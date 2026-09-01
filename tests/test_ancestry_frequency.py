"""Reference-frequency panels and EAF-profile decomposition."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ppb import (
    decompose_effect_allele_frequencies,
    estimate_frequency_composition,
    load_frequency_panel,
    match_effect_allele_frequencies,
    write_frequency_panel,
)


def _panel(tmp_path, *, m=1500, pops=("AFR", "EUR", "EAS"), seed=5):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.08, 0.80, m)
    offsets = rng.normal(0.0, 0.10, (m, len(pops)))
    af = np.clip(base[:, None] + offsets, 0.01, 0.99)
    ids = np.array([f"rs{i}" for i in range(m)])
    counted = np.full(m, "A")
    other = np.full(m, "C")
    path = write_frequency_panel(
        tmp_path / "panel.npz", ids=ids,
        chrom=np.array([str(i % 10 + 1) for i in range(m)]),
        pos=np.arange(m, dtype=np.int64) + 1,
        counted_allele=counted, other_allele=other, pops=pops, af=af,
        n_samples=np.array([661, 503, 504][:len(pops)]),
        source="synthetic unit-test panel", genome_build="GRCh37")
    return load_frequency_panel(path), af, ids, counted, other


def test_frequency_panel_round_trip_integrity_and_external_pin(tmp_path):
    panel, af, _ids, _counted, _other = _panel(tmp_path)
    assert panel.pops == ["AFR", "EUR", "EAS"]
    assert np.allclose(panel.af, af)
    assert len(panel.panel_sha256) == 64
    assert load_frequency_panel(
        tmp_path / "panel.npz",
        expected_sha256=panel.panel_sha256).panel_sha256 == panel.panel_sha256

    with np.load(tmp_path / "panel.npz", allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    fields["af"] = fields["af"].copy()
    fields["af"][0, 0] += 0.01
    np.savez_compressed(tmp_path / "tampered.npz", **fields)
    with pytest.raises(ValueError, match="content-hash"):
        load_frequency_panel(tmp_path / "tampered.npz")

    replacement = write_frequency_panel(
        tmp_path / "replacement.npz", ids=panel.ids, chrom=panel.chrom,
        pos=panel.pos, counted_allele=panel.counted_allele,
        other_allele=panel.other_allele,
        pops=panel.pops[::-1], af=panel.af[:, ::-1],
        n_samples=panel.n_samples[::-1], source=panel.source,
        genome_build=panel.genome_build)
    with pytest.raises(ValueError, match="externally registered"):
        load_frequency_panel(replacement, expected_sha256=panel.panel_sha256)

    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(b"PK\x03\x04not-a-zip")
    with pytest.raises(ValueError, match="cannot read ancestry panel"):
        load_frequency_panel(truncated)


def test_frequency_panel_rejects_strand_ambiguous_variants(tmp_path):
    with pytest.raises(ValueError, match="strand-ambiguous"):
        write_frequency_panel(
            tmp_path / "bad.npz", ids=["rs1"], chrom=["1"], pos=[1],
            counted_allele=["A"], other_allele=["T"], pops=["P1", "P2"],
            af=[[0.2, 0.8]], n_samples=[10, 10], source="invalid")


def test_matcher_aligns_swaps_and_counts_attrition(tmp_path):
    panel, _af, ids, counted, other = _panel(tmp_path, m=1200)
    observed = np.full(900, 0.3)
    observed[500:600] = 0.7
    observed[800:] = np.nan
    matched = match_effect_allele_frequencies(
        np.concatenate([ids[:500], ids[500:600], ids[:100],
                        ids[600:700], ids[700:800]]),
        np.concatenate([counted[:500], other[500:600], counted[:100],
                        np.full(100, "A"), counted[700:800]]),
        np.concatenate([other[:500], counted[500:600], other[:100],
                        np.full(100, "T"), other[700:800]]),
        observed, panel)
    assert np.allclose(matched.eaf, 0.3)
    assert matched.counts == {
        "n_input": 900,
        "n_dropped_duplicate": 100,
        "n_dropped_palindromic": 100,
        "n_dropped_not_in_panel": 0,
        "n_dropped_allele_mismatch": 0,
        "n_dropped_nonfinite_eaf": 100,
        "n_dropped_out_of_range_eaf": 0,
        "n_matched": 600,
    }


def test_matcher_distinguishes_missing_ids_from_allele_mismatch(tmp_path):
    panel, _af, _ids, _counted, _other = _panel(tmp_path, m=1200)
    matched = match_effect_allele_frequencies(
        ["rs0", "not-in-panel"], ["A", "A"], ["G", "C"],
        [0.2, 0.3], panel)
    assert matched.counts["n_dropped_allele_mismatch"] == 1
    assert matched.counts["n_dropped_not_in_panel"] == 1
    assert matched.counts["n_matched"] == 0


def test_known_mixture_recovery_and_uncertainty_scope(tmp_path):
    panel, af, ids, counted, other = _panel(tmp_path, m=3000)
    truth = np.array([0.60, 0.25, 0.15])
    rng = np.random.default_rng(7)
    target = np.clip(af @ truth + rng.normal(0.0, 0.002, len(ids)), 0, 1)
    flip = rng.random(len(ids)) < 0.30
    result = decompose_effect_allele_frequencies(
        ids, np.where(flip, other, counted),
        np.where(flip, counted, other), np.where(flip, 1 - target, target),
        panel)
    assert result["estimand"] == "af_projection_weights"
    assert result["status"] == "estimated"
    assert np.allclose(result["proportions"], truth, atol=0.01)
    assert result["proportions_se"] is not None
    assert result["af_corr"] > 0.99
    assert result["matching"]["n_matched"] == len(ids)
    assert "conditions on the fixed panel" in result["uncertainty_note"]

    fitted = np.asarray(result["proportions"])
    expected_reference_rms = np.sqrt(np.mean(np.sum(
        fitted[None, :] ** 2 * af * (1.0 - af)
        / (2.0 * panel.n_samples[None, :]), axis=1)))
    assert result["reference_sampling_rms"] == pytest.approx(
        expected_reference_rms)


def test_out_of_range_eaf_is_an_invalid_input(tmp_path):
    panel, af, ids, counted, other = _panel(tmp_path)
    target = af @ np.array([0.6, 0.25, 0.15])
    target[10] = 1.01
    result = decompose_effect_allele_frequencies(
        ids, counted, other, target, panel)
    assert result["status"] == "invalid_input"
    assert result["matching"]["n_dropped_out_of_range_eaf"] == 1


def test_acceptance_requires_one_thousand_variants(tmp_path):
    panel, af, ids, counted, other = _panel(tmp_path, m=999)
    result = decompose_effect_allele_frequencies(
        ids, counted, other, af @ np.array([0.5, 0.3, 0.2]), panel)
    assert result["status"] == "insufficient"
    assert result["validity_thresholds"]["minimum_matched_variants"] == 1000
    assert "engineering gates" in result["acceptance_gate_note"]


@pytest.mark.parametrize("failure", ["few_autosomes", "dominant_chromosome"])
def test_acceptance_requires_broad_chromosome_coverage(tmp_path, failure):
    panel, af, ids, counted, other = _panel(tmp_path, m=1200)
    if failure == "few_autosomes":
        chrom = np.array([str(i % 9 + 1) for i in range(len(ids))])
    else:
        chrom = np.concatenate([
            np.full(400, "1"),
            np.array([str(i % 9 + 2) for i in range(800)]),
        ])
    path = write_frequency_panel(
        tmp_path / f"{failure}.npz", ids=ids, chrom=chrom, pos=panel.pos,
        counted_allele=counted, other_allele=other, pops=panel.pops, af=af,
        n_samples=panel.n_samples, source=failure)
    coverage_panel = load_frequency_panel(path)
    result = decompose_effect_allele_frequencies(
        ids, counted, other, af @ np.array([0.5, 0.3, 0.2]),
        coverage_panel)
    assert result["status"] == "insufficient"
    if failure == "few_autosomes":
        assert result["chromosome_coverage"]["n_distinct_autosomes"] == 9
    else:
        assert result["chromosome_coverage"][
            "largest_chromosome_fraction"] > 0.25


def test_every_leave_one_chromosome_design_must_be_identifiable(tmp_path):
    rng = np.random.default_rng(12)
    m = 1500
    chrom = np.array([str(i % 10 + 1) for i in range(m)])
    base = rng.uniform(0.15, 0.85, m)
    common = rng.normal(0.0, 0.05, m)
    unique = np.where(chrom == "1", rng.normal(0.0, 0.05, m), 0.0)
    af = np.clip(np.column_stack([
        base + common + unique,
        base + common - unique,
        base - common,
    ]), 0.01, 0.99)
    ids = np.array([f"rs-loo-{i}" for i in range(m)])
    path = write_frequency_panel(
        tmp_path / "loo.npz", ids=ids, chrom=chrom, pos=np.arange(m),
        counted_allele=np.full(m, "A"), other_allele=np.full(m, "C"),
        pops=["P1", "P2", "P3"], af=af, n_samples=[200, 200, 200],
        source="leave-one-chromosome test")
    panel = load_frequency_panel(path)
    result = decompose_effect_allele_frequencies(
        ids, np.full(m, "A"), np.full(m, "C"),
        af @ np.array([0.4, 0.3, 0.3]), panel)
    failures = [
        item for item in result["leave_one_chromosome_identifiability"][
            "designs"] if not item["identifiable"]]
    assert [item["chromosome"] for item in failures] == ["1"]
    assert result["status"] == "nonidentifiable"
    assert result["proportions_se"] is None
    for item in result["leave_one_chromosome_identifiability"]["designs"]:
        if item["chromosome"] == "1":
            continue
        assert item["contrast_rank"] == item["expected_contrast_rank"]


def test_leave_one_chromosome_rank_is_k_minus_one_on_a_balanced_panel(tmp_path):
    rng = np.random.default_rng(13)
    m = 2000
    chrom = np.array([str(i % 10 + 1) for i in range(m)])
    base = rng.uniform(0.15, 0.85, m)
    af = np.clip(np.column_stack([
        base + rng.normal(0.0, 0.08, m),
        base + rng.normal(0.0, 0.08, m),
        base + rng.normal(0.0, 0.08, m),
    ]), 0.01, 0.99)
    ids = np.array([f"rs-bal-{i}" for i in range(m)])
    path = write_frequency_panel(
        tmp_path / "balanced.npz", ids=ids, chrom=chrom, pos=np.arange(m),
        counted_allele=np.full(m, "A"), other_allele=np.full(m, "C"),
        pops=["P1", "P2", "P3"], af=af, n_samples=[200, 200, 200],
        source="balanced leave-one-chromosome test")
    panel = load_frequency_panel(path)
    result = decompose_effect_allele_frequencies(
        ids, np.full(m, "A"), np.full(m, "C"),
        af @ np.array([0.5, 0.3, 0.2]), panel)
    assert result["status"] == "estimated"
    designs = result["leave_one_chromosome_identifiability"]["designs"]
    assert designs
    for item in designs:
        assert item["contrast_rank"] == item["expected_contrast_rank"] == 2
        assert item["identifiable"] is True


def test_rank_deficient_panel_is_nonidentifiable(tmp_path):
    rng = np.random.default_rng(10)
    m = 1200
    base = rng.uniform(0.05, 0.95, m)
    ids = np.array([f"rs-rank-{i}" for i in range(m)])
    path = write_frequency_panel(
        tmp_path / "rank.npz", ids=ids,
        chrom=np.array([str(i % 10 + 1) for i in range(m)]),
        pos=np.arange(m), counted_allele=np.full(m, "A"),
        other_allele=np.full(m, "C"), pops=["P1", "P2"],
        af=np.column_stack([base, base]), n_samples=[200, 200],
        source="rank-deficient test")
    panel = load_frequency_panel(path)
    result = decompose_effect_allele_frequencies(
        ids, np.full(m, "A"), np.full(m, "C"), base, panel)
    assert result["panel_confusability"]["contrast_rank"] == 0
    assert result["status"] == "nonidentifiable"
    assert result["boundary_note"] is not None


def test_off_panel_profile_fails_fixed_fit_gate(tmp_path):
    rng = np.random.default_rng(11)
    m = 2000
    base = rng.uniform(0.15, 0.85, m)
    contrasts = rng.normal(0.0, 0.025, (m, 3))
    contrasts -= contrasts.mean(axis=1, keepdims=True)
    af = np.clip(base[:, None] + contrasts, 0.01, 0.99)
    ids = np.array([f"rs-fit-{i}" for i in range(m)])
    path = write_frequency_panel(
        tmp_path / "fit.npz", ids=ids,
        chrom=np.array([str(i % 10 + 1) for i in range(m)]),
        pos=np.arange(m), counted_allele=np.full(m, "A"),
        other_allele=np.full(m, "C"), pops=["P1", "P2", "P3"], af=af,
        n_samples=[200, 200, 200], source="poor-fit test")
    panel = load_frequency_panel(path)
    target = np.clip(base + 0.07 * np.sin(np.arange(m)), 0.0, 1.0)
    matched = match_effect_allele_frequencies(
        ids, np.full(m, "A"), np.full(m, "C"), target, panel)
    result = estimate_frequency_composition(matched, panel)
    assert result["af_corr"] > 0.9
    assert result["residual_rms_over_contrast"] > 0.5
    assert result["status"] == "poor_fit"


def test_builder_orients_info_alt_frequency(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" \
        / "build_ancestry_panels.py"
    spec = importlib.util.spec_from_file_location("ppb_af_builder", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    ldref = tmp_path / "ldref.npz"
    np.savez_compressed(
        ldref, ids=np.array(["rs1"]), chrom=np.array(["22"]), pos=[101],
        counted_allele=np.array(["C"]), other_allele=np.array(["A"]))
    samples = tmp_path / "samples.panel"
    samples.write_text(
        "sample\tpop\tsuper_pop\tgender\n"
        "S1\tYRI\tAFR\tmale\nS2\tCEU\tEUR\tfemale\n")
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "22\t101\t.\tA\tC\t.\tPASS\tAFR_AF=0.2;EUR_AF=0.7\n")
    built = builder.build_panel(ldref, [vcf], samples, ("AFR", "EUR"))
    assert built["counted_allele"].tolist() == ["C"]
    assert np.allclose(built["af"], [[0.2, 0.7]])
