"""Tests for the file I/O layer and the ``ppb evaluate`` CLI."""

import gc
import json
import weakref

import numpy as np
import pytest

from ppb import (
    DenseLD,
    DenseLDInt8,
    VariantTable,
    evaluate,
    evaluate_ldrefs,
    read_bundle,
    read_sumstats,
    read_weights,
    standardized_marginal,
    write_bundle,
    write_ldref,
)
from ppb.cli import main


def _fixture(tmp_path, seed=0, m=8):
    """Write a small bundle + a matching weights file; return paths and truth."""
    rng = np.random.default_rng(seed)
    variants = VariantTable(
        chrom=np.ones(m, dtype=int), pos=np.arange(1, m + 1),
        a1=np.array(["A"] * m), a2=np.array(["G", "C"] * (m // 2)))
    A = rng.standard_normal((300, m))
    Xs = (A - A.mean(0)) / A.std(0)
    D = (Xs.T @ Xs) / Xs.shape[0]
    z = (Xs.T @ ((A[:, 0] - A[:, 0].mean()) / A[:, 0].std())) / Xs.shape[0]
    w = rng.standard_normal(m)

    bundle_path = tmp_path / "bundle.npz"
    write_bundle(bundle_path, variants, z, D=D)

    weights_path = tmp_path / "weights.tsv"
    with open(weights_path, "w", encoding="utf-8") as fh:
        fh.write("chr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\n")
        for i in range(m):
            fh.write(f"1\t{i + 1}\t{variants.a1[i]}\t{variants.a2[i]}\t{w[i]:.6f}\n")

    # Truth from the SAME parsed file the CLI reads, so the test checks plumbing,
    # not the file's 6-decimal rounding.
    wv, wr = read_weights(weights_path)
    truth = evaluate(DenseLD(D), variants, wv, wr, variants, z).r2
    return weights_path, bundle_path, truth


def _ldref_fixture(tmp_path):
    directory = tmp_path / "ldref"
    directory.mkdir()
    for chrom in ("1", "2"):
        variants = VariantTable(
            [chrom, chrom], [1, 2], ["A", "A"], ["C", "C"])
        block = DenseLDInt8.from_dense(np.eye(2)).packed()
        write_ldref(
            directory / f"ldref_chr{chrom}.npz",
            variants, [(block, np.arange(2))], af=[0.2, 0.3],
            psd_scan=False)

    # Interleaving exercises the non-contiguous chromosome partition path.
    variants = VariantTable(
        ["1", "2", "1", "2"], [1, 1, 2, 2],
        ["A", "A", "A", "A"], ["C", "C", "C", "C"])
    weights = np.array([1.0, 3.0, 2.0, 4.0])
    z = np.array([0.1, 0.3, 0.2, 0.4])
    paths = [directory / "ldref_chr1.npz", directory / "ldref_chr2.npz"]
    return directory, paths, variants, weights, z


def test_read_weights_recognises_pgs_catalog_columns(tmp_path):
    p = tmp_path / "w.tsv"
    p.write_text("#comment line\nchr_name\tchr_position\teffect_allele\t"
                 "other_allele\teffect_weight\n1\t100\tA\tG\t0.25\n2\t200\tc\tt\t-0.5\n",
                 encoding="utf-8")
    variants, w = read_weights(p)
    assert variants.n == 2
    assert list(variants.a1) == ["A", "C"]          # upper-cased
    assert np.allclose(w, [0.25, -0.5])


def test_read_weights_missing_column_errors(tmp_path):
    p = tmp_path / "w.tsv"
    p.write_text("chr\tpos\tbeta\n1\t100\t0.1\n", encoding="utf-8")  # no alleles
    with pytest.raises(ValueError):
        read_weights(p)


def test_bundle_roundtrip(tmp_path):
    variants = VariantTable([1, 1], [10, 20], ["A", "C"], ["G", "T"])
    z = np.array([0.1, -0.2])
    D = np.array([[1.0, 0.3], [0.3, 1.0]])
    genotype_sd = np.array([0.7, 0.9])
    write_bundle(tmp_path / "b.npz", variants, z, D=D, genotype_sd=genotype_sd)
    b = read_bundle(tmp_path / "b.npz")
    assert b["bundle_version"] == 2
    assert np.allclose(b["genotype_sd"], genotype_sd)
    assert b["variants"].n == 2
    assert np.allclose(b["z"], z)
    assert np.allclose(b["ld"].quad([1.0, 1.0]), np.array([1, 1]) @ D @ np.array([1, 1]))


def test_cli_evaluate_stdout_matches_truth(tmp_path, capsys):
    weights_path, bundle_path, truth = _fixture(tmp_path)
    rc = main(["evaluate", "--weights", str(weights_path), "--bundle", str(bundle_path),
               "--weight-scale", "standardized"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert abs(out["r2"] - truth) <= 1e-9
    assert out["n_variants_scored"] == 8
    assert out["weights_report"]["n_matched"] == 8


def test_cli_evaluate_writes_json_file(tmp_path):
    weights_path, bundle_path, truth = _fixture(tmp_path, seed=2)
    out_path = tmp_path / "result.json"
    rc = main(["evaluate", "--weights", str(weights_path),
               "--bundle", str(bundle_path), "--weight-scale", "standardized",
               "--out", str(out_path)])
    assert rc == 0
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert abs(result["r2"] - truth) <= 1e-9


def test_read_weights_ragged_row_errors_with_line_number(tmp_path):
    p = tmp_path / "w.tsv"
    p.write_text("chr\tpos\ta1\ta2\tweight\n1\t100\tA\tG\t0.25\n2\t200\tC\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        read_weights(p)


def test_read_weights_rejects_nonfinite_weight(tmp_path):
    p = tmp_path / "w.tsv"
    p.write_text(
        "chr\tpos\ta1\ta2\tweight\n1\t100\tA\tG\tnan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2.*finite"):
        read_weights(p)


def test_bundle_rejects_nonfinite_inputs(tmp_path):
    variants = VariantTable([1], [10], ["A"], ["G"])
    with pytest.raises(ValueError, match="z.*finite"):
        write_bundle(tmp_path / "bad.npz", variants, [np.nan], D=np.eye(1))
    with pytest.raises(ValueError, match="var_y"):
        write_bundle(tmp_path / "bad.npz", variants, [0.1], D=np.eye(1), var_y=0)


def test_cli_dosage_scale_uses_bundle_genotype_sd(tmp_path, capsys):
    variants = VariantTable([1, 1], [10, 20], ["A", "A"], ["G", "C"])
    z = np.array([0.2, 0.1])
    bundle = tmp_path / "bundle.npz"
    write_bundle(bundle, variants, z, D=np.eye(2), genotype_sd=[0.5, 2.0])
    weights = tmp_path / "weights.tsv"
    weights.write_text(
        "chr\tpos\ta1\ta2\teffect_weight\n"
        "1\t10\tA\tG\t1\n1\t20\tA\tC\t1\n", encoding="utf-8")
    assert main([
        "evaluate", "--weights", str(weights), "--bundle", str(bundle),
        "--weight-scale", "dosage"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["r2"] == pytest.approx(
        ((np.array([0.5, 2.0]) @ z) ** 2) / (0.5 ** 2 + 2.0 ** 2))
    assert result["weight_scale"] == "dosage"


def test_cli_dosage_scale_rejects_bundle_without_genotype_sd(tmp_path):
    weights_path, bundle_path, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="require.*genotype_sd"):
        main(["evaluate", "--weights", str(weights_path), "--bundle", str(bundle_path),
              "--weight-scale", "dosage"])


def test_bundle_summary_statistics_are_not_strand_resolved(tmp_path):
    """A bundle's z is stored in reference order; matching it against its own
    variant table has no strand to resolve, so it must not be reported as
    ambiguous removals. The weights are a submission and are still resolved.
    """
    m = 30
    a1 = np.array(["A"] * m)
    a2 = np.array(["T" if i % 3 == 0 else "G" for i in range(m)])  # every 3rd palindromic
    variants = VariantTable(np.array(["1"] * m), np.arange(1, m + 1), a1, a2)
    bundle = tmp_path / "b.npz"
    write_bundle(bundle, variants, np.full(m, 0.05), D=np.eye(m))

    weights = tmp_path / "w.tsv"
    weights.write_text(
        "chr\tpos\teffect_allele\tother_allele\teffect_weight\n"
        + "".join(f"1\t{i + 1}\t{a1[i]}\t{a2[i]}\t1.0\n" for i in range(m)),
        encoding="utf-8")

    out = tmp_path / "r.json"
    assert main(["evaluate", "--weights", str(weights), "--bundle", str(bundle),
                 "--weight-scale", "standardized", "--out", str(out)]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))

    n_palindromic = sum(1 for i in range(m) if i % 3 == 0)
    assert result["sumstats_report"]["n_ambiguous_removed"] == 0
    assert result["sumstats_report"]["n_matched"] == m
    # The weights side still drops them -- an external file's strand is unknown.
    assert result["weights_report"]["n_ambiguous_removed"] == n_palindromic
    assert result["n_variants_scored"] == m - n_palindromic


def test_mse_is_flagged_uninterpretable_for_dosage_weights(tmp_path):
    """R^2 is invariant to a global rescale of w; MSE is not. Dosage-scale
    weights carry an arbitrary scale, so their MSE means nothing and the result
    must say so rather than emit an equally authoritative-looking number."""
    m = 8
    variants = VariantTable(np.array(["1"] * m), np.arange(1, m + 1),
                            np.array(["A"] * m), np.array(["G"] * m))
    bundle = tmp_path / "b.npz"
    write_bundle(bundle, variants, np.full(m, 0.1), D=np.eye(m),
                 genotype_sd=np.full(m, 0.5))
    weights = tmp_path / "w.tsv"
    weights.write_text(
        "chr\tpos\teffect_allele\tother_allele\teffect_weight\n"
        + "".join(f"1\t{i + 1}\tA\tG\t1.0\n" for i in range(m)), encoding="utf-8")

    for scale, expected in (("dosage", False), ("standardized", True)):
        out = tmp_path / f"{scale}.json"
        assert main(["evaluate", "--weights", str(weights), "--bundle", str(bundle),
                     "--weight-scale", scale, "--out", str(out)]) == 0
        assert json.loads(out.read_text(encoding="utf-8"))["mse_interpretable"] is expected


def test_sharded_ldref_evaluation_forms_one_global_ratio(tmp_path):
    _, paths, variants, weights, z = _ldref_fixture(tmp_path)
    result = evaluate_ldrefs(
        paths, variants, weights, variants, z,
        # An unused column must not make standardized weights fail.
        genotype_sd=np.ones(variants.n))

    numerator = float(weights @ z)
    denominator = float(weights @ weights)
    assert result.r2 == pytest.approx(numerator ** 2 / denominator)
    assert result.mse == pytest.approx(1.0 - 2.0 * numerator + denominator)
    assert result.mse_interpretable is True
    assert result.n_ldref_files == 2
    assert result.chromosomes == ("1", "2")
    assert result.weights_report["n_matched"] == 4
    assert result.sumstats_report["n_matched"] == 4


def test_sharded_ldref_uses_joint_support_and_dosage_scale(tmp_path):
    _, paths, variants, weights, z = _ldref_fixture(tmp_path)
    subset = np.array([0, 1, 2])
    target = VariantTable(
        variants.chrom[subset], variants.pos[subset],
        variants.a1[subset], variants.a2[subset])
    target_z = z[subset]
    target_sd = np.array([0.5, 2.0, 1.0])

    result = evaluate_ldrefs(
        paths, variants, weights, target, target_z,
        weight_scale="dosage", genotype_sd=target_sd)

    scaled = weights[subset] * target_sd
    numerator = float(scaled @ target_z)
    denominator = float(scaled @ scaled)
    assert result.r2 == pytest.approx(numerator ** 2 / denominator)
    assert result.n_variants_scored == 3
    assert result.mse_interpretable is False
    assert result.genotype_sd_source == "sumstats_empirical"


def test_sharded_ldref_dosage_requires_exactly_one_sd_source(tmp_path):
    _, paths, variants, weights, z = _ldref_fixture(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_ldrefs(
            paths, variants, weights, variants, z, weight_scale="dosage")
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_ldrefs(
            paths, variants, weights, variants, z, weight_scale="dosage",
            genotype_sd=np.ones(variants.n), hwe_genotype_sd=True)

    result = evaluate_ldrefs(
        paths, variants, weights, variants, z, weight_scale="dosage",
        hwe_genotype_sd=True)
    sd = np.sqrt(2.0 * np.array([0.2, 0.2, 0.3, 0.3])
                 * (1.0 - np.array([0.2, 0.2, 0.3, 0.3])))
    scaled = weights * sd
    assert result.r2 == pytest.approx((scaled @ z) ** 2 / (scaled @ scaled))
    assert result.genotype_sd_source == "ldref_hwe"


def test_read_sumstats_and_cli_sharded_mode(tmp_path, capsys):
    directory, _, variants, weights, z = _ldref_fixture(tmp_path)
    weights_path = tmp_path / "weights.tsv"
    sumstats_path = tmp_path / "sumstats.tsv"
    header = "chrom\tpos\ta1\ta2"
    weights_path.write_text(
        header + "\tweight\n" + "".join(
            f"{variants.chrom[i]}\t{variants.pos[i]}\tA\tC\t{weights[i]}\n"
            for i in range(variants.n)),
        encoding="utf-8")
    sumstats_path.write_text(
        header + "\tz\tgenotype_sd\n" + "".join(
            f"{variants.chrom[i]}\t{variants.pos[i]}\tA\tC\t{z[i]}\t1\n"
            for i in range(variants.n)),
        encoding="utf-8")

    parsed, parsed_z, parsed_sd = read_sumstats(sumstats_path)
    assert parsed.n == variants.n
    assert np.array_equal(parsed_z, z)
    assert np.array_equal(parsed_sd, np.ones(variants.n))

    assert main([
        "evaluate", "--weights", str(weights_path),
        "--ldref-dir", str(directory), "--sumstats", str(sumstats_path),
        "--weight-scale", "standardized"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["r2"] == pytest.approx((weights @ z) ** 2 / (weights @ weights))
    assert result["source"] == "ldref_shards"
    assert result["n_ldref_files"] == 2


def test_cli_hwe_flag_overrides_optional_empirical_sd(tmp_path, capsys):
    directory, _, variants, weights, z = _ldref_fixture(tmp_path)
    weights_path = tmp_path / "weights.tsv"
    sumstats_path = tmp_path / "sumstats.tsv"
    rows = "".join(
        f"{variants.chrom[i]}\t{variants.pos[i]}\tA\tC\t"
        f"{weights[i]}\t{z[i]}\tnot-an-sd\n"
        for i in range(variants.n)
    )
    weights_path.write_text(
        "chrom\tpos\ta1\ta2\tweight\tz\tgenotype_sd\n" + rows,
        encoding="utf-8")
    sumstats_path.write_text(
        "chrom\tpos\ta1\ta2\tweight\tz\tgenotype_sd\n" + rows,
        encoding="utf-8")

    assert main([
        "evaluate", "--weights", str(weights_path),
        "--ldref-dir", str(directory), "--sumstats", str(sumstats_path),
        "--weight-scale", "dosage", "--hwe-genotype-sd"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["genotype_sd_source"] == "ldref_hwe"


def test_cli_standardized_weights_ignore_optional_empirical_sd(tmp_path, capsys):
    directory, _, variants, weights, z = _ldref_fixture(tmp_path)
    weights_path = tmp_path / "weights.tsv"
    sumstats_path = tmp_path / "sumstats.tsv"
    weights_path.write_text(
        "chrom\tpos\ta1\ta2\tweight\n" + "".join(
            f"{variants.chrom[i]}\t{variants.pos[i]}\tA\tC\t{weights[i]}\n"
            for i in range(variants.n)),
        encoding="utf-8")
    sumstats_path.write_text(
        "chrom\tpos\ta1\ta2\tz\tgenotype_sd\n" + "".join(
            f"{variants.chrom[i]}\t{variants.pos[i]}\tA\tC\t{z[i]}\tbad\n"
            for i in range(variants.n)),
        encoding="utf-8")

    assert main([
        "evaluate", "--weights", str(weights_path),
        "--ldref-dir", str(directory), "--sumstats", str(sumstats_path),
        "--weight-scale", "standardized"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["genotype_sd_source"] == "not_used"


def test_read_sumstats_validates_empirical_sd_by_default(tmp_path):
    path = tmp_path / "sumstats.tsv"
    path.write_text(
        "chrom\tpos\ta1\ta2\tz\tgenotype_sd\n"
        "1\t1\tA\tC\t0.1\tbad\n",
        encoding="utf-8")

    with pytest.raises(ValueError):
        read_sumstats(path)


def test_cli_validates_sharded_mode_before_reading_large_tables(tmp_path):
    with pytest.raises(ValueError, match="--sumstats is required"):
        main([
            "evaluate", "--weights", str(tmp_path / "missing.tsv"),
            "--ldref-dir", str(tmp_path / "missing-ldref"),
            "--weight-scale", "standardized"])


def test_sharded_evaluation_releases_ld_before_reading_next(monkeypatch):
    import ppb.io as io

    previous = None
    calls = 0

    class IdentityLD:
        m = 1

        @staticmethod
        def quad(w):
            return float(w @ w)

    def fake_read(path):
        nonlocal calls, previous
        gc.collect()
        if previous is not None:
            assert previous() is None
        calls += 1
        ld = IdentityLD()
        previous = weakref.ref(ld)
        chrom = str(calls)
        return {
            "variants": VariantTable([chrom], [1], ["A"], ["C"]),
            "ld": ld,
        }

    monkeypatch.setattr(io, "read_ldref", fake_read)
    both = VariantTable(["1", "2"], [1, 1], ["A", "A"], ["C", "C"])
    result = io.evaluate_ldrefs(
        ["one", "two"], both, np.ones(2), both, np.full(2, 0.1))

    assert calls == 2
    assert result.n_ldref_files == 2


def test_read_weight_file_keeps_ldpred3_frozen_columns(tmp_path):
    from ppb import read_weight_file

    path = tmp_path / "w.tsv"
    path.write_text(
        "ID\tCHR\tPOS\tA1\tA2\tWEIGHT\tAF_REF\tSD_REF\n"
        "rs1\t1\t100\tA\tG\t0.2\t0.4\t0.5\n"
        "rs2\t1\t200\tC\tT\t0.4\t0.1\t0.0\n",
        encoding="utf-8")
    wf = read_weight_file(path)
    assert wf.has_frozen_scale
    assert np.allclose(wf.weights, [0.2, 0.4])
    assert np.allclose(wf.sd_ref, [0.5, 0.0])
    assert list(wf.variants.a1) == ["A", "C"]


def test_cli_frozen_scale_uses_sd_ref_then_target_sd(tmp_path, capsys):
    variants = VariantTable([1, 1], [10, 20], ["A", "A"], ["G", "C"])
    z = np.array([0.2, 0.1])
    bundle = tmp_path / "bundle.npz"
    write_bundle(bundle, variants, z, D=np.eye(2), genotype_sd=[0.5, 2.0])
    weights = tmp_path / "weights.tsv"
    weights.write_text(
        "ID\tCHR\tPOS\tA1\tA2\tWEIGHT\tAF_REF\tSD_REF\n"
        "rs1\t1\t10\tA\tG\t0.2\t0.3\t0.5\n"
        "rs2\t1\t20\tA\tC\t0.4\t0.4\t2.0\n",
        encoding="utf-8")
    assert main([
        "evaluate", "--weights", str(weights), "--bundle", str(bundle),
        "--weight-scale", "frozen"]) == 0
    result = json.loads(capsys.readouterr().out)
    # WEIGHT/SD_REF = [0.4, 0.2], then × target sd [0.5, 2.0] → [0.2, 0.4]
    w = np.array([0.2, 0.4])
    assert result["r2"] == pytest.approx((w @ z) ** 2 / (w @ w))
    assert result["weight_scale"] == "frozen"
    assert result["mse_interpretable"] is False


def test_frozen_scale_refuses_to_guess_without_sd_ref(tmp_path):
    weights_path, bundle_path, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="sd_ref"):
        main(["evaluate", "--weights", str(weights_path),
              "--bundle", str(bundle_path), "--weight-scale", "frozen"])


def test_evaluate_emits_block_diagnostics_and_x3(tmp_path):
    from ppb import BlockDiagonalLD, DenseLD, evaluate

    variants = VariantTable(
        ["1", "1", "2", "2"], [1, 2, 1, 2],
        ["A", "A", "A", "A"], ["C", "C", "C", "C"])
    blocks = [
        (DenseLD(np.eye(2)), np.array([0, 1])),
        (DenseLD(np.eye(2)), np.array([2, 3])),
    ]
    ld = BlockDiagonalLD(blocks)
    w = np.array([1.0, 0.5, 0.25, 0.1])
    z = np.array([0.2, 0.1, 0.05, 0.02])
    result = evaluate(ld, variants, variants, w, variants, z, n_eff=500.0)
    assert result.jackknife is not None
    assert result.jackknife["n_blocks"] == 2
    assert result.sign_flip_null["z_ceiling"] == pytest.approx(np.sqrt(2.0))
    assert result.r2_corrected == pytest.approx(result.r2 - 1.0 / 500.0)
    assert result.r2_se_finite_sample == pytest.approx(
        2.0 * np.sqrt(result.r2 / 500.0))


def test_sharded_evaluation_emits_chromosome_jackknife(tmp_path):
    _, paths, variants, weights, z = _ldref_fixture(tmp_path)
    result = evaluate_ldrefs(paths, variants, weights, variants, z)
    assert result.jackknife["n_blocks"] == 2
    assert result.jackknife_chromosome["n_groups"] == 2
    assert set(result.per_chromosome) == {"1", "2"}


def test_evaluate_sumstats_frame_matches_reference_frame():
    from ppb import DenseLD, evaluate

    ref = VariantTable(["1", "1"], [10, 20], ["A", "A"], ["G", "C"])
    # Sumstats in reverse row order, with SDs attached to those rows.
    ss = VariantTable(["1", "1"], [20, 10], ["A", "A"], ["C", "G"])
    z = np.array([0.4, 0.1])           # pos 20, pos 10
    sd = np.array([2.0, 0.5])
    w = np.array([1.0, 1.0])
    via_sumstats = evaluate(
        DenseLD(np.eye(2)), ref, ref, w, ss, z,
        weight_scale="dosage", genotype_sd=sd,
        genotype_sd_frame="sumstats")
    via_ref = evaluate(
        DenseLD(np.eye(2)), ref, ref, w, ss, z,
        weight_scale="dosage", genotype_sd=np.array([0.5, 2.0]),
        genotype_sd_frame="reference")
    assert via_sumstats.r2 == pytest.approx(via_ref.r2)
    assert via_sumstats.genotype_sd_source == "sumstats_empirical"
    assert via_ref.genotype_sd_source == "reference_empirical"


def test_bundle_writer_rejects_asymmetric_D(tmp_path):
    variants = VariantTable([1, 1], [10, 20], ["A", "A"], ["G", "C"])
    bad = np.eye(2)
    bad[0, 1] = 0.5
    with pytest.raises(ValueError, match="symmetric"):
        write_bundle(tmp_path / "bad.npz", variants, [0.1, 0.2], D=bad)


def test_read_bundle_rejects_asymmetric_D(tmp_path):
    """The read boundary must fail closed on a legacy asymmetric bundle too."""
    path = tmp_path / "legacy.npz"
    bad = np.eye(2)
    bad[0, 1] = 0.5
    np.savez(
        path, chrom=np.array(["1", "1"]), pos=np.array([10, 20]),
        a1=np.array(["A", "A"]), a2=np.array(["G", "C"]),
        z=np.array([0.1, 0.2]), var_y=np.array(1.0),
        bundle_version=np.array(2, dtype=np.int64), D=bad)
    with pytest.raises(ValueError, match="symmetric"):
        read_bundle(path)


def test_read_sumstats_converts_beta_se_n(tmp_path):
    path = tmp_path / "sumstats.tsv"
    path.write_text(
        "chrom\tpos\ta1\ta2\tbeta\tse\tn\n"
        "1\t1\tA\tC\t0.01\t0.1\t4000\n"
        "1\t2\tA\tC\t-0.02\t0.05\t3000\n",
        encoding="utf-8")

    variants, z, sd = read_sumstats(path, scale="beta-se-n")
    assert np.allclose(
        z, standardized_marginal([0.01, -0.02], [0.1, 0.05], [4000, 3000]))
    assert variants.n == 2
    assert sd is None


def test_read_sumstats_beta_se_n_uses_n_eff_without_n_column(tmp_path):
    path = tmp_path / "sumstats.tsv"
    path.write_text(
        "chrom\tpos\ta1\ta2\tbeta\tse\n1\t1\tA\tC\t0.01\t0.1\n",
        encoding="utf-8")

    with pytest.raises(ValueError, match="n_eff"):
        read_sumstats(path, scale="beta-se-n")
    _, z, _ = read_sumstats(path, scale="beta-se-n", n_eff=5000.0)
    assert np.allclose(z, standardized_marginal([0.01], [0.1], 5000.0))


def test_read_sumstats_beta_se_n_validates_rows_with_line_numbers(tmp_path):
    path = tmp_path / "sumstats.tsv"
    path.write_text(
        "chrom\tpos\ta1\ta2\tbeta\tse\tn\n1\t1\tA\tC\t0.01\t0.0\t4000\n",
        encoding="utf-8")

    with pytest.raises(ValueError, match="line 2.*se"):
        read_sumstats(path, scale="beta-se-n")


def test_read_sumstats_rejects_unknown_scale(tmp_path):
    path = tmp_path / "sumstats.tsv"
    path.write_text("chrom\tpos\ta1\ta2\tz\n1\t1\tA\tC\t0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scale"):
        read_sumstats(path, scale="or")


def test_cli_bundle_rejects_sumstats_scale(tmp_path):
    weights_path, bundle_path, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="--sumstats-scale"):
        main(["evaluate", "--weights", str(weights_path),
              "--bundle", str(bundle_path), "--weight-scale", "standardized",
              "--sumstats-scale", "beta-se-n"])


def test_cli_sharded_mode_converts_beta_se_n(tmp_path, capsys):
    directory, paths, variants, weights, _ = _ldref_fixture(tmp_path)
    beta = np.array([0.05, 0.05, 0.05, 0.05])
    se = np.array([0.02] * 4)
    n_col = np.array([4000] * 4)
    expected = evaluate_ldrefs(
        paths, variants, weights, variants,
        standardized_marginal(beta, se, n_col)).r2

    weights_path = tmp_path / "weights.tsv"
    sumstats_path = tmp_path / "sumstats.tsv"
    weights_path.write_text(
        "chrom\tpos\ta1\ta2\tweight\n" + "".join(
            f"{variants.chrom[i]}\t{variants.pos[i]}\tA\tC\t{weights[i]}\n"
            for i in range(variants.n)), encoding="utf-8")
    sumstats_path.write_text(
        "chrom\tpos\ta1\ta2\tbeta\tse\tn\n" + "".join(
            f"{variants.chrom[i]}\t{variants.pos[i]}\tA\tC\t"
            f"{beta[i]}\t{se[i]}\t{n_col[i]}\n" for i in range(variants.n)),
        encoding="utf-8")

    assert main([
        "evaluate", "--weights", str(weights_path),
        "--ldref-dir", str(directory), "--sumstats", str(sumstats_path),
        "--sumstats-scale", "beta-se-n", "--weight-scale", "standardized",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["r2"] == pytest.approx(expected)


def test_cli_beta_se_n_without_n_column_requires_n_eff(tmp_path, capsys):
    directory, paths, variants, weights, _ = _ldref_fixture(tmp_path)
    beta = np.array([0.05, 0.05, 0.05, 0.05])
    se = np.array([0.02] * 4)
    weights_path = tmp_path / "weights.tsv"
    sumstats_path = tmp_path / "sumstats.tsv"
    weights_path.write_text(
        "chrom\tpos\ta1\ta2\tweight\n" + "".join(
            f"{variants.chrom[i]}\t{variants.pos[i]}\tA\tC\t{weights[i]}\n"
            for i in range(variants.n)), encoding="utf-8")
    sumstats_path.write_text(
        "chrom\tpos\ta1\ta2\tbeta\tse\n" + "".join(
            f"{variants.chrom[i]}\t{variants.pos[i]}\tA\tC\t"
            f"{beta[i]}\t{se[i]}\n" for i in range(variants.n)),
        encoding="utf-8")

    with pytest.raises(ValueError, match="n_eff"):
        main(["evaluate", "--weights", str(weights_path),
              "--ldref-dir", str(directory), "--sumstats", str(sumstats_path),
              "--sumstats-scale", "beta-se-n", "--weight-scale", "standardized"])

    assert main([
        "evaluate", "--weights", str(weights_path),
        "--ldref-dir", str(directory), "--sumstats", str(sumstats_path),
        "--sumstats-scale", "beta-se-n", "--weight-scale", "standardized",
        "--n-eff", "5000",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    expected = evaluate_ldrefs(
        paths, variants, weights, variants,
        standardized_marginal(beta, se, 5000.0)).r2
    assert result["r2"] == pytest.approx(expected)
    # The same N that converted the statistics enters the (X3) correction.
    assert result["r2_corrected"] == pytest.approx(expected - 1.0 / 5000.0)
