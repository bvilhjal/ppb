"""Offline contracts for the compact-reference LD-moment GWAS benchmark."""
from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import ancestry_ld_gwas_benchmark as benchmark


def _design_fields():
    ids = np.asarray(["rs1", "rs2", "rs3", "rs4"])
    block = np.asarray([0, 0, 1, 1], dtype=np.int64)
    return {
        "schema_version": np.asarray(1),
        "populations": np.asarray(["AFR", "EUR"]),
        "sample_counts": np.asarray([661, 503], dtype=np.int64),
        "ids": ids,
        "chrom": np.asarray(["1"] * 4),
        "pos": np.asarray([10, 20, 1000, 1010], dtype=np.int64),
        "counted_allele": np.asarray(["A", "C", "G", "T"]),
        "other_allele": np.asarray(["C", "A", "T", "C"]),
        "variant_block": block,
        "block_source_id": np.asarray(["1:1-100", "1:901-1100"]),
        "block_chrom": np.asarray(["1", "1"]),
        "block_start": np.asarray([1, 901], dtype=np.int64),
        "block_end": np.asarray([100, 1100], dtype=np.int64),
        "pair_i": np.asarray([0, 2], dtype=np.int64),
        "pair_j": np.asarray([1, 3], dtype=np.int64),
        "pair_block": np.asarray([0, 1], dtype=np.int64),
        "pair_design": np.asarray([[0.4, 0.1], [0.2, 0.5]]),
        "quadratic_design": np.asarray(
            [[0.3, 0.1, 0.2], [0.2, 0.1, 0.4]]),
        "bilinear_design": np.asarray([
            [1.0, 0.2, 0.7], [0.9, 0.1, 0.8],
            [0.8, 0.3, 1.0], [0.7, 0.2, 1.1],
        ]),
        "maf_threshold": np.asarray(0.05),
        "min_block_gap_bp": np.asarray(500_000),
        "min_variants_per_block": np.asarray(2),
        "max_variants_per_block": np.asarray(100),
        "ld_floor": np.asarray(0.05),
        "max_pairs_per_block": np.asarray(250),
        "source": np.asarray("1000 Genomes Phase 3"),
        "source_url": np.asarray("https://example.test/1000g"),
        "source_archive_md5": np.asarray("1" * 32),
        "af_panel_sha256": np.asarray("2" * 64),
        "genome_build": np.asarray("GRCh37"),
        "semantic_sha256": np.asarray("3" * 64),
    }


def _write_design(path: Path, **changes):
    fields = _design_fields()
    fields.update(changes)
    fields.pop("semantic_sha256", None)
    fields["semantic_sha256"] = np.asarray(
        benchmark._semantic_digest(fields)
    )
    np.savez_compressed(path, **fields)
    return path


def _write_sumstats(path: Path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("rsid\teffect_allele\tother_allele\tbeta\tse\tn\n")
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")
    return path


def _study(expected="AFR"):
    return replace(
        benchmark.STUDIES[0], key="SYNTHETIC", accession="GCST_SYNTHETIC",
        expected_superpopulation=expected,
    )


def test_loader_verifies_hash_shapes_and_selection(tmp_path):
    path = _write_design(tmp_path / "design.npz")
    digest = benchmark.sha256_file(path)
    design = benchmark.load_ld_design(path, digest)
    assert design["pair_design"].shape == (2, 2)
    assert design["quadratic_design"].shape == (2, 3)
    assert design["bilinear_design"].shape == (4, 3)
    assert len(design["semantic_sha256"]) == 64
    assert design["selection"]["maf_threshold"] == pytest.approx(0.05)
    with pytest.raises(ValueError, match="file hash"):
        benchmark.load_ld_design(path, "0" * 64)

    bad = _write_design(
        tmp_path / "bad.npz", pair_design=np.ones((2, 3)))
    with pytest.raises(ValueError, match="pair_design.*shape"):
        benchmark.load_ld_design(bad)


def test_harmonization_sign_missing_filter_and_design_selection(tmp_path, monkeypatch):
    design = benchmark.load_ld_design(_write_design(tmp_path / "design.npz"))
    sumstats = _write_sumstats(tmp_path / "sumstats.tsv.gz", [
        ("rs1", "A", "C", 2.0, 1.0, 1000),       # direct: z=2
        ("rs2", "A", "C", 3.0, 1.0, 1100),       # swapped: z=-3
        ("rs3", "G", "T", 4.0, ".", 1200),       # invalid SE
        ("rs4", "T", "C", 5.0, 1.0, "."),        # invalid N
        ("not-in-panel", "A", "C", 9.0, 1.0, 900),
    ])
    seen_a, seen_b = [], []

    def fake_a(z, ii, jj, matrix, pair_block, **kwargs):
        seen_a.append((z.copy(), ii.copy(), jj.copy(), matrix.copy()))
        return {"proportions": np.asarray([0.8, 0.2]),
                "n_pairs": len(ii), "scale": 1.0}

    def fake_b(z, matrix, variant_block, **kwargs):
        seen_b.append((z.copy(), matrix.copy(), kwargs["sample_size"].copy()))
        return {"proportions": None, "signal_note": "exploratory decline"}

    monkeypatch.setattr(benchmark, "estimate_pair_products_from_design", fake_a)
    monkeypatch.setattr(benchmark, "estimate_bilinear_from_design", fake_b)
    row = benchmark.benchmark_study(
        _study(), design, sumstats, {"mode": "synthetic"})

    assert np.array_equal(seen_a[0][0], [2.0, -3.0])
    assert np.array_equal(seen_a[0][1], [0])
    assert np.array_equal(seen_a[0][2], [1])
    assert seen_a[0][3].shape == (1, 2)
    assert np.array_equal(seen_b[0][0], [2.0, -3.0])
    assert np.array_equal(seen_b[0][2], [1000.0, 1100.0])
    assert row["input"]["n_harmonized_variants"] == 2
    assert row["input"]["n_available_pairs"] == 1
    assert row["predeclared_control"]["expected_rank"] == 1
    assert row["estimator_b_exploratory"]["status"] == "declined"
    assert len(seen_a) == 1 + benchmark.SIGN_FLIP_REPLICATES
    diagnostic = row["independent_sign_flip_diagnostic"]
    assert diagnostic["n_replicates"] == benchmark.SIGN_FLIP_REPLICATES
    assert diagnostic["n_estimated"] == benchmark.SIGN_FLIP_REPLICATES
    assert diagnostic["included_in_primary_verdict"] is False
    assert diagnostic["scaled_contrast_empirical_p"] == pytest.approx(1.0)
    assert diagnostic["normalized_contrast_empirical_p"] == pytest.approx(1.0)


def test_fixed_verdict_and_numbered_table(monkeypatch, capsys):
    studies = [_study("AFR"), replace(_study(None), key="POOLED")]
    design = {
        "populations": np.asarray(["AFR", "EUR"]),
        "sample_counts": np.asarray([2, 2]), "ids": np.asarray(["rs1"]),
        "pair_i": np.asarray([0]), "file_sha256": "1" * 64,
        "semantic_sha256": "2" * 64, "n_blocks": 1,
        "selection": {"maf_threshold": 0.05}, "provenance": {},
        "extra_scalars": {},
    }

    def fake(study, design, path, acquisition):
        expected = study.expected_superpopulation
        return {
            "study": {**study.__dict__},
            "input": {"n_harmonized_variants": 10, "n_available_pairs": 8},
            "estimator_a": {"status": "estimated", "proportions": [0.7, 0.3]},
            "estimator_b_exploratory": {"status": "declined"},
            "independent_sign_flip_diagnostic": {
                "n_estimated": 2, "n_replicates": 200,
                "scaled_contrast_empirical_p": 0.005,
                "normalized_contrast_empirical_p": 0.02,
                "descriptive_threshold_passed": True,
            },
            "predeclared_control": {
                "included_in_verdict": expected is not None,
                "expected_rank": 1 if expected else None, "passed": True,
            },
        }

    monkeypatch.setattr(benchmark, "benchmark_study", fake)
    inputs = {study.key: (Path("unused"), {}) for study in studies}
    result = benchmark.run_benchmark(studies, design, inputs)
    assert result["verdict"] == {
        "n_controls": 1, "n_passed": 1,
        "failed_accessions": [], "passed": True,
    }
    assert "not participant ancestry fractions" in result["interpretation"]
    assert result["benchmark"]["estimator_b_role"].startswith("exploratory")
    benchmark.print_table(result)
    output = capsys.readouterr().out
    assert output.startswith("Table 1.")
    assert "| SYNTHETIC (GCST_SYNTHETIC) | AFR | estimated" in output
    assert "Qualitative benchmark-rule verdict: PASS (1/1 controls passed)." in output


def test_primary_control_fails_when_a_declines(monkeypatch):
    study = _study("AFR")
    design = {
        "populations": np.asarray(["AFR", "EUR"]),
        "sample_counts": np.asarray([2, 2]), "ids": np.asarray(["rs1"]),
        "pair_i": np.asarray([0]), "file_sha256": "1" * 64,
        "semantic_sha256": None, "n_blocks": 1, "selection": {},
        "provenance": {}, "extra_scalars": {},
    }
    monkeypatch.setattr(benchmark, "benchmark_study", lambda *args: {
        "study": {**study.__dict__},
        "independent_sign_flip_diagnostic": {
            "descriptive_threshold_passed": False,
            "normalized_contrast_empirical_p": 1.0,
        },
        "predeclared_control": {
            "included_in_verdict": True, "passed": False,
        },
    })
    result = benchmark.run_benchmark(
        [study], design, {study.key: (Path("unused"), {})})
    assert result["verdict"]["passed"] is False
    assert result["verdict"]["failed_accessions"] == [study.accession]


def test_fetch_uses_full_hm3_panel_not_pruned_design_ids(monkeypatch):
    study = _study("AFR")
    captured = {}
    monkeypatch.setattr(benchmark, "_selected", lambda _: [study])
    monkeypatch.setattr(benchmark, "load_ld_design", lambda *args: {
        "ids": np.asarray(["pruned-only"])
    })
    monkeypatch.setattr(
        benchmark, "load_frequency_panel",
        lambda *args, **kwargs: SimpleNamespace(
            ids=np.asarray(["full-1", "full-2"])))

    def acquire(study, *, panel_ids, **kwargs):
        captured["ids"] = panel_ids
        return Path("sumstats.tsv.gz"), {"mode": "synthetic"}

    monkeypatch.setattr(benchmark, "acquire_study", acquire)
    monkeypatch.setattr(benchmark, "run_benchmark", lambda *args: {
        "verdict": {"passed": True}
    })
    monkeypatch.setattr(benchmark, "print_table", lambda result: None)
    assert benchmark.main([
        "--design", "custom.npz", "--design-sha256", "0" * 64,
        "--fetch",
    ]) == 0
    assert captured["ids"] == {"full-1", "full-2"}


def test_checked_in_snapshot_pins_design_nulls_and_qualitative_verdict():
    path = (
        Path(benchmark.__file__).resolve().parents[1]
        / "results" / "ancestry-ld" / "yengo-height-2026-08-30.json"
    )
    text = path.read_text(encoding="utf-8")
    snapshot = json.loads(text, parse_constant=lambda value: pytest.fail(
        f"non-finite JSON constant {value}"))
    assert snapshot["schema_version"] == 1
    assert snapshot["design"]["file_sha256"] == (
        benchmark.DEFAULT_DESIGN_SHA256)
    assert snapshot["software"]["ppb_commit"] == (
        "5dca2f8320d0991f949926dfaac1a30f61a9a128")
    assert snapshot["verdict"] == {
        "n_controls": 5, "n_passed": 5,
        "failed_accessions": [], "passed": True,
    }
    diagnostic = snapshot["sign_flip_diagnostic_verdict"]
    assert diagnostic["n_scaled_contrast_passed"] == 5
    assert diagnostic["n_normalized_contrast_passed"] == 0
    expected = {
        study.key: study.expected_superpopulation for study in benchmark.STUDIES
    }
    for row in snapshot["studies"]:
        assert row["input"]["n_harmonized_variants"] > 21_500
        null = row["independent_sign_flip_diagnostic"]
        assert null["n_replicates"] == benchmark.SIGN_FLIP_REPLICATES
        if expected[row["study"]["key"]] is not None:
            assert null["scaled_contrast_empirical_p"] == pytest.approx(1 / 201)
            assert null["normalized_contrast_empirical_p"] > 0.05
    assert "C:\\" not in text and "/Users/" not in text
