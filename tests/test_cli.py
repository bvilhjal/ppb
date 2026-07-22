"""Tests for the file I/O layer and the ``ppb evaluate`` CLI."""

import json

import numpy as np
import pytest

from ppb import DenseLD, VariantTable, evaluate, read_bundle, read_weights, write_bundle
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
