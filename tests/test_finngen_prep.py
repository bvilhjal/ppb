"""Tests for the FinnGen prep machinery (synthetic fixtures only, no network).

``scripts/finngen_prep.py`` joins a FinnGen endpoint file (GRCh38) to a PGS
Catalog scoring file (GRCh37 hmPOS positions) on rsID and writes the ppb
``chrom, pos, a1, a2, beta, se`` table. Everything here runs on small local
gzipped fixtures through the script's ``--weights`` / ``--endpoint-file`` /
``--manifest`` knobs, so the machinery is exercised without touching the
FinnGen bucket or storing any real summary statistics.
"""

import gzip

import pytest

from scripts.finngen_prep import n_eff, prep, read_manifest, read_weights


def _gzip_write(path, text):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        fh.write(text)


def _manifest(tmp_path):
    p = tmp_path / "manifest.tsv"
    p.write_text(
        "phenocode\tphenotype\tcategory\tnum_cases\tnum_controls\tpath_bucket\tpath_https\n"
        "T2D\tType 2 diabetes\tE4\t89727\t396292\tgs://x\turl\n"
        "I9_CHD\tCHD\tI9\t90714\t409472\tgs://x\turl\n",
        encoding="utf-8")
    return p


def _weights(tmp_path):
    p = tmp_path / "weights.tsv.gz"
    _gzip_write(p,
        "# PGS Catalog harmonized scoring file\n"
        "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\thm_chr\thm_pos\n"
        "rs1\t1\t100\tA\tG\t0.01\t1\t100\n"
        "rs2\t2\t200\tC\tT\t-0.02\t2\t200\n"
        "rs3\t3\t300\tG\tA\t0.03\t3\t300\n"
        "rs4\t4\t400\tA\tC\t0.04\t4\t400\n"
        "rs5\t5\t500\tA\tC\t0.05\t5\t500\n")
    return p


def _endpoint(tmp_path):
    p = tmp_path / "T2D.gz"
    _gzip_write(p,
        "#chrom\tpos\tref\talt\trsids\tnearest_genes\tpval\tmlogp\tbeta\tsebeta\taf_alt\taf_alt_cases\taf_alt_controls\tinfo\n"
        "1\t100\tA\tG\trs1\tGENE1\t0.1\t1\t0.01\t0.001\t0.5\t0.5\t0.5\t1\n"      # matched
        "2\t200\tC\tT\trs2\tGENE2\t0.2\t1\t-0.02\t0.0\t0.5\t0.5\t0.5\t1\n"    # bad se (0)
        "3\t300\tG\tA\trs3;rs3b\tGENE3\t0.3\t1\t0.03\t0.002\t0.5\t0.5\t0.5\t1\n"  # matched, first rsid of a list
        "4\t400\tA\tC\trs4\tGENE4\t0.4\t1\t0.04\t0.003\t0.5\t0.5\t0.5\t1\n"    # matched
        "5\t500\tC\tG\trs4\tGENE4b\t0.5\t1\t0.05\t0.004\t0.5\t0.5\t0.5\t1\n"   # duplicate rsid
        "6\t600\tA\tT\trsX\tGENE5\t0.6\t1\t0.06\t0.005\t0.5\t0.5\t0.5\t1\n"    # not in weights
        "7\t700\tG\tC\t\tGENE6\t0.7\t1\t0.07\t0.006\t0.5\t0.5\t0.5\t1\n"       # no rsid
        "8\t800\tC\tA\trs5\tGENE7\t0.8\t1\t0.08\tnan\t0.5\t0.5\t0.5\t1\n")
    return p


def test_n_eff_formula():
    # 4/(1/n_case + 1/n_ctrl), not the harmonic mean of the two.
    assert n_eff(89727, 396292) == pytest.approx(
        4.0 / (1.0 / 89727 + 1.0 / 396292))
    assert n_eff(100, 100) == pytest.approx(200.0)


def test_read_manifest_and_weights(tmp_path):
    manifest = read_manifest(_manifest(tmp_path))
    assert manifest["T2D"] == (89727, 396292)
    assert manifest["I9_CHD"] == (90714, 409472)

    weights, duplicates = read_weights(_weights(tmp_path))
    assert weights["rs1"] == ("1", 100, "A", "G")
    assert weights["rs3"] == ("3", 300, "G", "A")
    assert duplicates == 0
    assert len(weights) == 5


def test_read_weights_counts_duplicate_rsids(tmp_path):
    p = tmp_path / "w.tsv.gz"
    _gzip_write(p,
        "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\thm_chr\thm_pos\n"
        "rs1\t1\t100\tA\tG\t0.01\t1\t100\n"
        "rs1\t1\t100\tA\tG\t0.01\t1\t100\n")
    weights, duplicates = read_weights(p)
    assert len(weights) == 1
    assert duplicates == 1


def test_prep_joins_on_rsid_and_writes_ppb_table(tmp_path):
    out = tmp_path / "T2D.tsv"
    sidecar = tmp_path / "T2D.json"
    counts = prep("T2D", _weights(tmp_path), _manifest(tmp_path), out,
                  out_json=sidecar, endpoint_source=_endpoint(tmp_path))

    rows = out.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "chrom\tpos\ta1\ta2\tbeta\tse"
    # GRCh37 positions come from the weights table; alleles are FinnGen ref/alt.
    assert rows[1] == "1\t100\tA\tG\t0.01\t0.001"
    assert rows[2] == "3\t300\tG\tA\t0.03\t0.002"
    assert rows[3] == "4\t400\tA\tC\t0.04\t0.003"
    assert len(rows) == 4

    assert counts["matched"] == 3
    assert counts["duplicate_rsid"] == 1
    assert counts["not_in_weights"] == 1
    assert counts["no_rsid"] == 1
    assert counts["bad_se"] == 2            # se=0 and se=nan
    assert counts["n_eff"] == pytest.approx(round(n_eff(89727, 396292), 1))
    assert counts["weights_parsed"] == 5

    import json
    recorded = json.loads(sidecar.read_text(encoding="utf-8"))
    assert recorded["endpoint"] == "T2D"
    assert recorded["matched"] == 3


def test_prep_rejects_unknown_endpoint_with_close_matches(tmp_path):
    with pytest.raises(ValueError, match="not in the manifest"):
        prep("NOPE", _weights(tmp_path), _manifest(tmp_path),
             tmp_path / "out.tsv", endpoint_source=_endpoint(tmp_path))


def test_prep_rejects_missing_column(tmp_path):
    bad = tmp_path / "bad.tsv.gz"
    _gzip_write(bad, "chrom\tpos\tref\talt\trsids\n1\t100\tA\tG\trs1\n")
    with pytest.raises(ValueError, match="beta"):
        prep("T2D", _weights(tmp_path), _manifest(tmp_path),
             tmp_path / "out.tsv", endpoint_source=bad)


def test_n_eff_rejects_nonpositive_counts():
    with pytest.raises(ValueError, match="positive"):
        n_eff(0, 100)
