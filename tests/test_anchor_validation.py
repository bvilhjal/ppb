"""Contracts for the external anchor check.

The comparison itself needs the network, so the API is stubbed and what is
tested is the join and the arithmetic: that the European group is the one
selected out of nine, that partial-r is squared into R², and that a
non-overlapping registry row is required.
"""

import json

import pytest

from scripts import anchor_validation as av


def _payload(groups):
    """A PGS Catalog performance response with one entry per ancestry group."""
    return {"results": [
        {"sampleset": {"samples": [{"ancestry_additional": label,
                                    "ancestry_country": "UK",
                                    "sample_number": n}]},
         "performance_metrics": {
             "othermetrics": [{"name_long": "Partial Correlation (partial-r)",
                               "name_short": "partial-r", "estimate": r}],
             "effect_sizes": []}}
        for label, (r, n) in groups.items()]}


GROUPS = {
    "UK (+ Ireland)": (0.6133, 19953),
    "Poland (NE Europe)": (0.5922, 4126),
    "Nigeria (West Africa)": (0.2554, 3863),
}


def _record(trait="height", pgs="PGS002146", r2=0.252, overlap="none (declared)"):
    return {"trait": trait, "score": {"id": pgs, "n_variants": 1000},
            "target": {"overlap": overlap, "gwas": "GIANT 2014",
                       "trait_type": "quantitative"},
            "metrics": {"r2": r2, "n_variants_scored": 918}}


def test_published_accuracy_reads_every_ancestry_group():
    groups = av.published_accuracy("PGS002146", opener=lambda url: _payload(GROUPS))
    assert set(groups) == set(GROUPS)
    assert groups["UK (+ Ireland)"] == (0.6133, 19953)


def test_comparison_selects_the_european_group_not_the_last_one():
    """All nine published groups are UK Biobank with ancestry_country == 'UK';
    only ancestry_additional distinguishes them. Selecting on country alone
    silently compares against whichever group happens to come last."""
    rows, curves = av.compare([_record()], opener=lambda url: _payload(GROUPS))

    assert len(rows) == 1
    row = rows[0]
    assert row["partial_r_published"] == 0.6133          # the European group
    assert row["r2_published"] == pytest.approx(0.6133 ** 2)
    assert row["n_published"] == 19953
    assert row["ratio"] == pytest.approx(0.6133 ** 2 / 0.252)
    # The other groups survive as the portability curve.
    assert curves["height"]["Nigeria (West Africa)"] == pytest.approx(0.2554 ** 2)


def test_in_sample_rows_are_excluded():
    """Only declared non-overlapping evaluations are comparable to a held-out
    individual-level number; an in-sample row is an upper bound, not an estimate."""
    rows, _ = av.compare([_record(overlap="in-sample")],
                         opener=lambda url: _payload(GROUPS))
    assert rows == []


def test_a_score_without_a_european_group_is_skipped_not_guessed():
    missing = {k: v for k, v in GROUPS.items() if k != "UK (+ Ireland)"}
    rows, _ = av.compare([_record()], opener=lambda url: _payload(missing))
    assert rows == []


def test_support_is_reported_alongside_the_ratio():
    """A low ratio could be a restricted score rather than a scale problem, so
    the support has to travel with it."""
    rows, _ = av.compare([_record()], opener=lambda url: _payload(GROUPS))
    assert rows[0]["support"] == pytest.approx(0.918)


def test_summarize_returns_only_the_interpretable_rows():
    """Binary output is not an observed-scale partial correlation, so it must not
    enter the verdict."""
    records = [_record(),
               {**_record(trait="T2D", pgs="PGS002026", r2=0.044),
                "target": {"overlap": "none (declared)", "gwas": "DIAGRAM 2017",
                           "trait_type": "binary"}}]
    rows, curves = av.compare(records, opener=lambda url: _payload(GROUPS))
    assert len(rows) == 2
    quantitative = av.summarize(rows, curves)
    assert [r["trait"] for r in quantitative] == ["height"]


def test_script_is_runnable_as_documented(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "anchor_validation.py"
    done = subprocess.run([sys.executable, str(script), "--help"],
                          cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    # argparse rewraps the description, so match on the options it must expose
    # rather than on a prose fragment that wrapping can split.
    assert "--registry" in done.stdout and "--out" in done.stdout


def test_registry_rows_carry_what_the_comparison_needs():
    """Guard against the registry drifting away from the fields this join uses."""
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    records = json.loads((root / "results" / "baseline-2026-07.json").read_text())
    honest = [r for r in records if r["target"]["overlap"] == "none (declared)"]
    assert honest, "no non-overlapping rows to validate against"
    for record in honest:
        assert record["score"]["id"].startswith("PGS")
        assert record["metrics"]["r2"] > 0
        assert record["target"]["trait_type"] in {"quantitative", "binary"}
