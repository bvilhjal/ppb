"""Tests for the PGS Catalog independence screen (synthetic records, no network).

``scripts/pgs_independence.py`` implements the (H2) guard for the FinnGen /
non-UKB target path. The contract pinned here is that the screen fails
closed: a score with no cohort annotation, or with a cohort the matcher does
not recognize, is never declared independent of either biobank (the C2
defect: ``any([]) is False`` read absence of evidence as evidence of
absence for 28.3% of a real full-catalog run).
"""

import csv

from scripts import pgs_independence as pi


def _record(cohort_pairs, pgs_id="PGS000001"):
    """A minimal catalog record whose single training sample carries cohorts.

    ``cohort_pairs`` entries are ``(name_short, name_full)`` tuples; an empty
    short name exercises the full-name fallback.
    """
    return {
        "id": pgs_id,
        "name": "test score",
        "trait_efo": [{"id": "EFO:0000001"}],
        "samples_training": [{
            "sample_number": 1000, "ancestry_broad": "EUR",
            "cohorts": [{"name_short": s, "name_full": f}
                        for s, f in cohort_pairs],
        }],
    }


def test_cohorts_of_prefers_short_name_and_keeps_full():
    pairs = pi._cohorts_of(_record([("UKB", "United Kingdom Biobank"),
                                    ("", "FINRISK")]))
    assert pairs == [("UKB", "United Kingdom Biobank"), ("", "FINRISK")]
    assert pi._cohorts_of(_record([])) == []
    assert pi._cohorts_of({"id": "PGS1"}) == []
    # entries with both names empty carry no information and are dropped
    assert pi._cohorts_of(_record([("", "")])) == []


def test_label_matches_short_and_full_names():
    trained, unknown, evidence = pi._label(
        [("", "United Kingdom Biobank"), ("FINRISK", "FINRISK"),
         ("SomeCo", "Some cohort")])
    assert trained["UKB"] is True
    assert trained["FinnGen"] is True          # FINRISK feeds FinnGen
    assert trained["23andMe"] is False
    assert unknown == ["SomeCo"]
    assert "UKB:United Kingdom Biobank" in evidence
    assert "FinnGen:FINRISK" in evidence


def test_screen_status_fails_closed():
    ok_pairs = [("23andMe", "23andMe"), ("deCODE", "deCODE")]
    trained, unknown, evidence = pi._label(ok_pairs)
    assert pi._screen_status(ok_pairs, trained, unknown) == "independent"

    contaminated, _, _ = pi._label([("FinnGen", "FinnGen")])
    assert pi._screen_status([("FinnGen", "FinnGen")], contaminated, []) \
        == "contaminated"

    unrec, unknown2, _ = pi._label([("Mystery", "Mystery cohort")])
    assert pi._screen_status([("Mystery", "Mystery cohort")], unrec, unknown2) \
        == "unrecognized_cohorts"

    # the C2 regression: an empty cohort list must NOT be "independent"
    empty_trained, empty_unknown, _ = pi._label([])
    assert pi._screen_status([], empty_trained, empty_unknown) \
        == "no_annotation"


def test_main_writes_fail_closed_tsv(tmp_path, monkeypatch):
    records = [
        _record([("FinnGen", "FinnGen R4")], "PGS000001"),
        _record([("SomeCo", "Some cohort")], "PGS000002"),
        _record([], "PGS000003"),                              # no annotation
        _record([("23andMe", "23andMe")], "PGS000004"),        # clean
        _record([("", "United Kingdom Biobank")], "PGS000005"),
        _record([("FINRISK", "FINRISK")], "PGS000006"),
    ]
    monkeypatch.setattr(pi, "_fetch_json",
                        lambda url: {"results": records})
    out = tmp_path / "independence.tsv"
    assert pi.main(["--out", str(out)]) == 0
    rows = {r["pgs_id"]: r for r in csv.DictReader(
        out.read_text(encoding="utf-8").splitlines(), delimiter="\t")}

    assert rows["PGS000001"]["screen_status"] == "contaminated"
    assert rows["PGS000001"]["finngen_target_independent"] == "False"
    assert rows["PGS000001"]["finngen_trained"] == "True"
    assert rows["PGS000002"]["screen_status"] == "unrecognized_cohorts"
    assert rows["PGS000002"]["ukb_target_independent"] == "False"
    assert rows["PGS000003"]["screen_status"] == "no_annotation"
    assert rows["PGS000003"]["ukb_target_independent"] == "False"
    assert rows["PGS000003"]["finngen_target_independent"] == "False"
    assert rows["PGS000004"]["screen_status"] == "independent"
    assert rows["PGS000004"]["finngen_target_independent"] == "True"
    assert rows["PGS000004"]["ukb_target_independent"] == "True"
    assert rows["PGS000005"]["screen_status"] == "contaminated"  # full-name UKB
    assert rows["PGS000006"]["screen_status"] == "contaminated"  # Finnish alias
    assert "FinnGen:FINRISK" in rows["PGS000006"]["matched_evidence"]
    assert rows["PGS000004"]["matched_evidence"] == "23andMe:23andMe"
    # unrecognized cohorts appear once, not duplicated (the old writer
    # appended the unknown list to the cohort list)
    assert rows["PGS000002"]["cohorts"] == "SomeCo"
    assert rows["PGS000002"]["n_cohorts"] == "1"


def test_main_leaves_no_partial_file_when_a_fetch_fails(tmp_path, monkeypatch):
    def _fail(url):
        raise RuntimeError("API down")
    monkeypatch.setattr(pi, "_fetch_json", _fail)
    out = tmp_path / "independence.tsv"
    try:
        pi.main(["--out", str(out)])
    except RuntimeError:
        pass
    assert not out.exists()
    assert not list(tmp_path.glob("*.part.*"))
