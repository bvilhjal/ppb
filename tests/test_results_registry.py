"""Schema checks for the results registry (``results/*.json``).

The registry is the public-facing artifact: ``scripts/build_leaderboard.py``
renders it straight into the deployed site, and records are immutable once
merged. These tests enforce the rules stated in ``results/schema.md`` at PR
time, rather than letting a malformed record fail in the Pages deploy job.
"""

import json
from pathlib import Path

import pytest

from ppb.overlap import correct_numerator

ROOT = Path(__file__).resolve().parent.parent
PACKS = sorted((ROOT / "results").glob("*.json"))

ROLES = {"reference", "suspect", "suspect-unpaired"}
OVERLAP_DECLARATIONS = {"none (declared)", "in-sample"}


def _records():
    for pack in PACKS:
        for i, rec in enumerate(json.loads(pack.read_text(encoding="utf-8"))):
            yield f"{pack.name}[{i}]", rec


RECORDS = list(_records())
IDS = [name for name, _ in RECORDS]


def test_registry_is_not_empty():
    assert PACKS, "no result packs found in results/"
    assert RECORDS


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_record_has_required_fields(name, rec):
    for key in ("trait", "score", "target", "metrics", "overlap", "date", "ppb_commit"):
        assert key in rec, f"{name}: missing '{key}'"
    for key in ("id", "name", "training", "n_variants"):
        assert key in rec["score"], f"{name}: missing 'score.{key}'"
    for key in ("gwas", "cohort", "ancestry", "n_eff", "overlap"):
        assert key in rec["target"], f"{name}: missing 'target.{key}'"
    for key in ("num", "den", "r2", "w_match", "z_match"):
        assert key in rec["metrics"], f"{name}: missing 'metrics.{key}'"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_record_field_types_render(name, rec):
    """The fields the site formats numerically must actually be numbers."""
    assert isinstance(rec["target"]["n_eff"], int), f"{name}: target.n_eff must be an int"
    for key in ("num", "den", "r2", "w_match", "z_match"):
        assert isinstance(rec["metrics"][key], (int, float)), \
            f"{name}: metrics.{key} must be numeric"
    for key in ("gamma", "gamma_se", "z", "corrected_r2"):
        if rec["overlap"].get(key) is not None:
            assert isinstance(rec["overlap"][key], (int, float)), \
                f"{name}: overlap.{key} must be numeric"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_overlap_is_declared(name, rec):
    """Every record declares its training/target overlap (schema.md rule 1)."""
    assert rec["target"]["overlap"] in OVERLAP_DECLARATIONS, \
        f"{name}: target.overlap must be one of {sorted(OVERLAP_DECLARATIONS)}"
    assert rec["overlap"].get("role", "reference") in ROLES, \
        f"{name}: overlap.role must be one of {sorted(ROLES)}"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_suspects_carry_the_detector_fit(name, rec):
    """A 'suspect' is paired with a reference, so it must carry the detector fit.

    The corrected R^2 is required too, *unless* the record explains in a
    ``note`` why the correction does not apply (sparse / borderline scores are
    upper bounds -- schema.md rule 2, docs/OVERLAP.md). 'suspect-unpaired' has
    no reference to difference against and can never be corrected.
    """
    ov = rec["overlap"]
    if ov.get("role") == "suspect":
        for key in ("gamma", "gamma_se", "z", "reference"):
            assert ov.get(key) is not None, f"{name}: suspect is missing overlap.{key}"
        assert ov.get("corrected_r2") is not None or ov.get("note"), \
            f"{name}: suspect has no corrected_r2 and no note explaining why"
    if ov.get("role") == "suspect-unpaired":
        assert ov.get("corrected_r2") is None, \
            f"{name}: an unpaired suspect is an upper bound and cannot be corrected"


def _half_step(x):
    """Half of the last recorded decimal place of ``x`` -- its rounding radius.

    JSON floats round-trip through ``repr``, so the decimals actually written in
    the pack are recoverable and the identity below can be checked against the
    precision the record was stored at.
    """
    text = repr(float(x))
    if "e" in text or "E" in text:
        return 0.0                      # scientific notation: full precision
    _, _, frac = text.partition(".")
    return 0.5 * 10.0 ** -len(frac)


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_r2_matches_num_and_den(name, rec):
    """``metrics.r2 == num^2 / den`` — the identity schema.md defines.

    Checked against the interval implied by the recorded precision of ``num``
    and ``den``, so a reader can recompute (and audit) the headline number from
    the recorded quadratic forms.
    """
    m = rec["metrics"]
    assert m["den"] > 0, f"{name}: metrics.den must be positive"
    dn, dd, dr = _half_step(m["num"]), _half_step(m["den"]), _half_step(m["r2"])
    lo_num, hi_num = abs(m["num"]) - dn, abs(m["num"]) + dn
    lo = lo_num ** 2 / (m["den"] + dd) - dr      # r2 carries its own rounding too
    hi = hi_num ** 2 / max(m["den"] - dd, 1e-300) + dr
    assert lo <= m["r2"] <= hi, (
        f"{name}: r2={m['r2']} is outside [{lo:.6g}, {hi:.6g}], the range "
        f"num^2/den admits at the recorded precision of num and den")


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_corrected_r2_matches_the_numerator_correction(name, rec):
    """The corrected R^2 is the de-overlapped numerator over the same ``den``.

    ``overlap.m_total`` is the variant count the per-variant overlap term is
    summed over -- the LD reference's, not the score's. Records that omit it
    cannot be audited; schema.md requires it for new corrected records.
    """
    ov, m = rec["overlap"], rec["metrics"]
    if ov.get("corrected_r2") is None or ov.get("gamma") is None:
        pytest.skip("no correction recorded")
    if ov.get("m_total") is None:
        pytest.skip("record predates the overlap.m_total provenance field")
    num_corr = correct_numerator(m["num"], ov["gamma"], ov["m_total"])
    assert ov["corrected_r2"] == pytest.approx(num_corr ** 2 / m["den"], rel=5e-2), \
        f"{name}: corrected_r2 is not correct_numerator(num, gamma, m_total)^2 / den"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_match_fractions_are_fractions(name, rec):
    for key in ("w_match", "z_match"):
        assert 0.0 <= rec["metrics"][key] <= 1.0, f"{name}: metrics.{key} is not a fraction"
