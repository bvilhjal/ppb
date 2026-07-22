"""Schema checks for the results registry (``results/*.json``).

The registry is the public-facing artifact: ``scripts/build_leaderboard.py``
renders it straight into the deployed site, and records are immutable once
merged. These tests enforce the rules stated in ``results/schema.md`` at PR
time, rather than letting a malformed record fail in the Pages deploy job.
"""

import json
import math
import re
from datetime import date
from pathlib import Path

import pytest

from ppb.overlap import correct_numerator
from scripts.build_leaderboard import load_records

ROOT = Path(__file__).resolve().parent.parent
PACKS = sorted((ROOT / "results").glob("*.json"))

ROLES = {"reference", "suspect", "suspect-unpaired"}
OVERLAP_DECLARATIONS = {"none (declared)", "in-sample"}
PER_VARIANT_N_BASIS = "median of the per-variant N column"
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _reject_nonfinite_constant(value):
    raise ValueError(f"non-finite JSON constant {value!r}")


def _load_pack(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_constant,
    )


PACK_DATA = {}
PACK_ERRORS = {}
for _pack in PACKS:
    try:
        PACK_DATA[_pack] = _load_pack(_pack)
    except (json.JSONDecodeError, ValueError) as _error:
        PACK_ERRORS[_pack] = _error


def _records():
    for pack, data in PACK_DATA.items():
        if not isinstance(data, list):
            continue
        for i, rec in enumerate(data):
            if isinstance(rec, dict):
                yield f"{pack.name}[{i}]", rec


RECORDS = list(_records())
IDS = [name for name, _ in RECORDS]


def test_registry_is_not_empty():
    assert PACKS, "no result packs found in results/"
    assert RECORDS


@pytest.mark.parametrize("pack", PACKS, ids=lambda p: p.name)
def test_pack_is_strict_json_array_of_objects(pack):
    assert pack not in PACK_ERRORS, f"{pack.name}: invalid JSON: {PACK_ERRORS.get(pack)}"
    data = PACK_DATA[pack]
    assert isinstance(data, list), f"{pack.name}: top level must be a JSON array"
    assert data, f"{pack.name}: result pack must not be empty"
    for i, rec in enumerate(data):
        assert isinstance(rec, dict), f"{pack.name}[{i}]: record must be a JSON object"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_record_has_required_fields(name, rec):
    for key in (
        "trait", "score", "target", "ld_ref", "metrics", "overlap", "date",
        "ppb_commit",
    ):
        assert key in rec, f"{name}: missing '{key}'"
    for key in ("score", "target", "metrics", "overlap"):
        assert isinstance(rec[key], dict), f"{name}: '{key}' must be an object"
    for key in ("id", "name", "training", "n_variants"):
        assert key in rec["score"], f"{name}: missing 'score.{key}'"
    for key in ("gwas", "cohort", "ancestry", "n_eff", "n_eff_basis", "overlap"):
        assert key in rec["target"], f"{name}: missing 'target.{key}'"
    for key in ("num", "den", "r2", "w_match", "z_match"):
        assert key in rec["metrics"], f"{name}: missing 'metrics.{key}'"
    assert "role" in rec["overlap"], f"{name}: missing 'overlap.role'"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_record_text_fields_are_nonempty(name, rec):
    paths = (
        ("trait",), ("ld_ref",), ("score", "id"), ("score", "name"),
        ("score", "training"), ("target", "gwas"), ("target", "cohort"),
        ("target", "ancestry"), ("target", "n_eff_basis"),
    )
    for path in paths:
        value = rec
        for key in path:
            value = value[key]
        dotted = ".".join(path)
        assert isinstance(value, str) and value.strip(), \
            f"{name}: {dotted} must be a non-empty string"


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_record_numeric_fields_are_finite_real_numbers(name, rec):
    """Numerics rendered by the site cannot be booleans, NaN, or infinities."""
    for key in ("num", "den", "r2", "w_match", "z_match"):
        assert _finite_number(rec["metrics"][key]), \
            f"{name}: metrics.{key} must be a finite real number"
    for key in ("gamma", "gamma_se", "z", "corrected_r2"):
        if rec["overlap"].get(key) is not None:
            assert _finite_number(rec["overlap"][key]), \
                f"{name}: overlap.{key} must be a finite real number"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_counts_and_sample_sizes_are_positive_integers(name, rec):
    assert _positive_int(rec["score"]["n_variants"]), \
        f"{name}: score.n_variants must be a positive integer"
    assert _positive_int(rec["target"]["n_eff"]), \
        f"{name}: target.n_eff must be a positive integer"
    if rec["overlap"].get("m_total") is not None:
        assert _positive_int(rec["overlap"]["m_total"]), \
            f"{name}: overlap.m_total must be a positive integer"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_per_variant_n_provenance_is_consistent(name, rec):
    target = rec["target"]
    is_per_variant = target["n_eff_basis"] == PER_VARIANT_N_BASIS
    has_range = "n_eff_range" in target
    assert has_range == is_per_variant, (
        f"{name}: target.n_eff_range is required exactly when n_eff_basis is "
        f"'{PER_VARIANT_N_BASIS}'")
    if not has_range:
        return
    n_range = target["n_eff_range"]
    assert isinstance(n_range, list) and len(n_range) == 2, \
        f"{name}: target.n_eff_range must be [min, max]"
    assert all(_positive_int(x) for x in n_range), \
        f"{name}: target.n_eff_range values must be positive integers"
    lo, hi = n_range
    assert lo <= target["n_eff"] <= hi, \
        f"{name}: target.n_eff must lie within target.n_eff_range"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_provenance_is_sane(name, rec):
    stamp = rec["date"]
    assert isinstance(stamp, str) and DATE_RE.fullmatch(stamp), \
        f"{name}: date must be ISO YYYY-MM-DD"
    assert date.fromisoformat(stamp) <= date.today(), f"{name}: date is in the future"
    commit = rec["ppb_commit"]
    assert isinstance(commit, str) and COMMIT_RE.fullmatch(commit), \
        f"{name}: ppb_commit must be a 7-40 character lowercase Git object id"


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
    assert 0.0 <= m["r2"] <= 1.0, f"{name}: metrics.r2 must lie in [0, 1]"
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
    assert ov.get("m_total") is not None, \
        f"{name}: corrected record is missing overlap.m_total"
    assert 0.0 <= ov["corrected_r2"] <= 1.0, \
        f"{name}: overlap.corrected_r2 must lie in [0, 1]"
    num_corr = correct_numerator(m["num"], ov["gamma"], ov["m_total"])
    assert ov["corrected_r2"] == pytest.approx(num_corr ** 2 / m["den"], rel=5e-2), \
        f"{name}: corrected_r2 is not correct_numerator(num, gamma, m_total)^2 / den"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_match_fractions_are_fractions(name, rec):
    for key in ("w_match", "z_match"):
        assert 0.0 <= rec["metrics"][key] <= 1.0, f"{name}: metrics.{key} is not a fraction"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_overlap_z_matches_gamma_over_se(name, rec):
    ov = rec["overlap"]
    if ov.get("gamma") is None:
        pytest.skip("no detector fit recorded")
    assert ov["gamma_se"] > 0, f"{name}: overlap.gamma_se must be positive"
    assert ov["z"] == pytest.approx(ov["gamma"] / ov["gamma_se"], rel=1e-12), \
        f"{name}: overlap.z must equal overlap.gamma / overlap.gamma_se"


def test_evaluation_identities_are_unique():
    """One versioned evaluation must not render twice under different packs."""
    identities = []
    for _, rec in RECORDS:
        identities.append((
            rec["trait"], rec["score"]["id"], rec["target"]["gwas"],
            rec["target"]["cohort"], rec["target"]["ancestry"], rec["date"],
            rec["ppb_commit"],
        ))
    duplicates = {identity for identity in identities if identities.count(identity) > 1}
    assert not duplicates, f"duplicate evaluation identities: {sorted(duplicates)}"


def test_leaderboard_loader_reads_the_validated_registry():
    loaded = load_records()
    assert len(loaded) == len(RECORDS)
    assert all(rec.get("_pack") for rec in loaded)


@pytest.mark.parametrize(
    "payload,error",
    [
        ('[{"metrics": {"r2": NaN}}]', "non-finite"),
        ('[{"metrics": {"r2": Infinity}}]', "non-finite"),
        ('[{"metrics": {"r2": 1e999}}]', "non-finite"),
        ('{"record": {}}', "top level"),
        ('[]', "must not be empty"),
        ('[1]', "record 0"),
    ],
)
def test_leaderboard_loader_rejects_unsafe_pack_shapes(tmp_path, payload, error):
    (tmp_path / "bad.json").write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_records(tmp_path)
