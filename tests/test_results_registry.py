"""Schema checks for the results registry (``results/*.json``).

The registry is the public-facing artifact: ``scripts/build_leaderboard.py``
renders it straight into the deployed site, and records are immutable once
merged. These tests enforce the rules stated in ``results/schema.md`` at PR
time, rather than letting a malformed record fail in the Pages deploy job.
"""

import copy
import json
import math
import re
from datetime import date
from pathlib import Path

import pytest

from scripts.build_leaderboard import build, load_records, result_packs

ROOT = Path(__file__).resolve().parent.parent
# results/ also carries dated anchor snapshots (schema.md); they are
# provenance objects, not packs, and must not be validated as such.
PACKS = result_packs(ROOT / "results")

ROLES = {"reference", "suspect", "suspect-unpaired"}
OVERLAP_DECLARATIONS = {"none (declared)", "in-sample"}
TRAIT_TYPES = {"quantitative", "binary"}
SCALES = {
    "quantitative": "quantitative correlation R2",
    "binary": "standardized logistic-summary approximation (not liability R2)",
}
OVERLAP_METHOD = "scaled_signal_eiv_v1"
OVERLAP_STATUSES = {
    "not_applicable", "basis_unavailable", "insufficient_data",
    "excluded_basis", "nonidentifiable", "weak_identification",
    "heterogeneous", "unstable", "not_detected", "sign_reversal",
}
AVAILABLE_BASIS_KINDS = {"linear_trace", "jacobian_hutchinson"}
LEGACY_METHOD = "fixed_signal_variant_count_v0"
PER_VARIANT_N_BASIS = "median of the per-variant N column"
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Packs dated on/after this day must record their interpreter/package
# environment; earlier packs predate the field (schema.md grandfathering).
ENVIRONMENT_REQUIRED_FROM = date(2026, 8, 16)


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
    for key in (
        "gwas", "cohort", "ancestry", "trait_type", "n_eff",
        "n_eff_basis", "overlap",
    ):
        assert key in rec["target"], f"{name}: missing 'target.{key}'"
    for key in (
        "num", "den", "r2", "scale", "w_match", "z_match",
        "n_variants_scored",
    ):
        assert key in rec["metrics"], f"{name}: missing 'metrics.{key}'"
    for key in ("role", "method", "status"):
        assert key in rec["overlap"], f"{name}: missing 'overlap.{key}'"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_record_text_fields_are_nonempty(name, rec):
    paths = (
        ("trait",), ("ld_ref",), ("score", "id"), ("score", "name"),
        ("score", "training"), ("target", "gwas"), ("target", "cohort"),
        ("target", "ancestry"), ("target", "n_eff_basis"),
        ("metrics", "scale"), ("overlap", "method"),
        ("overlap", "status"),
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
    current_fit_numbers = (
        "alpha", "alpha_se", "gamma", "gamma_se", "gamma_z", "q_total",
        "q_fit", "numerator_target", "excluded_basis_fraction",
        "weighted_correlation", "vif", "condition_number",
        "heterogeneity_ratio", "corrected_r2",
    )
    for key in current_fit_numbers:
        if rec["overlap"].get(key) is not None:
            assert _finite_number(rec["overlap"][key]), \
                f"{name}: overlap.{key} must be a finite real number"
    basis = rec["overlap"].get("basis")
    if basis and basis.get("mc_se") is not None:
        assert _finite_number(basis["mc_se"]), \
            f"{name}: overlap.basis.mc_se must be a finite real number"
    legacy = rec["overlap"].get("legacy_unidentified")
    if legacy:
        for key in ("gamma", "gamma_se", "z", "corrected_r2"):
            if legacy.get(key) is not None:
                assert _finite_number(legacy[key]), \
                    f"{name}: overlap.legacy_unidentified.{key} must be finite"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_counts_and_sample_sizes_are_positive_integers(name, rec):
    assert _positive_int(rec["score"]["n_variants"]), \
        f"{name}: score.n_variants must be a positive integer"
    assert _positive_int(rec["target"]["n_eff"]), \
        f"{name}: target.n_eff must be a positive integer"
    n_scored = rec["metrics"]["n_variants_scored"]
    assert _positive_int(n_scored), \
        f"{name}: metrics.n_variants_scored must be a positive integer"
    assert n_scored <= rec["score"]["n_variants"], \
        f"{name}: metrics.n_variants_scored exceeds score.n_variants"
    for key in ("n_blocks", "n_groups"):
        if rec["overlap"].get(key) is not None:
            assert _positive_int(rec["overlap"][key]), \
                f"{name}: overlap.{key} must be a positive integer"
    legacy = rec["overlap"].get("legacy_unidentified")
    if legacy and legacy.get("m_total") is not None:
        assert _positive_int(legacy["m_total"]), \
            f"{name}: overlap.legacy_unidentified.m_total must be positive"


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
def test_environment_is_recorded_for_new_packs(name, rec):
    """A pack that names its commit but not its interpreter is not replayable;
    packs older than the field itself legitimately lack it."""
    environment = rec.get("environment")
    if environment is None:
        assert date.fromisoformat(rec["date"]) < ENVIRONMENT_REQUIRED_FROM, (
            f"{name}: packs dated {ENVIRONMENT_REQUIRED_FROM.isoformat()} or "
            "later must record environment.python/numpy/numba/ppb")
        return
    assert isinstance(environment, dict), \
        f"{name}: environment must be an object"
    for key in ("python", "numpy", "numba", "ppb"):
        version = environment.get(key)
        assert isinstance(version, str) and version.strip(), \
            f"{name}: environment.{key} must be a non-empty string"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_harmonization_loss_reasons_are_recorded_consistently(name, rec):
    """The per-reason losses must travel together and cannot exceed the
    unmatched fraction of the score; packs predating them legitimately lack
    the fields (schema.md grandfathering)."""
    metrics = rec["metrics"]
    for side in ("w", "z"):
        keys = [f"{side}_n_{reason}"
                for reason in ("ambiguous_removed", "mismatch", "unmatched")]
        present = [key for key in keys if key in metrics]
        assert len(present) in (0, len(keys)), \
            f"{name}: metrics per-reason losses must be recorded together"
        for key in present:
            value = metrics[key]
            assert isinstance(value, int) and not isinstance(value, bool) \
                and value >= 0, \
                f"{name}: metrics.{key} must be a non-negative integer"
    if "w_n_unmatched" in metrics:
        losses = sum(metrics[f"w_n_{reason}"]
                     for reason in ("ambiguous_removed", "mismatch", "unmatched"))
        n_variants = rec["score"]["n_variants"]
        assert metrics["w_match"] + losses / n_variants <= 1.0 + 1e-9, (
            f"{name}: weight-side losses exceed the unmatched fraction of the "
            "score; the loss counts and w_match disagree")


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_trait_type_and_metric_scale_agree(name, rec):
    trait_type = rec["target"]["trait_type"]
    assert trait_type in TRAIT_TYPES, \
        f"{name}: target.trait_type must be one of {sorted(TRAIT_TYPES)}"
    assert rec["metrics"]["scale"] == SCALES[trait_type], (
        f"{name}: metrics.scale does not match target.trait_type={trait_type!r}")


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_overlap_is_declared(name, rec):
    """Declared overlap, registry role, and current status must agree."""
    assert rec["target"]["overlap"] in OVERLAP_DECLARATIONS, \
        f"{name}: target.overlap must be one of {sorted(OVERLAP_DECLARATIONS)}"
    ov = rec["overlap"]
    assert ov["role"] in ROLES, \
        f"{name}: overlap.role must be one of {sorted(ROLES)}"
    assert ov["method"] == OVERLAP_METHOD, \
        f"{name}: overlap.method must be {OVERLAP_METHOD!r}"
    assert ov["status"] in OVERLAP_STATUSES, \
        f"{name}: unknown overlap.status {ov['status']!r}"
    if rec["target"]["overlap"] == "none (declared)":
        assert ov["role"] == "reference" and ov["status"] == "not_applicable", \
            f"{name}: declared non-overlap must be a not_applicable reference"
    else:
        assert ov["role"] != "reference" and ov["status"] != "not_applicable", \
            f"{name}: in-sample evaluations must remain upper bounds"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_overlap_basis_and_fail_closed_contract(name, rec):
    """Only an accepted basis-aware fit may publish a current correction."""
    ov = rec["overlap"]
    assert "z" not in ov and "m_total" not in ov, \
        f"{name}: pre-v1 fields must be nested under legacy_unidentified"

    if ov["role"] == "suspect":
        assert isinstance(ov.get("reference"), str) and ov["reference"].strip(), \
            f"{name}: paired suspect is missing overlap.reference"
    if ov["role"] == "suspect-unpaired":
        assert "reference" not in ov, f"{name}: unpaired suspect has a reference"

    if ov["status"] != "not_applicable":
        assert isinstance(ov.get("note"), str) and ov["note"].strip(), \
            f"{name}: refused correction must explain overlap.note"

    basis = ov.get("basis")
    if ov["status"] == "not_applicable":
        assert ov.get("corrected_r2") is None, \
            f"{name}: reference cannot carry corrected_r2"
        return
    assert isinstance(basis, dict), f"{name}: current overlap fit needs a basis object"
    assert isinstance(basis.get("provenance"), str) and basis["provenance"].strip(), \
        f"{name}: overlap.basis.provenance must be non-empty"
    if ov["status"] == "basis_unavailable":
        assert basis.get("kind") == "unavailable", \
            f"{name}: basis_unavailable status needs kind=unavailable"
        assert ov.get("corrected_r2") is None, \
            f"{name}: unavailable basis cannot carry corrected_r2"
        return

    assert basis.get("kind") in AVAILABLE_BASIS_KINDS, \
        f"{name}: available overlap basis has an unknown kind"
    assert isinstance(basis.get("support_hash"), str) and basis["support_hash"].strip(), \
        f"{name}: available overlap basis needs its exact support hash"
    assert ov.get("corrected_r2") is None, \
        f"{name}: the registry does not accept a corrected R2"


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
def test_registry_never_publishes_a_correction(name, rec):
    """The registry stores detection only; corrections are not a published path.

    The experimental apparatus (``experiments/overlap_detection.py``) can still
    fit one (``docs/OVERLAP.md``), but
    the correction is dominated by simply evaluating against the independent
    reference it requires, so no pack may present one as a result.
    """
    ov = rec["overlap"]
    assert ov["status"] != "correctable", \
        f"{name}: 'correctable' is not a registry status"
    assert ov.get("corrected_r2") is None, \
        f"{name}: the registry does not accept a corrected R2"

@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_match_fractions_are_fractions(name, rec):
    for key in ("w_match", "z_match"):
        assert 0.0 <= rec["metrics"][key] <= 1.0, f"{name}: metrics.{key} is not a fraction"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_overlap_z_matches_gamma_over_se(name, rec):
    ov = rec["overlap"]
    if ov.get("gamma") is None:
        pytest.skip("no current fit recorded")
    assert ov["gamma_se"] > 0, f"{name}: overlap.gamma_se must be positive"
    assert ov["gamma_z"] == pytest.approx(ov["gamma"] / ov["gamma_se"], rel=1e-12), \
        f"{name}: overlap.gamma_z must equal gamma / gamma_se"


def _synthetic_correctable_record():
    rec = copy.deepcopy(RECORDS[1][1])
    ov = {
        "role": "suspect",
        "method": OVERLAP_METHOD,
        "status": "correctable",
        "basis": {
            "kind": "linear_trace",
            "provenance": "synthetic auditable trainer operator",
            "support_hash": "sha256:synthetic",
        },
        "reference": "synthetic independent reference",
        "alpha": 1.2,
        "alpha_se": 0.1,
        "gamma": 0.4,
        "gamma_se": 0.1,
        "gamma_z": 4.0,
        "q_total": 0.5,
        "numerator_target": 1.0,
    }
    ov["corrected_r2"] = (
        ov["numerator_target"] - ov["gamma"] * ov["q_total"]
    ) ** 2 / rec["metrics"]["den"]
    rec["overlap"] = ov
    return rec

def test_a_correctable_record_is_rejected_by_the_registry():
    """A well-formed correctable fit is still refused: the status is not a
    registry outcome, and its corrected_r2 is not a publishable result."""
    rec = _synthetic_correctable_record()
    with pytest.raises(AssertionError, match="not a registry status"):
        test_registry_never_publishes_a_correction("synthetic", rec)
    with pytest.raises(AssertionError, match="unknown overlap.status"):
        test_overlap_is_declared("synthetic", rec)


def test_a_correctable_record_without_its_status_still_cannot_publish_r2():
    """Relabelling the status does not smuggle the correction back in."""
    rec = _synthetic_correctable_record()
    rec["overlap"]["status"] = "not_detected"
    rec["overlap"]["note"] = "relabelled"
    with pytest.raises(AssertionError, match="does not accept a corrected R2"):
        test_registry_never_publishes_a_correction("synthetic", rec)


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_legacy_overlap_diagnostics_are_quarantined_but_auditable(name, rec):
    """Pre-v1 values remain reproducible, but never become current evidence."""
    ov, m = rec["overlap"], rec["metrics"]
    legacy = ov.get("legacy_unidentified")
    if legacy is None:
        pytest.skip("no legacy diagnostic recorded")
    assert ov["role"] == "suspect", \
        f"{name}: legacy detector output needs its paired suspect context"
    assert legacy.get("method") == LEGACY_METHOD, \
        f"{name}: unknown legacy overlap method"
    assert isinstance(legacy.get("warning"), str) and legacy["warning"].strip(), \
        f"{name}: legacy diagnostic needs an explicit warning"
    for key in ("gamma", "gamma_se", "z", "m_total"):
        assert legacy.get(key) is not None, \
            f"{name}: legacy diagnostic is missing {key}"
    assert legacy["gamma_se"] > 0, \
        f"{name}: legacy gamma_se must be positive"
    assert legacy["z"] == pytest.approx(
        legacy["gamma"] / legacy["gamma_se"], rel=1e-12), \
        f"{name}: legacy z must equal gamma / gamma_se"
    if legacy.get("corrected_r2") is not None:
        old_num = m["num"] - legacy["gamma"] * legacy["m_total"]
        assert legacy["corrected_r2"] == pytest.approx(
            old_num ** 2 / m["den"], rel=5e-2), \
            f"{name}: legacy corrected_r2 cannot be reproduced"


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


def test_leaderboard_labels_scales_and_quarantines_legacy_values():
    rendered = build(load_records())
    assert "binary approximation (not liability R²)" in rendered
    assert "Corrections are not published" in rendered
    assert "validated R&sup2; correction" in rendered
    assert "91.8% target support" in rendered
    assert "Legacy v0 (unidentified)" not in rendered
    assert "old corrected R²=" not in rendered

    legacy_rec = copy.deepcopy(RECORDS[1][1])
    legacy = {
        "method": LEGACY_METHOD,
        "warning": "synthetic unidentified legacy model",
        "gamma": 1e-7,
        "gamma_se": 5e-8,
        "z": 2.0,
        "m_total": 1000,
    }
    old_num = legacy_rec["metrics"]["num"] - legacy["gamma"] * legacy["m_total"]
    legacy["corrected_r2"] = old_num ** 2 / legacy_rec["metrics"]["den"]
    legacy_rec["overlap"]["legacy_unidentified"] = legacy

    test_legacy_overlap_diagnostics_are_quarantined_but_auditable(
        "synthetic", legacy_rec)
    rendered_legacy = build([legacy_rec])
    assert "Legacy v0 (unidentified)" in rendered_legacy
    assert "legacy v0 (unidentified)" in rendered_legacy
    assert "old corrected R²=" in rendered_legacy


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


def test_leaderboard_loader_rejects_field_invalid_records(tmp_path):
    """A structurally valid pack that violates the field-level rules must be
    rejected with a ValueError naming the record and field -- not crash the
    render with KeyError (missing field) or ZeroDivisionError (n_variants 0)
    in the Pages deploy job."""
    base = copy.deepcopy(RECORDS[0][1])
    base.pop("_pack", None)

    def _in_sample_without_note(r):
        r["target"]["overlap"] = "in-sample"
        r["overlap"]["role"] = "suspect"
        r["overlap"]["status"] = "not_detected"
        r["overlap"].pop("note", None)

    mutations = [
        (lambda r: r["metrics"].pop("den"), r"missing 'metrics\.den'"),
        (lambda r: r["metrics"].update(den=0.0), "must be positive"),
        (lambda r: r["metrics"].update(r2=1.5), r"must lie in \[0, 1\]"),
        (lambda r: r["metrics"].update(w_match=-0.1), r"must lie in \[0, 1\]"),
        (lambda r: r["score"].update(n_variants=0), "positive integer"),
        (lambda r: r["target"].update(trait_type="categorical"), "trait_type"),
        (lambda r: r["metrics"].update(scale="made up"), "trait_type"),
        (_in_sample_without_note, "must explain itself"),
        (lambda r: r["overlap"].update(corrected_r2=0.5), "corrected R2"),
        (lambda r: r.update(date="2026-08-16"), "environment"),
        (lambda r: r["target"].update(n_eff_range=[10, 20]), "n_eff_range"),
    ]
    for mutate, match in mutations:
        rec = copy.deepcopy(base)
        mutate(rec)
        (tmp_path / "bad.json").write_text(json.dumps([rec]), encoding="utf-8")
        with pytest.raises(ValueError, match=match) as excinfo:
            load_records(tmp_path)
        assert str(excinfo.value).startswith("bad.json[0]:"), \
            "the rejection must name the record"


# ---------------------------------------------------------------------------
# Block-level uncertainty and negative control (optional; packs generated before
# scripts/regenerate_results.py recorded them legitimately lack these fields).


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_jackknife_blocks_are_internally_consistent(name, rec):
    """A recorded SE must be a finite non-negative number over a real grouping.

    The grouping is a partition of the blocks, so there can never be more
    delete-one groups than blocks, and a variance share is a fraction.
    """
    for key in ("jackknife", "jackknife_chromosome"):
        block = rec["metrics"].get(key)
        if block is None:
            continue
        assert isinstance(block, dict), f"{name}: metrics.{key} must be an object"
        for field in ("method", "se", "n_blocks", "n_groups", "max_variance_share"):
            assert field in block, f"{name}: missing metrics.{key}.{field}"
        assert isinstance(block["method"], str) and block["method"].strip()
        assert _finite_number(block["se"]) and block["se"] >= 0.0, \
            f"{name}: metrics.{key}.se must be finite and non-negative"
        assert _positive_int(block["n_blocks"]) and _positive_int(block["n_groups"])
        assert block["n_groups"] >= 2, \
            f"{name}: metrics.{key} needs at least 2 delete-one groups"
        assert block["n_groups"] <= block["n_blocks"], \
            f"{name}: metrics.{key} has more groups than blocks"
        share = block["max_variance_share"]
        assert _finite_number(share) and 0.0 <= share <= 1.0, \
            f"{name}: metrics.{key}.max_variance_share must be a fraction"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_sign_flip_null_is_internally_consistent(name, rec):
    """The control must be a real null: positive expectation, |z| within its
    Cauchy-Schwarz ceiling, and a ratio that matches the recorded R^2."""
    control = rec["metrics"].get("sign_flip_null")
    if control is None:
        return
    for field in ("method", "null_mean", "z", "ratio", "n_blocks", "z_ceiling"):
        assert field in control, f"{name}: missing metrics.sign_flip_null.{field}"
    assert _positive_int(control["n_blocks"])
    assert _finite_number(control["null_mean"]) and control["null_mean"] > 0.0, \
        f"{name}: sign_flip_null.null_mean must be positive"
    assert _finite_number(control["z"])
    ceiling = math.sqrt(control["n_blocks"])
    assert control["z_ceiling"] == pytest.approx(ceiling, rel=1e-9)
    assert abs(control["z"]) <= ceiling + 1e-9, \
        f"{name}: |z| = {control['z']} exceeds sqrt(n_blocks) = {ceiling}"
    assert control["ratio"] == pytest.approx(
        rec["metrics"]["r2"] / control["null_mean"], rel=1e-6), \
        f"{name}: sign_flip_null.ratio must equal r2 / null_mean"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_per_chromosome_sums_reproduce_the_headline_metrics(name, rec):
    """The per-chromosome partial sums exist so a reader can recompute the
    jackknife; they must therefore add back up to the recorded num and den."""
    per_chrom = rec["metrics"].get("per_chromosome")
    if per_chrom is None:
        return
    assert isinstance(per_chrom, dict) and per_chrom, \
        f"{name}: metrics.per_chromosome must be a non-empty object"
    total_u = total_v = 0.0
    for chrom, pair in per_chrom.items():
        assert isinstance(chrom, str) and chrom.strip()
        assert isinstance(pair, list) and len(pair) == 2, \
            f"{name}: per_chromosome[{chrom}] must be [u, v]"
        u, v = pair
        assert _finite_number(u) and _finite_number(v)
        assert v >= 0.0, f"{name}: per_chromosome[{chrom}] has negative score variance"
        total_u += u
        total_v += v
    assert total_u == pytest.approx(rec["metrics"]["num"], rel=1e-9), \
        f"{name}: per_chromosome u does not sum to metrics.num"
    assert total_v == pytest.approx(rec["metrics"]["den"], rel=1e-9), \
        f"{name}: per_chromosome v does not sum to metrics.den"


@pytest.mark.parametrize("name,rec", RECORDS, ids=IDS)
def test_missing_diagnostics_declare_why(name, rec):
    """A record without the block diagnostics must say so explicitly rather than
    leaving a reader to guess whether they were run and omitted."""
    note = rec["metrics"].get("diagnostics_unavailable")
    if note is not None:
        assert isinstance(note, str) and note.strip(), \
            f"{name}: metrics.diagnostics_unavailable must explain the omission"
        assert "jackknife" not in rec["metrics"], \
            f"{name}: cannot both declare diagnostics unavailable and record them"
