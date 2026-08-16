#!/usr/bin/env python
"""Build the static PPB leaderboard site from the results registry.

Reads every ``results/*.json`` result pack (schema: ``results/schema.md``) and
writes a self-contained ``site/index.html`` (inline CSS/JS, sortable table).
Run from anywhere:  python scripts/build_leaderboard.py
"""
import html
import json
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Records arrive by pull request, so every interpolated field is untrusted:
# escape it, and only build a catalog link for an id we recognise.
_PGS_ID = re.compile(r"^PGS\d{6}$")

TRAIT_SCALES = {
    "quantitative": "quantitative correlation R2",
    "binary": "standardized logistic-summary approximation (not liability R2)",
}
OVERLAP_DECLARATIONS = {"none (declared)", "in-sample"}
OVERLAP_METHOD = "scaled_signal_eiv_v1"
ENVIRONMENT_REQUIRED_FROM = date(2026, 8, 16)

# Packs generated before scripts/regenerate_results.py recorded the
# environment (including baseline-2026-07.json) legitimately lack it -- the
# same grandfathering the block diagnostics use in results/schema.md.


def _reject_nonfinite_constant(value):
    raise ValueError(f"non-finite JSON constant {value!r}")


def _reject_nonfinite_numbers(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite_numbers(item)
    elif isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite_numbers(item)


def result_packs(result_dir):
    """``results/*.json`` files that are result packs.

    Dated anchor snapshots written by ``scripts/anchor_validation.py``
    (``anchor-*.json``, schema.md) are performance provenance objects, not
    record arrays; parsing them as packs would reject the directory outright.
    """
    return [path for path in sorted(Path(result_dir).glob("*.json"))
            if not path.name.startswith("anchor-")]


def load_records(root=ROOT):
    """Load strict JSON result packs and reject unsafe shapes and fields."""
    records = []
    root = Path(root)
    result_dir = root / "results" if (root / "results").is_dir() else root
    for path in result_packs(result_dir):
        try:
            data = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=_reject_nonfinite_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path.name}: invalid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise ValueError(f"{path.name}: top level must be a JSON array")
        if not data:
            raise ValueError(f"{path.name}: result pack must not be empty")
        try:
            _reject_nonfinite_numbers(data)
        except ValueError as exc:
            raise ValueError(f"{path.name}: invalid JSON: {exc}") from exc
        for i, rec in enumerate(data):
            if not isinstance(rec, dict):
                raise ValueError(f"{path.name}: record {i} must be a JSON object")
            # Field-level rules before rendering: a pack that violates them
            # would otherwise crash the Pages deploy with a bare KeyError or
            # ZeroDivisionError instead of a named rejection (schema.md is
            # otherwise enforced only by the test suite).
            try:
                validate_record(rec)
            except ValueError as exc:
                raise ValueError(f"{path.name}[{i}]: {exc}") from exc
            rec["_pack"] = path.name
            records.append(rec)
    return records


def _required(rec, dotted):
    value = rec
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing '{dotted}'")
        value = value[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{dotted}' must be a non-empty string")
    return value


def _lookup(rec, dotted):
    value = rec
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing '{dotted}'")
        value = value[key]
    return value


def _positive_int(rec, dotted):
    value = _lookup(rec, dotted)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'{dotted}' must be a positive integer")
    return value


def _finite_number(rec, dotted):
    value = _lookup(rec, dotted)
    if not isinstance(value, (int, float)) or isinstance(value, bool) \
            or not math.isfinite(value):
        raise ValueError(f"'{dotted}' must be a finite real number")
    return value


def validate_record(rec, where="record"):
    """Enforce the field-level rules of ``results/schema.md``.

    Covers every field the site renders plus the binding field rules (labels,
    positivity, fraction ranges, scale agreement, overlap contract, date and
    commit provenance, and the pack-generation-date requirement for
    ``environment``). Internal consistency of the optional block diagnostics
    stays with the test suite; here a violation must surface as a clean
    ``ValueError`` naming the record and field.
    """
    for dotted in (
        "trait", "ld_ref", "score.id", "score.name", "score.training",
        "target.gwas", "target.cohort", "target.ancestry",
        "target.n_eff_basis", "metrics.scale",
    ):
        _required(rec, dotted)
    for dotted in ("score", "target", "metrics", "overlap"):
        if not isinstance(rec.get(dotted), dict):
            raise ValueError(f"missing '{dotted}'")

    _positive_int(rec, "score.n_variants")
    _positive_int(rec, "target.n_eff")
    _positive_int(rec, "metrics.n_variants_scored")
    if rec["metrics"]["n_variants_scored"] > rec["score"]["n_variants"]:
        raise ValueError("'metrics.n_variants_scored' exceeds 'score.n_variants'")

    if _finite_number(rec, "metrics.den") <= 0:
        raise ValueError("'metrics.den' must be positive")
    for dotted in ("metrics.num", "metrics.r2", "metrics.w_match",
                   "metrics.z_match"):
        _finite_number(rec, dotted)
    for key in ("r2", "w_match", "z_match"):
        if not 0.0 <= rec["metrics"][key] <= 1.0:
            raise ValueError(f"'metrics.{key}' must lie in [0, 1]")

    trait_type = rec["target"].get("trait_type")
    if trait_type not in TRAIT_SCALES:
        raise ValueError(
            f"'target.trait_type' must be one of {sorted(TRAIT_SCALES)}")
    if rec["metrics"]["scale"] != TRAIT_SCALES[trait_type]:
        raise ValueError("'metrics.scale' does not match 'target.trait_type'")

    if rec["target"].get("overlap") not in OVERLAP_DECLARATIONS:
        raise ValueError(f"'target.overlap' must be one of "
                         f"{sorted(OVERLAP_DECLARATIONS)}")
    if "n_eff_range" in rec["target"]:
        n_range = rec["target"]["n_eff_range"]
        if not (isinstance(n_range, list) and len(n_range) == 2
                and all(isinstance(x, int) and not isinstance(x, bool) and x > 0
                        for x in n_range)):
            raise ValueError("'target.n_eff_range' must be two positive integers")
        lo, hi = n_range
        if not lo <= rec["target"]["n_eff"] <= hi:
            raise ValueError("'target.n_eff' must lie within 'target.n_eff_range'")

    ov = rec["overlap"]
    if ov.get("role") not in ROLE_STYLE:
        raise ValueError(f"'overlap.role' must be one of {sorted(ROLE_STYLE)}")
    if ov.get("status") not in STATUS_LABEL:
        raise ValueError(f"'overlap.status' must be one of {sorted(STATUS_LABEL)}")
    if ov.get("method") != OVERLAP_METHOD:
        raise ValueError(f"'overlap.method' must be {OVERLAP_METHOD!r}")
    declared = rec["target"]["overlap"] == "none (declared)"
    if declared != (ov["status"] == "not_applicable") \
            or declared != (ov["role"] == "reference"):
        raise ValueError(
            "a declared non-overlap must be a not_applicable reference, and "
            "an in-sample evaluation must remain an upper bound")
    if ov["status"] != "not_applicable":
        note = ov.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(
                "a refused correction must explain itself in 'overlap.note'")
    if ov.get("corrected_r2") is not None:
        raise ValueError("the registry does not accept a corrected R2")

    stamp = rec.get("date")
    if not isinstance(stamp, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", stamp):
        raise ValueError("'date' must be ISO YYYY-MM-DD")
    if date.fromisoformat(stamp) > datetime.now(timezone.utc).date():
        raise ValueError("'date' is in the future")
    commit = rec.get("ppb_commit")
    if not isinstance(commit, str) \
            or not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise ValueError(
            "'ppb_commit' must be a 7-40 character lowercase Git object id")

    environment = rec.get("environment")
    if environment is None:
        if date.fromisoformat(stamp) >= ENVIRONMENT_REQUIRED_FROM:
            raise ValueError(
                "packs dated " + ENVIRONMENT_REQUIRED_FROM.isoformat() +
                " or later must record 'environment.python/numpy/numba/ppb'")
    else:
        if not isinstance(environment, dict):
            raise ValueError("'environment' must be an object")
        for key in ("python", "numpy", "numba", "ppb"):
            version = environment.get(key)
            if not isinstance(version, str) or not version.strip():
                raise ValueError(
                    f"'environment.{key}' must be a non-empty string")


def esc(s):
    """HTML-escape for text *and* attribute contexts (quotes included)."""
    return html.escape(str(s), quote=True)


def score_link(score_id):
    """Catalog link for a well-formed PGS id; plain escaped text otherwise."""
    sid = esc(score_id)
    if not _PGS_ID.match(str(score_id)):
        return sid
    return f'<a href="https://www.pgscatalog.org/score/{sid}/">{sid}</a>'


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


ROLE_STYLE = {
    "reference": "#e8f5e9",
    "suspect": "#fff8e1",
    "suspect-unpaired": "#ffebee",
}

STATUS_LABEL = {
    "not_applicable": "not applicable",
    "basis_unavailable": "basis unavailable",
    "insufficient_data": "insufficient data",
    "excluded_basis": "incomplete basis",
    "nonidentifiable": "not identifiable",
    "weak_identification": "weak identification",
    "heterogeneous": "heterogeneous",
    "unstable": "unstable",
    "not_detected": "not detected",
    "sign_reversal": "sign reversal refused",
}

SCALE_LABEL = {
    "quantitative correlation R2": "quantitative correlation R²",
    "standardized logistic-summary approximation (not liability R2)":
        "binary approximation (not liability R²)",
}


def row(rec, include_legacy=False):
    m, ov, sc, tg = rec["metrics"], rec["overlap"], rec["score"], rec["target"]
    role = ov.get("role", "reference")
    status = ov.get("status", "unknown")
    color = ROLE_STYLE.get(role, "#fff")
    prefix = "reference" if role == "reference" else "upper bound"
    label = f"{prefix} · {STATUS_LABEL.get(status, esc(status))}"
    corrected = "—"
    z = fmt(ov.get("gamma_z"), 1) if ov.get("gamma_z") is not None else "—"
    ref = esc(ov.get("reference", "—"))
    note = f'<br><small>{esc(ov["note"])}</small>' if ov.get("note") else ""
    legacy_cell = ""
    if include_legacy:
        legacy = ov.get("legacy_unidentified")
        if legacy:
            parts = []
            if legacy.get("z") is not None:
                parts.append(f"z={fmt(legacy['z'], 1)}")
            if legacy.get("corrected_r2") is not None:
                parts.append(f"old corrected R²={fmt(legacy['corrected_r2'])}")
            detail = " · ".join(parts) if parts else "diagnostic retained"
            value = (
                f'<abbr title="{esc(legacy.get("warning", "unidentified legacy model"))}">'
                f"legacy v0 (unidentified)</abbr><br><small>{detail}</small>")
        else:
            value = "—"
        legacy_cell = f"<td>{value}</td>"
    scale = SCALE_LABEL.get(m.get("scale"), esc(m.get("scale", "undeclared")))
    support = m["n_variants_scored"] / sc["n_variants"]
    # n_eff means different things per target; expose its basis rather than
    # letting a bare number read as one comparable quantity across rows.
    basis = tg.get("n_eff_basis", "")
    if tg.get("n_eff_range"):
        lo, hi = tg["n_eff_range"]
        basis = f"{basis} ({lo:,}–{hi:,})" if basis else f"per-variant N {lo:,}–{hi:,}"
    n_eff = (f'<abbr title="{esc(basis)}">n_eff {tg["n_eff"]:,}</abbr>'
             if basis else f"n_eff {tg['n_eff']:,}")
    return (
        f'<tr style="background:{color}">'
        f"<td>{esc(rec['trait'])}</td>"
        f"<td>{score_link(sc['id'])}"
        f"<br><small>{esc(sc['name'])} · {support:.1%} target support</small></td>"
        f"<td>{esc(tg['gwas'])}<br><small>{esc(tg['cohort'])} · {n_eff}</small></td>"
        f"<td>{esc(tg['ancestry'])}</td>"
        f'<td data-sort="{m["r2"]:.6f}"><b>{m["r2"]:.4f}</b>'
        f"<br><small>{scale}</small></td>"
        f"<td>{label}{note}</td>"
        f"<td>{z}</td>"
        f"<td>{corrected}</td>"
        f"{legacy_cell}"
        f"<td><small>{ref}</small></td>"
        "</tr>")


def build(records):
    records = list(records)
    include_legacy = any(
        r.get("overlap", {}).get("legacy_unidentified") for r in records)
    rows = "\n".join(row(r, include_legacy=include_legacy) for r in records)
    legacy_note = (
        " Historical v0 values in this pack are shown solely as unidentified "
        "legacy diagnostics." if include_legacy else "")
    legacy_header = (
        "<th>Legacy v0 (unidentified)</th>" if include_legacy else "")
    # Packs may be generated at different commits; list every one represented.
    commits = sorted({str(r.get("ppb_commit", "?")) for r in records})
    commit = esc(", ".join(commits)) if commits else "?"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PPB leaderboard — PGS accuracy from summary statistics</title>
<style>
 body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 2rem auto; max-width: 1200px; color: #222; }}
 h1 {{ font-size: 1.4rem; margin-bottom: 0.2rem; }}
 .sub {{ color: #666; font-size: 0.9rem; margin-bottom: 1.2rem; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
 th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.55rem; text-align: left; vertical-align: top; }}
 th {{ background: #263238; color: #fff; cursor: pointer; user-select: none; white-space: nowrap; }}
 th:hover {{ background: #37474f; }}
 tr:hover td {{ filter: brightness(0.97); }}
 small {{ color: #555; }}
 .legend {{ font-size: 0.85rem; margin: 0.8rem 0; }}
 .chip {{ display: inline-block; padding: 0.1rem 0.5rem; border: 1px solid #ccc; margin-right: 0.4rem; }}
 footer {{ margin-top: 1.5rem; font-size: 0.8rem; color: #777; }}
 a {{ color: #1565c0; }}
</style></head><body>
<h1>PPB leaderboard — polygenic-score accuracy from summary statistics</h1>
<p class="sub">R&sup2; = (w&#7488;z)&sup2; / (w&#7488;Dw), computed without individual-level data;
its declared scale is shown per row, and binary values are not liability-scale R&sup2;.
Every evaluation declares its training/target sample overlap; in-sample rows are <b>upper bounds</b>.
Corrections are not published: see <code>docs/OVERLAP.md</code> for why the fit is dominated by evaluating against the independent reference it requires.
<a href="https://github.com/bvilhjal/ppb/blob/main/docs/OVERLAP.md">Method details</a>.
{legacy_note}</p>
<p class="legend">
 <span class="chip" style="background:#e8f5e9">reference — declared non-overlapping target</span>
 <span class="chip" style="background:#fff8e1">upper bound — paired with a reference</span>
 <span class="chip" style="background:#ffebee">upper bound — no reference</span>
</p>
<table id="lb"><thead><tr>
<th onclick="srt(this,0)">Trait</th><th onclick="srt(this,1)">Score</th>
<th onclick="srt(this,2)">Target GWAS</th><th onclick="srt(this,3)">Anc.</th>
<th onclick="srt(this,4,1)">R&sup2; (declared scale)</th><th onclick="srt(this,5)">Status</th>
<th onclick="srt(this,6,1)">current γ z</th><th onclick="srt(this,7,1)">validated R&sup2; correction</th>
{legacy_header}<th>reference (R&sup2;)</th></tr></thead><tbody>
{rows}
</tbody></table>
<footer>Generated {now} from <a href="https://github.com/bvilhjal/ppb/tree/main/results">results/</a>
(ppb commit {commit}) by <code>scripts/build_leaderboard.py</code> ·
<a href="https://github.com/bvilhjal/ppb">github.com/bvilhjal/ppb</a> ·
<a href="https://github.com/bvilhjal/ppb/blob/main/results/schema.md">record schema</a>
</footer>
<script>
let dir = {{}};
function srt(th, col, num) {{
  dir[col] = !dir[col];
  const tb = document.querySelector("#lb tbody");
  [...tb.rows].sort((a, b) => {{
    let x = a.cells[col].dataset.sort || a.cells[col].innerText,
        y = b.cells[col].dataset.sort || b.cells[col].innerText;
    if (num) {{ x = parseFloat(x) || 0; y = parseFloat(y) || 0; return dir[col] ? x - y : y - x; }}
    return dir[col] ? x.localeCompare(y) : y.localeCompare(x);
  }}).forEach(r => tb.appendChild(r));
}}
</script>
</body></html>"""


def main():
    records = load_records()
    out = ROOT / "site" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(records), encoding="utf-8")
    print(f"{len(records)} records -> {out}")


if __name__ == "__main__":
    main()
