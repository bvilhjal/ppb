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
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Records arrive by pull request, so every interpolated field is untrusted:
# escape it, and only build a catalog link for an id we recognise.
_PGS_ID = re.compile(r"^PGS\d{6}$")


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


def load_records(root=ROOT):
    """Load strict JSON result packs and reject unsafe top-level shapes."""
    records = []
    root = Path(root)
    result_dir = root / "results" if (root / "results").is_dir() else root
    for path in sorted(result_dir.glob("*.json")):
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
            rec["_pack"] = path.name
            records.append(rec)
    return records


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
    "correctable": "correctable",
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
    corrected = (fmt(ov.get("corrected_r2"))
                 if status == "correctable" else "—")
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
A correction is displayed only for a basis-aware fit whose status is <code>correctable</code>.
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
