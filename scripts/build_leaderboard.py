#!/usr/bin/env python
"""Build the static PPB leaderboard site from the results registry.

Reads every ``results/*.json`` result pack (schema: ``results/schema.md``) and
writes a self-contained ``site/index.html`` (inline CSS/JS, sortable table).
Run from anywhere:  python scripts/build_leaderboard.py
"""
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Records arrive by pull request, so every interpolated field is untrusted:
# escape it, and only build a catalog link for an id we recognise.
_PGS_ID = re.compile(r"^PGS\d{6}$")


def load_records():
    records = []
    for path in sorted(ROOT.glob("results/*.json")):
        for rec in json.loads(path.read_text(encoding="utf-8")):
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


STATUS = {
    "reference": ("reference", "#e8f5e9"),
    "suspect": ("overlap suspect", "#fff8e1"),
    "suspect-unpaired": ("upper bound", "#ffebee"),
}


def row(rec):
    m, ov, sc, tg = rec["metrics"], rec["overlap"], rec["score"], rec["target"]
    role = ov.get("role", "reference")
    label, color = STATUS.get(role, (esc(role), "#fff"))
    corrected = fmt(ov.get("corrected_r2")) if ov.get("corrected_r2") is not None else "—"
    z = fmt(ov.get("z"), 1) if ov.get("z") is not None else "—"
    ref = esc(ov.get("reference", "—"))
    note = f'<br><small>{esc(ov["note"])}</small>' if ov.get("note") else ""
    return (
        f'<tr style="background:{color}">'
        f"<td>{esc(rec['trait'])}</td>"
        f"<td>{score_link(sc['id'])}"
        f"<br><small>{esc(sc['name'])}</small></td>"
        f"<td>{esc(tg['gwas'])}<br><small>{esc(tg['cohort'])} · n_eff {tg['n_eff']:,}</small></td>"
        f"<td>{esc(tg['ancestry'])}</td>"
        f'<td data-sort="{m["r2"]:.6f}"><b>{m["r2"]:.4f}</b></td>'
        f"<td>{label}{note}</td>"
        f"<td>{z}</td>"
        f"<td>{corrected}</td>"
        f"<td><small>{ref}</small></td>"
        "</tr>")


def build(records):
    rows = "\n".join(row(r) for r in records)
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
<p class="sub">R&sup2; = (w&#7488;z)&sup2; / (w&#7488;Dw), computed without individual-level data.
Every evaluation declares its training/target sample overlap; in-sample rows are <b>upper bounds</b>,
never accuracy measurements. Overlap z &gt; 2 flags detected training/target overlap
(<a href="https://github.com/bvilhjal/ppb/blob/main/docs/OVERLAP.md">method</a>).</p>
<p class="legend">
 <span class="chip" style="background:#e8f5e9">reference — declared non-overlapping target</span>
 <span class="chip" style="background:#fff8e1">overlap suspect — paired with a reference</span>
 <span class="chip" style="background:#ffebee">upper bound — no reference / sparse score</span>
</p>
<table id="lb"><thead><tr>
<th onclick="srt(this,0)">Trait</th><th onclick="srt(this,1)">Score</th>
<th onclick="srt(this,2)">Target GWAS</th><th onclick="srt(this,3)">Anc.</th>
<th onclick="srt(this,4,1)">R&sup2;</th><th onclick="srt(this,5)">Status</th>
<th onclick="srt(this,6,1)">overlap z</th><th onclick="srt(this,7,1)">R&sup2; corrected</th>
<th>reference (R&sup2;)</th></tr></thead><tbody>
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
