"""Fail when the committed ancestry-report PDF trails its sources.

The PDF cannot be rebuilt without a LaTeX toolchain, so this gate compares
git commit times instead of recompiling: if any source file is committed
newer than ``docs/ancestry_report/ancestry_report.pdf``, the evidence chain
has drifted and the check fails with the rebuild command.

Rebuild (needs pdflatex; bibliography is a self-contained
``thebibliography``, so no bibtex run)::

    cd docs/ancestry_report
    pdflatex ancestry_report.tex && pdflatex ancestry_report.tex
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "ancestry_report"
PDF = REPORT / "ancestry_report.pdf"
SOURCES = [
    REPORT / "ancestry_report.tex",
    REPORT / "make_figures.py",
    REPORT / "fig1_operating_characteristics.pdf",
    REPORT / "fig2_projection_weights.pdf",
    REPORT / "fig3_ld_diagnostics.pdf",
    *sorted((ROOT / "results" / "ancestry-frequency").glob("*.json")),
    *sorted((ROOT / "results" / "ancestry-ld").glob("*.json")),
    ROOT / "experiments" / "ancestry_ld_study.py",
]


def _commit_time(path: Path) -> int | None:
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", str(path)],
        cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout.strip()
    return int(out) if out else None


def _dirty(paths) -> list:
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--", *[str(p) for p in paths]],
        cwd=ROOT, capture_output=True, text=True)
    return [line for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    pdf_time = _commit_time(PDF)
    if pdf_time is None:
        print(f"FAIL: {PDF} is not tracked; commit a built PDF")
        return 1
    dirty = _dirty([REPORT, *SOURCES])
    if dirty:
        print("FAIL: report sources or PDF have uncommitted changes:")
        for line in dirty:
            print(f"  {line}")
        print("commit first; the time comparison below only sees commits")
        return 1
    stale = []
    for source in SOURCES:
        if not source.exists():
            print(f"FAIL: expected source {source} is missing")
            return 1
        source_time = _commit_time(source)
        if source_time is None:
            print(f"FAIL: {source} is not tracked; commit it")
            return 1
        if source_time > pdf_time:
            stale.append(source.relative_to(ROOT))
    if stale:
        print("FAIL: ancestry_report.pdf trails its sources:")
        for path in stale:
            print(f"  {path}")
        print("rebuild: cd docs/ancestry_report && "
              "pdflatex ancestry_report.tex && pdflatex ancestry_report.tex")
        return 1
    print("report PDF is current with its sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
