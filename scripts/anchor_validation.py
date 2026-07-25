#!/usr/bin/env python
"""Check PPB's within-ancestry anchor against published individual-level accuracy.

`docs/REAL_DATA.md` justified its numbers as "consistent with the literature".
That is the weakest form of validation available, and it is unnecessary: the same
PGS Catalog scores PPB evaluates carry *published* individual-level performance
from their source publication (Prive et al. 2022, AJHG), retrievable from the PGS
Catalog REST API. This turns the hand-wave into a table with ratios.

**What is and is not being compared.** The published metric is a partial
correlation in a held-out UK Biobank ancestry group; the European anchor is the
"UK (+ Ireland)" group. PPB's honest number is measured against a *consortium*
target (GIANT, GLGC, ...). Two different EUR cohorts, two slightly different
estimands: a magnitude check, not an equality test. A ratio near 1 is strong
evidence the pipeline is sound; a ratio far from 1 localises where to look.

Binary traits are not comparable at all -- PPB's output there is a standardized
summary-statistic approximation whose scale depends on the supplied effective
sample size, not an observed-scale partial correlation (`docs/LIMITATIONS.md`).
They are reported for completeness and excluded from the verdict.

The same query also returns each score's accuracy in **all nine** UKBB ancestry
groups. That is the individual-level portability curve PPB's flagship aims to
recover from summary statistics alone, for these exact scores -- a ready-made
validation target for Phase 4 (`FINISHING_PLAN.md`).

Run (needs network; no local data):

    python scripts/anchor_validation.py
    python scripts/anchor_validation.py --out anchor.json
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regenerate_results import TRAITS  # noqa: E402

API = "https://www.pgscatalog.org/rest/performance/search"
EUROPEAN_GROUP = "UK (+ Ireland)"
CA_BUNDLE = "/root/.ccr/ca-bundle.crt"


def _context():
    try:
        return ssl.create_default_context(cafile=CA_BUNDLE)
    except (FileNotFoundError, ssl.SSLError):
        return ssl.create_default_context()


def published_accuracy(pgs_id, *, opener=None):
    """``{ancestry group: (partial_r, n)}`` for one PGS Catalog score.

    The catalog stores a partial correlation, so partial R^2 is its square.
    """
    url = f"{API}?pgs_id={pgs_id}&limit=50"
    if opener is None:
        request = urllib.request.Request(
            url, headers={"User-Agent": "ppb-anchor-validation"})
        with urllib.request.urlopen(request, timeout=120, context=_context()) as fh:
            payload = json.load(fh)
    else:
        payload = opener(url)

    groups = {}
    for result in payload.get("results", []):
        samples = result.get("sampleset", {}).get("samples", [])
        if not samples:
            continue
        sample = samples[0]
        label = sample.get("ancestry_additional")
        for metric in result.get("performance_metrics", {}).get("othermetrics", []):
            name = (metric.get("name_long", "") + metric.get("name_short", "")).lower()
            if "partial" in name and label:
                groups[label] = (float(metric["estimate"]), sample["sample_number"])
    return groups


def compare(registry_records, *, opener=None):
    """Join the registry's non-overlapping rows to published accuracy."""
    honest = {r["trait"]: r for r in registry_records
              if r["target"]["overlap"] == "none (declared)"}
    rows, curves = [], {}
    for trait, record in honest.items():
        pgs = record["score"]["id"]
        groups = published_accuracy(pgs, opener=opener)
        if EUROPEAN_GROUP not in groups:
            continue
        r_pub, n_pub = groups[EUROPEAN_GROUP]
        rows.append(dict(
            trait=trait, pgs=pgs, target=record["target"]["gwas"],
            trait_type=record["target"]["trait_type"],
            r2_ppb=record["metrics"]["r2"],
            r2_published=r_pub ** 2, partial_r_published=r_pub, n_published=n_pub,
            support=(record["metrics"]["n_variants_scored"]
                     / record["score"]["n_variants"]),
            ratio=r_pub ** 2 / record["metrics"]["r2"]))
        curves[trait] = {g: v[0] ** 2 for g, v in groups.items()}
    return rows, curves


def summarize(rows, curves):
    print(f"{'trait':>7} {'target GWAS':>30} {'PPB R2':>7} {'published':>10} "
          f"{'pub/PPB':>8} {'support':>8}")
    for row in sorted(rows, key=lambda r: -r["r2_published"]):
        print(f"{row['trait']:>7} {row['target'][:30]:>30} {row['r2_ppb']:7.3f} "
              f"{row['r2_published']:10.3f} {row['ratio']:8.2f} {row['support']:8.1%}"
              + ("" if row["trait_type"] == "quantitative" else "   (binary: not comparable)"))

    quantitative = [r for r in rows if r["trait_type"] == "quantitative"]
    print("\nquantitative rows are the interpretable ones:")
    for row in sorted(quantitative, key=lambda r: r["ratio"]):
        print(f"  {row['trait']:>7} {row['ratio']:5.2f}x   target: {row['target']}")

    if curves:
        groups = sorted({g for c in curves.values() for g in c},
                        key=lambda g: -sum(c.get(g, 0) / c[EUROPEAN_GROUP]
                                           for c in curves.values()))
        traits = list(curves)
        print("\npublished individual-level portability, relative to "
              f"{EUROPEAN_GROUP} -- the curve the flagship aims to recover:")
        print(f"  {'group':>22}" + "".join(f"{t:>9}" for t in traits))
        for group in groups:
            cells = "".join(
                f"{curves[t][group] / curves[t][EUROPEAN_GROUP]:9.2f}"
                if group in curves[t] else f"{'-':>9}" for t in traits)
            print(f"  {group:>22}{cells}")
    return quantitative


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", default=str(ROOT / "results" / "baseline-2026-07.json"))
    ap.add_argument("--out", default=None, help="write the comparison as JSON")
    args = ap.parse_args(argv)

    records = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    rows, curves = compare(records)
    if not rows:
        print("no non-overlapping registry rows with published accuracy")
        return 1
    summarize(rows, curves)
    if args.out:
        Path(args.out).write_text(
            json.dumps(dict(rows=rows, portability=curves), indent=1) + "\n",
            encoding="utf-8")
        print(f"\n{len(rows)} comparisons -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
