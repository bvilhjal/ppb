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

The same query also returns each score's accuracy in **all eight** UKBB ancestry
groups. That is the individual-level portability curve PPB's flagship aims to
recover from summary statistics alone, for these exact scores -- a ready-made
validation target for Phase 4 (`FINISHING_PLAN.md`).

Run (needs network; no local data):

    python scripts/anchor_validation.py
    python scripts/anchor_validation.py --out results/anchor-<date>.json

The ``--out`` snapshot records the fetch date inside the file, so Tables 3–4
of ``docs/REAL_DATA.md`` can rest on a committed, dated snapshot rather than a
live query (review 2026-08-16, F5).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


API = "https://www.pgscatalog.org/rest/performance/search"
EUROPEAN_GROUP = "UK (+ Ireland)"
PAGE_LIMIT = 50
# A hard cap so a server that ignores `offset` (and would return the same full
# page forever) fails loudly instead of looping.
MAX_PAGES = 40


def _fetch(url, opener=None):
    if opener is not None:
        return opener(url)
    # Default certificate verification. An earlier version pinned a
    # dev-machine CA bundle here, which made the documented one-liner fail
    # everywhere else; there is no custom bundle to trust.
    request = urllib.request.Request(
        url, headers={"User-Agent": "ppb-anchor-validation"})
    with urllib.request.urlopen(request, timeout=120) as fh:
        return json.load(fh)


def _performance_results(pgs_id, *, opener=None):
    """Every performance record for one score, following pagination.

    ``limit=50`` can truncate a score with more performance entries than that
    (each ancestry group of each evaluation is its own record), so pages are
    followed until a short one. The response shape is checked defensively:
    anything but an object with a ``results`` list is an error naming the
    score, never a silently empty result.
    """
    results = []
    for page in range(MAX_PAGES):
        payload = _fetch(
            f"{API}?pgs_id={pgs_id}&limit={PAGE_LIMIT}"
            f"&offset={page * PAGE_LIMIT}", opener=opener)
        if not isinstance(payload, dict) \
                or not isinstance(payload.get("results"), list):
            raise ValueError(
                f"{pgs_id}: unexpected PGS Catalog response shape (expected "
                "an object with a 'results' list)")
        results.extend(payload["results"])
        if len(payload["results"]) < PAGE_LIMIT:
            return results
    raise ValueError(
        f"{pgs_id}: more than {MAX_PAGES * PAGE_LIMIT} performance records; "
        "refusing to page further without revisiting the query")


def published_accuracy(pgs_id, *, opener=None):
    """``{ancestry group: (partial_r, n)}`` for one PGS Catalog score.

    The catalog stores a partial correlation, so partial R^2 is its square.
    """
    groups = {}
    for result in _performance_results(pgs_id, opener=opener):
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
        # The date stamp inside the snapshot is what makes it a snapshot: a
        # committed file whose numbers were fetched on a recorded day.
        snapshot = dict(
            date=time.strftime("%Y-%m-%d", time.gmtime()),
            api=API, registry=str(args.registry),
            rows=rows, portability=curves)
        Path(args.out).write_text(
            json.dumps(snapshot, indent=1) + "\n", encoding="utf-8")
        print(f"\n{len(rows)} comparisons ({snapshot['date']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
