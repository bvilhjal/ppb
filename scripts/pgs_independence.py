#!/usr/bin/env python
"""Screen PGS Catalog scores for target-cohort independence.

PPB's (H2) requires that the evaluated score was trained independently of the
target GWAS's noise. The PGS Catalog REST API records each score's training
cohorts (``samples_training[].cohorts``), so the independence check is
programmatic: a score is evaluable against a FinnGen target iff no FinnGen
cohort entered its training, and against a UK Biobank-derived target iff no
UKB cohort did.

Run (stdlib only, needs network):

    python scripts/pgs_independence.py --out data/pgs_independence.tsv
    python scripts/pgs_independence.py --ids PGS002026 PGS000014

The TSV columns are: ``pgs_id``, ``name``, ``trait_efo``, ``n_dev``,
``ancestry_broad``, ``cohorts``, ``ukb_trained``, ``finngen_trained``,
``ukb_target_independent``, ``finngen_target_independent``. Cohort matching is
case-insensitive substring matching against the short names, which is
deliberately conservative: an unrecognized cohort name leaves the trained flag
false, so a score is only ever declared independent when no recognized
cohort of that biobank is present.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://www.pgscatalog.org/rest/score/all"
_UKB = ("ukb", "uk biobank")
_FINNGEN = ("finngen",)

# Biobanks whose scores are common in the catalog; anything not in these three
# sets is recorded verbatim in the cohorts column.
_ALIASES = {
    "23andMe": ("23andme",),
    "deCODE": ("decode",),
    "BBJ": ("bbj", "biobank japan"),
    "FinnGen": _FINNGEN,
    "UKB": _UKB,
}


def _fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def _cohorts_of(record):
    """Short cohort names of one score's training samples, lower-cased."""
    names = []
    for sample in record.get("samples_training") or []:
        for cohort in sample.get("cohorts") or []:
            short = (cohort.get("name_short") or "").strip()
            if short:
                names.append(short)
    return names


def _label(cohorts):
    """Recognized-biobank flags and a printable label per cohort set."""
    lowered = [c.lower() for c in cohorts]
    trained = {}
    for label, aliases in _ALIASES.items():
        trained[label] = any(any(a in c for a in aliases) for c in lowered)
    recognised = {c for c in cohorts if any(
        any(a in c.lower() for a in aliases) for aliases in _ALIASES.values())}
    unknown = [c for c in cohorts if c not in recognised]
    return trained, unknown


def _efo(record):
    efo = record.get("trait_efo") or []
    if isinstance(efo, list):
        ids = [e.get("id") if isinstance(e, dict) else str(e) for e in efo]
        return ",".join(i for i in ids if i)
    return str(efo)


def iter_all_scores():
    """Page through every score in the catalog (full records, not ids).

    The API rejects ``limit`` above 200 with HTTP 400, so the page size is
    capped there (~35 pages for the current catalog).
    """
    limit, offset, seen = 200, 0, 0
    while True:
        url = f"{API}?limit={limit}&offset={offset}"
        for attempt in range(3):
            try:
                page = _fetch_json(url)
                break
            except (urllib.error.URLError, json.JSONDecodeError):
                time.sleep(2.0 ** attempt)
        else:
            raise RuntimeError(f"could not fetch {url}")
        results = page.get("results") or []
        yield from results
        seen += len(results)
        offset += len(results)
        if len(results) < limit:
            break
        time.sleep(0.1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", nargs="*", default=None,
                    help="screen only these PGS ids (default: the whole catalog)")
    ap.add_argument("--out", default=None,
                    help="write TSV here (default: stdout)")
    args = ap.parse_args(argv)

    if args.ids:
        records = []
        for pgs in args.ids:
            url = f"https://www.pgscatalog.org/rest/score/{pgs}"
            records.append(_fetch_json(url))
    else:
        records = iter_all_scores()

    fieldnames = ("pgs_id", "name", "trait_efo", "n_dev", "ancestry_broad",
                  "cohorts", "ukb_trained", "finngen_trained",
                  "ukb_target_independent", "finngen_target_independent")
    out = open(args.out, "w", encoding="utf-8", newline="") if args.out \
        else sys.stdout
    writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t",
                            extrasaction="ignore")
    writer.writeheader()
    n_scores = n_finngen_ok = n_ukb_ok = 0
    by_trait = {}
    for record in records:
        cohorts = _cohorts_of(record)
        trained, unknown = _label(cohorts)
        efo = _efo(record)
        row = dict(
            pgs_id=record.get("id"), name=record.get("name") or "",
            trait_efo=efo, n_dev=sum(
                (s.get("sample_number") or 0)
                for s in (record.get("samples_training") or [])),
            ancestry_broad=",".join(sorted({
                s.get("ancestry_broad") or ""
                for s in (record.get("samples_training") or [])})),
            cohorts=",".join(cohorts + unknown),
            ukb_trained=trained["UKB"], finngen_trained=trained["FinnGen"],
            ukb_target_independent=not trained["UKB"],
            finngen_target_independent=not trained["FinnGen"],
        )
        writer.writerow(row)
        n_scores += 1
        n_finngen_ok += row["finngen_target_independent"]
        n_ukb_ok += row["ukb_target_independent"]
        for t in efo.split(",") or ["-"]:
            t = t.strip() or "-"
            by_trait.setdefault(t, [0, 0])
            by_trait[t][0] += 1
            by_trait[t][1] += row["finngen_target_independent"]
    if args.out:
        out.close()
        print(f"wrote {args.out}", file=sys.stderr)
    top = sorted(by_trait.items(), key=lambda kv: -kv[1][0])[:12]
    print(f"\n{n_scores} scores screened; {n_finngen_ok} FinnGen-independent, "
          f"{n_ukb_ok} UKB-independent", file=sys.stderr)
    print("largest traits (total / FinnGen-evaluable):", file=sys.stderr)
    for trait, (total, ok) in top:
        print(f"  {trait or '-':24s} {total:5d} / {ok}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
