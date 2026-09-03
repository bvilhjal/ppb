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
``ancestry_broad``, ``cohorts``, ``n_cohorts``, ``screen_status``,
``matched_evidence``, ``ukb_trained``, ``finngen_trained``,
``ukb_target_independent``, ``finngen_target_independent``.

The screen fails closed: ``*_target_independent`` is true only when
``screen_status`` is ``independent``.

- ``independent`` — cohort annotation is present, every cohort was
  recognized, and neither biobank appears among them;
- ``contaminated`` — a recognized cohort of one of the two biobanks is
  present;
- ``no_annotation`` — the catalog record carries no usable cohort
  annotation at all; absence of evidence is not evidence of independence;
- ``unrecognized_cohorts`` — some cohort could not be classified, so
  independence cannot be certified either way.

Matching is case-insensitive substring matching against both the short and
full cohort names (``name_short`` and ``name_full``), deliberately
conservative in both directions: an unrecognized name fails the screen, and a
recognized biobank alias (including Finnish cohorts such as FINRISK or NFBC,
which feed FinnGen) contaminates it. ``matched_evidence`` records which
cohort matched which biobank label so a human can audit every verdict.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://www.pgscatalog.org/rest/score/all"
_UKB = ("ukb", "uk biobank", "united kingdom biobank")
_FINNGEN = ("finngen", "finrisk", "nfbc")

# Cohort labels the screen recognizes; anything not matching one of these
# sets is recorded verbatim in the cohorts column and fails the screen.
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
    """``(name_short, name_full)`` pairs of one score's training cohorts.

    The short name is preferred; the full name is kept so matching can use
    either (a record may carry only ``name_full``, e.g. "United Kingdom
    Biobank"). Pairs with both names empty are dropped.
    """
    pairs = []
    for sample in record.get("samples_training") or []:
        for cohort in sample.get("cohorts") or []:
            short = (cohort.get("name_short") or "").strip()
            full = (cohort.get("name_full") or "").strip()
            if short or full:
                pairs.append((short, full))
    return pairs


def _label(pairs):
    """Classify a cohort list against the recognized biobank labels.

    Returns ``(trained, unknown, evidence)``: ``trained`` maps each label to
    whether any cohort matched it, ``unknown`` lists cohort display names no
    label matched, and ``evidence`` records ``label:cohort`` strings for
    every match so the TSV shows which alias fired.
    """
    trained = {label: False for label in _ALIASES}
    unknown, evidence = [], []
    for short, full in pairs:
        display = short or full
        fields = [f for f in (short, full) if f]
        hits = [label for label, aliases in _ALIASES.items()
                if any(alias in f.lower()
                       for alias in aliases for f in fields)]
        for label in hits:
            trained[label] = True
        if hits:
            evidence.append(f"{'+'.join(sorted(hits))}:{display}")
        else:
            unknown.append(display)
    return trained, unknown, evidence


def _screen_status(pairs, trained, unknown):
    """Fail-closed status: only full, recognized, clean annotation passes."""
    if not pairs:
        return "no_annotation"
    if trained["FinnGen"] or trained["UKB"]:
        return "contaminated"
    if unknown:
        return "unrecognized_cohorts"
    return "independent"


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
    limit, offset = 200, 0
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
                  "cohorts", "n_cohorts", "screen_status", "matched_evidence",
                  "ukb_trained", "finngen_trained",
                  "ukb_target_independent", "finngen_target_independent")
    out_path = args.out
    tmp = None
    if out_path:
        # Write to a sibling temp file and rename into place, so a mid-run
        # API failure cannot leave a plausible-looking partial TSV.
        tmp = f"{out_path}.part.{os.getpid()}"
        out = open(tmp, "w", encoding="utf-8", newline="")
    else:
        out = sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        n_scores = n_finngen_ok = n_ukb_ok = 0
        by_status, by_trait = {}, {}
        for record in records:
            pairs = _cohorts_of(record)
            trained, unknown, evidence = _label(pairs)
            status = _screen_status(pairs, trained, unknown)
            cohorts = [short or full for short, full in pairs]
            efo = _efo(record)
            row = dict(
                pgs_id=record.get("id"), name=record.get("name") or "",
                trait_efo=efo, n_dev=sum(
                    (s.get("sample_number") or 0)
                    for s in (record.get("samples_training") or [])),
                ancestry_broad=",".join(sorted({
                    s.get("ancestry_broad") or ""
                    for s in (record.get("samples_training") or [])})),
                cohorts=",".join(cohorts), n_cohorts=len(cohorts),
                screen_status=status,
                matched_evidence=";".join(evidence) or "-",
                ukb_trained=trained["UKB"], finngen_trained=trained["FinnGen"],
                ukb_target_independent=status == "independent",
                finngen_target_independent=status == "independent",
            )
            writer.writerow(row)
            n_scores += 1
            by_status[status] = by_status.get(status, 0) + 1
            n_finngen_ok += row["finngen_target_independent"]
            n_ukb_ok += row["ukb_target_independent"]
            for t in efo.split(","):
                t = t.strip() or "-"
                by_trait.setdefault(t, [0, 0])
                by_trait[t][0] += 1
                by_trait[t][1] += row["finngen_target_independent"]
    except BaseException:
        if tmp:
            out.close()
            os.unlink(tmp)
        raise
    if tmp:
        out.close()
        os.replace(tmp, out_path)
        print(f"wrote {out_path}", file=sys.stderr)
    top = sorted(by_trait.items(), key=lambda kv: -kv[1][0])[:12]
    print(f"\n{n_scores} scores screened; {n_finngen_ok} FinnGen-independent "
          f"and {n_ukb_ok} UKB-independent with evidence (screen_status "
          f"'independent')", file=sys.stderr)
    print("screen status counts: " + ", ".join(
        f"{k}={v}" for k, v in sorted(by_status.items())), file=sys.stderr)
    print("largest traits (total / FinnGen-evaluable):", file=sys.stderr)
    for trait, (total, ok) in top:
        print(f"  {trait or '-':24s} {total:5d} / {ok}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
