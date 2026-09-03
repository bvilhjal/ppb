#!/usr/bin/env python
"""Prepare a FinnGen endpoint as a ppb ``beta/se`` summary-statistics table.

The machinery half of a FinnGen evaluation: given a FinnGen DF13 endpoint and a
PGS Catalog score, emit the ``chrom, pos, a1, a2, beta, se`` table that
``ppb evaluate --sumstats-scale beta-se-n`` consumes, plus the trait-level
``n_eff`` computed from the endpoint's case/control counts.

    python scripts/finngen_prep.py --endpoint T2D --pgs PGS002026 \
        --out data/finngen/T2D.tsv --out-json data/finngen/T2D.json

Joining and positions. FinnGen summary statistics are GRCh38; the shipped ppb
LD reference and PGS Catalog ``hmPOS`` weight positions are GRCh37. Rather than
lifting over, this script joins the two tables **on rsID** and takes the
GRCh37 positions from the weights file, so the output rows are already on the
reference build. The join is cross-checked on alleles: a matched row is kept
only when FinnGen's ref/alt pair equals the weights file's allele pair (in
either order), so a merged rsID pointing at a different variant is dropped
and counted as ``allele_mismatch`` rather than silently mis-joined. Rows
without an rsID, with duplicate rsIDs among the matched set, two rsIDs
collapsing onto one emitted position, or non-finite beta/SE are likewise
skipped and counted. The join is rsID-only across builds (position
cross-checks are impossible), so a small number of misaligned variants
remains possible; the counts in the sidecar record how much the join
retained, and the sidecar names its inputs and builds.

Allele orientation. ``a1`` is FinnGen's ``alt`` — the allele ``beta`` belongs
to — and ``a2`` is ``ref``: ppb's convention is that ``a1`` is the effect
allele of the value the row carries, so no global sign flip is needed
downstream.

``n_eff``. FinnGen GWAS files carry no per-variant ``n``. The endpoint's
case/control counts come from the release manifest, and
``n_eff = 4 / (1/n_cases + 1/n_controls)`` is the balanced effective size for
the (M4) conversion and the (X3) correction. The estimate stays on the
observed scale at a case fraction of one half: for binary endpoints it is a
standardized summary-statistic approximation, **not** liability-scale R^2.

Network. With ``--pgs`` the weights file is fetched from the PGS Catalog;
with ``--endpoint`` the GWAS file and manifest are fetched from FinnGen's
public Google Cloud bucket. Nothing is stored except the two output files.
Pass ``--weights``, ``--endpoint-file`` and/or ``--manifest`` to use local
copies instead (the tests do exactly this).

Access note: FinnGen summary statistics are served without credentials, but
FinnGen asks users to register an access form before downloading; do so before
using this script on their data.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import gzip
import io
import json
import math
import os
import sys
import urllib.request

MANIFEST_URL = ("https://storage.googleapis.com/finngen-public-data-r13/"
                "summary_stats/finngen_R13_manifest.tsv")
ENDPOINT_URL = ("https://storage.googleapis.com/finngen-public-data-r13/"
                "summary_stats/finngen_R13_{endpoint}.gz")
PGS_API = "https://www.pgscatalog.org/rest/score/{pgs}"

# Column aliases, resolved per file: FinnGen (GRCh38) and PGS Catalog
# harmonized scoring files (GRCh37 hmPOS columns preferred).
_FINNGEN_COLS = dict(chrom=("#chrom", "chrom"), pos=("pos",), ref=("ref",),
                     alt=("alt",), rsids=("rsids",), beta=("beta",),
                     se=("sebeta",))
_WEIGHT_COLS = dict(rsid=("rsID", "rsid"), chrom=("hm_chr", "chr_name"),
                    pos=("hm_pos", "chr_position"), ea=("effect_allele", "ea"),
                    oa=("other_allele", "oa"))


def _resolve(header, aliases, table):
    lut = {name.strip().lower().lstrip("#"): i
           for i, name in enumerate(header)}
    cols = {}
    for field, names in aliases.items():
        for name in names:
            if name.lower() in lut:
                cols[field] = lut[name.lower()]
                break
        else:
            raise ValueError(
                f"{table} is missing a '{field}' column (looked for {names}); "
                f"header was {header}")
    return cols


def _open_text(path_or_url):
    """Return a text-mode line iterator over a (possibly gzipped) file or URL.

    Reads are streamed -- a decompressed HTTP response is never written to
    disk and never materialised whole in memory. URLs fetched by this script
    always end in ``.gz``; local files are sniffed by their magic bytes so a
    misnamed local copy still parses.
    """
    path_or_url = os.fspath(path_or_url)
    if path_or_url.startswith(("http://", "https://")):
        fh = urllib.request.urlopen(path_or_url, timeout=600)
        if path_or_url.rstrip("/").endswith((".gz", ".bgz")):
            fh = gzip.GzipFile(fileobj=fh)
    else:
        fh = open(path_or_url, "rb")
        gzipped = fh.read(2).startswith(b"\x1f\x8b")
        fh.seek(0)
        if gzipped:
            fh = gzip.GzipFile(fileobj=fh)
    return io.TextIOWrapper(fh, encoding="utf-8", errors="replace")


def _header_and_rows(fh, table, *, strip_leading_comments):
    """``(header_fields, row_iterator)`` skipping empty and comment lines.

    With ``strip_leading_comments`` (the ppb table convention, weights files),
    leading ``#`` lines are skipped and the first non-comment line is the
    header. Without it (FinnGen files), the first non-empty line is the header
    even though it starts with ``#chrom``. Lines after the header are always
    treated as comments when ``#``-prefixed.
    """
    lines = (ln for ln in fh if ln.strip())
    header = next(lines, None)
    if header is None:
        raise ValueError(f"{table} has no header")
    if strip_leading_comments:
        while header.lstrip().startswith("#"):
            header = next(lines, None)
            if header is None:
                raise ValueError(f"{table} has no header")
    rows = (ln for ln in lines if not ln.lstrip().startswith("#"))
    return header.strip().split("\t"), rows


def read_manifest(manifest):
    """``phenocode -> (n_cases, n_controls)`` from the FinnGen release manifest."""
    out = {}
    with _open_text(manifest) as fh:
        for i, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            out[row["phenocode"].strip()] = (
                int(row["num_cases"]), int(row["num_controls"]))
    if not out:
        raise ValueError(f"manifest {manifest!r} parsed no endpoints")
    return out


def read_weights(weights_source):
    """``(rsid -> match, counts)`` from a weights file.

    ``match`` is ``(chrom, pos, effect_allele, other_allele)``. GRCh37
    positions: ``hm_chr``/``hm_pos`` are preferred over ``chr_name``/
    ``chr_position`` so the output table is on the reference build. Duplicate
    rsIDs keep their first row; every skipped-row reason is counted so the
    reconciliation closes.
    """
    out, duplicates, no_rsid, short_rows = {}, 0, 0, 0
    with _open_text(weights_source) as fh:
        header, rows = _header_and_rows(fh, "weights file",
                                        strip_leading_comments=True)
        cols = _resolve(header, _WEIGHT_COLS, "weights file")
        reader = csv.reader(rows, delimiter="\t")
        for r in reader:
            if len(r) <= max(cols.values()):
                short_rows += 1
                continue
            rsid = r[cols["rsid"]].strip()
            if not rsid:
                no_rsid += 1
                continue
            if rsid in out:
                duplicates += 1
                continue
            out[rsid] = (r[cols["chrom"]].strip(), int(r[cols["pos"]]),
                         r[cols["ea"]].strip().upper(),
                         r[cols["oa"]].strip().upper())
    counts = dict(weights_rows=len(out) + duplicates + no_rsid + short_rows,
                  weights_parsed=len(out), weights_duplicate_rsids=duplicates,
                  weights_no_rsid=no_rsid, weights_short_rows=short_rows)
    return out, counts


def n_eff(n_cases, n_controls):
    """Balanced effective sample size ``4/(1/n_cases + 1/n_controls)``."""
    if n_cases <= 0 or n_controls <= 0:
        raise ValueError(
            f"n_eff needs positive case/control counts, got {n_cases}/{n_controls}")
    return 4.0 / (1.0 / n_cases + 1.0 / n_controls)


def fetch_weights_url(pgs):
    """URL of the ``hmPOS_GRCh37`` harmonized scoring file for a PGS id."""
    with urllib.request.urlopen(PGS_API.format(pgs=pgs), timeout=60) as resp:
        record = json.load(resp)
    for f in record.get("ftp_harmonized_scoring_files") or []:
        path = f.get("ftp_path") or ""
        if "hmPOS_GRCh37" in path:
            return path
    raise ValueError(
        f"{pgs} has no hmPOS_GRCh37 harmonized scoring file; candidates were "
        f"{[f.get('ftp_path') for f in record.get('ftp_harmonized_scoring_files') or []]}")


def prep(endpoint, weights_source, manifest, out_path, *, out_json=None,
         endpoint_source=None):
    """Join a FinnGen endpoint to a weights table and write the ppb TSV.

    Returns the counts dict (also written to ``out_json`` when given).
    """
    if endpoint_source is None:
        endpoint_source = ENDPOINT_URL.format(endpoint=endpoint)
    manifest_map = read_manifest(manifest)
    if endpoint not in manifest_map:
        closest = difflib.get_close_matches(endpoint, manifest_map, n=5,
                                            cutoff=0.6) or sorted(
            m for m in manifest_map
            if m.startswith(endpoint.split("_")[0]))[:5]
        raise ValueError(
            f"endpoint {endpoint!r} not in the manifest; closest matches: "
            f"{closest}")
    n_cases, n_controls = manifest_map[endpoint]
    neff = n_eff(n_cases, n_controls)
    weights, w_counts = read_weights(weights_source)

    counts = dict(n_cases=n_cases, n_controls=n_controls, n_eff=round(neff, 1),
                  endpoint_rows=0, short_rows=0, matched=0, no_rsid=0,
                  not_in_weights=0, bad_se=0, duplicate_rsid=0,
                  allele_mismatch=0, duplicate_position=0, **w_counts)
    seen = set()
    emitted = set()
    with _open_text(endpoint_source) as fh, open(
            out_path, "w", encoding="utf-8", newline="") as out_fh:
        header, rows = _header_and_rows(fh, "endpoint file",
                                        strip_leading_comments=False)
        cols = _resolve(header, _FINNGEN_COLS, "endpoint file")
        writer = csv.writer(out_fh, delimiter="\t")
        writer.writerow(("chrom", "pos", "a1", "a2", "beta", "se"))
        reader = csv.reader(rows, delimiter="\t")
        for r in reader:
            counts["endpoint_rows"] += 1
            if len(r) <= max(cols.values()):
                counts["short_rows"] += 1
                continue
            rsid = r[cols["rsids"]].strip().replace(";", ",").split(",")[0].strip()
            if not rsid:
                counts["no_rsid"] += 1
                continue
            if rsid in seen:
                counts["duplicate_rsid"] += 1
                continue
            match = weights.get(rsid)
            if match is None:
                counts["not_in_weights"] += 1
                continue
            beta, se = r[cols["beta"]].strip(), r[cols["se"]].strip()
            try:
                b, s = float(beta), float(se)
            except ValueError:
                counts["bad_se"] += 1
                continue
            if not math.isfinite(b) or not math.isfinite(s) or s <= 0.0:
                counts["bad_se"] += 1
                continue
            chrom, pos, ea, oa = match
            ref = r[cols["ref"]].strip().upper()
            alt = r[cols["alt"]].strip().upper()
            if {ref, alt} != {ea, oa}:
                # The rsID join landed on a different variant (merged rsID or
                # a cross-build collision): the weights-file position would
                # describe another variant, so drop and count.
                counts["allele_mismatch"] += 1
                seen.add(rsid)
                continue
            if (chrom, pos) in emitted:
                counts["duplicate_position"] += 1
                seen.add(rsid)
                continue
            seen.add(rsid)
            emitted.add((chrom, pos))
            counts["matched"] += 1
            # a1 is FinnGen's alt -- the allele beta belongs to; a2 is ref.
            # ppb's convention is that a1 is the effect allele of the value
            # the row carries, so no downstream sign flip is required.
            writer.writerow((chrom, pos, alt, ref, beta, se))
    if out_json is not None:
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(dict(
                endpoint=endpoint,
                weights_source=str(weights_source),
                endpoint_source=str(endpoint_source),
                manifest=str(manifest),
                genome_build=("output positions are GRCh37 (taken from the "
                              "weights file's hm_pos); the join is rsID-only "
                              "across builds (FinnGen itself is GRCh38)"),
                **counts), fh, indent=1)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", required=True,
                    help="FinnGen R13 endpoint code (e.g. T2D, I9_CHD)")
    ap.add_argument("--pgs", default=None,
                    help="PGS Catalog id whose hmPOS_GRCh37 scoring file "
                         "supplies rsids and GRCh37 positions")
    ap.add_argument("--weights", default=None,
                    help="local weights file (instead of --pgs)")
    ap.add_argument("--manifest", default=MANIFEST_URL,
                    help=f"manifest source (default: {MANIFEST_URL})")
    ap.add_argument("--endpoint-file", default=None,
                    help="local endpoint GWAS file (instead of fetching)")
    ap.add_argument("--out", required=True, help="output ppb sumstats TSV")
    ap.add_argument("--out-json", default=None,
                    help="sidecar JSON with n_eff and join counts")
    args = ap.parse_args(argv)

    if (args.pgs is None) == (args.weights is None):
        ap.error("provide exactly one of --pgs or --weights")
    weights_source = (fetch_weights_url(args.pgs) if args.pgs
                      else args.weights)
    counts = prep(args.endpoint, weights_source, args.manifest, args.out,
                  out_json=args.out_json, endpoint_source=args.endpoint_file)
    print(f"wrote {args.out}: {counts['matched']} variants joined "
          f"({counts['matched'] / max(counts['weights_parsed'], 1):.1%} of "
          f"weights); n_eff = {counts['n_eff']} "
          f"({counts['n_cases']} cases / {counts['n_controls']} controls)",
          file=sys.stderr)
    print(f"\nppb evaluate --weights <score weights file> --ldref-dir <ldref> \\\n"
          f"  --sumstats {args.out} --sumstats-scale beta-se-n "
          f"--n-eff {counts['n_eff']} \\\n"
          f"  --weight-scale dosage --hwe-genotype-sd",
          file=sys.stderr)
    print("PGS Catalog per-allele weights are dosage weights, so the scale is\n"
          "'dosage'; --hwe-genotype-sd supplies the per-variant SD from the LD\n"
          "reference's af_UKBB, which is an approximation for a Finnish target\n"
          "(same EUR-for-Finnish approximation as using the EUR LD reference).",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
