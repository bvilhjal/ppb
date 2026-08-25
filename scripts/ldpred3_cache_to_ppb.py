#!/usr/bin/env python
"""Convert an LDpred3 LD cache into ppb per-chromosome ``ldref_chr*.npz`` files.

The cache must carry chrom, pos, counted_allele, other_allele (the provenance
ldpred3 writes when the pipeline saves a cache). Memory-mapped caches and
low-rank (LR8/LR32) blocks are refused; shrink-toward-identity caches are
refused because they inflate R². See ``ppb.ldpred3_cache.convert_ldpred3_cache``.

Usage:
    python scripts/ldpred3_cache_to_ppb.py CACHE.npz OUT_DIR
    python scripts/ldpred3_cache_to_ppb.py CACHE.npz OUT_DIR --allow-lr-expand
"""

from __future__ import annotations

import argparse
import sys

from ppb.ldpred3_cache import convert_ldpred3_cache


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert an LDpred3 LD cache to ppb chromosome shards.")
    parser.add_argument("cache", help="LDpred3 save_ld_blocks .npz")
    parser.add_argument("out_dir", help="directory for ldref_chr*.npz")
    parser.add_argument(
        "--square", action="store_true",
        help="write full-square D8 (default: packed upper triangle)")
    parser.add_argument(
        "--allow-lr-expand", action="store_true",
        help="materialise LR8/LR32 blocks and requantise (approximation)")
    parser.add_argument(
        "--allow-shrunk", action="store_true",
        help="accept a cache that already shrank D toward I (inflates R²)")
    parser.add_argument(
        "--compress", action="store_true", help="np.savez_compressed")
    parser.add_argument(
        "--no-psd-scan", action="store_true",
        help="skip the Lanczos indefiniteness scan on large blocks (faster)")
    args = parser.parse_args(argv)
    report = convert_ldpred3_cache(
        args.cache, args.out_dir, packed=not args.square,
        allow_lr_expand=args.allow_lr_expand, allow_shrunk=args.allow_shrunk,
        compress=args.compress, psd_scan=not args.no_psd_scan)
    print(f"wrote {len(report.paths)} chromosome file(s) "
          f"({report.n_variants} variants, {report.n_blocks} blocks; "
          f"d8={report.n_d8} float={report.n_float_quantised} "
          f"lr_expanded={report.n_lr_expanded})")
    for path in report.paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
