#!/usr/bin/env python
"""Evaluate one score on a target-ancestry ``(z_B, D_B)`` pair (Phase 4).

This is the flagship run the method exists for. It is *not* a simulation.
Required local inputs, none of which this repository ships:

1. PGS weights (PGS Catalog dosage, or an LDpred3/BiPred/GWFM/MultiPGS file).
2. A target-ancestry GWAS table already converted to PPB's standardized ``z``
   (use ``ppb.standardized_marginal``, not ``ldpred3.standardize_betas``).
3. A target-ancestry LD reference: either ppb ``ldref_chr*.npz``, or an
   LDpred3 cache converted with ``scripts/ldpred3_cache_to_ppb.py``.
4. Optionally, published individual-level portability for the same score
   (``docs/REAL_DATA.md`` Table 4) to compare against.

If any input is missing the script exits 2 and prints what to collect.
It does not invent a cross-ancestry number.

    python scripts/cross_ancestry_eval.py \\
        --weights data/pgs_weights/PGS002146.txt \\
        --weight-scale dosage --hwe-genotype-sd \\
        --ldref-dir data/ldref_afr --sumstats data/height_afr_z.tsv \\
        --n-eff 80000 --out results/height_afr.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ppb.cli import _ldref_paths
from ppb.io import evaluate_ldrefs, read_sumstats, read_weight_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase-4 target-ancestry evaluation of a linear PGS.")
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--weight-scale", required=True,
        choices=("dosage", "standardized", "frozen"))
    parser.add_argument("--ldref-dir", required=True)
    parser.add_argument("--sumstats", required=True)
    parser.add_argument("--hwe-genotype-sd", action="store_true")
    parser.add_argument("--n-eff", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--keep-ambiguous", action="store_true")
    args = parser.parse_args(argv)

    missing = []
    for label, path in (
            ("weights", args.weights),
            ("sumstats", args.sumstats),
            ("ldref-dir", args.ldref_dir)):
        if not Path(path).exists():
            missing.append(f"{label}: {path}")
    if missing:
        print("Phase-4 inputs are not on disk:", file=sys.stderr)
        for line in missing:
            print(f"  {line}", file=sys.stderr)
        print(
            "Collect a target-ancestry GWAS + matched LD panel, convert the "
            "cache with scripts/ldpred3_cache_to_ppb.py, and rerun. "
            "See docs/CROSS_ANCESTRY.md exercise 7.",
            file=sys.stderr)
        return 2

    wf = read_weight_file(args.weights)
    needs_empirical_sd = (
        args.weight_scale in ("dosage", "frozen") and not args.hwe_genotype_sd)
    sumstats_variants, z, genotype_sd = read_sumstats(
        args.sumstats, read_genotype_sd=needs_empirical_sd)
    result = evaluate_ldrefs(
        _ldref_paths(args.ldref_dir),
        wf.variants, wf.weights,
        sumstats_variants, z,
        weight_scale=args.weight_scale,
        genotype_sd=(genotype_sd if needs_empirical_sd else None),
        sd_ref=wf.sd_ref,
        hwe_genotype_sd=args.hwe_genotype_sd,
        n_eff=args.n_eff,
        remove_ambiguous=not args.keep_ambiguous,
        mse_interpretable=args.weight_scale == "standardized",
    )
    text = json.dumps(result.to_dict(), indent=2, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}  (R^2 = {result.r2:.6f})")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
