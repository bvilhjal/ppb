"""Command-line interface: ``ppb evaluate``.

Evaluate a PGS weights file against either a compact benchmark bundle or
chromosome-sharded block-int8 LD references, and emit a JSON
:class:`~ppb.evaluate.EvaluationResult`.

    ppb evaluate --weights weights.tsv --bundle benchmark.npz [--out result.json]
    ppb evaluate --weights weights.tsv --ldref-dir ldref --sumstats z.tsv \
        [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import evaluate
from .io import evaluate_ldrefs, read_bundle, read_sumstats, read_weights


def _ldref_paths(directory):
    """Return naturally ordered ``ldref_chr*.npz`` files."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"LD-reference directory {str(directory)!r} does not exist")
    paths = list(directory.glob("ldref_chr*.npz"))
    if not paths:
        raise ValueError(
            f"LD-reference directory {str(directory)!r} has no "
            "ldref_chr*.npz files")

    def key(path):
        label = path.stem.removeprefix("ldref_chr").upper()
        if label.isdigit():
            return 0, int(label)
        return 1, label

    return sorted(paths, key=key)


def _cmd_evaluate(args) -> int:
    if args.bundle:
        if args.sumstats or args.hwe_genotype_sd:
            raise ValueError(
                "--sumstats and --hwe-genotype-sd apply only with --ldref-dir")
        if args.var_y != 1.0:
            raise ValueError(
                "--var-y applies only with --ldref-dir; bundles carry var_y")
        weights_variants, weights = read_weights(args.weights)
        bundle = read_bundle(args.bundle)
        result = evaluate(
            bundle["ld"], bundle["variants"],
            weights_variants, weights,
            bundle["variants"], bundle["z"],
            var_y=bundle["var_y"],
            weight_scale=args.weight_scale,
            genotype_sd=bundle["genotype_sd"],
            remove_ambiguous=not args.keep_ambiguous,
            # R^2 is invariant to a global rescale of w, MSE is not. Ordinary
            # PGS Catalog weights are in trait units (e.g. cm), so their MSE
            # is not on the standardized-phenotype scale and means nothing.
            mse_interpretable=args.weight_scale == "standardized",
        )
    else:
        if not args.sumstats:
            raise ValueError("--sumstats is required with --ldref-dir")
        ldref_paths = _ldref_paths(args.ldref_dir)
        weights_variants, weights = read_weights(args.weights)
        use_empirical_sd = (
            args.weight_scale == "dosage" and not args.hwe_genotype_sd
        )
        sumstats_variants, z, genotype_sd = read_sumstats(
            args.sumstats, read_genotype_sd=use_empirical_sd)
        result = evaluate_ldrefs(
            ldref_paths,
            weights_variants, weights,
            sumstats_variants, z,
            var_y=args.var_y,
            weight_scale=args.weight_scale,
            genotype_sd=(genotype_sd
                         if (args.weight_scale == "dosage"
                             and not args.hwe_genotype_sd) else None),
            hwe_genotype_sd=args.hwe_genotype_sd,
            remove_ambiguous=not args.keep_ambiguous,
            mse_interpretable=args.weight_scale == "standardized",
        )
    text = json.dumps(result.to_dict(), indent=2, allow_nan=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"wrote result to {args.out}  (R^2 = {result.r2:.6f}, "
              f"{result.n_variants_scored}/{result.n_reference} variants scored)")
    else:
        print(text)
    if not result.mse_interpretable:
        print("note: MSE is not interpretable for dosage-scale weights (the "
              "weight scale is arbitrary); R^2 is unaffected.", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ppb",
        description="Summary-statistics-based cross-ancestry polygenic-score portability benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser(
        "evaluate", help="evaluate PGS weights against a bundle or LDref shards")
    ev.add_argument("--weights", required=True, help="PGS weights file (TSV/CSV)")
    source = ev.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", help="benchmark bundle (.npz)")
    source.add_argument(
        "--ldref-dir",
        help="directory containing chromosome files named ldref_chr*.npz")
    ev.add_argument(
        "--sumstats",
        help=("standardized-z table for --ldref-dir: chrom, pos, a1, a2, z; "
              "optionally empirical genotype_sd"))
    ev.add_argument(
        "--weight-scale", required=True, choices=("dosage", "standardized"),
        help=("scale of the submitted weights: ordinary per-dosage weights "
              "need empirical genotype_sd or the explicit HWE approximation; "
              "standardized weights already multiply standardized genotypes"))
    ev.add_argument(
        "--hwe-genotype-sd", action="store_true",
        help=("with --ldref-dir and dosage weights, use sqrt(2*af*(1-af)) "
              "from each LD reference instead of empirical target genotype_sd"))
    ev.add_argument(
        "--var-y", type=float, default=1.0,
        help="phenotype variance for --ldref-dir (default: 1; bundles carry it)")
    ev.add_argument("--out", default=None, help="write JSON result here (default: stdout)")
    ev.add_argument("--keep-ambiguous", action="store_true",
                    help="keep strand-ambiguous palindromic SNPs (dropped by default)")
    ev.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
