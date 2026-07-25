"""Command-line interface: ``ppb evaluate``.

Evaluate a PGS weights file against a benchmark bundle and emit a JSON
:class:`~ppb.evaluate.EvaluationResult`.

    ppb evaluate --weights weights.tsv --bundle benchmark.npz [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import sys

from .evaluate import evaluate
from .io import read_bundle, read_weights


def _cmd_evaluate(args) -> int:
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
        # R^2 is invariant to a global rescale of w, MSE is not. Ordinary PGS
        # Catalog weights are in trait units (e.g. cm), so their MSE is not on
        # the bundle's standardized-phenotype scale and means nothing; only a
        # submission that declares standardized weights carries the scale MSE
        # needs. Report which case this run is rather than emitting a number
        # that looks equally authoritative either way.
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
        "evaluate", help="evaluate PGS weights against a benchmark bundle")
    ev.add_argument("--weights", required=True, help="PGS weights file (TSV/CSV)")
    ev.add_argument("--bundle", required=True, help="benchmark bundle (.npz)")
    ev.add_argument(
        "--weight-scale", required=True, choices=("dosage", "standardized"),
        help=("scale of the submitted weights: ordinary per-dosage weights "
              "need a bundle carrying target genotype_sd; standardized weights "
              "already multiply the standardized genotypes represented by LD"))
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
