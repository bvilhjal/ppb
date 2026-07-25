#!/usr/bin/env python
"""Negative controls for the real-data evaluation path.

Every fail-closed guard in ``ppb`` is unit-tested, but nothing checks that the
*real* pipeline returns ~0 when it should. A gauge error, a harmonization slip,
or a mis-scaled reference produces a plausible-looking number, not an exception.
Two controls, both from one extra sweep:

**Trait swap.** Evaluate every score against every consortium target, not just
its own. The diagonal should dominate its column and row. Off-diagonal cells are
*not* all expected to be zero -- BMI and T2D, or LDL and CAD, are genuinely
genetically correlated, and seeing that is itself a check that the pipeline
responds to real signal. What must not appear is a large off-diagonal between
traits with no plausible shared aetiology (height x LDL), which would indicate a
systematic artifact rather than biology.

**Block sign flip.** For each cell, ``ppb.sign_flip_null`` compares the estimate
against the accuracy the same score would report from block noise alone at its
own per-block magnitudes. This is an exact null (``D`` is block-diagonal, so
negating a whole block's weights leaves the denominator unchanged) and needs no
permuted phenotype. It calibrates small values: an R^2 of 0.025 means something
different at ``z = 30`` than at ``z = 1.5``.

Neither control detects a uniformly mis-scaled ``z`` -- observed and null move
together. See ``docs/LIMITATIONS.md``.

Run (needs the LD reference and the consortium targets under ``data/``):

    python scripts/negative_controls.py                     # every score x every target
    python scripts/negative_controls.py --out controls.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Run as `python scripts/negative_controls.py` and sys.path[0] is scripts/, not
# the repository root, so the sibling module below is not importable as
# `scripts.regenerate_results` without this. (`ppb` itself is pip-installed.)
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ppb import harmonize_to, read_ldref, read_weights, sign_flip_null  # noqa: E402
from scripts.regenerate_results import (  # noqa: E402
    CHROMS,
    DATA,
    LD_REF,
    TRAITS,
    _trait_type,
    git_commit,
    load_target,
)

# Trait pairs with a well-established shared genetic basis. Their off-diagonal
# cells are expected to be non-zero; listing them up front keeps the check from
# being quietly relaxed after the fact to accommodate whatever came out.
EXPECTED_CORRELATED = frozenset(
    frozenset(pair) for pair in (
        ("BMI", "T2D"), ("LDL", "CAD"), ("BMI", "CAD"), ("T2D", "CAD"),
    )
)


def paired_traits():
    """Traits that have a non-overlapping consortium target."""
    return [name for name, cfg in TRAITS.items() if "consortium" in cfg]


def sweep_matrix(traits, progress=True):
    """Evaluate every score against every target in one pass over the reference.

    Returns ``per_block[(score, target)] = (u, v)`` and ``chrom``, the
    chromosome tag per block. Each cell uses its own joint weight/summary-
    statistic support, exactly as ``regenerate_results.sweep`` does for the
    headline numbers -- a missing target association is never treated as zero.
    """
    scores = {}
    for trait in traits:
        variants, weights = read_weights(
            DATA / "pgs_weights" / f"{TRAITS[trait]['pgs']}_hmPOS_GRCh37.txt")
        scores[trait] = (variants, weights)

    targets = {}
    for trait in traits:
        path = DATA / "consortium" / f"{TRAITS[trait]['consortium']}_hm3plus.tsv"
        targets[trait] = load_target(path, trait_type=_trait_type(trait))

    cells = [(s, t) for s in traits for t in traits]
    u_b = {cell: [] for cell in cells}
    v_b = {cell: [] for cell in cells}
    chrom_tag = []

    for c in CHROMS:
        t0 = time.time()
        ref = read_ldref(DATA / "ldref_hm3_plus" / "npz" / f"ldref_chr{c}.npz")
        if "af" not in ref:
            raise ValueError(f"chr{c} LD reference carries no 'af' annotation")
        af = np.asarray(ref["af"], dtype=np.float64)
        bad = ~np.isfinite(af) | (af <= 0.0) | (af >= 1.0)
        if bad.any():
            raise ValueError(
                f"chr{c} LD reference has {int(bad.sum())} variant(s) with af "
                "outside (0, 1)")
        sd = np.sqrt(2.0 * af * (1.0 - af))

        aligned_w = {}
        for trait, (variants, weights) in scores.items():
            w_al, _, w_mask = harmonize_to(
                ref["variants"], variants, weights, return_mask=True)
            aligned_w[trait] = (w_al * sd, w_mask)
        aligned_z = {}
        for trait, (z_var, z, _) in targets.items():
            z_al, _, z_mask = harmonize_to(
                ref["variants"], z_var, z, return_mask=True)
            aligned_z[trait] = (z_al, z_mask)

        joint = {}
        for score, target in cells:
            ws, w_mask = aligned_w[score]
            z_al, z_mask = aligned_z[target]
            w_joint = ws.copy()
            w_joint[~(w_mask & z_mask)] = 0.0
            joint[(score, target)] = (w_joint, z_al)

        for backend, idx in ref["ld"].blocks:
            chrom_tag.append(c)
            for cell, (w_joint, z_al) in joint.items():
                u_b[cell].append(float(w_joint[idx] @ z_al[idx]))
                v_b[cell].append(backend.quad(w_joint[idx]))
        if progress:
            print(f"    chr{c:<2} {len(ref['ld'].blocks):>4} blocks  "
                  f"{time.time() - t0:6.1f}s", flush=True)
        del ref

    per_block = {cell: (np.asarray(u_b[cell], dtype=float),
                        np.asarray(v_b[cell], dtype=float))
                 for cell in cells}
    return per_block, np.asarray(chrom_tag)


def build_matrix(per_block, traits, n_draws=0, rng=None):
    """Turn per-block products into the control matrix.

    Each cell records the estimate, the sign-flip null it should be read
    against, and whether an off-diagonal cell was expected to be non-zero.
    """
    cells = []
    for score in traits:
        for target in traits:
            u, v = per_block[(score, target)]
            control = sign_flip_null(u, v, n_draws=n_draws, rng=rng)
            diagonal = score == target
            cells.append(dict(
                score=score, target=target, diagonal=diagonal,
                expected_correlated=bool(
                    not diagonal
                    and frozenset((score, target)) in EXPECTED_CORRELATED),
                **control.to_dict()))
    return cells


def summarize(cells, traits):
    """Print the matrix and the checks that should hold."""
    lookup = {(c["score"], c["target"]): c for c in cells}

    print("\nR^2 (rows = score, cols = target GWAS)")
    print(f"  {'':>8}" + "".join(f"{t:>10}" for t in traits))
    for score in traits:
        row = "".join(f"{lookup[(score, t)]['r2']:10.4f}" for t in traits)
        print(f"  {score:>8}{row}")

    print("\nsign-flip z (how coherently the LD blocks agree)")
    print(f"  {'':>8}" + "".join(f"{t:>10}" for t in traits))
    for score in traits:
        row = "".join(f"{lookup[(score, t)]['z']:10.1f}" for t in traits)
        print(f"  {score:>8}{row}")

    print("\nchecks")
    problems = []
    for score in traits:
        diag = lookup[(score, score)]
        worst = max((lookup[(score, t)] for t in traits if t != score),
                    key=lambda c: c["r2"])
        ok = diag["r2"] > worst["r2"]
        print(f"  {score:>8}: matched R^2 {diag['r2']:.4f} (z {diag['z']:6.1f}) vs "
              f"best mismatched {worst['r2']:.4f} on {worst['target']} "
              f"(z {worst['z']:6.1f})  {'ok' if ok else 'FAIL'}")
        if not ok:
            problems.append(f"{score}: mismatched target {worst['target']} scores higher")

    unexpected = [c for c in cells
                  if not c["diagonal"] and not c["expected_correlated"]
                  and abs(c["z"]) > 10.0]
    if unexpected:
        print("\n  unexpected off-diagonal signal (|z| > 10, no declared "
              "genetic correlation):")
        for c in sorted(unexpected, key=lambda c: -abs(c["z"])):
            print(f"    {c['score']:>8} score x {c['target']:>8} target: "
                  f"R^2 {c['r2']:.4f}, z {c['z']:.1f}")
            problems.append(f"{c['score']} x {c['target']}: |z| = {abs(c['z']):.1f}")
    else:
        print("\n  no unexpected off-diagonal signal")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("traits", nargs="*", default=None,
                    help="traits to include (default: all with a consortium target)")
    ap.add_argument("--out", default=None, help="write the JSON matrix here")
    ap.add_argument("--n-draws", type=int, default=0,
                    help="sign-flip permutation draws for an empirical p-value")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    available = paired_traits()
    traits = args.traits or available
    unknown = [t for t in traits if t not in available]
    if unknown:
        ap.error(f"no consortium target for: {', '.join(unknown)}")
    if len(traits) < 2:
        ap.error("the trait-swap control needs at least 2 traits")

    print(f"negative controls over {len(traits)} traits "
          f"({len(traits) ** 2} evaluations)", flush=True)
    per_block, _ = sweep_matrix(traits)
    cells = build_matrix(per_block, traits, n_draws=args.n_draws,
                         rng=np.random.default_rng(args.seed))
    problems = summarize(cells, traits)

    if args.out:
        Path(args.out).write_text(json.dumps(dict(
            ld_ref=LD_REF, traits=list(traits), cells=cells,
            date=time.strftime("%Y-%m-%d", time.gmtime()),
            ppb_commit=git_commit()), indent=1) + "\n", encoding="utf-8")
        print(f"\n{len(cells)} cells -> {args.out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
