"""Can LD-score regression detect a mis-scaled target `z`?

`R^2 = (w^T z)^2 / (w^T D w)` has `z` in the numerator alone, so a target
deflated by genomic control reports a proportionally deflated R^2 and nothing in
the estimator notices. Against published individual-level accuracy, both GIANT
targets come in 1.5-2x low while a GLGC target agrees to 2% -- the size and
pattern of genomic control, and PPB's largest measured error.

(C2) supplies the missing diagnostic from data PPB already has. Three questions:

1. **Consistency.** Does the estimator recover a known intercept, and how many
   variants does that take? The intercept is an extrapolation to zero LD score,
   so it is far noisier than the slope.
2. **The null and the sweep.** On simulated genotypes with the project's own
   block-diagonal LD scores, is the intercept 1 when nothing is wrong -- and does
   it track `1/lambda` when a genomic-control correction is applied?
3. **What it would take to explain GIANT.** The shortfall implies a specific
   intercept, which is a falsifiable prediction for when the real fit is run.

Run: ``python experiments/z_calibration.py``
"""

from __future__ import annotations

import argparse

import numpy as np

from ppb import ldscore_regression
from ppb.ld_backend import BlockDiagonalLD, DenseLD
from ppb.simulate import _diploid_dosages

# Blocks that differ in how tight their LD is, so LD scores have the spread the
# regression needs. Equal-sized equally-correlated blocks give one LD score and
# no leverage on the intercept at all.
SPECS = [(100, rho) for rho in (0.97, 0.93, 0.85, 0.7, 0.5, 0.3)]


def consistency(rng, *, n=100_000, h2=0.30, n_blocks=200):
    """Recovery of a known intercept as the variant count grows.

    chi2 is drawn straight from (C2), so any error here is the estimator's.
    """
    rows = []
    for m in (2_000, 10_000, 50_000, 200_000):
        ell = np.exp(rng.normal(1.0, 0.8, size=m))        # log-normal, like real ones
        chisq = (1.0 + n * h2 / m * ell) * rng.chisquare(1, size=m)
        fit = ldscore_regression(chisq, ell, n, n_variants=m,
                                 blocks=rng.integers(0, n_blocks, size=m))
        rows.append((m, fit.intercept, fit.intercept_se, fit.h2))
    return rows


def _cohort(rng, n, repeats):
    specs = SPECS * repeats
    parts = []
    for k, rho in specs:
        parts.append(_diploid_dosages(
            n, [k], rng.uniform(0.05, 0.95, size=k), rho, rng).astype(np.float32))
    g = np.hstack(parts)
    g = (g - g.mean(axis=0)) / g.std(axis=0)

    blocks, block_of, start = [], np.empty(g.shape[1], dtype=int), 0
    for b, (k, _) in enumerate(specs):
        idx = np.arange(start, start + k)
        blocks.append((DenseLD(np.corrcoef(g[:, idx].T.astype(np.float64))), idx))
        block_of[idx] = b
        start += k
    return g, BlockDiagonalLD(blocks).ld_scores(), block_of


def genomic_control_sweep(rng, *, n=6_000, repeats=30, h2=0.4, causal=0.1):
    """The null, then a deflation sweep, on genotypes and block-diagonal LD scores."""
    g, ell, block_of = _cohort(rng, n, repeats)
    m = g.shape[1]

    beta = rng.normal(size=m).astype(np.float32) * (rng.random(m) < causal)
    bv = g @ beta
    y = (bv / bv.std() * np.float32(np.sqrt(h2))
         + rng.normal(size=n).astype(np.float32) * np.float32(np.sqrt(1.0 - h2)))
    y = (y - y.mean()) / y.std()
    chisq = ((g.T @ y).astype(np.float64) / np.sqrt(n)) ** 2

    rows = []
    for lam in (1.0, 1.3, 1.6, 2.0):
        fit = ldscore_regression(chisq / lam, ell, n, blocks=block_of)
        rows.append((lam, fit.intercept, fit.intercept_se, fit.z_scale,
                     fit.deflation_detected))
    return m, ell, chisq.mean(), rows


def giant_prediction():
    """The intercept that would account for the observed real-data shortfall.

    `REAL_DATA.md` Table 3: both GIANT targets report 1.5-2x below published
    individual-level accuracy. If that is a uniform `z` rescaling, (C3) fixes the
    intercept exactly -- so the real fit either lands here or the explanation is
    something else.
    """
    return [(shortfall, 1.0 / shortfall, np.sqrt(shortfall))
            for shortfall in (1.5, 1.75, 2.0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("1. Consistency (chi2 drawn from (C2); true intercept 1.0, h2 0.30)")
    print(f"{'variants':>10}{'intercept':>12}{'se':>9}{'h2':>9}")
    print("-" * 40)
    for m, intercept, se, h2 in consistency(rng):
        print(f"{m:>10,}{intercept:>12.4f}{se:>9.4f}{h2:>9.4f}")

    m, ell, mean_chisq, rows = genomic_control_sweep(rng)
    print(f"\n2. Genotypes + block-diagonal LD scores ({m:,} variants, "
          f"mean chi2 {mean_chisq:.3f})")
    print(f"   LD score: min {ell.min():.2f} mean {ell.mean():.2f} max {ell.max():.2f}")
    print(f"{'lambda_GC':>10}{'intercept':>12}{'1/lambda':>11}{'se':>9}"
          f"{'z scale':>10}{'detected':>10}")
    print("-" * 62)
    for lam, intercept, se, z_scale, detected in rows:
        print(f"{lam:>10.2f}{intercept:>12.4f}{1.0 / lam:>11.4f}{se:>9.4f}"
              f"{z_scale:>10.4f}{str(detected):>10}")

    print("\n3. What would explain the GIANT shortfall (REAL_DATA.md Table 3)")
    print(f"{'R2 shortfall':>14}{'implied intercept':>20}{'implied z scale':>18}")
    print("-" * 52)
    for shortfall, intercept, z_scale in giant_prediction():
        print(f"{shortfall:>14.2f}{intercept:>20.3f}{z_scale:>18.3f}")
    print("   A real fit landing in that band would confirm genomic control as")
    print("   the cause; an intercept near 1 would refute it.")


if __name__ == "__main__":
    main()
