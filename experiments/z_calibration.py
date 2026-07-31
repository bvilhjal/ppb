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
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ppb.ld_backend import BlockDiagonalLD, DenseLD
from ppb.simulate import _diploid_dosages

# ---------------------------------------------------------------------------
# LD-score regression: is the target `z` on the scale PPB assumes?
#
# `R^2 = (w^T z)^2 / (w^T D w)` puts `z` in the numerator alone, so a target
# whose summary statistics are off by a factor `c` reports `c^2 R^2` and
# nothing cancels it. Real GWAS move that factor routinely -- genomic control,
# applied at study and meta level, inflates every standard error and deflates
# every `z` -- and PPB has until now performed no check on it. Against
# published individual-level accuracy, both GIANT targets come in 1.5-2x low
# while a GLGC target agrees to 2%.
#
# LD-score regression (Bulik-Sullivan et al. 2015,
# https://doi.org/10.1038/ng.3211) supplies the missing diagnostic from the
# target summary statistics and the LD reference PPB already ships:
#
# **(C2) LD-score regression.**
#
#     E[chi2_j] = intercept + (N h2 / M) l_j
#
# `l_j` is the LD score (C1). The slope carries heritability; the **intercept**
# is the part that matters here. With neither confounding nor correction it is
# 1. Population stratification and cryptic relatedness push it *up*. Dividing
# every chi2 by a genomic-control lambda pushes it *down*, to `1/lambda`, and
# takes the slope with it.
#
# **(C3) Implied `z` scale.**
#
#     z_true = z_observed / sqrt(intercept),    R^2_true = R^2_observed / intercept
#
# **What is and is not identified.** The intercept is one number standing for
# two mechanisms -- confounding, which raises it, and over-correction, which
# lowers it -- so it cannot separate them, and (C3) is only a *correction*
# under the assumption that the departure is a uniform rescaling. That is
# exactly true for genomic control and false for stratification, which adds a
# constant to chi2 rather than scaling it.
#
# One direction survives that ambiguity. No confounding mechanism pushes the
# intercept below 1, and the block-diagonal reference's LD scores are
# systematically *low* (they omit the long-range terms the reference defines
# away), which biases the intercept *up*. **An intercept below 1 is therefore
# conservative evidence of over-correction**, and that is the case this
# diagnostic is for.
#
# This is a reimplementation of the LDSC estimator, not a port: the weighting
# is the standard two-step scheme but the software's regression-SNP selection,
# per-chromosome handling, and partitioned variants are not reproduced.
# ---------------------------------------------------------------------------


@dataclass
class LDScoreFit:
    """Result of (C2), with the `z`-scale reading of (C3) attached.

    ``z_scale`` is the factor the observed `z` must be **multiplied** by, and
    ``r2_scale`` its square -- the factor for an R^2. Both are ``None`` when the
    intercept is non-positive, which is a degenerate fit rather than an infinite
    correction.
    """

    intercept: float
    slope: float
    h2: float
    n_variants: int
    mean_chisq: float
    intercept_se: float | None = None
    h2_se: float | None = None
    n_blocks: int | None = None
    leverage: float = 0.0
    z_scale: float | None = None
    r2_scale: float | None = None
    deflation_detected: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _wls(x, y, weights):
    """Weighted least squares of ``y`` on ``[1, x]``; returns (intercept, slope)."""
    sw = weights.sum()
    mx = (weights * x).sum() / sw
    my = (weights * y).sum() / sw
    sxx = (weights * (x - mx) ** 2).sum()
    if sxx <= 0.0:
        raise ValueError("LD scores carry no variance; the regression is degenerate")
    slope = (weights * (x - mx) * (y - my)).sum() / sxx
    return my - slope * mx, slope


def _fit(chisq, ld_scores, n, m, *, steps=2):
    """Two-step weighted fit: weights depend on the h2 they are estimating."""
    ell = np.maximum(ld_scores, 1.0)
    intercept, slope = _wls(ld_scores, chisq, np.ones_like(chisq))
    for _ in range(steps):
        # 1/l for LD-induced dependence between neighbouring chi2, and
        # 1/(1 + N h2 l / M)^2 for the heteroscedasticity of chi2 itself.
        expected = np.maximum(intercept + max(slope, 0.0) * ell, 1e-8)
        weights = 1.0 / (ell * expected ** 2)
        intercept, slope = _wls(ld_scores, chisq, weights)
    return float(intercept), float(slope)


def ldscore_regression(chisq, ld_scores, n, *, n_variants=None, blocks=None,
                       steps: int = 2) -> LDScoreFit:
    """Fit (C2) and report what it implies for the scale of `z` (C3).

    ``chisq`` are the target's squared association statistics ``(beta/se)^2``,
    ``ld_scores`` the reference's (C1) on the same variants, and ``n`` the GWAS
    sample size. ``n_variants`` is the ``M`` heritability is defined over,
    defaulting to the number of variants regressed.

    ``blocks`` optionally assigns each variant to an LD block; delete-one-block
    jackknife standard errors are then reported. Blocks -- not variants -- are
    the independent units, so a naive regression standard error would be far too
    small.
    """
    chisq = np.asarray(chisq, dtype=np.float64)
    ell = np.asarray(ld_scores, dtype=np.float64)
    if chisq.shape != ell.shape or chisq.ndim != 1:
        raise ValueError(
            f"chisq {chisq.shape} and ld_scores {ell.shape} must be the same 1-D shape")
    if not np.isfinite(chisq).all() or np.any(chisq < 0.0):
        raise ValueError("chisq must be finite and non-negative")
    if not np.isfinite(ell).all() or np.any(ell <= 0.0):
        raise ValueError("ld_scores must be finite and positive")
    n = float(n)
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError(f"n must be a positive sample size; got {n}")
    m = float(n_variants if n_variants is not None else chisq.size)
    if not math.isfinite(m) or m <= 0.0:
        raise ValueError(f"n_variants must be positive; got {m}")
    if chisq.size < 3:
        raise ValueError("LD-score regression needs at least 3 variants")

    intercept, slope = _fit(chisq, ell, n, m, steps=steps)
    h2 = slope * m / n

    intercept_se = h2_se = None
    n_blocks = None
    if blocks is not None:
        groups = np.asarray(blocks)
        if groups.shape != chisq.shape:
            raise ValueError(
                f"blocks has shape {groups.shape}, expected {chisq.shape}")
        labels = np.unique(groups)
        n_blocks = int(labels.size)
        if n_blocks >= 2:
            drops = []
            for label in labels:
                keep = groups != label
                if keep.sum() < 3:
                    continue
                a, b = _fit(chisq[keep], ell[keep], n, m, steps=steps)
                drops.append((a, b * m / n))
            if len(drops) >= 2:
                arr = np.asarray(drops, dtype=np.float64)
                k = arr.shape[0]
                scale = (k - 1) / k
                intercept_se = float(math.sqrt(scale * ((arr[:, 0] - arr[:, 0].mean()) ** 2).sum()))
                h2_se = float(math.sqrt(scale * ((arr[:, 1] - arr[:, 1].mean()) ** 2).sum()))

    z_scale = r2_scale = None
    note = ""
    if intercept > 0.0:
        z_scale = 1.0 / math.sqrt(intercept)
        r2_scale = 1.0 / intercept
    else:
        note = (f"intercept {intercept:.4g} is not positive; the fit is degenerate "
                "and implies no scale")

    # The intercept is an extrapolation to l = 0, so its precision depends on how
    # far that is from the data. Report the distance in units of the spread.
    leverage = float(ell.mean() / ell.std()) if ell.std() > 0.0 else float("inf")

    # Deflation is a claim about a number that is noisy at small m: at 2,000
    # variants the intercept's standard error exceeds 1, so "intercept < 1" alone
    # is not evidence of anything. Require the interval to clear 1, and refuse to
    # claim detection at all when no jackknife was possible.
    deflated = False
    if intercept <= 0.0:
        pass
    elif intercept_se is None:
        if intercept < 1.0:
            note = (f"intercept {intercept:.4g} < 1, but no blocks were supplied so "
                    "it carries no standard error; pass `blocks` before reading "
                    "this as deflation")
    elif intercept + 2.0 * intercept_se < 1.0:
        deflated = True
        note = (f"intercept {intercept:.4g} +/- {intercept_se:.3g} is more than two "
                "standard errors below 1. No confounding mechanism lowers the "
                "intercept, and block-diagonal LD scores bias it upward, so this "
                "is conservative evidence that the target was over-corrected")
    elif intercept < 1.0:
        note = (f"intercept {intercept:.4g} +/- {intercept_se:.3g} is below 1 but "
                "not by two standard errors; not evidence of deflation")

    return LDScoreFit(
        intercept=intercept, slope=float(slope), h2=float(h2),
        n_variants=int(chisq.size), mean_chisq=float(chisq.mean()),
        intercept_se=intercept_se, h2_se=h2_se, n_blocks=n_blocks,
        leverage=leverage,
        z_scale=z_scale, r2_scale=r2_scale, deflation_detected=deflated, note=note,
    )


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
