"""LD-score regression: is the target `z` on the scale PPB assumes?

`R^2 = (w^T z)^2 / (w^T D w)` puts `z` in the numerator alone, so a target whose
summary statistics are off by a factor `c` reports `c^2 R^2` and nothing cancels
it. Real GWAS move that factor routinely -- genomic control, applied at study and
meta level, inflates every standard error and deflates every `z` -- and PPB has
until now performed no check on it. Against published individual-level accuracy,
both GIANT targets come in 1.5-2x low while a GLGC target agrees to 2%.

LD-score regression (Bulik-Sullivan et al. 2015,
https://doi.org/10.1038/ng.3211) supplies the missing diagnostic from the target
summary statistics and the LD reference PPB already ships:

**(C2) LD-score regression.**

    E[chi2_j] = intercept + (N h2 / M) l_j

`l_j` is the LD score (C1). The slope carries heritability; the **intercept** is
the part that matters here. With neither confounding nor correction it is 1.
Population stratification and cryptic relatedness push it *up*. Dividing every
chi2 by a genomic-control lambda pushes it *down*, to `1/lambda`, and takes the
slope with it.

**(C3) Implied `z` scale.**

    z_true = z_observed / sqrt(intercept),    R^2_true = R^2_observed / intercept

**What is and is not identified.** The intercept is one number standing for two
mechanisms -- confounding, which raises it, and over-correction, which lowers it
-- so it cannot separate them, and (C3) is only a *correction* under the
assumption that the departure is a uniform rescaling. That is exactly true for
genomic control and false for stratification, which adds a constant to chi2
rather than scaling it.

One direction survives that ambiguity. No confounding mechanism pushes the
intercept below 1, and the block-diagonal reference's LD scores are
systematically *low* (they omit the long-range terms the reference defines away),
which biases the intercept *up*. **An intercept below 1 is therefore
conservative evidence of over-correction**, and that is the case this diagnostic
is for.

This is a reimplementation of the LDSC estimator, not a port: the weighting is
the standard two-step scheme but the software's regression-SNP selection,
per-chromosome handling, and partitioned variants are not reproduced.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = ["LDScoreFit", "ldscore_regression"]


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
