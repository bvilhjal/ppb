"""Summary-statistic estimators of polygenic-score predictive accuracy.

    R^2 = (w^T z)^2 / (w^T D w)
    MSE = var_y - 2 w^T z + w^T D w

where ``w`` are PGS weights, ``z = (1/N) X^T y`` are the target marginal summary
statistics, and ``D = (1/N) X^T X`` is the LD matrix (supplied as an LD backend
that computes ``w^T D w``). ``X`` and ``y`` are standardized (mean 0, variance 1);
with standardized ``y``, ``var_y = 1``. See ``docs/METHOD.md``.
"""

from __future__ import annotations

import numpy as np

from .ld_backend import LDBackend


def _wz(weights, z):
    w = np.asarray(weights, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if w.shape != z.shape or w.ndim != 1:
        raise ValueError(
            f"weights and z must be 1-D of equal length; got {w.shape} and {z.shape}")
    return w, float(np.dot(w, z))


def r2(weights, z, ld: LDBackend) -> float:
    """Estimated prediction ``R^2`` from summary-level inputs.

    Raises ``ValueError`` if ``w^T D w`` is not strictly positive (an undefined
    or invalid ratio -- e.g. all-zero weights, or a non-PSD ``D``).
    """
    w, num = _wz(weights, z)
    den = ld.quad(w)
    if not den > 0.0:
        raise ValueError(f"w^T D w = {den!r} is not positive; R^2 is undefined")
    return (num * num) / den


def mse(weights, z, ld: LDBackend, var_y: float = 1.0) -> float:
    """Estimated mean squared error from the same summary-level inputs.

    Raises ``ValueError`` if ``w^T D w`` is negative, which a non-PSD ``D`` can
    produce and which would silently understate the error. Unlike :func:`r2`,
    zero is allowed: all-zero weights predict nothing, and ``MSE = var_y`` is
    the right answer for them rather than an undefined ratio.
    """
    w, wz = _wz(weights, z)
    den = ld.quad(w)
    if den < 0.0:
        raise ValueError(
            f"w^T D w = {den!r} is negative; MSE would be understated "
            "(a non-PSD LD approximation)")
    return float(var_y) - 2.0 * wz + den
