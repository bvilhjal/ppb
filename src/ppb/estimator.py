"""Summary-statistic estimators of polygenic-score predictive accuracy.

    R^2 = (w^T z)^2 / (w^T D w * var_y)
    MSE = var_y - 2 w^T z + w^T D w

where ``w`` are PGS weights, ``z = (1/N) X^T y`` are the target marginal summary
statistics, and ``D = (1/N) X^T X`` is the LD matrix (supplied as an LD backend
that computes ``w^T D w``). ``X`` is standardized; ``var_y`` is the variance of
``y`` on the scale used to form ``z``. With standardized ``y``, ``var_y = 1``.
See ``docs/METHOD.md``.
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
    if not np.isfinite(w).all() or not np.isfinite(z).all():
        raise ValueError("weights and z must contain only finite numbers")
    wz = float(np.dot(w, z))
    if not np.isfinite(wz):
        raise ValueError("w^T z is not finite")
    return w, wz


def _var_y(value) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("var_y must be finite and strictly positive")
    return value


def r2(weights, z, ld: LDBackend, var_y: float = 1.0) -> float:
    """Estimated prediction ``R^2`` from summary-level inputs.

    Raises ``ValueError`` if ``w^T D w`` is not strictly positive (an undefined
    or invalid ratio -- e.g. all-zero weights, or a non-PSD ``D``).
    """
    var_y = _var_y(var_y)
    w, num = _wz(weights, z)
    den = ld.quad(w)
    if not np.isfinite(den):
        raise ValueError(f"w^T D w = {den!r} is not finite")
    if not den > 0.0:
        raise ValueError(f"w^T D w = {den!r} is not positive; R^2 is undefined")
    value = (num * num) / (den * var_y)
    if not np.isfinite(value):
        raise ValueError("estimated R^2 is not finite")
    return value


def mse(weights, z, ld: LDBackend, var_y: float = 1.0) -> float:
    """Estimated mean squared error from the same summary-level inputs.

    Raises ``ValueError`` if ``w^T D w`` is negative, which a non-PSD ``D`` can
    produce and which would silently understate the error. Unlike :func:`r2`,
    zero is allowed: all-zero weights predict nothing, and ``MSE = var_y`` is
    the right answer for them rather than an undefined ratio.
    """
    var_y = _var_y(var_y)
    w, wz = _wz(weights, z)
    den = ld.quad(w)
    if not np.isfinite(den):
        raise ValueError(f"w^T D w = {den!r} is not finite")
    if den < 0.0:
        raise ValueError(
            f"w^T D w = {den!r} is negative; MSE would be understated "
            "(a non-PSD LD approximation)")
    value = var_y - 2.0 * wz + den
    if not np.isfinite(value):
        raise ValueError("estimated MSE is not finite")
    return value
