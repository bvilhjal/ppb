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

import warnings

import numpy as np

from .ld_backend import LDBackend


def frozen_to_dosage(weights, sd_ref):
    """Convert LDpred3 standardized ``WEIGHT`` to per-allele dosage weights.

    ``b_j = WEIGHT_j / SD_REF_j``. ``SD_REF`` is the *fit-cohort* dosage SD.
    Variants with ``SD_REF <= 0`` (monomorphic in the fit cohort) are set to
    zero rather than divided; when that zeroes a nonzero weight a warning
    reports how many, since those variants silently vanish from
    ``n_variants_scored``. The result still needs the *target* genotype SD
    before it multiplies ``D``.
    """
    w = np.asarray(weights, dtype=np.float64)
    sd = np.asarray(sd_ref, dtype=np.float64)
    if w.shape != sd.shape or w.ndim != 1:
        raise ValueError(
            f"weights and sd_ref must be 1-D of equal length; got {w.shape} "
            f"and {sd.shape}")
    if not np.isfinite(w).all() or not np.isfinite(sd).all():
        raise ValueError("weights and sd_ref must contain only finite numbers")
    if np.any(sd < 0.0):
        raise ValueError("sd_ref must be non-negative")
    out = np.zeros(w.shape[0], dtype=np.float64)
    positive = sd > 0.0
    out[positive] = w[positive] / sd[positive]
    n_zeroed = int(np.count_nonzero(~positive & (w != 0.0)))
    if n_zeroed:
        warnings.warn(
            f"frozen_to_dosage zeroed {n_zeroed} nonzero weight(s) with "
            "SD_REF == 0 (monomorphic in the fit cohort); they drop out of "
            "n_variants_scored",
            stacklevel=2)
    return out


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


def _r2_from_quad(num: float, den: float, var_y: float) -> float:
    if not np.isfinite(den):
        raise ValueError(f"w^T D w = {den!r} is not finite")
    if not den > 0.0:
        raise ValueError(f"w^T D w = {den!r} is not positive; R^2 is undefined")
    value = (num * num) / (den * var_y)
    if not np.isfinite(value):
        raise ValueError("estimated R^2 is not finite")
    return value


def _mse_from_quad(wz: float, den: float, var_y: float) -> float:
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


def corrected_r2(num: float, den: float, n_eff: float,
                 var_y: float = 1.0) -> tuple[float, float, float]:
    """Finite-sample correction (X3) and its delta-method SE.

    The plug-in ``(wᵀẑ)²`` is biased upward by ``≈ wᵀ D w / N`` on the
    standardized-phenotype scale of Algorithm V. Subtract that term from the
    squared numerator before dividing. The reported SE is the leading
    ``1/N`` sampling term, ``2 √(R² / N)``, not the block jackknife (G2).

    Returns ``(r2_raw, r2_corrected, se)``. A corrected value may be
    negative when the numerator is noise; it is not clamped.
    """
    var_y = _var_y(var_y)
    n_eff = float(n_eff)
    if not np.isfinite(n_eff) or n_eff <= 2.0:
        raise ValueError("n_eff must be finite and greater than 2")
    raw = _r2_from_quad(num, den, var_y)
    corrected = (num * num - den / n_eff) / (den * var_y)
    if not np.isfinite(corrected):
        raise ValueError("corrected R^2 is not finite")
    se = 2.0 * float(np.sqrt(max(raw, 0.0) / n_eff))
    return raw, float(corrected), se


def r2(weights, z, ld: LDBackend, var_y: float = 1.0) -> float:
    """Estimated prediction ``R^2`` from summary-level inputs.

    Raises ``ValueError`` if ``w^T D w`` is not strictly positive (an undefined
    or invalid ratio -- e.g. all-zero weights, or a non-PSD ``D``).
    """
    var_y = _var_y(var_y)
    w, num = _wz(weights, z)
    return _r2_from_quad(num, ld.quad(w), var_y)


def mse(weights, z, ld: LDBackend, var_y: float = 1.0) -> float:
    """Estimated mean squared error from the same summary-level inputs.

    Raises ``ValueError`` if ``w^T D w`` is negative, which a non-PSD ``D`` can
    produce and which would silently understate the error. Unlike :func:`r2`,
    zero is allowed: all-zero weights predict nothing, and ``MSE = var_y`` is
    the right answer for them rather than an undefined ratio.
    """
    var_y = _var_y(var_y)
    w, wz = _wz(weights, z)
    return _mse_from_quad(wz, ld.quad(w), var_y)
