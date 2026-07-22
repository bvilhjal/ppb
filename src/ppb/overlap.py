"""Detect shared training/target noise without erasing cohort signal.

The defensible detector compares block-level score products in an exposed
target and an independent reference while allowing their genuine signal to
differ by a multiplicative factor.  Correction additionally requires a
trainer-specific sensitivity basis.  Final weights alone do not determine
that basis, so an unknown trainer fails closed.

``overlap_slope`` and ``correct_numerator`` retain the original variant-count
model for reproducibility only.  New code should use :func:`fit_overlap` and
:func:`correct_overlap_numerator`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from .ld_backend import BlockDiagonalLD

_AVAILABLE_BASIS_KINDS = frozenset({"linear_trace", "jacobian_hutchinson"})


def block_products(w, z, ld: BlockDiagonalLD):
    """Return per-block score products and LD variances.

    The returned arrays are ``u_b = w_b' z_b`` and ``v_b = w_b' D_b w_b``.
    ``w`` and ``z`` must be aligned to the variants in ``ld``.
    """
    w = np.asarray(w, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if w.shape != z.shape or w.shape != (ld.m,):
        raise ValueError(
            f"w and z must have shape ({ld.m},); got {w.shape} and {z.shape}")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(z)):
        raise ValueError("w and z must contain only finite values")
    u = np.array([float(w[idx] @ z[idx]) for _, idx in ld.blocks])
    v = np.array([backend.quad(w[idx]) for backend, idx in ld.blocks])
    return u, v


@dataclass(frozen=True)
class OverlapBasis:
    """Trainer sensitivity to one unit of shared estimation noise.

    ``values[b]`` is the block basis ``q_b``.  For a linear trainer
    ``w = A z_train`` it is ``tr(A_b.T K_b)``, where ``K`` is the declared
    shared-noise covariance template.  ``support`` is the exact block support
    of the score numerator.  ``support_hash`` and ``provenance`` tie the basis
    to the evaluated score and trainer artifact.

    Use :meth:`unavailable` when the trainer operator cannot be reconstructed;
    variant count is deliberately not used as a fallback.
    """

    values: np.ndarray | None
    kind: str
    provenance: str
    support_hash: str | None = None
    support: np.ndarray | None = None
    mc_se: float | None = None

    def __post_init__(self):
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("basis provenance must be a non-empty string")
        if self.kind == "unavailable":
            if self.values is not None or self.support is not None:
                raise ValueError("an unavailable basis cannot contain values or support")
            return
        if self.kind not in _AVAILABLE_BASIS_KINDS:
            allowed = ", ".join(sorted(_AVAILABLE_BASIS_KINDS))
            raise ValueError(f"available basis kind must be one of: {allowed}")
        if not isinstance(self.support_hash, str) or not self.support_hash.strip():
            raise ValueError("an available basis requires a non-empty support_hash")
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("basis values must be a non-empty finite 1-D array")
        support = (np.ones(values.size, dtype=bool) if self.support is None
                   else np.asarray(self.support, dtype=bool))
        if support.shape != values.shape or not np.any(support):
            raise ValueError("basis support must select at least one block")
        if self.mc_se is not None and (not np.isfinite(self.mc_se) or self.mc_se < 0):
            raise ValueError("basis mc_se must be finite and non-negative")
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "support", support.copy())

    @classmethod
    def unavailable(cls, provenance: str) -> "OverlapBasis":
        """Represent an unknown trainer operator explicitly."""
        return cls(values=None, kind="unavailable", provenance=provenance)

    @property
    def available(self) -> bool:
        return self.kind != "unavailable"


@dataclass(frozen=True)
class OverlapFit:
    """Basis-aware dual-target fit and its correction eligibility."""

    status: str
    basis_kind: str
    basis_provenance: str
    support_hash: str | None
    alpha: float | None = None
    alpha_se: float | None = None
    gamma: float | None = None
    gamma_se: float | None = None
    gamma_z: float | None = None
    q_total: float | None = None
    q_fit: float | None = None
    numerator_target: float | None = None
    excluded_basis_fraction: float | None = None
    weighted_correlation: float | None = None
    vif: float | None = None
    condition_number: float | None = None
    heterogeneity_ratio: float | None = None
    n_blocks: int = 0
    n_groups: int = 0
    jackknife_alpha: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False)
    jackknife_gamma: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False)
    warnings: tuple[str, ...] = ()

    @property
    def can_correct(self) -> bool:
        return self.status == "correctable"


def _as_block_array(name, value, n=None):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 1 or (n is not None and out.shape != (n,)):
        expected = "a 1-D array" if n is None else f"shape ({n},)"
        raise ValueError(f"{name} must have {expected}; got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _profile_eiv(u_t, u_r, q, var_t, var_r, cov, alpha_bounds):
    """Profile gamma and minimize generalized Deming distance over alpha."""
    lo, hi = map(float, alpha_bounds)
    if not (0 < lo < hi):
        raise ValueError("alpha_bounds must satisfy 0 < lower < upper")

    def objective(alpha):
        variance = var_t + alpha * alpha * var_r - 2.0 * alpha * cov
        scale = var_t + alpha * alpha * var_r + 2.0 * alpha * np.abs(cov)
        minimum = 64.0 * np.finfo(float).eps * np.maximum(scale, np.finfo(float).tiny)
        if np.any(variance <= minimum) or not np.all(np.isfinite(variance)):
            return np.inf, np.nan
        weight = 1.0 / variance
        denominator = float(np.sum(weight * q * q))
        if denominator <= np.finfo(float).tiny:
            return np.inf, np.nan
        difference = u_t - alpha * u_r
        gamma = float(np.sum(weight * q * difference) / denominator)
        residual = difference - gamma * q
        return float(np.sum(weight * residual * residual)), gamma

    grid = np.geomspace(lo, hi, 257)
    scores = np.array([objective(alpha)[0] for alpha in grid])
    best = int(np.argmin(scores))
    if not np.isfinite(scores[best]):
        raise ValueError("no positive residual variance within alpha_bounds")
    boundary = best in (0, grid.size - 1)
    if boundary:
        alpha = float(grid[best])
        score, gamma = objective(alpha)
        return alpha, gamma, score, True

    left, right = float(grid[best - 1]), float(grid[best + 1])
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc, _ = objective(c)
    fd, _ = objective(d)
    for _ in range(80):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc, _ = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd, _ = objective(d)
    candidates = (float(grid[best]), (left + right) / 2.0)
    alpha = min(candidates, key=lambda value: objective(value)[0])
    score, gamma = objective(alpha)
    return float(alpha), float(gamma), float(score), False


def _identification(u_reference, q, weight):
    x = np.sqrt(weight) * u_reference
    z = np.sqrt(weight) * q
    nx, nz = float(np.linalg.norm(x)), float(np.linalg.norm(z))
    if nx <= np.finfo(float).tiny or nz <= np.finfo(float).tiny:
        return np.nan, np.inf, np.inf, False
    correlation = float(np.clip((x @ z) / (nx * nz), -1.0, 1.0))
    remaining = max(0.0, 1.0 - correlation * correlation)
    vif = np.inf if remaining <= 1e-12 else 1.0 / remaining
    gram = np.array([[1.0, correlation], [correlation, 1.0]])
    condition = float(np.linalg.cond(gram))
    identified = np.isfinite(vif) and np.linalg.matrix_rank(gram, tol=1e-10) == 2
    return correlation, float(vif), condition, bool(identified)


def _jackknife_se(estimates):
    estimates = np.asarray(estimates, dtype=np.float64)
    k = estimates.size
    if np.ptp(estimates) == 0.0:
        return 0.0
    return float(np.sqrt((k - 1) / k * np.sum((estimates - estimates.mean()) ** 2)))


def fit_overlap(
    u_target,
    u_reference,
    var_target,
    var_reference,
    *,
    basis: OverlapBasis | None,
    cov_target_reference=None,
    groups=None,
    alpha_bounds=(0.05, 20.0),
    detection_z=3.0,
    min_blocks=50,
    min_groups=15,
    max_vif=2.0,
    max_condition=5.0,
) -> OverlapFit:
    """Jointly estimate cohort signal scale ``alpha`` and noise coupling ``gamma``.

    The block model is ``u_target = alpha * signal + gamma * q + error`` and
    ``u_reference = signal + error``.  ``var_*`` are supplied sampling-noise
    variances of the block products, not LD score variances.  The fit minimizes
    generalized Deming distances and uses delete-group refits for uncertainty
    and influence checks.

    Correction is allowed only when the basis is available on the exact score
    support, both design components are identified, the one-sided detection
    threshold is met, and the jackknife is stable.  Status values explain every
    refusal; :func:`correct_overlap_numerator` enforces them.
    """
    u_t = _as_block_array("u_target", u_target)
    n = u_t.size
    if n == 0:
        raise ValueError("need at least one block")
    u_r = _as_block_array("u_reference", u_reference, n)
    var_t = _as_block_array("var_target", var_target, n)
    var_r = _as_block_array("var_reference", var_reference, n)
    if np.any(var_t < 0) or np.any(var_r < 0):
        raise ValueError("block noise variances must be non-negative")
    cov = (np.zeros(n, dtype=np.float64) if cov_target_reference is None
           else _as_block_array("cov_target_reference", cov_target_reference, n))
    covariance_scale = var_t * var_r
    tolerance = (64.0 * np.finfo(float).eps
                 * np.maximum(covariance_scale, np.finfo(float).tiny))
    if np.any(cov * cov > covariance_scale + tolerance):
        raise ValueError("each target/reference noise covariance must be positive semidefinite")
    if basis is None:
        basis = OverlapBasis.unavailable("no basis supplied")
    if not isinstance(basis, OverlapBasis):
        raise TypeError("basis must be an OverlapBasis or None")
    if not basis.available:
        return OverlapFit(
            status="basis_unavailable", basis_kind=basis.kind,
            basis_provenance=basis.provenance, support_hash=None,
            warnings=("trainer sensitivity basis is unavailable; correction refused",))
    if basis.values.shape != (n,):
        raise ValueError(f"basis must have one value per block ({n},); got {basis.values.shape}")

    q = basis.values
    support = basis.support
    q_total = float(np.sum(q[support]))
    numerator_target = float(np.sum(u_t[support]))
    informative = support & ((var_t > 0) | (var_r > 0))
    excluded_abs = float(np.sum(np.abs(q[support & ~informative])))
    total_abs = float(np.sum(np.abs(q[support])))
    excluded_fraction = (np.inf if total_abs == 0 and excluded_abs > 0
                         else excluded_abs / total_abs if total_abs > 0 else 0.0)
    q_fit = float(np.sum(q[informative]))

    if groups is None:
        k_default = min(20, max(1, int(np.sum(informative))))
        groups = np.floor(np.arange(n) * k_default / n).astype(int)
    groups = np.asarray(groups)
    if groups.shape != (n,):
        raise ValueError(f"groups must have one entry per block ({n},); got {groups.shape}")
    groups_fit = groups[informative]
    unique_groups = np.unique(groups_fit)
    n_blocks_fit = int(np.sum(informative))
    n_groups_fit = int(unique_groups.size)

    common = dict(
        basis_kind=basis.kind,
        basis_provenance=basis.provenance,
        support_hash=basis.support_hash,
        q_total=q_total,
        q_fit=q_fit,
        numerator_target=numerator_target,
        excluded_basis_fraction=excluded_fraction,
        n_blocks=n_blocks_fit,
        n_groups=n_groups_fit,
    )
    if n_blocks_fit < min_blocks or n_groups_fit < min_groups:
        return OverlapFit(
            status="insufficient_data", **common,
            warnings=(f"need at least {min_blocks} informative blocks and "
                      f"{min_groups} groups",))

    fit_args = tuple(array[informative] for array in (u_t, u_r, q, var_t, var_r, cov))
    alpha, gamma, objective, boundary = _profile_eiv(*fit_args, alpha_bounds)
    residual_variance = (fit_args[3] + alpha * alpha * fit_args[4]
                         - 2.0 * alpha * fit_args[5])
    residual_scale = (fit_args[3] + alpha * alpha * fit_args[4]
                      + 2.0 * alpha * np.abs(fit_args[5]))
    if np.any(residual_variance <= (64.0 * np.finfo(float).eps
                                    * np.maximum(residual_scale,
                                                 np.finfo(float).tiny))):
        return OverlapFit(
            status="unstable", alpha=alpha, gamma=gamma, **common,
            warnings=("fitted residual noise variance is numerically non-positive",))
    correlation, vif, condition, identified = _identification(
        fit_args[1], fit_args[2], 1.0 / residual_variance)
    heterogeneity_ratio = objective / max(1, n_blocks_fit - 2)

    loo_alpha = []
    loo_gamma = []
    loo_boundary = False
    for group in unique_groups:
        keep = groups_fit != group
        try:
            alpha_g, gamma_g, _, boundary_g = _profile_eiv(
                *(array[keep] for array in fit_args), alpha_bounds)
        except ValueError:
            alpha_g, gamma_g, boundary_g = np.nan, np.nan, True
        loo_alpha.append(alpha_g)
        loo_gamma.append(gamma_g)
        loo_boundary = loo_boundary or boundary_g
    loo_alpha = np.asarray(loo_alpha, dtype=np.float64)
    loo_gamma = np.asarray(loo_gamma, dtype=np.float64)
    finite_jackknife = bool(np.all(np.isfinite(loo_alpha)) and np.all(np.isfinite(loo_gamma)))
    alpha_se = _jackknife_se(loo_alpha) if finite_jackknife else np.nan
    gamma_se = _jackknife_se(loo_gamma) if finite_jackknife else np.nan
    if np.isfinite(gamma_se) and gamma_se > 0:
        gamma_z = gamma / gamma_se
    else:
        gamma_z = np.nan

    result = dict(
        alpha=alpha, alpha_se=alpha_se, gamma=gamma, gamma_se=gamma_se,
        gamma_z=gamma_z, weighted_correlation=correlation, vif=vif,
        condition_number=condition, heterogeneity_ratio=heterogeneity_ratio,
        jackknife_alpha=loo_alpha,
        jackknife_gamma=loo_gamma, **common)

    warnings_out = []
    incomplete_basis = excluded_fraction > 1e-12
    nonidentified = not identified or vif > 10.0 or condition > 30.0
    weak = vif > max_vif or condition > max_condition
    basis_unstable = (basis.mc_se is not None and
                      (q_total == 0 or basis.mc_se / abs(q_total) > 0.05))
    degenerate_jackknife = (not finite_jackknife or not np.isfinite(alpha_se)
                            or not np.isfinite(gamma_se)
                            or alpha_se <= 0 or gamma_se <= 0)
    basic_instability = boundary or loo_boundary or degenerate_jackknife

    if incomplete_basis:
        status = "excluded_basis"
        warnings_out.append("a fit-excluded block has non-zero basis mass")
    elif nonidentified:
        status = "nonidentifiable"
        warnings_out.append("signal scale and shared-noise basis are collinear")
    elif weak:
        status = "weak_identification"
        warnings_out.append("signal scale and shared-noise basis are weakly separated")
    elif heterogeneity_ratio > 2.0:
        status = "heterogeneous"
        warnings_out.append("block residual heterogeneity exceeds twice its expectation")
    elif basis_unstable or basic_instability:
        status = "unstable"
        warnings_out.append("basis or delete-group fit is unstable")
    elif not np.isfinite(gamma_z):
        status = "unstable"
        warnings_out.append("delete-group uncertainty is zero or non-finite")
    elif gamma <= 0 or gamma_z < detection_z:
        status = "not_detected"
    else:
        positive_fraction = float(np.mean(loo_gamma > 0))
        max_change = float(np.max(np.abs(loo_gamma - gamma)))
        allowed_change = max(2.0 * gamma_se, 0.25 * abs(gamma), 1e-15)
        corrected = numerator_target - gamma * q_total
        sign_reversal = (numerator_target != 0.0
                         and corrected * numerator_target < 0.0)
        if sign_reversal:
            status = "sign_reversal"
            warnings_out.append("correction would reverse the signed numerator")
        elif positive_fraction < 0.9 or max_change > allowed_change:
            status = "unstable"
            warnings_out.append("detected coupling is not stable across groups")
        else:
            status = "correctable"
    return OverlapFit(status=status, warnings=tuple(warnings_out), **result)


def correct_overlap_numerator(fit: OverlapFit) -> float:
    """Return the corrected target numerator owned by a correctable fit.

    The numerator and total sensitivity basis come from the exact support used
    to construct ``fit``; callers cannot accidentally supply a different block
    count or extrapolate across excluded active blocks.
    """
    if not isinstance(fit, OverlapFit):
        raise TypeError("fit must be an OverlapFit")
    if not fit.can_correct:
        raise ValueError(f"overlap correction refused: fit status is {fit.status}")
    return float(fit.numerator_target - fit.gamma * fit.q_total)


# ---------------------------------------------------------------------------
# Deprecated compatibility path


@dataclass
class OverlapEstimate:
    """Legacy fixed-signal, variant-count overlap estimate."""

    gamma: float
    se: float
    z: float
    n_blocks: int


def overlap_slope(u_target, u_reference, m, v_target, v_reference, *,
                  groups=None) -> OverlapEstimate:
    """Legacy through-origin variant-count slope; use :func:`fit_overlap`.

    This compatibility helper fixes the cohort signal scale to one and assumes
    variant count is the trainer sensitivity basis.  Those assumptions are not
    generally identifiable from final weights.
    """
    warnings.warn(
        "overlap_slope fixes alpha=1 and uses variant count as the basis; "
        "use fit_overlap with an explicit OverlapBasis",
        DeprecationWarning, stacklevel=2)
    du = np.asarray(u_target, dtype=np.float64) - np.asarray(u_reference, dtype=np.float64)
    dv = np.asarray(v_target, dtype=np.float64) + np.asarray(v_reference, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    ok = dv > 0
    if ok.sum() < 4:
        raise ValueError("need at least 4 blocks with positive score variance")
    if groups is None:
        n_all = ok.size
        groups = np.repeat(np.arange(20), int(np.ceil(n_all / 20)))[:n_all]
    groups = np.asarray(groups)
    if groups.shape != ok.shape:
        raise ValueError(
            f"groups must have one entry per block ({ok.size},); got {groups.shape}")
    du, dv, m, groups = du[ok], dv[ok], m[ok], groups[ok]
    wt = 1.0 / dv
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError("need at least 2 jackknife groups")

    def fit(keep):
        w_ = wt[keep]
        return float((w_ * m[keep] * du[keep]).sum()
                     / (w_ * m[keep] * m[keep]).sum())

    gamma = fit(np.ones(len(m), bool))
    ests = np.array([fit(groups != group) for group in uniq])
    se = _jackknife_se(ests)
    if se > 0:
        z = gamma / se
    else:
        z = math.copysign(np.inf, gamma) if gamma != 0.0 else np.nan
    return OverlapEstimate(gamma=gamma, se=se, z=z, n_blocks=int(ok.sum()))


def correct_numerator(num, gamma, m_total):
    """Legacy variant-count correction; use :func:`correct_overlap_numerator`."""
    warnings.warn(
        "correct_numerator uses an externally supplied variant count; use "
        "correct_overlap_numerator with a validated OverlapFit",
        DeprecationWarning, stacklevel=2)
    return float(num - gamma * m_total)
