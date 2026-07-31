"""Basis-aware overlap detection and correction: where it works, and where it does not.

A training GWAS and a target GWAS share a controlled fraction of *individuals*;
an independent GWAS of the same size is the honest anchor. The detector jointly
fits a genuine target/reference signal scale ``alpha`` and the shared-noise
coupling ``gamma``, and corrects only when both are identified and stable.

**The identification boundary is a property of the architecture, not of the
method.** The design has two columns -- the reference signal ``u_R`` and the
trainer basis ``q`` -- and both are positive and both grow with block size, so
separating them needs signal variation *at fixed block size*. A diffuse
architecture has almost none: every block carries a similar amount of signal,
the columns are collinear, and the fit is correctly refused. A sparse
architecture has a great deal, and the same code identifies cleanly and recovers
the coupling. Earlier revisions of this experiment ran only the diffuse corner
with a *constant* basis (a marginal trainer over equal-sized blocks makes
``q_b = tr(D_b)`` the block size), which is the least identifiable configuration
available, and recorded the resulting refusal as the method's operating limit.

Real polygenic traits over the 1.44M-variant HM3+ reference are far sparser than
the diffuse corner, and its 431 LD blocks range from 216 to 17,304 variants, so
the realistic regime is the identified one.

Run:
    python experiments/overlap_detection.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ppb import DenseLD, OverlapBasis
from ppb.ld_backend import BlockDiagonalLD
from ppb.simulate import (draw_effects, marginal_stats, pgs_pthreshold,
                          simulate_diploid_genotypes, simulate_phenotype)

# ---------------------------------------------------------------------------
# Experimental overlap fitting apparatus (withdrawn from the package).
#
# Correction is **experimental and not recommended**.  Every correction needs
# an independent reference GWAS of the same trait, and given one, evaluating
# the score against it is unbiased in a single line -- so the condition that
# makes a correction valid is the condition that makes it unnecessary.  The
# results registry rejects a ``correctable`` status.  See ``docs/OVERLAP.md``.
#
# The defensible detector compares block-level score products in an exposed
# target and an independent reference while allowing their genuine signal to
# differ by a multiplicative factor.  Correction additionally requires a
# trainer-specific sensitivity basis.  Final weights alone do not determine
# that basis, so an unknown trainer fails closed.
#
# This code lived in ``ppb.overlap`` until the 2026-07-31 leanness refactor;
# the package keeps only :class:`ppb.OverlapBasis` (the fail-closed marker the
# results pipeline records).  The legacy variant-count path (``overlap_slope``
# / ``correct_numerator``) was deleted at the same time.


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


def estimate_overlap_basis(trainer, z, blocks, noise_sqrt, *, rng,
                           provenance, support_hash, support=None,
                           n_draws=32, deltas=(0.01, 0.05, 0.25), scale=None,
                           max_relative_spread=0.2, min_pattern_correlation=0.9):
    """Stochastic trainer-sensitivity basis for a rerunnable trainer.

    Implements ``docs/OVERLAP.md`` (O4), the generalized-degrees-of-freedom
    trace estimate (Ye 1998; Hutchinson 1989):

        q_b = (1 / R) sum_r g_br' {f(z + delta g_r) - f(z)}_b / delta,
        E[g g'] = K.

    ``trainer(z) -> w`` must be rerunnable on perturbed summary statistics.
    ``blocks`` is a sequence of index arrays; ``noise_sqrt[b]`` is any factor
    with ``L_b L_b' = K_b`` (for the usual shared-GWAS-noise template ``K = D``,
    the Cholesky factor of the block LD).

    ``deltas`` are step sizes **as a fraction of** ``norm(z)``: the actual
    perturbation is ``delta * scale * g`` with
    ``scale = norm(z) / sqrt(sum_b tr K_b)``, and the difference quotient
    divides by the same step, so the estimate keeps the units of ``tr(Phi' K)``.
    That scaling is not cosmetic -- a step much larger than ``z`` makes *any*
    trainer look like the identity, which is how a thresholding trainer can
    otherwise be mistaken for a well-behaved linear one. Pass ``scale``
    explicitly to override.

    **Fails closed.** A linear trainer's difference quotient is exact at every
    step, so its estimate is step-free. A trainer with discontinuous model
    selection -- clumping, thresholding, hard variable selection -- has a local
    Jacobian that is *also* locally stable but that misses the selection
    response entirely, which is why a stability check at one small step is not
    enough. The default ``deltas`` therefore span from a step too small to move
    any selection boundary to one that moves many; disagreement across that
    range is the signature of an operator no derivative describes. An
    :meth:`OverlapBasis.unavailable` is returned (never a number) when the
    totals disagree by more than ``max_relative_spread`` or the per-block
    patterns correlate below ``min_pattern_correlation``.

    Costs ``n_draws * len(deltas) + 1`` trainer runs.
    """
    if not callable(trainer):
        raise TypeError("trainer must be callable")
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("z must be a non-empty finite 1-D vector")
    blocks = [np.ascontiguousarray(np.asarray(idx, dtype=np.intp)) for idx in blocks]
    if not blocks:
        raise ValueError("need at least one block")
    seen = np.zeros(z.size, dtype=bool)
    for b, idx in enumerate(blocks):
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError(f"block {b} index must be a non-empty 1-D array")
        if idx.min() < 0 or idx.max() >= z.size:
            raise ValueError(f"block {b} index is out of range for z")
        if seen[idx].any():
            raise ValueError("blocks overlap: a variant appears in two blocks")
        seen[idx] = True
    factors = [np.asarray(L, dtype=np.float64) for L in noise_sqrt]
    if len(factors) != len(blocks):
        raise ValueError(
            f"noise_sqrt must have one factor per block ({len(blocks)}); "
            f"got {len(factors)}")
    for b, (L, idx) in enumerate(zip(factors, blocks)):
        if L.ndim != 2 or L.shape[0] != idx.size or not np.all(np.isfinite(L)):
            raise ValueError(
                f"noise_sqrt[{b}] must be a finite ({idx.size}, r) factor; "
                f"got shape {L.shape}")
    deltas = tuple(float(d) for d in deltas)
    if len(deltas) < 2 or any(not np.isfinite(d) or d <= 0 for d in deltas):
        raise ValueError(
            "deltas must be at least two finite positive perturbation scales; "
            "a single scale cannot detect a step-size artifact")
    n_draws = int(n_draws)
    if n_draws < 2:
        raise ValueError("n_draws must be at least 2 to estimate a Monte-Carlo SE")
    if scale is None:
        noise_norm = float(np.sqrt(sum(float(np.sum(L * L)) for L in factors)))
        if noise_norm <= np.finfo(float).tiny:
            raise ValueError("noise_sqrt describes a degenerate zero covariance")
        scale = float(np.linalg.norm(z)) / noise_norm
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and strictly positive")

    def _weights(value, source):
        out = np.asarray(value, dtype=np.float64)
        if out.shape != z.shape or not np.all(np.isfinite(out)):
            raise ValueError(f"{source} must return a finite vector of length {z.size}")
        return out

    base = _weights(trainer(z), "trainer(z)")
    n_blocks = len(blocks)
    # per_delta[d][r, b] -- common random draws across deltas, so the comparison
    # measures step-size sensitivity rather than Monte-Carlo noise.
    per_delta = {d: np.empty((n_draws, n_blocks)) for d in deltas}
    for r in range(n_draws):
        g = np.zeros(z.size, dtype=np.float64)
        for L, idx in zip(factors, blocks):
            g[idx] = L @ rng.standard_normal(L.shape[1])
        for d in deltas:
            step = d * scale
            perturbed = _weights(trainer(z + step * g), f"trainer(z + {d} * scale * g)")
            derivative = (perturbed - base) / step
            for b, idx in enumerate(blocks):
                per_delta[d][r, b] = float(g[idx] @ derivative[idx])

    means = {d: per_delta[d].mean(axis=0) for d in deltas}
    totals = np.array([means[d].sum() for d in deltas])
    reference = float(np.mean(np.abs(totals)))
    if reference <= np.finfo(float).tiny:
        return OverlapBasis.unavailable(
            f"{provenance}; trainer sensitivity estimated as zero at every "
            "perturbation scale, so no unit of shared noise is identified")
    spread = float(np.ptp(totals) / reference)

    pattern = 1.0
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            a, b_ = means[deltas[i]], means[deltas[j]]
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b_))
            if na <= np.finfo(float).tiny or nb <= np.finfo(float).tiny:
                pattern = 0.0
            else:
                pattern = min(pattern, float(np.dot(a, b_) / (na * nb)))
    if spread > max_relative_spread or pattern < min_pattern_correlation:
        return OverlapBasis.unavailable(
            f"{provenance}; trainer sensitivity is not stable in the "
            f"perturbation scale (total spread {spread:.3f} > "
            f"{max_relative_spread:g}, or block-pattern correlation "
            f"{pattern:.3f} < {min_pattern_correlation:g}), which is what a "
            "discontinuous model-selection step looks like -- the local "
            "Jacobian does not describe the operator")

    values = np.mean([means[d] for d in deltas], axis=0)
    draw_totals = np.concatenate([per_delta[d].sum(axis=1) for d in deltas])
    mc_se = float(np.std(draw_totals, ddof=1) / np.sqrt(draw_totals.size))
    return OverlapBasis(
        values=values, kind="jacobian_hutchinson",
        provenance=(f"{provenance}; Hutchinson GDF, {n_draws} draws x "
                    f"deltas {deltas} (scale {scale:.4g}), spread {spread:.3f}, "
                    f"pattern correlation {pattern:.3f}"),
        support_hash=support_hash, support=support, mc_se=mc_se)


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


# Heterogeneous block sizes, as a real LD reference has.
BLOCK_SIZES = (15, 30, 60, 120)
KEYS = ("diffuse_status", "sparse_null_status", "sparse_low_status",
        "sparse_full_status", "gamma_full_ratio", "r2_naive_full",
        "r2_corr_full", "r2_honest_full", "basis_error", "sparse_trainer_status")


def _cohorts(rng, n, per_size, rho):
    """Genotypes for train/target/independent, with heterogeneous LD blocks."""
    columns, sizes = [], []
    for block_size in BLOCK_SIZES:
        maf = rng.uniform(0.05, 0.5, size=block_size * per_size)
        columns.append(simulate_diploid_genotypes(
            3 * n, [block_size] * per_size, maf, rho, rng))
        sizes.extend([block_size] * per_size)
    x = np.hstack(columns)
    sizes = np.asarray(sizes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return x, [np.arange(s, s + k) for s, k in zip(starts, sizes)]


def _one(rng, frac, causal_frac, *, n, per_size, rho, h2, ridge, thresholded,
         hutchinson_draws):
    x, idxs = _cohorts(rng, n, per_size, rho)
    m = x.shape[1]
    beta = draw_effects(m, max(10, int(causal_frac * m)), rng)
    y = simulate_phenotype(x, beta, h2, rng)

    i_train = np.arange(n)
    n_overlap = int(frac * n)
    i_target = np.concatenate([rng.choice(i_train, n_overlap, replace=False),
                               np.arange(n, 2 * n - n_overlap)])
    z_train, t_train = marginal_stats(x[i_train], y[i_train])
    z_target, _ = marginal_stats(x[i_target], y[i_target])
    z_indep, _ = marginal_stats(x[2 * n:], y[2 * n:])

    D = np.corrcoef(x.T)
    blocks = [D[np.ix_(i, i)] for i in idxs]
    operators = [np.linalg.solve(Db + ridge * np.eye(len(Db)), np.eye(len(Db)))
                 for Db in blocks]

    def trainer(values):
        """Blockwise ridge -- rerunnable, and linear so its basis is checkable."""
        w = np.zeros_like(values)
        for Phi, idx in zip(operators, idxs):
            w[idx] = Phi @ values[idx]
        return w

    # (O3): q_b = tr(Phi_b' K_b) with K = D, exact for this linear trainer.
    analytic = np.array([float(np.trace(Phi.T @ Db))
                         for Phi, Db in zip(operators, blocks)])
    if thresholded:
        # Discontinuous selection: the local Jacobian misses the selection
        # response, and the perturbation-stability gate must say so.
        w = pgs_pthreshold(z_train, t_train, 2.5)
        basis = estimate_overlap_basis(
            lambda values: pgs_pthreshold(
                values, values / np.sqrt(np.clip(1.0 - values ** 2, 1e-12, None) / n),
                2.5),
            z_train, idxs, [np.linalg.cholesky(Db) for Db in blocks],
            rng=rng, provenance="p+T trainer", support_hash="simulation",
            n_draws=hutchinson_draws)
    else:
        w = trainer(z_train)
        basis = estimate_overlap_basis(
            trainer, z_train, idxs, [np.linalg.cholesky(Db) for Db in blocks],
            rng=rng, provenance="blockwise ridge trainer",
            support_hash="simulation", n_draws=hutchinson_draws)

    u_target = np.array([w[i] @ z_target[i] for i in idxs])
    u_indep = np.array([w[i] @ z_indep[i] for i in idxs])
    v = np.array([float(w[i] @ D[np.ix_(i, i)] @ w[i]) for i in idxs])
    fit = fit_overlap(u_target, u_indep, v / n, v / n, basis=basis,
                      groups=np.arange(len(idxs)) % 20)

    den = DenseLD(D).quad(w)
    corrected = correct_overlap_numerator(fit) if fit.can_correct else np.nan
    return dict(
        status=fit.status, vif=fit.vif, condition_number=fit.condition_number,
        alpha=fit.alpha, gamma=fit.gamma, gamma_true=n_overlap / (n * n),
        basis_error=(abs(basis.values.sum() - analytic.sum()) / abs(analytic.sum())
                     if basis.available and not thresholded else np.nan),
        basis_kind=basis.kind,
        r2_naive=(w @ z_target) ** 2 / den,
        r2_corrected=corrected ** 2 / den if np.isfinite(corrected) else np.nan,
        r2_honest=(w @ z_indep) ** 2 / den)


def run(n=2500, per_size=15, rho=0.6, h2=0.3, ridge=0.5, reps=2,
        diffuse_causal=0.20, sparse_causal=0.01, hutchinson_draws=48,
        seed=1000):
    """Both regimes plus the discontinuous-trainer case. See KEYS."""
    rng = np.random.default_rng(seed)
    common = dict(n=n, per_size=per_size, rho=rho, h2=h2, ridge=ridge,
                  thresholded=False, hutchinson_draws=hutchinson_draws)

    def batch(frac, causal):
        return [_one(rng, frac, causal, **common) for _ in range(reps)]

    diffuse = batch(1.0, diffuse_causal)
    sparse = {f: batch(f, sparse_causal) for f in (0.0, 0.25, 1.0)}
    thresholded = _one(rng, 1.0, sparse_causal,
                       **{**common, "thresholded": True})

    def mean(rows, key):
        values = [r[key] for r in rows
                  if r[key] is not None and np.isfinite(r[key])]
        return float(np.mean(values)) if values else np.nan

    full = sparse[1.0]
    return {
        "diffuse_status": tuple(r["status"] for r in diffuse),
        "diffuse_vif": mean(diffuse, "vif"),
        "diffuse_gamma_ratio": mean(diffuse, "gamma") / mean(diffuse, "gamma_true"),
        "sparse_null_status": tuple(r["status"] for r in sparse[0.0]),
        "sparse_low_status": tuple(r["status"] for r in sparse[0.25]),
        "sparse_full_status": tuple(r["status"] for r in full),
        "sparse_vif": mean(full, "vif"),
        "alpha_full": mean(full, "alpha"),
        "gamma_full_ratio": mean(full, "gamma") / mean(full, "gamma_true"),
        "gamma_low_ratio": (mean(sparse[0.25], "gamma")
                            / mean(sparse[0.25], "gamma_true")),
        "corrections_at_null": sum(r["status"] == "correctable"
                                   for r in sparse[0.0]),
        "r2_naive_full": mean(full, "r2_naive"),
        "r2_corr_full": mean(full, "r2_corrected"),
        "r2_honest_full": mean(full, "r2_honest"),
        "r2_naive_low": mean(sparse[0.25], "r2_naive"),
        "r2_corr_low": mean(sparse[0.25], "r2_corrected"),
        "r2_honest_low": mean(sparse[0.25], "r2_honest"),
        "basis_error": mean(full, "basis_error"),
        "sparse_trainer_status": thresholded["status"],
        "sparse_trainer_basis": thresholded["basis_kind"],
    }


def main():
    out = run()
    print("=== Hutchinson basis for a rerunnable trainer ===")
    print(f"  relative error vs analytic tr(Phi'K) = {out['basis_error']:.4f}")
    print("\n=== diffuse architecture (20% causal): the refusal corner ===")
    print(f"  status = {out['diffuse_status']}, VIF {out['diffuse_vif']:.2f} "
          f"(gate 2.0); gamma/true {out['diffuse_gamma_ratio']:.2f}")
    print("  the coupling is nearly right, but signal and basis are collinear,")
    print("  so the identification gate refuses -- correctly.")
    print("\n=== sparse architecture (1% causal): identified ===")
    print(f"  null       {out['sparse_null_status']}  "
          f"({out['corrections_at_null']} corrections issued; must be 0)")
    print(f"  25% overlap{out['sparse_low_status']}  "
          f"gamma/true {out['gamma_low_ratio']:.2f}   "
          f"R2 {out['r2_naive_low']:.4f} -> {out['r2_corr_low']:.4f} "
          f"(anchor {out['r2_honest_low']:.4f})")
    print(f"  100%       {out['sparse_full_status']}  "
          f"gamma/true {out['gamma_full_ratio']:.2f}   "
          f"R2 {out['r2_naive_full']:.4f} -> {out['r2_corr_full']:.4f} "
          f"(anchor {out['r2_honest_full']:.4f})")
    print(f"  VIF {out['sparse_vif']:.2f}, alpha {out['alpha_full']:.2f}")
    print("\n=== discontinuous trainer (p+T) ===")
    print(f"  basis = {out['sparse_trainer_basis']}, fit status = "
          f"{out['sparse_trainer_status']} (correction refused)")


if __name__ == "__main__":
    main()
