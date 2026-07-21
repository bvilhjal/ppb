"""Detecting (and approximately correcting) training/target sample overlap.

The summary-statistic accuracy estimate ``R^2 = (w^T z)^2 / (w^T D w)`` assumes
the PGS weights ``w`` were trained independently of the target GWAS's
estimation noise. When the score's training data overlaps the target sample,
``w`` partially fits the shared noise and the numerator is inflated by a
first-moment, additive term::

    E[w^T z_hat] = w^T zeta  (true signal)  +  E[w^T eps]  (overlap term)

For a linear-in-z trainer (Z-score / marginal weights, LDpred2-inf; LDpred2
with a large causal fraction approximately so), the overlap term is a
*uniform per-variant* quantity::

    E[w^T eps] = (N_ov / (n1 n2)) * rho_eps * tr(A D) ~= gamma * m

with ``A`` the trainer's effective operator (``A = I`` for marginal weights) and
``gamma`` the per-variant overlap coefficient; for ``A = I`` exactly
``gamma = N_ov / (n1 n2)``. Estimation noise carries no LD structure, so this
term is flat across the genome while true signal concentrates in LD. That
asymmetry makes overlap identifiable from ``(w, z, D)`` alone -- no training
summary statistics needed.

Dual-target detector (the design validated here): given the same score
evaluated against a *suspect* target ``z_t`` and a *reference* target ``z_r``
of the same trait (genetic correlation ~ 1, no overlap), the per-block
difference ``u_b(z_t) - u_b(z_r)`` removes the signal, leaving the overlap
term, which is recovered by a weighted through-origin regression on the block
sizes ``m_b``. Subtracting ``gamma * m`` from the suspect numerator yields an
approximate de-overlapped R^2.

**Sparse-score failure mode.** With strong nonlinear shrinkage (p-value
thresholding, LDpred2-auto with small p), the noise fit is projected onto the
same variants that carry signal; the uniform term vanishes and overlap becomes
unidentifiable by shape alone. Such evaluations must be flagged as upper
bounds rather than corrected. See ``docs/OVERLAP.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ld_backend import BlockDiagonalLD


def block_products(w, z, ld: BlockDiagonalLD):
    """Per-LD-block ``u_b = w_b' z_b`` and ``v_b = w_b' D_b w_b``.

    ``w`` and ``z`` are aligned vectors over the variants of ``ld`` (a
    :class:`BlockDiagonalLD`). Returns ``(u, v)`` arrays of length ``n_blocks``.
    """
    w = np.asarray(w, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if w.shape != z.shape or w.shape != (ld.m,):
        raise ValueError(
            f"w and z must have shape ({ld.m},); got {w.shape} and {z.shape}")
    u = np.array([float(w[idx] @ z[idx]) for _, idx in ld.blocks])
    v = np.array([backend.quad(w[idx]) for backend, idx in ld.blocks])
    return u, v


@dataclass
class OverlapEstimate:
    """Dual-target overlap fit: per-variant coefficient ``gamma`` with a
    delete-group jackknife standard error, its z-score, and the block count."""

    gamma: float
    se: float
    z: float
    n_blocks: int


def overlap_slope(u_target, u_reference, m, v_target, v_reference, *,
                  groups=None) -> OverlapEstimate:
    """Weighted through-origin fit of ``u_target - u_reference`` on block sizes.

    ``m`` are the per-block variant counts, ``v_*`` the per-block score
    variances (from :func:`block_products`). Weights are ``1 / (v_t + v_r)``,
    the inverse noise variances of the block differences (up to the constant
    1/N). ``groups`` are leave-one-out jackknife groups (default: ~20
    contiguous block groups); chromosomes are the natural choice on real data.
    """
    du = np.asarray(u_target, dtype=np.float64) - np.asarray(u_reference, dtype=np.float64)
    dv = np.asarray(v_target, dtype=np.float64) + np.asarray(v_reference, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    ok = dv > 0
    if ok.sum() < 4:
        raise ValueError("need at least 4 blocks with positive score variance")
    du, dv, m = du[ok], dv[ok], m[ok]
    wt = 1.0 / dv
    if groups is None:
        groups = np.repeat(np.arange(20), np.ceil(len(du) / 20))[:len(du)]
    groups = np.asarray(groups)[ok]
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError("need at least 2 jackknife groups")

    def fit(keep):
        w_ = wt[keep]
        return float((w_ * m[keep] * du[keep]).sum() / (w_ * m[keep] * m[keep]).sum())

    gamma = fit(np.ones(len(m), bool))
    ests = np.array([fit(groups != g) for g in uniq])
    k = len(ests)
    se = float(np.sqrt((k - 1) / k * ((ests - ests.mean()) ** 2).sum()))
    return OverlapEstimate(gamma=gamma, se=se,
                           z=gamma / se if se > 0 else np.inf, n_blocks=int(ok.sum()))


def correct_numerator(num, gamma, m_total):
    """De-overlapped numerator ``w^T z - gamma * m_total`` (dense-score case).

    ``gamma`` from :func:`overlap_slope`, ``m_total`` the total number of
    variants. The corrected R^2 is ``correct_numerator(...)^2 / w^T D w``.
    See the module docstring for the sparse-score failure mode, where this
    correction does not apply.
    """
    return float(num - gamma * m_total)
