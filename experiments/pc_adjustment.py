"""PC adjustment removes population-structure confounding in PPB evaluation.

Two subpopulations (Balding-Nichols fst) create population structure. A
phenotype driven only by ancestry (no true genetic effect) is spuriously
"predicted" by a polygenic score built on the structured data -- classic
stratification. Residualizing genotypes and phenotype on the top principal
components before forming the target summary statistics ``z`` and LD ``D``
removes the confounding, so the estimated R^2 drops to ~0. Under a genuine
genetic signal with no confounding, adjustment is nearly harmless.

The estimator is unchanged; it just consumes the PC-adjusted ``z`` and ``D``.
Averaged over replicates, because a single 2-population ancestry axis is low-rank
and the per-draw spurious R^2 is noisy.

Run:
    python experiments/pc_adjustment.py --n-reps 10
"""

from __future__ import annotations

import argparse

import numpy as np

from ppb import DenseLD, r2
from ppb.simulate import (
    draw_effects,
    marginal_stats,
    simulate_phenotype,
    simulate_structured_genotypes,
)


# ---------------------------------------------------------------------------
# Covariate / principal-component adjustment.
#
# The benchmark residualizes the phenotype (and genotypes) on fixed covariates
# -- sex, age, and the top principal components -- before forming summary
# statistics, so population structure does not confound the estimated
# prediction accuracy.
#
# The estimator itself is unchanged: it consumes the covariate-adjusted ``z``
# and ``D``. This mirrors the covariate projection ``X~ = P_C X`` in the local
# `pldsc` project (``pldsc.ld.residualize_genotypes``), i.e. least-squares
# projection of an intercept plus the covariates out of each column.
# ---------------------------------------------------------------------------


def _design(n: int, covariates):
    """Design matrix: an intercept column plus any covariates."""
    intercept = np.ones((n, 1), dtype=np.float64)
    if covariates is None:
        return intercept
    C = np.asarray(covariates, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    if C.ndim != 2 or C.shape[0] != n:
        raise ValueError(f"covariates must have shape ({n}, k); got {C.shape}")
    if not np.isfinite(C).all():
        raise ValueError("covariates must be finite")
    return np.column_stack([intercept, C])


def residualize(M, covariates=None):
    """Project an intercept + ``covariates`` out of ``M`` (columns of a 2-D array,
    or a 1-D vector). Returns residuals with the same shape (mirrors
    ``pldsc.ld.residualize_genotypes``)."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim not in (1, 2):
        raise ValueError(f"M must be 1-D or 2-D; got shape {M.shape}")
    if not np.isfinite(M).all():
        raise ValueError("M must contain only finite values")
    twod = M.ndim == 2
    X = M if twod else M[:, None]
    design = _design(X.shape[0], covariates)
    coef, _, _, _ = np.linalg.lstsq(design, X, rcond=None)
    R = X - design @ coef
    return R if twod else R[:, 0]


def _standardize_cols(A, *, reference=None, name="input"):
    A = np.asarray(A, dtype=np.float64)
    if not np.isfinite(A).all():
        raise ValueError(f"{name} must contain only finite values")
    A = A - A.mean(axis=0)
    sd = A.std(axis=0)
    ref = np.asarray(reference if reference is not None else A, dtype=np.float64)
    ref_centered = ref - ref.mean(axis=0)
    ref_sd = ref_centered.std(axis=0)
    ref_rms = np.sqrt(np.mean(ref * ref, axis=0))
    rel_tol = np.sqrt(np.finfo(np.float64).eps)
    if np.any(ref_sd <= rel_tol * np.maximum(ref_rms, np.finfo(np.float64).tiny)):
        raise ValueError(f"{name} has a constant or numerically constant column")
    if np.any(sd <= rel_tol * ref_sd):
        raise ValueError(
            f"{name} has a near-zero residual-variance column after adjustment")
    return A / sd


def adjust(X, y, covariates=None):
    """Residualize genotypes ``X`` and phenotype ``y`` on the covariates, then
    re-standardize. Returns ``(X_adj, y_adj)`` ready to form adjusted ``z`` and ``D``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")
    if y.shape != (X.shape[0],):
        raise ValueError(f"y must have shape ({X.shape[0]},); got {y.shape}")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values")
    X_adj = _standardize_cols(
        residualize(X, covariates), reference=X, name="genotype residuals")
    yr = residualize(y, covariates)
    y_adj = _standardize_cols(
        yr, reference=y, name="phenotype residuals")
    return X_adj, y_adj


def principal_components(X, k):
    """Top-``k`` sample principal components of a genotype matrix (n x m).

    The left singular vectors of the standardized genotypes -- eigenvectors of the
    sample GRM. For structured samples the leading PCs are the ancestry axes.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    Xs = _standardize_cols(X)
    if k > min(Xs.shape):
        raise ValueError(
            f"k={k} exceeds the available components min(n, m)={min(Xs.shape)}")
    U, _, _ = np.linalg.svd(Xs, full_matrices=False)
    return np.ascontiguousarray(U[:, :k])


def _block_sizes(m, block_size):
    sizes, rem = [], m
    while rem > 0:
        sizes.append(min(block_size, rem)); rem -= block_size
    return sizes


def _ppb_r2(X, y, w, n_pcs):
    """PPB R^2 without and with PC adjustment (test-set LD, so exact)."""
    n = X.shape[0]
    unadj = r2(w, (X.T @ y) / n, DenseLD((X.T @ X) / n))
    Xa, ya = adjust(X, y, principal_components(X, n_pcs))
    adj = r2(w, (Xa.T @ ya) / n, DenseLD((Xa.T @ Xa) / n))
    return unadj, adj


def run(m=400, block_size=40, fst=0.3, rho=0.5, n=3000, n_causal=40,
        confound=4.0, n_pcs=2, n_reps=10, seed=0):
    rng = np.random.default_rng(seed)
    bs = _block_sizes(m, block_size)
    rows = {"null+confound": ([], []), "genetic,no confound": ([], [])}

    for _ in range(n_reps):
        Xtr, ltr = simulate_structured_genotypes(n, bs, fst, rho, rng)
        Xte, lte = simulate_structured_genotypes(n, bs, fst, rho, rng)

        # (A) null genetic signal + ancestry confounder -> spurious prediction.
        def confounded(X, labels):
            y = rng.standard_normal(X.shape[0]) + confound * (labels - labels.mean())
            return (y - y.mean()) / y.std()
        w = marginal_stats(Xtr, confounded(Xtr, ltr))[0]
        u, a = _ppb_r2(Xte, confounded(Xte, lte), w, n_pcs)
        rows["null+confound"][0].append(u); rows["null+confound"][1].append(a)

        # (B) true genetic signal, no confounding -> adjustment ~harmless.
        beta = draw_effects(m, n_causal, rng)
        w2 = marginal_stats(Xtr, simulate_phenotype(Xtr, beta, 0.5, rng))[0]
        u2, a2 = _ppb_r2(Xte, simulate_phenotype(Xte, beta, 0.5, rng), w2, n_pcs)
        rows["genetic,no confound"][0].append(u2); rows["genetic,no confound"][1].append(a2)

    return {name: (float(np.mean(u)), float(np.mean(a))) for name, (u, a) in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = run(n_reps=args.n_reps, seed=args.seed)
    print(f"{'scenario':<22}{'R2 unadjusted':>16}{'R2 PC-adjusted':>16}")
    print("-" * 54)
    for name, (unadj, adj) in res.items():
        print(f"{name:<22}{unadj:>16.4f}{adj:>16.4f}")


if __name__ == "__main__":
    main()
