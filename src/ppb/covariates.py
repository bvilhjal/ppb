"""Covariate / principal-component adjustment.

The benchmark residualizes the phenotype (and genotypes) on fixed covariates --
sex, age, and the top principal components -- before forming summary statistics,
so population structure does not confound the estimated prediction accuracy.

The estimator itself is unchanged: it consumes the covariate-adjusted ``z`` and
``D``. This mirrors the covariate projection ``X~ = P_C X`` in the local `pldsc`
project (``pldsc.ld.residualize_genotypes``), i.e. least-squares projection of an
intercept plus the covariates out of each column.
"""

from __future__ import annotations

import numpy as np


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
    twod = M.ndim == 2
    X = M if twod else M[:, None]
    design = _design(X.shape[0], covariates)
    coef, _, _, _ = np.linalg.lstsq(design, X, rcond=None)
    R = X - design @ coef
    return R if twod else R[:, 0]


def _standardize_cols(A):
    A = np.asarray(A, dtype=np.float64)
    A = A - A.mean(axis=0)
    sd = A.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return A / sd


def adjust(X, y, covariates=None):
    """Residualize genotypes ``X`` and phenotype ``y`` on the covariates, then
    re-standardize. Returns ``(X_adj, y_adj)`` ready to form adjusted ``z`` and ``D``.
    """
    X_adj = _standardize_cols(residualize(X, covariates))
    yr = residualize(np.asarray(y, dtype=np.float64), covariates)
    sd = yr.std()
    y_adj = (yr - yr.mean()) / (sd if sd > 0 else 1.0)
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
