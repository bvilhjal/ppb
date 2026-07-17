"""Simulation harness for the LD-reference experiment (Supplementary Fig. S1).

Reproduces the qualitative finding of Witteveen et al.: the summary-statistic
prediction ``R^2`` estimated with

    R^2 = (w^T z_target)^2 / (w^T D_ref w)

is (a) exact when ``D_ref`` is the test-set LD, (b) biased when ``D_ref`` is the
training-set LD, and (c) unbiased when ``D_ref`` is an independent LD reference
from the same population. ``z_target = (1/n) X_test^T y_test`` are the released
target summary statistics.

The genotype model here is intentionally simple (standardized Gaussian variants
with a block-AR(1) population LD) -- enough to exhibit the LD-reference behavior
without individual-level data or a real cohort.
"""

from __future__ import annotations

import numpy as np


def population_ld(m: int, block_size: int = 30, rho: float = 0.6) -> np.ndarray:
    """Block-diagonal AR(1) population correlation ``Sigma`` (unit diagonal)."""
    if m < 1 or block_size < 1 or not 0.0 <= rho < 1.0:
        raise ValueError("require m>=1, block_size>=1, rho in [0, 1)")
    Sigma = np.zeros((m, m))
    start = 0
    while start < m:
        k = min(block_size, m - start)
        d = np.arange(k)
        Sigma[start:start + k, start:start + k] = rho ** np.abs(d[:, None] - d[None, :])
        start += k
    return Sigma


def sample_genotypes(Sigma: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n`` standardized variants from ``N(0, Sigma)`` (columns mean 0, var 1)."""
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n, Sigma.shape[0])) @ L.T
    X = X - X.mean(axis=0)
    return X / X.std(axis=0)


def draw_effects(m: int, n_causal: int, rng: np.random.Generator) -> np.ndarray:
    """Sparse standard-normal causal effect vector with ``n_causal`` nonzeros."""
    beta = np.zeros(m)
    idx = rng.choice(m, size=min(n_causal, m), replace=False)
    beta[idx] = rng.standard_normal(idx.size)
    return beta


def simulate_phenotype(X, beta, h2, rng) -> np.ndarray:
    """Standardized phenotype with genetic variance ``h2`` from predictor ``X @ beta``."""
    if not 0.0 <= h2 <= 1.0:
        raise ValueError("h2 must be in [0, 1]")
    g = X @ beta
    sd = g.std()
    g = g / sd if sd > 0 else g
    y = np.sqrt(h2) * g + np.sqrt(1.0 - h2) * rng.standard_normal(X.shape[0])
    y = y - y.mean()
    return y / y.std()


def marginal_stats(X, y):
    """Marginal GWAS: per-variant ``z = (1/n) X^T y`` and its t-statistic.

    For standardized ``X`` and ``y`` the marginal least-squares slope equals ``z``
    and its standard error is ``sqrt((1 - z^2) / n)``.
    """
    n = X.shape[0]
    z = (X.T @ y) / n
    se = np.sqrt(np.clip(1.0 - z ** 2, 1e-12, None) / n)
    return z, z / se


def pgs_pthreshold(z, t, t_crit: float = 2.5) -> np.ndarray:
    """p-value-thresholded marginal weights: keep ``z_j`` where ``|t_j| >= t_crit``."""
    return np.where(np.abs(t) >= t_crit, z, 0.0)
