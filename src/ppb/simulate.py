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


# Acklam's rational approximation of the standard-normal quantile (inverse CDF),
# accurate to ~1e-9 -- so the simulator needs no scipy.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)


def _norm_ppf(p):
    """Standard-normal quantile Phi^{-1}(p), vectorized, no scipy (Acklam)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    x = np.zeros_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    x[lo] = ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
             / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1))
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    x[hi] = -((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
              / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1))
    q = p[mid] - 0.5
    r = q * q
    x[mid] = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q \
        / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)
    return x


def _diploid_dosages(n, block_sizes, maf, rho, rng):
    """Raw 0/1/2 dosages (n x m) with block AR(1) LD and per-variant ``maf``."""
    block_sizes = list(block_sizes)
    m = int(sum(block_sizes))
    maf = np.asarray(maf, dtype=np.float64)
    if maf.shape != (m,):
        raise ValueError(f"maf must have length m={m}; got {maf.shape}")
    thr = _norm_ppf(1.0 - maf)                        # P(z > thr) = maf per variant
    G = np.zeros((n, m), dtype=np.float64)
    col = 0
    for k in block_sizes:
        d = np.arange(k)
        L = np.linalg.cholesky(rho ** np.abs(d[:, None] - d[None, :]) + 1e-8 * np.eye(k))
        block_thr = thr[col:col + k]
        hap = np.zeros((n, k))
        for _ in range(2):                            # two haplotypes -> 0/1/2
            hap += (rng.standard_normal((n, k)) @ L.T > block_thr)
        G[:, col:col + k] = hap
        col += k
    return G


def _standardize_cols(G):
    sd = G.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (G - G.mean(axis=0)) / sd


def simulate_diploid_genotypes(n, block_sizes, maf, rho, rng):
    """Simulate standardized diploid genotypes (0/1/2 dosages) with block LD.

    ldpred3-inspired latent model: per block, two Gaussian haplotypes with AR(1)
    correlation ``rho`` are thresholded at the MAF-implied quantile and summed,
    giving realistic 0/1/2 dosages with allele frequency ``maf`` and within-block
    LD. Returns the standardized genotype matrix (columns mean 0, variance 1).
    """
    return _standardize_cols(_diploid_dosages(n, block_sizes, maf, rho, rng))


def bn_freqs(rng, m, fst):
    """Balding-Nichols allele frequencies for two populations ``fst`` apart.

    Frequencies are clipped away from 0/1 so every variant stays polymorphic.
    """
    anc = rng.uniform(0.1, 0.9, size=m)
    a = anc * (1 - fst) / fst
    b = (1 - anc) * (1 - fst) / fst
    return (np.clip(rng.beta(a, b), 1e-3, 1 - 1e-3),
            np.clip(rng.beta(a, b), 1e-3, 1 - 1e-3))


def simulate_structured_genotypes(n, block_sizes, fst, rho, rng, prop_pop1=0.5):
    """Two subpopulations (Balding-Nichols ``fst``) with block LD.

    Each individual is assigned to population 0 or 1 and its variants drawn at
    that population's allele frequencies. Because the frequencies differ, the
    leading principal component of the returned genotypes is the ancestry axis --
    a controllable source of population structure. Returns ``(X, labels)`` with
    ``X`` standardized over all individuals and ``labels`` the 0/1 assignment.
    """
    m = int(sum(block_sizes))
    f1, f2 = bn_freqs(rng, m, fst)
    labels = (rng.random(n) < prop_pop1).astype(int)
    G = np.zeros((n, m), dtype=np.float64)
    for pop, freq in ((0, f1), (1, f2)):
        idx = np.where(labels == pop)[0]
        if idx.size:
            G[idx] = _diploid_dosages(idx.size, block_sizes, freq, rho, rng)
    return _standardize_cols(G), labels


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
