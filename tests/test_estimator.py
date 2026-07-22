"""Equation-level tests: the summary-statistic estimators must equal the
individual-level quantities exactly (up to floating point) when the full LD
matrix and identical standardization are used. This is validation criterion #1
in the completion plan.
"""

import numpy as np
import pytest

from ppb import DenseLD, LowRankLD, mse, r2

TOL = 1e-10


def _standardize(A, axis=0):
    """Center and scale to population (ddof=0) unit variance along ``axis``."""
    A = np.asarray(A, dtype=np.float64)
    A = A - A.mean(axis=axis, keepdims=True)
    sd = np.sqrt((A ** 2).mean(axis=axis, keepdims=True))
    return A / sd


def _simulate(n=3000, m=40, seed=0):
    """A small LD-correlated, standardized (X, y) with arbitrary PGS weights w.

    Returns X, y, w, D=(1/n)X^T X, z=(1/n)X^T y.
    """
    rng = np.random.default_rng(seed)
    # Correlated genotypes via a low-rank factor model + noise, then standardize.
    load = rng.standard_normal((m, 6))
    latent = rng.standard_normal((n, 6))
    Z = latent @ load.T + 0.4 * rng.standard_normal((n, m))
    X = _standardize(Z, axis=0)                       # columns: mean 0, var 1
    beta = rng.standard_normal(m) * (rng.random(m) < 0.3)
    y = X @ beta + 2.0 * rng.standard_normal(n)
    y = _standardize(y, axis=0).ravel()               # mean 0, var 1
    w = rng.standard_normal(m)                         # arbitrary weights to score
    D = (X.T @ X) / n
    z = (X.T @ y) / n
    return X, y, w, D, z


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_r2_equals_individual_level_with_full_D(seed):
    X, y, w, D, z = _simulate(seed=seed)
    est = r2(w, z, DenseLD(D))
    ref = float(np.corrcoef(X @ w, y)[0, 1] ** 2)     # true predictive R^2
    assert abs(est - ref) <= TOL * max(1.0, ref)


@pytest.mark.parametrize("seed", [0, 3])
def test_mse_equals_individual_level_with_full_D(seed):
    X, y, w, D, z = _simulate(seed=seed)
    est = mse(w, z, DenseLD(D), var_y=1.0)
    ref = float(np.mean((y - X @ w) ** 2))
    assert abs(est - ref) <= TOL * max(1.0, ref)


def test_lowrank_factor_reproduces_dense_quadratic_form():
    """A full-rank factor U (U U^T = D) must give the same w^T D w and R^2."""
    X, y, w, D, z = _simulate(seed=1)
    evals, evecs = np.linalg.eigh(D)                  # D is symmetric PSD
    U = evecs * np.sqrt(np.clip(evals, 0.0, None))    # (m, m), U U^T = D
    dense, low = DenseLD(D), LowRankLD(U)
    assert abs(low.quad(w) - dense.quad(w)) <= TOL * abs(dense.quad(w))
    assert abs(r2(w, z, low) - r2(w, z, dense)) <= TOL


def test_lowrank_is_nonnegative_even_when_truncated():
    """Truncating the factor stays PSD: quad >= 0 (no negative denominator)."""
    _, _, w, D, _ = _simulate(seed=4)
    evals, evecs = np.linalg.eigh(D)
    order = np.argsort(evals)[::-1]
    U = (evecs[:, order] * np.sqrt(np.clip(evals[order], 0.0, None)))[:, :5]
    assert LowRankLD(U).quad(w) >= 0.0


def test_nonpositive_denominator_raises():
    with pytest.raises(ValueError):
        r2(np.zeros(3), np.zeros(3), DenseLD(np.eye(3)))
    with pytest.raises(ValueError):
        r2([1.0, 2.0], [1.0], DenseLD(np.eye(2)))     # length mismatch


def test_mse_rejects_a_negative_quadratic_form_but_allows_zero():
    """A negative w'Dw (non-PSD LD) would silently understate the error. Zero is
    fine though: all-zero weights predict nothing and MSE = var_y is correct,
    where r2 is genuinely undefined."""
    class _Neg:
        m = 3
        def quad(self, w):
            return -1.0

    class _Zero:
        m = 3
        def quad(self, w):
            return 0.0

    z = np.zeros(3)
    with pytest.raises(ValueError, match="negative"):
        mse(np.ones(3), z, _Neg())
    assert mse(np.zeros(3), z, _Zero(), var_y=2.0) == pytest.approx(2.0)


def test_nonunit_phenotype_variance_is_coherent_in_r2_and_mse():
    """y=2x has var(y)=4, z=cov(x,y)=2 and the perfect score has w=2."""
    ld = DenseLD(np.eye(1))
    assert r2([2.0], [2.0], ld, var_y=4.0) == pytest.approx(1.0)
    assert mse([2.0], [2.0], ld, var_y=4.0) == pytest.approx(0.0)


@pytest.mark.parametrize("var_y", [0.0, -1.0, np.nan, np.inf])
def test_estimators_reject_invalid_phenotype_variance(var_y):
    ld = DenseLD(np.eye(1))
    with pytest.raises(ValueError, match="var_y"):
        r2([1.0], [0.1], ld, var_y=var_y)
    with pytest.raises(ValueError, match="var_y"):
        mse([1.0], [0.1], ld, var_y=var_y)


@pytest.mark.parametrize("weights,z", [([np.nan], [0.1]), ([1.0], [np.inf])])
def test_estimators_reject_nonfinite_vectors(weights, z):
    with pytest.raises(ValueError, match="finite"):
        r2(weights, z, DenseLD(np.eye(1)))
