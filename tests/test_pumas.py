"""Tests for PUMAS-style subsampling and its agreement with PPB."""

import numpy as np
import pytest

from ppb import DenseLD, pumas_r2, r2, subsample_sumstats

from experiments.pumas_agreement import run  # noqa: E402


def _corr(m, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, m))
    C = A @ A.T + 0.5 * np.eye(m)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def test_subsample_partitions_are_consistent():
    """n_tr*z_tr + n_val*z_val == n_full*z_full exactly (t^(tr)+t^(val)=t)."""
    rng = np.random.default_rng(0)
    D = _corr(20, 1)
    z_full = rng.standard_normal(20) * 0.05
    n_full, n_train = 5000, 3750
    z_tr, z_val = subsample_sumstats(z_full, D, n_full, n_train, rng)
    recomposed = (n_train * z_tr + (n_full - n_train) * z_val) / n_full
    assert np.allclose(recomposed, z_full, atol=1e-10)


def test_subsample_is_unbiased_for_full_stats():
    """Averaged over draws, both partitions' marginal stats recover z_full."""
    rng = np.random.default_rng(2)
    D = _corr(15, 3)
    z_full = rng.standard_normal(15) * 0.05
    chol = np.linalg.cholesky(D + 1e-8 * np.eye(15))
    vals = [subsample_sumstats(z_full, D, 4000, 3000, rng, chol=chol)[1]
            for _ in range(4000)]
    assert np.allclose(np.mean(vals, axis=0), z_full, atol=5e-3)


def test_subsample_validation_covariance_scales_with_D():
    """Cov(z_val) ~ (n_tr / (n n_val)) D on the diagonal."""
    rng = np.random.default_rng(4)
    D = _corr(12, 5)
    z_full = np.zeros(12)
    n, ntr = 4000, 3000
    nval = n - ntr
    chol = np.linalg.cholesky(D + 1e-8 * np.eye(12))
    draws = np.array([subsample_sumstats(z_full, D, n, ntr, rng, chol=chol)[1]
                      for _ in range(6000)])
    expected_diag = (ntr / (n * nval)) * np.diag(D)
    assert np.allclose(draws.var(axis=0), expected_diag, rtol=0.15)


def test_subsample_rejects_bad_split():
    D = _corr(5, 6)
    with pytest.raises(ValueError):
        subsample_sumstats(np.zeros(5), D, 1000, 1000, np.random.default_rng(0))


def test_pumas_agrees_with_ppb_and_individual_level():
    rows = run(n_reps=40, seed=0)
    assert rows
    for arch, method, gold, ppb, pum in rows:
        # PPB with exact target cross-products is algebraically the truth.
        assert abs(ppb - gold) <= 1e-9, f"{arch}/{method}: PPB {ppb} != gold {gold}"
        # PUMAS subsampling recovers it from a single GWAS, no held-out cohort.
        assert abs(pum - gold) <= 0.02, f"{arch}/{method}: PUMAS {pum} vs gold {gold}"
