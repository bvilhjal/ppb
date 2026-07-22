"""Tests for PUMAS-style subsampling and its agreement with PPB."""

import numpy as np
import pytest

from ppb import DenseLD, pumas_r2, r2, subsample_sumstats

from experiments.pumas_agreement import run  # noqa: E402


class _SequenceRng:
    def __init__(self, values):
        self.values = [np.asarray(value, dtype=np.float64) for value in values]
        self.index = 0

    def standard_normal(self, size):
        value = self.values[self.index % len(self.values)]
        self.index += 1
        assert value.shape == (size,)
        return value.copy()


def _corr(m, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, m))
    C = A @ A.T + 0.5 * np.eye(m)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def _isotropic_noises(p):
    """Finite zero-mean sequence with empirical covariance exactly ``I``."""
    eye = np.sqrt(p) * np.eye(p)
    return [row for row in eye] + [-row for row in eye]


def test_subsample_partitions_are_consistent():
    """The complement construction after Equation 2 is exact."""
    rng = np.random.default_rng(0)
    D = _corr(20, 1)
    z_full = rng.standard_normal(20) * 0.05
    n_full, n_train = 5000, 3750
    z_tr, z_val = subsample_sumstats(z_full, D, n_full, n_train, rng)
    recomposed = (n_train * z_tr + (n_full - n_train) * z_val) / n_full
    assert np.allclose(recomposed, z_full, atol=1e-10)


def test_subsample_has_signal_dependent_covariance():
    """The nonzero-``z`` covariance contains ``z z.T`` (Equation 1)."""
    D = np.eye(3)
    z_full = np.array([0.8, 0.6, -0.5])
    var_y = 2.0
    n_full, n_train = 4000, 3000
    n_val = n_full - n_train
    moment = var_y * D + np.outer(z_full, z_full)
    cov_sqrt = np.linalg.cholesky(moment)
    noises = _isotropic_noises(3)
    rng = _SequenceRng(noises)
    draws = np.array([
        subsample_sumstats(
            z_full, D, n_full, n_train, rng, var_y=var_y,
            cov_sqrt=cov_sqrt,
        )[1]
        for _ in noises
    ])
    expected = (n_train / (n_full * n_val)) * moment
    observed = np.cov(draws, rowvar=False, bias=True)
    assert np.allclose(draws.mean(axis=0), z_full, atol=1e-14)
    assert np.allclose(observed, expected, rtol=1e-12, atol=1e-14)
    assert observed[0, 1] > 0.0


def test_subsample_rejects_bad_inputs():
    D = _corr(5, 6)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_train < n_full"):
        subsample_sumstats(np.zeros(5), D, 1000, 1000, rng)
    with pytest.raises(ValueError, match=r"expected \(5, 5\)"):
        subsample_sumstats(np.zeros(5), np.eye(4), 1000, 750, rng)
    with pytest.raises(ValueError, match="finite"):
        subsample_sumstats(np.array([np.nan]), np.eye(1), 1000, 750, rng)
    with pytest.raises(TypeError, match="not bool"):
        subsample_sumstats(np.zeros(1), np.eye(1), True, 1, rng)
    with pytest.raises(ValueError, match="positive semi-definite"):
        subsample_sumstats(
            np.array([10.0, 0.0]),
            np.array([[1.0, 2.0], [2.0, 1.0]]),
            1000,
            750,
            rng,
        )


def test_pumas_fits_each_pseudo_training_split():
    D = np.array([[1.0, 0.2], [0.2, 1.0]])
    z_full = np.array([0.2, -0.1])
    noise = np.array([0.75, -0.5])
    n_full, n_train = 1000, 750
    expected_train, expected_val = subsample_sumstats(
        z_full, D, n_full, n_train, _SequenceRng([noise]))
    seen = []

    def fit(z_train):
        seen.append(z_train.copy())
        return np.array([z_train[0], 1.0 + z_train[1]])

    observed = pumas_r2(
        z_full,
        D,
        n_full,
        _SequenceRng([noise]),
        fit=fit,
        n_reps=1,
    )
    expected_w = np.array([expected_train[0], 1.0 + expected_train[1]])
    expected = r2(expected_w, expected_val, DenseLD(D))
    assert len(seen) == 1
    assert np.allclose(seen[0], expected_train)
    assert observed == pytest.approx(expected)


def test_pumas_conditional_bias_correction_has_zero_null():
    z_full = np.zeros(1)
    D = np.eye(1)
    n_full = 1000
    kwargs = dict(
        independent_weights=np.ones(1), frac_val=0.25, n_reps=1)
    raw = pumas_r2(
        z_full,
        D,
        n_full,
        _SequenceRng([np.ones(1)]),
        validation_bias="none",
        **kwargs,
    )
    corrected = pumas_r2(
        z_full,
        D,
        n_full,
        _SequenceRng([np.ones(1)]),
        **kwargs,
    )
    expected_bias = 750 / (1000 * 250)
    assert raw == pytest.approx(expected_bias)
    assert corrected == pytest.approx(0.0, abs=1e-15)


def test_pumas_r2_is_coherent_under_phenotype_rescaling():
    D = np.array([[1.0, 0.3], [0.3, 1.0]])
    z_full = np.array([0.2, -0.1])
    weights = np.array([0.7, -0.4])
    noises = [np.array([0.2, -1.1]), np.array([-0.8, 0.4])]
    base = pumas_r2(
        z_full,
        D,
        2000,
        _SequenceRng(noises),
        independent_weights=weights,
        n_reps=2,
        var_y=1.0,
    )
    scaled = pumas_r2(
        3.0 * z_full,
        D,
        2000,
        _SequenceRng(noises),
        independent_weights=weights,
        n_reps=2,
        var_y=9.0,
    )
    assert scaled == pytest.approx(base, rel=1e-14, abs=1e-14)


def test_pumas_requires_one_valid_weight_source():
    args = (np.zeros(2), np.eye(2), 1000, np.random.default_rng(0))
    with pytest.raises(ValueError, match="exactly one"):
        pumas_r2(*args)
    with pytest.raises(ValueError, match="exactly one"):
        pumas_r2(*args, fit=lambda z: z, independent_weights=np.ones(2))
    with pytest.raises(ValueError, match="shape"):
        pumas_r2(*args, fit=lambda z: np.ones(3), n_reps=1)


def test_pumas_agrees_with_ppb_and_individual_level():
    rows = run(n_reps=40, seed=0)
    assert rows
    for arch, method, gold, ppb, pum in rows:
        # PPB with exact target cross-products is algebraically the truth.
        assert abs(ppb - gold) <= 1e-9, f"{arch}/{method}: PPB {ppb} != gold {gold}"
        # PUMAS-style repeated learning estimates out-of-sample performance.
        assert abs(pum - gold) <= 0.03, f"{arch}/{method}: PUMAS {pum} vs gold {gold}"
