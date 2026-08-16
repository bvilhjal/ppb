"""Tests for PUMAS-style subsampling and its agreement with PPB."""

import numpy as np
import pytest

from ppb import DenseLD, r2

from experiments.pumas_agreement import (  # noqa: E402
    pumas_r2,
    run,
    run_diploid,
    subsample_sumstats,
)


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


# Fixed replication list: the agreement claim is made on the cross-seed mean
# with an SE-derived bound, not on a single-seed realization (review F3 -- the
# single-seed |PUMAS - gold| reaches 0.038 at other seeds).
SEEDS = (0, 1, 2, 3, 4)


def _mean_se(values):
    """Cross-seed mean and standard error of a per-seed statistic."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def test_pumas_agrees_with_ppb_and_individual_level():
    by_cell = {}
    for seed in SEEDS:
        for arch, method, gold, ppb, pum in run(n_reps=40, seed=seed):
            by_cell.setdefault((arch, method), []).append((gold, ppb, pum))
    for (arch, method), rows in sorted(by_cell.items()):
        ppb_diffs = np.array([ppb - gold for gold, ppb, _ in rows])
        pum_diffs = np.array([pum - gold for gold, _, pum in rows])
        pum_mean, pum_se = _mean_se(pum_diffs)
        # PPB with exact target cross-products is algebraically the truth.
        worst_ppb = float(np.abs(ppb_diffs).max())
        assert worst_ppb <= 1e-9, (
            f"{arch}/{method}: worst |PPB - gold| {worst_ppb:.2e} across seeds")
        # PUMAS-style repeated learning estimates out-of-sample performance:
        # the cross-seed mean difference stays within 0.03 R^2, and so does its
        # upper 3-SE bound. The bound is one-sided because the fitted-weight
        # cells carry a small systematic negative offset (worst mean -0.024);
        # the upper side is where a real regression would breach tolerance.
        assert abs(pum_mean) <= 0.03, (
            f"{arch}/{method}: PUMAS mean-vs-gold {pum_mean:+.4f} R^2")
        assert pum_mean + 3 * pum_se <= 0.03, (
            f"{arch}/{method}: PUMAS vs gold {pum_mean:+.4f} +/- {3 * pum_se:.4f}")


def test_pumas_diploid_leg_agreement():
    """The diploid leg (review F4a): the same claim on 0/1/2 dosages.

    Equation 1's moment covariance is exact only for Gaussian genotypes; on
    bounded, MAF-skewed dosages it is a fourth-moment approximation, and this
    leg reports what that approximation costs rather than assuming it away.
    What is actually true across the five seeds: the exact-summary identity is
    untouched and the independent causal oracle stays unbiased, while scores
    *fitted* on the pseudo-training splits shift systematically downward
    (measured means -0.012 to -0.035 R^2). The bounds assert that direction,
    with margins for both "no worse than" and "genuinely one-sided".
    """
    by_cell = {}
    for seed in SEEDS:
        for arch, method, gold, ppb, pum in run_diploid(n_reps=40, seed=seed):
            assert abs(ppb - gold) <= 1e-9, (
                f"{arch}/{method} seed {seed}: PPB with exact cross-products "
                f"must stay algebraically exact on any generator")
            # The approximation's worst single-seed cost stays well bounded.
            assert abs(pum - gold) <= 0.05, (
                f"{arch}/{method} seed {seed}: |PUMAS - gold| {abs(pum - gold):.4f}")
            by_cell.setdefault((arch, method), []).append(pum - gold)
    for (arch, method), gaps in sorted(by_cell.items()):
        mean, se = _mean_se(gaps)
        if method == "causal":
            # Independent weights: the pseudo-split model stays unbiased.
            assert abs(mean) <= 0.03, (
                f"{arch}/causal: oracle shifted {mean:+.4f} on the diploid leg")
        else:
            # Fitted weights: the Gaussian V under-disperses the pseudo-splits
            # for dosages, so the shift is one-directional. Assert it as such.
            assert -0.07 <= mean <= -0.01, (
                f"{arch}/{method}: diploid shift {mean:+.4f} +/- {se:.4f} must "
                "stay small and downward -- a larger or upward shift would mean "
                "the approximation is broken or the finding went stale")
