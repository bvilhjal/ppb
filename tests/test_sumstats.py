"""Tests for GWAS summary-statistic conversion and per-variant sample sizes."""

import numpy as np
import pytest

from ppb import standardized_marginal, zstat_to_corr

from experiments.per_variant_n import run  # noqa: E402


def test_zstat_to_corr_recovers_exact_correlation():
    """From an exact marginal regression t-stat, r_j == sample correlation."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.standard_normal(n); x = (x - x.mean()) / x.std()
    y = 0.3 * x + rng.standard_normal(n); y = (y - y.mean()) / y.std()
    r_true = float((x @ y) / n)                        # sample correlation
    beta = (x @ y) / (x @ x)                           # OLS slope (= r for std x)
    resid = y - beta * x
    se = np.sqrt((resid @ resid) / (n - 2) / (x @ x))
    r = standardized_marginal(beta, se, n)
    assert abs(float(r) - r_true) < 1e-10


def test_zstat_to_corr_vectorized_and_per_variant_n():
    t = np.array([2.0, -3.0, 0.5])
    r = zstat_to_corr(t, np.array([1000, 500, 2000]))
    assert r.shape == (3,)
    assert np.sign(r[1]) == -1                          # sign preserved
    assert np.all(np.abs(r) < 1)


def test_standardized_marginal_validation():
    with pytest.raises(ValueError):
        standardized_marginal([0.1], [0.0], 1000)       # se must be > 0
    with pytest.raises(ValueError):
        zstat_to_corr([1.0], 2)                          # n must exceed 2
    with pytest.raises(ValueError, match="finite"):
        standardized_marginal([np.nan], [1.0], 1000)
    with pytest.raises(ValueError, match="finite"):
        zstat_to_corr([1.0], np.inf)


def test_uniform_N_biases_r2_but_per_variant_n_corrects_it():
    rows = run(seed=0)
    for frac, gold, per_variant, uniform in rows:
        # per-variant conversion tracks the individual-level truth.
        assert abs(per_variant - gold) <= 0.02 * max(gold, 1e-3), (
            f"per-variant {per_variant:.4f} != gold {gold:.4f} at frac {frac}")
        if frac >= 1.0:
            assert abs(uniform - gold) <= 0.02 * gold    # no heterogeneity: no bias
        else:
            # heterogeneous n: uniform-N underestimates, clearly worse than per-variant.
            assert uniform < gold
            assert abs(uniform - gold) > abs(per_variant - gold)
