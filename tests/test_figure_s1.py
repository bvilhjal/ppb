"""Reproduction test for the LD-reference experiment (Supplementary Fig. S1).

Encodes the paper's qualitative finding as checkable assertions: estimating R^2
with test-set LD is exact, with training-set LD is biased, and with an
independent LD reference is unbiased.
"""

import numpy as np
import pytest

from ppb.simulate import (
    marginal_stats,
    population_ld,
    sample_genotypes,
    simulate_phenotype,
)

from experiments.figure_s1 import run  # noqa: E402


# --- simulation primitives -------------------------------------------------

def test_population_ld_is_symmetric_psd_unit_diagonal():
    S = population_ld(60, block_size=20, rho=0.6)
    assert np.allclose(S, S.T)
    assert np.allclose(np.diag(S), 1.0)
    assert np.linalg.eigvalsh(S).min() > -1e-8


def test_sample_genotypes_are_standardized():
    rng = np.random.default_rng(0)
    X = sample_genotypes(population_ld(40, 20, 0.5), 5000, rng)
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(X.std(axis=0), 1.0, atol=1e-8)


def test_simulate_phenotype_has_target_genetic_variance():
    rng = np.random.default_rng(1)
    X = sample_genotypes(population_ld(50, 25, 0.5), 20000, rng)
    beta = np.zeros(50); beta[:10] = rng.standard_normal(10)
    h2 = 0.5
    y = simulate_phenotype(X, beta, h2, rng)
    assert abs(y.mean()) < 1e-8 and abs(y.std() - 1.0) < 1e-8
    g = X @ beta; g = g / g.std()
    assert abs(float(np.corrcoef(g, y)[0, 1] ** 2) - h2) < 0.05  # genetic var ~ h2


def test_marginal_stats_shapes():
    rng = np.random.default_rng(2)
    X = sample_genotypes(population_ld(30, 15, 0.5), 1000, rng)
    y = simulate_phenotype(X, np.r_[np.ones(5), np.zeros(25)], 0.4, rng)
    z, t = marginal_stats(X, y)
    assert z.shape == (30,) and t.shape == (30,)


# --- the reproduction ------------------------------------------------------

def _mean_pct_bias(records, pgs, ref):
    rows = [(tr, es) for (p, r, _, tr, es) in records if p == pgs and r == ref]
    true = np.array([tr for tr, _ in rows])
    est = np.array([es for _, es in rows])
    ok = true > 1e-6
    bias = float((100.0 * (est[ok] - true[ok]) / true[ok]).mean())
    corr = float(np.corrcoef(est, true)[0, 1])
    return bias, corr


def test_ld_reference_reproduces_figure_s1():
    records = run(m=300, n=2000, n_causal=30, n_phenos=150, seed=1)

    # (1) Test-set LD reproduces the individual-level R^2 exactly, for every PGS.
    for pgs in ("causal", "marginal", "pT"):
        bias, corr = _mean_pct_bias(records, pgs, "test")
        assert abs(bias) < 0.2, f"{pgs}/test bias {bias:.3f}% not ~0"
        assert corr > 0.999, f"{pgs}/test corr {corr:.4f} not ~1"

    # (2) For data-derived weights, training LD is biased and independent LD is not.
    for pgs in ("marginal", "pT"):
        b_train, _ = _mean_pct_bias(records, pgs, "train")
        b_indep, _ = _mean_pct_bias(records, pgs, "indep")
        assert abs(b_indep) < 3.0, f"{pgs}/indep bias {b_indep:.2f}% too large"
        assert abs(b_train) > 3.5, f"{pgs}/train bias {b_train:.2f}% not clearly biased"
        assert abs(b_train) > 2.0 * abs(b_indep), (
            f"{pgs}: train bias {b_train:.2f}% not clearly worse than indep {b_indep:.2f}%")

    # (3) All references preserve ranking (high correlation with the truth).
    for pgs in ("causal", "marginal", "pT"):
        for ref in ("test", "train", "indep"):
            _, corr = _mean_pct_bias(records, pgs, ref)
            assert corr > 0.95, f"{pgs}/{ref} corr {corr:.4f} too low"
