"""Tests for the diploid simulator and the method-benchmark concordance."""

import numpy as np
import pytest

from ppb.simulate import _norm_ppf, simulate_diploid_genotypes

from experiments.benchmark_methods import (  # noqa: E402
    METHODS,
    ldpred_inf_weights,
    run,
)


# --- simulation primitives -------------------------------------------------

def test_norm_ppf_matches_known_quantiles():
    got = _norm_ppf([0.025, 0.5, 0.975])
    assert np.allclose(got, [-1.959963985, 0.0, 1.959963985], atol=1e-6)


def test_diploid_genotypes_are_discrete_and_standardized():
    rng = np.random.default_rng(0)
    maf = np.full(20, 0.3)
    X = simulate_diploid_genotypes(4000, [20], maf, rho=0.5, rng=rng)
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(X.std(axis=0), 1.0, atol=1e-8)
    # underlying dosages are 0/1/2 -> at most 3 distinct standardized values.
    assert all(len(np.unique(X[:, j])) <= 3 for j in range(X.shape[1]))


def test_diploid_within_block_ld_exceeds_cross_block():
    rng = np.random.default_rng(1)
    maf = np.full(20, 0.3)
    X = simulate_diploid_genotypes(4000, [10, 10], maf, rho=0.7, rng=rng)
    C = np.corrcoef(X.T)
    within = np.abs(C[np.ix_(range(10), range(10))]).mean()
    cross = np.abs(C[np.ix_(range(10), range(10, 20))]).mean()
    assert within > cross


def test_ldpred_inf_weights_are_finite():
    rng = np.random.default_rng(2)
    D = np.eye(5) + 0.1 * rng.standard_normal((5, 5))
    D = 0.5 * (D + D.T)
    w = ldpred_inf_weights(rng.standard_normal(5), D, n_train=1000, h2=0.5)
    assert w.shape == (5,) and np.all(np.isfinite(w))


# --- the benchmark ---------------------------------------------------------

def _concordance(records, kind):
    rows = [(t, e) for (_, _, k, t, e) in records if k == kind]
    true = np.array([t for t, _ in rows]); est = np.array([e for _, e in rows])
    ok = true > 1e-6
    pearson = float(np.corrcoef(est, true)[0, 1])
    spearman = float(np.corrcoef(np.argsort(np.argsort(est)),
                                 np.argsort(np.argsort(true)))[0, 1])
    mean_bias = float((100.0 * (est[ok] - true[ok]) / true[ok]).mean())
    return pearson, spearman, mean_bias


def _best_method(records, kind, which):
    idx = {"true": 3, "ppb": 4}[which]
    means = {m: np.mean([r[idx] for r in records if r[1] == m and r[2] == kind])
             for m in METHODS}
    return max(means, key=means.get)


def test_benchmark_concordance_with_individual_level():
    records = run(m=200, block_size=25, n=1500, n_reps=6, seed=1)

    # PPB tracks the individual-level R^2 across methods and architectures.
    pearson, spearman, mean_bias = _concordance(records, "exact")
    assert pearson > 0.90, f"Pearson {pearson:.3f} too low"
    assert spearman > 0.85, f"Spearman {spearman:.3f} too low (ranking not preserved)"
    assert abs(mean_bias) < 6.0, f"mean bias {mean_bias:.2f}% too large"

    # The oracle (true causal effects) is the best method under both evaluations.
    assert _best_method(records, "exact", "true") == "causal"
    assert _best_method(records, "exact", "ppb") == "causal"

    # LR8 at 99% variance retention stays high-concordance.
    p99, s99, _ = _concordance(records, "lr8@0.99")
    assert p99 > 0.88 and s99 > 0.82
