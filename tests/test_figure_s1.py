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

# Fixed replication list: assertions are cross-seed means with 3-SE bounds, not
# single-seed snapshots (review F3 -- the independent-reference bias flips sign
# across seeds, and the training-reference bias is stable but seed-dependent).
SEEDS = (0, 1, 2, 3, 4)


def _mean_se(values):
    """Cross-seed mean and standard error of a per-seed statistic."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def _mean_pct_bias(records, pgs, ref):
    rows = [(tr, es) for (p, r, _, tr, es) in records if p == pgs and r == ref]
    true = np.array([tr for tr, _ in rows])
    est = np.array([es for _, es in rows])
    ok = true > 1e-6
    bias = float((100.0 * (est[ok] - true[ok]) / true[ok]).mean())
    corr = float(np.corrcoef(est, true)[0, 1])
    return bias, corr


def test_ld_reference_reproduces_figure_s1():
    cells = {}  # (pgs, ref) -> list of per-seed (mean %bias, corr)
    for seed in SEEDS:
        records = run(m=300, n=2000, n_causal=30, n_phenos=150, seed=seed)
        for pgs in ("causal", "marginal", "pT"):
            for ref in ("test", "train", "indep"):
                cells.setdefault((pgs, ref), []).append(
                    _mean_pct_bias(records, pgs, ref))

    def bias_stats(pgs, ref):
        return _mean_se([b for b, _ in cells[(pgs, ref)]])

    def worst_corr(pgs, ref):
        return min(c for _, c in cells[(pgs, ref)])

    # (1) Test-set LD reproduces the individual-level R^2 exactly, for every PGS.
    for pgs in ("causal", "marginal", "pT"):
        bias_m, bias_se = bias_stats(pgs, "test")
        assert abs(bias_m) + 3 * bias_se < 0.2, (
            f"{pgs}/test bias {bias_m:+.4f} +/- {3 * bias_se:.4f}% not ~0")
        assert worst_corr(pgs, "test") > 0.999, (
            f"{pgs}/test corr {worst_corr(pgs, 'test'):.4f} not ~1")

    # (2) For data-derived weights, training LD is biased and independent LD is not.
    for pgs in ("marginal", "pT"):
        train_m, train_se = bias_stats(pgs, "train")
        indep_m, indep_se = bias_stats(pgs, "indep")
        # The independent-reference bias is consistent with zero: |mean| within
        # 3 SE (it flips sign across seeds) and its 3-SE envelope stays under
        # 1%, well inside the original 3% tolerance.
        assert abs(indep_m) < 3 * indep_se, (
            f"{pgs}/indep bias {indep_m:+.3f} +/- {indep_se:.3f}% not ~0")
        assert abs(indep_m) + 3 * indep_se < 1.0, (
            f"{pgs}/indep bias {indep_m:+.3f} +/- {3 * indep_se:.3f}% too large")
        # The training-reference bias is large and stable: the mean sits more
        # than 3 SE beyond the original 3.5% "clearly biased" threshold.
        assert train_m + 3 * train_se < -3.5, (
            f"{pgs}/train bias {train_m:+.3f} +/- {3 * train_se:.3f}% not clearly biased")
        # And it is clearly worse than the independent reference: the per-seed
        # gap |train| - 2|indep| stays above 0 by 3 SE.
        gap_m, gap_se = _mean_se([
            abs(b_tr) - 2.0 * abs(b_in)
            for (b_tr, _), (b_in, _) in zip(cells[(pgs, "train")],
                                            cells[(pgs, "indep")])])
        assert gap_m - 3 * gap_se > 0.0, (
            f"{pgs}: |train|-2|indep| gap {gap_m:+.3f} +/- {3 * gap_se:.3f}% "
            "not clearly worse than indep")

    # (3) All references preserve ranking (high correlation with the truth).
    for pgs in ("causal", "marginal", "pT"):
        for ref in ("test", "train", "indep"):
            assert worst_corr(pgs, ref) > 0.95, (
                f"{pgs}/{ref} corr {worst_corr(pgs, ref):.4f} too low")
