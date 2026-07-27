"""Assortative mating, and the variance it hides from a block-diagonal reference.

`docs/LIMITATIONS.md` attributed an observed R² overestimate to assortative
mating. These tests pin the mechanism: AM inflates additive genetic variance via
LD between *unlinked* loci, a block-diagonal `D` cannot see it, and (P3) is a
rank-one correction that does not need the missing covariance.
"""

import numpy as np
import pytest

from ppb.simulate import assortatively_mated_offspring, simulate_haplotypes

from experiments.assortative_mating import (  # noqa: E402
    BLOCK_SIZES, M, coverage_sweep, predicted_c, r2_bias, rank_one_transfer, run)


def test_mating_hits_the_requested_spouse_correlation():
    rng = np.random.default_rng(0)
    hap = simulate_haplotypes(4000, BLOCK_SIZES, np.full(M, 0.3), 0.5, rng)
    y = hap.sum(axis=1, dtype=np.float64) @ rng.normal(size=M)

    for target in (0.0, 0.25, 0.5):
        _, realized = assortatively_mated_offspring(
            hap, y, BLOCK_SIZES, rng, spouse_correlation=target)
        assert realized == pytest.approx(target, abs=0.05)


def test_mating_conserves_population_size_and_allele_frequency():
    """Meiosis reshuffles LD; it must not drift the marginal frequencies."""
    rng = np.random.default_rng(1)
    hap = simulate_haplotypes(4000, BLOCK_SIZES, np.full(M, 0.4), 0.5, rng)
    y = hap.sum(axis=1, dtype=np.float64) @ rng.normal(size=M)
    child, _ = assortatively_mated_offspring(hap, y, BLOCK_SIZES, rng,
                                             spouse_correlation=0.4)

    assert child.shape == hap.shape
    before = hap.sum(axis=1).mean(axis=0) / 2.0
    after = child.sum(axis=1).mean(axis=0) / 2.0
    assert np.abs(after - before).max() < 0.05


@pytest.mark.parametrize("target,inflates", [(0.0, False), (0.5, True)])
def test_assortative_mating_inflates_the_genetic_variance(target, inflates):
    """Random mating leaves the linkage-equilibrium variance alone; AM does not."""
    rng = np.random.default_rng(2)
    _, _, _, history = run(rng, n=4000, generations=6, spouse_correlation=target)
    final = history[-1]["inflation"]
    if inflates:
        assert final > 1.15, f"AM should inflate the variance; got {final:.3f}"
        # ...and it should track the classical equilibrium, not merely rise.
        r, h2 = history[-1]["spouse_r"], history[-1]["h2"]
        assert final == pytest.approx(1.0 / (1.0 - r * h2), rel=0.15)
    else:
        assert final == pytest.approx(1.0, abs=0.08)


def test_block_diagonal_ld_misses_it_and_r2_comes_out_too_high():
    """The consequence for PPB: a denominator too small, so R² too large."""
    rng = np.random.default_rng(3)
    _, g, beta, history = run(rng, n=6000, generations=6, spouse_correlation=0.5)
    fit = r2_bias(rng, g, beta, h2=history[-1]["h2"])

    assert fit["sd_block"] < fit["sd_realized"], "block-diagonal D must understate"
    assert fit["r2_block"] > fit["true_r2"] * 1.05, "so R² must be inflated"
    # Full genome-wide LD is not an approximation here -- it is the answer.
    assert fit["r2_full"] == pytest.approx(fit["true_r2"], rel=0.02)


def test_the_rank_one_correction_transfers_and_needs_no_ld():
    """(P3): one scalar, fitted or predicted, corrects scores it never saw."""
    rng = np.random.default_rng(4)
    _, g, beta, history = run(rng, n=6000, generations=6, spouse_correlation=0.5)
    r, h2 = history[-1]["spouse_r"], history[-1]["h2"]
    fit = r2_bias(rng, g, beta, h2=h2)
    c_pred = predicted_c(fit, beta, spouse_r=r, h2=h2)
    c, rows = rank_one_transfer(rng, fit, beta, c_pred)

    assert c > 0.0 and c_pred > 0.0
    by_label = {row[0]: row for row in rows}

    causal = by_label["causal beta"]
    assert causal[2] > 5.0, "the uncorrected error must be worth correcting"
    assert causal[3] < 1.0, "the fitted c should very nearly close it"
    assert causal[4] < causal[2] / 3.0, "theory-only c must still help a lot"

    # The control: a score that ignores the effect direction is not distorted by
    # AM, and the correction must leave it alone rather than inflate it.
    random_w = by_label["random weights"]
    assert random_w[2] < 2.0 and random_w[4] < 2.0


def test_am_bias_grows_with_how_much_of_the_trait_a_score_captures():
    """Real PGS are incomplete, which bounds how much of this reaches them."""
    rng = np.random.default_rng(5)
    _, g, beta, history = run(rng, n=6000, generations=6, spouse_correlation=0.5)
    fit = r2_bias(rng, g, beta, h2=history[-1]["h2"])
    rows = coverage_sweep(rng, fit, beta)

    errors = [err for _, _, err in rows]
    assert errors == sorted(errors), f"bias must grow with coverage: {errors}"
    assert errors[0] < errors[-1] / 4.0, "a sparse score carries far less bias"
