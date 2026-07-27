"""The predicted score distribution must match one measured on real genotypes.

(P1) and (P2) claim the first two moments of a PGS are fixed by allele
frequencies and LD. These tests check that against simulated individuals rather
than against the algebra that produced them.
"""

import numpy as np
import pytest

from ppb import DenseLD, ScoreDistribution, score_distribution
from ppb.harmonize import VariantTable
from ppb.ld_backend import BlockDiagonalLD
from ppb.simulate import _diploid_dosages


def _table(m, *, a1="A", a2="G"):
    return VariantTable(
        chrom=np.full(m, "1"),
        pos=np.arange(1, m + 1),
        a1=np.full(m, a1),
        a2=np.full(m, a2),
    )


def test_independent_variants_match_the_binomial_moments():
    """With D = I the moments are exactly the sum of per-variant binomials."""
    rng = np.random.default_rng(0)
    m, n = 40, 200_000
    f = rng.uniform(0.1, 0.9, size=m)
    w = rng.normal(size=m)
    variants = _table(m)

    dist = score_distribution(DenseLD(np.eye(m)), variants, variants, w, f)

    assert dist.mean == pytest.approx(2.0 * float(w @ f))
    assert dist.variance == pytest.approx(float(np.sum(w**2 * 2 * f * (1 - f))))

    g = rng.binomial(2, f, size=(n, m)).astype(np.float64)
    scores = g @ w
    assert scores.mean() == pytest.approx(dist.mean, abs=6.0 * dist.sd / np.sqrt(n))
    assert scores.std() == pytest.approx(dist.sd, rel=0.01)


def test_predicted_moments_match_simulated_individuals_under_ld():
    """The whole point: predict a real cohort's score spread from f and D alone."""
    rng = np.random.default_rng(1)
    block_sizes = [25] * 8
    m, n = sum(block_sizes), 40_000
    f = rng.uniform(0.15, 0.85, size=m)
    g = _diploid_dosages(n, block_sizes, f, 0.6, rng)

    f_hat = g.mean(axis=0) / 2.0
    D = np.corrcoef(g.T)
    w = rng.normal(size=m) * rng.binomial(1, 0.3, size=m)   # a sparse score
    variants = _table(m)

    dist = score_distribution(DenseLD(D), variants, variants, w, f_hat)
    scores = g @ w

    assert scores.mean() == pytest.approx(dist.mean, abs=6.0 * dist.sd / np.sqrt(n))
    # sd rests on the HWE identity Var(g) = 2f(1-f); the simulator's dosages are
    # in HWE up to sampling, so this is the assumption under test. The tolerance
    # is deliberately tighter than the LD term itself: predicting this cohort
    # with D = I lands 3.2% out, so a regression that drops LD fails here.
    assert scores.std() == pytest.approx(dist.sd, rel=0.01)


def test_percentiles_track_the_empirical_ranks():
    rng = np.random.default_rng(2)
    block_sizes = [20] * 6
    m, n = sum(block_sizes), 20_000
    f = rng.uniform(0.2, 0.8, size=m)
    g = _diploid_dosages(n, block_sizes, f, 0.4, rng)
    w = rng.normal(size=m)
    variants = _table(m)

    dist = score_distribution(DenseLD(np.corrcoef(g.T)), variants, variants,
                              w, g.mean(axis=0) / 2.0)
    scores = g @ w
    predicted = dist.percentile(scores)
    empirical = 100.0 * (np.argsort(np.argsort(scores)) + 0.5) / n

    assert np.abs(predicted - empirical).mean() < 1.5
    assert dist.standardize(dist.mean) == pytest.approx(0.0)


def test_block_variance_share_flags_a_dominating_variant():
    """A score whose variance sits in one block is a mixture, not a normal."""
    rng = np.random.default_rng(3)
    sizes = [4, 4, 4]
    m = sum(sizes)
    f = np.full(m, 0.3)
    blocks, start = [], 0
    for k in sizes:
        blocks.append((DenseLD(np.eye(k)), np.arange(start, start + k)))
        start += k
    ld = BlockDiagonalLD(blocks)
    variants = _table(m)

    spread = score_distribution(ld, variants, variants, np.ones(m), f)
    assert spread.n_blocks == 3
    assert spread.max_variance_share == pytest.approx(1 / 3, rel=1e-6)

    w = np.zeros(m)
    w[0] = 50.0                       # one APOE-sized effect
    w[4:] = 0.1
    dominated = score_distribution(ld, variants, variants, w, f)
    assert dominated.max_variance_share > 0.99
    del rng


def test_allele_frequency_is_not_harmonized_but_weights_are():
    """A flipped submission must give the same distribution, not a mirrored one."""
    m = 12
    rng = np.random.default_rng(4)
    f = rng.uniform(0.2, 0.8, size=m)
    w = rng.normal(size=m)
    reference = _table(m)
    flipped = VariantTable(chrom=reference.chrom, pos=reference.pos,
                           a1=reference.a2, a2=reference.a1)

    direct = score_distribution(DenseLD(np.eye(m)), reference, reference, w, f)
    swapped = score_distribution(DenseLD(np.eye(m)), reference, flipped, w, f)

    # harmonize_to negates a swapped weight, so the mean flips sign about 0 only
    # after accounting for f being the reference allele's frequency throughout.
    assert swapped.mean == pytest.approx(-direct.mean)
    assert swapped.variance == pytest.approx(direct.variance)


def test_inbreeding_inflates_the_variance_but_not_the_mean():
    m = 10
    f = np.full(m, 0.4)
    w = np.ones(m)
    variants = _table(m)
    ld = DenseLD(np.eye(m))

    base = score_distribution(ld, variants, variants, w, f)
    structured = score_distribution(ld, variants, variants, w, f, inbreeding=0.5)

    assert structured.mean == pytest.approx(base.mean)
    assert structured.variance == pytest.approx(1.5 * base.variance)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.2, np.nan])
def test_impossible_allele_frequencies_are_rejected(bad):
    m = 5
    f = np.full(m, 0.3)
    f[2] = bad
    variants = _table(m)
    with pytest.raises(ValueError, match="allele_frequency"):
        score_distribution(DenseLD(np.eye(m)), variants, variants,
                           np.ones(m), f)


def test_a_score_with_no_matched_weight_has_no_scale():
    m = 6
    variants = _table(m)
    other = VariantTable(chrom=np.full(m, "2"), pos=variants.pos,
                         a1=variants.a1, a2=variants.a2)
    with pytest.raises(ValueError, match="no variance has no standardized scale"):
        score_distribution(DenseLD(np.eye(m)), variants, other,
                           np.ones(m), np.full(m, 0.3))


def test_inbreeding_outside_the_unit_interval_is_rejected():
    m = 4
    variants = _table(m)
    with pytest.raises(ValueError, match="inbreeding"):
        score_distribution(DenseLD(np.eye(m)), variants, variants,
                           np.ones(m), np.full(m, 0.3), inbreeding=-0.2)


def test_result_round_trips_through_a_dict():
    m = 5
    variants = _table(m)
    dist = score_distribution(DenseLD(np.eye(m)), variants, variants,
                              np.ones(m), np.full(m, 0.25))
    payload = dist.to_dict()
    assert payload["sd"] == dist.sd
    assert ScoreDistribution(**payload).mean == dist.mean
