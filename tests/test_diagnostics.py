"""Block-level uncertainty and negative controls.

Two things are checked beyond input validation: that the jackknife standard
error actually matches the sampling variability of the estimate, and that the
sign-flip null is calibrated -- a genuinely null score must not look
significant, and a real one must.
"""

import numpy as np
import pytest

from ppb import (
    BlockDiagonalLD,
    DenseLD,
    r2_block_jackknife,
    r2_from_blocks,
    sign_flip_null,
)
from ppb.diagnostics import _blocks


def _corr_block(m, rho=0.7):
    d = np.arange(m)
    return rho ** np.abs(d[:, None] - d[None, :])


# --------------------------------------------------------------------------
# The property the sign-flip null rests on


def test_flipping_a_whole_block_preserves_the_denominator_exactly():
    """``D`` is block-diagonal, so negating every weight in one block leaves
    ``w' D w`` bit-for-bit unchanged while negating that block's ``w' z``.

    This is what makes the sign-flipped scores a legitimate null family: same
    denominator, same per-block magnitudes, no coherent association.
    """
    sizes = [40, 25, 35]
    blocks, start = [], 0
    for m in sizes:
        blocks.append((DenseLD(_corr_block(m)), np.arange(start, start + m)))
        start += m
    ld = BlockDiagonalLD(blocks)

    rng = np.random.default_rng(0)
    w = rng.standard_normal(start)
    z = rng.standard_normal(start) * 0.05
    base_den = ld.quad(w)
    base_num = float(w @ z)

    for flip in range(len(sizes)):
        flipped = w.copy()
        _, idx = blocks[flip]
        flipped[idx] *= -1.0
        assert ld.quad(flipped) == base_den            # exactly, not approximately
        block_u = float(w[idx] @ z[idx])
        assert float(flipped @ z) == pytest.approx(base_num - 2.0 * block_u)


# --------------------------------------------------------------------------
# Jackknife


def test_jackknife_se_matches_the_sampling_variability_of_r2():
    """The reported SE must track the actual spread of the estimate.

    Blocks are independent, so ``u_b`` is drawn per replicate around a coherent
    per-block mean and the empirical SD across replicates is the target.
    """
    rng = np.random.default_rng(0)
    n_blocks = 200
    v = rng.uniform(0.5, 1.5, n_blocks)
    mu = rng.normal(0.02, 0.01, n_blocks)
    noise = 0.02

    estimates, ses = [], []
    for _ in range(300):
        u = mu + noise * rng.standard_normal(n_blocks)
        fit = r2_block_jackknife(u, v)
        estimates.append(fit.r2)
        ses.append(fit.se)

    empirical = float(np.std(estimates))
    reported = float(np.mean(ses))
    assert 0.8 < reported / empirical < 1.25, f"{reported} vs {empirical}"


def test_jackknife_grouping_changes_resolution_not_the_estimate():
    rng = np.random.default_rng(1)
    n_blocks = 66
    u = rng.normal(0.02, 0.01, n_blocks)
    v = rng.uniform(0.5, 1.5, n_blocks)
    chrom = np.repeat(np.arange(22), 3)

    per_block = r2_block_jackknife(u, v)
    per_chrom = r2_block_jackknife(u, v, groups=chrom)

    assert per_block.r2 == per_chrom.r2 == r2_from_blocks(u, v)
    assert per_block.n_groups == n_blocks
    assert per_chrom.n_groups == 22
    assert per_chrom.se > 0.0
    # Same data, comparable magnitude -- the grouping is a resolution choice.
    assert 0.3 < per_chrom.se / per_block.se < 3.0


def test_jackknife_reports_a_dominating_block():
    """One block carrying the estimate must be visible, not averaged away."""
    rng = np.random.default_rng(7)
    n_blocks = 60
    v = np.full(n_blocks, 1.0)
    u = rng.normal(0.01, 0.002, n_blocks)
    even = r2_block_jackknife(u, v)

    u_spiked = u.copy()
    u_spiked[7] = 0.5                       # one block dominates the numerator
    spiked = r2_block_jackknife(u_spiked, v)

    # Equal contributions would be 1/60 = 0.017; the spiked block takes nearly all.
    assert even.max_variance_share < 0.15
    assert spiked.max_variance_share > 0.9
    assert int(np.argmax(np.abs(spiked.delete_values - spiked.r2))) == 7


def test_jackknife_variance_share_is_even_when_all_groups_agree():
    u, v = np.full(20, 0.01), np.full(20, 1.0)
    fit = r2_block_jackknife(u, v)
    assert fit.max_variance_share == pytest.approx(1.0 / 20, abs=0.02)


def test_jackknife_rejects_degenerate_input():
    u, v = np.ones(4), np.ones(4)
    with pytest.raises(ValueError, match="at least 2 groups"):
        r2_block_jackknife(u, v, groups=np.zeros(4))
    with pytest.raises(ValueError, match="one entry per block"):
        r2_block_jackknife(u, v, groups=np.arange(3))
    with pytest.raises(ValueError, match="not positive"):
        r2_block_jackknife(u, np.zeros(4))
    with pytest.raises(ValueError, match="positive semi-definite"):
        r2_block_jackknife(u, np.array([1.0, -1.0, 1.0, 1.0]))


def test_jackknife_requires_every_delete_one_subset_to_be_defined():
    """All the score variance in one group means that group cannot be deleted."""
    u = np.array([0.1, 0.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="jackknife needs every delete-one"):
        r2_block_jackknife(u, v, groups=np.array([0, 1, 1, 1]))


# --------------------------------------------------------------------------
# Sign-flip null


def test_sign_flip_null_is_calibrated_under_a_null_score():
    """A score with no coherent block association must not look significant."""
    rng = np.random.default_rng(2)
    n_blocks = 200
    v = rng.uniform(0.5, 1.5, n_blocks)

    z_scores, p_values = [], []
    for _ in range(300):
        u = 0.02 * rng.standard_normal(n_blocks)      # no coherent signal
        control = sign_flip_null(u, v, n_draws=500, rng=rng)
        z_scores.append(control.z)
        p_values.append(control.p_value)

    z_scores = np.asarray(z_scores)
    assert abs(z_scores.mean()) < 0.2, f"z mean {z_scores.mean()}"
    assert 0.8 < z_scores.std() < 1.25, f"z sd {z_scores.std()}"
    # A permutation p-value with the add-one correction is valid but slightly
    # conservative, so bound the rejection rate rather than pinning it.
    assert np.mean(np.asarray(p_values) < 0.05) < 0.12


def test_sign_flip_null_separates_a_real_score_from_a_null_one():
    rng = np.random.default_rng(3)
    n_blocks = 200
    v = rng.uniform(0.5, 1.5, n_blocks)

    real = sign_flip_null(np.abs(rng.normal(0.02, 0.005, n_blocks)), v,
                          n_draws=500, rng=rng)
    null = sign_flip_null(0.02 * rng.standard_normal(n_blocks), v,
                          n_draws=500, rng=rng)

    assert real.z > 8.0 and real.ratio > 20.0
    assert real.p_value < 0.01
    assert abs(null.z) < 3.0
    assert null.ratio < 10.0


def test_sign_flip_null_mean_is_the_exact_rademacher_expectation():
    """``E[R^2]`` under random block signs is ``sum(u^2) / sum(v)``."""
    rng = np.random.default_rng(4)
    u = rng.normal(0.0, 0.03, 40)
    v = rng.uniform(0.5, 1.5, 40)
    control = sign_flip_null(u, v)

    signs = rng.integers(0, 2, size=(40000, u.size)) * 2 - 1
    empirical = float(np.mean((signs @ u) ** 2) / v.sum())
    assert control.null_mean == pytest.approx(float(np.dot(u, u) / v.sum()))
    assert control.null_mean == pytest.approx(empirical, rel=0.05)


def test_sign_flip_null_is_blind_to_a_uniform_rescale_of_z():
    """Documented limitation: it controls association, not calibration."""
    rng = np.random.default_rng(5)
    u = np.abs(rng.normal(0.02, 0.005, 80))
    v = rng.uniform(0.5, 1.5, 80)
    base = sign_flip_null(u, v)
    rescaled = sign_flip_null(2.5 * u, v)

    assert rescaled.r2 == pytest.approx(base.r2 * 2.5 ** 2)
    assert rescaled.ratio == pytest.approx(base.ratio)
    assert rescaled.z == pytest.approx(base.z)


def test_sign_flip_null_rejects_degenerate_input():
    with pytest.raises(ValueError, match="degenerate"):
        sign_flip_null(np.zeros(5), np.ones(5))
    with pytest.raises(ValueError, match="n_draws must be non-negative"):
        sign_flip_null(np.ones(5), np.ones(5), n_draws=-1)


def test_block_arrays_are_validated():
    with pytest.raises(ValueError, match="equal length"):
        _blocks(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="at least one block"):
        _blocks(np.empty(0), np.empty(0))
    with pytest.raises(ValueError, match="finite"):
        _blocks(np.array([np.nan, 1.0]), np.ones(2))


# --------------------------------------------------------------------------
# Against the estimator itself


def test_block_diagnostics_agree_with_the_genome_wide_estimator():
    """``r2_from_blocks`` on per-block products must equal ``ppb.r2`` on the whole."""
    from ppb import r2

    sizes = [30, 45, 25, 50]
    blocks, start = [], 0
    for m in sizes:
        blocks.append((DenseLD(_corr_block(m)), np.arange(start, start + m)))
        start += m
    ld = BlockDiagonalLD(blocks)

    rng = np.random.default_rng(6)
    w = rng.standard_normal(start)
    z = rng.standard_normal(start) * 0.05

    u = np.array([float(w[idx] @ z[idx]) for _, idx in blocks])
    v = np.array([backend.quad(w[idx]) for backend, idx in blocks])

    assert r2_from_blocks(u, v) == pytest.approx(r2(w, z, ld), rel=1e-12)
    assert sign_flip_null(u, v).r2 == pytest.approx(r2(w, z, ld), rel=1e-12)
    assert r2_block_jackknife(u, v).r2 == pytest.approx(r2(w, z, ld), rel=1e-12)


def test_sign_flip_z_is_bounded_by_the_square_root_of_the_block_count():
    """Cauchy-Schwarz: ``sum(u) <= sqrt(n) * ||u||``, with equality when every
    block contributes the same signed amount. z is a coherence measure on a
    fixed scale, not an unbounded significance statistic."""
    rng = np.random.default_rng(8)
    n_blocks = 150
    v = rng.uniform(0.5, 1.5, n_blocks)
    ceiling = np.sqrt(n_blocks)

    for u in (rng.normal(0.02, 0.01, n_blocks),
              np.abs(rng.normal(0.02, 0.001, n_blocks)),
              rng.standard_normal(n_blocks)):
        assert abs(sign_flip_null(u, v).z) <= ceiling + 1e-9

    perfectly_coherent = np.full(n_blocks, 0.02)
    assert sign_flip_null(perfectly_coherent, v).z == pytest.approx(ceiling)
