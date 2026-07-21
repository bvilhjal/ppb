"""Tests for the block-diagonal LD backend and the low-rank (LR8) builder."""

import numpy as np
import pytest

from ppb import BlockDiagonalLD, DenseLD, LowRankLD, lowrank_ld, r2

TOL = 1e-9


def _rng(seed):
    return np.random.default_rng(seed)


def _corr_block(k, seed, strength=0.6):
    """A random PSD correlation matrix (unit diagonal) of size k."""
    rng = _rng(seed)
    A = rng.standard_normal((k, max(2, k // 2)))
    C = A @ A.T + strength * np.eye(k)
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)                              # unit diagonal
    return 0.5 * (C + C.T)


def _block_diagonal_matrix(sizes, seed):
    """A genuinely block-diagonal D (off-block entries zero) and its blocks."""
    blocks, mats = [], []
    pos = 0
    for b, k in enumerate(sizes):
        C = _corr_block(k, seed + b)
        mats.append(C)
        blocks.append((C, np.arange(pos, pos + k)))
        pos += k
    m = pos
    D = np.zeros((m, m))
    for C, idx in blocks:
        D[np.ix_(idx, idx)] = C
    return D, blocks


def test_block_diagonal_matches_full_dense_exactly():
    sizes = [5, 8, 3, 6]
    D, blocks = _block_diagonal_matrix(sizes, seed=0)
    rng = _rng(10)
    w = rng.standard_normal(D.shape[0])
    z = rng.standard_normal(D.shape[0]) * 0.1
    bd = BlockDiagonalLD([(DenseLD(C), idx) for C, idx in blocks])
    assert bd.m == D.shape[0]
    assert abs(bd.quad(w) - DenseLD(D).quad(w)) <= TOL * abs(DenseLD(D).quad(w))
    assert abs(r2(w, z, bd) - r2(w, z, DenseLD(D))) <= TOL


def test_block_diagonal_with_lowrank_blocks():
    """Full-variance low-rank blocks reproduce the exact block-diagonal quad."""
    sizes = [6, 7]
    D, blocks = _block_diagonal_matrix(sizes, seed=3)
    w = _rng(11).standard_normal(D.shape[0])
    bd_exact = BlockDiagonalLD([(DenseLD(C), idx) for C, idx in blocks])
    bd_lr = BlockDiagonalLD(
        [(lowrank_ld(C, variance=1.0), idx) for C, idx in blocks])
    assert abs(bd_lr.quad(w) - bd_exact.quad(w)) <= 1e-8 * abs(bd_exact.quad(w))


def test_lowrank_full_variance_reproduces_dense():
    C = _corr_block(12, seed=5)
    w = _rng(12).standard_normal(12)
    low = lowrank_ld(C, variance=1.0)
    assert low.rank == 12
    assert abs(low.quad(w) - DenseLD(C).quad(w)) <= 1e-8 * abs(DenseLD(C).quad(w))


def test_lowrank_truncation_is_nonnegative_and_lower_rank():
    C = _corr_block(20, seed=6)
    w = _rng(13).standard_normal(20)
    low = lowrank_ld(C, variance=0.9)
    assert 1 <= low.rank <= 20
    assert low.quad(w) >= 0.0                            # PSD by construction


def test_lowrank_reconstruction_has_unit_diagonal():
    C = _corr_block(10, seed=7)
    low = lowrank_ld(C, variance=0.95)
    recon_diag = np.einsum("ij,ij->i", low.U, low.U)
    assert np.allclose(recon_diag, 1.0, atol=1e-8)


def test_block_diagonal_rejects_overlapping_blocks():
    C = _corr_block(4, seed=8)
    with pytest.raises(ValueError):
        BlockDiagonalLD([(DenseLD(C), [0, 1, 2, 3]),
                         (DenseLD(C), [3, 4, 5, 6])])   # index 3 overlaps


def test_block_diagonal_rejects_size_mismatch():
    C = _corr_block(4, seed=9)
    with pytest.raises(ValueError):
        BlockDiagonalLD([(DenseLD(C), [0, 1, 2])])      # backend.m=4, idx=3


def test_block_diagonal_rejects_uncovered_variants():
    # A gapped partition silently drops the uncovered variants' diagonal mass
    # from w^T D w while they still enter w^T z -- it must fail loudly.
    C = _corr_block(2, seed=10)
    with pytest.raises(ValueError, match="cover"):
        BlockDiagonalLD([(DenseLD(C), [0, 1]), (DenseLD(C), [3, 4])])


def test_lowrank_ld_rejects_asymmetric_input():
    """np.linalg.eigh reads a single triangle, so an asymmetric matrix would be
    silently reinterpreted as its own lower triangle and factored into a
    perfectly valid-looking operator for a *different* matrix."""
    rng = np.random.default_rng(5)
    A = rng.standard_normal((12, 12))
    C = A @ A.T + 0.5 * np.eye(12)
    C = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    lowrank_ld(C)                                   # symmetric: fine
    asym = C.copy()
    asym[0, 1] += 5.0
    with pytest.raises(ValueError, match="symmetric"):
        lowrank_ld(asym)


def test_lowrank_ld_extends_rank_so_every_variant_is_supported():
    """Truncation must not leave a variant with no support in the kept factors.

    Regression: the variance rule dropped a whole basis direction of an identity
    block, leaving that row of U exactly zero. U U^T then had diagonal 0 there,
    quad() silently omitted the variant's self term, and R^2 came out too high
    with no error raised.
    """
    low = lowrank_ld(np.eye(100), variance=0.99)
    assert low.rank == 100                                  # bumped up from 99
    norms = np.sqrt((low.U * low.U).sum(axis=1))
    assert np.allclose(norms, 1.0)                          # unit diagonal restored
    assert low.quad(np.eye(100)[0]) == pytest.approx(1.0)
    assert low.quad(np.ones(100)) == pytest.approx(100.0)


def test_lowrank_ld_supports_a_variant_in_zero_ld_with_the_block():
    """The realistic shape of the same defect: one variant uncorrelated with all
    others, so the leading eigenvectors of the block carry nothing for it."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((30, 30))
    C = A @ A.T + 0.5 * np.eye(30)
    C = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    full = np.eye(31)
    full[:30, :30] = C
    low = lowrank_ld(full, variance=0.99)
    assert np.allclose(np.sqrt((low.U * low.U).sum(axis=1)), 1.0)
    assert low.quad(np.eye(31)[30]) == pytest.approx(1.0)   # the lone variant


def test_lowrank_ld_rejects_non_positive_variance_variants():
    bad = np.eye(5)
    bad[2, 2] = 0.0
    with pytest.raises(ValueError, match="non-positive variance"):
        lowrank_ld(bad)


def test_lowrank_ld_max_rank_is_a_hard_cap():
    """max_rank must be honoured when satisfiable, and raise -- not silently
    exceeded -- when it cannot support every variant."""
    rng = np.random.default_rng(8)
    A = rng.standard_normal((40, 40))
    C = A @ A.T + 0.5 * np.eye(40)
    C = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    assert lowrank_ld(C, variance=0.99, max_rank=12).rank <= 12
    with pytest.raises(ValueError, match="max_rank"):
        lowrank_ld(np.eye(50), variance=0.99, max_rank=10)
