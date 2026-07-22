"""Tests for int8-quantised LD backends (D8 / LR8)."""

import numpy as np
import pytest

from ppb import (
    DenseLD,
    DenseLDInt8,
    LowRankLD,
    LowRankLDInt8,
    PackedDenseLDInt8,
    lowrank_ld,
    quantize_lowrank,
    r2,
)


def _corr_block(k, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((k, k))
    C = A @ A.T + 0.5 * np.eye(k)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def test_dense_int8_matches_float_within_quantization():
    C = _corr_block(30, seed=0)
    w = np.random.default_rng(1).standard_normal(30)
    exact = DenseLD(C).quad(w)
    approx = DenseLDInt8.from_dense(C).quad(w)
    assert abs(approx - exact) <= 0.02 * abs(exact)


def test_dense_int8_diagonal_is_exact():
    C = _corr_block(10, seed=2)
    d8 = DenseLDInt8.from_dense(C)
    for i in range(10):
        e = np.zeros(10); e[i] = 1.0
        assert abs(d8.quad(e) - 1.0) < 1e-12          # 127/127 == 1 exactly


def test_lowrank_int8_matches_float_within_quantization():
    C = _corr_block(40, seed=3)
    w = np.random.default_rng(4).standard_normal(40)
    low = lowrank_ld(C, variance=1.0)                 # full-rank float factor
    q = quantize_lowrank(low)
    assert abs(q.quad(w) - low.quad(w)) <= 0.02 * abs(low.quad(w))
    assert q.quad(w) >= 0.0                            # still PSD


def test_lowrank_int8_is_smaller_than_float():
    C = _corr_block(50, seed=5)
    low = lowrank_ld(C, variance=0.99)
    q = quantize_lowrank(low)
    assert q.U8.dtype == np.int8
    assert q.nbytes * 8 == low.U.nbytes               # int8 vs float64: 8x smaller


def test_r2_via_int8_backends_close_to_float():
    C = _corr_block(40, seed=6)
    rng = np.random.default_rng(7)
    w = rng.standard_normal(40)
    z = rng.standard_normal(40) * 0.1
    exact = r2(w, z, DenseLD(C))
    assert abs(r2(w, z, DenseLDInt8.from_dense(C)) - exact) <= 0.02 * abs(exact)
    q = quantize_lowrank(lowrank_ld(C, variance=1.0))
    assert abs(r2(w, z, q) - exact) <= 0.02 * abs(exact)


def test_int8_rejects_minus_128():
    with pytest.raises(ValueError):
        DenseLDInt8(np.array([[127, -128], [-128, 127]], dtype=np.int8))
    with pytest.raises(ValueError):
        LowRankLDInt8(np.array([[-128]], dtype=np.int8), scale=0.01)


def test_lr8_rejects_invalid_rank_and_zero_rows():
    with pytest.raises(ValueError, match="rank must be"):
        LowRankLDInt8(np.empty((2, 0), dtype=np.int8), scale=0.01)
    with pytest.raises(ValueError, match="rank must be"):
        LowRankLDInt8(np.ones((2, 3), dtype=np.int8), scale=0.01)
    with pytest.raises(ValueError, match="all-zero row"):
        LowRankLDInt8(
            np.array([[1, 0], [0, 0]], dtype=np.int8), scale=0.01)


def test_lr8_quad_is_invariant_to_scale():
    """Row normalisation divides ``scale`` out, so ``quad`` cannot depend on it.

    Documents the contract: ``scale`` dequantises the stored factor
    (``U ~= U8 * scale``) but does not enter the quadratic form.
    """
    rng = np.random.default_rng(11)
    U8 = rng.integers(-127, 128, size=(40, 6)).astype(np.int8)
    w = rng.standard_normal(40)
    base = LowRankLDInt8(U8, scale=1.0).quad(w)
    for scale in (1e-4, 0.25, 7.0, 1e3):
        assert LowRankLDInt8(U8, scale=scale).quad(w) == pytest.approx(base, rel=1e-12)


def test_quantize_lowrank_rejects_non_unit_rows():
    """LR8 re-normalizes rows on every quad, so quantising a factor with
    non-unit rows would return a *different* operator, not a lossy copy.

    Regression: this silently changed w'Dw by ~35x instead of the ~1% the int8
    round trip is documented to cost.
    """
    rng = np.random.default_rng(21)
    U = rng.standard_normal((10, 4)) * 3.0          # rows far from unit norm
    with pytest.raises(ValueError, match="unit-norm rows"):
        quantize_lowrank(LowRankLD(U))


def test_quantize_lowrank_round_trip_is_within_quantisation_error():
    """The legitimate path: a factor from lowrank_ld has unit rows, and the int8
    round trip then agrees with the float factor to quantisation accuracy."""
    rng = np.random.default_rng(22)
    C = _corr_block(48, seed=3)
    low = lowrank_ld(C, variance=1.0)
    w = rng.standard_normal(48)
    exact, approx = DenseLD(C).quad(w), low.quad(w)
    assert approx == pytest.approx(exact, rel=1e-9)     # full rank reproduces D
    assert quantize_lowrank(low).quad(w) == pytest.approx(approx, rel=0.02)


def test_packed_matches_square_and_halves_the_bytes():
    """The packed triangle is the same operator as the square block, at half
    the size -- an LD matrix is symmetric, so the lower triangle is redundant."""
    rng = np.random.default_rng(31)
    C = _corr_block(120, seed=9)
    sq = DenseLDInt8.from_dense(C)
    pk = sq.packed()
    assert pk.nbytes == 120 * 121 // 2
    assert sq.nbytes / pk.nbytes == pytest.approx(2.0, rel=0.02)
    assert np.array_equal(pk.to_dense_int8(), sq.D8)        # lossless
    for _ in range(5):
        w = rng.standard_normal(120)
        assert pk.quad(w) == pytest.approx(sq.quad(w), rel=1e-12)


def test_packed_rejects_a_wrong_length_payload():
    with pytest.raises(ValueError, match="needs"):
        PackedDenseLDInt8(np.zeros(10, dtype=np.int8), 10)   # needs 55, not 10


def test_packed_rejects_a_non_unit_diagonal():
    p8 = np.array([127, 20, 127], dtype=np.int8)
    p8[2] = 1
    with pytest.raises(ValueError, match="diagonal"):
        PackedDenseLDInt8(p8, 2)


def test_lowrank_ld_rejects_materially_indefinite_input():
    C = np.full((3, 3), -0.9)
    np.fill_diagonal(C, 1.0)
    with pytest.raises(ValueError, match="positive semi-definite"):
        lowrank_ld(C)
