"""Tests for int8-quantised LD backends (D8, packed and square)."""

import numpy as np
import pytest

from ppb import (
    DenseLD,
    DenseLDInt8,
    PackedDenseLDInt8,
    lowrank_ld,
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


def test_r2_via_int8_backends_close_to_float():
    C = _corr_block(40, seed=6)
    rng = np.random.default_rng(7)
    w = rng.standard_normal(40)
    z = rng.standard_normal(40) * 0.1
    exact = r2(w, z, DenseLD(C))
    assert abs(r2(w, z, DenseLDInt8.from_dense(C)) - exact) <= 0.02 * abs(exact)


def test_int8_rejects_minus_128():
    with pytest.raises(ValueError):
        DenseLDInt8(np.array([[127, -128], [-128, 127]], dtype=np.int8))
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
