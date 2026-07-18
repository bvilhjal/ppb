"""Tests for int8-quantised LD backends (D8 / LR8)."""

import numpy as np
import pytest

from ppb import (
    DenseLD,
    DenseLDInt8,
    LowRankLD,
    LowRankLDInt8,
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
