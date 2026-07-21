"""Tests for the overlap detector (ppb.overlap) and its simulation validation."""

import numpy as np
import pytest

from ppb.ld_backend import BlockDiagonalLD, DenseLD
from ppb.overlap import block_products, correct_numerator, overlap_slope

from experiments.overlap_detection import run


def _block_ld(m, bs):
    blocks = []
    for s in range(0, m, bs):
        k = min(bs, m - s)
        blocks.append((DenseLD(np.eye(k)), np.arange(s, s + k)))
    return BlockDiagonalLD(blocks)


def test_block_products_matches_direct():
    rng = np.random.default_rng(0)
    m, bs = 30, 10
    ld = _block_ld(m, bs)
    w, z = rng.normal(size=m), rng.normal(size=m)
    u, v = block_products(w, z, ld)
    assert u.shape == v.shape == (3,)
    for b, s in enumerate(range(0, m, bs)):
        assert u[b] == pytest.approx(w[s:s + bs] @ z[s:s + bs])
        assert v[b] == pytest.approx(w[s:s + bs] @ w[s:s + bs])


def test_overlap_slope_recovers_known_slope():
    rng = np.random.default_rng(1)
    n_blocks = 60
    m = rng.integers(50, 150, size=n_blocks).astype(float)
    v = rng.uniform(0.5, 2.0, size=n_blocks)
    gamma_true = 0.01
    u_r = rng.normal(0.0, np.sqrt(v), size=n_blocks)
    u_t = gamma_true * m + rng.normal(0.0, np.sqrt(v), size=n_blocks)
    est = overlap_slope(u_t, u_r, m, v, v, groups=np.arange(n_blocks) % 8)
    assert est.gamma == pytest.approx(gamma_true, rel=0.15)
    assert est.se > 0 and est.n_blocks == n_blocks


def test_correct_numerator():
    assert correct_numerator(2.0, 1e-4, 10000) == pytest.approx(1.0)


def test_simulation_dense_detector_calibrated():
    out = run(reps=2, seed=7)
    assert abs(out["gamma_null"]) < 2e-5                       # null calibrated
    assert out["gamma_low"] == pytest.approx(out["gamma_low_true"], rel=0.35)
    assert out["gamma_full"] == pytest.approx(out["gamma_full_true"], rel=0.35)
    # correction restores the honest anchor even at full overlap
    assert out["r2_corr_full"] == pytest.approx(out["r2_honest_full"], rel=0.4)
    assert out["r2_naive_full"] > 2 * out["r2_honest_full"]


def test_simulation_sparse_detector_blind():
    out = run(reps=2, seed=7)
    # documented failure mode: sparse scores hide overlap from the detector
    assert abs(out["gamma_sparse_full"]) < 0.2 * out["gamma_sparse_full_true"]
