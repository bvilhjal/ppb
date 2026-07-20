"""Tests for the LD-reference store (ppb.ldref)."""

import numpy as np
import pytest

from ppb.harmonize import VariantTable
from ppb.ld_backend import BlockDiagonalLD, DenseLD, DenseLDInt8
from ppb.ldref import read_ldref, write_ldref


def _pd_corr(rng, m):
    X = rng.normal(size=(m, m))
    C = X @ X.T + m * np.eye(m)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


@pytest.fixture
def synth():
    rng = np.random.default_rng(7)
    sizes = [5, 30, 12]
    n = sum(sizes)
    dense = [_pd_corr(rng, m) for m in sizes]
    variants = VariantTable(
        np.array(["2"] * n), np.arange(1000, 1000 + n, dtype=np.int64),
        np.array(["A"] * n), np.array(["G"] * n))
    blocks, start = [], 0
    for m, D in zip(sizes, dense):
        blocks.append((DenseLDInt8.from_dense(D), np.arange(start, start + m)))
        start += m
    w = rng.normal(size=n)
    return variants, blocks, dense, w, sizes


def test_roundtrip_structure_and_quad(tmp_path, synth):
    variants, blocks, dense, w, sizes = synth
    path = tmp_path / "ldref.npz"
    write_ldref(path, variants, blocks,
                rsid=np.array([f"rs{i}" for i in range(variants.n)]),
                af=np.full(variants.n, 0.3),
                pos_hg38=variants.pos + 10)

    out = read_ldref(path)
    assert isinstance(out["ld"], BlockDiagonalLD)
    assert out["ld"].m == variants.n
    assert out["variants"].n == variants.n
    assert np.array_equal(out["variants"].pos, variants.pos)
    assert out["rsid"][3] == "rs3"
    assert out["af"].shape == (variants.n,)
    assert out["pos_hg38"][0] == 1010

    # Exact equality with the in-memory int8 blocks, and close to dense truth.
    direct = sum(b.quad(w[idx]) for b, idx in blocks)
    assert out["ld"].quad(w) == pytest.approx(direct, rel=0, abs=1e-12)
    truth = sum(w[idx] @ D @ w[idx] for (_, idx), D in zip(blocks, dense))
    assert out["ld"].quad(w) == pytest.approx(truth, rel=2e-2)


def test_optional_annotations_absent(tmp_path, synth):
    variants, blocks, *_ = synth
    path = tmp_path / "ldref.npz"
    write_ldref(path, variants, blocks)
    out = read_ldref(path)
    assert "rsid" not in out and "af" not in out and "pos_hg38" not in out


def test_annotation_length_checked(tmp_path, synth):
    variants, blocks, *_ = synth
    with pytest.raises(ValueError, match="af"):
        write_ldref(tmp_path / "x.npz", variants, blocks, af=np.ones(3))


def test_blocks_must_tile_contiguously(tmp_path, synth):
    variants, blocks, *_ = synth
    d8, idx = blocks[0]
    gap = [(d8, idx + 2)]  # leaves [0, 2) uncovered
    with pytest.raises(ValueError, match="contiguous"):
        write_ldref(tmp_path / "x.npz", variants, gap)
    reordered = [(d8, np.arange(idx.size)[::-1])]
    with pytest.raises(ValueError, match="contiguous"):
        write_ldref(tmp_path / "y.npz", variants, reordered)
