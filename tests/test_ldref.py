"""Tests for the LD-reference store (ppb.ldref)."""

import numpy as np
import pytest

from ppb.harmonize import VariantTable
from ppb.ld_backend import (
    BlockDiagonalLD, DenseLD, DenseLDInt8, PackedDenseLDInt8)
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


def _packed(blocks):
    return [(b.packed(), i) for b, i in blocks]


def test_v1_layout_is_unchanged_by_v2_support(tmp_path, synth):
    """Square-only references must still be written in the v1 layout so older
    readers keep working; v2 arrays appear only when a packed block does."""
    variants, blocks, _dense, _w, _sizes = synth
    path = tmp_path / "v1.npz"
    write_ldref(path, variants, blocks)
    stored = set(np.load(path).files)
    assert not stored & {"format_version", "block_kind", "block_offset", "ld8p"}
    assert read_ldref(path)["format_version"] == 1


def test_packed_round_trip_halves_the_store_and_preserves_quad(tmp_path, synth):
    variants, blocks, _dense, w, _sizes = synth
    sq, pk = tmp_path / "sq.npz", tmp_path / "pk.npz"
    write_ldref(sq, variants, blocks)
    write_ldref(pk, variants, _packed(blocks))

    ref_sq, ref_pk = read_ldref(sq), read_ldref(pk)
    assert ref_pk["format_version"] == 2
    assert all(isinstance(b, PackedDenseLDInt8) for b, _ in ref_pk["ld"].blocks)

    bytes_sq = sum(b.nbytes for b, _ in ref_sq["ld"].blocks)
    bytes_pk = sum(b.nbytes for b, _ in ref_pk["ld"].blocks)
    assert bytes_pk < bytes_sq                          # m(m+1)/2 vs m^2

    # Not bit-identical -- the packed kernel doubles one triangle rather than
    # summing two -- but far inside int8 quantisation's own ~0.1% error.
    assert ref_pk["ld"].quad(w) == pytest.approx(ref_sq["ld"].quad(w), rel=1e-12)


def test_packed_blocks_preserve_the_int8_values_exactly(tmp_path, synth):
    """Packing is lossless: only the redundant lower triangle is dropped."""
    variants, blocks, _dense, _w, _sizes = synth
    path = tmp_path / "pk.npz"
    write_ldref(path, variants, _packed(blocks))
    got = read_ldref(path)["ld"].blocks
    for (orig, _), (packed, _) in zip(blocks, got):
        assert np.array_equal(packed.to_dense_int8(), orig.D8)


def test_mixed_square_and_packed_blocks_round_trip(tmp_path, synth):
    variants, blocks, _dense, w, _sizes = synth
    mixed = [blocks[0], (blocks[1][0].packed(), blocks[1][1]), blocks[2]]
    path = tmp_path / "mixed.npz"
    write_ldref(path, variants, mixed)

    got = read_ldref(path)
    assert [type(b).__name__ for b, _ in got["ld"].blocks] == [
        "DenseLDInt8", "PackedDenseLDInt8", "DenseLDInt8"]

    write_ldref(tmp_path / "allsq.npz", variants, blocks)
    assert got["ld"].quad(w) == pytest.approx(
        read_ldref(tmp_path / "allsq.npz")["ld"].quad(w), rel=1e-12)


def test_write_ldref_rejects_asymmetric_and_bad_diagonal_blocks(tmp_path, synth):
    """The converter builds squares by scattering a triangle and filling the
    diagonal, so those are the corruptions to expect -- and they would otherwise
    load cleanly and yield a wrong D."""
    variants, blocks, _dense, _w, _sizes = synth
    (b0, i0) = blocks[0]

    asym = b0.D8.copy()
    asym[0, 1] = asym[1, 0] + 3
    with pytest.raises(ValueError, match="not symmetric"):
        write_ldref(tmp_path / "a.npz", variants, [(DenseLDInt8(asym), i0)] + blocks[1:])

    baddiag = b0.D8.copy()
    baddiag[2, 2] = 100
    with pytest.raises(ValueError, match="diagonal"):
        write_ldref(tmp_path / "b.npz", variants, [(DenseLDInt8(baddiag), i0)] + blocks[1:])

    # ...and validate=False is the escape hatch for a deliberately odd block.
    write_ldref(tmp_path / "c.npz", variants,
                [(DenseLDInt8(baddiag), i0)] + blocks[1:], validate=False)


def test_compressed_reference_reads_back_identically(tmp_path, synth):
    variants, blocks, _dense, w, _sizes = synth
    raw, comp = tmp_path / "raw.npz", tmp_path / "comp.npz"
    write_ldref(raw, variants, _packed(blocks))
    write_ldref(comp, variants, _packed(blocks), compress=True)
    assert read_ldref(comp)["ld"].quad(w) == read_ldref(raw)["ld"].quad(w)
