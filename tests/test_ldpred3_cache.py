"""LDpred3 cache → ppb ldref converter."""

import numpy as np
import pytest

from ppb import (
    DenseLDInt8, convert_ldpred3_cache, read_ldref,
)


def _d8_eye(m):
    return DenseLDInt8.from_dense(np.eye(m)).D8


def _write_cache(path, *, kinds=None, residual=False, shrunk=False,
                 ridge=0.0, mmap=False, drop_alleles=False,
                 scales=(1.0, 1.0)):
    """A two-chromosome, two-block non-mmap LDpred3 cache."""
    sizes = np.array([2, 2], dtype=np.int64)
    if kinds is None:
        kinds = np.array([1, 1], dtype=np.int8)
    arrays = dict(
        schema_version=np.array([6], dtype=np.int16),
        ondisk=np.array([1 if mmap else 0], dtype=np.int8),
        ids=np.array(["rs1", "rs2", "rs3", "rs4"]),
        sizes=sizes,
        kinds=np.asarray(kinds, dtype=np.int8),
        provenance_complete=np.array([0], dtype=np.int8),
        ld_shrunk=np.array([1 if shrunk else 0], dtype=np.int8),
        ld_ridge=np.array([ridge], dtype=np.float64),
        chrom=np.array(["1", "1", "2", "2"]),
        pos=np.array([10, 20, 10, 20], dtype=np.int64),
        reference_af=np.array([0.1, 0.2, 0.3, 0.4]),
        n_ref=np.array([500], dtype=np.int64),
    )
    if scales is not None:
        arrays["scales"] = np.asarray(scales, dtype=np.float64)
    if not drop_alleles:
        arrays["counted_allele"] = np.array(["A", "A", "C", "C"])
        arrays["other_allele"] = np.array(["G", "G", "T", "T"])
    for i, kind in enumerate(kinds):
        k = int(sizes[i])
        if kind == 1:
            arrays[f"R{i}"] = _d8_eye(k)
        elif kind == 0:
            arrays[f"R{i}"] = np.eye(k, dtype=np.float32)
        elif kind in (2, 3):
            U = np.zeros((k, 1), dtype=np.float32 if kind == 2 else np.int8)
            if kind == 2:
                U[:, 0] = 1.0
            else:
                U[:, 0] = 127
            arrays[f"R{i}_U"] = U
            if residual or kind == 3:
                arrays[f"R{i}_D"] = np.zeros(k, dtype=np.float32)
        else:
            raise ValueError(kind)
    np.savez(path, **arrays)


def test_converts_d8_cache_into_chromosome_shards(tmp_path):
    cache = tmp_path / "ld.npz"
    _write_cache(cache)
    report = convert_ldpred3_cache(cache, tmp_path / "out", packed=True,
                                   psd_scan=False)
    assert report.n_d8 == 2
    assert report.chromosomes == ["1", "2"]
    assert report.n_variants == 4
    for chrom, af0 in (("1", 0.1), ("2", 0.3)):
        ref = read_ldref(tmp_path / "out" / f"ldref_chr{chrom}.npz")
        assert ref["variants"].n == 2
        assert ref["af"][0] == pytest.approx(af0)
        w = np.array([1.0, 0.0])
        assert ref["ld"].quad(w) == pytest.approx(1.0)


def test_quantises_float_dense_blocks(tmp_path):
    cache = tmp_path / "ld.npz"
    _write_cache(cache, kinds=[0, 0])
    report = convert_ldpred3_cache(cache, tmp_path / "out", packed=False,
                                   psd_scan=False)
    assert report.n_float_quantised == 2
    ref = read_ldref(tmp_path / "out" / "ldref_chr1.npz")
    assert ref["ld"].quad(np.array([1.0, 1.0])) == pytest.approx(2.0)


def test_refuses_low_rank_unless_expanded(tmp_path):
    cache = tmp_path / "ld.npz"
    _write_cache(cache, kinds=[2, 2], residual=True)
    with pytest.raises(ValueError, match="low-rank"):
        convert_ldpred3_cache(cache, tmp_path / "out", psd_scan=False)
    report = convert_ldpred3_cache(
        cache, tmp_path / "out", allow_lr_expand=True, packed=False,
        psd_scan=False)
    assert report.n_lr_expanded == 2


def test_refuses_lr8_without_block_scales(tmp_path):
    """An int8 factor without its quantization scale is unrecoverable."""
    cache = tmp_path / "ld.npz"
    _write_cache(cache, kinds=[3, 3], scales=None)
    with pytest.raises(ValueError, match="scales"):
        convert_ldpred3_cache(cache, tmp_path / "out", allow_lr_expand=True,
                              psd_scan=False)
    _write_cache(cache, kinds=[3, 3], scales=(1.0,))      # short: no entry for block 1
    with pytest.raises(ValueError, match="scales"):
        convert_ldpred3_cache(cache, tmp_path / "out", allow_lr_expand=True,
                              psd_scan=False)


def test_refuses_float_factor_with_nonunit_scale(tmp_path):
    """A per-block scale on a float (LR32) factor would be silently dropped."""
    cache = tmp_path / "ld.npz"
    _write_cache(cache, kinds=[2, 2], residual=True, scales=(2.0, 1.0))
    with pytest.raises(ValueError, match="scale=2.0"):
        convert_ldpred3_cache(cache, tmp_path / "out", allow_lr_expand=True,
                              psd_scan=False)


def test_refuses_shrunk_cache(tmp_path):
    cache = tmp_path / "ld.npz"
    _write_cache(cache, shrunk=True)
    with pytest.raises(ValueError, match="shrinkage"):
        convert_ldpred3_cache(cache, tmp_path / "out", psd_scan=False)
    convert_ldpred3_cache(
        cache, tmp_path / "out", allow_shrunk=True, psd_scan=False)


def test_refuses_mmap_and_missing_alleles(tmp_path):
    cache = tmp_path / "ld.npz"
    _write_cache(cache, mmap=True)
    with pytest.raises(ValueError, match="memory-mapped"):
        convert_ldpred3_cache(cache, tmp_path / "out", psd_scan=False)
    _write_cache(cache, drop_alleles=True)
    with pytest.raises(ValueError, match="variant table"):
        convert_ldpred3_cache(cache, tmp_path / "out", psd_scan=False)
