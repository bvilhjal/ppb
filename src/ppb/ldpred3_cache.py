"""Convert an LDpred3 LD cache into ppb's per-chromosome D8 store.

PPB does not depend on ldpred3. This module reads the published ``.npz``
layout (``save_ld_blocks``) and writes :func:`ppb.ldref.write_ldref` files.
Dense D8 blocks are copied; dense float blocks are requantised. Low-rank
blocks (LR32/LR8) are refused unless the caller expands them. A cache that
already carries ldpred3's identity shrinkage is refused: that transform
inflates ``R²`` (``docs/METHOD.md`` §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .harmonize import VariantTable
from .ld_backend import DenseLDInt8
from .ldref import write_ldref

# LDpred3 block kinds in save_ld_blocks / load_ld_blocks.
_KIND_D32 = 0
_KIND_D8 = 1
_KIND_LR32 = 2
_KIND_LR8 = 3
_Q8 = 127.0


@dataclass
class ConvertReport:
    """What the converter wrote, and what it refused."""

    paths: list[str] = field(default_factory=list)
    n_variants: int = 0
    n_blocks: int = 0
    n_d8: int = 0
    n_float_quantised: int = 0
    n_lr_expanded: int = 0
    chromosomes: list[str] = field(default_factory=list)
    schema_version: int = 0
    n_ref: int | None = None
    shrunk: bool = False


def _as_str1d(value, n, name):
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.size != n:
        raise ValueError(f"{name} must have length {n}")
    return np.asarray([str(x).strip() for x in arr], dtype=object)


def _as_int1d(value, n, name):
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.size != n:
        raise ValueError(f"{name} must have length {n}")
    return np.asarray(arr, dtype=np.int64)


def _flag(store, name, default=False):
    if name not in store.files:
        return default
    value = np.asarray(store[name]).reshape(-1)
    return bool(int(value[0]))


def _scalar_int(store, name, default=None):
    if name not in store.files:
        return default
    return int(np.asarray(store[name]).reshape(-1)[0])


def _require_variant_table(store, n):
    missing = [name for name in ("chrom", "pos", "counted_allele", "other_allele")
               if name not in store.files]
    if missing:
        raise ValueError(
            "LDpred3 cache has no variant table ("
            + ", ".join(missing)
            + "); ppb cannot write an LD reference without chrom, pos, "
            "counted_allele, and other_allele. Rebuild the cache with "
            "ldpred3.save_ld_blocks(..., chrom=, pos=, counted_allele=, "
            "other_allele=)")
    return VariantTable(
        _as_str1d(store["chrom"], n, "chrom"),
        _as_int1d(store["pos"], n, "pos"),
        _as_str1d(store["counted_allele"], n, "counted_allele"),
        _as_str1d(store["other_allele"], n, "other_allele"),
    )


def _load_dense_block(store, i, k, kind, *, allow_lr_expand):
    """Return a square int8 D8 block and a tag of how it was obtained."""
    if kind == _KIND_D8:
        key = f"R{i}"
        if key not in store.files:
            raise ValueError(f"LDpred3 cache is missing dense payload {key}")
        R = np.asarray(store[key])
        if R.shape != (k, k) or R.dtype != np.int8:
            raise ValueError(
                f"block {i} is marked D8 but has shape {R.shape} dtype {R.dtype}")
        return DenseLDInt8(R), "d8"
    if kind == _KIND_D32:
        key = f"R{i}"
        if key not in store.files:
            raise ValueError(f"LDpred3 cache is missing dense payload {key}")
        R = np.asarray(store[key], dtype=np.float64)
        if R.shape != (k, k):
            raise ValueError(
                f"block {i} is marked dense-float but has shape {R.shape}")
        np.fill_diagonal(R, 1.0)
        return DenseLDInt8.from_dense(R), "float"
    if kind in (_KIND_LR32, _KIND_LR8):
        if not allow_lr_expand:
            raise ValueError(
                f"block {i} is an LDpred3 low-rank factor (kind {kind}). "
                "ppb does not implement LR8; refuse, or pass "
                "allow_lr_expand=True to materialise U Uᵀ (+ residual) "
                "and requantise — that is an approximation, not the stored "
                "factor")
        ukey, dkey = f"R{i}_U", f"R{i}_D"
        if ukey not in store.files:
            raise ValueError(f"LDpred3 cache is missing low-rank payload {ukey}")
        U = np.asarray(store[ukey])
        if U.ndim != 2 or U.shape[0] != k:
            raise ValueError(f"block {i} low-rank factor has shape {U.shape}")
        scale = None
        if "scales" in store.files:
            scales = np.asarray(store["scales"], dtype=np.float64)
            if i < scales.size:
                scale = float(scales[i])
        if U.dtype == np.int8:
            if scale is None:
                raise ValueError(
                    f"block {i} is an int8 (LR8) factor but the cache has no "
                    "'scales' entry for it; the quantization scale is "
                    "unrecoverable, so the block cannot be expanded")
            U = U.astype(np.float64) * scale
        else:
            if scale is not None and scale != 1.0:
                raise ValueError(
                    f"block {i} is a float low-rank factor but carries "
                    f"scale={scale}; per-block scales apply to int8 (LR8) "
                    "factors only and would be silently ignored here")
            U = U.astype(np.float64)
        D = U @ U.T
        if dkey in store.files:
            residual = np.asarray(store[dkey], dtype=np.float64)
            if residual.shape != (k,):
                raise ValueError(
                    f"block {i} residual_diag has shape {residual.shape}")
            D[np.diag_indices(k)] += residual
        np.fill_diagonal(D, 1.0)
        return DenseLDInt8.from_dense(D), "lr"
    raise ValueError(f"block {i} has unknown LDpred3 kind {kind}")


def convert_ldpred3_cache(path, out_dir, *, packed: bool = True,
                          allow_lr_expand: bool = False,
                          allow_shrunk: bool = False,
                          compress: bool = False,
                          psd_scan: bool = True) -> ConvertReport:
    """Write per-chromosome ppb LD-reference files from one LDpred3 cache.

    Memory-mapped caches (``ondisk=1``) are refused: reopen them with
    ``ldpred3.load_ld_blocks`` and ``save_ld_blocks(..., mmap=False)``, or
    write a non-mmap cache. Shrinkage toward the identity is refused unless
    ``allow_shrunk`` is set, because ``wᵀ D_a w`` understates the denominator.
    ``psd_scan`` defaults to True, matching :func:`~ppb.ldref.write_ldref`;
    pass ``psd_scan=False`` to skip the Lanczos indefiniteness scan on large
    blocks (a write-time cost).
    """
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = np.load(path, allow_pickle=False)
    try:
        if _flag(store, "ondisk"):
            raise ValueError(
                f"{path} is a memory-mapped LDpred3 cache. ppb reads the "
                "in-archive R{{i}} layout only. Reload with "
                "ldpred3.load_ld_blocks and save_ld_blocks(..., mmap=False)")
        schema = _scalar_int(store, "schema_version", 1)
        if "ids" not in store.files or "sizes" not in store.files:
            raise ValueError(
                f"{path} is not an LDpred3 LD cache (missing ids/sizes)")
        ids = np.asarray(store["ids"])
        sizes = np.asarray(store["sizes"], dtype=np.int64)
        n = int(ids.size)
        if int(sizes.sum()) != n:
            raise ValueError(
                f"LDpred3 cache block sizes sum to {int(sizes.sum())} but "
                f"there are {n} variant ids")
        kinds = (np.asarray(store["kinds"], dtype=np.int64)
                 if "kinds" in store.files
                 else np.zeros(sizes.size, dtype=np.int64))
        if kinds.size != sizes.size:
            raise ValueError("kinds and sizes must have one entry per block")
        shrunk = _flag(store, "ld_shrunk")
        ridge = 0.0
        if "ld_ridge" in store.files:
            ridge = float(np.asarray(store["ld_ridge"]).reshape(-1)[0])
        if (shrunk or ridge != 0.0) and not allow_shrunk:
            raise ValueError(
                "LDpred3 cache applies identity shrinkage or a ridge "
                f"(ld_shrunk={shrunk}, ld_ridge={ridge}). That transform "
                "inflates ppb's R². Rebuild an unshrunk cache, or pass "
                "allow_shrunk=True if you accept the bias")
        variants = _require_variant_table(store, n)
        af = None
        if "reference_af" in store.files:
            af = np.asarray(store["reference_af"], dtype=np.float64)
            if af.shape != (n,):
                raise ValueError("reference_af length must match variant ids")
        rsid = np.asarray([str(x) for x in ids], dtype=object)

        report = ConvertReport(
            n_variants=n, n_blocks=int(sizes.size),
            schema_version=schema, shrunk=shrunk,
            n_ref=_scalar_int(store, "n_ref"),
        )
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        backends = []
        for i, (start, k, kind) in enumerate(zip(starts, sizes, kinds)):
            backend, tag = _load_dense_block(
                store, int(i), int(k), int(kind),
                allow_lr_expand=allow_lr_expand)
            if packed:
                backend = backend.packed()
            backends.append((backend, int(start), int(k), tag))
            if tag == "d8":
                report.n_d8 += 1
            elif tag == "float":
                report.n_float_quantised += 1
            else:
                report.n_lr_expanded += 1

        chrom = variants.norm_chrom
        order = list(dict.fromkeys(chrom.tolist()))
        for label in order:
            mask = chrom == label
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            if np.any(np.diff(idx) != 1):
                raise ValueError(
                    f"chromosome {label!r} is not a contiguous run in the "
                    "LDpred3 cache; ppb shards are one chromosome per file")
            local_start, local_stop = int(idx[0]), int(idx[-1]) + 1
            local_blocks = []
            offset = 0
            for backend, start, k, _tag in backends:
                stop = start + k
                if stop <= local_start or start >= local_stop:
                    continue
                if start < local_start or stop > local_stop:
                    raise ValueError(
                        f"LD block [{start}, {stop}) spans chromosome "
                        f"{label!r}; split the cache on recombination "
                        "blocks that do not cross chromosomes")
                local_blocks.append(
                    (backend, np.arange(offset, offset + k)))
                offset += k
            if offset != idx.size:
                raise ValueError(
                    f"chromosome {label!r}: blocks cover {offset} variants, "
                    f"table has {idx.size}")
            local_variants = VariantTable(
                variants.chrom[mask], variants.pos[mask],
                variants.a1[mask], variants.a2[mask])
            dest = out_dir / f"ldref_chr{label}.npz"
            write_ldref(
                dest, local_variants, local_blocks,
                rsid=rsid[mask],
                af=None if af is None else af[mask],
                compress=compress, psd_scan=psd_scan)
            report.paths.append(str(dest))
            report.chromosomes.append(str(label))
        return report
    finally:
        store.close()
