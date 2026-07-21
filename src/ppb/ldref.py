"""LD-reference store: block-diagonal int8 LD plus a variant table.

One ``.npz`` per chromosome. Unlike a benchmark *bundle* (``ppb.io``), an LD
reference carries no summary statistics -- it is the reusable ``D`` half of the
estimator, to be combined with target-ancestry ``z`` at evaluation time.

Schema (arrays in the ``.npz``):

- variant table: ``chrom``, ``pos``, ``a1`` (effect allele), ``a2``;
  optional annotations ``rsid``, ``af`` (allele frequency; used only through
  the symmetric ``2 f (1 - f)``), ``pos_hg38``;
- block structure: ``block_starts``, ``block_sizes`` -- contiguous runs that
  must tile ``[0, n)`` exactly;
- ``ld8``: concatenated per-block dense int8 LD matrices (C-order), each
  ``round(corr * 127)`` with diagonal 127 -- the :class:`DenseLDInt8` store.

The reference converted from bigsnpr's HapMap3+ European LD (see
``scripts/bigsnpr_ldref_to_ppb.py``) is written in this format.
"""

from __future__ import annotations

import numpy as np

from .harmonize import VariantTable
from .ld_backend import BlockDiagonalLD, DenseLDInt8, PackedDenseLDInt8


_KIND_SQUARE = 0        # full m x m int8 block, in `ld8`   (the v1 layout)
_KIND_PACKED = 2        # packed upper triangle, in `ld8p`
#                   1 is reserved for an int8 low-rank factor; see docs/METHOD.md.


def _validate_square(backend, b):
    """Structural checks on a square int8 block, run on every write.

    The converter builds these by scattering an upper triangle into a square and
    filling the diagonal, so an asymmetric or wrong-diagonal block is exactly the
    corruption shape to expect -- and one that would otherwise load cleanly and
    silently yield a wrong ``D``.
    """
    D8 = backend.D8
    if not np.array_equal(D8, D8.T):
        raise ValueError(f"block {b} is not symmetric; LD must be")
    bad = np.flatnonzero(np.diag(D8) != 127)
    if bad.size:
        raise ValueError(
            f"block {b} has {bad.size} diagonal entry/entries != 127 (e.g. index "
            f"{int(bad[0])} = {int(np.diag(D8)[bad[0]])}); the int8 LD diagonal "
            "must dequantise to exactly 1")


def write_ldref(path, variants: VariantTable, blocks, *, rsid=None, af=None,
                pos_hg38=None, compress=False, validate=True):
    """Write an LD-reference ``.npz``.

    ``blocks`` is a sequence of ``(backend, idx)`` pairs -- each backend a
    :class:`DenseLDInt8` or a :class:`PackedDenseLDInt8` -- whose ``idx`` are
    contiguous runs that, in order, tile ``[0, variants.n)`` exactly (the
    on-disk format stores only block starts and sizes). Optional per-variant
    annotations ``rsid``, ``af``, ``pos_hg38`` must be length ``variants.n``.

    Files containing only square blocks are written in the **v1** layout, byte
    for byte as before, so existing readers keep working. A packed block
    promotes the file to **v2**, which adds ``format_version``, ``block_kind``
    and ``block_offset`` and carries triangles in ``ld8p``. ``compress`` selects
    ``savez_compressed``. ``validate`` runs :func:`_validate_square` on square
    blocks (symmetry, unit diagonal); packed blocks carry only a triangle, so
    symmetry is structural and only the diagonal is checkable.
    """
    n = variants.n
    starts, sizes, kinds, offsets = [], [], [], []
    square_chunks, packed_chunks = [], []
    used = {_KIND_SQUARE: 0, _KIND_PACKED: 0}
    expect = 0
    for b, (backend, idx) in enumerate(blocks):
        if isinstance(backend, PackedDenseLDInt8):
            kind, payload = _KIND_PACKED, backend.p8
        elif isinstance(backend, DenseLDInt8):
            kind, payload = _KIND_SQUARE, backend.D8.ravel()
            if validate:
                _validate_square(backend, b)
        else:
            raise TypeError(
                "ldref blocks must be DenseLDInt8 or PackedDenseLDInt8; got "
                f"{type(backend).__name__} for block {b}")
        idx = np.asarray(idx, dtype=np.intp)
        m = backend.m
        if idx.shape != (m,) or not np.array_equal(idx, np.arange(expect, expect + m)):
            raise ValueError(
                f"block idx must be the contiguous run [{expect}, {expect + m})")
        starts.append(expect)
        sizes.append(m)
        kinds.append(kind)
        offsets.append(used[kind])
        used[kind] += payload.size
        (packed_chunks if kind == _KIND_PACKED else square_chunks).append(payload)
        expect += m
    if expect != n:
        raise ValueError(
            f"blocks tile [0, {expect}) but the variant table has {n} variants")

    def _cat(chunks):
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int8)

    arrays = dict(
        chrom=variants.chrom.astype(str), pos=variants.pos.astype(np.int64),
        a1=variants.a1, a2=variants.a2,
        block_starts=np.asarray(starts, dtype=np.int64),
        block_sizes=np.asarray(sizes, dtype=np.int64),
        ld8=_cat(square_chunks),
    )
    if packed_chunks:                       # only then does the file need v2
        arrays["format_version"] = np.asarray(2, dtype=np.int64)
        arrays["block_kind"] = np.asarray(kinds, dtype=np.int8)
        arrays["block_offset"] = np.asarray(offsets, dtype=np.int64)
        arrays["ld8p"] = _cat(packed_chunks)
    for name, value, dtype in (("rsid", rsid, str), ("af", af, np.float64),
                               ("pos_hg38", pos_hg38, np.int64)):
        if value is not None:
            value = np.asarray(value, dtype=dtype)
            if value.shape != (n,):
                raise ValueError(f"{name} has shape {value.shape}, expected ({n},)")
            arrays[name] = value
    (np.savez_compressed if compress else np.savez)(path, **arrays)


def read_ldref(path):
    """Read an LD-reference ``.npz`` (v1 or v2).

    Returns a dict with ``variants`` (:class:`VariantTable`), ``ld``
    (:class:`BlockDiagonalLD` of :class:`DenseLDInt8` and/or
    :class:`PackedDenseLDInt8` blocks), and any stored annotations (``rsid``,
    ``af``, ``pos_hg38``).

    Every v2 array has a default that reproduces the v1 parse exactly, so a v1
    file reads identically here to how it always did.
    """
    data = np.load(path, allow_pickle=False)
    variants = VariantTable(data["chrom"], data["pos"], data["a1"], data["a2"])
    starts, sizes, ld8 = data["block_starts"], data["block_sizes"], data["ld8"]
    nb = len(sizes)
    version = int(data["format_version"]) if "format_version" in data else 1
    if version > 2:
        raise ValueError(
            f"{path!r} is LD-reference format v{version}; this ppb reads v1 and v2")
    ld8p = data["ld8p"] if "ld8p" in data else np.empty(0, dtype=np.int8)
    kinds = (data["block_kind"] if "block_kind" in data
             else np.full(nb, _KIND_SQUARE, dtype=np.int8))
    if "block_offset" in data:
        offsets = data["block_offset"]
    else:                                   # v1: squares packed back to back
        offsets = np.concatenate([[0], np.cumsum(np.asarray(sizes, np.int64) ** 2)[:-1]])
    if len(kinds) != nb or len(offsets) != nb:
        raise ValueError(
            f"block_kind/block_offset must have one entry per block ({nb})")

    blocks, used = [], {_KIND_SQUARE: 0, _KIND_PACKED: 0}
    for start, size, kind, off in zip(starts, sizes, kinds, offsets):
        m, kind, off = int(size), int(kind), int(off)
        idx = np.arange(int(start), int(start) + m)
        if kind == _KIND_SQUARE:
            need = m * m
            chunk = ld8[off:off + need]
            if chunk.size != need:
                raise ValueError(
                    f"block at {start} needs {need} square entries, found {chunk.size}")
            blocks.append((DenseLDInt8(chunk.reshape(m, m)), idx))
        elif kind == _KIND_PACKED:
            need = m * (m + 1) // 2
            chunk = ld8p[off:off + need]
            if chunk.size != need:
                raise ValueError(
                    f"block at {start} needs {need} packed entries, found {chunk.size}")
            blocks.append((PackedDenseLDInt8(chunk, m), idx))
        else:
            raise ValueError(
                f"block at {start} has unknown representation {kind}; this ppb "
                "stores 0 (square int8) and 2 (packed upper triangle)")
        used[kind] += need
    # Catch a payload longer than the blocks consume, which would otherwise mean
    # blocks were silently read from the wrong offsets.
    if used[_KIND_SQUARE] != ld8.size or used[_KIND_PACKED] != ld8p.size:
        raise ValueError(
            f"payload/block mismatch: ld8 holds {ld8.size} (blocks consume "
            f"{used[_KIND_SQUARE]}), ld8p holds {ld8p.size} (blocks consume "
            f"{used[_KIND_PACKED]})")
    out = {"variants": variants, "ld": BlockDiagonalLD(blocks), "format_version": version}
    for name in ("rsid", "af", "pos_hg38"):
        if name in data:
            out[name] = np.asarray(data[name])
    return out
