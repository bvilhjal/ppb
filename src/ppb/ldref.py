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
from .ld_backend import BlockDiagonalLD, DenseLDInt8


def write_ldref(path, variants: VariantTable, blocks, *, rsid=None, af=None,
                pos_hg38=None):
    """Write an LD-reference ``.npz``.

    ``blocks`` is a sequence of ``(DenseLDInt8, idx)`` pairs whose ``idx`` are
    contiguous runs that, in order, tile ``[0, variants.n)`` exactly (the
    on-disk format stores only block starts and sizes). Optional per-variant
    annotations ``rsid``, ``af``, ``pos_hg38`` must be length ``variants.n``.
    """
    n = variants.n
    starts, sizes, chunks = [], [], []
    expect = 0
    for backend, idx in blocks:
        if not isinstance(backend, DenseLDInt8):
            raise TypeError("ldref blocks must be DenseLDInt8")
        idx = np.asarray(idx, dtype=np.intp)
        m = backend.m
        if idx.shape != (m,) or not np.array_equal(idx, np.arange(expect, expect + m)):
            raise ValueError(
                f"block idx must be the contiguous run [{expect}, {expect + m})")
        starts.append(expect)
        sizes.append(m)
        chunks.append(backend.D8.ravel())
        expect += m
    if expect != n:
        raise ValueError(
            f"blocks tile [0, {expect}) but the variant table has {n} variants")

    arrays = dict(
        chrom=variants.chrom.astype(str), pos=variants.pos.astype(np.int64),
        a1=variants.a1, a2=variants.a2,
        block_starts=np.asarray(starts, dtype=np.int64),
        block_sizes=np.asarray(sizes, dtype=np.int64),
        ld8=np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int8),
    )
    for name, value, dtype in (("rsid", rsid, str), ("af", af, np.float64),
                               ("pos_hg38", pos_hg38, np.int64)):
        if value is not None:
            value = np.asarray(value, dtype=dtype)
            if value.shape != (n,):
                raise ValueError(f"{name} has shape {value.shape}, expected ({n},)")
            arrays[name] = value
    np.savez(path, **arrays)


def read_ldref(path):
    """Read an LD-reference ``.npz``.

    Returns a dict with ``variants`` (:class:`VariantTable`), ``ld``
    (:class:`BlockDiagonalLD` of :class:`DenseLDInt8` blocks), and any stored
    annotations (``rsid``, ``af``, ``pos_hg38``).
    """
    data = np.load(path, allow_pickle=False)
    variants = VariantTable(data["chrom"], data["pos"], data["a1"], data["a2"])
    starts, sizes, ld8 = data["block_starts"], data["block_sizes"], data["ld8"]
    blocks, off = [], 0
    for start, size in zip(starts, sizes):
        m = int(size)
        d8 = ld8[off:off + m * m].reshape(m, m)
        blocks.append((DenseLDInt8(d8), np.arange(int(start), int(start) + m)))
        off += m * m
    if off != ld8.size:
        raise ValueError(
            f"ld8 holds {ld8.size} values but the blocks consume {off}")
    out = {"variants": variants, "ld": BlockDiagonalLD(blocks)}
    for name in ("rsid", "af", "pos_hg38"):
        if name in data:
            out[name] = np.asarray(data[name])
    return out
