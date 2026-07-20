#!/usr/bin/env python
"""Convert bigsnpr's precomputed HapMap3+ European LD reference to ppb ``.npz``.

Source (Privé, doi:10.6084/m9.figshare.21305061, CC BY 4.0): correlations for
1,444,196 HapMap3+ variants from European UK Biobank individuals; pairs beyond
3 cM are 0 and LD blocks are defined so that correlation is 0 outside blocks.
Needed files:

- ``map_hm3_plus.rds``        -- figshare file 37802721
- ``LD_with_blocks_chr{1..22}.rds`` -- from the article's linked Google Drive zip

The R ``dsCMatrix`` serializations are read directly with the ``rdata`` package
(``pip install rdata``; no R installation needed). Per chromosome this writes
``<out_dir>/ldref_chrN.npz`` loadable with :func:`ppb.read_ldref`: the variant
table (bigsnpr ``a1`` -> ppb effect allele ``a1``, ``a0`` -> ``a2``; GRCh37
``pos`` plus ``pos_hg38``; ``rsid``; ``af`` = ``af_UKBB``) and the LD as dense
int8 blocks (ppb D8 convention: ``round(clip(corr, -1, 1) * 127)``,
diagonal 127), keyed by the reference's own ``block_id`` runs.

Usage: python scripts/bigsnpr_ldref_to_ppb.py <data_dir> <out_dir> [chr ...]
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import rdata

from ppb.harmonize import VariantTable
from ppb.ld_backend import DenseLDInt8
from ppb.ldref import write_ldref


def read_dsCMatrix(path):
    """Read an R ``dsCMatrix`` (upper-triangle CSC) -> (i, j, x) with i <= j."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no rdata constructor for dsCMatrix
        obj = rdata.conversion.convert(rdata.parser.parse_file(str(path)))
    i = np.asarray(obj.i, dtype=np.int64)
    p = np.asarray(obj.p, dtype=np.int64)
    x = np.asarray(obj.x, dtype=np.float64)
    j = np.repeat(np.arange(p.size - 1), np.diff(p))
    return i, j, x


def quantize_block(i, j, x, m):
    """Scatter stored upper-triangle values into a dense int8 D8 block."""
    d8 = np.zeros((m, m), dtype=np.int8)
    q = np.clip(np.rint(np.clip(x, -1.0, 1.0) * 127.0), -127, 127).astype(np.int8)
    d8[i, j] = q
    d8[j, i] = q  # symmetric
    np.fill_diagonal(d8, 127)
    return d8


def convert_chr(data_dir: Path, out_dir: Path, chr: int, info) -> None:
    sel = np.flatnonzero(info["chr"].to_numpy() == chr)
    i, j, x = read_dsCMatrix(data_dir / f"LD_with_blocks_chr{chr}.rds")
    if i.size and (i.max() >= sel.size or j.max() >= sel.size):
        raise ValueError(f"chr{chr}: LD matrix larger than the variant table")

    blk = info["block_id"].to_numpy()[sel]
    edges = np.flatnonzero(np.diff(blk) != 0) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [sel.size]])
    if np.unique(blk).size != starts.size:
        raise ValueError(f"chr{chr}: block_id runs are not contiguous")

    blocks = []
    for s, e in zip(starts, ends):
        in_b = (i >= s) & (i < e)
        d8 = quantize_block(i[in_b] - s, j[in_b] - s, x[in_b], e - s)
        blocks.append((DenseLDInt8(d8), np.arange(s, e)))
    blk_of = np.repeat(np.arange(starts.size), ends - starts)
    n_off = int(np.count_nonzero(blk_of[i] != blk_of[j]))

    variants = VariantTable(
        info["chr"].to_numpy()[sel].astype(str), info["pos"].to_numpy()[sel],
        info["a1"].to_numpy()[sel], info["a0"].to_numpy()[sel])
    pos_hg38 = pd_na_to_int(info["pos_hg38"].to_numpy()[sel])
    out = out_dir / f"ldref_chr{chr}.npz"
    write_ldref(out, variants, blocks, rsid=info["rsid"].to_numpy()[sel],
                af=info["af_UKBB"].to_numpy()[sel], pos_hg38=pos_hg38)
    print(f"chr{chr}: {sel.size} variants, {starts.size} blocks, "
          f"{n_off} off-block entries -> {out}", flush=True)


def pd_na_to_int(a):
    """liftOver can leave pos_hg38 missing (NA); keep -1 as the sentinel."""
    a = np.asarray(a)
    return np.where(pd_isna(a), -1, a).astype(np.int64)


def pd_isna(a):
    try:
        import pandas as pd
        return pd.isna(a)
    except ImportError:  # pragma: no cover
        return np.zeros(a.shape, dtype=bool)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: bigsnpr_ldref_to_ppb.py <data_dir> <out_dir> [chr ...]")
    data_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = rdata.conversion.convert(
            rdata.parser.parse_file(str(data_dir / "map_hm3_plus.rds")))
    chrs = [int(c) for c in sys.argv[3:]] or list(range(1, 23))
    for chr in chrs:
        convert_chr(data_dir, out_dir, chr, info)
