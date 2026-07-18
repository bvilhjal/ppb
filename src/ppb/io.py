"""File formats for the CLI: PGS weights and benchmark bundles.

- **Weights**: a delimited text table (TSV/CSV) with columns for chromosome,
  position, effect allele, other allele, and the weight. Common PGS Catalog
  column names are recognised. ``#``-prefixed comment lines are skipped.
- **Bundle**: a ``.npz`` archive holding the benchmark's variant table, the target
  summary statistics ``z``, and the LD reference (dense ``D`` or low-rank ``U``).
"""

from __future__ import annotations

import numpy as np

from .harmonize import VariantTable
from .ld_backend import DenseLD, LowRankLD

# Canonical column -> accepted header aliases (lower-cased, '#' stripped).
_WEIGHT_ALIASES = {
    "chrom": ("chrom", "chr", "chr_name", "chromosome", "hm_chr"),
    "pos": ("pos", "position", "chr_position", "bp", "base_pair_location", "hm_pos"),
    "a1": ("a1", "effect_allele", "ea", "allele1", "effectallele"),
    "a2": ("a2", "other_allele", "oa", "noneffect_allele", "allele2", "otherallele"),
    "weight": ("weight", "effect_weight", "beta", "effect_size", "w", "effectweight"),
}


def _resolve_columns(header):
    lut = {name.strip().lower().lstrip("#"): i for i, name in enumerate(header)}
    cols = {}
    for canonical, aliases in _WEIGHT_ALIASES.items():
        for alias in aliases:
            if alias in lut:
                cols[canonical] = lut[alias]
                break
        else:
            raise ValueError(
                f"weights file is missing a '{canonical}' column "
                f"(looked for {aliases}); header was {header}")
    return cols


def read_weights(path):
    """Read a PGS weights file -> ``(VariantTable, weights)``."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        raise ValueError(f"weights file {path!r} has no data rows")
    delim = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else None)
    rows = [ln.split(delim) if delim else ln.split() for ln in lines]
    cols = _resolve_columns(rows[0])

    chrom, pos, a1, a2, weight = [], [], [], [], []
    for r in rows[1:]:
        chrom.append(r[cols["chrom"]].strip())
        pos.append(int(r[cols["pos"]]))
        a1.append(r[cols["a1"]].strip())
        a2.append(r[cols["a2"]].strip())
        weight.append(float(r[cols["weight"]]))
    variants = VariantTable(np.array(chrom), np.array(pos, dtype=np.int64),
                            np.array(a1), np.array(a2))
    return variants, np.array(weight, dtype=np.float64)


def write_bundle(path, variants: VariantTable, z, *, D=None, U=None, var_y=1.0):
    """Write a benchmark bundle ``.npz``. Provide exactly one of ``D`` or ``U``."""
    if (D is None) == (U is None):
        raise ValueError("provide exactly one of D (dense) or U (low-rank factor)")
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (variants.n,):
        raise ValueError(f"z has shape {z.shape}, expected ({variants.n},)")
    arrays = dict(chrom=variants.chrom.astype(str), pos=variants.pos.astype(np.int64),
                  a1=variants.a1, a2=variants.a2, z=z, var_y=np.array(float(var_y)))
    if D is not None:
        arrays["D"] = np.ascontiguousarray(D, dtype=np.float64)
    else:
        arrays["U"] = np.ascontiguousarray(U, dtype=np.float64)
    np.savez(path, **arrays)


def read_bundle(path):
    """Read a benchmark bundle ``.npz`` -> dict with ``variants``, ``z``, ``ld``, ``var_y``."""
    data = np.load(path, allow_pickle=False)
    variants = VariantTable(data["chrom"], data["pos"], data["a1"], data["a2"])
    if "D" in data:
        ld = DenseLD(data["D"])
    elif "U" in data:
        ld = LowRankLD(data["U"])
    else:
        raise ValueError(f"bundle {path!r} has neither a dense 'D' nor low-rank 'U'")
    var_y = float(data["var_y"]) if "var_y" in data else 1.0
    return {"variants": variants, "z": np.asarray(data["z"], dtype=np.float64),
            "ld": ld, "var_y": var_y}
