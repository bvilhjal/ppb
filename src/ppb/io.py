"""File formats for the CLI: PGS weights and benchmark bundles.

- **Weights**: a delimited text table (TSV/CSV) with columns for chromosome,
  position, effect allele, other allele, and the weight. Common PGS Catalog
  column names are recognised. ``#``-prefixed comment lines are skipped.
- **Bundle**: a ``.npz`` archive holding the benchmark's variant table, the target
  summary statistics ``z``, the LD reference (dense ``D`` or low-rank ``U``),
  and optionally target-cohort genotype SDs for dosage-scale weights.
"""

from __future__ import annotations

import numpy as np

from .harmonize import VariantTable
from .ld_backend import DenseLD, LowRankLD

BUNDLE_VERSION = 2

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
        lines = [(i, ln) for i, ln in enumerate(fh.read().splitlines(), start=1)
                 if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        raise ValueError(f"weights file {path!r} has no data rows")
    delim = "\t" if "\t" in lines[0][1] else ("," if "," in lines[0][1] else None)
    split = (lambda ln: ln.split(delim)) if delim else (lambda ln: ln.split())
    cols = _resolve_columns(split(lines[0][1]))

    need = max(cols.values()) + 1
    chrom, pos, a1, a2, weight = [], [], [], [], []
    for lineno, ln in lines[1:]:
        r = split(ln)
        if len(r) < need:
            raise ValueError(
                f"weights file {path!r} line {lineno}: expected {need} "
                f"fields, got {len(r)}")
        chrom.append(r[cols["chrom"]].strip())
        pos.append(int(r[cols["pos"]]))
        a1.append(r[cols["a1"]].strip())
        a2.append(r[cols["a2"]].strip())
        value = float(r[cols["weight"]])
        if not np.isfinite(value):
            raise ValueError(
                f"weights file {path!r} line {lineno}: weight must be finite")
        weight.append(value)
    variants = VariantTable(np.array(chrom), np.array(pos, dtype=np.int64),
                            np.array(a1), np.array(a2))
    return variants, np.array(weight, dtype=np.float64)


def write_bundle(path, variants: VariantTable, z, *, D=None, U=None, var_y=1.0,
                 genotype_sd=None):
    """Write a versioned benchmark bundle ``.npz``.

    Provide exactly one of ``D`` or ``U``. ``genotype_sd`` is the empirical
    target-cohort SD for each dosage variant. It is required later when a CLI
    submission declares ordinary dosage-scale rather than standardized weights.
    """
    if (D is None) == (U is None):
        raise ValueError("provide exactly one of D (dense) or U (low-rank factor)")
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (variants.n,):
        raise ValueError(f"z has shape {z.shape}, expected ({variants.n},)")
    if not np.isfinite(z).all():
        raise ValueError("z must contain only finite values")
    var_y = float(var_y)
    if not np.isfinite(var_y) or var_y <= 0.0:
        raise ValueError("var_y must be finite and strictly positive")
    operator = np.asarray(D if D is not None else U, dtype=np.float64)
    if not np.isfinite(operator).all():
        raise ValueError("D/U must contain only finite values")
    if D is not None and operator.shape != (variants.n, variants.n):
        raise ValueError(
            f"D has shape {operator.shape}, expected ({variants.n}, {variants.n})")
    if U is not None and (operator.ndim != 2 or operator.shape[0] != variants.n):
        raise ValueError(
            f"U has shape {operator.shape}, expected ({variants.n}, r)")
    arrays = dict(chrom=variants.chrom.astype(str), pos=variants.pos.astype(np.int64),
                  a1=variants.a1, a2=variants.a2, z=z, var_y=np.array(var_y),
                  bundle_version=np.array(BUNDLE_VERSION, dtype=np.int64))
    if genotype_sd is not None:
        genotype_sd = np.asarray(genotype_sd, dtype=np.float64)
        if genotype_sd.shape != (variants.n,):
            raise ValueError(
                f"genotype_sd has shape {genotype_sd.shape}, expected "
                f"({variants.n},)")
        if not np.isfinite(genotype_sd).all() or np.any(genotype_sd <= 0.0):
            raise ValueError("genotype_sd must contain only finite positive values")
        arrays["genotype_sd"] = genotype_sd
    if D is not None:
        arrays["D"] = np.ascontiguousarray(operator)
    else:
        arrays["U"] = np.ascontiguousarray(operator)
    np.savez(path, **arrays)


def read_bundle(path):
    """Read and validate a benchmark bundle.

    Bundles without ``bundle_version`` are legacy version 1 and remain usable
    for explicitly standardized weights. Version 2 may additionally carry
    ``genotype_sd`` for dosage-scale submissions.
    """
    data = np.load(path, allow_pickle=False)
    version = int(data["bundle_version"]) if "bundle_version" in data else 1
    if version not in (1, BUNDLE_VERSION):
        raise ValueError(f"bundle {path!r} uses unsupported version {version}")
    variants = VariantTable(data["chrom"], data["pos"], data["a1"], data["a2"])
    if "D" in data:
        operator = np.asarray(data["D"], dtype=np.float64)
        if not np.isfinite(operator).all():
            raise ValueError(f"bundle {path!r} has non-finite values in D")
        ld = DenseLD(operator)
    elif "U" in data:
        operator = np.asarray(data["U"], dtype=np.float64)
        if not np.isfinite(operator).all():
            raise ValueError(f"bundle {path!r} has non-finite values in U")
        ld = LowRankLD(operator)
    else:
        raise ValueError(f"bundle {path!r} has neither a dense 'D' nor low-rank 'U'")
    if ld.m != variants.n:
        raise ValueError(
            f"bundle {path!r} has {variants.n} variants but LD has m={ld.m}")
    z = np.asarray(data["z"], dtype=np.float64)
    if z.shape != (variants.n,) or not np.isfinite(z).all():
        raise ValueError(
            f"bundle {path!r} z must be a finite vector of length {variants.n}")
    var_y = float(data["var_y"]) if "var_y" in data else 1.0
    if not np.isfinite(var_y) or var_y <= 0.0:
        raise ValueError(f"bundle {path!r} var_y must be finite and strictly positive")
    genotype_sd = None
    if "genotype_sd" in data:
        genotype_sd = np.asarray(data["genotype_sd"], dtype=np.float64)
        if (genotype_sd.shape != (variants.n,)
                or not np.isfinite(genotype_sd).all()
                or np.any(genotype_sd <= 0.0)):
            raise ValueError(
                f"bundle {path!r} genotype_sd must be a finite positive vector "
                f"of length {variants.n}")
    return {"variants": variants, "z": z, "ld": ld, "var_y": var_y,
            "genotype_sd": genotype_sd, "bundle_version": version}
