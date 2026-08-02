"""File formats and file-backed evaluation paths for the CLI.

- **Weights**: a delimited text table (TSV/CSV) with columns for chromosome,
  position, effect allele, other allele, and the weight. Common PGS Catalog
  column names are recognised. ``#``-prefixed comment lines are skipped.
- **Summary statistics**: the same four variant columns plus standardized
  marginal ``z``. An optional empirical ``genotype_sd`` column puts ordinary
  dosage weights on the standardized-genotype scale.
- **Bundle**: a ``.npz`` archive holding the benchmark's variant table, the target
  summary statistics ``z``, the LD reference (dense ``D`` or low-rank ``U``),
  and optionally target-cohort genotype SDs for dosage-scale weights.
- **LD-reference shards**: one block-int8 :func:`~ppb.ldref.read_ldref` file per
  chromosome. :func:`evaluate_ldrefs` evaluates these one at a time, so a
  genome-wide run never materializes a dense or low-rank whole-genome bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .estimator import _mse_from_quad, _r2_from_quad, _var_y
from .evaluate import EvaluationResult
from .harmonize import (
    HarmonizeReport,
    VariantTable,
    _harmonize_to_details,
    harmonize_to,
)
from .ld_backend import DenseLD, LowRankLD
from .ldref import read_ldref

BUNDLE_VERSION = 2

# Canonical column -> accepted header aliases (lower-cased, '#' stripped).
_VARIANT_ALIASES = {
    "chrom": ("chrom", "chr", "chr_name", "chromosome", "hm_chr"),
    "pos": ("pos", "position", "chr_position", "bp", "base_pair_location", "hm_pos"),
    "a1": ("a1", "effect_allele", "ea", "allele1", "effectallele"),
    "a2": ("a2", "other_allele", "oa", "noneffect_allele", "allele2", "otherallele"),
}
_WEIGHT_ALIASES = {
    **_VARIANT_ALIASES,
    "weight": ("weight", "effect_weight", "beta", "effect_size", "w", "effectweight"),
}
_SUMSTATS_ALIASES = {**_VARIANT_ALIASES, "z": ("z",)}


def _resolve_columns(header, aliases, table_name):
    lut = {name.strip().lower().lstrip("#"): i for i, name in enumerate(header)}
    cols = {}
    for canonical, choices in aliases.items():
        for alias in choices:
            if alias in lut:
                cols[canonical] = lut[alias]
                break
        else:
            raise ValueError(
                f"{table_name} file is missing a '{canonical}' column "
                f"(looked for {choices}); header was {header}")
    return cols


def _read_value_table(path, *, aliases, value_name, table_name):
    """Read one finite numeric value beside the canonical variant columns."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = (
            (i, ln.rstrip("\r\n"))
            for i, ln in enumerate(fh, start=1)
            if ln.strip() and not ln.lstrip().startswith("#")
        )
        try:
            _, header_line = next(lines)
        except StopIteration:
            raise ValueError(f"{table_name} file {path!r} has no header") from None
        delim = ("\t" if "\t" in header_line
                 else ("," if "," in header_line else None))
        split = (lambda ln: ln.split(delim)) if delim else (lambda ln: ln.split())
        cols = _resolve_columns(split(header_line), aliases, table_name)
        need = max(cols.values()) + 1
        chrom, pos, a1, a2, value = [], [], [], [], []
        for lineno, ln in lines:
            r = split(ln)
            if len(r) < need:
                raise ValueError(
                    f"{table_name} file {path!r} line {lineno}: expected {need} "
                    f"fields, got {len(r)}")
            chrom.append(r[cols["chrom"]].strip())
            pos.append(int(r[cols["pos"]]))
            a1.append(r[cols["a1"]].strip())
            a2.append(r[cols["a2"]].strip())
            x = float(r[cols[value_name]])
            if not np.isfinite(x):
                raise ValueError(
                    f"{table_name} file {path!r} line {lineno}: "
                    f"{value_name} must be finite")
            value.append(x)
    if not value:
        raise ValueError(f"{table_name} file {path!r} has no data rows")
    variants = VariantTable(chrom, pos, a1, a2)
    return variants, np.array(value, dtype=np.float64)


def read_weights(path):
    """Read a PGS weights file -> ``(VariantTable, weights)``.

    The file is consumed line by line, so a million-row file does not coexist
    with a second, raw-text copy of itself.
    """
    return _read_value_table(
        path, aliases=_WEIGHT_ALIASES, value_name="weight",
        table_name="weights")


def read_sumstats(path, *, read_genotype_sd: bool = True):
    """Read a standardized-z table.

    Required columns are ``chrom``, ``pos``, ``a1``, ``a2``, and exactly ``z``.
    No beta/SE/N conversion is inferred. An optional empirical target-cohort
    ``genotype_sd`` column may be supplied on the same rows. Set
    ``read_genotype_sd=False`` when that column is deliberately irrelevant
    (standardized weights or an explicit HWE approximation); it is then neither
    parsed nor validated.

    Returns ``(variants, z, genotype_sd)``; the last item is ``None`` when the
    optional column is absent.
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = (
            (i, ln.rstrip("\r\n"))
            for i, ln in enumerate(fh, start=1)
            if ln.strip() and not ln.lstrip().startswith("#")
        )
        try:
            _, header_line = next(lines)
        except StopIteration:
            raise ValueError(f"sumstats file {path!r} has no header") from None
        delim = ("\t" if "\t" in header_line
                 else ("," if "," in header_line else None))
        split = (lambda ln: ln.split(delim)) if delim else (lambda ln: ln.split())
        header = split(header_line)
        cols = _resolve_columns(header, _SUMSTATS_ALIASES, "sumstats")
        lut = {name.strip().lower().lstrip("#"): i
               for i, name in enumerate(header)}
        sd_col = lut.get("genotype_sd") if read_genotype_sd else None
        need = max([*cols.values(), *([] if sd_col is None else [sd_col])]) + 1

        chrom, pos, a1, a2, z, genotype_sd = [], [], [], [], [], []
        for lineno, ln in lines:
            r = split(ln)
            if len(r) < need:
                raise ValueError(
                    f"sumstats file {path!r} line {lineno}: expected {need} "
                    f"fields, got {len(r)}")
            chrom.append(r[cols["chrom"]].strip())
            pos.append(int(r[cols["pos"]]))
            a1.append(r[cols["a1"]].strip())
            a2.append(r[cols["a2"]].strip())
            value = float(r[cols["z"]])
            if not np.isfinite(value):
                raise ValueError(
                    f"sumstats file {path!r} line {lineno}: z must be finite")
            z.append(value)
            if sd_col is not None:
                value = float(r[sd_col])
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"sumstats file {path!r} line {lineno}: genotype_sd "
                        "must be finite and strictly positive")
                genotype_sd.append(value)
    if not z:
        raise ValueError(f"sumstats file {path!r} has no data rows")
    variants = VariantTable(chrom, pos, a1, a2)
    sd = (np.array(genotype_sd, dtype=np.float64)
          if sd_col is not None else None)
    return variants, np.array(z, dtype=np.float64), sd


@dataclass
class LDRefEvaluationResult(EvaluationResult):
    """An :class:`EvaluationResult` with sharded-input provenance."""

    source: str = "ldref_shards"
    n_ldref_files: int = 0
    chromosomes: tuple[str, ...] = field(default_factory=tuple)
    genotype_sd_source: str = "not_used"


_REPORT_COUNTS = (
    "n_reference", "n_target", "n_matched", "n_sign_flipped",
    "n_strand_flipped", "n_ambiguous_removed", "n_mismatch", "n_unmatched",
)


def _chromosome_rows(variants):
    """Map canonical chromosome to row selector without copying table columns.

    Genome-wide files are normally chromosome-grouped, in which case every
    selector is a zero-copy slice. An interleaved file remains valid and falls
    back to one integer index array for each repeated chromosome.
    """
    chrom = variants.norm_chrom
    if variants.n == 0:
        return {}
    stops = np.flatnonzero(chrom[1:] != chrom[:-1]) + 1
    starts = np.concatenate(([0], stops))
    stops = np.concatenate((stops, [variants.n]))
    out = {}
    repeated = set()
    for start, stop in zip(starts.tolist(), stops.tolist()):
        key = str(chrom[start])
        if key in out:
            repeated.add(key)
        else:
            out[key] = slice(start, stop)
    for key in repeated:
        out[key] = np.flatnonzero(chrom == key)
    return out


def _subset_variant_values(variants, values, selector):
    """Build the transient chromosome table consumed by harmonization."""
    if selector is None:
        selector = slice(0, 0)
    return (
        VariantTable(
            variants.chrom[selector], variants.pos[selector],
            variants.a1[selector], variants.a2[selector]),
        values[selector],
    )


def _add_report(total, report):
    for name in _REPORT_COUNTS:
        total[name] += int(getattr(report, name))


def _missing_rows(groups, seen, n):
    """Number of input rows on chromosomes absent from the LD references."""
    missing = 0
    for chrom, selector in groups.items():
        if chrom in seen:
            continue
        if isinstance(selector, slice):
            missing += len(range(*selector.indices(n)))
        else:
            missing += int(np.asarray(selector).size)
    return missing


def evaluate_ldrefs(
        ldref_paths, weights_variants: VariantTable, weights,
        sumstats_variants: VariantTable, z, *, var_y: float = 1.0,
        weight_scale: str = "standardized", genotype_sd=None,
        hwe_genotype_sd: bool = False, remove_ambiguous: bool = True,
        mse_interpretable: bool | None = None) -> LDRefEvaluationResult:
    """Evaluate chromosome-sharded block-int8 LD references in one pass.

    Each path must be one single-chromosome file accepted by
    :func:`~ppb.ldref.read_ldref`; duplicate chromosomes are rejected. Weights
    and summary statistics are partitioned by chromosome once, then only that
    chromosome's rows are harmonized. The global numerator ``w'z`` and
    denominator ``w'Dw`` are summed before R2 and MSE are formed. Chromosome R2
    values are never averaged.

    ``genotype_sd`` is an empirical target-cohort vector in
    ``sumstats_variants`` order. As an explicit approximation,
    ``hwe_genotype_sd=True`` instead uses ``sqrt(2*af*(1-af))`` from each LD
    reference. Exactly one is required for dosage weights; neither is used for
    standardized weights. ``mse_interpretable`` follows ``weight_scale`` by
    default and may be overridden only when the weight and phenotype scales are
    known independently.
    """
    if isinstance(ldref_paths, (str, Path)):
        ldref_paths = [ldref_paths]
    else:
        ldref_paths = list(ldref_paths)
    if not ldref_paths:
        raise ValueError("ldref_paths must contain at least one file")

    weights = np.asarray(weights, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if weights.shape != (weights_variants.n,) or not np.isfinite(weights).all():
        raise ValueError(
            f"weights must be a finite vector of length {weights_variants.n}")
    if z.shape != (sumstats_variants.n,) or not np.isfinite(z).all():
        raise ValueError(
            f"z must be a finite vector of length {sumstats_variants.n}")
    var_y = _var_y(var_y)

    if weight_scale == "dosage":
        n_sd_sources = int(genotype_sd is not None) + int(hwe_genotype_sd)
        if n_sd_sources != 1:
            raise ValueError(
                "dosage-scale weights require exactly one genotype-SD source: "
                "the sumstats genotype_sd vector or hwe_genotype_sd=True")
        if genotype_sd is not None:
            genotype_sd = np.asarray(genotype_sd, dtype=np.float64)
            if (genotype_sd.shape != (sumstats_variants.n,)
                    or not np.isfinite(genotype_sd).all()
                    or np.any(genotype_sd <= 0.0)):
                raise ValueError(
                    "genotype_sd must be a finite positive vector of length "
                    f"{sumstats_variants.n}")
            sd_source = "sumstats_empirical"
        else:
            sd_source = "ldref_hwe"
    elif weight_scale == "standardized":
        if hwe_genotype_sd:
            raise ValueError(
                "hwe_genotype_sd is only used with weight_scale='dosage'")
        sd_source = "not_used"
    else:
        raise ValueError("weight_scale must be 'standardized' or 'dosage'")
    if mse_interpretable is None:
        mse_interpretable = weight_scale == "standardized"

    w_rows = _chromosome_rows(weights_variants)
    z_rows = _chromosome_rows(sumstats_variants)
    w_report = {name: 0 for name in _REPORT_COUNTS}
    z_report = {name: 0 for name in _REPORT_COUNTS}
    seen = set()
    chromosomes = []
    num = den = 0.0
    n_reference = n_scored = 0

    for path in ldref_paths:
        ref = read_ldref(path)
        reference = ref["variants"]
        ref_chrom = reference.norm_chrom
        chrom = str(ref_chrom[0])
        if not np.all(ref_chrom == chrom):
            raise ValueError(
                f"LD-reference file {str(path)!r} contains multiple chromosomes")
        if chrom in seen:
            raise ValueError(
                f"duplicate LD-reference chromosome {chrom!r} in {str(path)!r}")
        seen.add(chrom)
        chromosomes.append(chrom)

        w_var, w_local = _subset_variant_values(
            weights_variants, weights, w_rows.get(chrom))
        z_var, z_local = _subset_variant_values(
            sumstats_variants, z, z_rows.get(chrom))
        w_aligned, wrep, wmask = harmonize_to(
            reference, w_var, w_local, remove_ambiguous=remove_ambiguous,
            return_mask=True)
        z_aligned, zrep, zmask, _, z_source = _harmonize_to_details(
            reference, z_var, z_local, remove_ambiguous=remove_ambiguous,
            return_target_index=(weight_scale == "dosage"
                                 and not hwe_genotype_sd))
        _add_report(w_report, wrep)
        _add_report(z_report, zrep)

        if weight_scale == "dosage":
            if hwe_genotype_sd:
                if "af" not in ref:
                    raise ValueError(
                        f"LD-reference file {str(path)!r} has no af annotation "
                        "for HWE genotype-SD scaling")
                af = np.asarray(ref["af"], dtype=np.float64)
                bad_shape = af.shape != (reference.n,)
                bad = (~np.isfinite(af) | (af <= 0.0) | (af >= 1.0)
                       if not bad_shape else np.array([True]))
                if bad_shape or bad.any():
                    raise ValueError(
                        f"LD-reference file {str(path)!r} af must be a finite "
                        "vector in (0, 1) with one value per reference variant")
                sd_aligned = np.sqrt(2.0 * af * (1.0 - af))
            else:
                selector = z_rows.get(chrom, slice(0, 0))
                sd_local = genotype_sd[selector]
                sd_aligned = np.zeros(reference.n, dtype=np.float64)
                sd_aligned[zmask] = sd_local[z_source[zmask]]
            w_aligned *= sd_aligned

        joint = wmask & zmask
        w_aligned[~joint] = 0.0
        z_aligned[~joint] = 0.0
        local_num = float(w_aligned @ z_aligned)
        local_den = float(ref["ld"].quad(w_aligned))
        if not np.isfinite(local_num):
            raise ValueError(f"chromosome {chrom} w^T z is not finite")
        if not np.isfinite(local_den):
            raise ValueError(f"chromosome {chrom} w^T D w is not finite")
        num += local_num
        den += local_den
        n_reference += reference.n
        n_scored += int(np.count_nonzero(w_aligned))
        # Drop the large LD payload before opening the next shard. Assignment
        # evaluates its right-hand side first, so retaining ``ref`` here would
        # otherwise make peak memory include two chromosome files.
        del ref

    # A whole-table harmonization would count rows on chromosomes absent from
    # the reference as unmatched. Preserve that report contract after sharding.
    missing_w = _missing_rows(w_rows, seen, weights_variants.n)
    missing_z = _missing_rows(z_rows, seen, sumstats_variants.n)
    w_report["n_target"] += missing_w
    w_report["n_unmatched"] += missing_w
    z_report["n_target"] += missing_z
    z_report["n_unmatched"] += missing_z

    return LDRefEvaluationResult(
        r2=_r2_from_quad(num, den, var_y),
        mse=_mse_from_quad(num, den, var_y),
        n_reference=n_reference,
        n_variants_scored=n_scored,
        weight_scale=weight_scale,
        mse_interpretable=bool(mse_interpretable),
        weights_report=HarmonizeReport(**w_report).to_dict(),
        sumstats_report=HarmonizeReport(**z_report).to_dict(),
        n_ldref_files=len(ldref_paths),
        chromosomes=tuple(chromosomes),
        genotype_sd_source=sd_source,
    )


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
