"""File formats and file-backed evaluation paths for the CLI.

- **Weights**: a delimited text table (TSV/CSV) with columns for chromosome,
  position, effect allele, other allele, and the weight. Common PGS Catalog
  column names are recognised. ``#``-prefixed comment lines are skipped.
- **Summary statistics**: the same four variant columns plus standardized
  marginal ``z`` -- or ``beta``/``se`` with a per-variant ``n`` column (or a
  declared trait-level ``n_eff``), converted to ``z`` by (M4) when
  ``read_sumstats`` is asked to. An optional empirical ``genotype_sd`` column
  puts ordinary dosage weights on the standardized-genotype scale.
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

from .diagnostics import block_diagnostics
from .estimator import (
    _mse_from_quad, _r2_from_quad, _var_y, corrected_r2, frozen_to_dosage,
)
from .evaluate import EvaluationResult, _require_weight_scale
from .harmonize import (
    HarmonizeReport,
    VariantTable,
    _harmonize_to_details,
    harmonize_to,
)
from .ld_backend import DenseLD, LowRankLD, _require_symmetric
from .ldref import read_ldref
from .sumstats import standardized_marginal

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
_FROZEN_ALIASES = {
    "af_ref": ("af_ref",),
    "sd_ref": ("sd_ref",),
}
_SUMSTATS_ALIASES = {**_VARIANT_ALIASES, "z": ("z",)}
_BETA_ALIASES = ("beta", "effect")
_SE_ALIASES = ("se", "sebeta", "standard_error", "std_err")
_N_ALIASES = ("n", "n_eff", "neff", "sample_size")
_SUMSTATS_SCALES = ("z", "beta-se-n")


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


@dataclass
class WeightFile:
    """Parsed PGS weights, including optional LDpred3 frozen-scale columns.

    ``weights`` is the file's ``WEIGHT`` column as written. LDpred3 writes a
    *standardized* effect there. ``af_ref`` / ``sd_ref`` are the fit-cohort
    A1 frequency and dosage SD (``AF_REF`` / ``SD_REF``), present only on a
    frozen-deploy file. They are discovery-cohort moments, not target SDs.
    """

    variants: VariantTable
    weights: np.ndarray
    af_ref: np.ndarray | None = None
    sd_ref: np.ndarray | None = None

    @property
    def has_frozen_scale(self) -> bool:
        return self.sd_ref is not None


def read_weight_file(path) -> WeightFile:
    """Read weights, keeping optional ``AF_REF`` / ``SD_REF`` if both are present."""
    variants, weights = _read_value_table(
        path, aliases=_WEIGHT_ALIASES, value_name="weight",
        table_name="weights")
    af_ref = sd_ref = None
    with open(path, "r", encoding="utf-8") as fh:
        lines = (
            ln.rstrip("\r\n")
            for ln in fh
            if ln.strip() and not ln.lstrip().startswith("#")
        )
        try:
            header_line = next(lines)
        except StopIteration:
            raise ValueError(f"weights file {path!r} has no header") from None
        delim = ("\t" if "\t" in header_line
                 else ("," if "," in header_line else None))
        split = (lambda ln: ln.split(delim)) if delim else (lambda ln: ln.split())
        lut = {name.strip().lower().lstrip("#"): i
               for i, name in enumerate(split(header_line))}
        has_af = "af_ref" in lut
        has_sd = "sd_ref" in lut
        if has_af != has_sd:
            raise ValueError(
                f"weights file {path!r} must carry both AF_REF and SD_REF, "
                "or neither")
        if has_sd:
            af_col, sd_col = lut["af_ref"], lut["sd_ref"]
            af_vals, sd_vals = [], []
            for lineno, ln in enumerate(lines, start=2):
                r = split(ln)
                need = max(af_col, sd_col) + 1
                if len(r) < need:
                    raise ValueError(
                        f"weights file {path!r} line {lineno}: expected "
                        f"{need} fields, got {len(r)}")
                af = float(r[af_col])
                sd = float(r[sd_col])
                if not np.isfinite(af) or not (0.0 <= af <= 1.0):
                    raise ValueError(
                        f"weights file {path!r} line {lineno}: AF_REF must "
                        "be finite and in [0, 1]")
                if not np.isfinite(sd) or sd < 0.0:
                    raise ValueError(
                        f"weights file {path!r} line {lineno}: SD_REF must "
                        "be finite and non-negative")
                af_vals.append(af)
                sd_vals.append(sd)
            if len(af_vals) != variants.n:
                raise ValueError(
                    f"weights file {path!r} AF_REF/SD_REF rows "
                    f"({len(af_vals)}) do not match weight rows ({variants.n})")
            af_ref = np.array(af_vals, dtype=np.float64)
            sd_ref = np.array(sd_vals, dtype=np.float64)
    return WeightFile(
        variants=variants, weights=weights, af_ref=af_ref, sd_ref=sd_ref)


def read_weights(path):
    """Read a PGS weights file -> ``(VariantTable, weights)``.

    The file is consumed line by line, so a million-row file does not coexist
    with a second, raw-text copy of itself. Optional ``AF_REF`` / ``SD_REF``
    are ignored here; use :func:`read_weight_file` when the frozen-deploy
    scale is required.
    """
    parsed = read_weight_file(path)
    return parsed.variants, parsed.weights


def read_sumstats(path, *, read_genotype_sd: bool = True, scale: str = "z",
                  n_eff: float | None = None):
    """Read a standardized-``z`` table, or convert a ``beta``/``se`` table by (M4).

    Required variant columns are ``chrom``, ``pos``, ``a1``, ``a2``. With the
    default ``scale='z'`` the file must carry exactly ``z`` -- the standardized
    marginal correlation the estimator consumes -- and no beta/SE/N conversion
    is inferred. With ``scale='beta-se-n'`` the file must instead carry
    ``beta`` and ``se``; the standardized marginal ``z_j = t_j/sqrt(t_j^2 + n_j
    - 2)`` is then computed by :func:`ppb.standardized_marginal` from a
    per-variant ``n`` column when the file carries one, and from the declared
    trait-level ``n_eff`` when it does not. An optional empirical target-cohort
    ``genotype_sd`` column may be supplied on the same rows. Set
    ``read_genotype_sd=False`` when that column is deliberately irrelevant
    (standardized weights or an explicit HWE approximation); it is then neither
    parsed nor validated.

    Returns ``(variants, z, genotype_sd)``; the last item is ``None`` when the
    optional column is absent.
    """
    if scale not in _SUMSTATS_SCALES:
        raise ValueError(f"scale must be one of {_SUMSTATS_SCALES}; got {scale!r}")
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
        cols = _resolve_columns(
            header,
            _SUMSTATS_ALIASES if scale == "z" else _VARIANT_ALIASES,
            "sumstats")
        lut = {name.strip().lower().lstrip("#"): i
               for i, name in enumerate(header)}
        sd_col = lut.get("genotype_sd") if read_genotype_sd else None

        def _column(aliases, field):
            for alias in aliases:
                if alias in lut:
                    return lut[alias]
            raise ValueError(
                f"sumstats file {path!r} is missing a '{field}' column "
                f"(looked for {aliases}); header was {header}")

        beta_col = se_col = n_col = None
        if scale == "beta-se-n":
            beta_col = _column(_BETA_ALIASES, "beta")
            se_col = _column(_SE_ALIASES, "se")
            for alias in _N_ALIASES:
                if alias in lut:
                    n_col = lut[alias]
                    break
            if n_col is None:
                if n_eff is None:
                    raise ValueError(
                        f"sumstats file {path!r} has no 'n' column (looked for "
                        f"{_N_ALIASES}); pass n_eff to convert beta/SE with a "
                        "trait-level constant sample size")
                n_eff = float(n_eff)
                if not np.isfinite(n_eff) or n_eff <= 2.0:
                    raise ValueError("n_eff must be finite and greater than 2")

        need_cols = [*cols.values(), *([] if sd_col is None else [sd_col])]
        if beta_col is not None:
            need_cols += [beta_col, se_col] + ([] if n_col is None else [n_col])
        need = max(need_cols) + 1

        chrom, pos, a1, a2, z = [], [], [], [], []
        beta_v, se_v, n_v = [], [], []
        genotype_sd = []
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
            if scale == "z":
                value = float(r[cols["z"]])
                if not np.isfinite(value):
                    raise ValueError(
                        f"sumstats file {path!r} line {lineno}: z must be finite")
                z.append(value)
            else:
                b = float(r[beta_col])
                s = float(r[se_col])
                if not np.isfinite(b):
                    raise ValueError(
                        f"sumstats file {path!r} line {lineno}: beta must be finite")
                if not np.isfinite(s) or s <= 0.0:
                    raise ValueError(
                        f"sumstats file {path!r} line {lineno}: se must be "
                        "finite and positive")
                beta_v.append(b)
                se_v.append(s)
                if n_col is not None:
                    nj = float(r[n_col])
                    if not np.isfinite(nj) or nj <= 2.0:
                        raise ValueError(
                            f"sumstats file {path!r} line {lineno}: n must be "
                            "finite and greater than 2")
                    n_v.append(nj)
            if sd_col is not None:
                value = float(r[sd_col])
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"sumstats file {path!r} line {lineno}: genotype_sd "
                        "must be finite and strictly positive")
                genotype_sd.append(value)
    if not chrom:
        raise ValueError(f"sumstats file {path!r} has no data rows")
    variants = VariantTable(chrom, pos, a1, a2)
    if scale == "beta-se-n":
        n = np.array(n_v) if n_col is not None else n_eff
        z = standardized_marginal(np.array(beta_v), np.array(se_v), n)
    else:
        z = np.array(z, dtype=np.float64)
    sd = (np.array(genotype_sd, dtype=np.float64)
          if sd_col is not None else None)
    return variants, z, sd


@dataclass
class LDRefEvaluationResult(EvaluationResult):
    """An :class:`EvaluationResult` with sharded-input provenance."""

    source: str = "ldref_shards"
    n_ldref_files: int = 0
    chromosomes: tuple[str, ...] = field(default_factory=tuple)


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
        sd_ref=None, genotype_sd_frame: str = "sumstats",
        hwe_genotype_sd: bool = False, n_eff=None,
        remove_ambiguous: bool = True,
        mse_interpretable: bool | None = None) -> LDRefEvaluationResult:
    """Evaluate chromosome-sharded block-int8 LD references in one pass.

    Each path must be one single-chromosome file accepted by
    :func:`~ppb.ldref.read_ldref`; duplicate chromosomes are rejected. Weights
    and summary statistics are partitioned by chromosome once, then only that
    chromosome's rows are harmonized. The global numerator ``w'z`` and
    denominator ``w'Dw`` are summed before R2 and MSE are formed. Chromosome R2
    values are never averaged.

    ``genotype_sd`` is always in the order of the table that carries it.
    The default ``genotype_sd_frame='sumstats'`` is the GWAS-file column
    (``sumstats_variants`` order). ``'reference'`` is not used here: each
    shard has its own reference, so a single reference-order vector does
    not exist. As an explicit approximation, ``hwe_genotype_sd=True``
    uses ``sqrt(2*af*(1-af))`` from each LD reference. Exactly one SD
    source is required for dosage and frozen weights; neither is used
    for standardized weights.

    ``weight_scale='frozen'`` converts LDpred3 ``WEIGHT / SD_REF`` to
    dosage first (``sd_ref`` in ``weights_variants`` order). Block
    jackknife and sign-flip diagnostics are accumulated across shards.
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
    weight_scale = _require_weight_scale(weight_scale)
    if genotype_sd_frame != "sumstats":
        raise ValueError(
            "evaluate_ldrefs only accepts genotype_sd_frame='sumstats': "
            "each shard has its own reference, so a single reference-order "
            "genotype_sd vector does not exist")

    if weight_scale == "frozen":
        if sd_ref is None:
            raise ValueError(
                "frozen-scale weights require sd_ref (the file's SD_REF column)")
        weights = frozen_to_dosage(weights, sd_ref)

    needs_sd = weight_scale in ("dosage", "frozen")
    if needs_sd:
        n_sd_sources = int(genotype_sd is not None) + int(hwe_genotype_sd)
        if n_sd_sources != 1:
            raise ValueError(
                f"{weight_scale}-scale weights require exactly one "
                "genotype-SD source: the sumstats genotype_sd vector or "
                "hwe_genotype_sd=True")
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
    elif hwe_genotype_sd:
        raise ValueError(
            "hwe_genotype_sd is only used with weight_scale='dosage' or "
            "'frozen'")
    else:
        sd_source = "not_used"
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
    u_blocks, v_blocks, chrom_tag = [], [], []

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
            return_target_index=(needs_sd and not hwe_genotype_sd))
        _add_report(w_report, wrep)
        _add_report(z_report, zrep)

        if needs_sd:
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
        ld = ref["ld"]
        if hasattr(ld, "block_quads") and hasattr(ld, "blocks"):
            v_chr = ld.block_quads(w_aligned)
            u_chr = np.empty(len(ld.blocks), dtype=np.float64)
            for b, (_backend, idx) in enumerate(ld.blocks):
                u_chr[b] = float(w_aligned[idx] @ z_aligned[idx])
            local_num = float(u_chr.sum())
            local_den = float(v_chr.sum())
            u_blocks.append(u_chr)
            v_blocks.append(v_chr)
            chrom_tag.extend([chrom] * u_chr.size)
        else:
            local_num = float(w_aligned @ z_aligned)
            local_den = float(ld.quad(w_aligned))
        if not np.isfinite(local_num):
            raise ValueError(f"chromosome {chrom} w^T z is not finite")
        if not np.isfinite(local_den):
            raise ValueError(f"chromosome {chrom} w^T D w is not finite")
        num += local_num
        den += local_den
        n_reference += reference.n
        n_scored += int(np.count_nonzero(w_aligned))
        # Drop the large LD payload before opening the next shard. Assignment
        # evaluates its right-hand side first, so retaining ``ref`` / ``ld``
        # here would otherwise make peak memory include two chromosome files.
        del ld, ref

    # A whole-table harmonization would count rows on chromosomes absent from
    # the reference as unmatched. Preserve that report contract after sharding.
    missing_w = _missing_rows(w_rows, seen, weights_variants.n)
    missing_z = _missing_rows(z_rows, seen, sumstats_variants.n)
    w_report["n_target"] += missing_w
    w_report["n_unmatched"] += missing_w
    z_report["n_target"] += missing_z
    z_report["n_unmatched"] += missing_z

    result = LDRefEvaluationResult(
        r2=_r2_from_quad(num, den, var_y),
        mse=_mse_from_quad(num, den, var_y),
        n_reference=n_reference,
        n_variants_scored=n_scored,
        weight_scale=weight_scale,
        mse_interpretable=bool(mse_interpretable),
        weights_report=HarmonizeReport(**w_report).to_dict(),
        sumstats_report=HarmonizeReport(**z_report).to_dict(),
        genotype_sd_source=sd_source,
        n_ldref_files=len(ldref_paths),
        chromosomes=tuple(chromosomes),
    )
    if u_blocks:
        u_all = np.concatenate(u_blocks)
        v_all = np.concatenate(v_blocks)
        payload = block_diagnostics(
            u_all, v_all, chrom=np.asarray(chrom_tag, dtype=object),
            var_y=var_y)
        result.diagnostics_unavailable = payload.get("diagnostics_unavailable")
        result.jackknife = payload.get("jackknife")
        result.jackknife_chromosome = payload.get("jackknife_chromosome")
        result.per_chromosome = payload.get("per_chromosome")
        result.sign_flip_null = payload.get("sign_flip_null")
    if n_eff is not None:
        _raw, corrected, se = corrected_r2(num, den, n_eff, var_y)
        result.n_eff = float(n_eff)
        result.r2_corrected = corrected
        result.r2_se_finite_sample = se
    return result


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
    if D is not None:
        # Read-side validation would catch this through DenseLD, but fail at
        # the write boundary instead of shipping a bundle that cannot be read.
        _require_symmetric(operator)
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
