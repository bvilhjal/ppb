"""High-level evaluation: harmonize inputs to the LD reference, then estimate R^2.

This ties the pieces together into the operation a benchmark submission needs:
given PGS weights and target summary statistics (each with their own variant
tables and allele orientation), align both to the LD reference's variants and
compute the summary-statistic prediction accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .diagnostics import block_diagnostics
from .estimator import (
    _mse_from_quad, _r2_from_quad, _var_y, _wz, corrected_r2, frozen_to_dosage,
)
from .harmonize import (
    HarmonizeReport, VariantTable, _harmonize_to_details, harmonize_to,
    same_variants,
)
from .ld_backend import BlockDiagonalLD, LDBackend

WEIGHT_SCALES = ("standardized", "dosage", "frozen")


@dataclass
class EvaluationResult:
    """Machine-readable result of one evaluation.

    ``mse`` is meaningful only when the weights carry an absolute scale. ``R^2``
    is invariant to a global rescale of ``w``, so the usual per-allele PGS
    Catalog weights (in trait units, e.g. cm) give a correct ``R^2`` and a
    meaningless ``MSE``; ``mse_interpretable`` records which case applies. It
    defaults to True only for ``'standardized'`` weights. ``'dosage'`` and
    ``'frozen'`` weights become absolute only through the ``genotype_sd``
    multiply of step V3, whose provenance the caller vouches for (empirical
    target SD, HWE approximation, or something weaker); passing
    ``mse_interpretable=`` explicitly promotes a dosage-scale evaluation whose
    SD provenance the caller is prepared to stand behind.

    ``genotype_sd`` is always in the order of the table that carries it: a
    bundle stores it on the LD reference; a sumstats file stores it on the
    GWAS rows. ``genotype_sd_source`` records which of those, or HWE, was used.
    """

    r2: float
    mse: float
    n_reference: int
    n_variants_scored: int          # reference variants with a nonzero aligned weight
    weight_scale: str = "standardized"
    mse_interpretable: bool = True
    weights_report: dict = field(default_factory=dict)
    sumstats_report: dict = field(default_factory=dict)
    genotype_sd_source: str = "not_used"
    jackknife: dict | None = None
    jackknife_chromosome: dict | None = None
    per_chromosome: dict | None = None
    sign_flip_null: dict | None = None
    diagnostics_unavailable: str | None = None
    n_eff: float | None = None
    r2_corrected: float | None = None
    r2_se_finite_sample: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _require_weight_scale(weight_scale: str) -> str:
    if weight_scale not in WEIGHT_SCALES:
        raise ValueError(
            "weight_scale must be 'standardized', 'dosage', or 'frozen'")
    return weight_scale


def _block_products(ld: LDBackend, w, z):
    """Per-block ``u_b = w_bᵀ z_b`` and ``v_b = w_bᵀ D_b w_b``, or None."""
    if not isinstance(ld, BlockDiagonalLD):
        return None
    w = np.asarray(w, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    u = np.empty(len(ld.blocks), dtype=np.float64)
    for b, (_backend, idx) in enumerate(ld.blocks):
        u[b] = float(w[idx] @ z[idx])
    return u, ld.block_quads(w)


def _apply_finite_sample(result: EvaluationResult, num, den, n_eff, var_y):
    if n_eff is None:
        return result
    _raw, corrected, se = corrected_r2(num, den, n_eff, var_y)
    result.n_eff = float(n_eff)
    result.r2_corrected = corrected
    result.r2_se_finite_sample = se
    return result


def _attach_diagnostics(result: EvaluationResult, products, *,
                        chrom=None, var_y: float = 1.0):
    if products is None:
        result.diagnostics_unavailable = (
            "LD backend is not block-diagonal; jackknife and sign-flip "
            "need per-block products")
        return result
    payload = block_diagnostics(products[0], products[1], chrom=chrom,
                                var_y=var_y)
    result.diagnostics_unavailable = payload.get("diagnostics_unavailable")
    result.jackknife = payload.get("jackknife")
    result.jackknife_chromosome = payload.get("jackknife_chromosome")
    result.per_chromosome = payload.get("per_chromosome")
    result.sign_flip_null = payload.get("sign_flip_null")
    return result


def evaluate(ld: LDBackend, ld_variants: VariantTable,
             weights_variants: VariantTable, weights,
             sumstats_variants: VariantTable, z,
             *, var_y: float = 1.0, weight_scale: str = "standardized",
             genotype_sd=None, sd_ref=None, genotype_sd_frame: str = "reference",
             n_eff=None, remove_ambiguous: bool = True,
             mse_interpretable: bool | None = None) -> EvaluationResult:
    """Harmonize weights and summary statistics to ``ld_variants``, then evaluate.

    ``ld`` must be an LD backend defined over ``ld_variants`` (same order).
    ``weight_scale`` is one of:

    * ``'standardized'`` -- weights already multiply the standardized
      genotypes represented by ``ld`` (LDpred3/BiPred/GWFM ``WEIGHT``).
    * ``'dosage'`` -- ordinary per-allele weights; requires target-cohort
      ``genotype_sd``.
    * ``'frozen'`` -- LDpred3 ``WEIGHT`` plus fit-cohort ``sd_ref``
      (``SD_REF``). Converted to dosage by ``WEIGHT / SD_REF``, then
      treated as ``'dosage'``. ``SD_REF`` is *not* the target SD.

    ``genotype_sd`` is in the order of the table that carries it.
    ``genotype_sd_frame='reference'`` (default, bundles) is
    ``ld_variants`` order; ``'sumstats'`` is ``sumstats_variants`` order
    and is remapped through the same match used for ``z``.

    When ``ld`` is block-diagonal with at least two blocks, the result
    includes the jackknife (G2) and sign-flip null (G3). Pass ``n_eff``
    to also report the finite-sample correction (X3).

    Strand resolution (and therefore the palindrome drop) applies only to a
    table that came from somewhere else. When ``sumstats_variants`` *is* the
    reference table -- the usual case for a bundle, whose ``z`` is stored in
    reference order by construction -- there is no strand to resolve, so the
    pass is skipped rather than run and reported as ambiguous removals that
    never existed. ``weights_variants`` is a submission and is always resolved.

    Note the asymmetry with the sharded entry point,
    :func:`ppb.io.evaluate_ldrefs`: it has no whole-reference table to
    recognise, so it always harmonizes (and drops palindromes). The same
    inputs can therefore score a slightly different variant set -- and a
    slightly different R2 -- through the two paths when the summary
    statistics carry palindromic variants.
    """
    if ld.m != ld_variants.n:
        raise ValueError(
            f"LD backend has m={ld.m} but ld_variants has {ld_variants.n} variants")
    weight_scale = _require_weight_scale(weight_scale)
    if genotype_sd_frame not in ("reference", "sumstats"):
        raise ValueError("genotype_sd_frame must be 'reference' or 'sumstats'")

    if weight_scale == "frozen":
        if sd_ref is None:
            raise ValueError(
                "frozen-scale weights require sd_ref (the file's SD_REF column)")
        weights = frozen_to_dosage(weights, sd_ref)

    w_aligned, wrep, wmask = harmonize_to(
        ld_variants, weights_variants, weights,
        remove_ambiguous=remove_ambiguous, return_mask=True)
    z_source = None
    if same_variants(ld_variants, sumstats_variants):
        z_aligned, _ = _wz(z, z)                    # finite + shape validation
        if z_aligned.shape != (ld_variants.n,):
            raise ValueError(
                f"z has shape {z_aligned.shape}, expected ({ld_variants.n},)")
        z_aligned = z_aligned.copy()
        zmask = np.ones(ld_variants.n, dtype=bool)
        z_source = np.arange(ld_variants.n)
        zrep = HarmonizeReport(
            n_reference=ld_variants.n, n_target=ld_variants.n,
            n_matched=ld_variants.n, n_sign_flipped=0, n_strand_flipped=0,
            n_ambiguous_removed=0, n_mismatch=0, n_unmatched=0,
            n_ambiguous_indel_removed=0)
    else:
        z_aligned, zrep, zmask, _, z_source = _harmonize_to_details(
            ld_variants, sumstats_variants, z,
            remove_ambiguous=remove_ambiguous, return_target_index=True)

    sd_source = "not_used"
    if weight_scale in ("dosage", "frozen"):
        if genotype_sd is None:
            raise ValueError(
                f"{weight_scale}-scale weights require target-cohort genotype_sd")
        genotype_sd = np.asarray(genotype_sd, dtype=np.float64)
        if genotype_sd_frame == "reference":
            if genotype_sd.shape != (ld_variants.n,):
                raise ValueError(
                    f"genotype_sd has shape {genotype_sd.shape}, expected "
                    f"({ld_variants.n},) in reference order")
            if not np.isfinite(genotype_sd).all() or np.any(genotype_sd <= 0.0):
                raise ValueError(
                    "genotype_sd must contain only finite positive values")
            sd_aligned = genotype_sd
            sd_source = "reference_empirical"
        else:
            if genotype_sd.shape != (sumstats_variants.n,):
                raise ValueError(
                    f"genotype_sd has shape {genotype_sd.shape}, expected "
                    f"({sumstats_variants.n},) in sumstats order")
            if not np.isfinite(genotype_sd).all() or np.any(genotype_sd <= 0.0):
                raise ValueError(
                    "genotype_sd must contain only finite positive values")
            sd_aligned = np.zeros(ld_variants.n, dtype=np.float64)
            sd_aligned[zmask] = genotype_sd[z_source[zmask]]
            sd_source = "sumstats_empirical"
        w_aligned *= sd_aligned
    if mse_interpretable is None:
        mse_interpretable = weight_scale == "standardized"

    # Missing z is not zero association. Evaluate on the joint matched set so
    # exactly the same variants enter w^T z and w^T D w.
    joint = wmask & zmask
    w_aligned[~joint] = 0.0
    z_aligned[~joint] = 0.0

    # w^T D w is the expensive half of the estimator (a sweep over every LD
    # block); r2 and mse both need exactly this number, so compute it once.
    var_y = _var_y(var_y)
    w_checked, wz = _wz(w_aligned, z_aligned)
    products = _block_products(ld, w_checked, z_aligned)
    den = (float(products[1].sum()) if products is not None
           else ld.quad(w_checked))

    result = EvaluationResult(
        r2=_r2_from_quad(wz, den, var_y),
        mse=_mse_from_quad(wz, den, var_y),
        n_reference=ld_variants.n,
        n_variants_scored=int(np.count_nonzero(w_aligned)),
        weight_scale=weight_scale,
        mse_interpretable=bool(mse_interpretable),
        weights_report=wrep.to_dict(),
        sumstats_report=zrep.to_dict(),
        genotype_sd_source=sd_source,
    )
    _attach_diagnostics(result, products, var_y=var_y)
    return _apply_finite_sample(result, wz, den, n_eff, var_y)
