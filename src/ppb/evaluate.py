"""High-level evaluation: harmonize inputs to the LD reference, then estimate R^2.

This ties the pieces together into the operation a benchmark submission needs:
given PGS weights and target summary statistics (each with their own variant
tables and allele orientation), align both to the LD reference's variants and
compute the summary-statistic prediction accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .estimator import _mse_from_quad, _r2_from_quad, _var_y, _wz
from .harmonize import HarmonizeReport, VariantTable, harmonize_to, same_variants
from .ld_backend import LDBackend


@dataclass
class EvaluationResult:
    """Machine-readable result of one evaluation.

    ``mse`` is meaningful only when the weights carry an absolute scale. ``R^2``
    is invariant to a global rescale of ``w``, so the usual per-allele PGS
    Catalog weights (in trait units, e.g. cm) give a correct ``R^2`` and a
    meaningless ``MSE``; ``mse_interpretable`` records which case applies.
    """

    r2: float
    mse: float
    n_reference: int
    n_variants_scored: int          # reference variants with a nonzero aligned weight
    weight_scale: str = "standardized"
    mse_interpretable: bool = True
    weights_report: dict = field(default_factory=dict)
    sumstats_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(ld: LDBackend, ld_variants: VariantTable,
             weights_variants: VariantTable, weights,
             sumstats_variants: VariantTable, z,
             *, var_y: float = 1.0, weight_scale: str = "standardized",
             genotype_sd=None, remove_ambiguous: bool = True,
             mse_interpretable: bool | None = None) -> EvaluationResult:
    """Harmonize weights and summary statistics to ``ld_variants``, then evaluate.

    ``ld`` must be an LD backend defined over ``ld_variants`` (same order).
    ``weight_scale='standardized'`` means the weights already multiply the
    standardized genotypes represented by ``ld``. For ordinary per-dosage PGS
    weights, pass ``weight_scale='dosage'`` and the target-cohort
    ``genotype_sd`` in reference order; each weight is then multiplied by its
    genotype SD before evaluation.

    ``mse_interpretable`` defaults to true for standardized weights and false
    for dosage weights. Set it explicitly only when the submitted weights and
    target phenotype are known to share (or not share) an absolute scale.

    Strand resolution (and therefore the palindrome drop) applies only to a
    table that came from somewhere else. When ``sumstats_variants`` *is* the
    reference table -- the usual case for a bundle, whose ``z`` is stored in
    reference order by construction -- there is no strand to resolve, so the
    pass is skipped rather than run and reported as ambiguous removals that
    never existed. ``weights_variants`` is a submission and is always resolved.
    """
    if ld.m != ld_variants.n:
        raise ValueError(
            f"LD backend has m={ld.m} but ld_variants has {ld_variants.n} variants")

    w_aligned, wrep, wmask = harmonize_to(
        ld_variants, weights_variants, weights,
        remove_ambiguous=remove_ambiguous, return_mask=True)
    if same_variants(ld_variants, sumstats_variants):
        z_aligned, _ = _wz(z, z)                    # finite + shape validation
        if z_aligned.shape != (ld_variants.n,):
            raise ValueError(
                f"z has shape {z_aligned.shape}, expected ({ld_variants.n},)")
        z_aligned = z_aligned.copy()
        zmask = np.ones(ld_variants.n, dtype=bool)
        zrep = HarmonizeReport(
            n_reference=ld_variants.n, n_target=ld_variants.n,
            n_matched=ld_variants.n, n_sign_flipped=0, n_strand_flipped=0,
            n_ambiguous_removed=0, n_mismatch=0, n_unmatched=0)
    else:
        z_aligned, zrep, zmask = harmonize_to(
            ld_variants, sumstats_variants, z,
            remove_ambiguous=remove_ambiguous, return_mask=True)

    if weight_scale == "dosage":
        if genotype_sd is None:
            raise ValueError(
                "dosage-scale weights require target-cohort genotype_sd")
        genotype_sd = np.asarray(genotype_sd, dtype=np.float64)
        if genotype_sd.shape != (ld_variants.n,):
            raise ValueError(
                f"genotype_sd has shape {genotype_sd.shape}, expected "
                f"({ld_variants.n},)")
        if not np.isfinite(genotype_sd).all() or np.any(genotype_sd <= 0.0):
            raise ValueError("genotype_sd must contain only finite positive values")
        w_aligned *= genotype_sd
    elif weight_scale != "standardized":
        raise ValueError("weight_scale must be 'standardized' or 'dosage'")
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
    den = ld.quad(w_checked)

    return EvaluationResult(
        r2=_r2_from_quad(wz, den, var_y),
        mse=_mse_from_quad(wz, den, var_y),
        n_reference=ld_variants.n,
        n_variants_scored=int(np.count_nonzero(w_aligned)),
        weight_scale=weight_scale,
        mse_interpretable=bool(mse_interpretable),
        weights_report=wrep.to_dict(),
        sumstats_report=zrep.to_dict(),
    )
