"""High-level evaluation: harmonize inputs to the LD reference, then estimate R^2.

This ties the pieces together into the operation a benchmark submission needs:
given PGS weights and target summary statistics (each with their own variant
tables and allele orientation), align both to the LD reference's variants and
compute the summary-statistic prediction accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .estimator import mse as _mse
from .estimator import r2 as _r2
from .harmonize import HarmonizeReport, VariantTable, harmonize_to
from .ld_backend import LDBackend


@dataclass
class EvaluationResult:
    """Machine-readable result of one evaluation."""

    r2: float
    mse: float
    n_reference: int
    n_variants_scored: int          # reference variants with a nonzero aligned weight
    weight_scale: str = "standardized"
    weights_report: dict = field(default_factory=dict)
    sumstats_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(ld: LDBackend, ld_variants: VariantTable,
             weights_variants: VariantTable, weights,
             sumstats_variants: VariantTable, z,
             *, var_y: float = 1.0, weight_scale: str = "standardized",
             genotype_sd=None, remove_ambiguous: bool = True) -> EvaluationResult:
    """Harmonize weights and summary statistics to ``ld_variants``, then evaluate.

    ``ld`` must be an LD backend defined over ``ld_variants`` (same order).
    ``weight_scale='standardized'`` means the weights already multiply the
    standardized genotypes represented by ``ld``. For ordinary per-dosage PGS
    weights, pass ``weight_scale='dosage'`` and the target-cohort
    ``genotype_sd`` in reference order; each weight is then multiplied by its
    genotype SD before evaluation.
    """
    if ld.m != ld_variants.n:
        raise ValueError(
            f"LD backend has m={ld.m} but ld_variants has {ld_variants.n} variants")

    w_aligned, wrep, wmask = harmonize_to(
        ld_variants, weights_variants, weights,
        remove_ambiguous=remove_ambiguous, return_mask=True)
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

    # Missing z is not zero association. Evaluate on the joint matched set so
    # exactly the same variants enter w^T z and w^T D w.
    joint = wmask & zmask
    w_aligned[~joint] = 0.0
    z_aligned[~joint] = 0.0

    return EvaluationResult(
        r2=_r2(w_aligned, z_aligned, ld, var_y=var_y),
        mse=_mse(w_aligned, z_aligned, ld, var_y=var_y),
        n_reference=ld_variants.n,
        n_variants_scored=int(np.count_nonzero(w_aligned)),
        weight_scale=weight_scale,
        weights_report=wrep.to_dict(),
        sumstats_report=zrep.to_dict(),
    )
