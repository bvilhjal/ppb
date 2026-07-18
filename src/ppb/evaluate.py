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
    weights_report: dict = field(default_factory=dict)
    sumstats_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(ld: LDBackend, ld_variants: VariantTable,
             weights_variants: VariantTable, weights,
             sumstats_variants: VariantTable, z,
             *, var_y: float = 1.0, remove_ambiguous: bool = True) -> EvaluationResult:
    """Harmonize weights and summary statistics to ``ld_variants``, then evaluate.

    ``ld`` must be an LD backend defined over ``ld_variants`` (same order).
    """
    if ld.m != ld_variants.n:
        raise ValueError(
            f"LD backend has m={ld.m} but ld_variants has {ld_variants.n} variants")

    w_aligned, wrep = harmonize_to(
        ld_variants, weights_variants, weights, remove_ambiguous=remove_ambiguous)
    z_aligned, zrep = harmonize_to(
        ld_variants, sumstats_variants, z, remove_ambiguous=remove_ambiguous)

    return EvaluationResult(
        r2=_r2(w_aligned, z_aligned, ld),
        mse=_mse(w_aligned, z_aligned, ld, var_y=var_y),
        n_reference=ld_variants.n,
        n_variants_scored=int(np.count_nonzero(w_aligned)),
        weights_report=wrep.to_dict(),
        sumstats_report=zrep.to_dict(),
    )
