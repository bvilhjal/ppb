"""PPB: privacy-preserving benchmark for polygenic prediction.

Summary-statistic evaluation of polygenic scores. Given PGS weights ``w``, target
marginal summary statistics ``z``, and an LD backend for ``D``, the predictive
accuracy is estimated without individual-level data as

    R^2 = (w^T z)^2 / (w^T D w).

See ``docs/METHOD.md`` for the full specification.
"""

from .covariates import adjust, principal_components, residualize
from .estimator import mse, r2
from .evaluate import EvaluationResult, evaluate
from .harmonize import HarmonizeReport, VariantTable, harmonize_to
from .io import read_bundle, read_weights, write_bundle
from .ld_backend import (
    BlockDiagonalLD,
    DenseLD,
    DenseLDInt8,
    LDBackend,
    LowRankLD,
    LowRankLDInt8,
    lowrank_ld,
    quantize_lowrank,
)
from .pumas import pumas_r2, subsample_sumstats
from .sumstats import standardized_marginal, zstat_to_corr

__all__ = [
    "r2", "mse",
    "DenseLD", "LowRankLD", "BlockDiagonalLD", "LDBackend",
    "DenseLDInt8", "LowRankLDInt8", "quantize_lowrank",
    "lowrank_ld",
    "VariantTable", "harmonize_to", "HarmonizeReport",
    "evaluate", "EvaluationResult",
    "read_weights", "read_bundle", "write_bundle",
    "residualize", "adjust", "principal_components",
    "pumas_r2", "subsample_sumstats",
    "standardized_marginal", "zstat_to_corr",
]
__version__ = "0.0.1.dev0"
