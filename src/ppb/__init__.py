"""PPB: summary-statistics-based cross-ancestry polygenic-score portability benchmark.

Given PGS weights ``w``, target-ancestry marginal summary statistics ``z`` and an
LD backend for ``D``, the predictive accuracy (in the target ancestry) is
estimated without individual-level data as

    R^2 = (w^T z)^2 / (w^T D w).

The estimator is ancestry-agnostic in form; supplying target-ancestry ``z`` and
``D`` measures cross-ancestry portability. See ``docs/METHOD.md`` and
``docs/CROSS_ANCESTRY.md``.
"""

from .covariates import adjust, principal_components, residualize
from .estimator import mse, r2
from .evaluate import EvaluationResult, evaluate
from .harmonize import HarmonizeReport, VariantTable, harmonize_to
from .io import read_bundle, read_weights, write_bundle
from .ldref import read_ldref, write_ldref
from .overlap import OverlapEstimate, block_products, correct_numerator, overlap_slope
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
    "read_ldref", "write_ldref",
    "block_products", "overlap_slope", "correct_numerator", "OverlapEstimate",
    "residualize", "adjust", "principal_components",
    "pumas_r2", "subsample_sumstats",
    "standardized_marginal", "zstat_to_corr",
]
__version__ = "0.0.1.dev0"
