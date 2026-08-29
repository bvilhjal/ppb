"""PPB: summary-statistics-based cross-ancestry polygenic-score portability benchmark.

Given PGS weights ``w``, target-ancestry marginal summary statistics ``z`` and an
LD backend for ``D``, the predictive accuracy (in the target ancestry) is
estimated without individual-level data as

    R^2 = (w^T z)^2 / (w^T D w).

The estimator is ancestry-agnostic in form; supplying target-ancestry ``z`` and
``D`` measures cross-ancestry portability. See ``docs/METHOD.md`` and
``docs/CROSS_ANCESTRY.md``.
"""

from .diagnostics import block_diagnostics, r2_block_jackknife, sign_flip_null
from .ancestry import estimate_bilinear, estimate_pair_products
from .estimator import corrected_r2, frozen_to_dosage, mse, r2
from .evaluate import EvaluationResult, evaluate
from .score_distribution import score_distribution
from .harmonize import HarmonizeReport, VariantTable, harmonize_to
from .io import (
    LDRefEvaluationResult,
    WeightFile,
    evaluate_ldrefs,
    read_bundle,
    read_sumstats,
    read_weight_file,
    read_weights,
    write_bundle,
)
from .ldpred3_cache import convert_ldpred3_cache
from .ldref import read_ldref, write_ldref
from .overlap import OverlapBasis
from .ld_backend import (
    BlockDiagonalLD,
    DenseLD,
    DenseLDInt8,
    LDBackend,
    LowRankLD,
    PackedDenseLDInt8,
    lowrank_ld,
)
from .sumstats import standardized_marginal

__all__ = [
    "r2", "mse", "corrected_r2", "frozen_to_dosage",
    "r2_block_jackknife", "sign_flip_null", "block_diagnostics",
    "DenseLD", "LowRankLD", "BlockDiagonalLD", "LDBackend",
    "DenseLDInt8", "PackedDenseLDInt8",
    "lowrank_ld",
    "VariantTable", "harmonize_to", "HarmonizeReport",
    "evaluate", "EvaluationResult",
    "score_distribution",
    "read_weights", "read_weight_file", "WeightFile",
    "read_sumstats", "read_bundle", "write_bundle",
    "evaluate_ldrefs", "LDRefEvaluationResult",
    "read_ldref", "write_ldref",
    "convert_ldpred3_cache",
    "OverlapBasis",
    "standardized_marginal",
    "estimate_pair_products", "estimate_bilinear",
]
__version__ = "0.0.1.dev1"
