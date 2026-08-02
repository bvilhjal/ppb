"""PPB: summary-statistics-based cross-ancestry polygenic-score portability benchmark.

Given PGS weights ``w``, target-ancestry marginal summary statistics ``z`` and an
LD backend for ``D``, the predictive accuracy (in the target ancestry) is
estimated without individual-level data as

    R^2 = (w^T z)^2 / (w^T D w).

The estimator is ancestry-agnostic in form; supplying target-ancestry ``z`` and
``D`` measures cross-ancestry portability. See ``docs/METHOD.md`` and
``docs/CROSS_ANCESTRY.md``.
"""

from .diagnostics import r2_block_jackknife, sign_flip_null
from .estimator import mse, r2
from .evaluate import EvaluationResult, evaluate
from .score_distribution import score_distribution
from .harmonize import HarmonizeReport, VariantTable, harmonize_to
from .io import (
    LDRefEvaluationResult,
    evaluate_ldrefs,
    read_bundle,
    read_sumstats,
    read_weights,
    write_bundle,
)
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
    "r2", "mse",
    "r2_block_jackknife", "sign_flip_null",
    "DenseLD", "LowRankLD", "BlockDiagonalLD", "LDBackend",
    "DenseLDInt8", "PackedDenseLDInt8",
    "lowrank_ld",
    "VariantTable", "harmonize_to", "HarmonizeReport",
    "evaluate", "EvaluationResult",
    "score_distribution",
    "read_weights", "read_sumstats", "read_bundle", "write_bundle",
    "evaluate_ldrefs", "LDRefEvaluationResult",
    "read_ldref", "write_ldref",
    "OverlapBasis",
    "standardized_marginal",
]
__version__ = "0.0.1.dev1"
