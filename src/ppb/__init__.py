"""PPB: privacy-preserving benchmark for polygenic prediction.

Summary-statistic evaluation of polygenic scores. Given PGS weights ``w``, target
marginal summary statistics ``z``, and an LD backend for ``D``, the predictive
accuracy is estimated without individual-level data as

    R^2 = (w^T z)^2 / (w^T D w).

See ``docs/METHOD.md`` for the full specification.
"""

from .estimator import mse, r2
from .evaluate import EvaluationResult, evaluate
from .harmonize import HarmonizeReport, VariantTable, harmonize_to
from .io import read_bundle, read_weights, write_bundle
from .ld_backend import (
    BlockDiagonalLD,
    DenseLD,
    LDBackend,
    LowRankLD,
    lowrank_ld,
)

__all__ = [
    "r2", "mse",
    "DenseLD", "LowRankLD", "BlockDiagonalLD", "LDBackend",
    "lowrank_ld",
    "VariantTable", "harmonize_to", "HarmonizeReport",
    "evaluate", "EvaluationResult",
    "read_weights", "read_bundle", "write_bundle",
]
__version__ = "0.0.1.dev0"
