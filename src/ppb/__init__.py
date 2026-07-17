"""PPB: privacy-preserving benchmark for polygenic prediction.

Summary-statistic evaluation of polygenic scores. Given PGS weights ``w``, target
marginal summary statistics ``z``, and an LD backend for ``D``, the predictive
accuracy is estimated without individual-level data as

    R^2 = (w^T z)^2 / (w^T D w).

See ``docs/METHOD.md`` for the full specification.
"""

from .estimator import mse, r2
from .ld_backend import DenseLD, LDBackend, LowRankLD

__all__ = ["r2", "mse", "DenseLD", "LowRankLD", "LDBackend"]
__version__ = "0.0.1.dev0"
