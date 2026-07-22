"""Turn GWAS summary statistics into the standardized marginal statistics the
estimator needs, honouring per-variant sample sizes.

PPB works on the standardized (correlation) scale: ``z_j`` is the marginal
correlation of variant ``j`` with the (standardized) phenotype, and ``D`` is the
LD correlation matrix. A GWAS reports per-variant effect ``beta_j``, standard
error ``se_j``, and sample size ``n_j`` (which varies across a meta-analysis).
The marginal z-statistic ``t_j = beta_j / se_j`` maps to the standardized
correlation by

    r_j = t_j / sqrt(t_j^2 + n_j - 2).

Using a single uniform ``N`` when the true ``n_j`` vary biases every low-``n``
variant's contribution and therefore the estimated R^2 (PUMAS Eq. 20 carries the
same per-variant ``N``/``SE`` normalisation; here it enters through ``r_j``).
"""

from __future__ import annotations

import numpy as np


def zstat_to_corr(t, n):
    """Standardized marginal correlation from a marginal z-statistic ``t`` and
    per-variant sample size ``n``: ``r = t / sqrt(t^2 + n - 2)``.

    ``n`` may be a scalar or a per-variant array.
    """
    t = np.asarray(t, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    if not np.isfinite(t).all() or not np.isfinite(n).all():
        raise ValueError("t and n must contain only finite values")
    if np.any(n <= 2):
        raise ValueError("sample size n must exceed 2")
    try:
        t, n = np.broadcast_arrays(t, n)
    except ValueError as exc:
        raise ValueError("t and n must have broadcast-compatible shapes") from exc
    return t / np.hypot(t, np.sqrt(n - 2.0))


def standardized_marginal(beta, se, n):
    """Standardized marginal correlation ``z`` from GWAS ``beta``, ``se`` and
    per-variant sample size ``n`` (``t = beta / se``; see :func:`zstat_to_corr`)."""
    beta = np.asarray(beta, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    if beta.shape != se.shape:
        raise ValueError("beta and se must have the same shape")
    if not np.isfinite(beta).all() or not np.isfinite(se).all():
        raise ValueError("beta and se must contain only finite values")
    if np.any(se <= 0):
        raise ValueError("standard errors must be positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        t = beta / se
    if not np.isfinite(t).all():
        raise ValueError("beta / se must be finite")
    return zstat_to_corr(t, n)
