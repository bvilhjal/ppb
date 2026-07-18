"""PUMAS-style subsampling of GWAS summary statistics.

PUMAS (Zhao et al., Genome Biology 2021, doi:10.1186/s13059-021-02479-9) tunes
and evaluates polygenic scores from a *single* GWAS by Monte-Carlo subsampling
the marginal association statistics into pseudo training/validation sets, then
scoring on the pseudo-validation with the same summary-statistic R^2 the rest of
``ppb`` uses. It needs no separate held-out cohort.

In sufficient-statistic space ``t = X^T y`` (with ``t = n * z`` for marginal
``z = (1/n) X^T y``), the training partition is drawn from the full-GWAS
statistics via the conditional normal

    t^(tr) | t ~ N( (n_tr / n) t , (n_tr n_val / n) var_y * D ),   t^(val) = t - t^(tr)

and the two partitions are rescaled to marginal ``z``. PUMAS's published form
assumes LD-pruned (independent) SNPs so ``D`` is diagonal; here ``D`` is the full
LD matrix, the natural generalization. This is a PUMAS-*style* evaluator (the
subsampling idea + the shared R^2), not a bit-exact reimplementation of PUMAS's
per-SNP-sample-size normalization.
"""

from __future__ import annotations

import numpy as np

from .estimator import r2
from .ld_backend import DenseLD


def _dense_ld(ld):
    if isinstance(ld, DenseLD):
        return ld.D
    arr = np.asarray(ld, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise TypeError("PUMAS subsampling needs a dense LD matrix (DenseLD or square ndarray)")
    return arr


def subsample_sumstats(z_full, D, n_full, n_train, rng, *, var_y=1.0, chol=None):
    """Draw one PUMAS train/validation split of marginal statistics.

    Returns ``(z_train, z_val)`` on the marginal ``(1/n) X^T y`` scale.
    ``chol`` may be a precomputed lower-Cholesky of ``D`` (reused across draws).
    """
    z_full = np.asarray(z_full, dtype=np.float64)
    p = z_full.shape[0]
    n_val = n_full - n_train
    if n_train <= 0 or n_val <= 0:
        raise ValueError("require 0 < n_train < n_full")
    D = _dense_ld(D)
    if chol is None:
        chol = np.linalg.cholesky(D + 1e-8 * np.eye(p))
    t_full = n_full * z_full
    mean = (n_train / n_full) * t_full
    c = (n_train * n_val / n_full) * float(var_y)
    t_train = mean + np.sqrt(c) * (chol @ rng.standard_normal(p))
    t_val = t_full - t_train
    return t_train / n_train, t_val / n_val


def pumas_r2(w, z_full, D, n_full, rng, *, frac_val=0.25, n_reps=20, var_y=1.0):
    """PUMAS estimate of a PGS's prediction R^2 from a single GWAS.

    Subsamples ``n_reps`` pseudo-validation sets and averages the summary-statistic
    R^2 of ``w`` on each. ``w`` must be independent of this GWAS (else overfitting).
    """
    D = _dense_ld(D)
    chol = np.linalg.cholesky(D + 1e-8 * np.eye(D.shape[0]))
    n_train = int(round(n_full * (1.0 - frac_val)))
    ld = DenseLD(D)
    ests = np.empty(n_reps)
    for i in range(n_reps):
        _, z_val = subsample_sumstats(z_full, D, n_full, n_train, rng,
                                      var_y=var_y, chol=chol)
        ests[i] = r2(w, z_val, ld)
    return float(ests.mean())
