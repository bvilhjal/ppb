"""Numba kernels for LD quadratic forms.

The estimator only needs the scalar ``w^T D w``, so these kernels reduce over the
compact LD representation without ever materialising a dense ``M``-by-``M`` matrix.

They use explicit scalar-loop reductions (no ``@`` / ``np.dot`` inside ``njit``):
numba's matmul path requires a scipy BLAS the target env does not carry, and the
loop form mirrors ldpred3's ``_lr8_sweep_all`` / ``_d8_sweep_all`` kernels.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def lowrank_quad(U, w):
    """``w^T (U U^T) w = ||U^T w||^2`` for a low-rank factor ``U`` of shape (m, r).

    Serial reference kernel. ``s_j = sum_i U[i, j] w[i]`` then the result is
    ``sum_j s_j^2``.
    """
    m, r = U.shape
    total = 0.0
    for j in range(r):
        s = 0.0
        for i in range(m):
            s += U[i, j] * w[i]
        total += s * s
    return total


@njit(parallel=True, cache=True)
def lowrank_quad_par(U, w):
    """Parallel ``||U^T w||^2``.

    Each column ``j`` of ``U`` is independent, so ``s_j`` is computed in a
    ``prange`` into a per-column buffer and summed afterwards -- no cross-thread
    reduction race.
    """
    m, r = U.shape
    partial = np.zeros(r)
    for j in prange(r):
        s = 0.0
        for i in range(m):
            s += U[i, j] * w[i]
        partial[j] = s * s
    total = 0.0
    for j in range(r):
        total += partial[j]
    return total


@njit(cache=True)
def dense_quad(D, w):
    """``w^T D w`` for a dense symmetric block ``D`` of shape (m, m)."""
    m = D.shape[0]
    total = 0.0
    for i in range(m):
        wi = w[i]
        if wi == 0.0:
            continue
        row_acc = 0.0
        for j in range(m):
            row_acc += D[i, j] * w[j]
        total += wi * row_acc
    return total
