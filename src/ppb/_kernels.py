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


@njit(parallel=True, cache=True)
def lowrank_quad_int8(U8, rw):
    """``sum_j (sum_i U8[i, j] * rw[i])^2`` for int8 factor ``U8``.

    The caller folds the global quantisation ``scale^2`` in afterwards and passes
    ``rw = rowscale * w`` (rowscale restoring each row's unit norm).
    """
    m, r = U8.shape
    partial = np.zeros(r)
    for j in prange(r):
        s = 0.0
        for i in range(m):
            s += U8[i, j] * rw[i]
        partial[j] = s * s
    total = 0.0
    for j in range(r):
        total += partial[j]
    return total


@njit(cache=True)
def dense_quad_int8(D8, w):
    """``sum_ij D8[i, j] w[i] w[j]`` for int8 dense ``D8`` (caller divides by 127)."""
    m = D8.shape[0]
    total = 0.0
    for i in range(m):
        wi = w[i]
        if wi == 0.0:
            continue
        acc = 0.0
        for j in range(m):
            acc += D8[i, j] * w[j]
        total += wi * acc
    return total


@njit(parallel=True, cache=True)
def packed_quad_int8(p8, w, m):
    """``w^T D w`` for an int8 upper triangle packed row-major (caller divides by 127).

    Row ``i`` starts at ``i*m - i*(i-1)//2`` and holds ``D8[i, i:]``, so the
    whole matrix is ``m(m+1)/2`` bytes instead of ``m^2``. Using symmetry,

        w^T D w = sum_i w_i * ( D[i, i] w_i + 2 * sum_{j>i} D[i, j] w_j ).

    Rows are independent, so each accumulates into its own slot of ``partial``
    and the sum is taken afterwards -- no cross-thread reduction race. Row ``i``
    does ``m - i`` work, so the loop is deliberately left to numba's scheduler
    rather than chunked by hand.
    """
    partial = np.zeros(m)
    for i in prange(m):
        wi = w[i]
        if wi == 0.0:
            continue
        base = i * m - (i * (i - 1)) // 2
        off = 0.0
        for j in range(i + 1, m):
            off += p8[base + (j - i)] * w[j]
        partial[i] = wi * (p8[base] * wi + 2.0 * off)
    total = 0.0
    for i in range(m):
        total += partial[i]
    return total
