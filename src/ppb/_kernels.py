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


@njit(cache=True)
def dense_quad_sparse(D, w, nz):
    """Dense-float quadratic form restricted to the sorted support ``nz``."""
    total = 0.0
    for a in range(nz.size):
        i = nz[a]
        wi = w[i]
        row_acc = 0.0
        for b in range(nz.size):
            j = nz[b]
            row_acc += D[i, j] * w[j]
        total += wi * row_acc
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


@njit(cache=True)
def dense_quad_int8_sparse(D8, w, nz):
    """Square-D8 quadratic form restricted to the sorted support ``nz``."""
    total = 0.0
    for a in range(nz.size):
        i = nz[a]
        wi = w[i]
        row_acc = 0.0
        for b in range(nz.size):
            j = nz[b]
            row_acc += D8[i, j] * w[j]
        total += wi * row_acc
    return total


@njit(parallel=True, cache=True)
def dense_matvec_int8(D8, w, out):
    """``out = D8 @ w`` for a square int8 block (caller divides by 127).

    Rows are independent, so each writes its own slot of ``out``.
    """
    m = D8.shape[0]
    for i in prange(m):
        acc = 0.0
        for j in range(m):
            acc += D8[i, j] * w[j]
        out[i] = acc
    return out


@njit(parallel=True, cache=True)
def dense_int8_structure_counts(D8):
    """Count forbidden values, wrong diagonals, and asymmetric pairs in D8.

    The reductions avoid allocating full-size boolean comparison arrays during
    backend construction.  Only the upper triangle is visited; both entries of
    each off-diagonal pair are inspected there.
    """
    m = D8.shape[0]
    n_forbidden = 0
    n_bad_diag = 0
    n_asymmetric = 0
    for i in prange(m):
        if D8[i, i] == -128:
            n_forbidden += 1
        if D8[i, i] != 127:
            n_bad_diag += 1
        for j in range(i + 1, m):
            upper = D8[i, j]
            lower = D8[j, i]
            if upper == -128:
                n_forbidden += 1
            if lower == -128:
                n_forbidden += 1
            if upper != lower:
                n_asymmetric += 1
    return n_forbidden, n_bad_diag, n_asymmetric


@njit(parallel=True, cache=True)
def int8_forbidden_count(a):
    """Number of forbidden ``-128`` entries in a one-dimensional D8 payload."""
    n = 0
    for i in prange(a.size):
        if a[i] == -128:
            n += 1
    return n


@njit(parallel=True, cache=True)
def pack_upper_int8(D8, out):
    """Pack a square matrix's upper triangle row-by-row without index arrays."""
    m = D8.shape[0]
    for i in prange(m):
        base = i * m - (i * (i - 1)) // 2
        for j in range(i, m):
            out[base + (j - i)] = D8[i, j]
    return out


@njit(parallel=True, cache=True)
def unpack_upper_int8(p8, m, out):
    """Mirror a packed upper triangle into ``out`` without index arrays."""
    for i in prange(m):
        base = i * m - (i * (i - 1)) // 2
        for j in range(i, m):
            value = p8[base + (j - i)]
            out[i, j] = value
            out[j, i] = value
    return out


@njit(parallel=True, cache=True)
def packed_row_sq_sums_int8(p8, m):
    """Integer row sums of squares directly from a packed upper triangle.

    Rows are independent and parallel. Off-diagonal entries are consequently
    read once for each owning row, trading twice the payload reads for thread
    parallelism without a per-thread ``m``-vector reduction. The length-``m``
    int64 result is the only work array; no dense matrix or triangular index
    vectors are formed.
    """
    out = np.empty(m, dtype=np.int64)
    for i in prange(m):
        acc = np.int64(0)
        for j in range(i):
            base_j = j * m - (j * (j - 1)) // 2
            value = np.int64(p8[base_j + (i - j)])
            acc += value * value
        base_i = i * m - (i * (i - 1)) // 2
        for j in range(i, m):
            value = np.int64(p8[base_i + (j - i)])
            acc += value * value
        out[i] = acc
    return out


@njit(parallel=True, cache=True)
def packed_matvec_int8(p8, w, m, out):
    """``out = D @ w`` for an int8 upper triangle packed row-major.

    Row ``i`` starts at ``i*m - i*(i-1)//2`` and holds ``D8[i, i:]``. Entries
    below the diagonal are read from the transposed position: ``D8[i, j]`` for
    ``j < i`` lives at row ``j``'s offset ``(i - j)``.
    """
    for i in prange(m):
        acc = 0.0
        for j in range(i):
            base_j = j * m - (j * (j - 1)) // 2
            acc += p8[base_j + (i - j)] * w[j]
        base_i = i * m - (i * (i - 1)) // 2
        for j in range(i, m):
            acc += p8[base_i + (j - i)] * w[j]
        out[i] = acc
    return out


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


@njit(cache=True)
def packed_quad_int8_sparse(p8, w, m, nz):
    """Packed-D8 quadratic form restricted to sorted support ``nz``.

    The diagonal is added once and each supported off-diagonal pair twice.
    Work is ``O(len(nz)^2)`` rather than proportional to the full triangle.
    """
    total = 0.0
    for a in range(nz.size):
        i = nz[a]
        wi = w[i]
        base = i * m - (i * (i - 1)) // 2
        total += p8[base] * wi * wi
        for b in range(a + 1, nz.size):
            j = nz[b]
            total += 2.0 * p8[base + (j - i)] * wi * w[j]
    return total
