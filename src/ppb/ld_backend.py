"""LD backends: compute ``w^T D w`` from different LD representations.

The estimator is agnostic to how ``D`` is stored -- it only calls ``.quad(w)``.

- :class:`DenseLD`   -- an explicit dense ``D`` (reference / exact path).
- :class:`LowRankLD` -- a low-rank factor ``R ~= U U^T`` (the LR8 idea; ``U`` may
  later be int8-quantised). PSD by construction, so ``w^T D w >= 0`` always.

Block-diagonal composition and the int8 D8/LR8 on-disk store (reusing ldpred3's
representation) build on these next.
"""

from __future__ import annotations

import numpy as np

from ._kernels import dense_quad, lowrank_quad_par


class LDBackend:
    """Interface: a length ``m`` LD operator exposing the quadratic form."""

    m: int

    def quad(self, w) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def _check(self, w) -> np.ndarray:
        w = np.ascontiguousarray(np.asarray(w, dtype=np.float64))
        if w.shape != (self.m,):
            raise ValueError(
                f"weight vector has shape {w.shape}, expected ({self.m},)")
        return w


class DenseLD(LDBackend):
    """Dense LD matrix ``D`` (m x m). Exact reference backend."""

    def __init__(self, D):
        D = np.ascontiguousarray(np.asarray(D, dtype=np.float64))
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(f"D must be a square 2-D array; got shape {D.shape}")
        self.D = D
        self.m = D.shape[0]

    def quad(self, w) -> float:
        w = self._check(w)
        return float(dense_quad(self.D, w))


class LowRankLD(LDBackend):
    """Low-rank LD approximation ``R ~= U U^T`` with ``U`` of shape (m, r).

    ``quad(w) = w^T U U^T w = ||U^T w||^2`` -- O(m*r) time, O(m*r) memory, and
    non-negative by construction (no negative-denominator failure).
    """

    def __init__(self, U):
        U = np.ascontiguousarray(np.asarray(U, dtype=np.float64))
        if U.ndim != 2:
            raise ValueError(f"U must be 2-D (m, r); got shape {U.shape}")
        if not 1 <= U.shape[1] <= U.shape[0]:
            raise ValueError(
                f"rank must be in [1, m]; got U.shape={U.shape}")
        self.U = U
        self.m = U.shape[0]
        self.rank = U.shape[1]

    def quad(self, w) -> float:
        w = self._check(w)
        return float(lowrank_quad_par(self.U, w))


class BlockDiagonalLD(LDBackend):
    """Block-diagonal LD: ``w^T D w = sum_b w[idx_b]^T D_b w[idx_b]``.

    ``blocks`` is a sequence of ``(backend, idx)`` pairs, where ``backend`` is any
    :class:`LDBackend` over the block and ``idx`` are that block's variant
    positions in the global length-``m`` vector. This mirrors ldpred3's
    recombination-aware block LD: off-block covariance is taken to be zero, and
    each block may independently be dense (D8) or low-rank (LR8).
    """

    def __init__(self, blocks):
        self.blocks = []
        seen = np.zeros(0, dtype=bool)
        m = 0
        for backend, idx in blocks:
            idx = np.ascontiguousarray(np.asarray(idx, dtype=np.intp))
            if idx.ndim != 1 or idx.size == 0:
                raise ValueError("each block idx must be a non-empty 1-D array")
            if idx.size != backend.m:
                raise ValueError(
                    f"block backend has m={backend.m} but idx has {idx.size} entries")
            if idx.min() < 0:
                raise ValueError("block idx has negative positions")
            top = int(idx.max()) + 1
            if top > seen.size:
                seen = np.concatenate([seen, np.zeros(top - seen.size, dtype=bool)])
            if seen[idx].any():
                raise ValueError("blocks overlap: a variant appears in two blocks")
            seen[idx] = True
            m = max(m, top)
            self.blocks.append((backend, idx))
        if not self.blocks:
            raise ValueError("BlockDiagonalLD needs at least one block")
        self.m = m

    def quad(self, w) -> float:
        w = self._check(w)
        total = 0.0
        for backend, idx in self.blocks:
            total += backend.quad(np.ascontiguousarray(w[idx]))
        return total


def lowrank_ld(corr, variance=0.99, max_rank=None, min_eig=1e-6) -> LowRankLD:
    """Eigen-truncate a dense correlation block into a :class:`LowRankLD`.

    Keeps the fewest top eigenvectors explaining ``variance`` of the spectrum
    (capped at ``max_rank``), folds ``sqrt(eigenvalue)`` into ``U``, and
    row-normalizes so the reconstruction ``U U^T`` has unit diagonal -- the
    LR8 / SBayesRC-style construction (mirrors ``ldpred3.ld_repr.lowrank_ld``).
    ``variance=1.0`` keeps full rank and reproduces ``corr`` exactly.
    """
    corr = np.ascontiguousarray(np.asarray(corr, dtype=np.float64))
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or corr.shape[0] == 0:
        raise ValueError("corr must be a non-empty square matrix")
    if not np.isfinite(corr).all():
        raise ValueError("corr must be finite")
    variance = float(variance)
    if not 0.0 < variance <= 1.0:
        raise ValueError("variance must be in (0, 1]")
    m = corr.shape[0]
    evals, evecs = np.linalg.eigh(corr)
    evals = np.maximum(evals[::-1], 0.0)
    evecs = evecs[:, ::-1]
    total = float(evals.sum())
    if total <= 0.0:
        r = 1
    else:
        r = int(np.searchsorted(np.cumsum(evals), variance * total) + 1)
    r = max(1, min(r, m))
    if max_rank is not None:
        r = min(r, int(max_rank))
    U = evecs[:, :r] * np.sqrt(np.maximum(evals[:r], min_eig))
    d = np.sqrt(np.clip((U * U).sum(axis=1), 1e-12, None))
    return LowRankLD(U / d[:, None])
