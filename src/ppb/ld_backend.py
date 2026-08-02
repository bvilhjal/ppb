"""LD backends: compute ``w^T D w`` from different LD representations.

The estimator is agnostic to how ``D`` is stored -- it only calls ``.quad(w)``.

- :class:`DenseLD`   -- an explicit dense ``D`` (reference / exact path).
- :class:`LowRankLD` -- a low-rank factor ``R ~= U U^T``. PSD by construction,
  so ``w^T D w >= 0`` always.

Block-diagonal composition and the int8 D8 on-disk store (reusing ldpred3's
representation) build on these next.
"""

from __future__ import annotations

import numpy as np

from ._kernels import (
    dense_int8_structure_counts,
    dense_matvec_int8,
    dense_quad_int8_sparse,
    dense_quad_sparse,
    int8_forbidden_count,
    pack_upper_int8,
    packed_matvec_int8,
    packed_quad_int8,
    packed_quad_int8_sparse,
    packed_row_sq_sums_int8,
    unpack_upper_int8,
)

_Q8 = 127.0  # int8 quantisation: correlations in [-1, 1] <-> [-127, 127]
_SPARSE_QUAD_DIVISOR = 8
_LDSCORE_WORK_BYTES = 64 * 1024 ** 2


def _sparse_support(w):
    """Return sparse support, or ``None`` when the dense path is preferable.

    The one-eighth cutoff is deliberately conservative: support-index kernels
    trade contiguous row sweeps for gathers. Benchmarks over float and D8 blocks
    put the crossover above this density, while this threshold retains a clear
    win and caps the support array at ``m`` bytes on a 64-bit build.
    """
    nnz = int(np.count_nonzero(w))
    if nnz == 0:
        return np.empty(0, dtype=np.intp)
    if nnz * _SPARSE_QUAD_DIVISOR <= w.size:
        return np.flatnonzero(w)
    return None


def _all_finite(a, chunk_rows=512):
    """Finite check with a bounded boolean temporary for a large matrix."""
    if a.ndim < 2:
        return bool(np.isfinite(a).all())
    for start in range(0, a.shape[0], chunk_rows):
        if not np.isfinite(a[start:start + chunk_rows]).all():
            return False
    return True


class LDBackend:
    """Interface: a length ``m`` LD operator exposing the quadratic form."""

    m: int

    def ld_scores(self) -> np.ndarray:  # pragma: no cover - interface
        """Per-variant LD score ``l_j = sum_k D_jk^2`` (C1).

        The diagonal is included, so ``l_j >= 1`` for a correlation matrix. This
        is the regressor in LD-score regression (C2), and it is a property of the
        reference alone -- no summary statistics enter.
        """
        raise NotImplementedError

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
        support = _sparse_support(w)
        if support is not None:
            if support.size == 0:
                return 0.0
            return float(dense_quad_sparse(self.D, w, support))
        # NumPy's BLAS is available through the required numpy dependency; it
        # does not need scipy and is substantially faster than a scalar njit
        # sweep for a dense weight vector.
        return float(np.dot(w, self.D @ w))

    def ld_scores(self) -> np.ndarray:
        return np.einsum("ij,ij->i", self.D, self.D)


class LowRankLD(LDBackend):
    """Low-rank LD approximation ``R ~= U U^T`` with ``U`` of shape (m, r).

    ``quad(w) = w^T U U^T w = ||U^T w||^2`` -- O(m*r) time, O(m*r) memory, and
    non-negative by construction (no negative-denominator failure).
    """

    def __init__(self, U):
        U = np.asarray(U, dtype=np.float64)
        if U.ndim != 2:
            raise ValueError(f"U must be 2-D (m, r); got shape {U.shape}")
        if not 1 <= U.shape[1] <= U.shape[0]:
            raise ValueError(
                f"rank must be in [1, m]; got U.shape={U.shape}")
        # BLAS consumes U.T. Keeping factor columns contiguous avoids the
        # strided C-order access that otherwise dominates a low-rank sweep.
        U = np.asfortranarray(U)
        # An all-zero row means U U^T has a zero on the diagonal there, so quad()
        # drops that variant's self term and understates w^T D w -- an inflated
        # R^2 with nothing to show for it. Always a defect for an LD operator.
        empty = np.flatnonzero(~np.any(U, axis=1))
        if empty.size:
            raise ValueError(
                f"U has {empty.size} all-zero row(s) (e.g. index {int(empty[0])}); "
                "those variants would contribute nothing to w^T D w. Build the "
                "factor with lowrank_ld(), which guarantees every row support.")
        self.U = U
        self.m = U.shape[0]
        self.rank = U.shape[1]

    def quad(self, w) -> float:
        w = self._check(w)
        projected = self.U.T @ w
        return float(np.dot(projected, projected))

    def ld_scores(self) -> np.ndarray:
        # D = U U^T, so sum_k (u_j.u_k)^2 = u_j^T (U^T U) u_j -- O(m r^2), never
        # forming D. Exact for the factor, hence approximate for the LD it stands
        # in for, by exactly the low-rank truncation error.
        gram = self.U.T @ self.U
        out = np.empty(self.m, dtype=np.float64)
        # ``U @ gram`` routes the expensive contraction through BLAS. Chunk it
        # so the temporary stays bounded for a genome-scale factor.
        rows = max(1, _LDSCORE_WORK_BYTES // (8 * self.rank))
        for start in range(0, self.m, rows):
            stop = min(start + rows, self.m)
            transformed = self.U[start:stop] @ gram
            out[start:stop] = np.einsum(
                "ij,ij->i", self.U[start:stop], transformed)
        return out


class BlockDiagonalLD(LDBackend):
    """Block-diagonal LD: ``w^T D w = sum_b w[idx_b]^T D_b w[idx_b]``.

    ``blocks`` is a sequence of ``(backend, idx)`` pairs, where ``backend`` is any
    :class:`LDBackend` over the block and ``idx`` are that block's variant
    positions in the global length-``m`` vector. This mirrors ldpred3's
    recombination-aware block LD: off-block covariance is taken to be zero, and
    each block may independently be dense (D8) or low-rank.
    """

    def __init__(self, blocks):
        prepared = []
        selectors = []
        m = 0
        for backend, idx in blocks:
            raw = np.asarray(idx)
            if raw.ndim != 1 or raw.size == 0:
                raise ValueError("each block idx must be a non-empty 1-D array")
            if raw.dtype.kind not in "iu":
                raise ValueError("each block idx must contain only integers")
            idx = np.ascontiguousarray(raw, dtype=np.intp)
            if idx.size != backend.m:
                raise ValueError(
                    f"block backend has m={backend.m} but idx has {idx.size} entries")
            if idx.min() < 0:
                raise ValueError("block idx has negative positions")
            ordered = idx if np.all(idx[1:] > idx[:-1]) else np.sort(idx)
            if np.any(ordered[1:] == ordered[:-1]):
                raise ValueError(
                    "block idx contains duplicate positions; one backend row "
                    "must correspond to one global variant")
            top = int(idx.max()) + 1
            m = max(m, top)
            prepared.append((backend, idx))
            if idx.size == 1 or np.all(idx[1:] == idx[:-1] + 1):
                selectors.append(slice(int(idx[0]), int(idx[-1]) + 1))
            else:
                selectors.append(idx)

        if not prepared:
            raise ValueError("BlockDiagonalLD needs at least one block")

        # Allocate coverage once. Growing ``seen`` for every block copies all
        # preceding entries and turns construction into O(number_of_blocks * m).
        seen = np.zeros(m, dtype=bool)
        for _backend, idx in prepared:
            if seen[idx].any():
                raise ValueError("blocks overlap: a variant appears in two blocks")
            seen[idx] = True
        if not seen.all():
            raise ValueError(
                f"blocks must cover every variant in [0, {seen.size}); "
                f"{int((~seen).sum())} position(s) have no LD block")

        self.blocks = prepared
        self._selectors = selectors
        self.m = m

    def quad(self, w) -> float:
        return float(self.block_quads(w).sum())

    def ld_scores(self) -> np.ndarray:
        """Off-block entries are zero, so every LD score is a within-block sum.

        That is a real restriction and not a rounding one: a variant's true LD
        score includes long-range and cross-chromosome terms this reference
        defines away, so these are systematically *low*. See ``docs/CALIBRATION.md``.
        """
        out = np.zeros(self.m, dtype=np.float64)
        for (backend, _idx), selector in zip(self.blocks, self._selectors):
            out[selector] = backend.ld_scores()
        return out

    def block_quads(self, w) -> np.ndarray:
        """Per-block ``w[idx_b]^T D_b w[idx_b]``, in block order.

        ``quad`` is their sum. Exposed separately because off-block covariance
        is zero, so these are the variances of *independent* contributions --
        which is what the block jackknife (G2) and the score-distribution
        normality diagnostic both need.
        """
        w = self._check(w)
        out = np.empty(len(self.blocks), dtype=np.float64)
        for b, ((backend, _idx), selector) in enumerate(
                zip(self.blocks, self._selectors)):
            wb = w[selector]
            if not np.any(wb):
                out[b] = 0.0
                continue
            qb = backend.quad(wb)
            # Only the *total* being non-positive is checked downstream, and a
            # single indefinite block is easily masked by hundreds of positive
            # ones -- 431 in the shipped reference. That deflates w^T D w and so
            # inflates R^2, silently, in the one direction this project fails
            # closed on everywhere else. D8 blocks carry no PSD certificate
            # (docs/METHOD.md), so check where the signal actually is. The
            # tolerance admits float rounding on a genuinely PSD block, whose
            # error is O(eps * m * ||w||^2), and nothing more.
            if qb < 0.0:
                if qb < -1e-12 * float(wb @ wb):
                    raise ValueError(
                        f"block {b} has w^T D_b w = {qb!r} < 0: the LD block is not "
                        "positive semi-definite, and summing it would understate "
                        "w^T D w (inflating R^2). Use a PSD representation such as "
                        "a low-rank factor for this block.")
            out[b] = qb
        return out


def _int8_row_sq_sums(D8, chunk=512):
    """Row sums of squared dequantised entries, without materialising floats.

    The largest block in the shipped reference is 17,304 variants; a float64 copy
    would be 2.4 GB. Squares accumulate in int64 instead (127^2 per entry), and
    the rows are walked in chunks.
    """
    D8 = np.asarray(D8)
    m = D8.shape[0]
    out = np.empty(m, dtype=np.float64)
    for start in range(0, m, chunk):
        block = D8[start:start + chunk].astype(np.int32)
        out[start:start + chunk] = np.einsum(
            "ij,ij->i", block, block, dtype=np.int64)
    return out / (_Q8 * _Q8)


def _clip_int8(a):
    """Round to int8, mapping the forbidden -128 to -127 (as ldpred3 does)."""
    q = np.rint(a).astype(np.int64)
    q = np.clip(q, -127, 127)
    return q.astype(np.int8)


class DenseLDInt8(LDBackend):
    """int8-quantised dense LD (D8): ``D8[i, j] = round(corr * 127)``.

    ``quad(w) = (1/127) sum_ij D8[i, j] w[i] w[j]``. The diagonal (127) dequantises
    to exactly 1. ~8x smaller than float64.
    """

    def __init__(self, D8):
        D8 = np.asarray(D8)
        if D8.dtype != np.int8:
            raise ValueError(
                f"D8 must have dtype int8; got {D8.dtype}. "
                "Use DenseLDInt8.from_dense() for floating-point correlations.")
        D8 = np.ascontiguousarray(D8)
        if D8.ndim != 2 or D8.shape[0] != D8.shape[1]:
            raise ValueError(f"D8 must be square; got {D8.shape}")
        self.D8 = D8
        self.m = D8.shape[0]
        self._validate()

    def _validate(self):
        """Validate the public D8 invariant without matrix-sized temporaries."""
        forbidden, bad_diag, asymmetric = dense_int8_structure_counts(self.D8)
        if forbidden:
            raise ValueError(
                f"int8 LD must not contain -128; found {int(forbidden)} entry/entries")
        if bad_diag:
            raise ValueError(
                f"int8 LD has {int(bad_diag)} diagonal entry/entries != 127; "
                "the diagonal must dequantise to exactly 1")
        if asymmetric:
            raise ValueError(
                f"int8 LD is not symmetric; found {int(asymmetric)} "
                "asymmetric upper/lower pair(s)")

    @classmethod
    def from_dense(cls, D) -> "DenseLDInt8":
        D = np.asarray(D, dtype=np.float64)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(f"D must be square; got {D.shape}")
        if not _all_finite(D):
            raise ValueError("D must contain only finite correlations")
        return cls(_clip_int8(np.clip(D, -1.0, 1.0) * _Q8))

    @property
    def nbytes(self) -> int:
        return int(self.D8.nbytes)

    def quad(self, w) -> float:
        w = self._check(w)
        support = _sparse_support(w)
        if support is not None:
            if support.size == 0:
                return 0.0
            return float(dense_quad_int8_sparse(self.D8, w, support) / _Q8)
        out = np.empty(self.m, dtype=np.float64)
        dense_matvec_int8(self.D8, w, out)
        return float(np.dot(w, out) / _Q8)

    def ld_scores(self) -> np.ndarray:
        return _int8_row_sq_sums(self.D8)

    def matvec(self, w) -> np.ndarray:
        """``D w`` for the dequantised block."""
        w = self._check(w)
        out = np.empty(self.m, dtype=np.float64)
        dense_matvec_int8(self.D8, w, out)
        out /= _Q8
        return out

    def packed(self) -> "PackedDenseLDInt8":
        """This block as a :class:`PackedDenseLDInt8` (half the bytes, same quad)."""
        return PackedDenseLDInt8.from_dense_int8(self.D8)


class PackedDenseLDInt8(LDBackend):
    """D8 stored as its packed upper triangle: ``m(m+1)/2`` bytes, not ``m^2``.

    Exactly the same operator as :class:`DenseLDInt8` -- an LD matrix is
    symmetric, so the lower triangle is redundant. It stores the identical int8
    values and is **lossless**: ``to_dense_int8()`` round-trips a square block
    byte-for-byte.

    ``quad`` is *not* bit-identical to the square form, though. It sums each
    off-diagonal pair once and doubles it, where the square kernel adds both
    copies, so the two differ in floating-point summation order. Measured over
    blocks of m = 50..3000, the relative difference is at most **17 machine
    epsilon (~4e-15)** -- eleven orders of magnitude below int8 quantisation's
    own ~0.1% error, but not exactly zero, so a reference repacked to this form
    moves published R^2 values in their last digit or two.

    Halves the on-disk and in-memory size of an LD reference at no accuracy
    cost. Its dense-weight kernel is parallel over rows and reads only one
    triangle; the square backend's parallel matrix-vector path reads both.
    """

    def __init__(self, p8, m):
        p8 = np.asarray(p8)
        if p8.dtype != np.int8:
            raise ValueError(f"packed LD must have dtype int8; got {p8.dtype}")
        p8 = np.ascontiguousarray(p8)
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)):
            raise ValueError(f"packed LD size m must be an integer; got {m!r}")
        m = int(m)
        if p8.ndim != 1:
            raise ValueError(f"packed LD must be 1-D; got shape {p8.shape}")
        if m < 1:
            raise ValueError(f"packed LD size m must be >= 1; got {m}")
        expected = m * (m + 1) // 2
        if p8.size != expected:
            raise ValueError(
                f"packed LD for m={m} needs {expected} entries; got {p8.size}")
        self.p8 = p8
        self.m = m
        self._validate()

    def _validate(self):
        """Validate the packed D8 invariant with O(m) auxiliary memory."""
        forbidden = int(int8_forbidden_count(self.p8))
        if forbidden:
            raise ValueError(
                f"int8 LD must not contain -128; found {forbidden} entry/entries")
        # Every packed row starts with its diagonal. A corrupt diagonal cannot be
        # repaired or inferred from the missing triangle.
        m = self.m
        diag_idx = np.arange(m, dtype=np.intp)
        diag_idx = diag_idx * m - diag_idx * (diag_idx - 1) // 2
        bad = np.flatnonzero(self.p8[diag_idx] != 127)
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"packed LD has {bad.size} diagonal entry/entries != 127 "
                f"(e.g. index {i} = {int(self.p8[diag_idx[i]])}); the int8 LD "
                "diagonal must dequantise to exactly 1")

    @classmethod
    def from_dense_int8(cls, D8) -> "PackedDenseLDInt8":
        """Validate and pack a square int8 block without triangle-index arrays."""
        D8 = DenseLDInt8(D8).D8
        m = D8.shape[0]
        p8 = np.empty(m * (m + 1) // 2, dtype=np.int8)
        pack_upper_int8(D8, p8)
        # The input invariant was checked and packing only copies its validated
        # upper triangle. Avoid rescanning O(m^2) bytes in the public constructor.
        return cls._from_validated(p8, m)

    @classmethod
    def _from_validated(cls, p8, m):
        """Internal constructor for a freshly packed, already validated payload."""
        obj = cls.__new__(cls)
        obj.p8 = p8
        obj.m = m
        return obj

    def to_dense_int8(self) -> np.ndarray:
        """Rebuild the full square int8 block with bounded auxiliary memory."""
        out = np.empty((self.m, self.m), dtype=np.int8)
        return unpack_upper_int8(self.p8, self.m, out)

    @property
    def nbytes(self) -> int:
        return int(self.p8.nbytes)

    def quad(self, w) -> float:
        w = self._check(w)
        support = _sparse_support(w)
        if support is not None:
            if support.size == 0:
                return 0.0
            return float(
                packed_quad_int8_sparse(self.p8, w, self.m, support) / _Q8)
        return float(packed_quad_int8(self.p8, w, self.m) / _Q8)

    def ld_scores(self) -> np.ndarray:
        sums = packed_row_sq_sums_int8(self.p8, self.m)
        return np.asarray(sums, dtype=np.float64) / (_Q8 * _Q8)

    def matvec(self, w) -> np.ndarray:
        """``D w`` for the dequantised block."""
        w = self._check(w)
        out = np.empty(self.m, dtype=np.float64)
        packed_matvec_int8(self.p8, w, self.m, out)
        out /= _Q8
        return out


def min_eig_upper_bound(backend, *, iters: int = 48, seed: int = 0) -> float:
    """Lanczos estimate of the smallest eigenvalue of an int8 LD block.

    Returns the smallest Ritz value after ``iters`` Lanczos steps with full
    reorthogonalization. By the Rayleigh-Ritz property this is an **upper
    bound** on the true smallest eigenvalue, so it is a *detector*, not a
    certificate: a value below zero proves the block is indefinite, while a
    value above zero proves nothing. That asymmetry is the useful direction
    here -- the failure it guards against is an indefinite block silently
    deflating ``w^T D w``.

    Costs ``O(m^2 * iters)`` matrix-vector products against the int8 payload,
    which is affordable once at conversion time (seconds for the shipped
    reference's largest block) but not on every read. Exact eigendecomposition
    is cubic and remains the right check for modest blocks.
    """
    matvec = getattr(backend, "matvec", None)
    if matvec is None:
        raise TypeError(
            f"{type(backend).__name__} has no matvec(); the Lanczos estimate "
            "needs one (DenseLDInt8 and PackedDenseLDInt8 provide it)")
    m = backend.m
    k = max(2, min(int(iters), m))
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(m)
    q /= np.linalg.norm(q)

    Q = np.empty((k, m), dtype=np.float64)
    alpha = np.empty(k, dtype=np.float64)
    beta = np.zeros(k, dtype=np.float64)
    steps = k
    for i in range(k):
        Q[i] = q
        v = np.asarray(matvec(q), dtype=np.float64)
        alpha[i] = float(q @ v)
        v -= alpha[i] * q
        if i:
            v -= beta[i - 1] * Q[i - 1]
        # Lanczos loses orthogonality fast in floating point, and a spurious
        # duplicated Ritz value would be reported as a fresh eigenvalue.
        v -= Q[:i + 1].T @ (Q[:i + 1] @ v)
        nrm = float(np.linalg.norm(v))
        if nrm <= 1e-10 * max(1.0, abs(alpha[i])):
            steps = i + 1                   # invariant subspace: exact here
            break
        beta[i] = nrm
        q = v / nrm

    T = np.diag(alpha[:steps])
    if steps > 1:
        off = beta[:steps - 1]
        T += np.diag(off, 1) + np.diag(off, -1)
    return float(np.linalg.eigvalsh(T)[0])


def lowrank_ld(corr, variance=0.99, max_rank=None, min_eig=1e-6) -> LowRankLD:
    """Eigen-truncate a dense correlation block into a :class:`LowRankLD`.

    Keeps the fewest top eigenvectors explaining ``variance`` of the spectrum
    (capped at ``max_rank``), folds ``sqrt(eigenvalue)`` into ``U``, and
    row-normalizes so the reconstruction ``U U^T`` has unit diagonal -- the
    construction LR8 and SBayesRC share (mirrors ``ldpred3.ld_repr.lowrank_ld``).
    Only the construction: ``U`` stays float64 here, where LR8 goes on to
    quantise it to int8, which ppb does not implement (``docs/METHOD.md`` s2).
    ``variance=1.0`` keeps full rank and reproduces ``corr`` exactly.

    The retained rank is raised past the ``variance`` rule if that rule would
    leave any variant with no support among the kept eigenvectors -- see the
    comment at the extension loop for why that case is dangerous. Raises if
    ``max_rank`` forbids the extension, or if a variant has non-positive
    variance (no unit diagonal exists for it; drop such variants first).
    """
    corr = np.ascontiguousarray(np.asarray(corr, dtype=np.float64))
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or corr.shape[0] == 0:
        raise ValueError("corr must be a non-empty square matrix")
    if not np.isfinite(corr).all():
        raise ValueError("corr must be finite")
    # np.linalg.eigh reads a single triangle, so an asymmetric input would be
    # silently reinterpreted as its own lower triangle -- a wrong answer with no
    # error. Check rather than let that through.
    if not np.allclose(corr, corr.T, rtol=1e-8, atol=1e-10):
        raise ValueError(
            "corr must be symmetric; eigh would silently use only its lower "
            "triangle and return a factor for a different matrix")
    variance = float(variance)
    if not 0.0 < variance <= 1.0:
        raise ValueError("variance must be in (0, 1]")
    m = corr.shape[0]
    diag = np.diag(corr).copy()
    weak = np.flatnonzero(diag <= 0.0)
    if weak.size:
        raise ValueError(
            f"corr has {weak.size} variant(s) with non-positive variance (e.g. "
            f"index {int(weak[0])}, diagonal {diag[weak[0]]!r}); no unit "
            "diagonal can be reconstructed for them -- drop them first")
    evals, evecs = np.linalg.eigh(corr)
    # Tiny negative eigenvalues arise from floating-point roundoff, but silently
    # clipping a genuinely indefinite matrix manufactures a different LD
    # operator. Permit numerical dust only; a correlation input with material
    # negative curvature is invalid and must be repaired upstream.
    psd_tol = 1e-8 * max(1.0, float(np.max(np.abs(evals))))
    if evals[0] < -psd_tol:
        raise ValueError(
            "corr must be positive semi-definite; smallest eigenvalue is "
            f"{float(evals[0]):.6g} (tolerance {-psd_tol:.6g})")
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

    # Truncation can leave a variant with no support among the kept
    # eigenvectors -- its row of U is then (near) zero. Normalizing that row is
    # either impossible (exactly zero stays zero, so U U^T has diagonal 0 there,
    # quad() silently drops the variant's self term and R^2 comes out too high)
    # or meaningless (a near-zero row gets amplified into a numerically
    # arbitrary direction). Neither is acceptable in a denominator, so extend
    # the rank by the minimum that supports every variant. Costs nothing on
    # well-conditioned LD; it only bites degenerate spectra, e.g. an identity
    # block, where the variance rule drops a whole basis direction.
    def _row_energy(k):
        return np.einsum("ij,ij,j->i", evecs[:, :k], evecs[:, :k], evals[:k])

    # An explicit max_rank is a hard cap: extend up to it, then raise rather
    # than quietly returning more rank than the caller allowed.
    r_cap = m if max_rank is None else min(m, int(max_rank))
    floor = 1e-8 * diag                      # keep >=1e-8 of each variant's own variance
    row_e = _row_energy(r)
    short = np.flatnonzero(row_e < floor)
    while short.size and r < r_cap:
        row_e[short] += evecs[short, r] ** 2 * evals[r]
        r += 1
        short = short[row_e[short] < floor[short]]
    if short.size:
        raise ValueError(
            f"max_rank={max_rank} leaves {short.size} variant(s) with no "
            f"support in the retained eigenvectors (e.g. index {int(short[0])}); "
            "their reconstructed LD diagonal would be 0 rather than 1, "
            "understating w^T D w. Raise max_rank.")

    U = evecs[:, :r] * np.sqrt(np.maximum(evals[:r], min_eig))
    # Every row now carries real energy, so normalize by the true norm -- no
    # clip, which would silently under-normalize a small row instead.
    return LowRankLD(U / np.sqrt(np.einsum("ij,ij->i", U, U))[:, None])
