"""Ancestry-composition estimation from summary statistics and LD references.

Implements the two LD-moment estimators derived in
``docs/ancestry_report/ancestry_report.tex`` (equation numbers below refer to
that report). Both need only the marginal z-scores and one LD reference per
candidate ancestry -- no allele-frequency column, no individual-level data.

* **Estimator A, pair products** -- eqs. (12)--(16). Within-block products
  ``z_i z_j`` are regressed on the per-ancestry reference correlations
  ``R^(k)_ij``. The moment is linear in the composition pi and is driven by
  the correlated-noise covariance ``(1 - h^2) R^(A)``. It can remain
  identified at ``h^2 = 0`` only under the eligible score-covariance and
  LD-mixture working model; correctly calibrated marginal standard errors
  are necessary, not sufficient. With ``absorb_signal``
  (default) the ``K(K+1)/2`` quadratic columns ``(R^(k) R^(k'))_ij`` join the
  fit as non-negative absorbers -- they span the tagging term ``(n h^2 / m)
  (R^(A))^2`` exactly. Their fitted total exceeding twice its parametric SE is
  an uncalibrated activation heuristic, not evidence of real polygenic signal;
  ``signal = 0`` means only that the absorber was not retained.
* **Estimator B, bilinear LD-score regression** -- eqs. (17)--(20).
  Chi-square statistics are regressed on the ``K(K+1)/2`` bilinear scores
  ``l^(kk')_i = sum_j R^(k)_ij R^(k')_ij``; the coefficient matrix is
  ``(n h^2 / m) pi pi^T`` at truth, so pi is recovered from its row sums.
  Regressing on the plain per-ancestry scores ``l^(k)`` instead drops the
  cross terms and is misspecified whenever the ancestries' correlation
  vectors overlap (report Section 5) -- the tests demonstrate the bias.

References are block-diagonal correlation matrices. Everywhere a reference
is accepted it may be given either as a dense ``(m, m)`` matrix per ancestry
(sliced into ``blocks``) or -- the scalable form -- as a list over blocks of
square matrices per ancestry, with ``blocks`` then defaulting to contiguous
tiling. All computation is per block, so cost is linear in genome size.

Standard errors are delete-one-group jackknives over groups of LD blocks
(the chromosome analogue; eq. (21)), because the residuals are correlated
through LD and i.i.d. formulas do not apply.

Known limitations (report Section 10): the mixture approximation ignores the
within-block frequency-contrast (Wahlund) term and longer-range admixture LD;
Estimator B needs genuine polygenic signal; both assume the references are
harmonised to the same variants and that the eligible summary-statistic
working model is credible.
"""

from __future__ import annotations

import numpy as np


def _nnls(A, b):
    """Lawson-Hanson NNLS: minimize ``||A x - b||^2`` over ``x >= 0``.

    Deterministic, pure NumPy (the family has no scipy dependency).
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = A.shape[1]
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    w = A.T @ b
    tol = 1e-12 * max(1.0, float(np.linalg.norm(A, ord=np.inf)))
    for _ in range(30 * n + 1):
        if passive.all() or (w[~passive] <= tol).all():
            return x
        passive[int(np.argmax(np.where(passive, -np.inf, w)))] = True
        for _inner in range(30 * n + 1):
            idx = np.flatnonzero(passive)
            s = np.zeros(n)
            s[idx], *_ = np.linalg.lstsq(A[:, idx], b, rcond=None)
            if (s[idx] > tol).all():
                x = s
                break
            neg = idx[s[idx] <= tol]
            alpha = float(np.min(x[neg] / (x[neg] - s[neg])))
            x = x + alpha * (s - x)
            zeros = (x <= tol) & passive
            x[zeros] = 0.0
            passive[zeros] = False
        w = A.T @ (b - A @ x)
    return x


def _validate_blocks(blocks, m, *, label="block"):
    """Return integer parts that partition ``range(m)`` exactly."""
    out = []
    for block in blocks:
        raw = np.asarray(block)
        if (raw.ndim != 1 or raw.size == 0 or
                not np.issubdtype(raw.dtype, np.integer)):
            raise ValueError(
                f"each {label} must be a non-empty integer vector"
            )
        out.append(raw.astype(int, copy=False))
    flat = np.concatenate(out) if out else np.array([], dtype=int)
    if (flat.size != m or (flat < 0).any() or (flat >= m).any() or
            not np.array_equal(np.sort(flat), np.arange(m))):
        raise ValueError(
            f"{label}s must partition indices 0..{m - 1} exactly"
        )
    return out


def _as_ref_blocks(refs, blocks):
    """Canonicalise references to ``ref_blocks[k][b]`` plus block indices.

    Accepts, per ancestry, either a dense ``(m, m)`` correlation matrix
    (sliced by ``blocks``) or a list of per-block square matrices (with
    ``blocks`` defaulting to contiguous tiling of the variant axis).
    """
    refs = list(refs)
    if len(refs) < 2:
        raise ValueError("need at least two ancestry references")
    first = refs[0]
    if isinstance(first, np.ndarray) and first.ndim == 2:
        m = first.shape[0]
        if any(np.asarray(R).shape != (m, m) for R in refs):
            raise ValueError("all references must be square on the same variants")
        if blocks is None:
            blocks = [np.arange(m)]
        blocks = _validate_blocks(blocks, m)
        return [[np.asarray(R, dtype=np.float64)[np.ix_(b, b)]
                 for b in blocks] for R in refs], blocks
    n_blocks = len(first)
    if any(len(R) != n_blocks for R in refs):
        raise ValueError("all references must have the same number of blocks")
    sizes = [np.asarray(first[b]).shape[0] for b in range(n_blocks)]
    for R in refs:
        for b in range(n_blocks):
            if np.asarray(R[b]).shape != (sizes[b], sizes[b]):
                raise ValueError("per-block matrices must be square and aligned "
                                 "across ancestries")
    if blocks is None:
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        blocks = [np.arange(bounds[b], bounds[b + 1]) for b in range(n_blocks)]
    if len(blocks) != n_blocks:
        raise ValueError("blocks must match the number of reference blocks")
    blocks = _validate_blocks(blocks, sum(sizes))
    if any(len(blocks[b]) != sizes[b] for b in range(n_blocks)):
        raise ValueError("block indices must match the per-block matrix sizes")
    return [[np.asarray(R[b], dtype=np.float64) for b in range(n_blocks)]
            for R in refs], blocks


def _design_diagnostics(X, *, center=False):
    """Rank and scale-free condition number of a regression design.

    Columns are normalised before the singular-value calculation so that a
    harmless change of units does not masquerade as confusability. Designs
    fitted with a free intercept must be centred first.
    """
    X = np.asarray(X, dtype=np.float64)
    A = X - X.mean(axis=0) if center else X
    norms = np.linalg.norm(A, axis=0)
    p = A.shape[1]
    nonzero = norms > np.finfo(np.float64).tiny
    if not nonzero.any():
        return 0, np.inf
    singular = np.linalg.svd(A[:, nonzero] / norms[nonzero],
                             compute_uv=False)
    tol = max(A.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.sum(singular > tol))
    condition = (np.inf if rank < p or not nonzero.all() else
                 float(singular[0] / singular[-1]))
    return rank, condition


def _conditional_design_diagnostics(target, nuisance):
    """Diagnose ``target`` after projection off the nuisance column space.

    In a joint model ``target @ beta + nuisance @ gamma``, raw full rank of
    ``target`` does not identify ``beta`` when nuisance columns reproduce its
    directions. This returns rank and scale-free condition number for
    ``M_nuisance target``.
    """
    target = np.asarray(target, dtype=np.float64)
    nuisance = np.asarray(nuisance, dtype=np.float64)
    if target.ndim != 2 or nuisance.ndim != 2:
        raise ValueError("target and nuisance designs must be two-dimensional")
    if target.shape[0] != nuisance.shape[0]:
        raise ValueError("target and nuisance designs must have the same rows")
    if nuisance.shape[1] == 0:
        return _design_diagnostics(target)
    left, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
    if singular.size == 0 or singular[0] <= np.finfo(np.float64).tiny:
        return _design_diagnostics(target)
    tolerance = max(nuisance.shape) * np.finfo(np.float64).eps * singular[0]
    basis = left[:, singular > tolerance]
    residualized = target - basis @ (basis.T @ target)
    target_norm = np.linalg.norm(target, axis=0)
    residual_norm = np.linalg.norm(residualized, axis=0)
    projection_tolerance = (10.0 * max(target.shape) *
                            np.finfo(np.float64).eps)
    projected = residual_norm <= projection_tolerance * target_norm
    residualized[:, projected] = 0.0
    return _design_diagnostics(residualized)


def bilinear_ld_scores(refs, blocks=None):
    """Bilinear scores ``l^(kk')_i = sum_j R^(k)_ij R^(k')_ij``, eq. (11).

    The sum runs over the variant's own block (references are
    block-diagonal). Returns ``(L, pairs)`` with ``L`` of shape
    ``(m, K*(K+1)/2)`` and ``pairs`` the ``(k, k')`` index list, ``k <= k'``.
    """
    ref_blocks, blocks = _as_ref_blocks(refs, blocks)
    K = len(ref_blocks)
    pairs = [(k, kp) for k in range(K) for kp in range(k, K)]
    scores = np.empty((sum(len(block) for block in blocks), len(pairs)))
    for b, block in enumerate(blocks):
        scores[block] = np.stack(
            [np.einsum("ij,ij->i", ref_blocks[k][b], ref_blocks[kp][b])
             for k, kp in pairs], axis=1)
    return scores, pairs


def ld_scores(ref, blocks=None):
    """Per-variant LD scores ``l_i = sum_j R_ij^2`` of one reference."""
    L, _ = bilinear_ld_scores([ref, ref], blocks)
    return L[:, 1]


def pair_design(refs, blocks=None, *, floor=0.05, cap=250, quadratic=False):
    """Pair set P and design matrix for Estimator A, eqs. (12)--(15).

    Pairs are taken within LD blocks only (unlinked pairs carry no ancestry
    information, report Section 6), keeping those with
    ``max_k |R^(k)_ij| >= floor`` and at most ``cap`` pairs per block, the
    strongest first. With ``quadratic=True`` the ``K(K+1)/2`` signal-term
    columns ``(R^(k) R^(k'))_ij`` are also returned -- the exact span of the
    tagging term ``(R^(A))^2`` under the mixture model.

    Returns ``(ii, jj, L, pair_block[, L2])``: global pair row indices, the
    ``(n_pairs, K)`` design with ``L[p, k] = R^(k)_{ii,jj}``, the block index
    of each pair, and optionally the ``(n_pairs, K(K+1)/2)`` quadratic design.
    Its off-diagonal ``(k, k')`` column is the Jordan product
    ``(R^(k) R^(k') + R^(k') R^(k)) / 2``. Consequently the coefficient on
    that column is ``2 s pi_k pi_k'``, matching :func:`_bilinear_recover`.
    """
    ref_blocks, blocks = _as_ref_blocks(refs, blocks)
    K = len(ref_blocks)
    pairs = [(k, kp) for k in range(K) for kp in range(k, K)]
    ii_parts, jj_parts, L_parts, L2_parts, blk_parts = [], [], [], [], []
    for b, block in enumerate(blocks):
        block = np.asarray(block)
        stack = np.stack([ref_blocks[k][b] for k in range(K)])
        ti, tj = np.triu_indices(block.size, k=1)
        entries = stack[:, ti, tj]
        strength = np.max(np.abs(entries), axis=0)
        keep = np.flatnonzero(strength >= floor)
        if keep.size > cap:
            keep = keep[np.argsort(strength[keep])[::-1][:cap]]
        if keep.size == 0:
            continue
        ii_parts.append(block[ti[keep]])
        jj_parts.append(block[tj[keep]])
        L_parts.append(entries[:, keep].T)
        blk_parts.append(np.full(keep.size, b, dtype=int))
        if quadratic:
            quadratic_columns = []
            for k, kp in pairs:
                product = ref_blocks[k][b] @ ref_blocks[kp][b]
                if k != kp:
                    reverse = ref_blocks[kp][b] @ ref_blocks[k][b]
                    product = 0.5 * (product + reverse)
                quadratic_columns.append(product[ti[keep], tj[keep]])
            L2_parts.append(np.stack(quadratic_columns, axis=1))
    if not ii_parts:
        raise ValueError("no pairs pass the LD floor; check the references")
    out = (np.concatenate(ii_parts), np.concatenate(jj_parts),
           np.concatenate(L_parts), np.concatenate(blk_parts))
    if quadratic:
        out = out + (np.concatenate(L2_parts),)
    return out


def block_groups(n_blocks, groups=None, n_groups=10):
    """Group LD blocks into jackknife strata (contiguous pseudo-chromosomes).

    ``groups`` may give explicit lists of block indices; otherwise blocks are
    split contiguously into at most ``n_groups`` groups. Returns a list of
    block-index arrays.
    """
    if groups is not None:
        return _validate_blocks(groups, n_blocks, label="group")
    if n_blocks < 2:
        return [np.arange(max(n_blocks, 1))]
    g = min(n_groups, n_blocks)
    return [np.asarray(b, dtype=int) for b in np.array_split(np.arange(n_blocks), g)]


def _delete_one_jackknife(fit, n_blocks, groups):
    """Jackknife ``fit(kept_blocks) -> vector`` over leave-one-group-out fits.

    ``fit`` receives a boolean mask over blocks and returns the estimate
    vector on the remaining data. Returns ``(se, per_group_estimates)``; any
    non-finite refit yields ``se = None``.
    """
    estimates = []
    for grp in groups:
        kept = np.ones(n_blocks, dtype=bool)
        kept[np.asarray(grp, dtype=int)] = False
        estimates.append(np.asarray(fit(kept), dtype=np.float64))
    estimates = np.asarray(estimates)
    if not np.isfinite(estimates).all():
        return None, estimates
    g = len(groups)
    mean = estimates.mean(axis=0)
    se = np.sqrt((g - 1) / g * ((estimates - mean) ** 2).sum(axis=0))
    return se, estimates


def _fit_pair_nnls(L, y):
    """NNLS of pair products on the ancestry design; returns (pi, scale).

    With ``u = s * pi`` the joint problem over ``(pi in simplex, s >= 0)`` is
    exactly NNLS in ``u``: ``s = sum(u)``, ``pi = u / s`` (eq. (12)).
    """
    u = _nnls(L, y)
    s = float(u.sum())
    if s <= 1e-12:
        return None, 0.0
    return u / s, s


def _design_block_count(block_ids, n_blocks, *, label):
    """Validate integer design block IDs and return the represented count."""
    raw = np.asarray(block_ids)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{label} must be a one-dimensional integer vector")
    block_ids = raw.astype(int, copy=False)
    if block_ids.size == 0 or (block_ids < 0).any():
        raise ValueError(f"{label} must contain non-negative block IDs")
    minimum = int(block_ids.max()) + 1
    if n_blocks is None:
        return block_ids, minimum
    if (isinstance(n_blocks, (bool, np.bool_)) or
            not isinstance(n_blocks, (int, np.integer)) or n_blocks < minimum):
        raise ValueError(f"n_blocks must be an integer at least {minimum}")
    return block_ids, int(n_blocks)


def estimate_pair_products_from_design(
        z, ii, jj, design, pair_block, *, quadratic_design=None,
        n_blocks=None, groups=None, absorb_signal=True,
        max_design_condition=1e3):
    """Estimator A from a compact, precomputed reference-LD design.

    ``design[p, k]`` is ancestry ``k``'s LD correlation for summary-statistic
    pair ``(ii[p], jj[p])``. ``quadratic_design`` contains the corresponding
    ``K(K+1)/2`` Jordan-product absorber columns in lexicographic ``(k, k')``
    order. This is the scalable entry point: a reference builder can process
    and discard one LD block at a time. With
    ``absorb_signal=True`` (default) the ``K(K+1)/2`` quadratic columns
    ``(R^(k) R^(k'))_ij`` join the fit as non-negative absorbers: they span
    the tagging term ``(n h^2 / m) (R^(A))^2`` exactly under the mixture
    model, so the composition read from the linear part is protected from the
    tagging bias the raw eq. (12) fit shows at moderate signal. Because the
    absorbers add noise when there is nothing to absorb, an uncalibrated
    heuristic retains them only when their fitted total clears twice its
    parametric standard error (otherwise the fit falls back to the linear
    columns alone). This activation is not a test of real polygenic signal.
    When retained, the linear ancestry design must remain identifiable after
    projection off the quadratic nuisance space. The
    composition read from the *quadratic* part (row sums, as in Estimator B)
    is returned as ``proportions_signal`` -- agreement between the two is a
    specification diagnostic.

    Returns a dict with ``proportions``, ``proportions_se`` (delete-one-group
    jackknife, eq. (21)), the noise ``scale`` (``~ 1 - h^2``), the signal
    absorber coefficient total (0 means the absorber was not retained), pair
    counts, the maximum
    design correlation between linear ancestry columns (confusability
    diagnostic), and the residual RMS. A rank-deficient or scale-normalised
    design condition number above ``max_design_condition`` is refused before
    fitting; the default limit is 1,000.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError("z must be one-dimensional")
    if not np.isfinite(z).all():
        raise ValueError("z must contain only finite values")
    L = np.asarray(design, dtype=np.float64)
    if L.ndim != 2 or L.shape[1] < 2 or L.shape[0] == 0:
        raise ValueError(
            "design must be a non-empty two-dimensional array with at "
            "least two ancestry columns"
        )
    if not np.isfinite(L).all():
        raise ValueError("design must contain only finite values")
    K = L.shape[1]
    P = K * (K + 1) // 2
    raw_ii, raw_jj = np.asarray(ii), np.asarray(jj)
    if (raw_ii.shape != (L.shape[0],) or raw_jj.shape != (L.shape[0],) or
            not np.issubdtype(raw_ii.dtype, np.integer) or
            not np.issubdtype(raw_jj.dtype, np.integer)):
        raise ValueError(
            "ii and jj must be integer vectors matching the design rows"
        )
    ii = raw_ii.astype(int, copy=False)
    jj = raw_jj.astype(int, copy=False)
    if ((ii < 0).any() or (jj < 0).any() or (ii >= z.size).any() or
            (jj >= z.size).any() or (ii == jj).any()):
        raise ValueError("ii and jj must be distinct valid indices into z")
    pair_block, n_blocks = _design_block_count(
        pair_block, n_blocks, label="pair_block"
    )
    if pair_block.shape != (L.shape[0],):
        raise ValueError("pair_block must match the design rows")
    if absorb_signal:
        L2 = np.asarray(quadratic_design, dtype=np.float64)
        if L2.shape != (L.shape[0], P):
            raise ValueError(
                "quadratic_design must have shape "
                f"({L.shape[0]}, {P}) when absorb_signal=True"
            )
        if not np.isfinite(L2).all():
            raise ValueError(
                "quadratic_design must contain only finite values"
            )
    else:
        L2 = None
    y_all = z[ii] * z[jj]
    design_rank, design_condition = _design_diagnostics(L)
    if design_rank < K:
        raise ValueError(
            "reference pair design is not identifiable: "
            f"rank {design_rank}/{K}"
        )
    if design_condition > max_design_condition:
        raise ValueError(
            "reference pair design is too ill-conditioned for stable "
            f"estimation: condition number {design_condition:.3g} "
            f"(limit {max_design_condition:.3g})"
        )
    pairs = [(k, kp) for k in range(K) for kp in range(k, K)]

    def split(u):
        s_u = float(u[:K].sum())
        pi_u = u[:K] / s_u if s_u > 1e-12 else None
        pi_v, _, _ = _bilinear_recover(u[K:], pairs, K)
        return pi_u, s_u, pi_v, float(u[K:].sum()), u

    def fit(kept_blocks):
        mask = np.isin(pair_block, np.flatnonzero(kept_blocks))
        y = y_all[mask]
        if not absorb_signal:
            pi_u, s_u = _fit_pair_nnls(L[mask], y)
            u = np.zeros(K + P)
            u[:K] = 0.0 if pi_u is None else pi_u * s_u
            return pi_u, s_u, None, 0.0, u
        X = np.hstack([L, L2])[mask]
        u = _nnls(X, y)
        s_v = float(u[K:].sum())
        # Uncalibrated activation heuristic: the absorber total is compared
        # with an i.i.d. parametric SE. Passing is not evidence of real signal.
        resid = y - X @ u
        dof = max(int(mask.sum()) - K - P, 1)
        sigma2 = float(resid @ resid) / dof
        XtX_inv = np.linalg.pinv(X.T @ X)
        se_v = float(np.sqrt(max(sigma2 * XtX_inv[K:, K:].sum(), 0.0)))
        if s_v < 2.0 * se_v:
            pi_u, s_u = _fit_pair_nnls(L[mask], y)
            u = np.zeros(K + P)
            u[:K] = 0.0 if pi_u is None else pi_u * s_u
            return pi_u, s_u, None, 0.0, u
        return split(u)

    def conditional_diagnostics(kept_blocks, absorber_retained):
        if not absorber_retained:
            return _design_diagnostics(
                L[np.isin(pair_block, np.flatnonzero(kept_blocks))]
            )
        mask = np.isin(pair_block, np.flatnonzero(kept_blocks))
        return _conditional_design_diagnostics(L[mask], L2[mask])

    pi, s, pi_signal, signal, u_full = fit(np.ones(n_blocks, dtype=bool))
    absorber_retained = signal > 0.0
    conditional_rank, conditional_condition = conditional_diagnostics(
        np.ones(n_blocks, dtype=bool), absorber_retained
    )
    if conditional_rank < K:
        raise ValueError(
            "linear pair design is not identifiable conditional on the "
            "retained quadratic absorbers: "
            f"rank {conditional_rank}/{K}"
        )
    if conditional_condition > max_design_condition:
        raise ValueError(
            "linear pair design is too ill-conditioned conditional on the "
            "retained quadratic absorbers: "
            f"condition number {conditional_condition:.3g} "
            f"(limit {max_design_condition:.3g})"
        )
    if pi is None:
        raise ValueError(
            "degenerate fit: no positive fitted linear pair-product component"
        )
    grps = block_groups(n_blocks, groups)

    def stat(kept_blocks):
        out, _, _, out_signal, _ = fit(kept_blocks)
        rank, condition = conditional_diagnostics(
            kept_blocks, out_signal > 0.0
        )
        if rank < K or condition > max_design_condition:
            return np.full(K, np.nan)
        return np.full(K, np.nan) if out is None else out

    if len(grps) < 2:
        se, per_group = None, None
    else:
        se, per_group = _delete_one_jackknife(stat, n_blocks, grps)
    fitted = np.hstack([L, L2]) @ u_full if absorb_signal else s * (L @ pi)
    center = L - L.mean(axis=0)
    denom = np.sqrt((center ** 2).sum(axis=0))
    denom = np.where(denom == 0.0, np.nan, denom)
    design_corr = (center.T @ center) / np.outer(denom, denom)
    offdiag = design_corr[~np.eye(design_corr.shape[0], dtype=bool)]
    agreement = (None if pi_signal is None else
                 float(np.abs(pi - pi_signal).max()))
    boundary = bool((pi < 1e-6).any())
    boundary_note = ("estimate lies on the simplex boundary; ordinary "
                     "symmetric inference from jackknife SEs is invalid "
                     "there, so use a boundary-aware interval"
                     if boundary else None)
    return {
        "proportions": pi,
        "proportions_se": se,
        "proportions_signal": pi_signal,
        "channel_agreement": agreement,
        "scale": s,
        "signal": signal,
        "absorber_retained": absorber_retained,
        "absorber_note": (
            "retained by an uncalibrated two-parametric-SE heuristic"
            if absorber_retained else
            "not retained; signal=0 does not establish h2=0"
        ),
        "boundary_note": boundary_note,
        "n_pairs": int(y_all.size),
        "n_blocks": n_blocks,
        "design_rank": design_rank,
        "design_columns": K,
        "design_condition": design_condition,
        "conditional_design_rank": conditional_rank,
        "conditional_design_columns": K,
        "conditional_design_condition": conditional_condition,
        "max_design_correlation": float(np.nanmax(np.abs(offdiag))),
        "residual_rms": float(np.sqrt(np.mean((y_all - fitted) ** 2))),
        "jackknife_estimates": per_group,
        "jackknife_note": (None if se is not None else
                           "fewer than two block groups or a non-identifiable "
                           "leave-group fit; no jackknife SEs"),
    }


def estimate_pair_products(z, refs, *, blocks=None, floor=0.05, cap=250,
                           groups=None, absorb_signal=True,
                           max_design_condition=1e3):
    """Estimator A from z-scores and per-ancestry LD references.

    This compatibility wrapper builds the compact pair design from dense or
    block-native references, then delegates to
    :func:`estimate_pair_products_from_design`.
    """
    _, canonical_blocks = _as_ref_blocks(refs, blocks)
    z_array = np.asarray(z)
    if (z_array.ndim != 1 or
            z_array.size != sum(len(block) for block in canonical_blocks)):
        raise ValueError("z must be one-dimensional and match the references")
    if absorb_signal:
        ii, jj, L, pair_block, L2 = pair_design(
            refs, canonical_blocks, floor=floor, cap=cap, quadratic=True
        )
    else:
        ii, jj, L, pair_block = pair_design(
            refs, canonical_blocks, floor=floor, cap=cap, quadratic=False
        )
        L2 = None
    return estimate_pair_products_from_design(
        z_array, ii, jj, L, pair_block,
        quadratic_design=L2,
        n_blocks=len(canonical_blocks),
        groups=groups,
        absorb_signal=absorb_signal,
        max_design_condition=max_design_condition,
    )


def _bilinear_recover(b, pairs, K):
    """Composition from bilinear-regression coefficients, eqs. (18)--(19).

    The coefficient on the unique ``(k, k')`` column equals
    ``(n h^2 / m) pi_k^2`` for ``k == k'`` and ``2 (n h^2 / m) pi_k pi_k'``
    otherwise, so the symmetric coefficient matrix has off-diagonal
    ``b_kk' / 2``; row sums normalised give pi.
    """
    B = np.zeros((K, K))
    for coef, (k, kp) in zip(b, pairs):
        if k == kp:
            B[k, k] = coef
        else:
            B[k, kp] = B[kp, k] = coef / 2.0
    row = B.sum(axis=1)
    total = float(row.sum())
    if total <= 1e-12:
        return None, B, np.nan
    pi = row / total
    rank1 = total * np.outer(pi, pi)
    distance = float(np.linalg.norm(B - rank1) / max(np.linalg.norm(B), 1e-300))
    return pi, B, distance


def _block_cross_validated_signal(X, y, variant_block, groups):
    """Heuristic held-out check that chi-square varies reproducibly with LD.

    Each group is predicted from a non-negative fit to all other groups. The
    returned statistic is the mean held-out correlation divided by its
    between-group standard error. It is not a calibrated normal Z-statistic;
    the implementation declines when fewer than four groups are available.
    """
    if len(groups) < 4:
        return np.nan, np.nan, None
    correlations = []
    for group in groups:
        test = np.isin(variant_block, np.asarray(group, dtype=int))
        train = ~test
        if test.sum() < 2 or train.sum() <= X.shape[1] + 1:
            return np.nan, np.nan, None
        X_train = X[train]
        y_train = y[train]
        b = _nnls(X_train - X_train.mean(axis=0),
                  y_train - y_train.mean())
        predicted = X[test] @ b
        predicted = predicted - predicted.mean()
        observed = y[test] - y[test].mean()
        denominator = float(np.sqrt((predicted @ predicted) *
                                    (observed @ observed)))
        correlations.append(0.0 if denominator <= 1e-15 else
                            float(predicted @ observed) / denominator)
    correlations = np.asarray(correlations, dtype=np.float64)
    mean = float(correlations.mean())
    se = float(correlations.std(ddof=1) / np.sqrt(correlations.size))
    statistic = (np.inf if mean > 0.0 else 0.0) if se <= 1e-15 else mean / se
    return float(statistic), mean, correlations


def estimate_bilinear_from_design(
        z, bilinear_design, variant_block, *, n_blocks=None, groups=None,
        c=None, sample_size=None, max_design_condition=1e3,
        min_signal_z=4.0, max_rank1_distance=0.25,
        max_psd_violation=0.1):
    """Estimator B from precomputed bilinear LD-score columns.

    Regresses ``z^2`` on the ``K(K+1)/2`` bilinear scores with a free
    intercept and non-negativity constraints (eq. (17)), then recovers pi
    from the coefficient row sums (eq. (19)). ``sample_size`` may be a
    positive per-variant vector; its values multiply the predictors after
    division by their median, implementing the per-variant-``N_i`` working
    moment under the same model assumptions while keeping the coefficient
    scale near ``median(N_i) h^2 / m``.

    The exact-expectation default does not winsorize chi-squares. An explicit
    finite ``c`` retains the old capped fit only as an exploratory diagnostic;
    if any value is capped, accepted proportions are withheld because
    ``E[min(z_i^2, c)]`` is not linear in the uncapped mean.

    Acceptance requires a full-rank centred design with scale-normalised
    condition number no larger than 1,000 by default;
    reproducible signal in held-out block groups; and a fitted coefficient
    matrix sufficiently close to positive-semidefinite rank one. Failed
    checks leave the raw diagnostics in the result but set ``proportions``
    to ``None``. The default model checks allow at most 0.25 relative
    Frobenius distance from rank one and negative-eigenvalue magnitude no
    larger than 0.10 of the fitted matrix's spectral scale. These are fixed
    screening thresholds, not calibrated hypothesis tests.
    Jackknife SEs are reported only if every leave-group refit passes these
    same screens; otherwise the SE is withheld rather than mixing accepted and
    rejected fits.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError("z must be one-dimensional")
    if not np.isfinite(z).all():
        raise ValueError("z must contain only finite values")
    Lb = np.asarray(bilinear_design, dtype=np.float64)
    if Lb.ndim != 2 or Lb.shape[0] != z.size or Lb.shape[1] == 0:
        raise ValueError(
            "bilinear_design must be a non-empty matrix matching z rows"
        )
    if not np.isfinite(Lb).all():
        raise ValueError("bilinear_design must contain only finite values")
    P = Lb.shape[1]
    K = int((np.sqrt(1.0 + 8.0 * P) - 1.0) / 2.0)
    if K < 2 or K * (K + 1) // 2 != P:
        raise ValueError(
            "bilinear_design must have K(K+1)/2 columns for an integer K>=2"
        )
    pairs = [(k, kp) for k in range(K) for kp in range(k, K)]
    variant_block, n_blocks = _design_block_count(
        variant_block, n_blocks, label="variant_block"
    )
    if variant_block.shape != (z.size,):
        raise ValueError("variant_block must match z")
    m = z.size
    if sample_size is None:
        sample_size_scale = 1.0
        relative_n = np.ones(m)
    else:
        sample_size = np.asarray(sample_size, dtype=np.float64)
        if (sample_size.shape != (m,) or not np.isfinite(sample_size).all()
                or (sample_size <= 0.0).any()):
            raise ValueError(
                "sample_size must be a positive finite vector matching z"
            )
        sample_size_scale = float(np.median(sample_size))
        relative_n = sample_size / sample_size_scale
    X_all = Lb * relative_n[:, None]
    raw_chi2 = z * z
    if c is not None and (not np.isfinite(c) or c <= 0.0):
        raise ValueError("c must be a positive finite cap or None")
    chi2 = raw_chi2 if c is None else np.minimum(raw_chi2, c)
    n_trunc = 0 if c is None else int(np.sum(raw_chi2 > c))
    design_rank, design_condition = _design_diagnostics(X_all, center=True)

    def fit(mask):
        y = chi2[mask]
        X = X_all[mask]
        yc, Xc = y - y.mean(), X - X.mean(axis=0)
        b = _nnls(Xc, yc)
        pi, B, distance = _bilinear_recover(b, pairs, K)
        eigenvalues = np.linalg.eigvalsh(B)
        spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1e-300)
        psd_violation = float(max(0.0, -eigenvalues[0]) / spectral_scale)
        return pi, b, float(B.sum()), B, distance, psd_violation

    pi, b_full, s_full, B, distance, psd_violation = fit(
        np.ones(m, dtype=bool)
    )
    grps = block_groups(n_blocks, groups)

    def stat(kept_blocks):
        mask = np.isin(variant_block, np.flatnonzero(kept_blocks))
        pi_m, _, _, _, distance_m, psd_m = fit(mask)
        rank_m, condition_m = _design_diagnostics(X_all[mask], center=True)
        remaining_groups = []
        for group in grps:
            group = np.asarray(group, dtype=int)
            kept_group = group[kept_blocks[group]]
            if kept_group.size:
                remaining_groups.append(kept_group)
        signal_m, _, correlations_m = _block_cross_validated_signal(
            X_all[mask], chi2[mask], variant_block[mask], remaining_groups
        )
        accepted = (
            pi_m is not None and rank_m == P and
            condition_m <= max_design_condition and
            correlations_m is not None and not np.isnan(signal_m) and
            signal_m >= min_signal_z and np.isfinite(distance_m) and
            distance_m <= max_rank1_distance and
            psd_m <= max_psd_violation
        )
        return pi_m if accepted else np.full(K, np.nan)

    signal_z, heldout_signal, heldout_correlations = (
        _block_cross_validated_signal(X_all, chi2, variant_block, grps))
    design_ok = (design_rank == P and
                 design_condition <= max_design_condition)
    signal_ok = not np.isnan(signal_z) and signal_z >= min_signal_z
    rank1_ok = np.isfinite(distance) and distance <= max_rank1_distance
    psd_ok = psd_violation <= max_psd_violation
    exact_model = n_trunc == 0
    identified = (pi is not None and design_ok and signal_ok and rank1_ok
                  and psd_ok and exact_model)
    if design_rank < P:
        note = ("bilinear reference design is not identifiable: "
                f"rank {design_rank}/{P}")
    elif design_condition > max_design_condition:
        note = ("bilinear reference design is too ill-conditioned for "
                "stable estimation: condition number "
                f"{design_condition:.3g} (limit {max_design_condition:.3g})")
    elif not exact_model:
        note = ("chi-square capping changed the response expectation; the "
                "capped fit is approximate, so composition is withheld")
    elif heldout_correlations is None:
        note = ("fewer than four usable block groups; reproducible "
                "polygenic signal cannot be assessed")
    elif not signal_ok:
        note = ("insufficient reproducible LD-dependent chi-square signal "
                "across held-out block groups; composition is withheld")
    elif not psd_ok:
        note = ("the fitted coefficient matrix fails the configured "
                "positive-semidefinite compatibility screen")
    elif not rank1_ok:
        note = ("the fitted coefficient matrix fails the configured "
                "rank-one compatibility screen")
    else:
        note = None
    if identified and len(grps) >= 2:
        se, per_group = _delete_one_jackknife(stat, n_blocks, grps)
    else:
        se, per_group = None, None
    boundary = bool(pi is not None and (pi < 1e-6).any())
    boundary_note = ("fitted composition lies on the simplex boundary; "
                     "ordinary symmetric inference from jackknife SEs is "
                     "invalid there, so use a boundary-aware interval"
                     if boundary else None)
    return {
        "proportions": pi if identified else None,
        "proportions_se": se if identified else None,
        "proportions_raw": pi,
        "coefficient_matrix": B,
        "signal": s_full,
        "signal_z": float(signal_z),
        "heldout_signal_correlation": heldout_signal,
        "heldout_signal_correlations": heldout_correlations,
        "intercept": float(chi2.mean() - X_all.mean(axis=0) @ b_full),
        "rank1_distance": distance,
        "psd_violation": psd_violation,
        "design_rank": design_rank,
        "design_columns": P,
        "design_condition": design_condition,
        "sample_size_scale": sample_size_scale,
        "n_used": int(m),
        "n_truncated": n_trunc,
        "truncation_cap": None if c is None else float(c),
        "approximation_note": (None if exact_model else
                               "explicit chi-square capping makes this an "
                               "approximate diagnostic only"),
        "jackknife_estimates": per_group,
        "signal_note": note,
        "boundary_note": boundary_note,
        "jackknife_note": (
            None if se is not None else
            "model not accepted, fewer than two block groups, or at least "
            "one leave-group refit failed the same acceptance screens; no "
            "jackknife SEs"
        ),
    }


def estimate_bilinear(z, refs, *, blocks=None, groups=None, c=None,
                      sample_size=None, max_design_condition=1e3,
                      min_signal_z=4.0, max_rank1_distance=0.25,
                      max_psd_violation=0.1):
    """Estimator B from z-scores and per-ancestry LD references.

    This compatibility wrapper builds the compact bilinear LD-score design,
    then delegates to :func:`estimate_bilinear_from_design`.
    """
    _, canonical_blocks = _as_ref_blocks(refs, blocks)
    Lb, _ = bilinear_ld_scores(refs, canonical_blocks)
    variant_block = np.empty(Lb.shape[0], dtype=int)
    for b, block in enumerate(canonical_blocks):
        variant_block[block] = b
    return estimate_bilinear_from_design(
        z, Lb, variant_block,
        n_blocks=len(canonical_blocks),
        groups=groups,
        c=c,
        sample_size=sample_size,
        max_design_condition=max_design_condition,
        min_signal_z=min_signal_z,
        max_rank1_distance=max_rank1_distance,
        max_psd_violation=max_psd_violation,
    )
