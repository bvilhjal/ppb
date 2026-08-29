"""Ancestry-composition estimation from summary statistics and LD references.

Implements the two LD-moment estimators derived in
``docs/ancestry_report/ancestry_report.tex`` (equation numbers below refer to
that report). Both need only the marginal z-scores and one LD reference per
candidate ancestry -- no allele-frequency column, no individual-level data.

* **Estimator A, pair products** -- eqs. (11)-(12). Within-block products
  ``z_i z_j`` are regressed on the per-ancestry reference correlations
  ``R^(k)_ij``. The moment is linear in the composition pi and is driven by
  the correlated-noise covariance ``(1 - h^2) R^(A)``, so it is identified
  even for traits with no polygenic signal; honest standard errors are what
  it requires (report Section 10, Assumption 3). With ``absorb_signal``
  (default) the ``K(K+1)/2`` quadratic columns ``(R^(k) R^(k'))_ij`` join the
  fit as non-negative absorbers -- they span the tagging term ``(n h^2 / m)
  (R^(A))^2`` exactly -- but are kept only when they detect real signal
  (their fitted total >= 2x its parametric SE), so null traits fall back to
  the stable linear fit.
* **Estimator B, bilinear LD-score regression** -- eqs. (13)-(15). Truncated
  chi-square statistics are regressed on the ``K(K+1)/2`` bilinear scores
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
(the chromosome analogue; eq. (16)), because the residuals are correlated
through LD and i.i.d. formulas do not apply.

Known limitations (report Section 10): the mixture approximation ignores
admixture LD beyond the within-block frequency-contrast term; Estimator B
needs genuine polygenic signal; both assume the references are harmonised
to the same variants and that standard errors are honest.
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
        return [[np.asarray(R, dtype=np.float64)[np.ix_(b, b)]
                 for b in blocks] for R in refs], list(blocks)
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
    return [[np.asarray(R[b], dtype=np.float64) for b in range(n_blocks)]
            for R in refs], list(blocks)


def bilinear_ld_scores(refs, blocks=None):
    """Bilinear scores ``l^(kk')_i = sum_j R^(k)_ij R^(k')_ij``, eq. (10).

    The sum runs over the variant's own block (references are
    block-diagonal). Returns ``(L, pairs)`` with ``L`` of shape
    ``(m, K*(K+1)/2)`` and ``pairs`` the ``(k, k')`` index list, ``k <= k'``.
    """
    ref_blocks, blocks = _as_ref_blocks(refs, blocks)
    K = len(ref_blocks)
    pairs = [(k, kp) for k in range(K) for kp in range(k, K)]
    parts = []
    for b in range(len(blocks)):
        parts.append(np.stack(
            [np.einsum("ij,ij->i", ref_blocks[k][b], ref_blocks[kp][b])
             for k, kp in pairs], axis=1))
    return np.concatenate(parts), pairs


def ld_scores(ref, blocks=None):
    """Per-variant LD scores ``l_i = sum_j R_ij^2`` of one reference."""
    L, _ = bilinear_ld_scores([ref, ref], blocks)
    return L[:, 1]


def pair_design(refs, blocks=None, *, floor=0.05, cap=250, quadratic=False):
    """Pair set P and design matrix for Estimator A, eq. (11).

    Pairs are taken within LD blocks only (unlinked pairs carry no ancestry
    information, report Section 6), keeping those with
    ``max_k |R^(k)_ij| >= floor`` and at most ``cap`` pairs per block, the
    strongest first. With ``quadratic=True`` the ``K(K+1)/2`` signal-term
    columns ``(R^(k) R^(k'))_ij`` are also returned -- the exact span of the
    tagging term ``(R^(A))^2`` under the mixture model.

    Returns ``(ii, jj, L, pair_block[, L2])``: global pair row indices, the
    ``(n_pairs, K)`` design with ``L[p, k] = R^(k)_{ii,jj}``, the block index
    of each pair, and optionally the ``(n_pairs, K(K+1)/2)`` quadratic design.
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
            L2_parts.append(np.stack(
                [(ref_blocks[k][b] @ ref_blocks[kp][b])[ti[keep], tj[keep]]
                 for k, kp in pairs], axis=1))
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
        return [np.asarray(g, dtype=int) for g in groups]
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


def estimate_pair_products(z, refs, *, blocks=None, floor=0.05, cap=250,
                           groups=None, absorb_signal=True):
    """Estimator A: composition from within-block z-score pair products.

    ``z`` are the marginal z-scores (genome-wide; every chromosome's blocks
    should be represented), ``refs`` the per-ancestry reference correlations
    (dense ``(m, m)`` each, or per-block matrix lists). With
    ``absorb_signal=True`` (default) the ``K(K+1)/2`` quadratic columns
    ``(R^(k) R^(k'))_ij`` join the fit as non-negative absorbers: they span
    the tagging term ``(n h^2 / m) (R^(A))^2`` exactly under the mixture
    model, so the composition read from the linear part is protected from the
    tagging bias the raw eq. (12) fit shows at moderate signal. Because the
    absorbers add noise when there is nothing to absorb, they are kept only
    when their fitted total clears twice its parametric standard error
    (otherwise the fit falls back to the linear columns alone). The
    composition read from the *quadratic* part (row sums, as in Estimator B)
    is returned as ``proportions_signal`` -- agreement between the two is a
    specification diagnostic.

    Returns a dict with ``proportions``, ``proportions_se`` (delete-one-group
    jackknife, eq. (16)), the noise ``scale`` (``~ 1 - h^2``), the signal
    scale (``~ n h^2 / m``; 0 for a null trait), pair counts, the maximum
    design correlation between linear ancestry columns (confusability
    diagnostic), and the residual RMS.
    """
    z = np.asarray(z, dtype=np.float64)
    K = len(refs)
    ii, jj, L, pair_block, L2 = pair_design(refs, blocks, floor=floor, cap=cap,
                                            quadratic=True)
    y_all = z[ii] * z[jj]
    pairs = [(k, kp) for k in range(K) for kp in range(k, K)]
    P = len(pairs)

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
        # Gate: keep the absorbers only when they detect real signal,
        # otherwise they just add noise to the noise-channel fit.
        resid = y - X @ u
        dof = max(int(mask.sum()) - K - P - 1, 1)
        sigma2 = float(resid @ resid) / dof
        XtX_inv = np.linalg.pinv(X.T @ X)
        se_v = float(np.sqrt(max(sigma2 * XtX_inv[K:, K:].sum(), 0.0)))
        if s_v < 2.0 * se_v:
            pi_u, s_u = _fit_pair_nnls(L[mask], y)
            u = np.zeros(K + P)
            u[:K] = 0.0 if pi_u is None else pi_u * s_u
            return pi_u, s_u, None, 0.0, u
        return split(u)

    pi, s, pi_signal, signal, u_full = fit(np.ones(len(blocks), dtype=bool))
    if pi is None and pi_signal is not None:
        pi = pi_signal                  # signal-dominated regime fallback
    if pi is None:
        raise ValueError("degenerate fit: no pair-product signal at all")
    grps = block_groups(len(blocks), groups)

    def stat(kept_blocks):
        out, _, out_v, _, _ = fit(kept_blocks)
        out = out if out is not None else out_v
        return np.full(K, np.nan) if out is None else out

    if len(grps) < 2:
        se, per_group = None, None
    else:
        se, per_group = _delete_one_jackknife(stat, len(blocks), grps)
    fitted = np.hstack([L, L2]) @ u_full if absorb_signal else s * (L @ pi)
    center = L - L.mean(axis=0)
    denom = np.sqrt((center ** 2).sum(axis=0))
    denom = np.where(denom == 0.0, np.nan, denom)
    design_corr = (center.T @ center) / np.outer(denom, denom)
    offdiag = design_corr[~np.eye(design_corr.shape[0], dtype=bool)]
    agreement = (None if pi_signal is None else
                 float(np.abs(pi - pi_signal).max()))
    boundary = bool((pi < 1e-6).any())
    boundary_note = ("estimate lies on the simplex boundary; jackknife SEs "
                     "are not meaningful there -- the design cannot separate "
                     "all ancestries at this sample size"
                     if boundary else None)
    return {
        "proportions": pi,
        "proportions_se": se,
        "proportions_signal": pi_signal,
        "channel_agreement": agreement,
        "scale": s,
        "signal": signal,
        "boundary_note": boundary_note,
        "n_pairs": int(y_all.size),
        "n_blocks": int(len(blocks)),
        "max_design_correlation": float(np.nanmax(np.abs(offdiag))),
        "residual_rms": float(np.sqrt(np.mean((y_all - fitted) ** 2))),
        "jackknife_estimates": per_group,
        "jackknife_note": (None if se is not None else
                           "fewer than two block groups; no jackknife SEs"),
    }


def _bilinear_recover(b, pairs, K):
    """Composition from bilinear-regression coefficients, eqs. (14)-(15).

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


def estimate_bilinear(z, refs, *, blocks=None, groups=None, c=None):
    """Estimator B: composition from chi-squares on bilinear LD scores.

    Regresses ``min(z^2, c)`` on the ``K(K+1)/2`` bilinear scores with a free
    intercept and non-negativity constraints (eq. (13)), then recovers pi
    from the coefficient row sums (eq. (15)). The truncation defaults to
    ``c = 30 * max(1, median(z^2))`` -- the LDSC ``c ~ 30`` rule relative to
    the null scale, adapted to the dataset's signal level. Needs genuine
    polygenic signal: the fitted total signal ``s = sum(B)`` estimates
    ``n h^2 / m`` and must clear its own (parametric) noise floor -- otherwise
    ``proportions`` is ``None`` and ``signal_note`` says so, rather than a
    fabricated composition. The ``rank1_distance`` is the relative Frobenius
    distance of the coefficient matrix from its fitted rank-one form -- a
    specification diagnostic, since ``B = (n h^2 / m) pi pi^T`` at truth.
    """
    z = np.asarray(z, dtype=np.float64)
    K = len(refs)
    _, blocks = _as_ref_blocks(refs, blocks)
    Lb, pairs = bilinear_ld_scores(refs, blocks)
    if c is None:
        c = 30.0 * max(1.0, float(np.median(z * z)))
    chi2 = np.minimum(z * z, c)
    m = z.size
    variant_block = np.concatenate([
        np.full(len(blocks[b]), b, dtype=int) for b in range(len(blocks))])

    def fit(mask):
        y = chi2[mask]
        X = Lb[mask]
        yc, Xc = y - y.mean(), X - X.mean(axis=0)
        b = _nnls(Xc, yc)
        resid = yc - Xc @ b
        dof = max(int(mask.sum()) - X.shape[1] - 1, 1)
        sigma2 = float(resid @ resid) / dof
        XtX_inv = np.linalg.pinv(Xc.T @ Xc)
        se_signal = float(np.sqrt(max(
            sigma2 * XtX_inv.sum(), 0.0)))       # s = sum(b) = 1' b
        pi, B, _ = _bilinear_recover(b, pairs, K)
        return pi, b, float(B.sum()), se_signal

    pi, b_full, s_full, se_signal_full = fit(np.ones(m, dtype=bool))
    n_trunc = int(np.sum(z * z >= c))
    grps = block_groups(len(blocks), groups)

    def stat(kept_blocks):
        mask = np.isin(variant_block, np.flatnonzero(kept_blocks))
        pi_m, _, _, _ = fit(mask)
        return np.full(K, np.nan) if pi_m is None else pi_m

    if len(grps) < 2 or pi is None:
        se, per_group = None, None
    else:
        se, per_group = _delete_one_jackknife(stat, len(blocks), grps)
    signal_z = s_full / se_signal_full if se_signal_full > 0 else (
        np.inf if s_full > 1e-12 else 0.0)
    identified = pi is not None and signal_z >= 2.0
    _, B, distance = _bilinear_recover(b_full, pairs, K)
    if identified:
        note = None
    elif n_trunc > 0.05 * m:
        note = ("chi-square saturates the truncation cap for many variants "
                "(huge per-variant signal); raise c")
    else:
        note = ("no polygenic signal: chi-square is flat in the LD "
                "scores, so the composition is not identified")
    return {
        "proportions": pi if identified else None,
        "proportions_se": se if identified else None,
        "proportions_raw": pi,
        "coefficient_matrix": B,
        "signal": s_full,
        "signal_z": float(signal_z),
        "intercept": float(chi2.mean() - Lb.mean(axis=0) @ b_full),
        "rank1_distance": distance,
        "n_used": int(m),
        "n_truncated": n_trunc,
        "truncation_cap": float(c),
        "jackknife_estimates": per_group,
        "signal_note": note,
        "jackknife_note": (None if se is not None else
                           "fewer than two block groups or no signal; "
                           "no jackknife SEs"),
    }
