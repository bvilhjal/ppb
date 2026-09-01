"""Simulation harness for the LD-reference experiment (Supplementary Fig. S1).

Reproduces the qualitative finding of Witteveen et al.: the summary-statistic
prediction ``R^2`` estimated with

    R^2 = (w^T z_target)^2 / (w^T D_ref w)

is (a) exact when ``D_ref`` is the test-set LD, (b) biased when ``D_ref`` is the
training-set LD, and (c) unbiased when ``D_ref`` is an independent LD reference
from the same population. ``z_target = (1/n) X_test^T y_test`` are the released
target summary statistics.

The genotype model here is intentionally simple (standardized Gaussian variants
with a block-AR(1) population LD) -- enough to exhibit the LD-reference behavior
without individual-level data or a real cohort.
"""

from __future__ import annotations

import numpy as np


def population_ld(m: int, block_size: int = 30, rho: float = 0.6) -> np.ndarray:
    """Block-diagonal AR(1) population correlation ``Sigma`` (unit diagonal)."""
    if m < 1 or block_size < 1 or not 0.0 <= rho < 1.0:
        raise ValueError("require m>=1, block_size>=1, rho in [0, 1)")
    Sigma = np.zeros((m, m))
    start = 0
    while start < m:
        k = min(block_size, m - start)
        d = np.arange(k)
        Sigma[start:start + k, start:start + k] = rho ** np.abs(d[:, None] - d[None, :])
        start += k
    return Sigma


def sample_genotypes(Sigma: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n`` standardized variants from ``N(0, Sigma)`` (columns mean 0, var 1)."""
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n, Sigma.shape[0])) @ L.T
    X = X - X.mean(axis=0)
    return X / X.std(axis=0)


# Acklam's rational approximation of the standard-normal quantile (inverse CDF),
# accurate to ~1e-9 -- so the simulator needs no scipy.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)


def _norm_ppf(p):
    """Standard-normal quantile Phi^{-1}(p), vectorized, no scipy (Acklam)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    x = np.zeros_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    x[lo] = ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
             / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1))
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    x[hi] = -((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
              / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1))
    q = p[mid] - 0.5
    r = q * q
    x[mid] = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q \
        / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)
    return x


def simulate_haplotypes(n, block_sizes, maf, rho, rng):
    """Founder haplotypes ``(n, 2, m)`` of 0/1, with block AR(1) LD.

    The phased form of :func:`simulate_diploid_genotypes`'s model, needed
    whenever offspring must be produced by meiosis rather than drawn afresh --
    see :func:`assortatively_mated_offspring`.
    """
    block_sizes = list(block_sizes)
    m = int(sum(block_sizes))
    maf = np.asarray(maf, dtype=np.float64)
    if maf.shape != (m,):
        raise ValueError(f"maf must have length m={m}; got {maf.shape}")
    thr = _norm_ppf(1.0 - maf)                        # P(z > thr) = maf per variant
    H = np.zeros((n, 2, m), dtype=np.uint8)
    col = 0
    for k in block_sizes:
        d = np.arange(k)
        L = np.linalg.cholesky(rho ** np.abs(d[:, None] - d[None, :]) + 1e-8 * np.eye(k))
        block_thr = thr[col:col + k]
        for h in range(2):
            H[:, h, col:col + k] = rng.standard_normal((n, k)) @ L.T > block_thr
        col += k
    return H


def _diploid_dosages(n, block_sizes, maf, rho, rng):
    """Raw 0/1/2 dosages (n x m) with block AR(1) LD and per-variant ``maf``."""
    return simulate_haplotypes(n, block_sizes, maf, rho, rng).sum(axis=1, dtype=np.float64)


def _standardize_cols(G):
    sd = G.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (G - G.mean(axis=0)) / sd


def simulate_diploid_genotypes(n, block_sizes, maf, rho, rng):
    """Simulate standardized diploid genotypes (0/1/2 dosages) with block LD.

    ldpred3-inspired latent model: per block, two Gaussian haplotypes with AR(1)
    correlation ``rho`` are thresholded at the MAF-implied quantile and summed,
    giving realistic 0/1/2 dosages with allele frequency ``maf`` and within-block
    LD. Returns the standardized genotype matrix (columns mean 0, variance 1).
    """
    return _standardize_cols(_diploid_dosages(n, block_sizes, maf, rho, rng))


def bn_freqs(rng, m, fst):
    """Balding-Nichols allele frequencies for two populations ``fst`` apart.

    Frequencies are clipped away from 0/1 so every variant stays polymorphic.
    """
    if not 0.0 < fst < 1.0:
        raise ValueError(f"fst must be in (0, 1); got {fst}")
    anc = rng.uniform(0.1, 0.9, size=m)
    a = anc * (1 - fst) / fst
    b = (1 - anc) * (1 - fst) / fst
    return (np.clip(rng.beta(a, b), 1e-3, 1 - 1e-3),
            np.clip(rng.beta(a, b), 1e-3, 1 - 1e-3))


def bn_freqs_multi(rng, m, fst, k):
    """Balding-Nichols frequencies for ``k`` populations ``fst`` apart.

    Returns a ``(k, m)`` array. All ``k`` rows are i.i.d. Balding--Nichols
    draws from the same ancestral frequencies; no row is special. Each
    population marginalises to the same ancestral mean, so
    ancestry-informative variants arise from drift, not ascertainment.
    """
    if not 0.0 < fst < 1.0:
        raise ValueError(f"fst must be in (0, 1); got {fst}")
    if k < 2:
        raise ValueError("need at least two populations")
    anc = rng.uniform(0.1, 0.9, size=m)
    a = anc * (1 - fst) / fst
    b = (1 - anc) * (1 - fst) / fst
    return np.clip(rng.beta(a, b, size=(k, m)), 1e-3, 1 - 1e-3)


def population_ld_blocks(block_sizes, rho_blocks):
    """Block-diagonal AR(1) correlation with a per-block correlation strength."""
    block_sizes = list(block_sizes)
    rho_blocks = np.asarray(rho_blocks, dtype=np.float64)
    if rho_blocks.shape != (len(block_sizes),):
        raise ValueError("rho_blocks must have one value per block")
    if ((rho_blocks < 0) | (rho_blocks >= 1)).any():
        raise ValueError("rho_blocks must be in [0, 1)")
    m = int(sum(block_sizes))
    Sigma = np.zeros((m, m))
    col = 0
    for k, rho in zip(block_sizes, rho_blocks):
        d = np.arange(k)
        Sigma[col:col + k, col:col + k] = rho ** np.abs(d[:, None] - d[None, :])
        col += k
    return Sigma


def draw_ld_paths(rng, block_sizes, base_lo, base_hi, hotspot_frac=0.15,
                  hotspot_hi=0.3):
    """One ancestry's LD landscape as per-block, per-interval AR(1) paths.

    Each block draws a base correlation ``~ U(base_lo, base_hi)``; most
    intervals keep it, and a ``hotspot_frac`` of intervals drop to
    ``U(0, hotspot_hi)`` (recombination hotspots). Per-interval variation is
    what makes per-variant LD scores differ *within* a block -- without it
    every interior variant of a block has the same score, the bilinear scores
    of estimator B become functionally dependent columns, and the ancestry
    composition is not identifiable from the chi-square moment.
    """
    if not 0 <= base_lo <= base_hi < 1.0:
        raise ValueError("require 0 <= base_lo <= base_hi < 1")
    paths = []
    for k in block_sizes:
        base = rng.uniform(base_lo, base_hi)
        path = np.full(max(k - 1, 0), base)
        if path.size:
            hot = rng.random(path.size) < hotspot_frac
            path[hot] = rng.uniform(0.0, hotspot_hi, size=int(hot.sum()))
        paths.append(path)
    return paths


def constant_ld_paths(block_sizes, rho):
    """Per-block constant AR(1) paths (no hotspots) at strength ``rho``."""
    return [np.full(max(int(k) - 1, 0), rho) for k in block_sizes]


def _check_ld_paths(paths, block_sizes):
    block_sizes = list(block_sizes)
    if len(paths) != len(block_sizes):
        raise ValueError("one LD path per block is required")
    out = []
    for path, k in zip(paths, block_sizes):
        path = np.asarray(path, dtype=np.float64)
        if path.shape != (max(int(k) - 1, 0),) or ((path < 0) | (path >= 1)).any():
            raise ValueError("each LD path must be k-1 values in [0, 1)")
        out.append(path)
    return out


def block_correlations(paths):
    """Block-diagonal correlation matrix from per-block AR(1) interval paths.

    Within a block, ``R_ij = prod_{t=i}^{j-1} path_t`` -- the correlation of a
    (non-stationary) Gaussian AR(1) process, so the matrix is PSD for any path
    in ``[0, 1)``; hotspots in the path show up as sharp LD breaks.
    """
    blocks = []
    for path in paths:
        path = np.asarray(path, dtype=np.float64)
        if ((path < 0) | (path >= 1)).any():
            raise ValueError("LD paths must lie in [0, 1)")
        log_r = np.concatenate([[0.0], np.cumsum(np.log(np.clip(path, 1e-12, None)))])
        R = np.exp(log_r[None, :] - log_r[:, None])
        R = np.triu(R) + np.triu(R, 1).T
        blocks.append(R)
    m = sum(R.shape[0] for R in blocks)
    Sigma = np.zeros((m, m))
    col = 0
    for R in blocks:
        k = R.shape[0]
        Sigma[col:col + k, col:col + k] = R
        col += k
    return Sigma


def _hap_segment_paths(n, block_sizes, maf, paths, rng):
    """``(n, sum(block_sizes))`` founder haplotypes under per-interval AR(1) LD."""
    maf = np.asarray(maf, dtype=np.float64)
    thr = _norm_ppf(1.0 - maf)
    H = np.zeros((n, maf.size), dtype=np.uint8)
    col = 0
    for k, path in zip(block_sizes, paths):
        latent = rng.standard_normal((n, k))
        for t in range(1, k):
            rho = path[t - 1]
            latent[:, t] = rho * latent[:, t - 1] + np.sqrt(1.0 - rho * rho) * latent[:, t]
        H[:, col:col + k] = latent > thr[col:col + k]
        col += k
    return H


def simulate_admixture_references(n_ref, block_sizes, maf_pops, ld_paths, rng):
    """Per-ancestry LD references ``R^(k)`` from simulated reference panels.

    Each ancestry's panel is ``n_ref`` diploid individuals at that ancestry's
    allele frequencies and LD landscape (``ld_paths[k]``: one AR(1) interval
    path per block, see :func:`draw_ld_paths`); ``R^(k)`` is the panel
    correlation matrix. Reference sampling noise (~1/sqrt(n_ref) per entry)
    is included deliberately: the estimators must tolerate it.
    """
    maf_pops = np.asarray(maf_pops, dtype=np.float64)
    block_sizes = list(block_sizes)
    if len(ld_paths) != maf_pops.shape[0]:
        raise ValueError("need one LD landscape per ancestry")
    refs = []
    for k in range(maf_pops.shape[0]):
        paths = _check_ld_paths(ld_paths[k], block_sizes)
        hap = _hap_segment_paths(2 * n_ref, block_sizes, maf_pops[k], paths, rng)
        G = _standardize_cols(
            hap.reshape(n_ref, 2, -1).sum(axis=1, dtype=np.float64))
        R = np.corrcoef(G, rowvar=False)
        # Finite n_ref can still produce a monomorphic column; corrcoef then
        # writes NaN. Treat those variants as uninformative (identity).
        if not np.all(np.isfinite(R)):
            R = np.where(np.isfinite(R), R, 0.0)
            np.fill_diagonal(R, 1.0)
        refs.append(R)
    return refs


def simulate_admixed_genotypes(n, block_sizes, maf_pops, ld_paths, pi, rng,
                               segment_blocks=1):
    """Admixed genomes as ancestry mosaics over LD blocks.

    For each of the ``n`` individuals, both haplotypes draw their ancestry
    independently per *segment* of ``segment_blocks`` consecutive blocks, with
    probabilities ``pi``; the segment's haplotype is then drawn from that
    ancestry's LD landscape and allele frequencies. ``ld_paths[k]`` is
    ancestry ``k``'s landscape (one AR(1) interval path per block, see
    :func:`draw_ld_paths`). With the default ``segment_blocks=1`` there is no
    between-block admixture LD (old admixture), but the within-block
    Wahlund/frequency-contrast term of the report's Assumption 1 is present
    whenever the ancestries' frequencies differ, so the mixture approximation
    is stressed, not assumed.

    Returns the standardized ``(n, m)`` genotype matrix of the admixed
    population.
    """
    block_sizes = list(block_sizes)
    maf_pops = np.asarray(maf_pops, dtype=np.float64)
    K, m = maf_pops.shape
    pi = np.asarray(pi, dtype=np.float64)
    if pi.shape != (K,) or (pi < 0).any() or abs(pi.sum() - 1.0) > 1e-8:
        raise ValueError("pi must be a probability vector of length K")
    if len(ld_paths) != K:
        raise ValueError("need one LD landscape per ancestry")
    if segment_blocks < 1:
        raise ValueError("segment_blocks must be >= 1")
    paths_pops = [_check_ld_paths(ld_paths[k], block_sizes) for k in range(K)]
    starts = np.concatenate([[0], np.cumsum(block_sizes)])
    segments = [range(s, min(s + segment_blocks, len(block_sizes)))
                for s in range(0, len(block_sizes), segment_blocks)]
    G = np.zeros((n, m))
    individuals = np.repeat(np.arange(n), 2)
    for seg in segments:
        seg = list(seg)
        sizes = [block_sizes[b] for b in seg]
        cols = np.concatenate([np.arange(starts[b], starts[b] + block_sizes[b])
                               for b in seg])
        ancestry = rng.choice(K, size=(n, 2), p=pi)
        for k in range(K):
            mask = ancestry == k
            count = int(mask.sum())
            if count == 0:
                continue
            hap = _hap_segment_paths(count, sizes, maf_pops[k][cols],
                                     [paths_pops[k][b] for b in seg], rng)
            rows = np.repeat(individuals[mask.ravel()], cols.size)
            np.add.at(G, (rows, np.tile(cols, count)),
                      hap.ravel().astype(np.float64))
    return _standardize_cols(G)


def simulate_admixture_sumstats(n, block_sizes, fst, ld_paths, pi, h2, n_causal,
                                rng, n_ref=2000, segment_blocks=1,
                                return_r_admixed=False):
    """End-to-end admixed-GWAS simulation for the ancestry estimators.

    Draws Balding-Nichols frequencies for ``K = len(ld_paths)`` ancestries,
    estimates per-ancestry reference LD from ``n_ref``-individual panels,
    builds an admixed population as an ancestry mosaic with composition
    ``pi``, simulates a phenotype with heritability ``h2`` on the admixed
    genotypes, and runs the marginal GWAS. ``ld_paths[k]`` is ancestry ``k``'s
    LD landscape (one AR(1) interval path per block; use
    :func:`draw_ld_paths` for hotspot landscapes or
    :func:`constant_ld_paths` for plain AR(1) blocks). Returns a dict with
    the z-statistics ``z``, the reference matrices ``refs``, and the truth
    ``pi``/``maf_pops``; ``R_admixed`` (the realized admixed correlation
    matrix) only when ``return_r_admixed`` is set.
    """
    block_sizes = list(block_sizes)
    K = len(ld_paths)
    m = int(sum(block_sizes))
    maf_pops = bn_freqs_multi(rng, m, fst, K)
    refs = simulate_admixture_references(n_ref, block_sizes, maf_pops, ld_paths, rng)
    X = simulate_admixed_genotypes(n, block_sizes, maf_pops, ld_paths, pi, rng,
                                   segment_blocks=segment_blocks)
    beta = draw_effects(m, n_causal, rng)
    y = simulate_phenotype(X, beta, h2, rng)
    _, t = marginal_stats(X, y)
    starts = np.concatenate([[0], np.cumsum(block_sizes)])
    blocks = [np.arange(starts[b], starts[b] + block_sizes[b])
              for b in range(len(block_sizes))]
    out = {
        "z": t,
        "refs": [[R[np.ix_(b, b)] for b in blocks] for R in refs],
        "blocks": blocks,
        "pi": np.asarray(pi, dtype=np.float64),
        "maf_pops": maf_pops,
    }
    if return_r_admixed:
        out["R_admixed"] = np.corrcoef(X, rowvar=False)
    return out


def simulate_admixture_mvn(R_pops, pi, h2, n, rng, blocks=None):
    """Model-consistent z-scores: the report's pair-moment equation exactly.

    Draws ``z ~ N(0, Sigma)`` with, per LD block,
    ``Sigma_b = (1 - h^2) R^(A)_b + (n h^2 / m) R^(A)_b R^(A)_b`` and
    ``R^(A) = sum_k pi_k R^(k)`` -- the infinitesimal moment model with no
    admixture-LD or reference-estimation complications, so estimator error is
    pure Monte Carlo noise and jackknife SEs can be calibrated against it.
    References may be dense ``(m, m)`` matrices (then ``blocks`` must slice
    them) or per-ancestry lists of per-block matrices, the scalable form;
    computation is per block either way, so cost is linear in genome size.
    """
    pi = np.asarray(pi, dtype=np.float64)
    if abs(pi.sum() - 1.0) > 1e-8 or len(R_pops) != pi.size:
        raise ValueError("pi must be a probability vector matching the references")
    first = R_pops[0]
    if isinstance(first, np.ndarray) and first.ndim == 2:
        m = first.shape[0]
        if blocks is None:
            blocks = [np.arange(m)]
        ref_blocks = [[np.asarray(R)[np.ix_(b, b)] for b in blocks]
                      for R in R_pops]
    else:
        ref_blocks = [[np.asarray(Rb, dtype=np.float64) for Rb in R]
                      for R in R_pops]
        blocks = [np.arange(Rb.shape[0]) for Rb in ref_blocks[0]]
    m = int(sum(Rb.shape[0] for Rb in ref_blocks[0]))
    parts = []
    for b in range(len(blocks)):
        kb = ref_blocks[0][b].shape[0]
        R_A = sum(p * ref_blocks[k][b] for k, p in enumerate(pi))
        Sigma = (1.0 - h2) * R_A + (n * h2 / m) * (R_A @ R_A)
        Sigma = (Sigma + Sigma.T) / 2 + 1e-10 * np.eye(kb)
        parts.append(np.linalg.cholesky(Sigma) @ rng.standard_normal(kb))
    return np.concatenate(parts)


def simulate_structured_genotypes(n, block_sizes, fst, rho, rng, prop_pop1=0.5):
    """Two subpopulations (Balding-Nichols ``fst``) with block LD.

    Each individual is assigned to population 0 or 1 and its variants drawn at
    that population's allele frequencies. Because the frequencies differ, the
    leading principal component of the returned genotypes is the ancestry axis --
    a controllable source of population structure. Returns ``(X, labels)`` with
    ``X`` standardized over all individuals and ``labels`` the 0/1 assignment.
    """
    m = int(sum(block_sizes))
    f1, f2 = bn_freqs(rng, m, fst)
    labels = (rng.random(n) < prop_pop1).astype(int)
    G = np.zeros((n, m), dtype=np.float64)
    for pop, freq in ((0, f1), (1, f2)):
        idx = np.where(labels == pop)[0]
        if idx.size:
            G[idx] = _diploid_dosages(idx.size, block_sizes, freq, rho, rng)
    return _standardize_cols(G), labels


def draw_effects(m: int, n_causal: int, rng: np.random.Generator) -> np.ndarray:
    """Sparse standard-normal causal effect vector with ``n_causal`` nonzeros."""
    beta = np.zeros(m)
    idx = rng.choice(m, size=min(n_causal, m), replace=False)
    beta[idx] = rng.standard_normal(idx.size)
    return beta


def simulate_phenotype(X, beta, h2, rng) -> np.ndarray:
    """Standardized phenotype with genetic variance ``h2`` from predictor ``X @ beta``."""
    if not 0.0 <= h2 <= 1.0:
        raise ValueError("h2 must be in [0, 1]")
    g = X @ beta
    sd = g.std()
    g = g / sd if sd > 0 else g
    y = np.sqrt(h2) * g + np.sqrt(1.0 - h2) * rng.standard_normal(X.shape[0])
    y = y - y.mean()
    return y / y.std()


def marginal_stats(X, y):
    """Marginal GWAS: per-variant ``z = (1/n) X^T y`` and its t-statistic.

    For standardized ``X`` and ``y`` the marginal least-squares slope equals ``z``
    and its standard error is ``sqrt((1 - z^2) / n)``.
    """
    n = X.shape[0]
    z = (X.T @ y) / n
    se = np.sqrt(np.clip(1.0 - z ** 2, 1e-12, None) / n)
    return z, z / se


def pgs_pthreshold(z, t, t_crit: float = 2.5) -> np.ndarray:
    """p-value-thresholded marginal weights: keep ``z_j`` where ``|t_j| >= t_crit``."""
    return np.where(np.abs(t) >= t_crit, z, 0.0)


def _gametes(hap, parents, block_starts, rng, recombination):
    """One gamete per parent: walk each block, switching strand at ``recombination``.

    A fresh coin at every block start makes blocks assort independently, which is
    what "independent LD block" means: within-block LD survives meiosis, and any
    LD *between* blocks halves each generation unless something regenerates it.
    """
    n, m = parents.size, hap.shape[2]
    flip = rng.random((n, m)) < recombination
    flip[:, block_starts] = rng.random((n, block_starts.size)) < 0.5
    which = np.cumsum(flip, axis=1) % 2
    return hap[parents[:, None], which, np.arange(m)[None, :]]


def assortatively_mated_offspring(hap, phenotype, block_sizes, rng, *,
                                  spouse_correlation, recombination=0.01):
    """One generation of assortative mating: pair on phenotype, then meiose.

    Mates are paired to a target phenotypic correlation, so trait-increasing
    alleles at *unlinked* loci become positively correlated in the offspring --
    the linkage disequilibrium that inflates additive genetic variance above the
    linkage-equilibrium value ``sum_j beta_j^2 2f_j(1-f_j)``.

    Returns ``(offspring_hap, realized_spouse_correlation)``. The realized
    correlation is reported rather than assumed: the rank-pairing construction
    hits the target closely but not exactly, and the achieved value is the one
    any downstream theory must be checked against. Two offspring are produced
    per pair, so an even parent count ``n`` yields exactly ``n`` offspring;
    an odd ``n`` floors to ``n - 1`` (one parent is left unpaired).
    """
    if not 0.0 <= spouse_correlation < 1.0:
        raise ValueError(
            f"spouse_correlation must be in [0, 1); got {spouse_correlation}")
    y = np.asarray(phenotype, dtype=np.float64)
    if y.shape != (hap.shape[0],):
        raise ValueError(f"phenotype has shape {y.shape}, expected ({hap.shape[0]},)")

    order = rng.permutation(y.size)
    half = y.size // 2
    a, b = order[:half], order[half:2 * half]

    # Rank-pair group A on its phenotype and group B on a noised copy. The noise
    # variance sets the achieved correlation: corr = 1/sqrt(1 + var_e/var_y).
    if spouse_correlation == 0.0:
        a, b = a[rng.permutation(half)], b[rng.permutation(half)]
    else:
        var_e = y.var() * (1.0 / spouse_correlation ** 2 - 1.0)
        noisy = y[b] + rng.normal(scale=np.sqrt(var_e), size=half)
        a, b = a[np.argsort(y[a])], b[np.argsort(noisy)]
    realized = float(np.corrcoef(y[a], y[b])[0, 1])

    block_starts = np.concatenate([[0], np.cumsum(list(block_sizes))[:-1]])
    pairs = np.concatenate([a, a])          # two offspring per pair; odd n floors to n - 1
    mates = np.concatenate([b, b])
    offspring = np.empty((pairs.size, 2, hap.shape[2]), dtype=np.uint8)
    offspring[:, 0] = _gametes(hap, pairs, block_starts, rng, recombination)
    offspring[:, 1] = _gametes(hap, mates, block_starts, rng, recombination)
    return offspring, realized
