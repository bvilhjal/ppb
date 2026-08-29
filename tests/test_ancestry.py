"""Tests for the LD-moment ancestry estimators (ppb.ancestry) and their
simulation harness (ppb.simulate admixture helpers).

Structure: unit pins for the numerical pieces, then the two simulation arms
of docs/ancestry_report/ancestry_report.tex -- model-consistent MVN draws
(estimator correctness, jackknife calibration, tagging-bias absorption) and
individual-level admixed-GWAS mosaics (robustness to Wahlund/frequency
contrast and reference-estimation noise).
"""

import numpy as np
import pytest

from ppb.ancestry import (_nnls, bilinear_ld_scores, estimate_bilinear, estimate_pair_products, ld_scores,
                          pair_design)
from ppb.simulate import (block_correlations, bn_freqs_multi,
                          draw_ld_paths,
                          simulate_admixture_mvn, simulate_admixture_sumstats,
                          simulate_admixed_genotypes)

PI = np.array([0.65, 0.35])
RANGES_K2 = [(0.50, 0.90), (0.10, 0.50)]   # EUR-like / AFR-like LD landscapes


def _landscape(seed, n_blocks, ranges=RANGES_K2):
    """Reference blocks for ancestries with distinct hotspot LD landscapes."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(15, 36, n_blocks)
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    blocks = [np.arange(bounds[b], bounds[b + 1]) for b in range(n_blocks)]
    paths = [draw_ld_paths(rng, sizes, lo, hi) for lo, hi in ranges]
    mats = [block_correlations(p) for p in paths]
    ref_blocks = [[M[np.ix_(b, b)] for b in blocks] for M in mats]
    return sizes, blocks, paths, ref_blocks


@pytest.fixture(scope="module")
def small():
    return _landscape(42, 60)


@pytest.fixture(scope="module")
def large():
    return _landscape(7, 400)


# ---------------------------------------------------------------- unit pins

def test_nnls_exact_and_zero():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 3))
    x = np.array([1.5, 0.0, 0.5])
    b = A @ x
    est = _nnls(A, b)
    assert np.allclose(est, x, atol=1e-6)
    est = _nnls(A, -(A @ np.ones(3)))     # gradient at 0 non-positive -> x = 0
    assert np.all(est <= 1e-10)


def test_block_correlations_psd_and_unit():
    paths = [np.array([0.5, 0.1, 0.5]), np.array([0.7, 0.7])]
    R = block_correlations(paths)
    assert R.shape == (7, 7)            # path length k - 1 -> block of size k
    assert np.allclose(np.diag(R), 1.0)
    # product structure: R_03 = 0.5 * 0.1 * 0.5 in the first block
    assert R[0, 3] == pytest.approx(0.5 * 0.1 * 0.5)
    assert R[0, 4] == 0.0               # block-diagonal
    assert np.linalg.eigvalsh(R).min() > -1e-10


def test_ld_paths_validation():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="base_lo"):
        draw_ld_paths(rng, [10], 0.9, 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        block_correlations([np.array([0.5, 1.5])])
    p = draw_ld_paths(rng, [8, 8], 0.5, 0.9, hotspot_frac=0.0)
    assert all(np.all(path >= 0.5) for path in p)
    from ppb.simulate import simulate_admixture_references
    with pytest.raises(ValueError, match="one LD landscape per ancestry"):
        simulate_admixture_references(50, [8, 8], np.full((2, 16), 0.3),
                                      [p], rng)
    with pytest.raises(ValueError, match="one LD path per block"):
        simulate_admixture_references(50, [8, 8], np.full((2, 16), 0.3),
                                      [p, p[:1]], rng)


def test_pair_design_floor_cap_and_blocks(small):
    sizes, blocks, paths, RB = small
    ii, jj, L, pb = pair_design(RB, blocks, floor=0.2, cap=50)
    assert L.shape == (ii.size, 2)
    assert (np.abs(L).max(axis=1) >= 0.2 - 1e-12).all()
    assert np.bincount(pb).max() <= 50
    for b, block in enumerate(blocks):   # every pair stays within its block
        sel = pb == b
        assert np.isin(ii[sel], block).all() and np.isin(jj[sel], block).all()
    with pytest.raises(ValueError, match="no pairs"):
        pair_design(RB, blocks, floor=1.1)


def test_bilinear_scores_match_definition(small):
    sizes, blocks, paths, RB = small
    L, pairs = bilinear_ld_scores(RB, blocks)
    assert pairs == [(0, 0), (0, 1), (1, 1)]
    assert np.allclose(L[:, 0], ld_scores(RB[0], blocks))
    for b, block in enumerate(blocks):
        r0, r1 = RB[0][b], RB[1][b]
        assert np.allclose(L[block, 2], np.einsum("ij,ij->i", r1, r1))
        assert np.allclose(L[block, 1], np.einsum("ij,ij->i", r0, r1))


def test_reference_validation():
    sizes, blocks, paths, RB = _landscape(0, 6)
    with pytest.raises(ValueError, match="at least two"):
        estimate_pair_products(np.zeros(sum(sizes)), RB[:1], blocks=blocks)
    with pytest.raises(ValueError, match="same number of blocks"):
        estimate_bilinear(np.zeros(sum(sizes)), [RB[0], RB[1][:3]],
                          blocks=blocks)


# ------------------------------------------------- MVN arm (model-consistent)

def _rep_means(estimator, RBB, blocksB, pi, h2, n, reps, seed, **kw):
    ests, ses = [], []
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        z = simulate_admixture_mvn(RBB, pi, h2, n, rng)
        r = estimator(z, RBB, blocks=blocksB, **kw)
        if r["proportions"] is None:
            continue
        ests.append(r["proportions"])
        ses.append(r["proportions_se"])
    return np.asarray(ests), np.asarray(ses)


def test_pair_products_null_trait_recovery_and_calibration(large):
    """h2=0: A is unbiased and its jackknife SE tracks the empirical SD."""
    _, blocksB, _, RBB = large
    ests, ses = _rep_means(estimate_pair_products, RBB, blocksB, PI, 0.0,
                           1000, reps=8, seed=300)
    assert ests.shape == (8, 2)
    assert np.abs(ests.mean(axis=0) - PI).max() < 0.05
    ratio = ses.mean(axis=0) / ests.std(axis=0)
    assert np.all((ratio > 0.5) & (ratio < 2.0))


def test_pair_products_absorber_removes_tagging_bias(large):
    """The quadratic absorber removes most of the toward-high-LD tagging
    bias that saturates the raw fit at nh^2/m = 0.5."""
    _, blocksB, _, RBB = large
    m = sum(b.size for b in blocksB)
    n = 0.5 * m / 0.25                      # nh^2/m = 0.5
    raw, _ = _rep_means(estimate_pair_products, RBB, blocksB, PI, 0.25, n,
                        reps=12, seed=400, absorb_signal=False)
    cor, _ = _rep_means(estimate_pair_products, RBB, blocksB, PI, 0.25, n,
                        reps=12, seed=400, absorb_signal=True)
    raw_err = np.abs(raw.mean(axis=0) - PI).max()
    cor_err = np.abs(cor.mean(axis=0) - PI).max()
    assert raw_err > 0.25                   # raw fit saturates on the high-LD ancestry
    assert cor_err < 0.20
    assert cor_err < raw_err


def test_pair_products_weak_signal_regime(large):
    """At the report's realistic nh^2/m = 0.025, raw and absorbed fits both
    land near the truth (small tagging bias + NNLS boundary asymmetry)."""
    _, blocksB, _, RBB = large
    m = sum(b.size for b in blocksB)
    n = 0.025 * m / 0.25
    for absorb in (False, True):
        ests, _ = _rep_means(estimate_pair_products, RBB, blocksB, PI, 0.25, n,
                             reps=16, seed=500, absorb_signal=absorb)
        assert np.abs(ests.mean(axis=0) - PI).max() < 0.14


def test_bilinear_recovery_strong_signal(large):
    """B recovers the composition when chi-square carries real LD signal."""
    _, blocksB, _, RBB = large
    m = sum(b.size for b in blocksB)
    ests, ses = _rep_means(estimate_bilinear, RBB, blocksB, PI, 0.5,
                           8.0 * m / 0.5, reps=8, seed=600)
    assert ests.shape[0] >= 6               # guard accepts strong signal
    assert np.abs(ests.mean(axis=0) - PI).max() < 0.08


def test_bilinear_declines_without_signal(large):
    _, blocksB, _, RBB = large
    rng = np.random.default_rng(700)
    z = simulate_admixture_mvn(RBB, PI, 0.0, 4000, rng)
    r = estimate_bilinear(z, RBB, blocks=blocksB)
    assert r["proportions"] is None
    assert "no polygenic signal" in r["signal_note"]


def test_naive_ldscore_regression_is_misspecified(small):
    """Regressing chi^2 on the plain per-ancestry scores drops the cross
    term and is biased; the bilinear design is exact (report Section 5)."""
    sizes, blocks, paths, RB = small
    Lb, pairs = bilinear_ld_scores(RB, blocks)
    ell_a = Lb @ np.array([PI[0] ** 2, 2 * PI[0] * PI[1], PI[1] ** 2])
    chi2 = 0.75 + 2.0 * ell_a               # exact model expectation, no noise

    def demeaned_nnls(X):
        Xc = X - X.mean(axis=0)
        yc = chi2 - chi2.mean()
        return _nnls(Xc, yc)

    b_naive = demeaned_nnls(Lb[:, [0, 2]])  # only the plain scores l^(k)
    pi_naive = b_naive / b_naive.sum()
    b_full = demeaned_nnls(Lb)
    from ppb.ancestry import _bilinear_recover
    pi_bilinear, _, _ = _bilinear_recover(b_full, pairs, 2)
    assert np.abs(pi_bilinear - PI).max() < 1e-6
    assert np.abs(pi_naive - PI).max() > 0.02


def test_k4_recovery():
    rng = np.random.default_rng(11)
    sizes = rng.integers(15, 36, 300)
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    blocks = [np.arange(bounds[b], bounds[b + 1]) for b in range(300)]
    ranges = [(0.55, 0.90), (0.08, 0.38), (0.32, 0.62), (0.0, 0.15)]
    paths = [draw_ld_paths(rng, sizes, lo, hi) for lo, hi in ranges]
    RB = [[M[np.ix_(b, b)] for b in blocks]
          for M in (block_correlations(p) for p in paths)]
    pi4 = np.array([0.4, 0.3, 0.2, 0.1])
    ests, _ = _rep_means(estimate_pair_products, RB, blocks, pi4, 0.0, 1000,
                         reps=8, seed=800)
    assert np.abs(ests.mean(axis=0) - pi4).max() < 0.10


def test_single_block_reports_no_jackknife():
    sizes, blocks, paths, RB = _landscape(0, 1)
    rng = np.random.default_rng(0)
    z = simulate_admixture_mvn(RB, PI, 0.0, 1000, rng)
    r = estimate_pair_products(z, RB, blocks=blocks)
    assert r["proportions_se"] is None
    assert r["jackknife_note"]


# --------------------------------------- individual-level arm (mosaic GWAS)

@pytest.fixture(scope="module")
def il_design(small):
    """Shared allele frequencies and reference panels (references are a fixed
    resource in practice); tests resimulate only the admixed GWAS."""
    from ppb.simulate import simulate_admixture_references
    sizes, blocks, paths, _ = small
    m = sum(int(s) for s in sizes)
    out = {}
    for fst, seed in ((0.05, 1000), (0.2, 1100)):
        rng = np.random.default_rng(seed)
        maf = bn_freqs_multi(rng, m, fst, 2)
        refs = simulate_admixture_references(3000, [int(s) for s in sizes],
                                             maf, paths, rng)
        out[fst] = {"maf": maf, "rng": rng,
                    "refs": [[R[np.ix_(b, b)] for b in blocks] for R in refs]}
    return out


def _il_replicates(design, small, reps=8):
    from ppb.simulate import (draw_effects, marginal_stats, simulate_phenotype)
    sizes, blocks, paths, _ = small
    sizes = [int(s) for s in sizes]
    m = sum(sizes)
    ests, ses, notes = [], [], []
    for _ in range(reps):
        X = simulate_admixed_genotypes(3000, sizes, design["maf"], paths, PI,
                                       design["rng"])
        y = simulate_phenotype(X, draw_effects(m, 50, design["rng"]), 0.0,
                               design["rng"])
        _, t = marginal_stats(X, y)
        r = estimate_pair_products(t, design["refs"], blocks=blocks)
        ests.append(r["proportions"])
        ses.append(r["proportions_se"])
        notes.append(r["boundary_note"])
    return np.asarray(ests), np.asarray(ses), notes


def test_individual_level_null_trait(il_design, small):
    """The mosaic GWAS (not the moment model) drives A; replicate-averaged
    recovery, and boundary estimates (if any) must carry a warning note
    rather than a confident zero SE."""
    ests, ses, notes = _il_replicates(il_design[0.05], small)
    assert np.abs(ests.mean(axis=0) - PI).max() < 0.15
    boundary = (ests < 1e-6).any(axis=1)
    assert (ses[~boundary] > 0).all()
    for on_boundary, note in zip(boundary, notes):
        assert (note is not None) == bool(on_boundary)


def test_individual_level_high_fst(il_design, small):
    """fst = 0.2 stresses the mixture approximation (Wahlund term); the
    estimate should still land near the truth on average."""
    ests, _, _ = _il_replicates(il_design[0.2], small)
    assert np.abs(ests.mean(axis=0) - PI).max() < 0.17


def test_individual_level_bilinear_honesty(small):
    """B declines gracefully at h2 = 0 on real mosaic statistics (no
    fabricated composition), and reports its reason."""
    sizes, blocks, paths, _ = small
    rng = np.random.default_rng(1200)
    sim = simulate_admixture_sumstats(3000, sizes, 0.05, paths, PI, 0.0, 50,
                                      rng, n_ref=3000)
    r = estimate_bilinear(sim["z"], sim["refs"], blocks=blocks)
    assert r["proportions"] is None
    assert r["signal_note"]


def test_mosaic_realizes_the_mixture(small):
    """Simulator self-check: with shared frequencies the admixed within-block
    correlation is the ancestry mixture of the per-ancestry *observed*
    (thresholded-genotype) correlations -- reference panels at large n_ref
    stand in for the truth, since thresholding attenuates the latent AR(1)."""
    from ppb.simulate import simulate_admixture_references
    sizes, blocks, paths, RB = small
    m = sum(sizes)
    rng = np.random.default_rng(1300)
    maf = bn_freqs_multi(rng, m, 1e-6, 2)   # ~no frequency contrast
    refs = simulate_admixture_references(8000, sizes, maf, paths, rng)
    X = simulate_admixed_genotypes(6000, sizes, maf, paths, PI, rng)
    R_emp = np.corrcoef(X, rowvar=False)
    R_mix = PI[0] * refs[0] + PI[1] * refs[1]
    diffs = []
    for b, block in enumerate(blocks):
        i, j = np.triu_indices(len(block), k=1)
        diffs.append((R_emp[np.ix_(block, block)][i, j]
                      - R_mix[np.ix_(block, block)][i, j]))
    diffs = np.concatenate(diffs)
    assert np.sqrt(np.mean(diffs ** 2)) < 0.05
