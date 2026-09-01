"""Tests for the LD-moment ancestry estimators (ppb.ancestry) and their
simulation harness (ppb.simulate admixture helpers).

Structure: unit pins for the numerical pieces, then the two simulation arms
of docs/ancestry_report/ancestry_report.tex -- model-consistent MVN draws
(estimator behavior, rough jackknife scale, tagging-bias absorption) and
individual-level admixed-GWAS mosaics (behavior under one Wahlund/frequency-
contrast and reference-estimation-noise design).
"""

import numpy as np
import pytest

from ppb.ancestry import (_conditional_design_diagnostics,
                          _design_diagnostics, _nnls, bilinear_ld_scores,
                          block_groups,
                          estimate_bilinear, estimate_bilinear_from_design,
                          estimate_pair_products,
                          estimate_pair_products_from_design, ld_scores,
                          pair_design)
from ppb.simulate import (block_correlations, bn_freqs_multi,
                          draw_ld_paths,
                          simulate_admixture_mvn, simulate_admixture_sumstats,
                          simulate_admixed_genotypes)

PI = np.array([0.65, 0.35])
RANGES_K2 = [(0.50, 0.90), (0.10, 0.50)]  # synthetic high-/low-LD landscapes


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


def test_pair_quadratic_design_exactly_spans_mixture_square():
    """The non-commuting cross term needs both matrix-product orders."""
    rng = np.random.default_rng(91)

    def random_correlation():
        A = rng.standard_normal((7, 7))
        covariance = A @ A.T + np.eye(7)
        scale = np.sqrt(np.diag(covariance))
        return covariance / np.outer(scale, scale)

    R0, R1 = random_correlation(), random_correlation()
    assert np.linalg.norm(R0 @ R1 - R1 @ R0) > 0.1
    pi = np.array([0.6, 0.4])
    ii, jj, _, _, L2 = pair_design(
        [R0, R1], floor=0.0, cap=100, quadratic=True
    )
    coefficients = np.array([pi[0] ** 2, 2 * pi[0] * pi[1], pi[1] ** 2])
    mixture = pi[0] * R0 + pi[1] * R1
    assert np.allclose(L2 @ coefficients, (mixture @ mixture)[ii, jj])


def test_conditional_design_detects_nuisance_aliasing():
    """Raw full rank does not identify L coefficients conditional on L2."""
    L = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0],
                  [2.0, -1.0],
                  [-1.0, 2.0]])
    L2 = L[:, [0]]
    raw_rank, raw_condition = _design_diagnostics(L)
    rank, condition = _conditional_design_diagnostics(L, L2)
    assert raw_rank == 2 and np.isfinite(raw_condition)
    assert rank == 1
    assert np.isinf(condition)


@pytest.mark.parametrize("groups", [
    [[0], [0], [1], [2]],
    [[0], [], [1], [2]],
    [[0], [1], [3]],
    [[0], [1]],
])
def test_custom_block_groups_must_be_a_partition(groups):
    with pytest.raises(ValueError, match="groups must partition|each group"):
        block_groups(3, groups)


def test_bilinear_scores_match_definition(small):
    sizes, blocks, paths, RB = small
    L, pairs = bilinear_ld_scores(RB, blocks)
    assert pairs == [(0, 0), (0, 1), (1, 1)]
    assert np.allclose(L[:, 0], ld_scores(RB[0], blocks))
    for b, block in enumerate(blocks):
        r0, r1 = RB[0][b], RB[1][b]
        assert np.allclose(L[block, 2], np.einsum("ij,ij->i", r1, r1))
        assert np.allclose(L[block, 1], np.einsum("ij,ij->i", r0, r1))


def test_bilinear_scores_preserve_global_order_for_reversed_dense_blocks():
    R0 = np.array([
        [1.0, 0.8, 0.0, 0.0], [0.8, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.2], [0.0, 0.0, 0.2, 1.0],
    ])
    R1 = np.array([
        [1.0, 0.4, 0.0, 0.0], [0.4, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.7], [0.0, 0.0, 0.7, 1.0],
    ])
    blocks = [np.array([0, 1]), np.array([2, 3])]
    ordered, pairs = bilinear_ld_scores([R0, R1], blocks)
    reversed_rows, reversed_pairs = bilinear_ld_scores(
        [R0, R1], list(reversed(blocks))
    )
    assert reversed_pairs == pairs
    assert np.allclose(reversed_rows, ordered)
    with pytest.raises(ValueError, match="partition"):
        bilinear_ld_scores([R0, R1], [np.array([0, 1]), np.array([1, 2])])


def test_reference_validation():
    sizes, blocks, paths, RB = _landscape(0, 6)
    with pytest.raises(ValueError, match="at least two"):
        estimate_pair_products(np.zeros(sum(sizes)), RB[:1], blocks=blocks)
    with pytest.raises(ValueError, match="same number of blocks"):
        estimate_bilinear(np.zeros(sum(sizes)), [RB[0], RB[1][:3]],
                          blocks=blocks)
    with pytest.raises(ValueError, match="match the references"):
        estimate_pair_products(np.zeros(sum(sizes) + 1), RB, blocks=blocks)


def test_pair_product_precomputed_design_matches_reference_wrapper(small):
    _, blocks, _, refs = small
    z = simulate_admixture_mvn(
        refs, PI, 0.0, 1000, np.random.default_rng(121)
    )
    ii, jj, design, pair_block, quadratic = pair_design(
        refs, blocks, quadratic=True
    )
    expected = estimate_pair_products(z, refs, blocks=blocks)
    observed = estimate_pair_products_from_design(
        z, ii, jj, design, pair_block,
        quadratic_design=quadratic,
        n_blocks=len(blocks),
    )
    for key in (
            "proportions", "proportions_se", "proportions_signal",
            "jackknife_estimates"):
        if expected[key] is None:
            assert observed[key] is None
        else:
            assert np.allclose(observed[key], expected[key], equal_nan=True)
    for key in (
            "scale", "signal", "design_condition",
            "conditional_design_condition", "residual_rms"):
        assert observed[key] == pytest.approx(expected[key])
    assert observed["n_pairs"] == expected["n_pairs"]
    assert observed["n_blocks"] == expected["n_blocks"]


def test_pair_product_precomputed_design_validation(small):
    _, blocks, _, refs = small
    m = sum(len(block) for block in blocks)
    ii, jj, design, pair_block = pair_design(refs, blocks)
    with pytest.raises(ValueError, match="quadratic_design"):
        estimate_pair_products_from_design(
            np.ones(m), ii, jj, design, pair_block
        )
    with pytest.raises(ValueError, match="integer vectors"):
        estimate_pair_products_from_design(
            np.ones(m), ii.astype(float), jj, design, pair_block,
            absorb_signal=False,
        )
    with pytest.raises(ValueError, match="n_blocks"):
        estimate_pair_products_from_design(
            np.ones(m), ii, jj, design, pair_block,
            absorb_signal=False, n_blocks=int(pair_block.max()),
        )


def test_bilinear_precomputed_design_matches_reference_wrapper(small):
    _, blocks, _, refs = small
    design, _ = bilinear_ld_scores(refs, blocks)
    variant_block = np.empty(design.shape[0], dtype=int)
    for b, block in enumerate(blocks):
        variant_block[block] = b
    sample_size = np.linspace(5000.0, 20000.0, design.shape[0])
    relative_n = sample_size / np.median(sample_size)
    coefficients = np.array([PI[0] ** 2,
                             2 * PI[0] * PI[1], PI[1] ** 2])
    z = np.sqrt(1.0 + 4.0 * relative_n * (design @ coefficients))
    expected = estimate_bilinear(
        z, refs, blocks=blocks, sample_size=sample_size
    )
    observed = estimate_bilinear_from_design(
        z, design, variant_block, n_blocks=len(blocks),
        sample_size=sample_size,
    )
    for key in (
            "proportions", "proportions_se", "proportions_raw",
            "coefficient_matrix", "heldout_signal_correlations",
            "jackknife_estimates"):
        assert np.allclose(observed[key], expected[key], equal_nan=True)
    for key in (
            "signal", "signal_z", "heldout_signal_correlation",
            "intercept", "rank1_distance", "psd_violation",
            "design_condition", "sample_size_scale"):
        assert observed[key] == pytest.approx(expected[key])


def test_bilinear_precomputed_design_validation():
    with pytest.raises(ValueError, match=r"K\(K\+1\)/2"):
        estimate_bilinear_from_design(
            np.ones(20), np.ones((20, 4)), np.zeros(20, dtype=int)
        )
    with pytest.raises(ValueError, match="variant_block"):
        estimate_bilinear_from_design(
            np.ones(20), np.ones((20, 3)), np.zeros(20)
        )


def test_pair_products_block_native_defaults_and_identifiability():
    _, blocks, _, RB = _landscape(22, 8)
    z = simulate_admixture_mvn(RB, PI, 0.0, 1000, np.random.default_rng(23))
    result = estimate_pair_products(z, RB)
    assert result["n_blocks"] == len(blocks)
    assert result["design_rank"] == 2
    assert result["conditional_design_rank"] == 2
    assert np.isfinite(result["conditional_design_condition"])
    raw = estimate_pair_products(z, RB, absorb_signal=False)
    assert raw["signal"] == 0.0
    assert raw["absorber_retained"] is False
    assert raw["conditional_design_rank"] == raw["design_rank"]
    assert raw["conditional_design_condition"] == pytest.approx(
        raw["design_condition"]
    )
    with pytest.raises(ValueError, match="not identifiable"):
        estimate_pair_products(z, [RB[0], RB[0]])
    near_duplicate = [0.999 * first + 0.001 * second
                      for first, second in zip(RB[0], RB[1])]
    with pytest.raises(ValueError, match="condition number"):
        estimate_pair_products(z, [RB[0], near_duplicate])


# ------------------------------------------------- MVN arm (model-consistent)

def _rep_means(estimator, RBB, blocksB, pi, h2, n, reps, seed, **kw):
    ests, ses = [], []
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        z = simulate_admixture_mvn(RBB, pi, h2, n, rng)
        try:
            r = estimator(z, RBB, blocks=blocksB, **kw)
        except ValueError as exc:
            if "no positive fitted linear pair-product component" not in str(exc):
                raise
            continue
        if r["proportions"] is None:
            continue
        ests.append(r["proportions"])
        ses.append(
            np.full_like(r["proportions"], np.nan)
            if r["proportions_se"] is None else r["proportions_se"]
        )
    return np.asarray(ests), np.asarray(ses)


def test_pair_products_null_trait_recovery_and_jackknife_scale(large):
    """Under one working-model arm, A recovers pi and SEs have rough scale."""
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
    # Some draws yield no estimable positive linear component and are declined.
    assert 9 <= cor.shape[0] < 12
    raw_err = np.abs(raw.mean(axis=0) - PI).max()
    cor_err = np.abs(cor.mean(axis=0) - PI).max()
    assert raw_err > 0.25                   # raw fit saturates on the high-LD ancestry
    assert raw.mean(axis=0)[0] > PI[0]      # index 0 is the high-LD landscape
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


def test_pair_products_does_not_substitute_absorber_composition():
    """Quadratic nuisance coefficients cannot replace an absent A estimand."""
    _, blocks, _, refs = _landscape(7, 100)
    m = sum(block.size for block in blocks)
    z = simulate_admixture_mvn(
        refs, PI, 0.99, int(20 * m / 0.99), np.random.default_rng(104)
    )
    with pytest.raises(ValueError, match="linear pair-product component"):
        estimate_pair_products(z, refs, blocks=blocks)


def test_bilinear_recovery_strong_signal(large):
    """B accepts a noisy positive control satisfying its bilinear model."""
    _, blocksB, _, RBB = large
    Lb, _ = bilinear_ld_scores(RBB, blocksB)
    coefficients = np.array([PI[0] ** 2,
                             2 * PI[0] * PI[1],
                             PI[1] ** 2])
    rng = np.random.default_rng(600)
    chi2 = 1.0 + 4.0 * (Lb @ coefficients) + rng.normal(0.0, 0.05, Lb.shape[0])
    result = estimate_bilinear(np.sqrt(chi2), RBB, blocks=blocksB)
    assert np.allclose(result["proportions"], PI, atol=0.01)
    assert result["rank1_distance"] < 0.01
    assert result["psd_violation"] == 0.0
    assert result["boundary_note"] is None


def test_bilinear_boundary_note(small):
    _, blocks, _, RB = small
    Lb, _ = bilinear_ld_scores(RB, blocks)
    result = estimate_bilinear(
        np.sqrt(1.0 + 4.0 * Lb[:, 0]), RB, blocks=blocks
    )
    assert np.allclose(result["proportions"], [1.0, 0.0])
    assert "simplex boundary" in result["boundary_note"]


def test_bilinear_declines_without_signal(large):
    _, blocksB, _, RBB = large
    rng = np.random.default_rng(700)
    z = simulate_admixture_mvn(RBB, PI, 0.0, 4000, rng)
    r = estimate_bilinear(z, RBB, blocks=blocksB)
    assert r["proportions"] is None
    assert "insufficient reproducible" in r["signal_note"]


def test_bilinear_withholds_se_if_leave_group_refit_fails_screen(large):
    _, blocks, _, refs = large
    m = sum(block.size for block in blocks)
    z = simulate_admixture_mvn(
        refs, PI, 0.25, 0.5 * m / 0.25, np.random.default_rng(600)
    )
    result = estimate_bilinear(z, refs, blocks=blocks)
    assert result["proportions"] is not None
    assert result["proportions_se"] is None
    assert np.isnan(result["jackknife_estimates"]).any()
    assert "leave-group refit failed" in result["jackknife_note"]


def test_bilinear_declines_old_null_false_positives(large):
    """Every seed accepted by the former i.i.d. two-SE guard now declines."""
    _, blocksB, _, RBB = large
    old_false_positive_seeds = [713, 748, 758, 798, 812,
                                819, 844, 847, 854, 866]
    for seed in old_false_positive_seeds:
        z = simulate_admixture_mvn(
            RBB, PI, 0.0, 4000, np.random.default_rng(seed)
        )
        result = estimate_bilinear(z, RBB, blocks=blocksB)
        assert result["proportions"] is None
        assert result["signal_z"] < 4.0
        assert "held-out block groups" in result["signal_note"]


def test_bilinear_declines_nonidentifiable_references(small):
    _, blocks, _, RB = small
    z = simulate_admixture_mvn(
        RB, PI, 0.25, 4000, np.random.default_rng(94)
    )
    result = estimate_bilinear(z, [RB[0], RB[0]], blocks=blocks)
    assert result["proportions"] is None
    assert result["design_rank"] < result["design_columns"]
    assert "not identifiable" in result["signal_note"]

    near_duplicate = [0.99 * first + 0.01 * second
                      for first, second in zip(RB[0], RB[1])]
    result = estimate_bilinear(z, [RB[0], near_duplicate], blocks=blocks)
    assert result["design_rank"] == result["design_columns"]
    assert result["design_condition"] > 1e3
    assert result["proportions"] is None
    assert "condition number" in result["signal_note"]


def test_bilinear_declines_cross_only_coefficient_matrix(small):
    """A strong but indefinite cross-only coefficient matrix is impossible."""
    _, blocks, _, RB = small
    Lb, pairs = bilinear_ld_scores(RB, blocks)
    assert pairs == [(0, 0), (0, 1), (1, 1)]
    z = np.sqrt(1.0 + 4.0 * Lb[:, 1])
    result = estimate_bilinear(z, RB, blocks=blocks)
    assert result["proportions_raw"] is not None
    assert result["proportions"] is None
    assert result["psd_violation"] > 0.9
    assert "positive-semidefinite" in result["signal_note"]


def test_bilinear_variable_sample_size_uses_n_times_ld(small):
    _, blocks, _, RB = small
    Lb, _ = bilinear_ld_scores(RB, blocks)
    m = Lb.shape[0]
    sample_size = np.linspace(5000.0, 20000.0, m)
    relative_n = sample_size / np.median(sample_size)
    coefficients = np.array([PI[0] ** 2,
                             2 * PI[0] * PI[1],
                             PI[1] ** 2])
    z = np.sqrt(1.0 + 4.0 * relative_n * (Lb @ coefficients))
    result = estimate_bilinear(
        z, RB, blocks=blocks, sample_size=sample_size
    )
    assert np.allclose(result["proportions"], PI, atol=1e-8)
    assert result["sample_size_scale"] == pytest.approx(
        np.median(sample_size)
    )
    assert result["n_truncated"] == 0
    assert result["truncation_cap"] is None
    with pytest.raises(ValueError, match="sample_size"):
        estimate_bilinear(
            z, RB, blocks=blocks, sample_size=np.zeros(m)
        )


def test_bilinear_explicit_capping_is_not_accepted(small):
    _, blocks, _, RB = small
    z = simulate_admixture_mvn(
        RB, PI, 0.5, 10000, np.random.default_rng(95)
    )
    result = estimate_bilinear(z, RB, blocks=blocks, c=1.0)
    assert result["n_truncated"] > 0
    assert result["proportions"] is None
    assert result["approximation_note"] is not None
    assert "approximate" in result["signal_note"]


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
