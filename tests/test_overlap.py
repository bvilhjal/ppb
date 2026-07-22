"""Tests for the overlap detector (ppb.overlap) and its simulation validation."""

import numpy as np
import pytest

import ppb.overlap as overlap_module
from ppb.ld_backend import BlockDiagonalLD, DenseLD
from ppb.overlap import (
    OverlapBasis,
    block_products,
    correct_numerator,
    correct_overlap_numerator,
    fit_overlap,
    overlap_slope,
)

from experiments.overlap_detection import run


def _block_ld(m, bs):
    blocks = []
    for s in range(0, m, bs):
        k = min(bs, m - s)
        blocks.append((DenseLD(np.eye(k)), np.arange(s, s + k)))
    return BlockDiagonalLD(blocks)


def _known_basis(q, support=None, suffix="base"):
    return OverlapBasis(
        values=np.asarray(q, dtype=float),
        kind="linear_trace",
        provenance="synthetic w = A z_train",
        support_hash=f"synthetic-{suffix}",
        support=support,
    )


def _joint_data(alpha=1.3, gamma=0.25, noise=0.02, seed=10, n=160):
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.4, 2.0, size=n)
    signal = rng.normal(0.0, 1.0, size=n)
    var_r = np.full(n, noise ** 2)
    var_t = np.full(n, noise ** 2)
    u_r = signal + rng.normal(0.0, noise, size=n)
    u_t = alpha * signal + gamma * q + rng.normal(0.0, noise, size=n)
    groups = np.arange(n) % 16
    return u_t, u_r, var_t, var_r, q, groups


def test_block_products_matches_direct():
    rng = np.random.default_rng(0)
    m, bs = 30, 10
    ld = _block_ld(m, bs)
    w, z = rng.normal(size=m), rng.normal(size=m)
    u, v = block_products(w, z, ld)
    assert u.shape == v.shape == (3,)
    for b, s in enumerate(range(0, m, bs)):
        assert u[b] == pytest.approx(w[s:s + bs] @ z[s:s + bs])
        assert v[b] == pytest.approx(w[s:s + bs] @ w[s:s + bs])


def test_block_products_rejects_nonfinite_inputs():
    ld = _block_ld(10, 5)
    with pytest.raises(ValueError, match="finite"):
        block_products(np.ones(10), np.r_[np.ones(9), np.nan], ld)


def test_overlap_basis_restricts_scientific_basis_kinds():
    with pytest.raises(ValueError, match="basis kind"):
        OverlapBasis(
            values=np.ones(10), kind="variant_count",
            provenance="unsupported shortcut", support_hash="test")


def test_overlap_slope_recovers_known_slope():
    rng = np.random.default_rng(1)
    n_blocks = 60
    m = rng.integers(50, 150, size=n_blocks).astype(float)
    v = rng.uniform(0.5, 2.0, size=n_blocks)
    gamma_true = 0.01
    u_r = rng.normal(0.0, np.sqrt(v), size=n_blocks)
    u_t = gamma_true * m + rng.normal(0.0, np.sqrt(v), size=n_blocks)
    with pytest.warns(DeprecationWarning):
        est = overlap_slope(u_t, u_r, m, v, v, groups=np.arange(n_blocks) % 8)
    assert est.gamma == pytest.approx(gamma_true, rel=0.15)
    assert est.se > 0 and est.n_blocks == n_blocks


def test_correct_numerator():
    with pytest.warns(DeprecationWarning):
        corrected = correct_numerator(2.0, 1e-4, 10000)
    assert corrected == pytest.approx(1.0)


def test_fit_overlap_does_not_mistake_signal_rescaling_for_overlap():
    u_t, u_r, var_t, var_r, q, groups = _joint_data(
        alpha=1.35, gamma=0.0, seed=11)
    fit = fit_overlap(
        u_t, u_r, var_t, var_r, basis=_known_basis(q), groups=groups)
    assert fit.alpha == pytest.approx(1.35, rel=0.03)
    assert fit.gamma == pytest.approx(0.0, abs=0.02)
    assert fit.status == "not_detected"
    with pytest.raises(ValueError, match="not_detected"):
        correct_overlap_numerator(fit)


def test_fit_overlap_jointly_recovers_alpha_and_gamma():
    u_t, u_r, var_t, var_r, q, groups = _joint_data(
        alpha=1.25, gamma=0.30, seed=12)
    fit = fit_overlap(
        u_t, u_r, var_t, var_r, basis=_known_basis(q), groups=groups)
    assert fit.alpha == pytest.approx(1.25, rel=0.03)
    assert fit.gamma == pytest.approx(0.30, rel=0.08)
    assert fit.gamma_z > 3.0
    assert fit.status == "correctable"


def test_fit_overlap_refuses_exact_signal_basis_collinearity():
    rng = np.random.default_rng(13)
    n = 80
    q = rng.uniform(0.5, 2.0, size=n)
    u_r = 2.0 * q
    u_t = 1.4 * u_r + 0.3 * q
    variance = np.full(n, 0.01)
    fit = fit_overlap(
        u_t, u_r, variance, variance, basis=_known_basis(q),
        groups=np.arange(n) % 10, min_blocks=20, min_groups=8)
    assert fit.status == "nonidentifiable"
    assert np.isinf(fit.vif)
    with pytest.raises(ValueError, match="nonidentifiable"):
        correct_overlap_numerator(fit)


def test_fit_overlap_refuses_unknown_trainer_basis():
    n = 40
    fit = fit_overlap(
        np.ones(n), np.ones(n), np.ones(n), np.ones(n),
        basis=OverlapBasis.unavailable("only final catalog weights are available"),
        groups=np.arange(n) % 10)
    assert fit.status == "basis_unavailable"
    assert fit.q_total is None
    with pytest.raises(ValueError, match="basis_unavailable"):
        correct_overlap_numerator(fit)


def test_fit_overlap_refuses_excluded_nonzero_basis_mass():
    u_t, u_r, var_t, var_r, q, groups = _joint_data(seed=14, n=80)
    var_t[-1] = var_r[-1] = 0.0
    fit = fit_overlap(
        u_t, u_r, var_t, var_r, basis=_known_basis(q), groups=groups,
        min_blocks=20, min_groups=8)
    assert fit.status == "excluded_basis"
    assert fit.excluded_basis_fraction > 0
    with pytest.raises(ValueError, match="excluded_basis"):
        correct_overlap_numerator(fit)


def test_fit_owns_exact_basis_support_and_valid_correction():
    u_t, u_r, var_t, var_r, q, groups = _joint_data(
        alpha=1.2, gamma=0.4, noise=0.005, seed=15, n=80)
    u_r += 0.5
    u_t += 1.2 * 0.5
    # A score-inactive block may be excluded only when its sensitivity is zero.
    q[-1] = 0.0
    u_t[-1] = u_r[-1] = 0.0
    var_t[-1] = var_r[-1] = 0.0
    basis = _known_basis(q, suffix="inactive-final-block")
    fit = fit_overlap(
        u_t, u_r, var_t, var_r, basis=basis, groups=groups,
        min_blocks=20, min_groups=8)
    assert fit.status == "correctable"
    assert fit.q_total == pytest.approx(q.sum())
    assert fit.q_fit == pytest.approx(q[:-1].sum())
    assert fit.excluded_basis_fraction == 0.0
    expected = u_t.sum() - fit.gamma * q.sum()
    assert correct_overlap_numerator(fit) == pytest.approx(expected)


def test_fit_overlap_refuses_tiny_scale_invalid_noise_covariance():
    n = 50
    variance = np.full(n, 1e-8)
    covariance = np.full(n, 1.01e-8)
    with pytest.raises(ValueError, match="positive semidefinite"):
        fit_overlap(
            np.ones(n), np.ones(n), variance, variance,
            cov_target_reference=covariance,
            basis=_known_basis(np.linspace(0.5, 2.0, n)),
            groups=np.arange(n) % 15)


def test_fit_overlap_refuses_degenerate_jackknife(monkeypatch):
    n = 80
    rng = np.random.default_rng(16)
    q = rng.uniform(0.5, 2.0, size=n)
    u_r = rng.normal(size=n)
    u_t = 1.2 * u_r + 0.3 * q
    variance = np.full(n, 0.01)

    def fixed_fit(*args, **kwargs):
        return 1.2, 0.3, 0.0, False

    monkeypatch.setattr(overlap_module, "_profile_eiv", fixed_fit)
    fit = fit_overlap(
        u_t, u_r, variance, variance, basis=_known_basis(q),
        groups=np.arange(n) % 10, min_blocks=20, min_groups=8)
    assert fit.status == "unstable"
    assert fit.gamma_se == 0.0
    with pytest.raises(ValueError, match="unstable"):
        correct_overlap_numerator(fit)


def test_fit_overlap_refuses_sign_reversing_correction():
    u_t, u_r, var_t, var_r, q, groups = _joint_data(
        alpha=1.0, gamma=0.4, noise=0.01, seed=17)
    u_r -= 0.2
    u_t -= 0.2
    assert u_t.sum() > 0
    fit = fit_overlap(
        u_t, u_r, var_t, var_r, basis=_known_basis(q), groups=groups)
    assert fit.numerator_target - fit.gamma * fit.q_total < 0
    assert fit.status == "sign_reversal"
    with pytest.raises(ValueError, match="sign_reversal"):
        correct_overlap_numerator(fit)


def test_fit_overlap_refuses_unmodelled_block_heterogeneity():
    u_t, u_r, var_t, var_r, q, groups = _joint_data(
        alpha=1.2, gamma=0.3, noise=0.02, seed=18)
    u_t += np.where(np.arange(u_t.size) % 2, 0.5, -0.5)
    fit = fit_overlap(
        u_t, u_r, var_t, var_r, basis=_known_basis(q), groups=groups)
    assert fit.heterogeneity_ratio > 2.0
    assert fit.status == "heterogeneous"
    with pytest.raises(ValueError, match="heterogeneous"):
        correct_overlap_numerator(fit)


@pytest.fixture(scope="module")
def overlap_simulation():
    return run(reps=2, seed=7)


def test_simulation_known_dense_basis_is_fail_closed(overlap_simulation):
    out = overlap_simulation
    assert abs(out["gamma_null"]) < 2e-5                       # null calibrated
    assert out["alpha_null"] == pytest.approx(1.0, abs=0.15)
    assert set(out["status_null"]) == {"weak_identification"}
    assert set(out["status_low"]) == {"weak_identification"}
    assert out["gamma_low"] == pytest.approx(out["gamma_low_true"], rel=0.35)
    assert 0 < out["gamma_full"] < out["gamma_full_true"]
    assert out["correctable_full_fraction"] == 0.0
    assert set(out["status_full"]) == {"weak_identification"}
    assert np.isnan(out["r2_corr_full"])
    assert out["r2_naive_full"] > 2 * out["r2_honest_full"]


@pytest.mark.xfail(
    strict=True,
    reason="operating boundary: alpha/gamma separation does not recover this "
           "seeded physical-overlap simulation",
)
def test_simulation_known_dense_basis_recovers_honest_anchor(overlap_simulation):
    out = overlap_simulation
    diagnostics = (
        f"alpha={out['alpha_full']:.3f}, gamma={out['gamma_full']:.3g}, "
        f"status={out['status_full']}, VIF={out['vif_full']:.2f}, "
        f"condition={out['condition_full']:.2f}")
    assert out["gamma_full"] == pytest.approx(
        out["gamma_full_true"], rel=0.35), diagnostics
    assert out["r2_corr_full"] == pytest.approx(out["r2_honest_full"], rel=0.4)


def test_simulation_unknown_sparse_basis_is_refused(overlap_simulation):
    assert overlap_simulation["sparse_status"] == "basis_unavailable"


def test_overlap_slope_default_groups_with_empty_blocks():
    """Blocks with no score variance are dropped; the default jackknife groups
    must be indexed like the inputs, not like the filtered arrays (regression:
    this raised IndexError whenever any block was filtered out)."""
    rng = np.random.default_rng(3)
    n_blocks = 60
    m = np.full(n_blocks, 100.0)
    v = np.ones(n_blocks)
    v[[3, 17, 42]] = 0.0                      # e.g. a sparse score with no weight here
    u_t = 0.01 * m + rng.normal(size=n_blocks)
    u_r = rng.normal(size=n_blocks)
    with pytest.warns(DeprecationWarning):
        est = overlap_slope(u_t, u_r, m, v, v)    # default groups
    assert est.n_blocks == n_blocks - 3
    assert est.gamma == pytest.approx(0.01, rel=0.3)
    assert est.se > 0


def test_overlap_slope_rejects_misaligned_groups():
    n_blocks = 60
    m = np.full(n_blocks, 100.0)
    v = np.ones(n_blocks)
    u = np.zeros(n_blocks)
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="one entry per block"):
            overlap_slope(u, u, m, v, v, groups=np.arange(n_blocks - 1))
