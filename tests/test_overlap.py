"""Tests for the overlap detector and its simulation validation.

The fitting apparatus is experimental and lives in
``experiments/overlap_detection.py`` (withdrawn from the package, which keeps
only the fail-closed ``OverlapBasis`` marker); see ``docs/OVERLAP.md``.
"""

import numpy as np
import pytest

import experiments.overlap_detection as overlap_module
from ppb import OverlapBasis
from ppb.ld_backend import BlockDiagonalLD, DenseLD

from experiments.overlap_detection import (
    block_products,
    correct_overlap_numerator,
    estimate_overlap_basis,
    fit_overlap,
    run,
)


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
    return run(n=1800, per_size=15, reps=2, hutchinson_draws=16, seed=7)


def test_hutchinson_basis_recovers_the_analytic_trainer_trace(overlap_simulation):
    """The stochastic basis is only usable if it estimates the quantity it
    claims to. For a linear trainer that quantity is tr(Phi_b' K_b), which is
    computable exactly, so the estimate is checked against it rather than
    against itself."""
    assert overlap_simulation["basis_error"] < 0.05


def test_diffuse_architecture_is_refused_not_corrected(overlap_simulation):
    """The documented refusal corner. Signal and basis are both near-constant
    across blocks when every block carries similar signal, so they cannot be
    separated -- and the gate says so instead of correcting on a collinear fit."""
    out = overlap_simulation
    assert set(out["diffuse_status"]) <= {"weak_identification", "nonidentifiable"}
    assert out["diffuse_vif"] > 2.0


def test_sparse_architecture_identifies_and_corrects(overlap_simulation):
    """With per-block signal variation the same code identifies the design,
    recovers the coupling, and brings an inflated statistic back to the
    independent anchor."""
    out = overlap_simulation
    assert set(out["sparse_full_status"]) == {"correctable"}
    assert out["sparse_vif"] < 2.0
    assert out["gamma_full_ratio"] == pytest.approx(1.0, rel=0.4)
    # The naive in-sample statistic is wildly inflated; the corrected one lands
    # near the honest anchor and, crucially, does not overshoot below it.
    assert out["r2_naive_full"] > 10 * out["r2_honest_full"]
    assert out["r2_corr_full"] < 0.25 * out["r2_naive_full"]
    assert 0.5 * out["r2_honest_full"] < out["r2_corr_full"] < 3.0 * out["r2_honest_full"]


def test_partial_overlap_is_corrected_to_the_anchor(overlap_simulation):
    out = overlap_simulation
    assert set(out["sparse_low_status"]) == {"correctable"}
    assert out["gamma_low_ratio"] == pytest.approx(1.0, rel=0.5)
    assert out["r2_corr_low"] == pytest.approx(out["r2_honest_low"], rel=1.0)
    assert out["r2_corr_low"] < 0.5 * out["r2_naive_low"]


def test_no_correction_is_issued_at_the_null(overlap_simulation):
    """The false-positive check: with zero shared individuals the detector must
    never issue a correction, whatever else it concludes."""
    out = overlap_simulation
    assert out["corrections_at_null"] == 0
    assert "correctable" not in out["sparse_null_status"]


def test_discontinuous_trainer_fails_the_perturbation_stability_gate(
        overlap_simulation):
    """A p+T trainer has a locally stable Jacobian that misses its own selection
    response, so it must be refused at the basis, before any fit."""
    out = overlap_simulation
    assert out["sparse_trainer_basis"] == "unavailable"
    assert out["sparse_trainer_status"] == "basis_unavailable"


# ---------------------------------------------------------------------------
# Stochastic trainer-sensitivity basis (OVERLAP.md (O4))


def _ridge_setup(sizes=(12, 20, 8, 15), rho=0.6, lam=0.8, seed=0):
    rng = np.random.default_rng(seed)
    idxs, factors, blocks, start = [], [], [], 0
    for k in sizes:
        d = np.arange(k)
        Db = rho ** np.abs(d[:, None] - d[None, :])
        blocks.append(Db)
        factors.append(np.linalg.cholesky(Db))
        idxs.append(np.arange(start, start + k))
        start += k
    operators = [np.linalg.solve(Db + lam * np.eye(len(Db)), np.eye(len(Db)))
                 for Db in blocks]

    def trainer(values):
        w = np.zeros_like(values)
        for A, idx in zip(operators, idxs):
            w[idx] = A @ values[idx]
        return w

    analytic = np.array([float(np.trace(A.T @ Db))
                         for A, Db in zip(operators, blocks)])
    z = rng.standard_normal(start) * 0.05
    return trainer, z, idxs, factors, analytic


def test_estimate_overlap_basis_matches_the_analytic_linear_trace():
    trainer, z, idxs, factors, analytic = _ridge_setup()
    basis = estimate_overlap_basis(
        trainer, z, idxs, factors, rng=np.random.default_rng(1),
        provenance="ridge", support_hash="h", n_draws=400)

    assert basis.kind == "jacobian_hutchinson"
    assert basis.available
    assert basis.mc_se is not None and basis.mc_se > 0
    assert basis.values.sum() == pytest.approx(analytic.sum(), rel=0.05)
    for estimated, exact in zip(basis.values, analytic):
        assert estimated == pytest.approx(exact, rel=0.15)


def test_estimate_overlap_basis_is_step_size_free_for_a_linear_trainer():
    """A linear trainer's difference quotient is exact at any step, so widely
    separated deltas must agree -- that is what makes the stability gate a test
    of the *trainer* rather than of the step size."""
    trainer, z, idxs, factors, _ = _ridge_setup()
    coarse = estimate_overlap_basis(
        trainer, z, idxs, factors, rng=np.random.default_rng(2),
        provenance="ridge", support_hash="h", n_draws=64,
        deltas=(0.001, 0.1, 1.0))
    assert coarse.available
    assert "spread 0.000" in coarse.provenance


def test_estimate_overlap_basis_refuses_a_discontinuous_trainer():
    """Hard thresholding: locally the Jacobian is a constant selection matrix,
    but which variants are selected responds to the same shared noise, and no
    single derivative describes that. Must fail closed, not return a number."""
    _, z, idxs, factors, _ = _ridge_setup()

    def thresholded(values):
        return np.where(np.abs(values) >= 0.05, values, 0.0)

    basis = estimate_overlap_basis(
        thresholded, z, idxs, factors, rng=np.random.default_rng(3),
        provenance="p+T", support_hash="h", n_draws=200)
    assert not basis.available
    assert basis.kind == "unavailable"
    assert "not stable in the perturbation scale" in basis.provenance
    # And it must be usable directly as a fail-closed input to the fit.
    assert basis.values is None


def test_estimate_overlap_basis_reports_a_zero_sensitivity_trainer():
    _, z, idxs, factors, _ = _ridge_setup()
    basis = estimate_overlap_basis(
        lambda values: np.zeros_like(values), z, idxs, factors,
        rng=np.random.default_rng(4), provenance="constant",
        support_hash="h", n_draws=8)
    assert not basis.available
    assert "estimated as zero" in basis.provenance


def test_estimate_overlap_basis_validates_its_inputs():
    trainer, z, idxs, factors, _ = _ridge_setup()
    rng = np.random.default_rng(5)
    common = dict(rng=rng, provenance="p", support_hash="h", n_draws=4)

    with pytest.raises(TypeError, match="trainer must be callable"):
        estimate_overlap_basis(None, z, idxs, factors, **common)
    with pytest.raises(ValueError, match="non-empty finite 1-D"):
        estimate_overlap_basis(trainer, np.array([np.nan, 1.0]), idxs, factors, **common)
    with pytest.raises(ValueError, match="one factor per block"):
        estimate_overlap_basis(trainer, z, idxs, factors[:-1], **common)
    with pytest.raises(ValueError, match="blocks overlap"):
        estimate_overlap_basis(trainer, z, [idxs[0], idxs[0]], factors[:2], **common)
    with pytest.raises(ValueError, match="out of range"):
        estimate_overlap_basis(trainer, z, [np.array([10 ** 6])], factors[:1], **common)
    with pytest.raises(ValueError, match="at least two finite positive"):
        estimate_overlap_basis(trainer, z, idxs, factors,
                               **{**common, "deltas": (0.1,)})
    with pytest.raises(ValueError, match="n_draws must be at least 2"):
        estimate_overlap_basis(trainer, z, idxs, factors,
                               **{**common, "n_draws": 1})
    with pytest.raises(ValueError, match="scale must be finite"):
        estimate_overlap_basis(trainer, z, idxs, factors,
                               **{**common, "scale": 0.0})
    with pytest.raises(ValueError, match="finite vector of length"):
        estimate_overlap_basis(lambda v: v[:-1], z, idxs, factors, **common)


def test_estimated_basis_drives_a_real_correction():
    """End to end at the object level: an estimated basis is accepted by
    fit_overlap and owns the support the correction is computed on."""
    trainer, z, idxs, factors, _ = _ridge_setup()
    basis = estimate_overlap_basis(
        trainer, z, idxs, factors, rng=np.random.default_rng(6),
        provenance="ridge", support_hash="h", n_draws=64)

    rng = np.random.default_rng(7)
    n = 200
    q = np.tile(basis.values, n // len(basis.values) + 1)[:n]
    # Block products of a real score are predominantly positive, so give the
    # latent signal a positive mean; a mean-zero signal sums to nothing and the
    # correction then reverses the numerator's sign rather than shrinking it.
    # The spread matters as much as the mean: signal and basis are both
    # positive columns, so separating them needs per-block signal variation --
    # the same thing that decides the sparse/diffuse boundary in the simulation.
    signal = rng.normal(1.0, 2.0, size=n)
    noise = 0.02
    u_r = signal + rng.normal(0.0, noise, size=n)
    u_t = 1.2 * signal + 0.05 * q + rng.normal(0.0, noise, size=n)
    fit = fit_overlap(
        u_t, u_r, np.full(n, noise ** 2), np.full(n, noise ** 2),
        basis=OverlapBasis(values=q, kind=basis.kind,
                           provenance=basis.provenance,
                           support_hash=basis.support_hash),
        groups=np.arange(n) % 16)
    assert fit.basis_kind == "jacobian_hutchinson"
    assert fit.status == "correctable"
    assert correct_overlap_numerator(fit) < fit.numerator_target
