"""Tests for covariate / PC adjustment and its effect on PPB evaluation."""

import numpy as np
import pytest

from ppb import adjust, principal_components, residualize
from ppb.simulate import simulate_structured_genotypes

from experiments.pc_adjustment import run  # noqa: E402


# --- covariate utilities ---------------------------------------------------

def test_residualize_removes_covariates_and_intercept():
    rng = np.random.default_rng(0)
    C = rng.standard_normal((200, 3))
    M = rng.standard_normal((200, 5)) + C @ rng.standard_normal((3, 5)) + 2.0
    R = residualize(M, C)
    design = np.column_stack([np.ones(200), C])
    assert np.allclose(design.T @ R, 0.0, atol=1e-8)   # orthogonal to covariates


def test_residualize_is_idempotent():
    rng = np.random.default_rng(1)
    C = rng.standard_normal((150, 2))
    M = rng.standard_normal((150, 4))
    R1 = residualize(M, C)
    assert np.allclose(residualize(R1, C), R1, atol=1e-9)


def test_residualize_1d_vector():
    rng = np.random.default_rng(2)
    C = rng.standard_normal(100)
    y = 3.0 + 2.0 * C + 0.1 * rng.standard_normal(100)
    r = residualize(y, C)
    assert r.shape == (100,)
    assert abs(np.corrcoef(r, C)[0, 1]) < 1e-6         # covariate projected out


def test_adjust_standardizes_outputs():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 6))
    y = rng.standard_normal(300)
    C = rng.standard_normal((300, 2))
    Xa, ya = adjust(X, y, C)
    assert np.allclose(Xa.mean(0), 0, atol=1e-8) and np.allclose(Xa.std(0), 1, atol=1e-8)
    assert abs(ya.mean()) < 1e-8 and abs(ya.std() - 1) < 1e-8


def test_principal_components_recover_ancestry_axis():
    rng = np.random.default_rng(4)
    X, labels = simulate_structured_genotypes(2000, [30, 30, 30], fst=0.3,
                                              rho=0.5, rng=rng)
    pc1 = principal_components(X, 1)[:, 0]
    assert abs(np.corrcoef(pc1, labels)[0, 1]) > 0.9   # PC1 == ancestry axis


def test_structured_genotypes_shapes():
    rng = np.random.default_rng(5)
    X, labels = simulate_structured_genotypes(500, [20, 20], fst=0.2, rho=0.5, rng=rng)
    assert X.shape == (500, 40) and labels.shape == (500,)
    assert set(np.unique(labels)).issubset({0, 1})


# --- the effect on PPB evaluation ------------------------------------------

def test_pc_adjustment_removes_structure_confounding():
    res = run(m=200, block_size=25, n=1500, n_reps=5, seed=0)
    null_unadj, null_adj = res["null+confound"]
    gen_unadj, gen_adj = res["genetic,no confound"]

    # Stratification: unadjusted R^2 is inflated; PC adjustment drives it to ~0.
    assert null_adj < 0.01, f"adjusted spurious R^2 {null_adj:.4f} not ~0"
    assert null_unadj > 0.02, f"unadjusted spurious R^2 {null_unadj:.4f} not inflated"
    assert null_unadj > 3 * null_adj

    # Genuine genetic prediction survives adjustment (not driven to ~0).
    assert gen_adj > 0.03, f"adjusted genetic R^2 {gen_adj:.4f} lost too much signal"


def test_principal_components_rejects_too_many_components():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((50, 40))
    with pytest.raises(ValueError, match="available components"):
        principal_components(X, 51)              # min(n, m) = 40
    assert principal_components(X, 40).shape == (50, 40)
