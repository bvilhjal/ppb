"""LD scores (C1) and LD-score regression (C2)-(C3).

The point of these is a diagnostic PPB has never had: whether the target's `z` is
on the scale the estimator assumes. So the tests check both that it fires on a
real deflation and that it stays quiet when there is nothing to find -- a
detector that cannot do the second is worthless.
"""

import numpy as np
import pytest

from ppb.ld_backend import (
    BlockDiagonalLD, DenseLD, DenseLDInt8, LowRankLD)

from experiments.z_calibration import (  # noqa: E402
    consistency, genomic_control_sweep, ldscore_regression)


# --------------------------------------------------------------------------
# (C1) LD scores


def test_ld_scores_match_the_definition_across_backends():
    """Every representation must agree on sum_k D_jk^2, within its own error."""
    rng = np.random.default_rng(0)
    D = np.corrcoef(rng.normal(size=(600, 25)).T)
    exact = (D ** 2).sum(axis=1)

    assert DenseLD(D).ld_scores() == pytest.approx(exact, abs=1e-12)
    assert DenseLDInt8.from_dense(D).ld_scores() == pytest.approx(exact, abs=5e-3)
    assert DenseLDInt8.from_dense(D).packed().ld_scores() == pytest.approx(exact, abs=5e-3)
    chol = np.linalg.cholesky(D + 1e-10 * np.eye(25))
    assert LowRankLD(chol).ld_scores() == pytest.approx(exact, abs=1e-8)


def test_an_ld_score_is_at_least_one_and_at_most_the_block_size():
    """The diagonal contributes 1; perfect LD across a block contributes m_b."""
    rng = np.random.default_rng(1)
    D = np.corrcoef(rng.normal(size=(500, 12)).T)
    scores = DenseLD(D).ld_scores()
    assert np.all(scores >= 1.0 - 1e-9)
    assert np.all(scores <= 12.0 + 1e-9)
    assert DenseLD(np.eye(12)).ld_scores() == pytest.approx(np.ones(12))
    assert DenseLD(np.ones((5, 5))).ld_scores() == pytest.approx(np.full(5, 5.0))


def test_block_diagonal_ld_scores_are_within_block_sums():
    rng = np.random.default_rng(2)
    a = np.corrcoef(rng.normal(size=(400, 8)).T)
    b = np.corrcoef(rng.normal(size=(400, 5)).T)
    ld = BlockDiagonalLD([(DenseLD(a), np.arange(8)),
                          (DenseLD(b), np.arange(8, 13))])
    scores = ld.ld_scores()
    assert scores.shape == (13,)
    assert scores[:8] == pytest.approx((a ** 2).sum(axis=1))
    assert scores[8:] == pytest.approx((b ** 2).sum(axis=1))


# --------------------------------------------------------------------------
# (C2)-(C3) regression


def _model(rng, m, *, n=100_000, h2=0.3, lam=1.0, n_blocks=200):
    ell = np.exp(rng.normal(1.0, 0.8, size=m))
    chisq = (1.0 + n * h2 / m * ell) * rng.chisquare(1, size=m) / lam
    return chisq, ell, n, rng.integers(0, n_blocks, size=m)


def test_it_recovers_a_known_intercept_and_heritability():
    rng = np.random.default_rng(3)
    chisq, ell, n, blocks = _model(rng, 200_000)
    fit = ldscore_regression(chisq, ell, n, n_variants=200_000, blocks=blocks)

    assert fit.intercept == pytest.approx(1.0, abs=4.0 * fit.intercept_se)
    assert fit.h2 == pytest.approx(0.3, rel=0.1)
    assert fit.n_blocks == 200
    assert fit.z_scale == pytest.approx(1.0 / np.sqrt(fit.intercept))
    assert fit.r2_scale == pytest.approx(1.0 / fit.intercept)


def test_the_intercept_tracks_one_over_lambda_under_genomic_control():
    rng = np.random.default_rng(4)
    for lam in (1.25, 1.5, 2.0):
        chisq, ell, n, blocks = _model(rng, 200_000, lam=lam)
        fit = ldscore_regression(chisq, ell, n, n_variants=200_000, blocks=blocks)
        assert fit.intercept == pytest.approx(1.0 / lam, rel=0.05)
        assert fit.z_scale == pytest.approx(np.sqrt(lam), rel=0.05)
        assert fit.deflation_detected


def test_deflation_is_not_claimed_without_evidence():
    """At 2,000 variants the intercept's own SE exceeds 1, so nothing is claimed.

    A detector that fires on noise is worse than none: it would license dividing
    a correct R² by a number pulled out of the sampling error.
    """
    rng = np.random.default_rng(5)
    chisq, ell, n, blocks = _model(rng, 2_000, lam=1.5, n_blocks=100)
    fit = ldscore_regression(chisq, ell, n, n_variants=2_000, blocks=blocks)

    assert fit.intercept_se > 0.3, "this regime is supposed to be badly determined"
    assert not fit.deflation_detected
    # Small m can even drive the intercept negative, which is nonsense rather
    # than an enormous correction -- that path must refuse a scale entirely.
    assert fit.note, "a refusal must say why"
    assert any(reason in fit.note for reason in
               ("not by two standard errors", "no standard error", "not positive"))
    if fit.intercept <= 0.0:
        assert fit.z_scale is None and fit.r2_scale is None


def test_deflation_is_not_claimed_when_no_blocks_are_given():
    """Without a jackknife there is no standard error, so no verdict is possible."""
    rng = np.random.default_rng(6)
    chisq, ell, n, _ = _model(rng, 50_000, lam=2.0)
    fit = ldscore_regression(chisq, ell, n, n_variants=50_000)

    assert fit.intercept < 1.0 and fit.intercept_se is None
    assert not fit.deflation_detected
    assert "no standard error" in fit.note


def test_the_null_does_not_produce_a_false_positive():
    rng = np.random.default_rng(7)
    for seed_shift in range(3):
        chisq, ell, n, blocks = _model(
            np.random.default_rng(100 + seed_shift), 200_000, lam=1.0)
        fit = ldscore_regression(chisq, ell, n, n_variants=200_000, blocks=blocks)
        assert not fit.deflation_detected, f"false positive at intercept {fit.intercept}"


def test_leverage_reports_how_far_the_intercept_is_extrapolated():
    rng = np.random.default_rng(8)
    spread, _, n, blocks = _model(rng, 20_000)
    ell_spread = np.exp(rng.normal(1.0, 0.8, size=20_000))
    ell_flat = np.full(20_000, 3.0) + rng.normal(0.0, 0.01, size=20_000)

    wide = ldscore_regression(spread, ell_spread, n, blocks=blocks)
    narrow = ldscore_regression(spread, ell_flat, n, blocks=blocks)
    assert narrow.leverage > 10.0 * wide.leverage


@pytest.mark.parametrize("bad,match", [
    (dict(chisq=[1.0, -1.0, 2.0]), "non-negative"),
    (dict(ld_scores=[1.0, 0.0, 2.0]), "positive"),
    (dict(n=0.0), "positive sample size"),
    (dict(chisq=[1.0, 2.0]), "at least 3"),
])
def test_invalid_input_is_rejected(bad, match):
    kwargs = dict(chisq=[1.0, 2.0, 3.0], ld_scores=[1.0, 2.0, 3.0], n=1000.0)
    kwargs.update(bad)
    if len(kwargs["chisq"]) != len(kwargs["ld_scores"]):
        kwargs["ld_scores"] = kwargs["ld_scores"][:len(kwargs["chisq"])]
    with pytest.raises(ValueError, match=match):
        ldscore_regression(kwargs["chisq"], kwargs["ld_scores"], kwargs["n"])


def test_flat_ld_scores_are_refused_rather_than_extrapolated():
    with pytest.raises(ValueError, match="no variance"):
        ldscore_regression([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], 1000.0)


# --------------------------------------------------------------------------
# The experiment's claims


def test_the_estimator_is_consistent_in_the_variant_count():
    """The intercept's SE must fall with m -- it is why real use needs ~10^5."""
    rng = np.random.default_rng(9)
    rows = consistency(rng)
    ses = [se for _, _, se, _ in rows]
    assert ses == sorted(ses, reverse=True), f"SE should shrink with m: {ses}"
    assert ses[0] > 0.5 and ses[-1] < 0.02
    for m, intercept, se, h2 in rows:
        assert intercept == pytest.approx(1.0, abs=4.0 * se)
        assert h2 == pytest.approx(0.3, rel=0.15)


def test_block_diagonal_ld_scores_give_a_clean_null_and_a_true_positive():
    """The end-to-end claim, on genotypes rather than on the model's own draws."""
    rng = np.random.default_rng(10)
    _, _, _, rows = genomic_control_sweep(rng, n=4_000, repeats=20)
    by_lambda = {lam: row for lam, *row in [(r[0], *r[1:]) for r in rows]}

    null_intercept, null_se, _, null_detected = by_lambda[1.0]
    assert not null_detected, "block-diagonal LD scores must not fabricate deflation"
    assert null_intercept == pytest.approx(1.0, abs=4.0 * null_se)

    for lam in (1.6, 2.0):
        intercept, se, z_scale, detected = by_lambda[lam]
        assert detected, f"lambda={lam} deflation should be caught"
        assert intercept == pytest.approx(1.0 / lam, rel=0.15)
        assert z_scale == pytest.approx(np.sqrt(lam), rel=0.08)
