"""Liability-scale rescaling (M6), checked against a threshold-model simulation.

The transformation is not verified against itself: a liability is simulated, a
case/control sample is ascertained from it exactly as a real study would be, and
(M6) is asked to recover the liability-scale R² that generated the data.
"""

import math

import numpy as np
import pytest

from ppb import liability_r2, observed_to_liability_factor
from ppb.liability import _probit


@pytest.mark.parametrize("p,expected", [
    (0.5, 0.0), (0.975, 1.959963984540054), (0.99, 2.3263478740408408),
    (0.001, -3.090232306167813), (0.8413447460685429, 1.0),
])
def test_probit_matches_known_quantiles(p, expected):
    assert _probit(p) == pytest.approx(expected, abs=1e-9)


def test_the_balanced_common_trait_factor_is_exactly_half_pi():
    """At K = P = 1/2 the factor collapses to pi/2 -- an exact anchor.

    phi(0)^2 = 1/(2 pi), so the factor is (1/16) / ((1/(2 pi)) (1/4)) = pi/2.
    """
    assert observed_to_liability_factor(0.5) == pytest.approx(math.pi / 2, rel=1e-12)


def test_without_ascertainment_dichotomizing_always_loses_information():
    """With P = K there is no enrichment, so the factor is the pure scale change
    and must exceed 1: a 0/1 outcome explains less than the liability behind it."""
    for k in (0.5, 0.2, 0.1, 0.05, 0.01, 0.001):
        assert observed_to_liability_factor(k, case_fraction=k) > 1.0


def test_a_balanced_sample_of_a_rare_trait_overstates_accuracy():
    """Enrichment cuts the other way: at P = 0.5 a rare trait's observed R² is
    inflated, so the factor drops below 1 and falls as the trait gets rarer."""
    factors = [observed_to_liability_factor(k) for k in (0.2, 0.1, 0.05, 0.01, 0.001)]
    assert factors == sorted(factors, reverse=True)
    assert factors[0] > 1.0 > factors[2]


def test_rescaling_is_linear_in_the_observed_statistic():
    f = observed_to_liability_factor(0.05)
    assert liability_r2(0.0, 0.05) == 0.0
    assert liability_r2(0.04, 0.05) == pytest.approx(0.04 * f)
    assert liability_r2(0.08, 0.05) == pytest.approx(2.0 * liability_r2(0.04, 0.05))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_impossible_prevalence_and_case_fraction_are_rejected(bad):
    with pytest.raises(ValueError, match="prevalence"):
        observed_to_liability_factor(bad)
    with pytest.raises(ValueError, match="case_fraction"):
        observed_to_liability_factor(0.05, case_fraction=bad)


def test_a_negative_observed_r2_is_rejected():
    with pytest.raises(ValueError, match="observed_r2"):
        liability_r2(-0.01, 0.05)


def _threshold_model(rng, n, r, k):
    """Ascertain a balanced case/control sample from a liability threshold model.

    Returns the observed-scale R² of the score in that sample. The liability-scale
    R² is ``r**2`` by construction, which is what (M6) has to recover.
    """
    score = rng.standard_normal(n)
    liability = r * score + math.sqrt(1.0 - r * r) * rng.standard_normal(n)
    case = liability > np.quantile(liability, 1.0 - k)
    cases, controls = np.flatnonzero(case), np.flatnonzero(~case)
    take = min(cases.size, controls.size, n // 10)
    idx = np.concatenate([rng.choice(cases, take, replace=False),
                          rng.choice(controls, take, replace=False)])
    return float(np.corrcoef(score[idx], case[idx].astype(float))[0, 1] ** 2)


@pytest.mark.parametrize("k,seed", [(0.5, 11), (0.2, 12), (0.05, 13)])
def test_it_recovers_the_liability_r2_that_generated_the_data(k, seed):
    """The regime PPB operates in: the registry's binary R² are 0.025-0.044.

    The seed is a literal. ``hash()`` of anything containing a string is
    randomized per interpreter, so seeding from one makes a Monte Carlo test
    silently non-reproducible -- it passed locally and failed in CI.
    """
    rng = np.random.default_rng(seed)
    r = 0.2                                        # liability R² = 0.04
    observed = _threshold_model(rng, 800_000, r, k)
    recovered = liability_r2(observed, k)
    assert recovered == pytest.approx(r * r, rel=0.12)

    # Demanding that the rescaling *improve* on the raw value only means
    # something where the rescaling does something. The factor crosses 1 near
    # K = 0.1 (1.57 at K = 0.5, 1.31 at 0.2, but 0.85 at 0.05), so near the
    # crossing the correction is a near-no-op and Monte Carlo noise decides the
    # comparison. Gate the claim on the factor, not on K.
    factor = observed_to_liability_factor(k)
    if abs(factor - 1.0) > 0.25:
        assert abs(observed - r * r) > abs(recovered - r * r)


def test_the_rescaling_degrades_for_a_large_r2_on_a_rare_trait():
    """(M6) is a small-R² result and must not be sold as exact.

    At K = 0.01 the raw statistic is ~50% high and the rescaled one ~17% low --
    much better, and still not a calibrated estimate.
    """
    rng = np.random.default_rng(7)
    r, k = 0.4, 0.01                               # liability R² = 0.16
    observed = _threshold_model(rng, 1_200_000, r, k)
    recovered = liability_r2(observed, k)

    assert observed > r * r * 1.2, "ascertainment should inflate the raw value"
    assert abs(recovered - r * r) < abs(observed - r * r), "rescaling still helps"
    assert not math.isclose(recovered, r * r, rel_tol=0.05), (
        "this regime is where the first-order transformation visibly breaks; "
        "if it has become exact, the documented caveat needs revisiting")
