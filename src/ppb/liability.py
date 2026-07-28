"""Liability-scale R^2 for a binary trait.

PPB's binary output is not a calibrated accuracy. ``standardized_marginal``
converts ``(beta, se, n)`` to a correlation, and for a case/control GWAS the
sample size supplied is the effective one, ``4/(1/n_case + 1/n_ctrl)`` -- the
size of a *balanced* study with equivalent power. So the resulting statistic
approximates R^2 on the **observed 0/1 scale at a case fraction of one half**,
which is neither the accuracy in the population (cases are far rarer than half)
nor the accuracy on the underlying liability.

The standard conversion (Lee, Goddard, Wray & Visscher 2012,
https://doi.org/10.1002/gepi.21614) rescales an observed-scale statistic from an
ascertained sample onto the liability scale:

**(M6) Observed-to-liability rescaling.**

    R^2_liability = R^2_observed * K^2 (1-K)^2 / (phi(t)^2 P (1-P))
    t = Phi^-1(1 - K)

``K`` is the population prevalence and ``P`` the case fraction the observed
statistic was computed at. The factor is not a small correction: at K = 0.01
and P = 0.5 it is 0.55, so a balanced-sample statistic nearly doubles the
liability-scale accuracy; at K = P = 0.5 it is 1.57, because dichotomizing a
continuous liability throws information away and the observed scale understates.

**Prevalence is external knowledge.** It is not in the summary statistics, it is
trait- *and population*-specific, and (M6) is monotone in it -- so a
liability-scale number inherits whatever uncertainty the assumed prevalence
carries. For cross-ancestry work this is a second place, alongside `z` and `D`,
where the target population must be named: the same score in the same cohort has
a different liability R^2 under a different assumed `K`.
"""

from __future__ import annotations

import math

__all__ = ["liability_r2", "observed_to_liability_factor"]


def _probit(p: float) -> float:
    """``Phi^-1(p)`` by Newton iteration on ``erf`` -- exact to machine precision."""
    x = 0.0
    for _ in range(60):
        cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        if pdf < 1e-300:
            break
        step = (cdf - p) / pdf
        x -= step
        if abs(step) < 1e-14:
            break
    return x


def observed_to_liability_factor(prevalence: float, *,
                                 case_fraction: float = 0.5) -> float:
    """The multiplier in (M6): ``K^2(1-K)^2 / (phi(t)^2 P(1-P))``.

    ``case_fraction`` must match the scale the observed statistic was computed
    at. It defaults to one half because PPB standardizes case/control summary
    statistics with an *effective* sample size, which is the balanced-study
    equivalent -- passing the study's true case fraction instead would describe a
    statistic PPB did not compute.
    """
    k = float(prevalence)
    p = float(case_fraction)
    if not math.isfinite(k) or not 0.0 < k < 1.0:
        raise ValueError(f"prevalence must be strictly between 0 and 1; got {k}")
    if not math.isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError(f"case_fraction must be strictly between 0 and 1; got {p}")
    t = _probit(1.0 - k)
    phi = math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
    return (k * (1.0 - k)) ** 2 / (phi ** 2 * p * (1.0 - p))


def liability_r2(observed_r2: float, prevalence: float, *,
                 case_fraction: float = 0.5) -> float:
    """Rescale an observed-scale binary R^2 onto the liability scale by (M6).

    ``observed_r2`` is PPB's binary statistic -- ``(w^T z)^2 / (w^T D w)`` with
    ``z`` standardized on an effective sample size. ``prevalence`` is the
    population risk, which the summary statistics do not contain and the caller
    must supply from outside; see this module's docstring on what that costs.

    The rescaling is a first-order result derived for the small-``R^2`` regime.
    It is applied to a statistic that is itself an approximation -- the
    logistic-beta-to-correlation conversion -- so the two stack, and the output
    is a calibrated *scale*, not a calibrated *estimate*.
    """
    r2 = float(observed_r2)
    if not math.isfinite(r2) or r2 < 0.0:
        raise ValueError(f"observed_r2 must be finite and non-negative; got {r2}")
    return r2 * observed_to_liability_factor(prevalence, case_fraction=case_fraction)
