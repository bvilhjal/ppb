"""Gauge-path validation: dosage weights x target genotype SD against truth.

``ppb.evaluate(weight_scale="dosage")`` -- the conversion every real
cross-ancestry evaluation performs -- checked against individual-level
``corr^2`` on the target cohort, on a shuffled submission with harmonization
in the path. See ``experiments/gauge_validation.py``.

Assertions are multi-seed means or robust per-seed bounds, chosen with
explicit margin over the measured values (seeds 0-4, 2026-08-16):

- homogeneous B: empirical-SD error max |3.7|% per seed, mean -0.8%;
  discovery-SD bias >= +14.9% per seed; HWE-empirical paired mean +0.03%.
- admixed B (sub_fst=0.1): per-seed error scatters to |13.9|% -- the
  independent LD panel, not the gauge, dominates at these n; the paired
  HWE-empirical mean is +0.9% (SE 0.7%), i.e. small, not zero.
"""

import numpy as np

from experiments.gauge_validation import run


SEEDS = (0, 1, 2, 3, 4)


def _rel(rows, key):
    return np.array([(r[key] - r["truth"]) / r["truth"] for r in rows])


def test_dosage_weights_with_empirical_sd_recover_truth():
    rows = [run(seed=s) for s in SEEDS]
    emp = _rel(rows, "r2_empirical_sd")
    # (X2) discharged with empirical B SDs: unbiased in the mean, and the
    # per-seed deviation is LD-panel sampling noise, bounded well away from
    # the wrong-gauge failure below.
    assert abs(emp.mean()) < 0.03
    assert np.abs(emp).max() < 0.06
    # The sumstats-frame remap must be the same evaluation, exactly.
    for r in rows:
        assert r["r2_sumstats_frame"] == r["r2_empirical_sd"]
    # The submission was shuffled and still recovered: harmonization ran.
    for r in rows:
        assert r["n_scored"] > 0


def test_discovery_ancestry_sd_is_a_wrong_gauge():
    rows = [run(seed=s) for s in SEEDS]
    asd = _rel(rows, "r2_discovery_sd")
    assert (asd > 0.08).all()          # measured min +14.9%


def test_hwe_sd_matches_empirical_in_a_homogeneous_target():
    rows = [run(seed=s) for s in SEEDS]
    paired = np.array([
        (r["r2_hwe_sd"] - r["r2_empirical_sd"]) / r["truth"] for r in rows])
    # Near-HWE homogeneous B: the approximation costs almost nothing
    # (measured paired mean +0.03%).
    assert abs(paired.mean()) < 0.01


def test_admixed_target_panel_noise_dominates_and_hwe_residual_is_small():
    rows = [run(seed=s, sub_fst=0.1) for s in SEEDS]
    emp = _rel(rows, "r2_empirical_sd")
    asd = _rel(rows, "r2_discovery_sd")
    paired = np.array([
        (r["r2_hwe_sd"] - r["r2_empirical_sd"]) / r["truth"] for r in rows])
    # The empirical gauge stays usable on average (no systematic gauge bias),
    # but the independent-panel denominator makes single draws scatter --
    # quote a block jackknife, not a point estimate, in this regime.
    assert abs(emp.mean()) < 0.10
    assert np.abs(emp).max() > 0.05
    assert asd.mean() > 0.05           # wrong gauge still biased on average
    # The HWE residual under structure is real but second-order at this fst:
    # positive on average, small, and not resolvable per-seed.
    assert -0.01 < paired.mean() < 0.03
