"""Gauge-path validation: dosage weights x target genotype SD against truth.

``ppb.evaluate(weight_scale="dosage")`` -- the conversion every real
cross-ancestry evaluation performs -- checked against individual-level
``corr^2`` on the target cohort, on a shuffled submission with harmonization
in the path. See ``experiments/gauge_validation.py``.

Assertions are cross-seed means against SE-derived bounds (the F3 style; a
hard per-seed bound on a noisy draw is a bootstrap failure waiting to
happen). Measured over seeds 0-19 at the shipped parameters, 2026-09-03:

- homogeneous B: empirical-SD error mean +0.1% (SE 0.3%), max |3.7|%;
  discovery-SD bias mean +42.5% (SE 11.4%, range +5.4% to +230% -- wide, so
  the per-seed floor quoted in the old docstring does not survive seeds);
  HWE-empirical paired mean +0.04% (SE 0.06%).
- admixed B (sub_fst=0.1): per-seed error scatters to |26.8|%, mean -2.0%
  (SE 1.9%) -- the independent LD panel, not the gauge, dominates at these
  n; the paired HWE-empirical mean is +0.19% (SE 0.42%), i.e. consistent
  with zero at this fst, not a measured residual.
"""

import numpy as np

from experiments.gauge_validation import run

SEEDS = tuple(range(20))


def _rel(rows, key):
    return np.array([(r[key] - r["truth"]) / r["truth"] for r in rows])


def _mean_se(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


def test_dosage_weights_with_empirical_sd_recover_truth():
    rows = [run(seed=s) for s in SEEDS]
    emp = _rel(rows, "r2_empirical_sd")
    mean, se = _mean_se(emp)
    # (X2) discharged with empirical B SDs: the cross-seed mean is within a
    # hair of zero, well inside its own SE and far from the wrong-gauge
    # bias of the test below.
    assert abs(mean) < 4.0 * se
    assert abs(mean) < 0.02
    # The sumstats-frame remap must be the same evaluation, exactly.
    for r in rows:
        assert r["r2_sumstats_frame"] == r["r2_empirical_sd"]
    # The submission was shuffled and still recovered: harmonization ran.
    for r in rows:
        assert r["n_scored"] > 0


def test_discovery_ancestry_sd_is_a_wrong_gauge():
    rows = [run(seed=s) for s in SEEDS]
    asd = _rel(rows, "r2_discovery_sd")
    mean, se = _mean_se(asd)
    # A wrong (discovery-cohort) gauge biases the measured R^2 upward by a
    # large, systematic amount. The per-seed range is wide (see the module
    # docstring), so the claim is the cross-seed mean, not a per-seed floor.
    assert mean - 3.0 * se > 0.05, (
        f"discovery-gauge bias {mean:+.4f} ± {3*se:.4f} not clearly positive")


def test_hwe_sd_matches_empirical_in_a_homogeneous_target():
    rows = [run(seed=s) for s in SEEDS]
    paired = np.array([
        (r["r2_hwe_sd"] - r["r2_empirical_sd"]) / r["truth"] for r in rows])
    mean, se = _mean_se(paired)
    # Near-HWE homogeneous B: the approximation costs almost nothing; the
    # paired mean is indistinguishable from zero at this fst.
    assert abs(mean) < 4.0 * se
    assert abs(mean) < 0.01


def test_admixed_target_panel_noise_dominates_and_hwe_residual_is_small():
    rows = [run(seed=s, sub_fst=0.1) for s in SEEDS]
    emp = _rel(rows, "r2_empirical_sd")
    asd = _rel(rows, "r2_discovery_sd")
    paired = np.array([
        (r["r2_hwe_sd"] - r["r2_empirical_sd"]) / r["truth"] for r in rows])
    emp_mean, emp_se = _mean_se(emp)
    asd_mean, asd_se = _mean_se(asd)
    paired_mean, paired_se = _mean_se(paired)
    # The empirical gauge stays unbiased on average (the cross-seed mean is
    # within its own SE), but the independent-panel denominator makes single
    # draws scatter -- quote a block jackknife, not a point estimate, here.
    assert abs(emp_mean) < 2.5 * emp_se
    assert np.abs(emp).max() > 0.05        # single draws genuinely scatter
    # A wrong gauge is still clearly biased under structure.
    assert asd_mean - 3.0 * asd_se > 0.05
    # The HWE residual under structure is consistent with zero at this fst
    # (0.5 SE), not a resolved positive offset; the old "+0.9% (SE 0.7%):
    # real" did not survive seeds.
    assert abs(paired_mean) < 4.0 * paired_se
    assert abs(paired_mean) < 0.01
