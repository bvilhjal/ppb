"""Cross-ancestry R^2 estimation: PPB measures PRS portability from summary data.

Encodes the verified claims: with target-ancestry (B) summary statistics AND a
B-matched LD reference the estimate is exact/unbiased and recovers the
portability loss; using discovery-ancestry (A) LD biases it via the LD-form
ratio; and substituting A summary statistics does not estimate R^2_B at all.
"""

import numpy as np

from experiments.cross_ancestry import pct_bias, run  # noqa: E402

# Dense marginal weights (t_crit=0) make the LD-form mismatch robust at test scale;
# the estimator itself is identical for any weights.
CFG = dict(m=400, block_size=40, fst=0.35, n_disc=7000, n_test=7000, n_ref=3500,
           n_causal=50, n_phenos=20, t_crit=0.0)


def test_cross_ancestry_estimator_and_failure_modes():
    r = run(**CFG, rg=0.8, seed=0)
    tb, ta = r["R2B_true"], r["R2A_true"]
    exact = pct_bias(r["exactB"], tb)
    indep = pct_bias(r["indepB"], tb)
    mismatch = pct_bias(r["mismatchA"], tb)
    a_only = pct_bias(r["aOnly"], tb)
    portability = tb.mean() / ta.mean()
    ld_ratio = r["ratioLD"].mean()
    rank = float(np.corrcoef(r["indepB"], tb)[0, 1])

    # (1) Exact: target sumstats z_B + target test LD reproduce individual-level R^2_B.
    assert abs(exact) < 0.05, f"exact %bias {exact}"
    # (2) Unbiased with an independent target-ancestry-B LD reference (realistic use).
    assert abs(indep) < 3.0, f"independent-B %bias {indep}"
    # (3) LD reference must match target ancestry: wrong-ancestry LD biases the
    #     denominator by the LD-form ratio w^T D_A w / w^T D_B w (> 1 here).
    assert ld_ratio > 1.02, f"LD-form ratio {ld_ratio}"
    assert mismatch < indep - 1.0, f"mismatch {mismatch} not clearly worse than indep {indep}"
    # (4) Impossibility: substituting discovery-ancestry sumstats z_A overstates R^2_B.
    assert a_only > 30.0, f"A-only overstatement {a_only}"
    # (5) Portability loss recovered at r_g = 0.8.
    assert portability < 0.75, f"portability R2_B/R2_A {portability}"
    # (6) Ranking of scores preserved across draws.
    assert rank > 0.9, f"rank corr {rank}"


def test_cross_ancestry_portability_control_rg1():
    r = run(**{**CFG, "n_phenos": 12}, rg=1.0, seed=0)
    tb, ta = r["R2B_true"], r["R2A_true"]
    # r_g = 1: portability is high (loss only from MAF/LD), and matched PPB stays exact.
    assert tb.mean() / ta.mean() > 0.9, f"control portability {tb.mean()/ta.mean()}"
    assert abs(pct_bias(r["exactB"], tb)) < 0.05
