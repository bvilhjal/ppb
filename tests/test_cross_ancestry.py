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
    # r_g = 1 with a shared LD architecture: portability is ~1. Balding-Nichols
    # gives the two ancestries different allele frequencies over the *same*
    # latent haplotype correlation, so F_ST has no LD channel here and there is
    # almost nothing left to lose. That is a property of the generator, not of
    # real ancestries -- see test_ld_divergence_dominates_the_mismatch_bias.
    assert tb.mean() / ta.mean() > 0.9, f"control portability {tb.mean()/ta.mean()}"
    assert abs(pct_bias(r["exactB"], tb)) < 0.05


def test_ld_divergence_dominates_the_mismatch_bias():
    """Both halves of the wrong-ancestry-LD claim, at r_g = 1 so LD is the only
    channel: with a shared LD architecture the bias is small and portability is
    ~1; give ancestry B its own LD decay and both move by an order of magnitude.

    This is why the headline -3.0% in docs/CROSS_ANCESTRY.md must be read as
    conditional on the simulated LD rather than as the size of the failure mode.
    """
    shared = run(**{**CFG, "n_phenos": 12}, rg=1.0, seed=0)
    diverged = run(**{**CFG, "n_phenos": 12}, rg=1.0, seed=0, rho=0.9, rho_b=0.6)

    tb_s, ta_s = shared["R2B_true"], shared["R2A_true"]
    tb_d, ta_d = diverged["R2B_true"], diverged["R2A_true"]
    bias_s = pct_bias(shared["mismatchA"], tb_s)
    bias_d = pct_bias(diverged["mismatchA"], tb_d)

    # Shared LD: little to get wrong, and little portability to lose.
    assert abs(bias_s) < 10.0, f"shared-LD mismatch bias {bias_s}"
    assert tb_s.mean() / ta_s.mean() > 0.9, "shared-LD portability should be ~1"
    # Diverged LD: the mismatch bias is several times larger, and real
    # portability loss appears with r_g held at 1.
    assert bias_d < 3.0 * bias_s, f"diverged {bias_d} vs shared {bias_s}"
    assert diverged["ratioLD"].mean() > 2.0 * shared["ratioLD"].mean()
    assert tb_d.mean() / ta_d.mean() < 0.9, "LD divergence should cost portability"
    # The estimator itself stays exact throughout -- it is the LD *input* that
    # is wrong in the mismatch column, not the identity.
    assert abs(pct_bias(diverged["exactB"], tb_d)) < 0.05
