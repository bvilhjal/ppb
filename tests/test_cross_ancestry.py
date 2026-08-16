"""Cross-ancestry R^2 estimation: PPB measures PRS portability from summary data.

Encodes the verified claims: with target-ancestry (B) summary statistics AND a
B-matched LD reference the estimate is exact/unbiased and recovers the
portability loss; using discovery-ancestry (A) LD biases it via the LD-form
ratio; and substituting A summary statistics does not estimate R^2_B at all.

Every Monte Carlo quantity is replicated over a fixed seed list and asserted as
a 3-SE bound on the cross-seed mean (review F3), not as a single-seed snapshot.
"""

import numpy as np

from experiments.cross_ancestry import pct_bias, run  # noqa: E402

# Dense marginal weights (t_crit=0) make the LD-form mismatch robust at test scale;
# the estimator itself is identical for any weights.
CFG = dict(m=400, block_size=40, fst=0.35, n_disc=7000, n_test=7000, n_ref=3500,
           n_causal=50, n_phenos=20, t_crit=0.0)

# Fixed replication list: the cross-seed mean and SE replace single-seed pins.
SEEDS = (0, 1, 2, 3, 4)


def _mean_se(values):
    """Cross-seed mean and standard error of a per-seed statistic."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def test_cross_ancestry_estimator_and_failure_modes():
    runs = [run(**CFG, rg=0.8, seed=s) for s in SEEDS]

    def ms(fn):
        return _mean_se([fn(r) for r in runs])

    exact_m, exact_se = ms(lambda r: pct_bias(r["exactB"], r["R2B_true"]))
    indep_m, indep_se = ms(lambda r: pct_bias(r["indepB"], r["R2B_true"]))
    a_only_m, a_only_se = ms(lambda r: pct_bias(r["aOnly"], r["R2B_true"]))
    port_m, port_se = ms(lambda r: r["R2B_true"].mean() / r["R2A_true"].mean())
    ld_ratio_m, ld_ratio_se = ms(lambda r: r["ratioLD"].mean())
    rank_m, rank_se = ms(
        lambda r: float(np.corrcoef(r["indepB"], r["R2B_true"])[0, 1]))
    # Per-seed gap between the wrong-ancestry-LD bias and the independent-B
    # bias: the quantity the mismatch claim is actually about.
    gap_m, gap_se = ms(lambda r: pct_bias(r["indepB"], r["R2B_true"])
                       - pct_bias(r["mismatchA"], r["R2B_true"]))

    # (1) Exact: target sumstats z_B + target test LD reproduce individual-level R^2_B.
    assert abs(exact_m) + 3 * exact_se < 0.05, (
        f"exact %bias {exact_m:+.4f} ± {3 * exact_se:.4f}")
    # (2) Unbiased with an independent target-ancestry-B LD reference (realistic use).
    assert abs(indep_m) + 3 * indep_se < 3.0, (
        f"independent-B %bias {indep_m:+.3f} ± {3 * indep_se:.3f}")
    # (3) LD reference must match target ancestry: wrong-ancestry LD biases the
    #     denominator by the LD-form ratio w^T D_A w / w^T D_B w, which sits
    #     clearly above 1 (> 3 SE), and the resulting mismatch bias is clearly
    #     worse than the independent-B reference (per-seed gap > 3 SE above 0).
    assert ld_ratio_m - 3 * ld_ratio_se > 1.0, (
        f"LD-form ratio {ld_ratio_m:.4f} ± {3 * ld_ratio_se:.4f}")
    assert gap_m - 3 * gap_se > 0.0, (
        f"mismatch-vs-indep gap {gap_m:+.3f} ± {3 * gap_se:.3f}")
    # (4) Impossibility: substituting discovery-ancestry sumstats z_A overstates R^2_B.
    assert a_only_m - 3 * a_only_se > 30.0, (
        f"A-only overstatement {a_only_m:+.2f} ± {3 * a_only_se:.2f}")
    # (5) Portability loss recovered at r_g = 0.8.
    assert port_m + 3 * port_se < 0.75, (
        f"portability R2_B/R2_A {port_m:.3f} ± {3 * port_se:.3f}")
    # (6) Ranking of scores preserved across draws.
    assert rank_m - 3 * rank_se > 0.9, (
        f"rank corr {rank_m:.4f} ± {3 * rank_se:.4f}")


def test_cross_ancestry_portability_control_rg1():
    runs = [run(**{**CFG, "n_phenos": 12}, rg=1.0, seed=s) for s in SEEDS]
    port_m, port_se = _mean_se(
        [r["R2B_true"].mean() / r["R2A_true"].mean() for r in runs])
    exact_m, exact_se = _mean_se(
        [pct_bias(r["exactB"], r["R2B_true"]) for r in runs])
    # r_g = 1 with a shared LD architecture: portability is ~1. Balding-Nichols
    # gives the two ancestries different allele frequencies over the *same*
    # latent haplotype correlation, so F_ST has no LD channel here and there is
    # almost nothing left to lose. That is a property of the generator, not of
    # real ancestries -- see test_ld_divergence_dominates_the_mismatch_bias.
    assert port_m - 3 * port_se > 0.9, (
        f"control portability {port_m:.4f} ± {3 * port_se:.4f}")
    assert abs(exact_m) + 3 * exact_se < 0.05, (
        f"control exact %bias {exact_m:+.5f} ± {3 * exact_se:.5f}")


def test_ld_divergence_dominates_the_mismatch_bias():
    """Both halves of the wrong-ancestry-LD claim, at r_g = 1 so LD is the only
    channel: with a shared LD architecture the bias is small and portability is
    ~1; give ancestry B its own LD decay and both move by an order of magnitude.

    This is why the headline -3.0% in docs/CROSS_ANCESTRY.md must be read as
    conditional on the simulated LD rather than as the size of the failure mode.
    """
    pairs = [
        (run(**{**CFG, "n_phenos": 12}, rg=1.0, seed=s),
         run(**{**CFG, "n_phenos": 12}, rg=1.0, seed=s, rho=0.9, rho_b=0.6))
        for s in SEEDS
    ]

    def ms(fn):
        return _mean_se([fn(shared, diverged) for shared, diverged in pairs])

    bias_s_m, bias_s_se = ms(
        lambda s, d: pct_bias(s["mismatchA"], s["R2B_true"]))
    excess_m, excess_se = ms(
        lambda s, d: pct_bias(d["mismatchA"], d["R2B_true"])
        - 3.0 * pct_bias(s["mismatchA"], s["R2B_true"]))
    ld_ratio_m, ld_ratio_se = ms(
        lambda s, d: d["ratioLD"].mean() / s["ratioLD"].mean())
    port_s_m, port_s_se = ms(
        lambda s, d: s["R2B_true"].mean() / s["R2A_true"].mean())
    port_d_m, port_d_se = ms(
        lambda s, d: d["R2B_true"].mean() / d["R2A_true"].mean())
    exact_d_m, exact_d_se = ms(
        lambda s, d: pct_bias(d["exactB"], d["R2B_true"]))

    # Shared LD: little to get wrong, and little portability to lose.
    assert abs(bias_s_m) + 3 * bias_s_se < 10.0, (
        f"shared-LD mismatch bias {bias_s_m:+.3f} ± {3 * bias_s_se:.3f}")
    assert port_s_m - 3 * port_s_se > 0.9, (
        f"shared-LD portability {port_s_m:.4f} ± {3 * port_s_se:.4f}")
    # Diverged LD: the mismatch bias is several times larger (its per-seed
    # excess over 3x the shared bias stays below 0 by 3 SE), and real
    # portability loss appears with r_g held at 1.
    assert excess_m + 3 * excess_se < 0.0, (
        f"diverged excess over 3x shared bias {excess_m:+.2f} ± {3 * excess_se:.2f}")
    assert ld_ratio_m - 3 * ld_ratio_se > 2.0, (
        f"diverged/shared LD-form ratio {ld_ratio_m:.3f} ± {3 * ld_ratio_se:.3f}")
    assert port_d_m + 3 * port_d_se < 0.9, (
        f"diverged-LD portability {port_d_m:.4f} ± {3 * port_d_se:.4f}")
    # The estimator itself stays exact throughout -- it is the LD *input* that
    # is wrong in the mismatch column, not the identity.
    assert abs(exact_d_m) + 3 * exact_d_se < 0.05, (
        f"diverged exact %bias {exact_d_m:+.5f} ± {3 * exact_d_se:.5f}")
