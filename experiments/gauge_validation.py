"""Gauge-path validation: dosage weights x target genotype SD against truth.

The cross-ancestry demonstrations in ``cross_ancestry.py`` simulate everything
within-cohort standardized, so ``w``, ``z_B`` and ``D_B`` already share one
gauge and the dosage-to-standardized conversion that every real evaluation
performs (``ppb.evaluate(weight_scale="dosage")``, step V3 / (X2)) is never
exercised against individual-level truth. This experiment closes that gap.

Setup. Two Balding-Nichols ancestries ``fst`` apart share a block-AR(1) LD
generator; a p+T score is trained in A and submitted as ordinary per-allele
(dosage) weights ``b_j = w_A,j / sd_A,j`` -- exactly how a PGS Catalog file
arrives. The B-side evaluation uses only summary-level inputs: an independent
B LD panel, B test-cohort marginal statistics ``z_B``, and per-variant
empirical B genotype SDs. Truth is ``corr(X_Bstd w_B, y_B)^2`` with the correct
B gauge ``w_B = b * sd_B``, computed on the individuals.

Three gauges are compared:

1. **empirical B SD** -- the (X2) rule. Must recover truth.
2. **HWE SD** ``sqrt(2 f_B (1-f_B))`` -- the documented approximation; close
   for a homogeneous, near-HWE B, and this is quantified, not assumed.
3. **A-cohort SD** -- the wrong-ancestry gauge; quantifies what a frozen-scale
   mixup costs.

With ``sub_fst > 0`` B is instead a 50/50 pool of two subpopulations that far
apart, so the HWE SD at the pooled frequencies departs from the empirical
pooled SD -- the structured-target case the (X2) rule exists for.

Measured (seeds 0-4, defaults, 2026-08-16). Homogeneous B: the empirical-SD
evaluation recovers truth with mean error -0.8% (worst seed 3.7%, which is
independent-LD-panel noise, not gauge error); the HWE SD is within +0.03% of
the empirical one (paired, shared noise cancelled); the discovery SD is
biased +15% to +34% every seed. Admixed B at ``sub_fst = 0.1``: single-draw
error scatters to |14|% because the panel denominator dominates at these n --
quote a block jackknife in this regime -- while the paired HWE-vs-empirical
residual is +0.9% (SE 0.7%): real, second-order at this fst, and unbounded
only in the worst case. At ``sub_fst = 0.2`` the panel noise swamps
everything; the gauge comparison needs either a larger panel or paired
differences, not bigger structure.

Run:
    python experiments/gauge_validation.py --seed 0 [--sub-fst 0.1]
"""

from __future__ import annotations

import argparse

import numpy as np

from ppb import BlockDiagonalLD, DenseLD, VariantTable, evaluate
from ppb.simulate import (
    bn_freqs,
    draw_effects,
    marginal_stats,
    pgs_pthreshold,
    simulate_haplotypes,
    simulate_phenotype,
)


def _block_sizes(m, block_size):
    sizes, rem = [], m
    while rem > 0:
        sizes.append(min(block_size, rem)); rem -= block_size
    return sizes


def _dosages(n, block_sizes, maf, rho, rng):
    """Raw 0/1/2 dosages and their empirical per-variant SDs."""
    G = simulate_haplotypes(n, block_sizes, maf, rho, rng).sum(
        axis=1, dtype=np.float64)
    return G, G.std(axis=0)


def _admixed_dosages(n, block_sizes, maf1, maf2, rho, rng):
    """Dosages from a 50/50 pool of two populations ``fst_sub`` apart.

    The pooled cohort is out of Hardy-Weinberg: ``Var(g_j)`` exceeds
    ``2 f_j (1-f_j)`` at the pooled frequency by a per-variant factor the HWE
    formula cannot see. This is the structured-target case (X2) warns about.
    """
    half = n // 2
    G1 = simulate_haplotypes(half, block_sizes, maf1, rho, rng).sum(
        axis=1, dtype=np.float64)
    G2 = simulate_haplotypes(n - half, block_sizes, maf2, rho, rng).sum(
        axis=1, dtype=np.float64)
    G = np.vstack([G1, G2])
    return G, G.std(axis=0)


def _variants(m, rng):
    """Non-palindromic variant table, chrom 1, positions 1..m, random alleles."""
    pairs = [("A", "C"), ("A", "G"), ("T", "C"), ("T", "G")]
    idx = rng.integers(0, len(pairs), size=m)
    a1 = np.array([pairs[i][0] for i in idx])
    a2 = np.array([pairs[i][1] for i in idx])
    return VariantTable(chrom=np.full(m, "1"), pos=np.arange(1, m + 1),
                        a1=a1, a2=a2)


def run(m=400, block_size=40, n_train=10000, n_test=15000, n_panel=6000,
        fst=0.15, rho=0.5, h2=0.5, n_causal=80, t_crit=2.0, seed=0,
        sub_fst=0.0):
    """One draw: train dosage weights in A, evaluate in B three ways.

    ``sub_fst > 0`` makes B a 50/50 pool of two subpopulations that far
    apart, so the HWE SD at the pooled frequencies departs from the empirical
    pooled SD -- the structured-target case the (X2) rule exists for.
    """
    rng = np.random.default_rng(seed)
    bs = _block_sizes(m, block_size)
    f_A, f_B = bn_freqs(rng, m, fst)
    if sub_fst > 0.0:
        f_B1, f_B2 = bn_freqs(rng, m, sub_fst)
        f_B = 0.5 * (f_B1 + f_B2)          # pooled frequencies for HWE

    G_A, sd_A = _dosages(n_train, bs, f_A, rho, rng)
    if sub_fst > 0.0:
        G_B, sd_B = _admixed_dosages(n_test, bs, f_B1, f_B2, rho, rng)
    else:
        G_B, sd_B = _dosages(n_test, bs, f_B, rho, rng)
    if sub_fst > 0.0:
        # In-sample B LD is mandatory for an admixed target (hard requirement
        # 2): the panel must carry the same structure as the z_B cohort.
        G_P, _ = _admixed_dosages(n_panel, bs, f_B1, f_B2, rho, rng)
    else:
        G_P, _ = _dosages(n_panel, bs, f_B, rho, rng)

    # A variant monomorphic anywhere has no standardized scale there; drop the
    # joint set (the package rule, docs/CROSS_ANCESTRY.md requirement 3).
    keep = (sd_A > 0) & (sd_B > 0) & (G_P.std(axis=0) > 0)
    if not keep.any():
        raise ValueError("every variant monomorphic in a cohort; rerun")
    G_A, G_B, G_P, f_B_kept = G_A[:, keep], G_B[:, keep], G_P[:, keep], f_B[keep]
    sd_A, sd_B = sd_A[keep], sd_B[keep]
    m_kept = int(keep.sum())

    def _std(G, sd):
        return (G - G.mean(axis=0)) / np.where(sd > 0, sd, 1.0)

    X_A, X_B, X_P = _std(G_A, sd_A), _std(G_B, sd_B), _std(G_P, G_P.std(axis=0))

    beta = draw_effects(m_kept, n_causal, rng)
    y_A = simulate_phenotype(X_A, beta, h2, rng)
    y_B = simulate_phenotype(X_B, beta, h2, rng)

    # Train in A on the standardized scale; ship per-allele dosage weights.
    w_std_A = pgs_pthreshold(*marginal_stats(X_A, y_A), t_crit=t_crit)
    b_dosage = w_std_A / sd_A

    # B-side summary inputs.
    z_B = (X_B.T @ y_B) / n_test
    D_P = (X_P.T @ X_P) / n_panel
    # The generator's blocks tile [0, m); re-tile over the kept variants so the
    # block boundaries survive the monomorphic drop.
    blocks, start = [], 0
    for size in bs:
        stop = min(start + size, m_kept)
        if stop > start:
            blocks.append((DenseLD(D_P[start:stop, start:stop]),
                           np.arange(start, stop)))
        start = stop
    ld = BlockDiagonalLD(blocks)

    ref = _variants(m_kept, rng)
    order = rng.permutation(m_kept)               # submission arrives shuffled
    sub = VariantTable(chrom=ref.chrom[order], pos=ref.pos[order],
                       a1=ref.a1[order], a2=ref.a2[order])

    def _eval(sds, frame="reference"):
        return evaluate(ld, ref, sub, b_dosage[order], ref, z_B,
                        weight_scale="dosage", genotype_sd=sds,
                        genotype_sd_frame=frame).r2

    r2_emp = _eval(sd_B)
    r2_sumstats_frame = _eval(sd_B, frame="sumstats")
    r2_hwe = _eval(np.sqrt(2.0 * f_B_kept * (1.0 - f_B_kept)))
    r2_A_sd = _eval(sd_A)

    w_B = b_dosage * sd_B                          # the correct B gauge
    truth = float(np.corrcoef(X_B @ w_B, y_B)[0, 1] ** 2)

    return {
        "truth": truth,
        "r2_empirical_sd": r2_emp,
        "r2_sumstats_frame": r2_sumstats_frame,
        "r2_hwe_sd": r2_hwe,
        "r2_discovery_sd": r2_A_sd,
        "n_monomorphic_dropped": int((~keep).sum()),
        "n_scored": int(np.count_nonzero(w_std_A)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sub-fst", type=float, default=0.0,
                    help=">0 evaluates against an admixed (50/50 pooled) target")
    args = ap.parse_args()
    res = run(seed=args.seed, sub_fst=args.sub_fst)
    print(f"truth (individual-level)      {res['truth']:.4f}")
    print(f"empirical B SD (X2 rule)      {res['r2_empirical_sd']:.4f}")
    print(f"sumstats-frame SD path        {res['r2_sumstats_frame']:.4f}")
    print(f"HWE SD at B frequencies       {res['r2_hwe_sd']:.4f}")
    print(f"discovery (A) SD -- wrong     {res['r2_discovery_sd']:.4f}")
    print(f"monomorphic dropped: {res['n_monomorphic_dropped']}; "
          f"nonzero weights: {res['n_scored']}")


if __name__ == "__main__":
    main()
