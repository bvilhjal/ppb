"""Cross-ancestry R^2 estimation: PPB measures PRS portability from summary data.

The PPB estimator R^2 = (w^T z)^2 / (w^T D w) is ancestry-agnostic in *form*. To
estimate the predictive R^2 of an ancestry-A-trained PGS in a TARGET ancestry B,
feed it ancestry-B target summary statistics z_B AND an ancestry-B-matched LD
reference D_B. Nothing else changes.

Two Balding-Nichols ancestries (per-population allele frequencies and LD) with a
shared causal architecture correlated across populations at r_g. A p-value-
thresholded PGS is trained in ancestry A and evaluated in B. All genotypes are
within-cohort standardized, so w, z, D share one gauge and the identity is exact;
real-data deployment must instead put w, z_B, D_B on one self-consistent B gauge
(see docs/CROSS_ANCESTRY.md).

Shows: with B target stats + B LD the estimate is exact (test LD) or unbiased
(independent B LD) and recovers the portability loss R^2_B < R^2_A; using
ancestry-A LD biases it; and substituting ancestry-A sumstats does not estimate
R^2_B at all (target-ancestry data is irreducibly required).

Run:
    python experiments/cross_ancestry.py --rg 0.8
"""

from __future__ import annotations

import argparse

import numpy as np

from ppb import DenseLD, r2
from ppb.simulate import (
    bn_freqs,
    marginal_stats,
    pgs_pthreshold,
    simulate_diploid_genotypes,
    simulate_phenotype,
)

KEYS = ("R2A_true", "R2B_true", "exactB", "indepB", "mismatchA", "aOnly",
        "R2A_ppb", "ratioLD")


def _block_sizes(m, bs):
    s, rem = [], m
    while rem > 0:
        s.append(min(bs, rem)); rem -= bs
    return s


def run(m=500, block_size=50, rho=0.5, fst=0.25, h2=0.5, n_causal=50,
        n_disc=20000, n_test=20000, n_ref=10000, rg=0.8, n_phenos=60,
        t_crit=4.0, seed=1):
    rng = np.random.default_rng(seed)
    B = _block_sizes(m, block_size)
    fA, fB = bn_freqs(rng, m, fst)                     # per-ancestry allele freqs
    # Genotype cohorts are fixed; phenotypes/effects loop.
    XAd = simulate_diploid_genotypes(n_disc, B, fA, rho, rng)   # A discovery -> weights
    XAt = simulate_diploid_genotypes(n_test, B, fA, rho, rng)   # A test -> R2_A truth
    XBt = simulate_diploid_genotypes(n_test, B, fB, rho, rng)   # B test -> R2_B truth + z_B
    XBr = simulate_diploid_genotypes(n_ref, B, fB, rho, rng)    # independent B LD ref
    XAr = simulate_diploid_genotypes(n_ref, B, fA, rho, rng)    # ancestry-A LD (mismatched)
    D_Btest = DenseLD((XBt.T @ XBt) / n_test)
    D_Bref = DenseLD((XBr.T @ XBr) / n_ref)
    D_Aref = DenseLD((XAr.T @ XAr) / n_ref)
    D_Atest = DenseLD((XAt.T @ XAt) / n_test)

    out = {k: [] for k in KEYS}
    for _ in range(n_phenos):
        idx = rng.choice(m, n_causal, replace=False)
        bA = np.zeros(m); bA[idx] = rng.standard_normal(n_causal)
        bB = np.zeros(m)
        bB[idx] = rg * bA[idx] + np.sqrt(1 - rg ** 2) * rng.standard_normal(n_causal)
        zA, tA = marginal_stats(XAd, simulate_phenotype(XAd, bA, h2, rng))
        w = pgs_pthreshold(zA, tA, t_crit)             # ancestry-A-trained PGS
        if not np.any(w):
            continue
        yAt = simulate_phenotype(XAt, bA, h2, rng)
        yBt = simulate_phenotype(XBt, bB, h2, rng)
        R2A = float(np.corrcoef(XAt @ w, yAt)[0, 1] ** 2)
        R2B = float(np.corrcoef(XBt @ w, yBt)[0, 1] ** 2)
        zB = (XBt.T @ yBt) / n_test                    # target-ancestry-B sumstats
        zAt = (XAt.T @ yAt) / n_test
        out["R2A_true"].append(R2A)
        out["R2B_true"].append(R2B)
        out["exactB"].append(r2(w, zB, D_Btest))       # z_B + exact B LD
        out["indepB"].append(r2(w, zB, D_Bref))        # z_B + independent B LD (realistic)
        out["mismatchA"].append(r2(w, zB, D_Aref))     # z_B + WRONG-ancestry LD
        out["aOnly"].append(r2(w, zAt, D_Aref))        # A-only -> estimates R2_A, not R2_B
        out["R2A_ppb"].append(r2(w, zAt, D_Atest))
        out["ratioLD"].append(D_Aref.quad(w) / D_Bref.quad(w))
    return {k: np.array(v) for k, v in out.items()}


def pct_bias(est, true):
    ok = true > 1e-6
    return float(100.0 * np.mean((est[ok] - true[ok]) / true[ok]))


def summarize(rec, rg):
    tb, ta = rec["R2B_true"], rec["R2A_true"]
    print(f"\n=== r_g = {rg}  (n = {len(tb)} draws) ===")
    print(f"  mean R2_A = {ta.mean():.4f}   mean R2_B = {tb.mean():.4f}   "
          f"portability R2_B/R2_A = {tb.mean() / ta.mean():.3f}")
    print(f"  exact-B     (z_B, D_Btest)  %bias vs R2_B = {pct_bias(rec['exactB'], tb):+.3f}")
    print(f"  independent (z_B, D_Bref)   %bias vs R2_B = {pct_bias(rec['indepB'], tb):+.3f}")
    print(f"  mismatch-A  (z_B, D_Aref)   %bias vs R2_B = {pct_bias(rec['mismatchA'], tb):+.3f}")
    print(f"  A-only      (z_A, D_Aref)   %bias vs R2_B = {pct_bias(rec['aOnly'], tb):+.3f}"
          f"   (tracks R2_A: {pct_bias(rec['aOnly'], ta):+.2f}% vs R2_A)")
    print(f"  LD-form ratio w^T D_A w / w^T D_B w        = {rec['ratioLD'].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rg", type=float, default=None, help="run a single r_g (default: 1.0 and 0.8)")
    ap.add_argument("--n-phenos", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    for rg in ([args.rg] if args.rg is not None else [1.0, 0.8]):
        summarize(run(rg=rg, n_phenos=args.n_phenos, seed=args.seed), rg)


if __name__ == "__main__":
    main()
