"""Benchmark PPB on simulated data (paper Figure 1 / Table 1 style).

Simulate realistic diploid genotypes (ldpred3-inspired), run several polygenic
score methods across polygenicity levels, and check that PPB's summary-statistic
prediction R^2 (using an independent LD reference, exact or LR8-approximated)
agrees with the individual-level R^2 -- both in absolute terms (bias) and in
ranking (Spearman), which is what lets PPB rank methods without individual data.

Methods (all self-contained, no external tools):
  causal   -- true causal effects (oracle)
  marginal -- marginal GWAS effects
  pT       -- p-value-thresholded marginal effects
  inf      -- LDpred-inf / ridge: (D_train + lambda I)^{-1} z_train

Run:
    python experiments/benchmark_methods.py --n-reps 20
"""

from __future__ import annotations

import argparse

import numpy as np

from ppb import DenseLD, lowrank_ld, r2
from ppb.simulate import (
    draw_effects,
    marginal_stats,
    pgs_pthreshold,
    simulate_diploid_genotypes,
    simulate_phenotype,
)

METHODS = ("causal", "marginal", "pT", "inf")


def _block_sizes(m, block_size):
    sizes, rem = [], m
    while rem > 0:
        sizes.append(min(block_size, rem))
        rem -= block_size
    return sizes


def ldpred_inf_weights(z_train, D_train, n_train, h2):
    """LDpred-infinitesimal / ridge weights: ``(D + (m / (h2 n)) I)^{-1} z``."""
    m = z_train.shape[0]
    lam = m / (h2 * n_train)
    return np.linalg.solve(D_train + lam * np.eye(m), z_train)


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run(m=400, block_size=40, rho=0.6, n=2500, h2=0.5, n_reps=20,
        t_crit=2.5, variances=(0.99, 0.95), seed=0):
    rng = np.random.default_rng(seed)
    block_sizes = _block_sizes(m, block_size)
    maf = rng.uniform(0.05, 0.5, size=m)

    Xtr = simulate_diploid_genotypes(n, block_sizes, maf, rho, rng)
    Xte = simulate_diploid_genotypes(n, block_sizes, maf, rho, rng)
    Xref = simulate_diploid_genotypes(n, block_sizes, maf, rho, rng)   # independent LD panel
    D_train = (Xtr.T @ Xtr) / n
    D_ref = (Xref.T @ Xref) / n
    ld = {"exact": DenseLD(D_ref)}
    for v in variances:
        ld[f"lr8@{v}"] = lowrank_ld(D_ref, variance=v)

    archs = {"sparse": max(5, m // 20), "medium": max(20, m // 4), "polygenic": m}

    records = []  # (arch, method, ld_kind, true_r2, ppb_r2)
    for arch, n_causal in archs.items():
        for _ in range(n_reps):
            beta = draw_effects(m, n_causal, rng)
            y_train = simulate_phenotype(Xtr, beta, h2, rng)
            y_test = simulate_phenotype(Xte, beta, h2, rng)
            z_train, t_train = marginal_stats(Xtr, y_train)
            z_target = (Xte.T @ y_test) / n
            weights = {
                "causal": beta,
                "marginal": z_train,
                "pT": pgs_pthreshold(z_train, t_train, t_crit),
                "inf": ldpred_inf_weights(z_train, D_train, n, h2),
            }
            for method, w in weights.items():
                if not np.any(w):
                    continue
                true_r2 = float(np.corrcoef(Xte @ w, y_test)[0, 1] ** 2)
                for kind, backend in ld.items():
                    records.append((arch, method, kind, true_r2, r2(w, z_target, backend)))
    return records


def summarize(records):
    kinds = sorted({k for _, _, k, _, _ in records}, key=lambda s: (s != "exact", s))
    print(f"{'LD reference':<12}{'n':>6}{'Pearson':>10}{'Spearman':>10}"
          f"{'mean %bias':>12}{'median %bias':>14}")
    print("-" * 64)
    for kind in kinds:
        rows = [(t, e) for (_, _, k, t, e) in records if k == kind]
        true = np.array([t for t, _ in rows]); est = np.array([e for _, e in rows])
        ok = true > 1e-6
        pct = 100.0 * (est[ok] - true[ok]) / true[ok]
        print(f"{kind:<12}{len(rows):>6}{np.corrcoef(est, true)[0, 1]:>10.4f}"
              f"{_spearman(est, true):>10.4f}{pct.mean():>12.2f}{np.median(pct):>14.2f}")

    print("\nMean individual-level vs PPB (exact) R^2 by method (ranking check):")
    for method in METHODS:
        rows = [(t, e) for (_, mth, k, t, e) in records if mth == method and k == "exact"]
        if rows:
            true = np.mean([t for t, _ in rows]); est = np.mean([e for _, e in rows])
            print(f"  {method:<9} true={true:.4f}  ppb={est:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=20)
    ap.add_argument("--m", type=int, default=400)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    records = run(m=args.m, n=args.n, n_reps=args.n_reps, seed=args.seed)
    summarize(records)


if __name__ == "__main__":
    main()
