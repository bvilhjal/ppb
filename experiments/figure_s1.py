"""Reproduce the LD-reference experiment (Supplementary Figure S1).

For many simulated phenotypes (h^2 in [0, 1]) and three PGS types, compare the
summary-statistic R^2 estimated with three LD references (test / train /
independent) against the individual-level R^2 on the test set. Prints a bias
table; optionally writes per-record CSV.

Run:
    python experiments/figure_s1.py                 # default 1000 phenotypes
    python experiments/figure_s1.py --n-phenos 200 --out fig_s1.csv
"""

from __future__ import annotations

import argparse

import numpy as np

from ppb import DenseLD, r2
from ppb.simulate import (
    draw_effects,
    marginal_stats,
    pgs_pthreshold,
    population_ld,
    sample_genotypes,
    simulate_phenotype,
)

PGS_TYPES = ("causal", "marginal", "pT")
LD_REFS = ("test", "train", "indep")


def run(m=300, block_size=30, rho=0.6, n=2000, n_causal=30,
        n_phenos=1000, t_crit=2.5, seed=0):
    rng = np.random.default_rng(seed)
    Sigma = population_ld(m, block_size, rho)
    X = {name: sample_genotypes(Sigma, n, rng) for name in ("train", "test", "indep")}
    # LD references depend only on genotypes -> build once.
    ld = {ref: DenseLD((X[ref].T @ X[ref]) / n) for ref in LD_REFS}

    records = []  # (pgs, ref, h2, true_r2, est_r2)
    for _ in range(n_phenos):
        h2 = rng.uniform(0.1, 0.9)
        beta = draw_effects(m, n_causal, rng)
        y_train = simulate_phenotype(X["train"], beta, h2, rng)
        y_test = simulate_phenotype(X["test"], beta, h2, rng)
        z_train, t_train = marginal_stats(X["train"], y_train)
        z_target = (X["test"].T @ y_test) / n           # released target sumstats
        pgs = {
            "causal": beta,
            "marginal": z_train,
            "pT": pgs_pthreshold(z_train, t_train, t_crit),
        }
        for name, w in pgs.items():
            if not np.any(w):
                continue
            true_r2 = float(np.corrcoef(X["test"] @ w, y_test)[0, 1] ** 2)
            for ref in LD_REFS:
                records.append((name, ref, h2, true_r2, r2(w, z_target, ld[ref])))
    return records


def summarize(records):
    print(f"{'PGS':<9}{'LD ref':<8}{'n':>5}{'mean %bias':>12}"
          f"{'median %bias':>14}{'corr(est,true)':>16}")
    print("-" * 64)
    for pgs in PGS_TYPES:
        for ref in LD_REFS:
            rows = [(t, e) for (p, r, _, t, e) in records if p == pgs and r == ref]
            if not rows:
                continue
            true = np.array([t for t, _ in rows])
            est = np.array([e for _, e in rows])
            ok = true > 1e-6
            pct = 100.0 * (est[ok] - true[ok]) / true[ok]
            corr = float(np.corrcoef(est, true)[0, 1]) if len(est) > 1 else float("nan")
            print(f"{pgs:<9}{ref:<8}{len(rows):>5}{pct.mean():>12.2f}"
                  f"{np.median(pct):>14.2f}{corr:>16.4f}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-phenos", type=int, default=1000)
    ap.add_argument("--m", type=int, default=300)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    records = run(m=args.m, n=args.n, n_phenos=args.n_phenos, seed=args.seed)
    summarize(records)
    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["pgs", "ld_ref", "h2", "true_r2", "est_r2"])
            wr.writerows(records)
        print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
