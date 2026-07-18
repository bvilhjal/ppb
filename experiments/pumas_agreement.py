"""PUMAS (single-GWAS subsampling) agrees with PPB (held-out target).

Both estimate the same summary-statistic prediction R^2. PPB uses a genuinely
held-out target cohort's cross-products; PUMAS (Zhao et al. 2021) manufactures a
pseudo-validation set by subsampling one GWAS's summary statistics. This shows
the two, and the individual-level truth, agree across PGS methods -- so PUMAS is
the internal-cross-validation cousin of PPB's external benchmark.

Run:
    python experiments/pumas_agreement.py
"""

from __future__ import annotations

import argparse

import numpy as np

from ppb import DenseLD, pumas_r2, r2
from ppb.simulate import (
    draw_effects,
    marginal_stats,
    pgs_pthreshold,
    population_ld,
    sample_genotypes,
    simulate_phenotype,
)

METHODS = ("causal", "marginal", "pT")


def run(m=300, block_size=30, rho=0.6, n_train=3000, n_target=8000,
        h2=0.5, n_reps=40, frac_val=0.25, seed=0):
    rng = np.random.default_rng(seed)
    Sigma = population_ld(m, block_size, rho)
    Xtr = sample_genotypes(Sigma, n_train, rng)      # PGS is built here (external)
    Xtg = sample_genotypes(Sigma, n_target, rng)     # target GWAS PUMAS subsamples

    rows = []
    for arch, n_causal in (("sparse", 20), ("polygenic", m)):
        beta = draw_effects(m, n_causal, rng)
        ytr = simulate_phenotype(Xtr, beta, h2, rng)
        ytg = simulate_phenotype(Xtg, beta, h2, rng)
        z_tr, t_tr = marginal_stats(Xtr, ytr)
        z_tg = (Xtg.T @ ytg) / n_target
        D_tg = (Xtg.T @ Xtg) / n_target
        pgs = {"causal": beta, "marginal": z_tr,
               "pT": pgs_pthreshold(z_tr, t_tr, 2.5)}
        for name, w in pgs.items():
            if not np.any(w):
                continue
            gold = float(np.corrcoef(Xtg @ w, ytg)[0, 1] ** 2)     # individual-level
            ppb = r2(w, z_tg, DenseLD(D_tg))                        # PPB exact target
            pum = pumas_r2(w, z_tg, D_tg, n_target,                 # PUMAS single-GWAS
                           np.random.default_rng(seed + 1),
                           frac_val=frac_val, n_reps=n_reps)
            rows.append((arch, name, gold, ppb, pum))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = run(n_reps=args.n_reps, seed=args.seed)
    print(f"{'arch':<11}{'method':<10}{'individual':>12}{'PPB(exact)':>12}"
          f"{'PUMAS':>12}")
    print("-" * 57)
    for arch, name, gold, ppb, pum in rows:
        print(f"{arch:<11}{name:<10}{gold:>12.4f}{ppb:>12.4f}{pum:>12.4f}")


if __name__ == "__main__":
    main()
