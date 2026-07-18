"""Per-variant sample sizes: using a uniform N biases the estimated R^2.

In a GWAS meta-analysis each variant has its own sample size ``n_j``. The
standardized marginal correlation is recovered per variant via
``r_j = t_j / sqrt(t_j^2 + n_j - 2)`` (``ppb.standardized_marginal``). Plugging a
single uniform ``N`` instead shrinks every low-``n`` variant and biases R^2
downward; the per-variant conversion recovers the individual-level truth.

Run:
    python experiments/per_variant_n.py
"""

from __future__ import annotations

import argparse

import numpy as np

from ppb import DenseLD, r2, standardized_marginal
from ppb.simulate import (
    draw_effects,
    marginal_stats,
    population_ld,
    sample_genotypes,
    simulate_phenotype,
)


def gwas_hetero(X, y, n_j, rng):
    """Per-variant marginal regression: variant ``j`` uses ``n_j[j]`` individuals."""
    N, m = X.shape
    beta = np.empty(m); se = np.empty(m)
    for j in range(m):
        idx = rng.choice(N, size=int(n_j[j]), replace=False)
        x = X[idx, j] - X[idx, j].mean()
        yy = y[idx] - y[idx].mean()
        b = (x @ yy) / (x @ x)
        resid = yy - b * x
        beta[j] = b
        se[j] = np.sqrt((resid @ resid) / (idx.size - 2) / (x @ x))
    return beta, se


def run(m=300, block_size=30, rho=0.6, n=8000, n_train=3000, h2=0.5,
        fracs=(1.0, 0.5, 0.25), seed=0):
    rng = np.random.default_rng(seed)
    Sigma = population_ld(m, block_size, rho)
    Xtr = sample_genotypes(Sigma, n_train, rng)
    X = sample_genotypes(Sigma, n, rng)
    beta = draw_effects(m, m, rng)
    ytr = simulate_phenotype(Xtr, beta, h2, rng)
    y = simulate_phenotype(X, beta, h2, rng)
    w = marginal_stats(Xtr, ytr)[0]                    # external predictive PGS
    D = DenseLD((X.T @ X) / n)
    gold = float(np.corrcoef(X @ w, y)[0, 1] ** 2)

    rows = []
    for frac in fracs:
        n_j = (np.full(m, n) if frac >= 1.0
               else rng.integers(int(frac * n), n + 1, size=m))
        b, se = gwas_hetero(X, y, n_j, rng)
        per_variant = r2(w, standardized_marginal(b, se, n_j), D)
        uniform = r2(w, standardized_marginal(b, se, n), D)
        rows.append((frac, gold, per_variant, uniform))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = run(seed=args.seed)
    print(f"{'n_j range':<16}{'individual':>12}{'per-variant n':>15}{'uniform N':>12}")
    print("-" * 55)
    for frac, gold, pv, uni in rows:
        label = "all = N" if frac >= 1.0 else f"[{frac:.2f}N, N]"
        print(f"{label:<16}{gold:>12.4f}{pv:>15.4f}{uni:>12.4f}")


if __name__ == "__main__":
    main()
