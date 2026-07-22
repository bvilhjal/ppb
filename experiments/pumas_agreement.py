"""PUMAS-style summary subsampling matches individual repeated learning.

For the individual-level reference, each score is trained on a random subset
and evaluated on its held-out complement.  PPB evaluates exactly the same
weights and complements from their cross-products.  The PUMAS-style path sees
only the full-sample summary statistics, rebuilds each data-trained score from
every pseudo-training split, and evaluates the paired pseudo-validation split.
The causal score is an explicitly independent oracle.

Run:
    python experiments/pumas_agreement.py
"""

from __future__ import annotations

import argparse

import numpy as np

from ppb import DenseLD, pumas_r2, r2
from ppb.simulate import (
    draw_effects,
    pgs_pthreshold,
    population_ld,
    sample_genotypes,
    simulate_phenotype,
)

METHODS = ("causal", "marginal", "pT")


def run(m=300, block_size=30, rho=0.6, n_full=3000, h2=0.5,
        n_reps=40, frac_val=0.25, seed=0):
    rng = np.random.default_rng(seed)
    Sigma = population_ld(m, block_size, rho)
    X = sample_genotypes(Sigma, n_full, rng)
    n_train = int(round(n_full * (1.0 - frac_val)))

    rows = []
    for arch_i, (arch, n_causal) in enumerate(
            (("sparse", 20), ("polygenic", m))):
        beta = draw_effects(m, n_causal, rng)
        y = simulate_phenotype(X, beta, h2, rng)
        z_full = (X.T @ y) / n_full
        D_full = (X.T @ X) / n_full

        def fit_marginal(z_train):
            return z_train

        def fit_pt(z_train):
            se = np.sqrt(np.clip(1.0 - z_train ** 2, 1e-12, None) / n_train)
            return pgs_pthreshold(z_train, z_train / se, 2.5)

        fits = {"causal": None, "marginal": fit_marginal, "pT": fit_pt}
        split_rng = np.random.default_rng(seed + 1000 + arch_i)
        splits = []
        for _ in range(n_reps):
            train_idx = split_rng.choice(n_full, n_train, replace=False)
            is_validation = np.ones(n_full, dtype=bool)
            is_validation[train_idx] = False
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[is_validation], y[is_validation]
            z_train = (X_train.T @ y_train) / n_train
            X_val_centered = X_val - X_val.mean(axis=0)
            y_val_centered = y_val - y_val.mean()
            z_val = (X_val_centered.T @ y_val_centered) / X_val.shape[0]
            D_val = (X_val_centered.T @ X_val_centered) / X_val.shape[0]
            var_y_val = float(np.mean(y_val_centered ** 2))
            splits.append(
                (z_train, z_val, D_val, var_y_val, X_val, y_val))

        for method_i, name in enumerate(METHODS):
            individual = []
            summary = []
            for z_train, z_val, D_val, var_y_val, X_val, y_val in splits:
                w = beta if name == "causal" else fits[name](z_train)
                prediction = X_val @ w
                individual.append(float(np.corrcoef(prediction, y_val)[0, 1] ** 2))
                summary.append(r2(w, z_val, DenseLD(D_val), var_y=var_y_val))
            gold = float(np.mean(individual))
            ppb = float(np.mean(summary))
            score_source = ({"independent_weights": beta}
                            if name == "causal" else {"fit": fits[name]})
            pum = pumas_r2(
                z_full,
                D_full,
                n_full,
                np.random.default_rng(seed + 100 * arch_i + method_i + 1),
                frac_val=frac_val,
                n_reps=n_reps,
                **score_source,
            )
            rows.append((arch, name, gold, ppb, pum))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = run(n_reps=args.n_reps, seed=args.seed)
    print(f"{'arch':<11}{'method':<10}{'individual':>12}{'PPB(exact)':>12}"
          f"{'PUMAS(CV)':>12}")
    print("-" * 57)
    for arch, name, gold, ppb, pum in rows:
        print(f"{arch:<11}{name:<10}{gold:>12.4f}{ppb:>12.4f}{pum:>12.4f}")


if __name__ == "__main__":
    main()
