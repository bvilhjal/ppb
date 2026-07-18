"""PC adjustment removes population-structure confounding in PPB evaluation.

Two subpopulations (Balding-Nichols fst) create population structure. A
phenotype driven only by ancestry (no true genetic effect) is spuriously
"predicted" by a polygenic score built on the structured data -- classic
stratification. Residualizing genotypes and phenotype on the top principal
components before forming the target summary statistics ``z`` and LD ``D``
removes the confounding, so the estimated R^2 drops to ~0. Under a genuine
genetic signal with no confounding, adjustment is nearly harmless.

The estimator is unchanged; it just consumes the PC-adjusted ``z`` and ``D``.
Averaged over replicates, because a single 2-population ancestry axis is low-rank
and the per-draw spurious R^2 is noisy.

Run:
    python experiments/pc_adjustment.py --n-reps 10
"""

from __future__ import annotations

import argparse

import numpy as np

from ppb import DenseLD, adjust, principal_components, r2
from ppb.simulate import (
    draw_effects,
    marginal_stats,
    simulate_phenotype,
    simulate_structured_genotypes,
)


def _block_sizes(m, block_size):
    sizes, rem = [], m
    while rem > 0:
        sizes.append(min(block_size, rem)); rem -= block_size
    return sizes


def _ppb_r2(X, y, w, n_pcs):
    """PPB R^2 without and with PC adjustment (test-set LD, so exact)."""
    n = X.shape[0]
    unadj = r2(w, (X.T @ y) / n, DenseLD((X.T @ X) / n))
    Xa, ya = adjust(X, y, principal_components(X, n_pcs))
    adj = r2(w, (Xa.T @ ya) / n, DenseLD((Xa.T @ Xa) / n))
    return unadj, adj


def run(m=400, block_size=40, fst=0.3, rho=0.5, n=3000, n_causal=40,
        confound=4.0, n_pcs=2, n_reps=10, seed=0):
    rng = np.random.default_rng(seed)
    bs = _block_sizes(m, block_size)
    rows = {"null+confound": ([], []), "genetic,no confound": ([], [])}

    for _ in range(n_reps):
        Xtr, ltr = simulate_structured_genotypes(n, bs, fst, rho, rng)
        Xte, lte = simulate_structured_genotypes(n, bs, fst, rho, rng)

        # (A) null genetic signal + ancestry confounder -> spurious prediction.
        def confounded(X, labels):
            y = rng.standard_normal(X.shape[0]) + confound * (labels - labels.mean())
            return (y - y.mean()) / y.std()
        w = marginal_stats(Xtr, confounded(Xtr, ltr))[0]
        u, a = _ppb_r2(Xte, confounded(Xte, lte), w, n_pcs)
        rows["null+confound"][0].append(u); rows["null+confound"][1].append(a)

        # (B) true genetic signal, no confounding -> adjustment ~harmless.
        beta = draw_effects(m, n_causal, rng)
        w2 = marginal_stats(Xtr, simulate_phenotype(Xtr, beta, 0.5, rng))[0]
        u2, a2 = _ppb_r2(Xte, simulate_phenotype(Xte, beta, 0.5, rng), w2, n_pcs)
        rows["genetic,no confound"][0].append(u2); rows["genetic,no confound"][1].append(a2)

    return {name: (float(np.mean(u)), float(np.mean(a))) for name, (u, a) in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = run(n_reps=args.n_reps, seed=args.seed)
    print(f"{'scenario':<22}{'R2 unadjusted':>16}{'R2 PC-adjusted':>16}")
    print("-" * 54)
    for name, (unadj, adj) in res.items():
        print(f"{name:<22}{unadj:>16.4f}{adj:>16.4f}")


if __name__ == "__main__":
    main()
