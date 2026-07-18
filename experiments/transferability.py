"""Can LD (D_A, D_B) be used to reweight a PGS onto more transferable variants?

A NEGATIVE result. The idea: down-weight variants whose LD neighbourhood is
discordant across ancestries (an A-placed tag "cashes out" in B only where D_B
agrees with D_A). We score each variant by LD-neighbourhood concordance
`s_j = cos(D_A row, D_B row)` (LD-only, no phenotype) and reweight by it.

It does not work:
- **LD-only** concordance shrinkage/pruning makes realized R²_B *worse* — the
  concordance score cannot tell a discordant *tag* (down-weight) from a discordant
  *causal* variant (keep) without the phenotype, so it removes signal.
- **Target-informed** efficiency reweighting shows *in-sample* gains that are
  overfitting: maximizing (wᵀz_B)²/(wᵀD_B w) over w has optimum w ∝ D_B⁻¹z_B — a
  ridge/BLUP refit in B (lassosum/LDpred2-in-B), not a reweighting of the score.
- The real headroom (naive → causal-oracle) requires *causal identification*
  (fine-mapping: PolyPred/PolyFun, PRS-CSx, SuSiEx), a construction task PPB does
  not and should not duplicate. See `docs/TRANSFERABILITY.md`.

Conclusion recorded so it is not re-litigated: PPB stays a measurement/diagnostic
tool; it does not reweight PGS for transferability.

Run:
    python experiments/transferability.py
"""

from __future__ import annotations

import argparse

import numpy as np

from ppb.simulate import (
    bn_freqs,
    marginal_stats,
    simulate_diploid_genotypes,
    simulate_phenotype,
)


def _blocks(m, bs):
    s, r = [], m
    while r > 0:
        s.append(min(bs, r)); r -= bs
    return s


def ld_concordance(DA, DB, block_sizes):
    """`s_j = cos(D_A row, D_B row)` over variant j's within-block LD (self excluded)."""
    m = DA.shape[0]; s = np.ones(m); lo = 0
    for k in block_sizes:
        idx = np.arange(lo, lo + k)
        for j in idx:
            nb = idx[idx != j]
            if nb.size == 0:
                continue
            a, b = DA[j, nb], DB[j, nb]
            na, nn = np.linalg.norm(a), np.linalg.norm(b)
            s[j] = (a @ b) / (na * nn) if na > 1e-12 and nn > 1e-12 else 1.0
        lo += k
    return s


def run(m=500, block_size=50, rho=0.5, fst=0.3, h2=0.5, n_causal=60,
        n=12000, n_ref=6000, rg=0.8, n_phenos=25, seed=1):
    rng = np.random.default_rng(seed); B = _blocks(m, block_size)
    fA, fB = bn_freqs(rng, m, fst)
    XAd = simulate_diploid_genotypes(n, B, fA, rho, rng)
    XBt = simulate_diploid_genotypes(n, B, fB, rho, rng)
    XBr = simulate_diploid_genotypes(n_ref, B, fB, rho, rng)
    XAr = simulate_diploid_genotypes(n_ref, B, fA, rho, rng)
    DA, DB = (XAr.T @ XAr) / n_ref, (XBr.T @ XBr) / n_ref
    tw = np.clip(ld_concordance(DA, DB, B), 0, 1)               # LD-only score

    out = {k: [] for k in ("naive", "shrink", "prune", "oracle", "ti_insample")}
    for _ in range(n_phenos):
        idx = rng.choice(m, n_causal, replace=False)
        bA = np.zeros(m); bA[idx] = rng.standard_normal(n_causal)
        bB = np.zeros(m)
        bB[idx] = rg * bA[idx] + np.sqrt(1 - rg ** 2) * rng.standard_normal(n_causal)
        w = marginal_stats(XAd, simulate_phenotype(XAd, bA, h2, rng))[0]
        yBt = simulate_phenotype(XBt, bB, h2, rng)
        R2 = lambda ww: float(np.corrcoef(XBt @ ww, yBt)[0, 1] ** 2)
        out["naive"].append(R2(w))
        out["shrink"].append(R2(w * tw ** 1.0))                # LD-only L2
        thr = np.quantile(tw, 0.30)
        out["prune"].append(R2(np.where(tw >= thr, w, 0.0)))   # LD-only L1
        out["oracle"].append(R2(bB))                           # causal upper bound
        zB_in = (XBt.T @ yBt) / n                              # IN-SAMPLE (overfits)
        eff = w * zB_in / (w * (DB @ w) + 1e-12)
        out["ti_insample"].append(R2(np.where(eff > np.quantile(eff, 0.30), w, 0.0)))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    print(f"{'r_g':>5}{'naive':>9}{'LD-shrink':>11}{'LD-prune':>10}"
          f"{'oracle':>9}{'target(in-sample)':>19}")
    for rg in (1.0, 0.8):
        r = run(rg=rg, seed=args.seed)
        d = lambda k: f"{r[k]:.3f} ({100*(r[k]-r['naive'])/r['naive']:+.0f}%)"
        print(f"{rg:>5}{r['naive']:>9.3f}{d('shrink'):>13}{d('prune'):>13}"
              f"{r['oracle']:>9.3f}{d('ti_insample'):>21}")


if __name__ == "__main__":
    main()
