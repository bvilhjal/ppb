"""Overlap detection and correction: simulation validation.

A training GWAS (n1) and a target GWAS (n2) share a controlled fraction of
individuals; an independent GWAS (n2) is the honest anchor. For a dense
(marginal, ``w = z_train``) score the per-variant overlap term is exactly
``gamma_true = N_ov / (n1 n2)``, and the dual-target detector
(:func:`ppb.overlap.overlap_slope`) recovers it: gamma_hat ~ 0 under the null
and ~ gamma_true under overlap, while the corrected numerator
(:func:`ppb.overlap.correct_numerator`) restores the honest-anchor R^2 even
at full overlap. For a p-value-thresholded (sparse) score the detector is
blind by construction (gamma_hat ~ 0 at every overlap level) -- the known
failure mode; sparse evaluations are upper bounds, not correctable.

Run:
    python experiments/overlap_detection.py
"""

from __future__ import annotations

import numpy as np

from ppb import DenseLD
from ppb.overlap import correct_numerator, overlap_slope
from ppb.simulate import (draw_effects, marginal_stats, pgs_pthreshold,
                          population_ld, simulate_diploid_genotypes,
                          simulate_phenotype)

KEYS = ("gamma_null", "gamma_low", "gamma_low_true", "gamma_full",
        "gamma_full_true", "r2_naive_full", "r2_corr_full", "r2_honest_full",
        "gamma_sparse_full", "gamma_sparse_full_true")


def _one(rng, frac, sparse, n1, n2, n_ind, m_blocks, bs, rho, h2, n_causal):
    m = m_blocks * bs
    sigma = population_ld(m, block_size=bs, rho=rho)
    maf = rng.uniform(0.05, 0.5, size=m)
    x = simulate_diploid_genotypes(n1 + n2 + n_ind, [bs] * m_blocks, maf, rho, rng)
    beta = draw_effects(m, n_causal, rng)
    y = simulate_phenotype(x, beta, h2, rng)

    i_tr = np.arange(n1)
    n_ov = int(frac * n2)
    i_ta = np.concatenate([rng.choice(i_tr, n_ov, replace=False),
                           np.arange(n1, n1 + n2 - n_ov)])
    i_in = np.arange(n1 + n2, n1 + n2 + n_ind)
    z_tr, t_tr = marginal_stats(x[i_tr], y[i_tr])
    z_ta, _ = marginal_stats(x[i_ta], y[i_ta])
    z_in, _ = marginal_stats(x[i_in], y[i_in])
    w = pgs_pthreshold(z_tr, t_tr, 2.5) if sparse else z_tr.copy()

    d = DenseLD(np.corrcoef(x.T))  # matched cohort LD -> exact estimator
    den = d.quad(w)
    num_naive, num_honest = w @ z_ta, w @ z_in

    u_ta = np.array([w[s:s + bs] @ z_ta[s:s + bs] for s in range(0, m, bs)])
    u_in = np.array([w[s:s + bs] @ z_in[s:s + bs] for s in range(0, m, bs)])
    v = np.array([d.quad(np.concatenate([np.zeros(s), w[s:s + bs],
                                         np.zeros(m - s - bs)]))
                  for s in range(0, m, bs)])
    est = overlap_slope(u_ta, u_in, np.full(m_blocks, float(bs)), v, v,
                        groups=np.arange(m_blocks) % 10)
    num_corr = correct_numerator(num_naive, est.gamma, m)
    return dict(gamma=est.gamma, gamma_true=n_ov / (n1 * n2),
                num_naive=num_naive, num_corr=num_corr,
                num_honest=num_honest, den=den)


def run(n1=4000, n2=4000, n_ind=4000, m_blocks=40, bs=75, rho=0.6, h2=0.3,
        n_causal=600, reps=5, seed=1000):
    """Overlap-detection validation across overlap fractions. Returns a dict of
    scalars (see KEYS) averaged over ``reps`` replicates."""
    rng = np.random.default_rng(seed)
    dense = {f: [_one(rng, f, False, n1, n2, n_ind, m_blocks, bs, rho, h2,
                      n_causal) for _ in range(reps)]
             for f in (0.0, 0.25, 1.0)}
    sparse_full = [_one(rng, 1.0, True, n1, n2, n_ind, m_blocks, bs, rho, h2,
                        n_causal) for _ in range(reps)]

    def mean(rows, k):
        return float(np.mean([r[k] for r in rows]))

    def r2(num, den):
        return num * num / den

    full = dense[1.0]
    return {
        "gamma_null": mean(dense[0.0], "gamma"),
        "gamma_low": mean(dense[0.25], "gamma"),
        "gamma_low_true": mean(dense[0.25], "gamma_true"),
        "gamma_full": mean(full, "gamma"),
        "gamma_full_true": mean(full, "gamma_true"),
        "r2_naive_full": float(np.mean([r2(r["num_naive"], r["den"]) for r in full])),
        "r2_corr_full": float(np.mean([r2(r["num_corr"], r["den"]) for r in full])),
        "r2_honest_full": float(np.mean([r2(r["num_honest"], r["den"]) for r in full])),
        "gamma_sparse_full": mean(sparse_full, "gamma"),
        "gamma_sparse_full_true": mean(sparse_full, "gamma_true"),
    }


def main():
    out = run()
    print("=== overlap detection (dense score) ===")
    print(f"  gamma null        = {out['gamma_null']:+.2e}   (truth 0)")
    print(f"  gamma 25% overlap = {out['gamma_low']:.2e}   (truth {out['gamma_low_true']:.2e})")
    print(f"  gamma 100%        = {out['gamma_full']:.2e}   (truth {out['gamma_full_true']:.2e})")
    print(f"  R2 at 100%: naive {out['r2_naive_full']:.4f} -> corrected {out['r2_corr_full']:.4f}"
          f"   (honest anchor {out['r2_honest_full']:.4f})")
    print("=== sparse score (known failure mode) ===")
    print(f"  gamma 100%        = {out['gamma_sparse_full']:.2e}   "
          f"(truth {out['gamma_sparse_full_true']:.2e}; detector blind -> flag as upper bound)")


if __name__ == "__main__":
    main()
