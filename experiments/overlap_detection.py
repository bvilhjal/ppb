"""Basis-aware overlap detection and correction: simulation validation.

A training GWAS (n1) and a target GWAS (n2) share a controlled fraction of
individuals; an independent GWAS (n2) is the honest anchor. For a dense
(marginal, ``w = z_train``) score the trainer operator and its block basis are
known exactly.  The detector jointly fits a genuine target/reference signal
scale and the shared-noise coefficient.  Correction is attempted only when
those components are identified and stable.  A thresholded score is passed as
an explicitly unavailable basis and therefore fails closed rather than
silently substituting variant count.

Run:
    python experiments/overlap_detection.py
"""

from __future__ import annotations

import numpy as np

from ppb import DenseLD
from ppb.overlap import (
    OverlapBasis,
    correct_overlap_numerator,
    fit_overlap,
)
from ppb.simulate import (draw_effects, marginal_stats, pgs_pthreshold,
                          population_ld, simulate_diploid_genotypes,
                          simulate_phenotype)

KEYS = ("alpha_null", "gamma_null", "gamma_low", "gamma_low_true",
        "gamma_full", "gamma_full_true", "r2_naive_full", "r2_corr_full",
        "r2_honest_full", "correctable_full_fraction", "sparse_status")


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
    if sparse:
        basis = OverlapBasis.unavailable(
            "p-value thresholding operator was not reconstructed")
    else:
        basis = OverlapBasis(
            values=np.full(m_blocks, float(bs)),
            kind="linear_trace",
            provenance="simulation: marginal trainer A = I, K = D",
            support_hash=f"simulation-m{m}-bs{bs}",
        )
    fit = fit_overlap(
        u_ta, u_in, v / n2, v / n_ind, basis=basis,
        groups=np.arange(m_blocks) % 20)
    num_corr = correct_overlap_numerator(fit) if fit.can_correct else np.nan
    return dict(alpha=fit.alpha, gamma=fit.gamma, status=fit.status,
                vif=fit.vif, condition_number=fit.condition_number,
                gamma_true=n_ov / (n1 * n2),
                num_naive=num_naive, num_corr=num_corr,
                num_honest=num_honest, den=den)


def run(n1=4000, n2=4000, n_ind=4000, m_blocks=60, bs=50, rho=0.6, h2=0.3,
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
        values = [r[k] for r in rows if r[k] is not None and np.isfinite(r[k])]
        return float(np.mean(values)) if values else np.nan

    def r2(num, den):
        return num * num / den

    full = dense[1.0]
    return {
        "alpha_null": mean(dense[0.0], "alpha"),
        "gamma_null": mean(dense[0.0], "gamma"),
        "gamma_low": mean(dense[0.25], "gamma"),
        "gamma_low_true": mean(dense[0.25], "gamma_true"),
        "gamma_full": mean(full, "gamma"),
        "gamma_full_true": mean(full, "gamma_true"),
        "alpha_full": mean(full, "alpha"),
        "vif_full": mean(full, "vif"),
        "condition_full": mean(full, "condition_number"),
        "r2_naive_full": float(np.mean([r2(r["num_naive"], r["den"]) for r in full])),
        "r2_corr_full": mean(
            [dict(value=r2(r["num_corr"], r["den"])) for r in full
             if np.isfinite(r["num_corr"])], "value"),
        "r2_honest_full": float(np.mean([r2(r["num_honest"], r["den"]) for r in full])),
        "correctable_full_fraction": float(np.mean([r["status"] == "correctable"
                                                     for r in full])),
        "status_null": tuple(r["status"] for r in dense[0.0]),
        "status_low": tuple(r["status"] for r in dense[0.25]),
        "status_full": tuple(r["status"] for r in full),
        "sparse_status": sparse_full[0]["status"],
    }


def main():
    out = run()
    print("=== basis-aware overlap fit (known dense trainer) ===")
    print(f"  null alpha        = {out['alpha_null']:.3f}   (truth approximately 1)")
    print(f"  gamma null        = {out['gamma_null']:+.2e}   (truth 0; {out['status_null']})")
    print(f"  gamma 25% overlap = {out['gamma_low']:.2e}   (truth {out['gamma_low_true']:.2e})")
    print(f"  gamma 100%        = {out['gamma_full']:.2e}   "
          f"(truth {out['gamma_full_true']:.2e}; {out['status_full']})")
    print(f"  full-overlap ID   = alpha {out['alpha_full']:.3f}, "
          f"VIF {out['vif_full']:.2f}, condition {out['condition_full']:.1f}")
    if np.isfinite(out["r2_corr_full"]):
        print(f"  R2 at 100%: naive {out['r2_naive_full']:.4f} -> "
              f"corrected {out['r2_corr_full']:.4f} "
              f"(honest anchor {out['r2_honest_full']:.4f})")
    else:
        print("  R2 correction refused by identification/stability gates")
    print("=== thresholded trainer with unknown sensitivity basis ===")
    print(f"  status            = {out['sparse_status']} (correction refused)")


if __name__ == "__main__":
    main()
