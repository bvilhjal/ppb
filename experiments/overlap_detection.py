"""Basis-aware overlap detection and correction: where it works, and where it does not.

A training GWAS and a target GWAS share a controlled fraction of *individuals*;
an independent GWAS of the same size is the honest anchor. The detector jointly
fits a genuine target/reference signal scale ``alpha`` and the shared-noise
coupling ``gamma``, and corrects only when both are identified and stable.

**The identification boundary is a property of the architecture, not of the
method.** The design has two columns -- the reference signal ``u_R`` and the
trainer basis ``q`` -- and both are positive and both grow with block size, so
separating them needs signal variation *at fixed block size*. A diffuse
architecture has almost none: every block carries a similar amount of signal,
the columns are collinear, and the fit is correctly refused. A sparse
architecture has a great deal, and the same code identifies cleanly and recovers
the coupling. Earlier revisions of this experiment ran only the diffuse corner
with a *constant* basis (a marginal trainer over equal-sized blocks makes
``q_b = tr(D_b)`` the block size), which is the least identifiable configuration
available, and recorded the resulting refusal as the method's operating limit.

Real polygenic traits over the 1.44M-variant HM3+ reference are far sparser than
the diffuse corner, and its 431 LD blocks range from 216 to 17,304 variants, so
the realistic regime is the identified one.

Run:
    python experiments/overlap_detection.py
"""

from __future__ import annotations

import numpy as np

from ppb import DenseLD, estimate_overlap_basis
from ppb.overlap import OverlapBasis, correct_overlap_numerator, fit_overlap
from ppb.simulate import (draw_effects, marginal_stats, pgs_pthreshold,
                          simulate_diploid_genotypes, simulate_phenotype)

# Heterogeneous block sizes, as a real LD reference has.
BLOCK_SIZES = (15, 30, 60, 120)
KEYS = ("diffuse_status", "sparse_null_status", "sparse_low_status",
        "sparse_full_status", "gamma_full_ratio", "r2_naive_full",
        "r2_corr_full", "r2_honest_full", "basis_error", "sparse_trainer_status")


def _cohorts(rng, n, per_size, rho):
    """Genotypes for train/target/independent, with heterogeneous LD blocks."""
    columns, sizes = [], []
    for block_size in BLOCK_SIZES:
        maf = rng.uniform(0.05, 0.5, size=block_size * per_size)
        columns.append(simulate_diploid_genotypes(
            3 * n, [block_size] * per_size, maf, rho, rng))
        sizes.extend([block_size] * per_size)
    x = np.hstack(columns)
    sizes = np.asarray(sizes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return x, [np.arange(s, s + k) for s, k in zip(starts, sizes)]


def _one(rng, frac, causal_frac, *, n, per_size, rho, h2, ridge, thresholded,
         hutchinson_draws):
    x, idxs = _cohorts(rng, n, per_size, rho)
    m = x.shape[1]
    beta = draw_effects(m, max(10, int(causal_frac * m)), rng)
    y = simulate_phenotype(x, beta, h2, rng)

    i_train = np.arange(n)
    n_overlap = int(frac * n)
    i_target = np.concatenate([rng.choice(i_train, n_overlap, replace=False),
                               np.arange(n, 2 * n - n_overlap)])
    z_train, t_train = marginal_stats(x[i_train], y[i_train])
    z_target, _ = marginal_stats(x[i_target], y[i_target])
    z_indep, _ = marginal_stats(x[2 * n:], y[2 * n:])

    D = np.corrcoef(x.T)
    blocks = [D[np.ix_(i, i)] for i in idxs]
    operators = [np.linalg.solve(Db + ridge * np.eye(len(Db)), np.eye(len(Db)))
                 for Db in blocks]

    def trainer(values):
        """Blockwise ridge -- rerunnable, and linear so its basis is checkable."""
        w = np.zeros_like(values)
        for Phi, idx in zip(operators, idxs):
            w[idx] = Phi @ values[idx]
        return w

    # (O3): q_b = tr(Phi_b' K_b) with K = D, exact for this linear trainer.
    analytic = np.array([float(np.trace(Phi.T @ Db))
                         for Phi, Db in zip(operators, blocks)])
    if thresholded:
        # Discontinuous selection: the local Jacobian misses the selection
        # response, and the perturbation-stability gate must say so.
        w = pgs_pthreshold(z_train, t_train, 2.5)
        basis = estimate_overlap_basis(
            lambda values: pgs_pthreshold(
                values, values / np.sqrt(np.clip(1.0 - values ** 2, 1e-12, None) / n),
                2.5),
            z_train, idxs, [np.linalg.cholesky(Db) for Db in blocks],
            rng=rng, provenance="p+T trainer", support_hash="simulation",
            n_draws=hutchinson_draws)
    else:
        w = trainer(z_train)
        basis = estimate_overlap_basis(
            trainer, z_train, idxs, [np.linalg.cholesky(Db) for Db in blocks],
            rng=rng, provenance="blockwise ridge trainer",
            support_hash="simulation", n_draws=hutchinson_draws)

    u_target = np.array([w[i] @ z_target[i] for i in idxs])
    u_indep = np.array([w[i] @ z_indep[i] for i in idxs])
    v = np.array([float(w[i] @ D[np.ix_(i, i)] @ w[i]) for i in idxs])
    fit = fit_overlap(u_target, u_indep, v / n, v / n, basis=basis,
                      groups=np.arange(len(idxs)) % 20)

    den = DenseLD(D).quad(w)
    corrected = correct_overlap_numerator(fit) if fit.can_correct else np.nan
    return dict(
        status=fit.status, vif=fit.vif, condition_number=fit.condition_number,
        alpha=fit.alpha, gamma=fit.gamma, gamma_true=n_overlap / (n * n),
        basis_error=(abs(basis.values.sum() - analytic.sum()) / abs(analytic.sum())
                     if basis.available and not thresholded else np.nan),
        basis_kind=basis.kind,
        r2_naive=(w @ z_target) ** 2 / den,
        r2_corrected=corrected ** 2 / den if np.isfinite(corrected) else np.nan,
        r2_honest=(w @ z_indep) ** 2 / den)


def run(n=2500, per_size=15, rho=0.6, h2=0.3, ridge=0.5, reps=2,
        diffuse_causal=0.20, sparse_causal=0.01, hutchinson_draws=48,
        seed=1000):
    """Both regimes plus the discontinuous-trainer case. See KEYS."""
    rng = np.random.default_rng(seed)
    common = dict(n=n, per_size=per_size, rho=rho, h2=h2, ridge=ridge,
                  thresholded=False, hutchinson_draws=hutchinson_draws)

    def batch(frac, causal):
        return [_one(rng, frac, causal, **common) for _ in range(reps)]

    diffuse = batch(1.0, diffuse_causal)
    sparse = {f: batch(f, sparse_causal) for f in (0.0, 0.25, 1.0)}
    thresholded = _one(rng, 1.0, sparse_causal,
                       **{**common, "thresholded": True})

    def mean(rows, key):
        values = [r[key] for r in rows
                  if r[key] is not None and np.isfinite(r[key])]
        return float(np.mean(values)) if values else np.nan

    full = sparse[1.0]
    return {
        "diffuse_status": tuple(r["status"] for r in diffuse),
        "diffuse_vif": mean(diffuse, "vif"),
        "diffuse_gamma_ratio": mean(diffuse, "gamma") / mean(diffuse, "gamma_true"),
        "sparse_null_status": tuple(r["status"] for r in sparse[0.0]),
        "sparse_low_status": tuple(r["status"] for r in sparse[0.25]),
        "sparse_full_status": tuple(r["status"] for r in full),
        "sparse_vif": mean(full, "vif"),
        "alpha_full": mean(full, "alpha"),
        "gamma_full_ratio": mean(full, "gamma") / mean(full, "gamma_true"),
        "gamma_low_ratio": (mean(sparse[0.25], "gamma")
                            / mean(sparse[0.25], "gamma_true")),
        "corrections_at_null": sum(r["status"] == "correctable"
                                   for r in sparse[0.0]),
        "r2_naive_full": mean(full, "r2_naive"),
        "r2_corr_full": mean(full, "r2_corrected"),
        "r2_honest_full": mean(full, "r2_honest"),
        "r2_naive_low": mean(sparse[0.25], "r2_naive"),
        "r2_corr_low": mean(sparse[0.25], "r2_corrected"),
        "r2_honest_low": mean(sparse[0.25], "r2_honest"),
        "basis_error": mean(full, "basis_error"),
        "sparse_trainer_status": thresholded["status"],
        "sparse_trainer_basis": thresholded["basis_kind"],
    }


def main():
    out = run()
    print("=== Hutchinson basis for a rerunnable trainer ===")
    print(f"  relative error vs analytic tr(Phi'K) = {out['basis_error']:.4f}")
    print("\n=== diffuse architecture (20% causal): the refusal corner ===")
    print(f"  status = {out['diffuse_status']}, VIF {out['diffuse_vif']:.2f} "
          f"(gate 2.0); gamma/true {out['diffuse_gamma_ratio']:.2f}")
    print("  the coupling is nearly right, but signal and basis are collinear,")
    print("  so the identification gate refuses -- correctly.")
    print("\n=== sparse architecture (1% causal): identified ===")
    print(f"  null       {out['sparse_null_status']}  "
          f"({out['corrections_at_null']} corrections issued; must be 0)")
    print(f"  25% overlap{out['sparse_low_status']}  "
          f"gamma/true {out['gamma_low_ratio']:.2f}   "
          f"R2 {out['r2_naive_low']:.4f} -> {out['r2_corr_low']:.4f} "
          f"(anchor {out['r2_honest_low']:.4f})")
    print(f"  100%       {out['sparse_full_status']}  "
          f"gamma/true {out['gamma_full_ratio']:.2f}   "
          f"R2 {out['r2_naive_full']:.4f} -> {out['r2_corr_full']:.4f} "
          f"(anchor {out['r2_honest_full']:.4f})")
    print(f"  VIF {out['sparse_vif']:.2f}, alpha {out['alpha_full']:.2f}")
    print("\n=== discontinuous trainer (p+T) ===")
    print(f"  basis = {out['sparse_trainer_basis']}, fit status = "
          f"{out['sparse_trainer_status']} (correction refused)")


if __name__ == "__main__":
    main()
