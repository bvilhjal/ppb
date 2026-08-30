"""Simulation study for the LD-moment ancestry estimators (the benchmark arm
recommended in ``docs/ancestry_report/ancestry_report.tex``, Section 12).

Pits Estimator A (pair products, with and without the quadratic signal
absorbers) and Estimator B (bilinear LD-score regression) against simulated
admixtures with known composition, across the regimes the report identifies:

* **signal regime** -- nh^2/m from 0 (null trait) to 0.5 (tagging-dominated);
  the raw pair-product fit's bias toward the high-LD ancestry grows with
  nh^2/m; the absorber is evaluated as a conditional correction rather than
  assumed to help at every signal level.
* **ancestry contrast** -- distinct vs overlapping LD landscapes
  (confusability shows up as inflated jackknife SEs and boundary estimates).
* **reference-panel size** -- an exploratory n_ref=500 versus 8000 arm using
  one realised panel at each size; it does not estimate reference uncertainty.
* **model mismatch** -- individual-level mosaic GWAS (Wahlund frequency
  contrast at fst = 0.05 and 0.2) versus the model-consistent MVN arm.

Run: ``python experiments/ancestry_ld_study.py`` (a few minutes). Results are
printed as a markdown table; ``--out PATH`` additionally dumps the JSON (the
``results/`` directory is the strict real-data registry -- study output does
not belong there). The encoded, fast version of these demonstrations lives in
``tests/test_ancestry.py``.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ppb.ancestry import estimate_bilinear, estimate_pair_products
from ppb.simulate import (block_correlations, bn_freqs_multi, draw_effects,
                          draw_ld_paths, marginal_stats,
                          simulate_admixture_mvn, simulate_admixture_references,
                          simulate_admixed_genotypes, simulate_phenotype)

PI2 = np.array([0.65, 0.35])
PI4 = np.array([0.4, 0.3, 0.2, 0.1])


def _gate_reason(result):
    note = result.get("signal_note") or ""
    if "not identifiable" in note or "ill-conditioned" in note:
        return "design"
    if "fewer than four" in note:
        return "too_few_groups"
    if "insufficient reproducible" in note:
        return "heldout_signal"
    if "positive-semidefinite" in note:
        return "non_psd"
    if "rank-one" in note:
        return "rank_one"
    if "capping" in note:
        return "capped"
    return "other"


def make_landscape(seed, n_blocks, ranges):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(15, 36, n_blocks)
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    blocks = [np.arange(bounds[b], bounds[b + 1]) for b in range(n_blocks)]
    paths = [draw_ld_paths(rng, sizes, lo, hi) for lo, hi in ranges]
    refs = [[M[np.ix_(b, b)] for b in blocks]
            for M in (block_correlations(p) for p in paths)]
    return [int(s) for s in sizes], blocks, paths, refs


def mvn_arm(refs, blocks, pi, h2, n, reps, seed, estimator, **kw):
    ests, ses, declined = [], [], 0
    decline_reasons = {}
    design_condition = None

    def count_decline(reason):
        decline_reasons[reason] = decline_reasons.get(reason, 0) + 1

    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        z = simulate_admixture_mvn(refs, pi, h2, n, rng)
        try:
            r = estimator(z, refs, blocks=blocks, **kw)
        except ValueError as exc:
            if "no positive fitted linear pair-product component" not in str(exc):
                raise
            declined += 1
            count_decline("no_linear_pair_signal")
            continue
        if "design_condition" in r:
            design_condition = r["design_condition"]
        if r["proportions"] is None:
            declined += 1
            count_decline(_gate_reason(r))
            continue
        ests.append(r["proportions"])
        ses.append(r["proportions_se"])
    diagnostic = {"decline_reasons": decline_reasons}
    if design_condition is not None:
        diagnostic["design_condition"] = float(design_condition)
    if not ests:
        return {"declined": declined, "reps": reps, **diagnostic}
    ests = np.asarray(ests)
    out = {
        "declined": declined,
        "reps": reps,
        "mean": ests.mean(axis=0).round(4).tolist(),
        "max_abs_error": float(np.abs(ests.mean(axis=0) - pi).max()),
        "empirical_sd": ests.std(axis=0).round(4).tolist(),
        **diagnostic,
    }
    ses = np.asarray([s for s in ses if s is not None])
    if ses.size:
        out["mean_jackknife_se"] = ses.mean(axis=0).round(4).tolist()
    return out


def il_arm(sizes, blocks, paths, pi, fst, n_ref, reps, seed, h2=0.0,
           n=3000, estimator=estimate_pair_products, **kw):
    m = sum(sizes)
    rng = np.random.default_rng(seed)
    maf = bn_freqs_multi(rng, m, fst, len(paths))
    refs = simulate_admixture_references(n_ref, sizes, maf, paths, rng)
    rb = [[R[np.ix_(b, b)] for b in blocks] for R in refs]
    ests, declined, decline_reasons = [], 0, {}
    for _ in range(reps):
        X = simulate_admixed_genotypes(n, sizes, maf, paths, pi, rng)
        y = simulate_phenotype(X, draw_effects(m, 50, rng), h2, rng)
        _, t = marginal_stats(X, y)
        try:
            r = estimator(t, rb, blocks=blocks, **kw)
        except ValueError as exc:
            if "no positive fitted linear pair-product component" not in str(exc):
                raise
            declined += 1
            decline_reasons["no_linear_pair_signal"] = (
                decline_reasons.get("no_linear_pair_signal", 0) + 1
            )
            continue
        if r["proportions"] is None:
            declined += 1
            reason = _gate_reason(r)
            decline_reasons[reason] = decline_reasons.get(reason, 0) + 1
            continue
        ests.append(r["proportions"])
    if not ests:
        return {"declined": declined, "reps": reps,
                "decline_reasons": decline_reasons}
    ests = np.asarray(ests)
    return {
        "declined": declined,
        "reps": reps,
        "mean": ests.mean(axis=0).round(4).tolist(),
        "max_abs_error": float(np.abs(ests.mean(axis=0) - pi).max()),
        "empirical_sd": ests.std(axis=0).round(4).tolist(),
        "decline_reasons": decline_reasons,
    }


def run(reps=16, il_reps=8):
    results = {}
    sizes, blocks, paths, RB = make_landscape(7, 400, [(0.50, 0.90), (0.10, 0.50)])
    m = sum(sizes)

    t0 = time.time()
    for nh2m in (0.0, 0.025, 0.1, 0.5):
        h2 = 0.25 if nh2m else 0.0
        n = int(nh2m * m / h2) if h2 else 1000
        for absorb in (False, True):
            key = f"A nh2/m={nh2m} absorb={int(absorb)}"
            results[key] = mvn_arm(RB, blocks, PI2, h2, n, reps, 300,
                                   estimate_pair_products,
                                   absorb_signal=absorb)
    for nh2m in (0.5, 2.0, 8.0):
        n = int(nh2m * m / 0.5)
        results[f"B nh2/m={nh2m}"] = mvn_arm(RB, blocks, PI2, 0.5, n, reps,
                                             600, estimate_bilinear)
    print(f"[mvn arms done in {time.time() - t0:.0f}s]")

    t0 = time.time()
    sizes4, blocks4, paths4, RB4 = make_landscape(
        11, 300, [(0.55, 0.90), (0.08, 0.38), (0.32, 0.62), (0.0, 0.15)])
    results["A K=4 h2=0"] = mvn_arm(RB4, blocks4, PI4, 0.0, 1000, reps, 800,
                                    estimate_pair_products)
    results["B K=4 nh2/m=8"] = mvn_arm(RB4, blocks4, PI4, 0.5,
                                       int(8.0 * sum(sizes4) / 0.5), reps, 900,
                                       estimate_bilinear)
    print(f"[K=4 done in {time.time() - t0:.0f}s]")

    t0 = time.time()
    sizes_c, blocks_c, paths_c, RB_c = make_landscape(
        21, 400, [(0.30, 0.60), (0.35, 0.65)])   # overlapping: confusable
    results["A confusable h2=0"] = mvn_arm(RB_c, blocks_c, PI2, 0.0, 1000,
                                           reps, 1000, estimate_pair_products)
    print(f"[confusability done in {time.time() - t0:.0f}s]")

    t0 = time.time()
    sizes_s, blocks_s, paths_s, _ = make_landscape(42, 60,
                                                   [(0.50, 0.90), (0.10, 0.50)])
    for fst in (0.05, 0.2):
        results[f"IL fst={fst} h2=0 A"] = il_arm(
            sizes_s, blocks_s, paths_s, PI2, fst, 3000, il_reps, 1100)
    results["IL fst=0.05 h2=0.5 B"] = il_arm(
        sizes_s, blocks_s, paths_s, PI2, 0.05, 3000, il_reps, 1200, h2=0.5,
        n=8000, estimator=estimate_bilinear)
    for n_ref in (500, 8000):
        results[f"IL fst=0.05 n_ref={n_ref} A"] = il_arm(
            sizes_s, blocks_s, paths_s, PI2, 0.05, n_ref, il_reps, 1300)
    print(f"[individual-level arms done in {time.time() - t0:.0f}s]")
    return results


def _fmt(row):
    declines = (f" | decline reasons {row['decline_reasons']}"
                if row.get("decline_reasons") else "")
    if "mean" not in row:
        return f"declined {row['declined']}/{row['reps']}{declines}"
    mean = ", ".join(f"{x:.3f}" for x in row["mean"])
    se = ("; SE " + ", ".join(f"{x:.3f}" for x in row["mean_jackknife_se"])
          if "mean_jackknife_se" in row else "")
    return (f"mean ({mean}) | max err {row['max_abs_error']:.3f} | "
            f"SD {', '.join(f'{x:.3f}' for x in row['empirical_sd'])}{se}"
            + (f" | declined {row['declined']}" if row["declined"] else "")
            + declines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--il-reps", type=int, default=8)
    ap.add_argument("--out", default=None,
                        help="optional JSON dump path (default: print only; "
                             "results/ is the strict real-data registry, so "
                             "study output does not belong there)")
    args = ap.parse_args()
    results = run(reps=args.reps, il_reps=args.il_reps)
    width = max(len(k) for k in results)
    print(f"\n{'arm'.ljust(width)}  result")
    print("-" * (width + 40))
    for key, row in results.items():
        print(f"{key.ljust(width)}  {_fmt(row)}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
