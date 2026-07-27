"""Benchmark (P1)-(P2): how accurate are the predicted moments, and when does
the normal percentile stop being trustworthy?

Four questions, each answered against simulated individuals whose true score
distribution is known by construction:

1. **Moment accuracy** across score density and LD strength. How close are the
   predicted mean and SD to the realized ones, and how wrong would you be if you
   ignored LD?
2. **Tail calibration.** A percentile is the product, not the SD. If the module
   says "99th percentile", what fraction of the cohort is actually above that
   threshold -- and does ``max_variance_share`` predict when the answer is bad?
3. **Structure.** The HWE identity ``Var(g) = 2f(1-f)`` fails in a structured
   cohort. How far off is the SD, and does the ``inbreeding`` knob recover it?
4. **Cost.** (P2) is the same quadratic form ``evaluate`` already computes, so
   the distribution should be free relative to an evaluation.

Run: ``python experiments/score_distribution.py``
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ppb import DenseLD, evaluate, score_distribution
from ppb.harmonize import VariantTable
from ppb.ld_backend import BlockDiagonalLD, DenseLDInt8
from ppb.simulate import _diploid_dosages, bn_freqs


def _variants(m):
    return VariantTable(chrom=np.full(m, "1"), pos=np.arange(1, m + 1),
                        a1=np.full(m, "A"), a2=np.full(m, "G"))


def _cohort(rng, block_sizes, n, rho, f):
    g = _diploid_dosages(n, block_sizes, f, rho, rng)
    return g, g.mean(axis=0) / 2.0, np.corrcoef(g.T)


def moment_accuracy(rng, *, n=40_000):
    """Predicted vs realized moments, and the cost of ignoring LD."""
    rows = []
    for label, block_sizes, rho in (
        ("weak LD",   [25] * 8, 0.2),
        ("moderate",  [25] * 8, 0.6),
        ("strong LD", [25] * 8, 0.9),
        ("few big blocks", [100] * 2, 0.9),
    ):
        m = sum(block_sizes)
        f = rng.uniform(0.05, 0.95, size=m)
        g, f_hat, D = _cohort(rng, block_sizes, n, rho, f)
        variants = _variants(m)
        for density, causal in (("sparse 5%", 0.05), ("dense", 1.0)):
            w = rng.normal(size=m) * (rng.random(m) < causal)
            if not np.any(w):
                continue
            scores = g @ w
            with_ld = score_distribution(DenseLD(D), variants, variants, w, f_hat)
            no_ld = score_distribution(DenseLD(np.eye(m)), variants, variants,
                                       w, f_hat)
            rows.append((
                f"{label}, {density}",
                abs(with_ld.mean - scores.mean()) / scores.std(),
                abs(with_ld.sd / scores.std() - 1.0) * 100.0,
                abs(no_ld.sd / scores.std() - 1.0) * 100.0,
            ))
    return rows


def tail_calibration(rng, *, n=200_000, m=400, block=20):
    """Does max_variance_share predict when the normal tail is wrong?

    Each row adds one variant carrying a growing share of the variance -- the
    APOE case -- and reports the realized fraction above the nominal 99th and
    99.9th percentile thresholds.
    """
    block_sizes = [block] * (m // block)
    f = np.concatenate([[0.05], rng.uniform(0.1, 0.9, size=m - 1)])
    g, f_hat, _ = _cohort(rng, block_sizes, n, 0.5, f)
    blocks, start = [], 0
    D_full = np.corrcoef(g.T)
    for k in block_sizes:
        idx = np.arange(start, start + k)
        blocks.append((DenseLD(D_full[np.ix_(idx, idx)]), idx))
        start += k
    ld = BlockDiagonalLD(blocks)
    variants = _variants(m)

    base = rng.normal(size=m)
    base[0] = 0.0
    rows = []
    for big in (0.0, 5.0, 12.0, 18.0, 24.0, 30.0, 40.0, 60.0):
        w = base.copy()
        w[0] = big
        dist = score_distribution(ld, variants, variants, w, f_hat)
        scores = g @ w
        out = [dist.max_variance_share]
        for nominal in (1.0, 0.1):
            z = {1.0: 2.3263478740408408, 0.1: 3.090232306167813}[nominal]
            threshold = dist.mean + z * dist.sd
            out.append(100.0 * np.mean(scores > threshold))
        # worst absolute percentile error over the whole cohort
        empirical = 100.0 * (np.argsort(np.argsort(scores)) + 0.5) / n
        out.append(np.abs(dist.percentile(scores) - empirical).max())
        rows.append((big, *out))
    return rows


def structured_cohort(rng, *, n=60_000, fst=0.05):
    """A within-ancestry panel's D and HWE, applied to a pooled cohort."""
    block_sizes = [25] * 8
    m = sum(block_sizes)
    p1, p2 = bn_freqs(rng, m, fst)
    g1 = _diploid_dosages(n // 2, block_sizes, p1, 0.5, rng)
    g2 = _diploid_dosages(n // 2, block_sizes, p2, 0.5, rng)
    g = np.vstack([g1, g2])
    w = rng.normal(size=m)
    variants = _variants(m)

    # The realistic mistake: a clean single-population LD panel, pooled f.
    D_panel = np.corrcoef(g1.T)
    f_pooled = g.mean(axis=0) / 2.0
    realized = (g @ w).std()

    rows = []
    for label, F in (("no correction", 0.0), (f"inbreeding=fst={fst}", fst)):
        dist = score_distribution(DenseLD(D_panel), variants, variants,
                                  w, f_pooled, inbreeding=F)
        rows.append((label, dist.sd, (dist.sd / realized - 1.0) * 100.0))
    # For contrast: the same score inside one homogeneous subpopulation.
    within = score_distribution(DenseLD(D_panel), variants, variants,
                                w, g1.mean(axis=0) / 2.0)
    rows.append(("within population 1", within.sd,
                 (within.sd / (g1 @ w).std() - 1.0) * 100.0))
    return rows, realized


def cost(rng, *, m=120_000, n_blocks=431):
    """(P2) versus a full evaluation on the same block-diagonal reference."""
    sizes = np.full(n_blocks, m // n_blocks)
    sizes[: m - sizes.sum()] += 1
    blocks, start = [], 0
    for k in sizes:
        a = rng.normal(size=(int(k) + 4, int(k)))
        Db = np.corrcoef(a.T)
        blocks.append((DenseLDInt8.from_dense(Db), np.arange(start, start + int(k))))
        start += int(k)
    ld = BlockDiagonalLD(blocks)
    variants = _variants(m)
    w = rng.normal(size=m) * (rng.random(m) < 0.1)
    f = rng.uniform(0.05, 0.95, size=m)
    z = rng.normal(size=m) / np.sqrt(100_000)

    t0 = time.perf_counter()
    evaluate(ld, variants, variants, w, variants, z)
    t_eval = time.perf_counter() - t0
    t0 = time.perf_counter()
    dist = score_distribution(ld, variants, variants, w, f)
    t_dist = time.perf_counter() - t0
    return m, n_blocks, t_eval, t_dist, dist.max_variance_share


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("1. Moment accuracy (40,000 individuals)")
    print(f"{'regime':<26}{'mean err/sd':>13}{'sd err %':>10}{'sd err %, no LD':>18}")
    print("-" * 67)
    for label, mean_err, sd_err, no_ld_err in moment_accuracy(rng):
        print(f"{label:<26}{mean_err:>13.4f}{sd_err:>10.2f}{no_ld_err:>18.2f}")

    print("\n2. Tail calibration (200,000 individuals; one variant grows)")
    print(f"{'w[0]':>6}{'max var share':>15}{'>nominal 1%':>13}"
          f"{'>nominal 0.1%':>15}{'max pctile err':>16}")
    print("-" * 65)
    for big, share, p1, p01, worst in tail_calibration(rng):
        print(f"{big:>6.0f}{share:>15.3f}{p1:>13.2f}{p01:>15.3f}{worst:>16.1f}")
    print("   nominal:                      1.00          0.100")

    print("\n3. Structured cohort (two populations, fst=0.05, pooled)")
    rows, realized = structured_cohort(rng)
    print(f"   realized sd in the pooled cohort: {realized:.4f}")
    print(f"{'assumption':<26}{'predicted sd':>14}{'error %':>10}")
    print("-" * 50)
    for label, sd, err in rows:
        print(f"{label:<26}{sd:>14.4f}{err:>10.2f}")

    print("\n4. Cost relative to an evaluation")
    m, nb, t_eval, t_dist, share = cost(rng)
    print(f"   {m:,} variants in {nb} int8 blocks")
    print(f"   evaluate           {t_eval * 1e3:8.1f} ms")
    print(f"   score_distribution {t_dist * 1e3:8.1f} ms"
          f"   ({t_dist / t_eval:.2f}x)")
    print(f"   max_variance_share {share:.4f}")


if __name__ == "__main__":
    main()
