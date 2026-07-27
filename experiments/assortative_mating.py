"""Assortative mating inflates the true score variance where PPB cannot see it.

`docs/LIMITATIONS.md` records that the source paper found Major Depression R^2
overestimated by ~4.5%, "hypothesised to stem from assortative mating". This
tests that hypothesis mechanically, by mating a simulated population on its
phenotype for several generations and asking what breaks.

The mechanism: AM correlates trait-increasing alleles at *unlinked* loci, which
inflates additive genetic variance above the linkage-equilibrium value. PPB's
denominator ``w^T D w`` uses a **block-diagonal** reference whose off-block
entries are zero by construction, so the inflation is invisible to it, the
denominator is too small, and R^2 comes out too high.

The interesting part is that the distortion has structure. AM correlates alleles
in proportion to their effects, so the missing covariance is approximately
rank-one along ``sd * beta``:

**(P3) Assortative-mating variance correction.**

    Var(S) ~= w_s^T D_block w_s + c (w_s^T v)^2,   v = sd * beta

One scalar therefore corrects every score on that reference -- no genome-wide LD
matrix required. Whether `c` is transferable, and whether it follows the
classical equilibrium prediction, is what this experiment measures.

Run: ``python experiments/assortative_mating.py``
"""

from __future__ import annotations

import argparse

import numpy as np

from ppb.simulate import assortatively_mated_offspring, simulate_haplotypes

BLOCKS, BLOCK_SIZE = 30, 20
M = BLOCKS * BLOCK_SIZE
BLOCK_SIZES = [BLOCK_SIZE] * BLOCKS


def _blocked(D):
    """Zero every off-block entry, as the shipped LD reference does."""
    out = np.zeros_like(D)
    for b in range(BLOCKS):
        i = np.arange(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)
        out[np.ix_(i, i)] = D[np.ix_(i, i)]
    return out


def _le_variance(beta, f):
    """Linkage-equilibrium variance: the sum of per-variant contributions."""
    return float(np.sum(beta ** 2 * 2.0 * f * (1.0 - f)))


def run(rng, *, n=20_000, generations=10, spouse_correlation=0.4, h2=0.6,
        recombination=0.01):
    """Mate a population assortatively and track what happens to the variance."""
    f0 = rng.uniform(0.15, 0.85, size=M)
    hap = simulate_haplotypes(n, BLOCK_SIZES, f0, 0.5, rng)
    beta = rng.normal(size=M)

    g = hap.sum(axis=1, dtype=np.float64)
    bv = g @ beta
    # Environmental variance is fixed at generation 0, so the realized h^2 rises
    # with the genetic variance -- which is what actually happens under AM.
    var_e = bv.var() * (1.0 - h2) / h2

    history = []
    for gen in range(generations):
        g = hap.sum(axis=1, dtype=np.float64)
        bv = g @ beta
        f = g.mean(axis=0) / 2.0
        y = bv + rng.normal(scale=np.sqrt(var_e), size=n)
        history.append({
            "gen": gen,
            "inflation": bv.var() / _le_variance(beta, f),
            "h2": bv.var() / y.var(),
        })
        hap, realized = assortatively_mated_offspring(
            hap, y, BLOCK_SIZES, rng, spouse_correlation=spouse_correlation,
            recombination=recombination)
        history[-1]["spouse_r"] = realized

    g = hap.sum(axis=1, dtype=np.float64)
    return hap, g, beta, history


def r2_bias(rng, g, beta, *, h2=0.6):
    """What PPB reports on an assortatively mated population, and the truth."""
    n = g.shape[0]
    f = g.mean(axis=0) / 2.0
    sd = np.sqrt(2.0 * f * (1.0 - f))
    bv = g @ beta
    y = bv + rng.normal(scale=np.sqrt(bv.var() * (1.0 - h2) / h2), size=n)

    D_full = np.corrcoef(g.T)
    D_block = _blocked(D_full)
    gz = (g - g.mean(axis=0)) / g.std(axis=0)
    z = gz.T @ ((y - y.mean()) / y.std()) / n

    ws = beta * sd
    truth = float(np.corrcoef(g @ beta, y)[0, 1] ** 2)
    return {
        "true_r2": truth,
        "r2_block": float((ws @ z) ** 2 / (ws @ D_block @ ws)),
        "r2_full": float((ws @ z) ** 2 / (ws @ D_full @ ws)),
        "sd_realized": float((g @ beta).std()),
        "sd_block": float(np.sqrt(ws @ D_block @ ws)),
        "sd_le": float(np.sqrt(_le_variance(beta, f))),
        "D_full": D_full, "D_block": D_block, "sd": sd, "f": f,
    }


def predicted_c(fit, beta, *, spouse_r, h2):
    """`c` from the classical equilibrium, with no genome-wide LD.

    ``w_s^T v = V_LE`` when the score is the causal effect vector, so

        c = (V_A - V_block) / V_LE^2,     V_A = V_LE / (1 - r h^2)

    Everything on the right is available without an off-block correlation: the
    spouse correlation and heritability come from the literature, and both
    variances from allele frequencies and the block reference.
    """
    ws = beta * fit["sd"]
    v_le = float(_le_variance(beta, fit["f"]))
    v_block = float(ws @ fit["D_block"] @ ws)
    v_a = v_le / (1.0 - spouse_r * h2)
    return (v_a - v_block) / v_le ** 2


def coverage_sweep(rng, fit, beta):
    """AM bias against how much of the effect direction a score captures.

    The AM term is ``(w_s^T v)^2`` while the linkage-equilibrium term is linear
    in the score's size, so a score covering a fraction `p` of the trait's
    effect-weighted variance carries roughly `p` times the full bias. Real PGS
    are far from complete, which bounds how much of this reaches them.
    """
    D_full, D_block, sd = fit["D_full"], fit["D_block"], fit["sd"]
    rows = []
    for p in (0.05, 0.1, 0.25, 0.5, 1.0):
        w = beta * (rng.random(M) < p)
        a = w * sd
        true = float(np.sqrt(a @ D_full @ a))
        blk = float(np.sqrt(a @ D_block @ a))
        captured = float((a @ (sd * beta)) / _le_variance(beta, fit["f"]))
        rows.append((p, captured, (true / blk - 1) * 100.0))
    return rows


def rank_one_transfer(rng, fit, beta, c_pred):
    """Fit `c` once on the causal score; does it correct unrelated scores?

    Reports the fitted `c` and the theory-only `c_pred` side by side, because
    only the second is available without the genome-wide LD this is correcting
    for -- a correction you can only apply once you already have the answer is
    not a correction.
    """
    D_full, D_block, sd = fit["D_full"], fit["D_block"], fit["sd"]
    v = sd * beta
    ws = beta * sd
    c = float(ws @ (D_full - D_block) @ ws / (ws @ v) ** 2)

    def err(a, cc):
        true = float(np.sqrt(a @ D_full @ a))
        got = float(np.sqrt(max(a @ D_block @ a + cc * (a @ v) ** 2, 0.0)))
        return abs(got / true - 1) * 100.0

    rows = []
    for label, w in (
        ("causal beta", beta),
        ("noisy beta", beta + rng.normal(size=M) * beta.std()),
        ("sparse 10% of beta", beta * (rng.random(M) < 0.10)),
        ("random weights", rng.normal(size=M)),
    ):
        a = w * sd
        true = float(np.sqrt(a @ D_full @ a))
        rows.append((label, true, err(a, 0.0), err(a, c), err(a, c_pred)))
    return c, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--generations", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"1. Variance inflation over {args.generations} generations "
          f"(n={args.n:,}, {M} variants in {BLOCKS} blocks)")
    print(f"{'spouse r':>10}{'gen 0':>10}{'final':>10}{'1/(1-r h2)':>13}{'final h2':>11}")
    print("-" * 54)
    equilibria = {}
    for target in (0.0, 0.2, 0.4, 0.6):
        _, g, beta, hist = run(rng, n=args.n, generations=args.generations,
                               spouse_correlation=target)
        r = np.mean([h["spouse_r"] for h in hist])
        first, last = hist[0]["inflation"], hist[-1]["inflation"]
        h2f = hist[-1]["h2"]
        print(f"{r:>10.3f}{first:>10.3f}{last:>10.3f}"
              f"{1.0 / (1.0 - r * h2f):>13.3f}{h2f:>11.3f}")
        equilibria[target] = (g, beta, r, h2f)

    g, beta, spouse_r, h2_eq = equilibria[0.4]
    fit = r2_bias(rng, g, beta, h2=h2_eq)
    print("\n2. What PPB reports on that population (spouse r = 0.4)")
    print(f"   realized score SD            {fit['sd_realized']:8.3f}")
    print(f"   block-diagonal D predicts    {fit['sd_block']:8.3f}"
          f"   ({fit['sd_block'] / fit['sd_realized']:.3f}x)")
    print(f"   linkage equilibrium predicts {fit['sd_le']:8.3f}"
          f"   ({fit['sd_le'] / fit['sd_realized']:.3f}x)")
    print(f"\n   true individual-level R2     {fit['true_r2']:8.4f}")
    print(f"   PPB R2, block-diagonal D     {fit['r2_block']:8.4f}"
          f"   ({fit['r2_block'] / fit['true_r2']:.3f}x)")
    print(f"   PPB R2, full genome-wide D   {fit['r2_full']:8.4f}"
          f"   ({fit['r2_full'] / fit['true_r2']:.3f}x)")

    print("\n3. AM bias against score coverage")
    print(f"{'sampled':>9}{'captured':>11}{'SD understated by %':>22}")
    print("-" * 42)
    for p, captured, err in coverage_sweep(rng, fit, beta):
        print(f"{p:>9.2f}{captured:>11.3f}{err:>22.2f}")

    c_pred = predicted_c(fit, beta, spouse_r=spouse_r, h2=h2_eq)
    c, rows = rank_one_transfer(rng, fit, beta, c_pred)
    print("\n4. Rank-one correction (P3)")
    print(f"   c fitted on the causal score  {c:.4e}")
    print(f"   c from 1/(1 - r h2), no LD    {c_pred:.4e}"
          f"   ({c_pred / c:.3f}x)   using r={spouse_r:.3f}, h2={h2_eq:.3f}")
    print(f"{'test score':<22}{'true SD':>10}{'no fix %':>10}"
          f"{'c fitted %':>12}{'c theory %':>12}")
    print("-" * 66)
    for label, true, raw, fit_err, pred_err in rows:
        print(f"{label:<22}{true:>10.3f}{raw:>10.2f}{fit_err:>12.2f}{pred_err:>12.2f}")


if __name__ == "__main__":
    main()
