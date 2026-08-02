# The distribution of a score in a population

Status: specification (binding). Symbols and labels:
[`NOTATION.md`](NOTATION.md). Results are labelled (P1)–(P2).

To report an individual's PGS as a standardized value or a percentile you need
the score's distribution in a reference population. The usual route scores a
panel of individuals and so needs individual genotypes. The first two moments do
not: they are fixed by allele frequencies and LD.

For per-allele weights `w` and dosages `g_j` in {0, 1, 2}, with `S = sum_j w_j g_j`:

**(P1) Score mean.**

    E[S] = 2 sum_j w_j f_j

**(P2) Score variance.**

    Var(S) = (w * sd)^T D (w * sd),   sd_j = sqrt(2 f_j (1 - f_j) (1 + F))

(P1) needs frequencies alone. (P2) is **the same quadratic form the estimator
already computes as its denominator** — `w^T D w` on the standardized gauge *is*
the score variance, so this and [`METHOD.md`](METHOD.md)'s (M2) share their
expensive half. Implemented as `ppb.score_distribution`.

## Why this is the tractable half of portability

[`CROSS_ANCESTRY.md`](CROSS_ANCESTRY.md) argues that `R²_B`'s **numerator** is a
functional of population B's *joint* genotype–phenotype distribution, so no
amount of A-side data determines it — one measurement is irreducible, a
phenotype measured in B. Its **denominator** is a functional of B's *genotype*
distribution alone.

(P1)–(P2) are exactly that denominator, and they are therefore obtainable for
any population with published allele frequencies and a matched LD panel, with no
phenotypes at all. They answer a different question from `R²`: not *how accurate
is this score in B* but *does the same raw score mean the same percentile in B*.
Those come apart, and the second is what breaks percentile-based reporting
across ancestries — both `mu` and `sigma` move with the population.

## What two moments can and cannot support

`D` gives exactly two moments. Per-variant third central moments are available
in closed form (`2f(1-f)(1-2f)`), and cross-block third moments vanish because
the blocks are independent, but within-block three-locus terms are not in a
correlation matrix and are not published. So a percentile rests on a normal
approximation.

That approximation is better founded than a bare appeal to the CLT. With a
block-diagonal reference `S` is **exactly** a sum of independent block
contributions — 431 of them in the shipped panel — so the CLT applies to a
genuine sum of independent terms, and its quality is governed by how evenly the
variance is spread. `ScoreDistribution.max_variance_share` reports the largest
block's share and is the number to read before quoting a tail: a score with one
dominating block, an APOE-sized effect, is a mixture whose tail is not normal.
Percentile claims live precisely there.

## Assortative mating moves the variance where `D` cannot see it

AM correlates trait-increasing alleles at **unlinked** loci, inflating the
additive genetic variance above its linkage-equilibrium value. A block-diagonal
reference sets exactly those covariances to zero, so (P2) understates the
variance — and since the same quadratic form is the estimator's denominator,
`R²` comes out too high. This is the mechanism behind the Major Depression
overestimate that [`LIMITATIONS.md`](LIMITATIONS.md) attributes to AM.

Measured in `experiments/assortative_mating.py`, mating a population on its
phenotype for ten generations at a spouse correlation of 0.4: the equilibrium
variance inflation is **1.377**, against the classical `1/(1 − r h²)` = 1.360.
The realized score SD is 18.35; the block-diagonal prediction is 15.87
(0.865×), barely better than ignoring LD entirely (0.846×). PPB's `R²` then
reads **1.336×** the individual-level truth, while full genome-wide LD
reproduces it exactly.

The distortion has structure, because AM correlates alleles *in proportion to
their effects*. The missing covariance is approximately rank one along
`v = sd ∘ beta`:

**(P3) Assortative-mating variance correction** — *specified, not implemented*.

    Var(S) ≈ w_sᵀ D_block w_s + c (w_sᵀ v)²,     c = (V_A − V_block) / V_LE²
    V_A = V_LE / (1 − r h²)

One scalar corrects every score on that reference. It matters that `c` is
*predictable* rather than merely fittable: a correction you can only calibrate
once you already own the genome-wide LD is no correction at all. Taking `r` and
`h²` from the literature and both variances from allele frequencies and the
block reference, `c` lands within 11% of the fitted value, which takes the
causal score's error from **13.5% to 1.4%**.

**How much reaches a real score.** The AM term is quadratic in `w_sᵀ v` while
the linkage-equilibrium term is linear in the score's size, so bias grows with
how much of the trait's effect-weighted variance a score captures: 0.5% at 5%
coverage, 3.9% at 25%, 15.6% at 100%. Real PGS are far from complete, which
bounds the damage. A score uncorrelated with the effect direction is not
distorted at all, and (P3) correctly leaves it alone — the control that shows
the structure is real and not generic variance inflation.

Everything here is simulation, (P3) needs the causal effect *direction* rather
than marginal estimates, and nothing in `ppb` applies it.

## Assumptions

- **Hardy–Weinberg.** `Var(g) = 2f(1-f)(1+F)`. The `inbreeding` parameter
  carries `F` for a structured or admixed population, but it does **not**
  correct `D`: admixture induces long-range LD that a block-diagonal
  within-ancestry panel cannot represent at all.
- **Panel match.** `f` and `D` must describe the individual's population, or
  both moments are wrong — and unlike `R²`, which is invariant to a global
  rescale of `w`, nothing here cancels.
- **Frequencies are not harmonized.** `allele_frequency` is supplied in
  reference order for the reference's own effect allele, as `genotype_sd` is in
  `ppb.evaluate`. For a submitted opposite effect allele, PPB uses
  `g_other = 2 - g_reference`: the sign change enters (P2), while the affine
  offset enters (P1). Reorienting the frequency itself would conflate these two
  operations.

## Validation

Checked against simulated individuals rather than against the algebra that
produced the formula: `tests/test_score_distribution.py` and the benchmark
`experiments/score_distribution.py`.

**Table 1. Moment accuracy**, 40,000 individuals, predicted from `f` and `D`
alone. "no LD" repeats the prediction with `D = I`. Four of the eight regimes the
benchmark runs, chosen to span the range of the "no LD" column.

| regime | SD error | SD error, no LD |
|---|---:|---:|
| weak LD, sparse 5% | 0.08% | 0.47% |
| moderate, dense | 0.02% | 2.76% |
| strong LD, sparse 5% | 0.19% | 9.59% |
| few big blocks, sparse 5% | 0.08% | 13.80% |

The mean is exact to under 10⁻⁴ SD everywhere. The SD is within **0.2%** in
all eight regimes — the worst is the 0.19% shown — and the LD term is worth up
to **13.8%**: it is not an optional refinement.

**Table 2. Tail calibration**, 200,000 individuals, one variant's weight `w[0]`
growing until its block takes over the variance. Each row reports the realized
percentage of the cohort above the *nominal* 99th and 99.9th percentile
thresholds. All eight rows of the run, at the script's own precision.

| `w[0]` | `max_variance_share` | above nominal 1% | above nominal 0.1% | worst percentile error |
|---:|---:|---:|---:|---:|
| 0 | 0.106 | 1.03% | 0.089% | 0.2 |
| 5 | 0.105 | 1.03% | 0.087% | 0.1 |
| 12 | 0.098 | 1.11% | 0.138% | 0.3 |
| 18 | 0.174 | 1.36% | 0.237% | 1.1 |
| 24 | 0.265 | 1.80% | 0.391% | 2.3 |
| 30 | 0.356 | 2.36% | 0.566% | 4.0 |
| 40 | 0.492 | 3.42% | 0.860% | 7.4 |
| 60 | 0.684 | 5.40% | 1.272% | 13.9 |

This is what makes `max_variance_share` actionable rather than decorative. Read
the first three rows together: until `w[0]` reaches about 12 the loaded variant's
block is not the largest one, the share stays near 0.10 whatever `w[0]` does, and
the differences between those rows are Monte Carlo noise rather than a trend.
From there the error is **monotone in the share**, and **the deeper the tail, the
earlier it fails**: at a share of 0.17 the 1% tail is only 1.4× over-populated
while the 0.1% tail is already 2.4× over-populated; by 0.49 they are 3.4× and
8.6×. Below ~0.1 the percentile is good to a fraction of a point across the whole
range. A score whose variance sits in one block does not have a normal tail, and
the tail is where the claims are.

**Table 3. A structured cohort**, two populations at `fst = 0.05`, pooled, using
a clean single-population LD panel — the realistic mistake.

| assumption | SD error |
|---|---:|
| no correction | −4.36% |
| `inbreeding = fst` | −2.00% |
| within one population (control) | +0.01% |

Pooling deflates the prediction by 4.4%, `F` recovers about half, and the same
score inside a homogeneous cohort is predicted exactly. The residual is the part
`F` cannot reach: structure changes `D`, and a within-ancestry panel does not
carry that.

**Cost.** On 120,000 variants in 431 int8 blocks, `score_distribution` runs in
about **0.45×** the time of a full `evaluate` (0.43× and 0.46× on two machines).
Unlike every other number in this document, that ratio is a wall-clock
measurement and does not reproduce exactly; what does reproduce is the reason for
it — `score_distribution` does one harmonization rather than two, and the two
calls share the single expensive quadratic form.
