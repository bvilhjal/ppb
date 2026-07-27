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
  `ppb.evaluate`. Harmonization negates a weight, whereas a frequency would have
  to be replaced by its complement; conflating the two silently mis-scales (P1).

## Validation

`tests/test_score_distribution.py` checks the predicted moments against
simulated individuals rather than against the algebra. On a 200-variant cohort
in block LD the predicted SD is within **0.02%** of the realized spread;
predicting the same cohort with `D = I` lands **3.2%** out, so the LD term is
load-bearing and the test tolerance is set below it.
