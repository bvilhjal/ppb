# Scope, limitations, and privacy claims

Status: scope. Symbols and labels: [`NOTATION.md`](NOTATION.md).

This consolidates what PPB does and does not claim. It complements the estimator
specification in [`METHOD.md`](METHOD.md) and the project plan in
[`../FINISHING_PLAN.md`](../FINISHING_PLAN.md).

## What PPB is

A tool (and planned benchmark) that measures the **cross-ancestry portability** of
**linear polygenic scores** — the predictive R² of a PGS in a *target ancestry* —
from **summary-level information only** (target-ancestry GWAS summary statistics +
a matched LD reference), without individual-level test records:

This is (X1) — the core identity (M1) with target-ancestry moments:

    R²_B = (wᵀ z_B)² / (wᵀ D_B w),   with z_B, D_B from the target ancestry B.

The within-ancestry case (`A = B`) is the special case and validation anchor; it
reproduces the method of Witteveen et al. (2022, bioRxiv, CC-BY), which was never
peer-reviewed. The cross-ancestry direction is new to this project (the source
paper is European-only) and is **validated in simulation only** — not yet run on
real cross-ancestry data. Claims are at preprint strength. See
[`CROSS_ANCESTRY.md`](CROSS_ANCESTRY.md).

PPB **measures** portability given target-ancestry data; it does **not predict**
portability from discovery-ancestry data alone (that is a different method class).

## Privacy claims — deliberately narrow

- PPB is **summary-statistics-based** / **individual-level-data-free**. It shares
  LD and GWAS summary statistics, not individual genotypes or phenotypes.
- "Privacy-preserving" here does **not** mean differential privacy, cryptographic
  protection, or immunity to membership inference. No formal privacy analysis has
  been done; until one exists, prefer "summary-statistics-based" in technical
  claims.
- Released test-side summary statistics are effectively public labels. The
  current public artifact is therefore **not** leakage-resistant for a
  competitive leaderboard; a hidden-test benchmark would be a separate redesign
  (see the plan, Gate D).

## What PPB does not establish

Predictive **association** only. It does not establish calibration, clinical
utility, causality, or individual privacy.

## Known limitations and failure modes

- **LD approximation bias.** Coarse LD approximation biases the estimate. In the
  source paper, small cM windows overestimate R² (~+1.9% at 2 cM; ldetect blocks
  ~+12%); 4 cM is the accuracy/size trade-off. In this implementation, aggressive
  low-rank (LR8) compression slightly *underestimates* R² (≈−2% at 95% variance
  retained; near-zero at 99%). int8 quantisation of the shipped reference moves
  the **genome-wide** R² by at most **0.02%** (measured against the float source
  for six real scores) — individual blocks are ~0.1–0.2%, but summing 431 blocks
  averages independent rounding down. The residual is *not* common-mode across
  scores, so it does not cancel in ratios or rankings; it is simply small (worst
  pairwise ratio error 0.022%). The loader now validates block coverage, offsets,
  dtypes, annotations, packed diagonals, and low-rank definiteness. D8
  quantization can perturb positive semidefiniteness, however, and full PSD is
  not certified for every large D8 block. See [`METHOD.md`](METHOD.md).
- **Training/target sample overlap — the dominant failure mode for a
  benchmark.** The estimator assumes `w` was trained independently of the target
  GWAS's noise. When it was not, the numerator is inflated additively — by more
  than tenfold for T2D in the real-data demonstration (0.509 against an in-sample
  Pan-UKB target vs 0.044 against non-overlapping DIAGRAM). PPB detects and labels this
  rather than correcting it. A correction would need an independent reference
  GWAS of the same trait, and given one, evaluating the score against it is
  unbiased in a single line — so the condition that makes a correction valid is
  the condition that makes it unnecessary. In-sample values are published as
  upper bounds. Detection is a moment, not a participant record: shared
  stratification or relatedness produce the same signature.
  See [`OVERLAP.md`](OVERLAP.md).
- **Incomplete target-summary support changes the score being evaluated.** A
  missing target association cannot be treated as a zero while its weight stays
  in `wᵀDw`. PPB now restricts both quadratic forms to the exact joint support
  and records `metrics.n_variants_scored`. The GIANT/GLGC demonstrations retain
  only 88.6–92.1% of score weights, so those results apply to restricted scores,
  not the complete PGS Catalog scores.
- **Binary-trait output is not liability-scale R².** The beta/SE-to-correlation
  conversion for case/control GWAS yields a standardized summary-statistic
  approximation whose scale depends on the supplied effective sample size and
  regression model. It is useful as a diagnostic or within-protocol comparison,
  but it is not a calibrated liability-scale accuracy estimate.
- **PUMAS is a dense Gaussian extension, not a bit-exact reimplementation.** Each
  pseudo-training draw is now refitted and evaluated on its paired
  pseudo-validation draw using the signal-dependent full-LD moment covariance.
  The covariance still plugs in the observed full-sample signal and does not
  reproduce the paper's LD-pruned, per-SNP-standard-error implementation. The
  conditional finite-validation correction is exact within this plug-in working
  model only for weights independent of the pseudo split; applying it to refitted
  weights is an explicit approximation. Binary-trait output is less interpretable
  still.
- **The estimate is not bounded by 1**, and a value near or above 1 is a
  diagnostic that a hypothesis has failed rather than a very good score. Derived
  in [`METHOD.md`](METHOD.md) §1.5.
- **Gauge (standardization) consistency.** `w`, `z` and `D` must share one
  self-consistent standardization; the ratio is invariant to a *global* rescale
  of `w` but not a per-variant one. The current real-data path uses the HWE
  scale `√(2f(1−f))` rather than empirical genotype SDs — acceptable for the
  within-ancestry EUR anchor, not for structured or admixed targets (see
  [`CROSS_ANCESTRY.md`](CROSS_ANCESTRY.md), "gauge self-consistency").
- **The scale of `z` passes straight through, squared.** The invariance above
  protects `w` only. If the target summary statistics are off by a factor `c`,
  then `R² → c²R²` — `z` appears in the numerator alone and nothing cancels it.
  That scale is set entirely by the reported `(beta, se, n)` triple that
  `ppb.standardized_marginal` consumes, and real GWAS routinely move it:
  **genomic control** (applied at study *and* meta level by GIANT and GLGC)
  inflates `se` and deflates the result; meta-analysis SE handling and SPA/SAIGE
  standard errors for binary traits do the same; and the analyst's choice of
  `n_eff` moves it directly — `REAL_DATA.md` records a 1.2× swing for DIAGRAM
  T2D from that choice alone, on a number quoted to three significant figures.
  This is now **measured, not hypothesised**: against published individual-level
  accuracy for the same scores, LDL (a GLGC target) agrees to 2% while both
  **GIANT** targets come in 1.5–2× low, implying a `z` deflation of 1.22–1.44 —
  the right size for GIANT's genomic control, and not explained by score support
  or by the cohort difference (`REAL_DATA.md`, Table 3).
  PPB performs **no diagnostic** on the scale of its input `z` and cannot detect
  a mis-scaled target. Reported values should be read as conditional on the
  source study's processing; a mean-χ²/LD-score or heritability sanity check
  against the target is the natural guard and is not implemented.
- **The real-data path pairs an adjusted `z` with an unadjusted `D`.**
  [`METHOD.md`](METHOD.md) §4 specifies forming *both* `z` and `D` from
  covariate-adjusted genotypes. Real GWAS `z` are partial correlations (adjusted
  for age, sex, PCs, centre) while the shipped bigsnpr LD reference is an
  unadjusted genotype correlation matrix. Within a homogeneous EUR sample the
  discrepancy is small, but it is largest exactly where the flagship
  cross-ancestry application will lean on it — structured targets. Relatedly,
  `standardized_marginal` uses the no-covariate degrees of freedom `n − 2`
  rather than `n − k − 2`; immaterial at GWAS N.
- **A meta-analysis target has no single generating population.** For GIANT
  height the per-variant `N` runs 50.0k–253k across dozens of contributing
  cohorts, so each `z_j` is a marginal correlation in a *different* mixture of
  cohorts while `D` is one panel's LD. The registry records the distribution
  (`n_eff_basis`, `n_eff_range`) rather than a headline N, but the estimand is
  then only approximately "the squared correlation in a population" — there is
  no single population. This compounds with the `z`-scale item above, since
  cohorts contribute different amounts of genomic-control deflation.
- **LD reference choice.** Test-set LD is exact; training-set LD is biased for
  data-derived weights; an independent same-population panel is unbiased
  (reproduced in `experiments/figure_s1.py`).
- **Trait-specific discrepancy.** The paper reports Major Depression R² is
  overestimated (~+4.5%), hypothesised to stem from assortative mating — a
  modelling limitation, not a bug.
- **Scope.** v0.1 is the **simulation-validated** cross-ancestry portability
  estimator plus the within-ancestry Witteveen anchor (quantitative traits are the
  interpretable primary use; binary outputs remain approximate). The
  estimator is ancestry-agnostic — given target-ancestry summary statistics and a
  matched LD reference it measures cross-ancestry (portability) R² (see
  [`CROSS_ANCESTRY.md`](CROSS_ANCESTRY.md)) — but this is **validated in simulation
  only, not yet run on real cross-ancestry data**, and it cannot *predict*
  portability from discovery-ancestry data alone. Real non-European / admixed
  target cohorts, binary-trait calibration, and local-ancestry handling are v1.0
  work (see `FINISHING_PLAN.md`).

## Reproduction status

- **Reproduced (qualitative, on simulated data):** the LD-reference behaviour
  (Fig. S1) and cross-method concordance / ranking (Fig. 1 / Table 1 style) —
  see `experiments/`.
- **Not yet reproduced:** the paper's **absolute** numbers on the 8 real traits
  (Fig. 2 / Table 2). This requires recovering the real UK Biobank-derived
  benchmark datasets (plan, Gate B) and is the outstanding gate for tagging
  `v0.1.0`.

## Standardization assumptions

Genotypes and phenotype are standardized (mean 0, variance 1); with standardized
`y`, `var_y = 1`. Weights and summary statistics must be harmonized to the LD
reference's variants and effect-allele orientation (`ppb.harmonize`). The CLI
requires the input weight scale explicitly: dosage-scale weights need the target
bundle's empirical `genotype_sd`, while already standardized weights must be
declared as such. Evaluation uses the joint weight/summary-statistic support and
rejects non-finite values rather than mixing numerator and denominator supports.

Population structure is controlled by residualizing genotypes and phenotype on
fixed covariates (sex, age, principal components) before forming `z` and `D`
(`ppb.covariates`). Adjustment removes the ancestry-aligned component of
prediction — spurious stratification *and* any genetic prediction mediated
through structure — so PC-adjusted R² is a conservative, within-structure
estimate. Choosing the number of PCs is left to the user; too few leaves residual
confounding, too many can absorb real signal.
