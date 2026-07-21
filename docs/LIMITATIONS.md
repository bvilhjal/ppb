# Scope, limitations, and privacy claims

This consolidates what PPB does and does not claim. It complements the estimator
specification in [`METHOD.md`](METHOD.md) and the project plan in
[`../FINISHING_PLAN.md`](../FINISHING_PLAN.md).

## What PPB is

A tool (and planned benchmark) that measures the **cross-ancestry portability** of
**linear polygenic scores** — the predictive R² of a PGS in a *target ancestry* —
from **summary-level information only** (target-ancestry GWAS summary statistics +
a matched LD reference), without individual-level test records:

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
  retained; near-zero at 99%). int8 quantisation adds ~1–2% error.
- **Training/target sample overlap — the dominant failure mode for a
  benchmark.** The estimator assumes `w` was trained independently of the target
  GWAS's noise. When it was not, the numerator is inflated additively — up to
  **15×** in the real-data demonstration (height R² 0.80 against an in-sample
  Pan-UKB target vs 0.21 against non-overlapping GIANT). It is detectable and
  approximately correctable from `(w, z, D)` alone for dense scores, but
  **undetectable for strongly shrunk / low-coverage scores**, which must be
  reported as upper bounds rather than corrected. Every result record must
  declare its overlap status. See [`OVERLAP.md`](OVERLAP.md).
- **The estimate is not bounded by 1.** `(wᵀz)²/(wᵀDw)` equals a squared
  correlation only when `z` and `D` describe the same sample as the phenotype.
  With a mismatched LD reference, overlap, or a `z` on an inconsistent gauge it
  can exceed 1; ppb does not clamp it. A value near or above 1 is a diagnostic
  that an assumption has failed, not a very good score.
- **Gauge (standardization) consistency.** `w`, `z` and `D` must share one
  self-consistent standardization; the ratio is invariant to a *global* rescale
  of `w` but not a per-variant one. The current real-data path uses the HWE
  scale `√(2f(1−f))` rather than empirical genotype SDs — acceptable for the
  within-ancestry EUR anchor, not for structured or admixed targets (see
  [`CROSS_ANCESTRY.md`](CROSS_ANCESTRY.md), "gauge self-consistency").
- **LD reference choice.** Test-set LD is exact; training-set LD is biased for
  data-derived weights; an independent same-population panel is unbiased
  (reproduced in `experiments/figure_s1.py`).
- **Trait-specific discrepancy.** The paper reports Major Depression R² is
  overestimated (~+4.5%), hypothesised to stem from assortative mating — a
  modelling limitation, not a bug.
- **Scope.** v0.1 is the **simulation-validated** cross-ancestry portability
  estimator plus the within-ancestry Witteveen anchor (continuous traits). The
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
reference's variants and effect-allele orientation (`ppb.harmonize`).

Population structure is controlled by residualizing genotypes and phenotype on
fixed covariates (sex, age, principal components) before forming `z` and `D`
(`ppb.covariates`). Adjustment removes the ancestry-aligned component of
prediction — spurious stratification *and* any genetic prediction mediated
through structure — so PC-adjusted R² is a conservative, within-structure
estimate. Choosing the number of PCs is left to the user; too few leaves residual
confounding, too many can absorb real signal.
