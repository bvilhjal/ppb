# PPB method specification

Status: draft, 2026-07-17

This is the contract for the reimplementation. PPB is being rebuilt from the
theory in the source preprint, not ported from the historical notebook. The
legacy code and datasets are used only as a **reference oracle** to check that
this implementation reproduces the published numbers; no legacy logic is copied.

Source: Witteveen, Pedersen, Meijsen, Andersen, Privé, Speed, Vilhjalmsson,
*Publicly Available Privacy-preserving Benchmarks for Polygenic Prediction*,
bioRxiv 2022, doi:10.1101/2022.10.10.510645 (CC-BY). Equation numbers below are
re-derived here and must be reconciled with the supplementary derivation once it
is obtained.

## 1. Core estimator

The performance measure is the squared Pearson correlation between the observed
phenotype and the polygenic-score prediction, expressed using only summary-level
inputs:

    R^2 = (w^T z)^2 / (w^T D w)

Symbols (all on standardized scales):

- `w`  — length-`M` vector of PGS weights being evaluated (the submission).
- `X`  — `N`-by-`M` standardized genotype matrix; each variant has mean 0,
         variance 1 across the target cohort.
- `y`  — length-`N` standardized target phenotype (mean 0, variance 1), after
         covariate adjustment (see §4).
- `D = (1/N) X^T X`  — `M`-by-`M` LD (genotype covariance/correlation) matrix.
- `z = (1/N) X^T y`  — length-`M` vector of marginal association summary
         statistics of `y` on each variant.

`M` ranges from ~1e5 to several million; `N` up to ~3.6e5 in the real benchmark.

### Derivation sketch (to be reconciled with the supplement)

Prediction is `p = X w`. With standardized `y` (var 1):

    R^2 = corr(p, y)^2 = cov(p, y)^2 / (var(p) var(y))
        = ( (1/N) w^T X^T y )^2 / ( (1/N) w^T X^T X w * 1 )
        = (w^T z)^2 / (w^T D w).

The privacy-preserving claim rests on this identity: a method developer needs
only `D` and `z` (both summary-level), never individual `X` or `y`, to evaluate a
weight vector `w`.

## 2. Exact vs. banded LD

- **Exact:** with the full `D`, the formula is exact — equal to the
  individual-level R^2 up to floating point. This is the equation-level test.
- **Banded (default):** the full `D` is too large to compute and share, so it is
  approximated by zeroing covariance outside a genetic-distance window. The
  published default is **4 cM**. Alternatives characterized in the paper:
  - 2 cM window overestimates R^2 by ~1.9% on average (worse for more polygenic
    architectures: +2.82% for the most polygenic vs +0.70% for the least).
  - ldetect LD blocks overestimate by ~12.4% on average (avoid as default).
  - Overestimation shrinks rapidly as the window grows; 4 cM is the accuracy/size
    trade-off used for the released benchmark.

The banding scheme (window in cM, or block definition) is a versioned parameter
of the benchmark and must be recorded with every result.

## 3. LD reference regimes

Three references for `D`, with their expected behavior (target of the Figure S1
reproduction, on synthetic data first):

- **Test data as LD reference** — exact estimates (within rounding).
- **Training data as LD reference** — biased R^2 estimates.
- **Independent LD reference** (samples in neither training nor test) — unbiased.

The released benchmark uses the validation set (a random 10K subset) for LD and
hyper-parameter selection only, not for estimating final performance.

## 4. Conventions to pin (verify against oracle before claiming a match)

- Genotype standardization: per-variant mean 0, variance 1 (allele-frequency
  based). Confirm exact scaling and how monomorphic/low-MAF variants are handled.
- Phenotype: residuals after adjusting for sex, age, and 10 principal components,
  then standardized (per the Privé et al. setup the paper follows).
- Allele alignment: `w` and `z` must be harmonized to the same effect allele; a
  sign convention and a variant-key (build, chr:pos:a1:a2) must be fixed.
- Per-variant sample size: the paper uses `1/N` scaling; confirm whether a single
  `N` or per-variant `n_j` is used for `z` and `D`, and how missingness enters.
- Variant set: HapMap3 (paper reports 1,117,493 variants; genome build to confirm
  — likely GRCh37 given the era and UKB pipeline).

## 5. Secondary measure (same inputs)

Mean squared error is computable from the same summary-level inputs. For
standardized `y` and predictor `p = X w`:

    MSE = (1/N) || y - X w ||^2
        = var(y) - 2 w^T z + w^T D w
        = 1 - 2 w^T z + w^T D w     (with standardized y).

## 6. Numerical tolerance

- **Equation-level (exact D, synthetic data):** summary-statistic R^2 must equal
  individual-level `corr(Xw, y)^2` to within `1e-10` relative error. This is a
  correctness test of the identity and the code, and must pass exactly.
- **Banded-LD (synthetic data):** reproduce the *sign and rough magnitude* of the
  published window-size biases (e.g. 4 cM near-zero deviation; 2 cM small positive
  deviation) rather than an exact number.
- **Golden result (real data):** one named published value — the natural
  candidate is Supplementary Figure S1 (LD-reference simulation) — reproduced
  within a tolerance declared before the run. Absolute-number reproduction of
  Figure 2 / Table 2 depends on recovering the real UK Biobank-derived datasets.

## 7. Reference benchmark facts (for the eventual real-data comparison)

- Cohort: UK Biobank, N = 362,320; validation = random 10K subset.
- 8 outcomes: Asthma, Breast cancer (BRCA), Coronary artery disease (CAD), Major
  depression (MDD), Prostate cancer (PRCA), Rheumatoid arthritis (RA), Type 1
  diabetes (T1D), Type 2 diabetes (T2D).
- Methods benchmarked in the paper: PRS-CS, SBayesR, LDpred2, lassosum.
- Known failure mode: MDD prediction R^2 is overestimated (~+4.5% at large
  windows), hypothesized to stem from assortative mating; a validation target for
  the "known failure modes" work, not a bug to eliminate.
