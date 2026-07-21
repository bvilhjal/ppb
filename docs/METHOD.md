# PPB method specification

Status: draft, revised 2026-07-18 (cross-ancestry focus).

This is the contract for the reimplementation. PPB is being rebuilt from the
theory in the source preprint, not ported from the historical notebook. The
legacy code and datasets are used only as a **reference oracle** to check that
this implementation reproduces the published numbers; no legacy logic is copied.

Source (foundation): Witteveen, Pedersen, Meijsen, Andersen, Privé, Speed,
Vilhjalmsson, *Publicly Available Privacy-preserving Benchmarks for Polygenic
Prediction*, bioRxiv 2022, doi:10.1101/2022.10.10.510645 (CC-BY). Equation numbers
below are re-derived here.

**Project focus: cross-ancestry portability.** The estimator below is
ancestry-agnostic in *form*; PPB's flagship application is measuring the R² of a
polygenic score in a **target ancestry B** by supplying the target-ancestry
moments `z_B`, `D_B` (the within-ancestry case is `A = B`). The cross-ancestry
derivation, per-ancestry inputs, gauge condition, finite-sample correction, and
failure modes are specified in **`docs/CROSS_ANCESTRY.md`**; this file specifies
the shared core estimator and its conventions.

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

The summary-statistics-based (individual-level-data-free) property rests on this
identity: a method developer needs only `D` and `z` (both summary-level), never
individual `X` or `y`, to evaluate a weight vector `w`. ("Privacy-preserving" here
means individual-level-data-free, not differential privacy or immunity to
membership inference — see `LIMITATIONS.md`.)

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
of the benchmark and must be recorded with every result. In the results registry
this is the `ld_ref` field (`results/schema.md`), which names the reference and
therefore its block definition; ppb does **not** implement cM-window banding at
all — it consumes caller-supplied blocks, and the shipped reference uses
bigsnpr's 431 HM3+ blocks rather than a cM window. The window-size biases quoted
above are the source paper's, and are not a property of this implementation.

### LD representation (int8 block LD, reimplemented independently, with numba)

Instead of a raw cM-banded dense matrix, PPB uses the same compact int8 block-LD
scheme as the local `ldpred3` project (Privé, a co-author of the source preprint,
develops ldpred3, the working successor of the LDpred/bigsnpr lineage), but
**reimplemented independently from the published scheme** — `ppb/ld_backend.py` is
original MIT-licensed code, not ported from ldpred3 (which cannot even be imported
into the Python 3.14 env). The whole estimator only ever needs two reductions over
`D`:

    numerator   = w^T z            (a plain dot product)
    denominator = w^T D w          (a quadratic form)

so `D` never needs to be materialised densely. `D` is stored **block-diagonal**
(recombination-aware blocks — ldpred3's `optimal_ld_blocks`, Privé 2022; ppb
itself takes caller-supplied blocks) and each block uses one of two int8
representations from `ldpred3/ld_repr.py`:

- **D8** — a dense int8 block (`round(corr * 127)`, dequantised by `/127`).
  ldpred3's `PackedSymmetricInt8LD` stores only the upper triangle and
  memory-maps it; **ppb's `DenseLDInt8` stores the full square array and loads
  it into RAM** (see "Implementation gaps" below).
- **LR8** — `LowRankLDInt8`: an int8 low-rank factor with `R ~= U U^T`, `U` shape
  `(m, r)`, rows unit-norm so the LD diagonal is 1. ldpred3 selects this for large
  blocks (>= ~1500); ppb implements it as a backend but does not store it — the
  measured trade-off is in "On-disk LD store" below.

The block quadratic form is then, per block `b`:

- D8 block:  `w_b^T D_b w_b`  over the int8 block (square or packed triangle);
- LR8 block: `s = U_b^T w_b` (length `r`), then `w_b^T D_b w_b = s^T s = ||s||^2`.

Total `w^T D w = sum_b (block quadratic form)`. This is O(sum k_b * r_b) time and
int8 (~1 byte/entry) memory — the efficiency win.

**PSD by construction (correctness bonus).** A low-rank `R = U U^T` is positive
semi-definite, so `w^T D w = ||U^T w||^2 >= 0` always. A block-diagonal of PSD
blocks is PSD. This *removes* the negative-denominator failure that a raw
cM-banded truncation (non-PSD) would introduce, so no ad-hoc clamping is needed
on the production path.

ldpred3 additionally applies **linear shrinkage toward the identity** to large
blocks (`shrink_ld_blocks`: `D_a = (1-a) D + a I`, with an MP-motivated intensity
`a = min(max_shrink, intensity * k / n_ref)`; defaults `max_shrink=0.5`,
`intensity=1.0`). ppb does **not** mirror it, and should not: it is not a
Marchenko-Pastur estimator (there is no MP edge or deconvolution anywhere in
ldpred3 — an earlier revision of this document called it one, wrongly), its only
measured benefit is conditioning a Gibbs sampler at small `n_ref`, and ppb never
solves or iterates with `D`. At ppb's `n_ref = 362,320` the intensity works out
to `a` = 0.048 for the largest block and 0.0006 for the smallest — numerically a
no-op. It would also bias ppb's estimand in a known direction:
`w^T D_a w = (1-a) w^T D w + a ||w||^2`, which for LD-tagged weights deflates the
denominator and *inflates* R².

**Kernels: numba.** The block sweeps for `w^T D w` are implemented as original
numba `@njit(parallel=True)` kernels in `ppb/_kernels.py` (the same scalar-loop
int8 sweep pattern ldpred3 uses, written independently — no code copied).

**Implemented in `ppb/ld_backend.py`:** `DenseLDInt8` (D8, `round(corr*127)`,
diagonal dequantises to exactly 1) and `LowRankLDInt8` (LR8, int8 factor with a
global `scale` and per-row rescaling to restore the unit diagonal), plus
`quantize_lowrank`. Both are ~8x smaller than float64 and agree with the float
path to within quantisation (~1-2%); LR8 stays PSD, so `w^T D w >= 0`.

### On-disk LD store: what is wired up, and why LR8 is not

The `.npz` LD-reference format (`ppb/ldref.py`) is versioned:

- **v1** — every block a full `m x m` int8 square in `ld8`. What the converted
  HM3+ reference originally shipped.
- **v2** — adds `format_version`, `block_kind` and `block_offset`, and carries
  packed upper triangles in `ld8p`. Written only when a packed block is present,
  so square-only references stay byte-for-byte v1 and older readers keep working;
  every v2 array has a default reproducing the v1 parse exactly. `block_kind = 1`
  is reserved for an int8 low-rank factor, which nothing writes (see below).

`scripts/repack_ldref.py` converts v1 to packed v2. Measured on the shipped
reference (1,444,196 variants in 431 blocks; block sizes min 216, median 1,901,
mean 3,351, max 17,304), per chr22 and scaling with `sum_b m_b^2`:

| layout | genome-wide | read (chr22) | note |
|---|---|---|---|
| v1 square, raw | 10.4 GB | 0.070 s | the original store |
| **v2 packed, raw** | **5.2 GB** | **0.034 s** | 2.00x smaller *and* 2x faster to read |
| v2 packed, compressed | ~1.8 GB | 0.086 s | `compress=True`; for distribution |

Packing is lossless — only the redundant lower triangle is dropped — and the
packed kernel is parallel over rows where the square one is serial (~6x faster
`quad` at m = 2000). `w^T D w` is *not* bit-identical across the two layouts,
because the packed kernel sums each off-diagonal pair once and doubles it:
measured at **at most 17 machine epsilon (~4e-15 relative)**, eleven orders of
magnitude below int8 quantisation's own ~0.1% error, but not exactly zero.

**LR8 is implemented as a backend but deliberately not as a storage format.**
Measured on this reference, the retained rank fraction at 99% variance is
essentially size-invariant (`r/m` ~ 0.42-0.51 from m = 220 to m = 17,304), so
LR8@0.99 would come to ~3.9 GB — only **1.34x** better than the lossless packed
triangle — while costing ~0.45% error in `w^T D w` on real PGS weights against
D8's own 0.12%. At 99.9% variance it is *larger* than the triangle. There is also
an LR8 error floor of ~0.25% from the PSD clamp, row renormalization and factor
quantisation that no retained-variance setting removes. Revisit only if a
distribution-size requirement forces below the packed triangle.

**Oracle vs. production banding — a deliberate deviation to validate.** The
preprint's published numbers use a plain cM-window banded `D` (non-PSD, and the
source of its documented small overestimation). The PPB production evaluator will
instead use the block-diagonal LR8/D8 representation. These are not identical
approximations, so the golden-result reproduction must (a) first match the
paper's banding to reproduce its numbers as the oracle check, then (b) show the
LR8/D8 path agrees within a declared tolerance and document any systematic
difference. Do not silently substitute the LR8 path for the paper's banding when
claiming to reproduce a published value.

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
  then standardized (per the Privé et al. setup the paper follows). Implemented in
  `ppb/covariates.py` (`residualize`/`adjust`, mirroring `pldsc`'s covariate
  projection `X~ = P_C X`; `principal_components` derives the PCs). Forming `z` and
  `D` from the PC-adjusted genotypes/phenotype removes population-structure
  confounding from the estimated R² (see `experiments/pc_adjustment.py`).
- Allele alignment: `w` and `z` must be harmonized to the same effect allele; a
  sign convention and a variant-key (build, chr:pos:a1:a2) must be fixed.
  Implemented in `ppb/harmonize.py` (`VariantTable`, `harmonize_to`) as a
  bigsnpr-`snp_match`-style pass, mirroring `ldpred3.harmonize`: match by
  normalized `(chrom, pos)`, flip the value sign on allele swaps and strand
  flips (reverse-complement, indel-aware), and drop strand-ambiguous
  palindromes. `ppb.evaluate` composes harmonization with the estimator.
- Per-variant sample size: handled by `ppb/sumstats.py`
  (`standardized_marginal(beta, se, n)` → `r_j = t_j/√(t_j²+n_j−2)`), which
  recovers the standardized marginal correlation per variant. Assuming a uniform
  `N` when the true `n_j` vary biases R² downward (see
  `experiments/per_variant_n.py`), so summary-statistic bundles should carry
  per-variant `n`.
- Variant set: the source paper reports HapMap3, 1,117,493 variants. This
  implementation uses the **HapMap3+ (HM3+)** set from bigsnpr's precomputed
  European LD reference — **1,444,196 variants in 431 blocks, GRCh37** (with
  `pos_hg38` carried alongside), confirmed by reading the converted reference,
  not inferred. Results are therefore not on the same variant set as the paper's,
  which the golden-result comparison in §6 must account for.

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
- **Golden result (real data):** the paper's absolute numbers on the 8 real
  traits (Figure 2 / Table 2), reproduced within a tolerance declared before the
  run; this depends on recovering the real UK Biobank-derived datasets (Gate B)
  and is the outstanding anchor. Supplementary Figure S1 (an LD-reference
  *simulation*) is reproduced qualitatively on synthetic data
  (`experiments/figure_s1.py`) and is not the real-data anchor.

## 7. Reference benchmark facts (for the eventual real-data comparison)

- Cohort: UK Biobank, N = 362,320; validation = random 10K subset.
- 8 outcomes: Asthma, Breast cancer (BRCA), Coronary artery disease (CAD), Major
  depression (MDD), Prostate cancer (PRCA), Rheumatoid arthritis (RA), Type 1
  diabetes (T1D), Type 2 diabetes (T2D).
- Methods benchmarked in the paper: PRS-CS, SBayesR, LDpred2, lassosum.
- Known failure mode: MDD prediction R^2 is overestimated (~+4.5% at large
  windows), hypothesized to stem from assortative mating; a validation target for
  the "known failure modes" work, not a bug to eliminate.
