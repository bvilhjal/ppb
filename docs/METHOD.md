# PPB method specification

Status: specification (binding). Revised 2026-07-25.

Symbols are defined in [`NOTATION.md`](NOTATION.md) and are not redefined here.
Equations carry the stable labels (M1)–(M6); see `NOTATION.md` §1 for the
scheme and §5 for the index tying each to its implementation and test.

This is the contract for the reimplementation. PPB is being rebuilt from the
theory in the source preprint, not ported from the historical notebook. The
legacy code and datasets are used only as a **reference oracle** to check that
this implementation reproduces the published numbers; no legacy logic is copied.

Source (foundation): Witteveen, Pedersen, Meijsen, Andersen, Privé, Speed,
Vilhjalmsson, *Publicly Available Privacy-preserving Benchmarks for Polygenic
Prediction*, bioRxiv 2022, doi:10.1101/2022.10.10.510645 (CC-BY). Results here are
re-derived rather than transcribed, and are labelled (M1)–(M6) in this project's
own scheme; they do not correspond to the preprint's numbering.

**Project focus: cross-ancestry portability.** The estimator below is
ancestry-agnostic in *form*; PPB's flagship application is measuring the R² of a
polygenic score in a **target ancestry B** by supplying the target-ancestry
moments `z_B`, `D_B` (the within-ancestry case is `A = B`). The cross-ancestry
derivation, per-ancestry inputs, gauge condition, finite-sample correction, and
failure modes are specified in **`docs/CROSS_ANCESTRY.md`**; this file specifies
the shared core estimator and its conventions.

## 1. Core estimator

**Prior art.** The identity below is not original to the source preprint and is
not claimed here. It is the **quasi-correlation** of Pattee & Pan (2020, *PLOS
Comput Biol* 16:e1008271), published explicitly as an out-of-sample PGS accuracy
metric, and the square of **lassosum's pseudovalidation** criterion
`βᵀr/√(βᵀRβ)` (Mak et al. 2017, *Genet Epidemiol* 41:469), where it tunes a
penalty rather than reporting an accuracy. PUMAS (Zhao et al. 2021) reaches the
same estimand by subsampling one GWAS. What the source preprint contributed is
the **shared public benchmark artifact** built on the identity; what this project
adds is the cross-ancestry framing, the failure-mode map, and the infrastructure.
See `docs/CROSS_ANCESTRY.md`, "Background".

### 1.1 The estimand

Fix a population. Its standardized genotype vector is `x` (length `M`, each entry
mean 0 and variance 1) and its standardized phenotype is `y` (mean 0, variance 1).
Write

    rho = E[x y]        the marginal correlations
    Sigma = E[x x^T]    the LD matrix

A polygenic score assigns the value `g = x^T w`. Since `var(g) = w^T Sigma w` and
`cov(g, y) = w^T rho`, the quantity PPB reports is

**(M1) Predictive accuracy of a linear score.**

    R^2(w) = corr(g, y)^2 = (w^T rho)^2 / (w^T Sigma w)

with `var_y` restored in the denominator when `y` is not standardized. Note what
`R^2(w)` is *not*: it is a property of a **fixed** `w` in a **named** population,
not of the procedure that produced `w`. Two populations give one score two
different accuracies, which is the entire subject of `CROSS_ANCESTRY.md`.

### 1.2 The identity: exact, and assumption-free

Suppose a single sample of `n` individuals supplies everything — genotypes `X`
(columns centered and scaled to unit variance), phenotype `y` (centered, scaled),
and the moments formed from them:

    z_hat = X^T y / n,     D_hat = X^T X / n

Then substituting into (M1) is not an approximation but an algebraic identity:

**(M2) In-sample identity.**

    (w^T z_hat)^2 / (w^T D_hat w) = corr(X w, y)^2      exactly

*Proof.* `Xw` is centered because the columns of `X` are, so
`cov(Xw, y) = (Xw)^T y / n = w^T z_hat` and `var(Xw) = (Xw)^T (Xw) / n = w^T D_hat w`,
while `var(y) = 1`. Divide. ∎

No probability enters. Nothing is assumed about how `w` arose, about
heritability, or about the genetic architecture. This is the statement
`tests/test_estimator.py` checks to `1e-10`, and it is the reason the method
needs no individual-level data: `z_hat` and `D_hat` are summary-level, and they
are sufficient. ("Privacy-preserving" here means individual-level-data-free — not
differential privacy, not immunity to membership inference; see
[`LIMITATIONS.md`](LIMITATIONS.md).)

### 1.3 The estimator: consistent under three hypotheses

PPB never operates in the regime of §1.2. In practice `z_hat` comes from a GWAS,
`D_hat` from a separate reference panel, and `w` from a third cohort entirely.
Then (M1) is no longer an identity; it is a plug-in estimator

    R^2_hat = (w^T z_hat)^2 / (w^T D_hat w)

and it estimates `R^2(w)` when three hypotheses hold:

- **(H1) One population.** `z_hat` estimates `rho` and `D_hat` estimates `Sigma`
  for the *same* population — the one whose accuracy is being reported.
- **(H2) Independence.** `w` is independent of the sampling noise in `z_hat` and
  in `D_hat`.
- **(H3) One gauge.** `w`, `z_hat` and `D_hat` express the same standardization
  of each variant (§4, and (X2)).

Under (H1)–(H3) both `w^T z_hat -> w^T rho` and `w^T D_hat w -> w^T Sigma w`, so
`R^2_hat -> R^2(w)`: the estimator is consistent. Two second-order facts follow,
and both are worth knowing before quoting a number.

**Bias.** Squaring a noisy numerator biases it upward. With
`var(w^T z_hat) ~= w^T Sigma w (1 - R^2) / N`,

    E[R^2_hat] - R^2(w) ~= (1 - R^2) / N   <=  1/N

so the absolute bias is at most `1/N` — 4e-6 at `N = 250,000`, negligible beside
any real R^2, but cheap to remove. That correction is (X3), specified and not yet
implemented.

**Variance.** The blocks of `D` are independent, so the sampling variability of
`R^2_hat` is estimable by a delete-one-block jackknife over the per-block products
of (M3) — (G2), which costs nothing because the sweep computes those products
anyway.

### 1.4 What each violated hypothesis costs

The value of stating (H1)–(H3) is that every documented failure mode of PPB is
one of them failing, with a known direction. This table is the map; the detail
lives where the third column points.

**Table 1. Violated hypotheses and their consequences.**

| what fails | mechanism | direction | where treated | measured |
|---|---|---|---|---|
| (H1) `D_hat` from the wrong population | `R^2_hat = R^2 · (w^T Sigma_B w)/(w^T Sigma_A w)` | either; downward when the wrong panel has more LD | (X1), `CROSS_ANCESTRY.md` | −3% to −65% in simulation, depending on how far the LD architectures diverge |
| (H1) `z_hat` from the wrong population | estimates that population's accuracy instead | overstates transfer | `CROSS_ANCESTRY.md` §impossible | `1/portability − 1`; +58% at the demo's `r_g = 0.8` |
| (H1) `z_hat` mis-scaled by `c` | `R^2_hat -> c^2 R^2_hat`; nothing cancels it | as `c^2`; genomic control deflates | [`LIMITATIONS.md`](LIMITATIONS.md) | 1.5–2× low on GIANT targets, (G5) |
| (H2) `w` fitted on the target's noise | numerator gains `gamma q` | **upward**, without limit | (O1), [`OVERLAP.md`](OVERLAP.md) | up to 30× on real data (T2D 0.509 vs 0.044) |
| (H2) `w` fitted on the LD panel's noise | `w` overfits that panel, so `w^T D_hat w` is *overstated* | downward | §3, `experiments/figure_s1.py` | −9.0% using training-set LD |
| (H3) per-variant gauge mismatch | scale factors `c_j` do not cancel | either | (X2) | small within EUR, unbounded under admixture |
| support(`w`) ≠ support(`z_hat`) | a *different*, restricted score is evaluated | usually downward | [`LIMITATIONS.md`](LIMITATIONS.md) | 88.6–92.1% support on the GIANT/GLGC rows |

### 1.5 Two consequences of the ratio form

**It is invariant to a global rescale of `w`, and to nothing else.** Replacing `w`
by `c w` multiplies numerator and denominator by `c^2`. This is why per-allele PGS
Catalog weights in trait units give a correct `R^2` with no rescaling — and why
(M5)'s MSE, which lacks that invariance, does not. It is *not* invariance to a
rescale of `z_hat`: `z` appears only in the numerator, so its scale passes
through squared. That asymmetry is the whole content of row three of Table 1.

**Outside §1.2 it is not bounded by 1.** In the in-sample regime `R^2_hat` is a
squared sample correlation and cannot exceed 1. Once numerator and denominator
come from different samples — every real use — nothing constrains the ratio, and
ppb does not clamp it. A value near or above 1 is therefore a *diagnostic that
one of (H1)–(H3) has failed*, not a very good score.

### 1.6 The evaluation procedure

The estimator is one line of algebra; an evaluation is not. Three variant tables
have to be reconciled before the line can be written down, and the order in which
that happens is part of the specification rather than an implementation detail.

**Algorithm V** (*eValuate a score from summary-level data*). Given PGS weights
`w` with their own variant table, target summary statistics with theirs, and a
block-diagonal LD reference `D` with a third, this algorithm computes the
`R^2_hat` of §1.3 together with the diagnostics (G2) and (G3). The reference's
table is the frame: the other two are brought to it, and nothing is ever brought
to them.

**V1.** [Convert the GWAS.] Form `z_j = t_j/sqrt(t_j^2 + n_j - 2)` by (M4) from
each variant's reported `(beta_j, se_j, n_j)`, using a per-variant `n_j` wherever
the file carries one — a single uniform `N` biases `R^2_hat` downward, by up to
42% when the sizes really do vary (`experiments/per_variant_n.py`). For a
case/control GWAS `n_j` is the effective size `4/(1/n_case + 1/n_ctrl)`, and what
comes out is on the observed scale at a case fraction of one half (§5).

**V2.** [Harmonize to the reference.] Match the weight and summary-statistic
tables to the reference's on normalized `(chrom, pos)`; negate the value on an
allele swap or a strand flip; drop strand-ambiguous palindromes. Record the
matched fractions — they are the evidence that this step did what it claims.

**V3.** [Fix the gauge.] If `w` is on the per-allele dosage scale, multiply each
`w_j` by the target cohort's empirical genotype standard deviation `sd_j`. If it
is already on the standardized scale that `D` represents, do nothing. There is no
third possibility, and the caller must declare which of the two applies. This
step is where (H3) is discharged, and (X2) is what it discharges.

**V4.** [Restrict to the joint support.] Let `J` be the set of reference variants
matched by *both* other tables, and set `w_j <- 0`, `z_j <- 0` for every `j`
outside `J`. A variant carrying a weight but no target association is not
evidence of no association; leaving it in the denominator alone would evaluate a
score nobody submitted. Record `|J|`.

**V5.** [Sweep the blocks.] For each block `b` accumulate `u_b <- w_b^T z_b` and
`v_b <- w_b^T D_b w_b`, the latter by the int8 kernel of §2. This is the only
pass over the LD reference that anything below needs.

**V6.** [Accumulate.] Report `R^2_hat <- (sum_b u_b)^2 / (var_y * sum_b v_b)`,
which is (G1), and `MSE <- var_y - 2 sum_b u_b + sum_b v_b`, which is (M5).
Refuse the evaluation if `sum_b v_b <= 0` — and refuse it also if any single
`v_b < 0`, because one indefinite block hiding among 430 sound ones deflates the
denominator and inflates `R^2_hat` without disturbing the total.

**V7.** [Diagnose.] From the same `u_b` and `v_b`, with no second pass, form the
delete-one-block jackknife (G2) and the sign-flip null (G3). Read `R^2_hat`
against `null_mean` rather than against zero. ∎

Steps V1–V4 are bookkeeping and V5–V7 are arithmetic, and that is not a
hierarchy. Two rows of Table 1 — the per-variant gauge mismatch and the support
mismatch — are V3 and V4 failing; and the mis-scaled `z_hat` that is this
project's largest *measured* error enters at V1, where the reported
`(beta, se, n)` triple is taken at face value. The arithmetic has never been the
problem.

**Running time.** V5 costs `O(sum_b m_b^2)`; V2 is linear in `M` on average (the
reference's position index is a hash table); everything else is linear. On the
shipped reference the constants invert that ordering — a full 22-chromosome sweep
of a 922,538-variant score spends more time in V2 than in V5 (§2, "On-disk LD
store"), which is worth knowing before optimizing the quadratic form.

**Implemented in** `ppb.evaluate` (V2–V6 for a single bundle) and
`scripts/regenerate_results.py` (V1–V7, genome-wide, one chromosome at a time).

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
**reimplemented independently from the published scheme** — `src/ppb/ld_backend.py` is
original MIT-licensed code, not ported from ldpred3 (which cannot even be imported
into the Python 3.14 env). The whole estimator only ever needs two reductions over
`D`:

    numerator   = w^T z            (a plain dot product)
    denominator = w^T D w          (a quadratic form)

so `D` never needs to be materialised densely. `D` is stored **block-diagonal**
(recombination-aware blocks — ldpred3's `optimal_ld_blocks`, Privé 2022; ppb
itself takes caller-supplied blocks) and each block uses one of two int8
representations from `ldpred3/ld_repr.py`:

- **D8** — `round(corr * 127)`, dequantised by `/127`, represented either as a
  full-square `DenseLDInt8` block or a losslessly packed-upper-triangle
  `PackedDenseLDInt8` block. The v2 reference uses the packed form, halving the
  int8 payload without changing the represented matrix.
- **LR8** — an int8 low-rank factor with `R ~= U U^T`, `U` shape
  `(m, r)`, rows unit-norm so the LD diagonal is 1. ldpred3 selects this for large
  blocks (>= ~1500). **ppb does not implement it** (removed 2026-07-31) — the
  measured trade-off is in "On-disk LD store" below.

The block quadratic form is then, per block `b`:

- D8 block:  `w_b^T D_b w_b`  over the int8 block (square or packed triangle);
- (ldpred3's) LR8 block: `s = U_b^T w_b` (length `r`), then `w_b^T D_b w_b = s^T s = ||s||^2`.

**(M3) Block-diagonal accumulation.**

    w^T D w = sum_b w_b^T D_b w_b

This is O(sum k_b * r_b) time and int8 (~1 byte/entry) memory — the efficiency
win. It is also what makes the block diagnostics of `REAL_DATA.md` free: the
per-block `u_b` and `v_b` of Table 4 in [`NOTATION.md`](NOTATION.md) are
computed on the way to the two totals, so (G2) and (G3) need no second pass.

**PSD is representation-specific.** A low-rank `R = U U^T` is positive
semi-definite by construction (`w^T D w = ||U^T w||^2 >= 0`). D8 does not:
rounding a PSD correlation matrix entrywise can introduce negative eigenvalues.
Three checks, in decreasing strength:

1. **Exact**, at load: blocks with `m <= 512` are eigendecomposed and rejected
   if materially indefinite. Cubic cost, so limited to the small tail.
2. **Detection**, at conversion: larger blocks get a Lanczos estimate of
   `lambda_min` (`ppb.ld_backend.min_eig_upper_bound`). The smallest Ritz value
   bounds the true minimum from *above*, so this can prove a block indefinite and
   can never prove one PSD — the useful direction, since the failure it guards
   against is an indefinite block silently deflating the denominator. It runs in
   `write_ldref`, not on every read: O(m^2 * iters), about 11 s at the largest
   shipped block (m = 17,304).
3. **Per block**, at evaluation: `BlockDiagonalLD.quad` rejects any block whose
   quadratic form is negative. A check on the total alone would let one bad block
   hide among 430 good ones, and the direction of that error inflates R^2.

Large D8 blocks still carry no *proof* of PSD; use a low-rank factor
(`LowRankLD`, PSD by construction) when that guarantee is
required. No denominator is silently clamped.

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
numba `@njit(parallel=True)` kernels in `src/ppb/_kernels.py` (the same scalar-loop
int8 sweep pattern ldpred3 uses, written independently — no code copied).

**Implemented in `src/ppb/ld_backend.py`:** `DenseLDInt8` and
`PackedDenseLDInt8` (D8, with a diagonal that dequantises to exactly 1).
Square D8 uses about one eighth the float64 storage; packed D8 uses about one
sixteenth. The PSD-by-construction fallback is the float `LowRankLD`.

### On-disk LD store: what is wired up, and why LR8 is not

The `.npz` LD-reference format (`src/ppb/ldref.py`) is versioned:

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

**Table 2. Measured LD-reference storage layouts.**

| layout | in memory | on disk | read (chr22) | note |
|---|---|---|---|---|
| v1 square, raw | 10.41 GB | 10.61 GB | 0.070 s | the original store |
| **v2 packed, raw** | **5.21 GB** | **5.40 GB** | **0.034 s** | 2.000x smaller *and* 2x faster to read |
| v2 packed, compressed | 5.21 GB | ~1.8 GB* | 0.086 s | `compress=True`; for distribution |

Memory and disk figures are measured over all 22 chromosomes; *the compressed
total is extrapolated from chr22 (25.2 -> 8.1 MB, 3.1x on top of packing).

Packing is lossless — only the redundant lower triangle is dropped — and the
packed kernel is parallel over rows where the square one is serial (~6x faster
`quad` at m = 2000). End to end, a full 22-chromosome sweep with a real PGS
(922,538 variants) ran 48.3 s -> 38.6 s; the gain is diluted because
`harmonize_to` dominates that loop, not the quadratic form.

`w^T D w` is *not* bit-identical across the two layouts, because the packed
kernel sums each off-diagonal pair once and doubles it. Measured: at most **11.3
machine epsilon** per chromosome on random weights, and **2.2e-16 relative** on
the real genome-wide height denominator. That is four orders of magnitude below
the 0.02% int8 error already in the reference (next section), so repacking is
numerically a non-event — but it is not exactly zero, so a regenerated registry
would differ in the last digit or two.

### How much does int8 quantisation move the published number?

Measured genome-wide against the float bigsnpr source (diagonal forced to 1 in
both, so this isolates quantisation), for the six real PGS Catalog scores of
`docs/REAL_DATA.md`:

**Table 3. Genome-wide D8 quantisation error.**

| trait | error in `w^T D w` | error in R² |
|---|---:|---:|
| T2D | +0.0196% | −0.0196% |
| BMI | +0.0122% | −0.0122% |
| BrCa | +0.0018% | −0.0018% |
| CAD | +0.0012% | −0.0012% |
| height | −0.0018% | +0.0018% |
| LDL | −0.0027% | +0.0027% |

Two things matter here.

**The genome-wide error is ~25x smaller than the per-block error.** Individual
blocks show ~0.1-0.2%; summing 431 blocks over 22 chromosomes averages
independent rounding down to a mean of 0.005% and a worst case of 0.02%. An
error budget quoted per block badly overstates what reaches the published R².

**It is not common-mode, so it does not cancel.** The signs differ (height and
LDL negative, the rest positive) and the spread across scores (sd 0.008%)
exceeds the mean (0.005%). So the intuition that a shared multiplicative bias
cancels in a ratio of quadratic forms — `experiments/cross_ancestry.py`'s
`w^T D_A w / w^T D_B w` — or in a ranking of scores on one reference is
**wrong**. What saves both is magnitude, not cancellation: the worst pairwise
ratio error is 0.022%, so only scores whose true R² differ by less than that
could swap order. Note that the cross-ancestry ratio takes its two denominators
from *different* references, where the errors are independent draws rather than
a shared bias, so expect ~sqrt(2) times the within-reference figure.

**LR8 is not implemented, as a backend or a storage format (removed
2026-07-31).** Measured on this reference, the retained rank fraction at 99%
variance is essentially size-invariant (`r/m` ~ 0.42-0.51 from m = 220 to
m = 17,304), so LR8@0.99 would come to ~3.9 GB — only **1.34x** better than the
lossless packed triangle — while costing ~0.45% error in `w^T D w` on real PGS
weights against D8's own 0.12%. At 99.9% variance it is *larger* than the
triangle. There is also an LR8 error floor of ~0.25% from the PSD clamp, row
renormalization and factor quantisation that no retained-variance setting
removes. Revisit only if a distribution-size requirement forces below the
packed triangle.

**Oracle vs. production banding — a deliberate deviation to validate.** The
preprint's published numbers use a plain cM-window banded `D` (non-PSD, and the
source of its documented small overestimation). The PPB production evaluator will
instead use the block-diagonal D8 representation. These are not identical
approximations, so the golden-result reproduction must (a) first match the
paper's banding to reproduce its numbers as the oracle check, then (b) show the
D8 path agrees within a declared tolerance and document any systematic
difference. Do not silently substitute the D8 path for the paper's banding when
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
  `experiments/pc_adjustment.py` (`residualize`/`adjust`, mirroring `pldsc`'s
  covariate projection `X~ = P_C X`; `principal_components` derives the PCs).
  Forming `z` and `D` from the PC-adjusted genotypes/phenotype removes
  population-structure confounding from the estimated R².
- Allele alignment: `w` and `z` must be harmonized to the same effect allele; a
  sign convention and a variant-key (build, chr:pos:a1:a2) must be fixed.
  Implemented in `src/ppb/harmonize.py` (`VariantTable`, `harmonize_to`) as a
  bigsnpr-`snp_match`-style pass, mirroring `ldpred3.harmonize`: match by
  normalized `(chrom, pos)`, flip the value sign on allele swaps and strand
  flips (reverse-complement, indel-aware), and drop strand-ambiguous
  palindromes. `ppb.evaluate` composes harmonization with the estimator.
- Per-variant sample size — **(M4)**: `z_j = t_j/√(t_j²+n_j−2)`, implemented by
  `src/ppb/sumstats.py` (`standardized_marginal(beta, se, n)`), which recovers the
  standardized marginal correlation per variant. Assuming a uniform
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

**(M5) Summary-statistic mean squared error.**

    MSE = (1/N) || y - X w ||^2
        = var_y - 2 w^T z + w^T D w

Unlike (M1), this is **not** invariant to a rescale of `w`, so it is meaningful
only when the weights carry an absolute scale. Ordinary PGS Catalog weights are
in trait units and theirs is not; `EvaluationResult.mse_interpretable` records
which case a given evaluation is in.

### Binary traits are on a third scale again

For a case/control GWAS, `standardized_marginal` is handed an **effective**
sample size, `4/(1/n_case + 1/n_ctrl)` — the size of a balanced study with
equivalent power. (M1) then approximates R² on the observed 0/1 scale **at a
case fraction of one half**, which is neither the accuracy in the population nor
the accuracy on the underlying liability. Converting to the liability scale is
the standard rescaling of [Lee et al.
2012](https://doi.org/10.1002/gepi.21614):

**(M6) Observed-to-liability rescaling.**

    R²_liability = R²_observed · K²(1−K)² / (φ(t)² P(1−P)),    t = Φ⁻¹(1−K)

`K` is the population prevalence and `P` the case fraction the observed
statistic was computed at — one half here, and passing a study's true case
fraction instead would describe a statistic PPB did not compute. PPB ships **no
implementation** of (M6): binary metrics are a v1.0 item and registry values
stay on the observed scale. (Table 4 was produced by a liability-threshold
simulation that no longer ships; the formula above is the record.)

The factor is not a small adjustment. At `K = P = 0.5` it is exactly **π/2**,
because dichotomizing a continuous liability discards information and the
observed scale *understates*. At `K = 0.01, P = 0.5` it is **0.55**, because a
balanced sample of a rare disease is enormously enriched and the observed scale
*overstates*.

**Table 4. What (M6) recovers**, from a liability-threshold simulation that
ascertains a balanced case/control sample exactly as a study would. The
liability-scale R² is known by construction.

| `K` | true | observed | rescaled | rescaled error |
|---:|---:|---:|---:|---:|
| 0.50 | 0.040 | 0.025 | 0.039 | 2.5% |
| 0.20 | 0.040 | 0.031 | 0.041 | 2.5% |
| 0.05 | 0.040 | 0.046 | 0.039 | 3.0% |
| 0.01 | 0.040 | 0.067 | 0.037 | 7.6% |
| 0.01 | 0.160 | 0.239 | 0.132 | 17.4% |

(M6) is a **first-order result for small R²**, applied to a statistic that is
itself an approximation, so the two stack. In the regime this registry occupies
— binary R² of 0.025–0.044 — it lands within about 8% across every prevalence
tested. The last row is the honest boundary: a large R² on a rare trait is still
much better rescaled than raw, and is not calibrated.

**Prevalence is external knowledge**, absent from the summary statistics and
specific to trait *and population*. (M6) is monotone in it, so a liability-scale
number carries whatever uncertainty the assumed `K` does — and for a
cross-ancestry evaluation it is a second place, alongside `z` and `D`, where the
target population must be named.

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

## 8. Exercises

Ratings follow [`README.md`](README.md), "Notes on the exercises". Answers begin
below the last question; nothing here is left to the reader.

**1.** `[00]` The estimator is invariant to a global rescale of exactly one of
`w`, `z_hat`, `D_hat`. Which one, and what happens under a global rescale of each
of the other two?

**2.** `[10]` Exhibit summary-level inputs `w`, `z_hat`, `D_hat` for which
`R^2_hat > 1`. Which of (H1)–(H3) does your example violate?

**3.** `[M15]` The proof of (M2) uses that the columns of `X` are centered and
that `var(y) = 1`. Where does it use that the columns are scaled to unit
variance? What does that scaling buy, then?

**4.** `[M20]` Verify §5's claim that the factor in (M6) is exactly `pi/2` when
`K = P = 1/2`, and say which way the observed scale errs.

**5.** `[M20]` Negating every weight in one LD block flips `u_b` and leaves `v_b`
alone. Why? Why does that make (G3) an *exact* null rather than a simulated one,
and what breaks if `D` carries off-block entries?

**6.** `[M25]` Derive §1.3's `E[R^2_hat] - R^2(w) ~= (1 - R^2)/N` from
`var(w^T z_hat) ~= w^T Sigma w (1 - R^2)/N`. Which term does the derivation drop,
and what does (X3) do about it?

**7.** `[20]` §2 finds the genome-wide D8 quantisation error about 25 times
smaller than the per-block error. Account for the factor, and state the condition
under which the argument would fail.

**8.** `[40]` Implement (X3). Explain why it is rated `[40]` and not `[20]`,
given that the formula is one subtraction.

**9.** `[50]` §2 leaves large D8 blocks with no *proof* of positive
semi-definiteness: the Lanczos scan can refute, never certify. Find a block
representation that is PSD by construction, no larger than the packed triangle,
and no more costly in genome-wide `R^2` than D8's measured 0.02%.

## Answers to the exercises

**1.** `w`. Replacing `w` by `c w` multiplies numerator and denominator by `c^2`.
Replacing `z_hat` by `c z_hat` sends `R^2_hat` to `c^2 R^2_hat` — `z` appears in
the numerator alone — and replacing `D_hat` by `c D_hat` sends it to
`R^2_hat / c`. Only the middle one happens in practice, and it is row three of
Table 1.

**2.** Take `D_hat = I` and `w = z_hat`; then
`R^2_hat = (z_hat^T z_hat)^2 / (z_hat^T z_hat) = ||z_hat||^2`, which exceeds 1 as
soon as the summary statistics have norm above one — two variants at
`z_j = 0.8` will do. It violates (H2), since `w` was chosen as a function of
`z_hat`. It is also the mechanism behind the leaderboard warning in
[`../results/schema.md`](../results/schema.md): the ratio is maximized at
`w ∝ D^{-1} z`, so a public bundle makes a high score a linear solve.

**3.** Nowhere. Replace `X` by `X C` for a positive diagonal `C`: then
`z_hat -> C z_hat` and `D_hat -> C D_hat C`, so the ratio evaluated at `w` on the
rescaled data equals the ratio evaluated at `C w` on the original — the identity
is *covariant* under a change of gauge rather than dependent on one, which is
also what `corr(X C w, y)^2` says on the right-hand side. Unit scaling is a
reporting convenience: it makes `D_hat` a correlation matrix and `z_hat` a vector
of marginal correlations. What (H3) demands is therefore only that `w`, `z_hat`
and `D_hat` use the *same* `C`; (X2) is what fixes `C` when the weights arrive
from another population, and step V3 of Algorithm V is where it is applied.

**4.** `t = Phi^{-1}(1/2) = 0` and `phi(0) = 1/sqrt(2 pi)`, so the factor is
`K^2 (1-K)^2 / (phi(t)^2 P (1-P)) = (1/16) / ((1/(2 pi)) (1/4)) = pi/2 ~= 1.5708`.
It exceeds 1, so the observed scale **understates**: dichotomizing a continuous
liability discards information, and the rescaling puts it back.

**5.** `u_b = w_b^T z_b` is linear in `w_b` and `v_b = w_b^T D_b w_b` is
quadratic, so negation flips the first and fixes the second. Because `D` is
*exactly* block-diagonal, `sum_b v_b` is the same for all `2^n_blocks` sign
patterns; the whole null family therefore shares the observed denominator, and
its mean `sum_b u_b^2 / sum_b v_b` is available in closed form with no
resampling. With off-block entries the cross terms `w_b^T D_bb' w_b'` change sign
whenever exactly one of the two blocks is flipped, the denominator moves with the
pattern, and the null stops being exact.

**6.** Hold the denominator at `w^T Sigma w`. Then
`E[(w^T z_hat)^2] = (w^T rho)^2 + var(w^T z_hat)`, and dividing through gives
`R^2 + (1 - R^2)/N`. The derivation drops the denominator's own sampling
variability, which enters at the same order — this is why the statement carries
`~=` and not `=`. (X3) removes the leading term directly, subtracting
`w^T D_B w / N_B` from the squared numerator before the division.

**7.** The per-block rounding errors are independent and of both signs, so their
sum grows like `sqrt(n_blocks)` while the total `sum_b v_b` grows like
`n_blocks`; the relative error therefore falls like `1/sqrt(n_blocks)`, and
`sqrt(431) ~= 20.8` is the right size for the measured factor. The argument fails
if the errors are common-mode — and §2 measures that they are not, the signs
differing across the six scores. That same fact is why the error does *not*
cancel in a ratio of two denominators, so magnitude rather than cancellation is
what saves the cross-ancestry ratio.

**8.** The formula is indeed one subtraction. The work is in what has to be true
around it. `N_B` is per-variant on a meta-analysis target, where no single sample
size exists ([`LIMITATIONS.md`](LIMITATIONS.md)), so the "sample size the
estimator used" that [`../results/schema.md`](../results/schema.md) records has to
be the same quantity that enters the correction. A corrected value also cannot
replace a published one without the (G2) standard error beside it, since the
correction is `~1/N` and the jackknife SE is routinely larger. Tracked as a v0.1
completion criterion in [`../FINISHING_PLAN.md`](../FINISHING_PLAN.md).

**9.** Open. The obvious candidate, LR8, satisfies the first two conditions and
fails the third: measured on this reference it is 1.34x smaller than the packed
triangle at 99% retained variance, but costs ~0.45% error in `w^T D w` against
D8's own 0.12%, with a ~0.25% floor from the PSD clamp, row renormalization and
factor quantisation that no retained-variance setting removes. §2 records the
measurements and the condition — a distribution-size requirement below the packed
triangle — under which it should be reconsidered anyway.
