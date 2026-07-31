# Notation, conventions, and an index of results

Every symbol PPB uses is defined here, once. The other documents cite this one
rather than redefining anything, so a symbol means the same thing in all of them
— and where it does not, §4 says so explicitly instead of leaving the reader to
discover it.

§5 indexes every numbered result to the code that implements it and the test that
pins it. A claim in these documents that is not in that table is a claim nothing
checks.

## 1. The labelling scheme

Equations are labelled by document, not by position in a global sequence, so a
label is stable when a document grows and unambiguous when cited from elsewhere:

| prefix | document | subject |
|---|---|---|
| **M** | [`METHOD.md`](METHOD.md) | the core estimator and its conventions |
| **X** | [`CROSS_ANCESTRY.md`](CROSS_ANCESTRY.md) | the cross-ancestry application |
| **O** | [`OVERLAP.md`](OVERLAP.md) | training/target shared-noise detection |
| **P** | [`SCORE_DISTRIBUTION.md`](SCORE_DISTRIBUTION.md) | the score's distribution in a population |
| **C** | [`CALIBRATION.md`](CALIBRATION.md) | whether the target `z` is on the assumed scale |
| **G** | [`REAL_DATA.md`](REAL_DATA.md) | genome-wide accumulation and block diagnostics |
| **R** | [`../results/schema.md`](../results/schema.md) | the results registry |
| **E** | [`../experiments/README.md`](../experiments/README.md) | simulation demonstrations |

So (M2) is the core identity wherever it appears, and (O4) is the stochastic
overlap basis whether you meet it in `OVERLAP.md` or in a docstring. Tables keep
per-document numbering, since they are read where they sit.

## 2. Core symbols

All of these are on standardized scales unless stated otherwise (§3).

**Table 1. Quantities in the estimator.**

| symbol | meaning | where it comes from |
|---|---|---|
| `M` | number of variants | the LD reference's variant set |
| `N` | sample size of the cohort a statistic was computed in | the GWAS |
| `w` | length-`M` vector of PGS weights being evaluated | the submission |
| `X` | `N × M` standardized genotype matrix | never materialised; conceptual |
| `y` | length-`N` standardized, covariate-adjusted phenotype | never materialised; conceptual |
| `D = (1/N) XᵀX` | LD matrix: variant-by-variant genotype correlation | an LD reference panel |
| `z = (1/N) Xᵀy` | marginal association of each variant with the phenotype | a GWAS, via Table 2 |
| `var_y` | variance of `y` on the scale used to form `z`; 1 when `y` is standardized | declared |
| `R²` | the estimand (M1); its plug-in estimate is `R²_hat` | computed |

**Table 2. Turning a GWAS into `z`.**

| symbol | meaning |
|---|---|
| `β_j`, `se_j` | reported effect and standard error for variant `j` |
| `n_j` | sample size for variant `j` — per-variant in a meta-analysis, trait-level otherwise |
| `t_j = β_j / se_j` | the marginal test statistic |
| `z_j = t_j / √(t_j² + n_j − 2)` | the standardized marginal correlation, i.e. entry `j` of `z` |

**Table 3. Ancestries.** Subscripts, never bare letters.

| symbol | meaning |
|---|---|
| `A` | the *discovery* ancestry — where `w` was trained |
| `B` | the *target* ancestry — where accuracy is being measured |
| `z_B`, `D_B`, `w_B` | the corresponding quantity in ancestry B |
| `R²_A`, `R²_B` | accuracy in each ancestry |
| portability | `R²_B / R²_A` |
| `r_g` | cross-population genetic correlation of causal effects |
| `F_ST` | population differentiation between A and B |
| `f_j`, `sd_j` | allele frequency and genotype standard deviation of variant `j` |

**Table 4a. An individual's score** ([`SCORE_DISTRIBUTION.md`](SCORE_DISTRIBUTION.md)).

| symbol | meaning |
|---|---|
| `g_j` | one individual's dosage at variant `j`, in {0, 1, 2} |
| `S = Σ_j w_j g_j` | that individual's raw polygenic score |
| `F` | Wright's inbreeding coefficient: `Var(g) = 2f(1−f)(1+F)` |

**Table 4. LD blocks and their storage.**

| symbol | meaning |
|---|---|
| `b` | block index; `D` is block-diagonal, so `wᵀDw = Σ_b w_bᵀ D_b w_b` |
| `m_b` | number of variants in block `b` |
| `u_b = w_bᵀ z_b` | that block's contribution to the numerator |
| `v_b = w_bᵀ D_b w_b` | that block's contribution to the denominator |
| D8 | int8 dense block: `round(corr × 127)`, dequantised by `/127` |
| LR8 | int8 low-rank factor `U` with `D_b ≈ U Uᵀ`; PSD by construction (ldpred3's representation — not implemented in ppb, see `METHOD.md` §2) |
| `ρ` | rank retained by a low-rank factor |
| `λ_min` | smallest eigenvalue of a block; negative means the block is not PSD |

Note that `u_b` and `v_b` are the *only* per-block quantities anything downstream
needs. The block jackknife (G2) and the sign-flip null (G3) both consume exactly
these, which is why they cost no extra pass over the LD reference.

**Table 5. Shared training/target noise** (`OVERLAP.md`).

| symbol | meaning |
|---|---|
| `Φ` | the trainer's linear operator: `w = Φ z_train` |
| `K` | covariance template for one unit of shared estimation noise; usually `D` |
| `q_b = tr(Φ_bᵀ K_b)` | the trainer's sensitivity to that noise, per block — the *basis* |
| `s_b` | latent genuine signal in block `b` |
| `α` | multiplicative target/reference signal scale |
| `γ` | shared-noise coupling; `γ > 0` is the overlap signature |
| `σ²_Tb`, `σ²_Rb`, `c_TR,b` | sampling-noise variances of the block products, and their covariance |
| `U_T`, `Q_total` | target numerator and total basis over the exact score support |

## 3. Conventions

1. **Standardization.** Genotypes have mean 0 and variance 1 per variant;
   the phenotype has mean 0 and variance 1 after covariate adjustment. `D` is
   then a correlation matrix with unit diagonal, and `z` is a vector of marginal
   correlations.
2. **One gauge.** `w`, `z`, and `D` must share a single self-consistent
   standardization. The ratio is invariant to a *global* rescale of `w` but not to a
   per-variant one — see (X2).
3. **Effect allele.** Every table carries `(chrom, pos, a1, a2)` with `a1` the
   effect allele. Harmonization flips signs on allele swaps and strand flips and
   drops strand-ambiguous palindromes.
4. **Sample size.** `n_j` is the sample size *the estimator was given*, never a
   study's published headline N. Where it varies across variants, the median and
   range are recorded rather than a single number.
5. **Blocks tile.** Block index sets partition `[0, M)` exactly. Off-block LD is
   taken to be zero.

The three hypotheses that turn the identity (M2) into an estimator — **(H1)** one
population, **(H2)** `w` independent of the sampling noise, **(H3)** one gauge —
are stated in [`METHOD.md`](METHOD.md) §1.3, and §1.4 tabulates what violating
each one costs. Every failure mode elsewhere in these documents is one of those
three failing.

## 4. Where a symbol is overloaded

Four collisions survive, because the alternative is renaming a public interface
for the sake of a table. They are scoped, and they do not co-occur.

| symbol | meaning 1 | meaning 2 | how to tell |
|---|---|---|---|
| `F` | Wright's inbreeding coefficient (Table 4a) | `F_ST`, differentiation between A and B (Table 3) | `F_ST` always carries its subscript. They are related but not interchangeable: pooling two populations `F_ST` apart induces an `F` of about that size, which is why `inbreeding=fst` is the natural first guess for a pooled cohort — and why it only recovers half the error, since `F` does not correct `D` |
| `z` | the marginal-correlation vector (Table 1) | `sign_flip_null.z`, the block-coherence statistic (G3) | the statistic is a scalar and always appears as a named field |
| `r` | marginal correlation of one variant | retained rank of a low-rank factor | context: `r_j` is per-variant, `r` alone is a rank |
| `n` | per-variant sample size `n_j` | number of blocks, `n_blocks` | `n_blocks` is always spelled out |

The block-coherence statistic in particular is a poor name — in a genomics
package `z` means a GWAS z-statistic — and it is called `z` only because the
`SignFlipNull` field was named before this table existed. It is bounded by
`√n_blocks`, which no z-statistic is; read it as a coherence measure, not a
significance test.

One collision was removed rather than documented: the trainer operator in
`OVERLAP.md` was called `A`, which is also the discovery ancestry. It is now `Φ`.

## 5. Index of results

Every numbered equation, the module that implements it, and the test that holds
it to account.

**Table 6. Numbered results, their implementation, and their test.**

| result | statement | implemented in | pinned by |
|---|---|---|---|
| (M1) | the estimand `R²(w) = (wᵀρ)²/(wᵀΣw)` | — (a definition) | — |
| (M2) | in-sample identity: (M1) with sample moments equals `corr(Xw, y)²` **exactly** | `ppb.estimator.r2` | `tests/test_estimator.py` (to 1e-10) |
| (M3) | block-diagonal accumulation `wᵀDw = Σ_b w_bᵀ D_b w_b` | `ppb.ld_backend.BlockDiagonalLD` | `tests/test_ld_backend.py` |
| (M4) | `z_j = t_j / √(t_j² + n_j − 2)` | `ppb.sumstats.standardized_marginal` | `tests/test_sumstats.py` |
| (M5) | `MSE = var_y − 2wᵀz + wᵀDw` | `ppb.estimator.mse` | `tests/test_estimator.py` |
| (M6) | observed-to-liability rescaling for binary traits | *not implemented* (documented formula; implementation removed 2026-07-31) | — |
| (X1) | (M1) with target-ancestry moments `z_B`, `D_B` | `ppb.estimator.r2` (same code) | `tests/test_cross_ancestry.py` |
| (X2) | gauge self-consistency; `w_B,j = w_A,j · sd_B,j / sd_A,j` | `ppb.evaluate` (`weight_scale`) | `tests/test_cli.py` |
| (X3) | finite-sample numerator bias `≈ wᵀD_Bw / N_B` | *not implemented* | — |
| (O1) | shared-noise block model | `experiments/overlap_detection.py` (`fit_overlap`) | `tests/test_overlap.py` |
| (O2) | generalized Deming objective | `experiments/overlap_detection.py` (`_profile_eiv`) | `tests/test_overlap.py` |
| (O3) | linear-trainer basis `q_b = tr(Φ_bᵀ K_b)` | caller-supplied `OverlapBasis` | `tests/test_overlap.py` |
| (O4) | stochastic basis (Hutchinson GDF) | `experiments/overlap_detection.py` (`estimate_overlap_basis`) | `tests/test_overlap.py` |
| (O5) | exact-support numerator and total basis | `experiments/overlap_detection.py` (`fit_overlap`) | `tests/test_overlap.py` |
| (O6) | corrected signed numerator (experimental; not a registry contract) | `experiments/overlap_detection.py` (`correct_overlap_numerator`) | `tests/test_overlap.py` |
| (P1) | score mean from allele frequencies | `ppb.score_distribution` | `tests/test_score_distribution.py` |
| (P2) | score variance from frequencies and LD | `ppb.score_distribution` | `tests/test_score_distribution.py` |
| (P3) | assortative-mating variance correction | *specified, not implemented* | `tests/test_assortative_mating.py` |
| (C1) | LD score `l_j = Σ_k D_jk²` | `LDBackend.ld_scores` | `tests/test_ldscore.py` |
| (C2) | LD-score regression | `experiments/z_calibration.py` (`ldscore_regression`) | `tests/test_ldscore.py` |
| (C3) | implied `z` scale from the intercept | `experiments/z_calibration.py` (`ldscore_regression`) | `tests/test_ldscore.py` |
| (G1) | genome-wide accumulation of (M1) | `scripts/regenerate_results.py` | `tests/test_regenerate_results.py` |
| (G2) | delete-one-group block jackknife | `ppb.diagnostics.r2_block_jackknife` | `tests/test_diagnostics.py` |
| (G3) | block-sign-flip null | `ppb.diagnostics.sign_flip_null` | `tests/test_diagnostics.py` |
| (G4) | trait-swap negative control | `scripts/negative_controls.py` | `tests/test_negative_controls.py` |
| (G5) | external check against published individual-level accuracy | `scripts/anchor_validation.py` | `tests/test_anchor_validation.py` |
| (R1) | registry score metric | `scripts/regenerate_results.py` | `tests/test_results_registry.py` |
| (E1) | PUMAS moment covariance `V = var_y D + z zᵀ` | `experiments/pumas_agreement.py` | `tests/test_pumas.py` |

(X3) is the one numbered result with no implementation. It is a stated v0.1
completion criterion, its constant has been verified by simulation, and it is
listed here rather than quietly omitted.
