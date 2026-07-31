# Experiments

Status: record. Symbols and labels: [`../docs/NOTATION.md`](../docs/NOTATION.md).

Validated demonstrations, each encoded as a test. Some reproduce within-ancestry
results from Witteveen et al. (2022) — `figure_s1.py`, `benchmark_methods.py`,
`pc_adjustment.py`, `per_variant_n.py`; `pumas_agreement.py` tests a PUMAS-style
dense Gaussian extension (Zhao et al. 2021); and **`cross_ancestry.py` is this
project's new cross-ancestry portability method, which is NOT from that
European-only paper.** These are not
part of the installable `ppb` package; run them from the repo root with the `ppb`
env's Python.

## `figure_s1.py` — LD-reference experiment (Supplementary Figure S1)

Reproduces, on simulated data, the paper's finding about which LD reference gives
an accurate summary-statistic `R²`:

- **test-set LD → exact** (it is algebraically the individual-level `R²`),
- **training-set LD → biased** (weights derived from the training samples overfit
  the training LD),
- **independent LD → unbiased** (an independent panel from the same population).

Run:

```bash
python experiments/figure_s1.py --n-phenos 1000
```

Observed (1000 simulated phenotypes, h² ∈ [0.1, 0.9], m=300, n=2000, seed=0):

**Table 1. LD-reference behavior in the Figure S1 simulation**

| PGS       | LD ref | mean % bias | corr(est, true) |
|-----------|--------|------------:|----------------:|
| causal    | test   |      −0.00  |          1.0000 |
| causal    | train  |      +0.05  |          0.9949 |
| causal    | indep  |      −0.06  |          0.9946 |
| marginal  | test   |      −0.00  |          1.0000 |
| marginal  | train  |    **−8.99**|          0.9959 |
| marginal  | indep  |      −0.40  |          0.9954 |
| pT        | test   |      −0.00  |          1.0000 |
| pT        | train  |    **−4.87**|          0.9956 |
| pT        | indep  |      −0.29  |          0.9952 |

The training-LD bias appears only for data-derived weights (marginal, p+T) and is
negligible for the true causal effects, which are independent of the samples —
the expected mechanism. The result is encoded as an assertion in
`tests/test_figure_s1.py`.

### Scope

This is the **qualitative** reproduction on a simplified Gaussian genotype model
(block-AR(1) population LD), which validates the estimator and the LD-reference
behavior without individual-level data. Reproducing the paper's **absolute**
numbers (its Figure 2 / Table 2) additionally requires recovering the real
UK Biobank-derived benchmark datasets — see `FINISHING_PLAN.md`, Gate B.

## `benchmark_methods.py` — method benchmark (Figure 1 / Table 1 style)

Simulates realistic **diploid 0/1/2 genotypes** (ldpred3-inspired: latent AR(1)
haplotypes thresholded at MAF quantiles), runs four PGS methods across
polygenicity levels, and checks that PPB's summary-statistic R^2 (independent LD
reference, exact or low-rank-approximated) agrees with the individual-level R^2.

The `lowrank@` rows below are float `LowRankLD` factors from `ppb.lowrank_ld` at
the stated retained-variance fraction, so what they measure is eigen-truncation
error with no quantisation in it. They are *not* the int8 LR8 representation of
[`../docs/METHOD.md`](../docs/METHOD.md) §2, which ppb does not implement — an
earlier version of this script keyed them `lr8@` and so implied otherwise.

Methods: `causal` (oracle true effects), `marginal` (GWAS betas), `pT`
(p-value-thresholded), `inf` (LDpred-infinitesimal / ridge).

Run:

```bash
python experiments/benchmark_methods.py --n-reps 20
```

Observed (m=400, n=2500, h²=0.5, 20 reps × 3 architectures):

**Table 2. Exact and compressed-LD agreement with individual-level accuracy**

| LD reference | Pearson | Spearman | mean % bias |
|--------------|--------:|---------:|------------:|
| exact        |  0.975  |  0.974   |    +0.19    |
| lowrank@0.99 |  0.976  |  0.974   |    −0.20    |
| lowrank@0.95 |  0.976  |  0.975   |    −1.91    |

Method ranking (mean R², individual-level vs PPB-exact) — correctly preserved (see
below).

## `pc_adjustment.py` — principal-component adjustment removes structure confounding

The paper residualizes the phenotype on sex, age, and 10 PCs before forming
summary statistics. This experiment shows why it matters. Two subpopulations
(Balding-Nichols fst) with a phenotype driven **only by ancestry** (no genetics)
are spuriously "predicted" by a polygenic score built on the structured data —
classic stratification. Residualizing genotypes and phenotype on the top PCs
(`adjust`, mirroring `pldsc`'s covariate projection) before forming `z` and
`D` removes it.

Run:

```bash
python experiments/pc_adjustment.py --n-reps 10
```

Observed (mean over replicates):

**Table 3. Effect of PC adjustment in the structure simulation**

| scenario            | R² unadjusted | R² PC-adjusted |
|---------------------|--------------:|---------------:|
| null + confound     |      ~0.09    |     ~0.0003    |
| genetic, no confound|      ~0.11    |     ~0.08      |

PC adjustment drives the stratification-induced spurious R² to ~0 while
preserving genuine genetic prediction — encoded in `tests/test_covariates.py`.
The per-draw spurious R² is noisy (a 2-population axis is low-rank), so the
result is averaged over replicates.

## `pumas_agreement.py` — PUMAS-style repeated learning compared with PPB

PPB uses a genuinely held-out target cohort. The PUMAS-style path instead draws
paired pseudo-training and pseudo-validation summary statistics from one GWAS,
refits each data-derived score with `fit(z_train)`, and evaluates it on the paired
`z_val` (`subsample_sumstats` / `pumas_r2`, both in this file). An independent fixed score is
accepted through a separate API so weights trained on the full input GWAS cannot
masquerade as independent.

The sampler uses the signal-dependent full-LD covariance (E1).

**(E1) Dense Gaussian moment covariance**

    V = var_y D + z_full z_fullᵀ.

This is a dense Gaussian extension of the moments in Zhao et al. (2021), with
the observed full-sample signal plugged in; it is not their LD-pruned,
per-SNP-standard-error implementation. The default finite-validation bias
correction is exact within this plug-in working model only for independent fixed
weights. For refitted weights the default is the raw pseudo-validation statistic;
the conditional correction is available only as an explicit approximation.

Run:

```bash
python experiments/pumas_agreement.py
```

Observed:

**Table 4. Seeded repeated-learning comparison**

| arch      | method   | individual | PPB (exact) | PUMAS  |
|-----------|----------|-----------:|------------:|-------:|
| sparse    | causal   |   0.5091   |   0.5091    | 0.5040 |
| sparse    | marginal |   0.3200   |   0.3200    | 0.3008 |
| sparse    | pT       |   0.3611   |   0.3611    | 0.3482 |
| polygenic | causal   |   0.5066   |   0.5066    | 0.5269 |
| polygenic | marginal |   0.3482   |   0.3482    | 0.3221 |
| polygenic | pT       |   0.3230   |   0.3230    | 0.3111 |

PPB with exact target cross-products equals the individual-level R² by
construction. In this seeded run the PUMAS-style values are within 0.027 of the
individual repeated-learning averages. That is a useful agreement check, not a
claim of bit-exact PUMAS reproduction. The remaining plug-in, dense-Gaussian,
Monte Carlo, fitted-weight bias, and binary-trait limitations are explicit in
this file's header notes; for a binary phenotype this statistic is not liability-scale R².
Encoded in `tests/test_pumas.py`.

## `per_variant_n.py` — per-variant sample sizes; uniform N biases R²

In a meta-analysis each variant has its own sample size `n_j`. The standardized
marginal correlation is recovered per variant with (M4)
(`ppb.standardized_marginal`).

**(M4) Per-variant standardized marginal correlation**

    z_j = t_j / √(t_j² + n_j − 2).

Using a single uniform `N` instead shrinks every low-`n` variant and biases R²
**downward**.

Run:

```bash
python experiments/per_variant_n.py
```

Observed (individual-level R² = 0.332):

**Table 5. Per-variant versus uniform sample-size conversion**

| n_j range   | individual | per-variant n | uniform N |
|-------------|-----------:|--------------:|----------:|
| all = N     |    0.3323  |    0.3323     |  0.3323   |
| [0.50N, N]  |    0.3323  |    0.3292     |  0.2448   |
| [0.25N, N]  |    0.3323  |    0.3317     |  **0.1939** |

Per-variant conversion recovers the truth; assuming uniform N underestimates R²
by up to ~42% when sample sizes vary. Encoded in `tests/test_sumstats.py`. (This
is why PUMAS's Eq. 20 carries per-SNP N/SE terms, and why real summary-statistic
bundles should ship per-variant `n`.)

## `cross_ancestry.py` — measuring PRS portability across ancestries

The estimator is ancestry-agnostic in *form*: to get the R² of an
ancestry-A-trained PGS in a target ancestry B, feed it **ancestry-B** target
sumstats `z_B` and a **B-matched** LD reference `D_B`. Two Balding-Nichols
ancestries, shared causal effects correlated at `r_g`. See
[`../docs/CROSS_ANCESTRY.md`](../docs/CROSS_ANCESTRY.md) for the derivation,
requirements, and the measure-vs-predict boundary.

Run:

```bash
python experiments/cross_ancestry.py --rg 0.8
```

Observed (m=500, n=20000, F_ST=0.25, h²=0.5):

**Table 6. Cross-ancestry portability and reference mismatch**

| | r_g = 1.0 | r_g = 0.8 |
|---|---|---|
| portability R²_B/R²_A | 0.998 | **0.648** |
| exact (z_B, test-B LD) | −0.000% | +0.000% |
| independent (z_B, indep-B LD) | −0.095% | −0.095% |
| mismatched (z_B, ancestry-A LD) | −3.02% | −3.02% |
| A-only (z_A, ancestry-A LD) | +0.06% | **+57.7%** |

With B target stats + B LD the estimate is exact/unbiased and recovers the
portability loss; ancestry-A LD biases it (the LD-form ratio 1.031); substituting
ancestry-A sumstats does not estimate R²_B at all. **Target-ancestry summary
statistics are irreducibly required.** The +57.7% is `1/portability − 1` — an
arithmetic consequence of the chosen `r_g`, not a measured effect size. It is
reported as the mean of the per-draw ratios, which Jensen's inequality puts above
the `1/0.648 − 1 = +54.4%` you get from the ratio of the means; see
[`../docs/CROSS_ANCESTRY.md`](../docs/CROSS_ANCESTRY.md).

**Read the mismatch row as conditional on the simulated LD.** Both ancestries
above share one LD architecture: Balding-Nichols differs their allele
frequencies, but `ppb.simulate` draws haplotypes with the same latent
correlation `rho**|i-j|` for both. That is why portability is 0.998 at
r_g = 1.0 — F_ST has no LD channel, so all loss in Table 6 comes from `r_g`.
`run()` takes `rho_b` / `block_size_b` to give B its own LD:

```bash
python experiments/cross_ancestry.py --ld-divergence
```

**Table 7. Wrong-ancestry-LD bias against the simulated LD (r_g = 1.0)**

| A LD | B LD | portability | mismatch % bias | LD ratio |
|---|---|---:|---:|---:|
| ρ 0.5, blocks 100 *(shipped)* | ρ 0.5, blocks 100 | 1.001 | −3.17 | 1.034 |
| ρ 0.8, blocks 100 | ρ 0.8, blocks 100 | 0.963 | −10.66 | 1.113 |
| ρ 0.9, blocks 100 | ρ 0.9, blocks 100 | 0.920 | −13.70 | 1.154 |
| ρ 0.9, blocks 100 | **ρ 0.6, blocks 100** | **0.780** | **−64.74** | 2.844 |
| ρ 0.9, blocks 100 | **ρ 0.6, blocks 50** | **0.771** | **−65.30** | 2.883 |

So the −3% headline is a property of the generator (AR(1) at ρ = 0.5 decays to
r = 0.06 in four variants; the real HM3+ blocks have a median of 1,901 variants),
and genuinely divergent LD produces real portability loss with `r_g` held at 1.
Both experiments are encoded in `tests/test_cross_ancestry.py`.

## `overlap_detection.py` — detecting/correcting training-target sample overlap

When the score's training data overlaps the target GWAS, `w` fits the shared
noise and the numerator `wᵀz` is inflated (the benchmark's Gate-D failure
mode). The retained detector now fits genuine target/reference signal scale and
shared-noise coupling jointly. Correction additionally requires a declared
trainer-sensitivity basis on the exact score support, block sampling-noise
variances, and stable identification. A known linear operator can supply that
basis; a rerunnable differentiable trainer can estimate it. Final weights alone
cannot, so the method fails closed as `basis_unavailable` instead of substituting
variant count. Method note: [`../docs/OVERLAP.md`](../docs/OVERLAP.md).

Run:

```bash
python experiments/overlap_detection.py
```

Observed (blockwise ridge trainer, Hutchinson basis, heterogeneous LD blocks,
two replicates):

**Table 8. Basis-aware overlap diagnostics by genetic architecture**

| architecture | overlap | fit status | γ̂/γ_true | R² naive | R² corrected | independent anchor |
|---|---:|---|---:|---:|---:|---:|
| diffuse (20% causal) | 100% | `weak_identification` | 0.76 | — | refused | — |
| sparse (1% causal) | 0% | `not_detected` | — | 0.056 | none issued | 0.041 |
| sparse (1% causal) | 25% | `correctable` | 0.84 | 0.158 | **0.050** | 0.047 |
| sparse (1% causal) | 100% | `correctable` | 0.99 | 1.094 | **0.062** | 0.038 |

**Identification is decided by the architecture, not by the code.** The design
has two positive columns — the reference signal and the trainer basis — that
both grow with block size, so separating them needs signal variation *at fixed
block size*. A diffuse trait has almost none and is correctly refused; a sparse
one has plenty, and the same code recovers the coupling and brings a statistic
inflated nearly thirtyfold back to within ~1.6× of its honest anchor. Partial
overlap lands on the anchor, and the null issues nothing.

An earlier version of this experiment ran only the diffuse corner *with a
constant basis* — a marginal trainer over equal-sized blocks makes
`q_b = tr(D_b)` the block size — and recorded the resulting refusal as the
method's operating limit. It was the least identifiable configuration available.

The Hutchinson basis (`estimate_overlap_basis` in this file, `docs/OVERLAP.md`
(O4) — it does not ship in the package, which keeps only the fail-closed
`ppb.OverlapBasis` marker) is
checked against the exact `tr(Phi_b'K_b)` of the linear trainer it is estimating
(0.2–0.6% error). A p+T trainer is refused at the basis: its local Jacobian is
locally stable but misses its own selection response, which the perturbation
sweep detects.

## `transferability.py` — LD-based reweighting does not improve portability (negative result)

Asks whether the two LD matrices `D_A` (discovery) and `D_B` (target) can be
used to move a PGS onto more transferable variants. They cannot. A per-variant
LD-concordance score (E2; LD-only, no phenotype) cannot distinguish a
discordant *tag* from a discordant *causal* variant, so reweighting by it removes
signal.

**(E2) Per-variant LD concordance**

    s_j = cos(D_A row_j, D_B row_j).

Maximizing the PPB statistic over `w` instead has the optimum (E3) — a
ridge/BLUP refit in B, i.e. score construction, not reweighting.

**(E3) Target-informed optimum**

    argmax_w (wᵀz_B)² / (wᵀD_Bw) ∝ D_B⁻¹z_B.

Recorded so the question is not re-litigated. Method note:
[`../docs/TRANSFERABILITY.md`](../docs/TRANSFERABILITY.md).

Run:

```bash
python experiments/transferability.py
```

Observed (F_ST = 0.3, m = 500), change in realized `R²_B` vs the naive score:

**Table 9. Effect of proposed LD-based reweighting rules**

| reweighting | R²_B vs naive |
|---|---|
| LD-only shrinkage `w·s^γ` | −3% to −9% |
| LD-only pruning (drop low-`s`) | −18% to −29% |
| target-informed efficiency prune (*in-sample*) | +15% (overfitting artifact) |
| causal-effect oracle (`w = β_B`) | +20–90% (the real headroom) |

Encoded in `tests/test_transferability.py`.

---

Method ranking from `benchmark_methods.py` (mean R², individual-level vs
PPB-exact) — correctly preserved:

**Table 10. Method ranking in the benchmark simulation**

| method   | individual-level | PPB |
|----------|-----------------:|----:|
| causal   | 0.501 | 0.501 |
| inf      | 0.425 | 0.425 |
| pT       | 0.337 | 0.338 |
| marginal | 0.328 | 0.329 |

So PPB estimates each method's accuracy to ~0.001 and ranks them correctly
(oracle > LDpred-inf > p+T ≈ marginal) — the benchmark's core utility, achieved
without individual-level data. Low-rank truncation at 99% variance retention is
near-exact; more aggressive compression (95%) introduces a small, expected bias.
Encoded as assertions in `tests/test_benchmark.py`.

## `score_distribution.py` — benchmark of (P1)–(P2)

How accurate are the predicted score moments, and when does the normal
percentile stop being trustworthy? Four measurements against simulated
individuals: moment accuracy across LD strength and score density (SD within
0.2%; ignoring LD costs up to 13.8%), tail calibration as one variant takes over
the variance (the nominal 1% tail holds 5.4% of the cohort at
`max_variance_share = 0.68`), a structured cohort (pooling deflates the SD by
4.4%, `inbreeding` recovers half), and cost (~0.45× an `evaluate` — a wall-clock
ratio, so machine-dependent, unlike the rest).

Run: `python experiments/score_distribution.py`. Numbers are recorded in
[`../docs/SCORE_DISTRIBUTION.md`](../docs/SCORE_DISTRIBUTION.md) and pinned by
`tests/test_score_distribution.py`.

## `assortative_mating.py` — AM inflates variance where a block-diagonal `D` cannot see it

Mates a simulated population on its phenotype for ten generations (haplotypes,
meiosis with recombination, rank-pairing to a target spouse correlation) and
asks what breaks. The equilibrium variance inflation tracks the classical
`1/(1 − r h²)` across spouse correlations 0 to 0.6; at 0.4, PPB's R² reads
**1.336×** the individual-level truth with a block-diagonal reference and
**1.000×** with full genome-wide LD. (P3), a rank-one correction whose scalar
comes from `r` and `h²` alone, takes the error from 13.5% to 1.4% without any
off-block covariance.

Run: `python experiments/assortative_mating.py`. Numbers are recorded in
[`../docs/SCORE_DISTRIBUTION.md`](../docs/SCORE_DISTRIBUTION.md) and pinned by
`tests/test_assortative_mating.py`.

## `z_calibration.py` — detecting a mis-scaled target `z`

The largest measured error in the project is the scale of the input `z`, not the
modelling: GIANT targets report 1.5–2× low. This asks whether LD-score
regression (C2) can find it. On 18,000 simulated variants with the project's own
block-diagonal LD scores, the intercept is **0.972 ± 0.041** at the null — no
fabricated deflation — and recovers an applied genomic-control `λ` to within 3%
across 1.3, 1.6 and 2.0. Consistency across variant counts shows the intercept
needs ~10⁵ variants, which is why the fit refuses a verdict without a jackknife.

Run: `python experiments/z_calibration.py`. Numbers are recorded in
[`../docs/CALIBRATION.md`](../docs/CALIBRATION.md) and pinned by
`tests/test_ldscore.py`.
