# Experiments

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
reference; exact or LR8-approximated) agrees with the individual-level R^2.

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
| lr8@0.99     |  0.976  |  0.974   |    −0.20    |
| lr8@0.95     |  0.976  |  0.975   |    −1.91    |

Method ranking (mean R², individual-level vs PPB-exact) — correctly preserved (see
below).

## `pc_adjustment.py` — principal-component adjustment removes structure confounding

The paper residualizes the phenotype on sex, age, and 10 PCs before forming
summary statistics. This experiment shows why it matters. Two subpopulations
(Balding-Nichols fst) with a phenotype driven **only by ancestry** (no genetics)
are spuriously "predicted" by a polygenic score built on the structured data —
classic stratification. Residualizing genotypes and phenotype on the top PCs
(`ppb.adjust`, mirroring `pldsc`'s covariate projection) before forming `z` and
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
`z_val` (`ppb.subsample_sumstats` / `ppb.pumas_r2`). An independent fixed score is
accepted through a separate API so weights trained on the full input GWAS cannot
masquerade as independent.

The sampler uses the signal-dependent full-LD covariance in Equation 1.

**Equation 1. Dense Gaussian moment covariance**

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
`ppb.pumas`; for a binary phenotype this statistic is not liability-scale R².
Encoded in `tests/test_pumas.py`.

## `per_variant_n.py` — per-variant sample sizes; uniform N biases R²

In a meta-analysis each variant has its own sample size `n_j`. The standardized
marginal correlation is recovered per variant with Equation 2
(`ppb.standardized_marginal`).

**Equation 2. Per-variant standardized marginal correlation**

    r_j = t_j / √(t_j² + n_j − 2).

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
ancestry-A sumstats does not estimate R²_B at all (+58%). **Target-ancestry
summary statistics are irreducibly required.** Encoded in
`tests/test_cross_ancestry.py`.

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

Observed (known marginal-score basis, five replicates per overlap fraction):

**Table 7. Seeded basis-aware overlap diagnostics**

| overlap | γ̂ | γ_true | fit status | correction |
|---:|---:|---:|---|---|
| 0% | −1.18e-05 | 0 | `weak_identification` | refused |
| 25% | 5.82e-05 | 6.25e-05 | `weak_identification` | refused |
| 100% | 1.91e-04 | 2.50e-04 | `weak_identification` | refused |

The overlap direction is visible, but the diffuse signal and basis remain too
collinear for a defensible correction (full-overlap VIF 2.76; condition number
8.9). The thresholded trainer is separately refused because its sensitivity
operator was not reconstructed. This is the point of fail-closed statistics:
"looks plausible" is not an identification theorem. Encoded in
`tests/test_overlap.py`.

## `transferability.py` — LD-based reweighting does not improve portability (negative result)

Asks whether the two LD matrices `D_A` (discovery) and `D_B` (target) can be
used to move a PGS onto more transferable variants. They cannot. A per-variant
LD-concordance score (Equation 3; LD-only, no phenotype) cannot distinguish a
discordant *tag* from a discordant *causal* variant, so reweighting by it removes
signal.

**Equation 3. Per-variant LD concordance**

    s_j = cos(D_A row_j, D_B row_j).

Maximizing the PPB statistic over `w` instead has the optimum in Equation 4 — a
ridge/BLUP refit in B, i.e. score construction, not reweighting.

**Equation 4. Target-informed optimum**

    argmax_w (wᵀz_B)² / (wᵀD_Bw) ∝ D_B⁻¹z_B.

Recorded so the question is not re-litigated. Method note:
[`../docs/TRANSFERABILITY.md`](../docs/TRANSFERABILITY.md).

Run:

```bash
python experiments/transferability.py
```

Observed (F_ST = 0.3, m = 500), change in realized `R²_B` vs the naive score:

**Table 8. Effect of proposed LD-based reweighting rules**

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

**Table 9. Method ranking in the benchmark simulation**

| method   | individual-level | PPB |
|----------|-----------------:|----:|
| causal   | 0.501 | 0.501 |
| inf      | 0.425 | 0.425 |
| pT       | 0.337 | 0.338 |
| marginal | 0.328 | 0.329 |

So PPB estimates each method's accuracy to ~0.001 and ranks them correctly
(oracle > LDpred-inf > p+T ≈ marginal) — the benchmark's core utility, achieved
without individual-level data. LR8 at 99% variance retention is near-exact; more
aggressive compression (95%) introduces a small, expected bias. Encoded as
assertions in `tests/test_benchmark.py`.
