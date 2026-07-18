# Experiments

Validated demonstrations, each encoded as a test. Some reproduce within-ancestry
results from Witteveen et al. (2022) — `figure_s1.py`, `benchmark_methods.py`,
`pc_adjustment.py`, `per_variant_n.py`; `pumas_agreement.py` demonstrates PUMAS
(Zhao et al. 2021); and **`cross_ancestry.py` is this project's new cross-ancestry
portability method, which is NOT from that European-only paper.** These are not
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

| scenario            | R² unadjusted | R² PC-adjusted |
|---------------------|--------------:|---------------:|
| null + confound     |      ~0.09    |     ~0.0003    |
| genetic, no confound|      ~0.11    |     ~0.08      |

PC adjustment drives the stratification-induced spurious R² to ~0 while
preserving genuine genetic prediction — encoded in `tests/test_covariates.py`.
The per-draw spurious R² is noisy (a 2-population axis is low-rank), so the
result is averaged over replicates.

## `pumas_agreement.py` — PUMAS (single-GWAS subsampling) agrees with PPB

PUMAS (Zhao et al. 2021) and PPB compute the same summary-statistic prediction
R². PPB uses a genuinely held-out target cohort; PUMAS manufactures a
pseudo-validation set by subsampling one GWAS's summary statistics
(`ppb.subsample_sumstats` / `ppb.pumas_r2`), needing no separate cohort. This
experiment shows both, and the individual-level truth, agree across PGS methods.

Run:

```bash
python experiments/pumas_agreement.py
```

Observed:

| arch      | method   | individual | PPB (exact) | PUMAS  |
|-----------|----------|-----------:|------------:|-------:|
| sparse    | causal   |   0.5002   |   0.5002    | 0.5027 |
| sparse    | marginal |   0.3509   |   0.3509    | 0.3508 |
| sparse    | pT       |   0.3758   |   0.3758    | 0.3752 |
| polygenic | marginal |   0.3505   |   0.3505    | 0.3509 |

PPB with exact target cross-products equals the individual-level R² by
construction; PUMAS recovers it from a single GWAS. So PUMAS is the
internal-cross-validation cousin of PPB's external benchmark — same estimator,
different source of the validation data. Encoded in `tests/test_pumas.py`.

## `per_variant_n.py` — per-variant sample sizes; uniform N biases R²

In a meta-analysis each variant has its own sample size `n_j`. The standardized
marginal correlation is recovered per variant via
`r_j = t_j / √(t_j² + n_j − 2)` (`ppb.standardized_marginal`). Using a single
uniform `N` instead shrinks every low-`n` variant and biases R² **downward**.

Run:

```bash
python experiments/per_variant_n.py
```

Observed (individual-level R² = 0.332):

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

---

Method ranking from `benchmark_methods.py` (mean R², individual-level vs
PPB-exact) — correctly preserved:

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
