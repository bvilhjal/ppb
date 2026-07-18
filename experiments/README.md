# Experiments

Reproduction drivers for results in Witteveen et al. (2022). These are not part
of the installable `ppb` package; run them from the repo root with the `ppb`
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

Method ranking (mean R², individual-level vs PPB-exact) — correctly preserved:

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
