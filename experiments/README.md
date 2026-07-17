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
