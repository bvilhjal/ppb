# PPB — cross-ancestry polygenic-score portability benchmark

[![CI](https://github.com/bvilhjal/ppb/actions/workflows/ci.yml/badge.svg)](https://github.com/bvilhjal/ppb/actions/workflows/ci.yml)

PPB **measures the cross-ancestry portability of a polygenic score** — its
predictive accuracy (R²) in a *target ancestry* — from **summary-level data only**
(target-ancestry GWAS summary statistics + a matched LD reference), without
individual-level test records. The estimator is

```
R²_B = (wᵀ z_B)² / (wᵀ D_B w)
```

where `w` are the PGS weights (trained in any ancestry) and **both** `z_B`
(marginal summary statistics of the trait) and `D_B` (the LD matrix) come from the
**target ancestry B**. The within-ancestry case is the special case `A = B` and
the validation anchor.

- **Measures, does not predict.** Given target-ancestry data it measures realized
  portability; it cannot forecast portability from discovery-ancestry data alone
  (substituting discovery-ancestry data — `z_A` and `D_A` — estimates R²_A and
  overstates R²_B by +58% at r_g=0.8 in simulation).
  See [`docs/CROSS_ANCESTRY.md`](docs/CROSS_ANCESTRY.md).
- **Foundation.** Built on the within-ancestry summary-statistic evaluator of
  Witteveen et al., *Publicly Available Privacy-preserving Benchmarks for Polygenic
  Prediction* (bioRxiv 2022,
  [doi:10.1101/2022.10.10.510645](https://doi.org/10.1101/2022.10.10.510645)). The
  cross-ancestry direction is **new to this project** and is not attributed to that
  paper (which is European-only).

> **Status:** early development. The estimator, LD backends, harmonization, and the
> cross-ancestry method are **validated in simulation against individual-level
> truth** (67 tests, CI green). It has **not** been run on real cross-ancestry
> data. See [`FINISHING_PLAN.md`](FINISHING_PLAN.md) for the roadmap and
> [`docs/METHOD.md`](docs/METHOD.md) / [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)
> for the specification and scope.

## Install

```bash
pip install -e .            # runtime: numpy, numba
pip install -e ".[test]"    # + pytest
```

Requires Python ≥ 3.11 (developed on 3.14). The LD quadratic-form kernels use
numba; no scipy is required.

## Usage

```python
from ppb import r2, DenseLD

# Cross-ancestry: z_B, D_B from the TARGET ancestry B; w harmonized to B's variants.
acc_B = r2(w, z_B, DenseLD(D_B))     # predictive R² of w in ancestry B
```

The estimator is ancestry-agnostic in form — within-ancestry is `z`/`D` from the
same population. It only needs `wᵀz` and `wᵀDw`, so `D` is never materialised
densely (dense, block-diagonal, low-rank, and int8 D8/LR8 backends are provided;
the low-rank factor is PSD, so `wᵀDw ≥ 0`).

Supporting pieces for real summary statistics: allele harmonization
(`ppb.harmonize`), PC/covariate adjustment (`ppb.covariates`), per-variant sample
sizes (`ppb.standardized_marginal`), and PUMAS-style single-GWAS subsampling
(`ppb.subsample_sumstats`).

## Command line

```bash
ppb evaluate --weights weights.tsv --bundle benchmark.npz [--out result.json]
```

- **weights**: a TSV/CSV with chromosome, position, effect allele, other allele,
  and weight (PGS Catalog column names recognised; `#` comment lines skipped).
- **bundle**: an `.npz` with the target-ancestry variant table (`chrom, pos, a1,
  a2`), summary statistics `z`, and an LD reference (dense `D` or low-rank `U`).
  Build one with `ppb.write_bundle(...)`. (A per-ancestry bundle schema carrying
  allele frequencies and per-variant `n` is on the v0.1 roadmap.)

The command harmonizes the weights to the bundle's variants and prints a JSON
`EvaluationResult` with `R²`, `MSE`, and harmonization counts.

## Experiments

`experiments/` holds validated demonstrations, each encoded as a test: the
cross-ancestry portability measurement (`cross_ancestry.py`), the within-ancestry
LD-reference behaviour (`figure_s1.py`), cross-method concordance
(`benchmark_methods.py`), PC adjustment (`pc_adjustment.py`), per-variant N
(`per_variant_n.py`), PUMAS agreement (`pumas_agreement.py`), and the negative
result on LD-based transferability reweighting (`transferability.py`, see
[`docs/TRANSFERABILITY.md`](docs/TRANSFERABILITY.md)). See
[`experiments/README.md`](experiments/README.md).

## Test

```bash
pytest -q
```

## License

Code: MIT (see [`LICENSE`](LICENSE)). The source preprint manuscript is CC-BY. The
legacy benchmark datasets are UK Biobank-derived; their redistribution is governed
by UK Biobank data-access terms, **not** the preprint's CC-BY — verify before
re-hosting.
