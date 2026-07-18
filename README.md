# PPB — Privacy-preserving Benchmark for Polygenic Prediction

PPB evaluates the predictive accuracy of linear polygenic scores (PGS) from
**summary-level information only** — LD and GWAS summary statistics — without
individual-level test records. Given PGS weights `w`, target marginal summary
statistics `z = (1/N) Xᵀy`, and an LD matrix `D = (1/N) XᵀX`, the prediction
accuracy is

```
R² = (wᵀz)² / (wᵀ D w)
```

This is a maintained reimplementation of the method in Witteveen et al.,
*Publicly Available Privacy-preserving Benchmarks for Polygenic Prediction*
(bioRxiv 2022, [doi:10.1101/2022.10.10.510645](https://doi.org/10.1101/2022.10.10.510645)).
See [`docs/METHOD.md`](docs/METHOD.md) for the estimator specification and
[`FINISHING_PLAN.md`](FINISHING_PLAN.md) for the project plan and status.

> Status: early v0.1 development. The core estimator and LD backends exist and are
> covered by equation-level tests; datasets, CLI, and schemas are in progress.

## Install

```bash
pip install -e .            # runtime: numpy, numba
pip install -e ".[test]"    # + pytest
```

Requires Python ≥ 3.11 (developed on 3.14). The LD quadratic-form kernels use
numba; no scipy is required.

## Usage

```python
import numpy as np
from ppb import r2, DenseLD, LowRankLD

# w: PGS weights, z: target marginal summary stats, D: LD matrix (all aligned)
acc = r2(w, z, DenseLD(D))                 # exact, dense D
acc = r2(w, z, LowRankLD(U))               # low-rank D ≈ U Uᵀ (LR8-style)
```

The estimator only ever needs `wᵀz` and `wᵀDw`, so `D` is never materialised
densely in the scalable path. The low-rank factor `R ≈ U Uᵀ` is positive
semi-definite by construction, so `wᵀDw ≥ 0` always holds.

## Command line

```bash
ppb evaluate --weights weights.tsv --bundle benchmark.npz [--out result.json]
```

- **weights**: a TSV/CSV with columns for chromosome, position, effect allele,
  other allele, and weight (PGS Catalog names like `chr_name`, `chr_position`,
  `effect_allele`, `other_allele`, `effect_weight` are recognised; `#` comment
  lines are skipped).
- **bundle**: an `.npz` with the benchmark's variant table (`chrom, pos, a1, a2`),
  target summary statistics `z`, and an LD reference (dense `D` or low-rank `U`).
  Build one with `ppb.write_bundle(...)`.

The command harmonizes the weights to the bundle's variants (flipping signs on
allele swaps/strand, dropping palindromes) and prints a JSON `EvaluationResult`
with the `R²`, `MSE`, and per-input harmonization counts.

## Test

```bash
pytest -q
```

## License

Code: MIT (see [`LICENSE`](LICENSE)). The method and benchmark datasets are
CC-BY per the source preprint; data terms are tracked separately from the code
license.
