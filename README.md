# PPB — cross-ancestry polygenic-score portability benchmark

[![CI](https://github.com/bvilhjal/ppb/actions/workflows/ci.yml/badge.svg)](https://github.com/bvilhjal/ppb/actions/workflows/ci.yml)

PPB **measures the cross-ancestry portability of a polygenic score** — its
predictive accuracy (R²) in a *target ancestry* — from **summary-level data only**
(target-ancestry GWAS summary statistics + a matched LD reference), without
individual-level test records. The estimator is

**(X1) Target-ancestry summary-statistic accuracy**

```
R²_B = (wᵀ z_B)² / (wᵀ D_B w)
```

where `w` are the PGS weights (trained in any ancestry) and **both** `z_B`
(marginal summary statistics of the trait) and `D_B` (the LD matrix) come from the
**target ancestry B**. The within-ancestry case is the special case `A = B` and
the validation anchor.

- **Measures, does not predict.** Given target-ancestry data it measures realized
  portability; it cannot forecast portability from discovery-ancestry data alone.
  Substituting discovery-ancestry data — `z_A` and `D_A` — estimates R²_A, so it
  overstates R²_B by `1/portability − 1` (+58% at the simulation's r_g=0.8).
  See [`docs/CROSS_ANCESTRY.md`](docs/CROSS_ANCESTRY.md).
- **Prior art.** The identity is **not new**: it is the quasi-correlation of
  Pattee & Pan (2020) and the square of lassosum's pseudovalidation criterion
  (Mak et al. 2017); PUMAS reaches the same estimand by subsampling.
  See [`docs/METHOD.md`](docs/METHOD.md) §1.
- **Foundation.** Built on the within-ancestry summary-statistic *benchmark* of
  Witteveen et al., *Publicly Available Privacy-preserving Benchmarks for Polygenic
  Prediction* (bioRxiv 2022,
  [doi:10.1101/2022.10.10.510645](https://doi.org/10.1101/2022.10.10.510645)).
  The cross-ancestry direction is this project's — that paper is European-only —
  and its novelty against the cross-population tuning literature is still to be
  checked. This repository attempts to finish Witteveen's unfinished project
  after he left science; preserving that provenance does not imply his
  endorsement or involvement.

> **Status:** early development. The estimator, LD backends, harmonization, and
> the cross-ancestry method are validated **in simulation** against
> individual-level truth; the within-ancestry anchor is demonstrated **on real
> data** ([`docs/REAL_DATA.md`](docs/REAL_DATA.md)). It has **not** been run on
> real cross-ancestry data. Roadmap: [`FINISHING_PLAN.md`](FINISHING_PLAN.md).
> Scope and failure modes: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Documentation

[`docs/README.md`](docs/README.md) is the roadmap: what each document is for,
which are binding, and where to start. [`docs/NOTATION.md`](docs/NOTATION.md)
defines every symbol once and indexes every numbered result — (M2), (X1), (O4),
(G2) and the rest — to the module that implements it and the test that pins it.

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
same population. It needs only `wᵀz` and `wᵀDw`, so `D` is never materialised
densely: dense, block-diagonal, low-rank, and int8 D8/LR8 backends are provided,
and the loader validates block tiling, offsets, dtypes, packed diagonals, and
low-rank definiteness ([`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) covers what
int8 quantization does and does not guarantee).

Block-level uncertainty and negative controls (`ppb.diagnostics`) come free from
the per-block products a genome-wide sweep already computes: `r2_block_jackknife`
gives a delete-one-block SE, and `sign_flip_null` an exact null — negating a
block's weights flips `wᵀ_b z_b` and leaves `wᵀ_b D_b w_b` alone, so the
sign-flipped scores share the denominator and carry no coherent signal.
`scripts/negative_controls.py` adds the trait-swap matrix.
See [`docs/REAL_DATA.md`](docs/REAL_DATA.md).

Supporting pieces for real summary statistics: allele harmonization
(`ppb.harmonize`), PC/covariate adjustment (`ppb.covariates`), per-variant sample
sizes (`ppb.standardized_marginal`), PUMAS-style single-GWAS subsampling
that refits each pseudo-training split (`ppb.subsample_sumstats` / `ppb.pumas_r2`),
and basis-aware detection/correction of training-target sample overlap
(`ppb.overlap`, see [`docs/OVERLAP.md`](docs/OVERLAP.md)). That correction works
in simulation — it brings a thirtyfold-inflated statistic back to its
independent anchor — but only given an independent reference GWAS, a
reconstructible trainer-sensitivity basis, and a trait sparse enough for the fit
to be identified. Final weights alone are not such a basis, so every published
score fails closed; a *rerunnable* trainer can supply one stochastically
(`ppb.estimate_overlap_basis`).

## LD reference (real data)

`scripts/bigsnpr_ldref_to_ppb.py` converts bigsnpr's precomputed **HapMap3+
European LD reference** (Privé,
[doi:10.6084/m9.figshare.21305061](https://doi.org/10.6084/m9.figshare.21305061),
CC BY 4.0 — 1,444,196 variants, UK Biobank Europeans, 431 independent LD
blocks, GRCh37 with `pos_hg38` alongside) into ppb's LD-reference `.npz`: one
file per chromosome holding the variant table (`chrom, pos, a1, a2`, `rsid`,
`af_UKBB`) and the LD as dense int8 blocks (`DenseLDInt8` / D8 convention) keyed
by the reference's own block structure — exactly block-diagonal (0 off-block
entries). Needs `pip install rdata` (reads the R `dsCMatrix` serializations
directly; no R required).

```bash
python scripts/bigsnpr_ldref_to_ppb.py <data_dir> <out_dir>   # all 22 chromosomes
```

```python
from ppb import read_ldref

ref = read_ldref("ldref_chr22.npz")   # variants, BlockDiagonalLD, rsid, af, pos_hg38
den = ref["ld"].quad(w)               # w^T D w for weights w on ref["variants"]
```

End-to-end evaluations of public PGS Catalog scores against real GWAS with this
reference are in [`docs/REAL_DATA.md`](docs/REAL_DATA.md); the pipelines are
`scripts/panukb_download.sh` + `scripts/panukb_filter_hm3plus.py` (Pan-UKB
targets), `scripts/consortium_prep.py` (consortium targets), and
`scripts/regenerate_results.py` (the ppb evaluation that produces the results
registry — `scripts/eval_consortium.py` / `scripts/eval_panukb.py` remain as
human-readable single-target tables).

## Command line

```bash
ppb evaluate --weights weights.tsv --bundle benchmark.npz \
  --weight-scale dosage [--out result.json]
```

- **weights**: a TSV/CSV with chromosome, position, effect allele, other allele,
  and weight (PGS Catalog column names recognised; `#` comment lines skipped).
- **weight scale**: this must be explicit. Use `dosage` for ordinary PGS Catalog
  per-allele weights; the bundle must then carry target-cohort empirical
  `genotype_sd`, and PPB converts `w_j` to `w_j * genotype_sd_j`. Use
  `standardized` only when the file already contains weights for the
  standardized genotypes represented by the bundle LD.
- **bundle**: an `.npz` with the target-ancestry variant table (`chrom, pos, a1,
  a2`), summary statistics `z`, and an LD reference (dense `D` or low-rank `U`).
  Version-2 bundles may also carry `genotype_sd`; build one with
  `ppb.write_bundle(..., genotype_sd=target_sd)`.

The command harmonizes the weights to the bundle's variants and prints a JSON
`EvaluationResult` with `R²`, `MSE`, and harmonization counts. For case/control
GWAS, this summary-statistic `R²` is an approximation on the chosen standardized
scale; it is **not** liability-scale `R²`.

## Experiments

`experiments/` holds validated demonstrations, each encoded as a test: the
cross-ancestry portability measurement (`cross_ancestry.py`), the within-ancestry
LD-reference behaviour (`figure_s1.py`), cross-method concordance
(`benchmark_methods.py`), PC adjustment (`pc_adjustment.py`), per-variant N
(`per_variant_n.py`), PUMAS-style repeated-learning agreement
(`pumas_agreement.py`), and basis-aware training-target sample-overlap analysis
(`overlap_detection.py`, see
[`docs/OVERLAP.md`](docs/OVERLAP.md)), and the negative
result on LD-based transferability reweighting (`transferability.py`, see
[`docs/TRANSFERABILITY.md`](docs/TRANSFERABILITY.md)). See
[`experiments/README.md`](experiments/README.md).

## Test

```bash
pytest -q
```

## Leaderboard (stage 1)

A static leaderboard is generated from the versioned results registry in
[`results/`](results/) (schema: [`results/schema.md`](results/schema.md)) by
`scripts/build_leaderboard.py` and published to GitHub Pages by the
`leaderboard` workflow (enable Pages with source "GitHub Actions" in repo
settings). It shows maintainer-run baselines only — every evaluation declares
its training/target overlap, and in-sample rows are marked as **upper bounds**.
It is **not a competition**: `(wᵀz)²/(wᵀDw)` is maximized at `w ∝ D⁻¹z`, so a
public bundle makes the top-scoring submission a linear solve rather than a
model (quantified in `FINISHING_PLAN.md`, Gate D). Ranking is meaningful here
only because every entry is a maintainer-run baseline.
The current final LDpred2 weights do not preserve a reconstructible training
operator, so their overlap basis is `basis_unavailable` and they do not receive a
headline corrected `R²`. Correction remains available for future scores whose
trainer basis passes the identification and stability gates in
[`docs/OVERLAP.md`](docs/OVERLAP.md). The registry rejects malformed,
non-finite, or provenance-incomplete packs before rendering. External submissions
enter as reviewed result packs via pull request; there is deliberately no
submission service yet (`FINISHING_PLAN.md`, Gate D).

## License

Code: MIT (see [`LICENSE`](LICENSE)). The source preprint manuscript is CC-BY. The
legacy benchmark datasets are UK Biobank-derived; their redistribution is governed
by UK Biobank data-access terms, **not** the preprint's CC-BY — verify before
re-hosting.
