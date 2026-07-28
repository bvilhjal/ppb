# Is the target `z` on the scale the estimator assumes?

Status: specification (binding). Symbols and labels:
[`NOTATION.md`](NOTATION.md). Results are labelled (C1)–(C3).

`R² = (wᵀz)²/(wᵀDw)` puts `z` in the numerator alone. A target whose summary
statistics carry a factor `c` reports `c²R²`, and nothing in the estimator
cancels it or notices. This is PPB's **largest measured error**: against
published individual-level accuracy, both GIANT targets come in 1.5–2× low while
a GLGC target agrees to 2% ([`REAL_DATA.md`](REAL_DATA.md), Table 3) — the size
and pattern of genomic control, which GIANT applies at study *and* meta level.

The diagnostic needs nothing PPB does not already have.

**(C1) LD score.**

    l_j = Σ_k D_jk²

A property of the reference alone; no summary statistics enter. Implemented as
`LDBackend.ld_scores()` on every backend.

**(C2) LD-score regression** ([Bulik-Sullivan et al.
2015](https://doi.org/10.1038/ng.3211)).

    E[χ²_j] = intercept + (N h² / M) · l_j

The slope carries heritability. The **intercept** is what matters here: it is 1
when nothing is wrong, confounding and cryptic relatedness push it *up*, and
dividing every `χ²` by a genomic-control `λ` pushes it *down* to `1/λ`, taking
the slope with it. Implemented as `ppb.ldscore_regression`.

**(C3) Implied `z` scale.**

    z_true = z_observed / √intercept        R²_true = R²_observed / intercept

## What is identified, and what is not

The intercept is one number standing for two mechanisms, so it cannot separate
them, and (C3) is a *correction* only under the assumption that the departure is
a uniform rescaling. That is exactly true for genomic control and false for
stratification, which adds a constant to `χ²` rather than scaling it.

**One direction survives the ambiguity.** No confounding mechanism pushes the
intercept below 1. And the block-diagonal reference's LD scores omit the
long-range terms it defines away, so they are systematically low — which biases
the intercept *up*. An intercept significantly below 1 is therefore
**conservative** evidence of over-correction, which is the case this exists for.

## Fail-closed conditions

`deflation_detected` is set only when the intercept is more than **two jackknife
standard errors below 1**. Two refusals matter more than the detection:

- **No blocks, no verdict.** Without a delete-one-block jackknife there is no
  standard error, so no claim is made however low the intercept looks. Blocks,
  not variants, are the independent units.
- **A small variant count decides nothing.** The intercept is an extrapolation
  to `l = 0`; at 2,000 variants its standard error exceeds 1, and it can even
  come out negative — nonsense rather than an enormous correction, so `z_scale`
  is withheld entirely. `leverage` reports how far the extrapolation reaches, in
  units of the LD-score spread. Equal-sized, equally-correlated blocks give one
  LD score and no leverage at all.

## Validation

`experiments/z_calibration.py`, pinned by `tests/test_ldscore.py`.

**Table 1. Consistency.** `χ²` drawn from (C2) itself, so any error is the
estimator's. True intercept 1.0, true `h²` 0.30.

| variants | intercept | SE | `h²` |
|---:|---:|---:|---:|
| 2,000 | 1.4260 | 1.2924 | 0.2989 |
| 10,000 | 1.1504 | 0.1588 | 0.2956 |
| 50,000 | 0.9516 | 0.0262 | 0.3104 |
| 200,000 | 1.0188 | 0.0078 | 0.2911 |

`h²` is well determined throughout; the **intercept needs on the order of 10⁵
variants**. The shipped reference carries 1,444,196, so real use is comfortable —
but this is why the gate above exists.

**Table 2. Genotypes and block-diagonal LD scores.** 18,000 simulated variants
in 180 blocks of differing LD tightness, mean `χ²` 1.43. A genomic-control `λ` is
applied and the fit asked to find it.

| `λ_GC` | intercept | `1/λ` | SE | implied `z` scale | detected |
|---:|---:|---:|---:|---:|:---|
| 1.00 | 0.9721 | 1.0000 | 0.0406 | 1.0142 | no |
| 1.30 | 0.7478 | 0.7692 | 0.0312 | 1.1564 | **yes** |
| 1.60 | 0.6076 | 0.6250 | 0.0254 | 1.2829 | **yes** |
| 2.00 | 0.4861 | 0.5000 | 0.0203 | 1.4344 | **yes** |

The null row is the one that matters most: block-diagonal LD scores do **not**
fabricate a deflation. Recovery is within ~3% of `1/λ` at every level, and the
implied `z` scale within 1.5% of `√λ`.

**Table 3. A falsifiable prediction for the real data.** If the GIANT shortfall
is a uniform `z` rescaling, (C3) fixes the intercept exactly.

| R² shortfall | implied intercept | implied `z` scale |
|---:|---:|---:|
| 1.50 | 0.667 | 1.225 |
| 1.75 | 0.571 | 1.323 |
| 2.00 | 0.500 | 1.414 |

A real fit landing in that band confirms genomic control as the cause; an
intercept near 1 refutes it and sends the explanation elsewhere. **This has not
been run** — it needs the target summary statistics, which are not in this
repository.

## Limitations

- A reimplementation of the LDSC estimator, not a port: the two-step weighting is
  standard, but regression-SNP selection, per-chromosome handling, and
  partitioned annotations are not reproduced. Do not expect the reference
  software's numbers to the last digit.
- Block-diagonal LD scores are systematically low. That is conservative for
  detecting deflation and **not** conservative for `h²`, which will come out
  biased; the `h²` here is a by-product, not a competitor to LDSC's.
- (C3) assumes a uniform rescaling. Applying it to an intercept raised by genuine
  stratification would deflate an R² whose inflation is not multiplicative.
- Nothing here has been run on real summary statistics, and no registry record
  carries a calibration.
