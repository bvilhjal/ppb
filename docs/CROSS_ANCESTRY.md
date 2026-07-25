# Cross-ancestry R² estimation — PPB's flagship method

Status: specification (binding). Symbols and labels:
[`NOTATION.md`](NOTATION.md). Results are labelled (X1)–(X3).

This is the project's central method (see `FINISHING_PLAN.md`, "Project focus"):
**measure the predictive R² of a polygenic score in a target ancestry** from
summary-level data. It *measures* portability given target-ancestry data; it does
not *predict* portability from discovery-ancestry data alone.

Derived and adversarially verified (18 claim-lens checks, 0 refuted, 5
qualifications folded in) and demonstrated against individual-level truth in
`experiments/cross_ancestry.py` / `tests/test_cross_ancestry.py`. **Status:
validated in simulation; not yet run on real cross-ancestry data.**

## Background: prior art, then cross-ancestry transferability

**The identity is not new, and this project does not claim it.** Evaluating a
linear score against summary-level moments via `(wᵀz)²/(wᵀDw)` predates the
work below:

- **lassosum pseudovalidation** (Mak, Porsch, Choi, Zhou & Sham 2017, *Genet
  Epidemiol* 41:469, [doi:10.1002/gepi.22050](https://doi.org/10.1002/gepi.22050))
  selects its penalty by maximizing `βᵀr/√(βᵀRβ)` — the square root of this
  estimator — with `r` from external summary statistics and `R` from a reference
  panel. It is used there as a *tuning criterion* rather than a reported accuracy.
- **Quasi-correlation** (Pattee & Pan 2020, *PLOS Comput Biol* 16:e1008271,
  [doi:10.1371/journal.pcbi.1008271](https://doi.org/10.1371/journal.pcbi.1008271))
  is the same quantity presented explicitly as a metric that "can be used to
  evaluate the performance of a polygenic risk score on out-of-sample data" and
  that "estimates the true correlation" — i.e. this estimand, named and
  published.
- **PUMAS** (Zhao et al. 2021) reaches the same estimand by subsampling a single
  GWAS into pseudo-training/validation splits instead of using an external target.

**Witteveen et al. (2022)** (bioRxiv, doi:10.1101/2022.10.10.510645; senior
author Vilhjalmsson) is the foundation this repository continues, and its
contribution is the **shared public artifact**: a released LD reference plus
target summary statistics for a fixed **within-ancestry (European)** cohort, so
that competing PGS methods could be compared on a common target without anyone
sharing individual-level data. A benchmark is a different thing from an
estimator, and that is what is being stewarded here.

**What this project adds** is therefore not the identity but: the cross-ancestry
framing and the failure-mode map below; the infrastructure (int8 block LD store,
tri-panel harmonization, a provenance-checked results registry); and the
evaluation discipline (exact joint support, fail-closed overlap correction,
declared gauge). Prior art for the cross-ancestry *application* specifically has
not been exhaustively surveyed — cross-population construction methods tune on
target-ancestry data, sometimes with criteria of this family — so treat
"new to this project" as a claim still to be checked against the literature
before publication.

## The estimator

The reframing rests on one observation: (M1) is **ancestry-agnostic in form**.
The identity holds in *any* population whose moments `(z, D)` are supplied, so
supplying the target ancestry's turns an evaluator into a measurement of
portability. For a fixed PGS `w` (trained in any ancestry A) evaluated in a
**target ancestry B**:

**(X1) Target-ancestry predictive accuracy.** This is (M1) with ancestry-B
moments substituted — the same code, the same algebra, different inputs.

$$R^2_B = \frac{(w_B^\top z_B)^2}{w_B^\top D_B\, w_B}$$

with **both** `z_B` and `D_B` from ancestry B:

**Table 1. Inputs and their required scale.**

| input | definition | source |
|---|---|---|
| `z_B` | target-trait marginal correlations in B, by (M3) | **B GWAS** of the trait |
| `D_B` | LD (genotype correlation) matrix in B, ideally from a panel independent of the `z_B` sample | **B reference panel** |
| `w_B` | weights on B's standardized-genotype scale | dosage weight `b_j` times empirical `sd_B,j`; only an already A-standardized weight uses `w_A,j × sd_B,j / sd_A,j` |

**Not required:** cross-population genetic correlation `r_g`, the discovery LD
`D_A`, discovery frequencies `p_A`, or `var(y_B)` (it cancels on the correlation
scale — it only enters the scale-dependent MSE). Because `z_B` is *measured* in B,
it already encodes B's true effects tagged through B LD; the estimator never
decomposes it into causal effects, so `r_g` has no slot.

## The one subtlety: gauge self-consistency

**(X2) Gauge condition.** `w`, `z_B` and `D_B` must be expressed on one and the
same ancestry-B standardization; put the weights there with the empirical
genotype SDs, `w_B,j = w_A,j · sd_B,j / sd_A,j`.

The ratio is invariant only to a **global** rescale of `w`, not a per-variant
one. So `w`, `z_B`, `D_B` must be on **one self-consistent B standardization**.
The HWE scale `√(2 p_B(1−p_B))` is correct only under Hardy-Weinberg in B; in a
**structured or admixed** target (F ≠ 0) it differs per variant from the true
genotype SD and the residual does **not** cancel. Operational rule: standardize
all three with the same **empirical B genotype SDs** (in-sample B moments), not
the `2p(1−p)` formula. (In the simulation harness everything is within-cohort
standardized, so this is automatic and the identity is exact to machine precision.)

The public evaluator makes the convention explicit: `weight_scale="dosage"`
requires a bundle carrying empirical `genotype_sd` and applies `b_j × sd_B,j`;
`weight_scale="standardized"` promises that the submitted weights already match
the standardized genotypes represented by `D_B`.

> **Known deviation in the current real-data path.** `scripts/regenerate_results.py`
> (and the older `scripts/eval_*.py`) put `w` on the standardized scale with
> `sd_j = √(2 f_j (1 − f_j))` using `af_UKBB` from the LD reference — i.e. the HWE
> formula this rule warns against, because the shipped reference carries allele
> frequencies but not empirical genotype SDs. For the current **within-ancestry
> EUR anchor** the two are close (a homogeneous, PC-adjusted UKBB sample is near
> HWE), so the demonstrated numbers are not materially affected. It is **not**
> acceptable for the flagship cross-ancestry application on a structured or
> admixed target, where the residual does not cancel. Carrying empirical per-variant
> SDs in the LD-reference schema is a prerequisite for the first real
> cross-ancestry run.

## Hard requirements

1. **`z_B` from a B GWAS of the same trait** (with per-variant `n_B`). Irreplaceable.
2. **`D_B` from a B panel** covering `w`'s variants; in-sample B LD is mandatory
   for admixed targets; regularized/PSD.
3. **One self-consistent B gauge** for `w`, `z_B`, `D_B`; drop variants monomorphic in B.
4. **Tri-panel allele harmonization** (weights, `z_B`, `D_B`); drop strand-ambiguous
   SNPs — cross-ancestry MAF differences make frequency-based strand tie-breaks unreliable.
5. **Sample independence** between the A training GWAS and the `z_B` GWAS; PC-adjust within B.
6. **(X3) Finite-sample numerator correction** *(not yet implemented)*: the plug-in
   `(wᵀẑ_B)²` is biased upward by `≈ wᵀD_B w / N_B` (an absolute R² bias of
   `≈ 1/N_B` — small even at modest GWAS N, but cheap to remove); subtract it
   (or use within-B PUMAS subsampling) and report an SE. Tracked as a v0.1
   completion item in `FINISHING_PLAN.md`.
7. **Matched coverage**: estimate on the intersection of `w`, `z_B`, `D_B`; report retained fraction.

## What is impossible without target-ancestry data

You **cannot** estimate `R²_B` from ancestry-A summaries (`z_A`, `D_A`, `p_A`)
plus a B LD panel. The denominator `wᵀD_B w` (PGS variance in B) *is* computable
from a B genotype reference with no trait data — the irreducible gap is the
numerator `wᵀz_B` (the PGS–phenotype covariance in B). Substituting `z_A`
estimates `R²_A`, not `R²_B`, and overstates cross-ancestry accuracy whenever
transfer is imperfect (**+58% at r_g = 0.8** in the demo).

Predicting `R²_B` without any B phenotype is a **portability-modeling** problem
(Wang et al. 2020), needing external parameters PPB does not supply: `h²_B` (sets
the numerator scale, cannot cancel), a cross-population effect-coupling model
(`r_g` is necessary but not sufficient), and both ancestries' LD/MAF.

## Demonstration (`experiments/cross_ancestry.py`)

Two Balding-Nichols ancestries, shared causal architecture correlated at `r_g`, a
p+T PGS trained in A and evaluated in B (m=500, n=20000, F_ST=0.25, h²=0.5):

| | r_g = 1.0 (control) | r_g = 0.8 (main) |
|---|---|---|
| portability R²_B/R²_A | 0.998 | **0.648** |
| exact (z_B, test-B LD) | −0.000% | +0.000% |
| independent (z_B, indep-B LD) | −0.095% | −0.095% |
| **mismatched (z_B, ancestry-A LD)** | −3.02% | −3.02% |
| **A-only (z_A, ancestry-A LD)** | +0.06% | **+57.7%** |
| LD-form ratio wᵀD_A w / wᵀD_B w | 1.031 | 1.031 |

So with B target stats + B LD the estimate is exact/unbiased and recovers the
portability loss; using ancestry-A LD biases it by the LD-form ratio; and
substituting ancestry-A sumstats does not estimate `R²_B` at all.

The **+57.7%** figure is arithmetic, not a measured effect size: substituting
`z_A` estimates `R²_A`, so the overstatement is `R²_A/R²_B − 1 = 1/0.648 − 1 =
+54.3%` plus the LD term. It is fully determined by the `r_g` and `F_ST` chosen
here, and scales as `1/portability − 1` — quote it that way rather than as a
number the method discovered.

### What this table does *not* show: LD-driven portability loss

**Both ancestries have the same LD architecture above.** `ppb.simulate` draws
haplotypes with latent correlation `rho**|i-j|` — the *same* matrix for both
populations — and Balding-Nichols differs them only in allele frequency. So
`F_ST` has no LD channel at all, which is why **portability is 0.998 at
r_g = 1.0**: ancestry divergence by itself costs nothing here, and every bit of
loss in the table comes from the `r_g` knob. Real portability loss is driven
substantially by LD and MAF divergence at `r_g ≈ 1` (Wang et al. 2020; Ding et
al. 2023), so this demonstration validates the estimator's *algebra* — never in
doubt, it is an identity — without exercising the mechanism the project exists
to measure.

`run()` takes `rho_b` and `block_size_b` to give ancestry B its own LD decay and
block boundaries. Sweeping them (`python experiments/cross_ancestry.py
--ld-divergence`, r_g = 1.0 throughout, m = 1000, dense weights, 25 draws):

**Table 2. Wrong-ancestry-LD bias against the simulated LD architecture.**

| A LD | B LD | portability at r_g = 1 | mismatch % bias | LD-form ratio |
|---|---|---:|---:|---:|
| ρ 0.5, blocks 100 *(shipped)* | ρ 0.5, blocks 100 | 1.001 | −3.17 | 1.034 |
| ρ 0.8, blocks 100 | ρ 0.8, blocks 100 | 0.963 | −10.66 | 1.113 |
| ρ 0.9, blocks 100 | ρ 0.9, blocks 100 | 0.920 | −13.70 | 1.154 |
| ρ 0.9, blocks 100 | **ρ 0.6, blocks 100** | **0.780** | **−64.74** | 2.844 |
| ρ 0.9, blocks 100 | **ρ 0.6, blocks 50** | **0.771** | **−65.30** | 2.883 |

Two conclusions. **The −3.0% headline is a property of the generator, not a
transferable magnitude** — it is measured at AR(1) ρ = 0.5, which decays to
r = 0.06 within four variants, against a real HM3+ reference whose blocks have a
median of 1,901 variants. Stronger LD alone takes it to −13.7%; genuinely
divergent LD takes it past −60%. And **LD divergence produces real portability
loss with `r_g` held at 1** (1.001 → 0.780), which is the mechanism the
shared-LD configuration cannot produce.

Pinned by `tests/test_cross_ancestry.py::test_ld_divergence_dominates_the_mismatch_bias`.
A defensible number for the mismatch failure mode needs either a coalescent
simulation with a real population split, or two real ancestry LD panels — the
latter is Phase-4 work and is the honest place to settle it.

## Positioning

This sits in the **measuring-portability** camp (summary-statistics option) — the
trans-ancestry analogue of PUMAS (real B `z`/`D` instead of subsampled
pseudo-validation). It is complementary to: **PRS-CSx / XPASS** (which *build*
cross-population weights — PPB scores whatever they emit), **Popcorn / S-LDXR**
(which estimate `r_g` — an input to the *predict* branch, not needed here), and
**Wang et al. 2020** (the deterministic *predict* branch that forecasts what PPB
measures). Individual-level portability benchmarks (Martin 2019, Ding 2023) are
the ground truth PPB substitutes for when only a B GWAS + LD panel exist.
