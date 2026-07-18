# Cross-ancestry R² estimation — PPB's flagship method

This is the project's central method (see `FINISHING_PLAN.md`, "Project focus"):
**measure the predictive R² of a polygenic score in a target ancestry** from
summary-level data. It *measures* portability given target-ancestry data; it does
not *predict* portability from discovery-ancestry data alone.

Derived and adversarially verified (18 claim-lens checks, 0 refuted, 5
qualifications folded in) and demonstrated against individual-level truth in
`experiments/cross_ancestry.py` / `tests/test_cross_ancestry.py`. **Status:
validated in simulation; not yet run on real cross-ancestry data.**

## Background: from Witteveen et al. to cross-ancestry transferability

The idea originates with **Witteveen et al. (2022)** (bioRxiv, doi:10.1101/
2022.10.10.510645; senior author Vilhjalmsson). Their contribution: the
out-of-sample predictive R² of a linear polygenic score can be computed from
**summary-level data alone** — an LD matrix `D` and marginal GWAS summary
statistics `z` — via `R² = (wᵀz)²/(wᵀDw)`, with no individual-level test
genotypes or phenotypes. They released this as a **within-ancestry (European)**
benchmark so that competing PGS methods could be compared on a shared target
without anyone sharing individual-level data. That work must be cited as the
foundation of everything below.

This project starts from that idea and makes one observation: the estimator is
**ancestry-agnostic in form** — the identity holds in *any* population whose
moments `(z, D)` are supplied. Supplying the **target ancestry's** summary
statistics `z_B` and LD `D_B` therefore turns the same estimator into a
measurement of **cross-ancestry transferability**: the realized predictive
accuracy in a target ancestry B of a PGS trained in any ancestry. That reframing
— from a within-ancestry benchmark into a summary-statistics-based probe of PRS
portability across ancestries — is this project's new contribution, and is *not*
part of the European-only original.

## The estimator

The PPB identity is ancestry-agnostic in *form*. For a fixed PGS `w` (trained in
any ancestry A) evaluated in a **target ancestry B**:

$$R^2_B = \frac{(w_B^\top z_B)^2}{w_B^\top D_B\, w_B}$$

with **both** `z_B` and `D_B` from ancestry B:

| input | definition | source |
|---|---|---|
| `z_B` | target-trait marginal correlations in B, `r_{B,j} = t_{B,j}/√(t_{B,j}²+n_{B,j}−2)` | **B GWAS** of the trait |
| `D_B` | LD (genotype correlation) matrix in B, ideally from a panel independent of the `z_B` sample | **B reference panel** |
| `w_B` | weights on B's standardized scale | per-allele β from A, rescaled to B's empirical genotype SDs (`sd_B/sd_A`) |

**Not required:** cross-population genetic correlation `r_g`, the discovery LD
`D_A`, discovery frequencies `p_A`, or `var(y_B)` (it cancels on the correlation
scale — it only enters the scale-dependent MSE). Because `z_B` is *measured* in B,
it already encodes B's true effects tagged through B LD; the estimator never
decomposes it into causal effects, so `r_g` has no slot.

## The one subtlety: gauge self-consistency

The ratio is invariant only to a **global** rescale of `w`, not a per-variant
one. So `w`, `z_B`, `D_B` must be on **one self-consistent B standardization**.
The HWE scale `√(2 p_B(1−p_B))` is correct only under Hardy-Weinberg in B; in a
**structured or admixed** target (F ≠ 0) it differs per variant from the true
genotype SD and the residual does **not** cancel. Operational rule: standardize
all three with the same **empirical B genotype SDs** (in-sample B moments), not
the `2p(1−p)` formula. (In the simulation harness everything is within-cohort
standardized, so this is automatic and the identity is exact to machine precision.)

## Hard requirements

1. **`z_B` from a B GWAS of the same trait** (with per-variant `n_B`). Irreplaceable.
2. **`D_B` from a B panel** covering `w`'s variants; in-sample B LD is mandatory
   for admixed targets; regularized/PSD.
3. **One self-consistent B gauge** for `w`, `z_B`, `D_B`; drop variants monomorphic in B.
4. **Tri-panel allele harmonization** (weights, `z_B`, `D_B`); drop strand-ambiguous
   SNPs — cross-ancestry MAF differences make frequency-based strand tie-breaks unreliable.
5. **Sample independence** between the A training GWAS and the `z_B` GWAS; PC-adjust within B.
6. **Finite-sample numerator correction** *(not yet implemented)*: the plug-in
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

## Positioning

This sits in the **measuring-portability** camp (summary-statistics option) — the
trans-ancestry analogue of PUMAS (real B `z`/`D` instead of subsampled
pseudo-validation). It is complementary to: **PRS-CSx / XPASS** (which *build*
cross-population weights — PPB scores whatever they emit), **Popcorn / S-LDXR**
(which estimate `r_g` — an input to the *predict* branch, not needed here), and
**Wang et al. 2020** (the deterministic *predict* branch that forecasts what PPB
measures). Individual-level portability benchmarks (Martin 2019, Ding 2023) are
the ground truth PPB substitutes for when only a B GWAS + LD panel exist.
