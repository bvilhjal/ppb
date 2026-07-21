# Detecting and correcting sample overlap from PGS weights alone

**Status:** validated in simulation (`experiments/overlap_detection.py`,
`tests/test_overlap.py`) and demonstrated on the real-data evaluations of
`docs/REAL_DATA.md`. To our knowledge the weight-space overlap detector and
correction below are new; the overlapping-noise insight itself is the
cross-trait LDSC intercept of Bulik-Sullivan et al. 2015 translated to the
PGS-evaluation setting (see "Related work").

## The problem

The accuracy estimator `R² = (wᵀz)² / (wᵀDw)` assumes the PGS weights `w` were
trained independently of the target GWAS's estimation noise. If the score's
training data overlaps the target sample, `w` partially memorizes the shared
noise and the numerator is inflated — in the real-data demonstration, by up to
15× (height: R² 0.80 with the overlapping Pan-UKB target vs 0.21 with the
non-overlapping GIANT target). For a benchmark that scores submitted PGS
models against released target summary statistics, overlap is the dominant
failure mode (see `FINISHING_PLAN.md`, Gate D).

## Moment decomposition

Write the target marginal statistics as `ẑ = ζ + ε` with `Cov(ε) = D/N`.
Then

```
E[wᵀẑ] = wᵀζ (true signal) + E[wᵀε] (overlap term)
```

Under no overlap, `w` is independent of `ε`, the second term vanishes, and only
the second-moment finite-sample bias `wᵀDw/N` remains (handled separately; see
`docs/METHOD.md`). Under overlap, `w = f(ẑ_train)` was fit to noise correlated
with `ε`, and the second term is a **first-moment, additive inflation**.

For a trainer that is linear in the training statistics, `w ≈ A ẑ_train`
(marginal/Z-score weights: `A = I`; LDpred2-inf: `A = (D + λI)⁻¹`; LDpred2 with
a large causal fraction approximately so),

```
E[wᵀε] = (N_ov / √(n₁ n₂)) · ρ_ε · tr(A D)  ≈  γ · m,
```

with `N_ov` the number of shared samples, `ρ_ε` the noise correlation of the
two traits, and `γ` the per-variant overlap coefficient. For `A = I` exactly,
`γ = N_ov / (n₁ n₂)`. The key structural fact: **estimation noise carries no LD
structure, so the overlap term is uniform per variant, while true signal
concentrates in LD.** That asymmetry makes overlap identifiable from
`(w, z, D)` alone — no training summary statistics required.

## Dual-target detector (validated design)

Given the same score evaluated against a *suspect* target `z_t` and a
*reference* target `z_r` of the same trait (genetic correlation ≈ 1, declared
non-overlapping):

1. Per LD block `b`: `u_b(z) = w_bᵀ z_b`, `v_b = w_bᵀ D_b w_b`
   (`ppb.block_products`).
2. Difference `Δu_b = u_b(z_t) − u_b(z_r)` removes the signal.
3. Weighted (weights `1/(v_t + v_r)`) through-origin regression `Δu_b ~ γ m_b`
   with a leave-group-out jackknife (`ppb.overlap_slope`).
4. Correction: `num_corr = num − γ·m` and `R²_corr = num_corr² / (wᵀDw)`
   (`ppb.correct_numerator`).

## Validation

**Simulation** (`experiments/overlap_detection.py`): training/target cohorts
with controlled overlap fraction, independent anchor, matched cohort LD.

| overlap | γ̂ | γ_true | R² naive | R² corrected | R² honest |
|---:|---:|---:|---:|---:|---:|
| 0% | ~0 | 0 | 0.074 | 0.076 | 0.075 |
| 25% | 5.8e-05 | 6.3e-05 | 0.163 | 0.080 | 0.077 |
| 100% | 2.3e-04 | 2.5e-04 | 0.616 | 0.102 | 0.088 |

The detector is calibrated under the null, recovers `γ_true`, and the
correction restores the honest anchor within ~15% even at full overlap
(slightly conservative: it under-corrects, because γ̂ is ~10% low there).

**Real data** (scores from `docs/REAL_DATA.md`, dual-target against the
consortium GWAS): overlap decisively detected for the polygenic scores
(height z = 12.2, BMI z = 12.3, T2D z = 5.0, CAD z = 5.6); corrections move
R² from the in-sample values most of the way toward the honest anchors
(e.g. height 0.80 → 0.45 vs honest 0.21 — the remainder is genuine
UKBB→GIANT transportability loss, not overlap; see Caveats).

## The sparse-score failure mode (important)

With strong nonlinear shrinkage — p-value thresholding, LDpred2-auto with a
small causal fraction — the effective trainer operator `A` projects the noise
fit onto the same variants that carry the signal. The uniform term vanishes
and overlap becomes **unidentifiable by shape alone**: in simulation the
detector reads γ̂ ≈ 0 at every overlap level (naive R² stays inflated 3×); on
real data the sparse LDL/BrCa scores show no detectable uniform term despite
known UKBB overlap. Such evaluations must be **flagged as upper bounds**, not
corrected. A shrinkage-proof alternative based on PUMAS-style subsample
scaling (overlap scales with the recycled-noise fraction while `w` is held
fixed) is the designated complement and is not yet implemented.

## Caveats

- **A reference target is required.** The detector needs a second,
  non-overlapping GWAS of the same trait (r_g ≈ 1) to difference against.
  Single-target versions (uniform-in-variant intercept, LD-slope amplification)
  are confounded by trainer shrinkage and not yet reliable.
- **Correction ≠ recovery.** The corrected R² removes the uniform noise-fit
  term; what remains can still differ from the honest anchor through cohort
  differences (r_g < 1, transportability loss) — those are signal, not bias,
  and are what the benchmark measures.
- **Overlap vs stratification.** Shared population stratification also
  produces genome-wide-uniform covariance between studies (the classic LDSC
  intercept caveat): the detector sees "shared noise structure," of which
  sample overlap is the dominant expected cause, not proof of overlap per se.
- **Power.** Like LDSC intercepts, γ̂ is imprecise when overlap is small; the
  useful operating point is ruling large overlap in or out.
- The overlap term is additive in the *numerator*; `wᵀDw` is unaffected.

## Related work

- Bulik-Sullivan et al. 2015 (cross-trait LD score regression): the intercept
  of the `E[z₁ z₂]` regression on LD score captures sample overlap between
  two GWAS. The present detector is the same moment decomposition with the
  PGS weights `w` as a noisy proxy for the unobserved `z₁` — which is what
  makes it usable when training summary statistics are unavailable.
- PUMAS / PUMA-CUBS (Zhao et al. 2021; 2024): summary-statistic PRS
  evaluation and tuning by GWAS subsampling; the subsample-scaling complement
  noted above builds on the same idea.
- Burgess et al. / Sanderson et al. 2022 (Mendelian-randomization overlap
  bias): analogous overlap corrections in a different estimand.
