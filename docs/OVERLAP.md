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
`docs/CROSS_ANCESTRY.md`, hard requirement 6). Under overlap, `w = f(ẑ_train)`
was fit to noise correlated with `ε`, and the second term is a **first-moment,
additive inflation**.

For a trainer that is linear in the training statistics, `w ≈ A ẑ_train`
(marginal/Z-score weights: `A = I`; LDpred2-inf: `A = (D + λI)⁻¹`; LDpred2 with
a large causal fraction approximately so),

```
E[wᵀε] = (N_ov / (n₁ n₂)) · ρ_ε · tr(A D)  ≈  γ · m,
```

with `N_ov` the number of shared samples, `ρ_ε` the noise correlation of the
two traits, and `γ` the per-variant overlap coefficient. For `A = I` exactly,
`tr(D) = m` (unit diagonal) so `γ = N_ov / (n₁ n₂)`.

Note the `n₁ n₂` denominator, not `√(n₁ n₂)`: PPB works on the **marginal
correlation** scale `z = (1/N) Xᵀy`, where `Cov(ε₁, ε₂) = ρ_ε N_ov/(n₁ n₂) · D`.
The familiar `√(n₁ n₂)` belongs to the χ²/z-score scale used by cross-trait
LDSC, and would be inconsistent with the `γ = N_ov/(n₁ n₂)` above. The
simulation validates the form used here: `experiments/overlap_detection.py`
sets `gamma_true = n_ov/(n1·n2)` and the detector recovers it (see Validation).

The key structural fact: **estimation noise carries no LD structure, so the
overlap term is uniform per variant, while true signal concentrates in LD.**
That asymmetry makes overlap identifiable from `(w, z, D)` alone — no training
summary statistics required.

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
correction restores the honest anchor within ~15% even at full overlap.
It **under-corrects** there (γ̂ runs ~10% low), so the corrected R² is left
slightly *above* the honest anchor — residual inflation, not a safety margin.
A corrected R² is therefore still an upper bound on the honest value, just a
much tighter one than the naive estimate.

**Real data** (scores from `docs/REAL_DATA.md`, dual-target against the
consortium GWAS): overlap decisively detected for the dense scores
(height z = 12.2, BMI z = 12.3, T2D z = 5.1, CAD z = 5.6). The correction moves
R² from the in-sample value a substantial but incomplete fraction of the way to
the honest anchor:

| trait | R² in-sample | R² corrected | R² honest | gap closed |
|---|---:|---:|---:|---:|
| BMI | 0.4048 | 0.1019 | 0.0555 | 87% |
| T2D | 0.5089 | 0.1707 | 0.0443 | 73% |
| height | 0.8031 | 0.4515 | 0.2108 | 59% |
| CAD | 0.2313 | 0.1305 | 0.0252 | 49% |

So 49–87% of the gap (mean 67%), not all of it. Two things are mixed in the
remainder and this design cannot separate them: the detector's known ~10%
under-estimation of γ (see the simulation above), and genuine
UKBB→consortium transportability loss, which is signal the benchmark is
*supposed* to measure rather than bias to remove (see Caveats). Do not read a
corrected R² as an overlap-free measurement — it remains an upper bound.

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
