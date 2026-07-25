# Basis-aware detection and correction of shared training/target noise

**Status:** fail-closed, and now demonstrated end to end in simulation. The API
is validated by focused counterexamples and by a controlled physical-overlap
simulation in which it recovers the coupling and returns an inflated statistic
to its independent anchor. It is not yet wired into the results registry, and no
real score has supplied a basis. Existing registry corrections were produced by
the deprecated fixed-scale, variant-count model and should be treated as legacy
estimates.

> **Revised (2026-07-25).** An earlier version of this document reported that
> the identification gate "refuses every correction, including at the null" and
> recorded that as the method's operating boundary. That was a property of the
> simulation, not of the method: it ran a *marginal* trainer over *equal-sized*
> blocks, which makes the basis `q_b = tr(D_b)` the block size — the same
> constant for every block — and a *diffuse* architecture, which makes the
> reference signal near-constant too. Two constant columns cannot be separated,
> so the refusal was correct and uninformative. With per-block signal variation
> the same code identifies cleanly and corrects. See "Validation" below.

## What is identifiable

The PPB numerator is a score-weighted target summary-statistic product. If the
score's training cohort overlaps its evaluation cohort, fitted training noise
can add to that numerator. A non-overlapping reference cohort helps, but merely
subtracting reference and target products assumes their genuine score signal is
identical. That assumption is false under ordinary cohort-scale or
transportability differences.

For block `b`, the implemented model is

**Equation 1. Shared-noise block model.**

    u_Rb = s_b + e_Rb
    u_Tb = alpha s_b + gamma q_b + e_Tb.

Here `s_b` is latent genuine signal, `alpha` is a positive target/reference
signal scale, `gamma` is shared-noise coupling, and `q_b` is the trainer's
sensitivity to a declared unit of shared estimation noise. `gamma` is not proof
of literal participant overlap: shared stratification or relatedness can produce
the same moment.

The implementation profiles the latent signal and minimizes the generalized
Deming objective

**Equation 2. Generalized Deming objective.**

    Q(alpha, gamma) = sum_b [
        (u_Tb - alpha u_Rb - gamma q_b)^2
        / (sigma_Tb^2 + alpha^2 sigma_Rb^2 - 2 alpha c_TR,b)
    ].

The caller must supply block sampling-noise variances. LD quadratic forms are
not automatically sampling-noise variances; for marginal correlations they
usually require the appropriate cohort-size scaling. Target/reference noise
covariance defaults to zero and is checked blockwise for positive
semidefiniteness.

This separation follows the central lesson of cross-trait LD-score regression:
genetic covariance and shared error need distinct model components. Its
intercept also captures shared confounding, not participant overlap uniquely
([Bulik-Sullivan et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4797329/)).

## The correction basis

For a known linear trainer, let its complete effective operator be `A`, and let
`K` be the covariance template corresponding to one unit of shared error. Then

**Equation 3. Linear-trainer overlap basis.**

    w = A z_train
    q_b = tr(A_b^T K_b).

Variant count is valid only in the special identity-operator normalization. It
is not a defensible fallback for shrinkage, LD-aware, clumped, thresholded, or
otherwise selected scores.

For a rerunnable differentiable trainer, the permitted stochastic basis is

**Equation 4. Stochastic overlap-basis estimate.**

    q_hat_b = (1/R) sum_{r=1}^R [
        g_br^T {f(z + delta g_r) - f(z)}_b / delta
    ]
    E[g_r g_r^T] = K.

This is a generalized-degrees-of-freedom trace estimate; see
[Ye 1998](https://doi.org/10.1080/01621459.1998.10474094) and
[Hutchinson 1989](https://doi.org/10.1080/03610918908812806). Discontinuous
selection requires perturbation-scale stability. If only final weights remain,
the operator is unknown and the basis is `unavailable`.

**Implemented as `ppb.estimate_overlap_basis`.** It costs
`n_draws × len(deltas) + 1` trainer runs and returns an `OverlapBasis` with an
`mc_se`, or an `unavailable` basis when the stability gates fail. Two details
decide whether it measures anything:

- **The step is scaled to the data.** `deltas` are fractions of `‖z‖`; the
  perturbation is `delta × ‖z‖ / sqrt(Σ_b tr K_b) × g` and the difference
  quotient divides by the same step, preserving the units of `tr(A'K)`. A step
  much larger than `z` makes *any* trainer look like the identity — a
  thresholding trainer perturbed that hard reports the basis of a linear one.
- **One small step is not a stability check.** A hard-thresholding trainer has a
  locally *constant* Jacobian (the selection matrix), so it looks perfectly
  stable near `z` while missing the fact that which variants are selected also
  responds to the shared noise. The default `deltas` therefore span from a step
  too small to move any selection boundary to one that moves many, and the gate
  gets its evidence from the disagreement. It is validated against the exact
  `tr(A_b' K_b)` for a linear trainer (0.2–0.6% error) and refuses a p+T trainer.

`OverlapBasis` intentionally permits only two available kinds:
`linear_trace` and `jacobian_hutchinson`. Arbitrary labels are rejected because
the basis units determine both `gamma` and the amount subtracted.

## Exact support and correction

The basis object includes block values, exact score support, provenance, and a
support hash. The fit owns both the target numerator and total basis:

**Equation 5. Target numerator and total basis.**

    U_T = sum_{b in S_score} u_Tb
    Q_total = sum_{b in S_score} q_b.

A zero-noise block is excluded from fitting. If such a block has nonzero basis
mass, correction is refused rather than extrapolated. Otherwise the guarded
correction is

**Equation 6. Corrected signed numerator.**

    U_corr = U_T - gamma_hat Q_total.

The usual PPB denominator is unchanged. The signed numerator must be inspected
before squaring; correction is refused if Equation (6) reverses its sign.

## Eligibility gates

**Table 1. Correction statuses and default gates**

| Status | Meaning |
|---|---|
| `basis_unavailable` | Trainer sensitivity cannot be reconstructed |
| `excluded_basis` | A fit-excluded block has nonzero basis mass |
| `insufficient_data` | Fewer than 50 informative blocks or 15 groups |
| `nonidentifiable` | Rank failure, VIF above 10, or condition number above 30 |
| `weak_identification` | VIF above 2 or condition number above 5 |
| `heterogeneous` | Generalized-Deming residual sum per degree of freedom exceeds 2 |
| `unstable` | Boundary fit, invalid basis Monte Carlo error, or degenerate/influential jackknife |
| `not_detected` | Positive coupling does not reach the one-sided z threshold of 3 |
| `sign_reversal` | Subtraction would reverse the target numerator |
| `correctable` | Every gate passes; correction is allowed |

The VIF and condition number use the variance-whitened, through-origin design
columns for reference signal and basis. Exact proportionality is therefore
reported as nonidentifiable, whether overlap is absent or present. Delete-group
refits provide standard errors and influence checks; chromosomes are the
preferred groups. Zero or non-finite jackknife uncertainty never produces an
infinite significance statistic. The residual heterogeneity gate rejects
block-specific cohort differences that the single multiplicative `alpha`
cannot explain.

## API

```python
basis = OverlapBasis(
    values=q_by_block,
    kind="linear_trace",
    provenance="trainer artifact sha256:...; K definition:...",
    support_hash="score-and-variant-support-sha256:...",
    support=score_block_support,
)

fit = fit_overlap(
    u_target,
    u_reference,
    var_target,
    var_reference,
    basis=basis,
    groups=chromosome,
)

if fit.can_correct:
    corrected_numerator = correct_overlap_numerator(fit)
```

`overlap_slope` and `correct_numerator` remain as deprecated compatibility
helpers so old result packs can still be reproduced. They fix the signal scale
to one and use externally supplied variant count; new analyses must not use
them as the headline path.

## Validation

**Table 2. Focused validation cases**

| No. | Case | Required behavior |
|---:|---|---|
| 1 | No overlap with target signal rescaled | Estimate `alpha`; do not correct |
| 2 | Joint non-collinear signal rescaling and overlap | Recover both components |
| 3 | Reference signal exactly proportional to basis | `nonidentifiable` |
| 4 | Final weights with unknown trainer | `basis_unavailable` |
| 5 | Excluded block with nonzero basis | `excluded_basis` |
| 6 | Excluded block with zero basis | Preserve exact total and permit an otherwise valid correction |
| 7 | Invalid tiny-scale noise covariance | Reject input |
| 8 | Zero delete-group uncertainty | `unstable` |
| 9 | Correction reverses signed numerator | `sign_reversal` |
| 10 | Non-multiplicative block differences | `heterogeneous` |
| 11 | Controlled participant-overlap simulation, diffuse architecture | Refused as weakly identified |
| 12 | Controlled participant-overlap simulation, sparse architecture | Recover the coupling; return the statistic to its independent anchor |
| 13 | Same simulation at zero overlap | No correction issued |
| 14 | Hutchinson basis for a linear trainer | Match the analytic `tr(A'K)` |
| 15 | Hutchinson basis for a thresholding trainer | `unavailable` on perturbation-scale instability |

**What decides identification.** The design has two columns — the reference
signal `u_R` and the basis `q` — and both are positive and both grow with block
size, so separating them needs signal variation *at fixed block size*. A diffuse
architecture has almost none, and the gate correctly refuses. A sparse one has a
great deal: most blocks carry no causal variant and a few carry a lot, which is
uncorrelated with the deterministic basis.

**Table 3. Physical-overlap simulation** (`experiments/overlap_detection.py`,
blockwise ridge trainer, Hutchinson basis, heterogeneous LD blocks, 2 replicates).

| architecture | overlap | status | `gamma`/true | R² naive | R² corrected | independent anchor |
|---|---:|---|---:|---:|---:|---:|
| diffuse (20% causal) | 100% | `weak_identification` | 0.76 | — | refused | — |
| sparse (1% causal) | 0% | `not_detected` | — | 0.056 | none issued | 0.041 |
| sparse (1% causal) | 25% | `correctable` | 0.84 | 0.158 | **0.050** | 0.047 |
| sparse (1% causal) | 100% | `correctable` | 0.99 | 1.094 | **0.062** | 0.038 |

So the machinery works where the design is identified: a statistic inflated
nearly thirty-fold comes back to within a factor of ~1.6 of the honest anchor,
partial overlap lands on it, and the null issues nothing. Residual
over-statement tracks the residual under-estimate of `gamma`, so the corrected
value should be read as an upper bound that is far tighter than the naive one —
not as an unbiased estimate.

**The realistic regime is the identified one.** Real polygenic traits over the
1.44M-variant HM3+ reference are far sparser than the diffuse corner, and its
431 blocks span 216 to 17,304 variants rather than being equal-sized. That is an
argument for expecting identification on real data, not evidence of it: no real
score has yet supplied a basis.

A thresholded trainer is still refused, now at the basis rather than by
declaration — `estimate_overlap_basis` returns `unavailable` because its
sensitivity is not stable in the perturbation scale.

## Remaining limitations

- Equation (1) assumes cohort signal differences are approximately
  multiplicative. Block-specific changes correlated with the basis cannot be
  separated from shared noise using these products alone.
- A declared independent reference is still required. Undeclared reference
  overlap invalidates Equation (2).
- A local Jacobian can miss discontinuous model-selection effects. Perturbation
  stability and trainer reruns are mandatory for that basis kind, and the
  stability sweep must span steps large enough to move selection boundaries —
  a check at one small step passes trivially for a thresholding trainer.
- Identification depends on the genetic architecture, which the analyst does not
  choose. A sufficiently diffuse trait can be refused even with a perfect basis;
  that is the gate working, not a bug, and there is no correction available in
  that case.
- The correction is demonstrated in simulation only. `gamma` is recovered to
  roughly 15%, and the residual leaves the corrected value slightly above the
  independent anchor — an upper bound, much tighter than the naive statistic.
- Delete-group uncertainty covers genomic heterogeneity only approximately; a
  future registry integration should carry intervals for the jointly corrected
  signed numerator and squared statistic.
- Prefer preventing overlap, recovering training provenance, or retraining the
  score. Statistical correction is secondary damage control, not a magic bath.
