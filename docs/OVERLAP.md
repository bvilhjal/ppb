# Basis-aware detection and correction of shared training/target noise

**Status:** experimental and fail-closed. The new API is validated by focused
counterexamples and a controlled simulation, but it is not yet wired into the
results registry. Existing registry corrections were produced by the deprecated
fixed-scale, variant-count model and should be treated as legacy estimates.

## What is identifiable

The PPB numerator is a score-weighted target summary-statistic product. If the
score's training cohort overlaps its evaluation cohort, fitted training noise
can add to that numerator. A non-overlapping reference cohort helps, but merely
subtracting reference and target products assumes their genuine score signal is
identical. That assumption is false under ordinary cohort-scale or
transportability differences.

For block `b`, the implemented model is

\[
u_{Rb}=s_b+e_{Rb},\qquad
u_{Tb}=\alpha s_b+\gamma q_b+e_{Tb}.
\tag{1}
\]

Here `s_b` is latent genuine signal, `alpha` is a positive target/reference
signal scale, `gamma` is shared-noise coupling, and `q_b` is the trainer's
sensitivity to a declared unit of shared estimation noise. `gamma` is not proof
of literal participant overlap: shared stratification or relatedness can produce
the same moment.

The implementation profiles the latent signal and minimizes the generalized
Deming objective

\[
Q(\alpha,\gamma)=\sum_b
\frac{\left(u_{Tb}-\alpha u_{Rb}-\gamma q_b\right)^2}
{\sigma^2_{Tb}+\alpha^2\sigma^2_{Rb}-2\alpha c_{TR,b}}.
\tag{2}
\]

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

\[
w=A z_{\mathrm{train}},\qquad
q_b=\operatorname{tr}\!\left(A_b^\mathsf{T}K_b\right).
\tag{3}
\]

Variant count is valid only in the special identity-operator normalization. It
is not a defensible fallback for shrinkage, LD-aware, clumped, thresholded, or
otherwise selected scores.

For a rerunnable differentiable trainer, the permitted stochastic basis is

\[
\widehat q_b=\frac{1}{R}\sum_{r=1}^{R}
\frac{g_{br}^{\mathsf T}\{f(z+\delta g_r)-f(z)\}_b}{\delta},
\qquad E[g_rg_r^\mathsf T]=K.
\tag{4}
\]

This is a generalized-degrees-of-freedom trace estimate; see
[Ye 1998](https://doi.org/10.1080/01621459.1998.10474094) and
[Hutchinson 1989](https://doi.org/10.1080/03610918908812806). Discontinuous
selection requires perturbation-scale stability. If only final weights remain,
the operator is unknown and the basis is `unavailable`.

`OverlapBasis` intentionally permits only two available kinds:
`linear_trace` and `jacobian_hutchinson`. Arbitrary labels are rejected because
the basis units determine both `gamma` and the amount subtracted.

## Exact support and correction

The basis object includes block values, exact score support, provenance, and a
support hash. The fit owns both the target numerator and total basis:

\[
U_T=\sum_{b\in S_{\mathrm{score}}}u_{Tb},\qquad
Q_{\mathrm{total}}=\sum_{b\in S_{\mathrm{score}}}q_b.
\tag{5}
\]

A zero-noise block is excluded from fitting. If such a block has nonzero basis
mass, correction is refused rather than extrapolated. Otherwise the guarded
correction is

\[
U_{\mathrm{corr}}=U_T-\widehat\gamma Q_{\mathrm{total}}.
\tag{6}
\]

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
| 11 | Controlled participant-overlap simulation | No correction at the null; current seeded architecture is refused as weakly identified |

The controlled simulation remains deliberately modest. With two deterministic
replicates, the new model absorbs genuine target/reference scale differences
and detects the large-overlap direction, but the diffuse signal and uniform
basis are weakly separated at every overlap fraction. It does **not** recover
the known full-overlap coupling or independent-anchor accuracy adequately. The
identification gate therefore refuses every correction, including at the null,
and the former recovery assertion is retained as a strict expected-failure
operating-boundary test. The thresholded trainer is also refused because its
sensitivity operator is not reconstructed.

## Remaining limitations

- Equation (1) assumes cohort signal differences are approximately
  multiplicative. Block-specific changes correlated with the basis cannot be
  separated from shared noise using these products alone.
- A declared independent reference is still required. Undeclared reference
  overlap invalidates Equation (2).
- A local Jacobian can miss discontinuous model-selection effects. Perturbation
  stability and trainer reruns are mandatory for that basis kind.
- Delete-group uncertainty covers genomic heterogeneity only approximately; a
  future registry integration should carry intervals for the jointly corrected
  signed numerator and squared statistic.
- Prefer preventing overlap, recovering training provenance, or retraining the
  score. Statistical correction is secondary damage control, not a magic bath.
