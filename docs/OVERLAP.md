# Basis-aware detection and correction of shared training/target noise

Status: specification (binding). Symbols and labels:
[`NOTATION.md`](NOTATION.md). Results are labelled (O1)–(O6).

## Does it work?

**Yes — when four conditions hold. When they do not, it refuses rather than
guesses.** Three of the four are arranged by whoever trained the score; the
fourth is a property of the trait, and no amount of care supplies it.

**Table 1. What a correction requires.**

| # | Condition | Why it is needed | If absent |
|---:|---|---|---|
| 1 | An independent reference GWAS of the same trait | (O1) has two rows. With the target row alone, `alpha s_b` and `gamma q_b` are one equation in two unknowns | nothing to fit |
| 2 | The trainer, not merely its output | the subtracted quantity `q_b = tr(Phi_b^T K_b)` is a functional of the *operator*: closed form (O3), or rerunnable and differentiable (O4) | `basis_unavailable` |
| 3 | Block sampling-noise variances | (O2) is a weighted fit, and LD quadratic forms are not automatically these variances | input rejected |
| 4 | A trait sparse enough to identify | both design columns are positive and both grow with block size, so separating them needs signal variation *at fixed block size* | `weak_identification` |

Conditions 1–3 are satisfied by keeping information that exists at training
time. Condition 4 is discovered rather than chosen: you run the fit and the gate
answers. Sparse architectures supply it in abundance — most blocks carry no
causal variant and a few carry many, which is uncorrelated with the
deterministic basis — and diffuse ones do not.

**Condition 2 is the one real scores fail today.** A published PGS artifact is a
list of weights, and the operator that produced them cannot be recovered from
that list. Every score in the registry therefore fails closed as
`basis_unavailable`, and its in-sample value stands as an upper bound. That is a
provenance problem rather than a statistical one: a trainer that retains its own
operator passes.

**What it delivers when all four hold** (Table 4, in simulation): a statistic
inflated nearly thirtyfold — 1.094 against an honest 0.038 — comes back to
0.062, and a 25%-overlap case lands on its anchor to two decimals. `gamma` is
recovered to within about 15%, always from below, so the corrected value settles
at or slightly above the honest anchor. Read it as an **upper bound** far
tighter than the naive one, not as an unbiased estimate.

**What it never delivers.** `gamma > 0` is evidence of shared *noise*, not proof
of shared *people*: it is identified from a moment, and shared stratification or
cryptic relatedness produce the same moment. Nor has any of this been run on
real data — no real score has yet supplied a basis, the correction is not wired
into the results registry, and the corrections sitting in existing packs came
from the deprecated fixed-scale variant-count model and are legacy estimates.

**The failure mode is safe.** Nine of the ten outcomes in Table 2 are refusals.
The estimator does not return a quietly wrong correction; it declines, and names
the gate that stopped it.

## What is identifiable

The PPB numerator is a score-weighted target summary-statistic product. If the
score's training cohort overlaps its evaluation cohort, fitted training noise
can add to that numerator. A non-overlapping reference cohort helps, but merely
subtracting reference and target products assumes their genuine score signal is
identical. That assumption is false under ordinary cohort-scale or
transportability differences.

### Where the model comes from

The block model below is not postulated; it falls out of writing down what a
shared individual does to the numerator. Let the training GWAS have `n1`
individuals and the target GWAS `n2`, sharing `n_ov` of them, and write both
estimates as truth plus noise:

    z_train = z + eps1,    z_target = z + eps2

Sharing individuals correlates the two noise terms. For standardized data the
per-individual cross-product has covariance `K` (in practice `K = D`, since
`cov(x y)` is the LD matrix up to `var_y`), and only the shared individuals
contribute, so

    Cov(eps1, eps2) = (n_ov / (n1 n2)) K  =:  gamma K

Now let the trainer be linear, `w = Phi z_train`. The numerator PPB computes on
the exposed target is `w' z_target`, and its expectation splits cleanly:

    E[w' z_target] = E[(Phi(z + eps1))' (z + eps2)]
                   = (Phi z)' z  +  E[eps1' Phi' eps2]
                   = (Phi z)' z  +  tr(Phi' Cov(eps2, eps1))
                   = (Phi z)' z  +  gamma tr(Phi' K)

The first term is the genuine signal — what an independent reference cohort also
measures. The second is pure contamination: it does not vanish as the GWAS grows,
it is proportional to `gamma = n_ov/(n1 n2)`, and its per-block shape is
`tr(Phi_b' K_b)`, which is (O3). **Both factors are needed.** `gamma` is a single
unknown scalar; the shape `q_b` must come from the trainer, and that is why final
weights alone are not enough.

Restricting to block `b` and allowing the reference cohort's genuine signal to
differ from the target's by a scale `alpha` (different cohort sizes, different
phenotype definitions, ordinary transportability) gives the fitted model:

**(O1) Shared-noise block model.**

    u_Rb = s_b + e_Rb
    u_Tb = alpha s_b + gamma q_b + e_Tb.

Here `s_b` is latent genuine signal, `alpha` is a positive target/reference
signal scale, `gamma` is shared-noise coupling, and `q_b` is the trainer's
sensitivity to a declared unit of shared estimation noise.

Two cautions the derivation makes visible. `alpha` is not a nuisance to be set to
1: doing so attributes every cohort-scale difference to overlap. And `gamma` is
identified from a *moment*, not from participant records — shared stratification
or cryptic relatedness produce the same correlation between `eps1` and `eps2`, so
`gamma > 0` is evidence of shared noise, not proof of shared people.

The implementation profiles the latent signal and minimizes the generalized
Deming objective

**(O2) Generalized Deming objective.**

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

For a known linear trainer, let its complete effective operator be `Phi`, and
let `K` be the covariance template corresponding to one unit of shared error.
(It was written `A` before 2026-07-25; that collided with the discovery ancestry
of `CROSS_ANCESTRY.md` — see [`NOTATION.md`](NOTATION.md) §4.) Then

**(O3) Linear-trainer overlap basis.**

    w = Phi z_train
    q_b = tr(Phi_b^T K_b).

Variant count is valid only in the special identity-operator normalization. It
is not a defensible fallback for shrinkage, LD-aware, clumped, thresholded, or
otherwise selected scores.

### What `q_b` measures, and why weights do not determine it

`Phi` is the *procedure*, not its output: the map that turns training summary
statistics into weights. And `q_b` has a plain reading — **how much of block
`b`'s training noise the procedure copies into the weights.** Three cases make
that concrete.

**A trainer that copies everything.** Take the weights straight from the
training statistics, `w = z_train`. Then `Phi = I` and, since `K = D` has a unit
diagonal, `q_b = tr(D_b) = m_b`, the number of variants in the block. Every
variant's noise is passed through intact, so the block contributes its full
variant count. *This is the only case in which variant count is the right
answer* — which is precisely what the deprecated `fixed_signal_variant_count_v0`
model assumed for every score.

**A trainer that shrinks.** Ridge, or LDpred-infinitesimal, solves
`w = (D + lambda I)^{-1} z_train`, so `Phi = (D + lambda I)^{-1}` and

    q_b = tr((D_b + lambda I)^{-1} D_b) = sum_i d_i / (d_i + lambda)

over the eigenvalues `d_i` of `D_b`. That expression is the ridge **effective
degrees of freedom** — the effective number of parameters fitted in the block.
It runs from `m_b` at `lambda = 0` down toward zero as `lambda` grows: on an
8-variant block it is 8.00, 7.25, 3.97, 0.73 at `lambda = 0, 0.1, 1, 10`. A
score that shrank hard absorbed little training noise and needs little
subtracted; one that barely shrank needs nearly the full variant count. Both
have `m_b` variants in the block, and the deprecated model would have given them
the same answer.

**A trainer that selects.** Keep the `k` largest `|z_train|` and zero the rest.
Locally `Phi` is a selection matrix and `q_b` looks like a count of survivors —
but *which* variants survive is itself a response to the training noise, and
that response is invisible to a derivative taken at one point. This is why
`estimate_overlap_basis` sweeps a range of step sizes rather than taking one
small one, and why p+T is refused.

**Two scores with identical weights can need different corrections.** Suppose a
published score has 100 non-zero weights. If those 100 were the top-scoring
variants in the training GWAS, the selection responded to training noise. If
they were a panel fixed in advance — prior biology, an earlier cohort — it did
not, and `Phi` is a constant selection matrix with `q_b = tr(S_b D_b)`, the
count of selected variants in the block. **The weight vector is identical in
both cases**; the corrections differ, and the second is correctable while the
first is refused. Nothing recoverable from a list of weights distinguishes them,
which is what "final weights alone are not a basis" means in practice.

The general statement: `q_b` is a functional of the *derivative* of the training
map, and a weight vector is that map evaluated at a single point. Infinitely
many maps pass through one point with different derivatives.

For a rerunnable differentiable trainer, the permitted stochastic basis is

**(O4) Stochastic overlap-basis estimate.**

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
  quotient divides by the same step, preserving the units of `tr(Phi'K)`. A step
  much larger than `z` makes *any* trainer look like the identity — a
  thresholding trainer perturbed that hard reports the basis of a linear one.
- **One small step is not a stability check.** A hard-thresholding trainer has a
  locally *constant* Jacobian (the selection matrix), so it looks perfectly
  stable near `z` while missing the fact that which variants are selected also
  responds to the shared noise. The default `deltas` therefore span from a step
  too small to move any selection boundary to one that moves many, and the gate
  gets its evidence from the disagreement. It is validated against the exact
  `tr(Phi_b' K_b)` for a linear trainer (0.2–0.6% error) and refuses a p+T trainer.

`OverlapBasis` intentionally permits only two available kinds:
`linear_trace` and `jacobian_hutchinson`. Arbitrary labels are rejected because
the basis units determine both `gamma` and the amount subtracted.

## Exact support and correction

The basis object includes block values, exact score support, provenance, and a
support hash. The fit owns both the target numerator and total basis:

**(O5) Target numerator and total basis.**

    U_T = sum_{b in S_score} u_Tb
    Q_total = sum_{b in S_score} q_b.

A zero-noise block is excluded from fitting. If such a block has nonzero basis
mass, correction is refused rather than extrapolated. Otherwise the guarded
correction is

**(O6) Corrected signed numerator.**

    U_corr = U_T - gamma_hat Q_total.

The usual PPB denominator is unchanged. The signed numerator must be inspected
before squaring; correction is refused if (O6) reverses its sign.

## Eligibility gates

**Table 2. Correction statuses and default gates**

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

**Table 3. Focused validation cases**

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
| 14 | Hutchinson basis for a linear trainer | Match the analytic `tr(Phi'K)` |
| 15 | Hutchinson basis for a thresholding trainer | `unavailable` on perturbation-scale instability |

**What decides identification** (condition 4 of Table 1). The design has two
columns — the reference signal `u_R` and the basis `q` — and both are positive
and both grow with block size, so separating them needs signal variation *at
fixed block size*. A diffuse architecture has almost none, and the gate
correctly refuses. A sparse one has a great deal: most blocks carry no causal
variant and a few carry a lot, which is uncorrelated with the deterministic
basis.

This is worth stating plainly because getting it wrong once cost this document a
false conclusion. Before 2026-07-25 it reported that the identification gate
"refuses every correction, including at the null," and recorded that as the
method's operating boundary. The boundary belonged to the simulation, not to the
method: a *marginal* trainer over *equal-sized* blocks makes the basis
`q_b = tr(D_b)` the block size — one constant for every block — and a *diffuse*
architecture makes the reference signal near-constant as well. Two constant
columns cannot be separated, so the refusal was correct and entirely
uninformative. The same code identifies cleanly once the blocks and the
architecture vary, which is what Table 4 shows.

**Table 4. Physical-overlap simulation** (`experiments/overlap_detection.py`,
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

- (O1) assumes cohort signal differences are approximately
  multiplicative. Block-specific changes correlated with the basis cannot be
  separated from shared noise using these products alone.
- A declared independent reference is still required. Undeclared reference
  overlap invalidates (O2).
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

## Conclusion

The correction works in the sense that matters. Where the design is identified
it recovers the coupling and brings an inflated statistic back to its honest
anchor; where the design is not identified it says so, rather than returning a
number that looks like an answer.

The binding constraint is archival rather than statistical. Every gate in Table 2
can pass, and not one of them can be reached, if the operator that produced the
weights was discarded — which is the state of every published score today. So
the order of preference is unchanged: prevent the overlap; failing that, keep
the provenance that makes it correctable; failing that, retrain. Statistical
correction is damage control applied after the fact, and it is worth precisely
what the surviving record of the training procedure is worth.
