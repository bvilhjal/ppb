# Shared training/target noise: detection and labelling

Status: scope — the detection/labelling policy is binding on published results;
the fitting apparatus is experimental, does not ship in the package, and lives
in [`../experiments/overlap_detection.py`](../experiments/overlap_detection.py).
Symbols and labels:
[`NOTATION.md`](NOTATION.md). Results are labelled (O1)–(O6).

## The short answer

If a score's training cohort overlaps its target GWAS, fitted training noise
adds to the PPB numerator. The reported R² is then an **upper bound**, not an
accuracy — by more than tenfold in the worst real case measured here.

**PPB detects and labels that; it does not correct it.** The fitting apparatus
in this document exists, is tested, and lives in
[`../experiments/overlap_detection.py`](../experiments/overlap_detection.py),
but it is
**not a recommended path** and the results registry will not accept a
correction from it. The reason is not that the method fails. It is that the
method is dominated:

> Every correction needs an independent reference GWAS of the same trait. But if
> you have one, evaluating the score against *it* gives you an unbiased estimate
> in one line. **The condition that makes the correction valid is the condition
> that makes it unnecessary** — see "Why correction is not offered" below.

What remains genuinely useful is the mechanism, because it says *why* an
in-sample number is untrustworthy and how far wrong it can be.

## What overlap does to the numerator

Let the training GWAS have `n1` individuals and the target `n2`, sharing
`n_ov`, and write both estimates as truth plus noise:

    z_train = z + eps1,    z_target = z + eps2

Sharing individuals correlates the two noise terms. For standardized data the
per-individual cross-product has covariance `K` (in practice `K = D`, since
`cov(x y)` is the LD matrix up to `var_y`), and only shared individuals
contribute:

    Cov(eps1, eps2) = (n_ov / (n1 n2)) K  =:  gamma K

For a linear trainer `w = Phi z_train`, the numerator PPB computes splits
cleanly:

    E[w' z_target] = E[(Phi(z + eps1))' (z + eps2)]
                   = (Phi z)' z  +  tr(Phi' Cov(eps2, eps1))
                   = (Phi z)' z  +  gamma tr(Phi' K)

The first term is genuine signal — what an independent cohort also measures. The
second is pure contamination: **it does not vanish as the GWAS grows**, it is
proportional to `gamma = n_ov/(n1 n2)`, and its per-block shape is
`tr(Phi_b' K_b)`.

Two consequences worth stating plainly. The contamination is a property of the
*training procedure*, not just of the weights — which is why it cannot be
undone from a weight list. And `gamma` is identified from a moment, not from
participant records: shared stratification or cryptic relatedness produce the
same correlation between `eps1` and `eps2`, so evidence of shared *noise* is
never proof of shared *people*.

## Detecting it

Evaluate the same score against an independent GWAS of the same trait and
compare. That is the whole method, it needs none of the apparatus below, and it
is what the registry does. The clearest real case is T2D: **0.509** against an
in-sample Pan-UKB target versus **0.044** against non-overlapping DIAGRAM — a
factor of 11.6.

Records whose target overlaps training are declared `target.overlap:
"in-sample"` and displayed as upper bounds, never as accuracy measurements
([`../results/schema.md`](../results/schema.md)).

## Why correction is not offered

Every correction below needs a reference GWAS of the same trait, declared
independent of training. Ask what that reference can be, and the case closes in
three branches. Write `alpha` for the scale by which the reference's genuine
signal differs from the target's — the parameter (O1) fits.

| the reference is… | then | so |
|---|---|---|
| exchangeable with the target (`alpha ≈ 1`) | it estimates the same quantity, unbiasedly | **use it directly** |
| a different cohort, same ancestry (`alpha ≠ 1`) | it is still a GWAS in essentially the target population | **use it directly** |
| a different ancestry (`alpha ≠ 1`) | (O1) does not hold | **correction is not licensed** |

The third row is the one that looks most promising and fails. (O1) assumes the
reference and target share a latent signal `s_b` up to one scalar. Across
ancestries the reference measures `w'D_A beta` and the target `w'D_B beta`,
which are related by a single number only if `D_A` is proportional to `D_B`
block by block — precisely what differing LD denies. The `heterogeneous` gate
should reject it, correctly.

The variance argument does not rescue it either. One might hope to spend a small
reference on estimating the single scalar `gamma`, keeping the large target's
precision for the leading term. But the correction's residual error is a
**bias** — `gamma` recovers about 15% low, the simulation lands 1.6× high, and
the output is an upper bound rather than an estimate. Bias does not shrink as
the target grows, and an unbiased estimate with a block-jackknife standard error
(G2) beats a biased bound at any sample size.

**Prefer, in order:** prevent the overlap; failing that, evaluate against an
independent target and report that; failing that, publish the in-sample value
labelled as the upper bound it is.

---

The rest of this document specifies the retained apparatus. It is
**experimental**. Read it to understand what a correction would require, not as
an invitation to attempt one.

## The fitted model (experimental)

Restricting to block `b` and allowing the reference's genuine signal to differ
from the target's by a scale `alpha`:

**(O1) Shared-noise block model.**

    u_Rb = s_b + e_Rb
    u_Tb = alpha s_b + gamma q_b + e_Tb.

`s_b` is latent genuine signal, `alpha` a positive target/reference signal
scale, `gamma` the shared-noise coupling, and `q_b` the trainer's sensitivity to
a declared unit of shared estimation noise. `alpha` is not a nuisance to be set
to 1: doing so attributes every cohort-scale difference to overlap.

The implementation profiles the latent signal and minimizes

**(O2) Generalized Deming objective.**

    Q(alpha, gamma) = sum_b [
        (u_Tb - alpha u_Rb - gamma q_b)^2
        / (sigma_Tb^2 + alpha^2 sigma_Rb^2 - 2 alpha c_TR,b)
    ].

The caller supplies block sampling-noise variances; LD quadratic forms are not
automatically such variances. This separation follows the lesson of cross-trait
LD-score regression — genetic covariance and shared error need distinct model
components, and its intercept likewise captures shared confounding rather than
participant overlap uniquely ([Bulik-Sullivan et al.
2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4797329/)).

## The correction basis

For a known linear trainer with effective operator `Phi`, and `K` the covariance
template for one unit of shared error:

**(O3) Linear-trainer overlap basis.**

    w = Phi z_train
    q_b = tr(Phi_b^T K_b).

### What `q_b` measures

`Phi` is the *procedure*, not its output, and `q_b` reads as **how much of block
`b`'s training noise the procedure copies into the weights**. Three cases:

- **Copy everything.** `w = z_train`, so `Phi = I` and `q_b = tr(D_b) = m_b`,
  the block's variant count. This is the only case in which variant count is the
  right answer — and it is what the deprecated `fixed_signal_variant_count_v0`
  model assumed for every score.
- **Shrink.** Ridge or LDpred-infinitesimal gives `Phi = (D + lambda I)^{-1}`,
  so `q_b = sum_i d_i/(d_i + lambda)` over the block's LD eigenvalues — the
  ridge **effective degrees of freedom**. On an 8-variant block that is 8.00,
  7.25, 3.97, 0.73 at `lambda = 0, 0.1, 1, 10`. A score that shrank hard
  absorbed little training noise; one that barely shrank needs nearly the full
  variant count.
- **Select.** Keep the `k` largest `|z_train|`. Locally `Phi` is a selection
  matrix, but *which* variants survive is itself a response to training noise,
  and that response is invisible to a derivative taken at one point.

**Weights do not determine `q_b`.** Suppose a score has 100 non-zero weights. If
those were the training top-100, the selection responded to training noise; if
they were a panel fixed in advance, it did not. The weight vector is identical;
one is correctable and the other refused. `q_b` is a functional of the
*derivative* of the training map, and a weight vector is that map at a single
point.

That also disposes of the tempting shortcut: for a p+T score you can read the
selected support off the weight list, so `Phi = diag(support)` looks
reconstructible. It is not admissible — that is variant-count-on-the-support,
which is (O3) under the one normalization it does not apply to.

**Table 1. Trainers and the basis they admit**, probed against
`estimate_overlap_basis` rather than classified on paper.

| method | linear in `z_train`? | verdict |
|---|---|---|
| marginal, `w = z_train` | yes, `Phi = I` | analytic; usually refused downstream as `nonidentifiable`, a constant column |
| ridge / LDpred-inf | yes, `(D_train + lambda I)^{-1}` | analytic **and** stochastic; the validated reference case |
| lassosum | no | **regime-dependent** — passes at moderate shrinkage, refused at heavy, and near the boundary not reproducible across seeds (1 of 8) |
| PRS-CS | no | **plausibly passes**; continuous shrinkage has no point mass at zero. Not verified against real PRS-CS |
| p+T / clumping+thresholding | no | refused, at 5/5 thresholds and 3/3 clump depths |
| LDpred2 grid/auto | no | refused; the spike-and-slab point mass is hard selection |
| published PGS artifact | — | `basis_unavailable` |

`linear_trace` is entirely caller-supplied — ppb ships no code computing
`tr(Phi_b^T K_b)`, and `estimate_overlap_basis` returns only
`jacobian_hutchinson` or `unavailable`. The ridge `Phi` contains `D_train`, the
*training* panel, and linearity holds only if `lambda` was fixed a priori.

For a rerunnable differentiable trainer the permitted stochastic basis is

**(O4) Stochastic overlap-basis estimate.**

    q_hat_b = (1/R) sum_{r=1}^R [
        g_br^T {f(z + delta g_r) - f(z)}_b / delta
    ]
    E[g_r g_r^T] = K.

A generalized-degrees-of-freedom trace estimate ([Ye
1998](https://doi.org/10.1080/01621459.1998.10474094), [Hutchinson
1989](https://doi.org/10.1080/03610918908812806)), implemented as
`estimate_overlap_basis` in
[`../experiments/overlap_detection.py`](../experiments/overlap_detection.py) at
a cost of `n_draws × len(deltas) + 1` = 97
trainer runs at defaults. Two details decide whether it measures anything:

- **The step is scaled to the data.** `deltas` are fractions of `‖z‖`. A step
  much larger than `z` makes *any* trainer look like the identity.
- **One small step is not a stability check.** Hard thresholding has a locally
  *constant* Jacobian, so it looks stable near `z` while missing that the
  selection itself responds to noise. The defaults span steps too small to move
  a selection boundary through steps that move many, and the gate takes its
  evidence from the disagreement — via two criteria, of which it is the
  block-pattern correlation, not the total spread, that actually catches p+T.

## Support, correction, and gates

**(O5) Target numerator and total basis.**

    U_T = sum_{b in S_score} u_Tb
    Q_total = sum_{b in S_score} q_b.

**(O6) Corrected signed numerator.**

    U_corr = U_T - gamma_hat Q_total.

The denominator is unchanged, and correction is refused if (O6) reverses the
numerator's sign. A zero-noise block is excluded from fitting; if it carries
basis mass, correction is refused rather than extrapolated. **(O6) is not a
registry contract** — no result pack may publish a corrected R².

**Table 2. Fit statuses.** Nine of the ten are refusals.

| Status | Meaning |
|---|---|
| `basis_unavailable` | Trainer sensitivity cannot be reconstructed |
| `excluded_basis` | A fit-excluded block has nonzero basis mass |
| `insufficient_data` | Fewer than 50 informative blocks or 15 groups |
| `nonidentifiable` | Rank failure, VIF above 10, or condition number above 30 |
| `weak_identification` | VIF above 2 or condition number above 5 |
| `heterogeneous` | Generalized-Deming residual per degree of freedom exceeds 2 |
| `unstable` | Boundary fit, invalid basis Monte Carlo error, or degenerate jackknife |
| `not_detected` | Positive coupling does not reach the one-sided z threshold of 3 |
| `sign_reversal` | Subtraction would reverse the target numerator |
| `correctable` | Every gate passes — an experimental outcome the registry rejects |

VIF and condition number use the variance-whitened, through-origin design
columns, so exact proportionality reports as `nonidentifiable` whether or not
overlap is present. Delete-group refits give standard errors and influence
checks, with chromosomes the preferred groups.

```python
fit = fit_overlap(u_target, u_reference, var_target, var_reference,
                  basis=basis, groups=chromosome)
```

## Validation

Focused counterexamples are enumerated in `tests/test_overlap.py` — one per gate,
plus the analytic-basis check and the p+T refusal.

**Table 3. Physical-overlap simulation** (`experiments/overlap_detection.py`,
blockwise ridge trainer, Hutchinson basis, heterogeneous LD blocks, 2 replicates).

| architecture | overlap | status | `gamma`/true | R² naive | R² corrected | independent anchor |
|---|---:|---|---:|---:|---:|---:|
| diffuse (20% causal) | 100% | `weak_identification` | 0.76 | — | refused | — |
| sparse (1% causal) | 0% | `not_detected` | — | 0.056 | none issued | 0.041 |
| sparse (1% causal) | 25% | `correctable` | 0.84 | 0.158 | 0.050 | 0.047 |
| sparse (1% causal) | 100% | `correctable` | 0.99 | 1.094 | 0.062 | 0.038 |

Read this table with the last column in view. Where the design is identified the
machinery does recover the coupling — 1.094 comes back to 0.062 — but the
independent anchor sits at 0.038, and the anchor is available by construction in
every row. That is the dominance argument in numbers.

**What decides identification.** The design has two columns, the reference
signal `u_R` and the basis `q`; both are positive and both grow with block size,
so separating them needs signal variation *at fixed block size*. Sparse
architectures supply it — most blocks carry no causal variant, a few carry many
— and diffuse ones do not.

That distinction cost this document a false conclusion once. Before 2026-07-25 it
reported that the gate "refuses every correction, including at the null" and
recorded that as the method's operating boundary. The boundary belonged to the
simulation: a *marginal* trainer over *equal-sized* blocks makes `q_b = tr(D_b)`
one constant for every block, and a *diffuse* architecture makes the reference
signal near-constant too. Two constant columns cannot be separated, so the
refusal was correct and uninformative.

## Limitations

- (O1) assumes cohort signal differences are approximately multiplicative.
  Block-specific differences correlated with the basis cannot be separated from
  shared noise using these products alone — which is why the cross-ancestry
  case, the one that would most justify a correction, is out of scope.
- A declared independent reference is still required, and undeclared reference
  overlap invalidates (O2).
- A local Jacobian can miss discontinuous model selection. The gate also does
  **not** see hyperparameter selection: an argmax over a penalty grid passes with
  spread 0.000, because the argmax does not switch under perturbations as large
  as `0.25‖z‖`.
- Passing `estimate_overlap_basis` is necessary but not sufficient —
  `fit_overlap` still rejects `mc_se/|q_total| > 0.05` as `unstable`.
- Identification depends on genetic architecture, which the analyst does not
  choose. A sufficiently diffuse trait is refused even with a perfect basis.
- Everything here is simulation. No real score has supplied a basis, and none
  can while published artifacts are weight lists.
