# Can LD be used to place PGS weights on more transferable variants?

**Question (project lead):** could the two LD matrices `D_A` (discovery) and `D_B`
(target) — and the summary-statistic R² machinery — be used to reweight/reselect a
PGS onto more transferable variants and improve cross-ancestry `R²_B`?

**Answer: a negative result.** LD-based reweighting does not work in PPB's
summary-statistic framework. PPB should stay a **measurement/diagnostic** tool and
not become a PGS constructor. This was derived, checked against prior art, and
**empirically confirmed in simulation** (`experiments/transferability.py`,
`tests/test_transferability.py`).

## The mechanism (why it seems appealing)

Realized target accuracy is `R²_B = (wᵀD_B β_B)² / (wᵀD_B w)`, so the numerator
depends on `w` only through `D_B w`: an A-placed weight "cashes out" as signal in
B only where the LD that *spent* it (`D_B`) agrees with the LD that *justified* it
(`D_A`). The transfer-loss numerator gap decomposes additively:

    wᵀz_B − wᵀz_A  =  wᵀ(D_B − D_A)β_A        (LD term — reweighting can touch this)
                    + (c_g−1)wᵀD_Bβ_A + wᵀD_B e   (effect heterogeneity, r_g<1 — irrecoverable)

So LD reweighting can only act on the first term; the `r_g < 1` part `e` is
orthogonal to anything an A-trained `w` encodes and cannot be recovered.

## Why it fails empirically

A per-variant LD-transferability score `s_j = cos(D_A row_j, D_B row_j)` (LD-only,
no phenotype) ranks variants by whether their tagging role is preserved A→B.
Reweighting by it (simulation, F_ST=0.3, m=500):

| reweighting | R²_B vs naive |
|---|---|
| LD-only shrinkage `w·s^γ` | **−3% to −9%** |
| LD-only pruning (drop low-`s`) | **−18% to −29%** |
| target-informed efficiency prune (*in-sample*) | +15% **(overfitting artifact)** |
| causal-effect oracle (`w = β_B`) | +20–90% (real headroom) |

Two hard failure modes explain this:
1. **The score can't tell a discordant *tag* from a discordant *causal* variant**
   without the phenotype, so LD-only reweighting removes signal → R²_B drops.
2. **Refit-collapse.** Maximizing `(wᵀz_B)²/(wᵀD_B w)` over `w` has optimum
   `w ∝ D_B⁻¹z_B` — a ridge/BLUP **refit in B** (i.e. lassosum/LDpred2-in-B),
   which discards the A-trained score and overfits finite-N `z_B`. The apparent
   in-sample gain above is exactly this overfitting (larger at r_g=0.8, where
   there is more target noise to fit).

And the `r_g < 1` **ceiling** is fatal for distant ancestries: in the demo the
LD-form ratio is `1.031` at both r_g=1.0 and r_g=0.8, so LD mismatch is ~3% of the
gap while effect heterogeneity is ~35%. LD reweighting cannot touch the 35%.

## Prior art (this space is crowded)

The real headroom — moving weight onto causal effects — is exactly what
fine-mapping / multi-ancestry construction methods already do: **PolyPred/PolyFun**
(tags → fine-mapped causal effects; +7–32% cross-ancestry), **PRS-CSx**,
**BridgePRS**, **SuSiEx/MultiSuSiE**, **IMPACT**, and multi-ancestry
**LDpred2/lassosum**. Optimizing a summary-stat R² objective is **lassosum
pseudovalidation** (Mak 2017) and **PUMAS**. A PPB reweighter would reinvent these
and inherit their overfitting.

## Decision

- **PPB does not reweight PGS for transferability.** The construction gains belong
  to fine-mapping/causal-effect methods; the naive LD-only version backfires.
- **PPB's role stays measurement/diagnostic**, consistent with its identity:
  - *measure* which existing placement transfers best — a target-ancestry,
    summary-stat `R²_B` selection objective for scoring candidate PGS (the
    trans-ancestry analogue of PUMAS), used with the finite-sample correction and
    held-out/PUMAS validation; and
  - *diagnose* where a PGS is LD-fragile in B via the decomposable per-block
    LD-form ratio `ρ_b = w_bᵀD_{A,b}w_b / w_bᵀD_{B,b}w_b` (grounded in Wang et al.
    2020). This is the one genuinely new, low-risk piece — an interpretable
    read-out, not a fix.

Both still require `z_B` (a target-ancestry GWAS of the trait), the hard input and
the reason cross-ancestry PRS is hard in the first place.

*Analysis basis: a derivation + prior-art + failure-mode workflow (the adversarial
verification and synthesis phases were cut off by a session limit; the empirical
simulation above is the decisive evidence and supersedes them).*
