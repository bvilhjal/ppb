# PPB Review — Idea, Theory, Plan, Documentation, Code

> **Historical snapshot — not a live document.** This is a point-in-time review
> of commit `bd5d0d9` (2026-07-18). The repository has moved on substantially
> since: the real-data evaluations, the sample-overlap detector, the results
> registry, and the stage-1 leaderboard all post-date it, and some findings here
> have since been fixed. For current status see [`FINISHING_PLAN.md`](FINISHING_PLAN.md);
> for current specification see [`docs/METHOD.md`](docs/METHOD.md).

**Date:** 2026-07-18
**Scope:** full review at commit `bd5d0d9` — the scientific idea (Witteveen
within-ancestry evaluator → cross-ancestry portability measurement), the theory
(`docs/METHOD.md`, `docs/CROSS_ANCESTRY.md`, `docs/LIMITATIONS.md`), the plan
(`FINISHING_PLAN.md`), the documentation (`README.md`, `experiments/README.md`),
and the code (`src/ppb/`, `tests/`, `experiments/`).
**Test suite:** 67 passed, 0 warnings (Python 3.14.6 + numba, conda env `ppb`);
all six experiment scripts re-run — every number quoted in the docs reproduces
exactly.

## Verdict

The idea is sound and the reframing is genuinely interesting: the Witteveen
identity `R² = (wᵀz)²/(wᵀDw)` is ancestry-agnostic in form, and supplying
target-ancestry moments `(z_B, D_B)` turns a within-ancestry benchmark into a
summary-statistics-based portability measurement. The derivation is correct,
the gauge analysis is right, the failure-mode claims are demonstrated, and the
claims discipline (simulation-only, measure-vs-predict, narrow privacy wording,
attribution) is unusually careful. **No critical issues.** One real code guard
missing (silent R² inflation on gapped block partitions), one method-status doc
gap, and a handful of minor code/doc items.

**Findings: 1 major · 7 minor · 4 nit.**

---

## Major

### M1. `BlockDiagonalLD` does not check block coverage — silently inflates R²

`src/ppb/ld_backend.py:91-114` validates block overlap and size but never checks
that the blocks *cover* `[0, m)`. With an interior gap, those variants' diagonal
mass is dropped from `wᵀDw` while `w_j·z_j` still enters the numerator via the
full vector. Reproduced: two 1×1 identity blocks at positions 0 and 2 of a
3-variant input gives `r2 = 1.125` where the correct value is `0.75` — a silent
~50% inflation of the headline metric. It requires caller misconfiguration (real
ldpred3-style partitions always tile, and zero weights at uncovered positions
are consistent), so this is a missing guard, not broken logic — but on the
headline metric it should fail loudly.

**Fix:** in `BlockDiagonalLD.__init__`, track the covered mask and raise
`ValueError` if `seen.all()` is false (or document the tiling precondition and
warn). Add a regression test with a gapped partition.

---

## Minor

### m1. Finite-sample numerator correction: hard requirement with no implementation, and the urgency is overstated

`docs/CROSS_ANCESTRY.md:76-78` lists "finite-sample numerator correction
`≈ wᵀD_B w / N_B` ... and report an SE" as hard requirement #6, but
`src/ppb/estimator.py:r2` is the raw plug-in — no correction, no SE, no test,
and no pointer in the doc that this is unimplemented. `FINISHING_PLAN.md` is
internally consistent (it is Phase-2 work and a v0.1 completion criterion), so
the gap is only the requirement list presenting it as part of the method.

Two honest calibrations:

- **The math checks out.** With `ẑ = z + ε`, `Var(ε) ≈ D/N`, the numerator is
  biased upward by `wᵀD_B w/N_B` — the documented correction is right (and with
  per-variant `n_j` it becomes the analogous sum, already half-built via
  `sumstats`).
- **The magnitude is small.** Dividing by the denominator, the absolute bias in
  R² is ≈ `1/N_B` — 5e-5 at N=20k, 1e-3 at N=1k — negligible next to typical
  R² of 0.05-0.15 even for small target-ancestry GWAS. "This matters because
  target-ancestry GWAS are usually small" (`CROSS_ANCESTRY.md:78`) overstates
  the urgency; the correction is cheap correctness hygiene, not a threat to the
  method's validity. Worth stating the `1/N` scale in the doc.
- The v0.1 criterion "a reported SE" is **unspecified**: delta method on the
  ratio, LD-block jackknife, or variant bootstrap? The plan should name the
  method before Phase 2 closes.

**Fix:** note the unimplemented status in `CROSS_ANCESTRY.md`, state the `1/N`
bias scale, and specify the SE method in `FINISHING_PLAN.md`.

### m2. `METHOD.md` names functions that do not exist in ppb

`docs/METHOD.md:88,109` says `D` "is stored block-diagonal
(recombination-aware blocks, `optimal_ld_blocks`...)" and that panel noise "is
handled by size-aware spectral shrinkage (`shrink_ld_blocks`, Marchenko-Pastur
...)". Neither function exists anywhere in the package (grep-confirmed): ppb
has no block-finding algorithm and no spectral shrinkage — `lowrank_ld` does
plain eigen-truncation. These are ldpred3-side references leaking into "the
contract for the reimplementation" in present tense.

**Fix:** mark both as ldpred3 references / roadmap items, or implement.

### m3. Gauge table gloss is imprecise

`docs/CROSS_ANCESTRY.md:48` defines `w_B` as "per-allele β from A, rescaled by
B allele frequencies". The correct rescale is the genotype-SD ratio:
`w_B,j = w_A,j · (sd_B,j / sd_A,j)` — and the doc's own gauge section
(`:56-65`) states this correctly (one empirical B gauge for all three inputs).
The table row invites the wrong `2p_B(1−p_B)`-only interpretation the gauge
section warns against.

**Fix:** "rescaled to B's empirical genotype SDs (sd_B/sd_A ratio)".

### m4. Equation-level test tolerance contradicts the spec

`docs/METHOD.md:181` requires agreement "within 1e-10 relative error ... must
pass exactly"; `tests/test_estimator.py:12` enforces `TOL = 1e-9` (absolute
below ref 1). Measured agreement is ~1.7e-16 over 20 seeds, so the code beats
both; only the test contradicts the spec.

**Fix:** tighten the test tolerance to 1e-10 relative.

### m5. `bn_freqs(fst=0)` returns silent NaN frequencies

`src/ppb/simulate.py:121-124` divides by `fst` with no validation: `fst=0`
produces `beta(inf, inf)` → NaN allele frequencies (RuntimeWarning only), which
then poisons `simulate_structured_genotypes` downstream.

**Fix:** `if fst <= 0: raise ValueError`.

### m6. `principal_components` silently returns fewer than `k` columns

`src/ppb/covariates.py:74-75`: `k > min(n, m)` silently caps (verified: k=100
on a rank-50 matrix returns shape (50, 50), no warning) — a silent change to
the covariate set the user asked for.

**Fix:** raise, or document and warn on the cap.

### m7. `read_weights` crashes on ragged rows

`src/ppb/io.py:54-59`: a short data row raises a bare `IndexError` instead of a
line-numbered `ValueError` (non-numeric weights do get a `ValueError`).

**Fix:** length-check each row and raise with the line number.

---

## Nits

| # | Location | Issue |
|---|----------|-------|
| n1 | `src/ppb/ld_backend.py:140` | `LowRankLDInt8.__init__` has an unused `m` parameter |
| n2 | `src/ppb/cli.py` | invalid input surfaces as a traceback (messages are informative; acceptable for a research CLI) |
| n3 | `src/ppb/harmonize.py:158-159` | a duplicate target row with a consumed reference candidate is counted as `n_mismatch` rather than duplicate |
| n4 | `src/ppb/simulate.py:175` | `marginal_stats` SE uses `n` where exact OLS is `n−2` (documented in the docstring; p-thresholding only) |

---

## Verified sound

### Idea and theory (re-derived independently)

- **Core identity** (`METHOD.md` §1): `corr(Xw, y)² = (wᵀz)²/(wᵀDw)` for
  standardized `y`, `z = Xᵀy/N`, `D = XᵀX/N` — algebraically exact for
  in-sample moments, which is the correct estimand for a benchmark. The
  estimator (`estimator.py`) implements exactly this and matches
  individual-level truth to ~1e-16 over 20 seeds.
- **Cross-ancestry extension** (`CROSS_ANCESTRY.md`): the identity holds with
  target moments `(z_B, D_B)`; `z_B` measured in B already encodes B's effects
  tagged through B LD, so **`r_g` has no slot** and **`var(y_B)` cancels** on
  the correlation scale — both claims verified algebraically. The "impossible
  without target data" argument (the numerator `wᵀz_B` is the irreducible gap)
  is correct.
- **Gauge analysis**: the ratio is invariant to a *global* rescale of `w` but
  not a per-variant one, so one self-consistent B standardization is required;
  the HWE `2p(1−p)` formula genuinely fails under structure/admixture. The doc
  gets this right (modulo the m3 table gloss).
- **The +58% claim**: the experiment (`experiments/cross_ancestry.py`) produces
  +57.7% overstatement for the A-only substitution at `r_g=0.8` (portability
  0.648); README/FINISHING_PLAN quote it accurately with "in simulation".
- **LD-reference regimes**: test-LD exact, training-LD biased (−9.0%/−4.9%),
  independent-LD unbiased — reproduced, and `tests/test_figure_s1.py` pins the
  ordering with numeric thresholds.
- **LR8/D8 PSD claim**: the LR8 quadratic form is a sum of squares
  (`||Uᵀw||²`-form, PSD even after int8 quantization with diagonal rowscale) —
  verified numerically (min quad > 0 over 200 random weight vectors); the
  effective diagonal is exactly 1 to 2e-16.
- **PC adjustment** (`covariates.py`, `LIMITATIONS.md:83-89`): residualizing
  both genotypes and phenotype on fixed covariates is the Frisch–Waugh-
  consistent construction; the "conservative, within-structure estimate"
  framing is accurate. Experiment: spurious stratification R² 0.078 → 0.0003
  with genetic signal preserved.
- **Per-variant N** (`sumstats.py`): `r = t/√(t²+n−2)` recovers the exact
  sample correlation (tested to 1e-10); the df simplification for
  covariate-adjusted GWAS is immaterial at GWAS scale. Uniform-N assumption
  biases R² downward as documented (tested).
- **PUMAS agreement** (`pumas.py`): conditional-normal moments match the PUMAS
  split exactly; unbiasedness over 4000 draws and PUMAS-vs-truth ≤ 0.02 are
  genuinely statistical tests, not smoke.
- **MSE identity**: `MSE = var_y − 2wᵀz + wᵀDw` — correct for standardized `y`.

### Plan and claims hygiene

- **Internal consistency**: "Current state (67 tests)", the v0.1 completion
  criteria, and the Phase 2/3 split agree — the unbuilt pieces (tri-panel
  harmonization, empirical gauge, finite-sample SE) are consistently listed as
  remaining, nowhere claimed as done. The m1/m2 doc gaps above are the only
  places the docs run ahead of the code.
- **Measure-vs-predict boundary** is stated identically and correctly in
  README, FINISHING_PLAN, CROSS_ANCESTRY, and LIMITATIONS. Positioning vs
  PRS-CSx/XPASS (construction), Popcorn/S-LDXR (`r_g` inputs), and Wang et al.
  2020 (predict branch) is accurate and fair.
- **Privacy claims** are deliberately narrow ("summary-statistics-based", no DP
  or membership-inference immunity, public test stats = public labels, Gate D
  leakage model). No overclaiming found anywhere.
- **Attribution**: the cross-ancestry direction is consistently marked as new
  to this project and not attributed to Witteveen et al.; the stewardship/
  consent gates (A) are appropriate given the senior-author situation.
- **Honest status**: "validated in simulation; not yet run on real
  cross-ancestry data" appears in every doc that makes the capability claim.

### Documentation and code

- **Every number in the docs reproduces**: all six experiment scripts re-run;
  per-variant-N, PUMAS, cross-ancestry, Fig. S1, benchmark, and PC-adjustment
  tables match `experiments/README.md` and `docs/CROSS_ANCESTRY.md` exactly.
- **README "67 tests, CI green"**: 67 tests collected and passing; CI config
  (`.github/workflows/ci.yml`, Python 3.11/3.12, `pip install -e ".[test]"`,
  `pytest -q`) matches `pyproject.toml`; the 3.14 exclusion is documented
  (numba wheels).
- **Every claimed validation has a real numeric test** (coverage map in the
  review notes): exactness, LD regimes, method concordance, PUMAS, PC
  adjustment, per-variant N, both cross-ancestry failure modes, harmonization
  edge cases (strand flips, palindromes, indels, chr aliases), CLI end-to-end.
  No trivially-asserting tests found; all simulation tests are seeded.
- **Packaging**: `pyproject.toml` matches README (Python ≥3.11; numpy + numba
  only — no other third-party imports anywhere; the hand-rolled `_norm_ppf`
  backs the "no scipy" claim); version synced; MIT license; archive preserved.

---

## Recommended actions (priority order)

1. **M1** — add the coverage check to `BlockDiagonalLD.__init__` (+ gapped-partition
   regression test).
2. **m1** — annotate requirement #6 in `CROSS_ANCESTRY.md` (unimplemented;
   bias scale ≈ `1/N_B`); name the SE method in `FINISHING_PLAN.md`.
3. **m2, m3** — fix the `METHOD.md` function references and the gauge table row.
4. **m4–m7** — tighten the estimator test tolerance; validate `fst > 0`; handle
   the PC `k` cap; line-numbered `read_weights` errors.
5. Nits at leisure.

---

## Addendum — commit `e83ca36` (LD-based transferability reweighting, negative result)

Reviewed the post-review commit adding `docs/TRANSFERABILITY.md`,
`experiments/transferability.py`, and `tests/test_transferability.py` (+ README /
FINISHING_PLAN decision records).

**The findings above are unaffected** — the commit touches none of the files the
findings live in (`ld_backend.py`, `estimator.py`, `METHOD.md`,
`CROSS_ANCESTRY.md`, `simulate.py`, `covariates.py`, `io.py`). M1 and m1–m7
stand as written.

**The new content checks out:**

- The transfer-loss decomposition (`wᵀz_B − wᵀz_A = wᵀ(D_B−D_A)β_A +
  (c_g−1)wᵀD_Bβ_A + wᵀD_B e`) is algebraically correct, and the refit-collapse
  observation (maximizing `(wᵀz_B)²/(wᵀD_B w)` gives `w ∝ D_B⁻¹z_B`, a
  B-side ridge/BLUP refit = lassosum-style pseudovalidation) is correct.
- The experiment is honest about its own trap: the "target-informed" gain is
  computed against the same in-sample `z_B` it is scored on and is labeled
  overfitting in the code. The guard tests assert the right direction on all
  three claims (LD-only degrades, target-informed only in-sample, oracle
  headroom needs causal identification), with non-trivial thresholds.
- The decision (PPB measures/diagnoses, does not construct) is consistent with
  the measure-vs-predict boundary this review verified, and the doc is candid
  about its truncated verification process. Positioning vs PolyPred/PRS-CSx/
  SuSiEx/BridgePRS is fair.

**Two new small items introduced by the commit:**

- The "67 tests" claims in `README.md:33` and `FINISHING_PLAN.md:81` are now
  stale — the suite is 70 passing (verified). Update both counts.
- `docs/TRANSFERABILITY.md`'s Decision references "the finite-sample correction"
  as if it exists — it is still unimplemented (see m1 above).

---

## Resolution (2026-07-18)

All findings addressed in the working tree. Test suite after the changes:
**75 passed** (70 + 5 new), zero warnings.

### Code

- **M1** — `BlockDiagonalLD.__init__` now raises `ValueError` when the blocks do
  not cover every variant in `[0, m)` (`src/ppb/ld_backend.py`), with a
  gapped-partition regression test
  (`tests/test_ld_backend.py::test_block_diagonal_rejects_uncovered_variants`).
- **m4** — `tests/test_estimator.py` tolerance tightened to `1e-10`, matching
  the `METHOD.md` spec (measured agreement ~1e-16).
- **m5** — `bn_freqs` validates `0 < fst < 1`
  (`tests/test_simulate.py`, new file).
- **m6** — `principal_components` raises for `k > min(n, m)` instead of
  silently capping (regression test in `tests/test_covariates.py`).
- **m7** — `read_weights` raises a line-numbered `ValueError` for ragged rows
  (regression test in `tests/test_cli.py`).
- **n1** — removed the unused `m` parameter from `LowRankLDInt8.__init__`.
- n2–n4 left as-is (review judged them acceptable/documented).

### Documentation

- **m1** — `CROSS_ANCESTRY.md` requirement 6 is now marked *(not yet
  implemented)*, states the absolute R² bias scale (`≈ 1/N_B`), and points to
  the v0.1 tracking item; `FINISHING_PLAN.md` names the SE method
  (delta-method, `SE(R²) ≈ 2√(R²/N_B)` from `Var(wᵀẑ) ≈ wᵀD_B w/N_B`);
  `TRANSFERABILITY.md`'s Decision references it as pending.
- **m2** — `METHOD.md` marks `optimal_ld_blocks` / `shrink_ld_blocks` as
  ldpred3-side (ppb takes caller-supplied blocks; `lowrank_ld` does plain
  eigen-truncation).
- **m3** — the gauge table row now reads "rescaled to B's empirical genotype
  SDs (`sd_B/sd_A`)".
- Addendum — stale "67 tests" counts updated to 70 in `README.md` and
  `FINISHING_PLAN.md` (the suite is now 75 with the new regression tests;
  update on next doc touch).
