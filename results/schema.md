# PPB results registry

Versioned, review-only result records — the stage-1 "leaderboard": a static
site (`site/`) generated from these files by `scripts/build_leaderboard.py`.
No submission service; records enter by maintainer-run baselines and reviewed
pull requests only (see `FINISHING_PLAN.md`, Gate D and the delivery plan).

Each `*.json` file here is a **result pack**: a JSON array of evaluation
records, one per (score × target GWAS) evaluation.

> **These are maintainer-run baselines, and the metric is not
> competition-safe.** `R² = (wᵀz)²/(wᵀDw)` is maximized at `w ∝ D⁻¹z`, so once
> a target's `z` and `D` are public the best-scoring submission is a single
> linear solve on the published bundle — no modelling required. In simulation
> a plain ridge refit on the released moments reports **1.45×** an honest
> score's R² at `m/N = 0.05` and **4.4×** at `m/N = 1`, with no improvement in
> true out-of-sample accuracy; PPB's real regime is `m/N ≈ 6–14`. The ridge
> parameter also doubles as a plausibility dial: a submission can be tuned to
> report ~1.35× while staying below 1, so the "R² > 1 means an assumption
> failed" diagnostic does **not** catch it. Ranking scores here is meaningful
> only because every entry is a maintainer-run baseline whose provenance is
> known. A competitive leaderboard requires a hidden target-`z` track
> (`FINISHING_PLAN.md`, Gate D), not merely reviewed submissions.

## Record schema

**Table 1. Result-record fields.**

| field | meaning |
|---|---|
| `trait` | trait label (short key) |
| `score.id` | PGS Catalog id (or other stable identifier) |
| `score.name` | score name |
| `score.training` | training cohort + method |
| `score.n_variants` | number of score variants |
| `target.gwas` | target GWAS label (study, year) |
| `target.cohort` | target cohort description |
| `target.ancestry` | target ancestry (evaluation frame) |
| `target.trait_type` | `"quantitative"` \| `"binary"` |
| `target.n_eff` | the sample size the estimator actually used to standardize `z` |
| `target.n_eff_basis` | how `n_eff` was derived — a trait-level constant, a binary-trait effective N, or the median of a per-variant `N` column |
| `target.n_eff_range` | `[min, max]` of the per-variant `N` column, when `n_eff` is a median over a varying column |
| `target.overlap` | `"none (declared)"` \| `"in-sample"` |
| `ld_ref` | LD reference id/version |
| `metrics.num`, `metrics.den` | `wᵀz`, `wᵀDw` over the target-specific joint weight/summary-statistic support |
| `metrics.r2` | registry R² from Equation 1 |
| `metrics.scale` | `"quantitative correlation R2"` \| `"standardized logistic-summary approximation (not liability R2)"` |
| `metrics.w_match`, `metrics.z_match` | harmonized-variant fractions |
| `metrics.n_variants_scored` | count of non-zero weights on the target-specific joint `w`/`z` support |
| `metrics.jackknife` | delete-one-**block** jackknife: `se`, `n_blocks`, `n_groups`, `max_variance_share` |
| `metrics.jackknife_chromosome` | the same, deleting whole chromosomes — the more conservative grouping when block sizes are uneven |
| `metrics.per_chromosome` | `{chrom: [u, v]}` partial sums, so a reader can recompute the chromosome jackknife from the pack alone |
| `metrics.sign_flip_null` | block-sign-flip negative control: `null_mean`, `z`, `ratio`, `n_blocks`, `z_ceiling` |
| `metrics.diagnostics_unavailable` | why the block diagnostics were not computed (fewer than 2 LD blocks); mutually exclusive with them |
| `overlap.role` | `"reference"` (declared non-overlapping) \| `"suspect"` (paired with a reference) \| `"suspect-unpaired"` (upper bound, no reference) |
| `overlap.method` | current contract: `"scaled_signal_eiv_v1"` |
| `overlap.status` | fit/correction eligibility from Table 2 |
| `overlap.basis` | trainer-sensitivity basis kind, provenance, and support hash; unavailable bases say so explicitly |
| `overlap.alpha`, `overlap.alpha_se` | fitted target/reference cohort-signal scale and jackknife SE |
| `overlap.gamma`, `overlap.gamma_se`, `overlap.gamma_z` | fitted shared-noise coupling and jackknife evidence |
| `overlap.q_total`, `overlap.numerator_target` | exact-support quantities used by Equation 3 |
| `overlap.corrected_r2` | basis-aware correction; permitted only when `status == "correctable"` |
| `overlap.reference` | label + R² of the reference evaluation |
| `overlap.note` | reason a correction is not applicable or was refused |
| `overlap.legacy_unidentified` | optional quarantined pre-v1 slope fields, retained for audit only and never treated as a current correction |
| `date`, `ppb_commit` | provenance |

**Equation 1. Registry score metric.**

```text
R²_registry = metrics.num² / metrics.den
```

For a binary trait, Equation 1 is a standardized logistic-summary
approximation. It is not observed-scale or liability-scale case/control R².

**Equation 2. Block-sign-flip null.**

```text
E[R²_null] = Σ_b u_b² / Σ_b v_b        z = Σ_b u_b / sqrt(Σ_b u_b²)
```

`D` is block-diagonal, so negating every weight in block `b` sends `u_b → −u_b`
and leaves `v_b` unchanged. The sign-flipped scores are therefore a family with
the same denominator and no coherent association, giving an exact null at no
extra cost. `null_mean` is the R² this score would report from block noise alone
at its own magnitudes — **read a small R² against that, not against zero**. `z`
measures how coherently the blocks agree and is bounded by `sqrt(n_blocks)`
(20.8 on the shipped 431-block reference), so it is a coherence measure on a
fixed scale, not an unbounded significance statistic. Neither this nor the
jackknife detects a uniformly mis-scaled `z`: observed and null move together
(`docs/LIMITATIONS.md`).

**Equation 3. Basis-aware numerator correction.**

```text
num_corrected = overlap.numerator_target - overlap.gamma × overlap.q_total
R²_corrected = num_corrected² / metrics.den
```

**Table 2. Current overlap statuses.**

| status | meaning |
|---|---|
| `not_applicable` | declared non-overlapping reference; no correction attempted |
| `basis_unavailable` | trainer sensitivity cannot be reconstructed; correction refused |
| `insufficient_data` | too few informative blocks or jackknife groups |
| `excluded_basis` | excluded blocks carry non-zero basis mass |
| `nonidentifiable` | cohort signal and overlap basis cannot be separated |
| `weak_identification` | separation is too weak for correction |
| `heterogeneous` | block residual heterogeneity exceeds the acceptance rule |
| `unstable` | numerical or delete-group stability rule failed |
| `not_detected` | positive overlap coupling was not detected |
| `sign_reversal` | proposed correction would reverse the signed numerator |
| `correctable` | every basis, identification, detection, and stability gate passed |

## Rules for records

- A result pack is a non-empty, strict-JSON array of objects. `NaN`,
  `Infinity`, booleans in numeric fields, and non-finite numeric values are
  invalid. The leaderboard loader rejects malformed pack structure even when
  it is run outside CI.
- `trait`, `ld_ref`, the score labels, the target labels, and
  `target.n_eff_basis` are required non-empty strings. `score.n_variants` and
  `target.n_eff` are positive integers; recorded metrics are finite real
  numbers. `metrics.den` is positive, while R² and match fractions lie in
  `[0, 1]`.
- `target.trait_type` and `metrics.scale` must agree exactly: quantitative
  traits use `"quantitative correlation R2"`; binary traits use
  `"standardized logistic-summary approximation (not liability R2)"`.
  `metrics.n_variants_scored` is a positive integer no larger than
  `score.n_variants`.
- Every record must declare `target.overlap`; in-sample records are displayed
  as **upper bounds**, never as accuracy measurements.
- Every record uses `overlap.method == "scaled_signal_eiv_v1"` and declares a
  Table 2 status. References are `not_applicable`. A `basis_unavailable`
  record must carry an unavailable basis with non-empty provenance. An
  available basis must declare `linear_trace` or `jacobian_hutchinson`, its
  provenance, and the exact score-support hash.
- Only `correctable` records may carry `overlap.corrected_r2`; they must also
  carry the finite current-fit fields needed to verify Equation 3. Every other
  fit status fails closed. Every refusal status (all except `correctable` and
  `not_applicable`) must explain the refusal in `overlap.note`.
- When present, `overlap.legacy_unidentified` is never current evidence. It
  records the old
  `fixed_signal_variant_count_v0` calculation, which fixed the signal scale
  at one and substituted LD-reference variant count for a trainer-sensitivity
  basis.
  Its numbers remain finite and auditable, but the leaderboard labels them
  legacy and never displays their `corrected_r2` as a validated correction.
  The exact-support baseline does not carry these obsolete values; they remain
  available in repository history at commit `dcd4fc3`.
- The block diagnostics are optional: packs generated before
  `scripts/regenerate_results.py` recorded them (including
  `baseline-2026-07.json`) legitimately lack them, and a record with fewer than
  2 LD blocks must instead carry `metrics.diagnostics_unavailable` explaining
  why. When present they must be internally consistent: a jackknife has at
  least 2 delete-one groups and never more groups than blocks, its `se` is
  finite and non-negative, and `max_variance_share` is a fraction; the
  sign-flip null has a positive `null_mean`, `|z| <= sqrt(n_blocks)`, and
  `ratio == metrics.r2 / null_mean`; and `per_chromosome` partial sums add back
  to `metrics.num` and `metrics.den`.
- `metrics.num` and `metrics.den` must be recorded with enough significant
  digits to reproduce `metrics.r2` — a reader must be able to recompute the
  headline number. Rounding both to 4 decimals leaves small-`den` traits with
  ~1 significant figure and makes `r2` unverifiable.
- `target.n_eff` is the sample size **the estimator used**, never the study's
  published headline N. Where the sumstats carry a per-variant `N`, no single
  number is well defined — record the median, the basis, and the range. (The
  published N can exceed every per-variant value in the HM3+-filtered file:
  GIANT BMI publishes 339,224 against a per-variant maximum of 322,153.)
- `target.n_eff_range` is present exactly when `target.n_eff_basis` is
  `"median of the per-variant N column"`. It contains two positive integers,
  and the recorded median lies within that inclusive range.
- `date` is a non-future ISO date (`YYYY-MM-DD`) and `ppb_commit` is a 7–40
  character lowercase hexadecimal Git object id. A versioned evaluation is
  identified by trait, score id, target GWAS/cohort/ancestry, date, and commit;
  that identity must be unique across all packs.
- Records are immutable once merged; corrections land as new packs. The
  documented exception is the 2026-07-22 repair of the unfinished baseline:
  commit `dcd4fc3` first quarantined the pre-v1 overlap fields, then the full
  real-data regeneration replaced the metrics with exact-support evaluations
  and removed those obsolete fields from the current pack. Both states remain
  auditable in Git history.

## Generating a pack

`python scripts/regenerate_results.py [traits...] --out results/<pack>.json`
emits records directly from the source data at full precision — never
hand-transcribe numbers from a script's printed table. One pass covers both
targets of a trait and records overlap eligibility/provenance; it takes ~5.5
min per trait.

These rules are enforced by `tests/test_results_registry.py`, so a malformed
pack fails CI on the pull request rather than in the Pages deploy job.
