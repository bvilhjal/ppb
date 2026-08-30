# PPB results registry

Status: specification (binding) — these rules are enforced by
`tests/test_results_registry.py`. Symbols and labels:
[`../docs/NOTATION.md`](../docs/NOTATION.md).

Versioned, review-only result records — the stage-1 "leaderboard": a static
site (`site/`) generated from these files by `scripts/build_leaderboard.py`.
No submission service; records enter by maintainer-run baselines and reviewed
pull requests only (see `FINISHING_PLAN.md`, Gate D and the delivery plan).

Each `*.json` file here is a **result pack**: a JSON array of evaluation
records, one per (score × target GWAS) evaluation. (The dated `anchor-*.json`
snapshots described below are provenance objects, not packs.)

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
| `target.n_eff_basis` | how `n_eff` was derived — the median of a per-variant `N` column, a trait-level sample size, or a binary-trait effective N (computed from recorded case/control counts when they are recorded; otherwise a trait-level constant explicitly labelled "case/control counts unrecorded") |
| `target.n_eff_range` | `[min, max]` of the per-variant `N` column, when `n_eff` is a median over a varying column |
| `target.overlap` | `"none (declared)"` \| `"in-sample"` |
| `ld_ref` | LD reference id/version |
| `metrics.num`, `metrics.den` | `wᵀz`, `wᵀDw` over the target-specific joint weight/summary-statistic support |
| `metrics.r2` | registry R² from (R1) |
| `metrics.scale` | `"quantitative correlation R2"` \| `"standardized logistic-summary approximation (not liability R2)"` |
| `metrics.w_match`, `metrics.z_match` | harmonized-variant fractions |
| `metrics.w_n_ambiguous_removed`, `metrics.w_n_mismatch`, `metrics.w_n_unmatched` | per-reason weight-harmonization loss counts (from `ppb.harmonize.HarmonizeReport`) |
| `metrics.z_n_ambiguous_removed`, `metrics.z_n_mismatch`, `metrics.z_n_unmatched` | the same per-reason counts, for the target summary statistics |
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
| `overlap.q_total`, `overlap.numerator_target` | exact-support quantities used by (O6) |
| `overlap.reference` | label + R² of the reference evaluation |
| `overlap.note` | reason a correction is not applicable or was refused |
| `overlap.legacy_unidentified` | optional quarantined pre-v1 slope fields, retained for audit only and never treated as a current correction |
| `date`, `ppb_commit` | provenance |
| `environment.python`, `environment.numpy`, `environment.numba`, `environment.ppb` | interpreter and package versions the pack was generated with |

**(R1) Registry score metric.**

```text
R²_registry = metrics.num² / metrics.den
```

For a binary trait, (R1) is a standardized logistic-summary
approximation. It is not observed-scale or liability-scale case/control R².

**(G3) Block-sign-flip null.** (Defined in `docs/REAL_DATA.md`; restated
here because the registry stores its outputs.)

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

**No correction is published.** The experimental apparatus in
[`../experiments/overlap_detection.py`](../experiments/overlap_detection.py)
can still fit one,
but every correction needs an independent reference GWAS of the same trait — and
given one, evaluating the score against it is unbiased in a single line. The
condition that makes a correction valid is the condition that makes it
unnecessary, so the registry records detection only and `correctable` is not a
status here. See [`../docs/OVERLAP.md`](../docs/OVERLAP.md).

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

## Rules for records

- A result pack is a non-empty, strict-JSON array of objects. `NaN`,
  `Infinity`, booleans in numeric fields, and non-finite numeric values are
  invalid. The leaderboard loader rejects malformed pack structure and the
  field-level rules above even when it is run outside CI.
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
- No record may carry `overlap.corrected_r2`, and `correctable` is not an
  accepted status. Every status except `not_applicable` must explain itself in
  `overlap.note`.
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
- A constant `n` column must be labelled with its actual derivation. The
  binary consortium intermediates carry constants injected by
  `scripts/consortium_prep.py` — an effective N computed from recorded
  case/control counts when they are recorded, otherwise a trait-level constant
  explicitly labelled "case/control counts unrecorded" — never as though the
  source GWAS itself had shipped the constant. Those constants scale R²
  directly (`z = t/√(t²+n−2)`), so the distinction is load-bearing.
- Records dated **2026-08-16 or later** must carry `environment` with
  non-empty `python`, `numpy`, `numba`, and `ppb` versions: a pack that names
  its commit but not its interpreter is not replayable. Packs generated before
  the field existed (including `baseline-2026-07.json`) legitimately lack it —
  the same grandfathering as the block diagnostics below.
- The per-reason harmonization loss counts
  (`w_n_ambiguous_removed`/`w_n_mismatch`/`w_n_unmatched` and the `z_`
  counterparts) are optional with that same grandfathering: they are recorded
  together or not at all, are non-negative integers when present, and the
  weight-side losses can never exceed the unmatched fraction of the score
  (`w_match + (w losses)/score.n_variants ≤ 1`).
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
targets of a trait and records overlap eligibility/provenance, the per-reason
harmonization losses, and the interpreter/package `environment`; it takes ~5.5
min per trait.

These rules are enforced by `tests/test_results_registry.py`, so a malformed
pack fails CI on the pull request rather than in the Pages deploy job.
`scripts/build_leaderboard.py` re-validates the field-level rules at load time
so a pack that never met the tests is still rejected (by name and field)
before it can crash the deploy.

## Input manifest

`inputs.tsv` catalogues every external input the real-data path consumes: the
consortium GWAS source files (`scripts/consortium_prep.py`), the PGS Catalog
weight files, the bigsnpr HM3+ LD reference (`LD_with_blocks_chr*.rds` and
`map_hm3_plus.rds`), and the Pan-UKB flat files. Tab-separated, with leading
`#` comment rows; columns:

| column | meaning |
|---|---|
| `artifact` | the file name exactly as the scripts reference it |
| `role` | what the artifact is and which path consumes it |
| `source_url_or_accession` | publisher URL or accession; `UNKNOWN` where the repo records none — never guessed |
| `checksum_md5` | publisher MD5 where one is recorded (the Pan-UKB files); the literal `unrecorded` otherwise |
| `date_recorded` | when the row was written, ISO `YYYY-MM-DD` |

The registry's reproducibility rests on this file. An `unrecorded` checksum
means a replaced upstream file cannot be detected after the fact, and an
`UNKNOWN` source means the artifact cannot be re-found from the repo alone;
the manifest marks exactly those gaps instead of hiding them (review
2026-08-16, F5). Structure and script coverage are enforced by
`tests/test_panukb_manifest.py`.

## Anchor snapshots

`anchor-*.json` are dated snapshots written by
`scripts/anchor_validation.py --out results/anchor-<date>.json`: the PGS
Catalog performance fetch behind Tables 3–4 of `docs/REAL_DATA.md`, with the
fetch date recorded inside the file. They are provenance objects, not result
packs — the leaderboard loader and the schema tests skip them, and they carry
no evaluation records.

## Ancestry benchmark snapshots

`results/ancestry-frequency/` and `results/ancestry-ld/` hold dated snapshots of
the experimental ancestry-composition channels — (A1) the effect-allele-frequency
projection and (A2) the pair-product LD projection — written by
`scripts/ancestry_frequency_gwas_benchmark.py` and
`scripts/ancestry_ld_gwas_benchmark.py`. Each records the fitted mixtures, the
sign-flip null replicates, input hashes, the reference design, and the software
revision, and each directory carries a `README.md` reporting the run.

Like anchor snapshots these are provenance objects rather than result packs:
they carry no evaluation records, so no leaderboard entry derives from them.
Unlike anchor snapshots they are excluded **by location, not by name** — the
loader globs `results/*.json` non-recursively (`result_packs` in
`scripts/build_leaderboard.py`), so anything in a subdirectory is outside the
registry by construction. Keep new snapshot classes in their own subdirectory
for that reason; a file dropped directly in `results/` is parsed as a pack and
will fail the schema tests.

These channels are experimental and their estimand is the best-fitting
reference mixture, not a fraction of participants. Neither is a registry
contract, and neither feeds (R1).
