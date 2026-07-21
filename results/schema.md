# PPB results registry

Versioned, review-only result records — the stage-1 "leaderboard": a static
site (`site/`) generated from these files by `scripts/build_leaderboard.py`.
No submission service; records enter by maintainer-run baselines and reviewed
pull requests only (see `FINISHING_PLAN.md`, Gate D and the delivery plan).

Each `*.json` file here is a **result pack**: a JSON array of evaluation
records, one per (score × target GWAS) evaluation.

## Record schema

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
| `target.n_eff` | (effective) sample size |
| `target.overlap` | `"none (declared)"` \| `"in-sample"` |
| `ld_ref` | LD reference id/version |
| `metrics.num`, `metrics.den` | `wᵀz`, `wᵀDw` over the LD-ref variants |
| `metrics.r2` | `num²/den` |
| `metrics.w_match`, `metrics.z_match` | harmonized-variant fractions |
| `overlap.role` | `"reference"` (declared non-overlapping) \| `"suspect"` (paired with a reference) \| `"suspect-unpaired"` (upper bound, no reference) |
| `overlap.gamma`, `overlap.gamma_se`, `overlap.z` | dual-target detector fit (`ppb.overlap_slope`), when a reference exists |
| `overlap.corrected_r2` | numerator-corrected R² (dense scores only) |
| `overlap.m_total` | variant count the per-variant overlap term is summed over — the **LD reference's**, not the score's (`ppb.correct_numerator`) |
| `overlap.reference` | label + R² of the reference evaluation |
| `overlap.note` | why a `suspect` carries no `corrected_r2` (sparse / borderline score) |
| `date`, `ppb_commit` | provenance |

## Rules for records

- Every record must declare `target.overlap`; in-sample records are displayed
  as **upper bounds**, never as accuracy measurements.
- `suspect` records must carry the detector fit and the corrected R² when the
  score is dense; sparse scores are flagged `upper bound` instead (see
  `docs/OVERLAP.md`). A `suspect` without a `corrected_r2` must say why in
  `overlap.note`.
- `metrics.num` and `metrics.den` must be recorded with enough significant
  digits to reproduce `metrics.r2` — a reader must be able to recompute the
  headline number. Rounding both to 4 decimals leaves small-`den` traits with
  ~1 significant figure and makes `r2` unverifiable.
- A record with a `corrected_r2` must carry `overlap.m_total`, or the
  correction cannot be checked.
- Records are immutable once merged; corrections land as new packs.

These rules are enforced by `tests/test_results_registry.py`, so a malformed
pack fails CI on the pull request rather than in the Pages deploy job.
