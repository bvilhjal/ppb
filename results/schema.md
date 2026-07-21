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
| `overlap.reference` | label + R² of the reference evaluation |
| `date`, `ppb_commit` | provenance |

## Rules for records

- Every record must declare `target.overlap`; in-sample records are displayed
  as **upper bounds**, never as accuracy measurements.
- `suspect` records must carry the detector fit and the corrected R² when the
  score is dense; sparse scores are flagged `upper bound` instead (see
  `docs/OVERLAP.md`).
- Records are immutable once merged; corrections land as new packs.
