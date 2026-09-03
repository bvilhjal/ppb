# Review archive

Point-in-time audits of ppb, kept because the reasoning behind a change is worth
more than the change. They are **historical records, not current documentation**:
each describes the tree as it stood on its date, and a finding listed as open may
since have been resolved. Where one disagrees with a specification, the
specification wins — start from [`../METHOD.md`](../METHOD.md) for the current
specification and [`../../FINISHING_PLAN.md`](../../FINISHING_PLAN.md) for current
status.

| Document | Commit | Date | Outcome |
|---|---|---|---|
| [REVIEW.md](REVIEW.md) | `bd5d0d9` | 2026-07-18 | Full review — idea, theory, plan, docs, code. All findings resolved. |
| [REVIEW-2026-07-25.md](REVIEW-2026-07-25.md) | `011bdee` | 2026-07-25 | Idea, theory, implementation. Resolved; one severity corrected in place after the original claim was disproved. |
| [REVIEW-2026-08-16.md](REVIEW-2026-08-16.md) | `5d2d680` | 2026-08-16 | Gauge validation, novelty, provenance, test rigor. Addressed same day; three external blockers noted. |
| [REVIEW-2026-09-02.md](REVIEW-2026-09-02.md) | `036e6a3` | 2026-09-02 | 3 Critical · 19 Major · 46 Minor · 16 Nit over the ~24,400 new lines. All resolved 2026-09-03 (§12). |

The detail behind each row follows.

- [`REVIEW.md`](REVIEW.md) — commit `bd5d0d9`, 2026-07-18. All findings resolved.
- [`REVIEW-2026-07-25.md`](REVIEW-2026-07-25.md) — commit `011bdee`. Findings
  resolved, with one severity corrected in place after the original claim was
  disproved; two follow-ups (block diagnostics, and the overlap salvage) recorded
  at the end.
- [`REVIEW-2026-08-16.md`](REVIEW-2026-08-16.md) — commit `5d2d680`, 2026-08-16.
  Findings addressed the same day (gauge validation, novelty sweep, provenance,
  multi-seed test rigor); three items remain open as external blockers —
  regenerating the EUR pack with the new diagnostics, the two novelty leads, and
  the first real cross-ancestry run. (The two novelty leads were closed by the
  2026-08-23 follow-up sweep recorded in [`NOVELTY.md`](../NOVELTY.md); the other
  two remain open.)
- [`REVIEW-2026-09-02.md`](REVIEW-2026-09-02.md) — commit `036e6a3`,
  2026-09-02. Covers the ~24,400 lines added since `5d2d680`: both ancestry
  channels and their builders, benchmarks, snapshots and technical report; the
  FinnGen target path; the simulator expansion. **3 Critical · 19 Major · 46 Minor
  · 16 Nit** (all resolved 2026-09-03; see below). All three Criticals are on
  the FinnGen path
  (wrong `--weight-scale` in the README's own worked example; the (H2) independence
  screen failing open on absent cohort annotation; `ref`/`alt` swapped in the
  emitted table). The estimator algebra re-verified clean to machine precision; the
  findings are at the perimeter — undeclared scales, diagnostics computed but gated
  by nothing, and artifacts that predate the code producing them. Also audits the
  2026-08-16 fixes: F4a holds, F1/F3/F7 hold structurally but not quantitatively,
  F4b fails open on its default path.
  **Resolved 2026-09-03** in commit `cceb245`: all three Criticals, all 19
  Majors, and the load-bearing Minors/Nits. The FinnGen path now declares
  `dosage --hwe-genotype-sd`, emits `a1 = alt`, and fails the independence
  screen closed on absent annotation. `_as_ref_blocks` validates the correlation
  matrices the estimator is provably scale-sensitive to; the two LD/PSD gates are
  reconciled (the per-block check now reports an indefinite block instead of
  disagreeing with the write gate by 10¹¹); the gauge remap is exercised, not
  tautological; `test_gauge.py` is cross-seed mean±SE; and CI runs `ruff` with the
  BLAS threads pinned. The one exception at the time was the published
  `ancestry_report.pdf`, then stale against its `.tex`; it was rebuilt in
  `7272408` (see the follow-up entry below).
- **Follow-up review, 2026-09-03** (review of `70e8d2d..cceb245`, 51 files;
  uncommitted when received, remediated in `0ed294a`). Withheld
  release-readiness: five P0 correctness/documentation findings plus P1 artifact
  drift. The simplex face solver is now centred/SVD with direct residual
  evaluation (plus an exact-mixture regression test); rejected frequency fits
  publish `proportions: null` with `proportions_raw` retained (Estimator B's
  contract, extended to the frequency channel); `r2()`/`mse()`/`evaluate()`
  refuse a materially indefinite LD block exactly like the diagnostics path
  (low-level `quad`/`block_quads` still warn-and-report for certified int8
  references); the Estimator A scale story is one interpretation everywhere
  (`s₀ = 1−h² ≤ 1`, rescaling `z` by `c` rescales `s` by `c²`); the
  dosage→standardized multiply is stated correctly with proposed-vs-implemented
  separated; correlation validation is tiled; panels are read-only after load;
  duplicate IDs are validated before marking seen; the sdist ships the
  checkout-only test drivers (22 collection errors → 0). Artifact status:
  the Yengo snapshots were regenerated on 2026-09-03 from current code
  (inputs re-fetched and hash-verified; the 2026-08-30 files remain as
  history) and the ancestry report is revised
  and rebuilt with `tectonic` (retitle, calibrated abstract, provenance
  table, operating-characteristics survey, diagnostics-forward display, three
  generated figures). Tagged PDF remains future work: the available engine
  (Tectonic 0.17.0, LaTeX format 2021-11-15) predates `\DocumentMetadata`
  tagging, which is guarded in for newer toolchains. A drift gate
  (`scripts/check_report_drift.py`, run in CI) fails while the committed PDF
  trails any source.

