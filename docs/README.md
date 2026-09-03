# Organization of these documents

PPB's documentation is written to be read in an order, not grepped. This page
says what each document is for, which of them are binding, and where to start
depending on why you are here.

## What is binding, and what is not

Three kinds of document, and it matters which you are reading:

- **Specification** — what the implementation must do. [`METHOD.md`](METHOD.md),
  [`CROSS_ANCESTRY.md`](CROSS_ANCESTRY.md),
  [`SCORE_DISTRIBUTION.md`](SCORE_DISTRIBUTION.md), and
  [`../results/schema.md`](../results/schema.md). A disagreement between one of
  these and the code is a defect in one of them.
- **Record** — what was actually run, and what came out.
  [`REAL_DATA.md`](REAL_DATA.md), [`../experiments/README.md`](../experiments/README.md),
  [`../results/`](../results/). Numbers here are reproducible from committed code
  and are dated.
- **Scope and intent** — what is claimed, what is refused, and what happens next.
  [`LIMITATIONS.md`](LIMITATIONS.md), [`TRANSFERABILITY.md`](TRANSFERABILITY.md),
  [`OVERLAP.md`](OVERLAP.md), [`CALIBRATION.md`](CALIBRATION.md),
  [`../FINISHING_PLAN.md`](../FINISHING_PLAN.md), and the reviews.

[`NOTATION.md`](NOTATION.md) underlies all three: every symbol is defined there
once, and every numbered result is indexed there to the code that implements it
and the test that pins it.

## Where to start

**To understand the method**, read in this order:

1. [`NOTATION.md`](NOTATION.md) §2–3 — the symbols and the standardization
   conventions. Ten minutes, and everything afterwards is unambiguous.
2. [`METHOD.md`](METHOD.md) §1 — the estimand (M1), the exact in-sample
   identity (M2), the three hypotheses that turn it into an estimator, a table
   of what each violated hypothesis costs, and Algorithm V (§1.6), which is the
   whole evaluation in seven steps. That table is the map of every failure mode
   in the rest of these documents, and Algorithm V is where each of them enters.
3. [`CROSS_ANCESTRY.md`](CROSS_ANCESTRY.md) — what changes when the target is a
   different ancestry. Mostly: nothing changes in the algebra, and a great deal
   changes in what the inputs must be.
4. [`LIMITATIONS.md`](LIMITATIONS.md) — before believing any number.

[`SCORE_DISTRIBUTION.md`](SCORE_DISTRIBUTION.md) is a short detour off step 2:
the estimator's denominator, read on its own, is the score's variance in the
population — which is what standardizes an individual's score.

**To evaluate a score**, read [`METHOD.md`](METHOD.md) §1.6 (Algorithm V — what
happens, in what order, and why the order is not free) and §4 (conventions you
must match), then the "Command line" section of the [root README](../README.md),
and [`LIMITATIONS.md`](LIMITATIONS.md) on gauge consistency and the scale of `z`.

**To judge whether a published number means anything**, read
[`REAL_DATA.md`](REAL_DATA.md) — both the results and the caveats — then
[`OVERLAP.md`](OVERLAP.md) if the score's training cohort might intersect the
target — the failure mode that dominates all others, and the one PPB labels
rather than corrects.

**To extend the code**, read [`NOTATION.md`](NOTATION.md) §5, which maps every
numbered result to its module and its test, then the module.

## The documents

**Table 1. What each document contains.**

| document | kind | contents |
|---|---|---|
| [`NOTATION.md`](NOTATION.md) | reference | symbols, conventions, overloadings, index of results |
| [`METHOD.md`](METHOD.md) | specification | the estimand and estimator (M1–M6), the hypotheses and what violating each costs, Algorithm V, LD representation, conventions, tolerances, exercises |
| [`CROSS_ANCESTRY.md`](CROSS_ANCESTRY.md) | specification | target-ancestry inputs (X1), the gauge condition (X2), Theorem 1 on what is impossible without target data, exercises |
| [`EMPIRICAL_SD.md`](EMPIRICAL_SD.md) | design (proposed) | extending the LD-reference schema to carry empirical per-variant SDs, and the panel options for the first non-European reference — the Phase-4 prerequisite |
| [`OVERLAP.md`](OVERLAP.md) | scope | shared training/target noise: the mechanism, how it is detected, and why correction is not offered (the experimental apparatus lives in `../experiments/overlap_detection.py`) |
| [`CALIBRATION.md`](CALIBRATION.md) | scope | whether the target `z` is on the assumed scale: LD scores (C1), LD-score regression (C2), the implied rescaling (C3); implemented in `../experiments/z_calibration.py` |
| [`SCORE_DISTRIBUTION.md`](SCORE_DISTRIBUTION.md) | specification | the score's mean (P1) and variance (P2) in a population, from allele frequencies and LD; what two moments can support |
| [`REAL_DATA.md`](REAL_DATA.md) | record | the within-ancestry anchor on real GWAS, its uncertainty (G2), controls (G3–G4), and the external check against published individual-level accuracy (G5) |
| [`LIMITATIONS.md`](LIMITATIONS.md) | scope | what PPB does and does not establish; every known failure mode |
| [`TRANSFERABILITY.md`](TRANSFERABILITY.md) | scope | a negative result: why PPB measures portability but does not construct scores |
| [`ancestry_report/ancestry_report.pdf`](ancestry_report/ancestry_report.pdf) | record | technical report: estimating the ancestry composition of a GWAS from summary statistics — frequency projection and two LD-moment channels, with derivations, encoded simulation gates, and the Yengo 2022 height snapshots; LaTeX source alongside. **Stale:** the committed PDF is a build of the `.tex` as it stood before commit `036e6a3` (it states the wrong error statistic and omits the fitted-scale caveat); the `.tex` is current, and the PDF needs a manual rebuild with a LaTeX toolchain, which nothing in the repo automates |
| [`../results/schema.md`](../results/schema.md) | specification | the result-pack format (R1) and the rules CI enforces |
| [`../experiments/README.md`](../experiments/README.md) | record | each simulation, what it demonstrates, and the numbers it produces |
| [`../FINISHING_PLAN.md`](../FINISHING_PLAN.md) | intent | objective, gates, delivery plan, claims discipline |
| [`../PAPER_PLAN.md`](../PAPER_PLAN.md) | intent | manuscript plan: claim stack, preconditions, venue, display map from existing artifacts, objections, writing checklist |
| [`../archive/README.md`](../archive/README.md) | record | what the original notebook is, and why it is an oracle rather than a source |

## Reviews

Point-in-time assessments, kept because the reasoning behind a change is worth
more than the change. None is a live document; where one disagrees with a
specification above, the specification wins.

- [`../REVIEW.md`](../REVIEW.md) — commit `bd5d0d9`, 2026-07-18. All findings resolved.
- [`../REVIEW-2026-07-25.md`](../REVIEW-2026-07-25.md) — commit `011bdee`. Findings
  resolved, with one severity corrected in place after the original claim was
  disproved; two follow-ups (block diagnostics, and the overlap salvage) recorded
  at the end.
- [`../REVIEW-2026-08-16.md`](../REVIEW-2026-08-16.md) — commit `5d2d680`, 2026-08-16.
  Findings addressed the same day (gauge validation, novelty sweep, provenance,
  multi-seed test rigor); three items remain open as external blockers —
  regenerating the EUR pack with the new diagnostics, the two novelty leads, and
  the first real cross-ancestry run. (The two novelty leads were closed by the
  2026-08-23 follow-up sweep recorded in [`NOVELTY.md`](NOVELTY.md); the other
  two remain open.)
- [`../REVIEW-2026-09-02.md`](../REVIEW-2026-09-02.md) — commit `036e6a3`,
  2026-09-02. Covers the ~24,400 lines added since `5d2d680`: both ancestry
  channels and their builders, benchmarks, snapshots and technical report; the
  FinnGen target path; the simulator expansion. **3 Critical · 19 Major · 46 Minor
  · 16 Nit, none yet addressed.** All three Criticals are on the FinnGen path
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
  BLAS threads pinned. The one exception is the published
  `ancestry_report.pdf`, which is stale against the `.tex` and needs a manual
  LaTeX rebuild (no toolchain in the repo).
- **Follow-up review, 2026-09-03** (uncommitted when received; review of
  `70e8d2d..cceb245`, 51 files). Withheld release-readiness: five P0
  correctness/documentation findings plus P1 artifact drift. Remediated in the
  working tree: the simplex face solver is now centred/SVD with direct residual
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
  checkout-only test drivers (22 collection errors → 0). Remaining artifact
  work: the Yengo snapshots are labelled historical with migration records
  (inputs unavailable in this checkout) and the ancestry report is revised
  and rebuilt with `tectonic` (retitle, calibrated abstract, provenance
  table, operating-characteristics survey, diagnostics-forward display, three
  generated figures). Tagged PDF remains future work: the available engine
  (Tectonic 0.17.0, LaTeX format 2021-11-15) predates `\DocumentMetadata`
  tagging, which is guarded in for newer toolchains. A drift gate
  (`scripts/check_report_drift.py`, run in CI) fails while the committed PDF
  trails any source.

## Conventions in the prose

- Numbers quoted in a **record** document are reproducible from committed code
  at the stated commit. Where a number is a simulation input rather than a
  measurement — the +58% discovery-substitution figure is the standing example —
  the text says so at the point of quoting.
- A **specification** document states requirements even when they are not yet
  implemented; anything unimplemented is marked at the point of statement and
  appears in [`NOTATION.md`](NOTATION.md) §5 without an implementation.
- Historical and attribution notes are given where the idea is introduced, not
  collected in a bibliography, so that credit is attached to the claim it
  belongs to.
- Procedures whose *order* is normative are set out as numbered algorithms
  (**Algorithm V**, with steps **V1**, **V2**, …), and results that come with a
  proof are stated as theorems. Both keep per-document numbering, as tables do;
  see [`NOTATION.md`](NOTATION.md) §1.

## Notes on the exercises

[`METHOD.md`](METHOD.md) and [`CROSS_ANCESTRY.md`](CROSS_ANCESTRY.md) end with
exercises, each followed by its answer. They are there because those two
documents are meant to be read rather than consulted, and because several of the
claims in them are one line of algebra away from being checkable by the reader —
who will then believe the rest for a better reason than that it is written down.

The bracketed number is a difficulty rating, on the scale Knuth uses in *The Art
of Computer Programming*:

**Table 2. Difficulty ratings.**

| rating | what to expect |
|---:|---|
| 00 | immediate — answerable from the text, with no pencil and paper |
| 10 | a minute's thought |
| 20 | a quarter of an hour |
| 30 | a couple of hours |
| 40 | a substantial piece of work; here, always a tracked completion item |
| 50 | an open problem — nothing in this repository solves it |

An `M` before the number (`[M20]`) marks an exercise that is mathematical rather
than about the code. The scale is logarithmic: a `[30]` is a great deal more than
three times a `[10]`.

The `[40]` and `[50]` exercises are not rhetorical. (X3) is implemented
(`ppb.estimator.corrected_r2`, shipped 2026-08-16; the published EUR pack has
not been regenerated with it), and the first real cross-ancestry evaluation is
Phase 4 of [`../FINISHING_PLAN.md`](../FINISHING_PLAN.md). They are exercises
in the sense that a reader could do them, and open in the sense that nobody
has.
