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
   identity (M2), the three hypotheses that turn it into an estimator, and a
   table of what each violated hypothesis costs. That table is the map of every
   failure mode in the rest of these documents.
3. [`CROSS_ANCESTRY.md`](CROSS_ANCESTRY.md) — what changes when the target is a
   different ancestry. Mostly: nothing changes in the algebra, and a great deal
   changes in what the inputs must be.
4. [`LIMITATIONS.md`](LIMITATIONS.md) — before believing any number.

[`SCORE_DISTRIBUTION.md`](SCORE_DISTRIBUTION.md) is a short detour off step 2:
the estimator's denominator, read on its own, is the score's variance in the
population — which is what standardizes an individual's score.

**To evaluate a score**, read [`METHOD.md`](METHOD.md) §4 (conventions you must
match), the "Command line" section of the [root README](../README.md), and
[`LIMITATIONS.md`](LIMITATIONS.md) on gauge consistency and the scale of `z`.

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
| [`METHOD.md`](METHOD.md) | specification | the estimand and estimator (M1–M5), the hypotheses and what violating each costs, LD representation, conventions, tolerances |
| [`CROSS_ANCESTRY.md`](CROSS_ANCESTRY.md) | specification | target-ancestry inputs (X1), the gauge condition (X2), what is impossible without target data |
| [`OVERLAP.md`](OVERLAP.md) | scope | shared training/target noise: the mechanism, how it is detected, and why correction is not offered (the experimental apparatus lives in `../experiments/overlap_detection.py`) |
| [`CALIBRATION.md`](CALIBRATION.md) | scope | whether the target `z` is on the assumed scale: LD scores (C1), LD-score regression (C2), the implied rescaling (C3); implemented in `../experiments/z_calibration.py` |
| [`SCORE_DISTRIBUTION.md`](SCORE_DISTRIBUTION.md) | specification | the score's mean (P1) and variance (P2) in a population, from allele frequencies and LD; what two moments can support |
| [`REAL_DATA.md`](REAL_DATA.md) | record | the within-ancestry anchor on real GWAS, its uncertainty (G2), controls (G3–G4), and the external check against published individual-level accuracy (G5) |
| [`LIMITATIONS.md`](LIMITATIONS.md) | scope | what PPB does and does not establish; every known failure mode |
| [`TRANSFERABILITY.md`](TRANSFERABILITY.md) | scope | a negative result: why PPB measures portability but does not construct scores |
| [`../results/schema.md`](../results/schema.md) | specification | the result-pack format (R1) and the rules CI enforces |
| [`../experiments/README.md`](../experiments/README.md) | record | each simulation, what it demonstrates, and the numbers it produces |
| [`../FINISHING_PLAN.md`](../FINISHING_PLAN.md) | intent | objective, gates, delivery plan, claims discipline |

## Reviews

Point-in-time assessments, kept because the reasoning behind a change is worth
more than the change. Neither is a live document; where one disagrees with a
specification above, the specification wins.

- [`../REVIEW.md`](../REVIEW.md) — commit `bd5d0d9`, 2026-07-18. All findings resolved.
- [`../REVIEW-2026-07-25.md`](../REVIEW-2026-07-25.md) — commit `011bdee`. Findings
  resolved, with one severity corrected in place after the original claim was
  disproved; two follow-ups (block diagnostics, and the overlap salvage) recorded
  at the end.

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
