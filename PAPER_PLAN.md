# PPB paper plan

Status: proposed.

Prepared: 2026-08-23.

Companion to [`FINISHING_PLAN.md`](FINISHING_PLAN.md), which plans the project;
this document plans the manuscript — the claims, their preconditions, the
venue decision, a display-by-display map from artifacts that already exist, the
anticipated objections, and a writing checklist. It creates no obligation for
the code: every display maps either to a committed artifact or to an item
[`FINISHING_PLAN.md`](FINISHING_PLAN.md) already tracks.

## 1. The claim stack

Three claims, in decreasing order of strength:

1. **Impossibility.** Without a phenotype measured in the target ancestry, the
   portability `R²_B` of a fixed score is not determined by any
   discovery-side quantity — Theorem 1 of
   [`docs/CROSS_ANCESTRY.md`](docs/CROSS_ANCESTRY.md), proved, with the
   corollary that exactly one measurement (a phenotype in B) is irreducible.
2. **Measurement.** Given target-ancestry summary statistics `z_B` and a
   matched LD reference `D_B`, the realized `R²_B` of an arbitrary published
   score is measurable without individual-level records, and comes with a
   finite-sample correction (X3), uncertainty (G2 block jackknife), an exact
   null (G3 sign-flip), and a failure-mode map (gauge E4; calibration C1–C4;
   overlap detection that fails closed rather than corrects).
3. **Evidence.** The estimator is validated against individual-level truth in
   simulation, anchored on real EUR consortium targets against published
   individual-level accuracies (G5), and — pending Phase 4 — applied to the
   first real cross-ancestry measurement.

The novelty position is verdict (d) of
[`docs/NOVELTY.md`](docs/NOVELTY.md), unchanged by the 2026-08-23 follow-up
sweep: PPB is not the first to compute the number (PTL-PRS reports
target-ancestry pseudo-R² as a fallback; MIXPRS tunes on functionals of the
same moments); it is the first located work to make that number the *object of
study*. Terminology follows the claims discipline in
[`FINISHING_PLAN.md`](FINISHING_PLAN.md): "summary-statistics-based", never
"privacy-preserving", in technical claims.

## 2. Preconditions

**Table 1. Preconditions for submission**

| # | precondition | status (2026-08-23) |
|---|---|---|
| 1 | Witteveen written consent + recorded authorship agreement (Gate A; the stewardship section of `FINISHING_PLAN.md` requires it before any new publication) | **open** — nothing sent |
| 2 | First real `(z_B, D_B)` evaluation (Phase 4); needs the empirical-SD B-side LD reference — design note: [`docs/EMPIRICAL_SD.md`](docs/EMPIRICAL_SD.md) | **open** — design proposed, panel decision pending |
| 3 | EUR baseline pack regenerated with jackknife / X3 / C2 (needs the GWAS inputs re-fetched; GIANT height/BMI rows stay lower bounds until then) | **open** — data wrangling, no code gap |
| 4 | Prior-art sweep re-run before submission | **done 2026-08-23** (both leads closed, three additions; see the `NOVELTY.md` addendum); re-run once more at submission time |
| 5 | Binary-trait scale decision: keep observed-scale R² with an interpretive citation, or reinstate (M6) liability conversion (removed deliberately 2026-07-31 as a v1.0 item) | **decision needed** — five of nine registry traits are binary |

Preconditions 1–3 gate submission; 4 is continuous; 5 is a writing-time
decision with a one-day implementation cost if "reinstate" wins.

## 3. Venue

**Table 2. Venue options**

| venue | fits if | notes |
|---|---|---|
| *AJHG* (data-driven methods article) | precondition 2 lands, with the published non-EUR accuracy column as validation | closest to the Pedersen/Krebs house-lineage papers; registry + leaderboard framing acceptable |
| *PLOS Genetics* | Phase 4 lands but validation is thinner (no individual-level cross-check for the chosen target) | tolerant of methods + infrastructure framing; Pain et al. 2021 precedent |
| *Bioinformatics* (application note) | Phase 4 stalls — estimator + registry as a tool paper | the LDpred2 route; weakest claim ("measures portability" stays simulation-validated) |

Recommendation: *AJHG* if precondition 2 includes validation against
published individual-level accuracies (the `docs/REAL_DATA.md` Table 4
column); *PLOS Genetics* as the fallback; the application note only as the
no-Phase-4 route. Post to bioRxiv at submission; the Witteveen foundation is
itself an unrefereed preprint, and the manuscript must say so wherever its
within-ancestry anchor is used.

## 4. Display plan

Manuscript numbering below; every row names the committed artifact the display
is generated from. "New" means produce from existing numbers, not new science.

**Table 3. Displays**

| display | content | source artifact | status |
|---|---|---|---|
| Figure 1 | The measure-vs-predict boundary; Theorem 1's two worlds (agree on all A-side quantities and B genotypes, differ in `R²_B`) | `docs/CROSS_ANCESTRY.md` Theorem 1 + corollary | new (schematic only) |
| Table 1 | What located works do with target-ancestry summary statistics | `docs/NOVELTY.md` Table 1, trimmed to ~10 rows | exists |
| Figure 2 | Estimator validity: (a) agreement with individual-level R²; (b) method ranking preserved | `experiments/README.md` Tables 2 and 10 | exists |
| Figure 3 | Input requirements and failure modes: (a) LD-reference choice (test/train/independent); (b) wrong-ancestry-LD bias under LD divergence; (c) gauge — empirical vs HWE vs discovery SD | `experiments/README.md` Tables 1, 6–7; `experiments/gauge_validation.py` (E4) numbers | exists |
| Table 2 | Agreement with PUMAS-style repeated learning; per-variant `n` vs uniform N | `experiments/README.md` Tables 4–5 | exists |
| Table 3 | Within-ancestry real-data anchor: 15 EUR evaluations with jackknife SEs, X3, C2 intercepts; in-sample rows as labelled upper bounds; published-accuracy anchor ratios | `docs/REAL_DATA.md` + regenerated `results/` pack | needs precondition 3 |
| **Table 4** | **Centerpiece: first real cross-ancestry measurement** — published PGS × non-EUR target(s), validated against published individual-level accuracies | `scripts/cross_ancestry_eval.py` + new pack | needs precondition 2 |
| Figure 4 | Diagnostics: (a) LD-score calibration intercept and the stratification arm; (b) overlap gate statuses; (c) sign-flip null in use | `docs/CALIBRATION.md`; `experiments/README.md` Table 8; `ppb.sign_flip_null` | exists |
| Figure 5 | Why the registry is shaped as it is: the closed-form attack (Gate D) and the hidden-target-`z` requirement | `FINISHING_PLAN.md` Table 2 | exists |
| Suppl. S1–S8 | Score distribution (P1–P2); assortative-mating bias and the (P3) status; the transferability negative result; PC adjustment; LD compression (D8, low-rank); Algorithm V; Theorem 1 proof; full prior-art table | `docs/SCORE_DISTRIBUTION.md`; `experiments/README.md` Tables 3, 9; `docs/METHOD.md`; `docs/CROSS_ANCESTRY.md`; `docs/NOVELTY.md` | exists |

Note for Table 4: until precondition 2 lands, the strongest cross-ancestry
display is simulated (`experiments/cross_ancestry.py`), and the manuscript's
centerpiece is a simulation — the reason the venue table has a fallback row.

## 5. Section outline

1. **Introduction** — the portability problem (Martin et al. 2019; Ding et al.
   2023); the gap: constructors are benchmarked individually on individual-level
   data, and no measurement layer exists; the measure-vs-predict boundary;
   contributions (theorem, estimator + error theory, registry).
2. **Results** — Theorem 1; validity against individual-level truth; failure
   modes (LD reference, gauge, overlap, calibration); the EUR anchor; the
   first real cross-ancestry measurement; the registry and its leakage model.
3. **Discussion** — complement to construction methods (PRS-CSx, XPASS,
   CT-SLEB, MUSSEL, MIXPRS — they emit, PPB scores); what is deliberately not
   claimed (prediction, calibration, clinical utility, privacy); limits
   (binary scale decision, admixed targets as v1.0).
4. **Methods** — (X1)–(X3), Algorithm V, harmonization, diagnostics, registry
   schema, data provenance. Most of this text exists in
   `docs/METHOD.md`/`docs/CROSS_ANCESTRY.md` and assembles rather than gets
   written.

Target length: ~4,000 words main text; Methods as long as it needs.

## 6. Anticipated objections and responses

**Table 4. Objections**

| objection | response |
|---|---|
| "This is pseudovalidation, known since 2017." | The identity is conceded (`NOVELTY.md` verdict a). The contribution is measurement-as-end: the impossibility theorem, the finite-sample correction, the failure-mode map, and fail-closed infrastructure — none of which accompany the nearest prior art (PTL-PRS validates pseudo-R² against true R², notes slight inflation, and proceeds; MIXPRS reports individual-level accuracy). |
| "Binary R² is not liability R²." | Stated up front. Either (M6) is reinstated (one day of work, formula already documented) or values are labelled standardized-summary approximations with Lee et al. (2011) cited for interpretation — precondition 5 decides. |
| "No real cross-ancestry validation." | Precondition 2; the published-accuracy column in `docs/REAL_DATA.md` Table 4 is the validation target. Without it the paper is a tool note (Table 2, row 3). |
| "Target GWAS are small." | X3 correction with delta-method SE is shipped and validated; the jackknife carries reference-panel noise. |
| "Why a registry if the metric has a closed-form maximizer?" | That is Figure 5: the attack is quantified, the stage-1 registry is maintainer-only, and the hidden-target-`z` track is stated as a requirement, not a refinement. |
| "A 1000G-scale panel is noisy LD." | The independent-panel regime is the validated one (`experiments/README.md` Table 6, −0.095% bias); panel-size sensitivity joins the validation runs in `docs/EMPIRICAL_SD.md` §5; the in-sample (UKB) upgrade path is documented. |

## 7. Writing checklist

Roughly three weeks of part-time work once preconditions 1–3 close, in order:

1. Figures/tables driver — one `scripts/make_paper_displays.py` producing
   every display in Table 3 from the registry and the experiments (the
   experiments already print their numbers; this only assembles). 2–3 days.
2. Methods assembly from the specification docs. 2–3 days.
3. Introduction and Discussion from §1 and §3 above. 3–4 days.
4. An adversarial self-review pass in the `REVIEW-*.md` style before anyone
   outside sees it. 1 day.
5. Supplement assembly (S1–S8), references from `docs/NOVELTY.md` and the
   `FINISHING_PLAN.md` reference list. 2 days.
6. Author list, consent record, and data-availability text — blocked on
   precondition 1.

## 8. Related-work status

The 2026-08-16 sweep's two open leads are closed (2026-08-23 follow-up,
recorded in the `docs/NOVELTY.md` appendix): the Wang et al. 2023 *Cell
Genomics* lead resolved to two individual-level applied papers, and no
Pattee-group cross-ancestry paper exists. The citation-chase from PTL-PRS
added TL-PRS (Zhao et al. 2022). Verdict (d) stands: the measurement slot is
open, and the manuscript should hedge exactly as `NOVELTY.md` prescribes. One
more sweep at submission time (its own instruction), ideally through Google
Scholar, which this environment could not reach.
