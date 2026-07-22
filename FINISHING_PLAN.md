# PPB plan — a cross-ancestry polygenic-score portability benchmark

Status: proposed

Prepared: 2026-07-17

Revised: 2026-07-18 — **project refocused on cross-ancestry portability** (see
"Project focus" below). Earlier revision 2026-07-17 after reviewing the source
preprint and successor repository.

Revised: 2026-07-20 — **first real-data runs** (within-ancestry anchor only):
bigsnpr HM3+ European LD reference converted to the block-diagonal int8 store
(`ppb.ldref`), and public PGS Catalog scores evaluated against non-overlapping
consortium GWAS across 6 traits (R² 0.025-0.21, consistent with the
literature); the overlapping-cohort (in-sample) failure mode demonstrated
quantitatively on the same scores. See `docs/REAL_DATA.md`. Still no real
cross-ancestry result.

Revised: 2026-07-22 — **Gate-D overlap correction retained and redesigned**
(`ppb.overlap`): the fit separates target/reference signal scale from shared
noise and requires a trainer-specific sensitivity basis on the exact score
support. Identification, stability, heterogeneity, and sign gates fail closed.
The current final LDpred2 artifacts do not preserve a reconstructible trainer
operator, so they are `basis_unavailable` upper bounds and must not carry a
headline corrected R². The controlled physical simulation deliberately records
the current weak-identification boundary rather than declaring a false recovery.
Method note: `docs/OVERLAP.md`.

Revised: 2026-07-22 — **scientific and publication-path hardening**: PUMAS-style
repeated learning now refits every pseudo-training split with signal-dependent
dense Gaussian moments; evaluator scaling/support and finite-value checks are
explicit; LD stores and result packs receive strict structural and provenance
validation. Binary summary-statistic R² remains an approximation, not
liability-scale R².

Revised: 2026-07-22 — **stage-1 leaderboard**: versioned results registry
(`results/`, schema in `results/schema.md`) + static site generator
(`scripts/build_leaderboard.py`) + GitHub Pages workflow. Baselines only;
in-sample rows are displayed as upper bounds, and only overlap fits satisfying
the new basis-aware gates may be corrected. This is the "first leaderboard from
reviewed result files" — no submission service before the protocol survives
external beta (unchanged).

## Objective

Build PPB into a maintained, versioned benchmark that **measures the
cross-ancestry portability of polygenic scores** — the predictive R² of a
polygenic score in a *target ancestry* — using only summary-level data
(target-ancestry GWAS summary statistics + a matched LD reference), without
individual-level test records.

The within-ancestry summary-statistic evaluator of Witteveen et al. (2022) is the
**foundation and validation anchor**: cross-ancestry evaluation uses Equation 1
with the target-population moments (`z_B`, `D_B`) supplied for target ancestry B.
The project's new scientific contribution is turning that estimator into a
portability-measurement tool and benchmark, and validating it across ancestries.

**Equation 1. Target-ancestry summary-statistic accuracy**

    R²_B = (wᵀz_B)² / (wᵀD_Bw).

## Project focus (decided 2026-07-18)

Cross-ancestry portability is the **main focus** of the resurrected project. This
sharpens, not discards, the original goal:

- **Foundation (kept):** reproduce and steward the Witteveen et al. within-ancestry
  summary-statistic R² evaluator. It is the special case (`A = B`) and the anchor
  every cross-ancestry result is validated against.
- **Flagship (new):** measure cross-ancestry PRS portability from summary
  statistics — a distinct, high-impact contribution motivated by the well-known
  loss of PRS accuracy in non-European ancestries (Martin et al. 2019). The
  original paper is European-only; **the cross-ancestry direction is new to this
  project and must not be attributed to Witteveen et al.**
- **Honest status:** the cross-ancestry estimator is currently validated **in
  simulation against individual-level truth** (`experiments/cross_ancestry.py`,
  `docs/CROSS_ANCESTRY.md`). It has **not** been run on real cross-ancestry data.
  "Measures portability" is a validated-method claim, not a real-world result.

## Positioning: measure, do not predict

PPB **measures** realized portability given target-ancestry data; it does **not
predict** portability from discovery-ancestry data alone. Substituting
discovery-ancestry summary statistics does not estimate the target R² (it
overstates it — +58% at r_g = 0.8 in simulation). Predicting portability without a
target-ancestry phenotype is a different method class (deterministic portability
models à la Wang et al. 2020, fed by cross-population r_g estimators such as
Popcorn / S-LDXR). PPB is complementary to cross-population PRS *construction*
methods (PRS-CSx, XPASS): they build weights, PPB scores whatever they emit. See
`docs/CROSS_ANCESTRY.md`.

**Decided (2026-07-18): PPB does not construct or reweight PGS.** Whether the two
LD matrices could reweight a score onto more transferable variants was
investigated and **declined** — LD-only reweighting empirically *degrades* R²_B
and target-informed reweighting collapses to an overfitting B-refit; the real
gains belong to fine-mapping methods (PolyPred etc.). PPB *measures* and
*diagnoses* transferability, not builds scores. See `docs/TRANSFERABILITY.md`.

## Implementation approach (decided 2026-07-17): reimplement from theory

The evaluator is rebuilt from the method described in the preprint rather than
ported from the historical notebook or the successor package. Reimplementing from
the published equations upgrades the golden-result check from a repackaging into a
genuine independent replication, and yields a clean installable package. The
legacy code and datasets are retained strictly as a **reference oracle** for
differential testing — never copied. The estimator, conventions, and tolerances
are specified in `docs/METHOD.md`; the cross-ancestry extension in
`docs/CROSS_ANCESTRY.md`.

**Stack and LD handling (decided 2026-07-17).** The evaluator is a Python package
named `ppb`, `numpy` + **`numba`** for the performance kernels (no scipy in the
core). The estimator only needs `wᵀz` and `wᵀDw`, so `D` is never materialised
densely: it is stored block-diagonal with **D8** (packed int8 dense) and **LR8**
(int8 low-rank `R ≈ U Uᵀ`) representations mirroring the local `ldpred3` project,
with numba sweep kernels. Low-rank factors are PSD by construction, so `wᵀDw ≥ 0`.
Toolchain: a dedicated `ppb` conda env on **Python 3.14** (Python 3.14.6,
numpy 2.4.6, numba 0.66.0, pytest, OpenBLAS). Gotcha: `conda create python=3.14`
resolves to the **free-threaded** build (`*_cp314t`), under which numpy/MKL matmul
hard-crashes (exit 127); pin `python=3.14.*=*cp314`, and keep `@`/`np.dot` out of
`njit`.

## Current state (2026-07-22)

**Built and tested locally (CI targets Python 3.11/3.12):**

- Core estimator (`r2`, `mse`) and LD backends: dense, block-diagonal, low-rank
  (LR8), and int8 (D8/LR8, ~8× smaller), validated against individual-level
  truth at representation-appropriate tolerances. The LD loader now checks exact
  tiling/coverage, offsets, dtypes, annotations, packed diagonals, LR8 zero rows,
  and low-rank definiteness; large D8 blocks are not claimed to have a universal
  PSD certificate.
- Allele harmonization (`harmonize`), covariate/PC adjustment (`covariates`),
  per-variant sample sizes (`sumstats`), a CLI (`ppb evaluate`) and `.npz` bundle
  format. Evaluation requires an explicit dosage/standardized weight scale, uses
  coherent joint weight/summary-statistic support, and rejects non-finite inputs
  and degenerate residual covariates.
- Simulation harness: block-AR(1) LD, diploid genotypes, Balding-Nichols
  two-population structure.
- Validated demonstrations (all encoded as tests): within-ancestry LD-reference
  behaviour (Fig. S1), cross-method concordance/ranking (Fig. 1 / Table 1 style),
  PC adjustment removing stratification, per-variant-N correction, PUMAS-style
  repeated learning that refits each pseudo-training split, and **cross-ancestry
  portability** (`experiments/cross_ancestry.py`).
- Basis-aware training/target shared-noise detection and guarded numerator
  correction (`ppb.overlap`, Gate D). Correction requires an identified trainer
  basis; the current LDpred2 final weights fail closed as `basis_unavailable`, and
  the physical simulation documents a weak-identification boundary
  (`docs/OVERLAP.md`).
- Stage-1 leaderboard: the versioned results registry (`results/`, schema
  enforced by `tests/test_results_registry.py`) regenerated end-to-end from
  source data by `scripts/regenerate_results.py`, rendered to a static site by
  `scripts/build_leaderboard.py`. Pack validation rejects non-finite metrics,
  malformed structures, inconsistent identities, and incomplete provenance before
  deployment.

**Historical starting point (for provenance):** this repo began as a single
~1.4 MB notebook (now `archive/PPB.ipynb`); the working legacy code/data live in
`mennowitteveen/pgsbenchmark`. The source is an unpublished 2022 bioRxiv preprint
(doi:10.1101/2022.10.10.510645, CC-BY; authors Witteveen, Pedersen, Meijsen,
Andersen, Privé, Speed, Vilhjalmsson — **Vilhjalmsson is senior author**). The
published European benchmark data is two fragile Google Drive HDF5 files
(validation ~10K, test ~352K UK-Biobank-derived; 8 traits). Because that test
data is already public, a leakage-resistant *competitive* benchmark is a redesign,
not a repackaging (Gate D).

**Real-data progress (2026-07-20):**

- The bigsnpr HM3+ European LD reference (Privé, figshare 21305061) is
  converted to the project's block-diagonal int8 store
  (`scripts/bigsnpr_ldref_to_ppb.py`, loader `ppb.read_ldref`): 1,444,196
  variants, 431 blocks, exactly block-diagonal, verified against the source
  matrices to quantization error.
- First real-data evaluations done (within-ancestry anchor, Phase-1/2
  overlap): public PGS Catalog `portability-ldpred2` scores vs six
  non-overlapping consortium GWAS — R² 0.21 (height), 0.10 (LDL), 0.056 (BMI),
  0.044 (T2D), 0.042 (breast cancer), 0.025 (CAD) — plus the in-sample
  (overlapping-cohort) failure mode quantified (same scores vs Pan-UKB GWAS:
  severe inflation). Binary-trait values are standardized summary-statistic
  approximations, not liability-scale R²; the final-weight overlap basis is
  unavailable, so no corrected value is claimed. Details and provenance in
  `docs/REAL_DATA.md`.

## Stewardship and provenance

This repository is an attempt to finish and extend Menno Witteveen's unfinished
project after he left science, led here by an existing senior author of the
original preprint. It is a stewardship continuation of the within-ancestry method
**and** a new cross-ancestry extension; preserving the record does not imply
Witteveen's endorsement, ownership of later changes, or current involvement, and
the cross-ancestry direction is not attributed to the original authors. The
project will:

- preserve the original repositories, history, licences, copyright, and citations,
  keeping code (MIT), the preprint manuscript (CC-BY), and the UK-Biobank-derived
  datasets (governed by UK Biobank access terms, *not* CC-BY) correctly
  distinguished;
- credit Witteveen and the original collaborators for the method, benchmark,
  software, and datasets they created, and clearly mark the cross-ancestry
  portability work as new to this project;
- request repository access, data backups, and handover context without making
  progress depend on Witteveen returning to active research;
- obtain the lead author's written consent to continue, and a recorded authorship
  agreement, before any new publication; and
- record the new maintainers and when they assume responsibility for releases,
  corrections, governance, and support.

## Product definition

### v0.1 — validated portability estimator (a tool)

Complete when:

- the estimator is an installable package with a CLI and a bundle schema that
  carries **per-ancestry** LD, allele frequencies, and per-variant `n`;
- the cross-ancestry estimator is **validated in simulation against
  individual-level truth** (exact with matched LD, unbiased with an independent
  target-ancestry LD panel, recovering the r_g<1 / MAF-LD portability loss), and
  the two failure modes (wrong-ancestry LD; discovery-sumstats substitution) are
  documented and tested;
- the **within-ancestry Witteveen result is reproduced** as the foundational
  anchor within a declared tolerance;
- the real-data-critical pieces exist and are tested: **tri-panel allele
  harmonization**, **empirical (in-sample) per-ancestry standardization** (not the
  HWE `2p(1−p)` formula, which breaks for admixed targets), the **finite-sample
  numerator correction** `≈ wᵀD_B w / N_B` with a reported delta-method SE
  (`Var(wᵀẑ) ≈ wᵀD_B w / N_B`, i.e. `SE(R²) ≈ 2√(R²/N_B)`), and per-variant `n`;
- synthetic fixtures, equation-level tests, and CI pass;
- scope, limitations, privacy claims, and the measure-vs-predict boundary are
  documented; the historical notebook is preserved out of production.

v0.1 does **not** require real cross-ancestry data; it is a method-and-tool
release validated in simulation plus the within-ancestry anchor.

### v1.0 — cross-ancestry portability benchmark (a product)

Complete when:

- the benchmark measures PRS portability across **at least two target ancestries
  including one non-European**, plus at least one **admixed** target, on real
  data with matched LD panels;
- datasets, protocols, baseline portability results, and a leaderboard are
  immutable and versioned; an open development track and a leakage-resistant
  hidden-test track exist;
- uncertainty and known failure modes are validated against **individual-level
  target-ancestry** evaluation;
- at least three independent researchers reproduce baseline portability numbers,
  and at least two external method teams submit without maintainer intervention.

## Scope

### Included in the v0.1 track

- The ancestry-agnostic estimator over the compact int8 block LD store.
- Exact (matched target LD) and approximate (independent target-ancestry panel)
  cross-ancestry evaluation, plus the within-ancestry anchor.
- Tri-panel harmonization, per-ancestry empirical standardization, per-variant `n`,
  finite-sample numerator correction with an SE, PC/covariate adjustment within B.
- Simulation-based cross-ancestry validation (Balding-Nichols ancestries, r_g<1).
- A machine-readable, per-ancestry input and result schema.

### Deferred to the v1.0 track (needs real data / more research)

- Real non-European and admixed target cohorts; local-ancestry handling.
- Binary-trait portability metrics and calibration.
- A hosted submission service and leaderboard.
- Nonlinear models, survival outcomes, clinical-utility claims.

## Delivery plan

**Table 1. Delivery plan**

| Phase | Work | Exit criterion | Estimate |
|---|---|---|---:|
| 0. Ownership & demand | Confirm maintainers/fork, Witteveen consent + authorship, interview PRS-portability method developers, inventory candidate target-ancestry datasets | Named owners, writable repo, consent recorded, a data-availability map, go/no-go | 1-2 weeks |
| 1. Foundation anchor | Recover the legacy European datasets as an oracle; reproduce one named within-ancestry published result | One checksummed dataset and one agreed golden result reproduced by the new code | 1-2 weeks |
| 2. v0.1 core (mostly done) | Finish the real-data-critical pieces (tri-panel harmonization, empirical per-ancestry gauge, finite-sample SE), per-ancestry bundle schema/CLI, docs, CI | A clean machine runs the synthetic within- and cross-ancestry examples; `v0.1.0` tagged | 2-3 weeks |
| 3. Cross-ancestry scientific validation | Simulate and quantify: LD-reference ancestry mismatch, MAF/gauge, finite-sample bias/coverage, admixture, r_g and causal-effect heterogeneity, per-variant N; compare with individual-level truth and with PUMAS | Predefined bias/coverage thresholds pass; operating boundaries documented | 6-8 weeks |
| 4. Real-data cross-ancestry beta | Acquire ≥1 non-European target GWAS + matched LD panel; measure portability of published PRS; validate against an individual-level target cohort | Real portability numbers reproduced within tolerance against individual-level truth | 6-10 weeks |
| 5. v1.0 release | Add a second/admixed ancestry track, leaderboard, governance, data review, DOI release, paper | External acceptance criteria pass; versioning/correction policy exists | 4-6 weeks |

Phase 3 is a research programme (a likely publication) that runs in parallel with
and after the v0.1 release, not a gate on shipping v0.1. Phases 0-1 depend on a
third party and are timeboxed (Gate A); the simulation, code, and validation work
block on nothing external and proceed regardless.

## Scientific validation

The validation plan will be frozen before the main experiments. It must cover,
against individual-level truth wherever possible:

1. **Within-ancestry anchor:** numerical equality with individual-level Pearson
   R² when matched target cross-products and identical preprocessing are used.
2. **Cross-ancestry core:** exactness with matched target-ancestry LD;
   unbiasedness with an independent target-ancestry panel; recovery of the
   portability loss as a function of r_g and F_ST.
3. **LD-reference ancestry mismatch:** the dominant cross-ancestry error; quantify
   the bias as the LD-form ratio `wᵀD_A w / wᵀD_B w` grows with F_ST and PGS density.
4. **Standardization / gauge:** empirical vs HWE per-ancestry scaling under
   structure and admixture; monomorphic-in-target handling.
5. **Finite-sample:** the upward numerator bias `≈ wᵀD_B w / N_B` at small target N,
   its correction, and interval coverage — target-ancestry GWAS are small.
6. **Admixture, sample overlap/relatedness, per-variant N, allele/build errors,
   imputation quality, and population structure within the target.**
7. **Metrics beyond rank correlation:** bias, RMSE, interval coverage, top-method
   selection probability, top-model regret.
8. **Comparison** with individual-level truth, PUMAS, quasi-correlation, and
   SummaryAUC where applicable.

Baselines should include fixed versions of P+T/SCT, lassosum2, LDpred2, PRS-CS,
SBayesR, **PRS-CSx / XPASS** (cross-population construction), and an ensemble.

## Benchmark protocol

- Publish an open development bundle; never expose the final hidden-test target
  phenotypes or the sufficient statistics used for a competitive leaderboard.
- Accept PGS Catalog-compatible per-allele weights (harmonized and rescaled to the
  target-ancestry gauge) or a reproducible container.
- Freeze, per benchmark version: genome build, variant set, allele conventions,
  **per-ancestry LD panels and allele frequencies**, target-ancestry definition,
  standardization gauge, finite-sample correction, tuning budget, submission quota,
  metrics, uncertainty method, exclusions, and ranking procedure.
- Store large artifacts in immutable DOI-backed storage with an object-store
  mirror; keep only manifests and tiny fixtures in Git.
- Record data version, code commit, parameters, environment/container digest,
  runtime, and resources for every baseline and submission.
- Build the first leaderboard from reviewed result files; no custom web service
  before the protocol survives external beta testing.

## Claims and terminology

Primary claim (narrow):

> PPB measures the cross-ancestry portability of a linear polygenic score — its
> predictive R² in a target ancestry — from target-ancestry GWAS summary
> statistics and a matched LD reference, without individual-level test records.
> The within-ancestry case is the special case A = B.

- "Privacy-preserving" must not imply differential privacy, cryptographic
  protection, or immunity to membership inference; use "summary-statistics-based"
  or "individual-level-data-free" in technical claims.
- PPB evaluates predictive **association / portability**; it does not establish
  calibration, clinical utility, causality, or individual privacy.
- PPB **measures** portability given target-ancestry data; it does **not predict**
  it from discovery-ancestry data alone.
- The cross-ancestry capability is currently **simulation-validated**; do not
  claim a real-world cross-ancestry result until Phase 4.
- The underlying within-ancestry work is an unpublished 2022 preprint; state
  claims at preprint strength until peer-reviewed.

## Decision gates

### Gate A: ownership & consent

Ask Witteveen for access/transfer, backups, handover context, consent to continue,
and an authorship agreement. Do not require his return to science; timebox at three
weeks, then proceed with a controlled fork preserving history and attribution
while the consent/credit request continues in parallel.

### Gate B: data — now target-ancestry data

Two data needs, distinct:
- **Foundation:** recover the legacy European datasets (two Google Drive HDF5
  files or a copy from the authors) as the within-ancestry oracle; checksum,
  verify provenance, re-host DOI-backed. (`pgsbenchmark.org` is out of scope.)
- **Flagship (v1.0):** a **non-European target-ancestry GWAS of a benchmark trait
  plus a matched (ideally in-sample) LD panel**, and — for validation — an
  individual-level target-ancestry cohort. This is the binding constraint for a
  real cross-ancestry result and is harder to source than the European data; map
  availability (e.g. Biobank Japan, PAGE, H3Africa, UGR, All of Us, admixed
  cohorts) in Phase 0. If no target-ancestry data is obtainable, v0.1 (the
  simulation-validated tool + within-ancestry anchor) still ships; v1.0 waits.

### Gate C: demand

Interview PRS-portability method developers and multi-ancestry benchmark
maintainers early. If a summary-statistics-based portability *measurement* does
not solve a current problem (e.g. they already have individual-level target
cohorts), release v0.1 and stop.

### Gate D: leakage model

Public target-ancestry test summary statistics are effectively public test labels;
a competitive leaderboard needs open development data and a hidden target-ancestry
test track with submission limits. v0.1 must not be marketed as competition-ready.

## Governance

Minimum core team: a scientific lead, a software maintainer, and a data steward.
Protocol changes require a public proposal, a new benchmark version, and rerunning
all baselines. Maintainer-authored methods get the same submission treatment as
external ones. Success six months after v1.0: at least five external submissions,
three independent citations/uses, and one external co-maintainer.

## First sprint

The first working days should produce:

1. A decision on upstream revival vs controlled fork, and the Witteveen consent
   request sent.
2. Three to five conversations with PRS-portability method developers (Gate C).
3. A **target-ancestry data-availability map** (candidate non-European and admixed
   GWAS + LD panels, access terms, individual-level validation cohorts).
4. Recovery of one legacy European dataset with a recorded checksum (the anchor).
5. The named within-ancestry golden result reproduced within a declared tolerance.
6. A short design for the per-ancestry bundle schema (LD, allele freqs,
   per-variant `n`, gauge) and the finite-sample-corrected estimator with an SE.
7. A written go/no-go for v0.1 and for pursuing the v1.0 cross-ancestry benchmark.

## References

Foundation:
- Witteveen, Pedersen, Meijsen, Andersen, Privé, Speed, Vilhjalmsson,
  *Publicly Available Privacy-preserving Benchmarks for Polygenic Prediction*,
  bioRxiv 2022, doi:10.1101/2022.10.10.510645 (CC-BY; not peer-reviewed):
  https://www.biorxiv.org/content/10.1101/2022.10.10.510645v1
- Historical successor repository: https://github.com/mennowitteveen/pgsbenchmark
- Legacy European data (verify availability): validation
  https://drive.google.com/file/d/1eZIiuz__tiqEtTyL14C_GwPEC6raSFSf ;
  test https://drive.google.com/file/d/12v67vJoAZqkvaStqiYzUrAezS4KTNdNP

Summary-statistic PRS evaluation (same estimator family):
- PUMAS: https://doi.org/10.1186/s13059-021-02479-9 ; PUMA-CUBS:
  https://doi.org/10.1186/s13059-024-03400-w
- Pattee & Pan, quasi-correlation: https://doi.org/10.1371/journal.pcbi.1008271
- Song et al., SummaryAUC: https://doi.org/10.1093/bioinformatics/btz176

Cross-ancestry portability (the new focus):
- Wang et al. 2020, deterministic portability limits, Nat Commun:
  https://www.nature.com/articles/s41467-020-17719-y
- Martin et al. 2019, portability disparities, Nat Genet (motivation).
- Ding et al. 2023, accuracy across the ancestry continuum, Nature.
- Kachuri et al. 2024, transferring PRS across populations (review):
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10961971/
- PRS-CSx (Ruan et al. 2022, Nat Genet); XPASS (Cai et al. 2021, AJHG);
  XPXP (Xiao et al. 2022, Bioinformatics) — cross-population construction.
- Popcorn (Brown et al. 2016, AJHG); S-LDXR (Shi et al. 2021, Nat Commun) —
  trans-ancestry genetic correlation (inputs to the predict branch).
