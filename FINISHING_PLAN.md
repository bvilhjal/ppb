# PPB completion plan

Status: proposed

Prepared: 2026-07-17

Revised: 2026-07-17, after reviewing the source preprint and successor
repository (see Current state and References).

## Objective

Finish PPB as a maintained, versioned benchmark for evaluating linear polygenic
scores without distributing individual-level test records.

The original notebook is provenance, not the target architecture. The first
release will rescue the published result as a reproducible research artifact.
A larger blind benchmark will proceed only if its data, ownership, and user
demand pass explicit viability gates.

**Implementation approach (decided 2026-07-17): reimplement from theory.** The
evaluator is rebuilt from the method described in the preprint rather than ported
from the historical notebook or the successor package. Reimplementing from the
published equations upgrades the golden-result check from a repackaging into a
genuine independent replication, and yields a clean installable package instead
of inheriting the dated legacy stack. The legacy code and datasets are retained
strictly as a **reference oracle** for differential testing — used to confirm the
new implementation reproduces the published numbers, never copied. The estimator,
conventions, and tolerances are specified in `docs/METHOD.md`.

**Stack and LD handling (decided 2026-07-17).** The evaluator is a Python package
named `ppb`, using `numpy`/`scipy` for the core and **`numba`** for the
performance kernels. The estimator only needs `w^T z` and `w^T D w`, so `D` is
never materialised densely. For the LD matrix, PPB reuses the compact block
representation from the local **`ldpred3`** project — **D8** (packed int8 dense
blocks) and **LR8** (int8 low-rank `R ~= U U^T` factors for large blocks) — and
numba sweep kernels mirroring ldpred3's `_lr8_sweep_all` / `_d8_sweep_all`. The
low-rank factor is PSD by construction, so `w^T D w >= 0` holds without clamping.
The reproduction of published numbers still uses the paper's original cM-banding
as the oracle, with the LR8/D8 path validated to agree within a declared
tolerance (see `docs/METHOD.md`). A real Python toolchain with numpy/numba (the
one ldpred3 already uses) must be located and pinned before tests can run; the
`python` on PATH is only the Microsoft Store stub.

## Current state (2026-07-17)

Starting coordinates, so the timeline below is read against reality rather than
a clean slate:

- **This repository (`bvilhjal/ppb`)** contains only `PPB.ipynb` (a ~1.4 MB
  notebook that is almost entirely base64-embedded output images), a two-line
  `README.md`, an MIT `LICENSE`, and this plan. It holds no packaged, reusable
  source. The real method and datasets live in the successor repository.
- **Successor repository (`mennowitteveen/pgsbenchmark`)** holds the working
  code as notebooks plus a `pgsbenchmark` package, with `PPB-demonstration.ipynb`
  as the entry point. Stack is dated: it targets "Python 3.2+", Unix only.
- **The paper was never formally published.** It exists as a 2022 bioRxiv
  preprint (doi:10.1101/2022.10.10.510645, CC-BY, `published_doi: NA`).
  Authors: Witteveen, Pedersen, Meijsen, Andersen, Privé, Speed, Vilhjalmsson.
  Witteveen is first/corresponding; **Vilhjalmsson is senior (last) author**, so
  this continuation has a standing co-author, not only an external steward.
  Consequently, "update the paper" at v1.0 can mean *finally publishing it*.
- **Benchmark data as published:** a ~10,000-individual external-sumstats and
  validation dataset and a ~352,000-individual test dataset, both HDF5,
  containing LD information and matched GWAS summary statistics, for 8 outcomes
  (somatic and psychiatric). The 352K test scale is UK Biobank-derived European.
- **Hosting is fragile.** The datasets are distributed as two bare Google Drive
  file IDs plus `pgsbenchmark.org`. A quick check on 2026-07-17 returned 404 for
  both Drive links via direct request (this may reflect Drive's consent gating
  rather than deletion) and an ambiguous response for the domain. Durable,
  checksummed, DOI-backed re-hosting is therefore an early requirement, and link
  recovery must be verified interactively before it is assumed intact.
- **Reproduction may be partly underway.** Recent commits here (`figure S1`,
  `Comparison using reference LD`) suggest work toward reproducing a published
  figure already exists; Phase 1 should be estimated from that baseline, not zero.

Because the existing test dataset (LD plus matched summary statistics) is
*already public*, the current benchmark already exposes test-side summary
statistics and cannot be un-published. Any leakage-resistant competitive
benchmark is therefore a genuine redesign, not a repackaging of the existing
release (see Gate D).

## Stewardship and provenance

This is explicitly an attempt to finish the successor project led by Menno
Witteveen after he left science. It is a stewardship and continuation effort,
not a claim that the original work is being replaced or newly invented. The
continuation is led by an existing senior author of the original preprint, which
grants standing to maintain and re-release the work but does not remove the
obligation to secure the lead author's consent and agree on credit before any
new publication.

The project will therefore:

- preserve the original repositories, commit history, licence, copyright, and
  citations, keeping the code licence (MIT) and the paper/data licence (CC-BY)
  distinct and correctly attached to each artifact;
- credit Witteveen and the original collaborators for the method, benchmark,
  software, and datasets they created;
- distinguish historical work from subsequent maintenance, validation, and
  extensions in documentation and publications;
- request repository access, data backups, and any available handover context
  without making progress depend on Witteveen returning to active research;
- obtain the lead author's written consent to continue the work and a recorded
  agreement on authorship and credit for any subsequent publication, before that
  publication is prepared; and
- record the new maintainers and the point at which they assume responsibility
  for releases, corrections, governance, and support.

## Product definition

### v0.1: reproducibility release

The project is complete at v0.1 when:

- an actively maintained canonical repository has been selected;
- at least one legacy benchmark dataset has durable hosting, a checksum,
  provenance, and documented redistribution rights;
- one named published result — a specific figure or table value from the
  preprint, chosen and written down before Phase 1 (the recent `figure S1` work
  is the natural candidate) — can be reproduced from a clean environment within a
  declared numerical tolerance;
- the essential evaluator is an installable package with a command-line
  interface;
- a synthetic fixture, equation-level tests, golden-result tests, and CI pass;
- supported inputs, preprocessing assumptions, limitations, and privacy claims
  are documented; and
- the historical notebook and results are preserved but are not part of the
  production implementation.

### v1.0: benchmark product

The larger project is complete at v1.0 when:

- an open development track and a leakage-resistant hidden-test track exist;
- datasets, protocols, baseline results, and leaderboards are immutable and
  versioned;
- the benchmark covers quantitative and binary traits, two independent target
  cohorts, and at least one non-European ancestry track;
- uncertainty and known failure modes have been validated against
  individual-level evaluation;
- at least three independent researchers reproduce baseline results; and
- at least two external method teams complete a submission without maintainer
  intervention.

## Scope

### Included in v0.1

- Continuous-trait PPB evaluation on the original European/HapMap3-style scope.
- Exact target-cross-product evaluation and approximate external-LD evaluation.
- A numba-backed evaluator computing `w^T z` and `w^T D w` over a compact block
  LD store (ldpred3 D8/LR8 int8 representation), never a dense `M`-by-`M` matrix.
- Variant, allele, build, order, duplicate, and missingness validation.
- A small public fixture and at least one recovered real benchmark trait.
- A machine-readable input and result schema.
- A generated static baseline table.

### Deferred until after v0.1

- Binary-trait metrics and calibration.
- Multi-ancestry and admixed-cohort tracks.
- A hosted submission service.
- Nonlinear prediction models, survival outcomes, and clinical-utility claims.
- Porting unfinished legacy prediction models and notebook utilities.

## Delivery plan

**Table 1. Delivery plan**

| Phase | Work | Exit criterion | Estimate |
|---|---|---|---:|
| 0. Viability and ownership | Confirm maintainers, choose upstream revival or a controlled fork, locate data backups, audit data rights, and interview likely users | Named owners, writable canonical repository, data-rights inventory, and a recorded go/no-go decision | 1 week |
| 1. Historical recovery (as oracle) | Recover the legacy datasets and generate reference outputs from the historical code to serve as a test oracle; reproduce one published result | One checksummed dataset and one agreed golden result, reproduced by the new implementation | 1-2 weeks |
| 2. v0.1 core | Reimplement the evaluator from theory (`docs/METHOD.md`), define schemas, add packaging, CLI, tests, CI, documentation, and durable data hosting | A clean machine completes the synthetic and real-data examples; `v0.1.0` is tagged | 2-3 weeks |
| 3. Scientific validation (parallel research track) | Test LD source and size, ancestry mismatch, sample overlap, data defects, uncertainty, and binary outcomes; compare with established summary-statistic evaluators | Predefined accuracy and interval-coverage thresholds pass; operating boundaries are documented | 6-8 weeks |
| 4. Blind beta | Publish development data, keep final test cross-products hidden, validate submissions, rate-limit evaluation, and generate a versioned leaderboard | Two outside teams submit successfully and 3-5 baselines reproduce | 4-6 weeks |
| 5. v1.0 release | Add independent cohorts and ancestry tracks, complete governance and data review, publish a DOI release, and update the paper | External acceptance criteria pass and public correction/versioning policies exist | 3-4 weeks |

Expected effort is 4-6 focused weeks for v0.1 and approximately 24-32
person-weeks, or 4-6 calendar months with parallel scientific and engineering
work, for v1.0.

Phase 3 is a research programme in its own right — potentially the content of a
publication — not a gate on shipping v0.1. It runs in parallel with, and after,
the v0.1 release rather than blocking it. v0.1 ships once Phase 2 exits;
Phase 3's results deepen the operating-boundary documentation and inform whether
v1.0 is worth attempting.

The Phase 0-1 estimates assume the third party leading the original work is
reachable and responsive. That is not guaranteed. The critical path is therefore
decoupled from that dependency: the synthetic-fixture, notebook-extraction, and
code-audit work begins immediately and blocks on nothing external, while the
ownership and data-recovery requests run alongside on their own timebox (Gate A).

## Scientific validation

The validation plan will be frozen before the main experiments. It must cover:

1. Numerical equality with individual-level Pearson R2 when exact target
   cross-products and identical preprocessing are used.
2. Approximation error from LD sample size, LD windowing, ancestry mismatch,
   target versus independent versus training LD, and admixed targets.
3. Sparse through highly polygenic architectures, multiple heritabilities,
   long-range LD, and large-effect loci.
4. Sample overlap, relatedness, per-variant sample-size variation, allele/build
   errors, imputation quality, and population structure.
5. Bias, RMSE, interval coverage, top-method selection probability, and
   top-model regret, not only rank correlation.
6. Comparison with individual-level truth, quasi-correlation, PUMAS, and
   SummaryAUC where applicable.

The initial baselines should include fixed versions of P+T or SCT, lassosum2,
LDpred2, PRS-CS or PRS-CSx, SBayesR or SBayesC, and an ensemble.

## Benchmark protocol

- Publish an open development bundle, but never expose final test phenotypes or
  sufficient statistics used for a competitive leaderboard.
- Accept PGS Catalog-compatible weights or a reproducible container.
- Freeze the genome build, variant set, allele conventions, train/development/
  test separation, tuning budget, submission quota, metrics, uncertainty
  method, exclusions, and ranking procedure for each benchmark version.
- Store large artifacts in immutable DOI-backed storage with an object-store
  mirror. Store only manifests and tiny fixtures in Git.
- Record data version, code commit, parameters, environment or container digest,
  runtime, and resources for every baseline and submission.
- Generate the first leaderboard from reviewed result files. Do not build a
  custom web service before the protocol survives external beta testing.

## Claims and terminology

The main claim should be narrow:

> PPB is a versioned, leakage-resistant benchmark for externally evaluating
> linear polygenic scores without releasing individual-level test records.

"Privacy-preserving" must not imply differential privacy, cryptographic
protection, or immunity to membership inference. Until a formal privacy review
supports stronger wording, use "summary-statistics-based" or
"individual-level-data-free" in technical claims.

PPB evaluates predictive association. It does not establish calibration,
clinical utility, causality, or individual privacy.

The underlying work is a 2022 preprint that was never peer-reviewed or
journal-published. Claims should be stated at preprint strength until a
peer-reviewed version exists; finally publishing the method is itself a
legitimate v1.0 outcome rather than a given.

## Decision gates

### Gate A: ownership

Ask Witteveen for access or transfer, data backups, handover context, consent to
continue, and an authorship agreement. The plan must not require his return to
science. Set an explicit timebox: if there is no substantive response within
three weeks, proceed with a controlled fork under an active owner while
preserving history, authorship, and attribution, and continue the consent and
credit request in parallel without blocking engineering work.

### Gate B: data

The benchmark data is already public and CC-BY, so the binding question is
durability and provenance, not permission. Recover both HDF5 datasets from the
Google Drive links and any mirror, verify them against a recorded checksum,
confirm the UK Biobank-derived summary-level nature is documented and compliant
with how they were originally released, and re-host them in durable,
DOI-backed storage with an object-store mirror. Treat the bare Google Drive links
and `pgsbenchmark.org` as single points of failure that may already have decayed
— verify recovery interactively before assuming it. If a dataset genuinely
cannot be recovered from any source, do not silently substitute another; either
redesign around openly licensable data as a new major version or finish only the
synthetic/reference artifact.

### Gate C: demand

Interview likely method developers and adjacent benchmark maintainers early —
during the first sprint, not after Phase 3 — because it is cheap and a "no
demand" answer should stop work before the multi-week validation and beta
phases, not after them. If hidden summary-statistic evaluation does not solve a
current problem, release v0.1 and stop.

### Gate D: leakage model

Public test summary statistics are effectively public test labels. A serious
competitive benchmark therefore requires open development data and hidden test
evaluation with submission limits. Note that the existing v0.1-era test dataset
is already public and cannot be withdrawn, so the leakage-resistant benchmark is
a new design built on freshly held-out data — not the current release with a
leaderboard bolted on. v0.1 must not be marketed as competition-ready.

## Governance

The minimum core team consists of a scientific lead, a software maintainer, and
a data steward. Protocol changes require a public proposal, a new benchmark
version, and rerunning all baselines. Maintainer-authored methods receive the
same submission treatment as external methods.

Success six months after v1.0 means at least five external submissions, three
independent citations or uses, and one external co-maintainer. Page views are
not a scientific endpoint.

## First sprint

The first five working days should produce:

1. A decision on upstream revival versus controlled fork.
2. A data inventory listing every expected artifact, known backup, owner,
   licence, provenance, and current availability, with the two Google Drive
   links and `pgsbenchmark.org` verified as live or dead.
3. An extraction and audit of `PPB.ipynb` and the successor package into plain
   source, so the actual historical logic is inventoried rather than trapped in
   notebook output; strip embedded outputs (for example with `nbstripout`) and
   decide whether the notebook stays in the main repository or moves to an
   archive path or provenance tag.
4. A pinned legacy environment or container capable of running the historical
   code.
5. One recovered dataset with a recorded checksum.
6. The named target result written down, then one reproduced golden result and
   its numerical tolerance.
7. Three to five short conversations with likely users to test demand (Gate C).
8. A written go/no-go decision for v0.1 and the larger v1.0 product.

The first milestone is one recoverable dataset and one reproduced result. Code
cleanup, branding, and a website come later.

## References

- Witteveen, Pedersen, Meijsen, Andersen, Privé, Speed, Vilhjalmsson,
  *Publicly Available Privacy-preserving Benchmarks for Polygenic Prediction*,
  bioRxiv 2022, doi:10.1101/2022.10.10.510645 (CC-BY; not peer-reviewed;
  `published_doi: NA` as of 2026-07-17):
  https://www.biorxiv.org/content/10.1101/2022.10.10.510645v1 — also mirrored at
  https://openreview.net/forum?id=ZRZDTY3L8Ne
- Historical successor repository:
  https://github.com/mennowitteveen/pgsbenchmark
- Benchmark platform and data (verify availability): http://www.pgsbenchmark.org;
  validation dataset https://drive.google.com/file/d/1eZIiuz__tiqEtTyL14C_GwPEC6raSFSf;
  test dataset https://drive.google.com/file/d/12v67vJoAZqkvaStqiYzUrAezS4KTNdNP
- Pattee and Pan, quasi-correlation:
  https://doi.org/10.1371/journal.pcbi.1008271
- Song et al., SummaryAUC: https://doi.org/10.1093/bioinformatics/btz176
- PUMAS: https://doi.org/10.1186/s13059-021-02479-9
- PUMAS-ensemble: https://doi.org/10.1186/s13059-024-03400-w
