# PPB completion plan

Status: proposed

Prepared: 2026-07-17

## Objective

Finish PPB as a maintained, versioned benchmark for evaluating linear polygenic
scores without distributing individual-level test records.

The original notebook is provenance, not the target architecture. The first
release will rescue the published result as a reproducible research artifact.
A larger blind benchmark will proceed only if its data, ownership, and user
demand pass explicit viability gates.

## Stewardship and provenance

This is explicitly an attempt to finish the successor project led by Menno
Witteveen after he left science. It is a stewardship and continuation effort,
not a claim that the original work is being replaced or newly invented.

The project will therefore:

- preserve the original repositories, commit history, licence, copyright, and
  citations;
- credit Witteveen and the original collaborators for the method, benchmark,
  software, and datasets they created;
- distinguish historical work from subsequent maintenance, validation, and
  extensions in documentation and publications;
- request repository access, data backups, and any available handover context
  without making progress depend on Witteveen returning to active research;
  and
- record the new maintainers and the point at which they assume responsibility
  for releases, corrections, governance, and support.

## Product definition

### v0.1: reproducibility release

The project is complete at v0.1 when:

- an actively maintained canonical repository has been selected;
- at least one legacy benchmark dataset has durable hosting, a checksum,
  provenance, and documented redistribution rights;
- one published result can be reproduced from a clean environment within a
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
| 1. Historical recovery | Recover the legacy datasets, recreate the historical environment, and reproduce one published result | One checksummed dataset and one exact command produce an agreed golden result | 1-2 weeks |
| 2. v0.1 core | Extract the evaluator, define schemas, add packaging, CLI, tests, CI, documentation, and durable data hosting | A clean machine completes the synthetic and real-data examples; `v0.1.0` is tagged | 2-3 weeks |
| 3. Scientific validation | Test LD source and size, ancestry mismatch, sample overlap, data defects, uncertainty, and binary outcomes; compare with established summary-statistic evaluators | Predefined accuracy and interval-coverage thresholds pass; operating boundaries are documented | 6-8 weeks |
| 4. Blind beta | Publish development data, keep final test cross-products hidden, validate submissions, rate-limit evaluation, and generate a versioned leaderboard | Two outside teams submit successfully and 3-5 baselines reproduce | 4-6 weeks |
| 5. v1.0 release | Add independent cohorts and ancestry tracks, complete governance and data review, publish a DOI release, and update the paper | External acceptance criteria pass and public correction/versioning policies exist | 3-4 weeks |

Expected effort is 4-6 focused weeks for v0.1 and approximately 24-32
person-weeks, or 4-6 calendar months with parallel scientific and engineering
work, for v1.0.

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

## Decision gates

### Gate A: ownership

Ask Witteveen for access or transfer, data backups, and any handover context he
is willing to provide. The plan must not require his return to science. If a
transfer cannot be arranged promptly, create a controlled fork under an active
owner while preserving history, authorship, and attribution.

### Gate B: data

If the original datasets cannot be recovered or legally redistributed, do not
silently replace them. Either redesign the benchmark around openly licensable
data as a new major version or finish only the synthetic/reference artifact.

### Gate C: demand

Before committing to the hosted benchmark, interview likely method developers
and adjacent benchmark maintainers. If hidden summary-statistic evaluation does
not solve a current problem, release v0.1 and stop.

### Gate D: leakage model

Public test summary statistics are effectively public test labels. A serious
competitive benchmark therefore requires open development data and hidden test
evaluation with submission limits.

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
   licence, provenance, and current availability.
3. A pinned legacy environment or container capable of running the historical
   code.
4. One recovered dataset with a recorded checksum.
5. One reproduced golden result and its numerical tolerance.
6. A written go/no-go decision for v0.1 and the larger v1.0 product.

The first milestone is one recoverable dataset and one reproduced result. Code
cleanup, branding, and a website come later.

## References

- Witteveen et al., *Publicly Available Privacy-preserving Benchmarks for
  Polygenic Prediction*: https://openreview.net/forum?id=ZRZDTY3L8Ne
- Historical successor repository:
  https://github.com/mennowitteveen/pgsbenchmark
- Pattee and Pan, quasi-correlation:
  https://doi.org/10.1371/journal.pcbi.1008271
- Song et al., SummaryAUC: https://doi.org/10.1093/bioinformatics/btz176
- PUMAS: https://doi.org/10.1186/s13059-021-02479-9
- PUMAS-ensemble: https://doi.org/10.1186/s13059-024-03400-w
