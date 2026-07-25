# Real-data demonstration (within-ancestry anchor)

Status: record. Numbers here are reproducible from committed code at the stated
commit. Symbols and labels: [`NOTATION.md`](NOTATION.md). Results are labelled
(G1)–(G5).

First run of ppb on real data: public PGS Catalog scores evaluated against real
GWAS summary statistics with the bigsnpr HapMap3+ European LD reference
(see the README "LD reference (real data)" section). This exercises the whole
stack — `read_ldref`, `harmonize_to`, `standardized_marginal`, per-variant `n`,
and the genome-wide accumulation (G1) across chromosomes. It
is a **within-ancestry (EUR→EUR) anchor**, the A = B special case; it is not a
cross-ancestry result.

**(G1) Genome-wide accumulation.** (M1) evaluated over the whole genome by
summing the per-block products of (M4):

    R² = (Σ_b u_b)² / (Σ_b v_b).

## Setup

- **Scores.** Privé et al. 2022, *Portability of 245 polygenic scores…* (AJHG,
  [doi:10.1016/j.ajhg.2021.11.008](https://doi.org/10.1016/j.ajhg.2021.11.008))
  — the PGS Catalog `portability-ldpred2_*` series, LDpred2 weights trained on
  UK Biobank with HM3 variants. 100% of score variants are inside the HM3+
  reference (no truncation).
- **LD reference.** bigsnpr HM3+ EUR (UK Biobank), converted per
  `scripts/bigsnpr_ldref_to_ppb.py` to per-chromosome block-diagonal int8 npz.
- **Target GWAS, two regimes:**
  - *Non-overlapping (honest):* consortium GWAS that do **not** include UK
    Biobank — GIANT 2014 height, GIANT 2015 BMI, GLGC 2013 LDL, DIAGRAM 2017
    T2D, CARDIoGRAMplusC4D 2015 CAD (GCST003116), BCAC 2017 breast cancer
    (GCST004988). All EUR, GRCh37.
  - *Overlapping (in-sample):* Pan-UK Biobank EUR GWAS — the same cohort the
    scores were trained on.
- `z` from beta/se with per-variant or effective `n`
  (`ppb.standardized_marginal`); `w` put on the standardized-genotype scale
  with `sd_j = sqrt(2 f_j (1−f_j))`, `f = af_UKBB` from the reference. The
  overall scale of `w` cancels in the ratio, so raw-unit (e.g. cm) weights are
  fine for R² (not for MSE).

## Results

### Non-overlapping consortium targets

**Table 1. Non-overlapping within-ancestry estimates**

| trait | score (PGS Catalog) | target GWAS | N used | score support | R² statistic |
|---|---|---|---:|---:|---:|
| height | PGS002146 | GIANT 2014 (Wood) | median 252k (50.0k–253k) | 91.8% | 0.252 |
| LDL | PGS002150 | GLGC 2013 (Teslovich) | median 89.9k (50.0k–173k) | 88.6% | 0.110 |
| BMI | PGS002161 | GIANT 2015 (Locke) | median 234k (50.0k–322k) | 92.1% | 0.065 |
| T2D | PGS002026 | DIAGRAM 2017 (Scott) | 88.8k | 99.9% | 0.044 |
| breast cancer | PGS002015 | BCAC 2017 (Michailidou) | 255k | 100.0% | 0.042 |
| CAD | PGS002048 | CARDIoGRAMplusC4D 2015 (Nikpay) | 163k | 99.9% | 0.025 |

`N used` is the sample-size input actually supplied to
`ppb.standardized_marginal`, as recorded in the results registry. The GIANT and
GLGC files carry per-variant `N`, so the table reports its median and range; no
single `n_eff` exists for those rows. Case/control studies use a trait-level
effective size (for example, T2D 88,810 and CAD 163,123). Published headline or
total sample sizes are not substituted for the values fed to the estimator.
`Score support` is the fraction of non-zero score weights retained on the joint
weight/target-summary-statistic support. The height, BMI, and LDL estimates
therefore describe explicitly restricted scores, not the complete catalog
scores; treating missing target associations as zero while retaining their
weights in the denominator produced the former lower values.
For the quantitative rows, the statistic targets squared correlation under the
documented standardization. The T2D, breast-cancer, and CAD rows are
case/control approximations, not liability-scale R².

### Same scores, overlapping Pan-UKB targets (in-sample failure mode)

**Table 2. Uncorrected in-sample statistics and independent anchors**

| trait | R² statistic (in-sample) | R² statistic (honest) |
|---|---:|---:|
| height | 0.803 | 0.252 |
| T2D | 0.509 | 0.044 |
| BMI | 0.405 | 0.065 |
| asthma | 0.311 | — |
| SBP | 0.249 | — |
| CAD (I25) | 0.231 | 0.025 |
| breast cancer | 0.176 | 0.042 |
| LDL | 0.148 | 0.110 |
| MDD | 0.021 | — |

The contrast exposes the expected failure mode: using the training cohort as the
target can produce a severely inflated statistic, whereas an independent target
measures out-of-sample association. The correction has not been removed, but it
now requires a trainer-specific sensitivity basis, an independent reference,
block sampling-noise variances, exact support, and a stable identifiable fit. The
PGS Catalog LDpred2 files contain final weights but no reconstructible trainer
operator, so their basis is `basis_unavailable`: every in-sample value in Table 2
is an upper bound, and none has a defensible corrected R². See
[`OVERLAP.md`](OVERLAP.md). All runs matched 100% of weight variants to the LD
reference and ≥99.9% of target-file variants, with positive `wᵀz` and strictly
positive `wᵀDw`. Joint score support was 88.6–100.0%; the lower-support
consortium rows are restricted-score estimates as Table 1 states. The
independent quantitative magnitudes are consistent with the literature (height
about 25%, LDL about 11%, BMI about 6%); binary-trait comparisons retain the
approximation described below.

## External validation against published individual-level accuracy

The claim above that these magnitudes are "consistent with the literature" is
the weakest form of validation available, and it is unnecessary. The same PGS
Catalog scores carry **published individual-level accuracy** from their source
publication (Privé et al. 2022), retrievable from the PGS Catalog REST API, so
the hand-wave can be replaced by ratios:

```bash
python scripts/anchor_validation.py          # needs network, no local data
```

**Table 3. PPB against published individual-level accuracy** (partial R² in the
held-out UK Biobank "UK (+ Ireland)" group; PPB against its consortium target).

| trait | target GWAS | PPB R² | published R² | published/PPB | support |
|---|---|---:|---:|---:|---:|
| LDL | GLGC 2013 | 0.110 | 0.112 | **1.02** | 88.6% |
| height | GIANT 2014 | 0.252 | 0.376 | **1.49** | 91.8% |
| BMI | GIANT 2015 | 0.065 | 0.134 | **2.08** | 92.1% |
| T2D | DIAGRAM 2017 | 0.044 | 0.017 | 0.38 | 99.9% |
| breast cancer | BCAC 2017 | 0.042 | 0.013 | 0.30 | 100.0% |
| CAD | CARDIoGRAM 2015 | 0.025 | 0.012 | 0.46 | 99.9% |

These are two different EUR cohorts and two slightly different estimands (a
covariate-adjusted partial correlation within UKBB, versus PPB against a
consortium meta-analysis), so this is a magnitude check rather than an equality
test. Read it that way, and it says three things.

**LDL agrees to 2%.** On the trait with the *lowest* score support in the table,
the summary-statistic estimate lands on the individual-level one. That is the
strongest single piece of evidence the pipeline has: harmonization, the gauge,
per-variant `n`, the LD reference, and (G1) all have to be right together to
produce it.

**Both GIANT targets are low by 1.5–2×, and nothing else is.** Support does not
explain it — LDL has the least and agrees best. The cohort difference does not
explain it — it would not single out one consortium. What does fit is the
mechanism `LIMITATIONS.md` already names: R² scales as the **square** of the
scale of `z`, and genomic control deflates `z`. The implied deflation is
`√1.49 = 1.22` for height and `√2.08 = 1.44` for BMI, which is the right size for
GIANT's study-and-meta-level genomic control. This is now a measured,
target-specific effect rather than a flagged possibility — and it is a reason to
treat GIANT-derived rows as **lower bounds** until a scale diagnostic exists.

**The binary rows are not comparable**, as the caveats below state: PPB's binary
output is a standardized summary-statistic approximation whose scale depends on
the supplied effective sample size, not an observed-scale partial correlation.
They run 2–3× *higher* by a roughly constant factor, which is what a scale
convention difference looks like rather than an error, and they are excluded from
the verdict.

### A ready-made target for the flagship

The same query returns each score's accuracy in **all nine** UKBB ancestry
groups — the individual-level portability curve, for these exact scores, that
the cross-ancestry method aims to recover from summary statistics alone:

**Table 4. Published individual-level portability, relative to UK (+ Ireland).**

| group | height | BMI | LDL | CAD | T2D | BrCa |
|---|---:|---:|---:|---:|---:|---:|
| UK (+ Ireland) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Poland | 0.93 | 1.02 | 1.08 | 1.77 | 0.44 | 0.51 |
| Italy | 0.88 | 0.96 | 0.85 | 0.97 | 0.76 | 0.93 |
| Iran | 0.65 | 0.64 | 0.79 | 0.45 | 1.83 | 0.74 |
| India | 0.60 | 0.71 | 0.43 | 1.34 | 1.47 | 0.50 |
| China | 0.49 | 0.54 | 0.58 | 0.23 | 0.36 | 1.02 |
| Caribbean | 0.27 | 0.33 | 0.77 | 0.11 | 0.59 | 0.18 |
| Nigeria | 0.17 | 0.18 | 0.45 | 0.01 | 0.38 | 0.55 |

The quantitative columns show the portability decay the project exists to
measure — height and BMI down to ~0.18 in the West African group. Given a
non-European target GWAS for one of these traits, (X1) can be evaluated against
this column rather than against a simulation. That is the Phase-4 validation
target, and it already exists.

## Uncertainty and negative controls

Two things a point estimate cannot tell you, both computed from the per-block
`u_b = wᵀ_b z_b` and `v_b = wᵀ_b D_b w_b` that the genome-wide sweep already
builds — no second pass over the LD reference (`ppb.diagnostics`):

- **(G2) `metrics.jackknife`** — a delete-one-block standard error (plus a
  delete-one-chromosome variant, and the per-chromosome partial sums so a reader
  can recompute it from the pack). Without it "0.252 vs 0.110" is a comparison
  with no stated precision, which is not enough for a benchmark whose purpose is
  ranking. It also captures genomic heterogeneity, which the `1/N` finite-sample
  term does not, and `max_variance_share` flags an estimate carried by one region.
- **(G3) `metrics.sign_flip_null`** — an exact negative control. Because `D` is
  block-diagonal, negating every weight in a block flips `u_b` and leaves `v_b`
  untouched, so the sign-flipped scores are a null family with the same
  denominator and the same per-block magnitudes. `null_mean = Σu_b²/Σv_b` is the
  R² this score would report from block noise alone: **read a small value like
  CAD's 0.025 against that, not against zero.** `z = Σu_b/√(Σu_b²)` measures how
  coherently the blocks agree, bounded by `√431 = 20.8` on this reference.

**(G4)** A third control needs its own sweep:

```bash
python scripts/negative_controls.py --out controls.json
```

It evaluates every score against **every** consortium target. The diagonal
should dominate. Off-diagonal cells are *not* all expected to be zero — BMI/T2D
and LDL/CAD are genuinely genetically correlated, and seeing that is itself
evidence the pipeline responds to real signal — so the expected-correlated pairs
are declared in the script rather than rationalised afterwards. What must not
appear is strong signal between traits with no shared aetiology (height × LDL);
that is an artifact, and the script exits non-zero when it finds one.

None of the three detects a uniformly mis-scaled `z`, which moves the estimate
and its null together. See the caveat below and [`LIMITATIONS.md`](LIMITATIONS.md).

## Caveats

- **Binary traits:** the reported number is a standardized summary-statistic
  approximation, **not liability-scale R²**. DIAGRAM T2D used n_eff = 88.8k
  (balanced case/control effective); Privé's pipeline uses 72.1k, which rescales
  the statistic by a constant ~1.2×. Pan-UKB binary GWAS are SAIGE logistic, so
  the z→correlation mapping is model- and prevalence-dependent.
- **Honest ≠ unbiased:** consortium targets are UKBB-free, but scores trained
  on close relatives of these consortia would still be optimistic. The
  portability-ldpred2 scores are UKBB-trained, so the consortium numbers are
  clean of direct sample overlap.
- **These numbers inherit the source studies' summary-statistic scale.**
  `R²` is proportional to the *square* of the scale of `z`, and `z` is derived
  entirely from the reported `beta`, `se` and `n`. Genomic control — applied at
  study and meta level by GIANT and GLGC — inflates `se` and therefore deflates
  these estimates by roughly the inflation factor; the `n_eff` choice does the
  same (the T2D note above is a 1.2× example). No step in this pipeline detects
  or corrects that, so read every row as conditional on how its source study
  processed its standard errors. See [`LIMITATIONS.md`](LIMITATIONS.md).
- **Adjusted `z`, unadjusted `D`.** The target GWAS report covariate-adjusted
  (partial) correlations; the bigsnpr LD reference is an unadjusted genotype
  correlation matrix. `METHOD.md` §4 specifies adjusting both. The mismatch is
  small within a homogeneous EUR sample and is not small in a structured target.
- **Meta-analysis targets:** GIANT/GLGC `z_j` come from different cohort subsets
  per variant (N 50.0k–253k), so there is no single population whose squared
  correlation is being estimated. The registry records the N distribution rather
  than a headline number for exactly this reason.
- **Incomplete target support:** a missing target association is not evidence of
  zero association. PPB now uses the exact joint support in both numerator and
  denominator and records `metrics.n_variants_scored`; where support is below
  100%, the result applies to the restricted score rather than the full catalog
  score.
- The EUR LD reference is from UK Biobank; the consortium GWAS are not — any
  residual LD mismatch is small within EUR and goes the other way from the
  cross-ancestry case (see `docs/CROSS_ANCESTRY.md`).

## Reproduce

```bash
scripts/panukb_download.sh                                   # Pan-UKB flat files (~21 GB)
python scripts/panukb_filter_hm3plus.py                      # -> data/panukb/*_hm3plus.tsv
python scripts/consortium_prep.py                            # -> data/consortium/*_hm3plus.tsv
python scripts/regenerate_results.py --out results/<pack>.json   # all 15 evaluations
```

`regenerate_results.py` is the reproducible path and the one that produces the
results registry: it sweeps each score against **both** its targets in a single
pass over the LD reference and writes the JSON records at full precision (~5.5
min per trait, ~35 min for all nine). It records the current final-weight overlap
basis as unavailable and does not manufacture a corrected statistic from variant
count. Future scores with a documented linear operator or stable rerunnable
trainer can use the retained basis-aware correction. The older
`scripts/eval_consortium.py` / `scripts/eval_panukb.py` are human-readable
diagnostics, not the publication path; they do not replace the registry's strict
joint-support and provenance checks. Do not transcribe rounded numbers from them
— at `wᵀDw ≈ 8e-4` four decimals leaves one significant figure and `r2` can no
longer be recomputed from the recorded `num`/`den`.

Scripts resolve `data/` relative to the repository root. PGS Catalog weights
(`data/pgs_weights/`) are downloaded directly from the PGS Catalog FTP
(`PGS{id}_hmPOS_GRCh37.txt.gz`).
