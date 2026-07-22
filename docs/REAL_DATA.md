# Real-data demonstration (within-ancestry anchor)

First run of ppb on real data: public PGS Catalog scores evaluated against real
GWAS summary statistics with the bigsnpr HapMap3+ European LD reference
(see the README "LD reference (real data)" section). This exercises the whole
stack — `read_ldref`, `harmonize_to`, `standardized_marginal`, per-variant `n`,
and the genome-wide estimator in Equation 1 accumulated across chromosomes. It
is a **within-ancestry (EUR→EUR) anchor**, the A = B special case; it is not a
cross-ancestry result.

**Equation 1. Genome-wide summary-statistic accuracy**

    R² = (wᵀz)² / (wᵀDw).

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
