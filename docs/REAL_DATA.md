# Real-data demonstration (within-ancestry anchor)

First run of ppb on real data: public PGS Catalog scores evaluated against real
GWAS summary statistics with the bigsnpr HapMap3+ European LD reference
(see the README "LD reference (real data)" section). This exercises the whole
stack — `read_ldref`, `harmonize_to`, `standardized_marginal`, per-variant `n`,
and the genome-wide estimator `R² = (wᵀz)² / (wᵀDw)` accumulated across
chromosomes. It is a **within-ancestry (EUR→EUR) anchor**, the A = B special
case; it is not a cross-ancestry result.

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

### Honest R² (non-overlapping consortium targets)

| trait | score (PGS Catalog) | target GWAS | n_eff | R² |
|---|---|---|---:|---:|
| height | PGS002146 | GIANT 2014 (Wood) | 253k | 0.211 |
| LDL | PGS002150 | GLGC 2013 (Teslovich) | 189k | 0.100 |
| BMI | PGS002161 | GIANT 2015 (Locke) | 339k | 0.056 |
| T2D | PGS002026 | DIAGRAM 2017 (Scott) | 159k | 0.044 |
| breast cancer | PGS002015 | BCAC 2017 (Michailidou) | 255k | 0.042 |
| CAD | PGS002048 | CARDIoGRAMplusC4D 2015 (Nikpay) | 184k | 0.025 |

### Same scores, overlapping Pan-UKB targets (in-sample failure mode)

| trait | R² (in-sample) | R² (honest) |
|---|---:|---:|
| height | 0.803 | 0.211 |
| T2D | 0.509 | 0.044 |
| BMI | 0.405 | 0.056 |
| asthma | 0.311 | — |
| SBP | 0.249 | — |
| CAD (I25) | 0.231 | 0.025 |
| breast cancer | 0.176 | 0.042 |
| LDL | 0.148 | 0.100 |
| MDD | 0.021 | — |

The overlap contrast is the expected behaviour of the estimator: with the
training cohort as target it recovers the in-sample fit (up to 15× inflation),
with an independent target it measures the score's real accuracy. This failure
mode is detectable and approximately correctable from the score weights alone —
see [`OVERLAP.md`](OVERLAP.md) (`ppb.overlap`; polygenic scores detected at
z = 5-12 here; sparse scores are flagged as upper bounds). All runs had
100% weight variants and ≥99.9% z variants matched, positive `wᵀz`, and
strictly positive `wᵀDw`. The magnitudes are consistent with the literature
(height ~20-25%, LDL ~10%, BMI ~5-8%, CAD/T2D ~3-5%).

## Caveats

- **Binary traits:** R² is on the standardized-latent scale (approximation,
  not liability R²). DIAGRAM T2D used n_eff = 88.8k (balanced case/control
  effective); Privé's pipeline uses 72.1k, which rescales R² by a constant
  ~1.2×. Pan-UKB binary GWAS are SAIGE logistic; the z→correlation mapping is
  approximate.
- **Honest ≠ unbiased:** consortium targets are UKBB-free, but scores trained
  on close relatives of these consortia would still be optimistic. The
  portability-ldpred2 scores are UKBB-trained, so the consortium numbers are
  clean of direct sample overlap.
- The EUR LD reference is from UK Biobank; the consortium GWAS are not — any
  residual LD mismatch is small within EUR and goes the other way from the
  cross-ancestry case (see `docs/CROSS_ANCESTRY.md`).

## Reproduce

```bash
scripts/panukb_download.sh                                   # Pan-UKB flat files (~21 GB)
python scripts/panukb_filter_hm3plus.py                      # -> data/panukb/*_hm3plus.tsv
python scripts/consortium_prep.py                            # -> data/consortium/*_hm3plus.tsv
python scripts/eval_consortium.py                            # honest R² table
python scripts/eval_panukb.py                                # in-sample (overlap) table
```

Scripts resolve `data/` relative to the repository root. PGS Catalog weights
(`data/pgs_weights/`) are downloaded directly from the PGS Catalog FTP
(`PGS{id}_hmPOS_GRCh37.txt.gz`).
