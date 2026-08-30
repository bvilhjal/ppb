# Ancestry LD-moment benchmark snapshots

This directory records real-GWAS stress tests of PPB's experimental LD-moment
channels. The estimand is the best-fitting 1000G LD-covariance mixture, not a
literal fraction of participants or meta-analysis weight.

**Table 1. Versioned snapshots.**

| Snapshot | Cohort | Positive controls | Qualitative rule |
|---|---|---:|---|
| `yengo-height-2026-08-30.json` | Yengo 2022 ancestry-stratified height | 5 | 5/5 declared references ranked first |

**Table 2. Estimator A weights and repeated sign-flip diagnostics.**

| Study | AFR | AMR | EAS | EUR | SAS | Declared rank | Null fits | p, scaled coefficient | p, normalized weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AFR | .702 | .170 | .000 | .113 | .014 | 1 | 117/200 | .0050 | .1692 |
| AMR | .000 | 1.000 | .000 | .000 | .000 | 1 | 112/200 | .0050 | .0945 |
| EAS | .000 | .000 | .961 | .021 | .017 | 1 | 132/200 | .0050 | .1045 |
| EUR | .000 | .045 | .000 | .955 | .000 | 1 | 130/200 | .0050 | .1443 |
| SAS | .000 | .004 | .000 | .110 | .887 | 1 | 119/200 | .0050 | .0647 |
| Pooled | .000 | .102 | .000 | .898 | .000 | descriptive | 120/200 | - | - |

The scaled statistic is

**(R1) Correct-reference coefficient contrast**

```text
s * (pi_expected - max_{k != expected} pi_k).
```

It distinguishes a large correctly directed covariance component from a
near-zero-scale null fit whose normalized weights happen to land on a simplex
vertex. All five scaled contrasts exceeded all 200 sign flips, giving the
plus-one empirical value `1/201 = 0.004975`. The normalized contrasts did not
clear `p <= 0.05`; normalized mixture magnitudes are therefore not validated.

**Table 3. Model and uncertainty diagnostics.**

| Study | A scale | Signal absorber | Declared-component jackknife SE | Boundary | B signal z | B result |
|---|---:|---|---:|---|---:|---|
| AFR | 2.031 | no | .050 | yes | 1.598 | declined |
| AMR | 1.877 | no | .000 | yes | 2.069 | declined |
| EAS | 3.186 | no | .117 | yes | .921 | declined |
| EUR | 25.286 | no | .109 | yes | 2.164 | declined |
| SAS | 1.257 | yes | .231 | yes | 1.475 | declined |
| Pooled | 27.460 | no | - | yes | 2.178 | declined |

AMR's zero symmetric SE is a boundary artifact, not certainty. All six fits
touch a simplex boundary, so ordinary symmetric intervals are invalid. The
SAS quadratic absorber was retained by an explicitly uncalibrated heuristic.
Estimator B declined for insufficient reproducible LD-dependent chi-square
signal; sparse marker pruning also omits genome-wide tagging paths, so this is
not a fair genome-scale verdict on B.

**Table 4. Fixed compact reference design.**

| Quantity | Value |
|---|---:|
| Full-panel common HM3 variants | 714,078 |
| Retained distant blocks | 177 |
| Retained variants / pairs | 21,892 / 44,250 |
| Minimum edge-to-edge block gap | 5 Mb |
| Maximum variants / pairs per block | 128 / 250 |
| AFR/AMR/EAS/EUR/SAS LD samples | 652 / 347 / 504 / 503 / 484 |
| Design file SHA-256 | `1ce09d4c...b93114e` |
| Design semantic SHA-256 | `7c332f45...679297c` |

The common rule is MAF at least 5% in every super-population of the full 1000G
AF panel; LD is estimated in the published 2,490-sample mostly-unrelated
subset. Pair selection and LD estimation reuse those same finite samples, so
reference winner's-curse and errors-in-variables uncertainty are absent from
the chromosome jackknife.

Reproduce from the tracked design and hash-verified cached summary statistics:

```bash
python scripts/ancestry_ld_gwas_benchmark.py \
  --out results/ancestry-ld/yengo-height-2026-08-30.json
```

Use `--fetch` to acquire the six pinned GWAS Catalog inputs through LDpred3's
pinned harvester. The snapshot records all 1,200 null replicates, input hashes,
selection settings, software revision, and the full estimator diagnostics.
