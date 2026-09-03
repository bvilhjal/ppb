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

**(A3) Correct-reference coefficient contrast**

```text
s * (pi_expected - max_{k != expected} pi_k).
```

It distinguishes a large correctly directed covariance component from a
near-zero-scale null fit whose normalized weights happen to land on a simplex
vertex. No scaled contrast was exceeded by any sign-flip replicate, giving the
plus-one empirical value `1/201 = 0.004975`. The normalized contrasts did not
clear `p <= 0.05`; normalized mixture magnitudes are therefore not validated.

Read that `1/201` with its convention in mind. Only 112-132 of the 200 flips
per study produced a fitted contrast at all (the "Null fits" column above); the
rest declined, and a declined replicate cannot exceed the observation, so it
counts toward the denominator but never the numerator. That is the permissive
reading and it favours significance. The stricter reading treats a decline as
unknown rather than as a non-exceedance, which for AFR would put the
denominator at 118 rather than 201 and the p-value at `1/118 = 0.008`. Both
readings clear `0.05` here, so the qualitative verdict is unchanged, but the
exact value is convention-dependent.

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

The fitted linear scales are not interpretable as \(1-h^2\). Under Estimator
A's working model the linear scale is \(s_0=1-h^2\le 1\); every recorded
scale here is larger than 1, and EUR / Pooled (25.3 / 27.5) ran without the
quadratic absorber. Those numbers are a diagnostic that the real pair
moments are not quantitatively model-consistent — tagging-term leakage into
the linear channel, uncalibrated \(z\)'s in a large height meta-analysis
(Assumption 3), or both. Ranking-first remains a compatibility statement,
not a calibrated \(1-h^2\) or participant-fraction estimate.

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

Reproduce from the tracked design and hash-verified cached summary statistics.
The snapshot's acquisition mode is `verified_cache`: raw source bytes were
verified at first download; each recorded run re-verifies the normalized-content
hash (`raw_source_sha256_verified_this_run: false`). Write to a **fresh dated
path** — the committed file is a dated archive of the run that produced it, and
overwriting it would replace that archive with a different one:

```bash
python scripts/ancestry_ld_gwas_benchmark.py \
  --out results/ancestry-ld/yengo-height-<today>.json
```

**Provenance (absolute, not relative to HEAD).** The committed snapshot was
produced by commit `5dca2f8` (2026-08-30). Compatibility last checked at
commit `cceb245` (2026-09-03): the estimator semantics have since changed, so
this archive is **historical** — do not read it as output of current code.

Semantic changes after `5dca2f8` that a fresh run would reflect:

- the fitted linear `scale` above one is a model-incompatibility diagnostic
  (`s₀ = 1−h² ≤ 1` under the calibrated working model; rescaling `z` by `c`
  rescales the scale by `c²`), not a sample-size effect;
- reference correlation validation is tiled (no multi-gigabyte temporaries);
- `r2()`/`mse()`/`evaluate()` refuse a materially indefinite LD block even
  when the summed denominator stays positive (previously only the multi-block
  diagnostics path refused).

**Migration record.** Regeneration is blocked in this checkout: the cached
normalized inputs live in gitignored `.work/` (absent) and `--fetch` fails
its own LDpred3 pin (the sibling checkout has drifted). To supersede this
archive, restore either path and write to a **fresh dated path** — never
overwrite this file:

```bash
python scripts/ancestry_ld_gwas_benchmark.py \
  --out results/ancestry-ld/yengo-height-<today>.json
```

A superseding snapshot must record the producing commit, input hashes, the
tracked design hash, the corrected scale interpretation, and the per-block
PSD gate above; the qualitative rules (declared reference ranks first, the
permissive/strict sign-flip readings) are unchanged. The snapshot records all
1,200 null replicates, input hashes, selection settings, software revision,
and the full estimator diagnostics.
