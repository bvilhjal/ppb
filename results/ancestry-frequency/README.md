# Ancestry-frequency benchmark snapshots

These provenance objects are separate from PPB's score-evaluation result packs
in `results/*.json`; the leaderboard does not read this subdirectory. Each
snapshot preserves the full decomposition result, immutable input hashes, panel
identity, software revisions, and the predeclared-control verdict.

**Table 1. Versioned snapshots.**

| Snapshot | Cohort | Controls | Verdict |
|---|---|---:|---|
| `yengo-height-2026-08-30.json` | Yengo 2022 ancestry-stratified height | 5 | 5/5 passed |

The five controls require `status=estimated` and the corresponding 1000
Genomes superpopulation to rank first. The pooled analysis is descriptive and
excluded. This evaluates deposited EAF-profile projection, not literal
participant-ancestry fractions.

**Table 2. Fixed real-data benchmark cohort** (Yengo et al., Nature 2022,
[doi:10.1038/s41586-022-05275-y](https://doi.org/10.1038/s41586-022-05275-y)).

| Key | GWAS Catalog accession | Reported sample | N | Predeclared top component |
|---|---|---|---:|---|
| AFR | `GCST90245989` | African ancestry | 168,193 | AFR |
| AMR | `GCST90245993` | Hispanic or Latin American | 58,709 | AMR (imperfect label proxy) |
| EAS | `GCST90245991` | East Asian ancestry | 363,856 | EAS |
| EUR | `GCST90245992` | European ancestry | 1,597,374 | EUR |
| SAS | `GCST90245994` | South Asian ancestry | 60,939 | SAS |
| POOLED | `GCST90245990` | five-ancestry pooled meta-analysis | 2,200,007 | descriptive only |

The acquisition path loads LDpred3's GWAS Catalog harvester from an exact
source revision, verifies each official compressed-file SHA-256, filters to
the panel variants, and pins the decompressed normalized content:

```bash
# Either fetch from the Catalog, or normalize already downloaded official files.
python scripts/ancestry_frequency_gwas_benchmark.py --fetch
python scripts/ancestry_frequency_gwas_benchmark.py --raw-dir <download-dir>
```

Leave-one-chromosome `contrast_rank` values in this snapshot can read `5`
against `expected_contrast_rank: 4` for a five-population panel. That was a
rounding artifact of the Gram diagnostic (the 1-vector null lifted above
tolerance). The rank half of the identifiability gate did not fire; the
condition half did the work. Current code counts rank in the \(K-1\)
contrast subspace. Do not re-read those recorded ranks as \(K\) independent
frequency axes.

Reproduce from the locally cached, hash-verified normalized inputs.
The snapshot's acquisition mode is `verified_cache`: raw source bytes were
verified at first download; each recorded run re-verifies the normalized-content
hash (`raw_source_sha256_verified_this_run: false`). Write to a **fresh dated
path** — the committed file is a dated archive of the run that produced it, and
overwriting it would replace that archive with a different one:

```bash
python scripts/ancestry_frequency_gwas_benchmark.py \
  --out results/ancestry-frequency/yengo-height-<today>.json
```

**Provenance (absolute, not relative to HEAD).** The committed snapshot was
produced by commit `67fa915` (2026-08-30). Compatibility last checked at
commit `cceb245` (2026-09-03): the estimator semantics have since changed, so
this archive is **historical** — do not read it as output of current code.

Semantic changes after `67fa915` that a fresh run would reflect:

- contrast rank is counted in the K−1 subspace (`contrast_rank: 4`, not 5);
- rejected fits publish `proportions: null` with the optimizer output kept as
  `proportions_raw` (this archive publishes weights on every status);
- the simplex face solver is centred/SVD with direct residual evaluation
  (the archived solver could elect a wrong face on near-exact mixtures);
- loaded panels are read-only; duplicate GWAS IDs are validated before they
  mark an identifier seen.

**Migration record.** To supersede this archive: restore the cached normalized
inputs (downloaded/normalized files live under gitignored `.work/`, not
committed; absent in this checkout) or re-acquire them, then write to a
**fresh dated path** — never overwrite this file:

```bash
python scripts/ancestry_frequency_gwas_benchmark.py \
  --out results/ancestry-frequency/yengo-height-<today>.json
```

A superseding snapshot must record the producing commit, input hashes,
`contrast_rank` in the K−1 subspace, and the `proportions: null` rejection
contract above; the 5/5 verdict rule is unchanged.
