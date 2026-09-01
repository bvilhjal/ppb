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
hash (`raw_source_sha256_verified_this_run: false`).

```bash
python scripts/ancestry_frequency_gwas_benchmark.py \
  --out results/ancestry-frequency/yengo-height-2026-08-30.json
```

Downloaded and normalized summary-statistic files live under ignored `.work/`;
they are not committed.
