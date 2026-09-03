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
hash (`raw_source_sha256_verified_this_run: false`). Write to a **fresh dated
path** — the committed file is a dated archive of the run that produced it, and
overwriting it would replace that archive with a different one:

```bash
python scripts/ancestry_frequency_gwas_benchmark.py \
  --out results/ancestry-frequency/yengo-height-<today>.json
```

**Provenance caveat.** The committed snapshot records `ppb_commit: 67fa915`,
three commits before HEAD: `e12d78e` and `036e6a3` changed
`src/ppb/ancestry_frequency.py` and the verdict aggregation. No scientific
number changes, but HEAD's `_contrast_rank_condition_from_gram` truncates to
K−1 singular values, so a fresh run records `contrast_rank: 4` where the
archive records 5 (the same caveat as the LD channel).

Downloaded and normalized summary-statistic files live under ignored `.work/`;
they are not committed.
