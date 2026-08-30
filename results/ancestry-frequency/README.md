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

Reproduce from the locally cached, hash-verified normalized inputs:

```bash
python scripts/ancestry_frequency_gwas_benchmark.py \
  --out results/ancestry-frequency/yengo-height-2026-08-30.json
```

Downloaded and normalized summary-statistic files live under ignored `.work/`;
they are not committed.
