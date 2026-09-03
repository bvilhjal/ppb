# Carrying empirical per-variant SDs: design note for the first non-European LD reference

Status: proposed design note — not yet a binding specification. When the
decisions in §6 are taken, what survives should be folded into
[`METHOD.md`](METHOD.md) §4 and [`../results/schema.md`](../results/schema.md),
and this note retired.

Prepared: 2026-08-23.

## 1. Problem

[`CROSS_ANCESTRY.md`](CROSS_ANCESTRY.md) records a **known deviation in the
current real-data path**: per-variant genotype SDs are recovered from allele
frequencies under Hardy-Weinberg, `sd = sqrt(2p(1−p))`. That formula is an
acceptable approximation on a homogeneous target and is **not** acceptable for
the flagship cross-ancestry application on a structured or admixed target;
the same document states that carrying empirical per-variant SDs in the
LD-reference schema is a **prerequisite for the first real cross-ancestry
run**. `../FINISHING_PLAN.md` already requires "empirical (in-sample)
per-ancestry standardization (not the HWE `2p(1−p)` formula, which breaks for
admixed targets)" in the v0.1 definition. This note designs that extension.

## 2. Where the SD enters — and where it does not

- **Enters, V3 gauge conversion (X2):** dosage→standardized weight conversion
  multiplies by the target-ancestry genotype SD
  (`w_std = w_dosage · sd`, since `w_dosage = w_std / sd`; [`METHOD.md`](METHOD.md)
  §V3 and `evaluate.py` implement the multiply — an earlier revision of this
  note said "divides" here, which was wrong).
- **Enters, (P1)–(P2):** the phenotype-free score mean and variance consume
  the panel's SDs.
- **Enters at panel-build time:** the D8 LD matrices are correlations of
  standardized genotypes, so whatever SD convention built the block must be
  recorded with it, or the block and the gauge conversion disagree.
- **Does not enter:** `z_B` itself — (M4) converts `beta/se/n` to the
  standardized marginal without any SD.

The empirical path is already validated:
[`../experiments/gauge_validation.py`](../experiments/gauge_validation.py)
(E4) shows that with empirical B SDs the full evaluation path recovers
individual-level truth (cross-seed mean +0.1%, SE 0.3%, worst seed 3.7%, the
residual being independent-panel noise); that discovery-side (A) SDs are
biased systematically upward (mean +42.5%, SE 11.4%; wide per-seed range
+5.4% to +230%); that on an admixed B (two subpopulations, F_ST 0.1)
single-draw estimates scatter to |27|% so a block jackknife must be quoted;
and that the paired HWE-vs-empirical residual there is +0.19% (SE 0.42%) —
consistent with zero at that structure (measured over seeds 0–19,
2026-09-03).

## 3. Schema proposal

Extend the LD-reference shards (`ldref_chr*.npz`) and their loaders:

- **`sd` (float32)** — the per-variant genotype SD, on the standardized-gauge
  convention used to build the block.
- **`sd_source` ∈ {"empirical", "hwe"}** — how it was obtained.
- **Panel metadata** — `panel_population`, `panel_n`, `panel_build`,
  `panel_source`, and a checksum, stored once per reference.
- **Format version bump.** Legacy shards without `sd` remain loadable as
  `sd_source="hwe"` on the within-ancestry EUR path (the current documented
  deviation, unchanged); cross-ancestry result packs must record an empirical
  source or be refused by the registry.
- **Registry:** packs carry the LD-panel identity and `sd_source` in the
  target block; `../results/inputs.tsv` gains rows for the new panel(s) with
  real checksums — closing, for the new panels, the provenance gap the review
  of 2026-08-16 recorded (F5) for the consortium inputs.

Tests to add: SD propagation through `read_ldref` → `evaluate`; version
refusal; legacy fallback labelling.

## 4. Panel options for the first B-side reference

**Table 1. Candidate non-European LD panels**

| panel | access | n per population | regime | notes |
|---|---|---:|---|---|
| 1000 Genomes (per-population) | public | ~50–500 | independent panel | the regime the simulation work already validates (unbiased, −0.095% in `experiments/README.md` Table 6); LD and SDs noisier; reproducible by anyone |
| HGDP | public | comparable | independent panel | more, and more structured, populations |
| gnomAD | public | large | — | no clean per-population genotype set; unsuitable for SD/LD construction |
| UK Biobank non-European groups | application | 10⁴+ | in-sample with Pan-UKB targets | the best reference if access exists; matches the earmarked targets in `REAL_DATA.md` Table 4 |
| PAGE / All of Us | controlled | large | varies | access-gated; useful for the v1.0 admixed track |

Recommendation: **1000 Genomes per-population panels first** — public,
checksummed, reproducible, and the independent-panel approximation is exactly
the regime the existing validation covers, with the block jackknife carrying
the panel noise; document the in-sample UKB upgrade as the path if
application access exists. A trait-specific second target (e.g. a BBJ
endpoint against a JPT panel) is a natural Phase-4b extension.

Blocks: ppb consumes caller-supplied blocks. Either inherit the 431-block EUR
map on the intersected variant set (keeps EUR and non-EUR evaluations on one
block grid; simpler jackknife comparability) or re-derive per population
(`optimal_ld_blocks`, Privé 2022, is the ldpred3-family precedent). Decision
§6.

## 4a. Proposed versus implemented (read this before quoting §3)

- **Implemented today:** `evaluate(..., weight_scale="dosage"/"frozen",
  genotype_sd=...)` multiplies dosage weights by caller-supplied per-variant
  SDs (`evaluate.py`, step V3); `genotype_sd_source` records which table
  carried them. The gauge validation E4 exercises this path end to end.
- **Proposed, not implemented:** the `sd` / `sd_source` shard fields, the
  `sd_source="hwe"` legacy fallback, the registry refusal of empirical-less
  cross-ancestry packs, and the converter/loader/CLI changes of §5–§6.
  Until those land, no LD-reference shard carries SDs and every empirical-SD
  run passes them explicitly through the current API.

## 5. Implementation steps

1. **Converter** `scripts/plink_to_ppb_ldref.py` (or extend
   `bigsnpr_ldref_to_ppb.py`): PLINK1/PGEN input → empirical SDs, D8 shards,
   panel metadata, block assignment (per §4 decision).
2. **Loader/schema changes** per §3; CLI passes `sd_source` through;
   `evaluate` already accepts per-variant SDs (E4 proves the path end to end).
3. **Validation runs**
   (a) *EUR sanity:* build a 1000G-EUR panel and compare X1 against the
   bigsnpr-EUR X1 across the 15 registry evaluations; declare the tolerance
   before running.
   (b) *Panel-size sensitivity (new, small):* sweep reference n in the
   cross-ancestry simulation and check that panel noise stays inside the
   jackknife SE — the direct answer to "a 1000G-scale panel is noisy LD"
   (`../PAPER_PLAN.md` Table 4).
   (c) E4 already pins the gauge behavior including the admixed arm.
4. **Phase 4 entry:** `scripts/cross_ancestry_eval.py` with real `(z_B, D_B)`;
   new result pack; leaderboard update.

## 6. Open decisions (maintainer)

1. Panel choice — 1000G-only, or 1000G now + UKB when access allows (data
   access is the deciding fact, not the code).
2. Blocks per ancestry — inherit the EUR map on the intersect, or re-derive.
3. `sd` dtype — float32 proposed; float16 saves little against one array.
4. Whether the legacy EUR path keeps the HWE fallback or is upgraded to
   1000G-EUR empirical SDs in the same pass (recommended: same pass, so the
   whole registry shares one gauge convention).
5. Admixed targets remain v1.0 scope (local-ancestry handling deferred) —
   this note does not change that.
