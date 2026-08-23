# Prior-art sweep: measuring cross-population portability from target-ancestry summary statistics

Status: novelty audit, closing finding F2 of [`../REVIEW-2026-08-16.md`](../REVIEW-2026-08-16.md).
Sweep executed 2026-08-16 by web search with primary-source verification (journal
pages, PubMed Central full texts, Europe PMC records). Every citation marked
"verified" below was seen in a fetched primary or indexing source during this
sweep; queries are listed in the appendix so the sweep is reproducible. Scope
limits: English-language literature reachable by web search; Google Scholar was
not directly accessible; preprint coverage is partial.

## The question under test

The claim audited is narrow. PPB's estimator evaluates a fixed PGS `w` in a
target ancestry B from summary-level inputs alone, `R²_B = (wᵀz_B)²/(wᵀD_B w)`,
and presents the resulting number as a **measurement of cross-population
portability** — not as a tuning criterion, not as a step in constructing `w`.
The sweep therefore separates four things: (a) the identity, (b) within-ancestry
summary-statistic benchmarking, (c) cross-population tuning criteria of this
family, (d) cross-population measurement as the contribution.

## Table 1. What the located works do with target-ancestry summary statistics

| work | year | what it does with target-ancestry summary stats | role | verified citation |
|---|---|---|---|---|
| lassosum (Mak et al.) | 2017 | none across ancestries: external GWAS statistics + reference LD enter a pseudovalidation criterion used to pick the lasso penalty | tunes | Mak, Porsch, Choi, Zhou & Sham, *Genet Epidemiol* 41(6):469–480, [doi:10.1002/gepi.22050](https://doi.org/10.1002/gepi.22050) — verified (Europe PMC) |
| quasi-correlation (Pattee & Pan) | 2020 | names `(wᵀz)²/(wᵀDw)`-type accuracy-from-summaries as an evaluation metric; demonstration is within a single ancestry | evaluates (metric proposed, within-ancestry demo) | Pattee & Pan, *PLOS Comput Biol* 16(10):e1008271, [doi:10.1371/journal.pcbi.1008271](https://doi.org/10.1371/journal.pcbi.1008271) — verified (Europe PMC) |
| SummaryAUC (Song et al.) | 2019 | approximates the AUC of a fixed score in a validation dataset from summary-level statistics (binary traits); schizophrenia demonstration | evaluates (within-ancestry, AUC) | Song, Liu, Shi, *Bioinformatics* 35(20):4038–4044, [doi:10.1093/bioinformatics/btz176](https://doi.org/10.1093/bioinformatics/btz176) — verified (Europe PMC) |
| PUMAS (Zhao et al.) | 2021 | subsamples a single GWAS into pseudo-training/validation splits to tune and benchmark scores; one ancestry per analysis | tunes (benchmarks within-ancestry) | Zhao et al., *Genome Biol* 22(1):257, [doi:10.1186/s13059-021-02479-9](https://doi.org/10.1186/s13059-021-02479-9) — verified (Europe PMC) |
| benchmarking framework (Pain et al.) | 2021 | compares tuning strategies for eight PRS methods, pseudovalidation among them, against individual-level truth; European UKB/TEDS only | evaluates methods (within-ancestry) | Pain et al., *PLOS Genet* 17(5):e1009021, [doi:10.1371/journal.pgen.1009021](https://doi.org/10.1371/journal.pgen.1009021) — verified (journal full text) |
| Witteveen et al. | 2022 | none across ancestries: releases the within-ancestry EUR public benchmark (LD reference + target summary statistics) this repository stewards | evaluates (infrastructure, within-ancestry) | Witteveen et al., bioRxiv, [doi:10.1101/2022.10.10.510645](https://doi.org/10.1101/2022.10.10.510645) — verified (Europe PMC) |
| Martin et al. | 2019 | none: individual-level measurement of PGS portability across ancestries (the ground truth PPB substitutes for) | evaluates (individual-level) | Martin et al., *Nat Genet* 51(4):584–591, [doi:10.1038/s41588-019-0379-x](https://doi.org/10.1038/s41588-019-0379-x) — verified (PubMed) |
| Wang et al. | 2020 | none: deterministic model *predicting* transferred R² from discovery-side quantities plus target LD/frequencies (the "predict" branch) | predicts | Wang, Guo, Ni, Yang, Visscher, Yengo, *Nat Commun* 11:3865, [doi:10.1038/s41467-020-17719-y](https://doi.org/10.1038/s41467-020-17719-y) — verified (Europe PMC) |
| XPASS (Cai et al.) | 2021 | integrates multi-population GWAS statistics through a cross-population genetic-correlation model to output weights | constructs | Cai et al., *Am J Hum Genet* 108(4):632–655, [doi:10.1016/j.ajhg.2021.03.002](https://doi.org/10.1016/j.ajhg.2021.03.002) — verified (Europe PMC) |
| PRS-CSx (Ruan et al.) | 2022 | couples effect-size estimation across populations via a shared signal prior on multi-population GWAS statistics + per-population LD references | constructs | Ruan et al., *Nat Genet* 54(5):573–580, [doi:10.1038/s41588-022-01054-7](https://doi.org/10.1038/s41588-022-01054-7) — verified (Europe PMC) |
| TL-PRS (Zhao et al.) | 2022 | fine-tunes a source-trained score to the target ancestry by gradient descent on target-ancestry GWAS statistics with reference-panel LD; individual-level data still used for validation and testing, pseudovalidation named as a workaround | constructs | Zhao, Fritsche, Smith, Mukherjee, Lee, *Am J Hum Genet* 109(11):1998–2008, [doi:10.1016/j.ajhg.2022.09.010](https://doi.org/10.1016/j.ajhg.2022.09.010) — verified (PMC full text; added 2026-08-23) |
| X-Wing (Miao et al.) | 2023 | quantifies local cross-population genetic correlation and combines population-specific scores from summary statistics | constructs | Miao et al., *Nat Commun* 14:832, [doi:10.1038/s41467-023-36544-7](https://doi.org/10.1038/s41467-023-36544-7) — verified (Europe PMC) |
| CT-SLEB (Zhang et al.) | 2023 | empirical-Bayes ancestry-specific effects from large EUR + smaller non-EUR GWAS statistics after clumping and thresholding | constructs | Zhang et al., *Nat Genet* 55(10):1757–1768, [doi:10.1038/s41588-023-01501-z](https://doi.org/10.1038/s41588-023-01501-z) — verified (Europe PMC) |
| Ding et al. | 2023 | none: individual-level portability measurement across the genetic ancestry continuum (All of Us) | evaluates (individual-level) | Ding et al., *Nature* 618(7966):774–781, [doi:10.1038/s41586-023-06079-4](https://doi.org/10.1038/s41586-023-06079-4) — verified (Europe PMC) |
| Wang et al. (global biobanks) | 2023 | none: individual-level multi-biobank comparison of strategies for developing scores across diverse cohorts, ancestry-specific effect sizes among them | evaluates (individual-level) | Ying Wang, Namba, Lopera, Kerminen, Tsuo, Läll, et al., *Cell Genomics* 3(1):100241, [doi:10.1016/j.xgen.2022.100241](https://doi.org/10.1016/j.xgen.2022.100241) — verified (PubMed metadata; added 2026-08-23) |
| Wang et al. (PRIMED) | 2023 | none: individual-level evaluation of how ancestry, genetic architecture, and methodology govern cross-population predictive performance | evaluates (individual-level) | Ying Wang, Kanai, Tan, …, Martin, *Cell Genomics* 3(10):100408, [doi:10.1016/j.xgen.2023.100408](https://doi.org/10.1016/j.xgen.2023.100408) — verified (PMC full text; added 2026-08-23) |
| SBayesRC (Zheng et al.) | 2024 | annotation-informed priors and wider genome coverage improve prediction *between* ancestries; training LD is matched to the GWAS, one population at a time | constructs | Zheng et al., *Nat Genet* 56(5):767–777, [doi:10.1038/s41588-024-01704-y](https://doi.org/10.1038/s41588-024-01704-y) — verified (Europe PMC) |
| MUSSEL (Jin et al.) | 2024 | ancestry-group-specific effect estimates borrowing information across multi-ancestry GWAS statistics and LD references | constructs | Jin et al., *Cell Genomics* 4(4):100539, [doi:10.1016/j.xgen.2024.100539](https://doi.org/10.1016/j.xgen.2024.100539) — verified (Europe PMC) |
| PUMAS framework, extended (Zhao et al.) | 2024 | summary-statistic optimization, benchmarking and ensembling of existing scores on one GWAS; analyses "constrained … to the European ancestral population"; cross-ancestry benchmarking named as future work requiring ancestry-specific GWAS and LD references | tunes + benchmarks (within-ancestry) | Zhao et al., *Genome Biol* 25(1):260, [doi:10.1186/s13059-024-03400-w](https://doi.org/10.1186/s13059-024-03400-w) — verified (journal full text) |
| PRStuning (Jiang et al.) | 2024 | none from the target: tunes hyperparameters from *training*-population summary statistics, avoiding target data entirely | tunes (target-free) | Jiang, Chen, Girgenti, Zhao, *Nat Commun* 15:24, [doi:10.1038/s41467-023-44009-0](https://doi.org/10.1038/s41467-023-44009-0) — verified (Europe PMC) |
| PTL-PRS (Cho & Lee) | 2025 | computes pseudo-R² of transfer-learned scores from **target-ancestry GWAS statistics and target LD**; uses it for early stopping and hyperparameter tuning, and — where no individual-level target data exist (COVID-19 analyses in EAS/AFR/SAS) — reports pseudo-R² by ancestry as the final accuracy | tunes primarily; **measures as a fallback** | Cho & Lee, *Bioinformatics* 41(10):btaf540, [doi:10.1093/bioinformatics/btaf540](https://doi.org/10.1093/bioinformatics/btaf540) — verified (PMC full text) |
| Momin et al. | 2026 | none: individual-level UKB evaluation of seven cross-ancestry construction methods; no quasi-correlation quantity computed | evaluates (individual-level) | Momin et al., *Genet Epidemiol* 50(1):e70029, [doi:10.1002/gepi.70029](https://doi.org/10.1002/gepi.70029) — verified (PMC full text) |
| review (Kachuri et al., PRIMED) | 2024 | none: taxonomy of *construction* methods for diverse populations; pseudovalidation discussed only as a way to avoid individual-level tuning sets; evaluation metrics (AUC, calibration) framed as individual-level | review | Kachuri et al., *Nat Rev Genet* 25(1):8–25, [doi:10.1038/s41576-023-00637-2](https://doi.org/10.1038/s41576-023-00637-2) — verified (PMC full text) |
| MIXPRS (Xu et al.) | 2026 | subsamples the **target-population GWAS** ("data fission" / pseudo-GWAS) into tuning and testing halves; estimates non-negative combination weights from summary-form score–outcome and score–score moments with target LD; prunes SNPs to blunt LD mismatch; accuracy itself reported on individual-level UKB/All of Us | tunes/combines | Xu et al., *Nat Genet* 58(7):1583–1594, [doi:10.1038/s41588-026-02637-x](https://doi.org/10.1038/s41588-026-02637-x) — verified (Europe PMC; journal page) |

## Verdict

**(a) The identity is published and conceded.** Mak et al. (2017) tune with its
square root; Pattee & Pan (2020) name it as an accuracy metric; PUMAS (Zhao et
al. 2021) reaches the same estimand by subsampling. Nothing in this sweep
changes that, and no claim on the algebra is available.

**(b) Within-ancestry summary-statistic benchmarking is occupied.** Pain et al.
(2021) benchmark eight methods with pseudovalidation among the tuning
strategies; Zhao et al. (2024) optimize, benchmark and ensemble existing scores
from a single (European) GWAS; Song et al. (2019) do the summary-statistic
evaluation for AUC. Witteveen et al. (2022) — the foundation this repository
continues — supply the shared within-ancestry artifact. PPB's within-ancestry
use is an application of known technique.

**(c) Cross-population tuning criteria of exactly this family are now
published.** This is the sweep's firmest negative result for any strong novelty
claim. PTL-PRS (Cho & Lee 2025) evaluates transfer-learned scores with a
pseudo-R² computed from target-ancestry GWAS statistics and target-population
LD, and uses it for early stopping and hyperparameter selection in non-European
targets. MIXPRS (Xu et al. 2026) subsamples the target-population GWAS and
estimates combination weights from summary-form moments that are functionals of
the same `wᵀz` and `wᵀDw` quantities. A sentence like "no one has used
target-ancestry summary statistics in the quasi-correlation identity" would be
false as of 2026.

**(d) Cross-population *measurement* as the contribution: not found — but it
must be hedged against one published near miss.** No located work makes
"measure the portability of an existing PGS in ancestry B from `(z_B, D_B)`"
its stated contribution, and three independent signals say the slot is open:
the PUMAS group's own benchmarking paper names cross-ancestry benchmarking as
future work; the PRIMED review's taxonomy contains no summary-statistics
measurement category; and the 2026 methods comparison (Momin et al.) evaluates
transferability exclusively on individual-level data. However, PTL-PRS's
COVID-19 analyses *report* target-ancestry pseudo-R² as final ancestry-specific
accuracy where individual-level data do not exist — the computation, in a
different-ancestry target, published. The honest formulation is therefore:
PPB is not the first to compute the number; it is the first located work to
make that number the *object of study* — with the failure-mode map (gauge
self-consistency, LD-reference mismatch, A-substitution as Theorem 1's
impossibility), the finite-sample correction, and fail-closed infrastructure —
rather than a criterion consumed by a constructor or a fallback reported
without error analysis (Cho & Lee validate pseudo-R² against true R² on eight
UKB traits, note slight inflation, and proceed; MIXPRS's accuracies are
individual-level).

**Where the claim survives.** The measurement framing, the error theory, and
the discipline around inputs (empirical-SD gauge rule, exact joint support,
overlap correction, provenance-checked registry) do not appear in any located
work. "New to this project" can be asserted for that bundle.

**Where it must be hedged.** (1) Cite Cho & Lee (2025) and Xu et al. (2026) as
nearest prior art and state the delta precisely: measurement-as-end vs
criterion-as-means; error analysis vs none; arbitrary published scores (PGS
Catalog) vs a method's own outputs. (2) Do not claim priority over "using
target-ancestry summary statistics with the identity" in any unqualified form.
(3) The claim is scholarly, not mathematical — an identity cannot be owned,
and the framing claim is falsifiable by one uncited paper, so this sweep should
be re-run (or extended through Google Scholar and citation-chasing from PTL-PRS
and MIXPRS) before submission. (4) Closed 2026-08-23. Both leads left open by
the 2026-08-16 sweep were resolved by the follow-up recorded in the appendix:
the "Wang et al. 2023 (*Cell Genomics*, ancestry-specific effect sizes)" lead
is two individual-level applied papers (the global-biobank strategies paper,
3(1):100241, and the PRIMED cross-population evaluation, 3(10):100408), both
now in Table 1 and neither measuring portability from summary statistics; and
no Pattee-group cross-ancestry paper exists in any indexed source reached —
the only Pattee & Pan method remains the 2020 quasi-correlation paper, and the
Wiley/ResearchGate snippet was almost certainly a journal article *citing* it
(the DOI resolving to a conference-abstracts issue). The citation-chase from
PTL-PRS added its direct predecessor TL-PRS (Zhao et al. 2022). Verdict (d) is
unchanged by all three additions.

## Appendix: search terms used

Web queries, grouped as run on 2026-08-16 (several were retried after
rate-limit errors; retries used the variants shown):

1. `PRS-CSx Ruan 2022 Nature Genetics improving polygenic prediction ancestrally diverse populations summary statistics`
2. `XPASS cross-population polygenic risk score method GWAS summary statistics`; variant: `"XPASS" polygenic risk score cross-ancestry method paper`
3. `CT-SLEB polygenic score cross-ancestry continuous transcriptome? sparse? method`; variants: `CT-SLEB polygenic score method cross-ancestry paper`, `CT-SLEB polygenic risk score method`
4. `MUSSEL multi-population polygenic score method summary statistics`; variant: `MUSSEL polygenic score trans-ancestry`
5. `MIXPRS multi-population multi-method polygenic score Nature Genetics 2026 summary statistics tuning`; variants: `MIXPRS polygenic score`, `MIXPRS multi-population polygenic score Nature Genetics`
6. `SBayesRC Zeng functional annotations polygenic score GCTB multi-population ancestry`
7. `Martin 2019 clinical use of current polygenic risk scores may exacerbate health disparities Nature Genetics`
8. `Wang 2020 polygenic score accuracy transferability theory model portability across ancestries predicts R2`; variants: `Wang 2020 polygenic risk score transferability accuracy prediction model across populations allele frequencies LD`, `Wang 2020 polygenic score transferability theory model`, `Wang 2020 polygenic score portability across ancestries`, `polygenic risk score transferability across populations Wang 2020`
9. `Ding 2023 polygenic scoring accuracy varies across genetic ancestry continuum Nature`
10. `pseudovalidation cross-population polygenic risk score target ancestry summary statistics evaluate accuracy`; variants: `pseudovalidation PRS target population summary statistics method`, `pseudovalidation cross-population polygenic risk score summary statistics`, `PRS pseudovalidation without individual-level data evaluate accuracy`
11. `"quasi-correlation" polygenic score ancestry cross-population evaluation`
12. `polygenic risk score transferability diverse populations review portability`
13. `"integrating cross-ancestry and ancestry-specific effects" polygenic risk score Genetic Epidemiology`
14. `PUMAS fine-tuning cross-ancestry trans-ancestry GWAS summary statistics subsampling diverse populations`
15. `SummaryAUC summary statistics AUC polygenic score without individual-level data method`
16. `Pattee J polygenic` (Europe PMC author query); `AUTH:"Pattee Jack" AND TITLE:"polygenic"`

Primary-source fetches used for verification (not search): nature.com
(MIXPRS), pmc.ncbi.nlm.nih.gov (PTL-PRS PMC12529095; Momin PMC12820924; Kachuri
PMC10961971; Zhao 2024 PMC11462675 full text), journals.plos.org (Pain et al.),
onlinelibrary.wiley.com (gepi.22256 — blocked), academic.oup.com (SummaryAUC —
blocked), biorxiv.org (data-fission preprint — timed out), and Europe PMC REST
queries by DOI/title for: lassosum, Pattee & Pan, PUMAS, Zhao 2024, Wang 2020,
Ding 2023, XPASS, CT-SLEB, PRS-CSx, MUSSEL, SBayesRC, X-Wing, PRStuning,
MIXPRS, PTL-PRS, Witteveen et al.

Addendum — follow-up sweep 2026-08-23 (closing the two open leads and the
PTL-PRS citation chase; run by web search with primary-source fetches):

17. `Wang 2023 "Cell Genomics" ancestry-specific effect sizes polygenic score`
18. `Pattee Pan "cross-ancestry" "ancestry-specific" effects polygenic score abstract Genetic Epidemiology Wiley`; variants restricted to `onlinelibrary.wiley.com`
19. `polygenic score accuracy evaluation "summary statistics" target ancestry without individual-level data method 2024 2025 PTL-PRS citing`
20. `"Polygenic prediction across populations is influenced by ancestry, genetic architecture, and methodology" Cell Genomics authors year`

Primary-source fetches: pmc.ncbi.nlm.nih.gov PMC9674947 (TL-PRS — title, DOI,
and target-summary-statistic usage confirmed from the full text) and PMC10589629
(Wang et al. 2023 3(10):100408 — DOI confirmed from the article header); the
3(1):100241 record was confirmed from PubMed metadata (PMID 36777179) and the
publisher page. The Pattee-group queries (18) surfaced only articles *citing*
Pattee & Pan (2020); no cross-ancestry method paper from that group was found.
No change to the verdict; the additions strengthen the "individual-level
evaluation" cluster rather than the measurement slot.
