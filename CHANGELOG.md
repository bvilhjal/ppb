# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions are
single-sourced from `ppb.__version__`; the package is pre-release
(`0.0.1.dev2`) and the API/schema may still change.

## [Unreleased]

### Added

- Refreshed ancestry benchmark snapshots under current code: new dated runs
  `results/ancestry-frequency/yengo-height-2026-09-03.json` and
  `results/ancestry-ld/yengo-height-2026-09-03.json` (both 5/5 controls
  passed; LD weights reproduce the 2026-08-30 archive exactly). Inputs were
  re-fetched from the GWAS Catalog with the pinned LDpred3 harvester supplied
  via a detached worktree; both channel READMEs now document that regeneration
  path and record the new snapshots.
- Ancestry report revised and rebuilt for the new snapshots (provenance
  table now records producing commit `0ed294a`, the re-fetch acquisition
  mode, the resolved regeneration path, and the default-thread environment
  of the snapshot runs; AMR's withheld SE, the SAS channel-agreement 0.500
  flag, and the K−1 contrast-rank contract are stated in prose and figures;
  `make_figures.py` re-points at the dated snapshots). Rebuilt with the same
  Tectonic 0.17.0 toolchain as the previous revision.

### Fixed

- Frequency simplex solver: each face is now a centred least-squares solve by
  SVD on the design (never the squared-condition Gram normal equations) with
  the residual norm evaluated directly. The old expanded-loss form could elect
  a materially wrong face on a near-exact mixture (e.g. `[0, 0.95, 0.05]` for
  truth `[0.9, 0.05, 0.05]`). Added an exact-mixture regression test.
- Rejected frequency fits publish no composition: every non-`estimated`
  status returns `proportions: null` and `proportions_se: null`; the optimizer
  output is retained as `proportions_raw`. This extends Estimator B's contract
  to the frequency channel.
- Fail-closed LD denominator: `r2()`, `mse()`, `evaluate()`, the sharded
  `evaluate_ldrefs` path, and `score_distribution` refuse a materially
  indefinite per-block `wᵀD_b w` (beyond `1e-9 · wᵀw` rounding tolerance),
  exactly like the multi-block diagnostics path already did. Low-level
  `quad()`/`block_quads()` still warn-and-report for certified int8
  references.
- Estimator A scale: one interpretation everywhere — under the calibrated
  working model the expected scale is `1 − h² ≤ 1`; rescaling `z` by `c`
  rescales the fitted scale by `c²`. Recorded scales of 1.26–27.46 are
  model-incompatibility diagnostics, not sample-size effects
  (the old order-`√N` story was wrong).
- Design note `docs/EMPIRICAL_SD.md` stated the dosage→standardized gauge
  conversion backwards ("divides"); the implementation multiplies
  (`w_std = w_dosage · sd`), as do `docs/METHOD.md` and `evaluate.py`. The
  note now separates implemented API behavior from the proposed shard schema.
- Reference correlation validation no longer materializes full auxiliary
  matrices (`R − Rᵀ`, identity mask, off-diagonal copy); tiled checks bound
  the peak temporary to ~32 MB.
- Loaded frequency panels are read-only, so post-load mutation cannot void
  the content-hash pin. Duplicate GWAS IDs are validated before marking an
  identifier seen, so an invalid first occurrence no longer poisons a valid
  later one.
- The sdist ships the checkout-only test drivers (`scripts/`, `experiments/`,
  `results/`, root `conftest.py`); shipped tests previously failed with
  22 collection errors. The wheel still installs only the `ppb` package.

### Added

- `examples/mini/`: tracked 8-variant bundle + weights with one runnable
  `ppb evaluate` command; `test_cli_mini_example` pins its numbers.
- Ancestry-report figures (`docs/ancestry_report/make_figures.py`):
  simulation bias/dispersion/decline rates, projection weights with boundary
  markers, and LD-channel diagnostics; plus a fixed-seed
  operating-characteristics table from `experiments/ancestry_ld_study.py`.
  The report is retitled, its abstract calibrated, a numbered
  artifact/provenance table added, the sign-flip calculation labelled
  descriptive, and the unverified "all citations verified" claim removed.
- `scripts/check_report_drift.py` (run in CI): fails while the committed
  report PDF trails any source.
- `CHANGELOG.md` (this file) and a stability map in `README.md` grouping
  public exports into Core/Stable, IO, Diagnostics, and Experimental.
- `require_psd_block_quads` in `ppb.ld_backend`: the shared per-block
  PSD gate used by every estimator that publishes a ratio.

### Changed

- Yengo snapshots in `results/ancestry-frequency/` and `results/ancestry-ld/`
  are labelled historical archives with migration records (inputs unavailable
  in this checkout); provenance is stated as absolute producing commit plus
  last-checked commit, never "N commits before HEAD".
- `README.md` novelty statement now reflects the completed sweep in
  `docs/NOVELTY.md`: the algebra is prior art; the claimed contribution is
  the measurement framing, explicit error theory, and fail-closed
  provenance infrastructure.
