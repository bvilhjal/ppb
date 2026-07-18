# Archive — historical provenance

This directory preserves the original research artifacts. They are **provenance,
not the production implementation**: the maintained evaluator is reimplemented
from the method's theory in `src/ppb/` (see `docs/METHOD.md`). Nothing here is
imported by the package, and it is not covered by the tests or CI.

## `PPB.ipynb`

The original exploratory notebook (≈1.4 MB, mostly embedded output images). Kept
for historical reference and to trace how the published figures were produced.

It is used only as a **reference oracle** — a source of expected numbers to check
the reimplementation against — never as code to port. See `FINISHING_PLAN.md`
(the reimplement-from-theory decision) for the rationale.

The working code and datasets of the successor project live in the external
repository `mennowitteveen/pgsbenchmark`.
