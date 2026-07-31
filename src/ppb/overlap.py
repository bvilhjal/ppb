"""Training/target sample overlap: the fail-closed basis marker.

PPB **detects and labels** shared training/target noise; it does not correct
it. Every correction needs an independent reference GWAS of the same trait, and
given one, evaluating the score against it is unbiased in a single line -- so
the condition that makes a correction valid is the condition that makes it
unnecessary. In-sample results are published as upper bounds, and the results
registry rejects a ``correctable`` status. See ``docs/OVERLAP.md``.

What remains in the package is the explicit ``unavailable`` marker: final
weights alone do not determine the trainer's sensitivity basis, so an unknown
trainer fails closed -- ``OverlapBasis.unavailable`` is what the results
pipeline records for a score whose trainer operator cannot be reconstructed.

The experimental fitting apparatus (``fit_overlap``,
``estimate_overlap_basis``, ``correct_overlap_numerator``) does not ship in the
package; it lives in ``experiments/overlap_detection.py`` as a validated
demonstration of the identification boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_AVAILABLE_BASIS_KINDS = frozenset({"linear_trace", "jacobian_hutchinson"})


@dataclass(frozen=True)
class OverlapBasis:
    """Trainer sensitivity to one unit of shared estimation noise.

    ``values[b]`` is the block basis ``q_b``.  For a linear trainer
    ``w = Phi z_train`` it is ``tr(Phi_b.T K_b)``, where ``K`` is the declared
    shared-noise covariance template.  (``Phi``, not ``A``: ``A`` is the
    discovery ancestry elsewhere -- see ``docs/NOTATION.md`` section 4.)
    ``support`` is the exact block support of the score numerator.
    ``support_hash`` and ``provenance`` tie the basis to the evaluated score and
    trainer artifact.

    Use :meth:`unavailable` when the trainer operator cannot be reconstructed;
    variant count is deliberately not used as a fallback.
    """

    values: np.ndarray | None
    kind: str
    provenance: str
    support_hash: str | None = None
    support: np.ndarray | None = None
    mc_se: float | None = None

    def __post_init__(self):
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("basis provenance must be a non-empty string")
        if self.kind == "unavailable":
            if self.values is not None or self.support is not None:
                raise ValueError("an unavailable basis cannot contain values or support")
            return
        if self.kind not in _AVAILABLE_BASIS_KINDS:
            allowed = ", ".join(sorted(_AVAILABLE_BASIS_KINDS))
            raise ValueError(f"available basis kind must be one of: {allowed}")
        if not isinstance(self.support_hash, str) or not self.support_hash.strip():
            raise ValueError("an available basis requires a non-empty support_hash")
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("basis values must be a non-empty finite 1-D array")
        support = (np.ones(values.size, dtype=bool) if self.support is None
                   else np.asarray(self.support, dtype=bool))
        if support.shape != values.shape or not np.any(support):
            raise ValueError("basis support must select at least one block")
        if self.mc_se is not None and (not np.isfinite(self.mc_se) or self.mc_se < 0):
            raise ValueError("basis mc_se must be finite and non-negative")
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "support", support.copy())

    @classmethod
    def unavailable(cls, provenance: str) -> "OverlapBasis":
        """Represent an unknown trainer operator explicitly."""
        return cls(values=None, kind="unavailable", provenance=provenance)

    @property
    def available(self) -> bool:
        return self.kind != "unavailable"
