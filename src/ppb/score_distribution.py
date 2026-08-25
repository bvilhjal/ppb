"""Distribution of a polygenic score in a population, from allele frequencies and LD.

To report an individual's PGS as a standardized value or a percentile you need
the score's distribution in a reference population. The usual route is to score
a panel of individuals, which needs individual genotypes. This module takes the
summary-level route instead: the first two moments of a linear score are exactly
determined by allele frequencies and LD.

For a per-allele weight vector ``w`` and dosages ``g_j`` in {0, 1, 2}, writing
``S = sum_j w_j g_j``:

**(P1) Score mean.**

    E[S] = 2 sum_j w_j f_j

**(P2) Score variance.**

    Var(S) = sum_jk w_j w_k D_jk sd_j sd_k = (w * sd)^T D (w * sd)
    sd_j = sqrt(2 f_j (1 - f_j) (1 + F))

(P1) needs allele frequencies alone. (P2) is the same quadratic form the
estimator already computes as its denominator (M2) -- ``w^T D w`` on the
standardized gauge *is* the score variance, so this module and ``ppb.evaluate``
share their expensive half.

Two moments is all ``D`` can give. Per-variant third central moments are
available in closed form, and cross-block third moments vanish because the
blocks are independent, but the within-block three-locus terms are not in a
correlation matrix and nobody publishes them. So a percentile from this module
rests on a normal approximation, and ``max_variance_share`` reports how much
that approximation is being asked to do: with a block-diagonal reference ``S``
is *exactly* a sum of independent block contributions, so the CLT applies to a
genuine sum of independent terms and its quality is governed by how evenly the
variance is spread over them. One dominating block -- an APOE-like variant --
makes the score a mixture whose tail is not normal, which matters because the
claims people make from percentiles ("top 1%") live precisely there.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .harmonize import VariantTable, _harmonize_to_details
from .ld_backend import BlockDiagonalLD, LDBackend

_erfc = np.vectorize(math.erfc, otypes=[np.float64])


@dataclass
class ScoreDistribution:
    """Mean and standard deviation of a PGS in the population behind ``af``/``D``.

    ``max_variance_share`` is the largest single LD block's share of the
    variance, and it is the diagnostic to read before quoting a percentile: the
    normal approximation is a statement about a sum of independent block
    contributions, and a share near 1 means there is effectively one term.
    """

    mean: float
    sd: float
    variance: float
    n_reference: int
    n_variants_scored: int
    n_blocks: int | None = None
    max_variance_share: float | None = None
    inbreeding: float = 0.0
    weights_report: dict = field(default_factory=dict)

    def standardize(self, raw_score):
        """Standardized score ``(S - mean) / sd`` for one or many raw scores."""
        return (np.asarray(raw_score, dtype=np.float64) - self.mean) / self.sd

    def percentile(self, raw_score):
        """Normal-approximation percentile in [0, 100].

        Exact only insofar as ``S`` is normal; see ``max_variance_share`` and
        this module's docstring. Do not read the extreme tail off this number
        without checking that the variance is spread over many blocks.

        The tail is evaluated with ``erfc`` on the side that carries the
        precision: ``1 + erf(x)`` saturates to exactly 0 or 2 for
        ``|z| >~ 6``, pinning the percentile to exactly 0/100, while the
        ``erfc`` form keeps the lower tail representable to ~1e-300 and the
        upper tail to within ~1e-14 of 100.
        """
        z = self.standardize(raw_score)
        x = z / math.sqrt(2.0)
        return np.where(x >= 0.0,
                        100.0 - 50.0 * _erfc(x),
                        50.0 * _erfc(-x))

    def to_dict(self) -> dict:
        return asdict(self)


def score_distribution(ld: LDBackend, ld_variants: VariantTable,
                       weights_variants: VariantTable, weights,
                       allele_frequency, *, inbreeding: float = 0.0,
                       remove_ambiguous: bool = True) -> ScoreDistribution:
    """Mean and SD of the score ``weights`` in the population described by ``ld``.

    ``weights`` are **per-allele (dosage) weights** -- ordinary PGS Catalog
    weights -- with their own variant table and allele orientation; they are
    harmonized to ``ld_variants``. Standardized weights have no dosage mean, so
    they are not accepted here.

    ``allele_frequency`` is the frequency of each reference variant's *effect*
    allele (``ld_variants``' ``a1``), in reference order. It belongs to the
    reference rather than to the submission, exactly as ``genotype_sd`` does in
    :func:`ppb.evaluate`, so it is never harmonized. For a swapped submission
    allele, harmonization supplies the covariance sign and the raw-score mean
    uses the complementary dosage ``2(1-f)`` explicitly.

    ``inbreeding`` is Wright's ``F``, giving ``Var(g) = 2f(1-f)(1+F)``. It
    corrects the per-variant variance for a structured or admixed population but
    does **not** correct ``D``: admixture induces long-range LD that a
    block-diagonal within-ancestry panel cannot represent at all.
    """
    if ld.m != ld_variants.n:
        raise ValueError(
            f"LD backend has m={ld.m} but ld_variants has {ld_variants.n} variants")

    f = np.asarray(allele_frequency, dtype=np.float64)
    if f.shape != (ld_variants.n,):
        raise ValueError(
            f"allele_frequency has shape {f.shape}, expected ({ld_variants.n},)")
    if not np.isfinite(f).all() or np.any(f <= 0.0) or np.any(f >= 1.0):
        raise ValueError(
            "allele_frequency must contain only finite values strictly between "
            "0 and 1; a monomorphic variant has no standardized scale")
    inbreeding = float(inbreeding)
    if not math.isfinite(inbreeding) or inbreeding < 0.0 or inbreeding > 1.0:
        raise ValueError("inbreeding must be a finite value in [0, 1]")

    w, report, _, orientation, _ = _harmonize_to_details(
        ld_variants, weights_variants, weights,
        remove_ambiguous=remove_ambiguous)
    w = np.asarray(w, dtype=np.float64)
    if not np.isfinite(w).all():
        raise ValueError("harmonized weights contain non-finite values")

    sd = np.sqrt(2.0 * f * (1.0 - f) * (1.0 + inbreeding))
    # A swapped raw dosage is 2 - g_ref, not -g_ref. Using f - 1 directly for
    # its already-negated aligned weight both restores that affine intercept
    # and avoids subtracting two large, nearly equal genome-wide sums.
    dosage_frequency = f - (orientation == -1)
    mean = 2.0 * float(w @ dosage_frequency)
    w_std = w * sd

    n_blocks = max_share = None
    if isinstance(ld, BlockDiagonalLD):
        per_block = ld.block_quads(w_std)
        variance = float(per_block.sum())
        n_blocks = int(per_block.size)
        if variance > 0.0:
            max_share = float(per_block.max() / variance)
    else:
        variance = float(ld.quad(w_std))

    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            f"score variance is {variance!r}; a score with no variance has no "
            "standardized scale (are all matched weights zero?)")

    return ScoreDistribution(
        mean=mean,
        sd=math.sqrt(variance),
        variance=variance,
        n_reference=ld_variants.n,
        n_variants_scored=int(np.count_nonzero(w)),
        n_blocks=n_blocks,
        max_variance_share=max_share,
        inbreeding=inbreeding,
        weights_report=report.to_dict(),
    )
