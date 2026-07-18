"""Variant schema and allele harmonization.

To evaluate a polygenic score, the weights ``w``, the target summary statistics
``z``, and the LD matrix ``D`` must all refer to the same variants in the same
order with a consistent effect allele. :func:`harmonize_to` aligns an incoming
table to a canonical reference variant set, flipping the sign of the value on
allele swaps and strand flips and dropping strand-ambiguous (palindromic) SNPs.
This mirrors ``bigsnpr::snp_match`` (Privé), the setup the benchmark follows.

Variants are matched by ``(chrom, pos)``; alleles resolve the orientation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

_COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
_AMBIGUOUS = frozenset([frozenset(("A", "T")), frozenset(("C", "G"))])
# PLINK numeric sex / mitochondrial codes -> canonical letters (mirrors ldpred3).
_CHROM_ALIASES = {"23": "X", "24": "Y", "25": "XY", "26": "MT", "M": "MT"}


def _norm_chrom(c) -> str:
    """Canonical chromosome label: drop a ``chr`` prefix, map sex/MT codes.

    Lets a ``chr1``/``1`` (or ``X``/``23``) labelling mismatch between inputs
    still match by position -- a common reason real-data runs match nothing.
    """
    s = str(c).strip()
    if s[:3].lower() == "chr":
        s = s[3:]
    s = s.upper()
    return _CHROM_ALIASES.get(s, s)


def _complement(allele: str):
    """Reverse-complement of an allele (handles multi-base indels); None if non-ACGT."""
    try:
        return "".join(_COMP[b] for b in reversed(allele))
    except KeyError:
        return None


@dataclass
class VariantTable:
    """A set of variants: chromosome, position, effect allele ``a1``, other ``a2``.

    Alleles are upper-cased on construction. All four arrays must be equal length.
    """

    chrom: np.ndarray
    pos: np.ndarray
    a1: np.ndarray
    a2: np.ndarray

    def __post_init__(self):
        self.chrom = np.asarray(self.chrom)
        self.pos = np.asarray(self.pos)
        self.a1 = np.char.upper(np.asarray(self.a1, dtype=str))
        self.a2 = np.char.upper(np.asarray(self.a2, dtype=str))
        n = self.chrom.shape[0]
        if not (self.pos.shape[0] == self.a1.shape[0] == self.a2.shape[0] == n):
            raise ValueError("chrom, pos, a1, a2 must have equal length")
        if self.chrom.ndim != 1:
            raise ValueError("variant fields must be 1-D")

    @property
    def n(self) -> int:
        return int(self.chrom.shape[0])


@dataclass
class HarmonizeReport:
    """Counts from a harmonization pass (all machine-readable via :meth:`to_dict`)."""

    n_reference: int
    n_target: int
    n_matched: int
    n_sign_flipped: int
    n_strand_flipped: int
    n_ambiguous_removed: int
    n_mismatch: int          # position found but alleles incompatible
    n_unmatched: int         # position not found in the reference

    def to_dict(self) -> dict:
        return asdict(self)


def _orient(t1, t2, r1, r2):
    """Return ``(sign, strand_flipped)`` to map target alleles onto (r1, r2), or None.

    ``sign`` is +1 (same orientation) or -1 (effect allele is the other allele).
    """
    if (t1, t2) == (r1, r2):
        return 1, False
    if (t1, t2) == (r2, r1):
        return -1, False
    c1, c2 = _complement(t1), _complement(t2)
    if c1 is not None and c2 is not None:
        if (c1, c2) == (r1, r2):
            return 1, True
        if (c1, c2) == (r2, r1):
            return -1, True
    return None


def harmonize_to(reference: VariantTable, target: VariantTable, value,
                 *, remove_ambiguous: bool = True):
    """Align ``target``'s ``value`` onto ``reference`` order.

    Returns ``(aligned, report)`` where ``aligned`` is a length ``reference.n``
    array (0 where ``reference`` had no matching target variant) with signs
    flipped for allele swaps / strand flips, and ``report`` is a
    :class:`HarmonizeReport`. Strand-ambiguous palindromic SNPs are dropped when
    ``remove_ambiguous`` (the default), since strand cannot be resolved from
    alleles alone.
    """
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (target.n,):
        raise ValueError(f"value has shape {value.shape}, expected ({target.n},)")

    pos_index: dict = {}
    for j in range(reference.n):
        pos_index.setdefault((_norm_chrom(reference.chrom[j]), int(reference.pos[j])), []).append(j)

    aligned = np.zeros(reference.n, dtype=np.float64)
    used = np.zeros(reference.n, dtype=bool)
    n_matched = n_sign = n_strand = n_ambig = n_mismatch = n_unmatched = 0

    for i in range(target.n):
        candidates = pos_index.get((_norm_chrom(target.chrom[i]), int(target.pos[i])))
        if not candidates:
            n_unmatched += 1
            continue
        t1, t2 = str(target.a1[i]), str(target.a2[i])
        if remove_ambiguous and frozenset((t1, t2)) in _AMBIGUOUS:
            n_ambig += 1
            continue
        matched = False
        for j in candidates:
            if used[j]:
                continue
            res = _orient(t1, t2, str(reference.a1[j]), str(reference.a2[j]))
            if res is None:
                continue
            sign, strand = res
            aligned[j] = sign * value[i]
            used[j] = True
            matched = True
            n_matched += 1
            if sign == -1:
                n_sign += 1
            if strand:
                n_strand += 1
            break
        if not matched:
            n_mismatch += 1          # position(s) present, but no allele orientation fit

    report = HarmonizeReport(
        n_reference=reference.n, n_target=target.n, n_matched=n_matched,
        n_sign_flipped=n_sign, n_strand_flipped=n_strand,
        n_ambiguous_removed=n_ambig, n_mismatch=n_mismatch, n_unmatched=n_unmatched)
    return aligned, report
