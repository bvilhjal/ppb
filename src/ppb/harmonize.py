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


def _norm_chrom_array(chrom) -> np.ndarray:
    """Canonical chromosome labels: drop a ``chr`` prefix, map sex/MT codes.

    Lets a ``chr1``/``1`` (or ``X``/``23``) labelling mismatch between inputs
    still match by position -- a common reason real-data runs match nothing.
    Vectorized and computed once per table: at HM3+ scale the per-element Python
    string work was a measurable share of a genome-wide harmonization sweep.
    """
    s = np.char.upper(np.char.strip(np.asarray(chrom, dtype=str)))
    prefixed = np.char.startswith(s, "CHR")
    if prefixed.any():
        s = s.astype(object)
        s[prefixed] = [v[3:] for v in s[prefixed]]
        s = s.astype(str)
    for alias, canonical in _CHROM_ALIASES.items():
        s = np.where(s == alias, canonical, s)
    return s


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

    Normalized chromosomes, the allele lists, and the position index are derived
    once and cached; assigning to a field drops them, so the table stays correct
    if it is rebuilt in place.
    """

    _CACHES = ("_norm_chrom_cache", "_allele_list_cache", "_position_index_cache")

    chrom: np.ndarray
    pos: np.ndarray
    a1: np.ndarray
    a2: np.ndarray

    def __setattr__(self, name, value):
        if name in {"chrom", "pos", "a1", "a2"}:
            for key in self._CACHES:
                self.__dict__.pop(key, None)
        object.__setattr__(self, name, value)

    def __post_init__(self):
        self.chrom = np.asarray(self.chrom)
        self.pos = np.asarray(self.pos)
        self.a1 = np.char.upper(np.asarray(self.a1, dtype=str))
        self.a2 = np.char.upper(np.asarray(self.a2, dtype=str))
        if not all(a.ndim == 1 for a in (self.chrom, self.pos, self.a1, self.a2)):
            raise ValueError("variant fields must be 1-D")
        n = self.chrom.shape[0]
        if not (self.pos.shape[0] == self.a1.shape[0] == self.a2.shape[0] == n):
            raise ValueError("chrom, pos, a1, a2 must have equal length")
        try:
            numeric_pos = np.asarray(self.pos, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("variant positions must be finite integers") from exc
        if (not np.isfinite(numeric_pos).all()
                or not np.equal(numeric_pos, np.floor(numeric_pos)).all()):
            raise ValueError("variant positions must be finite integers")
        for chrom in self.chrom:
            if isinstance(chrom, (float, np.floating)) and not np.isfinite(chrom):
                raise ValueError("chromosome labels must be finite")

    @property
    def n(self) -> int:
        return int(self.chrom.shape[0])

    @property
    def norm_chrom(self) -> np.ndarray:
        """Canonical chromosome labels, computed once and cached."""
        cached = self.__dict__.get("_norm_chrom_cache")
        if cached is None:
            cached = _norm_chrom_array(self.chrom)
            self.__dict__["_norm_chrom_cache"] = cached
        return cached

    def allele_lists(self) -> tuple[list, list]:
        """``(a1, a2)`` as Python lists, computed once and cached.

        The orientation loop indexes single alleles; going through numpy scalars
        for each one costs more than the comparison it feeds.
        """
        cached = self.__dict__.get("_allele_list_cache")
        if cached is None:
            cached = (self.a1.tolist(), self.a2.tolist())
            self.__dict__["_allele_list_cache"] = cached
        return cached

    def position_index(self) -> dict:
        """``(norm_chrom, pos) -> [row indices]``, computed once and cached.

        A reference table is harmonized against repeatedly -- the genome-wide
        sweep in ``scripts/regenerate_results.py`` matches weights and every
        target against the same chromosome table -- and rebuilding this map per
        call was pure repeat work.
        """
        cached = self.__dict__.get("_position_index_cache")
        if cached is None:
            cached = {}
            for j, key in enumerate(zip(self.norm_chrom.tolist(),
                                        self.pos.astype(np.int64).tolist())):
                cached.setdefault(key, []).append(j)
            self.__dict__["_position_index_cache"] = cached
        return cached


def same_variants(a: VariantTable, b: VariantTable) -> bool:
    """True when two tables describe the same variants in the same order.

    Used to recognise that a table is being matched against itself, where there
    is no orientation to resolve and no strand ambiguity to drop.
    """
    if a is b:
        return True
    if a.n != b.n:
        return False
    return bool(np.array_equal(a.norm_chrom, b.norm_chrom)
                and np.array_equal(np.asarray(a.pos, dtype=np.int64),
                                   np.asarray(b.pos, dtype=np.int64))
                and np.array_equal(a.a1, b.a1) and np.array_equal(a.a2, b.a2))


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
                 *, remove_ambiguous: bool = True, return_mask: bool = False):
    """Align ``target``'s ``value`` onto ``reference`` order.

    Returns ``(aligned, report)`` where ``aligned`` is a length ``reference.n``
    array (0 where ``reference`` had no matching target variant) with signs
    flipped for allele swaps / strand flips, and ``report`` is a
    :class:`HarmonizeReport`. With ``return_mask=True``, a third boolean array
    marks the reference variants that genuinely matched. This distinguishes a
    matched value of zero from a missing variant and lets callers form a joint
    intersection across inputs. Strand-ambiguous palindromic SNPs are dropped
    when ``remove_ambiguous`` (the default), since strand cannot be resolved
    from alleles alone.
    """
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (target.n,):
        raise ValueError(f"value has shape {value.shape}, expected ({target.n},)")
    if not np.isfinite(value).all():
        raise ValueError("value must contain only finite numbers")

    pos_index = reference.position_index()

    aligned = np.zeros(reference.n, dtype=np.float64)
    used = np.zeros(reference.n, dtype=bool)
    n_matched = n_sign = n_strand = n_ambig = n_mismatch = n_unmatched = 0

    target_keys = zip(target.norm_chrom.tolist(),
                      target.pos.astype(np.int64).tolist())
    target_a1, target_a2 = target.allele_lists()
    ref_a1, ref_a2 = reference.allele_lists()

    for i, key in enumerate(target_keys):
        candidates = pos_index.get(key)
        if not candidates:
            n_unmatched += 1
            continue
        t1, t2 = target_a1[i], target_a2[i]
        if remove_ambiguous and frozenset((t1, t2)) in _AMBIGUOUS:
            n_ambig += 1
            continue
        matched = False
        for j in candidates:
            if used[j]:
                continue
            res = _orient(t1, t2, ref_a1[j], ref_a2[j])
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
    if return_mask:
        return aligned, report, used
    return aligned, report
