"""Reference-frequency decomposition of GWAS effect-allele frequencies.

For variants aligned to a panel's counted allele, PPB estimates the
simplex-constrained projection

    f_i  ~=  sum_k pi_k p_ik,      pi_k >= 0,  sum_k pi_k = 1.

The coefficients describe how the supplied *frequency profile* projects onto
the chosen reference populations. They are not individual-level ancestry
fractions: case/control ascertainment, changing samples across variants,
meta-analysis, drift, and missing reference populations can all change the
estimand. A delete-one-chromosome locus jackknife describes regional
instability conditional on the fixed panel; it is not total sampling
uncertainty. Fixed fit and identifiability gates refuse the most misleading
decompositions.

Panel schema 2 deliberately retains the canonical
``smartpred-af-panel-v2`` digest domain so panels built by the original
SMARTpred implementation remain scientifically compatible. Registry and
web-service concerns belong to SMARTpred, not this module.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PANEL_SCHEMA = 2
_AF_DTYPE = np.dtype("<f8")
_INT_DTYPE = np.dtype("<i8")

_MIN_MATCHED_CRITICAL = 1000
_MIN_AF_CORR_WARNING = 0.85
_MIN_AUTOSOMES = 10
_MAX_CHROMOSOME_FRACTION = 0.25
_MAX_CONTRAST_CONDITION = 1e3
_MAX_RESIDUAL_RMS = 0.05
_MAX_RESIDUAL_TO_CONTRAST = 0.50
_BOUNDARY_WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FrequencyPanel:
    """One validated, immutable per-population allele-frequency panel."""

    ids: np.ndarray             # rsID, panel order
    chrom: np.ndarray
    pos: np.ndarray
    counted_allele: np.ndarray
    other_allele: np.ndarray
    pops: list                  # population codes, af column order
    af: np.ndarray              # (m, K) float64, counted-allele frequency
    n_samples: np.ndarray       # (K,) reference sample counts
    source: str
    source_url: str
    genome_build: str
    panel_sha256: str

    def __len__(self):
        return int(len(self.ids))


def _digest_field(digest, name, payload):
    """Add one length-delimited field to a panel's canonical digest."""
    name = name.encode("ascii")
    payload = bytes(payload)
    digest.update(len(name).to_bytes(2, "little"))
    digest.update(name)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _text_payload(values):
    return json.dumps(
        [str(value) for value in np.asarray(values).reshape(-1).tolist()],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")


def _numeric_payload(values, dtype):
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    shape = json.dumps(array.shape, separators=(",", ":")).encode("ascii")
    return (len(shape).to_bytes(2, "little") + shape
            + memoryview(array).cast("B").tobytes())


def _panel_digest(*, ids, chrom, pos, counted, other, pops, af, n_samples,
                  source, source_url, genome_build):
    """Hash every field that can change a panel's scientific meaning."""
    digest = hashlib.sha256()
    digest.update(b"smartpred-af-panel-v2\0")
    for name, values in (
            ("ids", ids), ("chrom", chrom), ("counted_allele", counted),
            ("other_allele", other), ("pops", pops)):
        _digest_field(digest, name, _text_payload(values))
    _digest_field(digest, "pos", _numeric_payload(pos, _INT_DTYPE))
    _digest_field(digest, "af", _numeric_payload(af, _AF_DTYPE))
    _digest_field(
        digest, "n_samples", _numeric_payload(n_samples, _INT_DTYPE))
    for name, value in (
            ("source", source), ("source_url", source_url),
            ("genome_build", genome_build)):
        _digest_field(digest, name, str(value).encode("utf-8"))
    return digest.hexdigest()


def _scalar(archive, name, path):
    if name not in archive:
        raise ValueError(f"ancestry panel {path} is missing {name!r}")
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"ancestry panel {path} field {name!r} must be scalar")
    return value.reshape(-1)[0].item()


def _allele_vector(archive, name, m, path):
    value = np.asarray(archive[name])
    if value.shape != (m,):
        raise ValueError(
            f"ancestry panel {path} field {name!r} must have length {m:,}")
    normalized = np.char.upper(np.char.strip(value.astype(str)))
    if not np.all(np.isin(normalized, ("A", "C", "G", "T"))):
        raise ValueError(
            f"ancestry panel {path} field {name!r} must be A/C/G/T alleles")
    return normalized


def write_frequency_panel(
        path, *, ids, chrom, pos, counted_allele, other_allele, pops, af,
        n_samples, source, source_url="", genome_build="") -> Path:
    """Atomically write one validated ancestry AF panel."""
    path = Path(path)
    ids = np.char.strip(np.asarray(ids).astype(str))
    m = len(ids)
    chrom = np.char.strip(np.asarray(chrom).astype(str))
    pos = np.asarray(pos, dtype=np.int64)
    if chrom.shape != (m,) or pos.shape != (m,):
        raise ValueError("panel chrom/pos must match the variant count")
    if ids.shape != (m,) or np.any(ids == "") or len(set(ids.tolist())) != m:
        raise ValueError("panel variant IDs must be non-empty and unique")
    counted = _allele_vector({"a": counted_allele}, "a", m, path)
    other = _allele_vector({"a": other_allele}, "a", m, path)
    palindromic = (counted == "A") & (other == "T") | \
                  (counted == "T") & (other == "A") | \
                  (counted == "C") & (other == "G") | \
                  (counted == "G") & (other == "C")
    if np.any(palindromic):
        raise ValueError(
            "panel contains strand-ambiguous variants; rebuild it without "
            "them")
    if np.any(counted == other):
        raise ValueError("panel contains homozygous allele pairs")
    pops = [str(pop).strip() for pop in np.asarray(pops).tolist()]
    if len(pops) < 2 or any(not pop for pop in pops) \
            or len(set(pops)) != len(pops):
        raise ValueError(
            "panel needs at least two non-empty, unique population codes")
    af = np.asarray(af, dtype=np.float64)
    n_samples = np.asarray(n_samples, dtype=np.int64)
    if af.shape != (m, len(pops)) or n_samples.shape != (len(pops),):
        raise ValueError("panel af must be (variants, populations)")
    if (not np.all(np.isfinite(af)) or np.any(af < 0.0) or np.any(af > 1.0)):
        raise ValueError("panel frequencies must be finite in [0, 1]")
    if np.any(n_samples <= 0):
        raise ValueError("panel populations need positive sample counts")
    source = str(source).strip()
    source_url = str(source_url).strip()
    genome_build = str(genome_build).strip()
    if not source:
        raise ValueError("panel source must be non-empty")
    digest = _panel_digest(
        ids=ids, chrom=chrom, pos=pos, counted=counted, other=other,
        pops=pops, af=af, n_samples=n_samples, source=source,
        source_url=source_url, genome_build=genome_build)
    path_tag = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{path_tag}.part.npz")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tmp,
            schema_version=np.array(PANEL_SCHEMA),
            ids=ids, chrom=chrom, pos=pos,
            counted_allele=counted, other_allele=other,
            pops=np.array(pops), af=af, n_samples=n_samples,
            source=np.array(source), source_url=np.array(source_url),
            genome_build=np.array(genome_build),
            panel_sha256=np.array(digest))
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return path



def load_frequency_panel(path, *, expected_sha256=None) -> FrequencyPanel:
    """Load and validate a panel, optionally against a trusted digest."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            schema = int(_scalar(archive, "schema_version", path))
            if schema != PANEL_SCHEMA:
                raise ValueError(
                    f"ancestry panel {path} has schema {schema}, expected "
                    f"{PANEL_SCHEMA}")
            stored = str(_scalar(archive, "panel_sha256", path)).lower()
            ids = np.asarray(archive["ids"])
            if ids.ndim != 1 or ids.dtype.kind not in "US":
                raise ValueError(
                    f"ancestry panel {path} ids must be a 1-d string vector")
            m = len(ids)
            counted = _allele_vector(archive, "counted_allele", m, path)
            other = _allele_vector(archive, "other_allele", m, path)
            pops = [str(pop).strip() for pop in
                    np.asarray(archive["pops"]).tolist()]
            af = np.asarray(archive["af"], dtype=np.float64)
            n_samples = np.asarray(archive["n_samples"], dtype=np.int64)
            chrom = np.char.strip(np.asarray(archive["chrom"]).astype(str))
            pos = np.asarray(archive["pos"], dtype=np.int64)
            source = str(_scalar(archive, "source", path)).strip()
            source_url = str(_scalar(archive, "source_url", path)).strip()
            genome_build = str(_scalar(archive, "genome_build", path)).strip()
    except (OSError, EOFError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"cannot read ancestry panel {path}: {exc}") from exc
    ids = np.char.strip(ids.astype(str))
    palindromic = ((counted == "A") & (other == "T")
                   | (counted == "T") & (other == "A")
                   | (counted == "C") & (other == "G")
                   | (counted == "G") & (other == "C"))
    if (len(pops) < 2 or any(not pop for pop in pops)
            or len(set(pops)) != len(pops)
            or af.shape != (m, len(pops))
            or n_samples.shape != (len(pops),)
            or not np.all(np.isfinite(af)) or np.any(af < 0.0)
            or np.any(af > 1.0) or np.any(n_samples <= 0)
            or chrom.shape != (m,) or pos.shape != (m,)
            or np.any(ids == "") or len(set(ids.tolist())) != m
            or np.any(counted == other) or np.any(palindromic)
            or not source):
        raise ValueError(f"ancestry panel {path} is structurally invalid")
    digest = _panel_digest(
        ids=ids, chrom=chrom, pos=pos, counted=counted, other=other,
        pops=pops, af=af, n_samples=n_samples, source=source,
        source_url=source_url, genome_build=genome_build)
    if not hmac.compare_digest(stored, digest):
        raise ValueError(
            f"ancestry panel {path} failed its content-hash check; it has "
            "changed since it was built")
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef"
                                      for char in expected):
            raise ValueError("expected ancestry-panel sha256 is invalid")
        if not hmac.compare_digest(expected, digest):
            raise ValueError(
                f"ancestry panel {path} does not match its externally "
                "registered sha256; refuse to use a replaced panel")
    return FrequencyPanel(
        ids=ids, chrom=chrom, pos=pos, counted_allele=counted,
        other_allele=other, pops=pops, af=af, n_samples=n_samples,
        source=source, source_url=source_url,
        genome_build=genome_build, panel_sha256=digest)



_AMBIGUOUS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}


@dataclass
class MatchedFrequencies:
    """GWAS EAF rows aligned to panel variants, with attrition counts."""

    panel_rows: np.ndarray     # index into panel variants, sorted
    eaf: np.ndarray            # GWAS EAF on the panel counted allele
    counts: dict


def match_effect_allele_frequencies(
        ids, effect_allele, other_allele, eaf,
        panel: FrequencyPanel) -> MatchedFrequencies:
    """Align array-like GWAS EAF fields to one reference-frequency panel.

    Palindromic variants and allele pairs absent from the panel are excluded
    and counted; duplicated IDs keep the first occurrence. Swapped alleles
    flip the frequency onto the panel's counted allele.
    """
    ss_id = np.asarray(ids)
    ss_ea = np.char.upper(
        np.char.strip(np.asarray(effect_allele).astype(str)))
    ss_oa = np.char.upper(
        np.char.strip(np.asarray(other_allele).astype(str)))
    eaf = np.asarray(eaf, dtype=float)
    n_input = len(ss_id)
    if (ss_id.shape != (n_input,) or ss_ea.shape != (n_input,)
            or ss_oa.shape != (n_input,) or eaf.shape != (n_input,)):
        raise ValueError(
            "ids, effect_allele, other_allele, and eaf must be 1-d arrays "
            "of equal length")

    # Panel IDs are unique by construction, so one ID->row map is enough.
    # Storing three-string tuple keys roughly triples the Python-object cost
    # at the million-variant scale for no additional information.
    panel_row = {str(pid): row for row, pid in enumerate(panel.ids)}

    rows, freqs = [], []
    n_dupe = n_missing = n_mismatch = n_ambiguous = n_nonfinite = 0
    n_out_of_range = 0
    seen = set()
    for i in range(n_input):
        pid = str(ss_id[i])
        ea, oa = ss_ea[i], ss_oa[i]
        value = eaf[i]
        if not np.isfinite(value):
            n_nonfinite += 1
            continue
        if value < 0.0 or value > 1.0:
            n_out_of_range += 1
            continue
        if pid in seen:
            n_dupe += 1
            continue
        seen.add(pid)
        if (ea, oa) in _AMBIGUOUS:
            n_ambiguous += 1
            continue
        row = panel_row.get(pid)
        if row is None:
            n_missing += 1
            continue
        pc = panel.counted_allele[row]
        po = panel.other_allele[row]
        if ea == pc and oa == po:
            flip = False
        elif ea == po and oa == pc:
            flip = True
        else:
            n_mismatch += 1
            continue
        rows.append(row)
        freqs.append(1.0 - value if flip else value)

    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    freqs = np.asarray(freqs, dtype=np.float64)[order] \
        if len(freqs) else np.zeros(0)
    rows = rows[order]
    counts = {
        "n_input": n_input,
        "n_dropped_duplicate": n_dupe,
        "n_dropped_palindromic": n_ambiguous,
        "n_dropped_not_in_panel": n_missing,
        "n_dropped_allele_mismatch": n_mismatch,
        "n_dropped_nonfinite_eaf": n_nonfinite,
        "n_dropped_out_of_range_eaf": n_out_of_range,
        "n_matched": int(len(rows)),
    }
    return MatchedFrequencies(panel_rows=rows, eaf=freqs, counts=counts)


def _simplex_least_squares(P, f):
    """Exactly minimize ‖f − Pπ‖² over π ≥ 0 and Σπ = 1.

    Population panels have small ``K`` (five for 1000 Genomes), so enumerating
    every non-empty simplex face is simpler and safer than approximating the
    equality constraint with a large penalty row.
    """
    P = np.asarray(P, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    if P.ndim != 2 or f.shape != (P.shape[0],) or P.shape[1] == 0:
        raise ValueError("simplex least squares needs P=(variants, populations)")
    K = P.shape[1]
    if K > 15:
        raise ValueError("simplex face enumeration supports at most 15 populations")
    gram_full = P.T @ P
    rhs_full = P.T @ f
    f_squared = float(f @ f)
    best = None
    best_loss = np.inf
    tolerance = 1e-10
    for mask in range(1, 1 << K):
        active = np.array([i for i in range(K) if mask & (1 << i)])
        q = len(active)
        gram = gram_full[np.ix_(active, active)]
        kkt = np.block([
            [gram, np.ones((q, 1))],
            [np.ones((1, q)), np.zeros((1, 1))],
        ])
        rhs = np.concatenate([rhs_full[active], [1.0]])
        candidate, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
        weights = candidate[:q]
        if abs(float(weights.sum()) - 1.0) > tolerance:
            continue
        if np.any(weights < -tolerance):
            continue
        weights[np.abs(weights) <= tolerance] = 0.0
        weights /= weights.sum()
        full = np.zeros(K)
        full[active] = weights
        loss = float(
            f_squared - 2.0 * full @ rhs_full + full @ gram_full @ full)
        if loss < best_loss:
            best, best_loss = full, loss
    if best is None:  # Every simplex vertex is feasible, barring non-finite data.
        raise ValueError("could not solve simplex least squares")
    return best


def _autosome_label(chrom):
    value = str(chrom).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    try:
        number = int(value)
    except ValueError:
        return None
    return str(number) if 1 <= number <= 22 else None


def _contrast_rank_condition_from_gram(gram, n_rows, n_populations):
    """Rank and condition of row-centred population contrasts.

    Each variant row lives in 1-perp, so the Gram has a structural null along
    the all-ones vector. Rounding can lift that eigenvalue just above a
    machine-eps cutoff and report rank K for a K-population design. Count
    rank in the (K-1)-dimensional contrast subspace by discarding the
    smallest eigenvalue before applying the tolerance.
    """
    expected_rank = max(n_populations - 1, 0)
    gram = np.asarray(gram, dtype=np.float64)
    k = int(gram.shape[0])
    if k == 0:
        return 0, expected_rank, None if expected_rank else 1.0
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    singular = np.sqrt(np.maximum(eigenvalues[::-1], 0.0))
    if k >= 1:
        singular = singular[: k - 1]
    largest = float(singular[0]) if len(singular) else 0.0
    tolerance = max(n_rows, n_populations, 1) * np.finfo(float).eps * largest
    rank = int(np.sum(singular > tolerance)) if largest > 0 else 0
    if expected_rank and rank >= expected_rank:
        condition = float(singular[0] / singular[expected_rank - 1])
    elif expected_rank:
        condition = None
    else:
        condition = 1.0
    return rank, expected_rank, condition


def estimate_frequency_composition(
        matched: MatchedFrequencies, panel: FrequencyPanel) -> dict:
    """Fit proportions, jackknife SEs, and diagnostics for one matched set."""
    from collections import defaultdict

    pops = panel.pops
    rows = matched.panel_rows
    f = matched.eaf
    P = panel.af[rows]
    chroms = np.asarray(panel.chrom)[rows]
    n = len(rows)
    result = {
        "channel": "frequency",
        "estimand": "af_projection_weights",
        "populations": list(pops),
        "n_used": n,
        "proportions": None,
        "proportions_se": None,
        "af_corr": None,
        "residual_rms": None,
        "residual_max_abs": None,
        "reference_sampling_rms": None,
        "panel_confusability": _panel_confusability(panel, rows=rows),
        "residual_rms_over_contrast": None,
        "validity_thresholds": {
            "minimum_matched_variants": _MIN_MATCHED_CRITICAL,
            "minimum_distinct_autosomes": _MIN_AUTOSOMES,
            "maximum_chromosome_fraction": _MAX_CHROMOSOME_FRACTION,
            "max_contrast_condition_number": _MAX_CONTRAST_CONDITION,
            "max_residual_rms": _MAX_RESIDUAL_RMS,
            "max_residual_rms_over_contrast": _MAX_RESIDUAL_TO_CONTRAST,
        },
        "jackknife_note": None,
        "uncertainty_note": (
            "delete-one-chromosome jackknife conditions on the fixed panel; "
            "it omits target and reference allele-frequency sampling "
            "uncertainty and model misspecification"),
        "boundary_note": None,
        "acceptance_gate_note": (
            "count, chromosome-coverage, contrast-rank, condition-number, "
            "and fit cutoffs are engineering gates, not calibrated "
            "statistical thresholds"),
        "chromosome_coverage": None,
        "leave_one_chromosome_identifiability": None,
        "status": "estimated",
        "warnings": [],
    }
    if n == 0:
        invalid = matched.counts.get("n_dropped_out_of_range_eaf", 0)
        if invalid:
            result["status"] = "invalid_input"
            result["warnings"].append(
                f"{invalid:,} finite effect-allele frequencies were outside "
                "[0, 1]; the input is invalid")
        else:
            result["status"] = "unavailable"
            result["warnings"].append("no variants matched the panel")
        return result

    pi = _simplex_least_squares(P, f)
    fitted = P @ pi
    if n >= 2 and np.std(fitted) > 0 and np.std(f) > 0:
        result["af_corr"] = float(np.corrcoef(f, fitted)[0, 1])
    residual = f - fitted
    result["residual_rms"] = float(np.sqrt(np.mean(residual ** 2)))
    result["residual_max_abs"] = float(np.max(np.abs(residual)))
    result["proportions"] = [float(value) for value in pi]
    reference_variance = np.sum(
        pi[None, :] ** 2 * P * (1.0 - P)
        / (2.0 * panel.n_samples[None, :]), axis=1)
    result["reference_sampling_rms"] = float(
        np.sqrt(np.mean(reference_variance)))
    if np.any(pi <= _BOUNDARY_WEIGHT_TOLERANCE):
        result["boundary_note"] = (
            "at least one fitted weight is on or near the simplex boundary; "
            "ordinary symmetric standard-error intervals are invalid")
    contrast_rms = result["panel_confusability"]["contrast_rms"]
    if contrast_rms is not None and contrast_rms > 0:
        result["residual_rms_over_contrast"] = float(
            result["residual_rms"] / contrast_rms)

    # Delete-one-chromosome jackknife: LD among nearby variants makes an
    # i.i.d. SE wrong; dropping whole chromosomes absorbs that correlation.
    by_chrom = defaultdict(list)
    for i, chrom in enumerate(chroms):
        label = _autosome_label(chrom) or str(chrom).strip()
        by_chrom[label].append(i)
    usable = [indices for indices in by_chrom.values() if len(indices) < n]

    autosomes = {_autosome_label(chrom) for chrom in by_chrom}
    autosomes.discard(None)
    largest_chromosome, largest_indices = max(
        by_chrom.items(), key=lambda item: (len(item[1]), str(item[0])))
    largest_fraction = len(largest_indices) / n
    result["chromosome_coverage"] = {
        "n_distinct_autosomes": len(autosomes),
        "largest_chromosome": str(largest_chromosome),
        "largest_chromosome_fraction": float(largest_fraction),
    }

    centered = P - P.mean(axis=1, keepdims=True)
    leave_one = []
    for chrom, indices in by_chrom.items():
        keep = np.ones(n, dtype=bool)
        keep[np.asarray(indices, dtype=np.int64)] = False
        kept = int(keep.sum())
        # Form the kept Gram directly. Downdating ``C.T @ C`` by the
        # dropped rows is algebraically exact but summation-order rounding
        # lifts the exact 1-vector null of a row-centred contrast by ~1e-6,
        # so a K-population design was recorded as rank K rather than K-1.
        kept_centered = centered[keep]
        gram = kept_centered.T @ kept_centered
        rank, expected_rank, condition = _contrast_rank_condition_from_gram(
            gram, kept, len(pops))
        identifiable = (rank >= expected_rank
                        and condition is not None
                        and condition <= _MAX_CONTRAST_CONDITION)
        leave_one.append({
            "chromosome": str(chrom),
            "n_retained": int(kept),
            "contrast_rank": rank,
            "expected_contrast_rank": expected_rank,
            "contrast_condition_number": condition,
            "identifiable": bool(identifiable),
        })
    result["leave_one_chromosome_identifiability"] = {
        "all_identifiable": all(item["identifiable"] for item in leave_one),
        "designs": leave_one,
    }

    identifiable_by_chrom = {
        item["chromosome"]: item["identifiable"] for item in leave_one}
    if len(usable) >= 2:
        leave_out = np.zeros(n, dtype=bool)
        estimates = []
        for chrom, indices in by_chrom.items():
            if len(indices) >= n:
                continue
            if not identifiable_by_chrom.get(str(chrom), False):
                continue
            leave_out[:] = False
            leave_out[indices] = True
            keep = ~leave_out
            estimates.append(_simplex_least_squares(P[keep], f[keep]))
        if len(estimates) >= 2:
            estimates = np.asarray(estimates)
            g = len(estimates)
            se = np.sqrt((g - 1) / g * np.sum(
                (estimates - estimates.mean(0)) ** 2, axis=0))
            result["proportions_se"] = [float(value) for value in se]
        else:
            result["jackknife_note"] = (
                "fewer than two leave-one-chromosome designs passed the "
                "rank/condition gate; jackknife standard errors were not "
                "computed")
    else:
        result["jackknife_note"] = (
            "fewer than two chromosomes carried matched variants; "
            "jackknife standard errors were not computed")

    if matched.counts.get("n_dropped_out_of_range_eaf", 0):
        count = matched.counts["n_dropped_out_of_range_eaf"]
        result["warnings"].append(
            f"{count:,} finite effect-allele frequencies were outside [0, 1]; "
            "the input is invalid")
        result["status"] = "invalid_input"
    if n < _MIN_MATCHED_CRITICAL:
        result["warnings"].append(
            f"only {n:,} matched variants (minimum {_MIN_MATCHED_CRITICAL:,})")
        if result["status"] == "estimated":
            result["status"] = "insufficient"
    if len(autosomes) < _MIN_AUTOSOMES:
        result["warnings"].append(
            f"matched variants span only {len(autosomes)} distinct autosomes "
            f"(minimum {_MIN_AUTOSOMES})")
        if result["status"] == "estimated":
            result["status"] = "insufficient"
    if largest_fraction > _MAX_CHROMOSOME_FRACTION:
        result["warnings"].append(
            f"chromosome {largest_chromosome} contributes "
            f"{largest_fraction:.1%} of matched variants (maximum "
            f"{_MAX_CHROMOSOME_FRACTION:.0%})")
        if result["status"] == "estimated":
            result["status"] = "insufficient"
    corr = result["af_corr"]
    if corr is not None and corr < _MIN_AF_CORR_WARNING:
        result["warnings"].append(
            f"observed-vs-fitted frequency correlation is {corr:.3f} "
            f"(descriptive warning threshold {_MIN_AF_CORR_WARNING}); the "
            "dataset may not match the panel's ancestries, build, or allele "
            "orientation")
    conf = result["panel_confusability"]
    if conf["contrast_rank"] < conf["expected_contrast_rank"]:
        result["warnings"].append(
            "the matched population-frequency contrasts are rank deficient "
            f"({conf['contrast_rank']} of {conf['expected_contrast_rank']}); "
            "the requested ancestry components are not identifiable")
        if result["status"] == "estimated":
            result["status"] = "nonidentifiable"
    condition = conf["contrast_condition_number"]
    if (condition is not None and np.isfinite(condition)
            and condition > _MAX_CONTRAST_CONDITION):
        result["warnings"].append(
            "the matched population-frequency contrasts are too ill "
            f"conditioned ({condition:.3g}; maximum "
            f"{_MAX_CONTRAST_CONDITION:.3g})")
        if result["status"] == "estimated":
            result["status"] = "nonidentifiable"
    failed_leave_one = [
        item for item in leave_one if not item["identifiable"]]
    if failed_leave_one:
        failed_labels = ", ".join(
            item["chromosome"] for item in failed_leave_one)
        result["warnings"].append(
            "population-frequency contrasts fail the rank/condition gate "
            "after leaving out chromosome(s): " + failed_labels)
        if result["status"] == "estimated":
            result["status"] = "nonidentifiable"
        result["proportions_se"] = None
        if "jackknife_note" not in result:
            result["jackknife_note"] = (
                "leave-one-chromosome designs failed identifiability; "
                "jackknife standard errors are withheld")
    relative_residual = result["residual_rms_over_contrast"]
    poor_absolute_fit = result["residual_rms"] > _MAX_RESIDUAL_RMS
    poor_relative_fit = (relative_residual is None
                         or relative_residual > _MAX_RESIDUAL_TO_CONTRAST)
    if poor_absolute_fit or poor_relative_fit:
        relative_text = ("undefined" if relative_residual is None
                         else f"{relative_residual:.3f}")
        result["warnings"].append(
            "the fitted mixture fails a fixed heuristic model-fit screen "
            f"(residual RMS {result['residual_rms']:.4f}, maximum "
            f"{_MAX_RESIDUAL_RMS:.4f}; residual/contrast RMS "
            f"{relative_text}, maximum {_MAX_RESIDUAL_TO_CONTRAST:.2f}); "
            "the EAF profile is not adequately represented by this panel; "
            "the screen is not calibrated to GWAS or reference-frequency "
            "sampling uncertainty")
        if result["status"] == "estimated":
            result["status"] = "poor_fit"
    if result["proportions"] is not None and max(pi) >= 0.999:
        result["warnings"].append(
            "a single population absorbs the whole composition; with a poor "
            "fit this can mean the dataset's ancestry is outside the "
            "panel, not a pure single-origin result")
    return result


def decompose_effect_allele_frequencies(
        ids, effect_allele, other_allele, eaf,
        panel: FrequencyPanel) -> dict:
    """Match arrays and return one JSON-friendly decomposition result.

    The returned dictionary preserves the estimator's result schema and adds
    ``matching`` with the complete row-attrition counts.
    """
    matched = match_effect_allele_frequencies(
        ids, effect_allele, other_allele, eaf, panel)
    result = estimate_frequency_composition(matched, panel)
    result["matching"] = dict(matched.counts)
    return result


def _panel_confusability(panel: FrequencyPanel, *, rows=None) -> dict:
    """Separability diagnostics in the matched population-contrast space."""
    P = panel.af if rows is None else panel.af[np.asarray(rows, dtype=np.int64)]
    K = len(panel.pops)
    correlations = {}
    for i in range(K):
        for j in range(i + 1, K):
            a, b = P[:, i], P[:, j]
            if np.std(a) > 0 and np.std(b) > 0:
                correlations[f"{panel.pops[i]}-{panel.pops[j]}"] = float(
                    np.corrcoef(a, b)[0, 1])
    expected_rank = max(K - 1, 0)
    centered = P - P.mean(axis=1, keepdims=True) if len(P) else P.copy()
    singular = np.linalg.svd(centered, compute_uv=False) if len(P) else np.zeros(0)
    largest = float(singular[0]) if len(singular) else 0.0
    tolerance = max(centered.shape, default=1) * np.finfo(float).eps * largest
    rank = int(np.sum(singular > tolerance)) if largest > 0 else 0
    if expected_rank and rank >= expected_rank:
        condition = float(singular[0] / singular[expected_rank - 1])
    elif expected_rank:
        condition = None
    else:
        condition = 1.0
    contrast_rms = (float(np.sqrt(np.mean(centered ** 2)))
                    if centered.size else None)
    return {
        "pairwise_af_correlation": correlations,
        # Historical aliases retained for old result consumers; both now refer
        # to the scientifically relevant matched contrast design.
        "gram_condition_number": condition,
        "effective_rank": rank,
        "contrast_condition_number": condition,
        "contrast_rank": rank,
        "expected_contrast_rank": expected_rank,
        "contrast_rms": contrast_rms,
        "n_variants": int(len(P)),
        "note": (
            "identifiability is determined from population-frequency "
            "contrasts on the matched variants; raw pairwise AF correlations "
            "are descriptive because their shared baseline can be misleading"),
    }
