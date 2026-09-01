#!/usr/bin/env python
"""Benchmark compact-reference LD-moment ancestry decomposition on Yengo height.

Estimator A is the predeclared benchmark channel.  Estimator B and an
independent sign-flip analysis are diagnostics only.  The estimand is the
best-fitting mixture of the fixed 1000 Genomes LD-covariance references, not
participant ancestry fractions.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppb  # noqa: E402
from ppb import (estimate_bilinear_from_design,  # noqa: E402
                 estimate_pair_products_from_design,
                 load_frequency_panel)
from scripts.build_ancestry_ld_design import _semantic_digest  # noqa: E402
from scripts.ancestry_frequency_gwas_benchmark import (  # noqa: E402
    DEFAULT_LDPRED3_REPO, DEFAULT_PANEL, DEFAULT_SUMSTATS_DIR, HEIGHT_PAPER,
    HEIGHT_PAPER_DOI, HEIGHT_PAPER_PMID, LDPRED3_REVISION, PANEL_SHA256,
    STUDIES, acquire_study, sha256_file,
)


DEFAULT_DESIGN = (
    ROOT / "data" / "ancestry_ld"
    / "kgp_phase3_common_distant_hm3_ld_design.npz"
)
DEFAULT_DESIGN_SHA256 = (
    "1ce09d4c224b11cde8996b97b0b9b119040fab53642ebc37e0ebc5eb1b93114e"
)
DESIGN_SCHEMA = 1
SIGN_FLIP_REPLICATES = 200
ARRAY_FIELDS = (
    "populations", "sample_counts", "ids", "chrom", "pos",
    "counted_allele", "other_allele", "variant_block", "block_source_id",
    "block_chrom", "block_start", "block_end", "pair_i", "pair_j",
    "pair_block", "pair_design", "quadratic_design", "bilinear_design",
)
SCALAR_FIELDS = (
    "schema_version", "maf_threshold", "min_block_gap_bp",
    "min_variants_per_block", "max_variants_per_block", "ld_floor",
    "max_pairs_per_block", "source", "source_url", "source_archive_md5",
    "af_panel_sha256", "genome_build",
)


def _vector(value, name, *, length=None, integer=False):
    value = np.asarray(value)
    if value.ndim != 1 or (length is not None and value.shape != (length,)):
        suffix = "" if length is None else f" of length {length:,}"
        raise ValueError(f"LD design field {name!r} must be a vector{suffix}")
    if integer and not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"LD design field {name!r} must be integer-valued")
    return value


def _scalar(archive, name, path):
    if name not in archive:
        raise ValueError(f"LD design {path} is missing {name!r}")
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"LD design field {name!r} must be scalar")
    return value.reshape(-1)[0].item()


def _text(values):
    return np.char.upper(np.char.strip(np.asarray(values).astype(str)))


def load_ld_design(path: Path, expected_sha256: str | None = None):
    """Load and structurally validate one compact, block-streamed LD design."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing LD design {path}")
    file_sha256 = sha256_file(path)
    if expected_sha256 and file_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"LD design file hash {file_sha256} does not match expected "
            f"{expected_sha256}"
        )
    with np.load(path, allow_pickle=False) as archive:
        missing = (set(ARRAY_FIELDS) | set(SCALAR_FIELDS)) - set(archive.files)
        if missing:
            raise ValueError(f"LD design {path} is missing {sorted(missing)}")
        raw = {name: np.asarray(archive[name]).copy() for name in archive.files}
        scalars = {name: _scalar(raw, name, path) for name in SCALAR_FIELDS}
        semantic = (_scalar(raw, "semantic_sha256", path)
                    if "semantic_sha256" in raw else None)
        arrays = {name: raw[name] for name in ARRAY_FIELDS}
        extra = {}
        for name in set(raw) - set(ARRAY_FIELDS) - set(SCALAR_FIELDS):
            if name == "semantic_sha256":
                continue
            value = raw[name]
            if value.size != 1:
                raise ValueError(
                    f"unrecognized LD design field {name!r} must be scalar"
                )
            extra[name] = value.reshape(-1)[0].item()

    if int(scalars["schema_version"]) != DESIGN_SCHEMA:
        raise ValueError(
            f"unsupported LD design schema {scalars['schema_version']}; "
            f"expected {DESIGN_SCHEMA}"
        )
    pops = np.char.strip(_vector(arrays["populations"], "populations").astype(str))
    k = pops.size
    if k < 2 or np.any(pops == "") or len(set(pops.tolist())) != k:
        raise ValueError("LD design populations must contain unique non-empty names")
    sample_counts = _vector(
        arrays["sample_counts"], "sample_counts", length=k, integer=True)
    if (sample_counts <= 0).any():
        raise ValueError("LD design sample_counts must be positive")

    ids = np.char.strip(_vector(arrays["ids"], "ids").astype(str))
    m = ids.size
    if m == 0 or np.any(ids == "") or len(set(ids.tolist())) != m:
        raise ValueError("LD design IDs must be non-empty and unique")
    chrom = np.char.strip(_vector(arrays["chrom"], "chrom", length=m).astype(str))
    pos = _vector(arrays["pos"], "pos", length=m, integer=True)
    counted = _text(_vector(arrays["counted_allele"], "counted_allele", length=m))
    other = _text(_vector(arrays["other_allele"], "other_allele", length=m))
    if (not np.all(np.isin(counted, ("A", "C", "G", "T"))) or
            not np.all(np.isin(other, ("A", "C", "G", "T"))) or
            np.any(counted == other)):
        raise ValueError("LD design alleles must be distinct A/C/G/T bases")
    palindromic = np.isin(
        np.char.add(counted, other), ("AT", "TA", "CG", "GC")
    )
    if palindromic.any():
        raise ValueError("LD design must exclude strand-ambiguous variants")
    if np.any(chrom == "") or np.any(pos <= 0):
        raise ValueError("LD design chromosome and position fields are invalid")

    block_source = _vector(arrays["block_source_id"], "block_source_id")
    b = block_source.size
    if b == 0:
        raise ValueError("LD design must contain at least one block")
    for name in ("block_chrom", "block_start", "block_end"):
        _vector(arrays[name], name, length=b,
                integer=name in {"block_start", "block_end"})
    variant_block = _vector(
        arrays["variant_block"], "variant_block", length=m, integer=True)
    if (variant_block < 0).any() or (variant_block >= b).any():
        raise ValueError("LD design variant_block contains invalid block indices")

    pair_i = _vector(arrays["pair_i"], "pair_i", integer=True)
    p = pair_i.size
    pair_j = _vector(arrays["pair_j"], "pair_j", length=p, integer=True)
    pair_block = _vector(
        arrays["pair_block"], "pair_block", length=p, integer=True)
    if (p == 0 or (pair_i < 0).any() or (pair_j < 0).any() or
            (pair_i >= m).any() or (pair_j >= m).any() or
            (pair_i == pair_j).any() or (pair_block < 0).any() or
            (pair_block >= b).any()):
        raise ValueError("LD design pair indices are empty or invalid")
    if (np.any(variant_block[pair_i] != pair_block) or
            np.any(variant_block[pair_j] != pair_block)):
        raise ValueError("LD design pairs must lie within their declared block")
    q = k * (k + 1) // 2
    expected_shapes = {
        "pair_design": (p, k),
        "quadratic_design": (p, q),
        "bilinear_design": (m, q),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(arrays[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(
                f"LD design field {name!r} must have finite shape {shape}"
            )
        arrays[name] = value
    if semantic is not None:
        semantic = str(semantic).lower()
        if len(semantic) != 64 or any(c not in "0123456789abcdef" for c in semantic):
            raise ValueError("LD design semantic_sha256 must be 64 hexadecimal digits")
        observed_semantic = _semantic_digest({
            name: value for name, value in raw.items()
            if name != "semantic_sha256"
        })
        if observed_semantic != semantic:
            raise ValueError(
                f"LD design semantic hash {observed_semantic} does not match "
                f"stored {semantic}"
            )

    selection = {
        name: scalars[name] for name in (
            "maf_threshold", "min_block_gap_bp", "min_variants_per_block",
            "max_variants_per_block", "ld_floor", "max_pairs_per_block")
    }
    if (not 0.0 < float(selection["maf_threshold"]) < 0.5 or
            int(selection["min_block_gap_bp"]) < 0 or
            int(selection["min_variants_per_block"]) < 2 or
            int(selection["max_variants_per_block"]) <
            int(selection["min_variants_per_block"]) or
            not 0.0 <= float(selection["ld_floor"]) <= 1.0 or
            int(selection["max_pairs_per_block"]) < 1):
        raise ValueError("LD design selection settings are invalid")
    return {
        **arrays, "populations": pops, "sample_counts": sample_counts,
        "ids": ids, "chrom": chrom, "pos": pos,
        "counted_allele": counted, "other_allele": other,
        "variant_block": variant_block, "pair_i": pair_i, "pair_j": pair_j,
        "pair_block": pair_block, "n_blocks": b, "file_sha256": file_sha256,
        "semantic_sha256": semantic, "selection": selection,
        "provenance": {
            name: scalars[name] for name in (
                "source", "source_url", "source_archive_md5",
                "af_panel_sha256", "genome_build")
        },
        "extra_scalars": extra,
    }


def _float(value):
    try:
        out = float((value or "").strip())
    except ValueError:
        return np.nan
    return out if np.isfinite(out) else np.nan


def read_harmonized_sumstats(path: Path, design):
    """Read raw z-scores and align their signs to the design counted allele."""
    opener = gzip.open if str(path).endswith(".gz") else open
    by_id = {variant: i for i, variant in enumerate(design["ids"].tolist())}
    found = {}
    with opener(path, "rt", encoding="utf-8", errors="strict", newline="") as src:
        reader = csv.DictReader(src, delimiter="\t")
        required = {"rsid", "effect_allele", "other_allele", "beta", "se", "n"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing normalized columns {sorted(missing)}")
        for row in reader:
            idx = by_id.get((row["rsid"] or "").strip())
            if idx is None or idx in found:
                continue
            beta, se, n = _float(row["beta"]), _float(row["se"]), _float(row["n"])
            effect = (row["effect_allele"] or "").strip().upper()
            other = (row["other_allele"] or "").strip().upper()
            if not (np.isfinite(beta) and np.isfinite(se) and se > 0.0 and
                    np.isfinite(n) and n > 0.0):
                continue
            if effect == design["counted_allele"][idx] and other == design["other_allele"][idx]:
                sign = 1.0
            elif effect == design["other_allele"][idx] and other == design["counted_allele"][idx]:
                sign = -1.0
            else:
                continue
            found[idx] = (sign * beta / se, n)
    indices = np.asarray(sorted(found), dtype=np.int64)
    z = np.asarray([found[i][0] for i in indices], dtype=np.float64)
    n = np.asarray([found[i][1] for i in indices], dtype=np.float64)
    return indices, z, n


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _fit_a(z, ii, jj, design, pair_block, quadratic, n_blocks, groups):
    try:
        estimate = estimate_pair_products_from_design(
            z, ii, jj, design, pair_block, quadratic_design=quadratic,
            n_blocks=n_blocks, groups=groups)
        return {"status": "estimated", **_jsonable(estimate)}
    except ValueError as exc:
        return {"status": "declined", "note": str(exc), "proportions": None}


def _fit_b(z, bilinear, variant_block, n, n_blocks, groups):
    try:
        estimate = estimate_bilinear_from_design(
            z, bilinear, variant_block, n_blocks=n_blocks, groups=groups,
            sample_size=n)
        status = "estimated" if estimate["proportions"] is not None else "declined"
        return {"status": status, **_jsonable(estimate)}
    except ValueError as exc:
        return {"status": "declined", "signal_note": str(exc), "proportions": None}


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return {
        name: float(value) for name, value in zip(
            ("q05", "median", "q95"),
            np.quantile(values, (0.05, 0.5, 0.95)), strict=True)
    }


def sign_flip_diagnostic(
        study, z, ii, jj, pair_design, pair_block, quadratic_design,
        n_blocks, populations, observed_proportions, observed_scale):
    """Repeated sign randomization; descriptive, not a calibrated model test."""
    seed = int.from_bytes(hashlib.sha256(
        ("ppb-ld-sign-flip-v2:" + study.accession).encode()
    ).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    null_groups = [np.arange(n_blocks, dtype=int)]  # no null jackknife
    expected = study.expected_superpopulation
    expected_index = (None if expected is None else
                      list(populations).index(expected))
    observed_contrast = observed_scaled_contrast = None
    if expected_index is not None and observed_proportions is not None:
        observed = np.asarray(observed_proportions, dtype=np.float64)
        observed_contrast = float(
            observed[expected_index] -
            np.max(np.delete(observed, expected_index))
        )
        observed_scaled_contrast = float(observed_scale * observed_contrast)
    records, scales, max_weights = [], [], []
    null_contrasts, null_scaled_contrasts = [], []
    for _ in range(SIGN_FLIP_REPLICATES):
        flipped = z * rng.choice(np.asarray([-1.0, 1.0]), size=z.size)
        fit = _fit_a(
            flipped, ii, jj, pair_design, pair_block, quadratic_design,
            n_blocks, null_groups)
        proportions = fit.get("proportions")
        if proportions is None:
            records.append({"status": "declined", "top_population": None,
                            "scale": None, "expected_contrast": None,
                            "scaled_expected_contrast": None})
            continue
        proportions = np.asarray(proportions, dtype=np.float64)
        scale = float(fit["scale"])
        top = str(populations[int(np.argmax(proportions))])
        contrast = (None if expected_index is None else float(
            proportions[expected_index] -
            np.max(np.delete(proportions, expected_index))))
        scaled_contrast = None if contrast is None else scale * contrast
        records.append({"status": "estimated", "top_population": top,
                        "scale": scale, "expected_contrast": contrast,
                        "scaled_expected_contrast": scaled_contrast})
        scales.append(scale)
        max_weights.append(float(proportions.max()))
        if contrast is not None:
            null_contrasts.append(contrast)
            null_scaled_contrasts.append(scaled_contrast)
    n_estimated = len(scales)
    normalized_p = scaled_p = None
    if observed_contrast is not None:
        # Declined fits contribute -infinity and cannot exceed the observed
        # correct-label contrast. The +1 correction includes the observation.
        normalized_p = float(
            (1 + np.sum(np.asarray(null_contrasts) >= observed_contrast)) /
            (SIGN_FLIP_REPLICATES + 1)
        )
        scaled_p = float(
            (1 + np.sum(
                np.asarray(null_scaled_contrasts) >= observed_scaled_contrast
            )) / (SIGN_FLIP_REPLICATES + 1)
        )
    return {
        "seed": seed,
        "n_replicates": SIGN_FLIP_REPLICATES,
        "included_in_primary_verdict": False,
        "rule": (
            "independently randomize every harmonized z-score sign; compare "
            "the expected-reference coefficient minus the largest competing "
            "coefficient; also report the normalized-weight contrast, which "
            "is unstable when the fitted linear scale is near zero"
        ),
        "observed_expected_contrast": observed_contrast,
        "observed_scaled_expected_contrast": observed_scaled_contrast,
        "scaled_contrast_empirical_p": scaled_p,
        "normalized_contrast_empirical_p": normalized_p,
        "descriptive_p_threshold": 0.05,
        "descriptive_threshold_passed": (
            None if scaled_p is None else scaled_p <= 0.05),
        "n_estimated": n_estimated,
        "estimated_fraction": n_estimated / SIGN_FLIP_REPLICATES,
        "scale_quantiles_among_estimated": _quantiles(scales),
        "max_weight_quantiles_among_estimated": _quantiles(max_weights),
        "replicates": records,
    }


def benchmark_study(study, design, path: Path, acquisition: dict):
    indices, z, n = read_harmonized_sumstats(path, design)
    inverse = np.full(design["ids"].size, -1, dtype=np.int64)
    inverse[indices] = np.arange(indices.size)
    keep_pair = ((inverse[design["pair_i"]] >= 0) &
                 (inverse[design["pair_j"]] >= 0))
    ii = inverse[design["pair_i"][keep_pair]]
    jj = inverse[design["pair_j"][keep_pair]]
    block_chrom = np.asarray(design["block_chrom"]).astype(str)
    groups = [
        np.flatnonzero(block_chrom == chromosome)
        for chromosome in dict.fromkeys(block_chrom.tolist())
    ]
    a = _fit_a(
        z, ii, jj, design["pair_design"][keep_pair],
        design["pair_block"][keep_pair], design["quadratic_design"][keep_pair],
        design["n_blocks"], groups)
    b = _fit_b(
        z, design["bilinear_design"][indices],
        design["variant_block"][indices], n, design["n_blocks"], groups)

    negative = sign_flip_diagnostic(
        study, z, ii, jj, design["pair_design"][keep_pair],
        design["pair_block"][keep_pair], design["quadratic_design"][keep_pair],
        design["n_blocks"], design["populations"], a["proportions"],
        a.get("scale"))

    expected = study.expected_superpopulation
    expected_rank = None
    if expected is not None and a["proportions"] is not None:
        weights = dict(zip(design["populations"], a["proportions"], strict=True))
        target = weights[expected]
        expected_rank = 1 + sum(value > target + 1e-12 for value in weights.values())
    passed = expected is None or (a["status"] == "estimated" and expected_rank == 1)
    return {
        "study": {
            **asdict(study), "trait": "height", "publication": HEIGHT_PAPER,
            "publication_doi": HEIGHT_PAPER_DOI,
            "publication_pmid": HEIGHT_PAPER_PMID, "url": study.url,
        },
        "input": {
            "normalized_filename": study.filename,
            "n_harmonized_variants": int(indices.size),
            "n_available_pairs": int(keep_pair.sum()), **acquisition,
        },
        "estimator_a": a,
        "estimator_b_exploratory": b,
        "independent_sign_flip_diagnostic": negative,
        "predeclared_control": {
            "included_in_verdict": expected is not None,
            "expected_top_population": expected,
            "expected_rank": expected_rank,
            "passed": bool(passed),
        },
    }


def _git_commit(repo):
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={Path(repo).resolve()}", "-C",
             str(repo), "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_benchmark(studies, design, inputs):
    rows = [benchmark_study(
        study, design, inputs[study.key][0], inputs[study.key][1])
        for study in studies]
    controls = [row for row in rows if row["predeclared_control"]["included_in_verdict"]]
    failures = [row["study"]["accession"] for row in controls
                if not row["predeclared_control"]["passed"]]
    scaled_flip_passes = [
        row["independent_sign_flip_diagnostic"]["descriptive_threshold_passed"]
        is True
        for row in controls
    ]
    normalized_flip_passes = [
        p is not None and p <= 0.05
        for p in (
            row["independent_sign_flip_diagnostic"]
            ["normalized_contrast_empirical_p"]
            for row in controls)
    ]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimand": "best-fitting 1000 Genomes LD-covariance mixture",
        "interpretation": (
            "Estimator A projects summary-statistic pair moments onto fixed "
            "1000 Genomes superpopulation LD covariances. These weights are "
            "not participant ancestry fractions or meta-analysis weights."
        ),
        "benchmark": {
            "name": "Yengo 2022 ancestry-stratified height LD-moment benchmark",
            "verdict_rule": (
                "Estimator A must estimate and rank the predeclared matching "
                "1000G superpopulation first in all five ancestry-specific "
                "studies. Pooled and Estimator B are excluded."
            ),
            "verdict_scope": (
                "qualitative label-ranking control only; not quantitative "
                "calibration or participant-ancestry validation"),
            "numeric_thresholds_tuned_on_these_data": False,
            "estimator_b_role": "exploratory; excluded from verdict",
            "sign_flip_role": (
                f"{SIGN_FLIP_REPLICATES} deterministic negative replicates; "
                "the standard p<=0.05 unnormalized-coefficient contrast "
                "threshold is descriptive and excluded from the primary "
                "verdict; normalized-weight contrast p-values are retained"),
        },
        "design": {
            "file_sha256": design["file_sha256"],
            "semantic_sha256": design["semantic_sha256"],
            "n_variants": int(design["ids"].size),
            "n_pairs": int(design["pair_i"].size),
            "n_blocks": int(design["n_blocks"]),
            "populations": design["populations"].tolist(),
            "sample_counts": design["sample_counts"].astype(int).tolist(),
            "selection": _jsonable(design["selection"]),
            "selection_interpretation": (
                "Common in every reference population, evenly subsampled "
                "within selected blocks, and restricted to mutually distant "
                "semi-independent LD blocks."
            ),
            "provenance": _jsonable(design["provenance"]),
            "extra_scalars": _jsonable(design["extra_scalars"]),
        },
        "caveats": [
            "Yengo studies are adjusted meta-analyses; their estimation-noise "
            "LD need not equal raw 1000G LD.",
            "The pruned, distant-block target is a deliberate tractability "
            "estimand. Its quadratic absorber omits tagging through discarded "
            "variants, so A is chiefly a noise-LD compatibility stress test "
            "and B declines are not a fair genome-wide scale-up test.",
            "Pairs are selected and estimated in the same finite 1000G panels; "
            "reference winner's-curse and errors-in-variables uncertainty are "
            "not included in the chromosome jackknife.",
            "The block jackknife conditions on a finite reference panel and "
            "does not include reference-LD estimation uncertainty.",
        ],
        "software": {
            "ppb_version": ppb.__version__, "ppb_commit": _git_commit(ROOT),
            "python": platform.python_version(), "numpy": np.__version__,
            "ldpred3_acquisition_revision": LDPRED3_REVISION,
        },
        "studies": rows,
        "verdict": {
            "n_controls": len(controls),
            "n_passed": len(controls) - len(failures),
            "failed_accessions": failures, "passed": not failures,
        },
        "sign_flip_diagnostic_verdict": {
            "included_in_primary_verdict": False,
            "n_controls": len(controls),
            "n_scaled_contrast_passed": int(sum(scaled_flip_passes)),
            "all_scaled_contrasts_passed": bool(all(scaled_flip_passes)),
            "n_normalized_contrast_passed": int(sum(normalized_flip_passes)),
            "all_normalized_contrasts_passed": bool(
                all(normalized_flip_passes)),
            "interpretation": (
                "The scaled contrast tests whether a non-negligible fitted "
                "LD-covariance component points to the declared reference. "
                "The normalized contrast exposes instability of simplex "
                "weights when that fitted component is near zero."
            ),
        },
    }


def print_table(result):
    pops = result["design"]["populations"]
    print("Table 1. Predeclared LD-moment controls and exploratory Estimator B")
    print()
    headings = ["Study", "Expected", "A status", "Variants", "Pairs", *pops,
                "A rank", "B status", "Null estimated", "Flip p(s)",
                "Flip p(pi)"]
    print("| " + " | ".join(headings) + " |")
    print("|" + "|".join(["---"] * 3 + ["---:"] * (len(headings) - 3)) + "|")
    for row in result["studies"]:
        a = row["estimator_a"]
        weights = (dict(zip(pops, a["proportions"], strict=True))
                   if a["proportions"] is not None else {})
        cells = [
            f"{row['study']['key']} ({row['study']['accession']})",
            row["study"]["expected_superpopulation"] or "descriptive",
            a["status"], f"{row['input']['n_harmonized_variants']:,}",
            f"{row['input']['n_available_pairs']:,}",
            *["-" if pop not in weights else f"{weights[pop]:.3f}" for pop in pops],
            "-" if row["predeclared_control"]["expected_rank"] is None else
            str(row["predeclared_control"]["expected_rank"]),
            row["estimator_b_exploratory"]["status"],
            (f"{row['independent_sign_flip_diagnostic']['n_estimated']}/"
             f"{row['independent_sign_flip_diagnostic']['n_replicates']}"),
            ("-" if row["independent_sign_flip_diagnostic"]
             ["scaled_contrast_empirical_p"] is None else
             f"{row['independent_sign_flip_diagnostic']['scaled_contrast_empirical_p']:.4f}"),
            ("-" if row["independent_sign_flip_diagnostic"]
             ["normalized_contrast_empirical_p"] is None else
             f"{row['independent_sign_flip_diagnostic']['normalized_contrast_empirical_p']:.4f}"),
        ]
        print("| " + " | ".join(cells) + " |")
    verdict = result["verdict"]
    print(f"\nQualitative benchmark-rule verdict: "
          f"{'PASS' if verdict['passed'] else 'FAIL'} "
          f"({verdict['n_passed']}/{verdict['n_controls']} controls passed).")
    diagnostic = result["sign_flip_diagnostic_verdict"]
    print(
        "Sign-flip diagnostic: "
        f"scaled contrast {diagnostic['n_scaled_contrast_passed']}/"
        f"{diagnostic['n_controls']}; normalized contrast "
        f"{diagnostic['n_normalized_contrast_passed']}/"
        f"{diagnostic['n_controls']} at descriptive p<=0.05."
    )


def _selected(tokens):
    lookup = {token.upper(): study for study in STUDIES
              for token in (study.key, study.accession)}
    if not tokens:
        return list(STUDIES)
    unknown = [token for token in tokens if token.upper() not in lookup]
    if unknown:
        raise ValueError(f"unknown studies: {', '.join(unknown)}")
    return list(dict.fromkeys(lookup[token.upper()] for token in tokens))


def _write_json_atomic(path: Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument(
        "--design-sha256",
        help="expected file SHA-256; the bundled default path is pinned automatically")
    parser.add_argument("--sumstats-dir", type=Path, default=DEFAULT_SUMSTATS_DIR)
    parser.add_argument("--ldpred3-repo", type=Path, default=DEFAULT_LDPRED3_REPO)
    parser.add_argument(
        "--af-panel", type=Path, default=DEFAULT_PANEL,
        help="full HM3 AF panel used only by --fetch/--raw-dir acquisition")
    parser.add_argument("--af-panel-sha256", default=PANEL_SHA256)
    parser.add_argument("--studies", nargs="*", metavar="KEY")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fetch", action="store_true")
    source.add_argument("--raw-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    try:
        studies = _selected(args.studies)
        if args.list:
            print("Table 1. Fixed GWAS Catalog LD benchmark cohort\n")
            print("| Key | Accession | Reported sample | N | Expected |")
            print("|---|---|---|---:|---|")
            for study in studies:
                print(f"| {study.key} | {study.accession} | {study.reported_sample} | "
                      f"{study.reported_n:,} | "
                      f"{study.expected_superpopulation or 'descriptive'} |")
            return 0
        expected_design_sha256 = args.design_sha256
        if (expected_design_sha256 is None and
                args.design.resolve() == DEFAULT_DESIGN.resolve()):
            expected_design_sha256 = DEFAULT_DESIGN_SHA256
        if expected_design_sha256 is None:
            parser.error(
                "--design-sha256 is required unless --design is the bundled "
                "default (an unpinned custom design is unsupported)")
        design = load_ld_design(args.design, expected_design_sha256)
        if args.fetch or args.raw_dir:
            acquisition_panel = load_frequency_panel(
                args.af_panel, expected_sha256=args.af_panel_sha256)
            ids = set(acquisition_panel.ids.tolist())
        else:
            ids = None
        inputs = {}
        for study in studies:
            path, provenance = acquire_study(
                study, panel_ids=ids, sumstats_dir=args.sumstats_dir,
                ldpred3_repo=args.ldpred3_repo, fetch=args.fetch,
                raw_dir=args.raw_dir)
            if provenance["mode"] == "missing":
                raise FileNotFoundError(
                    f"{study.accession}: {path} is absent; use --fetch or --raw-dir")
            inputs[study.key] = path, provenance
        result = run_benchmark(studies, design, inputs)
        print_table(result)
        if args.out:
            _write_json_atomic(args.out, result)
            print(f"\nSnapshot: {args.out}")
        return 0 if result["verdict"]["passed"] else 1
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
