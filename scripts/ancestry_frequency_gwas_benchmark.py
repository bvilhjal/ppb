#!/usr/bin/env python
"""Benchmark EAF-profile ancestry decomposition on a matched GWAS family.

The six studies are ancestry-stratified height analyses from the same Yengo
et al. publication.  Five are predeclared qualitative positive controls: the
reported ancestry's corresponding 1000 Genomes superpopulation must receive
the largest projection weight.  The pooled analysis is descriptive because
participant counts are not the variant-specific weights represented by a
deposited EAF profile.

This is deliberately an EAF-profile benchmark, not participant-ancestry
validation.  GWAS meta-analysis, per-variant missingness, ascertainment, drift,
and the provenance of deposited frequencies can separate those estimands.

Acquisition is optional.  ``--fetch`` or ``--raw-dir`` dynamically loads the
source-checkout-only LDpred3 GWAS Catalog harvester at one pinned revision.
Ordinary benchmark runs consume hash-pinned normalized files and do not import
LDpred3 or access the network.

Examples::

    python scripts/ancestry_frequency_gwas_benchmark.py --list
    python scripts/ancestry_frequency_gwas_benchmark.py --fetch
    python scripts/ancestry_frequency_gwas_benchmark.py --raw-dir ../.work/gwas-candidates
    python scripts/ancestry_frequency_gwas_benchmark.py --out result.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppb  # noqa: E402
from ppb import (  # noqa: E402
    decompose_effect_allele_frequencies,
    load_frequency_panel,
)


CATALOG_BASE = (
    "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics")
HEIGHT_PAPER = "Yengo et al., Nature 2022"
HEIGHT_PAPER_DOI = "10.1038/s41586-022-05275-y"
HEIGHT_PAPER_PMID = "36224396"
LDPRED3_REVISION = "621a2c4dc10a225fd7abcddebc66fbdee33c9cf0"
PANEL_SHA256 = "c91b6652671ceb373417e9d82c47db5048964923bf448801214e1f78b7291186"
DEFAULT_PANEL = (
    ROOT.parent / "reference-data" / "1000g-phase3" / "panels"
    / "kgp_phase3_20130502_superpops_hm3_grch37.npz")
DEFAULT_SUMSTATS_DIR = ROOT / ".work" / "ancestry-frequency-gwas"
DEFAULT_LDPRED3_REPO = ROOT.parent / "ldpred3"


@dataclass(frozen=True)
class CatalogStudy:
    """One immutable GWAS Catalog benchmark input."""

    key: str
    accession: str
    reported_sample: str
    reported_n: int
    expected_superpopulation: str | None
    expectation_basis: str
    ftp_relative_path: str
    remote_bytes: int
    source_sha256: str
    normalized_content_sha256: str

    @property
    def url(self) -> str:
        return f"{CATALOG_BASE}/{self.ftp_relative_path}"

    @property
    def filename(self) -> str:
        return f"{self.accession}.hm3.tsv.gz"


def _study(
    key,
    accession,
    reported_sample,
    reported_n,
    expected_superpopulation,
    expectation_basis,
    remote_bytes,
    source_sha256,
    normalized_content_sha256,
):
    group = "GCST90245001-GCST90246000"
    relative = f"{group}/{accession}/harmonised/{accession}.h.tsv.gz"
    return CatalogStudy(
        key=key,
        accession=accession,
        reported_sample=reported_sample,
        reported_n=reported_n,
        expected_superpopulation=expected_superpopulation,
        expectation_basis=expectation_basis,
        ftp_relative_path=relative,
        remote_bytes=remote_bytes,
        source_sha256=source_sha256,
        normalized_content_sha256=normalized_content_sha256,
    )


# Source hashes are over the official compressed bytes.  Normalized hashes are
# over decompressed LDpred3 output, which is stable across gzip timestamps.
STUDIES = (
    _study(
        "AFR", "GCST90245989", "African ancestry", 168_193, "AFR",
        "same-paper ancestry-specific analysis; AFR is the predeclared top rank",
        37_029_018,
        "c28fe7ecf786e1e5f3e0d658ce2d9a61d409f24b31aa429962b794afb6736a71",
        "80dcbb368e2295dc6ee37941784ced2175072d89ea256b16039e9d20a7a69d9a",
    ),
    _study(
        "AMR", "GCST90245993", "Hispanic or Latin American", 58_709, "AMR",
        "predeclared proxy: the Catalog label maps imperfectly to 1000G AMR",
        36_760_742,
        "c83e01128edac04f7b680b61428fc5497070de4be8366c9618eadac2dd7658b4",
        "921ce260f3be4465f2f3692a19a865dd06e9bd9cd7a24b0dda3d0749c347a385",
    ),
    _study(
        "EAS", "GCST90245991", "East Asian ancestry", 363_856, "EAS",
        "same-paper ancestry-specific analysis; EAS is the predeclared top rank",
        35_940_719,
        "dec82cdd1bb2ae4d0cfd695e6c57e51a02583e2a50a60ab487a22c12ae422195",
        "520f86b7795a8ad8301ef978f974c82d7ac6368f9eb7c2f0bf48d1743b53a494",
    ),
    _study(
        "EUR", "GCST90245992", "European ancestry", 1_597_374, "EUR",
        "same-paper ancestry-specific analysis; EUR is the predeclared top rank",
        39_622_810,
        "c771e1bb1ed2e859e487a8a63880f694e30e9e771843b2da3373dedbbea8c1a1",
        "eea3f4e81bc992b63725240448fb5557330d8e5864ca8f3b49115f24c2c78706",
    ),
    _study(
        "SAS", "GCST90245994", "South Asian ancestry", 60_939, "SAS",
        "same-paper ancestry-specific analysis; SAS is the predeclared top rank",
        35_445_178,
        "e54809498df59e218c6b252d1346be527e20adb1e1b20ee69396aa587cba80b3",
        "c8cee189a1ffb1feb9427e03693db9e797caad38344aa9b1143a31867dc07e09",
    ),
    _study(
        "POOLED", "GCST90245990", "five-ancestry pooled meta-analysis",
        2_200_007, None,
        "descriptive stress test; headcounts are not an EAF-mixture truth target",
        39_600_155,
        "9e807eaef46b6e1af06710defd64a8276689950ed0f73a3cdea682f6e0fe3e88",
        "43dd8f1ad7efa7d7a16dc16f3e5bb773f3d2c510982cba0570de917ddcf207b4",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_content(path: Path) -> str:
    """Hash decompressed bytes, excluding non-semantic gzip metadata."""
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    repo = Path(repo).resolve()
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        check=True,
        capture_output=True, text=True)
    return completed.stdout.strip()


def _load_ldpred3_harvester(repo: Path):
    """Load the private harvester only after verifying its source checkout."""
    repo = Path(repo).resolve()
    revision = _git(repo, "rev-parse", "HEAD")
    if revision != LDPRED3_REVISION:
        raise ValueError(
            f"LDpred3 checkout is {revision}, expected {LDPRED3_REVISION}")
    protected = (
        "benchmarks/gwas_catalog_harvest.py",
        "benchmarks/_paths.py",
    )
    dirty = _git(repo, "status", "--porcelain", "--", *protected)
    if dirty:
        raise ValueError(
            "LDpred3 acquisition code differs from the pinned commit:\n" + dirty)
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("benchmarks.gwas_catalog_harvest")
    finally:
        try:
            sys.path.remove(str(repo))
        except ValueError:
            pass
    source = Path(module.__file__).resolve()
    if repo not in source.parents:
        raise ValueError(f"loaded GWAS harvester from unexpected path {source}")
    return module


def _verify_normalized(study: CatalogStudy, path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"{study.accession}: missing normalized input {path}; use --fetch "
            "or --raw-dir")
    observed = sha256_gzip_content(path)
    if observed != study.normalized_content_sha256:
        raise ValueError(
            f"{study.accession}: normalized content hash {observed} does not "
            f"match pinned {study.normalized_content_sha256}")
    return observed


def acquire_study(
    study: CatalogStudy,
    *,
    panel_ids,
    sumstats_dir: Path,
    ldpred3_repo: Path,
    fetch: bool = False,
    raw_dir: Path | None = None,
) -> tuple[Path, dict]:
    """Create or verify one LDpred3-normalized, HM3-filtered input."""
    sumstats_dir = Path(sumstats_dir)
    output = sumstats_dir / study.filename
    if output.exists():
        content_hash = _verify_normalized(study, output)
        return output, {
            "mode": "verified_cache",
            "raw_source_sha256_verified_this_run": False,
            "normalized_content_sha256": content_hash,
        }
    if not fetch and raw_dir is None:
        return output, {
            "mode": "missing",
            "raw_source_sha256_verified_this_run": False,
            "normalized_content_sha256": None,
        }

    harvester = _load_ldpred3_harvester(ldpred3_repo)
    sumstats_dir.mkdir(parents=True, exist_ok=True)
    candidate = {
        "accession": study.accession,
        "path": study.ftp_relative_path,
    }
    created = False
    try:
        if raw_dir is not None:
            raw = Path(raw_dir) / f"{study.accession}.h.tsv.gz"
            if not raw.is_file():
                raise FileNotFoundError(f"{study.accession}: missing raw input {raw}")
            raw_hash = sha256_file(raw)
            if raw_hash != study.source_sha256:
                raise ValueError(
                    f"{study.accession}: raw source hash {raw_hash} does not "
                    f"match pinned {study.source_sha256}")
            made = harvester.ingest_local(
                candidate, panel_ids, str(sumstats_dir), str(raw))
            mode = "ldpred3_ingest_local"
        else:
            made = harvester.stream_filter(
                candidate, panel_ids, str(sumstats_dir))
            raw_hash = candidate.get("sha256")
            if raw_hash != study.source_sha256:
                raise ValueError(
                    f"{study.accession}: downloaded source hash {raw_hash} "
                    f"does not match pinned {study.source_sha256}")
            mode = "ldpred3_stream_filter"
        output = Path(made)
        created = True
        content_hash = _verify_normalized(study, output)
    except Exception:
        if created or output.exists():
            output.unlink(missing_ok=True)
        raise
    return output, {
        "mode": mode,
        "raw_source_sha256_verified_this_run": True,
        "normalized_content_sha256": content_hash,
    }


def _parse_eaf(value: str) -> float:
    text = (value or "").strip()
    if not text or text.lower() in {"na", "nan", "null", "."}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_harvested_eaf(path: Path):
    """Read only the four fields used by PPB from LDpred3-normalized output."""
    ids, effect, other, eaf = [], [], [], []
    with gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="") as src:
        reader = csv.DictReader(src, delimiter="\t")
        required = {"rsid", "effect_allele", "other_allele", "eaf"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path}: missing LDpred3-normalized columns {sorted(missing)}")
        for row in reader:
            ids.append((row["rsid"] or "").strip())
            effect.append((row["effect_allele"] or "").strip())
            other.append((row["other_allele"] or "").strip())
            eaf.append(_parse_eaf(row["eaf"]))
    return (
        np.asarray(ids),
        np.asarray(effect),
        np.asarray(other),
        np.asarray(eaf, dtype=np.float64),
    )


def benchmark_study(study: CatalogStudy, panel, path: Path, acquisition: dict):
    started = time.perf_counter()
    ids, effect, other, eaf = read_harvested_eaf(path)
    decomposition = decompose_effect_allele_frequencies(
        ids, effect, other, eaf, panel)
    expected = study.expected_superpopulation
    expected_rank = expected_weight = expected_top = None
    n_tied_at_top = None
    if expected is not None and decomposition["proportions"] is not None:
        weights = dict(zip(
            decomposition["populations"], decomposition["proportions"],
            strict=True))
        expected_weight = weights[expected]
        expected_rank = 1 + sum(
            value > expected_weight + 1e-12 for value in weights.values())
        n_tied_at_top = sum(
            abs(value - expected_weight) <= 1e-12
            for value in weights.values())
        # "Rank first" means a unique argmax: an exact tie at the top is not
        # a passed control.
        expected_top = expected_rank == 1 and n_tied_at_top == 1
    control_passed = (
        None if expected is None
        else (decomposition["status"] == "estimated" and expected_top is True)
    )
    return {
        "study": {
            **asdict(study),
            "trait": "height",
            "publication": HEIGHT_PAPER,
            "publication_doi": HEIGHT_PAPER_DOI,
            "publication_pmid": HEIGHT_PAPER_PMID,
            "url": study.url,
        },
        "input": {
            "normalized_filename": study.filename,
            "n_rows": int(len(ids)),
            **acquisition,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "decomposition": decomposition,
        "predeclared_control": {
            "included_in_verdict": expected is not None,
            "expected_top_population": expected,
            "expected_weight": expected_weight,
            "expected_rank": expected_rank,
            "n_tied_at_top": n_tied_at_top,
            # None (not True) for the descriptive-only studies: no control
            # was evaluated, so the field must not assert a pass.
            "passed": control_passed,
        },
    }


def _commit_or_unknown(repo: Path) -> str:
    try:
        return _git(repo, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_benchmark(studies, panel, inputs):
    rows = [
        benchmark_study(study, panel, inputs[study.key][0], inputs[study.key][1])
        for study in studies
    ]
    controls = [
        row for row in rows
        if row["predeclared_control"]["included_in_verdict"]
    ]
    failures = [
        row["study"]["accession"] for row in controls
        if not row["predeclared_control"]["passed"]
    ]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimand": "equal-marker EAF-profile projection weights",
        "interpretation": (
            "Projection of deposited effect-allele frequencies onto the "
            "fixed 1000 Genomes superpopulation panel; not automatically "
            "participant-ancestry fractions or meta-analysis weights."),
        "benchmark": {
            "name": "Yengo 2022 ancestry-stratified height EAF benchmark",
            "publication": HEIGHT_PAPER,
            "doi": HEIGHT_PAPER_DOI,
            "pmid": HEIGHT_PAPER_PMID,
            "verdict_rule": (
                "Each of the five ancestry-specific analyses must return "
                "status=estimated and rank its predeclared corresponding "
                "1000G superpopulation first. The pooled analysis is excluded."),
            "numeric_thresholds_tuned_on_these_data": False,
        },
        "panel": {
            "source": panel.source,
            "source_url": panel.source_url,
            "genome_build": panel.genome_build,
            "semantic_sha256": panel.panel_sha256,
            "n_variants": len(panel),
            "populations": list(panel.pops),
            "n_samples": [int(value) for value in panel.n_samples],
        },
        "software": {
            "ppb_version": ppb.__version__,
            "ppb_commit": _commit_or_unknown(ROOT),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ldpred3_acquisition_revision": LDPRED3_REVISION,
            "ldpred3_component": "benchmarks/gwas_catalog_harvest.py",
        },
        "studies": rows,
        "verdict": {
            "n_controls": len(controls),
            "n_passed": len(controls) - len(failures),
            "failed_accessions": failures,
            # A control-free selection must not report a vacuous PASS with
            # exit code 0 (e.g. --studies POOLED, which is descriptive-only
            # by construction and can never be a control).
            "passed": bool(controls) and not failures,
        },
    }


def print_table(result):
    pops = result["panel"]["populations"]
    print("Table 1. Predeclared ancestry-specific controls and pooled stress test")
    print()
    headings = [
        "Study", "Expected", "Status", "Matched", *pops,
        "AF r", "RMS", "Rank",
    ]
    print("| " + " | ".join(headings) + " |")
    print("|" + "|".join(["---"] * 3 + ["---:"] * (len(headings) - 3)) + "|")
    for row in result["studies"]:
        study = row["study"]
        dec = row["decomposition"]
        control = row["predeclared_control"]
        weights = (
            dict(zip(dec["populations"], dec["proportions"], strict=True))
            if dec["proportions"] is not None else {}
        )
        weight_cells = [
            "-" if pop not in weights else f"{weights[pop]:.3f}" for pop in pops
        ]
        rank = control["expected_rank"]
        cells = [
            f"{study['key']} ({study['accession']})",
            study["expected_superpopulation"] or "descriptive",
            dec["status"],
            f"{dec['matching']['n_matched']:,}",
            *weight_cells,
            "-" if dec["af_corr"] is None else f"{dec['af_corr']:.4f}",
            "-" if dec["residual_rms"] is None else f"{dec['residual_rms']:.4f}",
            "-" if rank is None else str(rank),
        ]
        print("| " + " | ".join(cells) + " |")
    verdict = result["verdict"]
    label = "PASS" if verdict["passed"] else "FAIL"
    print(
        f"\nVerdict: {label} ({verdict['n_passed']}/{verdict['n_controls']} "
        "predeclared controls passed).")


def _selected(tokens):
    by_name = {
        token.upper(): study
        for study in STUDIES
        for token in (study.key, study.accession)
    }
    if not tokens:
        return list(STUDIES)
    unknown = [token for token in tokens if token.upper() not in by_name]
    if unknown:
        raise ValueError(f"unknown studies: {', '.join(unknown)}")
    selected = []
    seen = set()
    for token in tokens:
        study = by_name[token.upper()]
        if study.accession not in seen:
            selected.append(study)
            seen.add(study.accession)
    return selected


def _write_json_atomic(path: Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--panel-sha256", default=PANEL_SHA256)
    parser.add_argument("--sumstats-dir", type=Path, default=DEFAULT_SUMSTATS_DIR)
    parser.add_argument("--ldpred3-repo", type=Path, default=DEFAULT_LDPRED3_REPO)
    parser.add_argument(
        "--studies", nargs="*", metavar="KEY", help="keys or accessions; default: all")
    acquisition = parser.add_mutually_exclusive_group()
    acquisition.add_argument(
        "--fetch", action="store_true",
        help="download and normalize with the pinned LDpred3 source checkout")
    acquisition.add_argument(
        "--raw-dir", type=Path,
        help="normalize pinned <accession>.h.tsv.gz files already in this directory")
    parser.add_argument("--out", type=Path, help="write a strict JSON snapshot")
    parser.add_argument(
        "--allow-partial-verdict", action="store_true",
        help="let a snapshot record fewer than five predeclared controls")
    parser.add_argument("--list", action="store_true", help="list the fixed cohort and exit")
    args = parser.parse_args(argv)

    try:
        studies = _selected(args.studies)
        if args.list:
            print("Table 1. Fixed GWAS Catalog benchmark cohort\n")
            print("| Key | Accession | Reported sample | N | Expected |")
            print("|---|---|---|---:|---|")
            for study in studies:
                expected = study.expected_superpopulation or "descriptive"
                if "imperfectly" in study.expectation_basis:
                    expected += " (imperfect label proxy)"
                print(
                    f"| {study.key} | {study.accession} | "
                    f"{study.reported_sample} | {study.reported_n:,} | "
                    f"{expected} |")
            return 0

        panel = load_frequency_panel(
            args.panel, expected_sha256=args.panel_sha256)
        panel_ids = set(panel.ids.tolist()) if (args.fetch or args.raw_dir) else None
        inputs = {}
        for study in studies:
            path, provenance = acquire_study(
                study,
                panel_ids=panel_ids,
                sumstats_dir=args.sumstats_dir,
                ldpred3_repo=args.ldpred3_repo,
                fetch=args.fetch,
                raw_dir=args.raw_dir,
            )
            if provenance["mode"] == "missing":
                raise FileNotFoundError(
                    f"{study.accession}: {path} is absent; use --fetch or --raw-dir")
            inputs[study.key] = (path, provenance)
        result = run_benchmark(studies, panel, inputs)
        print_table(result)
        if args.out:
            n_controls = result["verdict"]["n_controls"]
            if n_controls < 5 and not args.allow_partial_verdict:
                raise ValueError(
                    f"refusing to write a snapshot with only {n_controls} "
                    "predeclared controls selected (need 5); rerun with "
                    "--allow-partial-verdict to record a partial verdict")
            _write_json_atomic(args.out, result)
            print(f"\nSnapshot: {args.out}")
        return 0 if result["verdict"]["passed"] else 1
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
