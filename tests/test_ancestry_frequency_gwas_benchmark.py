"""Offline contracts for the real-GWAS EAF-profile benchmark."""

from __future__ import annotations

import gzip
import json
import string
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ppb import load_frequency_panel, write_frequency_panel
from scripts import ancestry_frequency_gwas_benchmark as benchmark


_EXPECTED_STUDIES = {
    "AFR": ("GCST90245989", "AFR"),
    "AMR": ("GCST90245993", "AMR"),
    "EAS": ("GCST90245991", "EAS"),
    "EUR": ("GCST90245992", "EUR"),
    "SAS": ("GCST90245994", "SAS"),
    "POOLED": ("GCST90245990", None),
}


def _write_normalized(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("rsid\teffect_allele\tother_allele\teaf\n")
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")
    return path


def _study_for(path: Path, *, expected="AFR"):
    content_hash = benchmark.sha256_gzip_content(path)
    return replace(
        benchmark.STUDIES[0],
        key="SYNTHETIC",
        accession="GCST_SYNTHETIC",
        reported_sample="synthetic test population",
        reported_n=10_000,
        expected_superpopulation=expected,
        source_sha256="1" * 64,
        normalized_content_sha256=content_hash,
    )


def test_manifest_has_five_predeclared_controls_and_one_descriptive_stress_test():
    assert len(benchmark.STUDIES) == 6
    observed = {
        study.key: (study.accession, study.expected_superpopulation)
        for study in benchmark.STUDIES
    }
    assert observed == _EXPECTED_STUDIES
    assert len({study.accession for study in benchmark.STUDIES}) == 6

    controls = [
        study for study in benchmark.STUDIES
        if study.expected_superpopulation is not None
    ]
    assert {study.expected_superpopulation for study in controls} == {
        "AFR", "AMR", "EAS", "EUR", "SAS",
    }
    assert all("predeclared" in study.expectation_basis for study in controls)
    pooled = next(study for study in benchmark.STUDIES if study.key == "POOLED")
    assert "descriptive" in pooled.expectation_basis

    hexadecimal = set(string.hexdigits.lower())
    for study in benchmark.STUDIES:
        for digest in (
            study.source_sha256, study.normalized_content_sha256,
        ):
            assert len(digest) == 64
            assert set(digest) <= hexadecimal
            assert digest != "0" * 64, (
                f"{study.accession} still has a placeholder content digest")
        assert study.url.startswith(benchmark.CATALOG_BASE + "/")
        assert study.filename == f"{study.accession}.hm3.tsv.gz"


def test_verdict_counts_only_the_five_predeclared_controls(monkeypatch):
    failed = "GCST90245993"

    def fake_benchmark_study(study, panel, path, acquisition):
        del panel, path, acquisition
        included = study.expected_superpopulation is not None
        return {
            "study": {"accession": study.accession},
            "predeclared_control": {
                "included_in_verdict": included,
                "passed": not included or study.accession != failed,
            },
        }

    monkeypatch.setattr(benchmark, "benchmark_study", fake_benchmark_study)
    class Panel:
        source = "test"
        source_url = ""
        genome_build = "GRCh37"
        panel_sha256 = "a" * 64
        pops = ["AFR", "AMR", "EAS", "EUR", "SAS"]
        n_samples = np.ones(5, dtype=int)

        def __len__(self):
            return 1

    panel = Panel()
    inputs = {
        study.key: (Path(study.filename), {"mode": "unused"})
        for study in benchmark.STUDIES
    }

    result = benchmark.run_benchmark(benchmark.STUDIES, panel, inputs)

    assert result["benchmark"]["numeric_thresholds_tuned_on_these_data"] is False
    assert result["verdict"] == {
        "n_controls": 5,
        "n_passed": 4,
        "failed_accessions": [failed],
        "passed": False,
    }


def test_control_free_selection_is_not_a_vacuous_pass(monkeypatch):
    """M10: ``--studies POOLED`` selects no predeclared control at all, so
    the verdict must be False (not vacuously True with exit code 0) and a
    snapshot must be refused without --allow-partial-verdict."""
    pooled = [s for s in benchmark.STUDIES
              if s.expected_superpopulation is None]

    def fake_benchmark_study(study, panel, path, acquisition):
        del panel, path, acquisition
        return {
            "study": {"accession": study.accession},
            "predeclared_control": {
                "included_in_verdict": False,
                # the real code records None, not True, for a descriptive row
                "passed": None,
            },
        }

    monkeypatch.setattr(benchmark, "benchmark_study", fake_benchmark_study)
    class Panel:
        source = "test"
        source_url = ""
        genome_build = "GRCh37"
        panel_sha256 = "a" * 64
        pops = ["AFR", "AMR", "EAS", "EUR", "SAS"]
        n_samples = np.ones(5, dtype=int)

        def __len__(self):
            return 1

    inputs = {study.key: (Path(study.filename), {"mode": "unused"})
              for study in pooled}
    result = benchmark.run_benchmark(pooled, Panel(), inputs)
    assert result["verdict"]["n_controls"] == 0
    assert result["verdict"]["passed"] is False


def test_normalized_reader_preserves_duplicates_and_maps_blank_eaf_to_nan(
        tmp_path):
    path = _write_normalized(tmp_path / "normalized.tsv.gz", [
        ("rs1", "A", "C", "0.25"),
        ("rs1", "A", "C", ""),
        ("rs2", "G", "T", "NA"),
        ("rs3", "A", "G", "."),
        ("rs4", "T", "C", "0.8"),
    ])

    ids, effect, other, eaf = benchmark.read_harvested_eaf(path)

    assert ids.tolist() == ["rs1", "rs1", "rs2", "rs3", "rs4"]
    assert effect.tolist() == ["A", "A", "G", "A", "T"]
    assert other.tolist() == ["C", "C", "T", "G", "C"]
    assert eaf[[0, 4]].tolist() == pytest.approx([0.25, 0.8])
    assert np.isnan(eaf[1:4]).all()


def test_normalized_reader_requires_all_four_ldpred3_columns(tmp_path):
    path = tmp_path / "missing-eaf.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("rsid\teffect_allele\tother_allele\nrs1\tA\tC\n")

    with pytest.raises(ValueError, match="missing LDpred3-normalized columns"):
        benchmark.read_harvested_eaf(path)


def test_synthetic_wrapper_runs_public_ppb_decomposition_end_to_end(tmp_path):
    rng = np.random.default_rng(20260830)
    m = 1_200
    populations = ["AFR", "AMR", "EAS", "EUR", "SAS"]
    ids = np.asarray([f"rs{i}" for i in range(m)])
    chrom = np.asarray([str(i % 10 + 1) for i in range(m)])
    counted = np.full(m, "A")
    other = np.full(m, "C")
    baseline = rng.uniform(0.12, 0.88, m)
    contrasts = rng.normal(0.0, 0.10, (m, len(populations)))
    contrasts -= contrasts.mean(axis=1, keepdims=True)
    af = np.clip(baseline[:, None] + contrasts, 0.01, 0.99)
    panel_path = write_frequency_panel(
        tmp_path / "panel.npz",
        ids=ids,
        chrom=chrom,
        pos=np.arange(1, m + 1),
        counted_allele=counted,
        other_allele=other,
        pops=populations,
        af=af,
        n_samples=[661, 347, 504, 503, 489],
        source="synthetic benchmark panel",
        genome_build="GRCh37",
    )
    panel = load_frequency_panel(panel_path)

    truth = np.asarray([0.60, 0.08, 0.10, 0.14, 0.08])
    target = np.clip(af @ truth + rng.normal(0.0, 0.001, m), 0.0, 1.0)
    swapped = np.arange(m) % 7 == 0
    effect = np.where(swapped, other, counted)
    non_effect = np.where(swapped, counted, other)
    observed = np.where(swapped, 1.0 - target, target)
    rows = list(zip(ids, effect, non_effect, observed, strict=True))
    rows.append(rows[0])
    path = _write_normalized(tmp_path / "GCST_SYNTHETIC.hm3.tsv.gz", rows)
    study = _study_for(path)
    acquisition = {
        "mode": "verified_cache",
        "raw_source_sha256_verified_this_run": False,
        "normalized_content_sha256": study.normalized_content_sha256,
    }

    result = benchmark.run_benchmark(
        [study], panel, {study.key: (path, acquisition)})

    assert result["estimand"] == "equal-marker EAF-profile projection weights"
    assert result["panel"]["semantic_sha256"] == panel.panel_sha256
    assert result["verdict"] == {
        "n_controls": 1,
        "n_passed": 1,
        "failed_accessions": [],
        "passed": True,
    }
    row = result["studies"][0]
    decomposition = row["decomposition"]
    assert decomposition["status"] == "estimated"
    assert decomposition["n_used"] == m
    assert decomposition["matching"]["n_input"] == m + 1
    assert decomposition["matching"]["n_dropped_duplicate"] == 1
    assert decomposition["chromosome_coverage"]["n_distinct_autosomes"] == 10
    assert np.asarray(decomposition["proportions"]) == pytest.approx(
        truth, abs=0.01)
    assert row["predeclared_control"]["expected_rank"] == 1
    assert row["predeclared_control"]["passed"] is True


def test_cached_input_is_verified_and_tampering_is_refused(tmp_path):
    sumstats_dir = tmp_path / "sumstats"
    path = _write_normalized(sumstats_dir / benchmark.STUDIES[0].filename, [
        ("rs1", "A", "C", "0.2"),
    ])
    study = replace(
        benchmark.STUDIES[0],
        normalized_content_sha256=benchmark.sha256_gzip_content(path),
    )

    observed_path, provenance = benchmark.acquire_study(
        study,
        panel_ids=None,
        sumstats_dir=sumstats_dir,
        ldpred3_repo=tmp_path / "unused",
    )
    assert observed_path == path
    assert provenance == {
        "mode": "verified_cache",
        "raw_source_sha256_verified_this_run": False,
        "normalized_content_sha256": study.normalized_content_sha256,
    }

    _write_normalized(path, [("rs1", "A", "C", "0.3")])
    with pytest.raises(ValueError, match="does not match pinned"):
        benchmark.acquire_study(
            study,
            panel_ids=None,
            sumstats_dir=sumstats_dir,
            ldpred3_repo=tmp_path / "unused",
        )


def test_failed_fetch_removes_untrusted_normalized_output(tmp_path, monkeypatch):
    sumstats_dir = tmp_path / "sumstats"
    study = replace(
        benchmark.STUDIES[0],
        source_sha256="a" * 64,
        normalized_content_sha256="b" * 64,
    )

    def fake_stream_filter(candidate, panel_ids, out_dir):
        del panel_ids
        path = _write_normalized(
            Path(out_dir) / study.filename,
            [("rs1", "A", "C", "0.2")],
        )
        candidate["sha256"] = "f" * 64
        return str(path)

    fake_harvester = SimpleNamespace(stream_filter=fake_stream_filter)
    monkeypatch.setattr(
        benchmark, "_load_ldpred3_harvester", lambda repo: fake_harvester)

    with pytest.raises(ValueError, match="downloaded source hash"):
        benchmark.acquire_study(
            study,
            panel_ids={"rs1"},
            sumstats_dir=sumstats_dir,
            ldpred3_repo=tmp_path / "ldpred3",
            fetch=True,
        )
    assert not (sumstats_dir / study.filename).exists()


def test_list_is_runnable_without_repo_cwd_or_network(tmp_path):
    script = Path(benchmark.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--list"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Table 1. Fixed GWAS Catalog benchmark cohort" in completed.stdout
    for study in benchmark.STUDIES:
        assert study.accession in completed.stdout
    assert "descriptive" in completed.stdout


def test_checked_in_snapshot_retains_full_provenance_and_predeclared_verdict():
    path = (
        Path(benchmark.__file__).resolve().parents[1]
        / "results" / "ancestry-frequency" / "yengo-height-2026-08-30.json"
    )
    text = path.read_text(encoding="utf-8")
    snapshot = json.loads(text, parse_constant=lambda value: pytest.fail(
        f"non-finite JSON constant {value}"))

    assert snapshot["schema_version"] == 1
    assert snapshot["estimand"] == "equal-marker EAF-profile projection weights"
    assert snapshot["panel"]["semantic_sha256"] == benchmark.PANEL_SHA256
    assert snapshot["software"]["ldpred3_acquisition_revision"] == (
        benchmark.LDPRED3_REVISION)
    assert len(snapshot["software"]["ppb_commit"]) == 40
    assert snapshot["verdict"] == {
        "n_controls": 5,
        "n_passed": 5,
        "failed_accessions": [],
        "passed": True,
    }
    assert [row["study"]["key"] for row in snapshot["studies"]] == [
        study.key for study in benchmark.STUDIES
    ]
    pins = {study.accession: study for study in benchmark.STUDIES}
    for row in snapshot["studies"]:
        study = pins[row["study"]["accession"]]
        assert row["study"]["source_sha256"] == study.source_sha256
        assert row["input"]["normalized_content_sha256"] == (
            study.normalized_content_sha256)
        assert row["decomposition"]["status"] == "estimated"
        assert row["decomposition"]["matching"]["n_matched"] > 1_000_000
    assert "C:\\" not in text and "/Users/" not in text
