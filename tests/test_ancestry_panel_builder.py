"""1000 Genomes frequency-panel builder provenance and orientation tests."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _builder(name="ppb_ancestry_panel_builder"):
    script = Path(__file__).resolve().parents[1] / "scripts" \
        / "build_ancestry_panels.py"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _samples(path):
    path.write_text(
        "sample\tpop\tsuper_pop\tgender\n"
        "S1\tYRI\tAFR\tmale\nS2\tLWK\tAFR\tfemale\n"
        "S3\tCEU\tEUR\tmale\nS4\tGBR\tEUR\tfemale\n")
    return path


def test_alt_counting_and_both_super_population_headers(tmp_path):
    module = _builder("ppb_builder_headers")
    counts = module.alt_allele_frequencies(
        ["0|0", "0|1", "1|1", "1|0", ".|.", "0|0"],
        {"AFR": [0, 1, 4], "EUR": [2, 3, 5]})
    assert counts == {"AFR": (1, 4), "EUR": (3, 6)}

    for header in ("super_pop", "super_population"):
        sample_file = tmp_path / f"samples-{header}.panel"
        sample_file.write_text(
            f"sample\tpop\t{header}\tgender\n"
            "S1\tYRI\tAFR\tmale\nS2\tCEU\tEUR\tfemale\n")
        groups = module.read_sample_groups(sample_file, ("AFR", "EUR"))
        assert [slot.sample for slot in groups["AFR"]] == ["S1"]
        assert [slot.sample for slot in groups["EUR"]] == ["S2"]


def test_download_mode_checks_official_tree_once(tmp_path, monkeypatch):
    module = _builder("ppb_builder_download")
    vcf_name = module.VCF_TEMPLATE.rsplit("/", 1)[-1].format(chrom="22")
    sample_name = module.SAMPLE_PANEL_URL.rsplit("/", 1)[-1]
    payloads = {vcf_name: b"release VCF", sample_name: b"sample panel"}
    tree_text = "".join(
        f"{module._tree_path(name)}\tfile\t{len(data)}\t"
        f"2021-01-01 00:00:00\t{hashlib.md5(data).hexdigest()}\t\n"
        for name, data in payloads.items())
    calls = []

    def fake_retrieve(url, destination):
        calls.append(url)
        if url == module.CHECKSUM_TREE_URL:
            Path(destination).write_text(tree_text)
        else:
            Path(destination).write_bytes(payloads[url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(module.urllib.request, "urlretrieve", fake_retrieve)
    monkeypatch.setattr(
        module, "build_panel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("build reached")))
    ldref = tmp_path / "ldref.npz"
    ldref.write_bytes(b"placeholder")
    with pytest.raises(RuntimeError, match="build reached"):
        module.main([
            "--ldref", str(ldref),
            "--download-dir", str(tmp_path / "downloads"),
            "--chrom", "22", "--out", str(tmp_path / "panel.npz")])
    assert calls.count(module.CHECKSUM_TREE_URL) == 1
    assert len(calls) == 3
    assert (tmp_path / "downloads" / vcf_name).read_bytes() == payloads[vcf_name]
    assert (tmp_path / "downloads" / sample_name).read_bytes() == payloads[
        sample_name]


def test_download_refuses_bad_existing_and_downloaded_files(
        tmp_path, monkeypatch):
    module = _builder("ppb_builder_checks")
    good = b"official bytes"
    expected = module.OfficialChecksum(
        "ftp/release/20130502/artifact.vcf.gz", len(good),
        hashlib.md5(good).hexdigest())
    existing = tmp_path / "existing.vcf.gz"
    existing.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="official current.tree requires"):
        module.download_file("https://example.test/artifact", existing, expected)

    downloaded = tmp_path / "downloaded.vcf.gz"
    monkeypatch.setattr(
        module.urllib.request, "urlretrieve",
        lambda _url, destination: Path(destination).write_bytes(b"wrong"))
    with pytest.raises(ValueError, match="official current.tree requires"):
        module.download_file(
            "https://example.test/artifact", downloaded, expected)
    assert not downloaded.exists()
    assert not (tmp_path / "downloaded.vcf.gz.part").exists()


def test_end_to_end_gt_fallback_orientation_and_manifest(tmp_path):
    module = _builder("ppb_builder_e2e")
    ldref = tmp_path / "ldref.npz"
    np.savez_compressed(
        ldref, ids=np.array(["rs1"]), chrom=np.array(["22"]),
        pos=np.array([101], dtype=np.int64),
        counted_allele=np.array(["C"]), other_allele=np.array(["A"]))
    samples = _samples(tmp_path / "samples.panel")
    vcf = tmp_path / "chr22.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        "S1\tS2\tS3\tS4\n"
        "22\t101\t.\tA\tC\t.\tPASS\t.\tGT:DP\t"
        "0|0:12\t0|1:9\t1|1:15\t0|1:11\n")
    built = module.build_panel(ldref, [vcf], samples, ("AFR", "EUR"))
    assert built["ids"].tolist() == ["rs1"]
    assert built["counted_allele"].tolist() == ["C"]
    assert built["other_allele"].tolist() == ["A"]
    assert np.allclose(built["af"], [[0.25, 0.75]])
    assert built["n_samples"].tolist() == [2, 2]
    assert built["frequency_source_counts"] == {"info": 0, "genotypes": 1}

    checksum_tree = tmp_path / "current.tree"
    checksum_tree.write_text("".join(
        f"{module._tree_path(path.name)}\tfile\t{path.stat().st_size}\t"
        f"2021-01-01 00:00:00\t"
        f"{hashlib.md5(path.read_bytes()).hexdigest()}\t\n"
        for path in (vcf, samples)))
    out = tmp_path / "built.npz"
    assert module.main([
        "--ldref", str(ldref), "--vcfs", str(vcf),
        "--samples", str(samples), "--pops", "AFR,EUR",
        "--checksum-tree", str(checksum_tree), "--out", str(out)]) == 0
    manifest = json.loads(
        (tmp_path / "built.npz.manifest.json").read_text())
    sources = {item["role"]: item for item in manifest["sources"]}
    assert set(sources) == {"vcf", "samples", "ldref"}
    assert sources["ldref"]["size"] == ldref.stat().st_size
    assert len(sources["ldref"]["sha256"]) == 64
    assert len(manifest["official_checksum_index"]["sha256"]) == 64
    assert sources["vcf"]["official_current_tree"]["verified"] is True
    assert sources["samples"]["official_current_tree"]["verified"] is True
    assert "official_current_tree" not in sources["ldref"]
    assert "matched the exact byte size and MD5" in manifest["provenance_status"]


def test_info_frequencies_match_genotype_fallback(tmp_path):
    module = _builder("ppb_builder_info")
    ldref = tmp_path / "ldref.npz"
    np.savez_compressed(
        ldref, ids=np.array(["rs1"]), chrom=np.array(["22"]), pos=[101],
        counted_allele=np.array(["C"]), other_allele=np.array(["A"]))
    samples = _samples(tmp_path / "samples.panel")
    info = tmp_path / "info.vcf"
    info.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "22\t101\t.\tA\tC\t.\tPASS\tAFR_AF=0.25004;EUR_AF=0.74996\n")
    gt = tmp_path / "gt.vcf"
    gt.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        "S1\tS2\tS3\tS4\n"
        "22\t101\t.\tA\tC\t.\tPASS\t.\tGT\t0|0\t0|1\t1|1\t0|1\n")
    from_info = module.build_panel(ldref, [info], samples, ("AFR", "EUR"))
    from_gt = module.build_panel(ldref, [gt], samples, ("AFR", "EUR"))
    assert np.allclose(from_info["af"], from_gt["af"], atol=5e-5)
    assert from_info["frequency_source_counts"] == {"info": 1, "genotypes": 0}
    assert from_gt["frequency_source_counts"] == {"info": 0, "genotypes": 1}


def test_coordinate_allele_join_reports_duplicates_and_mismatch(tmp_path):
    module = _builder("ppb_builder_join")
    ldref = tmp_path / "ldref.npz"
    np.savez_compressed(
        ldref,
        ids=np.array(["rs_ac", "rs_ag", "rs_dup1", "rs_dup2", "rs_mm"]),
        chrom=np.array(["22"] * 5), pos=np.array([101, 101, 202, 202, 303]),
        counted_allele=np.array(["C", "G", "C", "A", "C"]),
        other_allele=np.array(["A", "A", "A", "C", "A"]))
    samples = tmp_path / "samples.panel"
    samples.write_text(
        "sample\tpop\tsuper_pop\tgender\n"
        "S1\tYRI\tAFR\tmale\nS2\tCEU\tEUR\tfemale\n")
    vcf = tmp_path / "variants.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
        "22\t101\t.\tA\tG\t.\tPASS\tAFR_AF=0.1;EUR_AF=0.2\tGT\t0|0\t0|1\n"
        "22\t202\t.\tA\tC\t.\tPASS\tAFR_AF=0.2;EUR_AF=0.3\tGT\t0|0\t0|1\n"
        "22\t303\t.\tA\tG\t.\tPASS\tAFR_AF=0.3;EUR_AF=0.4\tGT\t0|0\t0|1\n")
    built = module.build_panel(ldref, [vcf], samples, ("AFR", "EUR"))
    assert built["ids"].tolist() == ["rs_ag"]
    assert built["join_diagnostics"] == {
        "allele_mismatch": 1, "ambiguous": 1, "duplicate": 0,
        "invalid_genotypes": 0}


def test_gt_fallback_skips_invalid_allele_codes(tmp_path):
    module = _builder("ppb_builder_invalid_gt")
    ldref = tmp_path / "ldref.npz"
    np.savez_compressed(
        ldref, ids=np.array(["rs_bad", "rs_good"]),
        chrom=np.array(["22", "22"]), pos=np.array([101, 202]),
        counted_allele=np.array(["C", "C"]),
        other_allele=np.array(["A", "A"]))
    samples = tmp_path / "samples.panel"
    samples.write_text(
        "sample\tpop\tsuper_pop\tgender\n"
        "S1\tYRI\tAFR\tmale\nS2\tCEU\tEUR\tfemale\n")
    vcf = tmp_path / "variants.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
        "22\t101\t.\tA\tC\t.\tPASS\t.\tGT\t0|2\t0|1\n"
        "22\t202\t.\tA\tC\t.\tPASS\t.\tGT\t0|1\t1|1\n")
    built = module.build_panel(ldref, [vcf], samples, ("AFR", "EUR"))
    assert built["ids"].tolist() == ["rs_good"]
    assert built["join_diagnostics"]["invalid_genotypes"] == 1
    assert built["frequency_source_counts"] == {"info": 0, "genotypes": 1}


def test_sites_vcf_skips_incomplete_or_nonfinite_info(tmp_path):
    module = _builder("ppb_builder_incomplete_info")
    ldref = tmp_path / "ldref.npz"
    np.savez_compressed(
        ldref, ids=np.array(["rs_incomplete", "rs_invalid", "rs_complete"]),
        chrom=np.array(["22"] * 3), pos=np.array([101, 151, 202]),
        counted_allele=np.array(["C"] * 3),
        other_allele=np.array(["A"] * 3))
    samples = tmp_path / "samples.panel"
    samples.write_text(
        "sample\tpop\tsuper_pop\tgender\n"
        "S1\tYRI\tAFR\tmale\nS2\tCEU\tEUR\tfemale\n")
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "22\t101\t.\tA\tC\t.\tPASS\tAFR_AF=0.1\n"
        "22\t151\t.\tA\tC\t.\tPASS\tAFR_AF=nan;EUR_AF=0.2\n"
        "22\t202\t.\tA\tC\t.\tPASS\tAFR_AF=0.2;EUR_AF=0.3\n")
    built = module.build_panel(ldref, [vcf], samples, ("AFR", "EUR"))
    assert built["ids"].tolist() == ["rs_complete"]
    assert built["frequency_source_counts"] == {"info": 1, "genotypes": 0}


def test_repeated_vcf_record_is_counted_not_silently_dropped(tmp_path):
    # A second VCF record for a row already captured is discarded, but the
    # discard has to be visible: its sibling reasons (allele mismatch,
    # ambiguous match) are counted, and an uncounted third reason would make
    # the reconciliation in the printed summary fail to add up.
    module = _builder("ppb_builder_dup")
    ldref = tmp_path / "ldref.npz"
    np.savez_compressed(
        ldref, ids=np.array(["rs_one"]), chrom=np.array(["22"]),
        pos=np.array([101]), counted_allele=np.array(["G"]),
        other_allele=np.array(["A"]))
    samples = tmp_path / "samples.panel"
    samples.write_text(
        "sample\tpop\tsuper_pop\tgender\n"
        "S1\tYRI\tAFR\tmale\nS2\tCEU\tEUR\tfemale\n")
    vcf = tmp_path / "variants.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
        "22\t101\t.\tA\tG\t.\tPASS\tAFR_AF=0.1;EUR_AF=0.2\tGT\t0|0\t0|1\n"
        "22\t101\t.\tA\tG\t.\tPASS\tAFR_AF=0.9;EUR_AF=0.8\tGT\t1|1\t1|1\n")
    built = module.build_panel(ldref, [vcf], samples, ("AFR", "EUR"))
    assert built["ids"].tolist() == ["rs_one"]
    assert built["join_diagnostics"]["duplicate"] == 1
    # First record wins; the repeat must not overwrite the retained frequency.
    assert built["af"][0][0] == pytest.approx(0.1)
