"""The downloader and tracked Pan-UKB publisher checksums stay in lockstep,
and the input manifest catalogs every external file the scripts consume."""

import csv
import re
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "panukb_download.sh"
CHECKSUMS = ROOT / "scripts" / "panukb_checksums.tsv"
INPUTS = ROOT / "results" / "inputs.tsv"

MANIFEST_COLUMNS = [
    "artifact", "role", "source_url_or_accession", "checksum_md5",
    "date_recorded",
]


def test_every_panukb_download_has_one_valid_published_checksum():
    text = SCRIPT.read_text(encoding="utf-8")
    downloads = set(re.findall(r"^\s{2}([\w-]+\.tsv\.bgz)(?: \\)?$", text, re.MULTILINE))
    with CHECKSUMS.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    assert downloads
    assert {row["filename"] for row in rows} == downloads
    assert len(rows) == len(downloads)
    for row in rows:
        assert int(row["size_bytes"]) > 0
        assert re.fullmatch(r"[0-9a-f]{32}", row["md5"])


def _manifest_rows():
    """``(header, rows)`` from results/inputs.tsv, skipping '#' comments."""
    lines = [ln for ln in INPUTS.read_text(encoding="utf-8").splitlines()
             if ln and not ln.startswith("#")]
    header = lines[0].split("\t")
    return header, [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def test_manifest_exists_with_the_documented_columns():
    header, rows = _manifest_rows()
    assert header == MANIFEST_COLUMNS
    # 6 consortium GWAS + 9 PGS weight files + 22 LD chromosomes + the map
    # RDS + 9 Pan-UKB files.
    assert len(rows) >= 47


def test_manifest_rows_are_well_formed():
    _, rows = _manifest_rows()
    artifacts = [row["artifact"] for row in rows]
    assert len(artifacts) == len(set(artifacts)), "duplicate artifact rows"
    for row in rows:
        for column in MANIFEST_COLUMNS:
            assert row[column].strip(), \
                f"{row['artifact']}: empty '{column}'"
        assert row["checksum_md5"] == "unrecorded" \
            or re.fullmatch(r"[0-9a-f]{32}", row["checksum_md5"]), \
            f"{row['artifact']}: checksum_md5 must be a publisher MD5 or 'unrecorded'"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["date_recorded"])
        assert date.fromisoformat(row["date_recorded"]) <= date.today()


def test_panukb_checksums_are_carried_into_the_manifest():
    _, rows = _manifest_rows()
    by_artifact = {row["artifact"]: row for row in rows}
    with CHECKSUMS.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            assert by_artifact[row["filename"]]["checksum_md5"] == row["md5"], \
                f"{row['filename']}: manifest checksum disagrees with the tracked one"


def test_every_script_referenced_external_file_is_catalogued():
    """The manifest must cover every file the real-data scripts name: a file
    the scripts read but the manifest misses is exactly the unversioned input
    the manifest exists to prevent."""
    from scripts import consortium_prep
    from scripts.regenerate_results import TRAITS

    _, rows = _manifest_rows()
    artifacts = {row["artifact"] for row in rows}

    missing = [cfg["src"] for cfg in consortium_prep.CFG.values()
               if cfg["src"] not in artifacts]
    missing += [f"{cfg['pgs']}_hmPOS_GRCh37.txt" for cfg in TRAITS.values()
                if f"{cfg['pgs']}_hmPOS_GRCh37.txt" not in artifacts]
    missing += [f"LD_with_blocks_chr{chrom}.rds" for chrom in range(1, 23)
                if f"LD_with_blocks_chr{chrom}.rds" not in artifacts]
    missing += [name for name in ("map_hm3_plus.rds",)
                if name not in artifacts]
    with CHECKSUMS.open(encoding="utf-8", newline="") as fh:
        missing += [row["filename"] for row in csv.DictReader(fh, delimiter="\t")
                    if row["filename"] not in artifacts]
    assert not missing, f"uncatalogued external inputs: {sorted(missing)}"


def test_download_script_validates_checksums_portably():
    """`md5sum --check --status` is GNU-only: on BSD/macOS it fails after the
    download completes. The portable digest helper must keep failing closed
    when no checksum tool exists at all."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--check" not in text, "GNU-only md5sum --check remains"
    assert "md5sum" in text and "md5 -q" in text and "hashlib" in text, \
        "the portable md5_of fallback chain is incomplete"
