"""The downloader and tracked Pan-UKB publisher checksums stay in lockstep."""

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "panukb_download.sh"
CHECKSUMS = ROOT / "scripts" / "panukb_checksums.tsv"


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
