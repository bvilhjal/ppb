#!/usr/bin/env python3
"""Build a PPB reference-frequency panel from 1000 Genomes.

The panel holds one counted-allele (ALT) frequency per HapMap3 rsID and per
1000 Genomes phase-3 super-population (default AFR, AMR, EAS, EUR, SAS). It
is the reference artifact for PPB's EAF-profile decomposition.

This is a heavy, one-off admin step, not part of CI:

  1. For a whole-genome panel, use the release's compact ``ALL.wgs...sites``
     VCF, whose INFO field contains super-population ALT frequencies, plus the
     sample panel. ``--download-dir downloads --chrom 22`` instead fetches one
     chromosome's genotype VCF for a smoke test or GT-count fallback.
  2. Run this script over the VCF and the HapMap3 LD reference from ldpred3's
     benchmarks (``--ldref`` supplies rsIDs, coordinates, and alleles).
  3. Pin the reported canonical content digest outside the artifact and pass
     it as ``expected_sha256`` when loading the panel in production.

Download mode verifies every VCF and the sample panel against the exact byte
size and MD5 published in IGSR's ``current.tree`` checksum index. The manifest
records those checks, the index's observed SHA-256, and observed SHA-256 values
for every input. A local-input build can request the same verification with
``--checksum-tree``. Independently pin the index and LD reference when stronger
source authentication is required. Strand-ambiguous variants (A/T, C/G) are
excluded outright: frequency-based strand resolution is unreliable across
ancestries.

Example:
  python scripts/build_ancestry_panels.py \
      --ldref ../ldpred3/benchmarks/.work/ldref-hm3-pin/ldpred3_ldref_hm3.npz \
      --vcfs downloads/ALL.wgs.phase3_shapeit2_mvncall_integrated_v5c.*.vcf.gz \
      --samples downloads/integrated_call_samples_v3.20130502.ALL.panel \
      --checksum-tree downloads/current.tree \
      --out panels/kgp_superpops_hm3.npz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import urllib.request
from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np

FTP_BASE = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
CHECKSUM_TREE_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/current.tree")
CHECKSUM_TREE_NAME = "current.tree"
CHECKSUM_TREE_RELEASE_PREFIX = "ftp/release/20130502/"
SAMPLE_PANEL_URL = (
    f"{FTP_BASE}/integrated_call_samples_v3.20130502.ALL.panel")
VCF_TEMPLATE = (
    f"{FTP_BASE}/ALL.chr{{chrom}}.phase3_shapeit2_mvncall_integrated_v5b"
    f".20130502.genotypes.vcf.gz")
DEFAULT_POPS = ("AFR", "AMR", "EAS", "EUR", "SAS")
_SNPS = frozenset("ACGT")


def sha256_of(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_of(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


OfficialChecksum = namedtuple(
    "OfficialChecksum", ("tree_path", "size", "md5"))


def _tree_path(filename: str) -> str:
    return CHECKSUM_TREE_RELEASE_PREFIX + Path(filename).name


def read_official_checksums(
        tree_path: Path, filenames) -> dict[str, OfficialChecksum]:
    """Read only the requested release-file records from IGSR current.tree."""
    wanted = {_tree_path(name): Path(name).name for name in filenames}
    found = {}
    with open(tree_path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5 or fields[0] not in wanted:
                continue
            tree_name, kind, size_text, _modified, md5 = fields[:5]
            if kind != "file":
                raise ValueError(
                    f"{tree_path}:{number}: {tree_name} is not a file")
            try:
                size = int(size_text)
            except ValueError as exc:
                raise ValueError(
                    f"{tree_path}:{number}: invalid size for {tree_name}") \
                    from exc
            md5 = md5.strip().lower()
            if (size < 0 or len(md5) != 32
                    or any(char not in "0123456789abcdef" for char in md5)):
                raise ValueError(
                    f"{tree_path}:{number}: invalid checksum for {tree_name}")
            basename = wanted[tree_name]
            record = OfficialChecksum(tree_name, size, md5)
            if basename in found and found[basename] != record:
                raise ValueError(
                    f"{tree_path}: conflicting records for {tree_name}")
            found[basename] = record
    missing = sorted(set(wanted.values()) - set(found))
    if missing:
        raise ValueError(
            f"{tree_path} lacks official checksum records for: "
            f"{', '.join(missing)}")
    return found


def verify_official_file(path: Path, expected: OfficialChecksum) -> None:
    """Fail closed unless one file matches IGSR's exact size and MD5."""
    observed_size = path.stat().st_size
    if observed_size != expected.size:
        raise ValueError(
            f"{path} has {observed_size} bytes; official current.tree "
            f"requires {expected.size}")
    observed_md5 = md5_of(path)
    if observed_md5 != expected.md5:
        raise ValueError(
            f"{path} has MD5 {observed_md5}; official current.tree "
            f"requires {expected.md5}")


def fetch_checksum_tree(dest: Path) -> None:
    """Fetch the official index once per download-mode invocation."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    print(f"downloading {CHECKSUM_TREE_URL}", file=sys.stderr)
    try:
        urllib.request.urlretrieve(CHECKSUM_TREE_URL, partial)
        partial.replace(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def download_file(url: str, dest: Path,
                  expected: OfficialChecksum) -> None:
    """Download or reuse one release file only after official verification."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        verify_official_file(dest, expected)
        return
    partial = dest.with_name(dest.name + ".part")
    print(f"downloading {url}", file=sys.stderr)
    try:
        urllib.request.urlretrieve(url, partial)
        verify_official_file(partial, expected)
        partial.replace(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def read_sample_groups(panel_path: Path,
                       pops: tuple[str, ...]) -> dict[str, list[_SampleSlot]]:
    """One named sample per requested super-population member."""
    groups = defaultdict(list)
    with open(panel_path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            sample_col = header.index("sample")
            super_name = next(
                name for name in ("super_pop", "super_population")
                if name in header)
            super_col = header.index(super_name)
        except ValueError as exc:
            raise SystemExit(
                f"{panel_path} lacks sample and super_pop/super_population "
                "columns") from exc
        except StopIteration as exc:
            raise SystemExit(
                f"{panel_path} lacks sample and super_pop/super_population "
                "columns") from exc
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max(sample_col, super_col):
                continue
            sample, super_pop = fields[sample_col], fields[super_col]
            if super_pop in pops:
                groups[super_pop].append(_SampleSlot(sample))
    missing = [pop for pop in pops if not groups[pop]]
    if missing:
        raise SystemExit(
            f"{panel_path} has no samples for requested populations: "
            f"{', '.join(missing)}")
    return groups


class _SampleSlot:
    __slots__ = ("sample",)

    def __init__(self, sample):
        self.sample = sample


def alt_allele_frequencies(genotypes, columns_by_pop):
    """ALT allele counts and called-allele totals per population.

    ``genotypes`` is one variant's per-sample GT string (``0|1`` style);
    ``columns_by_pop`` maps a population to the VCF sample-column indices it
    uses. Pure and unit-tested: the heavy VCF walk only feeds records in.
    """
    out = {}
    for pop, columns in columns_by_pop.items():
        alt = called = 0
        for column in columns:
            genotype = genotypes[column]
            for allele in genotype.replace("|", "/").split("/"):
                if allele == ".":
                    continue
                if allele not in ("0", "1"):
                    return None
                called += 1
                if allele == "1":
                    alt += 1
        out[pop] = (alt, called)
    return out


def info_superpop_frequencies(info: str, pops: tuple[str, ...]):
    """Return complete valid 1000G ``<POP>_AF`` annotations, or ``None``."""
    tags = {}
    for item in info.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            tags[key] = value
    out = {}
    for pop in pops:
        value = tags.get(f"{pop}_AF")
        if value is None or "," in value:
            return None
        try:
            frequency = float(value)
        except ValueError:
            return None
        if not math.isfinite(frequency) or not 0.0 <= frequency <= 1.0:
            return None
        out[pop] = frequency
    return out


def _normalise_chrom(value) -> str:
    chrom = str(value).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    if chrom.isdigit():
        return str(int(chrom))
    return chrom.upper()


def _columns_by_pop(sample_names, slots, vcf: Path):
    """Map sample-panel members to one VCF's zero-based sample columns."""
    sample_index = {sample: column
                    for column, sample in enumerate(sample_names)}
    if len(sample_index) != len(sample_names):
        raise SystemExit(f"{vcf} contains duplicate sample names")
    columns = {}
    for pop, group in slots.items():
        try:
            columns[pop] = tuple(sample_index[slot.sample] for slot in group)
        except KeyError as exc:
            raise SystemExit(
                f"sample {exc.args[0]!r} from the sample panel is absent "
                f"from {vcf}") from exc
    return columns


def build_panel(ldref: Path, vcfs: list[Path], samples: Path,
                pops: tuple[str, ...]) -> dict:
    """Join phase-3 VCF records to HM3 by GRCh37 coordinate and alleles."""
    with np.load(ldref, allow_pickle=False) as archive:
        ref_ids = np.asarray(archive["ids"]).astype(str)
        chrom = np.asarray(archive["chrom"]).astype(str)
        pos = np.asarray(archive["pos"]).astype(np.int64)
        ref_counted = np.char.upper(
            np.asarray(archive["counted_allele"]).astype(str))
        ref_other = np.char.upper(
            np.asarray(archive["other_allele"]).astype(str))

    m = len(ref_ids)
    if (chrom.shape != (m,) or pos.shape != (m,)
            or ref_counted.shape != (m,) or ref_other.shape != (m,)):
        raise ValueError("LD reference variant metadata arrays disagree")
    by_coordinate = defaultdict(list)
    for row, (row_chrom, row_pos) in enumerate(zip(chrom, pos)):
        by_coordinate[(_normalise_chrom(row_chrom), int(row_pos))].append(row)

    slots = read_sample_groups(samples, pops)
    found = {}                     # LD-reference row -> panel row arrays
    n_scanned = 0
    n_info = 0
    n_genotypes = 0
    n_invalid_genotypes = 0
    n_allele_mismatch = 0
    n_ambiguous = 0
    for vcf in vcfs:
        opener = gzip.open if str(vcf).endswith(".gz") else open
        with opener(vcf, "rt", encoding="utf-8") as fh:
            sample_names = None
            columns_by_pop = None
            for line in fh:
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    header = line.rstrip("\n").split("\t")
                    sample_names = header[9:]
                    continue
                n_scanned += 1
                # The ninth split preserves all 2,504 sample fields as one
                # tail. Non-HM3 records therefore avoid thousands of string
                # allocations before being rejected by coordinate.
                fields = line.rstrip("\n").split("\t", 9)
                if len(fields) < 8:
                    continue
                try:
                    coordinate = (_normalise_chrom(fields[0]), int(fields[1]))
                except ValueError:
                    continue
                candidates = by_coordinate.get(coordinate)
                if not candidates:
                    continue
                ref, alts = fields[3], fields[4].split(",")
                ref = ref.upper()
                alts = [allele.upper() for allele in alts]
                if len(alts) != 1 or ref not in _SNPS or alts[0] not in _SNPS:
                    continue
                if (ref, alts[0]) in (("A", "T"), ("T", "A"),
                                      ("C", "G"), ("G", "C")):
                    continue                 # strand-ambiguous
                allele_pair = frozenset((ref, alts[0]))
                matches = [row for row in candidates
                           if frozenset((ref_counted[row], ref_other[row]))
                           == allele_pair]
                if not matches:
                    n_allele_mismatch += 1
                    continue
                if len(matches) != 1:
                    n_ambiguous += 1
                    continue
                row = matches[0]
                if row in found:
                    continue
                if sample_names is None:
                    raise SystemExit(f"{vcf} has records before #CHROM")

                frequencies = info_superpop_frequencies(fields[7], pops)
                if frequencies is not None:
                    n_info += 1
                else:
                    if len(fields) < 10:
                        continue
                    if columns_by_pop is None:
                        columns_by_pop = _columns_by_pop(
                            sample_names, slots, vcf)
                    sample_fields = fields[9].split("\t")
                    format_fields = fields[8].split(":")
                    try:
                        gt_index = format_fields.index("GT")
                    except ValueError:
                        continue
                    genotypes = []
                    for sample in sample_fields:
                        values = sample.split(":")
                        genotypes.append(
                            values[gt_index] if gt_index < len(values) else ".")
                    counts = alt_allele_frequencies(
                        genotypes, columns_by_pop)
                    if counts is None:
                        n_invalid_genotypes += 1
                        continue
                    if any(called == 0 for _alt, called in counts.values()):
                        continue
                    frequencies = {
                        pop: alt / called
                        for pop, (alt, called) in counts.items()}
                    n_genotypes += 1
                # The counted allele is ALT because ``frequencies`` contains
                # ALT-allele counts. Recording REF here would silently replace
                # every frequency by its complement at match time.
                found[row] = (row, chrom[row], pos[row], alts[0], ref,
                              frequencies)
    rows = sorted(found.values(), key=lambda item: item[0])
    ids = np.array([ref_ids[row] for row, *_ in rows])
    chroms = np.array([c for _row, c, *_ in rows])
    positions = np.array([p for _row, _c, p, *_ in rows])
    counted = np.array([a1 for *_r, _c, _p, a1, _a2, _f in rows])
    other = np.array([a2 for *_r, _c, _p, _a1, a2, _f in rows])
    af = np.array([[freq[pop] for pop in pops]
                   for *_head, freq in rows], dtype=np.float64)
    n_samples = np.array([
        sum(1 for _slot in slots[pop]) for pop in pops], dtype=np.int64)
    print(f"scanned {n_scanned:,} VCF records; kept {len(rows):,} of "
          f"{m:,} reference variants (INFO AF: {n_info:,}; genotype AF: "
          f"{n_genotypes:,}; allele mismatches: {n_allele_mismatch:,}; "
          f"ambiguous matches: {n_ambiguous:,}; invalid genotype records: "
          f"{n_invalid_genotypes:,})", file=sys.stderr)
    return {"ids": ids, "chrom": chroms, "pos": positions,
            "counted_allele": counted, "other_allele": other,
            "pops": list(pops), "af": af, "n_samples": n_samples,
            "frequency_source_counts": {
                "info": n_info, "genotypes": n_genotypes},
            "join_diagnostics": {
                "allele_mismatch": n_allele_mismatch,
                "ambiguous": n_ambiguous,
                "invalid_genotypes": n_invalid_genotypes}}


def _manifest_source(role: str, path: Path, official_by_path) -> dict:
    record = {"role": role, "path": str(path), "size": path.stat().st_size,
              "sha256": sha256_of(path)}
    official = official_by_path.get(path.resolve())
    if official is not None:
        record["official_current_tree"] = {
            "path": official.tree_path, "size": official.size,
            "md5": official.md5, "verified": True}
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ldref", required=True, type=Path,
                        help="ldpred3 HapMap3 LD reference NPZ (rsID list)")
    parser.add_argument("--vcfs", type=str, default=None,
                        help="glob of 1000G phase-3 sites or genotype VCFs")
    parser.add_argument(
        "--samples", type=Path, default=None,
        help=("integrated_call_samples_v3.20130502.ALL.panel; omitted when "
              "--download-dir is used"))
    parser.add_argument("--pops", type=str, default=",".join(DEFAULT_POPS),
                        help="comma-separated super-populations")
    parser.add_argument("--out", required=True, type=Path,
                        help="output panel NPZ path")
    parser.add_argument("--download-dir", type=Path, default=None,
                        help="download VCFs here before building")
    parser.add_argument("--chrom", type=str, default=None,
                        help="comma-separated chromosomes (download mode)")
    parser.add_argument(
        "--checksum-tree", type=Path, default=None,
        help=("official IGSR current.tree for verifying local --vcfs and "
              "--samples; download mode fetches its own current copy"))
    args = parser.parse_args(argv)
    pops = tuple(pop.strip() for pop in args.pops.split(",") if pop.strip())
    if args.download_dir is not None and args.checksum_tree is not None:
        parser.error(
            "--checksum-tree is for local inputs; download mode fetches "
            "current.tree itself")

    vcfs = []
    if args.vcfs:
        import glob
        vcfs.extend(Path(name) for name in sorted(
            glob.glob(str(args.vcfs))))

    download_requests = []
    checksum_tree = args.checksum_tree
    checksum_tree_url = ""
    if args.download_dir is not None:
        if not args.chrom:
            parser.error("--download-dir needs --chrom")
        for chrom in args.chrom.split(","):
            chrom = chrom.strip()
            if not chrom:
                continue
            vcf = args.download_dir / VCF_TEMPLATE.rsplit("/", 1)[-1].format(
                chrom=chrom)
            vcfs.append(vcf)
            download_requests.append(
                (VCF_TEMPLATE.format(chrom=chrom), vcf))
        panel_file = args.download_dir / SAMPLE_PANEL_URL.rsplit("/", 1)[-1]
        download_requests.append((SAMPLE_PANEL_URL, panel_file))
        args.samples = panel_file
        checksum_tree = args.download_dir / CHECKSUM_TREE_NAME
        fetch_checksum_tree(checksum_tree)
        checksum_tree_url = CHECKSUM_TREE_URL

    if not vcfs:
        parser.error("no VCFs: give --vcfs and/or --download-dir --chrom")
    if args.samples is None:
        parser.error("--samples is required unless --download-dir is used")
    if checksum_tree is not None and not checksum_tree.exists():
        parser.error(f"{checksum_tree} does not exist")

    official_by_path = {}
    checksum_index_manifest = None
    if checksum_tree is not None:
        inputs = [*vcfs, args.samples]
        official_by_name = read_official_checksums(
            checksum_tree, [path.name for path in inputs])
        checksum_index_manifest = {
            "path": str(checksum_tree), "source_url": checksum_tree_url,
            "size": checksum_tree.stat().st_size,
            "sha256": sha256_of(checksum_tree)}
        downloaded_paths = set()
        for url, path in download_requests:
            download_file(url, path, official_by_name[path.name])
            downloaded_paths.add(path.resolve())
        for path in inputs:
            official = official_by_name[path.name]
            if path.resolve() not in downloaded_paths:
                if not path.exists():
                    parser.error(f"{path} does not exist")
                verify_official_file(path, official)
            official_by_path[path.resolve()] = official

    if not args.samples.exists():
        parser.error(f"{args.samples} does not exist")
    for vcf in vcfs:
        if not vcf.exists():
            parser.error(f"{vcf} does not exist")
    if not args.ldref.exists():
        parser.error(f"{args.ldref} does not exist")

    from ppb import load_frequency_panel, write_frequency_panel

    built = build_panel(args.ldref, vcfs, args.samples, pops)
    write_frequency_panel(
        args.out, ids=built["ids"], chrom=built["chrom"], pos=built["pos"],
        counted_allele=built["counted_allele"],
        other_allele=built["other_allele"], pops=built["pops"],
        af=built["af"], n_samples=built["n_samples"],
        source="1000 Genomes phase 3 (20130502) super-population ALT "
               "frequencies at HapMap3 sites",
        source_url=FTP_BASE, genome_build="hg19/GRCh37")
    panel = load_frequency_panel(args.out)  # fail closed on our own output

    manifest = args.out.with_name(args.out.name + ".manifest.json")
    sources = [_manifest_source(
        "vcf", vcf, official_by_path) for vcf in vcfs]
    sources.append(_manifest_source(
        "samples", args.samples, official_by_path))
    sources.append(_manifest_source("ldref", args.ldref, {}))
    if checksum_index_manifest is not None:
        provenance = (
            "Every 1000 Genomes VCF and sample input matched the exact byte "
            "size and MD5 in IGSR current.tree. The index SHA-256 and LD-"
            "reference SHA-256 are observed provenance; independently pin "
            "them when stronger source authentication is required.")
    else:
        provenance = (
            "Source SHA-256 values are observed provenance, not independent "
            "authentication; use --checksum-tree to verify local 1000 "
            "Genomes inputs against IGSR metadata.")
    manifest.write_text(json.dumps(
         {"panel": str(args.out),
          "panel_content_sha256": panel.panel_sha256, "sources": sources,
          "n_variants": int(len(built["ids"])), "pops": list(pops),
          "official_checksum_index": checksum_index_manifest,
          "frequency_source_counts": built["frequency_source_counts"],
          "join_diagnostics": built["join_diagnostics"],
          "provenance_status": provenance},
         indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({len(built['ids']):,} variants, "
          f"{len(pops)} populations) and {manifest}", file=sys.stderr)
    print(
        f"canonical panel sha256: {panel.panel_sha256}",
        file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
