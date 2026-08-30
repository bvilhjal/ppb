#!/usr/bin/env python3
"""Build a compact five-population 1000G LD-moment design.

The build is deliberately target-blind. It retains HapMap3 SNPs with MAF at
least 5% in every 1000 Genomes super-population, greedily keeps source LD
blocks separated by a fixed physical gap, and takes evenly spaced markers
within each retained block. Dense population LD matrices exist only while one
block is processed; the output contains the sufficient statistics consumed by
``ppb.estimate_*_from_design``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from ppb import load_frequency_panel


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT.parent / "reference-data" / "1000g-phase3"
DEFAULT_AF_PANEL = (
    REFERENCE_ROOT / "panels" /
    "kgp_phase3_20130502_superpops_hm3_grch37.npz"
)
AF_PANEL_SHA256 = (
    "c91b6652671ceb373417e9d82c47db5048964923bf448801214e1f78b7291186"
)
DEFAULT_LD_GEOMETRY = (
    ROOT.parent / "ldpred3" / "benchmarks" / ".work" / "ldref-hm3-pin" /
    "ldpred3_ldref_hm3.npz"
)
DEFAULT_BED_PREFIX = (
    REFERENCE_ROOT / "source" / "1000G_phase3_common_norel" /
    "1000G_phase3_common_norel"
)
DEFAULT_ARCHIVE = (
    REFERENCE_ROOT / "source" / "1000G_phase3_common_norel.zip"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "ancestry_ld" /
    "kgp_phase3_common_distant_hm3_ld_design.npz"
)

SOURCE_URL = "https://doi.org/10.6084/m9.figshare.9208979.v4"
SOURCE_ARCHIVE_MD5 = "594626c0a3bc5a154a58c8f573ea0406"
LD_GEOMETRY_SHA256 = (
    "35bc59d58598b4e414b602c9f61f2ebd01f0ef65856b0a81a722e79bac0a4654"
)
LD_GEOMETRY_SOURCE_REVISION = "621a2c4dc10a225fd7abcddebc66fbdee33c9cf0"
POPS = ("AFR", "AMR", "EAS", "EUR", "SAS")
PLINK_MAGIC = b"\x6c\x1b\x01"


def file_digest(path: Path, algorithm="sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_digest(fields) -> str:
    """Hash all scientific fields independently of NPZ zip metadata."""
    digest = hashlib.sha256()
    digest.update(b"ppb-ancestry-ld-design-v1\0")
    for name in sorted(fields):
        value = np.asarray(fields[name])
        name_bytes = name.encode("ascii")
        digest.update(len(name_bytes).to_bytes(2, "little"))
        digest.update(name_bytes)
        if value.dtype.kind in "USO":
            payload = json.dumps(
                value.reshape(-1).astype(str).tolist(),
                ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
        else:
            canonical = np.ascontiguousarray(value)
            payload = canonical.dtype.str.encode("ascii") + b"\0"
            payload += json.dumps(
                canonical.shape, separators=(",", ":")
            ).encode("ascii") + b"\0"
            payload += memoryview(canonical).cast("B").tobytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _load_geometry(panel_path: Path, geometry_path: Path, maf_threshold: float):
    panel = load_frequency_panel(
        panel_path, expected_sha256=AF_PANEL_SHA256
    )
    if tuple(panel.pops) != POPS:
        raise ValueError(f"AF panel populations must be {POPS}")
    with np.load(geometry_path, allow_pickle=False) as archive:
        required = (
            "ids", "chrom", "pos", "counted_allele", "other_allele",
            "sizes",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"LD geometry is missing {missing}")
        geometry = {name: np.asarray(archive[name]).copy() for name in required}
    for name, expected in (
            ("ids", panel.ids), ("chrom", panel.chrom), ("pos", panel.pos),
            ("counted_allele", panel.counted_allele),
            ("other_allele", panel.other_allele)):
        if not np.array_equal(geometry[name].astype(str), expected.astype(str)):
            raise ValueError(f"AF panel and LD geometry disagree in {name}")
    sizes = geometry["sizes"].astype(np.int64)
    if sizes.ndim != 1 or (sizes <= 0).any() or sizes.sum() != len(panel):
        raise ValueError("LD geometry sizes do not partition the AF panel")
    common = np.min(np.minimum(panel.af, 1.0 - panel.af), axis=1) >= maf_threshold
    return panel, geometry, common


def select_distant_blocks(
        chrom, pos, sizes, common, *, min_gap_bp, min_variants):
    """Greedily retain ordered blocks with a fixed edge-to-edge gap."""
    chrom = np.asarray(chrom).astype(str)
    pos = np.asarray(pos, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    common = np.asarray(common, dtype=bool)
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    selected = []
    for chromosome in map(str, range(1, 23)):
        last_end = None
        for block in range(len(sizes)):
            lo, hi = int(bounds[block]), int(bounds[block + 1])
            if chrom[lo] != chromosome:
                continue
            indices = np.flatnonzero(common[lo:hi]) + lo
            if indices.size < min_variants:
                continue
            start, end = int(pos[lo:hi].min()), int(pos[lo:hi].max())
            if last_end is None or start - last_end >= min_gap_bp:
                selected.append((block, chromosome, start, end, indices))
                last_end = end
    if not selected:
        raise ValueError("no LD blocks pass the fixed selection rules")
    return selected


def evenly_spaced(indices, maximum):
    """Choose at most ``maximum`` ordered entries without using LD or z."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= maximum:
        return indices
    take = np.rint(np.linspace(0, indices.size - 1, maximum)).astype(int)
    out = indices[take]
    if out.size != maximum or np.unique(out).size != maximum:
        raise RuntimeError("evenly-spaced pruning did not return unique markers")
    return out


def _read_samples(prefix: Path):
    fam_path = prefix.with_suffix(".fam")
    fam2_path = prefix.with_suffix(".fam2")
    sample_ids = []
    with fam_path.open("rt", encoding="utf-8", newline="") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"malformed PLINK FAM row in {fam_path}")
            sample_ids.append(fields[1])
    mapping = {}
    with fam2_path.open("rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sample.ID", "Super Population"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{fam2_path} lacks {sorted(required)}")
        for row in reader:
            mapping[row["sample.ID"].strip()] = row["Super Population"].strip()
    unknown = [sample for sample in sample_ids if sample not in mapping]
    if unknown:
        raise ValueError(f"{len(unknown)} FAM samples lack super-population labels")
    sample_pop = np.asarray([mapping[sample] for sample in sample_ids])
    unexpected = sorted(set(sample_pop) - set(POPS))
    if unexpected:
        raise ValueError(f"unexpected super-populations {unexpected}")
    counts = np.asarray([(sample_pop == pop).sum() for pop in POPS], dtype=np.int64)
    if (counts < 2).any():
        raise ValueError("each super-population needs at least two samples")
    return np.asarray(sample_ids), sample_pop, counts


def _index_bim(path: Path, wanted_ids):
    wanted_ids = set(map(str, wanted_ids))
    found = {}
    n_rows = 0
    with path.open("rt", encoding="utf-8", newline="") as handle:
        for n_rows, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"malformed PLINK BIM row {n_rows} in {path}")
            chrom, rsid, _, pos, a1, a2 = fields
            if rsid in wanted_ids:
                if rsid in found:
                    raise ValueError(f"duplicate selected BIM variant {rsid}")
                found[rsid] = (
                    n_rows - 1, chrom, int(pos), a1.upper(), a2.upper()
                )
    missing = wanted_ids - set(found)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"{len(missing)} selected variants absent from BIM: {preview}")
    return found, n_rows


def decode_plink_rows(raw, n_samples):
    """Decode variant-major BED bytes to A1 dosages; missing is NaN."""
    raw = np.asarray(raw, dtype=np.uint8)
    if raw.ndim != 2:
        raise ValueError("raw BED rows must be two-dimensional")
    shifts = np.asarray([0, 2, 4, 6], dtype=np.uint8)
    codes = ((raw[:, :, None] >> shifts) & 3).reshape(raw.shape[0], -1)
    codes = codes[:, :n_samples]
    lookup = np.asarray([2.0, np.nan, 1.0, 0.0])
    return lookup[codes].T


def _correlations(genotypes, sample_pop):
    references = []
    empirical_af = []
    for pop in POPS:
        values = np.asarray(genotypes[sample_pop == pop], dtype=np.float64)
        means = np.nanmean(values, axis=0)
        if not np.isfinite(means).all():
            raise ValueError(f"{pop}: a selected variant has no called genotypes")
        missing = np.isnan(values)
        if missing.any():
            values = values.copy()
            values[missing] = np.broadcast_to(means, values.shape)[missing]
        centered = values - values.mean(axis=0)
        sums = np.sum(centered * centered, axis=0)
        if (sums <= 0.0).any():
            raise ValueError(f"{pop}: a selected variant is monomorphic")
        standardized = centered / np.sqrt(sums / (values.shape[0] - 1.0))
        correlation = standardized.T @ standardized / (values.shape[0] - 1.0)
        references.append(correlation)
        empirical_af.append(means / 2.0)
    return references, np.stack(empirical_af, axis=1)


def compact_block_design(references, *, floor, cap):
    """Return A and B sufficient statistics for one aligned LD block."""
    references = [np.asarray(ref, dtype=np.float64) for ref in references]
    K = len(references)
    if K < 2 or any(ref.shape != references[0].shape for ref in references):
        raise ValueError("references must be aligned square matrices")
    if references[0].ndim != 2 or references[0].shape[0] != references[0].shape[1]:
        raise ValueError("references must be square")
    m = references[0].shape[0]
    ti, tj = np.triu_indices(m, k=1)
    linear = np.stack([ref[ti, tj] for ref in references], axis=1)
    strength = np.max(np.abs(linear), axis=1)
    keep = np.flatnonzero(strength >= floor)
    if keep.size > cap:
        order = np.argsort(strength[keep], kind="stable")[::-1]
        keep = keep[order[:cap]]
    if keep.size == 0:
        raise ValueError("no pairs pass the configured LD floor")
    ii, jj = ti[keep], tj[keep]
    linear = linear[keep]
    pairs = [(k, kp) for k in range(K) for kp in range(k, K)]
    quadratic = []
    bilinear = []
    for k, kp in pairs:
        forward = np.einsum(
            "ij,ij->i", references[k][ii], references[kp][jj]
        )
        if k != kp:
            reverse = np.einsum(
                "ij,ij->i", references[kp][ii], references[k][jj]
            )
            forward = 0.5 * (forward + reverse)
        quadratic.append(forward)
        bilinear.append(np.einsum(
            "ij,ij->i", references[k], references[kp]
        ))
    return (
        ii.astype(np.int64), jj.astype(np.int64), linear,
        np.stack(quadratic, axis=1), np.stack(bilinear, axis=1),
    )


def build_design(
        *, panel_path, geometry_path, bed_prefix, archive_path,
        maf_threshold, min_block_gap_bp, min_variants_per_block,
        max_variants_per_block, ld_floor, max_pairs_per_block):
    if archive_path is not None:
        observed_md5 = file_digest(archive_path, "md5")
        if observed_md5 != SOURCE_ARCHIVE_MD5:
            raise ValueError(
                f"source archive MD5 {observed_md5} does not match pinned "
                f"{SOURCE_ARCHIVE_MD5}"
            )
    observed_geometry_sha256 = file_digest(geometry_path)
    if observed_geometry_sha256 != LD_GEOMETRY_SHA256:
        raise ValueError(
            f"LD geometry SHA-256 {observed_geometry_sha256} does not match "
            f"pinned {LD_GEOMETRY_SHA256}"
        )
    panel, geometry, common = _load_geometry(
        panel_path, geometry_path, maf_threshold
    )
    selected_blocks = select_distant_blocks(
        geometry["chrom"], geometry["pos"], geometry["sizes"], common,
        min_gap_bp=min_block_gap_bp,
        min_variants=min_variants_per_block,
    )
    pruned = [
        (block, chrom, start, end,
         evenly_spaced(indices, max_variants_per_block))
        for block, chrom, start, end, indices in selected_blocks
    ]
    selected_indices = np.concatenate([item[4] for item in pruned])
    selected_ids = panel.ids[selected_indices].astype(str)
    bim_path = bed_prefix.with_suffix(".bim")
    bed_path = bed_prefix.with_suffix(".bed")
    bim, n_bim = _index_bim(bim_path, selected_ids)
    sample_ids, sample_pop, sample_counts = _read_samples(bed_prefix)
    row_bytes = (len(sample_ids) + 3) // 4
    expected_size = 3 + n_bim * row_bytes
    if bed_path.stat().st_size != expected_size:
        raise ValueError(
            f"BED size {bed_path.stat().st_size} != expected {expected_size}"
        )
    with bed_path.open("rb") as handle:
        if handle.read(3) != PLINK_MAGIC:
            raise ValueError("BED is not PLINK variant-major binary format")
    bed = np.memmap(
        bed_path, dtype=np.uint8, mode="r", offset=3,
        shape=(n_bim, row_bytes),
    )

    ids_parts, chrom_parts, pos_parts = [], [], []
    counted_parts, other_parts, variant_block_parts = [], [], []
    pair_i_parts, pair_j_parts, pair_block_parts = [], [], []
    linear_parts, quadratic_parts, bilinear_parts = [], [], []
    block_source, block_chrom, block_start, block_end = [], [], [], []
    empirical_differences = []
    offset = 0
    started = time.perf_counter()
    for output_block, (source_block, chromosome, start, end, indices) in enumerate(pruned):
        if output_block % 25 == 0:
            print(
                f"LD design: block {output_block + 1}/{len(pruned)}, "
                f"chromosome {chromosome}, elapsed "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
        ids = panel.ids[indices].astype(str)
        records = [bim[rsid] for rsid in ids]
        rows = np.asarray([record[0] for record in records], dtype=np.int64)
        genotypes = decode_plink_rows(np.asarray(bed[rows]), len(sample_ids))
        counted = panel.counted_allele[indices].astype(str)
        other = panel.other_allele[indices].astype(str)
        for j, (record, canonical, alternate) in enumerate(
                zip(records, counted, other, strict=True)):
            _, bim_chrom, bim_pos, a1, a2 = record
            if bim_chrom != chromosome or bim_pos != int(panel.pos[indices[j]]):
                raise ValueError(f"coordinate mismatch for {ids[j]}")
            if {a1, a2} != {canonical, alternate}:
                raise ValueError(f"allele mismatch for {ids[j]}")
            if canonical == a2:
                called = np.isfinite(genotypes[:, j])
                genotypes[called, j] = 2.0 - genotypes[called, j]
            elif canonical != a1:
                raise ValueError(f"cannot orient {ids[j]} to counted allele")
        references, empirical_af = _correlations(genotypes, sample_pop)
        empirical_differences.append(empirical_af - panel.af[indices])
        ii, jj, linear, quadratic, bilinear = compact_block_design(
            references, floor=ld_floor, cap=max_pairs_per_block
        )
        ids_parts.append(ids)
        chrom_parts.append(panel.chrom[indices].astype(str))
        pos_parts.append(panel.pos[indices].astype(np.int64))
        counted_parts.append(counted)
        other_parts.append(other)
        variant_block_parts.append(
            np.full(indices.size, output_block, dtype=np.int64)
        )
        pair_i_parts.append(ii + offset)
        pair_j_parts.append(jj + offset)
        pair_block_parts.append(
            np.full(ii.size, output_block, dtype=np.int64)
        )
        linear_parts.append(linear.astype(np.float32))
        quadratic_parts.append(quadratic.astype(np.float32))
        bilinear_parts.append(bilinear.astype(np.float32))
        block_source.append(source_block)
        block_chrom.append(int(chromosome))
        block_start.append(start)
        block_end.append(end)
        offset += indices.size

    differences = np.concatenate(empirical_differences)
    fields = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "populations": np.asarray(POPS),
        "sample_counts": sample_counts,
        "ids": np.concatenate(ids_parts),
        "chrom": np.concatenate(chrom_parts),
        "pos": np.concatenate(pos_parts),
        "counted_allele": np.concatenate(counted_parts),
        "other_allele": np.concatenate(other_parts),
        "variant_block": np.concatenate(variant_block_parts),
        "block_source_id": np.asarray(block_source, dtype=np.int64),
        "block_chrom": np.asarray(block_chrom, dtype=np.int64),
        "block_start": np.asarray(block_start, dtype=np.int64),
        "block_end": np.asarray(block_end, dtype=np.int64),
        "pair_i": np.concatenate(pair_i_parts),
        "pair_j": np.concatenate(pair_j_parts),
        "pair_block": np.concatenate(pair_block_parts),
        "pair_design": np.concatenate(linear_parts),
        "quadratic_design": np.concatenate(quadratic_parts),
        "bilinear_design": np.concatenate(bilinear_parts),
        "maf_threshold": np.asarray(maf_threshold, dtype=np.float64),
        "min_block_gap_bp": np.asarray(min_block_gap_bp, dtype=np.int64),
        "min_variants_per_block": np.asarray(
            min_variants_per_block, dtype=np.int64),
        "max_variants_per_block": np.asarray(
            max_variants_per_block, dtype=np.int64),
        "ld_floor": np.asarray(ld_floor, dtype=np.float64),
        "max_pairs_per_block": np.asarray(max_pairs_per_block, dtype=np.int64),
        "n_common_before_pruning": np.asarray(common.sum(), dtype=np.int64),
        "n_source_blocks": np.asarray(len(geometry["sizes"]), dtype=np.int64),
        "source": np.asarray(
            "1000 Genomes Project Phase 3; Privé HM3/UKB PLINK release v4"),
        "source_url": np.asarray(SOURCE_URL),
        "source_archive_md5": np.asarray(SOURCE_ARCHIVE_MD5),
        "ld_geometry_sha256": np.asarray(LD_GEOMETRY_SHA256),
        "ld_geometry_source_revision": np.asarray(
            LD_GEOMETRY_SOURCE_REVISION),
        "ld_geometry_role": np.asarray(
            "variant order, alleles, and 625 HM3 block sizes only"),
        "af_panel_sha256": np.asarray(panel.panel_sha256),
        "genome_build": np.asarray(panel.genome_build),
        "max_empirical_af_difference": np.asarray(
            np.max(np.abs(differences)), dtype=np.float64),
    }
    fields["semantic_sha256"] = np.asarray(_semantic_digest(fields))
    return fields


def write_design(path: Path, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **fields)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def print_summary(fields, path=None):
    print("Table 1. Fixed common-variant, distant-block LD design")
    print()
    print("| Quantity | Value |")
    print("|---|---:|")
    rows = (
        ("Common HM3 SNPs before pruning", int(fields["n_common_before_pruning"])),
        ("Retained distant blocks", len(fields["block_source_id"])),
        ("Retained variants", len(fields["ids"])),
        ("Retained pair products", len(fields["pair_i"])),
        ("Minimum edge-to-edge block gap (bp)", int(fields["min_block_gap_bp"])),
        ("Maximum variants per block", int(fields["max_variants_per_block"])),
        ("Maximum empirical AF difference", f"{float(fields['max_empirical_af_difference']):.6g}"),
    )
    for label, value in rows:
        print(f"| {label} | {value} |")
    counts = "/".join(map(str, fields["sample_counts"].tolist()))
    print(f"| AFR/AMR/EAS/EUR/SAS samples | {counts} |")
    print(f"\nSemantic SHA-256: {fields['semantic_sha256'].item()}")
    if path is not None:
        print(f"Artifact: {path}")
        print(f"File SHA-256: {file_digest(path)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--af-panel", type=Path, default=DEFAULT_AF_PANEL)
    parser.add_argument("--ld-geometry", type=Path, default=DEFAULT_LD_GEOMETRY)
    parser.add_argument("--bed-prefix", type=Path, default=DEFAULT_BED_PREFIX)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maf", type=float, default=0.05)
    parser.add_argument("--min-block-gap-bp", type=int, default=5_000_000)
    parser.add_argument("--min-variants-per-block", type=int, default=32)
    parser.add_argument("--max-variants-per-block", type=int, default=128)
    parser.add_argument("--ld-floor", type=float, default=0.05)
    parser.add_argument("--max-pairs-per-block", type=int, default=250)
    args = parser.parse_args(argv)
    if not 0.0 < args.maf < 0.5:
        parser.error("--maf must lie strictly between 0 and 0.5")
    if (args.min_block_gap_bp < 0 or args.min_variants_per_block < 2 or
            args.max_variants_per_block < 2 or args.max_pairs_per_block < 1 or
            not 0.0 <= args.ld_floor <= 1.0):
        parser.error("invalid fixed pruning or LD-pair settings")
    try:
        fields = build_design(
            panel_path=args.af_panel,
            geometry_path=args.ld_geometry,
            bed_prefix=args.bed_prefix,
            archive_path=args.archive,
            maf_threshold=args.maf,
            min_block_gap_bp=args.min_block_gap_bp,
            min_variants_per_block=args.min_variants_per_block,
            max_variants_per_block=args.max_variants_per_block,
            ld_floor=args.ld_floor,
            max_pairs_per_block=args.max_pairs_per_block,
        )
        write_design(args.out, fields)
        print_summary(fields, args.out)
        return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
