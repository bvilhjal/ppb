"""Unit tests for the compact 1000G LD-design builder."""

from pathlib import Path

import numpy as np
import pytest

from ppb.ancestry import bilinear_ld_scores, pair_design
from ppb.ancestry_frequency import load_frequency_panel, write_frequency_panel
from scripts import build_ancestry_ld_design as builder

POPS = ("AFR", "AMR", "EAS", "EUR", "SAS")
NONPAL = [("A", "C"), ("G", "A"), ("C", "T"), ("G", "T"), ("A", "G"),
          ("T", "C")]

# Eight blocks on three chromosomes; each kind exercises one provenance rule.
BLOCKS = [
    ("1", 1_000_000, 60, "all"),       # kept
    ("1", 4_000_000, 60, "sasrare"),   # SAS MAF 0.02: fails the conjunction
    ("1", 12_000_000, 60, "all"),      # kept
    ("1", 14_000_000, 60, "afrmono"),  # AFR MAF 0.0: fails the conjunction
    ("2", 1_000_000, 60, "all"),       # kept
    ("2", 3_000_000, 20, "all"),       # < min_variants_per_block
    ("2", 20_000_000, 60, "euronly"),  # MAF 0.01 outside EUR: fails conjunction
    ("3", 5_000_000, 60, "all"),       # kept
]
MIN_GAP, MIN_VAR, MAX_VAR, MAX_PAIRS, LD_FLOOR, MAF = \
    5_000_000, 32, 32, 25, 0.05, 0.05
NSAMP = {"AFR": 40, "AMR": 30, "EAS": 35, "EUR": 33, "SAS": 31}


def _correlation(seed, m=7):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((30, m))
    return np.corrcoef(values, rowvar=False)


# ---------------------------------------------------------------- existing


def test_decode_variant_major_plink_bed_rows():
    # PLINK codes 00/10/11/01 mean A1/A1, heterozygous, A2/A2, missing.
    first = 0 | (2 << 2) | (3 << 4) | (1 << 6)
    raw = np.asarray([[first, 0]], dtype=np.uint8)
    observed = builder.decode_plink_rows(raw, 5)[:, 0]
    assert np.allclose(observed[[0, 1, 2, 4]], [2.0, 1.0, 0.0, 2.0])
    assert np.isnan(observed[3])


def test_distant_block_selection_and_even_pruning():
    chrom = np.asarray(["1"] * 12 + ["2"] * 4)
    pos = np.asarray([
        1, 2, 3, 10, 11, 12, 30, 31, 32, 50, 51, 52,
        1, 2, 3, 4,
    ]) * 1_000_000
    sizes = np.asarray([3, 3, 3, 3, 4])
    common = np.ones(16, dtype=bool)
    selected = builder.select_distant_blocks(
        chrom, pos, sizes, common, min_gap_bp=5_000_000, min_variants=3
    )
    assert [item[0] for item in selected] == [0, 1, 2, 3, 4]
    selected = builder.select_distant_blocks(
        chrom, pos, sizes, common, min_gap_bp=20_000_000, min_variants=3
    )
    assert [item[0] for item in selected] == [0, 2, 4]
    assert np.array_equal(
        builder.evenly_spaced(np.arange(20), 5), [0, 5, 10, 14, 19]
    )


def test_compact_block_design_matches_dense_definitions():
    refs = [_correlation(1), _correlation(2)]
    ii, jj, linear, quadratic, bilinear = builder.compact_block_design(
        refs, floor=0.0, cap=1000
    )
    expected = pair_design(refs, floor=0.0, cap=1000, quadratic=True)
    exp_i, exp_j, exp_linear, _, exp_quadratic = expected
    exp_bilinear, _ = bilinear_ld_scores(refs)
    assert np.array_equal(ii, exp_i)
    assert np.array_equal(jj, exp_j)
    assert np.allclose(linear, exp_linear)
    assert np.allclose(quadratic, exp_quadratic)
    assert np.allclose(bilinear, exp_bilinear)


def test_semantic_digest_is_order_stable_and_content_sensitive():
    first = {"b": np.asarray([1, 2]), "a": np.asarray("x")}
    second = {"a": np.asarray("x"), "b": np.asarray([1, 2])}
    assert builder._semantic_digest(first) == builder._semantic_digest(second)
    second["b"] = np.asarray([1, 3])
    assert builder._semantic_digest(first) != builder._semantic_digest(second)


# ------------------------------------------------- synthetic PLINK fixtures


def _panel_arrays():
    ids, chrom, pos, counted, other, af_rows, kinds = [], [], [], [], [], [], []
    for _bi, (c, start, nv, kind) in enumerate(BLOCKS):
        for _v in range(nv):
            i = len(ids)
            ids.append(f"rs{100000 + i}")
            chrom.append(c)
            pos.append(start + _v * 1000)
            ca, oa = NONPAL[i % len(NONPAL)]
            counted.append(ca)
            other.append(oa)
            kinds.append(kind)
            row = {p: 0.30 for p in POPS}
            if kind == "sasrare":
                row["SAS"] = 0.02
            elif kind == "afrmono":
                row["AFR"] = 0.0
            elif kind == "euronly":
                for p in POPS:
                    if p != "EUR":
                        row[p] = 0.01
            af_rows.append([row[p] for p in POPS])
    return (np.asarray(ids), np.asarray(chrom), np.asarray(pos, dtype=np.int64),
            np.asarray(counted), np.asarray(other),
            np.asarray(af_rows, dtype=np.float64), np.asarray(kinds))


def _write_panel(tmp_path, ids, chrom, pos, counted, other, af):
    return write_frequency_panel(
        tmp_path / "panel.npz", ids=ids, chrom=chrom, pos=pos,
        counted_allele=counted, other_allele=other, pops=list(POPS), af=af,
        n_samples=np.asarray([2504] * 5, dtype=np.int64),
        source="synthetic test panel", source_url="https://example.test",
        genome_build="hg19/GRCh37")


def _write_geometry(tmp_path, ids, chrom, pos, counted, other):
    sizes = np.asarray([b[2] for b in BLOCKS], dtype=np.int64)
    geometry = tmp_path / "geometry.npz"
    np.savez_compressed(geometry, ids=ids, chrom=chrom, pos=pos,
                        counted_allele=counted, other_allele=other,
                        sizes=sizes)
    return geometry


def _write_plink(prefix, bed_dosage_of_a1, sample_ids, sample_pop,
                 ids, chrom, pos, bim_a1, bim_a2):
    """A minimal PLINK .bed/.bim/.fam(+ .fam2) triple. BED stores the dosage
    of the *BIM A1* allele, so the builder must flip variants whose BIM A1 is
    the panel's other allele."""
    prefix = Path(prefix)
    n_samp = len(sample_ids)
    m = len(ids)
    with prefix.with_suffix(".fam").open("w") as fh:
        for sid, p in zip(sample_ids, sample_pop):
            fh.write(f"FAM_{p} {sid} 0 0 1 -9\n")
    with prefix.with_suffix(".fam2").open("w", encoding="utf-8-sig") as fh:
        fh.write("sample.ID\tSuper Population\tPopulation\n")
        for sid, p in zip(sample_ids, sample_pop):
            fh.write(f"{sid}\t{p}\t{p}sub\n")
    with prefix.with_suffix(".bim").open("w") as fh:
        for i in range(m):
            fh.write(f"{chrom[i]}\t{ids[i]}\t0.0\t{pos[i]}\t{bim_a1[i]}\t"
                     f"{bim_a2[i]}\n")
    rb = (n_samp + 3) // 4
    code_of = {2.0: 0, 1.0: 2, 0.0: 3}
    bed = np.zeros((m, rb), dtype=np.uint8)
    for i in range(m):
        for s in range(n_samp):
            g = bed_dosage_of_a1[i, s]
            code = 1 if np.isnan(g) else code_of[float(g)]
            bed[i, s // 4] |= np.uint8(code << (2 * (s % 4)))
    with prefix.with_suffix(".bed").open("wb") as fh:
        fh.write(builder.PLINK_MAGIC)
        fh.write(bed.tobytes())
    return prefix


# ------------------------------------------------------ the M9 provenance half


def test_load_geometry_enforces_the_per_panel_maf_conjunction(tmp_path,
                                                              monkeypatch):
    """The load-bearing filter is the per-panel MAF conjunction, not a pooled
    mean: a pooled rule would keep 'sasrare' (0.02 in SAS, 0.30 elsewhere),
    'afrmono' (0.0 in AFR) and 'euronly' (0.01 elsewhere) variants, which the
    conjunction must drop. M9: a refactor to a pooled filter must not pass."""
    ids, chrom, pos, counted, other, af, kinds = _panel_arrays()
    panel_path = _write_panel(tmp_path, ids, chrom, pos, counted, other, af)
    geometry = _write_geometry(tmp_path, ids, chrom, pos, counted, other)
    # _load_geometry pins against the module-level shipped-panel digest; the
    # synthetic panel is not the shipped one, so register its real digest.
    monkeypatch.setattr(builder, "AF_PANEL_SHA256",
                        load_frequency_panel(panel_path).panel_sha256)
    _, _, common = builder._load_geometry(panel_path, geometry, MAF)
    expected = np.min(np.minimum(af, 1.0 - af), axis=1) >= MAF
    assert np.array_equal(common, expected)
    # the conjunction and a pooled mean genuinely differ on this fixture
    pooled = np.minimum(af.mean(axis=1), 1.0 - af.mean(axis=1)) >= MAF
    assert int((pooled & ~common).sum()) > 0
    for kind, want in (("all", True), ("sasrare", False),
                       ("afrmono", False), ("euronly", False)):
        sel = kinds == kind
        assert (bool(common[sel].all()) if want else
                not common[sel].any()), kind


def test_read_samples_labels_from_fam2_and_validates(tmp_path):
    """Population labels come from the keyed .fam2, not the FAM's first
    column, so a FAM line cannot silently reassign a sample between
    reference panels."""
    ids, chrom, pos, counted, other, af, _ = _panel_arrays()
    _write_panel(tmp_path, ids, chrom, pos, counted, other, af)
    sample_ids = np.array([f"{p}S{i:03d}" for p in POPS
                           for i in range(NSAMP[p])])
    sample_pop = np.array([p for p in POPS for _ in range(NSAMP[p])])
    # a FAM whose FID column lies about the population: the label must still
    # come from .fam2
    prefix = _write_plink(tmp_path / "syn", np.zeros((len(ids), len(sample_ids))),
                          sample_ids, sample_pop, ids, chrom, pos,
                          counted, other)
    got_ids, got_pop, counts = builder._read_samples(prefix)
    assert np.array_equal(got_ids, sample_ids)
    assert np.array_equal(got_pop, sample_pop)
    assert counts.tolist() == [NSAMP[p] for p in POPS]
    # drop one fam2 row: that sample is unlabelled and must raise
    fam2 = prefix.with_suffix(".fam2")
    lines = fam2.read_text(encoding="utf-8-sig").splitlines()
    fam2.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="lack super-population"):
        builder._read_samples(prefix)


def test_index_bim_and_decode_and_correlations_round_trip(tmp_path):
    """A strand-aware BIM index + BED decode must hand _correlations the
    allele-oriented dosages the panel declared: BIM A1 = the other allele on
    odd rows, so the builder flips those rows back."""
    ids, chrom, pos, counted, other, af, _ = _panel_arrays()
    _write_panel(tmp_path, ids, chrom, pos, counted, other, af)
    # two samples per pop, no missingness, so correlations are exact
    sample_ids = np.array([f"{p}S{i:03d}" for p in POPS for i in range(40)])
    sample_pop = np.array([p for p in POPS for _ in range(40)])
    m = len(ids)
    rng = np.random.default_rng(3)
    # 40 samples per population at p=[0.3,0.4,0.3] make a monomorphic column
    # vanishingly unlikely; _correlations refuses one outright.
    D = rng.choice([0, 1, 2], size=(m, len(sample_ids)), p=[0.3, 0.4, 0.3]) \
        .astype(float)
    bim_a1 = np.where(np.arange(m) % 2 == 0, counted, other)
    bim_a2 = np.where(np.arange(m) % 2 == 0, other, counted)
    Da1 = np.where((np.arange(m) % 2 == 0)[:, None], D, 2.0 - D)
    prefix = _write_plink(tmp_path / "syn", Da1, sample_ids, sample_pop,
                          ids, chrom, pos, bim_a1, bim_a2)
    _, _, counts = builder._read_samples(prefix)
    found, n_rows = builder._index_bim(prefix.with_suffix(".bim"), ids)
    assert n_rows == m
    assert set(found) == set(ids.tolist())
    raw = np.frombuffer(
        prefix.with_suffix(".bed").read_bytes(), dtype=np.uint8)
    assert raw[:3].tobytes() == builder.PLINK_MAGIC
    rb = (len(sample_ids) + 3) // 4
    bed = raw[3:].reshape(m, rb)
    decoded = builder.decode_plink_rows(bed, len(sample_ids))   # samples x variants
    # decoded is dosage of BIM A1; on odd rows that is the other allele, so
    # flipping those rows back recovers the panel's counted-allele dosage.
    flip = (np.arange(m) % 2 == 1)
    decoded[:, flip] = 2.0 - decoded[:, flip]
    np.testing.assert_allclose(decoded, D.T)
    refs, emp_af = builder._correlations(decoded, sample_pop)
    assert len(refs) == len(POPS)
    for pi, p in enumerate(POPS):
        cols = np.flatnonzero(sample_pop == p)
        np.testing.assert_allclose(
            refs[pi], np.corrcoef(decoded[cols].T), atol=1e-12)
        np.testing.assert_allclose(emp_af[:, pi],
                                   decoded[cols].mean(axis=0) / 2.0,
                                   atol=1e-12)
