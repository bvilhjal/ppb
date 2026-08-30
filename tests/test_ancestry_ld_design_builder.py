"""Unit tests for the compact 1000G LD-design builder."""

import numpy as np

from ppb.ancestry import bilinear_ld_scores, pair_design
from scripts import build_ancestry_ld_design as builder


def _correlation(seed, m=7):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((30, m))
    return np.corrcoef(values, rowvar=False)


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
