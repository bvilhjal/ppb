"""Tests for allele harmonization and high-level evaluation."""

import numpy as np
import pytest

from ppb import DenseLD, VariantTable, evaluate, harmonize_to, r2


def _ref():
    # 4 canonical variants; #4 (pos 400) is palindromic (A/T). The rest are
    # non-palindromic so exact/swap/strand cases are unambiguous.
    return VariantTable(
        chrom=[1, 1, 2, 2],
        pos=[100, 200, 300, 400],
        a1=["A", "C", "G", "A"],
        a2=["G", "T", "A", "T"],
    )


def test_exact_match_keeps_values_and_order():
    ref = _ref()
    tgt = VariantTable([1, 2], [100, 300], ["A", "G"], ["G", "A"])
    aligned, rep = harmonize_to(ref, tgt, [0.5, -0.3])
    assert np.allclose(aligned, [0.5, 0.0, -0.3, 0.0])
    assert rep.n_matched == 2 and rep.n_sign_flipped == 0


def test_allele_swap_flips_sign():
    ref = _ref()
    tgt = VariantTable([1], [100], ["G"], ["A"])          # a1/a2 swapped vs ref
    aligned, rep = harmonize_to(ref, tgt, [0.5])
    assert np.allclose(aligned, [-0.5, 0.0, 0.0, 0.0])
    assert rep.n_sign_flipped == 1 and rep.n_strand_flipped == 0


def test_strand_flip_keeps_sign():
    ref = _ref()
    tgt = VariantTable([1], [100], ["T"], ["C"])          # complement of A/G
    aligned, rep = harmonize_to(ref, tgt, [0.5])
    assert np.allclose(aligned, [0.5, 0.0, 0.0, 0.0])
    assert rep.n_strand_flipped == 1 and rep.n_sign_flipped == 0


def test_strand_flip_plus_swap_flips_sign():
    ref = _ref()
    tgt = VariantTable([1], [100], ["C"], ["T"])          # complement of G/A (swapped)
    aligned, rep = harmonize_to(ref, tgt, [0.5])
    assert np.allclose(aligned, [-0.5, 0.0, 0.0, 0.0])
    assert rep.n_strand_flipped == 1 and rep.n_sign_flipped == 1


def test_palindromic_removed_by_default():
    ref = _ref()
    tgt = VariantTable([2], [400], ["A"], ["T"])          # ambiguous A/T
    aligned, rep = harmonize_to(ref, tgt, [0.9])
    assert np.allclose(aligned, 0.0)
    assert rep.n_ambiguous_removed == 1 and rep.n_matched == 0


def test_palindromic_kept_when_disabled():
    ref = _ref()
    tgt = VariantTable([2], [400], ["A"], ["T"])
    aligned, rep = harmonize_to(ref, tgt, [0.9], remove_ambiguous=False)
    assert np.allclose(aligned, [0.0, 0.0, 0.0, 0.9])
    assert rep.n_ambiguous_removed == 0 and rep.n_matched == 1


def test_allele_mismatch_is_counted_as_mismatch():
    ref = _ref()
    tgt = VariantTable([1], [200], ["A"], ["C"])          # pos 200 is C/T in ref
    aligned, rep = harmonize_to(ref, tgt, [1.0])
    assert np.allclose(aligned, 0.0)
    assert rep.n_mismatch == 1 and rep.n_unmatched == 0 and rep.n_matched == 0


def test_missing_position_is_unmatched():
    ref = _ref()
    tgt = VariantTable([9], [999], ["A"], ["G"])
    _, rep = harmonize_to(ref, tgt, [1.0])
    assert rep.n_unmatched == 1 and rep.n_mismatch == 0


def test_chrom_naming_mismatch_still_matches():
    """'chr1'/'1' and 'X'/'23' labelling differences must not break matching."""
    ref = VariantTable(["chr1", "23"], [100, 500], ["A", "C"], ["G", "T"])
    tgt = VariantTable([1, "X"], [100, 500], ["A", "C"], ["G", "T"])
    aligned, rep = harmonize_to(ref, tgt, [0.5, 0.7])
    assert np.allclose(aligned, [0.5, 0.7])
    assert rep.n_matched == 2 and rep.n_unmatched == 0


def test_indel_reverse_complement_strand_flip():
    """Multi-base indels strand-resolve via reverse-complement."""
    ref = VariantTable([1], [100], ["AG"], ["C"])
    tgt = VariantTable([1], [100], ["CT"], ["G"])         # reverse-complement of AG / C
    aligned, rep = harmonize_to(ref, tgt, [0.5])
    assert np.allclose(aligned, [0.5])
    assert rep.n_matched == 1 and rep.n_strand_flipped == 1 and rep.n_sign_flipped == 0


def test_value_length_mismatch_raises():
    ref = _ref()
    tgt = VariantTable([1], [100], ["A"], ["G"])
    with pytest.raises(ValueError):
        harmonize_to(ref, tgt, [1.0, 2.0])


def test_evaluate_matches_prealigned_r2():
    """End-to-end: harmonizing swapped + reordered inputs recovers the same R^2
    as evaluating already-aligned vectors."""
    rng = np.random.default_rng(0)
    m = 6
    ref = VariantTable(                                    # all non-palindromic
        chrom=np.ones(m, dtype=int), pos=np.arange(1, m + 1),
        a1=["A"] * m, a2=["G", "C", "G", "C", "G", "C"])
    A = rng.standard_normal((400, m))
    Xs = (A - A.mean(0)) / A.std(0)
    ld = DenseLD((Xs.T @ Xs) / Xs.shape[0])
    w = rng.standard_normal(m)
    z = rng.standard_normal(m) * 0.1
    pre = r2(w, z, ld)

    # Present the same variants reordered, with variant 0's alleles swapped
    # (so its weight sign must flip back during harmonization).
    order = np.array([3, 0, 5, 1, 4, 2])
    a1s, a2s = np.array(ref.a1), np.array(ref.a2)
    a1s[0], a2s[0] = ref.a2[0], ref.a1[0]
    tw = w.copy(); tw[0] = -tw[0]
    weights_tbl = VariantTable(np.ones(m, dtype=int)[order], np.arange(1, m + 1)[order],
                               a1s[order], a2s[order])
    sumstats_tbl = VariantTable(np.ones(m, dtype=int), np.arange(1, m + 1),
                                np.array(ref.a1), np.array(ref.a2))

    res = evaluate(ld, ref, weights_tbl, tw[order], sumstats_tbl, z)
    assert abs(res.r2 - pre) <= 1e-9
    assert res.n_variants_scored == m
    assert res.weights_report["n_matched"] == m
    assert res.weights_report["n_sign_flipped"] == 1
    assert res.weights_report["n_ambiguous_removed"] == 0
