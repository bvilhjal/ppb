"""Contracts for the real-data negative-control sweep.

The sweep itself needs the LD reference and the consortium targets, so the data
access is monkeypatched (as in ``test_regenerate_results.py``) and what is
tested is the logic: that every cell gets its own joint support, that the
diagonal separates from the off-diagonal on planted signal, and that an
artifact-shaped result is reported as a problem rather than printed and ignored.
"""

import numpy as np
import pytest

from ppb.harmonize import VariantTable
from scripts import negative_controls as nc


def _variants(positions):
    n = len(positions)
    return VariantTable(
        np.repeat("1", n), np.asarray(positions),
        np.repeat("A", n), np.repeat("C", n))


class _IdentityBlock:
    m = 4

    @staticmethod
    def quad(x):
        return float(x @ x)


def _planted(traits, signal, cross=()):
    """Per-block products with a planted diagonal and declared cross-signal."""
    rng = np.random.default_rng(0)
    n_blocks = 120
    per_block = {}
    cross = {frozenset(pair): value for pair, value in cross}
    for score in traits:
        for target in traits:
            v = rng.uniform(0.5, 1.5, n_blocks)
            if score == target:
                u = rng.normal(signal, 0.2 * signal, n_blocks)
            else:
                shared = cross.get(frozenset((score, target)), 0.0)
                u = rng.normal(shared, 0.2 * signal, n_blocks)
            per_block[(score, target)] = (u, v)
    return per_block


def test_diagonal_separates_from_the_off_diagonal():
    traits = ["height", "BMI", "LDL"]
    cells = nc.build_matrix(_planted(traits, signal=0.05), traits)
    lookup = {(c["score"], c["target"]): c for c in cells}

    assert len(cells) == 9
    # z is bounded by sqrt(n_blocks) = sqrt(120) = 10.95, attained when every
    # block agrees exactly, so "close to the ceiling" is the strong statement.
    for score in traits:
        matched = lookup[(score, score)]
        assert matched["z"] > 9.0
        assert matched["ratio"] > 10.0
        for target in traits:
            if target == score:
                continue
            mismatched = lookup[(score, target)]
            assert mismatched["r2"] < matched["r2"]
            assert abs(mismatched["z"]) < 5.0

    assert nc.summarize(cells, traits) == []


def test_a_genuinely_correlated_pair_is_declared_not_discovered():
    """BMI x T2D is expected to be non-zero; that must not read as an artifact,
    and the expectation is declared up front rather than fitted after the fact."""
    traits = ["BMI", "T2D", "height"]
    per_block = _planted(traits, signal=0.05,
                         cross=((("BMI", "T2D"), 0.02),))
    cells = nc.build_matrix(per_block, traits)
    lookup = {(c["score"], c["target"]): c for c in cells}

    assert lookup[("BMI", "T2D")]["expected_correlated"] is True
    assert lookup[("BMI", "T2D")]["z"] > 8.0
    assert lookup[("BMI", "height")]["expected_correlated"] is False
    # A declared correlation is not reported as a problem.
    assert nc.summarize(cells, traits) == []


def test_unexpected_off_diagonal_signal_is_reported_as_a_problem():
    """Height x LDL has no plausible shared aetiology: strong signal there is an
    artifact, and the control must say so rather than print it quietly."""
    traits = ["height", "LDL", "BMI"]
    per_block = _planted(traits, signal=0.05,
                         cross=((("height", "LDL"), 0.04),))
    problems = nc.summarize(nc.build_matrix(per_block, traits), traits)

    assert problems, "an unexplained off-diagonal must be reported"
    assert any("height" in p and "LDL" in p for p in problems)


def test_a_mismatched_target_outscoring_its_own_is_a_problem():
    traits = ["height", "BMI"]
    per_block = _planted(traits, signal=0.01,
                         cross=((("height", "BMI"), 0.05),))
    problems = nc.summarize(nc.build_matrix(per_block, traits), traits)
    assert any("scores higher" in p for p in problems)


def test_expected_correlated_pairs_are_symmetric():
    for pair in nc.EXPECTED_CORRELATED:
        a, b = tuple(pair)
        assert frozenset((b, a)) in nc.EXPECTED_CORRELATED


def test_sweep_gives_every_cell_its_own_joint_support(monkeypatch):
    """A target missing a variant must remove it from that cell's numerator and
    denominator only -- never from another cell that does have it."""
    reference = _variants([1, 2, 3, 4])
    monkeypatch.setattr(nc, "CHROMS", ["1"])
    monkeypatch.setattr(nc, "TRAITS", {
        "A": dict(pgs="PGS1", consortium="A"),
        "B": dict(pgs="PGS2", consortium="B"),
    })
    monkeypatch.setattr(nc, "_trait_type", lambda trait: "quantitative")
    monkeypatch.setattr(
        nc, "read_weights",
        lambda path: (reference, np.array([1.0, 1.0, 1.0, 1.0])))
    # Target A covers every variant; target B is missing the last one.
    targets = {
        "A": (reference, np.array([0.1, 0.1, 0.1, 0.1]), {}),
        "B": (_variants([1, 2, 3]), np.array([0.1, 0.1, 0.1]), {}),
    }
    monkeypatch.setattr(
        nc, "load_target",
        lambda path, **kw: targets["A" if str(path).count("A_") else "B"])
    monkeypatch.setattr(nc, "read_ldref", lambda path: {
        "variants": reference,
        "af": np.full(4, 0.5),
        "ld": type("LD", (), {"blocks": [(_IdentityBlock(), np.arange(4))]})(),
    })

    per_block, chrom = nc.sweep_matrix(["A", "B"], progress=False)

    assert list(chrom) == ["1"]
    assert set(per_block) == {("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")}
    # sd = sqrt(2 * 0.5 * 0.5) = 0.7071 per variant, weights all 1.
    sd = np.sqrt(0.5)
    for score in ("A", "B"):
        u_full, v_full = per_block[(score, "A")]
        assert u_full[0] == pytest.approx(4 * sd * 0.1)
        assert v_full[0] == pytest.approx(4 * sd * sd)
        u_short, v_short = per_block[(score, "B")]
        assert u_short[0] == pytest.approx(3 * sd * 0.1)
        assert v_short[0] == pytest.approx(3 * sd * sd)


def test_sweep_rejects_a_reference_without_usable_allele_frequencies(monkeypatch):
    reference = _variants([1, 2, 3, 4])
    monkeypatch.setattr(nc, "CHROMS", ["1"])
    monkeypatch.setattr(nc, "TRAITS", {"A": dict(pgs="PGS1", consortium="A")})
    monkeypatch.setattr(nc, "_trait_type", lambda trait: "quantitative")
    monkeypatch.setattr(nc, "read_weights", lambda path: (reference, np.ones(4)))
    monkeypatch.setattr(nc, "load_target",
                        lambda path, **kw: (reference, np.full(4, 0.1), {}))
    ld = type("LD", (), {"blocks": [(_IdentityBlock(), np.arange(4))]})()

    monkeypatch.setattr(nc, "read_ldref",
                        lambda path: {"variants": reference, "ld": ld})
    with pytest.raises(ValueError, match="no 'af' annotation"):
        nc.sweep_matrix(["A"], progress=False)

    monkeypatch.setattr(nc, "read_ldref", lambda path: {
        "variants": reference, "ld": ld,
        "af": np.array([0.5, 0.0, 0.5, 0.5])})
    with pytest.raises(ValueError, match=r"af outside \(0, 1\)"):
        nc.sweep_matrix(["A"], progress=False)


def test_script_is_runnable_as_documented(tmp_path):
    """``python scripts/negative_controls.py`` puts scripts/ on sys.path, not the
    repository root, so the sibling-module import needs help. Run it from an
    unrelated directory so a stray cwd cannot mask the failure."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "negative_controls.py"
    done = subprocess.run([sys.executable, str(script), "--help"],
                          cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "Negative controls" in done.stdout
