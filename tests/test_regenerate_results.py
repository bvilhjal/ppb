"""Focused contracts for the real-data result regeneration path."""

import numpy as np
import pytest

from ppb.harmonize import VariantTable
from scripts import regenerate_results as regenerate


def _variants(positions):
    n = len(positions)
    return VariantTable(
        np.repeat("1", n), np.asarray(positions),
        np.repeat("A", n), np.repeat("C", n),
    )


class _IdentityBlock:
    @staticmethod
    def quad(x):
        return float(x @ x)


class _OneBlockLD:
    blocks = [(_IdentityBlock(), np.arange(3))]


def test_no_n_target_records_trait_specific_sample_size_basis(tmp_path):
    path = tmp_path / "target.tsv"
    path.write_text(
        "chrom\tpos\ta1\ta2\tbeta\tse\n1\t1\tA\tC\t0.1\t0.02\n",
        encoding="utf-8",
    )

    _, _, quantitative = regenerate.load_target(
        path, n_eff=1000, trait_type="quantitative")
    _, _, binary = regenerate.load_target(
        path, n_eff=800, trait_type="binary")

    assert quantitative["n_eff_basis"] == "trait-level sample size"
    assert binary["n_eff_basis"] == (
        "trait-level effective N (4/(1/n_case + 1/n_ctrl) for binary traits)")


def test_sweep_uses_each_targets_joint_variant_support(monkeypatch):
    reference = _variants([1, 2, 3])
    weights = np.array([1.0, 2.0, 3.0])
    monkeypatch.setattr(regenerate, "CHROMS", ["1"])
    monkeypatch.setattr(
        regenerate, "read_weights", lambda path: (reference, weights))
    monkeypatch.setattr(
        regenerate,
        "read_ldref",
        lambda path: {
            "variants": reference,
            "af": np.repeat(0.5, 3),
            "ld": _OneBlockLD(),
        },
    )
    targets = {
        "partial": (_variants([1, 3]), np.array([10.0, 30.0]), {}),
        "complete": (reference, np.array([10.0, 20.0, 30.0]), {}),
    }

    blocks, totals = regenerate.sweep("PGS000000", targets)

    # Dosage weights are multiplied by sqrt(2p(1-p)) = sqrt(1/2).
    assert blocks["u"]["partial"][0] == pytest.approx(100.0 / np.sqrt(2.0))
    assert blocks["v"]["partial"][0] == pytest.approx(5.0)
    assert blocks["u"]["complete"][0] == pytest.approx(140.0 / np.sqrt(2.0))
    assert blocks["v"]["complete"][0] == pytest.approx(7.0)
    assert totals["n_variants_scored"] == {"partial": 2, "complete": 3}


def test_build_records_fails_closed_and_labels_metric_scale(monkeypatch):
    target = (_variants([1]), np.array([0.1]), {
        "n_eff": 1000,
        "n_eff_basis": "test sample size",
    })
    monkeypatch.setattr(
        regenerate, "load_target",
        lambda path, n_eff=None, trait_type=None: target,
    )

    def fake_sweep(pgs, targets):
        u = {name: np.array([1.0 if name == "consortium" else 3.0])
             for name in targets}
        v = {name: np.array([4.0 if name == "consortium" else 9.0])
             for name in targets}
        totals = {
            "w_matched": 8,
            "w_total": 10,
            "z_matched": {name: 1 for name in targets},
            "z_total": {name: 1 for name in targets},
            "n_variants_scored": {
                name: 2 if name == "consortium" else 3 for name in targets
            },
        }
        return {"chrom": np.array(["1"]), "u": u, "v": v}, totals

    monkeypatch.setattr(regenerate, "sweep", fake_sweep)

    reference, suspect = regenerate.build_records(
        "height", regenerate.TRAITS["height"], "abc1234", "2026-07-22")
    assert reference["target"]["trait_type"] == "quantitative"
    assert reference["metrics"]["scale"] == "quantitative correlation R2"
    assert reference["metrics"]["den"] == 4.0
    assert reference["metrics"]["n_variants_scored"] == 2
    assert reference["overlap"]["method"] == regenerate.OVERLAP_METHOD
    assert suspect["metrics"]["den"] == 9.0
    assert suspect["metrics"]["n_variants_scored"] == 3
    assert suspect["overlap"]["status"] == "basis_unavailable"
    assert suspect["overlap"]["basis"]["kind"] == "unavailable"
    assert "trainer sensitivity" in suspect["overlap"]["basis"]["provenance"]
    assert "reference" in suspect["overlap"]
    for legacy in ("gamma", "gamma_se", "z", "m_total", "corrected_r2"):
        assert legacy not in suspect["overlap"]

    (unpaired,) = regenerate.build_records(
        "asthma", regenerate.TRAITS["asthma"], "abc1234", "2026-07-22")
    assert unpaired["target"]["trait_type"] == "binary"
    assert unpaired["metrics"]["scale"] == (
        "standardized logistic-summary approximation (not liability R2)")
    assert unpaired["overlap"]["status"] == "basis_unavailable"
    assert "no independent reference" in unpaired["overlap"]["note"]
