"""Focused contracts for the real-data result regeneration path."""

import gc
import weakref

import numpy as np
import pytest

from ppb.harmonize import VariantTable
from scripts import eval_consortium, eval_panukb
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


class _IdentityLD:
    @staticmethod
    def quad(x):
        return float(x @ x)


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


def test_sweep_harmonizes_only_the_current_chromosome(monkeypatch):
    variants = VariantTable(
        ["1", "1", "2", "2"], [1, 2, 1, 2],
        ["A"] * 4, ["C"] * 4)
    values = np.arange(1.0, 5.0)
    references = {
        chrom: VariantTable([chrom, chrom], [1, 2], ["A", "A"], ["C", "C"])
        for chrom in ("1", "2")
    }

    class LocalLD:
        def __init__(self):
            self.blocks = [(_IdentityBlock(), np.arange(2))]

    monkeypatch.setattr(regenerate, "CHROMS", ["1", "2"])
    monkeypatch.setattr(
        regenerate, "read_weights", lambda path: (variants, values))

    def fake_read(path):
        chrom = "1" if "chr1.npz" in str(path) else "2"
        return {
            "variants": references[chrom],
            "af": np.full(2, 0.5),
            "ld": LocalLD(),
        }

    monkeypatch.setattr(regenerate, "read_ldref", fake_read)
    original = regenerate.harmonize_to
    target_sizes = []

    def tracked(reference, target, value, **kwargs):
        target_sizes.append(target.n)
        return original(reference, target, value, **kwargs)

    monkeypatch.setattr(regenerate, "harmonize_to", tracked)
    _, totals = regenerate.sweep(
        "PGS000000", {"target": (variants, values, {})})

    # One weight and one target pass per chromosome, each over two local rows.
    assert target_sizes == [2, 2, 2, 2]
    assert totals["w_matched"] == 4
    assert totals["z_matched"]["target"] == 4


def test_sweep_releases_block_backend_before_next_ld_read(monkeypatch):
    variants = _variants([1, 2, 3])
    weights = np.ones(3)
    previous = None
    calls = 0

    class LocalLD:
        def __init__(self):
            self.blocks = [(_IdentityBlock(), np.arange(3))]

    monkeypatch.setattr(regenerate, "CHROMS", ["1", "2"])
    monkeypatch.setattr(
        regenerate, "read_weights", lambda path: (variants, weights))

    def fake_read(path):
        nonlocal previous, calls
        gc.collect()
        if previous is not None:
            assert previous() is None
        calls += 1
        ld = LocalLD()
        previous = weakref.ref(ld.blocks[0][0])
        chrom = str(calls)
        reference = VariantTable(
            [chrom] * 3, [1, 2, 3], ["A"] * 3, ["C"] * 3)
        return {"variants": reference, "af": np.full(3, 0.5), "ld": ld}

    monkeypatch.setattr(regenerate, "read_ldref", fake_read)
    target = VariantTable(
        ["1"] * 3 + ["2"] * 3, [1, 2, 3] * 2,
        ["A"] * 6, ["C"] * 6)
    regenerate.sweep(
        "PGS000000", {"target": (target, np.ones(6), {})})

    assert calls == 2


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


@pytest.mark.parametrize(
    "module,is_panukb",
    [(eval_consortium, False), (eval_panukb, True)],
)
def test_human_readable_evaluators_use_joint_support(
        monkeypatch, module, is_panukb):
    reference = _variants([1, 2, 3])
    partial = _variants([1, 3])
    weights = np.array([1.0, 2.0, 3.0])
    z = np.array([10.0, 30.0])
    monkeypatch.setattr(module, "map", lambda function, values: ("1",),
                        raising=False)
    monkeypatch.setattr(module, "read_weights", lambda path: (reference, weights))
    monkeypatch.setattr(
        module, "read_ldref",
        lambda path: {
            "variants": reference,
            "af": np.repeat(0.5, 3),
            "ld": _IdentityLD(),
        })
    monkeypatch.setattr(module, "standardized_marginal",
                        lambda beta, se, n: beta)
    if is_panukb:
        monkeypatch.setattr(
            module, "load_sumstats",
            lambda path: (partial, z, np.ones(2)))
        observed = module.evaluate("weights", "sumstats", 1000)
    else:
        monkeypatch.setattr(
            module, "load_sumstats",
            lambda path: (partial, z, np.ones(2), np.repeat(1000.0, 2)))
        observed = module.evaluate("weights", "sumstats")

    num, den, w_matched, w_total, z_matched, z_total, n_scored = observed
    assert num == pytest.approx(100.0 / np.sqrt(2.0))
    assert den == pytest.approx(5.0)
    assert (w_matched, w_total, z_matched, z_total, n_scored) == (3, 3, 2, 2, 2)


def _blocked_metrics(n_blocks=66, seed=0, coherent=True):
    rng = np.random.default_rng(seed)
    u = (np.abs(rng.normal(0.02, 0.004, n_blocks)) if coherent
         else rng.normal(0.0, 0.02, n_blocks))
    v = rng.uniform(0.5, 1.5, n_blocks)
    chrom = np.repeat([str(c) for c in range(1, 23)],
                      int(np.ceil(n_blocks / 22)))[:n_blocks]
    num, den = float(u.sum()), float(v.sum())
    return regenerate._metrics(
        num, den, 1.0, 1.0, 1000, "quantitative", u=u, v=v, chrom=chrom), u, v


def test_metrics_records_the_block_jackknife_and_sign_flip_null():
    metrics, u, v = _blocked_metrics()

    assert metrics["jackknife"]["n_groups"] == 66
    assert metrics["jackknife"]["se"] > 0.0
    assert metrics["jackknife_chromosome"]["n_groups"] == 22
    assert metrics["sign_flip_null"]["z"] > 5.0
    assert metrics["sign_flip_null"]["null_mean"] > 0.0
    # The control must calibrate the headline value, not restate it.
    assert metrics["sign_flip_null"]["ratio"] == pytest.approx(
        metrics["r2"] / metrics["sign_flip_null"]["null_mean"])
    # Per-chromosome partial sums must reconstruct the headline totals.
    assert sum(p[0] for p in metrics["per_chromosome"].values()) == pytest.approx(
        metrics["num"])
    assert sum(p[1] for p in metrics["per_chromosome"].values()) == pytest.approx(
        metrics["den"])


def test_metrics_declares_when_the_block_diagnostics_cannot_run():
    """A single-block sweep must say so, not emit a field that looks computed."""
    metrics = regenerate._metrics(
        0.1, 1.0, 1.0, 1.0, 10, "quantitative",
        u=np.array([0.1]), v=np.array([1.0]), chrom=np.array(["1"]))
    assert "jackknife" not in metrics
    assert "sign_flip_null" not in metrics
    assert "LD block" in metrics["diagnostics_unavailable"]


def test_metrics_without_block_products_omits_the_diagnostics():
    metrics = regenerate._metrics(0.1, 1.0, 1.0, 1.0, 10, "quantitative")
    assert set(metrics) == {"num", "den", "r2", "w_match", "z_match",
                            "n_variants_scored", "scale"}


def test_emitted_diagnostics_satisfy_the_registry_schema():
    """What the generator writes must pass the validators the registry enforces."""
    from tests import test_results_registry as registry

    metrics, _, _ = _blocked_metrics()
    record = {"metrics": {**metrics, "r2": metrics["r2"]},
              "score": {"n_variants": 5000}}
    registry.test_jackknife_blocks_are_internally_consistent("synthetic", record)
    registry.test_sign_flip_null_is_internally_consistent("synthetic", record)
    registry.test_per_chromosome_sums_reproduce_the_headline_metrics(
        "synthetic", record)
    registry.test_missing_diagnostics_declare_why("synthetic", record)


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m["jackknife"].update(n_groups=999), "more groups than blocks"),
    (lambda m: m["jackknife"].update(se=-1.0), "non-negative"),
    (lambda m: m["jackknife"].update(max_variance_share=1.5), "fraction"),
    (lambda m: m["jackknife"].update(n_groups=1), "at least 2 delete-one groups"),
])
def test_registry_rejects_a_malformed_jackknife(mutate, match):
    from tests import test_results_registry as registry

    metrics, _, _ = _blocked_metrics()
    mutate(metrics)
    with pytest.raises(AssertionError, match=match):
        registry.test_jackknife_blocks_are_internally_consistent(
            "synthetic", {"metrics": metrics})


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m["sign_flip_null"].update(z=99.0), r"exceeds sqrt\(n_blocks\)"),
    (lambda m: m["sign_flip_null"].update(null_mean=0.0), "must be positive"),
    (lambda m: m["sign_flip_null"].update(ratio=1.0), "must equal r2 / null_mean"),
    (lambda m: m["sign_flip_null"].update(z_ceiling=2.0), None),
])
def test_registry_rejects_a_malformed_sign_flip_null(mutate, match):
    from tests import test_results_registry as registry

    metrics, _, _ = _blocked_metrics()
    mutate(metrics)
    with pytest.raises(AssertionError, match=match):
        registry.test_sign_flip_null_is_internally_consistent(
            "synthetic", {"metrics": metrics})


def test_registry_rejects_per_chromosome_sums_that_do_not_reconstruct_the_total():
    from tests import test_results_registry as registry

    metrics, _, _ = _blocked_metrics()
    first = next(iter(metrics["per_chromosome"]))
    metrics["per_chromosome"][first][0] += 0.5
    with pytest.raises(AssertionError, match="does not sum to metrics.num"):
        registry.test_per_chromosome_sums_reproduce_the_headline_metrics(
            "synthetic", {"metrics": metrics})
