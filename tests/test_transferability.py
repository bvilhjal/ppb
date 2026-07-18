"""Guard test for the NEGATIVE transferability-reweighting result.

Records, as assertions, that LD-based reweighting of a PGS for cross-ancestry
transfer does not work in PPB's summary-stat framework: LD-only reweighting
degrades R²_B, and target-informed reweighting shows only an in-sample
(overfitting) gain. This prevents anyone re-adding a broken "reweight for
transferability" feature. See docs/TRANSFERABILITY.md.
"""

from experiments.transferability import run

CFG = dict(m=250, block_size=25, n=5000, n_ref=2500, n_causal=30, n_phenos=12)


def test_ld_only_reweighting_does_not_improve_cross_ancestry_r2():
    r = run(**CFG, rg=0.8, seed=0)
    # LD-only concordance shrinkage and pruning both make realized R²_B WORSE.
    assert r["shrink"] < r["naive"], f"LD-shrink {r['shrink']} unexpectedly >= naive {r['naive']}"
    assert r["prune"] < 0.9 * r["naive"], f"LD-prune {r['prune']} not clearly worse than naive"


def test_target_informed_reweighting_gain_is_in_sample_overfitting():
    r = run(**CFG, rg=0.8, seed=0)
    # Reweighting by efficiency computed on the SAME z_B it is scored against shows
    # an apparent gain — the refit-collapse / overfitting trap (real out-of-sample
    # gains require held-out/PUMAS validation and are not free).
    assert r["ti_insample"] > r["naive"], "expected an in-sample (overfit) inflation"


def test_real_headroom_requires_causal_identification():
    r = run(**CFG, rg=0.8, seed=0)
    # The causal-effect oracle far exceeds the naive transferred PGS — headroom is
    # real, but capturing it needs fine-mapping / causal identification (PolyPred,
    # PRS-CSx), not LD reweighting.
    assert r["oracle"] > 1.3 * r["naive"], f"oracle {r['oracle']} vs naive {r['naive']}"
