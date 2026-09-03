"""Figures for the ancestry report from tracked or reproducible inputs.

Figure 1 (simulation operating characteristics) reads the JSON dumped by
``python experiments/ancestry_ld_study.py --out <path>`` (default
``/tmp/ld_study.json``; study output does not belong in ``results/``).
Figures 2--3 read the committed historical Yengo snapshots and label them
as such in their captions; regenerate the snapshots first (see their
READMEs' migration records) before re-reading these figures as current.

Outputs: ``fig1_operating_characteristics.pdf``,
``fig2_projection_weights.pdf``, ``fig3_ld_diagnostics.pdf`` in this
directory. Also prints the LaTeX rows for the operating-characteristics
table. Requires matplotlib only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FREQUENCY_SNAPSHOT = (ROOT / "results" / "ancestry-frequency"
                      / "yengo-height-2026-08-30.json")
LD_SNAPSHOT = (ROOT / "results" / "ancestry-ld"
               / "yengo-height-2026-08-30.json")
HISTORICAL_TAG = "historical snapshots yengo-height-2026-08-30"


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fig1(study, out):
    """Bias, dispersion, jackknife calibration, and decline rates."""
    a_arms = [("0.0", "0"), ("0.025", "0"), ("0.025", "1"), ("0.1", "0"),
              ("0.1", "1"), ("0.5", "0"), ("0.5", "1")]
    keys = [f"A nh2/m={nh} absorb={ab}" for nh, ab in a_arms]
    rows = [study[k] for k in keys]
    x = np.arange(len(keys))
    err = np.array([r["max_abs_error"] for r in rows])
    sd = np.array([r["empirical_sd"][0] for r in rows])
    se = np.array([r["mean_jackknife_se"][0] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    axes[0].bar(x, err, color="0.35")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [f"{nh}\n{'abs' if ab == '1' else 'raw'}" for nh, ab in a_arms],
        fontsize=7)
    axes[0].set_ylabel("max |mean - truth|")
    axes[0].set_title("(a) bias of the replicate mean", fontsize=9)
    axes[0].axhline(0.05, color="k", linestyle="--", linewidth=0.8)

    width = 0.35
    axes[1].bar(x - width / 2, sd, width, label="empirical SD", color="0.35")
    axes[1].bar(x + width / 2, se, width, label="mean jackknife SE",
                color="0.75", edgecolor="k", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f"{nh}\n{'abs' if ab == '1' else 'raw'}" for nh, ab in a_arms],
        fontsize=7)
    axes[1].set_title("(b) dispersion vs jackknife", fontsize=9)
    axes[1].legend(fontsize=7, frameon=False)

    b_keys = ["B nh2/m=0.5", "B nh2/m=2.0", "B nh2/m=8.0",
              "B K=4 nh2/m=8", "IL fst=0.05 h2=0.5 B"]
    b_decl = [study[k]["declined"] for k in b_keys]
    b_reps = [study[k]["reps"] for k in b_keys]
    y = np.arange(len(b_keys))
    axes[2].barh(y, b_decl, color="0.35")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(b_keys, fontsize=7)
    axes[2].set_xlabel("declined / replicates")
    axes[2].set_title("(c) Estimator B declines", fontsize=9)
    for i, (ddecl, rreps) in enumerate(zip(b_decl, b_reps)):
        axes[2].text(ddecl + 0.2, i, f"{ddecl}/{rreps}", va="center",
                     fontsize=7)
    axes[2].set_xlim(0, max(b_reps) + 4)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Estimator operating characteristics (16 MVN reps; "
                 "truth (0.65, 0.35))", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig2(freq, ld, out):
    """Projection weights with boundary markers and conditional intervals."""
    pops = ["AFR", "AMR", "EAS", "EUR", "SAS"]
    studies = [s["study"]["key"] for s in freq["studies"]]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
    for ax, snap, key, title in (
            (axes[0], freq, "decomposition", "(a) frequency projection"),
            (axes[1], ld, "estimator_a", "(b) LD pair-product weights")):
        table = np.array(
            [s[key]["proportions"] for s in snap["studies"]])
        se = np.array(
            [s[key].get("proportions_se") or [np.nan] * 5
             for s in snap["studies"]])
        x = np.arange(len(studies))
        for j, pop in enumerate(pops):
            off = (j - 2) * 0.13
            vals = table[:, j]
            axes_dummy = ax.errorbar(
                x + off, vals, yerr=se[:, j], fmt="o", ms=4,
                capsize=2, label=pop)
            _ = axes_dummy
            boundary = vals <= 1e-6
            ax.plot((x + off)[boundary], vals[boundary], "x", ms=7,
                    mew=1.5, color="k")
        ax.set_xticks(x)
        ax.set_xticklabels(studies, fontsize=8)
        ax.set_ylabel("weight")
        ax.set_title(title + " -- " + HISTORICAL_TAG, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False, ncol=5, loc="upper right")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    fig.text(0.01, 0.01, "x: simplex-boundary weight (symmetric SE invalid "
             "there); intervals are conditional jackknife SEs.", fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig3(ld, out):
    """Collinearity, scale, channel disagreement, and sign-flip fit rates."""
    studies = [s["study"]["key"] for s in ld["studies"]]
    est = [s["estimator_a"] for s in ld["studies"]]
    diag = [s["independent_sign_flip_diagnostic"] for s in ld["studies"]]
    x = np.arange(len(studies))
    scale = np.array([e["scale"] for e in est])
    maxcorr = np.array([e["max_design_correlation"] for e in est])
    agree = np.array([e["channel_agreement"] for e in est])
    agree = np.where(np.array([a is None for a in agree]), np.nan,
                     agree).astype(float)
    fit_frac = np.array(
        [d["n_estimated"] / d["n_replicates"] for d in diag])

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0))
    axes[0, 0].bar(x, scale, color="0.35")
    axes[0, 0].axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(studies, fontsize=8)
    axes[0, 0].set_ylabel("fitted scale s")
    axes[0, 0].set_title("(a) scale vs working model s0 = 1-h2 <= 1",
                         fontsize=9)
    axes[0, 1].bar(x, maxcorr, color="0.35")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(studies, fontsize=8)
    axes[0, 1].set_ylabel("max |design correlation|")
    axes[0, 1].set_title("(b) linear-channel collinearity", fontsize=9)
    axes[1, 0].bar(x, agree, color="0.35")
    axes[1, 0].axhline(0.10, color="k", linestyle="--", linewidth=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(studies, fontsize=8)
    axes[1, 0].set_ylabel("max |pi_linear - pi_quad|")
    axes[1, 0].set_title("(c) channel agreement (warn > 0.10)", fontsize=9)
    axes[1, 1].bar(x, fit_frac, color="0.35")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(studies, fontsize=8)
    axes[1, 1].set_ylabel("fitted / 200 null replicates")
    axes[1, 1].set_title("(d) sign-flip fit rate (descriptive)",
                         fontsize=9)
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("LD-channel diagnostics -- " + HISTORICAL_TAG, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def latex_rows(study):
    print("% Operating characteristics (experiments/ancestry_ld_study.py).")
    for key in ["A nh2/m=0.0 absorb=0", "A nh2/m=0.025 absorb=0",
                "A nh2/m=0.025 absorb=1", "A nh2/m=0.1 absorb=0",
                "A nh2/m=0.1 absorb=1", "A nh2/m=0.5 absorb=0",
                "A nh2/m=0.5 absorb=1", "A K=4 h2=0",
                "A confusable h2=0", "IL fst=0.05 h2=0 A",
                "IL fst=0.2 h2=0 A", "IL fst=0.05 n_ref=500 A",
                "IL fst=0.05 n_ref=8000 A"]:
        r = study[key]
        mean = ", ".join(f"{v:.3f}" for v in r["mean"])
        sd = ", ".join(f"{v:.3f}" for v in r["empirical_sd"])
        se = ("; " + ", ".join(f"{v:.3f}"
              for v in r["mean_jackknife_se"])
              if "mean_jackknife_se" in r else "")
        print(f"% {key}: mean ({mean}) maxerr {r['max_abs_error']:.3f} "
              f"SD ({sd}){se} declined {r.get('declined', 0)}/{r['reps']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study-json", default="/tmp/ld_study.json")
    args = ap.parse_args()
    study = _load(args.study_json)
    freq = _load(FREQUENCY_SNAPSHOT)
    ld = _load(LD_SNAPSHOT)
    fig1(study, HERE / "fig1_operating_characteristics.pdf")
    fig2(freq, ld, HERE / "fig2_projection_weights.pdf")
    fig3(ld, HERE / "fig3_ld_diagnostics.pdf")
    print("wrote fig1/fig2/fig3 PDFs")
    latex_rows(study)


if __name__ == "__main__":
    main()
