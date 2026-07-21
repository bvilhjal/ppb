#!/usr/bin/env python
"""Regenerate the results registry (``results/*.json``) from the source data.

Supersedes the hand-transcription behind ``results/baseline-2026-07.json``: it
runs the genome-wide evaluation and the dual-target overlap detector in one
pass and emits the JSON record directly, at full precision, so every number in
the registry is reproducible from committed code (schema: ``results/schema.md``).

It replaces three previously separate paths:

- ``scripts/eval_consortium.py`` -- R^2 against non-UKBB consortium targets,
- ``scripts/eval_panukb.py``     -- R^2 against the in-sample Pan-UKB targets,
- ``data/overlap_test/wldsc.py`` -- the overlap fit, which lived in the
  *gitignored* ``data/`` tree with a hardcoded absolute path, so the recorded
  ``gamma``/``corrected_r2`` were not reproducible from the repository at all.
  The fit here uses the library's :func:`ppb.overlap.overlap_slope` rather than
  a private copy of the weighted through-origin regression.

For one score, the LD reference and the per-block score variance ``v_b`` are
target-independent, so both targets are evaluated in a single sweep over the
chromosomes -- the LD reference is read once per trait, not once per target.

Run (needs the ~24G LD reference and the target sumstats under ``data/``):

    python scripts/regenerate_results.py                  # every trait
    python scripts/regenerate_results.py height BMI       # a subset
    python scripts/regenerate_results.py --out results/baseline-2026-07.json
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ppb import harmonize_to, read_ldref, read_weights, standardized_marginal
from ppb.harmonize import VariantTable
from ppb.overlap import correct_numerator, overlap_slope

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHROMS = [str(c) for c in range(1, 23)]

LD_REF = "bigsnpr HM3+ EUR (figshare 21305061; ppb int8 blocks)"

# Correct the numerator only when the detector actually detects. Below this the
# fit is consistent with zero, and subtracting an insignificant gamma would fit
# noise rather than remove overlap -- so the evaluation stays an upper bound.
# This is the threshold the leaderboard legend and docs/OVERLAP.md state.
Z_DETECT = 2.0

# Pan-UKB effective sample sizes: n for quantitative traits, 4/(1/n_case + 1/n_ctrl)
# for binary ones (the same values scripts/eval_panukb.py derives from the counts).
_BINARY = {
    "T2D": (22634, 397897), "asthma": (31030, 389501), "MDD": (15674, 404857),
    "CAD": (31148, 389383), "BrCa": (11779, 408752),
}


def _n_eff(trait, quantitative):
    if trait in _BINARY:
        n_case, n_ctrl = _BINARY[trait]
        return 4.0 / (1.0 / n_case + 1.0 / n_ctrl)
    return float(quantitative)


# trait -> score, prose provenance, Pan-UKB sumstats stem + n_eff, consortium stem + label
TRAITS = {
    "height": dict(
        pgs="PGS002146", score_name="portability-ldpred2_height",
        panukb="height", panukb_n=_n_eff("height", 419596),
        consortium="height", consortium_gwas="GIANT 2014 (Wood et al.)",
        consortium_cohort="GIANT consortium (no UKBB)"),
    "BMI": dict(
        pgs="PGS002161", score_name="portability-ldpred2_BMI",
        panukb="BMI", panukb_n=_n_eff("BMI", 419163),
        consortium="BMI", consortium_gwas="GIANT 2015 (Locke et al.)",
        consortium_cohort="GIANT consortium (no UKBB)"),
    "LDL": dict(
        pgs="PGS002150", score_name="portability-ldpred2_LDL",
        panukb="LDL", panukb_n=_n_eff("LDL", 400223),
        consortium="LDL", consortium_gwas="GLGC 2013 (Teslovich et al.)",
        consortium_cohort="GLGC consortium (no UKBB)"),
    "CAD": dict(
        pgs="PGS002048", score_name="portability-ldpred2_CAD",
        panukb="IHD", panukb_n=_n_eff("CAD", None),
        consortium="CAD", consortium_gwas="CARDIoGRAMplusC4D 2015 (Nikpay et al.)",
        consortium_cohort="CARDIoGRAMplusC4D (no UKBB)"),
    "T2D": dict(
        pgs="PGS002026", score_name="portability-ldpred2_T2D",
        panukb="T2D", panukb_n=_n_eff("T2D", None),
        consortium="T2D", consortium_gwas="DIAGRAM 2017 (Scott et al.)",
        consortium_cohort="DIAGRAM consortium (no UKBB)"),
    "BrCa": dict(
        pgs="PGS002015", score_name="portability-ldpred2_BrCa",
        panukb="BrCa", panukb_n=_n_eff("BrCa", None),
        consortium="BrCa", consortium_gwas="BCAC 2017 (Michailidou et al.)",
        consortium_cohort="BCAC consortium (no UKBB)"),
    # Pan-UKB only: no non-overlapping reference target, so no detector and no
    # correction -- these are upper bounds (schema.md: "suspect-unpaired").
    "SBP": dict(
        pgs="PGS002228", score_name="portability-ldpred2_SBP",
        panukb="SBP", panukb_n=_n_eff("SBP", 396663)),
    "asthma": dict(
        pgs="PGS002061", score_name="portability-ldpred2_asthma",
        panukb="asthma", panukb_n=_n_eff("asthma", None)),
    "MDD": dict(
        pgs="PGS002036", score_name="portability-ldpred2_MDD",
        panukb="MDD", panukb_n=_n_eff("MDD", None)),
}


def load_target(path, n_eff=None):
    """Target sumstats -> ``(VariantTable, z, n_summary)``.

    Per-variant ``n`` is used when the file carries an ``n`` column (the
    consortium targets); otherwise the supplied trait-level ``n_eff`` is used
    (the Pan-UKB targets).

    ``n_summary`` describes the sample size the estimator actually saw, which is
    a *distribution* when ``n`` is per-variant: GIANT/GLGC files carry an ``N``
    that ranges from ~50k to the study maximum across variants. The registry
    records its median plus the basis, because a single headline number is not
    well defined here -- and the published study N (e.g. 339,224 for GIANT BMI)
    can exceed every per-variant value present in the HM3+-filtered file.
    """
    chrom, pos, a1, a2, beta, se, n = [], [], [], [], [], [], []
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        with_n = "n" in (reader.fieldnames or ())
        for r in reader:
            chrom.append(r["chrom"]); pos.append(int(r["pos"]))
            a1.append(r["a1"]); a2.append(r["a2"])
            beta.append(float(r["beta"])); se.append(float(r["se"]))
            if with_n:
                n.append(float(r["n"]))
    variants = VariantTable(np.array(chrom), np.array(pos, dtype=np.int64),
                            np.array(a1), np.array(a2))
    beta, se = np.array(beta), np.array(se)
    if with_n:
        n = np.array(n)
        constant = bool(n.min() == n.max())
        summary = dict(
            n_eff=int(round(float(np.median(n)))),
            n_eff_basis=("trait-level constant in the sumstats file" if constant
                         else "median of the per-variant N column"))
        if not constant:
            summary["n_eff_range"] = [int(n.min()), int(n.max())]
        return variants, standardized_marginal(beta, se, n), summary
    if n_eff is None:
        raise ValueError(f"{path} has no 'n' column and no n_eff was supplied")
    summary = dict(n_eff=int(round(float(n_eff))),
                   n_eff_basis="trait-level effective N (4/(1/n_case + 1/n_ctrl) "
                               "for binary traits)")
    return variants, standardized_marginal(beta, se, n_eff), summary


def sweep(pgs, targets):
    """One pass over chromosomes 1-22 for a score and its targets.

    Returns ``(per_block, totals)``. ``per_block`` holds the block-level
    ``chrom``, ``m`` (variant count), ``v`` (score variance ``w'Dw``) and one
    ``u`` (``w'z``) column per target; ``totals`` holds the harmonization match
    counts. ``v`` is target-independent, so it is computed once.
    """
    w_var, w = read_weights(DATA / "pgs_weights" / f"{pgs}_hmPOS_GRCh37.txt")
    chrom_tag, m_b, v_b = [], [], []
    u_b = {name: [] for name in targets}
    w_matched = 0
    z_matched = {name: 0 for name in targets}

    for c in CHROMS:
        t0 = time.time()
        ref = read_ldref(DATA / "ldref_hm3_plus" / "npz" / f"ldref_chr{c}.npz")
        w_al, rep_w = harmonize_to(ref["variants"], w_var, w)
        # Put the weights on the standardized-genotype scale the LD is defined on.
        sd = np.sqrt(2.0 * ref["af"] * (1.0 - ref["af"]))
        ws = w_al * sd
        w_matched += rep_w.n_matched

        aligned = {}
        for name, (z_var, z, _) in targets.items():
            z_al, rep_z = harmonize_to(ref["variants"], z_var, z)
            aligned[name] = z_al
            z_matched[name] += rep_z.n_matched

        for backend, idx in ref["ld"].blocks:
            chrom_tag.append(c)
            m_b.append(idx.size)
            v_b.append(backend.quad(ws[idx]))
            for name, z_al in aligned.items():
                u_b[name].append(float(ws[idx] @ z_al[idx]))
        print(f"    chr{c:<2} {len(ref['ld'].blocks):>4} blocks  "
              f"{time.time() - t0:6.1f}s", flush=True)
        del ref

    per_block = dict(
        chrom=np.array(chrom_tag), m=np.array(m_b, dtype=float),
        v=np.array(v_b, dtype=float),
        u={name: np.array(vals, dtype=float) for name, vals in u_b.items()})
    totals = dict(w_matched=w_matched, w_total=w_var.n,
                  z_matched=z_matched,
                  z_total={name: t[0].n for name, t in targets.items()})
    return per_block, totals


def _metrics(num, den, w_frac, z_frac):
    return dict(num=num, den=den, r2=num * num / den,
                w_match=w_frac, z_match=z_frac)


def build_records(trait, cfg, commit, date):
    """Evaluate one trait against its targets and return its registry records."""
    targets = {}
    if "consortium" in cfg:
        path = DATA / "consortium" / f"{cfg['consortium']}_hm3plus.tsv"
        targets["consortium"] = load_target(path)
    path = DATA / "panukb" / f"{cfg['panukb']}_hm3plus.tsv"
    targets["panukb"] = load_target(path, n_eff=cfg["panukb_n"])

    print(f"  {trait}: {cfg['pgs']}, targets = {', '.join(targets)}", flush=True)
    per_block, totals = sweep(cfg["pgs"], targets)

    den = float(per_block["v"].sum())
    m_total = int(per_block["m"].sum())
    n_variants = totals["w_total"]
    w_frac = totals["w_matched"] / totals["w_total"]
    score = dict(id=cfg["pgs"], name=cfg["score_name"],
                 training="UK Biobank (LDpred2)", n_variants=n_variants)

    def zfrac(name):
        return totals["z_matched"][name] / totals["z_total"][name]

    records = []
    paired = "consortium" in targets
    if paired:
        num_ref = float(per_block["u"]["consortium"].sum())
        records.append(dict(
            trait=trait, score=score,
            target=dict(gwas=cfg["consortium_gwas"], cohort=cfg["consortium_cohort"],
                        ancestry="EUR", **targets["consortium"][2],
                        overlap="none (declared)"),
            ld_ref=LD_REF,
            metrics=_metrics(num_ref, den, w_frac, zfrac("consortium")),
            overlap=dict(role="reference"),
            date=date, ppb_commit=commit))

    num_ov = float(per_block["u"]["panukb"].sum())
    overlap = dict(role="suspect" if paired else "suspect-unpaired")
    if paired:
        # Dual-target detector: the signal cancels in u_target - u_reference,
        # leaving the flat per-variant overlap term. Jackknife by chromosome.
        est = overlap_slope(
            per_block["u"]["panukb"], per_block["u"]["consortium"],
            per_block["m"], per_block["v"], per_block["v"],
            groups=per_block["chrom"])
        overlap.update(gamma=est.gamma, gamma_se=est.se, z=est.z,
                       m_total=m_total,
                       reference=f"{cfg['consortium_gwas'].split(' (')[0]} "
                                 f"(R² {num_ref * num_ref / den:.4f})")
        if abs(est.z) >= Z_DETECT:
            num_corr = correct_numerator(num_ov, est.gamma, m_total)
            overlap["corrected_r2"] = num_corr * num_corr / den
        else:
            # The uniform overlap term is invisible here. Low-coverage scores are
            # the documented failure mode: shrinkage projects the noise fit onto
            # the same variants that carry signal, so the flat term vanishes and
            # overlap is unidentifiable by shape alone (docs/OVERLAP.md).
            coverage = n_variants / m_total
            overlap["note"] = (
                f"no detectable uniform overlap term (z = {est.z:.2f}); score covers "
                f"{coverage:.0%} of the LD reference — detector blind by construction "
                f"for shrunk/low-coverage scores, so this is an upper bound, "
                f"not a corrected measurement")
    records.append(dict(
        trait=trait, score=score,
        target=dict(gwas="Pan-UK Biobank (2020)", cohort="UK Biobank",
                    ancestry="EUR", **targets["panukb"][2],
                    overlap="in-sample"),
        ld_ref=LD_REF,
        metrics=_metrics(num_ov, den, w_frac, zfrac("panukb")),
        overlap=overlap, date=date, ppb_commit=commit))
    return records


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("traits", nargs="*", default=None,
                    help=f"traits to evaluate (default: all of {', '.join(TRAITS)})")
    ap.add_argument("--out", default=None,
                    help="write the JSON pack here (default: stdout)")
    ap.add_argument("--date", default=None,
                    help="record date, YYYY-MM-DD (default: today, UTC)")
    args = ap.parse_args(argv)

    traits = args.traits or list(TRAITS)
    unknown = [t for t in traits if t not in TRAITS]
    if unknown:
        ap.error(f"unknown trait(s): {', '.join(unknown)}")

    date = args.date or time.strftime("%Y-%m-%d", time.gmtime())
    commit = git_commit()
    records = []
    for trait in traits:
        t0 = time.time()
        records.extend(build_records(trait, TRAITS[trait], commit, date))
        print(f"  {trait} done in {time.time() - t0:.0f}s", flush=True)

    text = json.dumps(records, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"{len(records)} records -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
