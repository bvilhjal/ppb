#!/usr/bin/env python
"""Regenerate the results registry (``results/*.json``) from the source data.

Supersedes the hand-transcription behind ``results/baseline-2026-07.json``: it
runs the genome-wide evaluation in one pass and emits the JSON record directly,
at full precision, so every number in the registry is reproducible from
committed code (schema: ``results/schema.md``).

It replaces three previously separate paths:

- ``scripts/eval_consortium.py`` -- R^2 against non-UKBB consortium targets,
- ``scripts/eval_panukb.py``     -- R^2 against the in-sample Pan-UKB targets,
- ``data/overlap_test/wldsc.py`` -- the legacy overlap fit, which lived in the
  *gitignored* ``data/`` tree with a hardcoded absolute path. The available
  artifacts contain only final LDpred2 weights, not the trainer sensitivity
  operator required by the basis-aware correction, so regenerated records fail
  closed with ``basis_unavailable`` rather than inventing a variant-count basis.

Both targets are evaluated in one sweep over the chromosomes, but each target
gets its own joint weight/sumstat support and score variance. Missing target
statistics therefore cannot remain in that target's denominator.

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

from ppb import (
    OverlapBasis,
    harmonize_to,
    read_ldref,
    read_weights,
    standardized_marginal,
)
from ppb.harmonize import VariantTable

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHROMS = [str(c) for c in range(1, 23)]

LD_REF = "bigsnpr HM3+ EUR (figshare 21305061; ppb int8 blocks)"

OVERLAP_METHOD = "scaled_signal_eiv_v1"
FINAL_WEIGHT_BASIS = OverlapBasis.unavailable(
    "only final PGS Catalog LDpred2 weights are available; "
    "the trainer sensitivity operator cannot be reconstructed"
)

METRIC_SCALES = {
    "quantitative": "quantitative correlation R2",
    "binary": "standardized logistic-summary approximation (not liability R2)",
}

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


def _trait_type(trait):
    return "binary" if trait in _BINARY else "quantitative"


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


def load_target(path, n_eff=None, trait_type=None):
    """Target sumstats -> ``(VariantTable, z, n_summary)``.

    Per-variant ``n`` is used when the file carries an ``n`` column (the
    consortium targets); otherwise the supplied trait-level ``n_eff`` is used
    (the Pan-UKB targets). ``trait_type`` distinguishes a quantitative sample
    size from a binary effective sample size in the recorded provenance.

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
    if trait_type not in METRIC_SCALES:
        raise ValueError(
            f"{path} has no 'n' column; trait_type must be 'quantitative' or 'binary'")
    basis = (
        "trait-level sample size" if trait_type == "quantitative" else
        "trait-level effective N (4/(1/n_case + 1/n_ctrl) for binary traits)"
    )
    summary = dict(n_eff=int(round(float(n_eff))), n_eff_basis=basis)
    return variants, standardized_marginal(beta, se, n_eff), summary


def sweep(pgs, targets):
    """One pass over chromosomes 1-22 for a score and its targets.

    Returns ``(per_block, totals)``. ``per_block`` holds one block-level score
    product ``u = w'z`` and score variance ``v = w'Dw`` per target. Both use
    that target's joint weight/sumstat intersection. ``totals`` holds the
    harmonization match counts.
    """
    w_var, w = read_weights(DATA / "pgs_weights" / f"{pgs}_hmPOS_GRCh37.txt")
    chrom_tag = []
    u_b = {name: [] for name in targets}
    v_b = {name: [] for name in targets}
    w_matched = 0
    z_matched = {name: 0 for name in targets}
    n_variants_scored = {name: 0 for name in targets}

    for c in CHROMS:
        t0 = time.time()
        ref = read_ldref(DATA / "ldref_hm3_plus" / "npz" / f"ldref_chr{c}.npz")
        w_al, rep_w, w_mask = harmonize_to(
            ref["variants"], w_var, w, return_mask=True)
        # Put the weights on the standardized-genotype scale the LD is defined on.
        sd = np.sqrt(2.0 * ref["af"] * (1.0 - ref["af"]))
        ws = w_al * sd
        w_matched += rep_w.n_matched

        aligned = {}
        for name, (z_var, z, _) in targets.items():
            z_al, rep_z, z_mask = harmonize_to(
                ref["variants"], z_var, z, return_mask=True)
            w_joint = ws.copy()
            w_joint[~(w_mask & z_mask)] = 0.0
            aligned[name] = (w_joint, z_al)
            z_matched[name] += rep_z.n_matched
            n_variants_scored[name] += int(np.count_nonzero(w_joint))

        for backend, idx in ref["ld"].blocks:
            chrom_tag.append(c)
            for name, (w_joint, z_al) in aligned.items():
                u_b[name].append(float(w_joint[idx] @ z_al[idx]))
                v_b[name].append(backend.quad(w_joint[idx]))
        print(f"    chr{c:<2} {len(ref['ld'].blocks):>4} blocks  "
              f"{time.time() - t0:6.1f}s", flush=True)
        del ref

    per_block = dict(
        chrom=np.array(chrom_tag),
        v={name: np.array(vals, dtype=float) for name, vals in v_b.items()},
        u={name: np.array(vals, dtype=float) for name, vals in u_b.items()})
    totals = dict(w_matched=w_matched, w_total=w_var.n,
                  z_matched=z_matched,
                  z_total={name: t[0].n for name, t in targets.items()},
                  n_variants_scored=n_variants_scored)
    return per_block, totals


def _metrics(num, den, w_frac, z_frac, n_variants_scored, trait_type):
    return dict(num=num, den=den, r2=num * num / den,
                w_match=w_frac, z_match=z_frac,
                n_variants_scored=n_variants_scored,
                scale=METRIC_SCALES[trait_type])


def _unavailable_overlap(role, reference=None):
    note = (
        "correction refused: only final LDpred2 weights are available, so the "
        "trainer sensitivity basis cannot be reconstructed; this in-sample "
        "estimate remains an upper bound"
    )
    if role == "suspect-unpaired":
        note += "; no independent reference target is available"
    overlap = dict(
        role=role,
        status="basis_unavailable",
        method=OVERLAP_METHOD,
        basis=dict(
            kind=FINAL_WEIGHT_BASIS.kind,
            provenance=FINAL_WEIGHT_BASIS.provenance,
        ),
        note=note,
    )
    if reference is not None:
        overlap["reference"] = reference
    return overlap


def build_records(trait, cfg, commit, date):
    """Evaluate one trait against its targets and return its registry records."""
    trait_type = _trait_type(trait)
    targets = {}
    if "consortium" in cfg:
        path = DATA / "consortium" / f"{cfg['consortium']}_hm3plus.tsv"
        targets["consortium"] = load_target(path, trait_type=trait_type)
    path = DATA / "panukb" / f"{cfg['panukb']}_hm3plus.tsv"
    targets["panukb"] = load_target(
        path, n_eff=cfg["panukb_n"], trait_type=trait_type)

    print(f"  {trait}: {cfg['pgs']}, targets = {', '.join(targets)}", flush=True)
    per_block, totals = sweep(cfg["pgs"], targets)

    den = {name: float(values.sum()) for name, values in per_block["v"].items()}
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
        den_ref = den["consortium"]
        records.append(dict(
            trait=trait, score=score,
            target=dict(gwas=cfg["consortium_gwas"], cohort=cfg["consortium_cohort"],
                        ancestry="EUR", **targets["consortium"][2],
                        overlap="none (declared)", trait_type=trait_type),
            ld_ref=LD_REF,
            metrics=_metrics(
                num_ref, den_ref, w_frac, zfrac("consortium"),
                totals["n_variants_scored"]["consortium"], trait_type),
            overlap=dict(
                role="reference", status="not_applicable",
                method=OVERLAP_METHOD,
            ),
            date=date, ppb_commit=commit))

    num_ov = float(per_block["u"]["panukb"].sum())
    den_ov = den["panukb"]
    if paired:
        reference = (f"{cfg['consortium_gwas'].split(' (')[0]} "
                     f"(R2 {num_ref * num_ref / den_ref:.4f})")
        overlap = _unavailable_overlap("suspect", reference)
    else:
        overlap = _unavailable_overlap("suspect-unpaired")
    records.append(dict(
        trait=trait, score=score,
        target=dict(gwas="Pan-UK Biobank (2020)", cohort="UK Biobank",
                    ancestry="EUR", **targets["panukb"][2],
                    overlap="in-sample", trait_type=trait_type),
        ld_ref=LD_REF,
        metrics=_metrics(
            num_ov, den_ov, w_frac, zfrac("panukb"),
            totals["n_variants_scored"]["panukb"], trait_type),
        overlap=overlap, date=date, ppb_commit=commit))
    return records


def git_commit():
    """Short HEAD, suffixed ``-dirty`` when the working tree has uncommitted changes.

    Without the suffix a pack can claim a commit that does not contain the code
    that produced it -- e.g. a run started before this script itself was
    committed. The marker makes that unreproducible provenance visible in the
    record instead of silently plausible.
    """
    def _git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout
    try:
        commit = _git("rev-parse", "--short", "HEAD").strip()
        return f"{commit}-dirty" if _git("status", "--porcelain").strip() else commit
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
