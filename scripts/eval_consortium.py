"""ppb evaluation: PGS Catalog portability-ldpred2 scores x consortium GWAS.

Same estimator as the Pan-UKB run, but the consortium GWAS are UKBB-free, so
there is no training-sample overlap and R^2 is an honest portability-style
measurement (EUR -> EUR within-ancestry). Per-variant n is used when present.
"""
import csv
import sys

import numpy as np

from ppb import harmonize_to, read_ldref, read_weights, standardized_marginal
from ppb.harmonize import VariantTable

from pathlib import Path
DATA = str(Path(__file__).resolve().parent.parent / "data")

TRAITS = {
    "height": "PGS002146", "BMI": "PGS002161", "LDL": "PGS002150",
    "CAD": "PGS002048", "T2D": "PGS002026", "BrCa": "PGS002015",
}


def load_sumstats(path):
    chrom, pos, a1, a2, beta, se, n = [], [], [], [], [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            chrom.append(r["chrom"]); pos.append(int(r["pos"]))
            a1.append(r["a1"]); a2.append(r["a2"])
            beta.append(float(r["beta"])); se.append(float(r["se"]))
            n.append(float(r["n"]))
    return (VariantTable(np.array(chrom), np.array(pos), np.array(a1), np.array(a2)),
            np.array(beta), np.array(se), np.array(n))


def evaluate(pgs_file, ss_file):
    w_var, w = read_weights(pgs_file)
    z_var, beta, se, n = load_sumstats(ss_file)
    z = standardized_marginal(beta, se, n)
    num = den = 0.0
    w_match = z_match = 0
    for c in map(str, range(1, 23)):
        ref = read_ldref(f"{DATA}/ldref_hm3_plus/npz/ldref_chr{c}.npz")
        w_al, rep_w = harmonize_to(ref["variants"], w_var, w)
        z_al, rep_z = harmonize_to(ref["variants"], z_var, z)
        sd = np.sqrt(2 * ref["af"] * (1 - ref["af"]))
        w_std = w_al * sd
        num += float(w_std @ z_al)
        den += ref["ld"].quad(w_std)
        w_match += rep_w.n_matched
        z_match += rep_z.n_matched
        del ref
    return num, den, w_match, w_var.n, z_match, z_var.n


if __name__ == "__main__":
    only = sys.argv[1:] or list(TRAITS)
    print(f"{'trait':8} {'PGS':10} {'w_match':>8} {'z_match':>8} "
          f"{'w^T z':>10} {'w^T D w':>10} {'R^2':>8}")
    for trait in only:
        num, den, wm, wn, zm, zn = evaluate(
            f"{DATA}/pgs_weights/{TRAITS[trait]}_hmPOS_GRCh37.txt",
            f"{DATA}/consortium/{trait}_hm3plus.tsv")
        r2 = num * num / den
        print(f"{trait:8} {TRAITS[trait]:10} {wm/wn:8.1%} {zm/zn:8.1%} "
              f"{num:10.4f} {den:10.4f} {r2:8.4f}", flush=True)
