"""Multi-trait ppb evaluation: PGS Catalog scores x Pan-UKB EUR GWAS x HM3+ EUR LD.

For each trait: R^2 = (w^T z)^2 / (w^T D w) accumulated genome-wide per
chromosome, w standardized with sd_j = sqrt(2 f_j (1 - f_j)) (f = af_UKBB from
the LD reference), z from beta/se with the trait's (effective) sample size.

Scores are Privé et al. 2022 (AJHG) portability-ldpred2 weights trained on UK
Biobank; the Pan-UKB GWAS is the same cohort -> sample overlap inflates R^2.
This is a machinery demonstration, not an unbiased accuracy estimate.
"""
import csv
import sys

import numpy as np

from ppb import harmonize_to, read_ldref, read_weights, standardized_marginal
from ppb.harmonize import VariantTable

from pathlib import Path
DATA = str(Path(__file__).resolve().parent.parent / "data")

# trait -> (PGS id, sumstats TSV, n or (n_cases, n_controls))
TRAITS = {
    "height": ("PGS002146", "height_hm3plus.tsv", 419596),
    "BMI": ("PGS002161", "BMI_hm3plus.tsv", 419163),
    "LDL": ("PGS002150", "LDL_hm3plus.tsv", 400223),
    "SBP": ("PGS002228", "SBP_hm3plus.tsv", 396663),
    "T2D": ("PGS002026", "T2D_hm3plus.tsv", (22634, 397897)),
    "asthma": ("PGS002061", "asthma_hm3plus.tsv", (31030, 389501)),
    "MDD": ("PGS002036", "MDD_hm3plus.tsv", (15674, 404857)),
    "IHD": ("PGS002048", "IHD_hm3plus.tsv", (31148, 389383)),
    "BrCa": ("PGS002015", "BrCa_hm3plus.tsv", (11779, 408752)),
}


def load_sumstats(path):
    chrom, pos, a1, a2, beta, se = [], [], [], [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            chrom.append(r["chrom"]); pos.append(int(r["pos"]))
            a1.append(r["a1"]); a2.append(r["a2"])
            beta.append(float(r["beta"])); se.append(float(r["se"]))
    return (VariantTable(np.array(chrom), np.array(pos), np.array(a1), np.array(a2)),
            np.array(beta), np.array(se))


def evaluate(pgs_file, ss_file, n):
    w_var, w = read_weights(pgs_file)
    z_var, beta, se = load_sumstats(ss_file)
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
    print(f"{'trait':8} {'PGS':10} {'n_eff':>7} {'w_match':>8} {'z_match':>8} "
          f"{'w^T z':>10} {'w^T D w':>10} {'R^2':>8}")
    for trait in only:
        pgs, ss, n = TRAITS[trait]
        if isinstance(n, tuple):
            n_cases, n_ctrl = n
            n = 4 / (1 / n_cases + 1 / n_ctrl)
        num, den, wm, wn, zm, zn = evaluate(
            f"{DATA}/pgs_weights/{pgs}_hmPOS_GRCh37.txt", f"{DATA}/panukb/{ss}", n)
        r2 = num * num / den
        print(f"{trait:8} {pgs:10} {n:7.0f} {wm/wn:8.1%} {zm/zn:8.1%} "
              f"{num:10.4f} {den:10.4f} {r2:8.4f}", flush=True)
