"""Stream-filter Pan-UKB flat sumstats to HM3+ variants -> slim per-trait TSV.

Keeps rows whose (chr, pos) is in the HM3+ LD reference and that pass QC
(non-missing beta/se, not low_confidence_EUR). Output columns:
chrom, pos, a1 (alt = effect allele), a2 (ref), beta, se, af
(af_EUR for continuous traits; af_controls_EUR for binary ones).

Usage: python filter_to_hm3plus.py [trait ...]  (default: all not yet done)
"""
import gzip
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import rdata

from pathlib import Path
DATA = str(Path(__file__).resolve().parent.parent / "data")
FILES = {
    "height": "continuous-50-both_sexes-irnt.tsv.bgz",
    "BMI": "continuous-21001-both_sexes-irnt.tsv.bgz",
    "LDL": "biomarkers-30780-both_sexes-irnt.tsv.bgz",
    "SBP": "continuous-4080-both_sexes-irnt.tsv.bgz",
    "T2D": "icd10-E11-both_sexes.tsv.bgz",
    "asthma": "icd10-J45-both_sexes.tsv.bgz",
    "MDD": "icd10-F32-both_sexes.tsv.bgz",
    "IHD": "icd10-I25-both_sexes.tsv.bgz",
    "BrCa": "icd10-C50-both_sexes.tsv.bgz",
}


def load_refset():
    info = rdata.conversion.convert(
        rdata.parser.parse_file(f"{DATA}/ldref_hm3_plus/map_hm3_plus.rds"))
    return set(zip(info["chr"].astype(str), info["pos"]))


def filter_trait(trait, fname, refset):
    src = f"{DATA}/panukb/{fname}"
    dst = f"{DATA}/panukb/{trait}_hm3plus.tsv"
    n_in = n_kept = 0
    with gzip.open(src, "rt") as fh, open(dst, "w") as out:
        header = fh.readline().rstrip("\n").split("\t")
        col = {h: i for i, h in enumerate(header)}
        ic, ip = col["chr"], col["pos"]
        ir, ia = col["ref"], col["alt"]
        ib, ise = col["beta_EUR"], col["se_EUR"]
        iaf = col["af_EUR"] if "af_EUR" in col else col["af_controls_EUR"]
        ilc = col["low_confidence_EUR"]
        out.write("chrom\tpos\ta1\ta2\tbeta\tse\taf\n")
        for ln in fh:
            n_in += 1
            r = ln.rstrip("\n").split("\t")
            if r[ilc] == "true" or r[ib] == "NA" or r[ise] == "NA":
                continue
            pos = int(r[ip])
            if (r[ic], pos) in refset:
                out.write(f"{r[ic]}\t{pos}\t{r[ia]}\t{r[ir]}\t{r[ib]}\t{r[ise]}\t{r[iaf]}\n")
                n_kept += 1
    print(f"{trait}: kept {n_kept}/{n_in} -> {dst}", flush=True)


if __name__ == "__main__":
    refset = load_refset()
    print(f"HM3+ positions: {len(refset)}", flush=True)
    traits = sys.argv[1:] or [t for t, f in FILES.items()]
    for trait in traits:
        dst = f"{DATA}/panukb/{trait}_hm3plus.tsv"
        if os.path.exists(dst):
            print(f"{trait}: already done, skipping", flush=True)
            continue
        filter_trait(trait, FILES[trait], refset)
    print("FILTER_DONE")
