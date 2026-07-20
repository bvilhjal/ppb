"""Parse consortium GWAS sumstats (non-UKBB) -> slim TSVs for ppb evaluation.

Output schema (same as the Pan-UKB filter): chrom, pos, a1, a2, beta, se, n
(n may be per-variant or a trait-level constant). GIANT files carry rsIDs
only; positions are resolved through the HM3+ map. Restricted to HM3+ variants.
"""
import csv
import gzip
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import rdata

from pathlib import Path
DATA = str(Path(__file__).resolve().parent.parent / "data")

# trait -> dict(src, kind, a1, a2, beta, se, n | n_col, chrom/pos | rsid)
CFG = {
    "height": dict(src="GIANT_HEIGHT_2014.txt.gz", rsid="MarkerName",
                   a1="Allele1", a2="Allele2", beta="b", se="SE", n_col="N"),
    "BMI": dict(src="GIANT_BMI_2015.txt.gz", rsid="SNP",
                a1="A1", a2="A2", beta="b", se="se", n_col="N"),
    "LDL": dict(src="GLGC_LDL_2013.txt.gz", chrpos="SNP_hg19",
                a1="A1", a2="A2", beta="beta", se="se", n_col="N"),
    "CAD": dict(src="CARDIO_CAD_2015_build37.tsv.gz", chrom="chromosome",
                pos="base_pair_location", a1="effect_allele", a2="other_allele",
                beta="beta", se="standard_error", n=163123),
    "T2D": dict(src="DIAGRAM_T2D_2017.txt", chrpos="Chr:Position",
                a1="Allele1", a2="Allele2", beta="Effect", se="StdErr", n=88810,
                gz=False),
    "BrCa": dict(src="BCAC_2017_build37.tsv.gz", chrom="chromosome",
                 pos="base_pair_location", a1="effect_allele", a2="other_allele",
                 beta="beta", se="standard_error", n=254739),
}


def main():
    info = rdata.conversion.convert(
        rdata.parser.parse_file(f"{DATA}/ldref_hm3_plus/map_hm3_plus.rds"))
    refset = set(zip(info["chr"].astype(str), info["pos"]))
    rs2pos = {}
    for c, p, r in zip(info["chr"].astype(str), info["pos"], info["rsid"]):
        rs2pos.setdefault(r, (c, p))

    for trait, cfg in CFG.items():
        opener = (lambda p: open(p, "rt")) if cfg.get("gz") is False \
            else (lambda p: gzip.open(p, "rt"))
        src = f"{DATA}/consortium/{cfg['src']}"
        dst = f"{DATA}/consortium/{trait}_hm3plus.tsv"
        n_in = n_kept = 0
        with opener(src) as fh, open(dst, "w") as out:
            rows = csv.DictReader(fh, delimiter="\t")
            out.write("chrom\tpos\ta1\ta2\tbeta\tse\tn\n")
            for r in rows:
                n_in += 1
                try:
                    if "rsid" in cfg:
                        cp = rs2pos.get(r[cfg["rsid"]])
                        if cp is None:
                            continue
                    elif "chrpos" in cfg:
                        c, p = r[cfg["chrpos"]].split(":")
                        cp = (c.replace("chr", ""), int(p))
                    else:
                        cp = (r[cfg["chrom"]], int(r[cfg["pos"]]))
                    if cp not in refset:
                        continue
                    beta, se = float(r[cfg["beta"]]), float(r[cfg["se"]])
                    if not np.isfinite(beta) or not np.isfinite(se) or se <= 0:
                        continue
                    n = r[cfg["n_col"]] if "n_col" in cfg else cfg["n"]
                    out.write(f"{cp[0]}\t{cp[1]}\t{r[cfg['a1']].upper()}\t"
                              f"{r[cfg['a2']].upper()}\t{beta}\t{se}\t{n}\n")
                    n_kept += 1
                except (ValueError, KeyError):
                    continue
        print(f"{trait}: kept {n_kept}/{n_in} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
