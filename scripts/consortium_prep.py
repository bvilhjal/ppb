"""Parse consortium GWAS sumstats (non-UKBB) -> slim TSVs for ppb evaluation.

Output schema (same as the Pan-UKB filter): chrom, pos, a1, a2, beta, se, n
(n may be per-variant or a trait-level constant). GIANT files carry rsIDs
only; positions are resolved through the HM3+ map. Restricted to HM3+ variants.

Every dropped row is counted per reason and reported: the previous version
discarded malformed rows silently and resolved rsID ties by unreported first
occurrence, so a truncated or malformed source file was indistinguishable from
a small HM3+ intersection (review 2026-08-16, Minor 5).
"""
import csv
import gzip
from collections import Counter
from pathlib import Path

import numpy as np

DATA = str(Path(__file__).resolve().parent.parent / "data")

# Effective sample sizes for the binary consortium targets, written as the
# trait-level constant `n` below. They directly scale the recorded R^2
# (z = t/sqrt(t^2 + n - 2)). The case/control counts behind these three values
# were NOT recorded when the constants were first introduced (review
# 2026-08-16, F6), so they are carried with an explicit None marker rather
# than back-filled from memory. To make one computed, fill n_case/n_ctrl from
# the source publication: n_eff() then derives the constant and the registry
# records the counts. The shipped values reproduce the 2026-07 baseline
# exactly and must not change without regenerating the pack.
BINARY_N_EFF = {
    "CAD": (163123, None, None),   # CARDIoGRAMplusC4D 2015 (Nikpay et al.)
    "T2D": (88810, None, None),    # DIAGRAM 2017 (Scott et al.)
    "BrCa": (254739, None, None),  # BCAC 2017 (Michailidou et al.)
}


def n_eff(trait):
    """Binary-trait effective N: 4/(1/n_case + 1/n_ctrl) when the counts are
    recorded, the shipped constant when they are not. The constant is returned
    unchanged (an int) so the written sumstats file is byte-identical to the
    pre-refactor output."""
    n_eff_const, n_case, n_ctrl = _spec(trait)
    if n_case is None or n_ctrl is None:
        return n_eff_const
    return 4.0 / (1.0 / n_case + 1.0 / n_ctrl)


def n_eff_basis(trait):
    """Registry label distinguishing a computed effective N from a constant
    whose derivation is unrecorded."""
    _, n_case, n_ctrl = _spec(trait)
    if n_case is None or n_ctrl is None:
        return ("binary effective N, trait-level constant "
                "(case/control counts unrecorded)")
    return (f"binary effective N computed from recorded counts "
            f"(n_case={n_case}, n_ctrl={n_ctrl})")


def _spec(trait):
    try:
        return BINARY_N_EFF[trait]
    except KeyError:
        raise ValueError(
            f"no binary effective-N provenance recorded for trait {trait!r}; "
            "add n_case/n_ctrl (or the shipped constant) to BINARY_N_EFF "
            "before injecting an n into its sumstats") from None


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
                beta="beta", se="standard_error", n=n_eff("CAD")),
    "T2D": dict(src="DIAGRAM_T2D_2017.txt", chrpos="Chr:Position",
                a1="Allele1", a2="Allele2", beta="Effect", se="StdErr",
                n=n_eff("T2D"), gz=False),
    "BrCa": dict(src="BCAC_2017_build37.tsv.gz", chrom="chromosome",
                 pos="base_pair_location", a1="effect_allele", a2="other_allele",
                 beta="beta", se="standard_error", n=n_eff("BrCa")),
}


def load_map():
    """HM3+ variant map from the bigsnpr reference (figshare 37802721).

    ``rdata`` is the optional ``ldref`` extra and is imported lazily: this
    module's constants and labels must stay importable without it.
    """
    import warnings

    warnings.filterwarnings("ignore")
    import rdata
    return rdata.conversion.convert(
        rdata.parser.parse_file(f"{DATA}/ldref_hm3_plus/map_hm3_plus.rds"))


def build_lookup(info):
    """Map info -> ``(refset, rs2pos, n_collisions)``.

    rsIDs mapping to more than one position are resolved by first occurrence
    (deterministic) and counted, so the tie-breaking is reported rather than
    silent.
    """
    refset = set(zip(info["chr"].astype(str), info["pos"]))
    rs2pos = {}
    n_collisions = 0
    for c, p, r in zip(info["chr"].astype(str), info["pos"], info["rsid"]):
        first = rs2pos.setdefault(r, (c, p))
        if first != (c, p):
            n_collisions += 1
    return refset, rs2pos, n_collisions


def _required_columns(cfg):
    cols = [cfg["a1"], cfg["a2"], cfg["beta"], cfg["se"]]
    cols += [cfg[k] for k in ("rsid", "chrpos", "chrom", "pos", "n_col")
             if k in cfg]
    return cols


def prep_trait(trait, cfg, refset, rs2pos, src=None, dst=None):
    """Filter one consortium file to the HM3+ refset, counting every drop.

    A row is dropped for exactly one recorded reason (or kept); a source file
    whose columns do not match the config fails loudly instead of dropping
    every row. ``src``/``dst`` override the ``data/consortium`` defaults for
    tests.
    """
    opener = (lambda p: open(p, "rt")) if cfg.get("gz") is False \
        else (lambda p: gzip.open(p, "rt"))
    src = src or f"{DATA}/consortium/{cfg['src']}"
    dst = dst or f"{DATA}/consortium/{trait}_hm3plus.tsv"
    n_in = n_kept = 0
    dropped = Counter()
    with opener(src) as fh, open(dst, "w") as out:
        rows = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in _required_columns(cfg)
                   if c not in (rows.fieldnames or ())]
        if missing:
            raise ValueError(
                f"{trait}: source file {cfg['src']} lacks column(s) {missing}; "
                "refusing to drop every row as a silent parse failure")
        out.write("chrom\tpos\ta1\ta2\tbeta\tse\tn\n")
        for r in rows:
            n_in += 1
            if "rsid" in cfg:
                cp = rs2pos.get(r[cfg["rsid"]])
                if cp is None:
                    dropped["rsid absent from the HM3+ map"] += 1
                    continue
            elif "chrpos" in cfg:
                try:
                    c, p = r[cfg["chrpos"]].split(":")
                    cp = (c.replace("chr", ""), int(p))
                except (ValueError, AttributeError):
                    dropped["malformed chr:pos"] += 1
                    continue
            else:
                try:
                    cp = (r[cfg["chrom"]], int(r[cfg["pos"]]))
                except (ValueError, TypeError):
                    dropped["malformed chrom/pos"] += 1
                    continue
            if cp not in refset:
                dropped["position outside the HM3+ refset"] += 1
                continue
            try:
                beta, se = float(r[cfg["beta"]]), float(r[cfg["se"]])
            except (ValueError, TypeError):
                dropped["malformed beta/se"] += 1
                continue
            if not np.isfinite(beta) or not np.isfinite(se) or se <= 0:
                dropped["non-finite beta/se or se <= 0"] += 1
                continue
            if "n_col" in cfg:
                n = r[cfg["n_col"]]
                if n is None:
                    dropped["missing N value"] += 1
                    continue
            else:
                n = cfg["n"]
            out.write(f"{cp[0]}\t{cp[1]}\t{r[cfg['a1']].upper()}\t"
                      f"{r[cfg['a2']].upper()}\t{beta}\t{se}\t{n}\n")
            n_kept += 1
    detail = ", ".join(f"{count} {reason}" for reason, count
                       in sorted(dropped.items())) or "none"
    print(f"{trait}: kept {n_kept}/{n_in} -> {dst} (dropped: {detail})",
          flush=True)


def main():
    info = load_map()
    refset, rs2pos, n_collisions = build_lookup(info)
    if n_collisions:
        print(f"HM3+ map: {n_collisions} row(s) whose rsID maps to more than "
              "one position; the first occurrence is kept", flush=True)
    for trait, cfg in CFG.items():
        prep_trait(trait, cfg, refset, rs2pos)


if __name__ == "__main__":
    main()
