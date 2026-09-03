"""Build the tracked miniature dataset in this directory.

Deterministic (seed 0, numpy only): 8 variants, one LD block, standardized
weights. Regenerate with ``python make_mini.py`` from this directory; the
outputs (``weights.tsv``, ``bundle.npz``) are tracked so the README's worked
example runs with no generation step. ``test_cli_mini_example`` pins the
CLI's numbers on these files.
"""

from pathlib import Path

import numpy as np

from ppb.harmonize import VariantTable
from ppb.io import write_bundle

SEED = 0
M = 8

HERE = Path(__file__).resolve().parent


def main() -> None:
    rng = np.random.default_rng(SEED)
    variants = VariantTable(
        chrom=np.ones(M, dtype=int), pos=np.arange(1, M + 1),
        a1=np.array(["A"] * M), a2=np.array(["G", "C"] * (M // 2)))
    A = rng.standard_normal((300, M))
    Xs = (A - A.mean(0)) / A.std(0)
    D = (Xs.T @ Xs) / Xs.shape[0]
    z = (Xs.T @ ((A[:, 0] - A[:, 0].mean()) / A[:, 0].std())) / Xs.shape[0]
    w = rng.standard_normal(M)

    write_bundle(HERE / "bundle.npz", variants, z, D=D)
    with open(HERE / "weights.tsv", "w", encoding="utf-8") as fh:
        fh.write(
            "chr_name\tchr_position\teffect_allele\tother_allele\t"
            "effect_weight\n")
        for i in range(M):
            fh.write(
                f"1\t{i + 1}\t{variants.a1[i]}\t{variants.a2[i]}\t"
                f"{w[i]:.6f}\n")
    print("wrote weights.tsv and bundle.npz")


if __name__ == "__main__":
    main()
