#!/usr/bin/env python
"""Repack an LD reference from full-square int8 blocks to packed upper triangles.

An LD matrix is symmetric, so the lower triangle is redundant: storing only the
upper triangle halves the reference (``m^2`` -> ``m(m+1)/2`` bytes per block)
and loses nothing. The packed kernel is also parallel over rows where the square
one is serial.

This rewrites v1 references (``block_kind`` absent, all square) into v2
(``format_version=2``, triangles in ``ld8p``). ``ppb.read_ldref`` reads both, so
old files keep working and a repacked file is a drop-in replacement.

``w^T D w`` is not bit-identical across the two layouts -- the packed kernel sums
each off-diagonal pair once and doubles it, so floating-point summation order
differs. The difference is ~17 machine epsilon (~4e-15 relative), eleven orders
of magnitude below int8 quantisation's own ~0.1% error, but it does move
published R^2 in the last digit or two. ``--verify`` measures it per chromosome
on random weights so the claim is checked, not assumed.

Run:
    python scripts/repack_ldref.py data/ldref_hm3_plus/npz  --verify
    python scripts/repack_ldref.py <in_dir> --out <out_dir> --compress
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ppb.ld_backend import DenseLDInt8, PackedDenseLDInt8
from ppb.ldref import read_ldref, write_ldref


def repack_one(src: Path, dst: Path, *, verify=False, compress=False, seed=0,
               verify_rtol=1e-12, verify_atol=1e-12):
    """Repack one chromosome ``.npz``. Returns a stats dict."""
    ref = read_ldref(src)
    variants, ld = ref["variants"], ref["ld"]
    before = sum(getattr(b, "nbytes", 0) for b, _ in ld.blocks)

    blocks = []
    for backend, idx in ld.blocks:
        if isinstance(backend, PackedDenseLDInt8):
            blocks.append((backend, idx))                  # already packed
        elif isinstance(backend, DenseLDInt8):
            blocks.append((backend.packed(), idx))
        else:
            raise TypeError(
                f"{src.name}: cannot repack a {type(backend).__name__} block")
    after = sum(b.nbytes for b, _ in blocks)

    rel = None
    if verify:
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(variants.n)
        old = ld.quad(w)
        new = sum(b.quad(w[i]) for b, i in blocks)
        delta = abs(new - old)
        rel = delta / max(abs(old), np.finfo(float).tiny)
        if not np.isclose(new, old, rtol=verify_rtol, atol=verify_atol):
            raise RuntimeError(
                f"{src.name}: verification failed: old w^T D w={old:.17g}, "
                f"packed={new:.17g}, absolute difference={delta:.3e}, "
                f"relative difference={rel:.3e} (rtol={verify_rtol:.3e}, "
                f"atol={verify_atol:.3e})")

    kwargs = {k: ref[k] for k in ("rsid", "af", "pos_hg38") if k in ref}
    write_ldref(dst, variants, blocks, compress=compress, **kwargs)
    return dict(name=src.name, n=variants.n, blocks=len(blocks),
                mem_before=before, mem_after=after,
                disk_before=src.stat().st_size, disk_after=Path(str(dst)).stat().st_size,
                rel=rel)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src_dir", help="directory of ldref_chr*.npz")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <src_dir>_packed)")
    ap.add_argument("--glob", default="ldref_chr*.npz")
    ap.add_argument("--compress", action="store_true",
                    help="write with savez_compressed")
    ap.add_argument("--verify", action="store_true",
                    help="compare w^T D w before/after on random weights")
    ap.add_argument("--verify-rtol", type=float, default=1e-12,
                    help="relative verification tolerance (default: 1e-12)")
    ap.add_argument("--verify-atol", type=float, default=1e-12,
                    help="absolute verification tolerance (default: 1e-12)")
    args = ap.parse_args(argv)

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out) if args.out else src_dir.with_name(src_dir.name + "_packed")
    files = sorted(src_dir.glob(args.glob))
    if not files:
        ap.error(f"no files matching {args.glob!r} in {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'file':22} {'variants':>9} {'blocks':>7} {'mem MB':>18} "
          f"{'disk MB':>18} {'rel diff':>10}")
    tot = {k: 0 for k in ("mem_before", "mem_after", "disk_before", "disk_after")}
    worst = 0.0
    for f in files:
        t0 = time.time()
        try:
            s = repack_one(
                f, out_dir / f.name, verify=args.verify,
                compress=args.compress, verify_rtol=args.verify_rtol,
                verify_atol=args.verify_atol)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for k in tot:
            tot[k] += s[k]
        if s["rel"] is not None:
            worst = max(worst, s["rel"])
        print(f"{s['name']:22} {s['n']:>9,} {s['blocks']:>7} "
              f"{s['mem_before']/1e6:8.1f} -> {s['mem_after']/1e6:6.1f} "
              f"{s['disk_before']/1e6:8.1f} -> {s['disk_after']/1e6:6.1f} "
              f"{('%.2e' % s['rel']) if s['rel'] is not None else '—':>10}"
              f"   {time.time()-t0:5.1f}s", flush=True)

    print(f"\ntotal in-memory {tot['mem_before']/1e9:.2f} GB -> "
          f"{tot['mem_after']/1e9:.2f} GB  ({tot['mem_before']/tot['mem_after']:.3f}x)")
    print(f"total on-disk   {tot['disk_before']/1e9:.2f} GB -> "
          f"{tot['disk_after']/1e9:.2f} GB  ({tot['disk_before']/tot['disk_after']:.3f}x)")
    if args.verify:
        print(f"worst relative change in w^T D w: {worst:.3e} "
              f"({worst/np.finfo(float).eps:.1f} machine epsilon)")
    print(f"\nwrote {len(files)} file(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
