"""Block-level uncertainty and negative controls for the summary-statistic R^2.

The genome-wide estimate is a ratio of sums over LD blocks:

    R^2 = (sum_b u_b)^2 / (sum_b v_b),   u_b = w_b' z_b,   v_b = w_b' D_b w_b.

Both tools here consume exactly those per-block products, which a genome-wide
sweep already computes on its way to the two totals (``scripts/regenerate_results.py``
builds them per chromosome). Neither needs a second pass over the LD reference.

- :func:`r2_block_jackknife` -- a delete-one-group jackknife standard error. A
  point estimate cannot support the comparison a benchmark exists to make; this
  supplies the interval, and it captures genomic heterogeneity, which the
  ``1/N`` finite-sample term does not.
- :func:`sign_flip_null` -- an exact negative control. Because ``D`` is
  block-diagonal, flipping the sign of *every* weight in a block leaves
  ``v_b = w_b' D_b w_b`` unchanged while flipping ``u_b``. So the sign-flipped
  scores are a family of scores with an identical denominator and no genuine
  association, and their null distribution is available in closed form.

Neither detects a mis-scaled ``z``: scaling ``z`` by ``c`` scales the estimate
and its null by ``c^2`` alike, so the ratio between them is invariant. See
``docs/LIMITATIONS.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class BlockJackknife:
    """Delete-one-group jackknife of the block-ratio ``R^2``."""

    r2: float                      # full-sample estimate
    se: float                      # jackknife standard error
    bias: float                    # jackknife bias estimate for the ratio
    n_blocks: int
    n_groups: int
    max_variance_share: float      # largest single group's share of the jackknife variance
    delete_values: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False)

    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop("delete_values")
        return out


@dataclass
class SignFlipNull:
    """Exact block-sign-flip negative control for the block-ratio ``R^2``."""

    r2: float                      # observed estimate
    null_mean: float               # E[R^2] under random block sign flips
    z: float                       # sum(u) / sqrt(sum(u^2))
    ratio: float                   # observed / null_mean
    n_blocks: int
    p_value: float | None = None   # empirical, when draws were requested
    n_draws: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _blocks(u, v):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.ndim != 1 or u.shape != v.shape:
        raise ValueError(
            f"u and v must be 1-D of equal length; got {u.shape} and {v.shape}")
    if u.size == 0:
        raise ValueError("need at least one block")
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("u and v must contain only finite values")
    # An indefinite block understates w^T D w and so inflates R^2. The LD
    # backend rejects one at source (BlockDiagonalLD.quad); reject it here too,
    # since these arrays can also arrive from a file.
    negative = np.flatnonzero(v < 0.0)
    if negative.size:
        raise ValueError(
            f"{negative.size} block(s) have v = w' D_b w < 0 (e.g. index "
            f"{int(negative[0])} = {v[negative[0]]!r}); the LD block is not "
            "positive semi-definite")
    return u, v


def _var_y(value) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("var_y must be finite and strictly positive")
    return value


def r2_from_blocks(u, v, var_y: float = 1.0) -> float:
    """``R^2`` from per-block products: ``(sum u)^2 / (sum v * var_y)``."""
    u, v = _blocks(u, v)
    var_y = _var_y(var_y)
    return _ratio(float(u.sum()), float(v.sum()), var_y)


def _ratio(num: float, den: float, var_y: float) -> float:
    if not den > 0.0:
        raise ValueError(f"sum of w' D_b w = {den!r} is not positive; R^2 is undefined")
    value = (num * num) / (den * var_y)
    if not np.isfinite(value):
        raise ValueError("R^2 is not finite")
    return value


def r2_block_jackknife(u, v, *, groups=None, var_y: float = 1.0) -> BlockJackknife:
    """Delete-one-group jackknife standard error for the genome-wide ``R^2``.

    ``u`` and ``v`` are the per-block ``w_b' z_b`` and ``w_b' D_b w_b``.
    ``groups`` labels the delete-one units; it defaults to one group per block.
    Chromosomes are the more conservative choice when block sizes are very
    uneven -- the shipped HM3+ reference runs from 216 to 17,304 variants per
    block -- because the jackknife weights every group equally regardless of
    how much of the genome it carries.

    ``max_variance_share`` is the fraction of the jackknife variance contributed
    by the single most influential group. Equal contributions give ``1/K``;
    a value far above that means one region of the genome is carrying the
    estimate, which is the failure this diagnostic is most likely to meet on
    real data (a long-range LD region, or the MHC). Note it is deliberately
    *not* the largest shift measured in SE units: a dominating group inflates
    the SE as much as its own deviation, so that ratio is near 1 either way and
    detects nothing.

    The reported ``se`` uses the same convention as the overlap fit in
    ``experiments/overlap_detection.py``:
    ``sqrt((K-1)/K * sum_g (theta_(g) - mean(theta_(.)))^2)``.
    """
    u, v = _blocks(u, v)
    var_y = _var_y(var_y)
    n = u.size

    if groups is None:
        groups = np.arange(n)
    groups = np.asarray(groups)
    if groups.shape != (n,):
        raise ValueError(
            f"groups must have one entry per block ({n},); got {groups.shape}")
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError(
            f"the jackknife needs at least 2 groups; got {unique.size}")

    total_u, total_v = float(u.sum()), float(v.sum())
    full = _ratio(total_u, total_v, var_y)

    delete = np.empty(unique.size, dtype=np.float64)
    for i, group in enumerate(unique):
        keep = groups != group
        den = float(v[keep].sum())
        if not den > 0.0:
            raise ValueError(
                f"deleting group {group!r} leaves sum(w' D_b w) = {den!r}; the "
                "jackknife needs every delete-one subset to have positive score "
                "variance")
        delete[i] = _ratio(float(u[keep].sum()), den, var_y)

    k = unique.size
    mean = float(delete.mean())
    squared = (delete - mean) ** 2
    total = float(squared.sum())
    se = float(np.sqrt((k - 1) / k * total))
    bias = float((k - 1) * (mean - full))
    # All groups identical: no group is more influential than any other, so
    # report the even share rather than 0/0.
    share = float(squared.max() / total) if total > 0.0 else 1.0 / k
    return BlockJackknife(
        r2=full, se=se, bias=bias, n_blocks=n, n_groups=k,
        max_variance_share=share, delete_values=delete)


def sign_flip_null(u, v, *, var_y: float = 1.0, n_draws: int = 0,
                   rng=None) -> SignFlipNull:
    """Exact block-sign-flip negative control for the genome-wide ``R^2``.

    Flipping the sign of every weight in LD block ``b`` sends ``u_b -> -u_b``
    and leaves ``v_b`` exactly unchanged, because ``D`` is block-diagonal and
    ``(-w_b)' D_b (-w_b) = w_b' D_b w_b``. The sign-flipped scores are therefore
    a family of scores with the *same* denominator, the same per-block
    magnitudes, and no coherent association -- a null that costs no extra sweep
    and needs no permuted phenotype.

    Under uniform random signs ``eps_b``, the numerator ``sum_b eps_b u_b`` has
    mean 0 and variance ``sum_b u_b^2``, so

        E[R^2_null] = sum_b u_b^2 / (sum_b v_b * var_y),
        z           = sum_b u_b / sqrt(sum_b u_b^2).

    ``null_mean`` is the accuracy this score would report from block noise alone
    at its own magnitudes -- read a small ``R^2`` against it rather than against
    zero. ``z`` is the sign-flip statistic: how coherently the blocks agree.
    With ``n_draws > 0`` an empirical one-sided p-value is added; the normal
    approximation to ``z`` is usually enough at 431 blocks.

    **``z`` is bounded by ``sqrt(n_blocks)``** by Cauchy-Schwarz, attained when
    every block contributes the same signed amount. So it is a coherence measure
    on a fixed scale, not an unbounded significance statistic: on the shipped
    431-block reference the ceiling is 20.8, and a value near it means every
    block agrees rather than that the evidence is overwhelming. Compare values
    across scores on one reference; do not compare them across references with
    different block counts.

    This is a control for *association*, not for calibration: a uniformly
    mis-scaled ``z`` moves the observed value and the null together.
    """
    u, v = _blocks(u, v)
    var_y = _var_y(var_y)
    total_v = float(v.sum())
    observed = _ratio(float(u.sum()), total_v, var_y)

    sum_sq = float(np.dot(u, u))
    if sum_sq <= 0.0:
        raise ValueError(
            "every block product is zero; the sign-flip null is degenerate")
    null_mean = sum_sq / (total_v * var_y)
    z = float(u.sum() / np.sqrt(sum_sq))

    p_value = None
    n_draws = int(n_draws)
    if n_draws < 0:
        raise ValueError("n_draws must be non-negative")
    if n_draws:
        rng = np.random.default_rng() if rng is None else rng
        signs = rng.integers(0, 2, size=(n_draws, u.size)) * 2 - 1
        draws = np.abs(signs @ u)
        # Add-one so a p-value is never 0, which no finite permutation supports.
        p_value = float((1.0 + np.count_nonzero(draws >= abs(u.sum())))
                        / (1.0 + n_draws))

    return SignFlipNull(
        r2=observed, null_mean=null_mean, z=z,
        ratio=observed / null_mean, n_blocks=u.size,
        p_value=p_value, n_draws=n_draws)


def block_diagnostics(u, v, *, chrom=None, var_y: float = 1.0) -> dict:
    """Jackknife (G2) and sign-flip null (G3) from per-block products.

    The payload matches ``scripts/regenerate_results.py`` so a CLI evaluation
    and a registry pack report the same fields. Returns
    ``diagnostics_unavailable`` instead when there are fewer than two blocks.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.size < 2:
        return {
            "diagnostics_unavailable": (
                f"{u.size} LD block(s): the block jackknife and sign-flip "
                "null need at least 2"),
        }
    block = r2_block_jackknife(u, v, var_y=var_y)
    out = {
        "jackknife": {
            "method": "delete-one-block",
            "se": block.se,
            "n_blocks": block.n_blocks,
            "n_groups": block.n_groups,
            "max_variance_share": block.max_variance_share,
        },
    }
    if chrom is not None:
        chrom = np.asarray(chrom)
        if chrom.shape != (u.size,):
            raise ValueError(
                f"chrom must have one label per block ({u.size},); "
                f"got {chrom.shape}")
        def _chrom_key(c):
            try:
                return (0, int(c))
            except (TypeError, ValueError):
                return (1, str(c))

        order = sorted(dict.fromkeys(chrom.tolist()), key=_chrom_key)
        if len(order) > 1:
            by_chrom = r2_block_jackknife(u, v, groups=chrom, var_y=var_y)
            out["jackknife_chromosome"] = {
                "method": "delete-one-chromosome",
                "se": by_chrom.se,
                "n_blocks": by_chrom.n_blocks,
                "n_groups": by_chrom.n_groups,
                "max_variance_share": by_chrom.max_variance_share,
            }
        out["per_chromosome"] = {
            c: [float(u[chrom == c].sum()), float(v[chrom == c].sum())]
            for c in order
        }
    try:
        control = sign_flip_null(u, v, var_y=var_y)
    except ValueError as exc:
        if "degenerate" not in str(exc):
            raise
        out["diagnostics_unavailable"] = str(exc)
        return out
    out["sign_flip_null"] = {
        "method": "block-sign-flip",
        "null_mean": control.null_mean,
        "z": control.z,
        "ratio": control.ratio,
        "n_blocks": control.n_blocks,
        "z_ceiling": float(np.sqrt(control.n_blocks)),
    }
    return out
