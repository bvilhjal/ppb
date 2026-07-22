"""PUMAS-style subsampling of GWAS summary statistics.

PUMAS (Zhao et al., Genome Biology 2021,
doi:10.1186/s13059-021-02479-9) trains a score on pseudo-training summary
statistics and evaluates it on the paired pseudo-validation statistics.  This
module implements that repeated-learning idea on PPB's standardized scale.

Let ``z = (1 / N) X.T @ y``, ``D = (1 / N) X.T @ X`` and
``t = N z``.  For jointly Gaussian genotypes and phenotype, the per-sample
cross-product has moment covariance

    V = var_y D + z z.T.                                      (Equation 1)

The conditional pseudo-split used here is

    t_train | t ~ N((N_train / N) t,
                    (N_train N_val / N) V),                   (Equation 2)

with ``t_val = t - t_train``.  Equation 1 is the Gaussian full-LD extension of
the signal-dependent diagonal/off-diagonal moments in Zhao et al. (their
Equations 8--11).  The paper's implementation uses LD-pruned variants and
per-variant regression standard errors, so this remains PUMAS-*style*, not a
bit-exact reimplementation.

For a quantitative phenotype, each paired score is evaluated as

    R2 = (w.T @ z_val)^2 / ((w.T @ D @ w) var_y).              (Equation 3)

Squaring a noisy validation numerator gives a positive finite-validation bias.
``pumas_r2`` exposes the paper-style raw statistic and a conditional plug-in
correction.  The correction is exact for weights independent of the pseudo
split; with fitted weights it is an approximation.  For binary phenotypes,
Equation 3 is only an approximate and less interpretable performance measure,
as Zhao et al. explicitly note.
"""

from __future__ import annotations

import operator

import numpy as np

from .ld_backend import DenseLD


def _dense_ld(ld) -> np.ndarray:
    if isinstance(ld, DenseLD):
        arr = ld.D
    else:
        arr = np.asarray(ld, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise ValueError(
            "PUMAS subsampling needs a non-empty square dense LD matrix")
    if not np.isfinite(arr).all():
        raise ValueError("D must contain only finite numbers")
    if not np.allclose(arr, arr.T, rtol=1e-8, atol=1e-10):
        raise ValueError("D must be symmetric")
    if np.any(np.diag(arr) <= 0.0):
        raise ValueError("D must have a strictly positive diagonal")
    eigenvalues = np.linalg.eigvalsh(arr)
    tol = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues[0] < -tol:
        raise ValueError(
            "D must be positive semi-definite; smallest eigenvalue is "
            f"{eigenvalues[0]:.6g}")
    return np.asarray(arr, dtype=np.float64)


def _positive_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    try:
        finite_value = np.isfinite(float(value))
    except OverflowError:
        finite_value = False
    if not finite_value:
        raise ValueError(f"{name} is too large for floating-point calculations")
    return value


def _var_y(value) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("var_y must be finite and strictly positive")
    return value


def _inputs(z_full, D, var_y):
    z_full = np.asarray(z_full, dtype=np.float64)
    if z_full.ndim != 1 or z_full.size == 0:
        raise ValueError("z_full must be a non-empty 1-D vector")
    if not np.isfinite(z_full).all():
        raise ValueError("z_full must contain only finite numbers")
    D = _dense_ld(D)
    if D.shape != (z_full.size, z_full.size):
        raise ValueError(
            f"D has shape {D.shape}, expected ({z_full.size}, {z_full.size})")
    return z_full, D, _var_y(var_y)


def _moment_covariance(z_full, D, var_y):
    with np.errstate(over="ignore", invalid="ignore"):
        moment = var_y * D + np.outer(z_full, z_full)
    if not np.isfinite(moment).all():
        raise ValueError("the moment covariance is not finite")
    return moment


def _psd_sqrt(matrix):
    """Return ``L`` with ``L @ L.T == matrix``, allowing singular PSD input."""
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(matrix)
        tol = 1e-10 * max(1.0, float(np.max(np.abs(values))))
        if values[0] < -tol:
            raise ValueError(
                "var_y * D + outer(z_full, z_full) must be positive "
                f"semi-definite; smallest eigenvalue is {values[0]:.6g}")
        return vectors * np.sqrt(np.maximum(values, 0.0))


def _draw_split(z_full, n_full, n_train, rng, cov_sqrt):
    n_val = n_full - n_train
    with np.errstate(over="ignore", invalid="ignore"):
        t_full = n_full * z_full
    if not np.isfinite(t_full).all():
        raise ValueError("n_full * z_full is not finite")
    mean = (n_train / n_full) * t_full
    scale = np.sqrt((n_train / n_full) * n_val)
    if not np.isfinite(scale):
        raise ValueError("the subsampling scale is not finite")
    noise = np.asarray(rng.standard_normal(z_full.size), dtype=np.float64)
    if noise.shape != z_full.shape or not np.isfinite(noise).all():
        raise ValueError(
            "rng.standard_normal(p) must return a finite vector of length p")
    with np.errstate(over="ignore", invalid="ignore"):
        t_train = mean + scale * (cov_sqrt @ noise)
        t_val = t_full - t_train
        z_train, z_val = t_train / n_train, t_val / n_val
    if not np.isfinite(z_train).all() or not np.isfinite(z_val).all():
        raise ValueError("subsampled summary statistics are not finite")
    return z_train, z_val


def subsample_sumstats(
    z_full,
    D,
    n_full,
    n_train,
    rng,
    *,
    var_y=1.0,
    cov_sqrt=None,
):
    """Draw one paired PUMAS-style train/validation summary-statistic split.

    Returns ``(z_train, z_val)`` on the marginal ``(1 / n) X.T @ y`` scale.
    ``cov_sqrt`` may be a precomputed square root of Equation 1; it is useful
    when repeatedly sampling the same inputs.
    """
    z_full, D, var_y = _inputs(z_full, D, var_y)
    n_full = _positive_int(n_full, "n_full")
    n_train = _positive_int(n_train, "n_train")
    if n_train >= n_full:
        raise ValueError("require 0 < n_train < n_full")
    if cov_sqrt is None:
        cov_sqrt = _psd_sqrt(_moment_covariance(z_full, D, var_y))
    else:
        cov_sqrt = np.asarray(cov_sqrt, dtype=np.float64)
        if cov_sqrt.shape != D.shape or not np.isfinite(cov_sqrt).all():
            raise ValueError(
                f"cov_sqrt must be finite with shape {D.shape}; got "
                f"{cov_sqrt.shape}")
    return _draw_split(z_full, n_full, n_train, rng, cov_sqrt)


def _weights(value, p, source):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (p,):
        raise ValueError(
            f"{source} returned weights with shape {value.shape}, expected ({p},)")
    if not np.isfinite(value).all():
        raise ValueError(f"{source} returned non-finite weights")
    return value


def pumas_r2(
    z_full,
    D,
    n_full,
    rng,
    *,
    fit=None,
    independent_weights=None,
    frac_val=0.25,
    n_reps=20,
    var_y=1.0,
    validation_bias="auto",
):
    """Estimate repeated-learning prediction ``R2`` from one GWAS.

    Exactly one score source is required:

    - ``fit(z_train)`` constructs weights separately from every pseudo-training
      split, which is the PUMAS model-tuning workflow.
    - ``independent_weights`` evaluates a fixed score that was constructed
      independently of ``z_full``.  Naming the independence requirement is
      intentional; weights trained on the full input GWAS would leak validation
      information.

    ``validation_bias='none'`` returns the mean of the paper-style squared
    pseudo-validation numerators.  ``'conditional'`` subtracts
    ``N_train / (N N_val) * w.T @ V @ w`` before dividing by the denominator in
    Equation 3.  Corrected estimates can be negative; clipping them would simply
    reintroduce positive null bias.  The default ``'auto'`` uses this exact
    conditional correction for independent weights and the paper-style raw
    statistic for fitted weights.  Applying ``'conditional'`` to fitted weights
    is allowed only as an explicit approximation because ``w(z_train)`` and the
    paired validation noise are dependent.
    """
    z_full, D, var_y = _inputs(z_full, D, var_y)
    n_full = _positive_int(n_full, "n_full")
    n_reps = _positive_int(n_reps, "n_reps")
    frac_val = float(frac_val)
    if not np.isfinite(frac_val) or not 0.0 < frac_val < 1.0:
        raise ValueError("frac_val must be finite and in (0, 1)")
    n_train = int(round(n_full * (1.0 - frac_val)))
    if not 0 < n_train < n_full:
        raise ValueError("frac_val leaves an empty training or validation split")
    if (fit is None) == (independent_weights is None):
        raise ValueError("provide exactly one of fit or independent_weights")
    if fit is not None and not callable(fit):
        raise TypeError("fit must be callable")
    if validation_bias not in {"auto", "conditional", "none"}:
        raise ValueError(
            "validation_bias must be 'auto', 'conditional', or 'none'")
    resolved_bias = validation_bias
    if resolved_bias == "auto":
        resolved_bias = "conditional" if fit is None else "none"

    p = z_full.size
    fixed = None
    if independent_weights is not None:
        fixed = _weights(independent_weights, p, "independent_weights")

    moment = _moment_covariance(z_full, D, var_y)
    cov_sqrt = _psd_sqrt(moment)
    ld = DenseLD(D)
    n_val = n_full - n_train
    conditional_scale = (n_train / n_full) / n_val
    estimates = np.empty(n_reps)
    for i in range(n_reps):
        z_train, z_val = _draw_split(
            z_full, n_full, n_train, rng, cov_sqrt)
        w = fixed if fixed is not None else _weights(
            fit(z_train), p, "fit(z_train)")
        den = ld.quad(w)
        if not np.isfinite(den) or den <= 0.0:
            raise ValueError(
                f"replicate {i}: w.T @ D @ w must be finite and positive; "
                f"got {den!r}")
        wz_val = float(np.dot(w, z_val))
        if not np.isfinite(wz_val):
            raise ValueError(f"replicate {i}: w.T @ z_val is not finite")
        with np.errstate(over="ignore", invalid="ignore"):
            numerator_sq = wz_val * wz_val
        if not np.isfinite(numerator_sq):
            raise ValueError(f"replicate {i}: squared numerator is not finite")
        if resolved_bias == "conditional":
            wz_full = float(np.dot(w, z_full))
            if not np.isfinite(wz_full):
                raise ValueError(f"replicate {i}: w.T @ z_full is not finite")
            with np.errstate(over="ignore", invalid="ignore"):
                moment_quad = var_y * den + wz_full * wz_full
                numerator_sq -= conditional_scale * moment_quad
            if not np.isfinite(moment_quad) or not np.isfinite(numerator_sq):
                raise ValueError(
                    f"replicate {i}: validation-bias correction is not finite")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            estimates[i] = numerator_sq / (den * var_y)
        if not np.isfinite(estimates[i]):
            raise ValueError(f"replicate {i}: R2 estimate is not finite")
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(estimates.mean())
    if not np.isfinite(value):
        raise ValueError("mean R2 estimate is not finite")
    return value
