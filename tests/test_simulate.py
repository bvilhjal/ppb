"""Tests for the simulation helpers."""

import numpy as np
import pytest

from ppb.simulate import bn_freqs


def test_bn_freqs_rejects_invalid_fst():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="fst"):
        bn_freqs(rng, 10, 0.0)                   # division by fst -> NaN freqs
    with pytest.raises(ValueError, match="fst"):
        bn_freqs(rng, 10, -0.1)


def test_bn_freqs_polymorphic_and_mean_preserving():
    rng = np.random.default_rng(1)
    fA, fB = bn_freqs(rng, 2000, 0.2)
    assert np.all((fA > 0) & (fA < 1))
    assert np.all((fB > 0) & (fB < 1))
    # E[frequency] equals the shared ancestral mean (0.5 for uniform ancestors)
    assert abs(fA.mean() - 0.5) < 0.03
    assert abs(fB.mean() - 0.5) < 0.03
