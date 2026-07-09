#!/usr/bin/env python3
"""
test_outlier_detection.py

Tests for pivtools_cli/piv/piv_backend/outlier_detection.py, focused on
median_outlier_detection replicating Westerweel & Scarano's PIVware
``pwValidate`` (vector-norm residual, single normalized threshold).

Usage:
    pytest unit-tests/test_outlier_detection.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_cli.piv.piv_backend.outlier_detection import (
    median_outlier_detection,
)


def _shear_field(n=9, slope=2.0):
    """Linear wall-normal shear in ux (like a boundary layer), quiescent uy.

    ux increases by ``slope`` per row so a valid vector differs strongly from
    its wall-normal neighbours; uy is zero everywhere (mean wall-normal ~ 0).
    """
    ux = slope * np.arange(n, dtype=np.float32)[:, None] * np.ones((1, n), np.float32)
    uy = np.zeros((n, n), dtype=np.float32)
    return ux, uy


def test_clean_shear_field_has_no_interior_outliers():
    """Strong-but-smooth shear must NOT be flagged: the median cancels the
    gradient, so the normalized residual is ~0 across the interior.

    Checked with a margin of 2 so every tested node has a full 5×5 window
    (the default ``size``) rather than a truncated one near the border.
    """
    ux, uy = _shear_field()
    mask = median_outlier_detection(ux, uy, epsilon=0.1, threshold=2.0)
    assert not mask[2:-2, 2:-2].any()


def test_window_size_is_selectable():
    """Both a 3×3 and a 5×5 window run and still flag a gross outlier."""
    ux, uy = _shear_field()
    ux[6, 6] = 100.0
    uy[6, 6] = 100.0
    for size in (3, 5):
        mask = median_outlier_detection(ux, uy, epsilon=0.1, threshold=2.0, size=size)
        assert mask[6, 6], f"gross outlier missed at size={size}"


def test_even_or_too_small_window_raises():
    """size must be an odd integer >= 3 (no silent coercion)."""
    ux, uy = _shear_field()
    for bad in (1, 2, 4):
        with pytest.raises(ValueError):
            median_outlier_detection(ux, uy, size=bad)


def test_legitimate_wall_normal_fluctuation_is_kept():
    """A v' fluctuation comparable to the local shear scale is retained.

    The per-component OR (old behaviour) would divide this by the near-zero
    uy MAD and flag it; the PIVware vector-norm normalizes by the combined
    residual (dominated by the ux shear), so the vector survives.
    """
    ux, uy = _shear_field()
    uy[4, 4] = 1.5  # genuine wall-normal fluctuation, ux perfectly consistent
    mask = median_outlier_detection(ux, uy, epsilon=0.1, threshold=2.0)
    assert not mask[4, 4]


def test_gross_wall_normal_spike_is_still_caught():
    """A quiet-component value far above the local shear scale is still an
    outlier under the vector-norm test."""
    ux, uy = _shear_field()
    uy[4, 4] = 10.0  # >> local shear spread -> genuinely spurious
    mask = median_outlier_detection(ux, uy, epsilon=0.1, threshold=2.0)
    assert mask[4, 4]


def test_gross_outlier_is_flagged():
    """A vector wrong in both components is detected."""
    ux, uy = _shear_field()
    ux[6, 6] = 100.0
    uy[6, 6] = 100.0
    mask = median_outlier_detection(ux, uy, epsilon=0.1, threshold=2.0)
    assert mask[6, 6]


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        median_outlier_detection(np.zeros((4, 4)), np.zeros((4, 5)))
