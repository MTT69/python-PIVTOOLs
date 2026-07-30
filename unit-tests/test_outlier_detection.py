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
    apply_outlier_detection,
    median_outlier_detection,
    sigma_outlier_detection,
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


# --------------------------------------------------------------------------
# combine="per_component" — for input pairs that are NOT vector components
# (the ensemble stress pair UU/VV). See the function docstring for why.
# --------------------------------------------------------------------------


def _disparate_pair(n=15):
    """Two fields whose local RESIDUAL scales differ ~100x, like (UU, VV).

    Both are checkerboards so every node has a well-defined, non-zero local
    scatter (a perfectly uniform field would make the sigma test degenerate),
    but A's residual is 1.0 and B's is 0.01. A spike in B that is huge on B's
    own scale is therefore negligible inside sqrt(dA^2 + dB^2).
    """
    i, j = np.indices((n, n))
    chk = ((i + j) % 2).astype(np.float32)
    A = (1.0 + 2.0 * chk).astype(np.float32)      # alternates 1.0 / 3.0
    B = (0.25 + 0.02 * chk).astype(np.float32)    # alternates 0.25 / 0.27
    return A, B


def test_norm_mode_is_unchanged_by_the_combine_parameter():
    """Default must stay byte-identical to the explicit vector-norm request."""
    ux, uy = _shear_field()
    ux[6, 6] = 100.0
    default = median_outlier_detection(ux, uy, epsilon=0.1, threshold=2.0)
    explicit = median_outlier_detection(
        ux, uy, epsilon=0.1, threshold=2.0, combine="norm"
    )
    assert np.array_equal(default, explicit)


def test_norm_hides_a_small_component_failure_and_per_component_catches_it():
    """The regression that motivated the mode.

    Corrupt ONLY the small-magnitude field. Its residual is swamped in
    sqrt(dA^2 + dB^2) by the larger field's scale, so the norm misses it; the
    per-component test scales each field by its own neighbourhood and catches it.
    """
    A, B = _disparate_pair()
    B[7, 7] += 0.30  # 30x B's local scatter, 0.3x A's

    norm = median_outlier_detection(
        A, B, epsilon=0.01, threshold=2.0, combine="norm"
    )
    per = median_outlier_detection(
        A, B, epsilon=0.01, threshold=2.0, combine="per_component"
    )
    assert not norm[7, 7], "vector norm unexpectedly caught the small-field spike"
    assert per[7, 7], "per-component test missed the small-field spike"


def test_per_component_equals_the_union_of_single_component_runs():
    """OR-ing two independent verdicts is exactly what the mode must do."""
    A, B = _disparate_pair()
    A[5, 5] += 6.0
    B[9, 3] += 0.30
    zeros = np.zeros_like(A)

    per = median_outlier_detection(A, B, epsilon=0.01, threshold=2.0,
                                   combine="per_component")
    a_only = median_outlier_detection(A, zeros, epsilon=0.01, threshold=2.0,
                                      combine="per_component")
    b_only = median_outlier_detection(zeros, B, epsilon=0.01, threshold=2.0,
                                      combine="per_component")
    assert np.array_equal(per, a_only | b_only)


def test_per_component_is_not_a_superset_of_norm():
    """Documents a real asymmetry rather than asserting a false invariant.

    Two components each moderately off can exceed the threshold jointly under
    the norm (sqrt(2) * 1.5 > 2) while neither exceeds it alone. So the modes
    are not nested, which is why the production stress path must decide which
    behaviour it wants rather than assuming per_component strictly dominates.
    """
    rng = np.random.default_rng(0)
    n = 200
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)

    norm = median_outlier_detection(A, B, epsilon=1e-6, threshold=2.0,
                                    combine="norm")
    per = median_outlier_detection(A, B, epsilon=1e-6, threshold=2.0,
                                   combine="per_component")
    interior = np.zeros_like(norm)
    interior[3:-3, 3:-3] = True  # keep clear of the border guard

    norm_not_per = int((norm & ~per & interior).sum())
    per_not_norm = int((per & ~norm & interior).sum())
    assert norm_not_per > 0, (
        "expected some nodes caught only by the norm (both components "
        "moderately off); the two modes should not be nested"
    )
    assert per_not_norm > 0, "expected some nodes caught only per-component"


def test_unknown_combine_raises():
    ux, uy = _shear_field()
    with pytest.raises(ValueError, match="combine"):
        median_outlier_detection(ux, uy, combine="magnitude")
    with pytest.raises(ValueError, match="combine"):
        sigma_outlier_detection(ux, uy, combine="magnitude")


def test_sigma_per_component_catches_a_small_component_failure():
    """sigma collapses to sqrt(ux^2+uy^2) too, so it needs the same mode."""
    A, B = _disparate_pair()
    B[7, 7] += 0.30
    norm = sigma_outlier_detection(A, B, sigma_threshold=2.0, combine="norm")
    per = sigma_outlier_detection(A, B, sigma_threshold=2.0,
                                  combine="per_component")
    assert per[7, 7]
    assert not norm[7, 7]


def test_apply_outlier_detection_forwards_combine():
    """The dispatcher must pass the mode through to median_2d."""
    A, B = _disparate_pair()
    B[7, 7] += 0.30
    methods = [{"type": "median_2d", "epsilon": 0.01, "threshold": 2.0, "size": 5}]
    norm = apply_outlier_detection(A, B, methods)
    per = apply_outlier_detection(A, B, methods, combine="per_component")
    assert per[7, 7]
    assert not norm[7, 7]
