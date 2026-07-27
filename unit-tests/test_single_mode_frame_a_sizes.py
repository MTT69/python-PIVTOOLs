"""
test_single_mode_frame_a_sizes.py

Tests for small Frame-A window sizes (SINGLE_MODE_A_SIZES = 4, 6) in ensemble
single mode.

In single mode ``ensemble_piv.window_size`` is NOT an FFT length: the
correlation runs at ``sum_window`` (``cpu_ensemble.py`` ->
``window_sizes_for_computation``) and ``window_size`` only sets the support of
the ``singlepix`` Frame-A weight mask plus the grid spacing. These tests pin the
two layers that consume it, so the claim "already generic in window_size" is
verified rather than assumed:

  * the mask builder (``CrossCorrelator._window_weight_fun`` -> singlepix)
  * the geometry (``compute_padding_for_single_mode``,
    ``compute_window_centers_single_mode``)

The key invariant is centre-of-mass parity with the production 16-in-48 case:
shrinking the A window must not shift where the A mask sits in the sum window,
or every displacement would acquire an offset.

Usage:
    pytest unit-tests/test_single_mode_frame_a_sizes.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_cli.piv.piv_backend.base import CrossCorrelator
from pivtools_core.fft_sizes import SINGLE_MODE_A_SIZES
from pivtools_core.window_utils import (
    compute_padding_for_single_mode,
    compute_window_centers_single_mode,
)

SUM = 48
IMAGE_SHAPE = (512, 512)


# ---------------------------------------------------------------------------
# Frame-A mask (singlepix)
# ---------------------------------------------------------------------------


class TestFrameAMask:
    @pytest.mark.parametrize("m", SINGLE_MODE_A_SIZES)
    def test_mask_is_sum_window_shaped(self, m):
        """The mask is always sum_window shaped -- that is what reaches C."""
        wa = CrossCorrelator._window_weight_fun((m, m), "singlepix", (SUM, SUM))
        assert wa.shape == (SUM, SUM)
        assert wa.dtype == np.float32

    @pytest.mark.parametrize("m", SINGLE_MODE_A_SIZES)
    def test_support_area_is_window_area(self, m):
        wa = CrossCorrelator._window_weight_fun((m, m), "singlepix", (SUM, SUM))
        assert wa.sum() == pytest.approx(float(m * m))
        assert set(np.unique(wa)) == {0.0, 1.0}

    @pytest.mark.parametrize("m", SINGLE_MODE_A_SIZES)
    def test_support_is_one_contiguous_block(self, m):
        wa = CrossCorrelator._window_weight_fun((m, m), "singlepix", (SUM, SUM))
        rows = np.flatnonzero(wa.any(axis=1))
        cols = np.flatnonzero(wa.any(axis=0))
        assert len(rows) == m and len(cols) == m
        # contiguous: index span equals count
        assert rows[-1] - rows[0] == m - 1
        assert cols[-1] - cols[0] == m - 1

    @pytest.mark.parametrize("m", [4, 6, 16])
    def test_centre_of_mass_matches_production_16_in_48(self, m):
        """The A-mask centre must not move when the A window shrinks.

        For any even m, ceil((SUM - m) / 2) + (m - 1) / 2 = (SUM - 1) / 2, so
        4-in-48 and 6-in-48 sit exactly where the production 16-in-48 mask
        sits. This is the load-bearing invariant: were it to differ, the new
        sizes would introduce a displacement offset relative to today's runs.
        """
        wa = CrossCorrelator._window_weight_fun((m, m), "singlepix", (SUM, SUM))
        rows = np.flatnonzero(wa.any(axis=1))
        cols = np.flatnonzero(wa.any(axis=0))
        assert rows.mean() == pytest.approx((SUM - 1) / 2)
        assert cols.mean() == pytest.approx((SUM - 1) / 2)

    def test_rectangular_support(self):
        wa = CrossCorrelator._window_weight_fun((4, 6), "singlepix", (SUM, SUM))
        assert wa.sum() == pytest.approx(24.0)
        assert len(np.flatnonzero(wa.any(axis=1))) == 4
        assert len(np.flatnonzero(wa.any(axis=0))) == 6


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


class TestPadding:
    # expected = (48 - m) / 2
    @pytest.mark.parametrize("m,expected", [(4, 22), (6, 21), (16, 16)])
    def test_symmetric_padding_for_even_sizes(self, m, expected):
        pad = compute_padding_for_single_mode((m, m), (SUM, SUM))
        assert pad == (expected, expected, expected, expected)

    @pytest.mark.parametrize("m", SINGLE_MODE_A_SIZES)
    def test_padding_spans_the_sum_window(self, m):
        pad_top, pad_bottom, pad_left, pad_right = compute_padding_for_single_mode(
            (m, m), (SUM, SUM)
        )
        assert pad_top + m + pad_bottom == SUM
        assert pad_left + m + pad_right == SUM


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------


class TestGridGeometry:
    @pytest.mark.parametrize("m", SINGLE_MODE_A_SIZES)
    def test_spacing_follows_small_window_not_sum_window(self, m):
        """Grid resolution follows the A window -- the point of the feature."""
        result = compute_window_centers_single_mode(
            image_shape=IMAGE_SHAPE,
            window_size=(m, m),
            sum_window=(SUM, SUM),
            overlap=0,
            validate=True,
        )
        assert result.win_spacing_x == m
        assert result.win_spacing_y == m

    @pytest.mark.parametrize("m", SINGLE_MODE_A_SIZES)
    def test_overlap_halves_spacing(self, m):
        result = compute_window_centers_single_mode(
            image_shape=IMAGE_SHAPE,
            window_size=(m, m),
            sum_window=(SUM, SUM),
            overlap=50,
            validate=True,
        )
        assert result.win_spacing_x == m // 2
        assert result.win_spacing_y == m // 2

    @pytest.mark.parametrize("m", SINGLE_MODE_A_SIZES)
    def test_centres_are_consistent_with_spacing_and_count(self, m):
        result = compute_window_centers_single_mode(
            image_shape=IMAGE_SHAPE,
            window_size=(m, m),
            sum_window=(SUM, SUM),
            overlap=0,
            validate=True,
        )
        assert len(result.win_ctrs_x) == result.n_win_x
        assert len(result.win_ctrs_y) == result.n_win_y
        assert np.allclose(np.diff(result.win_ctrs_x), m)
        assert np.allclose(np.diff(result.win_ctrs_y), m)

    def test_window_count_scales_as_inverse_spacing(self):
        """4 px spacing yields 4x the windows of 16 px along each axis.

        This is the cost driver for the feature: plane memory grows as
        1 / spacing^2, not with the FFT size (which is unchanged at sum_window).
        """
        counts = {}
        for m in (4, 16):
            result = compute_window_centers_single_mode(
                image_shape=IMAGE_SHAPE,
                window_size=(m, m),
                sum_window=(SUM, SUM),
                overlap=0,
                validate=True,
            )
            counts[m] = result.n_win_x
        assert counts[4] == pytest.approx(counts[16] * 4, rel=0.02)

    @pytest.mark.parametrize("m", SINGLE_MODE_A_SIZES)
    def test_all_windows_fit_inside_the_padded_image(self, m):
        """Every sum window must lie within the padded image, else C skips it."""
        result = compute_window_centers_single_mode(
            image_shape=IMAGE_SHAPE,
            window_size=(m, m),
            sum_window=(SUM, SUM),
            overlap=0,
            validate=True,
        )
        pad_top, pad_bottom, pad_left, pad_right = result.padding
        padded_h = IMAGE_SHAPE[0] + pad_top + pad_bottom
        padded_w = IMAGE_SHAPE[1] + pad_left + pad_right
        # win_ctrs are returned in padded coords by the util; the accumulator
        # subtracts padding afterwards. Extraction uses the sum window.
        assert result.win_ctrs_y.min() - (SUM - 1) / 2 >= -0.5
        assert result.win_ctrs_x.min() - (SUM - 1) / 2 >= -0.5
        assert result.win_ctrs_y.max() + (SUM + 1) / 2 <= padded_h + 0.5
        assert result.win_ctrs_x.max() + (SUM + 1) / 2 <= padded_w + 0.5

    def test_rectangular_window_size(self):
        result = compute_window_centers_single_mode(
            image_shape=IMAGE_SHAPE,
            window_size=(4, 6),
            sum_window=(SUM, SUM),
            overlap=0,
            validate=True,
        )
        assert result.win_spacing_y == 4
        assert result.win_spacing_x == 6
