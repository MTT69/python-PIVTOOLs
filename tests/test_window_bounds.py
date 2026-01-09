"""
Window bounds and positioning tests for PIV processing.

Tests verify:
1. Single mode first_ctr matches standard mode positioning convention
2. Bounds checking for various image sizes
3. Bounds checking for various window sizes
4. Window indices never go negative or exceed image bounds
5. Single mode with various sum_window sizes
6. Edge cases: small images, extreme overlap values
"""

import numpy as np
import pytest
from pivtools_core.window_utils import (
    compute_window_centers,
    compute_window_centers_single_mode,
    compute_padding_for_single_mode,
)


class TestSingleModeFirstCenter:
    """Test that single mode first center matches standard mode convention."""

    @pytest.mark.parametrize("win_size", [4, 8, 16, 32])
    def test_first_center_formula_consistency(self, win_size):
        """
        Verify single mode first center uses same formula as standard mode.

        first_ctr = pad + (win_size - 1) / 2.0

        This ensures the small window grid starts at the same position
        relative to the original image as standard mode would.
        """
        image_shape = (512, 512)
        overlap = 50
        sum_window_size = 64  # Larger than any win_size tested

        # Standard mode
        std_result = compute_window_centers(
            image_shape=image_shape,
            window_size=(win_size, win_size),
            overlap=overlap,
        )

        # Single mode
        single_result = compute_window_centers_single_mode(
            image_shape=image_shape,
            window_size=(win_size, win_size),
            sum_window=(sum_window_size, sum_window_size),
            overlap=overlap,
        )

        # Get padding offset
        pad_top, pad_bottom, pad_left, pad_right = single_result.padding

        # Single mode first center in original image coords = first_ctr - pad
        single_first_x_orig = single_result.win_ctrs_x[0] - pad_left
        single_first_y_orig = single_result.win_ctrs_y[0] - pad_top

        # Should match standard mode first center
        assert np.isclose(single_first_x_orig, std_result.win_ctrs_x[0], atol=0.01), \
            f"X first center mismatch: single_orig={single_first_x_orig}, std={std_result.win_ctrs_x[0]}"

        # Note: Y centers may differ due to anchoring strategy, but the formula should be consistent

    def test_16px_sum_window_first_center_is_7_5(self):
        """
        For 4x4 small window with 16x16 sum_window, first center in padded coords = 7.5.

        Calculation:
        - padding = (16-4)/2 = 6 for each side
        - first_ctr = pad + (win_size - 1) / 2 = 6 + 1.5 = 7.5
        """
        win_size = 4
        sum_window_size = 16
        expected_first = 6 + (win_size - 1) / 2.0  # 6 + 1.5 = 7.5

        result = compute_window_centers_single_mode(
            image_shape=(512, 512),
            window_size=(win_size, win_size),
            sum_window=(sum_window_size, sum_window_size),
            overlap=50,
        )

        assert np.isclose(result.win_ctrs_x[0], expected_first), \
            f"Expected first_ctr_x={expected_first}, got {result.win_ctrs_x[0]}"

    def test_standard_mode_16px_first_center_is_7_5(self):
        """Verify standard mode formula: first_ctr = (win_size - 1) / 2.0 = 7.5 for 16px."""
        result = compute_window_centers(
            image_shape=(512, 512),
            window_size=(16, 16),
            overlap=50,
        )

        expected_first = (16 - 1) / 2.0  # 7.5
        assert np.isclose(result.win_ctrs_x[0], expected_first), \
            f"Expected first_ctr_x={expected_first}, got {result.win_ctrs_x[0]}"


class TestWindowBoundsVariousImageSizes:
    """Test bounds checking for various image sizes."""

    @pytest.mark.parametrize("image_size", [256, 512, 1024, 2048])
    def test_standard_mode_bounds(self, image_size):
        """All windows should fit within image bounds for standard mode."""
        window_size = 64
        overlap = 50

        result = compute_window_centers(
            image_shape=(image_size, image_size),
            window_size=(window_size, window_size),
            overlap=overlap,
        )

        half_win = (window_size - 1) / 2.0

        # Check X bounds
        assert result.win_ctrs_x[0] - half_win >= 0, \
            f"First X center too close to left edge: {result.win_ctrs_x[0]}"
        assert result.win_ctrs_x[-1] + half_win < image_size, \
            f"Last X center too close to right edge: {result.win_ctrs_x[-1]}"

        # Check Y bounds (note: ascending order, so [0] is smallest, [-1] is largest)
        assert result.win_ctrs_y[0] - half_win >= 0, \
            f"First Y center too close to top edge: {result.win_ctrs_y[0]}"
        assert result.win_ctrs_y[-1] + half_win < image_size, \
            f"Last Y center too close to bottom edge: {result.win_ctrs_y[-1]}"

    @pytest.mark.parametrize("image_size", [256, 512, 1024, 2048])
    def test_single_mode_bounds_in_padded_coords(self, image_size):
        """Single mode SumWindow extractions should fit within padded image."""
        window_size = 4
        sum_window_size = 16
        overlap = 50

        result = compute_window_centers_single_mode(
            image_shape=(image_size, image_size),
            window_size=(window_size, window_size),
            sum_window=(sum_window_size, sum_window_size),
            overlap=overlap,
        )

        pad_top, pad_bottom, pad_left, pad_right = result.padding
        padded_size = image_size + pad_left + pad_right

        half_sum_win = (sum_window_size - 1) / 2.0

        # Check bounds in padded image
        assert result.win_ctrs_x[0] - half_sum_win >= 0, \
            f"First X SumWindow extraction starts before padded image: {result.win_ctrs_x[0] - half_sum_win}"
        assert result.win_ctrs_x[-1] + half_sum_win < padded_size, \
            f"Last X SumWindow extraction exceeds padded image: {result.win_ctrs_x[-1] + half_sum_win} >= {padded_size}"


class TestWindowBoundsVariousWindowSizes:
    """Test bounds checking for various window sizes."""

    @pytest.mark.parametrize("window_size", [4, 8, 16, 32, 64, 128])
    def test_standard_mode_various_windows(self, window_size):
        """Standard mode should handle various window sizes."""
        image_size = 512
        overlap = 50

        result = compute_window_centers(
            image_shape=(image_size, image_size),
            window_size=(window_size, window_size),
            overlap=overlap,
        )

        assert result.n_win_x >= 1, f"Should have at least 1 window for {window_size}x{window_size}"
        assert result.n_win_y >= 1, f"Should have at least 1 window for {window_size}x{window_size}"

    @pytest.mark.parametrize("window_size", [4, 8, 16, 32])
    def test_single_mode_various_windows(self, window_size):
        """Single mode should handle various small window sizes."""
        image_size = 512
        sum_window_size = 64  # Fixed large sum window
        overlap = 50

        result = compute_window_centers_single_mode(
            image_shape=(image_size, image_size),
            window_size=(window_size, window_size),
            sum_window=(sum_window_size, sum_window_size),
            overlap=overlap,
        )

        assert result.n_win_x >= 1
        assert result.n_win_y >= 1

        # Spacing should be based on small window
        expected_spacing = round((1 - overlap / 100) * window_size)
        assert result.win_spacing_x == expected_spacing, \
            f"Spacing mismatch: expected {expected_spacing}, got {result.win_spacing_x}"
        assert result.win_spacing_y == expected_spacing, \
            f"Spacing mismatch: expected {expected_spacing}, got {result.win_spacing_y}"


class TestExtractedWindowIndices:
    """Test that extracted window indices are valid (simulate C library behavior)."""

    def _simulate_c_extraction(self, center: float, window_size: int) -> tuple:
        """
        Simulate C library window extraction bounds calculation.

        From PIV_2d_cross_correlate.c lines 94-96, 261-262:
        row_min = (int)floor(center - (size-1)/2 + 0.5)

        Returns (row_min, row_max)
        """
        row_min = int(np.floor(center - (window_size - 1) / 2.0 + 0.5))
        row_max = row_min + window_size - 1
        return row_min, row_max

    def test_standard_mode_no_negative_indices(self):
        """Standard mode should never produce negative extraction indices."""
        for image_size in [256, 512, 1024]:
            for window_size in [32, 64, 128]:
                if window_size >= image_size:
                    continue

                result = compute_window_centers(
                    image_shape=(image_size, image_size),
                    window_size=(window_size, window_size),
                    overlap=50,
                )

                for center in result.win_ctrs_x:
                    row_min, row_max = self._simulate_c_extraction(center, window_size)
                    assert row_min >= 0, \
                        f"Negative index: center={center}, window={window_size}, min={row_min}"
                    assert row_max < image_size, \
                        f"Exceeds bounds: center={center}, window={window_size}, max={row_max}, size={image_size}"

    def test_single_mode_no_negative_indices(self):
        """Single mode should never produce negative extraction indices (in padded image)."""
        for image_size in [256, 512, 1024]:
            for window_size, sum_window_size in [(4, 16), (8, 32), (16, 64)]:
                result = compute_window_centers_single_mode(
                    image_shape=(image_size, image_size),
                    window_size=(window_size, window_size),
                    sum_window=(sum_window_size, sum_window_size),
                    overlap=50,
                )

                pad_top, pad_bottom, pad_left, pad_right = result.padding
                padded_size = image_size + pad_left + pad_right

                # Check X centers with SumWindow extraction size
                for center in result.win_ctrs_x:
                    row_min, row_max = self._simulate_c_extraction(center, sum_window_size)
                    assert row_min >= 0, \
                        f"Negative index: center={center}, sum_win={sum_window_size}, min={row_min}"
                    assert row_max < padded_size, \
                        f"Exceeds padded bounds: center={center}, max={row_max}, padded_size={padded_size}"

    def test_first_window_starts_at_pixel_zero(self):
        """
        For both modes, first window extraction should start at pixel 0.

        This verifies the formula first_ctr = (win_size - 1) / 2.0 produces
        row_min = 0 when passed to C extraction.
        """
        # Standard mode: 16px window
        std_result = compute_window_centers(
            image_shape=(512, 512),
            window_size=(16, 16),
            overlap=50,
        )
        row_min, _ = self._simulate_c_extraction(std_result.win_ctrs_x[0], 16)
        assert row_min == 0, f"Standard mode first extraction should start at 0, got {row_min}"

        # Single mode: 4px window with 16px sum_window
        single_result = compute_window_centers_single_mode(
            image_shape=(512, 512),
            window_size=(4, 4),
            sum_window=(16, 16),
            overlap=50,
        )
        # Single mode extracts with sum_window size from padded image
        row_min, _ = self._simulate_c_extraction(single_result.win_ctrs_x[0], 16)
        assert row_min == 0, f"Single mode first extraction should start at 0, got {row_min}"


class TestSingleModeVariousSumWindows:
    """Test single mode with various sum_window sizes."""

    @pytest.mark.parametrize("sum_window_size", [16, 32, 64])
    def test_padding_calculation(self, sum_window_size):
        """Padding should be (sum_window - window) / 2."""
        window_size = 4

        padding = compute_padding_for_single_mode(
            window_size=(window_size, window_size),
            sum_window=(sum_window_size, sum_window_size),
        )

        expected_pad = (sum_window_size - window_size) // 2
        assert padding[0] == expected_pad, f"Top padding: expected {expected_pad}, got {padding[0]}"
        assert padding[1] == expected_pad, f"Bottom padding: expected {expected_pad}, got {padding[1]}"
        assert padding[2] == expected_pad, f"Left padding: expected {expected_pad}, got {padding[2]}"
        assert padding[3] == expected_pad, f"Right padding: expected {expected_pad}, got {padding[3]}"

    @pytest.mark.parametrize("sum_window_size", [16, 32, 64])
    def test_window_count_independent_of_sum_window(self, sum_window_size):
        """Number of windows should depend on small window spacing, not sum_window."""
        image_size = 512
        window_size = 4
        overlap = 50

        result = compute_window_centers_single_mode(
            image_shape=(image_size, image_size),
            window_size=(window_size, window_size),
            sum_window=(sum_window_size, sum_window_size),
            overlap=overlap,
        )

        # Compare to standard mode with same small window
        std_result = compute_window_centers(
            image_shape=(image_size, image_size),
            window_size=(window_size, window_size),
            overlap=overlap,
        )

        # Window counts should match (or be very close)
        assert result.n_win_x == std_result.n_win_x, \
            f"X window count mismatch: single={result.n_win_x}, std={std_result.n_win_x}"
        assert result.n_win_y == std_result.n_win_y, \
            f"Y window count mismatch: single={result.n_win_y}, std={std_result.n_win_y}"


class TestEdgeCases:
    """Test edge cases: small images, extreme overlap."""

    def test_very_small_image(self):
        """Should handle images barely larger than window."""
        result = compute_window_centers(
            image_shape=(70, 70),
            window_size=(64, 64),
            overlap=0,
        )

        assert result.n_win_x == 1
        assert result.n_win_y == 1

    def test_overlap_near_zero(self):
        """Should handle 0% overlap."""
        result = compute_window_centers(
            image_shape=(512, 512),
            window_size=(64, 64),
            overlap=0,
        )

        assert result.win_spacing_x == 64
        assert result.win_spacing_y == 64

    def test_overlap_near_maximum(self):
        """Should handle 99% overlap (spacing = 1 pixel)."""
        result = compute_window_centers(
            image_shape=(512, 512),
            window_size=(64, 64),
            overlap=99,
        )

        # 99% overlap means spacing = round(0.01 * 64) = 1
        assert result.win_spacing_x == 1
        assert result.win_spacing_y == 1

        # Many more windows with high overlap
        assert result.n_win_x > 400  # Approximately (512 - 64) / 1 + 1

    def test_single_mode_minimum_sum_window(self):
        """Sum window equal to window size (edge case, no padding needed)."""
        result = compute_window_centers_single_mode(
            image_shape=(512, 512),
            window_size=(8, 8),
            sum_window=(8, 8),  # Same size
            overlap=50,
        )

        # Padding should be zero
        assert result.padding == (0, 0, 0, 0)

        # Should match standard mode exactly
        std_result = compute_window_centers(
            image_shape=(512, 512),
            window_size=(8, 8),
            overlap=50,
        )

        np.testing.assert_array_almost_equal(
            result.win_ctrs_x, std_result.win_ctrs_x,
            err_msg="X centers should match standard mode when sum_window == window_size"
        )


class TestPaddedCoordinateConversion:
    """Test conversion between original and padded coordinates."""

    def test_padding_offset_correctly_applied(self):
        """Verify single mode centers are correctly offset by padding."""
        image_size = 512
        window_size = 4
        sum_window_size = 16

        result = compute_window_centers_single_mode(
            image_shape=(image_size, image_size),
            window_size=(window_size, window_size),
            sum_window=(sum_window_size, sum_window_size),
            overlap=50,
        )

        pad_top, pad_bottom, pad_left, pad_right = result.padding

        # Verify padding is (sum - win) / 2 = (16 - 4) / 2 = 6
        assert pad_left == 6
        assert pad_top == 6

        # First center in padded coords should be pad + (win - 1) / 2
        expected_first_x = pad_left + (window_size - 1) / 2.0  # 6 + 1.5 = 7.5
        assert np.isclose(result.win_ctrs_x[0], expected_first_x), \
            f"Expected first_ctr_x={expected_first_x}, got {result.win_ctrs_x[0]}"

    def test_sum_window_extraction_valid_at_all_centers(self):
        """Verify SumWindow can be extracted at every center without going out of bounds."""
        for image_size in [256, 512]:
            for win_size, sum_win_size in [(4, 16), (8, 32), (4, 32)]:
                result = compute_window_centers_single_mode(
                    image_shape=(image_size, image_size),
                    window_size=(win_size, win_size),
                    sum_window=(sum_win_size, sum_win_size),
                    overlap=50,
                )

                pad_top, pad_bottom, pad_left, pad_right = result.padding
                padded_size = image_size + pad_left + pad_right
                half_sum = (sum_win_size - 1) / 2.0

                # Check all X centers
                for i, center in enumerate(result.win_ctrs_x):
                    assert center - half_sum >= 0, \
                        f"X center {i}: left edge negative ({center - half_sum})"
                    assert center + half_sum < padded_size, \
                        f"X center {i}: right edge exceeds bounds ({center + half_sum} >= {padded_size})"

                # Check all Y centers
                for i, center in enumerate(result.win_ctrs_y):
                    assert center - half_sum >= 0, \
                        f"Y center {i}: top edge negative ({center - half_sum})"
                    assert center + half_sum < padded_size, \
                        f"Y center {i}: bottom edge exceeds bounds ({center + half_sum} >= {padded_size})"
