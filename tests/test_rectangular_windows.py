"""
Comprehensive tests for rectangular window support in PyPIVTools PIV processing.

Verifies that window_size = (height, width) tuples where height != width are
correctly handled throughout:
- Window center computation
- Single mode padding
- Window weight creation
- Gaussian fitting with C library
- Coordinate grid generation

Convention: window_size is always (height, width) = (rows, cols)
            Arrays are row-major: array[y, x]

Usage:
    pytest tests/test_rectangular_windows.py -v
    pytest tests/test_rectangular_windows.py -v -m "not slow"  # Skip end-to-end
"""
import ctypes
import numpy as np
import pytest

from pivtools_core.window_utils import (
    compute_window_centers,
    compute_window_centers_single_mode,
    compute_padding_for_single_mode,
)
from pivtools_cli.piv.piv_backend.gaussian_fitting import (
    _load_marquadt_lib,
    set_offset_fitting,
)
from tests.test_initial_guess import generate_2d_gaussian


# =============================================================================
# Helper Functions
# =============================================================================

def generate_stacked_planes(shape, params_dict):
    """
    Generate stacked AA, BB, AB correlation planes from dict params.

    Parameters
    ----------
    shape : tuple
        (height, width) of output planes
    params_dict : dict
        Dictionary with keys: amp_A, amp_B, amp_AB, c_A, c_B, c_AB,
        sx_A, sy_A, sxy_A, sx_AB, sy_AB, sxy_AB, x0_A, y0_A, x0_AB, y0_AB

    Returns
    -------
    AA, BB, AB : np.ndarray
        Correlation plane arrays
    """
    h, w = shape
    Y, X = np.meshgrid(np.arange(1, h+1), np.arange(1, w+1), indexing='ij')

    # Extract params from dict
    amp_A, amp_B, amp_AB = params_dict['amp_A'], params_dict['amp_B'], params_dict['amp_AB']
    c_A, c_B, c_AB = params_dict['c_A'], params_dict['c_B'], params_dict['c_AB']
    sx_A, sy_A, sxy_A = params_dict['sx_A'], params_dict['sy_A'], params_dict['sxy_A']
    sx_AB, sy_AB, sxy_AB = params_dict['sx_AB'], params_dict['sy_AB'], params_dict['sxy_AB']
    x0_A, y0_A = params_dict['x0_A'], params_dict['y0_A']
    x0_AB, y0_AB = params_dict['x0_AB'], params_dict['y0_AB']

    # Generate planes - AA and BB share A covariance, centered at x0_A, y0_A
    AA = generate_2d_gaussian(X, Y, amp_A, x0_A, y0_A, sx_A, sy_A, sxy_A, c_A)
    BB = generate_2d_gaussian(X, Y, amp_B, x0_A, y0_A, sx_A, sy_A, sxy_A, c_B)

    # AB uses combined covariance (sx_A + sx_AB), centered at x0_AB, y0_AB
    sum_sx = sx_A + sx_AB
    sum_sy = sy_A + sy_AB
    sum_sxy = sxy_A + sxy_AB
    AB = generate_2d_gaussian(X, Y, amp_AB, x0_AB, y0_AB, sum_sx, sum_sy, sum_sxy, c_AB)

    return AA, BB, AB


def fit_single_window(lib, AA, BB, AB, initial_guess, win_size):
    """
    Fit a single rectangular window using the C library.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded Marquadt library
    AA, BB, AB : np.ndarray
        Correlation planes (2D arrays)
    initial_guess : np.ndarray
        16-element initial guess
    win_size : tuple
        (height, width)

    Returns
    -------
    result : np.ndarray
        16-element fitted parameters
    status : int
        1 = success, 0 = failure
    """
    h, w = win_size
    n_per_window = h * w

    # Build coordinate grids (1-based, matching C code)
    Y, X = np.meshgrid(np.arange(1, h+1), np.arange(1, w+1), indexing='ij')
    X1 = Y.ravel(order='C').astype(np.float64)  # Y coordinates
    X2 = X.ravel(order='C').astype(np.float64)  # X coordinates

    # Pack correlation data: [AA | BB | AB]
    y_all = np.concatenate([AA.ravel(), BB.ravel(), AB.ravel()]).astype(np.float64)

    # Output arrays
    result = np.zeros(16, dtype=np.float64)
    status = np.zeros(1, dtype=np.int32)

    # Call batch C function with separate win_height and win_width
    lib.fit_stacked_gaussian_batch_export(
        ctypes.c_size_t(1),
        ctypes.c_size_t(n_per_window),
        ctypes.c_size_t(h),  # win_height
        ctypes.c_size_t(w),  # win_width
        X2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        initial_guess.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        status.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    return result, status[0]


# =============================================================================
# Unit Tests: Window Center Computation
# =============================================================================

class TestRectangularWindowCenters:
    """Unit tests for window center computation with rectangular windows."""

    @pytest.mark.parametrize("win_size,expected_first_x", [
        ((8, 4), 1.5),    # Tall window: first_x = (4-1)/2 = 1.5
        ((4, 8), 3.5),    # Wide window: first_x = (8-1)/2 = 3.5
        ((16, 8), 3.5),   # 2:1 tall ratio
        ((8, 16), 7.5),   # 1:2 wide ratio
        ((32, 16), 7.5),  # 2:1 larger
    ])
    def test_first_center_x_formula(self, win_size, expected_first_x):
        """
        Verify first X center uses formula: (width - 1) / 2.0.

        For rectangular windows, first_x depends only on width.
        """
        result = compute_window_centers(
            image_shape=(512, 512),
            window_size=win_size,
            overlap=50,
        )

        assert np.isclose(result.win_ctrs_x[0], expected_first_x, atol=0.01), \
            f"X first center: expected {expected_first_x}, got {result.win_ctrs_x[0]}"

    @pytest.mark.parametrize("win_size", [
        (8, 4),   # Tall
        (4, 8),   # Wide
        (16, 8),  # 2:1 tall
        (8, 16),  # 1:2 wide
        (32, 16), # 2:1 larger
    ])
    def test_spacing_is_dimension_specific(self, win_size):
        """
        Verify spacing is computed independently for each dimension.

        win_spacing_y = round((1 - overlap/100) * height)
        win_spacing_x = round((1 - overlap/100) * width)
        """
        overlap = 50
        result = compute_window_centers(
            image_shape=(512, 512),
            window_size=win_size,
            overlap=overlap,
        )

        height, width = win_size
        expected_spacing_y = round((1 - overlap / 100) * height)
        expected_spacing_x = round((1 - overlap / 100) * width)

        assert result.win_spacing_y == expected_spacing_y, \
            f"Y spacing: expected {expected_spacing_y}, got {result.win_spacing_y}"
        assert result.win_spacing_x == expected_spacing_x, \
            f"X spacing: expected {expected_spacing_x}, got {result.win_spacing_x}"

    @pytest.mark.parametrize("win_size", [
        (8, 4),   # Tall: fewer Y windows, more X windows
        (4, 8),   # Wide: more Y windows, fewer X windows
    ])
    def test_window_count_asymmetric(self, win_size):
        """
        Verify asymmetric window counts for rectangular windows.

        Wider windows should produce fewer X windows.
        Taller windows should produce fewer Y windows.
        """
        image_shape = (256, 256)  # Square image
        overlap = 50

        result = compute_window_centers(
            image_shape=image_shape,
            window_size=win_size,
            overlap=overlap,
        )

        height, width = win_size

        # With 50% overlap on square image:
        # - Larger dimension produces fewer windows
        if height > width:
            # Tall window: expect more X windows than Y windows
            assert result.n_win_x > result.n_win_y, \
                f"Tall window should have n_win_x > n_win_y, got {result.n_win_x} vs {result.n_win_y}"
        else:
            # Wide window: expect more Y windows than X windows
            assert result.n_win_y > result.n_win_x, \
                f"Wide window should have n_win_y > n_win_x, got {result.n_win_y} vs {result.n_win_x}"

    def test_bounds_never_exceed_image(self):
        """
        Verify window extraction bounds never exceed image for all rectangular sizes.

        Simulates C library extraction: row_min = floor(center - (size-1)/2 + 0.5)
        """
        image_shape = (512, 512)
        test_sizes = [(8, 4), (4, 8), (16, 8), (8, 16), (32, 16), (16, 32)]

        for win_size in test_sizes:
            result = compute_window_centers(
                image_shape=image_shape,
                window_size=win_size,
                overlap=50,
            )

            height, width = win_size

            # Check X bounds
            for center_x in result.win_ctrs_x:
                col_min = int(np.floor(center_x - (width - 1) / 2.0 + 0.5))
                col_max = col_min + width - 1
                assert col_min >= 0, f"Negative col_min for win_size={win_size}"
                assert col_max < image_shape[1], f"col_max exceeds image width for win_size={win_size}"

            # Check Y bounds
            for center_y in result.win_ctrs_y:
                row_min = int(np.floor(center_y - (height - 1) / 2.0 + 0.5))
                row_max = row_min + height - 1
                assert row_min >= 0, f"Negative row_min for win_size={win_size}"
                assert row_max < image_shape[0], f"row_max exceeds image height for win_size={win_size}"

    def test_consistent_grid_dimensions(self):
        """Verify grid dimensions are computed consistently for different aspect ratios."""
        image_shape = (256, 256)

        # Compute with tall window
        result_tall = compute_window_centers(image_shape, (16, 8), 50)
        # Compute with wide window (rotated)
        result_wide = compute_window_centers(image_shape, (8, 16), 50)

        # Should have swapped n_win_x and n_win_y
        assert result_tall.n_win_x == result_wide.n_win_y
        assert result_tall.n_win_y == result_wide.n_win_x


# =============================================================================
# Unit Tests: Single Mode Padding
# =============================================================================

class TestRectangularSingleModePadding:
    """Unit tests for single mode padding with rectangular windows."""

    @pytest.mark.parametrize("window_size,sum_window,expected_padding", [
        # Symmetric case (baseline)
        ((4, 4), (16, 16), (6, 6, 6, 6)),
        # Rectangular small window, square sum
        ((8, 4), (16, 16), (4, 4, 6, 6)),   # pad_y = (16-8)/2=4, pad_x = (16-4)/2=6
        ((4, 8), (16, 16), (6, 6, 4, 4)),   # pad_y = (16-4)/2=6, pad_x = (16-8)/2=4
        # Both rectangular
        ((8, 4), (32, 16), (12, 12, 6, 6)), # pad_y = (32-8)/2=12, pad_x = (16-4)/2=6
        ((4, 8), (16, 32), (6, 6, 12, 12)), # pad_y = (16-4)/2=6, pad_x = (32-8)/2=12
    ])
    def test_asymmetric_padding_calculation(self, window_size, sum_window, expected_padding):
        """
        Verify padding is computed independently for each dimension.

        pad_top = ceil((sum_h - win_h) / 2)
        pad_bottom = floor((sum_h - win_h) / 2)
        """
        padding = compute_padding_for_single_mode(window_size, sum_window)

        assert padding == expected_padding, \
            f"Padding mismatch for window={window_size}, sum={sum_window}: " \
            f"expected {expected_padding}, got {padding}"

    @pytest.mark.parametrize("window_size,sum_window", [
        ((8, 4), (16, 16)),
        ((4, 8), (16, 16)),
        ((8, 4), (32, 16)),
        ((4, 8), (16, 32)),
    ])
    def test_padding_sum_equals_difference(self, window_size, sum_window):
        """
        Verify total padding equals difference: pad_top + pad_bottom = sum_h - win_h.
        """
        pad_top, pad_bottom, pad_left, pad_right = compute_padding_for_single_mode(
            window_size, sum_window
        )

        win_h, win_w = window_size
        sum_h, sum_w = sum_window

        assert pad_top + pad_bottom == sum_h - win_h, \
            f"Vertical padding sum mismatch: {pad_top}+{pad_bottom} != {sum_h}-{win_h}"
        assert pad_left + pad_right == sum_w - win_w, \
            f"Horizontal padding sum mismatch: {pad_left}+{pad_right} != {sum_w}-{win_w}"

    def test_single_mode_centers_with_rectangular_window(self):
        """
        Verify single mode computes correct centers with rectangular windows.

        Grid spacing should be based on small window dimensions.
        """
        image_shape = (512, 512)
        window_size = (8, 4)   # Tall window
        sum_window = (32, 32)  # Square sum window
        overlap = 50

        result = compute_window_centers_single_mode(
            image_shape=image_shape,
            window_size=window_size,
            sum_window=sum_window,
            overlap=overlap,
        )

        # Spacing based on small window
        assert result.win_spacing_y == 4  # round(0.5 * 8) = 4
        assert result.win_spacing_x == 2  # round(0.5 * 4) = 2

        # Different number of windows in each direction
        # (Tall window with smaller width produces more X windows)
        assert result.n_win_x > result.n_win_y, \
            "Tall small window should have more X windows than Y windows"

    def test_single_mode_centers_bounds_check(self):
        """Verify single mode centers stay within padded bounds."""
        image_shape = (512, 512)
        window_size = (8, 4)
        sum_window = (32, 16)

        result = compute_window_centers_single_mode(
            image_shape=image_shape,
            window_size=window_size,
            sum_window=sum_window,
            overlap=50,
        )

        pad_top, pad_bottom, pad_left, pad_right = result.padding
        padded_h = image_shape[0] + pad_top + pad_bottom
        padded_w = image_shape[1] + pad_left + pad_right

        half_sum_h = (sum_window[0] - 1) / 2.0
        half_sum_w = (sum_window[1] - 1) / 2.0

        # Check all centers
        for center_x in result.win_ctrs_x:
            assert center_x - half_sum_w >= 0, "X extraction starts before padded image"
            assert center_x + half_sum_w < padded_w, "X extraction exceeds padded image"

        for center_y in result.win_ctrs_y:
            assert center_y - half_sum_h >= 0, "Y extraction starts before padded image"
            assert center_y + half_sum_h < padded_h, "Y extraction exceeds padded image"


# =============================================================================
# Unit Tests: Window Weights
# =============================================================================

class TestRectangularWindowWeights:
    """Unit tests for window weight creation with rectangular dimensions."""

    @pytest.fixture
    def correlator(self):
        """Create a minimal correlator instance for testing weight function."""
        from pivtools_cli.piv.piv_backend.base import CrossCorrelator

        class TestCorrelator(CrossCorrelator):
            def correlate_batch(self, images, config, vector_masks=None):
                pass  # Not used in tests

        return TestCorrelator()

    def test_square_weight_shape(self, correlator):
        """Verify square window weight has correct shape."""
        weight = correlator._window_weight_fun((16, 16), 'square')

        assert weight.shape == (16, 16), f"Expected (16, 16), got {weight.shape}"
        assert np.allclose(weight, 1.0), "Square weight should be all ones"

    @pytest.mark.parametrize("win_size", [
        (8, 4),   # Tall
        (4, 8),   # Wide
        (16, 8),  # 2:1 tall
        (8, 16),  # 1:2 wide
    ])
    def test_rectangular_square_weight_shape(self, correlator, win_size):
        """Verify square weight type works with rectangular dimensions."""
        weight = correlator._window_weight_fun(win_size, 'square')

        assert weight.shape == win_size, f"Expected {win_size}, got {weight.shape}"
        assert np.allclose(weight, 1.0), "Square weight should be all ones"

    @pytest.mark.parametrize("win_size", [
        (8, 4),   # Tall
        (4, 8),   # Wide
        (16, 8),  # 2:1 tall
        (8, 16),  # 1:2 wide
    ])
    def test_rectangular_blackman_weight_shape(self, correlator, win_size):
        """Verify Blackman window weight has correct rectangular shape."""
        weight = correlator._window_weight_fun(win_size, 'blackman')

        assert weight.shape == win_size, f"Expected {win_size}, got {weight.shape}"
        assert weight.min() > 0, "Blackman weight should have no zeros"
        assert weight.max() <= 1.0, "Blackman weight should be <= 1.0"

    @pytest.mark.parametrize("win_size", [
        (8, 4),   # Tall
        (4, 8),   # Wide
        (16, 8),  # 2:1 tall
        (8, 16),  # 1:2 wide
    ])
    def test_rectangular_gaussian_weight_shape(self, correlator, win_size):
        """Verify Gaussian window weight has correct rectangular shape."""
        weight = correlator._window_weight_fun(win_size, 'gaussian')

        assert weight.shape == win_size, f"Expected {win_size}, got {weight.shape}"
        # Gaussian peaks at center
        center_y, center_x = win_size[0] // 2, win_size[1] // 2
        assert weight[center_y, center_x] == weight.max(), "Gaussian should peak at center"

    @pytest.mark.parametrize("win_size,sum_window", [
        ((8, 4), (16, 16)),
        ((4, 8), (16, 16)),
        ((8, 4), (32, 16)),
    ])
    def test_singlepix_weight_shape(self, correlator, win_size, sum_window):
        """
        Verify singlepix weight has SumWindow shape with inner region set.

        The small window region should be centered within the SumWindow.
        """
        weight = correlator._window_weight_fun(win_size, 'singlepix', sum_window)

        # Shape should be SumWindow
        assert weight.shape == sum_window, f"Expected {sum_window}, got {weight.shape}"

        # Center region should be 1.0, rest should be 0.0
        nonzero_count = np.count_nonzero(weight)
        expected_nonzero = win_size[0] * win_size[1]
        assert nonzero_count == expected_nonzero, \
            f"Expected {expected_nonzero} nonzero elements, got {nonzero_count}"


# =============================================================================
# Integration Tests: Gaussian Fitting with C Library
# =============================================================================

class TestRectangularGaussianFitting:
    """
    Integration tests for Gaussian fitting with rectangular windows.

    Uses the C library (marquadt_gaussian.c) to verify correct handling
    of separate win_height and win_width parameters.
    """

    @pytest.fixture
    def lib(self):
        """Load the Marquadt C library."""
        return _load_marquadt_lib()

    @pytest.mark.parametrize("win_size,description", [
        ((16, 8), "Tall 2:1 ratio"),
        ((8, 16), "Wide 1:2 ratio"),
        ((32, 16), "Tall 2:1 larger"),
        ((16, 32), "Wide 1:2 larger"),
        ((24, 8), "Tall 3:1 ratio"),
        ((8, 24), "Wide 1:3 ratio"),
    ])
    def test_gaussian_fit_rectangular_window(self, lib, win_size, description):
        """
        Verify Gaussian fitting works correctly with rectangular windows.

        Uses zero-noise synthetic data to verify parameter recovery.
        """
        set_offset_fitting(enabled=True)

        h, w = win_size
        center_x = w / 2 + 1  # 1-based
        center_y = h / 2 + 1  # 1-based

        # Ground truth parameters
        true_params = {
            'amp_A': 100.0, 'amp_B': 100.0, 'amp_AB': 80.0,
            'c_A': 5.0, 'c_B': 5.0, 'c_AB': 5.0,
            'sx_A': 4.0, 'sy_A': 4.0, 'sxy_A': 0.0,
            'sx_AB': 2.0, 'sy_AB': 2.0, 'sxy_AB': 0.0,
            'x0_A': center_x, 'y0_A': center_y,
            'x0_AB': center_x + 1.5, 'y0_AB': center_y + 1.0,
        }

        AA, BB, AB = generate_stacked_planes(win_size, true_params)

        # Use true params as initial guess (zero-noise test)
        initial_guess = np.array([
            true_params['amp_A'], true_params['amp_B'], true_params['amp_AB'],
            true_params['c_A'], true_params['c_B'], true_params['c_AB'],
            true_params['sx_A'], true_params['sy_A'], true_params['sxy_A'],
            true_params['sx_AB'], true_params['sy_AB'], true_params['sxy_AB'],
            true_params['x0_A'], true_params['y0_A'],
            true_params['x0_AB'], true_params['y0_AB'],
        ])

        result, status = fit_single_window(lib, AA, BB, AB, initial_guess, win_size)

        assert status == 1, f"Fitter failed for {description} (status={status})"

        # Verify key parameters recovered within tolerance
        tolerance_pct = 1.0  # 1% relative error

        # Check position recovery (displacement measurement)
        x0_AB_err = abs(result[14] - true_params['x0_AB']) / true_params['x0_AB'] * 100
        y0_AB_err = abs(result[15] - true_params['y0_AB']) / true_params['y0_AB'] * 100

        assert x0_AB_err < tolerance_pct, \
            f"{description}: x0_AB error {x0_AB_err:.3f}% exceeds {tolerance_pct}%"
        assert y0_AB_err < tolerance_pct, \
            f"{description}: y0_AB error {y0_AB_err:.3f}% exceeds {tolerance_pct}%"

        # Check sigma recovery (uncertainty measurement)
        sx_AB_err = abs(result[9] - true_params['sx_AB']) / true_params['sx_AB'] * 100
        sy_AB_err = abs(result[10] - true_params['sy_AB']) / true_params['sy_AB'] * 100

        assert sx_AB_err < tolerance_pct, \
            f"{description}: sx_AB error {sx_AB_err:.3f}% exceeds {tolerance_pct}%"
        assert sy_AB_err < tolerance_pct, \
            f"{description}: sy_AB error {sy_AB_err:.3f}% exceeds {tolerance_pct}%"

    def test_tall_vs_wide_produces_same_displacement(self, lib):
        """
        Verify that fitting with tall vs wide windows produces equivalent results.

        Given the same Gaussian parameters and window center, fitting should
        recover the same displacement regardless of window aspect ratio.
        """
        set_offset_fitting(enabled=True)

        disp_x, disp_y = 2.0, 1.5  # Known displacement

        # Test with tall window
        tall_size = (16, 8)
        h, w = tall_size
        center_tall = (w / 2 + 1, h / 2 + 1)

        params_tall = {
            'amp_A': 100.0, 'amp_B': 100.0, 'amp_AB': 80.0,
            'c_A': 5.0, 'c_B': 5.0, 'c_AB': 5.0,
            'sx_A': 3.0, 'sy_A': 3.0, 'sxy_A': 0.0,
            'sx_AB': 1.5, 'sy_AB': 1.5, 'sxy_AB': 0.0,
            'x0_A': center_tall[0], 'y0_A': center_tall[1],
            'x0_AB': center_tall[0] + disp_x, 'y0_AB': center_tall[1] + disp_y,
        }

        AA_tall, BB_tall, AB_tall = generate_stacked_planes(tall_size, params_tall)
        guess_tall = np.array([
            100.0, 100.0, 80.0, 5.0, 5.0, 5.0,
            3.0, 3.0, 0.0, 1.5, 1.5, 0.0,
            center_tall[0], center_tall[1],
            center_tall[0] + disp_x, center_tall[1] + disp_y,
        ])

        result_tall, status_tall = fit_single_window(
            lib, AA_tall, BB_tall, AB_tall, guess_tall, tall_size
        )

        # Test with wide window
        wide_size = (8, 16)
        h, w = wide_size
        center_wide = (w / 2 + 1, h / 2 + 1)

        params_wide = {
            'amp_A': 100.0, 'amp_B': 100.0, 'amp_AB': 80.0,
            'c_A': 5.0, 'c_B': 5.0, 'c_AB': 5.0,
            'sx_A': 3.0, 'sy_A': 3.0, 'sxy_A': 0.0,
            'sx_AB': 1.5, 'sy_AB': 1.5, 'sxy_AB': 0.0,
            'x0_A': center_wide[0], 'y0_A': center_wide[1],
            'x0_AB': center_wide[0] + disp_x, 'y0_AB': center_wide[1] + disp_y,
        }

        AA_wide, BB_wide, AB_wide = generate_stacked_planes(wide_size, params_wide)
        guess_wide = np.array([
            100.0, 100.0, 80.0, 5.0, 5.0, 5.0,
            3.0, 3.0, 0.0, 1.5, 1.5, 0.0,
            center_wide[0], center_wide[1],
            center_wide[0] + disp_x, center_wide[1] + disp_y,
        ])

        result_wide, status_wide = fit_single_window(
            lib, AA_wide, BB_wide, AB_wide, guess_wide, wide_size
        )

        assert status_tall == 1, "Tall fit failed"
        assert status_wide == 1, "Wide fit failed"

        # Both should recover the same displacement
        disp_tall_x = result_tall[14] - result_tall[12]
        disp_tall_y = result_tall[15] - result_tall[13]
        disp_wide_x = result_wide[14] - result_wide[12]
        disp_wide_y = result_wide[15] - result_wide[13]

        assert np.isclose(disp_tall_x, disp_x, atol=0.05), \
            f"Tall displacement X: expected {disp_x}, got {disp_tall_x}"
        assert np.isclose(disp_tall_y, disp_y, atol=0.05), \
            f"Tall displacement Y: expected {disp_y}, got {disp_tall_y}"
        assert np.isclose(disp_wide_x, disp_x, atol=0.05), \
            f"Wide displacement X: expected {disp_x}, got {disp_wide_x}"
        assert np.isclose(disp_wide_y, disp_y, atol=0.05), \
            f"Wide displacement Y: expected {disp_y}, got {disp_wide_y}"

    def test_gaussian_fit_without_offset(self, lib):
        """Verify rectangular fitting works with offset fitting disabled."""
        set_offset_fitting(enabled=False)

        win_size = (16, 8)  # Tall window
        h, w = win_size
        center_x = w / 2 + 1
        center_y = h / 2 + 1

        true_params = {
            'amp_A': 100.0, 'amp_B': 100.0, 'amp_AB': 80.0,
            'c_A': 0.0, 'c_B': 0.0, 'c_AB': 0.0,  # No offset
            'sx_A': 4.0, 'sy_A': 4.0, 'sxy_A': 0.0,
            'sx_AB': 2.0, 'sy_AB': 2.0, 'sxy_AB': 0.0,
            'x0_A': center_x, 'y0_A': center_y,
            'x0_AB': center_x + 1.0, 'y0_AB': center_y + 0.5,
        }

        AA, BB, AB = generate_stacked_planes(win_size, true_params)

        initial_guess = np.array([
            100.0, 100.0, 80.0, 0.0, 0.0, 0.0,
            4.0, 4.0, 0.0, 2.0, 2.0, 0.0,
            center_x, center_y,
            center_x + 1.0, center_y + 0.5,
        ])

        result, status = fit_single_window(lib, AA, BB, AB, initial_guess, win_size)

        assert status == 1, "Fitter failed without offset"

        # Check displacement recovery
        disp_x = result[14] - result[12]
        disp_y = result[15] - result[13]

        assert np.isclose(disp_x, 1.0, atol=0.05), f"X displacement: expected 1.0, got {disp_x}"
        assert np.isclose(disp_y, 0.5, atol=0.05), f"Y displacement: expected 0.5, got {disp_y}"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestRectangularWindowEdgeCases:
    """Test edge cases and error handling for rectangular windows."""

    def test_very_narrow_window(self):
        """Test extreme aspect ratio (8:1) window."""
        result = compute_window_centers(
            image_shape=(256, 256),
            window_size=(32, 4),  # Very narrow
            overlap=50,
        )

        assert result.n_win_x >= 1, "Should have at least one X window"
        assert result.n_win_y >= 1, "Should have at least one Y window"

        # Narrow width means more X windows
        assert result.n_win_x > result.n_win_y * 4, \
            f"Very narrow window should produce many more X windows: {result.n_win_x} vs {result.n_win_y}"

    def test_very_wide_window(self):
        """Test extreme aspect ratio (1:8) window."""
        result = compute_window_centers(
            image_shape=(256, 256),
            window_size=(4, 32),  # Very wide
            overlap=50,
        )

        assert result.n_win_x >= 1, "Should have at least one X window"
        assert result.n_win_y >= 1, "Should have at least one Y window"

        # Wide height means more Y windows
        assert result.n_win_y > result.n_win_x * 4, \
            f"Very wide window should produce many more Y windows: {result.n_win_y} vs {result.n_win_x}"

    def test_window_larger_than_image_dimension(self):
        """Test behavior when one window dimension exceeds image."""
        # Window width exceeds image width
        with pytest.raises(ValueError, match="exceeds image dimensions"):
            compute_window_centers(
                image_shape=(256, 128),
                window_size=(64, 256),  # Width > image width
                overlap=50,
            )

    def test_sum_window_smaller_than_window_raises(self):
        """Test that sum_window smaller than window raises error."""
        with pytest.raises(ValueError, match="must be >="):
            compute_padding_for_single_mode(
                window_size=(16, 16),
                sum_window=(8, 8),  # Smaller than window
            )

    def test_rectangular_single_mode_bounds(self):
        """Verify single mode extraction never exceeds padded bounds."""
        image_shape = (512, 512)
        window_size = (8, 4)   # Tall small window
        sum_window = (32, 16)  # Rectangular sum window

        result = compute_window_centers_single_mode(
            image_shape=image_shape,
            window_size=window_size,
            sum_window=sum_window,
            overlap=50,
        )

        pad_top, pad_bottom, pad_left, pad_right = result.padding
        padded_h = image_shape[0] + pad_top + pad_bottom
        padded_w = image_shape[1] + pad_left + pad_right

        half_sum_h = (sum_window[0] - 1) / 2.0
        half_sum_w = (sum_window[1] - 1) / 2.0

        # Check all centers
        for center_x in result.win_ctrs_x:
            assert center_x - half_sum_w >= 0, "X extraction starts before padded image"
            assert center_x + half_sum_w < padded_w, "X extraction exceeds padded image"

        for center_y in result.win_ctrs_y:
            assert center_y - half_sum_h >= 0, "Y extraction starts before padded image"
            assert center_y + half_sum_h < padded_h, "Y extraction exceeds padded image"


# =============================================================================
# Main entry point for running tests directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
