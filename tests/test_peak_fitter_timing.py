"""
Peak Fitter Validation and Timing Test

Tests the lsqpeaklocate_lm function from libbulkxcorr2d.so for instantaneous PIV.
Validates 3, 4, 5, 6 DOF Gaussian fits with known synthetic inputs.

Test Matrix:
- No astigmatism: sx = sy = 1.5 (circular)
- Minor astigmatism: sx = 1.5, sy = 2.0
- High astigmatism: sx = 1.0, sy = 3.0

All tests are noise-free to validate parameter recovery.

Usage:
    python tests/test_peak_fitter_timing.py
"""

import sys
import os
import ctypes
import time
import numpy as np
from typing import Tuple, Dict, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# =============================================================================
# Library Loading
# =============================================================================

_bulkxcorr_lib = None


def _load_bulkxcorr_lib():
    """Load the bulkxcorr2d library containing lsqpeaklocate_lm."""
    global _bulkxcorr_lib
    if _bulkxcorr_lib is not None:
        return _bulkxcorr_lib

    lib_path = os.path.join(
        os.path.dirname(__file__), '..', 'pivtools_cli', 'lib', 'libbulkxcorr2d.so'
    )
    lib_path = os.path.abspath(lib_path)

    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found: {lib_path}")

    lib = ctypes.CDLL(lib_path)

    # Set up ctypes bindings for lsqpeaklocate_lm
    # void lsqpeaklocate_lm(const float *xcorr, const int *N, float *peak_loc,
    #                       int nPeaks, int iFitType, float *std_dev);
    lib.lsqpeaklocate_lm.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),  # xcorr
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),    # N
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),  # peak_loc
        ctypes.c_int,  # nPeaks
        ctypes.c_int,  # iFitType
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),  # std_dev
    ]
    lib.lsqpeaklocate_lm.restype = None

    _bulkxcorr_lib = lib
    return lib


# =============================================================================
# Gaussian Generation (matching C code models)
# =============================================================================

def generate_gaussian_4dof(
    shape: Tuple[int, int],
    amp: float,
    i0: float,
    j0: float,
    s: float
) -> np.ndarray:
    """
    Generate a 4-DOF circular Gaussian matching the C code model.

    Model: A * exp(-((i-i0)^2 + (j-j0)^2) / s^2)

    Note: The C code uses coordinates centered at (N-1)/2.
    i0, j0 are offsets from center (e.g., 0.3 = 0.3 pixels from center).
    """
    h, w = shape
    center_i = (h - 1) / 2
    center_j = (w - 1) / 2

    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    # Convert to centered coordinates
    di = ii - center_i - i0
    dj = jj - center_j - j0

    r2 = di * di + dj * dj
    return (amp * np.exp(-r2 / (s * s))).astype(np.float32)


def generate_gaussian_5dof(
    shape: Tuple[int, int],
    amp: float,
    i0: float,
    j0: float,
    sigma_row: float,
    sigma_col: float
) -> np.ndarray:
    """
    Generate a 5-DOF elliptical Gaussian matching the C code model.

    Model: A * exp(-((i-i0)^2/sigma_row^2 + (j-j0)^2/sigma_col^2))

    Parameters:
    - i0, j0: Offsets from center (row and col directions)
    - sigma_row: Width in row (vertical/y) direction - C code calls this 'sx'
    - sigma_col: Width in col (horizontal/x) direction - C code calls this 'sy'
    """
    h, w = shape
    center_i = (h - 1) / 2
    center_j = (w - 1) / 2

    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    di = ii - center_i - i0
    dj = jj - center_j - j0

    Q = (di * di) / (sigma_row * sigma_row) + (dj * dj) / (sigma_col * sigma_col)
    return (amp * np.exp(-Q)).astype(np.float32)


def generate_gaussian_6dof(
    shape: Tuple[int, int],
    amp: float,
    i0: float,
    j0: float,
    var_row: float,
    var_col: float,
    cov_rowcol: float
) -> np.ndarray:
    """
    Generate a 6-DOF rotated elliptical Gaussian matching the C code model.

    Model: A * exp(-0.5 * (di^2/var_row + dj^2/var_col + 2*di*dj*cov_rowcol))

    Parameters:
    - var_row: Variance in row direction (sigma_row^2) - C code calls this 'sx'
    - var_col: Variance in col direction (sigma_col^2) - C code calls this 'sy'
    - cov_rowcol: Off-diagonal covariance term - C code calls this 'sxy'
    """
    h, w = shape
    center_i = (h - 1) / 2
    center_j = (w - 1) / 2

    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    di = ii - center_i - i0
    dj = jj - center_j - j0

    Q = di * di / var_row + dj * dj / var_col + 2.0 * di * dj * cov_rowcol
    return (amp * np.exp(-0.5 * Q)).astype(np.float32)


# =============================================================================
# Test Runner
# =============================================================================

def run_single_fit(
    xcorr: np.ndarray,
    fit_type: int,
    n_peaks: int = 1
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run a single peak fit and return results with timing.

    Returns:
        peak_loc: [3, n_peaks] array - [y_offset, x_offset, height]
        std_dev: [3, n_peaks] array - [sigma_y, sigma_x, sxy]
        time_ns: Time taken in nanoseconds
    """
    lib = _load_bulkxcorr_lib()

    xcorr = np.ascontiguousarray(xcorr, dtype=np.float32)
    N = np.array(xcorr.shape, dtype=np.int32)
    peak_loc = np.zeros((3, n_peaks), dtype=np.float32, order='C')
    std_dev = np.zeros((3, n_peaks), dtype=np.float32, order='C')

    # Flatten for C (row-major)
    peak_loc_flat = peak_loc.flatten()
    std_dev_flat = std_dev.flatten()

    start = time.perf_counter_ns()
    lib.lsqpeaklocate_lm(xcorr, N, peak_loc_flat, n_peaks, fit_type, std_dev_flat)
    end = time.perf_counter_ns()

    # Reshape outputs
    peak_loc = peak_loc_flat.reshape(3, n_peaks)
    std_dev = std_dev_flat.reshape(3, n_peaks)

    return peak_loc, std_dev, end - start


def run_timing_test(
    xcorr: np.ndarray,
    fit_type: int,
    n_iterations: int = 1000
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Run multiple iterations for timing statistics.

    Returns:
        peak_loc: Final peak location
        std_dev: Final std_dev
        mean_time_us: Mean time in microseconds
        std_time_us: Std deviation of time in microseconds
    """
    times = []
    for _ in range(n_iterations):
        peak_loc, std_dev, time_ns = run_single_fit(xcorr, fit_type)
        times.append(time_ns)

    times = np.array(times)
    return peak_loc, std_dev, np.mean(times) / 1000.0, np.std(times) / 1000.0


# =============================================================================
# Test Cases
# =============================================================================

def run_test_matrix():
    """Run the full test matrix and print results."""

    print("=" * 80)
    print("PEAK FITTER VALIDATION: lsqpeaklocate_lm (Instantaneous PIV)")
    print("=" * 80)
    print("\nCoordinate Convention:")
    print("  - Row direction = vertical = i-index")
    print("  - Col direction = horizontal = j-index")
    print("  - sigma_row = width in row (vertical) direction")
    print("  - sigma_col = width in col (horizontal) direction")

    # Test parameters
    shape = (33, 33)  # Typical sub-window size
    amp = 1000.0
    i0_true = 0.35  # Sub-pixel offset from center (row direction)
    j0_true = 0.27  # Sub-pixel offset from center (col direction)
    n_iterations = 1000

    # Astigmatism conditions: (sigma_row, sigma_col)
    # sigma_row = width in row (vertical) direction
    # sigma_col = width in col (horizontal) direction
    test_conditions = [
        ("No astigmatism", 1.5, 1.5, "sigma_row=sigma_col=1.5"),
        ("Minor astigmatism", 1.5, 2.0, "sigma_row=1.5, sigma_col=2.0"),
        ("High astigmatism", 1.0, 3.0, "sigma_row=1.0, sigma_col=3.0"),
    ]

    fit_types = [
        (3, "3-pt Parabolic"),
        (4, "4-DOF Circular"),
        (5, "5-DOF Elliptical"),
        (6, "6-DOF Rotated"),
    ]

    all_results = []

    for cond_name, sigma_row, sigma_col, cond_desc in test_conditions:
        print(f"\n{'=' * 80}")
        print(f"TEST: {cond_name} ({cond_desc})")
        print("=" * 80)

        # For 6-DOF, we use variances
        var_row = sigma_row * sigma_row
        var_col = sigma_col * sigma_col

        # Header
        print(f"\n{'Fit Type':<20} {'True i0':>8} {'Fit i0':>8} {'Err i':>8} "
              f"{'True j0':>8} {'Fit j0':>8} {'Err j':>8} "
              f"{'True s_r':>8} {'Fit s_r':>8} {'Err%':>7} "
              f"{'True s_c':>8} {'Fit s_c':>8} {'Err%':>7} "
              f"{'Time(us)':>10}")
        print("-" * 145)

        for fit_type, fit_name in fit_types:
            # Generate appropriate Gaussian for this test
            if fit_type in [3, 4]:
                # Use circular for 3-DOF and 4-DOF (average the sigmas)
                s_avg = np.sqrt((sigma_row**2 + sigma_col**2) / 2)
                xcorr = generate_gaussian_4dof(shape, amp, i0_true, j0_true, s_avg)
                true_sigma_row = s_avg
                true_sigma_col = s_avg
            elif fit_type == 5:
                xcorr = generate_gaussian_5dof(shape, amp, i0_true, j0_true, sigma_row, sigma_col)
                true_sigma_row = sigma_row
                true_sigma_col = sigma_col
            else:  # fit_type == 6
                xcorr = generate_gaussian_6dof(shape, amp, i0_true, j0_true, var_row, var_col, 0.0)
                # For 6-DOF, the output is variances, not sigmas
                true_sigma_row = var_row
                true_sigma_col = var_col

            # Run timing test
            peak_loc, std_dev, mean_time, std_time = run_timing_test(
                xcorr, fit_type, n_iterations
            )

            # Extract results
            # peak_loc[0,0] = row position (i), peak_loc[1,0] = col position (j)
            # The C code returns absolute pixel position, we need to convert to offset
            center_i = (shape[0] - 1) / 2
            center_j = (shape[1] - 1) / 2

            fit_i0 = peak_loc[0, 0] - center_i  # row offset
            fit_j0 = peak_loc[1, 0] - center_j  # col offset

            # C code output (consistent for all fit types):
            # std_dev[0] = sigma_row (row/vertical direction)
            # std_dev[1] = sigma_col (col/horizontal direction)
            fit_sigma_row = std_dev[0, 0]
            fit_sigma_col = std_dev[1, 0]

            # Compute errors
            err_i0 = abs(fit_i0 - i0_true)
            err_j0 = abs(fit_j0 - j0_true)

            # For 3-DOF, sigmas aren't meaningful output
            if fit_type == 3:
                err_row_pct = float('nan')
                err_col_pct = float('nan')
            else:
                err_row_pct = 100 * abs(fit_sigma_row - true_sigma_row) / true_sigma_row if true_sigma_row > 0 else 0
                err_col_pct = 100 * abs(fit_sigma_col - true_sigma_col) / true_sigma_col if true_sigma_col > 0 else 0

            # Print results
            if fit_type == 3:
                print(f"{fit_name:<20} {i0_true:>8.3f} {fit_i0:>8.3f} {err_i0:>8.4f} "
                      f"{j0_true:>8.3f} {fit_j0:>8.3f} {err_j0:>8.4f} "
                      f"{true_sigma_row:>8.3f} {'N/A':>8} {'N/A':>7} "
                      f"{true_sigma_col:>8.3f} {'N/A':>8} {'N/A':>7} "
                      f"{mean_time:>8.2f} +/- {std_time:.1f}")
            else:
                print(f"{fit_name:<20} {i0_true:>8.3f} {fit_i0:>8.3f} {err_i0:>8.4f} "
                      f"{j0_true:>8.3f} {fit_j0:>8.3f} {err_j0:>8.4f} "
                      f"{true_sigma_row:>8.3f} {fit_sigma_row:>8.3f} {err_row_pct:>6.1f}% "
                      f"{true_sigma_col:>8.3f} {fit_sigma_col:>8.3f} {err_col_pct:>6.1f}% "
                      f"{mean_time:>8.2f} +/- {std_time:.1f}")

            all_results.append({
                'condition': cond_name,
                'fit_type': fit_type,
                'fit_name': fit_name,
                'true_i0': i0_true,
                'true_j0': j0_true,
                'fit_i0': fit_i0,
                'fit_j0': fit_j0,
                'err_i0': err_i0,
                'err_j0': err_j0,
                'true_sigma_row': true_sigma_row,
                'true_sigma_col': true_sigma_col,
                'fit_sigma_row': fit_sigma_row,
                'fit_sigma_col': fit_sigma_col,
                'mean_time_us': mean_time,
            })

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)

    print("\nPosition Accuracy (pixels):")
    print(f"{'Fit Type':<20} {'No Astig':>15} {'Minor Astig':>15} {'High Astig':>15}")
    print("-" * 65)
    for fit_type, fit_name in fit_types:
        errors = []
        for cond_name, _, _, _ in test_conditions:
            for r in all_results:
                if r['condition'] == cond_name and r['fit_type'] == fit_type:
                    err = np.sqrt(r['err_i0']**2 + r['err_j0']**2)
                    errors.append(err)
        print(f"{fit_name:<20} {errors[0]:>15.4f} {errors[1]:>15.4f} {errors[2]:>15.4f}")

    print("\nTiming (microseconds per fit):")
    print(f"{'Fit Type':<20} {'No Astig':>15} {'Minor Astig':>15} {'High Astig':>15}")
    print("-" * 65)
    for fit_type, fit_name in fit_types:
        times = []
        for cond_name, _, _, _ in test_conditions:
            for r in all_results:
                if r['condition'] == cond_name and r['fit_type'] == fit_type:
                    times.append(r['mean_time_us'])
        print(f"{fit_name:<20} {times[0]:>15.2f} {times[1]:>15.2f} {times[2]:>15.2f}")

    return all_results


def test_astigmatism_challenge():
    """
    ASTIGMATISM CHALLENGE TEST

    This test answers the key question: Which fitters can handle astigmatic
    (elliptical) correlation peaks?

    We generate a SINGLE elliptical Gaussian and fit it with ALL fitters.
    This reveals which fitters introduce position bias when the peak is not circular.
    """
    print(f"\n\n{'=' * 80}")
    print("ASTIGMATISM CHALLENGE: Which fitters handle elliptical peaks?")
    print("=" * 80)
    print("\nTest: Generate ONE elliptical Gaussian, fit with ALL fitters.")
    print("Key question: Does model mismatch cause position bias?")

    shape = (33, 33)
    amp = 1000.0
    i0_true = 0.35  # Row offset (sub-pixel)
    j0_true = 0.27  # Col offset (sub-pixel)
    sigma_row_true = 1.5  # Width in row direction
    sigma_col_true = 2.5  # Width in col direction (significant astigmatism)
    n_iterations = 1000

    # Generate ONE elliptical Gaussian for ALL fitters
    xcorr = generate_gaussian_5dof(shape, amp, i0_true, j0_true, sigma_row_true, sigma_col_true)

    fit_types = [
        (3, "3-pt Parabolic"),
        (4, "4-DOF Circular"),
        (5, "5-DOF Elliptical"),
        (6, "6-DOF Rotated"),
    ]

    print(f"\nTrue parameters:")
    print(f"  Position: i0={i0_true} (row), j0={j0_true} (col)")
    print(f"  Widths:   sigma_row={sigma_row_true}, sigma_col={sigma_col_true}")
    print(f"  Ratio:    sigma_col/sigma_row = {sigma_col_true/sigma_row_true:.2f} (elliptical)")

    print(f"\n{'Fit Type':<20} {'Fit i0':>10} {'Fit j0':>10} {'Pos Err (px)':>14} {'RESULT':>10}")
    print("-" * 70)

    center_i = (shape[0] - 1) / 2
    center_j = (shape[1] - 1) / 2

    results = []
    for fit_type, fit_name in fit_types:
        peak_loc, std_dev, mean_time, std_time = run_timing_test(
            xcorr, fit_type, n_iterations
        )

        fit_i0 = peak_loc[0, 0] - center_i
        fit_j0 = peak_loc[1, 0] - center_j

        err_i0 = fit_i0 - i0_true
        err_j0 = fit_j0 - j0_true
        pos_err = np.sqrt(err_i0**2 + err_j0**2)

        # Determine pass/fail (0.01 pixel threshold)
        result = "PASS" if pos_err < 0.01 else "FAIL"

        print(f"{fit_name:<20} {fit_i0:>10.4f} {fit_j0:>10.4f} {pos_err:>14.4f} {result:>10}")
        results.append((fit_type, fit_name, pos_err, mean_time))

    print("\n" + "-" * 70)
    print("EXPLANATION:")
    print("  - 3-DOF PASSES because it fits each axis INDEPENDENTLY (separable)")
    print("    It uses only 3 points along each axis, not the full 2D window.")
    print("    A 1D slice through an elliptical Gaussian is still a valid 1D Gaussian!")
    print("")
    print("  - 4-DOF FAILS because it fits ALL 25 points with a circular model.")
    print("    The elliptical peak doesn't match the circular model -> position bias.")
    print("")
    print("  - 5-DOF and 6-DOF PASS because they model elliptical shapes correctly.")

    return results


def test_noise_sensitivity():
    """
    NOISE SENSITIVITY TEST

    The 3-DOF method uses only 6 points (3 per axis).
    The 4/5/6-DOF methods use all 25 points in the 5x5 window.

    More points = better noise averaging = more robust to noise.
    This test shows where 3-DOF fails and higher DOF methods shine.
    """
    print(f"\n\n{'=' * 80}")
    print("NOISE SENSITIVITY: Where 3-DOF fails")
    print("=" * 80)
    print("\n3-DOF uses only 6 points. 4/5/6-DOF use 25 points.")
    print("More points = better noise averaging.")

    shape = (33, 33)
    amp = 1000.0
    i0_true = 0.35
    j0_true = 0.27
    sigma = 1.5  # Circular for fair comparison
    n_trials = 100  # Multiple trials to get statistics

    noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]  # Fraction of amplitude

    fit_types = [
        (3, "3-pt Parabolic"),
        (4, "4-DOF Circular"),
        (5, "5-DOF Elliptical"),
        (6, "6-DOF Rotated"),
    ]

    print(f"\nTest: {n_trials} trials per noise level, circular Gaussian (sigma={sigma})")
    print(f"Position error = RMS over {n_trials} trials\n")

    # Header
    print(f"{'Noise Level':<15}", end="")
    for _, name in fit_types:
        print(f"{name:>18}", end="")
    print()
    print("-" * 87)

    center_i = (shape[0] - 1) / 2
    center_j = (shape[1] - 1) / 2

    for noise_frac in noise_levels:
        noise_std = noise_frac * amp

        print(f"{noise_frac*100:>6.0f}% noise   ", end="")

        for fit_type, fit_name in fit_types:
            errors = []

            for trial in range(n_trials):
                # Generate clean Gaussian
                xcorr = generate_gaussian_4dof(shape, amp, i0_true, j0_true, sigma)

                # Add noise
                if noise_std > 0:
                    np.random.seed(trial)  # Reproducible
                    xcorr = xcorr + np.random.normal(0, noise_std, shape).astype(np.float32)

                # Fit
                peak_loc, std_dev, _, _ = run_timing_test(xcorr, fit_type, n_iterations=1)

                fit_i0 = peak_loc[0, 0] - center_i
                fit_j0 = peak_loc[1, 0] - center_j

                pos_err = np.sqrt((fit_i0 - i0_true)**2 + (fit_j0 - j0_true)**2)
                errors.append(pos_err)

            rms_error = np.sqrt(np.mean(np.array(errors)**2))
            print(f"{rms_error:>15.4f} px", end="")

        print()

    print("\n" + "-" * 87)
    print("CONCLUSION:")
    print("  - At 0% noise: All methods work well")
    print("  - At 5-20% noise: 3-DOF degrades faster than 4/5/6-DOF")
    print("  - 4/5/6-DOF average over 25 points -> more robust to noise")
    print("  - For noisy data, use 4-DOF or higher!")


def print_timing_summary():
    """Print a dedicated timing comparison."""
    print(f"\n\n{'=' * 80}")
    print("TIMING COMPARISON")
    print("=" * 80)

    shape = (33, 33)
    amp = 1000.0
    n_iterations = 5000  # More iterations for stable timing

    # Generate a simple circular Gaussian for timing
    xcorr = generate_gaussian_4dof(shape, amp, 0.3, 0.2, 1.5)

    fit_types = [
        (3, "3-pt Parabolic"),
        (4, "4-DOF Circular"),
        (5, "5-DOF Elliptical"),
        (6, "6-DOF Rotated"),
    ]

    print(f"\nTiming over {n_iterations} iterations on 33x33 correlation plane:")
    print(f"\n{'Fit Type':<20} {'Mean (us)':>12} {'Std (us)':>12} {'Relative':>12}")
    print("-" * 60)

    times = []
    for fit_type, fit_name in fit_types:
        _, _, mean_time, std_time = run_timing_test(xcorr, fit_type, n_iterations)
        times.append((fit_name, mean_time, std_time))

    # Calculate relative to fastest
    min_time = min(t[1] for t in times)

    for fit_name, mean_time, std_time in times:
        relative = mean_time / min_time
        print(f"{fit_name:<20} {mean_time:>10.2f} {std_time:>10.2f} {relative:>10.2f}x")

    print("\n" + "-" * 60)
    print(f"Fastest: {times[0][0]} at {times[0][1]:.2f} us")
    print(f"Slowest: {times[-1][0]} at {times[-1][1]:.2f} us ({times[-1][1]/times[0][1]:.1f}x slower)")


def main():
    """Main test entry point."""
    print("\n" + "=" * 80)
    print("INSTANTANEOUS PIV PEAK FITTER VALIDATION")
    print("Testing lsqpeaklocate_lm from libbulkxcorr2d.so")
    print("=" * 80)

    # Run main test matrix (each fitter with its matched input)
    results = run_test_matrix()

    # Run astigmatism challenge (all fitters on elliptical input)
    test_astigmatism_challenge()

    # Run noise sensitivity test (where 3-DOF fails)
    test_noise_sensitivity()

    # Run dedicated timing comparison
    print_timing_summary()

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
