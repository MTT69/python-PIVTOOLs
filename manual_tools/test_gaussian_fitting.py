"""
Test Gaussian Fitting Validation Tool

Generates synthetic correlation planes (AA, BB, AB) and validates the stacked
Gaussian fitter under various conditions, comparing WITH and WITHOUT offset (+C) fitting.

Test Matrix:
- No noise, zero offset: Tests parameter recovery in ideal conditions
- Noisy, zero offset: Tests robustness to noise
- No noise, non-zero offset: Tests offset recovery and need for +C term

Also verifies that ensemble background subtraction produces zero-mean planes.

Usage:
    python manual_tools/test_gaussian_fitting.py
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import ctypes


def generate_2d_gaussian(
    shape: Tuple[int, int],
    amp: float,
    sigma_x: float,
    sigma_y: float,
    sigma_xy: float,
    x0: float,
    y0: float,
    offset: float = 0.0
) -> np.ndarray:
    """
    Generate a 2D Gaussian with full covariance matrix.

    Model: f(x,y) = amp * exp(-0.5 * Q) + offset
    where Q = (x-x0, y-y0)^T * C^{-1} * (x-x0, y-y0)
    and C = [[sigma_x, sigma_xy], [sigma_xy, sigma_y]] (covariance matrix)

    Parameters
    ----------
    shape : tuple
        (height, width) of output array
    amp : float
        Peak amplitude above offset
    sigma_x, sigma_y : float
        Diagonal elements of covariance matrix (variances)
    sigma_xy : float
        Off-diagonal covariance term
    x0, y0 : float
        Peak center position (1-indexed to match C code)
    offset : float
        Background offset (+C term)

    Returns
    -------
    np.ndarray
        2D array containing the Gaussian
    """
    h, w = shape
    # Create coordinate grids (1-indexed to match C code convention)
    y, x = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))

    # Compute inverse covariance matrix
    # C = [[sx, sxy], [sxy, sy]]
    # det(C) = sx*sy - sxy^2
    det_C = sigma_x * sigma_y - sigma_xy * sigma_xy
    if det_C <= 0:
        raise ValueError(f"Covariance matrix is not positive definite: det={det_C}")

    # C^{-1} = (1/det) * [[sy, -sxy], [-sxy, sx]]
    inv_xx = sigma_y / det_C
    inv_yy = sigma_x / det_C
    inv_xy = -sigma_xy / det_C

    # Compute quadratic form Q = dx^T * C^{-1} * dx
    dx = x - x0
    dy = y - y0
    Q = inv_xx * dx * dx + 2 * inv_xy * dx * dy + inv_yy * dy * dy

    return amp * np.exp(-0.5 * Q) + offset


def generate_stacked_planes(
    shape: Tuple[int, int],
    true_params: Dict[str, float],
    noise_std: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate stacked correlation planes (AA, BB, AB) from true parameters.

    The model matches the C library's stacked Gaussian:
    - AA: amp_A * G_A(x,y) + c_A
    - BB: amp_B * G_A(x,y) + c_B  (same Gaussian shape as AA)
    - AB: amp_AB * G_AB(x,y) + c_AB (different position/shape)

    Parameters
    ----------
    shape : tuple
        (height, width) of correlation planes
    true_params : dict
        Dictionary with keys matching 16-parameter layout:
        amp_A, amp_B, amp_AB, c_A, c_B, c_AB,
        sx_A, sy_A, sxy_A, sx_AB, sy_AB, sxy_AB,
        x0_A, y0_A, x0_AB, y0_AB
    noise_std : float
        Standard deviation of Gaussian noise to add (relative to max amplitude)

    Returns
    -------
    tuple of np.ndarray
        (AA, BB, AB) correlation planes
    """
    # Generate AA plane (autocorrelation A)
    AA = generate_2d_gaussian(
        shape,
        amp=true_params['amp_A'],
        sigma_x=true_params['sx_A'],
        sigma_y=true_params['sy_A'],
        sigma_xy=true_params['sxy_A'],
        x0=true_params['x0_A'],
        y0=true_params['y0_A'],
        offset=true_params['c_A']
    )

    # Generate BB plane (autocorrelation B) - same Gaussian shape as AA
    BB = generate_2d_gaussian(
        shape,
        amp=true_params['amp_B'],
        sigma_x=true_params['sx_A'],  # Same shape as AA
        sigma_y=true_params['sy_A'],
        sigma_xy=true_params['sxy_A'],
        x0=true_params['x0_A'],       # Same position as AA
        y0=true_params['y0_A'],
        offset=true_params['c_B']
    )

    # Generate AB plane (cross-correlation) - different position and potentially different shape
    # AB sigma is sum of A and AB sigmas (convolution property)
    AB = generate_2d_gaussian(
        shape,
        amp=true_params['amp_AB'],
        sigma_x=true_params['sx_A'] + true_params['sx_AB'],
        sigma_y=true_params['sy_A'] + true_params['sy_AB'],
        sigma_xy=true_params['sxy_A'] + true_params['sxy_AB'],
        x0=true_params['x0_AB'],
        y0=true_params['y0_AB'],
        offset=true_params['c_AB']
    )

    # Add noise if requested
    if noise_std > 0:
        max_amp = max(true_params['amp_A'], true_params['amp_B'], true_params['amp_AB'])
        noise_scale = noise_std * max_amp
        AA += np.random.normal(0, noise_scale, shape)
        BB += np.random.normal(0, noise_scale, shape)
        AB += np.random.normal(0, noise_scale, shape)

    return AA, BB, AB


# Module-level cache for the library
_marquadt_lib = None


def _load_marquadt_lib():
    """Load the Marquadt library directly."""
    global _marquadt_lib
    if _marquadt_lib is not None:
        return _marquadt_lib

    lib_path = os.path.join(
        os.path.dirname(__file__), '..', 'pivtools_cli', 'lib', 'libmarquadt.so'
    )
    lib_path = os.path.abspath(lib_path)

    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found: {lib_path}")

    lib = ctypes.CDLL(lib_path)

    # Set up ctypes bindings
    lib.set_disable_offset.argtypes = [ctypes.c_int]
    lib.set_disable_offset.restype = None

    _marquadt_lib = lib
    return lib


def set_offset_fitting(enabled: bool = True):
    """Enable or disable offset (+C) fitting in the Gaussian solver."""
    lib = _load_marquadt_lib()
    lib.set_disable_offset(0 if enabled else 1)


def fit_correlation_planes(
    AA: np.ndarray,
    BB: np.ndarray,
    AB: np.ndarray,
    initial_guess: np.ndarray,
    use_offset: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Fit stacked correlation planes using the C library.

    Parameters
    ----------
    AA, BB, AB : np.ndarray
        Correlation planes to fit
    initial_guess : np.ndarray
        16-element initial parameter guess
    use_offset : bool
        If True, fit offsets (+C). If False, fix offsets to zero.

    Returns
    -------
    tuple
        (fitted_params, status) where status=1 is success
    """
    # Set offset fitting mode
    set_offset_fitting(use_offset)

    # Load library
    lib = _load_marquadt_lib()

    h, w = AA.shape
    n_per_window = h * w

    # Create coordinate grids (1-indexed)
    y_coords, x_coords = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
    X1 = y_coords.flatten().astype(np.float64)  # Note: X1 is y-coord in C code
    X2 = x_coords.flatten().astype(np.float64)  # Note: X2 is x-coord in C code

    # Stack correlation planes: [AA, BB, AB]
    y_all = np.concatenate([AA.flatten(), BB.flatten(), AB.flatten()]).astype(np.float64)

    # Prepare output arrays
    out_params = np.zeros(16, dtype=np.float64)
    out_status = np.zeros(1, dtype=np.int32)

    # Call C function
    success = lib.fit_stacked_gaussian_batch_export(
        ctypes.c_size_t(1),  # num_windows
        ctypes.c_size_t(n_per_window),
        X2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        initial_guess.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_status.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )

    # Reset to default (offset enabled)
    set_offset_fitting(True)

    return out_params, out_status[0]


def compute_parameter_errors(true_params: Dict[str, float], fitted: np.ndarray) -> Dict[str, float]:
    """
    Compute relative errors between true and fitted parameters.

    Parameters
    ----------
    true_params : dict
        True parameter values
    fitted : np.ndarray
        16-element fitted parameter array

    Returns
    -------
    dict
        Relative errors (%) for each parameter
    """
    param_names = [
        'amp_A', 'amp_B', 'amp_AB',
        'c_A', 'c_B', 'c_AB',
        'sx_A', 'sy_A', 'sxy_A',
        'sx_AB', 'sy_AB', 'sxy_AB',
        'x0_A', 'y0_A', 'x0_AB', 'y0_AB'
    ]

    errors = {}
    for i, name in enumerate(param_names):
        true_val = true_params[name]
        fitted_val = fitted[i]
        if abs(true_val) > 1e-10:
            errors[name] = 100 * (fitted_val - true_val) / true_val
        else:
            errors[name] = fitted_val  # Absolute error when true is ~0

    return errors


def run_test_case(
    name: str,
    true_params: Dict[str, float],
    noise_std: float,
    use_offset: bool,
    shape: Tuple[int, int] = (65, 65)
) -> Dict:
    """
    Run a single test case.

    Parameters
    ----------
    name : str
        Test case name
    true_params : dict
        True parameters
    noise_std : float
        Noise level (relative to amplitude)
    use_offset : bool
        Whether to fit offsets
    shape : tuple
        Correlation plane shape

    Returns
    -------
    dict
        Results including fitted params, errors, residuals
    """
    # Generate synthetic data
    AA, BB, AB = generate_stacked_planes(shape, true_params, noise_std)

    # Create initial guess (use true values with some perturbation for realism)
    param_names = [
        'amp_A', 'amp_B', 'amp_AB',
        'c_A', 'c_B', 'c_AB',
        'sx_A', 'sy_A', 'sxy_A',
        'sx_AB', 'sy_AB', 'sxy_AB',
        'x0_A', 'y0_A', 'x0_AB', 'y0_AB'
    ]
    initial_guess = np.array([true_params[p] for p in param_names], dtype=np.float64)

    # Perturb initial guess (except positions which are usually well-estimated from peak finding)
    initial_guess[:6] *= 1.1  # Amplitudes and offsets +10%
    initial_guess[6:12] *= 1.2  # Sigmas +20%

    # Fit
    fitted, status = fit_correlation_planes(AA, BB, AB, initial_guess, use_offset)

    # Compute errors
    errors = compute_parameter_errors(true_params, fitted)

    # Compute residual sum of squares
    h, w = shape
    y_coords, x_coords = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))

    # Reconstruct fitted planes
    AA_fit = generate_2d_gaussian(
        shape, fitted[0], fitted[6], fitted[7], fitted[8],
        fitted[12], fitted[13], fitted[3] if use_offset else 0.0
    )
    BB_fit = generate_2d_gaussian(
        shape, fitted[1], fitted[6], fitted[7], fitted[8],
        fitted[12], fitted[13], fitted[4] if use_offset else 0.0
    )
    AB_fit = generate_2d_gaussian(
        shape, fitted[2],
        fitted[6] + fitted[9], fitted[7] + fitted[10], fitted[8] + fitted[11],
        fitted[14], fitted[15], fitted[5] if use_offset else 0.0
    )

    rss_AA = np.sum((AA - AA_fit) ** 2)
    rss_BB = np.sum((BB - BB_fit) ** 2)
    rss_AB = np.sum((AB - AB_fit) ** 2)

    return {
        'name': name,
        'use_offset': use_offset,
        'status': status,
        'fitted': fitted,
        'errors': errors,
        'rss': {'AA': rss_AA, 'BB': rss_BB, 'AB': rss_AB, 'total': rss_AA + rss_BB + rss_AB},
        'planes': {'AA': AA, 'BB': BB, 'AB': AB},
        'fitted_planes': {'AA': AA_fit, 'BB': BB_fit, 'AB': AB_fit}
    }


def verify_background_subtraction():
    """
    Verify that R = <A⋆B> - <A>⋆<B> produces zero-mean correlation planes.

    This tests the math behind ensemble background subtraction.
    """
    print("\n" + "=" * 60)
    print("BACKGROUND SUBTRACTION VERIFICATION")
    print("=" * 60)

    shape = (65, 65)
    center = (shape[0] // 2 + 1, shape[1] // 2 + 1)  # 1-indexed center

    # Generate a correlation plane with known offset (simulates <A⋆B> with background)
    offset = 100.0
    raw_corr = generate_2d_gaussian(
        shape, amp=1000.0,
        sigma_x=3.0, sigma_y=3.0, sigma_xy=0.0,
        x0=center[0], y0=center[1],
        offset=offset
    )

    # Generate the background correlation (simulates <A>⋆<B>)
    # In real ensemble PIV, this is computed from mean images
    # Here we just use a flat plane with the same offset
    background = np.full(shape, offset)

    # Apply subtraction
    corrected = raw_corr - background

    # Check mean
    raw_mean = np.mean(raw_corr)
    bg_mean = np.mean(background)
    corrected_mean = np.mean(corrected)

    print(f"\nRaw correlation mean:        {raw_mean:.4f}")
    print(f"Background mean:             {bg_mean:.4f}")
    print(f"Corrected correlation mean:  {corrected_mean:.4f}")
    print(f"Expected (near zero):        0.0")

    # The Gaussian peak contributes slightly to the mean, so we won't get exactly zero
    # But the offset should be removed
    peak_contribution = 1000.0 * 2 * np.pi * 3.0 / (shape[0] * shape[1])  # Approx integral/area
    print(f"\nExpected residual from Gaussian peak: ~{peak_contribution:.4f}")

    if abs(corrected_mean) < 10.0:  # Should be close to peak contribution
        print("PASS: Background subtraction removes offset as expected")
    else:
        print("FAIL: Corrected mean is too large")

    return corrected_mean


def print_results_table(results: list, true_params_list: list):
    """Print a formatted table of test results."""
    print("\n" + "=" * 80)
    print("PARAMETER RECOVERY RESULTS")
    print("=" * 80)

    # Group by test case (pair results by base name)
    test_cases = {}
    for r in results:
        # Extract base name by removing _with_C or _no_C suffix
        base_name = r['name'].replace('_with_C', '').replace('_no_C', '')
        if base_name not in test_cases:
            test_cases[base_name] = {'results': {}}
        key = 'with_C' if r['use_offset'] else 'no_C'
        test_cases[base_name]['results'][key] = r

    param_names = [
        'amp_A', 'amp_B', 'amp_AB',
        'c_A', 'c_B', 'c_AB',
        'sx_A', 'sy_A', 'sxy_A',
        'sx_AB', 'sy_AB', 'sxy_AB',
        'x0_A', 'y0_A', 'x0_AB', 'y0_AB'
    ]

    # Map base names to their true params
    true_params_map = {
        'clean_no_offset': true_params_list[0],
        'noisy_no_offset': true_params_list[1],
        'clean_with_offset': true_params_list[2],
    }

    for case_name, case_data in test_cases.items():
        modes = case_data['results']
        true_params = true_params_map.get(case_name, {})

        print(f"\n{'='*70}")
        print(f"TEST CASE: {case_name}")
        print(f"{'='*70}")
        print(f"{'Parameter':<12} {'True':>10} {'With +C':>12} {'Err%':>8} {'No +C':>12} {'Err%':>8}")
        print("-" * 70)

        for i, pname in enumerate(param_names):
            true_val = true_params.get(pname, 0)
            fit_with = modes['with_C']['fitted'][i] if 'with_C' in modes else 0
            fit_no = modes['no_C']['fitted'][i] if 'no_C' in modes else 0
            err_with = modes['with_C']['errors'].get(pname, 0) if 'with_C' in modes else 0
            err_no = modes['no_C']['errors'].get(pname, 0) if 'no_C' in modes else 0

            print(f"{pname:<12} {true_val:>10.3f} {fit_with:>12.3f} {err_with:>7.1f}% {fit_no:>12.3f} {err_no:>7.1f}%")

        # Print RSS comparison
        rss_with = modes['with_C']['rss']['total'] if 'with_C' in modes else 0
        rss_no = modes['no_C']['rss']['total'] if 'no_C' in modes else 0
        print("-" * 70)
        print(f"{'RSS Total:':<12} {'':>10} {rss_with:>12.2e} {'':>8} {rss_no:>12.2e}")

        # Determine winner
        if rss_with < rss_no * 0.1:
            print(">>> With +C significantly better")
        elif rss_no < rss_with * 0.1:
            print(">>> Without +C significantly better")
        else:
            print(">>> Similar performance")


def plot_results(results: list, output_dir: str = None):
    """Generate visualization plots."""
    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    for r in results:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f"{r['name']} (offset={'enabled' if r['use_offset'] else 'disabled'})")

        planes = r['planes']
        fitted = r['fitted_planes']

        for i, (name, ax_row) in enumerate(zip(['AA', 'BB', 'AB'], axes.T)):
            # True
            im = ax_row[0].imshow(planes[name], cmap='viridis')
            ax_row[0].set_title(f'{name} True')
            plt.colorbar(im, ax=ax_row[0])

            # Residual
            residual = planes[name] - fitted[name]
            im = ax_row[1].imshow(residual, cmap='RdBu_r', vmin=-np.abs(residual).max(), vmax=np.abs(residual).max())
            ax_row[1].set_title(f'{name} Residual')
            plt.colorbar(im, ax=ax_row[1])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"gauss_fit_{r['name']}.png"), dpi=150)
        plt.close()

    print(f"\nPlots saved to {output_dir}")


def main():
    """Main test runner."""
    print("=" * 60)
    print("GAUSSIAN FITTING VALIDATION TOOL")
    print("=" * 60)

    # Define true parameters for a typical PIV correlation
    shape = (65, 65)
    center = (shape[0] // 2 + 1, shape[1] // 2 + 1)  # 1-indexed center

    # Base true parameters
    base_params = {
        'amp_A': 1000.0,
        'amp_B': 950.0,
        'amp_AB': 900.0,
        'c_A': 0.0,
        'c_B': 0.0,
        'c_AB': 0.0,
        'sx_A': 3.0,      # Variance
        'sy_A': 3.0,
        'sxy_A': 0.0,
        'sx_AB': 1.5,     # Additional variance for AB
        'sy_AB': 1.5,
        'sxy_AB': 0.0,
        'x0_A': float(center[0]),
        'y0_A': float(center[1]),
        'x0_AB': float(center[0]) + 2.5,  # 2.5 pixel displacement
        'y0_AB': float(center[1]) + 1.3,
    }

    results = []

    # Test Case 1: Clean, no offset
    print("\n--- Test Case 1: Clean, No Offset ---")
    params_clean = base_params.copy()
    r1a = run_test_case("clean_no_offset_with_C", params_clean, noise_std=0.0, use_offset=True, shape=shape)
    r1b = run_test_case("clean_no_offset_no_C", params_clean, noise_std=0.0, use_offset=False, shape=shape)
    results.extend([r1a, r1b])
    print(f"  With +C: status={r1a['status']}, RSS={r1a['rss']['total']:.2e}")
    print(f"  No +C:   status={r1b['status']}, RSS={r1b['rss']['total']:.2e}")

    # Test Case 2: Noisy, no offset
    print("\n--- Test Case 2: Noisy, No Offset ---")
    params_noisy = base_params.copy()
    r2a = run_test_case("noisy_no_offset_with_C", params_noisy, noise_std=0.05, use_offset=True, shape=shape)
    r2b = run_test_case("noisy_no_offset_no_C", params_noisy, noise_std=0.05, use_offset=False, shape=shape)
    results.extend([r2a, r2b])
    print(f"  With +C: status={r2a['status']}, RSS={r2a['rss']['total']:.2e}")
    print(f"  No +C:   status={r2b['status']}, RSS={r2b['rss']['total']:.2e}")

    # Test Case 3: Clean, with offset
    print("\n--- Test Case 3: Clean, With Offset ---")
    params_offset = base_params.copy()
    params_offset['c_A'] = 100.0
    params_offset['c_B'] = 80.0
    params_offset['c_AB'] = 50.0
    r3a = run_test_case("clean_with_offset_with_C", params_offset, noise_std=0.0, use_offset=True, shape=shape)
    r3b = run_test_case("clean_with_offset_no_C", params_offset, noise_std=0.0, use_offset=False, shape=shape)
    results.extend([r3a, r3b])
    print(f"  With +C: status={r3a['status']}, RSS={r3a['rss']['total']:.2e}")
    print(f"  No +C:   status={r3b['status']}, RSS={r3b['rss']['total']:.2e}")

    # Collect true params for each test case
    true_params_list = [params_clean, params_noisy, params_offset]

    # Print detailed results
    print_results_table(results, true_params_list)

    # Verify background subtraction
    verify_background_subtraction()

    # Generate plots
    plot_results(results)

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
