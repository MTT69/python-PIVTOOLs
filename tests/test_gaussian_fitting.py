"""
Test the C Gaussian fitting library (marquadt_gaussian.c) for parameter recovery.

Verifies that with zero noise, the Levenberg-Marquardt fitter correctly recovers
all 16 Gaussian parameters. Tests both with and without offset (+c) fitting.

Usage:
    python tests/test_gaussian_fitting.py
"""
import ctypes
import numpy as np
from tests.test_initial_guess import generate_2d_gaussian
from pivtools_cli.piv.piv_backend.gaussian_fitting import (
    _load_marquadt_lib,
    set_offset_fitting,
)


def generate_stacked_planes(shape, params_dict, noise_std=0.0):
    """
    Generate stacked AA, BB, AB correlation planes from dict params.

    Wraps generate_2d_gaussian with dict-to-array conversion
    and optional noise injection.

    Parameters
    ----------
    shape : tuple
        (height, width) of output planes
    params_dict : dict
        Dictionary with keys: amp_A, amp_B, amp_AB, c_A, c_B, c_AB,
        sx_A, sy_A, sxy_A, sx_AB, sy_AB, sxy_AB, x0_A, y0_A, x0_AB, y0_AB
    noise_std : float
        Standard deviation of Gaussian noise as fraction of amp_A

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

    # Add noise if requested
    if noise_std > 0:
        noise_scale = amp_A * noise_std
        AA += np.random.randn(*shape) * noise_scale
        BB += np.random.randn(*shape) * noise_scale
        AB += np.random.randn(*shape) * noise_scale

    return AA, BB, AB


__all__ = ['generate_2d_gaussian', 'generate_stacked_planes',
           '_load_marquadt_lib', 'set_offset_fitting']


def fit_single_window(AA, BB, AB, initial_guess, win_size):
    """
    Fit a single window using the C library.

    Parameters
    ----------
    AA, BB, AB : np.ndarray
        Flattened correlation planes
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
    lib = _load_marquadt_lib()

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

    # Call batch C function with 1 window
    import ctypes
    lib.fit_stacked_gaussian_batch_export(
        ctypes.c_size_t(1),
        ctypes.c_size_t(n_per_window),
        X2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        initial_guess.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        status.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    return result, status[0]


def run_test_case(name, true_params, win_size, tolerance=0.5):
    """Run a single test case and report results."""
    param_names = [
        'amp_A', 'amp_B', 'amp_AB',
        'c_A', 'c_B', 'c_AB',
        'sx_A', 'sy_A', 'sxy_A',
        'sx_AB', 'sy_AB', 'sxy_AB',
        'x0_A', 'y0_A', 'x0_AB', 'y0_AB'
    ]

    # Generate synthetic planes
    params_dict = {
        'amp_A': true_params[0], 'amp_B': true_params[1], 'amp_AB': true_params[2],
        'c_A': true_params[3], 'c_B': true_params[4], 'c_AB': true_params[5],
        'sx_A': true_params[6], 'sy_A': true_params[7], 'sxy_A': true_params[8],
        'sx_AB': true_params[9], 'sy_AB': true_params[10], 'sxy_AB': true_params[11],
        'x0_A': true_params[12], 'y0_A': true_params[13],
        'x0_AB': true_params[14], 'y0_AB': true_params[15],
    }
    AA, BB, AB = generate_stacked_planes(win_size, params_dict)

    # Use true params as initial guess
    initial_guess = true_params.copy()

    result, status = fit_single_window(AA, BB, AB, initial_guess, win_size)

    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")

    if status != 1:
        print(f"FITTER FAILED (status={status})")
        return False

    print(f"{'Parameter':<12} {'True':>12} {'Fitted':>12} {'Abs Err':>12} {'Rel Err%':>10} {'Status'}")
    print(f"{'-'*80}")

    all_pass = True
    for i, pname in enumerate(param_names):
        true_val = true_params[i]
        fitted_val = result[i]
        abs_err = fitted_val - true_val
        if abs(true_val) > 1e-10:
            rel_err = 100 * abs_err / true_val
        else:
            rel_err = 0.0 if abs(abs_err) < 1e-6 else float('inf')

        # Check pass/fail
        if abs(true_val) > 1e-6:
            passed = abs(rel_err) < tolerance
        else:
            passed = abs(abs_err) < 0.01

        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(f"{pname:<12} {true_val:>12.6f} {fitted_val:>12.6f} "
              f"{abs_err:>12.6f} {rel_err:>10.3f} {status_str}")

    return all_pass


def test_zero_noise_with_offset():
    """Test parameter recovery with offset fitting enabled."""
    win_size = (32, 32)
    center = (win_size[1] / 2 + 1, win_size[0] / 2 + 1)

    set_offset_fitting(enabled=True)

    results = []

    # Test 1: Zero displacement, moderate sigmas
    true_params = np.array([
        100.0, 100.0, 80.0,
        5.0, 5.0, 5.0,
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0], center[1],
    ])
    results.append(("Zero disp, small offset", run_test_case("Zero disp, small offset", true_params, win_size)))

    # Test 2: Small displacement
    true_params = np.array([
        100.0, 100.0, 80.0,
        5.0, 5.0, 5.0,
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0] + 2.5, center[1] + 1.5,
    ])
    results.append(("Small disp (2.5, 1.5)", run_test_case("Small disp (2.5, 1.5)", true_params, win_size)))

    # Test 3: Zero offset
    true_params = np.array([
        100.0, 100.0, 80.0,
        0.0, 0.0, 0.0,
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0] + 2.0, center[1] + 1.0,
    ])
    results.append(("Zero offset", run_test_case("Zero offset", true_params, win_size)))

    # Test 4: Anisotropic sigmas
    true_params = np.array([
        100.0, 100.0, 80.0,
        5.0, 5.0, 5.0,
        6.0, 3.0, 0.0,
        3.0, 1.5, 0.0,
        center[0], center[1],
        center[0] + 2.0, center[1] + 1.0,
    ])
    results.append(("Anisotropic sigmas", run_test_case("Anisotropic sigmas", true_params, win_size)))

    # Test 5: Small sigma_AB
    true_params = np.array([
        100.0, 100.0, 80.0,
        5.0, 5.0, 5.0,
        4.0, 4.0, 0.0,
        0.3, 0.3, 0.0,
        center[0], center[1],
        center[0] + 2.0, center[1] + 1.0,
    ])
    results.append(("Small sigma_AB (low turb)", run_test_case("Small sigma_AB", true_params, win_size, tolerance=1.0)))

    # Test 6: Large offset (with offset fitting enabled - should still work)
    true_params = np.array([
        100.0, 100.0, 80.0,
        10.0, 10.0, 10.0,       # Large offset
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0] + 2.0, center[1] + 1.0,
    ])
    results.append(("Large offset (enabled)", run_test_case("Large offset (enabled)", true_params, win_size)))

    # Test 7: Very large offset (stress test)
    true_params = np.array([
        100.0, 100.0, 80.0,
        50.0, 50.0, 50.0,       # Very large offset (50% of amplitude)
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0] + 2.0, center[1] + 1.0,
    ])
    results.append(("Very large offset (50%)", run_test_case("Very large offset (50%)", true_params, win_size)))

    return all(r[1] for r in results)


def test_zero_noise_without_offset():
    """Test parameter recovery with offset fitting disabled."""
    win_size = (32, 32)
    center = (win_size[1] / 2 + 1, win_size[0] / 2 + 1)

    set_offset_fitting(enabled=False)

    results = []

    # Test 1: Zero offset in data (should work perfectly)
    true_params = np.array([
        100.0, 100.0, 80.0,
        0.0, 0.0, 0.0,
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0] + 2.5, center[1] + 1.5,
    ])
    results.append(("Zero offset (disabled)", run_test_case("Zero offset (disabled)", true_params, win_size)))

    # Test 2: Non-zero offset in data (expect bias)
    print("\n" + "#"*80)
    print("# Note: With non-zero offset but offset fitting disabled,")
    print("# we expect biased sigma estimates (model mismatch)")
    print("#"*80)

    true_params = np.array([
        100.0, 100.0, 80.0,
        10.0, 10.0, 10.0,
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0] + 2.0, center[1] + 1.0,
    ])
    # Use larger tolerance - we expect errors here
    run_test_case("Non-zero offset (disabled) - expect bias", true_params, win_size, tolerance=20.0)

    # Re-enable offset fitting
    set_offset_fitting(enabled=True)

    return all(r[1] for r in results)


def test_perturbed_initial_guess():
    """Test recovery when initial guess is perturbed from true values."""
    win_size = (32, 32)
    center = (win_size[1] / 2 + 1, win_size[0] / 2 + 1)

    set_offset_fitting(enabled=True)

    print("\n" + "="*80)
    print("PERTURBED INITIAL GUESS TEST")
    print("="*80)
    print("\nTesting fitter robustness to imperfect initial guesses...")

    true_params = np.array([
        100.0, 100.0, 80.0,
        5.0, 5.0, 5.0,
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0] + 2.5, center[1] + 1.5,
    ])

    params_dict = {
        'amp_A': true_params[0], 'amp_B': true_params[1], 'amp_AB': true_params[2],
        'c_A': true_params[3], 'c_B': true_params[4], 'c_AB': true_params[5],
        'sx_A': true_params[6], 'sy_A': true_params[7], 'sxy_A': true_params[8],
        'sx_AB': true_params[9], 'sy_AB': true_params[10], 'sxy_AB': true_params[11],
        'x0_A': true_params[12], 'y0_A': true_params[13],
        'x0_AB': true_params[14], 'y0_AB': true_params[15],
    }
    AA, BB, AB = generate_stacked_planes(win_size, params_dict)

    perturbations = [0.1, 0.2, 0.5, 1.0]  # 10%, 20%, 50%, 100% perturbation

    print(f"\n{'Perturbation':>12} {'sx_A_err%':>12} {'sx_AB_err%':>12} {'R_uu_err%':>12} {'Status'}")
    print("-"*60)

    all_pass = True
    np.random.seed(42)

    for pct in perturbations:
        # Create perturbed initial guess
        perturbation = 1.0 + pct * (2 * np.random.rand(16) - 1)  # Random +/- pct
        initial_guess = true_params * perturbation

        # Keep positions close to true (within 2 pixels)
        initial_guess[12:16] = true_params[12:16] + 2 * (np.random.rand(4) - 0.5)

        # Ensure sigmas stay positive
        initial_guess[6:12] = np.maximum(initial_guess[6:12], 0.1)

        result, status = fit_single_window(AA, BB, AB, initial_guess, win_size)

        if status != 1:
            print(f"{pct*100:>10.0f}% FAILED")
            all_pass = False
            continue

        sx_A_err = 100 * (result[6] - true_params[6]) / true_params[6]
        sx_AB_err = 100 * (result[9] - true_params[9]) / true_params[9]
        R_true = true_params[9] ** 2
        R_fit = result[9] ** 2
        R_err = 100 * (R_fit - R_true) / R_true

        passed = abs(sx_AB_err) < 1.0
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(f"{pct*100:>10.0f}% {sx_A_err:>12.3f} {sx_AB_err:>12.3f} {R_err:>12.3f} {status_str}")

    return all_pass


def test_with_noise():
    """Test recovery with Gaussian noise added to correlation planes."""
    win_size = (32, 32)
    center = (win_size[1] / 2 + 1, win_size[0] / 2 + 1)

    set_offset_fitting(enabled=True)

    print("\n" + "="*80)
    print("NOISE ROBUSTNESS TEST")
    print("="*80)
    print("\nTesting fitter with Gaussian noise added to correlation planes...")

    true_params = np.array([
        100.0, 100.0, 80.0,
        5.0, 5.0, 5.0,
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0] + 2.5, center[1] + 1.5,
    ])

    noise_levels = [0.01, 0.05, 0.10, 0.20]  # 1%, 5%, 10%, 20% of amplitude

    print(f"\n{'Noise%':>10} {'sx_A_err%':>12} {'sx_AB_err%':>12} {'R_uu_err%':>12} {'Status'}")
    print("-"*60)

    all_pass = True
    np.random.seed(42)

    for noise_pct in noise_levels:
        params_dict = {
            'amp_A': true_params[0], 'amp_B': true_params[1], 'amp_AB': true_params[2],
            'c_A': true_params[3], 'c_B': true_params[4], 'c_AB': true_params[5],
            'sx_A': true_params[6], 'sy_A': true_params[7], 'sxy_A': true_params[8],
            'sx_AB': true_params[9], 'sy_AB': true_params[10], 'sxy_AB': true_params[11],
            'x0_A': true_params[12], 'y0_A': true_params[13],
            'x0_AB': true_params[14], 'y0_AB': true_params[15],
        }
        AA, BB, AB = generate_stacked_planes(win_size, params_dict, noise_std=noise_pct)

        # Use true params as initial guess
        initial_guess = true_params.copy()

        result, status = fit_single_window(AA, BB, AB, initial_guess, win_size)

        if status != 1:
            print(f"{noise_pct*100:>8.0f}% FAILED")
            all_pass = False
            continue

        sx_A_err = 100 * (result[6] - true_params[6]) / true_params[6]
        sx_AB_err = 100 * (result[9] - true_params[9]) / true_params[9]
        R_true = true_params[9] ** 2
        R_fit = result[9] ** 2
        R_err = 100 * (R_fit - R_true) / R_true

        # Allow more error with noise - 5% for 10% noise, etc
        tolerance = noise_pct * 50  # Rough scaling
        passed = abs(sx_AB_err) < max(tolerance, 5.0)
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(f"{noise_pct*100:>8.0f}% {sx_A_err:>12.3f} {sx_AB_err:>12.3f} {R_err:>12.3f} {status_str}")

    return all_pass


def test_reynolds_stress_recovery():
    """Verify sigma_AB recovery for Reynolds stress calculation."""
    win_size = (32, 32)
    center = (win_size[1] / 2 + 1, win_size[0] / 2 + 1)

    set_offset_fitting(enabled=True)

    print("\n" + "="*80)
    print("REYNOLDS STRESS RECOVERY TEST")
    print("="*80)
    print("\nIn the stacked Gaussian model:")
    print("  sigma_total^2 = sigma_A^2 + sigma_AB^2")
    print("  Reynolds stress R_uu ~ sigma_AB^2")
    print("\nTesting sigma_AB recovery for various values...")

    sigma_AB_values = [0.3, 0.5, 1.0, 2.0, 3.0]

    print(f"\n{'sig_AB_true':>12} {'sig_AB_fit':>12} {'Error%':>10} {'R_uu_true':>12} {'R_uu_fit':>12} {'R_err%':>10}")
    print("-"*80)

    all_pass = True
    for sig_AB in sigma_AB_values:
        true_params = np.array([
            100.0, 100.0, 80.0,
            5.0, 5.0, 5.0,
            4.0, 4.0, 0.0,
            sig_AB, sig_AB, 0.0,
            center[0], center[1],
            center[0] + 2.0, center[1] + 1.0,
        ])

        params_dict = {
            'amp_A': true_params[0], 'amp_B': true_params[1], 'amp_AB': true_params[2],
            'c_A': true_params[3], 'c_B': true_params[4], 'c_AB': true_params[5],
            'sx_A': true_params[6], 'sy_A': true_params[7], 'sxy_A': true_params[8],
            'sx_AB': true_params[9], 'sy_AB': true_params[10], 'sxy_AB': true_params[11],
            'x0_A': true_params[12], 'y0_A': true_params[13],
            'x0_AB': true_params[14], 'y0_AB': true_params[15],
        }
        AA, BB, AB = generate_stacked_planes(win_size, params_dict)

        result, status = fit_single_window(AA, BB, AB, true_params.copy(), win_size)

        if status != 1:
            print(f"{sig_AB:>12.3f} FAILED")
            all_pass = False
            continue

        sig_AB_fit = result[9]  # sx_AB
        sig_err = 100 * (sig_AB_fit - sig_AB) / sig_AB

        R_true = sig_AB ** 2
        R_fit = sig_AB_fit ** 2
        R_err = 100 * (R_fit - R_true) / R_true if R_true > 1e-10 else 0

        status_str = "PASS" if abs(sig_err) < 1.0 else "FAIL"
        print(f"{sig_AB:>12.3f} {sig_AB_fit:>12.6f} {sig_err:>10.3f} "
              f"{R_true:>12.4f} {R_fit:>12.6f} {R_err:>10.3f} {status_str}")

        if abs(sig_err) > 1.0:
            all_pass = False

    return all_pass


def main():
    """Run all Gaussian fitting tests."""
    print("="*80)
    print("C LIBRARY GAUSSIAN FITTING VERIFICATION")
    print("Testing marquadt_gaussian.c with zero-noise synthetic data")
    print("="*80)

    try:
        lib = _load_marquadt_lib()
        print(f"\nLibrary loaded successfully")
    except Exception as e:
        print(f"\nFAILED to load library: {e}")
        return 1

    results = {}

    print("\n" + "="*80)
    print("PART 1: Zero noise with offset fitting ENABLED")
    print("="*80)
    results['with_offset'] = test_zero_noise_with_offset()

    print("\n" + "="*80)
    print("PART 2: Zero noise with offset fitting DISABLED")
    print("="*80)
    results['without_offset'] = test_zero_noise_without_offset()

    print("\n" + "="*80)
    print("PART 3: Perturbed initial guess")
    print("="*80)
    results['perturbed'] = test_perturbed_initial_guess()

    print("\n" + "="*80)
    print("PART 4: Noise robustness")
    print("="*80)
    results['noise'] = test_with_noise()

    print("\n" + "="*80)
    print("PART 5: Reynolds stress recovery")
    print("="*80)
    results['reynolds'] = test_reynolds_stress_recovery()

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
