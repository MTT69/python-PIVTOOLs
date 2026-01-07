"""
Test Gaussian fitter accuracy vs window size.

Does the fitter introduce window-size-dependent bias on ideal Gaussians?
If yes → fitter is the source of bias
If no → bias comes from correlation plane generation (FFT, particle truncation, etc.)

Usage:
    python test_fitter_window_size.py
"""
import sys
import numpy as np
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_initial_guess import generate_2d_gaussian
from pivtools_cli.piv.piv_backend.gaussian_fitting import (
    _load_marquadt_lib,
    set_offset_fitting,
)


def generate_stacked_planes(shape, params_dict, noise_std=0.0):
    """Generate AA, BB, AB correlation planes from params dict."""
    h, w = shape
    Y, X = np.meshgrid(np.arange(1, h+1), np.arange(1, w+1), indexing='ij')

    amp_A, amp_B, amp_AB = params_dict['amp_A'], params_dict['amp_B'], params_dict['amp_AB']
    c_A, c_B, c_AB = params_dict['c_A'], params_dict['c_B'], params_dict['c_AB']
    sx_A, sy_A, sxy_A = params_dict['sx_A'], params_dict['sy_A'], params_dict['sxy_A']
    sx_AB, sy_AB, sxy_AB = params_dict['sx_AB'], params_dict['sy_AB'], params_dict['sxy_AB']
    x0_A, y0_A = params_dict['x0_A'], params_dict['y0_A']
    x0_AB, y0_AB = params_dict['x0_AB'], params_dict['y0_AB']

    AA = generate_2d_gaussian(X, Y, amp_A, x0_A, y0_A, sx_A, sy_A, sxy_A, c_A)
    BB = generate_2d_gaussian(X, Y, amp_B, x0_A, y0_A, sx_A, sy_A, sxy_A, c_B)

    sum_sx = sx_A + sx_AB
    sum_sy = sy_A + sy_AB
    sum_sxy = sxy_A + sxy_AB
    AB = generate_2d_gaussian(X, Y, amp_AB, x0_AB, y0_AB, sum_sx, sum_sy, sum_sxy, c_AB)

    if noise_std > 0:
        noise_scale = amp_A * noise_std
        AA += np.random.randn(*shape) * noise_scale
        BB += np.random.randn(*shape) * noise_scale
        AB += np.random.randn(*shape) * noise_scale

    return AA, BB, AB


def fit_single_window(AA, BB, AB, initial_guess, win_size):
    """Fit a single window using the C library."""
    import ctypes
    lib = _load_marquadt_lib()

    h, w = win_size
    n_per_window = h * w

    Y, X = np.meshgrid(np.arange(1, h+1), np.arange(1, w+1), indexing='ij')
    X1 = Y.ravel(order='C').astype(np.float64)
    X2 = X.ravel(order='C').astype(np.float64)

    y_all = np.concatenate([AA.ravel(), BB.ravel(), AB.ravel()]).astype(np.float64)

    result = np.zeros(16, dtype=np.float64)
    status = np.zeros(1, dtype=np.int32)

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


def test_window_size_effect():
    """Test sigma recovery at different window sizes."""
    print("\n" + "="*80)
    print("WINDOW SIZE EFFECT ON GAUSSIAN FITTING")
    print("="*80)
    print("\nTesting fitter accuracy on ideal Gaussians at different window sizes...")
    print("If errors vary with window size → fitter introduces bias")
    print("If errors are constant (~0) → bias comes from elsewhere (FFT, particles)")

    window_sizes = [16, 32, 64, 128]

    # Test parameters matching RS test conditions
    # sig_A ~ 1.4 (autocorrelation), sig_AB ~ 1.4 (RS variance component)
    sig_A_true = 1.4
    sig_AB_true = 1.4  # This gives UU = sig_AB^2 = 1.96 px^2

    set_offset_fitting(enabled=False)

    print(f"\nTrue parameters:")
    print(f"  sig_A = {sig_A_true:.3f} px (autocorrelation width)")
    print(f"  sig_AB = {sig_AB_true:.3f} px (RS component)")
    print(f"  R_uu = sig_AB^2 = {sig_AB_true**2:.3f} px^2")

    print(f"\n{'Window':<10} {'sig_A_fit':<12} {'sig_A_err%':<12} {'sig_AB_fit':<12} {'sig_AB_err%':<12} {'R_uu_err%':<12}")
    print("-"*72)

    results = []
    for win in window_sizes:
        win_size = (win, win)
        center = (win / 2 + 1, win / 2 + 1)

        # Ensure Gaussian is well-sampled (peak not truncated)
        # For sig_A=1.4, 99% of Gaussian is within ~3.5 sigmas = 4.9 pixels from center
        # So even 16x16 should work (8 pixels from center)

        true_params = np.array([
            100.0, 100.0, 80.0,      # Amplitudes
            0.0, 0.0, 0.0,           # Offsets (disabled)
            sig_A_true, sig_A_true, 0.0,   # sig_A (autocorrelation)
            sig_AB_true, sig_AB_true, 0.0, # sig_AB (RS component)
            center[0], center[1],    # A position (center)
            center[0], center[1],    # AB position (zero displacement)
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
            print(f"{win}x{win:<7} FAILED")
            continue

        sig_A_fit = result[6]  # sx_A
        sig_AB_fit = result[9]  # sx_AB

        sig_A_err = 100 * (sig_A_fit - sig_A_true) / sig_A_true
        sig_AB_err = 100 * (sig_AB_fit - sig_AB_true) / sig_AB_true

        R_true = sig_AB_true ** 2
        R_fit = sig_AB_fit ** 2
        R_err = 100 * (R_fit - R_true) / R_true

        print(f"{win}x{win:<7} {sig_A_fit:<12.6f} {sig_A_err:<12.4f} "
              f"{sig_AB_fit:<12.6f} {sig_AB_err:<12.4f} {R_err:<12.4f}")

        results.append({
            'window': win,
            'sig_A_fit': sig_A_fit,
            'sig_A_err': sig_A_err,
            'sig_AB_fit': sig_AB_fit,
            'sig_AB_err': sig_AB_err,
            'R_err': R_err,
        })

    return results


def test_window_size_with_noise():
    """Test sigma recovery at different window sizes with realistic noise."""
    print("\n" + "="*80)
    print("WINDOW SIZE EFFECT WITH NOISE (5% of amplitude)")
    print("="*80)

    window_sizes = [16, 32, 64, 128]
    sig_A_true = 1.4
    sig_AB_true = 1.4
    noise_std = 0.05  # 5% noise

    set_offset_fitting(enabled=False)
    np.random.seed(42)

    print(f"\n{'Window':<10} {'sig_A_err%':<12} {'sig_AB_err%':<12} {'R_uu_err%':<12}")
    print("-"*50)

    for win in window_sizes:
        win_size = (win, win)
        center = (win / 2 + 1, win / 2 + 1)

        true_params = np.array([
            100.0, 100.0, 80.0,
            0.0, 0.0, 0.0,
            sig_A_true, sig_A_true, 0.0,
            sig_AB_true, sig_AB_true, 0.0,
            center[0], center[1],
            center[0], center[1],
        ])

        params_dict = {
            'amp_A': true_params[0], 'amp_B': true_params[1], 'amp_AB': true_params[2],
            'c_A': true_params[3], 'c_B': true_params[4], 'c_AB': true_params[5],
            'sx_A': true_params[6], 'sy_A': true_params[7], 'sxy_A': true_params[8],
            'sx_AB': true_params[9], 'sy_AB': true_params[10], 'sxy_AB': true_params[11],
            'x0_A': true_params[12], 'y0_A': true_params[13],
            'x0_AB': true_params[14], 'y0_AB': true_params[15],
        }

        AA, BB, AB = generate_stacked_planes(win_size, params_dict, noise_std=noise_std)

        result, status = fit_single_window(AA, BB, AB, true_params.copy(), win_size)

        if status != 1:
            print(f"{win}x{win:<7} FAILED")
            continue

        sig_A_fit = result[6]
        sig_AB_fit = result[9]

        sig_A_err = 100 * (sig_A_fit - sig_A_true) / sig_A_true
        sig_AB_err = 100 * (sig_AB_fit - sig_AB_true) / sig_AB_true
        R_err = 100 * (sig_AB_fit**2 - sig_AB_true**2) / (sig_AB_true**2)

        print(f"{win}x{win:<7} {sig_A_err:<12.4f} {sig_AB_err:<12.4f} {R_err:<12.4f}")


def test_fit_region_vs_window():
    """
    Test: What if we generate a large Gaussian but only fit a smaller region?
    This simulates the real PIV case where correlation planes are computed
    on small windows but contain information from larger particle images.
    """
    print("\n" + "="*80)
    print("FIT REGION SIZE TEST")
    print("="*80)
    print("\nGenerating Gaussian on 64x64, fitting on smaller extracted regions...")
    print("This simulates: large correlation plane, small fit region")

    # Generate on large grid
    full_size = (64, 64)
    center_full = (33, 33)  # 1-based center

    sig_A_true = 1.4
    sig_AB_true = 1.4

    set_offset_fitting(enabled=False)

    true_params = np.array([
        100.0, 100.0, 80.0,
        0.0, 0.0, 0.0,
        sig_A_true, sig_A_true, 0.0,
        sig_AB_true, sig_AB_true, 0.0,
        center_full[0], center_full[1],
        center_full[0], center_full[1],
    ])

    params_dict = {
        'amp_A': true_params[0], 'amp_B': true_params[1], 'amp_AB': true_params[2],
        'c_A': true_params[3], 'c_B': true_params[4], 'c_AB': true_params[5],
        'sx_A': true_params[6], 'sy_A': true_params[7], 'sxy_A': true_params[8],
        'sx_AB': true_params[9], 'sy_AB': true_params[10], 'sxy_AB': true_params[11],
        'x0_A': true_params[12], 'y0_A': true_params[13],
        'x0_AB': true_params[14], 'y0_AB': true_params[15],
    }

    AA_full, BB_full, AB_full = generate_stacked_planes(full_size, params_dict)

    # Extract and fit on smaller regions
    fit_regions = [64, 32, 16]

    print(f"\n{'Fit Region':<12} {'sig_A_err%':<12} {'sig_AB_err%':<12} {'R_uu_err%':<12}")
    print("-"*50)

    for fit_size in fit_regions:
        # Extract center region
        offset = (64 - fit_size) // 2
        AA = AA_full[offset:offset+fit_size, offset:offset+fit_size]
        BB = BB_full[offset:offset+fit_size, offset:offset+fit_size]
        AB = AB_full[offset:offset+fit_size, offset:offset+fit_size]

        # Adjust center for extracted region (1-based)
        new_center = (fit_size / 2 + 1, fit_size / 2 + 1)

        initial_guess = true_params.copy()
        initial_guess[12] = new_center[0]  # x0_A
        initial_guess[13] = new_center[1]  # y0_A
        initial_guess[14] = new_center[0]  # x0_AB
        initial_guess[15] = new_center[1]  # y0_AB

        result, status = fit_single_window(AA, BB, AB, initial_guess, (fit_size, fit_size))

        if status != 1:
            print(f"{fit_size}x{fit_size:<8} FAILED")
            continue

        sig_A_fit = result[6]
        sig_AB_fit = result[9]

        sig_A_err = 100 * (sig_A_fit - sig_A_true) / sig_A_true
        sig_AB_err = 100 * (sig_AB_fit - sig_AB_true) / sig_AB_true
        R_err = 100 * (sig_AB_fit**2 - sig_AB_true**2) / (sig_AB_true**2)

        print(f"{fit_size}x{fit_size:<8} {sig_A_err:<12.4f} {sig_AB_err:<12.4f} {R_err:<12.4f}")


def main():
    """Run all window size tests."""
    print("="*80)
    print("GAUSSIAN FITTER WINDOW SIZE DEPENDENCY TEST")
    print("="*80)

    try:
        lib = _load_marquadt_lib()
        print(f"\nLibrary loaded successfully")
    except Exception as e:
        print(f"\nFAILED to load library: {e}")
        return 1

    # Test 1: Basic window size effect (zero noise)
    results = test_window_size_effect()

    # Test 2: With noise
    test_window_size_with_noise()

    # Test 3: Fit region extraction
    test_fit_region_vs_window()

    # Summary
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if results:
        max_err = max(abs(r['sig_AB_err']) for r in results)
        if max_err < 0.01:
            print(f"\nFitter shows NO window-size bias (max error: {max_err:.6f}%)")
            print("→ The bias in ensemble PIV comes from correlation plane generation")
            print("  (particle truncation, FFT edge effects, etc.), NOT from Gaussian fitting")
        else:
            print(f"\nFitter shows window-size bias (max error: {max_err:.4f}%)")
            print("→ Part of the ensemble PIV bias may come from the Gaussian fitter")

    return 0


if __name__ == "__main__":
    exit(main())
