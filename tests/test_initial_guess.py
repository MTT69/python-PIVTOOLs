#!/usr/bin/env python3
"""
Test script to analyze initial guess quality for stacked Gaussian fitting.

Generates synthetic correlation planes with known parameters, runs them through
the initial guess generators, and compares to ground truth.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_2d_gaussian(X, Y, amp, cx, cy, sx, sy, sxy, offset):
    """
    Generate a 2D Gaussian with full covariance matrix.

    Covariance matrix: [[sx, sxy], [sxy, sy]]
    """
    # Build inverse covariance matrix
    det = sx * sy - sxy * sxy
    if det <= 0:
        det = 1e-6  # Prevent singular matrix

    inv_sx = sy / det
    inv_sy = sx / det
    inv_sxy = -sxy / det

    dx = X - cx
    dy = Y - cy

    exponent = -0.5 * (inv_sx * dx**2 + 2 * inv_sxy * dx * dy + inv_sy * dy**2)
    return amp * np.exp(exponent) + offset


def generate_stacked_gaussians(win_size, params):
    """
    Generate stacked AA, BB, AB correlation planes.

    Parameters (16 total):
    [0-2]: amp_A, amp_B, amp_AB
    [3-5]: c_A, c_B, c_AB (offsets)
    [6-8]: sx_A, sy_A, sxy_A
    [9-11]: sx_AB, sy_AB, sxy_AB
    [12-13]: x0_A, y0_A
    [14-15]: x0_AB, y0_AB
    """
    h, w = win_size
    Y, X = np.meshgrid(np.arange(1, h+1), np.arange(1, w+1), indexing='ij')

    amp_A, amp_B, amp_AB = params[0:3]
    c_A, c_B, c_AB = params[3:6]
    sx_A, sy_A, sxy_A = params[6:9]
    sx_AB, sy_AB, sxy_AB = params[9:12]
    x0_A, y0_A = params[12:14]
    x0_AB, y0_AB = params[14:16]

    # AA and BB share the A covariance, centered at x0_A, y0_A
    AA = generate_2d_gaussian(X, Y, amp_A, x0_A, y0_A, sx_A, sy_A, sxy_A, c_A)
    BB = generate_2d_gaussian(X, Y, amp_B, x0_A, y0_A, sx_A, sy_A, sxy_A, c_B)

    # AB uses combined covariance (sx_A + sx_AB), centered at x0_AB, y0_AB
    sum_sx = sx_A + sx_AB
    sum_sy = sy_A + sy_AB
    sum_sxy = sxy_A + sxy_AB
    AB = generate_2d_gaussian(X, Y, amp_AB, x0_AB, y0_AB, sum_sx, sum_sy, sum_sxy, c_AB)

    return AA, BB, AB


def estimate_sigma_moment(profile, peak_idx, min_sigma=0.5):
    """
    Estimate standard deviation using moment-based method.

    Uses weighted second moment with edge-based background subtraction.
    No thresholding - uses full profile for accurate variance estimation.
    """
    n = len(profile)

    # Background from edges (mean of first/last few points)
    n_edge = max(3, n // 8)
    bg = 0.5 * (np.mean(profile[:n_edge]) + np.mean(profile[-n_edge:]))

    # Subtract background
    profile_shifted = profile - bg

    # Use only positive values (after background subtraction)
    weights = np.maximum(profile_shifted, 0)

    total_w = weights.sum()
    if total_w < 1e-10:
        return min_sigma

    # Weighted moment calculation
    x_coords = np.arange(n, dtype=np.float64)

    # Centroid
    centroid = np.sum(x_coords * weights) / total_w

    # Second moment = variance
    variance = np.sum((x_coords - centroid)**2 * weights) / total_w

    return max(np.sqrt(variance), min_sigma)


def generate_initial_guess(AA, BB, AB, win_size):
    """
    Replicate the FIXED initial guess generation from gaussian_fitting.py.

    Key fixes:
    1. Amplitudes = peak_value - offset (not raw peak value)
    2. Fixed reasonable default variances (not fragile moment estimation)
    """
    h, w = win_size
    center_x = w / 2.0 + 1  # 1-based indexing
    center_y = h / 2.0 + 1
    central_idx_y = h // 2
    central_idx_x = w // 2

    # Raw peak values
    peak_A = AA[central_idx_y, central_idx_x]
    peak_B = BB[central_idx_y, central_idx_x]

    # AB peak location
    peak_flat_idx = np.argmax(AB)
    peak_y, peak_x = np.unravel_index(peak_flat_idx, AB.shape)
    peak_AB = AB[peak_y, peak_x]

    # Offsets from 5th percentile
    c_A = np.percentile(AA, 5)
    c_B = np.percentile(BB, 5)
    c_AB = np.percentile(AB, 5)

    # CRITICAL: Amplitudes = peak - offset (not raw peak!)
    # Model: f(x) = amp * exp(...) + c
    # At peak: peak_value = amp + c, so amp = peak_value - c
    amp_A = max(peak_A - c_A, 1e-6)
    amp_B = max(peak_B - c_B, 1e-6)
    amp_AB = max(peak_AB - c_AB, 1e-6)

    # Use fixed reasonable default VARIANCES based on window size
    # For LM fitting, reasonable defaults work better than fragile estimates
    scale_factor = h / 32.0

    # Default variance for autocorrelation (typical PIV: σ² ≈ 3 for 32x32)
    variance_A_x = 3.0 * scale_factor
    variance_A_y = 3.0 * scale_factor
    sigma_A_xy = 0.0

    # Default variance for displacement uncertainty (σ² ≈ 1.5 for 32x32)
    variance_AB_x = 1.5 * scale_factor
    variance_AB_y = 1.5 * scale_factor
    sigma_AB_xy = 0.0

    # Build guess array
    guess = np.array([
        amp_A, amp_B, amp_AB,
        c_A, c_B, c_AB,
        variance_A_x, variance_A_y, sigma_A_xy,
        variance_AB_x, variance_AB_y, sigma_AB_xy,
        center_x, center_y,
        peak_x + 1, peak_y + 1  # 1-based
    ])

    return guess


def analyze_guess_quality(true_params, guess, param_names):
    """Analyze and print quality of initial guess."""
    print("\n" + "="*80)
    print("INITIAL GUESS QUALITY ANALYSIS")
    print("="*80)
    print(f"{'Parameter':<15} {'True':>12} {'Guess':>12} {'Error':>12} {'Rel.Err%':>10}")
    print("-"*80)

    errors = []
    for i, name in enumerate(param_names):
        true_val = true_params[i]
        guess_val = guess[i]
        error = guess_val - true_val
        if abs(true_val) > 1e-6:
            rel_err = 100 * error / true_val
        else:
            rel_err = 0 if abs(error) < 1e-6 else float('inf')

        errors.append((name, true_val, guess_val, error, rel_err))
        print(f"{name:<15} {true_val:>12.4f} {guess_val:>12.4f} {error:>12.4f} {rel_err:>10.1f}")

    return errors


def run_test_case(name, true_params, win_size=(32, 32), add_noise=0.0):
    """Run a single test case."""
    print(f"\n{'#'*80}")
    print(f"# TEST CASE: {name}")
    print(f"{'#'*80}")

    # Generate synthetic data
    AA, BB, AB = generate_stacked_gaussians(win_size, true_params)

    # Add noise if requested
    if add_noise > 0:
        noise_scale = add_noise * np.max([AA.max(), BB.max(), AB.max()])
        AA = AA + np.random.normal(0, noise_scale, AA.shape)
        BB = BB + np.random.normal(0, noise_scale, BB.shape)
        AB = AB + np.random.normal(0, noise_scale, AB.shape)

    # Generate initial guess
    guess = generate_initial_guess(AA, BB, AB, win_size)

    # Parameter names
    param_names = [
        'amp_A', 'amp_B', 'amp_AB',
        'c_A', 'c_B', 'c_AB',
        'sx_A', 'sy_A', 'sxy_A',
        'sx_AB', 'sy_AB', 'sxy_AB',
        'x0_A', 'y0_A',
        'x0_AB', 'y0_AB'
    ]

    errors = analyze_guess_quality(true_params, guess, param_names)

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot correlation planes
    axes[0, 0].imshow(AA, origin='lower')
    axes[0, 0].set_title('AA (Auto-correlation A)')
    axes[0, 0].plot(true_params[12]-1, true_params[13]-1, 'r+', markersize=15, label='True center')
    axes[0, 0].plot(guess[12]-1, guess[13]-1, 'gx', markersize=15, label='Guess center')

    axes[0, 1].imshow(BB, origin='lower')
    axes[0, 1].set_title('BB (Auto-correlation B)')

    axes[0, 2].imshow(AB, origin='lower')
    axes[0, 2].set_title('AB (Cross-correlation)')
    axes[0, 2].plot(true_params[14]-1, true_params[15]-1, 'r+', markersize=15, label='True peak')
    axes[0, 2].plot(guess[14]-1, guess[15]-1, 'gx', markersize=15, label='Guess peak')
    axes[0, 2].legend()

    # Plot profiles
    center_y = win_size[0] // 2
    center_x = win_size[1] // 2

    axes[1, 0].plot(AA[center_y, :], 'b-', label='AA x-profile')
    axes[1, 0].axhline(AA[center_y, center_x]/2, color='r', linestyle='--', label='Half-max')
    axes[1, 0].set_title('AA X-Profile at Center')
    axes[1, 0].legend()

    peak_y = int(guess[15] - 1)
    peak_x = int(guess[14] - 1)
    axes[1, 1].plot(AB[peak_y, :], 'b-', label='AB x-profile')
    axes[1, 1].axhline(AB[peak_y, peak_x]/2, color='r', linestyle='--', label='Half-max')
    axes[1, 1].set_title('AB X-Profile at Peak')
    axes[1, 1].legend()

    # Error bar chart
    critical_params = ['sx_A', 'sy_A', 'sx_AB', 'sy_AB', 'x0_AB', 'y0_AB']
    critical_idx = [6, 7, 9, 10, 14, 15]
    rel_errors = [errors[i][4] for i in critical_idx]

    colors = ['green' if abs(e) < 20 else 'orange' if abs(e) < 50 else 'red' for e in rel_errors]
    axes[1, 2].bar(critical_params, rel_errors, color=colors)
    axes[1, 2].set_ylabel('Relative Error (%)')
    axes[1, 2].set_title('Critical Parameter Errors')
    axes[1, 2].axhline(0, color='k', linestyle='-')
    axes[1, 2].axhline(20, color='g', linestyle='--', alpha=0.5)
    axes[1, 2].axhline(-20, color='g', linestyle='--', alpha=0.5)

    plt.suptitle(f'Test Case: {name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'/tmp/initial_guess_test_{name.replace(" ", "_")}.png', dpi=150)
    plt.close()

    return errors, guess


def main():
    """Run comprehensive initial guess tests."""

    win_size = (32, 32)
    center = (win_size[1] / 2 + 1, win_size[0] / 2 + 1)  # 1-based

    print("\n" + "="*80)
    print("STACKED GAUSSIAN INITIAL GUESS ANALYSIS")
    print("="*80)
    print(f"Window size: {win_size}")
    print(f"Window center (1-based): {center}")

    # Test Case 1: Ideal case - zero displacement, moderate sigmas
    params_ideal = np.array([
        100.0, 100.0, 80.0,    # amplitudes
        5.0, 5.0, 5.0,          # offsets
        4.0, 4.0, 0.0,          # sx_A, sy_A, sxy_A
        2.0, 2.0, 0.0,          # sx_AB, sy_AB, sxy_AB
        center[0], center[1],   # x0_A, y0_A (at center)
        center[0], center[1],   # x0_AB, y0_AB (zero displacement)
    ])
    run_test_case("Ideal Zero Displacement", params_ideal, win_size)

    # Test Case 2: Small displacement (typical PIV)
    params_small_disp = np.array([
        100.0, 100.0, 80.0,
        5.0, 5.0, 5.0,
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0] + 2.5, center[1] + 1.5,  # 2.5 px X, 1.5 px Y displacement
    ])
    run_test_case("Small Displacement 2.5px", params_small_disp, win_size)

    # Test Case 3: Large displacement (near 1/4 rule limit)
    params_large_disp = np.array([
        100.0, 100.0, 80.0,
        5.0, 5.0, 5.0,
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0] + 7.0, center[1] + 5.0,  # Large displacement
    ])
    run_test_case("Large Displacement 7px", params_large_disp, win_size)

    # Test Case 4: Anisotropic sigmas (elliptical Gaussian)
    params_aniso = np.array([
        100.0, 100.0, 80.0,
        5.0, 5.0, 5.0,
        6.0, 3.0, 0.0,          # Elongated in X
        3.0, 1.5, 0.0,
        center[0], center[1],
        center[0] + 2.0, center[1] + 2.0,
    ])
    run_test_case("Anisotropic Sigmas", params_aniso, win_size)

    # Test Case 5: Very narrow Gaussian (small sigma)
    params_narrow = np.array([
        100.0, 100.0, 80.0,
        5.0, 5.0, 5.0,
        1.5, 1.5, 0.0,          # Narrow
        0.8, 0.8, 0.0,          # Very narrow AB
        center[0], center[1],
        center[0] + 2.0, center[1] + 1.0,
    ])
    run_test_case("Narrow Gaussian", params_narrow, win_size)

    # Test Case 6: Wide Gaussian (large sigma)
    params_wide = np.array([
        100.0, 100.0, 80.0,
        5.0, 5.0, 5.0,
        8.0, 8.0, 0.0,          # Wide
        4.0, 4.0, 0.0,
        center[0], center[1],
        center[0] + 2.0, center[1] + 1.0,
    ])
    run_test_case("Wide Gaussian", params_wide, win_size)

    # Test Case 7: Low SNR (with noise)
    params_noisy = np.array([
        100.0, 100.0, 80.0,
        5.0, 5.0, 5.0,
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0] + 2.0, center[1] + 1.0,
    ])
    run_test_case("Low SNR 10% Noise", params_noisy, win_size, add_noise=0.10)

    # Test Case 8: Very low SNR
    run_test_case("Very Low SNR 25% Noise", params_noisy, win_size, add_noise=0.25)

    # Test Case 9: Negative offset (post background subtraction)
    params_neg_offset = np.array([
        100.0, 100.0, 80.0,
        -10.0, -10.0, -15.0,    # Negative offsets
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0] + 2.0, center[1] + 1.0,
    ])
    run_test_case("Negative Offsets", params_neg_offset, win_size)

    # Test Case 10: Weak AB peak (low correlation)
    params_weak_ab = np.array([
        100.0, 100.0, 30.0,     # Weak AB
        5.0, 5.0, 5.0,
        4.0, 4.0, 0.0,
        2.0, 2.0, 0.0,
        center[0], center[1],
        center[0] + 2.0, center[1] + 1.0,
    ])
    run_test_case("Weak AB Peak", params_weak_ab, win_size)

    print("\n" + "="*80)
    print("TEST COMPLETE - Check /tmp/initial_guess_test_*.png for visualizations")
    print("="*80)


if __name__ == "__main__":
    np.random.seed(42)
    main()
