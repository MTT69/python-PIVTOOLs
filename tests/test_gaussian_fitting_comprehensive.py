"""
Comprehensive Gaussian Fitting Test: When is +C worth it?

Tests across multiple noise levels and offset magnitudes to determine
when the +C offset term helps vs hurts parameter recovery.

Key questions:
1. At what SNR does +C start absorbing signal instead of just noise?
2. For small offsets, is +C still beneficial?
3. What's the displacement accuracy impact (the most important parameter)?
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import ctypes

# Reuse functions from the main test
from PyPIVTools.tests.test_gaussian_fitting import (
    generate_2d_gaussian,
    generate_stacked_planes,
    _load_marquadt_lib,
    set_offset_fitting,
)


def fit_correlation_planes(
    AA: np.ndarray,
    BB: np.ndarray,
    AB: np.ndarray,
    initial_guess: np.ndarray,
    use_offset: bool = True
) -> Tuple[np.ndarray, int]:
    """Fit stacked correlation planes using the C library."""
    set_offset_fitting(use_offset)
    lib = _load_marquadt_lib()

    h, w = AA.shape
    n_per_window = h * w

    y_coords, x_coords = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
    X1 = y_coords.flatten().astype(np.float64)
    X2 = x_coords.flatten().astype(np.float64)

    y_all = np.concatenate([AA.flatten(), BB.flatten(), AB.flatten()]).astype(np.float64)

    out_params = np.zeros(16, dtype=np.float64)
    out_status = np.zeros(1, dtype=np.int32)

    lib.fit_stacked_gaussian_batch_export(
        ctypes.c_size_t(1),
        ctypes.c_size_t(n_per_window),
        X2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        initial_guess.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_status.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )

    set_offset_fitting(True)
    return out_params, out_status[0]


def run_monte_carlo(
    true_params: Dict[str, float],
    noise_std: float,
    use_offset: bool,
    n_trials: int = 50,
    shape: Tuple[int, int] = (65, 65)
) -> Dict:
    """
    Run Monte Carlo trials to get statistics on parameter recovery.

    Returns mean and std of errors for key parameters.
    """
    param_names = [
        'amp_A', 'amp_B', 'amp_AB',
        'c_A', 'c_B', 'c_AB',
        'sx_A', 'sy_A', 'sxy_A',
        'sx_AB', 'sy_AB', 'sxy_AB',
        'x0_A', 'y0_A', 'x0_AB', 'y0_AB'
    ]

    errors = {name: [] for name in param_names}
    rss_values = []
    successes = 0

    for trial in range(n_trials):
        # Generate synthetic data with noise
        np.random.seed(trial)  # Reproducible
        AA, BB, AB = generate_stacked_planes(shape, true_params, noise_std)

        # Create initial guess (perturbed from true)
        initial_guess = np.array([true_params[p] for p in param_names], dtype=np.float64)
        initial_guess[:6] *= 1.1  # Amplitudes/offsets +10%
        initial_guess[6:12] *= 1.2  # Sigmas +20%

        # Fit
        fitted, status = fit_correlation_planes(AA, BB, AB, initial_guess, use_offset)

        if status == 1:
            successes += 1

            # Compute errors
            for i, name in enumerate(param_names):
                true_val = true_params[name]
                fitted_val = fitted[i]
                if abs(true_val) > 1e-10:
                    err = fitted_val - true_val
                else:
                    err = fitted_val
                errors[name].append(err)

            # Compute RSS (only for AB plane since that's what matters for displacement)
            AB_fit = generate_2d_gaussian(
                shape, fitted[2],
                fitted[6] + fitted[9], fitted[7] + fitted[10], fitted[8] + fitted[11],
                fitted[14], fitted[15], fitted[5] if use_offset else 0.0
            )
            rss = np.sum((AB - AB_fit) ** 2)
            rss_values.append(rss)

    # Compute statistics
    results = {
        'success_rate': successes / n_trials,
        'rss_mean': np.mean(rss_values) if rss_values else np.inf,
        'rss_std': np.std(rss_values) if rss_values else np.inf,
    }

    for name in param_names:
        if errors[name]:
            results[f'{name}_bias'] = np.mean(errors[name])
            results[f'{name}_std'] = np.std(errors[name])
        else:
            results[f'{name}_bias'] = np.inf
            results[f'{name}_std'] = np.inf

    return results


def main():
    print("=" * 70)
    print("COMPREHENSIVE +C OFFSET TERM ANALYSIS")
    print("=" * 70)

    shape = (65, 65)
    center = (shape[0] // 2 + 1, shape[1] // 2 + 1)

    # True displacement (what we really care about)
    true_dx = 2.5
    true_dy = 1.3

    # Base parameters
    base_params = {
        'amp_A': 1000.0,
        'amp_B': 950.0,
        'amp_AB': 900.0,
        'c_A': 0.0,
        'c_B': 0.0,
        'c_AB': 0.0,
        'sx_A': 3.0,
        'sy_A': 3.0,
        'sxy_A': 0.0,
        'sx_AB': 1.5,
        'sy_AB': 1.5,
        'sxy_AB': 0.0,
        'x0_A': float(center[0]),
        'y0_A': float(center[1]),
        'x0_AB': float(center[0]) + true_dx,
        'y0_AB': float(center[1]) + true_dy,
    }

    # Test configurations
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]  # SNR: inf, 100, 50, 20, 10, 5
    offset_levels = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]  # As % of amplitude

    n_trials = 30  # Monte Carlo trials per configuration

    # =========================================================
    # TEST 1: Effect of noise level (no offset in data)
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 1: NOISE SENSITIVITY (no offset in data)")
    print("=" * 70)
    print("\nQuestion: Does +C hurt displacement accuracy when there's no offset?")
    print(f"\nTrue displacement: dx={true_dx}, dy={true_dy}")
    print(f"Monte Carlo trials per config: {n_trials}")

    print(f"\n{'Noise':>8} {'SNR':>6} | {'With +C dx bias':>15} {'dx std':>10} | {'No +C dx bias':>15} {'dx std':>10} | {'Winner':>10}")
    print("-" * 90)

    noise_results = []
    for noise in noise_levels:
        params = base_params.copy()
        snr = 1000 / (noise * 1000) if noise > 0 else float('inf')

        # Run with +C
        res_with = run_monte_carlo(params, noise, use_offset=True, n_trials=n_trials, shape=shape)

        # Run without +C
        res_no = run_monte_carlo(params, noise, use_offset=False, n_trials=n_trials, shape=shape)

        # Displacement is x0_AB - center[0] for dx, y0_AB - center[1] for dy
        # Error in x0_AB directly translates to error in displacement
        dx_bias_with = res_with['x0_AB_bias']
        dx_std_with = res_with['x0_AB_std']
        dx_bias_no = res_no['x0_AB_bias']
        dx_std_no = res_no['x0_AB_std']

        # Which is better? Lower total error (bias^2 + std^2)
        rmse_with = np.sqrt(dx_bias_with**2 + dx_std_with**2)
        rmse_no = np.sqrt(dx_bias_no**2 + dx_std_no**2)
        winner = "With +C" if rmse_with < rmse_no else "No +C" if rmse_no < rmse_with else "Tie"

        noise_results.append({
            'noise': noise, 'snr': snr,
            'with_c': res_with, 'no_c': res_no,
            'winner': winner
        })

        snr_str = f"{snr:.0f}" if snr < 1000 else "∞"
        print(f"{noise:>8.2f} {snr_str:>6} | {dx_bias_with:>+15.4f} {dx_std_with:>10.4f} | {dx_bias_no:>+15.4f} {dx_std_no:>10.4f} | {winner:>10}")

    # =========================================================
    # TEST 2: Effect of offset magnitude (with noise)
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 2: OFFSET MAGNITUDE SENSITIVITY (with 5% noise, SNR=20)")
    print("=" * 70)
    print("\nQuestion: How small can the offset be before +C stops helping?")

    fixed_noise = 0.05  # SNR=20

    print(f"\n{'Offset':>8} {'% Amp':>8} | {'With +C dx RMSE':>16} | {'No +C dx RMSE':>16} | {'Improvement':>12}")
    print("-" * 75)

    offset_results = []
    for offset in offset_levels:
        params = base_params.copy()
        params['c_A'] = offset
        params['c_B'] = offset * 0.8
        params['c_AB'] = offset * 0.5

        # Run with +C
        res_with = run_monte_carlo(params, fixed_noise, use_offset=True, n_trials=n_trials, shape=shape)

        # Run without +C
        res_no = run_monte_carlo(params, fixed_noise, use_offset=False, n_trials=n_trials, shape=shape)

        rmse_with = np.sqrt(res_with['x0_AB_bias']**2 + res_with['x0_AB_std']**2)
        rmse_no = np.sqrt(res_no['x0_AB_bias']**2 + res_no['x0_AB_std']**2)
        improvement = (rmse_no - rmse_with) / rmse_no * 100 if rmse_no > 0 else 0

        offset_results.append({
            'offset': offset,
            'with_c': res_with, 'no_c': res_no,
            'rmse_with': rmse_with, 'rmse_no': rmse_no,
            'improvement': improvement
        })

        pct = offset / 10  # as % of amplitude (1000)
        print(f"{offset:>8.1f} {pct:>7.1f}% | {rmse_with:>16.4f} | {rmse_no:>16.4f} | {improvement:>+11.1f}%")

    # =========================================================
    # TEST 3: Sigma recovery (important for uncertainty)
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 3: SIGMA RECOVERY (affects displacement uncertainty estimate)")
    print("=" * 70)
    print("\nQuestion: Does +C corrupt sigma estimates?")
    print(f"True sx_AB={base_params['sx_AB']}, sy_AB={base_params['sy_AB']}")

    print(f"\n{'Noise':>8} | {'With +C sx_AB bias':>20} {'std':>8} | {'No +C sx_AB bias':>20} {'std':>8}")
    print("-" * 80)

    for noise in [0.0, 0.05, 0.10]:
        params = base_params.copy()
        res_with = run_monte_carlo(params, noise, use_offset=True, n_trials=n_trials, shape=shape)
        res_no = run_monte_carlo(params, noise, use_offset=False, n_trials=n_trials, shape=shape)

        print(f"{noise:>8.2f} | {res_with['sx_AB_bias']:>+20.4f} {res_with['sx_AB_std']:>8.4f} | {res_no['sx_AB_bias']:>+20.4f} {res_no['sx_AB_std']:>8.4f}")

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATION")
    print("=" * 70)

    # Check if +C ever hurts in zero-offset case
    worst_with_c = max(nr['with_c']['x0_AB_std'] for nr in noise_results)
    worst_no_c = max(nr['no_c']['x0_AB_std'] for nr in noise_results)

    print(f"\n1. ZERO OFFSET CASE:")
    print(f"   - Worst dx std with +C: {worst_with_c:.4f} pixels")
    print(f"   - Worst dx std without +C: {worst_no_c:.4f} pixels")
    if worst_with_c <= worst_no_c * 1.1:
        print(f"   → +C does NOT significantly hurt when offset is zero")
    else:
        print(f"   → +C DOES hurt when offset is zero (consider removing)")

    # Check offset threshold where +C helps
    threshold_offset = None
    for or_ in offset_results:
        if or_['improvement'] > 10:  # 10% improvement threshold
            threshold_offset = or_['offset']
            break

    print(f"\n2. OFFSET THRESHOLD:")
    if threshold_offset is not None:
        print(f"   - +C provides >10% improvement when offset ≥ {threshold_offset}")
        print(f"   - This is {threshold_offset/10:.1f}% of peak amplitude")
    else:
        print(f"   - +C never provides >10% improvement (consider removing)")

    # Final recommendation
    print(f"\n3. RECOMMENDATION:")
    if worst_with_c <= worst_no_c * 1.1 and threshold_offset is not None and threshold_offset <= 20:
        print("   ✓ KEEP +C TERM")
        print("   - Minimal cost when offset=0")
        print("   - Significant benefit when even small offsets exist")
        print("   - Background subtraction may not be perfect in practice")
    else:
        print("   ✗ CONSIDER REMOVING +C TERM")
        print("   - Cost outweighs benefit for typical conditions")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
