"""
Compare C (GSL) vs Python (scipy/TRF) k-space fitters on stored correlation planes.

Usage:
    python scripts/compare_kspace_fitters.py <path_to_planes_pass_N.mat> [--windows N] [--snr-threshold T]

Example:
    python scripts/compare_kspace_fitters.py /data/ensemble/4000/Cam1/planes_pass_3.mat
    python scripts/compare_kspace_fitters.py /data/ensemble/4000/Cam1/planes_pass_3.mat --windows 50 --snr-threshold 1.0

Requires ensemble_piv.store_planes: true in config.yaml before running ensemble PIV.
"""

import argparse
import time

import numpy as np
from scipy.io import loadmat


def load_planes(mat_path):
    """Load stored correlation planes from planes_pass_N.mat."""
    data = loadmat(mat_path, squeeze_me=True)

    R_AA = data['AA']
    R_BB = data['BB']
    R_AB = data['AB']
    corr_size = tuple(data['corr_size'].astype(int))
    n_win_y = int(data['n_win_y'])
    n_win_x = int(data['n_win_x'])

    num_windows = n_win_y * n_win_x

    R_AA_flat = R_AA.reshape(num_windows, corr_size[0], corr_size[1]).reshape(-1).astype(np.float32)
    R_BB_flat = R_BB.reshape(num_windows, corr_size[0], corr_size[1]).reshape(-1).astype(np.float32)
    R_AB_flat = R_AB.reshape(num_windows, corr_size[0], corr_size[1]).reshape(-1).astype(np.float32)

    return R_AA_flat, R_BB_flat, R_AB_flat, corr_size, num_windows, n_win_y, n_win_x


def run_comparison(mat_path, max_windows=10, snr_threshold=3.0):
    """Run both fitters on the same data and compare."""
    print(f"Loading planes from: {mat_path}")
    R_AA, R_BB, R_AB, corr_size, num_windows, n_win_y, n_win_x = load_planes(mat_path)
    print(f"  Grid: {n_win_y} x {n_win_x} = {num_windows} windows, corr_size={corr_size}")

    if max_windows and max_windows < num_windows:
        n_per_win = corr_size[0] * corr_size[1]
        R_AA = R_AA[:max_windows * n_per_win]
        R_BB = R_BB[:max_windows * n_per_win]
        R_AB = R_AB[:max_windows * n_per_win]
        num_windows = max_windows
        print(f"  Using first {max_windows} windows")

    mask = np.zeros(num_windows, dtype=bool)

    # --- Run Python (scipy/TRF) fitter ---
    print(f"\n{'='*60}")
    print(f"Python fitter (scipy TRF, snr_threshold={snr_threshold})")
    print(f"{'='*60}")

    from pivtools_cli.piv.piv_backend.kspace_fitting_python import fit_windows_kspace_python

    t0 = time.perf_counter()
    py_params, py_status, py_init, py_diag = fit_windows_kspace_python(
        R_AA, R_BB, R_AB, mask, corr_size,
        snr_threshold=snr_threshold,
    )
    py_time = time.perf_counter() - t0

    py_success = np.sum(py_status == 0)
    print(f"  Time: {py_time:.2f}s")
    print(f"  Success: {py_success}/{num_windows} ({100*py_success/num_windows:.1f}%)")
    _print_status_breakdown(py_status)

    py_snr = py_diag[:, 0]
    py_n0 = py_diag[:, 1]

    # --- Run C (GSL) fitter ---
    print(f"\n{'='*60}")
    print(f"C fitter (GSL LM, snr_threshold={snr_threshold})")
    print(f"{'='*60}")

    from pivtools_cli.piv.piv_backend.kspace_fitting import fit_windows_kspace

    t0 = time.perf_counter()
    c_params, c_status, c_init, c_diag = fit_windows_kspace(
        R_AA, R_BB, R_AB, mask, corr_size, None, 0,
        snr_threshold=snr_threshold,
        return_diagnostics=True,
    )
    c_time = time.perf_counter() - t0

    c_success = np.sum(c_status == 0)
    print(f"  Time: {c_time:.2f}s")
    print(f"  Success: {c_success}/{num_windows} ({100*c_success/num_windows:.1f}%)")
    _print_status_breakdown(c_status)

    c_snr = c_diag[:, 0]
    c_n0 = c_diag[:, 1]

    # --- Per-window comparison ---
    print(f"\n{'='*60}")
    print("Per-window comparison")
    print(f"{'='*60}")

    center_x = corr_size[1] / 2.0 + 1
    center_y = corr_size[0] / 2.0 + 1

    header = (f"  {'Win':>4}  {'Py status':>10}  {'C status':>10}  "
              f"{'Py SNR':>8}  {'C SNR':>8}  {'Py N0':>10}  {'C N0':>10}  "
              f"{'Py mu_x':>8}  {'C mu_x':>8}  {'Py Sxx':>8}  {'C Sxx':>8}")
    print(header)
    print(f"  {'-'*len(header.strip())}")

    status_names = {-1: 'masked', 0: 'success', 1: 'no_conv', 2: 'low_snr',
                    3: 'big_disp', 5: 'neg_var', 11: 'stress_ol'}

    for wi in range(num_windows):
        py_st = status_names.get(int(py_status[wi]), str(py_status[wi]))
        c_st = status_names.get(int(c_status[wi]), str(c_status[wi]))

        py_snr_val = f"{py_snr[wi]:.4f}" if not np.isnan(py_snr[wi]) else "    -"
        c_snr_val = f"{c_snr[wi]:.4f}" if not np.isnan(c_snr[wi]) else "    -"
        py_n0_val = f"{py_n0[wi]:.6f}" if not np.isnan(py_n0[wi]) else "      -"
        c_n0_val = f"{c_n0[wi]:.6f}" if not np.isnan(c_n0[wi]) else "      -"

        py_mu_x = f"{py_params[wi, 14] - center_x:.4f}" if py_status[wi] == 0 else "    -"
        c_mu_x = f"{c_params[wi, 14] - center_x:.4f}" if c_status[wi] == 0 else "    -"
        py_sxx = f"{py_params[wi, 9]:.4f}" if py_status[wi] == 0 else "    -"
        c_sxx = f"{c_params[wi, 9]:.4f}" if c_status[wi] == 0 else "    -"

        # Highlight disagreements
        marker = ""
        if py_status[wi] != c_status[wi]:
            marker = " <-- STATUS DIFFERS"
        elif py_status[wi] == 0 and c_status[wi] == 0:
            sxx_diff = abs(py_params[wi, 9] - c_params[wi, 9])
            if sxx_diff > 0.01:
                marker = f" <-- Sxx diff={sxx_diff:.4f}"

        print(f"  {wi:>4}  {py_st:>10}  {c_st:>10}  "
              f"{py_snr_val:>8}  {c_snr_val:>8}  {py_n0_val:>10}  {c_n0_val:>10}  "
              f"{py_mu_x:>8}  {c_mu_x:>8}  {py_sxx:>8}  {c_sxx:>8}{marker}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Speed ratio: {py_time/c_time:.1f}x (C faster)")

    both_ok = (py_status == 0) & (c_status == 0)
    n_both = np.sum(both_ok)
    py_only = (py_status == 0) & (c_status != 0)
    c_only = (c_status == 0) & (py_status != 0)
    print(f"  Both succeeded: {n_both}")
    print(f"  Python only: {np.sum(py_only)}")
    if np.sum(py_only) > 0:
        print(f"    C status for these: {_status_counts(c_status[py_only])}")
    print(f"  C only: {np.sum(c_only)}")
    if np.sum(c_only) > 0:
        print(f"    Python status for these: {_status_counts(py_status[c_only])}")

    # SNR/N0 comparison
    valid = ~np.isnan(py_snr) & ~np.isnan(c_snr)
    if np.any(valid):
        snr_diff = np.abs(py_snr[valid] - c_snr[valid])
        n0_diff = np.abs(py_n0[valid] - c_n0[valid])
        print(f"\n  SNR agreement ({np.sum(valid)} windows with both valid):")
        print(f"    |SNR diff| — median: {np.median(snr_diff):.4f}, max: {np.max(snr_diff):.4f}")
        print(f"    |N0 diff|  — median: {np.median(n0_diff):.6f}, max: {np.max(n0_diff):.6f}")

    if n_both > 0:
        print(f"\n  Parameter comparison ({n_both} both-succeeded windows):")
        print(f"  {'Param':<10} {'|diff| med':>12} {'|diff| max':>12}")
        print(f"  {'-'*10} {'-'*12} {'-'*12}")
        for name, idx, offset in [('mu_x', 14, center_x), ('mu_y', 15, center_y),
                                   ('Sigma_xx', 9, 0), ('Sigma_yy', 10, 0), ('Sigma_xy', 11, 0)]:
            diff = np.abs((py_params[both_ok, idx] - offset) - (c_params[both_ok, idx] - offset))
            print(f"  {name:<10} {np.median(diff):>12.6f} {np.max(diff):>12.6f}")


def _print_status_breakdown(status):
    names = {-1: 'masked', 0: 'success', 1: 'no_converge', 2: 'low_snr',
             3: 'big_disp', 5: 'neg_var', 11: 'stress_outlier'}
    counts = {}
    for s in status:
        counts[int(s)] = counts.get(int(s), 0) + 1
    parts = [f"{names.get(k, f'status_{k}')}={v}" for k, v in sorted(counts.items()) if v > 0]
    print(f"  Status: {', '.join(parts)}")


def _status_counts(status):
    names = {-1: 'masked', 0: 'success', 1: 'no_converge', 2: 'low_snr',
             3: 'big_disp', 5: 'neg_var'}
    counts = {}
    for s in status:
        counts[int(s)] = counts.get(int(s), 0) + 1
    return ', '.join(f"{names.get(k, f'{k}')}={v}" for k, v in sorted(counts.items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare C vs Python k-space fitters')
    parser.add_argument('planes_mat', help='Path to planes_pass_N.mat')
    parser.add_argument('--windows', type=int, default=10,
                        help='Max windows to test (default: 10, 0=all)')
    parser.add_argument('--snr-threshold', type=float, default=3.0,
                        help='SNR threshold (default: 3.0)')
    args = parser.parse_args()

    max_win = args.windows if args.windows > 0 else None
    run_comparison(args.planes_mat, max_windows=max_win,
                   snr_threshold=args.snr_threshold)
