#!/usr/bin/env python3
"""
Diagnostic script for analyzing Gaussian fitting initial guesses vs results.

Loads debug plane .mat files (saved when ensemble_store_planes: true) and:
1. Compares initial guess vs fitted result for all 16 parameters
2. Identifies spatial patterns of failures
3. Quantifies sigma estimation errors
4. Computes what HWHM-based estimation WOULD have produced

Usage:
    python tests/diagnose_gaussian_fitting.py path/to/planes_pass_1.mat [-o output_dir]
"""

import argparse
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import label
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Parameter indices for 16-parameter layout
PARAM_NAMES = [
    'amp_A', 'amp_B', 'amp_AB',       # 0-2: Amplitudes
    'c_A', 'c_B', 'c_AB',             # 3-5: Offsets
    'sx_A', 'sy_A', 'sxy_A',          # 6-8: Sigma A (autocorrelation)
    'sx_AB', 'sy_AB', 'sxy_AB',       # 9-11: Sigma AB (cross-correlation)
    'x0_A', 'y0_A',                   # 12-13: Center A position
    'x0_AB', 'y0_AB'                  # 14-15: Center AB position (displacement)
]


def load_debug_planes(mat_path: Path) -> dict:
    """Load debug .mat file with correlation planes and fitting results."""
    data = loadmat(str(mat_path), squeeze_me=True)
    return {
        'AA': data['AA'],                           # (n_win_y, n_win_x, corr_h, corr_w)
        'BB': data['BB'],
        'AB': data['AB'],
        'gauss_results': data['gauss_results'],     # (n_win_y, n_win_x, 16)
        'initial_guesses': data['initial_guesses'], # (n_win_y, n_win_x, 16)
        'corr_size': tuple(data['corr_size'].flatten()),
        'n_win_y': int(data['n_win_y']),
        'n_win_x': int(data['n_win_x']),
        'pass_idx': int(data['pass_idx']),
    }


def estimate_sigma_hwhm(plane: np.ndarray, peak_y: int, peak_x: int,
                        min_sigma: float = 0.5) -> tuple:
    """
    HWHM-based sigma estimation for a single correlation plane.

    Uses Half-Width at Half-Maximum to estimate Gaussian sigma.
    """
    peak_val = plane[peak_y, peak_x]
    if peak_val < 1e-6:
        return min_sigma, min_sigma

    threshold = peak_val / 2.0
    hwhm_to_sigma = 1.0 / np.sqrt(2 * np.log(2))

    # X-direction profile at peak row
    x_profile = plane[peak_y, :]
    x_above = np.where(x_profile >= threshold)[0]
    if len(x_above) >= 2:
        hwhm_x = (x_above[-1] - x_above[0]) / 2.0
        sigma_x = max(hwhm_x * hwhm_to_sigma, min_sigma)
    else:
        sigma_x = min_sigma

    # Y-direction profile at peak column
    y_profile = plane[:, peak_x]
    y_above = np.where(y_profile >= threshold)[0]
    if len(y_above) >= 2:
        hwhm_y = (y_above[-1] - y_above[0]) / 2.0
        sigma_y = max(hwhm_y * hwhm_to_sigma, min_sigma)
    else:
        sigma_y = min_sigma

    return sigma_x, sigma_y


def compute_hwhm_sigmas_for_planes(data: dict) -> dict:
    """
    Compute what HWHM-based sigma estimation WOULD produce for all windows.

    This allows comparison with:
    - Current fixed defaults (3.0, 1.5)
    - Actual fitted values
    """
    n_win_y, n_win_x = data['n_win_y'], data['n_win_x']
    corr_h, corr_w = data['corr_size']

    # Output arrays
    hwhm_sigma_A_x = np.zeros((n_win_y, n_win_x))
    hwhm_sigma_A_y = np.zeros((n_win_y, n_win_x))
    hwhm_sigma_AB_x = np.zeros((n_win_y, n_win_x))
    hwhm_sigma_AB_y = np.zeros((n_win_y, n_win_x))

    # Center index for AA/BB autocorrelation
    center_y, center_x = corr_h // 2, corr_w // 2

    for iy in range(n_win_y):
        for ix in range(n_win_x):
            # AA autocorrelation (particle image size)
            AA_plane = data['AA'][iy, ix]
            hwhm_sigma_A_x[iy, ix], hwhm_sigma_A_y[iy, ix] = estimate_sigma_hwhm(
                AA_plane, center_y, center_x, min_sigma=0.5
            )

            # AB cross-correlation (find peak first)
            AB_plane = data['AB'][iy, ix]
            peak_idx = np.argmax(AB_plane)
            peak_y, peak_x = np.unravel_index(peak_idx, AB_plane.shape)

            # Raw AB sigma (includes particle contribution)
            sigma_AB_raw_x, sigma_AB_raw_y = estimate_sigma_hwhm(
                AB_plane, peak_y, peak_x, min_sigma=0.1
            )

            # Subtract particle contribution (quadrature)
            # sigma_total^2 = sigma_particle^2 + sigma_displacement^2
            hwhm_A_x = hwhm_sigma_A_x[iy, ix] * np.sqrt(2 * np.log(2))
            hwhm_A_y = hwhm_sigma_A_y[iy, ix] * np.sqrt(2 * np.log(2))
            hwhm_AB_x = sigma_AB_raw_x * np.sqrt(2 * np.log(2))
            hwhm_AB_y = sigma_AB_raw_y * np.sqrt(2 * np.log(2))

            min_hwhm = 0.1 * np.sqrt(2 * np.log(2))
            hwhm_diff_x = np.sqrt(max(hwhm_AB_x**2 - hwhm_A_x**2, min_hwhm**2))
            hwhm_diff_y = np.sqrt(max(hwhm_AB_y**2 - hwhm_A_y**2, min_hwhm**2))

            hwhm_sigma_AB_x[iy, ix] = hwhm_diff_x / np.sqrt(2 * np.log(2))
            hwhm_sigma_AB_y[iy, ix] = hwhm_diff_y / np.sqrt(2 * np.log(2))

    return {
        'sigma_A_x': hwhm_sigma_A_x,
        'sigma_A_y': hwhm_sigma_A_y,
        'sigma_AB_x': hwhm_sigma_AB_x,
        'sigma_AB_y': hwhm_sigma_AB_y,
    }


def compute_parameter_errors(initial: np.ndarray, fitted: np.ndarray) -> dict:
    """
    Compute per-parameter errors between initial guess and fitted result.
    """
    abs_err = fitted - initial

    # Relative error (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = np.abs(abs_err) / np.maximum(np.abs(fitted), 1e-10)
        rel_err = np.where(np.isfinite(rel_err), rel_err, np.nan)

    # Per-parameter statistics
    per_param_stats = {}
    for i, name in enumerate(PARAM_NAMES):
        param_abs = abs_err[:, :, i].ravel()
        param_rel = rel_err[:, :, i].ravel()
        valid_mask = np.isfinite(param_abs)

        per_param_stats[name] = {
            'mean_abs': np.nanmean(np.abs(param_abs)),
            'std_abs': np.nanstd(param_abs),
            'max_abs': np.nanmax(np.abs(param_abs[valid_mask])) if valid_mask.any() else np.nan,
            'p95_abs': np.nanpercentile(np.abs(param_abs), 95),
            'mean_rel': np.nanmean(param_rel),
        }

    return {
        'absolute_error': abs_err,
        'relative_error': rel_err,
        'per_param_stats': per_param_stats,
    }


def identify_failure_patterns(initial: np.ndarray, fitted: np.ndarray,
                              threshold_sigma_err: float = 2.0) -> dict:
    """
    Identify spatial patterns of fitting failures.
    """
    # Criteria for "failure"
    sx_AB_init = initial[:, :, 9]
    sy_AB_init = initial[:, :, 10]
    sx_AB_fit = fitted[:, :, 9]
    sy_AB_fit = fitted[:, :, 10]

    sigma_err_x = np.abs(sx_AB_fit - sx_AB_init) / np.maximum(sx_AB_init, 0.1)
    sigma_err_y = np.abs(sy_AB_fit - sy_AB_init) / np.maximum(sy_AB_init, 0.1)

    # Displacement change
    disp_err = np.sqrt(
        (fitted[:, :, 14] - initial[:, :, 14])**2 +
        (fitted[:, :, 15] - initial[:, :, 15])**2
    )

    # Negative fitted sigma (unphysical)
    neg_sigma = (sx_AB_fit < 0) | (sy_AB_fit < 0)

    # Combined failure mask
    failure_mask = (
        (sigma_err_x > threshold_sigma_err) |
        (sigma_err_y > threshold_sigma_err) |
        (disp_err > 3.0) |
        neg_sigma
    )

    # Find connected components
    cluster_labels, n_clusters = label(failure_mask)

    return {
        'failure_mask': failure_mask,
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters,
        'failure_rate': failure_mask.sum() / failure_mask.size,
        'sigma_err_x': sigma_err_x,
        'sigma_err_y': sigma_err_y,
        'disp_err': disp_err,
    }


def generate_diagnostic_report(mat_path: Path, output_dir: Path = None):
    """Generate comprehensive diagnostic report for a debug .mat file."""
    print(f"\n{'='*70}")
    print(f"Gaussian Fitting Diagnostic Report")
    print(f"File: {mat_path.name}")
    print(f"{'='*70}\n")

    data = load_debug_planes(mat_path)

    initial = data['initial_guesses']
    fitted = data['gauss_results']
    n_win_y, n_win_x = data['n_win_y'], data['n_win_x']
    pass_idx = data['pass_idx']

    print(f"Pass: {pass_idx + 1}")
    print(f"Grid size: {n_win_y} x {n_win_x} = {n_win_y * n_win_x} windows")
    print(f"Correlation window: {data['corr_size']}")

    # 1. Parameter error analysis
    print(f"\n{'='*70}")
    print("1. PARAMETER ERROR ANALYSIS (Initial Guess vs Fitted)")
    print("-" * 70)
    errors = compute_parameter_errors(initial, fitted)

    print(f"{'Parameter':<12} {'Mean|Err|':>10} {'P95|Err|':>10} {'Rel.Err':>10}")
    print("-" * 42)
    for name in ['sx_A', 'sy_A', 'sx_AB', 'sy_AB', 'x0_AB', 'y0_AB']:
        stats = errors['per_param_stats'][name]
        print(f"{name:<12} {stats['mean_abs']:>10.3f} {stats['p95_abs']:>10.3f} {stats['mean_rel']:>10.1%}")

    # 2. Sigma comparison: Fixed defaults vs HWHM vs Fitted
    print(f"\n{'='*70}")
    print("2. SIGMA ESTIMATION COMPARISON")
    print("-" * 70)

    # Compute what HWHM would give
    print("Computing HWHM-based sigma estimates...")
    hwhm_sigmas = compute_hwhm_sigmas_for_planes(data)

    print(f"\n{'Method':<20} {'sigma_A_x':>12} {'sigma_A_y':>12} {'sigma_AB_x':>12} {'sigma_AB_y':>12}")
    print("-" * 70)

    # Fixed defaults (what production currently uses)
    print(f"{'Fixed defaults':<20} {'3.00':>12} {'3.00':>12} {'1.50':>12} {'1.50':>12}")

    # Initial guess (from the file - should match fixed defaults for pass 0)
    print(f"{'Initial guess':<20} "
          f"{np.nanmedian(initial[:,:,6]):>12.2f} "
          f"{np.nanmedian(initial[:,:,7]):>12.2f} "
          f"{np.nanmedian(initial[:,:,9]):>12.2f} "
          f"{np.nanmedian(initial[:,:,10]):>12.2f}")

    # HWHM-based (what we propose)
    print(f"{'HWHM-based':<20} "
          f"{np.nanmedian(hwhm_sigmas['sigma_A_x']):>12.2f} "
          f"{np.nanmedian(hwhm_sigmas['sigma_A_y']):>12.2f} "
          f"{np.nanmedian(hwhm_sigmas['sigma_AB_x']):>12.2f} "
          f"{np.nanmedian(hwhm_sigmas['sigma_AB_y']):>12.2f}")

    # Fitted result (ground truth for comparison)
    print(f"{'Fitted result':<20} "
          f"{np.nanmedian(fitted[:,:,6]):>12.2f} "
          f"{np.nanmedian(fitted[:,:,7]):>12.2f} "
          f"{np.nanmedian(fitted[:,:,9]):>12.2f} "
          f"{np.nanmedian(fitted[:,:,10]):>12.2f}")

    # Show ranges
    print(f"\n{'Method':<20} {'Range sigma_A_x':>25} {'Range sigma_AB_x':>25}")
    print("-" * 70)
    print(f"{'HWHM-based':<20} "
          f"[{np.nanmin(hwhm_sigmas['sigma_A_x']):.2f}, {np.nanmax(hwhm_sigmas['sigma_A_x']):.2f}]"
          f"{' ':>10}"
          f"[{np.nanmin(hwhm_sigmas['sigma_AB_x']):.2f}, {np.nanmax(hwhm_sigmas['sigma_AB_x']):.2f}]")
    print(f"{'Fitted result':<20} "
          f"[{np.nanmin(fitted[:,:,6]):.2f}, {np.nanmax(fitted[:,:,6]):.2f}]"
          f"{' ':>10}"
          f"[{np.nanmin(fitted[:,:,9]):.2f}, {np.nanmax(fitted[:,:,9]):.2f}]")

    # 3. Failure pattern analysis
    print(f"\n{'='*70}")
    print("3. FAILURE PATTERN ANALYSIS")
    print("-" * 70)
    patterns = identify_failure_patterns(initial, fitted)
    print(f"Failure rate: {patterns['failure_rate']:.1%} ({int(patterns['failure_rate'] * n_win_y * n_win_x)} / {n_win_y * n_win_x})")
    print(f"Number of failure clusters: {patterns['n_clusters']}")

    if patterns['n_clusters'] > 0:
        # Analyze cluster sizes
        cluster_sizes = []
        for i in range(1, patterns['n_clusters'] + 1):
            cluster_sizes.append((patterns['cluster_labels'] == i).sum())
        cluster_sizes.sort(reverse=True)
        print(f"Largest clusters: {cluster_sizes[:5]}")

    # 4. Error correlation analysis
    print(f"\n{'='*70}")
    print("4. ERROR CORRELATION WITH INITIAL GUESS")
    print("-" * 70)

    # Does poor initial guess correlate with failure?
    sigma_init_error = np.abs(initial[:,:,9] - fitted[:,:,9])  # How wrong was initial guess
    failed_init_error = sigma_init_error[patterns['failure_mask']]
    success_init_error = sigma_init_error[~patterns['failure_mask']]

    print(f"Mean init error (FAILED windows):   {np.nanmean(failed_init_error):.3f}")
    print(f"Mean init error (SUCCESS windows):  {np.nanmean(success_init_error):.3f}")

    # 5. Generate visualizations
    if output_dir and HAS_MATPLOTLIB:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Failure mask
        im0 = axes[0, 0].imshow(patterns['failure_mask'], cmap='Reds', origin='lower')
        axes[0, 0].set_title('Failure Mask')
        plt.colorbar(im0, ax=axes[0, 0])

        # Sigma error
        im1 = axes[0, 1].imshow(patterns['sigma_err_x'], cmap='hot', vmax=5, origin='lower')
        axes[0, 1].set_title('Sigma X Relative Error')
        plt.colorbar(im1, ax=axes[0, 1])

        # Displacement error
        im2 = axes[0, 2].imshow(patterns['disp_err'], cmap='hot', vmax=5, origin='lower')
        axes[0, 2].set_title('Displacement Error (pixels)')
        plt.colorbar(im2, ax=axes[0, 2])

        # HWHM vs Fitted sigma_A comparison
        im3 = axes[1, 0].imshow(hwhm_sigmas['sigma_A_x'], cmap='viridis', origin='lower')
        axes[1, 0].set_title('HWHM sigma_A_x (proposed)')
        plt.colorbar(im3, ax=axes[1, 0])

        # Fitted sigma_A
        im4 = axes[1, 1].imshow(fitted[:,:,6], cmap='viridis', origin='lower',
                                 vmin=hwhm_sigmas['sigma_A_x'].min(),
                                 vmax=hwhm_sigmas['sigma_A_x'].max())
        axes[1, 1].set_title('Fitted sigma_A_x (ground truth)')
        plt.colorbar(im4, ax=axes[1, 1])

        # Cluster labels
        im5 = axes[1, 2].imshow(patterns['cluster_labels'], cmap='nipy_spectral', origin='lower')
        axes[1, 2].set_title(f'Failure Clusters (n={patterns["n_clusters"]})')
        plt.colorbar(im5, ax=axes[1, 2])

        plt.suptitle(f'Gaussian Fitting Diagnostics: {mat_path.name}', fontsize=14)
        plt.tight_layout()

        out_path = output_dir / f"diagnostic_{mat_path.stem}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"\nVisualization saved to: {out_path}")
    elif output_dir and not HAS_MATPLOTLIB:
        print("\nWarning: matplotlib not available, skipping visualization")

    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")

    return {
        'data': data,
        'errors': errors,
        'patterns': patterns,
        'hwhm_sigmas': hwhm_sigmas,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose Gaussian fitting quality from debug planes'
    )
    parser.add_argument('mat_file', type=Path,
                        help='Path to planes_pass_N.mat file')
    parser.add_argument('--output', '-o', type=Path,
                        help='Output directory for plots')
    args = parser.parse_args()

    if not args.mat_file.exists():
        print(f"Error: File not found: {args.mat_file}")
        return 1

    generate_diagnostic_report(args.mat_file, args.output)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
