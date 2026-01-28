#!/usr/bin/env python3
"""
K-Space Adaptive Window Analysis Script

This script analyzes how k_max bounds are determined for k-space fitting.
It loads saved correlation planes and generates diagnostic figures showing:

1. How k_max is computed from F_ref (particle image point-spread) decay
2. How k_max varies by window location (center vs edge vs corner)
3. Statistics of k_max across all windows
4. The effect of the 0.25 hard cap

Usage:
    python analyze_kspace_kmax.py <planes_path> [--output_dir <dir>]

Example:
    python analyze_kspace_kmax.py tests/rs_q4_test_5000/output_kspace_single/uncalibrated_piv/5000/Cam1/ensemble/planes_pass_1.mat
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import scipy.io as sio
from scipy.fft import fft2, fftshift, ifftshift, fftfreq
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def compute_kmax_from_profile(k_axis, F_profile, F_dc, threshold_frac=0.01,
                               min_k=0.05, max_k=0.25):
    """
    Compute k_max from where F_ref profile drops below threshold.

    Parameters
    ----------
    k_axis : np.ndarray
        Wavenumber axis (centered, from fftshift(fftfreq()))
    F_profile : np.ndarray
        |F_ref| profile along this axis
    F_dc : float
        DC value |F_ref(0)|
    threshold_frac : float
        Fraction of DC below which to cut off (default 0.01 = 1%)
    min_k, max_k : float
        Bounds on k_max

    Returns
    -------
    float
        k_max value (uncapped, for analysis)
    """
    threshold = F_dc * threshold_frac

    # Only look at positive k (right half of array after fftshift)
    center = len(k_axis) // 2
    k_pos = k_axis[center:]
    F_pos = F_profile[center:]

    # Find first index where F drops below threshold
    below_threshold = F_pos < threshold
    if np.any(below_threshold):
        idx = np.argmax(below_threshold)
        k_max = k_pos[max(0, idx - 1)]
    else:
        k_max = 0.5  # No crossing found up to Nyquist

    return k_max


def compute_kmax_from_sigma(sigma, snr, min_k=0.05, max_k=0.25):
    """
    Compute k_max from variance (Sigma) and SNR.

    k_max = sqrt(ln(SNR) / (2*pi^2 * Sigma))
    """
    if sigma <= 0 or snr <= 1:
        return max_k
    k_max = np.sqrt(np.log(snr) / (2 * np.pi**2 * sigma + 1e-12))
    return k_max


def analyze_single_window(R_AA, R_BB, R_AB, k_x, k_y, K_X, K_Y):
    """
    Analyze k-space properties of a single window.

    Returns dict with F_ref, SNR, k_max values, and profiles.
    """
    corr_h, corr_w = R_AA.shape
    center_idx_x = corr_w // 2
    center_idx_y = corr_h // 2

    # Compute FFTs (with ifftshift since correlation has peak at center)
    F_AA = fftshift(fft2(ifftshift(R_AA)))
    F_BB = fftshift(fft2(ifftshift(R_BB)))
    F_AB = fftshift(fft2(ifftshift(R_AB)))

    # Particle shape reference
    F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))
    F_ref_dc = np.abs(F_ref[center_idx_y, center_idx_x])

    # Extract profiles along axes
    F_ref_profile_x = np.abs(F_ref[center_idx_y, :])
    F_ref_profile_y = np.abs(F_ref[:, center_idx_x])

    # SNR estimate
    dc_power = F_ref_dc ** 2
    corner_region = np.abs(F_ref[:3, :3]) ** 2
    noise_power = np.median(corner_region) + 1e-12
    snr = dc_power / noise_power

    # k_max from F_ref decay (1% threshold)
    k_max_x_1pct = compute_kmax_from_profile(k_x, F_ref_profile_x, F_ref_dc, 0.01)
    k_max_y_1pct = compute_kmax_from_profile(k_y, F_ref_profile_y, F_ref_dc, 0.01)

    return {
        'F_ref': F_ref,
        'F_AB': F_AB,
        'F_ref_dc': F_ref_dc,
        'F_ref_profile_x': F_ref_profile_x,
        'F_ref_profile_y': F_ref_profile_y,
        'snr': snr,
        'noise_floor': np.sqrt(noise_power),
        'k_max_x_1pct': k_max_x_1pct,
        'k_max_y_1pct': k_max_y_1pct,
    }


def create_kmax_determination_figure(planes_data, output_dir, window_idx=(15, 15)):
    """
    Create figure showing how k_max is determined from F_ref decay.
    """
    AA = planes_data['AA']
    BB = planes_data['BB']
    AB = planes_data['AB']

    wy, wx = window_idx
    R_AA = AA[wy, wx]
    R_BB = BB[wy, wx]
    R_AB = AB[wy, wx]

    corr_h, corr_w = R_AA.shape
    center_idx = corr_w // 2

    # Build wavenumber grids
    k_x = fftshift(fftfreq(corr_w))
    k_y = fftshift(fftfreq(corr_h))
    K_X, K_Y = np.meshgrid(k_x, k_y, indexing='xy')

    # Analyze window
    analysis = analyze_single_window(R_AA, R_BB, R_AB, k_x, k_y, K_X, K_Y)

    F_ref = analysis['F_ref']
    F_ref_dc = analysis['F_ref_dc']
    snr = analysis['snr']
    k_max_x = analysis['k_max_x_1pct']
    k_max_y = analysis['k_max_y_1pct']

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1a: F_ref 2D
    ax = axes[0, 0]
    log_F_ref = np.log10(np.abs(F_ref) + 1)
    im = ax.imshow(log_F_ref, cmap='viridis', origin='lower',
                   extent=[k_x[0], k_x[-1], k_y[0], k_y[-1]])
    ax.axhline(0, color='r', linestyle='--', alpha=0.5, label='k_y=0 profile')
    ax.axvline(0, color='b', linestyle='--', alpha=0.5, label='k_x=0 profile')
    ax.set_title(f'log10(|F_ref|+1)\nWindow ({wy}, {wx})')
    ax.set_xlabel('k_x [cycles/px]')
    ax.set_ylabel('k_y [cycles/px]')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.legend(loc='upper right', fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 1b: F_ref profile along k_x
    ax = axes[0, 1]
    ax.semilogy(k_x, analysis['F_ref_profile_x'], 'b.-', lw=1.5, markersize=4,
                label='|F_ref(k_x, 0)|')
    ax.axhline(F_ref_dc * 0.01, color='r', linestyle='--', lw=2,
               label='1% of DC (threshold)')
    ax.axhline(analysis['noise_floor'], color='gray', linestyle='-.', lw=1.5,
               label='noise floor')

    k_max_x_capped = min(k_max_x, 0.25)
    ax.axvline(k_max_x_capped, color='green', linestyle='-', lw=2,
               label=f'k_max_x = {k_max_x_capped:.3f}')
    ax.axvline(-k_max_x_capped, color='green', linestyle='-', lw=2)
    ax.fill_betweenx([1, F_ref_dc*2], -k_max_x_capped, k_max_x_capped,
                     alpha=0.1, color='green')

    ax.set_xlabel('k_x [cycles/px]')
    ax.set_ylabel('|F_ref|')
    ax.set_title('Stage 1: F_ref decay → k_max (x-axis)')
    ax.legend(loc='lower left', fontsize=8)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(1, F_ref_dc * 2)
    ax.grid(True, alpha=0.3)

    # 1c: F_ref profile along k_y
    ax = axes[0, 2]
    ax.semilogy(k_y, analysis['F_ref_profile_y'], 'r.-', lw=1.5, markersize=4,
                label='|F_ref(0, k_y)|')
    ax.axhline(F_ref_dc * 0.01, color='r', linestyle='--', lw=2, label='1% of DC')
    ax.axhline(analysis['noise_floor'], color='gray', linestyle='-.', lw=1.5,
               label='noise floor')

    k_max_y_capped = min(k_max_y, 0.25)
    ax.axvline(k_max_y_capped, color='green', linestyle='-', lw=2,
               label=f'k_max_y = {k_max_y_capped:.3f}')
    ax.axvline(-k_max_y_capped, color='green', linestyle='-', lw=2)
    ax.fill_betweenx([1, F_ref_dc*2], -k_max_y_capped, k_max_y_capped,
                     alpha=0.1, color='green')

    ax.set_xlabel('k_y [cycles/px]')
    ax.set_ylabel('|F_ref|')
    ax.set_title('Stage 1: F_ref decay → k_max (y-axis)')
    ax.legend(loc='lower left', fontsize=8)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(1, F_ref_dc * 2)
    ax.grid(True, alpha=0.3)

    # 2a: k_max vs Sigma for different SNR
    ax = axes[1, 0]
    sigma_range = np.linspace(0.1, 10, 100)
    snr_values = [1e3, 1e6, 1e9, snr]
    colors = ['blue', 'orange', 'red', 'green']
    for snr_val, c in zip(snr_values, colors):
        k_max_curve = [min(compute_kmax_from_sigma(s, snr_val), 0.5) for s in sigma_range]
        label = f'SNR={snr_val:.0e}' if snr_val != snr else f'SNR={snr:.1e} (actual)'
        ax.plot(sigma_range, k_max_curve, c=c, lw=2, label=label)

    ax.axhline(0.25, color='gray', linestyle='--', label='cap = 0.25')
    ax.set_xlabel('Sigma (variance) [px²]')
    ax.set_ylabel('k_max [cycles/px]')
    ax.set_title('Stage 2: k_max from Sigma & SNR')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3)

    # 2b: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
K-SPACE ADAPTIVE WINDOW SUMMARY
================================

Window Location: ({wy}, {wx})
Correlation Size: {corr_h} x {corr_w}

F_ref DC Value: {F_ref_dc:.1f}
Noise Floor: {analysis['noise_floor']:.2f}
SNR: {snr:.2e}

From 1% F_ref Threshold:
  k_max_x (raw): {k_max_x:.4f}
  k_max_y (raw): {k_max_y:.4f}

After 0.25 Cap:
  k_max_x: {min(k_max_x, 0.25):.4f}
  k_max_y: {min(k_max_y, 0.25):.4f}

Interpretation:
  {'SNR is very high - limited by 0.25 cap, not noise'
   if snr > 1e6 else
   'SNR is moderate - may be noise limited'}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, family='monospace', verticalalignment='top')

    # 2c: Final k-region
    ax = axes[1, 2]
    log_F_AB = np.log10(np.abs(analysis['F_AB']) + 1)
    im = ax.imshow(log_F_AB, cmap='magma', origin='lower',
                   extent=[k_x[0], k_x[-1], k_y[0], k_y[-1]])

    k_max_final = min(k_max_x_capped, k_max_y_capped)
    ellipse = Ellipse((0, 0), 2*k_max_x_capped, 2*k_max_y_capped,
                      fill=False, edgecolor='lime', linewidth=3)
    ax.add_patch(ellipse)

    # 1% F_ref contour
    contour = ax.contour(K_X, K_Y, np.abs(F_ref) / F_ref_dc, levels=[0.01],
                         colors=['cyan'], linewidths=2, linestyles='--')

    ax.set_title('Final k-region used for fit')
    ax.set_xlabel('k_x [cycles/px]')
    ax.set_ylabel('k_y [cycles/px]')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    plt.colorbar(im, ax=ax, shrink=0.8, label='log10|F_AB|')

    fig.suptitle('K-Space Adaptive Window Determination', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = output_dir / 'kmax_determination.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

    return analysis


def create_window_comparison_figure(planes_data, output_dir):
    """
    Compare k_max determination across different window locations.
    """
    AA = planes_data['AA']
    BB = planes_data['BB']
    AB = planes_data['AB']

    n_win_y, n_win_x = AA.shape[:2]
    corr_h, corr_w = AA.shape[2:]

    k_x = fftshift(fftfreq(corr_w))
    k_y = fftshift(fftfreq(corr_h))
    K_X, K_Y = np.meshgrid(k_x, k_y, indexing='xy')

    # Select representative windows
    center_y, center_x = n_win_y // 2, n_win_x // 2
    window_locs = [
        (center_y, center_x, 'Center'),
        (3, center_x, 'Top Edge'),
        (3, 3, 'Corner'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for col, (wy, wx, name) in enumerate(window_locs):
        R_AA = AA[wy, wx]
        R_BB = BB[wy, wx]
        R_AB = AB[wy, wx]

        analysis = analyze_single_window(R_AA, R_BB, R_AB, k_x, k_y, K_X, K_Y)

        k_max = min(analysis['k_max_x_1pct'], 0.25)

        # Top row: F_ref
        ax = axes[0, col]
        log_F_ref = np.log10(np.abs(analysis['F_ref']) + 1)
        im = ax.imshow(log_F_ref, cmap='viridis', origin='lower',
                       extent=[k_x[0], k_x[-1], k_y[0], k_y[-1]])
        ellipse = Ellipse((0, 0), 2*k_max, 2*k_max,
                          fill=False, edgecolor='red', linewidth=2, linestyle='--')
        ax.add_patch(ellipse)
        ax.set_title(f'{name} ({wy},{wx})\nk_max={k_max:.3f}, SNR={analysis["snr"]:.1e}')
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.35, 0.35)
        if col == 0:
            ax.set_ylabel('k_y [cycles/px]')
        ax.set_xlabel('k_x [cycles/px]')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Bottom row: F_ref profile
        ax = axes[1, col]
        ax.semilogy(k_x, analysis['F_ref_profile_x'], 'b.-', lw=1.5, markersize=3)
        ax.axhline(analysis['F_ref_dc'] * 0.01, color='r', linestyle='--', lw=2,
                   label='1% threshold')
        ax.axvline(k_max, color='green', linestyle='-', lw=2,
                   label=f'k_max={k_max:.3f}')
        ax.axvline(-k_max, color='green', linestyle='-', lw=2)
        ax.fill_betweenx([1, analysis['F_ref_dc']*2], -k_max, k_max,
                         alpha=0.1, color='green')
        ax.set_xlabel('k_x [cycles/px]')
        if col == 0:
            ax.set_ylabel('|F_ref(k_x, 0)|')
        ax.set_xlim(-0.4, 0.4)
        ax.set_ylim(max(1, analysis['F_ref_dc']/1e6), analysis['F_ref_dc'] * 2)
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'F_ref profile')

    fig.suptitle('K-max Varies by Window Location', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = output_dir / 'kmax_by_location.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def create_statistics_figure(planes_data, output_dir):
    """
    Create histogram of k_max values across all windows.
    """
    AA = planes_data['AA']
    BB = planes_data['BB']

    n_win_y, n_win_x = AA.shape[:2]
    corr_h, corr_w = AA.shape[2:]

    k_x = fftshift(fftfreq(corr_w))
    k_y = fftshift(fftfreq(corr_h))
    K_X, K_Y = np.meshgrid(k_x, k_y, indexing='xy')
    center_idx = corr_w // 2

    # Collect k_max and SNR for all windows
    k_max_x_all = []
    k_max_y_all = []
    snr_all = []

    for wy in range(n_win_y):
        for wx in range(n_win_x):
            R_AA = AA[wy, wx]
            R_BB = BB[wy, wx]

            F_AA = fftshift(fft2(ifftshift(R_AA)))
            F_BB = fftshift(fft2(ifftshift(R_BB)))
            F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))
            F_ref_dc = np.abs(F_ref[center_idx, center_idx])

            F_ref_profile_x = np.abs(F_ref[center_idx, :])
            F_ref_profile_y = np.abs(F_ref[:, center_idx])

            k_max_x = compute_kmax_from_profile(k_x, F_ref_profile_x, F_ref_dc, 0.01)
            k_max_y = compute_kmax_from_profile(k_y, F_ref_profile_y, F_ref_dc, 0.01)

            dc_power = F_ref_dc ** 2
            corner_region = np.abs(F_ref[:3, :3]) ** 2
            noise_power = np.median(corner_region) + 1e-12
            snr = dc_power / noise_power

            k_max_x_all.append(k_max_x)
            k_max_y_all.append(k_max_y)
            snr_all.append(snr)

    k_max_x_all = np.array(k_max_x_all)
    k_max_y_all = np.array(k_max_y_all)
    k_max_all = np.minimum(k_max_x_all, k_max_y_all)
    k_max_capped = np.clip(k_max_all, 0.05, 0.25)
    snr_all = np.array(snr_all)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1: Distribution of raw k_max
    ax = axes[0]
    ax.hist(k_max_all, bins=30, alpha=0.7, edgecolor='blue',
            label='Raw k_max from 1% threshold')
    ax.axvline(0.25, color='r', linestyle='--', lw=2, label='Cap at 0.25')
    ax.axvline(np.median(k_max_all), color='green', linestyle='-', lw=2,
               label=f'Median = {np.median(k_max_all):.3f}')
    ax.set_xlabel('k_max [cycles/px]')
    ax.set_ylabel('Number of windows')
    ax.set_title('Distribution of k_max from F_ref decay')
    ax.legend()

    # 2: Distribution of SNR
    ax = axes[1]
    # Filter out zero/negative SNR values for log plot
    valid_snr = snr_all[snr_all > 0]
    ax.hist(np.log10(valid_snr), bins=30, alpha=0.7, edgecolor='blue', color='orange')
    ax.axvline(np.log10(np.median(valid_snr)), color='green', linestyle='-', lw=2,
               label=f'Median = {np.median(valid_snr):.1e}')
    ax.set_xlabel('log10(SNR)')
    ax.set_ylabel('Number of windows')
    ax.set_title('Distribution of SNR')
    ax.legend()

    # 3: Summary statistics
    ax = axes[2]
    ax.axis('off')

    pct_at_cap = 100 * np.sum(k_max_all >= 0.25) / len(k_max_all)

    stats_text = f"""
K-MAX STATISTICS SUMMARY
========================

Total windows: {len(k_max_all)}

Raw k_max (from 1% F_ref threshold):
  Min: {np.min(k_max_all):.4f}
  Median: {np.median(k_max_all):.4f}
  Max: {np.max(k_max_all):.4f}

After 0.25 cap:
  Min: {np.min(k_max_capped):.4f}
  Median: {np.median(k_max_capped):.4f}
  Max: {np.max(k_max_capped):.4f}

Windows hitting 0.25 cap: {pct_at_cap:.1f}%

SNR Statistics:
  Min: {np.min(snr_all):.2e}
  Median: {np.median(snr_all):.2e}
  Max: {np.max(snr_all):.2e}

Interpretation:
{f"  High SNR ensemble - {pct_at_cap:.0f}% of windows limited by" if pct_at_cap > 50 else
 f"  Moderate SNR ensemble - {100-pct_at_cap:.0f}% of windows limited by"}
{"  the 0.25 hard cap, not noise" if pct_at_cap > 50 else
 "  signal decay, not the hard cap"}
"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, family='monospace', verticalalignment='top')

    fig.suptitle('K-max Statistics Across All Windows', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = output_dir / 'kmax_statistics.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

    return {
        'k_max_raw': k_max_all,
        'k_max_capped': k_max_capped,
        'snr': snr_all,
        'pct_at_cap': pct_at_cap,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze k-space adaptive window determination',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('planes_path', type=str,
                        help='Path to planes_pass_*.mat file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures (default: same as planes file)')
    parser.add_argument('--window', type=str, default=None,
                        help='Window index for detailed analysis, e.g. "15,15"')

    args = parser.parse_args()

    planes_path = Path(args.planes_path)
    if not planes_path.exists():
        print(f"Error: {planes_path} not found")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else planes_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading correlation planes from: {planes_path}")
    planes_data = sio.loadmat(planes_path)

    # Get window index
    if args.window:
        wy, wx = map(int, args.window.split(','))
        window_idx = (wy, wx)
    else:
        # Default to center
        n_win_y, n_win_x = planes_data['AA'].shape[:2]
        window_idx = (n_win_y // 2, n_win_x // 2)

    print(f"\nCreating k_max determination figure for window {window_idx}...")
    analysis = create_kmax_determination_figure(planes_data, output_dir, window_idx)

    print(f"\nCreating window comparison figure...")
    create_window_comparison_figure(planes_data, output_dir)

    print(f"\nCreating statistics figure...")
    stats = create_statistics_figure(planes_data, output_dir)

    print(f"\n{'='*50}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Output saved to: {output_dir}")
    print(f"\nKey findings:")
    print(f"  - {stats['pct_at_cap']:.1f}% of windows hit the 0.25 k_max cap")
    print(f"  - Median SNR: {np.median(stats['snr']):.2e}")
    print(f"  - Median raw k_max: {np.median(stats['k_max_raw']):.4f}")


if __name__ == '__main__':
    main()
