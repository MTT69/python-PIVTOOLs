#!/usr/bin/env python3
"""
Focused near-wall analysis of gradient corrections.
Shows ABSOLUTE values rather than misleading percentages.
"""

import numpy as np
from pathlib import Path
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_ensemble_result(result_path: Path, pass_idx: int = -1) -> dict:
    """Load ensemble result .mat file."""
    data = loadmat(str(result_path))
    ensemble = data['ensemble_result']
    num_passes = ensemble.shape[1]

    if pass_idx == -1:
        pass_idx = num_passes - 1

    result = ensemble[0, pass_idx]

    return {
        'U': np.squeeze(result['ux']),
        'V': np.squeeze(result['uy']),
        'sig_A_x': np.squeeze(result['sig_A_x']),
        'sig_A_y': np.squeeze(result['sig_A_y']),
        'sig_AB_x': np.squeeze(result['sig_AB_x']),
        'sig_AB_y': np.squeeze(result['sig_AB_y']),
        'UU_stress': np.squeeze(result['UU_stress']),
        'VV_stress': np.squeeze(result['VV_stress']),
        'UV_stress': np.squeeze(result['UV_stress']) if 'UV_stress' in result.dtype.names else None,
        'window_size': np.squeeze(result['window_size']),
        'num_passes': num_passes,
        'pass_idx': pass_idx,
    }


def load_coordinates(coord_path: Path, pass_idx: int = -1) -> dict:
    """Load coordinates .mat file."""
    data = loadmat(str(coord_path))
    coords = data['coordinates']
    num_passes = coords.shape[1]

    if pass_idx == -1:
        pass_idx = num_passes - 1

    result = coords[0, pass_idx]

    return {
        'X': np.squeeze(result['x']),
        'Y': np.squeeze(result['y']),
    }


def compute_gradients(U: np.ndarray, V: np.ndarray, X: np.ndarray, Y: np.ndarray) -> dict:
    """Compute velocity gradients, accounting for coordinate orientation."""
    if X.ndim == 2:
        dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
        dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
    else:
        dx = X[1] - X[0] if len(X) > 1 else 1.0
        dy = Y[1] - Y[0] if len(Y) > 1 else 1.0

    # np.gradient uses the spacing sign to determine gradient direction
    # If Y decreases with index (image coords), dy will be negative,
    # and gradient will have correct physical sign
    dU_dy = np.gradient(U, dy, axis=0)
    dV_dx = np.gradient(V, dx, axis=1)

    return {'dU_dy': dU_dy, 'dV_dx': dV_dx, 'dx': dx, 'dy': dy}


def analyze_nearwall(result_path: Path, coord_path: Path, save_dir: Path):
    """
    Detailed near-wall analysis showing absolute correction magnitudes.
    """
    # Load data
    data = load_ensemble_result(result_path)
    coords = load_coordinates(coord_path, data['pass_idx'])
    grads = compute_gradients(data['U'], data['V'], coords['X'], coords['Y'])

    # Get y coordinates
    Y = coords['Y']
    if Y.ndim == 2:
        y = np.nanmean(Y, axis=1)
    else:
        y = Y

    # Row-average function
    def row_avg(arr):
        if arr is None:
            return None
        return np.nanmean(arr, axis=1)

    # Compute corrections
    sig_A_x = data['sig_A_x']
    sig_A_y = data['sig_A_y']
    dU_dy = grads['dU_dy']
    dV_dx = grads['dV_dx']

    UU_correction = 0.5 * sig_A_x * (dU_dy ** 2)
    VV_correction = 0.5 * sig_A_y * (dV_dx ** 2)
    UV_correction = 0.5 * sig_A_x * (dU_dy + dV_dx)

    # Row averages
    UU_raw = row_avg(data['UU_stress'])
    VV_raw = row_avg(data['VV_stress'])
    UV_raw = row_avg(data['UV_stress'])
    UU_corr = row_avg(UU_correction)
    VV_corr = row_avg(VV_correction)
    UV_corr = row_avg(UV_correction)
    dU_dy_avg = row_avg(dU_dy)
    sig_A_x_avg = row_avg(sig_A_x)

    # Find the wall (minimum y)
    y_wall = y.min()
    y_center = y.max() / 2

    # Focus on bottom 20% of domain (near-wall region)
    y_cutoff = y_wall + 0.2 * (y.max() - y_wall)
    nearwall_mask = y < y_cutoff

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Near-Wall Gradient Correction Analysis (ABSOLUTE VALUES)\n'
                 f'Window={data["window_size"]}, Pass {data["pass_idx"]+1}/{data["num_passes"]}',
                 fontsize=14, fontweight='bold')

    # --- Plot 1: UV stress components (absolute) ---
    ax = axes[0, 0]
    ax.plot(y[nearwall_mask], UV_raw[nearwall_mask], 'b-', label='UV raw', linewidth=2)
    ax.plot(y[nearwall_mask], UV_corr[nearwall_mask], 'r-', label='UV correction', linewidth=2)
    ax.plot(y[nearwall_mask], UV_raw[nearwall_mask] - UV_corr[nearwall_mask], 'g--',
            label='UV corrected', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('UV Stress (px²/frame²)')
    ax.set_title('UV Reynolds Stress: Near-Wall Region')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Correction as fraction of max UV ---
    ax = axes[0, 1]
    uv_max = np.nanmax(np.abs(UV_raw))
    uv_corr_pct_of_max = np.abs(UV_corr) / uv_max * 100
    ax.plot(y[nearwall_mask], uv_corr_pct_of_max[nearwall_mask], 'r-', linewidth=2)
    ax.axhline(y=10, color='orange', linestyle='--', label='10% of max UV')
    ax.axhline(y=50, color='red', linestyle='--', label='50% of max UV')
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('|UV correction| / |UV_max| (%)')
    ax.set_title('UV Correction as % of PEAK UV Stress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Plot 3: dU/dy profile ---
    ax = axes[0, 2]
    ax.plot(y[nearwall_mask], dU_dy_avg[nearwall_mask], 'b-', linewidth=2)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('dU/dy (1/frame)')
    ax.set_title('Velocity Gradient dU/dy (drives UV correction)')
    ax.grid(True, alpha=0.3)

    # --- Plot 4: UU stress near wall ---
    ax = axes[1, 0]
    ax.plot(y[nearwall_mask], UU_raw[nearwall_mask], 'b-', label='UU raw', linewidth=2)
    ax.plot(y[nearwall_mask], UU_corr[nearwall_mask], 'r-', label='UU correction', linewidth=2)
    ax.plot(y[nearwall_mask], UU_raw[nearwall_mask] - UU_corr[nearwall_mask], 'g--',
            label='UU corrected', linewidth=2)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('UU Stress (px²/frame²)')
    ax.set_title('UU Reynolds Stress: Near-Wall Region')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 5: VV stress near wall ---
    ax = axes[1, 1]
    ax.plot(y[nearwall_mask], VV_raw[nearwall_mask], 'b-', label='VV raw', linewidth=2)
    ax.plot(y[nearwall_mask], VV_corr[nearwall_mask], 'r-', label='VV correction', linewidth=2)
    ax.plot(y[nearwall_mask], VV_raw[nearwall_mask] - VV_corr[nearwall_mask], 'g--',
            label='VV corrected', linewidth=2)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('VV Stress (px²/frame²)')
    ax.set_title('VV Reynolds Stress: Near-Wall Region')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 6: Correction comparison ---
    ax = axes[1, 2]
    ax.plot(y[nearwall_mask], np.abs(UU_corr[nearwall_mask]), 'b-', label='|UU correction|', linewidth=2)
    ax.plot(y[nearwall_mask], np.abs(VV_corr[nearwall_mask]), 'g-', label='|VV correction|', linewidth=2)
    ax.plot(y[nearwall_mask], np.abs(UV_corr[nearwall_mask]), 'r-', label='|UV correction|', linewidth=2)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('|Correction| (px²/frame²)')
    ax.set_title('Absolute Correction Magnitudes Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    save_path = save_dir / "gradient_corrections_nearwall.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved near-wall figure to {save_path}")
    plt.close()

    # === SECOND FIGURE: Full profile with key metrics ===
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('UV Gradient Correction: The Key Problem\n'
                  'UV_correction = 0.5 * sig_A_x * (dU/dy + dV/dx)',
                  fontsize=14, fontweight='bold')

    # Full profile of UV stress
    ax = axes2[0, 0]
    ax.plot(y, UV_raw, 'b-', label='UV raw', linewidth=2)
    ax.plot(y, UV_corr, 'r-', label='UV correction', linewidth=2)
    ax.plot(y, UV_raw - UV_corr, 'g--', label='UV corrected', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('UV Stress (px²/frame²)')
    ax.set_title('UV Stress: Full Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # dU/dy full profile
    ax = axes2[0, 1]
    ax.plot(y, dU_dy_avg, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('dU/dy (1/frame)')
    ax.set_title('Velocity Gradient dU/dy')
    ax.grid(True, alpha=0.3)

    # Zoom on near-wall UV
    ax = axes2[1, 0]
    # Bottom wall
    bot_mask = y < y_cutoff
    ax.plot(y[bot_mask], UV_raw[bot_mask], 'b-', label='UV raw', linewidth=2)
    ax.plot(y[bot_mask], UV_corr[bot_mask], 'r-', label='UV correction', linewidth=2)
    ax.plot(y[bot_mask], UV_raw[bot_mask] - UV_corr[bot_mask], 'g--', label='UV corrected', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('UV Stress (px²/frame²)')
    ax.set_title('UV Stress: ZOOM on Near-Wall')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Analysis text
    ax = axes2[1, 1]
    ax.axis('off')

    # Compute key statistics
    uv_max = np.nanmax(np.abs(UV_raw))
    uv_corr_max = np.nanmax(np.abs(UV_corr))
    dU_dy_max = np.nanmax(np.abs(dU_dy_avg))
    sig_A_mean = np.nanmean(sig_A_x_avg)

    # Find where correction > 50% of local UV
    ratio = np.abs(UV_corr) / np.maximum(np.abs(UV_raw), 1e-10)
    high_ratio_mask = ratio > 0.5
    if np.any(high_ratio_mask):
        y_high = y[high_ratio_mask]
        y_range_high = f"{y_high.min():.1f} to {y_high.max():.1f} px"
        n_high = np.sum(high_ratio_mask)
    else:
        y_range_high = "None"
        n_high = 0

    analysis_text = f"""
KEY STATISTICS:

UV Stress:
  Max |UV_raw|:        {uv_max:.4f} px²/frame²
  Max |UV_correction|: {uv_corr_max:.4f} px²/frame²
  Ratio (corr/raw):    {uv_corr_max/uv_max*100:.1f}% of peak

Gradient:
  Max |dU/dy|:         {dU_dy_max:.4f} /frame
  Mean sig_A_x:        {sig_A_mean:.4f} px²

Problem Regions (|correction/raw| > 50%):
  Y range: {y_range_high}
  Affected rows: {n_high} of {len(y)} ({100*n_high/len(y):.1f}%)

THE INSIGHT:
The UV correction term (0.5 * sig_A * dU/dy) is
problematic where:
1. dU/dy is large (near walls)
2. UV stress is small (near walls, centerline)

This creates SIGN ERRORS near walls where
the correction can flip UV from negative
to positive (or vice versa).

RECOMMENDATION:
Either apply correction carefully OR
mask/exclude near-wall regions where
|correction| > |UV_raw| * threshold
"""

    ax.text(0.05, 0.95, analysis_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path2 = save_dir / "uv_correction_analysis.png"
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved UV analysis figure to {save_path2}")
    plt.close()

    # Print summary
    print("\n" + "="*70)
    print("NEAR-WALL GRADIENT CORRECTION SUMMARY")
    print("="*70)
    print(f"\nPeak |UV stress|: {uv_max:.4f} px²/frame²")
    print(f"Peak |UV correction|: {uv_corr_max:.4f} px²/frame²")
    print(f"Correction as % of peak UV: {uv_corr_max/uv_max*100:.1f}%")
    print(f"\nMax |dU/dy|: {dU_dy_max:.4f} /frame")
    print(f"Mean sig_A_x: {sig_A_mean:.4f} px²")
    print(f"\nRows where |correction| > 50% of |raw UV|: {n_high} ({100*n_high/len(y):.1f}%)")

    return {
        'y': y,
        'UV_raw': UV_raw,
        'UV_correction': UV_corr,
        'UU_raw': UU_raw,
        'UU_correction': UU_corr,
        'VV_raw': VV_raw,
        'VV_correction': VV_corr,
        'dU_dy': dU_dy_avg,
    }


if __name__ == "__main__":
    base_dir = Path("/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/planar_images/rectangle-test")
    result_path = base_dir / "uncalibrated_piv" / "1000" / "Cam1" / "ensemble" / "ensemble_result.mat"
    coord_path = base_dir / "uncalibrated_piv" / "1000" / "Cam1" / "ensemble" / "coordinates.mat"
    save_dir = base_dir / "uncalibrated_piv" / "1000" / "Cam1" / "ensemble"

    analyze_nearwall(result_path, coord_path, save_dir)
