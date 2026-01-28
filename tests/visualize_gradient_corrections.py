#!/usr/bin/env python3
"""
Visualize row-by-row gradient corrections for Reynolds stresses.

Shows how the gradient correction term varies with y-position and its
percentage contribution relative to the raw stresses.
"""

import numpy as np
from pathlib import Path
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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
    """Compute velocity gradients using central differences.

    NOTE: Uses signed spacing to correctly handle inverted coordinates.
    If Y decreases with row index (image coordinates), dy will be negative,
    giving the correct physical gradient direction.
    """
    if X.ndim == 2:
        dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
        dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
    else:
        dx = X[1] - X[0] if len(X) > 1 else 1.0
        dy = Y[1] - Y[0] if len(Y) > 1 else 1.0

    # np.gradient with signed spacing gives correct physical gradient
    dU_dy = np.gradient(U, dy, axis=0)
    dV_dx = np.gradient(V, dx, axis=1)
    dU_dx = np.gradient(U, dx, axis=1)
    dV_dy = np.gradient(V, dy, axis=0)

    return {
        'dU_dy': dU_dy,
        'dV_dx': dV_dx,
        'dU_dx': dU_dx,
        'dV_dy': dV_dy,
        'dx': dx,
        'dy': dy,
    }


def compute_row_averaged_corrections(data: dict, grads: dict, coords: dict) -> dict:
    """
    Compute gradient corrections and average across each row (constant y).

    Returns row-averaged values for analysis vs y-position.
    """
    sig_A_x = data['sig_A_x']
    sig_A_y = data['sig_A_y']
    UU_stress = data['UU_stress']
    VV_stress = data['VV_stress']
    UV_stress = data['UV_stress']

    dU_dy = grads['dU_dy']
    dV_dx = grads['dV_dx']

    # Gradient corrections (2D fields)
    UU_correction = 0.5 * sig_A_x * (dU_dy ** 2)
    VV_correction = 0.5 * sig_A_y * (dV_dx ** 2)
    UV_correction = 0.5 * sig_A_x * (dU_dy + dV_dx)

    # Corrected stresses
    UU_corrected = UU_stress - UU_correction
    VV_corrected = VV_stress - VV_correction
    UV_corrected = UV_stress - UV_correction if UV_stress is not None else None

    # Get y coordinates (row-wise average)
    Y = coords['Y']
    if Y.ndim == 2:
        y_values = np.nanmean(Y, axis=1)  # Average y per row
    else:
        y_values = Y

    # Row-average everything (axis=1 averages across columns)
    def row_avg(arr):
        if arr is None:
            return None
        return np.nanmean(arr, axis=1)

    # Percentage of correction relative to raw stress
    # Use absolute value of stress for percentage calc to handle sign
    UU_pct = np.abs(UU_correction / np.maximum(np.abs(UU_stress), 1e-10)) * 100
    VV_pct = np.abs(VV_correction / np.maximum(np.abs(VV_stress), 1e-10)) * 100
    UV_pct = np.abs(UV_correction / np.maximum(np.abs(UV_stress), 1e-10)) * 100 if UV_stress is not None else None

    return {
        'y': y_values,
        # Raw stresses (row-averaged)
        'UU_raw': row_avg(UU_stress),
        'VV_raw': row_avg(VV_stress),
        'UV_raw': row_avg(UV_stress),
        # Corrections (row-averaged)
        'UU_correction': row_avg(UU_correction),
        'VV_correction': row_avg(VV_correction),
        'UV_correction': row_avg(UV_correction),
        # Corrected (row-averaged)
        'UU_corrected': row_avg(UU_corrected),
        'VV_corrected': row_avg(VV_corrected),
        'UV_corrected': row_avg(UV_corrected),
        # Percentage (row-averaged)
        'UU_pct': row_avg(UU_pct),
        'VV_pct': row_avg(VV_pct),
        'UV_pct': row_avg(UV_pct),
        # Gradients (row-averaged)
        'dU_dy': row_avg(dU_dy),
        'dV_dx': row_avg(dV_dx),
        # Sigma values (row-averaged)
        'sig_A_x': row_avg(sig_A_x),
        'sig_A_y': row_avg(sig_A_y),
    }


def plot_gradient_corrections(result_path: Path, coord_path: Path,
                              pass_idx: int = -1, save_path: Path = None):
    """
    Create visualization of row-by-row gradient corrections.
    """
    # Load data
    data = load_ensemble_result(result_path, pass_idx)
    coords = load_coordinates(coord_path, data['pass_idx'])
    grads = compute_gradients(data['U'], data['V'], coords['X'], coords['Y'])

    # Compute row-averaged corrections
    row_data = compute_row_averaged_corrections(data, grads, coords)
    y = row_data['y']

    # Create figure with 3 rows, 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'Row-Averaged Gradient Corrections vs Y Position\n'
                 f'Pass {data["pass_idx"]+1}/{data["num_passes"]}, Window={data["window_size"]}',
                 fontsize=14, fontweight='bold')

    # --- UU Stress ---
    ax = axes[0, 0]
    ax.plot(y, row_data['UU_raw'], 'b-', label='UU raw', linewidth=2)
    ax.plot(y, row_data['UU_corrected'], 'g-', label='UU corrected', linewidth=2)
    ax.plot(y, row_data['UU_correction'], 'r--', label='Correction term', linewidth=1.5)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('UU Stress (px²/frame²)')
    ax.set_title('UU Reynolds Stress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    ax = axes[0, 1]
    ax.plot(y, row_data['UU_pct'], 'r-', linewidth=2)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('Correction / |Raw| (%)')
    ax.set_title('UU Gradient Correction Percentage')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- VV Stress ---
    ax = axes[1, 0]
    ax.plot(y, row_data['VV_raw'], 'b-', label='VV raw', linewidth=2)
    ax.plot(y, row_data['VV_corrected'], 'g-', label='VV corrected', linewidth=2)
    ax.plot(y, row_data['VV_correction'], 'r--', label='Correction term', linewidth=1.5)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('VV Stress (px²/frame²)')
    ax.set_title('VV Reynolds Stress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    ax = axes[1, 1]
    ax.plot(y, row_data['VV_pct'], 'r-', linewidth=2)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('Correction / |Raw| (%)')
    ax.set_title('VV Gradient Correction Percentage')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- UV Stress ---
    ax = axes[2, 0]
    if row_data['UV_raw'] is not None:
        ax.plot(y, row_data['UV_raw'], 'b-', label='UV raw', linewidth=2)
        ax.plot(y, row_data['UV_corrected'], 'g-', label='UV corrected', linewidth=2)
        ax.plot(y, row_data['UV_correction'], 'r--', label='Correction term', linewidth=1.5)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'UV stress not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('UV Stress (px²/frame²)')
    ax.set_title('UV Reynolds Stress')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    ax = axes[2, 1]
    if row_data['UV_pct'] is not None:
        ax.plot(y, row_data['UV_pct'], 'r-', linewidth=2)
        # Highlight regions where correction > 50%
        high_correction = row_data['UV_pct'] > 50
        if np.any(high_correction):
            ax.fill_between(y, 0, row_data['UV_pct'], where=high_correction,
                           alpha=0.3, color='red', label='>50% correction')
            ax.legend()
    else:
        ax.text(0.5, 0.5, 'UV stress not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('Correction / |Raw| (%)')
    ax.set_title('UV Gradient Correction Percentage')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close()

    return row_data


def plot_gradient_breakdown(result_path: Path, coord_path: Path,
                            pass_idx: int = -1, save_path: Path = None):
    """
    Additional plot showing the breakdown of what drives the UV correction.

    UV_correction = 0.5 * sig_A_x * (dU/dy + dV/dx)

    This helps identify whether dU/dy or dV/dx dominates.
    """
    # Load data
    data = load_ensemble_result(result_path, pass_idx)
    coords = load_coordinates(coord_path, data['pass_idx'])
    grads = compute_gradients(data['U'], data['V'], coords['X'], coords['Y'])

    # Compute row-averaged values
    row_data = compute_row_averaged_corrections(data, grads, coords)
    y = row_data['y']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gradient Correction Breakdown: What Drives UV Correction?\n'
                 f'Pass {data["pass_idx"]+1}/{data["num_passes"]}',
                 fontsize=14, fontweight='bold')

    # Plot velocity gradients
    ax = axes[0, 0]
    ax.plot(y, row_data['dU_dy'], 'b-', label='dU/dy', linewidth=2)
    ax.plot(y, row_data['dV_dx'], 'r-', label='dV/dx', linewidth=2)
    ax.plot(y, row_data['dU_dy'] + row_data['dV_dx'], 'k--',
            label='dU/dy + dV/dx', linewidth=1.5)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('Gradient (1/frame)')
    ax.set_title('Velocity Gradients')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Plot sig_A_x (particle width)
    ax = axes[0, 1]
    ax.plot(y, row_data['sig_A_x'], 'b-', label='sig_A_x', linewidth=2)
    ax.plot(y, row_data['sig_A_y'], 'r-', label='sig_A_y', linewidth=2)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('Sigma (px²)')
    ax.set_title('Autocorrelation Widths (Particle Size)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot UV stress components
    ax = axes[1, 0]
    ax.plot(y, row_data['UV_raw'], 'b-', label='UV raw', linewidth=2)
    ax.plot(y, row_data['UV_correction'], 'r-', label='UV correction', linewidth=2)
    ax.plot(y, row_data['UV_corrected'], 'g-', label='UV corrected', linewidth=2)
    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('Stress (px²/frame²)')
    ax.set_title('UV Stress: Raw, Correction, Corrected')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Plot percentage with analysis
    ax = axes[1, 1]
    ax.plot(y, row_data['UV_pct'], 'r-', linewidth=2, label='|Correction/Raw| %')
    ax.axhline(y=100, color='k', linestyle='--', linewidth=1, label='100% threshold')
    ax.axhline(y=50, color='orange', linestyle='--', linewidth=1, label='50% threshold')

    # Mark regions where correction dominates (>100%)
    dominates = row_data['UV_pct'] > 100
    if np.any(dominates):
        ax.fill_between(y, 0, row_data['UV_pct'], where=dominates,
                       alpha=0.3, color='red', label='Correction > Raw')

    ax.set_xlabel('Y position (px)')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('UV Correction Magnitude (% of Raw)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close()

    # Print summary statistics
    print("\n" + "="*70)
    print("GRADIENT CORRECTION ANALYSIS SUMMARY")
    print("="*70)

    print(f"\nUV Correction Statistics:")
    print(f"  Mean |correction/raw|: {np.nanmean(row_data['UV_pct']):.1f}%")
    print(f"  Max  |correction/raw|: {np.nanmax(row_data['UV_pct']):.1f}%")
    print(f"  Min  |correction/raw|: {np.nanmin(row_data['UV_pct']):.1f}%")

    if np.any(dominates):
        y_dom = y[dominates]
        print(f"\n  WARNING: Correction DOMINATES (>100%) in y range: "
              f"{y_dom.min():.1f} to {y_dom.max():.1f} px")
        print(f"  This affects {np.sum(dominates)} of {len(y)} rows ({100*np.sum(dominates)/len(y):.1f}%)")

    high_corr = row_data['UV_pct'] > 50
    if np.any(high_corr):
        y_high = y[high_corr]
        print(f"\n  HIGH CORRECTION (>50%) in y range: "
              f"{y_high.min():.1f} to {y_high.max():.1f} px")

    return row_data


def main():
    base_dir = Path("/Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools/tests/rs_particle_d3")

    # Test cases
    test_cases = [
        ("3-pass STD (64->32->16)", "output_3pass"),
        ("3-pass SINGLE sum64", "output_single_sumwin64"),
        ("1-pass SINGLE 32x32", "output_1pass_32_sum64"),
        ("1-pass SINGLE 16x16", "output_1pass_16_sum64"),
    ]

    for name, folder in test_cases:
        result_path = base_dir / folder / "uncalibrated_piv" / "500" / "Cam1" / "ensemble" / "ensemble_result.mat"
        coord_path = base_dir / folder / "uncalibrated_piv" / "500" / "Cam1" / "ensemble" / "coordinates.mat"

        if result_path.exists() and coord_path.exists():
            print(f"\n{'='*70}")
            print(f"Analyzing: {name}")
            print(f"{'='*70}")

            save_path = base_dir / folder / f"gradient_corrections_{folder}.png"
            plot_gradient_corrections(result_path, coord_path, save_path=save_path)

            breakdown_path = base_dir / folder / f"gradient_breakdown_{folder}.png"
            plot_gradient_breakdown(result_path, coord_path, save_path=breakdown_path)
        else:
            print(f"\nSkipping {name}: files not found at {result_path}")


def generate_synthetic_boundary_layer_demo():
    """
    Generate synthetic boundary layer data to demonstrate gradient correction effects.

    Near the wall (low y+):
    - Large velocity gradients (dU/dy is large)
    - Small Reynolds stresses (turbulence suppressed)
    -> Gradient correction can DOMINATE the measured stress

    Far from wall (high y+):
    - Smaller velocity gradients
    - Larger Reynolds stresses
    -> Gradient correction is small percentage
    """
    print("\n" + "="*70)
    print("SYNTHETIC BOUNDARY LAYER DEMONSTRATION")
    print("Showing how gradient corrections dominate at low y+")
    print("="*70)

    # Create synthetic y+ coordinate (wall units)
    y_plus = np.linspace(1, 200, 100)

    # Typical boundary layer mean velocity profile (log law + viscous sublayer)
    # U+ = y+ for y+ < 5, then log law
    U_plus = np.where(y_plus < 5, y_plus,
                      2.5 * np.log(y_plus) + 5.5)

    # Convert to pixels (arbitrary scaling)
    y_px = y_plus * 2  # 2 px per wall unit
    U_px = U_plus * 0.5  # velocity in px/frame

    # Velocity gradient dU/dy (very large near wall, small far from wall)
    # In wall units: dU+/dy+ = 1 for y+ < 5, then 2.5/y+
    dU_dy_plus = np.where(y_plus < 5, 1.0, 2.5 / y_plus)
    dU_dy = dU_dy_plus * 0.25  # Convert to px/frame per px (scaling)

    # V is small (wall-normal), dV/dx is small
    dV_dx = 0.01 * np.random.randn(len(y_plus))

    # Reynolds stresses (typical boundary layer profiles)
    # Near wall: u'v' peaks around y+ ~ 30, then decreases
    # UU_true: peaks around y+ ~ 15
    UU_true = 0.5 * (1 - np.exp(-y_plus/20)) * np.exp(-y_plus/100)
    VV_true = 0.2 * (1 - np.exp(-y_plus/30)) * np.exp(-y_plus/150)
    UV_true = -0.3 * (1 - np.exp(-y_plus/10)) * np.exp(-y_plus/80) * (y_plus/30)

    # Particle autocorrelation width (sig_A_x) - roughly constant
    sig_A_x = 2.5 * np.ones_like(y_plus)  # 2.5 px² typical for ~3px diameter particles
    sig_A_y = 2.5 * np.ones_like(y_plus)

    # Gradient corrections
    UU_correction = 0.5 * sig_A_x * (dU_dy ** 2)
    VV_correction = 0.5 * sig_A_y * (dV_dx ** 2)
    UV_correction = 0.5 * sig_A_x * (dU_dy + dV_dx)

    # "Measured" stresses (true + gradient bias)
    UU_measured = UU_true + UU_correction
    VV_measured = VV_true + VV_correction
    UV_measured = UV_true + UV_correction

    # Corrected (should recover true)
    UU_corrected = UU_measured - UU_correction
    VV_corrected = VV_measured - VV_correction
    UV_corrected = UV_measured - UV_correction

    # Percentage of correction
    UU_pct = np.abs(UU_correction / np.maximum(np.abs(UU_measured), 1e-10)) * 100
    VV_pct = np.abs(VV_correction / np.maximum(np.abs(VV_measured), 1e-10)) * 100
    UV_pct = np.abs(UV_correction / np.maximum(np.abs(UV_measured), 1e-10)) * 100

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Gradient Correction in Boundary Layer: Low y+ Problem\n'
                 '(Synthetic demonstration - typical turbulent BL profiles)',
                 fontsize=14, fontweight='bold')

    # --- UU Stress ---
    ax = axes[0, 0]
    ax.plot(y_plus, UU_measured, 'b-', label='UU measured (with gradient bias)', linewidth=2)
    ax.plot(y_plus, UU_true, 'g--', label='UU true', linewidth=2)
    ax.plot(y_plus, UU_correction, 'r:', label='Gradient correction', linewidth=2)
    ax.set_xlabel('y+')
    ax.set_ylabel('UU Stress')
    ax.set_title('UU Reynolds Stress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)

    ax = axes[0, 1]
    ax.plot(y_plus, UU_pct, 'r-', linewidth=2)
    ax.axhline(y=100, color='k', linestyle='--', alpha=0.5)
    ax.fill_between(y_plus, 0, UU_pct, where=UU_pct > 100, alpha=0.3, color='red')
    ax.set_xlabel('y+')
    ax.set_ylabel('|Correction/Measured| (%)')
    ax.set_title('UU Correction Percentage')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, min(200, np.nanmax(UU_pct)*1.1))

    # --- VV Stress ---
    ax = axes[1, 0]
    ax.plot(y_plus, VV_measured, 'b-', label='VV measured', linewidth=2)
    ax.plot(y_plus, VV_true, 'g--', label='VV true', linewidth=2)
    ax.plot(y_plus, VV_correction, 'r:', label='Gradient correction', linewidth=2)
    ax.set_xlabel('y+')
    ax.set_ylabel('VV Stress')
    ax.set_title('VV Reynolds Stress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)

    ax = axes[1, 1]
    ax.plot(y_plus, VV_pct, 'r-', linewidth=2)
    ax.axhline(y=100, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('y+')
    ax.set_ylabel('|Correction/Measured| (%)')
    ax.set_title('VV Correction Percentage')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)

    # --- UV Stress (the key one you're concerned about) ---
    ax = axes[2, 0]
    ax.plot(y_plus, UV_measured, 'b-', label='UV measured (gradient corrupted)', linewidth=2)
    ax.plot(y_plus, UV_true, 'g--', label='UV true', linewidth=2)
    ax.plot(y_plus, UV_correction, 'r:', label='Gradient correction (0.5*sig_A*dU/dy)', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('y+')
    ax.set_ylabel('UV Stress')
    ax.set_title('UV Reynolds Stress - GRADIENT DOMINATES AT LOW y+')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)

    ax = axes[2, 1]
    ax.plot(y_plus, UV_pct, 'r-', linewidth=2, label='|Correction/Measured|')
    ax.axhline(y=100, color='k', linestyle='--', linewidth=1, label='100% - correction = measured')
    ax.axhline(y=50, color='orange', linestyle='--', linewidth=1, label='50% threshold')
    ax.fill_between(y_plus, 0, UV_pct, where=UV_pct > 100, alpha=0.3, color='red',
                   label='DOMINATED by gradient')
    ax.fill_between(y_plus, 0, UV_pct, where=(UV_pct > 50) & (UV_pct <= 100),
                   alpha=0.2, color='orange')
    ax.set_xlabel('y+')
    ax.set_ylabel('|Correction/Measured| (%)')
    ax.set_title('UV Correction % - PROBLEM ZONE AT LOW y+')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, min(300, np.nanmax(UV_pct)*1.1))

    plt.tight_layout()

    # Save
    save_dir = Path("/Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools/tests")
    save_path = save_dir / "gradient_correction_synthetic_demo.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to {save_path}")

    plt.close()

    # Additional breakdown figure
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('WHY Gradient Correction Dominates UV at Low y+\n'
                  'UV_correction = 0.5 * sig_A_x * (dU/dy + dV/dx)',
                  fontsize=14, fontweight='bold')

    # Mean velocity profile
    ax = axes2[0, 0]
    ax.plot(y_plus, U_plus, 'b-', linewidth=2)
    ax.set_xlabel('y+')
    ax.set_ylabel('U+')
    ax.set_title('Mean Velocity Profile')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)

    # Velocity gradient
    ax = axes2[0, 1]
    ax.plot(y_plus, dU_dy, 'b-', linewidth=2, label='dU/dy (large near wall!)')
    ax.plot(y_plus, dV_dx, 'r-', linewidth=1, alpha=0.5, label='dV/dx (small)')
    ax.set_xlabel('y+')
    ax.set_ylabel('Gradient')
    ax.set_title('Velocity Gradients - dU/dy HUGE at Low y+')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)

    # UV true vs correction
    ax = axes2[1, 0]
    ax.plot(y_plus, np.abs(UV_true), 'g-', linewidth=2, label='|UV true| (small near wall)')
    ax.plot(y_plus, np.abs(UV_correction), 'r-', linewidth=2, label='|UV correction| (large near wall)')
    ax.set_xlabel('y+')
    ax.set_ylabel('|Stress|')
    ax.set_title('The Problem: True UV Small, Correction Large at Low y+')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)

    # Sign issue
    ax = axes2[1, 1]
    ax.plot(y_plus, UV_true, 'g-', linewidth=2, label='UV true (negative)')
    ax.plot(y_plus, UV_correction, 'r-', linewidth=2, label='UV correction (positive!)')
    ax.plot(y_plus, UV_measured, 'b--', linewidth=2, label='UV measured = true + correction')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('y+')
    ax.set_ylabel('Stress')
    ax.set_title('SIGN FLIP: dU/dy > 0 adds POSITIVE bias to negative UV')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)

    plt.tight_layout()

    save_path2 = save_dir / "gradient_correction_breakdown_demo.png"
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved breakdown figure to {save_path2}")

    plt.close()

    # Print analysis
    print("\n" + "="*70)
    print("KEY INSIGHT: Why Gradient Correction Dominates UV at Low y+")
    print("="*70)
    print("""
    The UV correction term is: 0.5 * sig_A_x * (dU/dy + dV/dx)

    At LOW y+ (near wall):
    - dU/dy is VERY LARGE (steep velocity profile)
    - True UV stress is SMALL (turbulence suppressed by viscosity)
    - The correction term >> true stress
    - Result: Measured UV dominated by gradient smearing artifact

    At HIGH y+ (away from wall):
    - dU/dy is small (flat velocity profile)
    - True UV stress is larger
    - Correction << true stress
    - Result: Measured UV is close to true

    CRITICAL PROBLEM:
    - In a boundary layer, UV should be NEGATIVE (momentum flux toward wall)
    - But dU/dy > 0 always (velocity increases away from wall)
    - So gradient correction adds POSITIVE bias
    - At low y+, this can flip the SIGN of measured UV!

    This is why you see unreliable UV measurements near walls in PIV.
    """)

    # Quantify where correction dominates
    low_y_mask = y_plus < 30
    print(f"\nQuantitative Analysis (y+ < 30):")
    print(f"  Mean |UV true|:       {np.mean(np.abs(UV_true[low_y_mask])):.4f}")
    print(f"  Mean |UV correction|: {np.mean(np.abs(UV_correction[low_y_mask])):.4f}")
    print(f"  Mean |UV measured|:   {np.mean(np.abs(UV_measured[low_y_mask])):.4f}")
    print(f"  Correction/True ratio: {np.mean(np.abs(UV_correction[low_y_mask])/np.maximum(np.abs(UV_true[low_y_mask]), 1e-10)):.1f}x")

    print(f"\nQuantitative Analysis (y+ > 100):")
    high_y_mask = y_plus > 100
    print(f"  Mean |UV true|:       {np.mean(np.abs(UV_true[high_y_mask])):.4f}")
    print(f"  Mean |UV correction|: {np.mean(np.abs(UV_correction[high_y_mask])):.4f}")
    print(f"  Correction/True ratio: {np.mean(np.abs(UV_correction[high_y_mask])/np.maximum(np.abs(UV_true[high_y_mask]), 1e-10)):.2f}x")


def analyze_jhtdb_channel_data():
    """Analyze the JHTDB channel flow data with gradient corrections."""
    base_dir = Path("/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/planar_images/rectangle-test")

    result_path = base_dir / "uncalibrated_piv" / "1000" / "Cam1" / "ensemble" / "ensemble_result.mat"
    coord_path = base_dir / "uncalibrated_piv" / "1000" / "Cam1" / "ensemble" / "coordinates.mat"

    if not result_path.exists():
        print(f"Data not found at {result_path}")
        return

    print("\n" + "="*70)
    print("JHTDB CHANNEL FLOW - GRADIENT CORRECTION ANALYSIS")
    print("="*70)

    save_dir = base_dir / "uncalibrated_piv" / "1000" / "Cam1" / "ensemble"

    # Main corrections plot
    row_data = plot_gradient_corrections(result_path, coord_path,
                                        save_path=save_dir / "gradient_corrections.png")

    # Breakdown plot
    plot_gradient_breakdown(result_path, coord_path,
                           save_path=save_dir / "gradient_breakdown.png")

    return row_data


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Run synthetic demonstration
        generate_synthetic_boundary_layer_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "--jhtdb":
        # Analyze JHTDB channel data
        analyze_jhtdb_channel_data()
    else:
        # Default: try JHTDB data first, then fall back to demo
        base_dir = Path("/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/planar_images/rectangle-test")
        result_path = base_dir / "uncalibrated_piv" / "1000" / "Cam1" / "ensemble" / "ensemble_result.mat"

        if result_path.exists():
            analyze_jhtdb_channel_data()
        else:
            print("No JHTDB data found. Running synthetic demonstration...")
            generate_synthetic_boundary_layer_demo()
