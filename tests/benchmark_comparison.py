#!/usr/bin/env python3
"""
Benchmark comparison of PIV results against JHTDB ground truth.

Compares:
- Mean velocity profile U+ vs y+
- Reynolds normal stresses uu+, vv+ vs y+
- Reynolds shear stress uv+ vs y+

Excludes first/last 5mm in x-direction (out-of-plane particle loss).
"""

import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path


def resolve_ground_truth_dir(gt_dir: Path | None, script_dir: Path) -> Path:
    """Resolve a ground-truth directory.

    Accepts either:
    - the ground-truth root directory (containing an `ensemble_statistics/` subdir), or
    - the `ensemble_statistics/` directory itself.
    """
    candidate = (script_dir / 'ground_truth') if gt_dir is None else Path(gt_dir)

    direct = candidate
    nested = candidate / 'ensemble_statistics'

    def has_expected_files(folder: Path) -> bool:
        return (folder / 'wall_units.mat').exists() and (folder / 'profiles.mat').exists()

    if has_expected_files(direct):
        return direct
    if has_expected_files(nested):
        return nested

    raise FileNotFoundError(
        "Could not find ground-truth files. Expected 'wall_units.mat' and 'profiles.mat' in "
        f"either '{direct}' or '{nested}'."
    )


def load_wall_units(wall_units_path):
    """Load wall units from .mat file."""
    wall = sio.loadmat(wall_units_path, squeeze_me=True, struct_as_record=False)
    wu = wall['wall_units']
    return {
        'u_tau': float(wu.u_tau),  # mm/s
        'nu': float(wu.nu),        # mm^2/s
        'delta_nu': float(wu.delta_nu),  # mm
        'h_mm': float(wu.h_mm),    # mm
        'Re_tau': float(wu.Re_tau)
    }


def load_ground_truth(profiles_path):
    """Load ground truth 1px profiles."""
    profiles = sio.loadmat(profiles_path, squeeze_me=True, struct_as_record=False)
    win1px = profiles['profiles'].win_1px
    return {
        'y_mm': win1px.y_mm,
        'y_plus': win1px.y_plus,
        'U': win1px.U,           # mm/s
        'V': win1px.V,           # mm/s
        'uu': win1px.uu,         # (mm/s)^2
        'vv': win1px.vv,         # (mm/s)^2
        'uv': win1px.uv,         # (mm/s)^2
        'U_plus': win1px.U_plus,
        'uu_plus': win1px.uu_plus,
        'vv_plus': win1px.vv_plus,
        'uv_plus': win1px.uv_plus,
    }


def load_piv_statistics(stats_path, run_idx=3):
    """
    Load PIV statistics from mean_stats.mat (instantaneous).

    Parameters
    ----------
    stats_path : Path
        Path to mean_stats.mat
    run_idx : int
        Run index (0-based). run_idx=3 corresponds to run 4 (16x16 window)
    """
    stats = sio.loadmat(stats_path, squeeze_me=True, struct_as_record=False)
    piv = stats['piv_result'][run_idx]
    coords = stats['coordinates'][run_idx]

    return {
        'ux': piv.ux,    # m/s (need to convert to mm/s)
        'uy': piv.uy,    # m/s
        'uu': piv.uu,    # (m/s)^2
        'vv': piv.vv,    # (m/s)^2
        'uv': piv.uv,    # (m/s)^2
        'x': coords.x,   # mm
        'y': coords.y,   # mm
    }


def load_ensemble_statistics(ensemble_path, coords_path, run_idx=3):
    """
    Load PIV statistics from ensemble_result.mat.

    Parameters
    ----------
    ensemble_path : Path
        Path to ensemble_result.mat
    coords_path : Path
        Path to coordinates.mat
    run_idx : int
        Run index (0-based). run_idx=3 corresponds to run 4
    """
    ens = sio.loadmat(ensemble_path, squeeze_me=True, struct_as_record=False)
    coords_data = sio.loadmat(coords_path, squeeze_me=True, struct_as_record=False)

    piv = ens['ensemble_result'][run_idx]
    coords = coords_data['coordinates'][run_idx]

    return {
        'ux': piv.ux,           # m/s
        'uy': piv.uy,           # m/s
        'uu': piv.UU_stress,    # (m/s)^2
        'vv': piv.VV_stress,    # (m/s)^2
        'uv': piv.UV_stress,    # (m/s)^2
        'x': coords.x,          # mm
        'y': coords.y,          # mm
    }


def compute_piv_profiles(piv_data, x_exclude_vectors=4):
    """
    Compute x-averaged PIV profiles, excluding edges.

    Parameters
    ----------
    piv_data : dict
        PIV statistics dictionary
    x_exclude_vectors : int
        Number of vectors to exclude from each side in x-direction

    Returns
    -------
    dict with y_mm, U, uu, vv, uv profiles
    """
    x = piv_data['x']
    y = piv_data['y']

    # Get unique y values (assuming regular grid)
    # The grid is (ny, nx), so y varies along axis 0
    y_unique = y[:, 0]  # First column gives y values
    x_unique = x[0, :]  # First row gives x values
    nx = len(x_unique)

    # Create mask excluding first/last x_exclude_vectors
    x_mask = np.zeros(nx, dtype=bool)
    x_mask[x_exclude_vectors:nx-x_exclude_vectors] = True

    print(f"  X range: {x_unique.min():.2f} to {x_unique.max():.2f} mm")
    print(f"  Excluding {x_exclude_vectors} vectors from each x-edge")
    print(f"  X points: {x_mask.sum()} / {nx}")

    # Convert velocities from m/s to mm/s
    ux_mm = piv_data['ux'] * 1000  # m/s -> mm/s
    uy_mm = piv_data['uy'] * 1000
    uu_mm2 = piv_data['uu'] * 1e6  # (m/s)^2 -> (mm/s)^2
    vv_mm2 = piv_data['vv'] * 1e6
    uv_mm2 = piv_data['uv'] * 1e6

    # Average over valid x range
    U_profile = np.nanmean(ux_mm[:, x_mask], axis=1)
    V_profile = np.nanmean(uy_mm[:, x_mask], axis=1)
    uu_profile = np.nanmean(uu_mm2[:, x_mask], axis=1)
    vv_profile = np.nanmean(vv_mm2[:, x_mask], axis=1)
    uv_profile = np.nanmean(uv_mm2[:, x_mask], axis=1)

    return {
        'y_mm': y_unique,
        'U': U_profile,
        'V': V_profile,
        'uu': uu_profile,
        'vv': vv_profile,
        'uv': uv_profile,
    }


def convert_to_wall_units(profiles, wall_units, y_offset_mm=0.0):
    """
    Convert profiles to wall units (plus units).

    Parameters
    ----------
    profiles : dict
        PIV profiles with y_mm, U, etc.
    wall_units : dict
        Wall unit parameters
    y_offset_mm : float
        Offset to add to y_mm before converting to y+ (for coordinate alignment)
    """
    u_tau = wall_units['u_tau']
    delta_nu = wall_units['delta_nu']
    u_tau2 = u_tau ** 2

    # Apply y offset (to align PIV coordinate system with ground truth)
    y_mm_aligned = profiles['y_mm'] + y_offset_mm

    return {
        'y_mm': y_mm_aligned,
        'y_plus': y_mm_aligned / delta_nu,
        'U_plus': profiles['U'] / u_tau,
        'V_plus': profiles['V'] / u_tau,
        'uu_plus': profiles['uu'] / u_tau2,
        'vv_plus': profiles['vv'] / u_tau2,
        'uv_plus': profiles['uv'] / u_tau2,
    }


def compute_errors(piv_plus, gt_plus, y_plus_range=(10, 500)):
    """
    Compute error metrics between PIV and ground truth.

    Parameters
    ----------
    piv_plus : dict
        PIV profiles in wall units
    gt_plus : dict
        Ground truth profiles in wall units
    y_plus_range : tuple
        y+ range for comparison (exclude near-wall and centerline regions)
    """
    # Interpolate ground truth to PIV y+ locations
    y_piv = piv_plus['y_plus']
    y_gt = gt_plus['y_plus']

    # Only compare in specified y+ range
    mask_piv = (y_piv >= y_plus_range[0]) & (y_piv <= y_plus_range[1])
    y_compare = y_piv[mask_piv]

    if len(y_compare) == 0:
        print(f"  Warning: No PIV points in y+ range {y_plus_range}")
        return {}

    errors = {}
    for var in ['U_plus', 'V_plus', 'uu_plus', 'vv_plus', 'uv_plus']:
        piv_vals = piv_plus[var][mask_piv]

        # Interpolate ground truth
        gt_interp = interp1d(y_gt, gt_plus[var], kind='linear',
                            bounds_error=False, fill_value=np.nan)
        gt_vals = gt_interp(y_compare)

        # Remove NaN values
        valid = ~np.isnan(piv_vals) & ~np.isnan(gt_vals)
        if valid.sum() == 0:
            continue

        piv_valid = piv_vals[valid]
        gt_valid = gt_vals[valid]

        # Compute metrics
        diff = piv_valid - gt_valid
        rms_error = np.sqrt(np.mean(diff**2))
        mean_abs_error = np.mean(np.abs(diff))

        # Relative RMS error (as percentage of GT range)
        gt_range = np.ptp(gt_valid)  # peak-to-peak
        rms_rel = (rms_error / gt_range * 100) if gt_range > 0 else np.nan

        # Correlation coefficient
        corr = np.corrcoef(piv_valid, gt_valid)[0, 1]

        # R-squared
        ss_res = np.sum(diff**2)
        ss_tot = np.sum((gt_valid - gt_valid.mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        errors[var] = {
            'rms': rms_error,
            'rms_rel': rms_rel,
            'mae': mean_abs_error,
            'corr': corr,
            'r2': r2,
            'n_points': valid.sum(),
        }

    return errors


def plot_comparison(piv_plus, gt_plus, wall_units, errors, output_dir, window_label='16x16'):
    """Generate comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Re_tau = wall_units['Re_tau']

    # Figure 1: Mean velocity profile (semi-log)
    fig, ax = plt.subplots(figsize=(10, 7))

    # Ground truth
    ax.semilogx(gt_plus['y_plus'], gt_plus['U_plus'], 'k-',
                linewidth=2, label='DNS (1px)', zorder=3)

    # PIV
    ax.semilogx(piv_plus['y_plus'], piv_plus['U_plus'], 'ro',
                markersize=4, alpha=0.7, label=f'PIV ({window_label})', zorder=2)

    # Log law reference
    y_log = np.logspace(1, np.log10(Re_tau), 100)
    kappa, B = 0.41, 5.2
    U_log = (1/kappa) * np.log(y_log) + B
    ax.semilogx(y_log, U_log, 'b--', linewidth=1, alpha=0.7,
                label=r'Log law: $U^+ = \frac{1}{\kappa}\ln(y^+) + B$')

    # Viscous sublayer
    y_visc = np.linspace(0.1, 10, 50)
    ax.semilogx(y_visc, y_visc, 'g--', linewidth=1, alpha=0.7,
                label=r'Viscous sublayer: $U^+ = y^+$')

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$U^+$', fontsize=14)
    ax.set_title(f'Mean Velocity Profile (Re$_\\tau$ = {Re_tau:.0f})', fontsize=16)
    ax.legend(fontsize=11)
    ax.set_xlim(1, Re_tau)
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)

    if 'U_plus' in errors:
        ax.text(0.02, 0.98, f"R² = {errors['U_plus']['r2']:.4f}\n"
                           f"RMS = {errors['U_plus']['rms_rel']:.1f}%",
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_dir / 'U_plus_profile.png', dpi=150)
    plt.close(fig)

    # Figure 2: Reynolds stresses (semi-log)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # uu+
    ax = axes[0]
    ax.semilogx(gt_plus['y_plus'], gt_plus['uu_plus'], 'k-', linewidth=2, label='DNS')
    ax.semilogx(piv_plus['y_plus'], piv_plus['uu_plus'], 'ro', markersize=4,
                alpha=0.7, label='PIV')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{u'u'}^+$", fontsize=12)
    ax.set_title('Streamwise Normal Stress', fontsize=14)
    ax.legend()
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)
    if 'uu_plus' in errors:
        ax.text(0.98, 0.98, f"R² = {errors['uu_plus']['r2']:.4f}",
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # vv+
    ax = axes[1]
    ax.semilogx(gt_plus['y_plus'], gt_plus['vv_plus'], 'k-', linewidth=2, label='DNS')
    ax.semilogx(piv_plus['y_plus'], piv_plus['vv_plus'], 'ro', markersize=4,
                alpha=0.7, label='PIV')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{v'v'}^+$", fontsize=12)
    ax.set_title('Wall-Normal Normal Stress', fontsize=14)
    ax.legend()
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)
    if 'vv_plus' in errors:
        ax.text(0.98, 0.98, f"R² = {errors['vv_plus']['r2']:.4f}",
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # -uv+
    ax = axes[2]
    ax.semilogx(gt_plus['y_plus'], -gt_plus['uv_plus'], 'k-', linewidth=2, label='DNS')
    ax.semilogx(piv_plus['y_plus'], -piv_plus['uv_plus'], 'ro', markersize=4,
                alpha=0.7, label='PIV')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$-\overline{u'v'}^+$", fontsize=12)
    ax.set_title('Reynolds Shear Stress', fontsize=14)
    ax.legend()
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)
    if 'uv_plus' in errors:
        ax.text(0.98, 0.98, f"R² = {errors['uv_plus']['r2']:.4f}",
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_dir / 'reynolds_stresses.png', dpi=150)
    plt.close(fig)

    # Figure 3: V+ profile (wall-normal mean velocity)
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(gt_plus['y_plus'], gt_plus['V_plus'], 'k-', linewidth=2, label='DNS')
    ax.plot(piv_plus['y_plus'], piv_plus['V_plus'], 'ro', markersize=4,
            alpha=0.7, label='PIV')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$V^+$', fontsize=14)
    ax.set_title(f'Mean Wall-Normal Velocity Profile (Re$_\\tau$ = {Re_tau:.0f})', fontsize=16)
    ax.legend(fontsize=11)
    ax.set_xlim(0, Re_tau)
    ax.grid(True, alpha=0.3)

    if 'V_plus' in errors:
        ax.text(0.02, 0.98, f"R² = {errors['V_plus']['r2']:.4f}\n"
                           f"Corr = {errors['V_plus']['corr']:.4f}",
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_dir / 'V_plus_profile.png', dpi=150)
    plt.close(fig)

    # Figure 4: All profiles on linear scale
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(gt_plus['y_plus'], gt_plus['U_plus'], 'k-', linewidth=2, label='DNS U+')
    ax.plot(piv_plus['y_plus'], piv_plus['U_plus'], 'ko', markersize=3,
            alpha=0.5, label='PIV U+')

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$U^+$', fontsize=14)
    ax.set_title('Mean Velocity Profile (Linear Scale)', fontsize=16)
    ax.legend(fontsize=11)
    ax.set_xlim(0, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'U_plus_linear.png', dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to: {output_dir}")


def main(mode='instantaneous', gt_dir=None, base_dir=None):
    """Main benchmark comparison function.

    Parameters
    ----------
    mode : str
        'instantaneous' or 'ensemble'
    """
    # Paths
    script_dir = Path(__file__).parent
    gt_dir = resolve_ground_truth_dir(gt_dir, script_dir)
    if base_dir is None:
        base_dir = Path('/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/'
                        'Documents/#current_processing/query_JHTDB/download_from_jhtdb/'
                        'bottom_channel/planar_images/window_validation')
    else:
        base_dir = Path(base_dir)

    if mode == 'ensemble':
        ensemble_path = base_dir / 'calibrated_piv/1000/Cam1/ensemble/ensemble_result.mat'
        coords_path = base_dir / 'calibrated_piv/1000/Cam1/ensemble/coordinates.mat'
        output_dir = script_dir / 'benchmark_results_ensemble'
    else:
        stats_path = base_dir / 'statistics/1000/Cam1/instantaneous/mean_stats/mean_stats.mat'
        output_dir = script_dir / 'benchmark_results'

    print("=" * 70)
    print(f"PIV BENCHMARK COMPARISON ({mode.upper()})")
    print("=" * 70)

    # Load data
    print("\n[1] Loading wall units...")
    wall_units = load_wall_units(gt_dir / 'wall_units.mat')
    print(f"  u_tau = {wall_units['u_tau']:.4f} mm/s")
    print(f"  nu = {wall_units['nu']:.4f} mm²/s")
    print(f"  delta_nu = {wall_units['delta_nu']:.4f} mm")
    print(f"  Re_tau = {wall_units['Re_tau']:.0f}")

    print("\n[2] Loading ground truth (1px window)...")
    gt = load_ground_truth(gt_dir / 'profiles.mat')
    print(f"  y+ range: {gt['y_plus'].min():.1f} to {gt['y_plus'].max():.1f}")
    print(f"  U range: {gt['U'].min():.2f} to {gt['U'].max():.2f} mm/s")

    print(f"\n[3] Loading PIV statistics ({mode}, run 4)...")
    if mode == 'ensemble':
        piv = load_ensemble_statistics(ensemble_path, coords_path, run_idx=3)
    else:
        piv = load_piv_statistics(stats_path, run_idx=3)
    print(f"  Grid size: {piv['ux'].shape}")
    print(f"  ux range: {np.nanmin(piv['ux'])*1000:.2f} to {np.nanmax(piv['ux'])*1000:.2f} mm/s")

    print("\n[4] Computing x-averaged PIV profiles...")
    piv_profiles = compute_piv_profiles(piv, x_exclude_vectors=4)
    print(f"  y range: {piv_profiles['y_mm'].min():.2f} to {piv_profiles['y_mm'].max():.2f} mm")
    print(f"  U range: {np.nanmin(piv_profiles['U']):.2f} to {np.nanmax(piv_profiles['U']):.2f} mm/s")

    print("\n[5] Converting to wall units...")
    # Calculate y-offset to align PIV coordinate system with ground truth
    # Ground truth has y=0 at the wall, PIV may have an offset
    y_offset_mm = -piv_profiles['y_mm'].min()  # Shift so y_min = 0
    print(f"  Applying y-offset: {y_offset_mm:.2f} mm (aligning y_min to wall)")

    piv_plus = convert_to_wall_units(piv_profiles, wall_units, y_offset_mm=y_offset_mm)
    print(f"  Aligned y range: {piv_plus['y_mm'].min():.2f} to {piv_plus['y_mm'].max():.2f} mm")
    print(f"  y+ range: {piv_plus['y_plus'].min():.1f} to {piv_plus['y_plus'].max():.1f}")
    print(f"  U+ range: {np.nanmin(piv_plus['U_plus']):.2f} to {np.nanmax(piv_plus['U_plus']):.2f}")

    # Ground truth - convert V to wall units and include pre-computed values
    gt_plus = {
        'y_plus': gt['y_plus'],
        'U_plus': gt['U_plus'],
        'V_plus': gt['V'] / wall_units['u_tau'],  # Convert V to wall units
        'uu_plus': gt['uu_plus'],
        'vv_plus': gt['vv_plus'],
        'uv_plus': gt['uv_plus'],
    }

    # Verify V sign convention matches (should be correct after save_results.py fix)
    # Sample at mid-channel to check sign
    y_mid_idx = len(piv_plus['y_plus']) // 4  # ~25% from wall
    piv_v_sample = piv_plus['V_plus'][y_mid_idx]
    gt_v_idx = np.argmin(np.abs(gt_plus['y_plus'] - piv_plus['y_plus'][y_mid_idx]))
    gt_v_sample = gt['V'][gt_v_idx] / wall_units['u_tau']

    print(f"\n  Sign check at y+ ≈ {piv_plus['y_plus'][y_mid_idx]:.0f}:")
    print(f"    PIV V+ = {piv_v_sample:+.4f}")
    print(f"    DNS V+ = {gt_v_sample:+.4f}")

    v_sign_match = np.sign(piv_v_sample) == np.sign(gt_v_sample) or abs(gt_v_sample) < 0.01
    if v_sign_match:
        print("    => V sign MATCHES ✓ (no flip needed)")
    else:
        print("    => V sign MISMATCH ✗ (PIV pipeline may still have sign issue)")

    # No manual flipping - the save_results.py fix should handle this
    # If signs still don't match, it indicates the fix didn't work correctly

    print("\n[6] Computing error metrics (y+ = 10-500)...")
    errors = compute_errors(piv_plus, gt_plus, y_plus_range=(10, 500))

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    for var, err in errors.items():
        var_name = {
            'U_plus': 'Mean Streamwise Velocity (U+)',
            'V_plus': 'Mean Wall-normal Velocity (V+)',
            'uu_plus': 'Streamwise Stress (uu+)',
            'vv_plus': 'Wall-normal Stress (vv+)',
            'uv_plus': 'Shear Stress (uv+)',
        }.get(var, var)

        print(f"\n{var_name}:")
        print(f"  RMS Error: {err['rms']:.4f} ({err['rms_rel']:.1f}% of range)")
        print(f"  MAE: {err['mae']:.4f}")
        print(f"  R²: {err['r2']:.4f}")
        print(f"  Correlation: {err['corr']:.4f}")
        print(f"  Points compared: {err['n_points']}")

    print("\n[7] Generating plots...")
    plot_comparison(piv_plus, gt_plus, wall_units, errors, output_dir)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


def plot_combined_comparison(all_results, gt_plus, wall_units, output_dir):
    """
    Generate combined comparison plots with all window sizes on one figure.

    Parameters
    ----------
    all_results : list of dict
        List of dicts with keys: 'piv_plus', 'errors', 'window_label', 'window_size'
    gt_plus : dict
        Ground truth profiles in wall units
    wall_units : dict
        Wall unit parameters
    output_dir : Path
        Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Re_tau = wall_units['Re_tau']

    # Color/marker cycle for different window sizes
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    # ==========================================================================
    # Figure 1: Mean velocity profile (semi-log) - ALL WINDOWS
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    # Ground truth
    ax.semilogx(gt_plus['y_plus'], gt_plus['U_plus'], 'k-',
                linewidth=2.5, label='DNS (1px)', zorder=10)

    # PIV results for each window
    for i, res in enumerate(all_results):
        piv_plus = res['piv_plus']
        label = res['window_label']
        ax.semilogx(piv_plus['y_plus'], piv_plus['U_plus'],
                    color=colors[i % len(colors)], marker=markers[i % len(markers)],
                    markersize=4, alpha=0.7, linestyle='none',
                    label=f'PIV ({label})', zorder=5-i*0.1)

    # Log law reference
    y_log = np.logspace(1, np.log10(Re_tau), 100)
    kappa, B = 0.41, 5.2
    U_log = (1/kappa) * np.log(y_log) + B
    ax.semilogx(y_log, U_log, 'b--', linewidth=1.5, alpha=0.5,
                label=r'Log law: $U^+ = \frac{1}{\kappa}\ln(y^+) + B$')

    # Viscous sublayer
    y_visc = np.linspace(0.1, 10, 50)
    ax.semilogx(y_visc, y_visc, 'g--', linewidth=1.5, alpha=0.5,
                label=r'Viscous sublayer: $U^+ = y^+$')

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$U^+$', fontsize=14)
    ax.set_title(f'Mean Velocity Profile - All Window Sizes (Re$_\\tau$ = {Re_tau:.0f})', fontsize=16)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(1, Re_tau)
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'U_plus_profile_combined.png', dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Figure 2: Reynolds stresses - ALL WINDOWS
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # uu+
    ax = axes[0]
    ax.plot(gt_plus['y_plus'], gt_plus['uu_plus'], 'k-', linewidth=2.5, label='DNS', zorder=10)
    for i, res in enumerate(all_results):
        piv_plus = res['piv_plus']
        label = res['window_label']
        ax.plot(piv_plus['y_plus'], piv_plus['uu_plus'],
                color=colors[i % len(colors)], marker=markers[i % len(markers)],
                markersize=3, alpha=0.7, linestyle='none', label=f'PIV ({label})')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{u'u'}^+$", fontsize=12)
    ax.set_title('Streamwise Normal Stress', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim(0, Re_tau)
    ax.grid(True, alpha=0.3)

    # vv+
    ax = axes[1]
    ax.plot(gt_plus['y_plus'], gt_plus['vv_plus'], 'k-', linewidth=2.5, label='DNS', zorder=10)
    for i, res in enumerate(all_results):
        piv_plus = res['piv_plus']
        label = res['window_label']
        ax.plot(piv_plus['y_plus'], piv_plus['vv_plus'],
                color=colors[i % len(colors)], marker=markers[i % len(markers)],
                markersize=3, alpha=0.7, linestyle='none', label=f'PIV ({label})')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{v'v'}^+$", fontsize=12)
    ax.set_title('Wall-Normal Normal Stress', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim(0, Re_tau)
    ax.grid(True, alpha=0.3)

    # -uv+
    ax = axes[2]
    ax.plot(gt_plus['y_plus'], -gt_plus['uv_plus'], 'k-', linewidth=2.5, label='DNS', zorder=10)
    for i, res in enumerate(all_results):
        piv_plus = res['piv_plus']
        label = res['window_label']
        ax.plot(piv_plus['y_plus'], -piv_plus['uv_plus'],
                color=colors[i % len(colors)], marker=markers[i % len(markers)],
                markersize=3, alpha=0.7, linestyle='none', label=f'PIV ({label})')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$-\overline{u'v'}^+$", fontsize=12)
    ax.set_title('Reynolds Shear Stress', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim(0, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'reynolds_stresses_combined.png', dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Figure 3: V+ profile - ALL WINDOWS
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(gt_plus['y_plus'], gt_plus['V_plus'], 'k-', linewidth=2.5, label='DNS', zorder=10)
    for i, res in enumerate(all_results):
        piv_plus = res['piv_plus']
        label = res['window_label']
        ax.plot(piv_plus['y_plus'], piv_plus['V_plus'],
                color=colors[i % len(colors)], marker=markers[i % len(markers)],
                markersize=4, alpha=0.7, linestyle='none', label=f'PIV ({label})')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$V^+$', fontsize=14)
    ax.set_title(f'Mean Wall-Normal Velocity Profile - All Window Sizes (Re$_\\tau$ = {Re_tau:.0f})', fontsize=16)
    ax.legend(fontsize=10)
    ax.set_xlim(0, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'V_plus_profile_combined.png', dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Figure 4: U+ linear scale - ALL WINDOWS
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(gt_plus['y_plus'], gt_plus['U_plus'], 'k-', linewidth=2.5, label='DNS U+', zorder=10)
    for i, res in enumerate(all_results):
        piv_plus = res['piv_plus']
        label = res['window_label']
        ax.plot(piv_plus['y_plus'], piv_plus['U_plus'],
                color=colors[i % len(colors)], marker=markers[i % len(markers)],
                markersize=3, alpha=0.6, linestyle='none', label=f'PIV ({label})')

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$U^+$', fontsize=14)
    ax.set_title('Mean Velocity Profile (Linear Scale) - All Window Sizes', fontsize=16)
    ax.legend(fontsize=10)
    ax.set_xlim(0, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'U_plus_linear_combined.png', dpi=150)
    plt.close(fig)

    print(f"\nCombined plots saved to: {output_dir}")


def main_multi_run(mode='ensemble', run_indices=None, window_sizes=None, run_labels=None, gt_dir=None, base_dir=None, y_plus_offset=0.0):
    """
    Main benchmark comparison function for multiple runs/window sizes.

    Parameters
    ----------
    mode : str
        'instantaneous' or 'ensemble'
    run_indices : list of int
        List of run indices (0-based) to process
    window_sizes : list of int
        Corresponding window sizes for labels (e.g., [16, 8, 6, 4])
    run_labels : list of str, optional
        Custom labels for output folders (e.g., ['run_1', 'run_2', 'run_3'])
    gt_dir : Path, optional
        Ground truth directory. Defaults to script_dir/ground_truth/ensemble_statistics
    base_dir : Path, optional
        Base directory containing PIV results. Defaults to window_validation folder.
    y_plus_offset : float, optional
        Offset to add to y+ coordinates (for calibration correction)
    """
    # Paths
    script_dir = Path(__file__).parent
    gt_dir = resolve_ground_truth_dir(gt_dir, script_dir)
    # Alternative: 2-pass ground truth
    # gt_dir = Path('/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/'
    #               'Documents/#current_processing/query_JHTDB/download_from_jhtdb/'
    #               'ensemble_statistics_2pass')
    if base_dir is None:
        base_dir = Path('/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/'
                        'Documents/#current_processing/query_JHTDB/download_from_jhtdb/'
                        'bottom_channel/planar_images/window_validation')

    if mode == 'ensemble':
        ensemble_path = base_dir / 'calibrated_piv/1000/Cam1/ensemble/ensemble_result.mat'
        coords_path = base_dir / 'calibrated_piv/1000/Cam1/ensemble/coordinates.mat'
        output_dir = script_dir / 'benchmark_results_ensemble'
    else:
        stats_path = base_dir / 'statistics/1000/Cam1/instantaneous/mean_stats/mean_stats.mat'
        output_dir = script_dir / 'benchmark_results'

    print("=" * 70)
    print(f"PIV BENCHMARK COMPARISON ({mode.upper()}) - MULTI-RUN")
    print("=" * 70)
    print(f"Processing runs: {run_indices}")
    print(f"Window sizes: {window_sizes}")

    # Load common data
    print("\n[1] Loading wall units...")
    wall_units = load_wall_units(gt_dir / 'wall_units.mat')
    print(f"  u_tau = {wall_units['u_tau']:.4f} mm/s")
    print(f"  nu = {wall_units['nu']:.4f} mm²/s")
    print(f"  delta_nu = {wall_units['delta_nu']:.4f} mm")
    print(f"  Re_tau = {wall_units['Re_tau']:.0f}")

    print("\n[2] Loading ground truth (1px window)...")
    gt = load_ground_truth(gt_dir / 'profiles.mat')
    print(f"  y+ range: {gt['y_plus'].min():.1f} to {gt['y_plus'].max():.1f}")

    # Ground truth in wall units
    gt_plus = {
        'y_plus': gt['y_plus'],
        'U_plus': gt['U_plus'],
        'V_plus': gt['V'] / wall_units['u_tau'],
        'uu_plus': gt['uu_plus'],
        'vv_plus': gt['vv_plus'],
        'uv_plus': gt['uv_plus'],
    }

    # Process each run
    all_results = []
    if run_labels is None:
        run_labels = [f'{ws}x{ws}' for ws in window_sizes]
    for i, (run_idx, win_size) in enumerate(zip(run_indices, window_sizes)):
        window_label = f'{win_size}x{win_size}'
        run_output_dir = output_dir / run_labels[i]

        print(f"\n{'='*70}")
        print(f"Processing Run {run_idx+1} (Window: {window_label})")
        print('='*70)

        try:
            if mode == 'ensemble':
                piv = load_ensemble_statistics(ensemble_path, coords_path, run_idx=run_idx)
            else:
                piv = load_piv_statistics(stats_path, run_idx=run_idx)

            print(f"  Grid size: {piv['ux'].shape}")
            print(f"  ux range: {np.nanmin(piv['ux'])*1000:.2f} to {np.nanmax(piv['ux'])*1000:.2f} mm/s")

            # Compute profiles
            piv_profiles = compute_piv_profiles(piv, x_exclude_vectors=4)
            print(f"  y range: {piv_profiles['y_mm'].min():.2f} to {piv_profiles['y_mm'].max():.2f} mm")

            # Convert to wall units
            y_offset_mm = -piv_profiles['y_mm'].min()
            piv_plus = convert_to_wall_units(piv_profiles, wall_units, y_offset_mm=y_offset_mm)
            # Apply y+ offset if specified
            if y_plus_offset != 0.0:
                piv_plus['y_plus'] = piv_plus['y_plus'] + y_plus_offset
                print(f"  y+ offset applied: {y_plus_offset:+.1f}")
            print(f"  y+ range: {piv_plus['y_plus'].min():.1f} to {piv_plus['y_plus'].max():.1f}")

            # Compute errors
            errors = compute_errors(piv_plus, gt_plus, y_plus_range=(10, 500))

            # Print error summary
            if 'U_plus' in errors:
                print(f"  U+ R² = {errors['U_plus']['r2']:.4f}, RMS = {errors['U_plus']['rms_rel']:.1f}%")
            if 'uu_plus' in errors:
                print(f"  uu+ R² = {errors['uu_plus']['r2']:.4f}")

            # Generate individual plots
            plot_comparison(piv_plus, gt_plus, wall_units, errors, run_output_dir,
                          window_label=window_label)

            # Store for combined plot
            all_results.append({
                'piv_plus': piv_plus,
                'errors': errors,
                'window_label': window_label,
                'window_size': win_size,
            })

        except Exception as e:
            print(f"  ERROR processing run {run_idx}: {e}")
            import traceback
            traceback.print_exc()

    # Generate combined plots if we have multiple results
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("Generating combined comparison plots...")
        print('='*70)
        plot_combined_comparison(all_results, gt_plus, wall_units, output_dir)

    # Print final summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\n{'Window':<12} {'U+ R²':<10} {'U+ RMS%':<10} {'uu+ R²':<10} {'vv+ R²':<10} {'-uv+ R²':<10}")
    print("-" * 62)
    for res in all_results:
        errs = res['errors']
        u_r2 = errs.get('U_plus', {}).get('r2', np.nan)
        u_rms = errs.get('U_plus', {}).get('rms_rel', np.nan)
        uu_r2 = errs.get('uu_plus', {}).get('r2', np.nan)
        vv_r2 = errs.get('vv_plus', {}).get('r2', np.nan)
        uv_r2 = errs.get('uv_plus', {}).get('r2', np.nan)
        print(f"{res['window_label']:<12} {u_r2:<10.4f} {u_rms:<10.1f} {uu_r2:<10.4f} {vv_r2:<10.4f} {uv_r2:<10.4f}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PIV Benchmark Comparison')
    parser.add_argument('--mode', '-m', choices=['instantaneous', 'ensemble'],
                        default='instantaneous', help='PIV mode (default: instantaneous)')
    parser.add_argument('--runs', '-r', type=str, default=None,
                        help='Comma-separated run indices (0-based), e.g., "0,1,2"')
    parser.add_argument('--windows', '-w', type=str, default=None,
                        help='Comma-separated window sizes for labels, e.g., "32,8,8"')
    parser.add_argument('--labels', '-l', type=str, default=None,
                        help='Comma-separated output folder labels, e.g., "run_1,run_2,run_3"')
    parser.add_argument('--gt-dir', '-g', type=str, default=None,
                        help='Ground truth directory path (defaults to tests/ground_truth; may also be tests/ground_truth/ensemble_statistics)')
    parser.add_argument('--base-dir', '-b', type=str, default=None,
                        help='Base directory containing PIV results')
    parser.add_argument('--y-plus-offset', '-y', type=float, default=0.0,
                        help='Offset to add to y+ coordinates (calibration correction)')
    args = parser.parse_args()

    if args.runs and args.windows:
        run_indices = [int(r) for r in args.runs.split(',')]
        window_sizes = [int(w) for w in args.windows.split(',')]
        run_labels = args.labels.split(',') if args.labels else None
        gt_dir = Path(args.gt_dir) if args.gt_dir else None
        base_dir = Path(args.base_dir) if args.base_dir else None
        main_multi_run(mode=args.mode, run_indices=run_indices, window_sizes=window_sizes,
                       run_labels=run_labels, gt_dir=gt_dir, base_dir=base_dir,
                       y_plus_offset=args.y_plus_offset)
    else:
        gt_dir = Path(args.gt_dir) if args.gt_dir else None
        base_dir = Path(args.base_dir) if args.base_dir else None
        main(mode=args.mode, gt_dir=gt_dir, base_dir=base_dir)
