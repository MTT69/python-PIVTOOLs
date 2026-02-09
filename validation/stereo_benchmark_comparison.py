#!/usr/bin/env python3
"""
Stereo PIV Benchmark Comparison against JHTDB DNS Ground Truth.

Compares 3-component velocity (U, V, W) and all 6 Reynolds stresses
(uu, vv, ww, uv, uw, vw) against DNS channel flow data.

Usage:
    python stereo_benchmark_comparison.py [--run RUN_IDX] [--x-min X_MIN] [--x-max X_MAX]
"""

import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_wall_units(wall_units_path):
    """Load wall units from .mat file (supports both v5 and v7.3/HDF5)."""
    wall_units_path = str(wall_units_path)
    try:
        wall = sio.loadmat(wall_units_path, squeeze_me=True, struct_as_record=False)
        wu = wall['wall_units']
        return {
            'u_tau': float(wu.u_tau),
            'nu': float(wu.nu),
            'delta_nu': float(wu.delta_nu),
            'h_mm': float(wu.h_mm),
            'Re_tau': float(wu.Re_tau)
        }
    except NotImplementedError:
        import h5py
        with h5py.File(wall_units_path, 'r') as f:
            d = f['diagnostics']
            u_tau = float(d['u_tau'][0, 0])
            Re_tau = float(d['Re_tau'][0, 0])
            delta_nu = float(d['delta_nu'][0, 0])
            return {
                'u_tau': u_tau,
                'nu': u_tau * delta_nu,  # nu = u_tau * delta_nu
                'delta_nu': delta_nu,
                'h_mm': Re_tau * delta_nu,  # h = Re_tau * delta_nu
                'Re_tau': Re_tau
            }


def load_ground_truth_3d(profiles_path):
    """Load ground truth 1px profiles including W component (supports both v5 and v7.3/HDF5)."""
    profiles_path = str(profiles_path)
    try:
        profiles = sio.loadmat(profiles_path, squeeze_me=True, struct_as_record=False)
        win1px = profiles['profiles'].win_1px
        return {
            'y_mm': win1px.y_mm,
            'y_plus': win1px.y_plus,
            'U': win1px.U,
            'V': win1px.V,
            'W': win1px.W,
            'uu': win1px.uu,
            'vv': win1px.vv,
            'ww': win1px.ww,
            'uv': win1px.uv,
            'uw': win1px.uw,
            'vw': win1px.vw,
            'U_plus': win1px.U_plus,
            'uu_plus': win1px.uu_plus,
            'vv_plus': win1px.vv_plus,
            'ww_plus': win1px.ww_plus,
            'uv_plus': win1px.uv_plus,
        }
    except NotImplementedError:
        import h5py
        with h5py.File(profiles_path, 'r') as f:
            rp = f['ref_profile']
            y_plus = rp['y_plus'][0, :]
            U = rp['U'][0, :]
            V = rp['V'][0, :]
            W = rp['W'][0, :]

            # Also load ensemble_stats for stress profiles (win_idx=0 = 16x16)
            es = f['ensemble_stats']

            def deref(field, idx=0):
                ref = es[field][idx, 0]
                return f[ref][:].flatten()

            y_mm_es = deref('y_mm')
            y_plus_es = deref('y_plus')

            # Wall units from diagnostics sibling file
            diag_path = str(Path(profiles_path).parent / 'diagnostics.mat')
            wu = load_wall_units(diag_path)
            u_tau = wu['u_tau']
            u_tau2 = u_tau ** 2

            uu = deref('uu_profile')
            vv = deref('vv_profile')
            ww = deref('ww_profile')
            uv = deref('uv_profile')
            uw = deref('uw_profile')
            vw = deref('vw_profile')

            return {
                'y_mm': y_mm_es,
                'y_plus': y_plus_es,
                'U': deref('U_profile'),
                'V': deref('V_profile'),
                'W': deref('W_profile'),
                'uu': uu,
                'vv': vv,
                'ww': ww,
                'uv': uv,
                'uw': uw,
                'vw': vw,
                'U_plus': deref('U_profile') / u_tau,
                'uu_plus': uu / u_tau2,
                'vv_plus': vv / u_tau2,
                'ww_plus': ww / u_tau2,
                'uv_plus': uv / u_tau2,
            }


def load_stereo_statistics(stats_path, coords_path, run_idx=3):
    """
    Load stereo PIV statistics from mean_stats.mat and coordinates from separate file.

    Parameters
    ----------
    stats_path : Path
        Path to mean_stats.mat
    coords_path : Path
        Path to coordinates.mat (from stereo_calibrated folder)
    run_idx : int
        Run index (0-based). run_idx=3 is typically finest resolution (16x16)
    """
    stats = sio.loadmat(stats_path, squeeze_me=True, struct_as_record=False)
    coords_data = sio.loadmat(coords_path, squeeze_me=True, struct_as_record=False)

    # Get piv_result for the requested run
    piv_result = stats['piv_result']
    if isinstance(piv_result, np.ndarray) and piv_result.ndim == 0:
        piv = piv_result.item()
    elif hasattr(piv_result, '__len__') and len(piv_result) > run_idx:
        piv = piv_result[run_idx]
    else:
        piv = piv_result

    # Get coordinates from separate file
    coords = coords_data['coordinates']
    if isinstance(coords, np.ndarray) and coords.ndim == 0:
        coord = coords.item()
    elif hasattr(coords, '__len__') and len(coords) > run_idx:
        coord = coords[run_idx]
    else:
        coord = coords

    return {
        # Velocities (m/s -> mm/s)
        'ux': piv.ux * 1000,
        'uy': piv.uy * 1000,
        'uz': piv.uz * 1000,
        # Normal stresses ((m/s)^2 -> (mm/s)^2)
        'uu': piv.uu * 1e6,
        'vv': piv.vv * 1e6,
        'ww': piv.ww * 1e6,
        # Shear stresses
        'uv': piv.uv * 1e6,
        'uw': piv.uw * 1e6,
        'vw': piv.vw * 1e6,
        # Coordinates (already in mm from calibrated file)
        'x': coord.x,
        'y': coord.y,
    }


def compute_stereo_profiles(piv_data, x_min=5.0, x_max=145.0):
    """
    Compute x-averaged stereo PIV profiles.

    Parameters
    ----------
    piv_data : dict
        Stereo PIV data dictionary
    x_min : float
        Minimum x to include (mm)
    x_max : float
        Maximum x to include (mm)

    Returns
    -------
    dict with y_mm and all velocity/stress profiles
    """
    x = piv_data['x']
    y = piv_data['y']

    # Stereo coordinates have NaN at edges (outside overlap region)
    # Find valid region
    valid_mask = ~np.isnan(x)
    valid_rows = np.any(valid_mask, axis=1)
    valid_cols = np.any(valid_mask, axis=0)

    # Find first valid column to get y values from
    first_valid_col = np.argmax(valid_cols)
    last_valid_col = len(valid_cols) - np.argmax(valid_cols[::-1]) - 1
    mid_col = (first_valid_col + last_valid_col) // 2

    # Find first valid row to get x values from
    first_valid_row = np.argmax(valid_rows)
    last_valid_row = len(valid_rows) - np.argmax(valid_rows[::-1]) - 1
    mid_row = (first_valid_row + last_valid_row) // 2

    # Get unique coordinates from valid region
    y_full = y[:, mid_col]
    x_unique = x[mid_row, :]

    print(f"  Valid row range: {first_valid_row} to {last_valid_row}")
    print(f"  Valid col range: {first_valid_col} to {last_valid_col}")
    print(f"  X range: {np.nanmin(x_unique):.2f} to {np.nanmax(x_unique):.2f} mm")
    print(f"  Y range: {np.nanmin(y_full):.2f} to {np.nanmax(y_full):.2f} mm")

    # Apply x range filter
    x_mask = (x_unique >= x_min) & (x_unique <= x_max) & ~np.isnan(x_unique)

    print(f"  Keeping X: {x_min:.2f} to {x_max:.2f} mm")
    print(f"  X points: {x_mask.sum()} / {len(x_unique)}")

    # Filter out rows with invalid y values
    y_valid_mask = ~np.isnan(y_full)

    # Compute profiles for all variables
    profiles = {}

    for var in ['ux', 'uy', 'uz', 'uu', 'vv', 'ww', 'uv', 'uw', 'vw']:
        data = piv_data[var]
        # Average along x for each row, then keep only valid y rows
        row_means = np.nanmean(data[:, x_mask], axis=1)
        profiles[var] = row_means[y_valid_mask]

    # Store y for valid rows only
    profiles['y_mm'] = y_full[y_valid_mask]

    # Rename for clarity
    profiles['U'] = profiles.pop('ux')
    profiles['V'] = profiles.pop('uy')
    profiles['W'] = profiles.pop('uz')

    return profiles


def convert_to_wall_units(profiles, wall_units, y_offset_mm=0.0):
    """
    Convert profiles to wall units.

    Parameters
    ----------
    profiles : dict
        PIV profiles with y_mm, U, V, W, stresses
    wall_units : dict
        Wall unit parameters
    y_offset_mm : float
        Offset to add to y_mm for coordinate alignment
    """
    u_tau = wall_units['u_tau']
    delta_nu = wall_units['delta_nu']
    u_tau2 = u_tau ** 2

    y_mm_aligned = profiles['y_mm'] + y_offset_mm

    return {
        'y_mm': y_mm_aligned,
        'y_plus': y_mm_aligned / delta_nu,
        # Velocities
        'U_plus': profiles['U'] / u_tau,
        'V_plus': profiles['V'] / u_tau,
        'W_plus': profiles['W'] / u_tau,
        # Normal stresses
        'uu_plus': profiles['uu'] / u_tau2,
        'vv_plus': profiles['vv'] / u_tau2,
        'ww_plus': profiles['ww'] / u_tau2,
        # Shear stresses
        'uv_plus': profiles['uv'] / u_tau2,
        'uw_plus': profiles['uw'] / u_tau2,
        'vw_plus': profiles['vw'] / u_tau2,
    }


def compute_errors(piv_plus, gt_plus, y_plus_range=(10, 500)):
    """Compute error metrics between PIV and ground truth."""
    y_piv = piv_plus['y_plus']
    y_gt = gt_plus['y_plus']

    mask_piv = (y_piv >= y_plus_range[0]) & (y_piv <= y_plus_range[1])
    y_compare = y_piv[mask_piv]

    if len(y_compare) == 0:
        print(f"  Warning: No PIV points in y+ range {y_plus_range}")
        return {}

    errors = {}
    variables = ['U_plus', 'V_plus', 'W_plus', 'uu_plus', 'vv_plus', 'ww_plus',
                 'uv_plus', 'uw_plus', 'vw_plus']

    for var in variables:
        if var not in piv_plus or var not in gt_plus:
            continue

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

        gt_range = np.ptp(gt_valid)
        rms_rel = (rms_error / gt_range * 100) if gt_range > 0 else np.nan

        corr = np.corrcoef(piv_valid, gt_valid)[0, 1] if len(piv_valid) > 1 else np.nan

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


def plot_velocity_comparison(piv_plus, gt_plus, wall_units, errors, output_dir):
    """Generate velocity comparison plots (U+, V+, W+)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Re_tau = wall_units['Re_tau']

    # ==========================================================================
    # Figure 1: U+ profile (semilog)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.semilogx(gt_plus['y_plus'], gt_plus['U_plus'], 'k-',
                linewidth=2, label='DNS (1px)', zorder=3)
    ax.semilogx(piv_plus['y_plus'], piv_plus['U_plus'], 'ro',
                markersize=4, alpha=0.7, label='Stereo PIV', zorder=2)

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
    ax.set_title(f'Mean Streamwise Velocity - Stereo PIV (Re$_\\tau$ = {Re_tau:.0f})', fontsize=16)
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

    # ==========================================================================
    # Figure 2: V+ profile
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(gt_plus['y_plus'], gt_plus['V_plus'], 'k-', linewidth=2, label='DNS')
    ax.plot(piv_plus['y_plus'], piv_plus['V_plus'], 'ro', markersize=4,
            alpha=0.7, label='Stereo PIV')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$V^+$', fontsize=14)
    ax.set_title('Mean Wall-Normal Velocity - Stereo PIV', fontsize=16)
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    if 'V_plus' in errors:
        ax.text(0.02, 0.98, f"R² = {errors['V_plus']['r2']:.4f}",
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_dir / 'V_plus_profile.png', dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Figure 3: W+ profile (spanwise - should be ~0 for channel flow)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(gt_plus['y_plus'], gt_plus['W_plus'], 'k-', linewidth=2, label='DNS')
    ax.plot(piv_plus['y_plus'], piv_plus['W_plus'], 'bo', markersize=4,
            alpha=0.7, label='Stereo PIV')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$W^+$', fontsize=14)
    ax.set_title('Mean Spanwise Velocity - Stereo PIV (should be ~0)', fontsize=16)
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    if 'W_plus' in errors:
        ax.text(0.02, 0.98, f"R² = {errors['W_plus']['r2']:.4f}",
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_dir / 'W_plus_profile.png', dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Figure 4: All velocities combined
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # U+
    ax = axes[0]
    ax.semilogx(gt_plus['y_plus'], gt_plus['U_plus'], 'k-', linewidth=2, label='DNS')
    ax.semilogx(piv_plus['y_plus'], piv_plus['U_plus'], 'ro', markersize=3, alpha=0.7, label='Stereo')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r'$U^+$', fontsize=12)
    ax.set_title('Streamwise Velocity', fontsize=14)
    ax.legend()
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # V+
    ax = axes[1]
    ax.plot(gt_plus['y_plus'], gt_plus['V_plus'], 'k-', linewidth=2, label='DNS')
    ax.plot(piv_plus['y_plus'], piv_plus['V_plus'], 'ro', markersize=3, alpha=0.7, label='Stereo')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r'$V^+$', fontsize=12)
    ax.set_title('Wall-Normal Velocity', fontsize=14)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # W+
    ax = axes[2]
    ax.plot(gt_plus['y_plus'], gt_plus['W_plus'], 'k-', linewidth=2, label='DNS')
    ax.plot(piv_plus['y_plus'], piv_plus['W_plus'], 'bo', markersize=3, alpha=0.7, label='Stereo')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r'$W^+$', fontsize=12)
    ax.set_title('Spanwise Velocity', fontsize=14)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'velocities_combined.png', dpi=150)
    plt.close(fig)


def plot_normal_stresses(piv_plus, gt_plus, wall_units, errors, output_dir):
    """Generate normal stress plots (uu+, vv+, ww+)."""
    output_dir = Path(output_dir)
    Re_tau = wall_units['Re_tau']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # uu+
    ax = axes[0]
    ax.plot(gt_plus['y_plus'], gt_plus['uu_plus'], 'k-', linewidth=2, label='DNS')
    ax.plot(piv_plus['y_plus'], piv_plus['uu_plus'], 'ro', markersize=3, alpha=0.7, label='Stereo')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{u'u'}^+$", fontsize=12)
    ax.set_title('Streamwise Normal Stress', fontsize=14)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)
    if 'uu_plus' in errors:
        ax.text(0.98, 0.98, f"R² = {errors['uu_plus']['r2']:.4f}",
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # vv+
    ax = axes[1]
    ax.plot(gt_plus['y_plus'], gt_plus['vv_plus'], 'k-', linewidth=2, label='DNS')
    ax.plot(piv_plus['y_plus'], piv_plus['vv_plus'], 'go', markersize=3, alpha=0.7, label='Stereo')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{v'v'}^+$", fontsize=12)
    ax.set_title('Wall-Normal Normal Stress', fontsize=14)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)
    if 'vv_plus' in errors:
        ax.text(0.98, 0.98, f"R² = {errors['vv_plus']['r2']:.4f}",
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ww+
    ax = axes[2]
    ax.plot(gt_plus['y_plus'], gt_plus['ww_plus'], 'k-', linewidth=2, label='DNS')
    ax.plot(piv_plus['y_plus'], piv_plus['ww_plus'], 'bo', markersize=3, alpha=0.7, label='Stereo')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{w'w'}^+$", fontsize=12)
    ax.set_title('Spanwise Normal Stress', fontsize=14)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)
    if 'ww_plus' in errors:
        ax.text(0.98, 0.98, f"R² = {errors['ww_plus']['r2']:.4f}",
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Normal Reynolds Stresses - Stereo PIV', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'normal_stresses.png', dpi=150)
    plt.close(fig)


def plot_shear_stresses(piv_plus, gt_plus, wall_units, errors, output_dir):
    """Generate shear stress plots (-uv+, -uw+, -vw+)."""
    output_dir = Path(output_dir)
    Re_tau = wall_units['Re_tau']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # -uv+
    ax = axes[0]
    ax.plot(gt_plus['y_plus'], -gt_plus['uv_plus'], 'k-', linewidth=2, label='DNS')
    ax.plot(piv_plus['y_plus'], -piv_plus['uv_plus'], 'ro', markersize=3, alpha=0.7, label='Stereo')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$-\overline{u'v'}^+$", fontsize=12)
    ax.set_title('Reynolds Shear Stress (u-v)', fontsize=14)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)
    if 'uv_plus' in errors:
        ax.text(0.98, 0.98, f"R² = {errors['uv_plus']['r2']:.4f}",
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # -uw+
    ax = axes[1]
    ax.plot(gt_plus['y_plus'], -gt_plus['uw_plus'], 'k-', linewidth=2, label='DNS')
    ax.plot(piv_plus['y_plus'], -piv_plus['uw_plus'], 'go', markersize=3, alpha=0.7, label='Stereo')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$-\overline{u'w'}^+$", fontsize=12)
    ax.set_title('Reynolds Shear Stress (u-w)', fontsize=14)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)
    if 'uw_plus' in errors:
        ax.text(0.98, 0.98, f"R² = {errors['uw_plus']['r2']:.4f}",
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # -vw+
    ax = axes[2]
    ax.plot(gt_plus['y_plus'], -gt_plus['vw_plus'], 'k-', linewidth=2, label='DNS')
    ax.plot(piv_plus['y_plus'], -piv_plus['vw_plus'], 'bo', markersize=3, alpha=0.7, label='Stereo')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$-\overline{v'w'}^+$", fontsize=12)
    ax.set_title('Reynolds Shear Stress (v-w)', fontsize=14)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)
    if 'vw_plus' in errors:
        ax.text(0.98, 0.98, f"R² = {errors['vw_plus']['r2']:.4f}",
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Reynolds Shear Stresses - Stereo PIV', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'shear_stresses.png', dpi=150)
    plt.close(fig)


def plot_combined_stresses(piv_plus, gt_plus, wall_units, errors, output_dir):
    """Plot uu+, vv+, ww+, -uv+ all on a single axis."""
    output_dir = Path(output_dir)
    Re_tau = wall_units['Re_tau']

    fig, ax = plt.subplots(figsize=(12, 8))

    # Ground truth / reference
    ax.plot(gt_plus['y_plus'], gt_plus['uu_plus'], 'k-', linewidth=2, label=r"Ref $\overline{u'u'}^+$")
    ax.plot(gt_plus['y_plus'], gt_plus['vv_plus'], 'k--', linewidth=2, label=r"Ref $\overline{v'v'}^+$")
    ax.plot(gt_plus['y_plus'], gt_plus['ww_plus'], 'k-.', linewidth=2, label=r"Ref $\overline{w'w'}^+$")
    ax.plot(gt_plus['y_plus'], -gt_plus['uv_plus'], 'k:', linewidth=2, label=r"Ref $-\overline{u'v'}^+$")

    # Stereo PIV
    ax.plot(piv_plus['y_plus'], piv_plus['uu_plus'], 'ro', markersize=3, alpha=0.6, label=r"Stereo $\overline{u'u'}^+$")
    ax.plot(piv_plus['y_plus'], piv_plus['vv_plus'], 'gs', markersize=3, alpha=0.6, label=r"Stereo $\overline{v'v'}^+$")
    ax.plot(piv_plus['y_plus'], piv_plus['ww_plus'], 'b^', markersize=3, alpha=0.6, label=r"Stereo $\overline{w'w'}^+$")
    ax.plot(piv_plus['y_plus'], -piv_plus['uv_plus'], 'mD', markersize=3, alpha=0.6, label=r"Stereo $-\overline{u'v'}^+$")

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'Stress$^+$', fontsize=14)
    ax.set_title(f'Reynolds Stresses - Stereo PIV vs Reference (Re$_\\tau$ = {Re_tau:.0f})', fontsize=16)
    ax.legend(fontsize=10, ncol=2, loc='upper right')
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'combined_stresses.png', dpi=150)
    plt.close(fig)


def main(run_idx=2, x_min=5.0, x_max=145.0, gt_dir=None, stereo_base=None):
    """Main stereo benchmark comparison function."""

    # Paths
    script_dir = Path(__file__).parent
    data_root = Path(r'C:/Users/mtt1e23/OneDrive - University of Southampton'
                     r'/Documents/#current_processing/4000_images_channel')

    if gt_dir is None:
        # Try local ground_truth first, fall back to ensemble_statistics_first_1000
        gt_dir_local = script_dir / 'ground_truth' / 'ensemble_statistics'
        if gt_dir_local.exists():
            gt_dir = gt_dir_local
        else:
            gt_dir = data_root / 'ensemble_statistics_first_1000'

    if stereo_base is None:
        stereo_base = data_root / 'stereo_images' / 'validation_first_1000'

    stats_path = stereo_base / 'statistics/1000/stereo/Cam1_Cam2/instantaneous/mean_stats/mean_stats.mat'
    coords_path = stereo_base / 'stereo_calibrated/1000/Cam1_Cam2/instantaneous/coordinates.mat'

    output_dir = script_dir / 'benchmark_results_stereo'

    print("=" * 70)
    print("STEREO PIV BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"Run index: {run_idx}")
    print(f"X range: {x_min} to {x_max} mm")

    # Load data - detect file format (old: wall_units.mat + profiles.mat, new: HDF5)
    wall_units_file = gt_dir / 'wall_units.mat'
    profiles_file = gt_dir / 'profiles.mat'
    if not wall_units_file.exists():
        wall_units_file = gt_dir / 'diagnostics.mat'
    if not profiles_file.exists():
        profiles_file = gt_dir / 'ensemble_statistics_full.mat'

    print("\n[1] Loading wall units...")
    print(f"  Source: {wall_units_file.name}")
    wall_units = load_wall_units(wall_units_file)
    print(f"  u_tau = {wall_units['u_tau']:.4f} mm/s")
    print(f"  delta_nu = {wall_units['delta_nu']:.4f} mm")
    print(f"  Re_tau = {wall_units['Re_tau']:.0f}")

    print("\n[2] Loading ground truth (3-component)...")
    print(f"  Source: {profiles_file.name}")
    gt = load_ground_truth_3d(profiles_file)
    print(f"  y+ range: {gt['y_plus'].min():.1f} to {gt['y_plus'].max():.1f}")
    print(f"  U range: {gt['U'].min():.2f} to {gt['U'].max():.2f} mm/s")
    print(f"  W range: {gt['W'].min():.2f} to {gt['W'].max():.2f} mm/s")

    print(f"\n[3] Loading stereo PIV statistics (run {run_idx})...")
    piv = load_stereo_statistics(stats_path, coords_path, run_idx=run_idx)
    print(f"  Grid size: {piv['ux'].shape}")
    print(f"  ux range: {np.nanmin(piv['ux']):.2f} to {np.nanmax(piv['ux']):.2f} mm/s")
    print(f"  uz (W) range: {np.nanmin(piv['uz']):.2f} to {np.nanmax(piv['uz']):.2f} mm/s")

    print("\n[4] Computing x-averaged profiles...")
    piv_profiles = compute_stereo_profiles(piv, x_min=x_min, x_max=x_max)
    print(f"  y range: {piv_profiles['y_mm'].min():.2f} to {piv_profiles['y_mm'].max():.2f} mm")

    print("\n[5] Converting to wall units...")
    y_offset_mm = -piv_profiles['y_mm'].min()
    print(f"  Applying y-offset: {y_offset_mm:.2f} mm")

    piv_plus = convert_to_wall_units(piv_profiles, wall_units, y_offset_mm=y_offset_mm)
    piv_plus['y_plus'] = piv_plus['y_plus'] + 1.0  # shift y+ by +1
    print(f"  y+ range: {piv_plus['y_plus'].min():.1f} to {piv_plus['y_plus'].max():.1f} (after +1 shift)")

    # Ground truth in wall units
    u_tau = wall_units['u_tau']
    u_tau2 = u_tau ** 2
    gt_plus = {
        'y_plus': gt['y_plus'],
        'U_plus': gt['U_plus'],
        'V_plus': gt['V'] / u_tau,
        'W_plus': gt['W'] / u_tau,
        'uu_plus': gt['uu_plus'],
        'vv_plus': gt['vv_plus'],
        'ww_plus': gt['ww_plus'],
        'uv_plus': gt['uv_plus'],
        'uw_plus': gt['uw'] / u_tau2,
        'vw_plus': gt['vw'] / u_tau2,
    }

    print("\n[6] Computing error metrics (y+ = 10-500)...")
    errors = compute_errors(piv_plus, gt_plus, y_plus_range=(10, 500))

    # Print results
    print("\n" + "=" * 70)
    print("STEREO BENCHMARK RESULTS")
    print("=" * 70)

    var_names = {
        'U_plus': 'Streamwise Velocity (U+)',
        'V_plus': 'Wall-normal Velocity (V+)',
        'W_plus': 'Spanwise Velocity (W+)',
        'uu_plus': 'Streamwise Stress (uu+)',
        'vv_plus': 'Wall-normal Stress (vv+)',
        'ww_plus': 'Spanwise Stress (ww+)',
        'uv_plus': 'Shear Stress (uv+)',
        'uw_plus': 'Shear Stress (uw+)',
        'vw_plus': 'Shear Stress (vw+)',
    }

    for var, err in errors.items():
        name = var_names.get(var, var)
        print(f"\n{name}:")
        print(f"  RMS Error: {err['rms']:.4f} ({err['rms_rel']:.1f}% of range)")
        print(f"  R²: {err['r2']:.4f}")
        print(f"  Correlation: {err['corr']:.4f}")

    print("\n[7] Generating plots...")
    plot_velocity_comparison(piv_plus, gt_plus, wall_units, errors, output_dir)
    plot_normal_stresses(piv_plus, gt_plus, wall_units, errors, output_dir)
    plot_shear_stresses(piv_plus, gt_plus, wall_units, errors, output_dir)
    plot_combined_stresses(piv_plus, gt_plus, wall_units, errors, output_dir)

    print(f"\nPlots saved to: {output_dir}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Variable':<20} {'R²':<10} {'RMS%':<10} {'Corr':<10}")
    print("-" * 50)
    for var in ['U_plus', 'V_plus', 'W_plus', 'uu_plus', 'vv_plus', 'ww_plus', 'uv_plus', 'uw_plus', 'vw_plus']:
        if var in errors:
            e = errors[var]
            print(f"{var:<20} {e['r2']:<10.4f} {e['rms_rel']:<10.1f} {e['corr']:<10.4f}")

    print("\n" + "=" * 70)
    print("STEREO BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stereo PIV Benchmark Comparison')
    parser.add_argument('--run', '-r', type=int, default=2,
                        help='Run index (0-based), default=2 (finest available)')
    parser.add_argument('--x-min', type=float, default=5.0,
                        help='Minimum x to include (mm), default=5.0')
    parser.add_argument('--x-max', type=float, default=145.0,
                        help='Maximum x to include (mm), default=145.0')
    args = parser.parse_args()

    main(run_idx=args.run, x_min=args.x_min, x_max=args.x_max)
