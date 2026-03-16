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
import matplotlib as mpl
from pathlib import Path

# Use LaTeX-style fonts everywhere (Computer Modern via mathtext — no LaTeX install needed)
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'text.usetex': False,
})


def log_smooth(y_plus, values, sigma_decades=0.06):
    """LOWESS-style smooth in log(y+) space, evaluated at original points.

    Each output point is a locally-weighted LINEAR regression of neighbours,
    where distance is measured in decades of y+. Using local linear fits
    instead of local averages gives:
      - Better peak tracking (local slope captures gradients)
      - Better edge behaviour (linear extrapolation, not mean bias)

    Parameters
    ----------
    y_plus : array
        y+ coordinates (positive)
    values : array
        Values to smooth
    sigma_decades : float
        Smoothing width in decades of y+ (0.06 ~ +/-15% local y+)

    Returns
    -------
    y_out, smoothed : arrays
        Sorted y+ and smoothed values (at original data points)
    """
    valid = (y_plus > 0) & ~np.isnan(values)
    yp = y_plus[valid]
    vals = values[valid]
    if len(yp) < 5:
        return yp, vals

    # Sort by y+
    order = np.argsort(yp)
    yp = yp[order]
    vals = vals[order]
    log_yp = np.log10(yp)

    # Local linear regression (LOWESS) at each point
    smoothed = np.empty_like(vals)
    for i in range(len(vals)):
        d = (log_yp - log_yp[i]) / sigma_decades
        w = np.exp(-0.5 * d * d)
        wsum = np.sum(w)
        wmean_x = np.sum(w * log_yp) / wsum
        wmean_y = np.sum(w * vals) / wsum
        dx = log_yp - wmean_x
        denom = np.sum(w * dx * dx)
        if denom > 1e-30:
            slope = np.sum(w * dx * vals) / denom
            smoothed[i] = wmean_y + slope * (log_yp[i] - wmean_x)
        else:
            smoothed[i] = wmean_y

    return yp, smoothed


def plot_ci_band(ax, y_plus, ci_lo, ci_hi, sign=1, color='k', alpha=0.3, zorder=1):
    """Plot a 95% CI shaded band around a reference line.

    Parameters
    ----------
    ax : matplotlib Axes
    y_plus : array
        x-axis values (y+ coordinates)
    ci_lo, ci_hi : array
        Lower/upper CI bounds (same units as the plotted variable)
    sign : int
        1 or -1 (for variables like -uv+ that flip sign)
    color : str
        Fill color
    alpha : float
        Fill transparency
    zorder : int
        Drawing order
    """
    lo = sign * ci_lo if sign == 1 else sign * ci_hi  # sign flip swaps lo/hi
    hi = sign * ci_hi if sign == 1 else sign * ci_lo
    ax.fill_between(y_plus, lo, hi, color=color, alpha=alpha, zorder=zorder,
                    linewidth=0)
    # Add thin edge lines so the CI is visible even when narrow
    ax.plot(y_plus, lo, color=color, linewidth=0.5, alpha=0.4, zorder=zorder)
    ax.plot(y_plus, hi, color=color, linewidth=0.5, alpha=0.4, zorder=zorder)


def load_wall_units(wall_units_path):
    """Load wall units from .mat file (scipy v5 or h5py v7.3).

    Supports three formats:
    - wall_units.mat with 'wall_units' struct
    - diagnostics.mat with 'diagnostics' group (HDF5)
    - direct_stats.mat with top-level u_tau, delta_nu, Re_tau keys
    """
    try:
        wall = sio.loadmat(wall_units_path, squeeze_me=True, struct_as_record=False)

        # Format: direct_stats.mat (top-level scalar keys)
        if 'u_tau' in wall and 'delta_nu' in wall and 'Re_tau' in wall:
            u_tau = float(wall['u_tau'])
            delta_nu = float(wall['delta_nu'])
            Re_tau = float(wall['Re_tau'])
            return {
                'u_tau': u_tau,
                'nu': u_tau * delta_nu,
                'delta_nu': delta_nu,
                'h_mm': float(wall['h_mm']) if 'h_mm' in wall else Re_tau * delta_nu,
                'Re_tau': Re_tau,
            }

        # Format: wall_units.mat (struct)
        wu = wall['wall_units']
        return {
            'u_tau': float(wu.u_tau),  # mm/s
            'nu': float(wu.nu),        # mm^2/s
            'delta_nu': float(wu.delta_nu),  # mm
            'h_mm': float(wu.h_mm),    # mm
            'Re_tau': float(wu.Re_tau)
        }
    except NotImplementedError:
        # MATLAB v7.3 (HDF5) format - use h5py
        import h5py
        with h5py.File(str(wall_units_path), 'r') as f:
            if 'wall_units' in f:
                grp = f['wall_units']
                result = {
                    'u_tau': float(np.array(grp['u_tau']).flat[0]),
                    'delta_nu': float(np.array(grp['delta_nu']).flat[0]),
                    'Re_tau': float(np.array(grp['Re_tau']).flat[0]),
                }
                if 'nu' in grp:
                    result['nu'] = float(np.array(grp['nu']).flat[0])
                else:
                    result['nu'] = result['u_tau'] * result['delta_nu']
                if 'h_mm' in grp:
                    result['h_mm'] = float(np.array(grp['h_mm']).flat[0])
                else:
                    result['h_mm'] = result['Re_tau'] * result['delta_nu']
                return result
            elif 'diagnostics' in f:
                grp = f['diagnostics']
                result = {
                    'u_tau': float(np.array(grp['u_tau']).flat[0]),
                    'delta_nu': float(np.array(grp['delta_nu']).flat[0]),
                    'Re_tau': float(np.array(grp['Re_tau']).flat[0]),
                }
                if 'nu' in grp:
                    result['nu'] = float(np.array(grp['nu']).flat[0])
                else:
                    result['nu'] = result['u_tau'] * result['delta_nu']
                if 'h_mm' in grp:
                    result['h_mm'] = float(np.array(grp['h_mm']).flat[0])
                else:
                    result['h_mm'] = result['Re_tau'] * result['delta_nu']
                return result
            else:
                raise ValueError(f"No 'wall_units' or 'diagnostics' group in {wall_units_path}")


def load_ground_truth(profiles_path, wall_units_path=None):
    """Load ground truth profiles (scipy v5 or h5py v7.3).

    Supports three formats:
    - profiles.mat with 'profiles.win_1px' struct
    - ensemble_statistics_full.mat with 'ref_profile' + 'ensemble_stats' (HDF5)
    - direct_stats.mat with top-level y_plus, U_plus, stress_plus arrays
    """
    try:
        profiles = sio.loadmat(profiles_path, squeeze_me=True, struct_as_record=False)

        # Format: direct_stats.mat (top-level arrays)
        if 'U_plus' in profiles and 'stress_plus' in profiles and 'y_plus' in profiles:
            y_plus_full = profiles['y_plus']
            Re_tau = float(profiles['Re_tau'])
            u_tau = float(profiles['u_tau'])
            delta_nu = float(profiles['delta_nu'])
            u_tau2 = u_tau ** 2

            # Select lower half of channel (y+ <= Re_tau)
            mask = y_plus_full <= Re_tau
            y_plus = y_plus_full[mask]
            y_mm = y_plus * delta_nu

            # U_plus: (N, 3) -> columns [U, V, W]
            U_plus = profiles['U_plus'][mask, 0]
            V_plus = profiles['U_plus'][mask, 1]

            # stress_plus: (N, 3, 3) -> Reynolds stress tensor in plus units
            uu_plus = profiles['stress_plus'][mask, 0, 0]
            vv_plus = profiles['stress_plus'][mask, 1, 1]
            uv_plus = profiles['stress_plus'][mask, 0, 1]

            result = {
                'y_mm': y_mm,
                'y_plus': y_plus,
                'U': U_plus * u_tau,      # mm/s
                'V': V_plus * u_tau,      # mm/s
                'uu': uu_plus * u_tau2,   # (mm/s)^2
                'vv': vv_plus * u_tau2,   # (mm/s)^2
                'uv': uv_plus * u_tau2,   # (mm/s)^2
                'U_plus': U_plus,
                'uu_plus': uu_plus,
                'vv_plus': vv_plus,
                'uv_plus': uv_plus,
            }

            # Load 95% confidence intervals if available
            if 'stress_ci_lo' in profiles and 'stress_ci_hi' in profiles:
                result['uu_plus_ci_lo'] = profiles['stress_ci_lo'][mask, 0, 0]
                result['uu_plus_ci_hi'] = profiles['stress_ci_hi'][mask, 0, 0]
                result['vv_plus_ci_lo'] = profiles['stress_ci_lo'][mask, 1, 1]
                result['vv_plus_ci_hi'] = profiles['stress_ci_hi'][mask, 1, 1]
                result['uv_plus_ci_lo'] = profiles['stress_ci_lo'][mask, 0, 1]
                result['uv_plus_ci_hi'] = profiles['stress_ci_hi'][mask, 0, 1]
            if 'umean_ci_lo' in profiles and 'umean_ci_hi' in profiles:
                result['U_plus_ci_lo'] = profiles['umean_ci_lo'][mask, 0]
                result['U_plus_ci_hi'] = profiles['umean_ci_hi'][mask, 0]
                result['V_plus_ci_lo'] = profiles['umean_ci_lo'][mask, 1]
                result['V_plus_ci_hi'] = profiles['umean_ci_hi'][mask, 1]

            return result

        # Format: profiles.mat (struct)
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
    except NotImplementedError:
        # MATLAB v7.3 (HDF5) format
        import h5py

        # Load wall units for normalisation
        if wall_units_path is not None:
            wu = load_wall_units(wall_units_path)
        else:
            wu_path = Path(profiles_path).parent / 'diagnostics.mat'
            if wu_path.exists():
                wu = load_wall_units(wu_path)
            else:
                raise ValueError("Need wall_units_path to normalise HDF5 ground truth")

        u_tau = wu['u_tau']
        u_tau2 = u_tau ** 2
        delta_nu = wu['delta_nu']

        with h5py.File(str(profiles_path), 'r') as f:
            ref = f['ref_profile']

            # Check if ref_profile has stress data directly
            if 'uu' in ref:
                y_mm = np.array(ref['y_mm']).flatten()
                U = np.array(ref['U']).flatten()
                V = np.array(ref['V']).flatten()
                uu = np.array(ref['uu']).flatten()
                vv = np.array(ref['vv']).flatten()
                uv = np.array(ref['uv']).flatten()
                y_plus = y_mm / delta_nu
                return {
                    'y_mm': y_mm, 'y_plus': y_plus,
                    'U': U, 'V': V, 'uu': uu, 'vv': vv, 'uv': uv,
                    'U_plus': U / u_tau, 'uu_plus': uu / u_tau2,
                    'vv_plus': vv / u_tau2, 'uv_plus': uv / u_tau2,
                }

            # Use ensemble_stats profiles (pre-averaged, consistent y_plus)
            if 'ensemble_stats' not in f:
                raise ValueError("No stress data in ref_profile and no ensemble_stats")

            es = f['ensemble_stats']
            # Use finest window (index 0) as reference
            win_idx = 0

            def _deref(field, idx=win_idx):
                refs = np.array(es[field]).flatten()
                return np.array(f[refs[idx]]).flatten()

            # Ensemble stats y_plus (255 points for 16x16 window)
            es_y_plus = _deref('y_plus')
            es_y_mm = es_y_plus * delta_nu

            # Stresses are already in plus units
            uu_plus = _deref('uu_plus')
            vv_plus = _deref('vv_plus')
            uv_plus = _deref('uv_plus')

            # Velocity: interpolate DNS onto ensemble y_plus grid
            dns_y_mm = np.array(ref['y_mm']).flatten()
            dns_U = np.array(ref['U']).flatten()
            dns_V = np.array(ref['V']).flatten()
            dns_y_plus = dns_y_mm / delta_nu

            U_interp = interp1d(dns_y_plus, dns_U, kind='linear',
                                bounds_error=False, fill_value=np.nan)(es_y_plus)
            V_interp = interp1d(dns_y_plus, dns_V, kind='linear',
                                bounds_error=False, fill_value=np.nan)(es_y_plus)

        return {
            'y_mm': es_y_mm,
            'y_plus': es_y_plus,
            'U': U_interp,
            'V': V_interp,
            'uu': uu_plus * u_tau2,
            'vv': vv_plus * u_tau2,
            'uv': uv_plus * u_tau2,
            'U_plus': U_interp / u_tau,
            'uu_plus': uu_plus,
            'vv_plus': vv_plus,
            'uv_plus': uv_plus,
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
        PIV statistics dictionary (velocities in m/s, stresses in (m/s)^2)
    x_exclude_vectors : int
        Number of vectors to exclude from each side in x-direction

    Returns
    -------
    dict with y_mm, U, uu, vv, uv profiles (in mm/s and (mm/s)^2)
    """
    x = piv_data['x']
    y = piv_data['y']

    # Get unique y values (assuming regular grid)
    y_unique = y[:, 0]
    x_unique = x[0, :]
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


def plot_comparison(piv_plus, gt_plus, wall_units, errors, output_dir, window_label='16x16', show_fit_lines=False):
    """Generate comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Re_tau = wall_units['Re_tau']

    # Check for CI data
    has_ci = 'uu_plus_ci_lo' in gt_plus

    # Figure 1: Mean velocity profile (semi-log)
    fig, ax = plt.subplots(figsize=(10, 7))

    # Ground truth with CI band
    if has_ci and 'U_plus_ci_lo' in gt_plus:
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['U_plus_ci_lo'],
                     gt_plus['U_plus_ci_hi'], color='k', alpha=0.15, zorder=1)
    ax.semilogx(gt_plus['y_plus'], gt_plus['U_plus'], 'k-',
                linewidth=2, label='DNS (1px)', zorder=3)

    # PIV
    ax.semilogx(piv_plus['y_plus'], piv_plus['U_plus'], 'ro',
                markersize=4, alpha=0.7, label=f'PIV ({window_label})', zorder=2)

    if show_fit_lines:
        y_log = np.logspace(1, np.log10(Re_tau), 100)
        kappa, B = 0.41, 5.2
        ax.semilogx(y_log, (1/kappa)*np.log(y_log)+B, 'b--', linewidth=1, alpha=0.7,
                    label=r'Log law: $U^+ = \frac{1}{\kappa}\ln(y^+) + B$')
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
    if has_ci:
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['uu_plus_ci_lo'],
                     gt_plus['uu_plus_ci_hi'], color='k', zorder=1)
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
    if has_ci:
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['vv_plus_ci_lo'],
                     gt_plus['vv_plus_ci_hi'], color='k', zorder=1)
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
    if has_ci:
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['uv_plus_ci_lo'],
                     gt_plus['uv_plus_ci_hi'], sign=-1, color='k', zorder=1)
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

    if has_ci and 'V_plus_ci_lo' in gt_plus:
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['V_plus_ci_lo'],
                     gt_plus['V_plus_ci_hi'], color='k', zorder=1)
    ax.plot(gt_plus['y_plus'], gt_plus['V_plus'], 'k-', linewidth=2, label='DNS')
    ax.plot(piv_plus['y_plus'], piv_plus['V_plus'], 'ro', markersize=4,
            alpha=0.7, label='PIV')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$V^+$', fontsize=14)
    ax.set_title(f'Mean Wall-Normal Velocity Profile (Re$_\\tau$ = {Re_tau:.0f})', fontsize=16)
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
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
    ax.set_title('Mean Velocity Profile', fontsize=16)
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'U_plus_linear.png', dpi=150)
    plt.close(fig)

    # Figure 5: Smoothed line plots (log-space moving average)
    # ---- U+ smoothed ----
    fig, ax = plt.subplots(figsize=(10, 7))

    if has_ci and 'U_plus_ci_lo' in gt_plus:
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['U_plus_ci_lo'],
                     gt_plus['U_plus_ci_hi'], color='k', zorder=1)
    ax.semilogx(gt_plus['y_plus'], gt_plus['U_plus'], 'k-',
                linewidth=2, label='DNS (1px)', zorder=3)

    ax.semilogx(piv_plus['y_plus'], piv_plus['U_plus'], 'ro',
                markersize=4, alpha=0.7, label=f'PIV ({window_label})', zorder=2)

    if show_fit_lines:
        y_log = np.logspace(1, np.log10(Re_tau), 100)
        kappa, B = 0.41, 5.2
        ax.semilogx(y_log, (1/kappa)*np.log(y_log)+B, 'b--', linewidth=1, alpha=0.7,
                    label=r'Log law')
        y_visc = np.linspace(0.1, 10, 50)
        ax.semilogx(y_visc, y_visc, 'g--', linewidth=1, alpha=0.7, label=r'$U^+=y^+$')

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$U^+$', fontsize=14)
    ax.set_title(f'Mean Velocity Profile - Smoothed ({window_label})', fontsize=16)
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
    fig.savefig(output_dir / 'U_plus_profile_smooth.png', dpi=150)
    plt.close(fig)

    # ---- Reynolds stresses smoothed ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    stress_configs = [
        ('uu_plus', r"$\overline{u'u'}^+$", 'Streamwise Normal Stress', 1),
        ('vv_plus', r"$\overline{v'v'}^+$", 'Wall-Normal Normal Stress', 1),
        ('uv_plus', r"$-\overline{u'v'}^+$", 'Reynolds Shear Stress', -1),
    ]
    for ax, (var, ylabel, title, sign) in zip(axes, stress_configs):
        gt_vals = sign * gt_plus[var]
        piv_vals = sign * piv_plus[var]

        ci_lo_key = f'{var}_ci_lo'
        ci_hi_key = f'{var}_ci_hi'
        if has_ci and ci_lo_key in gt_plus:
            plot_ci_band(ax, gt_plus['y_plus'], gt_plus[ci_lo_key],
                         gt_plus[ci_hi_key], sign=sign, color='k', zorder=1)
        ax.semilogx(gt_plus['y_plus'], gt_vals, 'k-', linewidth=2, label='DNS')
        ax.semilogx(piv_plus['y_plus'], piv_vals, 'ro', markersize=4, alpha=0.7, label='PIV')

        ax.set_xlabel(r'$y^+$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.set_xlim(1, Re_tau)
        ax.grid(True, alpha=0.3)
        if var in errors:
            ax.text(0.98, 0.98, f"R² = {errors[var]['r2']:.4f}",
                    transform=ax.transAxes, fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_dir / 'reynolds_stresses_smooth.png', dpi=150)
    plt.close(fig)

    # Figure 6: Trace invariant (u'u' + v'v') - rotation diagnostic
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Compute trace
    gt_trace = gt_plus['uu_plus'] + gt_plus['vv_plus']
    piv_trace = piv_plus['uu_plus'] + piv_plus['vv_plus']

    # Left: Individual components
    ax = axes[0]
    ax.semilogx(gt_plus['y_plus'], gt_plus['uu_plus'], 'k-', linewidth=2, label="DNS u'u'+")
    ax.semilogx(gt_plus['y_plus'], gt_plus['vv_plus'], 'k--', linewidth=2, label="DNS v'v'+")
    ax.semilogx(piv_plus['y_plus'], piv_plus['uu_plus'], 'ro', markersize=3, alpha=0.7, label="PIV u'u'+")
    ax.semilogx(piv_plus['y_plus'], piv_plus['vv_plus'], 'bs', markersize=3, alpha=0.7, label="PIV v'v'+")
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"Stress$^+$", fontsize=12)
    ax.set_title('Individual Normal Stresses', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # Middle: Trace comparison
    ax = axes[1]
    ax.semilogx(gt_plus['y_plus'], gt_trace, 'k-', linewidth=2, label='DNS')
    ax.semilogx(piv_plus['y_plus'], piv_trace, 'ro', markersize=4, alpha=0.7, label='PIV')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{u'u'}^+ + \overline{v'v'}^+$", fontsize=12)
    ax.set_title("Trace Invariant (u'u' + v'v')", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # Right: Ratio of components (rotation indicator)
    ax = axes[2]
    gt_ratio = gt_plus['uu_plus'] / (gt_plus['vv_plus'] + 1e-10)  # avoid div by zero
    piv_ratio = piv_plus['uu_plus'] / (piv_plus['vv_plus'] + 1e-10)
    ax.semilogx(gt_plus['y_plus'], gt_ratio, 'k-', linewidth=2, label='DNS')
    ax.semilogx(piv_plus['y_plus'], piv_ratio, 'ro', markersize=4, alpha=0.7, label='PIV')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{u'u'}^+ / \overline{v'v'}^+$", fontsize=12)
    ax.set_title("Stress Ratio (rotation indicator)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(1, Re_tau)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Rotation Diagnostic: Trace is invariant under rotation", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'trace_invariant.png', dpi=150)
    plt.close(fig)

    # Figure 7: Residuals (PIV - Ref) vs y+ — velocities and stresses
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Interpolate ground truth onto PIV y+ grid
    gt_interp_fn = {}
    for var in ['U_plus', 'V_plus', 'uu_plus', 'vv_plus', 'uv_plus']:
        gt_interp_fn[var] = interp1d(gt_plus['y_plus'], gt_plus[var], kind='linear',
                                     bounds_error=False, fill_value=np.nan)

    # Top row: velocity residuals
    vel_configs = [
        ('U_plus', r"$U^+_{\mathrm{PIV}} - U^+_{\mathrm{Ref}}$",
         'Mean Streamwise Velocity Residual', 1),
        ('V_plus', r"$V^+_{\mathrm{PIV}} - V^+_{\mathrm{Ref}}$",
         'Mean Wall-Normal Velocity Residual', 1),
    ]
    for ax, (var, ylabel, title, sign) in zip(axes[0, :2], vel_configs):
        gt_at_piv = gt_interp_fn[var](piv_plus['y_plus'])
        residual = sign * piv_plus[var] - sign * gt_at_piv

        ax.semilogx(piv_plus['y_plus'], residual, 'ro', markersize=3, alpha=0.5)
        yp_s, r_s = log_smooth(piv_plus['y_plus'], residual)
        ax.semilogx(yp_s, r_s, 'r-', linewidth=2, label=f'PIV ({window_label})')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)

        ax.set_xlabel(r'$y^+$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.set_xlim(1, Re_tau)
        ax.grid(True, alpha=0.3)

    axes[0, 2].set_visible(False)  # Empty top-right panel

    # Bottom row: stress residuals
    stress_configs = [
        ('uu_plus', r"$\overline{u'u'}^+_{\mathrm{PIV}} - \overline{u'u'}^+_{\mathrm{Ref}}$",
         'Streamwise Normal Stress Residual', 1),
        ('vv_plus', r"$\overline{v'v'}^+_{\mathrm{PIV}} - \overline{v'v'}^+_{\mathrm{Ref}}$",
         'Wall-Normal Normal Stress Residual', 1),
        ('uv_plus', r"$-\overline{u'v'}^+_{\mathrm{PIV}} - (-\overline{u'v'}^+_{\mathrm{Ref}})$",
         'Shear Stress Residual', -1),
    ]
    for ax, (var, ylabel, title, sign) in zip(axes[1, :], stress_configs):
        gt_at_piv = gt_interp_fn[var](piv_plus['y_plus'])
        residual = sign * piv_plus[var] - sign * gt_at_piv

        ax.semilogx(piv_plus['y_plus'], residual, 'ro', markersize=3, alpha=0.5)
        yp_s, r_s = log_smooth(piv_plus['y_plus'], residual)
        ax.semilogx(yp_s, r_s, 'r-', linewidth=2, label=f'PIV ({window_label})')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)

        ax.set_xlabel(r'$y^+$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.set_xlim(1, Re_tau)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'residuals.png', dpi=150)
    plt.close(fig)

    # Figure 8: Noise floor vs gradient correction decomposition
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Compute residuals
    uu_residual = piv_plus['uu_plus'] - gt_interp_fn['uu_plus'](piv_plus['y_plus'])
    vv_residual = piv_plus['vv_plus'] - gt_interp_fn['vv_plus'](piv_plus['y_plus'])
    gradient_only = uu_residual - vv_residual  # u'u' residual minus noise floor

    # Left: Noise floor (v'v' residual)
    ax = axes[0]
    ax.semilogx(piv_plus['y_plus'], vv_residual, 'bo', markersize=2, alpha=0.3)
    yp_s, r_s = log_smooth(piv_plus['y_plus'], vv_residual)
    ax.semilogx(yp_s, r_s, 'b-', linewidth=2.5, label=r"$v'v'$ residual (noise floor)")
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{v'v'}^+_{\mathrm{PIV}} - \overline{v'v'}^+_{\mathrm{Ref}}$", fontsize=12)
    ax.set_title('Noise Floor (isotropic)', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # Middle: Gradient-only residual
    ax = axes[1]
    ax.semilogx(piv_plus['y_plus'], gradient_only, 'ro', markersize=2, alpha=0.3)
    yp_s, r_s = log_smooth(piv_plus['y_plus'], gradient_only)
    ax.semilogx(yp_s, r_s, 'r-', linewidth=2.5,
                label=r"$(\overline{u'u'} - \overline{v'v'})_{\mathrm{PIV}} - (\overline{u'u'} - \overline{v'v'})_{\mathrm{Ref}}$")
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"Gradient-only residual$^+$", fontsize=12)
    ax.set_title(r"Gradient Correction Residual ($u'u' - v'v'$ removes noise)", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # Right: All three overlaid
    ax = axes[2]
    ax.semilogx(piv_plus['y_plus'], uu_residual, 'ro', markersize=2, alpha=0.15)
    yp_s_uu, r_s_uu = log_smooth(piv_plus['y_plus'], uu_residual)
    ax.semilogx(yp_s_uu, r_s_uu, 'r-', linewidth=2, label=r"$u'u'$ residual (total)")

    ax.semilogx(piv_plus['y_plus'], vv_residual, 'bo', markersize=2, alpha=0.15)
    yp_s_vv, r_s_vv = log_smooth(piv_plus['y_plus'], vv_residual)
    ax.semilogx(yp_s_vv, r_s_vv, 'b-', linewidth=2, label=r"$v'v'$ residual (noise floor)")

    yp_s_g, r_s_g = log_smooth(piv_plus['y_plus'], gradient_only)
    ax.semilogx(yp_s_g, r_s_g, 'g--', linewidth=2, label=r"$u'u' - v'v'$ residual (gradient only)")

    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"Residual$^+$", fontsize=12)
    ax.set_title('Decomposition: Total = Noise + Gradient', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Noise Floor vs Gradient Correction ({window_label})', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'noise_gradient_decomposition.png', dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to: {output_dir}")


def plot_combined_stresses(piv_plus, gt_plus, wall_units, errors, output_dir, window_label='16x16'):
    """Plot uu+, vv+, -uv+ all on one axis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Re_tau = wall_units['Re_tau']

    has_ci = 'uu_plus_ci_lo' in gt_plus

    fig, ax = plt.subplots(figsize=(12, 8))

    # CI bands (before reference lines so they render behind)
    if has_ci:
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['uu_plus_ci_lo'],
                     gt_plus['uu_plus_ci_hi'], color='k', alpha=0.12, zorder=1)
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['vv_plus_ci_lo'],
                     gt_plus['vv_plus_ci_hi'], color='k', alpha=0.12, zorder=1)
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['uv_plus_ci_lo'],
                     gt_plus['uv_plus_ci_hi'], sign=-1, color='k', alpha=0.12, zorder=1)

    # Reference (solid lines)
    ax.plot(gt_plus['y_plus'], gt_plus['uu_plus'], 'k-', linewidth=2,
            label=r"Ref $\overline{u'u'}^+$")
    ax.plot(gt_plus['y_plus'], gt_plus['vv_plus'], 'k--', linewidth=2,
            label=r"Ref $\overline{v'v'}^+$")
    ax.plot(gt_plus['y_plus'], -gt_plus['uv_plus'], 'k:', linewidth=2,
            label=r"Ref $-\overline{u'v'}^+$")

    # PIV markers
    marker_configs = [
        ('uu_plus', 1, 'r', 'o', r"PIV $\overline{u'u'}^+$"),
        ('vv_plus', 1, 'g', 's', r"PIV $\overline{v'v'}^+$"),
        ('uv_plus', -1, 'm', 'D', r"PIV $-\overline{u'v'}^+$"),
    ]
    for var, sign, col, mkr, label in marker_configs:
        piv_vals = sign * piv_plus[var]
        ax.plot(piv_plus['y_plus'], piv_vals, color=col, marker=mkr,
                markersize=4, alpha=0.7, linestyle='none', label=label, zorder=5)

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'Stress$^+$', fontsize=14)
    ax.set_title(f'Reynolds Stresses ({window_label}) - PIV vs Reference '
                 f'(Re$_\\tau$ = {Re_tau:.0f})', fontsize=16)
    ax.legend(fontsize=10, ncol=2, loc='upper right')
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'combined_stresses.png', dpi=150)
    plt.close(fig)
    print(f"  Combined stresses plot saved to: {output_dir / 'combined_stresses.png'}")


def main(mode='instantaneous', gt_dir=None, base_dir=None, ensemble_dir=None, num_frames=1000, output_dir_override=None, show_fit_lines=False):
    """Main benchmark comparison function.

    Parameters
    ----------
    mode : str
        'instantaneous' or 'ensemble'
    gt_dir : Path
        Ground truth directory path (required)
    base_dir : Path, optional
        Base directory containing PIV results
    ensemble_dir : Path, optional
        Direct path to ensemble directory containing ensemble_result.mat and coordinates.mat.
        If provided, overrides base_dir for ensemble mode.
    num_frames : int
        Number of frames subdirectory (e.g. 1000 or 4000). Used in path construction.
    output_dir_override : Path, optional
        Custom output directory. If None, uses default naming.
    """
    if gt_dir is None:
        raise ValueError("gt_dir is required. Please provide the ground truth directory path.")

    # Paths
    script_dir = Path(__file__).parent
    gt_dir = Path(gt_dir)

    if mode == 'ensemble':
        if ensemble_dir is not None:
            ensemble_dir = Path(ensemble_dir)
            ensemble_path = ensemble_dir / 'ensemble_result.mat'
            coords_path = ensemble_dir / 'coordinates.mat'
        elif base_dir is not None:
            base_dir = Path(base_dir)
            ensemble_path = base_dir / f'calibrated_piv/{num_frames}/Cam1/ensemble/ensemble_result.mat'
            coords_path = base_dir / f'calibrated_piv/{num_frames}/Cam1/ensemble/coordinates.mat'
        else:
            raise ValueError("Either ensemble_dir or base_dir must be provided for ensemble mode.")
        output_dir = output_dir_override or (script_dir / 'benchmark_results_ensemble')
    else:
        if base_dir is None:
            raise ValueError("base_dir is required for instantaneous mode.")
        base_dir = Path(base_dir)
        stats_path = base_dir / f'statistics/{num_frames}/Cam1/instantaneous/mean_stats/mean_stats.mat'
        output_dir = output_dir_override or (script_dir / 'benchmark_results')

    print("=" * 70)
    print(f"PIV BENCHMARK COMPARISON ({mode.upper()})")
    print("=" * 70)

    # Load data - auto-detect file names
    print("\n[1] Loading wall units...")
    wall_units_file = gt_dir / 'wall_units.mat'
    if not wall_units_file.exists():
        wall_units_file = gt_dir / 'diagnostics.mat'
    if not wall_units_file.exists():
        wall_units_file = gt_dir / 'direct_stats.mat'
    wall_units = load_wall_units(wall_units_file)
    print(f"  u_tau = {wall_units['u_tau']:.4f} mm/s")
    print(f"  nu = {wall_units['nu']:.4f} mm²/s")
    print(f"  delta_nu = {wall_units['delta_nu']:.4f} mm")
    print(f"  Re_tau = {wall_units['Re_tau']:.0f}")

    print("\n[2] Loading ground truth...")
    profiles_file = gt_dir / 'profiles.mat'
    if not profiles_file.exists():
        profiles_file = gt_dir / 'ensemble_statistics_full.mat'
    if not profiles_file.exists():
        profiles_file = gt_dir / 'direct_stats.mat'
    gt = load_ground_truth(profiles_file, wall_units_path=wall_units_file)
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
    piv_plus['y_plus'] = piv_plus['y_plus'] + 1.0  # shift y+ by +1
    print(f"  Aligned y range: {piv_plus['y_mm'].min():.2f} to {piv_plus['y_mm'].max():.2f} mm")
    print(f"  y+ range: {piv_plus['y_plus'].min():.1f} to {piv_plus['y_plus'].max():.1f} (y+ +1 applied)")
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
    # Thread CI bounds through if available
    for ci_key in ['U_plus_ci_lo', 'U_plus_ci_hi', 'V_plus_ci_lo', 'V_plus_ci_hi',
                   'uu_plus_ci_lo', 'uu_plus_ci_hi', 'vv_plus_ci_lo', 'vv_plus_ci_hi',
                   'uv_plus_ci_lo', 'uv_plus_ci_hi']:
        if ci_key in gt:
            gt_plus[ci_key] = gt[ci_key]

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
    plot_comparison(piv_plus, gt_plus, wall_units, errors, output_dir,
                    show_fit_lines=show_fit_lines)
    plot_combined_stresses(piv_plus, gt_plus, wall_units, errors, output_dir)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


def plot_combined_comparison(all_results, gt_plus, wall_units, output_dir, show_fit_lines=False):
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

    # Check for CI data
    has_ci = 'uu_plus_ci_lo' in gt_plus

    # ==========================================================================
    # Figure 1: Mean velocity profile (semi-log) - ALL WINDOWS
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    # Ground truth with CI band
    if has_ci and 'U_plus_ci_lo' in gt_plus:
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['U_plus_ci_lo'],
                     gt_plus['U_plus_ci_hi'], color='k', alpha=0.15, zorder=1)
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

    if show_fit_lines:
        y_log = np.logspace(1, np.log10(Re_tau), 100)
        kappa, B = 0.41, 5.2
        ax.semilogx(y_log, (1/kappa)*np.log(y_log)+B, 'b--', linewidth=1.5, alpha=0.5,
                    label=r'Log law: $U^+ = \frac{1}{\kappa}\ln(y^+) + B$')
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
    if has_ci:
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['uu_plus_ci_lo'],
                     gt_plus['uu_plus_ci_hi'], color='k', zorder=1)
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
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # vv+
    ax = axes[1]
    if has_ci:
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['vv_plus_ci_lo'],
                     gt_plus['vv_plus_ci_hi'], color='k', zorder=1)
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
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # -uv+
    ax = axes[2]
    if has_ci:
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['uv_plus_ci_lo'],
                     gt_plus['uv_plus_ci_hi'], sign=-1, color='k', zorder=1)
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
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'reynolds_stresses_combined.png', dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Figure 3: V+ profile - ALL WINDOWS
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    if has_ci and 'V_plus_ci_lo' in gt_plus:
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['V_plus_ci_lo'],
                     gt_plus['V_plus_ci_hi'], color='k', zorder=1)
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
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
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
    ax.set_title('Mean Velocity Profile - All Window Sizes', fontsize=16)
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'U_plus_linear_combined.png', dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Figure 5: Smoothed U+ combined - ALL WINDOWS
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    if has_ci and 'U_plus_ci_lo' in gt_plus:
        plot_ci_band(ax, gt_plus['y_plus'], gt_plus['U_plus_ci_lo'],
                     gt_plus['U_plus_ci_hi'], color='k', alpha=0.15, zorder=1)
    ax.semilogx(gt_plus['y_plus'], gt_plus['U_plus'], 'k-',
                linewidth=2.5, label='DNS (1px)', zorder=10)

    for i, res in enumerate(all_results):
        piv_plus = res['piv_plus']
        label = res['window_label']
        c = colors[i % len(colors)]
        ax.semilogx(piv_plus['y_plus'], piv_plus['U_plus'],
                    color=c, marker=markers[i % len(markers)],
                    markersize=4, alpha=0.7, linestyle='none',
                    label=f'PIV ({label})', zorder=5-i*0.1)

    if show_fit_lines:
        y_log = np.logspace(1, np.log10(Re_tau), 100)
        kappa, B = 0.41, 5.2
        ax.semilogx(y_log, (1/kappa)*np.log(y_log)+B, 'b--', linewidth=1.5, alpha=0.5,
                    label='Log law')
        y_visc = np.linspace(0.1, 10, 50)
        ax.semilogx(y_visc, y_visc, 'g--', linewidth=1.5, alpha=0.5, label=r'$U^+=y^+$')

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$U^+$', fontsize=14)
    ax.set_title(f'Mean Velocity Profile - Smoothed (Re$_\\tau$ = {Re_tau:.0f})', fontsize=16)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(1, Re_tau)
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'U_plus_profile_combined_smooth.png', dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Figure 6: Smoothed Reynolds stresses combined - ALL WINDOWS
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    stress_configs = [
        ('uu_plus', r"$\overline{u'u'}^+$", 'Streamwise Normal Stress', 1),
        ('vv_plus', r"$\overline{v'v'}^+$", 'Wall-Normal Normal Stress', 1),
        ('uv_plus', r"$-\overline{u'v'}^+$", 'Reynolds Shear Stress', -1),
    ]
    for ax, (var, ylabel, title, sign) in zip(axes, stress_configs):
        gt_vals = sign * gt_plus[var]
        ci_lo_key = f'{var}_ci_lo'
        ci_hi_key = f'{var}_ci_hi'
        if has_ci and ci_lo_key in gt_plus:
            plot_ci_band(ax, gt_plus['y_plus'], gt_plus[ci_lo_key],
                         gt_plus[ci_hi_key], sign=sign, color='k', zorder=1)
        ax.plot(gt_plus['y_plus'], gt_vals, 'k-', linewidth=2.5, label='DNS', zorder=10)

        for i, res in enumerate(all_results):
            piv_plus = res['piv_plus']
            label = res['window_label']
            c = colors[i % len(colors)]
            piv_vals = sign * piv_plus[var]
            ax.plot(piv_plus['y_plus'], piv_vals,
                    color=c, marker=markers[i % len(markers)],
                    markersize=4, alpha=0.7, linestyle='none',
                    label=f'PIV ({label})', zorder=5-i*0.1)

        ax.set_xlabel(r'$y^+$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=9)
        ax.set_xscale('log')
        ax.set_xlim(1, Re_tau)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'reynolds_stresses_combined_smooth.png', dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Figure 7: Trace invariant (u'u' + v'v') - ALL WINDOWS
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Compute ground truth trace
    gt_trace = gt_plus['uu_plus'] + gt_plus['vv_plus']

    # Left: Trace comparison
    ax = axes[0]
    ax.semilogx(gt_plus['y_plus'], gt_trace, 'k-', linewidth=2.5, label='DNS', zorder=10)
    for i, res in enumerate(all_results):
        piv_plus = res['piv_plus']
        label = res['window_label']
        piv_trace = piv_plus['uu_plus'] + piv_plus['vv_plus']
        ax.semilogx(piv_plus['y_plus'], piv_trace,
                    color=colors[i % len(colors)], marker=markers[i % len(markers)],
                    markersize=4, alpha=0.7, linestyle='none', label=f'PIV ({label})')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{u'u'}^+ + \overline{v'v'}^+$", fontsize=12)
    ax.set_title("Trace Invariant (rotation-invariant)", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # Right: Ratio of components (rotation indicator)
    ax = axes[1]
    gt_ratio = gt_plus['uu_plus'] / (gt_plus['vv_plus'] + 1e-10)
    ax.semilogx(gt_plus['y_plus'], gt_ratio, 'k-', linewidth=2.5, label='DNS', zorder=10)
    for i, res in enumerate(all_results):
        piv_plus = res['piv_plus']
        label = res['window_label']
        piv_ratio = piv_plus['uu_plus'] / (piv_plus['vv_plus'] + 1e-10)
        ax.semilogx(piv_plus['y_plus'], piv_ratio,
                    color=colors[i % len(colors)], marker=markers[i % len(markers)],
                    markersize=4, alpha=0.7, linestyle='none', label=f'PIV ({label})')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{u'u'}^+ / \overline{v'v'}^+$", fontsize=12)
    ax.set_title("Stress Ratio (rotation indicator)", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(1, Re_tau)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Rotation Diagnostic: If trace matches but ratio differs, rotation problem exists", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'trace_invariant_combined.png', dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Figure 8: Residuals (PIV - Ref) vs y+ - ALL WINDOWS
    # ==========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Interpolate ground truth
    gt_interp_fn = {}
    for var in ['U_plus', 'V_plus', 'uu_plus', 'vv_plus', 'uv_plus']:
        gt_interp_fn[var] = interp1d(gt_plus['y_plus'], gt_plus[var], kind='linear',
                                     bounds_error=False, fill_value=np.nan)

    # Top row: velocity residuals
    vel_configs = [
        ('U_plus', r"$U^+_{\mathrm{PIV}} - U^+_{\mathrm{Ref}}$",
         'Mean Streamwise Velocity Residual', 1),
        ('V_plus', r"$V^+_{\mathrm{PIV}} - V^+_{\mathrm{Ref}}$",
         'Mean Wall-Normal Velocity Residual', 1),
    ]
    for ax, (var, ylabel, title, sign) in zip(axes[0, :2], vel_configs):
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
        for i, res in enumerate(all_results):
            piv_plus = res['piv_plus']
            label = res['window_label']
            c = colors[i % len(colors)]
            gt_at_piv = gt_interp_fn[var](piv_plus['y_plus'])
            residual = sign * piv_plus[var] - sign * gt_at_piv
            ax.plot(piv_plus['y_plus'], residual,
                    color=c, marker=markers[i % len(markers)],
                    markersize=2, alpha=0.15, linestyle='none', zorder=2)
            yp_s, r_s = log_smooth(piv_plus['y_plus'], residual)
            ax.plot(yp_s, r_s, color=c, linewidth=2,
                    label=f'PIV ({label})', zorder=5-i*0.1)
        ax.set_xlabel(r'$y^+$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=9)
        ax.set_xscale('log')
        ax.set_xlim(1, Re_tau)
        ax.grid(True, alpha=0.3)

    axes[0, 2].set_visible(False)  # Empty top-right panel

    # Bottom row: stress residuals
    stress_configs = [
        ('uu_plus', r"$\overline{u'u'}^+_{\mathrm{PIV}} - \overline{u'u'}^+_{\mathrm{Ref}}$",
         'Streamwise Normal Stress Residual', 1),
        ('vv_plus', r"$\overline{v'v'}^+_{\mathrm{PIV}} - \overline{v'v'}^+_{\mathrm{Ref}}$",
         'Wall-Normal Normal Stress Residual', 1),
        ('uv_plus', r"$-\overline{u'v'}^+_{\mathrm{PIV}} - (-\overline{u'v'}^+_{\mathrm{Ref}})$",
         'Shear Stress Residual', -1),
    ]
    for ax, (var, ylabel, title, sign) in zip(axes[1, :], stress_configs):
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
        for i, res in enumerate(all_results):
            piv_plus = res['piv_plus']
            label = res['window_label']
            c = colors[i % len(colors)]
            gt_at_piv = gt_interp_fn[var](piv_plus['y_plus'])
            residual = sign * piv_plus[var] - sign * gt_at_piv
            ax.plot(piv_plus['y_plus'], residual,
                    color=c, marker=markers[i % len(markers)],
                    markersize=2, alpha=0.15, linestyle='none', zorder=2)
            yp_s, r_s = log_smooth(piv_plus['y_plus'], residual)
            ax.plot(yp_s, r_s, color=c, linewidth=2,
                    label=f'PIV ({label})', zorder=5-i*0.1)
        ax.set_xlabel(r'$y^+$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=9)
        ax.set_xscale('log')
        ax.set_xlim(1, Re_tau)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'residuals_combined.png', dpi=150)
    plt.close(fig)

    # ==========================================================================
    # Figure 9: Noise floor vs gradient correction decomposition - ALL WINDOWS
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: Noise floor (v'v' residual) — all windows
    ax = axes[0]
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    for i, res in enumerate(all_results):
        piv_plus = res['piv_plus']
        label = res['window_label']
        c = colors[i % len(colors)]
        vv_res = piv_plus['vv_plus'] - gt_interp_fn['vv_plus'](piv_plus['y_plus'])
        ax.plot(piv_plus['y_plus'], vv_res,
                color=c, marker=markers[i % len(markers)],
                markersize=2, alpha=0.15, linestyle='none', zorder=2)
        yp_s, r_s = log_smooth(piv_plus['y_plus'], vv_res)
        ax.plot(yp_s, r_s, color=c, linewidth=2,
                label=f'PIV ({label})', zorder=5-i*0.1)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{v'v'}^+_{\mathrm{PIV}} - \overline{v'v'}^+_{\mathrm{Ref}}$", fontsize=12)
    ax.set_title('Noise Floor (isotropic)', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # Middle: Gradient-only residual (u'u' - v'v') — all windows
    ax = axes[1]
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    for i, res in enumerate(all_results):
        piv_plus = res['piv_plus']
        label = res['window_label']
        c = colors[i % len(colors)]
        uu_res = piv_plus['uu_plus'] - gt_interp_fn['uu_plus'](piv_plus['y_plus'])
        vv_res = piv_plus['vv_plus'] - gt_interp_fn['vv_plus'](piv_plus['y_plus'])
        grad_only = uu_res - vv_res
        ax.plot(piv_plus['y_plus'], grad_only,
                color=c, marker=markers[i % len(markers)],
                markersize=2, alpha=0.15, linestyle='none', zorder=2)
        yp_s, r_s = log_smooth(piv_plus['y_plus'], grad_only)
        ax.plot(yp_s, r_s, color=c, linewidth=2,
                label=f'PIV ({label})', zorder=5-i*0.1)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"Gradient-only residual$^+$", fontsize=12)
    ax.set_title(r"Gradient Correction Residual ($u'u' - v'v'$ removes noise)", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # Right: All three overlaid for the finest window
    ax = axes[2]
    finest = all_results[-1]  # Last (finest) window
    piv_plus = finest['piv_plus']
    label = finest['window_label']
    uu_res = piv_plus['uu_plus'] - gt_interp_fn['uu_plus'](piv_plus['y_plus'])
    vv_res = piv_plus['vv_plus'] - gt_interp_fn['vv_plus'](piv_plus['y_plus'])
    grad_only = uu_res - vv_res

    ax.plot(piv_plus['y_plus'], uu_res, 'ro', markersize=2, alpha=0.1)
    yp_s, r_s = log_smooth(piv_plus['y_plus'], uu_res)
    ax.plot(yp_s, r_s, 'r-', linewidth=2, label=r"$u'u'$ residual (total)")

    ax.plot(piv_plus['y_plus'], vv_res, 'bo', markersize=2, alpha=0.1)
    yp_s, r_s = log_smooth(piv_plus['y_plus'], vv_res)
    ax.plot(yp_s, r_s, 'b-', linewidth=2, label=r"$v'v'$ residual (noise floor)")

    yp_s, r_s = log_smooth(piv_plus['y_plus'], grad_only)
    ax.plot(yp_s, r_s, 'g--', linewidth=2, label=r"$u'u' - v'v'$ residual (gradient only)")

    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"Residual$^+$", fontsize=12)
    ax.set_title(f'Decomposition ({label}): Total = Noise + Gradient', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'noise_gradient_decomposition_combined.png', dpi=150)
    plt.close(fig)

    print(f"\nCombined plots saved to: {output_dir}")


def main_multi_run(mode='ensemble', run_indices=None, window_sizes=None, run_labels=None, gt_dir=None, base_dir=None, ensemble_dir=None, y_plus_offset=0.0, num_frames=1000, output_dir_override=None, show_fit_lines=False):
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
    ensemble_dir : Path, optional
        Direct path to ensemble directory containing ensemble_result.mat and coordinates.mat.
        If provided, overrides base_dir for ensemble mode.
    y_plus_offset : float, optional
        Offset to add to y+ coordinates (for calibration correction)
    """
    # Paths
    script_dir = Path(__file__).parent
    if gt_dir is None:
        raise ValueError("gt_dir is required. Please provide the ground truth directory path.")
    gt_dir = Path(gt_dir)

    if mode == 'ensemble':
        if ensemble_dir is not None:
            ensemble_dir = Path(ensemble_dir)
            ensemble_path = ensemble_dir / 'ensemble_result.mat'
            coords_path = ensemble_dir / 'coordinates.mat'
        elif base_dir is not None:
            base_dir = Path(base_dir)
            ensemble_path = base_dir / f'calibrated_piv/{num_frames}/Cam1/ensemble/ensemble_result.mat'
            coords_path = base_dir / f'calibrated_piv/{num_frames}/Cam1/ensemble/coordinates.mat'
        else:
            raise ValueError("Either ensemble_dir or base_dir must be provided for ensemble mode.")
        output_dir = output_dir_override or (script_dir / 'benchmark_results_ensemble')
    else:
        if base_dir is None:
            raise ValueError("base_dir is required for instantaneous mode.")
        base_dir = Path(base_dir)
        stats_path = base_dir / f'statistics/{num_frames}/Cam1/instantaneous/mean_stats/mean_stats.mat'
        output_dir = output_dir_override or (script_dir / 'benchmark_results')

    print("=" * 70)
    print(f"PIV BENCHMARK COMPARISON ({mode.upper()}) - MULTI-RUN")
    print("=" * 70)
    print(f"Processing runs: {run_indices}")
    print(f"Window sizes: {window_sizes}")

    # Load common data - auto-detect file names
    print("\n[1] Loading wall units...")
    wall_units_file = gt_dir / 'wall_units.mat'
    if not wall_units_file.exists():
        wall_units_file = gt_dir / 'diagnostics.mat'
    if not wall_units_file.exists():
        wall_units_file = gt_dir / 'direct_stats.mat'
    wall_units = load_wall_units(wall_units_file)
    print(f"  u_tau = {wall_units['u_tau']:.4f} mm/s")
    print(f"  nu = {wall_units['nu']:.4f} mm²/s")
    print(f"  delta_nu = {wall_units['delta_nu']:.4f} mm")
    print(f"  Re_tau = {wall_units['Re_tau']:.0f}")

    print("\n[2] Loading ground truth...")
    profiles_file = gt_dir / 'profiles.mat'
    if not profiles_file.exists():
        profiles_file = gt_dir / 'ensemble_statistics_full.mat'
    if not profiles_file.exists():
        profiles_file = gt_dir / 'direct_stats.mat'
    gt = load_ground_truth(profiles_file, wall_units_path=wall_units_file)
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
    # Thread CI bounds through if available
    for ci_key in ['U_plus_ci_lo', 'U_plus_ci_hi', 'V_plus_ci_lo', 'V_plus_ci_hi',
                   'uu_plus_ci_lo', 'uu_plus_ci_hi', 'vv_plus_ci_lo', 'vv_plus_ci_hi',
                   'uv_plus_ci_lo', 'uv_plus_ci_hi']:
        if ci_key in gt:
            gt_plus[ci_key] = gt[ci_key]

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
            piv_plus['y_plus'] = piv_plus['y_plus'] + 1.0  # shift y+ by +1
            # Apply additional y+ offset if specified
            if y_plus_offset != 0.0:
                piv_plus['y_plus'] = piv_plus['y_plus'] + y_plus_offset
                print(f"  y+ offset applied: {y_plus_offset:+.1f}")
            print(f"  y+ range: {piv_plus['y_plus'].min():.1f} to {piv_plus['y_plus'].max():.1f} (y+ +1 applied)")

            # Compute errors
            errors = compute_errors(piv_plus, gt_plus, y_plus_range=(10, 500))

            # Print error summary
            if 'U_plus' in errors:
                print(f"  U+ R² = {errors['U_plus']['r2']:.4f}, RMS = {errors['U_plus']['rms_rel']:.1f}%")
            if 'uu_plus' in errors:
                print(f"  uu+ R² = {errors['uu_plus']['r2']:.4f}")

            # Generate individual plots
            plot_comparison(piv_plus, gt_plus, wall_units, errors, run_output_dir,
                          window_label=window_label, show_fit_lines=show_fit_lines)
            plot_combined_stresses(piv_plus, gt_plus, wall_units, errors, run_output_dir,
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
        plot_combined_comparison(all_results, gt_plus, wall_units, output_dir,
                                show_fit_lines=show_fit_lines)

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
    parser.add_argument('--gt-dir', '-g', type=str, required=True,
                        help='Ground truth directory path (required)')
    parser.add_argument('--base-dir', '-b', type=str, default=None,
                        help='Base directory containing PIV results')
    parser.add_argument('--ensemble-dir', '-e', type=str, default=None,
                        help='Direct path to ensemble directory containing ensemble_result.mat and coordinates.mat')
    parser.add_argument('--y-plus-offset', '-y', type=float, default=0.0,
                        help='Offset to add to y+ coordinates (calibration correction)')
    parser.add_argument('--num-frames', '-n', type=int, default=1000,
                        help='Frame count subdirectory in paths (default: 1000)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Custom output directory for results')
    parser.add_argument('--show-fit-lines', action='store_true', default=False,
                        help='Show log-law and viscous sublayer reference lines on U+ plots')
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    base_dir = Path(args.base_dir) if args.base_dir else None
    ensemble_dir = Path(args.ensemble_dir) if args.ensemble_dir else None
    output_dir_override = Path(args.output_dir) if args.output_dir else None

    if args.runs and args.windows:
        run_indices = [int(r) for r in args.runs.split(',')]
        window_sizes = [int(w) for w in args.windows.split(',')]
        run_labels = args.labels.split(',') if args.labels else None
        main_multi_run(mode=args.mode, run_indices=run_indices, window_sizes=window_sizes,
                       run_labels=run_labels, gt_dir=gt_dir, base_dir=base_dir,
                       ensemble_dir=ensemble_dir, y_plus_offset=args.y_plus_offset,
                       num_frames=args.num_frames, output_dir_override=output_dir_override,
                       show_fit_lines=args.show_fit_lines)
    else:
        main(mode=args.mode, gt_dir=gt_dir, base_dir=base_dir, ensemble_dir=ensemble_dir,
             num_frames=args.num_frames, output_dir_override=output_dir_override,
             show_fit_lines=args.show_fit_lines)
