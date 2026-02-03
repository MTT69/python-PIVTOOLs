#!/usr/bin/env python3
"""
Compare two ground truth datasets from JHTDB ensemble statistics.

Compares:
- Original ground truth (1-pass)
- New 2-pass ground truth

For all window sizes: 1px, 2px, 4px, 6px, 8px, 16px
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path


def load_profiles(profiles_path):
    """Load profiles from .mat file."""
    data = sio.loadmat(profiles_path, squeeze_me=True, struct_as_record=False)
    profiles = data['profiles']

    result = {}
    for win_name in ['win_1px', 'win_2px', 'win_4px', 'win_6px', 'win_8px', 'win_16px']:
        if hasattr(profiles, win_name):
            win = getattr(profiles, win_name)
            result[win_name] = {
                'y_mm': win.y_mm,
                'y_plus': win.y_plus,
                'U': win.U,
                'V': win.V,
                'U_plus': win.U_plus,
                'uu': win.uu,
                'vv': win.vv,
                'uv': win.uv,
                'uu_plus': win.uu_plus,
                'vv_plus': win.vv_plus,
                'uv_plus': win.uv_plus,
            }
    return result


def load_wall_units(wall_units_path):
    """Load wall units from .mat file."""
    wall = sio.loadmat(wall_units_path, squeeze_me=True, struct_as_record=False)
    wu = wall['wall_units']
    return {
        'u_tau': float(wu.u_tau),
        'nu': float(wu.nu),
        'delta_nu': float(wu.delta_nu),
        'h_mm': float(wu.h_mm),
        'Re_tau': float(wu.Re_tau)
    }


def compute_errors(prof1, prof2, y_plus_range=(10, 500)):
    """Compute error metrics between two profiles."""
    y1, y2 = prof1['y_plus'], prof2['y_plus']

    # Use the common y+ values (should be identical)
    mask1 = (y1 >= y_plus_range[0]) & (y1 <= y_plus_range[1])
    mask2 = (y2 >= y_plus_range[0]) & (y2 <= y_plus_range[1])

    errors = {}
    for var in ['U_plus', 'uu_plus', 'vv_plus', 'uv_plus']:
        v1 = prof1[var][mask1]
        v2 = prof2[var][mask2]

        # Interpolate if needed (in case grids differ slightly)
        if len(v1) != len(v2):
            from scipy.interpolate import interp1d
            f = interp1d(y2[mask2], v2, kind='linear', fill_value='extrapolate')
            v2 = f(y1[mask1])

        diff = v1 - v2
        rms = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))

        # Relative to range
        range_val = np.ptp(v1)
        rms_rel = (rms / range_val * 100) if range_val > 0 else np.nan

        # Correlation
        corr = np.corrcoef(v1, v2)[0, 1] if len(v1) > 1 else np.nan

        errors[var] = {
            'rms': rms,
            'rms_rel': rms_rel,
            'mae': mae,
            'corr': corr,
        }

    return errors


def plot_comparison(prof1, prof2, wall_units, output_dir, win_label='1px'):
    """Generate comparison plots for a single window size."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Re_tau = wall_units['Re_tau']

    # Figure 1: Mean velocity U+ (semi-log)
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.semilogx(prof1['y_plus'], prof1['U_plus'], 'b-', linewidth=2,
                label='Original (1-pass)', zorder=3)
    ax.semilogx(prof2['y_plus'], prof2['U_plus'], 'r--', linewidth=2,
                label='New (2-pass)', zorder=2)

    # Log law reference
    y_log = np.logspace(1, np.log10(Re_tau), 100)
    kappa, B = 0.41, 5.2
    U_log = (1/kappa) * np.log(y_log) + B
    ax.semilogx(y_log, U_log, 'k:', linewidth=1, alpha=0.5, label='Log law')

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$U^+$', fontsize=14)
    ax.set_title(f'Mean Velocity Profile Comparison ({win_label})', fontsize=16)
    ax.legend(fontsize=11)
    ax.set_xlim(1, Re_tau)
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f'U_plus_comparison_{win_label}.png', dpi=150)
    plt.close(fig)

    # Figure 2: Reynolds stresses (semi-log)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # uu+
    ax = axes[0]
    ax.semilogx(prof1['y_plus'], prof1['uu_plus'], 'b-', linewidth=2, label='Original')
    ax.semilogx(prof2['y_plus'], prof2['uu_plus'], 'r--', linewidth=2, label='2-pass')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{u'u'}^+$", fontsize=12)
    ax.set_title('Streamwise Normal Stress', fontsize=14)
    ax.legend()
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # vv+
    ax = axes[1]
    ax.semilogx(prof1['y_plus'], prof1['vv_plus'], 'b-', linewidth=2, label='Original')
    ax.semilogx(prof2['y_plus'], prof2['vv_plus'], 'r--', linewidth=2, label='2-pass')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\overline{v'v'}^+$", fontsize=12)
    ax.set_title('Wall-Normal Normal Stress', fontsize=14)
    ax.legend()
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # -uv+
    ax = axes[2]
    ax.semilogx(prof1['y_plus'], -prof1['uv_plus'], 'b-', linewidth=2, label='Original')
    ax.semilogx(prof2['y_plus'], -prof2['uv_plus'], 'r--', linewidth=2, label='2-pass')
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$-\overline{u'v'}^+$", fontsize=12)
    ax.set_title('Reynolds Shear Stress', fontsize=14)
    ax.legend()
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f'reynolds_stresses_comparison_{win_label}.png', dpi=150)
    plt.close(fig)

    # Figure 3: Difference plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # U+ difference
    ax = axes[0, 0]
    diff_U = prof2['U_plus'] - prof1['U_plus']
    ax.semilogx(prof1['y_plus'], diff_U, 'k-', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r'$\Delta U^+$ (2-pass - Original)', fontsize=12)
    ax.set_title('U+ Difference', fontsize=14)
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # uu+ difference
    ax = axes[0, 1]
    diff_uu = prof2['uu_plus'] - prof1['uu_plus']
    ax.semilogx(prof1['y_plus'], diff_uu, 'k-', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\Delta \overline{u'u'}^+$", fontsize=12)
    ax.set_title('uu+ Difference', fontsize=14)
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # vv+ difference
    ax = axes[1, 0]
    diff_vv = prof2['vv_plus'] - prof1['vv_plus']
    ax.semilogx(prof1['y_plus'], diff_vv, 'k-', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\Delta \overline{v'v'}^+$", fontsize=12)
    ax.set_title('vv+ Difference', fontsize=14)
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    # uv+ difference
    ax = axes[1, 1]
    diff_uv = prof2['uv_plus'] - prof1['uv_plus']
    ax.semilogx(prof1['y_plus'], diff_uv, 'k-', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$y^+$', fontsize=12)
    ax.set_ylabel(r"$\Delta \overline{u'v'}^+$", fontsize=12)
    ax.set_title('uv+ Difference', fontsize=14)
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f'differences_{win_label}.png', dpi=150)
    plt.close(fig)


def plot_all_windows_comparison(profiles1, profiles2, wall_units, output_dir):
    """Generate combined comparison plots for all window sizes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Re_tau = wall_units['Re_tau']
    window_labels = ['1px', '2px', '4px', '6px', '8px', '16px']
    window_keys = ['win_1px', 'win_2px', 'win_4px', 'win_6px', 'win_8px', 'win_16px']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Figure: U+ for all windows
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Original
    ax = axes[0]
    for i, (key, label) in enumerate(zip(window_keys, window_labels)):
        if key in profiles1:
            ax.semilogx(profiles1[key]['y_plus'], profiles1[key]['U_plus'],
                       color=colors[i], linewidth=1.5, label=label)
    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$U^+$', fontsize=14)
    ax.set_title('Original (1-pass)', fontsize=16)
    ax.legend(fontsize=10, title='Window')
    ax.set_xlim(1, Re_tau)
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)

    # 2-pass
    ax = axes[1]
    for i, (key, label) in enumerate(zip(window_keys, window_labels)):
        if key in profiles2:
            ax.semilogx(profiles2[key]['y_plus'], profiles2[key]['U_plus'],
                       color=colors[i], linewidth=1.5, label=label)
    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'$U^+$', fontsize=14)
    ax.set_title('New (2-pass)', fontsize=16)
    ax.legend(fontsize=10, title='Window')
    ax.set_xlim(1, Re_tau)
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Mean Velocity Profiles - All Window Sizes', fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'U_plus_all_windows.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Figure: uu+ for all windows
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for i, (key, label) in enumerate(zip(window_keys, window_labels)):
        if key in profiles1:
            ax.semilogx(profiles1[key]['y_plus'], profiles1[key]['uu_plus'],
                       color=colors[i], linewidth=1.5, label=label)
    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r"$\overline{u'u'}^+$", fontsize=14)
    ax.set_title('Original (1-pass)', fontsize=16)
    ax.legend(fontsize=10, title='Window')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, (key, label) in enumerate(zip(window_keys, window_labels)):
        if key in profiles2:
            ax.semilogx(profiles2[key]['y_plus'], profiles2[key]['uu_plus'],
                       color=colors[i], linewidth=1.5, label=label)
    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r"$\overline{u'u'}^+$", fontsize=14)
    ax.set_title('New (2-pass)', fontsize=16)
    ax.legend(fontsize=10, title='Window')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Streamwise Normal Stress - All Window Sizes', fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'uu_plus_all_windows.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """Main comparison function."""
    # Paths
    script_dir = Path(__file__).parent
    gt1_dir = script_dir / 'ground_truth' / 'ensemble_statistics'
    gt2_dir = Path('/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/'
                   'Documents/#current_processing/query_JHTDB/download_from_jhtdb/'
                   'ensemble_statistics_2pass')
    output_dir = script_dir / 'ground_truth_comparison'

    print("=" * 70)
    print("GROUND TRUTH COMPARISON: Original vs 2-Pass")
    print("=" * 70)

    # Load wall units (should be identical)
    print("\n[1] Loading wall units...")
    wall_units = load_wall_units(gt1_dir / 'wall_units.mat')
    print(f"  Re_tau = {wall_units['Re_tau']:.0f}")

    # Load profiles
    print("\n[2] Loading original ground truth (1-pass)...")
    profiles1 = load_profiles(gt1_dir / 'profiles.mat')
    print(f"  Windows loaded: {list(profiles1.keys())}")

    print("\n[3] Loading new ground truth (2-pass)...")
    profiles2 = load_profiles(gt2_dir / 'profiles.mat')
    print(f"  Windows loaded: {list(profiles2.keys())}")

    # Compare each window size
    print("\n[4] Computing differences...")
    print("\n" + "=" * 70)
    print("ERROR SUMMARY (y+ = 10-500)")
    print("=" * 70)
    print(f"\n{'Window':<10} {'U+ RMS%':<12} {'uu+ RMS%':<12} {'vv+ RMS%':<12} {'-uv+ RMS%':<12}")
    print("-" * 58)

    window_keys = ['win_1px', 'win_2px', 'win_4px', 'win_6px', 'win_8px', 'win_16px']
    window_labels = ['1px', '2px', '4px', '6px', '8px', '16px']

    for key, label in zip(window_keys, window_labels):
        if key in profiles1 and key in profiles2:
            errors = compute_errors(profiles1[key], profiles2[key])
            u_rms = errors['U_plus']['rms_rel']
            uu_rms = errors['uu_plus']['rms_rel']
            vv_rms = errors['vv_plus']['rms_rel']
            uv_rms = errors['uv_plus']['rms_rel']
            print(f"{label:<10} {u_rms:<12.2f} {uu_rms:<12.2f} {vv_rms:<12.2f} {uv_rms:<12.2f}")

    # Generate plots for each window
    print("\n[5] Generating comparison plots...")
    for key, label in zip(window_keys, window_labels):
        if key in profiles1 and key in profiles2:
            plot_comparison(profiles1[key], profiles2[key], wall_units,
                          output_dir / label, win_label=label)
            print(f"  {label}: plots saved")

    # Generate combined plots
    print("\n[6] Generating combined comparison plots...")
    plot_all_windows_comparison(profiles1, profiles2, wall_units, output_dir)

    print(f"\nAll plots saved to: {output_dir}")
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
