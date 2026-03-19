#!/usr/bin/env python3
"""
Cross-method PIV comparison: Instantaneous vs Ensemble vs Stereo.

Publication-quality plots comparing one pass from each method against DNS
ground truth. Designed for academic journal submission.

Produces:
  1. mean_velocity_comparison.png   — U+ vs y+ (semi-log)
  2. stresses_comparison.png        — 1×3 subplots (uu+, vv+, −uv+)
  3. combined_stresses_comparison.png — all stresses on one axis

Usage (Python API):
    from dns_validation.cross_method_comparison import compare_methods
    compare_methods(
        gt_dir='path/to/ground_truth',
        inst_stats_path='path/to/mean_stats.mat',
        ens_ensemble_path='path/to/ensemble_result.mat',
        ens_coords_path='path/to/coordinates.mat',
        stereo_stats_path='path/to/stereo/mean_stats.mat',
        output_dir='path/to/output',
        ...
    )
"""

import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Publication-quality font setup ────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'text.usetex': False,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ── Okabe–Ito colourblind-safe palette ────────────────────────────────────────
METHOD_STYLES = {
    'Instantaneous': {'color': '#0072B2', 'marker': 'o'},   # blue
    'Ensemble':      {'color': '#D55E00', 'marker': 's'},   # vermillion
    'Stereo':        {'color': '#009E73', 'marker': '^'},   # teal
}
DNS_COLOR = 'k'


def _get_style(name):
    """Look up style by method name prefix (e.g. 'Instantaneous (16x16)' → Instantaneous)."""
    for key in METHOD_STYLES:
        if name.startswith(key):
            return METHOD_STYLES[key]
    return {'color': 'gray', 'marker': 'x'}


# =============================================================================
# Data loading helpers (reuse from benchmark_comparison where possible)
# =============================================================================

def _load_wall_units(gt_dir):
    """Load wall units from ground truth directory (auto-detect format)."""
    from dns_validation.benchmark_comparison import load_wall_units
    gt_dir = Path(gt_dir)
    for name in ('wall_units.mat', 'diagnostics.mat', 'direct_stats.mat'):
        p = gt_dir / name
        if p.exists():
            return load_wall_units(p)
    raise FileNotFoundError(f"No wall-units file found in {gt_dir}")


def _load_ground_truth(gt_dir):
    """Load ground truth profiles + wall units."""
    from dns_validation.benchmark_comparison import load_ground_truth
    gt_dir = Path(gt_dir)
    wu = _load_wall_units(gt_dir)

    for name in ('profiles.mat', 'ensemble_statistics_full.mat', 'direct_stats.mat'):
        p = gt_dir / name
        if p.exists():
            gt = load_ground_truth(p, wall_units_path=gt_dir / 'direct_stats.mat')
            break
    else:
        raise FileNotFoundError(f"No ground-truth profiles in {gt_dir}")

    gt_plus = {
        'y_plus': gt['y_plus'],
        'U_plus': gt['U_plus'],
        'V_plus': gt['V'] / wu['u_tau'],
        'uu_plus': gt['uu_plus'],
        'vv_plus': gt['vv_plus'],
        'uv_plus': gt['uv_plus'],
    }
    # Thread CI bounds if available
    for key in ('U_plus_ci_lo', 'U_plus_ci_hi',
                'uu_plus_ci_lo', 'uu_plus_ci_hi',
                'vv_plus_ci_lo', 'vv_plus_ci_hi',
                'uv_plus_ci_lo', 'uv_plus_ci_hi'):
        if key in gt:
            gt_plus[key] = gt[key]
    return gt_plus, wu


def _profiles_from_regular_grid(piv_data, x_exclude_vectors=4):
    """X-averaged profiles from a regular (no NaN border) grid."""
    from dns_validation.benchmark_comparison import compute_piv_profiles
    return compute_piv_profiles(piv_data, x_exclude_vectors=x_exclude_vectors)


def _profiles_from_stereo_grid(piv_struct, coords_struct, x_exclude_vectors=4):
    """X-averaged profiles from a stereo grid with NaN borders."""
    x = coords_struct.x
    y = coords_struct.y

    # Find valid subgrid
    valid_cols = np.any(~np.isnan(y), axis=0)
    col_indices = np.where(valid_cols)[0]
    mid_col = col_indices[len(col_indices) // 2]

    y_col = y[:, mid_col]
    valid_rows = ~np.isnan(y_col)
    y_unique = y_col[valid_rows]

    # X exclusion
    x_start = col_indices[0] + x_exclude_vectors
    x_end = col_indices[-1] - x_exclude_vectors + 1
    x_mask = np.zeros(x.shape[1], dtype=bool)
    x_mask[x_start:x_end] = True

    def _avg(field):
        return np.nanmean(field[valid_rows][:, x_mask], axis=1)

    return {
        'y_mm': y_unique,
        'U': _avg(piv_struct.ux) * 1000,
        'V': _avg(piv_struct.uy) * 1000,
        'uu': _avg(piv_struct.uu) * 1e6,
        'vv': _avg(piv_struct.vv) * 1e6,
        'uv': _avg(piv_struct.uv) * 1e6,
    }


def _to_wall_units(profiles, wall_units, y_plus_offset):
    """Convert profiles to plus units with y-offset."""
    from dns_validation.benchmark_comparison import convert_to_wall_units
    y_offset_mm = -profiles['y_mm'].min()
    piv_plus = convert_to_wall_units(profiles, wall_units, y_offset_mm=y_offset_mm)
    piv_plus['y_plus'] = piv_plus['y_plus'] + 1.0 + y_plus_offset
    return piv_plus


def load_instantaneous(stats_path, run_idx, wall_units, y_plus_offset,
                       x_exclude=4):
    """Load instantaneous PIV and return wall-unit profiles."""
    from dns_validation.benchmark_comparison import load_piv_statistics
    piv = load_piv_statistics(Path(stats_path), run_idx=run_idx)
    profiles = _profiles_from_regular_grid(piv, x_exclude_vectors=x_exclude)
    return _to_wall_units(profiles, wall_units, y_plus_offset)


def load_ensemble(ensemble_path, coords_path, run_idx, wall_units,
                  y_plus_offset, x_exclude=4):
    """Load ensemble PIV and return wall-unit profiles."""
    from dns_validation.benchmark_comparison import load_ensemble_statistics
    piv = load_ensemble_statistics(Path(ensemble_path), Path(coords_path),
                                   run_idx=run_idx)
    profiles = _profiles_from_regular_grid(piv, x_exclude_vectors=x_exclude)
    return _to_wall_units(profiles, wall_units, y_plus_offset)


def load_stereo(stats_path, run_idx, wall_units, y_plus_offset,
                x_exclude=4, trim_top=0):
    """Load stereo PIV (NaN-aware) and return wall-unit profiles."""
    stats = sio.loadmat(str(stats_path), squeeze_me=True, struct_as_record=False)
    piv = stats['piv_result'][run_idx]
    coords = stats['coordinates'][run_idx]
    profiles = _profiles_from_stereo_grid(piv, coords, x_exclude_vectors=x_exclude)

    if trim_top > 0 and len(profiles['y_mm']) > trim_top:
        y = profiles['y_mm']
        # Trim from the high-y end
        if y[0] > y[-1]:
            sl = slice(trim_top, None)
        else:
            sl = slice(None, -trim_top)
        profiles = {k: v[sl] if isinstance(v, np.ndarray) else v
                    for k, v in profiles.items()}

    return _to_wall_units(profiles, wall_units, y_plus_offset)


# =============================================================================
# Plotting
# =============================================================================

def _ci_band(ax, y_plus, lo, hi, sign=1, **kwargs):
    """Shade a 95 % confidence-interval band."""
    if sign == -1:
        lo, hi = hi, lo
    ax.fill_between(y_plus, sign * lo, sign * hi,
                    color=DNS_COLOR, alpha=0.10, linewidth=0, **kwargs)


def plot_velocity_comparison(methods, gt_plus, wall_units, output_dir, title_suffix=''):
    """
    U+ vs y+ — one marker series per method, DNS as reference line.

    Parameters
    ----------
    methods : dict
        {method_name: piv_plus_dict}  (keys must be in METHOD_STYLES)
    gt_plus : dict
        Ground truth in wall units
    wall_units : dict
    output_dir : Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Re_tau = wall_units['Re_tau']

    fig, ax = plt.subplots(figsize=(7, 5))

    # DNS reference
    has_ci = 'U_plus_ci_lo' in gt_plus
    if has_ci:
        _ci_band(ax, gt_plus['y_plus'],
                 gt_plus['U_plus_ci_lo'], gt_plus['U_plus_ci_hi'])
    ax.semilogx(gt_plus['y_plus'], gt_plus['U_plus'], color=DNS_COLOR,
                linewidth=2, label='DNS', zorder=10)

    # PIV methods
    for name, piv_plus in methods.items():
        sty = _get_style(name)
        ax.semilogx(piv_plus['y_plus'], piv_plus['U_plus'],
                    color=sty['color'], marker=sty['marker'],
                    markersize=3.5, alpha=0.7, linestyle='none',
                    label=name, zorder=5)

    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$U^+$')
    title = 'Mean Velocity Profile Comparison with DNS'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title)
    ax.set_xlim(1, Re_tau)
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc='lower right', framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_dir / 'mean_velocity_comparison.png')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'mean_velocity_comparison.png'}")


def plot_stresses_subplots(methods, gt_plus, wall_units, output_dir, title_suffix=''):
    """
    1×3 subplot: uu+, vv+, −uv+ — one marker series per method per panel.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Re_tau = wall_units['Re_tau']
    has_ci = 'uu_plus_ci_lo' in gt_plus

    panels = [
        ('uu_plus', r"$\overline{u'u'}^+$", 1),
        ('vv_plus', r"$\overline{v'v'}^+$", 1),
        ('uv_plus', r"$-\overline{u'v'}^+$", -1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))

    for ax, (var, ylabel, sign) in zip(axes, panels):
        # CI band
        ci_lo_key, ci_hi_key = f'{var}_ci_lo', f'{var}_ci_hi'
        if has_ci and ci_lo_key in gt_plus:
            _ci_band(ax, gt_plus['y_plus'],
                     gt_plus[ci_lo_key], gt_plus[ci_hi_key], sign=sign)

        # DNS
        ax.plot(gt_plus['y_plus'], sign * gt_plus[var], color=DNS_COLOR,
                linewidth=1.8, label='DNS', zorder=10)

        # PIV methods
        for name, piv_plus in methods.items():
            sty = _get_style(name)
            ax.plot(piv_plus['y_plus'], sign * piv_plus[var],
                    color=sty['color'], marker=sty['marker'],
                    markersize=2.5, alpha=0.65, linestyle='none',
                    label=name, zorder=5)

        ax.set_xlabel(r'$y^+$')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.set_xlim(1, Re_tau)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    # Single shared legend below the title
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(methods) + 1,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)

    suptitle = 'Reynolds Stresses Method Comparison with DNS'
    if title_suffix:
        suptitle += f' ({title_suffix})'
    fig.suptitle(suptitle, y=1.08)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(output_dir / 'stresses_comparison.png')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'stresses_comparison.png'}")


def plot_combined_stresses(methods, gt_plus, wall_units, output_dir, title_suffix=''):
    """
    All stresses on one axis — line styles distinguish components,
    colours distinguish methods.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Re_tau = wall_units['Re_tau']
    has_ci = 'uu_plus_ci_lo' in gt_plus

    component_styles = {
        'uu_plus': {'linestyle': '-',  'label_tex': r"$\overline{u'u'}^+$",  'sign': 1},
        'vv_plus': {'linestyle': '--', 'label_tex': r"$\overline{v'v'}^+$",  'sign': 1},
        'uv_plus': {'linestyle': ':',  'label_tex': r"$-\overline{u'v'}^+$", 'sign': -1},
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    # ── DNS reference lines ──────────────────────────────────────────────
    for var, csty in component_styles.items():
        sign = csty['sign']
        # CI band
        ci_lo, ci_hi = f'{var}_ci_lo', f'{var}_ci_hi'
        if has_ci and ci_lo in gt_plus:
            _ci_band(ax, gt_plus['y_plus'],
                     gt_plus[ci_lo], gt_plus[ci_hi], sign=sign)
        ax.plot(gt_plus['y_plus'], sign * gt_plus[var],
                color=DNS_COLOR, linewidth=1.8, linestyle=csty['linestyle'],
                zorder=10)

    # ── PIV method markers ───────────────────────────────────────────────
    for name, piv_plus in methods.items():
        sty = _get_style(name)
        for var, csty in component_styles.items():
            sign = csty['sign']
            ax.plot(piv_plus['y_plus'], sign * piv_plus[var],
                    color=sty['color'], marker=sty['marker'],
                    markersize=2.5, alpha=0.55, linestyle='none',
                    zorder=5)

    # ── Build a clean two-part legend ────────────────────────────────────
    # Part 1: method colours (dummy markers)
    method_handles = []
    for name in methods:
        sty = _get_style(name)
        h = plt.Line2D([], [], color=sty['color'], marker=sty['marker'],
                        markersize=5, linestyle='none', label=name)
        method_handles.append(h)
    # DNS entry
    dns_handle = plt.Line2D([], [], color=DNS_COLOR, linewidth=1.8,
                             linestyle='-', label='DNS')
    method_handles.insert(0, dns_handle)

    # Part 2: component line styles (black dummy lines)
    comp_handles = []
    for var, csty in component_styles.items():
        h = plt.Line2D([], [], color='gray', linewidth=1.5,
                        linestyle=csty['linestyle'], label=csty['label_tex'])
        comp_handles.append(h)

    leg1 = ax.legend(handles=method_handles, loc='upper right',
                     framealpha=0.9, title='Method')
    ax.add_artist(leg1)
    ax.legend(handles=comp_handles, loc='upper left',
              framealpha=0.9, title='Component')

    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'Stress$^+$')
    title = 'Reynolds Stresses Method Comparison with DNS'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_xlim(1, Re_tau)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / 'combined_stresses_comparison.png')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'combined_stresses_comparison.png'}")


# =============================================================================
# Error summary
# =============================================================================

def print_error_table(methods, gt_plus):
    """Print R² summary for all methods."""
    from dns_validation.benchmark_comparison import compute_errors

    header = f"{'Method':<18} {'U+ R²':<10} {'U+ RMS%':<10} {'uu+ R²':<10} {'vv+ R²':<10} {'-uv+ R²':<10}"
    print("\n" + header)
    print("-" * len(header))

    for name, piv_plus in methods.items():
        errs = compute_errors(piv_plus, gt_plus, y_plus_range=(10, 500))
        u_r2  = errs.get('U_plus',  {}).get('r2',      float('nan'))
        u_rms = errs.get('U_plus',  {}).get('rms_rel', float('nan'))
        uu_r2 = errs.get('uu_plus', {}).get('r2',      float('nan'))
        vv_r2 = errs.get('vv_plus', {}).get('r2',      float('nan'))
        uv_r2 = errs.get('uv_plus', {}).get('r2',      float('nan'))
        print(f"{name:<18} {u_r2:<10.4f} {u_rms:<10.1f} {uu_r2:<10.4f} {vv_r2:<10.4f} {uv_r2:<10.4f}")

    print()


# =============================================================================
# Main entry point
# =============================================================================

def compare_methods(
    gt_dir,
    output_dir,
    # Instantaneous
    inst_stats_path=None, inst_run_idx=3, inst_y_offset=3.0,
    # Ensemble
    ens_ensemble_path=None, ens_coords_path=None,
    ens_run_idx=4, ens_y_offset=0.8,
    # Stereo
    stereo_stats_path=None, stereo_run_idx=3, stereo_y_offset=0.8,
    stereo_trim_top=10,
    # Shared
    x_exclude=4,
    title_suffix='',
    inst_window_label=None,
    ens_window_label=None,
    stereo_window_label=None,
    trim_near_wall=0,
):
    """
    Compare one pass from each method against DNS ground truth.

    Parameters
    ----------
    gt_dir : str or Path
        Directory containing ground truth (direct_stats.mat etc.)
    output_dir : str or Path
        Where to save figures.
    inst_stats_path : str or Path, optional
        Path to instantaneous mean_stats.mat
    inst_run_idx : int
        0-based run index for instantaneous
    inst_y_offset : float
        Additional y+ offset for instantaneous (on top of hardcoded +1)
    ens_ensemble_path, ens_coords_path : str or Path, optional
        Paths to ensemble_result.mat and coordinates.mat
    ens_run_idx : int
        0-based run index for ensemble
    ens_y_offset : float
        Additional y+ offset for ensemble
    stereo_stats_path : str or Path, optional
        Path to stereo mean_stats.mat
    stereo_run_idx : int
        0-based run index for stereo
    stereo_y_offset : float
        Additional y+ offset for stereo
    stereo_trim_top : int
        Trim this many high-y points from stereo (NaN border cleanup)
    x_exclude : int
        Vectors to exclude from each x-edge
    """
    gt_plus, wu = _load_ground_truth(gt_dir)
    print(f"DNS: Re_tau={wu['Re_tau']:.0f}, u_tau={wu['u_tau']:.4f} mm/s")

    methods = {}

    def _trim(piv_plus, n):
        """Remove n lowest y+ points (nearest wall)."""
        if n <= 0:
            return piv_plus
        yp = piv_plus['y_plus']
        if yp[0] > yp[-1]:
            # Sorted high-to-low: wall is at the end
            sl = slice(None, -n if n > 0 else None)
        else:
            # Sorted low-to-high: wall is at the start
            sl = slice(n, None)
        return {k: (v[sl] if isinstance(v, np.ndarray) and len(v) > n else v)
                for k, v in piv_plus.items()}

    if inst_stats_path is not None:
        label = f"Instantaneous ({inst_window_label})" if inst_window_label else "Instantaneous"
        print(f"\nLoading {label} (run {inst_run_idx}, y+ offset +{inst_y_offset})...")
        data = load_instantaneous(inst_stats_path, inst_run_idx, wu, inst_y_offset, x_exclude)
        methods[label] = _trim(data, trim_near_wall)
        p = methods[label]
        print(f"  y+ range: {p['y_plus'].min():.1f} – {p['y_plus'].max():.1f}")

    if ens_ensemble_path is not None and ens_coords_path is not None:
        label = f"Ensemble ({ens_window_label})" if ens_window_label else "Ensemble"
        print(f"\nLoading {label} (run {ens_run_idx}, y+ offset +{ens_y_offset})...")
        methods[label] = load_ensemble(
            ens_ensemble_path, ens_coords_path,
            ens_run_idx, wu, ens_y_offset, x_exclude)
        p = methods[label]
        print(f"  y+ range: {p['y_plus'].min():.1f} – {p['y_plus'].max():.1f}")

    if stereo_stats_path is not None:
        label = f"Stereo ({stereo_window_label})" if stereo_window_label else "Stereo"
        print(f"\nLoading {label} (run {stereo_run_idx}, y+ offset +{stereo_y_offset}, "
              f"trim top {stereo_trim_top})...")
        data = load_stereo(
            stereo_stats_path, stereo_run_idx, wu, stereo_y_offset,
            x_exclude, stereo_trim_top)
        methods[label] = _trim(data, trim_near_wall)
        p = methods[label]
        print(f"  y+ range: {p['y_plus'].min():.1f} – {p['y_plus'].max():.1f}")

    if not methods:
        raise ValueError("No data loaded — provide at least one method path.")

    print_error_table(methods, gt_plus)

    print("Generating plots...")
    output_dir = Path(output_dir)
    plot_velocity_comparison(methods, gt_plus, wu, output_dir, title_suffix=title_suffix)
    plot_stresses_subplots(methods, gt_plus, wu, output_dir, title_suffix=title_suffix)
    plot_combined_stresses(methods, gt_plus, wu, output_dir, title_suffix=title_suffix)
    print("Done.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Cross-method PIV comparison (Instantaneous / Ensemble / Stereo)')

    parser.add_argument('--gt-dir', '-g', required=True,
                        help='Ground truth directory')
    parser.add_argument('--output-dir', '-o', required=True,
                        help='Output directory for figures')

    # Instantaneous
    parser.add_argument('--inst-stats', type=str, default=None,
                        help='Path to instantaneous mean_stats.mat')
    parser.add_argument('--inst-run', type=int, default=3,
                        help='0-based run index (default: 3)')
    parser.add_argument('--inst-y-offset', type=float, default=3.0,
                        help='Additional y+ offset (default: 3.0)')

    # Ensemble
    parser.add_argument('--ens-dir', type=str, default=None,
                        help='Directory with ensemble_result.mat + coordinates.mat')
    parser.add_argument('--ens-run', type=int, default=4,
                        help='0-based run index (default: 4)')
    parser.add_argument('--ens-y-offset', type=float, default=0.8,
                        help='Additional y+ offset (default: 0.8)')

    # Stereo
    parser.add_argument('--stereo-stats', type=str, default=None,
                        help='Path to stereo mean_stats.mat')
    parser.add_argument('--stereo-run', type=int, default=3,
                        help='0-based run index (default: 3)')
    parser.add_argument('--stereo-y-offset', type=float, default=0.8,
                        help='Additional y+ offset (default: 0.8)')
    parser.add_argument('--stereo-trim-top', type=int, default=10,
                        help='Trim N highest y points from stereo (default: 10)')

    parser.add_argument('--x-exclude', type=int, default=4,
                        help='Vectors to exclude from each x-edge (default: 4)')

    args = parser.parse_args()

    ens_ensemble_path = None
    ens_coords_path = None
    if args.ens_dir:
        ens_dir = Path(args.ens_dir)
        ens_ensemble_path = ens_dir / 'ensemble_result.mat'
        ens_coords_path = ens_dir / 'coordinates.mat'

    compare_methods(
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        inst_stats_path=args.inst_stats,
        inst_run_idx=args.inst_run,
        inst_y_offset=args.inst_y_offset,
        ens_ensemble_path=ens_ensemble_path,
        ens_coords_path=ens_coords_path,
        ens_run_idx=args.ens_run,
        ens_y_offset=args.ens_y_offset,
        stereo_stats_path=args.stereo_stats,
        stereo_run_idx=args.stereo_run,
        stereo_y_offset=args.stereo_y_offset,
        stereo_trim_top=args.stereo_trim_top,
        x_exclude=args.x_exclude,
    )
