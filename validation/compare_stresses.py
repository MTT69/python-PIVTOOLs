#!/usr/bin/env python3
"""
Compare Reynolds stress profiles from PIV ensemble statistics (HDF5)
against DNS ground truth (profiles.txt).

Plots u'u'+, v'v'+, and -u'v'+ for all window sizes.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path


def load_dns_profiles(profiles_txt_path):
    """Load DNS ground truth from profiles.txt.

    Columns: y+, U+, uv+, uu+, vv+, ww+, tau_uv/tau_w, tau_nu/tau_w, tau/tau_w, P+, pp+, up+
    """
    data = np.loadtxt(profiles_txt_path, skiprows=2)
    return {
        'y_plus': data[:, 0],
        'U_plus': data[:, 1],
        'uv_plus': data[:, 2],
        'uu_plus': data[:, 3],
        'vv_plus': data[:, 4],
        'ww_plus': data[:, 5],
    }


def load_ensemble_stats(hdf5_path):
    """Load PIV ensemble statistics from HDF5 file.

    Returns dict keyed by window size with uu_plus, vv_plus, uv_plus, y_plus profiles.
    """
    results = {}
    with h5py.File(str(hdf5_path), 'r') as f:
        window_sizes = np.array(f['config/window_sizes']).flatten()
        es = f['ensemble_stats']

        for i, ws in enumerate(window_sizes):
            ws_int = int(ws)

            def _deref(field, idx=i):
                refs = np.array(es[field]).flatten()
                return np.array(f[refs[idx]]).flatten()

            results[ws_int] = {
                'y_plus': _deref('y_plus'),
                'uu_plus': _deref('uu_plus'),
                'vv_plus': _deref('vv_plus'),
                'uv_plus': _deref('uv_plus'),
                'U_plus': _deref('U_plus'),
                'window_size_px': ws_int,
            }

    return results


def main():
    # Paths
    ensemble_stats_path = Path(
        r'C:\Users\mtt1e23\OneDrive - University of Southampton\Documents'
        r'\#current_processing\4000_images_channel\ensemble_statistics_4000'
        r'\ensemble_statistics_full.mat'
    )
    dns_profiles_path = Path(
        r'c:\Users\mtt1e23\OneDrive - University of Southampton\Documents'
        r'\pivtools_fullstack\python-PIVTOOLs\validation\profiles.txt'
    )

    print("Loading DNS ground truth...")
    dns = load_dns_profiles(dns_profiles_path)
    print(f"  {len(dns['y_plus'])} points, y+ = [{dns['y_plus'].min():.2f}, {dns['y_plus'].max():.2f}]")

    print("Loading PIV ensemble statistics...")
    piv = load_ensemble_stats(ensemble_stats_path)
    for ws, data in sorted(piv.items()):
        print(f"  Window {ws}x{ws}: {len(data['y_plus'])} points, "
              f"y+ = [{data['y_plus'].min():.1f}, {data['y_plus'].max():.1f}]")

    # Colors and markers for each window size
    styles = {
        16: {'color': '#e41a1c', 'marker': 'o', 'ms': 3},
        8:  {'color': '#377eb8', 'marker': 's', 'ms': 3},
        6:  {'color': '#4daf4a', 'marker': '^', 'ms': 3},
        4:  {'color': '#984ea3', 'marker': 'D', 'ms': 2.5},
    }

    # =========================================================================
    # Figure 1: Three separate subplots (uu+, vv+, -uv+)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- uu+ ---
    ax = axes[0]
    ax.plot(dns['y_plus'], dns['uu_plus'], 'k-', linewidth=2, label='DNS', zorder=10)
    for ws in sorted(piv.keys(), reverse=True):
        s = styles[ws]
        ax.plot(piv[ws]['y_plus'], piv[ws]['uu_plus'],
                color=s['color'], marker=s['marker'], markersize=s['ms'],
                linestyle='none', alpha=0.7, label=f'PIV {ws}x{ws}')
    ax.set_xlabel(r'$y^+$', fontsize=13)
    ax.set_ylabel(r"$\overline{u'u'}^+$", fontsize=13)
    ax.set_title("Streamwise Normal Stress", fontsize=14)
    ax.set_xscale('log')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- vv+ ---
    ax = axes[1]
    ax.plot(dns['y_plus'], dns['vv_plus'], 'k-', linewidth=2, label='DNS', zorder=10)
    for ws in sorted(piv.keys(), reverse=True):
        s = styles[ws]
        ax.plot(piv[ws]['y_plus'], piv[ws]['vv_plus'],
                color=s['color'], marker=s['marker'], markersize=s['ms'],
                linestyle='none', alpha=0.7, label=f'PIV {ws}x{ws}')
    ax.set_xlabel(r'$y^+$', fontsize=13)
    ax.set_ylabel(r"$\overline{v'v'}^+$", fontsize=13)
    ax.set_title("Wall-Normal Normal Stress", fontsize=14)
    ax.set_xscale('log')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- -uv+ ---
    ax = axes[2]
    ax.plot(dns['y_plus'], -dns['uv_plus'], 'k-', linewidth=2, label='DNS', zorder=10)
    for ws in sorted(piv.keys(), reverse=True):
        s = styles[ws]
        ax.plot(piv[ws]['y_plus'], -piv[ws]['uv_plus'],
                color=s['color'], marker=s['marker'], markersize=s['ms'],
                linestyle='none', alpha=0.7, label=f'PIV {ws}x{ws}')
    ax.set_xlabel(r'$y^+$', fontsize=13)
    ax.set_ylabel(r"$-\overline{u'v'}^+$", fontsize=13)
    ax.set_title("Reynolds Shear Stress", fontsize=14)
    ax.set_xscale('log')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Reynolds Stresses: PIV Ensemble (4000 images) vs DNS Ground Truth",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(ensemble_stats_path.parent / 'stress_comparison_subplots.png', dpi=150,
                bbox_inches='tight')
    print(f"\nSaved: {ensemble_stats_path.parent / 'stress_comparison_subplots.png'}")

    # =========================================================================
    # Figure 2: All stresses on one axis
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    # DNS reference lines
    ax.plot(dns['y_plus'], dns['uu_plus'], 'k-', linewidth=2,
            label=r"DNS $\overline{u'u'}^+$", zorder=10)
    ax.plot(dns['y_plus'], dns['vv_plus'], 'k--', linewidth=2,
            label=r"DNS $\overline{v'v'}^+$", zorder=10)
    ax.plot(dns['y_plus'], -dns['uv_plus'], 'k:', linewidth=2,
            label=r"DNS $-\overline{u'v'}^+$", zorder=10)

    # PIV for finest window (4x4)
    ws = 4
    s = styles[ws]
    ax.plot(piv[ws]['y_plus'], piv[ws]['uu_plus'],
            color='#e41a1c', marker='o', markersize=3, linestyle='none', alpha=0.6,
            label=rf"PIV {ws}x{ws} $\overline{{u'u'}}^+$")
    ax.plot(piv[ws]['y_plus'], piv[ws]['vv_plus'],
            color='#377eb8', marker='s', markersize=3, linestyle='none', alpha=0.6,
            label=rf"PIV {ws}x{ws} $\overline{{v'v'}}^+$")
    ax.plot(piv[ws]['y_plus'], -piv[ws]['uv_plus'],
            color='#4daf4a', marker='^', markersize=3, linestyle='none', alpha=0.6,
            label=rf"PIV {ws}x{ws} $-\overline{{u'v'}}^+$")

    ax.set_xlabel(r'$y^+$', fontsize=14)
    ax.set_ylabel(r'Stress$^+$', fontsize=14)
    ax.set_title("Reynolds Stresses: PIV 4x4 vs DNS (Re$_{\\tau}$ = 1000)", fontsize=16)
    ax.set_xscale('log')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=10, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(ensemble_stats_path.parent / 'stress_comparison_combined.png', dpi=150)
    print(f"Saved: {ensemble_stats_path.parent / 'stress_comparison_combined.png'}")

    # =========================================================================
    # Figure 3: Per-stress comparison across all window sizes (linear y-axis)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (stress_key, label, title) in zip(axes, [
        ('uu_plus', r"$\overline{u'u'}^+$", "Streamwise Normal Stress"),
        ('vv_plus', r"$\overline{v'v'}^+$", "Wall-Normal Normal Stress"),
        ('uv_plus', r"$-\overline{u'v'}^+$", "Reynolds Shear Stress"),
    ]):
        negate = -1 if 'uv' in stress_key else 1
        ax.plot(dns['y_plus'], negate * dns[stress_key], 'k-', linewidth=2,
                label='DNS', zorder=10)
        for ws in sorted(piv.keys(), reverse=True):
            s = styles[ws]
            ax.plot(piv[ws]['y_plus'], negate * piv[ws][stress_key],
                    color=s['color'], marker=s['marker'], markersize=s['ms'],
                    linestyle='none', alpha=0.7, label=f'PIV {ws}x{ws}')
        ax.set_xlabel(r'$y^+$', fontsize=13)
        ax.set_ylabel(label, fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(0, 1000)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Reynolds Stresses (linear scale): PIV vs DNS", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(ensemble_stats_path.parent / 'stress_comparison_linear.png', dpi=150,
                bbox_inches='tight')
    print(f"Saved: {ensemble_stats_path.parent / 'stress_comparison_linear.png'}")

    plt.show()
    print("\nDone.")


if __name__ == '__main__':
    main()
