#!/usr/bin/env python3
"""Export CSV data for the 4 validation figures (clean/noisy × velocity/stress)."""

import numpy as np
import csv
from scipy.interpolate import interp1d
from dns_validation.benchmark_comparison import (
    load_wall_units, load_ground_truth, load_piv_statistics, load_ensemble_statistics,
    compute_piv_profiles, convert_to_wall_units
)
from pathlib import Path
import scipy.io as sio


def load_stereo_plus(stats_path, run_idx, wu, y_offset, trim_top=10):
    stats = sio.loadmat(str(stats_path), squeeze_me=True, struct_as_record=False)
    piv_s = stats['piv_result'][run_idx]
    coords_s = stats['coordinates'][run_idx]
    x, y = coords_s.x, coords_s.y
    valid_cols = np.any(~np.isnan(y), axis=0)
    col_indices = np.where(valid_cols)[0]
    mid_col = col_indices[len(col_indices) // 2]
    y_unique = y[:, mid_col]
    valid_rows = ~np.isnan(y_unique)
    y_unique = y_unique[valid_rows]
    if trim_top > 0:
        if y_unique[0] > y_unique[-1]:
            y_unique = y_unique[trim_top:]
            vi = np.where(valid_rows)[0][trim_top:]
        else:
            y_unique = y_unique[:-trim_top]
            vi = np.where(valid_rows)[0][:-trim_top]
        tm = np.zeros(valid_rows.shape, dtype=bool)
        tm[vi] = True
        valid_rows = tm
    xs = col_indices[0] + 4
    xe = col_indices[-1] - 3
    x_mask = np.zeros(x.shape[1], dtype=bool)
    x_mask[xs:xe] = True
    prof = {
        'y_mm': y_unique,
        'U': np.nanmean(piv_s.ux[valid_rows][:, x_mask] * 1000, axis=1),
        'V': np.nanmean(piv_s.uy[valid_rows][:, x_mask] * 1000, axis=1),
        'uu': np.nanmean(piv_s.uu[valid_rows][:, x_mask] * 1e6, axis=1),
        'vv': np.nanmean(piv_s.vv[valid_rows][:, x_mask] * 1e6, axis=1),
        'uv': np.nanmean(piv_s.uv[valid_rows][:, x_mask] * 1e6, axis=1),
    }
    plus = convert_to_wall_units(prof, wu, y_offset_mm=-prof['y_mm'].min())
    plus['y_plus'] = plus['y_plus'] + 1.0 + y_offset
    return plus


def trim_near_wall(plus, n=1):
    yp = plus['y_plus']
    if yp[0] > yp[-1]:
        sl = slice(None, -n)
    else:
        sl = slice(n, None)
    return {k: (v[sl] if isinstance(v, np.ndarray) and len(v) > n else v)
            for k, v in plus.items()}


def write_velocity_csv(path, dns_yp, dns_Up, methods):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['DNS_y_plus', 'DNS_U_plus']
        for name in methods:
            header += [f'{name}_y_plus', f'{name}_U_plus']
        w.writerow(header)
        max_len = max(len(dns_yp), max(len(m['y_plus']) for m in methods.values()))
        for i in range(max_len):
            row = []
            row.append(f'{dns_yp[i]:.6f}' if i < len(dns_yp) else '')
            row.append(f'{dns_Up[i]:.6f}' if i < len(dns_yp) else '')
            for name, m in methods.items():
                if i < len(m['y_plus']):
                    row.append(f'{m["y_plus"][i]:.6f}')
                    row.append(f'{m["U_plus"][i]:.6f}')
                else:
                    row += ['', '']
            w.writerow(row)
    print(f'  Saved: {path}')


def write_stress_csv(path, dns_yp, dns_uu, dns_vv, dns_uv, dns_ci, methods):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['DNS_y_plus', 'DNS_uu_plus', 'DNS_vv_plus', 'DNS_neg_uv_plus']
        if dns_ci:
            header += ['DNS_uu_ci_lo', 'DNS_uu_ci_hi', 'DNS_vv_ci_lo', 'DNS_vv_ci_hi',
                       'DNS_neg_uv_ci_lo', 'DNS_neg_uv_ci_hi']
        for name in methods:
            header += [f'{name}_y_plus', f'{name}_uu_plus', f'{name}_vv_plus', f'{name}_neg_uv_plus']
        w.writerow(header)
        max_len = max(len(dns_yp), max(len(m['y_plus']) for m in methods.values()))
        for i in range(max_len):
            row = []
            if i < len(dns_yp):
                row += [f'{dns_yp[i]:.6f}', f'{dns_uu[i]:.6f}', f'{dns_vv[i]:.6f}', f'{-dns_uv[i]:.6f}']
                if dns_ci:
                    row += [f'{dns_ci["uu_lo"][i]:.6f}', f'{dns_ci["uu_hi"][i]:.6f}',
                            f'{dns_ci["vv_lo"][i]:.6f}', f'{dns_ci["vv_hi"][i]:.6f}',
                            f'{-dns_ci["uv_hi"][i]:.6f}', f'{-dns_ci["uv_lo"][i]:.6f}']
            else:
                row += [''] * (10 if dns_ci else 4)
            for name, m in methods.items():
                if i < len(m['y_plus']):
                    row += [f'{m["y_plus"][i]:.6f}', f'{m["uu_plus"][i]:.6f}',
                            f'{m["vv_plus"][i]:.6f}', f'{-m["uv_plus"][i]:.6f}']
                else:
                    row += [''] * 4
            w.writerow(row)
    print(f'  Saved: {path}')


def get_dns_ci(gt):
    if 'uu_plus_ci_lo' not in gt:
        return None
    return {
        'uu_lo': gt['uu_plus_ci_lo'], 'uu_hi': gt['uu_plus_ci_hi'],
        'vv_lo': gt['vv_plus_ci_lo'], 'vv_hi': gt['vv_plus_ci_hi'],
        'uv_lo': gt['uv_plus_ci_lo'], 'uv_hi': gt['uv_plus_ci_hi'],
    }


if __name__ == '__main__':
    base = Path(r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
                r"\#current_processing\4000_images_channel")
    out_base = base / "Validation_results"

    # ===================== CLEAN FINE =====================
    print("=== Clean Fine ===")
    gt_c_dir = base / "ensemble_statistics_direct_4000"
    wu_c = load_wall_units(gt_c_dir / "direct_stats.mat")
    gt_c = load_ground_truth(gt_c_dir / "direct_stats.mat", wall_units_path=gt_c_dir / "direct_stats.mat")

    inst_c = load_piv_statistics(
        base / "planar_images/instantaneous/statistics/4000/Cam1/instantaneous/mean_stats/mean_stats.mat",
        run_idx=3)
    prof = compute_piv_profiles(inst_c, x_exclude_vectors=4)
    plus_inst_c = convert_to_wall_units(prof, wu_c, y_offset_mm=-prof['y_mm'].min())
    plus_inst_c['y_plus'] = plus_inst_c['y_plus'] + 4.0
    plus_inst_c = trim_near_wall(plus_inst_c)

    ens_c_dir = base / "planar_images/ensemble/calibrated_piv/4000/Cam1/ensemble"
    piv_e = load_ensemble_statistics(ens_c_dir / "ensemble_result.mat", ens_c_dir / "coordinates.mat", run_idx=3)
    prof = compute_piv_profiles(piv_e, x_exclude_vectors=4)
    plus_ens_c = convert_to_wall_units(prof, wu_c, y_offset_mm=-prof['y_mm'].min())
    plus_ens_c['y_plus'] = plus_ens_c['y_plus'] + 1.8

    plus_stereo_c = load_stereo_plus(
        base / "stereo_images/stereo/statistics/4000/stereo/Cam1_Cam2/instantaneous/mean_stats/mean_stats.mat",
        3, wu_c, 0.8)
    plus_stereo_c = trim_near_wall(plus_stereo_c)

    methods_c = {'Instantaneous_16x16': plus_inst_c, 'Ensemble_8x16': plus_ens_c, 'Stereo_16x16': plus_stereo_c}
    out_clean = out_base / "clean" / "cross_method_fine"

    write_velocity_csv(out_clean / "mean_velocity_comparison.csv", gt_c['y_plus'], gt_c['U_plus'], methods_c)
    write_stress_csv(out_clean / "combined_stresses_comparison.csv",
                     gt_c['y_plus'], gt_c['uu_plus'], gt_c['vv_plus'], gt_c['uv_plus'],
                     get_dns_ci(gt_c), methods_c)

    # ===================== NOISY COARSE =====================
    print("\n=== Noisy Coarse ===")
    gt_n_dir = base / "ensemble_stats_direct_reduced_22k_4000"
    wu_n = load_wall_units(gt_n_dir / "direct_stats.mat")
    gt_n = load_ground_truth(gt_n_dir / "direct_stats.mat", wall_units_path=gt_n_dir / "direct_stats.mat")

    inst_n = load_piv_statistics(
        base / "planar_images_reduced_22k_noisy/instantaneous/statistics/4000/Cam1/instantaneous/mean_stats/mean_stats.mat",
        run_idx=2)
    prof = compute_piv_profiles(inst_n, x_exclude_vectors=4)
    plus_inst_n = convert_to_wall_units(prof, wu_n, y_offset_mm=-prof['y_mm'].min())
    plus_inst_n['y_plus'] = plus_inst_n['y_plus'] + 4.0
    plus_inst_n = trim_near_wall(plus_inst_n)

    ens_n_dir = base / "planar_images_reduced_22k_noisy/ensemble/calibrated_piv/4000/Cam1/ensemble"
    piv_e = load_ensemble_statistics(ens_n_dir / "ensemble_result.mat", ens_n_dir / "coordinates.mat", run_idx=3)
    prof = compute_piv_profiles(piv_e, x_exclude_vectors=4)
    plus_ens_n = convert_to_wall_units(prof, wu_n, y_offset_mm=-prof['y_mm'].min())
    plus_ens_n['y_plus'] = plus_ens_n['y_plus'] + 2.0

    plus_stereo_n = load_stereo_plus(
        base / "stereo_images_reduced_22k_noisy/stereo/statistics/4000/stereo/Cam1_Cam2/instantaneous/mean_stats/mean_stats.mat",
        2, wu_n, 0.8)
    plus_stereo_n = trim_near_wall(plus_stereo_n)

    methods_n = {'Instantaneous_32x32': plus_inst_n, 'Ensemble_8x16': plus_ens_n, 'Stereo_32x32': plus_stereo_n}
    out_noisy = out_base / "noisy" / "cross_method_coarse"

    write_velocity_csv(out_noisy / "mean_velocity_comparison.csv", gt_n['y_plus'], gt_n['U_plus'], methods_n)
    write_stress_csv(out_noisy / "combined_stresses_comparison.csv",
                     gt_n['y_plus'], gt_n['uu_plus'], gt_n['vv_plus'], gt_n['uv_plus'],
                     get_dns_ci(gt_n), methods_n)

    print("\nDone.")
