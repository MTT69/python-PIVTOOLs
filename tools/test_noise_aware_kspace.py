"""
Manual test tool: Compare current k-space fitter vs noise-aware prototype.

Loads saved correlation planes from a planes_pass_*.mat file, runs both the
current fitter and a prototype noise-aware fitter on selected windows, and
prints a comparison table.

Usage:
    python tools/test_noise_aware_kspace.py <planes_mat_path> [--row ROW] [--col COL] [--n N]

Examples:
    # Test a single window at (row=255, col=127):
    python tools/test_noise_aware_kspace.py path/to/planes_pass_4.mat --row 255 --col 127

    # Test N windows along a column (wall-normal profile):
    python tools/test_noise_aware_kspace.py path/to/planes_pass_4.mat --col 127 --n 20

    # Test N random windows:
    python tools/test_noise_aware_kspace.py path/to/planes_pass_4.mat --n 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.fft import fft2, fftshift, ifftshift, fftfreq
from scipy.optimize import least_squares

# ─── Current k-space fitter (imported from codebase) ─────────────────────────

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from pivtools_cli.piv.piv_backend.kspace_fitting import _fit_single_window_kspace


# ─── Noise-aware prototype ───────────────────────────────────────────────────

def _compute_spectra(R_AA_2d, R_BB_2d, R_AB_2d, corr_size):
    """Compute FFTs and k-space grids from correlation planes."""
    corr_h, corr_w = corr_size
    center_idx_x = corr_w // 2
    center_idx_y = corr_h // 2

    k_x = fftfreq(corr_w, d=1.0)
    k_y = fftfreq(corr_h, d=1.0)
    k_x = np.fft.fftshift(k_x)
    k_y = np.fft.fftshift(k_y)
    K_X, K_Y = np.meshgrid(k_x, k_y)

    F_AA = fftshift(fft2(ifftshift(R_AA_2d)))
    F_BB = fftshift(fft2(ifftshift(R_BB_2d)))
    F_AB = fftshift(fft2(ifftshift(R_AB_2d)))
    F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))

    return {
        'F_AA': F_AA, 'F_BB': F_BB, 'F_AB': F_AB, 'F_ref': F_ref,
        'K_X': K_X, 'K_Y': K_Y, 'k_x': k_x, 'k_y': k_y,
        'center_idx_x': center_idx_x, 'center_idx_y': center_idx_y,
    }


def _fit_noise_aware_kspace(R_AA_2d, R_BB_2d, R_AB_2d, corr_size, center_x, center_y,
                            snr_threshold=3.0, oracle_N=None):
    """
    Noise-aware k-space fitter (prototype).

    Annular ring noise subtraction + transfer function fit with data-derived weights.
    If oracle_N is provided, uses that instead of estimating from corners.
    """
    corr_h, corr_w = corr_size
    spec = _compute_spectra(R_AA_2d, R_BB_2d, R_AB_2d, corr_size)
    F_AA, F_BB, F_AB, F_ref = spec['F_AA'], spec['F_BB'], spec['F_AB'], spec['F_ref']
    K_X, K_Y, k_x, k_y = spec['K_X'], spec['K_Y'], spec['k_x'], spec['k_y']
    center_idx_x, center_idx_y = spec['center_idx_x'], spec['center_idx_y']

    # SNR gate
    dc_power = np.abs(F_ref[center_idx_y, center_idx_x])**2
    K_R = np.sqrt(K_X**2 + K_Y**2)
    noise_ring = (K_R > 0.4) & (K_R < 0.5)
    noise_power = np.median(np.abs(F_ref[noise_ring])**2) + 1e-12
    snr = dc_power / noise_power

    amp_A = R_AA_2d[center_idx_y, center_idx_x]
    amp_B = R_BB_2d[center_idx_y, center_idx_x]
    amp_AB = np.max(R_AB_2d)

    if snr < snr_threshold:
        return _make_result(amp_A, amp_B, amp_AB, center_x, center_y,
                            status=2, info={'snr': snr})

    # ── Noise subtraction ──
    if oracle_N is not None:
        N_annular = oracle_N
    else:
        # Corners where BOTH |k_x| and |k_y| are high — particle Gaussian has
        # decayed in all directions regardless of anisotropy. Camera noise (white)
        # is flat, so corners correctly read the noise floor.
        noise_corners = (np.abs(K_X) > 0.35) & (np.abs(K_Y) > 0.35)
        N_annular = float(np.median(F_ref[noise_corners]))
        N_annular = max(N_annular, 0.0)

    epsilon = F_ref[center_idx_y, center_idx_x] * 1e-8
    F_ref_clean = np.maximum(F_ref - N_annular, epsilon)

    # ── Initial guesses using noise-corrected F_ref ──
    from pivtools_cli.piv.piv_backend.kspace_fitting import _estimate_displacement_from_peak
    mu_x_init, mu_y_init = _estimate_displacement_from_peak(
        R_AB_2d, center_idx_x, center_idx_y
    )

    # 1D regression for Sigma using F_ref_clean

    def _1d_sigma(F_AB_2d, F_ref_clean_2d, k_1d, center_other, axis):
        if axis == 'x':
            fab_prof = F_AB_2d[center_other, :]
            fref_prof = F_ref_clean_2d[center_other, :]
        else:
            fab_prof = F_AB_2d[:, center_other]
            fref_prof = F_ref_clean_2d[:, center_other]

        log_T = np.log(np.maximum(np.abs(fab_prof), 1e-12)) - np.log(np.maximum(fref_prof, 1e-12))
        k_sq = k_1d**2
        valid = (np.abs(k_1d) > 1.5 / len(k_1d)) & (fref_prof > np.max(fref_prof) * 0.01)
        if np.sum(valid) < 3:
            return 1.0
        A_mat = k_sq[valid].reshape(-1, 1)
        b_vec = log_T[valid] - log_T[center_other if axis == 'x' else len(k_1d) // 2]
        w = np.abs(fab_prof[valid]) / (np.max(np.abs(fab_prof[valid])) + 1e-12)
        A_w = A_mat * w[:, None]
        b_w = b_vec * w
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
            return max(-coeffs[0] / (2 * np.pi**2), 0.01)
        except Exception:
            return 1.0

    Sigma_xx_init = _1d_sigma(F_AB, F_ref_clean, k_x, center_idx_y, 'x')
    Sigma_yy_init = _1d_sigma(F_AB, F_ref_clean, k_y, center_idx_x, 'y')

    # ── Full 5-parameter transfer function fit with data-derived weights ──
    T_measured = F_AB / (F_ref_clean + epsilon)
    T_0 = T_measured[center_idx_y, center_idx_x]
    if np.abs(T_0) < 1e-6:
        return _make_result(amp_A, amp_B, amp_AB, center_x, center_y,
                            status=3, info={'snr': snr, 'N': N_annular})

    T_normalized = T_measured / T_0

    # Data-derived weights: w = F_ref_clean / F_ref_clean(0)
    F_ref_clean_dc = F_ref_clean[center_idx_y, center_idx_x]
    w = F_ref_clean / (F_ref_clean_dc + 1e-12)

    # Safety mask at k = 0.45 (outlier protection)
    k_safety = 0.45
    safety_mask = (K_X**2 + K_Y**2) <= k_safety**2

    K_X_safe = K_X[safety_mask]
    K_Y_safe = K_Y[safety_mask]
    T_norm_safe = T_normalized[safety_mask]
    w_safe = w[safety_mask]

    if len(K_X_safe) < 10:
        return _make_result(amp_A, amp_B, amp_AB, center_x, center_y,
                            status=4, info={'snr': snr, 'N': N_annular})

    def residual_func(params):
        mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy = params
        phase = -2 * np.pi * (K_X_safe * mu_x + K_Y_safe * mu_y)
        quad = Sigma_xx * K_X_safe**2 + 2 * Sigma_xy * K_X_safe * K_Y_safe + Sigma_yy * K_Y_safe**2
        T_model = np.exp(-2 * np.pi**2 * quad) * np.exp(1j * phase)
        diff = w_safe * (T_norm_safe - T_model)
        return np.concatenate([diff.real, diff.imag])

    p0 = [mu_x_init, mu_y_init, Sigma_xx_init, Sigma_yy_init, 0.0]
    max_disp = 0.75 * max(corr_w, corr_h)
    bounds = (
        [-max_disp, -max_disp, 0, 0, -50],
        [max_disp, max_disp, 100, 100, 50],
    )

    try:
        result = least_squares(residual_func, p0, bounds=bounds, method='trf',
                               max_nfev=250, ftol=1e-8, xtol=1e-8)
        mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy = result.x
    except Exception:
        return _make_result(amp_A, amp_B, amp_AB, center_x, center_y,
                            status=5, info={'snr': snr, 'N': N_annular})

    info = {
        'snr': snr,
        'N': N_annular,
        'mu_x': mu_x,
        'mu_y': mu_y,
        'Sigma_xx': Sigma_xx,
        'Sigma_yy': Sigma_yy,
        'Sigma_xy': Sigma_xy,
        'n_points': len(K_X_safe),
        'cost': result.cost,
    }
    return _make_result(amp_A, amp_B, amp_AB, center_x, center_y,
                        status=0, info=info,
                        mu=(mu_x, mu_y), Sigma=(Sigma_xx, Sigma_yy, Sigma_xy))


def _make_result(amp_A, amp_B, amp_AB, center_x, center_y,
                 status=0, info=None, mu=None, Sigma=None):
    """Build output dict matching the format of _fit_single_window_kspace."""
    params = np.zeros(16, dtype=np.float64)
    params[0] = amp_A
    params[1] = amp_B
    params[2] = amp_AB
    params[3:6] = 0.0
    params[6:9] = np.nan  # sigma_A not estimated (cancels in transfer function)

    if Sigma is not None:
        params[9] = Sigma[0]    # Sigma_xx (or sigma_A_x + Sigma_xx for v2)
        params[10] = Sigma[1]   # Sigma_yy
        params[11] = Sigma[2]   # Sigma_xy
    else:
        params[9:12] = 0.0

    params[12] = center_x
    params[13] = center_y
    if mu is not None:
        params[14] = center_x + mu[0]
        params[15] = center_y + mu[1]
    else:
        params[14] = center_x
        params[15] = center_y

    return {'params': params, 'status': status, 'info': info or {}}


# ─── Main test logic ─────────────────────────────────────────────────────────

def load_planes(mat_path):
    """Load saved correlation planes from a planes_pass_*.mat file."""
    data = sio.loadmat(str(mat_path))
    return {
        'AA': data['AA'],         # (n_win_y, n_win_x, corr_h, corr_w) — already bg-subtracted & normalized
        'BB': data['BB'],
        'AB': data['AB'],
        'AA_bg': data['AA_bg'],   # background (for reference)
        'BB_bg': data['BB_bg'],
        'AB_bg': data['AB_bg'],
        'norm_factors': data['norm_factors'],
        'gauss_results': data['gauss_results'],
        'corr_size': tuple(data['corr_size'].ravel()),
        'n_win_y': int(data['n_win_y'].ravel()[0]),
        'n_win_x': int(data['n_win_x'].ravel()[0]),
    }


def compute_oracle_N(clean_planes, noisy_planes, iy, ix):
    """Compute oracle noise floor: F_ref_noisy - F_ref_clean (per-pixel median)."""
    corr_size = clean_planes['corr_size']

    R_AA_clean = clean_planes['AA'][iy, ix].astype(np.float64)
    R_BB_clean = clean_planes['BB'][iy, ix].astype(np.float64)
    spec_clean = _compute_spectra(R_AA_clean, R_BB_clean, R_AA_clean, corr_size)

    R_AA_noisy = noisy_planes['AA'][iy, ix].astype(np.float64)
    R_BB_noisy = noisy_planes['BB'][iy, ix].astype(np.float64)
    spec_noisy = _compute_spectra(R_AA_noisy, R_BB_noisy, R_AA_noisy, corr_size)

    # Difference in F_ref: this IS the noise floor (spatially varying)
    diff = spec_noisy['F_ref'] - spec_clean['F_ref']

    # Use median of high-k region as the scalar N (noise is flat there)
    K_X, K_Y = spec_clean['K_X'], spec_clean['K_Y']
    K_R = np.sqrt(K_X**2 + K_Y**2)
    high_k = K_R > 0.3
    oracle_N = float(np.median(diff[high_k]))

    return max(oracle_N, 0.0), diff


def test_window(planes, iy, ix, verbose=True, oracle_N=None):
    """Run both fitters on a single window and return comparison dict."""
    corr_size = planes['corr_size']
    corr_h, corr_w = corr_size
    center_x = corr_w / 2.0
    center_y = corr_h / 2.0

    R_AA = planes['AA'][iy, ix].astype(np.float64)
    R_BB = planes['BB'][iy, ix].astype(np.float64)
    R_AB = planes['AB'][iy, ix].astype(np.float64)

    # Wavenumber grids (needed for current fitter)
    k_x = np.fft.fftshift(fftfreq(corr_w, d=1.0))
    k_y = np.fft.fftshift(fftfreq(corr_h, d=1.0))
    K_X, K_Y = np.meshgrid(k_x, k_y)

    # ── Current fitter ──
    result_current = _fit_single_window_kspace(
        R_AA, R_BB, R_AB, K_X, K_Y, k_x, k_y,
        corr_size, snr_threshold=3.0,
        center_x=center_x, center_y=center_y,
        use_soft_weighting=True,
        return_diagnostics=True,
    )

    # ── Noise-aware prototype ──
    result_noise_aware = _fit_noise_aware_kspace(
        R_AA, R_BB, R_AB, corr_size,
        center_x=center_x, center_y=center_y,
        snr_threshold=3.0,
        oracle_N=oracle_N,
    )

    # ── Existing Gaussian fit result (from saved data) ──
    gauss = planes['gauss_results'][iy, ix]

    # Extract comparison values
    p_cur = result_current['params']
    p_new = result_noise_aware['params']
    info = result_noise_aware.get('info', {})

    comp = {
        'iy': iy, 'ix': ix,
        'status_current': result_current['status'],
        'status_noise_aware': result_noise_aware['status'],
        # Stresses from current k-space
        'Sigma_xx_current': p_cur[9],
        'Sigma_yy_current': p_cur[10],
        'Sigma_xy_current': p_cur[11],
        # Stresses from noise-aware
        'Sigma_xx_noise_aware': p_new[9],
        'Sigma_yy_noise_aware': p_new[10],
        'Sigma_xy_noise_aware': p_new[11],
        # Stresses from Gaussian fitter (if available)
        'sig_AB_x_gauss': gauss[9],
        'sig_A_x_gauss': gauss[6],
        'sig_AB_y_gauss': gauss[10],
        'sig_A_y_gauss': gauss[7],
        'Sigma_xx_gauss': gauss[9] - gauss[6] if np.isfinite(gauss[6]) else gauss[9],
        'Sigma_yy_gauss': gauss[10] - gauss[7] if np.isfinite(gauss[7]) else gauss[10],
        # Displacements
        'mu_x_current': p_cur[14] - p_cur[12],
        'mu_y_current': p_cur[15] - p_cur[13],
        'mu_x_noise_aware': p_new[14] - p_new[12],
        'mu_y_noise_aware': p_new[15] - p_new[13],
        'mu_x_gauss': gauss[14] - gauss[12],
        'mu_y_gauss': gauss[15] - gauss[13],
        # Noise-aware extras
        'N_annular': info.get('N', np.nan),
        'snr': info.get('snr', np.nan),
        'n_points': info.get('n_points', 0),
    }

    if verbose:
        _print_window(comp)

    return comp


def _print_window(c):
    """Pretty-print comparison for one window."""
    print(f"\n{'='*70}")
    print(f"Window ({c['iy']}, {c['ix']})  |  SNR={c['snr']:.1f}  |  "
          f"Status: current={c['status_current']}, noise_aware={c['status_noise_aware']}")
    print(f"  N_annular={c['N_annular']:.6f}  |  n_points={c['n_points']}")
    print(f"{'-'*70}")

    print(f"  {'':20s} {'Current':>12s} {'Noise-Aware':>12s} {'Gaussian':>12s}  {'D(cur-new)':>10s}")
    print(f"  {'-'*66}")

    for label, key_cur, key_new, key_gauss in [
        ("Sigma_xx (uu)", 'Sigma_xx_current', 'Sigma_xx_noise_aware', 'Sigma_xx_gauss'),
        ("Sigma_yy (vv)", 'Sigma_yy_current', 'Sigma_yy_noise_aware', 'Sigma_yy_gauss'),
        ("Sigma_xy (uv)", 'Sigma_xy_current', 'Sigma_xy_noise_aware', None),
        ('mu_x', 'mu_x_current', 'mu_x_noise_aware', 'mu_x_gauss'),
        ('mu_y', 'mu_y_current', 'mu_y_noise_aware', 'mu_y_gauss'),
    ]:
        v_cur = c[key_cur]
        v_new = c[key_new]
        v_gauss = c.get(key_gauss, np.nan) if key_gauss else np.nan
        delta = v_cur - v_new if np.isfinite(v_cur) and np.isfinite(v_new) else np.nan
        g_str = f"{v_gauss:12.6f}" if np.isfinite(v_gauss) else f"{'N/A':>12s}"
        d_str = f"{delta:+10.6f}" if np.isfinite(delta) else f"{'N/A':>10s}"
        print(f"  {label:20s} {v_cur:12.6f} {v_new:12.6f} {g_str} {d_str}")


def main():
    parser = argparse.ArgumentParser(description="Test noise-aware k-space fitting on saved planes")
    parser.add_argument('mat_path', type=str, help='Path to planes_pass_*.mat file')
    parser.add_argument('--row', type=int, default=None, help='Window row index')
    parser.add_argument('--col', type=int, default=None, help='Window column index')
    parser.add_argument('--n', type=int, default=10, help='Number of windows to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for window selection')
    parser.add_argument('--clean-ref', type=str, default=None,
                        help='Path to CLEAN planes_pass_*.mat for oracle N computation')
    args = parser.parse_args()

    print(f"Loading planes from: {args.mat_path}")
    planes = load_planes(args.mat_path)
    n_win_y = planes['n_win_y']
    n_win_x = planes['n_win_x']
    print(f"Grid: {n_win_y} x {n_win_x}, corr_size: {planes['corr_size']}")

    clean_planes = None
    if args.clean_ref:
        print(f"Loading CLEAN reference from: {args.clean_ref}")
        clean_planes = load_planes(args.clean_ref)
        print(f"  Clean grid: {clean_planes['n_win_y']} x {clean_planes['n_win_x']}")
        print("  MODE: Oracle N (computed from clean-noisy F_ref difference)")

    # Select windows to test
    if args.row is not None and args.col is not None:
        # Single window
        windows = [(args.row, args.col)]
    elif args.col is not None:
        # Column profile (wall-normal line)
        step = max(1, n_win_y // args.n)
        windows = [(iy, args.col) for iy in range(0, n_win_y, step)][:args.n]
        print(f"Testing {len(windows)} windows along column {args.col} (wall-normal profile)")
    elif args.row is not None:
        # Row profile
        step = max(1, n_win_x // args.n)
        windows = [(args.row, ix) for ix in range(0, n_win_x, step)][:args.n]
        print(f"Testing {len(windows)} windows along row {args.row}")
    else:
        # Random windows
        rng = np.random.default_rng(args.seed)
        rows = rng.integers(0, n_win_y, size=args.n)
        cols = rng.integers(0, n_win_x, size=args.n)
        windows = list(zip(rows, cols))
        print(f"Testing {len(windows)} random windows")

    # Run comparisons
    results = []
    for iy, ix in windows:
        try:
            o_N = None
            if clean_planes is not None:
                # Grids may differ (clean 511x255, noisy 511x169) — check bounds
                if iy < clean_planes['n_win_y'] and ix < clean_planes['n_win_x']:
                    o_N, _ = compute_oracle_N(clean_planes, planes, iy, ix)
                else:
                    print(f"\n  Window ({iy}, {ix}): outside clean grid, skipping oracle")
            comp = test_window(planes, iy, ix, verbose=True, oracle_N=o_N)
            results.append(comp)
        except Exception as e:
            print(f"\n  Window ({iy}, {ix}): FAILED -- {e}")

    # Summary statistics
    if len(results) > 1:
        successful = [r for r in results if r['status_current'] == 0 and r['status_noise_aware'] == 0]
        if successful:
            print(f"\n{'='*70}")
            print(f"SUMMARY ({len(successful)} successful windows)")
            print(f"{'-'*70}")

            for label, key_cur, key_new in [
                ("Sigma_xx (uu)", 'Sigma_xx_current', 'Sigma_xx_noise_aware'),
                ("Sigma_yy (vv)", 'Sigma_yy_current', 'Sigma_yy_noise_aware'),
            ]:
                vals_cur = np.array([r[key_cur] for r in successful])
                vals_new = np.array([r[key_new] for r in successful])
                delta = vals_cur - vals_new

                print(f"\n  {label}:")
                print(f"    Current:     mean={vals_cur.mean():.6f}, std={vals_cur.std():.6f}")
                print(f"    Noise-aware: mean={vals_new.mean():.6f}, std={vals_new.std():.6f}")
                print(f"    Difference:  mean={delta.mean():+.6f}, std={delta.std():.6f}")
                if vals_cur.mean() > 0:
                    print(f"    Reduction:   {delta.mean() / vals_cur.mean() * 100:+.1f}%")

            N_vals = np.array([r['N_annular'] for r in successful])
            print(f"\n  Noise floor N (annular ring): mean={N_vals.mean():.6f}, std={N_vals.std():.6f}, "
                  f"min={N_vals.min():.6f}, max={N_vals.max():.6f}")


if __name__ == '__main__':
    main()
