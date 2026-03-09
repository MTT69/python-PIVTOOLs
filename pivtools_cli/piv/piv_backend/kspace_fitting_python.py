"""
DEPRECATED: Pure-Python K-Space Transfer Function Fitter (scipy/TRF)

This is the original Python implementation preserved from commit dfdd4df
for comparison testing against the C/GSL implementation. It uses scipy's
Trust Region Reflective (TRF) solver with proper box constraints.

DO NOT USE IN PRODUCTION — this module exists solely for debugging and
regression testing the C fitter (kspace_fitting.py / kspace_fitting.c).
"""

import logging
from typing import Optional

import numpy as np
from numpy.fft import fft2, fftshift, ifftshift, fftfreq
from scipy.optimize import least_squares

from pivtools_cli.piv.piv_backend.interpolation_noise_psd import (
    compute_noise_psd_2d, frac_distance
)

logger = logging.getLogger(__name__)


def fit_windows_kspace_python(
    R_AA: np.ndarray,
    R_BB: np.ndarray,
    R_AB: np.ndarray,
    mask_flat: np.ndarray,
    corr_size: tuple,
    snr_threshold: float = 3.0,
    use_soft_weighting: bool = True,
    predictor_displacements: Optional[np.ndarray] = None,
    interp_kernel: str = 'bicubic',
    k_max_cap: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure-Python k-space fitter for comparison testing.

    Same interface as fit_windows_kspace() in kspace_fitting.py but uses
    scipy least_squares with TRF method instead of C/GSL.

    Returns (gauss_flat, status_flat, initial_guess_flat) with same shapes.
    """
    corr_h, corr_w = corr_size
    n_per_window = corr_h * corr_w
    num_windows = len(mask_flat)

    expected_size = num_windows * n_per_window
    if R_AA.size != expected_size:
        raise ValueError(
            f"R_AA size {R_AA.size} != expected {expected_size} "
            f"(num_windows={num_windows}, corr_size={corr_size})"
        )

    gauss_flat = np.zeros((num_windows, 16), dtype=np.float64)
    status_flat = np.full(num_windows, -1, dtype=np.int32)
    initial_guess_flat = np.zeros((num_windows, 16), dtype=np.float64)
    diagnostics_flat = np.full((num_windows, 4), np.nan, dtype=np.float64)

    center_x = corr_w / 2.0 + 1
    center_y = corr_h / 2.0 + 1

    k_x = fftshift(fftfreq(corr_w))
    k_y = fftshift(fftfreq(corr_h))
    K_X, K_Y = np.meshgrid(k_x, k_y, indexing='xy')

    valid_indices = np.where(~mask_flat)[0]

    for idx in valid_indices:
        start = idx * n_per_window
        end = start + n_per_window

        R_AA_2d = R_AA[start:end].reshape(corr_h, corr_w)
        R_BB_2d = R_BB[start:end].reshape(corr_h, corr_w)
        R_AB_2d = R_AB[start:end].reshape(corr_h, corr_w)

        if predictor_displacements is not None:
            pred_dy = predictor_displacements[idx, 0]
            pred_dx = predictor_displacements[idx, 1]
        else:
            pred_dy, pred_dx = 0.0, 0.0

        result = _fit_single_window_kspace(
            R_AA_2d, R_BB_2d, R_AB_2d,
            K_X, K_Y, k_x, k_y,
            corr_size, snr_threshold,
            center_x, center_y,
            use_soft_weighting=use_soft_weighting,
            return_diagnostics=True,
            pred_dy=pred_dy,
            pred_dx=pred_dx,
            interp_kernel=interp_kernel,
            k_max_cap=k_max_cap,
        )

        gauss_flat[idx] = result['params']
        status_flat[idx] = result['status']
        initial_guess_flat[idx] = result['initial_guess']

        if 'diagnostics' in result:
            diag = result['diagnostics']
            diagnostics_flat[idx, 0] = diag.get('snr', np.nan)
            diagnostics_flat[idx, 1] = diag.get('N_floor', np.nan)
            diagnostics_flat[idx, 2] = diag.get('k_max_x', np.nan)
            diagnostics_flat[idx, 3] = diag.get('k_max_y', np.nan)

    return gauss_flat, status_flat, initial_guess_flat, diagnostics_flat


def _fit_fref_joint(F_ref, K_X, K_Y, P_noise):
    """Fit joint signal + colored noise model to F_ref.

    Model: F_ref(k) = (A * exp(-2*pi^2*(kx^2*sx^2 + ky^2*sy^2)) + N0) * P_noise(k)
    """
    center_y, center_x = F_ref.shape[0] // 2, F_ref.shape[1] // 2
    F_dc = F_ref[center_y, center_x]

    if F_dc < 1e-10:
        return {'success': False}

    F_ref_norm = F_ref / F_dc

    K_X_flat = K_X.ravel()
    K_Y_flat = K_Y.ravel()
    F_ref_flat = F_ref_norm.ravel()
    P_noise_flat = P_noise.ravel()

    p0 = np.array([1.0, 2.0, 2.0, 0.01])

    bounds_lo = [0.01, 0.1, 0.1, 0.0]
    bounds_hi = [10.0, 20.0, 20.0, 1.0]

    K_R = np.sqrt(K_X_flat ** 2 + K_Y_flat ** 2)
    weights = np.exp(-K_R ** 2 / (0.15 ** 2))
    weights = weights / (np.max(weights) + 1e-12)
    weights = np.maximum(weights, 0.1)

    def residual(params):
        A, sx, sy, N0 = params
        gaussian = A * np.exp(-2 * np.pi ** 2 * (K_X_flat ** 2 * sx ** 2
                                                   + K_Y_flat ** 2 * sy ** 2))
        model = (gaussian + N0) * P_noise_flat
        return weights * (F_ref_flat - model)

    try:
        result = least_squares(
            residual, p0,
            bounds=(bounds_lo, bounds_hi),
            method='trf',
            max_nfev=200,
            ftol=1e-10,
            xtol=1e-10,
        )

        A, sigma_x, sigma_y, N0 = result.x

        N0_abs = N0 * F_dc

        noise_floor_2d = N0_abs * P_noise
        epsilon = F_dc * 1e-8
        F_ref_clean = np.maximum(F_ref - noise_floor_2d, epsilon)

        return {
            'success': True,
            'F_ref_clean': F_ref_clean,
            'N0': N0_abs,
            'A': A * F_dc,
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
        }

    except Exception:
        return {'success': False}


def _fit_single_window_kspace(
    R_AA_2d, R_BB_2d, R_AB_2d,
    K_X, K_Y, k_x, k_y,
    corr_size, snr_threshold,
    center_x, center_y,
    use_soft_weighting=True,
    return_diagnostics=False,
    pred_dy=0.0, pred_dx=0.0,
    interp_kernel='bicubic',
    k_max_cap=None,
):
    corr_h, corr_w = corr_size
    center_idx_x = corr_w // 2
    center_idx_y = corr_h // 2

    amp_A = R_AA_2d[center_idx_y, center_idx_x]
    amp_B = R_BB_2d[center_idx_y, center_idx_x]
    amp_AB = np.max(R_AB_2d)

    default_params = _build_default_params(amp_A, amp_B, amp_AB, center_x, center_y)

    if amp_A < 1e-12 or amp_B < 1e-12:
        return {'params': default_params, 'status': 2, 'initial_guess': default_params.copy()}

    F_AA = fftshift(fft2(ifftshift(R_AA_2d)))
    F_BB = fftshift(fft2(ifftshift(R_BB_2d)))
    F_AB = fftshift(fft2(ifftshift(R_AB_2d)))

    F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))

    f_x = frac_distance(pred_dx / 2.0)
    f_y = frac_distance(pred_dy / 2.0)
    P_noise = compute_noise_psd_2d(K_X, K_Y, f_x, f_y, kernel=interp_kernel)

    joint_result = _fit_fref_joint(F_ref, K_X, K_Y, P_noise)

    if not joint_result['success']:
        return {'params': default_params, 'status': 1, 'initial_guess': default_params.copy()}

    F_ref = joint_result['F_ref_clean']
    N_floor = joint_result['N0']

    epsilon = np.max(np.abs(F_ref)) * 1e-8

    dc_power = np.abs(F_ref[center_idx_y, center_idx_x]) ** 2
    noise_power = N_floor ** 2 + 1e-12
    snr = dc_power / noise_power

    def _make_diag(k_max_x_val=None, k_max_y_val=None):
        if not return_diagnostics:
            return {}
        return {'diagnostics': {
            'snr': snr, 'N_floor': N_floor,
            'k_max_x': k_max_x_val or 0.0,
            'k_max_y': k_max_y_val or 0.0,
        }}

    if snr < snr_threshold:
        return {'params': default_params, 'status': 2,
                'initial_guess': default_params.copy(), **_make_diag()}

    F_ref_dc = np.abs(F_ref[center_idx_y, center_idx_x])
    threshold_frac = 0.01

    F_ref_profile_x = np.abs(F_ref[center_idx_y, :])
    k_max_x = _compute_kmax_from_profile(k_x, F_ref_profile_x, F_ref_dc, threshold_frac)

    F_ref_profile_y = np.abs(F_ref[:, center_idx_x])
    k_max_y = _compute_kmax_from_profile(k_y, F_ref_profile_y, F_ref_dc, threshold_frac)

    k_max_init_x = min(k_max_x, 0.45)
    k_max_init_y = min(k_max_y, 0.45)

    mu_x_init, mu_y_init = _estimate_displacement_from_peak(
        R_AB_2d, center_idx_x, center_idx_y
    )

    _, Sigma_xx_init = _fit_1d_axis(F_AB, F_ref, k_x, center_idx_y, k_max_init_x, axis='x')
    _, Sigma_yy_init = _fit_1d_axis(F_AB, F_ref, k_y, center_idx_x, k_max_init_y, axis='y')

    if Sigma_xx_init < 0 or Sigma_yy_init < 0:
        Sigma_xx_init = max(Sigma_xx_init, 0.1)
        Sigma_yy_init = max(Sigma_yy_init, 0.1)

    if k_max_cap is not None:
        k_max_limit = k_max_cap
    elif use_soft_weighting:
        k_max_limit = 0.45
    else:
        k_max_limit = 0.30

    k_max_x = min(_compute_kmax(Sigma_xx_init, snr, max_k=k_max_limit), k_max_x, k_max_limit)
    k_max_y = min(_compute_kmax(Sigma_yy_init, snr, max_k=k_max_limit), k_max_y, k_max_limit)

    initial_guess = np.array([
        mu_x_init, mu_y_init,
        Sigma_xx_init, Sigma_yy_init, 0.0,
        1.0,
    ])

    try:
        result = _fit_transfer_function_full(
            F_AB, F_ref, K_X, K_Y,
            k_max_x, k_max_y, initial_guess,
            use_soft_weighting=use_soft_weighting,
            noise_floor=noise_power,
            sigma_xx_estimate=Sigma_xx_init,
            sigma_yy_estimate=Sigma_yy_init,
            epsilon=epsilon,
        )

        if result is None:
            return {
                'params': default_params, 'status': 1,
                'initial_guess': _build_params_from_fit(
                    initial_guess, amp_A, amp_B, amp_AB, center_x, center_y),
                **_make_diag(k_max_x, k_max_y),
            }

        mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy, amplitude = result

        if Sigma_xx < 0 or Sigma_yy < 0:
            return {
                'params': default_params, 'status': 5,
                'initial_guess': _build_params_from_fit(
                    initial_guess, amp_A, amp_B, amp_AB, center_x, center_y),
                **_make_diag(k_max_x, k_max_y),
            }

        max_disp_x = 0.75 * corr_w
        max_disp_y = 0.75 * corr_h
        if abs(mu_x) > max_disp_x or abs(mu_y) > max_disp_y:
            return {
                'params': default_params, 'status': 3,
                'initial_guess': _build_params_from_fit(
                    initial_guess, amp_A, amp_B, amp_AB, center_x, center_y),
                **_make_diag(k_max_x, k_max_y),
            }

        params = _build_params_from_fit(
            result, amp_A, amp_B, amp_AB, center_x, center_y
        )

        return {
            'params': params, 'status': 0,
            'initial_guess': _build_params_from_fit(
                initial_guess, amp_A, amp_B, amp_AB, center_x, center_y),
            **_make_diag(k_max_x, k_max_y),
        }

    except Exception as e:
        logger.debug(f"K-space fit failed: {e}")
        return {'params': default_params, 'status': 1, 'initial_guess': default_params.copy()}


def _estimate_displacement_from_peak(R_AB_2d, center_idx_x, center_idx_y):
    corr_h, corr_w = R_AB_2d.shape
    peak_idx = np.argmax(R_AB_2d)
    peak_y, peak_x = np.unravel_index(peak_idx, R_AB_2d.shape)

    sub_x = 0.0
    if 0 < peak_x < corr_w - 1:
        left = R_AB_2d[peak_y, peak_x - 1]
        center = R_AB_2d[peak_y, peak_x]
        right = R_AB_2d[peak_y, peak_x + 1]
        if left > 0 and center > 0 and right > 0:
            ln_l, ln_c, ln_r = np.log(left), np.log(center), np.log(right)
            denom = 2 * (ln_l - 2 * ln_c + ln_r)
            if abs(denom) > 1e-12:
                sub_x = (ln_l - ln_r) / denom

    sub_y = 0.0
    if 0 < peak_y < corr_h - 1:
        top = R_AB_2d[peak_y - 1, peak_x]
        center = R_AB_2d[peak_y, peak_x]
        bottom = R_AB_2d[peak_y + 1, peak_x]
        if top > 0 and center > 0 and bottom > 0:
            ln_t, ln_c, ln_b = np.log(top), np.log(center), np.log(bottom)
            denom = 2 * (ln_t - 2 * ln_c + ln_b)
            if abs(denom) > 1e-12:
                sub_y = (ln_t - ln_b) / denom

    return (peak_x + sub_x) - center_idx_x, (peak_y + sub_y) - center_idx_y


def _fit_1d_axis(F_AB, F_ref, k_axis, other_center_idx, k_max, axis):
    if axis == 'x':
        F_AB_profile = F_AB[other_center_idx, :]
        F_ref_profile = F_ref[other_center_idx, :]
    else:
        F_AB_profile = F_AB[:, other_center_idx]
        F_ref_profile = F_ref[:, other_center_idx]

    log_mag_AB = np.log(np.maximum(np.abs(F_AB_profile), 1e-12))
    log_mag_ref = np.log(np.maximum(F_ref_profile, 1e-12))
    log_mag_T = log_mag_AB - log_mag_ref

    phase_profile = np.angle(F_AB_profile)

    N = len(k_axis)
    k_min = 1.5 / N
    valid_mask_mag = (np.abs(k_axis) > k_min) & (np.abs(k_axis) < k_max)
    k_valid_mag = k_axis[valid_mask_mag]
    log_mag_T_valid = log_mag_T[valid_mask_mag]
    F_AB_mag_valid = np.abs(F_AB_profile[valid_mask_mag])

    phase_k_max = min(k_max, 0.25)
    valid_mask_phase = (np.abs(k_axis) > k_min) & (np.abs(k_axis) < phase_k_max)
    k_valid_phase = k_axis[valid_mask_phase]
    phase_valid = phase_profile[valid_mask_phase]
    F_AB_valid_phase = F_AB_profile[valid_mask_phase]

    Sigma = 1.0
    mu = 0.0

    if len(k_valid_mag) >= 3:
        k_sq = k_valid_mag ** 2
        weights = F_AB_mag_valid / (np.max(F_AB_mag_valid) + 1e-12)
        try:
            A = (k_sq * weights).reshape(-1, 1)
            b = log_mag_T_valid * weights
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            slope = coeffs[0]
            Sigma = max(-slope / (2 * np.pi ** 2), 0.01)
        except Exception:
            pass

    if len(k_valid_phase) >= 3:
        weights_phase = np.abs(F_AB_valid_phase)
        weights_phase = weights_phase / (np.max(weights_phase) + 1e-12)
        try:
            A = (k_valid_phase * weights_phase).reshape(-1, 1)
            b = phase_valid * weights_phase
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            slope_phase = coeffs[0]
            mu = -slope_phase / (2 * np.pi)
        except Exception:
            pass

    return mu, Sigma


def _compute_kmax_from_profile(k_axis, F_profile, F_dc, threshold_frac=0.01,
                                min_k=0.05, max_k=0.45):
    threshold = F_dc * threshold_frac
    center = len(k_axis) // 2
    k_pos = k_axis[center:]
    F_pos = F_profile[center:]
    below_threshold = F_pos < threshold
    if np.any(below_threshold):
        idx = np.argmax(below_threshold)
        k_max = k_pos[max(0, idx - 1)]
    else:
        k_max = max_k
    return np.clip(k_max, min_k, max_k)


def _compute_kmax(sigma_sq, snr, min_k=0.05, max_k=0.45):
    if sigma_sq <= 0 or snr <= 1:
        return max_k
    k_max = np.sqrt(np.log(snr) / (2 * np.pi ** 2 * sigma_sq + 1e-12))
    return np.clip(k_max, min_k, max_k)


def _fit_transfer_function_full(
    F_AB, F_ref, K_X, K_Y,
    k_max_x, k_max_y, initial_guess,
    use_soft_weighting=True, noise_floor=1e-12,
    sigma_xx_estimate=1.0, sigma_yy_estimate=1.0, epsilon=1e-12,
):
    corr_h, corr_w = F_AB.shape
    center_idx_y = corr_h // 2
    center_idx_x = corr_w // 2

    T_measured = F_AB / (F_ref + epsilon)

    T_0 = T_measured[center_idx_y, center_idx_x]
    if np.abs(T_0) < 1e-6:
        return None

    T_normalized = T_measured / T_0

    k_mask = (K_X ** 2 / k_max_x ** 2 + K_Y ** 2 / k_max_y ** 2) <= 1.0

    K_X_flat = K_X[k_mask]
    K_Y_flat = K_Y[k_mask]
    T_norm_flat = T_normalized[k_mask]
    F_ref_flat = F_ref[k_mask]

    if len(K_X_flat) < 10:
        return None

    if use_soft_weighting:
        w_snr = np.abs(F_ref_flat) / (np.sqrt(noise_floor) + 1e-12)
        w_snr = w_snr / (np.max(w_snr) + 1e-12)

        k0_x_sq = 1.0 / (2 * np.pi ** 2 * max(sigma_xx_estimate, 0.01) + 1e-12)
        k0_y_sq = 1.0 / (2 * np.pi ** 2 * max(sigma_yy_estimate, 0.01) + 1e-12)
        w_soft = np.exp(-K_X_flat ** 2 / k0_x_sq - K_Y_flat ** 2 / k0_y_sq)

        weights = w_snr * w_soft
    else:
        weights = np.ones_like(K_X_flat)

    def residual_func(params):
        mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy = params
        phase = -2 * np.pi * (K_X_flat * mu_x + K_Y_flat * mu_y)
        phase_term = np.exp(1j * phase)
        quad_form = (Sigma_xx * K_X_flat ** 2
                     + 2 * Sigma_xy * K_X_flat * K_Y_flat
                     + Sigma_yy * K_Y_flat ** 2)
        decay_term = np.exp(-2 * np.pi ** 2 * quad_form)
        T_model = decay_term * phase_term
        diff = weights * (T_norm_flat - T_model)
        return np.concatenate([diff.real, diff.imag])

    p0 = initial_guess[:5]

    max_disp_x = 0.75 * corr_w
    max_disp_y = 0.75 * corr_h
    bounds = (
        [-max_disp_x, -max_disp_y, 0, 0, -50],
        [max_disp_x, max_disp_y, 100, 100, 50],
    )

    try:
        result = least_squares(
            residual_func, p0,
            bounds=bounds, method='trf',
            max_nfev=50 * len(p0),
            ftol=1e-8, xtol=1e-8,
        )

        n_points = len(T_norm_flat)
        if result.success or result.cost / n_points < 1.0:
            return np.array([
                result.x[0], result.x[1],
                result.x[2], result.x[3], result.x[4],
                1.0
            ])
        else:
            return None
    except Exception:
        return None


def _build_default_params(amp_A, amp_B, amp_AB, center_x, center_y):
    params = np.zeros(16, dtype=np.float64)
    params[0] = amp_A
    params[1] = amp_B
    params[2] = amp_AB
    params[3:6] = 0.0
    params[6:9] = np.nan
    params[9:12] = 0.0
    params[12] = center_x
    params[13] = center_y
    params[14] = center_x
    params[15] = center_y
    return params


def _build_params_from_fit(fit_result, amp_A, amp_B, amp_AB, center_x, center_y):
    if isinstance(fit_result, np.ndarray) and len(fit_result) == 6:
        mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy, amplitude = fit_result
    else:
        mu_x = fit_result[0] if len(fit_result) > 0 else 0.0
        mu_y = fit_result[1] if len(fit_result) > 1 else 0.0
        Sigma_xx = fit_result[2] if len(fit_result) > 2 else 0.0
        Sigma_yy = fit_result[3] if len(fit_result) > 3 else 0.0
        Sigma_xy = fit_result[4] if len(fit_result) > 4 else 0.0

    params = np.zeros(16, dtype=np.float64)
    params[0] = amp_A
    params[1] = amp_B
    params[2] = amp_AB
    params[3:6] = 0.0
    params[6:9] = np.nan
    params[9] = Sigma_xx
    params[10] = Sigma_yy
    params[11] = Sigma_xy
    params[12] = center_x
    params[13] = center_y
    params[14] = center_x + mu_x
    params[15] = center_y + mu_y
    return params
