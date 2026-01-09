"""
K-Space Transfer Function Fitting for Ensemble PIV

This module implements k-space (Fourier domain) fitting for Reynolds stress
extraction in ensemble PIV. The key advantage over physical-space Gaussian
fitting is that the particle image contribution cancels algebraically:

    F(R_AB) = sqrt(F(R_AA) * F(R_BB)) * T(k)

where T(k) is the transfer function encoding only velocity PDF parameters:

    T(k) = A * exp(-2*pi^2 * k^T * Sigma * k) * exp(-2*pi*i * k . mu)

This assumes a Gaussian velocity PDF (normal distribution) and reduces
the problem from 16 parameters (physical-space) to 6:
    - mu_x, mu_y: Mean displacement
    - Sigma_xx, Sigma_yy, Sigma_xy: Reynolds stress tensor components
    - A: Amplitude

**Anisotropic soft decay weighting**:
    The fitting uses combined SNR and soft decay weighting to optimally
    balance signal quality vs model validity:

    w(k) = w_snr(k) * w_soft(k)

    where:
    - w_snr = |F_ref| / noise_floor  (emphasizes high-signal regions)
    - w_soft = exp(-k_x^2/k0_x^2 - k_y^2/k0_y^2)  (anisotropic decay)
    - k0_x = 1 / sqrt(2*pi^2 * Sigma_xx)  (x decay wavenumber)
    - k0_y = 1 / sqrt(2*pi^2 * Sigma_yy)  (y decay wavenumber)

    The anisotropic weighting matches the elliptical decay of the transfer
    function when Sigma_xx != Sigma_yy, properly handling flows with
    different turbulence intensities in x and y directions.

    This avoids hard k_max caps by naturally down-weighting regions
    where the model becomes unreliable.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.fft import fft2, fftshift, ifftshift, fftfreq
from scipy.optimize import least_squares

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


def fit_windows_kspace(
    R_AA: np.ndarray,
    R_BB: np.ndarray,
    R_AB: np.ndarray,
    mask_flat: np.ndarray,
    corr_size: tuple,
    config,
    pass_idx: int,
    snr_threshold: float = 3.0,
    use_soft_weighting: bool = True,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit k-space transfer function to correlation planes.

    This function provides a drop-in replacement for fit_windows_openmp()
    using k-space fitting instead of Levenberg-Marquardt Gaussian fitting.

    Parameters
    ----------
    R_AA : np.ndarray
        Flattened auto-correlation A planes, shape (num_windows * corr_h * corr_w,)
    R_BB : np.ndarray
        Flattened auto-correlation B planes
    R_AB : np.ndarray
        Flattened cross-correlation AB planes
    mask_flat : np.ndarray
        Boolean mask, shape (num_windows,). True = skip this window.
    corr_size : tuple
        (height, width) of correlation window
    config : Config
        Configuration object
    pass_idx : int
        Current pass index
    snr_threshold : float
        Minimum SNR for including wavenumber in fit (default 3.0)
    use_soft_weighting : bool
        If True (default), use combined SNR × anisotropic soft decay weighting.
        If False, use uniform weighting within k_max bounds.

    Returns
    -------
    gauss_flat : np.ndarray
        Fitted parameters in 16-element format for compatibility.
        Shape: (num_windows, 16)

        Index mapping:
        [0-2]: amp_A, amp_B, amp_AB (peak values from correlation planes)
        [3-5]: c_A, c_B, c_AB = 0 (not used in k-space)
        [6-8]: sig_A_x, sig_A_y, sig_A_xy = NaN (not estimated in k-space)
        [9-11]: sig_AB_x, sig_AB_y, sig_AB_xy = Sigma_xx, Sigma_yy, Sigma_xy
        [12-13]: x0_A, y0_A = window center
        [14-15]: x0_AB, y0_AB = center + mu_x, center + mu_y (displacement)

    status_flat : np.ndarray
        Status codes, shape (num_windows,)
        -1 = masked/skipped, 0 = success, >0 = error code

    initial_guess_flat : np.ndarray
        Initial guesses used, shape (num_windows, 16)
    """
    corr_h, corr_w = corr_size
    n_per_window = corr_h * corr_w
    num_windows = len(mask_flat)

    # Validate input shapes
    expected_size = num_windows * n_per_window
    if R_AA.size != expected_size:
        raise ValueError(
            f"R_AA size {R_AA.size} != expected {expected_size} "
            f"(num_windows={num_windows}, corr_size={corr_size})"
        )

    # Pre-allocate output arrays
    gauss_flat = np.zeros((num_windows, 16), dtype=np.float64)
    status_flat = np.full(num_windows, -1, dtype=np.int32)  # Default: masked
    initial_guess_flat = np.zeros((num_windows, 16), dtype=np.float64)

    # Window center (1-based indexing for compatibility with Gaussian fitter)
    center_x = corr_w / 2.0 + 1
    center_y = corr_h / 2.0 + 1

    # Build wavenumber grids (cycles per pixel, centered)
    k_x = fftshift(fftfreq(corr_w))  # Range: -0.5 to 0.5
    k_y = fftshift(fftfreq(corr_h))
    K_X, K_Y = np.meshgrid(k_x, k_y, indexing='xy')

    # Find valid (non-masked) windows
    valid_indices = np.where(~mask_flat)[0]
    n_valid = len(valid_indices)

    if n_valid == 0:
        logger.warning("All windows masked, no k-space fitting performed")
        return gauss_flat, status_flat, initial_guess_flat

    logger.info(
        f"Pass {pass_idx + 1}: K-space fitting {n_valid}/{num_windows} windows "
        f"(soft_weighting={use_soft_weighting}, SNR threshold={snr_threshold})"
    )

    # Process each valid window
    n_success = 0
    n_failed = 0

    # Diagnostic accumulators (when debug=True)
    diag_snr = []
    diag_k_max_x = []
    diag_k_max_y = []
    diag_sigma_xx = []
    diag_sigma_yy = []
    diag_mu_x = []
    diag_mu_y = []
    status_counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for idx in valid_indices:
        # Extract window correlation planes
        start = idx * n_per_window
        end = start + n_per_window

        R_AA_2d = R_AA[start:end].reshape(corr_h, corr_w)
        R_BB_2d = R_BB[start:end].reshape(corr_h, corr_w)
        R_AB_2d = R_AB[start:end].reshape(corr_h, corr_w)

        # Fit this window
        result = _fit_single_window_kspace(
            R_AA_2d, R_BB_2d, R_AB_2d,
            K_X, K_Y, k_x, k_y,
            corr_size, snr_threshold,
            center_x, center_y,
            use_soft_weighting=use_soft_weighting,
            return_diagnostics=debug,
        )

        if result['status'] == 0:
            n_success += 1
        else:
            n_failed += 1

        status_counts[result['status']] = status_counts.get(result['status'], 0) + 1

        # Store results in 16-element format
        gauss_flat[idx] = result['params']
        status_flat[idx] = result['status']
        initial_guess_flat[idx] = result['initial_guess']

        # Collect diagnostics if available
        if debug and 'diagnostics' in result:
            diag = result['diagnostics']
            diag_snr.append(diag.get('snr', np.nan))
            diag_k_max_x.append(diag.get('k_max_x', np.nan))
            diag_k_max_y.append(diag.get('k_max_y', np.nan))
            if result['status'] == 0:
                diag_sigma_xx.append(result['params'][9])
                diag_sigma_yy.append(result['params'][10])
                diag_mu_x.append(result['params'][14] - center_x)
                diag_mu_y.append(result['params'][15] - center_y)

    # Log summary
    if n_valid > 0:
        success_rate = n_success / n_valid
        logger.info(
            f"Pass {pass_idx + 1}: K-space fitting success rate: {success_rate:.1%} "
            f"({n_success}/{n_valid} non-masked windows)"
        )

        # Log detailed diagnostics if debug mode
        if debug and len(diag_snr) > 0:
            logger.info(f"Pass {pass_idx + 1}: K-space diagnostics:")
            logger.info(f"  SNR: min={np.nanmin(diag_snr):.1f}, median={np.nanmedian(diag_snr):.1f}, max={np.nanmax(diag_snr):.1f}")
            logger.info(f"  k_max_x: min={np.nanmin(diag_k_max_x):.3f}, median={np.nanmedian(diag_k_max_x):.3f}, max={np.nanmax(diag_k_max_x):.3f}")
            logger.info(f"  k_max_y: min={np.nanmin(diag_k_max_y):.3f}, median={np.nanmedian(diag_k_max_y):.3f}, max={np.nanmax(diag_k_max_y):.3f}")
            if len(diag_sigma_xx) > 0:
                logger.info(f"  Sigma_xx (UU): min={np.nanmin(diag_sigma_xx):.4f}, median={np.nanmedian(diag_sigma_xx):.4f}, max={np.nanmax(diag_sigma_xx):.4f}")
                logger.info(f"  Sigma_yy (VV): min={np.nanmin(diag_sigma_yy):.4f}, median={np.nanmedian(diag_sigma_yy):.4f}, max={np.nanmax(diag_sigma_yy):.4f}")
                logger.info(f"  mu_x: min={np.nanmin(diag_mu_x):.4f}, median={np.nanmedian(diag_mu_x):.4f}, max={np.nanmax(diag_mu_x):.4f}")
                logger.info(f"  mu_y: min={np.nanmin(diag_mu_y):.4f}, median={np.nanmedian(diag_mu_y):.4f}, max={np.nanmax(diag_mu_y):.4f}")
            # Status breakdown
            status_names = {-1: 'masked', 0: 'success', 1: 'no_converge', 2: 'low_snr', 3: 'neg_var', 4: 'big_disp'}
            status_str = ', '.join([f"{status_names.get(k, f'status_{k}')}={v}" for k, v in sorted(status_counts.items()) if v > 0])
            logger.info(f"  Status breakdown: {status_str}")

    return gauss_flat, status_flat, initial_guess_flat


def _fit_single_window_kspace(
    R_AA_2d: np.ndarray,
    R_BB_2d: np.ndarray,
    R_AB_2d: np.ndarray,
    K_X: np.ndarray,
    K_Y: np.ndarray,
    k_x: np.ndarray,
    k_y: np.ndarray,
    corr_size: tuple,
    snr_threshold: float,
    center_x: float,
    center_y: float,
    use_soft_weighting: bool = True,
    return_diagnostics: bool = False,
) -> dict:
    """
    Fit k-space transfer function to a single window.

    Parameters
    ----------
    use_soft_weighting : bool
        If True, use SNR × anisotropic soft decay weighting

    Returns dict with 'params' (16-element), 'status' (int), 'initial_guess' (16-element).
    If return_diagnostics=True, also includes 'diagnostics' dict with SNR, k_max values, etc.
    """
    corr_h, corr_w = corr_size
    center_idx_x = corr_w // 2
    center_idx_y = corr_h // 2

    # Extract peak values for amplitude estimates
    amp_A = R_AA_2d[center_idx_y, center_idx_x]
    amp_B = R_BB_2d[center_idx_y, center_idx_x]
    amp_AB = np.max(R_AB_2d)

    # Default output (failure case)
    default_params = _build_default_params(
        amp_A, amp_B, amp_AB, center_x, center_y
    )

    # Check for valid input
    if amp_A < 1e-12 or amp_B < 1e-12:
        return {
            'params': default_params,
            'status': 2,  # SNR too low
            'initial_guess': default_params.copy(),
        }

    # Step 1: Compute FFTs (no windowing - correlation planes are already localized)
    # IMPORTANT: Correlation planes have peak at center (index N/2).
    # Use ifftshift before FFT to center signal at index 0 for correct phase.
    F_AA = fftshift(fft2(ifftshift(R_AA_2d)))
    F_BB = fftshift(fft2(ifftshift(R_BB_2d)))
    F_AB = fftshift(fft2(ifftshift(R_AB_2d)))

    # Step 2: Particle shape reference (algebraic cancellation)
    # Use geometric mean of magnitudes: sqrt(|F_AA| * |F_BB|)
    # This gives a real positive reference for normalization
    F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))

    # Step 3: We avoid explicit division T = F_AB / F_ref to prevent noise amplification
    # Instead, we will fit: F_AB = F_ref * T_model directly
    # For initial guesses, we use log differences: log|T| = log|F_AB| - log|F_ref|
    epsilon = np.max(np.abs(F_ref)) * 1e-8

    # Step 4: Estimate SNR and compute adaptive k-bounds from F_ref decay
    dc_power = np.abs(F_ref[center_idx_y, center_idx_x]) ** 2
    # Estimate noise from corners of k-space
    corner_region = np.abs(F_ref[:3, :3]) ** 2
    noise_power = np.median(corner_region) + 1e-12
    snr = dc_power / noise_power

    # Helper to build diagnostics dict
    def _make_diag(k_max_x_val=None, k_max_y_val=None):
        if not return_diagnostics:
            return {}
        return {'diagnostics': {
            'snr': snr,
            'k_max_x': k_max_x_val if k_max_x_val is not None else 0.0,
            'k_max_y': k_max_y_val if k_max_y_val is not None else 0.0,
        }}

    if snr < snr_threshold:
        return {
            'params': default_params,
            'status': 2,  # SNR too low
            'initial_guess': default_params.copy(),
            **_make_diag(),
        }

    # Compute k_max from F_ref decay along axes
    # Use where F_ref drops to 1% of DC as the boundary
    F_ref_dc = np.abs(F_ref[center_idx_y, center_idx_x])
    threshold_frac = 0.01

    # k_max along x (at k_y=0)
    F_ref_profile_x = np.abs(F_ref[center_idx_y, :])
    k_max_x = _compute_kmax_from_profile(k_x, F_ref_profile_x, F_ref_dc, threshold_frac)

    # k_max along y (at k_x=0)
    F_ref_profile_y = np.abs(F_ref[:, center_idx_x])
    k_max_y = _compute_kmax_from_profile(k_y, F_ref_profile_y, F_ref_dc, threshold_frac)

    # Use more conservative of adaptive and fixed bounds
    k_max_init = min(k_max_x, k_max_y, 0.25)

    # Step 5: Extract initial guesses via 1D linear regression (warm start)
    # Uses log|F_AB| - log|F_ref| = log|T| to avoid explicit division
    mu_x_init, Sigma_xx_init = _fit_1d_axis(
        F_AB, F_ref, k_x, center_idx_y, k_max_init, axis='x'
    )
    mu_y_init, Sigma_yy_init = _fit_1d_axis(
        F_AB, F_ref, k_y, center_idx_x, k_max_init, axis='y'
    )

    # Validate initial estimates
    if Sigma_xx_init < 0 or Sigma_yy_init < 0:
        # Fallback to safe defaults
        Sigma_xx_init = max(Sigma_xx_init, 0.1)
        Sigma_yy_init = max(Sigma_yy_init, 0.1)

    # Compute k_max bounds based on F_ref decay
    # With soft weighting, we use larger bounds (0.45) since weights handle the decay
    # Without soft weighting, use more conservative bounds (0.25)
    if use_soft_weighting:
        k_max_limit = 0.45  # Soft weights handle high-k attenuation
    else:
        k_max_limit = 0.25  # Hard cutoff needs conservative bound

    k_max_x = min(_compute_kmax(Sigma_xx_init, snr), k_max_x, k_max_limit)
    k_max_y = min(_compute_kmax(Sigma_yy_init, snr), k_max_y, k_max_limit)

    # Step 6: Full 6-parameter fit
    initial_guess = np.array([
        mu_x_init, mu_y_init,
        Sigma_xx_init, Sigma_yy_init, 0.0,  # Sigma_xy starts at 0
        1.0,  # Amplitude
    ])

    try:
        result = _fit_transfer_function_full(
            F_AB, F_ref, K_X, K_Y,
            k_max_x, k_max_y, initial_guess,
            use_soft_weighting=use_soft_weighting,
            noise_floor=noise_power,
            sigma_xx_estimate=Sigma_xx_init,
            sigma_yy_estimate=Sigma_yy_init,
        )

        if result is None:
            return {
                'params': default_params,
                'status': 1,  # Optimization did not converge
                'initial_guess': _build_params_from_fit(
                    initial_guess, amp_A, amp_B, amp_AB, center_x, center_y
                ),
                **_make_diag(k_max_x, k_max_y),
            }

        mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy, amplitude = result

        # Validate results
        if Sigma_xx < 0 or Sigma_yy < 0:
            return {
                'params': default_params,
                'status': 3,  # Negative variance
                'initial_guess': _build_params_from_fit(
                    initial_guess, amp_A, amp_B, amp_AB, center_x, center_y
                ),
                **_make_diag(k_max_x, k_max_y),
            }

        # Check displacement bounds (1/2 window rule)
        max_disp = min(corr_w, corr_h) / 2.0
        if abs(mu_x) > max_disp or abs(mu_y) > max_disp:
            return {
                'params': default_params,
                'status': 4,  # Displacement exceeds 1/2 window
                'initial_guess': _build_params_from_fit(
                    initial_guess, amp_A, amp_B, amp_AB, center_x, center_y
                ),
                **_make_diag(k_max_x, k_max_y),
            }

        # Build output in 16-element format
        final_result = result
        params = _build_params_from_fit(
            final_result, amp_A, amp_B, amp_AB, center_x, center_y
        )

        return {
            'params': params,
            'status': 0,  # Success
            'initial_guess': _build_params_from_fit(
                initial_guess, amp_A, amp_B, amp_AB, center_x, center_y
            ),
            **_make_diag(k_max_x, k_max_y),
        }

    except Exception as e:
        logger.debug(f"K-space fit failed: {e}")
        result = {
            'params': default_params,
            'status': 1,  # Optimization did not converge
            'initial_guess': default_params.copy(),
        }
        if return_diagnostics:
            result['diagnostics'] = {'snr': snr if 'snr' in dir() else 0.0, 'k_max_x': 0.0, 'k_max_y': 0.0}
        return result


def _fit_1d_axis(
    F_AB: np.ndarray,
    F_ref: np.ndarray,
    k_axis: np.ndarray,
    other_center_idx: int,
    k_max: float,
    axis: str,
) -> tuple[float, float]:
    """
    Extract mean displacement and variance via 1D linear regression.

    Uses log differences to avoid explicit division:
    - log|T| = log|F_AB| - log|F_ref| = -2*pi^2 * Sigma * k^2  (parabola)
    - phase(T) = phase(F_AB) - phase(F_ref) = -2*pi * k * mu  (linear)

    Parameters
    ----------
    F_AB : np.ndarray
        Complex cross-correlation spectrum, shape (corr_h, corr_w)
    F_ref : np.ndarray
        Reference spectrum sqrt(|F_AA|*|F_BB|), shape (corr_h, corr_w)
    k_axis : np.ndarray
        Wavenumber array for this axis
    other_center_idx : int
        Index of center in the other axis
    k_max : float
        Maximum wavenumber to include in magnitude fit
    axis : str
        'x' or 'y'

    Returns
    -------
    mu : float
        Mean displacement estimate
    Sigma : float
        Variance (stress) estimate
    """
    if axis == 'x':
        # Profile along k_x at k_y = 0
        F_AB_profile = F_AB[other_center_idx, :]
        F_ref_profile = F_ref[other_center_idx, :]
    else:
        # Profile along k_y at k_x = 0
        F_AB_profile = F_AB[:, other_center_idx]
        F_ref_profile = F_ref[:, other_center_idx]

    # Compute log|T| = log|F_AB| - log|F_ref| (avoids division)
    log_mag_AB = np.log(np.maximum(np.abs(F_AB_profile), 1e-12))
    log_mag_ref = np.log(np.maximum(F_ref_profile, 1e-12))
    log_mag_T = log_mag_AB - log_mag_ref

    # Phase of T = phase(F_AB) - phase(F_ref), but F_ref is real positive so phase(F_ref) = 0
    phase_profile = np.angle(F_AB_profile)

    # For magnitude (variance estimation): use k_max
    valid_mask_mag = (np.abs(k_axis) > 0.01) & (np.abs(k_axis) < k_max)
    k_valid_mag = k_axis[valid_mask_mag]
    log_mag_T_valid = log_mag_T[valid_mask_mag]
    # Weight by |F_AB| (higher signal regions more reliable)
    F_AB_mag_valid = np.abs(F_AB_profile[valid_mask_mag])

    # For phase (displacement estimation): use smaller k range to avoid phase wrapping
    # Phase estimation is most reliable at low k where phase is nearly linear
    phase_k_max = min(k_max, 0.25)
    valid_mask_phase = (np.abs(k_axis) > 0.02) & (np.abs(k_axis) < phase_k_max)
    k_valid_phase = k_axis[valid_mask_phase]
    phase_valid = phase_profile[valid_mask_phase]
    F_AB_valid_phase = F_AB_profile[valid_mask_phase]

    # Default values
    Sigma = 1.0
    mu = 0.0

    # Fit magnitude: ln|T| = ln|F_AB| - ln|F_ref| = -2*pi^2 * Sigma * k^2
    if len(k_valid_mag) >= 3:
        k_sq = k_valid_mag ** 2

        # Weighted least squares (weight by |F_AB| for SNR emphasis)
        weights = F_AB_mag_valid / (np.max(F_AB_mag_valid) + 1e-12)

        try:
            # Fit: log_mag_T = slope * k_sq + intercept
            A = np.vstack([k_sq * weights, weights]).T
            b = log_mag_T_valid * weights
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            slope = coeffs[0]
            Sigma = max(-slope / (2 * np.pi ** 2), 0.01)
        except:
            pass

    # Fit phase: phase(F_AB) = -2*pi * k * mu
    if len(k_valid_phase) >= 3:
        # Weight by |F_AB| to emphasize high-signal regions
        weights_phase = np.abs(F_AB_valid_phase)
        weights_phase = weights_phase / (np.max(weights_phase) + 1e-12)

        try:
            # Weighted linear fit: phase = slope * k (no intercept, phase at k=0 should be 0)
            A = np.vstack([k_valid_phase * weights_phase, weights_phase]).T
            b = phase_valid * weights_phase
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            slope_phase = coeffs[0]
            mu = -slope_phase / (2 * np.pi)
        except:
            pass

    return mu, Sigma


def _compute_kmax_from_profile(
    k_axis: np.ndarray,
    F_profile: np.ndarray,
    F_dc: float,
    threshold_frac: float = 0.01,
    min_k: float = 0.05,
    max_k: float = 0.25,
) -> float:
    """
    Compute k_max from where F_ref profile drops below threshold.

    Parameters
    ----------
    k_axis : np.ndarray
        Wavenumber axis (centered, from fftshift(fftfreq()))
    F_profile : np.ndarray
        |F_ref| profile along this axis
    F_dc : float
        DC value |F_ref(0)|
    threshold_frac : float
        Fraction of DC below which to cut off (default 0.01 = 1%)
    min_k, max_k : float
        Bounds on k_max

    Returns
    -------
    float
        k_max value
    """
    threshold = F_dc * threshold_frac

    # Only look at positive k (right half of array after fftshift)
    center = len(k_axis) // 2
    k_pos = k_axis[center:]
    F_pos = F_profile[center:]

    # Find first index where F drops below threshold
    below_threshold = F_pos < threshold
    if np.any(below_threshold):
        idx = np.argmax(below_threshold)  # First True
        k_max = k_pos[max(0, idx - 1)]  # Use one bin before the cutoff
    else:
        k_max = max_k

    return np.clip(k_max, min_k, max_k)


def _compute_kmax(sigma_sq: float, snr: float, min_k: float = 0.05, max_k: float = 0.25) -> float:
    """
    Compute adaptive k-max based on variance and SNR.

    Uses conservative bounds: k_max = sqrt(ln(SNR) / (2*pi^2 * sigma^2))
    but clamped to avoid regions where F_ref becomes unreliable.

    For typical PIV correlation planes, k_max should stay below ~0.25
    to avoid numerical issues from division by small F_ref values.

    Clamped to [min_k, max_k] for stability.
    """
    if sigma_sq <= 0 or snr <= 1:
        return max_k

    k_max = np.sqrt(np.log(snr) / (2 * np.pi ** 2 * sigma_sq + 1e-12))
    return np.clip(k_max, min_k, max_k)


def _fit_transfer_function_full(
    F_AB: np.ndarray,
    F_ref: np.ndarray,
    K_X: np.ndarray,
    K_Y: np.ndarray,
    k_max_x: float,
    k_max_y: float,
    initial_guess: np.ndarray,
    use_soft_weighting: bool = True,
    noise_floor: float = 1e-12,
    sigma_xx_estimate: float = 1.0,
    sigma_yy_estimate: float = 1.0,
) -> Optional[np.ndarray]:
    """
    Full 5-parameter nonlinear fit using normalized transfer function.

    Minimizes: ||w(k) * (T_norm - T_model)||^2

    where T_norm = T / T(0) is the transfer function normalized by its DC value.
    This normalization removes the amplitude ambiguity caused by F_AA ≠ F_BB
    at DC, which otherwise biases the stress estimates.

    The model is:
        T_model(k) = exp(-2*pi^2 * k^T * Sigma * k) * exp(-2*pi*i * k . mu)

    with T_model(0) = 1 by construction.

    Parameters
    ----------
    F_AB : np.ndarray
        Cross-correlation spectrum (complex), shape (corr_h, corr_w)
    F_ref : np.ndarray
        Reference spectrum sqrt(|F_AA|*|F_BB|), shape (corr_h, corr_w)
    K_X, K_Y : np.ndarray
        Wavenumber grids
    k_max_x, k_max_y : float
        Maximum wavenumbers in x and y directions
    initial_guess : np.ndarray
        Initial parameters [mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy, A]
        (A is ignored - kept for interface compatibility)
    use_soft_weighting : bool
        If True, use combined SNR × anisotropic soft decay weighting
    noise_floor : float
        Noise power estimate for SNR computation
    sigma_xx_estimate : float
        Initial Sigma_xx estimate for anisotropic soft decay (x direction)
    sigma_yy_estimate : float
        Initial Sigma_yy estimate for anisotropic soft decay (y direction)

    Returns
    -------
    np.ndarray or None
        Fitted parameters [mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy, A]
        where A is set to 1.0 (for interface compatibility).
        Returns None if optimization failed.
    """
    corr_h, corr_w = F_AB.shape
    center_idx_y = corr_h // 2
    center_idx_x = corr_w // 2

    # Compute transfer function T = F_AB / F_ref
    epsilon = np.max(np.abs(F_ref)) * 1e-8
    T_measured = F_AB / (F_ref + epsilon)

    # Get T(0) for normalization
    T_0 = T_measured[center_idx_y, center_idx_x]
    if np.abs(T_0) < 1e-6:
        return None  # T(0) too small, cannot normalize

    # Normalize T by T(0): T_norm(0) = 1
    T_normalized = T_measured / T_0

    # Build mask for valid k-points (elliptical region)
    k_mask = (K_X ** 2 / k_max_x ** 2 + K_Y ** 2 / k_max_y ** 2) <= 1.0

    # Flatten for optimization
    K_X_flat = K_X[k_mask]
    K_Y_flat = K_Y[k_mask]
    T_norm_flat = T_normalized[k_mask]
    F_ref_flat = F_ref[k_mask]

    if len(K_X_flat) < 10:
        return None  # Not enough valid points

    if use_soft_weighting:
        # Combined SNR × isotropic soft decay weighting
        # Note: Isotropic weighting empirically outperforms anisotropic
        # because it provides more balanced weight between x and y directions
        w_snr = np.abs(F_ref_flat) / (np.sqrt(noise_floor) + 1e-12)
        w_snr = w_snr / (np.max(w_snr) + 1e-12)

        # Isotropic soft decay using average sigma
        sigma_avg = (sigma_xx_estimate + sigma_yy_estimate) / 2
        k0_sq = 1.0 / (2 * np.pi ** 2 * max(sigma_avg, 0.01) + 1e-12)
        K_R_sq = K_X_flat ** 2 + K_Y_flat ** 2
        w_soft = np.exp(-K_R_sq / k0_sq)

        weights = w_snr * w_soft
        weights = weights / (np.max(weights) + 1e-12)
    else:
        weights = np.ones_like(K_X_flat)

    def residual_func(params):
        """Compute weighted residual: w * (T_norm - T_model)."""
        mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy = params

        # Phase term: exp(-2*pi*i * k . mu)
        phase = -2 * np.pi * (K_X_flat * mu_x + K_Y_flat * mu_y)
        phase_term = np.exp(1j * phase)

        # Quadratic form: k^T * Sigma * k
        quad_form = (
            Sigma_xx * K_X_flat ** 2
            + 2 * Sigma_xy * K_X_flat * K_Y_flat
            + Sigma_yy * K_Y_flat ** 2
        )

        # Gaussian decay: T_model = exp(-2*pi^2 * k·Σ·k) with A=1
        decay_term = np.exp(-2 * np.pi ** 2 * quad_form)
        T_model = decay_term * phase_term

        # Weighted residual
        diff = weights * (T_norm_flat - T_model)

        return np.concatenate([diff.real, diff.imag])

    # Initial guess (5 parameters - drop amplitude)
    p0 = initial_guess[:5]

    # Bounds: stresses >= 0, displacements bounded
    bounds = (
        [-10, -10, 0, 0, -50],  # Lower bounds
        [10, 10, 100, 100, 50],  # Upper bounds
    )

    try:
        result = least_squares(
            residual_func,
            p0,
            bounds=bounds,
            method='trf',
            max_nfev=200,  # More iterations for better convergence
            ftol=1e-8,
            xtol=1e-8,
        )

        if result.success or result.cost < 1e6:
            # Return 6-element array for interface compatibility (A=1.0)
            return np.array([
                result.x[0], result.x[1],  # mu_x, mu_y
                result.x[2], result.x[3], result.x[4],  # Sigma_xx, Sigma_yy, Sigma_xy
                1.0  # A (fixed)
            ])
        else:
            return None

    except Exception:
        return None


def _build_default_params(
    amp_A: float,
    amp_B: float,
    amp_AB: float,
    center_x: float,
    center_y: float,
) -> np.ndarray:
    """Build default 16-element parameter array for failure cases."""
    params = np.zeros(16, dtype=np.float64)
    params[0] = amp_A
    params[1] = amp_B
    params[2] = amp_AB
    params[3:6] = 0.0  # Offsets (not used)
    params[6:9] = np.nan  # sig_A (not estimated)
    params[9:12] = 0.0  # sig_AB = Sigma (default zero)
    params[12] = center_x  # x0_A
    params[13] = center_y  # y0_A
    params[14] = center_x  # x0_AB (no displacement)
    params[15] = center_y  # y0_AB
    return params


def _build_params_from_fit(
    fit_result,
    amp_A: float,
    amp_B: float,
    amp_AB: float,
    center_x: float,
    center_y: float,
) -> np.ndarray:
    """
    Build 16-element parameter array from k-space fit result.

    Maps k-space parameters to Gaussian-compatible output format.
    """
    if isinstance(fit_result, np.ndarray) and len(fit_result) == 6:
        mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy, amplitude = fit_result
    else:
        # Initial guess format
        mu_x = fit_result[0] if len(fit_result) > 0 else 0.0
        mu_y = fit_result[1] if len(fit_result) > 1 else 0.0
        Sigma_xx = fit_result[2] if len(fit_result) > 2 else 0.0
        Sigma_yy = fit_result[3] if len(fit_result) > 3 else 0.0
        Sigma_xy = fit_result[4] if len(fit_result) > 4 else 0.0

    params = np.zeros(16, dtype=np.float64)

    # Amplitudes (from correlation plane peaks)
    params[0] = amp_A
    params[1] = amp_B
    params[2] = amp_AB

    # Offsets (not used in k-space)
    params[3] = 0.0
    params[4] = 0.0
    params[5] = 0.0

    # Sigma A - particle shape (not estimated in k-space)
    params[6] = np.nan
    params[7] = np.nan
    params[8] = np.nan

    # Sigma AB = Reynolds stresses
    params[9] = Sigma_xx   # UU stress
    params[10] = Sigma_yy  # VV stress
    params[11] = Sigma_xy  # UV stress

    # Position A (window center)
    params[12] = center_x
    params[13] = center_y

    # Position AB (displacement from center)
    params[14] = center_x + mu_x
    params[15] = center_y + mu_y

    return params


def plot_kspace_diagnostic(
    R_AA_2d: np.ndarray,
    R_BB_2d: np.ndarray,
    R_AB_2d: np.ndarray,
    true_params: Optional[dict] = None,
    snr_threshold: float = 3.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[dict]:
    """
    Create diagnostic k-space plot for debugging and visualization.

    Parameters
    ----------
    R_AA_2d : np.ndarray
        Auto-correlation A plane, shape (h, w)
    R_BB_2d : np.ndarray
        Auto-correlation B plane, shape (h, w)
    R_AB_2d : np.ndarray
        Cross-correlation AB plane, shape (h, w)
    true_params : dict, optional
        Ground truth parameters {'mu_x', 'mu_y', 'Sigma_xx', 'Sigma_yy', 'Sigma_xy'}
    snr_threshold : float
        SNR threshold for k-bounds computation
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    dict
        Fitted parameters and diagnostics, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available for k-space diagnostic plot")
        return None

    corr_h, corr_w = R_AA_2d.shape
    center_idx_x = corr_w // 2
    center_idx_y = corr_h // 2

    # Build wavenumber grids
    k_x = fftshift(fftfreq(corr_w))
    k_y = fftshift(fftfreq(corr_h))
    K_X, K_Y = np.meshgrid(k_x, k_y, indexing='xy')

    # Compute FFTs
    # Use ifftshift before FFT since correlation planes have peak at center
    F_AA = fftshift(fft2(ifftshift(R_AA_2d)))
    F_BB = fftshift(fft2(ifftshift(R_BB_2d)))
    F_AB = fftshift(fft2(ifftshift(R_AB_2d)))

    # Particle shape reference (use magnitudes for real positive reference)
    F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))

    # Transfer function
    epsilon = np.max(np.abs(F_ref)) * 1e-8
    T_measured = F_AB / (F_ref + epsilon)
    T_mag = np.abs(T_measured)
    T_phase = np.angle(T_measured)

    # Compute SNR and adaptive k-bounds from F_ref decay
    dc_power = np.abs(F_ref[center_idx_y, center_idx_x]) ** 2
    corner_region = np.abs(F_ref[:3, :3]) ** 2
    noise_power = np.median(corner_region) + 1e-12
    snr = dc_power / noise_power

    # Compute k_max from F_ref decay along axes
    F_ref_dc = np.abs(F_ref[center_idx_y, center_idx_x])
    threshold_frac = 0.01

    F_ref_profile_x = np.abs(F_ref[center_idx_y, :])
    k_max_x = _compute_kmax_from_profile(k_x, F_ref_profile_x, F_ref_dc, threshold_frac)

    F_ref_profile_y = np.abs(F_ref[:, center_idx_x])
    k_max_y = _compute_kmax_from_profile(k_y, F_ref_profile_y, F_ref_dc, threshold_frac)

    # Use more conservative bound for initial fits
    k_max_init = min(k_max_x, k_max_y, 0.25)

    # 1D fits for initial estimates
    mu_x_init, Sigma_xx_init = _fit_1d_axis(T_measured, k_x, center_idx_y, k_max_init, axis='x')
    mu_y_init, Sigma_yy_init = _fit_1d_axis(T_measured, k_y, center_idx_x, k_max_init, axis='y')

    # Refine k_max based on Sigma estimates
    k_max_x = min(_compute_kmax(max(Sigma_xx_init, 0.01), snr), k_max_x, 0.25)
    k_max_y = min(_compute_kmax(max(Sigma_yy_init, 0.01), snr), k_max_y, 0.25)

    # Extract 1D profiles
    T_profile_x = T_measured[center_idx_y, :]
    T_profile_y = T_measured[:, center_idx_x]

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Row 1: Physical space correlation planes
    ax1 = fig.add_subplot(3, 4, 1)
    im1 = ax1.imshow(R_AA_2d, cmap='RdBu_r', origin='lower')
    ax1.set_title('R_AA (auto-corr A)')
    ax1.set_xlabel('x [px]')
    ax1.set_ylabel('y [px]')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(3, 4, 2)
    im2 = ax2.imshow(R_BB_2d, cmap='RdBu_r', origin='lower')
    ax2.set_title('R_BB (auto-corr B)')
    ax2.set_xlabel('x [px]')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    ax3 = fig.add_subplot(3, 4, 3)
    im3 = ax3.imshow(R_AB_2d, cmap='RdBu_r', origin='lower')
    ax3.set_title('R_AB (cross-corr)')
    ax3.set_xlabel('x [px]')
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # Row 1, col 4: log magnitude of F_ref with k-bounds
    ax4 = fig.add_subplot(3, 4, 4)
    log_F_ref = np.log10(np.abs(F_ref) + 1e-12)
    im4 = ax4.imshow(
        log_F_ref, cmap='viridis', origin='lower',
        extent=[k_x[0], k_x[-1], k_y[0], k_y[-1]]
    )
    # Add elliptical k-bounds
    ellipse = Ellipse(
        (0, 0), 2*k_max_x, 2*k_max_y,
        fill=False, edgecolor='red', linewidth=2, linestyle='--',
        label=f'k_max ({k_max_x:.2f}, {k_max_y:.2f})'
    )
    ax4.add_patch(ellipse)
    ax4.set_title('log10|F_ref| + k-bounds')
    ax4.set_xlabel('k_x [cycles/px]')
    ax4.set_ylabel('k_y [cycles/px]')
    ax4.set_xlim(-0.5, 0.5)
    ax4.set_ylim(-0.5, 0.5)
    ax4.legend(loc='upper right', fontsize=8)
    plt.colorbar(im4, ax=ax4, shrink=0.8)

    # Row 2: Transfer function magnitude and phase
    ax5 = fig.add_subplot(3, 4, 5)
    log_T_mag = np.log10(T_mag + 1e-12)
    im5 = ax5.imshow(
        log_T_mag, cmap='viridis', origin='lower',
        extent=[k_x[0], k_x[-1], k_y[0], k_y[-1]]
    )
    ellipse2 = Ellipse(
        (0, 0), 2*k_max_x, 2*k_max_y,
        fill=False, edgecolor='red', linewidth=2, linestyle='--'
    )
    ax5.add_patch(ellipse2)
    ax5.set_title('log10|T| (transfer function)')
    ax5.set_xlabel('k_x [cycles/px]')
    ax5.set_ylabel('k_y [cycles/px]')
    ax5.set_xlim(-0.5, 0.5)
    ax5.set_ylim(-0.5, 0.5)
    plt.colorbar(im5, ax=ax5, shrink=0.8)

    ax6 = fig.add_subplot(3, 4, 6)
    im6 = ax6.imshow(
        T_phase, cmap='twilight', origin='lower',
        extent=[k_x[0], k_x[-1], k_y[0], k_y[-1]],
        vmin=-np.pi, vmax=np.pi
    )
    ellipse3 = Ellipse(
        (0, 0), 2*k_max_x, 2*k_max_y,
        fill=False, edgecolor='white', linewidth=2, linestyle='--'
    )
    ax6.add_patch(ellipse3)
    ax6.set_title('phase(T) [rad]')
    ax6.set_xlabel('k_x [cycles/px]')
    ax6.set_ylabel('k_y [cycles/px]')
    ax6.set_xlim(-0.5, 0.5)
    ax6.set_ylim(-0.5, 0.5)
    plt.colorbar(im6, ax=ax6, shrink=0.8)

    # Row 2, cols 3-4: 1D magnitude profiles with fits
    ax7 = fig.add_subplot(3, 4, 7)
    valid_k_x = np.abs(k_x) < k_max_x
    k_x_valid = k_x[valid_k_x]
    mag_x = np.abs(T_profile_x[valid_k_x])
    ax7.semilogy(k_x_valid, mag_x, 'b.-', label='|T(k_x, 0)|', markersize=4)

    # Plot fitted Gaussian decay
    if Sigma_xx_init > 0:
        k_fit = np.linspace(-k_max_x, k_max_x, 100)
        mag_fit = np.exp(-2 * np.pi**2 * Sigma_xx_init * k_fit**2)
        ax7.semilogy(k_fit, mag_fit, 'r-', lw=2,
                     label=f'fit: Sigma_xx={Sigma_xx_init:.4f}')
    if true_params and 'Sigma_xx' in true_params:
        mag_true = np.exp(-2 * np.pi**2 * true_params['Sigma_xx'] * k_fit**2)
        ax7.semilogy(k_fit, mag_true, 'g--', lw=2,
                     label=f'true: Sigma_xx={true_params["Sigma_xx"]:.4f}')
    ax7.axvline(-k_max_x, color='gray', linestyle=':', alpha=0.7)
    ax7.axvline(k_max_x, color='gray', linestyle=':', alpha=0.7)
    ax7.set_xlabel('k_x [cycles/px]')
    ax7.set_ylabel('|T|')
    ax7.set_title('|T| profile along k_y=0')
    ax7.legend(fontsize=8)
    ax7.set_ylim(1e-3, 2)
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(3, 4, 8)
    valid_k_y = np.abs(k_y) < k_max_y
    k_y_valid = k_y[valid_k_y]
    mag_y = np.abs(T_profile_y[valid_k_y])
    ax8.semilogy(k_y_valid, mag_y, 'b.-', label='|T(0, k_y)|', markersize=4)

    if Sigma_yy_init > 0:
        k_fit_y = np.linspace(-k_max_y, k_max_y, 100)
        mag_fit_y = np.exp(-2 * np.pi**2 * Sigma_yy_init * k_fit_y**2)
        ax8.semilogy(k_fit_y, mag_fit_y, 'r-', lw=2,
                     label=f'fit: Sigma_yy={Sigma_yy_init:.4f}')
    if true_params and 'Sigma_yy' in true_params:
        mag_true_y = np.exp(-2 * np.pi**2 * true_params['Sigma_yy'] * k_fit_y**2)
        ax8.semilogy(k_fit_y, mag_true_y, 'g--', lw=2,
                     label=f'true: Sigma_yy={true_params["Sigma_yy"]:.4f}')
    ax8.axvline(-k_max_y, color='gray', linestyle=':', alpha=0.7)
    ax8.axvline(k_max_y, color='gray', linestyle=':', alpha=0.7)
    ax8.set_xlabel('k_y [cycles/px]')
    ax8.set_ylabel('|T|')
    ax8.set_title('|T| profile along k_x=0')
    ax8.legend(fontsize=8)
    ax8.set_ylim(1e-3, 2)
    ax8.grid(True, alpha=0.3)

    # Row 3: Phase profiles and summary
    ax9 = fig.add_subplot(3, 4, 9)
    phase_k_max = min(k_max_x, 0.25)
    valid_k_phase = (np.abs(k_x) > 0.02) & (np.abs(k_x) < phase_k_max)
    k_x_phase = k_x[valid_k_phase]
    phase_x = np.angle(T_profile_x[valid_k_phase])
    ax9.plot(k_x_phase, phase_x, 'b.-', label='phase(T(k_x, 0))', markersize=4)

    # Plot fitted phase slope
    k_fit_phase = np.linspace(-phase_k_max, phase_k_max, 100)
    phase_fit = -2 * np.pi * mu_x_init * k_fit_phase
    ax9.plot(k_fit_phase, phase_fit, 'r-', lw=2,
             label=f'fit: mu_x={mu_x_init:.4f}')
    if true_params and 'mu_x' in true_params:
        phase_true = -2 * np.pi * true_params['mu_x'] * k_fit_phase
        ax9.plot(k_fit_phase, phase_true, 'g--', lw=2,
                 label=f'true: mu_x={true_params["mu_x"]:.4f}')
    ax9.axhline(-np.pi, color='gray', linestyle=':', alpha=0.5)
    ax9.axhline(np.pi, color='gray', linestyle=':', alpha=0.5)
    ax9.set_xlabel('k_x [cycles/px]')
    ax9.set_ylabel('phase [rad]')
    ax9.set_title('phase(T) profile along k_y=0')
    ax9.legend(fontsize=8)
    ax9.set_ylim(-np.pi*1.2, np.pi*1.2)
    ax9.grid(True, alpha=0.3)

    ax10 = fig.add_subplot(3, 4, 10)
    phase_k_max_y = min(k_max_y, 0.25)
    valid_k_phase_y = (np.abs(k_y) > 0.02) & (np.abs(k_y) < phase_k_max_y)
    k_y_phase = k_y[valid_k_phase_y]
    phase_y = np.angle(T_profile_y[valid_k_phase_y])
    ax10.plot(k_y_phase, phase_y, 'b.-', label='phase(T(0, k_y))', markersize=4)

    k_fit_phase_y = np.linspace(-phase_k_max_y, phase_k_max_y, 100)
    phase_fit_y = -2 * np.pi * mu_y_init * k_fit_phase_y
    ax10.plot(k_fit_phase_y, phase_fit_y, 'r-', lw=2,
              label=f'fit: mu_y={mu_y_init:.4f}')
    if true_params and 'mu_y' in true_params:
        phase_true_y = -2 * np.pi * true_params['mu_y'] * k_fit_phase_y
        ax10.plot(k_fit_phase_y, phase_true_y, 'g--', lw=2,
                  label=f'true: mu_y={true_params["mu_y"]:.4f}')
    ax10.axhline(-np.pi, color='gray', linestyle=':', alpha=0.5)
    ax10.axhline(np.pi, color='gray', linestyle=':', alpha=0.5)
    ax10.set_xlabel('k_y [cycles/px]')
    ax10.set_ylabel('phase [rad]')
    ax10.set_title('phase(T) profile along k_x=0')
    ax10.legend(fontsize=8)
    ax10.set_ylim(-np.pi*1.2, np.pi*1.2)
    ax10.grid(True, alpha=0.3)

    # Summary panel
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.axis('off')
    summary_text = f"""K-Space Diagnostic Summary
{'='*35}
SNR estimate: {snr:.1f}
Initial k_max: {k_max_init:.3f}

Adaptive k-bounds:
  k_max_x: {k_max_x:.3f}
  k_max_y: {k_max_y:.3f}

1D Fit Results:
  mu_x: {mu_x_init:.4f} px
  mu_y: {mu_y_init:.4f} px
  Sigma_xx: {Sigma_xx_init:.4f} px^2
  Sigma_yy: {Sigma_yy_init:.4f} px^2"""

    if true_params:
        summary_text += f"""

Ground Truth:
  mu_x: {true_params.get('mu_x', 'N/A'):.4f} px
  mu_y: {true_params.get('mu_y', 'N/A'):.4f} px
  Sigma_xx: {true_params.get('Sigma_xx', 'N/A'):.4f} px^2
  Sigma_yy: {true_params.get('Sigma_yy', 'N/A'):.4f} px^2"""

    ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes,
              fontsize=10, family='monospace', verticalalignment='top')

    # log|T| vs k^2 plot (linearized)
    ax12 = fig.add_subplot(3, 4, 12)
    valid_k_sq = (np.abs(k_x) > 0.01) & (np.abs(k_x) < k_max_x)
    k_x_sq = k_x[valid_k_sq] ** 2
    log_mag_x = np.log(np.maximum(np.abs(T_profile_x[valid_k_sq]), 1e-12))
    ax12.plot(k_x_sq, log_mag_x, 'b.-', label='ln|T| vs k_x^2', markersize=4)

    # Linear fit line
    k_sq_fit = np.linspace(0, k_max_x**2, 100)
    log_fit = -2 * np.pi**2 * Sigma_xx_init * k_sq_fit
    ax12.plot(k_sq_fit, log_fit, 'r-', lw=2,
              label=f'slope = -2pi^2 * {Sigma_xx_init:.4f}')
    if true_params and 'Sigma_xx' in true_params:
        log_true = -2 * np.pi**2 * true_params['Sigma_xx'] * k_sq_fit
        ax12.plot(k_sq_fit, log_true, 'g--', lw=2,
                  label=f'true slope')
    ax12.set_xlabel('k_x^2 [cycles^2/px^2]')
    ax12.set_ylabel('ln|T|')
    ax12.set_title('Linearized magnitude fit')
    ax12.legend(fontsize=8)
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"K-space diagnostic saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        'mu_x_1d': mu_x_init,
        'mu_y_1d': mu_y_init,
        'Sigma_xx_1d': Sigma_xx_init,
        'Sigma_yy_1d': Sigma_yy_init,
        'k_max_x': k_max_x,
        'k_max_y': k_max_y,
        'snr': snr,
    }
