"""
Interpolation Kernel Noise PSD for Predictor-Aware K-Space Fitting

Computes the noise power spectral density (PSD) introduced by image
interpolation during predictor warping. When images are warped by a
sub-pixel displacement, the interpolation kernel acts as a low-pass filter
on the noise. The shape of this filter depends on the fractional pixel
displacement and the kernel type.

The noise PSD is computed analytically from the Discrete-Time Fourier
Transform (DTFT) of the interpolation kernel weights:

    P_noise(kx, ky) = |H(kx, fx)|^2 * |H(ky, fy)|^2

where H(k, f) is the 1D DTFT of the kernel at wavenumber k for fractional
displacement f. The 2D factorization is exact because both kernels are
separable.

Supported kernels:
    - Bicubic (Keys a=-0.75, 4-tap): matches fused_warp.c interp_mode=0
    - Lanczos-3 (6-tap windowed sinc): matches fused_warp.c interp_mode=1

Key property: At f=0 (integer displacement / pass 0), all weight is on one
tap, so H(k)=1 and P_noise=1 everywhere — seamlessly reduces to flat white
noise. No pass-index gating needed.
"""

import numpy as np


# =============================================================================
# Bicubic kernel (Keys a=-0.75, 4-tap) — matches fused_warp.c line 27
# =============================================================================

BICUBIC_A = -0.75


def bicubic_weights(f, a=BICUBIC_A):
    """Compute the 4 weights of Keys' bicubic kernel for fractional shift f.

    Parameters
    ----------
    f : float
        Fractional position within the central interval, in [0, 1).
    a : float
        Keys kernel parameter. Default -0.75 (matches cv2.INTER_CUBIC
        and fused_warp.c).

    Returns
    -------
    tuple of 4 floats
        (w[-1], w[0], w[1], w[2]) — weights for the 4 surrounding pixels.
    """
    f2 = f * f
    f3 = f2 * f
    w_m1 = a * f3 - 2 * a * f2 + a * f
    w_0 = (a + 2) * f3 - (a + 3) * f2 + 1.0
    w_1 = -(a + 2) * f3 + (2 * a + 3) * f2 - a * f
    w_2 = -a * f3 + a * f2
    return w_m1, w_0, w_1, w_2


def bicubic_dtft(k, f):
    """DTFT of the bicubic kernel at wavenumber k for fractional shift f.

    H(k, f) = w[-1]*e^{+i2pi*k} + w[0] + w[1]*e^{-i2pi*k} + w[2]*e^{-i4pi*k}

    Tap offsets relative to floor(position): [-1, 0, +1, +2].

    Parameters
    ----------
    k : array_like
        Wavenumber in cycles/pixel, range [-0.5, 0.5].
    f : float
        Fractional pixel displacement in [0, 0.5].

    Returns
    -------
    H : complex ndarray
        Frequency response of the kernel.
    """
    w_m1, w_0, w_1, w_2 = bicubic_weights(f)
    twopik = 2.0 * np.pi * k
    return (w_m1 * np.exp(1j * twopik)
            + w_0
            + w_1 * np.exp(-1j * twopik)
            + w_2 * np.exp(-2j * twopik))


# =============================================================================
# Lanczos-3 kernel (6-tap windowed sinc) — matches fused_warp.c interp_mode=1
# =============================================================================

def _lanczos3_single_weight(t):
    """Single Lanczos-3 weight: sinc(t)*sinc(t/3) for |t|<3, else 0."""
    t = float(t)
    if abs(t) < 1e-12:
        return 1.0
    if abs(t) >= 3.0:
        return 0.0
    pi_t = np.pi * t
    return (np.sin(pi_t) / pi_t) * (np.sin(pi_t / 3.0) / (pi_t / 3.0))


def lanczos3_weights(f):
    """Compute the 6 weights of Lanczos-3 kernel for fractional shift f.

    Taps at offsets [-2, -1, 0, +1, +2, +3] relative to floor(position).

    Parameters
    ----------
    f : float
        Fractional position within the central interval, in [0, 1).

    Returns
    -------
    tuple of 6 floats
        Normalized weights (sum = 1).
    """
    offsets = (-2, -1, 0, 1, 2, 3)
    raw = [_lanczos3_single_weight(f - d) for d in offsets]
    s = sum(raw)
    if s < 1e-12:
        return (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    return tuple(w / s for w in raw)


def lanczos3_dtft(k, f):
    """DTFT of the Lanczos-3 kernel at wavenumber k for fractional shift f.

    H(k, f) = sum of 6 weighted complex exponentials at offsets [-2,-1,0,+1,+2,+3].

    Parameters
    ----------
    k : array_like
        Wavenumber in cycles/pixel, range [-0.5, 0.5].
    f : float
        Fractional pixel displacement in [0, 0.5].

    Returns
    -------
    H : complex ndarray
        Frequency response of the kernel.
    """
    weights = lanczos3_weights(f)
    offsets = (-2, -1, 0, 1, 2, 3)
    twopik = 2.0 * np.pi * k
    H = np.zeros_like(k, dtype=complex)
    for w, d in zip(weights, offsets):
        H += w * np.exp(-1j * d * twopik)
    return H


# =============================================================================
# Unified API
# =============================================================================

def compute_noise_psd_2d(K_X, K_Y, f_x, f_y, kernel='bicubic'):
    """Compute 2D noise PSD from interpolation kernel DTFT.

    P_noise(kx, ky) = |H(kx, fx)|^2 * |H(ky, fy)|^2

    At f_x = f_y = 0 (no warping / pass 0), P_noise = 1.0 everywhere.

    Parameters
    ----------
    K_X, K_Y : ndarray
        2D wavenumber grids (cycles/pixel, centered), range [-0.5, 0.5].
    f_x, f_y : float
        Fractional pixel displacements (distance to nearest integer, 0..0.5).
    kernel : str
        'bicubic' or 'lanczos3'.

    Returns
    -------
    P_noise : ndarray
        2D noise power spectral density, same shape as K_X.
    """
    if kernel == 'lanczos3':
        dtft_fn = lanczos3_dtft
    else:
        dtft_fn = bicubic_dtft
    H_x = dtft_fn(K_X, f_x)
    H_y = dtft_fn(K_Y, f_y)
    return (np.abs(H_x) ** 2) * (np.abs(H_y) ** 2)


def frac_distance(x):
    """Distance to nearest integer: 0 at integers, 0.5 at half-pixels.

    Parameters
    ----------
    x : float or array_like
        Position(s) in pixels.

    Returns
    -------
    float or ndarray
        Fractional distance in [0, 0.5].
    """
    return np.abs(x - np.round(x))


def compute_composed_fractional_displacements(
    map_x, map_y, win_ctrs_y, win_ctrs_x,
    pred_dy_2d=None, pred_dx_2d=None,
):
    """Compute fractional pixel displacements for composed dewarp+predictor warp.

    For each window centre, computes the raw image coordinate that the
    composed warp samples at, then extracts the fractional part. This is
    the fractional displacement that determines P_noise for the single-pass
    interpolation.

    For pass 0 (no predictor), the fractional part comes from the dewarp map
    values at integer window centres — P_noise is NOT flat (unlike the old
    two-pass approach where the dewarp noise was unmodeled).

    Parameters
    ----------
    map_x, map_y : ndarray, shape (H_dw, W_dw)
        Dewarp remap tables (dewarped pixel → raw camera pixel).
    win_ctrs_y, win_ctrs_x : 1D ndarray
        Window centres in dewarped pixel coordinates.
    pred_dy_2d, pred_dx_2d : ndarray, shape (n_win_y, n_win_x), optional
        Per-window predictor displacements in dewarped pixels. None for pass 0.

    Returns
    -------
    frac_disp : ndarray, shape (n_windows, 2)
        Fractional displacements [f_y, f_x] in [0, 0.5] for each window.
        Averaged over the symmetric A (+pred/2) and B (-pred/2) warp.
    """
    from scipy.ndimage import map_coordinates

    n_win_y = len(win_ctrs_y)
    n_win_x = len(win_ctrs_x)
    n_windows = n_win_y * n_win_x

    # Build 2D grid of window centre coordinates
    ctrs_y_2d, ctrs_x_2d = np.meshgrid(win_ctrs_y, win_ctrs_x, indexing="ij")
    ctrs_y_flat = ctrs_y_2d.ravel()
    ctrs_x_flat = ctrs_x_2d.ravel()

    if pred_dy_2d is not None and pred_dx_2d is not None:
        half_dy = pred_dy_2d.ravel() / 2.0
        half_dx = pred_dx_2d.ravel() / 2.0
    else:
        half_dy = np.zeros(n_windows)
        half_dx = np.zeros(n_windows)

    frac_disp = np.zeros((n_windows, 2), dtype=np.float64)

    for sign, label in [(-1.0, "A"), (1.0, "B")]:
        shifted_y = ctrs_y_flat + sign * half_dy
        shifted_x = ctrs_x_flat + sign * half_dx

        # Bilinear interpolation of dewarp maps at shifted positions
        coords = np.array([shifted_y, shifted_x])
        raw_y = map_coordinates(map_y.astype(np.float64), coords, order=1, mode="nearest")
        raw_x = map_coordinates(map_x.astype(np.float64), coords, order=1, mode="nearest")

        frac_disp[:, 0] += frac_distance(raw_y)
        frac_disp[:, 1] += frac_distance(raw_x)

    # Average A and B fractional displacements
    frac_disp /= 2.0

    return frac_disp
