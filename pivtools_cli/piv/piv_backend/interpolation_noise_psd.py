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
    - B-spline (IIR-prefiltered interpolating cubic, 4-tap basis / discrete
      binomial denominator): matches fused_warp.c interp_mode=2

Key property: At f=0 (integer displacement / pass 0), H(k)=1 and P_noise=1
everywhere — seamlessly reduces to flat white noise, no pass-index gating
needed. For bicubic/lanczos this holds because the f=0 weights collapse to a
delta; for the B-spline it holds by an algebraic identity instead (the f=0
basis weights [1/6, 4/6, 1/6, 0] are NOT a delta, but their DTFT equals the
prefilter denominator b_hat(k) exactly, so the ratio is 1).

Magnitude: the interpolating B-spline is passive, |H| <= 1 for all (k, f)
(the prefilter's high-k gain is exactly cancelled by the basis; H=0 at
Nyquist for f=0.5). Bicubic (Keys a=-0.75) and Lanczos-3 overshoot slightly
mid-band (max |H| ~ 1.029 and ~ 1.05) — "interpolation cannot amplify" is
exact only for the B-spline.
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
# Interpolating cubic B-spline (IIR prefilter + 4-tap basis)
# — matches fused_warp.c interp_mode=2
# =============================================================================

def bspline_weights(f):
    """Compute the 4 cubic B-spline basis weights for fractional shift f.

    These are the SAMPLING weights only (the beta^3 basis evaluated at the
    4 taps); the interpolating property comes from dividing their DTFT by
    the discrete B-spline filter b_hat(k) in :func:`bspline_dtft` — the
    frequency-domain equivalent of the causal/anticausal IIR prefilter
    (pole z1 = sqrt(3)-2) applied to the image in fused_warp.c.

    Note: at f=0 the weights are (1/6, 4/6, 1/6, 0) — NOT a delta. Exact
    reproduction of integer samples happens through the prefilter division,
    not through the weights alone.

    Parameters
    ----------
    f : float
        Fractional position within the central interval, in [0, 1).

    Returns
    -------
    tuple of 4 floats
        (w[-1], w[0], w[1], w[2]) — weights for the 4 surrounding pixels
        (same tap layout as :func:`bicubic_weights`). Sum is exactly 1.
    """
    f2 = f * f
    f3 = f2 * f
    omf = 1.0 - f
    w_m1 = omf * omf * omf / 6.0
    w_0 = (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0
    w_1 = (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) / 6.0
    w_2 = f3 / 6.0
    return w_m1, w_0, w_1, w_2


def bspline_dtft(k, f):
    """DTFT of the interpolating cubic B-spline at wavenumber k, shift f.

    H(k, f) = [w[-1]*e^{+i2pi*k} + w[0] + w[1]*e^{-i2pi*k} + w[2]*e^{-i4pi*k}]
              / b_hat(k),
    b_hat(k) = (2 + cos(2*pi*k)) / 3

    The numerator is the basis-weight DTFT (same tap convention as
    :func:`bicubic_dtft`); the denominator is the frequency response of the
    discrete B-spline filter (beta^3 at integers = [1/6, 4/6, 1/6]), i.e.
    the analytic form of the IIR prefilter. b_hat is bounded in [1/3, 1] so
    the division needs no guard. |H| <= 1 for all (k, f); H(k, 0) = 1
    exactly; closed form at Nyquist: H(0.5, f) = 1 - 6f^2 + 4f^3.

    Parameters
    ----------
    k : array_like
        Wavenumber in cycles/pixel, range [-0.5, 0.5].
    f : float
        Fractional pixel displacement in [0, 0.5].

    Returns
    -------
    H : complex ndarray
        Frequency response of the effective (prefiltered) kernel.
    """
    w_m1, w_0, w_1, w_2 = bspline_weights(f)
    twopik = 2.0 * np.pi * k
    numerator = (w_m1 * np.exp(1j * twopik)
                 + w_0
                 + w_1 * np.exp(-1j * twopik)
                 + w_2 * np.exp(-2j * twopik))
    b_hat = (2.0 + np.cos(twopik)) / 3.0
    return numerator / b_hat


# =============================================================================
# Unified API
# =============================================================================

_KERNEL_DTFTS = {
    'bicubic': bicubic_dtft,
    'lanczos3': lanczos3_dtft,
    'bspline': bspline_dtft,
}


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
        'bicubic', 'lanczos3', or 'bspline'. Unknown names raise (the kernel
        MUST match the warp that produced the images — no silent default).

    Returns
    -------
    P_noise : ndarray
        2D noise power spectral density, same shape as K_X.
    """
    try:
        dtft_fn = _KERNEL_DTFTS[kernel]
    except KeyError:
        raise ValueError(
            f"Unknown interpolation kernel {kernel!r} for noise PSD; "
            f"expected one of {sorted(_KERNEL_DTFTS)}"
        ) from None
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
