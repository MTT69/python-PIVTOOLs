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
    - Quintic B-spline, prefiltered ('bspline5'): effective weights only
      (``bspline5_weights``), consumed by the analytic floor in
      kspace_floor_psd; no DTFT entry, the retired linear fitter never used it

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
    return (
        w_m1 * np.exp(1j * twopik)
        + w_0
        + w_1 * np.exp(-1j * twopik)
        + w_2 * np.exp(-2j * twopik)
    )


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
# Quintic B-spline, prefiltered (cardinal quintic spline) — 'bspline5'
# =============================================================================
#
# The warp first prefilters the image with the inverse of the discrete
# B-spline b = beta5(integers) = (1, 26, 66, 26, 1)/120, then samples the
# 6-tap beta5 basis. The composite is a linear interpolator with weights
#
#     c_n(f) = sum_{j=-2..3} beta5(f - j) * p(n - j),   n in Z,
#
# on pixel floor + n for a sample at floor + f, where p = b^{-1} is the
# two-sided impulse response of the prefilter. In z, b(z) = z^-2 Q(z) / 120
# with Q(z) = z^4 + 26 z^3 + 66 z^2 + 26 z + 1, whose roots are z1, z2 and
# their reciprocals. Residues of 120 z^(k+1) / Q(z) at the two poles inside
# the unit circle give, for every integer k,
#
#     p(k) = A1 z1^|k| + A2 z2^|k|,   A_i = 120 z_i / Q'(z_i).
#
# c_n is therefore exact in closed form for any n; only the stored tap range
# is a truncation. Because beta5 >= 0 and sum_j beta5(f - j) = 1,
# |c_n| <= max_j |p(n - j)| <= (|A1| + |A2|) |z1|^(|n| - d) with d = 3 for
# n > 0 (nearest basis tap +3) and d = 2 for n <= 0 (nearest tap -2). Summing
# the geometric tails beyond the stored taps [lo, hi] bounds the dropped
# weight (``bspline5_tail_bound``). The pole values and the closed form are
# pinned against scipy.ndimage.map_coordinates(order=5, mode='mirror') in
# unit-tests/test_kspace_floor_bspline5.py.

#: Quartic whose roots are the quintic B-spline prefilter poles (z^4..z^0):
#: 120 * z^2 * b(z) with b = beta5 at the integers -2..2.
BSPLINE5_POLY = np.array([1.0, 26.0, 66.0, 26.0, 1.0])
#: Poles inside the unit circle, closed form (|z1| > |z2|; z1 sets the decay).
BSPLINE5_Z1 = 0.5 * (np.sqrt(270.0 - np.sqrt(70980.0)) + np.sqrt(105.0) - 13.0)
BSPLINE5_Z2 = 0.5 * (np.sqrt(270.0 + np.sqrt(70980.0)) - np.sqrt(105.0) - 13.0)
_BSPLINE5_POLES = np.array([BSPLINE5_Z1, BSPLINE5_Z2])
#: Residues: p(k) = sum_i A_i z_i^|k|.
_BSPLINE5_RESIDUES = (
    120.0 * _BSPLINE5_POLES / np.polyval(np.polyder(BSPLINE5_POLY), _BSPLINE5_POLES)
)
#: Basis tap offsets relative to floor(position), as the warp samples them.
_BSPLINE5_BASIS_OFFSETS = np.arange(-2, 4)

#: Largest admissible L1 mass of the effective weights dropped by truncation.
#: The floor is linear in each axis' tap autocorrelation, so the truncation
#: error it carries is O(this tail) relative to the floor's scale; P is then
#: stored float32 (single_pass_accumulator, relative resolution 2^-24 = 6e-8).
#: 1e-9 keeps the truncation at least 60x below that storage resolution, and
#: is also the partition-of-unity accuracy the weights are tested to.
BSPLINE5_TAIL_TOL = 1e-9


def bspline5_tail_bound(offsets: np.ndarray) -> float:
    """Proven upper bound on sum_{n outside offsets} |c_n(f)|, for every f.

    ``offsets`` must be contiguous unit-step and cover the basis taps -2..3.
    Right tail: sum_{n > hi} C a^(n-3) = C a^(hi-2) / (1-a). Left tail:
    sum_{n < lo} C a^(|n|-2) = C a^(|lo|-1) / (1-a). C = |A1| + |A2|,
    a = |z1|.
    """
    offsets = np.asarray(offsets)
    lo, hi = int(offsets[0]), int(offsets[-1])
    if not np.array_equal(offsets, np.arange(lo, hi + 1)) or lo > -2 or hi < 3:
        raise ValueError(
            f"offsets must be contiguous and cover the basis taps -2..3, got "
            f"[{lo}, {hi}] (n = {offsets.size})"
        )
    C = float(np.abs(_BSPLINE5_RESIDUES).sum())
    a = abs(BSPLINE5_Z1)
    return C * (a ** (hi - 2) + a ** (-lo - 1)) / (1.0 - a)


def _bspline5_offsets_for(tol: float) -> np.ndarray:
    """Smallest symmetric tap layout [-(M-1), M] whose tail bound is <= tol."""
    M = 3
    while bspline5_tail_bound(np.arange(-(M - 1), M + 1)) > tol:
        M += 1
    return np.arange(-(M - 1), M + 1)


#: Stored taps of the effective quintic interpolator, chosen by the bound
#: (60 taps, -29..30, for BSPLINE5_TAIL_TOL = 1e-9).
BSPLINE5_OFFSETS = _bspline5_offsets_for(BSPLINE5_TAIL_TOL)


def _beta5_basis_weights(f: float) -> np.ndarray:
    """beta5(f - j) at the basis taps j = -2..3, for f in [0, 1)."""
    a, b, c = 1.0 - f, 2.0 - f, 3.0 - f
    e, g = 1.0 + f, 2.0 + f
    return (
        np.array(
            [
                a**5,
                b**5 - 6.0 * a**5,
                c**5 - 6.0 * b**5 + 15.0 * a**5,
                g**5 - 6.0 * e**5 + 15.0 * f**5,
                e**5 - 6.0 * f**5,
                f**5,
            ]
        )
        / 120.0
    )


def bspline5_weights(f: float, offsets: np.ndarray | None = None) -> np.ndarray:
    """Effective weights c_n(f) of the prefiltered quintic B-spline warp.

    Parameters
    ----------
    f : float
        Fractional position within the central interval, in [0, 1).
    offsets : ndarray of int, optional
        Pixel offsets n relative to floor(position). Defaults to
        ``BSPLINE5_OFFSETS`` (the truncated layout the floor uses); any
        integers are allowed, the closed form is exact for every n.

    Returns
    -------
    ndarray of float64
        c_n(f) on ``offsets``. On the default layout the weights sum to 1
        within ``BSPLINE5_TAIL_TOL`` and are a delta at f = 0 to round-off.
    """
    f = float(f)
    if not 0.0 <= f < 1.0:
        raise ValueError(f"fractional shift f must lie in [0, 1), got {f}")
    n = BSPLINE5_OFFSETS if offsets is None else np.asarray(offsets)
    lag = np.abs(n[:, None] - _BSPLINE5_BASIS_OFFSETS[None, :])  # (n, 6)
    p = (
        _BSPLINE5_RESIDUES[None, None, :]
        * _BSPLINE5_POLES[None, None, :] ** lag[:, :, None]
    ).sum(axis=2)
    return p @ _beta5_basis_weights(f)


# =============================================================================
# Unified API
# =============================================================================


def compute_noise_psd_2d(K_X, K_Y, f_x, f_y, kernel="bicubic"):
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
    if kernel == "lanczos3":
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
