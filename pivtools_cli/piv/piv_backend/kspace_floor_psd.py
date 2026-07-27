"""Analytic coloured noise-floor PSD P(k; fx, fy) for the k-space ensemble fitter.

White sensor noise does not reach the ensemble transfer ratio flat: the
image-deformation pipeline colours it. This module computes the expected
white-noise floor of F_ref = sqrt(|F_AA||F_BB|) through the exact production
chain — warp-kernel fractional-shift filtering, (weighted) per-pair
window-mean removal, sum-window correlation, central crop, and envelope
divide — as a closed form, with no Monte-Carlo step. The LM fitter consumes
it as the per-window attenuation ``1 - N0 * P(k;fx,fy) / F_ref``.

Derivation (validated against the offline ``simulate_floor`` referee at 1024
pairs across {single, std} x {lanczos, cubic} x {fx, fy, no-window-mean,
(fx, fy) grid}, worst per-cell median deviation 0.15-0.18%; scripts
``TurbulentChannel/scripts/analytic_floor_P_single.py`` + ``white_noise_floor_probe.py``):

Both auto buffers are built as ``g = w_B * (u - m)`` with ``m = (w_B . u) / sum(w_B)``,
where ``w_B`` is the correlation pixel weight and ``u`` the frac-shifted white
field. The fused warp is a same-kernel tensor product (fused_warp.c
``wy[m] * wx[n]``), so the warped-noise covariance is separable,
``C = C_x (x) C_y`` with ``C_a = Toeplitz(r(tau; f_a))`` and ``r`` the kernel
tap autocorrelation. ``g`` is linear in ``u`` with ``E[g] = 0``, so the
expected auto spectrum is the exact quadratic form which expands into three
pieces:

    E_S(k) = T1(k) - (2/sum(w_B)) * Re[conj(W) * V] + q * |W|^2 / sum(w_B)^2

    T1(k)  = sum_{tau_x, tau_y} r_x r_y A(tau_x, tau_y) e^{-i k . tau}
    A(tau) = 2-D zero-padded lag autocorrelation of w_B
    v      = C w_B (separable x-then-y tap correlation),  q = w_B . v
    W      = DFT2(w_B),  V = DFT2(w_B * v)

At k = 0: T1 = q, W = sum(w_B), V = q, so E_S(0) = 0 exactly — the
(weighted) window-mean hole. AB never enters the floor (A/B noise
independent, E[K_AB] = 0). ``hole=False`` drops the two rank-1 terms: the
floor of a run WITHOUT ``window_mean`` background subtraction (bMeanSubtract
is conditional in the C accumulator for BOTH chains); every other background
ingredient is a k-independent factor that cancels under the P normalization.

Post-chain, identical to production: ifft2 at the sum size, fftshift,
central (h, w) crop, divide the stored ``env_auto``, centred FFT, magnitude,
normalize to mean 1 at (fx, fy) = (0, 0).

Weight semantics: ``w_B`` is ALWAYS the actual correlation pixel weight of
the pass — ``_window_weight_fun(sum_window, 'bsingle', sum_window)`` for
single-mode passes, ``_window_weight_fun(win_size, ensemble_window_type)``
for std passes (the C correlator weights the windows and mean-subtracts per
buffer support in every mode). For the 'square' window type the std weight
is all-ones, which is the offline-validated flat-weight limit; non-square
tapers are covered by the same algebra but were not part of the offline
gates.

Scope: the colouring is exact for separable per-axis fractional shifts (the
fused-warp model), parameterized by ``(fx, fy) = frac_distance(pred/2)`` per
window. P provenance (kernel, hole flag, f grids) is recorded in the fit
diagnostics sidecar by the caller.
"""

from __future__ import annotations

import numpy as np

from .interpolation_noise_psd import bicubic_weights, lanczos3_weights
from .kspace_common import _fft_planes

# kernel name -> (tap offsets, weights function); keys match the production
# ensemble_image_warp_interpolation options (fused_warp.c taps: cubic = Keys
# a=-0.75 4-tap, lanczos = Lanczos-3 6-tap)
KERNEL_TAPS = {
    "lanczos": (np.array([-2, -1, 0, 1, 2, 3]), lanczos3_weights),
    "cubic": (np.array([-1, 0, 1, 2]), bicubic_weights),
}

N_FRACS_DEFAULT = 21  # f grid = linspace(0, 0.5, 21): the offline-validated density


def tap_autocorr(f: float, kernel: str = "lanczos") -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation r(tau) of the warp-kernel taps at fractional shift f."""
    if kernel not in KERNEL_TAPS:
        raise ValueError(f"unknown kernel '{kernel}' (valid: {set(KERNEL_TAPS)})")
    offsets, weights_fun = KERNEL_TAPS[kernel]
    c = np.asarray(weights_fun(f), dtype=np.float64)
    n = len(offsets)
    span = int(offsets[-1] - offsets[0])
    taus = np.arange(-span, span + 1)
    r = np.array(
        [
            sum(
                c[i] * c[j]
                for i in range(n)
                for j in range(n)
                if offsets[j] - offsets[i] == t
            )
            for t in taus
        ]
    )
    return taus, r


def _shift_pad(a: np.ndarray, tau: int, axis: int) -> np.ndarray:
    """a shifted by +tau along axis with zero padding: out(x) = a(x + tau)."""
    out = np.zeros_like(a)
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if tau >= 0:
        n = a.shape[axis] - tau
        src[axis] = slice(tau, None)
        dst[axis] = slice(None, n)
    else:
        src[axis] = slice(None, tau)
        dst[axis] = slice(-tau, None)
    out[tuple(dst)] = a[tuple(src)]
    return out


def analytic_floor_single(
    fx: float,
    fy: float,
    env_auto: np.ndarray,
    w_B: np.ndarray,
    kernel: str = "lanczos",
    hole: bool = True,
) -> np.ndarray:
    """Expected chain F_ref floor (h, w), centred k order, up to one scale.

    ``env_auto`` is the stored (h, w) production envelope; ``w_B`` the
    (sh, sw) correlation pixel weight (see module docstring for the per-chain
    weight semantics — the std chain is the same formula, not a special
    case). The central (h, w) crop of the (sh, sw) lag plane is implied by
    the two shapes; for std passes sh == h and the crop is a no-op.

    ``hole=False`` drops the two rank-1 mean-removal terms (background
    methods without ``window_mean``).
    """
    # float64 throughout: production weights arrive float32
    # (_window_weight_fun), and float32 tap sums break the exact E_S(0)=0
    # identity guard below (the offline scripts cast on load for the same
    # reason)
    env_auto = np.asarray(env_auto, dtype=np.float64)
    w_B = np.asarray(w_B, dtype=np.float64)
    h, w = env_auto.shape
    sh, sw = w_B.shape
    taus_x, r_x = tap_autocorr(fx, kernel)
    taus_y, r_y = tap_autocorr(fy, kernel)

    # v = C w_B: separable zero-padded correlation, x taps then y taps.
    # v(x) = sum_tau r(tau) w(x + tau); taps beyond the window edge vanish
    # because w_B is only supported inside the window.
    vx = np.zeros_like(w_B)
    for tau, rv in zip(taus_x, r_x):
        if rv != 0.0:
            vx += rv * _shift_pad(w_B, int(tau), axis=1)
    v = np.zeros_like(w_B)
    for tau, rv in zip(taus_y, r_y):
        if rv != 0.0:
            v += rv * _shift_pad(vx, int(tau), axis=0)

    sumB = w_B.sum()
    q = float((w_B * v).sum())

    # T1(k): expected weighted periodogram (no mean removal).
    # A(tau_x, tau_y) = 2-D zero-padded lag autocorrelation of w_B; support
    # is the tap-autocorr span per axis (<= +-5), so the double sum is tiny.
    my = np.arange(sh)
    mx = np.arange(sw)
    phase_y = np.exp(-2j * np.pi * np.outer(my, taus_y) / sh)  # (sh, n_ty)
    phase_x = np.exp(-2j * np.pi * np.outer(mx, taus_x) / sw)  # (sw, n_tx)
    T1 = np.zeros((sh, sw))
    for iy, (ty, ry) in enumerate(zip(taus_y, r_y)):
        if ry == 0.0:
            continue
        wy = _shift_pad(w_B, int(ty), axis=0)
        for ix, (tx, rx) in enumerate(zip(taus_x, r_x)):
            if rx == 0.0:
                continue
            A = float((w_B * _shift_pad(wy, int(tx), axis=1)).sum())
            if A == 0.0:
                continue
            T1 += (ry * rx * A) * np.real(np.outer(phase_y[:, iy], phase_x[:, ix]))

    if hole:
        W_hat = np.fft.fft2(w_B)
        V_hat = np.fft.fft2(w_B * v)
        E_S = (
            T1
            - (2.0 / sumB) * np.real(np.conj(W_hat) * V_hat)
            + (q / sumB**2) * np.abs(W_hat) ** 2
        )
        # weighted-mean hole: exact zero at DC (T1 = q, W = sumB, V = q).
        # This also cross-checks the independent T1 and v code paths:
        # T1(0,0) = sum r_x r_y A = w.(C w) = q.
        if abs(E_S[0, 0]) > 1e-9 * q:
            raise AssertionError(
                f"E_S(0,0) = {E_S[0, 0]:.3e} not ~0 (q = {q:.3e}) — algebra broken"
            )
        E_S[0, 0] = 0.0
    else:
        E_S = T1

    # post-chain identical to production: lag space at sum size, central
    # (h, w) crop, stored envelope divide, centred FFT
    R = np.fft.fftshift(np.fft.ifft2(E_S).real)
    y0, x0 = (sh - h) // 2, (sw - w) // 2
    F = _fft_planes((R[y0 : y0 + h, x0 : x0 + w] / env_auto)[None], h, w)[0]
    return np.abs(F)


def build_P_grid(
    env_auto: np.ndarray,
    kernel: str,
    hole: bool,
    weight_B: np.ndarray,
    *,
    n_fracs: int = N_FRACS_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic P over the full (fx, fy) fraction grid, normalized at (0, 0).

    Returns ``(fs, P_grid)`` with ``fs = linspace(0, 0.5, n_fracs)`` and
    ``P_grid`` shaped (n_fracs, n_fracs, h, w) indexed ``[iy, ix]`` (fy
    varies along axis 0, fx along axis 1). Normalization: unit mean at
    (fx, fy) = (0, 0), matching the offline P_grid convention — the fitted
    per-window N0 absorbs the absolute level, only the shape matters.
    """
    if weight_B.ndim != 2:
        raise ValueError(f"weight_B must be 2-D, got shape {weight_B.shape}")
    h, w = env_auto.shape
    sh, sw = weight_B.shape
    if sh < h or sw < w:
        raise ValueError(
            f"weight_B shape {weight_B.shape} smaller than envelope {env_auto.shape}"
        )
    fs = np.linspace(0.0, 0.5, n_fracs)
    P_grid = np.empty((n_fracs, n_fracs, h, w))
    for iy, fyv in enumerate(fs):
        for ix, fxv in enumerate(fs):
            P_grid[iy, ix] = analytic_floor_single(
                float(fxv), float(fyv), env_auto, weight_B, kernel, hole
            )
    P_grid /= P_grid[0, 0].mean()
    return fs, P_grid


def interp_P(
    fs: np.ndarray, P_grid: np.ndarray, fx: np.ndarray, fy: np.ndarray
) -> np.ndarray:
    """Bilinear interpolation of the (fx, fy) grid per window.

    ``fx``/``fy`` are (n,) per-window fractional shifts in [0, 0.5]
    (``frac_distance(pred/2)``). Returns (n, h*w) float64, flattened per
    window in C order — ready to slice alongside the flat plane arrays.
    """
    fx = np.asarray(fx, dtype=np.float64).ravel()
    fy = np.asarray(fy, dtype=np.float64).ravel()
    if fx.shape != fy.shape:
        raise ValueError(f"fx/fy length mismatch: {fx.shape} vs {fy.shape}")
    tol = 1e-9
    for name, f in (("fx", fx), ("fy", fy)):
        if f.min() < fs[0] - tol or f.max() > fs[-1] + tol:
            raise ValueError(
                f"{name} outside the P grid range [{fs[0]}, {fs[-1]}]: "
                f"min {f.min()}, max {f.max()}"
            )

    nf = len(fs)
    ix = np.clip(np.searchsorted(fs, fx) - 1, 0, nf - 2)
    iy = np.clip(np.searchsorted(fs, fy) - 1, 0, nf - 2)
    tx = np.clip((fx - fs[ix]) / (fs[ix + 1] - fs[ix]), 0.0, 1.0)
    ty = np.clip((fy - fs[iy]) / (fs[iy + 1] - fs[iy]), 0.0, 1.0)

    Pf = P_grid.reshape(nf, nf, -1)
    return (
        ((1.0 - ty) * (1.0 - tx))[:, None] * Pf[iy, ix]
        + ((1.0 - ty) * tx)[:, None] * Pf[iy, ix + 1]
        + (ty * (1.0 - tx))[:, None] * Pf[iy + 1, ix]
        + (ty * tx)[:, None] * Pf[iy + 1, ix + 1]
    )
