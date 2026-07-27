"""Seeded mini Monte-Carlo white-noise floor simulation — the test referee.

Compact port of ``TurbulentChannel/scripts/white_noise_floor_probe.simulate_floor``
(the validation oracle the analytic P was gated against offline at 1024 pairs,
worst per-cell median deviation 0.15-0.18%). Pushes pure white noise through
the exact production chain — fractional warp by the fused-warp kernel taps,
window extraction on the pass stride, (weighted) per-pair window-mean removal,
circular correlation accumulation, mean-image subtraction, central crop
(single chain), stored envelope divide, centred FFT — and returns the median
F_ref floor over windows.

Always ``normalize=False`` semantics (no peak normalization): that is the
model-P convention, where the per-window level is absorbed by the fitted N0.
"""

import numpy as np

from pivtools_cli.piv.piv_backend.kspace_common import _fft_planes
from pivtools_cli.piv.piv_backend.kspace_floor_psd import KERNEL_TAPS


def frac_shift(
    img: np.ndarray, f: float, kernel: str = "lanczos", axis: int = 1
) -> np.ndarray:
    """Shift img by fractional f along axis with the fused-warp kernel taps."""
    if f == 0.0:
        return img
    offsets, weights_fun = KERNEL_TAPS[kernel]
    w = weights_fun(f)
    out = np.zeros_like(img)
    for wt, d in zip(w, offsets):
        if wt != 0.0:
            out += wt * np.roll(img, -int(d), axis=axis)
    return out


def extract_windows(img: np.ndarray, h: int, w: int, sy: int, sx: int) -> np.ndarray:
    """(n, h, w) view of all windows on the stride grid."""
    H, W = img.shape
    ny = (H - h) // sy + 1
    nx = (W - w) // sx + 1
    s = img.strides
    v = np.lib.stride_tricks.as_strided(
        img, shape=(ny, nx, h, w), strides=(sy * s[0], sx * s[1], s[0], s[1])
    )
    return v.reshape(ny * nx, h, w)


def simulate_floor(
    fx: float,
    env_auto: np.ndarray,
    n_pairs: int,
    img_size: int,
    h: int,
    w: int,
    rng: np.random.Generator,
    *,
    weights: tuple[np.ndarray, np.ndarray] | None = None,
    kernel: str = "lanczos",
    fy: float = 0.0,
    window_mean: bool = True,
) -> np.ndarray:
    """Median F_ref(k) floor over windows for white noise at warp fraction (fx, fy).

    ``weights=None`` simulates the STD chain (flat weights, extraction and
    correlation at the (h, w) plane size). ``weights=(w_A, w_B)`` simulates
    the SINGLE chain: extraction at the sum-window size ``w_B.shape``, matched
    w_B-weighted means, singlepix/bsingle weighting, circular correlation at
    the sum window, central (h, w) crop, then the stored envelope divide.
    ``window_mean=False`` skips the per-pair (weighted) window-mean removal in
    both chains (background methods without 'window_mean').
    """
    if weights is not None:
        w_A = np.asarray(weights[0], dtype=np.float64)
        w_B = np.asarray(weights[1], dtype=np.float64)
        sh, sw = w_B.shape
        sy, sx = sh // 2, sw // 2
        sumB = w_B.sum()
        S_AA = S_BB = None
        M_Ab = M_B = None
        for _ in range(n_pairs):
            A = frac_shift(
                frac_shift(rng.standard_normal((img_size, img_size)), fx, kernel),
                fy,
                kernel,
                axis=0,
            )
            B = frac_shift(
                frac_shift(rng.standard_normal((img_size, img_size)), fx, kernel),
                fy,
                kernel,
                axis=0,
            )
            wa = extract_windows(A, sh, sw, sy, sx).copy()
            wb = extract_windows(B, sh, sw, sy, sx).copy()
            if window_mean:
                # matched means: every buffer removes the w_B-weighted mean
                mA = (wa * w_B).sum(axis=(1, 2), keepdims=True) / sumB
                mB = (wb * w_B).sum(axis=(1, 2), keepdims=True) / sumB
                wa = wa - mA
                wb = wb - mB
            FA_b = np.fft.fft2(wa * w_B)  # AA auto (sum-window weighted)
            FB = np.fft.fft2(wb * w_B)  # BB auto
            if S_AA is None:
                S_AA = np.abs(FA_b) ** 2
                S_BB = np.abs(FB) ** 2
                M_Ab, M_B = FA_b.copy(), FB.copy()
            else:
                S_AA += np.abs(FA_b) ** 2
                S_BB += np.abs(FB) ** 2
                M_Ab += FA_b
                M_B += FB
        n = float(n_pairs)
        K_AA = S_AA / n - np.abs(M_Ab / n) ** 2
        K_BB = S_BB / n - np.abs(M_B / n) ** 2
        y0, x0 = (sh - h) // 2, (sw - w) // 2
        R_AA = np.fft.fftshift(np.fft.ifft2(K_AA).real, axes=(-2, -1))[
            :, y0 : y0 + h, x0 : x0 + w
        ]
        R_BB = np.fft.fftshift(np.fft.ifft2(K_BB).real, axes=(-2, -1))[
            :, y0 : y0 + h, x0 : x0 + w
        ]
        R_AA = R_AA / env_auto
        R_BB = R_BB / env_auto
    else:
        sy, sx = h // 2, w // 2
        S_AA = S_BB = None
        S_A = S_B = None
        for _ in range(n_pairs):
            A = frac_shift(
                frac_shift(rng.standard_normal((img_size, img_size)), fx, kernel),
                fy,
                kernel,
                axis=0,
            )
            B = frac_shift(
                frac_shift(rng.standard_normal((img_size, img_size)), fx, kernel),
                fy,
                kernel,
                axis=0,
            )
            wa = extract_windows(A, h, w, sy, sx).copy()
            wb = extract_windows(B, h, w, sy, sx).copy()
            if window_mean:
                wa -= wa.mean(axis=(1, 2), keepdims=True)
                wb -= wb.mean(axis=(1, 2), keepdims=True)
            FA = np.fft.fft2(wa)
            FB = np.fft.fft2(wb)
            if S_AA is None:
                S_AA = np.abs(FA) ** 2
                S_BB = np.abs(FB) ** 2
                S_A, S_B = FA.copy(), FB.copy()
            else:
                S_AA += np.abs(FA) ** 2
                S_BB += np.abs(FB) ** 2
                S_A += FA
                S_B += FB
        n = float(n_pairs)
        K_AA = S_AA / n - np.abs(S_A / n) ** 2
        K_BB = S_BB / n - np.abs(S_B / n) ** 2
        R_AA = np.fft.fftshift(np.fft.ifft2(K_AA).real, axes=(-2, -1)) / env_auto
        R_BB = np.fft.fftshift(np.fft.ifft2(K_BB).real, axes=(-2, -1)) / env_auto
    F_AA = _fft_planes(R_AA, h, w)
    F_BB = _fft_planes(R_BB, h, w)
    Fr = np.sqrt(np.abs(F_AA) * np.abs(F_BB))
    return np.median(Fr, axis=0)  # (h, w)
