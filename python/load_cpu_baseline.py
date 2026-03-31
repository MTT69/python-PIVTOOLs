"""
Load the existing CPU C libraries (libbulkxcorr2d, libfusedwarp) for baseline comparison.

Falls back to scipy/numpy if the C libraries aren't found.
"""

import ctypes
import os
import sys
import numpy as np
from numpy.ctypeslib import ndpointer

# Path to the worktree's compiled C libraries (with exported xcorr functions)
_LIB_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "pivtools_cli", "lib"
))
_EXT = ".dll" if os.name == "nt" else ".so"


def _load_lib(name: str):
    """Try to load a C shared library from the main codebase."""
    path = os.path.join(_LIB_DIR, f"{name}{_EXT}")
    if not os.path.isfile(path):
        return None
    try:
        return ctypes.CDLL(path)
    except OSError as e:
        print(f"Warning: Could not load {path}: {e}")
        return None


def load_xcorr_lib():
    """Load libbulkxcorr2d and set up the convolve() function signature."""
    lib = _load_lib("libbulkxcorr2d")
    if lib is None:
        return None

    try:
        lib.convolve.argtypes = [
            ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fA
            ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fB
            ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fC (output)
            ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # N [h, w]
        ]
        lib.convolve.restype = ctypes.c_uint
        return lib
    except AttributeError:
        print("Warning: convolve() not found in libbulkxcorr2d (may not be exported on Windows)")
        return None


def load_fused_warp_lib():
    """Load libfusedwarp and set up fused_symmetric_warp_batch() signature."""
    lib = _load_lib("libfusedwarp")
    if lib is None:
        return None

    try:
        lib.fused_symmetric_warp_batch.argtypes = [
            ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # imgs_a
            ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # imgs_b
            ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # outs_a
            ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # outs_b
            ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # pred_dy
            ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # pred_dx
            ctypes.c_int,                                        # N
            ctypes.c_int, ctypes.c_int,                          # H, W
            ctypes.c_int, ctypes.c_int,                          # nPY, nPX
            ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # ctrs_y
            ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # ctrs_x
            ctypes.c_int,                                        # interp_mode
            ctypes.c_int,                                        # shared_predictor
        ]
        lib.fused_symmetric_warp_batch.restype = ctypes.c_int
        return lib
    except AttributeError:
        print("Warning: fused_symmetric_warp_batch() not found in libfusedwarp")
        return None


def cpu_xcorr_scipy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Cross-correlation using scipy FFT (fallback if C library not available).

    Replicates the xcorr.c pipeline:
      zero-pad centred → FFT → conjugate multiply → IFFT → normalise → fftshift → extract
    """
    from scipy.fft import fft2, ifft2, fftshift

    win_h, win_w = A.shape
    pad_h, pad_w = win_h * 2, win_w * 2

    # Zero-pad with centred placement
    padA = np.zeros((pad_h, pad_w), dtype=np.float32)
    padB = np.zeros((pad_h, pad_w), dtype=np.float32)
    oh, ow = win_h // 2, win_w // 2
    padA[oh:oh + win_h, ow:ow + win_w] = A
    padB[oh:oh + win_h, ow:ow + win_w] = B

    # FFT cross-correlation
    FA = fft2(padA)
    FB = fft2(padB)
    FC = FA * np.conj(FB)
    C = np.real(ifft2(FC)).astype(np.float32)

    # fftshift + extract central region
    C = fftshift(C)
    C = C[oh:oh + win_h, ow:ow + win_w].copy()

    return C


def cpu_bicubic_warp_scipy(img: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """
    Bicubic warp using scipy (fallback if C library not available).
    Uses map_coordinates with order=3 (cubic spline, NOT Keys a=-0.75).
    Results won't match exactly but are close enough for timing comparison.
    """
    from scipy.ndimage import map_coordinates

    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    coords = np.array([yy + dy, xx + dx])
    return map_coordinates(img, coords, order=3, mode='constant', cval=0.0).astype(np.float32)
