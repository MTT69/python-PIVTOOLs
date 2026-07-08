"""Shared helpers for the k-space ensemble fitters.

Hosts the pieces used by both the production LM fitter (``kspace_lm_fitting``) and the
dormant closed-form fitter (``kspace_linear_fitting``): the centred FFT/wavenumber-grid
conventions and the process-cached ThreadpoolController. Extracted 2026-07-08 so the
dormant linear module carries nothing the live path depends on.
"""

from typing import Optional

import numpy as np
from threadpoolctl import ThreadpoolController

# Process-cached ThreadpoolController (same pattern as sklearn.utils.parallel).
# Constructing one enumerates every DLL loaded in the process and re-opens each
# recognised BLAS/OpenMP library via ctypes; on Windows that enumeration is
# nondeterministically unsafe (threadpoolctl<=3.6.0 passes GetModuleFileNameExW's
# nSize argument by reference instead of by value, so returned paths can be
# silently truncated -> transient FileNotFoundError on libscipy_openblas64).
# Enumerate once per worker process and reuse; .limit() on a cached controller
# only sets/restores thread counts through already-resolved handles.
_THREADPOOL_CONTROLLER: Optional[ThreadpoolController] = None


def _get_threadpool_controller() -> ThreadpoolController:
    global _THREADPOOL_CONTROLLER
    if _THREADPOOL_CONTROLLER is None:
        _THREADPOOL_CONTROLLER = ThreadpoolController()
    return _THREADPOOL_CONTROLLER


def _kgrids(corr_h, corr_w):
    """Centred (fftshifted) wavenumber grids and derived quantities, cycles/pixel."""
    kx = np.fft.fftshift(np.fft.fftfreq(corr_w))
    ky = np.fft.fftshift(np.fft.fftfreq(corr_h))
    KX, KY = np.meshgrid(kx, ky, indexing="xy")  # shape (corr_h, corr_w)
    KR = np.sqrt(KX * KX + KY * KY)
    return KX, KY, KR


def _fft_planes(R, corr_h, corr_w):
    """Centred 2D FFT of a stack of correlation planes, shape (n, h, w) -> (n, h, w) complex.

    Convention: F = fftshift(fft2(ifftshift(R))).
    """
    R = R.reshape(-1, corr_h, corr_w)
    shifted = np.fft.ifftshift(R, axes=(1, 2))
    F = np.fft.fft2(shifted, axes=(1, 2))
    return np.fft.fftshift(F, axes=(1, 2))
