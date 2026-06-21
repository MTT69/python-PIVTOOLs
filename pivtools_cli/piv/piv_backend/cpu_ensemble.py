"""
Ensemble PIV Correlator for PyPIVTools

This module implements ensemble PIV processing where correlation planes from
multiple image pairs are averaged before peak fitting using Levenberg-Marquardt
Gaussian fitting.

Adapted from con_tools ensemble implementation to follow PyPIVTools production
conventions for config, masking, infilling, and save patterns.
"""

import ctypes
import logging
import os
import time
import traceback
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from pivtools_core.config import Config
from pivtools_core.window_utils import (
    compute_window_centers,
    compute_window_centers_single_mode,
)
from pivtools_cli.piv.piv_backend.base import CrossCorrelator
from pivtools_cli.piv.piv_result import PIVEnsembleBlockResult
from pivtools_cli.piv.piv_backend.infilling import apply_infilling

import matplotlib
matplotlib.use("Agg") 
class EnsembleCorrelatorCPU(CrossCorrelator):
    """
    Ensemble PIV correlator using CPU with Levenberg-Marquardt Gaussian
    fitting.

    This correlator averages correlation planes across multiple image pairs
    before fitting 2D stacked Gaussians to extract sub-pixel displacements
    and uncertainty estimates.
    """

    # Class-level cache for libraries to avoid DLL thrashing
    _lib_corr = None
    _lib_marq = None
    _lib_fw = None

    def __init__(
        self,
        config: Config,
        precomputed_cache: Optional[dict] = None,
        vector_masks: Optional[List[np.ndarray]] = None,
        active_pass_idx: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Store config for interpolation settings
        self.config = config
        self.printed_passes = set()

        # Load libraries ONLY if not already loaded in this process
        if EnsembleCorrelatorCPU._lib_corr is None:
            self._load_libraries()

        # Use the cached class attributes
        self.lib = EnsembleCorrelatorCPU._lib_corr
        self.marquadt_lib = EnsembleCorrelatorCPU._lib_marq
        self._fw_lib = EnsembleCorrelatorCPU._lib_fw
        self.lib.bulkxcorr2d.restype = ctypes.c_ubyte
        self.lib.bulkxcorr2d.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_bool,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ]

        # New ensemble accumulation function (Option C: window-parallel)
        self.lib.bulkxcorr2d_accumulate.restype = ctypes.c_ubyte
        self.lib.bulkxcorr2d_accumulate.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageA_stack
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageB_stack
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fMask
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nImageSize
            ctypes.c_int,                                                      # N_images
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsX
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsY
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nWindows
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWindowWeightA
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWindowWeightB
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nWindowSize (FFT computation)
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nFitWindowSize (output size)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fCorrelPlane_Sum (output)
        ]

        # Fused triple correlation function (AB + AA + BB in one C call)
        self.lib.bulkxcorr2d_accumulate_triple.restype = ctypes.c_ubyte
        self.lib.bulkxcorr2d_accumulate_triple.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageA_stack
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageB_stack
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fMask
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nImageSize
            ctypes.c_int,                                                      # N_images
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsX
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsY
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nWindows
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWindowWeightA_AB
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWindowWeightB_AB
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fAutoWeightA
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fAutoWeightB
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nWindowSize (FFT)
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nFitWindowSize (output)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fCorrAB_Sum (output)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fCorrAA_Sum (output)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fCorrBB_Sum (output)
        ]

        # Initialize window weights for each pass
        # For single mode, Frame A and Frame B use different weights
        self.win_weights_A = []
        self.win_weights_B = []
        self.window_sizes_for_computation = []  # FFT computation size (sum_window)
        self.window_sizes_for_corr = []  # Output size for storage/fitting

        # Get fit_window if enabled (applies only to single mode passes)
        fit_window = config.ensemble_sum_fitting_window  # None if disabled

        for pass_idx, win_size in enumerate(config.ensemble_window_sizes):
            runtype = config.ensemble_type[pass_idx]
            sum_window = tuple(config.ensemble_sum_window)

            if runtype == 'single':
                # Single mode: Frame A uses small weighted window
                weight_A = np.ascontiguousarray(
                    self._window_weight_fun(win_size, 'singlepix', sum_window)
                )
                weight_B = np.ascontiguousarray(
                    self._window_weight_fun(sum_window, 'bsingle', sum_window)
                )
                # Computation uses full sum_window for FFT
                self.window_sizes_for_computation.append(sum_window)
                # Output size: fit_window if enabled, else full sum_window
                corr_size = tuple(fit_window) if fit_window else sum_window
            else:
                # Standard mode: both frames use same window
                weight = np.ascontiguousarray(
                    self._window_weight_fun(win_size, config.ensemble_window_type)
                )
                weight_A = weight
                weight_B = weight
                # Standard mode: compute and output at same size
                self.window_sizes_for_computation.append(win_size)
                corr_size = win_size

            self.win_weights_A.append(weight_A)
            self.win_weights_B.append(weight_B)
            self.window_sizes_for_corr.append(corr_size)

        # Use precomputed cache if provided, otherwise compute it
        if precomputed_cache is not None:
            self._load_precomputed_cache(precomputed_cache)
        else:
            self._cache_window_padding_ensemble(config=config)
            self.H, self.W = config.image_shape
            self._cache_interpolation_grids_ensemble(config=config)

        # Initialize vector masks
        self.vector_masks = vector_masks if vector_masks is not None else []

        # Pre-allocate correlation plane buffers (reused across batches).
        # When active_pass_idx is set, only allocate for that single pass —
        # saves ~491 MB per worker for a 4-pass config at 2048x2048.
        passes_to_allocate = (
            [active_pass_idx] if active_pass_idx is not None
            else range(config.ensemble_num_passes)
        )
        self._corr_buffers = {}
        for pass_idx in passes_to_allocate:
            corr_size = self.window_sizes_for_corr[pass_idx]
            n_win_y = len(self.win_ctrs_y[pass_idx])
            n_win_x = len(self.win_ctrs_x[pass_idx])
            total_windows = n_win_y * n_win_x
            plane_size = total_windows * corr_size[0] * corr_size[1]

            # Pre-allocate buffers for AA, BB, AB
            self._corr_buffers[pass_idx] = {
                'AA': np.zeros(plane_size, dtype=np.float32),
                'BB': np.zeros(plane_size, dtype=np.float32),
                'AB': np.zeros(plane_size, dtype=np.float32),
            }

            logging.debug(
                f"Pre-allocated correlation buffers for pass {pass_idx}: "
                f"{plane_size * 4 * 3 / 1024 / 1024:.1f} MB"
            )

        # Disable OpenCV's internal threading — we manage parallelism ourselves
        # via the thread pool below. Without this, each cv2.remap spawns its own
        # threads, causing N_pool × N_opencv contention.
        cv2.setNumThreads(1)

        # Reusable thread pool for GIL-releasing operations (cv2.remap).
        # Sized to omp_threads — safe because the pool is idle while
        # bulkxcorr2d_accumulate runs (OpenMP) and vice versa.
        self._n_threads = max(1, int(config.omp_threads))
        self._pool = ThreadPoolExecutor(max_workers=self._n_threads)

        # Section-level profiling (disabled by default, zero overhead)
        self.profiling_enabled = False
        self.profile_data = {}  # {pass_idx: {section_name: elapsed_seconds}}

    @contextmanager
    def _profile_section(self, pass_idx, section):
        """Context manager for timing named sections within correlate_batch_for_accumulation."""
        if not self.profiling_enabled:
            yield
            return
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self.profile_data.setdefault(pass_idx, {})[section] = (
            self.profile_data.get(pass_idx, {}).get(section, 0.0) + elapsed
        )

    def get_profile_summary(self):
        """Return a copy of the profile data collected across all batches."""
        return dict(self.profile_data)

    def reset_profile_data(self):
        """Clear profile data for a new profiling run."""
        self.profile_data = {}

    @classmethod
    def _load_libraries(cls):
        """Load C libraries once per process to avoid DLL thrashing."""
        logging.debug("Loading C libraries (One-time init)...")

        lib_extension = ".dll" if os.name == "nt" else ".so"

        # On Windows a clang-cl /openmp build ships libomp.dll beside our DLLs;
        # Windows does not search a dependent DLL's own dir by default, so make it
        # discoverable for the loads below. No-op when libomp.dll isn't present (MSVC).
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
            )

        # Load marquadt library for Gaussian fitting
        marquadt_libpath = os.path.join(
            os.path.dirname(__file__), "..", "..", "lib", f"libmarquadt{lib_extension}"
        )
        marquadt_libpath = os.path.abspath(marquadt_libpath)

        if not os.path.isfile(marquadt_libpath):
            raise FileNotFoundError(
                f"Marquadt library not found: {marquadt_libpath}. "
                "Ensure GSL is installed and run 'pip install -e .' to build."
            )

        cls._lib_marq = ctypes.CDLL(marquadt_libpath)
        # Note: Only batch function is used now (fit_stacked_gaussian_batch_export)
        # Single-window function was removed in the optimized matrix-free LM solver

        # Load cross-correlation library
        lib_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "lib", f"libbulkxcorr2d{lib_extension}"
        )
        lib_path = os.path.abspath(lib_path)

        if not os.path.isfile(lib_path):
            raise FileNotFoundError(
                f"Cross-correlation library not found: {lib_path}. "
                "Ensure the library is built and available."
            )

        cls._lib_corr = ctypes.CDLL(lib_path)

        # Load fused warp library (required)
        fw_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "lib", f"libfusedwarp{lib_extension}"
        )
        fw_path = os.path.abspath(fw_path)
        if not os.path.isfile(fw_path):
            raise FileNotFoundError(f"Required library file not found: {fw_path}")
        cls._lib_fw = ctypes.CDLL(fw_path)
        cls._lib_fw.fused_symmetric_warp_batch.restype = ctypes.c_int
        cls._lib_fw.fused_symmetric_warp_batch.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # imgs_a (N,H,W)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # imgs_b (N,H,W)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # outs_a (N,H,W)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # outs_b (N,H,W)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # pred_dy
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # pred_dx
            ctypes.c_int,                # N
            ctypes.c_int, ctypes.c_int,  # H, W
            ctypes.c_int, ctypes.c_int,  # nPY, nPX
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # ctrs_y
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # ctrs_x
            ctypes.c_int,                # interp_mode
            ctypes.c_int,                # shared_predictor
        ]

        cls._lib_corr.bulkxcorr2d.restype = ctypes.c_ubyte
        cls._lib_corr.bulkxcorr2d.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_bool,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ]

        # New ensemble accumulation function (Option C: window-parallel)
        cls._lib_corr.bulkxcorr2d_accumulate.restype = ctypes.c_ubyte
        cls._lib_corr.bulkxcorr2d_accumulate.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageA_stack
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageB_stack
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fMask
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nImageSize
            ctypes.c_int,                                                      # N_images
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsX
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsY
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nWindows
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWindowWeightA
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWindowWeightB
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nWindowSize
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fCorrelPlane_Sum (output)
        ]

    def _run_correlation_kernel(
        self, img1, img2, weight1, weight2, b_mask, common_args
    ):
        """
        Wraps the heavy C arguments for bulkxcorr2d calls.

        Parameters
        ----------
        img1 : np.ndarray
            First image
        img2 : np.ndarray
            Second image
        weight1 : np.ndarray
            Weight for first image
        weight2 : np.ndarray
            Weight for second image
        b_mask : np.ndarray
            Mask array
        common_args : tuple
            Common arguments: (image_size, wx, wy, n_win, w_size, n_peaks,
                             i_peak, pk_x, pk_y, pk_h, sx, sy, sxy, out_plane)

        Returns
        -------
        tuple
            (error_code, out_plane)
        """
        # Unpack common args
        (N, img_size, wx, wy, n_win, w_size, n_peaks, i_peak, pk_x, pk_y,
         pk_h, sx, sy, sxy, out_plane) = common_args

        # Clear output plane before use
        out_plane.fill(0)

        err = self.lib.bulkxcorr2d(
            np.ascontiguousarray(img1),
            np.ascontiguousarray(img2),
            b_mask,
            img_size,
            N,
            wx, wy, n_win,
            weight1,
            True,  # b_ensemble
            weight2,
            w_size,
            n_peaks, i_peak,
            pk_x, pk_y, pk_h, sx, sy, sxy,
            out_plane
        )
        return err, out_plane

    def _run_correlation_accumulate(
        self,
        images_a: np.ndarray,
        images_b: np.ndarray,
        weight_a: np.ndarray,
        weight_b: np.ndarray,
        mask: np.ndarray,
        pass_idx: int,
        correl_sum: np.ndarray,
    ) -> int:
        """
        Compute cross-correlation with internal accumulation (Option C).

        This function uses the new bulkxcorr2d_accumulate C function which
        parallelizes over windows and accumulates across images internally.
        Output is the SUM across all N images (not individual planes).

        Memory usage: O(windows × corr_size²) instead of O(N × windows × corr_size²)

        Parameters
        ----------
        images_a : np.ndarray
            Stack of first images, shape (N, H, W)
        images_b : np.ndarray
            Stack of second images, shape (N, H, W)
        weight_a : np.ndarray
            Window weights for image A
        weight_b : np.ndarray
            Window weights for image B
        mask : np.ndarray
            Window mask (1 = skip, 0 = process)
        pass_idx : int
            Current pass index
        correl_sum : np.ndarray
            Pre-allocated output buffer, shape (windows × corr_size²)

        Returns
        -------
        int
            Error code (0 = success)
        """
        N = images_a.shape[0]
        H, W = images_a.shape[1], images_a.shape[2]

        image_size = np.array([H, W], dtype=np.int32)
        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])
        n_windows = np.array([n_win_y, n_win_x], dtype=np.int32)

        # Get computation size (FFT) and output size (storage/fitting)
        comp_size = self.window_sizes_for_computation[pass_idx]
        out_size = self.window_sizes_for_corr[pass_idx]

        win_size_arr = np.array([comp_size[0], comp_size[1]], dtype=np.int32)
        fit_size_arr = np.array([out_size[0], out_size[1]], dtype=np.int32)

        # Add padding offset for single mode passes.
        # win_ctrs are stored in original image coords; the C library
        # receives a padded image, so centers must be offset by the padding.
        # For standard mode passes, padding is (0,0,0,0) so this is a no-op.
        pad_top, pad_bottom, pad_left, pad_right = self.padding_per_pass[pass_idx]

        error_code = self.lib.bulkxcorr2d_accumulate(
            np.ascontiguousarray(images_a, dtype=np.float32),
            np.ascontiguousarray(images_b, dtype=np.float32),
            mask,
            image_size,
            N,
            (self.win_ctrs_x[pass_idx] + pad_left).astype(np.float32),
            (self.win_ctrs_y[pass_idx] + pad_top).astype(np.float32),
            n_windows,
            weight_a,
            weight_b,
            win_size_arr,   # FFT computation size
            fit_size_arr,   # Output size (may be smaller for central extraction)
            correl_sum,
        )

        return error_code

    def _run_correlation_accumulate_triple(
        self,
        images_a: np.ndarray,
        images_b: np.ndarray,
        weight_a_ab: np.ndarray,
        weight_b_ab: np.ndarray,
        auto_weight_a: np.ndarray,
        auto_weight_b: np.ndarray,
        mask: np.ndarray,
        pass_idx: int,
        correl_AB_sum: np.ndarray,
        correl_AA_sum: np.ndarray,
        correl_BB_sum: np.ndarray,
    ) -> int:
        """
        Compute AB, AA, BB cross-correlations in a single fused C call.

        Reuses FFT(A) and FFT(B) across all three products, eliminating
        redundant forward FFTs. ~50% fewer forward FFTs than 3× separate calls.

        Parameters
        ----------
        images_a, images_b : np.ndarray
            Image stacks, shape (N, H, W)
        weight_a_ab, weight_b_ab : np.ndarray
            Taper weights for AB cross-correlation (may be asymmetric)
        auto_weight_a, auto_weight_b : np.ndarray
            Taper weights for AA/BB auto-correlations (symmetric)
        mask : np.ndarray
            Window mask (1=skip, 0=process)
        pass_idx : int
            Current pass index
        correl_AB_sum, correl_AA_sum, correl_BB_sum : np.ndarray
            Pre-allocated output buffers

        Returns
        -------
        int
            Error code (0 = success)
        """
        N = images_a.shape[0]
        H, W = images_a.shape[1], images_a.shape[2]

        image_size = np.array([H, W], dtype=np.int32)
        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])
        n_windows = np.array([n_win_y, n_win_x], dtype=np.int32)

        comp_size = self.window_sizes_for_computation[pass_idx]
        out_size = self.window_sizes_for_corr[pass_idx]

        win_size_arr = np.array([comp_size[0], comp_size[1]], dtype=np.int32)
        fit_size_arr = np.array([out_size[0], out_size[1]], dtype=np.int32)

        pad_top, pad_bottom, pad_left, pad_right = self.padding_per_pass[pass_idx]

        error_code = self.lib.bulkxcorr2d_accumulate_triple(
            np.ascontiguousarray(images_a, dtype=np.float32),
            np.ascontiguousarray(images_b, dtype=np.float32),
            mask,
            image_size,
            N,
            (self.win_ctrs_x[pass_idx] + pad_left).astype(np.float32),
            (self.win_ctrs_y[pass_idx] + pad_top).astype(np.float32),
            n_windows,
            weight_a_ab,
            weight_b_ab,
            auto_weight_a,
            auto_weight_b,
            win_size_arr,
            fit_size_arr,
            correl_AB_sum,
            correl_AA_sum,
            correl_BB_sum,
        )

        return error_code

    def _cache_window_padding_ensemble(self, config: Config) -> None:
        """Cache window padding information for ensemble PIV.

        Uses unified base class implementation with ensemble-specific parameters.
        """
        self._cache_window_padding_unified(
            config=config,
            window_sizes=config.ensemble_window_sizes,
            window_type=config.ensemble_window_type,
            compute_window_fn=self._compute_window_centres_ensemble,
            first_pass_ksize=(1, 1),  # Ensemble uses (1, 1) for first pass
            predictor_boundary_conditions=config.ensemble_predictor_boundary_conditions,
        )

    def _compute_window_centres_ensemble(
        self, pass_idx: int, config: Config
    ) -> tuple[int, int, np.ndarray, np.ndarray, tuple]:
        """
        Compute window centers and spacing for ensemble PIV pass.

        Uses centralized window_utils for consistency with instantaneous mode.
        Supports both standard and single mode ensemble PIV.

        Returns:
            tuple: (win_spacing_x, win_spacing_y, win_ctrs_x, win_ctrs_y, padding)
                   padding is (top, bottom, left, right) - (0,0,0,0) for standard mode
        """
        win_y, win_x = config.ensemble_window_sizes[pass_idx]
        overlap = config.ensemble_overlaps[pass_idx]

        # Check if this pass uses single mode
        runtype = config.ensemble_type[pass_idx]

        if runtype == 'single':
            # Single mode: use sum window for positioning
            result = compute_window_centers_single_mode(
                image_shape=config.image_shape,
                window_size=(win_y, win_x),
                sum_window=tuple(config.ensemble_sum_window),
                overlap=overlap,
                validate=True
            )
            padding = result.padding  # (top, bottom, left, right)
            # Convert from padded coords to original image coords.
            # Single mode returns centers in padded image space; subtract
            # padding so all passes use the same coordinate system.
            # Padding is added back at C library call sites only.
            win_ctrs_x = np.ascontiguousarray(result.win_ctrs_x - padding[2])  # subtract pad_left
            win_ctrs_y = np.ascontiguousarray(result.win_ctrs_y - padding[0])  # subtract pad_top
        else:
            # Standard mode
            result = compute_window_centers(
                image_shape=config.image_shape,
                window_size=(win_y, win_x),
                overlap=overlap,
                validate=True
            )
            padding = (0, 0, 0, 0)  # No padding for standard mode
            win_ctrs_x = np.ascontiguousarray(result.win_ctrs_x)
            win_ctrs_y = np.ascontiguousarray(result.win_ctrs_y)

        return (
            result.win_spacing_x,
            result.win_spacing_y,
            win_ctrs_x,
            win_ctrs_y,
            padding,
        )

    def _cache_interpolation_grids_ensemble(self, config: Config) -> None:
        """Cache interpolation grids for predictor correction in ensemble PIV.

        Uses unified base class implementation with ensemble-specific parameters.
        """
        self._cache_interpolation_grids_unified(
            config=config,
            window_sizes=config.ensemble_window_sizes,
            include_first_pass=True,  # Ensemble needs first pass grids
        )

    def _load_precomputed_cache(self, cache: dict) -> None:
        """Load precomputed cache data for ensemble PIV."""
        self.win_ctrs_x = cache['win_ctrs_x']
        self.win_ctrs_y = cache['win_ctrs_y']
        self.win_spacing_x = cache['win_spacing_x']
        self.win_spacing_y = cache['win_spacing_y']
        self.win_ctrs_x_all = cache['win_ctrs_x_all']
        self.win_ctrs_y_all = cache['win_ctrs_y_all']
        self.n_pre_all = cache['n_pre_all']
        self.n_post_all = cache['n_post_all']
        self.ksize_filt = cache['ksize_filt']
        self.sd = cache['sd']
        self.G_smooth_predictor = cache['G_smooth_predictor']
        self.H = cache['H']
        self.W = cache['W']
        self.im_mesh = cache['im_mesh']
        self.cached_dense_maps = cache['cached_dense_maps']
        self.cached_predictor_maps = cache['cached_predictor_maps']
        self.win_weights_A = cache.get('win_weights_A', [])
        self.win_weights_B = cache.get('win_weights_B', [])
        self.window_sizes_for_corr = cache.get('window_sizes_for_corr', [])
        self.padding_per_pass = cache.get('padding_per_pass', [])
        self.predictor_bcs = cache.get('predictor_bcs', [])

    def get_cache_data(self) -> dict:
        """Extract cache data for sharing across workers."""
        return {
            'win_ctrs_x': self.win_ctrs_x,
            'win_ctrs_y': self.win_ctrs_y,
            'win_spacing_x': self.win_spacing_x,
            'win_spacing_y': self.win_spacing_y,
            'win_ctrs_x_all': self.win_ctrs_x_all,
            'win_ctrs_y_all': self.win_ctrs_y_all,
            'n_pre_all': self.n_pre_all,
            'n_post_all': self.n_post_all,
            'ksize_filt': self.ksize_filt,
            'sd': self.sd,
            'G_smooth_predictor': self.G_smooth_predictor,
            'H': self.H,
            'W': self.W,
            'im_mesh': self.im_mesh,
            'cached_dense_maps': self.cached_dense_maps,
            'cached_predictor_maps': self.cached_predictor_maps,
            'win_weights_A': self.win_weights_A,
            'win_weights_B': self.win_weights_B,
            'window_sizes_for_corr': self.window_sizes_for_corr,
            'padding_per_pass': self.padding_per_pass,
            'predictor_bcs': getattr(self, 'predictor_bcs', []),
        }

    def correlate_batch(self, images: np.ndarray, config: Config, vector_masks: List[np.ndarray] = None):
        """
        Not used for ensemble PIV - use correlate_batch_for_accumulation instead.

        This method exists only to satisfy the abstract base class requirement.
        Ensemble PIV uses a different workflow with correlate_batch_for_accumulation.
        """
        raise NotImplementedError(
            "Ensemble PIV does not use correlate_batch(). "
            "Use correlate_batch_for_accumulation() instead."
        )

    def correlate_batch_for_accumulation(
        self,
        images: np.ndarray,
        config: Config,
        pass_idx: int,
        predictor_field: Optional[np.ndarray] = None,
        save_diagnostics: bool = False,
        output_path: Optional[str] = None,
        is_first_batch: bool = False,
        clear_buffers: bool = True,
        copy_result: bool = True,
    ) -> dict:
        """
        Correlate batch and return SUMS for single-pass accumulation.

        Returns all three correlation planes (AA, BB, AB) needed for
        stacked Gaussian fitting, along with warped image sums.

        This method is used by UnifiedBatchPipeline for streaming ensemble PIV.

        Parameters
        ----------
        images : np.ndarray
            Image batch of shape (N, 2, H, W)
        config : Config
            Configuration object
        pass_idx : int
            PIV pass index
        predictor_field : Optional[np.ndarray]
            Predictor field from previous pass (shape: n_win_y+2, n_win_x+2, 2)
            containing [uy, ux]. None for pass 0.
        save_diagnostics : bool
            If True, save warped images for first pair
        output_path : Optional[str]
            Output directory for diagnostic images
        is_first_batch : bool
            If True, this is the first batch (save diagnostics for first pair)
        clear_buffers : bool
            If True (default), zero correlation buffers before correlating.
            Set False to accumulate across multiple calls (worker accumulation mode).
        copy_result : bool
            If True (default), copy correlation planes into result dict.
            Set False to leave them in self._corr_buffers and return a lightweight
            dict. Call get_accumulated_correlation() once after all batches.

        Returns
        -------
        dict
            Dictionary with keys:
            - corr_AA_sum: Auto-correlation A sum (not mean!)
            - corr_BB_sum: Auto-correlation B sum
            - corr_AB_sum: Cross-correlation sum
            - warp_A_sum: Sum of warped A images
            - warp_B_sum: Sum of warped B images
            - n_images: Number of images in batch
            - n_win_x: Number of windows in x
            - n_win_y: Number of windows in y
            - smoothed_predictor: Smoothed predictor field
        """
        from pivtools_core.window_utils import apply_single_mode_padding

        win_size = config.ensemble_window_sizes[pass_idx]
        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])
        total_windows = n_win_y * n_win_x

        # Check if single mode
        runtype = config.ensemble_type[pass_idx]
        is_single_mode = (runtype == 'single')

        N, _, H, W = images.shape

        # Reuse pre-allocated correlation buffers
        correl_AA_sum = self._corr_buffers[pass_idx]['AA']
        correl_BB_sum = self._corr_buffers[pass_idx]['BB']
        correl_AB_sum = self._corr_buffers[pass_idx]['AB']

        # Clear buffers (faster than reallocation)
        if clear_buffers:
            correl_AA_sum.fill(0)
            correl_BB_sum.fill(0)
            correl_AB_sum.fill(0)

        # Store smoothed predictor (will be set during warping if pass > 0)
        smoothed_predictor = None

        vector_mask = (
            self.vector_masks[pass_idx]
            if self.vector_masks and pass_idx < len(self.vector_masks)
            else None
        )

        # Process each image batch
        try:
            image_a_stack = images[:, 0, :, :].astype(np.float32, copy=False)
            image_b_stack = images[:, 1, :, :].astype(np.float32, copy=False)

            # Warp images if predictor field is provided (pass > 0)
            with self._profile_section(pass_idx, "predictor_corrector"):
                if pass_idx > 0:
                    if predictor_field is None:
                        logging.warning(
                            f"Pass {pass_idx + 1}: predictor_field is None! "
                            f"Cannot perform image warping. Correlating unwarped images."
                        )
                    else:
                        delta_ab_pred = self._get_im_mesh(
                            pass_idx, predictor_field
                        )
                        smoothed_predictor = delta_ab_pred
                        logging.debug(
                            f"Pass {pass_idx + 1}: Got smoothed predictor field "
                            f"(shape: {delta_ab_pred.shape})"
                        )

                        # Apply vector mask to zero out masked vectors
                        if vector_mask is not None:
                            smoothed_predictor[vector_mask] = 0

                with self._profile_section(pass_idx, "pc_fused_warp"):
                    if predictor_field is not None and pass_idx > 0:
                        images_a_prime, images_b_prime = self._fused_warp_batch(
                            image_a_stack, image_b_stack
                        )
                    else:
                        # No warping for pass 0
                        images_a_prime = image_a_stack
                        images_b_prime = image_b_stack

            # Compute warp sums BEFORE padding (for accumulation in original image space)
            # This ensures warp sums have shape (H, W) not (H_padded, W_padded)
            with self._profile_section(pass_idx, "warp_sum"):
                warp_A_sum = images_a_prime.sum(axis=0)
                warp_B_sum = images_b_prime.sum(axis=0)
            logging.debug(f"Pass {pass_idx}: warp_A_sum shape {warp_A_sum.shape} (expected {H}x{W})")

            # Apply padding for single mode (for correlation only)
            with self._profile_section(pass_idx, "single_mode_padding"):
                if is_single_mode:
                    sum_window = tuple(config.ensemble_sum_window)
                    images_a_prime, _ = apply_single_mode_padding(
                            images_a_prime, win_size, sum_window, pad_value=0.0
                        )
                    images_b_prime, _ = apply_single_mode_padding(
                            images_b_prime, win_size, sum_window, pad_value=0.0
                        )
                    H_padded, W_padded = images_a_prime.shape[-2:]
                    image_size = np.ascontiguousarray(np.array([H_padded, W_padded], dtype=np.int32))
                else:
                    image_size = np.ascontiguousarray(np.array([H, W], dtype=np.int32))

        except Exception as e:
            logging.error("Error preprocessing image: %s", e)
            traceback.print_exc()

        try:
            corr_size = self.window_sizes_for_corr[pass_idx]
            plane_size = total_windows * corr_size[0] * corr_size[1]

            # Use pre-allocated buffers (cleared above when clear_buffers=True)

            # Create mask for C library
            if vector_mask is not None:
                b_mask = np.ascontiguousarray(vector_mask.ravel(order='C').astype(np.float32))
            else:
                b_mask = np.zeros(total_windows, dtype=np.float32)

            # Fused triple cross-correlation: AB + AA + BB in one C call
            # Reuses FFT(A) and FFT(B) across all three products,
            # halving the number of forward FFTs vs 3× separate calls.
            with self._profile_section(pass_idx, "xcorr"):
                error_code = self._run_correlation_accumulate_triple(
                    images_a_prime, images_b_prime,
                    self.win_weights_A[pass_idx], self.win_weights_B[pass_idx],  # AB weights
                    self.win_weights_B[pass_idx], self.win_weights_B[pass_idx],  # AA/BB auto weights
                    b_mask, pass_idx,
                    correl_AB_sum, correl_AA_sum, correl_BB_sum,
                )

            # Reshape to (windows, corr_h, corr_w) for downstream processing
            correl_AA_sum = correl_AA_sum.reshape(total_windows, corr_size[0], corr_size[1])
            correl_AB_sum = correl_AB_sum.reshape(total_windows, corr_size[0], corr_size[1])
            correl_BB_sum = correl_BB_sum.reshape(total_windows, corr_size[0], corr_size[1])

            if error_code != 0:
                logging.error(f"Fused triple correlation error code: {error_code}")

        except Exception as e:
            logging.error("Error in correlation: %s", e)
            traceback.print_exc()

        # Copy buffers before returning - required because pre-allocated buffers
        # may be reused by subsequent correlation tasks before Dask finishes
        # serializing this result for network transfer
        with self._profile_section(pass_idx, "result_copy"):
            if copy_result:
                result = {
                    "corr_AA_sum": correl_AA_sum.copy(),
                    "corr_BB_sum": correl_BB_sum.copy(),
                    "corr_AB_sum": correl_AB_sum.copy(),
                    "warp_A_sum": warp_A_sum.copy(),
                    "warp_B_sum": warp_B_sum.copy(),
                    "n_images": N,
                    "n_win_x": n_win_x,
                    "n_win_y": n_win_y,
                    "smoothed_predictor": smoothed_predictor,
                    "padded_predictor": self.delta_ab_old.copy() if pass_idx > 0 else None,
                    "vector_mask": vector_mask,
                    "n_pre": self.n_pre_all[pass_idx],
                    "n_post": self.n_post_all[pass_idx],
                    "first_pair_A": images_a_prime[0].copy() if is_first_batch else None,
                    "first_pair_B": images_b_prime[0].copy() if is_first_batch else None,
                }
            else:
                # Lightweight return — correlation planes stay in self._corr_buffers
                result = {
                    "warp_A_sum": warp_A_sum,
                    "warp_B_sum": warp_B_sum,
                    "n_images": N,
                    "n_win_x": n_win_x,
                    "n_win_y": n_win_y,
                    "smoothed_predictor": smoothed_predictor if is_first_batch else None,
                    "padded_predictor": self.delta_ab_old.copy() if pass_idx > 0 and is_first_batch else None,
                    "vector_mask": vector_mask if is_first_batch else None,
                    "n_pre": self.n_pre_all[pass_idx] if is_first_batch else None,
                    "n_post": self.n_post_all[pass_idx] if is_first_batch else None,
                    "first_pair_A": images_a_prime[0].copy() if is_first_batch else None,
                    "first_pair_B": images_b_prime[0].copy() if is_first_batch else None,
                }
        return result

    def get_accumulated_correlation(self, pass_idx: int) -> dict:
        """Copy accumulated correlation buffers out. Call once after all batches."""
        corr_size = self.window_sizes_for_corr[pass_idx]
        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])
        total_windows = n_win_y * n_win_x
        return {
            "corr_AA_sum": self._corr_buffers[pass_idx]['AA'].reshape(
                total_windows, corr_size[0], corr_size[1]).copy(),
            "corr_BB_sum": self._corr_buffers[pass_idx]['BB'].reshape(
                total_windows, corr_size[0], corr_size[1]).copy(),
            "corr_AB_sum": self._corr_buffers[pass_idx]['AB'].reshape(
                total_windows, corr_size[0], corr_size[1]).copy(),
        }

    def compute_warp_sums_only(
        self,
        images: np.ndarray,
        config: Config,
        pass_idx: int,
        predictor_field: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute warped image sums only (no correlation) for 'image' background method.

        This is the first pass of the two-pass 'image' background subtraction method.
        It loads images, warps them if pass > 0, and accumulates the warped image sums
        which are used to compute mean images (Ā, B̄).

        Parameters
        ----------
        images : np.ndarray
            Image batch of shape (N, 2, H, W)
        config : Config
            Configuration object
        pass_idx : int
            PIV pass index
        predictor_field : Optional[np.ndarray]
            Predictor field from previous pass (shape: n_win_y+2, n_win_x+2, 2)
            containing [uy, ux]. None for pass 0.

        Returns
        -------
        dict
            Dictionary with keys:
            - warp_A_sum: Sum of warped A images (H, W)
            - warp_B_sum: Sum of warped B images (H, W)
            - n_images: Number of images in batch
            - smoothed_predictor: Smoothed predictor field (for pass > 0)
        """
        N, _, H, W = images.shape
        smoothed_predictor = None

        vector_mask = (
            self.vector_masks[pass_idx]
            if self.vector_masks and pass_idx < len(self.vector_masks)
            else None
        )

        # Extract A and B frames
        image_a_stack = images[:, 0, :, :].astype(np.float32, copy=False)
        image_b_stack = images[:, 1, :, :].astype(np.float32, copy=False)

        # Warp images if predictor field is provided (pass > 0)
        if pass_idx > 0 and predictor_field is not None:
            delta_ab_pred = self._get_im_mesh(
                pass_idx, predictor_field
            )
            smoothed_predictor = delta_ab_pred
            logging.debug(
                f"Pass {pass_idx + 1} [warp-only]: Got smoothed predictor field "
                f"(shape: {delta_ab_pred.shape})"
            )

            # Apply vector mask to zero out masked vectors
            if vector_mask is not None:
                smoothed_predictor[vector_mask] = 0

            # Warp images using fused C kernel
            images_a_prime, images_b_prime = self._fused_warp_batch(
                image_a_stack, image_b_stack
            )
        else:
            # No warping for pass 0
            images_a_prime = image_a_stack
            images_b_prime = image_b_stack

        # Compute warp sums
        warp_A_sum = images_a_prime.sum(axis=0)
        warp_B_sum = images_b_prime.sum(axis=0)

        logging.debug(
            f"Pass {pass_idx} [warp-only]: warp_A_sum shape {warp_A_sum.shape}, "
            f"n_images={N}"
        )

        return {
            "warp_A_sum": warp_A_sum.copy(),
            "warp_B_sum": warp_B_sum.copy(),
            "n_images": N,
            "smoothed_predictor": smoothed_predictor,
            "padded_predictor": self.delta_ab_old.copy() if pass_idx > 0 else None,
        }

    def correlate_mean_subtracted_batch(
        self,
        images: np.ndarray,
        config: Config,
        pass_idx: int,
        A_mean: np.ndarray,
        B_mean: np.ndarray,
        predictor_field: Optional[np.ndarray] = None,
        save_diagnostics: bool = False,
        output_path: Optional[str] = None,
        is_first_batch: bool = False,
    ) -> dict:
        """
        Correlate mean-subtracted images for 'image' background subtraction method.

        This is the second pass of the two-pass 'image' background method.
        It loads images, warps them if pass > 0, subtracts the pre-computed mean
        images, then correlates the mean-subtracted images.

        Formula: R_ensemble = <(A - Ā) ⊗ (B - B̄)>

        Parameters
        ----------
        images : np.ndarray
            Image batch of shape (N, 2, H, W)
        config : Config
            Configuration object
        pass_idx : int
            PIV pass index
        A_mean : np.ndarray
            Mean of warped A images (H, W) from first pass
        B_mean : np.ndarray
            Mean of warped B images (H, W) from first pass
        predictor_field : Optional[np.ndarray]
            Predictor field from previous pass
        save_diagnostics : bool
            If True, save warped images for first pair
        output_path : Optional[str]
            Output directory for diagnostic images
        is_first_batch : bool
            If True, this is the first batch

        Returns
        -------
        dict
            Dictionary with keys:
            - corr_AA_sum: Auto-correlation A sum (mean-subtracted)
            - corr_BB_sum: Auto-correlation B sum (mean-subtracted)
            - corr_AB_sum: Cross-correlation sum (mean-subtracted)
            - n_images: Number of images in batch
            - n_win_x: Number of windows in x
            - n_win_y: Number of windows in y
            - smoothed_predictor: Smoothed predictor field
            - vector_mask: Vector mask
            - n_pre, n_post: Padding values
        """
        from pivtools_core.window_utils import apply_single_mode_padding

        win_size = config.ensemble_window_sizes[pass_idx]
        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])
        total_windows = n_win_y * n_win_x

        runtype = config.ensemble_type[pass_idx]
        is_single_mode = (runtype == 'single')

        N, _, H, W = images.shape
        smoothed_predictor = None

        vector_mask = (
            self.vector_masks[pass_idx]
            if self.vector_masks and pass_idx < len(self.vector_masks)
            else None
        )

        # Extract and convert images
        image_a_stack = images[:, 0, :, :].astype(np.float32, copy=False)
        image_b_stack = images[:, 1, :, :].astype(np.float32, copy=False)

        # Warp images if predictor field is provided (pass > 0)
        if pass_idx > 0 and predictor_field is not None:
            delta_ab_pred = self._get_im_mesh(
                pass_idx, predictor_field
            )
            smoothed_predictor = delta_ab_pred

            if vector_mask is not None:
                smoothed_predictor[vector_mask] = 0

            images_a_prime, images_b_prime = self._fused_warp_batch(
                image_a_stack, image_b_stack
            )
        else:
            images_a_prime = image_a_stack
            images_b_prime = image_b_stack

        # SUBTRACT MEAN IMAGES - this is the key difference from standard method
        # A' = A - Ā, B' = B - B̄
        # In-place subtraction to avoid allocating new (N, H, W) arrays
        images_a_prime -= A_mean
        images_b_prime -= B_mean
        images_a_centered = images_a_prime
        images_b_centered = images_b_prime

        logging.debug(
            f"Pass {pass_idx} [mean-subtracted]: A_mean range [{A_mean.min():.2f}, {A_mean.max():.2f}], "
            f"centered A range [{images_a_centered.min():.2f}, {images_a_centered.max():.2f}]"
        )

        # Save diagnostics if requested
        if save_diagnostics and is_first_batch and output_path is not None:
            from pathlib import Path
            from pivtools_cli.preprocessing.diagnostics import save_warped_diagnostics
            save_warped_diagnostics(
                image_a_warped=images_a_prime[0],
                image_b_warped=images_b_prime[0],
                output_dir=Path(output_path),
                pass_idx=pass_idx,
                pair_idx=0,
                image_a_original=image_a_stack[0],
                image_b_original=image_b_stack[0],
            )

        # Apply padding for single mode (for correlation only)
        if is_single_mode:
            sum_window = tuple(config.ensemble_sum_window)
            images_a_centered, _ = apply_single_mode_padding(
                images_a_centered, win_size, sum_window, pad_value=0.0
            )
            images_b_centered, _ = apply_single_mode_padding(
                images_b_centered, win_size, sum_window, pad_value=0.0
            )
            H_padded, W_padded = images_a_centered.shape[-2:]
            image_size = np.ascontiguousarray(np.array([H_padded, W_padded], dtype=np.int32))
        else:
            image_size = np.ascontiguousarray(np.array([H, W], dtype=np.int32))

        # Correlate mean-subtracted images
        corr_size = self.window_sizes_for_corr[pass_idx]
        plane_size = total_windows * corr_size[0] * corr_size[1]

        correl_AB_sum = np.zeros(plane_size, dtype=np.float32)
        correl_AA_sum = np.zeros(plane_size, dtype=np.float32)
        correl_BB_sum = np.zeros(plane_size, dtype=np.float32)

        if vector_mask is not None:
            b_mask = np.ascontiguousarray(vector_mask.ravel(order='C').astype(np.float32))
        else:
            b_mask = np.zeros(total_windows, dtype=np.float32)

        # Fused triple cross-correlation: AB + AA + BB in one C call
        error_code = self._run_correlation_accumulate_triple(
            images_a_centered, images_b_centered,
            self.win_weights_A[pass_idx], self.win_weights_B[pass_idx],  # AB weights
            self.win_weights_B[pass_idx], self.win_weights_B[pass_idx],  # AA/BB auto weights
            b_mask, pass_idx,
            correl_AB_sum, correl_AA_sum, correl_BB_sum,
        )

        # Reshape to (windows, corr_h, corr_w)
        correl_AA_sum = correl_AA_sum.reshape(total_windows, corr_size[0], corr_size[1])
        correl_AB_sum = correl_AB_sum.reshape(total_windows, corr_size[0], corr_size[1])
        correl_BB_sum = correl_BB_sum.reshape(total_windows, corr_size[0], corr_size[1])

        if error_code != 0:
            logging.error(f"Fused triple correlation error code: {error_code}")

        # Return correlation sums only (no warp sums needed - mean already subtracted)
        return {
            "corr_AA_sum": correl_AA_sum.copy(),
            "corr_BB_sum": correl_BB_sum.copy(),
            "corr_AB_sum": correl_AB_sum.copy(),
            "n_images": N,
            "n_win_x": n_win_x,
            "n_win_y": n_win_y,
            "smoothed_predictor": smoothed_predictor,
            "padded_predictor": self.delta_ab_old.copy() if pass_idx > 0 else None,
            "vector_mask": vector_mask,
            "n_pre": self.n_pre_all[pass_idx],
            "n_post": self.n_post_all[pass_idx],
            "first_pair_A": images_a_prime[0].copy() if is_first_batch else None,
            "first_pair_B": images_b_prime[0].copy() if is_first_batch else None,
        }

    def _set_lib_arguments_ensemble(
        self,
        config: Config,
        win_size: list,
        pass_idx: int,
        N: int
    ):
        """
        Set up arguments for the cross-correlation library call.

        For single mode, win_size_arr should be SumWindow (the actual correlation size),
        not the small window size.
        """
        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])
        total_windows = n_win_y * n_win_x

        # Use actual correlation size (SumWindow for single mode)
        corr_size = self.window_sizes_for_corr[pass_idx]
        win_size_arr = np.ascontiguousarray(np.array(corr_size, dtype=np.int32))
        n_windows = np.ascontiguousarray(np.array([n_win_y, n_win_x], dtype=np.int32))

        if self.vector_masks and pass_idx < len(self.vector_masks):
            b_mask = np.ascontiguousarray(
                self.vector_masks[pass_idx].astype(np.float32)
            )
        else:
            b_mask = np.ascontiguousarray(np.zeros((n_win_y, n_win_x), dtype=np.float32))

        n_peaks = config.ensemble_num_peaks
        i_peak_finder = config.ensemble_peak_finder
        b_ensemble = True
        correl_plane_out = np.ascontiguousarray(
            np.zeros(N * total_windows * corr_size[0] * corr_size[1], dtype=np.float32)
        )
        out_shape = (N, n_peaks, n_win_y, n_win_x)

        pk_loc_x = pk_loc_y = pk_height = sx = sy = sxy = np.ascontiguousarray(
                np.zeros((1,), dtype=np.float32)
            )
        point_spread_a = np.zeros_like(correl_plane_out)
        point_spread_b = np.zeros_like(correl_plane_out)

        return (
            win_size_arr,
            n_windows,
            b_mask,
            n_peaks,
            i_peak_finder,
            b_ensemble,
            pk_loc_x,
            pk_loc_y,
            pk_height,
            sx,
            sy,
            sxy,
            correl_plane_out,
            point_spread_a,
            point_spread_b,
        )

    # Interpolation method mapping for cv2.remap (used only for predictor remap now)
    INTERP_FLAGS = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
    }

    def _fused_warp_batch(self, images_a, images_b):
        """Warp N image pairs using fused C kernel with self.delta_ab_old (shared predictor).

        Replaces the old _get_image_prime_batch which used cv2.remap per-image.
        The fused kernel does predictor upsampling + symmetric coordinate maps +
        image interpolation in a single OpenMP-parallel pass.
        """
        images_a = np.ascontiguousarray(images_a, dtype=np.float32)
        images_b = np.ascontiguousarray(images_b, dtype=np.float32)
        N, H, W = images_a.shape
        nPY, nPX = self.delta_ab_old.shape[0], self.delta_ab_old.shape[1]

        out_a = np.zeros((N, H, W), dtype=np.float32)
        out_b = np.zeros((N, H, W), dtype=np.float32)

        pred_dy = np.ascontiguousarray(self.delta_ab_old[..., 0], dtype=np.float32)
        pred_dx = np.ascontiguousarray(self.delta_ab_old[..., 1], dtype=np.float32)

        ret = self._fw_lib.fused_symmetric_warp_batch(
            images_a, images_b,
            out_a, out_b,
            pred_dy, pred_dx,
            N, H, W, nPY, nPX,
            self._fused_ctrs_y, self._fused_ctrs_x,
            self._fused_interp_mode,
            1,  # shared_predictor=1 → ensemble mode
        )
        if ret != 0:
            raise RuntimeError(f"fused_symmetric_warp_batch failed (ret={ret})")

        return out_a, out_b
    
    def _get_im_mesh(self, pass_idx: int, predictor_field: Optional[np.ndarray]):
        """Compute image meshes with predictor field warping.

        Uses interpolation method from config.ensemble_predictor_interpolation.
        """
        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])

        if predictor_field is None or pass_idx == 0:
            predictor_field = np.zeros((n_win_y, n_win_x, 2), dtype=np.float32)

        # Pad predictor field using PREVIOUS pass padding values.
        # The interpolation maps (cached_dense_maps, cached_predictor_maps)
        # were built from win_ctrs_x_all[pass_idx-1] which includes pre/post
        # padding. The input predictor must match this padded coordinate system.
        # This matches instantaneous mode's padding in cpu_instantaneous.py:480-489.
        if pass_idx > 0:
            prev_pass = pass_idx - 1
            pre_y, pre_x = self.n_pre_all[prev_pass]
            post_y, post_x = self.n_post_all[prev_pass]

            # Verify incoming predictor has expected shape (previous pass grid)
            expected_y = len(self.win_ctrs_y[prev_pass])
            expected_x = len(self.win_ctrs_x[prev_pass])
            if predictor_field.shape[0] != expected_y or predictor_field.shape[1] != expected_x:
                logging.warning(
                    f"Pass {pass_idx}: Predictor shape mismatch! "
                    f"Got {predictor_field.shape[:2]}, expected ({expected_y}, {expected_x}). "
                    f"This may cause edge artifacts."
                )

            # DEBUG: Log predictor edge values BEFORE padding
            logging.debug(
                f"Pass {pass_idx}: PRE-PADDING predictor edges - "
                f"top-left=({predictor_field[0,0,0]:.4f}, {predictor_field[0,0,1]:.4f}), "
                f"top-right=({predictor_field[0,-1,0]:.4f}, {predictor_field[0,-1,1]:.4f}), "
                f"bot-left=({predictor_field[-1,0,0]:.4f}, {predictor_field[-1,0,1]:.4f}), "
                f"bot-right=({predictor_field[-1,-1,0]:.4f}, {predictor_field[-1,-1,1]:.4f}), "
                f"center=({predictor_field[predictor_field.shape[0]//2, predictor_field.shape[1]//2, 0]:.4f}, "
                f"{predictor_field[predictor_field.shape[0]//2, predictor_field.shape[1]//2, 1]:.4f})"
            )

            predictor_field = np.pad(
                predictor_field,
                ((pre_y, post_y), (pre_x, post_x), (0, 0)),
                mode="edge",
            )

            # Verify padded shape matches expected interpolation grid
            expected_padded_y = len(self.win_ctrs_y_all[prev_pass])
            expected_padded_x = len(self.win_ctrs_x_all[prev_pass])
            if predictor_field.shape[0] != expected_padded_y or predictor_field.shape[1] != expected_padded_x:
                logging.error(
                    f"Pass {pass_idx}: CRITICAL - Padded predictor shape mismatch! "
                    f"Got {predictor_field.shape[:2]}, expected ({expected_padded_y}, {expected_padded_x}). "
                    f"Padding: pre=({pre_y},{pre_x}), post=({post_y},{post_x})"
                )

            # DEBUG: Log predictor edge values AFTER padding
            logging.debug(
                f"Pass {pass_idx}: POST-PADDING predictor edges - "
                f"top-left=({predictor_field[0,0,0]:.4f}, {predictor_field[0,0,1]:.4f}), "
                f"top-right=({predictor_field[0,-1,0]:.4f}, {predictor_field[0,-1,1]:.4f}), "
                f"bot-left=({predictor_field[-1,0,0]:.4f}, {predictor_field[-1,0,1]:.4f}), "
                f"bot-right=({predictor_field[-1,-1,0]:.4f}, {predictor_field[-1,-1,1]:.4f}), "
                f"center=({predictor_field[predictor_field.shape[0]//2, predictor_field.shape[1]//2, 0]:.4f}, "
                f"{predictor_field[predictor_field.shape[0]//2, predictor_field.shape[1]//2, 1]:.4f})"
            )

            # --- Apply predictor boundary conditions ---
            if self.predictor_bcs:
                padded_ctrs_y = self.win_ctrs_y_all[prev_pass]
                for bc in self.predictor_bcs:
                    bc_y_img = float(self.H - 1 - bc["y_position"]) if bc["edge"] == "bottom" \
                               else float(bc["y_position"])
                    bc_uy = bc["uy"]
                    bc_ux = bc["ux"]

                    if bc["edge"] == "bottom":
                        mask = padded_ctrs_y >= bc_y_img
                    else:  # "top"
                        mask = padded_ctrs_y <= bc_y_img

                    if mask.any():
                        # predictor_field[..., 0] = uy, [.._, 1] = ux
                        predictor_field[mask, :, 0] = bc_uy
                        predictor_field[mask, :, 1] = bc_ux
                        logging.debug(
                            f"Pass {pass_idx}: Applied {bc['edge']} BC at "
                            f"y_img={bc_y_img:.1f}: {mask.sum()} rows set to "
                            f"ux={bc_ux}, uy={bc_uy}"
                        )

        self.delta_ab_old = np.zeros_like(predictor_field).astype(np.float32)
        delta_ab_pred = np.zeros((n_win_y, n_win_x, 2), dtype=np.float32)

        # Smooth predictor field (configurable — disable for ensemble to preserve wall gradients)
        with self._profile_section(pass_idx, "pc_gaussian_smooth"):
            if self.config.ensemble_predictor_smoothing:
                self.delta_ab_old[..., 0] = gaussian_filter(
                    predictor_field[..., 0],
                    sigma=self.sd[pass_idx],
                    truncate=(self.ksize_filt[pass_idx][0] - 1) / (2 * self.sd[pass_idx]),
                    mode="nearest",
                )
                self.delta_ab_old[..., 1] = gaussian_filter(
                    predictor_field[..., 1],
                    sigma=self.sd[pass_idx],
                    truncate=(self.ksize_filt[pass_idx][0] - 1) / (2 * self.sd[pass_idx]),
                    mode="nearest",
                )
            else:
                self.delta_ab_old[..., 0] = predictor_field[..., 0]
                self.delta_ab_old[..., 1] = predictor_field[..., 1]

        # DEBUG: Log smoothed predictor edge values
        if pass_idx > 0:
            logging.debug(
                f"Pass {pass_idx}: POST-SMOOTH delta_ab_old edges - "
                f"shape={self.delta_ab_old.shape}, "
                f"top-left=({self.delta_ab_old[0,0,0]:.4f}, {self.delta_ab_old[0,0,1]:.4f}), "
                f"top-right=({self.delta_ab_old[0,-1,0]:.4f}, {self.delta_ab_old[0,-1,1]:.4f}), "
                f"bot-left=({self.delta_ab_old[-1,0,0]:.4f}, {self.delta_ab_old[-1,0,1]:.4f}), "
                f"bot-right=({self.delta_ab_old[-1,-1,0]:.4f}, {self.delta_ab_old[-1,-1,1]:.4f}), "
                f"center=({self.delta_ab_old[self.delta_ab_old.shape[0]//2, self.delta_ab_old.shape[1]//2, 0]:.4f}, "
                f"{self.delta_ab_old[self.delta_ab_old.shape[0]//2, self.delta_ab_old.shape[1]//2, 1]:.4f}), "
                f"sigma={self.sd[pass_idx]:.4f}, ksize={self.ksize_filt[pass_idx]}"
            )

        # Get interpolation method from config (default to cubic for backwards compatibility)
        interp_method = getattr(self.config, 'ensemble_predictor_interpolation', 'cubic')
        interp_flag = self.INTERP_FLAGS.get(interp_method, cv2.INTER_CUBIC)

        with self._profile_section(pass_idx, "pc_predictor_remap"):
            map_x, map_y = self.cached_predictor_maps[pass_idx]
            if map_x is None or map_y is None:
                raise ValueError(f"Predictor interpolation maps missing for pass {pass_idx}")

            for d in range(2):
                delta_ab_pred[..., d] = cv2.remap(
                    self.delta_ab_old[..., d],
                    map_x,
                    map_y,
                    interp_flag,
                    borderMode=cv2.BORDER_REPLICATE,
                )

        # DEBUG: Log predictor remap edge values
        if pass_idx > 0:
            logging.debug(
                f"Pass {pass_idx}: POST-REMAP delta_ab_pred - "
                f"shape={delta_ab_pred.shape}, "
                f"top-left=({delta_ab_pred[0,0,0]:.4f}, {delta_ab_pred[0,0,1]:.4f}), "
                f"top-right=({delta_ab_pred[0,-1,0]:.4f}, {delta_ab_pred[0,-1,1]:.4f}), "
                f"bot-left=({delta_ab_pred[-1,0,0]:.4f}, {delta_ab_pred[-1,0,1]:.4f}), "
                f"bot-right=({delta_ab_pred[-1,-1,0]:.4f}, {delta_ab_pred[-1,-1,1]:.4f}), "
                f"center=({delta_ab_pred[delta_ab_pred.shape[0]//2, delta_ab_pred.shape[1]//2, 0]:.4f}, "
                f"{delta_ab_pred[delta_ab_pred.shape[0]//2, delta_ab_pred.shape[1]//2, 1]:.4f})"
            )

        # Store window centres and interp mode for _fused_warp_batch
        if pass_idx > 0:
            prev_pass = pass_idx - 1
            self._fused_ctrs_y = np.ascontiguousarray(
                self.win_ctrs_y_all[prev_pass], dtype=np.float32)
            self._fused_ctrs_x = np.ascontiguousarray(
                self.win_ctrs_x_all[prev_pass], dtype=np.float32)
        else:
            self._fused_ctrs_y = np.ascontiguousarray(
                self.win_ctrs_y_all[0], dtype=np.float32)
            self._fused_ctrs_x = np.ascontiguousarray(
                self.win_ctrs_x_all[0], dtype=np.float32)

        image_interp = getattr(self.config, 'ensemble_image_warp_interpolation', 'cubic')
        self._fused_interp_mode = 0 if image_interp == 'cubic' else 1

        return delta_ab_pred
def plot_corr_planes(corr_avg_flat, n_win_y, n_win_x, win_h, win_w, pass_idx):
    """
    Visualize ensemble-averaged correlation planes for PIV in a grid
    matching the window structure.

    Parameters
    ----------
    corr_avg_flat : np.ndarray
        Flattened correlation planes,
        shape (n_win_y * n_win_x * win_h * win_w,)
    n_win_y : int
        Number of interrogation windows in y direction
    n_win_x : int
        Number of interrogation windows in x direction
    win_h : int
        Height of each correlation plane (pixels)
    win_w : int
        Width of each correlation plane (pixels)
    pass_idx : int
        Pass index for naming the output file
    """
    # Reshape into (n_win_y, n_win_x, win_h, win_w) using C order
    corr_planes = corr_avg_flat.reshape((n_win_y, n_win_x, win_h, win_w), order="C")

    fig, axes = plt.subplots(n_win_y, n_win_x, figsize=(3 * n_win_x, 3 * n_win_y))

    for i in range(n_win_y):
        for j in range(n_win_x):
            ax = axes[i, j]
            plane = corr_planes[i, j]
            im = ax.imshow(plane, origin="lower", cmap="viridis")
            ax.set_title(f"W{i},{j}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"corr{pass_idx}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: corr{pass_idx}.png")
    plt.close()
