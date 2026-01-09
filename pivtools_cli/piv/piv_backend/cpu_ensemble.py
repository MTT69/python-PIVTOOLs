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
import traceback
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

    def __init__(
        self,
        config: Config,
        precomputed_cache: Optional[dict] = None,
        vector_masks: Optional[List[np.ndarray]] = None,
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
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nWindowSize
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fCorrelPlane_Sum (output)
        ]

        # Initialize window weights for each pass
        # For single mode, Frame A and Frame B use different weights
        self.win_weights_A = []
        self.win_weights_B = []
        self.window_sizes_for_corr = []  # Actual correlation size

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
                corr_size = sum_window  # Correlation uses SumWindow size
            else:
                # Standard mode: both frames use same window
                weight = np.ascontiguousarray(
                    self._window_weight_fun(win_size, config.ensemble_window_type)
                )
                weight_A = weight
                weight_B = weight
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

        # Pre-allocate correlation plane buffers (reused across batches)
        self._corr_buffers = {}
        for pass_idx in range(config.ensemble_num_passes):
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

    @classmethod
    def _load_libraries(cls):
        """Load C libraries once per process to avoid DLL thrashing."""
        logging.info("Loading C libraries (One-time init)...")

        lib_extension = ".dll" if os.name == "nt" else ".so"

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
        corr_size = self.window_sizes_for_corr[pass_idx]
        win_size_arr = np.array([corr_size[0], corr_size[1]], dtype=np.int32)

        error_code = self.lib.bulkxcorr2d_accumulate(
            np.ascontiguousarray(images_a, dtype=np.float32),
            np.ascontiguousarray(images_b, dtype=np.float32),
            mask,
            image_size,
            N,
            self.win_ctrs_x[pass_idx].astype(np.float32),
            self.win_ctrs_y[pass_idx].astype(np.float32),
            n_windows,
            weight_a,
            weight_b,
            win_size_arr,
            correl_sum,
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
        else:
            # Standard mode
            result = compute_window_centers(
                image_shape=config.image_shape,
                window_size=(win_y, win_x),
                overlap=overlap,
                validate=True
            )
            padding = (0, 0, 0, 0)  # No padding for standard mode

        return (
            result.win_spacing_x,
            result.win_spacing_y,
            np.ascontiguousarray(result.win_ctrs_x),
            np.ascontiguousarray(result.win_ctrs_y),
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
        correl_AA_sum.fill(0)
        correl_BB_sum.fill(0)
        correl_AB_sum.fill(0)

        # Accumulators for warped images
        #warp_A_sum = np.zeros((H, W), dtype=np.float32)
        #warp_B_sum = np.zeros((H, W), dtype=np.float32)

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

                # For single-pass optimization: accumulate RAW warped images
                # Mean subtraction happens in finalize() via background correlation
                # Formula: R_ensemble = <A⋆B> - <A>⋆<B>

                # Warp images if predictor field is provided (pass > 0)
            if pass_idx > 0:
                if predictor_field is None:
                    logging.warning(
                            f"Pass {pass_idx + 1}: predictor_field is None! "
                            f"Cannot perform image warping. Correlating unwarped images."
                        )
                else:
                    im_mesh_A, im_mesh_B, delta_ab_pred = self._get_im_mesh(
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

            if predictor_field is not None and pass_idx > 0:
                images_a_prime, images_b_prime = self._get_image_prime_batch(
                        image_a_stack, image_b_stack, im_mesh_A, im_mesh_B
            )
            else:
                    # No warping for pass 0
                images_a_prime = image_a_stack
                images_b_prime = image_b_stack


            # Compute warp sums BEFORE padding (for accumulation in original image space)
            # This ensures warp sums have shape (H, W) not (H_padded, W_padded)
            warp_A_sum = images_a_prime.sum(axis=0)
            warp_B_sum = images_b_prime.sum(axis=0)
            logging.debug(f"Pass {pass_idx}: warp_A_sum shape {warp_A_sum.shape} (expected {H}x{W})")

                # Apply padding for single mode (for correlation only)
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

            # Use pre-allocated buffers (already cleared above at lines 567-570)

            # Create mask for C library
            if vector_mask is not None:
                b_mask = np.ascontiguousarray(vector_mask.ravel(order='C').astype(np.float32))
            else:
                b_mask = np.zeros(total_windows, dtype=np.float32)

            # Cross-correlation AB (accumulates internally - no reshape/sum needed!)
            error_code_AB = self._run_correlation_accumulate(
                images_a_prime, images_b_prime,
                self.win_weights_A[pass_idx], self.win_weights_B[pass_idx],
                b_mask, pass_idx, correl_AB_sum
            )

            # Auto-correlation AA
            error_code_AA = self._run_correlation_accumulate(
                images_a_prime, images_a_prime,
                self.win_weights_A[pass_idx], self.win_weights_A[pass_idx],
                b_mask, pass_idx, correl_AA_sum
            )

            # Auto-correlation BB
            error_code_BB = self._run_correlation_accumulate(
                images_b_prime, images_b_prime,
                self.win_weights_B[pass_idx], self.win_weights_B[pass_idx],
                b_mask, pass_idx, correl_BB_sum
            )

            # Reshape to (windows, corr_h, corr_w) for downstream processing
            correl_AA_sum = correl_AA_sum.reshape(total_windows, corr_size[0], corr_size[1])
            correl_AB_sum = correl_AB_sum.reshape(total_windows, corr_size[0], corr_size[1])
            correl_BB_sum = correl_BB_sum.reshape(total_windows, corr_size[0], corr_size[1])

            if error_code_AB != 0 or error_code_AA != 0 or error_code_BB != 0:
                logging.error("Correlation error codes: AB={}, AA={}, BB={}".format(
                    error_code_AB, error_code_AA, error_code_BB))

        except Exception as e:
            logging.error("Error in correlation: %s", e)
            traceback.print_exc()

        # Copy buffers before returning - required because pre-allocated buffers
        # may be reused by subsequent correlation tasks before Dask finishes
        # serializing this result for network transfer
        return {
            "corr_AA_sum": correl_AA_sum.copy(),
            "corr_BB_sum": correl_BB_sum.copy(),
            "corr_AB_sum": correl_AB_sum.copy(),
            "warp_A_sum": warp_A_sum.copy(),
            "warp_B_sum": warp_B_sum.copy(),
            "n_images": N,
            "n_win_x": n_win_x,
            "n_win_y": n_win_y,
            "smoothed_predictor": smoothed_predictor,  # For pass > 0
            "vector_mask": vector_mask,
            # Padding values for predictor field - used in finalize_pass to store
            # PADDED predictor matching instantaneous mode format
            "n_pre": self.n_pre_all[pass_idx],
            "n_post": self.n_post_all[pass_idx],
            # First-pair warped images for diagnostic comparison (only from first batch)
            "first_pair_A": images_a_prime[0].copy() if is_first_batch else None,
            "first_pair_B": images_b_prime[0].copy() if is_first_batch else None,
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
            im_mesh_A, im_mesh_B, delta_ab_pred = self._get_im_mesh(
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

            # Warp images
            images_a_prime, images_b_prime = self._get_image_prime_batch(
                image_a_stack, image_b_stack, im_mesh_A, im_mesh_B
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
            im_mesh_A, im_mesh_B, delta_ab_pred = self._get_im_mesh(
                pass_idx, predictor_field
            )
            smoothed_predictor = delta_ab_pred

            if vector_mask is not None:
                smoothed_predictor[vector_mask] = 0

            images_a_prime, images_b_prime = self._get_image_prime_batch(
                image_a_stack, image_b_stack, im_mesh_A, im_mesh_B
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

        # Cross-correlation AB (mean-subtracted)
        error_code_AB = self._run_correlation_accumulate(
            images_a_centered, images_b_centered,
            self.win_weights_A[pass_idx], self.win_weights_B[pass_idx],
            b_mask, pass_idx, correl_AB_sum
        )

        # Auto-correlation AA (mean-subtracted)
        error_code_AA = self._run_correlation_accumulate(
            images_a_centered, images_a_centered,
            self.win_weights_A[pass_idx], self.win_weights_A[pass_idx],
            b_mask, pass_idx, correl_AA_sum
        )

        # Auto-correlation BB (mean-subtracted)
        error_code_BB = self._run_correlation_accumulate(
            images_b_centered, images_b_centered,
            self.win_weights_B[pass_idx], self.win_weights_B[pass_idx],
            b_mask, pass_idx, correl_BB_sum
        )

        # Reshape to (windows, corr_h, corr_w)
        correl_AA_sum = correl_AA_sum.reshape(total_windows, corr_size[0], corr_size[1])
        correl_AB_sum = correl_AB_sum.reshape(total_windows, corr_size[0], corr_size[1])
        correl_BB_sum = correl_BB_sum.reshape(total_windows, corr_size[0], corr_size[1])

        if error_code_AB != 0 or error_code_AA != 0 or error_code_BB != 0:
            logging.error(
                f"Correlation error codes: AB={error_code_AB}, AA={error_code_AA}, BB={error_code_BB}"
            )

        # Return correlation sums only (no warp sums needed - mean already subtracted)
        return {
            "corr_AA_sum": correl_AA_sum.copy(),
            "corr_BB_sum": correl_BB_sum.copy(),
            "corr_AB_sum": correl_AB_sum.copy(),
            "n_images": N,
            "n_win_x": n_win_x,
            "n_win_y": n_win_y,
            "smoothed_predictor": smoothed_predictor,
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

    # Interpolation method mapping for cv2.remap
    INTERP_FLAGS = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
    }

    def _get_image_prime_batch(
        self,
        images_a: np.ndarray,
        images_b: np.ndarray,
        im_mesh_A: np.ndarray,
        im_mesh_B: np.ndarray,
    ):
        """Warp images using predictor field meshes.

        Uses interpolation method from config.ensemble_image_warp_interpolation.
        """
        images_a = images_a.astype(np.float32, copy=False)
        images_b = images_b.astype(np.float32, copy=False)
        map_A_x = im_mesh_A[..., 1].astype(np.float32, copy=False)
        map_A_y = im_mesh_A[..., 0].astype(np.float32, copy=False)
        map_B_x = im_mesh_B[..., 1].astype(np.float32, copy=False)
        map_B_y = im_mesh_B[..., 0].astype(np.float32, copy=False)

        N = images_a.shape[0]

        out_a = np.empty_like(images_a, dtype=np.float32)
        out_b = np.empty_like(images_b, dtype=np.float32)

        # Get interpolation method from config (default to cubic for backwards compatibility)
        interp_method = getattr(self.config, 'ensemble_image_warp_interpolation', 'cubic')
        interp_flag = self.INTERP_FLAGS.get(interp_method, cv2.INTER_CUBIC)

        for n in range(N):
            out_a[n] = cv2.remap(
                images_a[n],
                map_A_x,
                map_A_y,
                interpolation=interp_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            out_b[n] = cv2.remap(
                images_b[n],
                map_B_x,
                map_B_y,
                interpolation=interp_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

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

        self.delta_ab_old = np.zeros_like(predictor_field).astype(np.float32)
        delta_ab_pred = np.zeros((n_win_y, n_win_x, 2), dtype=np.float32)

        # Smooth predictor field
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

        self.delta_ab_dense = np.zeros((self.H, self.W, 2), dtype=np.float32)
        map_x_2d, map_y_2d = self.cached_dense_maps[pass_idx]

        if map_x_2d is None or map_y_2d is None:
            raise ValueError(f"Dense interpolation maps missing for pass {pass_idx}")

        for d in range(2):
            self.delta_ab_dense[..., d] = cv2.remap(
                self.delta_ab_old[..., d].astype(np.float32),
                map_x_2d,
                map_y_2d,
                interp_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        # DEBUG: Log dense remap edge values and check for zeros at edges
        if pass_idx > 0:
            edge_margin = 10  # pixels from edge
            edge_zeros = (
                (self.delta_ab_dense[:edge_margin, :, :] == 0).sum() +
                (self.delta_ab_dense[-edge_margin:, :, :] == 0).sum() +
                (self.delta_ab_dense[:, :edge_margin, :] == 0).sum() +
                (self.delta_ab_dense[:, -edge_margin:, :] == 0).sum()
            )
            center_zeros = (self.delta_ab_dense[edge_margin:-edge_margin, edge_margin:-edge_margin, :] == 0).sum()
            logging.debug(
                f"Pass {pass_idx}: POST-REMAP delta_ab_dense - "
                f"shape={self.delta_ab_dense.shape}, "
                f"edge_zeros(margin={edge_margin})={edge_zeros}, center_zeros={center_zeros}, "
                f"corners: TL=({self.delta_ab_dense[0,0,0]:.4f},{self.delta_ab_dense[0,0,1]:.4f}), "
                f"TR=({self.delta_ab_dense[0,-1,0]:.4f},{self.delta_ab_dense[0,-1,1]:.4f}), "
                f"BL=({self.delta_ab_dense[-1,0,0]:.4f},{self.delta_ab_dense[-1,0,1]:.4f}), "
                f"BR=({self.delta_ab_dense[-1,-1,0]:.4f},{self.delta_ab_dense[-1,-1,1]:.4f})"
            )

        map_x, map_y = self.cached_predictor_maps[pass_idx]
        if map_x is None or map_y is None:
            raise ValueError(f"Predictor interpolation maps missing for pass {pass_idx}")

        for d in range(2):
            delta_ab_pred[..., d] = cv2.remap(
                self.delta_ab_old[..., d],
                map_x,
                map_y,
                interp_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
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

        delta_0b = self.delta_ab_dense / 2
        delta_0a = -self.delta_ab_dense / 2
        im_mesh_A = self.im_mesh + delta_0a
        im_mesh_B = self.im_mesh + delta_0b

        return im_mesh_A, im_mesh_B, delta_ab_pred
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
