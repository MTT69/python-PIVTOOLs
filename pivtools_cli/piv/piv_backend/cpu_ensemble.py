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
        cls._lib_marq.fit_stacked_gaussian_batch.argtypes = [
            ctypes.c_size_t,                  # num_windows
            ctypes.c_size_t,                  # n (points per plane)
            ctypes.POINTER(ctypes.c_double),  # X1 (grid coords)
            ctypes.POINTER(ctypes.c_double),  # X2 (grid coords)
            ctypes.POINTER(ctypes.c_double),  # y_batch (correlation data)
            ctypes.POINTER(ctypes.c_double),  # initial_guess_batch
            ctypes.POINTER(ctypes.c_double),  # out_params_batch
            ctypes.POINTER(ctypes.c_int),     # out_status_batch
        ]
        cls._lib_marq.fit_stacked_gaussian_batch.restype = ctypes.c_int

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
        (img_size, wx, wy, n_win, w_size, n_peaks, i_peak, pk_x, pk_y,
         pk_h, sx, sy, sxy, out_plane) = common_args

        # Clear output plane before use
        out_plane.fill(0)

        err = self.lib.bulkxcorr2d(
            np.ascontiguousarray(img1),
            np.ascontiguousarray(img2),
            b_mask,
            img_size,
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

    def _cache_window_padding_ensemble(self, config: Config) -> None:
        """Cache window padding information for ensemble PIV.

        :param config: Configuration object.
        :type config: Config
        """
        self.win_ctrs_x: list[np.ndarray] = []
        self.win_ctrs_y: list[np.ndarray] = []
        self.win_spacing_x: list[int] = []
        self.win_spacing_y: list[int] = []
        self.win_ctrs_x_all: list[np.ndarray] = []
        self.win_ctrs_y_all: list[np.ndarray] = []
        self.n_pre_all: list[tuple[int, int]] = []
        self.n_post_all: list[tuple[int, int]] = []
        self.ksize_filt: list[tuple[int, int]] = []
        self.sd: list[float] = []
        self.G_smooth_predictor: list[np.ndarray] = []

        H, W = config.image_shape

        for pass_idx, _ in enumerate(config.ensemble_window_sizes):
            spacing_x, spacing_y, win_ctrs_x, win_ctrs_y = self._compute_window_centres_ensemble(
                pass_idx, config
            )

            win_ctrs_x_pre = np.arange(1, win_ctrs_x[0] - spacing_x / 2, spacing_x)
            if win_ctrs_x_pre.size == 0:
                win_ctrs_x_pre = np.array([1])
            win_ctrs_x_pre -= 1
            win_ctrs_x_post = np.arange(W, win_ctrs_x[-1] + spacing_x / 2, -spacing_x)
            if win_ctrs_x_post.size == 0:
                win_ctrs_x_post = np.array([W])
            win_ctrs_x_post -= 1
            win_ctrs_x_all = np.concatenate(
                [win_ctrs_x_pre, win_ctrs_x, win_ctrs_x_post[::-1]]
            )

            win_ctrs_y_pre = np.arange(1, win_ctrs_y[0] - spacing_y / 2, spacing_y)
            if win_ctrs_y_pre.size == 0:
                win_ctrs_y_pre = np.array([1])
            win_ctrs_y_pre -= 1
            win_ctrs_y_post = np.arange(H, win_ctrs_y[-1] + spacing_y / 2, -spacing_y)
            if win_ctrs_y_post.size == 0:
                win_ctrs_y_post = np.array([H])
            win_ctrs_y_post -= 1
            win_ctrs_y_all = np.concatenate(
                [win_ctrs_y_pre, win_ctrs_y, win_ctrs_y_post[::-1]]
            )

            n_pre = (len(win_ctrs_y_pre), len(win_ctrs_x_pre))
            n_post = (len(win_ctrs_y_post), len(win_ctrs_x_post))

            self.win_ctrs_x.append(win_ctrs_x.astype(np.float32))
            self.win_ctrs_y.append(win_ctrs_y.astype(np.float32))
            self.win_spacing_x.append(spacing_x)
            self.win_spacing_y.append(spacing_y)
            self.win_ctrs_x_all.append(win_ctrs_x_all.astype(np.float32))
            self.win_ctrs_y_all.append(win_ctrs_y_all.astype(np.float32))
            self.n_pre_all.append(n_pre)
            self.n_post_all.append(n_post)

            if pass_idx == 0:
                self.ksize_filt.append((1, 1))
                self.sd.append(np.sqrt(np.prod((1, 1))) / 3 * 0.65)
                self.G_smooth_predictor.append(np.ones((1, 1), dtype=np.float32))
            else:
                prev_counts = (
                    len(self.win_ctrs_y[pass_idx - 1]),
                    len(self.win_ctrs_x[pass_idx - 1]),
                )
                prev_spacing = (
                    self.win_spacing_y[pass_idx - 1],
                    self.win_spacing_x[pass_idx - 1],
                )
                k_filt = (
                    np.round(np.array(prev_counts) / np.array(prev_spacing)).astype(int)
                    + 1
                )
                k_filt_list = [int(k) for k in k_filt.tolist()]
                k_filt_tuple = (
                    k_filt_list[0] + (k_filt_list[0] % 2 == 0),
                    k_filt_list[1] + (k_filt_list[1] % 2 == 0),
                )
                self.ksize_filt.append(k_filt_tuple)
                self.sd.append(np.sqrt(np.prod(k_filt_tuple)) / 3 * 0.65)
                g_kernel = self._window_weight_fun(k_filt_tuple, config.ensemble_window_type)
                g_kernel = g_kernel.astype(np.float32)
                g_kernel /= max(np.sum(g_kernel), 1e-12)
                self.G_smooth_predictor.append(g_kernel)

    def _compute_window_centres_ensemble(
        self, pass_idx: int, config: Config
    ) -> tuple[int, int, np.ndarray, np.ndarray]:
        """
        Compute window centers and spacing for ensemble PIV pass.

        Uses centralized window_utils for consistency with instantaneous mode.
        Supports both standard and single mode ensemble PIV.
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
        else:
            # Standard mode
            result = compute_window_centers(
                image_shape=config.image_shape,
                window_size=(win_y, win_x),
                overlap=overlap,
                validate=True
            )

        return (
            result.win_spacing_x,
            result.win_spacing_y,
            np.ascontiguousarray(result.win_ctrs_x),
            np.ascontiguousarray(result.win_ctrs_y),
        )

    def _cache_interpolation_grids_ensemble(self, config: Config) -> None:
        """Cache interpolation grids for predictor correction in ensemble PIV.

        For pass_idx > 0, interpolation maps are based on the PREVIOUS pass's
        padded window centers, since the predictor field comes from the previous pass.
        """
        H, W = config.image_shape

        y_coords = np.arange(H, dtype=np.float32)
        x_coords = np.arange(W, dtype=np.float32)
        y_mesh, x_mesh = np.meshgrid(y_coords, x_coords, indexing="ij")
        self.im_mesh = np.stack([y_mesh, x_mesh], axis=-1)

        self.cached_dense_maps = []
        self.cached_predictor_maps = []

        for pass_idx in range(len(config.ensemble_window_sizes)):
            if pass_idx == 0:
                # First pass: use current pass coordinates
                points_y = self.win_ctrs_y_all[pass_idx]
                points_x = self.win_ctrs_x_all[pass_idx]
            else:
                # Subsequent passes: use PREVIOUS pass coordinates
                # because predictor field comes from previous pass
                points_y = self.win_ctrs_y_all[pass_idx - 1]
                points_x = self.win_ctrs_x_all[pass_idx - 1]

            # Dense interpolation maps (for image warping)
            map_x_1d = np.interp(x_coords, points_x, np.arange(len(points_x)))
            map_y_1d = np.interp(y_coords, points_y, np.arange(len(points_y)))
            map_y_2d, map_x_2d = np.meshgrid(
                map_y_1d.astype(np.float32), map_x_1d.astype(np.float32),
                indexing="ij"
            )
            self.cached_dense_maps.append((map_x_2d, map_y_2d))

            # Predictor interpolation maps (from prev pass grid to current pass grid)
            win_ctrs_x = self.win_ctrs_x[pass_idx]
            win_ctrs_y = self.win_ctrs_y[pass_idx]

            win_y, win_x = np.meshgrid(win_ctrs_y, win_ctrs_x, indexing="ij")
            ix = np.interp(win_x.ravel(), points_x, np.arange(len(points_x)))
            iy = np.interp(win_y.ravel(), points_y, np.arange(len(points_y)))
            map_x = ix.reshape(win_x.shape).astype(np.float32)
            map_y = iy.reshape(win_y.shape).astype(np.float32)

            self.cached_predictor_maps.append((map_x, map_y))

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
        corr_size = self.window_sizes_for_corr[pass_idx]
        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])

        # Check if single mode
        runtype = config.ensemble_type[pass_idx]
        is_single_mode = (runtype == 'single')

        total_windows = n_win_y * n_win_x
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
        warp_A_sum = np.zeros((H, W), dtype=np.float32)
        warp_B_sum = np.zeros((H, W), dtype=np.float32)

        # Store smoothed predictor (will be set during warping if pass > 0)
        smoothed_predictor = None

        vector_mask = (
            self.vector_masks[pass_idx]
            if self.vector_masks and pass_idx < len(self.vector_masks)
            else None
        )

        # Process each image pair
        for n in range(N):
            try:
                image_a = np.asarray(images[n, 0], dtype=np.float32)
                image_b = np.asarray(images[n, 1], dtype=np.float32)

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
                            pass_idx, predictor_field, interp="cubic"
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

                    # Warp images using cv2.remap
                    import cv2
                    image_a_prime = cv2.remap(
                        image_a,
                        im_mesh_A[..., 1].astype(np.float32),  # x coordinates
                        im_mesh_A[..., 0].astype(np.float32),  # y coordinates
                        cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    image_b_prime = cv2.remap(
                        image_b,
                        im_mesh_B[..., 1].astype(np.float32),
                        im_mesh_B[..., 0].astype(np.float32),
                        cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )

                    # Clip negative values from cubic interpolation ringing
                    # PIV images should have non-negative intensity values
                    image_a_prime = np.clip(image_a_prime, 0, None)
                    image_b_prime = np.clip(image_b_prime, 0, None)
                else:
                    # No warping for pass 0
                    image_a_prime = image_a
                    image_b_prime = image_b

                # Accumulate RAW warped images (for computing <A>, <B> later)
                warp_A_sum += image_a_prime
                warp_B_sum += image_b_prime

                # Save warped images for diagnostic purposes (first pair of first batch)
                if save_diagnostics and is_first_batch and n == 0 and output_path is not None:
                    from pathlib import Path
                    from pivtools_cli.preprocessing.diagnostics import save_warped_diagnostics
                    save_warped_diagnostics(
                        image_a_warped=image_a_prime,
                        image_b_warped=image_b_prime,
                        output_dir=Path(output_path),
                        pass_idx=pass_idx,
                        pair_idx=0,
                        image_a_original=image_a,  # Original pre-warp image
                        image_b_original=image_b,  # Original pre-warp image
                    )

                # Apply padding for single mode
                if is_single_mode:
                    sum_window = tuple(config.ensemble_sum_window)
                    image_a_prime, padding = apply_single_mode_padding(
                        image_a_prime, win_size, sum_window, pad_value=0.0
                    )
                    image_b_prime, _ = apply_single_mode_padding(
                        image_b_prime, win_size, sum_window, pad_value=0.0
                    )
                    H_padded, W_padded = image_a_prime.shape
                    image_size = np.ascontiguousarray(np.array([H_padded, W_padded], dtype=np.int32))
                else:
                    image_size = np.ascontiguousarray(np.array([H, W], dtype=np.int32))

            except Exception as e:
                logging.error("Error preprocessing image: %s", e)
                traceback.print_exc()
                continue

            try:
                # Set up library arguments
                (
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
                    correl_out_AB,
                    correl_out_AA,
                    correl_out_BB,
                ) = self._set_lib_arguments_ensemble(
                    config=config,
                    win_size=config.ensemble_window_sizes[pass_idx],
                    pass_idx=pass_idx,
                )

                # Pack common arguments
                common_args = (
                    image_size,
                    self.win_ctrs_x[pass_idx].astype(np.float32),
                    self.win_ctrs_y[pass_idx].astype(np.float32),
                    n_windows,
                    win_size_arr,
                    int(n_peaks),
                    int(i_peak_finder),
                    pk_loc_x,
                    pk_loc_y,
                    pk_height,
                    sx,
                    sy,
                    sxy,
                )

                # Cross-correlation AB
                error_code_AB, _ = self._run_correlation_kernel(
                    image_a_prime, image_b_prime,
                    self.win_weights_A[pass_idx], self.win_weights_B[pass_idx],
                    b_mask, common_args + (correl_out_AB,)
                )

                # Auto-correlation AA
                error_code_AA, _ = self._run_correlation_kernel(
                    image_a_prime, image_a_prime,
                    self.win_weights_A[pass_idx], self.win_weights_A[pass_idx],
                    b_mask, common_args + (correl_out_AA,)
                )

                # Auto-correlation BB
                error_code_BB, _ = self._run_correlation_kernel(
                    image_b_prime, image_b_prime,
                    self.win_weights_B[pass_idx], self.win_weights_B[pass_idx],
                    b_mask, common_args + (correl_out_BB,)
                )

                correl_AA_sum += correl_out_AA
                correl_BB_sum += correl_out_BB
                correl_AB_sum += correl_out_AB

                if error_code_AB != 0 or error_code_AA != 0 or error_code_BB != 0:
                    logging.error("Correlation error codes: AB={}, AA={}, BB={}".format(
                        error_code_AB, error_code_AA, error_code_BB))

            except Exception as e:
                logging.error("Error in correlation: %s", e)
                traceback.print_exc()
                continue

        # Copy buffers before returning (since they will be reused for next batch)
        return {
            "corr_AA_sum": correl_AA_sum.copy(),
            "corr_BB_sum": correl_BB_sum.copy(),
            "corr_AB_sum": correl_AB_sum.copy(),
            "warp_A_sum": warp_A_sum,
            "warp_B_sum": warp_B_sum,
            "n_images": N,
            "n_win_x": n_win_x,
            "n_win_y": n_win_y,
            "smoothed_predictor": smoothed_predictor,  # For pass > 0
            "vector_mask": vector_mask,
        }

    def _set_lib_arguments_ensemble(
        self,
        config: Config,
        win_size: list,
        pass_idx: int,
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

        # Output arrays
        pk_loc_x = np.ascontiguousarray(np.zeros((n_peaks, n_win_y, n_win_x), dtype=np.float32))
        pk_loc_y = np.ascontiguousarray(np.zeros((n_peaks, n_win_y, n_win_x), dtype=np.float32))
        pk_height = np.ascontiguousarray(np.zeros((n_peaks, n_win_y, n_win_x), dtype=np.float32))
        sx = np.ascontiguousarray(np.zeros((n_peaks, n_win_y, n_win_x), dtype=np.float32))
        sy = np.ascontiguousarray(np.zeros((n_peaks, n_win_y, n_win_x), dtype=np.float32))
        sxy = np.ascontiguousarray(np.zeros((n_peaks, n_win_y, n_win_x), dtype=np.float32))
        # Use correlation size (SumWindow for single mode) for output arrays
        correl_plane_out = np.ascontiguousarray(
            np.zeros(total_windows * corr_size[0] * corr_size[1], dtype=np.float32)
        )
        point_spread_a = np.ascontiguousarray(
            np.zeros(total_windows * corr_size[0] * corr_size[1], dtype=np.float32)
        )
        point_spread_b = np.ascontiguousarray(
            np.zeros(total_windows * corr_size[0] * corr_size[1], dtype=np.float32)
        )

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

    def _get_im_mesh(self, pass_idx: int, predictor_field: Optional[np.ndarray], interp: str = "cubic"):
        """Compute image meshes with predictor field warping."""
        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])

        if predictor_field is None or pass_idx == 0:
            predictor_field = np.zeros((n_win_y, n_win_x, 2), dtype=np.float32)

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

        interp_flag = cv2.INTER_CUBIC if interp == "cubic" else cv2.INTER_LINEAR
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

        delta_0b = self.delta_ab_dense / 2
        delta_0a = -self.delta_ab_dense / 2
        im_mesh_A = self.im_mesh + delta_0a
        im_mesh_B = self.im_mesh + delta_0b

        return im_mesh_A, im_mesh_B, delta_ab_pred
