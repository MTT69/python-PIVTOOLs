"""
Ensemble PIV Correlator for PyPIVTools

This module implements ensemble PIV processing where correlation planes from
multiple image pairs are averaged before peak fitting using Levenberg-Marquardt
Gaussian fitting.

Adapted from con_tools ensemble implementation to follow PyPIVTools production
conventions for config, masking, infilling, and save patterns.
"""

import ctypes
import gc
import logging
import os
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import dask.array as da
import numpy as np
from dask.distributed import Client, get_worker, wait
from scipy.ndimage import gaussian_filter
from scipy.io import savemat

from pivtools_core.config import Config
from pivtools_cli.piv.piv_backend.base import CrossCorrelator
from pivtools_cli.piv.piv_result import (
    PIVEnsembleBlockResult,
    PIVEnsemblePassResult,
    PIVEnsembleResult,
)
from pivtools_cli.piv.piv_backend.outlier_detection import apply_outlier_detection
from pivtools_cli.piv.piv_backend.infilling import apply_infilling


class EnsembleCorrelatorCPU(CrossCorrelator):
    """
    Ensemble PIV correlator using CPU with Levenberg-Marquardt Gaussian fitting.

    This correlator averages correlation planes across multiple image pairs before
    fitting 2D stacked Gaussians to extract sub-pixel displacements and uncertainty
    estimates.
    """

    def __init__(
        self,
        config: Config,
        precomputed_cache: Optional[dict] = None,
        vector_masks: Optional[List[np.ndarray]] = None,
    ) -> None:
        super().__init__()

        self.printed_passes = set()

        # Load marquadt library for Gaussian fitting
        lib_extension = ".dll" if os.name == "nt" else ".so"
        marquadt_libpath = os.path.join(
            os.path.dirname(__file__), "..", "..", "lib", f"libmarquadt{lib_extension}"
        )
        marquadt_libpath = os.path.abspath(marquadt_libpath)

        if not os.path.isfile(marquadt_libpath):
            raise FileNotFoundError(
                f"Marquadt library not found: {marquadt_libpath}. "
                "Ensure GSL is installed and run 'pip install -e .' to build."
            )

        self.marquadt_lib = ctypes.CDLL(marquadt_libpath)

        self.marquadt_lib.fit_stacked_gaussian_export.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.marquadt_lib.fit_stacked_gaussian_export.restype = ctypes.c_int

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

        self.lib = ctypes.CDLL(lib_path)
        self.lib.bulkxcorr2d.restype = ctypes.c_ubyte
        self.lib.bulkxcorr2d.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageA
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageB
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fMask
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # nImageSize
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsX
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsY
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # nWindows
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWindowWeightA
            ctypes.c_bool,  # bEnsemble
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWindowWeightB
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # nWindowSize
            ctypes.c_int,  # nPeaks
            ctypes.c_int,  # iPeakFinder
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fPkLocX (output)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fPkLocY (output)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fPkHeight (output)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fSx (output)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fSy (output)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fSxy (output)
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fCorrelPlane_Out (output)
        ]

        # Use ensemble window sizes from config
        self.win_weights = [
            np.ascontiguousarray(self._window_weight_fun(win_size, config.ensemble_window_type))
            for win_size in config.ensemble_window_sizes
        ]

        # Use precomputed cache if provided, otherwise compute it
        if precomputed_cache is not None:
            self._load_precomputed_cache(precomputed_cache)
        else:
            self._cache_window_padding_ensemble(config=config)
            self.H, self.W = config.image_shape
            self._cache_interpolation_grids_ensemble(config=config)

        # Initialize vector masks
        self.vector_masks = vector_masks if vector_masks is not None else []

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
        """Compute window centers and spacing for ensemble PIV pass."""
        win_y, win_x = config.ensemble_window_sizes[pass_idx]
        overlap = config.ensemble_overlaps[pass_idx]

        w_spacing_x = round((1 - overlap / 100) * win_x)
        w_spacing_y = round((1 - overlap / 100) * win_y)

        Nx, Ny = config.image_shape[1], config.image_shape[0]

        min_x = 0
        max_x = Nx - 1
        min_y = 0
        max_y = Ny - 1

        first_ctr_x = -0.5 + win_x / 2
        first_ctr_y = -0.5 + win_y / 2

        start_ctr_x = max(first_ctr_x, min_x)
        start_ctr_y = max(first_ctr_y, min_y)

        n_win_x = int(np.floor((max_x - start_ctr_x) / w_spacing_x)) + 1
        n_win_y = int(np.floor((max_y - start_ctr_y) / w_spacing_y)) + 1

        n_win_x = max(1, n_win_x)
        n_win_y = max(1, n_win_y)

        win_ctrs_x = np.linspace(
            start_ctr_x,
            start_ctr_x + w_spacing_x * (n_win_x - 1),
            n_win_x,
            dtype=np.float32,
        )
        win_ctrs_y = np.linspace(
            start_ctr_y,
            start_ctr_y + w_spacing_y * (n_win_y - 1),
            n_win_y,
            dtype=np.float32,
        )

        return (
            w_spacing_x,
            w_spacing_y,
            np.ascontiguousarray(win_ctrs_x),
            np.ascontiguousarray(win_ctrs_y),
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
        }

    def correlate_batch(
        self,
        images: np.ndarray,
        config: Config,
        pass_idx: int,
        warp_A_mean: np.ndarray,
        warp_B_mean: np.ndarray,
        im_mesh_A: np.ndarray,
        im_mesh_B: np.ndarray,
    ) -> PIVEnsembleBlockResult:
        """Run ensemble PIV correlation on a batch of image pairs."""
        win_size = config.ensemble_window_sizes[pass_idx]
        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])

        total_windows = n_win_y * n_win_x
        correl_plane_mean = np.ascontiguousarray(
            np.zeros(total_windows * win_size[0] * win_size[1], dtype=np.float32)
        )
        point_spread_a_mean = np.zeros_like(correl_plane_mean)
        point_spread_b_mean = np.zeros_like(correl_plane_mean)

        N, _, H, W = images.shape

        for n in range(N):
            try:
                image_a = np.asarray(images[n, 0], dtype=np.float32)
                image_b = np.asarray(images[n, 1], dtype=np.float32)
                image_size = np.ascontiguousarray(np.array([H, W], dtype=np.int32))

                image_a_prime, image_b_prime = self._get_image_prime(
                    image_a, image_b, im_mesh_A, im_mesh_B
                )
                image_a_prime = image_a_prime - warp_A_mean
                image_b_prime = image_b_prime - warp_B_mean

            except Exception as e:
                logging.error("Error in get image_prime: %s", e)
                traceback.print_exc()
                continue

            try:
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
                    correl_plane_out,
                    point_spread_a,
                    point_spread_b,
                ) = self._set_lib_arguments_ensemble(
                    config=config,
                    win_size=config.ensemble_window_sizes[pass_idx],
                    pass_idx=pass_idx,
                )

                # Cross-correlation AB
                error_code_correl = self.lib.bulkxcorr2d(
                    np.ascontiguousarray(image_a_prime),
                    np.ascontiguousarray(image_b_prime),
                    b_mask,
                    image_size,
                    self.win_ctrs_x[pass_idx].astype(np.float32),
                    self.win_ctrs_y[pass_idx].astype(np.float32),
                    n_windows,
                    self.win_weights[pass_idx],
                    b_ensemble,
                    self.win_weights[pass_idx],
                    win_size_arr,
                    int(n_peaks),
                    int(i_peak_finder),
                    pk_loc_x,
                    pk_loc_y,
                    pk_height,
                    sx,
                    sy,
                    sxy,
                    correl_plane_out,
                )

                # Auto-correlation AA
                error_code_auto_A = self.lib.bulkxcorr2d(
                    np.ascontiguousarray(image_a_prime),
                    np.ascontiguousarray(image_a_prime),
                    b_mask,
                    image_size,
                    self.win_ctrs_x[pass_idx].astype(np.float32),
                    self.win_ctrs_y[pass_idx].astype(np.float32),
                    n_windows,
                    self.win_weights[pass_idx],
                    b_ensemble,
                    self.win_weights[pass_idx],
                    win_size_arr,
                    int(n_peaks),
                    int(i_peak_finder),
                    pk_loc_x,
                    pk_loc_y,
                    pk_height,
                    sx,
                    sy,
                    sxy,
                    point_spread_a,
                )

                # Auto-correlation BB
                error_code_auto_b = self.lib.bulkxcorr2d(
                    np.ascontiguousarray(image_b_prime),
                    np.ascontiguousarray(image_b_prime),
                    b_mask,
                    image_size,
                    self.win_ctrs_x[pass_idx].astype(np.float32),
                    self.win_ctrs_y[pass_idx].astype(np.float32),
                    n_windows,
                    self.win_weights[pass_idx],
                    b_ensemble,
                    self.win_weights[pass_idx],
                    win_size_arr,
                    int(n_peaks),
                    int(i_peak_finder),
                    pk_loc_x,
                    pk_loc_y,
                    pk_height,
                    sx,
                    sy,
                    sxy,
                    point_spread_b,
                )

                correl_plane_mean += correl_plane_out
                point_spread_a_mean += point_spread_a
                point_spread_b_mean += point_spread_b

                if error_code_correl != 0 or error_code_auto_A != 0 or error_code_auto_b != 0:
                    logging.error(f"Error in cross-correlation: codes {error_code_correl}, {error_code_auto_A}, {error_code_auto_b}")
                    raise RuntimeError(f"Cross-correlation failed")

            except Exception as e:
                logging.error("Error in cross-correlation step: %s", e)
                traceback.print_exc()
                continue

        return PIVEnsembleBlockResult(
            correlation_plane_mean=correl_plane_mean / N,
            point_spread_a_mean=point_spread_a_mean / N,
            point_spread_b_mean=point_spread_b_mean / N,
            n_win_x=n_win_x,
            n_win_y=n_win_y,
        )

    def _set_lib_arguments_ensemble(
        self,
        config: Config,
        win_size: list,
        pass_idx: int,
    ):
        """Set up arguments for the cross-correlation library call."""
        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])
        total_windows = n_win_y * n_win_x

        win_size_arr = np.ascontiguousarray(np.array(win_size, dtype=np.int32))
        n_windows = np.ascontiguousarray(np.array([n_win_y, n_win_x], dtype=np.int32))

        # Masking
        if self.vector_masks and pass_idx < len(self.vector_masks):
            b_mask = np.ascontiguousarray(
                self.vector_masks[pass_idx].astype(np.float32).ravel()
            )
        else:
            b_mask = np.ascontiguousarray(np.zeros(total_windows, dtype=np.float32))

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
        correl_plane_out = np.ascontiguousarray(
            np.zeros(total_windows * win_size[0] * win_size[1], dtype=np.float32)
        )
        point_spread_a = np.ascontiguousarray(
            np.zeros(total_windows * win_size[0] * win_size[1], dtype=np.float32)
        )
        point_spread_b = np.ascontiguousarray(
            np.zeros(total_windows * win_size[0] * win_size[1], dtype=np.float32)
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

    def _get_image_prime(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        im_mesh_A: np.ndarray,
        im_mesh_B: np.ndarray,
    ):
        """Warp images using predictor field."""
        image_a_prime = cv2.remap(
            image_a.astype(np.float32),
            im_mesh_A[..., 1].astype(np.float32),
            im_mesh_A[..., 0].astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        image_b_prime = cv2.remap(
            image_b.astype(np.float32),
            im_mesh_B[..., 1].astype(np.float32),
            im_mesh_B[..., 0].astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return image_a_prime.astype(np.float32), image_b_prime.astype(np.float32)

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

    def _compute_warps(
        self, images: np.ndarray, predictor_field: Optional[np.ndarray], pass_idx: int,
        im_mesh_A: np.ndarray, im_mesh_B: np.ndarray
    ):
        """Compute mean warp fields across all images."""
        N, _, H, W = images.shape
        block_A_warp = np.zeros([H, W], dtype=np.float32)
        block_B_warp = np.zeros([H, W], dtype=np.float32)

        for n in range(N):
            image_a = np.asarray(images[n, 0], dtype=np.float32)
            image_b = np.asarray(images[n, 1], dtype=np.float32)

            image_a_prime = cv2.remap(
                image_a.astype(np.float32),
                im_mesh_A[..., 1].astype(np.float32),
                im_mesh_A[..., 0].astype(np.float32),
                cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            image_b_prime = cv2.remap(
                image_b.astype(np.float32),
                im_mesh_B[..., 1].astype(np.float32),
                im_mesh_B[..., 0].astype(np.float32),
                cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            block_A_warp = block_A_warp + image_a_prime
            block_B_warp = block_B_warp + image_b_prime

        return block_A_warp / N, block_B_warp / N


class EnsembleExecutor:
    """
    Orchestrator for ensemble PIV processing.

    Manages the multi-pass ensemble PIV pipeline including:
    - Predictor field initialization
    - Mean warp computation
    - Distributed correlation
    - Gaussian fitting
    - Velocity extraction
    """

    def __init__(self, config: Config, client: Client, vector_masks: Optional[List[np.ndarray]] = None):
        logging.info("Initialising Ensemble Executor")
        self.config = config
        self.client = client
        self.vector_masks = vector_masks if vector_masks is not None else []

        # Cache window padding information
        self._cache_window_padding(config)

    def _cache_window_padding(self, config: Config) -> None:
        """Cache window padding information for predictor field initialization."""
        self.n_pre_all: list[tuple[int, int]] = []
        self.n_post_all: list[tuple[int, int]] = []

        H, W = config.image_shape

        for pass_idx in range(len(config.ensemble_window_sizes)):
            spacing_x, spacing_y, win_ctrs_x, win_ctrs_y = self._compute_window_centres(
                pass_idx, config
            )

            win_ctrs_x_pre = np.arange(1, win_ctrs_x[0] - spacing_x / 2, spacing_x)
            if win_ctrs_x_pre.size == 0:
                win_ctrs_x_pre = np.array([1])

            win_ctrs_x_post = np.arange(W, win_ctrs_x[-1] + spacing_x / 2, -spacing_x)
            if win_ctrs_x_post.size == 0:
                win_ctrs_x_post = np.array([W])

            win_ctrs_y_pre = np.arange(1, win_ctrs_y[0] - spacing_y / 2, spacing_y)
            if win_ctrs_y_pre.size == 0:
                win_ctrs_y_pre = np.array([1])

            win_ctrs_y_post = np.arange(H, win_ctrs_y[-1] + spacing_y / 2, -spacing_y)
            if win_ctrs_y_post.size == 0:
                win_ctrs_y_post = np.array([H])

            n_pre = (len(win_ctrs_y_pre), len(win_ctrs_x_pre))
            n_post = (len(win_ctrs_y_post), len(win_ctrs_x_post))

            self.n_pre_all.append(n_pre)
            self.n_post_all.append(n_post)

    def _compute_window_centres(self, pass_idx: int, config: Config):
        """Compute window centers and spacing for a given pass."""
        win_y, win_x = config.ensemble_window_sizes[pass_idx]
        overlap = config.ensemble_overlaps[pass_idx]

        w_spacing_x = round((1 - overlap / 100) * win_x)
        w_spacing_y = round((1 - overlap / 100) * win_y)

        Nx, Ny = config.image_shape[1], config.image_shape[0]

        min_x = 0
        max_x = Nx - 1
        min_y = 0
        max_y = Ny - 1

        first_ctr_x = -0.5 + win_x / 2
        first_ctr_y = -0.5 + win_y / 2

        start_ctr_x = max(first_ctr_x, min_x)
        start_ctr_y = max(first_ctr_y, min_y)

        n_win_x = int(np.floor((max_x - start_ctr_x) / w_spacing_x)) + 1
        n_win_y = int(np.floor((max_y - start_ctr_y) / w_spacing_y)) + 1

        n_win_x = max(1, n_win_x)
        n_win_y = max(1, n_win_y)

        win_ctrs_x = np.linspace(
            start_ctr_x, start_ctr_x + w_spacing_x * (n_win_x - 1), n_win_x, dtype=np.float32
        )
        win_ctrs_y = np.linspace(
            start_ctr_y, start_ctr_y + w_spacing_y * (n_win_y - 1), n_win_y, dtype=np.float32
        )

        return w_spacing_x, w_spacing_y, win_ctrs_x, win_ctrs_y

    def run_ensemble_piv(self, images: da.Array, scattered_cache: dict) -> PIVEnsembleResult:
        """
        Run the complete ensemble PIV pipeline.

        Parameters
        ----------
        images : da.Array
            Dask array of image pairs (N, 2, H, W)
        scattered_cache : dict
            Pre-scattered correlator cache

        Returns
        -------
        PIVEnsembleResult
            Complete ensemble PIV results for all passes
        """
        logging.info("Starting ensemble PIV")
        piv_results = PIVEnsembleResult()

        # Persist images if configured
        if hasattr(self.config, 'cache_images') and self.config.cache_images:
            logging.info("Persisting images on workers for reuse across passes")
            images = images.persist()
            wait(images)

        blocks_delayed = images.to_delayed().ravel()
        predictor_field = None

        for pass_idx, win_size in enumerate(self.config.ensemble_window_sizes):
            logging.info(
                f"Processing ensemble PIV pass {pass_idx + 1}/{len(self.config.ensemble_window_sizes)}"
            )

            # Initialize predictor field from previous pass
            predictor_field = self._initialise_predictor_field(pass_idx, piv_results)

            # Compute mean warps
            warp_A_mean, warp_B_mean = self._get_warp_pass(
                blocks_delayed, pass_idx, predictor_field, scattered_cache
            )

            # Run ensemble correlation
            piv_block_result = self._run_ensemble_piv_pass(
                image_blocks=blocks_delayed,
                predictor_field=predictor_field,
                pass_idx=pass_idx,
                warp_A_mean=warp_A_mean,
                warp_B_mean=warp_B_mean,
                scattered_cache=scattered_cache,
            )

            # Save correlation planes and point spreads if debug enabled
            if hasattr(self.config, 'debug') and self.config.debug:
                try:
                    from pathlib import Path
                    outdir = Path(os.path.join(os.getcwd(), "debug_outputs"))
                    outdir.mkdir(parents=True, exist_ok=True)

                    win_size = self.config.ensemble_window_sizes[pass_idx]
                    n_win_y = piv_block_result.n_win_y
                    n_win_x = piv_block_result.n_win_x

                    # Reshape correlation planes: (y, x, win_h, win_w)
                    corr_reshaped = piv_block_result.correlation_plane_mean.reshape(
                        (n_win_y, n_win_x, win_size[0], win_size[1]), order='C'
                    )
                    psa_reshaped = piv_block_result.point_spread_a_mean.reshape(
                        (n_win_y, n_win_x, win_size[0], win_size[1]), order='C'
                    )
                    psb_reshaped = piv_block_result.point_spread_b_mean.reshape(
                        (n_win_y, n_win_x, win_size[0], win_size[1]), order='C'
                    )

                    savemat(
                        outdir / f"corr_ps_pass_{pass_idx}.mat",
                        {
                            "correlation_plane_mean": corr_reshaped,
                            "point_spread_a_mean": psa_reshaped,
                            "point_spread_b_mean": psb_reshaped,
                        },
                        do_compression=True,
                    )
                    logging.info(f"Saved correlation planes for pass {pass_idx}")
                except Exception as e:
                    logging.warning(
                        f"Failed to save correlation/pointspread mats for pass {pass_idx}: {e}"
                    )

            # Fit Gaussians to correlation planes
            gauss_results, statuses = self._evaluate_correlation_planes(
                piv_block_result, pass_idx, piv_results, scattered_cache
            )

            # Extract velocities from fitted parameters
            # Use the interpolated predictor field from the worker, not the padded one
            piv_pass_result = self._process_correlation_planes(
                piv_result=piv_block_result,
                gauss_results=gauss_results,
                statuses=statuses,
                pass_idx=pass_idx,
                predictor_field=piv_block_result.predictor_field,
            )

            piv_results.add_pass(piv_pass_result)

            # Save PIV pass result if debug enabled
            if hasattr(self.config, 'debug') and self.config.debug:
                try:
                    from pathlib import Path
                    outdir = Path(os.path.join(os.getcwd(), "debug_outputs"))
                    outdir.mkdir(parents=True, exist_ok=True)

                    # Save all PIV result fields
                    mat_dict = {}
                    result_fields = [
                        'ux_mat', 'uy_mat', 'UU_stress', 'VV_stress', 'UV_stress',
                        'peakheights_A', 'peakheights_B', 'peakheights_AB',
                        'nan_reason', 'sig_AB_x', 'sig_AB_y', 'sig_AB_xy',
                        'sig_A_x', 'sig_A_y', 'sig_A_xy',
                        'sig_PD_x', 'sig_PD_y', 'sig_PD_xy',
                        'win_ctrs_x', 'win_ctrs_y'
                    ]
                    for name in result_fields:
                        if hasattr(piv_pass_result, name):
                            val = getattr(piv_pass_result, name)
                            if val is not None:
                                mat_dict[name] = val

                    # Add predictor field if present
                    if piv_block_result.predictor_field is not None:
                        mat_dict["predictor_field_block"] = piv_block_result.predictor_field

                    if mat_dict:
                        savemat(
                            outdir / f"piv_pass_result_{pass_idx}.mat",
                            mat_dict, do_compression=True
                        )
                        logging.info(f"Saved PIV pass result for pass {pass_idx}")
                except Exception as e:
                    logging.warning(
                        f"Failed to save processed pass result mats for pass {pass_idx}: {e}"
                    )

        return piv_results

    def _initialise_predictor_field(self, pass_idx: int, piv_results: PIVEnsembleResult):
        """Initialize predictor field from previous pass results."""
        if pass_idx == 0:
            return None

        if not piv_results.passes:
            raise ValueError("piv_results must have passes for pass_idx > 0")

        prev_pass = piv_results.passes[pass_idx - 1]

        ux = prev_pass.ux_mat.copy()
        uy = prev_pass.uy_mat.copy()

        # Infill any NaN values using mid-pass infilling
        nan_mask = np.isnan(ux) | np.isnan(uy)
        if nan_mask.any():
            mid_infill_cfg = self.config.infilling_mid_pass
            ux, uy = apply_infilling(ux, uy, nan_mask, mid_infill_cfg)

        predictor_field = np.stack([uy, ux], axis=-1)

        # Pad with edge values
        pre_y, pre_x = self.n_pre_all[pass_idx - 1]
        post_y, post_x = self.n_post_all[pass_idx - 1]

        predictor_field = np.pad(
            predictor_field,
            pad_width=((pre_y, post_y), (pre_x, post_x), (0, 0)),
            mode="edge",
        )

        return predictor_field

    def _get_warp_pass(self, image_blocks, pass_idx, predictor_field, scattered_cache):
        """Compute mean warp fields across all images."""
        fut_warp_res = []
        for blk_group in image_blocks:
            fut_warps = self.client.submit(
                _compute_warps_worker,
                [blk_group],
                predictor_field,
                pass_idx,
                scattered_cache,
                self.config,
                pure=False,
            )
            fut_warp_res.append(fut_warps)

        worker_warps = self.client.gather(fut_warp_res)

        total_blocks = sum(n for _, _, n in worker_warps)
        warp_A_mean = sum(mA * n for mA, _, n in worker_warps) / total_blocks
        warp_B_mean = sum(mB * n for _, mB, n in worker_warps) / total_blocks

        return warp_A_mean, warp_B_mean

    def _run_ensemble_piv_pass(
        self,
        image_blocks,
        pass_idx: int,
        warp_A_mean: np.ndarray,
        warp_B_mean: np.ndarray,
        predictor_field: Optional[np.ndarray],
        scattered_cache: dict,
    ) -> PIVEnsembleBlockResult:
        """Run ensemble PIV correlation for one pass."""
        fut_results = []
        for blk in image_blocks:
            fut = self.client.submit(
                _ensemble_piv_pass_worker,
                [blk],
                self.config,
                pass_idx,
                warp_A_mean,
                warp_B_mean,
                predictor_field,
                scattered_cache,
                self.vector_masks,
                pure=False,
            )
            fut_results.append(fut)

        worker_results = self.client.gather(fut_results)
        return _aggregate_ensemble_results(worker_results)

    def _evaluate_correlation_planes(
        self, piv_block_result: PIVEnsembleBlockResult, pass_idx: int,
        piv_results: PIVEnsembleResult, scattered_cache: dict
    ):
        """Fit Gaussians to correlation planes."""
        win_size = self.config.ensemble_window_sizes[pass_idx]
        n_win_y = piv_block_result.n_win_y
        n_win_x = piv_block_result.n_win_x
        total_windows = n_win_y * n_win_x

        # Get predictor displacement guesses
        PD_guess_x, PD_guess_y = _get_pd_guess(
            pass_idx, total_windows, self.config, piv_results, n_win_x, n_win_y
        )

        # Distribute fitting across workers
        num_workers = len(self.client.scheduler_info()["workers"])
        workers = list(self.client.scheduler_info()["workers"].keys())
        windows_per_worker = (total_windows + num_workers - 1) // num_workers

        # Reshape correlation planes
        AA_flat = piv_block_result.point_spread_a_mean
        BB_flat = piv_block_result.point_spread_b_mean
        AB_flat = piv_block_result.correlation_plane_mean

        futures = []
        for i, worker in enumerate(workers):
            start_idx = i * windows_per_worker
            end_idx = min((i + 1) * windows_per_worker, total_windows)
            if start_idx >= end_idx:
                continue

            start_idx_data = start_idx * win_size[0] * win_size[1]
            end_idx_data = end_idx * win_size[0] * win_size[1]

            AA_chunk = self.client.scatter(AA_flat[start_idx_data:end_idx_data], workers=[worker])
            BB_chunk = self.client.scatter(BB_flat[start_idx_data:end_idx_data], workers=[worker])
            AB_chunk = self.client.scatter(AB_flat[start_idx_data:end_idx_data], workers=[worker])
            PD_guess_x_chunk = self.client.scatter(PD_guess_x[start_idx:end_idx], workers=[worker])
            PD_guess_y_chunk = self.client.scatter(PD_guess_y[start_idx:end_idx], workers=[worker])

            fut = self.client.submit(
                _fit_windows_batch,
                AA_chunk, BB_chunk, AB_chunk,
                PD_guess_x_chunk, PD_guess_y_chunk,
                win_size, self.config, pass_idx, scattered_cache,
                workers=[worker],
            )
            futures.append(fut)

        results = self.client.gather(futures)
        gauss_flat = np.concatenate([r[0] for r in results])
        status_flat = np.concatenate([r[1] for r in results])

        gauss_results = gauss_flat.reshape(n_win_y, n_win_x, -1)
        statuses = status_flat.reshape(n_win_y, n_win_x)

        logging.info(f"Gaussian fitting success rate: {np.sum(statuses == 0) / statuses.size:.1%}")
        return gauss_results, statuses

    def _process_correlation_planes(
        self, piv_result: PIVEnsembleBlockResult, gauss_results: np.ndarray,
        statuses: np.ndarray, pass_idx: int, predictor_field: Optional[np.ndarray]
    ) -> PIVEnsemblePassResult:
        """Extract velocities from fitted Gaussian parameters."""
        n_win_y, n_win_x, _ = gauss_results.shape
        win_size = self.config.ensemble_window_sizes[pass_idx]

        # Get window centers
        _, _, win_ctrs_x, win_ctrs_y = self._compute_window_centres(pass_idx, self.config)

        # Extract velocities and stresses from fitted parameters
        ux_mat = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        uy_mat = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        UU_stress = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        VV_stress = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        UV_stress = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        peakheights_A = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        peakheights_B = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        peakheights_AB = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        nan_reason = np.zeros((n_win_y, n_win_x), dtype=np.int32)

        # Sigma arrays
        sig_AB_x = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        sig_AB_y = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        sig_AB_xy = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        sig_A_x = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        sig_A_y = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        sig_A_xy = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        sig_PD_x = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        sig_PD_y = np.zeros((n_win_y, n_win_x), dtype=np.float32)
        sig_PD_xy = np.zeros((n_win_y, n_win_x), dtype=np.float32)

        x_offset = win_size[1] / 2 + 1
        y_offset = win_size[0] / 2 + 1

        for iy in range(n_win_y):
            for ix in range(n_win_x):
                params = gauss_results[iy, ix, :]
                status = statuses[iy, ix]

                if status != 0:
                    nan_reason[iy, ix] = 1
                    ux_mat[iy, ix] = np.nan
                    uy_mat[iy, ix] = np.nan
                    continue

                # Extract parameters
                peakheights_A[iy, ix] = params[0]
                peakheights_B[iy, ix] = params[1]
                peakheights_AB[iy, ix] = params[2]

                sig_A_x[iy, ix] = params[3]
                sig_A_y[iy, ix] = params[4]
                sig_A_xy[iy, ix] = params[5]

                sig_PD_x[iy, ix] = params[6]
                sig_PD_y[iy, ix] = params[7]
                sig_PD_xy[iy, ix] = params[8]

                # Displacement from correlation peak
                ux_mat[iy, ix] = params[11] - x_offset
                uy_mat[iy, ix] = params[12] - y_offset

                # Stress tensors
                UU_stress[iy, ix] = params[6]
                VV_stress[iy, ix] = params[7]
                UV_stress[iy, ix] = params[8]

                # Total variances
                sig_AB_x[iy, ix] = params[3] + params[6]
                sig_AB_y[iy, ix] = params[4] + params[7]
                sig_AB_xy[iy, ix] = params[5] + params[8]

        # Add predictor field back
        # predictor_field is already interpolated to current pass grid size by _get_im_mesh
        if predictor_field is not None and pass_idx > 0:
            # predictor_field format: [:, :, 0] is uy, [:, :, 1] is ux
            ux_mat += predictor_field[:, :, 1]
            uy_mat += predictor_field[:, :, 0]

        # Track NaN mask from fitting failures
        nan_mask = np.isnan(ux_mat) | np.isnan(uy_mat)

        # Apply outlier detection if enabled (matching instantaneous pipeline)
        if self.config.outlier_detection_enabled:
            outlier_methods = self.config.outlier_detection_methods
            if outlier_methods:
                # Get peak magnitude for peak_mag detection
                primary_peak_mag = peakheights_AB
                outlier_mask = apply_outlier_detection(
                    ux_mat, uy_mat, outlier_methods, peak_mag=primary_peak_mag
                )
                nan_mask |= outlier_mask
                logging.debug(f"Pass {pass_idx}: Detected {np.sum(outlier_mask)} outliers")

        # Mark outliers as NaN for infilling
        if nan_mask.any():
            ux_mat[nan_mask] = np.nan
            uy_mat[nan_mask] = np.nan

        # Apply infilling for mid-passes or final pass (matching instantaneous pipeline)
        is_final_pass = (pass_idx == len(self.config.ensemble_window_sizes) - 1)

        # Get infilling configs
        final_infill_cfg = self.config.infilling_final_pass
        mid_infill_cfg = self.config.infilling_mid_pass

        if is_final_pass:
            # Final pass infilling (optional)
            if final_infill_cfg.get('enabled', True) and np.isnan(ux_mat).any():
                infill_mask = np.isnan(ux_mat) | np.isnan(uy_mat)
                ux_mat, uy_mat = apply_infilling(
                    ux_mat, uy_mat, infill_mask, final_infill_cfg
                )
                logging.debug(f"Pass {pass_idx} (final): Infilled {np.sum(infill_mask)} vectors")
        else:
            # Mid-pass infilling (required for predictor)
            if np.isnan(ux_mat).any() or np.isnan(uy_mat).any():
                infill_mask = np.isnan(ux_mat) | np.isnan(uy_mat)
                ux_mat, uy_mat = apply_infilling(
                    ux_mat, uy_mat, infill_mask, mid_infill_cfg
                )
                logging.debug(f"Pass {pass_idx} (mid): Infilled {np.sum(infill_mask)} vectors")

        # Infill stress fields with same method as velocity
        stress_nan_mask = np.isnan(UU_stress) | np.isnan(VV_stress) | np.isnan(UV_stress)
        if stress_nan_mask.any():
            infill_cfg = final_infill_cfg if is_final_pass else mid_infill_cfg
            UU_stress, VV_stress = apply_infilling(
                UU_stress.astype(np.float32), VV_stress.astype(np.float32),
                stress_nan_mask, infill_cfg
            )
            # UV_stress needs separate call since apply_infilling takes pairs
            UV_stress_temp = UV_stress.astype(np.float32)
            UV_stress, _ = apply_infilling(
                UV_stress_temp, UV_stress_temp,
                stress_nan_mask, infill_cfg
            )

        # Apply masking
        if self.vector_masks and pass_idx < len(self.vector_masks):
            mask_pass = self.vector_masks[pass_idx]
            ux_mat[mask_pass] = 0.0
            uy_mat[mask_pass] = 0.0

        return PIVEnsemblePassResult(
            ux_mat=ux_mat,
            uy_mat=uy_mat,
            UU_stress=UU_stress,
            VV_stress=VV_stress,
            UV_stress=UV_stress,
            peakheights_A=peakheights_A,
            peakheights_B=peakheights_B,
            peakheights_AB=peakheights_AB,
            nan_reason=nan_reason,
            sig_AB_x=sig_AB_x,
            sig_AB_y=sig_AB_y,
            sig_AB_xy=sig_AB_xy,
            sig_A_x=sig_A_x,
            sig_A_y=sig_A_y,
            sig_A_xy=sig_A_xy,
            sig_PD_x=sig_PD_x,
            sig_PD_y=sig_PD_y,
            sig_PD_xy=sig_PD_xy,
            window_size=tuple(win_size),
            win_ctrs_x=win_ctrs_x,
            win_ctrs_y=win_ctrs_y,
        )


# Worker functions

def _compute_warps_worker(image_blocks, predictor_field, pass_idx, scattered_cache, config):
    """Worker function to compute mean warps."""
    correlator = EnsembleCorrelatorCPU(config, precomputed_cache=scattered_cache)
    im_mesh_A, im_mesh_B, _ = correlator._get_im_mesh(pass_idx, predictor_field)

    n_blocks = 0
    mean_A_warp = None
    mean_B_warp = None

    for block in image_blocks:
        arr = block.compute()
        block_A_warp, block_B_warp = correlator._compute_warps(
            arr, predictor_field, pass_idx, im_mesh_A, im_mesh_B
        )
        if mean_A_warp is None:
            mean_A_warp = block_A_warp
            mean_B_warp = block_B_warp
        else:
            mean_A_warp += block_A_warp
            mean_B_warp += block_B_warp
        n_blocks += 1

    return mean_A_warp / n_blocks, mean_B_warp / n_blocks, n_blocks


def _ensemble_piv_pass_worker(
    image_blocks, config, pass_idx, warp_A_mean, warp_B_mean,
    predictor_field, scattered_cache, vector_masks
):
    """Worker function to run ensemble PIV correlation."""
    correlator = EnsembleCorrelatorCPU(config, precomputed_cache=scattered_cache, vector_masks=vector_masks)
    im_mesh_A, im_mesh_B, pred_field = correlator._get_im_mesh(pass_idx, predictor_field)

    corr_sum = None
    psa_sum = None
    psb_sum = None
    n_blocks = 0

    for block in image_blocks:
        arr = block.compute()
        result = correlator.correlate_batch(
            arr, config, pass_idx, warp_A_mean, warp_B_mean, im_mesh_A, im_mesh_B
        )

        if corr_sum is None:
            corr_sum = result.correlation_plane_mean
            psa_sum = result.point_spread_a_mean
            psb_sum = result.point_spread_b_mean
        else:
            corr_sum += result.correlation_plane_mean
            psa_sum += result.point_spread_a_mean
            psb_sum += result.point_spread_b_mean

        n_blocks += 1

    return PIVEnsembleBlockResult(
        correlation_plane_mean=corr_sum / n_blocks,
        point_spread_a_mean=psa_sum / n_blocks,
        point_spread_b_mean=psb_sum / n_blocks,
        n_blocks=n_blocks,
        n_win_x=result.n_win_x,
        n_win_y=result.n_win_y,
        predictor_field=pred_field,
    )


def _aggregate_ensemble_results(worker_results: List[PIVEnsembleBlockResult]) -> PIVEnsembleBlockResult:
    """Aggregate results from all workers."""
    total_blocks = sum(r.n_blocks for r in worker_results)
    corr_sum = sum(r.correlation_plane_mean * r.n_blocks for r in worker_results)
    psa_sum = sum(r.point_spread_a_mean * r.n_blocks for r in worker_results)
    psb_sum = sum(r.point_spread_b_mean * r.n_blocks for r in worker_results)

    predictor_field = None
    for r in worker_results:
        if r.predictor_field is not None:
            predictor_field = r.predictor_field
            break

    return PIVEnsembleBlockResult(
        correlation_plane_mean=corr_sum / total_blocks,
        point_spread_a_mean=psa_sum / total_blocks,
        point_spread_b_mean=psb_sum / total_blocks,
        n_win_x=worker_results[0].n_win_x,
        n_win_y=worker_results[0].n_win_y,
        predictor_field=predictor_field,
    )


def _get_pd_guess(pass_idx, n_windows, config, piv_results, n_win_x, n_win_y):
    """Get predictor displacement guesses for Gaussian fitting."""
    if pass_idx == 0:
        return np.full(n_windows, 0.01), np.full(n_windows, 0.01)

    old_pd_x = piv_results.passes[pass_idx - 1].sig_PD_x.copy().astype(np.float32)
    old_pd_y = piv_results.passes[pass_idx - 1].sig_PD_y.copy().astype(np.float32)

    # Infill any NaN values using mid-pass infilling
    nan_mask = np.isnan(old_pd_x) | np.isnan(old_pd_y)
    if nan_mask.any():
        mid_infill_cfg = config.infilling_mid_pass
        old_pd_x, old_pd_y = apply_infilling(
            old_pd_x, old_pd_y, nan_mask, mid_infill_cfg
        )

    old_h, old_w = old_pd_x.shape
    new_h, new_w = n_win_y, n_win_x

    if (old_h, old_w) == (new_h, new_w):
        PD_guess_x = old_pd_x.ravel(order="C")
        PD_guess_y = old_pd_y.ravel(order="C")
    else:
        map_y, map_x = np.meshgrid(
            np.linspace(0, old_h - 1, new_h).astype(np.float32),
            np.linspace(0, old_w - 1, new_w).astype(np.float32),
            indexing="ij"
        )
        PD_guess_x = cv2.remap(old_pd_x, map_x, map_y, cv2.INTER_LINEAR).ravel(order="C")
        PD_guess_y = cv2.remap(old_pd_y, map_x, map_y, cv2.INTER_LINEAR).ravel(order="C")

    return np.maximum(PD_guess_x, 0.01), np.maximum(PD_guess_y, 0.01)


def _fit_windows_batch(AA_chunk, BB_chunk, AB_chunk, PD_guess_x_chunk, PD_guess_y_chunk,
                       win_size, config, pass_idx, scattered_cache):
    """Fit Gaussians to correlation windows on a worker."""
    correlator = EnsembleCorrelatorCPU(config, precomputed_cache=scattered_cache)

    num_windows = len(PD_guess_x_chunk)
    X1, X2, central_index, x_guess, y_guess = _get_pass_grid(pass_idx, config)

    results = []
    statuses = []

    for idx in range(num_windows):
        AA_win = _get_window(AA_chunk, idx, win_size)
        BB_win = _get_window(BB_chunk, idx, win_size)
        AB_win = _get_window(AB_chunk, idx, win_size)

        initial_guess, real_corr = _build_initial_guess(
            idx, pass_idx, AA_win, BB_win, AB_win, central_index,
            x_guess, y_guess, PD_guess_x_chunk[idx], PD_guess_y_chunk[idx],
            win_size, config
        )

        out_params = np.zeros(13, dtype=np.float64)
        out_status = np.zeros(1, dtype=np.int32)

        correlator.marquadt_lib.fit_stacked_gaussian_export(
            ctypes.c_size_t(win_size[0] * win_size[1]),
            X2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            X1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            real_corr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            initial_guess.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_status.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )

        results.append(out_params)
        statuses.append(out_status[0])

    return np.array(results), np.array(statuses)


def _get_window(flat_array, idx, win_size):
    """Extract one window from a flat array."""
    start = idx * win_size[0] * win_size[1]
    end = start + win_size[0] * win_size[1]
    return flat_array[start:end]


def _get_pass_grid(pass_idx, config):
    """Get grid coordinates for Gaussian fitting."""
    runtype = config.ensemble_type[pass_idx]
    wsize = config.ensemble_window_sizes[pass_idx]
    sum_window = config.ensemble_sum_window

    if runtype == "single":
        X1, X2 = np.meshgrid(
            np.linspace(1, sum_window[0], sum_window[0]),
            np.linspace(1, sum_window[1], sum_window[1]),
            indexing="ij",
        )
        X1 = X1.ravel(order="C")
        X2 = X2.ravel(order="C")
        central_index = int(sum_window[0] / 2 * sum_window[1] + sum_window[1] / 2 + 1)
        x_guess = sum_window[1] / 2 + 1
        y_guess = sum_window[0] / 2 + 1
    else:
        X1, X2 = np.meshgrid(
            np.linspace(1, wsize[0], wsize[0]),
            np.linspace(1, wsize[1], wsize[1]),
            indexing="ij",
        )
        X1 = X1.ravel(order="C")
        X2 = X2.ravel(order="C")
        central_index = int(wsize[0] / 2 * wsize[1] + wsize[1] / 2 + 1)
        x_guess = wsize[1] / 2 + 1
        y_guess = wsize[0] / 2 + 1

    return X1, X2, central_index, x_guess, y_guess


def _build_initial_guess(idx, pass_idx, AA_win, BB_win, AB_win, central_index,
                         x_guess, y_guess, PD_guess_x, PD_guess_y, win_size, config):
    """Build initial guess for Gaussian fitting."""
    if pass_idx == 0:
        max_idx = np.argmax(AB_win)
        guess_y_AB, guess_x_AB = np.unravel_index(max_idx, win_size, order="C")

        if config.ensemble_noisy:
            gaussian_radius = 8.0
            yy, xx = np.meshgrid(
                np.arange(1, win_size[0] + 1),
                np.arange(1, win_size[1] + 1),
                indexing="ij",
            )

            def gaussian_window(cx, cy):
                return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * gaussian_radius ** 2))

            cy_AA, cx_AA = np.unravel_index(central_index - 1, win_size, order="C")
            w_AA = gaussian_window(cx_AA + 1, cy_AA + 1).ravel(order="C")
            w_BB = gaussian_window(cx_AA + 1, cy_AA + 1).ravel(order="C")
            w_AB = gaussian_window(guess_x_AB + 1, guess_y_AB + 1).ravel(order="C")

            AA_win = AA_win * w_AA
            BB_win = BB_win * w_BB
            AB_win = AB_win * w_AB

        initial_guess = np.array([
            AA_win[central_index], BB_win[central_index], np.max(AB_win),
            1, 1, 0, PD_guess_x, PD_guess_y, 0.0,
            x_guess, y_guess, guess_x_AB, guess_y_AB,
        ])
    else:
        initial_guess = np.array([
            AA_win[central_index], BB_win[central_index], AB_win[central_index],
            1.0, 1.0, 0.0, PD_guess_x, PD_guess_y, 0.0,
            x_guess, y_guess, x_guess, y_guess,
        ], dtype=np.float64)

    real_corr = np.concatenate([AA_win, BB_win, AB_win]).astype(np.float64)
    return initial_guess, real_corr


# Convenience function for performing ensemble PIV
def perform_ensemble_piv(
    images: da.Array,
    config: Config,
    client: Client,
    scattered_cache: dict,
    vector_masks: Optional[List[np.ndarray]] = None,
) -> PIVEnsembleResult:
    """
    Perform ensemble PIV processing.

    Parameters
    ----------
    images : da.Array
        Dask array of image pairs (N, 2, H, W)
    config : Config
        Configuration object
    client : Client
        Dask client
    scattered_cache : dict
        Pre-scattered correlator cache
    vector_masks : list, optional
        Pre-computed vector masks for each pass

    Returns
    -------
    PIVEnsembleResult
        Complete ensemble PIV results
    """
    executor = EnsembleExecutor(config, client, vector_masks)
    return executor.run_ensemble_piv(images, scattered_cache)
