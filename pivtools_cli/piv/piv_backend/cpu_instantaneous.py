import atexit
import ctypes
import gc
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import List, Optional
import cv2
import dask.array as da
import numpy as np
from dask.distributed import get_worker
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

from concurrent.futures import ThreadPoolExecutor, as_completed
from pivtools_core.config import Config
from pivtools_core.window_utils import compute_window_centers
from pivtools_cli.piv.piv_backend.base import CrossCorrelator
from pivtools_cli.piv.piv_result import PIVPassResult, PIVResult
from pivtools_cli.piv.piv_backend.outlier_detection import apply_outlier_detection
from pivtools_cli.piv.piv_backend.infilling import apply_infilling

import matplotlib
matplotlib.use("Agg") 
from matplotlib import pyplot as plt
class InstantaneousCorrelatorCPU(CrossCorrelator):
    # Class variable to track if FFTW cleanup has been registered
    _cleanup_registered = False
    _lib_for_cleanup = None

    def __init__(self, config: Config, precomputed_cache: Optional[dict] = None) -> None:
        super().__init__()
        # Use platform-appropriate library extension
        lib_extension = ".dll" if os.name == "nt" else ".so"
        lib_path = os.path.join(
            os.path.dirname(__file__), "../..", "lib", f"libbulkxcorr2d{lib_extension}"
        )
        lib_path = os.path.abspath(lib_path)
        if not os.path.isfile(lib_path):
            raise FileNotFoundError(f"Required library file not found: {lib_path}")
        self.lib = ctypes.CDLL(lib_path)

        # Register FFTW cleanup to run at interpreter exit (once per process)
        if not InstantaneousCorrelatorCPU._cleanup_registered:
            self._register_fftw_cleanup()
            InstantaneousCorrelatorCPU._cleanup_registered = True
        self.lib.bulkxcorr2d.restype = ctypes.c_ubyte
        self.delta_ab_pred = None
        self.delta_ab_old = None
        self.prev_win_size = None
        self.prev_win_spacing = None
        # Updated to use C-contiguous (row-major) arrays
        self.lib.bulkxcorr2d.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageA
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageB
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fMask
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # nImageSize
            ctypes.c_int, 
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsX
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsY
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # nWindows
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="C_CONTIGUOUS"
            ),  # fWindowWeightA
            ctypes.c_bool,  # bEnsemble
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="C_CONTIGUOUS"
            ),  # fWindowWeightB
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # nWindowSize
            ctypes.c_int,  # nPeaks
            ctypes.c_int,  # iPeakFinder
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="C_CONTIGUOUS"
            ),  # fPkLocX (output)
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="C_CONTIGUOUS"
            ),  # fPkLocY (output)
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="C_CONTIGUOUS"
            ),  # fPkHeight (output)
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="C_CONTIGUOUS"
            ),  # fSx (output)
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="C_CONTIGUOUS"
            ),  # fSy (output)
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="C_CONTIGUOUS"
            ),  # fSxy (output)
            ctypes.c_void_p,  # fCorrelPlane_Out (nullable - use c_void_p for instantaneous NULL)
        ]
        # Window weights should be C-contiguous with shape (win_height, win_width)
        self.win_weights = [
            np.ascontiguousarray(self._window_weight_fun(win_size, config.window_type))
            for win_size in config.window_sizes
        ]

        # Use precomputed cache if provided, otherwise compute it
        if precomputed_cache is not None:
            self._load_precomputed_cache(precomputed_cache)
        else:
            self._cache_window_padding(config=config)
            self.H, self.W = config.image_shape
            # Cache interpolation grids for performance
            self._cache_interpolation_grids(config=config)

        # Initialize vector masks (will be set in correlate_batch)
        self.vector_masks = []

        # Store pass times for profiling
        self.pass_times = []

    def _register_fftw_cleanup(self) -> None:
        """Register FFTW cleanup to run at interpreter exit.

        This prevents segfaults during garbage collection by ensuring
        FFTW threads and wisdom are properly cleaned up before the
        library is unloaded.
        """
        # Store library reference at class level to keep it alive
        InstantaneousCorrelatorCPU._lib_for_cleanup = self.lib

        # Declare the cleanup function signature
        if hasattr(self.lib, 'fftw_library_cleanup'):
            self.lib.fftw_library_cleanup.argtypes = []
            self.lib.fftw_library_cleanup.restype = None

            @atexit.register
            def _cleanup_fftw():
                lib = InstantaneousCorrelatorCPU._lib_for_cleanup
                if lib is not None:
                    try:
                        lib.fftw_library_cleanup()
                    except Exception:
                        pass  # Ignore errors during cleanup

    def _load_precomputed_cache(self, cache: dict) -> None:
        """Load precomputed cache data to avoid redundant computation.

        :param cache: Dictionary containing precomputed cache data
        :type cache: dict
        """
        # Load window padding cache
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
        
        # Load image dimensions
        self.H = cache['H']
        self.W = cache['W']
        
        # Load interpolation grids cache
        self.im_mesh = cache['im_mesh']
        self.cached_dense_maps = cache['cached_dense_maps']
        self.cached_predictor_maps = cache['cached_predictor_maps']

    def get_cache_data(self) -> dict:
        """Extract cache data for sharing across workers.
        
        :return: Dictionary containing all cached data
        :rtype: dict
        """
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

    
    def correlate_batch(  # type: ignore[override]
        self, images: np.ndarray, config: Config, vector_masks: List[np.ndarray] | None = None
    ) -> PIVResult:
        """Run PIV correlation on a batch of image pairs with MATLAB-style indexing."""

        # Validate input shape
        if images.ndim != 4:
            raise ValueError(f"Expected 4D images array (N, 2, H, W), got {images.ndim}D array with shape {images.shape}")

        N, C, H, W = images.shape

        # Validate channel dimension
        if C != 2:
            raise ValueError(f"Expected 2 channels (image pairs), got {C} channels")

        # Validate non-zero dimensions
        if H == 0 or W == 0:
            raise ValueError(f"Invalid image dimensions: H={H}, W={W}. Images appear to be empty.")

        # Log image batch info
        logging.debug(f"Processing batch: N={N} pairs, shape=({H}, {W}), dtype={images.dtype}")

        self.delta_ab_pred = None
        self.delta_ab_old = None

        # Clear pass times for this batch
        self.pass_times = []

        # Use pre-computed vector masks
        self.vector_masks = vector_masks if vector_masks is not None else []

        y_coords = np.arange(self.H, dtype=np.float32)
        x_coords = np.arange(self.W, dtype=np.float32)
        y_mesh, x_mesh = np.meshgrid(y_coords, x_coords, indexing="ij")
        self.im_mesh = np.stack([y_mesh, x_mesh], axis=-1)
        logging.debug(f"Processing batch of {N} image pairs")

        try:
                # Convert images to C-contiguous (row-major) format
            #images_a = images[:, 0, :, :].astype(np.float32, copy=False)
            #images_b = images[:, 1, :, :].astype(np.float32, copy=False)

#            if not images_a.flags["C_CONTIGUOUS"]:
#                images_a = np.ascontiguousarray(images_a)
#            if not images_b.flags["C_CONTIGUOUS"]:
#                images_b = np.ascontiguousarray(images_b)

                # Pass image_size as [H, W] in C-contiguous format
            image_size = np.ascontiguousarray(np.array([H, W], dtype=np.int32))
            batch_results = [PIVResult() for _ in range(N)]

            for pass_idx, win_size in enumerate(config.window_sizes):

                pass_start = time.perf_counter()
                logging.debug(f"\n{'='*60}")
                logging.debug(f"PASS {pass_idx + 1} of {len(config.window_sizes)} (window_size={win_size})")
                logging.debug(f"{'='*60}")

                images_a_prime, images_b_prime, self.delta_ab_pred = (
                        self._predictor_corrector_batch(
                            pass_idx,
                            images[:, 0, :, :].astype(np.float32, copy=False),
                            images[:, 1, :, :].astype(np.float32, copy=False),
                            config=config
                        )
                    )

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
                    ) = self._set_lib_arguments(
                        config=config,
                        win_size=win_size,
                        pass_idx=pass_idx,
                        N=N
                    )
                

                    # Ensure images are C-contiguous before passing to C library
                image_a_prime_c = images_a_prime if images_a_prime.flags["C_CONTIGUOUS"] else np.ascontiguousarray(images_a_prime)
                image_b_prime_c = images_b_prime if images_b_prime.flags["C_CONTIGUOUS"] else np.ascontiguousarray(images_b_prime)
                    
                try:
                    error_code = self.lib.bulkxcorr2d(
                            image_a_prime_c,
                            image_b_prime_c,
                            b_mask,
                            image_size,
                            N,
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
                except Exception as e:
                    logging.error(f"    Exception type: {type(e).__name__}")
                    logging.error(traceback.format_exc())
                    raise
                # Free warped images immediately after C library call to reduce peak memory
                del images_a_prime, images_b_prime, image_a_prime_c, image_b_prime_c

                n_win_y = len(self.win_ctrs_y[pass_idx])
                n_win_x = len(self.win_ctrs_x[pass_idx])
                #plot_corr_planes(
                #    correl_plane_out[0].ravel(order="C"),
                #    n_win_y,
                #    n_win_x,
                #    win_size[0],
                #    win_size[1],
                #    pass_idx,
                #)
                if error_code != 0:
                    error_names = {
                            1: "ERROR_NOMEM (out of memory)",
                            2: "ERROR_NOPLAN_FWD (FFT forward plan failed)",
                            4: "ERROR_NOPLAN_BWD (FFT backward plan failed)",
                            8: "ERROR_NOPLAN (general plan error)",
                            9: "ERROR_OUT_OF_BOUNDS (array access out of bounds)"
                        }
                    error_msg = error_names.get(error_code, f"Unknown error code {error_code}")
                    logging.error(f"    bulkxcorr2d returned error code {error_code}: {error_msg}")
                    raise RuntimeError(f"bulkxcorr2d failed with error {error_code}: {error_msg}")
                n_win_y = int(n_windows[0])
                n_win_x = int(n_windows[1])

                win_height, win_width = win_size_arr

                # Large displacement threshold: more lenient for first pass (no predictor yet),
                # stricter for subsequent passes where predictor-corrector reduces residuals
                if pass_idx == 0:
                    # First pass: use win/2 threshold to allow larger displacements
                    thresh_x = win_width / 2.0
                    thresh_y = win_height / 2.0
                else:
                    # Subsequent passes: use win/4 threshold (predictor reduces residuals)
                    thresh_x = win_width / 4.0
                    thresh_y = win_height / 4.0

                mask_batch = np.broadcast_to(b_mask[None, :, :], (N, n_win_y, n_win_x))
                mask_bool_batch = mask_batch.astype(bool)
                large_disp_mask = (np.abs(pk_loc_x) > thresh_x) | (np.abs(pk_loc_y) > thresh_y)
                # Broadcast mask to match pk_loc_x shape (N, n_peaks, n_win_y, n_win_x)
                mask_for_peaks = np.broadcast_to(mask_bool_batch[:, None, :, :], pk_loc_x.shape)
                invalid_peaks = (
                    mask_for_peaks | large_disp_mask
                )

                pk_loc_x[invalid_peaks] = np.nan
                pk_loc_y[invalid_peaks] = np.nan
                pk_height[invalid_peaks] = np.nan
                dx = self.delta_ab_pred[..., 1]
                dy = self.delta_ab_pred[..., 0]

                dx_exp = dx[:, None, :, :]  # (N,1,n_win_y,n_win_x)
                dy_exp = dy[:, None, :, :]
                pk_loc_x += dx_exp
                pk_loc_y += dy_exp

                primary_idx = np.zeros((N, 1, n_win_y, n_win_x), dtype=np.intp)

                ux_mat = np.take_along_axis(pk_loc_x, primary_idx, axis=1)[
                    :, 0, :, :
                ]  # (N, ny, nx)
                uy_mat = np.take_along_axis(pk_loc_y, primary_idx, axis=1)[:, 0, :, :]

                nan_mask = (
                    np.isnan(ux_mat)
                    | np.isnan(uy_mat)
                    | mask_bool_batch
                )

                ux_mat[nan_mask] = 0.0
                uy_mat[nan_mask] = 0.0
                peak_choice = np.ones((N, n_win_y, n_win_x), dtype=np.intp)

                outliers_detected_count = 0
                if config.outlier_detection_enabled:
                    outlier_methods = config.outlier_detection_methods
                    if outlier_methods:
                        for im_idx in range(N):
                            primary_peak_mag_0 = pk_height[im_idx, 0]
                            outlier_mask = apply_outlier_detection(
                                ux_mat[im_idx],
                                uy_mat[im_idx],
                                outlier_methods,
                                peak_mag=primary_peak_mag_0,
                            )
                            outliers_detected_count += outlier_mask.sum()
                            nan_mask[im_idx] |= outlier_mask

                if config.secondary_peak:
                    for pk in range(1, n_peaks):
                        active = nan_mask & (peak_choice < n_peaks)
                        if not active.any():
                            break

                        peak_choice[active] += 1

                        idx = peak_choice[:, None, :, :] - 1
                        ux_mat = np.take_along_axis(pk_loc_x, idx, axis=1)[:, 0]
                        uy_mat = np.take_along_axis(pk_loc_y, idx, axis=1)[:, 0]

                        if config.outlier_detection_enabled and outlier_methods:
                            primary_peak_mag = np.take_along_axis(pk_height, idx, axis=1)[:, 0]
                            for im_idx in range(N):
                                if nan_mask[im_idx].any():
                                    outlier_mask = apply_outlier_detection(
                        ux_mat[im_idx],
                        uy_mat[im_idx],
                        outlier_methods,
                        peak_mag=primary_peak_mag[im_idx],
                    )
                                    nan_mask[im_idx] |= outlier_mask

                idx = peak_choice[:, None, :, :] - 1
                primary_peak_mag = np.take_along_axis(pk_height, idx, axis=1)[:, 0]
                nan_mask |= np.isnan(primary_peak_mag)

                shifted_pk_height = np.roll(pk_height, shift=-1, axis=1)
                shifted_pk_height[:, -1, :, :] = pk_height[:, -1, :, :]

                with np.errstate(divide="ignore", invalid="ignore"):
                    Q_mat = np.divide(
        pk_height,
        shifted_pk_height,
        out=np.zeros_like(pk_height),
        where=shifted_pk_height > 0,
    )

                Q = np.take_along_axis(Q_mat, idx, axis=1)[:, 0]

                ux_mat[nan_mask] = np.nan
                uy_mat[nan_mask] = np.nan
                primary_peak_mag[nan_mask] = np.nan
                Q[nan_mask] = 0.0

                ux_mat[mask_bool_batch] = 0.0
                uy_mat[mask_bool_batch] = 0.0
                peak_choice[nan_mask] = 0

                is_final_pass = (pass_idx == len(config.window_sizes) - 1)

                # Apply infilling to outlier/invalid vectors
                for im_idx in range(N):
                    if np.isnan(ux_mat[im_idx]).any() or np.isnan(uy_mat[im_idx]).any():
                        infill_mask = np.isnan(ux_mat[im_idx]) | np.isnan(uy_mat[im_idx])
                        cfg = (
                            config.infilling_final_pass
                            if is_final_pass
                            else config.infilling_mid_pass
                        )

                        if infill_mask.sum() == infill_mask.size:
                            logging.error(f"CRITICAL: 100% of data needs infilling for image {im_idx}!")

                        if cfg.get("enabled", True):
                            ux_mat[im_idx], uy_mat[im_idx] = apply_infilling(
                                ux_mat[im_idx],
                                uy_mat[im_idx],
                                infill_mask,
                                cfg,
                            )

                # Pass summary - only warn if >20% invalid vectors (excluding masked regions)
                mask_count = mask_bool_batch.sum()
                unmasked_total = ux_mat.size - mask_count
                true_invalid_count = nan_mask.sum() - mask_count  # nan_mask includes mask, subtract it
                invalid_rate = true_invalid_count / unmasked_total if unmasked_total > 0 else 0
                if invalid_rate > 0.20:
                    logging.warning(
                        f"Pass {pass_idx + 1}: {invalid_rate*100:.1f}% invalid vectors "
                        f"({true_invalid_count}/{unmasked_total})"
                    )

                pre_y, pre_x = self.n_pre_all[pass_idx]
                post_y, post_x = self.n_post_all[pass_idx]

                stacked = np.stack([uy_mat, ux_mat], axis=-1)  # (N, ny, nx, 2)

                self.delta_ab_old = np.pad(
                    stacked,
                    ((0, 0), (pre_y, post_y), (pre_x, post_x), (0, 0)),
                    mode="edge",
                )




                self.previous_win_spacing = (
                    self.win_spacing_y[pass_idx],
                    self.win_spacing_x[pass_idx],
                )
                self.prev_win_size = (n_win_y, n_win_x)
                #logging.info(f"Average ux: {np.nanmean(ux_mat):.4f}, uy: {np.nanmean(uy_mat):.4f} for pass {pass_idx + 1}")
                for im_idx in range(N):
                    pass_result = PIVPassResult(
                        n_windows=np.array([n_win_y, n_win_x], dtype=np.int32),
                        ux_mat=np.copy(ux_mat[im_idx]),
                        uy_mat=np.copy(uy_mat[im_idx]),
                        nan_mask=np.copy(nan_mask[im_idx]),
                        peak_mag=np.copy(pk_height[im_idx]),
                        peak_choice=np.copy(peak_choice[im_idx]),
                        predictor_field=np.copy(self.delta_ab_old[im_idx]),
                        b_mask=b_mask.reshape((n_win_y, n_win_x)).astype(bool),
                        window_size=win_size,
                        win_ctrs_x=self.win_ctrs_x[pass_idx],
                        win_ctrs_y=self.win_ctrs_y[pass_idx],

                    )
                    pass_time = time.perf_counter() - pass_start
                    self.pass_times.append((im_idx, pass_idx, pass_time))
                    batch_results[im_idx].add_pass(pass_result)

                # Explicit memory cleanup after each pass to prevent accumulation
                # Delete large intermediate arrays that are no longer needed
                del pk_loc_x, pk_loc_y, pk_height, Q_mat
                del sx, sy, sxy  # These are returned by C lib but not used
                del stacked, mask_batch, mask_bool_batch, mask_for_peaks
                del invalid_peaks, large_disp_mask
                # NOTE: gc.collect() was causing SIGSEGV during FFTW cleanup
                # The explicit del statements above should be sufficient

        except Exception as exc:
            logging.error("Error in correlate_batch: %s", exc)
            logging.error(traceback.format_exc())
            raise

        return batch_results

    def _compute_window_centres(
        self, pass_idx: int, config: Config
    ) -> tuple[int, int, np.ndarray, np.ndarray, tuple]:
        """
        Compute window centers and spacing for a given pass using centralized utilities.

        Uses pivtools_core.window_utils.compute_window_centers() for consistency
        across all PIV modes (instantaneous, ensemble, masking).

        :param pass_idx: Index of the current pass.
        :type pass_idx: int
        :param config: Configuration object containing window sizes, overlap, and image shape.
        :type config: Config
        :return: Tuple containing window spacing in x and y, arrays of window center coordinates
                 in x and y, and padding tuple (top, bottom, left, right) - always (0,0,0,0) for
                 instantaneous mode.
        :rtype: tuple[int, int, np.ndarray, np.ndarray, tuple]
        """
        win_height, win_width = config.window_sizes[pass_idx]
        overlap = config.overlap[pass_idx]

        logging.debug(f"_compute_window_centres pass {pass_idx}:")
        logging.debug(f"  Image shape (H, W) = {config.image_shape}")
        logging.debug(f"  Window size (H, W) = ({win_height}, {win_width})")
        logging.debug(f"  Overlap = {overlap}%")

        # Use centralized window center computation
        result = compute_window_centers(
            image_shape=config.image_shape,
            window_size=(win_height, win_width),
            overlap=overlap,
            validate=True
        )

        logging.debug(f"  Window spacing (X, Y) = ({result.win_spacing_x}, {result.win_spacing_y})")
        logging.debug(f"  Number of windows (X, Y) = ({result.n_win_x}, {result.n_win_y})")

        if len(result.win_ctrs_x) > 0:
            logging.debug(
                f"  win_ctrs_x: min={result.win_ctrs_x.min():.2f}, "
                f"max={result.win_ctrs_x.max():.2f}, len={len(result.win_ctrs_x)}"
            )
        else:
            logging.warning(f"  win_ctrs_x: EMPTY ARRAY (len=0)")

        if len(result.win_ctrs_y) > 0:
            logging.debug(
                f"  win_ctrs_y: min={result.win_ctrs_y.min():.2f}, "
                f"max={result.win_ctrs_y.max():.2f}, len={len(result.win_ctrs_y)}"
            )
        else:
            logging.warning(f"  win_ctrs_y: EMPTY ARRAY (len=0)")

        return (
            result.win_spacing_x,
            result.win_spacing_y,
            np.ascontiguousarray(result.win_ctrs_x),
            np.ascontiguousarray(result.win_ctrs_y),
            (0, 0, 0, 0),  # No padding for instantaneous mode
        )
    def _predictor_corrector_batch(
        self,
        pass_idx: int,
        images_a: np.ndarray,
        images_b: np.ndarray,
        interpolator="cubic",
        config: Config = None,
    ):
        interp_flag = cv2.INTER_CUBIC if interpolator == "cubic" else cv2.INTER_LINEAR
        N, H, W = images_a.shape
        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])

        self.delta_ab_pred = np.zeros((N, n_win_y, n_win_x, 2), dtype=np.float32)
        if pass_idx == 0:
            if self.delta_ab_old is None:
                self.delta_ab_old = np.zeros((N, n_win_y, n_win_x, 2), dtype=np.float32)

            self.prev_win_size = (n_win_y, n_win_x)
            self.prev_win_spacing = (
                self.win_spacing_y[pass_idx],
                self.win_spacing_x[pass_idx],
            )

            return (
                images_a.copy(),
                images_b.copy(),
                self.delta_ab_pred,
            )
        else:
            if self.delta_ab_old is None:
                raise RuntimeError(
                    "delta_ab_old is uninitialised before predictor step"
                )

            # DEBUG: Check delta_ab_old from previous pass
            logging.debug(f"  delta_ab_old from prev pass: shape={self.delta_ab_old.shape}, "
                        f"nan_count={np.isnan(self.delta_ab_old).sum()}, "
                        f"dx range=[{np.nanmin(self.delta_ab_old[...,1]):.4f}, {np.nanmax(self.delta_ab_old[...,1]):.4f}], "
                        f"dy range=[{np.nanmin(self.delta_ab_old[...,0]):.4f}, {np.nanmax(self.delta_ab_old[...,0]):.4f}]")

            sigma = self.sd[pass_idx]
            truncate = (self.ksize_filt[pass_idx][0] - 1) / (2 * sigma)
            # thread over images here
            with ThreadPoolExecutor(max_workers=int(config.omp_threads)) as ex:
                futures = [
                    ex.submit(self._smooth_one_delta_old, i, sigma, truncate)
                    for i in range(N)
                ]
                for f in as_completed(futures):
                    _ = f.result()

            delta_ab_dense = np.zeros((N, H, W, 2), dtype=np.float32)
            map_x_2d, map_y_2d = self.cached_dense_maps[pass_idx]
            if map_x_2d is None or map_y_2d is None:
                raise ValueError(
                    f"Dense interpolation maps missing for pass {pass_idx}"
                )
            # If installed properly cv2 should be multithreaded here
            # could explicitly use multithreading over images
            for i in range(N):
                for d in range(2):
                    delta_ab_dense[i, ..., d] = cv2.remap(
                        self.delta_ab_old[i, ..., d],
                        map_x_2d,
                        map_y_2d,
                        interp_flag,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    ).astype(np.float32)

            im_mesh_base = self.im_mesh.astype(np.float32)
            im_mesh_base_batched = im_mesh_base[None, ...]  # (1,H,W,2)
            im_mesh_A = im_mesh_base_batched - 0.5 * delta_ab_dense  # (N,H,W,2)
            im_mesh_B = im_mesh_base_batched + 0.5 * delta_ab_dense  # (N,H,W,2)
            map_x, map_y = self.cached_predictor_maps[pass_idx]
            if map_x is None or map_y is None:
                raise ValueError(
                    f"Predictor interpolation maps missing for pass {pass_idx}"
                )
            for i in range(N):
                for d in range(2):
                    self.delta_ab_pred[i, ..., d] = cv2.remap(
                        self.delta_ab_old[i, ..., d],
                        map_x,
                        map_y,
                        interp_flag,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0,
                    ).astype(np.float32)

            image_a_prime_batch = np.zeros_like(images_a, dtype=np.float32)
            image_b_prime_batch = np.zeros_like(images_b, dtype=np.float32)
            # If installed properly cv2 should be multithreaded here
            # could explicitly use multithreading over images
            for i in range(N):
                map_x_A = im_mesh_A[i, ..., 1]
                map_y_A = im_mesh_A[i, ..., 0]
                map_x_B = im_mesh_B[i, ..., 1]
                map_y_B = im_mesh_B[i, ..., 0]
                image_a_prime_batch[i] = cv2.remap(
                    images_a[i].astype(np.float32),
                    map_x_A.astype(np.float32),
                    map_y_A.astype(np.float32),
                    cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                image_b_prime_batch[i] = cv2.remap(
                    images_b[i].astype(np.float32),
                    map_x_B.astype(np.float32),
                    map_y_B.astype(np.float32),
                    cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            return image_a_prime_batch, image_b_prime_batch, self.delta_ab_pred

    def _smooth_one_delta_old(self, i, sigma, truncate):
        self.delta_ab_old[i, ..., 0] = gaussian_filter(
            self.delta_ab_old[i, ..., 0],
            sigma=sigma,
            truncate=truncate,
            mode="nearest",
        )
        self.delta_ab_old[i, ..., 1] = gaussian_filter(
            self.delta_ab_old[i, ..., 1],
            sigma=sigma,
            truncate=truncate,
            mode="nearest",
        )

    def _check_args(self, *args):
        """Check the arguments for consistency and validity if debug mode is enabled.
        Parameters
        ----------
        *args : list of tuples
            Each tuple contains (name, array) to be checked.

        """

        def _describe(arr):
            if isinstance(arr, np.ndarray):
                return (arr.shape, arr.dtype, arr.flags["C_CONTIGUOUS"])
            return (type(arr), arr)

        for name, arr in args:
            logging.info(f"{name}: {_describe(arr)}")
    
    def _predictor_corrector(
        self,
        pass_idx: int,
        image_a: np.ndarray,
        image_b: np.ndarray,
        interpolator="cubic",
        win_type="A",
    ):
        """Predictor-corrector step to adjust images based on previous displacement estimates."""

        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])
        self.delta_ab_pred = np.zeros((n_win_y, n_win_x, 2), dtype=np.float32)

        if pass_idx == 0:
            if self.delta_ab_old is None:
                self.delta_ab_old = np.zeros_like(self.delta_ab_pred)

            self.prev_win_size = (n_win_y, n_win_x)
            self.prev_win_spacing = (
                self.win_spacing_y[pass_idx],
                self.win_spacing_x[pass_idx],
            )
            return image_a.copy(), image_b.copy(), self.delta_ab_pred

        if self.delta_ab_old is None:
            raise RuntimeError("delta_ab_old is uninitialised before predictor step")

        interp_flag = cv2.INTER_CUBIC if interpolator == "cubic" else cv2.INTER_LINEAR

        self.delta_ab_old[..., 0] = gaussian_filter(
            self.delta_ab_old[..., 0],
            sigma=self.sd[pass_idx],
            truncate=(self.ksize_filt[pass_idx][0] - 1) / (2 * self.sd[pass_idx]),
            mode="nearest",
        )
        self.delta_ab_old[..., 1] = gaussian_filter(
            self.delta_ab_old[..., 1],
            sigma=self.sd[pass_idx],
            truncate=(self.ksize_filt[pass_idx][0] - 1) / (2 * self.sd[pass_idx]),
            mode="nearest",
        )

        self.delta_ab_dense = np.zeros((self.H, self.W, 2), dtype=np.float32)
        map_x_2d, map_y_2d = self.cached_dense_maps[pass_idx]
        if map_x_2d is None or map_y_2d is None:
            raise ValueError(f"Dense interpolation maps missing for pass {pass_idx}")
        
        # Verify cached dense maps have correct shape
        assert map_x_2d.shape == (self.H, self.W), f"Cached dense map X shape mismatch for pass {pass_idx}: {map_x_2d.shape} vs {(self.H, self.W)}"
        assert map_y_2d.shape == (self.H, self.W), f"Cached dense map Y shape mismatch for pass {pass_idx}: {map_y_2d.shape} vs {(self.H, self.W)}"
        logging.debug(f"Using cached dense interpolation maps for pass {pass_idx}")

        for d in range(2):
            self.delta_ab_dense[..., d] = cv2.remap(
                self.delta_ab_old[..., d].astype(np.float32),
                map_x_2d,
                map_y_2d,
                interp_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        delta_0b = self.delta_ab_dense / 2
        delta_0a = -delta_0b
        im_mesh_A = self.im_mesh + delta_0a
        im_mesh_B = self.im_mesh + delta_0b

        map_x, map_y = self.cached_predictor_maps[pass_idx]
        if map_x is None or map_y is None:
            raise ValueError(f"Predictor interpolation maps missing for pass {pass_idx}")
        
        # Verify cached predictor maps have correct shape
        expected_pred_shape = (len(self.win_ctrs_y[pass_idx]), len(self.win_ctrs_x[pass_idx]))
        assert map_x.shape == expected_pred_shape, f"Cached predictor map X shape mismatch for pass {pass_idx}: {map_x.shape} vs {expected_pred_shape}"
        assert map_y.shape == expected_pred_shape, f"Cached predictor map Y shape mismatch for pass {pass_idx}: {map_y.shape} vs {expected_pred_shape}"
        logging.debug(f"Using cached predictor interpolation maps for pass {pass_idx}")

        for d in range(2):
            remapped = cv2.remap(
                self.delta_ab_old[..., d].astype(np.float32),
                map_x,
                map_y,
                interp_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            self.delta_ab_pred[..., d] = remapped

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

        return image_a_prime, image_b_prime, self.delta_ab_pred
    
    def _set_lib_arguments(
        self,
        config: Config,
        win_size: np.ndarray,
        pass_idx: int,
        N: int,
    ):
        """Set library arguments for PIV computation.

        :param config: Configuration object.
        :type config: Config
        :param win_size: Window size.
        :type win_size: np.ndarray
        :param pass_idx: Pass index.
        :type pass_idx: int
        :return: Tuple of library arguments.
        :rtype: tuple
        """
        # Window size: [win_height, win_width] in C-contiguous format
        win_size = np.ascontiguousarray(np.array(win_size, dtype=np.int32))

        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])
        # nWindows: [n_win_y, n_win_x] where n_win_y = rows, n_win_x = cols
        n_windows = np.ascontiguousarray(
            np.array([n_win_y, n_win_x], dtype=np.int32)
        )

        total_windows = n_win_y * n_win_x

        # Use precomputed vector mask for this pass if available
        # Mask shape: (n_win_y, n_win_x) in C-contiguous format
        if hasattr(self, 'vector_masks') and self.vector_masks and pass_idx < len(self.vector_masks):
            cached_mask = self.vector_masks[pass_idx]
            b_mask = np.ascontiguousarray(cached_mask.astype(np.float32))
        else:
            b_mask = np.ascontiguousarray(np.zeros((n_win_y, n_win_x), dtype=np.float32))
            logging.debug("No vector mask applied for pass %d", pass_idx)

        n_peaks = np.int32(config.num_peaks if config.num_peaks else 1)
        if n_peaks < 1:
            raise ValueError(f"num_peaks must be >= 1, got {n_peaks}")
        i_peak_finder = np.int32(config.peak_finder)
        b_ensemble = False  # Instantaneous PIV, not ensemble

        # Output arrays shape: (n_peaks, n_win_y, n_win_x) in C-contiguous format
        #out_shape = (n_peaks, n_win_y, n_win_x)
        out_shape = (N, n_peaks, n_win_y, n_win_x)
        pk_loc_x = np.zeros(out_shape, dtype=np.float32)
        pk_loc_y = np.zeros(out_shape, dtype=np.float32)
        pk_height = np.zeros(out_shape, dtype=np.float32)
        sx = np.zeros(out_shape, dtype=np.float32)
        sy = np.zeros(out_shape, dtype=np.float32)
        sxy = np.zeros(out_shape, dtype=np.float32)

        # Instantaneous mode doesn't need correlation planes - pass None to skip memcpy in C
        correl_plane_out = None
        if config.debug:
            args = [
                ("mask", b_mask),
                ("win_ctrs_x", self.win_ctrs_x[pass_idx].astype(np.float32)),
                ("win_ctrs_y", self.win_ctrs_y[pass_idx].astype(np.float32)),
                ("n_windows", n_windows),
                ("window_weight_a", self.win_weights[pass_idx]),
                ("b_ensemble", b_ensemble),
                ("window_weight_b", self.win_weights[pass_idx]),
                ("win_size", win_size),
                ("n_peaks", int(n_peaks)),
                ("i_peak_finder", int(i_peak_finder)),
                ("pk_loc_x", pk_loc_x),
                ("pk_loc_y", pk_loc_y),
                ("pk_height", pk_height),
                ("sx", sx),
                ("sy", sy),
                ("sxy", sxy),
                ("correl_plane_out", correl_plane_out),
            ]
            self._check_args(*args)

        return (
            win_size,
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
        )

    
    def _cache_window_padding(self, config: Config) -> None:
        """Cache window padding information.

        Uses unified base class implementation with instantaneous-specific parameters.
        """
        self._cache_window_padding_unified(
            config=config,
            window_sizes=config.window_sizes,
            window_type=config.window_type,
            compute_window_fn=self._compute_window_centres,
            first_pass_ksize=(0, 0),  # Instantaneous uses (0, 0) for first pass
        )

    
    
    def _cache_interpolation_grids(self, config: Config) -> None:
        """Cache interpolation grid coordinates for reuse across passes.

        Uses unified base class implementation with instantaneous-specific parameters.
        """
        self._cache_interpolation_grids_unified(
            config=config,
            window_sizes=config.window_sizes,
            include_first_pass=False,  # Instantaneous doesn't need first pass grids
        )

        # Verify caching integrity (keep assertions for debugging)
        assert len(self.cached_dense_maps) == len(config.window_sizes), \
            f"Dense maps cache length mismatch: {len(self.cached_dense_maps)} vs {len(config.window_sizes)}"
        assert len(self.cached_predictor_maps) == len(config.window_sizes), \
            f"Predictor maps cache length mismatch: {len(self.cached_predictor_maps)} vs {len(config.window_sizes)}"

        for pass_idx in range(1, len(config.window_sizes)):
            assert self.cached_dense_maps[pass_idx] is not None, \
                f"Dense map for pass {pass_idx} is None"
            assert self.cached_predictor_maps[pass_idx] is not None, \
                f"Predictor map for pass {pass_idx} is None"
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
    logging.info(f"{corr_planes.shape}")
    try:
        fig, axes = plt.subplots(n_win_y, n_win_x, figsize=(3 * n_win_x, 3 * n_win_y))
        logging.info(f"Created figure with {n_win_y} rows and {n_win_x} columns of subplots")
        for i in range(n_win_y):
            for j in range(n_win_x):
                ax = axes[i, j]
                plane = corr_planes[i, j]
                im = ax.imshow(plane, origin="lower", cmap="viridis")
                ax.set_title(f"W{i},{j}")
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        logging.info(f"Plotted correlation planes for pass {pass_idx}")
        plt.tight_layout()
        plt.savefig(f"corr{pass_idx}.png", dpi=150, bbox_inches="tight")
        logging.info(f"Saved: corr{pass_idx}.png")
    except Exception as e:
        logging.error(f"Error occurred while plotting correlation planes: {e}")
    finally:    
        plt.close()