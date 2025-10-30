import ctypes
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

# Try to import line_profiler for detailed profiling
try:
    from line_profiler import profile
except ImportError:
    profile = lambda f: f

# Add src to path for unified imports

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from config import Config
from pypivtools.piv.piv_backend.base import CrossCorrelator
from pypivtools.piv.piv_result import PIVPassResult, PIVResult


class InstantaneousCorrelatorCPU(CrossCorrelator):
    @profile
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
        # Add vcpkg bin directory to DLL search path on Windows
        if os.name == "nt":
            vcpkg_bin = os.path.join(os.environ.get('FFTW_LIB_PATH', '').replace('lib', 'bin'))
            if vcpkg_bin and os.path.isdir(vcpkg_bin):
                os.add_dll_directory(vcpkg_bin)
        self.lib = ctypes.CDLL(lib_path)
        self.lib.bulkxcorr2d.restype = ctypes.c_ubyte
        self.delta_ab_pred = None
        self.delta_ab_old = None
        self.prev_win_size = None
        self.prev_win_spacing = None
        self.lib.bulkxcorr2d.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS"),  # fImageA
            np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS"),  # fImageB
            np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS"),  # fMask
            np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS"),  # nImageSize
            np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS"),  # fWinCtrsX
            np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS"),  # fWinCtrsY
            np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS"),  # nWindows
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="F_CONTIGUOUS"
            ),  # fWindowWeightA
            ctypes.c_bool,  # bEnsemble
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="F_CONTIGUOUS"
            ),  # fWindowWeightB
            np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS"),  # nWindowSize
            ctypes.c_int,  # nPeaks
            ctypes.c_int,  # iPeakFinder
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="F_CONTIGUOUS"
            ),  # fPkLocX (output)
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="F_CONTIGUOUS"
            ),  # fPkLocY (output)
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="F_CONTIGUOUS"
            ),  # fPkHeight (output)
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="F_CONTIGUOUS"
            ),  # fSx (output)
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="F_CONTIGUOUS"
            ),  # fSy (output)
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="F_CONTIGUOUS"
            ),  # fSxy (output)
            np.ctypeslib.ndpointer(
                dtype=np.float32, flags="F_CONTIGUOUS"
            ),  # fCorrelPlane_Out (output)
        ]
        self.win_weights = [
            np.asfortranarray(self._window_weight_fun(win_size, config.window_type))
            for win_size in config.window_sizes
        ]

        # Use precomputed cache if provided, otherwise compute it
        if precomputed_cache is not None:
            self._load_precomputed_cache(precomputed_cache)
            logging.info("Loaded precomputed correlator cache")
        else:
            self._cache_window_padding(config=config)
            self.H, self.W = config.image_shape
            # Cache interpolation grids for performance
            self._cache_interpolation_grids(config=config)

        # Initialize vector masks (will be set in correlate_batch)
        self.vector_masks = []

        # Store pass times for profiling
        self.pass_times = []

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

    @profile
    def correlate_batch(  # type: ignore[override]
        self, images: np.ndarray, config: Config, vector_masks: List[np.ndarray] | None = None
    ) -> PIVResult:
        """Run PIV correlation on a batch of image pairs with MATLAB-style indexing."""

        N, _, H, W = images.shape

        piv_result_all = PIVResult()
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

        for n in range(N):
            try:
                image_a = np.asarray(images[n, 0], dtype=np.float32)
                image_b = np.asarray(images[n, 1], dtype=np.float32)

                if not image_a.flags["F_CONTIGUOUS"]:
                    image_a = np.asfortranarray(image_a)
                if not image_b.flags["F_CONTIGUOUS"]:
                    image_b = np.asfortranarray(image_b)

                image_size = np.asfortranarray(np.array([H, W], dtype=np.int32))

                for pass_idx, win_size in enumerate(config.window_sizes):
                    pass_start = time.perf_counter()
                    image_a_prime, image_b_prime, self.delta_ab_pred = (
                        self._predictor_corrector(
                            pass_idx,
                            image_a,
                            image_b,
                            win_type=config.window_type,
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
                    )

                    error_code = self.lib.bulkxcorr2d(
                        np.asfortranarray(image_a_prime),
                        np.asfortranarray(image_b_prime),
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

                    if error_code != 0:
                        raise RuntimeError(f"bulkxcorr2d failed with error {error_code}")

                    n_win_y = int(n_windows[0])
                    n_win_x = int(n_windows[1])
                    mask_bool = b_mask.astype(bool)

                    pk_loc_x[:, mask_bool] = np.nan
                    pk_loc_y[:, mask_bool] = np.nan
                    pk_height[:, mask_bool] = np.nan

                    win_height, win_width = win_size_arr.astype(np.int32)
                    large_disp_mask = (
                        (np.abs(pk_loc_x) > win_width / 4.0)
                        | (np.abs(pk_loc_y) > win_height / 4.0)
                    )
                    pk_loc_x[large_disp_mask] = np.nan
                    pk_loc_y[large_disp_mask] = np.nan
                    pk_height[large_disp_mask] = np.nan

                    pk_loc_x += self.delta_ab_pred[..., 0][np.newaxis, :, :]
                    pk_loc_y += self.delta_ab_pred[..., 1][np.newaxis, :, :]

                    primary_idx = np.zeros((1, n_win_y, n_win_x), dtype=np.intp)
                    ux_mat = np.take_along_axis(pk_loc_x, primary_idx, axis=0)[0]
                    uy_mat = np.take_along_axis(pk_loc_y, primary_idx, axis=0)[0]
                    # Use direct indexing without meshgrid for outlier detection and peak selection
                    n_win_y = int(n_windows[0])
                    n_win_x = int(n_windows[1])
                    peak_choice = np.ones((n_win_y, n_win_x), dtype=np.int32)

                    # Initial peak selection
                    ux_mat = pk_loc_x[0]
                    uy_mat = pk_loc_y[0]

                    nan_mask = np.isnan(ux_mat) | np.isnan(uy_mat)
                    if config.outlier_detection:
                        nan_mask |= self._piv_2d_outlier(ux_mat, uy_mat)

                    # No-op edge mask (no vectors excluded by edges)
                    edge_mask = np.zeros((n_win_y, n_win_x), dtype=bool)

                    if config.secondary_peak:
                        for pk in range(1, n_peaks):
                            # Increment peak_choice for nan_mask locations
                            peak_choice[nan_mask] += 1
                            # Clamp peak_choice to valid range
                            peak_choice = np.clip(peak_choice, 1, n_peaks)
                            # Select new peak for nan_mask locations
                            ux_mat = np.choose(peak_choice - 1, pk_loc_x)
                            uy_mat = np.choose(peak_choice - 1, pk_loc_y)
                            if config.outlier_detection:
                                nan_mask |= self._piv_2d_outlier(ux_mat, uy_mat)
                            if not nan_mask.any():
                                break

                    # Select primary peak magnitude
                    primary_peak_mag = np.choose(peak_choice - 1, pk_height)
                    nan_mask |= np.isnan(primary_peak_mag)

                    nan_mask |= mask_bool
                    nan_mask |= primary_peak_mag < 0.2

                    # Q calculation (peak ratio)
                    shifted_pk_height = np.roll(pk_height, shift=-1, axis=0)
                    shifted_pk_height[-1, :, :] = pk_height[-1, :, :]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        Q_mat = np.divide(
                            pk_height,
                            shifted_pk_height,
                            out=np.zeros_like(pk_height),
                            where=shifted_pk_height > 0,
                        )

                    Q = np.choose(peak_choice - 1, Q_mat)

                    if nan_mask.any():
                        ux_mat[nan_mask] = np.nan
                        uy_mat[nan_mask] = np.nan
                        primary_peak_mag[nan_mask] = np.nan
                        Q[nan_mask] = 0.0

                    ux_mat[mask_bool] = 0.0
                    uy_mat[mask_bool] = 0.0

                    if np.isnan(ux_mat).any():
                        ux_mat = self._inpaint_nans_biharm(ux_mat)
                    if np.isnan(uy_mat).any():
                        uy_mat = self._inpaint_nans_biharm(uy_mat)

                    ux_mat[mask_bool] = 0.0
                    uy_mat[mask_bool] = 0.0
                    peak_choice[nan_mask] = 0

                    ux_mat = np.ascontiguousarray(ux_mat.astype(np.float32))
                    uy_mat = np.ascontiguousarray(uy_mat.astype(np.float32))
                    nan_mask = np.ascontiguousarray(nan_mask)
                    Q = np.ascontiguousarray(Q.astype(np.float32))
                    primary_peak_mag = np.ascontiguousarray(
                        np.where(nan_mask, 0.0, primary_peak_mag.astype(np.float32))
                    )
                    pk_height = np.asfortranarray(pk_height.astype(np.float32))
                    Q_mat = np.asfortranarray(Q_mat.astype(np.float32))

                    self.delta_ab_old = np.stack([ux_mat, uy_mat], axis=2)
                    pre_y, pre_x = self.n_pre_all[pass_idx]
                    post_y, post_x = self.n_post_all[pass_idx]
                    self.delta_ab_old = np.pad(
                        self.delta_ab_old,
                        ((pre_y, post_y), (pre_x, post_x), (0, 0)),
                        mode="edge",
                    )

                    self.previous_win_spacing = (
                        self.win_spacing_y[pass_idx],
                        self.win_spacing_x[pass_idx],
                    )
                    self.prev_win_size = (n_win_y, n_win_x)

                    pass_result = PIVPassResult(
                        n_windows=np.array([n_win_y, n_win_x], dtype=np.int32),
                        ux_mat=np.copy(ux_mat),
                        uy_mat=np.copy(uy_mat),
                        nan_mask=np.copy(nan_mask),
                        Q=np.copy(Q),
                        peak_mag=np.copy(pk_height),
                        peak_choice=np.copy(peak_choice),
                        predictor_field=np.copy(self.delta_ab_old),
                        b_mask=b_mask.reshape((n_win_y, n_win_x)).astype(bool),
                        window_size=win_size,
                        spacing=(self.win_spacing_y[pass_idx], self.win_spacing_x[pass_idx]),
                        win_ctrs_x=self.win_ctrs_x[pass_idx],
                        win_ctrs_y=self.win_ctrs_y[pass_idx],
                        edge_mask=edge_mask,
                    )
                    pass_time = time.perf_counter() - pass_start
                    self.pass_times.append((n, pass_idx, pass_time))
                    piv_result_all.add_pass(pass_result)

            except Exception as exc:
                logging.error("Error in correlate_batch for image %d: %s", n, exc)
                logging.error(traceback.format_exc())
                raise

        return piv_result_all

    def _compute_window_centres(
        self, pass_idx: int, config: Config
    ) -> tuple[int, int, np.ndarray, np.ndarray]:
        """
        Compute window centers and spacing for a given pass.
        
        Window centers are constrained to be at least EDGE_MARGIN pixels away
        from image boundaries to avoid unreliable edge vectors where particles
        move out of frame.

        :param pass_idx: Index of the current pass.
        :type pass_idx: int
        :param config: Configuration object containing window sizes, overlap, and image shape.
        :type config: Config
        :return: Tuple containing window spacing in x and y, and arrays of window center coordinates in x and y.
        :rtype: tuple[int, int, np.ndarray, np.ndarray]
        """
        # CRITICAL FIX: config.window_sizes is in (H, W) = (Y-size, X-size) format
        # because C library uses column-major order where first dimension is vertical (Y)
        win_y, win_x = config.window_sizes[pass_idx]
        overlap = config.overlap[pass_idx]

        w_spacing_x = round((1 - overlap / 100) * win_x)
        w_spacing_y = round((1 - overlap / 100) * win_y)

        Nx, Ny = config.image_shape[1], config.image_shape[0]
        
        # Temporarily disable exclusion of windows near image edges.
        # This removes the EDGE_MARGIN constraint so window centres cover
        # the full image. To re-enable edge exclusion, restore the
        # original EDGE_MARGIN logic below.
        #
        # Original (commented):
        # EDGE_MARGIN = 32  # pixels from edge to exclude
        # min_x = EDGE_MARGIN
        # max_x = Nx - EDGE_MARGIN - 1
        # min_y = EDGE_MARGIN
        # max_y = Ny - EDGE_MARGIN - 1

        # Use full image bounds for window centre placement
        min_x = 0
        max_x = Nx - 1
        min_y = 0
        max_y = Ny - 1

        # Original window center range
        first_ctr_x = -0.5 + win_x / 2
        first_ctr_y = -0.5 + win_y / 2
        
        # Adjust starting positions to respect edge margin
        start_ctr_x = max(first_ctr_x, min_x)
        start_ctr_y = max(first_ctr_y, min_y)
        
        # Calculate number of windows that fit in the valid region
        n_win_x = int(np.floor((max_x - start_ctr_x) / w_spacing_x)) + 1
        n_win_y = int(np.floor((max_y - start_ctr_y) / w_spacing_y)) + 1
        
        # Ensure at least one window
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
    @profile
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
    @profile
    def _piv_2d_outlier(
        self,
        ux: np.ndarray,
        uy: np.ndarray,
        epsilon: float = 0.2,
        threshold: float = 2.0,
    ):
        """Check for outliers in 2D PIV data.

        :param ux: Horizontal velocity component.
        :type ux: np.ndarray
        :param uy: Vertical velocity component.
        :type uy: np.ndarray
        :param epsilon: Regularization parameter, defaults to 0.2.
        :type epsilon: float, optional
        :param threshold: Outlier threshold, defaults to 2.0.
        :type threshold: float, optional
        :return: Boolean array indicating outliers.
        :rtype: np.ndarray
        """
        n_wx, n_wy = ux.shape

        ui = np.stack([ux, uy], axis=-1)

        r_0p = np.zeros((n_wx, n_wy, 2))
        n_neighbours = np.zeros((n_wx, n_wy, 2))

        for c in range(2):
            U = ui[..., c]
            U_pad = np.pad(U, 1, mode="constant", constant_values=np.nan)

            U_nn = np.zeros((n_wx, n_wy, 8))
            U_nn[..., 0] = U_pad[0:-2, 0:-2]
            U_nn[..., 1] = U_pad[0:-2, 1:-1]
            U_nn[..., 2] = U_pad[0:-2, 2:]
            U_nn[..., 3] = U_pad[1:-1, 0:-2]
            U_nn[..., 4] = U_pad[1:-1, 2:]
            U_nn[..., 5] = U_pad[2:, 2:]
            U_nn[..., 6] = U_pad[2:, 1:-1]
            U_nn[..., 7] = U_pad[2:, 0:-2]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                U_med = np.nanmedian(U_nn, axis=2)

            r_0 = np.abs(U_med - U)
            r_i = np.abs(U_nn - U_med[..., None])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                r_m = np.nanmedian(r_i, axis=2)

            r_0p[..., c] = r_0 / (r_m + epsilon)

            # Count valid neighbors using convolution with explicit boundaries
            # boundary='fill' with fillvalue=0 prevents circular wrap-around
            n_neigh = convolve2d(
                np.logical_not(np.isnan(U)).astype(float),
                np.ones((3, 3)),
                mode="same",
                boundary="fill",
                fillvalue=0,
            )
            n_neighbours[..., c] = n_neigh
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            r_0_combined = np.nanmax(r_0p, axis=2)
        b_filter = (
            (r_0_combined > threshold)
            | np.isnan(r_0p).any(axis=2)
            | (n_neighbours < 6).any(axis=2)
        )
        return b_filter
    
    # @profile
    # def _piv_2d_outlier(
    #     self,
    #     ux: np.ndarray,
    #     uy: np.ndarray,
    #     epsilon: float = 0.2,
    #     threshold: float = 2.0,
    # ):
    #     """Check for outliers in 2D PIV data.
        
    #     :param ux: Horizontal velocity component.
    #     :type ux: np.ndarray
    #     :param uy: Vertical velocity component.
    #     :type uy: np.ndarray
    #     :param epsilon: Regularization parameter, defaults to 0.2.
    #     :type epsilon: float, optional
    #     :param threshold: Outlier threshold, defaults to 2.0.
    #     :type threshold: float, optional
    #     :return: Boolean array indicating outliers.
    #     :rtype: np.ndarray
    #     """
    #     n_wx, n_wy = ux.shape
        
    #     # Stack arrays once for vectorized processing
    #     ui = np.stack([ux, uy], axis=-1)
        
    #     # Pre-allocate output arrays
    #     r_0p = np.zeros((n_wx, n_wy, 2))
    #     n_neighbours = np.zeros((n_wx, n_wy, 2))
        
    #     for c in range(2):
    #         U = ui[..., c]
    #         U_pad = np.pad(U, 1, mode="constant", constant_values=np.nan)
            
    #         # Use stride_tricks for faster neighbor extraction (avoids creating full U_nn array)
    #         # Or stick with slicing but use views more efficiently
    #         U_nn = np.array([
    #             U_pad[0:-2, 0:-2],
    #             U_pad[0:-2, 1:-1],
    #             U_pad[0:-2, 2:],
    #             U_pad[1:-1, 0:-2],
    #             U_pad[1:-1, 2:],
    #             U_pad[2:, 2:],
    #             U_pad[2:, 1:-1],
    #             U_pad[2:, 0:-2]
    #         ])  # Shape: (8, n_wx, n_wy)
            
    #         U_nn = np.moveaxis(U_nn, 0, -1)  # Shape: (n_wx, n_wy, 8)
            
    #         # Replace nanmedian with nanmean or percentile-based approximation
    #         # nanmedian is O(n log n), while nanmean is O(n)
    #         # For outlier detection, mean often works nearly as well
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore", category=RuntimeWarning)
    #             # Option 1: Use nanmean (much faster, ~50x speedup)
    #             U_med = np.nanmean(U_nn, axis=2)
                
    #             # Option 2: Use approximate median via sorting only valid values
    #             # (faster than nanmedian but slower than nanmean)
    #             # U_med = np.nanpercentile(U_nn, 50, axis=2)
            
    #         r_0 = np.abs(U_med - U)
    #         r_i = np.abs(U_nn - U_med[..., None])
            
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore", category=RuntimeWarning)
    #             r_m = np.nanmean(r_i, axis=2)  # Replace nanmedian with nanmean
            
    #         r_0p[..., c] = r_0 / (r_m + epsilon)
            
    #         # Neighbor counting - use correlate2d which can be faster
    #         valid_mask = ~np.isnan(U)
    #         n_neigh = convolve2d(
    #             valid_mask.astype(np.float32),  # Use float32 for speed
    #             np.ones((3, 3), dtype=np.float32),
    #             mode="same",
    #             boundary="fill",
    #             fillvalue=0,
    #         )
    #         n_neighbours[..., c] = n_neigh
        
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", category=RuntimeWarning)
    #         r_0_combined = np.nanmax(r_0p, axis=2)
        
    #     b_filter = (
    #         (r_0_combined > threshold)
    #         | np.isnan(r_0p).any(axis=2)
    #         | (n_neighbours < 6).any(axis=2)
    #     )
        
    #     return b_filter

    def _set_lib_arguments(
        self,
        config: Config,
        win_size: np.ndarray,
        pass_idx: int,
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
        win_size = np.asfortranarray(np.array(win_size, dtype=np.int32))

        n_win_y = len(self.win_ctrs_y[pass_idx])
        n_win_x = len(self.win_ctrs_x[pass_idx])
        n_windows = np.asfortranarray(
            np.array([n_win_y, n_win_x], dtype=np.int32)
        )

        total_windows = n_win_y * n_win_x

        # Use precomputed vector mask for this pass if available
        if hasattr(self, 'vector_masks') and self.vector_masks and pass_idx < len(self.vector_masks):
            cached_mask = self.vector_masks[pass_idx]
            b_mask = np.asfortranarray(cached_mask.astype(np.float32))
        else:
            b_mask = np.asfortranarray(np.zeros((n_win_y, n_win_x), dtype=np.float32))
            logging.debug("No vector mask applied for pass %d", pass_idx)

        n_peaks = np.int32(config.num_peaks)
        i_peak_finder = np.int32(config.peak_finder)
        b_ensemble = bool(config.ensemble_piv)

        out_shape = (n_peaks, n_win_y, n_win_x)
        pk_loc_x = np.asfortranarray(np.zeros(out_shape, dtype=np.float32))
        pk_loc_y = np.asfortranarray(np.zeros(out_shape, dtype=np.float32))
        pk_height = np.asfortranarray(np.zeros(out_shape, dtype=np.float32))
        sx = np.asfortranarray(np.zeros(out_shape, dtype=np.float32))
        sy = np.asfortranarray(np.zeros(out_shape, dtype=np.float32))
        sxy = np.asfortranarray(np.zeros(out_shape, dtype=np.float32))

        correl_plane_out = np.asfortranarray(
            np.zeros(total_windows * win_size[0] * win_size[1], dtype=np.float32)
        )

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

    @profile
    def _cache_window_padding(self, config: Config) -> None:
        """Cache window padding information.

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

        for pass_idx, _ in enumerate(config.window_sizes):
            spacing_x, spacing_y, win_ctrs_x, win_ctrs_y = self._compute_window_centres(
                pass_idx, config
            )

            win_ctrs_x_pre = np.arange(1, win_ctrs_x[0] - spacing_x / 2, spacing_x)
            if win_ctrs_x_pre.size == 0:
                win_ctrs_x_pre = np.array([1])
            win_ctrs_x_pre -= 1
            win_ctrs_x_post = np.arange(
                W, win_ctrs_x[-1] + spacing_x / 2, -spacing_x
            )
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
            win_ctrs_y_post = np.arange(
                H, win_ctrs_y[-1] + spacing_y / 2, -spacing_y
            )
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
                self.ksize_filt.append((0, 0))
                self.sd.append(0.0)
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
                g_kernel = self._window_weight_fun(k_filt_tuple, config.window_type)
                g_kernel = g_kernel.astype(np.float32)
                g_kernel /= max(np.sum(g_kernel), 1e-12)
                self.G_smooth_predictor.append(g_kernel)

        # Verify window padding cache integrity
        assert len(self.win_ctrs_x) == len(config.window_sizes), f"Window centers X cache length mismatch: {len(self.win_ctrs_x)} vs {len(config.window_sizes)}"
        assert len(self.win_ctrs_y) == len(config.window_sizes), f"Window centers Y cache length mismatch: {len(self.win_ctrs_y)} vs {len(config.window_sizes)}"
        assert len(self.win_spacing_x) == len(config.window_sizes), f"Window spacing X cache length mismatch: {len(self.win_spacing_x)} vs {len(config.window_sizes)}"
        assert len(self.win_spacing_y) == len(config.window_sizes), f"Window spacing Y cache length mismatch: {len(self.win_spacing_y)} vs {len(config.window_sizes)}"
        
        # Check that cached values are reasonable
        for pass_idx in range(len(config.window_sizes)):
            assert len(self.win_ctrs_x[pass_idx]) > 0, f"No X window centers cached for pass {pass_idx}"
            assert len(self.win_ctrs_y[pass_idx]) > 0, f"No Y window centers cached for pass {pass_idx}"
            assert self.win_spacing_x[pass_idx] > 0, f"Invalid X spacing for pass {pass_idx}: {self.win_spacing_x[pass_idx]}"
            assert self.win_spacing_y[pass_idx] > 0, f"Invalid Y spacing for pass {pass_idx}: {self.win_spacing_y[pass_idx]}"
        
        logging.info(f"Successfully cached window padding for {len(config.window_sizes)} passes")

    @profile
    @profile
    def _cache_interpolation_grids(self, config: Config) -> None:
        """Cache interpolation grid coordinates for reuse across passes.

        This significantly improves performance by avoiding repeated
        computation of coordinate grids.
        
        :param config: Configuration object.
        :type config: Config
        """
        # Cache the image mesh for dense interpolation
        y_coords = np.arange(self.H, dtype=np.float32)
        x_coords = np.arange(self.W, dtype=np.float32)
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
        self.im_mesh = np.stack([y_mesh, x_mesh], axis=-1)
        
        # Pre-cache coordinate mappings for each pass
        self.cached_dense_maps = []
        self.cached_predictor_maps = []
        
        for pass_idx in range(len(config.window_sizes)):
            if pass_idx == 0:
                self.cached_dense_maps.append(None)
                self.cached_predictor_maps.append(None)
            else:
                # Cache dense interpolation maps
                points = (
                    self.win_ctrs_y_all[pass_idx - 1],
                    self.win_ctrs_x_all[pass_idx - 1]
                )
                map_x_1d = np.interp(
                    x_coords, points[1], np.arange(len(points[1]))
                )
                map_y_1d = np.interp(
                    y_coords, points[0], np.arange(len(points[0]))
                )
                map_x_2d, map_y_2d = np.meshgrid(
                    map_x_1d.astype(np.float32),
                    map_y_1d.astype(np.float32)
                )
                self.cached_dense_maps.append((map_x_2d, map_y_2d))
                
                # Cache predictor interpolation maps
                win_x, win_y = np.meshgrid(
                    self.win_ctrs_x[pass_idx],
                    self.win_ctrs_y[pass_idx]
                )
                ix = np.interp(
                    win_x.ravel(), points[1], np.arange(len(points[1]))
                )
                iy = np.interp(
                    win_y.ravel(), points[0], np.arange(len(points[0]))
                )
                map_x = ix.reshape(win_x.shape).astype(np.float32)
                map_y = iy.reshape(win_x.shape).astype(np.float32)
                self.cached_predictor_maps.append((map_x, map_y))

        # Verify caching integrity
        assert len(self.cached_dense_maps) == len(config.window_sizes), f"Dense maps cache length mismatch: {len(self.cached_dense_maps)} vs {len(config.window_sizes)}"
        assert len(self.cached_predictor_maps) == len(config.window_sizes), f"Predictor maps cache length mismatch: {len(self.cached_predictor_maps)} vs {len(config.window_sizes)}"
        
        # Check that non-zero passes have cached maps
        for pass_idx in range(1, len(config.window_sizes)):
            assert self.cached_dense_maps[pass_idx] is not None, f"Dense map for pass {pass_idx} is None"
            assert self.cached_predictor_maps[pass_idx] is not None, f"Predictor map for pass {pass_idx} is None"
            dense_x, dense_y = self.cached_dense_maps[pass_idx]
            pred_x, pred_y = self.cached_predictor_maps[pass_idx]
            assert dense_x.shape == (self.H, self.W), f"Dense map X shape incorrect for pass {pass_idx}: {dense_x.shape} vs {(self.H, self.W)}"
            assert dense_y.shape == (self.H, self.W), f"Dense map Y shape incorrect for pass {pass_idx}: {dense_y.shape} vs {(self.H, self.W)}"
            expected_pred_shape = (len(self.win_ctrs_y[pass_idx]), len(self.win_ctrs_x[pass_idx]))
            assert pred_x.shape == expected_pred_shape, f"Predictor map X shape incorrect for pass {pass_idx}: {pred_x.shape} vs {expected_pred_shape}"
            assert pred_y.shape == expected_pred_shape, f"Predictor map Y shape incorrect for pass {pass_idx}: {pred_y.shape} vs {expected_pred_shape}"
        
        logging.info(f"Successfully cached interpolation grids for {len(config.window_sizes)} passes")

    # def _apply_mask_to_vectors(
    #     self,
    #     win_ctrs_x: np.ndarray,
    #     win_ctrs_y: np.ndarray,
    #     mask: np.ndarray
    # ) -> np.ndarray:
    #     """
    #     Apply user-defined mask to invalidate vectors in masked regions.
        
    #     A vector is invalidated if its window center falls within a masked region
    #     (where mask == True).
        
    #     Parameters
    #     ----------
    #     win_ctrs_x : np.ndarray
    #         1D array of window center x-coordinates
    #     win_ctrs_y : np.ndarray
    #         1D array of window center y-coordinates
    #     mask : np.ndarray
    #         Boolean mask array of shape (H, W) where True indicates masked regions
            
    #     Returns
    #     -------
    #     np.ndarray
    #         Boolean mask of shape (len(win_ctrs_y), len(win_ctrs_x)) where
    #         True indicates vectors to invalidate
    #     """
    #     grid_y, grid_x = np.meshgrid(win_ctrs_y, win_ctrs_x, indexing="ij")

    #     # Round to nearest pixel indices
    #     x_idx = np.round(grid_x).astype(int)
    #     y_idx = np.round(grid_y).astype(int)

    #     # Clip to valid image bounds
    #     x_idx = np.clip(x_idx, 0, mask.shape[1] - 1)
    #     y_idx = np.clip(y_idx, 0, mask.shape[0] - 1)

    #     # Sample mask at window center locations
    #     # mask[y, x] where True = masked region
    #     vector_mask = mask[y_idx, x_idx]

    #     return vector_mask
