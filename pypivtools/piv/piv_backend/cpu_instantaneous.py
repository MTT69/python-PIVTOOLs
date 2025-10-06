import ctypes
import logging
import os
import sys
import warnings
from pathlib import Path
import cv2
import dask.array as da
import numpy as np
from dask.distributed import get_worker
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

# Add src to path for unified imports

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from config import Config
from pypivtools.piv.piv_backend.base import CrossCorrelator

from pypivtools.piv.piv_result import PIVPassResult, PIVResult

class InstantaneousCorrelatorCPU(CrossCorrelator):

    def __init__(self, config: Config) -> None:
        super().__init__()
        lib_path = os.path.join(
            os.path.dirname(__file__), "../..", "lib", "libbulkxcorr2d.so"
        )
        lib_path = os.path.abspath(lib_path)
        self.lib = ctypes.CDLL(lib_path)
        self.lib.bulkxcorr2d.restype = ctypes.c_ubyte
        self.delta_ab_pred = None
        self.delta_ab_old = None
        self.prev_win_size = None
        self.prev_win_spacing = None
        self.lib.bulkxcorr2d.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS"),  # fImageA
            np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS"),  # fImageB
            np.ctypeslib.ndpointer(dtype=np.uint8, flags="F_CONTIGUOUS"),  # fMask
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

        self._cache_window_padding(config=config)
        self.H, self.W = config.image_shape
        
        # Cache interpolation grids for performance
        self._cache_interpolation_grids(config=config)

    def correlate_batch(self, images: np.ndarray, config: Config, mask: np.ndarray = None) -> PIVResult:
        """
        Run PIV correlation on a batch of image pairs using libbulkxcorr2d.

        Parameters
        ----------
        images : da.Array
            Dask array of shape (N, 2, H, W), where axis 1 = [ImageA, ImageB].
        config : Config
            Config object containing window sizes, overlap, etc.
        mask : np.ndarray, optional
            Boolean mask array of shape (H, W) where True indicates masked regions.
            Vectors in masked regions will be invalidated (set to NaN).

        Returns
        -------
        da.Array
            Output array of peak displacements per window for each image pair.
            Shape: (N, nPeaks, nWindowsX * nWindowsY, 2) [PkLocX, PkLocY]
        """

        N, C, H, W = images.shape

        piv_result_all = PIVResult()
        self.delta_ab_pred = None
        self.delta_ab_old = None
        self.mask = mask  # Store mask for use in vector invalidation
        im_i = np.arange(self.H)
        im_j = np.arange(self.W)
        im_imat, im_jmat = np.meshgrid(im_i, im_j, indexing="ij")
        self.im_mesh = np.stack([im_imat, im_jmat], axis=-1)

        progress_step = max(1, N // 10)
        # worker = get_worker()
        # logging.info("Worker: %s Starting processing of %d image pairs", worker.name, N)
        for n in range(N):
            if (n + 1) % progress_step == 0 or (n + 1) == N:

                pct = int(((n + 1) / N) * 100)
                # logging.info(
                #     "Worker: %s Processed %d%% of chunk - image pair %d/%d ",
                #     worker.name,
                #     pct,
                #     n + 1,
                #     N,
                # )
            try:

                image_a = np.asarray(images[n, 0], dtype=np.float32)
                image_b = np.asarray(images[n, 1], dtype=np.float32)
                if not image_a.flags["F_CONTIGUOUS"]:
                    image_a = np.asfortranarray(image_a)

                if not image_a.flags["F_CONTIGUOUS"]:
                    print("not contiguous")
                    image_a = np.asfortranarray(image_a)

                image_size = np.asfortranarray(np.array([H, W], dtype=np.int32))
                for pass_idx, win_size in enumerate(config.window_sizes):

                    image_a_prime, image_b_prime, self.delta_ab_pred = (
                        self._predictor_corrector(
                            pass_idx,
                            image_a,
                            image_b,
                            win_type=config.window_type,
                        )
                    )
                    try:
                        (
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
                            win_size,
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
                            logging.error(
                                f"Error in correlate_batch calling for image {n}: error code: {error_code} "
                            )
                            raise RuntimeError(
                                f"bulkxcorr2d failed with error {error_code}"
                            )
                    except Exception as e:
                        import traceback

                        logging.error(
                            f"Error in correlate_batch calling for image {n}:  {e} ",
                        )
                        logging.error(f"{traceback.format_exc()}")
                        raise e

                    try:
                        peak_choice = np.ones(
                            (n_windows[0], n_windows[1]), dtype=np.int32
                        )
                        self.peak_loc_x_after_bulk = np.copy(pk_loc_x)
                        self.peak_loc_y_after_bulk = np.copy(pk_loc_y)
                        for k in range(pk_loc_x.shape[0]):
                            pk_height[k, b_mask.astype(bool)] = np.nan
                        for k in range(pk_loc_y.shape[0]):
                            pk_height[k, b_mask.astype(bool)] = np.nan

                        b_large_disp = (np.abs(pk_loc_x) > win_size[0] / 4) | (
                            np.abs(pk_loc_y) > win_size[1] / 4
                        )
                        pk_loc_x[b_large_disp] = np.nan
                        pk_loc_y[b_large_disp] = np.nan
                        pk_height[b_large_disp] = np.nan

                        # delta_ab_pred is (n_x, n_y, 2) matching pk_loc (n_peaks, n_x, n_y)
                        pk_loc_x += self.delta_ab_pred[:, :, 0][None, :, :]
                        pk_loc_y += self.delta_ab_pred[:, :, 1][None, :, :]
                        rows, cols = np.indices(n_windows)
                        ux_mat = pk_loc_x[peak_choice - 1, rows, cols]
                        uy_mat = pk_loc_y[peak_choice - 1, rows, cols]

                        nan_mask = np.isnan(ux_mat) | np.isnan(uy_mat)
                        
                        # Mark edge vectors as invalid to prevent error propagation
                        # Window centers within EDGE_MARGIN of boundaries are excluded
                        EDGE_MARGIN = 64
                        win_ctrs_x_grid, win_ctrs_y_grid = np.meshgrid(
                            self.win_ctrs_x[pass_idx],
                            self.win_ctrs_y[pass_idx]
                        )
                        edge_mask = (
                            (win_ctrs_x_grid < EDGE_MARGIN) |
                            (win_ctrs_x_grid > self.W - EDGE_MARGIN - 1) |
                            (win_ctrs_y_grid < EDGE_MARGIN) |
                            (win_ctrs_y_grid > self.H - EDGE_MARGIN - 1)
                        )
                        # nan_mask |= edge_mask
                        
                        # Apply user-defined mask if provided
                        if self.mask is not None:
                            user_mask = self._apply_mask_to_vectors(
                                self.win_ctrs_x[pass_idx],
                                self.win_ctrs_y[pass_idx],
                                self.mask
                            )
                            # nan_mask |= user_mask

                        # nan_mask |= self._piv_2d_outlier(
                        #     ux_mat,
                        #     uy_mat,
                        # )

                        # if config.secondary_peak:
                        #     for pk in range(1, n_peaks):
                        #         peak_choice[nan_mask] += 1
                        #         ux_mat = pk_loc_x[peak_choice - 1, rows, cols]
                        #         uy_mat = pk_loc_y[peak_choice - 1, rows, cols]
                        #         nan_mask |= self._piv_2d_outlier(ux_mat, uy_mat)
                        #         if not nan_mask.any():
                        #             break

                        peak_mag = pk_height[peak_choice - 1, rows, cols]

                        nan_mask |= peak_mag < 0.2

                        shifted_pk_height = np.roll(pk_height, shift=-1, axis=0)
                        shifted_pk_height[-1, :, :] = pk_height[-1, :, :]

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            Q_mat = pk_height / shifted_pk_height

                        Q = Q_mat[peak_choice - 1, rows, cols]

                        if nan_mask.any():
                            ux_mat[nan_mask] = np.nan
                            uy_mat[nan_mask] = np.nan

                            if "b_mask" in locals():
                                ux_mat[b_mask.astype(bool)] = 0
                                uy_mat[b_mask.astype(bool)] = 0

                        ux_mat = self._inpaint_nans_opencv(ux_mat)
                        uy_mat = self._inpaint_nans_opencv(uy_mat)

                        peak_choice[nan_mask] = 0
                        Q[nan_mask] = 0
                        peak_mag[nan_mask] = 0

                        if "b_mask" in locals():
                            ux_mat[b_mask.astype(bool)] = 0
                            uy_mat[b_mask.astype(bool)] = 0

                    except Exception as e:
                        import traceback

                        logging.info(f"Error in correlate_batch for image {n}: {e}")
                        logging.info(traceback.format_exc())
                        raise e

                    self.delta_ab_old = np.stack(
                        [ux_mat, uy_mat], axis=2
                    )  # shape (rows, cols, 2)
                    n_pre = self.n_pre_all[pass_idx]
                    n_post = self.n_post_all[pass_idx]

                    self.delta_ab_old = np.pad(
                        self.delta_ab_old,
                        ((n_pre[0], n_post[0]), (n_pre[1], n_post[1]), (0, 0)),
                        mode="edge",
                    )

                    self.previous_win_spacing = [
                        self.win_spacing_x[pass_idx],
                        self.win_spacing_y[pass_idx],
                    ]
                    self.prev_win_size = win_size

                    pass_result = PIVPassResult(
                        n_windows=n_windows,
                        ux_mat=np.copy(ux_mat),  # processed velocities
                        uy_mat=np.copy(uy_mat),
                        nan_mask=np.copy(nan_mask),
                        Q=Q,
                        peak_mag=np.copy(pk_height),
                        peak_choice=peak_choice,
                        predictor_field=np.copy(self.delta_ab_old),
                    )
                    piv_result_all.add_pass(pass_result)
            except Exception as e:
                import traceback

                logging.info(f"Error in correlate_batch for image {n}: {e}")
                logging.info(traceback.format_exc())
                raise e
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
        win_x, win_y = config.window_sizes[pass_idx]
        overlap = config.overlap[pass_idx]

        w_spacing_x = round((1 - overlap / 100) * win_x)
        w_spacing_y = round((1 - overlap / 100) * win_y)

        Nx, Ny = config.image_shape[1], config.image_shape[0]
        
        # Exclude windows within EDGE_MARGIN pixels of image boundaries
        # This prevents unreliable edge vectors from propagating errors
        EDGE_MARGIN = 32  # pixels from edge to exclude
        
        # Calculate valid region for window centers
        min_x = EDGE_MARGIN
        max_x = Nx - EDGE_MARGIN - 1
        min_y = EDGE_MARGIN
        max_y = Ny - EDGE_MARGIN - 1

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

    def _predictor_corrector(
        self,
        pass_idx: int,
        image_a: np.ndarray,
        image_b: np.ndarray,
        interpolator="cubic",
        win_type="A",
    ):
        """Predictor-corrector step to adjust images based on previous displacement estimates.

        :param pass_idx: Index of the current pass.
        :type pass_idx: int
        :param image_a: First image in the pair.
        :type image_a: np.ndarray
        :param image_b: _Second image in the pair.
        :type image_b: np.ndarray
        :param interpolator: Interpolation method, defaults to "cubic"
        :type interpolator: str, optional
        :param win_type: Window type, defaults to "A"
        :type win_type: str, optional
        :return: Adjusted images and predicted displacements.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        # delta_ab_pred must match the C library's grid ordering: (n_x, n_y, 2)
        self.delta_ab_pred = np.zeros(
            (len(self.win_ctrs_x[pass_idx]), len(self.win_ctrs_y[pass_idx]), 2),
            dtype=np.float32,
        )
        if pass_idx == 0:
            if self.delta_ab_old is None:
                self.delta_ab_old = np.zeros_like(self.delta_ab_pred)

            self.prev_win_size = (
                len(self.win_ctrs_x[pass_idx]),
                len(self.win_ctrs_y[pass_idx]),
            )
            self.prev_win_spacing = (
                self.win_spacing_x[pass_idx],
                self.win_spacing_y[pass_idx],
            )
            self.image_a_prime = image_a
            self.image_b_prime = image_b
            return image_a.copy(), image_b.copy(), self.delta_ab_pred

        # Apply Gaussian smoothing with explicit boundary handling
        # mode='nearest' prevents circular/wrap-around effects at boundaries
        self.delta_ab_old[..., 0] = gaussian_filter(
            self.delta_ab_old[..., 0],
            sigma=self.sd[pass_idx],
            truncate=(self.ksize_filt[pass_idx][0] - 1) / (2 * self.sd[pass_idx]),
            mode='nearest'
        )
        self.delta_ab_old[..., 1] = gaussian_filter(
            self.delta_ab_old[..., 1],
            sigma=self.sd[pass_idx],
            truncate=(self.ksize_filt[pass_idx][0] - 1) / (2 * self.sd[pass_idx]),
            mode='nearest'
        )

        self.delta_ab_dense = np.zeros((self.H, self.W, 2), dtype=np.float32)

        # Use cached interpolation grids for performance
        map_x_2d, map_y_2d = self.cached_dense_maps[pass_idx]
        
        # Set interpolation method
        interp_flag = (
            cv2.INTER_CUBIC if interpolator == "cubic"
            else cv2.INTER_LINEAR
        )
        
        # Interpolate displacement field to dense grid
        # borderMode=BORDER_CONSTANT prevents circular/wrap-around effects
        for d in range(2):
            self.delta_ab_dense[..., d] = cv2.remap(
                self.delta_ab_old[..., d].astype(np.float32),
                map_x_2d,
                map_y_2d,
                interp_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        delta_0b = self.delta_ab_dense / 2
        delta_0a = -self.delta_ab_dense / 2
        self.delta_ab_dense_test = np.copy(self.delta_ab_dense)

        im_mesh_A = self.im_mesh + delta_0a
        im_mesh_B = self.im_mesh + delta_0b
        # delta_ab_pred must match the C library's grid ordering: (n_x, n_y, 2)
        self.delta_ab_pred = np.zeros(
            (len(self.win_ctrs_x[pass_idx]), len(self.win_ctrs_y[pass_idx]), 2),
            dtype=np.float32,
        )

        # Convolve with explicit padding to avoid circular convolution
        # mode='symmetric' prevents wrap-around at boundaries
        pad_amt = self.ksize_filt[pass_idx][0] // 2
        padded_x = np.pad(
            self.delta_ab_old[..., 0], pad_amt, mode="symmetric"
        )
        padded_y = np.pad(
            self.delta_ab_old[..., 1], pad_amt, mode="symmetric"
        )

        # mode='valid' with pre-padding ensures no circular convolution
        delta_ab_filt_x = convolve2d(
            padded_x, self.G_smooth_predictor[pass_idx], mode="valid"
        )
        delta_ab_filt_y = convolve2d(
            padded_y, self.G_smooth_predictor[pass_idx], mode="valid"
        )
        delta_ab_filt = np.stack([delta_ab_filt_x, delta_ab_filt_y], axis=-1)

        # Use cached interpolation grids for predictor
        map_x, map_y = self.cached_predictor_maps[pass_idx]

        # Interpolate filtered displacement to predictor grid
        # cv2.remap is faster than map_coordinates
        # borderMode=BORDER_CONSTANT prevents circular/wrap-around effects
        interp_flag = cv2.INTER_CUBIC
        for d in range(2):
            self.delta_ab_pred[..., d] = cv2.remap(
                delta_ab_filt[..., d].astype(np.float32),
                map_x,
                map_y,
                interp_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0
            )

        # Warp images using displacement field
        # cv2.remap is faster than custom C library
        # cv2.remap expects map coordinates in (x, y) format
        # borderMode=BORDER_CONSTANT prevents circular/wrap-around effects
        image_a_prime = cv2.remap(
            image_a.astype(np.float32),
            im_mesh_A[..., 1].astype(np.float32),
            im_mesh_A[..., 0].astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        image_b_prime = cv2.remap(
            image_b.astype(np.float32),
            im_mesh_B[..., 1].astype(np.float32),
            im_mesh_B[..., 0].astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return image_a_prime, image_b_prime, self.delta_ab_pred

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
        n_windows = np.asfortranarray(
            np.array(
                [len(self.win_ctrs_x[pass_idx]), len(self.win_ctrs_y[pass_idx])],
                dtype=np.int32,
            )
        )

        total_windows = n_windows[0] * n_windows[1]

        b_mask = np.asfortranarray(
            np.zeros((n_windows[0], n_windows[1]), dtype=np.uint8)
        )

        n_peaks = np.int32(config.num_peaks)
        i_peak_finder = np.int32(config.peak_finder)
        b_ensemble = bool(config.ensemble_piv)

        pk_loc_x = np.asfortranarray(
            np.zeros((n_peaks, n_windows[0], n_windows[1]), dtype=np.float32)
        )

        pk_loc_y = np.asfortranarray(
            np.zeros((n_peaks, n_windows[0], n_windows[1]), dtype=np.float32)
        )

        pk_height = np.asfortranarray(
            np.zeros(
                (n_peaks, n_windows[0], n_windows[1]),
                dtype=np.float32,
            )
        )

        sx = np.asfortranarray(
            np.zeros((n_peaks, n_windows[0], n_windows[1]), dtype=np.float32)
        )

        sy = np.asfortranarray(
            np.zeros((n_peaks, n_windows[0], n_windows[1]), dtype=np.float32)
        )

        sxy = np.asfortranarray(
            np.zeros((n_peaks, n_windows[0], n_windows[1]), dtype=np.float32)
        )

        correl_plane_out = np.asfortranarray(
            np.zeros(
                total_windows * win_size[0] * win_size[1],
                dtype=np.float32,
            )
        )

        if config.debug:
            args = [
                ("mask", b_mask),
                ("win_ctrs_x", self.win_ctrs_x.astype(np.float32)),
                ("win_ctrs_y", self.win_ctrs_y.astype(np.float32)),
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

        :param config: Configuration object.
        :type config: Config
        """
        self.win_ctrs_x = []
        self.win_ctrs_y = []
        self.win_spacing_x = []
        self.win_spacing_y = []
        self.win_ctrs_x_all = []
        self.win_ctrs_y_all = []
        self.n_pre_all = []
        self.n_post_all = []
        self.ksize_filt = [0]
        self.sd = [0]
        self.G_smooth_predictor = [0]

        H, W = config.image_shape

        for pass_idx, win_size in enumerate(config.window_sizes):
            w_spacing_x, w_spacing_y, win_ctrs_x, win_ctrs_y = (
                self._compute_window_centres(pass_idx, config)
            )

            win_ctrs_x_pre = np.arange(1, win_ctrs_x[0] - w_spacing_x / 2, w_spacing_x)
            if win_ctrs_x_pre.size == 0:
                win_ctrs_x_pre = np.array([1])
            win_ctrs_x_pre -= 1
            win_ctrs_x_post = np.arange(
                W, win_ctrs_x[-1] + w_spacing_x / 2, -w_spacing_x
            )
            if win_ctrs_x_post.size == 0:
                win_ctrs_x_post = np.array([W])
            win_ctrs_x_post -= 1
            win_ctrs_x_old = np.concatenate(
                [win_ctrs_x_pre, win_ctrs_x, win_ctrs_x_post[::-1]]
            )

            win_ctrs_y_pre = np.arange(1, win_ctrs_y[0] - w_spacing_y / 2, w_spacing_y)
            if win_ctrs_y_pre.size == 0:
                win_ctrs_y_pre = np.array([1])
            win_ctrs_y_pre -= 1
            win_ctrs_y_post = np.arange(
                H, win_ctrs_y[-1] + w_spacing_y / 2, -w_spacing_y
            )
            if win_ctrs_y_post.size == 0:
                win_ctrs_y_post = np.array([H])
            win_ctrs_y_post -= 1
            win_ctrs_y_old = np.concatenate(
                [win_ctrs_y_pre, win_ctrs_y, win_ctrs_y_post[::-1]]
            )

            n_pre = [len(win_ctrs_x_pre), len(win_ctrs_y_pre)]
            n_post = [len(win_ctrs_x_post), len(win_ctrs_y_post)]
            self.win_ctrs_x.append(win_ctrs_x.astype(np.float32))
            self.win_ctrs_y.append(win_ctrs_y.astype(np.float32))
            self.win_spacing_x.append(w_spacing_x)
            self.win_spacing_y.append(w_spacing_y)
            self.win_ctrs_x_all.append(win_ctrs_x_old)
            self.win_ctrs_y_all.append(win_ctrs_y_old)
            self.n_pre_all.append(n_pre)
            self.n_post_all.append(n_post)
            if pass_idx > 0:
                k_filt = (
                    np.round(
                        (
                            len(self.win_ctrs_x[pass_idx - 1]),
                            len(self.win_ctrs_y[pass_idx - 1]),
                        )
                        / np.array(
                            (
                                self.win_spacing_x[pass_idx - 1],
                                self.win_spacing_y[pass_idx - 1],
                            )
                        )
                    ).astype(int)
                    + 1
                )
                self.ksize_filt.append(tuple([k + (k % 2 == 0) for k in k_filt]))

                self.sd.append(np.sqrt(np.prod(self.ksize_filt[pass_idx])) / 3 * 0.65)
                self.G_smooth_predictor.append(
                    self._window_weight_fun(
                        self.ksize_filt[pass_idx], config.window_type
                    )
                )
                self.G_smooth_predictor[pass_idx] /= self.G_smooth_predictor[
                    pass_idx
                ].sum()

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

    def _apply_mask_to_vectors(
        self,
        win_ctrs_x: np.ndarray,
        win_ctrs_y: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply user-defined mask to invalidate vectors in masked regions.
        
        A vector is invalidated if its window center falls within a masked region
        (where mask == True).
        
        Parameters
        ----------
        win_ctrs_x : np.ndarray
            1D array of window center x-coordinates
        win_ctrs_y : np.ndarray
            1D array of window center y-coordinates
        mask : np.ndarray
            Boolean mask array of shape (H, W) where True indicates masked regions
            
        Returns
        -------
        np.ndarray
            Boolean mask of shape (len(win_ctrs_y), len(win_ctrs_x)) where
            True indicates vectors to invalidate
        """
        # Create meshgrid of window centers
        win_x_grid, win_y_grid = np.meshgrid(win_ctrs_x, win_ctrs_y)
        
        # Round to nearest pixel indices
        win_x_idx = np.round(win_x_grid).astype(int)
        win_y_idx = np.round(win_y_grid).astype(int)
        
        # Clip to valid image bounds
        win_x_idx = np.clip(win_x_idx, 0, mask.shape[1] - 1)
        win_y_idx = np.clip(win_y_idx, 0, mask.shape[0] - 1)
        
        # Sample mask at window center locations
        # mask[y, x] where True = masked region
        vector_mask = mask[win_y_idx, win_x_idx]
        
        return vector_mask
