"""
Stereo Ensemble PIV CPU Backend — Correlation-of-Correlations Pipeline

Manages dewarping, dual-camera correlation, and per-frame CoC computation.
Uses composition (not inheritance) with EnsembleCorrelatorCPU for per-camera
correlation via the C library.

Reference: manual_tools/Ensemble_stereo/stereo_coc_validation.py
"""

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from math import acos, sqrt
from typing import Optional, Tuple

import cv2
import numpy as np

from pivtools_core.config import Config
from pivtools_gui.stereo_reconstruction.self_calibration import (
    PinholeCamera,
    compute_dewarp_maps,
)


class StereoEnsembleCorrelatorCPU:
    """
    Stereo ensemble PIV correlator with Correlation-of-Correlations.

    Composition approach: creates an internal EnsembleCorrelatorCPU instance
    (configured for the dewarped image shape) for per-camera correlation calls.
    The stereo-specific logic (dewarping, CoC, dual-camera coordination) lives here.
    """

    def __init__(
        self,
        config: Config,
        cam1: PinholeCamera,
        cam2: PinholeCamera,
        output_size: Tuple[int, int],
        world_bounds: Tuple[float, float, float, float],
        self_cal_z: float = 0.0,
        self_cal_tilt_x: float = 0.0,
        self_cal_tilt_y: float = 0.0,
        precomputed_cache: Optional[dict] = None,
        vector_masks: Optional[list] = None,
    ):
        """
        Parameters
        ----------
        config : Config
            Configuration object
        cam1, cam2 : PinholeCamera
            Stereo camera models (cam1 = reference, cam2 = secondary)
        output_size : (H, W)
            Size of dewarped output images
        world_bounds : (x_min, x_max, y_min, y_max)
            World coordinate bounds in mm
        self_cal_z/tilt_x/tilt_y : float
            Self-calibration corrections
        precomputed_cache : dict, optional
            Pre-scattered cache from factory (for Dask workers)
        vector_masks : list, optional
            Per-pass vector masks
        """
        self.config = config
        self.cam1 = cam1
        self.cam2 = cam2
        self.output_size = output_size
        self.world_bounds = world_bounds

        # Build dewarp maps ONCE (fixed for all frames/passes)
        logging.info("Building dewarp maps for stereo ensemble PIV...")
        self.dewarp_maps_cam1 = compute_dewarp_maps(
            cam1, output_size, world_bounds, self_cal_z, self_cal_tilt_x, self_cal_tilt_y
        )
        self.dewarp_maps_cam2 = compute_dewarp_maps(
            cam2, output_size, world_bounds, self_cal_z, self_cal_tilt_x, self_cal_tilt_y
        )
        logging.info(
            f"Dewarp maps built: output_size={output_size}, "
            f"world_bounds={world_bounds}"
        )

        # Compute stereo geometry
        R_rel = cam2.R @ cam1.R.T
        trace_val = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
        full_angle = acos(trace_val)
        self.stereo_half_angle = full_angle / 2
        x_min, x_max, y_min, y_max = world_bounds
        H_dw, W_dw = output_size
        self.mm_per_pixel = ((x_max - x_min) / W_dw + (y_max - y_min) / H_dw) / 2
        logging.info(
            f"Stereo geometry: half_angle={np.degrees(self.stereo_half_angle):.2f} deg, "
            f"mm_per_pixel={self.mm_per_pixel:.4f}"
        )

        # Create internal ensemble correlator for per-camera correlation
        # Override image_shape in config to match dewarped size
        from .cpu_ensemble import EnsembleCorrelatorCPU

        self._correlator = EnsembleCorrelatorCPU(
            config=config,
            precomputed_cache=precomputed_cache,
            vector_masks=vector_masks,
        )

        # Thread pool for parallel dewarping
        self._n_threads = max(1, int(config.omp_threads))
        cv2.setNumThreads(1)
        self._pool = ThreadPoolExecutor(max_workers=self._n_threads)

    def get_cache_data(self) -> dict:
        """Return the correlator's precomputed cache for Dask scattering."""
        return self._correlator.get_cache_data()

    def dewarp_batch(
        self, images: np.ndarray, maps: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        Dewarp a batch of image pairs using precomputed remap tables.

        Parameters
        ----------
        images : np.ndarray
            Shape (N, 2, H_raw, W_raw) float32
        maps : (map_x, map_y)
            Float32 remap tables of shape (H_dw, W_dw)

        Returns
        -------
        np.ndarray
            Shape (N, 2, H_dw, W_dw) float32
        """
        map_x, map_y = maps
        N = images.shape[0]
        H_dw, W_dw = map_x.shape
        out = np.empty((N, 2, H_dw, W_dw), dtype=np.float32)

        def _dewarp_one(n, ch):
            out[n, ch] = cv2.remap(
                images[n, ch],
                map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

        futures = []
        for n in range(N):
            for ch in range(2):
                futures.append(self._pool.submit(_dewarp_one, n, ch))
        for f in futures:
            f.result()

        return out

    def correlate_batch_stereo(
        self,
        images_cam1: np.ndarray,
        images_cam2: np.ndarray,
        config: Config,
        pass_idx: int,
        predictor_field: Optional[np.ndarray] = None,
        is_first_batch: bool = False,
    ) -> dict:
        """
        Main method: dewarp, correlate both cameras, compute CoC.

        Parameters
        ----------
        images_cam1, images_cam2 : np.ndarray
            Raw image batches, shape (N, 2, H_raw, W_raw)
        config : Config
        pass_idx : int
        predictor_field : np.ndarray, optional
            Predictor from previous pass, shape (n_win_y, n_win_x, 2)
        is_first_batch : bool

        Returns
        -------
        dict
            Combined results with keys for both cameras' correlation sums,
            warp sums, and CoC sum.
        """
        N = images_cam1.shape[0]

        # Step 1: Dewarp both cameras to common world plane
        dw_cam1 = self.dewarp_batch(images_cam1, self.dewarp_maps_cam1)
        dw_cam2 = self.dewarp_batch(images_cam2, self.dewarp_maps_cam2)

        # Step 2: Per-camera ensemble correlation via internal correlator
        # The correlator handles predictor warping, padding, and C library calls
        result_cam1 = self._correlator.correlate_batch_for_accumulation(
            dw_cam1, config, pass_idx,
            predictor_field=predictor_field,
            is_first_batch=is_first_batch,
        )
        result_cam2 = self._correlator.correlate_batch_for_accumulation(
            dw_cam2, config, pass_idx,
            predictor_field=predictor_field,
            is_first_batch=False,
        )

        # Step 3: CoC — per-frame cross-correlation of correlation maps
        # For each frame n, cross-correlate cam1's AB map with cam2's AB map
        coc_sum = self._compute_coc_batch(
            dw_cam1, dw_cam2, config, pass_idx, predictor_field
        )

        # Build combined result
        combined = {
            # Camera 1
            "cam1_corr_AA_sum": result_cam1["corr_AA_sum"],
            "cam1_corr_BB_sum": result_cam1["corr_BB_sum"],
            "cam1_corr_AB_sum": result_cam1["corr_AB_sum"],
            "cam1_warp_A_sum": result_cam1["warp_A_sum"],
            "cam1_warp_B_sum": result_cam1["warp_B_sum"],
            # Camera 2
            "cam2_corr_AA_sum": result_cam2["corr_AA_sum"],
            "cam2_corr_BB_sum": result_cam2["corr_BB_sum"],
            "cam2_corr_AB_sum": result_cam2["corr_AB_sum"],
            "cam2_warp_A_sum": result_cam2["warp_A_sum"],
            "cam2_warp_B_sum": result_cam2["warp_B_sum"],
            # CoC
            "coc_sum": coc_sum,
            # Metadata (from cam1, should match cam2)
            "n_images": N,
            "n_win_x": result_cam1["n_win_x"],
            "n_win_y": result_cam1["n_win_y"],
            "smoothed_predictor": result_cam1["smoothed_predictor"],
            "vector_mask": result_cam1["vector_mask"],
            "n_pre": result_cam1["n_pre"],
            "n_post": result_cam1["n_post"],
        }

        # Include first-pair dewarped images for diagnostics
        if is_first_batch and N > 0:
            combined["first_pair_cam1_A"] = dw_cam1[0, 0]
            combined["first_pair_cam1_B"] = dw_cam1[0, 1]
            combined["first_pair_cam2_A"] = dw_cam2[0, 0]
            combined["first_pair_cam2_B"] = dw_cam2[0, 1]

        return combined

    def _compute_coc_batch(
        self,
        dw_cam1: np.ndarray,
        dw_cam2: np.ndarray,
        config: Config,
        pass_idx: int,
        predictor_field: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute per-frame Correlation-of-Correlations.

        For each frame n:
        1. Compute single-frame AB correlation for cam1 (N=1)
        2. Compute single-frame AB correlation for cam2 (N=1)
        3. FFT cross-correlate the two correlation maps
        4. Accumulate the cross-correlation

        Returns
        -------
        np.ndarray
            CoC sum, shape (n_windows, coc_h, coc_w) — accumulated across frames
        """
        from pivtools_core.window_utils import apply_single_mode_padding

        N = dw_cam1.shape[0]
        n_win_y = len(self._correlator.win_ctrs_y[pass_idx])
        n_win_x = len(self._correlator.win_ctrs_x[pass_idx])
        total_windows = n_win_y * n_win_x
        corr_size = self._correlator.window_sizes_for_corr[pass_idx]
        corr_h, corr_w = corr_size

        # CoC output size (full cross-correlation of two corr_size maps)
        coc_h = 2 * corr_h - 1
        coc_w = 2 * corr_w - 1

        coc_sum = np.zeros((total_windows, coc_h, coc_w), dtype=np.float32)

        # Temporary buffers for single-frame correlation
        plane_size = total_windows * corr_h * corr_w
        temp_cam1_AB = np.zeros(plane_size, dtype=np.float32)
        temp_cam2_AB = np.zeros(plane_size, dtype=np.float32)

        # Get mask
        vector_mask = (
            self._correlator.vector_masks[pass_idx]
            if self._correlator.vector_masks and pass_idx < len(self._correlator.vector_masks)
            else None
        )
        if vector_mask is not None:
            b_mask = np.ascontiguousarray(vector_mask.ravel(order='C').astype(np.float32))
        else:
            b_mask = np.zeros(total_windows, dtype=np.float32)

        # Handle predictor warping for pass > 0
        if pass_idx > 0 and predictor_field is not None:
            im_mesh_A, im_mesh_B, _ = self._correlator._get_im_mesh(pass_idx, predictor_field)
        else:
            im_mesh_A = im_mesh_B = None

        win_size = config.stereo_ensemble_window_sizes[pass_idx]
        runtype = config.stereo_ensemble_type[pass_idx]
        is_single_mode = (runtype == 'single')

        for n in range(N):
            # Extract single frame pair from each camera
            frame_cam1_a = dw_cam1[n, 0].astype(np.float32, copy=False)
            frame_cam1_b = dw_cam1[n, 1].astype(np.float32, copy=False)
            frame_cam2_a = dw_cam2[n, 0].astype(np.float32, copy=False)
            frame_cam2_b = dw_cam2[n, 1].astype(np.float32, copy=False)

            # Apply predictor warping if available
            if im_mesh_A is not None:
                frame_cam1_a = cv2.remap(
                    frame_cam1_a, im_mesh_A[..., 0], im_mesh_A[..., 1],
                    interpolation=cv2.INTER_LINEAR
                )
                frame_cam1_b = cv2.remap(
                    frame_cam1_b, im_mesh_B[..., 0], im_mesh_B[..., 1],
                    interpolation=cv2.INTER_LINEAR
                )
                frame_cam2_a = cv2.remap(
                    frame_cam2_a, im_mesh_A[..., 0], im_mesh_A[..., 1],
                    interpolation=cv2.INTER_LINEAR
                )
                frame_cam2_b = cv2.remap(
                    frame_cam2_b, im_mesh_B[..., 0], im_mesh_B[..., 1],
                    interpolation=cv2.INTER_LINEAR
                )

            # Apply single mode padding if needed
            if is_single_mode:
                sum_window = tuple(config.stereo_ensemble_sum_window)
                frame_cam1_a, _ = apply_single_mode_padding(
                    frame_cam1_a[np.newaxis], win_size, sum_window, pad_value=0.0
                )
                frame_cam1_a = frame_cam1_a[0]
                frame_cam1_b, _ = apply_single_mode_padding(
                    frame_cam1_b[np.newaxis], win_size, sum_window, pad_value=0.0
                )
                frame_cam1_b = frame_cam1_b[0]
                frame_cam2_a, _ = apply_single_mode_padding(
                    frame_cam2_a[np.newaxis], win_size, sum_window, pad_value=0.0
                )
                frame_cam2_a = frame_cam2_a[0]
                frame_cam2_b, _ = apply_single_mode_padding(
                    frame_cam2_b[np.newaxis], win_size, sum_window, pad_value=0.0
                )
                frame_cam2_b = frame_cam2_b[0]

            H_proc, W_proc = frame_cam1_a.shape
            image_size = np.array([H_proc, W_proc], dtype=np.int32)
            n_windows = np.array([n_win_y, n_win_x], dtype=np.int32)
            comp_size = self._correlator.window_sizes_for_computation[pass_idx]
            out_size = self._correlator.window_sizes_for_corr[pass_idx]
            win_size_arr = np.array([comp_size[0], comp_size[1]], dtype=np.int32)
            fit_size_arr = np.array([out_size[0], out_size[1]], dtype=np.int32)

            # Clear temp buffers
            temp_cam1_AB.fill(0)
            temp_cam2_AB.fill(0)

            # Single-frame cam1 AB correlation (N=1)
            self._correlator.lib.bulkxcorr2d_accumulate(
                np.ascontiguousarray(frame_cam1_a[np.newaxis], dtype=np.float32),
                np.ascontiguousarray(frame_cam1_b[np.newaxis], dtype=np.float32),
                b_mask,
                image_size, 1,
                self._correlator.win_ctrs_x[pass_idx].astype(np.float32),
                self._correlator.win_ctrs_y[pass_idx].astype(np.float32),
                n_windows,
                self._correlator.win_weights_A[pass_idx],
                self._correlator.win_weights_B[pass_idx],
                win_size_arr, fit_size_arr,
                temp_cam1_AB,
            )

            # Single-frame cam2 AB correlation (N=1)
            self._correlator.lib.bulkxcorr2d_accumulate(
                np.ascontiguousarray(frame_cam2_a[np.newaxis], dtype=np.float32),
                np.ascontiguousarray(frame_cam2_b[np.newaxis], dtype=np.float32),
                b_mask,
                image_size, 1,
                self._correlator.win_ctrs_x[pass_idx].astype(np.float32),
                self._correlator.win_ctrs_y[pass_idx].astype(np.float32),
                n_windows,
                self._correlator.win_weights_A[pass_idx],
                self._correlator.win_weights_B[pass_idx],
                win_size_arr, fit_size_arr,
                temp_cam2_AB,
            )

            # Reshape to per-window maps
            R11_frame = temp_cam1_AB.reshape(total_windows, corr_h, corr_w)
            R22_frame = temp_cam2_AB.reshape(total_windows, corr_h, corr_w)

            # Per-window FFT cross-correlate: C[w] = ifft2(fft2(R11[w]) * conj(fft2(R22[w])))
            # Use scipy.signal.fftconvolve for correct full correlation
            for w in range(total_windows):
                if vector_mask is not None and vector_mask.ravel()[w]:
                    continue  # Skip masked windows
                F1 = np.fft.fft2(R11_frame[w], s=(coc_h, coc_w))
                F2 = np.fft.fft2(R22_frame[w], s=(coc_h, coc_w))
                C = np.real(np.fft.ifft2(F1 * np.conj(F2)))
                coc_sum[w] += np.fft.fftshift(C).astype(np.float32)

        logging.debug(
            f"CoC batch: N={N}, total_windows={total_windows}, "
            f"coc_shape=({coc_h}, {coc_w})"
        )
        return coc_sum
