"""CPU backend for Stereo Ensemble PIV using Correlation-of-Correlations.

Composes EnsembleCorrelatorCPU for shared infrastructure (taper generation,
padding, predictor handling, libfusedwarp) and adds stereo-specific logic:
- Dewarping via cv2.remap with precomputed calibration maps
- Dual-camera C correlation + CoC accumulation via libstereo_coc

The CoC k-space spread fit (Σ_diff via the AC F_ref path) lives Python-side
in StereoEnsembleAccumulator._fit_coc_kspace_ac.
"""

from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from pivtools_core.config import Config


# ctypes convenience
c_float_p = ctypes.POINTER(ctypes.c_float)
c_int_p = ctypes.POINTER(ctypes.c_int)


class StereoEnsembleCorrelatorCPU:
    """Stereo ensemble PIV correlator using Correlation-of-Correlations.

    Composes EnsembleCorrelatorCPU for:
    - Taper weight generation (per-pass, asymmetric/symmetric)
    - Window center computation and caching
    - Predictor field construction (_get_im_mesh)
    - Fused image warping (_fused_warp_batch via libfusedwarp)
    - Correlation buffer pre-allocation

    Adds stereo-specific:
    - Dewarp maps (precomputed from calibration + self-cal)
    - bulkxcorr2d_stereo_coc_accumulate (dual-camera + CoC in one C call)
    """

    # Class-level library cache (loaded once, shared across instances)
    _lib_stereo_coc = None

    def __init__(
        self,
        config: Config,
        precomputed_cache: Optional[dict] = None,
        vector_masks: Optional[List[np.ndarray]] = None,
        active_pass_idx: Optional[int] = None,
        dewarp_maps: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
        mm_per_pixel: Optional[float] = None,
        stereo_angle: Optional[float] = None,
        dewarped_image_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initialize stereo ensemble correlator.

        Parameters
        ----------
        config : Config
            PIVTOOLs configuration.
        precomputed_cache : dict, optional
            Cache from get_cache_data() for worker scattering.
        vector_masks : list of ndarray, optional
            Per-pass vector masks.
        active_pass_idx : int, optional
            Allocate buffers only for this pass.
        dewarp_maps : dict, optional
            {cam_num: (map_x, map_y)} precomputed dewarp remap tables.
        mm_per_pixel : float, optional
            Dewarped pixel scale (mm per pixel).
        stereo_angle : float, optional
            tan(half-angle) between camera optical axes — Frame-B
            (dewarped) trig factor used downstream by Σ → R conversions
            in StereoEnsembleAccumulator. (Historical: this was sin(α);
            changed for the geometry-correct dewarped formula. See
            pivtools_core/stereo_ensemble.py::_compute_stereo_angle.)
        dewarped_image_shape : tuple, optional
            (H, W) of dewarped images. Needed if dewarp_maps not provided
            but config is used to derive shape.
        """
        self.config = config
        self.dewarp_maps = dewarp_maps or {}
        self.mm_per_pixel = mm_per_pixel
        self.stereo_angle = stereo_angle

        # Resolve the dewarped image shape — the inner correlator's window grid
        # MUST be built against this, not against the raw camera sensor. If not
        # given explicitly, derive it from the dewarp remap tables (map_x/map_y
        # have shape = dewarped image shape by construction in
        # compute_dewarp_maps). Without this override, EnsembleCorrelatorCPU
        # reads config.image_shape (the raw sensor shape) and creates window
        # centres that extend past the dewarped image — up to 26% of windows
        # end up outside the image bounds, producing dead-flat correlations
        # and LOW_SNR fitter failures. See cpu_ensemble.py:183 for where the
        # inner correlator reads config.image_shape.
        if dewarped_image_shape is None and self.dewarp_maps:
            any_maps = next(iter(self.dewarp_maps.values()))
            dewarped_image_shape = tuple(any_maps[0].shape)
        self.dewarped_image_shape = dewarped_image_shape

        # Create inner ensemble correlator — this gives us taper weights,
        # window centers, padding, predictor handling, and libfusedwarp.
        # Temporarily pin config._detected_image_shape to the dewarped shape so
        # the inner grid is sized for the dewarped image, then restore. Scoped
        # exactly to the inner construction — no other config consumer sees it.
        from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

        _prev_shape = getattr(config, "_detected_image_shape", None)
        try:
            if self.dewarped_image_shape is not None:
                config._detected_image_shape = tuple(self.dewarped_image_shape)
            self._inner = EnsembleCorrelatorCPU(
                config,
                precomputed_cache=precomputed_cache,
                vector_masks=vector_masks,
                active_pass_idx=active_pass_idx,
            )
        finally:
            config._detected_image_shape = _prev_shape

        # Set fused-warp kernel up front. The inner's _get_im_mesh sets this
        # too, but only runs for pass_idx > 0 — so pass 0 would fall back to
        # `hasattr(_inner, _fused_interp_mode) else 0` (cubic) at
        # _fused_dewarp_warp_batch:334. Pre-seed here so pass 0 honours the
        # stereo override.
        stereo_image_interp = config.stereo_ensemble_image_warp_interpolation
        self._inner._fused_interp_mode = 0 if stereo_image_interp == 'cubic' else 1

        # Load stereo-specific C libraries
        self._load_stereo_libraries()
        self.lib_stereo_coc = StereoEnsembleCorrelatorCPU._lib_stereo_coc
        self._setup_stereo_ctypes()

        # CoC uses same size as per-camera correlation output (no zero-padding)
        self._coc_window_sizes = {}
        n_passes = config.stereo_ensemble_num_passes
        for p in range(n_passes):
            if active_pass_idx is not None and p != active_pass_idx:
                continue
            corr_size = self._inner.window_sizes_for_corr[p]
            self._coc_window_sizes[p] = list(corr_size)

        # Pre-allocate stereo correlation buffers (7 per pass)
        self._stereo_buffers = {}
        for p in range(n_passes):
            if active_pass_idx is not None and p != active_pass_idx:
                continue
            self._allocate_stereo_buffers(p)

        # Thread pool for parallel dewarping
        omp_threads = max(1, int(config.omp_threads))
        self._dewarp_pool = ThreadPoolExecutor(max_workers=omp_threads)
        cv2.setNumThreads(1)  # Prevent OpenCV internal threading contention

    @classmethod
    def _load_stereo_libraries(cls):
        """Load libstereo_coc (cached at class level)."""
        if cls._lib_stereo_coc is not None:
            return

        lib_ext = ".dll" if os.name == "nt" else ".so"

        # Search paths: installed package, then development directory
        search_dirs = [
            Path(__file__).parent.parent.parent / "lib",  # installed
            Path.cwd() / "pivtools_cli" / "lib",          # development
        ]

        lib_name = f"libstereo_coc{lib_ext}"
        for search_dir in search_dirs:
            lib_path = search_dir / lib_name
            if lib_path.exists():
                cls._lib_stereo_coc = ctypes.CDLL(str(lib_path))
                logger.debug(f"Loaded {lib_name} from {lib_path}")
                return
        raise RuntimeError(
            f"Could not find {lib_name}. Run 'python setup.py build' first."
        )

    def _setup_stereo_ctypes(self):
        """Configure ctypes bindings for stereo C functions."""
        # bulkxcorr2d_stereo_coc_accumulate
        self.lib_stereo_coc.bulkxcorr2d_stereo_coc_accumulate.restype = ctypes.c_ubyte
        self.lib_stereo_coc.bulkxcorr2d_stereo_coc_accumulate.argtypes = [
            c_float_p, c_float_p,  # cam1 images A, B
            c_float_p, c_float_p,  # cam2 images A, B
            c_float_p,             # mask
            c_int_p,               # nImageSize
            ctypes.c_int,          # N_images
            c_float_p, c_float_p,  # win_ctrs X, Y
            c_int_p,               # nWindows
            c_float_p, c_float_p,  # AB weights A, B
            c_float_p, c_float_p,  # auto weights A, B
            c_int_p,               # nWindowSize
            c_int_p,               # nFitWindowSize
            c_float_p, c_float_p, c_float_p,  # cam1 AB, AA, BB output
            c_float_p, c_float_p, c_float_p,  # cam2 AB, AA, BB output
            c_float_p,             # CoC output
            c_float_p,             # cam1 autocorr(AB1) accumulated output
            c_float_p,             # cam2 autocorr(AB2) accumulated output
            c_float_p,             # CoC_AA = xcorr(cam1_AA(f), cam2_AA(f)) accumulated
            ctypes.c_int,          # nStorePlanes gate for CoC_A/CoC_B (0/1)
            c_float_p,             # CoC_A = xcorr(I1A_c, I2A_c) accumulated (store-only)
            c_float_p,             # CoC_B = xcorr(I1B_c, I2B_c) accumulated (store-only)
            ctypes.c_int,          # diag_window_idx (-1 = disabled)
            c_float_p,             # diag AB1 per-frame (or NULL)
            c_float_p,             # diag AB2 per-frame (or NULL)
            c_float_p,             # diag CoC per-frame (or NULL)
        ]

    def _allocate_stereo_buffers(self, pass_idx: int):
        """Pre-allocate correlation buffers for one pass."""
        corr_size = self._inner.window_sizes_for_corr[pass_idx]
        n_win_y = len(self._inner.win_ctrs_y[pass_idx])
        n_win_x = len(self._inner.win_ctrs_x[pass_idx])
        n_windows = n_win_y * n_win_x
        n_px_corr = corr_size[0] * corr_size[1]

        coc_size = self._coc_window_sizes[pass_idx]
        n_px_coc = coc_size[0] * coc_size[1]

        self._stereo_buffers[pass_idx] = {
            "cam1_AB": np.zeros(n_windows * n_px_corr, dtype=np.float32),
            "cam1_AA": np.zeros(n_windows * n_px_corr, dtype=np.float32),
            "cam1_BB": np.zeros(n_windows * n_px_corr, dtype=np.float32),
            "cam2_AB": np.zeros(n_windows * n_px_corr, dtype=np.float32),
            "cam2_AA": np.zeros(n_windows * n_px_corr, dtype=np.float32),
            "cam2_BB": np.zeros(n_windows * n_px_corr, dtype=np.float32),
            "CoC": np.zeros(n_windows * n_px_coc, dtype=np.float32),
            # Per-frame autocorr(AB_c) ensemble — diagnostic only, sized to CoC
            "cam1_AB_AC": np.zeros(n_windows * n_px_coc, dtype=np.float32),
            "cam2_AB_AC": np.zeros(n_windows * n_px_coc, dtype=np.float32),
            # Per-frame xcorr(cam1_AA(f), cam2_AA(f)) ensemble — leak-only
            # reference plane for the matched-pair correction.
            # Wiki: 2026-05-10-coc-matched-pair-leak.md §E.
            "CoC_AA": np.zeros(n_windows * n_px_coc, dtype=np.float32),
            # Store-only diagnostics: cross-camera xcorr of the raw
            # mean-subtracted sub-images at instant A / B (static-disparity
            # correlation). Computed in C only when store_planes is on;
            # never consumed by any fit/velocity/stress path.
            "CoC_A": np.zeros(n_windows * n_px_coc, dtype=np.float32),
            "CoC_B": np.zeros(n_windows * n_px_coc, dtype=np.float32),
        }

    def dewarp_batch(
        self,
        images: np.ndarray,
        camera_num: int,
    ) -> np.ndarray:
        """Dewarp a batch of images using precomputed maps.

        Parameters
        ----------
        images : ndarray, shape (N, 2, H_raw, W_raw)
            Raw image pairs.
        camera_num : int
            Camera number (key into self.dewarp_maps).

        Returns
        -------
        ndarray, shape (N, 2, H_dw, W_dw)
            Dewarped image pairs.
        """
        map_x, map_y = self.dewarp_maps[camera_num]
        N = images.shape[0]
        out_h, out_w = map_x.shape

        # Flatten to list of individual images for parallel processing
        flat_images = []
        for i in range(N):
            flat_images.append(images[i, 0])  # frame A
            flat_images.append(images[i, 1])  # frame B

        def _dewarp_single(img):
            return cv2.remap(
                img.astype(np.float32),
                map_x, map_y,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        results = list(self._dewarp_pool.map(_dewarp_single, flat_images))

        # Reshape back to (N, 2, H_dw, W_dw)
        output = np.empty((N, 2, out_h, out_w), dtype=np.float32)
        for i in range(N):
            output[i, 0] = results[2 * i]
            output[i, 1] = results[2 * i + 1]
        return output

    def _fused_dewarp_warp_batch(
        self,
        raw_imgs_a: np.ndarray,
        raw_imgs_b: np.ndarray,
        camera_num: int,
        pass_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combined dewarp + predictor warp in a single interpolation pass.

        Composes the dewarp remap tables with the predictor displacement,
        sampling the raw camera image exactly once via bicubic or Lanczos-3.

        Parameters
        ----------
        raw_imgs_a, raw_imgs_b : ndarray, shape (N, H_raw, W_raw)
            Raw camera images (frame A and B).
        camera_num : int
            Camera number (key into self.dewarp_maps).
        pass_idx : int
            Current pass index (0-based).

        Returns
        -------
        out_a, out_b : ndarray, shape (N, H_dw, W_dw)
            Dewarped + predictor-warped images.
        """
        map_x, map_y = self.dewarp_maps[camera_num]
        N, H_raw, W_raw = raw_imgs_a.shape
        H_dw, W_dw = map_x.shape

        out_a = np.zeros((N, H_dw, W_dw), dtype=np.float32)
        out_b = np.zeros((N, H_dw, W_dw), dtype=np.float32)

        # Get predictor (zero if pass 0)
        if pass_idx > 0 and hasattr(self._inner, "delta_ab_old") and self._inner.delta_ab_old is not None:
            pred_dy = np.ascontiguousarray(self._inner.delta_ab_old[..., 0], dtype=np.float32)
            pred_dx = np.ascontiguousarray(self._inner.delta_ab_old[..., 1], dtype=np.float32)
            nPY, nPX = pred_dy.shape
            ctrs_y = self._inner._fused_ctrs_y
            ctrs_x = self._inner._fused_ctrs_x
        else:
            # Pass 0: zero predictor, 1x1 grid at image centre
            pred_dy = np.zeros((1, 1), dtype=np.float32)
            pred_dx = np.zeros((1, 1), dtype=np.float32)
            nPY, nPX = 1, 1
            ctrs_y = np.array([H_dw / 2.0], dtype=np.float32)
            ctrs_x = np.array([W_dw / 2.0], dtype=np.float32)

        interp_mode = self._inner._fused_interp_mode if hasattr(self._inner, "_fused_interp_mode") else 0

        # One-shot diagnostic: log the kernel actually fed to fused_warp.c per
        # (pass, camera). Confirms the stereo_ensemble_piv.image_warp_interpolation
        # override is honoured end-to-end. Cheap (set-membership check, info log).
        if not hasattr(self, "_logged_interp_mode"):
            self._logged_interp_mode = set()
        log_key = (pass_idx, camera_num)
        if log_key not in self._logged_interp_mode:
            logger.info(
                f"  fused_warp: pass {pass_idx} cam{camera_num} interp_mode={interp_mode} "
                f"({'cubic' if interp_mode == 0 else 'lanczos3'})"
            )
            self._logged_interp_mode.add(log_key)

        dw_map_x = np.ascontiguousarray(map_x, dtype=np.float32)
        dw_map_y = np.ascontiguousarray(map_y, dtype=np.float32)

        ret = self._inner._lib_fw.fused_symmetric_warp_with_dewarp_batch(
            np.ascontiguousarray(raw_imgs_a, dtype=np.float32),
            np.ascontiguousarray(raw_imgs_b, dtype=np.float32),
            out_a, out_b,
            dw_map_x, dw_map_y,
            pred_dy, pred_dx,
            N, H_raw, W_raw, H_dw, W_dw,
            nPY, nPX,
            ctrs_y, ctrs_x,
            interp_mode,
            1,  # shared_predictor=1 for ensemble
        )
        if ret != 0:
            raise RuntimeError(f"fused_symmetric_warp_with_dewarp_batch failed (ret={ret})")

        return out_a, out_b

    def correlate_batch_stereo(
        self,
        images_cam1: np.ndarray,
        images_cam2: np.ndarray,
        config: Config,
        pass_idx: int,
        predictor_field: Optional[np.ndarray] = None,
        is_first_batch: bool = False,
        clear_buffers: bool = True,
        copy_result: bool = True,
    ) -> dict:
        """Full stereo correlation: dewarp -> predictor warp -> C accumulate.

        Parameters
        ----------
        images_cam1, images_cam2 : ndarray, shape (N, 2, H_raw, W_raw)
            Raw image pairs from both cameras.
        config : Config
            Configuration.
        pass_idx : int
            Current pass index (0-based).
        predictor_field : ndarray, optional
            Predictor displacement from previous pass, shape (n_win_y, n_win_x, 2).
        is_first_batch : bool
            True if this is the first batch in the sliding window.
        clear_buffers : bool
            Zero correlation buffers before accumulating.
        copy_result : bool
            If True, copy buffers to output dict. If False, return lightweight dict.

        Returns
        -------
        dict with keys: cam1_AB/AA/BB_sum, cam2_AB/AA/BB_sum, CoC_sum,
             warp_1A/1B/2A/2B_sum, n_images, n_win_x, n_win_y,
             smoothed_predictor, vector_mask
        """
        cam_pair = config.stereo_ensemble_camera_pair
        N = images_cam1.shape[0]

        # 1. Split raw images into A and B stacks: (N, H_raw, W_raw)
        raw_cam1_A = np.ascontiguousarray(images_cam1[:, 0], dtype=np.float32)
        raw_cam1_B = np.ascontiguousarray(images_cam1[:, 1], dtype=np.float32)
        raw_cam2_A = np.ascontiguousarray(images_cam2[:, 0], dtype=np.float32)
        raw_cam2_B = np.ascontiguousarray(images_cam2[:, 1], dtype=np.float32)

        # 2. Build predictor displacement field (pass > 0)
        if pass_idx > 0 and predictor_field is not None:
            self._inner._get_im_mesh(pass_idx, predictor_field)

        # 3. Fused dewarp + predictor warp (single interpolation pass)
        cam1_A, cam1_B = self._fused_dewarp_warp_batch(raw_cam1_A, raw_cam1_B, cam_pair[0], pass_idx)
        cam2_A, cam2_B = self._fused_dewarp_warp_batch(raw_cam2_A, raw_cam2_B, cam_pair[1], pass_idx)

        # 4. Compute warp sums (for background subtraction)
        warp_1A = cam1_A.sum(axis=0)
        warp_1B = cam1_B.sum(axis=0)
        warp_2A = cam2_A.sum(axis=0)
        warp_2B = cam2_B.sum(axis=0)

        # 5. Single-mode padding if needed
        from pivtools_core.window_utils import apply_single_mode_padding

        pass_type = config.stereo_ensemble_type[pass_idx] if pass_idx < len(config.stereo_ensemble_type) else "std"
        if pass_type == "single":
            win_size = config.stereo_ensemble_window_sizes[pass_idx]
            sum_window = config.stereo_ensemble_sum_window
            cam1_A, _ = apply_single_mode_padding(cam1_A, win_size, sum_window, pad_value=0)
            cam1_B, _ = apply_single_mode_padding(cam1_B, win_size, sum_window, pad_value=0)
            cam2_A, _ = apply_single_mode_padding(cam2_A, win_size, sum_window, pad_value=0)
            cam2_B, _ = apply_single_mode_padding(cam2_B, win_size, sum_window, pad_value=0)

        # 5. Clear buffers if needed
        if clear_buffers:
            for key in self._stereo_buffers[pass_idx]:
                self._stereo_buffers[pass_idx][key].fill(0)

        # 6. Call C library: fused dual-camera + CoC accumulation
        H, W = cam1_A.shape[1], cam1_A.shape[2]
        image_size = np.array([H, W], dtype=np.int32)

        padding = self._inner.padding_per_pass[pass_idx]
        pad_top, pad_left = padding[0], padding[2]
        win_ctrs_x = (self._inner.win_ctrs_x[pass_idx] + pad_left).astype(np.float32)
        win_ctrs_y = (self._inner.win_ctrs_y[pass_idx] + pad_top).astype(np.float32)

        n_win_y = len(self._inner.win_ctrs_y[pass_idx])
        n_win_x = len(self._inner.win_ctrs_x[pass_idx])
        n_windows = np.array([n_win_y, n_win_x], dtype=np.int32)

        # Build mask
        if self._inner.vector_masks and pass_idx < len(self._inner.vector_masks):
            mask = self._inner.vector_masks[pass_idx].ravel().astype(np.float32)
        else:
            mask = np.zeros(n_win_y * n_win_x, dtype=np.float32)

        # Window sizes for computation and output
        comp_size = self._inner.window_sizes_for_computation[pass_idx]
        win_size_arr = np.array(comp_size, dtype=np.int32)

        corr_size = self._inner.window_sizes_for_corr[pass_idx]
        fit_size_arr = np.array(corr_size, dtype=np.int32)

        # Taper weights
        weight_a_ab = np.ascontiguousarray(self._inner.win_weights_A[pass_idx], dtype=np.float32)
        weight_b_ab = np.ascontiguousarray(self._inner.win_weights_B[pass_idx], dtype=np.float32)
        # Symmetric auto weights (same weight for both A and B)
        auto_weight_a = np.ascontiguousarray(self._inner.win_weights_B[pass_idx], dtype=np.float32)
        auto_weight_b = np.ascontiguousarray(self._inner.win_weights_B[pass_idx], dtype=np.float32)

        buffers = self._stereo_buffers[pass_idx]

        # Diagnostic: store per-frame planes for the central window
        n_px_out = corr_size[0] * corr_size[1]
        diag_win_idx = (n_win_y // 2) * n_win_x + (n_win_x // 2)
        diag_ab1 = np.zeros(N * n_px_out, dtype=np.float32)
        diag_ab2 = np.zeros(N * n_px_out, dtype=np.float32)
        diag_coc = np.zeros(N * n_px_out, dtype=np.float32)

        error_code = self.lib_stereo_coc.bulkxcorr2d_stereo_coc_accumulate(
            cam1_A.ctypes.data_as(c_float_p),
            cam1_B.ctypes.data_as(c_float_p),
            cam2_A.ctypes.data_as(c_float_p),
            cam2_B.ctypes.data_as(c_float_p),
            mask.ctypes.data_as(c_float_p),
            image_size.ctypes.data_as(c_int_p),
            N,
            win_ctrs_x.ctypes.data_as(c_float_p),
            win_ctrs_y.ctypes.data_as(c_float_p),
            n_windows.ctypes.data_as(c_int_p),
            weight_a_ab.ctypes.data_as(c_float_p),
            weight_b_ab.ctypes.data_as(c_float_p),
            auto_weight_a.ctypes.data_as(c_float_p),
            auto_weight_b.ctypes.data_as(c_float_p),
            win_size_arr.ctypes.data_as(c_int_p),
            fit_size_arr.ctypes.data_as(c_int_p),
            buffers["cam1_AB"].ctypes.data_as(c_float_p),
            buffers["cam1_AA"].ctypes.data_as(c_float_p),
            buffers["cam1_BB"].ctypes.data_as(c_float_p),
            buffers["cam2_AB"].ctypes.data_as(c_float_p),
            buffers["cam2_AA"].ctypes.data_as(c_float_p),
            buffers["cam2_BB"].ctypes.data_as(c_float_p),
            buffers["CoC"].ctypes.data_as(c_float_p),
            buffers["cam1_AB_AC"].ctypes.data_as(c_float_p),
            buffers["cam2_AB_AC"].ctypes.data_as(c_float_p),
            buffers["CoC_AA"].ctypes.data_as(c_float_p),
            ctypes.c_int(
                1 if getattr(config, "stereo_ensemble_store_planes", False) else 0
            ),
            buffers["CoC_A"].ctypes.data_as(c_float_p),
            buffers["CoC_B"].ctypes.data_as(c_float_p),
            diag_win_idx,
            diag_ab1.ctypes.data_as(c_float_p),
            diag_ab2.ctypes.data_as(c_float_p),
            diag_coc.ctypes.data_as(c_float_p),
        )

        if error_code != 0:
            logger.warning(f"stereo_coc_accumulate returned error code {error_code}")

        # 7. Build result dict
        # Two predictor fields exposed for diagnostics + save:
        #   smoothed_predictor : delta_ab_old — pre-remap, on the previous
        #     pass's grid + boundary padding. Existing stereo accumulator
        #     logic depends on this shape; do not change semantics.
        #   upsampled_predictor : delta_ab_pred — post-remap, on this
        #     pass's grid. This is what the std ensemble saves under
        #     ``pred_x/y``. Added 2026-04-26 to expose the actual upsample
        #     output (before this, stereo only persisted the pre-remap
        #     field, mislabelled as ``pred_x/y``).
        result = {
            "warp_1A_sum": warp_1A,
            "warp_1B_sum": warp_1B,
            "warp_2A_sum": warp_2A,
            "warp_2B_sum": warp_2B,
            "n_images": N,
            "n_win_x": n_win_x,
            "n_win_y": n_win_y,
            "smoothed_predictor": getattr(self._inner, "delta_ab_old", None),
            "upsampled_predictor": getattr(self._inner, "delta_ab_pred", None),
            "vector_mask": mask if mask.any() else None,
        }

        # Save first dewarped+warped frame per camera (for diagnostics)
        # With fused dewarp+warp, cam1_A/B ARE the dewarped+warped output.
        if is_first_batch:
            result["diag_dw_cam1_A"] = cam1_A[0].copy()
            result["diag_dw_cam1_B"] = cam1_B[0].copy()
            result["diag_dw_cam2_A"] = cam2_A[0].copy()
            result["diag_dw_cam2_B"] = cam2_B[0].copy()

        # Per-frame diagnostic planes for central window
        corr_h, corr_w = corr_size
        result["diag_perframe_ab1"] = diag_ab1.reshape(N, corr_h, corr_w)
        result["diag_perframe_ab2"] = diag_ab2.reshape(N, corr_h, corr_w)
        result["diag_perframe_coc"] = diag_coc.reshape(N, corr_h, corr_w)
        result["diag_perframe_win_idx"] = diag_win_idx

        # Per-frame raw sub-images (post-dewarp, post-warp) at the diag window
        # for offline experimentation in `manual_tools/coc_window_experiments.py`.
        # These are EXACTLY the inputs the C kernel cross-correlated above.
        diag_jj_row = diag_win_idx // n_win_x
        diag_ii_col = diag_win_idx % n_win_x
        cy_diag = float(win_ctrs_y[diag_jj_row])
        cx_diag = float(win_ctrs_x[diag_ii_col])
        ws_h, ws_w = comp_size  # computation window size (h, w)
        diag_row_min = int(np.floor(cy_diag - (ws_h - 1) / 2.0 + 0.5))
        diag_col_min = int(np.floor(cx_diag - (ws_w - 1) / 2.0 + 0.5))
        if (
            0 <= diag_row_min and diag_row_min + ws_h <= cam1_A.shape[1]
            and 0 <= diag_col_min and diag_col_min + ws_w <= cam1_A.shape[2]
        ):
            r0, r1 = diag_row_min, diag_row_min + ws_h
            c0, c1 = diag_col_min, diag_col_min + ws_w
            result["diag_perframe_subimg_cam1_A"] = cam1_A[:, r0:r1, c0:c1].copy()
            result["diag_perframe_subimg_cam1_B"] = cam1_B[:, r0:r1, c0:c1].copy()
            result["diag_perframe_subimg_cam2_A"] = cam2_A[:, r0:r1, c0:c1].copy()
            result["diag_perframe_subimg_cam2_B"] = cam2_B[:, r0:r1, c0:c1].copy()
            result["diag_perframe_subimg_window_size"] = np.array(comp_size, dtype=np.int32)
            result["diag_perframe_subimg_topleft"] = np.array(
                [diag_row_min, diag_col_min], dtype=np.int32)
        else:
            logger.warning(
                f"Diagnostic sub-image extraction OOB: "
                f"row [{diag_row_min}, {diag_row_min + ws_h}) col "
                f"[{diag_col_min}, {diag_col_min + ws_w}) vs image "
                f"shape {cam1_A.shape[1:]}. Skipping sub-image dump."
            )

        if copy_result:
            result.update(self.get_accumulated_correlation_stereo(pass_idx))

        return result

    def get_accumulated_correlation_stereo(self, pass_idx: int) -> dict:
        """Copy accumulated correlation buffers (call ONCE after all batches)."""
        buffers = self._stereo_buffers[pass_idx]
        return {
            "cam1_AB_sum": buffers["cam1_AB"].copy(),
            "cam1_AA_sum": buffers["cam1_AA"].copy(),
            "cam1_BB_sum": buffers["cam1_BB"].copy(),
            "cam2_AB_sum": buffers["cam2_AB"].copy(),
            "cam2_AA_sum": buffers["cam2_AA"].copy(),
            "cam2_BB_sum": buffers["cam2_BB"].copy(),
            "CoC_sum": buffers["CoC"].copy(),
            "cam1_AB_AC_sum": buffers["cam1_AB_AC"].copy(),
            "cam2_AB_AC_sum": buffers["cam2_AB_AC"].copy(),
            "CoC_AA_sum": buffers["CoC_AA"].copy(),
            "CoC_A_sum": buffers["CoC_A"].copy(),
            "CoC_B_sum": buffers["CoC_B"].copy(),
        }

    # Delegate to inner correlator for shared infrastructure
    @property
    def win_ctrs_x(self):
        return self._inner.win_ctrs_x

    @property
    def win_ctrs_y(self):
        return self._inner.win_ctrs_y

    @property
    def window_sizes_for_corr(self):
        return self._inner.window_sizes_for_corr

    @property
    def padding_per_pass(self):
        return self._inner.padding_per_pass

    def get_cache_data(self):
        """Get serializable cache data for Dask worker scattering."""
        return self._inner.get_cache_data()
