"""
Stereo Ensemble Accumulator for Correlation-of-Correlations Pipeline

Manages dual-camera + CoC buffer accumulation across Dask workers,
then performs 3D velocity reconstruction and 6-component Reynolds stress
extraction in finalize_pass().

Follows the same accumulate → finalize → clear pattern as SinglePassAccumulator,
but with dual-camera correlation buffers and a CoC cross-correlation buffer.
"""

import gc
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares
from dask.distributed import Client

from pivtools_core.config import Config
from pivtools_cli.piv.stereo_ensemble_result import (
    PIVStereoEnsemblePassResult,
    PIVStereoEnsembleResult,
)
from pivtools_cli.piv.piv_backend.gaussian_fitting import _get_sigma_from_previous_pass
from pivtools_cli.piv.piv_backend.outlier_detection import apply_outlier_detection
from pivtools_cli.piv.piv_backend.infilling import apply_infilling


class StereoEnsembleAccumulator:
    """
    Accumulates dual-camera correlation planes and CoC cross-correlations
    for stereo ensemble PIV.

    Per-pass buffers:
    - cam1/cam2: sum_corr_AA, sum_corr_BB, sum_corr_AB, sum_warp_A, sum_warp_B
    - coc_sum: accumulated per-frame cross-camera correlation

    finalize_pass() performs:
    1. Per-camera Gaussian fitting → displacements + spreads
    2. Background subtraction per camera
    3. CoC Gaussian fitting → cross-camera spread
    4. 3D velocity reconstruction (ux, uy, uz)
    5. All 6 Reynolds stress extraction (UU, VV, WW, UV, UW, VW)
    6. Outlier detection + infilling
    7. Predictor field for next pass
    """

    def __init__(
        self,
        config: Config,
        stereo_half_angle: float,
        mm_per_pixel: float,
        vector_masks: Optional[list] = None,
    ):
        self.config = config
        self.stereo_half_angle = stereo_half_angle
        self.mm_per_pixel = mm_per_pixel
        self.vector_masks = vector_masks if vector_masks is not None else []
        self.n_images = 0
        self.passes_data = []
        self.passes_results = []

        H, W = config.image_shape

        for pass_idx in range(config.stereo_ensemble_num_passes):
            win_size = config.stereo_ensemble_window_sizes[pass_idx]
            overlap = config.stereo_ensemble_overlaps[pass_idx]
            runtype = config.stereo_ensemble_type[pass_idx]

            if runtype == 'single':
                fit_window = getattr(config, 'stereo_ensemble_sum_fitting_window', None)
                sum_window = config.stereo_ensemble_sum_window
                corr_size = tuple(fit_window) if fit_window else tuple(sum_window)
            else:
                corr_size = tuple(win_size)

            from pivtools_core.window_utils import (
                compute_window_centers,
                compute_window_centers_single_mode,
            )

            if runtype == 'single':
                result = compute_window_centers_single_mode(
                    image_shape=(H, W),
                    window_size=tuple(win_size),
                    sum_window=tuple(config.stereo_ensemble_sum_window),
                    overlap=overlap,
                    validate=True,
                )
            else:
                result = compute_window_centers(
                    image_shape=(H, W),
                    window_size=tuple(win_size),
                    overlap=overlap,
                    validate=True,
                )

            n_win_y = result.n_win_y
            n_win_x = result.n_win_x
            plane_size = n_win_y * n_win_x * corr_size[0] * corr_size[1]

            # CoC output size (full cross-correlation of two corr_size maps)
            coc_h = 2 * corr_size[0] - 1
            coc_w = 2 * corr_size[1] - 1
            total_windows = n_win_y * n_win_x

            self.passes_data.append({
                # Camera 1 correlation sums
                "cam1_sum_corr_AA": np.zeros(plane_size, dtype=np.float32),
                "cam1_sum_corr_BB": np.zeros(plane_size, dtype=np.float32),
                "cam1_sum_corr_AB": np.zeros(plane_size, dtype=np.float32),
                "cam1_sum_warp_A": np.zeros((H, W), dtype=np.float32),
                "cam1_sum_warp_B": np.zeros((H, W), dtype=np.float32),
                # Camera 2 correlation sums
                "cam2_sum_corr_AA": np.zeros(plane_size, dtype=np.float32),
                "cam2_sum_corr_BB": np.zeros(plane_size, dtype=np.float32),
                "cam2_sum_corr_AB": np.zeros(plane_size, dtype=np.float32),
                "cam2_sum_warp_A": np.zeros((H, W), dtype=np.float32),
                "cam2_sum_warp_B": np.zeros((H, W), dtype=np.float32),
                # CoC sum
                "coc_sum": np.zeros(
                    (total_windows, coc_h, coc_w), dtype=np.float32
                ),
                # Grid info
                "n_win_x": n_win_x,
                "n_win_y": n_win_y,
                "corr_size": corr_size,
                "win_size": win_size,
                "coc_size": (coc_h, coc_w),
                # First-pair warped images (for diagnostic saving)
                "first_pair_cam1_A": None,
                "first_pair_cam1_B": None,
                "first_pair_cam2_A": None,
                "first_pair_cam2_B": None,
            })

    def load_previous_passes(
        self, stereo_result: PIVStereoEnsembleResult, n_images: int
    ) -> None:
        """
        Load previous passes from existing stereo ensemble result for resume.

        Parameters
        ----------
        stereo_result : PIVStereoEnsembleResult
            Loaded result containing completed passes
        n_images : int
            Number of images (kept for API compatibility, not used)
        """
        # NOTE: Do NOT set self.n_images here! Each pass counts its own.
        self.passes_results = list(stereo_result.passes)
        logging.info(
            f"Loaded {len(self.passes_results)} previous passes for resume "
            f"(n_images={n_images})"
        )

    def accumulate_batch(self, batch_result: dict, pass_idx: int):
        """Add batch results to running sums."""
        pd = self.passes_data[pass_idx]

        # Camera 1
        pd["cam1_sum_corr_AA"] += batch_result["cam1_corr_AA_sum"].reshape(-1)
        pd["cam1_sum_corr_BB"] += batch_result["cam1_corr_BB_sum"].reshape(-1)
        pd["cam1_sum_corr_AB"] += batch_result["cam1_corr_AB_sum"].reshape(-1)
        pd["cam1_sum_warp_A"] += batch_result["cam1_warp_A_sum"]
        pd["cam1_sum_warp_B"] += batch_result["cam1_warp_B_sum"]

        # Camera 2
        pd["cam2_sum_corr_AA"] += batch_result["cam2_corr_AA_sum"].reshape(-1)
        pd["cam2_sum_corr_BB"] += batch_result["cam2_corr_BB_sum"].reshape(-1)
        pd["cam2_sum_corr_AB"] += batch_result["cam2_corr_AB_sum"].reshape(-1)
        pd["cam2_sum_warp_A"] += batch_result["cam2_warp_A_sum"]
        pd["cam2_sum_warp_B"] += batch_result["cam2_warp_B_sum"]

        # CoC
        pd["coc_sum"] += batch_result["coc_sum"]

        # Metadata
        if batch_result.get("smoothed_predictor") is not None:
            pd["smoothed_predictor"] = batch_result["smoothed_predictor"]
        if batch_result.get("n_pre") is not None:
            pd["n_pre"] = batch_result["n_pre"]
        if batch_result.get("n_post") is not None:
            pd["n_post"] = batch_result["n_post"]

        # Store first-pair warped images (only from first batch)
        for key in ["first_pair_cam1_A", "first_pair_cam1_B",
                     "first_pair_cam2_A", "first_pair_cam2_B"]:
            if batch_result.get(key) is not None and pd.get(key) is None:
                pd[key] = batch_result[key]

        self.n_images += batch_result["n_images"]

    def finalize_pass(
        self,
        pass_idx: int,
        client: Client,
        predictor_field: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None,
    ) -> PIVStereoEnsemblePassResult:
        """
        Finalize a stereo ensemble pass: fit both cameras, compute CoC,
        reconstruct 3D velocity, extract 6 Reynolds stresses.

        Parameters
        ----------
        pass_idx : int
        client : Client
            Dask client for distributed Gaussian fitting
        predictor_field : np.ndarray, optional
            Shape (n_win_y, n_win_x, 2) where [:,:,0]=Y, [:,:,1]=X
        output_path : Path, optional

        Returns
        -------
        PIVStereoEnsemblePassResult
        """
        from pivtools_core.window_utils import (
            compute_window_centers,
            compute_window_centers_single_mode,
        )

        logging.info(
            f"Stereo pass {pass_idx + 1}: Finalizing "
            f"(N={self.n_images} images, angle={np.degrees(self.stereo_half_angle):.2f} deg)"
        )

        pd = self.passes_data[pass_idx]
        N = self.n_images
        n_win_y = pd["n_win_y"]
        n_win_x = pd["n_win_x"]
        corr_size = pd["corr_size"]
        win_size = pd["win_size"]
        total_windows = n_win_y * n_win_x

        sin_th = np.sin(self.stereo_half_angle)

        # =====================================================================
        # Step 1: Per-camera Gaussian fitting
        # =====================================================================
        cam1_params = self._finalize_camera(
            pass_idx, client, predictor_field,
            pd["cam1_sum_corr_AA"], pd["cam1_sum_corr_BB"], pd["cam1_sum_corr_AB"],
            pd["cam1_sum_warp_A"], pd["cam1_sum_warp_B"],
            camera_label="cam1",
        )
        cam2_params = self._finalize_camera(
            pass_idx, client, predictor_field,
            pd["cam2_sum_corr_AA"], pd["cam2_sum_corr_BB"], pd["cam2_sum_corr_AB"],
            pd["cam2_sum_warp_A"], pd["cam2_sum_warp_B"],
            camera_label="cam2",
        )

        # =====================================================================
        # Step 2: CoC Gaussian fitting
        # =====================================================================
        coc_avg = pd["coc_sum"] / N  # (total_windows, coc_h, coc_w)
        coc_spread_xx, coc_spread_yy, coc_spread_xy = self._fit_coc_spreads(
            coc_avg, n_win_y, n_win_x
        )

        # =====================================================================
        # Step 3: 3D velocity reconstruction
        # =====================================================================
        d1_x = cam1_params["ux"]
        d1_y = cam1_params["uy"]
        d2_x = cam2_params["ux"]
        d2_y = cam2_params["uy"]

        # In-plane displacements (average of both cameras)
        ux_px = (d1_x + d2_x) / 2.0
        uy_px = (d1_y + d2_y) / 2.0
        # Out-of-plane from displacement difference
        uz_px = (d1_x - d2_x) / (2.0 * sin_th)

        # Convert pixels → mm → m/s
        # Velocity = displacement_mm / dt, but dt is applied during calibration
        # Here we store in dewarped pixel units; calibration step converts to m/s
        ux_mat = ux_px.astype(np.float64)
        uy_mat = uy_px.astype(np.float64)
        uz_mat = uz_px.astype(np.float64)

        # =====================================================================
        # Step 4: Extract all 6 Reynolds stress components
        # =====================================================================
        # Per-camera displacement variance (sig_AB - sig_A)
        Sigma_11_xx = np.maximum(
            cam1_params["sig_AB_x"] - cam1_params["sig_A_x"], 0.0
        )
        Sigma_11_yy = np.maximum(
            cam1_params["sig_AB_y"] - cam1_params["sig_A_y"], 0.0
        )
        Sigma_11_xy = cam1_params["sig_AB_xy"] - cam1_params["sig_A_xy"]

        Sigma_22_xx = np.maximum(
            cam2_params["sig_AB_x"] - cam2_params["sig_A_x"], 0.0
        )
        Sigma_22_yy = np.maximum(
            cam2_params["sig_AB_y"] - cam2_params["sig_A_y"], 0.0
        )
        Sigma_22_xy = cam2_params["sig_AB_xy"] - cam2_params["sig_A_xy"]

        # Standard 5 observables
        R_yy = (Sigma_11_yy + Sigma_22_yy) / 2.0  # VV stress
        R_xy = (Sigma_11_xy + Sigma_22_xy) / 2.0  # UV stress
        R_yz = (Sigma_11_xy - Sigma_22_xy) / (2.0 * sin_th)  # VW stress
        R_xz = (Sigma_11_xx - Sigma_22_xx) / (4.0 * sin_th)  # UW stress

        # Coupled observable: A = R_xx + sin²θ·R_zz
        A = (Sigma_11_xx + Sigma_22_xx) / 2.0

        # CoC extraction: Sigma_12_xx = (spread_11_xx + spread_22_xx - spread_C_xx) / 2
        # where spread_11_xx and spread_22_xx are the per-camera displacement variances
        # and spread_C_xx is the CoC spread
        Sigma_12_xx = (Sigma_11_xx + Sigma_22_xx - coc_spread_xx) / 2.0
        B = Sigma_12_xx  # = R_xx - sin²θ·R_zz

        # Decouple R_xx and R_zz
        R_xx = (A + B) / 2.0
        R_zz = (A - B) / (2.0 * sin_th ** 2)

        # Validate: R_xx >= 0, R_zz >= 0 (variances must be non-negative)
        invalid = R_zz < 0
        n_invalid = invalid.sum()
        if n_invalid > 0:
            logging.warning(
                f"Stereo pass {pass_idx + 1}: {n_invalid} windows have R_zz < 0, "
                f"falling back to coupled A"
            )
            R_zz[invalid] = 0.0
            R_xx[invalid] = A[invalid]

        invalid_xx = R_xx < 0
        n_invalid_xx = invalid_xx.sum()
        if n_invalid_xx > 0:
            logging.warning(
                f"Stereo pass {pass_idx + 1}: {n_invalid_xx} windows have R_xx < 0, "
                f"clamping to 0"
            )
            R_xx[invalid_xx] = 0.0

        UU_stress = R_xx.astype(np.float32)
        VV_stress = R_yy.astype(np.float32)
        WW_stress = R_zz.astype(np.float32)
        UV_stress = R_xy.astype(np.float32)
        UW_stress = R_xz.astype(np.float32)
        VW_stress = R_yz.astype(np.float32)

        logging.info(
            f"Stereo pass {pass_idx + 1}: Stress extraction complete — "
            f"UU={np.nanmean(UU_stress):.4f}, VV={np.nanmean(VV_stress):.4f}, "
            f"WW={np.nanmean(WW_stress):.4f}, UV={np.nanmean(UV_stress):.4f}, "
            f"UW={np.nanmean(UW_stress):.4f}, VW={np.nanmean(VW_stress):.4f}"
        )

        # Average peakheight from both cameras
        peakheight = (
            (cam1_params["peakheight"] + cam2_params["peakheight"]) / 2.0
        ).astype(np.float32)

        # =====================================================================
        # Step 5: Apply vector mask
        # =====================================================================
        nan_reason = np.zeros((n_win_y, n_win_x), dtype=np.int32)
        # Mark fitting failures from either camera
        nan_reason[cam1_params["statuses"] != 0] = cam1_params["statuses"][
            cam1_params["statuses"] != 0
        ]
        nan_reason[cam2_params["statuses"] != 0] = cam2_params["statuses"][
            cam2_params["statuses"] != 0
        ]

        vector_mask = None
        if self.vector_masks and pass_idx < len(self.vector_masks):
            vector_mask = self.vector_masks[pass_idx]

        if vector_mask is None:
            vector_mask = np.zeros((n_win_y, n_win_x), dtype=bool)

        if vector_mask.any():
            for arr in [
                ux_mat, uy_mat, uz_mat,
                UU_stress, VV_stress, WW_stress, UV_stress, UW_stress, VW_stress,
                peakheight,
            ]:
                arr[vector_mask] = 0.0
            nan_reason[vector_mask] = -1
            logging.info(
                f"Stereo pass {pass_idx + 1}: {vector_mask.sum()} vectors masked"
            )

        # =====================================================================
        # Step 6: Outlier detection and infilling
        # =====================================================================
        is_final_pass = (pass_idx == self.config.stereo_ensemble_num_passes - 1)

        # Start with fitting failures
        outlier_mask = (nan_reason > 0)
        if vector_mask is not None:
            outlier_mask = outlier_mask & ~vector_mask

        # Apply outlier detection on valid vectors
        if self.config.ensemble_outlier_detection_enabled:
            outlier_methods = self.config.ensemble_outlier_detection_methods
            if outlier_methods:
                valid_for_detection = ~outlier_mask & ~vector_mask
                if valid_for_detection.any():
                    detected = apply_outlier_detection(
                        ux_mat, uy_mat, outlier_methods, peak_mag=peakheight
                    )
                    outlier_mask |= (detected & valid_for_detection)

        logging.info(
            f"Stereo pass {pass_idx + 1}: {outlier_mask.sum()} outliers "
            f"({outlier_mask.sum() / outlier_mask.size * 100:.1f}%)"
        )

        # Set outliers to NaN
        for arr in [
            ux_mat, uy_mat, uz_mat,
            UU_stress, VV_stress, WW_stress, UV_stress, UW_stress, VW_stress,
            peakheight,
        ]:
            arr[outlier_mask] = np.nan

        nan_reason[outlier_mask & (nan_reason == 0)] = 10

        # Infilling
        infill_mask = outlier_mask.copy()
        if is_final_pass:
            infill_cfg = self.config.ensemble_infilling_final_pass
            if not infill_cfg.get('enabled', True):
                infill_mask = np.zeros_like(outlier_mask, dtype=bool)
        else:
            infill_cfg = self.config.ensemble_infilling_mid_pass

        if infill_mask.any():
            logging.info(
                f"Stereo pass {pass_idx + 1}: Infilling {infill_mask.sum()} vectors"
            )
            ux_mat, uy_mat = apply_infilling(
                ux_mat, uy_mat, infill_mask, infill_cfg
            )
            uz_temp = np.zeros_like(uz_mat)
            uz_mat, _ = apply_infilling(uz_mat, uz_temp, infill_mask, infill_cfg)

            UU_stress, VV_stress = apply_infilling(
                UU_stress, VV_stress, infill_mask, infill_cfg
            )
            WW_temp = np.zeros_like(WW_stress)
            WW_stress, _ = apply_infilling(
                WW_stress, WW_temp, infill_mask, infill_cfg
            )
            UV_stress, UW_stress = apply_infilling(
                UV_stress, UW_stress, infill_mask, infill_cfg
            )
            VW_temp = np.zeros_like(VW_stress)
            VW_stress, _ = apply_infilling(
                VW_stress, VW_temp, infill_mask, infill_cfg
            )
            ph_temp = np.zeros_like(peakheight)
            peakheight, _ = apply_infilling(
                peakheight, ph_temp, infill_mask, infill_cfg
            )

        # =====================================================================
        # Step 7: Build predictor field for next pass
        # =====================================================================
        pred_x = None
        pred_y = None
        n_pre = pd.get("n_pre")
        n_post = pd.get("n_post")
        if n_pre is not None and n_post is not None:
            pre_y, pre_x = n_pre
            post_y, post_x = n_post
            stacked = np.stack([uy_mat, ux_mat], axis=-1)  # (ny, nx, 2)
            padded = np.pad(
                stacked,
                ((pre_y, post_y), (pre_x, post_x), (0, 0)),
                mode="edge",
            )
            pred_y = padded[:, :, 0].copy()
            pred_x = padded[:, :, 1].copy()
        elif predictor_field is not None:
            pred_y = predictor_field[:, :, 0].copy()
            pred_x = predictor_field[:, :, 1].copy()

        # Compute window centers for this pass
        runtype = self.config.stereo_ensemble_type[pass_idx]
        if runtype == 'single':
            grid_result = compute_window_centers_single_mode(
                image_shape=self.config.image_shape,
                window_size=tuple(win_size),
                sum_window=tuple(self.config.stereo_ensemble_sum_window),
                overlap=self.config.stereo_ensemble_overlaps[pass_idx],
                validate=True,
            )
        else:
            grid_result = compute_window_centers(
                image_shape=self.config.image_shape,
                window_size=tuple(win_size),
                overlap=self.config.stereo_ensemble_overlaps[pass_idx],
                validate=True,
            )

        # Build result
        pass_result = PIVStereoEnsemblePassResult(
            ux_mat=ux_mat,
            uy_mat=uy_mat,
            uz_mat=uz_mat,
            UU_stress=UU_stress,
            VV_stress=VV_stress,
            WW_stress=WW_stress,
            UV_stress=UV_stress,
            UW_stress=UW_stress,
            VW_stress=VW_stress,
            d1_x=d1_x.astype(np.float64),
            d1_y=d1_y.astype(np.float64),
            d2_x=d2_x.astype(np.float64),
            d2_y=d2_y.astype(np.float64),
            sig_AB_x_cam1=cam1_params["sig_AB_x"],
            sig_AB_y_cam1=cam1_params["sig_AB_y"],
            sig_AB_xy_cam1=cam1_params["sig_AB_xy"],
            sig_A_x_cam1=cam1_params["sig_A_x"],
            sig_A_y_cam1=cam1_params["sig_A_y"],
            sig_A_xy_cam1=cam1_params["sig_A_xy"],
            sig_AB_x_cam2=cam2_params["sig_AB_x"],
            sig_AB_y_cam2=cam2_params["sig_AB_y"],
            sig_AB_xy_cam2=cam2_params["sig_AB_xy"],
            sig_A_x_cam2=cam2_params["sig_A_x"],
            sig_A_y_cam2=cam2_params["sig_A_y"],
            sig_A_xy_cam2=cam2_params["sig_A_xy"],
            Sigma_12_xx=Sigma_12_xx.astype(np.float32),
            peakheight=peakheight,
            nan_reason=nan_reason,
            b_mask=vector_mask,
            pred_x=pred_x,
            pred_y=pred_y,
            window_size=tuple(win_size),
            win_ctrs_x=grid_result.win_ctrs_x,
            win_ctrs_y=grid_result.win_ctrs_y,
            stereo_angle=self.stereo_half_angle,
            mm_per_pixel=self.mm_per_pixel,
        )

        self.passes_results.append(pass_result)

        # Save correlation planes if store_planes is enabled
        if self.config.stereo_ensemble_store_planes:
            try:
                from scipy.io import savemat
                if output_path is not None:
                    outdir = Path(output_path)
                else:
                    import os
                    outdir = Path(os.getcwd())
                outdir.mkdir(parents=True, exist_ok=True)

                planes_dict = {
                    'cam1_AA': pd["cam1_sum_corr_AA"].reshape(
                        n_win_y, n_win_x, corr_size[0], corr_size[1]
                    ) / N if pd["cam1_sum_corr_AA"] is not None else np.empty((0,)),
                    'cam1_BB': pd["cam1_sum_corr_BB"].reshape(
                        n_win_y, n_win_x, corr_size[0], corr_size[1]
                    ) / N if pd["cam1_sum_corr_BB"] is not None else np.empty((0,)),
                    'cam1_AB': pd["cam1_sum_corr_AB"].reshape(
                        n_win_y, n_win_x, corr_size[0], corr_size[1]
                    ) / N if pd["cam1_sum_corr_AB"] is not None else np.empty((0,)),
                    'cam2_AA': pd["cam2_sum_corr_AA"].reshape(
                        n_win_y, n_win_x, corr_size[0], corr_size[1]
                    ) / N if pd["cam2_sum_corr_AA"] is not None else np.empty((0,)),
                    'cam2_BB': pd["cam2_sum_corr_BB"].reshape(
                        n_win_y, n_win_x, corr_size[0], corr_size[1]
                    ) / N if pd["cam2_sum_corr_BB"] is not None else np.empty((0,)),
                    'cam2_AB': pd["cam2_sum_corr_AB"].reshape(
                        n_win_y, n_win_x, corr_size[0], corr_size[1]
                    ) / N if pd["cam2_sum_corr_AB"] is not None else np.empty((0,)),
                    'coc_sum': pd["coc_sum"].reshape(
                        n_win_y, n_win_x, *pd["coc_size"]
                    ) / N if pd["coc_sum"] is not None else np.empty((0,)),
                    'corr_size': corr_size,
                    'n_win_y': n_win_y,
                    'n_win_x': n_win_x,
                    'pass_idx': pass_idx,
                }

                savemat(
                    str(outdir / f"planes_pass_{pass_idx + 1}.mat"),
                    planes_dict,
                    do_compression=True,
                )
                logging.info(
                    f"Stereo pass {pass_idx + 1}: Saved correlation planes to "
                    f"{outdir / f'planes_pass_{pass_idx + 1}.mat'}"
                )

                # Save first-pair warped images to separate MAT file
                if pd.get("first_pair_cam1_A") is not None:
                    warped_dict = {
                        'cam1_A_warped': pd["first_pair_cam1_A"],
                        'cam1_B_warped': pd["first_pair_cam1_B"],
                        'cam2_A_warped': pd["first_pair_cam2_A"],
                        'cam2_B_warped': pd["first_pair_cam2_B"],
                        'pass_idx': pass_idx,
                    }
                    savemat(
                        str(outdir / f"warped_pass_{pass_idx + 1}.mat"),
                        warped_dict,
                        do_compression=True,
                    )
                    logging.info(
                        f"Stereo pass {pass_idx + 1}: Saved first-pair warped images to "
                        f"{outdir / f'warped_pass_{pass_idx + 1}.mat'}"
                    )
            except Exception as e:
                logging.warning(
                    f"Stereo pass {pass_idx + 1}: Failed to save diagnostics: {e}"
                )

        logging.info(f"Stereo pass {pass_idx + 1}: Finalization complete")
        return pass_result

    # =========================================================================
    # Per-camera fitting helper
    # =========================================================================

    def _finalize_camera(
        self,
        pass_idx: int,
        client: Client,
        predictor_field: Optional[np.ndarray],
        sum_corr_AA: np.ndarray,
        sum_corr_BB: np.ndarray,
        sum_corr_AB: np.ndarray,
        sum_warp_A: np.ndarray,
        sum_warp_B: np.ndarray,
        camera_label: str = "cam",
    ) -> dict:
        """
        Run background subtraction, normalization, Gaussian fitting, and
        parameter extraction for a single camera's correlation data.

        Returns dict with keys:
            ux, uy, sig_AB_x/y/xy, sig_A_x/y/xy, peakheight, statuses
        """
        from pivtools_cli.piv.piv_result import PIVEnsembleResult

        pd = self.passes_data[pass_idx]
        N = self.n_images
        n_win_y = pd["n_win_y"]
        n_win_x = pd["n_win_x"]
        corr_size = pd["corr_size"]
        win_size = pd["win_size"]
        total_windows = n_win_y * n_win_x

        # Average correlation planes
        R_AA_raw = sum_corr_AA / N
        R_BB_raw = sum_corr_BB / N
        R_AB_raw = sum_corr_AB / N

        # Background subtraction
        bg_method = getattr(
            self.config, 'ensemble_background_subtraction_method', 'correlation'
        )
        skip_bg = getattr(
            self.config, 'ensemble_skip_background_subtraction', False
        )

        if bg_method == 'image' or skip_bg:
            R_AA_ens = R_AA_raw
            R_BB_ens = R_BB_raw
            R_AB_ens = R_AB_raw
        else:
            # Compute mean images and correlate for background
            A_mean = sum_warp_A / N
            B_mean = sum_warp_B / N
            R_AA_bg, R_BB_bg, R_AB_bg = self._correlate_mean_images(
                A_mean, B_mean, pass_idx
            )
            R_AA_ens = R_AA_raw - R_AA_bg
            R_BB_ens = R_BB_raw - R_BB_bg
            R_AB_ens = R_AB_raw - R_AB_bg

        # Normalize by geometric mean of autocorrelation peaks
        AA_3d = R_AA_ens.reshape(total_windows, corr_size[0], corr_size[1])
        BB_3d = R_BB_ens.reshape(total_windows, corr_size[0], corr_size[1])
        AB_3d = R_AB_ens.reshape(total_windows, corr_size[0], corr_size[1])

        center_y, center_x = corr_size[0] // 2, corr_size[1] // 2
        AA_peaks = AA_3d[:, center_y, center_x]
        BB_peaks = BB_3d[:, center_y, center_x]
        norm_factors = np.sqrt(np.maximum(AA_peaks * BB_peaks, 1e-12))
        norm_3d = norm_factors[:, np.newaxis, np.newaxis]

        # Single mode AB scale correction
        runtype = self.config.stereo_ensemble_type[pass_idx]
        if runtype == 'single':
            sum_window = self.config.stereo_ensemble_sum_window
            particle_window = win_size
            sum_area = sum_window[0] * sum_window[1]
            particle_area = particle_window[0] * particle_window[1]
            AB_3d = AB_3d * np.sqrt(sum_area / particle_area)

        R_AA_ens = (AA_3d / norm_3d).reshape(-1).astype(np.float32)
        R_BB_ens = (BB_3d / norm_3d).reshape(-1).astype(np.float32)
        R_AB_ens = (AB_3d / norm_3d).reshape(-1).astype(np.float32)

        # Distributed Gaussian fitting
        # Build temporary PIVEnsembleResult for sigma propagation
        temp_result = PIVEnsembleResult()
        for pr in self.passes_results:
            # We don't have per-camera stored results — use None sigmas for pass 0
            # For pass > 0, we could propagate but keeping it simple: always fresh
            pass

        sigma_dict = _get_sigma_from_previous_pass(
            pass_idx, total_windows, self.config, temp_result, n_win_x, n_win_y
        )

        # Flatten mask
        if self.vector_masks and pass_idx < len(self.vector_masks):
            mask_flat = self.vector_masks[pass_idx].ravel(order='C').astype(bool)
        else:
            mask_flat = np.zeros(total_windows, dtype=bool)

        # Scatter + fit on Dask workers
        from pivtools_cli.piv.piv_backend.gaussian_fitting import fit_windows_openmp

        n_workers = len(client.scheduler_info()['workers'])
        windows_per_worker = (total_windows + n_workers - 1) // n_workers

        R_AA_f, R_BB_f, R_AB_f, mask_f = [], [], [], []
        sigma_f = [{} for _ in range(n_workers)]

        for wi in range(n_workers):
            s = wi * windows_per_worker * corr_size[0] * corr_size[1]
            e = min(
                (wi + 1) * windows_per_worker * corr_size[0] * corr_size[1],
                R_AA_ens.size,
            )
            sw = wi * windows_per_worker
            ew = min((wi + 1) * windows_per_worker, total_windows)

            R_AA_f.append(client.scatter(R_AA_ens[s:e], broadcast=False))
            R_BB_f.append(client.scatter(R_BB_ens[s:e], broadcast=False))
            R_AB_f.append(client.scatter(R_AB_ens[s:e], broadcast=False))
            mask_f.append(client.scatter(mask_flat[sw:ew], broadcast=False))

            for k, v in sigma_dict.items():
                if v is not None:
                    sigma_f[wi][k] = client.scatter(v[sw:ew], broadcast=False)
                else:
                    sigma_f[wi][k] = None

        fit_offset = getattr(self.config, 'ensemble_fit_offset', True)
        futures = [
            client.submit(
                fit_windows_openmp,
                R_AA_f[i], R_BB_f[i], R_AB_f[i],
                mask_f[i], sigma_f[i],
                corr_size, self.config, pass_idx,
                None, fit_offset,
            )
            for i in range(n_workers)
        ]

        results = client.gather(futures)
        gauss_flat = np.concatenate([r[0] for r in results])
        status_flat = np.concatenate([r[1] for r in results])

        del R_AA_ens, R_BB_ens, R_AB_ens
        gc.collect()

        gauss_results = gauss_flat.reshape(n_win_y, n_win_x, -1)
        statuses = status_flat.reshape(n_win_y, n_win_x)

        # Extract displacement
        win_center_x = corr_size[1] / 2.0 + 1
        win_center_y = corr_size[0] / 2.0 + 1
        ux = (gauss_results[:, :, 14] - win_center_x).astype(np.float32)
        uy = (gauss_results[:, :, 15] - win_center_y).astype(np.float32)

        # 3/4 window rule
        max_disp_x = 0.75 * corr_size[1]
        max_disp_y = 0.75 * corr_size[0]
        bad = (
            ~np.isfinite(ux) | ~np.isfinite(uy)
            | (np.abs(ux) > max_disp_x)
            | (np.abs(uy) > max_disp_y)
        )
        ux[bad] = np.nan
        uy[bad] = np.nan
        statuses[bad] = 6

        # Add predictor back for pass > 0
        if pass_idx > 0 and "smoothed_predictor" in pd and pd["smoothed_predictor"] is not None:
            smoothed_pred = pd["smoothed_predictor"]
            ux += smoothed_pred[:, :, 1]
            uy += smoothed_pred[:, :, 0]

        # Extract sigma parameters with clamping
        MAX_SIGMA = 1e6

        def _safe(arr):
            r = np.clip(arr, -MAX_SIGMA, MAX_SIGMA)
            r = np.where(np.isfinite(r), r, 0.0)
            return r.astype(np.float32)

        sig_A_x = _safe(gauss_results[:, :, 6])
        sig_A_y = _safe(gauss_results[:, :, 7])
        sig_A_xy = _safe(gauss_results[:, :, 8])
        sig_AB_x = _safe(gauss_results[:, :, 9])
        sig_AB_y = _safe(gauss_results[:, :, 10])
        sig_AB_xy = _safe(gauss_results[:, :, 11])

        # Peakheight
        amp_A = np.maximum(gauss_results[:, :, 0], 0)
        amp_B = np.maximum(gauss_results[:, :, 1], 0)
        amp_AB = np.maximum(gauss_results[:, :, 2], 0)
        geom_mean = np.sqrt(np.maximum(amp_A * amp_B, 1e-12))
        peakheight = np.clip(amp_AB / geom_mean, 0.0, 1.0).astype(np.float32)

        logging.info(
            f"Stereo pass {pass_idx + 1} [{camera_label}]: "
            f"ux range [{np.nanmin(ux):.3f}, {np.nanmax(ux):.3f}], "
            f"uy range [{np.nanmin(uy):.3f}, {np.nanmax(uy):.3f}]"
        )

        return {
            "ux": ux,
            "uy": uy,
            "sig_AB_x": sig_AB_x,
            "sig_AB_y": sig_AB_y,
            "sig_AB_xy": sig_AB_xy,
            "sig_A_x": sig_A_x,
            "sig_A_y": sig_A_y,
            "sig_A_xy": sig_A_xy,
            "peakheight": peakheight,
            "statuses": statuses,
        }

    def _correlate_mean_images(
        self,
        A_mean: np.ndarray,
        B_mean: np.ndarray,
        pass_idx: int,
    ) -> tuple:
        """Correlate mean images for background subtraction (same as SinglePassAccumulator)."""
        from pivtools_core.window_utils import apply_single_mode_padding

        pd = self.passes_data[pass_idx]
        win_size = pd["win_size"]
        corr_size = pd["corr_size"]
        n_win_y = pd["n_win_y"]
        n_win_x = pd["n_win_x"]
        total_windows = n_win_y * n_win_x

        runtype = self.config.stereo_ensemble_type[pass_idx]
        if runtype == 'single':
            sum_window = tuple(self.config.stereo_ensemble_sum_window)
            A_mean, _ = apply_single_mode_padding(
                A_mean, win_size, sum_window, pad_value=0.0
            )
            B_mean, _ = apply_single_mode_padding(
                B_mean, win_size, sum_window, pad_value=0.0
            )

        correl_AA_bg = np.zeros(
            total_windows * corr_size[0] * corr_size[1], dtype=np.float32
        )
        correl_BB_bg = np.zeros_like(correl_AA_bg)
        correl_AB_bg = np.zeros_like(correl_AA_bg)

        from pivtools_cli.piv.piv_backend.factory import make_correlator_backend

        correlator = make_correlator_backend(self.config, ensemble=True)
        comp_size = correlator.window_sizes_for_computation[pass_idx]
        out_size = correlator.window_sizes_for_corr[pass_idx]

        n_windows = np.array([n_win_y, n_win_x], dtype=np.int32)
        image_size = np.array([A_mean.shape[0], A_mean.shape[1]], dtype=np.int32)
        win_size_arr = np.array([comp_size[0], comp_size[1]], dtype=np.int32)
        fit_size_arr = np.array([out_size[0], out_size[1]], dtype=np.int32)

        if correlator.vector_masks and pass_idx < len(correlator.vector_masks):
            b_mask = np.ascontiguousarray(
                correlator.vector_masks[pass_idx].astype(np.float32)
            )
        else:
            b_mask = np.zeros((n_win_y, n_win_x), dtype=np.float32)

        A_stack = np.ascontiguousarray(
            A_mean[np.newaxis, :, :].astype(np.float32)
        )
        B_stack = np.ascontiguousarray(
            B_mean[np.newaxis, :, :].astype(np.float32)
        )

        for output_buf, img_a, img_b in [
            (correl_AB_bg, A_stack, B_stack),
            (correl_AA_bg, A_stack, A_stack),
            (correl_BB_bg, B_stack, B_stack),
        ]:
            # For AA/BB, use weight_B on both sides (same as SinglePassAccumulator)
            w_a = correlator.win_weights_B[pass_idx] if img_a is img_b else correlator.win_weights_A[pass_idx]
            w_b = correlator.win_weights_B[pass_idx]
            correlator.lib.bulkxcorr2d_accumulate(
                img_a, img_b, b_mask,
                image_size, 1,
                correlator.win_ctrs_x[pass_idx].astype(np.float32),
                correlator.win_ctrs_y[pass_idx].astype(np.float32),
                n_windows, w_a, w_b,
                win_size_arr, fit_size_arr,
                output_buf,
            )

        return correl_AA_bg, correl_BB_bg, correl_AB_bg

    # =========================================================================
    # CoC spread fitting
    # =========================================================================

    def _fit_coc_spreads(
        self,
        coc_avg: np.ndarray,
        n_win_y: int,
        n_win_x: int,
    ) -> tuple:
        """
        Fit 2D Gaussian to each window's CoC plane to extract spread parameters.

        Parameters
        ----------
        coc_avg : (total_windows, coc_h, coc_w) averaged CoC planes
        n_win_y, n_win_x : grid dimensions

        Returns
        -------
        (spread_xx, spread_yy, spread_xy) : each (n_win_y, n_win_x) float32
            Gaussian spread (variance) of the CoC plane in pixel² units
        """
        total_windows = n_win_y * n_win_x
        spread_xx = np.full(total_windows, np.nan, dtype=np.float32)
        spread_yy = np.full(total_windows, np.nan, dtype=np.float32)
        spread_xy = np.full(total_windows, np.nan, dtype=np.float32)

        for w in range(total_windows):
            plane = coc_avg[w].astype(np.float64)
            sx, sy, sxy = self._fit_coc_window(plane)
            spread_xx[w] = sx
            spread_yy[w] = sy
            spread_xy[w] = sxy

        n_valid = np.sum(np.isfinite(spread_xx))
        logging.info(
            f"CoC fitting: {n_valid}/{total_windows} windows fitted successfully"
        )

        return (
            spread_xx.reshape(n_win_y, n_win_x),
            spread_yy.reshape(n_win_y, n_win_x),
            spread_xy.reshape(n_win_y, n_win_x),
        )

    @staticmethod
    def _fit_coc_window(plane: np.ndarray) -> tuple:
        """
        Fit a 2D elliptical Gaussian to a single CoC correlation plane
        and return the spread (variance) parameters.

        The Gaussian model: G(i,j) = A * exp(-0.5*(di²*sx + dj²*sy + 2*di*dj*sxy))
        where sx, sy, sxy are inverse covariance parameters.

        The actual variance (spread) is obtained by inverting the 2x2 matrix:
            [[sx, sxy], [sxy, sy]]^-1 = [[var_yy, -var_xy], [-var_xy, var_xx]] / det

        Parameters
        ----------
        plane : (coc_h, coc_w) float64

        Returns
        -------
        (spread_xx, spread_yy, spread_xy) : variance in pixel² units
            Returns (NaN, NaN, NaN) if fitting fails.
        """
        h, w = plane.shape
        center_y, center_x = h // 2, w // 2

        # Search central 75% for coarse peak
        margin = max(min(h, w) // 8, 2)
        search = plane[margin:h - margin, margin:w - margin]
        if search.size == 0:
            return np.nan, np.nan, np.nan

        coarse_idx = np.unravel_index(np.argmax(search), search.shape)
        iy = coarse_idx[0] + margin
        ix = coarse_idx[1] + margin

        # Reject border peaks
        r = 2
        if iy < r or iy >= h - r or ix < r or ix >= w - r:
            return np.nan, np.nan, np.nan

        # Extract 5x5 region
        region = plane[iy - r:iy + r + 1, ix - r:ix + r + 1].copy()
        region_min = region.min()
        region = region - region_min

        if region.max() <= 0:
            return np.nan, np.nan, np.nan

        ii, jj = np.mgrid[-r:r + 1, -r:r + 1]
        ii_flat = ii.ravel().astype(np.float64)
        jj_flat = jj.ravel().astype(np.float64)
        data_flat = region.ravel().astype(np.float64)

        # Initial guess via 3-point parabolic
        cy = region[:, r]
        cx = region[r, :]
        eps = 1e-10
        log_cy = np.log(np.maximum(cy, eps))
        log_cx = np.log(np.maximum(cx, eps))

        denom_y = log_cy[r - 1] - 2 * log_cy[r] + log_cy[r + 1]
        denom_x = log_cx[r - 1] - 2 * log_cx[r] + log_cx[r + 1]

        sx_init = max(0.25, min(9.0, -denom_y if abs(denom_y) > eps else 1.0))
        sy_init = max(0.25, min(9.0, -denom_x if abs(denom_x) > eps else 1.0))
        di0 = 0.5 * (log_cy[r - 1] - log_cy[r + 1]) / denom_y if abs(denom_y) > eps else 0.0
        dj0 = 0.5 * (log_cx[r - 1] - log_cx[r + 1]) / denom_x if abs(denom_x) > eps else 0.0
        A_init = float(region.max())

        p0 = [A_init, di0, dj0, sx_init, sy_init, 0.0]

        def model(params, i_c, j_c):
            A, i0, j0, sx, sy, sxy = params
            di = i_c - i0
            dj = j_c - j0
            exp_arg = -0.5 * (di * di * sx + dj * dj * sy + 2.0 * di * dj * sxy)
            return A * np.exp(np.clip(exp_arg, -50, 50))

        def residuals(params, i_c, j_c, d):
            return model(params, i_c, j_c) - d

        try:
            result = least_squares(
                residuals, p0,
                args=(ii_flat, jj_flat, data_flat),
                method='lm', max_nfev=20,
            )
            A_fit, i0_fit, j0_fit, sx_fit, sy_fit, sxy_fit = result.x
        except Exception:
            return np.nan, np.nan, np.nan

        if A_fit <= 0 or abs(i0_fit) > r or abs(j0_fit) > r:
            return np.nan, np.nan, np.nan

        # Convert inverse covariance to covariance (spread)
        # [[sx, sxy], [sxy, sy]]^-1 = [[sy, -sxy], [-sxy, sx]] / det
        det = sx_fit * sy_fit - sxy_fit * sxy_fit
        if det <= 0:
            return np.nan, np.nan, np.nan

        # Note: the model uses (i=row, j=col), so sx → row variance, sy → col variance
        # In our coordinate system: row=y, col=x
        # spread_xx = var(col) = sy / det
        # spread_yy = var(row) = sx / det
        # spread_xy = -sxy / det
        spread_xx = sy_fit / det
        spread_yy = sx_fit / det
        spread_xy = -sxy_fit / det

        # Reject unreasonable values
        if spread_xx < 0 or spread_yy < 0 or spread_xx > 1e4 or spread_yy > 1e4:
            return np.nan, np.nan, np.nan

        return float(spread_xx), float(spread_yy), float(spread_xy)

    # =========================================================================
    # Result assembly and cleanup
    # =========================================================================

    def get_stereo_ensemble_result(self) -> PIVStereoEnsembleResult:
        """Get final result with all passes."""
        result = PIVStereoEnsembleResult()
        for pr in self.passes_results:
            result.add_pass(pr)
        logging.info(f"Assembled {len(self.passes_results)} stereo ensemble passes")
        return result

    def clear_pass_data(self, pass_idx: int):
        """Clear accumulated data for a pass to free memory."""
        if pass_idx >= len(self.passes_data):
            return

        self.n_images = 0
        pd = self.passes_data[pass_idx]

        mem_before = sum(
            v.nbytes for v in pd.values()
            if isinstance(v, np.ndarray)
        ) / (1024 ** 2)

        for key in list(pd.keys()):
            if isinstance(pd[key], np.ndarray):
                pd[key] = None
        pd["smoothed_predictor"] = None

        logging.debug(
            f"Stereo pass {pass_idx + 1}: Cleared data (~{mem_before:.1f} MB freed)"
        )
