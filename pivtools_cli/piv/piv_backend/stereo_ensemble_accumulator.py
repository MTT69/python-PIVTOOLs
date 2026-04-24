"""Accumulator and finalization for Stereo Ensemble PIV (CoC method).

Manages dual-camera correlation buffers + CoC buffer, and implements the
finalize_pass() pipeline that converts accumulated correlations into
3D velocity + 6 decoupled Reynolds stresses.

Mirrors SinglePassAccumulator but handles:
- Dual-camera (cam1 + cam2) correlation planes
- CoC (Correlation-of-Correlations) planes
- 3D velocity reconstruction from stereo geometry
- 6 Reynolds stress extraction via CoC decoupling
- Physical unit conversion (dewarped px → m/s, mm)
"""

from __future__ import annotations

import ctypes
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

c_float_p = ctypes.POINTER(ctypes.c_float)
c_int_p = ctypes.POINTER(ctypes.c_int)

from pivtools_core.config import Config
from pivtools_cli.piv.stereo_ensemble_result import (
    PIVStereoEnsemblePassResult,
)


class StereoEnsembleAccumulator:
    """Accumulates stereo ensemble correlation results and finalizes passes.

    Buffers per pass:
    - cam1: AA_sum, BB_sum, AB_sum  (flat float32)
    - cam2: AA_sum, BB_sum, AB_sum
    - CoC: CoC_sum
    - warp sums: 1A, 1B, 2A, 2B (2D images for background subtraction)
    """

    def __init__(
        self,
        config: Config,
        mm_per_pixel: float,
        stereo_angle: float,
        vector_masks: Optional[List[np.ndarray]] = None,
    ):
        self.config = config
        self.mm_per_pixel = mm_per_pixel
        self.stereo_angle = stereo_angle  # sin(theta)
        self.vector_masks = vector_masks or []
        self.n_images = 0
        self.passes_data: List[dict] = []
        self.passes_results: List[PIVStereoEnsemblePassResult] = []

    def set_pass_data(self, pass_idx: int, accumulated_result: dict):
        """Store accumulated worker result for a pass.

        Parameters
        ----------
        pass_idx : int
            Pass index (0-based).
        accumulated_result : dict
            Tree-reduced result from all workers with keys:
            cam1_AB/AA/BB_sum, cam2_AB/AA/BB_sum, CoC_sum,
            warp_1A/1B/2A/2B_sum, n_images, n_win_x, n_win_y,
            smoothed_predictor, vector_mask
        """
        # Extend list if needed
        while len(self.passes_data) <= pass_idx:
            self.passes_data.append(None)

        self.passes_data[pass_idx] = accumulated_result
        self.n_images = accumulated_result["n_images"]

    def finalize_pass(
        self,
        pass_idx: int,
        correlator,
        config: Config,
        output_path: Optional[str] = None,
    ) -> PIVStereoEnsemblePassResult:
        """Finalize one pass: fitting → 3D velocity → 6 stresses → outliers.

        Steps:
        1. Mean computation (divide accumulated sums by N)
        2. Background subtraction per camera (correlation method)
        3. Normalization per camera (geometric mean of autocorr peaks)
        4. Per-camera k-space fitting → displacements + spreads
        5. CoC k-space spread fitting → cross-camera covariance
        6. 3D velocity reconstruction
        7. 6 Reynolds stress extraction (standard + CoC decoupling)
        8. Physical unit conversion (dewarped px → m/s)
        9. Outlier detection + infilling
        10. Stress realizability check
        11. Predictor extraction for next pass

        Parameters
        ----------
        pass_idx : int
            Pass index (0-based).
        correlator : StereoEnsembleCorrelatorCPU
            Correlator instance (for kspace fitting, window info).
        config : Config
            Configuration.
        output_path : str, optional
            Path for diagnostic output.

        Returns
        -------
        PIVStereoEnsemblePassResult
        """
        pass_data = self.passes_data[pass_idx]
        N = pass_data["n_images"]
        n_win_y = pass_data["n_win_y"]
        n_win_x = pass_data["n_win_x"]
        n_windows = n_win_y * n_win_x
        corr_size = correlator.window_sizes_for_corr[pass_idx]
        corr_h, corr_w = corr_size
        n_px_corr = corr_h * corr_w

        coc_size = correlator._coc_window_sizes[pass_idx]
        coc_h, coc_w = coc_size
        n_px_coc = coc_h * coc_w

        sin_theta = self.stereo_angle
        sin2_theta = sin_theta * sin_theta
        mm_per_px = self.mm_per_pixel
        dt = config.dt

        logger.info(
            f"Pass {pass_idx + 1}: finalizing stereo ensemble "
            f"({N} frames, {n_win_y}x{n_win_x} windows, "
            f"corr {corr_h}x{corr_w}, CoC {coc_h}x{coc_w})"
        )

        # ── Step 1: Mean computation ────────────────────────────────────
        cam1_AB = pass_data["cam1_AB_sum"].reshape(n_windows, n_px_corr) / N
        cam1_AA = pass_data["cam1_AA_sum"].reshape(n_windows, n_px_corr) / N
        cam1_BB = pass_data["cam1_BB_sum"].reshape(n_windows, n_px_corr) / N
        cam2_AB = pass_data["cam2_AB_sum"].reshape(n_windows, n_px_corr) / N
        cam2_AA = pass_data["cam2_AA_sum"].reshape(n_windows, n_px_corr) / N
        cam2_BB = pass_data["cam2_BB_sum"].reshape(n_windows, n_px_corr) / N
        coc_avg = pass_data["CoC_sum"].reshape(n_windows, n_px_coc) / N

        # ── Diagnostic: save raw planes before any bg subtraction ──────
        diag_coc_raw = coc_avg.copy()
        diag_cam1_AB_raw = cam1_AB.copy()
        diag_cam2_AB_raw = cam2_AB.copy()
        diag_coc_bg = None
        diag_cam1_AB_bg = None
        diag_cam2_AB_bg = None

        # ── Step 1b: CoC structured background subtraction ────────────
        # Cross-correlate the mean AB planes WITHOUT mean subtraction to match
        # the C code (which now uses xcorr_preplanned instead of xcorr_meansub).
        bg_method = config.stereo_ensemble_background_subtraction_method
        if bg_method == "correlation":
            coc_bg = self._cross_correlate_planes(
                cam1_AB, cam2_AB,
                corr_h, corr_w, coc_h, coc_w,
            )
            diag_coc_bg = coc_bg.copy()
            coc_avg = coc_avg - coc_bg
            logger.debug("  CoC structured background subtracted")

        # ── Step 2: Per-camera background subtraction ─────────────────
        if bg_method == "correlation":
            cam1_AB_bg, cam1_AA_bg, cam1_BB_bg = self._correlate_mean_images(
                pass_data["warp_1A_sum"] / N,
                pass_data["warp_1B_sum"] / N,
                pass_idx, correlator, config,
            )
            cam2_AB_bg, cam2_AA_bg, cam2_BB_bg = self._correlate_mean_images(
                pass_data["warp_2A_sum"] / N,
                pass_data["warp_2B_sum"] / N,
                pass_idx, correlator, config,
            )
            diag_cam1_AB_bg = cam1_AB_bg.copy()
            diag_cam2_AB_bg = cam2_AB_bg.copy()
            cam1_AB = cam1_AB - cam1_AB_bg
            cam1_AA = cam1_AA - cam1_AA_bg
            cam1_BB = cam1_BB - cam1_BB_bg
            cam2_AB = cam2_AB - cam2_AB_bg
            cam2_AA = cam2_AA - cam2_AA_bg
            cam2_BB = cam2_BB - cam2_BB_bg

        # ── Step 3: Normalization (geometric mean of autocorr peaks) ────
        # Save bg-subtracted (unnormalized) auto-correlations for CoC fitting.
        # The CoC plane is NOT normalized, so F_ref must use the same scale.
        cam1_AA_raw = cam1_AA.copy()
        cam1_BB_raw = cam1_BB.copy()
        cam2_AA_raw = cam2_AA.copy()
        cam2_BB_raw = cam2_BB.copy()

        def _normalize_camera(AB, AA, BB):
            """Normalize by sqrt(AA_peak * BB_peak) per window."""
            center_idx = (corr_h // 2) * corr_w + (corr_w // 2)
            AA_peaks = AA[:, center_idx]
            BB_peaks = BB[:, center_idx]
            norm = np.sqrt(np.maximum(AA_peaks * BB_peaks, 1e-12))
            norm_3d = norm[:, np.newaxis]
            return AB / norm_3d, AA / norm_3d, BB / norm_3d

        cam1_AB, cam1_AA, cam1_BB = _normalize_camera(cam1_AB, cam1_AA, cam1_BB)
        cam2_AB, cam2_AA, cam2_BB = _normalize_camera(cam2_AB, cam2_AA, cam2_BB)

        # ── Step 3b: Compute composed fractional displacements for P_noise ─
        from pivtools_cli.piv.piv_backend.interpolation_noise_psd import (
            compute_composed_fractional_displacements,
        )
        from pivtools_cli.piv.piv_backend.kspace_fitting import fit_windows_kspace

        # Build mask (1=skip for C, inverted from Python bool)
        if self.vector_masks and pass_idx < len(self.vector_masks):
            mask_flat = self.vector_masks[pass_idx].ravel().astype(np.int32)
        else:
            mask_flat = np.zeros(n_windows, dtype=np.int32)

        # Composed fractional displacements for noise PSD
        # Uses dewarp maps + predictor to compute the fractional part of the
        # raw image coordinate at each window centre (single-pass noise model).
        composed_frac_disp = None
        if hasattr(correlator, "dewarp_maps") and correlator.dewarp_maps:
            cam_pair = config.stereo_ensemble_camera_pair
            # Use cam1 maps — both cameras have similar fractional displacement
            # patterns since they're dewarped to the same world grid
            map_x, map_y = correlator.dewarp_maps[cam_pair[0]]
            win_ctrs_y = correlator._inner.win_ctrs_y[pass_idx]
            win_ctrs_x = correlator._inner.win_ctrs_x[pass_idx]

            # Extract per-window predictor (pass > 0)
            # The smoothed_predictor is on the PREVIOUS pass grid — remap to
            # the current pass grid if sizes differ.
            smoothed_pred = pass_data.get("smoothed_predictor")
            if smoothed_pred is not None and smoothed_pred.ndim == 3 and smoothed_pred.shape[2] == 2:
                import cv2
                prev_ny, prev_nx = smoothed_pred.shape[:2]
                cur_ny = len(win_ctrs_y)
                cur_nx = len(win_ctrs_x)
                if prev_ny == cur_ny and prev_nx == cur_nx:
                    pred_dy_2d = smoothed_pred[:, :, 0]
                    pred_dx_2d = smoothed_pred[:, :, 1]
                else:
                    pred_dy_2d = cv2.resize(
                        smoothed_pred[:, :, 0].astype(np.float32),
                        (cur_nx, cur_ny), interpolation=cv2.INTER_LINEAR)
                    pred_dx_2d = cv2.resize(
                        smoothed_pred[:, :, 1].astype(np.float32),
                        (cur_nx, cur_ny), interpolation=cv2.INTER_LINEAR)
            else:
                pred_dy_2d = None
                pred_dx_2d = None

            composed_frac_disp = compute_composed_fractional_displacements(
                map_x, map_y, win_ctrs_y, win_ctrs_x,
                pred_dy_2d, pred_dx_2d,
            )

        # Encode composed fractional displacements for the per-camera C fitter.
        # The C code does: f = frac_distance(pred/2). We pass pred = 2*f so
        # that frac_distance(2*f / 2) = frac_distance(f) = f. No C changes needed.
        predictor_displacements = None
        if composed_frac_disp is not None:
            predictor_displacements = 2.0 * composed_frac_disp  # (n_windows, 2): [2*f_y, 2*f_x]

        # ── Step 4: Per-camera k-space fitting ──────────────────────────

        def _fit_camera(R_AA, R_BB, R_AB, cam_label):
            """Fit one camera's correlation planes via k-space transfer function.

            Uses T(k) = F(R_AB) / sqrt(F(R_AA) * F(R_BB)) which cancels
            particle image, giving displacement + turbulence-only spreads.
            """
            result = fit_windows_kspace(
                R_AA.ravel(), R_BB.ravel(), R_AB.ravel(),
                mask_flat, corr_size, config, pass_idx,
                use_soft_weighting=config.stereo_ensemble_kspace_soft_weighting,
                k_max_cap=config.stereo_ensemble_kspace_k_max_cap,
                predictor_displacements=predictor_displacements,
                return_diagnostics=True,
            )
            gauss_flat, status_flat, _, diagnostics = result

            gauss_2d = gauss_flat.reshape(n_windows, -1)

            # Extract displacement: params[14]-params[12] = mu_x, params[15]-params[13] = mu_y
            center_x = gauss_2d[:, 12]
            center_y = gauss_2d[:, 13]
            peak_x = gauss_2d[:, 14]
            peak_y = gauss_2d[:, 15]
            dx = peak_x - center_x  # displacement in dewarped pixels
            dy = peak_y - center_y

            # Extract spreads (variance of displacement distribution)
            sig_AB_x = gauss_2d[:, 9]   # Sigma_xx (total: particle + turbulence)
            sig_AB_y = gauss_2d[:, 10]  # Sigma_yy
            sig_AB_xy = gauss_2d[:, 11] # Sigma_xy
            sig_A_x = gauss_2d[:, 6]    # Particle image spread (NaN for kspace)
            sig_A_y = gauss_2d[:, 7]
            sig_A_xy = gauss_2d[:, 8]

            # Peak height
            peakheight = gauss_2d[:, 2]

            # N0 from Stage 1 diagnostics (for CoC noise model)
            N0 = diagnostics[:, 1] if diagnostics is not None else np.zeros(n_windows)

            n_ok = np.sum(status_flat == 0)
            logger.info(f"  {cam_label}: {n_ok}/{n_windows} windows fitted OK")

            return {
                "dx": dx, "dy": dy,
                "sig_AB_x": sig_AB_x, "sig_AB_y": sig_AB_y, "sig_AB_xy": sig_AB_xy,
                "sig_A_x": sig_A_x, "sig_A_y": sig_A_y, "sig_A_xy": sig_A_xy,
                "peakheight": peakheight, "status": status_flat,
                "N0": N0,
            }

        cam1_fit = _fit_camera(cam1_AA, cam1_BB, cam1_AB, "cam1")
        cam2_fit = _fit_camera(cam2_AA, cam2_BB, cam2_AB, "cam2")

        # ── Step 4b: Fit auto-correlations for particle width S ──────
        # Geometric mean sqrt(R_AA × R_BB) gives the particle image
        # auto-correlation shape. Fit as 2D Gaussian to get S (particle spread).
        S_cam1 = self._fit_autocorr_particle_width(
            cam1_AA_raw, cam1_BB_raw, corr_h, corr_w, n_windows, mask_flat)
        S_cam2 = self._fit_autocorr_particle_width(
            cam2_AA_raw, cam2_BB_raw, corr_h, corr_w, n_windows, mask_flat)
        # Average both cameras for robustness
        S_xx = (S_cam1["spread_xx"] + S_cam2["spread_xx"]) / 2.0
        S_yy = (S_cam1["spread_yy"] + S_cam2["spread_yy"]) / 2.0
        S_xy = (S_cam1["spread_xy"] + S_cam2["spread_xy"]) / 2.0
        logger.info(f"  Particle width S: median Sxx={np.median(S_xx):.4f}, "
                    f"Syy={np.median(S_yy):.4f}")

        # ── Step 5: Fit CoC directly (spatial domain Gaussian) ────────
        # The CoC plane IS Gaussian. Fit it directly to get total spread
        # (particle + turbulence). Particle cancels algebraically in Step 7.
        # See validation/coc_implicit_model_learnings.md for why we don't
        # use the implicit F_ref model.
        coc_spreads = self._fit_coc_spatial_gaussian(
            coc_avg, coc_h, coc_w, n_windows, mask_flat)
        n_coc_ok = np.sum(coc_spreads["status"] == 0)
        logger.info(f"  CoC spatial Gaussian fit: {n_coc_ok}/{n_windows} windows OK")

        # ── Step 6: 3D velocity reconstruction (dewarped pixels) ────────
        d1_x = cam1_fit["dx"].reshape(n_win_y, n_win_x)
        d1_y = cam1_fit["dy"].reshape(n_win_y, n_win_x)
        d2_x = cam2_fit["dx"].reshape(n_win_y, n_win_x)
        d2_y = cam2_fit["dy"].reshape(n_win_y, n_win_x)

        # Add back predictor (same as standard ensemble).
        # The predictor was remapped from the previous pass grid to the current
        # pass grid by _get_im_mesh() during correlation. The remapped predictor
        # is stored at the current pass's window centers — we need to sample it
        # at those positions. The smoothed_predictor in pass_data is on the
        # PREVIOUS pass grid and cannot be added directly.
        # For the first implementation, we use the per-camera displacement fields
        # which already include the predictor effect (the correlation measures
        # residual displacement after predictor warping, so total = residual + predictor).
        # The standard ensemble does: ux_mat += smoothed_pred (on the SAME grid).
        # For stereo, the predictor is shared but the grids differ between passes.
        # We reconstruct the predictor on the current grid by remapping.
        if pass_idx > 0 and "smoothed_predictor" in pass_data and pass_data["smoothed_predictor"] is not None:
            import cv2
            smoothed_pred = pass_data["smoothed_predictor"]
            if smoothed_pred.ndim == 3 and smoothed_pred.shape[2] == 2:
                # Remap predictor from previous pass grid to current pass grid
                # using the same interpolation as _get_im_mesh
                prev_ny, prev_nx = smoothed_pred.shape[:2]
                cur_ny, cur_nx = n_win_y, n_win_x

                if prev_ny == cur_ny and prev_nx == cur_nx:
                    # Same grid size — direct add
                    pred_uy = smoothed_pred[:, :, 0]
                    pred_ux = smoothed_pred[:, :, 1]
                else:
                    # Different grid — bilinear resize
                    pred_uy = cv2.resize(
                        smoothed_pred[:, :, 0].astype(np.float32),
                        (cur_nx, cur_ny),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    pred_ux = cv2.resize(
                        smoothed_pred[:, :, 1].astype(np.float32),
                        (cur_nx, cur_ny),
                        interpolation=cv2.INTER_LINEAR,
                    )

                d1_x = d1_x + pred_ux
                d1_y = d1_y + pred_uy
                d2_x = d2_x + pred_ux
                d2_y = d2_y + pred_uy

        # 3D velocity in dewarped pixels
        ux_px = (d1_x + d2_x) / 2.0         # in-plane x
        uy_px = (d1_y + d2_y) / 2.0         # in-plane y
        uz_px = (d1_x - d2_x) / (2.0 * sin_theta)  # out-of-plane

        # 3/4 window displacement validation
        max_disp_x = 0.75 * corr_w
        max_disp_y = 0.75 * corr_h
        invalid = (
            ~np.isfinite(ux_px) | ~np.isfinite(uy_px) | ~np.isfinite(uz_px)
            | (np.abs(d1_x) > max_disp_x) | (np.abs(d1_y) > max_disp_y)
            | (np.abs(d2_x) > max_disp_x) | (np.abs(d2_y) > max_disp_y)
        )
        ux_px[invalid] = np.nan
        uy_px[invalid] = np.nan
        uz_px[invalid] = np.nan

        # ── Step 7: 6 Reynolds stress extraction ────────────────────────
        #
        # Σ₁₁, Σ₂₂: particle-free turbulence spreads from per-camera T(k)
        # spread_C: TOTAL CoC spread (particle + turbulence) from spatial Gaussian fit
        # S: particle width from auto-correlation fit
        #
        # Algebraic particle cancellation:
        #   Σ₁₂ = (Σ₁₁ + Σ₂₂ + 2S - spread_C) / 2
        #
        # Proof: spread_C = 2S + Σ₁₁ + Σ₂₂ - 2Σ₁₂ (from theory)
        #   → (Σ₁₁ + Σ₂₂ + 2S - 2S - Σ₁₁ - Σ₂₂ + 2Σ₁₂) / 2 = Σ₁₂  ✓

        Sigma_11_xx = cam1_fit["sig_AB_x"].reshape(n_win_y, n_win_x)
        Sigma_11_yy = cam1_fit["sig_AB_y"].reshape(n_win_y, n_win_x)
        Sigma_11_xy = cam1_fit["sig_AB_xy"].reshape(n_win_y, n_win_x)
        Sigma_22_xx = cam2_fit["sig_AB_x"].reshape(n_win_y, n_win_x)
        Sigma_22_yy = cam2_fit["sig_AB_y"].reshape(n_win_y, n_win_x)
        Sigma_22_xy = cam2_fit["sig_AB_xy"].reshape(n_win_y, n_win_x)

        spread_C_xx = coc_spreads["spread_xx"].reshape(n_win_y, n_win_x)
        spread_C_yy = coc_spreads["spread_yy"].reshape(n_win_y, n_win_x)
        spread_C_xy = coc_spreads["spread_xy"].reshape(n_win_y, n_win_x)

        S_xx_2d = S_xx.reshape(n_win_y, n_win_x)
        S_yy_2d = S_yy.reshape(n_win_y, n_win_x)
        S_xy_2d = S_xy.reshape(n_win_y, n_win_x)

        Sigma_12_xx = (Sigma_11_xx + Sigma_22_xx + 2.0 * S_xx_2d - spread_C_xx) / 2.0

        # For backward compatibility, store the particle-free CoC spread too
        Sigma_CoC_xx = spread_C_xx - 2.0 * S_xx_2d
        Sigma_CoC_yy = spread_C_yy - 2.0 * S_yy_2d
        Sigma_CoC_xy = spread_C_xy - 2.0 * S_xy_2d

        # Debug — filter to windows where ALL fits succeeded
        coc_ok = coc_spreads["status"].reshape(n_win_y, n_win_x) == 0
        cam1_ok = cam1_fit["status"].reshape(n_win_y, n_win_x) == 0
        cam2_ok = cam2_fit["status"].reshape(n_win_y, n_win_x) == 0
        valid_mask = coc_ok & cam1_ok & cam2_ok & np.isfinite(Sigma_11_xx) & np.isfinite(spread_C_xx)
        if valid_mask.any():
            logger.info(
                f"  Stress intermediates (median of valid windows):\n"
                f"    Σ₁₁ (cam1 turb):    {np.median(Sigma_11_xx[valid_mask]):.6f}\n"
                f"    Σ₂₂ (cam2 turb):    {np.median(Sigma_22_xx[valid_mask]):.6f}\n"
                f"    S (particle):        {np.median(S_xx_2d[valid_mask]):.6f}\n"
                f"    spread_C (total):    {np.median(spread_C_xx[valid_mask]):.6f}\n"
                f"    Σ_CoC (particle-free): {np.median(Sigma_CoC_xx[valid_mask]):.6f}\n"
                f"    Σ₁₂ = (Σ₁₁+Σ₂₂+2S-C)/2: {np.median(Sigma_12_xx[valid_mask]):.6f}"
            )

        # Standard observables from T(k) turbulence variances
        A = (Sigma_11_xx + Sigma_22_xx) / 2.0  # = R_xx + sin²θ·R_zz

        # CoC decoupling: R_xx and R_zz
        R_xx_px2 = (A + Sigma_12_xx) / 2.0
        R_zz_px2 = np.where(
            sin2_theta > 1e-12,
            (A - Sigma_12_xx) / (2.0 * sin2_theta),
            0.0,
        )
        R_yy_px2 = (Sigma_11_yy + Sigma_22_yy) / 2.0
        R_xy_px2 = (Sigma_11_xy + Sigma_22_xy) / 2.0
        R_xz_px2 = (Sigma_11_xx - Sigma_22_xx) / (4.0 * sin_theta) if sin_theta > 1e-12 else np.zeros_like(A)
        R_yz_px2 = (Sigma_11_xy - Sigma_22_xy) / (2.0 * sin_theta) if sin_theta > 1e-12 else np.zeros_like(A)

        # Clamp negative normal stresses
        R_xx_px2 = np.maximum(R_xx_px2, 0.0)
        R_yy_px2 = np.maximum(R_yy_px2, 0.0)
        R_zz_px2 = np.maximum(R_zz_px2, 0.0)

        # ── Step 8: Convert to physical units ───────────────────────────
        # Velocity: px * mm_per_px / dt → mm/s → × 1e-3 → m/s
        vel_scale = mm_per_px / dt * 1e-3  # dewarped_px → m/s
        ux_ms = ux_px * vel_scale
        uy_ms = uy_px * vel_scale
        uz_ms = uz_px * vel_scale

        # Stresses: px² * (mm_per_px / dt)² → (mm/s)² → × 1e-6 → (m/s)²
        stress_scale = (mm_per_px / dt * 1e-3) ** 2
        UU_stress = R_xx_px2 * stress_scale
        VV_stress = R_yy_px2 * stress_scale
        WW_stress = R_zz_px2 * stress_scale
        UV_stress = R_xy_px2 * stress_scale
        UW_stress = R_xz_px2 * stress_scale
        VW_stress = R_yz_px2 * stress_scale
        Sigma_12_xx_phys = Sigma_12_xx * stress_scale

        # ── Step 9: Outlier detection + infilling ───────────────────────
        nan_reason = np.zeros((n_win_y, n_win_x), dtype=np.int32)

        # Mark masked windows
        if self.vector_masks and pass_idx < len(self.vector_masks):
            vmask = self.vector_masks[pass_idx]
            nan_reason[vmask > 0] = -1

        # Mark invalid displacement windows
        nan_reason[invalid & (nan_reason == 0)] = 3  # big displacement

        # Mark failed fits
        cam1_status = cam1_fit["status"].reshape(n_win_y, n_win_x)
        cam2_status = cam2_fit["status"].reshape(n_win_y, n_win_x)
        coc_status = coc_spreads["status"].reshape(n_win_y, n_win_x)
        fit_failed = (cam1_status != 0) | (cam2_status != 0) | (coc_status != 0)
        nan_reason[fit_failed & (nan_reason == 0)] = 1  # no converge

        # Apply outlier detection on velocity fields
        if config.ensemble_outlier_detection_enabled:
            from pivtools_cli.piv.piv_backend.outlier_detection import apply_outlier_detection
            methods = config.ensemble_outlier_detection_methods
            # Filter out velocity-specific methods not applicable to stereo
            applicable_methods = [
                m for m in methods
                if m.get("type") not in ("peak_mag", "div_vort")
            ]
            if applicable_methods:
                vel_outlier = apply_outlier_detection(ux_ms, uy_ms, applicable_methods)

                nan_reason[vel_outlier & (nan_reason == 0)] = 10

                # NaN-out outlier velocities
                ux_ms[vel_outlier] = np.nan
                uy_ms[vel_outlier] = np.nan
                uz_ms[vel_outlier] = np.nan

        # Infilling
        is_final_pass = (pass_idx == config.stereo_ensemble_num_passes - 1)
        infill_config = (
            config.ensemble_infilling_final_pass if is_final_pass
            else config.ensemble_infilling_mid_pass
        )

        if infill_config.get("enabled", True):
            from pivtools_cli.piv.piv_backend.infilling import apply_infilling

            nan_mask = ~np.isfinite(ux_ms)
            if nan_mask.any():
                ux_ms, uy_ms = apply_infilling(ux_ms, uy_ms, nan_mask, infill_config)
                # apply_infilling requires paired fields; pass uz with a copy
                # (the copy is infilled identically and discarded)
                uz_ms, _ = apply_infilling(uz_ms, uz_ms.copy(), nan_mask, infill_config)

        # ── Step 10: Stress realizability (3×3 positive semi-definite) ──
        if is_final_pass:
            stress_outlier = np.zeros((n_win_y, n_win_x), dtype=bool)

            # Cauchy-Schwarz: UV² <= UU*VV, UW² <= UU*WW, VW² <= VV*WW
            valid_stress = np.isfinite(UU_stress) & np.isfinite(VV_stress) & np.isfinite(WW_stress)
            stress_outlier |= valid_stress & (UV_stress ** 2 > UU_stress * VV_stress)
            stress_outlier |= valid_stress & (UW_stress ** 2 > UU_stress * WW_stress)
            stress_outlier |= valid_stress & (VW_stress ** 2 > VV_stress * WW_stress)

            if stress_outlier.any():
                nan_reason[stress_outlier & (nan_reason == 0)] = 11

                # Infill stress outliers using paired apply_infilling
                if infill_config.get("enabled", True):
                    from pivtools_cli.piv.piv_backend.infilling import apply_infilling

                    # NaN-out stress outliers
                    for field in [UU_stress, VV_stress, WW_stress, UV_stress, UW_stress, VW_stress]:
                        field[stress_outlier] = np.nan

                    stress_mask = ~np.isfinite(UU_stress)
                    if stress_mask.any():
                        # Infill in pairs (apply_infilling expects paired fields)
                        UU_stress, VV_stress = apply_infilling(UU_stress, VV_stress, stress_mask, infill_config)
                        WW_stress, UV_stress = apply_infilling(WW_stress, UV_stress, stress_mask, infill_config)
                        UW_stress, VW_stress = apply_infilling(UW_stress, VW_stress, stress_mask, infill_config)

        # ── Step 11: Predictor extraction for next pass ─────────────────
        # Predictor uses in-plane dewarped pixel displacements (before physical conversion)
        pred_x = None
        pred_y = None
        if pass_idx > 0 and "smoothed_predictor" in pass_data and pass_data["smoothed_predictor"] is not None:
            pred = pass_data["smoothed_predictor"]
            pred_y = pred[:, :, 0].copy()
            pred_x = pred[:, :, 1].copy()

        # Build mask
        b_mask = np.zeros((n_win_y, n_win_x), dtype=np.float64)
        if self.vector_masks and pass_idx < len(self.vector_masks):
            b_mask = self.vector_masks[pass_idx].astype(np.float64)

        # Average peakheight from both cameras
        ph1 = cam1_fit["peakheight"].reshape(n_win_y, n_win_x)
        ph2 = cam2_fit["peakheight"].reshape(n_win_y, n_win_x)
        peakheight = (ph1 + ph2) / 2.0

        # Build result
        result = PIVStereoEnsemblePassResult(
            ux=ux_ms,
            uy=uy_ms,
            uz=uz_ms,
            UU_stress=UU_stress,
            VV_stress=VV_stress,
            WW_stress=WW_stress,
            UV_stress=UV_stress,
            UW_stress=UW_stress,
            VW_stress=VW_stress,
            Sigma_12_xx=Sigma_12_xx_phys,
            d1_x=d1_x,
            d1_y=d1_y,
            d2_x=d2_x,
            d2_y=d2_y,
            peakheight=peakheight,
            nan_reason=nan_reason,
            b_mask=b_mask,
            stereo_angle=sin_theta,
            mm_per_pixel=mm_per_px,
            window_size=tuple(corr_size),
            win_ctrs_x=correlator.win_ctrs_x[pass_idx],
            win_ctrs_y=correlator.win_ctrs_y[pass_idx],
            pred_x=pred_x,
            pred_y=pred_y,
        )

        self.passes_results.append(result)

        # ── Diagnostic save ─────────────────────────────────────────────
        if output_path and config.stereo_ensemble_save_diagnostics:
            self._save_pass_diagnostics(
                pass_idx=pass_idx,
                output_path=output_path,
                pass_data=pass_data,
                N=N,
                # Averaged correlation planes (post bg-subtraction, post-normalization)
                cam1_AB=cam1_AB, cam1_AA=cam1_AA, cam1_BB=cam1_BB,
                cam2_AB=cam2_AB, cam2_AA=cam2_AA, cam2_BB=cam2_BB,
                coc_avg=coc_avg,
                # Per-camera fit results
                cam1_fit=cam1_fit, cam2_fit=cam2_fit,
                # CoC spread fit (implicit model, particle-free)
                coc_spreads=coc_spreads,
                cam1_AB_spread=None,
                cam2_AB_spread=None,
                # Stress intermediates
                Sigma_11_xx=Sigma_11_xx, Sigma_22_xx=Sigma_22_xx,
                Sigma_12_xx=Sigma_12_xx,
                # Velocity in dewarped px
                ux_px=ux_px, uy_px=uy_px, uz_px=uz_px,
                # Geometry
                corr_size=corr_size, coc_size=coc_size,
                # Raw/background diagnostic planes
                diag_coc_raw=diag_coc_raw,
                diag_coc_bg=diag_coc_bg,
                diag_cam1_AB_raw=diag_cam1_AB_raw,
                diag_cam2_AB_raw=diag_cam2_AB_raw,
                diag_cam1_AB_bg=diag_cam1_AB_bg,
                diag_cam2_AB_bg=diag_cam2_AB_bg,
            )

        logger.info(f"Pass {pass_idx + 1}: finalization complete")

        return result

    def _correlate_mean_images(
        self,
        mean_A: np.ndarray,
        mean_B: np.ndarray,
        pass_idx: int,
        correlator,
        config: Config,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Correlate mean images for background subtraction.

        Calls the inner ensemble correlator's triple correlation on mean images
        (N=1) to compute <A>⊗<B> for background subtraction:
        R_ensemble = <A⊗B> - <A>⊗<B>

        Returns (R_AB_bg, R_AA_bg, R_BB_bg) as flat arrays.
        """
        # Apply single-mode padding if needed (must match what the C function expects)
        mean_A_padded = mean_A.copy()
        mean_B_padded = mean_B.copy()
        pass_type = config.stereo_ensemble_type[pass_idx] if pass_idx < len(config.stereo_ensemble_type) else "std"
        if pass_type == "single":
            from pivtools_core.window_utils import apply_single_mode_padding
            win_size = config.stereo_ensemble_window_sizes[pass_idx]
            sum_window = config.stereo_ensemble_sum_window
            mean_A_padded, _ = apply_single_mode_padding(mean_A_padded, win_size, sum_window, pad_value=0)
            mean_B_padded, _ = apply_single_mode_padding(mean_B_padded, win_size, sum_window, pad_value=0)

        # Stack mean images as (1, H, W)
        A_stack = mean_A_padded[np.newaxis].astype(np.float32)
        B_stack = mean_B_padded[np.newaxis].astype(np.float32)

        H, W = mean_A_padded.shape
        image_size = np.array([H, W], dtype=np.int32)

        n_win_y = len(correlator.win_ctrs_y[pass_idx])
        n_win_x = len(correlator.win_ctrs_x[pass_idx])
        n_windows_arr = np.array([n_win_y, n_win_x], dtype=np.int32)
        n_windows = n_win_y * n_win_x

        corr_size = correlator.window_sizes_for_corr[pass_idx]
        n_px_corr = corr_size[0] * corr_size[1]

        padding = correlator.padding_per_pass[pass_idx]
        pad_top, pad_left = padding[0], padding[2]
        win_ctrs_x = (correlator.win_ctrs_x[pass_idx] + pad_left).astype(np.float32)
        win_ctrs_y = (correlator.win_ctrs_y[pass_idx] + pad_top).astype(np.float32)

        if self.vector_masks and pass_idx < len(self.vector_masks):
            mask = self.vector_masks[pass_idx].ravel().astype(np.float32)
        else:
            mask = np.zeros(n_windows, dtype=np.float32)

        comp_size = correlator._inner.window_sizes_for_computation[pass_idx]
        win_size_arr = np.array(comp_size, dtype=np.int32)
        fit_size_arr = np.array(corr_size, dtype=np.int32)

        weight_a = np.ascontiguousarray(correlator._inner.win_weights_A[pass_idx], dtype=np.float32)
        weight_b = np.ascontiguousarray(correlator._inner.win_weights_B[pass_idx], dtype=np.float32)

        # Output buffers
        bg_AB = np.zeros(n_windows * n_px_corr, dtype=np.float32)
        bg_AA = np.zeros(n_windows * n_px_corr, dtype=np.float32)
        bg_BB = np.zeros(n_windows * n_px_corr, dtype=np.float32)

        # Use the EXISTING libbulkxcorr2d triple function for background.
        # The lib's argtypes use np.ctypeslib.ndpointer — pass ndarrays directly.
        correlator._inner.lib.bulkxcorr2d_accumulate_triple(
            np.ascontiguousarray(A_stack, dtype=np.float32),
            np.ascontiguousarray(B_stack, dtype=np.float32),
            np.ascontiguousarray(mask, dtype=np.float32),
            np.ascontiguousarray(image_size, dtype=np.int32),
            1,  # N_images = 1 (single mean image)
            np.ascontiguousarray(win_ctrs_x, dtype=np.float32),
            np.ascontiguousarray(win_ctrs_y, dtype=np.float32),
            np.ascontiguousarray(n_windows_arr, dtype=np.int32),
            weight_a,
            weight_b,
            weight_b,  # auto weight A = symmetric
            weight_b,  # auto weight B = symmetric
            np.ascontiguousarray(win_size_arr, dtype=np.int32),
            np.ascontiguousarray(fit_size_arr, dtype=np.int32),
            bg_AB,
            bg_AA,
            bg_BB,
        )

        return (
            bg_AB.reshape(n_windows, n_px_corr),
            bg_AA.reshape(n_windows, n_px_corr),
            bg_BB.reshape(n_windows, n_px_corr),
        )

    @staticmethod
    def _fit_gaussian_2d(plane_2d, roi_half_size=20):
        """Fit a 2D Gaussian to a correlation plane using scipy.

        Parameters
        ----------
        plane_2d : ndarray, shape (h, w)
            The correlation plane (fftshifted, peak near centre).
        roi_half_size : int
            Half-size of the ROI around the peak for fitting.

        Returns
        -------
        dict with: mu_x, mu_y, spread_xx, spread_yy, spread_xy, amplitude, status
        """
        from scipy.optimize import least_squares as scipy_ls

        h, w = plane_2d.shape
        cy, cx = h // 2, w // 2

        # Normalise to avoid numerical issues with large values (O(10^19))
        plane_max = np.abs(plane_2d).max()
        if plane_max > 0:
            plane_2d = plane_2d / plane_max
        else:
            return {"mu_x": 0, "mu_y": 0, "spread_xx": 0, "spread_yy": 0,
                    "spread_xy": 0, "amplitude": 0, "status": 1}

        # Peak location (sub-pixel via argmax)
        peak_idx = np.argmax(plane_2d)
        peak_y, peak_x = divmod(peak_idx, w)

        # ROI around peak
        x0 = max(0, peak_x - roi_half_size)
        x1 = min(w, peak_x + roi_half_size + 1)
        y0 = max(0, peak_y - roi_half_size)
        y1 = min(h, peak_y + roi_half_size + 1)
        roi = plane_2d[y0:y1, x0:x1].astype(np.float64)

        # Coordinate grids relative to peak
        xg = np.arange(x0, x1, dtype=np.float64) - peak_x
        yg = np.arange(y0, y1, dtype=np.float64) - peak_y
        X, Y = np.meshgrid(xg, yg)

        # Initial guess from weighted moments
        A_init = float(roi.max() - roi.min())
        B_init = float(roi.min())
        wts = np.maximum(roi - B_init, 0)
        total = wts.sum()
        if total > 0:
            mx = float(np.sum(X * wts) / total)
            my = float(np.sum(Y * wts) / total)
            vx = max(float(np.sum((X - mx) ** 2 * wts) / total), 0.5)
            vy = max(float(np.sum((Y - my) ** 2 * wts) / total), 0.5)
            vxy = float(np.sum((X - mx) * (Y - my) * wts) / total)
        else:
            mx, my, vx, vy, vxy = 0.0, 0.0, 4.0, 4.0, 0.0

        p0 = np.array([A_init, B_init, mx, my, vx, vy, vxy])

        def residuals(params):
            A, B, px, py, sxx, syy, sxy = params
            det = sxx * syy - sxy * sxy
            if sxx <= 0 or syy <= 0 or det <= 0:
                return np.full(roi.size, 1e10)
            inv_xx = syy / det
            inv_yy = sxx / det
            inv_xy = -sxy / det
            dx = X - px
            dy = Y - py
            mahal = inv_xx * dx * dx + 2 * inv_xy * dx * dy + inv_yy * dy * dy
            model = A * np.exp(-0.5 * mahal) + B
            return (model - roi).ravel()

        try:
            result = scipy_ls(
                residuals, p0,
                bounds=([0, -np.inf, -roi_half_size, -roi_half_size, 0.1, 0.1, -500],
                        [np.inf, np.inf, roi_half_size, roi_half_size, 500, 500, 500]),
                method="trf", max_nfev=1000,
            )
            A, B, px, py, sxx, syy, sxy = result.x
            status = 0 if result.success else 1
        except Exception:
            px, py, sxx, syy, sxy, A = 0, 0, 0, 0, 0, 0
            status = 1

        return {
            "mu_x": peak_x - cx + px,
            "mu_y": peak_y - cy + py,
            "spread_xx": sxx,
            "spread_yy": syy,
            "spread_xy": sxy,
            "amplitude": A * plane_max,
            "status": status,
        }

    def _fit_coc_spatial_gaussian(self, coc_planes, coc_h, coc_w,
                                   n_windows, mask_flat, roi_half_size=20):
        """Fit each CoC plane as a spatial-domain 2D Gaussian.

        Returns dict with arrays: spread_xx, spread_yy, spread_xy,
                                  center_x, center_y, status
        """
        spread_xx = np.zeros(n_windows, dtype=np.float64)
        spread_yy = np.zeros(n_windows, dtype=np.float64)
        spread_xy = np.zeros(n_windows, dtype=np.float64)
        center_x = np.zeros(n_windows, dtype=np.float64)
        center_y = np.zeros(n_windows, dtype=np.float64)
        status = np.full(n_windows, -1, dtype=np.int32)

        n_ok = 0
        for wi in range(n_windows):
            if mask_flat[wi] == 1:
                continue
            plane = coc_planes[wi].reshape(coc_h, coc_w)
            fit = self._fit_gaussian_2d(plane, roi_half_size)
            spread_xx[wi] = fit["spread_xx"]
            spread_yy[wi] = fit["spread_yy"]
            spread_xy[wi] = fit["spread_xy"]
            center_x[wi] = fit["mu_x"]
            center_y[wi] = fit["mu_y"]
            status[wi] = fit["status"]
            if fit["status"] == 0:
                n_ok += 1

        return {
            "spread_xx": spread_xx,
            "spread_yy": spread_yy,
            "spread_xy": spread_xy,
            "center_x": center_x,
            "center_y": center_y,
            "status": status,
        }

    def _fit_autocorr_particle_width(self, R_AA, R_BB, corr_h, corr_w,
                                      n_windows, mask_flat, roi_half_size=10):
        """Fit geometric mean of auto-correlations to get particle width S.

        The geometric mean sqrt(R_AA × R_BB) is a clean particle
        auto-correlation peak at the origin. Fit as 2D Gaussian.

        Returns dict with arrays: spread_xx, spread_yy, spread_xy, status
        """
        spread_xx = np.zeros(n_windows, dtype=np.float64)
        spread_yy = np.zeros(n_windows, dtype=np.float64)
        spread_xy = np.zeros(n_windows, dtype=np.float64)
        status = np.full(n_windows, -1, dtype=np.int32)

        for wi in range(n_windows):
            if mask_flat[wi] == 1:
                continue
            aa = R_AA[wi].reshape(corr_h, corr_w).astype(np.float64)
            bb = R_BB[wi].reshape(corr_h, corr_w).astype(np.float64)
            # Geometric mean — use abs to handle any small negative values
            # from bg subtraction near edges
            geo = np.sqrt(np.maximum(aa, 0) * np.maximum(bb, 0))
            fit = self._fit_gaussian_2d(geo, roi_half_size)
            spread_xx[wi] = fit["spread_xx"]
            spread_yy[wi] = fit["spread_yy"]
            spread_xy[wi] = fit["spread_xy"]
            status[wi] = fit["status"]

        return {
            "spread_xx": spread_xx,
            "spread_yy": spread_yy,
            "spread_xy": spread_xy,
            "status": status,
        }

    def _cross_correlate_planes(
        self,
        planes1: np.ndarray,
        planes2: np.ndarray,
        corr_h: int,
        corr_w: int,
        coc_h: int,
        coc_w: int,
    ) -> np.ndarray:
        """Cross-correlate two sets of correlation planes (for CoC bg subtraction).

        Plain FFT cross-correlation without mean subtraction — matches the
        C code's xcorr_preplanned.

        Parameters
        ----------
        planes1, planes2 : ndarray, shape (n_windows, corr_h * corr_w)
            Per-window correlation planes from cameras 1 and 2.
        corr_h, corr_w : int
            Dimensions of input correlation planes.
        coc_h, coc_w : int
            Dimensions of output CoC planes (same as corr for now).

        Returns
        -------
        ndarray, shape (n_windows, coc_h * coc_w)
        """
        n_windows = planes1.shape[0]
        result = np.zeros((n_windows, coc_h * coc_w), dtype=np.float32)

        for wi in range(n_windows):
            p1 = planes1[wi].reshape(corr_h, corr_w)
            p2 = planes2[wi].reshape(corr_h, corr_w)

            # Plain FFT cross-correlation (no mean subtraction, no ifftshift)
            F1 = np.fft.fft2(p1)
            F2 = np.fft.fft2(p2)
            coc = np.fft.fftshift(np.real(np.fft.ifft2(F1 * np.conj(F2))))
            result[wi] = coc.ravel().astype(np.float32)

        return result

    def _cross_correlate_planes_meansub(
        self,
        planes1: np.ndarray,
        planes2: np.ndarray,
        corr_h: int,
        corr_w: int,
        coc_h: int,
        coc_w: int,
    ) -> np.ndarray:
        """Cross-correlate two sets of planes with spatial mean subtraction.

        Like _cross_correlate_planes but removes the spatial mean from each
        plane before cross-correlating. This matches the C xcorr_meansub
        behaviour and correctly removes spatially-structured static content
        from the CoC background.

        Parameters
        ----------
        planes1, planes2 : ndarray, shape (n_windows, corr_h * corr_w)
            Per-window correlation planes (mean AB from each camera).
        corr_h, corr_w : int
            Dimensions of input correlation planes.
        coc_h, coc_w : int
            Dimensions of output CoC planes.

        Returns
        -------
        ndarray, shape (n_windows, coc_h * coc_w)
        """
        n_windows = planes1.shape[0]
        result = np.zeros((n_windows, coc_h * coc_w), dtype=np.float32)

        for wi in range(n_windows):
            p1 = planes1[wi].reshape(corr_h, corr_w).astype(np.float64)
            p2 = planes2[wi].reshape(corr_h, corr_w).astype(np.float64)

            # Spatial mean subtraction (matches C xcorr_meansub)
            p1 = p1 - p1.mean()
            p2 = p2 - p2.mean()

            # FFT cross-correlation (no ifftshift — matches C code)
            F1 = np.fft.fft2(p1)
            F2 = np.fft.fft2(p2)
            coc = np.fft.fftshift(np.real(np.fft.ifft2(F1 * np.conj(F2))))
            result[wi] = coc.ravel().astype(np.float32)

        return result

    def _save_pass_diagnostics(
        self,
        pass_idx: int,
        output_path: str,
        pass_data: dict,
        N: int,
        cam1_AB, cam1_AA, cam1_BB,
        cam2_AB, cam2_AA, cam2_BB,
        coc_avg,
        cam1_fit, cam2_fit,
        coc_spreads,
        cam1_AB_spread, cam2_AB_spread,
        Sigma_11_xx, Sigma_22_xx, Sigma_12_xx,
        ux_px, uy_px, uz_px,
        corr_size, coc_size,
        diag_coc_raw=None,
        diag_coc_bg=None,
        diag_cam1_AB_raw=None,
        diag_cam2_AB_raw=None,
        diag_cam1_AB_bg=None,
        diag_cam2_AB_bg=None,
    ):
        """Save comprehensive diagnostics for one pass.

        Saves to: {output_path}/diagnostics_pass_{pass_idx+1}.mat

        Contents:
        - dewarped_cam1_A/B, dewarped_cam2_A/B: first dewarped frame
        - warped_cam1_A/B, warped_cam2_A/B: first warped-dewarped frame (pass>0)
        - cam1_AB/AA/BB_avg, cam2_AB/AA/BB_avg: averaged correlation planes
          (post bg-subtraction, post-normalization) for a sample grid of windows
        - coc_avg_planes: averaged CoC planes for sample windows
        - cam1_dx/dy, cam2_dx/dy: per-camera displacements (raw, before predictor add-back)
        - coc_spread_xx/yy/xy: CoC fitted spreads
        - coc_center_x/y: CoC peak displacement (should be ~0)
        - coc_status: fit status per window
        - Sigma_11_xx, Sigma_22_xx, Sigma_12_xx: variance fields
        - ux_px, uy_px, uz_px: 3D velocity in dewarped pixels
        - predictor_field: predictor used for this pass (if pass > 0)
        """
        import scipy.io
        from pathlib import Path

        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"diagnostics_pass_{pass_idx + 1}.mat"

        n_win_y = pass_data["n_win_y"]
        n_win_x = pass_data["n_win_x"]
        n_windows = n_win_y * n_win_x
        corr_h, corr_w = corr_size
        coc_h, coc_w = coc_size

        diag = {}

        # 1. Dewarped first frame images
        for key in ["diag_dw_cam1_A", "diag_dw_cam1_B",
                     "diag_dw_cam2_A", "diag_dw_cam2_B",
                     "diag_warped_cam1_A", "diag_warped_cam1_B",
                     "diag_warped_cam2_A", "diag_warped_cam2_B"]:
            if key in pass_data and pass_data[key] is not None:
                # Strip 'diag_' prefix for cleaner field names
                save_key = key.replace("diag_", "")
                diag[save_key] = pass_data[key].astype(np.float32)

        # 2. Sample correlation planes (save a grid of windows, not all)
        # Pick up to 25 evenly-spaced windows for inspection
        max_sample = 25
        if n_windows <= max_sample:
            sample_indices = np.arange(n_windows)
        else:
            sample_indices = np.linspace(0, n_windows - 1, max_sample, dtype=int)

        n_sample = len(sample_indices)

        # Per-camera averaged correlation planes (reshaped to per-window 2D)
        for name, planes in [("cam1_AB_avg", cam1_AB), ("cam1_AA_avg", cam1_AA),
                              ("cam1_BB_avg", cam1_BB),
                              ("cam2_AB_avg", cam2_AB), ("cam2_AA_avg", cam2_AA),
                              ("cam2_BB_avg", cam2_BB)]:
            planes_2d = planes.reshape(n_windows, corr_h, corr_w)
            diag[name] = planes_2d[sample_indices].astype(np.float32)

        # CoC averaged planes
        coc_2d = coc_avg.reshape(n_windows, coc_h, coc_w)
        diag["coc_avg_planes"] = coc_2d[sample_indices].astype(np.float32)

        # 3. FFT magnitude of sample CoC windows (for verifying k-space fit)
        try:
            n_fft_samples = min(5, n_sample)
            fft_sample_idx = sample_indices[:n_fft_samples]
            fft_mags = []
            for wi in fft_sample_idx:
                coc_plane = coc_2d[wi]
                # ifftshift → FFT → magnitude
                shifted = np.fft.ifftshift(coc_plane)
                F = np.fft.fft2(shifted)
                F_mag = np.fft.fftshift(np.abs(F))
                fft_mags.append(F_mag)
            diag["coc_fft_magnitude"] = np.array(fft_mags, dtype=np.float32)
            diag["coc_fft_sample_indices"] = fft_sample_idx.astype(np.int32)
        except Exception as e:
            logger.debug(f"Could not compute FFT diagnostics: {e}")

        # 4. Per-camera displacements (raw, before predictor add-back)
        diag["cam1_dx_raw"] = cam1_fit["dx"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["cam1_dy_raw"] = cam1_fit["dy"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["cam2_dx_raw"] = cam2_fit["dx"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["cam2_dy_raw"] = cam2_fit["dy"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["cam1_fit_status"] = cam1_fit["status"].reshape(n_win_y, n_win_x).astype(np.int32)
        diag["cam2_fit_status"] = cam2_fit["status"].reshape(n_win_y, n_win_x).astype(np.int32)

        # 5. Unnormalized spread fitting diagnostics (all particle-inclusive)
        diag["coc_spread_xx"] = coc_spreads["spread_xx"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["coc_spread_yy"] = coc_spreads["spread_yy"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["coc_spread_xy"] = coc_spreads["spread_xy"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["coc_center_x"] = coc_spreads["center_x"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["coc_center_y"] = coc_spreads["center_y"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["coc_fit_status"] = coc_spreads["status"].reshape(n_win_y, n_win_x).astype(np.int32)
        # Per-camera AB total spreads (only if available — removed in implicit model)
        if cam1_AB_spread is not None:
            diag["cam1_AB_total_spread_xx"] = cam1_AB_spread["spread_xx"].reshape(n_win_y, n_win_x).astype(np.float32)
            diag["cam1_AB_total_spread_yy"] = cam1_AB_spread["spread_yy"].reshape(n_win_y, n_win_x).astype(np.float32)
        if cam2_AB_spread is not None:
            diag["cam2_AB_total_spread_xx"] = cam2_AB_spread["spread_xx"].reshape(n_win_y, n_win_x).astype(np.float32)
            diag["cam2_AB_total_spread_yy"] = cam2_AB_spread["spread_yy"].reshape(n_win_y, n_win_x).astype(np.float32)

        # 6. Variance fields (dewarped px² units)
        diag["Sigma_11_xx"] = Sigma_11_xx.astype(np.float32)
        diag["Sigma_22_xx"] = Sigma_22_xx.astype(np.float32)
        diag["Sigma_12_xx"] = Sigma_12_xx.astype(np.float32)

        # 7. 3D velocity in dewarped pixels (before physical conversion)
        diag["ux_dewarped_px"] = ux_px.astype(np.float32)
        diag["uy_dewarped_px"] = uy_px.astype(np.float32)
        diag["uz_dewarped_px"] = uz_px.astype(np.float32)

        # 8. Predictor field (if pass > 0)
        if "smoothed_predictor" in pass_data and pass_data["smoothed_predictor"] is not None:
            pred = pass_data["smoothed_predictor"]
            diag["predictor_uy"] = pred[:, :, 0].astype(np.float32)
            diag["predictor_ux"] = pred[:, :, 1].astype(np.float32)

        # 9. Background subtraction diagnostics (raw → bg → clean)
        # These let us inspect whether the CoC bg subtraction is helping/hurting
        if diag_coc_raw is not None:
            coc_raw_2d = diag_coc_raw.reshape(n_windows, coc_h, coc_w)
            diag["coc_raw_planes"] = coc_raw_2d[sample_indices].astype(np.float32)
        if diag_coc_bg is not None:
            coc_bg_2d = diag_coc_bg.reshape(n_windows, coc_h, coc_w)
            diag["coc_bg_planes"] = coc_bg_2d[sample_indices].astype(np.float32)
        if diag_cam1_AB_raw is not None:
            cam1_AB_raw_2d = diag_cam1_AB_raw.reshape(n_windows, corr_h, corr_w)
            diag["cam1_AB_raw_planes"] = cam1_AB_raw_2d[sample_indices].astype(np.float32)
        if diag_cam2_AB_raw is not None:
            cam2_AB_raw_2d = diag_cam2_AB_raw.reshape(n_windows, corr_h, corr_w)
            diag["cam2_AB_raw_planes"] = cam2_AB_raw_2d[sample_indices].astype(np.float32)
        if diag_cam1_AB_bg is not None:
            cam1_AB_bg_2d = diag_cam1_AB_bg.reshape(n_windows, corr_h, corr_w)
            diag["cam1_AB_bg_planes"] = cam1_AB_bg_2d[sample_indices].astype(np.float32)
        if diag_cam2_AB_bg is not None:
            cam2_AB_bg_2d = diag_cam2_AB_bg.reshape(n_windows, corr_h, corr_w)
            diag["cam2_AB_bg_planes"] = cam2_AB_bg_2d[sample_indices].astype(np.float32)

        diag["n_images"] = np.int32(N)
        diag["pass_idx"] = np.int32(pass_idx)
        diag["corr_size"] = np.array(corr_size, dtype=np.int32)
        diag["coc_size"] = np.array(coc_size, dtype=np.int32)
        diag["stereo_angle"] = np.float64(self.stereo_angle)
        diag["mm_per_pixel"] = np.float64(self.mm_per_pixel)
        diag["sample_window_indices"] = sample_indices.astype(np.int32)

        # Per-frame diagnostic planes for one window (all N frames)
        for key in ["diag_perframe_ab1", "diag_perframe_ab2", "diag_perframe_coc"]:
            if key in pass_data and pass_data[key] is not None:
                diag[key] = pass_data[key].astype(np.float32)
        if "diag_perframe_win_idx" in pass_data:
            diag["diag_perframe_win_idx"] = np.int32(pass_data["diag_perframe_win_idx"])

        scipy.io.savemat(str(filepath), diag, do_compression=True)
        logger.info(f"  Saved diagnostics to {filepath}")


def extract_stereo_predictor_field(
    pass_result: PIVStereoEnsemblePassResult,
    mm_per_pixel: float,
    dt: float,
) -> np.ndarray:
    """Extract predictor field from a stereo ensemble pass result.

    Converts physical velocity (m/s) back to dewarped pixel displacements
    for use as predictor in the next pass.

    Returns
    -------
    ndarray, shape (n_win_y, n_win_x, 2) with [uy_px, ux_px] convention.
    """
    # Reverse the physical conversion: m/s → dewarped px
    vel_scale = mm_per_pixel / dt * 1e-3
    ux_px = pass_result.ux / vel_scale
    uy_px = pass_result.uy / vel_scale

    return np.stack([uy_px, ux_px], axis=-1).astype(np.float32)
