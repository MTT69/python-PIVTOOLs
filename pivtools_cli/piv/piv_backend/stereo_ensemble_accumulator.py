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
        # Per-frame autocorr(AB_c) ensemble averages — diagnostic only.
        # No background subtraction or normalisation applied; saved as-is
        # below in Step 5b. Carries no Σ_cc / Σ_12 content (autocorrelation
        # is shift-invariant in the input, so per-frame d_c cancels and
        # the ensemble is a redundant 2S estimator).
        cam1_AB_autocorr = pass_data["cam1_AB_AC_sum"].reshape(n_windows, n_px_coc) / N
        cam2_AB_autocorr = pass_data["cam2_AB_AC_sum"].reshape(n_windows, n_px_coc) / N

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

            # Extract per-window predictor for the noise-PSD fractional
            # displacement (pass > 0). Use the upsampled_predictor view —
            # it is already on the current pass grid (n_win_y, n_win_x, 2).
            # The padded smoothed_predictor must NOT be cv2.resized here:
            # cv2.resize stretches corner-to-corner with no awareness of
            # the pad zone and contaminates the boundary windows.
            upsampled = pass_data.get("upsampled_predictor")
            if upsampled is not None and upsampled.ndim == 3 and upsampled.shape[2] == 2:
                cur_ny = len(win_ctrs_y)
                cur_nx = len(win_ctrs_x)
                if upsampled.shape[:2] != (cur_ny, cur_nx):
                    raise ValueError(
                        f"upsampled_predictor shape {upsampled.shape[:2]} does not "
                        f"match current pass grid ({cur_ny}, {cur_nx}); "
                        f"the predictor build is out of sync with the accumulator."
                    )
                pred_dy_2d = upsampled[:, :, 0]
                pred_dx_2d = upsampled[:, :, 1]
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

        # ── Step 5: Fit CoC k-space transfer function with AC F_ref ──
        # Σ_diff = Σ_11 + Σ_22 − 2·Σ_12 from log-curvature of
        # |F[CoC]| / √(|F[cam1_AB_autocorr]|·|F[cam2_AB_autocorr]|).
        # AC F_ref cancels both particle width AND within-frame variance
        # so the recovered Σ_diff is the clean displacement covariance.
        k_max_cap = config.stereo_ensemble_kspace_k_max_cap or 0.35
        coc_kspace_ac = self._fit_coc_kspace_ac(
            coc_avg, cam1_AB_autocorr, cam2_AB_autocorr,
            coc_h, coc_w, n_windows, mask_flat, k_max=k_max_cap,
        )
        n_coc_ok = int(np.sum(coc_kspace_ac["status"] == 0))
        logger.info(f"  CoC k-space AC fit: {n_coc_ok}/{n_windows} windows OK")

        # ── Step 5b: Save per-window correlation planes (gated) ──────────
        # Honours `stereo_ensemble_piv.store_planes` (falling back to
        # `ensemble_piv.store_planes` via Config). Mirrors the std-ensemble
        # idiom in `single_pass_accumulator.py:1170-1224`. Saves the same
        # planes the fitters consume, plus unnormalised AA/BB for inspecting
        # the particle auto-correlation directly.
        if getattr(config, "stereo_ensemble_store_planes", False) and output_path is not None:
            import scipy.io as _sio
            from pathlib import Path as _Path

            planes_dir = _Path(output_path)
            planes_dir.mkdir(parents=True, exist_ok=True)
            planes_path = planes_dir / f"planes_pass_{pass_idx + 1}.mat"
            planes_dict = {
                # Normalised, bg-subtracted ensemble planes — what the
                # k-space fitter actually sees per camera.
                "cam1_AB": cam1_AB.reshape(n_win_y, n_win_x, corr_h, corr_w).astype(np.float32),
                "cam1_AA": cam1_AA.reshape(n_win_y, n_win_x, corr_h, corr_w).astype(np.float32),
                "cam1_BB": cam1_BB.reshape(n_win_y, n_win_x, corr_h, corr_w).astype(np.float32),
                "cam2_AB": cam2_AB.reshape(n_win_y, n_win_x, corr_h, corr_w).astype(np.float32),
                "cam2_AA": cam2_AA.reshape(n_win_y, n_win_x, corr_h, corr_w).astype(np.float32),
                "cam2_BB": cam2_BB.reshape(n_win_y, n_win_x, corr_h, corr_w).astype(np.float32),
                # Unnormalised AA / BB (post-bg) — what the particle-width fit
                # consumes; shows the absolute auto-correlation amplitude.
                "cam1_AA_raw": cam1_AA_raw.reshape(n_win_y, n_win_x, corr_h, corr_w).astype(np.float32),
                "cam1_BB_raw": cam1_BB_raw.reshape(n_win_y, n_win_x, corr_h, corr_w).astype(np.float32),
                "cam2_AA_raw": cam2_AA_raw.reshape(n_win_y, n_win_x, corr_h, corr_w).astype(np.float32),
                "cam2_BB_raw": cam2_BB_raw.reshape(n_win_y, n_win_x, corr_h, corr_w).astype(np.float32),
                # CoC ensemble plane after structured-bg subtraction.
                "coc_avg": coc_avg.reshape(n_win_y, n_win_x, coc_h, coc_w).astype(np.float32),
                # Diagnostic: ensemble averages of per-frame autocorr(AB_c).
                # Per-frame autocorr(AB_c) = G(η, 2S) (shift-invariant in d_c),
                # so each plane's peak sits at the origin with width ~2S and
                # carries no Σ_cc / Σ_12 content. Useful only as a redundant
                # consistency check on the particle width S — does NOT enter
                # the Σ_12 extraction identity.
                "cam1_AB_autocorr": cam1_AB_autocorr.reshape(n_win_y, n_win_x, coc_h, coc_w).astype(np.float32),
                "cam2_AB_autocorr": cam2_AB_autocorr.reshape(n_win_y, n_win_x, coc_h, coc_w).astype(np.float32),
                # Geometry + provenance.
                "corr_size": np.array(corr_size, dtype=np.int32),
                "coc_size": np.array(coc_size, dtype=np.int32),
                "n_win_y": np.int32(n_win_y),
                "n_win_x": np.int32(n_win_x),
                "pass_idx": np.int32(pass_idx),
                "n_images": np.int32(N),
            }
            _sio.savemat(str(planes_path), planes_dict, do_compression=True)
            logger.info(f"  Saved per-window correlation planes → {planes_path.name}")

        # ── Step 6: 3D velocity reconstruction (dewarped pixels) ────────
        d1_x = cam1_fit["dx"].reshape(n_win_y, n_win_x)
        d1_y = cam1_fit["dy"].reshape(n_win_y, n_win_x)
        d2_x = cam2_fit["dx"].reshape(n_win_y, n_win_x)
        d2_y = cam2_fit["dy"].reshape(n_win_y, n_win_x)

        # Add back predictor on the current pass grid.
        #
        # The stereo correlator publishes two predictor views in pass_data
        # (see cpu_stereo_ensemble.py:538-539):
        #   "smoothed_predictor"  = delta_ab_old, the PADDED previous-pass
        #                            grid (used for the cv2.remap input).
        #   "upsampled_predictor" = delta_ab_pred, the post-remap predictor
        #                            already on the CURRENT pass grid.
        # Use the upsampled view directly. It is the same shape as the
        # per-camera residuals (n_win_y, n_win_x, 2) and is already aligned
        # to the current pass's window centres, so no resize is needed.
        # Resizing the padded smoothed_predictor with cv2.resize stretches
        # the input corner-to-corner with no awareness of the pad zone,
        # producing a 2-cell underprediction band on every boundary of
        # the result.
        if pass_idx > 0 and pass_data.get("upsampled_predictor") is not None:
            upsampled = pass_data["upsampled_predictor"]
            if upsampled.ndim == 3 and upsampled.shape[2] == 2:
                if upsampled.shape[:2] != (n_win_y, n_win_x):
                    raise ValueError(
                        f"upsampled_predictor shape {upsampled.shape[:2]} does not "
                        f"match current pass grid ({n_win_y}, {n_win_x}); "
                        f"the predictor build is out of sync with the accumulator."
                    )
                pred_uy = upsampled[:, :, 0]
                pred_ux = upsampled[:, :, 1]
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
        # Σ₁₁, Σ₂₂: particle-free turbulence spreads from per-camera T(k).
        # Σ_diff:   Σ₁₁ + Σ₂₂ − 2·Σ₁₂, from k-space CoC fit with AC F_ref.
        #
        # Σ_diff is the curvature of log|F[CoC]| − log F_ref_AC over a
        # |k| ≤ k_max ring. The AC reference geom-means |F[cam1_AB_AC]|
        # and |F[cam2_AB_AC]|, so it cancels both the particle image
        # and the within-frame variance — Σ_diff is then the clean
        # cross-camera displacement covariance.
        #
        # Cross-camera covariance:
        #   Σ₁₂ = (Σ₁₁ + Σ₂₂ − Σ_diff) / 2
        #
        # See manual_tools/coc_kspace_vs_gaussian.py:267 (fit) and
        # wiki/concepts/stereo-coc-extraction.md (algebra).

        Sigma_11_xx = cam1_fit["sig_AB_x"].reshape(n_win_y, n_win_x)
        Sigma_11_yy = cam1_fit["sig_AB_y"].reshape(n_win_y, n_win_x)
        Sigma_11_xy = cam1_fit["sig_AB_xy"].reshape(n_win_y, n_win_x)
        Sigma_22_xx = cam2_fit["sig_AB_x"].reshape(n_win_y, n_win_x)
        Sigma_22_yy = cam2_fit["sig_AB_y"].reshape(n_win_y, n_win_x)
        Sigma_22_xy = cam2_fit["sig_AB_xy"].reshape(n_win_y, n_win_x)

        Sigma_diff_xx = coc_kspace_ac["sigma_diff_xx"].reshape(n_win_y, n_win_x)
        Sigma_diff_yy = coc_kspace_ac["sigma_diff_yy"].reshape(n_win_y, n_win_x)
        Sigma_diff_xy = coc_kspace_ac["sigma_diff_xy"].reshape(n_win_y, n_win_x)

        Sigma_12_xx = (Sigma_11_xx + Sigma_22_xx - Sigma_diff_xx) / 2.0
        Sigma_12_yy = (Sigma_11_yy + Sigma_22_yy - Sigma_diff_yy) / 2.0
        Sigma_12_xy = (Sigma_11_xy + Sigma_22_xy - Sigma_diff_xy) / 2.0

        # Debug — filter to windows where ALL fits succeeded
        coc_ok = coc_kspace_ac["status"].reshape(n_win_y, n_win_x) == 0
        cam1_ok = cam1_fit["status"].reshape(n_win_y, n_win_x) == 0
        cam2_ok = cam2_fit["status"].reshape(n_win_y, n_win_x) == 0
        valid_mask = (
            coc_ok & cam1_ok & cam2_ok
            & np.isfinite(Sigma_11_xx) & np.isfinite(Sigma_diff_xx)
        )
        if valid_mask.any():
            logger.info(
                f"  Stress intermediates (median of valid windows):\n"
                f"    Σ₁₁ (cam1 turb):              {np.median(Sigma_11_xx[valid_mask]):.6f}\n"
                f"    Σ₂₂ (cam2 turb):              {np.median(Sigma_22_xx[valid_mask]):.6f}\n"
                f"    Σ_diff (CoC k-space AC):      {np.median(Sigma_diff_xx[valid_mask]):.6f}\n"
                f"    Σ₁₂ = (Σ₁₁+Σ₂₂-Σ_diff)/2:     {np.median(Sigma_12_xx[valid_mask]):.6f}"
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
        coc_status = coc_kspace_ac["status"].reshape(n_win_y, n_win_x)
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

        # Vector-masked windows (out-of-FOV / user-masked) — never infill into
        # them. The new k-space AC fitter writes NaN at these windows, which
        # the stress-realizability infilling below would otherwise extrapolate
        # outwards from valid neighbours, producing saturated values at the
        # FOV boundary. Convention (per user, 2026-04-30): masked stresses
        # match masked velocities → 0, not NaN. Revisit when the codebase
        # moves to NaN-as-mask globally.
        vector_masked = (
            self.vector_masks[pass_idx].astype(bool)
            if self.vector_masks and pass_idx < len(self.vector_masks)
            else np.zeros((n_win_y, n_win_x), dtype=bool)
        )

        # Infilling
        is_final_pass = (pass_idx == config.stereo_ensemble_num_passes - 1)
        infill_config = (
            config.ensemble_infilling_final_pass if is_final_pass
            else config.ensemble_infilling_mid_pass
        )

        if infill_config.get("enabled", True):
            from pivtools_cli.piv.piv_backend.infilling import apply_infilling

            nan_mask = ~np.isfinite(ux_ms) & ~vector_masked
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

                    stress_mask = ~np.isfinite(UU_stress) & ~vector_masked
                    if stress_mask.any():
                        # Infill in pairs (apply_infilling expects paired fields)
                        UU_stress, VV_stress = apply_infilling(UU_stress, VV_stress, stress_mask, infill_config)
                        WW_stress, UV_stress = apply_infilling(WW_stress, UV_stress, stress_mask, infill_config)
                        UW_stress, VW_stress = apply_infilling(UW_stress, VW_stress, stress_mask, infill_config)

        # Force vector-masked windows to 0 across all velocity + stress fields
        # — matches the velocity convention (per-camera fitter writes 0 at
        # skipped windows) and keeps the saved .mat consistent with the
        # b_mask field. Sigma_12_xx_phys also zeroed (was NaN from the AC
        # fitter at masked windows).
        if vector_masked.any():
            for fld in (ux_ms, uy_ms, uz_ms,
                        UU_stress, VV_stress, WW_stress,
                        UV_stress, UW_stress, VW_stress,
                        Sigma_12_xx_phys):
                fld[vector_masked] = 0.0

        # ── Step 11: Predictor extraction for save ──────────────────────
        # Two predictor views are persisted (dewarped pixel units, in-plane):
        #   pred_x/y         — POST-remap, on this pass's grid. Sourced
        #                      from pass_data["upsampled_predictor"]
        #                      (= self._inner.delta_ab_pred). This is the
        #                      field that warped this pass's images.
        #   padded_pred_x/y  — PRE-remap, on previous pass's grid +
        #                      boundary padding. Sourced from
        #                      pass_data["smoothed_predictor"]
        #                      (= self._inner.delta_ab_old). The input
        #                      to the cv2.remap upsampling step.
        # Channel layout in both arrays: [..., 0] = y-component (uy),
        # [..., 1] = x-component (ux).
        pred_x = None
        pred_y = None
        padded_pred_x = None
        padded_pred_y = None
        if pass_idx > 0:
            upsampled = pass_data.get("upsampled_predictor")
            if upsampled is not None:
                pred_y = upsampled[:, :, 0].copy()
                pred_x = upsampled[:, :, 1].copy()
            smoothed = pass_data.get("smoothed_predictor")
            if smoothed is not None:
                padded_pred_y = smoothed[:, :, 0].copy()
                padded_pred_x = smoothed[:, :, 1].copy()

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
            padded_pred_x=padded_pred_x,
            padded_pred_y=padded_pred_y,
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
                # CoC k-space AC fit (Σ_diff + status + sub-pixel CoC peak)
                coc_kspace_ac=coc_kspace_ac,
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

    def _fit_coc_kspace_ac(
        self,
        coc_planes: np.ndarray,
        cam1_AC_planes: np.ndarray,
        cam2_AC_planes: np.ndarray,
        coc_h: int,
        coc_w: int,
        n_windows: int,
        mask_flat: np.ndarray,
        k_max: float = 0.35,
    ) -> Dict[str, np.ndarray]:
        """Fit Σ_diff per window from the CoC k-space transfer function with AC F_ref.

        Σ_diff = Σ_11 + Σ_22 − 2·Σ_12 is recovered from the log-quadratic
        curvature of |F[CoC]| / F_ref, where the AC reference

            F_ref(k) = √(|F[cam1_AB_autocorr]|(k) · |F[cam2_AB_autocorr]|(k))

        cancels both the particle image width and the within-frame variance,
        leaving Σ_disp uncontaminated. See ``manual_tools/coc_kspace_vs_gaussian.py``
        (``fit_kspace_quadratic``) and ``manual_tools/coc_vs_ab_autocorr_inspector.py``
        (``_kspace_sigma_diff_one_window``) for the reference implementation.

        Parameters
        ----------
        coc_planes, cam1_AC_planes, cam2_AC_planes : ndarray, shape (n_windows, coc_h*coc_w)
            Flat per-window planes. The CoC plane is the ensemble-averaged
            cross-correlation of AB1 and AB2 (after structured-bg subtraction).
            The AC planes are ensemble averages of the per-frame autocorrelations
            of AB1 and AB2 — already centred (peak at the geometric centre).
        mask_flat : ndarray, shape (n_windows,) of int32
            1 = skip the window (status forced to 1), 0 = fit.
        k_max : float
            Fit ring upper bound in cycles per pixel.

        Returns
        -------
        dict
            ``sigma_diff_xx/yy/xy`` (float64, shape (n_windows,)) — variance
            tensor entries of the recovered Σ_diff Gaussian, NaN where the
            fit failed or the window was masked.
            ``status`` (int32, shape (n_windows,)) — 0 if fit succeeded
            (positive-definite Σ_diff), 1 otherwise.
            ``residual_norm`` (float64, shape (n_windows,)) — RMS LSQ
            residual over the |k| ≤ k_max ring.
            ``center_x``, ``center_y`` (float64, shape (n_windows,)) —
            sub-pixel CoC peak offset from the plane centre, computed
            via argmax + parabolic interpolation. Diagnostic only.
        """
        # ── Batched FFT: ifftshift → fft2 → |·| → fftshift ──────────────
        # The accumulator holds planes with the correlation peak at the
        # centre (fftshifted), so we undo the shift before FFT.
        coc_2d = coc_planes.reshape(n_windows, coc_h, coc_w).astype(np.float64)
        ac1_2d = cam1_AC_planes.reshape(n_windows, coc_h, coc_w).astype(np.float64)
        ac2_2d = cam2_AC_planes.reshape(n_windows, coc_h, coc_w).astype(np.float64)

        def _batched_fft_mag(planes: np.ndarray) -> np.ndarray:
            shifted = np.fft.ifftshift(planes, axes=(-2, -1))
            F = np.fft.fft2(shifted)
            return np.fft.fftshift(np.abs(F), axes=(-2, -1))

        F_coc = _batched_fft_mag(coc_2d)
        F_ac1 = _batched_fft_mag(ac1_2d)
        F_ac2 = _batched_fft_mag(ac2_2d)
        F_ref_AC = np.sqrt(F_ac1 * F_ac2)

        eps = 1e-30
        log_T = np.log(np.maximum(F_coc, eps)) - np.log(np.maximum(F_ref_AC, eps))

        # ── k-grid (cycles/pixel, fftshifted) ──────────────────────────
        ky = np.fft.fftshift(np.fft.fftfreq(coc_h))
        kx = np.fft.fftshift(np.fft.fftfreq(coc_w))
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX * KX + KY * KY)
        valid = (K <= k_max) & np.isfinite(K)
        n_valid = int(valid.sum())

        sigma_diff_xx = np.full(n_windows, np.nan, dtype=np.float64)
        sigma_diff_yy = np.full(n_windows, np.nan, dtype=np.float64)
        sigma_diff_xy = np.full(n_windows, np.nan, dtype=np.float64)
        status = np.ones(n_windows, dtype=np.int32)
        residual_norm = np.full(n_windows, np.inf, dtype=np.float64)

        if n_valid >= 8:
            kx_v = KX[valid].astype(np.float64)
            ky_v = KY[valid].astype(np.float64)
            # Design matrix: log T = c + b1·kx² + b2·ky² + b3·kx·ky
            M = np.column_stack([
                np.ones_like(kx_v), kx_v * kx_v, ky_v * ky_v, kx_v * ky_v,
            ])

            log_T_flat = log_T.reshape(n_windows, coc_h * coc_w)
            valid_flat = valid.ravel()
            log_T_v = log_T_flat[:, valid_flat]  # shape (n_windows, n_valid)
            finite_rows = np.all(np.isfinite(log_T_v), axis=1) & (mask_flat != 1)

            inv_2pi2 = 1.0 / (2.0 * np.pi * np.pi)
            inv_4pi2 = 1.0 / (4.0 * np.pi * np.pi)

            if finite_rows.any():
                # numpy ≥ 2 emits benign divide/invalid warnings from
                # pinv/matmul internals even on well-conditioned inputs.
                # Suppress them; failure is caught by the status check.
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    pinv = np.linalg.pinv(M)
                    log_T_v_clean = log_T_v[finite_rows]  # (n_finite, n_valid)
                    coeffs_clean = pinv @ log_T_v_clean.T  # (4, n_finite)
                    b1 = coeffs_clean[1]
                    b2 = coeffs_clean[2]
                    b3 = coeffs_clean[3]
                    sxx_clean = -b1 * inv_2pi2
                    syy_clean = -b2 * inv_2pi2
                    sxy_clean = -b3 * inv_4pi2

                    res = M @ coeffs_clean - log_T_v_clean.T
                    rn_clean = np.linalg.norm(res, axis=0) / np.sqrt(M.shape[0])

                det = sxx_clean * syy_clean - sxy_clean * sxy_clean
                pd = (sxx_clean > 0) & (syy_clean > 0) & (det > 0)
                ok_clean = (
                    pd
                    & np.isfinite(sxx_clean)
                    & np.isfinite(syy_clean)
                    & np.isfinite(sxy_clean)
                )

                # Scatter back into per-window arrays.
                idx = np.where(finite_rows)[0]
                sigma_diff_xx[idx[ok_clean]] = sxx_clean[ok_clean]
                sigma_diff_yy[idx[ok_clean]] = syy_clean[ok_clean]
                sigma_diff_xy[idx[ok_clean]] = sxy_clean[ok_clean]
                status[idx[ok_clean]] = 0
                residual_norm[idx] = rn_clean

        # Sub-pixel CoC peak centre (argmax + 1-D parabolic refinement)
        # — diagnostic only, does not enter the Σ extraction.
        center_x = np.zeros(n_windows, dtype=np.float64)
        center_y = np.zeros(n_windows, dtype=np.float64)
        cx0 = coc_w // 2
        cy0 = coc_h // 2
        flat_max = coc_2d.reshape(n_windows, -1).argmax(axis=1)
        py = (flat_max // coc_w).astype(np.int64)
        px = (flat_max % coc_w).astype(np.int64)
        for wi in range(n_windows):
            iy, ix = int(py[wi]), int(px[wi])
            sub_x = 0.0
            sub_y = 0.0
            if 0 < ix < coc_w - 1:
                a = coc_2d[wi, iy, ix - 1]
                b = coc_2d[wi, iy, ix]
                c = coc_2d[wi, iy, ix + 1]
                denom = a - 2.0 * b + c
                if abs(denom) > 1e-20:
                    sub_x = 0.5 * (a - c) / denom
            if 0 < iy < coc_h - 1:
                a = coc_2d[wi, iy - 1, ix]
                b = coc_2d[wi, iy, ix]
                c = coc_2d[wi, iy + 1, ix]
                denom = a - 2.0 * b + c
                if abs(denom) > 1e-20:
                    sub_y = 0.5 * (a - c) / denom
            center_x[wi] = (ix - cx0) + sub_x
            center_y[wi] = (iy - cy0) + sub_y

        return {
            "sigma_diff_xx": sigma_diff_xx,
            "sigma_diff_yy": sigma_diff_yy,
            "sigma_diff_xy": sigma_diff_xy,
            "status": status,
            "residual_norm": residual_norm,
            "center_x": center_x,
            "center_y": center_y,
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
        coc_kspace_ac,
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
        - Sigma_diff_xx/yy/xy: Σ_11+Σ_22−2Σ_12 from k-space AC fit
        - coc_fit_status: AC fit status per window (0=ok, 1=fail)
        - coc_residual_norm: AC LSQ residual per window
        - coc_center_x/y: sub-pixel CoC peak offset from centre (diagnostic)
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

        # 5. CoC k-space AC fit diagnostics: Σ_diff (xx/yy/xy), status, residual,
        #    sub-pixel CoC peak offset (diagnostic only — not used in extraction).
        diag["Sigma_diff_xx"] = coc_kspace_ac["sigma_diff_xx"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["Sigma_diff_yy"] = coc_kspace_ac["sigma_diff_yy"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["Sigma_diff_xy"] = coc_kspace_ac["sigma_diff_xy"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["coc_fit_status"] = coc_kspace_ac["status"].reshape(n_win_y, n_win_x).astype(np.int32)
        diag["coc_residual_norm"] = coc_kspace_ac["residual_norm"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["coc_center_x"] = coc_kspace_ac["center_x"].reshape(n_win_y, n_win_x).astype(np.float32)
        diag["coc_center_y"] = coc_kspace_ac["center_y"].reshape(n_win_y, n_win_x).astype(np.float32)

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

        # Per-frame raw sub-images at the diag window (post-dewarp, post-warp).
        # Used by `manual_tools/coc_window_experiments.py` to iterate on
        # CoC pre-processing strategies in pure Python without re-running
        # the production pipeline.
        for key in ["diag_perframe_subimg_cam1_A", "diag_perframe_subimg_cam1_B",
                    "diag_perframe_subimg_cam2_A", "diag_perframe_subimg_cam2_B"]:
            if key in pass_data and pass_data[key] is not None:
                diag[key] = pass_data[key].astype(np.float32)
        for key in ["diag_perframe_subimg_window_size",
                    "diag_perframe_subimg_topleft"]:
            if key in pass_data and pass_data[key] is not None:
                diag[key] = pass_data[key].astype(np.int32)

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
