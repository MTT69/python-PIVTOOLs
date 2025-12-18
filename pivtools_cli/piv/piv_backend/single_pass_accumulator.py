"""
Single Pass Accumulator for Ensemble PIV

This module implements the SinglePassAccumulator class for ensemble PIV processing,
handling accumulation of correlation planes and single-pass optimization.
"""

import gc
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pivtools_core.config import Config
from pivtools_cli.piv.piv_result import PIVEnsemblePassResult, PIVEnsembleResult
from pivtools_cli.piv.piv_backend.gaussian_fitting import _get_sigma_from_previous_pass
from pivtools_cli.piv.piv_backend.outlier_detection import apply_outlier_detection
from pivtools_cli.piv.piv_backend.infilling import apply_infilling


class SinglePassAccumulator:
    """
    Accumulates correlation planes for single-pass ensemble PIV.

    Single-pass formula:
        R_AA = <A⋆A> - <A>⋆<A>
        R_BB = <B⋆B> - <B>⋆<B>
        R_AB = <A⋆B> - <A>⋆<B>

    This eliminates the need to store all warped images, only accumulating:
    - sum(A), sum(B): For computing means
    - sum(A⋆A), sum(B⋆B), sum(A⋆B): For correlation planes

    Used by UnifiedBatchPipeline for streaming ensemble PIV processing.
    """

    def __init__(self, config: Config, vector_masks: Optional[list[np.ndarray]] = None):
        self.config = config
        self.vector_masks = vector_masks if vector_masks is not None else []
        self.n_images = 0
        self.passes_data = []
        self.passes_results = []  # Store completed pass results

        H, W = config.image_shape

        # Initialize accumulators for each pass
        for pass_idx in range(config.ensemble_num_passes):
            win_size = config.ensemble_window_sizes[pass_idx]
            overlap = config.ensemble_overlaps[pass_idx]
            runtype = config.ensemble_type[pass_idx]

            # Determine correlation size
            if runtype == 'single':
                corr_size = tuple(config.ensemble_sum_window)
            else:
                corr_size = win_size

            # Compute grid size
            from pivtools_core.window_utils import compute_window_centers, compute_window_centers_single_mode

            if runtype == 'single':
                result = compute_window_centers_single_mode(
                    image_shape=(H, W),
                    window_size=tuple(win_size),
                    sum_window=tuple(config.ensemble_sum_window),
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

            self.passes_data.append({
                # Running sums for mean computation
                "sum_warp_A": np.zeros((H, W), dtype=np.float32),
                "sum_warp_B": np.zeros((H, W), dtype=np.float32),

                # Running correlation plane sums (THREE planes for stacked Gaussian)
                "sum_corr_AA": np.zeros(plane_size, dtype=np.float32),
                "sum_corr_BB": np.zeros(plane_size, dtype=np.float32),
                "sum_corr_AB": np.zeros(plane_size, dtype=np.float32),

                # Grid info
                "n_win_x": n_win_x,
                "n_win_y": n_win_y,
                "corr_size": corr_size,
                "win_size": win_size,
            })

    def accumulate_batch(self, batch_result: dict, pass_idx: int):
        """
        Add batch results to running sums.

        Parameters
        ----------
        batch_result : dict
            Results from correlate_batch_for_accumulation containing sums
        pass_idx : int
            PIV pass index
        """
        pass_data = self.passes_data[pass_idx]

        # Accumulate warped images
        pass_data["sum_warp_A"] += batch_result["warp_A_sum"]
        pass_data["sum_warp_B"] += batch_result["warp_B_sum"]

        # Accumulate correlation planes (NO averaging yet)
        pass_data["sum_corr_AA"] += batch_result["corr_AA_sum"].reshape(-1)
        pass_data["sum_corr_BB"] += batch_result["corr_BB_sum"].reshape(-1)
        pass_data["sum_corr_AB"] += batch_result["corr_AB_sum"].reshape(-1)

        # Store smoothed predictor (for pass > 0)
        # All batches should have the same smoothed predictor, so just overwrite
        if batch_result.get("smoothed_predictor") is not None:
            pass_data["smoothed_predictor"] = batch_result["smoothed_predictor"]
            logging.debug(
                f"Pass {pass_idx + 1}: Stored smoothed predictor in passes_data "
                f"(shape: {batch_result['smoothed_predictor'].shape})"
            )

        self.n_images += batch_result["n_images"]

    def _correlate_mean_images(
        self, A_mean: np.ndarray, B_mean: np.ndarray, pass_idx: int
    ) -> tuple:
        """
        Correlate mean images to compute background correlation.

        This implements the background term in the single-pass formula:
            R_ensemble = <A⋆B> - <A>⋆<B>

        Where <A>⋆<B> is the correlation of the mean images (background).

        Parameters
        ----------
        A_mean : np.ndarray
            Mean of all warped A images, shape (H, W)
        B_mean : np.ndarray
            Mean of all warped B images, shape (H, W)
        pass_idx : int
            PIV pass index

        Returns
        -------
        tuple
            (R_AA_bg, R_BB_bg, R_AB_bg): Background correlation planes
        """
        from pivtools_core.window_utils import apply_single_mode_padding

        # Get configuration for this pass
        win_size = self.config.ensemble_window_sizes[pass_idx]
        corr_size = self.passes_data[pass_idx]["corr_size"]
        n_win_y = self.passes_data[pass_idx]["n_win_y"]
        n_win_x = self.passes_data[pass_idx]["n_win_x"]

        # Check if single mode
        runtype = self.config.ensemble_type[pass_idx]
        is_single_mode = (runtype == 'single')

        total_windows = n_win_y * n_win_x
        H, W = A_mean.shape

        # Apply padding for single mode
        if is_single_mode:
            sum_window = tuple(self.config.ensemble_sum_window)
            A_mean, padding = apply_single_mode_padding(
                A_mean, win_size, sum_window, pad_value=0.0
            )
            B_mean, _ = apply_single_mode_padding(
                B_mean, win_size, sum_window, pad_value=0.0
            )

        # Allocate output correlation planes
        correl_AA_bg = np.ascontiguousarray(
            np.zeros(total_windows * corr_size[0] * corr_size[1], dtype=np.float32)
        )
        correl_BB_bg = np.ascontiguousarray(
            np.zeros(total_windows * corr_size[0] * corr_size[1], dtype=np.float32)
        )
        correl_AB_bg = np.ascontiguousarray(
            np.zeros(total_windows * corr_size[0] * corr_size[1], dtype=np.float32)
        )

        # Create temporary correlator to get library and arguments
        from pivtools_cli.piv.piv_backend.factory import make_correlator_backend
        correlator = make_correlator_backend(self.config, ensemble=True)

        # Set up correlation arguments
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
            correl_out,
            point_spread_a,
            point_spread_b,
        ) = correlator._set_lib_arguments_ensemble(
            config=self.config,
            win_size=win_size,
            pass_idx=pass_idx,
            N=1
        )

        # Image size for correlation
        image_size = np.array([A_mean.shape[0], A_mean.shape[1]], dtype=np.int32)

        # Cross-correlation AB
        correlator.lib.bulkxcorr2d(
            np.ascontiguousarray(A_mean, dtype=np.float32),
            np.ascontiguousarray(B_mean, dtype=np.float32),
            b_mask,
            image_size,
            1,
            correlator.win_ctrs_x[pass_idx].astype(np.float32),
            correlator.win_ctrs_y[pass_idx].astype(np.float32),
            n_windows,
            correlator.win_weights_A[pass_idx],
            b_ensemble,
            correlator.win_weights_B[pass_idx],
            win_size_arr,
            int(n_peaks),
            int(i_peak_finder),
            pk_loc_x,
            pk_loc_y,
            pk_height,
            sx,
            sy,
            sxy,
            correl_AB_bg,
        )

        # Auto-correlation AA
        correlator.lib.bulkxcorr2d(
            np.ascontiguousarray(A_mean, dtype=np.float32),
            np.ascontiguousarray(A_mean, dtype=np.float32),
            b_mask,
            image_size,
            1,
            correlator.win_ctrs_x[pass_idx].astype(np.float32),
            correlator.win_ctrs_y[pass_idx].astype(np.float32),
            n_windows,
            correlator.win_weights_A[pass_idx],
            b_ensemble,
            correlator.win_weights_A[pass_idx],
            win_size_arr,
            int(n_peaks),
            int(i_peak_finder),
            pk_loc_x,
            pk_loc_y,
            pk_height,
            sx,
            sy,
            sxy,
            correl_AA_bg,
        )
        logging.info(f"Pass {pass_idx}: AA_bg after bulkxcorr2d: [{correl_AA_bg.min():.3e}, {correl_AA_bg.max():.3e}], has_inf={np.isinf(correl_AA_bg).any()}, has_nan={np.isnan(correl_AA_bg).any()}")

        # Auto-correlation BB
        correlator.lib.bulkxcorr2d(
            np.ascontiguousarray(B_mean, dtype=np.float32),
            np.ascontiguousarray(B_mean, dtype=np.float32),
            b_mask,
            image_size,
            1,
            correlator.win_ctrs_x[pass_idx].astype(np.float32),
            correlator.win_ctrs_y[pass_idx].astype(np.float32),
            n_windows,
            correlator.win_weights_B[pass_idx],
            b_ensemble,
            correlator.win_weights_B[pass_idx],
            win_size_arr,
            int(n_peaks),
            int(i_peak_finder),
            pk_loc_x,
            pk_loc_y,
            pk_height,
            sx,
            sy,
            sxy,
            correl_BB_bg,
        )
        logging.info(f"Pass {pass_idx}: BB_bg after bulkxcorr2d: [{correl_BB_bg.min():.3e}, {correl_BB_bg.max():.3e}], has_inf={np.isinf(correl_BB_bg).any()}, has_nan={np.isnan(correl_BB_bg).any()}")

        logging.info(f"Pass {pass_idx}: Computed background correlations from mean images")

        return correl_AA_bg, correl_BB_bg, correl_AB_bg

    def finalize_pass(
        self, pass_idx: int,
        predictor_field: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None
    ):
        """
        Finalize a single pass with single-pass optimization.

        Uses pure OpenMP parallelization for Gaussian fitting (no Dask overhead).
        The correlation planes are already on the main process after reduction,
        so we call the C library directly with OpenMP parallelization.

        Parameters
        ----------
        pass_idx : int
            Pass index to finalize
        predictor_field : Optional[np.ndarray], default None
            Predictor displacement field used for warping in this pass.
            Shape: (n_win_y, n_win_x, 2) where [:, :, 0] is Y, [:, :, 1] is X.
            For pass > 0, this MUST be provided to add back to fitted displacements.
        output_path : Optional[Path], default None
            Directory where debug correlation planes and guesses will be saved.
            If None, uses current working directory.

        Returns
        -------
        PIVEnsemblePassResult
            Result for this pass
        """
        from pivtools_core.window_utils import compute_window_centers, compute_window_centers_single_mode

        logging.info(f"Finalizing pass {pass_idx + 1} with single-pass optimization")

        pass_data = self.passes_data[pass_idx]
        N = self.n_images

        temp_piv_results = PIVEnsembleResult()
        for pr in self.passes_results:
            temp_piv_results.add_pass(pr)

        logging.info(f"Pass {pass_idx + 1}: Applying single-pass optimization")
        # Step 1: Compute mean warped images
        A_mean = pass_data["sum_warp_A"] / N
        B_mean = pass_data["sum_warp_B"] / N

        # Step 2: Compute average correlation planes (RAW with background)
        R_AA_raw = pass_data["sum_corr_AA"] / N
        R_BB_raw = pass_data["sum_corr_BB"] / N
        R_AB_raw = pass_data["sum_corr_AB"] / N


        # Step 3: Correlate means for background subtraction
        R_AA_bg, R_BB_bg, R_AB_bg = self._correlate_mean_images(A_mean, B_mean, pass_idx)

        # Step 4: Background subtraction (SINGLE-PASS OPTIMIZATION)
        #         R_ensemble = <A⋆B> - <A>⋆<B>
        R_AA_ensemble = R_AA_raw - R_AA_bg
        R_BB_ensemble = R_BB_raw - R_BB_bg
        R_AB_ensemble = R_AB_raw - R_AB_bg


        # Step 5: Get configuration for this pass
        win_size = pass_data["win_size"]
        corr_size = pass_data["corr_size"]
        n_win_y = pass_data["n_win_y"]
        n_win_x = pass_data["n_win_x"]
        total_windows = n_win_y * n_win_x
        
        # Step 6: Perform distributed Gaussian fitting

        # Get sigma values from previous pass (if applicable)
        # For pass 0: All None (sigmas computed from HWHM in _build_initial_guess)
        # For pass > 0: Interpolated from previous pass after outlier detection & infilling
        # Returns dict with keys: sig_AB_x, sig_AB_y, sig_AB_xy, sig_A_x, sig_A_y, sig_A_xy
        sigma_dict = _get_sigma_from_previous_pass(
            pass_idx, total_windows, self.config, temp_piv_results,
            n_win_x, n_win_y
        )

        # Flatten mask for fitting
        if self.vector_masks and pass_idx < len(self.vector_masks):
            mask_flat = self.vector_masks[pass_idx].ravel(order='C').astype(bool)
        else:
            mask_flat = np.zeros(total_windows, dtype=bool)

        # Pure OpenMP Gaussian fitting (no Dask overhead)
        # The C library uses OpenMP internally for parallel fitting
        from pivtools_cli.piv.piv_backend.gaussian_fitting import fit_windows_openmp

        logging.info(f"Pass {pass_idx + 1}: Starting OpenMP Gaussian fitting...")

        gauss_flat, status_flat, initial_guess_flat = fit_windows_openmp(
            R_AA_ensemble,
            R_BB_ensemble,
            R_AB_ensemble,
            mask_flat,
            sigma_dict,
            corr_size,
            self.config,
            pass_idx,
            # num_threads=None means use ALL CPU cores (not limited by Dask worker settings)
            # This is appropriate because finalize_pass() runs on the main process
            # after all Dask workers have returned their results
            num_threads=None,
        )

        # Release large arrays after fitting
        if not (hasattr(self.config, 'ensemble_store_planes') and self.config.ensemble_store_planes):
            del R_AA_ensemble, R_BB_ensemble, R_AB_ensemble
            gc.collect()

        gauss_results = gauss_flat.reshape(n_win_y, n_win_x, -1)
        statuses = status_flat.reshape(n_win_y, n_win_x)
        initial_guesses = initial_guess_flat.reshape(n_win_y, n_win_x, -1)

        # Calculate success rate excluding masked vectors
        # Status -1 indicates masked/skipped windows (not fitted)
        # Status 0 indicates successful fit
        non_masked_windows = np.sum(statuses != -1)
        successful_fits = np.sum(statuses == 0)
        if non_masked_windows > 0:
            success_rate = successful_fits / non_masked_windows
            logging.info(
                f"Pass {pass_idx + 1}: Gaussian fitting success rate: {success_rate:.1%} "
                f"({successful_fits}/{non_masked_windows} non-masked windows)"
            )
        else:
            logging.warning(f"Pass {pass_idx + 1}: All windows masked, no fitting performed")

        # Step 7: Extract velocities from fitted parametes

        # Determine correlation size for grid
        runtype = self.config.ensemble_type[pass_idx]
        if runtype == 'single':
            grid_result = compute_window_centers_single_mode(
                image_shape=self.config.image_shape,
                window_size=tuple(win_size),
                sum_window=tuple(self.config.ensemble_sum_window),
                overlap=self.config.ensemble_overlaps[pass_idx],
                validate=True,
            )
        else:
            grid_result = compute_window_centers(
                image_shape=self.config.image_shape,
                window_size=tuple(win_size),
                overlap=self.config.ensemble_overlaps[pass_idx],
                validate=True,
            )

        # Extract velocity components and stresses from Gaussian parameters
        # gauss_results has shape (n_win_y, n_win_x, 13)
        #
        # CORRECT Parameter ordering from marquadt_gaussian.c:
        # [0] amp_A, [1] amp_B, [2] amp_AB,
        # [3] sx_A, [4] sy_A, [5] sxy_A,
        # [6] sx_AB, [7] sy_AB, [8] sxy_AB,
        # [9] x0_A, [10] y0_A,
        # [11] x0_AB, [12] y0_AB
        #
        # Displacement is computed from peak positions (x0_AB, y0_AB) relative to window center

        # Get window center (zero displacement location)
        win_center_x = corr_size[1] / 2.0 + 1
        win_center_y = corr_size[0] / 2.0 + 1

        # Extract peak positions from fitted Gaussian centers (16-param layout)
        x0_AB = gauss_results[:, :, 14].astype(np.float32)  # X position of AB peak
        y0_AB = gauss_results[:, :, 15].astype(np.float32)  # Y position of AB peak
        logging.info(f"x0_AB:{x0_AB}")
        logging.info(f"y0_AB:{y0_AB}")
        # Compute displacements as offset from window center
        ux_mat = x0_AB - win_center_x  # X displacement in pixels
        uy_mat = y0_AB - win_center_y  # Y displacement in pixels

        # =========================================================
        # DISPLACEMENT VALIDATION: 3/4 Window Rule
        # =========================================================
        # Displacements larger than 3/4 of the window size are physically
        # implausible and indicate fitting failures. Set to NaN.
        max_disp_x = 0.75 * corr_size[1]
        max_disp_y = 0.75 * corr_size[0]

        # Check for invalid displacements (inf, nan, or > 3/4 window)
        invalid_disp = (
            ~np.isfinite(ux_mat) | ~np.isfinite(uy_mat) |
            (np.abs(ux_mat) > max_disp_x) | (np.abs(uy_mat) > max_disp_y)
        )
        n_invalid = invalid_disp.sum()
        if n_invalid > 0:
            logging.warning(
                f"Pass {pass_idx + 1}: {n_invalid} vectors exceed 3/4 window rule "
                f"or have inf/nan - setting to NaN"
            )
            ux_mat[invalid_disp] = np.nan
            uy_mat[invalid_disp] = np.nan
            # Mark as failed in statuses
            statuses[invalid_disp] = 6  # 6 = displacement rule violation

        if pass_idx > 0:
            # Use the SMOOTHED predictor that was actually used for image warping
            # This is stored in passes_data[pass_idx] during accumulate_batch
            pass_data = self.passes_data[pass_idx]
            if "smoothed_predictor" in pass_data and pass_data["smoothed_predictor"] is not None:
                smoothed_pred = pass_data["smoothed_predictor"]
                logging.info(
                    f"Pass {pass_idx + 1}: Using smoothed predictor field from image warping"
                )

                # smoothed_pred is already on the window grid from _get_im_mesh
                # Shape: (n_win_y, n_win_x, 2) where [:,:,0]=Y, [:,:,1]=X
                ux_mat += smoothed_pred[:, :, 1]  # Add X-displacement
                uy_mat += smoothed_pred[:, :, 0]  # Add Y-displacement
                # Note: Final displacement range is logged after outlier detection/infilling
            else:
                logging.warning(
                    f"Pass {pass_idx + 1}: No smoothed predictor found! "
                    f"This will result in incorrect absolute displacements. "
                    f"Residual displacements will be returned without predictor correction."
                )

        # =========================================================
        # Extract Gaussian parameters with overflow protection
        # =========================================================
        # Clamp to reasonable ranges before float32 cast to prevent overflow
        MAX_AMP = 1e10   # Max reasonable amplitude
        MAX_SIGMA = 1e6  # Max reasonable variance

        def safe_extract(arr, max_val, fill_invalid=0.0):
            """Extract and clamp array, replacing non-finite with fill value."""
            result = np.clip(arr, -max_val, max_val)
            result = np.where(np.isfinite(result), result, fill_invalid)
            return result.astype(np.float32)

        # Amplitudes (positive values expected)
        amp_A = safe_extract(gauss_results[:, :, 0], MAX_AMP, 0.0)
        amp_B = safe_extract(gauss_results[:, :, 1], MAX_AMP, 0.0)
        amp_AB = safe_extract(gauss_results[:, :, 2], MAX_AMP, 0.0)

        # Normalized peak height: AB / sqrt(A * B), clamped to [0, 1]
        geom_mean = np.sqrt(np.maximum(amp_A * amp_B, 1e-12))
        peakheight = np.clip(amp_AB / geom_mean, 0.0, 1.0).astype(np.float32)

        # Gaussian offset terms (can be negative after background subtraction)
        c_A = safe_extract(gauss_results[:, :, 3], MAX_AMP, 0.0)
        c_B = safe_extract(gauss_results[:, :, 4], MAX_AMP, 0.0)
        c_AB = safe_extract(gauss_results[:, :, 5], MAX_AMP, 0.0)

        # Gaussian widths for A autocorrelation (variances, must be positive)
        sig_A_x = safe_extract(gauss_results[:, :, 6], MAX_SIGMA, 0.0)
        sig_A_y = safe_extract(gauss_results[:, :, 7], MAX_SIGMA, 0.0)
        sig_A_xy = safe_extract(gauss_results[:, :, 8], MAX_SIGMA, 0.0)

        # Gaussian widths for AB cross-correlation (predictor displacement uncertainty)
        sig_AB_x = safe_extract(gauss_results[:, :, 9], MAX_SIGMA, 0.0)
        sig_AB_y = safe_extract(gauss_results[:, :, 10], MAX_SIGMA, 0.0)
        sig_AB_xy = safe_extract(gauss_results[:, :, 11], MAX_SIGMA, 0.0)

        UU_stress = sig_AB_x
        VV_stress = sig_AB_y
        UV_stress = sig_AB_xy

        # =========================================================
        # STEP 7a: Apply Vector Mask FIRST (before outlier detection)
        # =========================================================
        # This matches instantaneous behavior: masked regions are set to zero
        # and excluded from outlier detection
        nan_reason = statuses.astype(np.int32)
        vector_mask = None
        if self.vector_masks and pass_idx < len(self.vector_masks):
            vector_mask = self.vector_masks[pass_idx]

        # Ensure vector_mask is always an array (even if no masking enabled)
        if vector_mask is None:
            vector_mask = np.zeros((n_win_y, n_win_x), dtype=bool)

        if vector_mask is not None:
            ux_mat[vector_mask] = 0.0
            uy_mat[vector_mask] = 0.0
            UU_stress[vector_mask] = 0.0
            VV_stress[vector_mask] = 0.0
            UV_stress[vector_mask] = 0.0
            peakheight[vector_mask] = 0.0
            sig_A_x[vector_mask] = 0.0
            sig_A_y[vector_mask] = 0.0
            sig_A_xy[vector_mask] = 0.0
            sig_AB_x[vector_mask] = 0.0
            sig_AB_y[vector_mask] = 0.0
            sig_AB_xy[vector_mask] = 0.0

            # Set nan_reason to indicate masked vectors
            nan_reason[vector_mask] = -1  # -1 = masked vector (not correlated)
            logging.info(f"Pass {pass_idx + 1}: {vector_mask.sum()} vectors masked (set to zero)")

        # =========================================================
        # STEP 7b: Outlier Detection and Infilling
        # =========================================================

        # Determine if this is final pass
        is_final_pass = (pass_idx == self.config.ensemble_num_passes - 1)

        # --- Combined Outlier Detection ---
        # Start with fitting failures (statuses != 0 indicates failed fit)
        # Exclude already-masked vectors from outlier detection
        outlier_mask = (statuses != 0)
        if vector_mask is not None:
            outlier_mask = outlier_mask & ~vector_mask  # Don't double-count masked regions

        # Apply additional outlier detection on valid fits if enabled
        if self.config.ensemble_outlier_detection_enabled:
            outlier_methods = self.config.ensemble_outlier_detection_methods
            if outlier_methods:
                # Only apply detection to non-failed, non-masked fits
                valid_for_detection = ~outlier_mask
                if vector_mask is not None:
                    valid_for_detection = valid_for_detection & ~vector_mask
                if valid_for_detection.any():
                    detected_outliers = apply_outlier_detection(
                        ux_mat, uy_mat,
                        outlier_methods,
                        peak_mag=peakheight
                    )
                    # Only mark as outliers within valid detection region
                    outlier_mask |= (detected_outliers & valid_for_detection)

        logging.info(
            f"Pass {pass_idx + 1}: Outlier detection found {outlier_mask.sum()} outliers "
            f"({outlier_mask.sum() / outlier_mask.size * 100:.1f}%)"
        )

        # --- Propagate outlier mask to ALL fields ---
        # Set outlier locations to NaN for all fields
        ux_mat[outlier_mask] = np.nan
        uy_mat[outlier_mask] = np.nan
        UU_stress[outlier_mask] = np.nan
        VV_stress[outlier_mask] = np.nan
        UV_stress[outlier_mask] = np.nan
        sig_A_x[outlier_mask] = np.nan
        sig_A_y[outlier_mask] = np.nan
        sig_A_xy[outlier_mask] = np.nan
        sig_AB_x[outlier_mask] = np.nan
        sig_AB_y[outlier_mask] = np.nan
        sig_AB_xy[outlier_mask] = np.nan
        peakheight[outlier_mask] = np.nan

        # Update nan_reason for detected outliers (code 10 = outlier on valid fit)
        nan_reason[outlier_mask & (statuses == 0)] = 10

        # --- Infilling ---
        infill_mask = outlier_mask.copy()

        if is_final_pass:
            # Final pass: use final_pass config (may be disabled)
            infill_cfg = self.config.ensemble_infilling_final_pass
            if not infill_cfg.get('enabled', True):
                logging.info(f"Pass {pass_idx + 1}: Final pass infilling disabled")
                infill_mask = np.zeros_like(outlier_mask, dtype=bool)  # Skip infilling
        else:
            # Mid-pass: always infill (required for predictor)
            infill_cfg = self.config.ensemble_infilling_mid_pass

        if infill_mask.any():
            logging.info(
                f"Pass {pass_idx + 1}: Infilling {infill_mask.sum()} vectors using "
                f"'{infill_cfg.get('method', 'biharmonic')}'"
            )

            # Infill displacement fields
            ux_mat, uy_mat = apply_infilling(ux_mat, uy_mat, infill_mask, infill_cfg)

            # Infill stress fields
            UU_stress, VV_stress = apply_infilling(UU_stress, VV_stress, infill_mask, infill_cfg)
            # UV_stress needs special handling (paired with zero array)
            UV_temp = np.zeros_like(UV_stress)
            UV_stress, _ = apply_infilling(UV_stress, UV_temp, infill_mask, infill_cfg)

            # Infill sigma fields (A autocorrelation)
            sig_A_x, sig_A_y = apply_infilling(sig_A_x, sig_A_y, infill_mask, infill_cfg)
            sig_A_xy_temp = np.zeros_like(sig_A_xy)
            sig_A_xy, _ = apply_infilling(sig_A_xy, sig_A_xy_temp, infill_mask, infill_cfg)

            # Infill sigma fields (AB cross-correlation)
            sig_AB_x, sig_AB_y = apply_infilling(sig_AB_x, sig_AB_y, infill_mask, infill_cfg)
            sig_AB_xy_temp = np.zeros_like(sig_AB_xy)
            sig_AB_xy, _ = apply_infilling(sig_AB_xy, sig_AB_xy_temp, infill_mask, infill_cfg)

            # Infill peakheight (paired with zero array)
            peakheight_temp = np.zeros_like(peakheight)
            peakheight, _ = apply_infilling(peakheight, peakheight_temp, infill_mask, infill_cfg)

        logging.info(
            f"Pass {pass_idx + 1}: Post-processing complete. "
            f"ux range: [{np.nanmin(ux_mat):.3f}, {np.nanmax(ux_mat):.3f}], "
            f"uy range: [{np.nanmin(uy_mat):.3f}, {np.nanmax(uy_mat):.3f}]"
        )

        # Extract predictor field components if available
        # Use the SMOOTHED predictor that was actually used for warping
        pred_x = None
        pred_y = None
        if pass_idx > 0 and "smoothed_predictor" in pass_data and pass_data["smoothed_predictor"] is not None:
            smoothed_pred = pass_data["smoothed_predictor"]
            logging.info(
                f"Pass {pass_idx + 1}: Storing SMOOTHED predictor field in pass result "
                f"(shape: {smoothed_pred.shape})"
            )
            pred_y = smoothed_pred[:, :, 0].copy()  # Y component
            pred_x = smoothed_pred[:, :, 1].copy()  # X component
        elif predictor_field is not None:
            # Fallback to raw predictor if smoothed not available (shouldn't happen)
            logging.warning(
                f"Pass {pass_idx + 1}: Smoothed predictor not available, "
                f"falling back to raw predictor field"
            )
            pred_y = predictor_field[:, :, 0].copy()  # Y component
            pred_x = predictor_field[:, :, 1].copy()  # X component

        # Create pass result
        pass_result = PIVEnsemblePassResult(
            ux_mat=ux_mat,
            uy_mat=uy_mat,
            UU_stress=UU_stress,
            VV_stress=VV_stress,
            UV_stress=UV_stress,
            peakheight=peakheight,
            nan_reason=nan_reason,
            sig_AB_x=sig_AB_x,
            sig_AB_y=sig_AB_y,
            sig_AB_xy=sig_AB_xy,
            sig_A_x=sig_A_x,
            sig_A_y=sig_A_y,
            sig_A_xy=sig_A_xy,
            c_A=c_A,
            c_B=c_B,
            c_AB=c_AB,
            b_mask=vector_mask,
            pred_x=pred_x,
            pred_y=pred_y,
            window_size=tuple(win_size),
            win_ctrs_x=grid_result.win_ctrs_x,
            win_ctrs_y=grid_result.win_ctrs_y,
        )

        # Store result in accumulator
        self.passes_results.append(pass_result)

        # Save correlation planes if store_planes is enabled
        if hasattr(self.config, 'ensemble_store_planes') and self.config.ensemble_store_planes:
            try:
                from pathlib import Path
                from scipy.io import savemat
                import os
                if output_path is not None:
                    outdir = Path(output_path)
                else:
                    outdir = Path(os.getcwd())
                outdir.mkdir(parents=True, exist_ok=True)

                # Create correlator to get window weights
                from pivtools_cli.piv.piv_backend.factory import make_correlator_backend
                correlator_for_weights = make_correlator_backend(self.config, ensemble=True)

                # Save correlation planes in 4D format (n_win_y, n_win_x, corr_h, corr_w)
                planes_dict = {
                    'AA': R_AA_ensemble.reshape(n_win_y, n_win_x, corr_size[0], corr_size[1]),
                    'BB': R_BB_ensemble.reshape(n_win_y, n_win_x, corr_size[0], corr_size[1]),
                    'AB': R_AB_ensemble.reshape(n_win_y, n_win_x, corr_size[0], corr_size[1]),
                    'gauss_results': gauss_results,  # All fitted parameters
                    'initial_guesses': initial_guesses,  # Initial guess parameters for fitting
                    'corr_size': corr_size,
                    'n_win_y': n_win_y,
                    'n_win_x': n_win_x,
                    'pass_idx': pass_idx,
                    # Window weights used in cross-correlation
                    'win_weight_A': correlator_for_weights.win_weights_A[pass_idx],
                    'win_weight_B': correlator_for_weights.win_weights_B[pass_idx],
                }

                savemat(
                    outdir / f"planes_pass_{pass_idx + 1}.mat",
                    planes_dict,
                    do_compression=True
                )
                logging.info(f"Pass {pass_idx + 1}: Saved correlation planes to {outdir}/planes_pass_{pass_idx + 1}.mat")
            except Exception as e:
                logging.warning(f"Pass {pass_idx + 1}: Failed to save correlation planes: {e}")

        logging.info(f"Pass {pass_idx + 1}: Finalization complete")

        return pass_result

    def get_ensemble_result(self) -> PIVEnsembleResult:
        """
        Get final ensemble result with all passes.

        Returns
        -------
        PIVEnsembleResult
            Complete ensemble PIV result with all passes
        """
        from pivtools_cli.piv.piv_result import PIVEnsembleResult

        piv_results = PIVEnsembleResult()
        for pass_result in self.passes_results:
            piv_results.add_pass(pass_result)

        logging.info(f"Assembled {len(self.passes_results)} ensemble passes")
        return piv_results

    def clear_pass_data(self, pass_idx: int):
        """
        Clear accumulated data for a specific pass to free memory.

        This is called after a pass has been finalized and saved to disk,
        allowing the memory to be reclaimed. The pass result is kept in
        passes_results for assembling the final output.

        Parameters
        ----------
        pass_idx : int
            Pass index to clear
        """
        if pass_idx >= len(self.passes_data):
            logging.warning(f"Cannot clear pass {pass_idx}: index out of range")
            return

        pass_data = self.passes_data[pass_idx]

        # Get memory usage before clearing
        mem_before = (
            pass_data["sum_warp_A"].nbytes +
            pass_data["sum_warp_B"].nbytes +
            pass_data["sum_corr_AA"].nbytes +
            pass_data["sum_corr_BB"].nbytes +
            pass_data["sum_corr_AB"].nbytes
        ) / (1024 ** 2)  # Convert to MB

        # Clear large arrays (keep metadata for grid info)
        pass_data["sum_warp_A"] = None
        pass_data["sum_warp_B"] = None
        pass_data["sum_corr_AA"] = None
        pass_data["sum_corr_BB"] = None
        pass_data["sum_corr_AB"] = None
        pass_data["smoothed_predictor"] = None

        logging.info(
            f"Pass {pass_idx + 1}: Cleared accumulated data "
            f"(freed ~{mem_before:.1f} MB)"
        )