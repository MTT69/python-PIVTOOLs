"""
Single Pass Accumulator for Ensemble PIV

This module implements the SinglePassAccumulator class for ensemble PIV processing,
handling accumulation of correlation planes and single-pass optimization.

NaN Reason Codes (nan_reason / status codes)
============================================
These codes indicate why a vector was marked as invalid or failed. Stored in
PIVPassResult.nan_reason. Produced by the k-space fitter and the accumulator.

Code  Stage                   Meaning
----  -----                   -------
 -1   Pre-fitting             Masked vector (outside ROI)
  0   Success                 Fit succeeded, all checks passed
  1   Fitting                 Linear LS underdetermined / singular
  2   Post-fit validation     SNR too low
  3   Post-fit validation     Displacement > 3/4 window
  5   Post-fit validation     Negative variance
  6   Displacement check      Displacement > 3/4 window (in accumulator)
 10   Outlier detection       Velocity outlier (median test)
 11   Outlier detection       Stress outlier (median test / realizability violation)
"""

import gc
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from dask.distributed import Client

from pivtools_cli.piv.piv_backend.base import CrossCorrelator
from pivtools_cli.piv.piv_backend.infilling import apply_infilling
from pivtools_cli.piv.piv_backend.outlier_detection import apply_outlier_detection
from pivtools_cli.piv.piv_result import PIVEnsemblePassResult, PIVEnsembleResult
from pivtools_core.config import Config


def _linear_pair_envelope(
    weight_a: np.ndarray, weight_b: np.ndarray, corr_size: tuple[int, int]
) -> np.ndarray:
    """Linear pair-count envelope E(Δ) = (W_A ⋆ W_B)(Δ), normalized to E = 1 at zero lag.

    The averaged correlation planes are attenuated by the LINEAR cross-correlation of
    the two window weight functions (Westerweel's loss-of-correlation F_i): the
    coherent peak is squeezed narrower and pulled toward zero lag. Dividing the
    accumulated planes by E removes that bias before the k-space fit. The μ² pedestal
    carries the CIRCULAR weight correlation instead (flat for these weights), which is
    handled by the fitter's noise-floor model, not here.

    Both weights must share one shape (the FFT computation size). The envelope is
    computed on the full linear support via 2x zero-padding and centre-cropped to
    ``corr_size`` (the stored plane size), with zero lag at ``corr_size // 2`` — the
    same convention as the C correlator's central extraction and the k-space fitter's
    fftshift centre. Validated <1% against the production C path
    (manual_tools/kspace/envelope_probe_empirical.py, 2026-07-06).
    """
    wa = np.asarray(weight_a, dtype=np.float64)
    wb = np.asarray(weight_b, dtype=np.float64)
    if wa.shape != wb.shape:
        raise ValueError(
            f"window weights must share a shape, got {wa.shape} vs {wb.shape}"
        )
    h, w = wa.shape
    ch, cw = corr_size
    if ch > h or cw > w:
        raise ValueError(
            f"corr_size {tuple(corr_size)} exceeds envelope support {(h, w)}"
        )

    fa = np.fft.fft2(wa, s=(2 * h, 2 * w))
    fb = np.fft.fft2(wb, s=(2 * h, 2 * w))
    full = np.fft.fftshift(np.fft.ifft2(fa * np.conj(fb)).real)  # zero lag at (h, w)
    env = full[h - ch // 2 : h - ch // 2 + ch, w - cw // 2 : w - cw // 2 + cw]

    centre = env[ch // 2, cw // 2]
    if abs(centre - env.max()) > 1e-9 * env.max():
        peak = np.unravel_index(np.argmax(env), env.shape)
        raise ValueError(
            f"envelope centre {centre:.12f} != max {env.max():.12f} at {peak} — "
            f"centering is off"
        )
    env = env / centre
    if np.any(env <= 0.0):
        raise ValueError(
            "pair-count envelope has non-positive entries inside the stored plane; "
            "cannot divide (check window weights against corr_size)"
        )
    return env


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

        # Section-level profiling (disabled by default, zero overhead)
        self.profiling_enabled = False
        self.profile_data = {}  # {pass_idx: {section_name: elapsed_seconds}}

        H, W = config.image_shape

        # Initialize accumulators for each pass
        for pass_idx in range(config.ensemble_num_passes):
            win_size = config.ensemble_window_sizes[pass_idx]
            overlap = config.ensemble_overlaps[pass_idx]
            runtype = config.ensemble_type[pass_idx]

            # Determine correlation size (output size after central extraction)
            if runtype == "single":
                fit_window = config.ensemble_sum_fitting_window
                corr_size = (
                    tuple(fit_window)
                    if fit_window
                    else tuple(config.ensemble_sum_window)
                )
            else:
                corr_size = win_size

            # Compute grid size
            from pivtools_core.window_utils import (
                compute_window_centers,
                compute_window_centers_single_mode,
            )

            if runtype == "single":
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

            self.passes_data.append(
                {
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
                    # First-pair warped images (for diagnostic saving)
                    "first_pair_A": None,
                    "first_pair_B": None,
                }
            )

    @contextmanager
    def _profile_section(self, pass_idx, section):
        """Context manager for timing named sections within finalize_pass."""
        if not self.profiling_enabled:
            yield
            return
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self.profile_data.setdefault(pass_idx, {})[section] = elapsed

    def get_profile_summary(self):
        """Return a copy of the profile data collected during finalization."""
        return dict(self.profile_data)

    def reset_profile_data(self):
        """Clear profile data for a new profiling run."""
        self.profile_data = {}

    def load_previous_passes(
        self, ensemble_result: PIVEnsembleResult, n_images: int
    ) -> None:
        """
        Load previous passes from existing ensemble result for resume functionality.

        This method allows resuming ensemble PIV from a specific pass by loading
        completed passes from a previously saved ensemble_result.mat file.

        Parameters
        ----------
        ensemble_result : PIVEnsembleResult
            Loaded ensemble result containing completed passes
        n_images : int
            Number of images used to generate the loaded result
            (kept for API compatibility but not used - each pass counts its own images)
        """
        # NOTE: Do NOT set self.n_images here! Each pass should count its own images
        # via accumulate_batch(). Setting it here causes double-counting when resuming.
        self.passes_results = list(ensemble_result.passes)
        logging.info(
            f"Loaded {len(self.passes_results)} previous passes for resume "
            f"(n_images={n_images})"
        )

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

        # Accumulate warped images (shape validation for single mode debugging)
        logging.debug(
            f"Pass {pass_idx}: accumulator shape {pass_data['sum_warp_A'].shape}, "
            f"batch warp shape {batch_result['warp_A_sum'].shape}"
        )
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

        # Store padded predictor (on previous pass grid + boundary padding)
        if batch_result.get("padded_predictor") is not None:
            pass_data["padded_predictor"] = batch_result["padded_predictor"]

        # Store padding values for predictor storage in finalize_pass
        # These are needed to pad the final velocities like instantaneous does
        if batch_result.get("n_pre") is not None:
            pass_data["n_pre"] = batch_result["n_pre"]
        if batch_result.get("n_post") is not None:
            pass_data["n_post"] = batch_result["n_post"]

        # Store first-pair warped images (only from first batch)
        if (
            batch_result.get("first_pair_A") is not None
            and pass_data["first_pair_A"] is None
        ):
            pass_data["first_pair_A"] = batch_result["first_pair_A"]
            pass_data["first_pair_B"] = batch_result["first_pair_B"]
            logging.debug(
                f"Pass {pass_idx + 1}: Stored first-pair warped images "
                f"(shape: {batch_result['first_pair_A'].shape})"
            )

        self.n_images += batch_result["n_images"]

    def _correlate_mean_images(
        self,
        A_mean: np.ndarray,
        B_mean: np.ndarray,
        pass_idx: int,
        window_mean_subtract: bool = False,
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
        window_mean_subtract : bool
            'correlation+window_mean' mode: remove each window's weighted mean
            from the mean images before correlating (bMeanSubtract=1), so the
            background term matches sums whose pairs had their window means
            removed in C. Exact by linearity of the window-mean operator:
            mean(A_i − m(A_i)) = Ā − m(Ā). Must stay False for plain
            'correlation' (raw sums need the raw <A>⊗<B> term).

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
        is_single_mode = runtype == "single"

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

        # Allocate output correlation planes (at output/fitting size)
        correl_AA_bg = np.ascontiguousarray(
            np.zeros(total_windows * corr_size[0] * corr_size[1], dtype=np.float32)
        )
        correl_BB_bg = np.ascontiguousarray(
            np.zeros(total_windows * corr_size[0] * corr_size[1], dtype=np.float32)
        )
        correl_AB_bg = np.ascontiguousarray(
            np.zeros(total_windows * corr_size[0] * corr_size[1], dtype=np.float32)
        )

        # Create temporary correlator to get library and window weights
        from pivtools_cli.piv.piv_backend.factory import make_correlator_backend

        correlator = make_correlator_backend(self.config, ensemble=True)

        # Get computation size (for FFT) and output size (for extraction)
        # These must match how raw correlations are computed in bulkxcorr2d_accumulate
        comp_size = correlator.window_sizes_for_computation[pass_idx]
        out_size = correlator.window_sizes_for_corr[pass_idx]

        # Set up arrays for bulkxcorr2d_accumulate
        n_windows = np.array([n_win_y, n_win_x], dtype=np.int32)
        image_size = np.array([A_mean.shape[0], A_mean.shape[1]], dtype=np.int32)
        win_size_arr = np.array([comp_size[0], comp_size[1]], dtype=np.int32)
        fit_size_arr = np.array([out_size[0], out_size[1]], dtype=np.int32)

        # Create mask
        if correlator.vector_masks and pass_idx < len(correlator.vector_masks):
            b_mask = np.ascontiguousarray(
                correlator.vector_masks[pass_idx].astype(np.float32)
            )
        else:
            b_mask = np.ascontiguousarray(
                np.zeros((n_win_y, n_win_x), dtype=np.float32)
            )

        # Use bulkxcorr2d_accumulate_triple for all three background correlations
        # in one fused call.  This computes AB, AA, BB with shared window
        # extraction (same A_mean, B_mean images) — 3 forward FFTs instead of 6.
        # Stack mean image as (1, H, W) for the N_images=1 interface
        A_mean_stack = np.ascontiguousarray(A_mean[np.newaxis, :, :].astype(np.float32))
        B_mean_stack = np.ascontiguousarray(B_mean[np.newaxis, :, :].astype(np.float32))

        # Add padding offset for single mode passes.
        # win_ctrs are in original image coords; the C library receives a
        # padded image, so centers must be offset by the padding.
        # For standard mode passes, padding is (0,0,0,0) so this is a no-op.
        pad_top, pad_bottom, pad_left, pad_right = correlator.padding_per_pass[pass_idx]

        win_ctrs_x_padded = (correlator.win_ctrs_x[pass_idx] + pad_left).astype(
            np.float32
        )
        win_ctrs_y_padded = (correlator.win_ctrs_y[pass_idx] + pad_top).astype(
            np.float32
        )

        correlator.lib.bulkxcorr2d_accumulate_triple(
            A_mean_stack,
            B_mean_stack,
            b_mask,
            image_size,
            1,  # N_images = 1
            win_ctrs_x_padded,
            win_ctrs_y_padded,
            n_windows,
            correlator.win_weights_A[pass_idx],  # AB weight A
            correlator.win_weights_B[pass_idx],  # AB weight B
            correlator.win_weights_B[pass_idx],  # auto weight A (symmetric)
            correlator.win_weights_B[pass_idx],  # auto weight B (symmetric)
            win_size_arr,  # FFT computation size
            fit_size_arr,  # Output size (central extraction)
            # bMeanSubtract: 1 in 'correlation+window_mean' so the background
            # gets the same per-window mean removal as the per-pair sums
            1 if window_mean_subtract else 0,
            0,  # bPerPairNorm off — background term must stay unnormalized
            correl_AB_bg,
            correl_AA_bg,
            correl_BB_bg,
        )
        logging.debug(
            f"Pass {pass_idx}: AA_bg after bulkxcorr2d_accumulate_triple: [{correl_AA_bg.min():.3e}, {correl_AA_bg.max():.3e}]"
        )
        logging.debug(
            f"Pass {pass_idx}: BB_bg after bulkxcorr2d_accumulate_triple: [{correl_BB_bg.min():.3e}, {correl_BB_bg.max():.3e}]"
        )

        logging.debug(
            f"Pass {pass_idx}: Computed background correlations from mean images"
        )

        return correl_AA_bg, correl_BB_bg, correl_AB_bg

    def finalize_pass(
        self,
        pass_idx: int,
        client: Client,
        predictor_field: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None,
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
        from pivtools_core.window_utils import (
            compute_window_centers,
            compute_window_centers_single_mode,
        )

        logging.info(f"Finalizing pass {pass_idx + 1} with single-pass optimization")

        pass_data = self.passes_data[pass_idx]
        N = self.n_images

        logging.info(f"Pass {pass_idx + 1}: Applying single-pass optimization")

        # Check background subtraction method
        bg_method = self.config.ensemble_background_subtraction_method
        bg_base = self.config.ensemble_bg_base_method
        skip_bg_subtraction = self.config.ensemble_skip_background_subtraction

        with self._profile_section(pass_idx, "mean_computation"):
            # Step 1: Compute mean warped images (always needed for diagnostics/metadata)
            A_mean = pass_data["sum_warp_A"] / N
            B_mean = pass_data["sum_warp_B"] / N

            # Step 2: Compute average correlation planes
            R_AA_raw = pass_data["sum_corr_AA"] / N
            R_BB_raw = pass_data["sum_corr_BB"] / N
            R_AB_raw = pass_data["sum_corr_AB"] / N

        # Step 3-4: Background subtraction depends on method
        with self._profile_section(pass_idx, "bg_subtraction"):
            if bg_base == "image":
                # IMAGE method (plain or '+window_mean'): mean was subtracted
                # per pair BEFORE correlation (and in the combined mode the C
                # correlator removed each window's weighted mean on top).
                # Correlation planes are already background-subtracted:
                # R = <(A-Ā)⊗(B-B̄)>
                logging.info(
                    f"Pass {pass_idx + 1}: Using '{bg_method}' background method "
                    f"(subtracted per pair before accumulation)"
                )
                R_AA_ensemble = R_AA_raw
                R_BB_ensemble = R_BB_raw
                R_AB_ensemble = R_AB_raw
                # Set background to zero for diagnostic logging
                R_AA_bg = np.zeros_like(R_AA_raw)
                R_BB_bg = np.zeros_like(R_BB_raw)
                R_AB_bg = np.zeros_like(R_AB_raw)
            elif bg_method == "window_mean":
                # WINDOW_MEAN method: each window's weighted mean was subtracted
                # per pair inside the C correlator — the pedestal never entered
                # the sums, so no <A>⊗<B> term is subtracted here.
                logging.info(
                    f"Pass {pass_idx + 1}: Using 'window_mean' background method "
                    f"(per-pair per-window mean subtracted in C)"
                )
                R_AA_ensemble = R_AA_raw
                R_BB_ensemble = R_BB_raw
                R_AB_ensemble = R_AB_raw
                R_AA_bg = np.zeros_like(R_AA_raw)
                R_BB_bg = np.zeros_like(R_BB_raw)
                R_AB_bg = np.zeros_like(R_AB_raw)
            elif skip_bg_subtraction:
                # Skip background subtraction (debug mode)
                logging.warning(
                    f"Pass {pass_idx + 1}: SKIPPING background subtraction (debug mode)"
                )
                R_AA_ensemble = R_AA_raw
                R_BB_ensemble = R_BB_raw
                R_AB_ensemble = R_AB_raw
                R_AA_bg = np.zeros_like(R_AA_raw)
                R_BB_bg = np.zeros_like(R_BB_raw)
                R_AB_bg = np.zeros_like(R_AB_raw)
            else:
                # CORRELATION method (plain or '+window_mean'):
                # R_ensemble = <A⊗B> - <A>⊗<B>. In the combined mode the
                # per-pair sums had each window's weighted mean removed in C,
                # so the mean images get the identical treatment
                # (window_mean_subtract=True) — by linearity the subtracted
                # term is exactly the stationary residual left in the sums.
                R_AA_bg, R_BB_bg, R_AB_bg = self._correlate_mean_images(
                    A_mean,
                    B_mean,
                    pass_idx,
                    window_mean_subtract=(
                        self.config.ensemble_window_mean_in_correlator
                    ),
                )
                R_AA_ensemble = R_AA_raw - R_AA_bg
                R_BB_ensemble = R_BB_raw - R_BB_bg
                R_AB_ensemble = R_AB_raw - R_AB_bg

        # Step 4a: Variance decomposition check
        # By Jensen's inequality, <A⊗A> >= <A>⊗<A>, so R_AA_raw >= R_AA_bg at center.
        # If this is violated, the correlation sums are inconsistent with the warp sums
        # (e.g. C library zeroed buffers between accumulation calls).
        _n_win = pass_data["n_win_y"] * pass_data["n_win_x"]
        _cs = pass_data["corr_size"]
        _cy, _cx = _cs[0] // 2, _cs[1] // 2
        _aa_raw_centers = R_AA_raw.reshape(_n_win, _cs[0], _cs[1])[:, _cy, _cx]
        _aa_bg_centers = R_AA_bg.reshape(_n_win, _cs[0], _cs[1])[:, _cy, _cx]
        _aa_ens_centers = R_AA_ensemble.reshape(_n_win, _cs[0], _cs[1])[:, _cy, _cx]
        _n_negative = int(np.sum(_aa_ens_centers < 0))
        logging.info(
            f"Pass {pass_idx + 1}: Variance check (N={N}): "
            f"AA_raw_center=[{_aa_raw_centers.min():.4e}, {_aa_raw_centers.max():.4e}], "
            f"AA_bg_center=[{_aa_bg_centers.min():.4e}, {_aa_bg_centers.max():.4e}], "
            f"AA_ensemble_center=[{_aa_ens_centers.min():.4e}, {_aa_ens_centers.max():.4e}], "
            f"negative={_n_negative}/{_n_win}"
        )
        if _n_negative > 0:
            logging.warning(
                f"Pass {pass_idx + 1}: {_n_negative}/{_n_win} windows have negative AA variance."
            )

        # Step 5: Get configuration for this pass
        win_size = pass_data["win_size"]
        corr_size = pass_data["corr_size"]
        n_win_y = pass_data["n_win_y"]
        n_win_x = pass_data["n_win_x"]
        total_windows = n_win_y * n_win_x

        # Step 5b: Normalize correlation planes by geometric mean of autocorrelation peaks
        # This improves the condition number of the stacked Gaussian solver
        # by ensuring all three planes have similar scale (~1.0 at peaks)
        with self._profile_section(pass_idx, "normalization"):
            AA_3d = R_AA_ensemble.reshape(total_windows, corr_size[0], corr_size[1])
            BB_3d = R_BB_ensemble.reshape(total_windows, corr_size[0], corr_size[1])
            AB_3d = R_AB_ensemble.reshape(total_windows, corr_size[0], corr_size[1])

            # Envelope divide: remove the window-overlap (pair-count) envelope from the
            # coherent planes before fitting (Westerweel loss-of-correlation). Every
            # plane is divided by its own TRUE envelope (2026-07-13; replaces the old
            # plateau-only single-mode design of env_ab = 1 + plateau guard):
            #   std    — all three planes share E = W ⋆ W: divide AA, BB and AB.
            #   single — autos carry the sum-window envelope B ⋆ B; AB carries the
            #            asymmetric A ⋆ B trapezoid (the divide is a no-op wherever a
            #            sum-window margin puts the peak on the E = 1 plateau).
            # Safe only because the per-pair pedestal is removed at source
            # (window_mean) — a surviving pedestal is amplified by the divide into the
            # 2026-07-08 "bowl" that diverges the LM fit. The weights used here must
            # match cpu_ensemble.py (singlepix/bsingle) so E equals the actual
            # attenuation the C correlator applied.
            # E(0) = 1, so the peak values used for normalization below are unchanged.
            envelope_runtype = self.config.ensemble_type[pass_idx]
            if envelope_runtype == "single":
                env_sum_window = tuple(self.config.ensemble_sum_window)
                env_weight_a = CrossCorrelator._window_weight_fun(
                    tuple(win_size), "singlepix", env_sum_window
                )
                env_weight_b = CrossCorrelator._window_weight_fun(
                    env_sum_window, "bsingle", env_sum_window
                )
                env_auto = _linear_pair_envelope(env_weight_b, env_weight_b, corr_size)
                env_ab = _linear_pair_envelope(env_weight_a, env_weight_b, corr_size)
            else:
                env_weight = CrossCorrelator._window_weight_fun(
                    tuple(win_size), self.config.ensemble_window_type
                )
                env_auto = _linear_pair_envelope(env_weight, env_weight, corr_size)
                env_ab = env_auto
            env_auto_f32 = env_auto.astype(np.float32)[None, :, :]
            env_ab_f32 = env_ab.astype(np.float32)[None, :, :]
            # E(0)=1 so the peak/center (and norm_factors below) are unchanged; only the
            # off-peak floor is affected.
            AA_3d /= env_auto_f32
            BB_3d /= env_auto_f32
            AB_3d /= env_ab_f32
            logging.info(
                f"Pass {pass_idx + 1}: Envelope divide applied ({envelope_runtype}: "
                f"AA/BB by their pair-count envelope, AB by its true envelope; "
                f"min E_auto = {env_auto.min():.3f}, min E_ab = {env_ab.min():.3f})"
            )

            # Central index (autocorrelation peak is at center)
            center_y, center_x = corr_size[0] // 2, corr_size[1] // 2

            # Extract central peak values for each window
            AA_peaks = AA_3d[:, center_y, center_x]
            BB_peaks = BB_3d[:, center_y, center_x]

            # Geometric mean with safety floor to avoid division by zero
            norm_factors = np.sqrt(np.maximum(AA_peaks * BB_peaks, 1e-12))

            # Reshape for broadcasting: (n_windows, 1, 1)
            norm_factors_3d = norm_factors[:, np.newaxis, np.newaxis]

            # For single mode, apply asymmetric window correction to AB
            # AA/BB use weight_B (full sum_window) on both sides, so they scale with sum_window^2
            # AB uses weight_A (particle window) × weight_B (sum_window), so it scales with
            # particle_window × sum_window. The normalization by sqrt(AA*BB) over-corrects AB.
            # We need to scale AB up by sqrt(sum_window_area / particle_window_area) to compensate.
            runtype = self.config.ensemble_type[pass_idx]
            if runtype == "single":
                sum_window = self.config.ensemble_sum_window
                particle_window = win_size
                sum_area = sum_window[0] * sum_window[1]
                particle_area = particle_window[0] * particle_window[1]
                # AB scales as sqrt(particle × sum), but norm_factors assumes sqrt(sum × sum)
                # Correction factor: sqrt(sum_area / particle_area)
                ab_scale_correction = np.sqrt(sum_area / particle_area)
                logging.debug(
                    f"Pass {pass_idx + 1}: Single mode AB scale correction = {ab_scale_correction:.3f} "
                    f"(sum_window={sum_window}, particle_window={particle_window})"
                )
                AB_3d = AB_3d * ab_scale_correction

            # Normalize all three planes
            AA_3d_norm = AA_3d / norm_factors_3d
            BB_3d_norm = BB_3d / norm_factors_3d
            AB_3d_norm = AB_3d / norm_factors_3d

            # Flatten back to original format
            R_AA_ensemble = AA_3d_norm.reshape(-1).astype(np.float32)
            R_BB_ensemble = BB_3d_norm.reshape(-1).astype(np.float32)
            R_AB_ensemble = AB_3d_norm.reshape(-1).astype(np.float32)

        logging.debug(
            f"Pass {pass_idx + 1}: Normalized planes by geometric mean "
            f"(min={norm_factors.min():.4f}, max={norm_factors.max():.4f}, "
            f"median={np.median(norm_factors):.4f})"
        )

        # Debug: Verify correlation plane sizes match expected dimensions
        expected_size = total_windows * corr_size[0] * corr_size[1]
        logging.debug(
            f"Pass {pass_idx}: Correlation plane sizes - "
            f"R_AA: {R_AA_ensemble.size}, R_BB: {R_BB_ensemble.size}, R_AB: {R_AB_ensemble.size}, "
            f"expected: {expected_size} ({total_windows} windows × {corr_size[0]}×{corr_size[1]})"
        )

        # Step 6: Perform distributed k-space fitting

        # Flatten mask for fitting
        if self.vector_masks and pass_idx < len(self.vector_masks):
            mask_flat = self.vector_masks[pass_idx].ravel(order="C").astype(bool)
            logging.info(
                f"mask shape: {self.vector_masks[pass_idx].shape}, flat shape: {mask_flat.shape}"
            )
            # Validate mask size matches data grid
            if mask_flat.size != total_windows:
                raise ValueError(
                    f"Vector mask size mismatch in pass {pass_idx + 1}: "
                    f"mask has {mask_flat.size} elements (shape {self.vector_masks[pass_idx].shape}) "
                    f"but data grid has {total_windows} windows ({n_win_y}×{n_win_x}). "
                    f"The mask must match the PIV grid dimensions for each pass."
                )
        else:
            mask_flat = np.zeros(total_windows, dtype=bool)

        # K-space fitting, dispatched across Dask workers.
        # Reading the property here also validates fit_method (a stale
        # 'gaussian' or 'kspace_linear' config raises loudly rather than
        # silently falling back).
        fit_method = self.config.ensemble_fit_method
        logging.info(
            f"Pass {pass_idx + 1}: Starting k-space transfer function fitting..."
        )

        with self._profile_section(pass_idx, "scatter"):
            n_workers = len(client.scheduler_info()["workers"])
            list(client.scheduler_info()["workers"].keys())
            windows_per_worker = (total_windows + n_workers - 1) // n_workers
            R_AA_futures = []
            R_BB_futures = []
            R_AB_futures = []
            mask_flat_futures = []
            for worker_idx in range(n_workers):
                # Use corr_size (not win_size) for slicing - correlation planes are sized at SumWindow
                start_idx = (
                    worker_idx * windows_per_worker * corr_size[0] * corr_size[1]
                )
                end_idx = min(
                    (worker_idx + 1) * windows_per_worker * corr_size[0] * corr_size[1],
                    R_AA_ensemble.size,
                )
                start_idx_win = worker_idx * windows_per_worker
                end_idx_win = min(
                    (worker_idx + 1) * windows_per_worker,
                    total_windows,
                )

                R_AA_futures.append(
                    client.scatter(
                        R_AA_ensemble[start_idx:end_idx],
                        broadcast=False,
                    )
                )
                R_BB_futures.append(
                    client.scatter(
                        R_BB_ensemble[start_idx:end_idx],
                        broadcast=False,
                    )
                )
                R_AB_futures.append(
                    client.scatter(
                        R_AB_ensemble[start_idx:end_idx],
                        broadcast=False,
                    )
                )
                mask_flat_futures.append(
                    client.scatter(
                        mask_flat[start_idx_win:end_idx_win],
                        broadcast=False,
                    )
                )

        # K-space fit, dispatched across Dask workers. 'kspace' — the only
        # production method — is the one-stage 7-param joint LM fit of the raw
        # transfer ratio (mu, Sigma, gain g, in-model noise floor N0);
        # see kspace_lm_fitting.py for the model. Returns the 16-element
        # gauss_flat contract.
        save_fit_diagnostics = (
            fit_method == "kspace" and self.config.ensemble_save_diagnostics
        )
        with self._profile_section(pass_idx, "fitting"):
            from pivtools_cli.piv.piv_backend.kspace_lm_fitting import (
                fit_windows_kspace_lm,
            )

            futures = [
                client.submit(
                    fit_windows_kspace_lm,
                    R_AA_futures[i],
                    R_BB_futures[i],
                    R_AB_futures[i],
                    mask_flat_futures[i],
                    corr_size,
                    self.config,
                    pass_idx,
                    self.config.debug,  # diagnostics when debug=True
                    save_fit_diagnostics,  # return per-window diag dict
                )
                for i in range(len(R_AA_futures))
            ]

            results = client.gather(futures)
        gauss_flat = np.concatenate([r[0] for r in results])
        status_flat = np.concatenate([r[1] for r in results])
        initial_guess_flat = np.concatenate([r[2] for r in results])

        # Persist the LM fitter's per-window diagnostics next to the planes
        if save_fit_diagnostics:
            from scipy.io import savemat

            diag_chunks = [r[3] for r in results]
            fit_diag = {
                key: np.concatenate([d[key] for d in diag_chunks]).reshape(
                    n_win_y, n_win_x
                )
                for key in diag_chunks[0]
            }
            fit_diag["status"] = status_flat.reshape(n_win_y, n_win_x)
            fit_diag["pass_idx"] = pass_idx
            fit_diag["n_pairs"] = N
            fit_diag["bg_method"] = bg_method
            fit_diag["per_pair_normalization"] = (
                self.config.ensemble_per_pair_normalization
            )
            fit_diag["kspace_shape"] = self.config.ensemble_kspace_shape
            diag_outdir = Path(output_path) if output_path else Path(os.getcwd())
            diag_outdir.mkdir(parents=True, exist_ok=True)
            savemat(
                diag_outdir / f"fit_diagnostics_pass_{pass_idx + 1}.mat",
                fit_diag,
                do_compression=True,
            )
            logging.info(
                f"Pass {pass_idx + 1}: Saved LM fit diagnostics to "
                f"{diag_outdir}/fit_diagnostics_pass_{pass_idx + 1}.mat"
            )

        # Release large arrays after fitting
        if not (
            hasattr(self.config, "ensemble_store_planes")
            and self.config.ensemble_store_planes
        ):
            del R_AA_ensemble, R_BB_ensemble, R_AB_ensemble
            gc.collect()

        gauss_results = gauss_flat.reshape(n_win_y, n_win_x, -1)
        statuses = status_flat.reshape(n_win_y, n_win_x)
        initial_guesses = initial_guess_flat.reshape(n_win_y, n_win_x, -1)

        # Step 7: Extract velocities from fitted parameters
        with self._profile_section(pass_idx, "velocity_extraction"):
            # Determine correlation size for grid
            runtype = self.config.ensemble_type[pass_idx]
            if runtype == "single":
                grid_result = compute_window_centers_single_mode(
                    image_shape=self.config.image_shape,
                    window_size=tuple(win_size),
                    sum_window=tuple(self.config.ensemble_sum_window),
                    overlap=self.config.ensemble_overlaps[pass_idx],
                    validate=True,
                )
                # Convert to original image coords (subtract padding offset)
                grid_result_ctrs_x = grid_result.win_ctrs_x - grid_result.padding[2]
                grid_result_ctrs_y = grid_result.win_ctrs_y - grid_result.padding[0]
            else:
                grid_result = compute_window_centers(
                    image_shape=self.config.image_shape,
                    window_size=tuple(win_size),
                    overlap=self.config.ensemble_overlaps[pass_idx],
                    validate=True,
                )
                grid_result_ctrs_x = grid_result.win_ctrs_x
                grid_result_ctrs_y = grid_result.win_ctrs_y

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
                ~np.isfinite(ux_mat)
                | ~np.isfinite(uy_mat)
                | (np.abs(ux_mat) > max_disp_x)
                | (np.abs(uy_mat) > max_disp_y)
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
                if (
                    "smoothed_predictor" in pass_data
                    and pass_data["smoothed_predictor"] is not None
                ):
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
        with self._profile_section(pass_idx, "parameter_extraction"):
            # Clamp to reasonable ranges before float32 cast to prevent overflow
            MAX_AMP = 1e10  # Max reasonable amplitude
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
            # In single mode, apply amplitude correction for asymmetric weighting:
            # AB uses (particle_window × sum_window) while AA/BB use (sum_window × sum_window)
            # This reduces AB amplitude by sqrt(particle_area / sum_area)
            # Correction factor: sqrt(sum_area / particle_area)
            geom_mean = np.sqrt(np.maximum(amp_A * amp_B, 1e-12))
            peakheight_raw = amp_AB / geom_mean

            runtype = self.config.ensemble_type[pass_idx]
            if runtype == "single":
                particle_window = self.config.ensemble_window_sizes[pass_idx]
                sum_window = self.config.ensemble_sum_window
                particle_area = particle_window[0] * particle_window[1]
                sum_area = sum_window[0] * sum_window[1]
                amplitude_correction = np.sqrt(sum_area / particle_area)
                peakheight_raw *= amplitude_correction
                logging.debug(
                    f"Pass {pass_idx + 1}: Applied single mode amplitude correction "
                    f"sqrt({sum_area}/{particle_area}) = {amplitude_correction:.3f}"
                )

            peakheight = np.clip(peakheight_raw, 0.0, 1.0).astype(np.float32)

            # Gaussian offset terms (can be negative after background subtraction)
            c_A = safe_extract(gauss_results[:, :, 3], MAX_AMP, 0.0)
            c_B = safe_extract(gauss_results[:, :, 4], MAX_AMP, 0.0)
            c_AB = safe_extract(gauss_results[:, :, 5], MAX_AMP, 0.0)

            # Gaussian widths for A autocorrelation (particle size, from AA/BB peaks)
            sig_A_x = safe_extract(gauss_results[:, :, 6], MAX_SIGMA, 0.0)
            sig_A_y = safe_extract(gauss_results[:, :, 7], MAX_SIGMA, 0.0)
            sig_A_xy = safe_extract(gauss_results[:, :, 8], MAX_SIGMA, 0.0)

            # Gaussian widths for AB cross-correlation (TOTAL width, used directly by C fitter)
            # With decoupled parameterization, sig_AB is the raw fitted total width,
            # NOT an additive term on top of sig_A.
            sig_AB_x = safe_extract(gauss_results[:, :, 9], MAX_SIGMA, 0.0)
            sig_AB_y = safe_extract(gauss_results[:, :, 10], MAX_SIGMA, 0.0)
            sig_AB_xy = safe_extract(gauss_results[:, :, 11], MAX_SIGMA, 0.0)

            # Compute displacement uncertainty = sig_AB - sig_A
            # This represents the additional width from displacement variance
            # (what was previously stored directly in sig_AB fields)
            # Constraint: displacement uncertainty >= 0
            UU_stress = np.maximum(sig_AB_x - sig_A_x, 0.0)
            VV_stress = np.maximum(sig_AB_y - sig_A_y, 0.0)
            UV_stress = sig_AB_xy - sig_A_xy  # Cross-term can be negative

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
                logging.info(
                    f"Pass {pass_idx + 1}: {vector_mask.sum()} vectors masked (set to zero)"
                )

        # =========================================================
        # STEP 7b: Outlier Detection and Infilling
        # =========================================================
        # Determine if this is final pass
        is_final_pass = pass_idx == self.config.ensemble_num_passes - 1

        with self._profile_section(pass_idx, "outlier_detection"):
            # --- Combined Outlier Detection ---
            # Start with fitting failures (statuses != 0 indicates failed fit)
            # Exclude already-masked vectors from outlier detection
            outlier_mask = statuses != 0
            if vector_mask is not None:
                outlier_mask = (
                    outlier_mask & ~vector_mask
                )  # Don't double-count masked regions

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
                            ux_mat, uy_mat, outlier_methods, peak_mag=peakheight
                        )
                        # Only mark as outliers within valid detection region
                        outlier_mask |= detected_outliers & valid_for_detection

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

        with self._profile_section(pass_idx, "infilling"):
            # --- Infilling ---
            infill_mask = outlier_mask.copy()

            if is_final_pass:
                # Final pass: use final_pass config (may be disabled)
                infill_cfg = self.config.ensemble_infilling_final_pass
                if not infill_cfg.get("enabled", True):
                    logging.info(f"Pass {pass_idx + 1}: Final pass infilling disabled")
                    infill_mask = np.zeros_like(
                        outlier_mask, dtype=bool
                    )  # Skip infilling
            else:
                # Mid-pass: always infill (required for predictor)
                infill_cfg = self.config.ensemble_infilling_mid_pass

            if infill_mask.any():
                logging.info(
                    f"Pass {pass_idx + 1}: Infilling {infill_mask.sum()} vectors using "
                    f"'{infill_cfg.get('method', 'biharmonic')}'"
                )

                # Infill displacement fields
                ux_mat, uy_mat = apply_infilling(
                    ux_mat, uy_mat, infill_mask, infill_cfg
                )

                # Infill stress fields
                UU_stress, VV_stress = apply_infilling(
                    UU_stress, VV_stress, infill_mask, infill_cfg
                )
                # UV_stress needs special handling (paired with zero array)
                UV_temp = np.zeros_like(UV_stress)
                UV_stress, _ = apply_infilling(
                    UV_stress, UV_temp, infill_mask, infill_cfg
                )

                # Infill sigma fields (A autocorrelation)
                sig_A_x, sig_A_y = apply_infilling(
                    sig_A_x, sig_A_y, infill_mask, infill_cfg
                )
                sig_A_xy_temp = np.zeros_like(sig_A_xy)
                sig_A_xy, _ = apply_infilling(
                    sig_A_xy, sig_A_xy_temp, infill_mask, infill_cfg
                )

                # Infill sigma fields (AB cross-correlation)
                sig_AB_x, sig_AB_y = apply_infilling(
                    sig_AB_x, sig_AB_y, infill_mask, infill_cfg
                )
                sig_AB_xy_temp = np.zeros_like(sig_AB_xy)
                sig_AB_xy, _ = apply_infilling(
                    sig_AB_xy, sig_AB_xy_temp, infill_mask, infill_cfg
                )

                # Infill peakheight (paired with zero array)
                peakheight_temp = np.zeros_like(peakheight)
                peakheight, _ = apply_infilling(
                    peakheight, peakheight_temp, infill_mask, infill_cfg
                )

        # =========================================================
        # STEP 7c: Stress-specific outlier detection (final pass only)
        # =========================================================
        with self._profile_section(pass_idx, "stress_outlier_detection"):
            # Velocity outlier detection can miss windows with plausible velocity
            # but bad stress estimates (e.g., fitter converged on peak position
            # but not on widths). Run median test on stress fields + realizability.
            if is_final_pass and self.config.ensemble_outlier_detection_enabled:
                # Filter config methods to those applicable to stress fields
                # (exclude velocity-specific methods like peak_mag and div_vort)
                STRESS_APPLICABLE_METHODS = {"median_2d", "sigma"}
                outlier_methods = self.config.ensemble_outlier_detection_methods
                stress_methods = [
                    m
                    for m in outlier_methods
                    if m.get("type", "").lower() in STRESS_APPLICABLE_METHODS
                ]

                stress_outlier_mask = np.zeros((n_win_y, n_win_x), dtype=bool)
                if stress_methods:
                    stress_outlier_mask = apply_outlier_detection(
                        UU_stress,
                        VV_stress,
                        stress_methods,
                    )

                # Realizability: Cauchy-Schwarz |UV|² ≤ UU·VV
                with np.errstate(invalid="ignore"):
                    realizability_violation = (
                        np.isfinite(UV_stress)
                        & np.isfinite(UU_stress)
                        & np.isfinite(VV_stress)
                        & (UV_stress**2 > UU_stress * VV_stress)
                    )
                stress_outlier_mask |= realizability_violation

                # Don't re-flag already-masked regions
                stress_outlier_mask &= ~vector_mask

                n_stress_outliers = int(stress_outlier_mask.sum())
                n_realiz = int(realizability_violation.sum())
                if n_stress_outliers > 0:
                    logging.info(
                        f"Pass {pass_idx + 1}: Stress outlier detection found "
                        f"{n_stress_outliers} outliers "
                        f"({n_realiz} realizability, "
                        f"{n_stress_outliers - n_realiz} median test)"
                    )

                    UU_stress[stress_outlier_mask] = np.nan
                    VV_stress[stress_outlier_mask] = np.nan
                    UV_stress[stress_outlier_mask] = np.nan
                    nan_reason[stress_outlier_mask & (nan_reason == 0)] = 11

                    # Infill stress fields only (velocity is already good)
                    if infill_cfg.get("enabled", True):
                        UU_stress, VV_stress = apply_infilling(
                            UU_stress, VV_stress, stress_outlier_mask, infill_cfg
                        )
                        UV_temp = np.zeros_like(UV_stress)
                        UV_stress, _ = apply_infilling(
                            UV_stress, UV_temp, stress_outlier_mask, infill_cfg
                        )
                else:
                    logging.info(
                        f"Pass {pass_idx + 1}: Stress outlier detection found 0 outliers"
                    )

        with self._profile_section(pass_idx, "result_construction"):
            # Store the predictor that was actually used for this pass (delta_ab_pred).
            # This is at the current pass's window centers — same grid as ux/uy —
            # so it can be directly compared to the output for diagnostics.
            # For pass 0, predictor is zero (no previous pass).
            pred_x = None
            pred_y = None
            padded_pred_x = None
            padded_pred_y = None
            if (
                pass_idx > 0
                and "smoothed_predictor" in pass_data
                and pass_data["smoothed_predictor"] is not None
            ):
                smoothed_pred = pass_data["smoothed_predictor"]
                pred_y = smoothed_pred[:, :, 0].copy()  # Y component
                pred_x = smoothed_pred[:, :, 1].copy()  # X component
                logging.debug(
                    f"Pass {pass_idx + 1}: Storing actual predictor field used for this pass "
                    f"(shape: {pred_x.shape})"
                )
            # Compute padded predictor from THIS pass's velocity field.
            # This represents what will be fed to the next pass as the predictor
            # (after edge-padding with n_pre/n_post boundary nodes).
            # Previously this stored self.delta_ab_old from the correlator, which was
            # the padded predictor from the PREVIOUS pass — off by one.
            n_pre = pass_data.get("n_pre")
            n_post = pass_data.get("n_post")
            if n_pre is not None and n_post is not None:
                pre_y, pre_x = n_pre
                post_y, post_x = n_post
                # Stack uy (dim 0) and ux (dim 1) into (n_win_y, n_win_x, 2)
                velocity_field = np.stack([uy_mat, ux_mat], axis=-1)
                padded = np.pad(
                    velocity_field,
                    ((pre_y, post_y), (pre_x, post_x), (0, 0)),
                    mode="edge",
                )
                padded_pred_y = padded[:, :, 0].copy()
                padded_pred_x = padded[:, :, 1].copy()
                logging.debug(
                    f"Pass {pass_idx + 1}: Storing padded predictor from current pass "
                    f"(velocity {ux_mat.shape} -> padded {padded_pred_x.shape}, "
                    f"pre=({pre_y},{pre_x}), post=({post_y},{post_x}))"
                )

            # DEBUG: Log edge values in final pass result to trace edge artifact source
            logging.debug(
                f"Pass {pass_idx + 1}: FINAL RESULT edge values - "
                f"ux_mat: TL={ux_mat[0,0]:.4f}, TR={ux_mat[0,-1]:.4f}, "
                f"BL={ux_mat[-1,0]:.4f}, BR={ux_mat[-1,-1]:.4f}, "
                f"center={ux_mat[ux_mat.shape[0]//2, ux_mat.shape[1]//2]:.4f}, "
                f"uy_mat: TL={uy_mat[0,0]:.4f}, TR={uy_mat[0,-1]:.4f}, "
                f"BL={uy_mat[-1,0]:.4f}, BR={uy_mat[-1,-1]:.4f}, "
                f"center={uy_mat[uy_mat.shape[0]//2, uy_mat.shape[1]//2]:.4f}, "
                f"NaN at edges: top_row={np.isnan(ux_mat[0,:]).sum()}, "
                f"bot_row={np.isnan(ux_mat[-1,:]).sum()}, "
                f"left_col={np.isnan(ux_mat[:,0]).sum()}, "
                f"right_col={np.isnan(ux_mat[:,-1]).sum()}"
            )

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
                padded_pred_x=padded_pred_x,
                padded_pred_y=padded_pred_y,
                window_size=tuple(win_size),
                win_ctrs_x=grid_result_ctrs_x,
                win_ctrs_y=grid_result_ctrs_y,
            )

        # Store result in accumulator
        self.passes_results.append(pass_result)

        # Save correlation planes if store_planes is enabled
        if (
            hasattr(self.config, "ensemble_store_planes")
            and self.config.ensemble_store_planes
        ):
            try:
                from scipy.io import savemat

                if output_path is not None:
                    outdir = Path(output_path)
                else:
                    outdir = Path(os.getcwd())
                outdir.mkdir(parents=True, exist_ok=True)

                # Create correlator to get window weights
                from pivtools_cli.piv.piv_backend.factory import make_correlator_backend

                correlator_for_weights = make_correlator_backend(
                    self.config, ensemble=True
                )

                # Save correlation planes in 4D format (n_win_y, n_win_x, corr_h, corr_w)
                # Note: All planes (AA, BB, AB and backgrounds) are saved in NORMALIZED form
                # (divided by geometric mean of autocorrelation peaks)

                # Normalize background planes with the same norm_factors used for ensemble planes
                norm_factors_3d = norm_factors[:, np.newaxis, np.newaxis]
                AA_bg_3d = R_AA_bg.reshape(total_windows, corr_size[0], corr_size[1])
                BB_bg_3d = R_BB_bg.reshape(total_windows, corr_size[0], corr_size[1])
                AB_bg_3d = R_AB_bg.reshape(total_windows, corr_size[0], corr_size[1])
                AA_bg_norm = (AA_bg_3d / norm_factors_3d).reshape(
                    n_win_y, n_win_x, corr_size[0], corr_size[1]
                )
                BB_bg_norm = (BB_bg_3d / norm_factors_3d).reshape(
                    n_win_y, n_win_x, corr_size[0], corr_size[1]
                )
                AB_bg_norm = (AB_bg_3d / norm_factors_3d).reshape(
                    n_win_y, n_win_x, corr_size[0], corr_size[1]
                )

                planes_dict = {
                    "AA": R_AA_ensemble.reshape(
                        n_win_y, n_win_x, corr_size[0], corr_size[1]
                    ),
                    "BB": R_BB_ensemble.reshape(
                        n_win_y, n_win_x, corr_size[0], corr_size[1]
                    ),
                    "AB": R_AB_ensemble.reshape(
                        n_win_y, n_win_x, corr_size[0], corr_size[1]
                    ),
                    # Background planes from correlating mean images: <A>⋆<A>, <B>⋆<B>, <A>⋆<B> (normalized)
                    "AA_bg": AA_bg_norm,
                    "BB_bg": BB_bg_norm,
                    "AB_bg": AB_bg_norm,
                    "norm_factors": norm_factors.reshape(
                        n_win_y, n_win_x
                    ),  # Geometric mean used for normalization
                    "gauss_results": gauss_results,  # All fitted parameters
                    "initial_guesses": initial_guesses,  # Initial guess parameters for fitting
                    "corr_size": corr_size,
                    "n_win_y": n_win_y,
                    "n_win_x": n_win_x,
                    "pass_idx": pass_idx,
                    # Window weights used in cross-correlation
                    "win_weight_A": correlator_for_weights.win_weights_A[pass_idx],
                    "win_weight_B": correlator_for_weights.win_weights_B[pass_idx],
                    # Pair-count envelopes already divided out of AA/BB/AB above.
                    # Downstream tools must check 'env_divided' before deciding whether
                    # to divide; the presence of env_auto/env_ab alone means nothing.
                    "env_auto": env_auto,
                    "env_ab": env_ab,
                    "env_divided": True,
                    # Accumulation-mode provenance: with 'window_mean' the per-pair
                    # weighted window mean was removed inside the C correlator (the
                    # *_bg planes stored above are zeros), and per-pair
                    # normalization rescales every pair's planes before they enter
                    # the sums — both change what AA/BB/AB mean, so record them.
                    "n_pairs": N,
                    "bg_method": bg_method,
                    "per_pair_normalization": (
                        self.config.ensemble_per_pair_normalization
                    ),
                    "skip_bg_subtraction": skip_bg_subtraction,
                }

                savemat(
                    outdir / f"planes_pass_{pass_idx + 1}.mat",
                    planes_dict,
                    do_compression=True,
                )
                logging.info(
                    f"Pass {pass_idx + 1}: Saved correlation planes to {outdir}/planes_pass_{pass_idx + 1}.mat"
                )

                # Save first-pair warped images to separate MAT file
                if pass_data.get("first_pair_A") is not None:
                    warped_dict = {
                        "A_warped": pass_data["first_pair_A"],
                        "B_warped": pass_data["first_pair_B"],
                        "pass_idx": pass_idx,
                    }
                    savemat(
                        outdir / f"warped_pass_{pass_idx + 1}.mat",
                        warped_dict,
                        do_compression=True,
                    )
                    logging.info(
                        f"Pass {pass_idx + 1}: Saved first-pair warped images to {outdir}/warped_pass_{pass_idx + 1}.mat"
                    )
            except Exception as e:
                logging.warning(
                    f"Pass {pass_idx + 1}: Failed to save correlation planes: {e}"
                )

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

        # Reset n_images for next pass (fixes cumulative count bug)
        self.n_images = 0

        pass_data = self.passes_data[pass_idx]

        # Get memory usage before clearing
        mem_before = (
            pass_data["sum_warp_A"].nbytes
            + pass_data["sum_warp_B"].nbytes
            + pass_data["sum_corr_AA"].nbytes
            + pass_data["sum_corr_BB"].nbytes
            + pass_data["sum_corr_AB"].nbytes
        ) / (
            1024**2
        )  # Convert to MB

        # Clear large arrays (keep metadata for grid info)
        pass_data["sum_warp_A"] = None
        pass_data["sum_warp_B"] = None
        pass_data["sum_corr_AA"] = None
        pass_data["sum_corr_BB"] = None
        pass_data["sum_corr_AB"] = None
        pass_data["smoothed_predictor"] = None

        logging.debug(
            f"Pass {pass_idx + 1}: Cleared accumulated data "
            f"(freed ~{mem_before:.1f} MB)"
        )
