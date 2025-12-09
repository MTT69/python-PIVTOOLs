"""
Unified batched PIV pipeline for HPC clusters.

Supports:
- Flexible worker allocation (5-100+ cores)
- Both instantaneous and ensemble modes
- Temporal and spatial filters
- Single-pass ensemble correlation
"""

import gc
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import dask
import dask.array as da
from dask.distributed import Client, wait

from pivtools_core.config import Config
from pivtools_cli.piv.piv_backend.factory import make_correlator_backend
from pivtools_cli.piv.save_results import (
    save_piv_result_distributed,
    save_ensemble_result_distributed,
)


class UnifiedBatchPipeline:
    """
    Unified batched processing pipeline for PIV.

    Architecture:
    - Filter workers: Apply batch filters (POD, time) with multi-threading
    - Correlation workers: PIV correlation in parallel
    - Pipelined execution: Filter batch N while correlating batch N-1

    Scales from 5 to 100+ cores with configurable worker allocation.
    """

    def __init__(
        self,
        client: Client,
        config: Config,
        mode: str = "instantaneous",
    ):
        self.client = client
        self.config = config
        self.mode = mode  # "instantaneous" or "ensemble"

        # Determine batch size
        if config.batch_size > 0:
            self.batch_size = config.batch_size
        else:
            self.batch_size = config.auto_batch_size

        logging.info(f"Batch size: {self.batch_size}")

        # Allocate workers
        self._allocate_workers()

        logging.info(
            f"Worker allocation: {self.num_filter_workers} filter, "
            f"{self.num_corr_workers} correlation (total: {self.total_workers})"
        )

    def _allocate_workers(self):
        """Allocate workers between filtering and correlation."""
        workers = list(self.client.scheduler_info()["workers"].keys())
        self.total_workers = len(workers)

        # Get allocation from config
        filter_count, corr_count = self.config.get_filter_worker_allocation(
            self.total_workers
        )

        self.num_filter_workers = filter_count
        self.num_corr_workers = corr_count

        # Assign workers to roles
        self.filter_workers = workers[:filter_count]
        self.corr_workers = workers[filter_count:]

        logging.info(f"Assigned {len(self.filter_workers)} filter workers")
        logging.info(f"Assigned {len(self.corr_workers)} correlation workers")

    def process(
        self,
        images: da.Array,
        output_path: Path,
        vector_masks: Optional[List[np.ndarray]] = None,
        pixel_mask: Optional[np.ndarray] = None,
    ):
        """
        Unified entry point for PIV processing.

        Args:
            images: Dask array of shape (N, 2, H, W)
            output_path: Directory for saving results
            vector_masks: Pre-computed masks for each pass (for vector validation)
            pixel_mask: Boolean mask of shape (H, W) where True = masked regions.
                Applied during preprocessing to zero pixel intensities.

        Returns:
            List of saved file paths (instantaneous) or single path (ensemble)
        """
        # Store pixel mask for use in filtering
        self.pixel_mask = pixel_mask

        if self.mode == "ensemble":
            return self._process_ensemble(images, output_path, vector_masks)
        else:
            return self._process_instantaneous(images, output_path, vector_masks)

    def _process_ensemble(
        self,
        images: da.Array,
        output_path: Path,
        vector_masks: Optional[List[np.ndarray]],
    ):
        """
        Ensemble PIV with cross-batch tree reduction.

        Architecture:
        - 1 filter worker (temporal filters need full batch in RAM)
        - N-1 correlation workers (embarrassingly parallel)
        - Tree reduction across ALL batches (not per-batch)
        - Only final accumulated result transferred to main
        """
        from pivtools_cli.piv.piv_backend.single_pass_accumulator import (
            SinglePassAccumulator,
        )

        # Create accumulator for single-pass math
        accumulator = SinglePassAccumulator(self.config, vector_masks)

        # Scatter immutable data once (broadcast to all workers)
        scattered_cache, scattered_masks = self._scatter_immutable_data(vector_masks)

        num_batches = (images.shape[0] + self.batch_size - 1) // self.batch_size
        num_passes = self.config.ensemble_num_passes

        # Check for resume configuration
        resume_from_pass = self.config.ensemble_resume_from_pass
        start_pass_idx = 0  # Default: start from pass 0
        predictor_field: Optional[np.ndarray] = None  # Initialize here

        if resume_from_pass > 0:
            # Validate resume configuration
            if resume_from_pass > num_passes:
                raise ValueError(
                    f"resume_from_pass={resume_from_pass} exceeds total passes ({num_passes}). "
                    "Check your ensemble_piv.window_size configuration."
                )

            # Look for existing ensemble_result.mat in the output directory
            resume_path = output_path / "ensemble_result.mat"
            if not resume_path.exists():
                raise FileNotFoundError(
                    f"Resume file not found: {resume_path}. "
                    f"Cannot resume from pass {resume_from_pass} without existing results. "
                    "Ensure ensemble_result.mat exists in the output directory from a previous run."
                )

            start_pass_idx = resume_from_pass - 1  # Convert to 0-based
            logging.info(f"RESUME MODE: Starting from pass {resume_from_pass} (skipping passes 1-{resume_from_pass - 1})")
            logging.info(f"Loading predictor field from: {resume_path}")

            # Load the existing ensemble result (predictor-only mode for memory efficiency)
            # This loads only ux/uy and sig_AB_x/sig_AB_y, skipping stress/peakheight
            # The full data is preserved in ensemble_result.mat and merged during save
            existing_result = self._load_ensemble_result_from_file(
                resume_path, predictor_only=True
            )

            # Validate it has enough passes
            if len(existing_result.passes) < start_pass_idx:
                raise ValueError(
                    f"Existing ensemble_result.mat contains only {len(existing_result.passes)} passes, "
                    f"but resume_from_pass={resume_from_pass} requires at least {start_pass_idx} passes."
                )

            # Extract predictor field from the pass just before where we resume
            last_pass = existing_result.passes[start_pass_idx - 1]
            predictor_field = self._extract_predictor_field(last_pass, start_pass_idx - 1)
            logging.info(f"Extracted predictor field from pass {start_pass_idx}")

            # Populate accumulator with previous passes for sigma interpolation
            # This ensures _get_sigma_from_previous_pass() can access pass N-1 results
            for prev_pass in existing_result.passes:
                accumulator.passes_results.append(prev_pass)
            logging.info(f"Loaded {len(existing_result.passes)} previous pass results into accumulator")

        logging.info(f"Processing passes {start_pass_idx + 1}-{num_passes} with {num_batches} batches each for ensemble PIV")

        # Multi-pass loop with cross-batch tree reduction
        for pass_idx in range(start_pass_idx, num_passes):
            logging.info("")
            logging.info(f"======== PASS {pass_idx + 1}/{num_passes} ========")

            # Scatter predictor field for this pass (if pass > 0)
            scattered_predictor = None
            if predictor_field is not None:
                scattered_predictor = self.client.scatter(predictor_field, broadcast=True)
                logging.info(f"[Pass {pass_idx + 1}] Broadcast predictor field from previous pass")

            # === CROSS-BATCH PROCESSING WITH INCREMENTAL REDUCTION ===
            # Process batches and reduce incrementally to bound memory usage
            # Instead of holding all futures, we reduce each batch's results
            # with the running accumulator future
            running_accumulated = None  # Future holding accumulated result so far

            for batch_idx in range(num_batches):
                # Determine batch slice
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, images.shape[0])
                batch_slice = images[batch_start:batch_end]

                worker = self.filter_workers[batch_idx % len(self.filter_workers)]
                is_first_batch = (batch_idx == 0)

                # Submit filter task
                logging.info(f"[Pass {pass_idx+1}, Batch {batch_idx+1}/{num_batches}] Submitting filter -> {worker}")
                filter_future = self.client.submit(
                    _filter_batch_worker,
                    batch_slice,
                    self.config,
                    batch_idx,
                    output_path,
                    self.scattered_pixel_mask,
                    is_first_batch,
                    workers=[worker],
                    priority=10,
                    pure=False,
                )

                # Wait for filter to complete (sequential filtering due to RAM constraints)
                filtered_batch = filter_future.result()
                logging.info(f"[Pass {pass_idx+1}, Batch {batch_idx+1}/{num_batches}] Filter complete")

                # Submit CHUNKED correlation tasks (one chunk per worker)
                # This replaces per-pair mapping with local accumulation on workers
                chunk_futures = self._correlate_ensemble_batch_chunked(
                    filtered_batch,
                    scattered_cache,
                    scattered_masks,
                    scattered_predictor,
                    pass_idx,
                    output_path=output_path,
                    batch_start_index=batch_idx * self.batch_size,
                )

                # Gather chunk results (small number = num_corr_workers)
                # Network traffic reduced from O(N) to O(num_workers)
                logging.info(
                    f"[Pass {pass_idx+1}, Batch {batch_idx+1}/{num_batches}] "
                    f"Gathering {len(chunk_futures)} chunk results"
                )
                chunk_results = self.client.gather(chunk_futures)

                # Local reduction on main process (fast - only num_workers results)
                batch_reduced = chunk_results[0]
                for i in range(1, len(chunk_results)):
                    batch_reduced = _reduce_ensemble_results(batch_reduced, chunk_results[i])

                del chunk_futures
                del chunk_results

                # Merge with running accumulator
                if running_accumulated is None:
                    running_accumulated = batch_reduced
                else:
                    running_accumulated = _reduce_ensemble_results(
                        running_accumulated, batch_reduced
                    )
                del batch_reduced

                logging.info(
                    f"[Pass {pass_idx+1}, Batch {batch_idx+1}/{num_batches}] "
                    f"Batch reduced and merged"
                )

                # Periodic garbage collection to release worker memory
                # Every 2 batches to avoid too much overhead
                if batch_idx % 2 == 1:
                    self.client.run(gc.collect)

            # Final accumulated result is already on main process (no .result() needed)
            # The chunked approach gathers after each batch, so no deferred transfer
            assert running_accumulated is not None, "No batches processed - running_accumulated is None"
            pass_accumulated = running_accumulated

            # Clean up worker memory
            del running_accumulated
            self.client.run(gc.collect)
            logging.debug(f"[Pass {pass_idx + 1}] Released worker memory")

            accumulator.accumulate_batch(pass_accumulated, pass_idx=pass_idx)

            # Finalize this pass
            logging.info(f"[Pass {pass_idx + 1}] Finalizing pass (single-pass optimization)")
            pass_result = accumulator.finalize_pass(pass_idx, self.client, scattered_cache, predictor_field, output_path)

            # PROGRESSIVE SAVING: Append pass to ensemble_result.mat and clear memory
            # Pass accumulator to avoid expensive file reload (uses in-memory passes)
            self._append_ensemble_pass_progressive(
                pass_result, pass_idx, output_path, accumulator=accumulator
            )
            accumulator.clear_pass_data(pass_idx)

            # MEMORY CLEANUP: Clear scattered predictor from previous pass
            if scattered_predictor is not None:
                del scattered_predictor
                scattered_predictor = None

            # Force garbage collection on main process
            gc.collect()

            # Force garbage collection on ALL workers to release unmanaged memory
            # This is critical between passes to prevent OOM
            self.client.run(gc.collect)
            logging.info(f"[Pass {pass_idx + 1}] Triggered garbage collection on all workers")

            # Extract predictor field for next pass
            if pass_idx < num_passes - 1:
                predictor_field = self._extract_predictor_field(pass_result, pass_idx)
                logging.info(f"[Pass {pass_idx + 1}] Extracted predictor field for next pass")

        # PROGRESSIVE SAVING: Final result already assembled in ensemble_result.mat
        final_path = output_path / "ensemble_result.mat"
        logging.info(f"All passes saved progressively to {final_path}")
        return str(final_path)

    def _extract_predictor_field(self, pass_result, pass_idx: int) -> np.ndarray:
        """
        Extract predictor field from pass result for next pass.

        Masked windows have zero displacement (not infilled), which propagates
        naturally to the next pass. This prevents infilling artifacts at edges
        and boundaries.

        Parameters
        ----------
        pass_result : PIVEnsemblePassResult
            Result from current pass
        pass_idx : int
            Current pass index

        Returns
        -------
        np.ndarray
            Predictor field of shape (n_win_y+2, n_win_x+2, 2) containing [uy, ux]
            Padded by 1 on each edge for boundary extrapolation.
        """
        # Get displacement fields (masked windows have zero displacement)
        uy = pass_result.uy_mat.copy()
        ux = pass_result.ux_mat.copy()

        # Stack as [uy, ux] along last dimension
        predictor_field = np.stack([uy, ux], axis=-1).astype(np.float32)

        # Pad predictor field to allow for extrapolation at boundaries
        # mode='edge' replicates edge values (including zeros from masked windows)
        n_pre_x = 1
        n_post_x = 1
        n_pre_y = 1
        n_post_y = 1

        predictor_field = np.pad(
            predictor_field,
            ((n_pre_y, n_post_y), (n_pre_x, n_post_x), (0, 0)),
            mode='edge'
        )

        # Get statistics for non-zero values (excluding masked windows)
        nonzero_mask = (ux != 0) | (uy != 0)
        if nonzero_mask.any():
            ux_nz = ux[nonzero_mask]
            uy_nz = uy[nonzero_mask]
            logging.info(
                f"Predictor field extracted: shape={predictor_field.shape}, "
                f"non-zero ux range=[{ux_nz.min():.3f}, {ux_nz.max():.3f}], "
                f"non-zero uy range=[{uy_nz.min():.3f}, {uy_nz.max():.3f}], "
                f"n_masked={(~nonzero_mask).sum()}"
            )
        else:
            logging.info(
                f"Predictor field extracted: shape={predictor_field.shape}, "
                f"all zeros (all windows masked)"
            )

        return predictor_field

    def _append_ensemble_pass_progressive(
        self, pass_result, pass_idx: int, output_path: Path, accumulator=None
    ):
        """
        Append a single ensemble pass to ensemble_result.mat.

        For fresh runs (no resume):
            - Pass 1: Creates new ensemble_result.mat with just pass 1
            - Pass 2+: Uses passes from accumulator (no file reload needed)

        For resume runs:
            - Load existing file fully (to preserve stress/peakheight from prior passes)
            - Replace/append only the newly computed passes

        Parameters
        ----------
        pass_result : PIVEnsemblePassResult
            Result from current pass
        pass_idx : int
            Pass index
        output_path : Path
            Output directory
        accumulator : SinglePassAccumulator, optional
            Accumulator containing previous pass results. If provided, avoids
            expensive file reload by using in-memory passes.
        """
        from pivtools_cli.piv.piv_result import PIVEnsembleResult

        ensemble_filepath = output_path / "ensemble_result.mat"
        resume_from_pass = self.config.ensemble_resume_from_pass
        start_pass_idx = resume_from_pass - 1 if resume_from_pass > 0 else 0

        # Check if we're in resume mode and this is the first pass being saved
        is_resume_mode = resume_from_pass > 0
        is_first_resumed_pass = is_resume_mode and pass_idx == start_pass_idx

        if is_first_resumed_pass and ensemble_filepath.exists():
            # RESUME MODE: Load existing file fully to preserve all fields from prior passes
            logging.info(
                f"Pass {pass_idx + 1}: Resume mode - loading existing file to preserve prior passes"
            )
            existing_result = self._load_ensemble_result_from_file(
                ensemble_filepath, predictor_only=False
            )

            # Build new ensemble result with existing passes + new pass
            ensemble_result = PIVEnsembleResult()

            # Add existing passes up to (but not including) the resume point
            for i in range(start_pass_idx):
                if i < len(existing_result.passes):
                    ensemble_result.add_pass(existing_result.passes[i])

            # Add the newly computed pass
            ensemble_result.add_pass(pass_result)

            logging.info(
                f"Pass {pass_idx + 1}: Merged {start_pass_idx} existing passes + 1 new pass"
            )
        elif is_resume_mode and pass_idx > start_pass_idx:
            # Subsequent passes in resume mode - load from file and append
            logging.info(
                f"Pass {pass_idx + 1}: Resume mode - appending to existing file"
            )
            existing_result = self._load_ensemble_result_from_file(
                ensemble_filepath, predictor_only=False
            )

            ensemble_result = PIVEnsembleResult()
            for prev_pass in existing_result.passes:
                ensemble_result.add_pass(prev_pass)
            ensemble_result.add_pass(pass_result)

            logging.info(
                f"Pass {pass_idx + 1}: Appended to {len(existing_result.passes)} existing passes"
            )
        else:
            # FRESH RUN: Use accumulator's in-memory passes (no file I/O needed)
            ensemble_result = PIVEnsembleResult()

            if accumulator is not None and hasattr(accumulator, 'passes_results'):
                # Use passes from accumulator (already in memory)
                # passes_results already contains the current pass after finalize_pass
                for prev_pass in accumulator.passes_results:
                    ensemble_result.add_pass(prev_pass)
                logging.info(
                    f"Pass {pass_idx + 1}: Building ensemble from {len(ensemble_result.passes)} "
                    f"in-memory passes (no file reload)"
                )
            else:
                # Fallback: single pass only
                ensemble_result.add_pass(pass_result)
                logging.info(f"Pass {pass_idx + 1}: Creating ensemble_result.mat with pass 1")

        # Save back to ensemble_result.mat
        save_ensemble_result_distributed(
            ensemble_result,
            output_path,
            runs_to_save=self.config.ensemble_runs_0based,
            filename="ensemble_result.mat",
        )

        logging.info(f"Pass {pass_idx + 1}: Saved to {ensemble_filepath} (progressive saving)")

    def _load_ensemble_result_from_file(
        self, filepath: Path, predictor_only: bool = False
    ):
        """
        Load ensemble result from .mat file.

        Parameters
        ----------
        filepath : Path
            Path to ensemble_result.mat file
        predictor_only : bool
            If True, only load ux/uy fields (for predictor extraction).
            This significantly reduces memory usage and I/O time.

        Returns
        -------
        PIVEnsembleResult
            Loaded ensemble result with all passes (minimal data if predictor_only)
        """
        from pivtools_cli.piv.piv_result import PIVEnsembleResult, PIVEnsemblePassResult
        from scipy.io import loadmat

        if not filepath.exists():
            raise FileNotFoundError(f"Ensemble result file not found: {filepath}")

        # Load .mat file
        mat_data = loadmat(filepath, squeeze_me=False, struct_as_record=False)
        ensemble_data = mat_data["ensemble_result"]

        # ensemble_data is a struct array with shape (n_passes,) or (n_passes, 1)
        # Flatten to 1D if needed
        if isinstance(ensemble_data, np.ndarray):
            ensemble_data = ensemble_data.ravel()
        else:
            # Single element, convert to list
            ensemble_data = [ensemble_data]

        num_passes = len(ensemble_data)
        ensemble_result = PIVEnsembleResult()

        for pass_idx in range(num_passes):
            pass_struct = ensemble_data[pass_idx]

            # Helper to safely get field value
            def get_field(field_name, default=None):
                if hasattr(pass_struct, field_name):
                    val = getattr(pass_struct, field_name)
                    # Handle empty arrays
                    if isinstance(val, np.ndarray) and val.size == 0:
                        return default if default is not None else np.array([])
                    return val
                return default if default is not None else np.array([])

            if predictor_only:
                # Minimal load: only fields needed for predictor and sigma
                # - ux/uy: for predictor field extraction
                # - all sig_* fields: for sigma interpolation in resume
                # Skip stress, peakheight to reduce memory
                pass_result = PIVEnsemblePassResult(
                    ux_mat=get_field('ux'),
                    uy_mat=get_field('uy'),
                    UU_stress=None,
                    VV_stress=None,
                    UV_stress=None,
                    peakheight=None,
                    nan_reason=np.array([], dtype=np.int32),
                    sig_AB_x=get_field('sig_AB_x'),  # Needed for sigma interp
                    sig_AB_y=get_field('sig_AB_y'),  # Needed for sigma interp
                    sig_AB_xy=get_field('sig_AB_xy'),  # Needed for sigma interp
                    sig_A_x=get_field('sig_A_x'),  # Needed for sigma interp
                    sig_A_y=get_field('sig_A_y'),  # Needed for sigma interp
                    sig_A_xy=get_field('sig_A_xy'),  # Needed for sigma interp
                    b_mask=None,
                    pred_x=None,
                    pred_y=None,
                    window_size=(0, 0),
                    win_ctrs_x=None,
                    win_ctrs_y=None,
                )
            else:
                # Full load: all fields
                # Get nan_reason and convert to int32 if it's not empty
                nan_reason_raw = get_field('nan_reason', None)
                if nan_reason_raw is not None and isinstance(
                    nan_reason_raw, np.ndarray
                ) and nan_reason_raw.size > 0:
                    nan_reason = nan_reason_raw.astype(np.int32)
                else:
                    nan_reason = np.array([], dtype=np.int32)

                # Get b_mask and convert to bool if it's not empty
                b_mask_raw = get_field('b_mask', None)
                if b_mask_raw is not None and isinstance(
                    b_mask_raw, np.ndarray
                ) and b_mask_raw.size > 0:
                    b_mask = b_mask_raw.astype(bool)
                else:
                    b_mask = None

                # Get window_size
                window_size_raw = get_field('window_size', np.array([0, 0]))
                if isinstance(window_size_raw, np.ndarray) and window_size_raw.size >= 2:
                    window_size = tuple(window_size_raw.ravel()[:2])
                else:
                    window_size = (0, 0)

                pass_result = PIVEnsemblePassResult(
                    ux_mat=get_field('ux'),
                    uy_mat=get_field('uy'),
                    UU_stress=get_field('UU_stress'),
                    VV_stress=get_field('VV_stress'),
                    UV_stress=get_field('UV_stress'),
                    peakheight=get_field('peakheight'),
                    nan_reason=nan_reason,
                    sig_AB_x=get_field('sig_AB_x'),
                    sig_AB_y=get_field('sig_AB_y'),
                    sig_AB_xy=get_field('sig_AB_xy'),
                    sig_A_x=get_field('sig_A_x'),
                    sig_A_y=get_field('sig_A_y'),
                    sig_A_xy=get_field('sig_A_xy'),
                    b_mask=b_mask,
                    pred_x=get_field('pred_x', None),
                    pred_y=get_field('pred_y', None),
                    window_size=window_size,
                    win_ctrs_x=get_field('win_ctrs_x'),
                    win_ctrs_y=get_field('win_ctrs_y'),
                )
            ensemble_result.add_pass(pass_result)

        mode_str = "predictor-only" if predictor_only else "full"
        logging.info(
            f"Loaded ensemble result ({mode_str}) with {num_passes} passes "
            f"from {filepath}"
        )
        return ensemble_result

    def _process_instantaneous(
        self,
        images: da.Array,
        output_path: Path,
        vector_masks: Optional[List[np.ndarray]],
    ):
        """Instantaneous PIV with batched processing."""
        # Scatter immutable data
        scattered_cache, scattered_masks = self._scatter_immutable_data(vector_masks)

        # Process batches
        num_batches = (images.shape[0] + self.batch_size - 1) // self.batch_size
        logging.info(f"Processing {num_batches} batches for instantaneous PIV")

        all_saved_paths = []

        # Initialize first batch
        batch_idx = 0
        batch_start = 0
        batch_end = min(self.batch_size, images.shape[0])
        batch_slice = images[batch_start:batch_end]
        worker = self.filter_workers[batch_idx % len(self.filter_workers)]
        logging.info(f"Submitting batch {batch_idx} to filter worker {worker}")
        filter_future = self.client.submit(
            _filter_batch_worker,
            batch_slice,
            self.config,
            batch_idx,
            None,  # output_path (not used for instantaneous diagnostics)
            self.scattered_pixel_mask,  # Pass pixel mask for preprocessing
            True,  # is_first_batch - use all cores, no correlation running yet
            workers=[worker],
            priority=10,
            pure=False,
        )

        while batch_idx < num_batches:
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, images.shape[0])

            # Wait for current filter
            filtered_batch = filter_future.result()
            logging.info(f"[Batch {batch_idx+1}/{num_batches}] Filtering complete")

            # Start correlation for this batch
            logging.info(f"[Batch {batch_idx+1}/{num_batches}] Starting correlation...")
            corr_futures = self._process_instantaneous_batch(
                filtered_batch,
                output_path,
                scattered_cache,
                scattered_masks,
                start_frame=batch_start + 1,
            )

            # Submit next filter if there is one
            next_batch_idx = batch_idx + 1
            if next_batch_idx < num_batches:
                next_batch_start = next_batch_idx * self.batch_size
                next_batch_end = min(next_batch_start + self.batch_size, images.shape[0])
                next_batch_slice = images[next_batch_start:next_batch_end]
                next_worker = self.filter_workers[next_batch_idx % len(self.filter_workers)]
                logging.info(f"Submitting batch {next_batch_idx} to filter worker {next_worker}")
                filter_future = self.client.submit(
                    _filter_batch_worker,
                    next_batch_slice,
                    self.config,
                    next_batch_idx,
                    None,  # output_path (not used for instantaneous diagnostics)
                    self.scattered_pixel_mask,  # Pass pixel mask for preprocessing
                    False,  # is_first_batch - use reduced cores, correlation is running
                    workers=[next_worker],
                    priority=10,
                    pure=False,
                )

            # Wait for current correlation
            saved_paths = self.client.gather(corr_futures)
            all_saved_paths.extend(saved_paths)
            logging.info(f"[Batch {batch_idx+1}/{num_batches}] Correlation complete")

            batch_idx += 1

        logging.info("Instantaneous PIV processing completed")
        logging.info(f"Returning {len(all_saved_paths)} saved paths")
        return all_saved_paths

    def _scatter_immutable_data(
        self, vector_masks: Optional[List[np.ndarray]]
    ) -> Tuple:
        """Scatter cache, masks, and pixel mask once (broadcast to all workers)."""
        # Create and scatter correlator cache
        temp_correlator = make_correlator_backend(
            self.config,
            ensemble=(self.mode == "ensemble"),
        )
        correlator_cache = temp_correlator.get_cache_data()
        scattered_cache = self.client.scatter(correlator_cache, broadcast=True)
        logging.info("Broadcast correlator cache to all workers")

        # Scatter vector masks if present (for correlation validation)
        scattered_masks = None
        if vector_masks:
            scattered_masks = self.client.scatter(vector_masks, broadcast=True)
            mask_size = sum(m.nbytes for m in vector_masks) / 1024
            logging.info(f"Broadcast vector masks ({mask_size:.1f} KB)")

        # Scatter pixel mask if present (for preprocessing)
        scattered_pixel_mask = None
        if self.pixel_mask is not None:
            scattered_pixel_mask = self.client.scatter(self.pixel_mask, broadcast=True)
            mask_size = self.pixel_mask.nbytes / 1024
            logging.info(f"Broadcast pixel mask ({mask_size:.1f} KB)")

        # Store for use in filter workers
        self.scattered_pixel_mask = scattered_pixel_mask

        return scattered_cache, scattered_masks

    def _submit_filter_batches(
        self, images: da.Array, num_batches: int
    ) -> List:
        """Submit all batch filtering tasks (parallel)."""
        filter_futures = []

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, images.shape[0])
            batch_slice = images[batch_start:batch_end]

            # Round-robin across filter workers
            worker = self.filter_workers[batch_idx % len(self.filter_workers)]

            logging.info(f"Submitting batch {batch_idx} (frames {batch_start+1}-{batch_end}) to filter worker {worker}")

            future = self.client.submit(
                _filter_batch_worker,
                batch_slice,
                self.config,
                batch_idx,
                None,  # output_path
                None,  # pixel_mask
                batch_idx == 0,  # is_first_batch - only first uses all cores
                workers=[worker],
                priority=10,
                pure=False,
            )
            filter_futures.append(future)

        return filter_futures

    def _correlate_ensemble_batch_chunked(
        self,
        filtered_batch: np.ndarray,
        scattered_cache,
        scattered_masks,
        scattered_predictor,
        pass_idx: int,
        output_path: Optional[Path] = None,
        batch_start_index: int = 0,
    ) -> List:
        """
        Submit correlation tasks in CHUNKS (one chunk per worker).

        Instead of mapping individual pairs (O(N) network transfers), this method
        splits the batch into chunks where each worker processes multiple pairs
        locally and returns a single accumulated result.

        Parameters
        ----------
        filtered_batch : np.ndarray
            Filtered image batch of shape (N, 2, H, W)
        scattered_cache : dict
            Pre-scattered correlator cache
        scattered_masks : Optional
            Pre-scattered vector masks
        scattered_predictor : Optional
            Pre-scattered predictor field
        pass_idx : int
            Current pass index
        output_path : Optional[Path]
            Output directory for diagnostic images
        batch_start_index : int
            Global index of first pair in this batch (for diagnostics)

        Returns
        -------
        List
            List of futures for chunk results (length <= num_corr_workers)
        """
        n_images = filtered_batch.shape[0]
        n_workers = len(self.corr_workers)

        # Calculate chunk size: each worker gets roughly equal work
        # ceil division ensures all pairs are covered
        chunk_size = max(1, (n_images + n_workers - 1) // n_workers)

        # Split into chunks and scatter to workers FIRST
        # This avoids embedding large arrays in the task graph (72MB+ warning)
        chunks = []
        chunk_start_indices = []
        for i in range(0, n_images, chunk_size):
            chunks.append(filtered_batch[i : i + chunk_size])
            chunk_start_indices.append(batch_start_index + i)

        # Scatter chunks to workers (efficient transfer, not embedded in graph)
        scattered_chunks = self.client.scatter(chunks, workers=self.corr_workers)

        # Submit tasks using scattered references
        futures = []
        for idx, scattered_chunk in enumerate(scattered_chunks):
            worker_idx = idx % n_workers
            worker = self.corr_workers[worker_idx]

            future = self.client.submit(
                _correlate_chunk_sum_worker,
                scattered_chunk,  # Reference to scattered data, not raw array
                chunk_start_indices[idx],
                self.config,
                scattered_cache,
                scattered_masks,
                scattered_predictor,
                pass_idx,
                output_path,
                workers=[worker],
                pure=False
            )
            futures.append(future)

        logging.debug(
            f"[Pass {pass_idx+1}] Submitted {len(futures)} chunk tasks "
            f"({n_images} pairs / {chunk_size} per chunk)"
        )

        return futures

    def _process_instantaneous_batch(
        self,
        filtered_batch: np.ndarray,
        output_path: Path,
        scattered_cache,
        scattered_masks,
        start_frame: int,
    ) -> List[str]:
        """Process instantaneous PIV for one batch."""
        # Split into individual pairs
        pairs = [filtered_batch[i] for i in range(filtered_batch.shape[0])]
        frame_numbers = list(range(start_frame, start_frame + len(pairs)))

        # Scatter to correlation workers
        scattered_pairs = self.client.scatter(pairs, workers=self.corr_workers)

        # Submit PIV tasks
        from pivtools_cli.piv.piv import _process_and_save_single_pair

        futures = self.client.map(
            _process_and_save_single_pair,
            scattered_pairs,
            frame_numbers,
            config=self.config,
            scattered_masks=scattered_masks,
            scattered_cache=scattered_cache,
            output_path=output_path,
            runs_to_save=self.config.instantaneous_runs_0based,
            vector_format=self.config.vector_format,
            workers=self.corr_workers,
            pure=False,
        )

        # Gather saved paths
        return futures


# Worker functions

def _reduce_ensemble_results(r1: dict, r2: dict) -> dict:
    """
    Combine two ensemble correlation results on a worker.

    Used for tree reduction to accumulate results without
    transferring large correlation planes to main process.

    Parameters
    ----------
    r1, r2 : dict
        Results from correlate_batch_for_accumulation containing:
        - corr_AA_sum, corr_BB_sum, corr_AB_sum: Flattened correlation planes
        - warp_A_sum, warp_B_sum: Warped image sums
        - n_images: Image count
        - n_win_x, n_win_y: Grid dimensions
        - smoothed_predictor, vector_mask: Pass-level metadata

    Returns
    -------
    dict
        Combined result with summed arrays
    """
    return {
        "corr_AA_sum": r1["corr_AA_sum"] + r2["corr_AA_sum"],
        "corr_BB_sum": r1["corr_BB_sum"] + r2["corr_BB_sum"],
        "corr_AB_sum": r1["corr_AB_sum"] + r2["corr_AB_sum"],
        "warp_A_sum": r1["warp_A_sum"] + r2["warp_A_sum"],
        "warp_B_sum": r1["warp_B_sum"] + r2["warp_B_sum"],
        "n_images": r1["n_images"] + r2["n_images"],
        "n_win_x": r1["n_win_x"],
        "n_win_y": r1["n_win_y"],
        "smoothed_predictor": r1.get("smoothed_predictor"),
        "vector_mask": r1.get("vector_mask"),
    }

def _filter_batch_worker(
    batch_images: da.Array,
    config: Config,
    batch_idx: int,
    output_path: Optional[Path] = None,
    pixel_mask: Optional[np.ndarray] = None,
    is_first_batch: bool = False,
) -> np.ndarray:
    """
    Apply all filters to batch on filter worker.

    Uses multi-threading for CPU-intensive operations (POD SVD, etc.).
    Thread count is controlled by config.omp_threads for consistency.

    Args:
        batch_images: Dask array slice for this batch
        config: Configuration object
        batch_idx: Batch index (for diagnostics)
        output_path: Output directory for diagnostic images
        pixel_mask: Boolean mask (H, W) where True = masked regions to zero
        is_first_batch: Unused, kept for API compatibility.
    """
    import os

    # Use config.omp_threads for all batches (consistent threading)
    worker_cores = int(config.omp_threads)

    os.environ["OMP_NUM_THREADS"] = str(worker_cores)
    os.environ["MKL_NUM_THREADS"] = str(worker_cores)

    # Load batch with threading scheduler
    with dask.config.set(scheduler='threads', num_workers=worker_cores):
        batch = batch_images.compute()

    # Apply all filters (temporal and spatial) including pixel masking
    from pivtools_cli.preprocessing.preprocess import apply_filters_to_batch

    save_diagnostics = (
        hasattr(config, 'ensemble_save_diagnostics') and
        config.ensemble_save_diagnostics and
        batch_idx == 0
    )

    batch_filtered = apply_filters_to_batch(
        batch,
        config,
        save_diagnostics=save_diagnostics,
        output_dir=output_path,
        batch_idx=batch_idx,
        pixel_mask=pixel_mask,
    )

    return batch_filtered


def _correlate_chunk_sum_worker(
    image_chunk: np.ndarray,
    chunk_start_idx: int,
    config: Config,
    scattered_cache: dict,
    scattered_masks: Optional[List[np.ndarray]],
    scattered_predictor: Optional[np.ndarray],
    pass_idx: int,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Process a CHUNK of images and return SUMMED correlation planes.

    Keeps summation local to worker RAM, minimizing network traffic.
    Each worker processes multiple pairs and returns a single accumulated result.

    Parameters
    ----------
    image_chunk : np.ndarray
        Chunk of image pairs, shape (N_chunk, 2, H, W)
    chunk_start_idx : int
        Global index of first pair in this chunk (for diagnostics)
    config : Config
        Configuration object
    scattered_cache : dict
        Pre-scattered correlator cache
    scattered_masks : Optional[List[np.ndarray]]
        Pre-scattered vector masks
    scattered_predictor : Optional[np.ndarray]
        Pre-scattered predictor field
    pass_idx : int
        Current pass index
    output_path : Optional[Path]
        Output directory for diagnostic images

    Returns
    -------
    dict
        Accumulated correlation results with keys:
        - corr_AA_sum, corr_BB_sum, corr_AB_sum: Summed correlation planes
        - warp_A_sum, warp_B_sum: Summed warped images
        - n_images: Total image count in chunk
        - n_win_x, n_win_y: Grid dimensions
        - smoothed_predictor, vector_mask: Metadata
    """
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    correlator = EnsembleCorrelatorCPU(
        config,
        precomputed_cache=scattered_cache,
        vector_masks=scattered_masks,
    )

    accumulated_result = None

    # Process each pair in the chunk LOCALLY (no network transfer)
    for local_i in range(image_chunk.shape[0]):
        pair = image_chunk[local_i : local_i + 1]  # Shape (1, 2, H, W)
        global_idx = chunk_start_idx + local_i

        # Save diagnostics only for first pair of entire batch
        save_diagnostics = (
            hasattr(config, 'ensemble_save_diagnostics') and
            config.ensemble_save_diagnostics and
            pass_idx == 0 and
            global_idx == 0
        )

        result = correlator.correlate_batch_for_accumulation(
            pair,
            config,
            pass_idx=pass_idx,
            predictor_field=scattered_predictor,
            save_diagnostics=save_diagnostics,
            output_path=str(output_path) if output_path else None,
            is_first_batch=(global_idx == 0),
        )

        if accumulated_result is None:
            accumulated_result = result
        else:
            # In-place summation to minimize memory allocations
            accumulated_result["corr_AA_sum"] += result["corr_AA_sum"]
            accumulated_result["corr_BB_sum"] += result["corr_BB_sum"]
            accumulated_result["corr_AB_sum"] += result["corr_AB_sum"]
            accumulated_result["warp_A_sum"] += result["warp_A_sum"]
            accumulated_result["warp_B_sum"] += result["warp_B_sum"]
            accumulated_result["n_images"] += result["n_images"]

    return accumulated_result
