"""
Unified batched PIV pipeline for HPC clusters.

Supports:
- Flexible worker allocation (5-100+ cores)
- Both instantaneous and ensemble modes
- Temporal and spatial filters
- Single-pass ensemble correlation
"""

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
    ):
        """
        Unified entry point for PIV processing.

        Args:
            images: Dask array of shape (N, 2, H, W)
            output_path: Directory for saving results
            vector_masks: Pre-computed masks for each pass

        Returns:
            List of saved file paths (instantaneous) or single path (ensemble)
        """
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
        """Ensemble PIV with single-pass correlation accumulation and multi-pass support."""
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

            # Load the existing ensemble result
            existing_result = self._load_ensemble_result_from_file(resume_path)

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

        # Multi-pass loop with pipelined batch processing
        # predictor_field is already initialized: None if fresh start, or loaded from resume file

        for pass_idx in range(start_pass_idx, num_passes):
            logging.info("")
            logging.info(f"======== PASS {pass_idx + 1}/{num_passes} ========")

            # Scatter predictor field for this pass (if pass > 0)
            scattered_predictor = None
            if predictor_field is not None:
                scattered_predictor = self.client.scatter(predictor_field, broadcast=True)
                logging.info(f"[Pass {pass_idx + 1}] Broadcast predictor field from previous pass")

            # === PIPELINED BATCH PROCESSING ===
            # Initialize first batch
            batch_idx = 0
            batch_slice = images[0:min(self.batch_size, images.shape[0])]
            worker = self.filter_workers[0]
            logging.info(f"[Pass {pass_idx+1}] Initializing pipeline: batch 0 -> filter worker {worker}")

            filter_future = self.client.submit(
                _filter_batch_worker,
                batch_slice,
                self.config,
                batch_idx,
                output_path,  # Pass output_path for diagnostics
                workers=[worker],
                priority=10,
                pure=False,
            )

            # Process batches with overlapping filter/correlation
            while batch_idx < num_batches:
                # Wait for current filter to complete
                filtered_batch = filter_future.result()
                logging.info(f"[Pass {pass_idx+1}, Batch {batch_idx+1}/{num_batches}] Filter complete")

                # Determine if this is the first batch (for diagnostic saving)
                is_first_batch = (batch_idx == 0)

                # Start correlation for THIS batch (non-blocking)
                corr_futures = self._correlate_ensemble_batch_async(
                    filtered_batch,
                    scattered_cache,
                    scattered_masks,
                    scattered_predictor,
                    pass_idx,
                    output_path=output_path,
                    is_first_batch=is_first_batch,
                )

                # OVERLAP: Submit NEXT filter while THIS batch correlates
                next_batch_idx = batch_idx + 1
                if next_batch_idx < num_batches:
                    next_start = next_batch_idx * self.batch_size
                    next_end = min(next_start + self.batch_size, images.shape[0])
                    next_slice = images[next_start:next_end]
                    next_worker = self.filter_workers[next_batch_idx % len(self.filter_workers)]

                    logging.info(
                        f"[Pass {pass_idx+1}] Pipeline: batch {next_batch_idx} -> "
                        f"filter worker {next_worker} (while batch {batch_idx+1} correlates)"
                    )

                    filter_future = self.client.submit(
                        _filter_batch_worker,
                        next_slice,
                        self.config,
                        next_batch_idx,
                        output_path,  # Pass output_path for diagnostics
                        workers=[next_worker],
                        priority=10,
                        pure=False,
                    )

                # Wait for current correlation to complete and accumulate
                results = self.client.gather(corr_futures)
                for result in results:
                    accumulator.accumulate_batch(result, pass_idx=pass_idx)

                logging.info(f"[Pass {pass_idx+1}, Batch {batch_idx+1}/{num_batches}] Correlation complete")

                batch_idx += 1

            # Finalize this pass
            logging.info(f"[Pass {pass_idx + 1}] Finalizing pass (single-pass optimization)")
            pass_result = accumulator.finalize_pass(pass_idx, self.client, scattered_cache, predictor_field, output_path)

            # PROGRESSIVE SAVING: Append pass to ensemble_result.mat and clear memory
            self._append_ensemble_pass_progressive(pass_result, pass_idx, output_path)
            accumulator.clear_pass_data(pass_idx)

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
        self, pass_result, pass_idx: int, output_path: Path
    ):
        """
        Append a single ensemble pass to ensemble_result.mat.

        For pass 1: Creates new ensemble_result.mat with just pass 1
        For pass 2+: Loads existing ensemble_result.mat, appends pass, saves back

        Parameters
        ----------
        pass_result : PIVEnsemblePassResult
            Result from current pass
        pass_idx : int
            Pass index
        output_path : Path
            Output directory
        """
        from pivtools_cli.piv.piv_result import PIVEnsembleResult
        from scipy.io import loadmat

        ensemble_filepath = output_path / "ensemble_result.mat"

        if pass_idx == 0:
            # First pass: Create new ensemble result
            ensemble_result = PIVEnsembleResult()
            ensemble_result.add_pass(pass_result)
            logging.info(f"Pass {pass_idx + 1}: Creating ensemble_result.mat with pass 1")
        else:
            # Subsequent passes: Load existing, append, save back
            logging.info(f"Pass {pass_idx + 1}: Loading existing ensemble_result.mat to append pass {pass_idx + 1}")

            # Load existing ensemble result
            ensemble_result = self._load_ensemble_result_from_file(ensemble_filepath)

            # Append new pass
            ensemble_result.add_pass(pass_result)
            logging.info(f"Pass {pass_idx + 1}: Appended to ensemble result (now has {len(ensemble_result.passes)} passes)")

        # Save back to ensemble_result.mat
        save_ensemble_result_distributed(
            ensemble_result,
            output_path,
            runs_to_save=self.config.ensemble_runs_0based,
            filename="ensemble_result.mat",
        )

        logging.info(f"Pass {pass_idx + 1}: Saved to {ensemble_filepath} (progressive saving)")

    def _load_ensemble_result_from_file(self, filepath: Path):
        """
        Load ensemble result from .mat file.

        Parameters
        ----------
        filepath : Path
            Path to ensemble_result.mat file

        Returns
        -------
        PIVEnsembleResult
            Loaded ensemble result with all passes
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

            # Get nan_reason and convert to int32 if it's not empty
            nan_reason_raw = get_field('nan_reason', None)
            if nan_reason_raw is not None and isinstance(nan_reason_raw, np.ndarray) and nan_reason_raw.size > 0:
                nan_reason = nan_reason_raw.astype(np.int32)
            else:
                nan_reason = np.array([], dtype=np.int32)

            # Get b_mask and convert to bool if it's not empty
            b_mask_raw = get_field('b_mask', None)
            if b_mask_raw is not None and isinstance(b_mask_raw, np.ndarray) and b_mask_raw.size > 0:
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

        logging.info(f"Loaded ensemble result with {num_passes} passes from {filepath}")
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
        """Scatter cache and masks once (broadcast to all workers)."""
        # Create and scatter correlator cache
        temp_correlator = make_correlator_backend(
            self.config,
            ensemble=(self.mode == "ensemble"),
        )
        correlator_cache = temp_correlator.get_cache_data()
        scattered_cache = self.client.scatter(correlator_cache, broadcast=True)
        logging.info("Broadcast correlator cache to all workers")

        # Scatter masks if present
        scattered_masks = None
        if vector_masks:
            scattered_masks = self.client.scatter(vector_masks, broadcast=True)
            mask_size = sum(m.nbytes for m in vector_masks) / 1024
            logging.info(f"Broadcast vector masks ({mask_size:.1f} KB)")

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
                workers=[worker],
                priority=10,
                pure=False,
            )
            filter_futures.append(future)

        return filter_futures

    def _correlate_ensemble_batch_async(
        self,
        filtered_batch: np.ndarray,
        scattered_cache,
        scattered_masks,
        scattered_predictor,
        pass_idx: int,
        output_path: Optional[Path] = None,
        is_first_batch: bool = False,
    ) -> List:
        """
        Submit correlation tasks and return futures (non-blocking).

        Returns futures instead of blocking on gather, allowing filter/correlation overlap.

        Parameters
        ----------
        filtered_batch : np.ndarray
            Filtered image batch
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
        is_first_batch : bool
            Whether this is the first batch (for diagnostics)

        Returns
        -------
        List
            List of futures for correlation results
        """
        # Split into individual pairs
        pairs = [filtered_batch[i] for i in range(filtered_batch.shape[0])]
        pair_indices = list(range(len(pairs)))

        # Scatter pairs to correlation workers
        scattered_pairs = self.client.scatter(pairs, workers=self.corr_workers)

        # Submit correlation tasks (returns futures immediately)
        corr_futures = self.client.map(
            _correlate_ensemble_pair_worker,
            scattered_pairs,
            pair_indices,  # Pass pair index for diagnostic check
            config=self.config,
            scattered_cache=scattered_cache,
            scattered_masks=scattered_masks,
            scattered_predictor=scattered_predictor,
            pass_idx=pass_idx,
            output_path=output_path,
            is_first_batch=is_first_batch,
            workers=self.corr_workers,
            pure=False,
        )

        return corr_futures

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

def _filter_batch_worker(
    batch_images: da.Array,
    config: Config,
    batch_idx: int,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Apply all filters to batch on filter worker.

    Uses multi-threading for CPU-intensive operations (POD SVD, etc.).
    Sets OMP_NUM_THREADS to use all cores on this worker.
    """
    import os

    # Use ALL cores on this worker
    worker_cores = os.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(worker_cores)
    os.environ["MKL_NUM_THREADS"] = str(worker_cores)

    # Load batch with threading scheduler
    with dask.config.set(scheduler='threads', num_workers=worker_cores):
        batch = batch_images.compute()

    # Apply all filters (temporal and spatial)
    # Pass diagnostic parameters for first batch
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
    )

    return batch_filtered


def _correlate_ensemble_pair_worker(
    image_pair: np.ndarray,
    pair_idx: int,
    config: Config,
    scattered_cache: dict,
    scattered_masks: Optional[List[np.ndarray]],
    scattered_predictor: Optional[np.ndarray],
    pass_idx: int,
    output_path: Optional[Path] = None,
    is_first_batch: bool = False,
) -> dict:
    """
    Correlate single pair for ensemble accumulation.

    Returns correlation plane sums (AA, BB, AB) and warp sums.
    """
    import os

    # Single thread per worker (parallelism across workers)
    os.environ["OMP_NUM_THREADS"] = "1"

    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    correlator = EnsembleCorrelatorCPU(
        config,
        precomputed_cache=scattered_cache,
        vector_masks=scattered_masks,
    )

    # Determine if we should save diagnostics (first pair of first batch)
    save_diagnostics = (
        hasattr(config, 'ensemble_save_diagnostics') and
        config.ensemble_save_diagnostics and
        is_first_batch and
        pair_idx == 0
    )

    # Correlate and return sums (for accumulation)
    result = correlator.correlate_batch_for_accumulation(
        image_pair[np.newaxis, ...],  # Add batch dimension
        config,
        pass_idx=pass_idx,
        predictor_field=scattered_predictor,
        save_diagnostics=save_diagnostics,
        output_path=str(output_path) if output_path else None,
        is_first_batch=is_first_batch,
    )

    return result
