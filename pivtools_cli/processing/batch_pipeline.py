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

        logging.info(f"Processing {num_passes} pass(es) with {num_batches} batches each for ensemble PIV")

        # Multi-pass loop
        predictor_field = None  # No predictor for pass 0

        for pass_idx in range(num_passes):
            logging.info("")
            logging.info(f"======== PASS {pass_idx + 1}/{num_passes} ========")

            # Scatter predictor field for this pass (if pass > 0)
            scattered_predictor = None
            if predictor_field is not None:
                scattered_predictor = self.client.scatter(predictor_field, broadcast=True)
                logging.info(f"[Pass {pass_idx + 1}] Broadcast predictor field from previous pass")

            # Process all batches for this pass
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, images.shape[0])
                batch_slice = images[batch_start:batch_end]

                # Assign to filter worker
                worker = self.filter_workers[batch_idx % len(self.filter_workers)]
                logging.info(f"[Pass {pass_idx + 1}] Submitting batch {batch_idx} to filter worker {worker}")

                # Submit filter task and wait
                future = self.client.submit(
                    _filter_batch_worker,
                    batch_slice,
                    self.config,
                    batch_idx,
                    workers=[worker],
                    priority=10,
                    pure=False,
                )
                filtered_batch = future.result()
                logging.info(f"[Pass {pass_idx + 1}, Batch {batch_idx+1}/{num_batches}] Filtering complete")

                # Distribute to correlation workers and accumulate
                self._correlate_and_accumulate_batch(
                    filtered_batch,
                    accumulator,
                    scattered_cache,
                    scattered_masks,
                    scattered_predictor,
                    pass_idx,
                    batch_idx+1,
                    num_batches,
                )

            # Finalize this pass
            logging.info(f"[Pass {pass_idx + 1}] Finalizing pass (single-pass optimization)")
            pass_result = accumulator.finalize_pass(pass_idx, self.client, scattered_cache, predictor_field, output_path)

            # Extract predictor field for next pass
            if pass_idx < num_passes - 1:
                predictor_field = self._extract_predictor_field(pass_result, pass_idx)
                logging.info(f"[Pass {pass_idx + 1}] Extracted predictor field for next pass")

        # Get final ensemble result (all passes combined)
        logging.info("Assembling final ensemble result from all passes")
        ensemble_result = accumulator.get_ensemble_result()

        # Save
        saved_path = save_ensemble_result_distributed(
            ensemble_result,
            output_path,
            runs_to_save=self.config.ensemble_runs_0based,
        )

        logging.info(f"Ensemble result saved to {saved_path}")
        return saved_path

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

    def _correlate_and_accumulate_batch(
        self,
        filtered_batch: np.ndarray,
        accumulator,
        scattered_cache,
        scattered_masks,
        scattered_predictor,
        pass_idx: int,
        batch_idx: int,
        total_batches: int,
    ):
        """Correlate batch and accumulate to ensemble result."""
        # Split into individual pairs
        pairs = [filtered_batch[i] for i in range(filtered_batch.shape[0])]

        # Scatter pairs to correlation workers
        scattered_pairs = self.client.scatter(pairs, workers=self.corr_workers)

        # Submit correlation tasks
        corr_futures = self.client.map(
            _correlate_ensemble_pair_worker,
            scattered_pairs,
            config=self.config,
            scattered_cache=scattered_cache,
            scattered_masks=scattered_masks,
            scattered_predictor=scattered_predictor,
            pass_idx=pass_idx,
            workers=self.corr_workers,
            pure=False,
        )

        # Gather results and accumulate
        results = self.client.gather(corr_futures)
        for result in results:
            accumulator.accumulate_batch(result, pass_idx=pass_idx)

        logging.info(f"[Pass {pass_idx + 1}, Batch {batch_idx}/{total_batches}] Correlation complete")

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
    from pivtools_cli.preprocessing.preprocess import apply_filters_to_batch
    batch_filtered = apply_filters_to_batch(batch, config)

    return batch_filtered


def _correlate_ensemble_pair_worker(
    image_pair: np.ndarray,
    config: Config,
    scattered_cache: dict,
    scattered_masks: Optional[List[np.ndarray]],
    scattered_predictor: Optional[np.ndarray],
    pass_idx: int,
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

    # Correlate and return sums (for accumulation)
    result = correlator.correlate_batch_for_accumulation(
        image_pair[np.newaxis, ...],  # Add batch dimension
        config,
        pass_idx=pass_idx,
        predictor_field=scattered_predictor,
    )

    return result
