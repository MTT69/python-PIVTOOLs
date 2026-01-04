"""
Dask-Native Ensemble PIV Processing

Entry point for ensemble PIV with true Dask patterns:
- rechunk: Group images for batched processing
- map_blocks: Apply filters lazily per-chunk
- persist: Cache filtered chunks on workers
- submit: One task per chunk for correlation
- gather: Collect correlation planes to client for reduction

Usage:
    python -m pivtools_core.ensemble
"""
from pivtools_core.config import Config
import os
config = Config()
omp_threads = str(config.omp_threads)
os.environ["OMP_NUM_THREADS"] = omp_threads

import gc
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

from dask.distributed import Client, as_completed

from pivtools_core.validation import (
    validate_config,
    log_validation_result,
)
from pivtools_core.image_handling.load_images import (
    load_images,
    load_mask_for_camera,
    compute_vector_mask,
)
from pivtools_cli.piv_cluster.cluster import start_cluster
from pivtools_cli.piv.save_results import (
    get_ensemble_output_path,
    save_ensemble_result_distributed,
    save_ensemble_coordinates_from_config_distributed,
)
from pivtools_cli.piv.piv_result import PIVEnsembleResult
from pivtools_cli.piv.piv_backend.factory import make_correlator_backend
from pivtools_cli.piv.piv_backend.single_pass_accumulator import SinglePassAccumulator
from pivtools_cli.processing.dask_pipeline import (
    rechunk_for_batched_processing,
    create_filter_pipeline,
    scatter_immutable_data,
    correlate_and_reduce_on_worker,
    reduce_ensemble_results,
    extract_predictor_field,
)


logger = logging.getLogger(__name__)

# Global references for clean shutdown
_client = None
_cluster = None
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle termination signals for clean shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    logger.info(f"Received signal {sig_name}, initiating clean shutdown...")
    print(f"\n[CANCELLED] Received signal {sig_name}, shutting down...", flush=True)

    # Close Dask client and cluster if they exist
    try:
        if _client is not None:
            logger.info("Cancelling pending futures...")
            # Cancel all pending futures to stop work immediately
            try:
                _client.cancel(_client.futures, force=True)
            except Exception:
                pass  # May fail if no futures or already cancelled

            logger.info("Closing Dask client...")
            _client.close(timeout=5)
    except Exception as e:
        logger.warning(f"Error closing client: {e}")

    try:
        if _cluster is not None:
            logger.info("Closing Dask cluster...")
            _cluster.close(timeout=5)
    except Exception as e:
        logger.warning(f"Error closing cluster: {e}")

    logger.info("Shutdown complete.")
    print("[CANCELLED] Shutdown complete.", flush=True)

    # Force exit to ensure all subprocesses terminate
    os._exit(1)


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def run_ensemble_piv(
    config: Config,
    client: Client,
    camera_num: int,
    source_path: Path,
    output_path: Path,
    base_path: Path,
    vector_masks: Optional[List] = None,
    pixel_mask: Optional = None,
) -> str:
    """
    Run Dask-native ensemble PIV processing for one camera.

    Flow:
    1. Load images (lazy)
    2. Rechunk for batched processing
    3. Apply all filters via map_blocks
    4. Persist filtered chunks on workers
    5. For each pass:
       - Submit correlation tasks for all chunks in parallel
       - Gather correlation planes to client
       - Reduce batch results
       - Finalize pass (Gaussian fitting, outlier detection, infilling)
       - Extract predictor for next pass
    6. Save ensemble result

    Args:
        config: Configuration object
        client: Dask distributed client
        camera_num: Camera number to process
        source_path: Path to source images
        output_path: Path for output files
        base_path: Base path for intermediate outputs (e.g., filter debug)
        vector_masks: Pre-computed vector masks per pass
        pixel_mask: Pixel mask for preprocessing

    Returns:
        Path to saved ensemble result
    """
    # 1. Load images (lazy)
    logger.info(f"Loading images for camera {camera_num}...")
    images = load_images(camera_num, config, source=source_path)
    logger.info(f"  Loaded: shape={images.shape}, {len(images.chunks[0])} chunks")

    # 2. Scatter immutable data once
    logger.info("Scattering immutable data...")
    scattered = scatter_immutable_data(
        client, config, vector_masks, pixel_mask, ensemble=True
    )

    # 3. Rechunk for batched processing
    batch_size = config.batch_size
    logger.info(f"Rechunking to batch_size={batch_size}...")
    images = rechunk_for_batched_processing(images, batch_size)
    logger.info(f"  Rechunked: {len(images.chunks[0])} chunks of size {images.chunks[0][0]}")

    # 4. Apply all filters via map_blocks
    logger.info("Creating filter pipeline...")
    # Pass base_path to save intermediate filter outputs when filters are configured
    save_intermediate = base_path if config.filters else None
    images = create_filter_pipeline(images, config, pixel_mask, save_intermediate_base=save_intermediate)

    # 5. Persist filtered chunks on workers (don't wait - enables pipelining!)
    logger.info("Persisting filtered images on workers...")
    images = images.persist()
    # NOTE: No wait() here! Dask handles dependencies automatically.
    # First pass correlation tasks will start as their chunks become ready.

    # 6. Multi-pass ensemble processing
    from dask.distributed import futures_of
    block_futures = futures_of(images)
    num_chunks = len(block_futures)
    num_passes = config.ensemble_num_passes
    accumulator = SinglePassAccumulator(config, vector_masks)
    predictor_field = None

    # Scatter config once to avoid repeated serialization
    scattered_config = client.scatter(config, broadcast=True)

    logger.info(f"Processing {num_passes} passes with {num_chunks} chunks each...")

    for pass_idx in range(num_passes):
        if _shutdown_requested:
            logger.info("Shutdown requested, stopping...")
            break

        logger.info("")
        logger.info(f"======== PASS {pass_idx + 1}/{num_passes} ========")

        # Scatter predictor for this pass
        scattered_predictor = None
        if predictor_field is not None:
            scattered_predictor = client.scatter(predictor_field, broadcast=True)
            logger.info(f"  Broadcast predictor field from previous pass")

        # Worker-side accumulation: distribute chunks across workers
        # Each worker processes multiple chunks and returns one accumulated result
        # This reduces network traffic from O(num_chunks) to O(num_workers)
        workers = list(client.scheduler_info()["workers"].keys())
        num_workers = len(workers)
        chunks_per_worker = (num_chunks + num_workers - 1) // num_workers

        logger.info(f"  Distributing {num_chunks} chunks across {num_workers} workers...")
        logger.info(f"  (~{chunks_per_worker} chunks per worker)")

        worker_futures = []
        for i, worker in enumerate(workers):
            start_idx = i * chunks_per_worker
            end_idx = min((i + 1) * chunks_per_worker, num_chunks)
            if start_idx >= end_idx:
                continue

            # Get the futures for this worker's assigned chunks
            worker_batch_futures = block_futures[start_idx:end_idx]

            # Submit task to process all assigned chunks on this worker
            # Dask will resolve the futures before the function executes
            future = client.submit(
                correlate_and_reduce_on_worker,
                worker_batch_futures,  # List of futures - resolved to numpy arrays
                scattered_config,
                pass_idx,
                scattered_predictor,
                scattered['cache'],
                scattered['masks'],
                workers=[worker],
                pure=False,  # Stateful computation
            )
            worker_futures.append(future)

        # Gather results with progress tracking
        pass_start = time.time()
        worker_results = []
        for i, future in enumerate(as_completed(worker_futures)):
            result = future.result()
            worker_results.append(result)
            logger.info(f"  Worker {i+1}/{len(worker_futures)} complete ({result['n_images']} images)")

        # Final local reduction (fast - only num_workers elements)
        accumulated = worker_results[0]
        for r in worker_results[1:]:
            accumulated = reduce_ensemble_results(accumulated, r)

        # Accumulate and finalize pass
        accumulator.accumulate_batch(accumulated, pass_idx=pass_idx)
        pass_result = accumulator.finalize_pass(
            client=client, pass_idx=pass_idx, predictor_field=predictor_field, output_path=output_path
        )
        # NOTE: finalize_pass() already appends to passes_results internally

        # Extract predictor for next pass
        if pass_idx < num_passes - 1:
            predictor_field = extract_predictor_field(pass_result)

        # Clean up - free accumulated correlation planes to reduce memory usage
        accumulator.clear_pass_data(pass_idx)
        del worker_futures, worker_results, accumulated
        if scattered_predictor is not None:
            del scattered_predictor
        gc.collect()
        # NOTE: gc.collect on workers causes SIGSEGV with FFTW - removed

        pass_elapsed = time.time() - pass_start
        logger.info(f"  Pass {pass_idx + 1} complete in {pass_elapsed:.1f}s")

    # 7. Build and save ensemble result
    logger.info("")
    logger.info("Building ensemble result...")
    ensemble_result = PIVEnsembleResult()
    for pass_result in accumulator.passes_results:
        ensemble_result.add_pass(pass_result)

    logger.info("Saving ensemble result...")
    save_ensemble_result_distributed(
        ensemble_result,
        output_path,
        runs_to_save=config.ensemble_runs_0based,
        filename="ensemble_result.mat",
    )

    # Save coordinates
    logger.info("Saving coordinates...")
    temp_correlator = make_correlator_backend(config, ensemble=True)
    correlator_cache = temp_correlator.get_cache_data()

    save_ensemble_coordinates_from_config_distributed(
        config,
        output_path,
        correlator_cache=correlator_cache,
        runs_to_save=config.ensemble_runs_0based,
    )

    final_path = output_path / "ensemble_result.mat"
    logger.info(f"  Saved to {final_path}")

    return str(final_path)


def main():
    """Main entry point for ensemble PIV processing."""
    start_time = time.time()

    # Load configuration
    #config = Config()

    # Validate configuration
    is_valid, error_msg, warnings = validate_config(config)
    log_validation_result(is_valid, error_msg, warnings, config)

    if not is_valid:
        sys.exit(1)

    # Set up environment
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
    worker_omp_threads = str(config.omp_threads)

    global _client, _cluster

    try:
        # Start Dask cluster
        logger.info("Starting Dask cluster...")
        cluster, client = start_cluster(
            #n_workers_per_node=config.dask_workers_per_node,
            #memory_limit=config.dask_memory_limit,
            config=config,
            #worker_omp_threads=worker_omp_threads,
        )
        _cluster = cluster
        _client = client
        logger.info("Dask cluster started successfully")

        # Log worker info
        info = client.scheduler_info()
        for w, meta in info["workers"].items():
            logger.info(f"Worker {w}: pid={meta.get('pid')}, host={meta.get('host')}")

        # Process each path and camera
        camera_numbers = config.camera_numbers
        active_path_indices = config.active_paths

        logger.info("")
        logger.info("=" * 80)
        logger.info("ENSEMBLE PIV PROCESSING (DASK-NATIVE)")
        logger.info(f"Processing {len(active_path_indices)} path(s), {len(camera_numbers)} camera(s)")
        logger.info("=" * 80)

        for path_set_num, path_idx in enumerate(active_path_indices, start=1):
            if _shutdown_requested:
                logger.info("Shutdown requested, stopping...")
                break

            source_path = config.source_paths[path_idx]
            base_path = config.base_paths[path_idx]

            logger.info("")
            logger.info(f"PATH SET {path_set_num} of {len(active_path_indices)}")
            logger.info(f"  Source: {source_path}")
            logger.info(f"  Base: {base_path}")

            for camera_num in camera_numbers:
                if _shutdown_requested:
                    logger.info("Shutdown requested, stopping...")
                    break

                logger.info("")
                logger.info(f"Processing camera {camera_num}...")

                # Load mask
                mask = load_mask_for_camera(camera_num, config, source_path_idx=path_idx)

                # Compute vector masks
                vector_masks = None
                if config.masking_enabled and mask is not None:
                    logger.info("Computing vector masks...")
                    vector_masks = compute_vector_mask(mask, config)
                    logger.info(f"  Vector masks: {len(vector_masks)} passes")

                # Get output path
                output_path = get_ensemble_output_path(
                    config,
                    camera_num,
                    use_uncalibrated=True,
                    base_path_idx=path_idx,
                )

                # Run PIV
                logger.info("=" * 60)
                logger.info(f"ENSEMBLE PIV: Camera {camera_num}")
                logger.info(f"  Image files: {config.num_images}")
                logger.info(f"  Frame pairs: {config.num_frame_pairs}")
                logger.info(f"  Batch size: {config.batch_size}")
                logger.info(f"  Passes: {config.ensemble_num_passes}")
                logger.info(f"  Window sizes: {config.ensemble_window_sizes}")
                logger.info(f"  Output: {output_path}")
                logger.info("=" * 60)

                result_path = run_ensemble_piv(
                    config=config,
                    client=client,
                    camera_num=camera_num,
                    source_path=source_path,
                    output_path=output_path,
                    base_path=base_path,
                    vector_masks=vector_masks,
                    pixel_mask=mask,
                )

                logger.info("")
                logger.info("=" * 60)
                logger.info(f"ENSEMBLE PIV COMPLETE: {result_path}")
                logger.info("=" * 60)

                # Clean up (local only - gc.collect on workers causes SIGSEGV with FFTW)
                gc.collect()

    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        traceback.print_exc()

    finally:
        # Clean shutdown
        try:
            if _client is not None:
                _client.close()
        except Exception as e:
            logger.warning(f"Error closing client: {e}")

        try:
            if _cluster is not None:
                _cluster.close()
        except Exception as e:
            logger.warning(f"Error closing cluster: {e}")

        end_time = time.time()
        elapsed = end_time - start_time

        if _shutdown_requested:
            print(f"[CANCELLED] Run cancelled after {elapsed:.2f} seconds", flush=True)
        else:
            print(f"Total elapsed time: {elapsed:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
