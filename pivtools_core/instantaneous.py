"""
Dask-Native Instantaneous PIV Processing

Entry point for instantaneous PIV with true Dask patterns:
- rechunk: Group images for batched processing
- map_blocks: Apply filters lazily per-chunk
- sliding window: Parallel I/O with bounded memory
- submit: One task per chunk for correlation

Usage:
    python -m pivtools_core.instantaneous
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

from pivtools_core.config import Config
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
    save_coordinates_from_config_distributed,
    get_output_path,
)
from pivtools_cli.piv.piv_backend.factory import make_correlator_backend
from pivtools_cli.processing.dask_pipeline import (
    rechunk_for_batched_processing,
    create_filter_pipeline,
    scatter_immutable_data,
    correlate_and_save_batch,
)


logger = logging.getLogger(__name__)


def process_instantaneous_sliding_window(
    client,
    images,
    num_chunks,
    scattered_config,
    scattered,
    output_path,
    config,
):
    """
    Process instantaneous PIV using sliding window.

    Unlike ensemble, no accumulation chain - tasks are independent.
    Can process in any order using as_completed().

    Memory: ~2 batches per worker (bounded)
    I/O: Parallel (max_in_flight concurrent disk reads)
    """
    from dask.distributed import as_completed

    num_workers = config.dask_workers_per_node
    max_in_flight = min(num_workers * 2, num_chunks)

    pending = {}  # correlation Future -> chunk_idx
    next_to_submit = 0
    all_saved_paths = []
    completed_count = 0
    last_reported_pct = -1  # Track last reported percentage to avoid duplicates

    logger.debug(f"  Sliding window: max_in_flight={max_in_flight}")

    # Fill initial window (NO worker pinning → parallel I/O!)
    while next_to_submit < min(max_in_flight, num_chunks):
        # Calculate frame number for this chunk
        chunk_start = sum(images.chunks[0][:next_to_submit])

        # Submit filter (parallel I/O - no worker pinning)
        filter_future = client.compute(images.blocks[next_to_submit])

        corr_future = client.submit(
            correlate_and_save_batch,
            filter_future,
            chunk_start + 1,  # 1-indexed frame number
            scattered_config,
            scattered['cache'],
            scattered['masks'],
            output_path,
            config.instantaneous_runs_0based,
            config.vector_format,
            pure=False,
        )
        pending[corr_future] = next_to_submit
        next_to_submit += 1

    logger.debug(f"  Initial window: {len(pending)} tasks in flight")

    # Process as tasks complete (any order is fine for instantaneous)
    ac = as_completed(list(pending.keys()))
    for completed in ac:
        chunk_idx = pending[completed]
        saved_paths = completed.result()
        all_saved_paths.extend(saved_paths)

        del pending[completed]
        completed_count += 1

        # Submit replacement to keep window full
        if next_to_submit < num_chunks:
            chunk_start = sum(images.chunks[0][:next_to_submit])
            filter_future = client.compute(images.blocks[next_to_submit])
            corr_future = client.submit(
                correlate_and_save_batch,
                filter_future,
                chunk_start + 1,
                scattered_config,
                scattered['cache'],
                scattered['masks'],
                output_path,
                config.instantaneous_runs_0based,
                config.vector_format,
                pure=False,
            )
            pending[corr_future] = next_to_submit
            ac.add(corr_future)
            next_to_submit += 1

        # Progress logging - report each new percentage point
        current_pct = round((completed_count / num_chunks) * 100)
        if current_pct > last_reported_pct:
            logger.info(f"  Correlation progress: {current_pct}%")
            last_reported_pct = current_pct

    return all_saved_paths


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

    # Suppress noisy distributed logs during teardown
    import logging as _logging
    for name in ["distributed.worker", "distributed.scheduler", "distributed.nanny",
                 "distributed.core", "distributed.comm", "distributed.comm.tcp",
                 "distributed.batched", "tornado.application", "tornado.general"]:
        _logging.getLogger(name).setLevel(_logging.CRITICAL)

    # Close Dask client and cluster if they exist
    try:
        if _client is not None:
            logger.info("Cancelling pending futures...")
            try:
                _client.cancel(_client.futures, force=True)
            except Exception:
                pass

            logger.info("Closing Dask client...")
            _client.close(timeout=5)
    except Exception as e:
        logger.warning(f"Error closing client: {e}")

    # Small delay to let workers finish cleanly
    import time as _time
    _time.sleep(0.5)

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


def run_instantaneous_piv(
    config: Config,
    client: Client,
    camera_num: int,
    source_path: Path,
    output_path: Path,
    base_path: Path,
    vector_masks: Optional[List] = None,
    pixel_mask: Optional = None,
) -> List[str]:
    """
    Run Dask-native instantaneous PIV processing for one camera.

    Flow:
    1. Load images (lazy)
    2. Rechunk for batched processing
    3. Apply all filters via map_blocks
    4. Process using sliding window (parallel I/O with bounded memory)
    5. Save coordinates

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
        List of saved file paths
    """
    # 1. Load images (lazy)
    logger.info(f"Loading images for camera {camera_num}...")
    images = load_images(camera_num, config, source=source_path)
    logger.info(f"  Loaded: shape={images.shape}, {len(images.chunks[0])} chunks of size {images.chunks[0][0]}")

    # 2. Scatter immutable data once
    logger.info("Scattering immutable data...")
    scattered = scatter_immutable_data(
        client, config, vector_masks, pixel_mask, ensemble=False
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

    # 5. Process using sliding window for parallel I/O with bounded memory
    num_chunks = len(images.chunks[0])
    logger.info(f"Processing {num_chunks} chunks using sliding window...")

    # Scatter config once to avoid repeated serialization
    scattered_config = client.scatter(config, broadcast=True)

    # Use sliding window: parallel I/O without loading all data into RAM
    all_saved_paths = process_instantaneous_sliding_window(
        client=client,
        images=images,
        num_chunks=num_chunks,
        scattered_config=scattered_config,
        scattered=scattered,
        output_path=output_path,
        config=config,
    )

    # Save coordinates
    logger.info("Saving coordinates...")
    temp_correlator = make_correlator_backend(config)
    correlator_cache = temp_correlator.get_cache_data()
    scattered_cache_for_coords = client.scatter(correlator_cache, broadcast=True)

    coords_future = client.submit(
        save_coordinates_from_config_distributed,
        config,
        output_path,
        scattered_cache_for_coords,
        config.instantaneous_runs_0based,
    )
    coords_future.result()
    logger.info(f"  Coordinates saved to {output_path}")

    return all_saved_paths


def main():
    """Main entry point for instantaneous PIV processing."""
    start_time = time.time()


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
            n_workers_per_node=config.dask_workers_per_node,
            memory_limit=config.dask_memory_limit,
            config=config,
            worker_omp_threads=worker_omp_threads,
        )
        _cluster = cluster
        _client = client
        logger.info("Dask cluster started successfully")

        # Log worker info
        info = client.scheduler_info()
        for w, meta in info["workers"].items():
            logger.info(f"Worker {w}: pid={meta.get('pid')}, host={meta.get('host')}")

        # Generate run timestamp for config traceability
        from datetime import datetime
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Process each path and camera
        camera_numbers = config.camera_numbers
        active_path_indices = config.active_paths

        logger.info("")
        logger.info("=" * 80)
        logger.info("INSTANTANEOUS PIV PROCESSING (DASK-NATIVE)")
        logger.info(f"Processing {len(active_path_indices)} path(s), {len(camera_numbers)} camera(s)")
        logger.info("=" * 80)

        for path_set_num, path_idx in enumerate(active_path_indices, start=1):
            if _shutdown_requested:
                logger.info("Shutdown requested, stopping...")
                break

            source_path = config.source_paths[path_idx]
            base_path = config.base_paths[path_idx]

            # Save timestamped config copy for traceability
            config_copy_path = config.save_timestamped_copy(base_path, timestamp=run_timestamp)
            logger.info(f"Config saved to: {config_copy_path}")

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
                    vector_masks = compute_vector_mask(mask, config, ensemble=False)
                    logger.info(f"  Vector masks: {len(vector_masks)} passes")

                # Get output path
                output_path = get_output_path(
                    config,
                    camera_num,
                    use_uncalibrated=True,
                    base_path_idx=path_idx,
                    piv_type="instantaneous",
                )

                # Run PIV
                logger.info("=" * 60)
                logger.info(f"INSTANTANEOUS PIV: Camera {camera_num}")
                logger.info(f"  Image files: {config.num_images}")
                logger.info(f"  Frame pairs: {config.num_frame_pairs}")
                logger.info(f"  Batch size: {config.batch_size}")
                logger.info(f"  Output: {output_path}")
                logger.info("=" * 60)

                saved_paths = run_instantaneous_piv(
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
                logger.info(f"INSTANTANEOUS PIV COMPLETE: {len(saved_paths)} results saved")
                logger.info("=" * 60)

                # Clean up (local only - gc.collect on workers causes SIGSEGV with FFTW)
                gc.collect()

    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        traceback.print_exc()

    finally:
        # Clean shutdown - suppress noisy distributed logs during teardown
        import logging as _logging
        import sys as _sys
        import io as _io

        for name in ["distributed.worker", "distributed.scheduler", "distributed.nanny",
                     "distributed.core", "distributed.comm", "distributed.comm.tcp",
                     "distributed.batched", "tornado.application", "tornado.general"]:
            _logging.getLogger(name).setLevel(_logging.CRITICAL)

        # Suppress stderr during cluster shutdown to hide Tornado tracebacks
        _old_stderr = _sys.stderr
        _sys.stderr = _io.StringIO()

        try:
            if _client is not None:
                _client.close(timeout=5)
        except Exception as e:
            pass  # Suppress errors during shutdown

        # Small delay to let workers finish cleanly
        time.sleep(0.5)

        try:
            if _cluster is not None:
                _cluster.close(timeout=5)
        except Exception as e:
            pass  # Suppress errors during shutdown

        # Wait a bit more for async cleanup to complete
        time.sleep(0.2)

        # Restore stderr
        _sys.stderr = _old_stderr

        end_time = time.time()
        elapsed = end_time - start_time

        if _shutdown_requested:
            print(f"[CANCELLED] Run cancelled after {elapsed:.2f} seconds", flush=True)
        else:
            print(f"Total elapsed time: {elapsed:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
