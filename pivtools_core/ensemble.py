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
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

from dask.distributed import Client

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
    create_filter_pipeline,
    scatter_immutable_data,
    reduce_ensemble_results,
    reduce_ensemble_results_inplace,
    extract_predictor_field,
    correlate_batch_ensemble,
)


logger = logging.getLogger(__name__)


def process_pass_sliding_window(
    client,
    images,
    num_chunks,
    workers,
    scattered_config,
    pass_idx,
    scattered_predictor,
    scattered,
    config,
    output_path,
):
    """
    Process one pass using sliding window with dynamic worker-affinity reduction.

    Architecture (mirrors instantaneous pipeline's as_completed pattern):
      - Phase 1: Fill initial window — submit filter→correlation tasks with NO
        worker pinning. Dask's scheduler decides placement (optimal load balancing
        and data locality).
      - Phase 2: as_completed loop — when each correlation completes,
        client.who_has() discovers which worker holds the result. Reduction is
        pinned there via workers=[worker], achieving zero cross-worker transfer
        during accumulation. A replacement task is submitted to keep the window full.
      - Phase 3: Tree reduction — merges K per-worker accumulators in O(log₂ K)
        rounds using safe + operator (Dask retry compatible).

    Memory per worker: max_in_flight_per_worker batches + 1 accumulator
    I/O: Parallel (max_in_flight concurrent disk reads)
    Network: Zero cross-worker transfer during Phase 2; only K results merge in Phase 3
    """
    from dask.distributed import as_completed

    if num_chunks == 0:
        raise ValueError("No chunks to process")

    num_workers = len(workers)
    max_in_flight_per_worker = config.dask_max_in_flight_per_worker
    max_in_flight = min(num_workers * max_in_flight_per_worker, num_chunks)

    diag_path = str(output_path) if config.ensemble_save_diagnostics else None

    # Per-worker accumulated futures (dynamically populated via who_has)
    accumulated = {}   # worker_address → accumulated_future

    pending = {}       # corr_future → chunk_idx
    next_to_submit = 0
    completed_count = 0
    last_reported_pct = -1

    logger.info(
        f"  Sliding window: {num_workers} workers x {max_in_flight_per_worker} per worker "
        f"= {max_in_flight} max_in_flight ({num_chunks} chunks total)"
    )

    # --- PHASE 1: Fill initial window (NO pinning — Dask decides placement) ---
    while next_to_submit < min(max_in_flight, num_chunks):
        filter_future = client.compute(images.blocks[next_to_submit])
        corr_future = client.submit(
            correlate_batch_ensemble,
            filter_future,
            scattered_config, pass_idx, scattered_predictor,
            scattered['cache'], scattered['masks'],
            batch_idx=next_to_submit,
            output_path=diag_path if next_to_submit == 0 else None,
            pure=False,
        )
        pending[corr_future] = next_to_submit
        next_to_submit += 1

    logger.info(f"  Initial window filled: {len(pending)}/{max_in_flight} tasks in flight")

    # --- PHASE 2: as_completed + dynamic worker-affinity reduction ---
    ac = as_completed(list(pending.keys()))
    for completed in ac:
        pending.pop(completed)
        completed_count += 1

        # Discover which worker holds the completed result
        who = client.who_has(completed)
        worker = list(who[completed.key])[0]

        # Reduce into that worker's accumulator (pinned — zero transfer)
        if worker not in accumulated:
            accumulated[worker] = completed
        else:
            accumulated[worker] = client.submit(
                reduce_ensemble_results_inplace, accumulated[worker], completed,
                workers=[worker],
                pure=False,
            )

        # Submit replacement to keep window full
        if next_to_submit < num_chunks:
            filter_future = client.compute(images.blocks[next_to_submit])
            corr_future = client.submit(
                correlate_batch_ensemble,
                filter_future,
                scattered_config, pass_idx, scattered_predictor,
                scattered['cache'], scattered['masks'],
                batch_idx=next_to_submit,
                output_path=None,
                pure=False,
            )
            pending[corr_future] = next_to_submit
            ac.add(corr_future)
            next_to_submit += 1

        # Progress logging
        current_pct = round((completed_count / num_chunks) * 100)
        if current_pct > last_reported_pct:
            logger.info(
                f"  Correlation progress: {current_pct}% ({len(pending)} in flight)"
            )
            last_reported_pct = current_pct

    # --- PHASE 3: Tree reduction of per-worker accumulators ---
    futures = list(accumulated.values())
    logger.info(f"  Tree reduction: {len(futures)} worker results")

    round_idx = 0
    while len(futures) > 1:
        new_futures = []
        for i in range(0, len(futures), 2):
            if i + 1 < len(futures):
                merged = client.submit(
                    reduce_ensemble_results, futures[i], futures[i + 1],
                    pure=False,
                )
                new_futures.append(merged)
            else:
                new_futures.append(futures[i])
        round_idx += 1
        logger.debug(f"  Tree round {round_idx}: {len(futures)} → {len(new_futures)}")
        futures = new_futures

    final = futures[0].result()
    logger.info(f"  Accumulated: {final['n_images']} images")
    return final


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
            # Cancel all pending futures to stop work immediately
            try:
                _client.cancel(_client.futures, force=True)
            except Exception:
                pass  # May fail if no futures or already cancelled

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
    # 1. Load images (lazy, batch-sized chunks — no rechunk needed)
    batch_size = config.batch_size
    logger.info(f"Loading images for camera {camera_num} (batch_size={batch_size})...")
    images = load_images(camera_num, config, source=source_path, batch_size=batch_size)
    logger.info(f"  Loaded: shape={images.shape}, {len(images.chunks[0])} chunks of size {images.chunks[0][0]}")

    # 2. Scatter immutable data once
    logger.info("Scattering immutable data...")
    scattered = scatter_immutable_data(
        client, config, vector_masks, pixel_mask, ensemble=True
    )

    # 3. Apply all filters via map_blocks
    logger.info("Creating filter pipeline...")
    save_intermediate = base_path if config.filters else None
    images = create_filter_pipeline(images, config, pixel_mask, save_intermediate_base=save_intermediate)

    # 4. Decide memory strategy based on dataset size
    num_chunks = len(images.chunks[0])

    if num_chunks == 1:
        # Small dataset: all images fit in one batch, persist to avoid re-loading per pass
        from dask.distributed import wait
        logger.info(f"Small dataset ({num_chunks} chunk): persisting to avoid re-load per pass")
        images = images.persist()
        wait(images)
    elif config.ensemble_persist_images:
        # HPC with lots of RAM: persist all filtered images in worker memory
        # With batch-loading (no rechunk), tasks are fully independent —
        # workers distribute batch-loads evenly from the start
        from dask.distributed import wait
        logger.info(f"Persisting {num_chunks} filtered image chunks to worker RAM (ensemble_persist_images=true)...")
        images = images.persist()
        wait(images)
        logger.info("  All filtered images cached in worker RAM")
    else:
        # Desktop / memory-constrained: sliding window re-computes filters per pass
        logger.info(f"Large dataset ({num_chunks} chunks): using sliding window for parallel I/O")
    num_passes = config.ensemble_num_passes
    accumulator = SinglePassAccumulator(config, vector_masks)
    predictor_field = None

    # Check for resume from previous pass
    resume_from_pass = config.ensemble_resume_from_pass  # 1-based, 0 = no resume
    start_pass_idx = 0  # 0-based

    if resume_from_pass > 0:
        # Convert to 0-based: resume_from_pass=6 means start at pass_idx=5
        start_pass_idx = resume_from_pass - 1

        # Validation
        if start_pass_idx < 1:
            raise ValueError(
                f"resume_from_pass={resume_from_pass} invalid: must resume from pass 2 or higher"
            )
        if start_pass_idx >= num_passes:
            raise ValueError(
                f"resume_from_pass={resume_from_pass} exceeds num_passes={num_passes}"
            )

        # Load existing ensemble_result.mat
        existing_result_path = output_path / "ensemble_result.mat"
        if not existing_result_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: {existing_result_path} not found. "
                f"Ensure previous passes completed successfully."
            )

        logger.info(f"Resuming from pass {resume_from_pass} (loading passes 1-{resume_from_pass-1})...")

        from pivtools_cli.piv.save_results import load_ensemble_result
        loaded_result, n_loaded = load_ensemble_result(
            existing_result_path,
            passes_to_load=list(range(start_pass_idx))  # Load passes 0..start_pass_idx-1
        )

        # Validate loaded passes
        if n_loaded < start_pass_idx:
            raise ValueError(
                f"Loaded only {n_loaded} passes but need {start_pass_idx} for resume_from_pass={resume_from_pass}"
            )

        # Load previous passes into accumulator
        # Note: n_images is tracked per-batch, but for resume we use total images from config
        # The final save combines all passes with their original statistics
        accumulator.load_previous_passes(loaded_result, config.num_images)

        # Extract predictor from last loaded pass
        last_pass = loaded_result.passes[-1]
        predictor_field = extract_predictor_field(last_pass)

        logger.info(f"  Loaded {n_loaded} passes from {existing_result_path}")
        logger.info(f"  Predictor extracted from pass {start_pass_idx} (shape: {predictor_field.shape})")

    # Scatter config once to avoid repeated serialization
    scattered_config = client.scatter(config, broadcast=True)

    logger.info(f"Processing passes {start_pass_idx + 1} to {num_passes} with {num_chunks} chunks each...")

    for pass_idx in range(start_pass_idx, num_passes):
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

        # Sliding window accumulation: parallel I/O with bounded memory
        # Chunks are distributed across workers in round-robin fashion
        workers = list(client.ncores().keys())
        num_workers = len(workers)

        logger.info(f"  Distributing {num_chunks} chunks across {num_workers} workers...")

        pass_start = time.time()

        # Process pass using sliding window for parallel I/O
        accumulated = process_pass_sliding_window(
            client=client,
            images=images,
            num_chunks=num_chunks,
            workers=workers,
            scattered_config=scattered_config,
            pass_idx=pass_idx,
            scattered_predictor=scattered_predictor,
            scattered=scattered,
            config=config,
            output_path=output_path,
        )

        corr_elapsed = time.time() - pass_start

        # Accumulate and finalize pass
        finalize_start = time.time()
        accumulator.accumulate_batch(accumulated, pass_idx=pass_idx)
        pass_result = accumulator.finalize_pass(
            client=client, pass_idx=pass_idx, predictor_field=predictor_field, output_path=output_path
        )
        # NOTE: finalize_pass() already appends to passes_results internally
        finalize_elapsed = time.time() - finalize_start

        # Extract predictor for next pass
        if pass_idx < num_passes - 1:
            predictor_field = extract_predictor_field(pass_result)

        # Clean up - free accumulated correlation planes to reduce memory usage
        accumulator.clear_pass_data(pass_idx)
        del accumulated
        if scattered_predictor is not None:
            del scattered_predictor
        gc.collect()
        # NOTE: gc.collect on workers causes SIGSEGV with FFTW - removed

        pass_elapsed = time.time() - pass_start
        logger.info(
            f"  Pass {pass_idx + 1} complete in {pass_elapsed:.1f}s "
            f"(correlation: {corr_elapsed:.1f}s, finalize: {finalize_elapsed:.1f}s)"
        )

    # 7. Build and save ensemble result
    logger.info("")
    logger.info("Building ensemble result...")
    ensemble_result = PIVEnsembleResult()
    for pass_result in accumulator.passes_results:
        ensemble_result.add_pass(pass_result)

    # Backup existing result if resuming (safety measure)
    if resume_from_pass > 0:
        existing_path = output_path / "ensemble_result.mat"
        if existing_path.exists():
            backup_path = output_path / f"ensemble_result_before_pass{resume_from_pass}.mat.bak"
            shutil.copy2(existing_path, backup_path)
            logger.info(f"Backed up previous result to {backup_path}")

    # Get correlator cache for coordinates and image height
    temp_correlator = make_correlator_backend(config, ensemble=True)
    correlator_cache = temp_correlator.get_cache_data()
    image_height = temp_correlator.H

    # Check if gradient correction is enabled (only for uncalibrated data)
    gradient_correction = config.ensemble_gradient_correction
    if gradient_correction:
        logger.info("Gradient correction enabled for Reynolds stresses")

    logger.info("Saving ensemble result...")
    save_ensemble_result_distributed(
        ensemble_result,
        output_path,
        runs_to_save=config.ensemble_runs_0based,
        filename="ensemble_result.mat",
        gradient_correction=gradient_correction,
        image_height=image_height,
    )

    # Save coordinates
    logger.info("Saving coordinates...")

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
        print(f"Dask dashboard available at: {client.dashboard_link}", flush=True)
        if config.open_dashboard:
            import webbrowser
            webbrowser.open(client.dashboard_link)

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
        logger.info("ENSEMBLE PIV PROCESSING (DASK-NATIVE)")
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
                logger.info(f"DEBUG: mask loaded = {mask is not None}, masking_enabled = {config.masking_enabled}")
                if mask is not None:
                    logger.info(f"DEBUG: pixel mask shape = {mask.shape}")

                # Compute vector masks
                vector_masks = None
                if config.masking_enabled and mask is not None:
                    logger.info("Computing vector masks...")
                    logger.info(f"DEBUG: config.image_shape = {config.image_shape}")
                    vector_masks = compute_vector_mask(mask, config, ensemble=True)
                    logger.info(f"  Vector masks: {len(vector_masks)} passes")
                    for i, vm in enumerate(vector_masks):
                        logger.info(f"    Pass {i}: mask shape = {vm.shape}")
                else:
                    logger.info(f"DEBUG: Skipping vector mask computation (enabled={config.masking_enabled}, mask={mask is not None})")

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
