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
    compute_warp_sums_on_worker,
    correlate_mean_subtracted_on_worker,
    reduce_ensemble_results,
    extract_predictor_field,
    # Single-batch chained accumulation functions (lazy loading)
    correlate_single_batch_and_accumulate,
    warp_single_batch_and_accumulate,
    correlate_mean_subtracted_single_batch,
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

    # 5. Decide memory strategy based on dataset size
    num_chunks = len(images.chunks[0])

    if num_chunks == 1:
        # Small dataset: all images fit in one batch, persist to avoid re-loading per pass
        from dask.distributed import wait, futures_of
        logger.info(f"Small dataset ({num_chunks} chunk): persisting to avoid re-load per pass")
        images = images.persist()
        wait(images)  # Wait since there's only one chunk
        block_futures = futures_of(images)
        use_streaming = False
    else:
        # Large dataset: stream batches to minimize memory (re-loads per pass)
        logger.info(f"Large dataset ({num_chunks} chunks): streaming on-demand to save memory")
        block_futures = None  # Not used in streaming mode
        use_streaming = True
    num_passes = config.ensemble_num_passes
    accumulator = SinglePassAccumulator(config, vector_masks)
    predictor_field = None

    # Import for predictor extraction (used both for resume and after each pass)
    from pivtools_cli.processing.dask_pipeline import extract_predictor_field

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

        # Worker-side accumulation: distribute chunks across workers
        # Each worker processes multiple chunks and returns one accumulated result
        # This reduces network traffic from O(num_chunks) to O(num_workers)
        workers = list(client.scheduler_info()["workers"].keys())
        num_workers = len(workers)
        chunks_per_worker = (num_chunks + num_workers - 1) // num_workers

        logger.info(f"  Distributing {num_chunks} chunks across {num_workers} workers...")
        logger.info(f"  (~{chunks_per_worker} chunks per worker)")

        # Check background subtraction method
        bg_method = config.ensemble_background_subtraction_method
        pass_start = time.time()

        if bg_method == 'image':
            # TWO-PASS method: First compute mean images, then correlate mean-subtracted
            logger.info(f"  [Image BG] Using two-pass background subtraction")

            # ---- PASS 1: Compute warped image sums ----
            # Use chained submission to preserve lazy loading
            logger.info(f"  [Image BG] Sub-pass 1/2: Computing mean images...")
            warp_sum_futures = []
            for i, worker in enumerate(workers):
                start_idx = i * chunks_per_worker
                end_idx = min((i + 1) * chunks_per_worker, num_chunks)
                if start_idx >= end_idx:
                    continue

                # Chain tasks: each depends on previous sum + one new batch
                accumulated_future = None
                for j, chunk_idx in enumerate(range(start_idx, end_idx)):
                    # Get batch: streaming computes on-demand, persisted uses cached future
                    if use_streaming:
                        block_future = client.compute(images.blocks[chunk_idx], workers=[worker])
                    else:
                        block_future = block_futures[chunk_idx]
                    accumulated_future = client.submit(
                        warp_single_batch_and_accumulate,
                        accumulated_future,
                        block_future,
                        scattered_config,
                        pass_idx,
                        scattered_predictor,
                        scattered['cache'],
                        scattered['masks'],
                        workers=[worker],
                        pure=False,
                    )

                # Append only the final result of the chain
                warp_sum_futures.append(accumulated_future)

            # Gather warp sums and compute global mean
            warp_results = []
            for i, future in enumerate(as_completed(warp_sum_futures)):
                result = future.result()
                warp_results.append(result)
                logger.debug(f"    Warp sums from worker {i+1}/{len(warp_sum_futures)}")

            # Reduce warp sums to get global mean
            total_warp_A = warp_results[0]["warp_A_sum"].copy()
            total_warp_B = warp_results[0]["warp_B_sum"].copy()
            total_n_images = warp_results[0]["n_images"]
            smoothed_predictor = warp_results[0].get("smoothed_predictor")

            for r in warp_results[1:]:
                total_warp_A += r["warp_A_sum"]
                total_warp_B += r["warp_B_sum"]
                total_n_images += r["n_images"]
                if r.get("smoothed_predictor") is not None:
                    smoothed_predictor = r["smoothed_predictor"]

            # Compute mean images
            A_mean = total_warp_A / total_n_images
            B_mean = total_warp_B / total_n_images
            logger.info(f"  [Image BG] Mean images computed from {total_n_images} images")
            logger.debug(f"    A_mean range: [{A_mean.min():.2f}, {A_mean.max():.2f}]")

            # Scatter mean images to workers
            mean_images = {"A_mean": A_mean, "B_mean": B_mean}
            scattered_means = client.scatter(mean_images, broadcast=True)

            # Clean up warp sum results
            del warp_results, total_warp_A, total_warp_B, warp_sum_futures
            gc.collect()

            # ---- PASS 2: Correlate mean-subtracted images ----
            # Use chained submission to preserve lazy loading
            logger.info(f"  [Image BG] Sub-pass 2/2: Correlating mean-subtracted images...")
            worker_futures = []
            for i, worker in enumerate(workers):
                start_idx = i * chunks_per_worker
                end_idx = min((i + 1) * chunks_per_worker, num_chunks)
                if start_idx >= end_idx:
                    continue

                # Chain tasks: each depends on previous sum + one new batch
                accumulated_future = None
                for j, chunk_idx in enumerate(range(start_idx, end_idx)):
                    # Get batch: streaming computes on-demand, persisted uses cached future
                    if use_streaming:
                        block_future = client.compute(images.blocks[chunk_idx], workers=[worker])
                    else:
                        block_future = block_futures[chunk_idx]
                    accumulated_future = client.submit(
                        correlate_mean_subtracted_single_batch,
                        accumulated_future,
                        block_future,
                        scattered_config,
                        pass_idx,
                        scattered_predictor,
                        scattered_means,
                        scattered['cache'],
                        scattered['masks'],
                        j == 0,                # is_first_batch
                        workers=[worker],
                        pure=False,
                    )

                # Append only the final result of the chain
                worker_futures.append(accumulated_future)

            # DIAGNOSTIC: Check where accumulated futures live before gathering
            logger.debug(f"  [LOCALITY] Checking future locations before gather...")
            for i, fut in enumerate(worker_futures):
                if hasattr(fut, 'key') and fut.key:
                    who = client.who_has(fut)
                    locations = list(who.get(fut.key, []))
                    logger.debug(f"  [LOCALITY] Final future {i}: on workers {[w[-20:] for w in locations]}")

            # Gather correlation results
            worker_results = []
            for i, future in enumerate(as_completed(worker_futures)):
                result = future.result()
                worker_results.append(result)
                logger.info(f"  Worker {i+1}/{len(worker_futures)} complete ({result['n_images']} images)")

            # Final local reduction
            accumulated = worker_results[0]
            for r in worker_results[1:]:
                accumulated = reduce_ensemble_results(accumulated, r)

            # For 'image' method, update warp sums with actual mean images
            # (workers return zero warp sums for compatibility with reduce_ensemble_results)
            # finalize_pass() needs these for diagnostic logging but skips background correlation
            accumulated["warp_A_sum"] = A_mean * total_n_images
            accumulated["warp_B_sum"] = B_mean * total_n_images

            # Clean up scattered means
            del scattered_means
            gc.collect()

        else:
            # SINGLE-PASS method: correlate raw images, subtract correlated means after
            # Use chained submission to preserve lazy loading - each task depends on
            # (previous sum, ONE batch) so Dask only resolves one batch per task
            worker_futures = []
            for i, worker in enumerate(workers):
                start_idx = i * chunks_per_worker
                end_idx = min((i + 1) * chunks_per_worker, num_chunks)
                if start_idx >= end_idx:
                    continue

                # Chain tasks: each depends on previous sum + one new batch
                accumulated_future = None
                for j, chunk_idx in enumerate(range(start_idx, end_idx)):
                    # Get batch: streaming computes on-demand, persisted uses cached future
                    if use_streaming:
                        block_future = client.compute(images.blocks[chunk_idx], workers=[worker])
                    else:
                        block_future = block_futures[chunk_idx]
                    # Overwrite accumulated_future - releases Client's ref to previous future
                    accumulated_future = client.submit(
                        correlate_single_batch_and_accumulate,
                        accumulated_future,    # Previous sum (Future or None)
                        block_future,          # ONE batch
                        scattered_config,
                        pass_idx,
                        scattered_predictor,
                        scattered['cache'],
                        scattered['masks'],
                        j == 0,                # is_first_batch
                        str(output_path) if config.ensemble_save_diagnostics and j == 0 else None,
                        workers=[worker],      # Keep on same worker!
                        pure=False,
                    )

                # CRITICAL: Append ONLY the final result of the chain
                # This must be OUTSIDE the inner j loop
                worker_futures.append(accumulated_future)

            # DIAGNOSTIC: Check where accumulated futures live before gathering
            logger.debug(f"  [LOCALITY] Checking future locations before gather...")
            for i, fut in enumerate(worker_futures):
                if hasattr(fut, 'key') and fut.key:
                    who = client.who_has(fut)
                    locations = list(who.get(fut.key, []))
                    logger.debug(f"  [LOCALITY] Final future {i}: on workers {[w[-20:] for w in locations]}")

            # Gather results with progress tracking
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
