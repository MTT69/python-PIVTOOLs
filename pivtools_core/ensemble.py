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
import numpy as np
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
    reduce_warp_sums,
    extract_predictor_field,
    correlate_worker_batches,
)


logger = logging.getLogger(__name__)


def process_pass_worker_accumulate(
    client,
    num_chunks,
    workers,
    scattered_config,
    pass_idx,
    scattered_predictor,
    scattered,
    config,
    output_path,
    camera_num,
    source_path,
    pixel_mask,
    images=None,
):
    """
    Process one pass using per-worker accumulation.

    Each worker gets ONE long-running task that:
    1. Creates ONE EnsembleCorrelatorCPU
    2. Loops over its assigned batches, accumulating into internal buffers
    3. Returns the accumulated result ONCE

    Two modes:
    - images=None (non-persisted): Assigns chunks round-robin. Workers
      reconstruct the lazy image+filter pipeline locally and load from disk.
    - images provided (persisted dask array): Discovers where Dask placed
      each chunk via who_has(), groups by home worker, passes chunk futures
      as batch_images (Dask resolves them locally — zero transfer).

    Then tree-reduces K per-worker results into one final result.
    """
    from dask.distributed import as_completed, Variable
    import threading

    num_workers = len(workers)
    diag_path = str(output_path) if config.ensemble_save_diagnostics else None

    if images is not None:
        # Persisted mode: discover where Dask placed each chunk, group by home worker
        from dask.distributed import futures_of
        all_futures = futures_of(images)
        # futures_of returns one future per chunk (in graph order)
        # Map each to its home worker via who_has
        who_map = client.who_has(all_futures)
        worker_chunks = {}  # worker_address → [(chunk_idx, future), ...]
        for chunk_idx, fut in enumerate(all_futures):
            holders = who_map.get(fut.key, [])
            home_worker = holders[0] if holders else workers[chunk_idx % num_workers]
            if home_worker not in worker_chunks:
                worker_chunks[home_worker] = []
            worker_chunks[home_worker].append((chunk_idx, fut))
    else:
        # Non-persisted mode: assign chunks round-robin across workers
        worker_chunks_rr = {w: [] for w in workers}
        for chunk_idx in range(num_chunks):
            w = workers[chunk_idx % num_workers]
            worker_chunks_rr[w].append(chunk_idx)
        # Convert to same format, remove empty workers
        worker_chunks = {
            w: [(idx, None) for idx in indices]
            for w, indices in worker_chunks_rr.items()
            if indices
        }

    logger.info(
        f"  Worker accumulation: {len(worker_chunks)} workers, "
        f"{num_chunks} chunks ({[len(c) for c in worker_chunks.values()]} per worker)"
        f"{' (persisted)' if images is not None else ' (from disk)'}"
    )

    # 'image' background method: phase A — a full sweep over the pairs that
    # only accumulates warped image sums, reduced to global mean images which
    # phase B then subtracts before correlating. Costs a second pass through
    # the data; only runs when the method is explicitly selected.
    scattered_mean_images = None
    image_bg_sums = None
    if config.ensemble_background_subtraction_method == 'image':
        logger.info(
            "  'image' background method: phase A sweep (warped image sums "
            "for the ensemble-mean images) — this doubles the passes over the data"
        )
        phase_a_futures = []
        for worker, chunk_info in worker_chunks.items():
            chunk_indices = [idx for idx, _ in chunk_info]
            chunk_futs = [fut for _, fut in chunk_info]
            batch_images = chunk_futs if all(f is not None for f in chunk_futs) else None
            fut = client.submit(
                correlate_worker_batches,
                batch_indices=chunk_indices,
                config=scattered_config,
                pass_idx=pass_idx,
                predictor_field=scattered_predictor,
                cache=scattered['cache'],
                masks=scattered['masks'],
                camera_num=camera_num,
                source_path=str(source_path),
                pixel_mask=pixel_mask,
                batch_images=batch_images,
                warp_sums_only=True,
                workers=[worker],
                pure=False,
            )
            phase_a_futures.append(fut)
        sums_futures = list(phase_a_futures)
        while len(sums_futures) > 1:
            merged = []
            for i in range(0, len(sums_futures), 2):
                if i + 1 < len(sums_futures):
                    merged.append(client.submit(
                        reduce_warp_sums, sums_futures[i], sums_futures[i + 1],
                        pure=False,
                    ))
                else:
                    merged.append(sums_futures[i])
            sums_futures = merged
        image_bg_sums = sums_futures[0].result()
        n_phase_a = image_bg_sums["n_images"]
        A_mean = (image_bg_sums["warp_A_sum"] / n_phase_a).astype(np.float32)
        B_mean = (image_bg_sums["warp_B_sum"] / n_phase_a).astype(np.float32)
        scattered_mean_images = client.scatter((A_mean, B_mean), broadcast=True)
        logger.info(f"  Phase A complete: mean images from {n_phase_a} pairs")

    # Create per-worker progress variables
    progress_vars = []
    worker_list = list(worker_chunks.keys())
    for i in range(len(worker_list)):
        var = Variable(f"_corr_p{pass_idx}_{i}", client)
        var.set(0)
        progress_vars.append(var)

    # Submit one task per worker
    worker_futures = []
    for wi, (worker, chunk_info) in enumerate(worker_chunks.items()):
        chunk_indices = [idx for idx, _ in chunk_info]
        chunk_futs = [fut for _, fut in chunk_info]
        batch_images = chunk_futs if all(f is not None for f in chunk_futs) else None

        fut = client.submit(
            correlate_worker_batches,
            batch_indices=chunk_indices,
            config=scattered_config,
            pass_idx=pass_idx,
            predictor_field=scattered_predictor,
            cache=scattered['cache'],
            masks=scattered['masks'],
            camera_num=camera_num,
            source_path=str(source_path),
            pixel_mask=pixel_mask,
            output_path=diag_path if 0 in chunk_indices else None,
            batch_images=batch_images,
            progress_var_name=f"_corr_p{pass_idx}_{wi}",
            mean_images=scattered_mean_images,
            workers=[worker],
            pure=False,
        )
        worker_futures.append(fut)

    # Start progress polling thread
    stop_progress = threading.Event()
    last_pct = [0]

    def poll_progress():
        while not stop_progress.is_set():
            try:
                total_done = sum(v.get(timeout=2) for v in progress_vars)
                pct = min(100, int(100 * total_done / num_chunks))
                if pct > last_pct[0]:
                    logger.info(f"  Pass {pass_idx + 1} correlation: {pct}%")
                    last_pct[0] = pct
            except Exception:
                pass
            stop_progress.wait(3)

    progress_thread = threading.Thread(target=poll_progress, daemon=True)
    progress_thread.start()

    # Capture pre-pass transfer bytes for locality verification
    pre_transfer = {}
    try:
        for addr, info in client.scheduler_info()["workers"].items():
            metrics = info.get("metrics", {})
            pre_transfer[addr] = {
                "in": metrics.get("transfer_incoming_bytes", 0),
                "out": metrics.get("transfer_outgoing_bytes", 0),
            }
    except Exception:
        pass  # Non-critical diagnostics

    # Wait for all workers (check errors without pulling data to client)
    completed_count = 0
    pinned_workers = list(worker_chunks.keys())
    ac = as_completed(worker_futures)
    for completed in ac:
        completed_count += 1
        exc = completed.exception()
        if exc is not None:
            for f in worker_futures:
                f.cancel()
            raise exc
        logger.debug(f"  Worker {completed_count}/{len(worker_futures)} complete")

    # Stop progress polling and clean up variables
    stop_progress.set()
    progress_thread.join(timeout=5)
    for v in progress_vars:
        try:
            v.delete()
        except Exception:
            pass

    # Verify locality: check each result is on the worker we pinned it to
    for fut, pinned_worker in zip(worker_futures, pinned_workers):
        try:
            holders = client.who_has(fut)
            if holders:
                held_on = list(holders.values())[0]
                is_local = pinned_worker in held_on
                logger.debug(
                    f"  Locality: pinned={pinned_worker.split('/')[-1]} "
                    f"held_on={[h.split('/')[-1] for h in held_on]} "
                    f"{'OK' if is_local else 'MOVED'}"
                )
        except Exception:
            pass

    # Tree reduction of per-worker results
    futures = list(worker_futures)
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
        logger.debug(f"  Tree round {round_idx}: {len(futures)} -> {len(new_futures)}")
        futures = new_futures

    final = futures[0].result()
    logger.info(f"  Accumulated: {final['n_images']} images")

    # 'image' method: phase-B results carry zero warp sums (means were consumed
    # in phase A) — inject the phase-A global sums so finalize_pass's mean
    # images stay meaningful for logging/diagnostics.
    if image_bg_sums is not None:
        final["warp_A_sum"] = image_bg_sums["warp_A_sum"]
        final["warp_B_sum"] = image_bg_sums["warp_B_sum"]

    # Report transfer bytes during this pass
    try:
        post_transfer = {}
        for addr, info in client.scheduler_info()["workers"].items():
            metrics = info.get("metrics", {})
            post_transfer[addr] = {
                "in": metrics.get("transfer_incoming_bytes", 0),
                "out": metrics.get("transfer_outgoing_bytes", 0),
            }
        total_in = sum(
            post_transfer[a]["in"] - pre_transfer.get(a, {}).get("in", 0)
            for a in post_transfer
        )
        total_out = sum(
            post_transfer[a]["out"] - pre_transfer.get(a, {}).get("out", 0)
            for a in post_transfer
        )
        logger.debug(
            f"  Transfer during pass: in={total_in / 1e6:.1f} MB, "
            f"out={total_out / 1e6:.1f} MB"
        )
    except Exception:
        pass

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
        # Desktop / memory-constrained: workers re-load from disk per pass
        logger.info(f"Large dataset ({num_chunks} chunks): workers load from disk per pass")
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

        # Backup pre-resume result before any intermediate saves overwrite it
        backup_path = output_path / f"ensemble_result_before_pass{resume_from_pass}.mat.bak"
        shutil.copy2(existing_result_path, backup_path)
        logger.info(f"  Backed up previous result to {backup_path}")
        logger.info(f"  Predictor extracted from pass {start_pass_idx} (shape: {predictor_field.shape})")

    # Scatter config once to avoid repeated serialization
    scattered_config = client.scatter(config, broadcast=True)

    # Scatter pixel_mask for worker-accumulation mode
    scattered_pixel_mask = None
    if pixel_mask is not None:
        scattered_pixel_mask = client.scatter(pixel_mask, broadcast=True)

    images_are_persisted = (num_chunks == 1) or config.ensemble_persist_images

    # Pre-compute correlator metadata for intermediate saves and coordinates
    temp_correlator = make_correlator_backend(config, ensemble=True)
    correlator_cache = temp_correlator.get_cache_data()
    image_height = temp_correlator.H
    del temp_correlator
    gradient_correction = config.ensemble_gradient_correction

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

        workers = list(client.ncores().keys())
        num_workers = len(workers)

        pass_start = time.time()

        # Single path for both persisted and non-persisted
        accumulated = process_pass_worker_accumulate(
            client=client,
            num_chunks=num_chunks,
            workers=workers,
            scattered_config=scattered_config,
            pass_idx=pass_idx,
            scattered_predictor=scattered_predictor,
            scattered=scattered,
            config=config,
            output_path=output_path,
            camera_num=camera_num,
            source_path=source_path,
            pixel_mask=scattered_pixel_mask,
            images=images if images_are_persisted else None,
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

        # Save intermediate result after each pass (enables debugging mid-run)
        intermediate_result = PIVEnsembleResult()
        for pr in accumulator.passes_results:
            intermediate_result.add_pass(pr)
        save_ensemble_result_distributed(
            intermediate_result,
            output_path,
            runs_to_save=config.ensemble_runs_0based,
            filename="ensemble_result.mat",
            gradient_correction=gradient_correction,
            image_height=image_height,
        )
        logger.info(f"  Saved intermediate result (pass {pass_idx + 1}/{num_passes})")
        del intermediate_result

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

    if gradient_correction:
        logger.info("Gradient correction enabled for Reynolds stresses")

    logger.info("Saving final ensemble result...")
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
                logger.debug(f"mask loaded = {mask is not None}, masking_enabled = {config.masking_enabled}")
                if mask is not None:
                    logger.debug(f"pixel mask shape = {mask.shape}")

                # Compute vector masks
                vector_masks = None
                if config.masking_enabled and mask is not None:
                    logger.info("Computing vector masks...")
                    logger.debug(f"config.image_shape = {config.image_shape}")
                    vector_masks = compute_vector_mask(mask, config, ensemble=True)
                    logger.info(f"  Vector masks: {len(vector_masks)} passes")
                    for i, vm in enumerate(vector_masks):
                        logger.info(f"    Pass {i}: mask shape = {vm.shape}")
                else:
                    logger.debug(f"Skipping vector mask computation (enabled={config.masking_enabled}, mask={mask is not None})")

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
