import gc
import logging
import os
import sys
import tracemalloc
import time
from pathlib import Path

import yaml
import dask
import dask.array as da
from dask.delayed import delayed

# Add src to path for unified imports
from pivtools_core.config import Config
from pivtools_core.image_handling.load_images import load_images, load_mask_for_camera
from pivtools_core.image_handling.load_images import compute_vector_mask

from pivtools_cli.piv.piv import perform_piv_and_save
from pivtools_cli.piv.save_results import (
    save_coordinates_from_config_distributed,
    get_output_path,
)
from pivtools_cli.piv_cluster.cluster import start_cluster
from pivtools_cli.preprocessing.filters import filter_images
from pivtools_cli.preprocessing.preprocess import (
    preprocess_images,
    has_batch_filters,
    get_batch_filter_specs,
    get_spatial_filter_specs,
    apply_filters_to_single_batch,
)

def main():
    """Main PIV processing function"""
    start_time = time.time()  # Start timer

    config = Config()
    logging.info("Config YAML:\n" + yaml.dump(config.data))
    
    # Store original OMP_NUM_THREADS for workers
    original_omp_threads = os.environ.get("OMP_NUM_THREADS", "1")
    
    # For batch filtering in main process, use all CPU cores
    max_threads = str(os.cpu_count() or 1)
    os.environ["OMP_NUM_THREADS"] = max_threads
    logging.info(f"Set OMP_NUM_THREADS to {max_threads} for main process batch filtering")
    
    # For workers, use config value (set later when starting cluster)
    worker_omp_threads = str(config.omp_threads)
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
    if config.debug:
        tracemalloc.start()

    try:
        cluster, client = start_cluster(
            n_workers_per_node=config.dask_workers_per_node,
            threads_per_worker=config.dask_threads_per_worker,
            memory_limit=config.dask_memory_limit,
            config=config,
            worker_omp_threads=worker_omp_threads,
        )
        logging.info("Dask cluster started successfully")

    except Exception as e:
        logging.error("Error starting Dask cluster: %s", e)
        exit(1)

    try:

        info = client.scheduler_info()
        for w, meta in info["workers"].items():
            logging.info("Dask Worker Info:")
            logging.info("Worker %s", w)
            logging.info("  pid: %s", meta.get("pid"))
            logging.info("  host: %s", meta.get("host"))
            logging.info("  local_dir: %s", meta.get("local_directory"))
            logging.info("  nanny: %s", meta.get("nanny"))

        camera_numbers = config.camera_numbers
        source_path = config.source_paths[0]

        for camera_num in camera_numbers:
            logging.info("Processing camera: Cam%d", camera_num)

            # Load images from source path (lazy loading - no memory consumption yet)
            images = load_images(camera_num, config, source=source_path)

            # Load mask once per camera (if masking is enabled)
            mask = load_mask_for_camera(camera_num, config, source_path_idx=0)

            # Pre-compute vector masks once per camera (if masking is enabled)
            vector_masks = None
            if config.masking_enabled and mask is not None:
                logging.info("Pre-computing vector masks for Cam%d", camera_num)
                vector_masks = compute_vector_mask(mask, config)
                logging.info("Vector masks computed: %d passes", len(vector_masks))

            # Get output path for this camera (uncalibrated PIV)
            output_path = get_output_path(
                config,
                camera_num,
                use_uncalibrated=True
            )

            # Check if we have batch filters (POD, time)
            if has_batch_filters(config):
                # BATCH FILTER PIPELINE: Process batch → filter → PIV → save → repeat
                # This keeps memory low by never accumulating all images
                logging.info("=" * 80)
                logging.info("BATCH FILTER MODE DETECTED")
                logging.info("=" * 80)

                batch_filter_specs = get_batch_filter_specs(config)
                batch_size = config.batch_size
                num_images = config.num_images
                total_batches = (num_images + batch_size - 1) // batch_size

                logging.info(f"Total images: {num_images}")
                logging.info(f"Batch size: {batch_size}")
                logging.info(f"Total batches: {total_batches}")
                logging.info(f"Batch filters: {[f.get('type') for f in batch_filter_specs]}")
                logging.info(f"Workers configured: {config.dask_workers_per_node}")
                logging.info("=" * 80)

                # Pre-compute and broadcast correlator cache ONCE (not per batch)
                from pivtools_cli.piv.piv_backend.factory import make_correlator_backend
                temp_correlator = make_correlator_backend(config)
                correlator_cache = temp_correlator.get_cache_data()
                scattered_cache = client.scatter(correlator_cache, broadcast=True)
                logging.info("Broadcast correlator cache to all workers (ONCE)")

                # Broadcast vector masks ONCE (not per batch)
                scattered_masks = None
                if vector_masks is not None:
                    scattered_masks = client.scatter(vector_masks, broadcast=True)
                    total_mask_size = sum(m.nbytes for m in vector_masks) / 1024
                    logging.info(f"Broadcast vector masks to all workers (ONCE, {total_mask_size:.1f} KB)")

                all_saved_paths = []

                # Process each batch: filter in main → distribute to workers for PIV
                for batch_idx in range(total_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, num_images)
                    batch_num = batch_idx + 1

                    # Extract batch slice (still lazy)
                    batch_images = images[batch_start:batch_end]

                    # Apply batch filters in main process with multi-threading
                    batch_filtered = apply_filters_to_single_batch(
                        batch_images,
                        batch_filter_specs,
                        config,
                        batch_num,
                        total_batches,
                    )

                    # Split batch into individual pairs for low-memory distribution
                    logging.info(f"[Batch {batch_num}] Splitting batch into {batch_end - batch_start} individual pairs for low-memory distribution")
                    pairs = [batch_filtered[i] for i in range(batch_filtered.shape[0])]  # List of (2, H, W) np.ndarray
                    del batch_filtered  # Free main memory ASAP
                    gc.collect()

                    # Scatter individual pairs (Dask balances across workers)
                    scattered_pairs = client.scatter(pairs)
                    pair_mb = pairs[0].nbytes / (1024 ** 2)
                    logging.info(f"[Batch {batch_num}] {len(pairs)} pairs scattered individually (~{pair_mb:.1f} MB each)")

                    # Create Dask arrays from scattered pairs (each becomes a delayed task)
                    dask_images = [
                        da.from_delayed(
                            scattered_pair,
                            shape=(2, *config.image_shape),
                            dtype=pairs[0].dtype
                        )
                        for scattered_pair in scattered_pairs
                    ]

                    # Stack into batch DA (still lazy, chunks=1 per image)
                    batch_filtered_da = da.stack(dask_images, axis=0)

                    # Apply spatial filters lazily on workers if any
                    spatial_filter_specs = get_spatial_filter_specs(config)
                    if spatial_filter_specs:
                        logging.info(f"[Batch {batch_num}] Applying {len(spatial_filter_specs)} spatial filter(s) lazily on workers")
                        original_filters = config.data['filters']
                        config.data['filters'] = spatial_filter_specs
                        batch_filtered_da = filter_images(batch_filtered_da, config)
                        config.data['filters'] = original_filters

                    # Distribute batch to workers for PIV and save
                    # Use pre-scattered cache and masks (already broadcasted once)
                    logging.info(f"[Batch {batch_num}] Starting parallel PIV processing on {config.dask_workers_per_node} workers...")
                    saved_paths, _ = perform_piv_and_save(
                        batch_filtered_da,
                        config,
                        client,
                        output_path,
                        start_frame=batch_start + 1,
                        runs_to_save=config.instantaneous_runs_0based,
                        vector_masks=None,  # Already scattered
                        scattered_cache=scattered_cache,
                        scattered_masks=scattered_masks,
                    )

                    all_saved_paths.extend(saved_paths)
                    logging.info(f"[Batch {batch_num}] {len(saved_paths)} images processed and saved")

                    # Free memory before next batch
                    del pairs, scattered_pairs, dask_images, batch_filtered_da
                    gc.collect()

                logging.info("")
                logging.info("=" * 80)
                logging.info(f"All {total_batches} batches complete!")
                logging.info(f"Total images processed: {len(all_saved_paths)}")
                logging.info("=" * 80)

            else:
                # SPATIAL FILTER PIPELINE: Standard lazy processing
                # Preprocess images (spatial filters only, stays lazy)
                processed_images = preprocess_images(images, config)

                # Perform PIV and save in parallel on workers with lazy loading
                all_saved_paths, scattered_cache = perform_piv_and_save(
                    processed_images,
                    config,
                    client,
                    output_path,
                    start_frame=1,
                    runs_to_save=config.instantaneous_runs_0based,
                    vector_masks=vector_masks,
                )

                logging.info(
                    "PIV and save completed: %d frames saved to %s",
                    len(all_saved_paths), output_path
                )

            # Submit coordinate saving task (runs once per camera)
            coords_future = client.submit(
                save_coordinates_from_config_distributed,
                config,
                output_path,
                scattered_cache,
                config.instantaneous_runs_0based,
            )

            # Wait for coordinates to be saved
            coords_future.result()
            logging.info("Coordinates saved to %s", output_path)

        if config.debug:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage: {current / 10**6:.2f} MB")
            print(f"Peak memory usage: {peak / 10**6:.2f} MB")

            tracemalloc.stop()
    except Exception as e:
        import traceback
        print(f"Error: {e}", flush=True)
        print("Traceback:", flush=True)
        traceback.print_exc()
    finally:
        client.close()
        end_time = time.time()  # End timer
        elapsed = end_time - start_time
        print(f"Total elapsed time: {elapsed:.2f} seconds", flush=True)

if __name__ == "__main__":
    main()
