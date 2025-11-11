import logging
import os
import sys
import tracemalloc
import time
from pathlib import Path

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
from pivtools_cli.preprocessing.preprocess import preprocess_images

def main():
    """Main PIV processing function"""
    start_time = time.time()  # Start timer

    config = Config()
    os.environ["OMP_NUM_THREADS"] = config.omp_threads
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
    if config.debug:
        tracemalloc.start()

    try:
        cluster, client = start_cluster(
            n_workers_per_node=config.dask_workers_per_node,
            threads_per_worker=config.dask_threads_per_worker,
            memory_limit=config.dask_memory_limit,
            config=config,
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
        base_path = config.base_paths[0]

        for camera_num in camera_numbers:
            logging.info("Processing camera: Cam%d", camera_num)

            # Load images from source path (lazy loading - no memory consumption yet)
            images = load_images(camera_num, config, source=source_path)
            
            # Preprocess images (applies filters from config)
            # This intelligently handles batching:
            # - Batch filters (time, pod): rechunks to batch_size
            # - Single-image filters: keeps single-image chunks
            # - No filters: skips preprocessing entirely
            processed_images = preprocess_images(images, config)
            
            # Load mask once per camera (if masking is enabled)
            mask = load_mask_for_camera(camera_num, config, source_path_idx=0)
            
            # Pre-compute vector masks once per camera (if masking is enabled)
            vector_masks = None
            if config.masking_enabled and mask is not None:
                logging.info("Pre-computing vector masks for Cam%d", camera_num)
                vector_masks = compute_vector_mask(mask, config)
                logging.info("Vector masks computed: %d passes", len(vector_masks))
            
            # Get output path for this camera (uncalibrated PIV)
            # Path: base_path/uncalibrated_piv/{num_images}/Cam{camera_num}/instantaneous
            output_path = get_output_path(
                config,
                camera_num,
                use_uncalibrated=True
            )
            
            # Perform PIV and save in parallel on workers with TRUE lazy loading
            # Each worker processes ONE image at a time: load → PIV → save → free
            # Memory per worker: ~280 MB (constant regardless of total images)
            # Dask scheduler handles task distribution automatically
            saved_paths, scattered_cache = perform_piv_and_save(
                processed_images,  # Use preprocessed images
                config,
                client,
                output_path,
                start_frame=1,
                runs_to_save=config.instantaneous_runs_0based,
                vector_masks=vector_masks,  # Pass pre-computed vector masks
            )
            
            # Submit coordinate saving task (runs once per camera) with shared cache
            coords_future = client.submit(
                save_coordinates_from_config_distributed,
                config,
                output_path,
                scattered_cache,  # Use the same scattered cache
                config.instantaneous_runs_0based,
            )
            
            # All PIV processing completed with true lazy loading!
            # Workers processed images one-by-one, keeping memory footprint minimal
            logging.info(
                "PIV and save completed: %d frames saved to %s",
                len(saved_paths), output_path
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
