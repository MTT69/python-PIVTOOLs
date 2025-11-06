import logging
import os
import sys
import tracemalloc
import time
from pathlib import Path

from dask.distributed import wait, performance_report

# Add src to path for unified imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import Config
from image_handling.load_images import load_images, load_mask_for_camera
from pypivtools.piv.piv import perform_piv_and_save
from pypivtools.piv.save_results import save_coordinates_from_config_distributed, get_output_path
from pypivtools.piv_cluster.cluster import start_cluster

def run_camera_processing(client, config, camera_num, source_path, base_path):
    logging.info("Processing camera: Cam%d", camera_num)

    # Load images from source path
    images = load_images(camera_num, config, source=source_path)
    # processed_images = preprocess_images(images, config)
    
    # Load mask once per camera (if masking is enabled)
    mask = load_mask_for_camera(camera_num, config, source_path_idx=0)
    
    # Pre-compute vector masks once per camera (if masking is enabled)
    vector_masks = None
    if config.masking_enabled and mask is not None:
        from image_handling.load_images import compute_vector_mask
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
    
    # Perform PIV and save in parallel on workers
    # This avoids gathering all results to main process
    save_futures, scattered_cache = perform_piv_and_save(
        images,
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
    
    # Wait for all PIV+save tasks to complete
    wait(save_futures)
    logging.info(
        "PIV and save completed: %d frames saved to %s",
        len(save_futures), output_path
    )
    
    # Wait for coordinates to be saved
    coords_future.result()
    logging.info("Coordinates saved to %s", output_path)


def main():
    start_time = time.time()  # Start timer

    config = Config()
    os.environ["OMP_NUM_THREADS"] = config.omp_threads
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

    # Dask profiling report (can be opened in browser)
    profile_report_path = Path(config.base_paths[0]) / "dask_profile_report.html"

    try:
        with performance_report(filename=str(profile_report_path)):
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
                run_camera_processing(client, config, camera_num, source_path, base_path)

        logging.info("Dask profiling report saved to %s", profile_report_path)

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
