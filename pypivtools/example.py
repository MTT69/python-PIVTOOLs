import logging
import os
import tracemalloc
import time

from dask.distributed import wait

from pypivtools.config import Config
from pypivtools.image_handling.load_images import load_images
from pypivtools.piv.piv import perform_piv_and_save
from pypivtools.piv.save_results import (
    save_coordinates_from_config_distributed,
    get_output_path,
)
from pypivtools.piv_cluster.cluster import start_cluster

if __name__ == "__main__":

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

    try:

        info = client.scheduler_info()
        for w, meta in info["workers"].items():
            logging.info("Dask Worker Info:")
            logging.info("Worker %s", w)
            logging.info("  pid: %s", meta.get("pid"))
            logging.info("  host: %s", meta.get("host"))
            logging.info("  local_dir: %s", meta.get("local_directory"))
            logging.info("  nanny: %s", meta.get("nanny"))

        camera_folders = config.cameras

        for camera in camera_folders:
            logging.info("Processing camera: %s", camera)

            images = load_images(camera, config)
            # processed_images = preprocess_images(images, config)
            
            # Get output path for this camera (uncalibrated PIV)
            # Path: base_path/uncalibrated_piv/{num_images}/{camera}/{type}
            output_path = get_output_path(
                config,
                camera,
                use_uncalibrated=True
            )
            
            # Perform PIV and save in parallel on workers
            # This avoids gathering all results to main process
            save_futures = perform_piv_and_save(
                images,
                config,
                client,
                output_path,
                start_frame=1,
                pass_index=None,  # Save all passes
            )
            
            # Submit coordinate saving task (runs once per camera)
            coords_future = client.submit(
                save_coordinates_from_config_distributed,
                config,
                output_path,
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

        if config.debug:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage: {current / 10**6:.2f} MB")
            print(f"Peak memory usage: {peak / 10**6:.2f} MB")

            tracemalloc.stop()
    except Exception as e:
        print(f"Error {e}", flush=True)
    finally:
        client.close()
        end_time = time.time()  # End timer
        elapsed = end_time - start_time
        print(f"Total elapsed time: {elapsed:.2f} seconds", flush=True)
