import logging
import os
import sys
import time
from pathlib import Path

from dask.distributed import Client, wait, performance_report
from dask import config as dask_config

# Add src to path for unified imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import Config
from image_handling.load_images import load_images, load_mask_for_camera
from pypivtools.piv.piv import perform_piv_and_save
from pypivtools.piv.save_results import save_coordinates_from_config_distributed, get_output_path
from pypivtools.piv_cluster.cluster import start_cluster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def run_camera_processing(client, config, camera_num, source_path, base_path):
    logging.info(f"Processing camera: Cam{camera_num}")

    # Load images
    images = load_images(camera_num, config, source=source_path)

    # Load mask and pre-compute vector masks
    mask = load_mask_for_camera(camera_num, config, source_path_idx=0)
    vector_masks = None
    if config.masking_enabled and mask is not None:
        from image_handling.load_images import compute_vector_mask
        vector_masks = compute_vector_mask(mask, config)
        logging.info(f"Pre-computed vector masks: {len(vector_masks)} passes")

    # Output path for uncalibrated PIV
    output_path = get_output_path(config, camera_num, use_uncalibrated=True)

    # Perform PIV and save in parallel
    save_futures, scattered_cache = perform_piv_and_save(
        images, config, client, output_path,
        start_frame=1,
        pass_index=None,
        vector_masks=vector_masks
    )

    # Save coordinates using the same scattered cache
    coords_future = client.submit(save_coordinates_from_config_distributed, config, output_path, scattered_cache)

    # Wait for tasks
    wait(save_futures)
    coords_future.result()

    logging.info(f"Completed processing for Cam{camera_num}. Saved {len(save_futures)} frames to {output_path}")


def main():
    start_time = time.time()
    config = Config()

    try:
        cluster, client = start_cluster(
            n_workers_per_node=config.dask_workers_per_node,
            threads_per_worker=config.dask_threads_per_worker,
            memory_limit=config.dask_memory_limit,
            config=config
        )
        logging.info("Dask cluster started successfully")
    except Exception as e:
        logging.error("Error starting Dask cluster: %s", e)
        sys.exit(1)

    # Dask profiling report (can be opened in browser)
    profile_report_path = Path(config.base_paths[0]) / "dask_profile_report.html"

    try:
        with performance_report(filename=str(profile_report_path)):
            # Optional: print worker info
            # Optional: print worker info safely
            for w, meta in client.scheduler_info()["workers"].items():
                logging.info(
                    f"Worker {w}: "
                    f"pid={meta.get('pid', 'N/A')}, "
                    f"host={meta.get('host', 'N/A')}, "
                    f"memory={meta.get('memory_limit', 'N/A')}, "
                    f"local_dir={meta.get('local_directory', 'N/A')}, "
                    f"nanny={meta.get('nanny', 'N/A')}"
    )


            source_path = config.source_paths[0]
            base_path = config.base_paths[0]

            for camera_num in config.camera_numbers:
                run_camera_processing(client, config, camera_num, source_path, base_path)

        logging.info(f"Dask profiling report saved to {profile_report_path}")

    except Exception as e:
        import traceback
        logging.error("Error during processing: %s", e)
        traceback.print_exc()
    finally:
        client.close()
        elapsed = time.time() - start_time
        logging.info(f"Total elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
