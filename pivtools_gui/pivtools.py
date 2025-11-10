import tracemalloc

import dask
import dask.config
from dask.distributed import Client, LocalCluster

from .config import Config
from .image_handling.load_images import load_images
from .image_handling.readers import list_supported_formats
from .post_processing.POD.pod_decompose import pod_decompose, pod_rebuild
from pivtools_cli.preprocessing.preprocess import preprocess_images
from .vector_statistics.ensemble_statistics import ensemble_statistics
from .vector_statistics.instantaneous_statistics import instantaneous_statistics

tracemalloc.start()


if __name__ == "__main__":
    # Print supported image formats
    print(f"Supported image formats: {', '.join(list_supported_formats())}")

    # Start a local Dask cluster and client for distributed computation and dashboard
    cluster = LocalCluster()
    client = Client(cluster)
    print(f"Dask dashboard available at: {client.dashboard_link}")

    # Print cluster resources
    print(f"Number of workers: {len(client.scheduler_info()['workers'])}")
    print("Threads per worker:")
    for addr, worker in client.scheduler_info()["workers"].items():
        print(f"  {addr}: {worker['nthreads']} threads")

    dask.config.set(scheduler="threads")

    config = Config()
    base_paths = config.base_paths
    sources = config.source_paths
    camera_numbers = config.camera_numbers

    for idx, base_path in enumerate(base_paths):
        source = sources[idx]
        for cam_num in camera_numbers:
            print(f"Processing source: {source}, camera: {cam_num}")
            images = load_images(cam_num, config, source=source)
            processed_images = preprocess_images(images, config)
            processed_images.compute()
            # perform PIV
            instantaneous_statistics(cam_num, config, base=base_path)
            ensemble_statistics(cam_num, config, base=base_path)
            pod_decompose(cam_num, config, base=base_path, k_modes=10)
            # Rebuild calibrated fields at prescribed energy (if configured)
            pod_rebuild(cam_num, config, base=base_path)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6:.2f} MB")
    print(f"Peak memory usage: {peak / 10**6:.2f} MB")

    tracemalloc.stop()
