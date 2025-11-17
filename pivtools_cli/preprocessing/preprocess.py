import logging

import dask
import dask.array as da
from dask import config as dask_config
import numpy as np

from pivtools_core.config import Config

from pivtools_cli.preprocessing.filters import filter_images, requires_batch


def get_batch_size_for_filters(config: Config) -> int:
    """
    Determine the optimal batch size based on enabled filters.
    
    Some filters (time, pod) require multiple images to compute properly.
    Others can work on single images.
    
    Args:
        config (Config): Configuration object with filters defined
        
    Returns:
        int: Recommended batch size (1 for single-image filters, >1 for batch filters)
    """
    if not config.filters:
        return 1  # No preprocessing, no batching needed
    
    for filter_spec in config.filters:
        filter_type = filter_spec.get("type")
        if requires_batch(filter_type):
            # Time and POD filters need batches
            # Use batch size from config
            batch_size = config.batch_size
            logging.info(
                f"Filter '{filter_type}' requires batching. Using batch_size={batch_size}"
            )
            return batch_size
    
    # No batch-requiring filters, can process images one-by-one
    return 1


def apply_filters_to_single_batch(
    batch_images: da.Array,
    batch_filter_specs: list,
    config: Config,
    batch_num: int,
    total_batches: int,
) -> np.ndarray:
    """
    Apply batch filters to a single batch in main process with multi-threading.

    This function:
    1. Loads the batch into main process memory
    2. Applies batch filters using multi-threading (utilizes all CPU cores)
    3. Returns processed batch as numpy array

    Args:
        batch_images (da.Array): Lazy Dask array slice for one batch (B, 2, H, W)
        batch_filter_specs (list): List of batch filter configurations
        config (Config): Configuration object
        batch_num (int): Current batch number (for logging)
        total_batches (int): Total number of batches (for logging)

    Returns:
        np.ndarray: Filtered batch as numpy array (B, 2, H, W)
    """
    import os
    import multiprocessing as mp

    # Get number of threads that will be used
    num_threads = os.cpu_count() or 1

    logging.info("")
    logging.info(f">>> BATCH {batch_num}/{total_batches} <<<")

    # Load batch in main process using threading scheduler
    logging.info(f"[Batch {batch_num}] Loading into main process memory...")
    with dask_config.set(scheduler='threads', num_workers=num_threads):
        batch_computed = batch_images.compute()

    mem_mb = batch_computed.nbytes / (1024 ** 2)
    num_images = batch_computed.shape[0]
    logging.info(f"[Batch {batch_num}] Loaded {num_images} images ({mem_mb:.1f} MB)")

    # Convert to Dask array for filter application
    batch_da = da.from_array(batch_computed, chunks=batch_computed.shape)

    # Apply batch filters with multi-threading
    filter_names = ', '.join(f.get('type') for f in batch_filter_specs)
    logging.info(f"[Batch {batch_num}] Applying filters: {filter_names}")
    logging.info(f"[Batch {batch_num}] Using {num_threads} threads for filtering")

    original_filters = config.data['filters']
    config.data['filters'] = batch_filter_specs

    with dask_config.set(scheduler='threads', num_workers=num_threads):
        batch_filtered = filter_images(batch_da, config)
        batch_filtered_computed = batch_filtered.compute()

    config.data['filters'] = original_filters

    logging.info(f"[Batch {batch_num}] Filtering complete")

    return batch_filtered_computed


def has_batch_filters(config: Config) -> bool:
    """Check if config contains batch filters (time, POD)."""
    if not config.filters:
        return False
    return any(requires_batch(f.get("type")) for f in config.filters)


def get_batch_filter_specs(config: Config) -> list:
    """Get list of batch filter specifications from config."""
    return [f for f in config.filters if requires_batch(f.get("type"))]


def get_spatial_filter_specs(config: Config) -> list:
    """Get list of spatial filter specifications from config."""
    return [f for f in config.filters if not requires_batch(f.get("type"))]


def preprocess_images(images: da.Array, config: Config) -> da.Array:
    """
    Apply spatial filters to images (lazy evaluation).

    NOTE: This function should NOT be used when batch filters are present.
    For batch filters, use the batch-by-batch processing in example.py.

    Args:
        images (da.Array): Dask array containing the images (N, 2, H, W)
        config (Config): Configuration object with filters defined

    Returns:
        da.Array: Filtered Dask array of images (lazy)
    """
    if not config.filters:
        logging.info("No filters configured, skipping preprocessing")
        return images

    # Only apply spatial filters (lazy)
    spatial_filter_specs = get_spatial_filter_specs(config)

    if spatial_filter_specs:
        logging.info(
            f"Applying {len(spatial_filter_specs)} spatial filter(s) "
            "(lazy evaluation on workers)"
        )
        original_filters = config.data['filters']
        config.data['filters'] = spatial_filter_specs
        images = filter_images(images, config)
        config.data['filters'] = original_filters

    return images

