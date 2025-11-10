import sys
from pathlib import Path
import logging

import dask.array as da

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


def preprocess_images(images: da.Array, config: Config) -> da.Array:
    """
    Preprocess images based on the provided configuration.
    
    This function intelligently handles batching:
    - For batch filters (time, pod): rechunks to batch size along first axis
    - For single-image filters: preserves single-image chunks for efficiency
    
    Args:
        images (da.Array): Dask array containing the images (N, 2, H, W)
        config (Config): Configuration object with filters defined
    
    Returns:
        da.Array: Filtered Dask array of images
    """
    if not config.filters:
        logging.info("No filters configured, skipping preprocessing")
        return images
    
    # Determine if batching is needed
    batch_size = get_batch_size_for_filters(config)
    
    if batch_size > 1:
        # Rechunk for batch processing along first dimension
        logging.info(f"Rechunking images for batch processing (batch_size={batch_size})")
        images = images.rechunk((batch_size, 2, -1, -1))
    
    # Apply filters
    images = filter_images(images, config)
    
    if batch_size > 1:
        # Rechunk back to single images for PIV processing
        logging.info("Rechunking back to single images for PIV processing")
        images = images.rechunk((1, 2, -1, -1))
    
    return images

