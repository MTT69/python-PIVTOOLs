import logging
from pathlib import Path
from typing import Optional

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

    # Validate filtered output
    if batch_filtered_computed.shape != batch_computed.shape:
        logging.error(
            f"[Batch {batch_num}] Shape mismatch! "
            f"Input: {batch_computed.shape}, Output: {batch_filtered_computed.shape}"
        )
        raise ValueError("Filtering changed image shape unexpectedly")

    if batch_filtered_computed.size == 0:
        logging.error(f"[Batch {batch_num}] Filtering produced empty array!")
        raise ValueError("Filtering produced empty array")

    logging.info(
        f"[Batch {batch_num}] Filtering complete. "
        f"Output shape: {batch_filtered_computed.shape}, "
        f"dtype: {batch_filtered_computed.dtype}, "
        f"range: [{batch_filtered_computed.min():.2f}, {batch_filtered_computed.max():.2f}]"
    )

    return batch_filtered_computed


def apply_filters_to_batch(
    batch: np.ndarray,
    config: Config,
    save_diagnostics: bool = False,
    output_dir: Optional[Path] = None,
    batch_idx: int = 0,
    pixel_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply ALL filters (temporal and spatial) to a batch, including pixel masking.

    Unified function that handles both batch and spatial filters.
    Used by UnifiedBatchPipeline for consistent filter application.

    The pixel mask is applied FIRST before any other filters to ensure masked
    regions have zero intensity throughout the entire preprocessing pipeline.

    Args:
        batch: Numpy array of shape (N, 2, H, W)
        config: Configuration object
        save_diagnostics: If True, save diagnostic images for first batch
        output_dir: Output directory for diagnostic images
        batch_idx: Batch index (diagnostics only saved for batch 0)
        pixel_mask: Optional boolean mask of shape (H, W) where True indicates
            regions to mask (set to zero intensity). Applied before other filters.

    Returns:
        Filtered batch of same shape
    """
    from pathlib import Path

    filters = config.filters

    # Track filter stages for diagnostics
    filter_stages = {}
    if save_diagnostics and batch_idx == 0:
        filter_stages["00_original"] = batch.copy()

    # Apply pixel mask FIRST (before any other filters)
    # This ensures masked regions are zeroed throughout preprocessing
    if pixel_mask is not None:
        from pivtools_cli.preprocessing.filters import apply_pixel_mask_to_batch
        batch = apply_pixel_mask_to_batch(batch, pixel_mask)
        if save_diagnostics and batch_idx == 0:
            filter_stages["01_pixel_mask"] = batch.copy()

    if not filters:
        return batch

    # Offset filter indices if pixel mask was applied
    filter_offset = 2 if pixel_mask is not None else 1

    for filter_idx, filter_spec in enumerate(filters):
        filter_type = filter_spec.get("type")

        if filter_type == "pod":
            # Call POD filter block function directly (already have numpy array)
            from pivtools_cli.preprocessing.filters import _pod_filter_block
            batch = _pod_filter_block(batch)
        elif filter_type == "time":
            # Call time filter block function directly (already have numpy array)
            from pivtools_cli.preprocessing.filters import _subtract_local_min
            batch = _subtract_local_min(batch)
        else:
            # Spatial filters (gaussian, median, etc.)
            batch = apply_spatial_filter_to_batch(batch, filter_spec, config)

        # Save after each filter for diagnostics
        if save_diagnostics and batch_idx == 0:
            stage_name = f"{filter_idx + filter_offset:02d}_{filter_type}"
            filter_stages[stage_name] = batch.copy()

    # Save diagnostic images if enabled
    if save_diagnostics and batch_idx == 0 and output_dir is not None:
        from pivtools_cli.preprocessing.diagnostics import save_filter_diagnostics
        save_filter_diagnostics(
            original_batch=filter_stages.get("00_original"),
            filtered_stages=filter_stages,
            output_dir=Path(output_dir),
            batch_idx=batch_idx,
            pair_idx=0,
        )

    return batch


def apply_spatial_filter_to_batch(
    batch: np.ndarray,
    filter_spec: dict,
    config: Config,
) -> np.ndarray:
    """
    Apply spatial filter to each frame in batch.

    Args:
        batch: Shape (N, 2, H, W)
        filter_spec: Filter specification dict
        config: Configuration

    Returns:
        Filtered batch of same shape
    """
    from pivtools_cli.preprocessing.filters import (
        gaussian_filter_dask,
        median_filter_dask,
        norm_filter,
        sbg_filter,
    )

    filter_type = filter_spec.get("type")
    N = batch.shape[0]

    # Apply to each frame
    for n in range(N):
        for frame_idx in range(2):  # A and B frames
            frame_da = da.from_array(batch[n, frame_idx], chunks=batch[n, frame_idx].shape)
            if filter_type == "gaussian":
                sigma = filter_spec.get("sigma", 1.0)
                batch[n, frame_idx] = gaussian_filter_dask(frame_da, sigma=sigma).compute()
            elif filter_type == "median":
                size = filter_spec.get("size", (5, 5))
                if isinstance(size, list):
                    size = tuple(size)
                batch[n, frame_idx] = median_filter_dask(frame_da, size=size).compute()
            elif filter_type == "norm":
                size = filter_spec.get("size", (7, 7))
                max_gain = filter_spec.get("max_gain", 1.0)
                if isinstance(size, list):
                    size = tuple(size)
                batch[n, frame_idx] = norm_filter(frame_da, size=size, max_gain=max_gain).compute()
            elif filter_type == "sbg":
                bg = filter_spec.get("bg", None)
                batch[n, frame_idx] = sbg_filter(frame_da, bg=bg).compute()
            # Add other spatial filters as needed

    return batch


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
    For batch filters, use the processing in instantaneous.py or ensemble.py.

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

