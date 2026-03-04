import logging
from pathlib import Path
from typing import Optional

import numpy as np

from pivtools_core.config import Config

from pivtools_cli.preprocessing.filters import requires_batch
from pivtools_cli.processing.dask_pipeline import _apply_spatial_filters_numpy


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

    # Separate filters into spatial and temporal
    spatial_specs = [f for f in filters if f.get("type") not in ("pod", "time")]
    temporal_specs = [f for f in filters if f.get("type") in ("pod", "time")]

    # Apply spatial filters using unified numpy function
    if spatial_specs:
        batch = _apply_spatial_filters_numpy(batch, spatial_specs)
        if save_diagnostics and batch_idx == 0:
            filter_stages["02_spatial_filters"] = batch.copy()

    # Apply temporal filters
    for filter_spec in temporal_specs:
        filter_type = filter_spec.get("type")
        if filter_type == "pod":
            from pivtools_cli.preprocessing.pod_filter import pod_filter_batch
            batch = pod_filter_batch(batch, eps_auto_psi=filter_spec.get('eps_auto_psi', 0.01), eps_auto_sigma=filter_spec.get('eps_auto_sigma', 0.01))
        elif filter_type == "time":
            from pivtools_cli.preprocessing.pod_filter import time_filter_batch
            batch = time_filter_batch(batch)

        if save_diagnostics and batch_idx == 0:
            filter_stages[f"03_{filter_type}"] = batch.copy()

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


def has_batch_filters(config: Config) -> bool:
    """Check if config contains batch filters (time, POD)."""
    if not config.filters:
        return False
    return any(requires_batch(f.get("type")) for f in config.filters)
