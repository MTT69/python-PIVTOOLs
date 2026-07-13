import logging
from pathlib import Path
from typing import Optional

import numpy as np

from pivtools_cli.preprocessing.filters import requires_batch
from pivtools_cli.processing.dask_pipeline import (
    apply_all_filters_slim,
    get_filter_specs,
)
from pivtools_core.config import Config


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

    Thin wrapper around apply_all_filters_slim that extracts filter specs
    from config and handles diagnostics saving.

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
    filter_stages = {}
    if save_diagnostics and batch_idx == 0:
        filter_stages["00_original"] = batch.copy()

    filter_specs = get_filter_specs(config)

    batch = apply_all_filters_slim(
        batch, filter_specs=filter_specs, pixel_mask=pixel_mask
    )

    if save_diagnostics and batch_idx == 0 and output_dir is not None:
        filter_stages["final"] = batch.copy()
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
