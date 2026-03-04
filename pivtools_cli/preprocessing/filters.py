import logging

import numpy as np


def apply_pixel_mask_to_batch(batch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply pixel mask to a numpy batch of images.

    Sets pixel intensity to zero in masked regions. Used by the batch pipeline
    for both instantaneous and ensemble PIV processing.

    Args:
        batch (np.ndarray): Image batch of shape (N, 2, H, W).
        mask (np.ndarray): Boolean mask of shape (H, W) where True indicates
            regions to mask (set to zero). If None, returns batch unchanged.

    Returns:
        np.ndarray: Batch with masked regions set to zero intensity.
    """
    if mask is None:
        return batch

    # Ensure mask is boolean
    mask = np.asarray(mask, dtype=bool)

    # Validate shapes
    if batch.ndim != 4:
        logging.error(f"Pixel mask: Expected 4D batch (N, 2, H, W), got {batch.ndim}D")
        return batch

    _, _, H, W = batch.shape
    if mask.shape != (H, W):
        logging.error(f"Pixel mask shape {mask.shape} doesn't match image shape ({H}, {W})")
        return batch

    # Set masked pixels to 0
    result = batch.copy()
    result[:, :, mask] = 0

    masked_pixels = np.sum(mask)
    logging.debug(f"Applied pixel mask: {masked_pixels} pixels zeroed per frame")

    return result


# Filters that require batches of images to operate correctly
BATCH_FILTERS = {"time", "pod"}


def requires_batch(filter_type: str) -> bool:
    """
    Check if a filter requires batches of images to operate.

    Args:
        filter_type (str): Type of filter (e.g., 'time', 'pod', 'gaussian')

    Returns:
        bool: True if filter needs multiple images, False otherwise
    """
    return filter_type in BATCH_FILTERS
