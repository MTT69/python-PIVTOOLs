"""
Diagnostic image saving utilities for ensemble PIV.

This module provides functions to save diagnostic images during ensemble PIV processing,
including:
- Original images before filtering
- Images after each filter step
- Warped images for each pass

All images are saved as 8-bit TIFFs for easy visualization.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np


def _normalize_to_8bit(image: np.ndarray, clip_negatives: bool = True) -> np.ndarray:
    """
    Normalize image to 8-bit range for saving.

    Parameters
    ----------
    image : np.ndarray
        Input image (any dtype)
    clip_negatives : bool
        If True, clip negative values to 0 before normalization.
        This prevents cubic interpolation ringing artifacts from
        shifting the entire image range.

    Returns
    -------
    np.ndarray
        Image normalized to uint8 [0, 255]
    """
    img = image.astype(np.float32)

    # Log if there are negative values (diagnostic)
    if img.min() < 0:
        logging.debug(
            f"Image has negative values: min={img.min():.2f}, max={img.max():.2f}. "
            f"This may be from cubic interpolation ringing."
        )

    # Clip negative values to prevent them from shifting the normalization
    if clip_negatives:
        img = np.clip(img, 0, None)

    img_min = img.min()
    img_max = img.max()

    if img_max - img_min > 1e-10:
        img = (img - img_min) / (img_max - img_min) * 255
    else:
        img = np.zeros_like(img)

    return img.astype(np.uint8)


def save_diagnostic_image(
    image: np.ndarray,
    output_dir: Path,
    filename: str,
) -> None:
    """
    Save a single image as 8-bit TIFF.

    Parameters
    ----------
    image : np.ndarray
        Image to save (H, W)
    output_dir : Path
        Directory to save to
    filename : str
        Filename (with .tif extension)
    """
    import tifffile

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    img_8bit = _normalize_to_8bit(image)
    tifffile.imwrite(filepath, img_8bit)

    logging.debug(f"Saved diagnostic image: {filepath}")


def save_filter_diagnostics(
    original_batch: np.ndarray,
    filtered_stages: dict,
    output_dir: Path,
    batch_idx: int = 0,
    pair_idx: int = 0,
) -> None:
    """
    Save diagnostic images showing filter effects.

    Saves the first image pair from the first batch:
    - Original A and B images
    - After each filter step

    Parameters
    ----------
    original_batch : np.ndarray
        Original batch before filtering, shape (N, 2, H, W)
    filtered_stages : dict
        Dictionary mapping filter name to filtered batch at that stage
        e.g., {"original": batch, "time": batch_after_time, "pod": batch_after_pod}
    output_dir : Path
        Base output directory (filters subdir will be created)
    batch_idx : int
        Batch index (only saves if batch_idx == 0)
    pair_idx : int
        Pair index within batch (only saves if pair_idx == 0)
    """
    if batch_idx != 0 or pair_idx != 0:
        return

    filters_dir = output_dir / "filters"
    filters_dir.mkdir(parents=True, exist_ok=True)

    # Save each stage
    for stage_name, batch in filtered_stages.items():
        if batch is None or batch.size == 0:
            continue

        # Get first pair
        frame_a = batch[0, 0]  # First pair, frame A
        frame_b = batch[0, 1]  # First pair, frame B

        save_diagnostic_image(frame_a, filters_dir, f"{stage_name}_A.tif")
        save_diagnostic_image(frame_b, filters_dir, f"{stage_name}_B.tif")

    logging.info(f"Saved filter diagnostic images to {filters_dir}")


def _compute_image_stats(image: np.ndarray) -> dict:
    """
    Compute intensity statistics for an image.

    Parameters
    ----------
    image : np.ndarray
        Image array (H, W)

    Returns
    -------
    dict
        Statistics: mean, std, min, max, histogram_counts, histogram_bins
    """
    img = image.astype(np.float32)
    # Compute histogram with 256 bins for intensity analysis
    hist_counts, hist_bins = np.histogram(img.ravel(), bins=256)

    return {
        'mean': float(img.mean()),
        'std': float(img.std()),
        'min': float(img.min()),
        'max': float(img.max()),
        'shape': list(img.shape),
        'histogram_counts': hist_counts.tolist(),
        'histogram_bins': hist_bins.tolist(),
    }


def save_warped_diagnostics(
    image_a_warped: np.ndarray,
    image_b_warped: np.ndarray,
    output_dir: Path,
    pass_idx: int,
    pair_idx: int = 0,
    image_a_original: Optional[np.ndarray] = None,
    image_b_original: Optional[np.ndarray] = None,
) -> None:
    """
    Save warped image pair for a pass with intensity statistics.

    For pass 1 only: also saves the original (post-filter, pre-warp) images once.
    For all passes: saves the warped images and intensity statistics JSON.

    Only saves the first image pair of each pass.

    Parameters
    ----------
    image_a_warped : np.ndarray
        Warped frame A, shape (H, W)
    image_b_warped : np.ndarray
        Warped frame B, shape (H, W)
    output_dir : Path
        Base output directory (filters subdir will be created)
    pass_idx : int
        PIV pass index (0-based)
    pair_idx : int
        Pair index within batch (only saves if pair_idx == 0)
    image_a_original : Optional[np.ndarray]
        Original (pre-warp) frame A, shape (H, W). Only saved for pass 1.
    image_b_original : Optional[np.ndarray]
        Original (pre-warp) frame B, shape (H, W). Only saved for pass 1.
    """
    import json

    if pair_idx != 0:
        return

    filters_dir = output_dir / "filters"
    filters_dir.mkdir(parents=True, exist_ok=True)

    # Check if diagnostics for this pass already saved by another worker
    # (race condition with multiple Dask workers)
    stats_file = filters_dir / f"pass{pass_idx + 1}_intensity_stats.json"
    if stats_file.exists():
        logging.debug(f"Pass {pass_idx + 1} diagnostics already saved by another worker, skipping")
        return

    # Collect stats for JSON output
    pass_stats = {
        'pass_idx': pass_idx + 1,
        'frame_A_warped': _compute_image_stats(image_a_warped),
        'frame_B_warped': _compute_image_stats(image_b_warped),
    }

    # Save original (post-filter, pre-warp) images only for pass 1
    # These are the same for all passes, so only need to save once
    if pass_idx == 0:
        if image_a_original is not None:
            save_diagnostic_image(
                image_a_original, filters_dir, "filtered_A.tif"
            )
            pass_stats['frame_A_original'] = _compute_image_stats(image_a_original)
        if image_b_original is not None:
            save_diagnostic_image(
                image_b_original, filters_dir, "filtered_B.tif"
            )
            pass_stats['frame_B_original'] = _compute_image_stats(image_b_original)

    # Save warped images for each pass
    save_diagnostic_image(
        image_a_warped, filters_dir, f"pass{pass_idx + 1}_A_warped.tif"
    )
    save_diagnostic_image(
        image_b_warped, filters_dir, f"pass{pass_idx + 1}_B_warped.tif"
    )

    # Save intensity statistics as JSON for quantitative analysis
    with open(stats_file, 'w') as f:
        json.dump(pass_stats, f, indent=2)

    logging.info(
        f"Saved warped diagnostic images for pass {pass_idx + 1} to {filters_dir}"
    )
    logging.info(
        f"Pass {pass_idx + 1} intensity stats: "
        f"A_mean={pass_stats['frame_A_warped']['mean']:.2f}, "
        f"A_std={pass_stats['frame_A_warped']['std']:.2f}, "
        f"B_mean={pass_stats['frame_B_warped']['mean']:.2f}, "
        f"B_std={pass_stats['frame_B_warped']['std']:.2f}"
    )
