"""
Dask-Centric Pipeline Utilities

This module provides utilities for the Dask-native PIV processing pipeline.
Uses true Dask patterns: rechunk, map_blocks, persist, scatter, submit, gather.

Key patterns:
- rechunk_for_batched_processing: Align chunks with batch_size
- apply_all_filters: Unified filter function for map_blocks
- scatter_immutable_data: Broadcast cache/masks once to all workers
"""

import gc
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import dask.array as da
import numpy as np
from dask.distributed import Client
from scipy.io import savemat

from pivtools_core.config import Config
from pivtools_cli.piv.piv_backend.factory import make_correlator_backend


logger = logging.getLogger(__name__)


# =============================================================================
# INTERMEDIATE FILTER OUTPUT SAVING
# =============================================================================

def _save_intermediate_frame(
    block: np.ndarray,
    save_dir: Path,
    filename_prefix: str,
) -> None:
    """
    Save the first frame pair (A and B) from a block as single-precision .mat files.

    Args:
        block: Image batch of shape (N, 2, H, W)
        save_dir: Directory to save files
        filename_prefix: Prefix for filenames (e.g., 'before_filtering', 'after_gaussian')
    """
    if block.ndim != 4 or block.shape[0] == 0:
        return

    # Ensure directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract first frame pair
    frame_A = block[0, 0, :, :].astype(np.float32)
    frame_B = block[0, 1, :, :].astype(np.float32)

    # Save as .mat files
    savemat(save_dir / f"{filename_prefix}_A.mat", {"frame": frame_A}, do_compression=True)
    savemat(save_dir / f"{filename_prefix}_B.mat", {"frame": frame_B}, do_compression=True)

    logger.debug(f"Saved intermediate frames to {save_dir / filename_prefix}_*.mat")


# =============================================================================
# FILTER HELPERS
# =============================================================================

def get_spatial_filter_specs(config: Config) -> List[dict]:
    """
    Get list of spatial filter specifications from config.

    Spatial filters operate element-wise and don't need temporal context.

    Returns:
        List of filter spec dicts (e.g., [{'type': 'gaussian', 'sigma': 1.0}])
    """
    TEMPORAL_FILTERS = {'time', 'pod'}
    filters = config.filters or []
    return [f for f in filters if f.get('type') not in TEMPORAL_FILTERS]


def get_temporal_filter_specs(config: Config) -> List[dict]:
    """
    Get list of temporal filter specifications from config.

    Temporal filters (POD, time) need multiple images in the batch.

    Returns:
        List of filter spec dicts (e.g., [{'type': 'pod'}])
    """
    TEMPORAL_FILTERS = {'time', 'pod'}
    filters = config.filters or []
    return [f for f in filters if f.get('type') in TEMPORAL_FILTERS]


def has_temporal_filters(config: Config) -> bool:
    """Check if config includes any temporal filters (POD, time)."""
    return len(get_temporal_filter_specs(config)) > 0


def has_spatial_filters(config: Config) -> bool:
    """Check if config includes any spatial filters."""
    return len(get_spatial_filter_specs(config)) > 0


# =============================================================================
# DASK PIPELINE FUNCTIONS
# =============================================================================

def rechunk_for_batched_processing(images: da.Array, batch_size: int) -> da.Array:
    """
    Rechunk images for batched processing.

    Groups images into chunks of batch_size along the first dimension.
    This ensures temporal coherence for filters like POD and enables
    efficient batched correlation.

    Args:
        images: Dask array of shape (N, 2, H, W) with chunks (1, 2, H, W)
        batch_size: Number of image pairs per chunk

    Returns:
        Dask array of shape (N, 2, H, W) with chunks (batch_size, 2, H, W)
    """
    return images.rechunk((batch_size, 2, -1, -1))


def apply_all_filters_slim(
    block: np.ndarray,
    spatial_specs: List[dict],
    temporal_specs: List[dict],
    pixel_mask: Optional[np.ndarray] = None,
    save_intermediate_base: Optional[str] = None,
    num_frame_pairs: Optional[int] = None,
    block_id: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """
    Unified filter function for map_blocks (slim version).

    This version takes filter specs directly instead of the full config object,
    avoiding repeated serialization of the entire config for every chunk.

    Applies all configured filters in order:
    1. Pixel mask (zero masked regions)
    2. Spatial filters (gaussian, median, norm, etc.)
    3. Temporal filters (POD, time) - only if configured

    This function is called by dask.array.map_blocks on each chunk.
    The chunk is already computed when it reaches this function.

    Args:
        block: Image batch of shape (N, 2, H, W)
        spatial_specs: List of spatial filter specifications
        temporal_specs: List of temporal filter specifications
        pixel_mask: Boolean mask (H, W) where True = masked (optional)
        save_intermediate_base: Base path for saving intermediate outputs (optional)
            If provided, saves frames to {base}/basic_filters/{num_frame_pairs}/{batch_no}/
        num_frame_pairs: Number of frame pairs (for path construction)
        block_id: Block ID from map_blocks (automatically populated by dask)

    Returns:
        Filtered block of same shape
    """
    from pivtools_cli.preprocessing.pod_filter import pod_filter_batch, time_filter_batch

    # Validate input
    if block.ndim != 4:
        logger.warning(f"apply_all_filters_slim: Expected 4D block, got {block.ndim}D")
        return block

    N, C, H, W = block.shape
    logger.debug(f"Applying filters to block: shape={block.shape}")

    # Determine if we should save intermediate outputs
    save_intermediate = (
        save_intermediate_base is not None and
        num_frame_pairs is not None and
        block_id is not None and
        (spatial_specs or temporal_specs or pixel_mask is not None)
    )

    if save_intermediate:
        batch_no = block_id[0]  # First dimension is the batch index
        save_dir = Path(save_intermediate_base) / "basic_filters" / str(num_frame_pairs) / f"batch_{batch_no:04d}"
        logger.info(f"Saving intermediate filter outputs to {save_dir}")

    # Single copy at start - all subsequent operations modify in-place
    # This avoids multiple copies if both mask and filters are applied
    needs_copy = (pixel_mask is not None or spatial_specs or temporal_specs)
    if needs_copy:
        block = block.copy()

    # Save before any filtering
    if save_intermediate:
        _save_intermediate_frame(block, save_dir, "00_before_filtering")

    filter_idx = 1  # Counter for filter ordering in filenames

    # 1. Apply pixel mask (zero masked regions) - now in-place
    if pixel_mask is not None:
        if pixel_mask.shape == (H, W):
            block[:, :, pixel_mask] = 0
            logger.debug(f"Applied pixel mask: {np.sum(pixel_mask)} pixels zeroed")
            if save_intermediate:
                _save_intermediate_frame(block, save_dir, f"{filter_idx:02d}_after_pixel_mask")
                filter_idx += 1
        else:
            logger.warning(f"Pixel mask shape {pixel_mask.shape} != image shape ({H}, {W})")

    # 2. Apply spatial filters (one at a time for intermediate saving)
    if spatial_specs:
        for spec in spatial_specs:
            filter_type = spec.get('type')
            block = _apply_spatial_filters_numpy(block, [spec])
            if save_intermediate:
                _save_intermediate_frame(block, save_dir, f"{filter_idx:02d}_after_{filter_type}")
                filter_idx += 1

    # 3. Apply temporal filters (POD, time)
    for spec in temporal_specs:
        filter_type = spec.get('type')

        if filter_type == 'pod':
            eps_auto_psi = spec.get('eps_auto_psi', 0.01)
            eps_auto_sigma = spec.get('eps_auto_sigma', 0.01)
            block = pod_filter_batch(
                block, None,  # Don't need config, specs are already extracted
                eps_auto_psi=eps_auto_psi,
                eps_auto_sigma=eps_auto_sigma
            )
        elif filter_type == 'time':
            block = time_filter_batch(block)

        if save_intermediate:
            _save_intermediate_frame(block, save_dir, f"{filter_idx:02d}_after_{filter_type}")
            filter_idx += 1

    return block


# Keep the old function for backward compatibility (can be removed later)
def apply_all_filters(
    block: np.ndarray,
    config: Config,
    pixel_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Unified filter function for map_blocks (legacy version).

    DEPRECATED: Use apply_all_filters_slim instead to avoid serializing
    the full config object for every chunk.
    """
    spatial_specs = get_spatial_filter_specs(config)
    temporal_specs = get_temporal_filter_specs(config)
    return apply_all_filters_slim(block, spatial_specs, temporal_specs, pixel_mask)


def _apply_spatial_filters_numpy(
    block: np.ndarray,
    filter_specs: List[dict],
) -> np.ndarray:
    """
    Apply spatial filters to a numpy block.

    Uses scipy.ndimage for numpy-based filtering (more efficient than
    dask_image when we already have the block computed).

    Args:
        block: Image batch of shape (N, 2, H, W)
        filter_specs: List of filter specifications

    Returns:
        Filtered block
    """
    from scipy.ndimage import (
        gaussian_filter as scipy_gaussian,
        median_filter as scipy_median,
        maximum_filter as scipy_maximum,
        minimum_filter as scipy_minimum,
        uniform_filter as scipy_uniform,
    )

    for spec in filter_specs:
        filter_type = spec.get('type')

        if filter_type == 'gaussian':
            sigma = spec.get('sigma', 1.0)
            # Apply to spatial dimensions only (last 2)
            block = scipy_gaussian(block, sigma=(0, 0, sigma, sigma))

        elif filter_type == 'median':
            size = spec.get('size', (5, 5))
            if isinstance(size, list):
                size = tuple(size)
            # Ensure odd size
            size = tuple(s + (s + 1) % 2 for s in size)
            block = scipy_median(block, size=(1, 1) + size)

        elif filter_type == 'norm':
            size = spec.get('size', (7, 7))
            max_gain = spec.get('max_gain', 1.0)
            if isinstance(size, list):
                size = tuple(size)
            size = tuple(s + (s + 1) % 2 for s in size)
            spatial_size = (1, 1) + size

            block_float = block.astype(np.float32)
            local_min = scipy_minimum(block_float, size=spatial_size)
            local_max = scipy_maximum(block_float, size=spatial_size)
            denom = np.maximum(local_max - local_min, 1.0 / max_gain)
            block = ((block_float - local_min) / denom).astype(block.dtype)

        elif filter_type == 'maxnorm':
            size = spec.get('size', (7, 7))
            max_gain = spec.get('max_gain', 1.0)
            if isinstance(size, list):
                size = tuple(size)
            size = tuple(s + (s + 1) % 2 for s in size)
            spatial_size = (1, 1) + size

            block_float = block.astype(np.float32)
            local_max = scipy_maximum(block_float, size=spatial_size)
            local_min = scipy_minimum(block_float, size=spatial_size)
            contrast = local_max - local_min
            smoothed = scipy_uniform(contrast, size=spatial_size)
            denom = np.maximum(smoothed, 1.0 / max_gain)
            block = (np.maximum(block_float, 0) / denom).astype(block.dtype)

        elif filter_type == 'lmax':
            size = spec.get('size', (7, 7))
            if isinstance(size, list):
                size = tuple(size)
            size = tuple(s + (s + 1) % 2 for s in size)
            block = scipy_maximum(block, size=(1, 1) + size)

        elif filter_type == 'clip':
            threshold = spec.get('threshold')
            n = spec.get('n', 2.0)
            if threshold is not None:
                lower, upper = threshold
                block = np.clip(block, lower, upper)
            else:
                # Median-based threshold
                med = np.median(block, axis=(2, 3), keepdims=True)
                std = np.std(block, axis=(2, 3), keepdims=True)
                upper = med + n * std
                block = np.clip(block, 0, upper)

        elif filter_type == 'invert':
            offset = spec.get('offset', 0)
            block = offset - block

        elif filter_type == 'sbg':
            bg = spec.get('bg', 0)
            block = np.maximum(0, block - bg)

        else:
            logger.warning(f"Unknown spatial filter type: {filter_type}")

    return block


def create_filter_pipeline(
    images: da.Array,
    config: Config,
    pixel_mask: Optional[np.ndarray] = None,
    save_intermediate_base: Optional[Path] = None,
) -> da.Array:
    """
    Create a lazy filter pipeline using map_blocks.

    This wraps apply_all_filters_slim for use with Dask arrays.
    Filter specs are extracted once here to avoid serializing the
    full config object for every chunk.

    Args:
        images: Dask array of shape (N, 2, H, W), already rechunked
        config: Configuration object
        pixel_mask: Optional pixel mask (H, W)
        save_intermediate_base: Optional base path for saving intermediate filter outputs.
            If provided, saves frames to {base}/basic_filters/{num_frame_pairs}/{batch_no}/

    Returns:
        Dask array with filters applied lazily
    """
    logger.info("Creating filter pipeline...")

    # Extract filter specs ONCE here (avoids serializing full config per chunk)
    spatial_specs = get_spatial_filter_specs(config)
    temporal_specs = get_temporal_filter_specs(config)

    if spatial_specs:
        logger.info(f"  Spatial filters: {[f.get('type') for f in spatial_specs]}")
    if temporal_specs:
        logger.info(f"  Temporal filters: {[f.get('type') for f in temporal_specs]}")
    if pixel_mask is not None:
        logger.info(f"  Pixel mask: {np.sum(pixel_mask)} masked pixels")
    if save_intermediate_base is not None:
        logger.info(f"  Saving intermediate outputs to: {save_intermediate_base}/basic_filters/...")

    # If no filters and no mask, return unchanged
    if not spatial_specs and not temporal_specs and pixel_mask is None:
        logger.info("  No filters configured, returning images unchanged")
        return images

    # Prepare intermediate saving parameters
    save_base_str = str(save_intermediate_base) if save_intermediate_base is not None else None
    num_frame_pairs = config.num_frame_pairs if save_intermediate_base is not None else None

    # Apply filters via map_blocks using the slim version
    # This only serializes the filter specs (small dicts), not the full config
    # Use block_id to get the batch number for intermediate saving
    filtered = images.map_blocks(
        apply_all_filters_slim,
        spatial_specs=spatial_specs,
        temporal_specs=temporal_specs,
        pixel_mask=pixel_mask,
        save_intermediate_base=save_base_str,
        num_frame_pairs=num_frame_pairs,
        dtype=images.dtype,
        block_id=True,  # Tell dask to pass block_id to the function
    )

    return filtered


# =============================================================================
# DATA SCATTERING
# =============================================================================

def scatter_immutable_data(
    client: Client,
    config: Config,
    vector_masks: Optional[List[np.ndarray]] = None,
    pixel_mask: Optional[np.ndarray] = None,
    ensemble: bool = False,
) -> Dict[str, Any]:
    """
    Scatter immutable data once to all workers.

    This broadcasts the correlator cache and masks to all workers,
    avoiding repeated transfers per task.

    Args:
        client: Dask distributed client
        config: Configuration object
        vector_masks: Pre-computed vector masks per pass
        pixel_mask: Pixel mask for preprocessing
        ensemble: Whether this is ensemble mode

    Returns:
        Dict with 'cache' and 'masks' keys containing scattered futures
    """
    logger.info("Scattering immutable data to workers...")

    # Create and scatter correlator cache
    temp_correlator = make_correlator_backend(config, ensemble=ensemble)
    correlator_cache = temp_correlator.get_cache_data()
    scattered_cache = client.scatter(correlator_cache, broadcast=True)

    cache_size = sum(
        v.nbytes if hasattr(v, 'nbytes') else 0
        for v in correlator_cache.values()
        if hasattr(v, 'nbytes')
    )
    logger.info(f"  Scattered correlator cache (~{cache_size / 1024:.1f} KB)")

    # Scatter vector masks if present
    scattered_masks = None
    if vector_masks:
        scattered_masks = client.scatter(vector_masks, broadcast=True)
        mask_size = sum(m.nbytes for m in vector_masks) / 1024
        logger.info(f"  Scattered vector masks ({mask_size:.1f} KB)")

    return {
        'cache': scattered_cache,
        'masks': scattered_masks,
    }


# =============================================================================
# CORRELATION HELPERS
# =============================================================================

def correlate_and_save_batch(
    batch: np.ndarray,
    start_img_idx: int,
    config: Config,
    scattered_cache: dict,
    scattered_masks: Optional[List[np.ndarray]],
    output_path: Path,
    runs_to_save: List[int],
    vector_format: str,
) -> List[str]:
    """
    Process multiple image pairs on one worker.

    This reduces task overhead by processing a batch of pairs instead
    of submitting one task per pair.

    Args:
        batch: Image batch of shape (N, 2, H, W)
        start_img_idx: Frame number of first pair (1-indexed)
        config: Configuration object
        scattered_cache: Pre-scattered correlator cache
        scattered_masks: Pre-scattered vector masks
        output_path: Directory for saving results
        runs_to_save: Which PIV runs to save
        vector_format: Format string for output files

    Returns:
        List of saved file paths
    """
    from pivtools_cli.piv.piv import _process_and_save_batch

    saved_paths = []

    saved_paths = _process_and_save_batch(
            batch,
            start_img_idx,
            config,
            scattered_masks,
            scattered_cache,
            output_path,
            runs_to_save,
            vector_format,
        )
    #saved_paths.append(path)

    return saved_paths


def correlate_batch_for_accumulation_distributed(
    batch: np.ndarray,
    config: Config,
    pass_idx: int,
    scattered_predictor: Optional[np.ndarray],
    scattered_cache: dict,
    scattered_masks: Optional[List[np.ndarray]],
    is_first_batch: bool = False,
    output_path: Optional[str] = None,
) -> dict:
    """
    Wrapper for distributed ensemble correlation.

    Creates a correlator from the scattered cache and processes
    the batch for accumulation.

    Args:
        batch: Image batch of shape (N, 2, H, W)
        config: Configuration object
        pass_idx: Current pass index
        scattered_predictor: Pre-scattered predictor field
        scattered_cache: Pre-scattered correlator cache
        scattered_masks: Pre-scattered vector masks
        is_first_batch: If True, capture first-pair warped images for diagnostics
        output_path: Path for saving diagnostic images

    Returns:
        Dict with correlation sums (corr_AA_sum, corr_BB_sum, corr_AB_sum, etc.)
    """
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    correlator = EnsembleCorrelatorCPU(
        config,
        precomputed_cache=scattered_cache,
        vector_masks=scattered_masks,
    )

    result = correlator.correlate_batch_for_accumulation(
        batch,
        config,
        pass_idx=pass_idx,
        predictor_field=scattered_predictor,
        is_first_batch=is_first_batch,
        save_diagnostics=config.ensemble_save_diagnostics,
        output_path=output_path,
    )

    return result


def reduce_ensemble_results(r1: dict, r2: dict) -> dict:
    """
    Combine two ensemble correlation results.

    Used to reduce batch results into accumulated sums.

    Args:
        r1, r2: Results from correlate_batch_for_accumulation containing:
            - corr_AA_sum, corr_BB_sum, corr_AB_sum: Correlation planes
            - warp_A_sum, warp_B_sum: Warped image sums
            - n_images: Image count

    Returns:
        Combined result with summed arrays
    """
    # Keep first-pair images from whichever result has them (only one should)
    first_pair_A = r1.get("first_pair_A") if r1.get("first_pair_A") is not None else r2.get("first_pair_A")
    first_pair_B = r1.get("first_pair_B") if r1.get("first_pair_B") is not None else r2.get("first_pair_B")

    return {
        "corr_AA_sum": r1["corr_AA_sum"] + r2["corr_AA_sum"],
        "corr_BB_sum": r1["corr_BB_sum"] + r2["corr_BB_sum"],
        "corr_AB_sum": r1["corr_AB_sum"] + r2["corr_AB_sum"],
        "warp_A_sum": r1["warp_A_sum"] + r2["warp_A_sum"],
        "warp_B_sum": r1["warp_B_sum"] + r2["warp_B_sum"],
        "n_images": r1["n_images"] + r2["n_images"],
        "n_win_x": r1["n_win_x"],
        "n_win_y": r1["n_win_y"],
        "smoothed_predictor": r1.get("smoothed_predictor"),
        "vector_mask": r1.get("vector_mask"),
        # Padding values for predictor field storage
        "n_pre": r1.get("n_pre"),
        "n_post": r1.get("n_post"),
        # First-pair warped images for diagnostic saving
        "first_pair_A": first_pair_A,
        "first_pair_B": first_pair_B,
    }


def extract_predictor_field(pass_result) -> np.ndarray:
    """
    Extract predictor field from pass result for next pass.

    Args:
        pass_result: PIVEnsemblePassResult from finalize_pass

    Returns:
        Predictor field of shape (n_win_y, n_win_x, 2) containing [uy, ux]
        NOTE: Returns UNPADDED field. Padding is applied inside _get_im_mesh()
        using pass-specific n_pre_all/n_post_all values to match the
        interpolation grid coordinates (win_ctrs_x_all, win_ctrs_y_all).
    """
    uy = pass_result.uy_mat.copy()
    ux = pass_result.ux_mat.copy()

    # Stack as [uy, ux] along last dimension
    predictor_field = np.stack([uy, ux], axis=-1).astype(np.float32)

    # NOTE: No padding here - _get_im_mesh() in cpu_ensemble.py handles
    # proper padding using n_pre_all/n_post_all which vary by pass configuration

    # DEBUG: Log predictor field statistics to prove it's the WHOLE-PASS MEAN
    # This proves the predictor is NOT batch-dependent - it's computed from
    # ALL images accumulated across ALL batches in the previous pass
    logger.info(
        f"[PREDICTOR DEBUG] Extracted from pass_result (whole-pass mean field):\n"
        f"  Shape: {predictor_field.shape} (n_win_y, n_win_x, 2)\n"
        f"  ux (mean displacement x): mean={np.nanmean(ux):.6f}, std={np.nanstd(ux):.6f}, "
        f"range=[{np.nanmin(ux):.4f}, {np.nanmax(ux):.4f}]\n"
        f"  uy (mean displacement y): mean={np.nanmean(uy):.6f}, std={np.nanstd(uy):.6f}, "
        f"range=[{np.nanmin(uy):.4f}, {np.nanmax(uy):.4f}]\n"
        f"  Source: pass_result.ux_mat, pass_result.uy_mat (ensemble mean from ALL batches)"
    )

    return predictor_field


# =============================================================================
# WORKER-SIDE ACCUMULATION
# =============================================================================

def correlate_and_reduce_on_worker(
    batch_list: List[np.ndarray],
    config: Config,
    pass_idx: int,
    scattered_predictor: Optional[np.ndarray],
    scattered_cache: dict,
    scattered_masks: Optional[List[np.ndarray]],
    output_path: Optional[str] = None,
) -> dict:
    """
    Process multiple batches on one worker, returning single accumulated result.

    This reduces network traffic from O(num_batches) to O(num_workers) by
    accumulating correlation sums locally on the worker before returning.

    Each worker processes all batches assigned to it, correlating each batch
    and summing the correlation planes in-place. Only the final accumulated
    result is returned to the client.

    Args:
        batch_list: List of image batches, each of shape (N, 2, H, W)
        config: Configuration object
        pass_idx: Current pass index
        scattered_predictor: Pre-scattered predictor field (or None for pass 0)
        scattered_cache: Pre-scattered correlator cache
        scattered_masks: Pre-scattered vector masks
        output_path: Path for saving diagnostic images

    Returns:
        Dict with accumulated correlation sums:
            - corr_AA_sum, corr_BB_sum, corr_AB_sum: Summed correlation planes
            - warp_A_sum, warp_B_sum: Summed warped images
            - n_images: Total image count processed
            - n_win_x, n_win_y: Grid dimensions
            - smoothed_predictor, vector_mask: Metadata from last batch
    """
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    # NOTE: This function is deprecated in favor of correlate_single_batch_and_accumulate
    # which uses chained submission for lazy loading. Kept for reference/fallback.

    # Create correlator once for all batches on this worker
    correlator = EnsembleCorrelatorCPU(
        config,
        precomputed_cache=scattered_cache,
        vector_masks=scattered_masks,
    )

    accumulated = None
    is_first_batch_processed = False

    for batch in batch_list:
        # Skip empty batches (can happen with uneven distribution)
        if batch is None or (hasattr(batch, 'shape') and batch.shape[0] == 0):
            continue

        # Correlate this batch (pass is_first_batch for first non-empty batch)
        result = correlator.correlate_batch_for_accumulation(
            batch,
            config,
            pass_idx=pass_idx,
            predictor_field=scattered_predictor,
            is_first_batch=not is_first_batch_processed,
            save_diagnostics=config.ensemble_save_diagnostics,
            output_path=output_path,
        )
        is_first_batch_processed = True

        if accumulated is None:
            # First batch - initialize with copy to avoid aliasing
            accumulated = {
                "corr_AA_sum": result["corr_AA_sum"].copy(),
                "corr_BB_sum": result["corr_BB_sum"].copy(),
                "corr_AB_sum": result["corr_AB_sum"].copy(),
                "warp_A_sum": result["warp_A_sum"].copy(),
                "warp_B_sum": result["warp_B_sum"].copy(),
                "n_images": result["n_images"],
                "n_win_x": result["n_win_x"],
                "n_win_y": result["n_win_y"],
                "smoothed_predictor": result.get("smoothed_predictor"),
                "vector_mask": result.get("vector_mask"),
                # Padding values for PADDED predictor storage (matching instantaneous)
                "n_pre": result.get("n_pre"),
                "n_post": result.get("n_post"),
                # First-pair warped images for diagnostic saving
                "first_pair_A": result.get("first_pair_A"),
                "first_pair_B": result.get("first_pair_B"),
            }
        else:
            # Subsequent batches - in-place accumulation
            accumulated["corr_AA_sum"] += result["corr_AA_sum"]
            accumulated["corr_BB_sum"] += result["corr_BB_sum"]
            accumulated["corr_AB_sum"] += result["corr_AB_sum"]
            accumulated["warp_A_sum"] += result["warp_A_sum"]
            accumulated["warp_B_sum"] += result["warp_B_sum"]
            accumulated["n_images"] += result["n_images"]
            # Keep metadata from latest result
            if result.get("smoothed_predictor") is not None:
                accumulated["smoothed_predictor"] = result["smoothed_predictor"]
            if result.get("vector_mask") is not None:
                accumulated["vector_mask"] = result["vector_mask"]

    # Handle case where all batches were empty/None
    if accumulated is None:
        raise ValueError("No valid batches to process on this worker")

    return accumulated


def compute_warp_sums_on_worker(
    batch_list: List[np.ndarray],
    config: Config,
    pass_idx: int,
    scattered_predictor: Optional[np.ndarray],
    scattered_cache: dict,
    scattered_masks: Optional[List[np.ndarray]],
) -> dict:
    """
    First pass for 'image' background method: compute warped image sums only.

    This is the first half of the two-pass 'image' background subtraction method.
    It loads images, warps them if pass > 0, and accumulates the warped image sums
    to compute mean images (Ā, B̄).

    Args:
        batch_list: List of image batches, each of shape (N, 2, H, W)
        config: Configuration object
        pass_idx: Current pass index
        scattered_predictor: Pre-scattered predictor field (or None for pass 0)
        scattered_cache: Pre-scattered correlator cache
        scattered_masks: Pre-scattered vector masks

    Returns:
        Dict with accumulated warp sums:
            - warp_A_sum: Summed warped A images (H, W)
            - warp_B_sum: Summed warped B images (H, W)
            - n_images: Total image count processed
            - smoothed_predictor: Smoothed predictor from last batch
    """
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    # NOTE: This function is deprecated in favor of warp_single_batch_and_accumulate
    # which uses chained submission for lazy loading. Kept for reference/fallback.

    # Create correlator once for all batches on this worker
    correlator = EnsembleCorrelatorCPU(
        config,
        precomputed_cache=scattered_cache,
        vector_masks=scattered_masks,
    )

    accumulated = None

    for batch in batch_list:
        # Skip empty batches
        if batch is None or (hasattr(batch, 'shape') and batch.shape[0] == 0):
            continue

        # Compute warp sums only (no correlation)
        result = correlator.compute_warp_sums_only(
            batch,
            config,
            pass_idx=pass_idx,
            predictor_field=scattered_predictor,
        )

        if accumulated is None:
            accumulated = {
                "warp_A_sum": result["warp_A_sum"].copy(),
                "warp_B_sum": result["warp_B_sum"].copy(),
                "n_images": result["n_images"],
                "smoothed_predictor": result.get("smoothed_predictor"),
            }
        else:
            accumulated["warp_A_sum"] += result["warp_A_sum"]
            accumulated["warp_B_sum"] += result["warp_B_sum"]
            accumulated["n_images"] += result["n_images"]
            if result.get("smoothed_predictor") is not None:
                accumulated["smoothed_predictor"] = result["smoothed_predictor"]

    if accumulated is None:
        raise ValueError("No valid batches to process on this worker")

    return accumulated


def correlate_mean_subtracted_on_worker(
    batch_list: List[np.ndarray],
    config: Config,
    pass_idx: int,
    scattered_predictor: Optional[np.ndarray],
    scattered_means: dict,
    scattered_cache: dict,
    scattered_masks: Optional[List[np.ndarray]],
) -> dict:
    """
    Second pass for 'image' background method: correlate mean-subtracted images.

    This is the second half of the two-pass 'image' background subtraction method.
    It loads images, warps them if pass > 0, subtracts the pre-computed mean images,
    then correlates the mean-subtracted images.

    Formula: R_ensemble = <(A - Ā) ⊗ (B - B̄)>

    Args:
        batch_list: List of image batches, each of shape (N, 2, H, W)
        config: Configuration object
        pass_idx: Current pass index
        scattered_predictor: Pre-scattered predictor field (or None for pass 0)
        scattered_means: Dict with 'A_mean' and 'B_mean' arrays (H, W)
        scattered_cache: Pre-scattered correlator cache
        scattered_masks: Pre-scattered vector masks

    Returns:
        Dict with accumulated correlation sums:
            - corr_AA_sum, corr_BB_sum, corr_AB_sum: Summed correlation planes
            - n_images: Total image count processed
            - n_win_x, n_win_y: Grid dimensions
            - smoothed_predictor, vector_mask: Metadata from last batch
    """
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    # Create correlator once for all batches on this worker
    correlator = EnsembleCorrelatorCPU(
        config,
        precomputed_cache=scattered_cache,
        vector_masks=scattered_masks,
    )

    # Extract mean images
    A_mean = scattered_means["A_mean"]
    B_mean = scattered_means["B_mean"]

    accumulated = None
    is_first_batch_processed = False

    for batch in batch_list:
        # Skip empty batches
        if batch is None or (hasattr(batch, 'shape') and batch.shape[0] == 0):
            continue

        # Correlate mean-subtracted images
        result = correlator.correlate_mean_subtracted_batch(
            batch,
            config,
            pass_idx=pass_idx,
            A_mean=A_mean,
            B_mean=B_mean,
            predictor_field=scattered_predictor,
            is_first_batch=not is_first_batch_processed,
        )
        is_first_batch_processed = True

        if accumulated is None:
            # Get image shape from first pair for dummy warp sums
            H, W = batch.shape[2], batch.shape[3]
            accumulated = {
                "corr_AA_sum": result["corr_AA_sum"].copy(),
                "corr_BB_sum": result["corr_BB_sum"].copy(),
                "corr_AB_sum": result["corr_AB_sum"].copy(),
                # Dummy warp sums (zeros) - not needed for 'image' method but
                # required for reduce_ensemble_results compatibility
                "warp_A_sum": np.zeros((H, W), dtype=np.float32),
                "warp_B_sum": np.zeros((H, W), dtype=np.float32),
                "n_images": result["n_images"],
                "n_win_x": result["n_win_x"],
                "n_win_y": result["n_win_y"],
                "smoothed_predictor": result.get("smoothed_predictor"),
                "vector_mask": result.get("vector_mask"),
                "n_pre": result.get("n_pre"),
                "n_post": result.get("n_post"),
                "first_pair_A": result.get("first_pair_A"),
                "first_pair_B": result.get("first_pair_B"),
            }
        else:
            accumulated["corr_AA_sum"] += result["corr_AA_sum"]
            accumulated["corr_BB_sum"] += result["corr_BB_sum"]
            accumulated["corr_AB_sum"] += result["corr_AB_sum"]
            accumulated["n_images"] += result["n_images"]
            # warp sums stay at zero for 'image' method
            if result.get("smoothed_predictor") is not None:
                accumulated["smoothed_predictor"] = result["smoothed_predictor"]
            if result.get("vector_mask") is not None:
                accumulated["vector_mask"] = result["vector_mask"]

    if accumulated is None:
        raise ValueError("No valid batches to process on this worker")

    return accumulated


# =============================================================================
# SINGLE-BATCH CHAINED ACCUMULATION FUNCTIONS
# =============================================================================
#
# These functions process ONE batch at a time and are designed for chained
# Dask submission. This preserves lazy loading - Dask only resolves the
# dependencies needed for each task (one batch + previous accumulated sum).
#
# CRITICAL: Use `+` NOT `+=` to create NEW arrays. This ensures idempotency -
# if Dask retries a failed task, the input `accumulated` is untouched.


def correlate_single_batch_and_accumulate(
    accumulated: Optional[dict],
    batch: np.ndarray,
    config: Config,
    pass_idx: int,
    predictor_field: Optional[np.ndarray],
    cache: dict,
    masks: Optional[List[np.ndarray]],
    is_first_batch: bool = False,
    output_path: Optional[str] = None,
) -> dict:
    """
    Correlate ONE batch and accumulate with previous sum.

    Designed for chained submission where each task depends on:
    - accumulated: Future from previous task (or None for first batch)
    - batch: ONE batch Future (lazily resolved by Dask)

    This keeps memory usage to ~100MB (1 batch + accumulated sum) instead
    of loading all batches upfront.

    CRITICAL: We use `+` NOT `+=` to create NEW arrays. This ensures
    idempotency - if Dask retries a failed task, the input `accumulated`
    is untouched and won't cause double-counting.

    Args:
        accumulated: Previous accumulated result (None for first batch)
        batch: Single image batch of shape (N, 2, H, W)
        config: Configuration object
        pass_idx: Current pass index
        predictor_field: Predictor field (or None for pass 0)
        cache: Correlator cache
        masks: Vector masks
        is_first_batch: Whether this is the first batch (for diagnostics)
        output_path: Path for saving diagnostic images

    Returns:
        Dict with accumulated correlation sums
    """
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    # Skip empty batches
    if batch is None or (hasattr(batch, 'shape') and batch.shape[0] == 0):
        return accumulated if accumulated is not None else {}

    # TEMPORARY DEBUG: Monitor memory per batch to confirm lazy loading
    import psutil
    import os as os_module
    process = psutil.Process(os_module.getpid())
    mem_before = process.memory_info().rss / 1024**3
    n_images_so_far = accumulated["n_images"] if accumulated else 0
    batch_images = batch.shape[0] if hasattr(batch, 'shape') else 0

    correlator = EnsembleCorrelatorCPU(config, precomputed_cache=cache, vector_masks=masks)

    result = correlator.correlate_batch_for_accumulation(
        batch,
        config,
        pass_idx=pass_idx,
        predictor_field=predictor_field,
        is_first_batch=is_first_batch,
        save_diagnostics=config.ensemble_save_diagnostics,
        output_path=output_path,
    )

    mem_after = process.memory_info().rss / 1024**3
    worker_total = n_images_so_far + batch_images
    logger.info(
        f"Batch +{batch_images} images ({worker_total} processed), "
        f"mem: {mem_before:.2f} -> {mem_after:.2f} GB"
    )

    # DIAGNOSTIC: Track data locality across batches
    from distributed import get_worker
    try:
        worker = get_worker()
        worker_addr = worker.address
        # Check if accumulated data has a provenance marker
        acc_from = accumulated.get("_worker_addr", "none") if accumulated else "none"
        logger.debug(
            f"[LOCALITY] Worker {worker_addr[-20:]}: batch +{batch_images} images, "
            f"accumulated_from={acc_from[-20:] if acc_from != 'none' else 'none'}"
        )
        # Tag this result with our worker address for tracking
        result["_worker_addr"] = worker_addr
    except Exception:
        pass  # Running outside worker context

    if accumulated is None:
        # First batch - return result directly (makes copies)
        return {
            "corr_AA_sum": result["corr_AA_sum"].copy(),
            "corr_BB_sum": result["corr_BB_sum"].copy(),
            "corr_AB_sum": result["corr_AB_sum"].copy(),
            "warp_A_sum": result["warp_A_sum"].copy(),
            "warp_B_sum": result["warp_B_sum"].copy(),
            "n_images": result["n_images"],
            "n_win_x": result["n_win_x"],
            "n_win_y": result["n_win_y"],
            "smoothed_predictor": result.get("smoothed_predictor"),
            "vector_mask": result.get("vector_mask"),
            "n_pre": result.get("n_pre"),
            "n_post": result.get("n_post"),
            "first_pair_A": result.get("first_pair_A"),
            "first_pair_B": result.get("first_pair_B"),
            "_worker_addr": result.get("_worker_addr"),  # DIAGNOSTIC: track locality
        }
    else:
        # SAFE: Create shallow copy of container, then NEW arrays for sums
        # This leaves `accumulated` untouched for Dask retry safety
        new_accumulated = accumulated.copy()
        new_accumulated["corr_AA_sum"] = accumulated["corr_AA_sum"] + result["corr_AA_sum"]
        new_accumulated["corr_BB_sum"] = accumulated["corr_BB_sum"] + result["corr_BB_sum"]
        new_accumulated["corr_AB_sum"] = accumulated["corr_AB_sum"] + result["corr_AB_sum"]
        new_accumulated["warp_A_sum"] = accumulated["warp_A_sum"] + result["warp_A_sum"]
        new_accumulated["warp_B_sum"] = accumulated["warp_B_sum"] + result["warp_B_sum"]
        new_accumulated["n_images"] = accumulated["n_images"] + result["n_images"]
        # Metadata updates (overwrite is fine - scalars/small refs)
        for key in ["smoothed_predictor", "vector_mask", "n_pre", "n_post", "_worker_addr"]:
            if result.get(key) is not None:
                new_accumulated[key] = result[key]
        return new_accumulated


def warp_single_batch_and_accumulate(
    accumulated: Optional[dict],
    batch: np.ndarray,
    config: Config,
    pass_idx: int,
    predictor_field: Optional[np.ndarray],
    cache: dict,
    masks: Optional[List[np.ndarray]],
) -> dict:
    """
    Compute warp sums for ONE batch (first pass of 'image' background method).

    Single-batch version of compute_warp_sums_on_worker for chained submission.

    Args:
        accumulated: Previous accumulated result (None for first batch)
        batch: Single image batch of shape (N, 2, H, W)
        config: Configuration object
        pass_idx: Current pass index
        predictor_field: Predictor field (or None for pass 0)
        cache: Correlator cache
        masks: Vector masks

    Returns:
        Dict with accumulated warp sums
    """
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    # Skip empty batches
    if batch is None or (hasattr(batch, 'shape') and batch.shape[0] == 0):
        return accumulated if accumulated is not None else {}

    correlator = EnsembleCorrelatorCPU(config, precomputed_cache=cache, vector_masks=masks)

    result = correlator.compute_warp_sums_only(
        batch,
        config,
        pass_idx=pass_idx,
        predictor_field=predictor_field,
    )

    if accumulated is None:
        return {
            "warp_A_sum": result["warp_A_sum"].copy(),
            "warp_B_sum": result["warp_B_sum"].copy(),
            "n_images": result["n_images"],
            "smoothed_predictor": result.get("smoothed_predictor"),
        }
    else:
        # SAFE: Create NEW arrays using + (not +=) for retry safety
        new_accumulated = accumulated.copy()
        new_accumulated["warp_A_sum"] = accumulated["warp_A_sum"] + result["warp_A_sum"]
        new_accumulated["warp_B_sum"] = accumulated["warp_B_sum"] + result["warp_B_sum"]
        new_accumulated["n_images"] = accumulated["n_images"] + result["n_images"]
        if result.get("smoothed_predictor") is not None:
            new_accumulated["smoothed_predictor"] = result["smoothed_predictor"]
        return new_accumulated


def correlate_mean_subtracted_single_batch(
    accumulated: Optional[dict],
    batch: np.ndarray,
    config: Config,
    pass_idx: int,
    predictor_field: Optional[np.ndarray],
    mean_images: dict,
    cache: dict,
    masks: Optional[List[np.ndarray]],
    is_first_batch: bool = False,
) -> dict:
    """
    Correlate mean-subtracted images for ONE batch (second pass of 'image' method).

    Single-batch version of correlate_mean_subtracted_on_worker for chained submission.

    Args:
        accumulated: Previous accumulated result (None for first batch)
        batch: Single image batch of shape (N, 2, H, W)
        config: Configuration object
        pass_idx: Current pass index
        predictor_field: Predictor field (or None for pass 0)
        mean_images: Dict with 'A_mean' and 'B_mean' arrays
        cache: Correlator cache
        masks: Vector masks
        is_first_batch: Whether this is the first batch (for diagnostics)

    Returns:
        Dict with accumulated correlation sums
    """
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    # Skip empty batches
    if batch is None or (hasattr(batch, 'shape') and batch.shape[0] == 0):
        return accumulated if accumulated is not None else {}

    correlator = EnsembleCorrelatorCPU(config, precomputed_cache=cache, vector_masks=masks)

    A_mean = mean_images["A_mean"]
    B_mean = mean_images["B_mean"]

    result = correlator.correlate_mean_subtracted_batch(
        batch,
        config,
        pass_idx=pass_idx,
        A_mean=A_mean,
        B_mean=B_mean,
        predictor_field=predictor_field,
        is_first_batch=is_first_batch,
    )

    if accumulated is None:
        # Get image shape from first pair for dummy warp sums
        H, W = batch.shape[2], batch.shape[3]
        return {
            "corr_AA_sum": result["corr_AA_sum"].copy(),
            "corr_BB_sum": result["corr_BB_sum"].copy(),
            "corr_AB_sum": result["corr_AB_sum"].copy(),
            # Dummy warp sums (zeros) - not needed for 'image' method but
            # required for reduce_ensemble_results compatibility
            "warp_A_sum": np.zeros((H, W), dtype=np.float32),
            "warp_B_sum": np.zeros((H, W), dtype=np.float32),
            "n_images": result["n_images"],
            "n_win_x": result["n_win_x"],
            "n_win_y": result["n_win_y"],
            "smoothed_predictor": result.get("smoothed_predictor"),
            "vector_mask": result.get("vector_mask"),
            "n_pre": result.get("n_pre"),
            "n_post": result.get("n_post"),
            "first_pair_A": result.get("first_pair_A"),
            "first_pair_B": result.get("first_pair_B"),
        }
    else:
        # SAFE: Create NEW arrays using + (not +=) for retry safety
        new_accumulated = accumulated.copy()
        new_accumulated["corr_AA_sum"] = accumulated["corr_AA_sum"] + result["corr_AA_sum"]
        new_accumulated["corr_BB_sum"] = accumulated["corr_BB_sum"] + result["corr_BB_sum"]
        new_accumulated["corr_AB_sum"] = accumulated["corr_AB_sum"] + result["corr_AB_sum"]
        new_accumulated["n_images"] = accumulated["n_images"] + result["n_images"]
        # warp sums stay at zero for 'image' method
        for key in ["smoothed_predictor", "vector_mask", "n_pre", "n_post"]:
            if result.get(key) is not None:
                new_accumulated[key] = result[key]
        return new_accumulated