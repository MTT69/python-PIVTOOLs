"""
Dask-Centric Pipeline Utilities

This module provides utilities for the Dask-native PIV processing pipeline.
Uses true Dask patterns: map_blocks, persist, scatter, submit, gather.

Key patterns:
- apply_all_filters_slim: Unified filter function for map_blocks
- scatter_immutable_data: Broadcast cache/masks once to all workers
- correlate_worker_batches: Per-worker accumulation for ensemble processing
- reduce_ensemble_results: Merge accumulated dicts (tree reduction)
"""

import logging
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


# =============================================================================
# DASK PIPELINE FUNCTIONS
# =============================================================================

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
        logger.debug(f"Saving intermediate filter outputs to {save_dir}")

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
                block,
                eps_auto_psi=eps_auto_psi,
                eps_auto_sigma=eps_auto_sigma,
            )
        elif filter_type == 'time':
            block = time_filter_batch(block)

        if save_intermediate:
            _save_intermediate_frame(block, save_dir, f"{filter_idx:02d}_after_{filter_type}")
            filter_idx += 1

    return block


def _apply_spatial_filters_numpy(
    block: np.ndarray,
    filter_specs: List[dict],
) -> np.ndarray:
    """
    Apply spatial filters to a numpy block.

    Uses scipy.ndimage for direct numpy-based filtering on computed arrays.

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

    # In-place arithmetic below requires floating-point dtype
    if not np.issubdtype(block.dtype, np.floating):
        block = block.astype(np.float32)

    for spec in filter_specs:
        filter_type = spec.get('type')

        if filter_type == 'gaussian':
            sigma = spec.get('sigma', 1.0)
            # Apply to spatial dimensions only (last 2), in-place
            scipy_gaussian(block, sigma=(0, 0, sigma, sigma), output=block)

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

            local_min = scipy_minimum(block, size=spatial_size)
            local_max = scipy_maximum(block, size=spatial_size)
            local_max -= local_min                                 # range in-place
            np.maximum(local_max, 1.0 / max_gain, out=local_max)  # clamp in-place
            block -= local_min                                     # in-place
            block /= local_max                                     # in-place

        elif filter_type == 'maxnorm':
            size = spec.get('size', (7, 7))
            max_gain = spec.get('max_gain', 1.0)
            if isinstance(size, list):
                size = tuple(size)
            size = tuple(s + (s + 1) % 2 for s in size)
            spatial_size = (1, 1) + size

            local_max = scipy_maximum(block, size=spatial_size)
            scipy_uniform(local_max, size=spatial_size, output=local_max)  # smooth in-place
            np.maximum(local_max, 1.0 / max_gain, out=local_max)
            np.maximum(block, 0, out=block)
            block /= local_max

        elif filter_type == 'lmax':
            size = spec.get('size', (7, 7))
            if isinstance(size, list):
                size = tuple(size)
            size = tuple(s + (s + 1) % 2 for s in size)
            block = scipy_maximum(block, size=(1, 1) + size)

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
    logger.debug("Creating filter pipeline...")

    # Extract filter specs ONCE here (avoids serializing full config per chunk)
    spatial_specs = get_spatial_filter_specs(config)
    temporal_specs = get_temporal_filter_specs(config)

    if spatial_specs:
        logger.debug(f"  Spatial filters: {[f.get('type') for f in spatial_specs]}")
    if temporal_specs:
        logger.debug(f"  Temporal filters: {[f.get('type') for f in temporal_specs]}")
    if pixel_mask is not None:
        logger.debug(f"  Pixel mask: {np.sum(pixel_mask)} masked pixels")
    if save_intermediate_base is not None:
        logger.debug(f"  Saving intermediate outputs to: {save_intermediate_base}/basic_filters/...")

    # If no filters and no mask, return unchanged
    if not spatial_specs and not temporal_specs and pixel_mask is None:
        logger.debug("  No filters configured, returning images unchanged")
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
        "padded_predictor": r1.get("padded_predictor"),
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

    logger.debug(
        f"Predictor extracted from pass_result (whole-pass mean field): "
        f"shape={predictor_field.shape}, "
        f"ux: mean={np.nanmean(ux):.6f}, std={np.nanstd(ux):.6f}, "
        f"range=[{np.nanmin(ux):.4f}, {np.nanmax(ux):.4f}], "
        f"uy: mean={np.nanmean(uy):.6f}, std={np.nanstd(uy):.6f}, "
        f"range=[{np.nanmin(uy):.4f}, {np.nanmax(uy):.4f}]"
    )

    return predictor_field


# =============================================================================
# ENSEMBLE CORRELATION & REDUCTION
# =============================================================================


def _log_worker_memory(label, pass_idx, batch_idx=-1):
    """Log worker RSS using psutil (zero-cost if psutil not available)."""
    try:
        import psutil
        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        logger.debug(
            f"[Memory] pass={pass_idx} batch={batch_idx} {label}: RSS={rss_mb:.0f} MB"
        )
    except ImportError:
        pass


def _deep_dict_nbytes(d):
    """Sum .nbytes of all numpy arrays in a dict."""
    total = 0
    for v in d.values():
        if hasattr(v, 'nbytes'):
            total += v.nbytes
    return total


def correlate_worker_batches(
    batch_indices: list,
    config: Config,
    pass_idx: int,
    predictor_field: Optional[np.ndarray],
    cache: dict,
    masks: Optional[List[np.ndarray]],
    camera_num: int = 0,
    source_path: str = "",
    pixel_mask: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    batch_images: Optional[List[np.ndarray]] = None,
    progress_var_name: Optional[str] = None,
) -> dict:
    """Accumulate correlation across multiple batches on one worker.

    Two modes:
    - batch_images=None: Reconstructs lazy image+filter pipeline locally,
      loads each batch from disk sequentially.
    - batch_images provided: Uses pre-filtered image arrays directly
      (Dask auto-resolves futures before function entry).

    In both modes, the EnsembleCorrelatorCPU's internal buffers accumulate
    across batches (C library's native += behavior). Correlation planes
    are copied out ONCE at the end.
    """
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    # Reconstruct lazy image pipeline only when loading from disk
    images = None
    if batch_images is None:
        from pivtools_core.image_handling.load_images import load_images
        images = load_images(
            camera_num, config, source=Path(source_path),
            batch_size=config.batch_size,
        )
        images = create_filter_pipeline(images, config, pixel_mask)

    # Create ONE correlator for all batches
    correlator = EnsembleCorrelatorCPU(
        config, precomputed_cache=cache, vector_masks=masks,
        active_pass_idx=pass_idx,
    )

    # Set up progress reporting via Dask Variable
    progress_var = None
    if progress_var_name:
        try:
            from distributed import Variable, get_client
            progress_var = Variable(progress_var_name, get_client())
        except Exception:
            pass

    warp_A_total = None
    warp_B_total = None
    n_total = 0
    metadata = {}

    for i, batch_idx in enumerate(batch_indices):
        # Get batch data: from persisted images or lazy pipeline
        if batch_images is not None:
            batch_data = batch_images[i]
        else:
            batch_data = images.blocks[batch_idx].compute(scheduler='synchronous')

        is_first = (batch_idx == 0)
        diag_path = output_path if is_first else None

        # Accumulate into correlator's internal buffers
        lightweight = correlator.correlate_batch_for_accumulation(
            batch_data, config,
            pass_idx=pass_idx,
            predictor_field=predictor_field,
            is_first_batch=is_first,
            save_diagnostics=config.ensemble_save_diagnostics if is_first else False,
            output_path=diag_path,
            clear_buffers=(i == 0),
            copy_result=False,
        )

        # Accumulate warp sums
        if warp_A_total is None:
            warp_A_total = lightweight["warp_A_sum"].copy()
            warp_B_total = lightweight["warp_B_sum"].copy()
        else:
            warp_A_total += lightweight["warp_A_sum"]
            warp_B_total += lightweight["warp_B_sum"]

        n_total += lightweight["n_images"]

        # Capture metadata from first batch that has it
        for key in ["smoothed_predictor", "padded_predictor", "vector_mask",
                     "n_pre", "n_post", "first_pair_A", "first_pair_B"]:
            if metadata.get(key) is None and lightweight.get(key) is not None:
                metadata[key] = lightweight[key]

        del batch_data, lightweight

        if progress_var is not None:
            try:
                progress_var.set(i + 1)
            except Exception:
                pass

    # Copy accumulated correlation buffers ONCE
    result = correlator.get_accumulated_correlation(pass_idx)
    result["warp_A_sum"] = warp_A_total
    result["warp_B_sum"] = warp_B_total
    result["n_images"] = n_total
    result["n_win_x"] = len(correlator.win_ctrs_x[pass_idx])
    result["n_win_y"] = len(correlator.win_ctrs_y[pass_idx])
    result.update(metadata)

    return result
