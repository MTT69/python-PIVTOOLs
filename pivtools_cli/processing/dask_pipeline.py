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
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
from dask.distributed import Client
from scipy.io import savemat

from pivtools_cli.piv.piv_backend.factory import make_correlator_backend
from pivtools_core.config import Config

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
    savemat(
        save_dir / f"{filename_prefix}_A.mat", {"frame": frame_A}, do_compression=True
    )
    savemat(
        save_dir / f"{filename_prefix}_B.mat", {"frame": frame_B}, do_compression=True
    )

    logger.debug(f"Saved intermediate frames to {save_dir / filename_prefix}_*.mat")


# =============================================================================
# FILTER HELPERS
# =============================================================================

TEMPORAL_FILTERS = {"time", "pod"}


def get_filter_specs(config: Config) -> List[dict]:
    """
    Get the ordered list of filter specifications from config.

    Returns the filters in the exact order the user defined them,
    preserving interleaved spatial/temporal ordering.

    Returns:
        List of filter spec dicts (e.g., [{'type': 'gaussian', 'sigma': 1.0}, {'type': 'pod'}])
    """
    return config.filters or []


# =============================================================================
# DASK PIPELINE FUNCTIONS
# =============================================================================


def apply_all_filters_slim(
    block: np.ndarray,
    filter_specs: Optional[List[dict]] = None,
    pixel_mask: Optional[np.ndarray] = None,
    save_intermediate_base: Optional[str] = None,
    num_frame_pairs: Optional[int] = None,
    block_id: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """
    Unified filter function for map_blocks (slim version).

    This version takes filter specs directly instead of the full config object,
    avoiding repeated serialization of the entire config for every chunk.

    Applies all configured filters in the user-defined order:
    1. Pixel mask (zero masked regions)
    2. Filters in order (spatial and temporal interleaved as configured)

    This function is called by dask.array.map_blocks on each chunk.
    The chunk is already computed when it reaches this function.

    Args:
        block: Image batch of shape (N, 2, H, W)
        filter_specs: Ordered list of all filter specifications (spatial and temporal
            interleaved in the user's chosen order)
        pixel_mask: Boolean mask (H, W) where True = masked (optional)
        save_intermediate_base: Base path for saving intermediate outputs (optional)
            If provided, saves frames to {base}/basic_filters/{num_frame_pairs}/{batch_no}/
        num_frame_pairs: Number of frame pairs (for path construction)
        block_id: Block ID from map_blocks (automatically populated by dask)

    Returns:
        Filtered block of same shape
    """
    from pivtools_cli.preprocessing.pod_filter import (
        pod_filter_batch,
        time_filter_batch,
    )

    if filter_specs is None:
        filter_specs = []

    # Validate input
    if block.ndim != 4:
        logger.warning(f"apply_all_filters_slim: Expected 4D block, got {block.ndim}D")
        return block

    N, C, H, W = block.shape
    logger.debug(f"Applying filters to block: shape={block.shape}")

    # Determine if we should save intermediate outputs
    save_intermediate = (
        save_intermediate_base is not None
        and num_frame_pairs is not None
        and block_id is not None
        and (filter_specs or pixel_mask is not None)
    )

    if save_intermediate:
        batch_no = block_id[0]  # First dimension is the batch index
        save_dir = (
            Path(save_intermediate_base)
            / "basic_filters"
            / str(num_frame_pairs)
            / f"batch_{batch_no:04d}"
        )
        logger.debug(f"Saving intermediate filter outputs to {save_dir}")

    # Single copy at start - all subsequent operations modify in-place
    # This avoids multiple copies if both mask and filters are applied
    needs_copy = pixel_mask is not None or filter_specs
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
                _save_intermediate_frame(
                    block, save_dir, f"{filter_idx:02d}_after_pixel_mask"
                )
                filter_idx += 1
        else:
            logger.warning(
                f"Pixel mask shape {pixel_mask.shape} != image shape ({H}, {W})"
            )

    # 2. Apply filters in user-defined order (spatial and temporal interleaved)
    for spec in filter_specs:
        filter_type = spec.get("type")

        if filter_type in TEMPORAL_FILTERS:
            # Temporal filter (needs full batch)
            if filter_type == "pod":
                eps_auto_psi = spec.get("eps_auto_psi", 0.01)
                eps_auto_sigma = spec.get("eps_auto_sigma", 0.01)
                block = pod_filter_batch(
                    block,
                    eps_auto_psi=eps_auto_psi,
                    eps_auto_sigma=eps_auto_sigma,
                )
            elif filter_type == "time":
                block = time_filter_batch(block)
        else:
            # Spatial filter (element-wise)
            block = _apply_spatial_filters_numpy(block, [spec])

        if save_intermediate:
            _save_intermediate_frame(
                block, save_dir, f"{filter_idx:02d}_after_{filter_type}"
            )
            filter_idx += 1

    return block


def _normalize_kernel_size(size, default: Tuple[int, int] = (7, 7)) -> Tuple[int, int]:
    """Normalize a filter kernel size to an odd-valued 2-tuple.

    Handles all representations that may arrive from JSON, YAML, or config:
    scalar int/float, list, or tuple.  Even values are bumped to the next odd.
    """
    if size is None:
        size = default
    if isinstance(size, (int, float)):
        size = (int(size), int(size))
    elif isinstance(size, list):
        size = tuple(int(s) for s in size)
    # Ensure odd
    return tuple(s + (s + 1) % 2 for s in size)


def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """Build a 1-D Gaussian kernel matching MATLAB's fspecial('gaussian')."""
    half = (size - 1) / 2.0
    x = np.arange(size) - half
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    k /= k.sum()
    return k


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
    from scipy.ndimage import correlate as scipy_correlate
    from scipy.ndimage import maximum_filter as scipy_maximum
    from scipy.ndimage import median_filter as scipy_median
    from scipy.ndimage import minimum_filter as scipy_minimum
    from scipy.ndimage import uniform_filter as scipy_uniform

    # In-place arithmetic below requires floating-point dtype
    if not np.issubdtype(block.dtype, np.floating):
        block = block.astype(np.float32)

    for spec in filter_specs:
        filter_type = spec.get("type")

        if filter_type == "gaussian":
            # FIR Gaussian kernel matching MATLAB fspecial('gaussian', size, sigma).
            # Uses explicit kernel + correlation (not scipy IIR gaussian_filter).
            size = _normalize_kernel_size(spec.get("size"), default=(7, 7))
            sigma = spec.get("sigma", 1.0)
            ky = _gaussian_kernel_1d(size[0], sigma)
            kx = _gaussian_kernel_1d(size[1], sigma)
            kernel_2d = np.outer(ky, kx).astype(np.float32)
            for i in range(block.shape[0]):
                for j in range(block.shape[1]):
                    block[i, j] = scipy_correlate(
                        block[i, j], kernel_2d, mode="constant"
                    )

        elif filter_type == "median":
            size = _normalize_kernel_size(spec.get("size"), default=(5, 5))
            block = scipy_median(block, size=(1, 1) + size)

        elif filter_type == "norm":
            size = _normalize_kernel_size(spec.get("size"), default=(7, 7))
            max_gain = spec.get("max_gain", 1.0)
            gain_floor = 1.0 / max_gain
            # Per-frame to avoid 3x block-size memory allocation
            for i in range(block.shape[0]):
                for j in range(block.shape[1]):
                    frame = block[i, j]
                    local_min = scipy_minimum(frame, size=size)
                    local_range = scipy_maximum(frame, size=size)
                    local_range -= local_min
                    np.maximum(local_range, gain_floor, out=local_range)
                    frame -= local_min
                    frame /= local_range

        elif filter_type == "maxnorm":
            # Background normalization: divides by a smoothed local minimum
            # (local background level), with a maximum gain limit. Equalizes
            # illumination gradients while preserving particle contrast.
            # Name kept as 'maxnorm' for config backward compatibility.
            # Matches MATLAB filter_maxnorm (minmaxfiltnd single-output = min).
            size = _normalize_kernel_size(spec.get("size"), default=(7, 7))
            max_gain = spec.get("max_gain", 1.0)
            gain_floor = 1.0 / max_gain
            for i in range(block.shape[0]):
                for j in range(block.shape[1]):
                    frame = block[i, j]
                    local_min = scipy_minimum(frame, size=size, mode="nearest")
                    scipy_uniform(
                        local_min, size=size, output=local_min, mode="constant"
                    )
                    np.maximum(local_min, gain_floor, out=local_min)
                    np.maximum(frame, 0, out=frame)
                    frame /= local_min

        elif filter_type == "norm2":
            # Smoothed range normalization: like 'norm' but box-smooths both
            # the min and max envelopes before subtracting/dividing. Gives a
            # more stable normalization that's less sensitive to single-pixel
            # noise spikes. Matches MATLAB filter_norm2.
            size = _normalize_kernel_size(spec.get("size"), default=(7, 7))
            max_gain = spec.get("max_gain", 1.0)
            gain_floor = 1.0 / max_gain
            for i in range(block.shape[0]):
                for j in range(block.shape[1]):
                    frame = block[i, j]
                    local_min = scipy_minimum(frame, size=size, mode="nearest")
                    local_max = scipy_maximum(frame, size=size, mode="nearest")
                    scipy_uniform(
                        local_min, size=size, output=local_min, mode="constant"
                    )
                    scipy_uniform(
                        local_max, size=size, output=local_max, mode="constant"
                    )
                    local_max -= local_min
                    np.maximum(local_max, gain_floor, out=local_max)
                    frame -= local_min
                    frame /= local_max

        elif filter_type == "ssmin":
            # Sliding minimum background subtraction: median-smooths the
            # image first (removes noise), then extracts the local minimum
            # (background envelope), box-smooths it, and subtracts. Output
            # is clipped to >= 0. Matches MATLAB filter_ssmin.
            size = _normalize_kernel_size(spec.get("size"), default=(7, 7))
            for i in range(block.shape[0]):
                for j in range(block.shape[1]):
                    frame = block[i, j]
                    bg = scipy_median(frame, size=(3, 3), mode="constant")
                    bg = scipy_minimum(bg, size=size, mode="nearest")
                    scipy_uniform(bg, size=size, output=bg, mode="constant")
                    frame -= bg
                    np.maximum(frame, 0, out=frame)

        elif filter_type == "meannorm":
            # Per-frame mean normalization: divide every frame by its own
            # spatial mean intensity, equalizing pair-to-pair brightness
            # (laser energy drift). Kills the brightness-covariance pedestal
            # that ensemble background subtraction cannot remove. Global gain
            # only — spatial illumination variation within a frame is the job
            # of norm/norm2/maxnorm.
            for i in range(block.shape[0]):
                for j in range(block.shape[1]):
                    frame = block[i, j]
                    frame_mean = float(frame.mean())
                    if frame_mean <= 0:
                        raise ValueError(
                            f"meannorm filter: frame ({i},{j}) has non-positive "
                            f"mean intensity {frame_mean} — cannot normalize."
                        )
                    frame /= frame_mean

        elif filter_type == "lmax":
            size = _normalize_kernel_size(spec.get("size"), default=(7, 7))
            block = scipy_maximum(block, size=(1, 1) + size)

        elif filter_type == "invert":
            img_max = block.max(axis=(-2, -1), keepdims=True)
            np.subtract(img_max, block, out=block)

        elif filter_type == "clahe":
            import cv2

            clip_limit = spec.get("clip_limit", 2.0)
            tile_size = _normalize_kernel_size(
                spec.get("tile_grid_size"), default=(8, 8)
            )
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            for i in range(block.shape[0]):
                for j in range(block.shape[1]):
                    frame = block[i, j]
                    fmin, fmax = frame.min(), frame.max()
                    if fmax > fmin:
                        norm = ((frame - fmin) / (fmax - fmin) * 65535).astype(
                            np.uint16
                        )
                        result = clahe.apply(norm)
                        block[i, j] = (
                            result.astype(np.float32) / 65535.0 * (fmax - fmin) + fmin
                        )

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
    filter_specs = get_filter_specs(config)

    if filter_specs:
        logger.debug(f"  Filters (in order): {[f.get('type') for f in filter_specs]}")
    if pixel_mask is not None:
        logger.debug(f"  Pixel mask: {np.sum(pixel_mask)} masked pixels")
    if save_intermediate_base is not None:
        logger.debug(
            f"  Saving intermediate outputs to: {save_intermediate_base}/basic_filters/..."
        )

    # If no filters and no mask, return unchanged
    if not filter_specs and pixel_mask is None:
        logger.debug("  No filters configured, returning images unchanged")
        return images

    # Prepare intermediate saving parameters
    save_base_str = (
        str(save_intermediate_base) if save_intermediate_base is not None else None
    )
    num_frame_pairs = (
        config.num_frame_pairs if save_intermediate_base is not None else None
    )

    # Apply filters via map_blocks using the slim version
    # This only serializes the filter specs (small dicts), not the full config
    # Use block_id to get the batch number for intermediate saving
    filtered = images.map_blocks(
        apply_all_filters_slim,
        filter_specs=filter_specs,
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
        v.nbytes if hasattr(v, "nbytes") else 0
        for v in correlator_cache.values()
        if hasattr(v, "nbytes")
    )
    logger.info(f"  Scattered correlator cache (~{cache_size / 1024:.1f} KB)")

    # Scatter vector masks if present
    scattered_masks = None
    if vector_masks:
        scattered_masks = client.scatter(vector_masks, broadcast=True)
        mask_size = sum(m.nbytes for m in vector_masks) / 1024
        logger.info(f"  Scattered vector masks ({mask_size:.1f} KB)")

    return {
        "cache": scattered_cache,
        "masks": scattered_masks,
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
    # saved_paths.append(path)

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
    first_pair_A = (
        r1.get("first_pair_A")
        if r1.get("first_pair_A") is not None
        else r2.get("first_pair_A")
    )
    first_pair_B = (
        r1.get("first_pair_B")
        if r1.get("first_pair_B") is not None
        else r2.get("first_pair_B")
    )

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


def reduce_warp_sums(r1: dict, r2: dict) -> dict:
    """Combine two phase-A ('image' background method) warp-sum results."""
    return {
        "warp_A_sum": r1["warp_A_sum"] + r2["warp_A_sum"],
        "warp_B_sum": r1["warp_B_sum"] + r2["warp_B_sum"],
        "n_images": r1["n_images"] + r2["n_images"],
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
        if hasattr(v, "nbytes"):
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
    warp_sums_only: bool = False,
    mean_images: Optional[tuple] = None,
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

    'image' background-method phases (two sweeps over the data):
    - warp_sums_only=True: phase A — only accumulate warped image sums
      (compute_warp_sums_only), no correlation. Returns warp sums + n_images.
    - mean_images=(A_mean, B_mean): phase B — correlate the mean-subtracted
      images (correlate_mean_subtracted_batch). Per-batch planes are summed
      in Python (this path predates the C buffer-accumulation API); warp sums
      in the result are zeros — the driver injects the phase-A global sums.
    """
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    # Reconstruct lazy image pipeline only when loading from disk
    images = None
    if batch_images is None:
        from pivtools_core.image_handling.load_images import load_images

        images = load_images(
            camera_num,
            config,
            source=Path(source_path),
            batch_size=config.batch_size,
        )
        images = create_filter_pipeline(images, config, pixel_mask)

    # Create ONE correlator for all batches
    correlator = EnsembleCorrelatorCPU(
        config,
        precomputed_cache=cache,
        vector_masks=masks,
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
    corr_totals = None
    n_total = 0
    metadata = {}

    for i, batch_idx in enumerate(batch_indices):
        # Get batch data: from persisted images or lazy pipeline
        if batch_images is not None:
            batch_data = batch_images[i]
        else:
            batch_data = images.blocks[batch_idx].compute(scheduler="synchronous")

        is_first = batch_idx == 0
        diag_path = output_path if is_first else None

        if warp_sums_only:
            # Phase A of the 'image' method: warp sums only, no correlation
            lightweight = correlator.compute_warp_sums_only(
                batch_data,
                config,
                pass_idx=pass_idx,
                predictor_field=predictor_field,
            )
        elif mean_images is not None:
            # Phase B of the 'image' method: correlate mean-subtracted images
            A_mean, B_mean = mean_images
            lightweight = correlator.correlate_mean_subtracted_batch(
                batch_data,
                config,
                pass_idx=pass_idx,
                A_mean=A_mean,
                B_mean=B_mean,
                predictor_field=predictor_field,
                save_diagnostics=(
                    config.ensemble_save_diagnostics if is_first else False
                ),
                output_path=diag_path,
                is_first_batch=is_first,
            )
            # Per-batch planes are returned as copies; sum them here in Python
            if corr_totals is None:
                corr_totals = {
                    k: lightweight[k].copy()
                    for k in ("corr_AA_sum", "corr_BB_sum", "corr_AB_sum")
                }
            else:
                for k in ("corr_AA_sum", "corr_BB_sum", "corr_AB_sum"):
                    corr_totals[k] += lightweight[k]
        else:
            # Production path: accumulate into correlator's internal C buffers
            lightweight = correlator.correlate_batch_for_accumulation(
                batch_data,
                config,
                pass_idx=pass_idx,
                predictor_field=predictor_field,
                is_first_batch=is_first,
                save_diagnostics=(
                    config.ensemble_save_diagnostics if is_first else False
                ),
                output_path=diag_path,
                clear_buffers=(i == 0),
                copy_result=False,
            )

        # Accumulate warp sums (absent from the phase-B result)
        if lightweight.get("warp_A_sum") is not None:
            if warp_A_total is None:
                warp_A_total = lightweight["warp_A_sum"].copy()
                warp_B_total = lightweight["warp_B_sum"].copy()
            else:
                warp_A_total += lightweight["warp_A_sum"]
                warp_B_total += lightweight["warp_B_sum"]

        n_total += lightweight["n_images"]

        # Capture metadata from first batch that has it
        for key in [
            "smoothed_predictor",
            "padded_predictor",
            "vector_mask",
            "n_pre",
            "n_post",
            "first_pair_A",
            "first_pair_B",
        ]:
            if metadata.get(key) is None and lightweight.get(key) is not None:
                metadata[key] = lightweight[key]

        del batch_data, lightweight

        if progress_var is not None:
            try:
                progress_var.set(i + 1)
            except Exception:
                pass

    if warp_sums_only:
        return {
            "warp_A_sum": warp_A_total,
            "warp_B_sum": warp_B_total,
            "n_images": n_total,
        }

    if mean_images is not None:
        # Warp sums were consumed in phase A; zeros here keep the reduction
        # shape-consistent, and the driver injects the phase-A global sums.
        A_mean, _ = mean_images
        result = corr_totals
        result["warp_A_sum"] = np.zeros_like(A_mean, dtype=np.float32)
        result["warp_B_sum"] = np.zeros_like(A_mean, dtype=np.float32)
    else:
        # Copy accumulated correlation buffers ONCE
        result = correlator.get_accumulated_correlation(pass_idx)
        result["warp_A_sum"] = warp_A_total
        result["warp_B_sum"] = warp_B_total
    result["n_images"] = n_total
    result["n_win_x"] = len(correlator.win_ctrs_x[pass_idx])
    result["n_win_y"] = len(correlator.win_ctrs_y[pass_idx])
    result.update(metadata)

    return result
