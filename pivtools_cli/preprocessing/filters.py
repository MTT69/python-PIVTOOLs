import logging
import sys
from pathlib import Path

import dask.array as da
import numpy as np
from dask_image.ndfilters import (
    gaussian_filter,
    maximum_filter,
    median_filter,
    minimum_filter,
    uniform_filter,
)


from pivtools_core.config import Config


def time_filter(images: da.Array) -> da.Array:
    """
    Time filter images

    Args:
        images (da.Array): Dask array containing the images.

    Returns:
        da.Array: Filtered Dask array of images.
    """

    processed_images = images.map_blocks(_subtract_local_min, dtype=images.dtype)

    return processed_images


def _subtract_local_min(chunk):

    if chunk.size == 0:
        logging.warning("Time filter: Empty chunk detected, skipping")
        return chunk

    # Validate chunk shape
    if chunk.ndim != 4:
        logging.error(f"Time filter: Expected 4D chunk (N, 2, H, W), got {chunk.ndim}D with shape {chunk.shape}")
        return chunk

    N, C, H, W = chunk.shape

    if C != 2:
        logging.error(f"Time filter: Expected 2 channels, got {C}")
        return chunk

    if N == 0:
        logging.warning(f"Time filter: Batch dimension is 0, cannot compute minimum across empty set")
        return chunk

    # Convert to float32 to avoid uint8 underflow during subtraction
    chunk = chunk.astype(np.float32)

    # Compute minimum across batch dimension (axis=0)
    frame1_min = chunk[:, 0, :, :].min(axis=0)  # Shape: (H, W)
    frame2_min = chunk[:, 1, :, :].min(axis=0)  # Shape: (H, W)

    # Subtract minimum from each frame
    chunk[:, 0, :, :] -= frame1_min
    chunk[:, 1, :, :] -= frame2_min

    # Clip negative values to 0 (negative intensities don't make sense for images)
    chunk = np.maximum(chunk, 0)

    # Return as the input dtype (should be float32 from config)
    return chunk.astype(chunk.dtype)


def pod_filter(images: da.Array) -> da.Array:
    """
    POD filter images using Proper Orthogonal Decomposition (Mendez et al.)
    
    This filter automatically identifies and removes coherent structures (signal modes)
    from image sequences, leaving behind the random fluctuations. The process:
    
    1. Computes covariance matrices for each frame pair
    2. Performs SVD to extract eigenvectors (PSI) and eigenvalues
    3. Automatically identifies the first "noise mode" based on:
       - Mean of eigenvector < eps_auto_psi (0.01)
       - Eigenvalue difference < eps_auto_sigma * max_eigenvalue (0.01)
    4. Removes all signal modes (modes before the noise mode) from the images
    
    Args:
        images (da.Array): Dask array containing the images (N, 2, H, W).

    Returns:
        da.Array: Filtered Dask array of images with signal modes removed.
    """
    processed_images = images.map_blocks(_pod_filter_block, dtype=images.dtype)
    return processed_images


def _pod_filter_block(block, tile_size=2048):
    """
    Memory-efficient POD filtering using spatial tiling.

    This implementation uses the mathematical equivalence:
        TC @ PHI.T = PSI @ PSI.T @ M

    The singular values in TC (= PSI @ Sigma) and PHI (= M.T @ PSI @ Sigma^-1)
    cancel out, allowing direct projection without storing the massive spatial
    mode matrix PHI.

    For each frame (frame1 and frame2 separately):
    1. Computes temporal modes (PSI) via SVD of covariance matrix
    2. Identifies signal modes using automatic thresholding (Mendez et al.)
    3. Applies filtering using spatial tiling to minimize memory usage

    Peak memory is dominated by the initial float32 conversion, not by the
    number of modes removed. This scales safely to high-resolution images.

    Args:
        block: numpy array of shape (N, 2, H, W)
        tile_size: Size of spatial tiles for memory-efficient processing (default 2048)

    Returns:
        numpy array of same shape, filtered (signal removed, noise retained)
    """
    import gc

    N, C, H, W = block.shape
    # POD filtering produces negative values, so output must be float32
    output = np.empty(block.shape, dtype=np.float32)

    # Process each channel (frame 0/1) sequentially to halve memory requirements
    for frame_idx in range(C):
        logging.debug(f"POD filter: Processing frame {frame_idx}...")

        # Step 1: Compute Temporal Modes (PSI) on full resolution
        # Reshape to (N, Pixels) and promote to float32
        M_full = block[:, frame_idx].reshape(N, -1).astype(np.float32)

        # Compute covariance matrix (N x N) - this is small and fast
        Cov = M_full @ M_full.T
        PSI, S, _ = np.linalg.svd(Cov, full_matrices=False)

        # Identify noise floor using Mendez et al. criteria
        n_remove = _find_pod_auto_mode(PSI, S, N)

        logging.debug(f"POD filter: Frame {frame_idx} - removing {n_remove} modes")

        if n_remove == 0:
            output[:, frame_idx] = block[:, frame_idx]
            del M_full, Cov, S, PSI
            gc.collect()
            continue

        # Isolate the signal modes to remove (copy before deleting PSI)
        PSI_bad = PSI[:, :n_remove].copy()

        # Free memory - we don't need M_full or full PSI anymore
        del M_full, Cov, S, PSI
        gc.collect()

        # Step 2: Apply filter using spatial tiling
        # The projection PSI @ PSI.T @ M is equivalent to TC @ PHI.T
        # but avoids storing the massive PHI matrix

        for y in range(0, H, tile_size):
            y_end = min(y + tile_size, H)
            for x in range(0, W, tile_size):
                x_end = min(x + tile_size, W)

                # Extract tile from original block
                tile = block[:, frame_idx, y:y_end, x:x_end]

                # Reshape to matrix (N, P_tile) and promote to float32
                M_tile = tile.reshape(N, -1).astype(np.float32)

                # Project onto bad modes and subtract: M - PSI @ PSI.T @ M
                # This is mathematically equivalent to the full POD reconstruction
                t_coeffs = PSI_bad.T @ M_tile  # (n_remove, P_tile)
                bad_signal = PSI_bad @ t_coeffs  # (N, P_tile)
                M_tile -= bad_signal

                # Store result
                h_t, w_t = y_end - y, x_end - x
                output[:, frame_idx, y:y_end, x:x_end] = M_tile.reshape(N, h_t, w_t)

                del M_tile, t_coeffs, bad_signal

        del PSI_bad
        gc.collect()

    # Return float32 - POD output has negative values that would be clipped by uint8
    return output


def _find_pod_auto_mode(PSI, eigvals, N):
    """
    Find the first mode that meets noise criteria (Mendez et al.).

    Returns the number of signal modes to remove (modes before the noise mode).
    If no noise mode is found, returns 0 (no filtering applied).

    The criteria for identifying a noise mode:
    - Mean of eigenvector < eps_auto_psi (0.01)
    - Eigenvalue difference < eps_auto_sigma * max_eigenvalue (0.01)

    Args:
        PSI: Eigenvectors from SVD of covariance matrix (N, N)
        eigvals: Eigenvalues from SVD (N,)
        N: Number of snapshots

    Returns:
        int: Number of signal modes to remove
    """
    eps_auto_psi = 0.01
    eps_auto_sigma = 0.01

    # Protect against division by zero
    norm_factor = eigvals[N // 2] if eigvals[N // 2] > 1e-10 else 1.0

    for i in range(N - 1):
        mean_psi = np.abs(np.mean(PSI[:, i]))
        sig_diff = np.abs(eigvals[i] - eigvals[i + 1]) / norm_factor

        if mean_psi < eps_auto_psi and sig_diff < eps_auto_sigma * eigvals[0]:
            # Found noise mode at index i, so remove modes 0 to i-1
            return i

    # No noise mode found, don't filter
    return 0


def clip_filter(images: da.Array, threshold=None, n=2.0) -> da.Array:
    """Clip images to a specified threshold or use a median-based threshold.

    Args:
        images (da.Array): Dask array of shape (N, 2, H, W).
        threshold (tuple[float, float], optional): Clipping threshold. Defaults to None.
        n (float, optional): Number of standard deviations for upper limit if threshold is None. Defaults to 2.0.

    Returns:
        da.Array: Clipped images, same shape & chunking as input.
    """
    if threshold is not None:
        lower, upper = threshold
        return da.clip(images, lower, upper)
    else:
        med = da.median(images, axis=(2, 3), keepdims=True)
        std = da.std(images, axis=(2, 3), keepdims=True)
        upper = med + n * std
        lower = da.zeros_like(upper)
        return da.clip(images, lower, upper)


def invert_filter(images: da.Array, offset: float = 0) -> da.Array:
    """
    Invert images per-frame using Dask, with a scalar offset.

    Args:
        images (da.Array): Dask array of shape (N,2,H,W)
        offset (float): scalar offset to subtract the image from

    Returns:
        da.Array: inverted images, same shape & chunking as input
    """
    return offset - images


def levelize_filter(images: da.Array, white: da.Array = None) -> da.Array:
    """
    Levelize images by dividing by a 'white' reference image.
    If white is None, returns the images unchanged.

    Args:
        images (da.Array): Dask array of shape (N,2,H,W)
        white (da.Array or None): white image, shape (H,W)

    Returns:
        da.Array: Levelized images
    """
    if white is None:
        return images

    return images / white


def lmax_filter(images: da.Array, size=(7, 7)) -> da.Array:
    """
    Apply a local maximum filter on a Dask array of images.

    Args:
        images (da.Array): Dask array of shape (N,2,H,W)
        size (tuple): Kernel size (height, width)

    Returns:
        da.Array: Filtered images
    """

    size = tuple(s + (s + 1) % 2 for s in size)

    return maximum_filter(images, size=(1, 1) + size)


def maxnorm_filter(images: da.Array, size=(7, 7), max_gain=1.0) -> da.Array:
    """
    Normalize images by local max-min contrast with smoothing and max gain limit.

    Args:
        images (da.Array): Dask array of shape (N,2,H,W)
        size (tuple): Kernel size (height, width)
        max_gain (float): Maximum allowed normalization gain

    Returns:
        da.Array: Filtered images
    """
    size = tuple(s + (s + 1) % 2 for s in size)
    spatial_size = (1, 1) + size

    images_float = images.astype("float32")

    local_max = maximum_filter(images_float, size=spatial_size)
    local_min = minimum_filter(images_float, size=spatial_size)
    contrast = local_max - local_min
    smoothed = uniform_filter(contrast, size=spatial_size)

    denom = da.maximum(smoothed, 1.0 / max_gain)
    normalized = da.maximum(images_float, 0) / denom

    return normalized.astype(images.dtype)


def median_filter_dask(images: da.Array, size=(5, 5)) -> da.Array:
    """
    Apply a median filter to a batch of images with shape (N, 2, H, W).

    Args:
        images (da.Array): Dask array of shape (N, 2, H, W).
        size (tuple): Kernel size (height, width). Default (5, 5).

    Returns:
        da.Array: Median-filtered images with the same shape.
    """

    return median_filter(images, size=(1, 1) + size)


def norm_filter(images: da.Array, size=(7, 7), max_gain=1.0) -> da.Array:
    """
    Normalize an image by subtracting a sliding minimum and dividing by a
    sliding maximum-minimum, subject to a maximum gain.

    Args:
        images (da.Array): Dask array of shape (N, C, H, W).
        size (tuple): Kernel size (height, width). Default (7, 7).
        max_gain (float): Maximum normalization gain. Default 1.0.

    Returns:
        da.Array: Normalized Dask array of images.
    """

    size = tuple(s + (s + 1) % 2 for s in size)

    spatial_size = (1, 1) + size

    images_float = images.astype("float32")

    local_min = minimum_filter(images_float, size=spatial_size)
    local_max = maximum_filter(images_float, size=spatial_size)

    denom = da.maximum(local_max - local_min, 1.0 / max_gain)
    normalized = (images_float - local_min) / denom

    return normalized.astype(images.dtype)


def sbg_filter(images: da.Array, bg=None) -> da.Array:
    """
    Subtract a background image from each input image and clip at zero.

    Args:
        images (da.Array): Dask array of shape (N, 2, H, W).
        bg (np.ndarray or da.Array or None): Background image to subtract.
            If None, defaults to zeros (no effect).
            Must be broadcastable to (N, 2, H, W).

    Returns:
        da.Array: Background-subtracted and clipped images.
    """
    if bg is None:
        bg = 0

    return da.maximum(0, images - bg)


def gaussian_filter_dask(images: da.Array, sigma=1.0) -> da.Array:
    """
    Apply a Gaussian filter to a batch of images with shape (N, 2, H, W).

    Args:
        images (da.Array): Dask array of shape (N, 2, H, W).
        sigma (float or tuple): Standard deviation for Gaussian kernel.

    Returns:
        da.Array: Gaussian-filtered images with the same shape.
    """
    return gaussian_filter(images, sigma=(0, 0, sigma, sigma))


def pixel_mask_filter(images: da.Array, mask: np.ndarray = None) -> da.Array:
    """
    Apply pixel mask to images by setting masked regions to zero intensity.

    This filter should be applied before PIV processing to zero out regions
    that should not contribute to correlation (e.g., boundaries, obstructions).

    Args:
        images (da.Array): Dask array of shape (N, 2, H, W).
        mask (np.ndarray): Boolean mask of shape (H, W) where True indicates
            regions to mask (set to zero). If None, returns images unchanged.

    Returns:
        da.Array: Images with masked regions set to zero intensity.
    """
    if mask is None:
        return images

    # Ensure mask is boolean
    mask = np.asarray(mask, dtype=bool)

    # Apply mask using map_blocks for efficiency
    def _apply_pixel_mask(block, mask):
        # block shape: (N, 2, H, W)
        # mask shape: (H, W)
        # Set masked pixels to 0
        result = block.copy()
        result[:, :, mask] = 0
        return result

    return images.map_blocks(
        _apply_pixel_mask,
        mask=mask,
        dtype=images.dtype
    )


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


FILTER_MAP = {
    "time": time_filter,
    "pod": pod_filter,
    "clip": clip_filter,
    "invert": invert_filter,
    "levelize": levelize_filter,
    "lmax": lmax_filter,
    "maxnorm": maxnorm_filter,
    "median": median_filter_dask,
    "sbg": sbg_filter,
    "norm": norm_filter,
    "gaussian": gaussian_filter_dask,
}

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


def filter_images(images: da.Array, config: Config) -> da.Array:
    """
    Apply a sequence of filters defined in the config.

    Args:
        images: Dask array of shape (N, C, H, W)
        preprocessing_config: dict with key 'filters', a list of filter dicts
    """
    for filt in config.filters:
        logging.info("Applying filter: %s", filt)
        ftype = filt.get("type")
        if ftype not in FILTER_MAP:
            raise ValueError(f"Unknown filter type: {ftype}")

        func = FILTER_MAP[ftype]
        kwargs = {k: v for k, v in filt.items() if k != "type"}
        
        # Convert list parameters to tuples (for size, threshold, etc.)
        for key in ['size', 'threshold']:
            if key in kwargs and isinstance(kwargs[key], list):
                kwargs[key] = tuple(kwargs[key])
        
        images = func(images, **kwargs)

    return images
