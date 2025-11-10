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
        logging.info("Empty chunk detected, skipping")
        return chunk
    frame1_min = chunk[:, 0, :, :].min(axis=0)
    frame2_min = chunk[:, 1, :, :].min(axis=0)
    chunk[:, 0, :, :] -= frame1_min
    chunk[:, 1, :, :] -= frame2_min
    return chunk


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


def _pod_filter_block(block):
    """
    Apply POD filtering to a block of images.
    
    For each frame (frame1 and frame2 separately):
    - Reshapes images to vectors (N, H*W)
    - Computes covariance matrix C = M @ M.T
    - Performs SVD: C = PSI @ S @ PSI.T
    - Identifies signal modes using automatic thresholding
    - Reconstructs and subtracts signal modes from original images
    
    Args:
        block: numpy array of shape (N, 2, H, W)
    
    Returns:
        numpy array of same shape, filtered (signal removed, noise retained)
    """
    N, _, H, W = block.shape
    M1 = block[:, 0].reshape(N, -1).astype(np.float32)
    M2 = block[:, 1].reshape(N, -1).astype(np.float32)

    C1 = M1 @ M1.T
    C2 = M2 @ M2.T
    PSI1, S1, _ = np.linalg.svd(C1, full_matrices=False)
    PSI2, S2, _ = np.linalg.svd(C2, full_matrices=False)

    eps_auto_psi = 0.01
    eps_auto_sigma = 0.01

    def _find_auto_mode(PSI, eigvals):
        """
        Find the first mode that meets noise criteria.
        Returns the number of signal modes to keep (modes before the noise mode).
        If no noise mode is found, returns 0 (no filtering applied).
        """
        for i in range(N - 1):
            mean_psi = np.abs(np.mean(PSI[:, i]))
            sig_diff = np.abs(eigvals[i] - eigvals[i + 1]) / eigvals[N // 2]
            if mean_psi < eps_auto_psi and sig_diff < eps_auto_sigma * eigvals[0]:
                # Found noise mode at index i, so keep modes 0 to i-1
                return i
        # No noise mode found, don't filter (return 0)
        return 0

    N1 = _find_auto_mode(PSI1, S1)
    N2 = _find_auto_mode(PSI2, S2)

    def _evaluate_phi_tcoeff(M, PSI, N_auto):
        """
        Compute spatial modes (PHI) and temporal coefficients (TCoeff) for POD.
        
        Args:
            M: Data matrix (N_images, N_pixels)
            PSI: Eigenvectors from SVD of covariance matrix
            N_auto: Number of modes to compute
            
        Returns:
            PHI: List of spatial modes (normalized)
            TC: List of temporal coefficients for each mode
        """
        PHI = []
        TC = []
        for i in range(N_auto):
            phi = M.T @ PSI[:, i]
            phi /= np.linalg.norm(phi)
            PHI.append(phi)
            TC.append(M @ phi)
        return PHI, TC

    PHI1, TC1 = _evaluate_phi_tcoeff(M1, PSI1, N1)
    PHI2, TC2 = _evaluate_phi_tcoeff(M2, PSI2, N2)

    F1 = M1.copy()
    F2 = M2.copy()
    for j in range(N1):
        F1 -= np.outer(TC1[j], PHI1[j])
    for j in range(N2):
        F2 -= np.outer(TC2[j], PHI2[j])

    filtered = np.stack([F1.reshape(N, H, W), F2.reshape(N, H, W)], axis=1)

    return filtered.astype(block.dtype)


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


def _transpose_block(block):
    return block.transpose(0, 1, 3, 2)


def transpose_filter(images: da.Array) -> da.Array:

    if images.ndim != 4:
        raise ValueError(f"Expected 4D array (N, C, H, W), got {images.ndim}D array.")

    return images.map_blocks(_transpose_block, dtype=images.dtype)


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
    "transpose": transpose_filter,
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
