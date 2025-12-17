"""
POD Filter - Simplified Implementation for Dask-Centric Pipeline

Proper Orthogonal Decomposition (POD) filtering for PIV image preprocessing.
Removes background modes from image batches based on automatic mode selection.

Key design choices:
- Process A channel then B channel sequentially to minimize peak memory
- Use covariance method: C = M @ M.T (N x N matrix, small!)
- In-place subtraction to avoid copying large arrays
- Images stored as float32, SVD computed in float64 for numerical stability
"""

import gc
import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def find_auto_mode(
    PSI: np.ndarray,
    eigvals: np.ndarray,
    n_images: int,
    eps_auto_psi: float = 0.01,
    eps_auto_sigma: float = 0.01,
) -> int:
    """
    Automatic mode selection based on eigenvector mean and eigenvalue difference.

    Finds the first mode that meets noise criteria (Mendez et al.):
    - Mean of eigenvector < eps_auto_psi
    - Eigenvalue difference < eps_auto_sigma * max_eigenvalue

    Parameters
    ----------
    PSI : np.ndarray
        Eigenvector matrix from SVD, shape (n_images, n_images)
    eigvals : np.ndarray
        Eigenvalues (singular values from SVD of covariance matrix)
    n_images : int
        Number of images in the batch
    eps_auto_psi : float
        Threshold for mean eigenvector criterion (default 0.01)
    eps_auto_sigma : float
        Threshold for eigenvalue difference criterion (default 0.01)

    Returns
    -------
    int
        Number of signal modes to remove (modes before noise floor)
    """
    # Protect against division by zero
    mid_idx = n_images // 2
    norm_factor = eigvals[mid_idx] if eigvals[mid_idx] > 1e-10 else 1.0

    # Handle edge case of very small first eigenvalue
    if eigvals[0] < 1e-10:
        return 0

    for i in range(n_images - 1):
        mean_psi = np.abs(np.mean(PSI[:, i]))
        sig_diff = np.abs(eigvals[i] - eigvals[i + 1]) / norm_factor

        if mean_psi < eps_auto_psi and sig_diff < eps_auto_sigma * eigvals[0]:
            return i

    return 0


def pod_filter_single_channel(
    images: np.ndarray,
    eps_auto_psi: float = 0.01,
    eps_auto_sigma: float = 0.01,
    verbose: bool = True,
) -> np.ndarray:
    """
    POD filter for a single channel of images.

    Memory-optimized implementation using the covariance method:
    - Compute C = M @ M.T (N x N matrix, small regardless of image size!)
    - SVD gives temporal modes (PSI) and energy (singular values)
    - Find noise floor using auto-thresholding
    - Subtract signal modes in-place

    Algorithm:
    1. M = images.reshape(N, n_pixels) - working matrix
    2. C = M @ M.T - covariance matrix (N x N)
    3. PSI, S, _ = np.linalg.svd(C) - temporal modes and energies
    4. n_remove = find_auto_mode(PSI, S) - find noise floor
    5. For each mode i in range(n_remove):
         phi = M.T @ PSI[:, i]  # Spatial mode (computed on-the-fly)
         phi /= np.linalg.norm(phi)  # Normalize
         tcoeff = M @ phi  # Temporal coefficients
         np.subtract(M, np.outer(tcoeff, phi), out=M)  # In-place subtraction!
    6. Return M.reshape(N, H, W)

    Parameters
    ----------
    images : np.ndarray
        Stack of images, shape (N, H, W), dtype typically float32
    eps_auto_psi : float
        Threshold for mean eigenvector criterion (default 0.01)
    eps_auto_sigma : float
        Threshold for eigenvalue difference criterion (default 0.01)
    verbose : bool
        Print progress information

    Returns
    -------
    np.ndarray
        Filtered images of same shape and dtype as input
    """
    n_images, height, width = images.shape
    n_pixels = height * width
    original_dtype = images.dtype

    # Reshape to (N, pixels) - this is our working matrix
    # Convert to float64 for numerical stability in SVD
    M = images.reshape(n_images, n_pixels).astype(np.float64)

    # Compute covariance matrix C = M @ M.T
    # This is N x N (small!) regardless of image resolution
    C = M @ M.T

    # SVD of covariance matrix
    # PSI contains temporal modes (eigenvectors)
    # singular_values are related to eigenvalues
    PSI, singular_values, _ = np.linalg.svd(C, full_matrices=False)

    # Free covariance matrix
    del C

    # Find automatic mode threshold (noise floor)
    n_remove = find_auto_mode(
        PSI, singular_values, n_images, eps_auto_psi, eps_auto_sigma
    )

    if verbose:
        logger.info(f"  POD: {n_images} images, removing {n_remove} signal modes")

    if n_remove == 0:
        # No modes to remove - convert back and return
        del PSI, singular_values
        return images

    # Process one mode at a time to minimize memory
    # Never store all spatial modes (PHI) simultaneously
    for mode_i in range(n_remove):
        # Compute spatial mode for this temporal mode: phi = M.T @ PSI[:, mode_i]
        psi_i = PSI[:, mode_i]
        phi = M.T @ psi_i

        # Normalize spatial mode
        norm = np.linalg.norm(phi)
        if norm > 1e-10:
            phi /= norm

        # Compute temporal coefficients
        tcoeff = M @ phi

        # Subtract mode IN-PLACE - critical for memory efficiency
        # This avoids creating a copy of the entire M matrix
        np.subtract(M, np.outer(tcoeff, phi), out=M)

        # phi and tcoeff are overwritten on next iteration

    # Free eigenvector matrix
    del PSI, singular_values
    gc.collect()

    # Reshape back and convert to original dtype
    filtered = M.reshape(n_images, height, width).astype(original_dtype)

    del M
    gc.collect()

    return filtered


def pod_filter_batch(
    batch: np.ndarray,
    config=None,
    eps_auto_psi: float = 0.01,
    eps_auto_sigma: float = 0.01,
    verbose: bool = True,
) -> np.ndarray:
    """
    Apply POD filter to a batch of image pairs.

    Processes A channel first, then B channel SEQUENTIALLY to minimize
    peak memory usage. Only one channel's working matrix is in memory
    at a time.

    Parameters
    ----------
    batch : np.ndarray
        Stack of image pairs, shape (N, 2, H, W)
        - N: number of image pairs
        - 2: channels (A=0, B=1)
        - H, W: image dimensions
    config : Config, optional
        Configuration object (can extract eps values if present)
    eps_auto_psi : float
        Threshold for mean eigenvector criterion
    eps_auto_sigma : float
        Threshold for eigenvalue difference criterion
    verbose : bool
        Print progress information

    Returns
    -------
    np.ndarray
        Filtered batch of same shape
    """
    # Extract parameters from config if available
    if config is not None:
        # Check if config has POD-specific parameters
        filters = getattr(config, 'filters', []) or []
        for f in filters:
            if isinstance(f, dict) and f.get('type') == 'pod':
                eps_auto_psi = f.get('eps_auto_psi', eps_auto_psi)
                eps_auto_sigma = f.get('eps_auto_sigma', eps_auto_sigma)
                break

    n_pairs = batch.shape[0]

    if verbose:
        logger.info(f"POD Filter: Processing {n_pairs} image pairs")

    # Process A channel (index 0)
    if verbose:
        logger.info("  Processing channel A...")
    batch[:, 0] = pod_filter_single_channel(
        batch[:, 0], eps_auto_psi, eps_auto_sigma, verbose=verbose
    )

    # Force garbage collection between channels
    gc.collect()

    # Process B channel (index 1)
    if verbose:
        logger.info("  Processing channel B...")
    batch[:, 1] = pod_filter_single_channel(
        batch[:, 1], eps_auto_psi, eps_auto_sigma, verbose=verbose
    )

    gc.collect()

    if verbose:
        logger.info("POD Filter: Complete")

    return batch


def time_filter_batch(batch: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    Time filter: subtract per-pixel minimum across the temporal batch.

    This removes static background by computing the minimum intensity
    at each pixel across all frames and subtracting it.

    Parameters
    ----------
    batch : np.ndarray
        Stack of image pairs, shape (N, 2, H, W)
    verbose : bool
        Print progress information

    Returns
    -------
    np.ndarray
        Filtered batch of same shape
    """
    if verbose:
        logger.info(f"Time Filter: Processing {batch.shape[0]} image pairs")

    # Process each channel
    for channel in range(batch.shape[1]):
        # Compute minimum across temporal dimension (axis 0)
        min_vals = batch[:, channel].min(axis=0, keepdims=True)
        # Subtract in-place
        batch[:, channel] -= min_vals

    if verbose:
        logger.info("Time Filter: Complete")

    return batch


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing POD filter with synthetic images...")

    n_images = 20
    height, width = 100, 100

    # Generate synthetic images with common background + noise
    np.random.seed(42)
    background = np.random.rand(height, width).astype(np.float32) * 100

    # Create batch with A and B channels
    batch = np.zeros((n_images, 2, height, width), dtype=np.float32)
    for i in range(n_images):
        batch[i, 0] = background + np.random.rand(height, width).astype(np.float32) * 10
        batch[i, 1] = background + np.random.rand(height, width).astype(np.float32) * 10

    print(f"Input batch shape: {batch.shape}")
    print(f"Input batch dtype: {batch.dtype}")

    # Test POD filter
    filtered = pod_filter_batch(batch.copy())

    print(f"Output batch shape: {filtered.shape}")
    print(f"Output batch dtype: {filtered.dtype}")

    # Test time filter
    time_filtered = time_filter_batch(batch.copy())
    print(f"Time-filtered batch shape: {time_filtered.shape}")

    print("\nTest complete!")
