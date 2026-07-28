"""
POD Filter - Simplified Implementation for Dask-Centric Pipeline

Proper Orthogonal Decomposition (POD) filtering for PIV image preprocessing.
Removes background modes from image batches based on automatic mode selection.

Key design choices:
- Process A channel then B channel sequentially to minimize peak memory
- Use covariance method: C = M @ M.T (N x N matrix, small!)
- Data GEMMs (covariance accumulation, mode projection) run in float32;
  the N x N SVD stays float64 for numerical stability
- Mode removal is a single temporal-side projection
  M -= PSI_k @ (PSI_k.T @ M), not sequential rank-1 deflation — identical
  in exact arithmetic, without N x n_pixels temporaries
- No gc.collect() here: this code runs on Dask workers, where collection
  is forbidden (see CLAUDE.md gotcha — GC only on the client)

Precision validation (2026-07-28, real Cam4 data, 65 batches x 2 channels):
float32 vs float64 selected identical mode counts on 130/130 batch-channels
(K spanning 2-11); max pixel difference 3.7e-4 of full scale.
"""

import logging

import numpy as np

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
        Number of modes to remove (matches MATLAB N_auto convention)
    """
    # Protect against division by zero
    # MATLAB round() rounds 0.5 away from zero; Python round() uses
    # banker's rounding (to nearest even). Use math.ceil to match MATLAB
    # for both even and odd N: ceil(N/2) == round(N/2) in MATLAB.
    mid_idx = -(-n_images // 2) - 1  # ceil(n_images/2), then 0-based
    norm_factor = eigvals[mid_idx] if eigvals[mid_idx] > 1e-10 else 1.0

    # Handle edge case of very small first eigenvalue
    if eigvals[0] < 1e-10:
        return 0

    for i in range(n_images - 1):
        mean_psi = np.abs(np.mean(PSI[:, i]))
        sig_diff = np.abs(eigvals[i] - eigvals[i + 1]) / norm_factor

        if mean_psi < eps_auto_psi and sig_diff < eps_auto_sigma * eigvals[0]:
            # MATLAB sets N_auto = i (1-based) at first noise-like mode,
            # then removes modes 1:N_auto. Python i is 0-based, so the
            # equivalent count is i + 1.
            return i + 1

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
    - Remove all selected modes with one temporal-side projection

    Algorithm:
    1. M = float32 working matrix, shape (N, n_pixels)
    2. C = (M @ M.T).astype(float64) - covariance matrix (N x N)
    3. PSI, S, _ = np.linalg.svd(C) - temporal modes and energies (float64)
    4. n_remove = find_auto_mode(PSI, S) - find noise floor
    5. M -= PSI_k @ (PSI_k.T @ M) with PSI_k = PSI[:, :n_remove] (float32)
       Snapshot-POD identity: because the temporal eigenvectors are
       orthonormal, this equals sequential rank-1 deflation exactly, with
       no N x n_pixels temporaries.

    Precision: the float32 data GEMMs were validated against the float64
    implementation on real Cam4 data (2026-07-28): identical mode counts on
    130/130 batch-channels, max pixel difference 3.7e-4 of full scale. The
    SVD itself stays float64.

    Parameters
    ----------
    images : np.ndarray
        Stack of images, shape (N, H, W), dtype typically float32.
        NOTE: a C-contiguous float32 input is filtered IN PLACE and the
        return value aliases it (the working matrix is a zero-copy view).
        Strided views and other dtypes are copied. Pass ``images.copy()``
        if the original must survive.
    eps_auto_psi : float
        Threshold for mean eigenvector criterion (default 0.01)
    eps_auto_sigma : float
        Threshold for eigenvalue difference criterion (default 0.01)
    verbose : bool
        Print progress information

    Returns
    -------
    np.ndarray
        Filtered images of same shape, float32 (may alias the input, see
        above; input returned unchanged when no modes are removed)
    """
    n_images, height, width = images.shape
    n_pixels = height * width

    # Contiguous float32 working matrix. Callers pass strided views
    # (batch[:, ch]); one explicit copy here replaces the old silent double
    # copy (non-contiguous reshape + float64 astype).
    M = np.ascontiguousarray(images, dtype=np.float32).reshape(n_images, n_pixels)

    # Covariance is N x N (small!) regardless of image resolution.
    # Accumulate in float32 (BLAS), decompose in float64.
    C = (M @ M.T).astype(np.float64)

    # SVD of covariance matrix
    # PSI contains temporal modes (eigenvectors)
    # singular_values are related to eigenvalues
    PSI, singular_values, _ = np.linalg.svd(C, full_matrices=False)

    # Find automatic mode threshold (noise floor)
    n_remove = find_auto_mode(
        PSI, singular_values, n_images, eps_auto_psi, eps_auto_sigma
    )

    if verbose:
        logger.info(f"  POD: {n_images} images, removing {n_remove} signal modes")

    if n_remove == 0:
        return images

    # Remove all selected modes in one projection. Intentionally NO
    # temporal-mean subtraction first: this is snapshot POD on raw
    # intensities (MATLAB parity) - the DC background IS the leading mode
    # and removing it is the filter's purpose.
    Pk = PSI[:, :n_remove].astype(np.float32)
    M -= Pk @ (Pk.T @ M)

    return M.reshape(n_images, height, width)


def pod_filter_batch(
    batch: np.ndarray,
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
        Stack of image pairs, shape (N, 2, H, W). Filtered IN PLACE
        (channel results are written back into this array) and also
        returned for convenience.
        - N: number of image pairs
        - 2: channels (A=0, B=1)
        - H, W: image dimensions
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
    n_pairs = batch.shape[0]

    if verbose:
        logger.debug(f"POD Filter: Processing {n_pairs} image pairs")

    # Process A channel (index 0)
    if verbose:
        logger.info("  Processing channel A...")
    batch[:, 0] = pod_filter_single_channel(
        batch[:, 0], eps_auto_psi, eps_auto_sigma, verbose=verbose
    )

    # Process B channel (index 1)
    if verbose:
        logger.info("  Processing channel B...")
    batch[:, 1] = pod_filter_single_channel(
        batch[:, 1], eps_auto_psi, eps_auto_sigma, verbose=verbose
    )

    if verbose:
        logger.debug("POD Filter: Complete")

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
        logger.debug(f"Time Filter: Processing {batch.shape[0]} image pairs")

    # Process each channel
    for channel in range(batch.shape[1]):
        # Compute minimum across temporal dimension (axis 0)
        min_vals = batch[:, channel].min(axis=0, keepdims=True)
        # Subtract in-place
        batch[:, channel] -= min_vals

    if verbose:
        logger.debug("Time Filter: Complete")

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
