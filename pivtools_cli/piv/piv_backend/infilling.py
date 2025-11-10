"""
Infilling methods for PIV velocity fields.

This module provides various methods for filling NaN/masked values in PIV data:
- Local median infilling
- K-nearest neighbors (KNN) regression
- Biharmonic inpainting
- Griddata linear interpolation
- Radial basis function (RBF) interpolation
"""

import numpy as np
import bottleneck as bn
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage as ndi
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import cKDTree
from skimage.restoration import inpaint_biharmonic

try:
    from sklearn.neighbors import KNeighborsRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def infill_local_median(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    ksize: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill masked pixels with the median of their ksize×ksize neighbors (center excluded).
    
    NaNs in the neighborhood are ignored (nanmedian).

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    mask : np.ndarray
        Boolean mask where True indicates pixels to fill.
    ksize : int, optional
        Kernel size for local median window, defaults to 3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Infilled (ux, uy) arrays.
    """
    ux = np.asarray(ux, dtype=np.float32)
    uy = np.asarray(uy, dtype=np.float32)
    H, W = ux.shape
    pad = ksize // 2

    # Pad with NaN so out-of-bounds are ignored by nanmedian
    ux_pad = np.pad(ux, pad, mode="constant", constant_values=np.nan)
    uy_pad = np.pad(uy, pad, mode="constant", constant_values=np.nan)

    # Sliding 2D windows (zero-copy views)
    win_x = sliding_window_view(ux_pad, (ksize, ksize))  # (H, W, k, k)
    win_y = sliding_window_view(uy_pad, (ksize, ksize))

    # Flatten the window and drop the center element to exclude "self"
    KK = ksize * ksize
    center_idx = KK // 2  # works for odd ksize
    keep = np.ones(KK, dtype=bool)
    keep[center_idx] = False

    nb_x = win_x.reshape(H, W, KK)[..., keep]  # (H, W, KK-1)
    nb_y = win_y.reshape(H, W, KK)[..., keep]

    # If mask is sparse, compute medians only where needed
    if mask is not None and mask.any():
        idx = np.where(mask)
        med_x_vals = bn.nanmedian(nb_x[idx], axis=-1)
        med_y_vals = bn.nanmedian(nb_y[idx], axis=-1)

        ux_out = ux.copy()
        uy_out = uy.copy()
        ux_out[idx] = med_x_vals
        uy_out[idx] = med_y_vals
        return ux_out, uy_out
    else:
        # Dense case: compute whole-field medians, then apply mask
        med_x = bn.nanmedian(nb_x, axis=-1)
        med_y = bn.nanmedian(nb_y, axis=-1)
        ux_out = np.where(mask, med_x, ux)
        uy_out = np.where(mask, med_y, uy)
        return ux_out, uy_out


def infill_biharmonic(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infill using biharmonic inpainting.
    
    Uses scikit-image's biharmonic inpainting algorithm which solves
    a partial differential equation to smoothly fill masked regions.

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    mask : np.ndarray
        Boolean mask where True indicates pixels to fill (outliers only).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Infilled (ux, uy) arrays.
    """
    # inpaint_biharmonic fills NaN regions, so we need to temporarily set outliers to NaN
    ux_temp = ux.copy()
    uy_temp = uy.copy()
    ux_temp[mask] = np.nan
    uy_temp[mask] = np.nan
    
    # Create a combined mask of what needs filling (outliers + any pre-existing NaNs)
    combined_mask = np.isnan(ux_temp) | np.isnan(uy_temp)
    
    # Biharmonic inpainting
    ux_filled = inpaint_biharmonic(ux_temp, combined_mask)
    uy_filled = inpaint_biharmonic(uy_temp, combined_mask)
    
    # Keep original valid values, only replace the outliers
    ux_out = ux.copy()
    uy_out = uy.copy()
    ux_out[mask] = ux_filled[mask]
    uy_out[mask] = uy_filled[mask]
    
    return ux_out, uy_out


def infill_griddata_linear(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    method: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infill using scipy griddata interpolation.
    
    Uses Delaunay triangulation for linear/cubic interpolation with
    nearest-neighbor fallback for edge regions.

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    mask : np.ndarray
        Boolean mask where True indicates pixels to fill.
    method : str, optional
        Interpolation method: 'linear', 'cubic', or 'nearest', defaults to 'linear'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Infilled (ux, uy) arrays.
    """
    H, W = ux.shape
    yy, xx = np.mgrid[0:H, 0:W]
    
    # Valid points are those that are NOT masked for filling AND are finite
    valid = ~mask & np.isfinite(ux) & np.isfinite(uy)
    
    if not valid.any():
        # No valid points to interpolate from
        return ux.copy(), uy.copy()
    
    pts = np.c_[yy[valid], xx[valid]]

    ux_f = griddata(pts, ux[valid], (yy, xx), method=method)
    uy_f = griddata(pts, uy[valid], (yy, xx), method=method)

    # Fallback to nearest for anything left unfilled (edges)
    if np.isnan(ux_f).any() or np.isnan(uy_f).any():
        ux_nn = griddata(pts, ux[valid], (yy, xx), method="nearest")
        uy_nn = griddata(pts, uy[valid], (yy, xx), method="nearest")
        ux_f = np.where(np.isnan(ux_f), ux_nn, ux_f)
        uy_f = np.where(np.isnan(uy_f), uy_nn, uy_f)

    # Only replace the masked outliers, keep everything else
    ux_out = ux.copy()
    uy_out = uy.copy()
    ux_out[mask] = ux_f[mask]
    uy_out[mask] = uy_f[mask]
    
    return ux_out, uy_out


def infill_rbf_local(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    neighbors: int = 64,
    kernel: str = "thin_plate_spline",
    epsilon: float = None,
    smoothing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infill masked PIV vectors with local RBFs (memory-safe).
    
    Uses radial basis function interpolation with local neighborhood
    support to keep memory usage reasonable for large fields.

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    mask : np.ndarray
        Boolean mask where True indicates pixels to fill.
    neighbors : int, optional
        Number of nearest neighbors used per query, defaults to 64.
        Typical range: 32-128.
    kernel : str, optional
        RBF kernel type, defaults to "thin_plate_spline".
        Options: "thin_plate_spline", "multiquadric", "inverse_multiquadric",
        "cubic", "quintic", "linear", "gaussian".
    epsilon : float, optional
        Shape parameter (length scale). Auto-determined if None.
        Try 1-3 pixels for gaussian/multiquadric; TPS ignores epsilon.
    smoothing : float, optional
        Regularization parameter, defaults to 0.0.
        Try 1e-3 to 1e-1 if data is noisy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Infilled (ux, uy) arrays.
    """
    H, W = ux.shape
    yy, xx = np.mgrid[0:H, 0:W]
    
    # Valid training points: not masked for filling AND finite
    valid = (~mask) & np.isfinite(ux) & np.isfinite(uy)
    
    if valid.sum() < 4:
        # Not enough points—just return originals
        return ux.copy(), uy.copy()

    X_train = np.c_[xx[valid].ravel(), yy[valid].ravel()].astype(np.float64)
    # Only query points that need filling (the mask)
    X_query = np.c_[xx[mask].ravel(), yy[mask].ravel()].astype(np.float64)

    u_train = ux[valid].astype(np.float64)
    v_train = uy[valid].astype(np.float64)

    # Build local RBF models (KD-tree inside; memory ~ O(N))
    rbf_u = RBFInterpolator(
        X_train, u_train,
        kernel=kernel,
        epsilon=epsilon,
        neighbors=neighbors,
        smoothing=smoothing
    )
    rbf_v = RBFInterpolator(
        X_train, v_train,
        kernel=kernel,
        epsilon=epsilon,
        neighbors=neighbors,
        smoothing=smoothing
    )

    u_pred = rbf_u(X_query)
    v_pred = rbf_v(X_query)

    # Keep originals; fill only masked cells
    ux_filled = ux.copy()
    uy_filled = uy.copy()
    ux_filled[mask] = u_pred
    uy_filled[mask] = v_pred

    return ux_filled, uy_filled


def infill_knn(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int = 32,
    weights: str = "distance",
    algorithm: str = "kd_tree",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infill masked PIV vectors using K-nearest-neighbor regression.
    
    Uses scikit-learn's KNN regressor for fast, local interpolation.
    Requires scikit-learn to be installed.

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    mask : np.ndarray
        Boolean mask where True indicates pixels to fill.
    n_neighbors : int, optional
        Number of neighbors for interpolation, defaults to 32.
        Typical range: 16-64.
    weights : str, optional
        Weighting scheme: "uniform" or "distance", defaults to "distance".
        "distance" usually gives smoother results.
    algorithm : str, optional
        Nearest neighbor search algorithm, defaults to "kd_tree".
        Options: "auto", "ball_tree", "kd_tree", "brute".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Infilled (ux, uy) arrays.
    
    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for KNN infilling. "
            "Install with: pip install scikit-learn"
        )
    
    H, W = ux.shape
    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    mask = mask.astype(bool)

    # Early exit if nothing to fill
    if not np.any(mask):
        return ux, uy

    yy, xx = np.mgrid[0:H, 0:W]
    valid = (~mask) & np.isfinite(ux) & np.isfinite(uy)

    X_train = np.c_[xx[valid], yy[valid]]
    X_query = np.c_[xx[mask], yy[mask]]

    # Clamp n_neighbors to avoid errors
    n_valid = X_train.shape[0]
    if n_valid < n_neighbors:
        if n_valid < 1:
            return ux, uy
        n_neighbors = n_valid

    # Fit single KNN model for both components (vectorized)
    knn = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        n_jobs=-1  # Use all CPU cores
    )
    
    # Stack u and v as multi-output targets
    y_train = np.column_stack([ux[valid], uy[valid]])
    knn.fit(X_train, y_train)
    
    # Single prediction call for both components
    predictions = knn.predict(X_query)

    ux_filled = ux.copy()
    uy_filled = uy.copy()
    ux_filled[mask] = predictions[:, 0]
    uy_filled[mask] = predictions[:, 1]
    
    return ux_filled, uy_filled


def apply_infilling(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    method_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the configured infilling method.
    
    This function dispatches to the appropriate infilling method based on
    the configuration dictionary.

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    mask : np.ndarray
        Boolean mask where True indicates pixels to fill.
    method_cfg : dict
        Configuration dictionary with 'method' and 'parameters' keys.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Infilled (ux, uy) arrays.
    
    Raises
    ------
    ValueError
        If unknown infilling method is specified.
    """
    method = method_cfg.get('method', 'local_median').lower()
    params = method_cfg.get('parameters', {})
    
    if method == 'local_median':
        ksize = params.get('ksize', 3)
        return infill_local_median(ux, uy, mask, ksize=ksize)
    
    elif method == 'knn':
        n_neighbors = params.get('n_neighbors', 32)
        weights = params.get('weights', 'distance')
        algorithm = params.get('algorithm', 'kd_tree')
        return infill_knn(ux, uy, mask, n_neighbors=n_neighbors, 
                         weights=weights, algorithm=algorithm)
    
    elif method == 'biharmonic':
        return infill_biharmonic(ux, uy, mask)
    
    elif method == 'griddata_linear':
        interp_method = params.get('method', 'linear')
        return infill_griddata_linear(ux, uy, mask, method=interp_method)
    
    elif method == 'rbf_local':
        neighbors = params.get('neighbors', 64)
        kernel = params.get('kernel', 'thin_plate_spline')
        epsilon = params.get('epsilon', None)
        smoothing = params.get('smoothing', 0.0)
        return infill_rbf_local(ux, uy, mask, neighbors=neighbors,
                               kernel=kernel, epsilon=epsilon, smoothing=smoothing)
    
    else:
        raise ValueError(f"Unknown infilling method: {method}")
