"""
Outlier detection methods for PIV velocity fields.

This module provides various methods for detecting outliers in PIV data:
- Peak magnitude thresholding
- Median-based 2D outlier detection
- Sigma-based outlier detection
- Divergence/vorticity-based outlier detection
"""

import numpy as np
import bottleneck as bn
from scipy.signal import convolve2d
from scipy import ndimage as ndi


def peak_magnitude_detection(
    peak_mag: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Detect outliers based on peak magnitude threshold.
    
    Vectors with peak magnitudes below the threshold are marked as outliers.

    Parameters
    ----------
    peak_mag : np.ndarray
        Peak magnitude array from PIV correlation.
    threshold : float, optional
        Minimum acceptable peak magnitude, defaults to 0.5.

    Returns
    -------
    np.ndarray
        Boolean array indicating outliers (True = outlier).
    """
    b_filter = (peak_mag < threshold) | np.isnan(peak_mag)
    return b_filter


def median_outlier_detection(
    ux: np.ndarray,
    uy: np.ndarray,
    epsilon: float = 0.2,
    threshold: float = 2.0,
) -> np.ndarray:
    """
    Fast median-based outlier detection for 2D PIV velocity fields,
    using bottleneck for nan-aware reductions.
    
    This method compares each vector to the median of its 8 neighbors.
    Outliers are identified based on the normalized residual exceeding a threshold.

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    epsilon : float, optional
        Regularization term for division stability, defaults to 0.2.
    threshold : float, optional
        Threshold for outlier detection, defaults to 2.0.

    Returns
    -------
    np.ndarray
        Boolean mask of outliers (True = outlier).
    """
    if ux.shape != uy.shape:
        raise ValueError("ux and uy must have identical shapes")

    n_wx, n_wy = ux.shape
    ui = np.stack((ux, uy), axis=-1).astype(np.float32, copy=False)

    r_0p = np.zeros((n_wx, n_wy, 2), dtype=np.float32)
    n_neighbours = np.zeros((n_wx, n_wy, 2), dtype=np.float32)

    ones3 = np.ones((3, 3), dtype=np.float32)

    for c in range(2):
        U = ui[..., c]
        U_pad = np.pad(U, 1, mode="constant", constant_values=np.nan)

        # Collect 8-neighbor pixels
        U_nn = np.stack([
            U_pad[:-2, :-2],  # top-left
            U_pad[:-2, 1:-1],  # top
            U_pad[:-2, 2:],    # top-right
            U_pad[1:-1, :-2],  # left
            U_pad[1:-1, 2:],   # right
            U_pad[2:, 2:],     # bottom-right
            U_pad[2:, 1:-1],   # bottom
            U_pad[2:, :-2],    # bottom-left
        ], axis=-1)  # (H, W, 8)

        # --- bottleneck median operations (C-accelerated) ---
        U_med = bn.nanmedian(U_nn, axis=-1)  # neighbor median
        r_0 = np.abs(U_med - U)
        r_i = np.abs(U_nn - U_med[..., None])
        r_m = bn.nanmedian(r_i, axis=-1)     # median absolute deviation

        r_0p[..., c] = r_0 / (r_m + epsilon)

        # Valid neighbor count via convolution (3×3 kernel)
        valid = (~np.isnan(U)).astype(np.float32)
        n_neigh = convolve2d(valid, ones3, mode="same", boundary="fill", fillvalue=0.0)
        n_neighbours[..., c] = n_neigh

    r_0_combined = bn.nanmax(r_0p, axis=2)

    # Boolean mask: true = outlier
    b_filter = (
        (r_0_combined > threshold)
        | np.isnan(r_0p).any(axis=2)
        | (n_neighbours < 6).any(axis=2)
    )
    return b_filter


def sigma_outlier_detection(
    ux: np.ndarray,
    uy: np.ndarray,
    sigma_threshold: float = 2.0,
) -> np.ndarray:
    """
    Detect outliers based on local standard deviation (sigma-based).
    
    Vectors that deviate from the 8-neighbor mean by more than sigma_threshold
    times the local standard deviation are marked as outliers.

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    sigma_threshold : float, optional
        Number of standard deviations for outlier threshold, defaults to 2.0.

    Returns
    -------
    np.ndarray
        Boolean mask of outliers (True = outlier).
    """
    v = np.sqrt(ux**2 + uy**2).astype(np.float32, copy=False)
    finite = np.isfinite(v).astype(np.float32)
    v0 = np.where(np.isfinite(v), v, 0.0)

    # 3x3 window means for value, value^2, and for the finite-mask
    m9_v = ndi.uniform_filter(v0, size=3, mode='constant', cval=0.0)
    m9_v2 = ndi.uniform_filter(v0 * v0, size=3, mode='constant', cval=0.0)
    m9_cnt = ndi.uniform_filter(finite, size=3, mode='constant', cval=0.0)

    # uniform_filter returns MEAN; convert to SUM by * 9
    sum9 = m9_v * 9.0
    sumsq9 = m9_v2 * 9.0
    cnt9 = m9_cnt * 9.0

    # Exclude the center pixel to get 8-neighbour stats
    center_val = np.where(np.isfinite(v), v, 0.0)
    center_cnt = finite
    sum8 = sum9 - center_val
    sumsq8 = sumsq9 - center_val * center_val
    cnt8 = cnt9 - center_cnt

    # Mean and std over the 8 neighbours (ignore divisions by <=0 count)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean8 = sum8 / cnt8
        var8 = (sumsq8 / cnt8) - mean8 * mean8
    var8 = np.maximum(var8, 0.0)
    std8 = np.sqrt(var8)

    # Outlier mask
    b_filter = (
        (np.abs(v - mean8) > sigma_threshold * std8)
        | ~np.isfinite(v)
        | ~np.isfinite(mean8)
        | (std8 == 0)
        | (cnt8 < 1)
    )
    return b_filter


def div_vort_outliers(
    ux: np.ndarray,
    uy: np.ndarray,
    div_thresh: float = None,
    vort_thresh: float = None,
) -> np.ndarray:
    """
    Detect outliers based on divergence and vorticity thresholds.
    
    Computes divergence and vorticity using central differences, then
    identifies outliers as vectors with extreme values. Thresholds are
    automatically computed from the field statistics if not provided.

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    div_thresh : float, optional
        Divergence threshold. If None, computed as 6*MAD.
    vort_thresh : float, optional
        Vorticity threshold. If None, computed as 6*MAD.

    Returns
    -------
    np.ndarray
        Boolean mask of outliers (True = outlier).
    """
    # Central differences
    dudx = 0.5 * (np.roll(ux, -1, 1) - np.roll(ux, 1, 1))
    dudy = 0.5 * (np.roll(ux, -1, 0) - np.roll(ux, 1, 0))
    dvdx = 0.5 * (np.roll(uy, -1, 1) - np.roll(uy, 1, 1))
    dvdy = 0.5 * (np.roll(uy, -1, 0) - np.roll(uy, 1, 0))
    
    div = dudx + dvdy
    vort = dvdx - dudy
    
    # Robust thresholds from field using MAD (Median Absolute Deviation)
    if div_thresh is None:
        div_finite = div[np.isfinite(div)]
        if div_finite.size > 0:
            div_median = np.median(div_finite)
            mad = 1.4826 * np.median(np.abs(div_finite - div_median))
            div_thresh = 6 * mad
        else:
            div_thresh = np.inf
    
    if vort_thresh is None:
        vort_finite = vort[np.isfinite(vort)]
        if vort_finite.size > 0:
            vort_median = np.median(vort_finite)
            mad = 1.4826 * np.median(np.abs(vort_finite - vort_median))
            vort_thresh = 6 * mad
        else:
            vort_thresh = np.inf
    
    return (np.abs(div) > div_thresh) | (np.abs(vort) > vort_thresh)


def apply_outlier_detection(
    ux: np.ndarray,
    uy: np.ndarray,
    methods: list,
    peak_mag: np.ndarray = None,
) -> np.ndarray:
    """
    Apply multiple outlier detection methods and combine results.
    
    This function applies a stack of outlier detection methods configured
    in the YAML file and combines their results with logical OR.

    Parameters
    ----------
    ux : np.ndarray
        Horizontal velocity component (2D).
    uy : np.ndarray
        Vertical velocity component (2D).
    methods : list
        List of method dictionaries from config, each with 'type' and parameters.
    peak_mag : np.ndarray, optional
        Peak magnitude array (required for 'peak_mag' method).

    Returns
    -------
    np.ndarray
        Combined boolean mask of outliers (True = outlier).
    """
    combined_mask = np.zeros(ux.shape, dtype=bool)
    
    for method_cfg in methods:
        method_type = method_cfg.get('type', '').lower()
        
        if method_type == 'peak_mag':
            if peak_mag is None:
                raise ValueError("peak_mag array required for 'peak_mag' outlier detection")
            threshold = method_cfg.get('threshold', 0.5)
            mask = peak_magnitude_detection(peak_mag, threshold=threshold)
            combined_mask |= mask
            
        elif method_type == 'median_2d':
            epsilon = method_cfg.get('epsilon', 0.2)
            threshold = method_cfg.get('threshold', 2.0)
            mask = median_outlier_detection(ux, uy, epsilon=epsilon, threshold=threshold)
            combined_mask |= mask
            
        elif method_type == 'sigma':
            sigma_threshold = method_cfg.get('sigma_threshold', 2.0)
            mask = sigma_outlier_detection(ux, uy, sigma_threshold=sigma_threshold)
            combined_mask |= mask
            
        elif method_type == 'div_vort':
            div_thresh = method_cfg.get('div_thresh', None)
            vort_thresh = method_cfg.get('vort_thresh', None)
            mask = div_vort_outliers(ux, uy, div_thresh=div_thresh, vort_thresh=vort_thresh)
            combined_mask |= mask
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method_type}")
    
    return combined_mask
