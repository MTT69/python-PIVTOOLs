"""
Gaussian Fitting Utilities for Ensemble PIV

This module contains helper functions for Gaussian fitting
in ensemble PIV processing.
"""

import ctypes

import cv2
import numpy as np

from pivtools_core.config import Config
from pivtools_cli.piv.piv_backend.infilling import apply_infilling

# Module-level cache for the Marquadt library
_marquadt_lib = None


def _load_marquadt_lib():
    """Load the Marquadt library for Gaussian fitting."""
    global _marquadt_lib
    if _marquadt_lib is not None:
        return _marquadt_lib
    import ctypes
    import os

    lib_extension = ".dll" if os.name == "nt" else ".so"
    
    # Try multiple possible paths for the library
    possible_paths = [
        # Absolute path to the project lib directory
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "lib", f"libmarquadt{lib_extension}")),
        # From current working directory
        os.path.abspath(os.path.join("pivtools_cli", "lib", f"libmarquadt{lib_extension}")),
        # Hardcoded absolute path (for debugging)
        os.path.abspath("/Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools/pivtools_cli/lib/libmarquadt.so"),
    ]
    
    for path in possible_paths:
        marquadt_libpath = os.path.abspath(path)
        if os.path.isfile(marquadt_libpath):
            break
    else:
        raise FileNotFoundError(
            f"Marquadt library not found. Tried paths: {possible_paths}"
        )
    
    marquadt_lib = ctypes.CDLL(marquadt_libpath)
    marquadt_lib.fit_stacked_gaussian_export.argtypes = [
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    marquadt_lib.fit_stacked_gaussian_export.restype = ctypes.c_int
    return marquadt_lib


def _get_sigma_from_previous_pass(
    pass_idx: int,
    n_windows: int,
    config: Config,
    piv_results,
    n_win_x: int,
    n_win_y: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Interpolate sigma fields from previous pass for initial guess.

    Displacement and amplitude guesses are ALWAYS determined from finding
    peaks in the correlation planes (after image warping). Only sigma values
    are propagated from the previous pass to provide better initial
    uncertainty estimates.

    For pass 0: Returns None (sigmas estimated from HWHM directly)
    For pass > 0: Returns interpolated sigma fields from previous pass
    after NaN infilling with Gaussian kernel smoothing.

    Parameters
    ----------
    pass_idx : int
        Current pass index
    n_windows : int
        Total number of windows (flattened)
    config : Config
        Configuration object
    piv_results : PIVEnsembleResult
        Previous pass results
    n_win_x : int
        Number of windows in x direction
    n_win_y : int
        Number of windows in y direction

    Returns
    -------
    sigma_x : np.ndarray or None
        Sigma values in x direction (None for pass 0)
    sigma_y : np.ndarray or None
        Sigma values in y direction (None for pass 0)
    """
    if pass_idx == 0:
        # Pass 0: Sigmas computed from HWHM in _build_initial_guess
        return None, None

    # Retrieve sigma fields from previous pass
    old_sigma_x = (piv_results.passes[pass_idx - 1].sig_AB_x.copy()
                   .astype(np.float32))
    old_sigma_y = (piv_results.passes[pass_idx - 1].sig_AB_y.copy()
                   .astype(np.float32))

    # Apply NaN infilling with Gaussian smoothing to suppress outliers
    sigma_nan_mask = np.isnan(old_sigma_x) | np.isnan(old_sigma_y)
    if sigma_nan_mask.any():
        mid_infill_cfg = config.infilling_mid_pass
        old_sigma_x, old_sigma_y = apply_infilling(
            old_sigma_x, old_sigma_y, sigma_nan_mask, mid_infill_cfg
        )

    old_h, old_w = old_sigma_x.shape
    new_h, new_w = n_win_y, n_win_x

    # Interpolate to current grid using cubic interpolation
    if (old_h, old_w) == (new_h, new_w):
        # Same grid size - no interpolation needed
        sigma_x = old_sigma_x.ravel(order="C")
        sigma_y = old_sigma_y.ravel(order="C")
    else:
        # Different grid size - use cubic interpolation for smooth upsampling
        map_y, map_x = np.meshgrid(
            np.linspace(0, old_h - 1, new_h).astype(np.float32),
            np.linspace(0, old_w - 1, new_w).astype(np.float32),
            indexing="ij"
        )
        sigma_x = cv2.remap(
            old_sigma_x, map_x, map_y, cv2.INTER_CUBIC
        ).ravel(order="C")
        sigma_y = cv2.remap(
            old_sigma_y, map_x, map_y, cv2.INTER_CUBIC
        ).ravel(order="C")

    # No artificial min/max constraints - use interpolated values as-is
    return sigma_x, sigma_y


def _validate_fitted_params(
    gauss_params: np.ndarray,
    win_size: tuple,
    pass_idx: int,
    runtype: str,
    sum_window: tuple,
    AA_central: float,
    BB_central: float
) -> tuple[bool, int]:
    """
    Validate fitted Gaussian parameters following MATLAB logic.

    Based on process_correlation_planes.m lines 129-183.

    Parameters
    ----------
    gauss_params : np.ndarray
        Fitted parameters [amp_A, amp_B, amp_AB, sx_A, sy_A, sxy_A,
                          sx_AB, sy_AB, sxy_AB, x0_A, y0_A, x0_AB, y0_AB]
    win_size : tuple
        (height, width) of correlation window
    pass_idx : int
        Current pass index
    runtype : str
        'single' or 'standard'
    sum_window : tuple
        SumWindow size (for single mode)
    AA_central : float
        Central value of AA autocorrelation
    BB_central : float
        Central value of BB autocorrelation

    Returns
    -------
    is_valid : bool
        True if parameters pass all checks
    nan_reason : int
        Reason code if invalid (0 if valid)
        1 = solver didn't converge (handled before this)
        2 = AB peak height invalid (not in [0,1])
        3 = breaks 1/2 displacement rule
        4 = Gaussian spread too large
        5 = negative sigmas
    """
    # Extract parameters
    amp_A, amp_B, amp_AB = gauss_params[0:3]
    sx_A, sy_A, sxy_A = gauss_params[3:6]
    sx_AB, sy_AB, sxy_AB = gauss_params[6:9]
    x0_A, y0_A = gauss_params[9:11]
    x0_AB, y0_AB = gauss_params[11:13]

    # Check 1: AB peak height validity
    if AA_central > 1e-12 and BB_central > 1e-12:
        AB_normalized = amp_AB / np.sqrt(AA_central * BB_central)
        if not np.isreal(AB_normalized) or AB_normalized < 0 or AB_normalized > 1:
            return False, 2

    # Check 2: 1/2 displacement rule 
    if runtype == 'single':
        center_x = sum_window[1] / 2.0
        center_y = sum_window[0] / 2.0
        half_x = sum_window[1] / 2.0
        half_y = sum_window[0] / 2.0
    else:
        center_x = win_size[1] / 2.0
        center_y = win_size[0] / 2.0
        half_x = win_size[1] / 2.0
        half_y = win_size[0] / 2.0

    # For pass > 0 or single mode, check peak is within central half
    if pass_idx > 0 or runtype == 'single':
        if (abs(x0_AB - center_x) > half_x or
            abs(y0_AB - center_y) > half_y):
            return False, 3

    # Check 3: Negative sigmas
    if sx_AB < 0 or sy_AB < 0:
        return False, 5

    return True, 0


def _fit_windows_batch_optimized(
    scattered_AA, scattered_BB, scattered_AB,
    scattered_sigma_x, scattered_sigma_y, scattered_mask,
    start_idx, end_idx,
    win_size, config, pass_idx, scattered_cache, outdir=None
):
    """
    Optimized Gaussian fitting with broadcast data and pre-allocated arrays.

    Receives broadcast arrays and extracts local chunk to minimize serialization.

    Parameters
    ----------
    scattered_AA, scattered_BB, scattered_AB : np.ndarray
        Broadcast correlation planes (full arrays)
    scattered_sigma_x, scattered_sigma_y : np.ndarray or None
        Broadcast sigma values (full arrays, or None for pass 0)
    scattered_mask : np.ndarray
        Broadcast mask array (full array)
    start_idx, end_idx : int
        Window indices for this worker
    win_size : tuple
        (height, width) of correlation window
    config : Config
        Configuration object
    pass_idx : int
        Current pass index
    scattered_cache : dict
        Scattered correlator cache
    outdir : Optional[Path]
        Output directory for debug info

    Returns
    -------
    results : np.ndarray
        Fitted parameters for each window
    statuses : np.ndarray
        Fitting status codes
    initial_guesses : np.ndarray
        Initial guesses used for fitting
    """
    marquadt_lib = _load_marquadt_lib()

    # Extract local chunk from broadcast data (no network transfer)
    plane_size = win_size[0] * win_size[1]
    start_data = start_idx * plane_size
    end_data = end_idx * plane_size

    AA_chunk = scattered_AA[start_data:end_data]
    BB_chunk = scattered_BB[start_data:end_data]
    AB_chunk = scattered_AB[start_data:end_data]

    mask_chunk = scattered_mask[start_idx:end_idx]
    sigma_x_chunk = scattered_sigma_x[start_idx:end_idx] if scattered_sigma_x is not None else None
    sigma_y_chunk = scattered_sigma_y[start_idx:end_idx] if scattered_sigma_y is not None else None

    num_windows = end_idx - start_idx
    X1, X2, central_index, x_guess, y_guess = _get_pass_grid(pass_idx, config)

    # Pre-allocate output arrays (avoid per-window allocation)
    results = np.zeros((num_windows, 13), dtype=np.float64)
    statuses = np.zeros(num_windows, dtype=np.int32)
    initial_guesses = np.zeros((num_windows, 13), dtype=np.float64)

    # Process windows
    for idx in range(num_windows):
        # Skip if masked
        if mask_chunk[idx]:
            statuses[idx] = -1  # Status -1 indicates masked/skipped window
            continue

        # Extract window
        AA_win = _get_window(AA_chunk, idx, win_size)
        BB_win = _get_window(BB_chunk, idx, win_size)
        AB_win = _get_window(AB_chunk, idx, win_size)

        # Get sigma values
        sigma_x_val = sigma_x_chunk[idx] if sigma_x_chunk is not None else None
        sigma_y_val = sigma_y_chunk[idx] if sigma_y_chunk is not None else None

        # Build initial guess
        initial_guess, real_corr = _build_initial_guess(
            idx, pass_idx, AA_win, BB_win, AB_win, central_index,
            x_guess, y_guess, sigma_x_val, sigma_y_val,
            win_size, config
        )

        # Call C library (unavoidable per-window call)
        out_params = results[idx]
        out_status = np.zeros(1, dtype=np.int32)

        marquadt_lib.fit_stacked_gaussian_export(
            ctypes.c_size_t(win_size[0] * win_size[1]),
            X2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            X1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            real_corr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            initial_guess.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_status.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )

        statuses[idx] = out_status[0]
        initial_guesses[idx] = initial_guess

        # Validate if converged
        if statuses[idx] == 0:
            is_valid, nan_reason_code = _validate_fitted_params(
                out_params, win_size, pass_idx,
                config.ensemble_type[pass_idx],
                tuple(config.ensemble_sum_window),
                float(AA_win[central_index]),
                float(BB_win[central_index])
            )
            if not is_valid:
                statuses[idx] = nan_reason_code

    return results, statuses, initial_guesses


def _get_window(flat_array, idx, win_size):
    """Extract one window from a flat array."""
    start = idx * win_size[0] * win_size[1]
    end = start + win_size[0] * win_size[1]
    return flat_array[start:end]


def _get_pass_grid(pass_idx, config):
    """Get grid coordinates for Gaussian fitting."""
    runtype = config.ensemble_type[pass_idx]
    wsize = config.ensemble_window_sizes[pass_idx]
    sum_window = config.ensemble_sum_window

    if runtype == "single":
        X1, X2 = np.meshgrid(
            np.linspace(1, sum_window[0], sum_window[0]),
            np.linspace(1, sum_window[1], sum_window[1]),
            indexing="ij",
        )
        X1 = X1.ravel(order="C")
        X2 = X2.ravel(order="C")
        
        # FIX: Integer math, 0-based indexing for flat array access
        # center_y * width + center_x
        central_index = (sum_window[0] // 2) * sum_window[1] + (sum_window[1] // 2)
        
        x_guess = sum_window[1] / 2 + 1
        y_guess = sum_window[0] / 2 + 1
    else:
        X1, X2 = np.meshgrid(
            np.linspace(1, wsize[0], wsize[0]),
            np.linspace(1, wsize[1], wsize[1]),
            indexing="ij",
        )
        X1 = X1.ravel(order="C")
        X2 = X2.ravel(order="C")
        
        central_index = (wsize[0] // 2) * wsize[1] + (wsize[1] // 2)
        
        x_guess = wsize[1] / 2 + 1
        y_guess = wsize[0] / 2 + 1

    return X1, X2, central_index, x_guess, y_guess


def _estimate_sigma_from_plane(
    corr_plane: np.ndarray,
    peak_idx: int,
    win_size: tuple,
    central_idx: int,
    min_sigma: float = 0.5
) -> tuple[float, float, float, float]:
    """
    Estimate Gaussian sigma from correlation plane shape.

    Uses half-width at half-maximum (HWHM) to estimate sigma.

    Parameters
    ----------
    corr_plane : np.ndarray
        Flattened correlation plane
    peak_idx : int
        Index of peak in flattened array
    win_size : tuple
        (height, width) of correlation window
    central_idx : int
        Index of window center (for fallback)
    min_sigma : float
        Minimum sigma value (safety bound)

    Returns
    -------
    sigma_x, sigma_y, hwhm_x, hwhm_y : float, float, float, float
        Estimated Gaussian widths and HWHM values in x and y directions
    """
    # Reshape to 2D for analysis
    plane_2d = corr_plane.reshape(win_size[0], win_size[1])
    peak_y, peak_x = np.unravel_index(peak_idx, win_size, order="C")
    peak_val = plane_2d[peak_y, peak_x]

    # Handle edge case: peak too low
    if peak_val < 1e-6:
        return min_sigma, min_sigma, min_sigma * np.sqrt(2 * np.log(2)), \
               min_sigma * np.sqrt(2 * np.log(2))

    # Find half-maximum threshold
    threshold = peak_val / 2.0

    # Estimate sigma_x: find width in x-direction at peak_y
    x_profile = plane_2d[peak_y, :]
    x_above = np.where(x_profile >= threshold)[0]
    if len(x_above) >= 2:
        hwhm_x = (x_above[-1] - x_above[0]) / 2.0
        sigma_x = hwhm_x / np.sqrt(2 * np.log(2))  # HWHM to sigma conversion
    else:
        hwhm_x = min_sigma * np.sqrt(2 * np.log(2))
        sigma_x = min_sigma

    # Estimate sigma_y: find width in y-direction at peak_x
    y_profile = plane_2d[:, peak_x]
    y_above = np.where(y_profile >= threshold)[0]
    if len(y_above) >= 2:
        hwhm_y = (y_above[-1] - y_above[0]) / 2.0
        sigma_y = hwhm_y / np.sqrt(2 * np.log(2))
    else:
        hwhm_y = min_sigma * np.sqrt(2 * np.log(2))
        sigma_y = min_sigma

    # Apply safety bounds
    sigma_x = max(sigma_x, min_sigma)
    sigma_y = max(sigma_y, min_sigma)

    return sigma_x, sigma_y, hwhm_x, hwhm_y


def _build_initial_guess(
    idx, pass_idx, AA_win, BB_win, AB_win, central_index,
    x_guess, y_guess, sigma_x, sigma_y, win_size, config
):
    """
    Build initial guess for Gaussian fitting.

    Displacement and amplitude guesses are ALWAYS found by locating peaks
    in the correlation planes (after image warping for pass > 0).
    - Displacement: Peak location in AB cross-correlation
    - Amplitude: Peak values at those locations

    Sigma guesses come from:
    - Pass 0: Computed as HWHM_cross - HWHM_auto
    - Pass > 0: Interpolated from previous pass (after infilling)

    Parameters
    ----------
    idx : int
        Window index (unused, kept for compatibility)
    pass_idx : int
        Current pass index
    AA_win : np.ndarray
        AA autocorrelation window (flattened, after warping)
    BB_win : np.ndarray
        BB autocorrelation window (flattened, after warping)
    AB_win : np.ndarray
        AB cross-correlation window (flattened, after warping)
    central_index : int
        Index of window center (for auto-correlation peaks)
    x_guess : float
        X coordinate for center A position
    y_guess : float
        Y coordinate for center A position
    sigma_x : float or None
        Sigma in x direction (None for pass 0, interpolated value for pass > 0)
    sigma_y : float or None
        Sigma in y direction (None for pass 0, interpolated value for pass > 0)
    win_size : tuple
        (height, width) of correlation window
    config : Config
        Configuration object (unused, kept for compatibility)

    Returns
    -------
    initial_guess : np.ndarray
        Initial parameter guess for Gaussian fitting
    real_corr : np.ndarray
        Concatenated correlation planes
    """

    # Always find peak position in AB cross-correlation (after warping)
    max_idx = np.argmax(AB_win)
    guess_y_AB, guess_x_AB = np.unravel_index(max_idx, win_size, order="C")

    # Sigma A: Always estimated from AA autocorrelation HWHM
    sigma_A_x, sigma_A_y, hwhm_A_x, hwhm_A_y = _estimate_sigma_from_plane(
        AA_win, central_index, win_size, central_index
    )

    # Sigma AB estimation
    if pass_idx == 0 or sigma_x is None or sigma_y is None:
        # Pass 0: Compute as HWHM_cross - HWHM_auto
        # This removes the contribution of particle image size
        _, _, hwhm_AB_x, hwhm_AB_y = _estimate_sigma_from_plane(
            AB_win, max_idx, win_size, central_index, min_sigma=0.5
        )
        # Compute difference (ensures positive value for numerical stability)
        hwhm_diff_x = max(hwhm_AB_x - hwhm_A_x, 0.1 * np.sqrt(2 * np.log(2)))
        hwhm_diff_y = max(hwhm_AB_y - hwhm_A_y, 0.1 * np.sqrt(2 * np.log(2)))
        # Convert to sigma
        sigma_AB_x = hwhm_diff_x / np.sqrt(2 * np.log(2))
        sigma_AB_y = hwhm_diff_y / np.sqrt(2 * np.log(2))
    else:
        # Pass > 0: Use interpolated values from previous pass
        # No artificial constraints - trust the interpolated values
        sigma_AB_x = float(sigma_x)
        sigma_AB_y = float(sigma_y)

    initial_guess = np.array([
        float(AA_win[central_index]),    # Amp A at center
        float(BB_win[central_index]),    # Amp B at center
        float(AB_win[max_idx]),          # Amp AB at peak (not center!)
        sigma_A_x, sigma_A_y, 0.0,       # Sigma A (adaptive from AA plane)
        sigma_AB_x, sigma_AB_y, 0.0,     # Sigma AB (HWHM diff for pass 0,
                                         # from previous pass for pass > 0)
        x_guess, y_guess,                # Center A (x, y)
        float(guess_x_AB + 1),           # Center AB x (1-based indexing)
        float(guess_y_AB + 1),           # Center AB y (1-based indexing)
    ], dtype=np.float64)

    real_corr = np.concatenate([AA_win, BB_win, AB_win]).astype(np.float64)
    return initial_guess, real_corr