"""
Gaussian Fitting Utilities for Ensemble PIV

This module contains helper functions for Gaussian fitting
in ensemble PIV processing.

NaN Reason Codes (nan_reason / status codes)
============================================
These codes indicate why a vector was marked as invalid or failed.
Stored in PIVPassResult.nan_reason array.

Code  Stage                   Description
----  -----                   -----------
 -1   Pre-fitting             Masked vector (not correlated, e.g., outside ROI)
  0   Success                 Fit succeeded and passed all validation
  1   C solver                Levenberg-Marquardt solver did not converge
  2   Post-fit validation     AB peak height invalid (normalized height not in [0,1])
  3   Post-fit validation     Breaks 1/2 displacement rule (peak too far from center)
  5   Post-fit validation     Negative sigma values (unphysical Gaussian width)
  6   Displacement check      Displacement exceeds 3/4 window rule (too large)
 10   Outlier detection       Fit succeeded but vector flagged as outlier by
                              median-based displacement outlier detection

Validation Pipeline
===================
1. C solver attempts Levenberg-Marquardt fit → code 1 if fails
2. _validate_fitted_params() checks fitted parameters → codes 2, 3, 5
3. Displacement magnitude check (3/4 rule) → code 6
4. Median-based outlier detection on displacement field → code 10
5. Vectors with code 0 after all checks are valid

Initial Guess Sources
=====================
For pass 0:
  - Amplitudes: Peak values from current correlation planes
  - Offsets: 5th percentile of current correlation planes
  - Sigmas: HWHM estimation from AA (particle size) and AB (displacement uncertainty)
  - Positions: Peak finding in AB cross-correlation

For pass > 0:
  - Amplitudes: Peak values from WARPED correlation planes (re-computed)
  - Offsets: 5th percentile of WARPED correlation planes (re-computed)
  - Sigmas: Interpolated from previous pass (outlier-detected + infilled + validated)
  - Positions: Peak finding in WARPED AB cross-correlation
"""

import ctypes
import logging

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
    import logging

    lib_extension = ".dll" if os.name == "nt" else ".so"

    # Try multiple possible paths for the library
    possible_paths = [
        # Installed package location (relative to this file)
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "lib", f"libmarquadt{lib_extension}")),
        # Development: from current working directory
        os.path.abspath(os.path.join("pivtools_cli", "lib", f"libmarquadt{lib_extension}")),
    ]

    for path in possible_paths:
        marquadt_libpath = os.path.abspath(path)
        if os.path.isfile(marquadt_libpath):
            break
    else:
        raise FileNotFoundError(
            f"Marquadt library not found. Tried paths: {possible_paths}"
        )
    logging.debug(f"Loading Marquadt library from: {marquadt_libpath}")
    marquadt_lib = ctypes.CDLL(marquadt_libpath)

    # Set up ctypes bindings for the offset control function
    marquadt_lib.set_disable_offset.argtypes = [ctypes.c_int]
    marquadt_lib.set_disable_offset.restype = None

    # Set up ctypes bindings for center pixel masking control
    marquadt_lib.set_mask_center.argtypes = [ctypes.c_int]
    marquadt_lib.set_mask_center.restype = None

    _marquadt_lib = marquadt_lib
    return marquadt_lib


def set_offset_fitting(enabled: bool = True):
    """
    Enable or disable offset (+C) fitting in the Gaussian solver.

    When offset fitting is disabled, the Gaussian model uses:
        y = amp * exp(...) + 0
    Instead of:
        y = amp * exp(...) + c

    This is useful for testing how offset fitting affects parameter recovery.

    Parameters
    ----------
    enabled : bool, default True
        If True, fit offsets normally (default behavior).
        If False, fix offsets to zero during optimization.
    """
    lib = _load_marquadt_lib()
    lib.set_disable_offset(0 if enabled else 1)


def set_center_masking(enabled: bool = True):
    """
    Enable or disable center pixel masking for autocorrelation.

    When enabled, the center pixel of AA/BB autocorrelation planes is excluded
    from fitting to remove the camera self-noise spike at zero lag. The
    cross-correlation (AB) center pixel is NOT masked since it contains valid
    displacement signal.

    Parameters
    ----------
    enabled : bool, default True
        If True, mask the center pixel (recommended for real camera data).
        If False, include all pixels (for synthetic data or testing).
    """
    lib = _load_marquadt_lib()
    lib.set_mask_center(1 if enabled else 0)


def _validate_sigma_field(
    sigma_field: np.ndarray,
    min_val: float = 0.1,
    max_val: float = 20.0
) -> np.ndarray:
    """
    Validate and clean sigma field before propagation to next pass.

    Replaces invalid values (NaN, inf, out-of-bounds) with local median.
    This prevents bad sigma estimates from propagating through passes
    and causing "pockets of bad results".

    Parameters
    ----------
    sigma_field : np.ndarray
        2D sigma field from previous pass
    min_val : float
        Minimum valid sigma value
    max_val : float
        Maximum valid sigma value

    Returns
    -------
    np.ndarray
        Validated sigma field with same shape
    """
    from scipy.ndimage import median_filter

    # Identify invalid values
    invalid = ~np.isfinite(sigma_field) | (sigma_field < min_val) | (sigma_field > max_val)

    if invalid.any():
        # Use median of valid neighbors to fill invalid values
        # First, set invalid to nan for median calculation
        temp_field = np.where(invalid, np.nan, sigma_field)

        # Median filter with nan handling
        median_field = median_filter(
            np.nan_to_num(temp_field, nan=np.nanmedian(temp_field)),
            size=3,
            mode='nearest'
        )

        sigma_field = np.where(invalid, median_field, sigma_field)

    # Final clip to bounds
    return np.clip(sigma_field, min_val, max_val)


def _get_sigma_from_previous_pass(
    pass_idx: int,
    n_windows: int,
    config: Config,
    piv_results,
    n_win_x: int,
    n_win_y: int
) -> dict:
    """
    Interpolate sigma fields from previous pass for initial guess.

    Displacement and amplitude guesses are ALWAYS determined from finding
    peaks in the correlation planes (after image warping). Only sigma values
    are propagated from the previous pass to provide better initial
    uncertainty estimates.

    For pass 0: Returns dict with all None values (sigmas estimated from HWHM)
    For pass > 0: Returns interpolated sigma fields from previous pass,
                  including all components for both A (autocorrelation) and
                  AB (cross-correlation) Gaussians.

    Note: NaN infilling is now handled uniformly in finalize_pass() for all
    fields (ux, uy, stresses, sigmas). Sigma fields passed here should
    already be infilled.

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
    dict with keys:
        'sig_AB_x', 'sig_AB_y', 'sig_AB_xy': Cross-correlation sigma components
        'sig_A_x', 'sig_A_y', 'sig_A_xy': Autocorrelation sigma components
        All values are np.ndarray (flattened) or None for pass 0
    """
    if pass_idx == 0:
        # Pass 0: Sigmas computed from HWHM in _build_initial_guess
        return {
            'sig_AB_x': None, 'sig_AB_y': None, 'sig_AB_xy': None,
            'sig_A_x': None, 'sig_A_y': None, 'sig_A_xy': None,
        }

    prev_pass = piv_results.passes[pass_idx - 1]
    old_h, old_w = prev_pass.sig_AB_x.shape
    new_h, new_w = n_win_y, n_win_x

    # Collect all sigma fields from previous pass
    sigma_fields = {
        'sig_AB_x': prev_pass.sig_AB_x.copy().astype(np.float32),
        'sig_AB_y': prev_pass.sig_AB_y.copy().astype(np.float32),
        'sig_AB_xy': prev_pass.sig_AB_xy.copy().astype(np.float32),
        'sig_A_x': prev_pass.sig_A_x.copy().astype(np.float32),
        'sig_A_y': prev_pass.sig_A_y.copy().astype(np.float32),
        'sig_A_xy': prev_pass.sig_A_xy.copy().astype(np.float32),
    }

    # Validate sigma fields before interpolation to prevent bad values from propagating
    # Use different bounds for different sigma types:
    # - sig_A (particle size): typically 0.5-15 pixels
    # - sig_AB (displacement uncertainty): typically 0.1-10 pixels
    # - sig_xy (cross-terms): can be negative, typically -10 to 10
    sigma_fields['sig_AB_x'] = _validate_sigma_field(sigma_fields['sig_AB_x'], 0.1, 15.0)
    sigma_fields['sig_AB_y'] = _validate_sigma_field(sigma_fields['sig_AB_y'], 0.1, 15.0)
    sigma_fields['sig_A_x'] = _validate_sigma_field(sigma_fields['sig_A_x'], 0.3, 20.0)
    sigma_fields['sig_A_y'] = _validate_sigma_field(sigma_fields['sig_A_y'], 0.3, 20.0)
    # Cross-terms can be negative, use symmetric bounds
    sigma_fields['sig_AB_xy'] = _validate_sigma_field(sigma_fields['sig_AB_xy'], -15.0, 15.0)
    sigma_fields['sig_A_xy'] = _validate_sigma_field(sigma_fields['sig_A_xy'], -20.0, 20.0)

    result = {}

    # Interpolate each field to current grid
    if (old_h, old_w) == (new_h, new_w):
        # Same grid size - no interpolation needed
        for key, field in sigma_fields.items():
            result[key] = field.ravel(order="C")
    else:
        # Different grid size - pad before interpolation to avoid edge zeros
        # Pad with 2 pixels on each side (matches cubic interpolation kernel requirement)
        # This mirrors velocity field handling in cpu_ensemble.py lines 878-882
        pad_size = 2
        for key, field in sigma_fields.items():
            # Pad field with edge values before interpolation (like velocity fields)
            padded = np.pad(field, pad_size, mode='edge')

            # Adjust interpolation coordinates to account for padding
            map_y, map_x = np.meshgrid(
                np.linspace(pad_size, old_h - 1 + pad_size, new_h).astype(np.float32),
                np.linspace(pad_size, old_w - 1 + pad_size, new_w).astype(np.float32),
                indexing="ij"
            )

            result[key] = cv2.remap(
                padded, map_x, map_y, cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE  # Extra safety for any remaining edge cases
            ).ravel(order="C")

    return result


def _estimate_offset(corr_plane: np.ndarray) -> float:
    """
    Estimate background offset from correlation plane.

    Uses 5th percentile for robustness against peak values and outliers.

    Parameters
    ----------
    corr_plane : np.ndarray
        Flattened correlation plane

    Returns
    -------
    float
        Estimated background offset value
    """
    return float(np.percentile(corr_plane, 5))


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
    # Extract parameters (16 total: 3 amps + 3 offsets + 6 sigmas + 4 positions)
    amp_A, amp_B, amp_AB = gauss_params[0:3]
    c_A, c_B, c_AB = gauss_params[3:6]  # offsets (unused in validation)
    sx_A, sy_A, sxy_A = gauss_params[6:9]
    sx_AB, sy_AB, sxy_AB = gauss_params[9:12]
    x0_A, y0_A = gauss_params[12:14]
    x0_AB, y0_AB = gauss_params[14:16]

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


def _fit_windows_batch_from_scattered(
    scattered_chunk: dict,
    win_size: tuple,
    config,
    pass_idx: int,
    scattered_cache: dict,
    outdir=None
):
    """
    Wrapper that unpacks a scattered chunk dict and calls the optimized fitter.

    This function allows correlation plane data to be pre-scattered to workers
    before task submission, avoiding large task graph serialization. When chunks
    are passed directly to client.submit(), they get embedded in the task graph.
    By scattering first and passing futures, only small references are in the graph.

    Parameters
    ----------
    scattered_chunk : dict
        Dictionary containing pre-scattered correlation plane chunks:
        - 'AA': Auto-correlation A chunk (flattened)
        - 'BB': Auto-correlation B chunk (flattened)
        - 'AB': Cross-correlation chunk (flattened)
        - 'mask': Boolean mask chunk for this worker's windows
        - 'sigma': Dict of sigma values for initial guesses
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
    tuple
        (results, statuses, initial_guesses) from _fit_windows_batch_optimized
    """
    return _fit_windows_batch_optimized(
        scattered_chunk['AA'],
        scattered_chunk['BB'],
        scattered_chunk['AB'],
        scattered_chunk['sigma'],
        scattered_chunk['mask'],
        win_size, config, pass_idx, scattered_cache, outdir
    )


def _fit_windows_batch_optimized(
    AA_chunk, BB_chunk, AB_chunk,
    sigma_chunk, mask_chunk,
    win_size, config, pass_idx, scattered_cache, outdir=None
):
    """
    Optimized Gaussian fitting with pre-chunked correlation planes.

    Uses batch C function with OpenMP parallelization for significant speedup.
    All data is pre-chunked per-worker before submission - no extraction needed.
    Uses sparse allocation: only allocates arrays for non-masked windows.

    At high resolution (4K+), correlation planes can reach GB in size.
    Pre-chunking avoids broadcasting full planes to all workers, reducing
    per-worker memory by ~87% with 8 workers.

    Parameters
    ----------
    AA_chunk, BB_chunk, AB_chunk : np.ndarray
        Pre-chunked correlation planes for this worker's windows only.
        Shape: (n_worker_windows * corr_h * corr_w,) flattened
    sigma_chunk : dict
        Per-worker sigma values with keys:
        'sig_AB_x', 'sig_AB_y', 'sig_AB_xy': Cross-correlation sigmas
        'sig_A_x', 'sig_A_y', 'sig_A_xy': Autocorrelation sigmas
        All values are np.ndarray (already chunked) or None for pass 0
    mask_chunk : np.ndarray
        Per-worker mask array (already chunked for this worker)
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
        Fitted parameters for each window (shape: num_windows, 16)
    statuses : np.ndarray
        Fitting status codes (shape: num_windows,)
    initial_guesses : np.ndarray
        Initial guesses used for fitting (shape: num_windows, 16)
    """
    marquadt_lib = _load_marquadt_lib()

    # All data is pre-chunked - no extraction needed
    # num_windows derived from mask_chunk length
    num_windows = len(mask_chunk)
    X1, X2, central_index, x_guess, y_guess = _get_pass_grid(pass_idx, config)

    valid_indices = np.where(~mask_chunk)[0]
    n_valid = len(valid_indices)

    if n_valid == 0:
        # All windows masked - return immediately with default values
        results = np.zeros((num_windows, 16), dtype=np.float64)
        statuses = np.full(num_windows, -1, dtype=np.int32)  # -1 = masked
        initial_guesses = np.zeros((num_windows, 16), dtype=np.float64)
        return results, statuses, initial_guesses

    n_per_window = win_size[0] * win_size[1]

    # Allocate batch arrays for valid windows only
    results_valid = np.zeros((n_valid, 16), dtype=np.float64)
    statuses_valid = np.zeros(n_valid, dtype=np.int32)
    initial_guesses_valid = np.zeros((n_valid, 16), dtype=np.float64)

    # Build batch arrays: y_all contains [AA|BB|AB] for each valid window
    y_all = np.zeros(n_valid * 3 * n_per_window, dtype=np.float64)

    # Build initial guesses and pack correlation data for all valid windows
    for i, idx in enumerate(valid_indices):
        # Extract window from correlation planes
        AA_win = _get_window(AA_chunk, idx, win_size)
        BB_win = _get_window(BB_chunk, idx, win_size)
        AB_win = _get_window(AB_chunk, idx, win_size)

        # Get sigma values for this window (all 6 components for pass > 0)
        sigma_vals = {}
        for key in ['sig_AB_x', 'sig_AB_y', 'sig_AB_xy', 'sig_A_x', 'sig_A_y', 'sig_A_xy']:
            if sigma_chunk[key] is not None:
                sigma_vals[key] = sigma_chunk[key][idx]
            else:
                sigma_vals[key] = None

        # Build initial guess
        initial_guess, real_corr = _build_initial_guess(
            idx, pass_idx, AA_win, BB_win, AB_win, central_index,
            x_guess, y_guess, sigma_vals,
            win_size, config
        )
        initial_guesses_valid[i] = initial_guess

        # Pack correlation data into batch array
        offset = i * 3 * n_per_window
        y_all[offset:offset + 3 * n_per_window] = real_corr

    # Call batch C function with OpenMP parallelization
    # Pass win_height and win_width separately to support rectangular windows
    success_count = marquadt_lib.fit_stacked_gaussian_batch_export(
        ctypes.c_size_t(n_valid),
        ctypes.c_size_t(n_per_window),
        ctypes.c_size_t(win_size[0]),  # win_height
        ctypes.c_size_t(win_size[1]),  # win_width
        X2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # Note: X2 is x-coord
        X1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # Note: X1 is y-coord
        y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        initial_guesses_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        results_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        statuses_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    # Post-process: validate fitted parameters (C returns 1 for success)
    for i, idx in enumerate(valid_indices):
        if statuses_valid[i] == 1:  # 1 = success from C code
            AA_win = _get_window(AA_chunk, idx, win_size)
            BB_win = _get_window(BB_chunk, idx, win_size)
            is_valid, nan_reason_code = _validate_fitted_params(
                results_valid[i], win_size, pass_idx,
                config.ensemble_type[pass_idx],
                tuple(config.ensemble_sum_window),
                float(AA_win[central_index]),
                float(BB_win[central_index])
            )
            if not is_valid:
                statuses_valid[i] = nan_reason_code
            else:
                statuses_valid[i] = 0  # Convert to 0 = success for Python

    # Expand back to full size for return
    results = np.zeros((num_windows, 16), dtype=np.float64)
    statuses = np.full(num_windows, -1, dtype=np.int32)  # -1 = masked default
    initial_guesses = np.zeros((num_windows, 16), dtype=np.float64)

    results[valid_indices] = results_valid
    statuses[valid_indices] = statuses_valid
    initial_guesses[valid_indices] = initial_guesses_valid

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
    x_guess, y_guess, sigma_vals, win_size, config
):
    """
    Build initial guess for Gaussian fitting.

    Displacement and amplitude guesses are ALWAYS found by locating peaks
    in the correlation planes (after image warping for pass > 0).
    - Displacement: Peak location in AB cross-correlation
    - Amplitude: Peak values at those locations

    Sigma guesses come from:
    - Pass 0: Computed from HWHM of correlation planes (all cross-terms = 0)
    - Pass > 0: Interpolated from previous pass (after outlier detection and infilling)
                All 6 components (sig_A_x/y/xy, sig_AB_x/y/xy) are used.

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
    sigma_vals : dict
        Dictionary with keys: 'sig_AB_x', 'sig_AB_y', 'sig_AB_xy',
                             'sig_A_x', 'sig_A_y', 'sig_A_xy'
        All values are float or None (None for pass 0)
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

    # Check if we have sigma values from previous pass
    has_prev_sigmas = (
        sigma_vals['sig_AB_x'] is not None and
        sigma_vals['sig_AB_y'] is not None
    )

    if pass_idx == 0 or not has_prev_sigmas:
        # Pass 0: Estimate sigmas from HWHM of correlation planes

        # Sigma A: from AA autocorrelation HWHM
        sigma_A_x, sigma_A_y, hwhm_A_x, hwhm_A_y = _estimate_sigma_from_plane(
            AA_win, central_index, win_size, central_index
        )
        # No cross-term for pass 0 (assume axis-aligned Gaussian)
        sigma_A_xy = 0.0

        # Sigma AB: Compute as HWHM_cross - HWHM_auto
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
        # No cross-term for pass 0 (assume axis-aligned Gaussian)
        sigma_AB_xy = 0.0
    else:
        # Pass > 0: Use interpolated values from previous pass
        # All 6 components are interpolated after outlier detection and infilling

        # Sigma A (autocorrelation) from previous pass
        sigma_A_x = float(sigma_vals['sig_A_x']) if sigma_vals['sig_A_x'] is not None else 1.0
        sigma_A_y = float(sigma_vals['sig_A_y']) if sigma_vals['sig_A_y'] is not None else 1.0
        sigma_A_xy = float(sigma_vals['sig_A_xy']) if sigma_vals['sig_A_xy'] is not None else 0.0

        # Sigma AB (cross-correlation) from previous pass
        sigma_AB_x = float(sigma_vals['sig_AB_x'])
        sigma_AB_y = float(sigma_vals['sig_AB_y'])
        sigma_AB_xy = float(sigma_vals['sig_AB_xy']) if sigma_vals['sig_AB_xy'] is not None else 0.0

    # Estimate offset values from correlation plane backgrounds (5th percentile)
    # OR zero if offset fitting is disabled
    if config.ensemble_fit_offset:
        c_A_guess = _estimate_offset(AA_win)
        c_B_guess = _estimate_offset(BB_win)
        c_AB_guess = _estimate_offset(AB_win)
    else:
        c_A_guess = 0.0
        c_B_guess = 0.0
        c_AB_guess = 0.0

    # Build 16-parameter initial guess:
    # [0-2] amplitudes, [3-5] offsets, [6-8] sigma_A, [9-11] sigma_AB, [12-15] positions
    initial_guess = np.array([
        float(AA_win[central_index]),    # [0] Amp A at center
        float(BB_win[central_index]),    # [1] Amp B at center
        float(AB_win[max_idx]),          # [2] Amp AB at peak (not center!)
        c_A_guess, c_B_guess, c_AB_guess,    # [3-5] Offsets (re-estimated each pass)
        sigma_A_x, sigma_A_y, sigma_A_xy,    # [6-8] Sigma A
        sigma_AB_x, sigma_AB_y, sigma_AB_xy, # [9-11] Sigma AB
        x_guess, y_guess,                    # [12-13] Center A (x, y)
        float(guess_x_AB + 1),               # [14] Center AB x (1-based indexing)
        float(guess_y_AB + 1),               # [15] Center AB y (1-based indexing)
    ], dtype=np.float64)

    real_corr = np.concatenate([AA_win, BB_win, AB_win]).astype(np.float64)
    return initial_guess, real_corr


def _build_initial_guesses_vectorized(
    R_AA: np.ndarray,
    R_BB: np.ndarray,
    R_AB: np.ndarray,
    valid_indices: np.ndarray,
    sigma_dict: dict,
    win_size: tuple,
    central_index: int,
    x_guess: float,
    y_guess: float,
    pass_idx: int,
    config: Config,
    num_windows:int,

) -> tuple:
    """
    Vectorized initial guess generation for all valid windows.

    This replaces the per-window loop with batch numpy operations for
    significant speedup (10-50x for typical window counts).

    Parameters
    ----------
    R_AA, R_BB, R_AB : np.ndarray
        Flattened correlation planes
    valid_indices : np.ndarray
        Indices of valid (non-masked) windows
    sigma_dict : dict
        Sigma values from previous pass (or all None for pass 0)
    win_size : tuple
        (height, width) of correlation window
    central_index : int
        Index of window center in flattened window
    x_guess, y_guess : float
        Center coordinates for auto-correlation peak
    pass_idx : int
        Current pass index
    config : Config
        Configuration object

    Returns
    -------
    initial_guesses : np.ndarray
        Initial parameters, shape (n_valid, 16)
    y_all : np.ndarray
        Packed correlation data for C function
    """
    n_valid = len(valid_indices)
    win_h, win_w = win_size
    n_per_window = win_h * win_w

    # Reshape correlation planes to (num_windows, win_h, win_w) for batch operations
    # First reshape to (total_windows, win_h * win_w), then extract valid and reshape
    total_windows = len(R_AA) // n_per_window

    # Extract valid windows in batch
    # Shape: (n_valid, n_per_window)
    AA_valid = R_AA.reshape(num_windows, n_per_window)[valid_indices]
    BB_valid = R_BB.reshape(num_windows, n_per_window)[valid_indices]
    AB_valid = R_AB.reshape(num_windows, n_per_window)[valid_indices]

    # Reshape to 3D for spatial operations: (n_valid, win_h, win_w)
    AA_3d = AA_valid.reshape(n_valid, win_h, win_w)
    BB_3d = BB_valid.reshape(n_valid, win_h, win_w)
    AB_3d = AB_valid.reshape(n_valid, win_h, win_w)

    # Vectorized peak finding in AB cross-correlation
    # Find peak location for each window
    AB_flat_view = AB_valid  # Shape: (n_valid, n_per_window)
    max_indices = np.argmax(AB_flat_view, axis=1)  # Shape: (n_valid,)

    # Convert flat indices to 2D coordinates
    peak_y = max_indices // win_w
    peak_x = max_indices % win_w

    # Extract values at peak locations (vectorized)
    amp_AB = AB_flat_view[np.arange(n_valid), max_indices]  # AB amplitude at peak
    amp_A = AA_valid[:, central_index]  # AA amplitude at center
    amp_B = BB_valid[:, central_index]  # BB amplitude at center

    # Check if we have sigma values from previous pass
    has_prev_sigmas = (
        sigma_dict['sig_AB_x'] is not None and
        sigma_dict['sig_AB_y'] is not None
    )

    if pass_idx == 0 or not has_prev_sigmas:
        # Pass 0: Estimate sigmas from HWHM of correlation planes
        # Data-driven initial guesses instead of fixed defaults.
        #
        # Uses Half-Width at Half-Maximum (HWHM) which is robust to background
        # and provides accurate estimates for Gaussian-shaped peaks.
        #
        # sigma_A: from AA autocorrelation (particle image size)
        # sigma_AB: TOTAL width from AB cross-correlation (NOT difference!)
        #           The C code now uses sigma_AB directly for cross-correlation model.

        # Sigma A: from AA autocorrelation at center (particle image size)
        central_indices = np.full(n_valid, central_index, dtype=np.int64)
        sigma_A_x, sigma_A_y, hwhm_A_x, hwhm_A_y = _estimate_sigma_batch_hwhm(
            AA_3d, central_indices, win_size, min_sigma=0.5
        )
        sigma_A_xy = np.zeros(n_valid, dtype=np.float64)  # Assume axis-aligned for pass 0

        # Sigma AB: TOTAL width from AB cross-correlation (used directly by C code)
        # This is the raw sigma from the cross-correlation peak, not the difference.
        sigma_AB_x, sigma_AB_y, hwhm_AB_x, hwhm_AB_y = _estimate_sigma_batch_hwhm(
            AB_3d, max_indices, win_size, min_sigma=0.5
        )
        sigma_AB_xy = np.zeros(n_valid, dtype=np.float64)  # Assume axis-aligned for pass 0

        # Apply reasonable bounds - extreme values indicate estimation failure
        sigma_A_x = np.clip(sigma_A_x, 0.5, 20.0)
        sigma_A_y = np.clip(sigma_A_y, 0.5, 20.0)
        # sigma_AB is total width, so should be >= sigma_A
        sigma_AB_x = np.clip(sigma_AB_x, 0.5, 25.0)
        sigma_AB_y = np.clip(sigma_AB_y, 0.5, 25.0)

        # Fallback: if HWHM estimation produces extreme values, use defaults
        # This handles cases where correlation planes are too noisy
        default_var_A = 3.0
        default_var_AB = 4.0  # Total width default (> sigma_A default)

        bad_A = (sigma_A_x > 15.0) | (sigma_A_y > 15.0) | \
                (sigma_A_x < 0.3) | (sigma_A_y < 0.3)
        bad_AB = (sigma_AB_x > 20.0) | (sigma_AB_y > 20.0) | \
                 (sigma_AB_x < 0.3) | (sigma_AB_y < 0.3)

        sigma_A_x = np.where(bad_A, default_var_A, sigma_A_x)
        sigma_A_y = np.where(bad_A, default_var_A, sigma_A_y)
        sigma_AB_x = np.where(bad_AB, default_var_AB, sigma_AB_x)
        sigma_AB_y = np.where(bad_AB, default_var_AB, sigma_AB_y)

        # Ensure sigma_AB >= sigma_A (physical constraint)
        sigma_AB_x = np.maximum(sigma_AB_x, sigma_A_x)
        sigma_AB_y = np.maximum(sigma_AB_y, sigma_A_y)
    else:
        # Pass > 0: Use interpolated values from previous pass
        sigma_A_x = sigma_dict['sig_A_x'][valid_indices].astype(np.float64)
        sigma_A_y = sigma_dict['sig_A_y'][valid_indices].astype(np.float64)
        sigma_A_xy = sigma_dict['sig_A_xy'][valid_indices].astype(np.float64) if sigma_dict['sig_A_xy'] is not None else np.zeros(n_valid)
        sigma_AB_x = sigma_dict['sig_AB_x'][valid_indices].astype(np.float64)
        sigma_AB_y = sigma_dict['sig_AB_y'][valid_indices].astype(np.float64)
        sigma_AB_xy = sigma_dict['sig_AB_xy'][valid_indices].astype(np.float64) if sigma_dict['sig_AB_xy'] is not None else np.zeros(n_valid)

        # Handle any NaN values with safe defaults
        sigma_A_x = np.where(np.isnan(sigma_A_x), 1.0, sigma_A_x)
        sigma_A_y = np.where(np.isnan(sigma_A_y), 1.0, sigma_A_y)
        sigma_A_xy = np.where(np.isnan(sigma_A_xy), 0.0, sigma_A_xy)
        sigma_AB_x = np.where(np.isnan(sigma_AB_x), 1.0, sigma_AB_x)
        sigma_AB_y = np.where(np.isnan(sigma_AB_y), 1.0, sigma_AB_y)
        sigma_AB_xy = np.where(np.isnan(sigma_AB_xy), 0.0, sigma_AB_xy)

    # Estimate offsets from 5th percentile (vectorized) OR zero if disabled
    # This represents the background level in each correlation plane
    if config.ensemble_fit_offset:
        c_A = np.percentile(AA_valid, 5, axis=1)
        c_B = np.percentile(BB_valid, 5, axis=1)
        c_AB = np.percentile(AB_valid, 5, axis=1)
    else:
        # Offsets disabled - set to zero
        c_A = np.zeros(n_valid, dtype=np.float64)
        c_B = np.zeros(n_valid, dtype=np.float64)
        c_AB = np.zeros(n_valid, dtype=np.float64)

    # CRITICAL: Amplitudes must be peak value ABOVE offset, not raw peak value!
    # Model: f(x) = amp * exp(...) + c
    # At peak: peak_value = amp * exp(0) + c = amp + c
    # Therefore: amp = peak_value - c
    # When offsets are zero, amp = peak_value (no correction needed)
    amp_A_corrected = amp_A - c_A
    amp_B_corrected = amp_B - c_B
    amp_AB_corrected = amp_AB - c_AB

    # Ensure amplitudes are positive (clamp to small positive value)
    amp_A_corrected = np.maximum(amp_A_corrected, 1e-6)
    amp_B_corrected = np.maximum(amp_B_corrected, 1e-6)
    amp_AB_corrected = np.maximum(amp_AB_corrected, 1e-6)

    # Build initial guesses array: (n_valid, 16)
    # [0-2] amplitudes, [3-5] offsets, [6-8] sigma_A, [9-11] sigma_AB, [12-15] positions
    initial_guesses = np.zeros((n_valid, 16), dtype=np.float64)
    initial_guesses[:, 0] = amp_A_corrected
    initial_guesses[:, 1] = amp_B_corrected
    initial_guesses[:, 2] = amp_AB_corrected
    initial_guesses[:, 3] = c_A
    initial_guesses[:, 4] = c_B
    initial_guesses[:, 5] = c_AB
    initial_guesses[:, 6] = sigma_A_x
    initial_guesses[:, 7] = sigma_A_y
    initial_guesses[:, 8] = sigma_A_xy
    initial_guesses[:, 9] = sigma_AB_x
    initial_guesses[:, 10] = sigma_AB_y
    initial_guesses[:, 11] = sigma_AB_xy
    initial_guesses[:, 12] = x_guess
    initial_guesses[:, 13] = y_guess
    initial_guesses[:, 14] = (peak_x + 1).astype(np.float64)  # 1-based indexing
    initial_guesses[:, 15] = (peak_y + 1).astype(np.float64)  # 1-based indexing

    # Pack correlation data: [AA|BB|AB] for each window
    y_all = np.zeros(n_valid * 3 * n_per_window, dtype=np.float64)
    for i in range(n_valid):
        offset = i * 3 * n_per_window
        y_all[offset:offset + n_per_window] = AA_valid[i]
        y_all[offset + n_per_window:offset + 2 * n_per_window] = BB_valid[i]
        y_all[offset + 2 * n_per_window:offset + 3 * n_per_window] = AB_valid[i]

    # Compute weights for AA/BB residuals (decoupled sigma fitting)
    # Weight = sqrt(sigma_AB / sigma_A) for variance normalization
    weights_auto = _compute_autocorrelation_weights(
        sigma_A_x, sigma_A_y, sigma_AB_x, sigma_AB_y
    )

    # Log weight statistics for verification
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"Decoupled fitting weights: min={weights_auto.min():.3f}, "
        f"max={weights_auto.max():.3f}, mean={weights_auto.mean():.3f}, "
        f"sigma_A_mean={np.mean(np.sqrt(sigma_A_x * sigma_A_y)):.2f}, "
        f"sigma_AB_mean={np.mean(np.sqrt(sigma_AB_x * sigma_AB_y)):.2f}"
    )

    return initial_guesses, y_all, weights_auto


def _estimate_sigma_batch_vectorized(
    corr_planes_3d: np.ndarray,
    peak_indices: np.ndarray,
    win_size: tuple,
    min_sigma: float = 0.5,
) -> tuple:
    """
    Vectorized sigma estimation using MOMENT-BASED method for correlation planes.

    Uses weighted second moment around the peak to estimate standard deviation:
        σ² = Σ(x - x_peak)² * w(x) / Σw(x)

    where w(x) is the correlation value (thresholded to positive values).
    This is more robust than HWHM for discrete/noisy data.

    Parameters
    ----------
    corr_planes_3d : np.ndarray
        Correlation planes, shape (n_windows, win_h, win_w)
    peak_indices : np.ndarray
        Flat indices of peaks in each window, shape (n_windows,)
    win_size : tuple
        (height, width) of window
    min_sigma : float
        Minimum sigma value (standard deviation)

    Returns
    -------
    sigma_x, sigma_y : np.ndarray
        Estimated standard deviations (NOT variance), shape (n_windows,)
    """
    n_windows = corr_planes_3d.shape[0]
    win_h, win_w = win_size

    # Convert flat indices to 2D coordinates
    peak_y = peak_indices // win_w
    peak_x = peak_indices % win_w

    # Get peak values
    peak_vals = corr_planes_3d[np.arange(n_windows), peak_y, peak_x]

    # Initialize sigma arrays
    sigma_x = np.full(n_windows, min_sigma, dtype=np.float64)
    sigma_y = np.full(n_windows, min_sigma, dtype=np.float64)

    # Coordinate arrays for moment calculation
    x_coords = np.arange(win_w, dtype=np.float64)
    y_coords = np.arange(win_h, dtype=np.float64)

    # Edge-based background estimation: use mean of border pixels
    n_edge = max(3, win_w // 8)

    for i in range(n_windows):
        if peak_vals[i] < 1e-6:
            continue

        py, px = int(peak_y[i]), int(peak_x[i])
        plane = corr_planes_3d[i]

        # X-direction: profile at peak_y with edge-based background
        x_profile = plane[py, :]
        bg_x = 0.5 * (np.mean(x_profile[:n_edge]) + np.mean(x_profile[-n_edge:]))
        x_shifted = x_profile - bg_x
        weights_x = np.maximum(x_shifted, 0)

        total_w = weights_x.sum()
        if total_w > 1e-10:
            centroid_x = np.sum(x_coords * weights_x) / total_w
            var_x = np.sum((x_coords - centroid_x)**2 * weights_x) / total_w
            sigma_x[i] = max(np.sqrt(var_x), min_sigma)

        # Y-direction: profile at peak_x with edge-based background
        y_profile = plane[:, px]
        bg_y = 0.5 * (np.mean(y_profile[:n_edge]) + np.mean(y_profile[-n_edge:]))
        y_shifted = y_profile - bg_y
        weights_y = np.maximum(y_shifted, 0)

        total_w = weights_y.sum()
        if total_w > 1e-10:
            centroid_y = np.sum(y_coords * weights_y) / total_w
            var_y = np.sum((y_coords - centroid_y)**2 * weights_y) / total_w
            sigma_y[i] = max(np.sqrt(var_y), min_sigma)

    return sigma_x, sigma_y


def _estimate_sigma_batch_hwhm(
    corr_planes_3d: np.ndarray,
    peak_indices: np.ndarray,
    win_size: tuple,
    min_sigma: float = 0.5,
) -> tuple:
    """
    Vectorized HWHM-based sigma estimation for correlation planes.

    Uses Half-Width at Half-Maximum (HWHM) to estimate Gaussian sigma:
        sigma = HWHM / sqrt(2 * ln(2))

    This is more robust than moment-based for peaks with clear structure
    and handles background subtraction implicitly (threshold = peak/2).

    Parameters
    ----------
    corr_planes_3d : np.ndarray
        Correlation planes, shape (n_windows, win_h, win_w)
    peak_indices : np.ndarray
        Flat indices of peaks in each window, shape (n_windows,)
    win_size : tuple
        (height, width) of window
    min_sigma : float
        Minimum sigma value (standard deviation)

    Returns
    -------
    sigma_x, sigma_y : np.ndarray
        Estimated standard deviations, shape (n_windows,)
    hwhm_x, hwhm_y : np.ndarray
        Raw HWHM values for debugging/subtraction, shape (n_windows,)
    """
    n_windows = corr_planes_3d.shape[0]
    win_h, win_w = win_size

    # Convert flat indices to 2D coordinates
    peak_y = peak_indices // win_w
    peak_x = peak_indices % win_w

    # Initialize outputs with defaults
    sigma_x = np.full(n_windows, min_sigma, dtype=np.float64)
    sigma_y = np.full(n_windows, min_sigma, dtype=np.float64)
    hwhm_x = np.full(n_windows, min_sigma * np.sqrt(2 * np.log(2)), dtype=np.float64)
    hwhm_y = np.full(n_windows, min_sigma * np.sqrt(2 * np.log(2)), dtype=np.float64)

    # HWHM to sigma conversion factor
    hwhm_to_sigma = 1.0 / np.sqrt(2 * np.log(2))  # ~0.849

    # Get peak values for all windows (vectorized)
    peak_vals = corr_planes_3d[np.arange(n_windows), peak_y, peak_x]

    # Process each window (loop required for profile thresholding)
    for i in range(n_windows):
        if peak_vals[i] < 1e-6:
            continue

        py, px = int(peak_y[i]), int(peak_x[i])
        plane = corr_planes_3d[i]
        threshold = peak_vals[i] / 2.0

        # X-direction: profile at peak_y
        x_profile = plane[py, :]
        x_above = np.where(x_profile >= threshold)[0]
        if len(x_above) >= 2:
            hwhm_x[i] = (x_above[-1] - x_above[0]) / 2.0
            sigma_x[i] = max(hwhm_x[i] * hwhm_to_sigma, min_sigma)

        # Y-direction: profile at peak_x
        y_profile = plane[:, px]
        y_above = np.where(y_profile >= threshold)[0]
        if len(y_above) >= 2:
            hwhm_y[i] = (y_above[-1] - y_above[0]) / 2.0
            sigma_y[i] = max(hwhm_y[i] * hwhm_to_sigma, min_sigma)

    return sigma_x, sigma_y, hwhm_x, hwhm_y


def _compute_autocorrelation_weights(
    sigma_A_x: np.ndarray,
    sigma_A_y: np.ndarray,
    sigma_AB_x: np.ndarray,
    sigma_AB_y: np.ndarray,
    min_weight: float = 0.5,
    max_weight: float = 5.0
) -> np.ndarray:
    """
    Compute weights for AA/BB residuals in decoupled sigma fitting.

    The weight is sqrt(sigma_AB / sigma_A), which normalizes the variance
    contribution of autocorrelation vs cross-correlation data. This ensures
    both data sources have comparable influence on the optimizer.

    When sigma_AB >> sigma_A (large displacement uncertainty), autocorrelation
    residuals should be weighted more heavily to preserve sigma_A accuracy.

    Parameters
    ----------
    sigma_A_x, sigma_A_y : np.ndarray
        Particle size sigma (from autocorrelation), shape (n_windows,)
    sigma_AB_x, sigma_AB_y : np.ndarray
        Total cross-correlation width (NOT additive term), shape (n_windows,)
    min_weight, max_weight : float
        Bounds to prevent extreme weights from destabilizing the optimizer

    Returns
    -------
    np.ndarray
        Per-window weights, shape (n_windows,)
    """
    # Use geometric mean of x and y components
    sigma_A_mean = np.sqrt(sigma_A_x * sigma_A_y)
    sigma_AB_mean = np.sqrt(sigma_AB_x * sigma_AB_y)

    # Avoid division by zero
    sigma_A_mean = np.maximum(sigma_A_mean, 0.1)

    # Weight = sqrt(sigma_AB / sigma_A)
    weights = np.sqrt(sigma_AB_mean / sigma_A_mean)

    return np.clip(weights, min_weight, max_weight)


def fit_windows_openmp(
    R_AA: np.ndarray,
    R_BB: np.ndarray,
    R_AB: np.ndarray,
    mask_flat: np.ndarray,
    sigma_dict: dict,
    corr_size: tuple,
    config,
    pass_idx: int,
    num_threads: int = None,
    fit_offset: bool = True,
) -> tuple:
    """
    Fit Gaussian peaks using pure OpenMP (no Dask overhead).

    This function is designed for use after correlation planes have been
    reduced to the main process. It bypasses Dask's scatter/submit/gather
    pattern and calls the C library directly with OpenMP parallelization.

    The C function fit_stacked_gaussian_batch_export() uses OpenMP internally
    with `#pragma omp parallel for schedule(dynamic, 16)`.

    IMPORTANT: This function uses ALL available CPU cores by default, not
    limited by Dask worker thread settings. This is appropriate because
    finalize_pass() runs on the main process after all Dask workers have
    returned their results.

    Parameters
    ----------
    R_AA : np.ndarray
        Flattened auto-correlation A plane
    R_BB : np.ndarray
        Flattened auto-correlation B plane
    R_AB : np.ndarray
        Flattened cross-correlation AB plane
    mask_flat : np.ndarray
        Boolean mask array (True = skip this window)
    sigma_dict : dict
        Sigma values from previous pass (or all None for pass 0)
        Keys: 'sig_AB_x', 'sig_AB_y', 'sig_AB_xy', 'sig_A_x', 'sig_A_y', 'sig_A_xy'
    corr_size : tuple
        (height, width) of correlation window
    config : Config
        Configuration object
    pass_idx : int
        Current pass index
    num_threads : int, optional
        Number of OpenMP threads. Defaults to ALL available CPU cores
        (os.cpu_count()), NOT limited by config.omp_threads which is for
        Dask workers.

    Returns
    -------
    gauss_flat : np.ndarray
        Fitted parameters, shape (num_windows, 16)
    status_flat : np.ndarray
        Status codes, shape (num_windows,)
        -1 = masked/skipped, 0 = success, >0 = error code
    initial_guess_flat : np.ndarray
        Initial guesses used, shape (num_windows, 16)
    """
    import os
    import logging

    logger = logging.getLogger(__name__)

    # Set OpenMP thread count - use ALL CPU cores by default since this runs
    # on the main process after Dask workers have finished
    if num_threads is not None:
        omp_threads = num_threads
    else:
        # Default to all available cores, not Dask's omp_threads setting
        omp_threads = os.cpu_count() or 4
    R_AA = np.asarray(R_AA)
    R_BB = np.asarray(R_BB)
    R_AB = np.asarray(R_AB)
    mask_flat = np.asarray(mask_flat)
    num_windows = len(mask_flat)
    
    if sigma_dict is not None:
        sigma_dict = {k: np.asarray(v) if v is not None else None for k, v in sigma_dict.items()}
    # Use the proper thread setter (calls omp_set_num_threads if available)
    marquadt_lib = _load_marquadt_lib()

    # Configure offset fitting for THIS worker process
    # Must be called on the worker, not main process, due to process isolation
    set_offset_fitting(fit_offset)

    # Configure center pixel masking (reads from config)
    # True = mask AA/BB center pixel to remove camera self-noise spike
    mask_center = getattr(config, 'ensemble_mask_center_pixel', True)
    set_center_masking(mask_center)

    # Get grid info
    win_size = corr_size  # (height, width)
    num_windows = len(mask_flat)
    X1, X2, central_index, x_guess, y_guess = _get_pass_grid(pass_idx, config)

    # Find valid (non-masked) windows
    mask_flat = np.asarray(mask_flat) 
    valid_indices = np.where(~mask_flat)[0]
    n_valid = len(valid_indices)

    logger.debug(f"fit_windows_openmp: Fitting {n_valid}/{num_windows} windows (pass {pass_idx + 1})")

    if n_valid == 0:
        # All windows masked - return immediately
        results = np.zeros((num_windows, 16), dtype=np.float64)
        statuses = np.full(num_windows, -1, dtype=np.int32)
        initial_guesses = np.zeros((num_windows, 16), dtype=np.float64)
        return results, statuses, initial_guesses

    n_per_window = win_size[0] * win_size[1]

    # Build initial guesses using vectorized implementation (much faster)
    # Also returns weights for decoupled sigma fitting
    logger.debug(f"fit_windows_openmp: Building initial guesses (vectorized)...{num_windows}")
    initial_guesses_valid, y_all, weights_auto = _build_initial_guesses_vectorized(
        R_AA, R_BB, R_AB, valid_indices, sigma_dict, win_size,
        central_index, x_guess, y_guess, pass_idx, config, num_windows=num_windows
    )

    # Allocate result arrays
    results_valid = np.zeros((n_valid, 16), dtype=np.float64)
    statuses_valid = np.zeros(n_valid, dtype=np.int32)

    # Call batch C function - OpenMP parallelizes internally
    # Pass win_height and win_width separately to support rectangular windows
    # Pass weights_auto for decoupled sigma fitting (variance normalization)
    logger.debug(f"fit_windows_openmp: Calling C batch function with {n_valid} windows")
    success_count = marquadt_lib.fit_stacked_gaussian_batch_export(
        ctypes.c_size_t(n_valid),
        ctypes.c_size_t(n_per_window),
        ctypes.c_size_t(win_size[0]),  # win_height
        ctypes.c_size_t(win_size[1]),  # win_width
        X2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # X2 is x-coord
        X1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # X1 is y-coord
        y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        initial_guesses_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        weights_auto.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # Per-window weights
        results_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        statuses_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    # Count C fitter successes and failures
    c_success = np.sum(statuses_valid == 1)
    c_failure = n_valid - c_success
    logger.info(
        f"fit_windows_openmp: C fitter results: {c_success}/{n_valid} succeeded, "
        f"{c_failure} failed ({c_failure/n_valid*100:.1f}% C failures)"
    )

    # Post-process: validate fitted parameters (vectorized where possible)
    # Get central values for validation
    total_windows = len(R_AA) // n_per_window
    AA_all = R_AA.reshape(total_windows, n_per_window)
    BB_all = R_BB.reshape(total_windows, n_per_window)
    AA_central_valid = AA_all[valid_indices, central_index]
    BB_central_valid = BB_all[valid_indices, central_index]

    # Validate each successful fit (C returns 1 for success)
    # Track rejection reasons for debugging
    rejection_counts = {2: 0, 3: 0, 5: 0}  # AB height, displacement rule, negative sigma

    for i in range(n_valid):
        if statuses_valid[i] == 1:  # 1 = success from C code
            is_valid, nan_reason_code = _validate_fitted_params(
                results_valid[i], win_size, pass_idx,
                config.ensemble_type[pass_idx],
                tuple(config.ensemble_sum_window),
                float(AA_central_valid[i]),
                float(BB_central_valid[i])
            )
            if not is_valid:
                statuses_valid[i] = nan_reason_code
                if nan_reason_code in rejection_counts:
                    rejection_counts[nan_reason_code] += 1
            else:
                statuses_valid[i] = 0  # Convert to 0 = success for Python

    # Log rejection breakdown at INFO level for visibility
    total_rejections = sum(rejection_counts.values())
    if total_rejections > 0:
        logger.info(
            f"fit_windows_openmp: Validation rejections ({total_rejections} total): "
            f"AB_height={rejection_counts[2]}, "
            f"displacement_rule={rejection_counts[3]}, "
            f"neg_sigma={rejection_counts[5]}"
        )

    # Expand back to full size
    results = np.zeros((num_windows, 16), dtype=np.float64)
    statuses = np.full(num_windows, -1, dtype=np.int32)  # -1 = masked default
    initial_guesses = np.zeros((num_windows, 16), dtype=np.float64)

    results[valid_indices] = results_valid
    statuses[valid_indices] = statuses_valid
    initial_guesses[valid_indices] = initial_guesses_valid

    # Log success rate
    successful_fits = np.sum(statuses == 0)
    if n_valid > 0:
        success_rate = successful_fits / n_valid
        logger.info(f"fit_windows_openmp: Success rate {success_rate:.1%} ({successful_fits}/{n_valid})")

    return results, statuses, initial_guesses