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

    # Single-window function (kept for compatibility)
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

    # Batch function with OpenMP parallelization
    marquadt_lib.fit_stacked_gaussian_batch_export.argtypes = [
        ctypes.c_size_t,  # num_windows
        ctypes.c_size_t,  # n_per_window
        ctypes.POINTER(ctypes.c_double),  # X1
        ctypes.POINTER(ctypes.c_double),  # X2
        ctypes.POINTER(ctypes.c_double),  # y_all
        ctypes.POINTER(ctypes.c_double),  # initial_guesses
        ctypes.POINTER(ctypes.c_double),  # out_params
        ctypes.POINTER(ctypes.c_int),     # out_statuses
    ]
    marquadt_lib.fit_stacked_gaussian_batch_export.restype = ctypes.c_int

    _marquadt_lib = marquadt_lib
    return marquadt_lib


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

    result = {}

    # Interpolate each field to current grid
    if (old_h, old_w) == (new_h, new_w):
        # Same grid size - no interpolation needed
        for key, field in sigma_fields.items():
            result[key] = field.ravel(order="C")
    else:
        # Different grid size - use cubic interpolation for smooth upsampling
        map_y, map_x = np.meshgrid(
            np.linspace(0, old_h - 1, new_h).astype(np.float32),
            np.linspace(0, old_w - 1, new_w).astype(np.float32),
            indexing="ij"
        )
        for key, field in sigma_fields.items():
            result[key] = cv2.remap(
                field, map_x, map_y, cv2.INTER_CUBIC
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
    success_count = marquadt_lib.fit_stacked_gaussian_batch_export(
        ctypes.c_size_t(n_valid),
        ctypes.c_size_t(n_per_window),
        X2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # Note: X2 is x-coord
        X1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # Note: X1 is y-coord
        y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        initial_guesses_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        results_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        statuses_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    # Post-process: validate fitted parameters
    for i, idx in enumerate(valid_indices):
        if statuses_valid[i] == 0:
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
    c_A_guess = _estimate_offset(AA_win)
    c_B_guess = _estimate_offset(BB_win)
    c_AB_guess = _estimate_offset(AB_win)

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