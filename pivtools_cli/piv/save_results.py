"""
Module for saving PIV results to .mat files compatible with post-processing code.
"""
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.io
from pivtools_core.config import Config
from pivtools_core.paths import get_data_paths

from pivtools_cli.piv.piv_result import (
    PIVResult, PIVPassResult,
    PIVEnsembleResult, PIVEnsemblePassResult,
)
from pivtools_cli.piv.gradient_correction import apply_gradient_correction_to_pass


def save_piv_result_distributed(
    piv_result: PIVResult,
    output_path: Path,
    frame_number: int,
    runs_to_save: Optional[List[int]] = None,
    vector_fmt: str = "B%05d.mat",
) -> str:
    """
    Save a PIV result to disk. Designed to be submitted to Dask workers.
    
    This function can be called on Dask workers to save results in parallel,
    avoiding the memory bottleneck of gathering all results to main.
    Memory-efficient: uses direct serialization without unnecessary copies.
    
    Parameters
    ----------
    piv_result : PIVResult
        The PIV result object containing one or more passes with complete data.
    output_path : Path
        Directory where the .mat file will be saved.
    frame_number : int
        Frame number (1-based) for the filename (e.g., 1 -> B00001.mat).
    runs_to_save : Optional[List[int]]
        List of pass indices (0-based) to save. If None, save all passes.
        For passes not in this list, empty arrays will be saved.
    vector_fmt : str
        Format string for the filename, e.g., "B%05d.mat".
        
    Returns
    -------
    str
        Path to the saved file (for verification/logging).
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = output_path / (vector_fmt % frame_number)

    if len(piv_result.passes) == 0:
        logging.warning(
            f"PIVResult has no passes for frame {frame_number}. "
            "Skipping save."
        )
        return str(filename)
    
    # Create single struct with arrays indexed by pass number
    # All data is already in piv_result, no external lists needed
    mat_data = _create_piv_struct_all_passes(piv_result, runs_to_save)
    
    # Save to .mat file with compression to reduce I/O
    scipy.io.savemat(filename, {"piv_result": mat_data}, oned_as="row", do_compression=True)
    logging.debug(f"Worker saved PIV result to {filename}")
    
    return str(filename)


def save_coordinates_from_config_distributed(
    config: Config,
    output_path: Path,
    correlator_cache: Optional[dict] = None,
    runs_to_save: Optional[List[int]] = None,
) -> str:
    """
    Generate and save coordinate grids. Designed for Dask workers.

    Parameters
    ----------
    config : Config
        Configuration object containing window sizes and overlap.
    output_path : Path
        Directory where coordinates.mat will be saved.
    correlator_cache : Optional[dict]
        Precomputed correlator cache to avoid redundant computation.
    runs_to_save : Optional[List[int]]
        List of pass indices (0-based) to save with data. If None, save all passes.
        For passes not in this list, empty coordinate grids will be saved.

    Returns
    -------
    str
        Path to the saved coordinates file.
    """
    from pivtools_cli.piv.piv_backend.cpu_instantaneous import (
        InstantaneousCorrelatorCPU
    )

    # Create a temporary correlator with optional precomputed cache
    correlator = InstantaneousCorrelatorCPU(config, precomputed_cache=correlator_cache)

    # Extract the cached window centers
    win_ctrs_x_list = correlator.win_ctrs_x
    win_ctrs_y_list = correlator.win_ctrs_y

    num_passes = len(config.window_sizes)

    if runs_to_save is None:
        runs_to_save = list(range(num_passes))

    # Create MATLAB-style struct array with fields 'x' and 'y', shape (num_passes,)
    dtype = [('x', object), ('y', object)]
    coords_struct = np.empty((num_passes,), dtype=dtype)

    # Get image height from correlator for coordinate conversion
    H = correlator.H

    for i in range(num_passes):
        if i in runs_to_save:
            x_centers = win_ctrs_x_list[i]
            y_centers = win_ctrs_y_list[i]

            # Convert pixel coords to physical coords (y=0 at bottom, increasing upward)
            # win_ctrs_y is in ascending pixel order [low_pixel_y, ..., high_pixel_y]
            # physical_y = (H-1) - pixel_y converts to descending physical [high_phys, ..., low_phys]
            # This already matches MATLAB convention: row 1 = highest y, row end = lowest y
            y_physical = (H - 1) - y_centers

            # Create meshgrid (MATLAB-style 1-based coords)
            x_grid, y_grid = np.meshgrid(x_centers + 1, y_physical + 1, indexing='xy')

            # Convert to half precision for space saving
            x_grid = _convert_to_half_precision(x_grid, 'x_grid')
            y_grid = _convert_to_half_precision(y_grid, 'y_grid')

            coords_struct['x'][i] = x_grid
            coords_struct['y'][i] = y_grid
        else:
            # Empty arrays for non-selected passes
            coords_struct['x'][i] = np.array([], dtype=np.float16)
            coords_struct['y'][i] = np.array([], dtype=np.float16)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = output_path / "coordinates.mat"
    scipy.io.savemat(filename, {"coordinates": coords_struct}, oned_as="row", do_compression=True)
    logging.info(f"Worker saved coordinates to {filename}")

    return str(filename)


def save_ensemble_result_distributed(
    ensemble_result: PIVEnsembleResult,
    output_path: Path,
    runs_to_save: Optional[List[int]] = None,
    filename: str = "ensemble_result.mat",
    gradient_correction: bool = False,
    image_height: Optional[int] = None,
) -> str:
    """
    Save an ensemble PIV result to disk. Designed to be submitted to Dask workers.

    This function saves the complete ensemble result (all passes) in a single file,
    since ensemble PIV processes all image pairs to produce one averaged result.

    Parameters
    ----------
    ensemble_result : PIVEnsembleResult
        The ensemble PIV result object containing one or more passes with complete data.
    output_path : Path
        Directory where the .mat file will be saved.
    runs_to_save : Optional[List[int]]
        List of pass indices (0-based) to save. If None, save all passes.
        For passes not in this list, empty arrays will be saved.
    filename : str
        Name of the output file.
    gradient_correction : bool
        If True, apply gradient correction to Reynolds stresses before saving.
    image_height : int, optional
        Image height for coordinate conversion. Required if gradient_correction=True.

    Returns
    -------
    str
        Path to the saved file (for verification/logging).
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename

    if len(ensemble_result.passes) == 0:
        logging.warning(
            f"PIVEnsembleResult has no passes. Skipping save."
        )
        return str(filepath)

    # Create single struct with arrays indexed by pass number
    mat_data = _create_ensemble_struct_all_passes(
        ensemble_result, runs_to_save, gradient_correction, image_height
    )

    # Save to .mat file with compression to reduce I/O
    scipy.io.savemat(filepath, {"ensemble_result": mat_data}, oned_as="row", do_compression=True)
    logging.info(f"Saved ensemble result to {filepath}")

    return str(filepath)


def save_ensemble_coordinates_from_config_distributed(
    config: Config,
    output_path: Path,
    correlator_cache: Optional[dict] = None,
    runs_to_save: Optional[List[int]] = None,
) -> str:
    """
    Generate and save coordinate grids for ensemble PIV. Designed for Dask workers.

    Parameters
    ----------
    config : Config
        Configuration object containing ensemble window sizes and overlap.
    output_path : Path
        Directory where coordinates.mat will be saved.
    correlator_cache : Optional[dict]
        Precomputed correlator cache to avoid redundant computation.
    runs_to_save : Optional[List[int]]
        List of pass indices (0-based) to save with data. If None, save all passes.

    Returns
    -------
    str
        Path to the saved coordinates file.
    """
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    # Create correlator with optional precomputed cache
    # This handles both standard and single mode correctly
    correlator = EnsembleCorrelatorCPU(config, precomputed_cache=correlator_cache)

    # Extract the cached window centers (computed in _compute_window_centres_ensemble)
    # These are CORRECT for both standard and single mode
    win_ctrs_x_list = correlator.win_ctrs_x
    win_ctrs_y_list = correlator.win_ctrs_y

    # Get padding info per pass (for single mode coordinate conversion)
    padding_per_pass = getattr(correlator, 'padding_per_pass', [])

    num_passes = config.ensemble_num_passes

    if runs_to_save is None:
        runs_to_save = list(range(num_passes))

    # Create MATLAB-style struct array with fields 'x' and 'y', shape (num_passes,)
    dtype = [('x', object), ('y', object)]
    coords_struct = np.empty((num_passes,), dtype=dtype)

    # Get image height from correlator for coordinate conversion
    H = correlator.H

    for i in range(num_passes):
        if i in runs_to_save:
            # Use cached window centers from correlator (handles single mode)
            x_centers = win_ctrs_x_list[i]
            y_centers = win_ctrs_y_list[i]

            # Get padding for this pass (for single mode, coords are in padded space)
            if padding_per_pass and i < len(padding_per_pass):
                pad_top, pad_bottom, pad_left, pad_right = padding_per_pass[i]
            else:
                pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

            # For single mode: convert from padded coords to original image coords
            # by subtracting the top/left padding offsets
            y_centers_original = y_centers - pad_top
            x_centers_original = x_centers - pad_left

            # Convert pixel coords to physical coords (y=0 at bottom, increasing upward)
            # win_ctrs_y is in ascending pixel order [low_pixel_y, ..., high_pixel_y]
            # physical_y = (H-1) - pixel_y converts to descending physical [high_phys, ..., low_phys]
            # This already matches MATLAB convention: row 1 = highest y, row end = lowest y
            y_physical = (H - 1) - y_centers_original

            # Create meshgrid (MATLAB-style 1-based coords)
            x_grid, y_grid = np.meshgrid(x_centers_original + 1, y_physical + 1, indexing='xy')

            # Convert to half precision for space saving
            x_grid = _convert_to_half_precision(x_grid, 'x_grid')
            y_grid = _convert_to_half_precision(y_grid, 'y_grid')

            coords_struct['x'][i] = x_grid
            coords_struct['y'][i] = y_grid
        else:
            # Empty arrays for non-selected passes
            coords_struct['x'][i] = np.array([], dtype=np.float16)
            coords_struct['y'][i] = np.array([], dtype=np.float16)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = output_path / "coordinates.mat"
    scipy.io.savemat(filename, {"coordinates": coords_struct}, oned_as="row", do_compression=True)
    logging.info(f"Saved ensemble coordinates to {filename}")

    return str(filename)


def _create_ensemble_struct_all_passes(
    ensemble_result: PIVEnsembleResult,
    runs_to_save: Optional[List[int]] = None,
    gradient_correction: bool = False,
    image_height: Optional[int] = None,
) -> np.ndarray:
    """
    Create a MATLAB-compatible struct array with one element per pass for ensemble results.

    This creates a struct ARRAY (N×1) where each element represents one pass,
    matching the instantaneous format:
        ensemble_result(1).ux = pass1_ux_matrix
        ensemble_result(2).ux = pass2_ux_matrix

    Parameters
    ----------
    ensemble_result : PIVEnsembleResult
        Ensemble PIV result object containing one or more passes with complete data.
    runs_to_save : Optional[List[int]]
        List of pass indices (0-based) to save with data. If None, save all passes.
    gradient_correction : bool
        If True, apply gradient correction to Reynolds stresses.
    image_height : int, optional
        Image height for coordinate conversion. Required if gradient_correction=True.

    Returns
    -------
    np.ndarray
        Structured numpy array (struct array) compatible with scipy.io.savemat.
    """
    n_passes = len(ensemble_result.passes)

    # Always save all passes, but empty arrays for non-selected passes
    passes_to_save = list(range(n_passes))

    # If runs_to_save is specified, only fill data for those passes
    if runs_to_save is None:
        runs_to_save = passes_to_save

    # Get dtype from first pass for creating empty arrays
    first_pass = ensemble_result.passes[0]
    if first_pass.ux_mat is not None and first_pass.ux_mat.size > 0:
        data_dtype = first_pass.ux_mat.dtype
    else:
        data_dtype = np.float64

    # Create structured dtype with all ensemble fields
    dtype = [
        ('ux', object),
        ('uy', object),
        ('UU_stress', object),
        ('VV_stress', object),
        ('UV_stress', object),
        ('peakheight', object),
        ('nan_reason', object),
        ('sig_AB_x', object),
        ('sig_AB_y', object),
        ('sig_AB_xy', object),
        ('sig_A_x', object),
        ('sig_A_y', object),
        ('sig_A_xy', object),
        ('c_A', object),
        ('c_B', object),
        ('c_AB', object),
        ('win_ctrs_x', object),
        ('win_ctrs_y', object),
        ('window_size', object),
        ('b_mask', object),
        ('pred_x', object),
        ('pred_y', object),
    ]

    # Create struct ARRAY with one element per pass (like instantaneous)
    ensemble_struct = np.empty((n_passes,), dtype=dtype)

    # Initialize all passes with empty arrays
    empty = np.empty((0, 0), dtype=data_dtype)
    for i in range(n_passes):
        ensemble_struct['ux'][i] = empty
        ensemble_struct['uy'][i] = empty
        ensemble_struct['UU_stress'][i] = empty
        ensemble_struct['VV_stress'][i] = empty
        ensemble_struct['UV_stress'][i] = empty
        ensemble_struct['peakheight'][i] = empty
        ensemble_struct['nan_reason'][i] = empty
        ensemble_struct['sig_AB_x'][i] = empty
        ensemble_struct['sig_AB_y'][i] = empty
        ensemble_struct['sig_AB_xy'][i] = empty
        ensemble_struct['sig_A_x'][i] = empty
        ensemble_struct['sig_A_y'][i] = empty
        ensemble_struct['sig_A_xy'][i] = empty
        ensemble_struct['c_A'][i] = empty
        ensemble_struct['c_B'][i] = empty
        ensemble_struct['c_AB'][i] = empty
        ensemble_struct['win_ctrs_x'][i] = empty
        ensemble_struct['win_ctrs_y'][i] = empty
        ensemble_struct['window_size'][i] = empty
        ensemble_struct['b_mask'][i] = empty
        ensemble_struct['pred_x'][i] = empty
        ensemble_struct['pred_y'][i] = empty

    # Fill with actual data for selected passes
    for local_idx, global_pass_idx in enumerate(passes_to_save):
        if global_pass_idx not in runs_to_save:
            continue  # Skip filling for non-selected passes
        pass_result = ensemble_result.passes[global_pass_idx]

        # Velocity fields - no row reversal needed since win_ctrs_y is now ascending
        # pixel→physical conversion already produces correct row order for MATLAB
        # Negate uy because +uy in image coords means downward, but we want +uy = upward
        ux_physical = pass_result.ux_mat
        uy_physical = -pass_result.uy_mat if pass_result.uy_mat is not None else None

        if ux_physical is not None:
            ensemble_struct['ux'][local_idx] = _convert_to_half_precision(ux_physical, 'ux')
        if uy_physical is not None:
            ensemble_struct['uy'][local_idx] = _convert_to_half_precision(uy_physical, 'uy')

        # Stress tensors - no row reversal needed
        # UV_stress is negated because V is negated (UV = u'v' -> u'(-v') = -u'v')
        UU_to_save = pass_result.UU_stress
        VV_to_save = pass_result.VV_stress
        UV_to_save = -pass_result.UV_stress if pass_result.UV_stress is not None else None

        # Apply gradient correction if enabled
        if gradient_correction and pass_result.sig_A_x is not None:
            logging.info(f"Applying gradient correction to pass {global_pass_idx + 1}")
            UU_to_save, VV_to_save, UV_to_save = apply_gradient_correction_to_pass(
                ux=ux_physical,
                uy=uy_physical,
                UU_stress=UU_to_save,
                VV_stress=VV_to_save,
                UV_stress=UV_to_save,
                sig_A_x=pass_result.sig_A_x,
                sig_A_y=pass_result.sig_A_y,
                win_ctrs_x=pass_result.win_ctrs_x,
                win_ctrs_y=pass_result.win_ctrs_y,
                image_height=image_height if image_height else 0,
            )

        if UU_to_save is not None:
            ensemble_struct['UU_stress'][local_idx] = _convert_to_half_precision(UU_to_save, 'UU_stress')
        if VV_to_save is not None:
            ensemble_struct['VV_stress'][local_idx] = _convert_to_half_precision(VV_to_save, 'VV_stress')
        if UV_to_save is not None:
            ensemble_struct['UV_stress'][local_idx] = _convert_to_half_precision(UV_to_save, 'UV_stress')

        # Normalized peak height - no row reversal needed
        if pass_result.peakheight is not None:
            ensemble_struct['peakheight'][local_idx] = _convert_to_half_precision(pass_result.peakheight, 'peakheight')

        # NaN reason - no row reversal needed
        if pass_result.nan_reason is not None:
            ensemble_struct['nan_reason'][local_idx] = pass_result.nan_reason

        # Sigma parameters (AB) - no row reversal needed
        if pass_result.sig_AB_x is not None:
            ensemble_struct['sig_AB_x'][local_idx] = _convert_to_half_precision(pass_result.sig_AB_x, 'sig_AB_x')
        if pass_result.sig_AB_y is not None:
            ensemble_struct['sig_AB_y'][local_idx] = _convert_to_half_precision(pass_result.sig_AB_y, 'sig_AB_y')
        if pass_result.sig_AB_xy is not None:
            ensemble_struct['sig_AB_xy'][local_idx] = _convert_to_half_precision(pass_result.sig_AB_xy, 'sig_AB_xy')

        # Sigma parameters (A) - no row reversal needed
        if pass_result.sig_A_x is not None:
            ensemble_struct['sig_A_x'][local_idx] = _convert_to_half_precision(pass_result.sig_A_x, 'sig_A_x')
        if pass_result.sig_A_y is not None:
            ensemble_struct['sig_A_y'][local_idx] = _convert_to_half_precision(pass_result.sig_A_y, 'sig_A_y')
        if pass_result.sig_A_xy is not None:
            ensemble_struct['sig_A_xy'][local_idx] = _convert_to_half_precision(pass_result.sig_A_xy, 'sig_A_xy')

        # Gaussian offset parameters (background levels) - no row reversal needed
        if pass_result.c_A is not None:
            ensemble_struct['c_A'][local_idx] = _convert_to_half_precision(pass_result.c_A, 'c_A')
        if pass_result.c_B is not None:
            ensemble_struct['c_B'][local_idx] = _convert_to_half_precision(pass_result.c_B, 'c_B')
        if pass_result.c_AB is not None:
            ensemble_struct['c_AB'][local_idx] = _convert_to_half_precision(pass_result.c_AB, 'c_AB')

        # Window centers and size (1D arrays)
        if pass_result.win_ctrs_x is not None:
            ensemble_struct['win_ctrs_x'][local_idx] = _convert_to_half_precision(pass_result.win_ctrs_x, 'win_ctrs_x')
        if pass_result.win_ctrs_y is not None:
            ensemble_struct['win_ctrs_y'][local_idx] = _convert_to_half_precision(pass_result.win_ctrs_y, 'win_ctrs_y')
        if pass_result.window_size is not None:
            ensemble_struct['window_size'][local_idx] = pass_result.window_size

        # Mask - no row reversal needed
        if pass_result.b_mask is not None:
            ensemble_struct['b_mask'][local_idx] = pass_result.b_mask.astype(bool)

        # Predictor fields - no row reversal needed, negate y component for physical coords
        if pass_result.pred_x is not None:
            ensemble_struct['pred_x'][local_idx] = _convert_to_half_precision(pass_result.pred_x, 'pred_x')
        if pass_result.pred_y is not None:
            ensemble_struct['pred_y'][local_idx] = _convert_to_half_precision(-pass_result.pred_y, 'pred_y')

    return ensemble_struct


def _create_piv_struct_all_passes(
    piv_result: PIVResult,
    runs_to_save: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Create a MATLAB-compatible struct with arrays indexed by pass number.
    
    This creates a single struct where each field (ux, uy, b_mask, etc.) is
    an array with one element per pass, matching the expected format:
        piv_result["ux"][pass_idx] = 2D array for that pass
    
    All required data (including window centers and masks) is extracted from
    the PIVResult object, which contains all necessary information in each
    PIVPassResult.
    
    Parameters
    ----------
    piv_result : PIVResult
        PIV result object containing one or more passes with complete data.
    runs_to_save : Optional[List[int]]
        List of pass indices (0-based) to save with data. If None, save all passes.
        For passes not in this list, empty arrays will be saved.
        
    Returns
    -------
    np.ndarray
        Structured numpy array compatible with scipy.io.savemat.
    """
    n_passes = len(piv_result.passes)
    
    # Always save all passes, but empty arrays for non-selected passes
    n_passes_to_save = n_passes
    passes_to_save = list(range(n_passes))
    
    # If runs_to_save is specified, only fill data for those passes
    if runs_to_save is None:
        runs_to_save = passes_to_save
    
    # Create structured dtype with all fields
    dtype = [
        ('ux', object),
        ('uy', object),
        ('b_mask', object),
        ('nan_mask', object),
        ('win_ctrs_x', object),
        ('win_ctrs_y', object),
        ('peak_mag', object),
        ('peak_choice', object),
        ('n_windows', object),
        ('predictor_field', object),
        ('window_size', object),
    ]
    
    # Create the struct with shape (n_passes_to_save,)
    piv_struct = np.empty((n_passes_to_save,), dtype=dtype)
    
    # Get dtype from first pass for creating empty arrays
    first_pass = piv_result.passes[0]
    if first_pass.ux_mat is not None and first_pass.ux_mat.size > 0:
        data_dtype = first_pass.ux_mat.dtype
    else:
        data_dtype = np.float64
    
    # Initialize all passes with empty arrays
    empty = np.empty((0, 0), dtype=data_dtype)
    for i in range(n_passes_to_save):
        piv_struct['ux'][i] = empty
        piv_struct['uy'][i] = empty
        piv_struct['b_mask'][i] = empty
        piv_struct['nan_mask'][i] = empty
        piv_struct['win_ctrs_x'][i] = empty
        piv_struct['win_ctrs_y'][i] = empty
        piv_struct['peak_mag'][i] = empty
        piv_struct['peak_choice'][i] = empty
        piv_struct['n_windows'][i] = empty
        piv_struct['predictor_field'][i] = empty
        piv_struct['window_size'][i] = empty
    
    # Fill with actual data for selected passes
    for local_idx, global_pass_idx in enumerate(passes_to_save):
        if global_pass_idx not in runs_to_save:
            continue  # Skip filling for non-selected passes
        pass_result = piv_result.passes[global_pass_idx]

        # No row reversal needed since win_ctrs_y is now ascending
        # pixel→physical conversion already produces correct row order for MATLAB
        # Negate uy because +uy in image coords means downward, but we want +uy = upward
        if pass_result.ux_mat is not None:
            piv_struct['ux'][local_idx] = _convert_to_half_precision(pass_result.ux_mat, 'ux')
        if pass_result.uy_mat is not None:
            piv_struct['uy'][local_idx] = _convert_to_half_precision(-pass_result.uy_mat, 'uy')

        # Use b_mask from pass_result (no row reversal needed)
        if pass_result.b_mask is not None:
            piv_struct['b_mask'][local_idx] = pass_result.b_mask
        elif pass_result.nan_mask is not None:
            # Fallback to nan_mask if b_mask not available
            piv_struct['b_mask'][local_idx] = pass_result.nan_mask

        if pass_result.nan_mask is not None:
            piv_struct['nan_mask'][local_idx] = pass_result.nan_mask

        # Window centers are 1D arrays
        if pass_result.win_ctrs_x is not None:
            piv_struct['win_ctrs_x'][local_idx] = _convert_to_half_precision(pass_result.win_ctrs_x, 'win_ctrs_x')
        if pass_result.win_ctrs_y is not None:
            piv_struct['win_ctrs_y'][local_idx] = _convert_to_half_precision(pass_result.win_ctrs_y, 'win_ctrs_y')

        if pass_result.peak_mag is not None:
            piv_struct['peak_mag'][local_idx] = _convert_to_half_precision(pass_result.peak_mag, 'peak_mag')
        if pass_result.peak_choice is not None:
            piv_struct['peak_choice'][local_idx] = pass_result.peak_choice if pass_result.peak_choice.ndim == 2 else pass_result.peak_choice
        if pass_result.n_windows is not None:
            piv_struct['n_windows'][local_idx] = pass_result.n_windows
        # predictor_field shape is (n_win_y+pad, n_win_x+pad, 2) where dim 2 is [uy, ux]
        # No row reversal needed, negate uy component (index 0) for physical coords
        if pass_result.predictor_field is not None:
            pred = pass_result.predictor_field.copy()
            pred[..., 0] = -pred[..., 0]  # Negate uy component
            piv_struct['predictor_field'][local_idx] = _convert_to_half_precision(pred, 'predictor_field')
        if pass_result.window_size is not None:
            piv_struct['window_size'][local_idx] = pass_result.window_size
    
    return piv_struct


# Note: get_data_paths is imported from src/paths.py at the top of this file


def get_output_path(
    config: Config,
    camera: Union[int, str],
    create: bool = True,
    use_uncalibrated: bool = True,
    piv_type: Optional[str] = None,
    base_path_idx: int = 0,
) -> Path:
    """
    Get the output path for a specific camera's PIV results using the GUI path structure.

    Follows the standardized directory structure:
    - Uncalibrated: base_path/uncalibrated_piv/{num_images}/Cam{camera}/instantaneous
    - Calibrated: base_path/calibrated_piv/{num_images}/Cam{camera}/instantaneous

    Parameters
    ----------
    config : Config
        Configuration object.
    camera : Union[int, str]
        Camera number (int) or camera folder name (str, e.g., "Cam1").
    create : bool
        If True, create the directory if it doesn't exist.
    use_uncalibrated : bool
        If True, save to uncalibrated_piv directory.
        If False, save to calibrated_piv directory.
    piv_type : Optional[str]
        Override the PIV type ("instantaneous" or "ensemble"). If None, determine from config.
    base_path_idx : int
        Index into config.base_paths to use. Defaults to 0.

    Returns
    -------
    Path
        Output path for PIV results.
    """
    base_path = config.base_paths[base_path_idx]

    # Convert camera to int if it's a string
    if isinstance(camera, str):
        if camera.startswith("Cam"):
            camera_num = int(camera[3:])
        else:
            camera_num = int(camera)
    else:
        camera_num = camera

    # Get PIV type - default to instantaneous if not specified
    if piv_type is None:
        piv_type = "instantaneous" if config.data.get("processing", {}).get("instantaneous", True) else "ensemble"

    # Use get_data_paths from src/paths.py (positional args: base_dir, num_images, cam, type_name)
    paths = get_data_paths(
        base_path,
        config.num_frame_pairs,
        camera_num,
        piv_type,
        endpoint="",
        use_uncalibrated=use_uncalibrated
    )

    output_path = paths["data_dir"]

    if create:
        output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def get_ensemble_output_path(
    config: Config,
    camera: Union[int, str],
    create: bool = True,
    use_uncalibrated: bool = True,
    base_path_idx: int = 0,
) -> Path:
    """
    Get the output path for ensemble PIV results.

    Convenience function that calls get_output_path with piv_type="ensemble".

    Parameters
    ----------
    config : Config
        Configuration object.
    camera : Union[int, str]
        Camera number (int) or camera folder name (str, e.g., "Cam1").
    create : bool
        If True, create the directory if it doesn't exist.
    use_uncalibrated : bool
        If True, save to uncalibrated_piv directory.
    base_path_idx : int
        Index into config.base_paths to use. Defaults to 0.

    Returns
    -------
    Path
        Output path for ensemble PIV results.
    """
    return get_output_path(
        config,
        camera,
        create=create,
        use_uncalibrated=use_uncalibrated,
        piv_type="ensemble",
        base_path_idx=base_path_idx,
    )


def _convert_to_half_precision(arr: np.ndarray, field_name: str = "unknown") -> np.ndarray:
    """
    Convert float arrays to half precision (float16) for space saving.

    Clamps values to float16 range to prevent overflow warnings.
    float16 max is ~65504.
    """
    if arr is None or arr.size == 0:
        return arr
    if arr.dtype.kind == 'f':
        # float16 range: approximately [-65504, 65504]
        FLOAT16_MAX = 65504.0

        # Check for values that would overflow
        finite_vals = arr[np.isfinite(arr)]
        if finite_vals.size > 0:
            max_abs = np.abs(finite_vals).max()
            if max_abs > FLOAT16_MAX:
                logging.warning(
                    f"Field '{field_name}' has values exceeding float16 range: "
                    f"max |value| = {max_abs:.2e}, clamping to ±{FLOAT16_MAX}"
                )

        # Clamp to float16 range before conversion
        clamped = np.clip(arr, -FLOAT16_MAX, FLOAT16_MAX)
        # Preserve NaN values (clip converts them to the clamp boundary)
        clamped = np.where(np.isnan(arr), np.nan, clamped)
        return clamped.astype(np.float16)
    return arr


# =============================================================================
# Load Functions for Resume from Pass
# =============================================================================


def load_ensemble_result(
    file_path: Path,
    passes_to_load: Optional[List[int]] = None,
) -> Tuple[PIVEnsembleResult, int]:
    """
    Load ensemble result from .mat file for resume functionality.

    Parameters
    ----------
    file_path : Path
        Path to ensemble_result.mat file
    passes_to_load : Optional[List[int]]
        List of pass indices (0-based) to load. If None, load all passes.

    Returns
    -------
    tuple
        (PIVEnsembleResult, n_passes_loaded)

    Notes
    -----
    Sign conventions are reversed during load to match save conventions:
    - uy was negated on save -> negate back on load
    - UV_stress was negated on save -> negate back on load
    - pred_y was negated on save -> negate back on load
    """
    mat = scipy.io.loadmat(str(file_path), struct_as_record=False, squeeze_me=True)

    if "ensemble_result" not in mat:
        raise KeyError(f"'ensemble_result' not found in {file_path}")

    data = mat["ensemble_result"]
    ensemble_result = PIVEnsembleResult()

    # Handle array of structs (multiple passes)
    if isinstance(data, np.ndarray) and data.dtype == object and data.size > 0:
        n_passes = data.size
        if passes_to_load is None:
            passes_to_load = list(range(n_passes))

        for pass_idx in passes_to_load:
            if pass_idx >= n_passes:
                logging.warning(f"Pass {pass_idx} not found in file (has {n_passes} passes)")
                continue
            struct = data.flat[pass_idx]
            pass_result = _struct_to_ensemble_pass_result(struct)
            ensemble_result.add_pass(pass_result)
    else:
        # Single pass case
        if passes_to_load is None or 0 in passes_to_load:
            pass_result = _struct_to_ensemble_pass_result(data)
            ensemble_result.add_pass(pass_result)

    logging.info(f"Loaded {len(ensemble_result.passes)} passes from {file_path}")
    return ensemble_result, len(ensemble_result.passes)


def _struct_to_ensemble_pass_result(struct) -> PIVEnsemblePassResult:
    """
    Convert MATLAB struct to PIVEnsemblePassResult.

    Reverses sign conventions applied during save.
    """
    def get_array(name: str) -> Optional[np.ndarray]:
        """Safely get attribute and convert to numpy array."""
        val = getattr(struct, name, None)
        if val is None:
            return None
        arr = np.asarray(val)
        if arr.size == 0:
            return None
        # Convert from float16 back to float32 for computation
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        return arr

    # Core velocity fields - uy was negated on save
    ux = get_array('ux')
    uy = get_array('uy')
    if uy is not None:
        uy = -uy  # Reverse negation from save

    # Stress tensors - UV_stress was negated on save
    UU_stress = get_array('UU_stress')
    VV_stress = get_array('VV_stress')
    UV_stress = get_array('UV_stress')
    if UV_stress is not None:
        UV_stress = -UV_stress  # Reverse negation from save

    # Predictor fields - pred_y was negated on save
    pred_x = get_array('pred_x')
    pred_y = get_array('pred_y')
    if pred_y is not None:
        pred_y = -pred_y  # Reverse negation from save

    # Window size (tuple)
    window_size_val = getattr(struct, 'window_size', None)
    window_size = None
    if window_size_val is not None:
        ws = np.asarray(window_size_val).flatten()
        if ws.size >= 2:
            window_size = (int(ws[0]), int(ws[1]))

    # Mask
    b_mask = get_array('b_mask')
    if b_mask is not None:
        b_mask = b_mask.astype(bool)

    return PIVEnsemblePassResult(
        ux_mat=ux,
        uy_mat=uy,
        UU_stress=UU_stress,
        VV_stress=VV_stress,
        UV_stress=UV_stress,
        peakheight=get_array('peakheight'),
        nan_reason=get_array('nan_reason'),
        sig_AB_x=get_array('sig_AB_x'),
        sig_AB_y=get_array('sig_AB_y'),
        sig_AB_xy=get_array('sig_AB_xy'),
        sig_A_x=get_array('sig_A_x'),
        sig_A_y=get_array('sig_A_y'),
        sig_A_xy=get_array('sig_A_xy'),
        c_A=get_array('c_A'),
        c_B=get_array('c_B'),
        c_AB=get_array('c_AB'),
        b_mask=b_mask,
        pred_x=pred_x,
        pred_y=pred_y,
        window_size=window_size,
        win_ctrs_x=get_array('win_ctrs_x'),
        win_ctrs_y=get_array('win_ctrs_y'),
    )
