"""
Module for saving PIV results to .mat files compatible with post-processing code.
"""
import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import scipy.io
from pivtools_core.config import Config
from pivtools_core.paths import get_data_paths

from pivtools_cli.piv.piv_result import (
    PIVResult, PIVPassResult,
    PIVEnsembleResult, PIVEnsemblePassResult,
)


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

    for i in range(num_passes):
        if i in runs_to_save:
            x_centers = win_ctrs_x_list[i]
            y_centers = win_ctrs_y_list[i]

            # Create 2D coordinate grids with smallest y at the bottom
            x_grid, y_grid = np.meshgrid(x_centers + 1, y_centers[::-1] + 1, indexing='xy')

            # Convert to half precision for space saving
            x_grid = _convert_to_half_precision(x_grid)
            y_grid = _convert_to_half_precision(y_grid)

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
    mat_data = _create_ensemble_struct_all_passes(ensemble_result, runs_to_save)

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

    num_passes = config.ensemble_num_passes

    if runs_to_save is None:
        runs_to_save = list(range(num_passes))

    # Create MATLAB-style struct array with fields 'x' and 'y', shape (num_passes,)
    dtype = [('x', object), ('y', object)]
    coords_struct = np.empty((num_passes,), dtype=dtype)

    for i in range(num_passes):
        if i in runs_to_save:
            # Use cached window centers from correlator (handles single mode)
            x_centers = win_ctrs_x_list[i]
            y_centers = win_ctrs_y_list[i]

            # Create 2D coordinate grids with smallest y at the bottom
            x_grid, y_grid = np.meshgrid(x_centers + 1, y_centers[::-1] + 1, indexing='xy')

            # Convert to half precision for space saving
            x_grid = _convert_to_half_precision(x_grid)
            y_grid = _convert_to_half_precision(y_grid)

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

        # Velocity fields
        if pass_result.ux_mat is not None:
            ensemble_struct['ux'][local_idx] = _convert_to_half_precision(pass_result.ux_mat)
        if pass_result.uy_mat is not None:
            ensemble_struct['uy'][local_idx] = _convert_to_half_precision(pass_result.uy_mat)

        # Stress tensors
        if pass_result.UU_stress is not None:
            ensemble_struct['UU_stress'][local_idx] = _convert_to_half_precision(pass_result.UU_stress)
        if pass_result.VV_stress is not None:
            ensemble_struct['VV_stress'][local_idx] = _convert_to_half_precision(pass_result.VV_stress)
        if pass_result.UV_stress is not None:
            ensemble_struct['UV_stress'][local_idx] = _convert_to_half_precision(pass_result.UV_stress)

        # Normalized peak height
        if pass_result.peakheight is not None:
            ensemble_struct['peakheight'][local_idx] = _convert_to_half_precision(pass_result.peakheight)

        # NaN reason
        if pass_result.nan_reason is not None:
            ensemble_struct['nan_reason'][local_idx] = pass_result.nan_reason

        # Sigma parameters (AB)
        if pass_result.sig_AB_x is not None:
            ensemble_struct['sig_AB_x'][local_idx] = _convert_to_half_precision(pass_result.sig_AB_x)
        if pass_result.sig_AB_y is not None:
            ensemble_struct['sig_AB_y'][local_idx] = _convert_to_half_precision(pass_result.sig_AB_y)
        if pass_result.sig_AB_xy is not None:
            ensemble_struct['sig_AB_xy'][local_idx] = _convert_to_half_precision(pass_result.sig_AB_xy)

        # Sigma parameters (A)
        if pass_result.sig_A_x is not None:
            ensemble_struct['sig_A_x'][local_idx] = _convert_to_half_precision(pass_result.sig_A_x)
        if pass_result.sig_A_y is not None:
            ensemble_struct['sig_A_y'][local_idx] = _convert_to_half_precision(pass_result.sig_A_y)
        if pass_result.sig_A_xy is not None:
            ensemble_struct['sig_A_xy'][local_idx] = _convert_to_half_precision(pass_result.sig_A_xy)

        # Gaussian offset parameters (background levels)
        if pass_result.c_A is not None:
            ensemble_struct['c_A'][local_idx] = _convert_to_half_precision(pass_result.c_A)
        if pass_result.c_B is not None:
            ensemble_struct['c_B'][local_idx] = _convert_to_half_precision(pass_result.c_B)
        if pass_result.c_AB is not None:
            ensemble_struct['c_AB'][local_idx] = _convert_to_half_precision(pass_result.c_AB)

        # Window centers and size
        if pass_result.win_ctrs_x is not None:
            ensemble_struct['win_ctrs_x'][local_idx] = _convert_to_half_precision(pass_result.win_ctrs_x)
        if pass_result.win_ctrs_y is not None:
            ensemble_struct['win_ctrs_y'][local_idx] = _convert_to_half_precision(pass_result.win_ctrs_y)
        if pass_result.window_size is not None:
            ensemble_struct['window_size'][local_idx] = pass_result.window_size

        # Mask
        if pass_result.b_mask is not None:
            ensemble_struct['b_mask'][local_idx] = pass_result.b_mask.astype(bool)

        # Predictor fields
        if pass_result.pred_x is not None:
            ensemble_struct['pred_x'][local_idx] = _convert_to_half_precision(pass_result.pred_x)
        if pass_result.pred_y is not None:
            ensemble_struct['pred_y'][local_idx] = _convert_to_half_precision(pass_result.pred_y)

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
        
        # Save ux and uy directly without swapping - coordinate system is now correct
        if pass_result.ux_mat is not None:
            piv_struct['ux'][local_idx] = _convert_to_half_precision(pass_result.ux_mat)
        if pass_result.uy_mat is not None:
            piv_struct['uy'][local_idx] = _convert_to_half_precision(pass_result.uy_mat)

        # Use b_mask from pass_result (already computed during PIV)
        if pass_result.b_mask is not None:
            piv_struct['b_mask'][local_idx] = pass_result.b_mask
        elif pass_result.nan_mask is not None:
            # Fallback to nan_mask if b_mask not available
            piv_struct['b_mask'][local_idx] = pass_result.nan_mask

        if pass_result.nan_mask is not None:
            piv_struct['nan_mask'][local_idx] = pass_result.nan_mask

        # Window centers are always stored in pass_result
        if pass_result.win_ctrs_x is not None:
            piv_struct['win_ctrs_x'][local_idx] = _convert_to_half_precision(pass_result.win_ctrs_x)
        if pass_result.win_ctrs_y is not None:
            piv_struct['win_ctrs_y'][local_idx] = _convert_to_half_precision(pass_result.win_ctrs_y)
            
        if pass_result.peak_mag is not None:
            piv_struct['peak_mag'][local_idx] = _convert_to_half_precision(pass_result.peak_mag)
        if pass_result.peak_choice is not None:
            piv_struct['peak_choice'][local_idx] = pass_result.peak_choice
        if pass_result.n_windows is not None:
            piv_struct['n_windows'][local_idx] = pass_result.n_windows
        if pass_result.predictor_field is not None:
            piv_struct['predictor_field'][local_idx] = _convert_to_half_precision(pass_result.predictor_field)
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


def _convert_to_half_precision(arr: np.ndarray) -> np.ndarray:
    """
    Convert float arrays to half precision (float16) for space saving.
    """
    if arr is None or arr.size == 0:
        return arr
    if arr.dtype.kind == 'f':
        return arr.astype(np.float16)
    return arr
