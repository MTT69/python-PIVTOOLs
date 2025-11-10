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

from pivtools_cli.piv.piv_result import PIVResult, PIVPassResult


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
            x_grid, y_grid = np.meshgrid(x_centers+1, y_centers[::-1]+1, indexing='xy')

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
        
    Returns
    -------
    Path
        Output path for PIV results.
    """
    base_path = config.base_paths[0]
    
    # Convert camera to int if it's a string
    if isinstance(camera, str):
        if camera.startswith("Cam"):
            camera_num = int(camera[3:])
        else:
            camera_num = int(camera)
    else:
        camera_num = camera
    
    # Get PIV type - default to instantaneous
    piv_type = "instantaneous" if config.data.get("processing", {}).get("instantaneous", True) else "ensemble"
    
    # Use get_data_paths from src/paths.py (positional args: base_dir, num_images, cam, type_name)
    paths = get_data_paths(
        base_path,
        config.num_images,
        camera_num,
        piv_type,
        endpoint="",
        use_uncalibrated=use_uncalibrated
    )
    
    output_path = paths["data_dir"]
    
    if create:
        output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path


def _convert_to_half_precision(arr: np.ndarray) -> np.ndarray:
    """
    Convert float arrays to half precision (float16) for space saving.
    """
    if arr is None or arr.size == 0:
        return arr
    if arr.dtype.kind == 'f':
        return arr.astype(np.float16)
    return arr
