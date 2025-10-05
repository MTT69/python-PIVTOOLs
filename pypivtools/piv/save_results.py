"""
Module for saving PIV results to .mat files compatible with post-processing code.
"""
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.io

from pypivtools.config import Config
from pypivtools.piv.piv_result import PIVResult, PIVPassResult


def save_piv_result_distributed(
    piv_result: PIVResult,
    output_path: Path,
    frame_number: int,
    pass_index: Optional[int] = None,
) -> str:
    """
    Save a PIV result to disk. Designed to be submitted to Dask workers.
    
    This function can be called on Dask workers to save results in parallel,
    avoiding the memory bottleneck of gathering all results to main.
    
    Parameters
    ----------
    piv_result : PIVResult
        The PIV result object containing one or more passes.
    output_path : Path
        Directory where the .mat file will be saved.
    frame_number : int
        Frame number (1-based) for the filename (e.g., 1 -> B00001.mat).
    pass_index : Optional[int]
        If specified, save only this pass (0-based).
        If None, save all passes.
        
    Returns
    -------
    str
        Path to the saved file (for verification/logging).
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / f"B{frame_number:05d}.mat"
    
    if len(piv_result.passes) == 0:
        logging.warning(
            f"PIVResult has no passes for frame {frame_number}. "
            "Skipping save."
        )
        return str(filename)
    
    # Create single struct with arrays indexed by pass number
    mat_data = _create_piv_struct_all_passes(piv_result, pass_index)
    
    # Save to .mat file
    scipy.io.savemat(filename, {"piv_result": mat_data}, oned_as="row")
    logging.info(f"Worker saved PIV result to {filename}")
    
    return str(filename)


def save_piv_result_to_mat(
    piv_result: PIVResult,
    output_path: Path,
    frame_number: int,
    pass_index: Optional[int] = None,
) -> None:
    """
    Save a single PIVResult to a .mat file compatible with the post-processing code.
    
    Parameters
    ----------
    piv_result : PIVResult
        The PIV result object containing one or more passes.
    output_path : Path
        Directory where the .mat file will be saved.
    frame_number : int
        Frame number (1-based) for the filename (e.g., 1 -> B00001.mat).
    pass_index : Optional[int]
        If specified, save only this pass (0-based). If None, save all passes as multiple runs.
    
    Notes
    -----
    - If pass_index is None, saves all passes as separate runs in the .mat file
    - If pass_index is specified, saves only that specific pass
    - Output filename format: B{frame_number:05d}.mat
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / f"B{frame_number:05d}.mat"
    
    if len(piv_result.passes) == 0:
        logging.warning(
            f"PIVResult has no passes for frame {frame_number}. "
            "Skipping save."
        )
        return
    
    # Create single struct with arrays indexed by pass number
    mat_data = _create_piv_struct_all_passes(piv_result, pass_index)
    
    # Save to .mat file
    scipy.io.savemat(filename, {"piv_result": mat_data}, oned_as="row")
    logging.info(f"Saved PIV result to {filename}")


def save_piv_results_batch(
    piv_results: List[PIVResult],
    output_path: Path,
    start_frame: int = 1,
    pass_index: Optional[int] = None,
    config: Optional[Config] = None,
) -> None:
    """
    Save multiple PIV results to individual .mat files.
    
    Parameters
    ----------
    piv_results : List[PIVResult]
        List of PIV result objects to save.
    output_path : Path
        Directory where the .mat files will be saved.
    start_frame : int
        Starting frame number (1-based) for filenames.
    pass_index : Optional[int]
        If specified, save only this pass (0-based) from each result.
    config : Optional[Config]
        Configuration object (for future use).
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, piv_result in enumerate(piv_results):
        frame_number = start_frame + i
        try:
            save_piv_result_to_mat(piv_result, output_path, frame_number, pass_index)
        except Exception as e:
            logging.error(f"Failed to save frame {frame_number}: {e}")


def save_coordinates(
    piv_result: PIVResult,
    win_ctrs_x_list: List[np.ndarray],
    win_ctrs_y_list: List[np.ndarray],
    output_path: Path,
) -> None:
    """
    Save coordinate grids to coordinates.mat file.
    
    Parameters
    ----------
    piv_result : PIVResult
        A sample PIV result (to determine number of passes).
    win_ctrs_x_list : List[np.ndarray]
        List of x-coordinates for each pass.
    win_ctrs_y_list : List[np.ndarray]
        List of y-coordinates for each pass.
    output_path : Path
        Directory where coordinates.mat will be saved.
        
    Notes
    -----
    Creates coordinate meshgrids for each pass and saves them as a struct array.
    Compatible with load_coords_from_directory() in post-processing code.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_passes = len(piv_result.passes)
    
    if len(win_ctrs_x_list) != num_passes or len(win_ctrs_y_list) != num_passes:
        raise ValueError(
            f"Number of coordinate arrays ({len(win_ctrs_x_list)}, {len(win_ctrs_y_list)}) "
            f"does not match number of passes ({num_passes})"
        )
    
    # Create structured array for coordinates
    coords_data = np.empty(num_passes, dtype=object)
    
    for i in range(num_passes):
        # Create meshgrid from window centers
        x_centers = win_ctrs_x_list[i]
        y_centers = win_ctrs_y_list[i]
        
        # Create 2D coordinate grids
        x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='xy')
        
        # Create struct with x and y fields
        coords_struct = np.empty(1, dtype=[('x', 'O'), ('y', 'O')])
        coords_struct['x'][0] = x_grid
        coords_struct['y'][0] = y_grid
        coords_data[i] = coords_struct[0]
    
    filename = output_path / "coordinates.mat"
    scipy.io.savemat(filename, {"coordinates": coords_data}, oned_as="row")
    logging.info(f"Saved coordinates to {filename}")


def save_coordinates_from_config_distributed(
    config: Config,
    output_path: Path,
) -> str:
    """
    Generate and save coordinate grids. Designed for Dask workers.
    
    Parameters
    ----------
    config : Config
        Configuration object containing window sizes and overlap.
    output_path : Path
        Directory where coordinates.mat will be saved.
        
    Returns
    -------
    str
        Path to the saved coordinates file.
    """
    from pypivtools.piv.piv_backend.cpu_instantaneous import (
        InstantaneousCorrelatorCPU
    )
    
    # Create a temporary correlator just to compute window centers
    correlator = InstantaneousCorrelatorCPU(config)
    
    # Extract the cached window centers
    win_ctrs_x_list = correlator.win_ctrs_x
    win_ctrs_y_list = correlator.win_ctrs_y
    
    num_passes = len(config.window_sizes)
    
    # Create structured array for coordinates
    coords_data = np.empty(num_passes, dtype=object)
    
    for i in range(num_passes):
        x_centers = win_ctrs_x_list[i]
        y_centers = win_ctrs_y_list[i]
        
        # Create 2D coordinate grids
        x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='xy')
        
        # Create struct with x and y fields
        coords_struct = np.empty(1, dtype=[('x', 'O'), ('y', 'O')])
        coords_struct['x'][0] = x_grid
        coords_struct['y'][0] = y_grid
        coords_data[i] = coords_struct[0]
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / "coordinates.mat"
    scipy.io.savemat(filename, {"coordinates": coords_data}, oned_as="row")
    logging.info(f"Worker saved coordinates to {filename}")
    
    return str(filename)


def save_coordinates_from_config(
    config: Config,
    output_path: Path,
) -> None:
    """
    Generate and save coordinate grids based on configuration.
    
    Parameters
    ----------
    config : Config
        Configuration object containing window sizes and overlap.
    output_path : Path
        Directory where coordinates.mat will be saved.
        
    Notes
    -----
    This is a convenience function that computes window centers from the config
    and saves them. Useful when you don't have a PIVResult object yet.
    """
    from pypivtools.piv.piv_backend.cpu_instantaneous import (
        InstantaneousCorrelatorCPU
    )
    
    # Create a temporary correlator just to compute window centers
    correlator = InstantaneousCorrelatorCPU(config)
    
    # Extract the cached window centers
    win_ctrs_x_list = correlator.win_ctrs_x
    win_ctrs_y_list = correlator.win_ctrs_y
    
    num_passes = len(config.window_sizes)
    
    # Create structured array for coordinates
    coords_data = np.empty(num_passes, dtype=object)
    
    for i in range(num_passes):
        x_centers = win_ctrs_x_list[i]
        y_centers = win_ctrs_y_list[i]
        
        # Create 2D coordinate grids
        x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='xy')
        
        # Create struct with x and y fields
        coords_struct = np.empty(1, dtype=[('x', 'O'), ('y', 'O')])
        coords_struct['x'][0] = x_grid
        coords_struct['y'][0] = y_grid
        coords_data[i] = coords_struct[0]
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / "coordinates.mat"
    scipy.io.savemat(filename, {"coordinates": coords_data}, oned_as="row")
    logging.info(f"Saved coordinates to {filename}")


def _create_piv_struct_all_passes(
    piv_result: PIVResult,
    pass_index: Optional[int] = None,
) -> np.ndarray:
    """
    Create a MATLAB-compatible struct with arrays indexed by pass number.
    
    This creates a single struct where each field (ux, uy, b_mask, etc.) is
    an array with one element per pass, matching the expected format:
        piv_result["ux"][pass_idx] = 2D array for that pass
    
    Parameters
    ----------
    piv_result : PIVResult
        PIV result object containing one or more passes.
    pass_index : Optional[int]
        If specified, save only this pass (0-based).
        If None, save all passes.
        
    Returns
    -------
    np.ndarray
        Structured numpy array compatible with scipy.io.savemat.
    """
    n_passes = len(piv_result.passes)
    
    # Determine which passes to save
    if pass_index is not None:
        if pass_index >= n_passes:
            raise IndexError(
                f"Pass index {pass_index} out of range. "
                f"PIVResult has {n_passes} passes."
            )
        n_passes_to_save = 1
        passes_to_save = [pass_index]
    else:
        n_passes_to_save = n_passes
        passes_to_save = list(range(n_passes))
    
    # Create structured dtype with all fields
    dtype = [
        ('ux', object),
        ('uy', object),
        ('b_mask', object),
        ('Q', object),
        ('peak_mag', object),
        ('peak_choice', object),
        ('n_windows', object),
        ('predictor_field', object),
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
        piv_struct['Q'][i] = empty
        piv_struct['peak_mag'][i] = empty
        piv_struct['peak_choice'][i] = empty
        piv_struct['n_windows'][i] = empty
        piv_struct['predictor_field'][i] = empty
    
    # Fill with actual data for selected passes
    for local_idx, global_pass_idx in enumerate(passes_to_save):
        pass_result = piv_result.passes[global_pass_idx]
        
        if pass_result.ux_mat is not None:
            piv_struct['ux'][local_idx] = pass_result.ux_mat
        if pass_result.uy_mat is not None:
            piv_struct['uy'][local_idx] = pass_result.uy_mat
        if pass_result.nan_mask is not None:
            piv_struct['b_mask'][local_idx] = pass_result.nan_mask
        if pass_result.Q is not None:
            piv_struct['Q'][local_idx] = pass_result.Q
        if pass_result.peak_mag is not None:
            piv_struct['peak_mag'][local_idx] = pass_result.peak_mag
        if pass_result.peak_choice is not None:
            piv_struct['peak_choice'][local_idx] = pass_result.peak_choice
        if pass_result.n_windows is not None:
            piv_struct['n_windows'][local_idx] = pass_result.n_windows
        if pass_result.predictor_field is not None:
            piv_struct['predictor_field'][local_idx] = (
                pass_result.predictor_field
            )
    
    return piv_struct


def get_data_paths(
    base_dir: Path,
    num_images: int,
    camera: str,
    type_name: str,
    endpoint: str = "",
    use_uncalibrated: bool = True,
) -> dict:
    """
    Construct directories for PIV data following standardized structure.
    
    This function follows the same pattern as the post-processing code.
    
    Parameters
    ----------
    base_dir : Path
        Base directory for all data.
    num_images : int
        Number of images being processed.
    camera : str
        Camera identifier (e.g., "Cam1" or "1").
    type_name : str
        PIV type name (e.g., "instantaneous", "ensemble").
    endpoint : str
        Optional subfolder within the data directory.
    use_uncalibrated : bool
        If True, use uncalibrated_piv directory.
        If False, use calibrated_piv directory.
        
    Returns
    -------
    dict
        Dictionary with 'data_dir' key pointing to output directory.
    """
    base_dir = Path(base_dir)
    
    # Ensure camera has "Cam" prefix
    if not camera.startswith("Cam"):
        camera = f"Cam{camera}"
    
    # Build path following the standard structure
    if use_uncalibrated:
        data_dir = (
            base_dir / "uncalibrated_piv" / str(num_images) /
            camera / type_name
        )
    else:
        data_dir = (
            base_dir / "calibrated_piv" / str(num_images) /
            camera / type_name
        )
    
    if endpoint:
        data_dir = data_dir / endpoint
    
    return dict(data_dir=data_dir)


def get_output_path(
    config: Config,
    camera: str,
    create: bool = True,
    use_uncalibrated: bool = True,
) -> Path:
    """
    Get the output path for a specific camera's PIV results.
    
    Follows the standardized directory structure:
    - Uncalibrated: base_path/uncalibrated_piv/{num_images}/{camera}/{type}
    - Calibrated: base_path/calibrated_piv/{num_images}/{camera}/{type}
    
    Parameters
    ----------
    config : Config
        Configuration object.
    camera : str
        Camera folder name (e.g., "Cam1").
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
    base_path = Path(config.base_path)
    
    # Ensure camera has "Cam" prefix
    if not camera.startswith("Cam"):
        camera = f"Cam{camera}"
    
    # Get PIV type from config (e.g., "instantaneous")
    piv_type = (
        config.piv_type if hasattr(config, 'piv_type')
        else 'instantaneous'
    )
    
    # Build path following the standard structure
    if use_uncalibrated:
        output_path = (
            base_path / "uncalibrated_piv" / str(config.num_images) /
            camera / piv_type
        )
    else:
        output_path = (
            base_path / "calibrated_piv" / str(config.num_images) /
            camera / piv_type
        )
    
    if create:
        output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path
