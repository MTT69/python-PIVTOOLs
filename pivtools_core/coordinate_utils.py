"""
Coordinate extraction utilities for PIV data.

Provides centralized coordinate handling used across calibration and plotting modules.
"""

from typing import Tuple

import numpy as np


def extract_coordinates(coords: np.ndarray, run: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract x, y coordinates for a given run from a coordinates structure.

    Handles both multi-run (object array) and single-run coordinate structs,
    as typically found in MATLAB .mat files from PIV processing.

    Args:
        coords: Coordinates array from scipy.io.loadmat.
            May be a numpy object array (for multi-run data) or a single struct.
        run: 1-based run number to extract coordinates for.

    Returns:
        Tuple of (x, y) coordinate arrays as numpy arrays.

    Raises:
        ValueError: If run is out of range for multi-run data,
            or if run != 1 for single-run data.

    Example:
        >>> mat = loadmat("coordinates.mat", struct_as_record=False, squeeze_me=True)
        >>> coords = mat["coordinates"]
        >>> x, y = extract_coordinates(coords, run=1)
    """
    if isinstance(coords, np.ndarray) and coords.dtype == object:
        # Multi-run case: coords is an object array
        max_coords_runs = coords.size
        if run < 1 or run > max_coords_runs:
            raise ValueError(f"run out of range for coordinates (1..{max_coords_runs})")
        c_el = coords[run - 1]
        cx, cy = np.asarray(c_el.x), np.asarray(c_el.y)
    else:
        # Single-run case: coords is a single struct
        if run != 1:
            raise ValueError("coordinates contains a single run; use run=1")
        c_el = coords
        cx, cy = np.asarray(c_el.x), np.asarray(c_el.y)
    return cx, cy


def extract_coordinate_bounds(
    coords: np.ndarray, run: int
) -> Tuple[float, float, float, float]:
    """
    Extract coordinate bounds (xmin, xmax, ymin, ymax) for a run.

    Useful for determining plot limits and validating data ranges.

    Args:
        coords: Coordinates array from scipy.io.loadmat.
        run: 1-based run number.

    Returns:
        Tuple of (xmin, xmax, ymin, ymax) as floats.

    Example:
        >>> xmin, xmax, ymin, ymax = extract_coordinate_bounds(coords, run=1)
        >>> ax.set_xlim(xmin, xmax)
    """
    x, y = extract_coordinates(coords, run)
    return float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y))


def get_num_coordinate_runs(coords: np.ndarray) -> int:
    """
    Get the number of runs in a coordinates structure.

    Args:
        coords: Coordinates array from scipy.io.loadmat.

    Returns:
        Number of runs (1 for single-run data, >1 for multi-run).
    """
    if isinstance(coords, np.ndarray) and coords.dtype == object:
        return coords.size
    return 1
