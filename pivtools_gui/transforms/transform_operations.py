"""
Pure transformation functions for PIV vector fields.

This module contains the low-level transformation operations that can be
applied to piv_result and coordinate data. Functions are designed to be
picklable for use with multiprocessing.

Supported transformations:
- flip_ud: Flip data vertically (upside down)
- flip_lr: Flip data horizontally (left-right)
- rotate_90_cw: Rotate 90 degrees clockwise
- rotate_90_ccw: Rotate 90 degrees counter-clockwise
- swap_ux_uy: Swap ux and uy velocity components
- invert_ux_uy: Negate ux and uy velocity components
"""

import copy
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy.io import loadmat, savemat

from pivtools_core.coordinate_utils import extract_coordinates


# Valid transformation names
VALID_TRANSFORMATIONS = [
    "flip_ud",
    "flip_lr",
    "rotate_90_cw",
    "rotate_90_ccw",
    "swap_ux_uy",
    "invert_ux_uy",
]


def apply_transformation_to_piv_result(pr: Any, transformation: str) -> None:
    """
    Apply a geometric transformation to a single piv_result element.

    Modifies the piv_result element in-place.

    Args:
        pr: A piv_result struct with vector field attributes (ux, uy, etc.)
        transformation: One of VALID_TRANSFORMATIONS

    Note:
        For rotate operations, both the spatial data and velocity components
        are transformed to maintain physical consistency.
    """
    logger.debug(f"Applying transformation {transformation} to piv_result")
    vector_attrs = ["ux", "uy", "uz", "b_mask", "x", "y"]

    if transformation == "flip_ud":
        # Flip upside down
        for attr in vector_attrs:
            if hasattr(pr, attr):
                arr = np.asarray(getattr(pr, attr))
                if arr.ndim >= 2 and arr.size > 0:
                    setattr(pr, attr, np.flipud(arr))

    elif transformation == "rotate_90_cw":
        # Rotate 90 degrees clockwise
        for attr in vector_attrs:
            if hasattr(pr, attr):
                arr = np.asarray(getattr(pr, attr))
                if arr.ndim >= 2 and arr.size > 0:
                    setattr(pr, attr, np.rot90(arr, k=-1))

    elif transformation == "rotate_90_ccw":
        # Rotate 90 degrees counter-clockwise
        for attr in vector_attrs:
            if hasattr(pr, attr):
                arr = np.asarray(getattr(pr, attr))
                if arr.ndim >= 2 and arr.size > 0:
                    setattr(pr, attr, np.rot90(arr, k=1))

    elif transformation == "swap_ux_uy":
        # Swap ux and uy velocity components
        if hasattr(pr, "ux") and hasattr(pr, "uy"):
            ux = getattr(pr, "ux")
            uy = getattr(pr, "uy")
            setattr(pr, "ux", uy)
            setattr(pr, "uy", ux)

    elif transformation == "invert_ux_uy":
        # Invert (negate) ux and uy velocity components
        if hasattr(pr, "ux"):
            ux = np.asarray(getattr(pr, "ux"))
            setattr(pr, "ux", -ux)
        if hasattr(pr, "uy"):
            uy = np.asarray(getattr(pr, "uy"))
            setattr(pr, "uy", -uy)

    elif transformation == "flip_lr":
        # Flip left-right
        for attr in vector_attrs:
            if hasattr(pr, attr):
                arr = np.asarray(getattr(pr, attr))
                if arr.ndim >= 2 and arr.size > 0:
                    setattr(pr, attr, np.fliplr(arr))

    else:
        logger.warning(f"Unknown transformation: {transformation}")


def apply_transformation_to_coordinates(
    coords: np.ndarray, run: int, transformation: str
) -> None:
    """
    Apply a geometric transformation to coordinates for a specific run.

    Modifies the coordinates in-place.

    Args:
        coords: Coordinates array (may be object array for multi-run)
        run: 1-based run number
        transformation: One of VALID_TRANSFORMATIONS

    Note:
        Some transformations (flip_ud, flip_lr) don't affect coordinates.
        Rotation transformations update both x and y coordinate arrays.
    """
    if transformation == "flip_ud":
        # Coordinates stay the same for flip_ud
        pass

    elif transformation == "rotate_90_cw":
        # Rotate coordinates 90 degrees clockwise: new_x = old_y, new_y = -old_x
        cx, cy = extract_coordinates(coords, run)
        if cx.size > 0 and cy.size > 0:
            cx_rot = np.rot90(cy, k=-1)
            cy_rot = np.rot90(-cx, k=-1)

            if isinstance(coords, np.ndarray) and coords.dtype == object:
                coords[run - 1].x = cx_rot
                coords[run - 1].y = cy_rot
            else:
                coords.x = cx_rot
                coords.y = cy_rot

    elif transformation == "rotate_90_ccw":
        # Rotate coordinates 90 degrees counter-clockwise: new_x = -old_y, new_y = old_x
        cx, cy = extract_coordinates(coords, run)
        if cx.size > 0 and cy.size > 0:
            cx_rot = np.rot90(-cy, k=1)
            cy_rot = np.rot90(cx, k=1)

            if isinstance(coords, np.ndarray) and coords.dtype == object:
                coords[run - 1].x = cx_rot
                coords[run - 1].y = cy_rot
            else:
                coords.x = cx_rot
                coords.y = cy_rot

    elif transformation == "flip_lr":
        # Coordinates stay the same for flip_lr
        pass

    # swap_ux_uy and invert_ux_uy don't affect coordinates


def backup_original_data(
    mat: Dict, coords_mat: Optional[Dict] = None
) -> Tuple[Dict, Optional[Dict]]:
    """
    Create backup copies of piv_result and coordinates as _original.

    Only creates backups if they don't already exist.

    Args:
        mat: Dictionary containing piv_result from loadmat
        coords_mat: Optional dictionary containing coordinates from loadmat

    Returns:
        Tuple of (updated_mat, updated_coords_mat) with _original fields added
    """
    # Backup piv_result if not already backed up
    if "piv_result_original" not in mat:
        logger.debug("Creating backup: piv_result -> piv_result_original")
        mat["piv_result_original"] = copy.deepcopy(mat["piv_result"])

    # Backup coordinates if provided and not already backed up
    if coords_mat is not None and "coordinates_original" not in coords_mat:
        logger.debug("Creating backup: coordinates -> coordinates_original")
        coords_mat["coordinates_original"] = copy.deepcopy(coords_mat["coordinates"])

    return mat, coords_mat


def restore_original_data(
    mat: Dict, coords_mat: Optional[Dict] = None
) -> Tuple[Dict, Optional[Dict]]:
    """
    Restore piv_result and coordinates from _original backups and remove backups.

    Args:
        mat: Dictionary containing piv_result and piv_result_original
        coords_mat: Optional dictionary containing coordinates

    Returns:
        Tuple of (updated_mat, updated_coords_mat) with original data restored
    """
    # Restore piv_result from backup
    if "piv_result_original" in mat:
        logger.debug("Restoring: piv_result_original -> piv_result")
        mat["piv_result"] = mat["piv_result_original"]
        del mat["piv_result_original"]
        # Clear transformation list
        mat["pending_transformations"] = []

    # Restore coordinates from backup
    if coords_mat is not None and "coordinates_original" in coords_mat:
        logger.debug("Restoring: coordinates_original -> coordinates")
        coords_mat["coordinates"] = coords_mat["coordinates_original"]
        del coords_mat["coordinates_original"]

    return mat, coords_mat


def has_original_backup(mat: Dict) -> bool:
    """
    Check if original backup exists for this frame.

    Args:
        mat: Dictionary containing piv_result data

    Returns:
        True if piv_result_original exists in mat
    """
    return "piv_result_original" in mat


def process_frame_worker(
    frame: int,
    mat_file: Path,
    coords_file: Optional[Path],
    transformations: List[str],
) -> bool:
    """
    Worker function for processing a single frame in parallel.

    This function must be at module level for pickle serialization
    when using ProcessPoolExecutor.

    Args:
        frame: Frame number (for logging)
        mat_file: Path to the .mat file containing piv_result
        coords_file: Optional path to coordinates.mat
        transformations: List of transformations to apply in order

    Returns:
        True if successful, False if an error occurred
    """
    try:
        mat = loadmat(str(mat_file), struct_as_record=False, squeeze_me=True)
        piv_result = mat["piv_result"]

        # Load coordinates if they exist
        coords = None
        if coords_file and coords_file.exists():
            coords_mat = loadmat(str(coords_file), struct_as_record=False, squeeze_me=True)
            coords = coords_mat.get("coordinates")

        # Apply transformations to all non-empty runs
        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
            num_runs = piv_result.size
            for run_idx in range(num_runs):
                pr = piv_result[run_idx]
                # Only apply to non-empty runs
                try:
                    if hasattr(pr, "ux"):
                        ux = np.asarray(pr.ux)
                        if ux.size > 0 and not np.all(np.isnan(ux)):
                            for trans in transformations:
                                apply_transformation_to_piv_result(pr, trans)
                                if coords is not None:
                                    apply_transformation_to_coordinates(
                                        coords, run_idx + 1, trans
                                    )
                except Exception as e:
                    logger.warning(
                        f"Error checking run {run_idx + 1} in frame {frame}: {e}, skipping"
                    )
        else:
            # Single run
            for trans in transformations:
                apply_transformation_to_piv_result(piv_result, trans)
                if coords is not None:
                    apply_transformation_to_coordinates(coords, 1, trans)

        # Save back the mat file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            savemat(str(mat_file), mat, do_compression=True)

        # Save coordinates if they were loaded
        if coords is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                savemat(str(coords_file), {"coordinates": coords}, do_compression=True)

        return True

    except Exception as e:
        logger.error(f"Error processing frame {frame}: {e}")
        return False


def validate_transformations(transformations: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate a list of transformation names.

    Args:
        transformations: List of transformation names to validate

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if all transformations are valid
        - error_message: Error description if invalid, None if valid
    """
    if not transformations:
        return False, "No transformations provided"

    invalid = [t for t in transformations if t not in VALID_TRANSFORMATIONS]
    if invalid:
        return False, f"Invalid transformations: {invalid}. Valid: {VALID_TRANSFORMATIONS}"

    return True, None
