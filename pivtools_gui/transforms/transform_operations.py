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
- rotate_180: Rotate 180 degrees
- swap_ux_uy: Swap ux and uy velocity components
- invert_ux_uy: Negate ux and uy velocity components

Parametric transformations (require :factor suffix, e.g., "scale_velocity:1000"):
- scale_velocity: Multiply ux, uy, uz by factor (for unit conversions like m/s to mm/s)
- scale_coords: Multiply x, y coordinates by factor (for unit conversions like m to mm)
"""

import copy
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict

import numpy as np
from loguru import logger
from scipy.io import loadmat, savemat

from pivtools_core.coordinate_utils import extract_coordinates
from pivtools_core.vector_loading import is_run_valid


# =============================================================================
# Constants
# =============================================================================

COORDINATES_FILENAME = "coordinates.mat"
LOADMAT_OPTIONS = {"struct_as_record": False, "squeeze_me": True}


# =============================================================================
# TypedDict Return Types
# =============================================================================


class TransformResult(TypedDict, total=False):
    """Return type for single-frame transform operations."""
    success: bool
    has_original: bool
    pending_transformations: List[str]
    error: str


class TransformAllResult(TypedDict, total=False):
    """Return type for batch transform operations."""
    success: bool
    total_frames: int
    total_cameras: int
    elapsed_time: float
    error: str


# =============================================================================
# Mat File Helpers
# =============================================================================


def load_mat_for_transform(mat_file: Path) -> dict:
    """
    Load mat file with standardized options for transform operations.

    Args:
        mat_file: Path to the .mat file

    Returns:
        Dictionary containing mat file contents
    """
    return loadmat(str(mat_file), **LOADMAT_OPTIONS)


def save_mat_from_transform(mat_file: Path, mat_dict: dict) -> None:
    """
    Save mat file with compression, suppressing warnings.

    Args:
        mat_file: Path to save the .mat file
        mat_dict: Dictionary to save
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        savemat(str(mat_file), mat_dict, do_compression=True)


# Valid transformation names
# Note: scale_velocity and scale_coords are parametric transforms
# that require a :factor suffix (e.g., "scale_velocity:1000")
VALID_TRANSFORMATIONS = [
    "flip_ud",
    "flip_lr",
    "rotate_90_cw",
    "rotate_90_ccw",
    "rotate_180",
    "swap_ux_uy",
    "invert_ux_uy",
    "scale_velocity",  # Parametric: requires :factor suffix
    "scale_coords",    # Parametric: requires :factor suffix
]

# Parametric transforms that require a numeric factor
PARAMETRIC_TRANSFORMS = ["scale_velocity", "scale_coords"]


def parse_parametric_transform(transformation: str) -> Tuple[str, Optional[float]]:
    """
    Parse a transformation string that may contain parameters.

    Format: "transform_name:parameter"
    Examples:
        "flip_ud" -> ("flip_ud", None)
        "scale_velocity:1000" -> ("scale_velocity", 1000.0)
        "scale_coords:0.001" -> ("scale_coords", 0.001)

    Args:
        transformation: Transform string, optionally with :parameter suffix

    Returns:
        Tuple of (base_transform_name, parameter_value)

    Raises:
        ValueError: If parametric transform is missing parameter or has invalid format
    """
    if ":" not in transformation:
        return (transformation, None)

    parts = transformation.split(":", 1)
    base_name = parts[0]

    if base_name not in PARAMETRIC_TRANSFORMS:
        raise ValueError(f"Unknown parametric transform: {base_name}")

    try:
        factor = float(parts[1])
    except (ValueError, IndexError):
        raise ValueError(f"Invalid parameter for {base_name}: expected numeric factor")

    if factor == 0:
        raise ValueError(f"Scale factor cannot be zero for {base_name}")

    return (base_name, factor)


def apply_transformation_to_piv_result(pr: Any, transformation: str) -> None:
    """
    Apply a geometric transformation to a single piv_result element.

    Modifies the piv_result element in-place.

    Args:
        pr: A piv_result struct with vector field attributes (ux, uy, etc.)
        transformation: One of VALID_TRANSFORMATIONS, optionally with :parameter suffix
            for parametric transforms (e.g., "scale_velocity:1000")

    Note:
        For rotate operations, both the spatial data and velocity components
        are transformed to maintain physical consistency.

    Parametric transforms:
        - scale_velocity:factor - multiply ux, uy, uz by factor
        - scale_coords:factor - multiply x, y by factor
    """
    logger.debug(f"Applying transformation {transformation} to piv_result")

    # Parse transformation to extract base name and optional parameter
    base_name, param = parse_parametric_transform(transformation)

    vector_attrs = ["ux", "uy", "uz", "b_mask", "x", "y"]

    # Handle parametric transforms first
    if base_name == "scale_velocity":
        if param is None:
            logger.error("scale_velocity requires a factor parameter")
            return
        # Scale velocity components only (not coordinates or mask)
        for attr in ["ux", "uy", "uz"]:
            if hasattr(pr, attr):
                arr = np.asarray(getattr(pr, attr))
                if arr.ndim >= 2 and arr.size > 0:
                    setattr(pr, attr, arr * param)
        return

    elif base_name == "scale_coords":
        if param is None:
            logger.error("scale_coords requires a factor parameter")
            return
        # Scale coordinate grids only (not velocities or mask)
        for attr in ["x", "y"]:
            if hasattr(pr, attr):
                arr = np.asarray(getattr(pr, attr))
                if arr.ndim >= 2 and arr.size > 0:
                    setattr(pr, attr, arr * param)
        return

    # Handle non-parametric transforms (use base_name for comparison)
    if base_name == "flip_ud":
        # Flip upside down
        for attr in vector_attrs:
            if hasattr(pr, attr):
                arr = np.asarray(getattr(pr, attr))
                if arr.ndim >= 2 and arr.size > 0:
                    setattr(pr, attr, np.flipud(arr))

    elif base_name == "rotate_90_cw":
        # Rotate 90 degrees clockwise
        for attr in vector_attrs:
            if hasattr(pr, attr):
                arr = np.asarray(getattr(pr, attr))
                if arr.ndim >= 2 and arr.size > 0:
                    setattr(pr, attr, np.rot90(arr, k=-1))

    elif base_name == "rotate_90_ccw":
        # Rotate 90 degrees counter-clockwise
        for attr in vector_attrs:
            if hasattr(pr, attr):
                arr = np.asarray(getattr(pr, attr))
                if arr.ndim >= 2 and arr.size > 0:
                    setattr(pr, attr, np.rot90(arr, k=1))

    elif base_name == "rotate_180":
        # Rotate 180 degrees (equivalent to two 90-degree rotations)
        for attr in vector_attrs:
            if hasattr(pr, attr):
                arr = np.asarray(getattr(pr, attr))
                if arr.ndim >= 2 and arr.size > 0:
                    setattr(pr, attr, np.rot90(arr, k=2))

    elif base_name == "swap_ux_uy":
        # Swap ux and uy velocity components
        if hasattr(pr, "ux") and hasattr(pr, "uy"):
            ux = getattr(pr, "ux")
            uy = getattr(pr, "uy")
            setattr(pr, "ux", uy)
            setattr(pr, "uy", ux)

    elif base_name == "invert_ux_uy":
        # Invert (negate) ux and uy velocity components
        if hasattr(pr, "ux"):
            ux = np.asarray(getattr(pr, "ux"))
            setattr(pr, "ux", -ux)
        if hasattr(pr, "uy"):
            uy = np.asarray(getattr(pr, "uy"))
            setattr(pr, "uy", -uy)

    elif base_name == "flip_lr":
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
        transformation: One of VALID_TRANSFORMATIONS, optionally with :parameter suffix

    Note:
        Some transformations (flip_ud, flip_lr) don't affect coordinates.
        Rotation transformations update both x and y coordinate arrays.
        scale_coords scales x and y by a factor.
        scale_velocity doesn't affect coordinates.
    """
    # Parse transformation to extract base name and optional parameter
    base_name, param = parse_parametric_transform(transformation)

    # Handle parametric transforms
    if base_name == "scale_coords":
        if param is None:
            return
        # Scale x and y coordinates
        cx, cy = extract_coordinates(coords, run)
        if cx.size > 0 and cy.size > 0:
            cx_scaled = cx * param
            cy_scaled = cy * param

            if isinstance(coords, np.ndarray) and coords.dtype == object:
                coords[run - 1].x = cx_scaled
                coords[run - 1].y = cy_scaled
            else:
                coords.x = cx_scaled
                coords.y = cy_scaled
        return

    # scale_velocity doesn't affect coordinates
    if base_name == "scale_velocity":
        return

    # Handle non-parametric transforms
    if base_name == "flip_ud":
        # Coordinates stay the same for flip_ud
        pass

    elif base_name == "rotate_90_cw":
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

    elif base_name == "rotate_90_ccw":
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

    elif base_name == "rotate_180":
        # Rotate coordinates 180 degrees: new_x = -old_x, new_y = -old_y
        cx, cy = extract_coordinates(coords, run)
        if cx.size > 0 and cy.size > 0:
            cx_rot = np.rot90(-cx, k=2)
            cy_rot = np.rot90(-cy, k=2)

            if isinstance(coords, np.ndarray) and coords.dtype == object:
                coords[run - 1].x = cx_rot
                coords[run - 1].y = cy_rot
            else:
                coords.x = cx_rot
                coords.y = cy_rot

    elif base_name == "flip_lr":
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
        mat = load_mat_for_transform(mat_file)
        piv_result = mat["piv_result"]

        # Load coordinates if they exist
        coords = None
        if coords_file and coords_file.exists():
            coords_mat = load_mat_for_transform(coords_file)
            coords = coords_mat.get("coordinates")

        # Apply transformations to all non-empty runs
        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
            num_runs = piv_result.size
            for run_idx in range(num_runs):
                pr = piv_result[run_idx]
                # Only apply to non-empty runs - use centralized validation
                try:
                    if is_run_valid(pr, fields=("ux",), require_2d=False, reject_all_nan=True):
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
        save_mat_from_transform(mat_file, mat)

        # Save coordinates if they were loaded
        if coords is not None:
            save_mat_from_transform(coords_file, {"coordinates": coords})

        return True

    except Exception as e:
        logger.error(f"Error processing frame {frame}: {e}")
        return False


# =============================================================================
# Canonical State for Transform Simplification
# =============================================================================


@dataclass
class TransformCanonicalState:
    """Canonical state representation for transform simplification.

    Transforms are represented in a canonical form that enables
    simplification of non-adjacent operations. This uses algebraic
    group properties:
    - Rotations: cyclic group Z4 (mod 4 arithmetic)
    - Flips: Z2 (even count = identity)
    - Velocity ops: Z2 (even count = identity)
    - Scaling: multiplicative accumulation
    """

    rotation: int = 0  # 0=none, 1=90cw, 2=180, 3=90ccw
    flip_ud: bool = False
    flip_lr: bool = False
    swap_ux_uy: bool = False
    invert_ux_uy: bool = False
    velocity_scale: float = 1.0
    coord_scale: float = 1.0

    def apply_transform(self, transformation: str) -> None:
        """Apply a transformation to update canonical state."""
        base_name, param = parse_parametric_transform(transformation)

        if base_name == "rotate_90_cw":
            self.rotation = (self.rotation + 1) % 4
        elif base_name == "rotate_90_ccw":
            self.rotation = (self.rotation + 3) % 4  # -1 mod 4 = 3
        elif base_name == "rotate_180":
            self.rotation = (self.rotation + 2) % 4
        elif base_name == "flip_ud":
            self.flip_ud = not self.flip_ud
        elif base_name == "flip_lr":
            self.flip_lr = not self.flip_lr
        elif base_name == "swap_ux_uy":
            self.swap_ux_uy = not self.swap_ux_uy
        elif base_name == "invert_ux_uy":
            self.invert_ux_uy = not self.invert_ux_uy
        elif base_name == "scale_velocity" and param is not None:
            self.velocity_scale *= param
        elif base_name == "scale_coords" and param is not None:
            self.coord_scale *= param

    def to_minimal_operations(self) -> List[str]:
        """Convert canonical state to minimal list of operations.

        Returns the shortest sequence of operations that achieves
        the same result as the original transformation sequence.
        """
        result = []

        # Emit rotation (prefer single operation over multiple)
        if self.rotation == 1:
            result.append("rotate_90_cw")
        elif self.rotation == 2:
            result.append("rotate_180")
        elif self.rotation == 3:
            result.append("rotate_90_ccw")
        # rotation == 0 means no rotation needed

        # Emit flips
        if self.flip_ud:
            result.append("flip_ud")
        if self.flip_lr:
            result.append("flip_lr")

        # Emit velocity operations
        if self.swap_ux_uy:
            result.append("swap_ux_uy")
        if self.invert_ux_uy:
            result.append("invert_ux_uy")

        # Emit scale operations (only if not identity)
        if abs(self.velocity_scale - 1.0) > 1e-10:
            result.append(f"scale_velocity:{self.velocity_scale}")
        if abs(self.coord_scale - 1.0) > 1e-10:
            result.append(f"scale_coords:{self.coord_scale}")

        return result


def validate_transformations(
    transformations: List[str], allow_empty: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate a list of transformation names.

    Handles both simple transforms and parametric transforms (e.g., "scale_velocity:1000").

    Args:
        transformations: List of transformation names to validate
        allow_empty: If True, empty list is valid (for status/clear operations)

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if all transformations are valid
        - error_message: Error description if invalid, None if valid
    """
    if not transformations:
        if allow_empty:
            return True, None
        return False, "No transformations provided"

    invalid = []
    for t in transformations:
        try:
            base_name, param = parse_parametric_transform(t)
            # Check if base name is valid
            if base_name not in VALID_TRANSFORMATIONS:
                invalid.append(t)
            # If parametric, ensure parameter was provided
            elif base_name in PARAMETRIC_TRANSFORMS and param is None:
                invalid.append(f"{t} (missing factor parameter)")
        except ValueError as e:
            invalid.append(f"{t} ({str(e)})")

    if invalid:
        return False, f"Invalid transformations: {invalid}. Valid: {VALID_TRANSFORMATIONS}"

    return True, None


def simplify_transformations(ops: List[str]) -> List[str]:
    """
    Simplify a list of transformations using canonical state representation.

    This function reduces any sequence of transformations to a minimal
    equivalent sequence by tracking cumulative state rather than
    looking at adjacent pairs. This enables cancellation of non-adjacent
    operations.

    Examples:
        >>> simplify_transformations(["rotate_90_cw", "rotate_90_ccw"])
        []
        >>> simplify_transformations(["rotate_90_cw", "flip_ud", "rotate_90_ccw"])
        ["flip_ud"]
        >>> simplify_transformations(["rotate_90_cw"] * 4)
        []
        >>> simplify_transformations(["flip_ud", "flip_ud"])
        []
        >>> simplify_transformations(["scale_velocity:2", "scale_velocity:0.5"])
        []

    Args:
        ops: List of transformation operation names

    Returns:
        Minimal equivalent list of transformations
    """
    if not ops:
        return []

    # Build canonical state by applying all transforms
    state = TransformCanonicalState()
    for op in ops:
        try:
            state.apply_transform(op)
        except ValueError:
            # Unknown transform - log warning but continue
            logger.warning(f"Unknown transformation during simplification: {op}")

    return state.to_minimal_operations()
