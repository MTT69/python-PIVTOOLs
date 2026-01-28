import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import dask
import dask.array as da
import numpy as np
import scipy.io

from pivtools_core.config import Config


# =============================================================================
# Run Validation API
# =============================================================================


@dataclass
class RunValidationResult:
    """Result of run validation."""

    valid_runs: List[int]  # List of valid run indices (0-based by default)
    total_runs: int  # Total number of runs in file
    single_run: bool  # True if file has single run (not object array)


def is_run_valid(
    struct: Any,
    fields: Sequence[str] = ("ux", "uy"),
    require_2d: bool = True,
    reject_all_nan: bool = True,
) -> bool:
    """
    Check if a single piv_result/coordinates struct has valid data.

    Args:
        struct: A MATLAB struct object with field attributes
        fields: Field names to check (default: ("ux", "uy") for vectors,
                use ("x", "y") for coordinates)
        require_2d: If True, require arrays to be 2-dimensional
        reject_all_nan: If True, reject runs where all values are NaN

    Returns:
        True if all specified fields have valid data
    """
    for field in fields:
        arr = getattr(struct, field, None)
        if arr is None:
            return False
        arr = np.asarray(arr)

        # Size check (always applied)
        if arr.size == 0:
            return False

        # Dimension check (optional)
        if require_2d and arr.ndim != 2:
            return False

        # NaN check (optional)
        if reject_all_nan and np.all(np.isnan(arr)):
            return False

    return True


def find_valid_runs(
    file_path: Union[str, Path],
    var_name: str = "piv_result",
    fields: Sequence[str] = ("ux", "uy"),
    require_2d: bool = True,
    reject_all_nan: bool = True,
    one_based: bool = False,
) -> RunValidationResult:
    """
    Find all runs with valid data in a .mat file.

    Args:
        file_path: Path to .mat file
        var_name: Variable name in mat file ("piv_result" or "coordinates")
        fields: Field names to validate (default: ("ux", "uy"))
        require_2d: If True, require arrays to be 2-dimensional
        reject_all_nan: If True, reject runs where all values are NaN
        one_based: If True, return 1-based indices; if False, 0-based

    Returns:
        RunValidationResult with valid_runs list, total_runs count, and single_run flag

    Raises:
        FileNotFoundError: If file does not exist
        KeyError: If var_name not found in mat file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    mat = scipy.io.loadmat(str(file_path), struct_as_record=False, squeeze_me=True)
    if var_name not in mat:
        raise KeyError(f"Variable '{var_name}' not found in {file_path}")

    data = mat[var_name]
    valid_runs: List[int] = []
    offset = 1 if one_based else 0

    # Check if multi-run (object array) or single-run
    if isinstance(data, np.ndarray) and data.dtype == object and data.size > 0:
        # Multiple runs - use .flat for consistent access
        total_runs = data.size
        for run_idx in range(total_runs):
            struct = data.flat[run_idx]
            if is_run_valid(struct, fields, require_2d, reject_all_nan):
                valid_runs.append(run_idx + offset)
        return RunValidationResult(
            valid_runs=valid_runs, total_runs=total_runs, single_run=False
        )
    else:
        # Single run
        if is_run_valid(data, fields, require_2d, reject_all_nan):
            valid_runs.append(offset)  # 0 for 0-based, 1 for 1-based
        return RunValidationResult(
            valid_runs=valid_runs, total_runs=1, single_run=True
        )


def get_first_valid_run(
    file_path: Union[str, Path],
    var_name: str = "piv_result",
    fields: Sequence[str] = ("ux", "uy"),
    require_2d: bool = True,
    reject_all_nan: bool = True,
    one_based: bool = False,
) -> Optional[int]:
    """
    Return first valid run index, or None if no valid runs.

    Args:
        Same as find_valid_runs()

    Returns:
        First valid run index, or None if no valid runs
    """
    result = find_valid_runs(
        file_path, var_name, fields, require_2d, reject_all_nan, one_based
    )
    return result.valid_runs[0] if result.valid_runs else None


def get_highest_valid_run(
    file_path: Union[str, Path],
    var_name: str = "piv_result",
    fields: Sequence[str] = ("ux", "uy"),
    require_2d: bool = True,
    reject_all_nan: bool = True,
    one_based: bool = False,
) -> Optional[int]:
    """
    Return highest valid run index, or None if no valid runs.

    Args:
        Same as find_valid_runs()

    Returns:
        Highest valid run index, or None if no valid runs
    """
    result = find_valid_runs(
        file_path, var_name, fields, require_2d, reject_all_nan, one_based
    )
    return result.valid_runs[-1] if result.valid_runs else None


def find_valid_piv_runs(
    file_path: Union[str, Path],
    one_based: bool = False,
    result_key: str = "piv_result",
) -> RunValidationResult:
    """
    Find valid runs in a piv_result or ensemble_result file (checks ux, uy).

    Args:
        file_path: Path to .mat file containing piv_result or ensemble_result
        one_based: If True, return 1-based indices; if False, 0-based
        result_key: Key name in mat file ("piv_result" or "ensemble_result")

    Returns:
        RunValidationResult with valid_runs list, total_runs count, and single_run flag
    """
    return find_valid_runs(
        file_path, var_name=result_key, fields=("ux", "uy"), one_based=one_based
    )


def find_valid_coord_runs(
    file_path: Union[str, Path], one_based: bool = False
) -> RunValidationResult:
    """
    Find valid runs in a coordinates file (checks x, y).

    Args:
        file_path: Path to .mat file containing coordinates
        one_based: If True, return 1-based indices; if False, 0-based

    Returns:
        RunValidationResult with valid_runs list, total_runs count, and single_run flag
    """
    return find_valid_runs(
        file_path,
        var_name="coordinates",
        fields=("x", "y"),
        require_2d=False,  # Coordinates may be 1D or 2D
        one_based=one_based,
    )


def find_non_empty_run(
    piv_result: Any,
    var: str,
    run: int = 1,
    require_2d: bool = False,
    reject_all_nan: bool = True,
) -> Tuple[Optional[Any], int]:
    """
    Find a non-empty run in piv_result for a specified variable.

    Searches starting from the given run index, returning the first run that
    has valid (non-empty, optionally non-NaN) data for the specified variable.

    This is a higher-level helper that wraps is_run_valid() for the common
    pattern of finding valid data starting from a specific run.

    Args:
        piv_result: piv_result array (may be multi-run object array or single struct)
        var: Variable name to check (e.g., "ux", "uy", "uz")
        run: 1-based run number to start searching from (default: 1)
        require_2d: If True, require arrays to be 2-dimensional
        reject_all_nan: If True, reject runs where all values are NaN

    Returns:
        Tuple of (piv_result_element, run_number) where:
        - piv_result_element: The valid piv_result struct, or None if not found
        - run_number: The 1-based run number of the valid data

    Raises:
        ValueError: If run != 1 for single-run data and run is out of range

    Example:
        >>> mat = loadmat("B00001.mat", struct_as_record=False, squeeze_me=True)
        >>> piv_result = mat["piv_result"]
        >>> pr, run = find_non_empty_run(piv_result, "ux", run=1)
        >>> if pr is not None:
        ...     ux = np.asarray(pr.ux)
    """
    pr = None

    # Check if multi-run: must be ndarray with dtype=object (array of mat_struct objects)
    # scipy.io with struct_as_record=False returns mat_struct objects accessed via getattr()
    is_multi_run = (
        isinstance(piv_result, np.ndarray) and
        piv_result.dtype == object and
        piv_result.size > 0
    )

    if is_multi_run:
        # Multi-run case - iterate through runs
        max_runs = piv_result.size
        current_run = run
        while current_run <= max_runs:
            # Use .flat for consistent access regardless of array shape
            pr_candidate = piv_result.flat[current_run - 1]
            if is_run_valid(
                pr_candidate,
                fields=(var,),
                require_2d=require_2d,
                reject_all_nan=reject_all_nan,
            ):
                pr = pr_candidate
                run = current_run
                break
            current_run += 1
    else:
        # Single-run case (scalar struct or squeezed single-element array)
        if run != 1:
            raise ValueError("piv_result contains a single run; use run=1")
        if is_run_valid(
            piv_result,
            fields=(var,),
            require_2d=require_2d,
            reject_all_nan=reject_all_nan,
        ):
            pr = piv_result
            run = 1
        else:
            pr = None

    return pr, run


def _stack_velocity_components(pr, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """
    Stack velocity components into array, including uz for stereo data if present.

    Returns:
        Stacked array of shape (3, H, W) for 2D or (4, H, W) for stereo.
        Components: [ux, uy, b_mask] or [ux, uy, uz, b_mask]
    """
    b_mask = (
        np.asarray(pr.b_mask).astype(ux.dtype, copy=False)
        if ux.size > 0
        else np.array([])
    )

    # Check for uz (stereo data)
    uz_raw = getattr(pr, 'uz', None)
    if uz_raw is not None:
        uz = np.asarray(uz_raw)
        if uz.size > 0 and uz.shape == ux.shape:
            return np.stack([ux, uy, uz, b_mask], axis=0)  # (4, H, W) for stereo

    return np.stack([ux, uy, b_mask], axis=0)  # (3, H, W) for 2D


def read_mat_contents(
    file_path: str,
    run_index: Optional[int] = None,
    return_all_runs: bool = False,
    var_name: str = "piv_result",
) -> np.ndarray:
    """
    Reads piv_result or ensemble_result from a .mat file.
    If multiple runs are present, selects the specified run_index (0-based).
    If run_index is None, selects the first run with valid (non-empty) data.
    If return_all_runs is True, returns all runs in shape (R, 3_or_4, H, W).
    Otherwise returns shape (1, 3_or_4, H, W) for the selected run.

    For stereo data (with uz), returns 4 components: [ux, uy, uz, b_mask].
    For 2D data, returns 3 components: [ux, uy, b_mask].

    Args:
        file_path: Path to the .mat file
        run_index: Optional specific run to load (0-based)
        return_all_runs: If True, return all runs
        var_name: Key name in mat file ("piv_result" or "ensemble_result")
    """
    mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    if var_name not in mat:
        raise KeyError(f"'{var_name}' not found in {file_path}")
    piv_result = mat[var_name]

    # Multiple runs case: numpy array of structs
    if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
        total_runs = piv_result.size

        if return_all_runs:
            # Return all runs
            all_runs = []
            for idx in range(total_runs):
                pr = piv_result[idx]
                ux = np.asarray(pr.ux)
                uy = np.asarray(pr.uy)

                if ux.size > 0 and uy.size > 0:
                    stacked = _stack_velocity_components(pr, ux, uy)  # (3 or 4, H, W)
                else:
                    # Empty run - create placeholder with consistent shape if possible
                    stacked = np.array([[], [], []])  # Will be reshaped later
                all_runs.append(stacked)
            
            try:
                return np.array(all_runs)  # (R, 3_or_4, H, W)
            except ValueError as e:
                # Fallback to object array if shapes are inconsistent
                # Use manual assignment to avoid numpy broadcasting errors with mixed shapes
                out = np.empty(len(all_runs), dtype=object)
                for i, r in enumerate(all_runs):
                    out[i] = r
                return out

        # Single run selection
        if run_index is None:
            # Find first valid run using unified validation
            for idx in range(total_runs):
                pr = piv_result[idx]
                # Use lenient validation here (no ndim/NaN check) for backward compatibility
                if is_run_valid(pr, fields=("ux", "uy"), require_2d=False, reject_all_nan=False):
                    run_index = idx
                    break
            else:
                raise ValueError(f"No valid runs found in {file_path}")
        if run_index < 0 or run_index >= total_runs:
            raise ValueError(
                f"Invalid run_index {run_index} for {file_path} (total_runs={total_runs})"
            )
        pr = piv_result[run_index]
        ux = np.asarray(pr.ux)
        uy = np.asarray(pr.uy)
        stacked = _stack_velocity_components(pr, ux, uy)[None, ...]  # (1, 3_or_4, H, W)
        return stacked

    # Single run struct
    if run_index is not None and run_index != 0:
        raise ValueError(
            f"Invalid run_index {run_index} for single-run file {file_path}"
        )
    pr = piv_result
    ux = np.asarray(pr.ux)
    uy = np.asarray(pr.uy)
    stacked = _stack_velocity_components(pr, ux, uy)[None, ...]  # (1, 3_or_4, H, W)
    return stacked


def load_vectors_from_directory(
    data_dir: Path, config: Config, runs: Optional[Sequence[int]] = None
) -> da.Array:
    """
    Load .mat vector files for requested runs.
    - runs: list of 1-based run numbers to include; if None or empty, include all runs in the files.
    Returns Dask array with shape (N_existing, R, C, H, W) where C=3 for 2D or C=4 for stereo.
    """
    data_dir = Path(data_dir)
    fmt = config.vector_format  # e.g. "B%05d.mat"
    expected_paths = [
        data_dir / (fmt % i) for i in range(1, config.num_frame_pairs + 1)
    ]
    existing_paths = [p for p in expected_paths if p.exists()]

    missing_count = len(expected_paths) - len(existing_paths)
    if missing_count == len(expected_paths):
        raise FileNotFoundError(
            f"No vector files found using pattern {fmt} in {data_dir}"
        )
    if missing_count:
        warnings.warn(
            f"{missing_count} vector files missing in {data_dir} (loaded {len(existing_paths)})"
        )

    # Convert runs (1-based) to zero-based indices for reading
    zero_based_runs: Optional[Sequence[int]] = None
    if runs:
        zero_based_runs = [r - 1 for r in runs]

    # Detect shape/dtype from first readable file
    first_arr = None
    for p in existing_paths:
        try:
            first_arr = read_mat_contents(
                str(p), run_index=zero_based_runs[0] if zero_based_runs else None
            )
            # Debugging: print shape, dtype, and file info
            if first_arr.ndim != 4:
                warnings.warn(
                    f"[DEBUG] Unexpected array ndim={first_arr.ndim} in {p.name}"
                )
            break
        except Exception as e:
            warnings.warn(f"Failed to read {p.name} during probing: {e}")
            raise
    if first_arr is None:
        raise FileNotFoundError(f"Could not read any valid vector files in {data_dir}")

    shape, dtype = first_arr.shape, first_arr.dtype  # (R, C, H, W) where C=3 or 4

    delayed_items = [
        dask.delayed(read_mat_contents)(
            str(p), run_index=zero_based_runs[0] if zero_based_runs else None
        )
        for p in existing_paths
    ]
    arrays = [da.from_delayed(di, shape=shape, dtype=dtype) for di in delayed_items]
    stacked = da.stack(arrays, axis=0)  # (N, R, 3, H, W)
    return stacked.rechunk({0: config.piv_chunk_size})


def load_coords_from_directory(
    data_dir: Path, runs: Optional[Sequence[int]] = None
) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
    """
    Locate and read the coordinates.mat file in data_dir and return (x_list, y_list).
    - runs: list of 1-based run numbers to include; if None or empty, include all runs present in the coords file.
    - Returns:
        x_list: list of x arrays in the same order as 'runs' (or all runs if None)
        y_list: list of y arrays in the same order as 'runs' (or all runs if None)
    """
    data_dir = Path(data_dir)
    coords_path = data_dir / "coordinates.mat"
    if not coords_path.exists():
        raise FileNotFoundError(f"No coordinates.mat file found in {data_dir}")

    mat = scipy.io.loadmat(coords_path, struct_as_record=False, squeeze_me=True)
    if "coordinates" not in mat:
        raise KeyError(f"'coordinates' variable not found in {coords_path.name}")
    coords = mat["coordinates"]

    def _xy_from_struct(obj):
        return np.asarray(obj.x), np.asarray(obj.y)

    x_list, y_list = [], []

    if isinstance(coords, np.ndarray) and coords.dtype == object:
        if runs:
            zero_based = [r - 1 for r in runs if 1 <= r <= coords.size]
            if len(zero_based) != len(runs):
                missing = sorted(set(runs) - set([z + 1 for z in zero_based]))
                warnings.warn(
                    f"Skipping out-of-range run indices {missing} for coordinates"
                )
        else:
            zero_based = list(range(coords.size))

        for idx in zero_based:
            x, y = _xy_from_struct(coords[idx])
            x_list.append(x)
            y_list.append(y)
    else:
        if runs and 1 not in runs:
            warnings.warn(
                "Requested runs do not include run 1 present in coordinates; returning empty coords"
            )
            return [], []
        x, y = _xy_from_struct(coords)
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list


# =============================================================================
# Variable Extraction API
# =============================================================================


# Variables to exclude from plotting (metadata, not 2D plottable)
EXCLUDED_VARS = {"win_ctrs_x", "win_ctrs_y", "window_size", "n_windows", "predictor_field"}


def get_plottable_vars(
    file_path: Union[str, Path],
    var_name: str = "piv_result",
    fields_to_check: Sequence[str] = ("ux", "uy"),
) -> List[str]:
    """
    Extract names of 2D plottable variables from a .mat file.

    Args:
        file_path: Path to .mat file
        var_name: Variable name in mat file ("piv_result" or "ensemble_result")
        fields_to_check: Fields to use for finding valid struct element

    Returns:
        List of variable names that are 2D arrays suitable for plotting
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return []

    mat = scipy.io.loadmat(str(file_path), struct_as_record=False, squeeze_me=True)
    if var_name not in mat:
        return []

    data = mat[var_name]

    # Get first valid struct element
    if isinstance(data, np.ndarray) and data.dtype == object and data.size > 0:
        struct = None
        for el in data.flat:
            if is_run_valid(el, fields=fields_to_check, require_2d=False, reject_all_nan=False):
                struct = el
                break
        if struct is None and data.size > 0:
            struct = data.flat[0]
    else:
        struct = data

    return get_plottable_vars_from_struct(struct)


def get_plottable_vars_from_struct(struct: Any) -> List[str]:
    """
    Extract names of 2D plottable variables from a loaded struct.

    Args:
        struct: A loaded MATLAB struct with field attributes

    Returns:
        List of variable names that are 2D arrays suitable for plotting
    """
    if struct is None:
        return []

    # Get all attribute names
    all_vars = [n for n in dir(struct) if not n.startswith("_") and not callable(getattr(struct, n, None))]

    plottable = []
    for var_name in all_vars:
        if var_name in EXCLUDED_VARS:
            continue
        try:
            val = getattr(struct, var_name, None)
            if val is None:
                continue
            arr = np.asarray(val)
            if arr.ndim == 2 and arr.size > 0:
                plottable.append(var_name)
        except Exception:
            continue

    return plottable


def find_valid_ensemble_runs(
    file_path: Union[str, Path], one_based: bool = False
) -> RunValidationResult:
    """
    Find valid runs in an ensemble_result file (checks ux, uy).

    Args:
        file_path: Path to .mat file containing ensemble_result
        one_based: If True, return 1-based indices; if False, 0-based

    Returns:
        RunValidationResult with valid_runs list, total_runs count, and single_run flag
    """
    return find_valid_runs(
        file_path, var_name="ensemble_result", fields=("ux", "uy"), one_based=one_based
    )


# =============================================================================
# Mask I/O
# =============================================================================


def save_mask_to_mat(file_path: str, mask: np.ndarray, polygons):
    """
    Save the given mask array to a .mat file.
    """
    scipy.io.savemat(file_path, {"mask": mask, "polygons": polygons}, do_compression=True)


def read_mask_from_mat(file_path: str):
    """
    Read the mask and polygons from a .mat file.
    Returns:
        mask: np.ndarray
        polygons: list of dicts with fields 'index', 'name', 'points'
    """
    # Load without squeeze_me to avoid 0-d array issues with single-element cells
    # Use struct_as_record=True (default) so structs become record arrays with dict-like access
    mat = scipy.io.loadmat(file_path, squeeze_me=False, struct_as_record=True)
    mask = mat.get("mask", None)
    polygons_raw = mat.get("polygons", None)
    if mask is None or polygons_raw is None:
        raise ValueError(f"Missing 'mask' or 'polygons' in {file_path}")

    # Squeeze the mask manually if needed
    mask = np.squeeze(mask)
    
    # polygons_raw is a numpy object array (MATLAB cell array)
    # Flatten it to iterate (it might be [[obj1], [obj2]] or [[obj]])
    polygons_flat = polygons_raw.flatten()
    
    polygons = []
    for poly in polygons_flat:
        # poly is a structured array (record) with named fields accessible via indexing
        # Extract scalar values from 0-d arrays
        idx_raw = poly['index'] if isinstance(poly, np.void) else poly['index'][0, 0]
        name_raw = poly['name'] if isinstance(poly, np.void) else poly['name'][0, 0]
        pts_raw = poly['points'] if isinstance(poly, np.void) else poly['points'][0, 0]
        
        idx = int(idx_raw.item() if hasattr(idx_raw, 'item') else idx_raw)
        name = str(name_raw.item() if hasattr(name_raw, 'item') else name_raw)
        
        # pts might be a 2D array, convert to list of lists
        points = pts_raw.tolist() if isinstance(pts_raw, np.ndarray) else list(pts_raw)
        polygons.append({"index": idx, "name": name, "points": points})

    return mask, polygons
