import warnings
from pathlib import Path
from typing import Optional, Sequence, Tuple

import dask
import dask.array as da
import numpy as np
import scipy.io

from .config import Config


def read_mat_contents(
    file_path: str, run_index: Optional[int] = None, return_all_runs: bool = False
) -> np.ndarray:
    """
    Reads piv_result from a .mat file.
    If multiple runs are present, selects the specified run_index (0-based).
    If run_index is None, selects the first run with valid (non-empty) data.
    If return_all_runs is True, returns all runs in shape (R, 3, H, W).
    Otherwise returns shape (1, 3, H, W) for the selected run.
    """
    mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    piv_result = mat["piv_result"]

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
                b_mask = (
                    np.asarray(pr.b_mask).astype(ux.dtype, copy=False)
                    if ux.size > 0
                    else np.array([])
                )
                if ux.size > 0 and uy.size > 0:
                    stacked = np.stack([ux, uy, b_mask], axis=0)  # (3, H, W)
                else:
                    # Empty run - create placeholder with consistent shape if possible
                    stacked = np.array([[], [], []])  # Will be reshaped later
                all_runs.append(stacked)
            return np.array(all_runs)  # (R, 3, H, W)

        # Single run selection (existing logic)
        if run_index is None:
            # Find first valid run (non-empty ux, uy)
            for idx in range(total_runs):
                pr = piv_result[idx]
                ux = np.asarray(pr.ux)
                uy = np.asarray(pr.uy)
                if ux.size > 0 and uy.size > 0:
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
        b_mask = np.asarray(pr.b_mask).astype(ux.dtype, copy=False)
        stacked = np.stack([ux, uy, b_mask], axis=0)[None, ...]  # (1, 3, H, W)
        return stacked

    # Single run struct
    if run_index is not None and run_index != 0:
        raise ValueError(
            f"Invalid run_index {run_index} for single-run file {file_path}"
        )
    pr = piv_result
    ux = np.asarray(pr.ux)
    uy = np.asarray(pr.uy)
    b_mask = np.asarray(pr.b_mask).astype(ux.dtype, copy=False)

    if return_all_runs:
        stacked = np.stack([ux, uy, b_mask], axis=0)[None, ...]  # (1, 3, H, W)
        return stacked
    else:
        stacked = np.stack([ux, uy, b_mask], axis=0)[None, ...]  # (1, 3, H, W)
        return stacked


def load_vectors_from_directory(
    data_dir: Path, config: Config, runs: Optional[Sequence[int]] = None
) -> da.Array:
    """
    Load .mat vector files for requested runs.
    - runs: list of 1-based run numbers to include; if None or empty, include all runs in the files.
    Returns Dask array with shape (N_existing, R, 3, H, W).
    """
    data_dir = Path(data_dir)
    fmt = config.vector_format  # e.g. "B%05d.mat"
    expected_paths = [data_dir / (fmt % i) for i in range(1, config.num_images + 1)]
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

    shape, dtype = first_arr.shape, first_arr.dtype  # (R, 3, H, W), dtype

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
