from pathlib import Path
from typing import Tuple, Optional, Sequence
import numpy as np
import dask
import dask.array as da
import scipy.io
import warnings
from config import Config


def read_mat_contents(file_path: str, run_indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Reads piv_result from a .mat file.
    If multiple runs are present (object array of structs), returns stacked runs with shape (R, 3, H, W).
    Otherwise returns a single-run with shape (1, 3, H, W).
    run_indices are zero-based indices of runs to extract; if None, extract all runs.
    """
    mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    piv_result = mat["piv_result"]

    # Multiple runs case: numpy array of structs
    if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
        total_runs = piv_result.size
        indices = list(range(total_runs)) if run_indices is None else [i for i in run_indices if 0 <= i < total_runs]
        if run_indices is not None and len(indices) != len(run_indices):
            missing = sorted(set(run_indices) - set(indices))
            warnings.warn(f"Skipping out-of-range run indices {missing} for {file_path} (total_runs={total_runs})")

        run_arrays = []
        for idx in indices:
            pr = piv_result[idx]
            ux = np.asarray(pr.ux)
            uy = np.asarray(pr.uy)
            b_mask = np.asarray(pr.b_mask).astype(ux.dtype, copy=False)
            run_arrays.append(np.stack([ux, uy, b_mask], axis=0))
        return np.stack(run_arrays, axis=0)  # (R, 3, H, W)

    # Single run struct
    pr = piv_result
    ux = np.asarray(pr.ux)
    uy = np.asarray(pr.uy)
    b_mask = np.asarray(pr.b_mask).astype(ux.dtype, copy=False)
    return np.stack([ux, uy, b_mask], axis=0)[None, ...]  # (1, 3, H, W)


def load_vectors_from_directory(data_dir: Path, config: Config, runs: Optional[Sequence[int]] = None) -> da.Array:
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
        raise FileNotFoundError(f"No vector files found using pattern {fmt} in {data_dir}")
    if missing_count:
        warnings.warn(f"{missing_count} vector files missing in {data_dir} (loaded {len(existing_paths)})")

    # Convert runs (1-based) to zero-based indices for reading
    zero_based_runs: Optional[Sequence[int]] = None
    if runs:
        zero_based_runs = [r - 1 for r in runs]

    # Detect shape/dtype from first readable file
    first_arr = None
    for p in existing_paths:
        try:
            first_arr = read_mat_contents(str(p), run_indices=zero_based_runs)
            break
        except Exception as e:
            warnings.warn(f"Failed to read {p.name} during probing: {e}")
    if first_arr is None:
        raise FileNotFoundError(f"Could not read any valid vector files in {data_dir}")

    shape, dtype = first_arr.shape, first_arr.dtype  # (R, 3, H, W), dtype

    delayed_items = [
        dask.delayed(read_mat_contents)(str(p), run_indices=zero_based_runs)
        for p in existing_paths
    ]
    arrays = [da.from_delayed(di, shape=shape, dtype=dtype) for di in delayed_items]
    stacked = da.stack(arrays, axis=0)  # (N, R, 3, H, W)
    return stacked.rechunk({0: config.piv_chunk_size})
