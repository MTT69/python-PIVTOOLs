"""PIV output sets in the exact on-disk layout the calibration apply path reads.

The layout mirrors a real instantaneous run (``uncalibrated_piv/N/CamX/instantaneous``):

* ``coordinates.mat`` holds a ``coordinates`` struct array of shape ``(1, n_passes)``
  with fields ``x``, ``y`` (1-based MATLAB pixel positions, full 2D meshes). An
  unfilled pass is an empty float array.
* ``NNNNN.mat`` holds a ``piv_result`` struct array of shape ``(1, n_passes)``. Pass 0
  is empty in real runs; filled passes carry float64 ``ux``/``uy`` and uint8
  ``b_mask``/``nan_mask`` plus the window-centre vectors.
* ``ensemble_result.mat`` holds an ``ensemble_result`` struct per pass with ``ux``,
  ``uy`` and one ``UU``/``VV``/``UV`` Reynolds-stress triple.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.io

# (rows, cols) of each pass' vector grid; None is an empty pass. Pass 0 is empty in
# real runs and the first filled pass is the coarse one.
DEFAULT_PASSES: Tuple[Optional[Tuple[int, int]], ...] = (None, (6, 8), (12, 16))


@dataclass(frozen=True)
class GridSpec:
    """Where the synthetic vector grid sits on the image.

    ``origin_px`` is the first window centre (0-based pixels) and ``extent_px`` the
    ``(width, height)`` the grid spans, shared by every pass so finer passes cover the
    same field of view. The default is a small grid in the image corner (fine for mono,
    where only the pixel -> mm map matters); stereo needs a grid inside both cameras'
    overlap, so pass one centred on the principal point.
    """

    origin_px: Tuple[float, float] = (16.0, 16.0)
    extent_px: Tuple[float, float] = (256.0, 192.0)


# Frame k carries displacement k * (BASE_UX_PX, BASE_UY_PX) px/frame everywhere, so a
# calibrated file's magnitude identifies its frame as well as checking the scale.
BASE_UX_PX = 0.5
BASE_UY_PX = -0.25

# Ensemble stress triple (px^2/frame^2): arbitrary but non-degenerate and distinct.
ENSEMBLE_UU_PX2 = 0.04
ENSEMBLE_VV_PX2 = 0.02
ENSEMBLE_UV_PX2 = -0.005

ENSEMBLE_FILE = "ensemble_result.mat"
VECTOR_NAME_FORMAT = "{:05d}.mat"


def grid_px(
    shape: Tuple[int, int], grid: GridSpec = GridSpec()
) -> Tuple[np.ndarray, np.ndarray]:
    """0-based image-down pixel meshes ``(x, y)`` of a pass (what ``read_coordinates`` returns)."""
    rows, cols = shape
    xs = grid.origin_px[0] + grid.extent_px[0] / cols * np.arange(cols)
    ys = grid.origin_px[1] + grid.extent_px[1] / rows * np.arange(rows)
    return np.meshgrid(xs, ys)


def _struct(fields: Dict[str, list], n_passes: int) -> np.ndarray:
    """A ``(1, n_passes)`` object struct array, the shape MATLAB-style savers produce."""
    st = np.empty((1, n_passes), dtype=[(k, object) for k in fields])
    for k, vals in fields.items():
        for i, v in enumerate(vals):
            st[k][0, i] = v
    return st


def _empty() -> np.ndarray:
    return np.empty((0, 0), dtype=np.float32)


def write_coordinates(
    out_dir: Path,
    passes: Sequence[Optional[Tuple[int, int]]],
    grid: GridSpec = GridSpec(),
) -> Path:
    """Write ``coordinates.mat`` (1-based MATLAB pixels) for ``passes``."""
    xs, ys = [], []
    for shape in passes:
        if shape is None:
            xs.append(_empty())
            ys.append(_empty())
        else:
            gx, gy = grid_px(shape, grid)
            xs.append(gx + 1.0)
            ys.append(gy + 1.0)
    path = Path(out_dir) / "coordinates.mat"
    scipy.io.savemat(
        str(path), {"coordinates": _struct({"x": xs, "y": ys}, len(passes))}
    )
    return path


def _frame_piv_result(
    frame: int, passes: Sequence[Optional[Tuple[int, int]]], grid: GridSpec
) -> np.ndarray:
    cols: Dict[str, list] = {
        k: []
        for k in (
            "ux",
            "uy",
            "b_mask",
            "nan_mask",
            "win_ctrs_x",
            "win_ctrs_y",
        )
    }
    for shape in passes:
        if shape is None:
            for k in cols:
                cols[k].append(_empty())
            continue
        rows, ncols = shape
        gx, gy = grid_px(shape, grid)
        cols["ux"].append(np.full(shape, frame * BASE_UX_PX, dtype=np.float64))
        cols["uy"].append(np.full(shape, frame * BASE_UY_PX, dtype=np.float64))
        cols["b_mask"].append(np.zeros(shape, dtype=np.uint8))
        cols["nan_mask"].append(np.zeros(shape, dtype=np.uint8))
        cols["win_ctrs_x"].append(gx[0:1, :] + 1.0)
        cols["win_ctrs_y"].append(gy[:, 0:1].T + 1.0)
    return _struct(cols, len(passes))


def _ensemble_result(passes: Sequence[Optional[Tuple[int, int]]]) -> np.ndarray:
    cols: Dict[str, list] = {k: [] for k in ("ux", "uy", "UU", "VV", "UV")}
    for shape in passes:
        if shape is None:
            for k in cols:
                cols[k].append(_empty())
            continue
        cols["ux"].append(np.full(shape, BASE_UX_PX, dtype=np.float64))
        cols["uy"].append(np.full(shape, BASE_UY_PX, dtype=np.float64))
        cols["UU"].append(np.full(shape, ENSEMBLE_UU_PX2, dtype=np.float64))
        cols["VV"].append(np.full(shape, ENSEMBLE_VV_PX2, dtype=np.float64))
        cols["UV"].append(np.full(shape, ENSEMBLE_UV_PX2, dtype=np.float64))
    return _struct(cols, len(passes))


def write_vector_set(
    out_dir: Path,
    n_frames: int,
    passes: Sequence[Optional[Tuple[int, int]]] = DEFAULT_PASSES,
    missing: Sequence[int] = (),
    with_ensemble: bool = True,
    grid: GridSpec = GridSpec(),
) -> Dict[str, object]:
    """Write one camera's uncalibrated PIV directory in the real layout.

    Parameters
    ----------
    out_dir
        Created if absent. Receives ``coordinates.mat``, ``00001.mat`` ..
        ``{n_frames:05d}.mat`` and (optionally) ``ensemble_result.mat``.
    n_frames
        Frames numbered from 1.
    passes
        Per-pass ``(rows, cols)``; ``None`` is an empty pass (pass 0 in real runs).
    missing
        1-based frame numbers to leave out, to build a camera whose frame set differs
        from its partner's.
    with_ensemble
        Also write ``ensemble_result.mat`` with one UU/VV/UV triple.
    grid
        Placement of the vector grid on the image.

    Returns
    -------
    dict
        ``frames`` (the 1-based frame numbers written), ``passes`` and ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = {int(m) for m in missing}
    unknown = skip - set(range(1, n_frames + 1))
    if unknown:
        raise ValueError(f"missing frames {sorted(unknown)} not in 1..{n_frames}")
    write_coordinates(out_dir, passes, grid)
    written = []
    for k in range(1, n_frames + 1):
        if k in skip:
            continue
        scipy.io.savemat(
            str(out_dir / VECTOR_NAME_FORMAT.format(k)),
            {"piv_result": _frame_piv_result(k, passes, grid)},
        )
        written.append(k)
    if with_ensemble:
        scipy.io.savemat(
            str(out_dir / ENSEMBLE_FILE), {"ensemble_result": _ensemble_result(passes)}
        )
    return {"frames": written, "passes": tuple(passes), "out_dir": out_dir}


def write_stereo_vector_sets(
    cam1_dir: Path,
    cam2_dir: Path,
    n_frames: int,
    passes: Sequence[Optional[Tuple[int, int]]] = DEFAULT_PASSES,
    missing_in_cam2: Sequence[int] = (),
    with_ensemble: bool = False,
    grid: GridSpec = GridSpec(),
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Two cameras' vector sets; ``missing_in_cam2`` frames exist only for camera 1."""
    one = write_vector_set(
        cam1_dir, n_frames, passes, with_ensemble=with_ensemble, grid=grid
    )
    two = write_vector_set(
        cam2_dir,
        n_frames,
        passes,
        missing=missing_in_cam2,
        with_ensemble=with_ensemble,
        grid=grid,
    )
    return one, two


def apple_double(directory: Path) -> Path:
    """Drop a ``._*`` AppleDouble twin beside every file, as macOS does on external drives.

    The twin is binary junk, so any code that globs it without the dot-name filter
    chokes on it. Returns ``directory``.
    """
    directory = Path(directory)
    for f in [
        p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")
    ]:
        (directory / f"._{f.name}").write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 60)
    return directory


def config_with_cameras(
    cameras: Sequence[int],
    base_paths: Sequence[Path] = (),
    num_frame_pairs: int = 10,
    vector_format: str = "%05d.mat",
) -> types.SimpleNamespace:
    """Config stand-in exposing what the apply route and planner read.

    ``camera_numbers`` is ``list(cameras)`` (e.g. ``[2, 3, 4]`` for a rig whose first
    camera is not 1).
    """
    return types.SimpleNamespace(
        camera_numbers=[int(c) for c in cameras],
        base_paths=[Path(b) for b in base_paths],
        num_frame_pairs=int(num_frame_pairs),
        vector_format=vector_format,
    )
