"""calibration2.runio — read/write the production PIV layout for apply.

Reads the uncalibrated PIV output (``coordinates.mat`` + per-frame ``B*.mat``)
written by ``pivtools_cli.piv.save_results`` and writes calibrated output in the
same struct layout, so downstream tooling is unchanged. Crosses the MATLAB/pixel
boundary via ``frames`` (the saved coords are 1-based image-down).

Calibrated coordinates are world mm; calibrated velocities are m/s. No implicit
Y-flips — the sign is carried by the camera model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.io

from . import frames
from .apply import calibrate_coordinates, calibrate_displacements
from .camera_model import CameraModel
from .record import MonoRecord, StereoRecord


# .mat files in a PIV output dir that are NOT per-frame vector files. A vector glob
# derived from "%05d.mat" is "*.mat", which would otherwise sweep these in.
_NON_VECTOR_MATS = {"coordinates.mat", "mask.mat"}


def _vector_files(uncal_dir: Path, vector_glob: str) -> List[Path]:
    """Per-frame vector files in ``uncal_dir`` matching ``vector_glob``, sorted.

    Excludes non-vector ``.mat`` files (coordinates/mask) and hidden / macOS
    AppleDouble sidecars (``._00001.mat`` on external drives) that a bare ``*.mat``
    glob would otherwise sweep in and choke ``read_vectors``.
    """
    return sorted(
        b for b in Path(uncal_dir).glob(vector_glob)
        if b.name not in _NON_VECTOR_MATS and not b.name.startswith(".")
    )


def _as2d(a) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype == object:
        arr = arr.item()
    return np.asarray(arr, dtype=np.float64)


def read_coordinates(coords_path: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return {pass_idx: (x_px, y_px)} image-down pixel grids (MATLAB 1-based -> 0-based)."""
    mat = scipy.io.loadmat(str(coords_path), squeeze_me=True)
    coords = mat["coordinates"]
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    n = coords.shape[0] if coords.ndim else 1
    xfield = coords["x"]
    yfield = coords["y"]
    for i in range(n):
        x = _as2d(xfield[i] if n > 1 else xfield)
        y = _as2d(yfield[i] if n > 1 else yfield)
        if x.size == 0:
            continue
        out[i] = (frames.matlab_to_pixel(x), frames.matlab_to_pixel(y))
    return out


def read_vectors(bmat_path: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Return {pass_idx: (ux_px, uy_px, b_mask)} from a per-frame B*.mat."""
    mat = scipy.io.loadmat(str(bmat_path), squeeze_me=True)
    pr = mat["piv_result"]
    out: Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
    n = pr.shape[0] if pr.ndim else 1
    uxf, uyf = pr["ux"], pr["uy"]
    bmf = pr["b_mask"] if "b_mask" in pr.dtype.names else None
    for i in range(n):
        ux = _as2d(uxf[i] if n > 1 else uxf)
        uy = _as2d(uyf[i] if n > 1 else uyf)
        if ux.size == 0:
            continue
        b = None
        if bmf is not None:
            bb = bmf[i] if n > 1 else bmf
            bb = np.asarray(bb.item() if np.asarray(bb).dtype == object else bb)
            b = bb if bb.size else None
        out[i] = (ux, uy, b)
    return out


def _coords_struct(per_pass: Dict[int, Tuple[np.ndarray, np.ndarray]], n_passes: int) -> np.ndarray:
    st = np.empty((n_passes,), dtype=[("x", object), ("y", object)])
    empty = np.array([], dtype=np.float64)
    for i in range(n_passes):
        if i in per_pass:
            st["x"][i], st["y"][i] = per_pass[i]
        else:
            st["x"][i] = empty
            st["y"][i] = empty
    return st


def _vectors_struct(per_pass: Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]], n_passes: int) -> np.ndarray:
    st = np.empty((n_passes,), dtype=[("ux", object), ("uy", object), ("b_mask", object)])
    empty = np.array([], dtype=np.float64)
    for i in range(n_passes):
        if i in per_pass:
            ux, uy, b = per_pass[i]
            st["ux"][i], st["uy"][i] = ux, uy
            st["b_mask"][i] = b if b is not None else empty
        else:
            st["ux"][i] = st["uy"][i] = st["b_mask"][i] = empty
    return st


def calibrate_mono_run(
    record: MonoRecord,
    uncal_dir: Path,
    out_dir: Path,
    dt: float,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    vector_glob: str = "B*.mat",
) -> List[str]:
    """Apply a mono model to a whole run: coordinates.mat + every vector .mat.

    Reads ``<uncal_dir>/coordinates.mat`` and ``<uncal_dir>/<vector_glob>``; writes
    calibrated world-mm coordinates and m/s velocities to ``<out_dir>``. ``vector_glob``
    matches the per-frame vector files (default ``B*.mat``; PIV's ``vector_fmt`` may
    differ, e.g. ``[0-9]*.mat``). ``progress_cb(done, total)`` runs after each frame.
    """
    uncal_dir = Path(uncal_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = record.camera_model

    coords_px = read_coordinates(uncal_dir / "coordinates.mat")
    n_passes = (max(coords_px) + 1) if coords_px else 1

    # Calibrated coordinates (world mm), per pass.
    cal_coords: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for i, (xpx, ypx) in coords_px.items():
        stacked = np.stack([xpx, ypx], axis=-1)
        world = calibrate_coordinates(model, stacked, z_world, tilt_x, tilt_y)
        cal_coords[i] = (world[..., 0], world[..., 1])
    scipy.io.savemat(
        str(out_dir / "coordinates.mat"),
        {"coordinates": _coords_struct(cal_coords, n_passes)},
        oned_as="row", do_compression=True,
    )

    written: List[str] = []
    bmats = _vector_files(uncal_dir, vector_glob)
    total = len(bmats)
    for k, bmat in enumerate(bmats):
        vecs = read_vectors(bmat)
        cal_vecs: Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
        for i, (ux, uy, b) in vecs.items():
            if i not in coords_px:
                continue
            xpx, ypx = coords_px[i]
            coords_stack = np.stack([xpx, ypx], axis=-1)
            disp_stack = np.stack([ux, uy], axis=-1)
            u, v = calibrate_displacements(model, coords_stack, disp_stack, dt, z_world, tilt_x, tilt_y)
            cal_vecs[i] = (u, v, b)
        out_b = out_dir / bmat.name
        scipy.io.savemat(
            str(out_b),
            {"piv_result": _vectors_struct(cal_vecs, n_passes)},
            oned_as="row", do_compression=True,
        )
        written.append(str(out_b))
        if progress_cb:
            progress_cb(k + 1, total)
    return written


def reconstruct_stereo_run(
    record: StereoRecord,
    uncal_dir1: Path,
    uncal_dir2: Path,
    out_dir: Path,
    dt: float,
    pass_index: Optional[int] = None,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    vector_glob: str = "B*.mat",
) -> List[str]:
    """3C reconstruction over a run: cam1 + cam2 uncalibrated -> stereo (U,V,W) m/s.

    Writes world coordinates (x,y,z mm) and 3C velocities (ux,uy,uz m/s) on cam1's
    grid into ``out_dir``. Uses the final populated pass unless ``pass_index`` set.
    """
    from .stereo_model import reconstruct_3c_field

    uncal_dir1 = Path(uncal_dir1)
    uncal_dir2 = Path(uncal_dir2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords1 = read_coordinates(uncal_dir1 / "coordinates.mat")
    coords2 = read_coordinates(uncal_dir2 / "coordinates.mat")
    if not coords1 or not coords2:
        raise RuntimeError("missing coordinates.mat for stereo reconstruction")
    p = pass_index if pass_index is not None else max(coords1)
    x1, y1 = coords1[p]
    x2, y2 = coords2[p]
    c1 = np.stack([x1, y1], axis=-1)
    c2 = np.stack([x2, y2], axis=-1)

    written: List[str] = []
    coords_written = False
    bmats1 = {b.name: b for b in _vector_files(uncal_dir1, vector_glob)}
    bmats2 = {b.name: b for b in _vector_files(uncal_dir2, vector_glob)}
    common = sorted(set(bmats1) & set(bmats2))
    total = len(common)
    for k, name in enumerate(common):
        v1 = read_vectors(bmats1[name])
        v2 = read_vectors(bmats2[name])
        if p not in v1 or p not in v2:
            continue
        ux1, uy1, _ = v1[p]
        ux2, uy2, _ = v2[p]
        wx, wy, wz, U, V, Wc = reconstruct_3c_field(
            record.model1, record.model2, c1, ux1, uy1, c2, ux2, uy2,
            dt, z_world, tilt_x, tilt_y,
        )
        n_passes = p + 1
        # The reconstructed world grid (wx, wy) is frame-invariant — write coordinates
        # once, on the first processed frame, not once per frame.
        if not coords_written:
            cs = _coords_struct({p: (wx, wy)}, n_passes)
            scipy.io.savemat(str(out_dir / "coordinates.mat"),
                             {"coordinates": cs}, oned_as="row", do_compression=True)
            coords_written = True
        st = np.empty((n_passes,), dtype=[("ux", object), ("uy", object), ("uz", object)])
        empty = np.array([], dtype=np.float64)
        for i in range(n_passes):
            if i == p:
                st["ux"][i], st["uy"][i], st["uz"][i] = U, V, Wc
            else:
                st["ux"][i] = st["uy"][i] = st["uz"][i] = empty
        out_b = out_dir / name
        scipy.io.savemat(str(out_b), {"piv_result": st}, oned_as="row", do_compression=True)
        written.append(str(out_b))
        if progress_cb:
            progress_cb(k + 1, total)
    return written
