"""Stale calibration files: real sidecars / records with one always-written field removed.

A stale file is one written by an older build, missing a field every current writer
emits. The fixtures here save a REAL sidecar or record through the production writer,
then delete one key (optionally nested) with ``scipy.io`` so the file is exactly what
an older writer would have left. ``drop_mat_field`` raises if the field was never there,
so a test cannot pass by deleting nothing.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.io

from pivtools_gui.calibration import inputs_store as INP
from pivtools_gui.calibration import record as REC
from pivtools_gui.calibration.camera_model import CameraModel, Polynomial3DModel
from pivtools_gui.calibration.detection.base import DetectionResult

FieldPath = Union[str, Sequence[str]]

# Plausible pinhole intrinsics shared by the record builders.
_FX_PX = 1500.0
_IMAGE_SIZE = (1280, 1024)
_N_DETECTED_POINTS = 6
_SIDECAR_DET_KEY = "0123456789abcdef"

# Always-written fields of each ``WorldFrame`` block.
WORLD_FRAME_FIELDS: Tuple[str, ...] = (
    "mode",
    "origin_px",
    "x_axis_px",
    "y_axis_px",
    "swap_axes",
    "col_sign",
    "row_sign",
    "origin_grid",
    "origin_mm",
    "world_offset_mm",
)


# ---------------------------------------------------------------------------
# Removing a field from a .mat file
# ---------------------------------------------------------------------------


def _drop_in_struct(arr: np.ndarray, path: Tuple[str, ...]) -> np.ndarray:
    """Remove the field at ``path`` from a loaded struct array (recursing into nested structs and cells)."""
    names = arr.dtype.names
    if names is None and arr.dtype == object:
        # A cell array (e.g. the joint record's ``camera_models``): drop from every cell.
        out = arr.copy()
        for idx in np.ndindex(arr.shape):
            out[idx] = _drop_in_struct(arr[idx], path)
        return out
    if names is None:
        raise KeyError(f"cannot descend into {path!r}: value is not a struct")
    head = path[0]
    if head not in names:
        raise KeyError(f"field {head!r} not in struct with fields {names}")
    if len(path) == 1:
        keep = [(n, arr.dtype[n]) for n in names if n != head]
        out = np.empty(arr.shape, dtype=keep)
        for n, _ in keep:
            out[n] = arr[n]
        return out
    out = arr.copy()
    for idx in np.ndindex(arr.shape):
        out[head][idx] = _drop_in_struct(arr[head][idx], path[1:])
    return out


# How ``record._meta_to_dict`` writes an empty block (a struct with no fields). scipy loads
# it back as an object array holding None, which it cannot write, so a rewrite restores it.
_EMPTY_BLOCK = {"_empty": 0}


def _restore_empty_blocks(value):
    """Re-encode loaded empty structs (object arrays of None) so the file can be re-saved."""
    if not isinstance(value, np.ndarray):
        return value
    if value.dtype.names is not None:
        out = value.copy()
        for n in value.dtype.names:
            for idx in np.ndindex(value.shape):
                out[n][idx] = _restore_empty_blocks(value[n][idx])
        return out
    if value.dtype == object and value.size and all(e is None for e in value.flat):
        return _EMPTY_BLOCK
    return value


def rewrite_mat(path: Path, drop: Optional[FieldPath] = None) -> Path:
    """Re-save ``path`` through ``scipy.io``, optionally deleting one (nested) field.

    ``drop`` is a top-level variable name, or a tuple descending into struct fields,
    e.g. ``("world_frame", "origin_px")`` or ``("camera_models", "cam_rms")``.
    With ``drop=None`` the file is rewritten unchanged — the control that shows the
    rewrite itself does not break the loader.
    """
    path = Path(path)
    mat = scipy.io.loadmat(str(path))
    if drop is not None:
        fp = (drop,) if isinstance(drop, str) else tuple(drop)
        if len(fp) == 1:
            if fp[0] not in mat:
                raise KeyError(f"{path.name}: variable {fp[0]!r} not present to drop")
            del mat[fp[0]]
        else:
            if fp[0] not in mat:
                raise KeyError(f"{path.name}: variable {fp[0]!r} not present")
            mat[fp[0]] = _drop_in_struct(mat[fp[0]], fp[1:])
    out = {
        k: _restore_empty_blocks(v) for k, v in mat.items() if not k.startswith("__")
    }
    with warnings.catch_warnings():
        # "_empty" starts with an underscore: scipy drops the field (which is the point).
        warnings.simplefilter("ignore", scipy.io.matlab.MatWriteWarning)
        scipy.io.savemat(str(path), out)
    return path


# ---------------------------------------------------------------------------
# Real sidecar
# ---------------------------------------------------------------------------


def _detection() -> DetectionResult:
    n = _N_DETECTED_POINTS
    r = np.arange(n, dtype=np.float64)
    return DetectionResult(
        success=True,
        board_type="dotboard",
        image_points=np.column_stack([r, r + 0.5]),
        board_local_points=np.column_stack([r, r, np.zeros(n)]),
        grid_indices=np.column_stack([r.astype(int), (r % 3).astype(int)]),
        board_to_pixel=np.eye(3) * 2.0,
        spacing_mm=15.0,
    )


def real_sidecar(model_dir: Path) -> Path:
    """Write a complete ``inputs.mat`` (detections, clicks, board params, key) and return its path."""
    return INP.save_inputs(
        model_dir,
        path_type="joint",
        board_type="dotboard",
        coords={"cameras": [1, 2], "datum_camera": 1},
        detections={1: [_detection(), _detection()], 2: [_detection()]},
        image_size_by_cam={1: _IMAGE_SIZE, 2: _IMAGE_SIZE},
        det_key=_SIDECAR_DET_KEY,
        board_params={"spacing_mm": 15.0},
    )


def stale_sidecar(model_dir: Path, drop_field: FieldPath) -> Path:
    """A real sidecar in ``model_dir`` with ``drop_field`` removed; returns the file path."""
    return rewrite_mat(real_sidecar(model_dir), drop_field)


# ---------------------------------------------------------------------------
# Real records
# ---------------------------------------------------------------------------


def _pinhole(tz: float = 700.0, yaw: float = 0.0) -> CameraModel:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    K = np.array([[_FX_PX, 0, 640.0], [0, _FX_PX, 512.0], [0, 0, 1.0]])
    return CameraModel(
        K=K,
        dist=np.array([-0.01, 0.02, 0.001, -0.0005, 0.0]),
        R=R,
        t=np.array([[1.0], [2.0], [tz]]),
        image_size=_IMAGE_SIZE,
        rms=0.2,
    )


def _poly3d() -> Polynomial3DModel:
    coeffs_u = np.zeros(20)
    coeffs_v = np.zeros(20)
    coeffs_u[1], coeffs_v[4] = 100.0, 100.0
    return Polynomial3DModel(
        coeffs_u=coeffs_u,
        coeffs_v=coeffs_v,
        x0=0.0,
        sx=100.0,
        y0=0.0,
        sy=100.0,
        z0=-1.5,
        sz=1.5,
        image_size=_IMAGE_SIZE,
        rms_px=0.1,
    )


def _world_frame() -> REC.WorldFrame:
    return REC.WorldFrame(
        mode="clicks",
        origin_px=np.array([10.0, 20.0]),
        x_axis_px=np.array([110.0, 20.0]),
        y_axis_px=np.array([10.0, 120.0]),
        origin_grid=np.array([0.0, 0.0]),
        origin_mm=np.array([5.0, -3.0]),
        world_offset_mm=np.array([1.0, 2.0]),
    )


def _mono(model_dir: Path) -> Path:
    rec = REC.MonoRecord(
        camera=1,
        board_type="dotboard",
        camera_model=_pinhole(),
        world_frame=_world_frame(),
        per_view_rms=[0.1, 0.2],
        board_meta={"spacing_mm": 15.0, "dt": 1e-4},
    )
    return REC.save_mono(rec, model_dir)


def _stereo_pinhole(model_dir: Path) -> Path:
    m1, m2 = _pinhole(yaw=0.2), _pinhole(yaw=-0.2)
    R = m2.R @ m1.R.T
    rec = REC.StereoRecord(
        cam1=1,
        cam2=2,
        board_type="dotboard",
        model1=m1,
        model2=m2,
        R_stereo=R,
        T_stereo=m2.t - R @ m1.t,
        world_frame=_world_frame(),
        per_view_rms1=[0.1],
        per_view_rms2=[0.12],
        board_meta={"spacing_mm": 15.0, "dt": 1e-4},
        self_cal={},
    )
    return REC.save_stereo(rec, model_dir)


def _stereo_poly(model_dir: Path) -> Path:
    rec = REC.StereoRecord(
        cam1=1,
        cam2=2,
        board_type="stepped",
        model1=_poly3d(),
        model2=_poly3d(),
        R_stereo=None,
        T_stereo=None,
        world_frame=_world_frame(),
        per_view_rms1=[0.1],
        per_view_rms2=[0.12],
        board_meta={"spacing_mm": 15.0, "dt": 1e-4},
        self_cal={},
    )
    return REC.save_stereo(rec, model_dir)


def _joint(model_dir: Path) -> Path:
    board = {
        (0, 0): np.array([0.0, 0.0, 0.0]),
        (1, 0): np.array([15.0, 0.0, 0.1]),
        (0, 1): np.array([0.0, 15.0, -0.2]),
    }
    rec = REC.JointRecord(
        cameras=[1, 2],
        board_type="dotboard",
        models={1: _pinhole(yaw=0.2), 2: _pinhole(yaw=-0.2)},
        board=board,
        world_frame=_world_frame(),
        spacing_mm=15.0,
        per_camera_rms={1: 0.18, 2: 0.21},
        rms_px=0.2,
        board_meta={"n_union": 3, "converged": 1, "dt": 1e-4},
        pose_diversity={"n_views": np.array([3, 3])},
    )
    return REC.save_joint(rec, model_dir)


# kind -> (writer, loader). ``stereo_poly`` is the polynomial3d stereo pair.
RECORD_KINDS: Dict[str, Tuple[Callable[[Path], Path], Callable[[Path], object]]] = {
    "mono": (_mono, REC.load_mono),
    "stereo": (_stereo_pinhole, REC.load_stereo),
    "stereo_poly": (_stereo_poly, REC.load_stereo),
    "joint": (_joint, REC.load_joint),
}


def real_record(kind: str, model_dir: Path) -> Path:
    """Write a complete record of ``kind`` (mono | stereo | stereo_poly | joint)."""
    return RECORD_KINDS[kind][0](model_dir)


def stale_record(kind: str, model_dir: Path, drop_field: FieldPath) -> Path:
    """A real record of ``kind`` in ``model_dir`` with ``drop_field`` removed; returns the file."""
    return rewrite_mat(real_record(kind, model_dir), drop_field)
