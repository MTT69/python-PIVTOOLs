"""calibration.record — one ``.mat`` model record, saved into the calibration source.

A single MATLAB-inspectable record per model, replacing the three legacy formats
(pinhole .mat, stereo .mat, polynomial in config.yaml). Models are written **into
the calibration source folder** (``config.get_calibration_source(idx)``), so a
project can hold many runs and each calibration lives with the images it came from.
No ``base_path/calibration`` fallback.

Directory layout (root = ``<source>/calibration`` or, for container sources,
``<source>.parent/calibration``):

    <root>/Cam{N}/{board}_planar/model/model.mat                 (mono)
    <root>/stereo_cam{A}_cam{B}/model/stereo_model.mat           (stereo)

Every record stamps ``contract_version`` and the resolved world frame so a stale
model can never be silently misread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.io import loadmat, savemat

from .camera_model import (
    CameraModel,
    DistortionModel,
    Polynomial3DModel,
    PolynomialModel,
    ScaleFactorModel,
)
from .frames import CONTRACT_VERSION

# A mono model is a 3D pinhole, a single-plane polynomial, a single-view 3D
# polynomial, or a uniform scale factor. All expose ``back_project_to_plane`` so the
# apply path treats them identically; the pinhole and 3D polynomial additionally
# expose ``project`` / ``jacobian`` for stereo reconstruction.
MonoModel = Union[CameraModel, PolynomialModel, Polynomial3DModel, ScaleFactorModel]


# ---------------------------------------------------------------------------
# Path resolution (into the calibration source folder)
# ---------------------------------------------------------------------------

def root_for_source(source: Path) -> Path:
    """Calibration-output root for an explicit source path.

    Directory sources -> ``<source>/calibration``. Container-file sources
    (``.set``/``.cine``) -> ``<source>.parent/calibration``.
    """
    source = Path(source)
    base = source.parent if source.suffix else source
    return base / "calibration"


def mono_model_dir_for_source(source: Path, camera: int, board: str) -> Path:
    return root_for_source(source) / f"Cam{camera}" / f"{board}_planar" / "model"


def stereo_model_dir_for_source(source: Path, cam1: int, cam2: int) -> Path:
    return root_for_source(source) / f"stereo_cam{cam1}_cam{cam2}" / "model"


def calibration_output_root(config, source_path_idx: int = 0) -> Path:
    """Root for calibration output, derived from the configured calibration SOURCE."""
    return root_for_source(config.get_calibration_source(source_path_idx))


def mono_model_dir(config, source_path_idx: int, camera: int, board: str) -> Path:
    return mono_model_dir_for_source(
        config.get_calibration_source(source_path_idx), camera, board
    )


def stereo_model_dir(config, source_path_idx: int, cam1: int, cam2: int) -> Path:
    return stereo_model_dir_for_source(
        config.get_calibration_source(source_path_idx), cam1, cam2
    )


# ---------------------------------------------------------------------------
# World frame provenance
# ---------------------------------------------------------------------------

@dataclass
class WorldFrame:
    """How the world frame was defined (provenance, stored in the record).

    ``mode`` is ``"clicks"`` (origin/+X/+Y picked on camera 1 and snapped to dots)
    or ``"default"`` (board-local min-corner origin, +X along the longer grid axis,
    +Y completing right-handed in the image-down sense). The clicks are stored for
    provenance; the resulting transform is baked into the camera ``R, t``.
    """

    mode: str = "default"
    origin_px: Optional[np.ndarray] = None
    x_axis_px: Optional[np.ndarray] = None
    y_axis_px: Optional[np.ndarray] = None
    # Resolved board-grid orientation. The world point for a feature at grid
    # (col, row) is, with du=col-origin_col, dv=row-origin_row:
    #   not swap_axes:  X = col_sign*du*spacing,  Y = row_sign*dv*spacing
    #       swap_axes:  X = col_sign*dv*spacing,  Y = row_sign*du*spacing
    # so col_sign is the +X sign, row_sign the +Y sign, swap selects which grid
    # delta feeds X. origin_grid is the (col,row) of the clicked origin dot,
    # letting the same frame be re-applied to other views/cameras by grid index.
    swap_axes: bool = False
    col_sign: int = 1
    row_sign: int = 1
    origin_grid: Optional[np.ndarray] = None
    # World (X, Y) mm assigned to the origin dot. None / absent == (0, 0). A non-zero
    # value translates every world point by this offset before the pose fit, so the
    # baked R, t (and hence the calibrated PIV coordinates) read in the user's frame.
    origin_mm: Optional[np.ndarray] = None
    # Per-camera (X, Y) mm placement into the SHARED multi-camera rig frame, computed
    # post-fit by the global-coordinate chain (datum + overlap pairs) and added to
    # every calibrated coordinate at apply time. None / absent == (0, 0) == this camera
    # is its own frame. Distinct from origin_mm (which is fit-time, baked into R, t):
    # world_offset_mm is applied AFTER back-projection. Regenerating the model builds a
    # fresh WorldFrame, so a stale global placement can never bleed into the next run.
    world_offset_mm: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class MonoRecord:
    camera: int
    board_type: str            # 'dotboard' | 'charuco'
    camera_model: MonoModel    # CameraModel (pinhole) or PolynomialModel
    world_frame: WorldFrame = field(default_factory=WorldFrame)
    per_view_rms: List[float] = field(default_factory=list)
    board_meta: Dict[str, Any] = field(default_factory=dict)
    contract_version: int = CONTRACT_VERSION


@dataclass
class StereoRecord:
    cam1: int
    cam2: int
    board_type: str
    # Pinhole (CameraModel) or single-view 3D polynomial (Polynomial3DModel), both in
    # the shared world frame. A polynomial pair has no extrinsic pose, so R/T are None.
    model1: Union[CameraModel, Polynomial3DModel]
    model2: Union[CameraModel, Polynomial3DModel]
    R_stereo: Optional[np.ndarray]   # cam1 -> cam2; None for a polynomial pair
    T_stereo: Optional[np.ndarray]   # cam1 -> cam2 (3,1); None for a polynomial pair
    world_frame: WorldFrame = field(default_factory=WorldFrame)
    per_view_rms1: List[float] = field(default_factory=list)
    per_view_rms2: List[float] = field(default_factory=list)
    board_meta: Dict[str, Any] = field(default_factory=dict)
    # Stereo self-calibration result (Wieneke disparity minimisation), written by a
    # post-model self-cal run. Empty == no self-cal == sheet at the datum plane.
    # Keys: z_offset (mm), tilt_x/tilt_y (rad), converged, final_rms_disparity,
    # n_iterations, n_images, window_size, overlap, source ("auto"|"manual").
    # Regenerating the stereo model builds a fresh record with this empty, so a stale
    # offset can never bleed into the next run.
    self_cal: Dict[str, Any] = field(default_factory=dict)
    contract_version: int = CONTRACT_VERSION

    @property
    def sc_z_offset(self) -> float:
        return float(self.self_cal.get("z_offset", 0.0))

    @property
    def sc_tilt_x(self) -> float:
        return float(self.self_cal.get("tilt_x", 0.0))

    @property
    def sc_tilt_y(self) -> float:
        return float(self.self_cal.get("tilt_y", 0.0))


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _empty_if_none(arr: Optional[np.ndarray]) -> np.ndarray:
    return np.array([]) if arr is None else np.asarray(arr, dtype=np.float64)


def _world_frame_to_dict(wf: WorldFrame) -> Dict[str, Any]:
    return {
        "mode": str(wf.mode),
        "origin_px": _empty_if_none(wf.origin_px),
        "x_axis_px": _empty_if_none(wf.x_axis_px),
        "y_axis_px": _empty_if_none(wf.y_axis_px),
        "swap_axes": int(bool(wf.swap_axes)),
        "col_sign": int(wf.col_sign),
        "row_sign": int(wf.row_sign),
        "origin_grid": _empty_if_none(wf.origin_grid),
        "origin_mm": _empty_if_none(wf.origin_mm),
        "world_offset_mm": _empty_if_none(wf.world_offset_mm),
    }


def _world_frame_from(obj) -> WorldFrame:
    def arr_or_none(v):
        a = np.asarray(v, dtype=np.float64).reshape(-1)
        return None if a.size == 0 else a
    return WorldFrame(
        mode=str(_scalar(getattr(obj, "mode", "default"))),
        origin_px=arr_or_none(getattr(obj, "origin_px", [])),
        x_axis_px=arr_or_none(getattr(obj, "x_axis_px", [])),
        y_axis_px=arr_or_none(getattr(obj, "y_axis_px", [])),
        swap_axes=bool(int(_scalar(getattr(obj, "swap_axes", 0)))),
        col_sign=int(_scalar(getattr(obj, "col_sign", 1))),
        row_sign=int(_scalar(getattr(obj, "row_sign", 1))),
        origin_grid=arr_or_none(getattr(obj, "origin_grid", [])),
        origin_mm=arr_or_none(getattr(obj, "origin_mm", [])),
        world_offset_mm=arr_or_none(getattr(obj, "world_offset_mm", [])),
    )


def _camera_to_dict(cm: CameraModel) -> Dict[str, Any]:
    return {
        "camera_matrix": np.asarray(cm.K, dtype=np.float64),
        "dist_coeffs": np.asarray(cm.dist, dtype=np.float64).reshape(1, -1),
        "R": np.asarray(cm.R, dtype=np.float64),
        "t": np.asarray(cm.t, dtype=np.float64).reshape(3, 1),
        "rvec": cm.rvec.reshape(3, 1),
        "image_width": int(cm.image_size[0]),
        "image_height": int(cm.image_size[1]),
        "distortion_model": str(cm.distortion_model.value),
        "rms": float(cm.rms),
    }


def _camera_from(obj) -> CameraModel:
    return CameraModel(
        K=np.asarray(obj.camera_matrix, dtype=np.float64).reshape(3, 3),
        dist=np.asarray(obj.dist_coeffs, dtype=np.float64).reshape(-1),
        R=np.asarray(obj.R, dtype=np.float64).reshape(3, 3),
        t=np.asarray(obj.t, dtype=np.float64).reshape(3, 1),
        image_size=(int(_scalar(obj.image_width)), int(_scalar(obj.image_height))),
        distortion_model=DistortionModel(str(_scalar(obj.distortion_model))),
        rms=float(_scalar(obj.rms)),
    )


def _polynomial_to_dict(pm: PolynomialModel) -> Dict[str, Any]:
    return {
        "coeffs_x": np.asarray(pm.coeffs_x, dtype=np.float64).reshape(1, -1),
        "coeffs_y": np.asarray(pm.coeffs_y, dtype=np.float64).reshape(1, -1),
        "x0": float(pm.x0),
        "sx": float(pm.sx),
        "y0": float(pm.y0),
        "sy": float(pm.sy),
        "image_width": int(pm.image_size[0]),
        "image_height": int(pm.image_size[1]),
        "rms_x_mm": float(pm.rms_x_mm),
        "rms_y_mm": float(pm.rms_y_mm),
    }


def _polynomial_from(obj) -> PolynomialModel:
    return PolynomialModel(
        coeffs_x=np.asarray(obj.coeffs_x, dtype=np.float64).reshape(-1),
        coeffs_y=np.asarray(obj.coeffs_y, dtype=np.float64).reshape(-1),
        x0=float(_scalar(obj.x0)),
        sx=float(_scalar(obj.sx)),
        y0=float(_scalar(obj.y0)),
        sy=float(_scalar(obj.sy)),
        image_size=(int(_scalar(obj.image_width)), int(_scalar(obj.image_height))),
        rms_x_mm=float(_scalar(obj.rms_x_mm)),
        rms_y_mm=float(_scalar(obj.rms_y_mm)),
    )


def _polynomial3d_to_dict(pm: Polynomial3DModel) -> Dict[str, Any]:
    return {
        "coeffs_u": np.asarray(pm.coeffs_u, dtype=np.float64).reshape(1, -1),
        "coeffs_v": np.asarray(pm.coeffs_v, dtype=np.float64).reshape(1, -1),
        "x0": float(pm.x0),
        "sx": float(pm.sx),
        "y0": float(pm.y0),
        "sy": float(pm.sy),
        "z0": float(pm.z0),
        "sz": float(pm.sz),
        "image_width": int(pm.image_size[0]),
        "image_height": int(pm.image_size[1]),
        "rms_px": float(pm.rms_px),
        "plane_rms_px": np.asarray(pm.plane_rms_px, dtype=np.float64).reshape(1, -1),
        "world_z_toward_camera": float(pm.world_z_toward_camera),
    }


def _polynomial3d_from(obj) -> Polynomial3DModel:
    return Polynomial3DModel(
        coeffs_u=np.asarray(obj.coeffs_u, dtype=np.float64).reshape(-1),
        coeffs_v=np.asarray(obj.coeffs_v, dtype=np.float64).reshape(-1),
        x0=float(_scalar(obj.x0)),
        sx=float(_scalar(obj.sx)),
        y0=float(_scalar(obj.y0)),
        sy=float(_scalar(obj.sy)),
        z0=float(_scalar(obj.z0)),
        sz=float(_scalar(obj.sz)),
        image_size=(int(_scalar(obj.image_width)), int(_scalar(obj.image_height))),
        rms_px=float(_scalar(obj.rms_px)),
        plane_rms_px=tuple(np.asarray(obj.plane_rms_px, dtype=np.float64).reshape(-1)),
        world_z_toward_camera=float(_scalar(obj.world_z_toward_camera)),
    )


def _scale_factor_to_dict(sf: ScaleFactorModel) -> Dict[str, Any]:
    return {
        "origin_px": np.asarray(sf.origin_px, dtype=np.float64).reshape(1, 2),
        "mm_per_pixel": float(sf.mm_per_pixel),
        "swap_axes": int(sf.swap_axes),
        "col_sign": int(sf.col_sign),
        "row_sign": int(sf.row_sign),
        "image_width": int(sf.image_size[0]),
        "image_height": int(sf.image_size[1]),
    }


def _scale_factor_from(obj) -> ScaleFactorModel:
    return ScaleFactorModel(
        origin_px=np.asarray(obj.origin_px, dtype=np.float64).reshape(2),
        mm_per_pixel=float(_scalar(obj.mm_per_pixel)),
        image_size=(int(_scalar(obj.image_width)), int(_scalar(obj.image_height))),
        swap_axes=int(_scalar(obj.swap_axes)),
        col_sign=int(_scalar(obj.col_sign)),
        row_sign=int(_scalar(obj.row_sign)),
    )


def _scalar(v):
    """Coerce a squeeze_me=loadmat scalar/0-d array to a python scalar/str."""
    a = np.asarray(v)
    if a.ndim == 0:
        return a.item()
    if a.size == 1:
        return a.reshape(-1)[0]
    return v


def _meta_to_dict(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, str):
            out[k] = v
        else:
            out[k] = np.asarray(v)
    return out or {"_empty": 0}


def _meta_from(obj) -> Dict[str, Any]:
    if obj is None:
        return {}
    out: Dict[str, Any] = {}
    for name in getattr(obj, "_fieldnames", []):
        if name == "_empty":
            continue
        out[name] = _scalar(getattr(obj, name))
    return out


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def _model_type_of(cm: MonoModel) -> str:
    """The ``model_type`` tag a model serialises under (the load dispatch key)."""
    if isinstance(cm, ScaleFactorModel):
        return "scale_factor"
    if isinstance(cm, Polynomial3DModel):
        return "polynomial3d"
    if isinstance(cm, PolynomialModel):
        return "polynomial"
    return "pinhole"


def _model_to_dict(cm: MonoModel, data: Dict[str, Any]) -> None:
    """Write the type-specific model block into ``data`` (keyed by model_type)."""
    if isinstance(cm, ScaleFactorModel):
        data["scale_factor_model"] = _scale_factor_to_dict(cm)
    elif isinstance(cm, Polynomial3DModel):
        data["polynomial3d_model"] = _polynomial3d_to_dict(cm)
    elif isinstance(cm, PolynomialModel):
        data["polynomial_model"] = _polynomial_to_dict(cm)
    else:
        data["camera_model"] = _camera_to_dict(cm)


def _model_from(mat, model_type: str) -> MonoModel:
    """Reconstruct a model from a loaded ``.mat`` given its ``model_type`` tag."""
    if model_type == "scale_factor":
        return _scale_factor_from(mat["scale_factor_model"])
    if model_type == "polynomial3d":
        return _polynomial3d_from(mat["polynomial3d_model"])
    if model_type == "polynomial":
        return _polynomial_from(mat["polynomial_model"])
    return _camera_from(mat["camera_model"])


def save_mono(record: MonoRecord, model_dir: Path) -> Path:
    """Write a mono model record to ``<model_dir>/model.mat``."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "model.mat"
    cm = record.camera_model
    data = {
        "contract_version": int(record.contract_version),
        "model_type": _model_type_of(cm),
        "board_type": str(record.board_type),
        "camera": int(record.camera),
        "world_frame": _world_frame_to_dict(record.world_frame),
        "per_view_rms": np.asarray(record.per_view_rms, dtype=np.float64).reshape(1, -1),
        "board_meta": _meta_to_dict(record.board_meta),
    }
    _model_to_dict(cm, data)
    savemat(str(path), data, oned_as="row")
    return path


def load_mono(path: Path) -> MonoRecord:
    path = Path(path)
    if path.is_dir():
        path = path / "model.mat"
    if not path.exists():
        raise FileNotFoundError(f"calibration model not found: {path}")
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    # Old files predate model_type and have only a pinhole "camera_model".
    model_type = str(_scalar(mat.get("model_type", "pinhole")))
    model: MonoModel = _model_from(mat, model_type)
    return MonoRecord(
        camera=int(_scalar(mat["camera"])),
        board_type=str(_scalar(mat["board_type"])),
        camera_model=model,
        world_frame=_world_frame_from(mat["world_frame"]),
        per_view_rms=list(np.asarray(mat["per_view_rms"], dtype=np.float64).reshape(-1)),
        board_meta=_meta_from(mat.get("board_meta")),
        contract_version=int(_scalar(mat["contract_version"])),
    )


def save_stereo(record: StereoRecord, model_dir: Path) -> Path:
    """Write a stereo model record to ``<model_dir>/stereo_model.mat``."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "stereo_model.mat"
    model_type = _model_type_of(record.model1)
    if model_type == "polynomial3d":
        m1, m2 = _polynomial3d_to_dict(record.model1), _polynomial3d_to_dict(record.model2)
    else:
        m1, m2 = _camera_to_dict(record.model1), _camera_to_dict(record.model2)
    data = {
        "contract_version": int(record.contract_version),
        "board_type": str(record.board_type),
        "model_type": model_type,
        "cam1": int(record.cam1),
        "cam2": int(record.cam2),
        "model1": m1,
        "model2": m2,
        # A polynomial pair has no extrinsic pose; store empty R/T (read back as None).
        "R_stereo": _empty_if_none(record.R_stereo).reshape(3, 3)
        if record.R_stereo is not None else np.array([]),
        "T_stereo": _empty_if_none(record.T_stereo).reshape(3, 1)
        if record.T_stereo is not None else np.array([]),
        "world_frame": _world_frame_to_dict(record.world_frame),
        "per_view_rms1": np.asarray(record.per_view_rms1, dtype=np.float64).reshape(1, -1),
        "per_view_rms2": np.asarray(record.per_view_rms2, dtype=np.float64).reshape(1, -1),
        "board_meta": _meta_to_dict(record.board_meta),
        "self_cal": _meta_to_dict(record.self_cal),
    }
    savemat(str(path), data, oned_as="row")
    return path


def load_stereo(path: Path) -> StereoRecord:
    path = Path(path)
    if path.is_dir():
        path = path / "stereo_model.mat"
    if not path.exists():
        raise FileNotFoundError(f"calibration stereo model not found: {path}")
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    # Old stereo files predate model_type and are pinhole-only.
    model_type = str(_scalar(mat.get("model_type", "pinhole")))
    if model_type == "polynomial3d":
        model1 = _polynomial3d_from(mat["model1"])
        model2 = _polynomial3d_from(mat["model2"])
    else:
        model1 = _camera_from(mat["model1"])
        model2 = _camera_from(mat["model2"])
    R_raw = np.asarray(mat["R_stereo"], dtype=np.float64).reshape(-1)
    T_raw = np.asarray(mat["T_stereo"], dtype=np.float64).reshape(-1)
    R_stereo = R_raw.reshape(3, 3) if R_raw.size == 9 else None
    T_stereo = T_raw.reshape(3, 1) if T_raw.size == 3 else None
    return StereoRecord(
        cam1=int(_scalar(mat["cam1"])),
        cam2=int(_scalar(mat["cam2"])),
        board_type=str(_scalar(mat["board_type"])),
        model1=model1,
        model2=model2,
        R_stereo=R_stereo,
        T_stereo=T_stereo,
        world_frame=_world_frame_from(mat["world_frame"]),
        per_view_rms1=list(np.asarray(mat["per_view_rms1"], dtype=np.float64).reshape(-1)),
        per_view_rms2=list(np.asarray(mat["per_view_rms2"], dtype=np.float64).reshape(-1)),
        board_meta=_meta_from(mat.get("board_meta")),
        self_cal=_meta_from(mat.get("self_cal")),
        contract_version=int(_scalar(mat["contract_version"])),
    )
