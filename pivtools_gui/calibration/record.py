"""calibration.record — one ``.mat`` model record, saved into the calibration source.

A single MATLAB-inspectable record per model, replacing the three legacy formats
(pinhole .mat, stereo .mat, polynomial in config.yaml). Models are written **into
the calibration source folder** (``config.get_calibration_source(idx)``), so a
project can hold many runs and each calibration lives with the images it came from.
No ``base_path/calibration`` fallback.

Directory layout (root = ``<source>/calibration`` or, for container sources,
``<source>.parent/calibration``):

    <root>/Cam{N}/{board}_planar/model/model_{model_type}.mat            (mono)
    <root>/stereo_cam{A}_cam{B}/model/stereo_model_{model_type}.mat      (stereo)

Filenames are per model type (``model_pinhole.mat`` / ``model_polynomial.mat`` ...)
so fitting a second type never clobbers the first. ``resolve_mono_path`` /
``resolve_stereo_path`` pick the file; when several types exist and no type is
requested, resolution errors listing them — never a silent pick.

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


def joint_model_dir_for_source(source: Path, board: str) -> Path:
    """Rig-level dir for the unified joint multi-camera record (one file, all cameras)."""
    return root_for_source(source) / f"joint_{board}" / "model"


def joint_model_dir(config, source_path_idx: int, board: str) -> Path:
    return joint_model_dir_for_source(config.get_calibration_source(source_path_idx), board)


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
class JointRecord:
    """Unified multi-camera calibration: all cameras + one shared released board, one file.

    The joint solve (``joint.run_joint``) puts every camera in ONE world frame and fits a
    single released board they all agree on. Storage is unified (this record), but downstream
    consumers still want a per-camera ``CameraModel``; ``models[cam]`` and the
    ``load_camera_model`` resolver provide that view without callers knowing the file is joint.
    """

    cameras: List[int]
    board_type: str                                  # 'dotboard' | 'charuco'
    models: Dict[int, CameraModel]                   # per-camera pinhole (datum-view pose)
    board: Dict[Tuple[int, int], np.ndarray]         # global index -> released (x,y,z) mm
    world_frame: WorldFrame = field(default_factory=WorldFrame)
    spacing_mm: float = 0.0
    board_release: str = "full3d"
    per_camera_rms: Dict[int, float] = field(default_factory=dict)
    rms_px: float = 0.0
    board_meta: Dict[str, Any] = field(default_factory=dict)
    model_type: str = "pinhole"
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
    """Mat-safe meta: strings pass through, dicts nest as structs, rest -> ndarray."""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, str):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = _meta_to_dict(v)
        else:
            out[k] = np.asarray(v)
    return out or {"_empty": 0}


def _meta_from(obj) -> Dict[str, Any]:
    """Inverse of ``_meta_to_dict``: a loaded mat-struct (has ``_fieldnames``)
    recurses to a nested dict; every other field is coerced by ``_scalar``. Note
    ``squeeze_me`` collapses size-1 arrays to scalars, so single-view diagnostics
    return as scalars — callers must ``np.asarray(...).reshape(-1)`` before indexing."""
    if obj is None:
        return {}
    out: Dict[str, Any] = {}
    for name in getattr(obj, "_fieldnames", []):
        if name == "_empty":
            continue
        v = getattr(obj, name)
        out[name] = _meta_from(v) if hasattr(v, "_fieldnames") else _scalar(v)
    return out


def geometry_meta(
    board: str,
    params: Any,
    *,
    model_type: Optional[str] = None,
    datum_frame: Optional[int] = None,
    datum_camera: Optional[int] = None,
) -> Dict[str, Any]:
    """Board geometry + datum, for stamping into ``board_meta["geometry"]``.

    Makes a saved model self-describing: the geometry that produced it travels with the
    record, so neither config nor the GUI panel is needed to interpret a model. Duck-typed
    on ``params`` (DotboardParams / CharucoParams / SteppedParams) to avoid importing the
    detection layer into the storage layer. Every key is a valid MATLAB identifier (no
    leading ``_`` or digit) so ``_meta_to_dict`` round-trips it as a nested struct.
    """
    g: Dict[str, Any] = {"board_type": str(board)}
    if board == "dotboard":
        g["dot_spacing_mm"] = float(params.dot_spacing_mm)
        g["k_neighbors"] = int(getattr(params, "k_neighbors", 9))
    elif board == "charuco":
        g["squares_h"] = int(params.squares_h)
        g["squares_v"] = int(params.squares_v)
        g["square_size_m"] = float(params.square_size_m)
        g["marker_ratio"] = float(params.marker_ratio)
        g["aruco_dict"] = str(params.aruco_dict)
        g["min_corners"] = int(params.min_corners)
    elif board == "stepped":
        g["dot_spacing_mm"] = float(params.dot_spacing_mm)
        g["step_height_mm"] = float(params.step_height_mm)
        g["board_thickness_mm"] = float(params.board_thickness_mm)
        if getattr(params, "level_offset_mm", None) is not None:
            g["level_offset_mm"] = float(params.level_offset_mm)
    if model_type is not None:
        g["model_type"] = str(model_type)
    if datum_frame is not None:
        g["datum_frame"] = int(datum_frame)
    if datum_camera is not None:
        g["datum_camera"] = int(datum_camera)
    return g


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


KNOWN_MODEL_TYPES = ("pinhole", "polynomial", "polynomial3d", "scale_factor")

# Stereo records hold either a pinhole pair (rig R/t) or a polynomial3d pair
# (reconstruction-only); the planar polynomial and scale factor are mono-only.
STEREO_MODEL_TYPES = ("pinhole", "polynomial3d")


def _resolve_record_path(model_dir: Path, model_type: Optional[str],
                         known: Tuple[str, ...], prefix: str, kind: str) -> Path:
    """Resolve the per-type record file in ``model_dir`` for an optional type.

    Files are named ``{prefix}_{type}.mat``. A requested type -> that file (or
    ``FileNotFoundError`` naming what is present). No type and exactly one
    candidate -> that file. Several candidates and no requested type ->
    ``ValueError`` listing them (no silent pick). Nothing usable ->
    ``FileNotFoundError``.
    """
    model_dir = Path(model_dir)
    present = {t: model_dir / f"{prefix}_{t}.mat" for t in known
               if (model_dir / f"{prefix}_{t}.mat").exists()}
    if model_type is not None:
        if model_type not in known:
            raise ValueError(
                f"unknown {kind} model_type {model_type!r} (known: {', '.join(known)})")
        if model_type in present:
            return present[model_type]
        found = f" (found: {', '.join(sorted(present))})" if present else ""
        raise FileNotFoundError(f"no {model_type} {kind} model in {model_dir}{found}")
    if len(present) == 1:
        return next(iter(present.values()))
    if not present:
        raise FileNotFoundError(f"{kind} calibration model not found in {model_dir}")
    raise ValueError(
        f"multiple {kind} models in {model_dir}: {', '.join(sorted(present))} — "
        f"specify model_type to pick one")


def resolve_mono_path(model_dir: Path, model_type: Optional[str] = None) -> Path:
    """Path of the mono record in ``model_dir`` (see ``_resolve_record_path``)."""
    return _resolve_record_path(model_dir, model_type, KNOWN_MODEL_TYPES,
                                "model", "mono")


def resolve_stereo_path(model_dir: Path, model_type: Optional[str] = None) -> Path:
    """Path of the stereo record in ``model_dir`` (see ``_resolve_record_path``)."""
    return _resolve_record_path(model_dir, model_type, STEREO_MODEL_TYPES,
                                "stereo_model", "stereo")


# Joint records currently hold a pinhole rig (the DaVis-matching solve).
JOINT_MODEL_TYPES = ("pinhole",)


def resolve_joint_path(model_dir: Path, model_type: Optional[str] = None) -> Path:
    """Path of the joint record in ``model_dir`` (see ``_resolve_record_path``)."""
    return _resolve_record_path(model_dir, model_type, JOINT_MODEL_TYPES,
                                "joint_model", "joint")


def _model_from(mat, model_type: str) -> MonoModel:
    """Reconstruct a model from a loaded ``.mat`` given its ``model_type`` tag.

    Unknown tags raise rather than silently loading as pinhole — a corrupted or
    future-format record must fail visibly. A missing tag also raises at the call
    sites (``load_mono`` / ``load_stereo``); there is no legacy pinhole default.
    """
    if model_type == "scale_factor":
        return _scale_factor_from(mat["scale_factor_model"])
    if model_type == "polynomial3d":
        return _polynomial3d_from(mat["polynomial3d_model"])
    if model_type == "polynomial":
        return _polynomial_from(mat["polynomial_model"])
    if model_type == "pinhole":
        return _camera_from(mat["camera_model"])
    raise ValueError(
        f"unknown calibration model_type tag {model_type!r} "
        f"(known: {', '.join(KNOWN_MODEL_TYPES)})"
    )


def save_mono(record: MonoRecord, model_dir: Path) -> Path:
    """Write a mono model record to ``<model_dir>/model_{model_type}.mat``."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    cm = record.camera_model
    path = model_dir / f"model_{_model_type_of(cm)}.mat"
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


def load_mono(path: Path, model_type: Optional[str] = None) -> MonoRecord:
    """Load a mono record. ``path`` is the file or its model dir; for a dir the
    per-type resolver picks the file (``model_type`` required only when several
    types exist). A requested type that mismatches the file's stored tag raises."""
    path = Path(path)
    if path.is_dir():
        path = resolve_mono_path(path, model_type)
    if not path.exists():
        raise FileNotFoundError(f"calibration model not found: {path}")
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "model_type" not in mat:
        raise ValueError(f"{path} has no model_type tag — not a PIVTOOLs per-type record")
    tag = str(_scalar(mat["model_type"]))
    if model_type is not None and tag != model_type:
        raise ValueError(f"requested {model_type!r} model but {path} stores {tag!r}")
    model: MonoModel = _model_from(mat, tag)
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
    """Write a stereo model record to ``<model_dir>/stereo_model_{model_type}.mat``."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_type = _model_type_of(record.model1)
    path = model_dir / f"stereo_model_{model_type}.mat"
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


def load_stereo(path: Path, model_type: Optional[str] = None) -> StereoRecord:
    """Load a stereo record. ``path`` is the file or its model dir; for a dir the
    per-type resolver picks the file (``model_type`` required only when several
    types exist). A requested type that mismatches the file's stored tag raises."""
    path = Path(path)
    if path.is_dir():
        path = resolve_stereo_path(path, model_type)
    if not path.exists():
        raise FileNotFoundError(f"calibration stereo model not found: {path}")
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "model_type" not in mat:
        raise ValueError(f"{path} has no model_type tag — not a PIVTOOLs per-type stereo record")
    tag = str(_scalar(mat["model_type"]))
    if model_type is not None and tag != model_type:
        raise ValueError(f"requested {model_type!r} stereo model but {path} stores {tag!r}")
    model_type = tag
    if model_type == "polynomial3d":
        model1 = _polynomial3d_from(mat["model1"])
        model2 = _polynomial3d_from(mat["model2"])
    elif model_type == "pinhole":
        model1 = _camera_from(mat["model1"])
        model2 = _camera_from(mat["model2"])
    else:
        raise ValueError(
            f"unknown stereo model_type tag {model_type!r} "
            f"(known for stereo: {', '.join(STEREO_MODEL_TYPES)})"
        )
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


def save_joint(record: JointRecord, model_dir: Path) -> Path:
    """Write a unified joint multi-camera record to ``<model_dir>/joint_model_{type}.mat``."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"joint_model_{record.model_type}.mat"
    cams = [int(c) for c in record.cameras]
    cam_models = []
    for c in cams:
        d = _camera_to_dict(record.models[c])
        d["camera"] = int(c)
        d["cam_rms"] = float(record.per_camera_rms.get(c, record.models[c].rms))
        cam_models.append(d)
    keys = sorted(record.board)
    data = {
        "contract_version": int(record.contract_version),
        "model_type": str(record.model_type),
        "board_type": str(record.board_type),
        "cameras": np.asarray(cams, dtype=np.int64).reshape(1, -1),
        "camera_models": cam_models,                       # -> struct array (one per camera)
        "board_index": np.asarray(keys, dtype=np.int64).reshape(-1, 2),
        "board_xyz": np.asarray([record.board[k] for k in keys], dtype=np.float64).reshape(-1, 3),
        "world_frame": _world_frame_to_dict(record.world_frame),
        "spacing_mm": float(record.spacing_mm),
        "board_release": str(record.board_release),
        "per_camera_rms": np.asarray([record.per_camera_rms.get(c, np.nan) for c in cams],
                                     dtype=np.float64).reshape(1, -1),
        "rms_px": float(record.rms_px),
        "board_meta": _meta_to_dict(record.board_meta),
    }
    savemat(str(path), data, oned_as="row")
    return path


def load_joint(path: Path, model_type: Optional[str] = None) -> JointRecord:
    """Load a unified joint record (file or its model dir)."""
    path = Path(path)
    if path.is_dir():
        path = resolve_joint_path(path, model_type)
    if not path.exists():
        raise FileNotFoundError(f"calibration joint model not found: {path}")
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "model_type" not in mat:
        raise ValueError(f"{path} has no model_type tag — not a PIVTOOLs joint record")
    tag = str(_scalar(mat["model_type"]))
    if model_type is not None and tag != model_type:
        raise ValueError(f"requested {model_type!r} joint model but {path} stores {tag!r}")
    cams = [int(c) for c in np.asarray(mat["cameras"], dtype=np.int64).reshape(-1)]
    cm_raw = mat["camera_models"]
    cm_list = list(np.atleast_1d(cm_raw))               # struct array -> list of mat_structs
    models, per_cam_rms = {}, {}
    for obj in cm_list:
        c = int(_scalar(obj.camera))
        models[c] = _camera_from(obj)
        per_cam_rms[c] = float(_scalar(getattr(obj, "cam_rms", models[c].rms)))
    idx = np.asarray(mat["board_index"], dtype=np.int64).reshape(-1, 2)
    xyz = np.asarray(mat["board_xyz"], dtype=np.float64).reshape(-1, 3)
    board = {(int(i[0]), int(i[1])): xyz[n] for n, i in enumerate(idx)}
    return JointRecord(
        cameras=cams,
        board_type=str(_scalar(mat["board_type"])),
        models=models,
        board=board,
        world_frame=_world_frame_from(mat["world_frame"]),
        spacing_mm=float(_scalar(mat["spacing_mm"])),
        board_release=str(_scalar(mat["board_release"])),
        per_camera_rms=per_cam_rms,
        rms_px=float(_scalar(mat["rms_px"])),
        board_meta=_meta_from(mat.get("board_meta")),
        model_type=tag,
        contract_version=int(_scalar(mat["contract_version"])),
    )


def _find_joint_record(model_dir: Path, camera: int,
                       model_type: Optional[str]) -> Optional["JointRecord"]:
    """The joint record covering ``camera`` for a per-camera mono ``model_dir``, or None.

    The mono dir is ``<root>/Cam{N}/{board}_planar/model``; the joint record (if any) lives at
    the rig-level sibling ``<root>/joint_{board}/model``. Returns None when no joint record is
    present or it does not include ``camera`` (so callers fall back to the legacy mono file).
    """
    model_dir = Path(model_dir)
    parent = model_dir.parent.name
    if not parent.endswith("_planar") or len(model_dir.parents) < 3:
        return None
    board = parent[: -len("_planar")]
    joint_dir = model_dir.parents[2] / f"joint_{board}" / "model"
    if not joint_dir.is_dir():
        return None
    try:
        jp = resolve_joint_path(joint_dir, model_type)
    except (FileNotFoundError, ValueError):
        return None
    jr = load_joint(jp)
    return jr if camera in jr.models else None


def _present_mono_types(model_dir: Path) -> set:
    """Mono record model types present on disk in ``model_dir`` (by ``model_{type}.mat``)."""
    model_dir = Path(model_dir)
    return {t for t in KNOWN_MODEL_TYPES if (model_dir / f"model_{t}.mat").exists()}


def mono_record_for_camera(model_dir: Path, camera: int,
                           model_type: Optional[str] = None) -> MonoRecord:
    """A per-camera ``MonoRecord``, preferring a joint record, else the legacy mono file.

    The apply path consumes a ``MonoRecord`` and handles every model type, so this returns one:
    from the unified joint file (wrapped as a pinhole MonoRecord) when present, else the
    legacy per-camera record unchanged. Only where the model is loaded changes — never the
    back-projection math.

    The unified joint record is pinhole-only (``JOINT_MODEL_TYPES``), so it is preferred ONLY
    for a pinhole or unspecified request — never for a polynomial/scale-factor one, otherwise a
    leftover pinhole joint file would silently shadow a per-camera polynomial calibration (e.g.
    after running ``detect-joint`` in both modes on one source). For an unspecified request,
    a pinhole joint record coexisting with a non-pinhole per-camera record is genuinely
    ambiguous, so we raise (matching ``_resolve_record_path``'s "no silent pick") rather than
    guess; a coexisting *pinhole* mono is fine — the joint solve supersedes it.
    """
    if model_type in (None, "pinhole"):
        jr = _find_joint_record(model_dir, camera, "pinhole")
        if jr is not None:
            if model_type is None:
                others = _present_mono_types(model_dir) - {"pinhole"}
                if others:
                    raise ValueError(
                        f"{model_dir}: a pinhole joint record and per-camera "
                        f"{sorted(others)} record(s) both exist — pass model_type to choose "
                        f"(the joint solve is pinhole; a polynomial joint run writes per-camera "
                        f"files)")
            return MonoRecord(
                camera=int(camera), board_type=jr.board_type, camera_model=jr.models[camera],
                world_frame=jr.world_frame,
                per_view_rms=[jr.per_camera_rms.get(camera, jr.models[camera].rms)],
                board_meta={**jr.board_meta, "spacing_mm": jr.spacing_mm, "joint": 1})
    return load_mono(model_dir, model_type)


def load_camera_model(model_dir: Path, camera: int,
                      model_type: Optional[str] = None) -> Tuple[CameraModel, WorldFrame]:
    """Per-camera (CameraModel, WorldFrame), joint-preferred. Pinhole only (raises otherwise)."""
    rec = mono_record_for_camera(model_dir, camera, model_type)
    if not isinstance(rec.camera_model, CameraModel):
        raise ValueError(
            f"load_camera_model: {model_dir} holds a {type(rec.camera_model).__name__}, "
            f"not a pinhole CameraModel"
        )
    return rec.camera_model, rec.world_frame
