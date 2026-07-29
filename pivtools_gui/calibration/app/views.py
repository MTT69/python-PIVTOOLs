"""calibration Flask blueprint — backend for the unified Calibration GUI.

One backend, two front-ends: these routes call the SAME ``Calibrator`` /
``StereoCalibrator`` / ``record`` code as the CLI. Image loading + validation reuse
the app-wide all-format reader (``calibration_loader.read_calibration_image`` /
``validate_calibration_images``), so standard images AND LaVision .im7/.set +
Phantom .cine all work, with per-camera-subfolder + indexing resolved from config —
exactly like the rest of the app.

Source model: the GUI persists the image-source config under ``config.calibration``
(``calibration_sources``, ``image_format``, ``image_type``, ``num_images``,
``use_camera_subfolders``, ``camera_subfolders``, ``zero_based_indexing``) and passes
a ``source_path_idx``; the backend reads that config via ``get_config()``. The fitted
model is the DaVis-matching pinhole (k1,k2,p1,p2) — there is no distortion choice.

Routes (prefix ``/calibration`` under the app's ``/backend``):
- POST /calibration/validate        -> validate the image source (found-N, preview, suggested pattern)
- POST /calibration/detect_datum    -> detect the datum view, cache it, return dot pixels
- GET  /calibration/datum_image     -> datum-view PNG for the click overlay
- GET  /calibration/frame_image     -> any frame as PNG (frame navigation)
- POST /calibration/detect_frame    -> detect an arbitrary frame + report frame count
- POST /calibration/snap_fiducial   -> snap a click to the nearest detected dot
- POST /calibration/generate_model  -> run the full calibration (mono/stereo), save + figures
- GET  /calibration/model           -> summary of a saved model
- POST /calibration/measure         -> distance in mm between two pixels (back-projection)
- GET  /calibration/figures|figure  -> list / serve the proof figures for a model
- POST /calibration/global/compute  -> planar N-camera datum-chain shifts
- POST /calibration/set_datum       -> shift a type's coordinates.mat (viewer datum/offset)
- POST /calibration/apply (+ /apply/status/<id>) -> apply the model to PIV output (job)
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, List, Optional

import cv2
import numpy as np
import scipy.io
from flask import Blueprint, Response, jsonify, request

from pivtools_cli import calibration_cli as c2
from pivtools_core import calibration_settings as cs
from pivtools_core.config import get_config
from pivtools_core.coordinate_utils import extract_coordinates, get_num_coordinate_runs
from pivtools_core.image_handling.calibration_loader import (
    get_calibration_frame_count,
    read_calibration_image,
    validate_calibration_images,
)
from pivtools_core.image_handling.path_utils import infer_image_type
from pivtools_core.paths import get_data_paths, vector_glob_from_format
from pivtools_gui.calibration import apply as c2apply
from pivtools_gui.calibration import global_coords as gc2
from pivtools_gui.calibration import record as rec
from pivtools_gui.calibration import runio as c2runio
from pivtools_gui.calibration import self_cal as c2sc
from pivtools_gui.calibration import world_frame as WF
from pivtools_gui.calibration.camera_model import (
    DistortionModel,
    Polynomial3DModel,
    PolynomialModel,
    ScaleFactorModel,
)
from pivtools_gui.calibration.global_grid import (
    first_view_orientation_candidates,
    resolve_global_grid_partial,
)
from pivtools_gui.calibration.inputs_store import (
    joint_det_key,
    save_inputs,
    try_load_inputs,
)
from pivtools_gui.calibration.joint_driver import run_joint_from_spec
from pivtools_gui.calibration.pipeline import Calibrator, build_scale_factor_record
from pivtools_gui.calibration.settings_seed import seed_settings
from pivtools_gui.calibration.stereo_model import StereoCalibrator
from pivtools_gui.services.job_manager import job_manager
from pivtools_gui.utils import (
    camera_number,
    get_display_contrast_stats,
    numpy_to_base64,
)
from pivtools_gui.utils.worker_pool import get_max_workers

calibration_bp = Blueprint("calibration", __name__)
logger = logging.getLogger(__name__)


def _detect_parallel(board, params, imgs, spacing_mm=None, on_done=None):
    """Detect calibration targets in many views concurrently.

    Detection is OpenCV-bound (the GIL is released inside cv2), so threads scale and
    the pool honours the ``processing.post_processing_workers`` knob. Each task builds
    its own detector — cv2 detector objects are not documented thread-safe. Results
    return in image order; ``on_done()`` fires once per completed view (any order).
    """
    dets: List[Any] = [None] * len(imgs)
    if not imgs:
        return dets
    max_workers = get_max_workers(len(imgs))

    def _one(k):
        d = c2._build_detector(board, params).detect(imgs[k])
        if spacing_mm is not None:
            d.spacing_mm = spacing_mm
        return k, d

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_one, k) for k in range(len(imgs))]
        for future in as_completed(futures):
            k, d = future.result()
            dets[k] = d
            if on_done is not None:
                on_done()
    return dets

# Datum-view detections cached for the snap workflow, keyed by camera number.
_datum_cache = {}
_datum_lock = threading.Lock()

# Joint multi-camera: detections cached per (source, board, n_views, format, params) so the
# live resolve-grid loop and the generate job do not re-detect every view on every call. The
# in-memory layer is the fast path; the persistent layer is the sidecar inputs.mat (see
# _joint_detect), keyed by _joint_det_key so a param change forces a re-detect.
_joint_detect_cache: dict = {}
_joint_detect_lock = threading.Lock()


def _joint_inputs_dir(source_idx: int, board: str) -> Optional[Path]:
    """Model dir holding this rig's sidecar ``inputs.mat``, or None if the source is unknown.

    The sidecar lives beside the solved joint model (``joint_{board}/model/``); the detections
    persist there so they survive a server restart and a model delete — detecting every view of
    every camera is the slow part of the live overlay.
    """
    try:
        return rec.joint_model_dir_for_source(_source_path(source_idx), board)
    except Exception:
        return None


# The GUI offers exactly one model — the DaVis PinholeOpenCV (k1,k2,p1,p2).
_MODEL = DistortionModel.STANDARD


def _cfg() -> dict:
    return c2._cfg2(get_config())


def _source_idx(get: Callable[[str], Any]) -> int:
    v = get("source_path_idx")
    return int(v) if v not in (None, "") else int(_cfg().get("source_idx", 0))


def _source_path(source_idx: int) -> Path:
    """Resolved calibration source directory for this index (where the model is saved)."""
    return get_config().get_calibration_source(int(source_idx))


def _settings_idx(source_idx: int) -> dict:
    """Settings sidecar for a source index; the defaults template when absent.

    An unconfigured source list also yields the template — helpers that only
    need a knob default (datum frame, camera number) must not fail before the
    route's own source handling does. A sidecar that EXISTS but fails
    validation still raises: silently reverting a corrupt file to the defaults
    template would be exactly the stale-state class this store eliminates.
    """
    try:
        source = _source_path(source_idx)
    except (ValueError, IndexError):
        return cs.default_settings()
    settings = cs.try_load_settings(source)
    return settings if settings is not None else cs.default_settings()


def _settings_for(get: Callable[[str], Any]) -> dict:
    return _settings_idx(_source_idx(get))


def _rig(get: Callable[[str], Any]) -> dict:
    return _settings_for(get).get("rig") or {}


def _methods(get: Callable[[str], Any]) -> dict:
    return _settings_for(get).get("methods") or {}


def _model_type_arg(
    get: Callable[[str], Any], board: Optional[str] = None, stereo: bool = False
) -> Optional[str]:
    """Requested record type for a model load, now that types have per-type files.

    Explicit ``model_type`` request param first, else the board's configured
    ``model_type`` (the GUI persists its selector there). For stereo loads a
    mono-only configured type (``polynomial`` / ``scale_factor``) is dropped —
    it names the mono fit, not the stereo record. ``None`` lets the resolver
    pick the single record present (ambiguity raises, listing the types).
    """
    mt = get("model_type")
    if not mt and board:
        mt = _methods(get).get(board, {}).get("model_type")
    if not mt:
        return None
    mt = str(mt)
    if stereo and mt not in rec.STEREO_MODEL_TYPES:
        return None
    return mt


def _generate_dt(get: Callable[[str], Any], source: Path) -> float:
    """dt to stamp into a generated record: request > settings ``rig.dt`` > error.

    Velocity scales linearly with dt, so it has no safe default; generation is
    the moment the user is present to supply it, and every record it produces
    carries it (apply then resolves request > model-stamped > error, with no
    config source).
    """
    v = get("dt")
    if v not in (None, ""):
        return float(v)
    settings = cs.try_load_settings(source)
    dt = ((settings or {}).get("rig") or {}).get("dt")
    if dt is None:
        raise ValueError(
            "dt is required to generate a calibration model — velocity has no "
            "safe default. Set dt in the Calibration tab (saved to the "
            "source's calibration/settings.yaml) or pass 'dt' in the request."
        )
    return float(dt)


def _resolve_board(get: Callable[[str], Any], overrides: Optional[dict] = None):
    """Resolve (cfg, board, params, detector). Board params may be overridden per-call.

    ``cfg`` (the YAML calibration block) supplies only the ``active`` pointer;
    board geometry comes from the source's settings sidecar + request overrides.
    """
    cfg = _cfg()
    board = get("board") or cfg.get("active", "charuco")
    params = c2._board_params(_methods(get), board, overrides)
    detector = c2._build_detector(board, params)
    return cfg, board, params, detector


@calibration_bp.route("/calibration/settings", methods=["GET"])
def get_calibration_settings():
    """Per-source settings sidecar for the requested source.

    ``exists: false`` ships a seed built from the model records on disk (the
    persisted YAML is presumed stale and never consulted); nothing is written
    until the client saves. A present-but-corrupt sidecar is an error, never
    silently reseeded.
    """
    try:
        source = _source_path(_source_idx(request.args.get))
    except (ValueError, IndexError) as e:
        return jsonify({"error": str(e)}), 400
    try:
        existing = cs.try_load_settings(source)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if existing is not None:
        return jsonify({"exists": True, "settings": existing})
    return jsonify({"exists": False, "settings": seed_settings(source)})


@calibration_bp.route("/calibration/settings", methods=["POST"])
def save_calibration_settings():
    """Deep-merge a partial settings payload into the source's sidecar.

    Unknown TOP-LEVEL blocks are rejected — a typo'd block name would be
    written and then ignored forever, the silent-staleness class this store
    exists to kill. Keys nested inside known blocks stay open: the deliberately
    untemplated knobs (``fit.use_release_object``, ``methods.<board>.fix_k2``)
    are legal precisely because they are absent from the template.
    """
    data = request.get_json(force=True) or {}
    partial = data.get("settings")
    if not isinstance(partial, dict):
        return jsonify({"error": "settings must be a mapping"}), 400
    unknown = set(partial) - set(cs.default_settings())
    if unknown:
        return (
            jsonify(
                {
                    "error": "unknown settings block(s): "
                    + ", ".join(sorted(unknown))
                }
            ),
            400,
        )
    try:
        source = _source_path(_source_idx(data.get))
    except (ValueError, IndexError) as e:
        return jsonify({"error": str(e)}), 400
    try:
        if not cs.settings_path(source).exists():
            # First save for this source: persist the record-recovered seed as
            # the base, THEN merge the client partial over it. Without this the
            # seed shown by GET (dt, geometry, camera_pair from the newest
            # records) would be discarded whenever the first POST is partial —
            # the whole migration path for existing datasets rides on it.
            cs.save_settings(source, seed_settings(source))
        merged = cs.save_settings(source, partial)
    except (ValueError, PermissionError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"status": "success", "settings": merged})


def _load_one(
    camera: int,
    frame: int,
    source_idx: int,
    image_format: Optional[str],
    image_type: Optional[str],
    normalize_uint8: bool = True,
) -> np.ndarray:
    """Load one calibration frame (any format) via the app-wide reader.

    ``normalize_uint8`` defaults True because detection (OpenCV) needs an 8-bit
    array. Display endpoints that drive the auto-contrast slider must pass
    ``normalize_uint8=False`` so the native bit depth reaches
    ``get_display_contrast_stats`` — collapsing a 12/16-bit frame to uint8 first
    makes that function short-circuit to a dead [0, 100] window.
    """
    return read_calibration_image(
        int(frame),
        int(camera),
        get_config(),
        int(source_idx),
        image_format=image_format,
        image_type=image_type,
        normalize_uint8=normalize_uint8,
    )


def _load_views(
    camera: int,
    frame_total: int,
    source_idx: int,
    image_format: Optional[str],
    image_type: Optional[str],
) -> List[np.ndarray]:
    return [
        _load_one(camera, k + 1, source_idx, image_format, image_type)
        for k in range(int(frame_total))
    ]


def _frame_total(get: Callable[[str], Any], camera: int, source_idx: int) -> int:
    """Resolve frame total: request value, else settings ``image.n_views``, else auto-detect."""
    v = get("frame_total")
    if v not in (None, ""):
        return int(v)
    n = int((_settings_idx(source_idx).get("image") or {}).get("n_views") or 0)
    if n > 0:
        return n
    try:
        return get_calibration_frame_count(int(camera), get_config(), int(source_idx))
    except Exception as exc:
        logger.warning(
            "frame-count auto-detect failed for cam %s source %s: %s",
            camera,
            source_idx,
            exc,
        )
        return 0


def _datum_frame(get: Callable[[str], Any]) -> int:
    v = get("datum_frame")
    return int(v) if v not in (None, "") else int(_rig(get).get("datum_frame") or 1)


def _png_response(img: np.ndarray):
    """Encode an image as an 8-bit PNG HTTP response (contrast-stretched if needed)."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        mx = float(img.max()) or 1.0
        img = (img.astype(np.float64) / mx * 255.0).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    return Response(buf.tobytes(), mimetype="image/png")


def _clicks_from(payload) -> Optional[dict]:
    if not payload:
        return None
    return {
        "origin": np.asarray(payload["origin"], dtype=float),
        "x_axis": np.asarray(payload["x_axis"], dtype=float),
        "y_axis": np.asarray(payload["y_axis"], dtype=float),
    }


def _origin_mm_from(payload) -> Optional[tuple]:
    """World (X, Y) mm of the origin dot, carried inside the clicks payload."""
    if not payload:
        return None
    om = payload.get("origin_mm")
    if om is None:
        return None
    return (float(om[0]), float(om[1]))


def _figure_rank(name: str) -> int:
    """Display priority for a proof figure: the dewarped board first (clearest at a glance),
    then detection, then the fit/reprojection residual, then the supporting geometry."""
    n = name.lower()
    if "dewarp" in n:
        return 0
    if n.startswith("detection"):
        return 1
    if n.startswith("reprojection") or n.startswith("polynomial_fit"):
        return 2
    if n.startswith("world_frame"):
        return 3
    if n.startswith("boards_3d"):
        return 4
    return 5


def _list_figures(fig_dir) -> List[str]:
    if not fig_dir or not Path(fig_dir).is_dir():
        return []
    # Skip macOS AppleDouble companions (._*) that appear when writing to non-HFS drives.
    names = [p.name for p in Path(fig_dir).glob("*.png") if not p.name.startswith("._")]
    # Rank by figure kind (dewarp first), alphabetical within a kind. Filenames are unchanged —
    # the /calibration/figure?name= serve endpoint still resolves each by basename.
    return sorted(names, key=lambda n: (_figure_rank(n), n))


def _job_status_response(job_id: str):
    """Shared poll response for every calibration job route: timed status or 404."""
    data = job_manager.get_job_with_timing(job_id)
    if data is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(data)


def _world_frame_payload(wf) -> dict:
    """Saved world frame for the GUI to restore the picked origin/+X/+Y + origin mm."""

    def lst(a):
        return None if a is None else [float(x) for x in np.asarray(a).reshape(-1)]

    return {
        "mode": str(wf.mode),
        "origin": lst(wf.origin_px),
        "x_axis": lst(wf.x_axis_px),
        "y_axis": lst(wf.y_axis_px),
        "origin_mm": lst(wf.origin_mm) or [0.0, 0.0],
    }


def _geometry_payload(board_meta) -> Optional[dict]:
    """The stamped board geometry (``board_meta['geometry']``) coerced to JSON-native scalars —
    Flask jsonify rejects numpy types, and the .mat round-trip can return size-1 numpy scalars.
    None for records saved before geometry stamping, so the GUI falls back to its panel defaults.
    """
    g = board_meta.get("geometry") if isinstance(board_meta, dict) else None
    if not isinstance(g, dict):
        return None
    return {
        k: (v if isinstance(v, str) else (v.item() if hasattr(v, "item") else v))
        for k, v in g.items()
    }


def _inputs_world_frame(model_dir) -> Optional[dict]:
    """World-frame picks recovered from the model dir's ``inputs.mat`` (the clicks survive a
    model delete), shaped like ``_world_frame_payload`` so the GUI restores origin/+X/+Y and a
    deleted model can be re-solved without re-clicking. None when no usable clicks are stored.
    """
    try:
        side = try_load_inputs(model_dir)
    except Exception:
        return None
    c = side.coords if side else None
    if not isinstance(c, dict) or not c.get("origin"):
        return None

    def lst(v):
        return [float(x) for x in v] if v else None

    return {
        "mode": "clicks",
        "origin": lst(c.get("origin")),
        "x_axis": lst(c.get("x_axis")),
        "y_axis": lst(c.get("y_axis")),
        "origin_mm": lst(c.get("origin_mm")) or [0.0, 0.0],
    }


def _intrinsics(cm) -> dict:
    """Pinhole intrinsics of a CameraModel, shaped for the GUI results card."""
    return {
        "fx": float(cm.K[0, 0]),
        "fy": float(cm.K[1, 1]),
        "cx": float(cm.K[0, 2]),
        "cy": float(cm.K[1, 2]),
        "camera_matrix": cm.K.tolist(),
        "dist_coeffs": cm.dist.tolist(),
        "rms": float(cm.rms),
        "image_width": int(cm.image_size[0]),
        "image_height": int(cm.image_size[1]),
    }


def _polynomial_summary(pm: PolynomialModel) -> dict:
    """Polynomial coefficients of a PolynomialModel, shaped for the GUI results card."""
    return {
        "model_type": "polynomial",
        "coeffs_x": [float(c) for c in pm.coeffs_x],
        "coeffs_y": [float(c) for c in pm.coeffs_y],
        "x0": float(pm.x0),
        "sx": float(pm.sx),
        "y0": float(pm.y0),
        "sy": float(pm.sy),
        "rms_x_mm": float(pm.rms_x_mm),
        "rms_y_mm": float(pm.rms_y_mm),
        "image_width": int(pm.image_size[0]),
        "image_height": int(pm.image_size[1]),
    }


def _polynomial3d_summary(pm: Polynomial3DModel) -> dict:
    """Single-view 3D polynomial summary (world->image cubic), shaped for the card."""
    return {
        "model_type": "polynomial3d",
        "rms": float(pm.rms_px),
        "plane_rms": [float(v) for v in pm.plane_rms_px],
        "coeffs_u": [float(c) for c in pm.coeffs_u],
        "coeffs_v": [float(c) for c in pm.coeffs_v],
        "x0": float(pm.x0),
        "sx": float(pm.sx),
        "y0": float(pm.y0),
        "sy": float(pm.sy),
        "z0": float(pm.z0),
        "sz": float(pm.sz),
        "image_width": int(pm.image_size[0]),
        "image_height": int(pm.image_size[1]),
    }


def _scale_factor_summary(
    sf: ScaleFactorModel, dt: float, frame_idx=None, wf=None
) -> dict:
    """Scale-factor params shaped for the GUI results card + restore-on-load.

    ``frame_idx`` (the 1-based frame the origin was picked on), when known, lets the GUI
    restore the origin/axis overlay on that same frame rather than always frame 1.

    ``wf`` (the record's WorldFrame) carries the PICKED origin pixel + its world
    ``origin_mm`` — the model's own origin_px is the world-zero pixel, which differs
    from the picked point once origin_mm is non-zero (the offset is baked in by
    shifting it). The GUI restores what the user picked/typed, so prefer wf.
    """
    px_per_mm = (1.0 / sf.mm_per_pixel) if sf.mm_per_pixel else float("nan")
    picked = (
        wf.origin_px if (wf is not None and wf.origin_px is not None) else sf.origin_px
    )
    origin_mm = (
        [float(wf.origin_mm[0]), float(wf.origin_mm[1])]
        if (wf is not None and wf.origin_mm is not None)
        else [0.0, 0.0]
    )
    out = {
        "model_type": "scale_factor",
        "origin_px": [float(picked[0]), float(picked[1])],
        "origin_mm": origin_mm,
        "mm_per_pixel": float(sf.mm_per_pixel),
        "px_per_mm": float(px_per_mm),
        "dt": float(dt),
        "col_sign": int(sf.col_sign),
        "row_sign": int(sf.row_sign),
        "swap_axes": int(sf.swap_axes),
        "x_dir": "right" if sf.col_sign >= 0 else "left",
        "y_dir": "up" if sf.row_sign < 0 else "down",
        "image_width": int(sf.image_size[0]),
        "image_height": int(sf.image_size[1]),
    }
    if frame_idx is not None:
        out["frame_idx"] = int(frame_idx)
    return out


def _figures_dir(get: Callable[[str], Any]) -> Path:
    """Resolve the figures dir for a model from request locators (mono or stereo)."""
    cfg = _cfg()
    board = get("board") or cfg.get("active", "charuco")
    source = _source_path(_source_idx(get))
    if str(get("joint")) in ("1", "true", "True"):
        return rec.joint_model_dir_for_source(source, board).parent / "figures"
    if str(get("stereo")) in ("1", "true", "True"):
        pair = get("camera_pair") or _rig(get).get("camera_pair") or [1, 2]
        if isinstance(pair, str):
            pair = [int(x) for x in pair.split(",")]
        return (
            rec.stereo_model_dir_for_source(source, int(pair[0]), int(pair[1])).parent
            / "figures"
        )
    camera = int(get("camera") or _rig(get).get("camera") or 1)
    return rec.mono_model_dir_for_source(source, camera, board).parent / "figures"


def _meta_float(meta: dict, key: str) -> Optional[float]:
    try:
        return float(meta[key]) if meta and key in meta else None
    except (TypeError, ValueError):
        return None


def _meta_int(meta: dict, key: str) -> Optional[int]:
    v = _meta_float(meta, key)
    return int(v) if v is not None else None


# ---------------------------------------------------------------------------
# Image source validation
# ---------------------------------------------------------------------------


@calibration_bp.route("/calibration/validate", methods=["POST"])
def validate():
    """Validate the calibration image source (all formats): found-count, preview, suggestion."""
    data = request.get_json() or {}
    camera = int(data.get("camera") or _rig(data.get).get("camera") or 1)
    source_idx = _source_idx(data.get)
    try:
        result = validate_calibration_images(
            camera,
            get_config(),
            source_idx,
            image_format=data.get("image_format"),
            num_images=data.get("frame_total"),
            image_type=data.get("image_type"),
        )
    except FileNotFoundError as exc:
        # The only FileNotFoundError reaching here is the missing settings
        # sidecar (validate_images_generic returns read failures in its result
        # dict, and get_calibration_source raises ValueError/IndexError). That
        # is the expected first-visit state of every pre-sidecar source, not a
        # fault — the message is already actionable, so no traceback.
        logger.warning("calibration validate: %s", exc)
        return jsonify({"valid": False, "error": str(exc)}), 200
    except Exception as exc:
        logger.exception("calibration validate failed")
        return jsonify({"valid": False, "error": str(exc)}), 200
    return jsonify(result)


# ---------------------------------------------------------------------------
# Detection + frame navigation
# ---------------------------------------------------------------------------


@calibration_bp.route("/calibration/detect_datum", methods=["POST"])
def detect_datum():
    """Detect the datum view for a camera; cache it and return dot pixels for clicking."""
    data = request.get_json() or {}
    cfg, board, params, detector = _resolve_board(data.get, data.get("board_params"))
    camera = int(data.get("camera") or _rig(data.get).get("camera") or 1)
    source_idx = _source_idx(data.get)
    frame = _datum_frame(data.get)
    try:
        img = _load_one(
            camera, frame, source_idx, data.get("image_format"), data.get("image_type")
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    det = detector.detect(img)
    if not det.success:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "board not detected",
                    "diagnostics": det.diagnostics,
                }
            ),
            200,
        )

    with _datum_lock:
        _datum_cache[camera] = det
    h, w = np.asarray(img).shape[:2]
    return jsonify(
        {
            "success": True,
            "camera": camera,
            "board": board,
            "frame": int(frame),
            "n_points": det.n,
            "width": int(w),
            "height": int(h),
            "image_points": det.image_points.tolist(),
            "grid_indices": det.grid_indices.tolist(),
        }
    )


@calibration_bp.route("/calibration/datum_image", methods=["GET"])
def datum_image():
    """Return the datum-view image (PNG) for a camera, for the click overlay."""
    camera = int(
        request.args.get("camera") or _rig(request.args.get).get("camera") or 1
    )
    source_idx = _source_idx(request.args.get)
    frame = _datum_frame(request.args.get)
    try:
        img = _load_one(
            camera,
            frame,
            source_idx,
            request.args.get("image_format"),
            request.args.get("image_type"),
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"error": str(exc)}), 404
    return _png_response(img)


@calibration_bp.route("/calibration/frame_image", methods=["GET"])
def frame_image():
    """Serve any calibration frame (1-based image index) as a PNG."""
    camera = int(
        request.args.get("camera") or _rig(request.args.get).get("camera") or 1
    )
    source_idx = _source_idx(request.args.get)
    frame = int(request.args.get("frame", 1))
    try:
        img = _load_one(
            camera,
            frame,
            source_idx,
            request.args.get("image_format"),
            request.args.get("image_type"),
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"error": str(exc)}), 404
    return _png_response(img)


@calibration_bp.route("/calibration/frame", methods=["GET"])
def frame_json():
    """Serve a calibration frame as JSON: ``{image, stats, width, height, frame_count}``.

    Matches the v1 ``/calibration/get_frame`` shape so the original
    ``CalibrationImageViewer`` (contrast sliders + colormap + prefetch) works
    unchanged on the calibration backend. All formats via the app-wide reader;
    ``stats.vmin_pct``/``vmax_pct`` drive the auto-contrast window.
    """
    _cfg()
    camera = camera_number(request.args.get("camera", default=1, type=int))
    idx = request.args.get("idx", default=1, type=int)
    source_idx = _source_idx(request.args.get)
    output_format = (
        request.args.get("format", default="jpeg", type=str) or "jpeg"
    ).lower()
    quality = request.args.get("quality", default=85, type=int)
    try:
        frame_count = get_calibration_frame_count(camera, get_config(), source_idx)
    except Exception:
        frame_count = 0
    if frame_count > 0 and idx > frame_count:
        return (
            jsonify(
                {
                    "error": f"Frame index {idx} exceeds available frames ({frame_count})",
                    "frame_count": frame_count,
                    "requested_idx": idx,
                }
            ),
            400,
        )
    try:
        # Display path: keep native bit depth so get_display_contrast_stats and
        # numpy_to_base64 see the real dynamic range (sqrt + percentile window),
        # matching the PIV viewer. Detection routes still load uint8 for OpenCV.
        img = _load_one(
            camera,
            idx,
            source_idx,
            request.args.get("image_format"),
            request.args.get("image_type"),
            normalize_uint8=False,
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"error": str(exc), "frame_count": frame_count}), 404

    arr = np.asarray(img)
    arr_f = arr.astype(np.float32)
    stats = {
        "min": float(arr_f.min()),
        "max": float(arr_f.max()),
        "mean": float(arr_f.mean()),
        "dtype": str(arr.dtype),
    }
    contrast = get_display_contrast_stats(arr)
    stats["vmin_pct"] = contrast["vmin_pct"]
    stats["vmax_pct"] = contrast["vmax_pct"]
    return jsonify(
        {
            "image": numpy_to_base64(arr, format=output_format, jpeg_quality=quality),
            "mime_type": f"image/{output_format}",
            "width": int(arr.shape[1]),
            "height": int(arr.shape[0]),
            "stats": stats,
            "frame_count": frame_count,
            "current_idx": idx,
        }
    )


@calibration_bp.route("/calibration/detect_frame", methods=["POST"])
def detect_frame():
    """Detect one arbitrary frame (browsing/overlay) and report the frame count.

    Does NOT replace the datum cache used by ``snap_fiducial`` — world-frame fiducials
    are picked on the datum frame; this route is for inspection.
    """
    data = request.get_json() or {}
    cfg, board, params, detector = _resolve_board(data.get, data.get("board_params"))
    camera = int(data.get("camera") or _rig(data.get).get("camera") or 1)
    source_idx = _source_idx(data.get)
    frame = int(data.get("frame", 1))
    try:
        frame_count = get_calibration_frame_count(camera, get_config(), source_idx)
    except Exception:
        frame_count = 0
    try:
        img = _load_one(
            camera, frame, source_idx, data.get("image_format"), data.get("image_type")
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return (
            jsonify({"success": False, "error": str(exc), "frame_count": frame_count}),
            200,
        )
    det = detector.detect(img)
    h, w = np.asarray(img).shape[:2]
    if not det.success:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "board not detected",
                    "frame": int(frame),
                    "frame_count": frame_count,
                    "width": int(w),
                    "height": int(h),
                    "diagnostics": det.diagnostics,
                }
            ),
            200,
        )
    return jsonify(
        {
            "success": True,
            "camera": camera,
            "board": board,
            "frame": int(frame),
            "frame_count": frame_count,
            "n_points": det.n,
            "width": int(w),
            "height": int(h),
            "image_points": det.image_points.tolist(),
            "grid_indices": det.grid_indices.tolist(),
        }
    )


@calibration_bp.route("/calibration/detect_views", methods=["POST"])
def detect_views():
    """Detect every calibration view for one camera in a single round-trip.

    Backs the "Detect Dots" button for pinhole models, where the overlay should show
    the full set of views the bundle fit will use. Frames that fail to load or detect
    are skipped (a board is legitimately absent from some views) — not an error.
    Like ``detect_frame`` this is overlay-only and does NOT touch the datum cache used
    by ``snap_fiducial``.
    """
    data = request.get_json() or {}
    cfg, board, params, detector = _resolve_board(data.get, data.get("board_params"))
    camera = int(data.get("camera") or _rig(data.get).get("camera") or 1)
    source_idx = _source_idx(data.get)
    frame_total = int(data.get("frame_total", 1))
    image_format = data.get("image_format")
    image_type = data.get("image_type")
    try:
        frame_count = get_calibration_frame_count(camera, get_config(), source_idx)
    except Exception:
        frame_count = 0
    frames: dict[str, dict] = {}
    width = height = 0
    for frame in range(1, frame_total + 1):
        try:
            img = _load_one(camera, frame, source_idx, image_format, image_type)
        except (FileNotFoundError, ValueError, IndexError):
            continue
        det = detector.detect(img)
        h, w = np.asarray(img).shape[:2]
        width, height = int(w), int(h)
        if det.success:
            frames[str(frame)] = {
                "image_points": det.image_points.tolist(),
                "grid_indices": det.grid_indices.tolist(),
                "n_points": det.n,
            }
    return jsonify(
        {
            "success": True,
            "camera": camera,
            "board": board,
            "frames": frames,
            "n_detected": len(frames),
            "frame_count": frame_count,
            "width": width,
            "height": height,
        }
    )


@calibration_bp.route("/calibration/snap_fiducial", methods=["POST"])
def snap_fiducial():
    """Snap a click to the nearest detected dot; return its pixel + grid index."""
    data = request.get_json() or {}
    camera = int(data.get("camera", 1))
    cx, cy = float(data.get("click_x", 0)), float(data.get("click_y", 0))
    with _datum_lock:
        det = _datum_cache.get(camera)
    if det is None:
        return jsonify({"error": "no cached detection; call detect_datum first"}), 400
    idx = WF._snap(det.image_points, (cx, cy))
    px = det.image_points[idx]
    gi = det.grid_indices[idx]
    return jsonify(
        {
            "snapped_x": float(px[0]),
            "snapped_y": float(px[1]),
            "grid_col": int(gi[0]),
            "grid_row": int(gi[1]),
        }
    )


# ---------------------------------------------------------------------------
# Model generation
# ---------------------------------------------------------------------------


@calibration_bp.route("/calibration/generate_model", methods=["POST"])
def generate_model():
    """Run the full calibration (mono or stereo), save it + proof figures into the source."""
    data = request.get_json() or {}
    cfg, board, params, detector = _resolve_board(data.get, data.get("board_params"))
    source_idx = _source_idx(data.get)
    image_format, image_type = data.get("image_format"), data.get("image_type")
    stereo = bool(data.get("stereo", False))
    model_type = str(
        data.get("model_type", "pinhole")
    )  # polynomial is planar (mono) only
    make_figs = not bool(data.get("no_figures", False))
    spacing = c2._spacing_mm(board, params)
    datum_frame = _datum_frame(data.get)
    datum_index = datum_frame - 1  # position within the loaded views (frames 1..N)

    try:
        source = _source_path(source_idx)
        # Resolve dt up front: a missing dt must fail before the expensive
        # detection pass, while the user is still at the form.
        gen_dt = _generate_dt(data.get, source)
        if stereo:
            pair = data.get("camera_pair") or _rig(data.get).get("camera_pair") or [1, 2]
            cam1, cam2 = int(pair[0]), int(pair[1])
            frame_total = _frame_total(data.get, cam1, source_idx)
            if not (0 <= datum_index < frame_total):
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"datum frame {datum_frame} is outside 1..{frame_total}",
                        }
                    ),
                    200,
                )
            imgs1 = _load_views(cam1, frame_total, source_idx, image_format, image_type)
            imgs2 = _load_views(cam2, frame_total, source_idx, image_format, image_type)
            model_dir = rec.stereo_model_dir_for_source(source, cam1, cam2)
            fig_dir = (model_dir.parent / "figures") if make_figs else None
            # Detection sidecar/cache (parity with the joint path): reuse stored detections
            # when the request params still match (det_key), else detect fresh. Persisted
            # below so a re-run skips detection and the detections survive a model delete.
            det_key = joint_det_key(
                board,
                frame_total,
                image_format,
                infer_image_type(image_format),
                [cam1, cam2],
                params,
            )
            force_redetect = bool(data.get("force_redetect", False))
            side = None if force_redetect else try_load_inputs(model_dir)
            cache_hit = (
                side is not None
                and side.det_key == det_key
                and bool(side.detections)
                and cam1 in side.detections
                and cam2 in side.detections
            )
            if cache_hit:
                det1, det2 = side.detections[cam1], side.detections[cam2]
            else:
                det1 = _detect_parallel(board, params, imgs1)
                det2 = _detect_parallel(board, params, imgs2)
            # World-frame picks: the live request, else the cam1 clicks stored in the sidecar
            # (so a deleted-model re-solve rebuilds the shared frame without re-clicking).
            clicks_payload = data.get("clicks") or (side.coords if side else None)
            sc = StereoCalibrator(
                detector=detector,
                board_type=board,
                distortion_model=_MODEL,
                fix_k2=bool(data.get("fix_k2", False)),
            )
            record = sc.run_stereo(
                imgs1,
                imgs2,
                cam1=cam1,
                cam2=cam2,
                clicks=_clicks_from(clicks_payload),
                clicks2=_clicks_from(data.get("clicks2")),
                origin_mm=_origin_mm_from(clicks_payload),
                datum_index=datum_index,
                spacing_mm=spacing,
                figure_dir=fig_dir,
                det1=det1,
                det2=det2,
            )
            record.board_meta["dt"] = gen_dt
            if image_format:
                record.board_meta["image_format"] = str(image_format)
            path = rec.save_stereo(record, model_dir)
            save_inputs(
                model_dir,
                path_type="stereo",
                board_type=board,
                detections={cam1: list(det1), cam2: list(det2)},
                image_size_by_cam={
                    cam1: (
                        int(record.model1.image_size[0]),
                        int(record.model1.image_size[1]),
                    ),
                    cam2: (
                        int(record.model2.image_size[0]),
                        int(record.model2.image_size[1]),
                    ),
                },
                det_key=det_key,
                board_params=rec.geometry_meta(board, params),
                coords=clicks_payload,
            )
            ang = float(
                np.degrees(
                    np.arccos(np.clip((np.trace(record.R_stereo) - 1) / 2, -1, 1))
                )
            )
            return jsonify(
                {
                    "success": True,
                    "stereo": True,
                    "model_path": str(path),
                    "rms_cam1": record.model1.rms,
                    "rms_cam2": record.model2.rms,
                    "per_view_rms1": list(record.per_view_rms1),
                    "per_view_rms2": list(record.per_view_rms2),
                    "intrinsics1": _intrinsics(record.model1),
                    "intrinsics2": _intrinsics(record.model2),
                    "num_pairs_used": record.board_meta.get(
                        "n_stereo_views", len(record.per_view_rms1)
                    ),
                    "stereo_angle_deg": ang,
                    "baseline_mm": float(np.linalg.norm(record.T_stereo)),
                    "stereo_rms_px": _finite_or_none(
                        record.board_meta.get("stereo_rms_px")
                    ),
                    "method": record.board_meta.get("stereo_method", "stereoCalibrate"),
                    "detections_cached": bool(cache_hit),
                    "figures": _list_figures(fig_dir),
                }
            )
        else:
            camera = int(data.get("camera") or _rig(data.get).get("camera") or 1)
            model_dir = rec.mono_model_dir_for_source(source, camera, board)
            # Detection sidecar (parity with stereo/joint): reuse stored detections + clicks when
            # the request params still match (det_key), so a model can be regenerated without
            # re-detecting or re-clicking — and, with figures off, without the images on disk.
            force_redetect = bool(data.get("force_redetect", False))
            side = None if force_redetect else try_load_inputs(model_dir)
            cached = (side.detections or {}).get(camera) if side else None
            # View count from the request/config, falling back to the sidecar's cached count so a
            # re-solve still resolves when the images (and thus the auto-count) are gone.
            frame_total = _frame_total(data.get, camera, source_idx) or (
                len(cached) if cached else 0
            )
            if not (0 <= datum_index < frame_total):
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"datum frame {datum_frame} is outside 1..{frame_total}",
                        }
                    ),
                    200,
                )
            det_key = joint_det_key(
                board,
                frame_total,
                image_format,
                infer_image_type(image_format),
                [camera],
                params,
            )
            cached_size = None
            cache_hit = side is not None and side.det_key == det_key and bool(cached)
            if cache_hit:
                cached_size = side.image_size_by_cam.get(camera)
            # World-frame picks: the live request, else the stored coords (re-solve w/o re-click).
            clicks_payload = data.get("clicks") or (side.coords if side else None)
            # Load images only to detect or to draw figures; a cached re-solve with figures off
            # touches no image files. (A cached hit lacking a stored size also needs an image.)
            need_images = make_figs or not cache_hit or cached_size is None
            imgs = (
                _load_views(camera, frame_total, source_idx, image_format, image_type)
                if need_images
                else []
            )
            # Figures need the images; if they could not be loaded (a re-solve from the sidecar
            # after the images were removed), skip them rather than fail the solve.
            fig_dir = (model_dir.parent / "figures") if (make_figs and imgs) else None
            calr = Calibrator(
                detector=detector,
                board_type=board,
                model_type=model_type,
                distortion_model=_MODEL,
                fix_k2=bool(data.get("fix_k2", False)),
            )
            dets = (
                list(cached) if cache_hit else _detect_parallel(board, params, imgs)
            )
            record = calr.run_mono(
                imgs,
                camera=camera,
                clicks=_clicks_from(clicks_payload),
                origin_mm=_origin_mm_from(clicks_payload),
                datum_index=datum_index,
                spacing_mm=spacing,
                image_size=cached_size,
                figure_dir=fig_dir,
                detections=dets,
            )
            record.board_meta["dt"] = gen_dt
            if image_format:
                record.board_meta["image_format"] = str(image_format)
            path = rec.save_mono(record, model_dir)
            isz = record.camera_model.image_size
            save_inputs(
                model_dir,
                path_type="mono",
                board_type=board,
                detections={camera: list(dets)},
                image_size_by_cam={camera: (int(isz[0]), int(isz[1]))},
                det_key=det_key,
                board_params=rec.geometry_meta(board, params),
                coords=clicks_payload,
            )
            cm = record.camera_model
            resp = {
                "success": True,
                "stereo": False,
                "model_path": str(path),
                "camera": camera,
                "num_images_used": len(record.per_view_rms),
                "per_view_rms": list(record.per_view_rms),
                "detections_cached": bool(cache_hit),
                "figures": _list_figures(fig_dir),
            }
            if isinstance(cm, PolynomialModel):
                resp.update(_polynomial_summary(cm))
            elif isinstance(cm, Polynomial3DModel):
                resp.update(_polynomial3d_summary(cm))
            else:
                resp.update(
                    {
                        "model_type": "pinhole",
                        "rms": cm.rms,
                        "fx": float(cm.K[0, 0]),
                        "fy": float(cm.K[1, 1]),
                        "cx": float(cm.K[0, 2]),
                        "cy": float(cm.K[1, 2]),
                        "camera_matrix": cm.K.tolist(),
                        "dist_coeffs": cm.dist.tolist(),
                        "image_width": int(cm.image_size[0]),
                        "image_height": int(cm.image_size[1]),
                    }
                )
            return jsonify(resp)
    except (
        Exception
    ) as exc:  # surface failures to the GUI (full traceback to the server log)
        logger.exception("generate_model failed")
        return jsonify({"success": False, "error": str(exc)}), 200


@calibration_bp.route("/calibration/scale_factor/generate", methods=["POST"])
def scale_factor_generate():
    """Build + save a scale-factor mono model from picked origin/axes + px/mm + dt.

    No detection, no fit — a direct uniform pixel->mm map. One frame is loaded only to
    stamp the image size and draw the proof figure. An explicit ``use_image: false``
    (the GUI's "Use calibration images" toggle off) skips the frame entirely: the size
    comes from the existing saved model and no figure is drawn — there is no silent
    fallback; without an existing model the request fails.
    """
    data = request.get_json() or {}
    source_idx = _source_idx(data.get)
    camera = int(data.get("camera") or _rig(data.get).get("camera") or 1)
    make_figs = not bool(data.get("no_figures", False))
    try:
        px_per_mm = float(data["px_per_mm"])
        origin_px = data["origin_px"]
        if origin_px is None or len(origin_px) != 2:
            raise ValueError("origin_px must be [x, y] pixels — click the origin first")
        x_dir = str(data.get("x_dir", "right"))
        y_dir = str(data.get("y_dir", "up"))
        swap = bool(data.get("swap_axes", False))
        frame_idx = int(data.get("frame_idx", 1))
        origin_mm_raw = data.get("origin_mm")
        if origin_mm_raw is not None and len(origin_mm_raw) != 2:
            raise ValueError("origin_mm must be [X, Y] millimetres")
        origin_mm = (
            (float(origin_mm_raw[0]), float(origin_mm_raw[1]))
            if origin_mm_raw is not None
            else (0.0, 0.0)
        )
        source = _source_path(source_idx)
        dt = _generate_dt(data.get, source)
        model_dir = rec.mono_model_dir_for_source(source, camera, "scale_factor")
        use_image = bool(data.get("use_image", True))
        if use_image:
            image = _load_one(
                camera,
                frame_idx,
                source_idx,
                data.get("image_format"),
                data.get("image_type"),
            )
            h, w = np.asarray(image).shape[:2]
            image_size = (int(w), int(h))
        else:
            image = None
            try:
                existing = rec.load_mono(model_dir, "scale_factor")
            except FileNotFoundError:
                raise ValueError(
                    f"Cannot generate without calibration images: no existing "
                    f"scale-factor model for Camera {camera} to take the image "
                    f"size from. Enable 'Use calibration images' and generate "
                    f"once with an image on disk first."
                )
            image_size = (
                int(existing.camera_model.image_size[0]),
                int(existing.camera_model.image_size[1]),
            )
        record = build_scale_factor_record(
            camera=camera,
            origin_px=origin_px,
            px_per_mm=px_per_mm,
            image_size=image_size,
            dt=dt,
            x_dir=x_dir,
            y_dir=y_dir,
            swap_axes=swap,
            frame_idx=frame_idx,
            origin_mm=origin_mm,
        )
        path = rec.save_mono(record, model_dir)
        fig_dir = model_dir.parent / "figures"
        if make_figs and image is not None:
            from pivtools_gui.calibration import figures as c2figs

            sf = record.camera_model
            # Draw at the PICKED origin (world_frame) — the model's own origin_px is the
            # world-zero pixel, off the picked point (possibly off-image) when origin_mm != 0.
            c2figs.write_scale_factor_figure(
                fig_dir,
                image=image,
                origin_px=record.world_frame.origin_px,
                col_sign=sf.col_sign,
                row_sign=sf.row_sign,
                swap_axes=bool(sf.swap_axes),
                mm_per_pixel=sf.mm_per_pixel,
                dt=dt,
                origin_mm=origin_mm,
            )
        resp = {
            "success": True,
            "stereo": False,
            "model_path": str(path),
            "camera": camera,
        }
        resp.update(
            _scale_factor_summary(
                record.camera_model, dt, frame_idx=frame_idx, wf=record.world_frame
            )
        )
        resp["figures"] = (
            _list_figures(fig_dir) if (make_figs and image is not None) else []
        )
        return jsonify(resp)
    except Exception as exc:
        logger.exception("scale_factor_generate failed")
        return jsonify({"success": False, "error": str(exc)}), 200


@calibration_bp.route("/calibration/model", methods=["GET"])
def load_model():
    """Return a summary of a saved model (mono or stereo) for the summary panel."""
    cfg = _cfg()
    board = request.args.get("board") or cfg.get("active", "charuco")
    try:
        source = _source_path(_source_idx(request.args.get))
    except (ValueError, IndexError):
        return jsonify({"exists": False}), 200
    if request.args.get("stereo") in ("1", "true", "True"):
        pair = request.args.get("camera_pair", "1,2").split(",")
        cam1, cam2 = int(pair[0]), int(pair[1])
        mdir = rec.stereo_model_dir_for_source(source, cam1, cam2)
        try:
            mpath = rec.resolve_stereo_path(
                mdir, _model_type_arg(request.args.get, board, stereo=True)
            )
            r = rec.load_stereo(mpath)
        except FileNotFoundError:
            return (
                jsonify({"exists": False, "world_frame": _inputs_world_frame(mdir)}),
                200,
            )
        except ValueError as exc:  # several types saved, none requested
            return jsonify({"exists": False, "error": str(exc)}), 200
        common = {
            "exists": True,
            "stereo": True,
            "cam1": cam1,
            "cam2": cam2,
            "model_path": str(mpath),
            "per_view_rms1": list(r.per_view_rms1),
            "per_view_rms2": list(r.per_view_rms2),
            "num_pairs_used": r.board_meta.get("n_stereo_views", len(r.per_view_rms1)),
            "world_frame_mode": r.world_frame.mode,
            "world_frame": _world_frame_payload(r.world_frame),
            "image_width": int(r.model1.image_size[0]),
            "image_height": int(r.model1.image_size[1]),
            "spacing_mm": _meta_float(r.board_meta, "spacing_mm"),
            "n_views": _meta_int(r.board_meta, "n_views"),
            "geometry": _geometry_payload(r.board_meta),
        }
        if isinstance(r.model1, Polynomial3DModel):
            # A polynomial pair has no extrinsic pose -> no baseline/angle (DaVis poly).
            common.update(
                {
                    "model_type": "polynomial3d",
                    "rms_cam1": float(r.model1.rms_px),
                    "rms_cam2": float(r.model2.rms_px),
                    "plane_rms_cam1": [float(v) for v in r.model1.plane_rms_px],
                    "plane_rms_cam2": [float(v) for v in r.model2.plane_rms_px],
                    "stereo_config": r.board_meta.get("stereo_config"),
                    "stereo_angle_deg": None,
                    "baseline_mm": None,
                }
            )
        else:
            common.update(
                {
                    "model_type": "pinhole",
                    "rms_cam1": r.model1.rms,
                    "rms_cam2": r.model2.rms,
                    "intrinsics1": _intrinsics(r.model1),
                    "intrinsics2": _intrinsics(r.model2),
                    "distortion_model": r.model1.distortion_model.value,
                    # The PIV-meaningful angle is the optical-axis angle, computed and
                    # stored at fit time (board_meta["relative_angle_deg"]). The R_stereo
                    # rotation angle below is a DIFFERENT quantity (it conflates the axis
                    # change with camera roll, e.g. ~178deg in transmission) — keep it only
                    # as a fallback for legacy models saved before the angle was persisted.
                    "stereo_angle_deg": (
                        float(r.board_meta["relative_angle_deg"])
                        if r.board_meta.get("relative_angle_deg") is not None
                        else float(
                            np.degrees(
                                np.arccos(
                                    np.clip((np.trace(r.R_stereo) - 1) / 2, -1, 1)
                                )
                            )
                        )
                    ),
                    "baseline_mm": float(np.linalg.norm(r.T_stereo)),
                    "stereo_rms_px": _finite_or_none(r.board_meta.get("stereo_rms_px")),
                    "method": r.board_meta.get("stereo_method"),
                }
            )
        return jsonify(common)
    camera = int(request.args.get("camera", 1))
    mdir = rec.mono_model_dir_for_source(source, camera, board)
    try:
        mpath = rec.resolve_mono_path(mdir, _model_type_arg(request.args.get, board))
        r = rec.load_mono(mpath)
    except FileNotFoundError:
        return jsonify({"exists": False, "world_frame": _inputs_world_frame(mdir)}), 200
    except ValueError as exc:  # several types saved, none requested
        return jsonify({"exists": False, "error": str(exc)}), 200
    cm = r.camera_model
    summary = {
        "exists": True,
        "stereo": False,
        "camera": camera,
        "model_path": str(mpath),
        "num_images_used": len(r.per_view_rms),
        "per_view_rms": list(r.per_view_rms),
        "world_frame_mode": r.world_frame.mode,
        "world_frame": _world_frame_payload(r.world_frame),
        "spacing_mm": _meta_float(r.board_meta, "spacing_mm"),
        "n_views": _meta_int(r.board_meta, "n_views"),
        "geometry": _geometry_payload(r.board_meta),
    }
    if isinstance(cm, ScaleFactorModel):
        summary.update(
            _scale_factor_summary(
                cm,
                _meta_float(r.board_meta, "dt") or 1.0,
                frame_idx=_meta_int(r.board_meta, "frame_idx"),
                wf=r.world_frame,
            )
        )
    elif isinstance(cm, Polynomial3DModel):
        summary.update(_polynomial3d_summary(cm))
    elif isinstance(cm, PolynomialModel):
        summary.update(_polynomial_summary(cm))
    else:
        summary.update(
            {
                "model_type": "pinhole",
                "rms": cm.rms,
                "fx": float(cm.K[0, 0]),
                "fy": float(cm.K[1, 1]),
                "cx": float(cm.K[0, 2]),
                "cy": float(cm.K[1, 2]),
                "camera_matrix": cm.K.tolist(),
                "dist_coeffs": cm.dist.tolist(),
                "distortion_model": cm.distortion_model.value,
                "image_width": int(cm.image_size[0]),
                "image_height": int(cm.image_size[1]),
            }
        )
    return jsonify(summary)


# ---------------------------------------------------------------------------
# Measure tool + figure serving
# ---------------------------------------------------------------------------


@calibration_bp.route("/calibration/measure", methods=["POST"])
def measure():
    """Distance in mm between two pixels, via the saved model's back-projection."""
    data = request.get_json() or {}
    cfg = _cfg()
    board = data.get("board") or cfg.get("active", "charuco")
    p1 = np.asarray(data.get("p1"), dtype=float)
    p2 = np.asarray(data.get("p2"), dtype=float)
    if p1.shape != (2,) or p2.shape != (2,):
        return jsonify({"error": "p1 and p2 must be [x, y] pixel pairs"}), 400
    z = float(data.get("z_world", 0.0))
    tx = float(data.get("tilt_x", 0.0))
    ty = float(data.get("tilt_y", 0.0))
    try:
        source = _source_path(_source_idx(data.get))
        if bool(data.get("stereo", False)):
            pair = data.get("camera_pair") or _rig(data.get).get("camera_pair") or [1, 2]
            cam1, cam2 = int(pair[0]), int(pair[1])
            model = rec.load_stereo(
                rec.stereo_model_dir_for_source(source, cam1, cam2),
                model_type=_model_type_arg(data.get, board, stereo=True),
            ).model1
        else:
            camera = int(data.get("camera") or _rig(data.get).get("camera") or 1)
            # Joint-preferred: a measure after a joint solve uses the unified rig model.
            model = rec.mono_record_for_camera(
                rec.mono_model_dir_for_source(source, camera, board),
                camera,
                _model_type_arg(data.get, board),
            ).camera_model
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return (
            jsonify(
                {
                    "error": f"no saved model — generate the model first to measure in mm ({exc})"
                }
            ),
            200,
        )
    world = c2apply.calibrate_coordinates(model, np.vstack([p1, p2]), z, tx, ty)
    if not np.all(np.isfinite(world)):
        return (
            jsonify({"error": "back-projection failed (ray missed the sheet plane)"}),
            200,
        )
    return jsonify(
        {
            "distance_mm": float(np.linalg.norm(world[1] - world[0])),
            "distance_px": float(np.linalg.norm(p2 - p1)),
            "world_p1": [float(world[0, 0]), float(world[0, 1])],
            "world_p2": [float(world[1, 0]), float(world[1, 1])],
        }
    )


@calibration_bp.route("/calibration/figures", methods=["GET"])
def list_figures():
    """List the proof figures written beside a model."""
    try:
        fig_dir = _figures_dir(request.args.get)
    except (ValueError, IndexError):
        return jsonify({"exists": False, "figures": []}), 200
    figs = _list_figures(fig_dir)
    return jsonify({"exists": bool(figs), "dir": str(fig_dir), "figures": figs})


@calibration_bp.route("/calibration/figure", methods=["GET"])
def get_figure():
    """Serve one proof-figure PNG (basename only — no path traversal)."""
    name = request.args.get("name", "")
    if not name or Path(name).name != name:
        return jsonify({"error": "invalid figure name"}), 400
    try:
        fig_dir = Path(_figures_dir(request.args.get)).resolve()
    except (ValueError, IndexError):
        return jsonify({"error": "source not configured"}), 404
    path = (fig_dir / name).resolve()
    if path.parent != fig_dir or not path.is_file():
        return jsonify({"error": "figure not found"}), 404
    return Response(path.read_bytes(), mimetype="image/png")


# ---------------------------------------------------------------------------
# Global coordinates — planar N-camera stitching
# ---------------------------------------------------------------------------


class _GlobalChainError(Exception):
    """User-facing failure while resolving the global datum chain."""


def _global_chain(data):
    """Shared setup for /global/compute and /global/save.

    Returns ``(dirs, records, shifts, datum_physical)`` — the per-camera model dirs +
    loaded records, the computed per-camera (shift_x, shift_y) mm, and the datum
    physical point. Raises ``_GlobalChainError`` (with a user-facing message) on a
    missing datum, a missing model, or a broken chain.
    """
    cfg = _cfg()
    board = data.get("board") or cfg.get("active", "charuco")
    datum_camera = int(data.get("datum_camera", 1))
    datum_pixel = data.get("datum_pixel")
    datum_physical = data.get("datum_physical", [0.0, 0.0])
    overlap_pairs = data.get("overlap_pairs") or []
    z = float(data.get("z_world", 0.0))
    tx = float(data.get("tilt_x", 0.0))
    ty = float(data.get("tilt_y", 0.0))
    if not datum_pixel:
        raise _GlobalChainError(
            "datum_pixel not set — click a point on the datum camera"
        )

    cams = {datum_camera}
    for p in overlap_pairs:
        cams.add(int(p["camera_a"]))
        cams.add(int(p["camera_b"]))
    try:
        source = _source_path(_source_idx(data.get))
        dirs = {cam: rec.mono_model_dir_for_source(source, cam, board) for cam in cams}
        mtype = _model_type_arg(data.get, board)
        records = {cam: rec.load_mono(d, model_type=mtype) for cam, d in dirs.items()}
    except (FileNotFoundError, ValueError, IndexError) as exc:
        raise _GlobalChainError(
            f"missing mono model — calibrate each camera first ({exc})"
        )
    try:
        shifts = gc2.compute_camera_shifts(
            records, datum_camera, datum_pixel, datum_physical, overlap_pairs, z, tx, ty
        )
    except (ValueError, KeyError) as exc:
        raise _GlobalChainError(str(exc))
    return dirs, records, shifts, datum_physical


def _shifts_payload(shifts, datum_physical, **extra):
    # Stringify camera keys (Flask jsonify mixes int/str keys badly).
    return {
        "camera_shifts": {
            str(c): [float(s[0]), float(s[1])] for c, s in shifts.items()
        },
        "datum_physical": [float(datum_physical[0]), float(datum_physical[1])],
        **extra,
    }


@calibration_bp.route("/calibration/global/compute", methods=["POST"])
def global_compute():
    """Per-camera world shifts from a datum + overlap pairs — preview only, persists nothing."""
    try:
        _dirs, _records, shifts, datum_physical = _global_chain(
            request.get_json() or {}
        )
    except _GlobalChainError as exc:
        return jsonify({"error": str(exc)}), 200
    return jsonify(_shifts_payload(shifts, datum_physical))


@calibration_bp.route("/calibration/global/save", methods=["POST"])
def global_save():
    """Compute the datum-chain shifts and BAKE them into each camera's model.

    Same body as ``/global/compute``, but persistent: each reachable camera's
    ``world_offset_mm`` is written into its model record so the apply step reads it
    and emits coordinates in the shared rig frame. Regenerating a camera's model
    clears its offset (fresh world frame), so re-save the global frame after any
    recalibration. Cameras not reached by the chain are left untouched.
    """
    try:
        dirs, records, shifts, datum_physical = _global_chain(request.get_json() or {})
    except _GlobalChainError as exc:
        return jsonify({"error": str(exc)}), 200

    for cam, (sx, sy) in shifts.items():
        r = records[cam]
        r.world_frame.world_offset_mm = np.array(
            [float(sx), float(sy)], dtype=np.float64
        )
        rec.save_mono(r, dirs[cam])

    return jsonify(
        _shifts_payload(
            shifts,
            datum_physical,
            success=True,
            cameras_saved=sorted(int(c) for c in shifts),
        )
    )


# ---------------------------------------------------------------------------
# Apply the model to PIV output (background job)
# ---------------------------------------------------------------------------


def _apply_units(data, full_cfg, source, board, stereo, type_name):
    """Resolve apply units for a Flask request — thin wrapper over ``runio.plan_apply_units``.

    Translates the request body into the shared planner's args: an explicit
    ``uncalibrated_dir``/``calibrated_dir`` (mono) or ``uncalibrated_dir_cam1/2`` (stereo)
    forces a single ad-hoc unit; otherwise units are derived from config across
    ``active_paths`` x cameras. The planner does the config derivation + per-unit model load
    (so the CLI's ``--all-paths`` apply uses the identical logic).
    """
    get = data.get
    explicit = None
    if stereo:
        if get("calibrated_dir") and get("uncalibrated_dir_cam1"):
            explicit = {
                "uncal1": data["uncalibrated_dir_cam1"],
                "uncal2": data["uncalibrated_dir_cam2"],
                "out": data["calibrated_dir"],
            }
    elif get("calibrated_dir") and get("uncalibrated_dir"):
        explicit = {"uncal": data["uncalibrated_dir"], "out": data["calibrated_dir"]}
    return c2runio.plan_apply_units(
        full_cfg,
        source,
        board,
        stereo,
        type_name,
        active_paths=get("active_paths"),
        camera_pair=(get("camera_pair") or _rig(get).get("camera_pair") or [1, 2]),
        camera=get("camera"),
        explicit=explicit,
        model_type=_model_type_arg(get, board, stereo=stereo),
    )


@calibration_bp.route("/calibration/set_datum", methods=["POST"])
def calibration_set_datum():
    """Set a new datum (origin) and/or apply offsets to ALL runs in a type's coordinates.

    Serves the vector viewer's Coordinate System panel. The datum ``x``/``y`` (a clicked
    physical position) is subtracted from every coordinate, then ``x_offset``/``y_offset``
    is added — offsets are additive per request, not absolute. Every run in the type's
    ``coordinates.mat`` is rewritten; stereo ``z`` is preserved untouched. Statistics
    coordinates (``statistics/.../mean_stats/coordinates.mat``) are NOT modified — the
    viewer hides coordinate editing for mean-stats variables.

    JSON body: ``base_path`` (optional, wins over idx) or ``base_path_idx``; ``camera``;
    ``run`` (logging only); ``type_name`` (default ``instantaneous``); ``x``/``y``
    (optional datum, both required together); ``x_offset``/``y_offset`` (default 0);
    ``merged``; ``use_stereo`` + ``camera_pair``.
    """
    data = request.get_json() or {}
    full_cfg = get_config()

    base_path = data.get("base_path")
    if not base_path:
        base_idx = int(data.get("base_path_idx", 0))
        try:
            base_path = full_cfg.base_paths[base_idx]
        except IndexError:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"base_path_idx {base_idx} out of range",
                    }
                ),
                400,
            )

    camera = camera_number(data.get("camera", 1))
    run = int(data.get("run", 1))  # logging only: the update spans all runs
    type_name = data.get("type_name", "instantaneous")
    x0 = data.get("x")
    y0 = data.get("y")
    x_offset = float(data.get("x_offset") or 0)
    y_offset = float(data.get("y_offset") or 0)
    use_merged = bool(data.get("merged"))
    use_stereo = bool(data.get("use_stereo"))
    camera_pair = data.get("camera_pair")
    stereo_camera_pair = None
    if use_stereo and isinstance(camera_pair, (list, tuple)) and len(camera_pair) >= 2:
        stereo_camera_pair = (int(camera_pair[0]), int(camera_pair[1]))

    try:
        paths = get_data_paths(
            base_dir=Path(base_path),
            num_frame_pairs=full_cfg.num_frame_pairs,
            cam=camera,
            type_name=type_name,
            use_merged=use_merged,
            use_stereo=use_stereo,
            stereo_camera_pair=stereo_camera_pair,
        )
        coords_path = Path(paths["data_dir"]) / "coordinates.mat"
        if not coords_path.exists():
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Coordinates file not found: {coords_path}",
                    }
                ),
                404,
            )

        mat = scipy.io.loadmat(
            str(coords_path), struct_as_record=False, squeeze_me=True
        )
        if "coordinates" not in mat:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Variable 'coordinates' not found in {coords_path}",
                    }
                ),
                400,
            )
        coordinates = mat["coordinates"]
        num_runs = get_num_coordinate_runs(coordinates)
        is_multi = isinstance(coordinates, np.ndarray) and coordinates.dtype == object
        has_z = hasattr(coordinates[0] if is_multi else coordinates, "z")

        fields = [("x", object), ("y", object)] + ([("z", object)] if has_z else [])
        coords_struct = np.empty((num_runs,), dtype=fields)
        for i in range(num_runs):
            cx, cy = extract_coordinates(coordinates, i + 1)
            if x0 is not None and y0 is not None:
                cx = cx - float(x0)
                cy = cy - float(y0)
            coords_struct["x"][i] = cx + x_offset
            coords_struct["y"][i] = cy + y_offset
            if has_z:
                el = coordinates[i] if is_multi else coordinates
                coords_struct["z"][i] = np.asarray(el.z)

        scipy.io.savemat(
            str(coords_path), {"coordinates": coords_struct}, do_compression=True
        )
        logger.info(
            f"[set_datum] {type_name} Cam{camera} (from run {run}): datum=({x0}, {y0}) "
            f"offset=({x_offset}, {y_offset}) -> {num_runs} runs in {coords_path}"
        )
        return jsonify(
            {
                "success": True,
                "type_name": type_name,
                "num_runs_updated": num_runs,
                "coords_path": str(coords_path),
                "x0": x0,
                "y0": y0,
                "x_offset": x_offset,
                "y_offset": y_offset,
            }
        )
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("[set_datum] failed")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@calibration_bp.route("/calibration/apply", methods=["POST"])
def apply_model():
    """Start applying a saved model to PIV output (mono apply / stereo 3C) as a job.

    Directories are derived from config: the model from the calibration ``source``, the
    PIV data from each base path in ``active_paths`` x ``camera_numbers`` (mono) or the
    stereo pair. One job spans all units; progress reports completed units.
    """
    data = request.get_json() or {}
    full_cfg = get_config()
    cfg = _cfg()
    board = data.get("board") or cfg.get("active", "charuco")
    stereo = bool(data.get("stereo", False))
    # Light-sheet plane: request > 0.0 for mono (the board plane, a geometric
    # identity); stereo additionally defaults from the record's self-cal below.
    z = float(data.get("z_world", 0.0))
    tx = float(data.get("tilt_x", 0.0))
    ty = float(data.get("tilt_y", 0.0))
    # Stereo cam2 resample kernel (validated up front, before the worker starts).
    interpolator = data.get("interpolator")
    if interpolator not in ("linear", "cubic", "lanczos", None, ""):
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"interpolator must be linear|cubic|lanczos, got {interpolator!r}",
                }
            ),
            200,
        )
    # Derive the per-frame vector glob from the PIV vector_format (e.g. "%05d.mat" ->
    # "*.mat"); the old hardcoded "B*.mat" matched nothing under the default naming, so
    # apply wrote coordinates.mat but no calibrated vectors.
    vector_glob = data.get("vector_glob") or vector_glob_from_format(
        full_cfg.vector_format
    )
    type_name = data.get("type_name") or "instantaneous"

    try:
        source = _source_path(_source_idx(data.get))
        if not interpolator:
            # Settings-sidecar knob (defaulted there); no config source. The
            # sidecar value gets the same membership check as a request value —
            # a typo'd rig.interpolator must fail here, not inside the worker.
            settings = cs.try_load_settings(source)
            interpolator = ((settings or {}).get("rig") or {}).get(
                "interpolator", "lanczos"
            )
            if interpolator not in ("linear", "cubic", "lanczos"):
                raise ValueError(
                    f"rig.interpolator in the settings sidecar must be "
                    f"linear|cubic|lanczos, got {interpolator!r}"
                )
        units = _apply_units(data, full_cfg, source, board, stereo, type_name)
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"model not found — generate it first ({exc})",
                }
            ),
            200,
        )
    except KeyError as exc:
        return (
            jsonify({"success": False, "error": f"missing directory field {exc}"}),
            200,
        )

    # Stereo: default the laser-sheet (z_world, tilt) from the saved self-calibration
    # block unless the request explicitly overrides it. So once self-cal is stored, 3C
    # reconstruction sits on the true sheet automatically.
    if stereo and units:
        # StereoRecord.sc_* mirror the CLI apply's fallback — one implementation
        # of "self-cal leg, 0.0 when absent" on the record, not two dict reads.
        rec0 = units[0]["record"]
        if "z_world" not in data:
            z = rec0.sc_z_offset
        if "tilt_x" not in data:
            tx = rec0.sc_tilt_x
        if "tilt_y" not in data:
            ty = rec0.sc_tilt_y

    if not units:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "no datasets selected — configure base paths first",
                }
            ),
            200,
        )

    # dt is resolved PER UNIT: request override > model-stamped (every generated
    # record carries it in board_meta). Resolving per unit (not once from units[0])
    # means a multi-camera rig whose cameras carry different stamped dt calibrates
    # each camera with ITS OWN dt instead of camera 1's. Velocity has no safe
    # default and no config source, so an unresolved dt fails loudly here, before
    # the job thread starts.
    explicit_dt = data.get("dt")
    try:
        for u in units:
            model_dt = (getattr(u["record"], "board_meta", None) or {}).get("dt")
            u["dt"] = c2runio.resolve_dt(explicit_dt, model_dt)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 200

    # Built here (not in the worker) so it captures the source/board this request
    # resolved, independent of any config change made while the job runs.
    applied_pointer = c2runio.build_applied_pointer(source, board, stereo)

    n_units = len(units)
    out_dirs = [str(u["out"]) for u in units]
    job_id = job_manager.create_job(
        "calibration_apply", stereo=stereo, out_dir=out_dirs[0]
    )

    def _run():
        try:
            job_manager.update_job(
                job_id, status="running", progress=0, processed=0, total=n_units
            )
            total_written = 0
            for i, u in enumerate(units):
                # Which dataset is running, published before the first frame ticks so the
                # label is never a unit behind. Frame counts reset to 0 ("not yet reported")
                # rather than carrying the previous unit's totals into this one.
                label = u["label"]
                job_manager.update_job(
                    job_id, unit_label=label, frame_done=0, frame_total=0
                )

                def cb(done, total, _i=i, _label=label):
                    pct = int(((_i + done / max(total, 1)) / n_units) * 100)
                    job_manager.update_job(
                        job_id,
                        progress=pct,
                        processed=_i,
                        total=n_units,
                        frame_done=done,
                        frame_total=total,
                        unit_label=_label,
                    )

                if u["stereo"]:
                    written = c2runio.reconstruct_stereo_run(
                        u["record"],
                        u["uncal1"],
                        u["uncal2"],
                        u["out"],
                        u["dt"],
                        None,
                        z,
                        tx,
                        ty,
                        progress_cb=cb,
                        vector_glob=vector_glob,
                        interpolator=interpolator,
                    )
                else:
                    written = c2runio.calibrate_mono_run(
                        u["record"],
                        u["uncal"],
                        u["out"],
                        u["dt"],
                        z,
                        tx,
                        ty,
                        progress_cb=cb,
                        vector_glob=vector_glob,
                    )
                total_written += len(written)
                # Record what actually calibrated these vectors in the run's archived
                # config. The snapshot was written when PIV started, so its calibration
                # block still names whatever dataset was configured before this one.
                c2runio.stamp_unit(u, applied_pointer)
                job_manager.update_job(job_id, processed=i + 1, total=n_units)
            job_manager.complete_job(
                job_id, n_frames=total_written, out_dir="; ".join(out_dirs)
            )
        except (
            Exception
        ) as exc:  # surface in the job status (full traceback to the server log)
            logger.exception("apply job %s failed", job_id)
            job_manager.fail_job(job_id, str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"success": True, "job_id": job_id})


@calibration_bp.route("/calibration/apply/status/<job_id>", methods=["GET"])
def apply_status(job_id):
    """Poll an apply job's status (same shape as the v1 vector-job status)."""
    return _job_status_response(job_id)


# ---------------------------------------------------------------------------
# Joint multi-camera calibration (continuous global grid + shared released board)
#
# The DaVis-matching solve: one shared board every camera observes, tied by a global dot
# index. ChArUco gets the grid free from corner ids (zero clicks); dotboard needs a datum
# world frame + cross-camera link clicks (the GlobalGridSpec). resolve_grid drives the live
# overlay; generate runs the solve as a job; model loads the saved record. All three share the
# run_joint_from_spec driver with the CLI (calibration_cli.detect_joint_command), so the GUI
# and headless paths cannot drift.
# ---------------------------------------------------------------------------


def _joint_int_list(value) -> List[int]:
    """Coerce a cameras value (list, or "1,2,3" string) to a list of ints."""
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [int(x) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def _joint_setup(get: Callable[[str], Any]):
    """Resolve the shared joint inputs from a request.

    Returns (cfg, gg, board, source, source_idx, cameras, n_views, image_format, image_type,
    params, spacing, datum_camera, datum_view). Raises ValueError for the caller to surface.
    """
    cfg = _cfg()
    board = get("board") or cfg.get("active", "dotboard")
    if board not in ("dotboard", "charuco"):
        raise ValueError(f"joint: board must be dotboard|charuco, got {board!r}")
    source_idx = _source_idx(get)
    source = _source_path(source_idx)
    settings = _settings_idx(source_idx)
    image_settings = settings.get("image") or {}
    # The clicked coords (datum + anchors + camera_extends + cameras) live in the sidecar
    # inputs.mat, not config. Same dict shape as the old global_grid block, so the gg.get(...)
    # reads below and _joint_spec / _joint_origin_mm are unchanged. A request override (the live
    # wizard) still wins in _joint_spec.
    side = try_load_inputs(rec.joint_model_dir_for_source(source, board))
    gg = dict(side.coords) if (side and side.coords) else {}
    cameras = (
        _joint_int_list(get("cameras"))
        or _joint_int_list(gg.get("cameras"))
        or _joint_int_list(get_config().camera_numbers)
    )
    if not cameras:
        raise ValueError("joint: no cameras — set the rig cameras in the joint wizard")

    def _int_or(name, fallback):
        v = get(name)
        return int(v) if v not in (None, "") else int(fallback)

    datum_camera = _int_or("datum_camera", gg.get("datum_camera", cameras[0]))
    datum_view = _int_or("datum_view", gg.get("datum_view", 0))
    n_views = _int_or(
        "n_views",
        (
            get("frame_total")
            if get("frame_total") not in (None, "")
            else image_settings.get("n_views") or 10
        ),
    )
    if n_views < 1:
        raise ValueError("joint: n_views must be >= 1")
    image_format = get("image_format") or image_settings.get("image_format")
    image_type = get("image_type") or image_settings.get("image_type")
    params = c2._board_params(settings.get("methods") or {}, board)
    spacing = c2._spacing_mm(board, params)
    return (
        cfg,
        gg,
        board,
        source,
        source_idx,
        cameras,
        n_views,
        image_format,
        image_type,
        params,
        spacing,
        datum_camera,
        datum_view,
    )


def _joint_detect(
    board,
    params,
    cameras,
    n_views,
    source_idx,
    image_format,
    image_type,
    spacing,
    *,
    refresh=False,
    progress_cb=None,
):
    """Detect every view of every camera, cached by (source, board, n_views, format, params).

    Returns (detections_by_cam, image_size_by_cam). A view that fails detection is kept in place
    as an unsuccessful DetectionResult, NOT raised on: a single bad frame must not blank the whole
    overlay or abort the solve. The resolvers skip failed views (reporting them per-view) and the
    solve simply uses the views that detected. The cache makes the live resolve loop and the
    generate job cheap; pass refresh=True after the images change on disk.
    """
    # The camera set is part of the key: a hit only detected the cameras it was asked for, so
    # reusing it for a different set would hand the solver a stale subset (caught by run_joint's
    # expected_cameras guard, but with a confusing message blaming the grid, not the cache).
    key = (
        int(source_idx),
        str(board),
        int(n_views),
        str(image_format),
        str(image_type),
        tuple(sorted(int(c) for c in cameras)),
        repr(params),
    )
    det_key = joint_det_key(
        board, n_views, image_format, infer_image_type(image_format), cameras, params
    )
    if not refresh:
        with _joint_detect_lock:
            hit = _joint_detect_cache.get(key)
        if hit is not None:
            return hit["detections"], hit["image_size_by_cam"]
        # Fall back to the persistent sidecar (survives restarts + a model delete); promote it
        # into memory on a hit. det_key guards against serving detections from a different
        # n_views / format / board param.
        idir = _joint_inputs_dir(source_idx, board)
        rec_in = try_load_inputs(idir) if idir is not None else None
        # Reuse only when the sidecar loaded cleanly AND its detections were made with the same
        # params (det_key); absent / corrupt / stale-key all fall through to a fresh detect.
        if rec_in is not None and rec_in.detections and rec_in.det_key == det_key:
            payload = {
                "detections": rec_in.detections,
                "image_size_by_cam": rec_in.image_size_by_cam,
            }
            with _joint_detect_lock:
                _joint_detect_cache[key] = payload
            return payload["detections"], payload["image_size_by_cam"]

    total = len(cameras) * int(n_views)
    done = 0
    detections: dict = {}
    image_size_by_cam: dict = {}
    for cam in cameras:
        imgs = _load_views(cam, n_views, source_idx, image_format, image_type)
        if not imgs:
            raise ValueError(f"joint: no images loaded for cam{cam}")

        def _on_done():
            nonlocal done
            done += 1
            if progress_cb is not None:
                progress_cb(done, total)

        detections[cam] = _detect_parallel(
            board, params, imgs, spacing_mm=spacing, on_done=_on_done
        )
        h, w = imgs[0].shape[:2]
        image_size_by_cam[cam] = (int(w), int(h))

    # A camera with NO successful view cannot be calibrated at all — that is the one fatal case
    # (almost always a wrong path/format, not a single bad target). Individual failed views are
    # fine: they are dropped downstream and reported per-view.
    blank = [c for c in cameras if not any(d.success for d in detections[c])]
    if blank:
        raise ValueError(
            f"joint: camera(s) {blank} detected no calibration target in any image — check the "
            f"image path, format and board parameters"
        )

    payload = {"detections": detections, "image_size_by_cam": image_size_by_cam}
    with _joint_detect_lock:
        _joint_detect_cache[key] = payload
    # Persist into the sidecar for the next session (best-effort: a read-only source just keeps
    # the in-memory cache). Merges, so an in-progress coords/click commit is preserved.
    idir = _joint_inputs_dir(source_idx, board)
    if idir is not None:
        try:
            save_inputs(
                idir,
                path_type="joint",
                board_type=board,
                detections=detections,
                image_size_by_cam=image_size_by_cam,
                det_key=det_key,
                board_params=rec.geometry_meta(board, params),
            )
        except (OSError, ValueError):
            pass
    return detections, image_size_by_cam


def _joint_spec(get, gg, board):
    """GlobalGridSpec from the request's global_grid block (preferred) or config; None for
    ChArUco. Raises ValueError on an incomplete dotboard spec (caller decides how to surface).
    """
    if board == "charuco":
        return None
    block = get("global_grid") or gg
    try:
        return c2._global_grid_spec_from_cfg(block)
    except (
        SystemExit
    ) as exc:  # the CLI builder signals validation errors via SystemExit
        raise ValueError(str(exc))


def _joint_origin_mm(board, gg) -> tuple:
    if board == "dotboard":
        om = (gg.get("datum_clicks", {}) or {}).get("origin_mm", [0.0, 0.0])
        return (float(om[0]), float(om[1]))
    return (0.0, 0.0)


@calibration_bp.route("/calibration/joint/inputs", methods=["GET"])
def joint_inputs():
    """The saved joint clicked-coords block (datum + anchors + camera_extends), or null.

    The Set-Global-Coordinates wizard restores its state from this — it replaces reading the
    old ``config.calibration.global_grid`` block.
    """
    get = request.args.get
    cfg = _cfg()
    board = get("board") or cfg.get("active", "dotboard")
    try:
        source = _source_path(_source_idx(get))
    except (ValueError, IndexError):
        return jsonify({"coords": None}), 200
    side = try_load_inputs(rec.joint_model_dir_for_source(source, board))
    return jsonify({"coords": (side.coords if side else None)}), 200


@calibration_bp.route("/calibration/joint/inputs/save", methods=["POST"])
def joint_inputs_save():
    """Persist the joint clicked-coords block into the sidecar ``inputs.mat``.

    Merges, so the detections already stored by ``_joint_detect`` are untouched. Replaces the
    old ``config.calibration.global_grid`` write.
    """
    data = request.get_json() or {}
    get = data.get
    cfg = _cfg()
    board = get("board") or cfg.get("active", "dotboard")
    coords = get("global_grid")
    if coords is None:
        return jsonify({"success": False, "error": "missing 'global_grid' block"}), 200
    try:
        source = _source_path(_source_idx(get))
        save_inputs(
            rec.joint_model_dir_for_source(source, board),
            path_type="joint",
            board_type=board,
            coords=coords,
        )
    except (ValueError, IndexError, OSError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 200
    return jsonify({"success": True}), 200


@calibration_bp.route("/calibration/joint/resolve_grid", methods=["POST"])
def joint_resolve_grid():
    """Detect all views + resolve the global grid for the live overlay.

    One entry per (camera, view) with its detected dot pixels and, when resolvable, the global
    (gx, gy) index of each dot. A dotboard spec still incomplete (no datum / missing links) is
    NOT an error here — those views return points but no indices plus a reason, so the GUI shows
    the dots and guides the next click.
    """
    data = request.get_json() or {}
    get = data.get
    try:
        (
            cfg,
            gg,
            board,
            source,
            source_idx,
            cameras,
            n_views,
            image_format,
            image_type,
            params,
            spacing,
            datum_camera,
            datum_view,
        ) = _joint_setup(get)
        detections, _ = _joint_detect(
            board,
            params,
            cameras,
            n_views,
            source_idx,
            image_format,
            image_type,
            spacing,
            refresh=bool(get("refresh")),
        )
    except (
        ValueError,
        TypeError,
    ) as exc:  # TypeError: a malformed payload (non-int cameras etc.)
        return jsonify({"success": False, "error": str(exc)}), 200

    resolved = {}
    unresolved = []
    errors = []
    try:
        spec = _joint_spec(get, gg, board)
        # Partial (non-raising) resolve: a half-built dotboard spec resolves the datum + linked
        # views and reports a per-view reason for the rest, so the overlay fills in as the user
        # clicks instead of blanking on the first unresolved view. ChArUco resolves every view.
        resolved, unresolved = resolve_global_grid_partial(
            detections, spec, spacing_mm=spacing
        )
    except ValueError as exc:
        errors.append(
            str(exc)
        )  # whole-spec construction failure — overlay still shows the dots

    reason_by_view = {(int(c), int(v)): r for c, v, r in unresolved}

    views = []
    for cam in cameras:
        for v, d in enumerate(detections[cam]):
            gi = resolved.get((cam, v))
            # A view that failed detection carries no dots — say so explicitly so the user can skip
            # that frame rather than wonder why the overlay is empty there.
            reason = reason_by_view.get((cam, v))
            if not d.success and reason is None:
                reason = "detection failed — skip this frame or replace the image"
            # The detector's LOCAL (col, row) grid indices let the overlay draw the dot mesh
            # immediately on Preview — before any anchoring assigns a global index. Per-view
            # connectivity is identical whichever indexing is used.
            local_idx = (
                None
                if (not d.success or d.grid_indices is None)
                else np.asarray(d.grid_indices, dtype=int).tolist()
            )
            # When a thin-overlap bridge leaves a mirror ambiguity the dots cannot break, offer the
            # competing layouts so the GUI can ask the user which footprint is real (confirm-on-
            # overlay). Returns [] for any view that resolves cleanly or carries no such bridge, so
            # only the genuinely ambiguous view sprouts a picker.
            candidates = (
                (
                    []
                    if gi is not None
                    else first_view_orientation_candidates(
                        detections, spec, cam, v, spacing_mm=spacing
                    )
                )
                if not errors
                else []
            )
            views.append(
                {
                    "camera": cam,
                    "view": v,
                    "n": int(d.n),
                    "points": np.asarray(d.image_points, dtype=float).tolist(),
                    "grid_indices": local_idx,
                    "global_index": (
                        None if gi is None else np.asarray(gi, dtype=int).tolist()
                    ),
                    "resolved": gi is not None,
                    "reason": reason,
                    "candidates": candidates,
                }
            )
    return jsonify(
        {
            "success": True,
            "board": board,
            "cameras": cameras,
            "n_views": n_views,
            "spacing_mm": float(spacing),
            "datum_camera": datum_camera,
            "datum_view": datum_view,
            "views": views,
            "n_resolved": sum(1 for vw in views if vw["resolved"]),
            "n_views_total": len(views),
            "errors": errors,
        }
    )


def _finite_or_none(x) -> Optional[float]:
    """Float for JSON, or None when non-finite.

    Flask's default ``jsonify`` emits a bare ``NaN``/``Infinity`` token — invalid JSON that
    ``JSON.parse`` rejects, which silently stalls the frontend job poll. The polynomial joint
    solve legitimately has no reprojection-px RMS or cross-camera agreement (both NaN), so those
    fields must serialise as ``null``, not ``NaN``.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


@calibration_bp.route("/calibration/joint/generate", methods=["POST"])
def joint_generate():
    """Run the joint solve (pinhole) or per-camera polynomial as a background job."""
    data = request.get_json() or {}
    get = data.get
    try:
        (
            cfg,
            gg,
            board,
            source,
            source_idx,
            cameras,
            n_views,
            image_format,
            image_type,
            params,
            spacing,
            datum_camera,
            datum_view,
        ) = _joint_setup(get)
        model_type = get("model_type") or _methods(get).get(board, {}).get(
            "model_type", "pinhole"
        )
        if model_type not in ("pinhole", "polynomial"):
            raise ValueError(
                f"joint: model_type must be pinhole|polynomial, got {model_type!r}"
            )
        board_release = get("board_release") or gg.get("board_release", "full3d")
        spec = _joint_spec(get, gg, board)
        origin_mm = _joint_origin_mm(board, gg)
        gen_dt = _generate_dt(get, source)
    except (
        ValueError,
        TypeError,
    ) as exc:  # TypeError: a malformed payload (non-int cameras etc.)
        return jsonify({"success": False, "error": str(exc)}), 200

    job_id = job_manager.create_job(
        "calibration_joint", board=board, model_type=model_type
    )

    def _run():
        try:
            job_manager.update_job(job_id, status="running", progress=0)

            def detect_cb(done, total):
                job_manager.update_job(
                    job_id,
                    progress=int(done / max(total, 1) * 80),
                    processed=done,
                    total=total,
                )

            detections, image_size_by_cam = _joint_detect(
                board,
                params,
                cameras,
                n_views,
                source_idx,
                image_format,
                image_type,
                spacing,
                progress_cb=detect_cb,
            )
            job_manager.update_job(job_id, progress=85)

            # Proof figures land beside the joint record. The driver needs raw images for the
            # detection overlays + dewarp; re-load them lazily one camera at a time (the detect
            # step above discarded them) so we never hold the whole rig in memory at once.
            figure_dir = (
                rec.joint_model_dir_for_source(source, board).parent / "figures"
            )
            _img_cache: dict = {}

            def image_loader(cam, view):
                if _img_cache.get("cam") != cam:
                    _img_cache["cam"] = cam
                    try:
                        _img_cache["imgs"] = _load_views(
                            cam, n_views, source_idx, image_format, image_type
                        )
                    except Exception:
                        # Visible, not silent (CLAUDE.md): the calibration is unaffected, but the
                        # image-based figures (detection overlays, dewarp) are skipped for this
                        # camera — say so once rather than returning blank frames quietly.
                        logger.warning(
                            "joint figures: could not load images for cam%s — its "
                            "detection/dewarp figures are skipped",
                            cam,
                        )
                        _img_cache["imgs"] = []
                imgs = _img_cache.get("imgs") or []
                return imgs[view] if 0 <= view < len(imgs) else None

            res = run_joint_from_spec(
                detections,
                image_size_by_cam,
                source=source,
                board=board,
                model_type=model_type,
                spacing_mm=spacing,
                dt=gen_dt,
                datum_camera=datum_camera,
                datum_view=datum_view,
                board_release=board_release,
                origin_mm=origin_mm,
                spec=spec,
                cameras=cameras,
                distortion_model=_MODEL,
                fix_aspect_ratio=True,
                n_views=n_views,
                figure_dir=figure_dir,
                image_loader=image_loader,
                board_params=params,
            )
            job_manager.complete_job(
                job_id,
                out_dir="; ".join(str(p) for p in res.paths),
                model_type=res.model_type,
                cameras=[int(c) for c in res.cameras],
                per_camera_rms={
                    str(c): _finite_or_none(r) for c, r in res.per_camera_rms.items()
                },
                rms_units=res.rms_units,
                rms_px=_finite_or_none(res.rms_px),
                converged=bool(res.converged),
                cross_camera_board_agreement_mm=_finite_or_none(
                    res.cross_camera_board_agreement_mm
                ),
                n_board_dots=int(res.n_board_dots),
                paths=[str(p) for p in res.paths],
            )
        except (
            Exception
        ) as exc:  # surface in the job status (full traceback to the server log)
            logger.exception("joint generate job %s failed", job_id)
            job_manager.fail_job(job_id, str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"success": True, "job_id": job_id})


@calibration_bp.route("/calibration/joint/generate/status/<job_id>", methods=["GET"])
def joint_generate_status(job_id):
    """Poll a joint-generate job's status (same shape as the apply job)."""
    return _job_status_response(job_id)


@calibration_bp.route("/calibration/joint/model", methods=["GET"])
def joint_model():
    """Load a saved joint model for display: the unified pinhole JointRecord, or per-camera
    polynomial records (which have no shared board to unify)."""
    get = request.args.get
    cfg = _cfg()
    board = get("board") or cfg.get("active", "dotboard")
    source = _source_path(_source_idx(get))
    model_type = _model_type_arg(get, board)
    side = try_load_inputs(rec.joint_model_dir_for_source(source, board))
    gg = dict(side.coords) if (side and side.coords) else {}

    if model_type == "polynomial":
        cameras = (
            _joint_int_list(get("cameras"))
            or _joint_int_list(gg.get("cameras"))
            or _joint_int_list(get_config().camera_numbers)
        )
        out = {}
        geom = None
        for cam in cameras:
            try:
                mono = rec.load_mono(
                    rec.mono_model_dir_for_source(source, cam, board), "polynomial"
                )
            except (FileNotFoundError, ValueError):
                continue
            if geom is None:
                geom = _geometry_payload(mono.board_meta)
            # Full coefficient summary (coeffs_x/y, normalisation, mm RMS) so the results card can
            # show the actual fitted polynomial, mirroring the pinhole per-camera parameters.
            out[str(cam)] = _polynomial_summary(mono.camera_model)
        if not out:
            return jsonify({"exists": False})
        return jsonify(
            {
                "exists": True,
                "model_type": "polynomial",
                "board": board,
                "cameras": [int(c) for c in out],
                "per_camera": out,
                "geometry": geom,
            }
        )

    try:
        jp = rec.resolve_joint_path(
            rec.joint_model_dir_for_source(source, board), "pinhole"
        )
        jr = rec.load_joint(jp)
    except (FileNotFoundError, ValueError):
        return jsonify({"exists": False})
    meta = jr.board_meta or {}

    # Per-camera intrinsics + extrinsics for the results panel. Position is the camera centre in
    # the world/board frame (C = -R^T t); rotation is the world->camera orientation as Tait-Bryan
    # euler angles in degrees (cv2.RQDecomp3x3, the same convention DaVis reports). Focal length is
    # px only — mm would need the sensor pixel size, which the model does not store.
    per_camera = {}
    positions = {}
    for c in jr.cameras:
        m = jr.models[c]
        K = np.asarray(m.K, dtype=np.float64)
        R = np.asarray(m.R, dtype=np.float64)
        t = np.asarray(m.t, dtype=np.float64).reshape(3)
        pos = (-R.T @ t).ravel()
        positions[int(c)] = pos
        euler = cv2.RQDecomp3x3(R)[0]
        per_camera[str(c)] = {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "dist": [float(x) for x in np.asarray(m.dist, dtype=np.float64).ravel()],
            "image_size": [int(m.image_size[0]), int(m.image_size[1])],
            "rms_px": float(jr.per_camera_rms.get(c, float("nan"))),
            "position_mm": [float(x) for x in pos],
            "rotation_deg": [
                float(x) for x in np.asarray(euler, dtype=np.float64).ravel()
            ],
        }

    cams_sorted = sorted(int(c) for c in jr.cameras)
    baselines_mm = {
        f"{a}-{b}": float(np.linalg.norm(positions[a] - positions[b]))
        for i, a in enumerate(cams_sorted)
        for b in cams_sorted[i + 1 :]
    }

    return jsonify(
        {
            "exists": True,
            "model_type": "pinhole",
            "board": jr.board_type,
            "cameras": [int(c) for c in jr.cameras],
            "per_camera_rms": {str(c): float(r) for c, r in jr.per_camera_rms.items()},
            "rms_px": float(jr.rms_px),
            "spacing_mm": float(jr.spacing_mm),
            "board_release": jr.board_release,
            "converged": bool(meta.get("converged", 0)),
            "cross_camera_board_agreement_mm": float(
                meta.get("cross_camera_board_agreement_mm", 0.0)
            ),
            "n_board_dots": int(meta.get("n_board_dots", len(jr.board))),
            "image_sizes": {
                str(c): [
                    int(jr.models[c].image_size[0]),
                    int(jr.models[c].image_size[1]),
                ]
                for c in jr.cameras
            },
            "per_camera": per_camera,
            "baselines_mm": baselines_mm,
            "geometry": _geometry_payload(meta),
        }
    )


# ---------------------------------------------------------------------------
# Stereo self-calibration (Wieneke disparity minimisation)
#
# A post-model step: it loads the saved stereo model + recorded PIV particle frames
# from a base_path dataset, recovers the laser sheet (z_offset, tilt_x, tilt_y), and
# writes the result INTO the stereo record (so apply picks it up automatically) plus
# the six diagnostic figures into the calibration source folder.
# ---------------------------------------------------------------------------


def _stereo_locators(get: Callable[[str], Any]):
    """(source, cam1, cam2, model_dir, self_cal_figdir) from request locators."""
    source = _source_path(_source_idx(get))
    pair = get("camera_pair") or _rig(get).get("camera_pair") or [1, 2]
    if isinstance(pair, str):
        pair = [int(x) for x in pair.split(",")]
    cam1, cam2 = int(pair[0]), int(pair[1])
    model_dir = rec.stereo_model_dir_for_source(source, cam1, cam2)
    figdir = model_dir.parent / "figures" / "self_cal"
    return source, cam1, cam2, model_dir, figdir


def _self_cal_payload(record) -> dict:
    """The saved self_cal block, shaped for the GUI (radians + degrees).

    For a baked record the applied z_offset/tilt are zero (the correction lives in the
    extrinsics); the recovered sheet is surfaced from the fitted_* provenance fields so
    the GUI still shows what self-cal found. Legacy (pre-bake) records fall back to the
    applied fields.
    """
    import math as _m

    sc = record.self_cal or {}
    if not sc:
        return {"has_self_calibration": False}
    z = float(sc.get("fitted_z_offset", sc.get("z_offset", 0.0)))
    tx = float(sc.get("fitted_tilt_x", sc.get("tilt_x", 0.0)))
    ty = float(sc.get("fitted_tilt_y", sc.get("tilt_y", 0.0)))
    # Non-finite floats (a diverged self-cal) would make jsonify emit bare NaN/Infinity —
    # invalid JSON that the frontend poll silently rejects (a hang). Send null instead.
    return {
        "has_self_calibration": True,
        "baked": bool(int(sc.get("baked", 0))),
        "z_offset": _finite_or_none(z),
        "tilt_x": _finite_or_none(tx),
        "tilt_y": _finite_or_none(ty),
        "tilt_x_deg": _finite_or_none(_m.degrees(tx)),
        "tilt_y_deg": _finite_or_none(_m.degrees(ty)),
        "converged": bool(int(sc.get("converged", 0))),
        "final_rms_disparity": _finite_or_none(sc.get("final_rms_disparity", 0.0)),
        "n_iterations": int(sc.get("n_iterations", 0)),
        "n_images": int(sc.get("n_images", 0)),
        "window_size": int(sc.get("window_size", 0)),
        "overlap": _finite_or_none(sc.get("overlap", 0.0)),
        "source": str(sc.get("source", "auto")),
    }


@calibration_bp.route("/calibration/self_cal/result", methods=["GET"])
def self_cal_result():
    """Current self-calibration block + saved figure list (restore on model load)."""
    try:
        _, _, _, model_dir, figdir = _stereo_locators(request.args.get)
        board = request.args.get("board") or _cfg().get("active", "charuco")
        record = rec.load_stereo(
            model_dir, model_type=_model_type_arg(request.args.get, board, stereo=True)
        )
    except (FileNotFoundError, ValueError, IndexError):
        return jsonify({"exists": False, "has_self_calibration": False}), 200
    payload = _self_cal_payload(record)
    payload["exists"] = True
    payload["figures"] = _list_figures(figdir)
    return jsonify(payload)


@calibration_bp.route("/calibration/self_cal/run", methods=["POST"])
def self_cal_run():
    """Run iterative self-calibration as a background job; write the result into the record."""
    data = request.get_json() or {}
    full_cfg = get_config()
    try:
        _, cam1, cam2, model_dir, figdir = _stereo_locators(data.get)
        board = data.get("board") or _cfg().get("active", "charuco")
        record = rec.load_stereo(
            model_dir, model_type=_model_type_arg(data.get, board, stereo=True)
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"no stereo model — generate it first ({exc})",
                }
            ),
            200,
        )

    # Self-cal nudges the rigid pose, so it needs pinhole extrinsics. A polynomial pair
    # has no R/t to rebake; reject it here, before the expensive correlation loop, rather
    # than failing late inside rebake_record.
    if not (hasattr(record.model1, "R") and hasattr(record.model1, "t")):
        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        "self-calibration needs a pinhole stereo model (it corrects the "
                        f"extrinsics); this model is {type(record.model1).__name__} with "
                        "no pose to rebake — regenerate as pinhole, or skip self-cal for "
                        "the polynomial fit"
                    ),
                }
            ),
            200,
        )

    base_idx = int(data.get("base_path_idx", 0))
    n_images = int(data.get("n_images", 20))
    window_size = int(data.get("window_size", 64))
    overlap = float(data.get("overlap", 50.0))
    apply_filters = bool(data.get("apply_filters", True))

    job_id = job_manager.create_job("calibration_self_cal", out_dir=str(figdir))

    def _run():
        import math as _m

        try:
            job_manager.update_job(
                job_id, status="running", progress=10, stage="loading_images"
            )
            imgs1, imgs2 = c2sc.load_particle_pairs(
                full_cfg, base_idx, cam1, cam2, n_images, apply_filters
            )
            job_manager.update_job(job_id, progress=25, stage="self_calibrating")
            result = c2sc.run(
                record,
                imgs1,
                imgs2,
                window_size=window_size,
                overlap=overlap,
                figure_dir=figdir,
            )
            c2sc.rebake_record(record, result.z_offset, result.tilt_x, result.tilt_y)
            record.self_cal = c2sc.baked_block(
                result, n_images=len(imgs1), window_size=window_size, overlap=overlap
            )
            rec.save_stereo(record, model_dir)
            # Guard every float against NaN/Inf (a diverged fit) so the jsonify'd status
            # payload stays valid JSON and the poll never hangs — null surfaces instead.
            history = [
                dict(
                    iteration=h.iteration,
                    rms_disparity=_finite_or_none(h.rms_disparity),
                    delta_z=_finite_or_none(h.delta_z),
                    delta_tilt_x=_finite_or_none(h.delta_tilt_x),
                    delta_tilt_y=_finite_or_none(h.delta_tilt_y),
                    cumulative_z=_finite_or_none(h.cumulative_z),
                    cumulative_tilt_x=_finite_or_none(h.cumulative_tilt_x),
                    cumulative_tilt_y=_finite_or_none(h.cumulative_tilt_y),
                )
                for h in result.history
            ]
            job_manager.complete_job(
                job_id,
                converged=bool(result.converged),
                n_iterations=int(result.n_iterations),
                z_offset=_finite_or_none(result.z_offset),
                tilt_x=_finite_or_none(result.tilt_x),
                tilt_y=_finite_or_none(result.tilt_y),
                tilt_x_deg=_finite_or_none(_m.degrees(result.tilt_x)),
                tilt_y_deg=_finite_or_none(_m.degrees(result.tilt_y)),
                final_rms_disparity=_finite_or_none(result.final_rms_disparity),
                history=history,
                figures=_list_figures(figdir),
            )
        except Exception as exc:
            logger.exception("self-cal job %s failed", job_id)
            job_manager.fail_job(job_id, str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"success": True, "job_id": job_id})


@calibration_bp.route("/calibration/self_cal/status/<job_id>", methods=["GET"])
def self_cal_status(job_id):
    """Poll a self-calibration job (convergence history + figure list on completion)."""
    return _job_status_response(job_id)


@calibration_bp.route("/calibration/self_cal/figure", methods=["GET"])
def self_cal_figure():
    """Serve one self-cal diagnostic PNG from the model's figures/self_cal folder."""
    name = request.args.get("name", "")
    if not name or Path(name).name != name:
        return jsonify({"error": "invalid figure name"}), 400
    try:
        _, _, _, _, figdir = _stereo_locators(request.args.get)
    except (ValueError, IndexError):
        return jsonify({"error": "source not configured"}), 404
    figdir_r = Path(figdir).resolve()
    path = (figdir_r / name).resolve()
    if path.parent != figdir_r or not path.is_file():
        return jsonify({"error": "figure not found"}), 404
    return Response(path.read_bytes(), mimetype="image/png")
