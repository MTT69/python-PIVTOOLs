"""calibration2 Flask blueprint — backend for the unified Calibration GUI.

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

Routes (prefix ``/calibration2`` under the app's ``/backend``):
- POST /calibration2/validate        -> validate the image source (found-N, preview, suggested pattern)
- POST /calibration2/detect_datum    -> detect the datum view, cache it, return dot pixels
- GET  /calibration2/datum_image     -> datum-view PNG for the click overlay
- GET  /calibration2/frame_image     -> any frame as PNG (frame navigation)
- POST /calibration2/detect_frame    -> detect an arbitrary frame + report frame count
- POST /calibration2/snap_fiducial   -> snap a click to the nearest detected dot
- POST /calibration2/generate_model  -> run the full calibration (mono/stereo), save + figures
- GET  /calibration2/model           -> summary of a saved model
- POST /calibration2/measure         -> distance in mm between two pixels (back-projection)
- GET  /calibration2/figures|figure  -> list / serve the proof figures for a model
- POST /calibration2/global/compute  -> planar N-camera datum-chain shifts
- POST /calibration2/apply (+ /apply/status/<id>) -> apply the model to PIV output (job)
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional

import cv2
import numpy as np
from flask import Blueprint, jsonify, request, Response

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths, vector_glob_from_format
from pivtools_core.image_handling.calibration_loader import (
    get_calibration_frame_count,
    read_calibration_image,
    validate_calibration_images,
)
from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.utils import (
    camera_number,
    get_display_contrast_stats,
    numpy_to_base64,
)
from pivtools_gui.calibration2 import apply as c2apply
from pivtools_gui.calibration2 import global_coords as gc2
from pivtools_gui.calibration2 import record as rec
from pivtools_gui.calibration2 import runio as c2runio
from pivtools_gui.calibration2 import world_frame as WF
from pivtools_gui.calibration2.camera_model import DistortionModel
from pivtools_gui.calibration2.pipeline import Calibrator
from pivtools_gui.calibration2.stereo_model import StereoCalibrator
from pivtools_cli import calibration2_cli as c2

calibration2_bp = Blueprint("calibration2", __name__)
logger = logging.getLogger(__name__)

# Datum-view detections cached for the snap workflow, keyed by camera number.
_datum_cache = {}
_datum_lock = threading.Lock()

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


def _resolve_board(get: Callable[[str], Any], overrides: Optional[dict] = None):
    """Resolve (cfg, board, params, detector). Board params may be overridden per-call."""
    cfg = _cfg()
    board = get("board") or cfg.get("active", "charuco")
    params = c2._board_params(cfg, board, overrides)
    detector = c2._build_detector(board, params)
    return cfg, board, params, detector


def _load_one(camera: int, frame: int, source_idx: int,
              image_format: Optional[str], image_type: Optional[str]) -> np.ndarray:
    """Load one calibration frame (any format) via the app-wide reader."""
    return read_calibration_image(
        int(frame), int(camera), get_config(), int(source_idx),
        image_format=image_format, image_type=image_type)


def _load_views(camera: int, frame_total: int, source_idx: int,
                image_format: Optional[str], image_type: Optional[str]) -> List[np.ndarray]:
    return [_load_one(camera, k + 1, source_idx, image_format, image_type)
            for k in range(int(frame_total))]


def _frame_total(get: Callable[[str], Any], camera: int, source_idx: int) -> int:
    """Resolve frame total: request value, else config num_images, else auto-detect."""
    v = get("frame_total")
    if v not in (None, ""):
        return int(v)
    n = int(_cfg().get("num_images") or 0)
    if n > 0:
        return n
    try:
        return get_calibration_frame_count(int(camera), get_config(), int(source_idx))
    except Exception:
        return 0


def _datum_frame(get: Callable[[str], Any]) -> int:
    v = get("datum_frame")
    return int(v) if v not in (None, "") else int(_cfg().get("datum_frame", 1))


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


def _list_figures(fig_dir) -> List[str]:
    if not fig_dir or not Path(fig_dir).is_dir():
        return []
    # Skip macOS AppleDouble companions (._*) that appear when writing to non-HFS drives.
    return sorted(p.name for p in Path(fig_dir).glob("*.png") if not p.name.startswith("._"))


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


def _intrinsics(cm) -> dict:
    """Pinhole intrinsics of a CameraModel, shaped for the GUI results card."""
    return {
        "fx": float(cm.K[0, 0]), "fy": float(cm.K[1, 1]),
        "cx": float(cm.K[0, 2]), "cy": float(cm.K[1, 2]),
        "camera_matrix": cm.K.tolist(), "dist_coeffs": cm.dist.tolist(),
        "rms": float(cm.rms),
        "image_width": int(cm.image_size[0]), "image_height": int(cm.image_size[1]),
    }


def _figures_dir(get: Callable[[str], Any]) -> Path:
    """Resolve the figures dir for a model from request locators (mono or stereo)."""
    cfg = _cfg()
    board = get("board") or cfg.get("active", "charuco")
    source = _source_path(_source_idx(get))
    if str(get("stereo")) in ("1", "true", "True"):
        pair = get("camera_pair") or cfg.get("camera_pair", [1, 2])
        if isinstance(pair, str):
            pair = [int(x) for x in pair.split(",")]
        return rec.stereo_model_dir_for_source(source, int(pair[0]), int(pair[1])).parent / "figures"
    camera = int(get("camera") or cfg.get("camera", 1))
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

@calibration2_bp.route("/calibration2/validate", methods=["POST"])
def validate():
    """Validate the calibration image source (all formats): found-count, preview, suggestion."""
    data = request.get_json() or {}
    camera = int(data.get("camera", _cfg().get("camera", 1)))
    source_idx = _source_idx(data.get)
    try:
        result = validate_calibration_images(
            camera, get_config(), source_idx,
            image_format=data.get("image_format"),
            num_images=data.get("frame_total"),
            image_type=data.get("image_type"))
    except Exception as exc:
        logger.exception("calibration2 validate failed")
        return jsonify({"valid": False, "error": str(exc)}), 200
    return jsonify(result)


# ---------------------------------------------------------------------------
# Detection + frame navigation
# ---------------------------------------------------------------------------

@calibration2_bp.route("/calibration2/detect_datum", methods=["POST"])
def detect_datum():
    """Detect the datum view for a camera; cache it and return dot pixels for clicking."""
    data = request.get_json() or {}
    cfg, board, params, detector = _resolve_board(data.get, data.get("board_params"))
    camera = int(data.get("camera", cfg.get("camera", 1)))
    source_idx = _source_idx(data.get)
    frame = _datum_frame(data.get)
    try:
        img = _load_one(camera, frame, source_idx, data.get("image_format"), data.get("image_type"))
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    det = detector.detect(img)
    if not det.success:
        return jsonify({"success": False, "error": "board not detected", "diagnostics": det.diagnostics}), 200

    with _datum_lock:
        _datum_cache[camera] = det
    h, w = np.asarray(img).shape[:2]
    return jsonify({
        "success": True, "camera": camera, "board": board, "frame": int(frame),
        "n_points": det.n, "width": int(w), "height": int(h),
        "image_points": det.image_points.tolist(),
        "grid_indices": det.grid_indices.tolist(),
    })


@calibration2_bp.route("/calibration2/datum_image", methods=["GET"])
def datum_image():
    """Return the datum-view image (PNG) for a camera, for the click overlay."""
    cfg = _cfg()
    camera = int(request.args.get("camera", cfg.get("camera", 1)))
    source_idx = _source_idx(request.args.get)
    frame = _datum_frame(request.args.get)
    try:
        img = _load_one(camera, frame, source_idx,
                        request.args.get("image_format"), request.args.get("image_type"))
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"error": str(exc)}), 404
    return _png_response(img)


@calibration2_bp.route("/calibration2/frame_image", methods=["GET"])
def frame_image():
    """Serve any calibration frame (1-based image index) as a PNG."""
    cfg = _cfg()
    camera = int(request.args.get("camera", cfg.get("camera", 1)))
    source_idx = _source_idx(request.args.get)
    frame = int(request.args.get("frame", 1))
    try:
        img = _load_one(camera, frame, source_idx,
                        request.args.get("image_format"), request.args.get("image_type"))
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"error": str(exc)}), 404
    return _png_response(img)


@calibration2_bp.route("/calibration2/frame", methods=["GET"])
def frame_json():
    """Serve a calibration frame as JSON: ``{image, stats, width, height, frame_count}``.

    Matches the v1 ``/calibration/get_frame`` shape so the original
    ``CalibrationImageViewer`` (contrast sliders + colormap + prefetch) works
    unchanged on the calibration2 backend. All formats via the app-wide reader;
    ``stats.vmin_pct``/``vmax_pct`` drive the auto-contrast window.
    """
    cfg = _cfg()
    camera = camera_number(request.args.get("camera", default=1, type=int))
    idx = request.args.get("idx", default=1, type=int)
    source_idx = _source_idx(request.args.get)
    output_format = (request.args.get("format", default="jpeg", type=str) or "jpeg").lower()
    quality = request.args.get("quality", default=85, type=int)
    try:
        frame_count = get_calibration_frame_count(camera, get_config(), source_idx)
    except Exception:
        frame_count = 0
    if frame_count > 0 and idx > frame_count:
        return jsonify({
            "error": f"Frame index {idx} exceeds available frames ({frame_count})",
            "frame_count": frame_count, "requested_idx": idx,
        }), 400
    try:
        img = _load_one(camera, idx, source_idx,
                        request.args.get("image_format"), request.args.get("image_type"))
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"error": str(exc), "frame_count": frame_count}), 404

    arr = np.asarray(img)
    arr_f = arr.astype(np.float32)
    stats = {
        "min": float(arr_f.min()), "max": float(arr_f.max()),
        "mean": float(arr_f.mean()), "dtype": str(arr.dtype),
    }
    contrast = get_display_contrast_stats(arr)
    stats["vmin_pct"] = contrast["vmin_pct"]
    stats["vmax_pct"] = contrast["vmax_pct"]
    return jsonify({
        "image": numpy_to_base64(arr, format=output_format, jpeg_quality=quality),
        "mime_type": f"image/{output_format}",
        "width": int(arr.shape[1]), "height": int(arr.shape[0]),
        "stats": stats, "frame_count": frame_count, "current_idx": idx,
    })


@calibration2_bp.route("/calibration2/detect_frame", methods=["POST"])
def detect_frame():
    """Detect one arbitrary frame (browsing/overlay) and report the frame count.

    Does NOT replace the datum cache used by ``snap_fiducial`` — world-frame fiducials
    are picked on the datum frame; this route is for inspection.
    """
    data = request.get_json() or {}
    cfg, board, params, detector = _resolve_board(data.get, data.get("board_params"))
    camera = int(data.get("camera", cfg.get("camera", 1)))
    source_idx = _source_idx(data.get)
    frame = int(data.get("frame", 1))
    try:
        frame_count = get_calibration_frame_count(camera, get_config(), source_idx)
    except Exception:
        frame_count = 0
    try:
        img = _load_one(camera, frame, source_idx, data.get("image_format"), data.get("image_type"))
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"success": False, "error": str(exc), "frame_count": frame_count}), 200
    det = detector.detect(img)
    h, w = np.asarray(img).shape[:2]
    if not det.success:
        return jsonify({"success": False, "error": "board not detected", "frame": int(frame),
                        "frame_count": frame_count, "width": int(w), "height": int(h),
                        "diagnostics": det.diagnostics}), 200
    return jsonify({
        "success": True, "camera": camera, "board": board, "frame": int(frame),
        "frame_count": frame_count, "n_points": det.n, "width": int(w), "height": int(h),
        "image_points": det.image_points.tolist(),
        "grid_indices": det.grid_indices.tolist(),
    })


@calibration2_bp.route("/calibration2/snap_fiducial", methods=["POST"])
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
    return jsonify({
        "snapped_x": float(px[0]), "snapped_y": float(px[1]),
        "grid_col": int(gi[0]), "grid_row": int(gi[1]),
    })


# ---------------------------------------------------------------------------
# Model generation
# ---------------------------------------------------------------------------

@calibration2_bp.route("/calibration2/generate_model", methods=["POST"])
def generate_model():
    """Run the full calibration (mono or stereo), save it + proof figures into the source."""
    data = request.get_json() or {}
    cfg, board, params, detector = _resolve_board(data.get, data.get("board_params"))
    source_idx = _source_idx(data.get)
    image_format, image_type = data.get("image_format"), data.get("image_type")
    stereo = bool(data.get("stereo", False))
    make_figs = not bool(data.get("no_figures", False))
    spacing = c2._spacing_mm(board, params)
    datum_frame = _datum_frame(data.get)
    datum_index = datum_frame - 1  # position within the loaded views (frames 1..N)

    try:
        source = _source_path(source_idx)
        if stereo:
            pair = data.get("camera_pair") or cfg.get("camera_pair", [1, 2])
            cam1, cam2 = int(pair[0]), int(pair[1])
            frame_total = _frame_total(data.get, cam1, source_idx)
            if not (0 <= datum_index < frame_total):
                return jsonify({"success": False,
                                "error": f"datum frame {datum_frame} is outside 1..{frame_total}"}), 200
            imgs1 = _load_views(cam1, frame_total, source_idx, image_format, image_type)
            imgs2 = _load_views(cam2, frame_total, source_idx, image_format, image_type)
            model_dir = rec.stereo_model_dir_for_source(source, cam1, cam2)
            fig_dir = (model_dir.parent / "figures") if make_figs else None
            sc = StereoCalibrator(detector=detector, board_type=board, distortion_model=_MODEL)
            record = sc.run_stereo(
                imgs1, imgs2, cam1=cam1, cam2=cam2,
                clicks=_clicks_from(data.get("clicks")),
                clicks2=_clicks_from(data.get("clicks2")),
                origin_mm=_origin_mm_from(data.get("clicks")),
                datum_index=datum_index, spacing_mm=spacing, figure_dir=fig_dir)
            path = rec.save_stereo(record, model_dir)
            ang = float(np.degrees(np.arccos(np.clip((np.trace(record.R_stereo) - 1) / 2, -1, 1))))
            return jsonify({
                "success": True, "stereo": True, "model_path": str(path),
                "rms_cam1": record.model1.rms, "rms_cam2": record.model2.rms,
                "per_view_rms1": list(record.per_view_rms1),
                "per_view_rms2": list(record.per_view_rms2),
                "intrinsics1": _intrinsics(record.model1),
                "intrinsics2": _intrinsics(record.model2),
                "num_pairs_used": len(record.per_view_rms1),
                "stereo_angle_deg": ang, "baseline_mm": float(np.linalg.norm(record.T_stereo)),
                "figures": _list_figures(fig_dir),
            })
        else:
            camera = int(data.get("camera", cfg.get("camera", 1)))
            frame_total = _frame_total(data.get, camera, source_idx)
            if not (0 <= datum_index < frame_total):
                return jsonify({"success": False,
                                "error": f"datum frame {datum_frame} is outside 1..{frame_total}"}), 200
            imgs = _load_views(camera, frame_total, source_idx, image_format, image_type)
            model_dir = rec.mono_model_dir_for_source(source, camera, board)
            fig_dir = (model_dir.parent / "figures") if make_figs else None
            calr = Calibrator(detector=detector, board_type=board, distortion_model=_MODEL)
            record = calr.run_mono(
                imgs, camera=camera, clicks=_clicks_from(data.get("clicks")),
                origin_mm=_origin_mm_from(data.get("clicks")),
                datum_index=datum_index, spacing_mm=spacing, figure_dir=fig_dir)
            path = rec.save_mono(record, model_dir)
            cm = record.camera_model
            return jsonify({
                "success": True, "stereo": False, "model_path": str(path),
                "camera": camera, "rms": cm.rms,
                "fx": float(cm.K[0, 0]), "fy": float(cm.K[1, 1]),
                "cx": float(cm.K[0, 2]), "cy": float(cm.K[1, 2]),
                "camera_matrix": cm.K.tolist(), "dist_coeffs": cm.dist.tolist(),
                "image_width": int(cm.image_size[0]), "image_height": int(cm.image_size[1]),
                "num_images_used": len(record.per_view_rms),
                "per_view_rms": list(record.per_view_rms),
                "figures": _list_figures(fig_dir),
            })
    except Exception as exc:  # surface failures to the GUI (full traceback to the server log)
        logger.exception("generate_model failed")
        return jsonify({"success": False, "error": str(exc)}), 200


@calibration2_bp.route("/calibration2/model", methods=["GET"])
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
            r = rec.load_stereo(mdir)
        except FileNotFoundError:
            return jsonify({"exists": False}), 200
        return jsonify({
            "exists": True, "stereo": True, "cam1": cam1, "cam2": cam2,
            "model_path": str(mdir / "stereo_model.mat"),
            "rms_cam1": r.model1.rms, "rms_cam2": r.model2.rms,
            "per_view_rms1": list(r.per_view_rms1), "per_view_rms2": list(r.per_view_rms2),
            "intrinsics1": _intrinsics(r.model1),
            "intrinsics2": _intrinsics(r.model2),
            "num_pairs_used": len(r.per_view_rms1),
            "distortion_model": r.model1.distortion_model.value,
            "world_frame_mode": r.world_frame.mode,
            "world_frame": _world_frame_payload(r.world_frame),
            "image_width": int(r.model1.image_size[0]), "image_height": int(r.model1.image_size[1]),
            "spacing_mm": _meta_float(r.board_meta, "spacing_mm"),
            "n_views": _meta_int(r.board_meta, "n_views"),
            "stereo_angle_deg": float(np.degrees(np.arccos(np.clip((np.trace(r.R_stereo) - 1) / 2, -1, 1)))),
            "baseline_mm": float(np.linalg.norm(r.T_stereo)),
        })
    camera = int(request.args.get("camera", 1))
    mdir = rec.mono_model_dir_for_source(source, camera, board)
    try:
        r = rec.load_mono(mdir)
    except FileNotFoundError:
        return jsonify({"exists": False}), 200
    cm = r.camera_model
    return jsonify({
        "exists": True, "stereo": False, "camera": camera,
        "model_path": str(mdir / "model.mat"),
        "rms": cm.rms,
        "fx": float(cm.K[0, 0]), "fy": float(cm.K[1, 1]),
        "cx": float(cm.K[0, 2]), "cy": float(cm.K[1, 2]),
        "camera_matrix": cm.K.tolist(), "dist_coeffs": cm.dist.tolist(),
        "num_images_used": len(r.per_view_rms),
        "per_view_rms": list(r.per_view_rms),
        "distortion_model": cm.distortion_model.value,
        "world_frame_mode": r.world_frame.mode,
        "world_frame": _world_frame_payload(r.world_frame),
        "image_width": int(cm.image_size[0]), "image_height": int(cm.image_size[1]),
        "spacing_mm": _meta_float(r.board_meta, "spacing_mm"),
        "n_views": _meta_int(r.board_meta, "n_views"),
    })


# ---------------------------------------------------------------------------
# Measure tool + figure serving
# ---------------------------------------------------------------------------

@calibration2_bp.route("/calibration2/measure", methods=["POST"])
def measure():
    """Distance in mm between two pixels, via the saved model's back-projection."""
    data = request.get_json() or {}
    cfg = _cfg()
    board = data.get("board") or cfg.get("active", "charuco")
    p1 = np.asarray(data.get("p1"), dtype=float)
    p2 = np.asarray(data.get("p2"), dtype=float)
    if p1.shape != (2,) or p2.shape != (2,):
        return jsonify({"error": "p1 and p2 must be [x, y] pixel pairs"}), 400
    z = float(data.get("z_world", cfg.get("z_world", 0.0)))
    tx = float(data.get("tilt_x", cfg.get("tilt_x", 0.0)))
    ty = float(data.get("tilt_y", cfg.get("tilt_y", 0.0)))
    try:
        source = _source_path(_source_idx(data.get))
        if bool(data.get("stereo", False)):
            pair = data.get("camera_pair") or cfg.get("camera_pair", [1, 2])
            cam1, cam2 = int(pair[0]), int(pair[1])
            model = rec.load_stereo(rec.stereo_model_dir_for_source(source, cam1, cam2)).model1
        else:
            camera = int(data.get("camera", cfg.get("camera", 1)))
            model = rec.load_mono(rec.mono_model_dir_for_source(source, camera, board)).camera_model
    except (FileNotFoundError, ValueError, IndexError):
        return jsonify({"error": "no saved model — generate the model first to measure in mm"}), 200
    world = c2apply.calibrate_coordinates(model, np.vstack([p1, p2]), z, tx, ty)
    if not np.all(np.isfinite(world)):
        return jsonify({"error": "back-projection failed (ray missed the sheet plane)"}), 200
    return jsonify({
        "distance_mm": float(np.linalg.norm(world[1] - world[0])),
        "distance_px": float(np.linalg.norm(p2 - p1)),
        "world_p1": [float(world[0, 0]), float(world[0, 1])],
        "world_p2": [float(world[1, 0]), float(world[1, 1])],
    })


@calibration2_bp.route("/calibration2/figures", methods=["GET"])
def list_figures():
    """List the proof figures written beside a model."""
    try:
        fig_dir = _figures_dir(request.args.get)
    except (ValueError, IndexError):
        return jsonify({"exists": False, "figures": []}), 200
    figs = _list_figures(fig_dir)
    return jsonify({"exists": bool(figs), "dir": str(fig_dir), "figures": figs})


@calibration2_bp.route("/calibration2/figure", methods=["GET"])
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

@calibration2_bp.route("/calibration2/global/compute", methods=["POST"])
def global_compute():
    """Compute per-camera world shifts from a datum + overlap pairs (planar datum chain).

    Pure compute: datum + pairs are owned by the frontend config and passed in the
    request; nothing is persisted here. Each camera's saved mono model is loaded to
    back-project the clicked pixels to physical mm.
    """
    data = request.get_json() or {}
    cfg = _cfg()
    board = data.get("board") or cfg.get("active", "charuco")
    datum_camera = int(data.get("datum_camera", 1))
    datum_pixel = data.get("datum_pixel")
    datum_physical = data.get("datum_physical", [0.0, 0.0])
    overlap_pairs = data.get("overlap_pairs") or []
    z = float(data.get("z_world", cfg.get("z_world", 0.0)))
    tx = float(data.get("tilt_x", cfg.get("tilt_x", 0.0)))
    ty = float(data.get("tilt_y", cfg.get("tilt_y", 0.0)))
    if not datum_pixel:
        return jsonify({"error": "datum_pixel not set — click a point on the datum camera"}), 200

    cams = {datum_camera}
    for p in overlap_pairs:
        cams.add(int(p["camera_a"]))
        cams.add(int(p["camera_b"]))
    try:
        source = _source_path(_source_idx(data.get))
        records = {cam: rec.load_mono(rec.mono_model_dir_for_source(source, cam, board)) for cam in cams}
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"error": f"missing mono model — calibrate each camera first ({exc})"}), 200

    try:
        shifts = gc2.compute_camera_shifts(
            records, datum_camera, datum_pixel, datum_physical, overlap_pairs, z, tx, ty)
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 200

    # Stringify camera keys (Flask jsonify mixes int/str keys badly).
    return jsonify({
        "camera_shifts": {str(c): [float(s[0]), float(s[1])] for c, s in shifts.items()},
        "datum_physical": [float(datum_physical[0]), float(datum_physical[1])],
    })


# ---------------------------------------------------------------------------
# Apply the model to PIV output (background job)
# ---------------------------------------------------------------------------

def _apply_units(data, full_cfg, source, board, stereo, type_name):
    """Resolve every (base_path x camera/pair) apply unit, deriving I/O dirs from config.

    The model is loaded from the shared calibration ``source``; the PIV data comes from
    each selected base path. ``active_paths`` indexes ``config.base_paths`` (default: all
    configured base paths). Explicit ``uncalibrated_dir``/``calibrated_dir`` (single unit)
    still override the derivation for ad-hoc runs. Each unit carries a loaded model record
    so a missing model fails loudly here, before the job thread starts.
    """
    get = data.get
    nfp = full_cfg.num_frame_pairs
    base_paths = full_cfg.base_paths
    active = get("active_paths")
    base_indices = [int(i) for i in active] if active else list(range(len(base_paths)))

    units = []
    if stereo:
        pair = get("camera_pair") or _cfg().get("camera_pair", [1, 2])
        cam1, cam2 = int(pair[0]), int(pair[1])
        record = rec.load_stereo(rec.stereo_model_dir_for_source(source, cam1, cam2))
        if get("calibrated_dir") and get("uncalibrated_dir_cam1"):
            return [dict(stereo=True, record=record,
                         uncal1=Path(data["uncalibrated_dir_cam1"]),
                         uncal2=Path(data["uncalibrated_dir_cam2"]),
                         out=Path(data["calibrated_dir"]), label="manual")]
        for bi in base_indices:
            base = base_paths[bi]
            uncal1 = get_data_paths(base, nfp, cam1, type_name, use_uncalibrated=True)["data_dir"]
            uncal2 = get_data_paths(base, nfp, cam2, type_name, use_uncalibrated=True)["data_dir"]
            out = get_data_paths(base, nfp, cam1, type_name, use_stereo=True,
                                 stereo_camera_pair=(cam1, cam2))["data_dir"]
            units.append(dict(stereo=True, record=record, uncal1=uncal1, uncal2=uncal2,
                              out=out, label=f"{base.name}/Cam{cam1}_Cam{cam2}"))
    else:
        if get("calibrated_dir") and get("uncalibrated_dir"):
            camera = int(get("camera") or full_cfg.camera_numbers[0])
            record = rec.load_mono(rec.mono_model_dir_for_source(source, camera, board))
            return [dict(stereo=False, record=record, uncal=Path(data["uncalibrated_dir"]),
                         out=Path(data["calibrated_dir"]), label="manual")]
        for bi in base_indices:
            base = base_paths[bi]
            for cam in full_cfg.camera_numbers:
                record = rec.load_mono(rec.mono_model_dir_for_source(source, cam, board))
                uncal = get_data_paths(base, nfp, cam, type_name, use_uncalibrated=True)["data_dir"]
                out = get_data_paths(base, nfp, cam, type_name)["data_dir"]
                units.append(dict(stereo=False, record=record, uncal=uncal, out=out,
                                  label=f"{base.name}/Cam{cam}"))
    return units


@calibration2_bp.route("/calibration2/apply", methods=["POST"])
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
    dt = float(data.get("dt", cfg.get("dt", 1.0)))
    z = float(data.get("z_world", cfg.get("z_world", 0.0)))
    tx = float(data.get("tilt_x", cfg.get("tilt_x", 0.0)))
    ty = float(data.get("tilt_y", cfg.get("tilt_y", 0.0)))
    # Derive the per-frame vector glob from the PIV vector_format (e.g. "%05d.mat" ->
    # "*.mat"); the old hardcoded "B*.mat" matched nothing under the default naming, so
    # apply wrote coordinates.mat but no calibrated vectors.
    vector_glob = data.get("vector_glob") or vector_glob_from_format(full_cfg.vector_format)
    type_name = data.get("type_name") or "instantaneous"

    try:
        source = _source_path(_source_idx(data.get))
        units = _apply_units(data, full_cfg, source, board, stereo, type_name)
    except (FileNotFoundError, ValueError, IndexError) as exc:
        return jsonify({"success": False, "error": f"model not found — generate it first ({exc})"}), 200
    except KeyError as exc:
        return jsonify({"success": False, "error": f"missing directory field {exc}"}), 200

    if not units:
        return jsonify({"success": False, "error": "no datasets selected — configure base paths first"}), 200

    n_units = len(units)
    out_dirs = [str(u["out"]) for u in units]
    job_id = job_manager.create_job("calibration2_apply", stereo=stereo, out_dir=out_dirs[0])

    def _run():
        try:
            job_manager.update_job(job_id, status="running", progress=0, processed=0, total=n_units)
            total_written = 0
            for i, u in enumerate(units):
                def cb(done, total, _i=i):
                    pct = int(((_i + done / max(total, 1)) / n_units) * 100)
                    job_manager.update_job(job_id, progress=pct, processed=_i, total=n_units)

                if u["stereo"]:
                    written = c2runio.reconstruct_stereo_run(
                        u["record"], u["uncal1"], u["uncal2"], u["out"], dt, None, z, tx, ty,
                        progress_cb=cb, vector_glob=vector_glob)
                else:
                    written = c2runio.calibrate_mono_run(
                        u["record"], u["uncal"], u["out"], dt, z, tx, ty,
                        progress_cb=cb, vector_glob=vector_glob)
                total_written += len(written)
                job_manager.update_job(job_id, processed=i + 1, total=n_units)
            job_manager.complete_job(job_id, n_frames=total_written, out_dir="; ".join(out_dirs))
        except Exception as exc:  # surface in the job status (full traceback to the server log)
            logger.exception("apply job %s failed", job_id)
            job_manager.fail_job(job_id, str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"success": True, "job_id": job_id})


@calibration2_bp.route("/calibration2/apply/status/<job_id>", methods=["GET"])
def apply_status(job_id):
    """Poll an apply job's status (same shape as the v1 vector-job status)."""
    data = job_manager.get_job_with_timing(job_id)
    if data is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(data)
