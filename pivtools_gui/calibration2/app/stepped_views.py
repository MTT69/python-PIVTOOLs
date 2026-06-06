"""calibration2 stepped-board Flask blueprint — the GUI API for the dual-level board.

The stepped flow is the one calibration2 path that is genuinely STATEFUL across
requests: the operator detects a multi-pose sequence, then labels each pose's two
levels (peak/trough) and picks the datum fiducials, and only THEN is the model fit.
The heavy per-level grid dicts the stepped calibrator consumes (homographies, BFS
vectors — stashed under ``_``-prefixed diagnostics keys) must survive between the
detect phase and the generate phase without round-tripping through the browser. So,
exactly like v1's ``stepped_planar``/``stepped_board`` blueprints, this keeps a
TTL'd server-side **sequence cache** keyed by an opaque ``sequence_id`` and never
serialises the ``_``-prefixed keys.

Everything that is NOT stepped-specific — image source validation, frame serving,
model load, the measure tool, figure serving, and apply — is the SAME board-agnostic
``/calibration2/*`` route in ``views.py`` driven with ``board="stepped"``; only the
sequence detect + per-pose level labelling + the stepped fit live here.

Routes (prefix ``/calibration2/stepped`` under the app's ``/backend``):
- POST /detect_sequence              -> detect N poses for 1 (mono) or 2 (stereo) cameras (job)
- GET  /detect_sequence/status/<id>  -> poll the detection job
- GET  /sequence_pose_detection      -> one pose's JSON-safe detection (overlay + click-to-label)
- POST /identify_pose_level          -> snap a click + report which level (A/B) it landed on
- POST /snap_fiducial                -> snap a fiducial click against a camera's datum pose
- POST /generate_model               -> run the stepped fit (mono or stereo), save + figures (job)
- GET  /generate_model/status/<id>   -> poll the fit job
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from flask import Blueprint, jsonify, request

from pivtools_core.config import get_config
from pivtools_core.image_handling.calibration_loader import read_calibration_image
from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.calibration2 import record as rec
from pivtools_gui.calibration2 import world_frame as WF
from pivtools_gui.calibration2.camera_model import DistortionModel
from pivtools_gui.calibration2.detection.base import DetectionResult
from pivtools_gui.calibration2.stepped_calibrate import (
    calibrate_stepped_mono,
    calibrate_stepped_stereo,
)
from pivtools_cli import calibration2_cli as c2

calibration2_stepped_bp = Blueprint("calibration2_stepped", __name__)
logger = logging.getLogger(__name__)

_MODEL = DistortionModel.STANDARD

# ---------------------------------------------------------------------------
# Sequence cache — heavy detections + images held server-side between the
# detect phase and the generate phase, keyed by an opaque sequence_id with a TTL.
# ---------------------------------------------------------------------------

_SEQUENCE_TTL_SECONDS = 2 * 60 * 60  # 2 hours
_sequence_cache: Dict[str, dict] = {}
_sequence_lock = threading.Lock()


def _gc_sequences(now: Optional[float] = None) -> None:
    """Drop cache entries older than the TTL."""
    if now is None:
        now = time.time()
    with _sequence_lock:
        stale = [
            sid for sid, e in _sequence_cache.items()
            if now - e.get("created_at", now) > _SEQUENCE_TTL_SECONDS
        ]
        for sid in stale:
            _sequence_cache.pop(sid, None)


def _lookup(sequence_id: str):
    """Return (entry, None) or (None, error_response) for a sequence id."""
    _gc_sequences()
    with _sequence_lock:
        entry = _sequence_cache.get(sequence_id)
    if entry is None:
        return None, (jsonify({
            "error": (
                f"sequence {sequence_id} not found — it may have expired or the "
                f"backend restarted; re-run detect_sequence"
            ),
        }), 410)
    return entry, None


def _cfg() -> dict:
    return c2._cfg2(get_config())


def _source_idx(data: dict) -> int:
    v = data.get("source_path_idx", data.get("source_idx"))
    return int(v) if v not in (None, "") else int(_cfg().get("source_idx", 0))


def _resolve_cameras(data: dict, cfg: dict) -> List[int]:
    """Resolve the camera list: explicit ``cameras``, else stereo pair, else mono camera."""
    cams = data.get("cameras")
    if cams:
        return [int(c) for c in cams]
    if bool(data.get("stereo", False)):
        pair = data.get("camera_pair") or cfg.get("camera_pair", [1, 2])
        if isinstance(pair, str):
            pair = [int(x) for x in pair.split(",")]
        return [int(pair[0]), int(pair[1])]
    return [int(data.get("camera", cfg.get("camera", 1)))]


def _strip_internal(diag: dict) -> dict:
    """JSON-safe copy of a detection's diagnostics, dropping ``_``-prefixed keys."""
    return {k: v for k, v in (diag or {}).items() if not k.startswith("_")}


def _detection_payload(det: Optional[DetectionResult], image_size: Tuple[int, int]) -> dict:
    """One pose's JSON-safe detection: points + per-level breakdown for the overlay."""
    if det is None or not det.success:
        return {"ok": False, "image_size": list(image_size)}
    diag = _strip_internal(det.diagnostics)
    return {
        "ok": True,
        "n_points": det.n,
        "image_size": list(image_size),
        "image_points": det.image_points.tolist(),
        "grid_indices": (None if det.grid_indices is None else det.grid_indices.tolist()),
        "level_a": diag.get("level_a"),
        "level_b": diag.get("level_b"),
        "level_labels": diag.get("level_labels"),
        "stitch": diag.get("stitch"),
    }


def _load_one(camera: int, frame: int, source_idx: int,
              image_format: Optional[str], image_type: Optional[str]) -> np.ndarray:
    return read_calibration_image(
        int(frame), int(camera), get_config(), int(source_idx),
        image_format=image_format, image_type=image_type)


# ===========================================================================
# ROUTE 1: detect sequence (1 or 2 cameras, N poses) — background job
# ===========================================================================

@calibration2_stepped_bp.route("/calibration2/stepped/detect_sequence", methods=["POST"])
def detect_sequence():
    """Detect a stepped-board pose sequence for one or two cameras and cache it.

    Request JSON: ``source_path_idx``, (``cameras`` | ``stereo``+``camera_pair`` |
    ``camera``), ``num_frames``, ``start_frame_idx``, ``datum_frame_idx``,
    ``board_params`` (geometry overrides), ``image_format``, ``image_type``.
    Returns ``{job_id, sequence_id}``; poll the job for per-pose summaries.
    """
    data = request.get_json() or {}
    cfg = _cfg()
    cameras = _resolve_cameras(data, cfg)
    source_idx = _source_idx(data)
    num_frames = int(data.get("num_frames", 1))
    start_frame_idx = int(data.get("start_frame_idx", 1))
    datum_frame_idx = int(data.get("datum_frame_idx", start_frame_idx))
    image_format = data.get("image_format")
    image_type = data.get("image_type")
    params = c2._board_params(cfg, "stepped", data.get("board_params"))

    if num_frames < 1:
        return jsonify({"error": "num_frames must be >= 1"}), 400
    frame_indices = list(range(start_frame_idx, start_frame_idx + num_frames))
    if datum_frame_idx not in frame_indices:
        return jsonify({
            "error": f"datum_frame_idx {datum_frame_idx} must lie in {frame_indices}",
        }), 400

    sequence_id = uuid.uuid4().hex
    job_id = job_manager.create_job(
        "calibration2_stepped_detect_sequence",
        cameras=cameras, num_frames=num_frames, datum_frame_idx=datum_frame_idx,
        sequence_id=sequence_id, total=num_frames * len(cameras), processed=0,
        stage="starting",
    )

    def _run():
        try:
            job_manager.update_job(job_id, status="running", stage="detecting", progress=0)
            detector = c2._build_detector("stepped", params)
            detections: Dict[int, List[Optional[DetectionResult]]] = {c: [] for c in cameras}
            images: Dict[int, List[Optional[np.ndarray]]] = {c: [] for c in cameras}
            image_size: Dict[int, Tuple[int, int]] = {}
            per_frame_status: Dict[str, str] = {}
            done = 0
            total = num_frames * len(cameras)
            for camera in cameras:
                for frame in frame_indices:
                    det: Optional[DetectionResult] = None
                    img: Optional[np.ndarray] = None
                    try:
                        img = _load_one(camera, frame, source_idx, image_format, image_type)
                        h, w = np.asarray(img).shape[:2]
                        image_size[camera] = (int(w), int(h))
                        det = detector.detect(img)
                        if not det.success:
                            per_frame_status[f"{camera}:{frame}"] = (
                                f"not detected: {det.diagnostics.get('error', 'unknown')}")
                            det = None
                        else:
                            per_frame_status[f"{camera}:{frame}"] = "ok"
                    except Exception as exc:
                        logger.warning("stepped detect cam%s frame%s failed: %s", camera, frame, exc)
                        per_frame_status[f"{camera}:{frame}"] = f"failed: {exc}"
                    detections[camera].append(det)
                    images[camera].append(img)
                    done += 1
                    job_manager.update_job(
                        job_id, processed=done, total=total,
                        progress=int(done / max(total, 1) * 90),
                        stage=f"cam{camera}_frame{frame}")

            with _sequence_lock:
                _sequence_cache[sequence_id] = {
                    "cameras": cameras,
                    "frame_indices": frame_indices,
                    "datum_frame_idx": datum_frame_idx,
                    "source_idx": source_idx,
                    "params": params,
                    "image_format": image_format,
                    "image_type": image_type,
                    "detections": detections,
                    "images": images,
                    "image_size": image_size,
                    "created_at": time.time(),
                }

            # Per-camera, per-pose JSON-safe summary + the datum detection for the overlay.
            poses: Dict[str, list] = {}
            datum_detection: Dict[str, dict] = {}
            datum_pos = frame_indices.index(datum_frame_idx)
            for camera in cameras:
                size = image_size.get(camera, (0, 0))
                cam_poses = []
                for fi, det in zip(frame_indices, detections[camera]):
                    s = {"frame_idx": fi, "is_datum": fi == datum_frame_idx,
                         "ok": det is not None}
                    if det is not None:
                        diag = _strip_internal(det.diagnostics)
                        s["n_level_a"] = (diag.get("level_a") or {}).get("n_points", 0)
                        s["n_level_b"] = (diag.get("level_b") or {}).get("n_points", 0)
                        s["n_points"] = det.n
                    else:
                        s["error"] = per_frame_status.get(f"{camera}:{fi}", "failed")
                    cam_poses.append(s)
                poses[str(camera)] = cam_poses
                datum_detection[str(camera)] = _detection_payload(
                    detections[camera][datum_pos], size)

            job_manager.complete_job(
                job_id, sequence_id=sequence_id, cameras=cameras,
                frame_indices=frame_indices, datum_frame_idx=datum_frame_idx,
                poses=poses, datum_detection=datum_detection,
                image_size={str(c): list(image_size.get(c, (0, 0))) for c in cameras},
                per_frame_status=per_frame_status)
        except Exception as exc:
            logger.exception("stepped detect_sequence job %s failed", job_id)
            job_manager.fail_job(job_id, f"{type(exc).__name__}: {exc}")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "sequence_id": sequence_id, "status": "starting"})


@calibration2_stepped_bp.route(
    "/calibration2/stepped/detect_sequence/status/<job_id>", methods=["GET"])
def detect_sequence_status(job_id: str):
    """Poll a stepped sequence-detection job."""
    data = job_manager.get_job_with_timing(job_id)
    if data is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(data)


# ===========================================================================
# ROUTE 2: per-pose detection + level identification
# ===========================================================================

def _pose_det(entry: dict, camera: int, frame_idx: int):
    """(detection, error_response). Validates the camera + frame against the cache."""
    if camera not in entry["cameras"]:
        return None, (jsonify({"error": f"camera {camera} not in this sequence"}), 400)
    if frame_idx not in entry["frame_indices"]:
        return None, (jsonify({"error": f"frame_idx {frame_idx} not in sequence"}), 400)
    pos = entry["frame_indices"].index(frame_idx)
    return entry["detections"][camera][pos], None


@calibration2_stepped_bp.route(
    "/calibration2/stepped/sequence_pose_detection", methods=["GET"])
def sequence_pose_detection():
    """Return one pose's JSON-safe detection (points + per-level breakdown)."""
    sequence_id = request.args.get("sequence_id")
    camera = int(request.args.get("camera", 1))
    frame_idx = int(request.args.get("frame_idx", 0))
    if not sequence_id:
        return jsonify({"error": "sequence_id is required"}), 400
    entry, err = _lookup(sequence_id)
    if err is not None:
        return err
    det, err = _pose_det(entry, camera, frame_idx)
    if err is not None:
        return err
    size = entry["image_size"].get(camera, (0, 0))
    return jsonify(_detection_payload(det, size))


@calibration2_stepped_bp.route(
    "/calibration2/stepped/identify_pose_level", methods=["POST"])
def identify_pose_level():
    """Snap a click to the nearest dot and report which level (A/B) it sits on.

    The peak/trough mapping is the operator's (via the datum ``clicked_level``); this
    only reports the stable A/B membership the detector assigns, plus the snapped dot.
    """
    data = request.get_json() or {}
    sequence_id = data.get("sequence_id")
    camera = int(data.get("camera", 1))
    frame_idx = int(data.get("frame_idx", 0))
    cx, cy = float(data.get("click_x", 0)), float(data.get("click_y", 0))
    if not sequence_id:
        return jsonify({"error": "sequence_id is required"}), 400
    entry, err = _lookup(sequence_id)
    if err is not None:
        return err
    det, err = _pose_det(entry, camera, frame_idx)
    if err is not None:
        return err
    if det is None:
        return jsonify({"error": f"no detection for camera {camera} frame {frame_idx}"}), 400

    idx = WF._snap(det.image_points, (cx, cy))
    sx, sy = float(det.image_points[idx, 0]), float(det.image_points[idx, 1])

    diag = _strip_internal(det.diagnostics)
    a_centers = (diag.get("level_a") or {}).get("centers") or []
    b_centers = (diag.get("level_b") or {}).get("centers") or []

    def _min_d2(centers):
        if not centers:
            return float("inf")
        c = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
        return float(np.min((c[:, 0] - sx) ** 2 + (c[:, 1] - sy) ** 2))

    level = "A" if _min_d2(a_centers) <= _min_d2(b_centers) else "B"
    return jsonify({"level": level, "snapped_x": sx, "snapped_y": sy})


# ===========================================================================
# ROUTE 3: snap a fiducial against a camera's datum pose
# ===========================================================================

@calibration2_stepped_bp.route("/calibration2/stepped/snap_fiducial", methods=["POST"])
def snap_fiducial():
    """Snap a fiducial click to the nearest dot on a camera's datum pose."""
    data = request.get_json() or {}
    sequence_id = data.get("sequence_id")
    camera = int(data.get("camera", 1))
    cx, cy = float(data.get("click_x", 0)), float(data.get("click_y", 0))
    if not sequence_id:
        return jsonify({"error": "sequence_id is required"}), 400
    entry, err = _lookup(sequence_id)
    if err is not None:
        return err
    det, err = _pose_det(entry, camera, entry["datum_frame_idx"])
    if err is not None:
        return err
    if det is None:
        return jsonify({
            "error": (
                f"datum pose {entry['datum_frame_idx']} has no detection for camera "
                f"{camera}; re-run detect_sequence"),
        }), 400
    idx = WF._snap(det.image_points, (cx, cy))
    px = det.image_points[idx]
    gi = det.grid_indices[idx] if det.grid_indices is not None else (0, 0)
    return jsonify({
        "snapped_x": float(px[0]), "snapped_y": float(px[1]),
        "grid_col": int(gi[0]), "grid_row": int(gi[1]),
    })


# ===========================================================================
# ROUTE 4: generate model (mono or stereo) — background job
# ===========================================================================

def _camera_spec(data: dict, camera: int) -> dict:
    """Pull a camera's {fiducials, clicked_level, pose_levels} from the request.

    Accepts either a per-camera ``cameras`` map (``{"<cam>": {...}}``) or, for the
    mono case, the spec at the top level.
    """
    by_cam = data.get("cameras")
    if isinstance(by_cam, dict) and str(camera) in by_cam:
        return by_cam[str(camera)]
    return data


def _validate_spec(spec: dict, frame_indices: List[int], camera: int):
    """(fiducials, clicked_level, pose_levels_dict) or (None, error_response)."""
    fiducials = spec.get("fiducials") or {}
    clicked_level = spec.get("clicked_level")
    pose_levels_raw = spec.get("pose_levels")
    if clicked_level not in ("peak", "trough"):
        return None, (jsonify({
            "error": f"camera {camera}: clicked_level must be 'peak' or 'trough'"}), 400)
    for key in ("origin", "x_axis", "y_axis"):
        v = fiducials.get(key)
        if v is None or len(v) != 2:
            return None, (jsonify({
                "error": f"camera {camera}: fiducial '{key}' must be [x, y]"}), 400)
    if not isinstance(pose_levels_raw, dict):
        return None, (jsonify({
            "error": f"camera {camera}: pose_levels must be a {{frame_idx: peak|trough}} map"}), 400)
    try:
        pose_levels = {int(k): v for k, v in pose_levels_raw.items()}
    except (TypeError, ValueError) as exc:
        return None, (jsonify({
            "error": f"camera {camera}: pose_levels keys must be integers ({exc})"}), 400)
    for fi in frame_indices:
        if pose_levels.get(fi) not in ("peak", "trough"):
            return None, (jsonify({
                "error": f"camera {camera}: pose_levels missing/invalid for frame {fi}"}), 400)
    return (fiducials, clicked_level, pose_levels), None


def _aligned(entry: dict, camera: int, pose_levels: dict):
    """(detections_list, images_list, pose_levels_list) aligned to frame order."""
    frame_indices = entry["frame_indices"]
    dets = entry["detections"][camera]
    imgs = entry["images"][camera]
    levels = [pose_levels[fi] for fi in frame_indices]
    return dets, imgs, levels


@calibration2_stepped_bp.route("/calibration2/stepped/generate_model", methods=["POST"])
def generate_model():
    """Run the stepped fit (mono or stereo) from a detected sequence, save + figures.

    Request JSON: ``sequence_id``, ``stereo`` (bool), ``no_figures`` (bool), and a
    per-camera spec — ``cameras: {"<cam>": {fiducials, clicked_level, pose_levels}}``
    (or the spec at the top level for mono). Stereo also takes ``stereo_config``
    ('auto'|'same_side'|'transmission'). Returns ``{job_id}``; poll for the result.
    """
    data = request.get_json() or {}
    sequence_id = data.get("sequence_id")
    if not sequence_id:
        return jsonify({"error": "sequence_id is required"}), 400
    entry, err = _lookup(sequence_id)
    if err is not None:
        return err

    cfg = _cfg()
    cameras = entry["cameras"]
    frame_indices = entry["frame_indices"]
    datum_index = frame_indices.index(entry["datum_frame_idx"])
    board = entry["params"].board()
    make_figs = not bool(data.get("no_figures", False))
    stereo = bool(data.get("stereo", False))
    if stereo and len(cameras) < 2:
        return jsonify({"error": "stereo requested but the sequence has one camera"}), 400
    model_type = str(data.get("model_type", "pinhole"))
    if model_type not in ("pinhole", "polynomial3d"):
        return jsonify({"error": f"model_type must be 'pinhole' or 'polynomial3d', "
                                 f"got {model_type!r}"}), 400

    # Validate every camera's spec up front (fail fast, before the job thread).
    specs: Dict[int, tuple] = {}
    use_cams = cameras[:2] if stereo else cameras[:1]
    for camera in use_cams:
        parsed, verr = _validate_spec(_camera_spec(data, camera), frame_indices, camera)
        if verr is not None:
            return verr
        specs[camera] = parsed

    try:
        source = get_config().get_calibration_source(int(entry["source_idx"]))
    except (ValueError, IndexError) as exc:
        return jsonify({"error": f"calibration source not configured ({exc})"}), 400

    job_id = job_manager.create_job(
        "calibration2_stepped_generate_model", stereo=stereo, stage="starting")

    def _run():
        try:
            job_manager.update_job(job_id, status="running", stage="fitting", progress=10)
            if stereo:
                cam1, cam2 = use_cams[0], use_cams[1]
                fid1, lvl1, pl1 = specs[cam1]
                fid2, lvl2, pl2 = specs[cam2]
                d1, i1, levels1 = _aligned(entry, cam1, pl1)
                d2, i2, levels2 = _aligned(entry, cam2, pl2)
                model_dir = rec.stereo_model_dir_for_source(source, cam1, cam2)
                fig_dir = (model_dir.parent / "figures") if make_figs else None
                record = calibrate_stepped_stereo(
                    detections1=d1, detections2=d2, fiducials1=fid1, fiducials2=fid2,
                    clicked_level1=lvl1, clicked_level2=lvl2,
                    pose_levels1=levels1, pose_levels2=levels2,
                    board=board, image_size1=entry["image_size"][cam1],
                    image_size2=entry["image_size"][cam2],
                    cam1=cam1, cam2=cam2, datum_index=datum_index,
                    stereo_config=str(data.get("stereo_config", "auto")),
                    model_type=model_type,
                    distortion_model=_MODEL, images1=i1, images2=i2, figure_dir=fig_dir)
                path = rec.save_stereo(record, model_dir)
                job_manager.complete_job(
                    job_id, stereo=True, model_type=model_type, model_path=str(path),
                    rms_cam1=_model_rms(record.model1), rms_cam2=_model_rms(record.model2),
                    per_view_rms1=list(record.per_view_rms1),
                    per_view_rms2=list(record.per_view_rms2),
                    plane_rms_cam1=_plane_rms(record.model1),
                    plane_rms_cam2=_plane_rms(record.model2),
                    num_pairs_used=len(record.per_view_rms1),
                    stereo_config=record.board_meta.get("stereo_config"),
                    baseline_mm=record.board_meta.get("baseline_mm"),
                    relative_angle_deg=record.board_meta.get("relative_angle_deg"),
                    figures=_list_figures(fig_dir))
            else:
                camera = use_cams[0]
                fid, lvl, pl = specs[camera]
                d, i, levels = _aligned(entry, camera, pl)
                model_dir = rec.mono_model_dir_for_source(source, camera, "stepped")
                fig_dir = (model_dir.parent / "figures") if make_figs else None
                record = calibrate_stepped_mono(
                    detections=d, fiducials=fid, clicked_level=lvl, pose_levels=levels,
                    board=board, image_size=entry["image_size"][camera],
                    camera=camera, datum_index=datum_index, distortion_model=_MODEL,
                    model_type=model_type, images=i, figure_dir=fig_dir)
                path = rec.save_mono(record, model_dir)
                cm = record.camera_model
                done = dict(
                    stereo=False, model_type=model_type, model_path=str(path),
                    camera=camera, rms=_model_rms(cm),
                    num_views_used=len(record.per_view_rms),
                    per_view_rms=list(record.per_view_rms),
                    plane_rms=_plane_rms(cm),
                    clicked_level=record.board_meta.get("clicked_level"),
                    figures=_list_figures(fig_dir))
                if model_type == "pinhole":
                    # Intrinsics are pinhole-only; a polynomial has no K.
                    done.update(fx=float(cm.K[0, 0]), fy=float(cm.K[1, 1]),
                                cx=float(cm.K[0, 2]), cy=float(cm.K[1, 2]))
                job_manager.complete_job(job_id, **done)
        except Exception as exc:
            logger.exception("stepped generate_model job %s failed", job_id)
            job_manager.fail_job(job_id, f"{type(exc).__name__}: {exc}")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "starting"})


@calibration2_stepped_bp.route(
    "/calibration2/stepped/generate_model/status/<job_id>", methods=["GET"])
def generate_model_status(job_id: str):
    """Poll a stepped model-generation job."""
    data = job_manager.get_job_with_timing(job_id)
    if data is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(data)


def _list_figures(fig_dir) -> List[str]:
    if not fig_dir or not Path(fig_dir).is_dir():
        return []
    return sorted(p.name for p in Path(fig_dir).glob("*.png") if not p.name.startswith("._"))


def _model_rms(model) -> float:
    """Overall reprojection RMS (px) for a pinhole (``rms``) or poly3d (``rms_px``) model."""
    return float(getattr(model, "rms", None) if hasattr(model, "rms") else model.rms_px)


def _plane_rms(model) -> List[float]:
    """Per-plane reprojection RMS (px); only the 3D polynomial has it, else empty."""
    return [float(v) for v in getattr(model, "plane_rms_px", ())]
