"""calibration stepped-board Flask blueprint — the GUI API for the dual-level board.

The stepped flow is the one calibration path that is genuinely STATEFUL across
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
``/calibration/*`` route in ``views.py`` driven with ``board="stepped"``; only the
sequence detect + per-pose level labelling + the stepped fit live here.

Routes (prefix ``/calibration/stepped`` under the app's ``/backend``):
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

from pivtools_cli import calibration_cli as c2
from pivtools_core.config import get_config
from pivtools_core.image_handling.calibration_loader import read_calibration_image
from pivtools_gui.calibration import record as rec
from pivtools_gui.calibration import world_frame as WF
from pivtools_gui.calibration.camera_model import DistortionModel
from pivtools_gui.calibration.detection.base import DetectionResult
from pivtools_gui.calibration.inputs_store import (
    joint_det_key,
    save_inputs,
    try_load_inputs,
)
from pivtools_gui.calibration.stepped_calibrate import (
    calibrate_stepped_mono,
    calibrate_stepped_stereo,
)
from pivtools_gui.services.job_manager import job_manager

calibration_stepped_bp = Blueprint("calibration_stepped", __name__)
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
            sid
            for sid, e in _sequence_cache.items()
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
        return None, (
            jsonify(
                {
                    "error": (
                        f"sequence {sequence_id} not found — it may have expired or the "
                        f"backend restarted; re-run detect_sequence"
                    ),
                }
            ),
            410,
        )
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


def _overlay_block(lv: Optional[dict]) -> Optional[dict]:
    """Slim per-level overlay ``{centers, grid_indices, n_points}`` for the GUI dot mesh,
    derived from a raw per-level grid dict (``DetectionResult.level_data['a'|'b']``).

    The fit input is the single source — the overlay is a JSON-safe projection of it, so a
    sidecar-restored detection draws the same mesh as a freshly-detected one."""
    if not isinstance(lv, dict):
        return None
    centers = np.asarray(lv.get("centers", []), dtype=np.float64).reshape(-1, 2)
    gi = np.asarray(lv.get("grid_indices", []), dtype=np.int64).reshape(-1, 2)
    return {
        "centers": centers.tolist(),
        "grid_indices": gi.tolist(),
        "n_points": int(len(centers)),
    }


def _overlay_blocks(det: DetectionResult) -> Tuple[Optional[dict], Optional[dict]]:
    """``(level_a, level_b)`` overlay blocks for a detection (None when no level_data)."""
    ld = det.level_data or {}
    return _overlay_block(ld.get("a")), _overlay_block(ld.get("b"))


def _detection_payload(
    det: Optional[DetectionResult], image_size: Tuple[int, int]
) -> dict:
    """One pose's JSON-safe detection: points + per-level breakdown for the overlay."""
    if det is None or not det.success:
        return {"ok": False, "image_size": list(image_size)}
    diag = _strip_internal(det.diagnostics)
    level_a, level_b = _overlay_blocks(det)
    return {
        "ok": True,
        "n_points": det.n,
        "image_size": list(image_size),
        "image_points": det.image_points.tolist(),
        "grid_indices": (
            None if det.grid_indices is None else det.grid_indices.tolist()
        ),
        "level_a": level_a,
        "level_b": level_b,
        "level_labels": diag.get("level_labels"),
        "stitch": diag.get("stitch"),
    }


def _summarize_sequence(entry: dict, per_frame_status: Optional[dict] = None):
    """Per-camera pose summaries + the datum detection payload, for the overlay.

    Shared by ``detect_sequence`` (live detection) and ``restore_sequence`` (rehydrated
    from the sidecar) so both return the identical shape the GUI consumes. ``entry`` is a
    sequence-cache entry (or a sidecar-rebuilt one); ``per_frame_status`` supplies the
    per-pose failure text for live detections (absent on a sidecar restore -> "failed").
    Returns ``(poses, datum_detection)``.
    """
    per_frame_status = per_frame_status or {}
    frame_indices = entry["frame_indices"]
    datum_frame_idx = entry["datum_frame_idx"]
    detections = entry["detections"]
    image_size = entry["image_size"]
    datum_pos = frame_indices.index(datum_frame_idx)
    poses: Dict[str, list] = {}
    datum_detection: Dict[str, dict] = {}
    for camera in entry["cameras"]:
        size = image_size.get(camera, (0, 0))
        cam_poses = []
        for fi, det in zip(frame_indices, detections[camera]):
            s = {
                "frame_idx": fi,
                "is_datum": fi == datum_frame_idx,
                "ok": det is not None,
            }
            if det is not None:
                la, lb = _overlay_blocks(det)
                s["n_level_a"] = (la or {}).get("n_points", 0)
                s["n_level_b"] = (lb or {}).get("n_points", 0)
                s["n_points"] = det.n
            else:
                s["error"] = per_frame_status.get(f"{camera}:{fi}", "failed")
            cam_poses.append(s)
        poses[str(camera)] = cam_poses
        datum_detection[str(camera)] = _detection_payload(
            detections[camera][datum_pos], size
        )
    return poses, datum_detection


def _load_one(
    camera: int,
    frame: int,
    source_idx: int,
    image_format: Optional[str],
    image_type: Optional[str],
) -> np.ndarray:
    return read_calibration_image(
        int(frame),
        int(camera),
        get_config(),
        int(source_idx),
        image_format=image_format,
        image_type=image_type,
    )


# ---------------------------------------------------------------------------
# Sidecar persistence — mirror the flat/joint inputs.mat so a stepped model can be
# regenerated from disk (detections + clicks) without re-detecting or re-clicking.
# The in-memory _sequence_cache stays the fast in-session layer; this is the durable
# store keyed by det_key, with the interactive spec stashed in coords.
# ---------------------------------------------------------------------------


def _sidecar_model_dir(source, cameras: List[int]) -> Path:
    """Model dir whose ``inputs.mat`` carries this sequence (stereo pair vs mono)."""
    if len(cameras) >= 2:
        return rec.stereo_model_dir_for_source(source, cameras[0], cameras[1])
    return rec.mono_model_dir_for_source(source, cameras[0], "stepped")


def _stepped_det_key(
    cameras, frame_indices, datum_frame_idx, image_format, image_type, params
) -> str:
    """det_key for a stepped sequence: folds the frame window + datum into the key so a
    different selection invalidates the cached detections (board geometry rides in
    ``params`` via its repr)."""
    sig = (tuple(int(f) for f in frame_indices), int(datum_frame_idx), params)
    return joint_det_key(
        "stepped", len(frame_indices), image_format, image_type, cameras, sig
    )


def _failed_det() -> DetectionResult:
    """success=False placeholder so a failed pose keeps its positional slot in the
    sidecar; mapped back to ``None`` on load."""
    return DetectionResult(
        success=False,
        board_type="stepped",
        image_points=np.empty((0, 2)),
        board_local_points=np.empty((0, 3)),
    )


def _dets_for_sidecar(cache_dets: dict) -> Dict[int, list]:
    """Cache detections (``None`` for failed poses) -> save-safe (``None`` -> placeholder)."""
    return {
        int(c): [d if d is not None else _failed_det() for d in lst]
        for c, lst in cache_dets.items()
    }


def _dets_from_sidecar(side_dets: dict) -> Dict[int, list]:
    """Loaded sidecar detections -> cache shape (failed placeholder -> ``None``)."""
    return {
        int(c): [d if (d is not None and d.success) else None for d in lst]
        for c, lst in side_dets.items()
    }


def _save_sequence_sidecar(source, entry: dict) -> None:
    """Persist a detected sequence's detections + descriptor + det_key + geometry,
    merging into any existing ``coords`` so a prior generate's clicks are preserved."""
    cameras = entry["cameras"]
    model_dir = _sidecar_model_dir(source, cameras)
    det_key = _stepped_det_key(
        cameras,
        entry["frame_indices"],
        entry["datum_frame_idx"],
        entry["image_format"],
        entry["image_type"],
        entry["params"],
    )
    prev = try_load_inputs(model_dir)
    coords = dict(prev.coords) if (prev and prev.coords) else {}
    coords["sequence"] = {
        "frame_indices": [int(f) for f in entry["frame_indices"]],
        "datum_frame_idx": int(entry["datum_frame_idx"]),
        "cameras": [int(c) for c in cameras],
        "source_idx": int(entry["source_idx"]),
        "image_format": entry["image_format"],
        "image_type": entry["image_type"],
    }
    save_inputs(
        model_dir,
        path_type="stepped",
        board_type="stepped",
        detections=_dets_for_sidecar(entry["detections"]),
        image_size_by_cam={
            int(c): tuple(entry["image_size"].get(c, (0, 0))) for c in cameras
        },
        det_key=det_key,
        board_params=rec.geometry_meta("stepped", entry["params"]),
        coords=coords,
    )


def _entry_from_sidecar(data: dict, cfg: dict, cameras: List[int]):
    """Rebuild a sequence ``entry`` from a model dir's sidecar (no live sequence).

    Returns ``(entry, model_dir, side, None)`` or ``(None, None, None, error)``.
    Images are left ``None`` (the sidecar stores no pixels); call ``_reread_images``
    when figures are needed.
    """
    try:
        source = get_config().get_calibration_source(_source_idx(data))
    except (ValueError, IndexError) as exc:
        return (
            None,
            None,
            None,
            (jsonify({"error": f"calibration source not configured ({exc})"}), 400),
        )
    model_dir = _sidecar_model_dir(source, cameras)
    side = try_load_inputs(model_dir)
    if side is None or not side.detections:
        return (
            None,
            None,
            None,
            (
                jsonify(
                    {
                        "error": (
                            "no saved detections for this camera set — run detect_sequence "
                            "first"
                        )
                    }
                ),
                410,
            ),
        )
    seq = (side.coords or {}).get("sequence") or {}
    frame_indices = [int(f) for f in seq.get("frame_indices", [])]
    if not frame_indices:
        return (
            None,
            None,
            None,
            (
                jsonify(
                    {
                        "error": (
                            "saved inputs are missing the sequence descriptor; "
                            "re-run detect_sequence"
                        )
                    }
                ),
                410,
            ),
        )
    datum_frame_idx = int(seq.get("datum_frame_idx", frame_indices[0]))
    if datum_frame_idx not in frame_indices:
        # A truncated / desynced sidecar — guard here so callers get a clean error rather
        # than an unhandled ValueError from frame_indices.index(datum_frame_idx) downstream.
        return (
            None,
            None,
            None,
            (
                jsonify(
                    {
                        "error": (
                            "saved datum_frame_idx is not in the saved frame window; "
                            "re-run detect_sequence"
                        )
                    }
                ),
                410,
            ),
        )
    params = c2._board_params(
        cfg, "stepped", data.get("board_params"), sidecar=side.board_params
    )
    dets = _dets_from_sidecar(side.detections)
    entry = {
        "cameras": [int(c) for c in cameras],
        "frame_indices": frame_indices,
        "datum_frame_idx": datum_frame_idx,
        "source_idx": int(seq.get("source_idx", _source_idx(data))),
        "params": params,
        "image_format": seq.get("image_format"),
        "image_type": seq.get("image_type"),
        "detections": {int(c): dets.get(int(c), []) for c in cameras},
        "images": {int(c): [None] * len(frame_indices) for c in cameras},
        "image_size": {int(k): tuple(v) for k, v in side.image_size_by_cam.items()},
        "created_at": time.time(),
    }
    return entry, model_dir, side, None


def _reread_images(entry: dict) -> None:
    """Fill ``entry['images']`` from the source (a sidecar restore stores no pixels)."""
    for camera in entry["cameras"]:
        imgs: List[Optional[np.ndarray]] = []
        for fi in entry["frame_indices"]:
            try:
                imgs.append(
                    _load_one(
                        camera,
                        fi,
                        entry["source_idx"],
                        entry["image_format"],
                        entry["image_type"],
                    )
                )
            except Exception as exc:
                logger.warning(
                    "stepped figure image reload cam%s frame%s failed: %s",
                    camera,
                    fi,
                    exc,
                )
                imgs.append(None)
        entry["images"][camera] = imgs


def _images_missing(entry: dict) -> bool:
    """True if any camera's frame image is absent (None / non-2-D).

    Figures need pixels. Both the sidecar restore and ``restore_sequence`` cache image-free
    entries (``images=[None]*N``); only a freshly-detected sequence carries them. Gating the
    figure-image reload on this — not on *how* the entry was sourced (the old ``restored``
    flag missed the ``restore_sequence`` cache) — covers every path.
    """
    imgs = entry.get("images") or {}
    n = len(entry["frame_indices"])
    for c in entry["cameras"]:
        seq = imgs.get(int(c), [])
        if len(seq) != n or any(im is None or np.asarray(im).ndim < 2 for im in seq):
            return True
    return False


def _spec_to_coords(specs: dict, use_cams: List[int], stereo: bool, data: dict) -> dict:
    """The per-camera clicks block to persist in the sidecar ``coords`` (re-loadable)."""
    cams: Dict[str, dict] = {}
    for c in use_cams:
        fid, lvl, pl = specs[c]
        cams[str(c)] = {
            "fiducials": {
                k: [float(fid[k][0]), float(fid[k][1])]
                for k in ("origin", "x_axis", "y_axis")
            },
            "clicked_level": str(lvl),
            "pose_levels": {str(k): str(v) for k, v in pl.items()},
        }
    out: Dict[str, object] = {"cameras": cams}
    if stereo:
        out["stereo_config"] = str(data.get("stereo_config", "auto"))
    return out


# ===========================================================================
# ROUTE 1: detect sequence (1 or 2 cameras, N poses) — background job
# ===========================================================================


@calibration_stepped_bp.route("/calibration/stepped/detect_sequence", methods=["POST"])
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
        return (
            jsonify(
                {
                    "error": f"datum_frame_idx {datum_frame_idx} must lie in {frame_indices}",
                }
            ),
            400,
        )

    sequence_id = uuid.uuid4().hex
    job_id = job_manager.create_job(
        "calibration_stepped_detect_sequence",
        cameras=cameras,
        num_frames=num_frames,
        datum_frame_idx=datum_frame_idx,
        sequence_id=sequence_id,
        total=num_frames * len(cameras),
        processed=0,
        stage="starting",
    )

    def _run():
        try:
            job_manager.update_job(
                job_id, status="running", stage="detecting", progress=0
            )
            detector = c2._build_detector("stepped", params)
            detections: Dict[int, List[Optional[DetectionResult]]] = {
                c: [] for c in cameras
            }
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
                        img = _load_one(
                            camera, frame, source_idx, image_format, image_type
                        )
                        h, w = np.asarray(img).shape[:2]
                        image_size[camera] = (int(w), int(h))
                        det = detector.detect(img)
                        if not det.success:
                            per_frame_status[f"{camera}:{frame}"] = (
                                f"not detected: {det.diagnostics.get('error', 'unknown')}"
                            )
                            det = None
                        else:
                            per_frame_status[f"{camera}:{frame}"] = "ok"
                    except Exception as exc:
                        logger.warning(
                            "stepped detect cam%s frame%s failed: %s",
                            camera,
                            frame,
                            exc,
                        )
                        per_frame_status[f"{camera}:{frame}"] = f"failed: {exc}"
                    detections[camera].append(det)
                    images[camera].append(img)
                    done += 1
                    job_manager.update_job(
                        job_id,
                        processed=done,
                        total=total,
                        progress=int(done / max(total, 1) * 90),
                        stage=f"cam{camera}_frame{frame}",
                    )

            cache_entry = {
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
            with _sequence_lock:
                _sequence_cache[sequence_id] = cache_entry

            # Persist to the model dir's inputs.mat so the model can be regenerated from
            # disk (detections + clicks) without re-detecting. Best-effort: a failure is
            # logged, never fatal — the in-memory cache still serves this session.
            try:
                src = get_config().get_calibration_source(int(source_idx))
                _save_sequence_sidecar(src, cache_entry)
            except Exception as exc:
                logger.warning(
                    "stepped detect: sidecar save failed (non-fatal): %s", exc
                )

            # Per-camera, per-pose JSON-safe summary + the datum detection for the overlay.
            poses, datum_detection = _summarize_sequence(cache_entry, per_frame_status)

            job_manager.complete_job(
                job_id,
                sequence_id=sequence_id,
                cameras=cameras,
                frame_indices=frame_indices,
                datum_frame_idx=datum_frame_idx,
                poses=poses,
                datum_detection=datum_detection,
                image_size={str(c): list(image_size.get(c, (0, 0))) for c in cameras},
                per_frame_status=per_frame_status,
            )
        except Exception as exc:
            logger.exception("stepped detect_sequence job %s failed", job_id)
            job_manager.fail_job(job_id, f"{type(exc).__name__}: {exc}")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "sequence_id": sequence_id, "status": "starting"})


@calibration_stepped_bp.route(
    "/calibration/stepped/detect_sequence/status/<job_id>", methods=["GET"]
)
def detect_sequence_status(job_id: str):
    """Poll a stepped sequence-detection job."""
    from .views import _job_status_response  # local: views imports this module first

    return _job_status_response(job_id)


@calibration_stepped_bp.route("/calibration/stepped/inputs", methods=["GET"])
def stepped_inputs():
    """Whether a persisted detection sidecar exists for this source + camera set, and
    its descriptor + saved clicks. Lets the GUI hydrate the operator's clicks and offer
    a regenerate without re-detecting. Returns ``{"exists": False}`` when none.

    Query: ``source_path_idx``, and ``camera_pair`` (stereo) or ``camera`` (mono).
    """
    pair = request.args.get("camera_pair")
    if pair:
        cameras = [int(x) for x in pair.split(",")]
    else:
        cameras = [int(request.args.get("camera", 1))]
    try:
        source = get_config().get_calibration_source(_source_idx(request.args))
    except (ValueError, IndexError):
        return jsonify({"exists": False})
    side = try_load_inputs(_sidecar_model_dir(source, cameras))
    if side is None or not side.detections:
        return jsonify({"exists": False})
    coords = side.coords or {}
    return jsonify(
        {
            "exists": True,
            "det_key": side.det_key,
            "sequence": coords.get("sequence") or {},
            "cameras": coords.get("cameras") or {},
            "stereo_config": coords.get("stereo_config"),
            "n_detected": {str(c): len(v) for c, v in side.detections.items()},
        }
    )


@calibration_stepped_bp.route("/calibration/stepped/inputs/save", methods=["POST"])
def stepped_inputs_save():
    """Persist the operator's clicks (fiducials / clicked_level / pose_levels) to the model
    dir's ``inputs.mat`` as they are picked — so reopening restores them without a full
    regenerate (config no longer holds clicks). Merges into the existing sidecar via
    ``save_inputs`` (detections + sequence descriptor preserved). No-op until a sequence is
    detected (clicks are made against the datum pose, so a sidecar normally already exists).

    Body: ``source_path_idx``, ``camera_pair`` (stereo) or ``camera`` (mono), and
    ``cameras`` = ``{"<cam>": {fiducials, clicked_level, pose_levels}}`` plus ``stereo_config``.
    """
    data = request.get_json() or {}
    pair = data.get("camera_pair")
    if pair:
        cameras = [int(x) for x in (pair.split(",") if isinstance(pair, str) else pair)]
    else:
        cameras = [int(data.get("camera", 1))]
    try:
        source = get_config().get_calibration_source(_source_idx(data))
    except (ValueError, IndexError):
        return jsonify({"saved": False})
    model_dir = _sidecar_model_dir(source, cameras)
    side = try_load_inputs(model_dir)
    if side is None or not side.detections:
        return jsonify(
            {"saved": False}
        )  # nothing detected yet — no sidecar to attach to
    coords = dict(side.coords or {})
    cams_in = data.get("cameras") or {}
    cams_out = dict(coords.get("cameras") or {})
    for cam in cameras:
        c = cams_in.get(str(cam))
        if isinstance(c, dict):
            cams_out[str(cam)] = (
                c  # stored verbatim; restore handles partial/null picks
            )
    coords["cameras"] = cams_out
    if data.get("stereo_config"):
        coords["stereo_config"] = str(data["stereo_config"])
    save_inputs(model_dir, path_type="stepped", board_type="stepped", coords=coords)
    return jsonify({"saved": True})


@calibration_stepped_bp.route("/calibration/stepped/restore_sequence", methods=["GET"])
def restore_sequence():
    """Rehydrate a saved sequence from the model dir's ``inputs.mat`` into the in-memory
    cache, so the GUI re-shows the detection overlay + peak/trough + fiducials on auto-load
    without re-detecting. Registers a fresh ``sequence_id`` (the cache is per-session) and
    returns the same ``poses`` / ``datum_detection`` shape as ``detect_sequence`` plus the
    saved per-camera clicks. Image-free: detections come straight from the sidecar.

    Query: ``source_path_idx``, and ``camera_pair`` (stereo) or ``camera`` (mono).
    Returns ``{"exists": False}`` when there is no usable sidecar (the normal first-use
    state).
    """
    pair = request.args.get("camera_pair")
    if pair:
        cameras = [int(x) for x in pair.split(",")]
    else:
        cameras = [int(request.args.get("camera", 1))]
    entry, _model_dir, side, err = _entry_from_sidecar(request.args, _cfg(), cameras)
    if err is not None:
        return jsonify({"exists": False})
    sequence_id = uuid.uuid4().hex
    with _sequence_lock:
        _sequence_cache[sequence_id] = entry
    poses, datum_detection = _summarize_sequence(entry)
    coords = side.coords or {}
    return jsonify(
        {
            "exists": True,
            "sequence_id": sequence_id,
            "cameras": [int(c) for c in entry["cameras"]],
            "frame_indices": [int(f) for f in entry["frame_indices"]],
            "datum_frame_idx": int(entry["datum_frame_idx"]),
            "poses": poses,
            "datum_detection": datum_detection,
            "clicks": coords.get("cameras") or {},
            "stereo_config": coords.get("stereo_config"),
        }
    )


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


@calibration_stepped_bp.route(
    "/calibration/stepped/sequence_pose_detection", methods=["GET"]
)
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


@calibration_stepped_bp.route(
    "/calibration/stepped/identify_pose_level", methods=["POST"]
)
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
        return (
            jsonify({"error": f"no detection for camera {camera} frame {frame_idx}"}),
            400,
        )

    idx = WF._snap(det.image_points, (cx, cy))
    sx, sy = float(det.image_points[idx, 0]), float(det.image_points[idx, 1])

    la, lb = _overlay_blocks(det)
    a_centers = (la or {}).get("centers") or []
    b_centers = (lb or {}).get("centers") or []

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


@calibration_stepped_bp.route("/calibration/stepped/snap_fiducial", methods=["POST"])
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
        return (
            jsonify(
                {
                    "error": (
                        f"datum pose {entry['datum_frame_idx']} has no detection for camera "
                        f"{camera}; re-run detect_sequence"
                    ),
                }
            ),
            400,
        )
    idx = WF._snap(det.image_points, (cx, cy))
    px = det.image_points[idx]
    gi = det.grid_indices[idx] if det.grid_indices is not None else (0, 0)
    return jsonify(
        {
            "snapped_x": float(px[0]),
            "snapped_y": float(px[1]),
            "grid_col": int(gi[0]),
            "grid_row": int(gi[1]),
        }
    )


# ===========================================================================
# ROUTE 4: generate model (mono or stereo) — background job
# ===========================================================================


def _camera_spec(
    data: dict, camera: int, sidecar_coords: Optional[dict] = None
) -> dict:
    """Pull a camera's {fiducials, clicked_level, pose_levels} from the request.

    Accepts either a per-camera ``cameras`` map (``{"<cam>": {...}}``) or, for the mono
    case, the spec at the top level. When the request carries no spec for this camera
    (e.g. a regenerate-from-sidecar with an otherwise-empty body), falls back to the
    clicks persisted in the sidecar ``coords``.
    """
    by_cam = data.get("cameras")
    if isinstance(by_cam, dict) and str(camera) in by_cam:
        return by_cam[str(camera)]
    if any(data.get(k) for k in ("fiducials", "clicked_level", "pose_levels")):
        return data
    if sidecar_coords:
        saved = (sidecar_coords.get("cameras") or {}).get(str(camera))
        if saved:
            return saved
    return data


def _validate_spec(spec: dict, frame_indices: List[int], camera: int):
    """(fiducials, clicked_level, pose_levels_dict) or (None, error_response)."""
    fiducials = spec.get("fiducials") or {}
    clicked_level = spec.get("clicked_level")
    pose_levels_raw = spec.get("pose_levels")
    if clicked_level not in ("peak", "trough"):
        return None, (
            jsonify(
                {"error": f"camera {camera}: clicked_level must be 'peak' or 'trough'"}
            ),
            400,
        )
    for key in ("origin", "x_axis", "y_axis"):
        v = fiducials.get(key)
        if v is None or len(v) != 2:
            return None, (
                jsonify({"error": f"camera {camera}: fiducial '{key}' must be [x, y]"}),
                400,
            )
    if not isinstance(pose_levels_raw, dict):
        return None, (
            jsonify(
                {
                    "error": f"camera {camera}: pose_levels must be a {{frame_idx: peak|trough}} map"
                }
            ),
            400,
        )
    try:
        pose_levels = {int(k): v for k, v in pose_levels_raw.items()}
    except (TypeError, ValueError) as exc:
        return None, (
            jsonify(
                {"error": f"camera {camera}: pose_levels keys must be integers ({exc})"}
            ),
            400,
        )
    for fi in frame_indices:
        if pose_levels.get(fi) not in ("peak", "trough"):
            return None, (
                jsonify(
                    {
                        "error": f"camera {camera}: pose_levels missing/invalid for frame {fi}"
                    }
                ),
                400,
            )
    return (fiducials, clicked_level, pose_levels), None


def _warn_missing_datum_image(
    entry: dict, datum_index: int, camera: int, imgs: list
) -> None:
    """Loudly attribute a missing datum frame (the cause of the dewarp-figure failure).

    The stepped fit reads ``image_size`` from ``entry`` and never touches the pixels, so a
    frame that ``_reread_images`` left as ``None`` (sidecar restore + a per-frame load
    failure) only surfaces deep in the figure writer. Name the cam/frame/source here so the
    loader failure is attributable; the figure writer still degrades gracefully.
    """
    img = imgs[datum_index] if 0 <= datum_index < len(imgs) else None
    if img is not None and np.asarray(img).ndim >= 2 and np.asarray(img).size > 0:
        return
    frame = entry["frame_indices"][datum_index]
    logger.warning(
        "stepped stereo: cam%s datum frame %s image missing (shape=%s) — dewarp figure "
        "will show a placeholder. source_idx=%s format=%s; check the earlier image-reload "
        "warning for the load failure.",
        camera,
        frame,
        None if img is None else np.asarray(img).shape,
        entry.get("source_idx"),
        entry.get("image_format"),
    )


def _aligned(entry: dict, camera: int, pose_levels: dict):
    """(detections_list, images_list, pose_levels_list) aligned to frame order."""
    frame_indices = entry["frame_indices"]
    dets = entry["detections"][camera]
    imgs = entry["images"][camera]
    levels = [pose_levels[fi] for fi in frame_indices]
    return dets, imgs, levels


@calibration_stepped_bp.route("/calibration/stepped/generate_model", methods=["POST"])
def generate_model():
    """Run the stepped fit (mono or stereo) from a detected sequence, save + figures.

    Request JSON: ``sequence_id`` (optional — omit to regenerate from the saved
    sidecar), ``stereo`` (bool), ``no_figures`` (bool), and a per-camera spec —
    ``cameras: {"<cam>": {fiducials, clicked_level, pose_levels}}`` (or top-level for
    mono); the spec falls back to the sidecar clicks when the body omits it. Stereo
    also takes ``stereo_config`` ('auto'|'same_side'|'transmission') and an optional
    ``assumed_poses`` map ``{"<cam>": [frame_idx, ...]}`` of poses whose level was
    assumed (datum) rather than user-verified. Returns ``{job_id}``; poll for the result.
    """
    data = request.get_json() or {}
    cfg = _cfg()
    stereo = bool(data.get("stereo", False))
    model_type = str(data.get("model_type", "pinhole"))
    if model_type not in ("pinhole", "polynomial3d"):
        return (
            jsonify(
                {
                    "error": f"model_type must be 'pinhole' or 'polynomial3d', "
                    f"got {model_type!r}"
                }
            ),
            400,
        )
    # Stereo polynomial3d is temporarily disabled — it builds no stereo pose. Reject loudly
    # (the UI also hides it); mono stepped polynomial3d remains available.
    if stereo and model_type == "polynomial3d":
        return (
            jsonify(
                {
                    "error": "stereo polynomial3d calibration is temporarily disabled — "
                    "use pinhole for stereo"
                }
            ),
            400,
        )
    cameras_req = _resolve_cameras(data, cfg)

    # Resolve the detected sequence: the live in-session cache first, else the persisted
    # sidecar (a regenerate after restart / cache expiry / a fresh session with no clicks).
    sequence_id = data.get("sequence_id")
    entry = None
    model_dir = None
    side = None
    if sequence_id:
        entry, err = _lookup(sequence_id)
        if err is not None:
            entry = None  # expired/missing -> fall through to the sidecar
    if entry is None:
        entry, model_dir, side, err = _entry_from_sidecar(data, cfg, cameras_req)
        if err is not None:
            return err

    cameras = entry["cameras"]
    frame_indices = entry["frame_indices"]
    datum_index = frame_indices.index(entry["datum_frame_idx"])
    board = entry["params"].board()
    make_figs = not bool(data.get("no_figures", False))
    if stereo and len(cameras) < 2:
        return (
            jsonify({"error": "stereo requested but the sequence has one camera"}),
            400,
        )
    use_cams = cameras[:2] if stereo else cameras[:1]

    try:
        source = get_config().get_calibration_source(int(entry["source_idx"]))
    except (ValueError, IndexError) as exc:
        return jsonify({"error": f"calibration source not configured ({exc})"}), 400
    if model_dir is None:
        model_dir = _sidecar_model_dir(source, use_cams)
    if side is None:
        side = try_load_inputs(model_dir)
    sidecar_coords = side.coords if (side and side.coords) else None

    # Validate every camera's spec up front (request first, sidecar clicks as fallback).
    specs: Dict[int, tuple] = {}
    for camera in use_cams:
        parsed, verr = _validate_spec(
            _camera_spec(data, camera, sidecar_coords), frame_indices, camera
        )
        if verr is not None:
            return verr
        specs[camera] = parsed

    # Persist the clicks + descriptor so a later session regenerates from the file alone.
    try:
        coords = dict(sidecar_coords) if sidecar_coords else {}
        coords.update(_spec_to_coords(specs, use_cams, stereo, data))
        save_inputs(model_dir, path_type="stepped", board_type="stepped", coords=coords)
    except Exception as exc:
        logger.warning(
            "stepped generate: sidecar coords save failed (non-fatal): %s", exc
        )

    # Figures need pixels. Both the sidecar restore AND restore_sequence cache image-free
    # entries (images=[None]*N); only a freshly-detected sequence carries them. Gate the
    # reload on whether images are actually missing — not on how the entry was sourced — so
    # every restore path is covered.
    if make_figs and _images_missing(entry):
        _reread_images(entry)

    # Poses whose level was assumed (datum) rather than user-verified, recorded for
    # honesty in the saved model (the GUI gates + warns; here we only stamp what it
    # reports). Cam-prefixed keys keep them MATLAB-identifier-safe (gotcha #5).
    assumed_raw = data.get("assumed_poses") or {}
    assumed_poses = {
        f"cam{c}": [
            int(f) for f in (assumed_raw.get(str(c)) or assumed_raw.get(c) or [])
        ]
        for c in use_cams
    }

    job_id = job_manager.create_job(
        "calibration_stepped_generate_model", stereo=stereo, stage="starting"
    )

    def _run():
        from .views import _intrinsics  # local: views imports this module first

        try:
            job_manager.update_job(
                job_id, status="running", stage="fitting", progress=10
            )
            fig_dir = (model_dir.parent / "figures") if make_figs else None
            if stereo:
                cam1, cam2 = use_cams[0], use_cams[1]
                fid1, lvl1, pl1 = specs[cam1]
                fid2, lvl2, pl2 = specs[cam2]
                d1, i1, levels1 = _aligned(entry, cam1, pl1)
                d2, i2, levels2 = _aligned(entry, cam2, pl2)
                if fig_dir is not None:
                    _warn_missing_datum_image(entry, datum_index, cam1, i1)
                    _warn_missing_datum_image(entry, datum_index, cam2, i2)
                record = calibrate_stepped_stereo(
                    detections1=d1,
                    detections2=d2,
                    fiducials1=fid1,
                    fiducials2=fid2,
                    clicked_level1=lvl1,
                    clicked_level2=lvl2,
                    pose_levels1=levels1,
                    pose_levels2=levels2,
                    board=board,
                    image_size1=entry["image_size"][cam1],
                    image_size2=entry["image_size"][cam2],
                    cam1=cam1,
                    cam2=cam2,
                    datum_index=datum_index,
                    stereo_config=str(data.get("stereo_config", "auto")),
                    model_type=model_type,
                    fix_k2=bool(data.get("fix_k2", False)),
                    distortion_model=_MODEL,
                    images1=i1,
                    images2=i2,
                    figure_dir=fig_dir,
                )
                record.board_meta["assumed_poses"] = assumed_poses
                path = rec.save_stereo(record, model_dir)
                done = dict(
                    stereo=True,
                    model_type=model_type,
                    model_path=str(path),
                    rms_cam1=_model_rms(record.model1),
                    rms_cam2=_model_rms(record.model2),
                    per_view_rms1=list(record.per_view_rms1),
                    per_view_rms2=list(record.per_view_rms2),
                    plane_rms_cam1=_plane_rms(record.model1),
                    plane_rms_cam2=_plane_rms(record.model2),
                    num_pairs_used=record.board_meta.get(
                        "n_stereo_views", len(record.per_view_rms1)
                    ),
                    stereo_config=record.board_meta.get("stereo_config"),
                    baseline_mm=record.board_meta.get("baseline_mm"),
                    relative_angle_deg=record.board_meta.get("relative_angle_deg"),
                    method=record.board_meta.get("stereo_method"),
                    assumed_poses=assumed_poses,
                    figures=_list_figures(fig_dir),
                )
                if model_type == "pinhole":
                    # Per-camera intrinsics so the card fills without a second fetch.
                    # Compose has no joint reprojection RMS -> no stereo_rms_px reported.
                    done["intrinsics1"] = _intrinsics(record.model1)
                    done["intrinsics2"] = _intrinsics(record.model2)
                job_manager.complete_job(job_id, **done)
            else:
                camera = use_cams[0]
                fid, lvl, pl = specs[camera]
                d, i, levels = _aligned(entry, camera, pl)
                record = calibrate_stepped_mono(
                    detections=d,
                    fiducials=fid,
                    clicked_level=lvl,
                    pose_levels=levels,
                    board=board,
                    image_size=entry["image_size"][camera],
                    camera=camera,
                    datum_index=datum_index,
                    distortion_model=_MODEL,
                    model_type=model_type,
                    images=i,
                    figure_dir=fig_dir,
                )
                record.board_meta["assumed_poses"] = assumed_poses
                path = rec.save_mono(record, model_dir)
                cm = record.camera_model
                done = dict(
                    stereo=False,
                    model_type=model_type,
                    model_path=str(path),
                    camera=camera,
                    rms=_model_rms(cm),
                    num_views_used=len(record.per_view_rms),
                    per_view_rms=list(record.per_view_rms),
                    plane_rms=_plane_rms(cm),
                    clicked_level=record.board_meta.get("clicked_level"),
                    assumed_poses=assumed_poses,
                    figures=_list_figures(fig_dir),
                )
                if model_type == "pinhole":
                    # Intrinsics are pinhole-only; a polynomial has no K.
                    done.update(
                        fx=float(cm.K[0, 0]),
                        fy=float(cm.K[1, 1]),
                        cx=float(cm.K[0, 2]),
                        cy=float(cm.K[1, 2]),
                    )
                job_manager.complete_job(job_id, **done)
        except Exception as exc:
            logger.exception("stepped generate_model job %s failed", job_id)
            job_manager.fail_job(job_id, f"{type(exc).__name__}: {exc}")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "starting"})


@calibration_stepped_bp.route(
    "/calibration/stepped/generate_model/status/<job_id>", methods=["GET"]
)
def generate_model_status(job_id: str):
    """Poll a stepped model-generation job."""
    from .views import _job_status_response  # local: views imports this module first

    return _job_status_response(job_id)


def _list_figures(fig_dir) -> List[str]:
    if not fig_dir or not Path(fig_dir).is_dir():
        return []
    return sorted(
        p.name for p in Path(fig_dir).glob("*.png") if not p.name.startswith("._")
    )


def _model_rms(model) -> float:
    """Overall reprojection RMS (px) for a pinhole (``rms``) or poly3d (``rms_px``) model."""
    return float(getattr(model, "rms", None) if hasattr(model, "rms") else model.rms_px)


def _plane_rms(model) -> List[float]:
    """Per-plane reprojection RMS (px); only the 3D polynomial has it, else empty."""
    return [float(v) for v in getattr(model, "plane_rms_px", ())]
