"""
Stepped Board Calibration Views.

Routes for stepped dotboard stereo calibration:
- /calibrate/stepped_board/detect       - Start detection job (single frame, both cameras)
- /calibrate/stepped_board/detect/job/  - Poll detection job status
- /calibrate/stepped_board/snap_fiducial - Snap click to nearest blob (single-frame cache)
- /calibrate/stepped_board/generate_model - Start model generation job (single-frame)
- /calibrate/stepped_board/generate_model/job/ - Poll model generation status
- /calibrate/stepped_board/model        - Load saved model

Multi-view sequence routes (Piece 4 UX):
- /calibrate/stepped_board/detect_sequence        - Start detection across a range of frames
- /calibrate/stepped_board/detect_sequence/job/   - Poll sequence detection job
- /calibrate/stepped_board/sequence_pose_detection - Fetch detection data for a single pose
- /calibrate/stepped_board/identify_pose_level    - Snap click + identify which level (A/B)
- /calibrate/stepped_board/snap_fiducial_sequence - Snap click on datum pose of a sequence
- /calibrate/stepped_board/generate_model_sequence - Start multi-view model generation
- /calibrate/stepped_board/generate_model_sequence/job/ - Poll model generation
"""

import threading
import time
import uuid

import numpy as np
from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_gui.calibration.calibration_stepped.stepped_calibration_production import SteppedCalibrator
from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.calibration.app.shared_views import set_active_calibration_method
from pivtools_gui.utils import camera_number

stepped_board_bp = Blueprint("stepped_board", __name__)

# Single-frame detection cache: keyed by (source_path_idx, cam1, cam2, frame_idx)
_detection_cache = {}
_detection_cache_lock = threading.Lock()

# Multi-view sequence cache: keyed by sequence_id (UUID).
# Each entry holds:
#   'calibrator': SteppedCalibrator instance (shared — params/config constant)
#   'source_path_idx': int
#   'cam1': int, 'cam2': int
#   'frame_indices': list[int]    (1-based in the viewer's convention)
#   'datum_frame_idx': int        (1-based, must be in frame_indices)
#   'detections_per_pose': list[dict]  (one per frame, keyed by str(cam_num),
#                                       internal numpy arrays included)
#   'created_at': float           (unix time, for TTL cleanup)
_sequence_cache = {}
_sequence_cache_lock = threading.Lock()
_SEQUENCE_CACHE_TTL_SECONDS = 3600  # 1 hour


def _strip_internal_keys(det: dict) -> dict:
    """Return a JSON-safe copy of a detection dict, dropping _-prefixed keys."""
    return {k: v for k, v in det.items() if not k.startswith('_')}


def _gc_sequence_cache(now: float | None = None) -> None:
    """Evict stale sequence cache entries older than the TTL."""
    if now is None:
        now = time.time()
    with _sequence_cache_lock:
        stale = [
            sid for sid, entry in _sequence_cache.items()
            if now - entry.get('created_at', 0) > _SEQUENCE_CACHE_TTL_SECONDS
        ]
        for sid in stale:
            logger.info(f"Evicting stale sequence cache entry {sid}")
            _sequence_cache.pop(sid, None)


# ============================================================================
# ROUTE 1: Detect Dots (Start Job)
# ============================================================================


@stepped_board_bp.route("/calibrate/stepped_board/detect", methods=["POST"])
def stepped_board_detect():
    """Start stepped board detection for both cameras."""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    cam1 = camera_number(data.get("cam1", 1))
    cam2 = camera_number(data.get("cam2", 2))
    frame_idx = int(data.get("frame_idx", 0))

    set_active_calibration_method("stepped_board")

    job_id = job_manager.create_job(
        "stepped_board_detect",
        cam1=cam1,
        cam2=cam2,
        frame_idx=frame_idx,
        stage="starting",
    )

    def run_detection():
        try:
            job_manager.update_job(job_id, status="running", stage="detecting_cam1")

            cfg = get_config()
            calibrator = SteppedCalibrator(
                config=cfg,
                source_path_idx=source_path_idx,
                camera_pair=[cam1, cam2],
            )

            # Detect camera 1
            det1 = calibrator.detect_single_camera(cam1, frame_idx)
            job_manager.update_job(job_id, progress=50, stage="detecting_cam2")

            # Detect camera 2
            det2 = calibrator.detect_single_camera(cam2, frame_idx)

            # Cache detection data (including internal numpy arrays for model generation)
            cache_key = (source_path_idx, cam1, cam2, frame_idx)
            with _detection_cache_lock:
                _detection_cache.clear()
                _detection_cache[cache_key] = {
                    str(cam1): det1,
                    str(cam2): det2,
                    'calibrator': calibrator,
                }

            # Build JSON-safe result (strip internal numpy arrays)
            result = {}
            for cam_key, det in [(str(cam1), det1), (str(cam2), det2)]:
                result[cam_key] = {
                    'blobs': det['blobs'],
                    'level_A': det['level_A'],
                    'level_B': det['level_B'],
                    'image_size': det['image_size'],
                }

            job_manager.complete_job(job_id, detections=result)
            logger.info(f"Stepped board detection completed for cameras {cam1}, {cam2}")

        except Exception as e:
            logger.exception(f"Stepped board detection failed: {type(e).__name__}: {e}")
            job_manager.fail_job(job_id, f"{type(e).__name__}: {e}")

    thread = threading.Thread(target=run_detection)
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "starting",
    })


# ============================================================================
# ROUTE 2: Detection Job Status
# ============================================================================


@stepped_board_bp.route("/calibrate/stepped_board/detect/job/<job_id>", methods=["GET"])
def stepped_board_detect_status(job_id: str):
    """Get detection job status."""
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job_data)


# ============================================================================
# ROUTE 3: Snap Fiducial Click
# ============================================================================


@stepped_board_bp.route("/calibrate/stepped_board/snap_fiducial", methods=["POST"])
def stepped_board_snap_fiducial():
    """Snap a click to the nearest detected blob. Synchronous (fast)."""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    frame_idx = int(data.get("frame_idx", 0))
    click_x = float(data.get("click_x", 0))
    click_y = float(data.get("click_y", 0))

    # Find cached detection data for this camera using exact key
    cam1_val = camera_number(data.get("cam1", 1))
    cam2_val = camera_number(data.get("cam2", 2))
    cache_key = (source_path_idx, cam1_val, cam2_val, frame_idx)
    with _detection_cache_lock:
        cached = _detection_cache.get(cache_key, {})
        detection_data = cached.get(str(camera))

    if detection_data is None:
        return jsonify({"error": "No detection data cached. Run detection first."}), 400

    result = SteppedCalibrator.snap_to_nearest((click_x, click_y), detection_data)

    return jsonify(result)


# ============================================================================
# ROUTE 4: Load Saved Model
# ============================================================================


@stepped_board_bp.route("/calibrate/stepped_board/model", methods=["GET"])
def stepped_board_load_model():
    """Load saved stepped board model.

    Query params: base_path_idx, cam1, cam2
    """
    base_path_idx = request.args.get("base_path_idx", default=0, type=int)
    cam1 = camera_number(request.args.get("cam1", default=1, type=int))
    cam2 = camera_number(request.args.get("cam2", default=2, type=int))

    try:
        cfg = get_config()
        base_root = cfg.base_paths[base_path_idx]

        # Check per-camera models
        cam1_model_path = (
            base_root / "calibration" / f"Cam{cam1}"
            / "stepped_board" / "model" / "camera_model.mat"
        )
        cam2_model_path = (
            base_root / "calibration" / f"Cam{cam2}"
            / "stepped_board" / "model" / "camera_model.mat"
        )
        stereo_model_path = (
            base_root / "calibration" / f"stereo_cam{cam1}_cam{cam2}"
            / "model" / "stereo_model.mat"
        )

        if not cam1_model_path.exists() and not cam2_model_path.exists():
            return jsonify({"exists": False})

        result = {"exists": True}

        import scipy.io

        for cam_num, path in [(cam1, cam1_model_path), (cam2, cam2_model_path)]:
            if path.exists():
                mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
                K = mat["camera_matrix"]
                result[f"cam{cam_num}"] = {
                    "rms_error": float(mat.get("rms_error", 0)),
                    "focal_length": [float(K[0, 0]), float(K[1, 1])],
                    "principal_point": [float(K[0, 2]), float(K[1, 2])],
                    "distortion": mat["dist_coeffs"].flatten().tolist(),
                    "dot_spacing_mm": float(mat.get("dot_spacing_mm", 0)),
                }

        if stereo_model_path.exists():
            mat = scipy.io.loadmat(str(stereo_model_path), squeeze_me=True, struct_as_record=False)
            result["stereo"] = {
                "stereo_rms_error": float(mat.get("stereo_rms_error", 0)),
                "relative_angle_deg": float(mat.get("relative_angle_deg", 0)),
                "baseline_mm": float(np.linalg.norm(mat["translation_vector"])),
            }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error loading stepped board model: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# MULTI-VIEW SEQUENCE ROUTES (Piece 4)
# ============================================================================
#
# The sequence routes support capturing N frame pairs and running a joint
# multi-view fit against them, with a single fiducial click per camera on
# the operator-selected datum pose. This is how we break the fx<->tz ridge
# at PIV magnification.
#
# Flow:
#   1. POST /detect_sequence          → detect N frames, get sequence_id
#   2. GET  /detect_sequence/job/<id> → poll job
#   3. POST /snap_fiducial_sequence   × 6 (3 per camera on datum)
#   4. POST /generate_model_sequence  → joint multi-view fit
#   5. GET  /generate_model_sequence/job/<id> → poll job


def _make_sequence_id() -> str:
    return uuid.uuid4().hex


def _lookup_sequence(sequence_id: str):
    """Return the cache entry for sequence_id or (None, error_response)."""
    _gc_sequence_cache()
    with _sequence_cache_lock:
        entry = _sequence_cache.get(sequence_id)
    if entry is None:
        return None, (jsonify({
            "error": (
                f"Sequence {sequence_id} not found — it may have expired or "
                f"the backend was restarted. Re-run detect_sequence."
            ),
        }), 410)
    return entry, None


@stepped_board_bp.route("/calibrate/stepped_board/detect_sequence", methods=["POST"])
def stepped_board_detect_sequence():
    """Start multi-frame detection for both cameras across a frame range.

    Request JSON:
        source_path_idx: int
        cam1: int
        cam2: int
        num_frames: int             (how many frames to detect, >= 1)
        start_frame_idx: int        (1-based first frame of the range)
        datum_frame_idx: int        (1-based, must lie in the range)
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    cam1 = camera_number(data.get("cam1", 1))
    cam2 = camera_number(data.get("cam2", 2))
    num_frames = int(data.get("num_frames", 1))
    start_frame_idx = int(data.get("start_frame_idx", 1))
    datum_frame_idx = int(data.get("datum_frame_idx", start_frame_idx))

    if num_frames < 1:
        return jsonify({"error": "num_frames must be >= 1"}), 400
    frame_indices = list(range(start_frame_idx, start_frame_idx + num_frames))
    if datum_frame_idx not in frame_indices:
        return jsonify({
            "error": (
                f"datum_frame_idx {datum_frame_idx} must lie in the "
                f"frame range {frame_indices}"
            ),
        }), 400

    set_active_calibration_method("stepped_board")

    sequence_id = _make_sequence_id()
    job_id = job_manager.create_job(
        "stepped_board_detect_sequence",
        cam1=cam1, cam2=cam2,
        num_frames=num_frames,
        start_frame_idx=start_frame_idx,
        datum_frame_idx=datum_frame_idx,
        sequence_id=sequence_id,
        processed_frames=0,
        total_frames=num_frames,
        per_frame_status={},
        stage="starting",
    )

    def run_sequence_detection():
        per_frame_status: dict = {}
        try:
            job_manager.update_job(job_id, status="running", stage="detecting")
            cfg = get_config()
            calibrator = SteppedCalibrator(
                config=cfg,
                source_path_idx=source_path_idx,
                camera_pair=[cam1, cam2],
            )

            detections_per_pose = []
            for i, frame_idx in enumerate(frame_indices):
                pose_entry = {}
                pose_status = {'cam1': 'ok', 'cam2': 'ok'}

                for cam_num in (cam1, cam2):
                    tag = 'cam1' if cam_num == cam1 else 'cam2'
                    try:
                        det = calibrator.detect_single_camera(cam_num, frame_idx)
                        pose_entry[str(cam_num)] = det
                        pose_status[tag] = 'ok'
                    except Exception as exc:
                        logger.warning(
                            f"Sequence detection pose {frame_idx} cam{cam_num} "
                            f"failed: {exc}"
                        )
                        pose_entry[str(cam_num)] = None
                        pose_status[tag] = f'failed: {exc}'

                detections_per_pose.append(pose_entry)
                per_frame_status[str(frame_idx)] = pose_status

                job_manager.update_job(
                    job_id,
                    processed_frames=i + 1,
                    progress=int(((i + 1) / num_frames) * 90),
                    stage=f"detected_frame_{frame_idx}",
                    per_frame_status=per_frame_status,
                )

            with _sequence_cache_lock:
                _sequence_cache[sequence_id] = {
                    'calibrator': calibrator,
                    'source_path_idx': source_path_idx,
                    'cam1': cam1,
                    'cam2': cam2,
                    'frame_indices': frame_indices,
                    'datum_frame_idx': datum_frame_idx,
                    'detections_per_pose': detections_per_pose,
                    'created_at': time.time(),
                }

            # Build a JSON-safe per-pose detection summary for the frontend.
            pose_summaries = []
            for frame_idx, pose_entry in zip(frame_indices, detections_per_pose):
                summary = {
                    'frame_idx': frame_idx,
                    'is_datum': frame_idx == datum_frame_idx,
                    'cam1': None,
                    'cam2': None,
                }
                for cam_num in (cam1, cam2):
                    det = pose_entry.get(str(cam_num))
                    tag = 'cam1' if cam_num == cam1 else 'cam2'
                    if det is None:
                        summary[tag] = {'ok': False, 'error': per_frame_status[str(frame_idx)][tag]}
                    else:
                        summary[tag] = {
                            'ok': True,
                            'n_blobs': len(det.get('blobs', [])),
                            'n_level_A': det.get('level_A', {}).get('n_points', 0),
                            'n_level_B': det.get('level_B', {}).get('n_points', 0),
                            'image_size': det.get('image_size'),
                        }
                pose_summaries.append(summary)

            # Extract datum frame's full detection for frontend overlay.
            datum_pose_index = frame_indices.index(datum_frame_idx)
            datum_pose_entry = detections_per_pose[datum_pose_index]
            datum_detection = {}
            for cam_num in (cam1, cam2):
                tag = 'cam1' if cam_num == cam1 else 'cam2'
                det = datum_pose_entry.get(str(cam_num))
                datum_detection[tag] = _strip_internal_keys(det) if det else None

            job_manager.complete_job(
                job_id,
                sequence_id=sequence_id,
                frame_indices=frame_indices,
                datum_frame_idx=datum_frame_idx,
                poses=pose_summaries,
                per_frame_status=per_frame_status,
                datum_detection=datum_detection,
            )
            logger.info(
                f"Stepped sequence detection complete: sequence_id={sequence_id}, "
                f"{num_frames} frames, datum={datum_frame_idx}"
            )
        except Exception as e:
            logger.exception(f"Sequence detection failed: {e}")
            job_manager.fail_job(job_id, f"{type(e).__name__}: {e}")

    thread = threading.Thread(target=run_sequence_detection)
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "sequence_id": sequence_id,
        "status": "starting",
    })


@stepped_board_bp.route("/calibrate/stepped_board/detect_sequence/job/<job_id>", methods=["GET"])
def stepped_board_detect_sequence_status(job_id: str):
    """Poll a sequence detection job."""
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job_data)


@stepped_board_bp.route("/calibrate/stepped_board/sequence_pose_detection", methods=["GET"])
def stepped_board_sequence_pose_detection():
    """Fetch detection data for a single pose from a sequence cache.

    Query params:
        sequence_id: str
        frame_idx: int (1-based)
    Returns JSON: {cam1: {blobs, level_A, level_B, image_size}, cam2: {...}}
    """
    sequence_id = request.args.get("sequence_id")
    frame_idx = int(request.args.get("frame_idx", 0))
    if not sequence_id:
        return jsonify({"error": "sequence_id is required"}), 400
    entry, err = _lookup_sequence(sequence_id)
    if err is not None:
        return err

    frame_indices = entry['frame_indices']
    if frame_idx not in frame_indices:
        return jsonify({"error": f"frame_idx {frame_idx} not in sequence"}), 400

    pose_index = frame_indices.index(frame_idx)
    pose_entry = entry['detections_per_pose'][pose_index]
    cam1, cam2 = entry['cam1'], entry['cam2']

    result = {}
    for cam_num in (cam1, cam2):
        tag = 'cam1' if cam_num == cam1 else 'cam2'
        det = pose_entry.get(str(cam_num))
        result[tag] = _strip_internal_keys(det) if det else None
    return jsonify(result)


@stepped_board_bp.route("/calibrate/stepped_board/identify_pose_level", methods=["POST"])
def stepped_board_identify_pose_level():
    """Snap a click to the nearest blob and report which level it belongs to.

    Request JSON:
        sequence_id: str
        frame_idx: int (1-based)
        camera: int
        click_x: float
        click_y: float
    Returns: {level: 'A'|'B', snapped_x, snapped_y}
    """
    data = request.get_json() or {}
    sequence_id = data.get("sequence_id")
    frame_idx = int(data.get("frame_idx", 0))
    camera = camera_number(data.get("camera", 1))
    click_x = float(data.get("click_x", 0))
    click_y = float(data.get("click_y", 0))

    if not sequence_id:
        return jsonify({"error": "sequence_id is required"}), 400
    entry, err = _lookup_sequence(sequence_id)
    if err is not None:
        return err

    frame_indices = entry['frame_indices']
    if frame_idx not in frame_indices:
        return jsonify({"error": f"frame_idx {frame_idx} not in sequence"}), 400

    pose_index = frame_indices.index(frame_idx)
    det = entry['detections_per_pose'][pose_index].get(str(camera))
    if det is None:
        return jsonify({"error": f"No detection for camera {camera} on frame {frame_idx}"}), 400

    # Snap to nearest blob
    snap_result = SteppedCalibrator.snap_to_nearest((click_x, click_y), det)
    sx, sy = snap_result['snapped_x'], snap_result['snapped_y']

    # Determine which level the snapped point belongs to
    level_A_centers = det.get('level_A', {}).get('centers', [])
    level_B_centers = det.get('level_B', {}).get('centers', [])

    min_dist_A = float('inf')
    for c in level_A_centers:
        d = (c[0] - sx) ** 2 + (c[1] - sy) ** 2
        if d < min_dist_A:
            min_dist_A = d

    min_dist_B = float('inf')
    for c in level_B_centers:
        d = (c[0] - sx) ** 2 + (c[1] - sy) ** 2
        if d < min_dist_B:
            min_dist_B = d

    level = 'A' if min_dist_A <= min_dist_B else 'B'
    return jsonify({
        "level": level,
        "snapped_x": sx,
        "snapped_y": sy,
    })


@stepped_board_bp.route("/calibrate/stepped_board/snap_fiducial_sequence", methods=["POST"])
def stepped_board_snap_fiducial_sequence():
    """Snap a fiducial click against the datum pose of a sequence.

    Request JSON:
        sequence_id: str
        camera: int
        click_x: float
        click_y: float
    """
    data = request.get_json() or {}
    sequence_id = data.get("sequence_id")
    camera = camera_number(data.get("camera", 1))
    click_x = float(data.get("click_x", 0))
    click_y = float(data.get("click_y", 0))

    if not sequence_id:
        return jsonify({"error": "sequence_id is required"}), 400
    entry, err = _lookup_sequence(sequence_id)
    if err is not None:
        return err

    datum_frame_idx = entry['datum_frame_idx']
    frame_indices = entry['frame_indices']
    try:
        datum_pose_index = frame_indices.index(datum_frame_idx)
    except ValueError:
        return jsonify({"error": "datum_frame_idx no longer in frame_indices"}), 500

    det = entry['detections_per_pose'][datum_pose_index].get(str(camera))
    if det is None:
        return jsonify({
            "error": (
                f"Datum pose {datum_frame_idx} has no detection for camera "
                f"{camera} — detection likely failed. Re-run detect_sequence "
                f"or pick a different datum."
            ),
        }), 400

    result = SteppedCalibrator.snap_to_nearest((click_x, click_y), det)
    return jsonify(result)


@stepped_board_bp.route("/calibrate/stepped_board/generate_model_sequence", methods=["POST"])
def stepped_board_generate_model_sequence():
    """Start multi-view model generation from a detected sequence.

    Request JSON:
        sequence_id: str
        cam1_fiducials: {origin: [x, y], x_axis: [x, y], y_axis: [x, y]}
        cam2_fiducials: {...}
        stereo_config: 'auto' | 'same_side' | 'transmission'
            Default 'auto' — the backend fits cam2 twice (once per
            configuration) and picks the one with lower RMS. Explicit
            override still accepted but rarely needed.
        cam1_clicked_level: 'peak' | 'trough' — REQUIRED. Label of the
            face the operator's origin fiducial click landed on, cam1.
        cam2_clicked_level: 'peak' | 'trough' — REQUIRED. Same for cam2.
        cam1_pose_levels: {frame_idx: 'peak'|'trough'} — REQUIRED. Per-
            pose peak/trough label for cam1, keyed by 1-based frame
            number. One entry per frame in the sequence. Replaces the
            old dot-product auto-labeller.
        cam2_pose_levels: {frame_idx: 'peak'|'trough'} — REQUIRED. Same
            for cam2.
    """
    data = request.get_json() or {}
    sequence_id = data.get("sequence_id")
    cam1_fiducials = data.get("cam1_fiducials", {})
    cam2_fiducials = data.get("cam2_fiducials", {})
    stereo_config = data.get("stereo_config") or "auto"
    cam1_clicked_level = data.get("cam1_clicked_level")
    cam2_clicked_level = data.get("cam2_clicked_level")
    cam1_pose_levels_raw = data.get("cam1_pose_levels")
    cam2_pose_levels_raw = data.get("cam2_pose_levels")

    if not sequence_id:
        return jsonify({"error": "sequence_id is required"}), 400

    if cam1_clicked_level not in ("peak", "trough"):
        return jsonify({
            "error": "cam1_clicked_level must be 'peak' or 'trough'",
        }), 400
    if cam2_clicked_level not in ("peak", "trough"):
        return jsonify({
            "error": "cam2_clicked_level must be 'peak' or 'trough'",
        }), 400
    if not isinstance(cam1_pose_levels_raw, dict):
        return jsonify({
            "error": (
                "cam1_pose_levels is required — a dict keyed by frame_idx "
                "with 'peak' or 'trough' values. No auto-detect fallback."
            ),
        }), 400
    if not isinstance(cam2_pose_levels_raw, dict):
        return jsonify({
            "error": (
                "cam2_pose_levels is required — a dict keyed by frame_idx "
                "with 'peak' or 'trough' values. No auto-detect fallback."
            ),
        }), 400

    # Coerce JSON keys to int (they arrive as strings after JSON round-trip).
    try:
        cam1_pose_levels = {int(k): v for k, v in cam1_pose_levels_raw.items()}
        cam2_pose_levels = {int(k): v for k, v in cam2_pose_levels_raw.items()}
    except (TypeError, ValueError) as exc:
        return jsonify({
            "error": f"cam*_pose_levels keys must be integer frame indices: {exc}",
        }), 400

    # Validate fiducials shape
    for cam_label, fids in [("cam1", cam1_fiducials), ("cam2", cam2_fiducials)]:
        for key in ["origin", "x_axis", "y_axis"]:
            if key not in fids or fids[key] is None:
                return jsonify({"error": f"Missing fiducial '{key}' for {cam_label}"}), 400
            if not isinstance(fids[key], (list, tuple)) or len(fids[key]) != 2:
                return jsonify({"error": f"Fiducial '{key}' for {cam_label} must be [x, y]"}), 400

    entry, err = _lookup_sequence(sequence_id)
    if err is not None:
        return err

    calibrator = entry['calibrator']
    cam1 = entry['cam1']
    cam2 = entry['cam2']
    frame_indices = entry['frame_indices']
    datum_frame_idx = entry['datum_frame_idx']
    detections_per_pose = entry['detections_per_pose']

    # Validate every frame in the sequence has a label for both cameras.
    for f in frame_indices:
        if f not in cam1_pose_levels:
            return jsonify({
                "error": f"cam1_pose_levels missing frame_idx {f}",
            }), 400
        if f not in cam2_pose_levels:
            return jsonify({
                "error": f"cam2_pose_levels missing frame_idx {f}",
            }), 400
        if cam1_pose_levels[f] not in ("peak", "trough"):
            return jsonify({
                "error": (
                    f"cam1_pose_levels[{f}]={cam1_pose_levels[f]!r}, "
                    f"expected 'peak' or 'trough'"
                ),
            }), 400
        if cam2_pose_levels[f] not in ("peak", "trough"):
            return jsonify({
                "error": (
                    f"cam2_pose_levels[{f}]={cam2_pose_levels[f]!r}, "
                    f"expected 'peak' or 'trough'"
                ),
            }), 400

    try:
        datum_pose_index = frame_indices.index(datum_frame_idx)
    except ValueError:
        return jsonify({"error": "datum_frame_idx no longer in frame_indices"}), 500

    set_active_calibration_method("stepped_board")

    job_id = job_manager.create_job(
        "stepped_board_sequence_model",
        sequence_id=sequence_id,
        cam1=cam1, cam2=cam2,
        num_poses=len(frame_indices),
        datum_frame_idx=datum_frame_idx,
        stage="starting",
    )

    def run_model_generation():
        try:
            job_manager.update_job(job_id, status="running", stage="fitting")
            fiducials = {str(cam1): cam1_fiducials, str(cam2): cam2_fiducials}
            params = {
                'stereo_config': stereo_config,
                'cam1_clicked_level': cam1_clicked_level,
                'cam2_clicked_level': cam2_clicked_level,
                'cam1_pose_levels': cam1_pose_levels,
                'cam2_pose_levels': cam2_pose_levels,
                'frame_indices': frame_indices,
            }

            def progress_callback(pd):
                job_manager.update_job(
                    job_id,
                    progress=pd.get("progress", 0),
                    stage=pd.get("stage", "fitting"),
                )

            result = calibrator.generate_model(
                detections_per_pose, fiducials, params,
                datum_pose_index=datum_pose_index,
                progress_callback=progress_callback,
            )

            if result.get('success'):
                job_manager.complete_job(
                    job_id,
                    cam1_rms=result.get('cam1_rms'),
                    cam2_rms=result.get('cam2_rms'),
                    stereo_rms=result.get('stereo_rms'),
                    warnings=result.get('warnings') or [],
                    relative_angle_deg=result.get('relative_angle_deg'),
                    baseline_mm=result.get('baseline_mm'),
                    cam1_details=result.get('cam1_details'),
                    cam2_details=result.get('cam2_details'),
                    num_poses_total=result.get('num_poses_total'),
                    cam1_poses_used=result.get('cam1_poses_used'),
                    cam2_poses_used=result.get('cam2_poses_used'),
                    cam1_clicked_level_resolved=result.get('cam1_clicked_level_resolved'),
                    cam2_clicked_level_resolved=result.get('cam2_clicked_level_resolved'),
                    stereo_config_resolved=result.get('stereo_config_resolved'),
                    stereo_config_rms_same_side=result.get('stereo_config_rms_same_side'),
                    stereo_config_rms_transmission=result.get('stereo_config_rms_transmission'),
                )
                logger.info(
                    f"Stepped sequence model generated: "
                    f"cam1 fx={result['cam1_details']['focal_length'][0]:.1f}, "
                    f"cam2 fx={result['cam2_details']['focal_length'][0]:.1f}"
                )
            else:
                job_manager.fail_job(job_id, result.get('error', 'Model generation failed'))
        except Exception as e:
            logger.exception(f"Sequence model generation failed: {e}")
            job_manager.fail_job(job_id, f"{type(e).__name__}: {e}")

    thread = threading.Thread(target=run_model_generation)
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "starting"})


@stepped_board_bp.route("/calibrate/stepped_board/generate_model_sequence/job/<job_id>", methods=["GET"])
def stepped_board_generate_model_sequence_status(job_id: str):
    """Poll a sequence model-generation job."""
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job_data)
