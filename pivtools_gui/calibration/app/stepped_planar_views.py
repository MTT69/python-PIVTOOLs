"""Stepped Planar Calibration Views.

Flask routes for per-camera 3D calibration using both Z levels of a
stepped board. Mirrors the stepped stereo sequence flow but emits
per-camera pinhole models without stereo composition.

Flow (per camera):
  1. POST /detect_sequence          → detect N frames for one camera, get sequence_id
  2. GET  /detect_sequence/job/<id> → poll detection progress
  2b. GET /sequence_pose_detection  → fetch detection data for a single pose
  2c. POST /identify_pose_level    → snap click + identify which level (A/B)
  3. POST /snap_fiducial            → snap click to nearest dot (×3 for origin/x/y)
  4. POST /generate_camera_model    → joint multi-view pinhole fit for one camera
  5. GET  /generate_camera_model/job/<id> → poll fit progress

Multi-camera is handled by calling `/detect_sequence` + `/generate_camera_model`
once per camera, either from the client sequentially or from the
convenience route `/generate_camera_model_all` (parallel thread pool).
Each camera has its own sequence_id and its own fiducial clicks and
peak/trough declaration — this preserves per-camera mounting flexibility
and matches how the stepped stereo flow handles per-camera state.
"""
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_gui.calibration.calibration_stepped.stepped_planar_calibrator import (
    SteppedPlanarCalibrator,
)
from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.calibration.app.shared_views import set_active_calibration_method
from pivtools_gui.utils import camera_number

stepped_planar_bp = Blueprint("stepped_planar", __name__)


# ----------------------------------------------------------------------
# Sequence cache — same shape as stepped_board_views but per-camera.
# ----------------------------------------------------------------------

_SEQUENCE_TTL_SECONDS = 2 * 60 * 60  # 2 hours

_sequence_cache: dict = {}
_sequence_cache_lock = threading.Lock()


def _gc_sequence_cache(now: float | None = None) -> None:
    """Drop sequence cache entries older than the TTL."""
    if now is None:
        now = time.time()
    with _sequence_cache_lock:
        stale = [
            sid for sid, entry in _sequence_cache.items()
            if now - entry.get('created_at', now) > _SEQUENCE_TTL_SECONDS
        ]
        for sid in stale:
            _sequence_cache.pop(sid, None)


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
                f"Sequence {sequence_id} not found — it may have expired "
                f"or the backend was restarted. Re-run detect_sequence."
            ),
        }), 410)
    return entry, None


def _strip_internal_keys(det: dict) -> dict:
    """Return a JSON-safe copy of a detection dict, dropping _-prefixed keys."""
    return {k: v for k, v in det.items() if not k.startswith('_')}


def _make_calibrator(cfg, source_path_idx: int) -> SteppedPlanarCalibrator:
    """Build a stepped planar calibrator from config. Camera number is
    passed per-call (not stored on the instance) so one calibrator can
    process any camera."""
    return SteppedPlanarCalibrator(
        config=cfg,
        source_path_idx=source_path_idx,
    )


# ============================================================================
# ROUTE 1: Detect Sequence (one camera, N frames)
# ============================================================================


@stepped_planar_bp.route("/calibration/stepped_planar/detect_sequence", methods=["POST"])
def stepped_planar_detect_sequence():
    """Run multi-frame detection for one camera across a frame range.

    Request JSON:
        source_path_idx: int
        camera: int
        num_frames: int             (how many frames to detect, >= 1)
        start_frame_idx: int        (1-based first frame of the range)
        datum_frame_idx: int        (1-based, must lie in the range)

    Returns:
        {job_id, sequence_id}
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
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

    set_active_calibration_method("stepped_planar")

    sequence_id = _make_sequence_id()
    job_id = job_manager.create_job(
        "stepped_planar_detect_sequence",
        camera=camera,
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
            calibrator = _make_calibrator(cfg, source_path_idx)

            detections_per_pose: list = []
            for i, frame_idx in enumerate(frame_indices):
                pose_entry: dict = {}
                try:
                    det = calibrator.detect_single_camera(camera, frame_idx)
                    pose_entry[str(camera)] = det
                    per_frame_status[str(frame_idx)] = 'ok'
                except Exception as exc:
                    logger.warning(
                        f"Stepped-planar detection pose {frame_idx} cam{camera} "
                        f"failed: {exc}"
                    )
                    pose_entry[str(camera)] = None
                    per_frame_status[str(frame_idx)] = f'failed: {exc}'

                detections_per_pose.append(pose_entry)

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
                    'camera': camera,
                    'frame_indices': frame_indices,
                    'datum_frame_idx': datum_frame_idx,
                    'detections_per_pose': detections_per_pose,
                    'created_at': time.time(),
                }

            # Build a JSON-safe per-pose summary.
            pose_summaries = []
            for frame_idx, pose_entry in zip(frame_indices, detections_per_pose):
                det = pose_entry.get(str(camera))
                summary = {
                    'frame_idx': frame_idx,
                    'is_datum': frame_idx == datum_frame_idx,
                    'ok': det is not None,
                }
                if det is not None:
                    summary.update({
                        'n_blobs': len(det.get('blobs', [])),
                        'n_level_A': det.get('level_A', {}).get('n_points', 0),
                        'n_level_B': det.get('level_B', {}).get('n_points', 0),
                        'image_size': det.get('image_size'),
                    })
                else:
                    summary['error'] = per_frame_status[str(frame_idx)]
                pose_summaries.append(summary)

            # Extract datum frame's full detection for frontend overlay.
            datum_pose_index = frame_indices.index(datum_frame_idx)
            datum_det = detections_per_pose[datum_pose_index].get(str(camera))
            datum_detection = _strip_internal_keys(datum_det) if datum_det else None

            job_manager.complete_job(
                job_id,
                sequence_id=sequence_id,
                camera=camera,
                frame_indices=frame_indices,
                datum_frame_idx=datum_frame_idx,
                poses=pose_summaries,
                per_frame_status=per_frame_status,
                datum_detection=datum_detection,
            )
            logger.info(
                f"Stepped-planar sequence detection complete: camera={camera}, "
                f"sequence_id={sequence_id}, {num_frames} frames, "
                f"datum={datum_frame_idx}"
            )
        except Exception as e:
            logger.exception(f"Stepped-planar sequence detection failed: {e}")
            job_manager.fail_job(job_id, f"{type(e).__name__}: {e}")

    thread = threading.Thread(target=run_sequence_detection)
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "sequence_id": sequence_id,
        "status": "starting",
    })


@stepped_planar_bp.route(
    "/calibration/stepped_planar/detect_sequence/job/<job_id>",
    methods=["GET"],
)
def stepped_planar_detect_sequence_status(job_id: str):
    """Poll a stepped-planar sequence detection job."""
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job_data)


# ============================================================================
# ROUTE 2b: Fetch per-pose detection data + identify level
# ============================================================================


@stepped_planar_bp.route("/calibration/stepped_planar/sequence_pose_detection", methods=["GET"])
def stepped_planar_sequence_pose_detection():
    """Fetch detection data for a single pose from a sequence cache.

    Query params:
        sequence_id: str
        frame_idx: int (1-based)
    Returns JSON: {blobs, level_A, level_B, image_size}
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

    camera = entry['camera']
    pose_index = frame_indices.index(frame_idx)
    det = entry['detections_per_pose'][pose_index].get(str(camera))
    if det is None:
        return jsonify({"error": f"No detection for frame {frame_idx}"}), 400
    return jsonify(_strip_internal_keys(det))


@stepped_planar_bp.route("/calibration/stepped_planar/identify_pose_level", methods=["POST"])
def stepped_planar_identify_pose_level():
    """Snap a click to the nearest blob and report which level it belongs to.

    Request JSON:
        sequence_id: str
        frame_idx: int (1-based)
        click_x: float
        click_y: float
    Returns: {level: 'A'|'B', snapped_x, snapped_y}
    """
    data = request.get_json() or {}
    sequence_id = data.get("sequence_id")
    frame_idx = int(data.get("frame_idx", 0))
    click_x = float(data.get("click_x", 0))
    click_y = float(data.get("click_y", 0))

    if not sequence_id:
        return jsonify({"error": "sequence_id is required"}), 400
    entry, err = _lookup_sequence(sequence_id)
    if err is not None:
        return err

    camera = entry['camera']
    frame_indices = entry['frame_indices']
    if frame_idx not in frame_indices:
        return jsonify({"error": f"frame_idx {frame_idx} not in sequence"}), 400

    pose_index = frame_indices.index(frame_idx)
    det = entry['detections_per_pose'][pose_index].get(str(camera))
    if det is None:
        return jsonify({"error": f"No detection for frame {frame_idx}"}), 400

    snap_result = SteppedPlanarCalibrator.snap_to_nearest((click_x, click_y), det)
    sx, sy = snap_result['snapped_x'], snap_result['snapped_y']

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


# ============================================================================
# ROUTE 3: Snap Fiducial Click
# ============================================================================


@stepped_planar_bp.route("/calibration/stepped_planar/snap_fiducial", methods=["POST"])
def stepped_planar_snap_fiducial():
    """Snap a fiducial click against the datum pose of a sequence.

    Request JSON:
        sequence_id: str
        click_x: float
        click_y: float
    """
    data = request.get_json() or {}
    sequence_id = data.get("sequence_id")
    click_x = float(data.get("click_x", 0))
    click_y = float(data.get("click_y", 0))

    if not sequence_id:
        return jsonify({"error": "sequence_id is required"}), 400
    entry, err = _lookup_sequence(sequence_id)
    if err is not None:
        return err

    camera = entry['camera']
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
                f"{camera} — detection likely failed. Re-run detect_sequence."
            ),
        }), 400

    result = SteppedPlanarCalibrator.snap_to_nearest((click_x, click_y), det)
    return jsonify(result)


# ============================================================================
# ROUTE 4: Generate Camera Model (from a sequence)
# ============================================================================


@stepped_planar_bp.route(
    "/calibration/stepped_planar/generate_camera_model",
    methods=["POST"],
)
def stepped_planar_generate_camera_model():
    """Start per-camera stepped-planar model generation from a detected
    sequence.

    Request JSON:
        sequence_id: str
        fiducials: {origin: [x, y], x_axis: [x, y], y_axis: [x, y]}
        clicked_level: 'peak' | 'trough'
        pose_levels: {frame_idx: 'peak'|'trough'}  — REQUIRED. Per-pose
            label for THIS camera; one entry per frame in the sequence.
    """
    data = request.get_json() or {}
    sequence_id = data.get("sequence_id")
    fiducials = data.get("fiducials", {})
    clicked_level = data.get("clicked_level")
    pose_levels_raw = data.get("pose_levels")

    if not sequence_id:
        return jsonify({"error": "sequence_id is required"}), 400
    if clicked_level not in ("peak", "trough"):
        return jsonify({
            "error": "clicked_level must be 'peak' or 'trough'",
        }), 400
    if not isinstance(pose_levels_raw, dict):
        return jsonify({
            "error": (
                "pose_levels is required — a dict keyed by frame_idx "
                "with 'peak' or 'trough' values. No auto-detect fallback."
            ),
        }), 400
    try:
        pose_levels = {int(k): v for k, v in pose_levels_raw.items()}
    except (TypeError, ValueError) as exc:
        return jsonify({
            "error": f"pose_levels keys must be integer frame indices: {exc}",
        }), 400
    for key in ("origin", "x_axis", "y_axis"):
        if key not in fiducials or fiducials[key] is None:
            return jsonify({"error": f"Missing fiducial '{key}'"}), 400
        if not isinstance(fiducials[key], (list, tuple)) or len(fiducials[key]) != 2:
            return jsonify({"error": f"Fiducial '{key}' must be [x, y]"}), 400

    entry, err = _lookup_sequence(sequence_id)
    if err is not None:
        return err

    calibrator = entry['calibrator']
    camera = entry['camera']
    frame_indices = entry['frame_indices']
    datum_frame_idx = entry['datum_frame_idx']
    detections_per_pose = entry['detections_per_pose']

    for f in frame_indices:
        if f not in pose_levels:
            return jsonify({
                "error": f"pose_levels missing frame_idx {f}",
            }), 400
        if pose_levels[f] not in ("peak", "trough"):
            return jsonify({
                "error": (
                    f"pose_levels[{f}]={pose_levels[f]!r}, "
                    f"expected 'peak' or 'trough'"
                ),
            }), 400

    try:
        datum_pose_index = frame_indices.index(datum_frame_idx)
    except ValueError:
        return jsonify({"error": "datum_frame_idx no longer in frame_indices"}), 500

    set_active_calibration_method("stepped_planar")

    job_id = job_manager.create_job(
        "stepped_planar_generate_camera_model",
        sequence_id=sequence_id,
        camera=camera,
        num_poses=len(frame_indices),
        datum_frame_idx=datum_frame_idx,
        stage="starting",
    )

    def run_model_generation():
        try:
            job_manager.update_job(job_id, status="running", stage="fitting")

            def progress_callback(pd):
                job_manager.update_job(
                    job_id,
                    progress=pd.get("progress", 0),
                    stage=pd.get("stage", "fitting"),
                )

            result = calibrator.generate_camera_model(
                cam_num=camera,
                detections_per_pose=detections_per_pose,
                fiducials_for_camera=fiducials,
                clicked_level=clicked_level,
                frame_indices=frame_indices,
                pose_levels=pose_levels,
                datum_pose_index=datum_pose_index,
                progress_callback=progress_callback,
            )

            if result.get('success'):
                job_manager.complete_job(
                    job_id,
                    camera=result.get('cam_num'),
                    rms=result.get('rms'),
                    K=result.get('K'),
                    dist=result.get('dist'),
                    num_poses=result.get('num_poses'),
                    pose_indices=result.get('pose_indices'),
                    model_path=result.get('model_path'),
                )
                logger.info(
                    f"Stepped-planar camera {camera} model generated: "
                    f"rms={result.get('rms'):.4f}px, "
                    f"num_poses={result.get('num_poses')}"
                )
            else:
                job_manager.fail_job(
                    job_id, result.get('error', 'Model generation failed'),
                )
        except Exception as e:
            logger.exception(f"Stepped-planar model generation failed: {e}")
            job_manager.fail_job(job_id, f"{type(e).__name__}: {e}")

    thread = threading.Thread(target=run_model_generation)
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "starting"})


@stepped_planar_bp.route(
    "/calibration/stepped_planar/generate_camera_model/job/<job_id>",
    methods=["GET"],
)
def stepped_planar_generate_camera_model_status(job_id: str):
    """Poll a stepped-planar camera-model generation job."""
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job_data)


# ============================================================================
# ROUTE 5: Load Saved Model
# ============================================================================


@stepped_planar_bp.route("/calibration/stepped_planar/model", methods=["GET"])
def stepped_planar_load_model():
    """Load a saved per-camera stepped-planar pinhole model.

    Query params:
        source_path_idx: int
        camera: int

    Returns the camera model fields in the same shape as stepped_board
    loading, minus the stereo-specific fields.
    """
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])
        model_path = (
            base_root / "calibration" / f"Cam{camera}"
            / "stepped_planar" / "model" / "camera_model.mat"
        )
        if not model_path.exists():
            return jsonify({
                "exists": False,
                "message": f"No saved stepped-planar model for camera {camera}",
            })

        model_data = scipy.io.loadmat(
            str(model_path), struct_as_record=False, squeeze_me=True
        )
        K = np.asarray(model_data["camera_matrix"])
        camera_model = {
            "model_type": "pinhole",
            "camera_matrix": K.tolist(),
            "dist_coeffs": np.asarray(model_data["dist_coeffs"]).flatten().tolist(),
            "reprojection_error": float(model_data.get("rms_error", 0)),
            "focal_length": [float(K[0, 0]), float(K[1, 1])],
            "principal_point": [float(K[0, 2]), float(K[1, 2])],
            "num_poses": int(model_data.get("num_poses", 0)),
            "image_width": int(model_data.get("image_width", 0)),
            "image_height": int(model_data.get("image_height", 0)),
        }

        return jsonify({
            "exists": True,
            "camera_model": camera_model,
        })
    except Exception as e:
        logger.exception(f"Error loading stepped_planar model: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROUTE 6: Generate Camera Model for All Cameras (parallel)
# ============================================================================


@stepped_planar_bp.route(
    "/calibration/stepped_planar/generate_camera_model_all",
    methods=["POST"],
)
def stepped_planar_generate_camera_model_all():
    """Run stepped-planar model generation on every configured camera
    in parallel, each from its own sequence_id + fiducials.

    Request JSON:
        per_camera: {
            "1": {sequence_id, fiducials, clicked_level},
            "2": {sequence_id, fiducials, clicked_level},
            ...
        }

    This assumes the client has already run `/detect_sequence` once per
    camera and snapped fiducials on each datum — same pattern as stepped
    stereo, just per-camera.
    """
    data = request.get_json() or {}
    per_camera = data.get("per_camera", {})
    if not per_camera:
        return jsonify({"error": "per_camera is required"}), 400

    # Validate
    validated: list = []
    for cam_key, spec in per_camera.items():
        try:
            cam_num = camera_number(cam_key)
        except Exception:
            return jsonify({"error": f"Invalid camera key {cam_key!r}"}), 400
        sid = spec.get("sequence_id")
        fids = spec.get("fiducials", {})
        cl = spec.get("clicked_level")
        pose_levels_raw = spec.get("pose_levels")
        if not sid:
            return jsonify({"error": f"camera {cam_num}: sequence_id missing"}), 400
        if cl not in ("peak", "trough"):
            return jsonify({
                "error": f"camera {cam_num}: clicked_level must be peak or trough",
            }), 400
        if not isinstance(pose_levels_raw, dict):
            return jsonify({
                "error": (
                    f"camera {cam_num}: pose_levels is required (dict "
                    f"keyed by frame_idx, values 'peak' or 'trough')"
                ),
            }), 400
        try:
            pose_levels = {int(k): v for k, v in pose_levels_raw.items()}
        except (TypeError, ValueError) as exc:
            return jsonify({
                "error": f"camera {cam_num}: pose_levels keys must be integers: {exc}",
            }), 400
        for key in ("origin", "x_axis", "y_axis"):
            if key not in fids or fids[key] is None:
                return jsonify({
                    "error": f"camera {cam_num}: missing fiducial '{key}'",
                }), 400
        validated.append((cam_num, sid, fids, cl, pose_levels))

    set_active_calibration_method("stepped_planar")

    cameras = [cam for cam, _, _, _, _ in validated]
    job_id = job_manager.create_job(
        "stepped_planar_generate_camera_model_all",
        processed_cameras=0,
        total_cameras=len(cameras),
        current_camera=None,
        camera_results={},
    )

    def process_camera(cam_num, sequence_id, fiducials, clicked_level, pose_levels):
        entry, err = _lookup_sequence(sequence_id)
        if err is not None:
            return cam_num, {"success": False, "error": (
                f"sequence {sequence_id} expired or not found"
            )}

        calibrator = entry['calibrator']
        frame_indices = entry['frame_indices']
        datum_frame_idx = entry['datum_frame_idx']
        detections_per_pose = entry['detections_per_pose']
        try:
            datum_pose_index = frame_indices.index(datum_frame_idx)
        except ValueError:
            return cam_num, {
                "success": False,
                "error": "datum_frame_idx no longer in frame_indices",
            }

        for f in frame_indices:
            if f not in pose_levels or pose_levels[f] not in ("peak", "trough"):
                return cam_num, {
                    "success": False,
                    "error": (
                        f"camera {cam_num}: pose_levels missing or invalid "
                        f"entry for frame_idx {f}"
                    ),
                }

        def progress_callback(pd, cam_num=cam_num):
            job_manager.update_job(
                job_id,
                current_camera=cam_num,
                current_camera_progress=pd.get("progress", 0),
            )

        result = calibrator.generate_camera_model(
            cam_num=cam_num,
            detections_per_pose=detections_per_pose,
            fiducials_for_camera=fiducials,
            clicked_level=clicked_level,
            frame_indices=frame_indices,
            pose_levels=pose_levels,
            datum_pose_index=datum_pose_index,
            progress_callback=progress_callback,
        )
        return cam_num, result

    def run_all():
        try:
            camera_results: dict = {}
            job_manager.update_job(job_id, status="running")
            with ThreadPoolExecutor(max_workers=max(1, len(cameras))) as executor:
                futures = {
                    executor.submit(
                        process_camera, cam_num, sid, fids, cl, pl,
                    ): cam_num
                    for (cam_num, sid, fids, cl, pl) in validated
                }
                for future in as_completed(futures):
                    cam_num = futures[future]
                    try:
                        _, result = future.result()
                        if result.get("success"):
                            camera_results[cam_num] = {
                                "status": "completed",
                                "rms": result.get("rms"),
                                "num_poses": result.get("num_poses"),
                                "model_path": result.get("model_path"),
                            }
                            logger.info(
                                f"Stepped-planar camera {cam_num} completed"
                            )
                        else:
                            camera_results[cam_num] = {
                                "status": "failed",
                                "error": result.get("error", "Unknown error"),
                            }
                    except Exception as e:
                        logger.error(f"Stepped-planar camera {cam_num} failed: {e}")
                        camera_results[cam_num] = {
                            "status": "failed", "error": str(e),
                        }

                    completed = sum(
                        1 for r in camera_results.values()
                        if r["status"] in ("completed", "failed")
                    )
                    job_manager.update_job(
                        job_id,
                        processed_cameras=completed,
                        camera_results=camera_results,
                    )

            job_manager.complete_job(
                job_id,
                camera_results=camera_results,
                processed_cameras=len(cameras),
            )
        except Exception as e:
            logger.exception(
                f"Stepped-planar multi-camera job {job_id} failed: {e}"
            )
            job_manager.fail_job(job_id, str(e))

    thread = threading.Thread(target=run_all)
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "starting",
        "cameras": cameras,
    })
