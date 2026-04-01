"""
Stepped Board Calibration Views.

Routes for stepped dotboard stereo calibration:
- /calibrate/stepped_board/detect       - Start detection job (both cameras)
- /calibrate/stepped_board/detect/job/  - Poll detection job status
- /calibrate/stepped_board/snap_fiducial - Snap click to nearest blob
- /calibrate/stepped_board/generate_model - Start model generation job
- /calibrate/stepped_board/generate_model/job/ - Poll model generation status
- /calibrate/stepped_board/model        - Load saved model
"""

import threading

import numpy as np
from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_gui.calibration.calibration_stepped.stepped_calibration_production import SteppedCalibrator
from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.calibration.app.shared_views import set_active_calibration_method
from pivtools_gui.utils import camera_number

stepped_board_bp = Blueprint("stepped_board", __name__)

# Detection cache: keyed by (source_path_idx, cam1, cam2, frame_idx)
_detection_cache = {}
_detection_cache_lock = threading.Lock()


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
# ROUTE 4: Generate Model (Start Job)
# ============================================================================


@stepped_board_bp.route("/calibrate/stepped_board/generate_model", methods=["POST"])
def stepped_board_generate_model():
    """Start model generation from detections and fiducials."""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    cam1 = camera_number(data.get("cam1", 1))
    cam2 = camera_number(data.get("cam2", 2))
    frame_idx = int(data.get("frame_idx", 0))

    cam1_fiducials = data.get("cam1_fiducials", {})
    cam2_fiducials = data.get("cam2_fiducials", {})
    stereo_config = data.get("stereo_config", "transmission")
    cam1_clicked_level = data.get("cam1_clicked_level") or "peak"
    cam2_clicked_level = data.get("cam2_clicked_level") or "peak"

    # Validate fiducials
    for cam_label, fids in [("cam1", cam1_fiducials), ("cam2", cam2_fiducials)]:
        for key in ["origin", "x_axis", "y_axis"]:
            if key not in fids or fids[key] is None:
                return jsonify({"error": f"Missing fiducial '{key}' for {cam_label}"}), 400
            if not isinstance(fids[key], (list, tuple)) or len(fids[key]) != 2:
                return jsonify({"error": f"Fiducial '{key}' for {cam_label} must be [x, y]"}), 400

    set_active_calibration_method("stepped_board")

    job_id = job_manager.create_job(
        "stepped_board_model",
        cam1=cam1,
        cam2=cam2,
        stage="starting",
    )

    def run_model_generation():
        try:
            job_manager.update_job(job_id, status="running", stage="loading_detections")

            # Get cached detection data — run detection implicitly if not cached
            with _detection_cache_lock:
                cache_key = (source_path_idx, cam1, cam2, frame_idx)
                cached = _detection_cache.get(cache_key)

            if cached is None:
                logger.info("No cached detection data — running detection implicitly")
                job_manager.update_job(job_id, stage="detecting_cam1")

                cfg = get_config()
                calibrator = SteppedCalibrator(
                    config=cfg,
                    source_path_idx=source_path_idx,
                    camera_pair=[cam1, cam2],
                )

                det1 = calibrator.detect_single_camera(cam1, frame_idx)
                job_manager.update_job(job_id, progress=15, stage="detecting_cam2")
                det2 = calibrator.detect_single_camera(cam2, frame_idx)

                with _detection_cache_lock:
                    _detection_cache.clear()
                    _detection_cache[cache_key] = {
                        str(cam1): det1,
                        str(cam2): det2,
                        'calibrator': calibrator,
                    }
                    cached = _detection_cache[cache_key]
                logger.info("Implicit detection completed — proceeding to model generation")

            calibrator = cached.get('calibrator')
            if calibrator is None:
                job_manager.fail_job(job_id, "Calibrator not found in cache.")
                return

            detections = {
                str(cam1): cached[str(cam1)],
                str(cam2): cached[str(cam2)],
            }

            fiducials = {
                str(cam1): cam1_fiducials,
                str(cam2): cam2_fiducials,
            }

            params = {
                'stereo_config': stereo_config,
                'cam1_clicked_level': cam1_clicked_level,
                'cam2_clicked_level': cam2_clicked_level,
            }

            def progress_callback(progress_data):
                job_manager.update_job(
                    job_id,
                    progress=progress_data.get("progress", 0),
                    stage=progress_data.get("stage", "generating"),
                )

            result = calibrator.generate_model(
                detections, fiducials, params, progress_callback=progress_callback
            )

            if result.get('success'):
                job_manager.complete_job(
                    job_id,
                    cam1_rms=result.get('cam1_rms'),
                    cam2_rms=result.get('cam2_rms'),
                    stereo_rms=result.get('stereo_rms'),
                    relative_angle_deg=result.get('relative_angle_deg'),
                    baseline_mm=result.get('baseline_mm'),
                    cam1_details=result.get('cam1_details'),
                    cam2_details=result.get('cam2_details'),
                )
                logger.info(f"Stepped board model generated for cameras {cam1}-{cam2}")
            else:
                job_manager.fail_job(job_id, result.get('error', 'Model generation failed'))

        except Exception as e:
            logger.exception(f"Stepped board model generation failed: {type(e).__name__}: {e}")
            job_manager.fail_job(job_id, f"{type(e).__name__}: {e}")

    thread = threading.Thread(target=run_model_generation)
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "starting",
    })


# ============================================================================
# ROUTE 5: Model Generation Job Status
# ============================================================================


@stepped_board_bp.route("/calibrate/stepped_board/generate_model/job/<job_id>", methods=["GET"])
def stepped_board_model_status(job_id: str):
    """Get model generation job status."""
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job_data)


# ============================================================================
# ROUTE 6: Load Saved Model
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
