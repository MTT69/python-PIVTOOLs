"""
ChArUco Calibration Views.

Provides Flask endpoints for ChArUco board camera calibration with progress tracking.
Uses ChArUcoCalibrator service for actual calibration logic.
"""

import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_core.batch_utils import iter_batch_targets
from pivtools_core.image_handling.calibration_loader import (
    read_calibration_image,
    validate_calibration_images,
)
from pivtools_core.image_handling.path_utils import build_calibration_camera_path

from pivtools_gui.calibration.calibration_charuco import ChArUcoCalibrator
from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.utils import camera_number 

charuco_bp = Blueprint("charuco", __name__)


@charuco_bp.route("/calibration/charuco/validate_images", methods=["POST"])
def charuco_validate_images():
    """
    Validate ChArUco calibration images exist and are readable.

    Uses the unified calibration_loader for consistent path handling.
    Note: Frontend now uses usePinholeValidation which calls the unified
    /calibration/validate_images endpoint. This endpoint is kept for
    backward compatibility.

    Request JSON:
        source_path_idx: int
        camera: int
        image_format: str (optional - overrides config)
        num_images: int (optional - overrides config)
        subfolder: str (optional - overrides config)

    Returns:
        JSON with validation status, file count, preview image
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))

    # Extract optional override parameters
    image_format = data.get("image_format")
    num_images = data.get("num_images")
    if num_images is not None:
        num_images = int(num_images)
    subfolder = data.get("subfolder")

    try:
        cfg = get_config()

        # Use unified validation function with optional overrides
        result = validate_calibration_images(
            camera,
            cfg,
            source_path_idx,
            image_format=image_format,
            num_images=num_images,
            subfolder=subfolder,
        )

        # Add extra context
        result["camera"] = camera
        result["source_path_idx"] = source_path_idx
        result["checked"] = True

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error validating ChArUco images: {e}")
        return jsonify({
            "valid": False,
            "checked": True,
            "found_count": 0,
            "error": str(e),
        }), 500


@charuco_bp.route("/calibration/charuco/calibrate", methods=["POST"])
def charuco_calibrate():
    """
    Start ChArUco calibration job with progress tracking.

    Uses unified calibration config for path handling.
    Board parameters are read from config (saved by frontend auto-save).

    Request JSON:
        source_path_idx: int
        camera: int

    Returns:
        JSON with job_id, status, message
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))

    cfg = get_config()
    charuco_cfg = cfg.charuco_calibration

    # Read ChArUco board parameters from config (saved by frontend auto-save)
    squares_h = int(charuco_cfg.get("squares_h", 10))
    squares_v = int(charuco_cfg.get("squares_v", 9))
    square_size = float(charuco_cfg.get("square_size", 0.03))
    marker_ratio = float(charuco_cfg.get("marker_ratio", 0.5))
    aruco_dict = charuco_cfg.get("aruco_dict", "DICT_4X4_1000")
    min_corners = int(charuco_cfg.get("min_corners", 6))
    dt = float(charuco_cfg.get("dt", 1.0))
    source_root = Path(cfg.source_paths[source_path_idx])
    base_root = Path(cfg.base_paths[source_path_idx])

    # Get calibration path settings from unified config
    file_pattern = cfg.calibration_image_format
    calibration_subfolder = cfg.calibration_subfolder

    # Create job
    job_id = job_manager.create_job(
        "charuco",
        processed_images=0,
        valid_images=0,
        total_images=0,
        current_camera=camera,
    )

    def run_calibration():
        try:
            job_manager.update_job(job_id, status="running")

            # Get calibration input path using config settings
            cam_input_path = build_calibration_camera_path(cfg, source_path_idx, camera)

            # Use config-based paths with explicit input path override
            calibrator = ChArUcoCalibrator(
                source_dir=source_root,
                base_dir=base_root,
                camera_count=1,
                file_pattern=file_pattern,
                squares_h=squares_h,
                squares_v=squares_v,
                square_size=square_size,
                marker_ratio=marker_ratio,
                aruco_dict=aruco_dict,
                min_corners=min_corners,
                dt=dt,
                calibration_subfolder=calibration_subfolder,
                calibration_input_path=cam_input_path,
            )

            def progress_callback(progress_data):
                job_manager.update_job(
                    job_id,
                    processed_images=progress_data.get("processed_images", 0),
                    valid_images=progress_data.get("valid_images", 0),
                    total_images=progress_data.get("total_images", 0),
                    progress=progress_data.get("progress", 0),
                )

            result = calibrator.process_camera(camera, progress_callback=progress_callback)

            if result.get("success"):
                job_manager.complete_job(
                    job_id,
                    camera_matrix=result.get("camera_matrix"),
                    dist_coeffs=result.get("dist_coeffs"),
                    rms_error=result.get("rms_error"),
                    num_images_used=result.get("num_images_used"),
                    model_path=result.get("model_path"),
                )
                logger.info(
                    f"ChArUco calibration completed. "
                    f"RMS error: {result['rms_error']:.4f}, "
                    f"Images used: {result['num_images_used']}"
                )
            else:
                job_manager.fail_job(job_id, result.get("error", "Calibration failed"))

        except Exception as e:
            logger.error(f"ChArUco calibration job {job_id} failed: {e}")
            job_manager.fail_job(job_id, str(e))

    # Start job in background thread
    thread = threading.Thread(target=run_calibration)
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "starting",
        "message": f"ChArUco calibration job started for camera {camera}",
        "board_config": {
            "squares_h": squares_h,
            "squares_v": squares_v,
            "square_size": square_size,
            "marker_ratio": marker_ratio,
            "aruco_dict": aruco_dict,
        },
    })


@charuco_bp.route("/calibration/charuco/calibrate_all", methods=["POST"])
def charuco_calibrate_all():
    """
    Start ChArUco calibration for all cameras.

    Uses unified calibration config for path handling.
    Board parameters are read from config (saved by frontend auto-save).

    Request JSON:
        source_path_idx: int

    Returns:
        JSON with job_id, status, message, cameras
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))

    cfg = get_config()
    charuco_cfg = cfg.charuco_calibration

    # Read ChArUco board parameters from config (saved by frontend auto-save)
    squares_h = int(charuco_cfg.get("squares_h", 10))
    squares_v = int(charuco_cfg.get("squares_v", 9))
    square_size = float(charuco_cfg.get("square_size", 0.03))
    marker_ratio = float(charuco_cfg.get("marker_ratio", 0.5))
    aruco_dict = charuco_cfg.get("aruco_dict", "DICT_4X4_1000")
    min_corners = int(charuco_cfg.get("min_corners", 6))
    dt = float(charuco_cfg.get("dt", 1.0))

    camera_numbers = cfg.camera_numbers
    source_root = Path(cfg.source_paths[source_path_idx])
    base_root = Path(cfg.base_paths[source_path_idx])

    # Get calibration path settings from unified config
    file_pattern = cfg.calibration_image_format
    calibration_subfolder = cfg.calibration_subfolder

    # Create job with camera-aware tracking
    job_id = job_manager.create_job(
        "charuco",
        processed_cameras=0,
        total_cameras=len(camera_numbers),
        current_camera=None,
        camera_progress={
            f"Cam{cam}": {"status": "pending", "valid_images": 0}
            for cam in camera_numbers
        },
    )

    def run_calibration():
        try:
            job_manager.update_job(job_id, status="running")

            # Get calibration input path using config settings
            # When camera_subfolders is empty, path is same for all cameras
            first_camera = camera_numbers[0] if camera_numbers else 1
            cam_input_path = build_calibration_camera_path(cfg, source_path_idx, first_camera)

            # Use config-based paths with explicit input path override
            calibrator = ChArUcoCalibrator(
                source_dir=source_root,
                base_dir=base_root,
                camera_count=len(camera_numbers),
                file_pattern=file_pattern,
                squares_h=squares_h,
                squares_v=squares_v,
                square_size=square_size,
                marker_ratio=marker_ratio,
                aruco_dict=aruco_dict,
                min_corners=min_corners,
                dt=dt,
                calibration_subfolder=calibration_subfolder,
                calibration_input_path=cam_input_path,
            )

            def progress_callback(progress_data):
                current_camera = progress_data.get("current_camera")
                camera_progress = job_manager.get_job(job_id).get("camera_progress", {})

                if current_camera:
                    cam_key = f"Cam{current_camera}"
                    camera_progress[cam_key] = {
                        "status": "running",
                        "processed_images": progress_data.get("processed_images", 0),
                        "valid_images": progress_data.get("valid_images", 0),
                    }

                job_manager.update_job(
                    job_id,
                    current_camera=current_camera,
                    processed_cameras=progress_data.get("processed_cameras", 0),
                    camera_progress=camera_progress,
                    progress=int(
                        (progress_data.get("processed_cameras", 0) / len(camera_numbers)) * 100
                    ),
                )

            result = calibrator.process_all_cameras(progress_callback=progress_callback)

            # Update final camera statuses
            camera_progress = {}
            for cam_num, cam_result in result.get("camera_results", {}).items():
                status = "completed" if cam_result.get("success") else "failed"
                camera_progress[f"Cam{cam_num}"] = {
                    "status": status,
                    "rms_error": cam_result.get("rms_error"),
                    "num_images_used": cam_result.get("num_images_used"),
                    "error": cam_result.get("error"),
                }

            job_manager.complete_job(
                job_id,
                camera_progress=camera_progress,
                current_camera=None,
            )

            logger.info(
                f"ChArUco calibration completed for {result['processed_cameras']} cameras"
            )

        except Exception as e:
            logger.error(f"ChArUco calibration job {job_id} failed: {e}")
            job_manager.fail_job(job_id, str(e))

    # Start job in background thread
    thread = threading.Thread(target=run_calibration)
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "starting",
        "message": f"ChArUco calibration job started for {len(camera_numbers)} camera(s)",
        "cameras": camera_numbers,
    })


@charuco_bp.route("/calibration/charuco/status/<job_id>", methods=["GET"])
def charuco_status(job_id):
    """
    Get ChArUco calibration job status.

    Args:
        job_id: Job ID to query

    Returns:
        JSON with job status, progress, timing info
    """
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job_data)


@charuco_bp.route("/calibration/charuco/load_results", methods=["GET"])
def charuco_load_results():
    """
    Load previously computed ChArUco calibration results.

    Returns the overall camera model and per-frame detection data including
    corner IDs and pixel coordinates for all calibration frames.

    Query params:
        source_path_idx: int
        camera: int

    Returns:
        JSON with:
        - exists: bool
        - camera_model: dict with camera_matrix, dist_coeffs, rvecs, tvecs, etc.
        - frames: list of dicts, each with:
            - frame_index: int (1-indexed)
            - corners: list of [x, y] pixel coordinates
            - corner_ids: list of corner IDs
            - original_filename: str
        - board: dict with board parameters
    """
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])
        cam_output_base = base_root / "calibration" / f"Cam{camera}"

        # Check both old and new paths
        model_folder = cam_output_base / "charuco_planar" / "model"
        model_file = model_folder / "camera_model.mat"
        indices_folder = cam_output_base / "charuco_planar" / "indices"

        # Fall back to old path structure if new doesn't exist
        if not model_file.exists():
            model_folder = cam_output_base / "model"
            model_file = model_folder / "camera_model.mat"
            indices_folder = cam_output_base / "indices"

        if not model_file.exists():
            return jsonify({
                "exists": False,
                "message": "No saved camera model found"
            })

        results = {"exists": True}

        # Load camera model
        import scipy.io
        model_data = scipy.io.loadmat(
            str(model_file), struct_as_record=False, squeeze_me=True
        )

        camera_model = {
            "camera_matrix": model_data["camera_matrix"].tolist(),
            "dist_coeffs": model_data["dist_coeffs"].tolist(),
            "reprojection_error": float(model_data.get("reprojection_error", 0)),
            "num_images_used": int(model_data.get("num_images", 0)),
            "focal_length": [
                float(model_data["camera_matrix"][0, 0]),
                float(model_data["camera_matrix"][1, 1]),
            ],
            "principal_point": [
                float(model_data["camera_matrix"][0, 2]),
                float(model_data["camera_matrix"][1, 2]),
            ],
        }

        # Include rvecs/tvecs if available
        if "rvecs" in model_data:
            rvecs = model_data["rvecs"]
            if isinstance(rvecs, np.ndarray):
                camera_model["rvecs"] = rvecs.tolist()
        if "tvecs" in model_data:
            tvecs = model_data["tvecs"]
            if isinstance(tvecs, np.ndarray):
                camera_model["tvecs"] = tvecs.tolist()

        # Include dot_spacing_mm if available
        if "dot_spacing_mm" in model_data:
            camera_model["dot_spacing_mm"] = float(model_data["dot_spacing_mm"])

        results["camera_model"] = camera_model

        # Load board parameters if available
        if "board_params" in model_data:
            bp = model_data["board_params"]
            results["board"] = {
                "squares_h": int(getattr(bp, "squares_h", 10)),
                "squares_v": int(getattr(bp, "squares_v", 9)),
                "square_size_m": float(getattr(bp, "square_size", 0.03)),
                "marker_ratio": float(getattr(bp, "marker_ratio", 0.5)),
                "aruco_dict": str(getattr(bp, "aruco_dict", "DICT_4X4_1000")),
            }

        # Load per-frame detection data (corners and IDs)
        frames = []
        if indices_folder.exists():
            # Find all indexing files
            indexing_files = sorted(indices_folder.glob("indexing_*.mat"))

            for idx_file in indexing_files:
                try:
                    # Extract frame number from filename
                    frame_num = int(idx_file.stem.split("_")[1])

                    frame_data = scipy.io.loadmat(
                        str(idx_file), struct_as_record=False, squeeze_me=True
                    )

                    frame_info = {"frame_index": frame_num}

                    # Corner pixel coordinates
                    if "corners" in frame_data:
                        corners = frame_data["corners"]
                        if isinstance(corners, np.ndarray):
                            frame_info["corners"] = corners.reshape(-1, 2).tolist()

                    # Corner IDs
                    if "corner_ids" in frame_data:
                        ids = frame_data["corner_ids"]
                        if isinstance(ids, np.ndarray):
                            frame_info["corner_ids"] = ids.flatten().tolist()

                    # Original filename
                    if "original_filename" in frame_data:
                        frame_info["original_filename"] = str(
                            frame_data["original_filename"]
                        )

                    # Corner count
                    if "corner_count" in frame_data:
                        frame_info["corner_count"] = int(frame_data["corner_count"])

                    frames.append(frame_info)

                except Exception as e:
                    logger.warning(f"Could not load frame data from {idx_file}: {e}")
                    continue

        results["frames"] = frames
        results["num_frames"] = len(frames)

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error loading ChArUco calibration results: {e}")
        return jsonify({"error": str(e)}), 500


@charuco_bp.route("/calibration/charuco/calibrate_batch", methods=["POST"])
def charuco_calibrate_batch():
    """
    Start ChArUco calibration job with batch processing support.

    Request JSON:
        active_paths: list of path indices (default: from config)
        cameras: list of camera numbers (default: from config)

    Board parameters are read from config.yaml.
    Processes each (path, camera) combination as a separate sub-job.

    Returns:
        JSON with parent_job_id, sub_jobs list, status
    """
    data = request.get_json() or {}
    logger.info(f"Received batch ChArUco calibration request: {data}")

    try:
        cfg = get_config()
        base_paths = cfg.base_paths
        source_paths = cfg.source_paths
        charuco_cfg = cfg.charuco_calibration

        # Get batch parameters
        active_paths = data.get("active_paths")
        if active_paths is None:
            active_paths = cfg.calibration_active_paths

        cameras = data.get("cameras")
        if cameras is None:
            cameras = cfg.calibration_cameras

        # Validate paths
        valid_paths = [i for i in active_paths if 0 <= i < len(base_paths)]
        if not valid_paths:
            return jsonify({"error": "No valid path indices provided"}), 400

        if not cameras:
            return jsonify({"error": "No cameras specified"}), 400

        # Generate batch targets (no merged for calibration)
        targets = iter_batch_targets(
            base_paths=base_paths,
            active_paths=valid_paths,
            cameras=cameras,
            include_merged=False,
            source_paths=source_paths,
        )

        if not targets:
            return jsonify({"error": "No targets to process"}), 400

        # Create parent job
        parent_job_id = job_manager.create_job(
            "charuco_batch",
            total_targets=len(targets),
        )
        sub_jobs = []

        # Read charuco parameters
        squares_h = int(charuco_cfg.get("squares_h", 10))
        squares_v = int(charuco_cfg.get("squares_v", 9))
        square_size = float(charuco_cfg.get("square_size", 0.03))
        marker_ratio = float(charuco_cfg.get("marker_ratio", 0.5))
        aruco_dict = charuco_cfg.get("aruco_dict", "DICT_4X4_1000")
        min_corners = int(charuco_cfg.get("min_corners", 6))
        dt = float(charuco_cfg.get("dt", 1.0))
        file_pattern = cfg.calibration_image_format
        calibration_subfolder = cfg.calibration_subfolder

        # Launch a job for each target
        for target in targets:
            base_dir = target.base_path
            source_dir = target.source_path or base_dir
            cam_num = target.camera

            # Create sub-job
            job_id = job_manager.create_job(
                "charuco",
                camera=cam_num,
                path_idx=target.path_idx,
                parent_job_id=parent_job_id,
                processed_images=0,
                valid_images=0,
                total_images=0,
            )
            sub_jobs.append({
                "job_id": job_id,
                "camera": cam_num,
                "path_idx": target.path_idx,
                "label": target.label,
            })

            # Launch thread
            thread = threading.Thread(
                target=_run_charuco_calibration_job,
                args=(
                    job_id,
                    base_dir,
                    source_dir,
                    cam_num,
                    target.path_idx,
                    cfg,
                    squares_h,
                    squares_v,
                    square_size,
                    marker_ratio,
                    aruco_dict,
                    min_corners,
                    dt,
                    file_pattern,
                    calibration_subfolder,
                ),
            )
            thread.daemon = True
            thread.start()

        # Update parent job
        job_manager.update_job(parent_job_id, sub_jobs=sub_jobs, status="running")

        return jsonify({
            "parent_job_id": parent_job_id,
            "sub_jobs": sub_jobs,
            "total_targets": len(targets),
            "processed_targets": len(sub_jobs),
            "status": "starting",
            "message": f"ChArUco calibration started for {len(sub_jobs)} target(s) "
            f"across {len(valid_paths)} path(s)",
        })

    except Exception as e:
        logger.error(f"Error starting batch ChArUco calibration: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _run_charuco_calibration_job(
    job_id: str,
    base_dir: Path,
    source_dir: Path,
    camera: int,
    path_idx: int,
    cfg,
    squares_h: int,
    squares_v: int,
    square_size: float,
    marker_ratio: float,
    aruco_dict: str,
    min_corners: int,
    dt: float,
    file_pattern: str,
    calibration_subfolder: str,
):
    """Run ChArUco calibration job in a background thread."""
    try:
        logger.info(f"[ChArUco] Starting job {job_id} for Cam{camera} at path {path_idx}")

        job_manager.update_job(job_id, status="running")

        # Build camera input path
        cam_input_path = build_calibration_camera_path(cfg, path_idx, camera)

        # Create calibrator
        calibrator = ChArUcoCalibrator(
            source_dir=source_dir,
            base_dir=base_dir,
            camera_count=1,
            file_pattern=file_pattern,
            squares_h=squares_h,
            squares_v=squares_v,
            square_size=square_size,
            marker_ratio=marker_ratio,
            aruco_dict=aruco_dict,
            min_corners=min_corners,
            dt=dt,
            calibration_subfolder=calibration_subfolder,
            calibration_input_path=cam_input_path,
        )

        def progress_callback(progress_data):
            job_manager.update_job(
                job_id,
                processed_images=progress_data.get("processed_images", 0),
                valid_images=progress_data.get("valid_images", 0),
                total_images=progress_data.get("total_images", 0),
                progress=progress_data.get("progress", 0),
            )

        # Run calibration
        result = calibrator.process_camera(camera, progress_callback=progress_callback)

        if result.get("success"):
            job_manager.complete_job(
                job_id,
                camera_matrix=result.get("camera_matrix"),
                dist_coeffs=result.get("dist_coeffs"),
                rms_error=result.get("rms_error"),
                num_images_used=result.get("num_images_used"),
                model_path=result.get("model_path"),
            )
            logger.info(f"[ChArUco] Job {job_id} completed for Cam{camera}")
        else:
            job_manager.fail_job(job_id, result.get("error", "Calibration failed"))
            logger.error(f"[ChArUco] Job {job_id} failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"[ChArUco] Job {job_id} error: {e}", exc_info=True)
        job_manager.fail_job(job_id, str(e))


@charuco_bp.route("/calibration/charuco/batch_status/<job_id>", methods=["GET"])
def charuco_batch_status(job_id: str):
    """Get batch ChArUco calibration job status with aggregated sub-job info."""
    job_data = job_manager.get_job(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    # If parent job, aggregate sub-job status
    if "sub_jobs" in job_data:
        sub_job_statuses = []
        all_completed = True
        any_failed = False
        total_progress = 0

        for sub_job in job_data["sub_jobs"]:
            sub_id = sub_job["job_id"]
            sub_status = job_manager.get_job(sub_id)
            if sub_status:
                sub_status["camera"] = sub_job.get("camera")
                sub_status["label"] = sub_job.get("label", "")
                sub_job_statuses.append(sub_status)

                if sub_status["status"] != "completed":
                    all_completed = False
                if sub_status["status"] == "failed":
                    any_failed = True

                total_progress += sub_status.get("progress", 0)

        job_data["sub_job_statuses"] = sub_job_statuses
        job_data["overall_progress"] = total_progress / max(1, len(sub_job_statuses))

        if any_failed:
            job_data["status"] = "failed"
        elif all_completed:
            job_data["status"] = "completed"
        else:
            job_data["status"] = "running"

    # Add timing info
    job_data = job_manager.add_timing_info(job_data)

    return jsonify(job_data)
