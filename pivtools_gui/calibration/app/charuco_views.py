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

from ..calibration_charuco import ChArUcoCalibrator
from ..services.job_manager import job_manager
from ...utils import camera_number, numpy_to_png_base64

charuco_bp = Blueprint("charuco", __name__)


@charuco_bp.route("/calibration/charuco/validate_images", methods=["POST"])
def charuco_validate_images():
    """
    Validate ChArUco calibration images exist and are readable.

    Request JSON:
        source_path_idx: int
        camera: int
        file_pattern: str (default: "*.tif")

    Returns:
        JSON with validation status, file count, preview image
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    file_pattern = data.get("file_pattern", "*.tif")

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])
        cam_input_dir = source_root / "calibration" / f"Cam{camera}"

        if not cam_input_dir.exists():
            return jsonify({
                "valid": False,
                "checked": True,
                "found_count": 0,
                "file_pattern": file_pattern,
                "camera_path": str(cam_input_dir),
                "sample_files": [],
                "first_image_preview": None,
                "error": f"Calibration directory not found: {cam_input_dir}",
            })

        # Find images
        image_files = sorted(cam_input_dir.glob(file_pattern))
        sample_files = [f.name for f in image_files[:5]]

        if not image_files:
            # Try to suggest patterns
            all_files = list(cam_input_dir.glob("*.*"))
            image_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
            found_files = [f.name for f in all_files if f.suffix.lower() in image_exts][:5]

            return jsonify({
                "valid": False,
                "checked": True,
                "found_count": 0,
                "file_pattern": file_pattern,
                "camera_path": str(cam_input_dir),
                "sample_files": found_files,
                "first_image_preview": None,
                "error": f"No images found matching pattern: {file_pattern}",
            })

        # Try to load first image for preview
        preview_b64 = None
        image_size = None

        try:
            img = cv2.imread(str(image_files[0]), cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Convert to grayscale and normalize
                if img.ndim == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img.copy()

                if gray.dtype == np.uint16:
                    gray = (gray / 256).astype(np.uint8)

                image_size = [int(gray.shape[1]), int(gray.shape[0])]

                # Normalize for display
                disp = gray.astype(float) - gray.min()
                if disp.max() > 0:
                    disp = disp / disp.max()
                disp8 = (disp * 255).astype(np.uint8)
                preview_b64 = numpy_to_png_base64(disp8)

        except Exception as e:
            logger.warning(f"Could not read first image for preview: {e}")

        return jsonify({
            "valid": True,
            "checked": True,
            "found_count": len(image_files),
            "file_pattern": file_pattern,
            "camera_path": str(cam_input_dir),
            "sample_files": sample_files,
            "first_image_preview": preview_b64,
            "image_size": image_size,
            "error": None,
        })

    except Exception as e:
        logger.error(f"Error validating ChArUco images: {e}")
        return jsonify({
            "valid": False,
            "checked": True,
            "found_count": 0,
            "file_pattern": file_pattern,
            "error": str(e),
        }), 500


@charuco_bp.route("/calibration/charuco/detect", methods=["POST"])
def charuco_detect():
    """
    Detect ChArUco board in a single image.

    Request JSON:
        source_path_idx: int
        camera: int
        image_index: int
        file_pattern: str
        squares_h: int (default: 10)
        squares_v: int (default: 9)
        square_size: float (default: 0.03)
        marker_ratio: float (default: 0.5)
        aruco_dict: str (default: "DICT_4X4_1000")

    Returns:
        JSON with detection results, corner count, preview
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    image_index = int(data.get("image_index", 0))
    file_pattern = data.get("file_pattern", "*.tif")
    squares_h = int(data.get("squares_h", 10))
    squares_v = int(data.get("squares_v", 9))
    square_size = float(data.get("square_size", 0.03))
    marker_ratio = float(data.get("marker_ratio", 0.5))
    aruco_dict = data.get("aruco_dict", "DICT_4X4_1000")

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])
        base_root = Path(cfg.base_paths[source_path_idx])

        cam_input_dir = source_root / "calibration" / f"Cam{camera}"
        image_files = sorted(cam_input_dir.glob(file_pattern))

        if image_index >= len(image_files):
            return jsonify({"error": "Image index out of range", "found": False}), 404

        # Create calibrator just for detection
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
        )

        # Load and detect
        img_path = image_files[image_index]
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            return jsonify({"error": f"Could not read image: {img_path}", "found": False}), 500

        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)

        found, corners, ids, marker_corners, marker_ids = calibrator.detect_charuco_corners(img)

        if not found:
            return jsonify({
                "found": False,
                "corner_count": 0,
                "marker_count": len(marker_ids) if marker_ids is not None else 0,
                "message": "ChArUco board not detected (insufficient corners)",
            })

        # Create visualization
        if len(img.shape) == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()

        if marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(vis, marker_corners)
        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)

        # Convert to base64
        preview_b64 = numpy_to_png_base64(vis)

        return jsonify({
            "found": True,
            "corner_count": len(corners),
            "corners": corners.reshape(-1, 2).tolist(),
            "corner_ids": ids.flatten().tolist(),
            "marker_count": len(marker_ids) if marker_ids is not None else 0,
            "detection_preview": preview_b64,
            "image_path": str(img_path),
            "image_filename": img_path.name,
        })

    except Exception as e:
        logger.error(f"Error detecting ChArUco board: {e}")
        return jsonify({"error": str(e), "found": False}), 500


@charuco_bp.route("/calibration/charuco/calibrate", methods=["POST"])
def charuco_calibrate():
    """
    Start ChArUco calibration job with progress tracking.

    Request JSON:
        source_path_idx: int
        camera: int
        file_pattern: str
        squares_h: int
        squares_v: int
        square_size: float
        marker_ratio: float
        aruco_dict: str
        min_corners: int

    Returns:
        JSON with job_id, status, message
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    file_pattern = data.get("file_pattern", "*.tif")
    squares_h = int(data.get("squares_h", 10))
    squares_v = int(data.get("squares_v", 9))
    square_size = float(data.get("square_size", 0.03))
    marker_ratio = float(data.get("marker_ratio", 0.5))
    aruco_dict = data.get("aruco_dict", "DICT_4X4_1000")
    min_corners = int(data.get("min_corners", 6))
    dt = float(data.get("dt", 1.0))

    cfg = get_config()
    source_root = Path(cfg.source_paths[source_path_idx])
    base_root = Path(cfg.base_paths[source_path_idx])

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

    Request JSON:
        source_path_idx: int
        file_pattern: str
        squares_h: int
        squares_v: int
        square_size: float
        marker_ratio: float
        aruco_dict: str
        min_corners: int
        dt: float

    Returns:
        JSON with job_id, status, message, cameras
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    file_pattern = data.get("file_pattern", "*.tif")
    squares_h = int(data.get("squares_h", 10))
    squares_v = int(data.get("squares_v", 9))
    square_size = float(data.get("square_size", 0.03))
    marker_ratio = float(data.get("marker_ratio", 0.5))
    aruco_dict = data.get("aruco_dict", "DICT_4X4_1000")
    min_corners = int(data.get("min_corners", 6))
    dt = float(data.get("dt", 1.0))

    cfg = get_config()
    camera_numbers = cfg.camera_numbers
    source_root = Path(cfg.source_paths[source_path_idx])
    base_root = Path(cfg.base_paths[source_path_idx])

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

    Query params:
        source_path_idx: int
        camera: int

    Returns:
        JSON with calibration results (camera_matrix, dist_coeffs, etc.)
    """
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])
        cam_output_base = base_root / "calibration" / f"Cam{camera}"

        model_file = cam_output_base / "model" / "camera_model.mat"
        json_file = cam_output_base / "model" / "camera_model.json"

        if not model_file.exists() and not json_file.exists():
            return jsonify({"exists": False, "message": "No saved results found"})

        # Prefer JSON for easier parsing
        if json_file.exists():
            import json
            with open(json_file) as f:
                results = json.load(f)
            return jsonify({"exists": True, "results": results})

        # Fall back to .mat file
        import scipy.io
        mat_data = scipy.io.loadmat(str(model_file), struct_as_record=False, squeeze_me=True)

        results = {
            "camera_matrix": mat_data["camera_matrix"].tolist(),
            "dist_coeffs": mat_data["dist_coeffs"].tolist(),
            "rms_error": float(mat_data.get("reprojection_error", 0)),
            "num_images_used": int(mat_data.get("num_images", 0)),
        }

        if "board_params" in mat_data:
            bp = mat_data["board_params"]
            results["board"] = {
                "squares_h": int(getattr(bp, "squares_h", 10)),
                "squares_v": int(getattr(bp, "squares_v", 9)),
                "square_size_m": float(getattr(bp, "square_size", 0.03)),
                "marker_ratio": float(getattr(bp, "marker_ratio", 0.5)),
                "aruco_dict": str(getattr(bp, "aruco_dict", "DICT_4X4_1000")),
            }

        return jsonify({"exists": True, "results": results})

    except Exception as e:
        logger.error(f"Error loading ChArUco calibration results: {e}")
        return jsonify({"error": str(e)}), 500
