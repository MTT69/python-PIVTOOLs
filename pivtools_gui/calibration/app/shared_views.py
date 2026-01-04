"""
Shared Calibration Views.

Provides utility endpoints used across calibration methods, including:
- Unified calibration image fetching
- Calibration configuration management
- Vector calibration (unified for dotboard and charuco)
"""

import threading
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from pivtools_core.image_handling.calibration_loader import (
    read_calibration_image,
    validate_calibration_images,
    get_calibration_frame_count,
)

from pivtools_core.coordinate_utils import extract_coordinates
from pivtools_gui.calibration.vector_calibration_production import VectorCalibrator
from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.utils import camera_number, numpy_to_png_base64

calibration_shared_bp = Blueprint("calibration_shared", __name__)


@calibration_shared_bp.route("/calibration/set_datum", methods=["POST"])
def calibration_set_datum():
    """
    Set a new datum (origin) for the coordinates of a given run, and/or apply offsets.

    Request JSON:
        base_path_idx or source_path_idx: int
        camera: int
        run: int
        type_name: str (default: "instantaneous")
        x: float (optional) - New x origin
        y: float (optional) - New y origin
        x_offset: float (optional) - X offset to apply
        y_offset: float (optional) - Y offset to apply

    Returns:
        JSON with status, run, shape
    """
    data = request.get_json() or {}
    base_path_idx = int(data.get("base_path_idx", data.get("source_path_idx", 0)))
    camera = camera_number(data.get("camera", 1))
    run = int(data.get("run", 1))
    type_name = data.get("type_name", "instantaneous")
    x0 = data.get("x")
    y0 = data.get("y")
    x_offset = data.get("x_offset", 0)
    y_offset = data.get("y_offset", 0)

    logger.debug("updating datum for run %d", run)

    try:
        cfg = get_config()
        source_root = Path(
            getattr(cfg, "base_paths", getattr(cfg, "source_paths", []))[base_path_idx]
        )
        paths = get_data_paths(
            base_dir=source_root,
            num_frame_pairs=cfg.num_frame_pairs,
            cam=camera,
            type_name=type_name,
            calibration=False,
        )
        data_dir = paths["data_dir"]
        coords_path = data_dir / "coordinates.mat"

        if not coords_path.exists():
            return jsonify({"error": f"Coordinates file not found: {coords_path}"}), 404

        mat = scipy.io.loadmat(coords_path, struct_as_record=False, squeeze_me=True)
        if "coordinates" not in mat:
            return (
                jsonify({"error": "Variable 'coordinates' not found in coords mat"}),
                400,
            )
        coordinates = mat["coordinates"]

        run_idx = run - 1

        # Use extract_coordinates from pivtools_core.coordinate_utils
        cx, cy = extract_coordinates(coordinates, run)

        logger.debug(
            f"[set_datum] Run {run} - original first x,y: {cx.flat[0]}, {cy.flat[0]}"
        )
        logger.debug(
            f"[set_datum] Datum to set: x0={x0}, y0={y0}, x_offset={x_offset}, y_offset={y_offset}"
        )

        # Only apply datum shift if x/y are provided
        if x0 is not None and y0 is not None:
            x0 = float(x0)
            y0 = float(y0)
            cx = cx - x0
            cy = cy - y0
            logger.debug(
                f"[set_datum] After datum shift, first x,y: {cx.flat[0]}, {cy.flat[0]}"
            )

        # Always apply offsets if present
        if x_offset is not None and y_offset is not None:
            x_offset = float(x_offset)
            y_offset = float(y_offset)
            cx = cx + x_offset
            cy = cy + y_offset
            logger.debug(f"[set_datum] After offset, first x,y: {cx.flat[0]}, {cy.flat[0]}")

        # Convert to proper MATLAB struct format
        num_runs = len(coordinates) if hasattr(coordinates, "__len__") else 1
        if num_runs == 1 and not hasattr(coordinates, "__len__"):
            num_runs = 1
            coordinates = [coordinates]

        dtype = [("x", object), ("y", object)]
        coords_struct = np.empty((num_runs,), dtype=dtype)

        # Copy all existing coordinates
        for i in range(num_runs):
            if i == run_idx:
                coords_struct["x"][i] = cx
                coords_struct["y"][i] = cy
            else:
                existing_x, existing_y = extract_coordinates(coordinates, i + 1)
                coords_struct["x"][i] = existing_x
                coords_struct["y"][i] = existing_y

        scipy.io.savemat(
            coords_path, {"coordinates": coords_struct}, do_compression=True
        )
        return jsonify({"status": "ok", "run": run, "shape": [cx.shape, cy.shape]})

    except Exception as e:
        logger.error(f"[set_datum] ERROR: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_shared_bp.route("/calibration/status", methods=["GET"])
def calibration_status():
    """
    Get calibration status - unified endpoint for all calibration types.

    Query params:
        source_path_idx: int
        camera: int
        type: str (optional) - Calibration type

    Returns:
        JSON with status info
    """
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))
    cal_type = request.args.get("type", None)

    # For now, return not_started for all status requests
    # This prevents 404 errors in the frontend
    return jsonify(
        {
            "status": "not_started",
            "source_path_idx": source_path_idx,
            "camera": camera,
            "type": cal_type,
        }
    )


# ============================================================================
# CALIBRATION IMAGE VIEWER ROUTES
# ============================================================================


@calibration_shared_bp.route("/calibration/get_frame", methods=["GET"])
def calibration_get_frame():
    """
    Fetch single calibration image for viewer using centralized loader.

    Query params:
        camera: int - Camera number (1-based)
        idx: int - Frame index (1-based unless zero_based_indexing is configured)
        source_path_idx: int - Index into source_paths list (default: 0)
        format: str - Output format: 'jpeg' or 'png' (default: 'jpeg')
        auto_limits: bool - Apply auto contrast limits (default: true)
        quality: int - JPEG quality 1-100 (default: 85)

    Returns:
        JSON with:
        - image: base64 encoded image
        - width: image width in pixels
        - height: image height in pixels
        - stats: {min, max, mean, vmin_pct, vmax_pct}
        - frame_count: total number of calibration frames
        - current_idx: current frame index
    """
    import io
    import base64
    from PIL import Image

    camera = camera_number(request.args.get("camera", default=1, type=int))
    idx = request.args.get("idx", default=1, type=int)
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    output_format = request.args.get("format", default="jpeg", type=str).lower()
    quality = request.args.get("quality", default=85, type=int)

    try:
        cfg = get_config()

        # Read calibration image using centralized loader
        img = read_calibration_image(idx, camera, cfg, source_path_idx)

        if img is None:
            return jsonify({"error": "Could not read calibration image"}), 500

        # Calculate statistics
        img_float = img.astype(np.float64)
        img_min = float(img_float.min())
        img_max = float(img_float.max())
        data_range = img_max - img_min

        stats = {
            "min": img_min,
            "max": img_max,
            "mean": float(img_float.mean()),
            "dtype": str(img.dtype),
        }

        # Calculate vmin/vmax as percentages (0-100) of the data range
        # This matches the logic in app.py get_percentile_stats()
        p1 = float(np.percentile(img_float, 1))
        p99 = float(np.percentile(img_float, 99))

        if data_range > 0:
            vmin_pct = 100.0 * (p1 - img_min) / data_range
            vmax_pct = 100.0 * (p99 - img_min) / data_range
        else:
            vmin_pct = 0.0
            vmax_pct = 100.0

        stats["vmin_pct"] = round(vmin_pct, 2)
        stats["vmax_pct"] = round(vmax_pct, 2)

        # Normalize for display using raw percentile values (not percentages)
        if p99 > p1:
            disp = (img_float - p1) / (p99 - p1)
            disp = np.clip(disp, 0, 1)
        else:
            disp = np.zeros_like(img_float)

        disp8 = (disp * 255).astype(np.uint8)

        # Convert to PIL Image
        pil_img = Image.fromarray(disp8)

        # Encode to base64
        buffer = io.BytesIO()
        if output_format == "png":
            pil_img.save(buffer, format="PNG")
            mime_type = "image/png"
        else:
            pil_img.save(buffer, format="JPEG", quality=quality)
            mime_type = "image/jpeg"

        b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Get total frame count
        frame_count = get_calibration_frame_count(camera, cfg, source_path_idx)

        return jsonify({
            "image": b64_image,
            "mime_type": mime_type,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "stats": stats,
            "frame_count": frame_count,
            "current_idx": idx,
        })

    except FileNotFoundError as e:
        logger.warning(f"Calibration image not found: {e}")
        return jsonify({"error": str(e)}), 404

    except Exception as e:
        logger.error(f"Error fetching calibration frame: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_shared_bp.route("/calibration/validate_images", methods=["POST"])
def calibration_validate_images():
    """
    Unified calibration image validation endpoint.

    Uses centralized calibration_loader for consistent validation
    across all calibration types.

    Request JSON:
        camera: int - Camera number (1-based)
        source_path_idx: int - Index into source_paths list (default: 0)
        image_format: str - Override for calibration image pattern (optional)
        num_images: int - Override for expected image count (optional)
        subfolder: str - Override for calibration subfolder (optional)
        image_type: str - Override for image type (optional)

    Returns:
        JSON with validation result from validate_calibration_images()
    """
    data = request.get_json() or {}
    camera = camera_number(data.get("camera", 1))
    source_path_idx = int(data.get("source_path_idx", 0))

    # Extract optional override parameters from request
    image_format = data.get("image_format")
    num_images = data.get("num_images")
    if num_images is not None:
        num_images = int(num_images)
    subfolder = data.get("subfolder")
    image_type = data.get("image_type")

    try:
        cfg = get_config()

        # Pass override parameters to validation function
        result = validate_calibration_images(
            camera,
            cfg,
            source_path_idx,
            image_format=image_format,
            num_images=num_images,
            subfolder=subfolder,
            image_type=image_type,
        )

        # Add extra context - use passed values if provided, else config values
        result["camera"] = camera
        result["source_path_idx"] = source_path_idx
        result["image_format"] = image_format if image_format else cfg.calibration_image_format
        result["image_type"] = image_type if image_type else cfg.calibration_image_type

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error validating calibration images: {e}")
        return jsonify({
            "valid": False,
            "error": str(e),
            "camera": camera,
            "source_path_idx": source_path_idx,
        }), 500


@calibration_shared_bp.route("/calibration/config", methods=["GET", "POST"])
def calibration_config():
    """
    Get or update calibration image configuration.

    All calibration settings are now unified under the 'calibration' block.

    GET: Return current calibration image settings
    POST: Update calibration image settings

    Request JSON (POST):
        image_format: str - Pattern for calibration images
        num_images: int - Number of calibration images expected
        image_type: str - 'standard' | 'cine' | 'lavision_set' | 'lavision_im7'
        zero_based_indexing: bool - Start indexing from 0
        subfolder: str - Subfolder for calibration images (e.g., "calibration")

    Note: use_camera_subfolders is read-only - derived from paths.camera_subfolders

    Returns:
        JSON with current calibration config
    """
    cfg = get_config()

    if request.method == "GET":
        return jsonify({
            "image_format": cfg.calibration_image_format,
            "num_images": cfg.calibration_image_count,
            "image_type": cfg.calibration_image_type,
            "zero_based_indexing": cfg.calibration_zero_based_indexing,
            "use_camera_subfolders": cfg.calibration_use_camera_subfolders,
            "subfolder": cfg.calibration_subfolder,
            "is_container_format": cfg.calibration_is_container_format,
            "camera_subfolders": cfg.calibration_camera_subfolders,
            "path_order": cfg.calibration_path_order,
        })

    # POST - Update config
    data = request.get_json() or {}

    try:
        # Get or create unified calibration block
        if "calibration" not in cfg.data:
            cfg.data["calibration"] = {}

        cal_block = cfg.data["calibration"]

        # Update fields if provided
        if "image_format" in data:
            cal_block["image_format"] = data["image_format"]
        if "num_images" in data:
            cal_block["num_images"] = int(data["num_images"])
        if "image_type" in data:
            cal_block["image_type"] = data["image_type"]
        if "zero_based_indexing" in data:
            cal_block["zero_based_indexing"] = bool(data["zero_based_indexing"])
        # use_camera_subfolders can now be set explicitly (especially for IM7 formats)
        if "use_camera_subfolders" in data:
            cal_block["use_camera_subfolders"] = bool(data["use_camera_subfolders"])
        if "subfolder" in data:
            cal_block["subfolder"] = str(data["subfolder"])
        # NEW: camera_subfolders - independent from PIV camera subfolders
        if "camera_subfolders" in data:
            cal_block["camera_subfolders"] = list(data["camera_subfolders"]) if data["camera_subfolders"] else []
        # NEW: path_order - controls whether camera folder comes before or after calibration subfolder
        if "path_order" in data:
            cal_block["path_order"] = str(data["path_order"])

        # Save config
        cfg.save()

        return jsonify({
            "status": "ok",
            "image_format": cfg.calibration_image_format,
            "num_images": cfg.calibration_image_count,
            "image_type": cfg.calibration_image_type,
            "zero_based_indexing": cfg.calibration_zero_based_indexing,
            "use_camera_subfolders": cfg.calibration_use_camera_subfolders,
            "subfolder": cfg.calibration_subfolder,
            "is_container_format": cfg.calibration_is_container_format,
            "camera_subfolders": cfg.calibration_camera_subfolders,
            "path_order": cfg.calibration_path_order,
        })

    except Exception as e:
        logger.error(f"Error updating calibration config: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# VECTOR CALIBRATION ROUTES
# ============================================================================


def detect_calibration_model_type(base_dir: Path, camera_num: int) -> Optional[str]:
    """
    Detect which calibration model type exists for a camera.

    Checks for existing calibration model files and returns the type.

    Args:
        base_dir: Base directory containing calibration data
        camera_num: Camera number (1-based)

    Returns:
        'charuco' if charuco model exists, 'dotboard' if dotboard model exists, None otherwise
    """
    calib_base = base_dir / "calibration" / f"Cam{camera_num}"

    # Check for charuco model first
    charuco_model = calib_base / "charuco_planar" / "model" / "camera_model.mat"
    if charuco_model.exists():
        return "charuco"

    # Check for dotboard model
    dotboard_model = calib_base / "dotboard_planar" / "model" / "dotboard_model.mat"
    if dotboard_model.exists():
        return "dotboard"

    return None


@calibration_shared_bp.route("/calibration/vectors/calibrate", methods=["POST"])
def vectors_calibrate():
    """
    Start vector calibration job for one or all cameras.

    Request JSON:
        source_path_idx: int - Index into base_paths list (default: 0)
        camera: int - Single camera number (optional, use if not providing cameras)
        cameras: list[int] - List of camera numbers (optional, for all cameras)
        type_name: str - 'instantaneous' or 'ensemble' (default: 'instantaneous')

    Returns:
        JSON with job_id and status
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    type_name = data.get("type_name", "instantaneous")

    # Support single camera OR all cameras
    cameras = data.get("cameras")
    if cameras is None:
        camera = camera_number(data.get("camera", 1))
        cameras = [camera]
    else:
        cameras = [camera_number(c) for c in cameras]

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])
        num_frame_pairs = cfg.num_frame_pairs

        # Get dt from the active calibration type or dotboard config
        active_type = getattr(cfg, "calibration_active", "dotboard")
        if active_type == "charuco":
            dt = getattr(cfg, "calibration_charuco_dt", 1.0)
        else:
            dt = getattr(cfg, "calibration_dotboard_dt", 1.0)

        # Create multi-camera job
        job_id = job_manager.create_job(
            "vectors",
            processed_cameras=0,
            total_cameras=len(cameras),
            current_camera=None,
            type_name=type_name,
            camera_progress={},
        )

        def run_calibration():
            try:
                camera_results = {}
                for idx, camera_num in enumerate(cameras):
                    # Update job with current camera
                    job_manager.update_job(
                        job_id,
                        status="running",
                        current_camera=camera_num,
                        processed_cameras=idx,
                    )

                    # Auto-detect model type
                    model_type = detect_calibration_model_type(base_root, camera_num)
                    if not model_type:
                        logger.warning(f"No calibration model found for camera {camera_num}")
                        camera_results[camera_num] = {
                            "status": "skipped",
                            "reason": "No calibration model found",
                        }
                        continue

                    logger.info(
                        f"Starting vector calibration for camera {camera_num} "
                        f"with model_type={model_type}"
                    )

                    # Create progress callback for this camera
                    def make_progress_cb(cam_num):
                        def progress_cb(progress_data):
                            # progress_data is a dict with processed_frames, total_frames, progress, etc.
                            try:
                                current_job = job_manager.get_job(job_id) or {}
                                job_manager.update_job(
                                    job_id,
                                    camera_progress={
                                        **current_job.get("camera_progress", {}),
                                        cam_num: {
                                            "current": progress_data.get("processed_frames", 0),
                                            "total": progress_data.get("total_frames", 0),
                                            "progress": progress_data.get("progress", 0),
                                        },
                                    },
                                )
                            except Exception as e:
                                logger.warning(f"Progress callback error: {e}")
                        return progress_cb

                    calibrator = VectorCalibrator(
                        base_dir=str(base_root),
                        camera_num=camera_num,
                        model_type=model_type,
                        dt=dt,
                        type_name=type_name,
                        runs=None,  # Always all runs
                    )
                    calibrator.process_run(
                        num_frame_pairs,
                        progress_cb=make_progress_cb(camera_num),
                    )

                    camera_results[camera_num] = {"status": "completed"}

                # Mark job complete
                job_manager.complete_job(
                    job_id,
                    camera_results=camera_results,
                    processed_cameras=len(cameras),
                )

            except Exception as e:
                logger.error(f"Vector calibration failed: {e}")
                job_manager.fail_job(job_id, str(e))

        thread = threading.Thread(target=run_calibration, daemon=True)
        thread.start()

        return jsonify({
            "job_id": job_id,
            "status": "starting",
            "cameras": cameras,
            "type_name": type_name,
        })

    except Exception as e:
        logger.error(f"Error starting vector calibration: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_shared_bp.route("/calibration/vectors/status/<job_id>", methods=["GET"])
def vectors_status(job_id):
    """
    Get vector calibration job status.

    Args:
        job_id: The job ID returned from /calibration/vectors/calibrate

    Returns:
        JSON with job status, progress, and results
    """
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job_data)
