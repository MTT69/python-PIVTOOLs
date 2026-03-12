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

from pivtools_core.config import get_config, reload_config, Config
from pivtools_core.paths import get_data_paths
from pivtools_core.image_handling.calibration_loader import (
    read_calibration_image,
    validate_calibration_images,
    get_calibration_frame_count,
)

from pivtools_core.coordinate_utils import extract_coordinates
from pivtools_gui.calibration.vector_calibration_production import VectorCalibrator
from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.utils import camera_number, numpy_to_png_base64, numpy_to_base64

calibration_shared_bp = Blueprint("calibration_shared", __name__)


def set_active_calibration_method(method_name: str, cfg: "Config" = None) -> "Config":
    """Set the active calibration method in config and save.

    Call at the start of any calibration endpoint so that downstream code
    (e.g. GlobalCoordinateAligner, statistics workflow selection) sees the
    correct method even if the user didn't explicitly click "Set as Active".

    Returns the (possibly reloaded) Config instance.
    """
    if cfg is None:
        cfg = reload_config()
    current = cfg.active_calibration_method
    if current != method_name:
        cfg.data.setdefault("calibration", {})["active"] = method_name
        cfg.save()
        logger.info(f"Auto-set active calibration method: {current!r} -> {method_name!r}")
    return cfg


# Calibration image cache: {(source_path_idx, camera, idx, output_format): response_dict}
_cal_image_cache = {}
_cal_cache_lock = threading.Lock()
_CAL_CACHE_MAX = 10


@calibration_shared_bp.route("/calibration/set_datum", methods=["POST"])
def calibration_set_datum():
    """
    Set a new datum (origin) and/or apply offsets to ALL runs in the specified type's coordinates.

    Request JSON:
        base_path_idx or source_path_idx: int
        camera: int
        run: int (used only for logging, offsets apply to ALL runs)
        type_name: str (default: "instantaneous") - which coordinates file to update
        x: float (optional) - New x origin (subtracted from all coordinates)
        y: float (optional) - New y origin (subtracted from all coordinates)
        x_offset: float (optional) - X offset to apply (added to all coordinates)
        y_offset: float (optional) - Y offset to apply (added to all coordinates)
        use_stereo: bool (optional) - Whether to use stereo data paths
        camera_pair: list[int] (optional) - Camera pair for stereo data (e.g., [1, 2])

    Returns:
        JSON with status and updated file path
    """
    data = request.get_json() or {}
    base_path_idx = int(data.get("base_path_idx", data.get("source_path_idx", 0)))
    camera = camera_number(data.get("camera", 1))
    run = int(data.get("run", 1))  # For logging only
    type_name = data.get("type_name", "instantaneous")
    x0 = data.get("x")
    y0 = data.get("y")
    x_offset = data.get("x_offset", 0)
    y_offset = data.get("y_offset", 0)
    use_stereo = data.get("use_stereo", False)
    camera_pair = data.get("camera_pair")
    if camera_pair and isinstance(camera_pair, list):
        camera_pair = tuple(camera_pair)

    logger.debug(f"updating datum/offset for all runs in {type_name} (triggered from run {run}, stereo={use_stereo})")

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
            use_stereo=use_stereo,
            stereo_camera_pair=camera_pair,
        )
        data_dir = paths["data_dir"]
        coords_path = data_dir / "coordinates.mat"

        if not coords_path.exists():
            return jsonify({"error": f"Coordinates file not found: {coords_path}"}), 404

        mat = scipy.io.loadmat(str(coords_path), struct_as_record=False, squeeze_me=True)
        if "coordinates" not in mat:
            return jsonify({"error": "Variable 'coordinates' not found in mat"}), 400

        coordinates = mat["coordinates"]

        # Determine number of runs in this coordinates file
        if hasattr(coordinates, "__len__") and not isinstance(coordinates, np.void):
            num_runs = len(coordinates)
        else:
            num_runs = 1

        # Check if stereo coordinates (has z field)
        first_coord = coordinates[0] if num_runs > 0 else coordinates
        has_z = hasattr(first_coord, 'z') if hasattr(first_coord, '__getattr__') else 'z' in first_coord.dtype.names if hasattr(first_coord, 'dtype') and first_coord.dtype.names else False

        logger.debug(f"[set_datum] Processing {type_name} with {num_runs} runs, has_z={has_z}")

        # Use appropriate dtype based on whether stereo (with z) or planar (x,y only)
        if has_z:
            dtype = [("x", object), ("y", object), ("z", object)]
        else:
            dtype = [("x", object), ("y", object)]
        coords_struct = np.empty((num_runs,), dtype=dtype)

        # Apply datum/offset to ALL runs
        for i in range(num_runs):
            cx, cy = extract_coordinates(mat["coordinates"], i + 1)

            # Extract z if stereo
            if has_z:
                c_el = coordinates[i] if num_runs > 1 else coordinates
                cz = np.asarray(c_el.z) if hasattr(c_el, 'z') else np.asarray(c_el['z'])

            # Log first valid (non-NaN) values for debugging
            if i == 0:
                # Find first non-NaN value for better logging
                valid_x = cx[~np.isnan(cx)] if cx.size > 0 else cx
                valid_y = cy[~np.isnan(cy)] if cy.size > 0 else cy
                first_x = valid_x.flat[0] if valid_x.size > 0 else np.nan
                first_y = valid_y.flat[0] if valid_y.size > 0 else np.nan
                logger.debug(
                    f"[set_datum] {type_name} run 1 - original first valid x,y: {first_x}, {first_y}"
                )

            # Apply datum shift if x/y are provided
            if x0 is not None and y0 is not None:
                cx = cx - float(x0)
                cy = cy - float(y0)

            # Apply offsets if present
            if x_offset is not None and y_offset is not None:
                cx = cx + float(x_offset)
                cy = cy + float(y_offset)

            coords_struct["x"][i] = cx
            coords_struct["y"][i] = cy
            if has_z:
                coords_struct["z"][i] = cz  # Preserve z unchanged

            if i == 0:
                valid_x = cx[~np.isnan(cx)] if cx.size > 0 else cx
                valid_y = cy[~np.isnan(cy)] if cy.size > 0 else cy
                first_x = valid_x.flat[0] if valid_x.size > 0 else np.nan
                first_y = valid_y.flat[0] if valid_y.size > 0 else np.nan
                logger.debug(
                    f"[set_datum] {type_name} run 1 - after transform first valid x,y: {first_x}, {first_y}"
                )

        scipy.io.savemat(
            str(coords_path), {"coordinates": coords_struct}, do_compression=True
        )
        logger.info(f"[set_datum] Updated {type_name} coordinates ({num_runs} runs): {coords_path}")

        return jsonify({
            "status": "ok",
            "type_name": type_name,
            "num_runs_updated": num_runs,
            "coords_path": str(coords_path),
            "x0": x0,
            "y0": y0,
            "x_offset": x_offset,
            "y_offset": y_offset,
        })

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
    camera = camera_number(request.args.get("camera", default=1, type=int))
    idx = request.args.get("idx", default=1, type=int)
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    output_format = request.args.get("format", default="jpeg", type=str).lower()
    quality = request.args.get("quality", default=85, type=int)

    # Check calibration image cache
    cal_cache_key = (source_path_idx, camera, idx, output_format)
    with _cal_cache_lock:
        if cal_cache_key in _cal_image_cache:
            return jsonify(_cal_image_cache[cal_cache_key])

    try:
        cfg = get_config()

        # Early bounds validation
        frame_count = get_calibration_frame_count(camera, cfg, source_path_idx)
        if frame_count > 0 and idx > frame_count:
            return jsonify({
                "error": f"Frame index {idx} exceeds available frames ({frame_count})",
                "frame_count": frame_count,
                "requested_idx": idx
            }), 400

        # Read calibration image using centralized loader
        img = read_calibration_image(idx, camera, cfg, source_path_idx)

        if img is None:
            return jsonify({"error": "Could not read calibration image"}), 500

        # Calculate statistics (use float32 to halve memory vs float64)
        img_float = img.astype(np.float32)
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
        # For large images (>4MP), subsample for percentile computation
        total_pixels = img_float.size
        if total_pixels > 4_000_000:
            # Stride-sample: take every Nth pixel for fast percentile
            stride = max(2, int(np.sqrt(total_pixels / 1_000_000)))
            img_sampled = img_float[::stride, ::stride].ravel()
        else:
            img_sampled = img_float.ravel()

        p1 = float(np.percentile(img_sampled, 1))
        p99 = float(np.percentile(img_sampled, 99))

        if data_range > 0:
            vmin_pct = 100.0 * (p1 - img_min) / data_range
            vmax_pct = 100.0 * (p99 - img_min) / data_range
        else:
            vmin_pct = 0.0
            vmax_pct = 100.0

        stats["vmin_pct"] = round(vmin_pct, 2)
        stats["vmax_pct"] = round(vmax_pct, 2)

        # Encode to base64 using shared utility (min-max normalization for
        # non-uint8, as-is for uint8).  This matches app.py get_frame_pair
        # so the frontend vmin/vmax contrast controls work correctly.
        b64_image = numpy_to_base64(img, format=output_format, jpeg_quality=quality)
        mime_type = f"image/{output_format}"

        result = {
            "image": b64_image,
            "mime_type": mime_type,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "stats": stats,
            "frame_count": frame_count,
            "current_idx": idx,
        }

        # Cache the result
        with _cal_cache_lock:
            _cal_image_cache[cal_cache_key] = result
            # Evict oldest entries if cache is full
            if len(_cal_image_cache) > _CAL_CACHE_MAX:
                oldest_key = next(iter(_cal_image_cache))
                del _cal_image_cache[oldest_key]

        return jsonify(result)

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
        source_path_idx: int - Index into calibration_sources list (default: 0)
        image_format: str - Override for calibration image pattern (optional)
        num_images: int - Override for expected image count (optional)
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
        calibration_sources: list[str] - Direct paths to calibration images (REQUIRED)
        use_camera_subfolders: bool - Whether to use camera subfolders
        camera_subfolders: list[str] - Custom camera folder names

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
            "calibration_sources": [str(p) for p in cfg.calibration_sources],
            "is_container_format": cfg.calibration_is_container_format,
            "camera_subfolders": cfg.calibration_camera_subfolders,
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
        # calibration_sources - direct paths to calibration images
        if "calibration_sources" in data:
            cal_block["calibration_sources"] = [s for s in data["calibration_sources"] if s] if data["calibration_sources"] else []
        # camera_subfolders - independent from PIV camera subfolders
        if "camera_subfolders" in data:
            cal_block["camera_subfolders"] = list(data["camera_subfolders"]) if data["camera_subfolders"] else []

        # Save config
        cfg.save()

        return jsonify({
            "status": "ok",
            "image_format": cfg.calibration_image_format,
            "num_images": cfg.calibration_image_count,
            "image_type": cfg.calibration_image_type,
            "zero_based_indexing": cfg.calibration_zero_based_indexing,
            "use_camera_subfolders": cfg.calibration_use_camera_subfolders,
            "calibration_sources": [str(p) for p in cfg.calibration_sources],
            "is_container_format": cfg.calibration_is_container_format,
            "camera_subfolders": cfg.calibration_camera_subfolders,
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

        # Get dt from config (uses active calibration method to determine source)
        dt = cfg.dt

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

                # Pre-compute alignment shifts (no data file I/O)
                alignment = None
                if cfg.global_coordinates_enabled:
                    try:
                        from pivtools_gui.calibration.global_coordinate_alignment import GlobalCoordinateAligner
                        logger.debug("Pre-computing global coordinate alignment shifts...")
                        aligner = GlobalCoordinateAligner(base_root, cfg)
                        alignment = aligner.precompute_camera_shifts(type_name)
                        if alignment:
                            logger.debug(f"Alignment pre-computed: {len(alignment['camera_shifts'])} cameras, invert_ux={alignment['invert_ux']}")
                    except Exception as align_err:
                        logger.error(f"Failed to pre-compute alignment: {align_err}")
                        camera_results["global_alignment"] = {
                            "status": "failed",
                            "error": str(align_err),
                        }

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
                        camera_results[str(camera_num)] = {
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
                        z_world=cfg.self_calibration_z_offset,
                        tilt_x=cfg.self_calibration_tilt_x,
                        tilt_y=cfg.self_calibration_tilt_y,
                    )
                    calibrator.process_run(
                        num_frame_pairs,
                        progress_cb=make_progress_cb(camera_num),
                        alignment=alignment,
                    )

                    camera_results[str(camera_num)] = {"status": "completed"}

                # Report alignment result from pre-computed data
                if alignment and "global_alignment" not in camera_results:
                    camera_results["global_alignment"] = {
                        "status": "completed",
                        "invert_ux": alignment["invert_ux"],
                        "cameras": {
                            str(cam): {"shift_x": sx, "shift_y": sy}
                            for cam, (sx, sy) in alignment["camera_shifts"].items()
                        },
                    }

                # Save calibration snapshot
                try:
                    snapshot_path = cfg.save_calibration_snapshot(base_root)
                    camera_results["calibration_snapshot"] = str(snapshot_path)
                except Exception as snap_err:
                    logger.warning(f"Failed to save calibration snapshot: {snap_err}")

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


# ============================================================================
# CALIBRATION SNAPSHOT ROUTES
# ============================================================================


@calibration_shared_bp.route("/calibration/snapshot", methods=["GET"])
def calibration_snapshot_check():
    """
    Check if a calibration snapshot exists for the given base path.

    Query params:
        base_path_idx: int (default 0)

    Returns:
        JSON with {exists: bool, date?: str, calibration_method?: str}
    """
    try:
        base_path_idx = int(request.args.get("base_path_idx", 0))
        cfg = get_config()
        base_path = Path(cfg.base_paths[base_path_idx])

        snapshot = Config.load_calibration_snapshot(base_path)
        return jsonify({
            "exists": True,
            "date": snapshot.get("date"),
            "calibration_method": snapshot.get("calibration_method"),
        })
    except FileNotFoundError:
        return jsonify({"exists": False})
    except (IndexError, ValueError) as e:
        return jsonify({"exists": False, "error": str(e)})
    except Exception as e:
        logger.error(f"Error checking calibration snapshot: {e}")
        return jsonify({"exists": False, "error": str(e)})


@calibration_shared_bp.route("/calibration/snapshot/load", methods=["POST"])
def calibration_snapshot_load():
    """
    Load a calibration snapshot, replacing the current calibration config.

    Preserves source-related fields (calibration_sources, image_format, etc.)
    so only the calibration results are restored.

    Request JSON:
        base_path_idx: int (default 0)

    Returns:
        JSON with {status: "success", date, calibration_method}
    """
    try:
        data = request.get_json() or {}
        base_path_idx = int(data.get("base_path_idx", 0))

        cfg = get_config()
        base_path = Path(cfg.base_paths[base_path_idx])

        snapshot = Config.load_calibration_snapshot(base_path)
        snapshot_cal = snapshot.get("calibration", {})

        if not snapshot_cal:
            return jsonify({"error": "Snapshot contains no calibration data"}), 400

        # Merge snapshot into current calibration config.
        # Snapshot may only contain the active method — preserve other methods.
        current_cal = cfg.data.get("calibration", {})
        current_cal.update(snapshot_cal)
        cfg.data["calibration"] = current_cal
        cfg.save()
        reload_config()

        return jsonify({
            "status": "success",
            "date": snapshot.get("date"),
            "calibration_method": snapshot.get("calibration_method"),
        })
    except FileNotFoundError:
        return jsonify({"error": "No calibration snapshot found"}), 404
    except Exception as e:
        logger.error(f"Error loading calibration snapshot: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_shared_bp.route("/calibration/clear_image_cache", methods=["POST"])
def calibration_clear_image_cache():
    """Clear the calibration image cache."""
    with _cal_cache_lock:
        _cal_image_cache.clear()
    return jsonify({"status": "ok"})


# ============================================================================
# GLOBAL COORDINATE ALIGNMENT ROUTES
# ============================================================================


@calibration_shared_bp.route("/calibration/global_coordinates/pixel_to_physical", methods=["POST"])
def global_coordinates_pixel_to_physical():
    """
    Preview what physical coordinate a pixel maps to.

    Request JSON:
        pixel_x: float - Pixel X coordinate
        pixel_y: float - Pixel Y coordinate
        camera: int - Camera number (1-based)
        source_path_idx: int - Index into base_paths
        type_name: str - 'instantaneous' or 'ensemble'

    Returns:
        JSON with physical_x, physical_y, method
    """
    data = request.get_json() or {}
    pixel_x = float(data.get("pixel_x", 0))
    pixel_y = float(data.get("pixel_y", 0))
    cam = camera_number(data.get("camera", 1))
    source_path_idx = int(data.get("source_path_idx", 0))
    type_name = data.get("type_name", "instantaneous")

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])

        from pivtools_gui.calibration.global_coordinate_alignment import GlobalCoordinateAligner
        aligner = GlobalCoordinateAligner(base_root, cfg)
        physical_x, physical_y = aligner.pixel_to_calibrated_physical(
            (pixel_x, pixel_y), cam, type_name
        )

        return jsonify({
            "physical_x": float(physical_x),
            "physical_y": float(physical_y),
            "method": cfg.active_calibration_method,
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"pixel_to_physical error: {e}")
        return jsonify({"error": str(e)}), 500
