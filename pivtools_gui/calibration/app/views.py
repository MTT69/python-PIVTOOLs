import glob
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import cv2
import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger

# Import production calibration classes
from ..calibration_planar.planar_calibration_production import (
    PlanarCalibrator,
)
from ..vector_calibration_production import VectorCalibrator
from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from pivtools_core.image_handling.load_images import read_image
from ...plotting.app.views import extract_coordinates
from ...utils import camera_number, numpy_to_png_base64


def cache_key(source_path_idx, camera):
    return (int(source_path_idx), str(camera))


calibration_cache = {}
calibration_bp = Blueprint("calibration", __name__)


# Global job tracking
calibration_jobs = {}
vector_jobs = {}
scale_factor_jobs = {}

# ============================================================================
# VECTOR CALIBRATION ROUTES WITH JOB MANAGEMENT
# ============================================================================


@calibration_bp.route("/calibration/vectors/calibrate_all", methods=["POST"])
def vectors_calibrate_all():
    """Start vector calibration job using production methods"""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    model_index = int(data.get("model_index", 0))
    dt = float(data.get("dt", 1.0))
    image_count = int(data.get("image_count", 1000))
    vector_pattern = data.get("vector_pattern", "%05d.mat")
    type_name = data.get("type_name", "instantaneous")

    job_id = str(uuid.uuid4())

    def run_vector_calibration():
        try:
            vector_jobs[job_id] = {
                "status": "starting",
                "progress": 0,
                "processed_frames": 0,
                "total_frames": image_count,
                "start_time": time.time(),
                "error": None,
            }

            cfg = get_config()
            Path(cfg.source_paths[source_path_idx])
            base_root = Path(cfg.base_paths[source_path_idx])

            def progress_callback(data):
                vector_jobs[job_id].update(
                    {
                        "status": "running",
                        "progress": data.get("progress", 0),
                        "processed_frames": data.get("processed_frames", 0),
                        "successful_frames": data.get("successful_frames", 0),
                    }
                )

            # Create calibrator
            calibrator = VectorCalibrator(
                base_dir=base_root,
                camera_num=camera,
                model_index=model_index,
                dt=dt,
                vector_pattern=vector_pattern,
                type_name=type_name,
            )

            # Run calibration with progress callback
            calibrator.process_run(image_count, progress_callback)

            vector_jobs[job_id]["status"] = "completed"
            vector_jobs[job_id]["progress"] = 100

        except Exception as e:
            logger.error(f"Vector calibration job {job_id} failed: {e}")
            vector_jobs[job_id]["status"] = "failed"
            vector_jobs[job_id]["error"] = str(e)

    # Start job in background thread
    thread = threading.Thread(target=run_vector_calibration)
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "job_id": job_id,
            "status": "starting",
            "message": f"Vector calibration job started for camera {camera}",
            "model_used": f"index_{model_index}",
            "image_count": image_count,
        }
    )


@calibration_bp.route("/calibration/vectors/status/<job_id>", methods=["GET"])
def vectors_status(job_id):
    """Get vector calibration job status"""
    if job_id not in vector_jobs:
        return jsonify({"error": "Job not found"}), 404

    job_data = vector_jobs[job_id].copy()

    # Add timing info
    if "start_time" in job_data:
        elapsed = time.time() - job_data["start_time"]
        job_data["elapsed_time"] = elapsed

        if job_data["status"] == "running" and job_data.get("progress", 0) > 0:
            estimated_total = elapsed / (job_data["progress"] / 100.0)
            job_data["estimated_remaining"] = max(0, estimated_total - elapsed)

    return jsonify(job_data)


# ============================================================================
# PRODUCTION PLANAR CALIBRATION ROUTES
# ===========================================================================


@calibration_bp.route("/calibration/set_datum", methods=["POST"])
def calibration_set_datum():
    """
    Set a new datum (origin) for the coordinates of a given run, and/or apply offsets.
    Expects JSON: source_path_idx, camera, run, x, y, x_offset, y_offset
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
        # Accept both base_paths and source_paths for compatibility
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

        # Use extract_coordinates from plotting.app.views
        cx, cy = extract_coordinates(coordinates, run)

        # Print for debugging
        print(f"[set_datum] Run {run} - original first x,y: {cx.flat[0]}, {cy.flat[0]}")
        print(
            f"[set_datum] Datum to set: x0={x0}, y0={y0}, x_offset={x_offset}, y_offset={y_offset}"
        )

        # Only apply datum shift if x/y are provided (not None)
        if x0 is not None and y0 is not None:
            x0 = float(x0)
            y0 = float(y0)
            cx = cx - x0
            cy = cy - y0
            print(
                f"[set_datum] After datum shift, first x,y: {cx.flat[0]}, {cy.flat[0]}"
            )

        # Always apply offsets if present
        if x_offset is not None and y_offset is not None:
            x_offset = float(x_offset)
            y_offset = float(y_offset)
            cx = cx + x_offset
            cy = cy + y_offset
            print(f"[set_datum] After offset, first x,y: {cx.flat[0]}, {cy.flat[0]}")

        # Convert to proper MATLAB struct format (not cell array)
        # Create structured numpy array with dtype [('x', object), ('y', object)]
        num_runs = len(coordinates) if hasattr(coordinates, '__len__') else 1
        if num_runs == 1 and not hasattr(coordinates, '__len__'):
            num_runs = 1
            coordinates = [coordinates]
        
        dtype = [('x', object), ('y', object)]
        coords_struct = np.empty((num_runs,), dtype=dtype)
        
        # Copy all existing coordinates
        for i in range(num_runs):
            if i == run_idx:
                # Use modified coordinates for this run
                coords_struct['x'][i] = cx
                coords_struct['y'][i] = cy
            else:
                # Copy existing coordinates
                existing_x, existing_y = extract_coordinates(coordinates, i + 1)
                coords_struct['x'][i] = existing_x
                coords_struct['y'][i] = existing_y

        scipy.io.savemat(coords_path, {"coordinates": coords_struct}, do_compression=True)
        return jsonify({"status": "ok", "run": run, "shape": [cx.shape, cy.shape]})
    except Exception as e:
        print(f"[set_datum] ERROR: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# PRODUCTION PLANAR CALIBRATION ROUTES
# ============================================================================


@calibration_bp.route("/calibration/planar/get_image", methods=["GET"])
def planar_get_image():
    """Get calibration image for production planar calibration"""
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))
    image_index = request.args.get("image_index", default=0, type=int)
    file_pattern = request.args.get("file_pattern", default="calib%05d.tif")

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])
        cam_input_dir = source_root / "calibration" / f"Cam{camera}"

        logger.info(f"Looking for images in: {cam_input_dir}")
        logger.info(f"File pattern: {file_pattern}")

        if not cam_input_dir.exists():
            return (
                jsonify({"error": f"Camera directory not found: {cam_input_dir}"}),
                404,
            )

        # Find calibration images
        if "%" in file_pattern:
            # Handle numbered patterns like calib%05d_enhanced.tif
            image_files = []
            i = 1
            while True:
                filename = file_pattern % i
                filepath = cam_input_dir / filename
                if filepath.exists():
                    image_files.append(str(filepath))
                    i += 1
                else:
                    break
        else:
            # Handle glob patterns like planar_calibration_plate_*.tif
            image_files = sorted(glob.glob(str(cam_input_dir / file_pattern)))

        logger.info(
            f"Found {len(image_files)} images: {[Path(f).name for f in image_files[:5]]}"
        )

        if not image_files:
            return (
                jsonify({"error": f"No images found with pattern {file_pattern}"}),
                404,
            )

        if image_index >= len(image_files):
            return (
                jsonify(
                    {
                        "error": f"Image index {image_index} out of range (0-{len(image_files)-1})"
                    }
                ),
                404,
            )

        img_path = image_files[image_index]
        logger.info(f"Loading image at index {image_index}: {Path(img_path).name}")

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return jsonify({"error": f"Could not load image: {img_path}"}), 500

        # Convert to grayscale if needed and normalize for display
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Normalize to 0-255 uint8 for display
        disp = gray - gray.min()
        if disp.max() > 0:
            disp = disp / disp.max()
        disp8 = (disp * 255).astype(np.uint8)

        # Convert to base64 PNG
        b64 = numpy_to_png_base64(disp8)

        return jsonify(
            {
                "image": b64,
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
                "path": str(img_path),
                "filename": Path(img_path).name,
                "total_images": len(image_files),
                "current_index": image_index,
                "all_filenames": [Path(f).name for f in image_files],
            }
        )

    except Exception as e:
        logger.error(f"Error getting planar calibration image: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/calibration/planar/validate_images", methods=["POST"])
def planar_validate_images():
    """Validate calibration images exist and are readable.

    Supports both standard image formats (tif, png, jpg) and container formats (.set, .im7).
    Returns validation status, file counts, and a preview image.
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    file_pattern = data.get("file_pattern", "calib%05d.tif")

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])

        # Detect container format
        is_container = '.set' in file_pattern.lower() or '.im7' in file_pattern.lower()

        # Setup paths based on format
        if is_container:
            cam_input_dir = source_root / "calibration"
            container_file = cam_input_dir / file_pattern
        else:
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
                "image_size": None,
                "format_detected": None,
                "container_format": is_container,
                "error": f"Calibration directory not found: {cam_input_dir}"
            })

        # Find calibration images
        image_files = []
        sample_files = []

        if is_container:
            # Container format: check if file exists and try to read first image
            if container_file.exists():
                image_files = [str(container_file)]
                sample_files = [container_file.name]
                # For containers, we can't easily count images without reading
                # Just indicate it exists
            else:
                return jsonify({
                    "valid": False,
                    "checked": True,
                    "found_count": 0,
                    "file_pattern": file_pattern,
                    "camera_path": str(cam_input_dir),
                    "sample_files": [],
                    "first_image_preview": None,
                    "image_size": None,
                    "format_detected": Path(file_pattern).suffix.lstrip('.'),
                    "container_format": True,
                    "error": f"Container file not found: {container_file}"
                })
        elif "%" in file_pattern:
            # Handle numbered patterns like calib%05d.tif
            i = 1
            while True:
                filename = file_pattern % i
                filepath = cam_input_dir / filename
                if filepath.exists():
                    image_files.append(str(filepath))
                    if len(sample_files) < 5:
                        sample_files.append(filename)
                    i += 1
                else:
                    break
        else:
            # Handle glob patterns like planar_calibration_plate_*.tif
            image_files = sorted(glob.glob(str(cam_input_dir / file_pattern)))
            sample_files = [Path(f).name for f in image_files[:5]]

        if not image_files:
            # Try to suggest a pattern based on files that exist in the directory
            suggested_pattern = None
            found_files = []
            error_msg = f"No images found matching pattern: {file_pattern}"

            if cam_input_dir.exists():
                # List all image files in the directory
                all_files = []
                for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    all_files.extend([f.name for f in cam_input_dir.glob(ext)])
                    all_files.extend([f.name for f in cam_input_dir.glob(ext.upper())])

                if all_files:
                    all_files = sorted(set(all_files))[:5]  # First 5 unique files
                    found_files = all_files

                    # Try to suggest a pattern from the first file
                    import re
                    sample = all_files[0]
                    ext = Path(sample).suffix

                    # Try to infer pattern from digits in filename
                    # Match sequences of digits and replace with format specifier
                    digit_match = re.search(r'(\d+)', sample)
                    if digit_match:
                        num_digits = len(digit_match.group(1))
                        suggested_pattern = re.sub(r'\d+', f'%0{num_digits}d', sample, count=1)

                    if suggested_pattern and suggested_pattern != file_pattern:
                        error_msg += f". Found files like: {', '.join(all_files[:3])}"
                        if len(all_files) > 3:
                            error_msg += f" (+more)"
                        error_msg += f". Try pattern: {suggested_pattern}"

            return jsonify({
                "valid": False,
                "checked": True,
                "found_count": 0,
                "file_pattern": file_pattern,
                "camera_path": str(cam_input_dir),
                "sample_files": found_files,
                "first_image_preview": None,
                "image_size": None,
                "format_detected": None,
                "container_format": is_container,
                "suggested_pattern": suggested_pattern,
                "error": error_msg
            })

        # Try to read the first image for preview
        preview_b64 = None
        image_size = None
        format_detected = None

        try:
            if is_container:
                # Read from container with camera number and index
                if '.set' in file_pattern.lower():
                    img = read_image(str(container_file), camera_no=camera, im_no=1)
                else:  # .im7
                    img = read_image(str(container_file), camera_no=camera)
                format_detected = Path(file_pattern).suffix.lstrip('.')
            else:
                img = read_image(image_files[0])
                format_detected = Path(image_files[0]).suffix.lstrip('.')

            if img is not None:
                # Convert to grayscale if needed
                if img.ndim == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img.copy()

                image_size = [int(gray.shape[1]), int(gray.shape[0])]

                # Normalize to 0-255 uint8 for display
                disp = gray.astype(float) - gray.min()
                if disp.max() > 0:
                    disp = disp / disp.max()
                disp8 = (disp * 255).astype(np.uint8)

                # Convert to base64 PNG
                preview_b64 = numpy_to_png_base64(disp8)

        except Exception as e:
            logger.warning(f"Could not read first image for preview: {e}")
            return jsonify({
                "valid": False,
                "checked": True,
                "found_count": len(image_files) if not is_container else 1,
                "file_pattern": file_pattern,
                "camera_path": str(cam_input_dir),
                "sample_files": sample_files,
                "first_image_preview": None,
                "image_size": None,
                "format_detected": format_detected,
                "container_format": is_container,
                "error": f"Images found but could not be read: {str(e)}"
            })

        return jsonify({
            "valid": True,
            "checked": True,
            "found_count": len(image_files) if not is_container else "container",
            "file_pattern": file_pattern,
            "camera_path": str(cam_input_dir),
            "sample_files": sample_files,
            "first_image_preview": preview_b64,
            "image_size": image_size,
            "format_detected": format_detected,
            "container_format": is_container,
            "error": None
        })

    except Exception as e:
        logger.error(f"Error validating planar calibration images: {e}")
        return jsonify({
            "valid": False,
            "checked": True,
            "found_count": 0,
            "file_pattern": file_pattern,
            "camera_path": None,
            "sample_files": [],
            "first_image_preview": None,
            "image_size": None,
            "format_detected": None,
            "container_format": False,
            "error": str(e)
        }), 500


@calibration_bp.route("/calibration/planar/detect_grid", methods=["POST"])
def planar_detect_grid():
    """Detect grid in calibration image using production methods"""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    image_index = int(data.get("image_index", 0))
    file_pattern = data.get("file_pattern", "calib%05d.tif")
    pattern_cols = int(data.get("pattern_cols", 10))
    pattern_rows = int(data.get("pattern_rows", 10))
    enhance_dots = bool(data.get("enhance_dots", True))
    asymmetric = bool(data.get("asymmetric", False))
    dt = float(data.get("dt", 1.0))

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])
        base_root = Path(cfg.base_paths[source_path_idx])

        # Create a temporary calibrator instance
        calibrator = PlanarCalibrator(
            source_dir=source_root,
            base_dir=base_root,
            camera_count=1,  # Just for this camera
            file_pattern=file_pattern,
            pattern_cols=pattern_cols,
            pattern_rows=pattern_rows,
            asymmetric=asymmetric,
            enhance_dots=enhance_dots,
        )

        # Get the image path
        cam_input_dir = source_root / "calibration" / f"Cam{camera}"
        if "%" in file_pattern:
            image_files = []
            i = 1
            while True:
                filename = file_pattern % i
                filepath = cam_input_dir / filename
                if filepath.exists():
                    image_files.append(str(filepath))
                    i += 1
                else:
                    break
        else:
            image_files = sorted(glob.glob(str(cam_input_dir / file_pattern)))

        if image_index >= len(image_files):
            return jsonify({"error": "Image index out of range"}), 404

        img_path = image_files[image_index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Detect grid
        found, grid_points = calibrator.detect_grid_in_image(img)

        if not found:
            return jsonify({"error": "Grid not detected", "found": False})

        # Convert to list for JSON serialization
        grid_points_list = grid_points.tolist()

        return jsonify(
            {
                "found": True,
                "grid_points": grid_points_list,
                "count": len(grid_points_list),
                "pattern_size": [pattern_cols, pattern_rows],
            }
        )

    except Exception as e:
        logger.error(f"Error detecting grid: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/calibration/planar/compute", methods=["POST"])
def planar_compute():
    """Compute full planar calibration using production methods"""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    image_index = int(data.get("image_index", 0))
    file_pattern = data.get("file_pattern", "calib%05d.tif")
    pattern_cols = int(data.get("pattern_cols", 10))
    pattern_rows = int(data.get("pattern_rows", 10))
    dot_spacing_mm = float(data.get("dot_spacing_mm", 28.89))
    enhance_dots = bool(data.get("enhance_dots", True))
    asymmetric = bool(data.get("asymmetric", False))
    dt = float(data.get("dt", 1.0))

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])
        base_root = Path(cfg.base_paths[source_path_idx])
        cam_output_base = base_root / "calibration" / f"Cam{camera}"

        # Get the image path using same logic as get_image
        cam_input_dir = source_root / "calibration" / f"Cam{camera}"
        if "%" in file_pattern:
            image_files = []
            i = 1
            while True:
                filename = file_pattern % i
                filepath = cam_input_dir / filename
                if filepath.exists():
                    image_files.append(str(filepath))
                    i += 1
                else:
                    break
        else:
            image_files = sorted(glob.glob(str(cam_input_dir / file_pattern)))

        logger.info(
            f"Compute: Found {len(image_files)} images for pattern '{file_pattern}'"
        )
        logger.info(f"Compute: All files: {[Path(f).name for f in image_files[:5]]}")

        if image_index >= len(image_files):
            return (
                jsonify(
                    {
                        "error": f"Image index {image_index} out of range (0-{len(image_files)-1})"
                    }
                ),
                404,
            )

        img_path = image_files[image_index]
        logger.info(
            f"Compute: Processing image at index {image_index}: {Path(img_path).name}"
        )

        # Create calibrator instance
        calibrator = PlanarCalibrator(
            source_dir=source_root,
            base_dir=base_root,
            camera_count=1,
            file_pattern=file_pattern,
            pattern_cols=pattern_cols,
            pattern_rows=pattern_rows,
            dot_spacing_mm=dot_spacing_mm,
            asymmetric=asymmetric,
            enhance_dots=enhance_dots,
            dt=dt,  # CRITICAL: Pass dt to calibrator
            selected_image_idx=image_index + 1,  # 1-based index for production script
        )

        # Run calibration for this camera and image
        calibrator.process_camera(camera)

        # After batch, load results for requested image index
        indices_folder = cam_output_base / "indices"
        model_folder = cam_output_base / "model"
        dewarp_folder = cam_output_base / "dewarp"
        grid_file = indices_folder / f"indexing_{image_index+1}.mat"
        model_file = model_folder / "camera_model.mat"
        grid_png_file = indices_folder / f"indexes_{image_index+1}.png"
        dewarped_file = dewarp_folder / f"dewarped_{image_index+1}.tif"
        results = {}

        # Load grid data first to get pattern info
        grid_data_dict = None
        if grid_file.exists():
            grid_data = scipy.io.loadmat(
                grid_file, struct_as_record=False, squeeze_me=True
            )
            grid_points = grid_data["grid_points"]
            pattern_size = grid_data["pattern_size"]
            dot_spacing_mm = float(grid_data["dot_spacing_mm"])
            cols, rows = pattern_size
            px_per_mm = None
            if grid_points.shape[0] >= 2:
                first_row = grid_points[:cols]
                x_vals = first_row[:, 0]
                px_per_mm = (
                    (x_vals.max() - x_vals.min()) / (cols - 1) / dot_spacing_mm
                    if dot_spacing_mm > 0
                    else None
                )
            grid_data_dict = {
                "grid_points": grid_points.tolist(),
                "homography": grid_data["homography"].tolist(),
                "reprojection_error": float(grid_data["reprojection_error"]),
                "reprojection_error_x_mean": float(
                    grid_data.get("reprojection_error_x_mean", 0)
                ),
                "reprojection_error_y_mean": float(
                    grid_data.get("reprojection_error_y_mean", 0)
                ),
                "pattern_size": pattern_size.tolist(),
                "dot_spacing_mm": dot_spacing_mm,
                "pixels_per_mm": px_per_mm,
                "timestamp": str(grid_data.get("timestamp", "")),
                "original_filename": str(grid_data.get("original_filename", "")),
            }
            results["grid_data"] = grid_data_dict

        # Load grid PNG visualization
        if grid_png_file.exists():
            import base64

            try:
                with open(grid_png_file, "rb") as f:
                    grid_png_b64 = base64.b64encode(f.read()).decode("utf-8")
                if grid_data_dict:
                    grid_data_dict["grid_png"] = grid_png_b64
                else:
                    results["grid_png"] = grid_png_b64
                logger.info(f"Loaded grid PNG from {grid_png_file}")
            except Exception as e:
                logger.error(f"Error loading grid PNG {grid_png_file}: {e}")

        # Load camera model
        if model_file.exists():
            model_data = scipy.io.loadmat(
                model_file, struct_as_record=False, squeeze_me=True
            )
            results["camera_model"] = {
                "camera_matrix": model_data["camera_matrix"].tolist(),
                "dist_coeffs": model_data["dist_coeffs"].tolist(),
                "reprojection_error": float(model_data["reprojection_error"]),
                "reprojection_error_x_mean": float(
                    model_data.get("reprojection_error_x_mean", 0)
                ),
                "reprojection_error_y_mean": float(
                    model_data.get("reprojection_error_y_mean", 0)
                ),
                "focal_length": [
                    float(model_data["camera_matrix"][0, 0]),
                    float(model_data["camera_matrix"][1, 1]),
                ],
                "principal_point": [
                    float(model_data["camera_matrix"][0, 2]),
                    float(model_data["camera_matrix"][1, 2]),
                ],
                "timestamp": str(model_data.get("timestamp", "")),
            }

        if dewarped_file.exists():
            dewarped_img = cv2.imread(str(dewarped_file), cv2.IMREAD_UNCHANGED)
            if dewarped_img is not None:
                if dewarped_img.ndim == 3:
                    dewarped_gray = cv2.cvtColor(dewarped_img, cv2.COLOR_BGR2GRAY)
                else:
                    dewarped_gray = dewarped_img.copy()
                disp = dewarped_gray - dewarped_gray.min()
                if disp.max() > 0:
                    disp = disp / disp.max()
                disp8 = (disp * 255).astype(np.uint8)
                results["dewarped_image"] = numpy_to_png_base64(disp8)
                results["dewarped_size"] = [
                    int(dewarped_gray.shape[1]),
                    int(dewarped_gray.shape[0]),
                ]

        return jsonify(
            {
                "status": "success",
                "results": results,
                "processed_file": Path(img_path).name,
                "image_index": image_index,
                "output_files": {
                    "grid": str(grid_file),
                    "model": str(model_file),
                    "grid_png": str(grid_png_file),
                    "dewarped": str(dewarped_file),
                },
            }
        )

    except Exception as e:
        logger.error(f"Error computing planar calibration: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/calibration/planar/load_results", methods=["GET"])
def planar_load_results():
    """Load previously computed planar calibration results"""
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))
    image_index = request.args.get("image_index", default=0, type=int)

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])
        cam_output_base = base_root / "calibration" / f"Cam{camera}"

        # Check if results exist
        grid_file = cam_output_base / "grid" / f"indexing_{image_index}.mat"
        model_file = cam_output_base / "models" / f"{image_index}.mat"

        if not grid_file.exists() or not model_file.exists():
            return jsonify({"exists": False, "message": "No saved results found"})

        # Load the results (same logic as in planar_compute)
        results = {}

        # Load grid data
        grid_data = scipy.io.loadmat(grid_file, struct_as_record=False, squeeze_me=True)
        # Estimate pixels per mm from grid points and dot spacing
        grid_points = grid_data["grid_points"]
        pattern_size = grid_data["pattern_size"]
        dot_spacing_mm = float(grid_data["dot_spacing_mm"])
        cols, rows = pattern_size
        # Only estimate if enough points
        px_per_mm = None
        if grid_points.shape[0] >= 2:
            # Use first row of grid points
            first_row = grid_points[:cols]
            x_vals = first_row[:, 0]
            px_per_mm = (
                (x_vals.max() - x_vals.min()) / (cols - 1) / dot_spacing_mm
                if dot_spacing_mm > 0
                else None
            )
        results["grid_data"] = {
            "grid_points": grid_points.tolist(),
            "homography": grid_data["homography"].tolist(),
            "reprojection_error": float(grid_data["reprojection_error"]),
            "reprojection_error_x_mean": float(
                grid_data.get("reprojection_error_x_mean", 0)
            ),
            "reprojection_error_y_mean": float(
                grid_data.get("reprojection_error_y_mean", 0)
            ),
            "pattern_size": pattern_size.tolist(),
            "dot_spacing_mm": dot_spacing_mm,
            "pixels_per_mm": px_per_mm,
            "timestamp": str(grid_data.get("timestamp", "")),
            "original_filename": str(grid_data.get("original_filename", "")),
        }

        # Try to load saved grid visualization PNG
        grid_png_file = cam_output_base / "indices" / f"indexes_{image_index}.png"
        if grid_png_file.exists():
            try:
                with open(grid_png_file, "rb") as f:
                    grid_png_b64 = base64.b64encode(f.read()).decode("utf-8")
                results["grid_data"]["grid_png"] = grid_png_b64
            except Exception as e:
                logger.warning(f"Could not load grid PNG: {e}")

        # Load camera model
        model_data = scipy.io.loadmat(
            model_file, struct_as_record=False, squeeze_me=True
        )
        results["camera_model"] = {
            "camera_matrix": model_data["camera_matrix"].tolist(),
            "dist_coeffs": model_data["dist_coeffs"].tolist(),
            "reprojection_error": float(model_data["reprojection_error"]),
            "reprojection_error_x_mean": float(
                model_data.get("reprojection_error_x_mean", 0)
            ),
            "reprojection_error_y_mean": float(
                model_data.get("reprojection_error_y_mean", 0)
            ),
            "focal_length": [
                float(model_data["camera_matrix"][0, 0]),
                float(model_data["camera_matrix"][1, 1]),
            ],
            "principal_point": [
                float(model_data["camera_matrix"][0, 2]),
                float(model_data["camera_matrix"][1, 2]),
            ],
        }

        # Try to load dewarped image
        dewarped_pattern = f"{grid_data.get('original_filename', 'unknown').split('.')[0]}_dewarped.tif"
        dewarped_file = cam_output_base / "dewarped" / dewarped_pattern

        if dewarped_file.exists():
            dewarped_img = cv2.imread(str(dewarped_file), cv2.IMREAD_UNCHANGED)
            if dewarped_img is not None:
                if dewarped_img.ndim == 3:
                    dewarped_gray = cv2.cvtColor(dewarped_img, cv2.COLOR_BGR2GRAY)
                else:
                    dewarped_gray = dewarped_img.copy()

                disp = dewarped_gray - dewarped_gray.min()
                if disp.max() > 0:
                    disp = disp / disp.max()
                disp8 = (disp * 255).astype(np.uint8)

                results["dewarped_image"] = numpy_to_png_base64(disp8)
                results["dewarped_size"] = [
                    int(dewarped_gray.shape[1]),
                    int(dewarped_gray.shape[0]),
                ]

        return jsonify({"exists": True, "results": results})

    except Exception as e:
        logger.error(f"Error loading planar calibration results: {e}")
        return jsonify({"error": str(e)}), 500


def _process_single_image(idx, source_root, base_root, file_pattern, pattern_cols, pattern_rows, dot_spacing_mm, asymmetric, enhance_dots, dt, camera):
    """Helper function for parallel processing of single calibration images"""
    try:
        # Create a separate calibrator instance for this image
        calibrator = PlanarCalibrator(
            source_dir=source_root,
            base_dir=base_root,
            camera_count=1,
            file_pattern=file_pattern,
            pattern_cols=pattern_cols,
            pattern_rows=pattern_rows,
            dot_spacing_mm=dot_spacing_mm,
            asymmetric=asymmetric,
            enhance_dots=enhance_dots,
            dt=dt,
            selected_image_idx=idx + 1,  # 1-based
        )
        calibrator.process_camera(camera)
        return idx, True
    except Exception as e:
        logger.error(f"Error processing image {idx}: {e}")
        return idx, False


@calibration_bp.route("/calibration/planar/calibrate_all", methods=["POST"])
def planar_calibrate_all():
    """Start batch planar calibration job for all images for a camera"""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    file_pattern = data.get("file_pattern", "calib%05d.tif")
    pattern_cols = int(data.get("pattern_cols", 10))
    pattern_rows = int(data.get("pattern_rows", 10))
    dot_spacing_mm = float(data.get("dot_spacing_mm", 28.89))
    enhance_dots = bool(data.get("enhance_dots", True))
    asymmetric = bool(data.get("asymmetric", False))
    dt = float(data.get("dt", 1.0))

    job_id = str(uuid.uuid4())

    def run_planar_calibration():
        try:
            calibration_jobs[job_id] = {
                "status": "starting",
                "progress": 0,
                "processed_indices": [],
                "total_images": 0,
                "start_time": time.time(),
                "error": None,
            }
            cfg = get_config()
            source_root = Path(cfg.source_paths[source_path_idx])
            base_root = Path(cfg.base_paths[source_path_idx])
            cam_input_dir = source_root / "calibration" / f"Cam{camera}"
            # Find calibration images
            if "%" in file_pattern:
                image_files = []
                i = 1
                while True:
                    filename = file_pattern % i
                    filepath = cam_input_dir / filename
                    if filepath.exists():
                        image_files.append(str(filepath))
                        i += 1
                    else:
                        break
            else:
                image_files = sorted(glob.glob(str(cam_input_dir / file_pattern)))
            total_images = len(image_files)
            calibration_jobs[job_id]["total_images"] = total_images
            if total_images == 0:
                calibration_jobs[job_id]["status"] = "failed"
                calibration_jobs[job_id]["error"] = "No calibration images found"
                return
            # Create calibrator
            calibrator = PlanarCalibrator(
                source_dir=source_root,
                base_dir=base_root,
                camera_count=1,
                file_pattern=file_pattern,
                pattern_cols=pattern_cols,
                pattern_rows=pattern_rows,
                dot_spacing_mm=dot_spacing_mm,
                asymmetric=asymmetric,
                enhance_dots=enhance_dots,
                dt=dt,
            )
            
            # Process all images in parallel
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers = min(os.cpu_count(), total_images, 8)) as executor:
                futures = [executor.submit(_process_single_image, idx, source_root, base_root, file_pattern, pattern_cols, pattern_rows, dot_spacing_mm, asymmetric, enhance_dots, dt, camera) for idx in range(total_images)]
                for future in as_completed(futures):
                    idx, success = future.result()
                    calibration_jobs[job_id]["processed_indices"].append(idx)
                    calibration_jobs[job_id]["progress"] = int(
                        (len(calibration_jobs[job_id]["processed_indices"]) / total_images) * 100
                    )
                    calibration_jobs[job_id]["status"] = "running"
            calibration_jobs[job_id]["status"] = "completed"
            calibration_jobs[job_id]["progress"] = 100
        except Exception as e:
            logger.error(f"Planar calibration job {job_id} failed: {e}")
            calibration_jobs[job_id]["status"] = "failed"
            calibration_jobs[job_id]["error"] = str(e)

    thread = threading.Thread(target=run_planar_calibration)
    thread.daemon = True
    thread.start()
    return jsonify(
        {
            "job_id": job_id,
            "status": "starting",
            "message": f"Planar calibration job started for camera {camera}",
            "total_images": None,
        }
    )


@calibration_bp.route(
    "/calibration/planar/calibrate_all/status/<job_id>", methods=["GET"]
)
def planar_calibrate_all_status(job_id):
    """Get batch planar calibration job status"""
    if job_id not in calibration_jobs:
        return jsonify({"error": "Job not found"}), 404
    job_data = calibration_jobs[job_id].copy()
    if "start_time" in job_data:
        elapsed = time.time() - job_data["start_time"]
        job_data["elapsed_time"] = elapsed
        if job_data["status"] == "running" and job_data.get("progress", 0) > 0:
            estimated_total = elapsed / (job_data["progress"] / 100.0)
            job_data["estimated_remaining"] = max(0, estimated_total - elapsed)
    return jsonify(job_data)


# ============================================================================
# SCALE FACTOR CALIBRATION ROUTES
# ============================================================================


@calibration_bp.route("/calibration/scale_factor/calibrate_vectors", methods=["POST"])
def scale_factor_calibrate_vectors():
    """Start scale factor calibration job with progress tracking for all cameras."""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    dt = float(data.get("dt", 1.0))
    px_per_mm = float(data.get("px_per_mm", 1.0))
    image_count = int(data.get("image_count", 1000))
    type_name = data.get("type_name", "instantaneous")

    job_id = str(uuid.uuid4())

    def run_scale_factor_calibration():
        try:
            cfg = get_config()
            camera_numbers = cfg.camera_numbers
            total_cameras = len(camera_numbers)
            
            # Initialize job with camera-aware tracking
            scale_factor_jobs[job_id] = {
                "status": "starting",
                "progress": 0,
                "processed_runs": 0,
                "processed_files": 0,
                "total_files": 0,
                "current_camera": None,
                "total_cameras": total_cameras,
                "processed_cameras": 0,
                "camera_progress": {},
                "start_time": time.time(),
                "error": None,
            }

            base_root = Path(cfg.base_paths[source_path_idx])
            if type_name == "instantaneous":
                runs = cfg.instantaneous_runs
            else:
                runs = cfg.instantaneous_runs  # Default to instantaneous if unknown
            
            # First pass: count total files across all cameras
            total_files_all_cameras = 0
            camera_file_counts = {}
            
            for cam_num in camera_numbers:
                paths_uncal = get_data_paths(
                    base_dir=base_root,
                    num_frame_pairs=cfg.num_frame_pairs,
                    cam=cam_num,
                    type_name=type_name,
                    use_uncalibrated=True,
                )
                data_dir_uncal = paths_uncal["data_dir"]
                coords_path_uncal = data_dir_uncal / "coordinates.mat"
                
                camera_files = 0
                # Count coordinate file
                if coords_path_uncal.exists():
                    camera_files += 1
                
                # Count vector files
                for run in range(1, image_count + 1):
                    vector_file_uncal = data_dir_uncal / (cfg.vector_format % run)
                    if vector_file_uncal.exists():
                        camera_files += 1
                
                camera_file_counts[cam_num] = camera_files
                total_files_all_cameras += camera_files
                scale_factor_jobs[job_id]["camera_progress"][f"Cam{cam_num}"] = {
                    "total_files": camera_files,
                    "processed_files": 0,
                    "status": "pending"
                }
            
            scale_factor_jobs[job_id]["total_files"] = total_files_all_cameras
            
            if total_files_all_cameras == 0:
                scale_factor_jobs[job_id]["status"] = "failed"
                scale_factor_jobs[job_id]["error"] = "No data files found to process for any camera"
                return
            
            scale_factor_jobs[job_id]["status"] = "running"
            
            # Process each camera sequentially
            total_processed_files = 0
            
            for cam_idx, cam_num in enumerate(camera_numbers, 1):
                scale_factor_jobs[job_id]["current_camera"] = cam_num
                scale_factor_jobs[job_id]["camera_progress"][f"Cam{cam_num}"]["status"] = "running"
                logger.info(f"Processing camera {cam_num} ({cam_idx}/{total_cameras})")
                
                paths_uncal = get_data_paths(
                    base_dir=base_root,
                    num_frame_pairs=cfg.num_frame_pairs,
                    cam=cam_num,
                    type_name=type_name,
                    use_uncalibrated=True,
                )
                paths_calib = get_data_paths(
                    base_dir=base_root,
                    num_frame_pairs=cfg.num_frame_pairs,
                    cam=cam_num,
                    type_name=type_name,
                    use_uncalibrated=False,
                )
                data_dir_uncal = paths_uncal["data_dir"]
                data_dir_cal = paths_calib["data_dir"]
                data_dir_cal.mkdir(parents=True, exist_ok=True)
                coords_path_uncal = data_dir_uncal / "coordinates.mat"
                coords_path_cal = data_dir_cal / "coordinates.mat"
                
                # Collect vector files for this camera
                vector_files = []
                for run in range(1, image_count + 1):
                    vector_file_uncal = data_dir_uncal / (cfg.vector_format % run)
                    vector_file_cal = data_dir_cal / (cfg.vector_format % run)
                    if vector_file_uncal.exists():
                        vector_files.append((run, vector_file_uncal, vector_file_cal))
                
                camera_processed_files = 0
                
                # --- Process coordinates as struct array ---
                if coords_path_uncal.exists():
                    mat = scipy.io.loadmat(
                        str(coords_path_uncal), struct_as_record=False, squeeze_me=True
                    )
                    coordinates = mat.get("coordinates", None)
                    if coordinates is not None:
                        # Build output struct array
                        coord_dtype = np.dtype([("x", "O"), ("y", "O")])
                        out_coords = np.empty(len(coordinates), dtype=coord_dtype)
                        processed_runs = 0
                        # Zero-base x and y before offset
                        for run_idx, run_coords in enumerate(coordinates):
                            x = getattr(run_coords, "x", None)
                            y = getattr(run_coords, "y", None)
                            if x is not None and y is not None:
                                # Zero-base: subtract first value
                                x0 = x.flat[0] if x.size > 0 else 0
                                y0 = y.flat[0] if y.size > 0 else 0
                                x_calib = (x - x0) / px_per_mm 
                                y_calib = -np.flipud((y - y0) / px_per_mm)
                                out_coords[run_idx] = (x_calib, y_calib)
                                processed_runs += 1
                            else:
                                out_coords[run_idx] = (np.array([]), np.array([]))
                        scipy.io.savemat(str(coords_path_cal), {"coordinates": out_coords}, do_compression=True)
                        logger.info(f"Cam{cam_num}: Updated coordinates for {processed_runs} runs")
                    
                    camera_processed_files += 1
                    total_processed_files += 1
                    scale_factor_jobs[job_id]["processed_files"] = total_processed_files
                    scale_factor_jobs[job_id]["camera_progress"][f"Cam{cam_num}"]["processed_files"] = camera_processed_files
                    scale_factor_jobs[job_id]["progress"] = int(
                        (total_processed_files / total_files_all_cameras) * 100
                    )
                
                # --- Process vector files as struct array ---
                if len(vector_files) > 0:
                    # Process vector files in parallel
                    vector_file_args = [
                        (run, vector_file_uncal, vector_file_cal, px_per_mm, dt)
                        for run, vector_file_uncal, vector_file_cal in vector_files
                    ]
                    successful_files = 0
                    failed_files = 0

                    with ProcessPoolExecutor(max_workers=min(4, len(vector_files))) as executor:
                        futures = [executor.submit(_process_vector_file_for_calibration, args) for args in vector_file_args]
                        for future in as_completed(futures):
                            try:
                                result = future.result()  # Get the return value
                                if result:
                                    successful_files += 1
                                else:
                                    failed_files += 1
                            except Exception as e:
                                logger.error(f"Future failed with exception: {e}")
                                failed_files += 1
                            
                            camera_processed_files += 1
                            total_processed_files += 1
                            scale_factor_jobs[job_id]["processed_files"] = total_processed_files
                            scale_factor_jobs[job_id]["camera_progress"][f"Cam{cam_num}"]["processed_files"] = camera_processed_files
                            scale_factor_jobs[job_id]["progress"] = int(
                                (total_processed_files / total_files_all_cameras) * 100
                            )

                    # Check if any files were successfully processed for this camera
                    if failed_files > 0:
                        logger.warning(f"Cam{cam_num}: Completed with {failed_files} failed vector files")

                    if successful_files == 0 and len(vector_files) > 0:
                        logger.error(f"Cam{cam_num}: No vector files were successfully processed")
                        scale_factor_jobs[job_id]["camera_progress"][f"Cam{cam_num}"]["status"] = "failed"
                    else:
                        logger.info(f"Cam{cam_num}: {successful_files} vector files processed successfully")
                        scale_factor_jobs[job_id]["camera_progress"][f"Cam{cam_num}"]["status"] = "completed"
                else:
                    scale_factor_jobs[job_id]["camera_progress"][f"Cam{cam_num}"]["status"] = "completed"
                
                scale_factor_jobs[job_id]["processed_cameras"] = cam_idx

            # Final status
            scale_factor_jobs[job_id]["status"] = "completed"
            scale_factor_jobs[job_id]["progress"] = 100
            scale_factor_jobs[job_id]["current_camera"] = None
            logger.info(
                f"Scale factor calibration completed for all {total_cameras} cameras. Total files processed: {total_processed_files}/{total_files_all_cameras}"
            )
            
        except Exception as e:
            logger.error(f"Scale factor calibration job {job_id} failed: {e}")
            scale_factor_jobs[job_id]["status"] = "failed"
            scale_factor_jobs[job_id]["error"] = str(e)

    # Start job in background thread
    thread = threading.Thread(target=run_scale_factor_calibration)
    thread.daemon = True
    thread.start()

    cfg = get_config()
    camera_numbers = cfg.camera_numbers

    return jsonify(
        {
            "job_id": job_id,
            "status": "starting",
            "message": f"Scale factor calibration job started for {len(camera_numbers)} camera(s): {camera_numbers}",
            "cameras": camera_numbers,
            "image_count": image_count,
        }
    )


def _process_vector_file_for_calibration(args):
    """Helper function for parallel vector file processing."""
    run, vector_file_uncal, vector_file_cal, px_per_mm, dt = args
    try:
        logger.info(f"Processing vector file: {vector_file_uncal}")
        mat = scipy.io.loadmat(
            str(vector_file_uncal), struct_as_record=False, squeeze_me=True
        )
        # Only support struct format (not cell arrays)
        if "piv_result" not in mat:
            logger.warning(
                f"Vector file {vector_file_uncal} missing 'piv_result' field."
            )
            return False
        
        piv_result = mat["piv_result"]
        # Calibrate all runs in piv_result (assume struct array)
        piv_dtype = np.dtype([("ux", "O"), ("uy", "O"), ("b_mask", "O")])
        out_piv = np.empty(len(piv_result), dtype=piv_dtype)
        for idx, cell in enumerate(piv_result):
            ux = getattr(cell, "ux", None)
            uy = getattr(cell, "uy", None)
            b_mask = getattr(
                cell,
                "b_mask",
                np.zeros_like(ux) if ux is not None else np.array([]),
            )
            if ux is not None and uy is not None:
                ux_calib = ux / px_per_mm / dt / 1000
                uy_calib = uy / px_per_mm / dt / 1000
                out_piv[idx] = (ux_calib, uy_calib, b_mask)
            else:
                out_piv[idx] = (np.array([]), np.array([]), np.array([]))
        scipy.io.savemat(str(vector_file_cal), {"piv_result": out_piv}, do_compression=True)
        return True
    except Exception as e:
        logger.error(f"Error processing vector file {vector_file_uncal}: {e}", exc_info=True)
        return False


@calibration_bp.route("/calibration/scale_factor/status/<job_id>", methods=["GET"])
def scale_factor_status(job_id):
    """Get scale factor calibration job status"""
    if job_id not in scale_factor_jobs:
        return jsonify({"error": "Job not found"}), 404

    job_data = scale_factor_jobs[job_id].copy()

    # Add timing info
    if "start_time" in job_data:
        elapsed = time.time() - job_data["start_time"]
        job_data["elapsed_time"] = elapsed

        if job_data["status"] == "running" and job_data.get("progress", 0) > 0:
            estimated_total = elapsed / (job_data["progress"] / 100.0)
            job_data["estimated_remaining"] = max(0, estimated_total - elapsed)

    return jsonify(job_data)


@calibration_bp.route("/calibration/status", methods=["GET"])
def calibration_status():
    """Get calibration status - unified endpoint for all calibration types"""
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
# POLYNOMIAL CALIBRATION ROUTES
# ============================================================================


@calibration_bp.route("/calibration/polynomial/read_xml", methods=["POST"])
def read_polynomial_calibration_xml():
    """Read Calibration.xml and extract polynomial coefficients for all cameras"""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))

    try:
        cfg = get_config()
        # Ensure source_paths exists and has the index
        if not hasattr(cfg, "source_paths") or source_path_idx >= len(cfg.source_paths):
            return jsonify({"error": "Invalid source_path_idx"}), 400

        source_root = Path(cfg.source_paths[source_path_idx])
        xml_path = source_root / "Properties" / "Calibration" / "Calibration.xml"

        if not xml_path.exists():
            return jsonify({"error": f"Calibration.xml not found at {xml_path}"}), 404

        logger.info(f"Reading Calibration.xml from: {xml_path}")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        cameras_data = {}

        # Find all CoordinateMapper elements
        mappers = root.findall(".//CoordinateMapper")

        for mapper in mappers:
            cam_id = mapper.get("CameraIdentifier")
            if not cam_id:
                continue

            # Initialize camera entry
            cam_data = {}

            # Get PolynomialParameters -> PolynomialMapping
            poly_params = mapper.find("PolynomialParameters")
            if poly_params is None:
                continue

            mapping = poly_params.find("PolynomialMapping")
            if mapping is None:
                continue

            # Extract Origin
            origin = mapping.find("Origin")
            if origin is not None:
                cam_data["origin"] = {k: float(v) for k, v in origin.attrib.items()}

            # Extract NormalisationFactor
            norm = mapping.find("NormalisationFactor")
            if norm is not None:
                cam_data["normalisation"] = {
                    k: float(v) for k, v in norm.attrib.items()
                }

            # Extract Polynomial3rdOrder Coefficients
            poly3 = mapping.find("Polynomial3rdOrder")
            if poly3 is not None:
                coeffs_a = poly3.find("CoefficientsA")
                if coeffs_a is not None:
                    cam_data["coefficients_a"] = {
                        k: float(v) for k, v in coeffs_a.attrib.items()
                    }

                coeffs_b = poly3.find("CoefficientsB")
                if coeffs_b is not None:
                    cam_data["coefficients_b"] = {
                        k: float(v) for k, v in coeffs_b.attrib.items()
                    }

            cameras_data[cam_id] = cam_data

        return jsonify(
            {"status": "success", "file": str(xml_path), "cameras": cameras_data}
        )

    except Exception as e:
        logger.error(f"Error reading Calibration.xml: {e}")
        return jsonify({"error": str(e)}), 500

