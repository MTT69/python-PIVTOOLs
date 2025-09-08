import glob
import json  # Add missing import
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger

# Import production calibration classes
from calibration.calibration_planar.planar_calibration_production import (
    PlanarCalibrator,
)
from common.utils import camera_number, numpy_to_png_base64
from config import get_config
from paths import get_data_paths
from plotting.app.views import extract_coordinates
from stereo_reconstruction.stereo_calibration_production import StereoCalibrator


def cache_key(source_path_idx, camera):
    return (int(source_path_idx), str(camera))


calibration_cache = {}
calibration_bp = Blueprint("calibration", __name__)


def _cfg():
    return get_config()


def _resolve_calibration_image(source_path_idx: int, camera: int, index: int = 1):
    cfg = get_config()
    source_root = Path(cfg.source_paths[source_path_idx])
    # Use get_data_paths to get calibration directory
    paths = get_data_paths(
        base_dir=source_root,
        num_images=1,
        cam=camera,
        type_name="",
        calibration=True,
    )
    calib_dir = paths["calib_dir"]
    filename = cfg.calibration_filename(index)
    img_path = calib_dir / filename
    return img_path


# Global job tracking
calibration_jobs = {}

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
            num_images=getattr(cfg, "num_images", 1),
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

        coordinates[run_idx].x = cx
        coordinates[run_idx].y = cy

        scipy.io.savemat(coords_path, {"coordinates": coordinates})
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

    # Get default file pattern from config if using default
    if file_pattern == "calib%05d.tif":
        try:
            cfg = get_config()
            # Try to get from calibration.pinhole.file_pattern
            pinhole_config = cfg.calibration.get("pinhole", {})
            file_pattern = pinhole_config.get("file_pattern", file_pattern)
            logger.info(f"Using file pattern from config: {file_pattern}")
        except Exception as e:
            logger.warning(f"Could not load file pattern from config: {e}")

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

    # Get default values from config
    try:
        cfg = get_config()
        pinhole_config = cfg.calibration.get("pinhole", {})

        # Only override if still using defaults
        if file_pattern == "calib%05d.tif":
            file_pattern = pinhole_config.get("file_pattern", file_pattern)
        if pattern_cols == 10:
            pattern_cols = int(pinhole_config.get("pattern_cols", pattern_cols))
        if pattern_rows == 10:
            pattern_rows = int(pinhole_config.get("pattern_rows", pattern_rows))
        if dt == 1.0:
            dt = float(pinhole_config.get("dt", dt))

        logger.info(
            f"Using config values - file_pattern: {file_pattern}, pattern: {pattern_cols}x{pattern_rows}, dt: {dt}"
        )
    except Exception as e:
        logger.warning(f"Could not load config defaults: {e}")

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

    # Get default values from config
    try:
        cfg = get_config()
        pinhole_config = cfg.calibration.get("pinhole", {})

        # Only override if still using defaults
        if file_pattern == "calib%05d.tif":
            file_pattern = pinhole_config.get("file_pattern", file_pattern)
        if pattern_cols == 10:
            pattern_cols = int(pinhole_config.get("pattern_cols", pattern_cols))
        if pattern_rows == 10:
            pattern_rows = int(pinhole_config.get("pattern_rows", pattern_rows))
        if dot_spacing_mm == 28.89:
            dot_spacing_mm = float(pinhole_config.get("dot_spacing_mm", dot_spacing_mm))
        if dt == 1.0:
            dt = float(pinhole_config.get("dt", dt))

        logger.info(
            f"Using config values - file_pattern: {file_pattern}, pattern: {pattern_cols}x{pattern_rows}, spacing: {dot_spacing_mm}mm, dt: {dt}s"
        )
    except Exception as e:
        logger.warning(f"Could not load config defaults: {e}")

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])
        base_root = Path(cfg.base_paths[source_path_idx])

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
        )

        # Process this single image
        cam_output_base = base_root / "calibration" / f"Cam{camera}"
        objp = calibrator.make_object_points()
        objp_2d = objp[:, :2]

        # Setup directories
        (cam_output_base / "grid").mkdir(parents=True, exist_ok=True)
        (cam_output_base / "models").mkdir(parents=True, exist_ok=True)
        (cam_output_base / "dewarped").mkdir(parents=True, exist_ok=True)

        # Process the image - pass the actual image index from the UI
        calibrator._process_single_image(
            img_path, image_index, cam_output_base, objp_2d
        )

        # Load the saved results
        grid_file = cam_output_base / "grid" / f"indexing_{image_index}.mat"
        model_file = cam_output_base / "models" / f"{image_index}.mat"
        dewarped_file = (
            cam_output_base / "dewarped" / f"{Path(img_path).stem}_dewarped.tif"
        )

        logger.info("Compute: Looking for results files:")
        logger.info(f"  Grid: {grid_file} (exists: {grid_file.exists()})")
        logger.info(f"  Model: {model_file} (exists: {model_file.exists()})")
        logger.info(f"  Dewarped: {dewarped_file} (exists: {dewarped_file.exists()})")

        results = {}
        if grid_file.exists():
            grid_data = scipy.io.loadmat(
                grid_file, struct_as_record=False, squeeze_me=True
            )
            results["grid_data"] = {
                "grid_points": grid_data["grid_points"].tolist(),
                "homography": grid_data["homography"].tolist(),
                "reprojection_error": float(grid_data["reprojection_error"]),
                "pattern_size": grid_data["pattern_size"].tolist(),
                "dot_spacing_mm": float(grid_data["dot_spacing_mm"]),
            }

        if model_file.exists():
            model_data = scipy.io.loadmat(
                model_file, struct_as_record=False, squeeze_me=True
            )
            results["camera_model"] = {
                "camera_matrix": model_data["camera_matrix"].tolist(),
                "dist_coeffs": model_data["dist_coeffs"].tolist(),
                "reprojection_error": float(model_data["reprojection_error"]),
                "focal_length": [
                    float(model_data["camera_matrix"][0, 0]),
                    float(model_data["camera_matrix"][1, 1]),
                ],
                "principal_point": [
                    float(model_data["camera_matrix"][0, 2]),
                    float(model_data["camera_matrix"][1, 2]),
                ],
            }

        # Load and encode dewarped image
        if dewarped_file.exists():
            dewarped_img = cv2.imread(str(dewarped_file), cv2.IMREAD_UNCHANGED)
            if dewarped_img is not None:
                # Convert to grayscale if needed and normalize for display
                if dewarped_img.ndim == 3:
                    dewarped_gray = cv2.cvtColor(dewarped_img, cv2.COLOR_BGR2GRAY)
                else:
                    dewarped_gray = dewarped_img.copy()

                # Normalize for display
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
        results["grid_data"] = {
            "grid_points": grid_data["grid_points"].tolist(),
            "homography": grid_data["homography"].tolist(),
            "reprojection_error": float(grid_data["reprojection_error"]),
            "pattern_size": grid_data["pattern_size"].tolist(),
            "dot_spacing_mm": float(grid_data["dot_spacing_mm"]),
            "timestamp": str(grid_data.get("timestamp", "")),
        }

        # Load camera model
        model_data = scipy.io.loadmat(
            model_file, struct_as_record=False, squeeze_me=True
        )
        results["camera_model"] = {
            "camera_matrix": model_data["camera_matrix"].tolist(),
            "dist_coeffs": model_data["dist_coeffs"].tolist(),
            "reprojection_error": float(model_data["reprojection_error"]),
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


# ============================================================================
# STEREO CALIBRATION ROUTES
# ============================================================================


def compute_camera_positions_and_angle(R, T):
    # Camera 1 at origin, Camera 2 at T (in Camera 1's frame)
    cam1_pos = np.array([0, 0, 0])
    cam2_pos = T.flatten()
    # Angle between cameras (from translation vector)
    norm_T = np.linalg.norm(cam2_pos)
    angle_deg = None
    if norm_T > 0:
        # If rotation matrix is available, get the angle between z axes
        z1 = np.array([0, 0, 1])
        z2 = R @ z1
        cos_theta = np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2))
        angle_deg = float(np.arccos(np.clip(cos_theta, -1, 1)) * 180.0 / np.pi)
    return cam1_pos.tolist(), cam2_pos.tolist(), angle_deg


@calibration_bp.route("/calibration/stereo/compute", methods=["POST"])
def stereo_compute():
    """Compute stereo calibration for a camera pair"""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera_pair = data.get("camera_pair", [1, 2])
    file_pattern = data.get("file_pattern", "planar_calibration_plate_*.tif")
    pattern_cols = int(data.get("pattern_cols", 10))
    pattern_rows = int(data.get("pattern_rows", 10))
    dot_spacing_mm = float(data.get("dot_spacing_mm", 28.89))
    enhance_dots = bool(data.get("enhance_dots", True))
    asymmetric = bool(data.get("asymmetric", False))

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])
        base_root = Path(cfg.base_paths[source_path_idx])

        # Create stereo calibrator instance
        calibrator = StereoCalibrator(
            source_dir=source_root,
            base_dir=base_root,
            camera_pairs=[camera_pair],
            file_pattern=file_pattern,
            pattern_cols=pattern_cols,
            pattern_rows=pattern_rows,
            dot_spacing_mm=dot_spacing_mm,
            asymmetric=asymmetric,
            enhance_dots=enhance_dots,
        )

        # Process the camera pair
        calibrator.process_camera_pair(camera_pair[0], camera_pair[1])

        # Try to load stereo results
        stereo_dir = base_root / "calibration" / f"Cam{camera_pair[0]}" / "stereo"
        stereo_file = stereo_dir / f"stereo_cam{camera_pair[0]}_cam{camera_pair[1]}.mat"

        results = {"status": "computed"}
        if stereo_file.exists():
            try:
                stereo_data = scipy.io.loadmat(
                    stereo_file, struct_as_record=False, squeeze_me=True
                )
                # Extract matrices
                mtx1 = stereo_data.get("camera_matrix_1", np.eye(3))
                mtx2 = stereo_data.get("camera_matrix_2", np.eye(3))
                R = stereo_data.get("rotation_matrix", np.eye(3))
                T = stereo_data.get("translation_vector", np.zeros((3, 1)))
                cam1_pos, cam2_pos, angle_deg = compute_camera_positions_and_angle(R, T)
                results["stereo_model"] = {
                    "fundamental_matrix": stereo_data.get("F", np.eye(3)).tolist(),
                    "essential_matrix": stereo_data.get("E", np.eye(3)).tolist(),
                    "stereo_reprojection_error": float(
                        stereo_data.get("stereo_reprojection_error", 0)
                    ),
                    "camera_pair": camera_pair,
                    "num_images": int(
                        stereo_data.get(
                            "num_images", stereo_data.get("num_image_pairs", 0)
                        )
                    ),
                    "camera_matrix_1": mtx1.tolist(),
                    "camera_matrix_2": mtx2.tolist(),
                    "rotation_matrix": R.tolist(),
                    "translation_vector": T.tolist(),
                    "camera_positions": [cam1_pos, cam2_pos],
                    "angle_of_separation_deg": angle_deg,
                }
            except Exception as e:
                logger.warning(f"Could not load stereo results: {e}")
        return jsonify(results)

    except Exception as e:
        logger.error(f"Error computing stereo calibration: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/calibration/stereo/load_results", methods=["GET"])
def stereo_load_results():
    """Load previously computed stereo calibration results"""
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera_pair = request.args.get("camera_pair", default="1,2")

    try:
        # Parse camera pair
        cam1, cam2 = map(int, camera_pair.split(","))

        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])

        # Check for stereo results
        stereo_dir = base_root / "calibration" / f"Cam{cam1}" / "stereo"
        stereo_file = stereo_dir / f"stereo_cam{cam1}_cam{cam2}.mat"

        if not stereo_file.exists():
            return jsonify(
                {"exists": False, "message": "No saved stereo results found"}
            )

        stereo_data = scipy.io.loadmat(
            stereo_file, struct_as_record=False, squeeze_me=True
        )
        mtx1 = stereo_data.get("camera_matrix_1", np.eye(3))
        mtx2 = stereo_data.get("camera_matrix_2", np.eye(3))
        R = stereo_data.get("rotation_matrix", np.eye(3))
        T = stereo_data.get("translation_vector", np.zeros((3, 1)))
        cam1_pos, cam2_pos, angle_deg = compute_camera_positions_and_angle(R, T)
        results = {
            "fundamental_matrix": stereo_data.get("F", np.eye(3)).tolist(),
            "essential_matrix": stereo_data.get("E", np.eye(3)).tolist(),
            "stereo_reprojection_error": float(
                stereo_data.get("stereo_reprojection_error", 0)
            ),
            "camera_pair": [cam1, cam2],
            "num_images": int(
                stereo_data.get("num_images", stereo_data.get("num_image_pairs", 0))
            ),
            "timestamp": str(stereo_data.get("timestamp", "")),
            "camera_matrix_1": mtx1.tolist(),
            "camera_matrix_2": mtx2.tolist(),
            "rotation_matrix": R.tolist(),
            "translation_vector": T.tolist(),
            "camera_positions": [cam1_pos, cam2_pos],
            "angle_of_separation_deg": angle_deg,
        }
        return jsonify({"exists": True, "results": results})

    except Exception as e:
        logger.error(f"Error loading stereo calibration results: {e}")
        return jsonify({"error": str(e)}), 500


def write_status_file(
    output_dir: Path, status: str, details: dict = None, type: str = "pinhole"
):
    status_path = output_dir / f"calibration_status_{type}.json"
    payload = {"status": status, "timestamp": datetime.now().isoformat()}
    if details:
        payload.update(details)
    try:
        with open(status_path, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        logger.warning(f"Could not write status file: {e}")


@calibration_bp.route("/calibration/planar/calibrate_all", methods=["POST"])
def planar_calibrate_all():
    """Start batch calibration of all images for planar calibration - simplified"""
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

    # Get default values from config
    try:
        cfg = get_config()
        pinhole_config = cfg.calibration.get("pinhole", {})

        # Only override if still using defaults
        if file_pattern == "calib%05d.tif":
            file_pattern = pinhole_config.get("file_pattern", file_pattern)
        if pattern_cols == 10:
            pattern_cols = int(pinhole_config.get("pattern_cols", pattern_cols))
        if pattern_rows == 10:
            pattern_rows = int(pinhole_config.get("pattern_rows", pattern_rows))
        if dot_spacing_mm == 28.89:
            dot_spacing_mm = float(pinhole_config.get("dot_spacing_mm", dot_spacing_mm))
        if dt == 1.0:
            dt = float(pinhole_config.get("dt", dt))

        logger.info(
            f"Batch calibration using config values - file_pattern: {file_pattern}, pattern: {pattern_cols}x{pattern_rows}, spacing: {dot_spacing_mm}mm, dt: {dt}s"
        )
    except Exception as e:
        logger.warning(f"Could not load config defaults for batch calibration: {e}")

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])
        base_root = Path(cfg.base_paths[source_path_idx])
        cam_output_base = base_root / "calibration" / f"Cam{camera}"

        # Find all calibration images
        cam_input_dir = source_root / "calibration" / f"Cam{camera}"

        if not cam_input_dir.exists():
            return (
                jsonify({"error": f"Camera directory not found: {cam_input_dir}"}),
                404,
            )

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

        if not image_files:
            return (
                jsonify({"error": f"No images found with pattern {file_pattern}"}),
                404,
            )

        logger.info(
            f"Starting batch calibration of {len(image_files)} images for Camera {camera}"
        )

        # Write status: running
        write_status_file(
            cam_output_base, "running", {"total_images": len(image_files)}
        )

        # Start background processing - simplified without progress tracking
        def process_batch():
            try:
                # Create calibrator instance - PASS dt parameter
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
                )

                # Process all images
                calibrator.process_camera(camera)
                logger.info(
                    f"Batch calibration completed successfully for Camera {camera}"
                )
                # Write status: completed
                write_status_file(
                    cam_output_base, "completed", {"total_images": len(image_files)}
                )
            except Exception as e:
                logger.error(f"Batch calibration failed for Camera {camera}: {e}")
                write_status_file(cam_output_base, "error", {"error": str(e)})

        # Start processing thread
        thread = threading.Thread(target=process_batch)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "status": "started",
                "message": f"Batch calibration started for Camera {camera} with {len(image_files)} images. Check logs for progress.",
                "total_images": len(image_files),
            }
        )
    except Exception as e:
        logger.error(f"Error starting batch calibration: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/calibration/vectors/calibrate_all", methods=["POST"])
def vectors_calibrate_all():
    """Start batch vector calibration using existing camera models"""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    model_index = int(data.get("model_index", 0))
    dt = float(data.get("dt", 1.0))
    image_count = int(data.get("image_count", 1000))
    vector_pattern = data.get("vector_pattern", "%05d.mat")
    type_name = data.get("type_name", "instantaneous")
    runs_to_process = data.get(
        "runs_to_process", None
    )  # List of run numbers or None for all

    # Get runs from config if not specified
    if runs_to_process is None:
        try:
            cfg = get_config()
            # Try to get runs from instantaneous_piv config
            config_runs = cfg.instantaneous_runs
            if config_runs:
                runs_to_process = config_runs
                logger.info(f"Using runs from config: {runs_to_process}")
            # Also try to get image_count from config
            if image_count == 1000:  # If still default
                config_image_count = cfg.num_images
                if config_image_count:
                    image_count = config_image_count
                    logger.info(f"Using image count from config: {image_count}")
        except Exception as e:
            logger.warning(f"Could not load runs from config: {e}")

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])
        cam_output_base = base_root / "calibration" / f"Cam{camera}"

        # Write status: running
        write_status_file(
            cam_output_base,
            "running",
            {"model_index": model_index, "image_count": image_count},
        )

        def process_vectors():
            try:
                from calibration.vector_calibration_production import VectorCalibrator

                calibrator = VectorCalibrator(
                    base_dir=base_root,
                    camera_num=camera,
                    model_index=model_index,
                    dt=dt,
                    vector_pattern=vector_pattern,
                    type_name=type_name,
                )
                calibrator.process_run(image_count, runs_to_process)
                logger.info(
                    f"Vector calibration completed successfully for Camera {camera}"
                )
                # Write status: completed
                write_status_file(
                    cam_output_base,
                    "completed",
                    {"model_index": model_index, "image_count": image_count},
                )
            except Exception as e:
                logger.error(f"Vector calibration failed for Camera {camera}: {e}")
                write_status_file(cam_output_base, "error", {"error": str(e)})

        thread = threading.Thread(target=process_vectors)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "status": "started",
                "message": f"Vector calibration started for Camera {camera}. Using runs {runs_to_process} with {image_count} images. Check logs for progress.",
                "model_used": model_index,
                "runs_to_process": runs_to_process,
                "image_count": image_count,
            }
        )
    except Exception as e:
        logger.error(f"Error starting vector calibration: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/calibration/scale_factor/calibrate_vectors", methods=["POST"])
def scale_factor_calibrate_vectors():
    """Calibrate vectors and coordinates using scale factor method (pixels/mm and dt)"""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    dt = float(data.get("dt", 1.0))
    px_per_mm = float(data.get("px_per_mm", 1.0))
    image_count = int(data.get("image_count", 1000))
    x_offset = data.get("x_offset", [0])
    y_offset = data.get("y_offset", [0])
    type_name = data.get("type_name", "instantaneous")

    # Get runs from config if not specified
    try:
        cfg = get_config()
        config_runs = getattr(cfg, "instantaneous_runs", None)
        if config_runs:
            runs_to_process = config_runs
            logger.info(f"Using runs from config: {runs_to_process}")
        else:
            runs_to_process = None
        # Also try to get image_count from config
        if image_count == 1000:  # If still default
            config_image_count = getattr(cfg, "num_images", None)
            if config_image_count:
                image_count = config_image_count
                logger.info(f"Using image count from config: {image_count}")
    except Exception as e:
        logger.warning(f"Could not load config settings: {e}")
        runs_to_process = None

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])
        Path(cfg.source_paths[source_path_idx])

        # Use same path logic as pinhole calibration for input
        data_paths = get_data_paths(
            base_dir=base_root,
            num_images=image_count,
            cam=camera,
            type_name=type_name,
            use_uncalibrated=True,
        )
        uncalib_data_dir = data_paths["data_dir"]

        # IMPORTANT: Output to calibrated PIV directory (same as pinhole)
        calib_data_paths = get_data_paths(
            base_dir=base_root,
            num_images=image_count,
            cam=camera,
            type_name=type_name,
            calibration=False,
        )
        output_dir = calib_data_paths["data_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        # Status tracking in calibration directory
        cam_output_base = base_root / "calibration" / f"Cam{camera}"

        # Write status: running
        write_status_file(
            cam_output_base,
            "running",
            {
                "method": "scale_factor",
                "image_count": image_count,
                "runs_to_process": runs_to_process,
                "output_dir": str(output_dir),
            },
            type="scale_factor",
        )

        def process_scale_factor():
            try:
                import numpy as np
                import scipy.io

                from post_processing.vector_loading import (
                    load_coords_from_directory,
                    read_mat_contents,
                )

                logger.info(f"Starting scale factor calibration for Camera {camera}")
                logger.info(f"Uncalibrated data directory: {uncalib_data_dir}")
                logger.info(f"Calibrated output directory: {output_dir}")
                logger.info(f"Scale: {px_per_mm} px/mm, dt: {dt} s")
                logger.info(f"Offsets: x={x_offset}, y={y_offset}")

                # Load coordinates using same logic as pinhole vector calibration
                x_coords_list, y_coords_list = [], []
                try:
                    if (uncalib_data_dir / "coordinates.mat").exists():
                        # Load ALL runs (not filtered by runs_to_process yet)
                        x_coords_list, y_coords_list = load_coords_from_directory(
                            uncalib_data_dir, None
                        )
                        logger.info(
                            f"Loaded coordinates for {len(x_coords_list)} runs from uncalibrated data"
                        )

                        # Log which runs actually have data
                        for i, (x_coords, y_coords) in enumerate(
                            zip(x_coords_list, y_coords_list)
                        ):
                            if x_coords.size > 0 and y_coords.size > 0:
                                logger.info(
                                    f"  Run {i+1}: has coordinate data ({x_coords.shape})"
                                )
                            else:
                                logger.info(f"  Run {i+1}: empty coordinates")
                except Exception as e:
                    logger.warning(f"Could not load coordinates.mat: {e}")

                # If no coordinates, create a default grid from first vector file
                vector_pattern = getattr(cfg, "vector_format", "%05d.mat")
                first_vector_file = uncalib_data_dir / (vector_pattern % 1)

                if not x_coords_list or not y_coords_list:
                    if not first_vector_file.exists():
                        logger.error(
                            "No coordinates.mat or vector files found for scale factor calibration."
                        )
                        write_status_file(
                            cam_output_base,
                            "error",
                            {"error": "No coordinates or vector files found"},
                            type="scale_factor",
                        )
                        return

                    # Create default grid from first vector file
                    arr = read_mat_contents(str(first_vector_file))
                    H, W = arr.shape[2], arr.shape[3]
                    y_grid, x_grid = np.meshgrid(
                        np.arange(H), np.arange(W), indexing="ij"
                    )
                    x_coords_list = [x_grid]
                    y_coords_list = [y_grid]
                    logger.info(f"Created default grid coordinates: shape=({H},{W})")

                # CRITICAL: Determine the maximum run number present in the uncalibrated data
                max_run_in_data = len(x_coords_list)
                logger.info(
                    f"Maximum run number in uncalibrated data: {max_run_in_data}"
                )
                logger.info(f"Runs to process (filter): {runs_to_process}")

                # Create coordinate structure using proper numpy structured array format
                # This creates the MATLAB struct array format: coordinates(idx).x, coordinates(idx).y
                coord_dtype = np.dtype([("x", "O"), ("y", "O")])
                coordinates = np.empty(max_run_in_data, dtype=coord_dtype)

                for run_num in range(1, max_run_in_data + 1):
                    run_idx = run_num - 1  # Convert to 0-based index

                    # Check if this run has data AND if we want to process it
                    has_data = (
                        run_idx < len(x_coords_list)
                        and x_coords_list[run_idx].size > 0
                        and y_coords_list[run_idx].size > 0
                    )

                    should_process = (
                        runs_to_process is None or run_num in runs_to_process
                    )

                    if has_data and should_process:
                        x_px = np.asarray(x_coords_list[run_idx])
                        y_px = np.asarray(y_coords_list[run_idx])

                        # Get camera-specific offsets
                        cam_idx = camera - 1
                        x_off = float(
                            x_offset[cam_idx]
                            if isinstance(x_offset, list) and cam_idx < len(x_offset)
                            else (
                                x_offset[0] if isinstance(x_offset, list) else x_offset
                            )
                        )
                        y_off = float(
                            y_offset[cam_idx]
                            if isinstance(y_offset, list) and cam_idx < len(y_offset)
                            else (
                                y_offset[0] if isinstance(y_offset, list) else y_offset
                            )
                        )

                        # If no offset specified, set bottom-left corner to (0,0)
                        if x_off == 0 and y_off == 0:
                            x_off = np.min(x_px)
                            y_off = np.max(y_px)
                            logger.info(
                                f"Auto-offset for run {run_num}: x_off={x_off:.1f}, y_off={y_off:.1f}"
                            )

                        # COORDINATES: Convert to mm
                        x_mm = (x_px - x_off) / px_per_mm
                        y_mm = (y_off - y_px) / px_per_mm

                        # Set struct fields directly - this creates coordinates(run_idx).x format
                        coordinates[run_idx] = (x_mm, y_mm)

                        logger.info(
                            f"Run {run_num}: calibrated {x_px.shape} coordinates to mm"
                        )
                    else:
                        # Empty struct for runs without data or not being processed
                        coordinates[run_idx] = (np.array([]), np.array([]))

                        if has_data and not should_process:
                            logger.info(
                                f"Run {run_num}: has data but not in processing list - saved as empty"
                            )
                        elif not has_data:
                            logger.info(f"Run {run_num}: no data - saved as empty")

                # Save coordinates with proper MATLAB struct array format
                coords_output = {"coordinates": coordinates}
                coords_path = output_dir / "coordinates.mat"
                scipy.io.savemat(
                    str(coords_path), coords_output
                )  # Remove struct_as_record parameter
                logger.info(
                    f"Saved coordinates with {len(coordinates)} total runs (proper MATLAB struct format): {coords_path}"
                )

                # Process vector files - SAME STRUCTURE PRESERVATION with proper MATLAB struct format
                n_saved = 0

                # Get coordinates for vector processing (use first available run with data)
                x_coords_for_vectors = None
                y_coords_for_vectors = None
                for run_idx, (x_coords, y_coords) in enumerate(
                    zip(x_coords_list, y_coords_list)
                ):
                    if x_coords.size > 0 and y_coords.size > 0:
                        run_num = run_idx + 1
                        if runs_to_process is None or run_num in runs_to_process:
                            x_coords_for_vectors = x_coords
                            y_coords_for_vectors = y_coords
                            logger.info(
                                f"Using coordinates from run {run_num} for vector processing"
                            )
                            break

                if (
                    x_coords_for_vectors is not None
                    and y_coords_for_vectors is not None
                ):
                    # Get camera-specific offsets for vector processing
                    cam_idx = camera - 1
                    x_off = float(
                        x_offset[cam_idx]
                        if isinstance(x_offset, list) and cam_idx < len(x_offset)
                        else (x_offset[0] if isinstance(x_offset, list) else x_offset)
                    )
                    y_off = float(
                        y_offset[cam_idx]
                        if isinstance(y_offset, list) and cam_idx < len(y_offset)
                        else (y_offset[0] if isinstance(y_offset, list) else y_offset)
                    )

                    # Auto-offset if not specified
                    if x_off == 0 and y_off == 0:
                        x_off = np.min(x_coords_for_vectors)
                        y_off = np.max(y_coords_for_vectors)

                    # Calibrate vectors for each image
                    for i in range(1, image_count + 1):
                        vector_file = uncalib_data_dir / (vector_pattern % i)
                        if not vector_file.exists():
                            continue

                        try:
                            arr = read_mat_contents(str(vector_file))
                            ux_px = arr[0, 0, :, :]  # Pixels per frame
                            uy_px = arr[0, 1, :, :]  # Pixels per frame
                            b_mask = arr[0, 2, :, :]

                            # VECTORS: Convert to m/s
                            ux_ms = (
                                ux_px / px_per_mm / dt
                            ) / 1000.0  # Convert mm/s to m/s
                            uy_ms = (
                                -uy_px / px_per_mm / dt
                            ) / 1000.0  # Convert mm/s to m/s (flip Y)

                            # Create piv_result structure array with proper MATLAB struct format
                            # This creates: piv_result(idx).ux, piv_result(idx).uy, piv_result(idx).b_mask
                            piv_dtype = np.dtype(
                                [("ux", "O"), ("uy", "O"), ("b_mask", "O")]
                            )
                            piv_result = np.empty(max_run_in_data, dtype=piv_dtype)

                            for run_num in range(1, max_run_in_data + 1):
                                run_idx = run_num - 1
                                has_data = (
                                    run_idx < len(x_coords_list)
                                    and x_coords_list[run_idx].size > 0
                                    and y_coords_list[run_idx].size > 0
                                )
                                should_process = (
                                    runs_to_process is None
                                    or run_num in runs_to_process
                                )

                                if has_data and should_process:
                                    # Calibrated vector data for this run
                                    piv_result[run_idx] = (ux_ms, uy_ms, b_mask)
                                else:
                                    # Empty struct for runs without data or not being processed
                                    piv_result[run_idx] = (
                                        np.array([]),
                                        np.array([]),
                                        np.array([]),
                                    )

                            # Save with proper MATLAB struct array format
                            out_file = output_dir / (vector_pattern % i)
                            scipy.io.savemat(
                                str(out_file), {"piv_result": piv_result}
                            )  # Remove struct_as_record parameter
                            n_saved += 1

                            if n_saved % 100 == 0:
                                logger.info(f"Processed {n_saved} vector files...")

                        except Exception as e:
                            logger.error(
                                f"Error processing vector file {vector_file}: {e}"
                            )
                            continue

                logger.info(
                    f"Scale factor calibration completed successfully for Camera {camera}"
                )
                logger.info(
                    f"Saved {n_saved} calibrated vector files (m/s) in proper MATLAB struct format to {output_dir}"
                )
                logger.info(
                    f"Structure format: {max_run_in_data} total runs with proper indexing"
                )
                logger.info("MATLAB format: piv_result(run).ux, coordinates(run).x")

                # Save status: completed
                write_status_file(
                    cam_output_base,
                    "completed",
                    {
                        "method": "scale_factor",
                        "n_saved": n_saved,
                        "image_count": image_count,
                        "runs_processed": runs_to_process or "all",
                        "total_runs_in_structure": max_run_in_data,
                        "output_dir": str(output_dir),
                        "units": "vectors: m/s, coordinates: mm",
                        "format": "MATLAB struct arrays",
                    },
                    type="scale_factor",
                )

            except Exception as e:
                logger.error(f"Scale factor vector calibration failed: {e}")
                write_status_file(
                    cam_output_base, "error", {"error": str(e)}, type="scale_factor"
                )

        # Start processing thread
        thread = threading.Thread(target=process_scale_factor)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "status": "started",
                "message": f"Scale factor calibration started for Camera {camera} with {image_count} images. Preserving full run structure. Check logs for progress.",
                "runs_to_process": runs_to_process,
                "image_count": image_count,
                "output_directory": str(output_dir),
                "units": "Vectors: m/s, Coordinates: mm",
                "structure_note": "Full run indexing preserved - if uncalibrated data has runs 4,5,6 then calibrated output will have all 6 indices with runs 1,2,3 as empty and 4,5,6 with calibrated data",
            }
        )

    except Exception as e:
        logger.error(f"Error starting scale factor calibration: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/calibration/status", methods=["GET"])
def calibration_status():
    """
    Returns the status of calibration for a given camera.
    Query params: source_path_idx, camera, type
    """
    source_path_idx = int(request.args.get("source_path_idx", 0))
    camera = camera_number(request.args.get("camera", 1))
    cal_type = request.args.get("type", "pinhole")
    cfg = get_config()
    base_root = Path(cfg.base_paths[source_path_idx])
    cam_output_base = base_root / "calibration" / f"Cam{camera}"
    status_path = cam_output_base / f"calibration_status_{cal_type}.json"
    if status_path.exists():
        with open(status_path, "r") as f:
            status = json.load(f)
        return jsonify(status)
    else:
        return jsonify({"status": "not_started"})
