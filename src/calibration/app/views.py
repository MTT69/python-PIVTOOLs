import glob
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
