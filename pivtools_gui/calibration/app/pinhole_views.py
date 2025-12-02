"""
Pinhole (Planar) Calibration Views.

Provides Flask endpoints for planar calibration and vector calibration
using pinhole camera model.
"""

import base64
import glob
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_core.image_handling.load_images import read_image

from ..calibration_planar.planar_calibration_production import PlanarCalibrator
from ..vector_calibration_production import VectorCalibrator
from ..services.job_manager import job_manager
from ...utils import camera_number, numpy_to_png_base64

pinhole_bp = Blueprint("pinhole", __name__)


# ============================================================================
# VECTOR CALIBRATION ROUTES (using pinhole camera model)
# ============================================================================


@pinhole_bp.route("/calibration/vectors/calibrate_all", methods=["POST"])
def vectors_calibrate_all():
    """
    Start vector calibration job using production methods.

    Request JSON:
        source_path_idx: int
        camera: int
        model_index: int - Index of camera model to use
        dt: float - Time between frames
        image_count: int
        vector_pattern: str - Pattern for vector files (default: "%05d.mat")
        type_name: str - Type of data (default: "instantaneous")

    Returns:
        JSON with job_id, status, model_used, image_count
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    model_index = int(data.get("model_index", 0))
    dt = float(data.get("dt", 1.0))
    image_count = int(data.get("image_count", 1000))
    vector_pattern = data.get("vector_pattern", "%05d.mat")
    type_name = data.get("type_name", "instantaneous")

    # Create job
    job_id = job_manager.create_job(
        "vector",
        processed_frames=0,
        successful_frames=0,
        total_frames=image_count,
    )

    def run_vector_calibration():
        try:
            job_manager.update_job(job_id, status="running")

            cfg = get_config()
            base_root = Path(cfg.base_paths[source_path_idx])

            def progress_callback(progress_data):
                job_manager.update_job(
                    job_id,
                    progress=progress_data.get("progress", 0),
                    processed_frames=progress_data.get("processed_frames", 0),
                    successful_frames=progress_data.get("successful_frames", 0),
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

            job_manager.complete_job(job_id)

        except Exception as e:
            logger.error(f"Vector calibration job {job_id} failed: {e}")
            job_manager.fail_job(job_id, str(e))

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


@pinhole_bp.route("/calibration/vectors/status/<job_id>", methods=["GET"])
def vectors_status(job_id):
    """Get vector calibration job status."""
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job_data)


# ============================================================================
# PLANAR CALIBRATION ROUTES
# ============================================================================


def _find_calibration_images(cam_input_dir: Path, file_pattern: str) -> list:
    """
    Find calibration images matching the given pattern.

    Args:
        cam_input_dir: Directory containing calibration images
        file_pattern: Pattern for filenames (supports %d or glob patterns)

    Returns:
        List of absolute file paths
    """
    if "%" in file_pattern:
        # Handle numbered patterns like calib%05d.tif
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

    return image_files


@pinhole_bp.route("/calibration/planar/get_image", methods=["GET"])
def planar_get_image():
    """Get calibration image for production planar calibration."""
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))
    image_index = request.args.get("image_index", default=0, type=int)
    file_pattern = request.args.get("file_pattern", default="calib%05d.tif")

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])
        cam_input_dir = source_root / "calibration" / f"Cam{camera}"

        logger.info(f"Looking for images in: {cam_input_dir}")

        if not cam_input_dir.exists():
            return (
                jsonify({"error": f"Camera directory not found: {cam_input_dir}"}),
                404,
            )

        image_files = _find_calibration_images(cam_input_dir, file_pattern)

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


@pinhole_bp.route("/calibration/planar/validate_images", methods=["POST"])
def planar_validate_images():
    """
    Validate calibration images exist and are readable.

    Supports both standard image formats and container formats (.set, .im7).
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    file_pattern = data.get("file_pattern", "calib%05d.tif")

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])

        # Detect container format
        is_container = (
            ".set" in file_pattern.lower() or ".im7" in file_pattern.lower()
        )

        # Setup paths based on format
        if is_container:
            cam_input_dir = source_root / "calibration"
            container_file = cam_input_dir / file_pattern
        else:
            cam_input_dir = source_root / "calibration" / f"Cam{camera}"

        if not cam_input_dir.exists():
            return jsonify(
                {
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
                    "error": f"Calibration directory not found: {cam_input_dir}",
                }
            )

        # Find calibration images
        image_files = []
        sample_files = []

        if is_container:
            if container_file.exists():
                image_files = [str(container_file)]
                sample_files = [container_file.name]
            else:
                return jsonify(
                    {
                        "valid": False,
                        "checked": True,
                        "found_count": 0,
                        "file_pattern": file_pattern,
                        "camera_path": str(cam_input_dir),
                        "sample_files": [],
                        "first_image_preview": None,
                        "image_size": None,
                        "format_detected": Path(file_pattern).suffix.lstrip("."),
                        "container_format": True,
                        "error": f"Container file not found: {container_file}",
                    }
                )
        else:
            image_files = _find_calibration_images(cam_input_dir, file_pattern)
            sample_files = [Path(f).name for f in image_files[:5]]

        if not image_files:
            # Try to suggest a pattern based on files that exist
            suggested_pattern = None
            found_files = []
            error_msg = f"No images found matching pattern: {file_pattern}"

            if cam_input_dir.exists():
                all_files = []
                for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                    all_files.extend([f.name for f in cam_input_dir.glob(ext)])
                    all_files.extend([f.name for f in cam_input_dir.glob(ext.upper())])

                if all_files:
                    all_files = sorted(set(all_files))[:5]
                    found_files = all_files

                    import re

                    sample = all_files[0]
                    digit_match = re.search(r"(\d+)", sample)
                    if digit_match:
                        num_digits = len(digit_match.group(1))
                        suggested_pattern = re.sub(
                            r"\d+", f"%0{num_digits}d", sample, count=1
                        )

                    if suggested_pattern and suggested_pattern != file_pattern:
                        error_msg += f". Found files like: {', '.join(all_files[:3])}"
                        if len(all_files) > 3:
                            error_msg += " (+more)"
                        error_msg += f". Try pattern: {suggested_pattern}"

            return jsonify(
                {
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
                    "error": error_msg,
                }
            )

        # Try to read the first image for preview
        preview_b64 = None
        image_size = None
        format_detected = None

        try:
            if is_container:
                if ".set" in file_pattern.lower():
                    img = read_image(str(container_file), camera_no=camera, im_no=1)
                else:
                    img = read_image(str(container_file), camera_no=camera)
                format_detected = Path(file_pattern).suffix.lstrip(".")
            else:
                img = read_image(image_files[0])
                format_detected = Path(image_files[0]).suffix.lstrip(".")

            if img is not None:
                if img.ndim == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img.copy()

                image_size = [int(gray.shape[1]), int(gray.shape[0])]

                disp = gray.astype(float) - gray.min()
                if disp.max() > 0:
                    disp = disp / disp.max()
                disp8 = (disp * 255).astype(np.uint8)

                preview_b64 = numpy_to_png_base64(disp8)

        except Exception as e:
            logger.warning(f"Could not read first image for preview: {e}")
            return jsonify(
                {
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
                    "error": f"Images found but could not be read: {str(e)}",
                }
            )

        return jsonify(
            {
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
                "error": None,
            }
        )

    except Exception as e:
        logger.error(f"Error validating planar calibration images: {e}")
        return jsonify(
            {
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
                "error": str(e),
            }
        ), 500


@pinhole_bp.route("/calibration/planar/detect_grid", methods=["POST"])
def planar_detect_grid():
    """Detect grid in calibration image using production methods."""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    image_index = int(data.get("image_index", 0))
    file_pattern = data.get("file_pattern", "calib%05d.tif")
    pattern_cols = int(data.get("pattern_cols", 10))
    pattern_rows = int(data.get("pattern_rows", 10))
    enhance_dots = bool(data.get("enhance_dots", True))
    asymmetric = bool(data.get("asymmetric", False))

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])
        base_root = Path(cfg.base_paths[source_path_idx])

        calibrator = PlanarCalibrator(
            source_dir=source_root,
            base_dir=base_root,
            camera_count=1,
            file_pattern=file_pattern,
            pattern_cols=pattern_cols,
            pattern_rows=pattern_rows,
            asymmetric=asymmetric,
            enhance_dots=enhance_dots,
        )

        cam_input_dir = source_root / "calibration" / f"Cam{camera}"
        image_files = _find_calibration_images(cam_input_dir, file_pattern)

        if image_index >= len(image_files):
            return jsonify({"error": "Image index out of range"}), 404

        img_path = image_files[image_index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        found, grid_points = calibrator.detect_grid_in_image(img)

        if not found:
            return jsonify({"error": "Grid not detected", "found": False})

        return jsonify(
            {
                "found": True,
                "grid_points": grid_points.tolist(),
                "count": len(grid_points.tolist()),
                "pattern_size": [pattern_cols, pattern_rows],
            }
        )

    except Exception as e:
        logger.error(f"Error detecting grid: {e}")
        return jsonify({"error": str(e)}), 500


@pinhole_bp.route("/calibration/planar/compute", methods=["POST"])
def planar_compute():
    """Compute full planar calibration using production methods."""
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

        cam_input_dir = source_root / "calibration" / f"Cam{camera}"
        image_files = _find_calibration_images(cam_input_dir, file_pattern)

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
            dt=dt,
            selected_image_idx=image_index + 1,
        )

        # Run calibration for this camera and image
        calibrator.process_camera(camera)

        # Load results
        indices_folder = cam_output_base / "indices"
        model_folder = cam_output_base / "model"
        dewarp_folder = cam_output_base / "dewarp"
        grid_file = indices_folder / f"indexing_{image_index+1}.mat"
        model_file = model_folder / "camera_model.mat"
        grid_png_file = indices_folder / f"indexes_{image_index+1}.png"
        dewarped_file = dewarp_folder / f"dewarped_{image_index+1}.tif"
        results = {}

        # Load grid data
        if grid_file.exists():
            grid_data = scipy.io.loadmat(
                grid_file, struct_as_record=False, squeeze_me=True
            )
            grid_points = grid_data["grid_points"]
            pattern_size = grid_data["pattern_size"]
            dot_spacing = float(grid_data["dot_spacing_mm"])
            cols, rows = pattern_size

            px_per_mm = None
            if grid_points.shape[0] >= 2:
                first_row = grid_points[:cols]
                x_vals = first_row[:, 0]
                px_per_mm = (
                    (x_vals.max() - x_vals.min()) / (cols - 1) / dot_spacing
                    if dot_spacing > 0
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
                "dot_spacing_mm": dot_spacing,
                "pixels_per_mm": px_per_mm,
                "timestamp": str(grid_data.get("timestamp", "")),
                "original_filename": str(grid_data.get("original_filename", "")),
            }
            results["grid_data"] = grid_data_dict

        # Load grid PNG visualization
        if grid_png_file.exists():
            try:
                with open(grid_png_file, "rb") as f:
                    grid_png_b64 = base64.b64encode(f.read()).decode("utf-8")
                if "grid_data" in results:
                    results["grid_data"]["grid_png"] = grid_png_b64
                else:
                    results["grid_png"] = grid_png_b64
            except Exception as e:
                logger.error(f"Error loading grid PNG: {e}")

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

        # Load dewarped image
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


@pinhole_bp.route("/calibration/planar/load_results", methods=["GET"])
def planar_load_results():
    """Load previously computed planar calibration results."""
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))
    image_index = request.args.get("image_index", default=0, type=int)

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])
        cam_output_base = base_root / "calibration" / f"Cam{camera}"

        grid_file = cam_output_base / "grid" / f"indexing_{image_index}.mat"
        model_file = cam_output_base / "models" / f"{image_index}.mat"

        if not grid_file.exists() or not model_file.exists():
            return jsonify({"exists": False, "message": "No saved results found"})

        results = {}

        # Load grid data
        grid_data = scipy.io.loadmat(grid_file, struct_as_record=False, squeeze_me=True)
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

        # Try to load grid PNG
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


def _process_single_image(
    idx,
    source_root,
    base_root,
    file_pattern,
    pattern_cols,
    pattern_rows,
    dot_spacing_mm,
    asymmetric,
    enhance_dots,
    dt,
    camera,
):
    """Helper function for parallel processing of single calibration images."""
    try:
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
            selected_image_idx=idx + 1,
        )
        calibrator.process_camera(camera)
        return idx, True
    except Exception as e:
        logger.error(f"Error processing image {idx}: {e}")
        return idx, False


@pinhole_bp.route("/calibration/planar/calibrate_all", methods=["POST"])
def planar_calibrate_all():
    """Start batch planar calibration job for all images for a camera."""
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

    job_id = job_manager.create_job(
        "planar",
        processed_indices=[],
        total_images=0,
    )

    def run_planar_calibration():
        try:
            cfg = get_config()
            source_root = Path(cfg.source_paths[source_path_idx])
            base_root = Path(cfg.base_paths[source_path_idx])
            cam_input_dir = source_root / "calibration" / f"Cam{camera}"

            image_files = _find_calibration_images(cam_input_dir, file_pattern)
            total_images = len(image_files)

            job_manager.update_job(job_id, total_images=total_images, status="running")

            if total_images == 0:
                job_manager.fail_job(job_id, "No calibration images found")
                return

            # Process all images in parallel
            max_workers = min(os.cpu_count() or 4, total_images, 8)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _process_single_image,
                        idx,
                        source_root,
                        base_root,
                        file_pattern,
                        pattern_cols,
                        pattern_rows,
                        dot_spacing_mm,
                        asymmetric,
                        enhance_dots,
                        dt,
                        camera,
                    )
                    for idx in range(total_images)
                ]

                processed_indices = []
                for future in as_completed(futures):
                    idx, success = future.result()
                    processed_indices.append(idx)
                    progress = int((len(processed_indices) / total_images) * 100)
                    job_manager.update_job(
                        job_id,
                        processed_indices=processed_indices,
                        progress=progress,
                    )

            job_manager.complete_job(job_id)

        except Exception as e:
            logger.error(f"Planar calibration job {job_id} failed: {e}")
            job_manager.fail_job(job_id, str(e))

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


@pinhole_bp.route(
    "/calibration/planar/calibrate_all/status/<job_id>", methods=["GET"]
)
def planar_calibrate_all_status(job_id):
    """Get batch planar calibration job status."""
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job_data)
