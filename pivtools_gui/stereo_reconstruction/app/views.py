#!/usr/bin/env python3
"""
stereo_reconstruction/app/views.py

Production-ready Flask endpoints for stereo camera calibration and 3D vector reconstruction.

This module provides comprehensive web API endpoints for:

STEREO CALIBRATION:
- Get available calibration images for camera pairs
- Run stereo calibration with real-time progress tracking
- Load and display calibration results with key metrics
- Get grid detection visualization images for quality assessment
- Validate calibration parameters before processing
- Preview grid detection on sample images
"""

import base64
import glob
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger
from scipy.io.matlab.mio5_params import mat_struct

from ...config import get_config

# Import production calibration and reconstruction classes
from ..stereo_calibration_production import StereoCalibrator
from ..stereo_reconstruction_production import StereoReconstructor
from ...utils import camera_number, numpy_to_png_base64

stereo_bp = Blueprint("stereo", __name__)

# Global job tracking for async operations
stereo_jobs = {}
job_id_counter = 0


def generate_job_id():
    global job_id_counter
    job_id_counter += 1
    return f"stereo_job_{job_id_counter}_{int(time.time())}"


def _to_dict(obj):
    """Recursively convert mat_struct to dict"""
    if isinstance(obj, mat_struct):
        result = {}
        for field in obj._fieldnames:
            value = getattr(obj, field)
            result[field] = _to_dict(value)
        return result
    elif isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    else:
        return obj


# ============================================================================
# STEREO CALIBRATION ROUTES
# ============================================================================


@stereo_bp.route("/stereo/calibration/get_images", methods=["GET"])
def stereo_get_calibration_images():
    """Get available calibration images for a camera pair"""
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    cam1 = camera_number(request.args.get("cam1", default=1, type=int))
    cam2 = camera_number(request.args.get("cam2", default=2, type=int))
    file_pattern = request.args.get(
        "file_pattern", default="planar_calibration_plate_*.tif"
    )

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])

        # Get images for both cameras
        cam1_dir = source_root / "calibration" / f"Cam{cam1}"
        cam2_dir = source_root / "calibration" / f"Cam{cam2}"

        if not cam1_dir.exists() or not cam2_dir.exists():
            return (
                jsonify(
                    {"error": f"Camera directories not found: {cam1_dir} or {cam2_dir}"}
                ),
                404,
            )

        def get_image_files(cam_dir, pattern):
            if "%" in pattern:
                files = []
                i = 1
                while True:
                    filename = pattern % i
                    filepath = cam_dir / filename
                    if filepath.exists():
                        files.append(str(filepath))
                        i += 1
                    else:
                        break
            else:
                files = sorted(glob.glob(str(cam_dir / pattern)))
            return files

        cam1_files = get_image_files(cam1_dir, file_pattern)
        cam2_files = get_image_files(cam2_dir, file_pattern)

        # Find matching files
        cam1_dict = {Path(f).name: f for f in cam1_files}
        cam2_dict = {Path(f).name: f for f in cam2_files}
        common_names = sorted(set(cam1_dict.keys()) & set(cam2_dict.keys()))

        matching_pairs = []
        for name in common_names:
            matching_pairs.append(
                {
                    "filename": name,
                    "cam1_path": cam1_dict[name],
                    "cam2_path": cam2_dict[name],
                }
            )

        return jsonify(
            {
                "camera_pair": [cam1, cam2],
                "total_pairs": len(matching_pairs),
                "matching_files": matching_pairs,
                "cam1_total": len(cam1_files),
                "cam2_total": len(cam2_files),
                "file_pattern": file_pattern,
            }
        )

    except Exception as e:
        logger.error(f"Error getting stereo calibration images: {e}")
        return jsonify({"error": str(e)}), 500


@stereo_bp.route("/stereo/calibration/run", methods=["POST"])
def stereo_run_calibration():
    """Run stereo calibration with progress tracking"""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera_pairs = data.get("camera_pairs", [[1, 2]])
    file_pattern = data.get("file_pattern", "planar_calibration_plate_*.tif")
    pattern_cols = int(data.get("pattern_cols", 10))
    pattern_rows = int(data.get("pattern_rows", 10))
    dot_spacing_mm = float(data.get("dot_spacing_mm", 28.89))
    asymmetric = bool(data.get("asymmetric", False))
    enhance_dots = bool(data.get("enhance_dots", True))

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])
        base_root = Path(cfg.base_paths[source_path_idx])

        job_id = generate_job_id()

        # Initialize job status
        stereo_jobs[job_id] = {
            "status": "starting",
            "progress": 0,
            "total_pairs": len(camera_pairs),
            "current_pair": None,
            "processed_pairs": 0,
            "results": {},
            "error": None,
            "start_time": datetime.now().isoformat(),
            "camera_pairs": camera_pairs,
        }

        def run_calibration():
            try:
                stereo_jobs[job_id]["status"] = "running"

                # Progress callback
                def progress_callback(pair_idx, pair_total, current_status):
                    if job_id in stereo_jobs:
                        progress = int((pair_idx / pair_total) * 100)
                        stereo_jobs[job_id]["progress"] = progress
                        stereo_jobs[job_id]["current_pair"] = current_status
                        stereo_jobs[job_id][
                            "current_stage"
                        ] = f"processing_pair_{pair_idx + 1}"
                        logger.info(
                            f"Job {job_id} progress: {progress}% - {current_status}"
                        )

                # Create calibrator
                calibrator = StereoCalibrator(
                    source_dir=source_root,
                    base_dir=base_root,
                    camera_pairs=camera_pairs,
                    file_pattern=file_pattern,
                    pattern_cols=pattern_cols,
                    pattern_rows=pattern_rows,
                    dot_spacing_mm=dot_spacing_mm,
                    asymmetric=asymmetric,
                    enhance_dots=enhance_dots,
                )

                # Process each camera pair
                for i, (cam1, cam2) in enumerate(camera_pairs):
                    if job_id not in stereo_jobs:
                        break  # Job was cancelled

                    progress_callback(i, len(camera_pairs), f"Camera {cam1}-{cam2}")

                    try:
                        calibrator.process_camera_pair(cam1, cam2)
                        stereo_jobs[job_id]["processed_pairs"] += 1

                        # Load results for this pair
                        stereo_file = (
                            base_root
                            / "calibration"
                            / f"stereo_model_cam{cam1}-cam{cam2}.mat"
                        )
                        if stereo_file.exists():
                            stereo_data = scipy.io.loadmat(
                                str(stereo_file),
                                squeeze_me=True,
                                struct_as_record=False,
                            )

                            # Extract key metrics
                            results = {
                                "camera_pair": [cam1, cam2],
                                "stereo_reprojection_error": float(
                                    stereo_data.get("stereo_reprojection_error", 0)
                                ),
                                "relative_angle_deg": float(
                                    stereo_data.get("relative_angle_deg", 0)
                                ),
                                "num_image_pairs": int(
                                    stereo_data.get("num_image_pairs", 0)
                                ),
                                "translation_vector": (
                                    stereo_data.get("translation_vector", []).tolist()
                                    if hasattr(
                                        stereo_data.get("translation_vector", []),
                                        "tolist",
                                    )
                                    else []
                                ),
                                "camera_matrix_1": (
                                    stereo_data.get("camera_matrix_1", []).tolist()
                                    if hasattr(
                                        stereo_data.get("camera_matrix_1", []), "tolist"
                                    )
                                    else []
                                ),
                                "camera_matrix_2": (
                                    stereo_data.get("camera_matrix_2", []).tolist()
                                    if hasattr(
                                        stereo_data.get("camera_matrix_2", []), "tolist"
                                    )
                                    else []
                                ),
                                "dist_coeffs_1": (
                                    stereo_data.get("dist_coeffs_1", []).tolist()
                                    if hasattr(
                                        stereo_data.get("dist_coeffs_1", []), "tolist"
                                    )
                                    else []
                                ),
                                "dist_coeffs_2": (
                                    stereo_data.get("dist_coeffs_2", []).tolist()
                                    if hasattr(
                                        stereo_data.get("dist_coeffs_2", []), "tolist"
                                    )
                                    else []
                                ),
                                "focal_length_1": (
                                    [
                                        float(stereo_data["camera_matrix_1"][0, 0]),
                                        float(stereo_data["camera_matrix_1"][1, 1]),
                                    ]
                                    if "camera_matrix_1" in stereo_data
                                    else []
                                ),
                                "focal_length_2": (
                                    [
                                        float(stereo_data["camera_matrix_2"][0, 0]),
                                        float(stereo_data["camera_matrix_2"][1, 1]),
                                    ]
                                    if "camera_matrix_2" in stereo_data
                                    else []
                                ),
                                "principal_point_1": (
                                    [
                                        float(stereo_data["camera_matrix_1"][0, 2]),
                                        float(stereo_data["camera_matrix_1"][1, 2]),
                                    ]
                                    if "camera_matrix_1" in stereo_data
                                    else []
                                ),
                                "principal_point_2": (
                                    [
                                        float(stereo_data["camera_matrix_2"][0, 2]),
                                        float(stereo_data["camera_matrix_2"][1, 2]),
                                    ]
                                    if "camera_matrix_2" in stereo_data
                                    else []
                                ),
                                "timestamp": str(stereo_data.get("timestamp", "")),
                                "successful_filenames": (
                                    stereo_data.get("successful_filenames", []).tolist()
                                    if hasattr(
                                        stereo_data.get("successful_filenames", []),
                                        "tolist",
                                    )
                                    else []
                                ),
                            }

                            stereo_jobs[job_id]["results"][
                                f"cam{cam1}_cam{cam2}"
                            ] = results

                    except Exception as e:
                        logger.error(f"Failed to calibrate pair {cam1}-{cam2}: {e}")
                        stereo_jobs[job_id]["results"][f"cam{cam1}_cam{cam2}"] = {
                            "error": str(e)
                        }

                # Complete
                stereo_jobs[job_id]["status"] = "completed"
                stereo_jobs[job_id]["progress"] = 100
                stereo_jobs[job_id]["end_time"] = datetime.now().isoformat()

            except Exception as e:
                logger.error(f"Stereo calibration job {job_id} failed: {e}")
                stereo_jobs[job_id]["status"] = "failed"
                stereo_jobs[job_id]["error"] = str(e)

        # Start calibration in background thread
        thread = threading.Thread(target=run_calibration)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "job_id": job_id,
                "status": "started",
                "message": "Stereo calibration started",
            }
        )

    except Exception as e:
        logger.error(f"Error starting stereo calibration: {e}")
        return jsonify({"error": str(e)}), 500


@stereo_bp.route("/stereo/calibration/status/<job_id>", methods=["GET"])
def stereo_calibration_status(job_id):
    """Get status of running stereo calibration job"""
    if not job_id or job_id == "undefined":
        return jsonify({"error": "Invalid job ID"}), 400

    if job_id not in stereo_jobs:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(stereo_jobs[job_id])


@stereo_bp.route("/stereo/calibration/load_results", methods=["GET"])
def stereo_load_calibration_results():
    """Load previously computed stereo calibration results"""
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    cam1 = camera_number(request.args.get("cam1", default=1, type=int))
    cam2 = camera_number(request.args.get("cam2", default=2, type=int))

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])

        # Load stereo calibration results
        stereo_file = (
            base_root / "calibration" / f"stereo_model_cam{cam1}-cam{cam2}.mat"
        )

        if not stereo_file.exists():
            return jsonify({"exists": False, "message": "No stereo calibration found"})

        stereo_data = scipy.io.loadmat(
            str(stereo_file), squeeze_me=True, struct_as_record=False
        )

        results = {
            "camera_pair": [cam1, cam2],
            "calibration_quality": {
                "stereo_reprojection_error": float(
                    stereo_data.get("stereo_reprojection_error", 0)
                ),
                "relative_angle_deg": float(stereo_data.get("relative_angle_deg", 0)),
                "num_image_pairs": int(stereo_data.get("num_image_pairs", 0)),
                "baseline_distance": float(
                    np.linalg.norm(stereo_data.get("translation_vector", [0, 0, 0]))
                ),
            },
            "camera_intrinsics": {
                "camera_matrix_1": (
                    stereo_data.get("camera_matrix_1", []).tolist()
                    if hasattr(stereo_data.get("camera_matrix_1", []), "tolist")
                    else []
                ),
                "camera_matrix_2": (
                    stereo_data.get("camera_matrix_2", []).tolist()
                    if hasattr(stereo_data.get("camera_matrix_2", []), "tolist")
                    else []
                ),
                "dist_coeffs_1": (
                    stereo_data.get("dist_coeffs_1", []).tolist()
                    if hasattr(stereo_data.get("dist_coeffs_1", []), "tolist")
                    else []
                ),
                "dist_coeffs_2": (
                    stereo_data.get("dist_coeffs_2", []).tolist()
                    if hasattr(stereo_data.get("dist_coeffs_2", []), "tolist")
                    else []
                ),
                "focal_length_1": (
                    [
                        float(stereo_data["camera_matrix_1"][0, 0]),
                        float(stereo_data["camera_matrix_1"][1, 1]),
                    ]
                    if "camera_matrix_1" in stereo_data
                    else []
                ),
                "focal_length_2": (
                    [
                        float(stereo_data["camera_matrix_2"][0, 0]),
                        float(stereo_data["camera_matrix_2"][1, 1]),
                    ]
                    if "camera_matrix_2" in stereo_data
                    else []
                ),
                "principal_point_1": (
                    [
                        float(stereo_data["camera_matrix_1"][0, 2]),
                        float(stereo_data["camera_matrix_1"][1, 2]),
                    ]
                    if "camera_matrix_1" in stereo_data
                    else []
                ),
                "principal_point_2": (
                    [
                        float(stereo_data["camera_matrix_2"][0, 2]),
                        float(stereo_data["camera_matrix_2"][1, 2]),
                    ]
                    if "camera_matrix_2" in stereo_data
                    else []
                ),
            },
            "stereo_geometry": {
                "translation_vector": (
                    stereo_data.get("translation_vector", []).tolist()
                    if hasattr(stereo_data.get("translation_vector", []), "tolist")
                    else []
                ),
                "rotation_matrix": (
                    stereo_data.get("rotation_matrix", []).tolist()
                    if hasattr(stereo_data.get("rotation_matrix", []), "tolist")
                    else []
                ),
                "fundamental_matrix": (
                    stereo_data.get("fundamental_matrix", []).tolist()
                    if hasattr(stereo_data.get("fundamental_matrix", []), "tolist")
                    else []
                ),
                "essential_matrix": (
                    stereo_data.get("essential_matrix", []).tolist()
                    if hasattr(stereo_data.get("essential_matrix", []), "tolist")
                    else []
                ),
            },
            "rectification": {
                "rectification_R1": (
                    stereo_data.get("rectification_R1", []).tolist()
                    if hasattr(stereo_data.get("rectification_R1", []), "tolist")
                    else []
                ),
                "rectification_R2": (
                    stereo_data.get("rectification_R2", []).tolist()
                    if hasattr(stereo_data.get("rectification_R2", []), "tolist")
                    else []
                ),
                "projection_P1": (
                    stereo_data.get("projection_P1", []).tolist()
                    if hasattr(stereo_data.get("projection_P1", []), "tolist")
                    else []
                ),
                "projection_P2": (
                    stereo_data.get("projection_P2", []).tolist()
                    if hasattr(stereo_data.get("projection_P2", []), "tolist")
                    else []
                ),
                "disparity_to_depth_Q": (
                    stereo_data.get("disparity_to_depth_Q", []).tolist()
                    if hasattr(stereo_data.get("disparity_to_depth_Q", []), "tolist")
                    else []
                ),
            },
            "metadata": {
                "timestamp": str(stereo_data.get("timestamp", "")),
                "successful_filenames": (
                    stereo_data.get("successful_filenames", []).tolist()
                    if hasattr(stereo_data.get("successful_filenames", []), "tolist")
                    else []
                ),
                "image_size": (
                    stereo_data.get("image_size", []).tolist()
                    if hasattr(stereo_data.get("image_size", []), "tolist")
                    else []
                ),
            },
        }

        # Quality assessment
        reprojection_error = results["calibration_quality"]["stereo_reprojection_error"]
        if reprojection_error > 1.0:
            results["quality_warning"] = (
                f"High reprojection error: {reprojection_error:.3f} pixels"
            )
        elif reprojection_error > 0.5:
            results["quality_warning"] = (
                f"Moderate reprojection error: {reprojection_error:.3f} pixels"
            )
        else:
            results["quality_status"] = "Good calibration quality"

        return jsonify({"exists": True, "results": results})

    except Exception as e:
        logger.error(f"Error loading stereo calibration results: {e}")
        return jsonify({"error": str(e)}), 500


@stereo_bp.route("/stereo/calibration/get_grid_images", methods=["GET"])
def stereo_get_grid_images():
    """Get grid detection visualization images for a camera pair"""
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    cam1 = camera_number(request.args.get("cam1", default=1, type=int))
    cam2 = camera_number(request.args.get("cam2", default=2, type=int))
    image_index = request.args.get("image_index", default=1, type=int)  # 1-based

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])

        results = {}

        # Load grid images for both cameras
        for cam_num in [cam1, cam2]:
            cam_stereo_dir = base_root / "calibration" / f"Cam{cam_num}" / "stereo"
            grid_png_file = cam_stereo_dir / f"grid_detection_{image_index}.png"
            grid_mat_file = cam_stereo_dir / f"grid_detection_{image_index}.mat"

            cam_data = {"camera": cam_num}

            # Load PNG visualization if available
            if grid_png_file.exists():
                try:
                    with open(grid_png_file, "rb") as f:
                        grid_png_b64 = base64.b64encode(f.read()).decode("utf-8")
                    cam_data["grid_image"] = grid_png_b64
                except Exception as e:
                    logger.error(f"Error loading grid PNG {grid_png_file}: {e}")

            # Load grid data
            if grid_mat_file.exists():
                try:
                    grid_data = scipy.io.loadmat(
                        str(grid_mat_file), squeeze_me=True, struct_as_record=False
                    )
                    cam_data.update(
                        {
                            "grid_points": (
                                grid_data.get("grid_points", []).tolist()
                                if hasattr(grid_data.get("grid_points", []), "tolist")
                                else []
                            ),
                            "reprojection_error": float(
                                grid_data.get("reprojection_error", 0)
                            ),
                            "reprojection_error_x_mean": float(
                                grid_data.get("reprojection_error_x_mean", 0)
                            ),
                            "reprojection_error_y_mean": float(
                                grid_data.get("reprojection_error_y_mean", 0)
                            ),
                            "original_filename": str(
                                grid_data.get("original_filename", "")
                            ),
                            "pattern_size": (
                                grid_data.get("pattern_size", []).tolist()
                                if hasattr(grid_data.get("pattern_size", []), "tolist")
                                else []
                            ),
                            "dot_spacing_mm": float(grid_data.get("dot_spacing_mm", 0)),
                            "timestamp": str(grid_data.get("timestamp", "")),
                        }
                    )
                except Exception as e:
                    logger.error(f"Error loading grid data {grid_mat_file}: {e}")

            results[f"cam{cam_num}"] = cam_data

        # Count available grid images
        cam1_stereo_dir = base_root / "calibration" / f"Cam{cam1}" / "stereo"
        available_indices = []
        for i in range(1, 100):  # Check first 100 indices
            if (cam1_stereo_dir / f"grid_detection_{i}.png").exists():
                available_indices.append(i)

        return jsonify(
            {
                "camera_pair": [cam1, cam2],
                "image_index": image_index,
                "results": results,
                "available_indices": available_indices,
                "total_available": len(available_indices),
            }
        )

    except Exception as e:
        logger.error(f"Error getting stereo grid images: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# STEREO VECTOR CALIBRATION ROUTES
# ============================================================================


@stereo_bp.route("/stereo/vectors/run", methods=["POST"])
def stereo_run_vector_calibration():
    """Run stereo vector calibration (3D reconstruction) with progress tracking"""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera_pairs = data.get("camera_pairs", [[1, 2]])
    image_count = int(data.get("image_count", 1000))
    vector_pattern = data.get("vector_pattern", "%05d.mat")
    type_name = data.get("type_name", "instantaneous")
    max_distance = float(data.get("max_correspondence_distance", 5.0))
    min_angle = float(data.get("min_triangulation_angle", 5.0))
    dt = float(data.get("dt", 1.0))  # NEW: time between frames in seconds

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])

        job_id = generate_job_id()

        # Initialize job status
        stereo_jobs[job_id] = {
            "status": "starting",
            "progress": 0,
            "total_frames": image_count * len(camera_pairs),
            "processed_frames": 0,
            "current_pair": None,
            "current_frame": 0,
            "results": {},
            "error": None,
            "start_time": datetime.now().isoformat(),
            "camera_pairs": camera_pairs,
            "type": "vector_calibration",
        }

        def run_vector_calibration():
            try:
                stereo_jobs[job_id]["status"] = "running"

                # Progress callback
                def progress_callback(update_data):
                    if job_id in stereo_jobs:
                        pair = update_data.get("camera_pair", [])
                        processed = update_data.get("processed", 0)
                        successful = update_data.get("successful", 0)
                        total = update_data.get("total", 1)

                        stereo_jobs[job_id]["current_pair"] = (
                            f"Camera {pair[0]}-{pair[1]}" if pair else "Unknown"
                        )
                        stereo_jobs[job_id]["current_frame"] = processed
                        stereo_jobs[job_id]["processed_frames"] = successful
                        stereo_jobs[job_id]["progress"] = int((processed / total) * 100)

                # Create reconstructor
                reconstructor = StereoReconstructor(
                    base_dir=base_root,
                    camera_pairs=camera_pairs,
                    image_count=image_count,
                    vector_pattern=vector_pattern,
                    type_name=type_name,
                    max_distance=max_distance,
                    min_angle=min_angle,
                    progress_cb=progress_callback,
                    dt=dt,  # Pass dt to reconstructor
                )

                # Run reconstruction - always processes all runs now
                reconstructor.run()

                # Load results for each pair
                for cam1, cam2 in camera_pairs:
                    output_cam = reconstructor.determine_output_camera(cam1, cam2)
                    output_dir = (
                        base_root
                        / "calibrated_piv"
                        / str(image_count)
                        / f"cam{output_cam}"
                        / type_name
                    )
                    summary_file = output_dir / "stereo_reconstruction_summary.mat"

                    if summary_file.exists():
                        try:
                            summary_data = scipy.io.loadmat(
                                str(summary_file),
                                squeeze_me=True,
                                struct_as_record=False,
                            )
                            reconstruction_summary = summary_data.get(
                                "reconstruction_summary", {}
                            )

                            # Handle mat_struct objects by converting to dict-like access
                            def safe_get(obj, key, default=None):
                                if hasattr(obj, key):
                                    return getattr(obj, key)
                                elif hasattr(obj, "get"):
                                    return obj.get(key, default)
                                else:
                                    return default

                            # Extract values safely from mat_struct
                            total_processed = safe_get(
                                reconstruction_summary, "total_frames_processed", 0
                            )
                            total_attempted = safe_get(
                                reconstruction_summary, "total_frames_attempted", 0
                            )
                            output_dir = safe_get(
                                reconstruction_summary, "output_directory", ""
                            )
                            config = safe_get(
                                reconstruction_summary, "configuration", {}
                            )
                            timestamp = safe_get(
                                reconstruction_summary, "timestamp", ""
                            )

                            results = {
                                "camera_pair": [cam1, cam2],
                                "output_camera": output_cam,
                                "total_frames_processed": (
                                    int(total_processed)
                                    if total_processed is not None
                                    else 0
                                ),
                                "total_frames_attempted": (
                                    int(total_attempted)
                                    if total_attempted is not None
                                    else 0
                                ),
                                "success_rate": (
                                    float(total_processed)
                                    / max(1, float(total_attempted))
                                    * 100
                                    if total_processed is not None
                                    and total_attempted is not None
                                    else 0
                                ),
                                "output_directory": (
                                    str(output_dir) if output_dir is not None else ""
                                ),
                                "configuration": config if config is not None else {},
                                "timestamp": (
                                    str(timestamp) if timestamp is not None else ""
                                ),
                            }

                            stereo_jobs[job_id]["results"][
                                f"cam{cam1}_cam{cam2}"
                            ] = results

                        except Exception as e:
                            logger.error(
                                f"Failed to load results for pair {cam1}-{cam2}: {e}"
                            )
                            stereo_jobs[job_id]["results"][f"cam{cam1}_cam{cam2}"] = {
                                "error": str(e)
                            }

                # Complete vector calibration
                stereo_jobs[job_id]["status"] = "completed"
                stereo_jobs[job_id]["progress"] = 100
                stereo_jobs[job_id]["current_stage"] = "completed"
                stereo_jobs[job_id]["current_pair"] = "All vector pairs completed"
                stereo_jobs[job_id]["end_time"] = datetime.now().isoformat()

                # Calculate summary statistics for vector calibration
                total_pairs = len(camera_pairs)
                stereo_jobs[job_id]["summary"] = {
                    "total_pairs": total_pairs,
                    "type": "vector_calibration",
                    "completion_time": datetime.now().isoformat(),
                }

                logger.info(
                    f"Stereo vector calibration job {job_id} COMPLETED for {total_pairs} pairs"
                )

                # Small delay to ensure results are fully processed
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Stereo vector calibration job {job_id} failed: {e}")
                stereo_jobs[job_id]["status"] = "failed"
                stereo_jobs[job_id]["error"] = str(e)

        # Start vector calibration in background thread
        thread = threading.Thread(target=run_vector_calibration)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "job_id": job_id,
                "status": "started",
                "message": "Stereo vector calibration started",
            }
        )

    except Exception as e:
        logger.error(f"Error starting stereo vector calibration: {e}")
        return jsonify({"error": str(e)}), 500


@stereo_bp.route("/stereo/vectors/status/<job_id>", methods=["GET"])
def stereo_vector_status(job_id):
    """Get status of running stereo vector calibration job"""
    if not job_id or job_id == "undefined":
        return jsonify({"error": "Invalid job ID"}), 400

    if job_id not in stereo_jobs:
        return jsonify({"error": "Job not found"}), 404

    # Recursively convert mat_struct objects to dicts for JSON serialization
    job_data = _to_dict(stereo_jobs[job_id])
    return jsonify(job_data)


@stereo_bp.route("/stereo/vectors/load_results", methods=["GET"])
def stereo_load_vector_results():
    """Load previously computed stereo vector calibration results"""
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    cam1 = camera_number(request.args.get("cam1", default=1, type=int))
    cam2 = camera_number(request.args.get("cam2", default=2, type=int))
    image_count = request.args.get("image_count", default=1000, type=int)
    type_name = request.args.get("type_name", default="instantaneous")

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])

        # Determine output camera (same logic as StereoReconstructor)
        output_cam = cam1  # Use first camera as output camera
        output_dir = (
            base_root
            / "calibrated_piv"
            / str(image_count)
            / f"cam{output_cam}"
            / type_name
        )
        summary_file = output_dir / "stereo_reconstruction_summary.mat"

        if not summary_file.exists():
            return jsonify(
                {"exists": False, "message": "No stereo vector calibration found"}
            )

        summary_data = scipy.io.loadmat(
            str(summary_file), squeeze_me=True, struct_as_record=False
        )
        reconstruction_summary = summary_data.get("reconstruction_summary", {})

        # Handle mat_struct objects by converting to dict-like access
        def safe_get(obj, key, default=None):
            if hasattr(obj, key):
                return getattr(obj, key)
            elif hasattr(obj, "get"):
                return obj.get(key, default)
            else:
                return default

        # Extract values safely from mat_struct
        total_processed = safe_get(reconstruction_summary, "total_frames_processed", 0)
        total_attempted = safe_get(reconstruction_summary, "total_frames_attempted", 0)
        output_dir = safe_get(reconstruction_summary, "output_directory", "")
        config = safe_get(reconstruction_summary, "configuration", {})
        timestamp = safe_get(reconstruction_summary, "timestamp", "")

        results = {
            "camera_pair": [cam1, cam2],
            "output_camera": output_cam,
            "total_frames_processed": (
                int(total_processed) if total_processed is not None else 0
            ),
            "total_frames_attempted": (
                int(total_attempted) if total_attempted is not None else 0
            ),
            "success_rate": (
                float(total_processed) / max(1, float(total_attempted)) * 100
                if total_processed is not None and total_attempted is not None
                else 0
            ),
            "output_directory": str(output_dir) if output_dir is not None else "",
            "configuration": config if config is not None else {},
            "timestamp": str(timestamp) if timestamp is not None else "",
        }

        # Check if coordinate files exist
        coords_file = output_dir / "coordinates.mat"
        results["coordinates_exist"] = coords_file.exists()

        # Count vector files
        vector_pattern = safe_get(
            safe_get(reconstruction_summary, "configuration", {}),
            "vector_pattern",
            "%05d.mat",
        )
        vector_count = 0
        output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir
        for i in range(1, image_count + 1):
            vector_file = output_path / (vector_pattern % i)
            if vector_file.exists():
                vector_count += 1
        results["vector_files_count"] = vector_count

        return jsonify({"exists": True, "results": results})

    except Exception as e:
        logger.error(f"Error loading stereo vector results: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# UTILITY ROUTES
# ============================================================================


@stereo_bp.route("/stereo/jobs", methods=["GET"])
def stereo_list_jobs():
    """List all stereo jobs and their status"""
    return jsonify(
        {
            "jobs": {
                job_id: {
                    "status": job_data["status"],
                    "progress": job_data["progress"],
                    "start_time": job_data.get("start_time", ""),
                    "end_time": job_data.get("end_time", ""),
                    "type": job_data.get("type", "unknown"),
                    "camera_pairs": job_data.get("camera_pairs", []),
                }
                for job_id, job_data in stereo_jobs.items()
            },
            "total_jobs": len(stereo_jobs),
        }
    )


@stereo_bp.route("/stereo/jobs/<job_id>", methods=["DELETE"])
def stereo_delete_job(job_id):
    """Delete a completed or failed job"""
    if job_id not in stereo_jobs:
        return jsonify({"error": "Job not found"}), 404

    job_status = stereo_jobs[job_id]["status"]
    if job_status in ["completed", "failed"]:
        del stereo_jobs[job_id]
        return jsonify({"message": "Job deleted successfully"})
    else:
        return jsonify({"error": "Cannot delete running job"}), 400


@stereo_bp.route("/stereo/config", methods=["GET"])
def stereo_get_config():
    """Get stereo calibration configuration from config file"""
    try:
        cfg = get_config()
        stereo_config = cfg.calibration.get("stereo", {})

        return jsonify(
            {
                "file_pattern": stereo_config.get(
                    "file_pattern", "planar_calibration_plate_*.tif"
                ),
                "pattern_cols": stereo_config.get("pattern_cols", 10),
                "pattern_rows": stereo_config.get("pattern_rows", 10),
                "dot_spacing_mm": stereo_config.get("dot_spacing_mm", 28.89),
                "asymmetric": stereo_config.get("asymmetric", False),
                "enhance_dots": stereo_config.get("enhance_dots", True),
                "max_correspondence_distance": stereo_config.get(
                    "max_correspondence_distance", 5.0
                ),
                "min_triangulation_angle": stereo_config.get(
                    "min_triangulation_angle", 5.0
                ),
                "vector_pattern": stereo_config.get("vector_pattern", "%05d.mat"),
                "type_name": stereo_config.get("type_name", "instantaneous"),
            }
        )

    except Exception as e:
        logger.error(f"Error getting stereo config: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ADDITIONAL PRODUCTION-READY ENDPOINTS
# ============================================================================


@stereo_bp.route("/stereo/calibration/list_available", methods=["GET"])
def stereo_list_available_calibrations():
    """List all available stereo calibration models"""
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])
        calibration_dir = base_root / "calibration"

        available_models = []

        if calibration_dir.exists():
            # Look for stereo model files
            for stereo_file in calibration_dir.glob("stereo_model_cam*-cam*.mat"):
                filename = stereo_file.name
                # Extract camera numbers from filename
                import re

                match = re.match(r"stereo_model_cam(\d+)-cam(\d+)\.mat", filename)
                if match:
                    cam1, cam2 = map(int, match.groups())

                    # Try to load basic info
                    try:
                        stereo_data = scipy.io.loadmat(
                            str(stereo_file), squeeze_me=True, struct_as_record=False
                        )

                        model_info = {
                            "camera_pair": [cam1, cam2],
                            "filename": filename,
                            "path": str(stereo_file),
                            "stereo_reprojection_error": float(
                                stereo_data.get("stereo_reprojection_error", 0)
                            ),
                            "relative_angle_deg": float(
                                stereo_data.get("relative_angle_deg", 0)
                            ),
                            "num_image_pairs": int(
                                stereo_data.get("num_image_pairs", 0)
                            ),
                            "timestamp": str(stereo_data.get("timestamp", "")),
                            "file_size_mb": stereo_file.stat().st_size / (1024 * 1024),
                        }

                        available_models.append(model_info)

                    except Exception as e:
                        logger.warning(f"Could not load stereo model {filename}: {e}")
                        available_models.append(
                            {
                                "camera_pair": [cam1, cam2],
                                "filename": filename,
                                "path": str(stereo_file),
                                "error": str(e),
                                "file_size_mb": stereo_file.stat().st_size
                                / (1024 * 1024),
                            }
                        )

        return jsonify(
            {
                "available_models": available_models,
                "total_models": len(available_models),
                "calibration_directory": str(calibration_dir),
            }
        )

    except Exception as e:
        logger.error(f"Error listing available calibrations: {e}")
        return jsonify({"error": str(e)}), 500


@stereo_bp.route("/stereo/calibration/validate", methods=["POST"])
def stereo_validate_calibration():
    """Validate stereo calibration parameters before running"""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera_pairs = data.get("camera_pairs", [[1, 2]])
    file_pattern = data.get("file_pattern", "planar_calibration_plate_*.tif")

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])

        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "camera_pairs": [],
        }

        for cam1, cam2 in camera_pairs:
            pair_result = {
                "camera_pair": [cam1, cam2],
                "cam1_images": 0,
                "cam2_images": 0,
                "matching_pairs": 0,
                "directories_exist": True,
                "warnings": [],
                "errors": [],
            }

            # Check directories
            cam1_dir = source_root / "calibration" / f"Cam{cam1}"
            cam2_dir = source_root / "calibration" / f"Cam{cam2}"

            if not cam1_dir.exists():
                pair_result["errors"].append(
                    f"Camera {cam1} directory not found: {cam1_dir}"
                )
                pair_result["directories_exist"] = False
                validation_results["valid"] = False

            if not cam2_dir.exists():
                pair_result["errors"].append(
                    f"Camera {cam2} directory not found: {cam2_dir}"
                )
                pair_result["directories_exist"] = False
                validation_results["valid"] = False

            if pair_result["directories_exist"]:
                # Count images
                def get_image_files(cam_dir, pattern):
                    if "%" in pattern:
                        files = []
                        i = 1
                        while True:
                            filename = pattern % i
                            filepath = cam_dir / filename
                            if filepath.exists():
                                files.append(str(filepath))
                                i += 1
                            else:
                                break
                    else:
                        files = sorted(glob.glob(str(cam_dir / pattern)))
                    return files

                cam1_files = get_image_files(cam1_dir, file_pattern)
                cam2_files = get_image_files(cam2_dir, file_pattern)

                pair_result["cam1_images"] = len(cam1_files)
                pair_result["cam2_images"] = len(cam2_files)

                # Find matching files
                cam1_dict = {Path(f).name: f for f in cam1_files}
                cam2_dict = {Path(f).name: f for f in cam2_files}
                common_names = sorted(set(cam1_dict.keys()) & set(cam2_dict.keys()))
                pair_result["matching_pairs"] = len(common_names)

                # Validation checks
                if len(cam1_files) == 0:
                    pair_result["errors"].append(
                        f"No images found for Camera {cam1} with pattern {file_pattern}"
                    )
                    validation_results["valid"] = False

                if len(cam2_files) == 0:
                    pair_result["errors"].append(
                        f"No images found for Camera {cam2} with pattern {file_pattern}"
                    )
                    validation_results["valid"] = False

                if len(common_names) < 3:
                    pair_result["errors"].append(
                        f"Need at least 3 matching image pairs, found {len(common_names)}"
                    )
                    validation_results["valid"] = False
                elif len(common_names) < 6:
                    pair_result["warnings"].append(
                        f"Only {len(common_names)} matching pairs found. Recommend 6+ for robust calibration"
                    )

                if abs(len(cam1_files) - len(cam2_files)) > 2:
                    pair_result["warnings"].append(
                        f"Significant image count difference: Cam{cam1}={len(cam1_files)}, Cam{cam2}={len(cam2_files)}"
                    )

            validation_results["camera_pairs"].append(pair_result)
            validation_results["warnings"].extend(pair_result["warnings"])
            validation_results["errors"].extend(pair_result["errors"])

        return jsonify(validation_results)

    except Exception as e:
        logger.error(f"Error validating stereo calibration: {e}")
        return jsonify({"error": str(e)}), 500


@stereo_bp.route("/stereo/calibration/preview_grid", methods=["POST"])
def stereo_preview_grid_detection():
    """Preview grid detection on a single image pair"""
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    cam1 = camera_number(data.get("cam1", 1))
    cam2 = camera_number(data.get("cam2", 2))
    filename = data.get("filename", "")
    pattern_cols = int(data.get("pattern_cols", 10))
    pattern_rows = int(data.get("pattern_rows", 10))
    enhance_dots = bool(data.get("enhance_dots", True))
    asymmetric = bool(data.get("asymmetric", False))

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])

        # Load images
        cam1_dir = source_root / "calibration" / f"Cam{cam1}"
        cam2_dir = source_root / "calibration" / f"Cam{cam2}"

        cam1_file = cam1_dir / filename
        cam2_file = cam2_dir / filename

        if not cam1_file.exists() or not cam2_file.exists():
            return jsonify({"error": f"Image files not found: {filename}"}), 404

        # Create temporary calibrator for grid detection
        calibrator = StereoCalibrator(
            source_dir=source_root,
            base_dir=Path("/tmp"),  # Temporary base dir
            camera_pairs=[[cam1, cam2]],
            file_pattern="dummy",
            pattern_cols=pattern_cols,
            pattern_rows=pattern_rows,
            asymmetric=asymmetric,
            enhance_dots=enhance_dots,
        )

        results = {}

        # Process both images
        for cam_num, img_file in [(cam1, cam1_file), (cam2, cam2_file)]:
            img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
            if img is None:
                results[f"cam{cam_num}"] = {
                    "error": f"Could not load image: {img_file}"
                }
                continue

            # Detect grid
            found, grid_points = calibrator.detect_grid_in_image(img)

            cam_result = {
                "camera": cam_num,
                "found": found,
                "image_size": [img.shape[1], img.shape[0]],  # width, height
                "filename": filename,
            }

            if found and grid_points is not None:
                cam_result.update(
                    {
                        "grid_points": grid_points.tolist(),
                        "num_points": len(grid_points),
                        "expected_points": pattern_cols * pattern_rows,
                    }
                )

                # Create visualization
                if img.ndim == 3:
                    vis_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    vis_img = img.copy()

                # Draw detected points
                vis_img_color = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
                for i, (x, y) in enumerate(grid_points):
                    cv2.circle(vis_img_color, (int(x), int(y)), 5, (255, 0, 0), 2)
                    if i < 20:  # Only label first 20 points to avoid clutter
                        cv2.putText(
                            vis_img_color,
                            str(i),
                            (int(x + 8), int(y + 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 0),
                            1,
                        )

                # Convert to base64
                cam_result["preview_image"] = numpy_to_png_base64(vis_img_color)
            else:
                cam_result["error"] = "Grid not detected"

                # Still provide original image
                if img.ndim == 3:
                    vis_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    vis_img = img.copy()
                cam_result["preview_image"] = numpy_to_png_base64(vis_img)

            results[f"cam{cam_num}"] = cam_result

        return jsonify(
            {
                "camera_pair": [cam1, cam2],
                "filename": filename,
                "results": results,
                "both_detected": all(
                    results.get(f"cam{c}", {}).get("found", False) for c in [cam1, cam2]
                ),
            }
        )

    except Exception as e:
        logger.error(f"Error previewing grid detection: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ENHANCED ERROR HANDLING AND VALIDATION
# ============================================================================


def validate_camera_pair(cam1, cam2):
    """Validate camera pair numbers"""
    if cam1 == cam2:
        raise ValueError("Camera numbers must be different")
    if cam1 < 1 or cam2 < 1:
        raise ValueError("Camera numbers must be positive integers")
    return True


def validate_calibration_params(data):
    """Validate calibration parameters"""
    errors = []
    warnings = []

    pattern_cols = data.get("pattern_cols", 10)
    pattern_rows = data.get("pattern_rows", 10)
    dot_spacing_mm = data.get("dot_spacing_mm", 28.89)

    if pattern_cols < 3 or pattern_rows < 3:
        errors.append("Grid pattern must be at least 3x3")

    if pattern_cols * pattern_rows < 20:
        warnings.append("Small grid pattern may affect calibration accuracy")

    if dot_spacing_mm <= 0:
        errors.append("Dot spacing must be positive")

    return errors, warnings


def cleanup_job_history():
    """Clean up old completed jobs to prevent memory leaks"""
    global stereo_jobs
    cutoff_time = time.time() - 3600  # Keep jobs for 1 hour

    jobs_to_remove = []
    for job_id, job_data in stereo_jobs.items():
        if job_data["status"] in ["completed", "failed"]:
            start_time_str = job_data.get("start_time", "")
            try:
                start_time = datetime.fromisoformat(start_time_str).timestamp()
                if start_time < cutoff_time:
                    jobs_to_remove.append(job_id)
            except Exception:
                jobs_to_remove.append(job_id)  # Remove invalid timestamps

    for job_id in jobs_to_remove:
        del stereo_jobs[job_id]

    logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")


# ============================================================================
# ENHANCED CALIBRATION ENDPOINTS WITH BETTER ERROR HANDLING
# ============================================================================


@stereo_bp.route("/stereo/calibration/run_enhanced", methods=["POST"])
def stereo_run_calibration_enhanced():
    """Enhanced stereo calibration with comprehensive validation and error handling"""
    data = request.get_json() or {}

    try:
        # Input validation
        source_path_idx = int(data.get("source_path_idx", 0))
        camera_pairs = data.get("camera_pairs", [[1, 2]])
        file_pattern = data.get("file_pattern", "planar_calibration_plate_*.tif")

        # Validate calibration parameters
        errors, warnings = validate_calibration_params(data)
        if errors:
            return (
                jsonify(
                    {
                        "error": "Validation failed",
                        "details": errors,
                        "warnings": warnings,
                    }
                ),
                400,
            )

        # Validate camera pairs
        for cam1, cam2 in camera_pairs:
            validate_camera_pair(cam1, cam2)

        # Clean up old jobs
        cleanup_job_history()

        # Get configuration
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])
        base_root = Path(cfg.base_paths[source_path_idx])

        # Pre-validate that directories exist
        for cam1, cam2 in camera_pairs:
            cam1_dir = source_root / "calibration" / f"Cam{cam1}"
            cam2_dir = source_root / "calibration" / f"Cam{cam2}"

            if not cam1_dir.exists():
                return (
                    jsonify(
                        {"error": f"Camera {cam1} directory not found: {cam1_dir}"}
                    ),
                    404,
                )
            if not cam2_dir.exists():
                return (
                    jsonify(
                        {"error": f"Camera {cam2} directory not found: {cam2_dir}"}
                    ),
                    404,
                )

        job_id = generate_job_id()

        # Enhanced job status with more detailed tracking
        stereo_jobs[job_id] = {
            "status": "starting",
            "progress": 0,
            "total_pairs": len(camera_pairs),
            "current_pair": None,
            "current_stage": "initialization",
            "processed_pairs": 0,
            "successful_pairs": 0,
            "failed_pairs": 0,
            "results": {},
            "error": None,
            "warnings": warnings,
            "start_time": datetime.now().isoformat(),
            "camera_pairs": camera_pairs,
            "configuration": {
                "file_pattern": file_pattern,
                "pattern_cols": data.get("pattern_cols", 10),
                "pattern_rows": data.get("pattern_rows", 10),
                "dot_spacing_mm": data.get("dot_spacing_mm", 28.89),
                "asymmetric": data.get("asymmetric", False),
                "enhance_dots": data.get("enhance_dots", True),
            },
        }

        def run_calibration_enhanced():
            try:
                stereo_jobs[job_id]["status"] = "running"
                stereo_jobs[job_id]["current_stage"] = "grid_detection"

                # Create calibrator with enhanced error handling
                calibrator = StereoCalibrator(
                    source_dir=source_root,
                    base_dir=base_root,
                    camera_pairs=camera_pairs,
                    file_pattern=file_pattern,
                    pattern_cols=data.get("pattern_cols", 10),
                    pattern_rows=data.get("pattern_rows", 10),
                    dot_spacing_mm=data.get("dot_spacing_mm", 28.89),
                    asymmetric=data.get("asymmetric", False),
                    enhance_dots=data.get("enhance_dots", True),
                )

                # Process each camera pair with detailed progress tracking
                for i, (cam1, cam2) in enumerate(camera_pairs):
                    if job_id not in stereo_jobs:
                        return  # Job was cancelled

                    pair_name = f"Camera {cam1}-{cam2}"
                    stereo_jobs[job_id]["current_pair"] = pair_name
                    stereo_jobs[job_id]["current_stage"] = f"processing_pair_{i + 1}"
                    stereo_jobs[job_id]["progress"] = int(
                        (i / len(camera_pairs)) * 90
                    )  # Leave 10% for finalization

                    try:
                        # Update progress through different stages
                        base_progress = int(
                            (i / len(camera_pairs)) * 80
                        )  # 80% for actual processing

                        # Stage 1: Loading images (0-20% of this pair's progress)
                        stereo_jobs[job_id]["current_stage"] = "loading_images"
                        stereo_jobs[job_id]["progress"] = base_progress + 0
                        logger.info(
                            f"Job {job_id}: Loading images for pair {pair_name}"
                        )

                        # Stage 2: Individual camera calibration (20-60% of this pair's progress)
                        stereo_jobs[job_id][
                            "current_stage"
                        ] = "calibrating_individual_cameras"
                        stereo_jobs[job_id]["progress"] = base_progress + int(
                            20 / len(camera_pairs)
                        )
                        logger.info(
                            f"Job {job_id}: Calibrating individual cameras for pair {pair_name}"
                        )

                        # Stage 3: Stereo calibration (60-90% of this pair's progress)
                        stereo_jobs[job_id]["current_stage"] = "stereo_calibration"
                        stereo_jobs[job_id]["progress"] = base_progress + int(
                            50 / len(camera_pairs)
                        )
                        logger.info(
                            f"Job {job_id}: Performing stereo calibration for pair {pair_name}"
                        )

                        # Actually process the camera pair
                        logger.info(f"Processing stereo pair {pair_name}")
                        calibrator.process_camera_pair(cam1, cam2)

                        # Stage 4: Saving results (90-100% of this pair's progress)
                        stereo_jobs[job_id]["current_stage"] = "saving_results"
                        stereo_jobs[job_id]["progress"] = base_progress + int(
                            70 / len(camera_pairs)
                        )
                        logger.info(
                            f"Job {job_id}: Saving results for pair {pair_name}"
                        )

                        stereo_jobs[job_id]["processed_pairs"] += 1
                        stereo_jobs[job_id]["successful_pairs"] += 1

                        # Load and validate results
                        stereo_file = (
                            base_root
                            / "calibration"
                            / f"stereo_model_cam{cam1}-cam{cam2}.mat"
                        )
                        if stereo_file.exists():
                            stereo_data = scipy.io.loadmat(
                                str(stereo_file),
                                squeeze_me=True,
                                struct_as_record=False,
                            )

                            # Comprehensive results extraction
                            results = {
                                "camera_pair": [cam1, cam2],
                                "calibration_quality": {
                                    "stereo_reprojection_error": float(
                                        stereo_data.get("stereo_reprojection_error", 0)
                                    ),
                                    "relative_angle_deg": float(
                                        stereo_data.get("relative_angle_deg", 0)
                                    ),
                                    "num_image_pairs": int(
                                        stereo_data.get("num_image_pairs", 0)
                                    ),
                                    "baseline_distance": float(
                                        np.linalg.norm(
                                            stereo_data.get(
                                                "translation_vector", [0, 0, 0]
                                            )
                                        )
                                    ),
                                },
                                "camera_intrinsics": {
                                    "camera_matrix_1": (
                                        stereo_data.get("camera_matrix_1", []).tolist()
                                        if hasattr(
                                            stereo_data.get("camera_matrix_1", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                    "camera_matrix_2": (
                                        stereo_data.get("camera_matrix_2", []).tolist()
                                        if hasattr(
                                            stereo_data.get("camera_matrix_2", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                    "dist_coeffs_1": (
                                        stereo_data.get("dist_coeffs_1", []).tolist()
                                        if hasattr(
                                            stereo_data.get("dist_coeffs_1", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                    "dist_coeffs_2": (
                                        stereo_data.get("dist_coeffs_2", []).tolist()
                                        if hasattr(
                                            stereo_data.get("dist_coeffs_2", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                    "focal_length_1": (
                                        [
                                            float(stereo_data["camera_matrix_1"][0, 0]),
                                            float(stereo_data["camera_matrix_1"][1, 1]),
                                        ]
                                        if "camera_matrix_1" in stereo_data
                                        else []
                                    ),
                                    "focal_length_2": (
                                        [
                                            float(stereo_data["camera_matrix_2"][0, 0]),
                                            float(stereo_data["camera_matrix_2"][1, 1]),
                                        ]
                                        if "camera_matrix_2" in stereo_data
                                        else []
                                    ),
                                    "principal_point_1": (
                                        [
                                            float(stereo_data["camera_matrix_1"][0, 2]),
                                            float(stereo_data["camera_matrix_1"][1, 2]),
                                        ]
                                        if "camera_matrix_1" in stereo_data
                                        else []
                                    ),
                                    "principal_point_2": (
                                        [
                                            float(stereo_data["camera_matrix_2"][0, 2]),
                                            float(stereo_data["camera_matrix_2"][1, 2]),
                                        ]
                                        if "camera_matrix_2" in stereo_data
                                        else []
                                    ),
                                },
                                "stereo_geometry": {
                                    "translation_vector": (
                                        stereo_data.get(
                                            "translation_vector", []
                                        ).tolist()
                                        if hasattr(
                                            stereo_data.get("translation_vector", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                    "rotation_matrix": (
                                        stereo_data.get("rotation_matrix", []).tolist()
                                        if hasattr(
                                            stereo_data.get("rotation_matrix", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                    "fundamental_matrix": (
                                        stereo_data.get(
                                            "fundamental_matrix", []
                                        ).tolist()
                                        if hasattr(
                                            stereo_data.get("fundamental_matrix", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                    "essential_matrix": (
                                        stereo_data.get("essential_matrix", []).tolist()
                                        if hasattr(
                                            stereo_data.get("essential_matrix", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                },
                                "rectification": {
                                    "rectification_R1": (
                                        stereo_data.get("rectification_R1", []).tolist()
                                        if hasattr(
                                            stereo_data.get("rectification_R1", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                    "rectification_R2": (
                                        stereo_data.get("rectification_R2", []).tolist()
                                        if hasattr(
                                            stereo_data.get("rectification_R2", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                    "projection_P1": (
                                        stereo_data.get("projection_P1", []).tolist()
                                        if hasattr(
                                            stereo_data.get("projection_P1", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                    "projection_P2": (
                                        stereo_data.get("projection_P2", []).tolist()
                                        if hasattr(
                                            stereo_data.get("projection_P2", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                    "disparity_to_depth_Q": (
                                        stereo_data.get(
                                            "disparity_to_depth_Q", []
                                        ).tolist()
                                        if hasattr(
                                            stereo_data.get("disparity_to_depth_Q", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                },
                                "metadata": {
                                    "timestamp": str(stereo_data.get("timestamp", "")),
                                    "successful_filenames": (
                                        stereo_data.get(
                                            "successful_filenames", []
                                        ).tolist()
                                        if hasattr(
                                            stereo_data.get("successful_filenames", []),
                                            "tolist",
                                        )
                                        else []
                                    ),
                                    "image_size": (
                                        stereo_data.get("image_size", []).tolist()
                                        if hasattr(
                                            stereo_data.get("image_size", []), "tolist"
                                        )
                                        else []
                                    ),
                                },
                            }

                            # Quality assessment
                            reprojection_error = results["calibration_quality"][
                                "stereo_reprojection_error"
                            ]
                            if reprojection_error > 1.0:
                                results["quality_warning"] = (
                                    f"High reprojection error: {reprojection_error:.3f} pixels"
                                )
                            elif reprojection_error > 0.5:
                                results["quality_warning"] = (
                                    f"Moderate reprojection error: {reprojection_error:.3f} pixels"
                                )
                            else:
                                results["quality_status"] = "Good calibration quality"

                            stereo_jobs[job_id]["results"][
                                f"cam{cam1}_cam{cam2}"
                            ] = results
                            logger.info(
                                f"Successfully processed pair {cam1}-{cam2} with reprojection error {reprojection_error:.3f}"
                            )

                        else:
                            raise FileNotFoundError(
                                f"Stereo calibration file not created: {stereo_file}"
                            )

                    except Exception as e:
                        logger.error(f"Failed to calibrate pair {pair_name}: {e}")
                        stereo_jobs[job_id]["failed_pairs"] += 1
                        stereo_jobs[job_id]["processed_pairs"] += 1
                        stereo_jobs[job_id]["results"][f"cam{cam1}_cam{cam2}"] = {
                            "error": str(e),
                            "camera_pair": [cam1, cam2],
                            "failed": True,
                        }

                # Finalization
                stereo_jobs[job_id]["current_stage"] = "completed"
                stereo_jobs[job_id]["current_pair"] = "All pairs completed"
                stereo_jobs[job_id]["progress"] = 100
                stereo_jobs[job_id]["status"] = "completed"
                stereo_jobs[job_id]["end_time"] = datetime.now().isoformat()

                # Calculate summary statistics
                total_pairs = len(camera_pairs)
                successful_pairs = stereo_jobs[job_id]["successful_pairs"]
                failed_pairs = stereo_jobs[job_id]["failed_pairs"]

                stereo_jobs[job_id]["summary"] = {
                    "total_pairs": total_pairs,
                    "successful_pairs": successful_pairs,
                    "failed_pairs": failed_pairs,
                    "success_rate": (
                        (successful_pairs / total_pairs) * 100 if total_pairs > 0 else 0
                    ),
                }

                logger.info(
                    f"Stereo calibration job {job_id} COMPLETED. Success rate: {successful_pairs}/{total_pairs}"
                )
                logger.info(
                    f"Job {job_id} final status: {stereo_jobs[job_id]['status']}"
                )
                logger.info(
                    f"Job {job_id} results keys: {list(stereo_jobs[job_id]['results'].keys())}"
                )

                # Small delay to ensure results are fully processed before frontend can poll
                time.sleep(0.1)

                # Calculate summary statistics
                total_pairs = len(camera_pairs)
                successful_pairs = stereo_jobs[job_id]["successful_pairs"]
                failed_pairs = stereo_jobs[job_id]["failed_pairs"]

                stereo_jobs[job_id]["summary"] = {
                    "total_pairs": total_pairs,
                    "successful_pairs": successful_pairs,
                    "failed_pairs": failed_pairs,
                    "success_rate": (
                        (successful_pairs / total_pairs) * 100 if total_pairs > 0 else 0
                    ),
                }

                logger.info(
                    f"Stereo calibration job {job_id} COMPLETED. Success rate: {successful_pairs}/{total_pairs}"
                )
                logger.info(
                    f"Job {job_id} final status: {stereo_jobs[job_id]['status']}"
                )
                logger.info(
                    f"Job {job_id} results keys: {list(stereo_jobs[job_id]['results'].keys())}"
                )

                # Small delay to ensure results are fully processed before frontend can poll
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Stereo calibration job {job_id} failed: {e}")
                stereo_jobs[job_id]["status"] = "failed"
                stereo_jobs[job_id]["current_stage"] = "failed"
                stereo_jobs[job_id]["error"] = str(e)
                stereo_jobs[job_id]["end_time"] = datetime.now().isoformat()

        # Start calibration in background thread
        thread = threading.Thread(target=run_calibration_enhanced)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "job_id": job_id,
                "status": "started",
                "message": "Enhanced stereo calibration started",
                "warnings": warnings,
                "estimated_duration_minutes": len(camera_pairs) * 2,  # Rough estimate
            }
        )

    except ValueError as e:
        return jsonify({"error": "Validation error", "details": str(e)}), 400
    except Exception as e:
        logger.error(f"Error starting enhanced stereo calibration: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# BATCH PROCESSING AND MONITORING
# ============================================================================


@stereo_bp.route("/stereo/calibration/batch_status", methods=["GET"])
def stereo_batch_status():
    """Get comprehensive status of all running and recent jobs"""
    try:
        cleanup_job_history()  # Clean up old jobs

        running_jobs = []
        completed_jobs = []
        failed_jobs = []

        for job_id, job_data in stereo_jobs.items():
            job_summary = {
                "job_id": job_id,
                "status": job_data["status"],
                "progress": job_data["progress"],
                "current_stage": job_data.get("current_stage", "unknown"),
                "current_pair": job_data.get("current_pair"),
                "start_time": job_data.get("start_time", ""),
                "end_time": job_data.get("end_time", ""),
                "camera_pairs": job_data.get("camera_pairs", []),
                "successful_pairs": job_data.get("successful_pairs", 0),
                "failed_pairs": job_data.get("failed_pairs", 0),
                "total_pairs": job_data.get("total_pairs", 0),
            }

            if job_data["status"] == "running":
                running_jobs.append(job_summary)
            elif job_data["status"] == "completed":
                completed_jobs.append(job_summary)
            elif job_data["status"] == "failed":
                failed_jobs.append(job_summary)

        return jsonify(
            {
                "running_jobs": running_jobs,
                "completed_jobs": completed_jobs[-10:],  # Last 10 completed jobs
                "failed_jobs": failed_jobs[-10:],  # Last 10 failed jobs
                "system_status": {
                    "total_active_jobs": len(running_jobs),
                    "total_jobs_today": len(stereo_jobs),
                    "memory_usage_mb": len(str(stereo_jobs)) / 1024,  # Rough estimate
                },
            }
        )

    except Exception as e:
        logger.error(f"Error getting batch status: {e}")
        return jsonify({"error": str(e)}), 500


@stereo_bp.route("/stereo/calibration/cancel/<job_id>", methods=["POST"])
def stereo_cancel_job(job_id):
    """Cancel a running calibration job"""
    try:
        if job_id not in stereo_jobs:
            return jsonify({"error": "Job not found"}), 404

        job_data = stereo_jobs[job_id]
        if job_data["status"] not in ["running", "starting"]:
            return (
                jsonify(
                    {"error": f"Cannot cancel job with status: {job_data['status']}"}
                ),
                400,
            )

        # Mark job as cancelled
        stereo_jobs[job_id]["status"] = "cancelled"
        stereo_jobs[job_id]["end_time"] = datetime.now().isoformat()
        stereo_jobs[job_id]["error"] = "Job cancelled by user"

        return jsonify({"message": "Job cancelled successfully", "job_id": job_id})

    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        return jsonify({"error": str(e)}), 500
