"""
Stereo Dotboard Calibration Views.

Clean API for stereo dotboard calibration using circle grid detection:
- /calibration/stereo/dotboard/validate - Validate calibration images for both cameras
- /calibration/stereo/dotboard/frame/<idx> - Get single calibration frame (with camera param)
- /calibration/stereo/dotboard/generate_model - Start stereo calibration job
- /calibration/stereo/dotboard/job/<job_id> - Poll job status
- /calibration/stereo/dotboard/model - Load saved stereo model + detections
- /calibration/stereo/dotboard/reconstruct - Start 3D vector reconstruction job
"""

import re
import threading
from pathlib import Path

import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_core.batch_utils import iter_batch_targets
from pivtools_core.image_handling.calibration_loader import (
    read_calibration_image,
    validate_calibration_images,
)

from pivtools_gui.stereo_reconstruction.stereo_dotboard_calibration_production import StereoDotboardCalibrator
from pivtools_gui.stereo_reconstruction.stereo_reconstruction_production import StereoReconstructor
from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.utils import camera_number, numpy_to_png_base64

stereo_dotboard_bp = Blueprint("stereo_dotboard", __name__)


# ============================================================================
# ROUTE 1: Validate Calibration Images (Both Cameras)
# ============================================================================


@stereo_dotboard_bp.route("/calibration/stereo/dotboard/validate", methods=["POST"])
def stereo_dotboard_validate():
    """
    Validate calibration images exist for BOTH cameras in a stereo pair.

    Request JSON:
        source_path_idx: int
        cam1: int
        cam2: int

    Returns:
        JSON with:
        - valid: bool (True only if both cameras valid)
        - cam1: { valid, found_count, error, ... }
        - cam2: { valid, found_count, error, ... }
        - matching_count: int (number of frames found in both)
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    cam1 = camera_number(data.get("cam1", 1))
    cam2 = camera_number(data.get("cam2", 2))

    try:
        cfg = get_config()

        # Validate both cameras
        result1 = validate_calibration_images(
            camera=cam1,
            config=cfg,
            source_path_idx=source_path_idx,
        )
        result2 = validate_calibration_images(
            camera=cam2,
            config=cfg,
            source_path_idx=source_path_idx,
        )

        # Compute matching count
        # For container formats, both must be valid
        is_container1 = result1.get("found_count") == "container"
        is_container2 = result2.get("found_count") == "container"

        if is_container1 and is_container2:
            # Both containers - count as matching if both valid
            matching_count = "container" if (result1.get("valid") and result2.get("valid")) else 0
        elif is_container1 or is_container2:
            # Mixed format - take the non-container count
            matching_count = result2.get("found_count", 0) if is_container1 else result1.get("found_count", 0)
        else:
            # Both standard formats - take minimum
            count1 = result1.get("found_count", 0)
            count2 = result2.get("found_count", 0)
            if isinstance(count1, int) and isinstance(count2, int):
                matching_count = min(count1, count2)
            else:
                matching_count = 0

        # Overall validity
        both_valid = result1.get("valid", False) and result2.get("valid", False)

        return jsonify({
            "valid": both_valid,
            "cam1": result1,
            "cam2": result2,
            "matching_count": matching_count,
            "container_format": is_container1 or is_container2,
        })

    except Exception as e:
        logger.error(f"Stereo validation error: {e}")
        return jsonify({
            "valid": False,
            "cam1": {"valid": False, "error": str(e)},
            "cam2": {"valid": False, "error": str(e)},
            "matching_count": 0,
            "error": str(e),
        }), 500


# ============================================================================
# ROUTE 2: Get Single Calibration Frame
# ============================================================================


@stereo_dotboard_bp.route("/calibration/stereo/dotboard/frame/<int:idx>", methods=["GET"])
def stereo_dotboard_frame(idx: int):
    """
    Get a single calibration frame for stereo viewing.

    Query params:
        source_path_idx: int
        camera: int (which camera to display - cam1 or cam2 value)

    Returns:
        JSON with image (base64), width, height, stats
    """
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))

    try:
        cfg = get_config()

        # Read image using centralized loader
        img = read_calibration_image(idx, camera, cfg, source_path_idx)

        if img is None:
            return jsonify({"error": f"Could not read frame {idx} for camera {camera}"}), 404

        # Calculate stats
        stats = {
            "min": float(img.min()),
            "max": float(img.max()),
            "mean": float(img.mean()),
            "vmin_pct": float(np.percentile(img, 1)),
            "vmax_pct": float(np.percentile(img, 99)),
        }

        # Normalize to uint8 for display
        vmin = np.percentile(img, 1)
        vmax = np.percentile(img, 99)
        if vmax > vmin:
            disp = ((img - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.uint8)
        else:
            disp = np.zeros_like(img, dtype=np.uint8)

        # Encode to base64
        b64 = numpy_to_png_base64(disp)

        return jsonify({
            "image": b64,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "stats": stats,
            "frame_idx": idx,
            "camera": camera,
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error reading stereo frame {idx}: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROUTE 3: Generate Stereo Model (Start Job)
# ============================================================================


@stereo_dotboard_bp.route("/calibration/stereo/dotboard/generate_model", methods=["POST"])
def stereo_dotboard_generate_model():
    """
    Start stereo calibration job for a camera pair.

    Request JSON:
        source_path_idx: int
        cam1: int
        cam2: int

    All calibration parameters are read from config.yaml.
    Uses StereoDotboardCalibrator.process_camera_pair() for the actual calibration.

    Returns:
        JSON with job_id, status
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    cam1 = camera_number(data.get("cam1", 1))
    cam2 = camera_number(data.get("cam2", 2))

    # Create job
    job_id = job_manager.create_job(
        "stereo_dotboard",
        processed_pairs=0,
        valid_pairs=0,
        total_pairs=0,
        cam1=cam1,
        cam2=cam2,
        stage="starting",
    )

    def run_calibration():
        try:
            job_manager.update_job(job_id, status="running", stage="detecting")

            # Get config
            cfg = get_config()
            stereo_cfg = cfg.stereo_dotboard_calibration

            # Create calibrator using config
            calibrator = StereoDotboardCalibrator(
                config=cfg,
                camera_pair=[cam1, cam2],
                source_path_idx=source_path_idx,
            )

            def progress_callback(progress_data):
                job_manager.update_job(
                    job_id,
                    progress=progress_data.get("progress", 0),
                    stage=progress_data.get("stage", "detecting"),
                    processed_pairs=progress_data.get("processed_pairs", 0),
                    valid_pairs=progress_data.get("valid_pairs", 0),
                    total_pairs=progress_data.get("total_pairs", 0),
                )

            # Run calibration
            result = calibrator.process_camera_pair(
                cam1=cam1,
                cam2=cam2,
                progress_callback=progress_callback,
                save_visualizations=True,
            )

            if result.get("success"):
                job_manager.complete_job(
                    job_id,
                    stereo_rms_error=result.get("stereo_rms_error"),
                    cam1_rms_error=result.get("cam1_rms_error"),
                    cam2_rms_error=result.get("cam2_rms_error"),
                    num_pairs_used=result.get("num_pairs_used"),
                    relative_angle_deg=result.get("relative_angle_deg"),
                    model_path=result.get("model_path"),
                )
                logger.info(f"Stereo dotboard calibration completed for cameras {cam1}-{cam2}")
            else:
                job_manager.fail_job(job_id, result.get("error", "Calibration failed"))

        except Exception as e:
            logger.error(f"Stereo dotboard calibration job {job_id} failed: {e}")
            job_manager.fail_job(job_id, str(e))

    thread = threading.Thread(target=run_calibration)
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "starting",
        "message": f"Stereo calibration started for cameras {cam1} and {cam2}",
    })


# ============================================================================
# ROUTE 4: Get Job Status
# ============================================================================


@stereo_dotboard_bp.route("/calibration/stereo/dotboard/job/<job_id>", methods=["GET"])
def stereo_dotboard_job_status(job_id: str):
    """
    Get stereo calibration job status.

    Returns:
        JSON with status, progress, stage, processed_pairs, valid_pairs, total_pairs,
        elapsed_time, estimated_remaining, error (if failed)
    """
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job_data)


# ============================================================================
# ROUTE 5: Load Saved Model + Detections
# ============================================================================


@stereo_dotboard_bp.route("/calibration/stereo/dotboard/model", methods=["GET"])
def stereo_dotboard_load_model():
    """
    Load saved stereo calibration model and detection coordinates.

    Query params:
        source_path_idx: int
        cam1: int
        cam2: int

    Returns:
        JSON with:
        - exists: bool
        - stereo_model: { camera matrices, distortion, rotation, translation, etc. }
        - detections_cam1: { frame_idx: { grid_points, ... } }
        - detections_cam2: { frame_idx: { grid_points, ... } }
        - summary: { total_pairs, pattern_params, etc. }
    """
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    cam1 = camera_number(request.args.get("cam1", default=1, type=int))
    cam2 = camera_number(request.args.get("cam2", default=2, type=int))

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])
        output_dir = base_root / "calibration" / f"stereo_cam{cam1}_cam{cam2}"

        model_file = output_dir / "model" / "stereo_model.mat"
        indices_folder = output_dir / "indices"

        # Check if model exists
        if not model_file.exists():
            return jsonify({
                "exists": False,
                "message": f"No saved stereo model found for cameras {cam1}-{cam2}",
            })

        # Load stereo model
        model_data = scipy.io.loadmat(
            str(model_file), struct_as_record=False, squeeze_me=True
        )

        # Extract stereo-specific data
        stereo_model = {
            # Camera 1 intrinsics
            "camera_matrix_1": model_data["camera_matrix_1"].tolist(),
            "dist_coeffs_1": model_data["dist_coeffs_1"].flatten().tolist(),
            "focal_length_1": [
                float(model_data["camera_matrix_1"][0, 0]),
                float(model_data["camera_matrix_1"][1, 1]),
            ],
            "principal_point_1": [
                float(model_data["camera_matrix_1"][0, 2]),
                float(model_data["camera_matrix_1"][1, 2]),
            ],
            # Camera 2 intrinsics
            "camera_matrix_2": model_data["camera_matrix_2"].tolist(),
            "dist_coeffs_2": model_data["dist_coeffs_2"].flatten().tolist(),
            "focal_length_2": [
                float(model_data["camera_matrix_2"][0, 0]),
                float(model_data["camera_matrix_2"][1, 1]),
            ],
            "principal_point_2": [
                float(model_data["camera_matrix_2"][0, 2]),
                float(model_data["camera_matrix_2"][1, 2]),
            ],
            # Stereo geometry
            "rotation_matrix": model_data["rotation_matrix"].tolist(),
            "translation_vector": model_data["translation_vector"].flatten().tolist(),
            "essential_matrix": model_data["essential_matrix"].tolist(),
            "fundamental_matrix": model_data["fundamental_matrix"].tolist(),
            # Rectification
            "rectification_R1": model_data["rectification_R1"].tolist(),
            "rectification_R2": model_data["rectification_R2"].tolist(),
            "projection_P1": model_data["projection_P1"].tolist(),
            "projection_P2": model_data["projection_P2"].tolist(),
            "disparity_to_depth_Q": model_data["disparity_to_depth_Q"].tolist(),
            # Quality metrics
            "stereo_rms_error": float(model_data.get("stereo_rms_error", 0)),
            "cam1_rms_error": float(model_data.get("cam1_rms_error", 0)),
            "cam2_rms_error": float(model_data.get("cam2_rms_error", 0)),
            "relative_angle_deg": float(model_data.get("relative_angle_deg", 0)),
            "num_image_pairs": int(model_data.get("num_image_pairs", 0)),
            # Baseline distance (magnitude of translation vector)
            "baseline_distance_mm": float(np.linalg.norm(model_data["translation_vector"])),
        }

        # Load per-frame detections for both cameras
        detections_cam1 = {}
        detections_cam2 = {}
        if indices_folder.exists():
            for idx_file in sorted(indices_folder.glob("indexing_*.mat")):
                try:
                    grid_data = scipy.io.loadmat(
                        str(idx_file), struct_as_record=False, squeeze_me=True
                    )

                    # Extract original source frame index from stored filename
                    # Format is "frame_00004" where 4 is the actual source frame index
                    original_filename = grid_data.get("original_filename", "")
                    if hasattr(original_filename, "item"):
                        original_filename = original_filename.item()  # Handle numpy scalar

                    # Parse frame number from original filename (e.g., "frame_00004" → 4)
                    match = re.search(r'(\d+)$', str(original_filename))
                    if match:
                        source_frame_idx = int(match.group(1))
                    else:
                        # Fallback to filename-based index for backward compatibility
                        source_frame_idx = int(idx_file.stem.split("_")[1])

                    if "grid_points_cam1" in grid_data:
                        detections_cam1[str(source_frame_idx)] = {
                            "grid_points": grid_data["grid_points_cam1"].tolist(),
                        }
                    if "grid_points_cam2" in grid_data:
                        detections_cam2[str(source_frame_idx)] = {
                            "grid_points": grid_data["grid_points_cam2"].tolist(),
                        }

                except Exception as e:
                    logger.warning(f"Could not load {idx_file}: {e}")

        # Summary info
        pattern_params = {}
        if "pattern_params" in model_data:
            pp = model_data["pattern_params"]
            if hasattr(pp, "_fieldnames"):
                for field in pp._fieldnames:
                    val = getattr(pp, field)
                    pattern_params[field] = val if not hasattr(val, "tolist") else val.tolist()
            elif isinstance(pp, dict):
                pattern_params = pp

        # Handle image_size - ensure it's converted from numpy array
        image_size = model_data.get("image_size", [0, 0]) if "image_size" in model_data else [0, 0]
        if hasattr(image_size, "tolist"):
            image_size = image_size.tolist()

        summary = {
            "total_pairs": stereo_model["num_image_pairs"],
            "frames_with_detections": max(len(detections_cam1), len(detections_cam2)),
            "pattern_params": pattern_params,
            "image_size": image_size,
        }

        return jsonify({
            "exists": True,
            "stereo_model": stereo_model,
            "detections_cam1": detections_cam1,
            "detections_cam2": detections_cam2,
            "summary": summary,
        })

    except Exception as e:
        logger.error(f"Error loading stereo model: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROUTE 6: Reconstruct 3D Vectors (Start Job)
# ============================================================================


@stereo_dotboard_bp.route("/calibration/stereo/dotboard/reconstruct", methods=["POST"])
def stereo_dotboard_reconstruct():
    """
    Start 3D vector reconstruction from stereo calibration.

    Request JSON:
        source_path_idx: int
        cam1: int
        cam2: int
        type_name: str ('instantaneous' or 'ensemble')

    Returns:
        JSON with job_id, status
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    cam1 = camera_number(data.get("cam1", 1))
    cam2 = camera_number(data.get("cam2", 2))
    type_name = data.get("type_name", "instantaneous")

    # Create job
    job_id = job_manager.create_job(
        "stereo_reconstruct",
        camera_pair=[cam1, cam2],
        type_name=type_name,
        processed_frames=0,
        successful_frames=0,
        total_frames=0,
    )

    def run_reconstruction():
        try:
            job_manager.update_job(job_id, status="running")

            # Get config
            cfg = get_config()
            base_root = Path(cfg.base_paths[source_path_idx])
            num_frame_pairs = cfg.num_frame_pairs

            # Create reconstructor
            reconstructor = StereoReconstructor(
                base_dir=str(base_root),
                camera_pair=[cam1, cam2],
                model_type="dotboard",  # Using dotboard stereo model
                type_name=type_name,
                config=cfg,
            )

            def progress_callback(progress_data):
                job_manager.update_job(
                    job_id,
                    progress=progress_data.get("progress", 0),
                    processed_frames=progress_data.get("processed_frames", 0),
                    successful_frames=progress_data.get("successful_frames", 0),
                    total_frames=progress_data.get("total_frames", 0),
                )

            # Run reconstruction
            reconstructor.process_run(
                num_frame_pairs=num_frame_pairs,
                progress_cb=progress_callback,
            )

            job_manager.complete_job(
                job_id,
                message=f"3D reconstruction completed for cameras {cam1}-{cam2}",
            )
            logger.info(f"Stereo reconstruction completed for cameras {cam1}-{cam2}")

        except Exception as e:
            logger.error(f"Stereo reconstruction job {job_id} failed: {e}")
            job_manager.fail_job(job_id, str(e))

    thread = threading.Thread(target=run_reconstruction)
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "starting",
        "message": f"3D reconstruction started for cameras {cam1} and {cam2}",
        "type_name": type_name,
    })


# ============================================================================
# ROUTE 7: Get Reconstruction Job Status
# ============================================================================


@stereo_dotboard_bp.route("/calibration/stereo/dotboard/reconstruct/status/<job_id>", methods=["GET"])
def stereo_dotboard_reconstruct_status(job_id: str):
    """
    Get stereo reconstruction job status.

    Returns:
        JSON with status, progress, processed_frames, successful_frames, total_frames,
        elapsed_time, error (if failed)
    """
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job_data)


# ============================================================================
# ROUTE 8: Generate Stereo Model Batch (Multi-Path)
# ============================================================================


@stereo_dotboard_bp.route("/calibration/stereo/dotboard/generate_model_batch", methods=["POST"])
def stereo_dotboard_generate_model_batch():
    """
    Start stereo dotboard calibration with batch processing support.

    For stereo calibration, we process one camera pair per path (no camera loop).

    Request JSON:
        active_paths: list of path indices (default: from config)
        cam1: int (default: from config stereo_dotboard.camera_pair[0])
        cam2: int (default: from config stereo_dotboard.camera_pair[1])

    Returns:
        JSON with parent_job_id, sub_jobs list, status
    """
    data = request.get_json() or {}
    logger.info(f"Received batch stereo dotboard calibration request: {data}")

    try:
        cfg = get_config()
        base_paths = cfg.base_paths
        stereo_cfg = cfg.stereo_dotboard_calibration

        # Get batch parameters
        active_paths = data.get("active_paths")
        if active_paths is None:
            active_paths = cfg.calibration_active_paths

        # Get camera pair from request or config
        cam1 = camera_number(data.get("cam1", stereo_cfg.get("camera_pair", [1, 2])[0]))
        cam2 = camera_number(data.get("cam2", stereo_cfg.get("camera_pair", [1, 2])[1]))

        # Validate paths
        valid_paths = [i for i in active_paths if 0 <= i < len(base_paths)]
        if not valid_paths:
            return jsonify({"error": "No valid path indices provided"}), 400

        # Create parent job
        parent_job_id = job_manager.create_job(
            "stereo_dotboard_batch",
            total_targets=len(valid_paths),
            cam1=cam1,
            cam2=cam2,
        )
        sub_jobs = []

        # Launch a job for each path (one camera pair per path)
        for path_idx in valid_paths:
            base_dir = Path(base_paths[path_idx])

            # Create sub-job
            job_id = job_manager.create_job(
                "stereo_dotboard",
                path_idx=path_idx,
                parent_job_id=parent_job_id,
                cam1=cam1,
                cam2=cam2,
                processed_pairs=0,
                valid_pairs=0,
                total_pairs=0,
                stage="starting",
            )
            sub_jobs.append({
                "job_id": job_id,
                "path_idx": path_idx,
                "camera_pair": [cam1, cam2],
                "label": f"Path {path_idx}",
            })

            # Launch thread
            thread = threading.Thread(
                target=_run_stereo_dotboard_job,
                args=(
                    job_id,
                    base_dir,
                    path_idx,
                    cam1,
                    cam2,
                    cfg,
                ),
            )
            thread.daemon = True
            thread.start()

        # Update parent job
        job_manager.update_job(parent_job_id, sub_jobs=sub_jobs, status="running")

        return jsonify({
            "parent_job_id": parent_job_id,
            "sub_jobs": sub_jobs,
            "total_targets": len(valid_paths),
            "processed_targets": len(sub_jobs),
            "status": "starting",
            "message": f"Stereo dotboard calibration started for {len(sub_jobs)} path(s)",
        })

    except Exception as e:
        logger.error(f"Error starting batch stereo dotboard calibration: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _run_stereo_dotboard_job(
    job_id: str,
    base_dir: Path,
    path_idx: int,
    cam1: int,
    cam2: int,
    cfg,
):
    """Run stereo dotboard calibration job in a background thread."""
    try:
        logger.info(f"[StereoDotboard] Starting job {job_id} for path {path_idx}")

        job_manager.update_job(job_id, status="running", stage="detecting")

        # Create calibrator
        calibrator = StereoDotboardCalibrator(
            config=cfg,
            camera_pair=[cam1, cam2],
            source_path_idx=path_idx,
        )

        def progress_callback(progress_data):
            job_manager.update_job(
                job_id,
                progress=progress_data.get("progress", 0),
                stage=progress_data.get("stage", "detecting"),
                processed_pairs=progress_data.get("processed_pairs", 0),
                valid_pairs=progress_data.get("valid_pairs", 0),
                total_pairs=progress_data.get("total_pairs", 0),
            )

        # Run calibration
        result = calibrator.process_camera_pair(
            cam1=cam1,
            cam2=cam2,
            progress_callback=progress_callback,
            save_visualizations=True,
        )

        if result.get("success"):
            job_manager.complete_job(
                job_id,
                stereo_rms_error=result.get("stereo_rms_error"),
                cam1_rms_error=result.get("cam1_rms_error"),
                cam2_rms_error=result.get("cam2_rms_error"),
                num_pairs_used=result.get("num_pairs_used"),
                relative_angle_deg=result.get("relative_angle_deg"),
                model_path=result.get("model_path"),
            )
            logger.info(f"[StereoDotboard] Job {job_id} completed for path {path_idx}")
        else:
            job_manager.fail_job(job_id, result.get("error", "Calibration failed"))
            logger.error(f"[StereoDotboard] Job {job_id} failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"[StereoDotboard] Job {job_id} error: {e}", exc_info=True)
        job_manager.fail_job(job_id, str(e))


@stereo_dotboard_bp.route("/calibration/stereo/dotboard/batch_status/<job_id>", methods=["GET"])
def stereo_dotboard_batch_status(job_id: str):
    """Get batch stereo dotboard calibration job status with aggregated sub-job info."""
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
                sub_status["label"] = sub_job.get("label", "")
                sub_status["camera_pair"] = sub_job.get("camera_pair", [])
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
