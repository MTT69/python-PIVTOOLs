"""
Pinhole (Planar) Calibration Views.

Clean API for planar calibration:
- /calibration/planar/validate - Validate calibration images
- /calibration/planar/frame/<idx> - Get single calibration frame
- /calibration/planar/generate_model - Start calibration job for single camera
- /calibration/planar/generate_model_all - Start calibration job for all cameras
- /calibration/planar/job/<job_id> - Poll job status
- /calibration/planar/model - Load saved camera model + detections
"""

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
from pivtools_core.image_handling.path_utils import build_calibration_camera_path

from pivtools_gui.calibration.calibration_planar.planar_calibration_production import MultiViewCalibrator
from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.utils import camera_number, numpy_to_png_base64

pinhole_bp = Blueprint("pinhole", __name__)


# ============================================================================
# ROUTE 1: Validate Calibration Images
# ============================================================================


@pinhole_bp.route("/calibration/planar/validate", methods=["POST"])
def planar_validate():
    """
    Validate calibration images exist and are readable.

    Request JSON:
        source_path_idx: int
        camera: int

    Reads image_format, num_images, subfolder from config.

    Returns:
        JSON with valid, found_count, sample_files, first_image_preview, etc.
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))

    try:
        cfg = get_config()

        # Use centralized validation
        result = validate_calibration_images(
            camera=camera,
            config=cfg,
            source_path_idx=source_path_idx,
        )

        # Add extra fields
        result["container_format"] = cfg.calibration_image_type in ("lavision_set", "lavision_im7", "cine")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            "valid": False,
            "found_count": 0,
            "error": str(e),
        }), 500


# ============================================================================
# ROUTE 2: Get Single Calibration Frame
# ============================================================================


@pinhole_bp.route("/calibration/planar/frame/<int:idx>", methods=["GET"])
def planar_frame(idx: int):
    """
    Get a single calibration frame (image only, no overlay).

    Query params:
        source_path_idx: int
        camera: int

    Returns:
        JSON with image (base64), width, height, stats
    """
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))

    try:
        cfg = get_config()

        # Read image
        img = read_calibration_image(idx, camera, cfg, source_path_idx)

        if img is None:
            return jsonify({"error": f"Could not read frame {idx}"}), 404

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
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error reading frame {idx}: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROUTE 3: Generate Camera Model (Start Job)
# ============================================================================


@pinhole_bp.route("/calibration/planar/generate_model", methods=["POST"])
def planar_generate_model():
    """
    Start camera model generation job.

    Request JSON:
        source_path_idx: int
        camera: int

    All calibration parameters are read from config.yaml.
    Uses MultiViewCalibrator.process_single_camera() for the actual calibration.

    Returns:
        JSON with job_id, status
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))

    # Create job
    job_id = job_manager.create_job(
        "planar",
        processed_images=0,
        valid_images=0,
        total_images=0,
        camera=camera,
    )

    def run_calibration():
        try:
            job_manager.update_job(job_id, status="running")

            # Get config and parameters
            cfg = get_config()
            pinhole_cfg = cfg.data.get("calibration", {}).get("pinhole", {})

            # Get paths
            base_root = Path(cfg.base_paths[source_path_idx])
            subfolder = cfg.calibration_subfolder

            # Build source directory path
            # For calibration images, the source is typically base_path/Cam{N}/subfolder
            source_dir = build_calibration_camera_path(cfg, source_path_idx, camera, subfolder)

            # Debug logging
            logger.info(f"[DEBUG] base_root: {base_root}")
            logger.info(f"[DEBUG] subfolder from config: '{subfolder}'")
            logger.info(f"[DEBUG] source_dir (from build_calibration_camera_path): {source_dir}")
            logger.info(f"[DEBUG] source_dir exists: {source_dir.exists()}")
            logger.info(f"[DEBUG] calibration_image_format: {cfg.calibration_image_format}")
            logger.info(f"[DEBUG] use_camera_subfolders: {cfg.calibration_use_camera_subfolders}")
            if source_dir.exists():
                files = list(source_dir.iterdir())[:10]
                logger.info(f"[DEBUG] Files in source_dir: {[f.name for f in files]}")

            # Create calibrator using the production class
            calibrator = MultiViewCalibrator(
                source_dir=str(source_dir),
                base_dir=str(base_root),
                camera_count=1,  # Processing single camera
                file_pattern=cfg.calibration_image_format,
                pattern_cols=pinhole_cfg.get("pattern_cols", 10),
                pattern_rows=pinhole_cfg.get("pattern_rows", 10),
                dot_spacing_mm=pinhole_cfg.get("dot_spacing_mm", 28.89),
                asymmetric=pinhole_cfg.get("asymmetric", False),
                enhance_dots=pinhole_cfg.get("enhance_dots", True),
                calibration_subfolder="",  # Already included in source_dir
            )

            def progress_callback(progress_data):
                job_manager.update_job(
                    job_id,
                    processed_images=progress_data.get("processed_images", 0),
                    valid_images=progress_data.get("valid_images", 0),
                    total_images=progress_data.get("total_images", 0),
                    progress=progress_data.get("progress", 0),
                )

            # Run calibration using the production class method
            result = calibrator.process_single_camera(
                cam_num=camera,
                progress_callback=progress_callback,
                save_visualizations=False,  # Skip figure generation for GUI
            )

            if result.get("success"):
                job_manager.complete_job(
                    job_id,
                    camera_matrix=result.get("camera_matrix"),
                    dist_coeffs=result.get("dist_coeffs"),
                    rms_error=result.get("rms_error"),
                    num_images_used=result.get("num_images_used"),
                    model_path=result.get("model_path"),
                )
                logger.info(f"Planar calibration completed for camera {camera}")
            else:
                job_manager.fail_job(job_id, result.get("error", "Calibration failed"))

        except Exception as e:
            logger.error(f"Planar calibration job {job_id} failed: {e}")
            job_manager.fail_job(job_id, str(e))

    thread = threading.Thread(target=run_calibration)
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "starting",
        "message": f"Camera model generation started for camera {camera}",
    })


# ============================================================================
# ROUTE 4: Get Job Status
# ============================================================================


@pinhole_bp.route("/calibration/planar/job/<job_id>", methods=["GET"])
def planar_job_status(job_id: str):
    """
    Get calibration job status.

    Returns:
        JSON with status, progress, processed_images, valid_images, total_images,
        elapsed_time, estimated_remaining, error (if failed)
    """
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job_data)


# ============================================================================
# ROUTE 5: Load Saved Model + Detections
# ============================================================================


@pinhole_bp.route("/calibration/planar/model", methods=["GET"])
def planar_load_model():
    """
    Load saved camera model and detection coordinates.

    Query params:
        source_path_idx: int
        camera: int

    Returns:
        JSON with exists, camera_model, detections (per-frame), summary
    """
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))

    try:
        cfg = get_config()
        base_root = Path(cfg.base_paths[source_path_idx])
        cam_output_base = base_root / "calibration" / f"Cam{camera}" / "pinhole_planar"

        model_file = cam_output_base / "model" / "pinhole_model.mat"
        indices_folder = cam_output_base / "indices"

        # Check if model exists
        if not model_file.exists():
            return jsonify({
                "exists": False,
                "message": f"No saved camera model found for camera {camera}",
            })

        # Load camera model
        model_data = scipy.io.loadmat(
            str(model_file), struct_as_record=False, squeeze_me=True
        )

        camera_model = {
            "camera_matrix": model_data["camera_matrix"].tolist(),
            "dist_coeffs": model_data["dist_coeffs"].flatten().tolist(),
            "reprojection_error": float(model_data.get("reprojection_error", 0)),
            "focal_length": [
                float(model_data["camera_matrix"][0, 0]),
                float(model_data["camera_matrix"][1, 1]),
            ],
            "principal_point": [
                float(model_data["camera_matrix"][0, 2]),
                float(model_data["camera_matrix"][1, 2]),
            ],
            "num_images_used": int(model_data.get("num_images_used", 0)),
        }

        # Load per-frame detections
        detections = {}
        if indices_folder.exists():
            for idx_file in sorted(indices_folder.glob("indexing_*.mat")):
                try:
                    frame_num = int(idx_file.stem.split("_")[1])
                    grid_data = scipy.io.loadmat(
                        str(idx_file), struct_as_record=False, squeeze_me=True
                    )

                    detections[str(frame_num)] = {
                        "grid_points": grid_data["grid_points"].tolist(),
                    }

                    if "reprojection_error" in grid_data:
                        detections[str(frame_num)]["reprojection_error"] = float(
                            grid_data["reprojection_error"]
                        )

                except Exception as e:
                    logger.warning(f"Could not load {idx_file}: {e}")

        # Summary
        summary = {
            "total_frames": len(detections),
            "frames_with_detections": len(detections),
            "pattern_size": [
                int(model_data.get("pattern_cols", 10)),
                int(model_data.get("pattern_rows", 10)),
            ],
            "dot_spacing_mm": float(model_data.get("dot_spacing_mm", 28.89)),
        }

        return jsonify({
            "exists": True,
            "camera_model": camera_model,
            "detections": detections,
            "summary": summary,
        })

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROUTE 6: Generate Camera Model for All Cameras
# ============================================================================


@pinhole_bp.route("/calibration/planar/generate_model_all", methods=["POST"])
def planar_generate_model_all():
    """
    Start camera model generation job for all configured cameras.

    Request JSON:
        source_path_idx: int

    All calibration parameters are read from config.yaml.
    Processes each camera in sequence using MultiViewCalibrator.process_single_camera().

    Returns:
        JSON with job_id, status, cameras list
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))

    try:
        cfg = get_config()
        camera_numbers = cfg.camera_numbers

        if not camera_numbers:
            return jsonify({"error": "No cameras configured"}), 400

        # Create multi-camera job
        job_id = job_manager.create_job(
            "planar_all",
            processed_cameras=0,
            total_cameras=len(camera_numbers),
            current_camera=None,
            camera_results={},
        )

        def run_calibration():
            try:
                camera_results = {}
                pinhole_cfg = cfg.data.get("calibration", {}).get("pinhole", {})
                base_root = Path(cfg.base_paths[source_path_idx])
                subfolder = cfg.calibration_subfolder

                for idx, camera in enumerate(camera_numbers):
                    # Update job with current camera
                    job_manager.update_job(
                        job_id,
                        status="running",
                        current_camera=camera,
                        processed_cameras=idx,
                    )

                    try:
                        # Build source directory path
                        source_dir = build_calibration_camera_path(
                            cfg, source_path_idx, camera, subfolder
                        )

                        logger.info(f"Processing camera {camera} from {source_dir}")

                        # Create calibrator
                        calibrator = MultiViewCalibrator(
                            source_dir=str(source_dir),
                            base_dir=str(base_root),
                            camera_count=1,
                            file_pattern=cfg.calibration_image_format,
                            pattern_cols=pinhole_cfg.get("pattern_cols", 10),
                            pattern_rows=pinhole_cfg.get("pattern_rows", 10),
                            dot_spacing_mm=pinhole_cfg.get("dot_spacing_mm", 28.89),
                            asymmetric=pinhole_cfg.get("asymmetric", False),
                            enhance_dots=pinhole_cfg.get("enhance_dots", True),
                            calibration_subfolder="",
                        )

                        # Run calibration
                        result = calibrator.process_single_camera(
                            cam_num=camera,
                            progress_callback=None,
                            save_visualizations=False,
                        )

                        if result.get("success"):
                            camera_results[camera] = {
                                "status": "completed",
                                "rms_error": result.get("rms_error"),
                                "num_images_used": result.get("num_images_used"),
                            }
                            logger.info(f"Camera {camera} calibration completed")
                        else:
                            camera_results[camera] = {
                                "status": "failed",
                                "error": result.get("error", "Unknown error"),
                            }

                    except Exception as e:
                        logger.error(f"Camera {camera} calibration failed: {e}")
                        camera_results[camera] = {
                            "status": "failed",
                            "error": str(e),
                        }

                # Mark job complete
                job_manager.complete_job(
                    job_id,
                    camera_results=camera_results,
                    processed_cameras=len(camera_numbers),
                )

            except Exception as e:
                logger.error(f"Multi-camera calibration job {job_id} failed: {e}")
                job_manager.fail_job(job_id, str(e))

        thread = threading.Thread(target=run_calibration)
        thread.daemon = True
        thread.start()

        return jsonify({
            "job_id": job_id,
            "status": "starting",
            "cameras": camera_numbers,
            "message": f"Camera model generation started for {len(camera_numbers)} cameras",
        })

    except Exception as e:
        logger.error(f"Error starting multi-camera calibration: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROUTE 7: Generate Camera Model Batch (Multi-Path + Multi-Camera)
# ============================================================================


@pinhole_bp.route("/calibration/planar/generate_model_batch", methods=["POST"])
def planar_generate_model_batch():
    """
    Start camera model generation job with batch processing support.

    Request JSON:
        active_paths: list of path indices (default: from config)
        cameras: list of camera numbers (default: from config)

    All calibration parameters are read from config.yaml.
    Processes each (path, camera) combination as a separate sub-job.

    Returns:
        JSON with parent_job_id, sub_jobs list, status
    """
    data = request.get_json() or {}
    logger.info(f"Received batch planar calibration request: {data}")

    try:
        cfg = get_config()
        base_paths = cfg.base_paths
        source_paths = cfg.source_paths

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
            "planar_batch",
            total_targets=len(targets),
        )
        sub_jobs = []

        # Launch a job for each target
        for target in targets:
            base_dir = target.base_path
            source_dir = target.source_path or base_dir
            cam_num = target.camera

            # Create sub-job
            job_id = job_manager.create_job(
                "planar",
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
                target=_run_planar_calibration_job,
                args=(
                    job_id,
                    base_dir,
                    source_dir,
                    cam_num,
                    target.path_idx,
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
            "total_targets": len(targets),
            "processed_targets": len(sub_jobs),
            "status": "starting",
            "message": f"Planar calibration started for {len(sub_jobs)} target(s) "
            f"across {len(valid_paths)} path(s)",
        })

    except Exception as e:
        logger.error(f"Error starting batch planar calibration: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _run_planar_calibration_job(
    job_id: str,
    base_dir: Path,
    source_dir: Path,
    camera: int,
    path_idx: int,
    cfg,
):
    """Run planar calibration job in a background thread."""
    try:
        logger.info(f"[Planar] Starting job {job_id} for Cam{camera} at path {path_idx}")

        job_manager.update_job(job_id, status="running")

        pinhole_cfg = cfg.data.get("calibration", {}).get("pinhole", {})
        subfolder = cfg.calibration_subfolder

        # Build source directory path
        cam_source_dir = build_calibration_camera_path(cfg, path_idx, camera, subfolder)

        logger.info(f"[Planar] Camera source dir: {cam_source_dir}")

        # Create calibrator
        calibrator = MultiViewCalibrator(
            source_dir=str(cam_source_dir),
            base_dir=str(base_dir),
            camera_count=1,
            file_pattern=cfg.calibration_image_format,
            pattern_cols=pinhole_cfg.get("pattern_cols", 10),
            pattern_rows=pinhole_cfg.get("pattern_rows", 10),
            dot_spacing_mm=pinhole_cfg.get("dot_spacing_mm", 28.89),
            asymmetric=pinhole_cfg.get("asymmetric", False),
            enhance_dots=pinhole_cfg.get("enhance_dots", True),
            calibration_subfolder="",
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
        result = calibrator.process_single_camera(
            cam_num=camera,
            progress_callback=progress_callback,
            save_visualizations=False,
        )

        if result.get("success"):
            job_manager.complete_job(
                job_id,
                camera_matrix=result.get("camera_matrix"),
                dist_coeffs=result.get("dist_coeffs"),
                rms_error=result.get("rms_error"),
                num_images_used=result.get("num_images_used"),
                model_path=result.get("model_path"),
            )
            logger.info(f"[Planar] Job {job_id} completed for Cam{camera}")
        else:
            job_manager.fail_job(job_id, result.get("error", "Calibration failed"))
            logger.error(f"[Planar] Job {job_id} failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"[Planar] Job {job_id} error: {e}", exc_info=True)
        job_manager.fail_job(job_id, str(e))


@pinhole_bp.route("/calibration/planar/batch_status/<job_id>", methods=["GET"])
def planar_batch_status(job_id: str):
    """Get batch planar calibration job status with aggregated sub-job info."""
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
