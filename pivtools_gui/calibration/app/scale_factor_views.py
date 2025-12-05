"""
Scale Factor Calibration Views.

Provides Flask endpoints for scale factor calibration with progress tracking.
Uses ScaleFactorCalibrator service for actual calibration logic.
"""

import threading
from pathlib import Path

from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config, reload_config

from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.calibration.scale_factor_calibration_production import ScaleFactorCalibrator

scale_factor_bp = Blueprint("scale_factor", __name__)


@scale_factor_bp.route("/calibration/scale_factor/calibrate_vectors", methods=["POST"])
def scale_factor_calibrate_vectors():
    """
    Start scale factor calibration job with progress tracking for all cameras.

    Request JSON:
        source_path_idx: int - Index into config source_paths
        image_count: int - Number of images to process
        type_name: str - Type of data (default: "instantaneous")

    Note: dt and px_per_mm are read from config.calibration.scale_factor

    Returns:
        JSON with job_id, status, message, cameras, image_count
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    image_count = int(data.get("image_count", 1000))
    type_name = data.get("type_name", "instantaneous")

    # Reload config to get latest dt/px_per_mm values
    cfg = reload_config()
    camera_numbers = cfg.camera_numbers
    base_root = Path(cfg.base_paths[source_path_idx])

    # Log the scale factor settings being used
    sf_cfg = cfg.data.get("calibration", {}).get("scale_factor", {})
    logger.info(
        f"Scale factor calibration using dt={sf_cfg.get('dt', 1.0)}, "
        f"px_per_mm={sf_cfg.get('px_per_mm', 1.0)}"
    )

    # Create job with camera-aware initial data
    job_id = job_manager.create_job(
        "scale_factor",
        processed_runs=0,
        processed_files=0,
        total_files=0,
        current_camera=None,
        total_cameras=len(camera_numbers),
        processed_cameras=0,
        camera_progress={
            f"Cam{cam}": {"total_files": 0, "processed_files": 0, "status": "pending"}
            for cam in camera_numbers
        },
    )

    def run_calibration():
        try:
            job_manager.update_job(job_id, status="running")

            # Create calibrator instance - reads dt/px_per_mm from config
            calibrator = ScaleFactorCalibrator(
                base_path=base_root,
                source_path_idx=source_path_idx,
                type_name=type_name,
                config=cfg,
            )

            def progress_callback(progress_data):
                """Update job with progress from calibrator."""
                current_camera = progress_data.get("current_camera")
                camera_processed = progress_data.get("camera_processed_files", 0)
                camera_total = progress_data.get("camera_total_files", 0)
                overall_processed = progress_data.get("overall_processed_files", 0)
                overall_total = progress_data.get("overall_total_files", 0)
                overall_progress = progress_data.get("overall_progress", 0)

                update_data = {
                    "current_camera": current_camera,
                    "processed_files": overall_processed,
                    "total_files": overall_total,
                    "progress": overall_progress,
                    "processed_cameras": progress_data.get("processed_cameras", 0),
                }

                if current_camera:
                    cam_key = f"Cam{current_camera}"
                    camera_progress = job_manager.get_job(job_id).get(
                        "camera_progress", {}
                    )
                    camera_progress[cam_key] = {
                        "total_files": camera_total,
                        "processed_files": camera_processed,
                        "status": "running",
                    }
                    update_data["camera_progress"] = camera_progress

                job_manager.update_job(job_id, **update_data)

            # Run calibration
            result = calibrator.process_all_cameras(
                cameras=camera_numbers,
                image_count=image_count,
                progress_callback=progress_callback,
            )

            # Update final camera statuses
            camera_progress = {}
            for cam_num, cam_result in result.get("camera_results", {}).items():
                status = "completed"
                if cam_result.get("failed_files", 0) > 0:
                    if cam_result.get("successful_files", 0) == 0:
                        status = "failed"
                    else:
                        status = "completed_with_errors"

                camera_progress[f"Cam{cam_num}"] = {
                    "total_files": cam_result.get("total_files", 0),
                    "processed_files": cam_result.get("processed_files", 0),
                    "successful_files": cam_result.get("successful_files", 0),
                    "failed_files": cam_result.get("failed_files", 0),
                    "status": status,
                }

            job_manager.complete_job(
                job_id,
                camera_progress=camera_progress,
                successful_files=result.get("successful_files", 0),
                failed_files=result.get("failed_files", 0),
                current_camera=None,
            )

            logger.info(
                f"Scale factor calibration completed. "
                f"Processed {result['processed_files']}/{result['total_files']} files "
                f"across {result['processed_cameras']} cameras"
            )

        except Exception as e:
            logger.error(f"Scale factor calibration job {job_id} failed: {e}")
            job_manager.fail_job(job_id, str(e))

    # Start job in background thread
    thread = threading.Thread(target=run_calibration)
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "job_id": job_id,
            "status": "starting",
            "message": f"Scale factor calibration job started for {len(camera_numbers)} camera(s): {camera_numbers}",
            "cameras": camera_numbers,
            "image_count": image_count,
        }
    )


@scale_factor_bp.route("/calibration/scale_factor/status/<job_id>", methods=["GET"])
def scale_factor_status(job_id):
    """
    Get scale factor calibration job status.

    Args:
        job_id: Job ID to query

    Returns:
        JSON with job status, progress, timing info
    """
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job_data)
