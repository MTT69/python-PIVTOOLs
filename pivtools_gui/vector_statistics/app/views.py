"""
Vector Statistics API views

Provides endpoints for computing instantaneous statistics (mean and Reynolds stresses)
with progress tracking.

Pattern matches: pinhole_views.py
"""

import threading
import time
from pathlib import Path

from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from ...calibration.services.job_manager import job_manager
from ...utils import camera_number
from ..instantaneous_statistics import VectorStatisticsProcessor

statistics_bp = Blueprint("statistics", __name__)


@statistics_bp.route("/statistics/calculate", methods=["POST"])
def calculate_statistics():
    """
    Start statistics calculation job.

    Expects JSON with:
        base_path_idx: int
        cameras: list of camera numbers
        include_merged: bool
        type_name: str (default: "instantaneous")
        requested_statistics: list of statistic names (optional)

    Returns:
        JSON with parent_job_id, sub_jobs list, status
    """
    data = request.get_json() or {}
    logger.info(f"Received statistics calculation request: {data}")

    base_path_idx = int(data.get("base_path_idx", 0))
    cameras = data.get("cameras", [])
    include_merged = bool(data.get("include_merged", False))
    type_name = data.get("type_name", "instantaneous")
    requested_statistics = data.get("requested_statistics", None)

    try:
        cfg = get_config()
        base_paths = getattr(cfg, "base_paths", getattr(cfg, "source_paths", []))
        if not base_paths or base_path_idx >= len(base_paths):
            return jsonify({"error": "Invalid base_path_idx"}), 400

        base_dir = Path(base_paths[base_path_idx])
        vector_format = getattr(cfg, "vector_format", "%05d.mat")
        num_frame_pairs = getattr(cfg, "num_frame_pairs", 100)

        # Create parent job to track all sub-jobs
        parent_job_id = job_manager.create_job("statistics_parent")
        sub_jobs = []

        # Process merged data if requested
        if include_merged:
            first_cam = cameras[0] if cameras else 1
            merged_paths = get_data_paths(
                base_dir=base_dir,
                num_frame_pairs=num_frame_pairs,
                cam=first_cam,
                type_name=type_name,
                use_merged=True,
            )

            if merged_paths["data_dir"].exists():
                job_id = job_manager.create_job(
                    "statistics",
                    camera="Merged",
                    parent_job_id=parent_job_id,
                )
                sub_jobs.append({"job_id": job_id, "type": "merged"})

                thread = threading.Thread(
                    target=_run_statistics_job,
                    args=(
                        job_id,
                        merged_paths["data_dir"],
                        base_dir,
                        num_frame_pairs,
                        vector_format,
                        type_name,
                        True,  # use_merged
                        first_cam,
                        requested_statistics,
                    ),
                )
                thread.daemon = True
                thread.start()
            else:
                logger.warning(f"Merged data directory not found: {merged_paths['data_dir']}")

        # Process each camera
        for cam in cameras:
            cam_num = camera_number(cam)

            # Get data paths for this camera
            cam_paths = get_data_paths(
                base_dir=base_dir,
                num_frame_pairs=num_frame_pairs,
                cam=cam_num,
                type_name=type_name,
                use_merged=False,
            )

            job_id = job_manager.create_job(
                "statistics",
                camera=f"Cam{cam_num}",
                parent_job_id=parent_job_id,
            )
            sub_jobs.append({"job_id": job_id, "type": f"camera_{cam_num}"})

            thread = threading.Thread(
                target=_run_statistics_job,
                args=(
                    job_id,
                    cam_paths["data_dir"],
                    base_dir,
                    num_frame_pairs,
                    vector_format,
                    type_name,
                    False,  # use_merged
                    cam_num,
                    requested_statistics,
                ),
            )
            thread.daemon = True
            thread.start()

        # Update parent job with sub_jobs list
        job_manager.update_job(parent_job_id, sub_jobs=sub_jobs, status="running")

        return jsonify({
            "parent_job_id": parent_job_id,
            "sub_jobs": sub_jobs,
            "status": "starting",
            "message": f"Statistics calculation started for {len(cameras)} camera(s)"
            + (" and merged data" if include_merged else ""),
        })

    except Exception as e:
        logger.error(f"Error starting statistics calculation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _run_statistics_job(
    job_id: str,
    data_dir: Path,
    base_dir: Path,
    num_frame_pairs: int,
    vector_format: str,
    type_name: str,
    use_merged: bool,
    camera: int,
    requested_statistics: list,
):
    """
    Run statistics calculation in a background thread.
    Uses VectorStatisticsProcessor.process() which handles parallelism internally.
    """
    try:
        cam_folder = "Merged" if use_merged else f"Cam{camera}"
        logger.info(f"[Statistics] Starting job {job_id} for {cam_folder}")

        job_manager.update_job(job_id, status="running")

        def progress_callback(progress: int):
            job_manager.update_job(job_id, progress=progress)

        # Create processor and run
        processor = VectorStatisticsProcessor(
            data_dir=data_dir,
            base_dir=base_dir,
            num_frame_pairs=num_frame_pairs,
            vector_format=vector_format,
            type_name=type_name,
            use_merged=use_merged,
            camera=camera,
        )

        result = processor.process(
            requested_statistics=requested_statistics,
            save_figures=True,
            progress_callback=progress_callback,
        )

        if result["success"]:
            job_manager.complete_job(
                job_id,
                output_file=result.get("output_file"),
                num_runs=result.get("num_runs"),
            )
            logger.info(f"[Statistics] Job {job_id} completed for {cam_folder}")
        else:
            job_manager.fail_job(job_id, result.get("error", "Unknown error"))
            logger.error(f"[Statistics] Job {job_id} failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"[Statistics] Job {job_id} error: {e}", exc_info=True)
        job_manager.fail_job(job_id, str(e))


@statistics_bp.route("/statistics/status/<job_id>", methods=["GET"])
def get_statistics_status(job_id):
    """Get statistics calculation job status."""
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
                sub_status["type"] = sub_job["type"]
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
