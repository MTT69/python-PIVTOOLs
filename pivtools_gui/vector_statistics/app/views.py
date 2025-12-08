"""
Vector Statistics API views

Provides endpoints for computing instantaneous statistics (mean and Reynolds stresses)
with progress tracking.

Supports batch processing: multi-path + multi-camera + optional merged data.

Pattern matches: pinhole_views.py
"""

import threading
import time
from pathlib import Path

from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from pivtools_core.batch_utils import iter_batch_targets
from ...calibration.services.job_manager import job_manager
from ...utils import camera_number
from ..instantaneous_statistics import VectorStatisticsProcessor

statistics_bp = Blueprint("statistics", __name__)


@statistics_bp.route("/statistics/calculate", methods=["POST"])
def calculate_statistics():
    """
    Start statistics calculation job with batch processing support.

    Expects JSON with:
        active_paths: list of path indices (default: from config or [0])
        cameras: list of camera numbers
        include_merged: bool
        type_name: str (default: "instantaneous")
        requested_statistics: list of statistic names (optional)

    Supports multi-path batch processing: for each path, processes all cameras
    and optionally merged data.

    Returns:
        JSON with parent_job_id, sub_jobs list, status
    """
    data = request.get_json() or {}
    logger.info(f"Received statistics calculation request: {data}")

    # Get parameters - support both old (base_path_idx) and new (active_paths) API
    active_paths = data.get("active_paths")
    if active_paths is None:
        # Backward compatibility: convert single base_path_idx to list
        base_path_idx = data.get("base_path_idx")
        if base_path_idx is not None:
            active_paths = [int(base_path_idx)]

    cameras = data.get("cameras", [])
    include_merged = bool(data.get("include_merged", False))
    type_name = data.get("type_name", "instantaneous")
    requested_statistics = data.get("requested_statistics", None)

    try:
        cfg = get_config()
        base_paths = cfg.base_paths

        # Use config defaults if not provided in request
        if active_paths is None:
            active_paths = cfg.statistics_active_paths
        if not cameras:
            cameras = cfg.statistics_cameras

        # Validate paths
        valid_paths = [i for i in active_paths if 0 <= i < len(base_paths)]
        if not valid_paths:
            return jsonify({"error": "No valid path indices provided"}), 400

        vector_format = cfg.vector_format
        num_frame_pairs = cfg.num_frame_pairs

        # Generate batch targets using unified utility
        targets = iter_batch_targets(
            base_paths=base_paths,
            active_paths=valid_paths,
            cameras=cameras,
            include_merged=include_merged,
        )

        if not targets:
            return jsonify({"error": "No targets to process"}), 400

        # Create parent job to track all sub-jobs
        parent_job_id = job_manager.create_job(
            "statistics_parent",
            total_targets=len(targets),
        )
        sub_jobs = []

        # Launch a job for each target
        for target in targets:
            base_dir = target.base_path

            # Determine data directory for this target
            use_merged = target.is_merged
            cam_num = target.camera if target.camera else 1

            target_paths = get_data_paths(
                base_dir=base_dir,
                num_frame_pairs=num_frame_pairs,
                cam=cam_num,
                type_name=type_name,
                use_merged=use_merged,
            )

            data_dir = target_paths["data_dir"]

            # Check if data exists
            if not data_dir.exists():
                logger.warning(f"Data directory not found: {data_dir}, skipping target")
                continue

            # Create sub-job
            job_id = job_manager.create_job(
                "statistics",
                camera=target.label,
                path_idx=target.path_idx,
                parent_job_id=parent_job_id,
            )
            sub_jobs.append({
                "job_id": job_id,
                "type": "merged" if use_merged else f"camera_{cam_num}",
                "path_idx": target.path_idx,
                "label": target.label,
            })

            # Launch thread
            thread = threading.Thread(
                target=_run_statistics_job,
                args=(
                    job_id,
                    data_dir,
                    base_dir,
                    num_frame_pairs,
                    vector_format,
                    type_name,
                    use_merged,
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
            "total_targets": len(targets),
            "processed_targets": len(sub_jobs),
            "status": "starting",
            "message": f"Statistics calculation started for {len(sub_jobs)} target(s) "
            f"across {len(valid_paths)} path(s)",
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
