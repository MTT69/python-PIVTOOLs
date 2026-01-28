"""
Vector Statistics API views

Provides endpoints for computing instantaneous statistics (mean and Reynolds stresses)
with progress tracking.

Simplified API: single base_path_idx + process_merged boolean.
- process_merged=False: Processes all cameras from config.camera_numbers
- process_merged=True: Processes merged data only
"""

import threading
from pathlib import Path

from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from ...calibration.services.job_manager import job_manager
from ..instantaneous_statistics import VectorStatisticsProcessor

statistics_bp = Blueprint("statistics", __name__)


@statistics_bp.route("/statistics/constraints", methods=["GET"])
def get_statistics_constraints():
    """
    Return constraints for statistics calculation.

    Used by frontend to show allowed source endpoints and workflow options.
    Statistics have no blocking constraints - all endpoints allowed.

    For stereo setups:
    - is_stereo=True indicates two cameras combined into single 3D field
    - Only "stereo" workflow option available (processes single stereo result)

    Returns:
        JSON with allowed_source_endpoints, workflow_options, is_stereo
    """
    cfg = get_config()
    is_stereo_config = cfg.is_stereo_setup

    # File-based detection for stereo data
    # Check if stereo_calibrated folder has data (regardless of calibration.active setting)
    has_stereo_data = False
    if cfg.base_paths:
        base_path = Path(cfg.base_paths[0])
        stereo_calibrated_dir = base_path / "stereo_calibrated" / str(cfg.num_frame_pairs)
        if stereo_calibrated_dir.exists():
            has_stereo_data = any(stereo_calibrated_dir.iterdir())

    # Stereo workflow available if EITHER config says stereo OR stereo data exists
    is_stereo = is_stereo_config or has_stereo_data

    # For stereo, only stereo workflow available (single combined result)
    # For planar, per_camera/after_merge/both options available
    # If BOTH stereo data and non-stereo data exist, offer all options
    if is_stereo and not has_stereo_data:
        # Config is stereo-only (no planar data expected)
        workflow_options = ["stereo"]
    elif has_stereo_data and not is_stereo_config:
        # Stereo data exists but config is planar - offer all options including stereo
        workflow_options = ["per_camera", "after_merge", "both", "stereo"]
    elif has_stereo_data and is_stereo_config:
        # Both stereo config and stereo data - primarily stereo but allow others if data exists
        workflow_options = ["stereo"]
    else:
        workflow_options = ["per_camera", "after_merge", "both"]

    return jsonify({
        "allowed_source_endpoints": cfg.get_allowed_endpoints("statistics"),
        "workflow_options": workflow_options,
        "current_workflow": cfg.statistics_workflow,
        "current_source_endpoint": cfg.statistics_source_endpoint,
        "is_stereo": is_stereo,
        "has_stereo_data": has_stereo_data,  # File-based detection result
    })


@statistics_bp.route("/statistics/calculate", methods=["POST"])
def calculate_statistics():
    """
    Start statistics calculation job.

    API:
        base_path_idx: int - Single path index (default: 0)
        workflow: str - Workflow mode (default: from config)
            - "per_camera": Process all cameras from config.camera_numbers
            - "after_merge": Process merged data only
            - "both": Process all cameras then merged data
        process_merged: bool - DEPRECATED, use workflow instead
        type_name: str (default: "instantaneous")
        requested_statistics: list of statistic names (optional)

    Returns:
        JSON with parent_job_id, sub_jobs list, status
    """
    data = request.get_json() or {}
    logger.info(f"Received statistics calculation request: {data}")

    base_path_idx = int(data.get("base_path_idx", 0))
    type_name = data.get("type_name", "instantaneous")
    requested_statistics = data.get("requested_statistics", None)

    try:
        cfg = get_config()
        base_paths = cfg.base_paths

        # Validate path index first (needed for stereo detection)
        if base_path_idx < 0 or base_path_idx >= len(base_paths):
            return jsonify({"error": f"Invalid base_path_idx: {base_path_idx}"}), 400

        base_dir = base_paths[base_path_idx]

        # Get workflow - support both new workflow param and deprecated process_merged
        workflow = data.get("workflow")
        source_endpoint = data.get("source_endpoint", "regular")
        if workflow is None:
            # Check deprecated process_merged for backward compat
            if "process_merged" in data:
                process_merged = bool(data.get("process_merged", False))
                workflow = "after_merge" if process_merged else "per_camera"
            else:
                workflow = cfg.statistics_workflow

        # Check if stereo setup (config-based)
        is_stereo_config = cfg.is_stereo_setup

        # File-based detection for stereo data
        base_dir_path = Path(base_dir)
        stereo_calibrated_dir = base_dir_path / "stereo_calibrated" / str(cfg.num_frame_pairs)
        has_stereo_data = stereo_calibrated_dir.exists() and any(stereo_calibrated_dir.iterdir()) if stereo_calibrated_dir.exists() else False

        # Stereo available if EITHER config says stereo OR stereo data exists
        is_stereo = is_stereo_config or has_stereo_data

        # Auto-select stereo workflow if source_endpoint is stereo
        if source_endpoint == "stereo" and has_stereo_data:
            workflow = "stereo"

        # Validate workflow - allow stereo if data exists
        if is_stereo_config and not has_stereo_data:
            # Config-only stereo (no data yet) - only stereo workflow
            valid_workflows = ("stereo",)
            workflow = "stereo"
        elif has_stereo_data:
            # Stereo data exists - allow stereo workflow alongside others
            valid_workflows = ("per_camera", "after_merge", "both", "stereo")
        else:
            valid_workflows = ("per_camera", "after_merge", "both")

        if workflow not in valid_workflows:
            return jsonify({
                "error": f"Invalid workflow '{workflow}'. Must be one of: {valid_workflows}"
            }), 400
        vector_format = cfg.vector_format
        num_frame_pairs = cfg.num_frame_pairs

        # Build targets based on workflow
        targets = []
        if workflow == "stereo":
            # Stereo: single combined 3D result, use stereo path structure
            stereo_pairs = cfg.stereo_pairs
            if stereo_pairs:
                cam_pair = stereo_pairs[0]
                targets.append({
                    "camera": cam_pair[0],  # Reference camera
                    "camera_pair": cam_pair,
                    "is_merged": False,
                    "is_stereo": True,
                    "label": f"Stereo Cam{cam_pair[0]}_Cam{cam_pair[1]}",
                })
            else:
                # Fallback if no stereo pairs configured
                targets.append({
                    "camera": 1,
                    "camera_pair": (1, 2),
                    "is_merged": False,
                    "is_stereo": True,
                    "label": "Stereo Cam1_Cam2",
                })
        elif workflow == "after_merge":
            # Process merged data only
            targets.append({
                "camera": None,
                "is_merged": True,
                "label": "Merged",
            })
        elif workflow == "both":
            # Process all cameras first, then merged
            for cam in cfg.camera_numbers:
                targets.append({
                    "camera": cam,
                    "is_merged": False,
                    "label": f"Cam{cam}",
                })
            targets.append({
                "camera": None,
                "is_merged": True,
                "label": "Merged",
            })
        else:  # per_camera (default)
            # Process all cameras from config.camera_numbers
            for cam in cfg.camera_numbers:
                targets.append({
                    "camera": cam,
                    "is_merged": False,
                    "label": f"Cam{cam}",
                })

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
            use_merged = target["is_merged"]
            use_stereo = target.get("is_stereo", False)
            stereo_camera_pair = target.get("camera_pair")
            cam_num = target["camera"] if target["camera"] else 1

            target_paths = get_data_paths(
                base_dir=base_dir,
                num_frame_pairs=num_frame_pairs,
                cam=cam_num,
                type_name=type_name,
                use_merged=use_merged,
                use_stereo=use_stereo,
                stereo_camera_pair=stereo_camera_pair,
            )

            data_dir = target_paths["data_dir"]

            # Check if data exists
            if not data_dir.exists():
                logger.warning(f"Data directory not found: {data_dir}, skipping target")
                continue

            # Create sub-job
            job_id = job_manager.create_job(
                "statistics",
                camera=target["label"],
                path_idx=base_path_idx,
                parent_job_id=parent_job_id,
            )
            sub_jobs.append({
                "job_id": job_id,
                "type": "merged" if use_merged else f"camera_{cam_num}",
                "path_idx": base_path_idx,
                "label": target["label"],
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
                    use_stereo,
                    stereo_camera_pair,
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
            "message": f"Statistics calculation started for {len(sub_jobs)} target(s)",
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
    use_stereo: bool = False,
    stereo_camera_pair: tuple = None,
):
    """
    Run statistics calculation in a background thread.
    Uses VectorStatisticsProcessor.process() which handles parallelism internally.
    """
    try:
        # Determine folder name for logging
        if use_stereo and stereo_camera_pair:
            cam_folder = f"Cam{stereo_camera_pair[0]}_Cam{stereo_camera_pair[1]}"
        elif use_merged:
            cam_folder = "Merged"
        else:
            cam_folder = f"Cam{camera}"
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
            use_stereo=use_stereo,
            stereo_camera_pair=stereo_camera_pair,
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
