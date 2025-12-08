"""
Vector Merging API views.

Thin route handlers that delegate to VectorMerger class.
Provides endpoints for merging vector fields from multiple cameras
with progress tracking and multiprocessing support.
"""

import threading
from pathlib import Path

from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from pivtools_core.batch_utils import iter_batch_targets

from ...calibration.services.job_manager import job_manager
from ...utils import camera_number
from ..vector_merger import VectorMerger

merging_bp = Blueprint("merging", __name__)


@merging_bp.route("/merge_vectors/merge_one", methods=["POST"])
def merge_one_frame():
    """
    Merge vectors for a single frame.

    Request JSON:
        frame_idx: int - Frame number to merge (default: 1)

    All other parameters read from config.yaml merging block:
        - base_path_idx: Which base_path to use
        - cameras: Camera numbers to merge
        - type_name: Vector type (instantaneous, ensemble, etc.)

    Returns:
        JSON with status, frame, runs_merged, message
    """
    data = request.get_json() or {}
    cfg = get_config()

    # All config from config.yaml
    base_path_idx = cfg.merging_base_path_idx
    cameras = [camera_number(c) for c in cfg.merging_cameras]
    type_name = cfg.merging_type_name

    # Only frame_idx accepted from request (for single frame testing)
    frame_idx = int(data.get("frame_idx", 1))

    try:
        base_dir = Path(cfg.base_paths[base_path_idx])

        logger.info(f"Merging frame {frame_idx} for cameras {cameras}")

        # Create merger instance
        merger = VectorMerger(
            base_dir=base_dir,
            cameras=cameras,
            type_name=type_name,
        )

        # Find valid runs
        valid_runs, total_runs = merger.find_valid_runs()

        if not valid_runs:
            return jsonify({"error": "No valid runs found in vector files"}), 400

        logger.info(
            f"Found {len(valid_runs)} valid runs: {valid_runs} (total: {total_runs})"
        )

        # Merge the frame
        merged_runs = merger.merge_single_frame(frame_idx, valid_runs)

        if not merged_runs:
            return jsonify({"error": f"Failed to merge frame {frame_idx}"}), 500

        # Save the result
        merger.save_frame_result(frame_idx, merged_runs, total_runs)

        # Save coordinates
        coords_file = merger.output_dir / "coordinates.mat"
        if not coords_file.exists():
            merger.save_coordinates(merged_runs, total_runs)

        return jsonify({
            "status": "success",
            "frame": frame_idx,
            "runs_merged": len(valid_runs),
            "message": f"Successfully merged frame {frame_idx}",
        })

    except Exception as e:
        logger.error(f"Error merging frame {frame_idx}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@merging_bp.route("/merge_vectors/merge_all", methods=["POST"])
def merge_all_frames():
    """
    Start vector merging job for all frames with multiprocessing.

    All parameters read from config.yaml merging block:
        - base_path_idx: Which base_path to use
        - cameras: Camera numbers to merge
        - type_name: Vector type (instantaneous, ensemble, etc.)

    Returns:
        JSON with job_id, status, message
    """
    cfg = get_config()

    # All config from config.yaml
    base_path_idx = cfg.merging_base_path_idx
    cameras = [camera_number(c) for c in cfg.merging_cameras]
    type_name = cfg.merging_type_name

    try:
        base_dir = Path(cfg.base_paths[base_path_idx])

        # Create job
        job_id = job_manager.create_job(
            "merging",
            cameras=cameras,
            total_frames=cfg.num_frame_pairs,
            processed_frames=0,
        )

        def run_merge_job():
            try:
                job_manager.update_job(job_id, status="running")

                # Create merger instance
                merger = VectorMerger(
                    base_dir=base_dir,
                    cameras=cameras,
                    type_name=type_name,
                )

                def progress_callback(progress_data):
                    job_manager.update_job(
                        job_id,
                        progress=progress_data.get("progress", 0),
                        processed_frames=progress_data.get("processed_frames", 0),
                        message=progress_data.get("message", ""),
                    )

                # Run merge
                result = merger.merge_all_frames(
                    progress_callback=progress_callback,
                )

                if result["success"]:
                    job_manager.complete_job(
                        job_id,
                        processed_count=result.get("processed_count", 0),
                        output_dir=result.get("output_dir", ""),
                        valid_runs=result.get("valid_runs", []),
                    )
                    logger.info(f"Merge job {job_id} completed successfully")
                else:
                    job_manager.fail_job(
                        job_id,
                        result.get("error", "Merge operation failed"),
                    )

            except Exception as e:
                logger.error(f"Merge job {job_id} failed: {e}", exc_info=True)
                job_manager.fail_job(job_id, str(e))

        # Start job in background thread
        thread = threading.Thread(target=run_merge_job)
        thread.daemon = True
        thread.start()

        return jsonify({
            "job_id": job_id,
            "status": "starting",
            "message": f"Vector merging job started for cameras {cameras}",
            "total_frames": cfg.num_frame_pairs,
        })

    except Exception as e:
        logger.error(f"Error starting merge job: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@merging_bp.route("/merge_vectors/status/<job_id>", methods=["GET"])
def merge_status(job_id: str):
    """
    Get vector merging job status with timing information.

    Returns:
        JSON with status, progress, processed_frames, total_frames,
        elapsed_time, estimated_remaining, error (if failed)
    """
    job_data = job_manager.get_job_with_timing(job_id)

    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job_data)


@merging_bp.route("/merge_vectors/merge_all_batch", methods=["POST"])
def merge_all_frames_batch():
    """
    Start vector merging job with multi-path batch support.

    Request JSON:
        active_paths: list of path indices (default: from config)
        cameras: list of camera numbers (optional, defaults to config)
        type_name: str (optional, defaults to config)

    Returns:
        JSON with parent_job_id, sub_jobs list, status
    """
    data = request.get_json() or {}
    logger.info(f"Received batch merge request: {data}")

    cfg = get_config()
    base_paths = cfg.base_paths

    # Get batch parameters
    active_paths = data.get("active_paths")
    if active_paths is None:
        active_paths = cfg.merging_active_paths

    # Cameras from request or config
    cameras_raw = data.get("cameras")
    if cameras_raw:
        cameras = [camera_number(c) for c in cameras_raw]
    else:
        cameras = [camera_number(c) for c in cfg.merging_cameras]

    type_name = data.get("type_name", cfg.merging_type_name)

    # Validate paths
    valid_paths = [i for i in active_paths if 0 <= i < len(base_paths)]
    if not valid_paths:
        return jsonify({"error": "No valid path indices provided"}), 400

    # Need at least 2 cameras for merging
    if len(cameras) < 2:
        return jsonify({"error": "Need at least 2 cameras for merging"}), 400

    try:
        # Create parent job
        parent_job_id = job_manager.create_job(
            "merging_parent",
            total_targets=len(valid_paths),
        )
        sub_jobs = []

        # Launch a job for each path (no camera loop - merging uses all cameras)
        for path_idx in valid_paths:
            base_dir = Path(base_paths[path_idx])

            # Verify data exists for at least 2 cameras
            cameras_with_data = []
            for cam in cameras:
                paths = get_data_paths(
                    base_dir=base_dir,
                    num_frame_pairs=cfg.num_frame_pairs,
                    cam=cam,
                    type_name=type_name,
                )
                if paths["data_dir"].exists():
                    cameras_with_data.append(cam)

            if len(cameras_with_data) < 2:
                logger.warning(
                    f"Skipping path {path_idx}: only {len(cameras_with_data)} cameras have data"
                )
                continue

            # Create sub-job
            job_id = job_manager.create_job(
                "merging",
                cameras=cameras_with_data,
                path_idx=path_idx,
                parent_job_id=parent_job_id,
                total_frames=cfg.num_frame_pairs,
                processed_frames=0,
            )
            sub_jobs.append({
                "job_id": job_id,
                "path_idx": path_idx,
                "cameras": cameras_with_data,
                "label": f"Path {path_idx}",
            })

            # Launch thread
            thread = threading.Thread(
                target=_run_merge_job,
                args=(
                    job_id,
                    base_dir,
                    cameras_with_data,
                    type_name,
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
            "message": f"Vector merging started for {len(sub_jobs)} path(s)",
        })

    except Exception as e:
        logger.error(f"Error starting batch merge job: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _run_merge_job(
    job_id: str,
    base_dir: Path,
    cameras: list,
    type_name: str,
):
    """Run merge job in a background thread."""
    try:
        logger.info(f"[Merge] Starting job {job_id} for cameras {cameras}")

        job_manager.update_job(job_id, status="running")

        # Create merger instance
        merger = VectorMerger(
            base_dir=base_dir,
            cameras=cameras,
            type_name=type_name,
        )

        def progress_callback(progress_data):
            job_manager.update_job(
                job_id,
                progress=progress_data.get("progress", 0),
                processed_frames=progress_data.get("processed_frames", 0),
                message=progress_data.get("message", ""),
            )

        # Run merge
        result = merger.merge_all_frames(
            progress_callback=progress_callback,
        )

        if result["success"]:
            job_manager.complete_job(
                job_id,
                processed_count=result.get("processed_count", 0),
                output_dir=result.get("output_dir", ""),
                valid_runs=result.get("valid_runs", []),
            )
            logger.info(f"[Merge] Job {job_id} completed successfully")
        else:
            job_manager.fail_job(
                job_id,
                result.get("error", "Merge operation failed"),
            )
            logger.error(f"[Merge] Job {job_id} failed")

    except Exception as e:
        logger.error(f"[Merge] Job {job_id} error: {e}", exc_info=True)
        job_manager.fail_job(job_id, str(e))


@merging_bp.route("/merge_vectors/batch_status/<job_id>", methods=["GET"])
def merge_batch_status(job_id: str):
    """Get batch merge job status with aggregated sub-job info."""
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
                sub_status["cameras"] = sub_job.get("cameras", [])
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


@merging_bp.route("/merge_vectors/validate", methods=["POST"])
def merge_validate():
    """
    Validate that vector data exists for all cameras before merging.

    All parameters read from config.yaml merging block:
        - base_path_idx: Which base_path to use
        - cameras: Camera numbers to check
        - type_name: Vector type (instantaneous, ensemble, etc.)

    Returns:
        JSON with valid, cameras_found, valid_runs, total_runs, num_frame_pairs
    """
    cfg = get_config()

    # All config from config.yaml
    base_path_idx = cfg.merging_base_path_idx
    cameras = [camera_number(c) for c in cfg.merging_cameras]
    type_name = cfg.merging_type_name

    try:
        base_dir = Path(cfg.base_paths[base_path_idx])

        # Check which cameras have valid data directories
        cameras_found = []
        for camera in cameras:
            paths = get_data_paths(
                base_dir=base_dir,
                num_frame_pairs=cfg.num_frame_pairs,
                cam=camera,
                type_name=type_name,
            )
            if paths["data_dir"].exists():
                cameras_found.append(camera)

        if len(cameras_found) < 2:
            return jsonify({
                "valid": False,
                "cameras_found": cameras_found,
                "cameras_requested": cameras,
                "error": f"Need at least 2 cameras with data, found {len(cameras_found)}",
            })

        # Create merger to find valid runs
        merger = VectorMerger(
            base_dir=base_dir,
            cameras=cameras_found,
            type_name=type_name,
        )
        valid_runs, total_runs = merger.find_valid_runs()

        return jsonify({
            "valid": len(valid_runs) > 0,
            "cameras_found": cameras_found,
            "cameras_requested": cameras,
            "valid_runs": valid_runs,
            "total_runs": total_runs,
            "num_frame_pairs": cfg.num_frame_pairs,
            "output_dir": str(merger.output_dir),
        })

    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        return jsonify({
            "valid": False,
            "error": str(e),
        }), 500
