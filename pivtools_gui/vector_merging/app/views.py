"""
Vector Merging API views.

Thin route handlers that delegate to VectorMerger class.
Provides endpoints for merging vector fields from multiple cameras
with progress tracking and multiprocessing support.

Simplified: Merging always uses all cameras from config.camera_numbers.
"""

import threading
from pathlib import Path

from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths

from ...calibration.services.job_manager import job_manager
from ...utils import camera_number
from ..vector_merger import VectorMerger

merging_bp = Blueprint("merging", __name__)


@merging_bp.route("/merge_vectors/constraints", methods=["GET"])
def get_merge_constraints():
    """
    Return constraints for vector merging.

    Used by frontend to disable options that are not valid.
    Stereo 3D vector data cannot be merged (uz component present).

    Returns:
        JSON with allowed_source_endpoints, is_stereo_setup, stereo_blocked, stereo_reason
    """
    cfg = get_config()

    is_stereo = cfg.is_stereo_setup
    stereo_reason = None
    if is_stereo:
        stereo_reason = (
            "3D stereo vectors cannot be merged. Vector merging is only "
            "supported for planar 2D PIV data."
        )

    return jsonify({
        "allowed_source_endpoints": cfg.get_allowed_endpoints("merging"),
        "is_stereo_setup": is_stereo,
        "stereo_blocked": is_stereo,
        "stereo_reason": stereo_reason,
    })


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
    # Always use all cameras from config.camera_numbers
    cameras = [camera_number(c) for c in cfg.camera_numbers]
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

    Simplified API:
        base_path_idx: int - Single path index (default: from config)
        type_name: str - Vector type (default: from config)

    Always merges all cameras from config.camera_numbers.

    Returns:
        JSON with job_id, status, message
    """
    data = request.get_json() or {}
    cfg = get_config()

    # Check stereo constraint first
    if cfg.is_stereo_setup:
        return jsonify({
            "error": (
                "Cannot merge stereo 3D vector data. Vector merging is only "
                "supported for planar 2D PIV data."
            ),
            "is_stereo_blocked": True,
        }), 400

    # Get parameters from request or config
    base_path_idx = int(data.get("base_path_idx", cfg.merging_base_path_idx))
    type_name = data.get("type_name", cfg.merging_type_name)

    # Always use all cameras from config.camera_numbers
    cameras = [camera_number(c) for c in cfg.camera_numbers]

    # Need at least 2 cameras for merging
    if len(cameras) < 2:
        return jsonify({"error": "Need at least 2 cameras for merging"}), 400

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

    # Check stereo constraint first
    if cfg.is_stereo_setup:
        return jsonify({
            "valid": False,
            "error": (
                "Cannot merge stereo 3D vector data. Vector merging is only "
                "supported for planar 2D PIV data."
            ),
            "is_stereo_blocked": True,
        }), 400

    # All config from config.yaml
    base_path_idx = cfg.merging_base_path_idx
    # Always use all cameras from config.camera_numbers
    cameras = [camera_number(c) for c in cfg.camera_numbers]
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
