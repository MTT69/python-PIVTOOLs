"""
Transformation views for PIV vector field operations.

Contains endpoints for:
- Single frame transformation (transform_frame)
- Clear transformations (clear_transform)
- Check transformation status (check_transform_status)
- Batch transformation (transform_all_frames, transform_all_frames/status)
"""

import threading
import time
import uuid

from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_gui.transforms import (
    VALID_TRANSFORMATIONS,
    VectorTransformProcessor,
)
from ...utils import camera_number


transform_bp = Blueprint("transform", __name__)

# Module-level job tracking for transformation jobs
transformation_jobs = {}


# =============================================================================
# Single Frame Transformation
# =============================================================================


@transform_bp.route("/transform_frame", methods=["POST"])
def transform_frame():
    """
    Apply transformation to a frame's data and coordinates.

    Request JSON:
        base_path: str
        camera: int
        frame: int
        transformation: str (one of VALID_TRANSFORMATIONS)
        merged: bool (optional)
        type_name: str (optional, default "instantaneous")

    Returns:
        JSON with success, pending_transformations, has_original
    """
    logger.info("transform_frame endpoint called")
    try:
        data = request.get_json() or {}
        base_path = data.get("base_path", "")
        camera = camera_number(data.get("camera", 1))
        frame = int(data.get("frame", 1))
        transformation = data.get("transformation", "")
        merged_raw = data.get("merged", False)
        merged = bool(merged_raw)
        type_name = data.get("type_name", "instantaneous")

        logger.info(
            f"transform_frame: base_path={base_path}, camera={camera}, "
            f"frame={frame}, transformation={transformation}"
        )

        if not base_path:
            return jsonify({"success": False, "error": "base_path required"}), 400

        if transformation not in VALID_TRANSFORMATIONS:
            return jsonify({
                "success": False,
                "error": f"Invalid transformation. Valid: {VALID_TRANSFORMATIONS}"
            }), 400

        # Use the processor
        processor = VectorTransformProcessor(
            base_path=base_path,
            transformations=[transformation],
            camera=camera,
            type_name=type_name,
            use_merged=merged,
        )

        result = processor.transform_single_frame(
            frame=frame,
            camera=camera,
            transformation=transformation,
        )

        if result["success"]:
            return jsonify({
                "success": True,
                "message": f"Transformation {transformation} applied successfully",
                "pending_transformations": result.get("pending_transformations", []),
                "has_original": result.get("has_original", True),
            })
        else:
            return jsonify({"success": False, "error": result.get("error", "Unknown error")}), 400

    except ValueError as e:
        logger.warning(f"transform_frame: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception(f"transform_frame: unexpected error: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500


# =============================================================================
# Clear Transformations
# =============================================================================


@transform_bp.route("/clear_transform", methods=["POST"])
def clear_transform():
    """
    Reset transformations for a specific frame by restoring from _original backups.

    Request JSON:
        base_path: str
        camera: int
        frame: int
        merged: bool (optional)
        type_name: str (optional)

    Returns:
        JSON with success, has_original
    """
    logger.info("clear_transform endpoint called")
    try:
        data = request.get_json() or {}
        base_path = data.get("base_path", "")
        camera = camera_number(data.get("camera", 1))
        frame = int(data.get("frame", 1))
        merged_raw = data.get("merged", False)
        merged = bool(merged_raw)
        type_name = data.get("type_name", "instantaneous")

        logger.info(f"clear_transform: base_path={base_path}, camera={camera}, frame={frame}")

        if not base_path:
            return jsonify({"success": False, "error": "base_path required"}), 400

        processor = VectorTransformProcessor(
            base_path=base_path,
            transformations=[],  # No transformations needed for clear
            camera=camera,
            type_name=type_name,
            use_merged=merged,
        )

        result = processor.clear_transformations(frame=frame, camera=camera)

        if result["success"]:
            return jsonify({
                "success": True,
                "message": "Transformations reset to original",
                "has_original": result.get("has_original", False),
            })
        else:
            return jsonify({"success": False, "error": result.get("error", "Unknown error")}), 400

    except ValueError as e:
        logger.warning(f"clear_transform: validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception(f"clear_transform: unexpected error: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500


# =============================================================================
# Check Transformation Status
# =============================================================================


@transform_bp.route("/check_transform_status", methods=["GET"])
def check_transform_status():
    """
    Check if a frame has pending transformations and original backup.

    Query params:
        base_path: str
        camera: int
        frame: int
        merged: str (optional)
        type_name: str (optional)

    Returns:
        JSON with has_original, pending_transformations
    """
    try:
        base_path = request.args.get("base_path", "")
        camera = camera_number(request.args.get("camera", 1))
        frame = int(request.args.get("frame", 1))
        merged_raw = request.args.get("merged", "0")
        merged = merged_raw in ("1", "true", "True", "TRUE")
        type_name = request.args.get("type_name", "instantaneous")

        if not base_path:
            return jsonify({"success": False, "error": "base_path required"}), 400

        processor = VectorTransformProcessor(
            base_path=base_path,
            transformations=[],
            camera=camera,
            type_name=type_name,
            use_merged=merged,
        )

        result = processor.get_transform_status(frame=frame, camera=camera)

        if result["success"]:
            return jsonify({
                "success": True,
                "has_original": result.get("has_original", False),
                "pending_transformations": result.get("pending_transformations", []),
            })
        else:
            return jsonify({"success": False, "error": result.get("error", "Unknown error")}), 400

    except Exception as e:
        logger.exception(f"check_transform_status: unexpected error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Batch Transformation
# =============================================================================


@transform_bp.route("/transform_all_frames", methods=["POST"])
def transform_all_frames():
    """
    Apply transformations to all frames across all cameras.

    Gets pending transformations from the source frame and applies them
    to all other frames.

    Request JSON:
        base_path: str
        camera: int (source camera)
        frame: int (source frame with pending transformations)
        merged: bool (optional)
        type_name: str (optional)

    Returns:
        JSON with job_id, status
    """
    logger.info("transform_all_frames endpoint called")
    data = request.get_json() or {}
    base_path = data.get("base_path", "")
    source_camera = camera_number(data.get("camera", 1))
    source_frame = int(data.get("frame", 1))
    merged_raw = data.get("merged", False)
    merged = bool(merged_raw)
    type_name = data.get("type_name", "instantaneous")

    logger.info(
        f"transform_all_frames: base_path={base_path}, "
        f"source_camera={source_camera}, source_frame={source_frame}"
    )

    if not base_path:
        return jsonify({"success": False, "error": "base_path required"}), 400

    # Get pending transformations from source frame
    try:
        processor = VectorTransformProcessor(
            base_path=base_path,
            transformations=[],  # Will get from source frame
            camera=None,  # Process all cameras
            type_name=type_name,
            use_merged=merged,
        )

        status_result = processor.get_transform_status(frame=source_frame, camera=source_camera)
        if not status_result["success"]:
            return jsonify({"success": False, "error": status_result.get("error", "Unknown error")}), 400

        transformations = status_result.get("pending_transformations", [])
        if not transformations:
            return jsonify({"success": False, "error": "No pending transformations found on source frame"}), 400

    except Exception as e:
        logger.exception(f"Error getting transform status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    job_id = str(uuid.uuid4())

    def run_transformation():
        try:
            transformation_jobs[job_id] = {
                "status": "starting",
                "progress": 0,
                "processed_cameras": 0,
                "processed_frames": 0,
                "total_frames": 0,
                "start_time": time.time(),
                "error": None,
            }

            # Create processor with the actual transformations
            proc = VectorTransformProcessor(
                base_path=base_path,
                transformations=transformations,
                camera=None,  # All cameras
                type_name=type_name,
                use_merged=merged,
            )

            def progress_callback(info):
                transformation_jobs[job_id].update({
                    "status": "running",
                    "progress": info.get("progress", 0),
                    "processed_frames": info.get("processed_frames", 0),
                    "total_frames": info.get("total_frames", 0),
                    "processed_cameras": info.get("processed_cameras", 0),
                })

            result = proc.transform_all_frames(
                source_frame=source_frame,
                source_camera=source_camera,
                progress_callback=progress_callback,
            )

            if result["success"]:
                transformation_jobs[job_id].update({
                    "status": "completed",
                    "progress": 100,
                    "total_frames": result.get("total_frames", 0),
                })
            else:
                transformation_jobs[job_id].update({
                    "status": "failed",
                    "error": result.get("error", "Unknown error"),
                })

        except Exception as e:
            logger.exception(f"Transformation job {job_id} failed: {e}")
            transformation_jobs[job_id].update({
                "status": "failed",
                "error": str(e),
            })

    # Start job in background thread
    thread = threading.Thread(target=run_transformation)
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "starting",
        "message": f"Transformations {transformations} job started across all cameras",
        "transformations": transformations,
    })


@transform_bp.route("/transform_all_frames/status/<job_id>", methods=["GET"])
def transform_all_frames_status(job_id):
    """
    Get transformation job status.

    Returns:
        JSON with status, progress, processed_frames, etc.
    """
    if job_id not in transformation_jobs:
        return jsonify({"error": "Job not found"}), 404

    job_data = transformation_jobs[job_id].copy()

    # Add timing info
    if "start_time" in job_data:
        elapsed = time.time() - job_data["start_time"]
        job_data["elapsed_time"] = elapsed

        if job_data["status"] == "running" and job_data.get("progress", 0) > 0:
            estimated_total = elapsed / (job_data["progress"] / 100.0)
            job_data["estimated_remaining"] = max(0, estimated_total - elapsed)

    return jsonify(job_data)
