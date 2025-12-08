"""
Transformation views for PIV vector field operations.

Contains endpoints for:
- Add transformation to camera (add_transform)
- Clear camera transformations (clear_transform)
- Get camera transform status (get_transform_status)
- Apply all transforms (apply_transforms) - batch operation

NOTE: Transform options should only be shown when viewing raw PIV vectors
(var_source="inst"), not when viewing statistics.

NOTE: Statistics files are NOT transformed. Users must manually
recalculate statistics after applying transforms.
"""

import threading

from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.transforms import VALID_TRANSFORMATIONS
from pivtools_gui.transforms.transform_operations import simplify_transformations
from pivtools_gui.transforms.transform_production import TransformProcessor
from pivtools_gui.utils import camera_number

transform_bp = Blueprint("transform", __name__)


# =============================================================================
# Add Transformation to Camera
# =============================================================================


@transform_bp.route("/transform/add", methods=["POST"])
def add_transform():
    """
    Add a transformation to a camera's pending list.

    Request JSON:
        camera: int
        transformation: str (one of VALID_TRANSFORMATIONS)

    Returns:
        JSON with success, operations (simplified list), original_count
    """
    try:
        data = request.get_json() or {}
        camera = camera_number(data.get("camera", 1))
        transformation = data.get("transformation", "")

        logger.info(f"add_transform: camera={camera}, transformation={transformation}")

        if transformation not in VALID_TRANSFORMATIONS:
            return jsonify({
                "success": False,
                "error": f"Invalid transformation. Valid: {VALID_TRANSFORMATIONS}"
            }), 400

        config = get_config()

        # Get current operations and add new one
        current_ops = config.get_camera_transforms(camera)
        new_ops = current_ops + [transformation]

        # Simplify
        simplified_ops = simplify_transformations(new_ops)

        # Save to config
        config.set_camera_transforms(camera, simplified_ops)
        config.save()

        logger.info(f"add_transform: simplified {len(new_ops)} -> {len(simplified_ops)} operations")

        return jsonify({
            "success": True,
            "camera": camera,
            "operations": simplified_ops,
            "original_count": len(new_ops),
            "simplified_count": len(simplified_ops),
            "message": f"Added {transformation}, simplified to {len(simplified_ops)} operation(s)",
        })

    except Exception as e:
        logger.exception(f"add_transform error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Clear Camera Transformations
# =============================================================================


@transform_bp.route("/transform/clear", methods=["POST"])
def clear_transform():
    """
    Clear all pending transformations for a camera.

    Request JSON:
        camera: int

    Returns:
        JSON with success
    """
    try:
        data = request.get_json() or {}
        camera = camera_number(data.get("camera", 1))

        logger.info(f"clear_transform: camera={camera}")

        config = get_config()
        config.clear_camera_transforms(camera)
        config.save()

        return jsonify({
            "success": True,
            "camera": camera,
            "message": f"Cleared all transforms for camera {camera}",
        })

    except Exception as e:
        logger.exception(f"clear_transform error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Get Camera Transform Status
# =============================================================================


@transform_bp.route("/transform/status", methods=["GET"])
def get_transform_status():
    """
    Get pending transformations for a camera.

    Query params:
        camera: int

    Returns:
        JSON with operations list, has_pending
    """
    try:
        camera = camera_number(request.args.get("camera", 1))

        config = get_config()
        operations = config.get_camera_transforms(camera)

        return jsonify({
            "success": True,
            "camera": camera,
            "operations": operations,
            "has_pending": len(operations) > 0,
        })

    except Exception as e:
        logger.exception(f"get_transform_status error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Get All Cameras Transform Status
# =============================================================================


@transform_bp.route("/transform/status/all", methods=["GET"])
def get_all_transform_status():
    """
    Get pending transformations for all cameras.

    Returns:
        JSON with cameras dict (camera_num -> operations list)
    """
    try:
        config = get_config()
        all_transforms = config.transforms_cameras

        return jsonify({
            "success": True,
            "cameras": all_transforms,
            "has_any_pending": any(len(ops) > 0 for ops in all_transforms.values()),
        })

    except Exception as e:
        logger.exception(f"get_all_transform_status error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Preview Simplification
# =============================================================================


@transform_bp.route("/transform/simplify", methods=["POST"])
def preview_simplify():
    """
    Preview what a list of transformations would simplify to.

    Request JSON:
        operations: list of transformation names

    Returns:
        JSON with simplified operations
    """
    try:
        data = request.get_json() or {}
        operations = data.get("operations", [])

        simplified = simplify_transformations(operations)

        return jsonify({
            "success": True,
            "original": operations,
            "simplified": simplified,
            "reduced_by": len(operations) - len(simplified),
        })

    except Exception as e:
        logger.exception(f"preview_simplify error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Apply All Transforms (Batch)
# =============================================================================


@transform_bp.route("/transform/apply", methods=["POST"])
def apply_transforms():
    """
    Apply all pending transformations to data files.

    Request JSON:
        cameras: list of camera numbers (optional, defaults to all with pending)
        type_name: str (optional, default "instantaneous")

    NOTE: Statistics files are NOT transformed.
    Users must manually recalculate statistics after this operation.

    Returns:
        JSON with job_id
    """
    try:
        data = request.get_json() or {}
        cameras = data.get("cameras")
        type_name = data.get("type_name", "instantaneous")

        config = get_config()

        # Get cameras with pending transforms
        all_transforms = config.transforms_cameras

        if cameras:
            camera_transforms = {c: all_transforms.get(c, []) for c in cameras if all_transforms.get(c)}
        else:
            camera_transforms = {c: ops for c, ops in all_transforms.items() if ops}

        if not camera_transforms:
            return jsonify({
                "success": False,
                "error": "No pending transformations to apply"
            }), 400

        # Create job
        job_id = job_manager.create_job(
            "transform",
            cameras=list(camera_transforms.keys()),
            camera_transforms={str(k): v for k, v in camera_transforms.items()},
            processed_cameras=0,
            total_cameras=len(camera_transforms),
        )

        def run_transforms():
            try:
                job_manager.update_job(job_id, status="running")

                processor = TransformProcessor(
                    base_dir=config.base_paths[0],
                    camera_transforms=camera_transforms,
                    type_name=type_name,
                    config=config,
                )

                def progress_callback(info):
                    job_manager.update_job(
                        job_id,
                        progress=info.get("progress", 0),
                        current_camera=info.get("current_camera"),
                        processed_cameras=info.get("processed_cameras", 0),
                    )

                result = processor.process_all_cameras(progress_callback)

                if result["success"]:
                    # Clear transforms from config after successful apply
                    for cam in camera_transforms.keys():
                        config.clear_camera_transforms(cam)
                    config.save()

                    job_manager.complete_job(
                        job_id,
                        camera_results={str(k): v for k, v in result["camera_results"].items()},
                        statistics_warning="Statistics files were NOT transformed. Please recalculate statistics if needed.",
                    )
                else:
                    job_manager.fail_job(job_id, "Some cameras failed")

            except Exception as e:
                logger.exception(f"Transform job {job_id} failed: {e}")
                job_manager.fail_job(job_id, str(e))

        thread = threading.Thread(target=run_transforms)
        thread.daemon = True
        thread.start()

        return jsonify({
            "job_id": job_id,
            "status": "starting",
            "cameras": list(camera_transforms.keys()),
            "message": "Transformations started. Note: Statistics will need to be recalculated.",
        })

    except Exception as e:
        logger.exception(f"apply_transforms error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@transform_bp.route("/transform/job/<job_id>", methods=["GET"])
def transform_job_status(job_id: str):
    """Get transform job status."""
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job_data)


# =============================================================================
# Get Valid Transformations
# =============================================================================


@transform_bp.route("/transform/valid", methods=["GET"])
def get_valid_transformations():
    """
    Get list of valid transformation names.

    Returns:
        JSON with valid transformations list
    """
    return jsonify({
        "success": True,
        "valid_transformations": VALID_TRANSFORMATIONS,
    })
