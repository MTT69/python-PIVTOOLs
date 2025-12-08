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
from pivtools_core.batch_utils import iter_batch_targets
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
        type_name: str (optional, defaults to config.transforms_type_name)
        base_path_idx: int (optional, defaults to config.transforms_base_path_idx)

    NOTE: Statistics files are NOT transformed.
    Users must manually recalculate statistics after this operation.

    Returns:
        JSON with job_id
    """
    try:
        data = request.get_json() or {}
        cameras = data.get("cameras")

        config = get_config()

        # Get type_name and base_path_idx from request or config
        type_name = data.get("type_name", config.transforms_type_name)
        base_path_idx = int(data.get("base_path_idx", config.transforms_base_path_idx))

        # Validate base_path_idx
        if base_path_idx >= len(config.base_paths):
            return jsonify({
                "success": False,
                "error": f"Invalid base_path_idx: {base_path_idx}. Only {len(config.base_paths)} base paths configured."
            }), 400

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
            base_path_idx=base_path_idx,
        )

        def run_transforms():
            try:
                job_manager.update_job(job_id, status="running")

                processor = TransformProcessor(
                    base_dir=config.base_paths[base_path_idx],
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
# Apply Transforms (Batch - Multi-Path + Merged)
# =============================================================================


@transform_bp.route("/transform/apply_batch", methods=["POST"])
def apply_transforms_batch():
    """
    Apply all pending transformations with batch processing support.

    Supports multi-path and merged data processing.

    Request JSON:
        active_paths: list of path indices (default: from config)
        cameras: list of camera numbers (optional, defaults to all with pending)
        include_merged: bool (default: from config)
        type_name: str (optional, defaults to config.transforms_type_name)

    NOTE: Statistics files are NOT transformed.
    Users must manually recalculate statistics after this operation.

    Returns:
        JSON with parent_job_id, sub_jobs list, status
    """
    try:
        data = request.get_json() or {}
        logger.info(f"Received batch transform request: {data}")

        config = get_config()
        base_paths = config.base_paths

        # Get batch parameters
        active_paths = data.get("active_paths")
        if active_paths is None:
            active_paths = config.transforms_active_paths

        cameras = data.get("cameras")
        include_merged = data.get("include_merged")
        if include_merged is None:
            include_merged = config.transforms_include_merged

        type_name = data.get("type_name", config.transforms_type_name)

        # Validate paths
        valid_paths = [i for i in active_paths if 0 <= i < len(base_paths)]
        if not valid_paths:
            return jsonify({"error": "No valid path indices provided"}), 400

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

        # Build camera list from those with transforms
        cameras_with_ops = list(camera_transforms.keys())

        # Generate batch targets
        targets = iter_batch_targets(
            base_paths=base_paths,
            active_paths=valid_paths,
            cameras=cameras_with_ops,
            include_merged=include_merged,
        )

        if not targets:
            return jsonify({"error": "No targets to process"}), 400

        # Create parent job
        parent_job_id = job_manager.create_job(
            "transform_parent",
            total_targets=len(targets),
        )
        sub_jobs = []

        # Launch a job for each target
        for target in targets:
            base_dir = target.base_path
            use_merged = target.is_merged
            cam_num = target.camera if target.camera else 1

            # Get transforms for this camera
            cam_transforms = camera_transforms.get(cam_num, [])
            if not cam_transforms and not use_merged:
                continue

            # For merged data, apply transforms from first camera (or skip if none)
            if use_merged:
                # Use first camera's transforms for merged data
                cam_transforms = list(camera_transforms.values())[0] if camera_transforms else []
                if not cam_transforms:
                    continue

            # Create sub-job
            job_id = job_manager.create_job(
                "transform",
                camera=target.label,
                path_idx=target.path_idx,
                parent_job_id=parent_job_id,
                operations=cam_transforms,
            )
            sub_jobs.append({
                "job_id": job_id,
                "type": "merged" if use_merged else f"camera_{cam_num}",
                "path_idx": target.path_idx,
                "label": target.label,
            })

            # Launch thread
            thread = threading.Thread(
                target=_run_transform_job,
                args=(
                    job_id,
                    base_dir,
                    cam_num,
                    cam_transforms,
                    type_name,
                    use_merged,
                    config,
                ),
            )
            thread.daemon = True
            thread.start()

        # Update parent job
        job_manager.update_job(parent_job_id, sub_jobs=sub_jobs, status="running")

        # Clear transforms from config after starting jobs
        for cam in camera_transforms.keys():
            config.clear_camera_transforms(cam)
        config.save()

        return jsonify({
            "parent_job_id": parent_job_id,
            "sub_jobs": sub_jobs,
            "total_targets": len(targets),
            "processed_targets": len(sub_jobs),
            "status": "starting",
            "message": f"Transformations started for {len(sub_jobs)} target(s). "
            "Note: Statistics will need to be recalculated.",
        })

    except Exception as e:
        logger.exception(f"apply_transforms_batch error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def _run_transform_job(
    job_id: str,
    base_dir,
    camera: int,
    operations: list,
    type_name: str,
    use_merged: bool,
    config,
):
    """Run transform job in a background thread."""
    try:
        cam_folder = "Merged" if use_merged else f"Cam{camera}"
        logger.info(f"[Transform] Starting job {job_id} for {cam_folder}")

        job_manager.update_job(job_id, status="running")

        # Build camera transforms dict for this job
        camera_transforms = {camera: operations}

        processor = TransformProcessor(
            base_dir=base_dir,
            camera_transforms=camera_transforms,
            type_name=type_name,
            config=config,
            use_merged=use_merged,
        )

        def progress_callback(info):
            job_manager.update_job(
                job_id,
                progress=info.get("progress", 0),
            )

        result = processor.process_all_cameras(progress_callback)

        if result["success"]:
            job_manager.complete_job(
                job_id,
                camera_results={str(k): v for k, v in result["camera_results"].items()},
                statistics_warning="Statistics files were NOT transformed.",
            )
            logger.info(f"[Transform] Job {job_id} completed for {cam_folder}")
        else:
            job_manager.fail_job(job_id, "Transform failed")
            logger.error(f"[Transform] Job {job_id} failed")

    except Exception as e:
        logger.error(f"[Transform] Job {job_id} error: {e}", exc_info=True)
        job_manager.fail_job(job_id, str(e))


@transform_bp.route("/transform/batch_status/<job_id>", methods=["GET"])
def get_transform_batch_status(job_id):
    """Get batch transform job status with aggregated sub-job info."""
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
