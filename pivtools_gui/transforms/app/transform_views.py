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
# Constraints Endpoint
# =============================================================================


@transform_bp.route("/transform/constraints", methods=["GET"])
def get_transform_constraints():
    """
    Return constraints for transform operations.

    Used by frontend to show allowed source endpoints.
    Transforms have no constraints - all data types can be transformed.

    Returns:
        JSON with allowed_source_endpoints, is_stereo
    """
    cfg = get_config()

    return jsonify({
        "allowed_source_endpoints": cfg.get_allowed_endpoints("transforms"),
        "is_stereo": cfg.is_stereo_setup,
        # No blocking constraints for transforms - all endpoints allowed
    })


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
# Apply Transforms (Batch)
# =============================================================================


@transform_bp.route("/transform/apply_batch", methods=["POST"])
def apply_transforms_batch():
    """
    Apply all pending transformations.

    Simplified API:
        base_path_idx: int - Single path index (default: 0)
        process_merged: bool - If true: merged only; if false: all cameras with pending transforms
        process_stereo: bool - If true: process stereo data; if false: per-camera
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

        base_path_idx = int(data.get("base_path_idx", 0))
        process_merged = bool(data.get("process_merged", False))
        process_stereo = bool(data.get("process_stereo", False))
        type_name = data.get("type_name", config.transforms_type_name)

        # Validate path index
        if base_path_idx < 0 or base_path_idx >= len(base_paths):
            return jsonify({"error": f"Invalid base_path_idx: {base_path_idx}"}), 400

        base_dir = base_paths[base_path_idx]

        # Get cameras with pending transforms
        all_transforms = config.transforms_cameras
        camera_transforms = {c: ops for c, ops in all_transforms.items() if ops}

        if not camera_transforms:
            return jsonify({
                "success": False,
                "error": "No pending transformations to apply"
            }), 400

        # Build targets based on process_merged/process_stereo flags
        targets = []
        stereo_camera_pair = None
        if process_stereo:
            # Process stereo data - use stereo camera pair from config
            stereo_pairs = config.stereo_pairs
            if stereo_pairs:
                stereo_camera_pair = stereo_pairs[0]
            else:
                stereo_camera_pair = (1, 2)
            cam_transforms = list(camera_transforms.values())[0]
            targets.append({
                "camera": stereo_camera_pair[0],
                "is_merged": False,
                "is_stereo": True,
                "stereo_camera_pair": stereo_camera_pair,
                "label": f"Stereo Cam{stereo_camera_pair[0]}_Cam{stereo_camera_pair[1]}",
                "operations": cam_transforms,
            })
        elif process_merged:
            # Process merged data only - use first camera's transforms
            cam_transforms = list(camera_transforms.values())[0]
            targets.append({
                "camera": None,
                "is_merged": True,
                "is_stereo": False,
                "label": "Merged",
                "operations": cam_transforms,
            })
        else:
            # Process all cameras with pending transforms
            for cam_num, cam_transforms in camera_transforms.items():
                targets.append({
                    "camera": cam_num,
                    "is_merged": False,
                    "is_stereo": False,
                    "label": f"Cam{cam_num}",
                    "operations": cam_transforms,
                })

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
            use_merged = target["is_merged"]
            use_stereo = target.get("is_stereo", False)
            target_stereo_pair = target.get("stereo_camera_pair")
            cam_num = target["camera"] if target["camera"] else 1
            cam_transforms = target["operations"]

            # Determine job type label
            if use_stereo:
                job_type = f"stereo_{target_stereo_pair[0]}_{target_stereo_pair[1]}"
            elif use_merged:
                job_type = "merged"
            else:
                job_type = f"camera_{cam_num}"

            # Create sub-job
            job_id = job_manager.create_job(
                "transform",
                camera=target["label"],
                path_idx=base_path_idx,
                parent_job_id=parent_job_id,
                operations=cam_transforms,
            )
            sub_jobs.append({
                "job_id": job_id,
                "type": job_type,
                "path_idx": base_path_idx,
                "label": target["label"],
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
                    use_stereo,
                    target_stereo_pair,
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
    use_stereo: bool,
    stereo_camera_pair,
    config,
):
    """Run transform job in a background thread."""
    try:
        if use_stereo and stereo_camera_pair:
            cam_folder = f"Stereo Cam{stereo_camera_pair[0]}_Cam{stereo_camera_pair[1]}"
        elif use_merged:
            cam_folder = "Merged"
        else:
            cam_folder = f"Cam{camera}"
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
            use_stereo=use_stereo,
            stereo_camera_pair=stereo_camera_pair,
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
