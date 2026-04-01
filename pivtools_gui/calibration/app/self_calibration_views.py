"""
Self-Calibration Views.

Routes for stereo self-calibration (Wieneke 2005):
- /calibrate/self_calibration/dewarp_preview  - Red-cyan overlay of source frame pair
- /calibrate/self_calibration/run             - Start self-cal job
- /calibrate/self_calibration/job/<job_id>    - Poll progress / get results
- /calibrate/self_calibration/status          - Current self-cal state from config
"""

import base64
import io
import math
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config, reload_config
from pivtools_gui.calibration.services.job_manager import job_manager
from pivtools_gui.calibration.services.self_calibration_service import (
    generate_dewarp_overlay,
    load_source_images,
    load_stereo_cameras,
    compute_stereo_world_bounds,
    result_to_dict,
    run_self_cal_job,
    save_self_cal_to_config,
)
from pivtools_gui.stereo_reconstruction.self_calibration import estimate_pixel_scale

self_calibration_bp = Blueprint("self_calibration", __name__)


# ============================================================================
# ROUTE 1: Dewarp Preview
# ============================================================================

@self_calibration_bp.route(
    "/calibrate/self_calibration/dewarp_preview", methods=["POST"]
)
def dewarp_preview():
    """Generate red-cyan dewarp overlay of one source frame pair.

    Request JSON:
        source_path_idx: int (default 0)
        cam1: int
        cam2: int
        method: str ("dotboard" or "charuco")
        frame_idx: int (1-based, default 1)
        sub_frame: str ("A" or "B", default "A")
        z_offset: float (optional, for corrected view)
        tilt_x: float (optional)
        tilt_y: float (optional)

    Returns JSON:
        overlay: base64 PNG (red-cyan composite)
        cam1_image: base64 PNG (dewarped cam1 grayscale)
        cam2_image: base64 PNG (dewarped cam2 grayscale)
        total_frames: int
        world_bounds: [xmin, xmax, ymin, ymax]
        mm_per_pixel: float
        image (deprecated alias for overlay)
    """
    try:
        data = request.get_json(force=True)
        config = get_config()

        source_path_idx = data.get("source_path_idx", 0)
        cam1_num = int(data["cam1"])
        cam2_num = int(data["cam2"])
        method = data.get("method", "dotboard")
        frame_idx = data.get("frame_idx", 1)
        sub_frame = data.get("sub_frame", "A").upper()
        z_offset = float(data.get("z_offset", 0.0))
        tilt_x = float(data.get("tilt_x", 0.0))
        tilt_y = float(data.get("tilt_y", 0.0))

        # Strip stereo_ prefix if present
        if method.startswith("stereo_"):
            method = method.replace("stereo_", "")

        base_dir = str(config.base_paths[0])

        # Load cameras
        cam1, cam2, md1, md2 = load_stereo_cameras(
            base_dir, cam1_num, cam2_num, method
        )

        # Compute bounds
        world_bounds = compute_stereo_world_bounds(md1, md2)
        mm_per_pixel = estimate_pixel_scale(cam1, cam2, world_bounds)

        # Load single frame pair
        from pivtools_core.image_handling.load_images import read_pair
        from pivtools_core.image_handling.path_utils import build_piv_camera_path

        # Validate source path exists
        if source_path_idx >= len(config.source_paths):
            return jsonify({"error": f"Source path index {source_path_idx} out of range "
                           f"(have {len(config.source_paths)} source paths configured)"}), 400
        source_path = config.source_paths[source_path_idx]
        if not Path(str(source_path)).exists():
            return jsonify({"error": f"Source path does not exist: {source_path}. "
                           f"Check Paths > Source in the Setup tab."}), 400

        cam1_path = build_piv_camera_path(config, source_path_idx, cam1_num)
        cam2_path = build_piv_camera_path(config, source_path_idx, cam2_num)

        # Select frame A or B from the pair
        sub_idx = 1 if sub_frame == "B" else 0

        pair1 = read_pair(frame_idx, cam1_path, cam1_num, config)
        img1 = pair1[sub_idx]
        if img1.ndim == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        pair2 = read_pair(frame_idx, cam2_path, cam2_num, config)
        img2 = pair2[sub_idx]
        if img2.ndim == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Generate overlay + individual dewarped images
        overlay, cam1_u8, cam2_u8 = generate_dewarp_overlay(
            cam1, cam2, img1, img2,
            world_bounds, mm_per_pixel,
            z_offset=z_offset, tilt_x=tilt_x, tilt_y=tilt_y,
        )

        # Encode all images to base64 PNG
        from PIL import Image

        def _encode_b64(arr):
            pil_img = Image.fromarray(arr)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG", compress_level=1)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        total_frames = config.num_frame_pairs

        return jsonify({
            "overlay": _encode_b64(overlay),
            "cam1_image": _encode_b64(cam1_u8),
            "cam2_image": _encode_b64(cam2_u8),
            "image": _encode_b64(overlay),  # backward compat
            "total_frames": total_frames,
            "world_bounds": list(world_bounds),
            "mm_per_pixel": mm_per_pixel,
        })

    except Exception as e:
        logger.error(f"Dewarp preview failed: {e}")
        return jsonify({"error": str(e)}), 400


# ============================================================================
# ROUTE 2: Run Self-Calibration
# ============================================================================

@self_calibration_bp.route(
    "/calibrate/self_calibration/run", methods=["POST"]
)
def run_self_calibration_route():
    """Start self-calibration job (daemon thread + job_manager).

    Request JSON:
        source_path_idx: int (default 0)
        cam1: int
        cam2: int
        method: str
        n_images: int (default 20)
        window_size: int (default 64)
        overlap: float (default 50.0)
        convergence_threshold: float (default 0.1)
        quality_threshold: float (default 0.3)
    """
    try:
        data = request.get_json(force=True)
        config = get_config()

        source_path_idx = data.get("source_path_idx", 0)
        cam1_num = int(data["cam1"])
        cam2_num = int(data["cam2"])
        method = data.get("method", "dotboard")
        if method.startswith("stereo_"):
            method = method.replace("stereo_", "")

        sc_cfg = config.self_calibration_config
        n_images = int(data.get("n_images", sc_cfg.get("n_images", 20)))
        window_size = int(data.get("window_size", sc_cfg.get("window_size", 64)))
        overlap = float(data.get("overlap", sc_cfg.get("overlap", 50.0)))
        convergence_threshold = float(data.get("convergence_threshold", 0.1))
        quality_threshold = float(data.get("quality_threshold", 0.3))

        base_dir = str(config.base_paths[0])

        job_id = job_manager.create_job(
            "self_calibration",
            cam1=cam1_num,
            cam2=cam2_num,
            method=method,
            n_images=n_images,
        )

        def run_job():
            try:
                job_manager.update_job(job_id, status="running", progress=0)

                def progress_cb(progress_data):
                    try:
                        job_manager.update_job(
                            job_id,
                            status=progress_data.get("status", "running"),
                            progress=progress_data.get("progress", 0),
                        )
                    except Exception:
                        pass

                result = run_self_cal_job(
                    config=config,
                    base_dir=base_dir,
                    source_path_idx=source_path_idx,
                    cam1_num=cam1_num,
                    cam2_num=cam2_num,
                    method=method,
                    n_images=n_images,
                    window_size=window_size,
                    overlap=overlap,
                    convergence_threshold=convergence_threshold,
                    quality_threshold=quality_threshold,
                    progress_callback=progress_cb,
                )

                # Save to config
                fresh_config = reload_config()
                save_self_cal_to_config(
                    fresh_config, result,
                    n_images=n_images,
                    window_size=window_size,
                    overlap=overlap,
                )

                # Complete job with serializable results
                job_manager.complete_job(
                    job_id,
                    result=result_to_dict(result),
                )

            except Exception as e:
                logger.error(f"Self-calibration job failed: {e}")
                job_manager.fail_job(job_id, str(e))

        thread = threading.Thread(target=run_job, daemon=True)
        thread.start()

        return jsonify({"job_id": job_id})

    except Exception as e:
        logger.error(f"Failed to start self-calibration: {e}")
        return jsonify({"error": str(e)}), 400


# ============================================================================
# ROUTE 3: Poll Job Status
# ============================================================================

@self_calibration_bp.route(
    "/calibrate/self_calibration/job/<job_id>", methods=["GET"]
)
def get_self_cal_job(job_id):
    """Poll self-calibration job progress.

    On completion, returns full convergence history + scalar results.
    """
    job = job_manager.get_job_with_timing(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ============================================================================
# ROUTE 4: Current Status from Config
# ============================================================================

@self_calibration_bp.route(
    "/calibrate/self_calibration/status", methods=["GET"]
)
def get_self_cal_status():
    """Return current self-cal state from config."""
    config = get_config()
    sc = config.self_calibration_config
    return jsonify({
        "has_self_calibration": config.has_self_calibration,
        "z_offset": config.self_calibration_z_offset,
        "tilt_x": config.self_calibration_tilt_x,
        "tilt_y": config.self_calibration_tilt_y,
        "tilt_x_deg": math.degrees(config.self_calibration_tilt_x),
        "tilt_y_deg": math.degrees(config.self_calibration_tilt_y),
        "converged": sc.get("converged", False),
        "n_iterations": sc.get("n_iterations", 0),
        "final_rms_disparity": sc.get("final_rms_disparity", 0.0),
        "n_images": sc.get("n_images", 20),
        "window_size": sc.get("window_size", 64),
        "overlap": sc.get("overlap", 50.0),
    })
