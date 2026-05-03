"""Flask routes for importing DaVis PinholeOpenCV calibration models."""

import threading
from pathlib import Path

from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config, reload_config
from pivtools_gui.calibration.services.job_manager import job_manager

davis_pinhole_bp = Blueprint("davis_pinhole", __name__)


@davis_pinhole_bp.route("/calibrate/davis_pinhole/import", methods=["POST"])
def import_davis_pinhole():
    """Import a DaVis Calibration.xml (PinholeOpenCV) into PIVTOOLs format.

    Request JSON:
        xml_path     (str)  Path to DaVis Calibration.xml
        dt           (float) Time step in seconds (default: from config)
        base_path_idx (int) Which base path to write models into (default: 0)
        camera_map   (dict) Optional {"1": 1, "2": 2, ...} davis_id -> cam_no
    """
    from pivtools_gui.calibration.davis_pinhole.davis_pinhole_import import (
        save_davis_pinhole_models,
    )

    data = request.get_json(force=True) or {}

    xml_path = data.get("xml_path")
    if not xml_path:
        return jsonify({"error": "xml_path is required"}), 400

    config = get_config()
    dt = float(data.get("dt") or config.dt)
    base_path_idx = int(data.get("base_path_idx", 0))

    if base_path_idx >= len(config.base_paths):
        return jsonify({"error": f"base_path_idx {base_path_idx} out of range"}), 400

    base_paths = [config.base_paths[base_path_idx]]

    raw_map = data.get("camera_map")
    camera_map = (
        {int(k): int(v) for k, v in raw_map.items()} if raw_map else None
    )

    result = save_davis_pinhole_models(xml_path, base_paths, dt, camera_map)

    if result["errors"]:
        return jsonify({"error": "; ".join(result["errors"])}), 500

    # Update config to use davis_pinhole method
    config.data.setdefault("calibration", {})["active"] = "davis_pinhole"
    config.save()
    reload_config()

    return jsonify({
        "success": True,
        "cameras": result["cameras"],
    })


@davis_pinhole_bp.route("/calibrate/davis_pinhole/calibrate_all", methods=["POST"])
def davis_pinhole_calibrate_all():
    """Apply saved DaVis pinhole model to all uncalibrated vector files.

    Request JSON:
        base_path_idx (int)  Which base path to read/write (default: 0)
        type_name     (str)  "instantaneous" or "ensemble" (default: "instantaneous")
    """
    from pivtools_gui.calibration.vector_calibration_production import VectorCalibrator

    data = request.get_json(force=True) or {}
    cfg = get_config()

    base_path_idx = int(data.get("base_path_idx", 0))
    if base_path_idx >= len(cfg.base_paths):
        return jsonify({"error": f"base_path_idx {base_path_idx} out of range"}), 400

    type_name = data.get("type_name", "instantaneous")
    base_dir = Path(cfg.base_paths[base_path_idx])
    camera_numbers = cfg.camera_numbers
    num_frame_pairs = cfg.num_frame_pairs
    dt = float(cfg.dt)
    vec_fmt = cfg.vector_format
    if isinstance(vec_fmt, list):
        vec_fmt = vec_fmt[0]

    job_id = job_manager.create_job(
        "davis_pinhole_calibrate",
        total_cameras=len(camera_numbers),
        processed_cameras=0,
        current_camera=None,
        camera_results={},
    )

    def run_calibration():
        try:
            job_manager.update_job(job_id, status="running")
            camera_results = {}

            for idx, cam_num in enumerate(camera_numbers):
                job_manager.update_job(
                    job_id,
                    current_camera=cam_num,
                    processed_cameras=idx,
                    progress=int(idx / len(camera_numbers) * 100),
                )
                try:
                    calibrator = VectorCalibrator(
                        base_dir=base_dir,
                        camera_num=cam_num,
                        model_type="davis_pinhole",
                        dt=dt,
                        vector_pattern=vec_fmt,
                        type_name=type_name,
                    )
                    calibrator.process_run(num_frame_pairs=num_frame_pairs)
                    camera_results[cam_num] = {"status": "completed"}
                except Exception as e:
                    logger.error(f"Camera {cam_num} failed: {e}")
                    camera_results[cam_num] = {"status": "failed", "error": str(e)}

            try:
                cfg.save_calibration_snapshot(base_dir)
            except Exception as snap_err:
                logger.warning(f"Snapshot save failed: {snap_err}")

            job_manager.complete_job(
                job_id,
                camera_results=camera_results,
                processed_cameras=len(camera_numbers),
                current_camera=None,
            )
        except Exception as e:
            logger.error(f"davis_pinhole calibrate_all job {job_id} failed: {e}")
            job_manager.fail_job(job_id, str(e))

    thread = threading.Thread(target=run_calibration)
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "starting",
        "cameras": camera_numbers,
        "num_frame_pairs": num_frame_pairs,
    })


@davis_pinhole_bp.route("/calibrate/davis_pinhole/job/<job_id>", methods=["GET"])
def davis_pinhole_job_status(job_id):
    """Poll calibration job status."""
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job_data)
