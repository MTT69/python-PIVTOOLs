"""
Polynomial Calibration Views.

Clean API for polynomial (DaVis XML) calibration:
- /calibration/polynomial/validate_xml - Validate Calibration.xml exists
- /calibration/polynomial/read_xml - Read and parse XML coefficients
- /calibration/polynomial/load_xml_to_config - Load XML data into config.yaml
- /calibration/polynomial/calibrate - Start calibration job for single camera
- /calibration/polynomial/calibrate_all - Start calibration job for all cameras
- /calibration/polynomial/job/<job_id> - Poll job status
- /calibration/polynomial/set_datum - Set coordinate datum/origin
"""

import threading
from pathlib import Path

import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config, reload_config
from pivtools_core.paths import get_data_paths
from pivtools_core.coordinate_utils import extract_coordinates
from pivtools_core.batch_utils import iter_batch_targets
from pivtools_gui.utils import camera_number
from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
    read_calibration_xml,
    PolynomialVectorCalibrator,
)
from pivtools_gui.calibration.services.job_manager import job_manager


polynomial_bp = Blueprint("polynomial", __name__)


# ============================================================================
# ROUTE 1: Validate XML Exists
# ============================================================================


@polynomial_bp.route("/calibration/polynomial/validate_xml", methods=["POST"])
def polynomial_validate_xml():
    """
    Validate that Calibration.xml exists at the specified path.

    Request JSON:
        source_path_idx: int

    XML path is derived from config (calibration.polynomial.xml_path or source_path/Calibration.xml).

    Returns:
        JSON with valid, xml_path, cameras, error
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))

    try:
        cfg = get_config()
        poly_cfg = cfg.polynomial_calibration

        # Get XML path from config or derive from source path
        xml_path_config = poly_cfg.get("xml_path", "")

        if xml_path_config:
            xml_path = Path(xml_path_config)
        else:
            # Fallback to derived path
            source_root = Path(cfg.source_paths[source_path_idx])
            calib_subfolder = cfg.calibration_subfolder
            if calib_subfolder:
                xml_path = source_root / calib_subfolder / "Calibration.xml"
            else:
                xml_path = source_root / "Calibration.xml"

        if not xml_path.exists():
            return jsonify({
                "valid": False,
                "xml_path": None,
                "error": f"Calibration.xml not found at {xml_path}",
                "searched_path": str(xml_path)
            })

        # Try to parse and extract camera info
        result = read_calibration_xml(
            source_path_idx=source_path_idx,
            xml_path=str(xml_path),
            config=cfg,
        )
        cameras = list(result.get("cameras", {}).keys())

        return jsonify({
            "valid": True,
            "xml_path": str(xml_path),
            "cameras": cameras,
            "camera_count": len(cameras)
        })

    except Exception as e:
        logger.error(f"Error validating XML: {e}")
        return jsonify({
            "valid": False,
            "xml_path": None,
            "error": f"Failed to parse XML: {str(e)}"
        })


# ============================================================================
# ROUTE 2: Read XML Coefficients
# ============================================================================


@polynomial_bp.route("/calibration/polynomial/read_xml", methods=["POST", "GET"])
def polynomial_read_xml():
    """
    Read Calibration.xml and extract polynomial coefficients for all cameras.

    Request JSON:
        source_path_idx: int

    XML path read from config.

    Returns:
        JSON with cameras, coefficients, status
    """
    logger.debug(f"Accessed polynomial_read_xml with method: {request.method}")

    if request.method == "GET":
        return jsonify({
            "status": "ready",
            "message": "Endpoint reachable. Send POST with source_path_idx."
        })

    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))

    try:
        cfg = get_config()
        poly_cfg = cfg.polynomial_calibration
        xml_path = poly_cfg.get("xml_path", "")

        result = read_calibration_xml(
            source_path_idx=source_path_idx,
            xml_path=xml_path if xml_path else None,
            config=cfg,
        )
        return jsonify(result)

    except ValueError as e:
        logger.error(f"ValueError in polynomial_read_xml: {e}")
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError in polynomial_read_xml: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error reading Calibration.xml: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROUTE 3: Load XML to Config
# ============================================================================


@polynomial_bp.route("/calibration/polynomial/load_xml_to_config", methods=["POST"])
def polynomial_load_xml_to_config():
    """
    Load Calibration.xml and save camera params to config.yaml.

    Request JSON:
        source_path_idx: int

    XML path read from config.

    Returns:
        JSON with status and list of updated cameras
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))

    try:
        cfg = get_config()
        poly_cfg = cfg.polynomial_calibration
        xml_path = poly_cfg.get("xml_path", "")

        # Read XML
        result = read_calibration_xml(
            source_path_idx=source_path_idx,
            xml_path=xml_path if xml_path else None,
            config=cfg,
        )

        if result.get("status") != "success":
            return jsonify({"error": "Failed to parse XML"}), 400

        # Ensure polynomial section exists
        if "polynomial" not in cfg.data.get("calibration", {}):
            cfg.data.setdefault("calibration", {})["polynomial"] = {}
        if "cameras" not in cfg.data["calibration"]["polynomial"]:
            cfg.data["calibration"]["polynomial"]["cameras"] = {}

        cameras_updated = []

        for cam_key, cam_data in result.get("cameras", {}).items():
            # Extract camera number from key (e.g., "Camera1" -> 1)
            import re
            match = re.search(r'\d+', cam_key)
            if not match:
                continue
            cam_num = int(match.group())

            # Extract origin
            origin_data = cam_data.get("origin", {})
            origin = {
                "x": origin_data.get("s_o", origin_data.get("x", 0.0)),
                "y": origin_data.get("t_o", origin_data.get("y", 0.0)),
            }

            # Extract normalisation
            norm_data = cam_data.get("normalisation", {})
            normalisation = {
                "nx": norm_data.get("nx", 512.0),
                "ny": norm_data.get("ny", 384.0),
            }

            # Extract mm_per_pixel
            mm_per_pixel = cam_data.get("mm_per_pixel", 0.0)

            # Convert coefficients from dict to array
            coefficients_x = [0.0] * 10
            coefficients_y = [0.0] * 10

            term_mapping = {
                'o': 0, 's': 1, 's2': 2, 's3': 3,
                't': 4, 't2': 5, 't3': 6,
                'st': 7, 's2t': 8, 'st2': 9
            }

            coeffs_a = cam_data.get("coefficients_a", {})
            for key, val in coeffs_a.items():
                suffix = key.split('_', 1)[1] if '_' in key else key
                if suffix in term_mapping:
                    coefficients_x[term_mapping[suffix]] = float(val)

            coeffs_b = cam_data.get("coefficients_b", {})
            for key, val in coeffs_b.items():
                suffix = key.split('_', 1)[1] if '_' in key else key
                if suffix in term_mapping:
                    coefficients_y[term_mapping[suffix]] = float(val)

            # Save to config
            cfg.data["calibration"]["polynomial"]["cameras"][cam_num] = {
                "origin": origin,
                "normalisation": normalisation,
                "mm_per_pixel": mm_per_pixel,
                "coefficients_x": coefficients_x,
                "coefficients_y": coefficients_y,
            }
            cameras_updated.append(cam_num)

        cfg.save()
        logger.info(f"Loaded XML data to config for cameras: {cameras_updated}")

        return jsonify({
            "status": "success",
            "cameras_updated": cameras_updated,
            "xml_file": result.get("file", "")
        })

    except FileNotFoundError as e:
        logger.error(f"XML file not found: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error loading XML to config: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROUTE 4: Calibrate Single Camera
# ============================================================================


@polynomial_bp.route("/calibration/polynomial/calibrate", methods=["POST"])
def polynomial_calibrate():
    """
    Run polynomial calibration on vector files for a single camera.

    Request JSON:
        source_path_idx: int
        camera: int

    All calibration parameters (dt, coefficients, num_frame_pairs) read from config.

    Returns:
        JSON with job_id, status
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))

    try:
        # Get config
        cfg = reload_config()
        poly_cfg = cfg.polynomial_calibration

        if not hasattr(cfg, "source_paths") or source_path_idx >= len(cfg.source_paths):
            return jsonify({"error": "Invalid source_path_idx"}), 400

        base_dir = Path(cfg.base_paths[source_path_idx])
        num_frame_pairs = cfg.num_frame_pairs
        type_name = poly_cfg.get("piv_type", "instantaneous")
        dt = poly_cfg.get("dt", cfg.dt if hasattr(cfg, "dt") else 1.0)

        # Get uncalibrated directory to verify it exists
        paths = get_data_paths(
            base_dir,
            num_frame_pairs=num_frame_pairs,
            cam=camera,
            type_name=type_name,
            use_uncalibrated=True
        )
        uncalib_dir = paths["data_dir"]

        if not uncalib_dir.exists():
            logger.error(f"Uncalibrated data directory not found: {uncalib_dir}")
            return jsonify({"error": f"Uncalibrated data directory not found: {uncalib_dir}"}), 404

        # Get vector format from config
        vec_fmt = cfg.vector_format
        if isinstance(vec_fmt, list):
            vec_fmt = vec_fmt[0]

        # Create job using job_manager
        job_id = job_manager.create_job(
            "polynomial",
            camera=camera,
            progress=0,
            processed_frames=0,
            total_frames=num_frame_pairs,
        )

        def run_job():
            try:
                job_manager.update_job(job_id, status="running")

                def progress_callback(prog_data):
                    job_manager.update_job(
                        job_id,
                        progress=prog_data.get("progress", 0),
                        processed_frames=prog_data.get("processed_frames", 0),
                        successful_frames=prog_data.get("successful_frames", 0),
                    )

                # Create calibrator - reads parameters from config
                calibrator = PolynomialVectorCalibrator(
                    base_dir=base_dir,
                    camera_num=camera,
                    dt=dt,
                    vector_pattern=vec_fmt,
                    type_name=type_name,
                    config=cfg,
                )

                # Run calibration
                result = calibrator.process_vectors(progress_callback=progress_callback)

                if result.get("success"):
                    job_manager.complete_job(
                        job_id,
                        processed_frames=result.get("processed_frames", 0),
                        successful_frames=result.get("successful_frames", 0),
                    )
                    logger.info(f"Calibration job {job_id} completed successfully")
                else:
                    job_manager.fail_job(job_id, result.get("error", "Unknown error"))

            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                job_manager.fail_job(job_id, str(e))

        thread = threading.Thread(target=run_job)
        thread.daemon = True
        thread.start()

        return jsonify({
            "status": "started",
            "job_id": job_id,
            "message": f"Started calibration for camera {camera} (up to {num_frame_pairs} files)"
        })

    except Exception as e:
        logger.error(f"Error starting calibration: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROUTE 5: Calibrate All Cameras
# ============================================================================


@polynomial_bp.route("/calibration/polynomial/calibrate_all", methods=["POST"])
def polynomial_calibrate_all():
    """
    Run polynomial calibration for ALL cameras.

    Request JSON:
        source_path_idx: int

    All parameters (dt, num_frame_pairs, use_xml, xml_path) read from config.

    Returns:
        JSON with job_id, status, cameras
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))

    try:
        # Reload config to get latest settings
        cfg = reload_config()
        camera_numbers = cfg.camera_numbers
        base_root = Path(cfg.base_paths[source_path_idx])
        num_frame_pairs = cfg.num_frame_pairs

        # Read polynomial config
        poly_cfg = cfg.polynomial_calibration
        dt = poly_cfg.get("dt", cfg.dt if hasattr(cfg, "dt") else 1.0)
        use_xml = poly_cfg.get("use_xml", True)
        xml_path = poly_cfg.get("xml_path", "")
        type_name = poly_cfg.get("piv_type", "instantaneous")

        logger.info(f"use_xml: {use_xml}, xml_path: {xml_path}")

        xml_result = None
        if use_xml:
            # Read XML for coefficients
            try:
                xml_result = read_calibration_xml(
                    source_path_idx=source_path_idx,
                    xml_path=xml_path if xml_path else None,
                    config=cfg
                )
            except FileNotFoundError as e:
                return jsonify({"error": str(e)}), 404

            if xml_result.get("status") != "success":
                return jsonify({"error": "Failed to read calibration XML"}), 400

        # Log settings
        source_mode = "XML" if use_xml else "config"
        logger.info(
            f"Polynomial calibration starting for {len(camera_numbers)} cameras, "
            f"dt={dt}, num_frame_pairs={num_frame_pairs}, source={source_mode}"
        )

        # Create job using job_manager
        job_id = job_manager.create_job(
            "polynomial_all",
            processed_cameras=0,
            total_cameras=len(camera_numbers),
            current_camera=None,
            camera_results={},
        )

        def run_calibration():
            try:
                job_manager.update_job(job_id, status="running")

                # Get vector format from config
                vec_fmt = cfg.vector_format
                if isinstance(vec_fmt, list):
                    vec_fmt = vec_fmt[0]

                def progress_callback(progress_data):
                    job_manager.update_job(
                        job_id,
                        current_camera=progress_data.get("current_camera"),
                        processed_cameras=progress_data.get("processed_cameras", 0),
                        progress=progress_data.get("overall_progress", 0),
                    )

                if use_xml and xml_result:
                    # Use XML data for calibration
                    result = PolynomialVectorCalibrator.process_all_cameras(
                        base_dir=base_root,
                        cameras=camera_numbers,
                        xml_data=xml_result,
                        dt=dt,
                        vector_pattern=vec_fmt,
                        type_name=type_name,
                        config=cfg,
                        progress_callback=progress_callback,
                    )
                else:
                    # Use config values directly (no XML)
                    overall_result = {
                        "total_cameras": len(camera_numbers),
                        "processed_cameras": 0,
                        "successful_files": 0,
                        "failed_files": 0,
                        "camera_results": {},
                    }

                    for idx, cam_num in enumerate(camera_numbers):
                        logger.info(f"Processing camera {cam_num} ({idx + 1}/{len(camera_numbers)})")

                        # Create calibrator - will read params from config
                        calibrator = PolynomialVectorCalibrator(
                            base_dir=base_root,
                            camera_num=cam_num,
                            dt=dt,
                            vector_pattern=vec_fmt,
                            type_name=type_name,
                            config=cfg,
                        )

                        def camera_progress(data):
                            progress_callback({
                                "current_camera": cam_num,
                                "processed_cameras": idx,
                                "total_cameras": len(camera_numbers),
                                "overall_progress": int(((idx + data.get("progress", 0) / 100) / len(camera_numbers)) * 100),
                            })

                        try:
                            cam_result = calibrator.process_vectors(progress_callback=camera_progress)
                            overall_result["camera_results"][cam_num] = {
                                "status": "completed",
                                "processed_frames": cam_result.get("processed_frames", 0),
                                "successful_frames": cam_result.get("successful_frames", 0),
                            }
                            overall_result["successful_files"] += cam_result.get("successful_frames", 0)
                        except Exception as e:
                            logger.error(f"Camera {cam_num} failed: {e}")
                            overall_result["camera_results"][cam_num] = {
                                "status": "failed",
                                "error": str(e),
                            }
                            overall_result["failed_files"] += 1

                        overall_result["processed_cameras"] = idx + 1

                    result = overall_result

                job_manager.complete_job(
                    job_id,
                    camera_results=result.get("camera_results", {}),
                    successful_files=result.get("successful_files", 0),
                    failed_files=result.get("failed_files", 0),
                    current_camera=None,
                )

                logger.info(
                    f"Polynomial calibration completed. "
                    f"Processed {result['processed_cameras']} cameras"
                )

            except Exception as e:
                logger.error(f"Polynomial calibration job {job_id} failed: {e}")
                job_manager.fail_job(job_id, str(e))

        thread = threading.Thread(target=run_calibration)
        thread.daemon = True
        thread.start()

        return jsonify({
            "job_id": job_id,
            "status": "starting",
            "message": f"Polynomial calibration job started for {len(camera_numbers)} camera(s): {camera_numbers}",
            "cameras": camera_numbers,
            "num_frame_pairs": num_frame_pairs,
        })

    except Exception as e:
        logger.error(f"Error starting polynomial calibration: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROUTE 6: Job Status
# ============================================================================


@polynomial_bp.route("/calibration/polynomial/job/<job_id>", methods=["GET"])
def polynomial_job_status(job_id):
    """Get polynomial calibration job status."""
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job_data)


# ============================================================================
# ROUTE 7: Set Datum
# ============================================================================


@polynomial_bp.route("/calibration/polynomial/set_datum", methods=["POST"])
def polynomial_set_datum():
    """
    Set a new datum (origin) for the coordinates of a given run, and/or apply offsets.

    Request JSON:
        source_path_idx: int (or base_path_idx)
        camera: int
        run: int
        x: float (optional - datum x)
        y: float (optional - datum y)
        x_offset: float (optional)
        y_offset: float (optional)

    type_name read from config.
    """
    data = request.get_json() or {}
    base_path_idx = int(data.get("base_path_idx", data.get("source_path_idx", 0)))
    camera = camera_number(data.get("camera", 1))
    run = int(data.get("run", 1))
    x0 = data.get("x")
    y0 = data.get("y")
    x_offset = data.get("x_offset", 0)
    y_offset = data.get("y_offset", 0)

    logger.debug("updating datum for run %d", run)

    try:
        cfg = get_config()
        poly_cfg = cfg.polynomial_calibration
        type_name = poly_cfg.get("piv_type", "instantaneous")

        # Accept both base_paths and source_paths for compatibility
        source_root = Path(
            getattr(cfg, "base_paths", getattr(cfg, "source_paths", []))[base_path_idx]
        )
        paths = get_data_paths(
            base_dir=source_root,
            num_frame_pairs=cfg.num_frame_pairs,
            cam=camera,
            type_name=type_name,
            calibration=False,
        )
        data_dir = paths["data_dir"]
        coords_path = data_dir / "coordinates.mat"

        if not coords_path.exists():
            return jsonify({"error": f"Coordinates file not found: {coords_path}"}), 404

        mat = scipy.io.loadmat(coords_path, struct_as_record=False, squeeze_me=True)
        if "coordinates" not in mat:
            return jsonify({"error": "Variable 'coordinates' not found in coords mat"}), 400

        coordinates = mat["coordinates"]
        run_idx = run - 1

        # Use extract_coordinates from pivtools_core.coordinate_utils
        cx, cy = extract_coordinates(coordinates, run)

        # Only apply datum shift if x/y are provided (not None)
        if x0 is not None and y0 is not None:
            x0 = float(x0)
            y0 = float(y0)
            cx = cx - x0
            cy = cy - y0

        # Always apply offsets if present
        if x_offset is not None and y_offset is not None:
            x_offset = float(x_offset)
            y_offset = float(y_offset)
            cx = cx + x_offset
            cy = cy + y_offset

        # Convert to proper MATLAB struct format
        num_runs = len(coordinates) if hasattr(coordinates, '__len__') else 1
        if num_runs == 1 and not hasattr(coordinates, '__len__'):
            num_runs = 1
            coordinates = [coordinates]

        dtype = [('x', object), ('y', object)]
        coords_struct = np.empty((num_runs,), dtype=dtype)

        # Copy all existing coordinates
        for i in range(num_runs):
            if i == run_idx:
                # Use modified coordinates for this run
                coords_struct['x'][i] = cx
                coords_struct['y'][i] = cy
            else:
                # Copy existing coordinates
                existing_x, existing_y = extract_coordinates(coordinates, i + 1)
                coords_struct['x'][i] = existing_x
                coords_struct['y'][i] = existing_y

        scipy.io.savemat(coords_path, {"coordinates": coords_struct}, do_compression=True)
        return jsonify({"status": "ok", "run": run, "shape": [cx.shape, cy.shape]})

    except Exception as e:
        logger.error(f"[set_datum] ERROR: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROUTE 8: Calibrate Batch (Multi-Path + Multi-Camera)
# ============================================================================


@polynomial_bp.route("/calibration/polynomial/calibrate_batch", methods=["POST"])
def polynomial_calibrate_batch():
    """
    Run polynomial calibration with batch processing support.

    Request JSON:
        active_paths: list of path indices (default: from config)
        cameras: list of camera numbers (default: from config)

    All parameters (dt, coefficients, use_xml) read from config.

    Returns:
        JSON with parent_job_id, sub_jobs list, status
    """
    data = request.get_json() or {}
    logger.info(f"Received batch polynomial calibration request: {data}")

    try:
        cfg = reload_config()
        base_paths = cfg.base_paths
        source_paths = cfg.source_paths
        poly_cfg = cfg.polynomial_calibration

        # Get batch parameters
        active_paths = data.get("active_paths")
        if active_paths is None:
            active_paths = cfg.calibration_active_paths

        cameras = data.get("cameras")
        if cameras is None:
            cameras = cfg.calibration_cameras

        # Validate paths
        valid_paths = [i for i in active_paths if 0 <= i < len(base_paths)]
        if not valid_paths:
            return jsonify({"error": "No valid path indices provided"}), 400

        if not cameras:
            return jsonify({"error": "No cameras specified"}), 400

        # Generate batch targets (no merged for calibration)
        targets = iter_batch_targets(
            base_paths=base_paths,
            active_paths=valid_paths,
            cameras=cameras,
            include_merged=False,
            source_paths=source_paths,
        )

        if not targets:
            return jsonify({"error": "No targets to process"}), 400

        # Read polynomial config
        dt = poly_cfg.get("dt", cfg.dt if hasattr(cfg, "dt") else 1.0)
        use_xml = poly_cfg.get("use_xml", True)
        xml_path = poly_cfg.get("xml_path", "")
        type_name = poly_cfg.get("piv_type", "instantaneous")

        # Get vector format
        vec_fmt = cfg.vector_format
        if isinstance(vec_fmt, list):
            vec_fmt = vec_fmt[0]

        # Create parent job
        parent_job_id = job_manager.create_job(
            "polynomial_batch",
            total_targets=len(targets),
        )
        sub_jobs = []

        # Launch a job for each target
        for target in targets:
            base_dir = target.base_path
            source_dir = target.source_path or base_dir
            cam_num = target.camera

            # Create sub-job
            job_id = job_manager.create_job(
                "polynomial",
                camera=cam_num,
                path_idx=target.path_idx,
                parent_job_id=parent_job_id,
                progress=0,
                processed_frames=0,
                total_frames=cfg.num_frame_pairs,
            )
            sub_jobs.append({
                "job_id": job_id,
                "camera": cam_num,
                "path_idx": target.path_idx,
                "label": target.label,
            })

            # Launch thread
            thread = threading.Thread(
                target=_run_polynomial_calibration_job,
                args=(
                    job_id,
                    base_dir,
                    source_dir,
                    cam_num,
                    target.path_idx,
                    dt,
                    use_xml,
                    xml_path,
                    type_name,
                    vec_fmt,
                    cfg,
                ),
            )
            thread.daemon = True
            thread.start()

        # Update parent job
        job_manager.update_job(parent_job_id, sub_jobs=sub_jobs, status="running")

        return jsonify({
            "parent_job_id": parent_job_id,
            "sub_jobs": sub_jobs,
            "total_targets": len(targets),
            "processed_targets": len(sub_jobs),
            "status": "starting",
            "message": f"Polynomial calibration started for {len(sub_jobs)} target(s) "
            f"across {len(valid_paths)} path(s)",
        })

    except Exception as e:
        logger.error(f"Error starting batch polynomial calibration: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _run_polynomial_calibration_job(
    job_id: str,
    base_dir: Path,
    source_dir: Path,
    camera: int,
    path_idx: int,
    dt: float,
    use_xml: bool,
    xml_path: str,
    type_name: str,
    vec_fmt: str,
    cfg,
):
    """Run polynomial calibration job in a background thread."""
    try:
        logger.info(f"[Polynomial] Starting job {job_id} for Cam{camera} at path {path_idx}")

        job_manager.update_job(job_id, status="running")

        def progress_callback(prog_data):
            job_manager.update_job(
                job_id,
                progress=prog_data.get("progress", 0),
                processed_frames=prog_data.get("processed_frames", 0),
                successful_frames=prog_data.get("successful_frames", 0),
            )

        # Read XML if needed
        xml_data = None
        if use_xml:
            try:
                xml_data = read_calibration_xml(
                    source_path_idx=path_idx,
                    xml_path=xml_path if xml_path else None,
                    config=cfg,
                )
            except Exception as e:
                logger.warning(f"Could not read XML for path {path_idx}: {e}")

        # Create calibrator
        calibrator = PolynomialVectorCalibrator(
            base_dir=base_dir,
            camera_num=camera,
            dt=dt,
            vector_pattern=vec_fmt,
            type_name=type_name,
            config=cfg,
        )

        # Run calibration
        result = calibrator.process_vectors(progress_callback=progress_callback)

        if result.get("success"):
            job_manager.complete_job(
                job_id,
                processed_frames=result.get("processed_frames", 0),
                successful_frames=result.get("successful_frames", 0),
            )
            logger.info(f"[Polynomial] Job {job_id} completed for Cam{camera}")
        else:
            job_manager.fail_job(job_id, result.get("error", "Calibration failed"))
            logger.error(f"[Polynomial] Job {job_id} failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"[Polynomial] Job {job_id} error: {e}", exc_info=True)
        job_manager.fail_job(job_id, str(e))


@polynomial_bp.route("/calibration/polynomial/batch_status/<job_id>", methods=["GET"])
def polynomial_batch_status(job_id: str):
    """Get batch polynomial calibration job status with aggregated sub-job info."""
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
                sub_status["camera"] = sub_job.get("camera")
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
