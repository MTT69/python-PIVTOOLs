import threading
from pathlib import Path
import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger


from pivtools_core.config import get_config, reload_config
from pivtools_core.paths import get_data_paths
from pivtools_core.coordinate_utils import extract_coordinates
from pivtools_gui.utils import camera_number
from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
    read_calibration_xml,
    PolynomialVectorCalibrator,
    convert_davis_coeffs_to_array,
)
from pivtools_gui.calibration.services.job_manager import job_manager


def cache_key(source_path_idx, camera):
    return (int(source_path_idx), str(camera))


calibration_cache = {}
calibration_poly_bp = Blueprint("calibration_poly", __name__)


# ============================================================================
# POLYNOMIAL CALIBRATION ROUTES
# ============================================================================


@calibration_poly_bp.route("/calibration_poly/read_xml", methods=["POST", "GET"], strict_slashes=False)
def read_polynomial_calibration_xml():
    """Read Calibration.xml and extract polynomial coefficients for all cameras"""
    logger.debug(f"Accessed read_polynomial_calibration_xml with method: {request.method}")

    if request.method == "GET":
        return jsonify({"status": "ready", "message": "Endpoint reachable. Send POST with source_path_idx."})

    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))

    try:
        result = read_calibration_xml(source_path_idx)
        return jsonify(result)

    except ValueError as e:
        logger.error(f"ValueError in read_polynomial_calibration_xml: {e}")
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError in read_polynomial_calibration_xml: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error reading Calibration.xml: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_poly_bp.route("/calibration_poly/validate_xml", methods=["POST"], strict_slashes=False)
def validate_polynomial_xml():
    """
    Validate that Calibration.xml exists in the calibration subfolder.

    Request JSON:
        source_path_idx: int

    Returns:
        JSON with valid, xml_path, cameras, error
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))

    try:
        cfg = get_config()
        source_root = Path(cfg.source_paths[source_path_idx])

        # XML is in the calibration subfolder (with calibration images)
        calib_subfolder = cfg.data.get("calibration", {}).get("subfolder", "")
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
        result = read_calibration_xml(source_path_idx)
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


@calibration_poly_bp.route("/calibration_poly/polynomial_calibrate_vectors", methods=["POST"], strict_slashes=False)
def run_polynomial_calibration():
    """
    Run polynomial calibration on vector files for a single camera.

    Expects JSON payload with:
    - source_path_idx: int
    - camera: int
    - type_name: str (optional, default "instantaneous")

    Note: Coefficients and parameters are read from config.
    num_frame_pairs is read from config for path construction and file iteration.
    """
    data = request.get_json() or {}

    try:
        source_path_idx = int(data.get("source_path_idx", 0))
        camera = int(data.get("camera", 1))
        type_name = data.get("type_name", "instantaneous")

        # Get config
        cfg = reload_config()
        if not hasattr(cfg, "source_paths") or source_path_idx >= len(cfg.source_paths):
            return jsonify({"error": "Invalid source_path_idx"}), 400

        base_dir = Path(cfg.base_paths[source_path_idx])
        num_frame_pairs = cfg.num_frame_pairs

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
                    vector_pattern=vec_fmt,
                    type_name=type_name,
                    config=cfg,
                )

                # Run calibration - reads num_frame_pairs from config
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


@calibration_poly_bp.route("/calibration_poly/status/<job_id>", methods=["GET"])
def calibration_poly_status(job_id):
    """Get polynomial calibration job status"""
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job_data)


@calibration_poly_bp.route("/calibration_poly/set_datum", methods=["POST"])
def calibration_poly_set_datum():
    """
    Set a new datum (origin) for the coordinates of a given run, and/or apply offsets.
    Expects JSON: source_path_idx, camera, run, x, y, x_offset, y_offset
    """
    data = request.get_json() or {}
    base_path_idx = int(data.get("base_path_idx", data.get("source_path_idx", 0)))
    camera = camera_number(data.get("camera", 1))
    run = int(data.get("run", 1))
    type_name = data.get("type_name", "instantaneous")
    x0 = data.get("x")
    y0 = data.get("y")
    x_offset = data.get("x_offset", 0)
    y_offset = data.get("y_offset", 0)
    logger.debug("updating datum for run %d", run)
    try:
        cfg = get_config()
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
            return (
                jsonify({"error": "Variable 'coordinates' not found in coords mat"}),
                400,
            )
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

        # Convert to proper MATLAB struct format (not cell array)
        # Create structured numpy array with dtype [('x', object), ('y', object)]
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


@calibration_poly_bp.route("/calibration_poly/polynomial_calibrate_vectors_all", methods=["POST"], strict_slashes=False)
def run_polynomial_calibration_all():
    """
    Run polynomial calibration for ALL cameras.

    Request JSON:
        source_path_idx: int
        type_name: str (optional)

    Note: dt and num_frame_pairs read from config, coefficients from XML
    """
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    type_name = data.get("type_name", "instantaneous")

    try:
        # Reload config to get latest settings
        cfg = reload_config()
        camera_numbers = cfg.camera_numbers
        base_root = Path(cfg.base_paths[source_path_idx])
        num_frame_pairs = cfg.num_frame_pairs

        # Read dt from config
        poly_cfg = cfg.polynomial_calibration
        dt = poly_cfg.get("dt", cfg.dt if hasattr(cfg, "dt") else 1.0)

        # Read XML for coefficients
        try:
            xml_result = read_calibration_xml(source_path_idx, config=cfg)
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404

        if xml_result.get("status") != "success":
            return jsonify({"error": "Failed to read calibration XML"}), 400

        # Log settings
        logger.info(
            f"Polynomial calibration starting for {len(camera_numbers)} cameras, "
            f"dt={dt}, num_frame_pairs={num_frame_pairs}"
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


@calibration_poly_bp.route("/calibration/polynomial/status/<job_id>", methods=["GET"])
def polynomial_job_status(job_id):
    """Get polynomial calibration job status (using job_manager)."""
    job_data = job_manager.get_job_with_timing(job_id)
    if job_data is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job_data)
