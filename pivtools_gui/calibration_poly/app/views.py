import glob
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import cv2
import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger


from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from ...plotting.app.views import extract_coordinates
from ...utils import camera_number, numpy_to_png_base64
from ..davis_polynomial_calibration import read_calibration_xml, PolynomialVectorCalibrator, convert_davis_coeffs_to_array


def cache_key(source_path_idx, camera):
    return (int(source_path_idx), str(camera))


calibration_cache = {}
calibration_poly_bp = Blueprint("calibration_poly", __name__)


# Global job tracking
calibration_jobs = {}
vector_jobs = {}
scale_factor_jobs = {}


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


@calibration_poly_bp.route("/calibration_poly/polynomial_calibrate_vectors", methods=["POST"], strict_slashes=False)
def run_polynomial_calibration():
    """
    Run polynomial calibration on vector files.
    Expects JSON payload with:
    - source_path_idx
    - camera
    - dt
    - mm_per_pixel
    - dx_coeff (dict or list)
    - dy_coeff (dict or list)
    - x_origin, y_origin, nx, ny
    - num_images (optional)
    """
    data = request.get_json() or {}
    
    try:
        source_path_idx = int(data.get("source_path_idx", 0))
        camera = int(data.get("camera", 1))
        dt = float(data.get("dt", 1.0))
        mm_per_pixel = float(data.get("mm_per_pixel", 1.0))
        
        # Helper to parse coefficients
        def parse_coeffs(coeffs_input):
            arr = np.zeros(10, dtype=float)
            if isinstance(coeffs_input, list):
                for i, v in enumerate(coeffs_input):
                    if i < 10: arr[i] = float(v)
            elif isinstance(coeffs_input, dict):
                for k, v in coeffs_input.items():
                    # Extract index from keys like "A0", "0", etc.
                    idx_str = ''.join(filter(str.isdigit, str(k)))
                    if idx_str:
                        idx = int(idx_str)
                        if 0 <= idx < 10:
                            arr[idx] = float(v)
            return arr

        dx_coeff = parse_coeffs(data.get("dx_coeff", {}))
        dy_coeff = parse_coeffs(data.get("dy_coeff", {}))
        
        x_origin = float(data.get("x_origin", 0))
        y_origin = float(data.get("y_origin", 0))
        nx = float(data.get("nx", 1))
        ny = float(data.get("ny", 1))
        
        # If normalization factors are default (1), try to load from XML
        if nx == 1 and ny == 1:
            try:
                logger.info(f"nx/ny not provided or 1. Attempting to load from XML for source_path_idx {source_path_idx}")
                calib_data = read_calibration_xml(source_path_idx)
                
                # Find camera data
                cam_key = None
                if "cameras" in calib_data:
                    for key in calib_data["cameras"]:
                        if str(camera) in key:
                            cam_key = key
                            break
                
                if cam_key:
                    cam_params = calib_data["cameras"][cam_key]
                    
                    # Update Origin if default
                    if x_origin == 0 and y_origin == 0:
                        origin_dict = cam_params.get("origin", {})
                        x_origin = origin_dict.get("s_o", origin_dict.get("x", origin_dict.get("X", 0.0)))
                        y_origin = origin_dict.get("t_o", origin_dict.get("y", origin_dict.get("Y", 0.0)))
                        logger.info(f"Loaded origin from XML: ({x_origin}, {y_origin})")

                    # Update Normalisation
                    norm_dict = cam_params.get("normalisation", {})
                    nx = norm_dict.get("nx", norm_dict.get("x", norm_dict.get("X", 1.0)))
                    ny = norm_dict.get("ny", norm_dict.get("y", norm_dict.get("Y", 1.0)))
                    logger.info(f"Loaded normalization from XML: nx={nx}, ny={ny}")
                    
                    # Update mm_per_pixel if default
                    if mm_per_pixel == 1.0:
                        mm_per_pixel = cam_params.get("mm_per_pixel", 1.0)
                        logger.info(f"Loaded mm_per_pixel from XML: {mm_per_pixel}")
                        
                    # Update coefficients if empty
                    if np.all(dx_coeff == 0) and np.all(dy_coeff == 0):
                        dx_coeff = convert_davis_coeffs_to_array(cam_params.get("coefficients_a", {}))
                        dy_coeff = convert_davis_coeffs_to_array(cam_params.get("coefficients_b", {}))
                        logger.info("Loaded coefficients from XML")

            except Exception as e:
                logger.warning(f"Failed to load parameters from XML: {e}")

        vector_pattern = data.get("vector_format", "%05d.mat")
        type_name = data.get("type_name", "instantaneous")
        # run = int(data.get("run", 1)) # No longer needed as we process all runs

        # Resolve paths
        cfg = get_config()
        if not hasattr(cfg, "source_paths") or source_path_idx >= len(cfg.source_paths):
             return jsonify({"error": "Invalid source_path_idx"}), 400
             
        base_dir = Path(cfg.base_paths[source_path_idx])
        
        # Get uncalibrated directory
        paths = get_data_paths(
            base_dir,
            num_images=cfg.num_images,
            cam=camera,
            type_name=type_name,
            use_uncalibrated=True
        )
        uncalib_dir = paths["data_dir"]
        
        if not uncalib_dir.exists():
             logger.error(f"Uncalibrated data directory not found: {uncalib_dir}")
             return jsonify({"error": f"Uncalibrated data directory not found: {uncalib_dir}"}), 404

        # Determine number of images
        num_images = int(data.get("num_images", 0))
        if num_images == 0:
            # Simple glob count
            glob_pattern = vector_pattern.replace("%05d", "*").replace("%d", "*")
            files = list(uncalib_dir.glob(glob_pattern))
            num_images = len(files)
            
        if num_images == 0:
             logger.error("No vector files found to calibrate")
             return jsonify({"error": "No vector files found to calibrate"}), 404

        # Get output directory
        calib_paths = get_data_paths(
            base_dir,
            num_images=1,
            cam=camera,
            type_name=type_name
        )
        calib_dir = calib_paths["data_dir"]

        # Start background job
        job_id = str(uuid.uuid4())
        
        def run_job():
            try:
                calibration_jobs[job_id] = {
                    "status": "starting",
                    "progress": 0,
                    "processed_frames": 0,
                    "total_frames": num_images,
                    "start_time": time.time(),
                    "error": None,
                }

                def progress_callback(data):
                    calibration_jobs[job_id].update(
                        {
                            "status": "running",
                            "progress": data.get("progress", 0),
                            "processed_frames": data.get("processed_frames", 0),
                            "successful_frames": data.get("successful_frames", 0),
                        }
                    )

                calibrator = PolynomialVectorCalibrator(
                    base_dir=base_dir,
                    camera_num=camera,
                    dt=dt,
                    mm_per_pixel=mm_per_pixel,
                    dx_coeff=dx_coeff,
                    dy_coeff=dy_coeff,
                    x_origin=x_origin,
                    y_origin=y_origin,
                    nx=nx,
                    ny=ny,
                    vector_pattern=vector_pattern,
                    type_name=type_name
                )
                
                calibrator.process_run(num_images, progress_cb=progress_callback)
                
                calibration_jobs[job_id]["status"] = "completed"
                calibration_jobs[job_id]["progress"] = 100
                logger.info(f"Calibration job {job_id} completed successfully")

            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                calibration_jobs[job_id]["status"] = "failed"
                calibration_jobs[job_id]["error"] = str(e)

        thread = threading.Thread(target=run_job)
        thread.daemon = True
        thread.start()
        calibration_jobs[job_id] = {"status": "starting"}
        
        return jsonify({
            "status": "started", 
            "job_id": job_id,
            "message": f"Started calibration for {num_images} images"
        })

    except Exception as e:
        logger.error(f"Error starting calibration: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_poly_bp.route("/calibration_poly/status/<job_id>", methods=["GET"])
def calibration_poly_status(job_id):
    """Get polynomial calibration job status"""
    if job_id not in calibration_jobs:
        return jsonify({"error": "Job not found"}), 404

    job_data = calibration_jobs[job_id].copy()

    # Add timing info
    if "start_time" in job_data:
        elapsed = time.time() - job_data["start_time"]
        job_data["elapsed_time"] = elapsed

        if job_data["status"] == "running" and job_data.get("progress", 0) > 0:
            estimated_total = elapsed / (job_data["progress"] / 100.0)
            job_data["estimated_remaining"] = max(0, estimated_total - elapsed)

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
            num_images=getattr(cfg, "num_images", 1),
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

        # Use extract_coordinates from plotting.app.views
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

