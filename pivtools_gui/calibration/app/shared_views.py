"""
Shared Calibration Views.

Provides utility endpoints used across calibration methods.
"""

from pathlib import Path

import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths

from ...plotting.app.views import extract_coordinates
from ...utils import camera_number

calibration_shared_bp = Blueprint("calibration_shared", __name__)


@calibration_shared_bp.route("/calibration/set_datum", methods=["POST"])
def calibration_set_datum():
    """
    Set a new datum (origin) for the coordinates of a given run, and/or apply offsets.

    Request JSON:
        base_path_idx or source_path_idx: int
        camera: int
        run: int
        type_name: str (default: "instantaneous")
        x: float (optional) - New x origin
        y: float (optional) - New y origin
        x_offset: float (optional) - X offset to apply
        y_offset: float (optional) - Y offset to apply

    Returns:
        JSON with status, run, shape
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

        # Use extract_coordinates from plotting.app.views
        cx, cy = extract_coordinates(coordinates, run)

        logger.debug(
            f"[set_datum] Run {run} - original first x,y: {cx.flat[0]}, {cy.flat[0]}"
        )
        logger.debug(
            f"[set_datum] Datum to set: x0={x0}, y0={y0}, x_offset={x_offset}, y_offset={y_offset}"
        )

        # Only apply datum shift if x/y are provided
        if x0 is not None and y0 is not None:
            x0 = float(x0)
            y0 = float(y0)
            cx = cx - x0
            cy = cy - y0
            logger.debug(
                f"[set_datum] After datum shift, first x,y: {cx.flat[0]}, {cy.flat[0]}"
            )

        # Always apply offsets if present
        if x_offset is not None and y_offset is not None:
            x_offset = float(x_offset)
            y_offset = float(y_offset)
            cx = cx + x_offset
            cy = cy + y_offset
            logger.debug(f"[set_datum] After offset, first x,y: {cx.flat[0]}, {cy.flat[0]}")

        # Convert to proper MATLAB struct format
        num_runs = len(coordinates) if hasattr(coordinates, "__len__") else 1
        if num_runs == 1 and not hasattr(coordinates, "__len__"):
            num_runs = 1
            coordinates = [coordinates]

        dtype = [("x", object), ("y", object)]
        coords_struct = np.empty((num_runs,), dtype=dtype)

        # Copy all existing coordinates
        for i in range(num_runs):
            if i == run_idx:
                coords_struct["x"][i] = cx
                coords_struct["y"][i] = cy
            else:
                existing_x, existing_y = extract_coordinates(coordinates, i + 1)
                coords_struct["x"][i] = existing_x
                coords_struct["y"][i] = existing_y

        scipy.io.savemat(
            coords_path, {"coordinates": coords_struct}, do_compression=True
        )
        return jsonify({"status": "ok", "run": run, "shape": [cx.shape, cy.shape]})

    except Exception as e:
        logger.error(f"[set_datum] ERROR: {e}")
        return jsonify({"error": str(e)}), 500


@calibration_shared_bp.route("/calibration/status", methods=["GET"])
def calibration_status():
    """
    Get calibration status - unified endpoint for all calibration types.

    Query params:
        source_path_idx: int
        camera: int
        type: str (optional) - Calibration type

    Returns:
        JSON with status info
    """
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))
    cal_type = request.args.get("type", None)

    # For now, return not_started for all status requests
    # This prevents 404 errors in the frontend
    return jsonify(
        {
            "status": "not_started",
            "source_path_idx": source_path_idx,
            "camera": camera,
            "type": cal_type,
        }
    )
