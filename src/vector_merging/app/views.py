"""
Vector Merging API views
Provides endpoints for merging vector fields from multiple cameras
with progress tracking.
"""

import sys
import threading
import uuid
from pathlib import Path

import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger
from scipy.interpolate import griddata

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_config
from paths import get_data_paths
from utils import camera_number
from vector_loading import load_coords_from_directory, load_vectors_from_directory

merging_bp = Blueprint("merging", __name__)

# Global job tracking
merging_jobs = {}


def create_distance_weights(x, y, x_bounds, y_bounds):
    """
    Create distance-based weights for blending.
    Higher weights near center, lower weights near edges.
    """
    # Normalize coordinates to [0, 1] within bounds
    x_norm = (x - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
    y_norm = (y - y_bounds[0]) / (y_bounds[1] - y_bounds[0])

    # Distance from edges (0 at edge, 0.5 at center)
    x_dist = np.minimum(x_norm, 1 - x_norm)
    y_dist = np.minimum(y_norm, 1 - y_norm)

    # Combined distance weight (Hanning-like)
    weights = np.sin(np.pi * x_dist) * np.sin(np.pi * y_dist)
    return weights


def merge_two_vector_fields(x1, y1, ux1, uy1, x2, y2, ux2, uy2, grid_spacing=None):
    """
    Merge two vector fields with smart overlap handling.

    Args:
        x1, y1, ux1, uy1: Camera 1 coordinates and vectors
        x2, y2, ux2, uy2: Camera 2 coordinates and vectors
        grid_spacing: Target grid spacing for merged field

    Returns:
        X_merged, Y_merged, ux_merged, uy_merged: Merged field
    """
    logger.debug("Starting vector field merging...")

    # Determine merged coordinate bounds
    x_min = min(np.nanmin(x1), np.nanmin(x2))
    x_max = max(np.nanmax(x1), np.nanmax(x2))
    y_min = min(np.nanmin(y1), np.nanmin(y2))
    y_max = max(np.nanmax(y1), np.nanmax(y2))

    logger.debug(
        f"Merged bounds: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]"
    )

    # Auto-determine grid spacing if not provided
    if grid_spacing is None:
        dx1 = np.median(np.diff(np.unique(x1)))
        dy1 = np.median(np.diff(np.unique(y1)))
        dx2 = np.median(np.diff(np.unique(x2)))
        dy2 = np.median(np.diff(np.unique(y2)))
        grid_spacing = min(dx1, dy1, dx2, dy2)
        logger.debug(f"Auto grid spacing: {grid_spacing:.3f}")

    # Create merged grid
    x_merged = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_merged = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    X_merged, Y_merged = np.meshgrid(x_merged, y_merged)

    logger.debug(f"Merged grid shape: {X_merged.shape}")

    # Initialize merged arrays
    ux_merged = np.full_like(X_merged, np.nan)
    uy_merged = np.full_like(X_merged, np.nan)
    uz_merged = None  # For 2D PIV
    weight_sum = np.zeros_like(X_merged)

    # Process each camera
    for cam_idx, (x_cam, y_cam, ux_cam, uy_cam) in enumerate(
        [(x1, y1, ux1, uy1), (x2, y2, ux2, uy2)]
    ):
        logger.debug(f"Processing camera {cam_idx + 1}...")

        # Get valid (non-NaN) points
        valid_mask = ~(np.isnan(ux_cam) | np.isnan(uy_cam))
        if not np.any(valid_mask):
            logger.warning(f"No valid data for camera {cam_idx + 1}")
            continue

        x_valid = x_cam[valid_mask]
        y_valid = y_cam[valid_mask]
        ux_valid = ux_cam[valid_mask]
        uy_valid = uy_cam[valid_mask]

        logger.debug(f"Camera {cam_idx + 1}: {len(x_valid)} valid points")

        # Interpolate to merged grid
        ux_interp = griddata(
            (x_valid, y_valid),
            ux_valid,
            (X_merged, Y_merged),
            method="linear",
            fill_value=np.nan,
        )
        uy_interp = griddata(
            (x_valid, y_valid),
            uy_valid,
            (X_merged, Y_merged),
            method="linear",
            fill_value=np.nan,
        )

        # Create weights based on distance from edges of this camera's domain
        x_bounds = [np.nanmin(x_cam), np.nanmax(x_cam)]
        y_bounds = [np.nanmin(y_cam), np.nanmax(y_cam)]
        weights = create_distance_weights(X_merged, Y_merged, x_bounds, y_bounds)

        # Only apply weights where data is valid
        valid_interp = ~(np.isnan(ux_interp) | np.isnan(uy_interp))
        weights = np.where(valid_interp, weights, 0)

        # Accumulate weighted values
        ux_merged = np.where(
            weight_sum == 0,
            np.where(valid_interp, ux_interp, np.nan),
            np.where(
                valid_interp,
                (ux_merged * weight_sum + ux_interp * weights) / (weight_sum + weights),
                ux_merged,
            ),
        )
        uy_merged = np.where(
            weight_sum == 0,
            np.where(valid_interp, uy_interp, np.nan),
            np.where(
                valid_interp,
                (uy_merged * weight_sum + uy_interp * weights) / (weight_sum + weights),
                uy_merged,
            ),
        )
        weight_sum += weights

    logger.debug(f"Merged field has {np.sum(~np.isnan(ux_merged))} valid points")
    return X_merged, Y_merged, ux_merged, uy_merged, uz_merged


def find_non_empty_runs_in_file(data_dir: Path, vector_format: str) -> list:
    """
    Find which runs have non-empty vector data by checking the first vector file.
    Returns list of 1-based run numbers that contain valid data.
    """
    if not data_dir.exists():
        return []

    # Get first vector file to check run structure
    first_file = data_dir / (vector_format % 1)
    if not first_file.exists():
        return []

    try:
        mat = scipy.io.loadmat(str(first_file), struct_as_record=False, squeeze_me=True)
        if "piv_result" not in mat:
            return []

        piv_result = mat["piv_result"]
        valid_runs = []

        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
            # Multiple runs
            for idx, cell in enumerate(piv_result):
                if hasattr(cell, "ux") and np.asarray(cell.ux).size > 0:
                    valid_runs.append(idx + 1)  # 1-based
        else:
            # Single run
            if hasattr(piv_result, "ux") and np.asarray(piv_result.ux).size > 0:
                valid_runs.append(1)

        return valid_runs
    except Exception as e:
        logger.error(f"Error checking runs in {first_file}: {e}")
        return []


def merge_vectors_for_frame(
    base_dir: Path,
    cameras: list,
    frame_idx: int,
    type_name: str,
    endpoint: str,
    num_images: int,
    vector_format: str,
    valid_runs: list,
):
    """
    Merge vectors from multiple cameras for a single frame.
    Returns merged data structure matching the expected format.
    """
    camera_data = {}

    # Load data from each camera
    for camera in cameras:
        paths = get_data_paths(
            base_dir=base_dir,
            num_images=num_images,
            cam=camera,
            type_name=type_name,
            endpoint=endpoint,
        )

        data_dir = paths["data_dir"]
        if not data_dir.exists():
            logger.warning(f"Data directory does not exist for camera {camera}")
            continue

        # Load coordinates
        try:
            coords_x_list, coords_y_list = load_coords_from_directory(
                data_dir, runs=valid_runs
            )
        except Exception as e:
            logger.error(f"Failed to load coordinates for camera {camera}: {e}")
            continue

        # Load vector file
        vector_file = data_dir / (vector_format % frame_idx)
        if not vector_file.exists():
            logger.warning(f"Vector file does not exist: {vector_file}")
            continue

        try:
            mat = scipy.io.loadmat(
                str(vector_file), struct_as_record=False, squeeze_me=True
            )
            if "piv_result" not in mat:
                logger.warning(f"No piv_result in {vector_file}")
                continue

            piv_result = mat["piv_result"]
            camera_data[camera] = {
                "piv_result": piv_result,
                "coords_x": coords_x_list,
                "coords_y": coords_y_list,
            }
        except Exception as e:
            logger.error(f"Failed to load vector file {vector_file}: {e}")
            continue

    if len(camera_data) < 2:
        raise ValueError(
            f"Need at least 2 cameras with data, only found {len(camera_data)}"
        )

    # Merge data for each run
    merged_runs = []

    for run_idx, run_num in enumerate(valid_runs):
        # Extract data for this run from each camera
        run_data = {}

        for camera, data in camera_data.items():
            piv_result = data["piv_result"]

            if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
                # Multiple runs
                if run_idx < len(piv_result):
                    cell = piv_result[run_idx]
                    ux = np.asarray(cell.ux)
                    uy = np.asarray(cell.uy)
                    b_mask = np.asarray(cell.b_mask).astype(bool)
                else:
                    continue
            else:
                # Single run
                if run_idx == 0:
                    ux = np.asarray(piv_result.ux)
                    uy = np.asarray(piv_result.uy)
                    b_mask = np.asarray(piv_result.b_mask).astype(bool)
                else:
                    continue

            # Apply mask
            ux_masked = np.where(b_mask, np.nan, ux)
            uy_masked = np.where(b_mask, np.nan, uy)

            # Get coordinates for this run
            x_coords = data["coords_x"][run_idx]
            y_coords = data["coords_y"][run_idx]

            run_data[camera] = {
                "x": x_coords,
                "y": y_coords,
                "ux": ux_masked,
                "uy": uy_masked,
            }

        # Merge the fields for this run
        if len(run_data) == 2:
            cameras_list = list(run_data.keys())
            cam1_data = run_data[cameras_list[0]]
            cam2_data = run_data[cameras_list[1]]

            X_merged, Y_merged, ux_merged, uy_merged, uz_merged = (
                merge_two_vector_fields(
                    cam1_data["x"],
                    cam1_data["y"],
                    cam1_data["ux"],
                    cam1_data["uy"],
                    cam2_data["x"],
                    cam2_data["y"],
                    cam2_data["ux"],
                    cam2_data["uy"],
                )
            )

            # Create b_mask (True where data is invalid/NaN)
            b_mask_merged = np.isnan(ux_merged) | np.isnan(uy_merged)

            # Replace NaN with 0 for saving
            ux_merged = np.nan_to_num(ux_merged, nan=0.0)
            uy_merged = np.nan_to_num(uy_merged, nan=0.0)

            merged_runs.append(
                {
                    "ux": ux_merged,
                    "uy": uy_merged,
                    "uz": uz_merged if uz_merged is not None else np.zeros_like(ux_merged),
                    "b_mask": b_mask_merged.astype(np.uint8),
                    "x": X_merged,
                    "y": Y_merged,
                }
            )
        else:
            logger.warning(f"Could not merge run {run_num}, skipping")

    return merged_runs


@merging_bp.route("/merge_vectors/merge", methods=["POST"])
def merge_vectors():
    """Start vector merging job with progress tracking."""
    data = request.get_json() or {}
    base_path_idx = int(data.get("base_path_idx", 0))
    cameras = data.get("cameras", [1, 2])  # List of camera numbers
    type_name = data.get("type_name", "instantaneous")
    endpoint = data.get("endpoint", "")
    num_images = int(data.get("image_count", 1000))

    job_id = str(uuid.uuid4())

    def run_merge():
        try:
            cfg = get_config()
            base_dir = Path(cfg.base_paths[base_path_idx])
            vector_format = cfg.vector_format

            merging_jobs[job_id] = {
                "status": "starting",
                "progress": 0,
                "total_frames": num_images,
                "processed_frames": 0,
                "message": "Initializing merge operation...",
            }

            logger.info(
                f"Starting vector merge for cameras {cameras}, {num_images} frames"
            )

            # Find valid runs from first camera
            first_cam_paths = get_data_paths(
                base_dir=base_dir,
                num_images=num_images,
                cam=cameras[0],
                type_name=type_name,
                endpoint=endpoint,
            )

            valid_runs = find_non_empty_runs_in_file(
                first_cam_paths["data_dir"], vector_format
            )

            if not valid_runs:
                raise ValueError("No valid runs found in vector files")

            logger.info(f"Found {len(valid_runs)} valid runs: {valid_runs}")

            merging_jobs[job_id]["valid_runs"] = valid_runs
            merging_jobs[job_id]["progress"] = 5

            # Create output directory (use Merged in the path structure)
            # Note: We use cam=cameras[0] but the folder will be "Merged" due to use_merged=True
            output_paths = get_data_paths(
                base_dir=base_dir,
                num_images=num_images,
                cam=cameras[0],  # This gets overridden by use_merged
                type_name=type_name,
                endpoint=endpoint,
                use_merged=True,
            )

            output_dir = output_paths["data_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Output directory: {output_dir}")

            merging_jobs[job_id]["status"] = "running"
            merging_jobs[job_id]["message"] = "Merging vector fields..."

            # Process each frame
            for frame_idx in range(1, num_images + 1):
                try:
                    merged_runs = merge_vectors_for_frame(
                        base_dir,
                        cameras,
                        frame_idx,
                        type_name,
                        endpoint,
                        num_images,
                        vector_format,
                        valid_runs,
                    )

                    # Save merged data in MATLAB format matching expected structure
                    output_file = output_dir / (vector_format % frame_idx)

                    # Create piv_result structure matching the format expected by read_mat_contents
                    if len(merged_runs) > 1:
                        # Multiple runs - create object array
                        piv_dtype = np.dtype(
                            [("ux", "O"), ("uy", "O"), ("uz", "O"), ("b_mask", "O")]
                        )
                        piv_result = np.empty(len(merged_runs), dtype=piv_dtype)
                        for idx, run in enumerate(merged_runs):
                            piv_result[idx]["ux"] = run["ux"]
                            piv_result[idx]["uy"] = run["uy"]
                            piv_result[idx]["uz"] = run["uz"]
                            piv_result[idx]["b_mask"] = run["b_mask"]
                    else:
                        # Single run - create single struct
                        class PIVResult:
                            def __init__(self, ux, uy, uz, b_mask):
                                self.ux = ux
                                self.uy = uy
                                self.uz = uz
                                self.b_mask = b_mask

                        piv_result = PIVResult(
                            merged_runs[0]["ux"],
                            merged_runs[0]["uy"],
                            merged_runs[0]["uz"],
                            merged_runs[0]["b_mask"],
                        )

                    scipy.io.savemat(
                        str(output_file),
                        {"piv_result": piv_result},
                        do_compression=True,
                    )

                    merging_jobs[job_id]["processed_frames"] = frame_idx
                    merging_jobs[job_id]["progress"] = int((frame_idx / num_images) * 90) + 5

                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {e}")
                    # Continue with next frame

            # Save coordinates for merged data
            # Use coordinates from first run of merged data
            coords_file = output_dir / "coordinates.mat"
            if merged_runs:
                if len(valid_runs) > 1:
                    # Multiple runs - create object array
                    coords_dtype = np.dtype([("x", "O"), ("y", "O")])
                    coordinates = np.empty(len(valid_runs), dtype=coords_dtype)
                    for idx in range(len(valid_runs)):
                        # Load the last processed frame to get coordinates
                        last_frame_file = output_dir / (vector_format % num_images)
                        if last_frame_file.exists():
                            mat = scipy.io.loadmat(
                                str(last_frame_file),
                                struct_as_record=False,
                                squeeze_me=True,
                            )
                            if "piv_result" in mat:
                                # Get coordinates from the saved data
                                # We need to save coordinates separately during merge
                                pass
                    
                    # For now, save coordinates from the last merged runs
                    for idx, run in enumerate(merged_runs):
                        coordinates[idx]["x"] = run["x"]
                        coordinates[idx]["y"] = run["y"]
                else:
                    # Single run - create single struct
                    class Coordinates:
                        def __init__(self, x, y):
                            self.x = x
                            self.y = y

                    coordinates = Coordinates(merged_runs[0]["x"], merged_runs[0]["y"])

                scipy.io.savemat(
                    str(coords_file), {"coordinates": coordinates}, do_compression=True
                )

            merging_jobs[job_id]["status"] = "completed"
            merging_jobs[job_id]["progress"] = 100
            merging_jobs[job_id]["message"] = f"Successfully merged {num_images} frames"
            logger.info(f"Merge complete: {output_dir}")

        except Exception as e:
            logger.error(f"Error in merge job: {e}", exc_info=True)
            merging_jobs[job_id]["status"] = "failed"
            merging_jobs[job_id]["error"] = str(e)
            merging_jobs[job_id]["message"] = f"Merge failed: {str(e)}"

    # Start job in background thread
    thread = threading.Thread(target=run_merge)
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "job_id": job_id,
            "status": "starting",
            "message": f"Vector merging job started for cameras {cameras}",
            "image_count": num_images,
        }
    )


@merging_bp.route("/merge_vectors/status/<job_id>", methods=["GET"])
def merge_status(job_id):
    """Get vector merging job status"""
    if job_id not in merging_jobs:
        return jsonify({"error": "Job not found"}), 404

    job_data = merging_jobs[job_id].copy()
    return jsonify(job_data)
