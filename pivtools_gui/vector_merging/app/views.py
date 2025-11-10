"""
Vector Merging API views
Provides endpoints for merging vector fields from multiple cameras
with progress tracking and multiprocessing support.
"""

import sys
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import numpy as np
import scipy.io
from flask import Blueprint, jsonify, request
from loguru import logger
from scipy.interpolate import interpn

sys.path.append(str(Path(__file__).parent.parent.parent))

from ...config import get_config
from ...paths import get_data_paths
from ...utils import camera_number
from ...vector_loading import load_coords_from_directory, load_vectors_from_directory

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


def merge_two_vector_fields(x1, y1, ux1, uy1, mask1, x2, y2, ux2, uy2, mask2, grid_spacing=None):
    """
    Merge two vector fields with smart overlap handling.
    OPTIMIZED VERSION: Build interpolators once, reuse for both components.
    Uses data from whichever camera is available, with weighted blending in overlap regions.
    Respects original masks to prevent interpolation into masked regions.
    Fills unknown regions with NaN (keeps full extent from both cameras).

    Args:
        x1, y1, ux1, uy1, mask1: Camera 1 coordinates, vectors, and mask (True = masked/invalid)
        x2, y2, ux2, uy2, mask2: Camera 2 coordinates, vectors, and mask (True = masked/invalid)
        grid_spacing: Target grid spacing for merged field

    Returns:
        X_merged, Y_merged, ux_merged, uy_merged, uz_merged: Merged field
    """
    logger.debug("Starting optimized vector field merging...")

    # Check for empty arrays
    if x1.size == 0 or x2.size == 0:
        raise ValueError("Cannot merge: one or both coordinate arrays are empty")

    # Get full extent of both cameras (no cropping)
    y1_min, y1_max = np.nanmin(y1), np.nanmax(y1)
    y2_min, y2_max = np.nanmin(y2), np.nanmax(y2)
    
    # Find overlapping y-range (for info only)
    y_overlap_min = max(y1_min, y2_min)
    y_overlap_max = min(y1_max, y2_max)
    
    logger.debug(f"Camera 1 y-range: [{y1_min:.2f}, {y1_max:.2f}]")
    logger.debug(f"Camera 2 y-range: [{y2_min:.2f}, {y2_max:.2f}]")
    logger.debug(f"Overlapping y-range: [{y_overlap_min:.2f}, {y_overlap_max:.2f}]")
    
    # Use FULL y-range from both cameras (no cropping)
    y_min = min(y1_min, y2_min)
    y_max = max(y1_max, y2_max)
    
    # Full x-range from both cameras
    x_min = min(np.nanmin(x1), np.nanmin(x2))
    x_max = max(np.nanmax(x1), np.nanmax(x2))
    
    logger.debug(f"Merged bounds (FULL extent): x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")

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

    # Flatten query points once for all interpolations
    query_points = np.column_stack([X_merged.ravel(), Y_merged.ravel()])

    # Initialize merged arrays
    ux_merged = np.zeros(X_merged.size)
    uy_merged = np.zeros(X_merged.size)
    weight_sum = np.zeros(X_merged.size)

    # Process each camera
    for cam_idx, (x_cam, y_cam, ux_cam, uy_cam, mask_cam) in enumerate(
        [(x1, y1, ux1, uy1, mask1), (x2, y2, ux2, uy2, mask2)]
    ):
        logger.debug(f"Processing camera {cam_idx + 1}...")

        # Apply mask: set masked values to NaN (will propagate through interpn)
        ux_cam_masked = np.where(mask_cam, np.nan, ux_cam)
        uy_cam_masked = np.where(mask_cam, np.nan, uy_cam)
        
        # Check if we have any valid data
        if np.all(np.isnan(ux_cam_masked)) or np.all(np.isnan(uy_cam_masked)):
            logger.warning(f"No valid data for camera {cam_idx + 1}")
            continue

        # Extract unique x and y coordinates (assumes structured grid)
        # For structured grids from meshgrid: rows have constant y, columns have constant x
        x_coords_1d = x_cam[0, :]  # First row (all x values)
        y_coords_1d = y_cam[:, 0]  # First column (all y values)
        
        logger.debug(f"Camera {cam_idx + 1}: grid shape {x_cam.shape}, x range [{x_coords_1d[0]:.2f}, {x_coords_1d[-1]:.2f}], y range [{y_coords_1d[0]:.2f}, {y_coords_1d[-1]:.2f}]")

        # Use interpn for FAST structured grid interpolation
        # interpn expects points as (y, x) for 2D arrays
        # bounds_error=False allows extrapolation, fill_value=np.nan for out-of-bounds
        try:
            ux_interp = interpn(
                (y_coords_1d, x_coords_1d),
                ux_cam_masked,
                (Y_merged, X_merged),
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            
            uy_interp = interpn(
                (y_coords_1d, x_coords_1d),
                uy_cam_masked,
                (Y_merged, X_merged),
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )
        except Exception as e:
            logger.error(f"Interpolation failed for camera {cam_idx + 1}: {e}")
            continue
        
        # interpn automatically propagates NaN from masked regions
        ux_interp_flat = ux_interp.ravel()
        uy_interp_flat = uy_interp.ravel()

        # Create weights based on distance from edges
        x_bounds = [np.nanmin(x_cam), np.nanmax(x_cam)]
        y_bounds = [np.nanmin(y_cam), np.nanmax(y_cam)]
        weights = create_distance_weights(X_merged, Y_merged, x_bounds, y_bounds)

        # Flatten weights and filter by valid data (interpn already handled masking via NaN)
        weights_flat = weights.ravel()
        valid_interp = ~(np.isnan(ux_interp_flat) | np.isnan(uy_interp_flat))
        weights_flat = np.where(valid_interp, weights_flat, 0)

        # Accumulate weighted values (vectorized)
        ux_interp_flat = np.where(np.isnan(ux_interp_flat), 0, ux_interp_flat)
        uy_interp_flat = np.where(np.isnan(uy_interp_flat), 0, uy_interp_flat)

        ux_merged += ux_interp_flat * weights_flat
        uy_merged += uy_interp_flat * weights_flat
        weight_sum += weights_flat

    # Normalize by total weights (safe division)
    # Set unknown regions (no data from either camera) to NaN
    valid_weights = weight_sum > 0
    ux_merged = np.where(valid_weights, ux_merged / np.maximum(weight_sum, 1e-10), np.nan)
    uy_merged = np.where(valid_weights, uy_merged / np.maximum(weight_sum, 1e-10), np.nan)

    # Reshape back to 2D
    ux_merged = ux_merged.reshape(X_merged.shape)
    uy_merged = uy_merged.reshape(X_merged.shape)

    logger.debug(f"Merged field has {np.sum(valid_weights)} valid points")
    logger.debug(f"Merged field has {np.sum(~valid_weights.reshape(X_merged.shape))} NaN points (unknown regions)")
    
    uz_merged = np.zeros_like(ux_merged)  # For 2D PIV
    
    # MATLAB convention: lowest y-coordinate at bottom (first row)
    # NumPy meshgrid creates arrays where row 0 is at top, so flip vertically
    X_merged = np.flipud(X_merged)
    Y_merged = np.flipud(Y_merged)
    ux_merged = np.flipud(ux_merged)
    uy_merged = np.flipud(uy_merged)
    uz_merged = np.flipud(uz_merged)
    
    return X_merged, Y_merged, ux_merged, uy_merged, uz_merged


def find_non_empty_runs_in_file(data_dir: Path, vector_format: str) -> tuple:
    """
    Find which runs have non-empty vector data by checking the first vector file.
    Returns tuple of (list of 1-based valid run numbers, total number of runs in file).
    """
    if not data_dir.exists():
        return [], 0

    # Get first vector file to check run structure
    first_file = data_dir / (vector_format % 1)
    if not first_file.exists():
        return [], 0

    try:
        mat = scipy.io.loadmat(str(first_file), struct_as_record=False, squeeze_me=True)
        if "piv_result" not in mat:
            return [], 0

        piv_result = mat["piv_result"]
        valid_runs = []
        total_runs = 0

        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
            # Multiple runs
            total_runs = len(piv_result)
            for idx, cell in enumerate(piv_result):
                if hasattr(cell, "ux") and np.asarray(cell.ux).size > 0:
                    valid_runs.append(idx + 1)  # 1-based
        else:
            # Single run
            total_runs = 1
            if hasattr(piv_result, "ux") and np.asarray(piv_result.ux).size > 0:
                valid_runs.append(1)

        # TEMPORARY: Hardcode to only use run 4
        logger.warning("TEMPORARY: Hardcoded to only process run 4")
        valid_runs = [4] if 4 in valid_runs else []
        
        return valid_runs, total_runs
    except Exception as e:
        logger.error(f"Error checking runs in {first_file}: {e}")
        return [], 0


def _process_single_frame_merge(args):
    """
    Helper function for parallel processing of single frame merging.
    Must be a top-level function for multiprocessing.
    """
    frame_idx, base_dir, cameras, type_name, endpoint, num_images, vector_format, valid_runs, total_runs = args
    try:
        merged_runs_dict = merge_vectors_for_frame(
            base_dir,
            cameras,
            frame_idx,
            type_name,
            endpoint,
            num_images,
            vector_format,
            valid_runs,
        )
        
        # Create output directory (use Merged in the path structure)
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
        
        # Save merged data
        output_file = output_dir / (vector_format % frame_idx)
        
        # Check if we have any merged runs to save
        if len(merged_runs_dict) == 0:
            logger.warning(f"No runs could be merged for frame {frame_idx}")
            return frame_idx, False, None
        
        # Create piv_result structure preserving run indices
        # Create array with ALL runs, filling empty ones with empty arrays
        piv_dtype = np.dtype(
            [("ux", "O"), ("uy", "O"), ("uz", "O"), ("b_mask", "O")]
        )
        piv_result = np.empty(total_runs, dtype=piv_dtype)
        
        # Fill all runs (0-based array indices)
        for run_idx in range(total_runs):
            run_num = run_idx + 1  # 1-based run number
            if run_num in merged_runs_dict:
                # This run was merged
                run_data = merged_runs_dict[run_num]
                piv_result[run_idx]["ux"] = run_data["ux"]
                piv_result[run_idx]["uy"] = run_data["uy"]
                piv_result[run_idx]["uz"] = run_data["uz"]
                piv_result[run_idx]["b_mask"] = run_data["b_mask"]
            else:
                # Empty run - preserve structure
                piv_result[run_idx]["ux"] = np.array([])
                piv_result[run_idx]["uy"] = np.array([])
                piv_result[run_idx]["uz"] = np.array([])
                piv_result[run_idx]["b_mask"] = np.array([])

        scipy.io.savemat(
            str(output_file),
            {"piv_result": piv_result},
            do_compression=True,
        )
        
        return frame_idx, True, merged_runs_dict
    except Exception as e:
        logger.error(f"Error processing frame {frame_idx}: {e}", exc_info=True)
        return frame_idx, False, None


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
    merged_runs = {}  # Dictionary mapping run_num -> merged data

    for run_idx, run_num in enumerate(valid_runs):
        logger.debug(f"Processing run {run_num} (index {run_idx})")
        # Extract data for this run from each camera
        run_data = {}

        for camera, data in camera_data.items():
            piv_result = data["piv_result"]

            if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
                # Multiple runs - run_num is 1-based, array is 0-based
                array_idx = run_num - 1
                logger.debug(f"Camera {camera}: Accessing piv_result[{array_idx}] for run {run_num}")
                if array_idx < len(piv_result):
                    cell = piv_result[array_idx]
                    ux = np.asarray(cell.ux)
                    uy = np.asarray(cell.uy)
                    b_mask = np.asarray(cell.b_mask).astype(bool)
                    logger.debug(f"Camera {camera}: Loaded ux.shape={ux.shape}, uy.shape={uy.shape}")
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

            # Skip empty runs (similar to calibration approach)
            if ux.size == 0 or uy.size == 0:
                logger.debug(f"Skipping empty run {run_num} for camera {camera}")
                continue

            # Apply mask (set masked values to NaN for interpolation)
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
                "mask": b_mask,  # Store the mask for the optimized merge function
            }

        # Merge the fields for this run - need at least 2 cameras with valid data
        if len(run_data) < 2:
            logger.warning(f"Could not merge run {run_num}: insufficient cameras with valid data (got {len(run_data)}), skipping")
            continue
            
        cameras_list = list(run_data.keys())
        cam1_data = run_data[cameras_list[0]]
        cam2_data = run_data[cameras_list[1]]

        # Verify coordinates are not empty
        if cam1_data["x"].size == 0 or cam2_data["x"].size == 0:
            logger.warning(f"Empty coordinates for run {run_num}, skipping")
            continue

        X_merged, Y_merged, ux_merged, uy_merged, uz_merged = (
            merge_two_vector_fields(
                cam1_data["x"],
                cam1_data["y"],
                cam1_data["ux"],
                cam1_data["uy"],
                cam1_data["mask"],
                cam2_data["x"],
                cam2_data["y"],
                cam2_data["ux"],
                cam2_data["uy"],
                cam2_data["mask"],
            )
        )

        # Create b_mask (True where data is invalid/NaN)
        b_mask_merged = np.isnan(ux_merged) | np.isnan(uy_merged)

        # Replace NaN with 0 for saving (MATLAB compatibility)
        ux_merged_save = np.nan_to_num(ux_merged, nan=0.0)
        uy_merged_save = np.nan_to_num(uy_merged, nan=0.0)
        uz_merged_save = np.nan_to_num(uz_merged if uz_merged is not None else np.zeros_like(ux_merged), nan=0.0)

        # Store with run_num as key to preserve run indices
        merged_runs[run_num] = {
            "ux": ux_merged_save,
            "uy": uy_merged_save,
            "uz": uz_merged_save,
            "b_mask": b_mask_merged.astype(np.uint8),
            "x": X_merged,
            "y": Y_merged,
        }

    return merged_runs


@merging_bp.route("/merge_vectors/merge_one", methods=["POST"])
def merge_one_frame():
    """Merge vectors for a single frame."""
    data = request.get_json() or {}
    base_path_idx = int(data.get("base_path_idx", 0))
    cameras = data.get("cameras", [1, 2])
    frame_idx = int(data.get("frame_idx", 1))
    type_name = data.get("type_name", "instantaneous")
    endpoint = data.get("endpoint", "")
    num_images = int(data.get("image_count", 1000))

    try:
        cfg = get_config()
        base_dir = Path(cfg.base_paths[base_path_idx])
        vector_format = cfg.vector_format

        logger.info(f"Merging frame {frame_idx} for cameras {cameras}")

        # Find valid runs
        first_cam_paths = get_data_paths(
            base_dir=base_dir,
            num_images=num_images,
            cam=cameras[0],
            type_name=type_name,
            endpoint=endpoint,
        )

        valid_runs, total_runs = find_non_empty_runs_in_file(
            first_cam_paths["data_dir"], vector_format
        )

        if not valid_runs:
            return jsonify({"error": "No valid runs found in vector files"}), 400

        logger.info(f"Found {len(valid_runs)} valid runs: {valid_runs} (total runs: {total_runs})")

        # Merge the frame
        _, success, merged_runs = _process_single_frame_merge(
            (frame_idx, base_dir, cameras, type_name, endpoint, num_images, vector_format, valid_runs, total_runs)
        )

        if not success:
            return jsonify({"error": f"Failed to merge frame {frame_idx}"}), 500

        # Save coordinates if this is the first frame
        output_paths = get_data_paths(
            base_dir=base_dir,
            num_images=num_images,
            cam=cameras[0],
            type_name=type_name,
            endpoint=endpoint,
            use_merged=True,
        )
        output_dir = output_paths["data_dir"]
        coords_file = output_dir / "coordinates.mat"

        if not coords_file.exists() and merged_runs:
            # Create coordinates structure preserving run indices
            coords_dtype = np.dtype([("x", "O"), ("y", "O")])
            coordinates = np.empty(total_runs, dtype=coords_dtype)
            
            # Fill all runs
            for run_idx in range(total_runs):
                run_num = run_idx + 1
                if run_num in merged_runs:
                    coordinates[run_idx]["x"] = merged_runs[run_num]["x"]
                    coordinates[run_idx]["y"] = merged_runs[run_num]["y"]
                else:
                    # Empty run
                    coordinates[run_idx]["x"] = np.array([])
                    coordinates[run_idx]["y"] = np.array([])

            scipy.io.savemat(
                str(coords_file), {"coordinates": coordinates}, do_compression=True
            )

        return jsonify({
            "status": "success",
            "frame": frame_idx,
            "runs_merged": len(valid_runs),
            "message": f"Successfully merged frame {frame_idx}"
        })

    except Exception as e:
        logger.error(f"Error merging frame {frame_idx}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@merging_bp.route("/merge_vectors/merge_all", methods=["POST"])
def merge_all_frames():
    """Start vector merging job for all frames with multiprocessing."""
    data = request.get_json() or {}
    base_path_idx = int(data.get("base_path_idx", 0))
    cameras = data.get("cameras", [1, 2])
    type_name = data.get("type_name", "instantaneous")
    endpoint = data.get("endpoint", "")
    num_images = int(data.get("image_count", 1000))
    max_workers = min(os.cpu_count(), num_images, 8)
    job_id = str(uuid.uuid4())

    def run_merge_all():
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
                "start_time": time.time(),
            }

            logger.info(
                f"Starting vector merge for cameras {cameras}, {num_images} frames with {max_workers} workers"
            )

            # Find valid runs from first camera
            first_cam_paths = get_data_paths(
                base_dir=base_dir,
                num_images=num_images,
                cam=cameras[0],
                type_name=type_name,
                endpoint=endpoint,
            )

            valid_runs, total_runs = find_non_empty_runs_in_file(
                first_cam_paths["data_dir"], vector_format
            )

            if not valid_runs:
                raise ValueError("No valid runs found in vector files")

            logger.info(f"Found {len(valid_runs)} valid runs: {valid_runs} (total runs: {total_runs})")

            merging_jobs[job_id]["valid_runs"] = valid_runs
            merging_jobs[job_id]["progress"] = 2

            # Create output directory
            output_paths = get_data_paths(
                base_dir=base_dir,
                num_images=num_images,
                cam=cameras[0],
                type_name=type_name,
                endpoint=endpoint,
                use_merged=True,
            )

            output_dir = output_paths["data_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Output directory: {output_dir}")

            merging_jobs[job_id]["status"] = "running"
            merging_jobs[job_id]["message"] = "Merging vector fields with multiprocessing..."
            merging_jobs[job_id]["progress"] = 5

            # Prepare arguments for all frames
            frame_args = [
                (frame_idx, base_dir, cameras, type_name, endpoint, num_images, vector_format, valid_runs, total_runs)
                for frame_idx in range(1, num_images + 1)
            ]

            # Process frames in parallel
            processed_count = 0
            last_merged_runs = None
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_single_frame_merge, args) for args in frame_args]
                
                for future in as_completed(futures):
                    frame_idx, success, merged_runs = future.result()
                    processed_count += 1
                    
                    if success and merged_runs:
                        last_merged_runs = merged_runs
                    
                    merging_jobs[job_id]["processed_frames"] = processed_count
                    merging_jobs[job_id]["progress"] = int((processed_count / num_images) * 90) + 5
                    
                    if processed_count % 10 == 0:
                        logger.info(f"Merged {processed_count}/{num_images} frames")

            # Save coordinates for merged data
            coords_file = output_dir / "coordinates.mat"
            if last_merged_runs:
                # Create coordinates structure preserving run indices
                coords_dtype = np.dtype([("x", "O"), ("y", "O")])
                coordinates = np.empty(total_runs, dtype=coords_dtype)
                
                # Fill all runs
                for run_idx in range(total_runs):
                    run_num = run_idx + 1
                    if run_num in last_merged_runs:
                        coordinates[run_idx]["x"] = last_merged_runs[run_num]["x"]
                        coordinates[run_idx]["y"] = last_merged_runs[run_num]["y"]
                    else:
                        # Empty run
                        coordinates[run_idx]["x"] = np.array([])
                        coordinates[run_idx]["y"] = np.array([])

                scipy.io.savemat(
                    str(coords_file), {"coordinates": coordinates}, do_compression=True
                )

            merging_jobs[job_id]["status"] = "completed"
            merging_jobs[job_id]["progress"] = 100
            merging_jobs[job_id]["message"] = f"Successfully merged {num_images} frames with {len(valid_runs)} runs each"
            logger.info(f"Merge complete: {output_dir}")

        except Exception as e:
            logger.error(f"Error in merge job: {e}", exc_info=True)
            merging_jobs[job_id]["status"] = "failed"
            merging_jobs[job_id]["error"] = str(e)
            merging_jobs[job_id]["message"] = f"Merge failed: {str(e)}"

    # Start job in background thread
    thread = threading.Thread(target=run_merge_all)
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "job_id": job_id,
            "status": "starting",
            "message": f"Vector merging job started for cameras {cameras}",
            "image_count": num_images,
            "max_workers": max_workers,
        }
    )


@merging_bp.route("/merge_vectors/status/<job_id>", methods=["GET"])
def merge_status(job_id):
    """Get vector merging job status with timing information"""
    if job_id not in merging_jobs:
        return jsonify({"error": "Job not found"}), 404

    job_data = merging_jobs[job_id].copy()
    
    # Add timing info
    if "start_time" in job_data:
        elapsed = time.time() - job_data["start_time"]
        job_data["elapsed_time"] = elapsed
        
        if job_data["status"] == "running" and job_data.get("progress", 0) > 0:
            # Estimate remaining time
            progress_fraction = job_data["progress"] / 100.0
            if progress_fraction > 0:
                estimated_total = elapsed / progress_fraction
                estimated_remaining = estimated_total - elapsed
                job_data["estimated_remaining"] = estimated_remaining
    
    return jsonify(job_data)
