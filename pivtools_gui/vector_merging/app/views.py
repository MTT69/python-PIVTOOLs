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

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from ...utils import camera_number
from pivtools_core.vector_loading import load_coords_from_directory, load_vectors_from_directory
from scipy.interpolate import RegularGridInterpolator

merging_bp = Blueprint("merging", __name__)

# Global job tracking
merging_jobs = {}


def _convert_to_half_precision(arr: np.ndarray) -> np.ndarray:
    """
    Convert float arrays to half precision (float16) for space saving.
    """
    if arr is None or arr.size == 0:
        return arr
    if arr.dtype.kind == 'f':
        return arr.astype(np.float16)
    return arr



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
        valid_runs = [4] if 4 in valid_runs else []  # --- TEMPORARY OVERRIDE FOR TESTING ---
        return valid_runs, total_runs
    except Exception as e:
        logger.error(f"Error checking runs in {first_file}: {e}")
        return [], 0


def merge_n_camera_fields(camera_data_dict):
    """
    Merge n cameras using unified grid with distance-based Hanning blend.
    Based on production merging logic from simple_merge.py.

    Args:
        camera_data_dict: Dict mapping camera_idx -> {
            'x': x coordinates (1D or 2D),
            'y': y coordinates (1D or 2D),
            'ux': x velocity (masked with NaN),
            'uy': y velocity (masked with NaN),
            'mask': boolean mask
        }

    Returns:
        Tuple of (X_merged, Y_merged, ux_merged, uy_merged, uz_merged)
    """
    if len(camera_data_dict) < 2:
        raise ValueError(f"Need at least 2 cameras, got {len(camera_data_dict)}")

    # Get first camera for reference
    first_cam_idx = min(camera_data_dict.keys())
    first_cam = camera_data_dict[first_cam_idx]

    # Compute grid spacing from first camera
    x_first = np.asarray(first_cam['x'])
    y_first = np.asarray(first_cam['y'])

    if x_first.ndim == 1:
        x_first_vec = x_first
        y_first_vec = y_first
    else:
        x_first_vec = x_first[0, :]
        y_first_vec = y_first[:, 0]

    dx = abs(np.median(np.diff(x_first_vec)))
    dy = abs(np.median(np.diff(y_first_vec)))

    # Combined bounds from all cameras
    x_min = min(cam_data['x'].min() for cam_data in camera_data_dict.values())
    x_max = max(cam_data['x'].max() for cam_data in camera_data_dict.values())
    y_min = min(cam_data['y'].min() for cam_data in camera_data_dict.values())
    y_max = max(cam_data['y'].max() for cam_data in camera_data_dict.values())

    # Create unified grid
    nx = int(np.round((x_max - x_min) / dx)) + 1
    ny = int(np.round((y_max - y_min) / dy)) + 1
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    xg, yg = np.meshgrid(x_grid, y_grid, indexing='xy')

    logger.debug(f"Unified grid: {nx} x {ny}, X:[{x_min:.2f}, {x_max:.2f}], Y:[{y_min:.2f}, {y_max:.2f}]")

    # Interpolate all cameras to unified grid
    points = np.stack([yg.ravel(), xg.ravel()], axis=-1)
    camera_interp = {}

    for cam_idx, cam_data in camera_data_dict.items():
        logger.debug(f"Interpolating camera {cam_idx}...")

        # Get camera coordinates and data
        x_cam = np.asarray(cam_data['x'])
        y_cam = np.asarray(cam_data['y'])
        ux_cam = np.asarray(cam_data['ux'])
        uy_cam = np.asarray(cam_data['uy'])
        mask_cam = np.asarray(cam_data['mask'])

        # Extract vectors for 1D coords
        if x_cam.ndim == 1:
            x_vec, y_vec = x_cam, y_cam
        else:
            x_vec = x_cam[0, :]
            y_vec = y_cam[:, 0]

        # Reshape data if needed
        if ux_cam.ndim == 1:
            ny_cam, nx_cam = len(y_vec), len(x_vec)
            ux_cam = ux_cam.reshape(ny_cam, nx_cam)
            uy_cam = uy_cam.reshape(ny_cam, nx_cam)
            mask_cam = mask_cam.reshape(ny_cam, nx_cam)

        # Ensure y_vec is ascending for RegularGridInterpolator
        if y_vec[1] < y_vec[0]:
            y_vec = y_vec[::-1]
            ux_cam = np.flipud(ux_cam)
            uy_cam = np.flipud(uy_cam)
            mask_cam = np.flipud(mask_cam)

        # Create interpolators (replace NaN with 0 for interpolation)
        valid_ux = np.where(np.isnan(ux_cam), 0, ux_cam)
        valid_uy = np.where(np.isnan(uy_cam), 0, uy_cam)
        interp_ux = RegularGridInterpolator(
            (y_vec, x_vec), valid_ux, method='nearest',
            bounds_error=False, fill_value=np.nan
        )
        interp_uy = RegularGridInterpolator(
            (y_vec, x_vec), valid_uy, method='nearest',
            bounds_error=False, fill_value=np.nan
        )
        interp_mask = RegularGridInterpolator(
            (y_vec, x_vec), mask_cam.astype(float), method='nearest',
            bounds_error=False, fill_value=1.0
        )

        # Interpolate to unified grid
        ux_interp = interp_ux(points).reshape(yg.shape)
        uy_interp = interp_uy(points).reshape(yg.shape)
        mask_interp = interp_mask(points).reshape(yg.shape) > 0.5

        # Store interpolated data and valid region
        camera_interp[cam_idx] = {
            'ux': ux_interp,
            'uy': uy_interp,
            'mask': mask_interp,
            'valid': ~np.isnan(ux_interp) & ~mask_interp,
            'x_center': np.mean(x_cam),
            'y_center': np.mean(y_cam)
        }

    # Determine stacking direction (horizontal or vertical)
    cam_centers = [(camera_interp[idx]['x_center'], camera_interp[idx]['y_center'])
                   for idx in sorted(camera_data_dict.keys())]
    x_spread = max(c[0] for c in cam_centers) - min(c[0] for c in cam_centers)
    y_spread = max(c[1] for c in cam_centers) - min(c[1] for c in cam_centers)

    if x_spread >= y_spread:
        stack_direction = 'horizontal'
        logger.debug(f"Detected horizontal stacking (x_spread={x_spread:.2f} mm)")
    else:
        stack_direction = 'vertical'
        logger.debug(f"Detected vertical stacking (y_spread={y_spread:.2f} mm)")

    # Create weight maps for each camera using distance-based Hanning blend
    logger.debug("Computing blend weights...")
    camera_weights = {}

    for cam_idx in camera_data_dict.keys():
        # Initialize weight to 1 where this camera is valid
        weight = np.where(camera_interp[cam_idx]['valid'], 1.0, 0.0)

        # For overlap regions, use distance-based weighting
        if stack_direction == 'horizontal':
            # Weight based on distance from camera center in x-direction
            cam_x_center = camera_interp[cam_idx]['x_center']
            for other_idx in camera_data_dict.keys():
                if other_idx == cam_idx:
                    continue

                # Find overlap region
                valid_this = camera_interp[cam_idx]['valid']
                valid_other = camera_interp[other_idx]['valid']
                overlap = valid_this & valid_other

                if np.any(overlap):
                    # Get x-coordinates of overlap
                    x_overlap = xg[overlap]
                    x_min_overlap = x_overlap.min()
                    x_max_overlap = x_overlap.max()

                    # Determine which camera is left/right
                    other_x_center = camera_interp[other_idx]['x_center']
                    if cam_x_center < other_x_center:
                        # This camera is on the left - high on left, low on right
                        x_norm = (x_overlap - x_min_overlap) / (x_max_overlap - x_min_overlap)
                        overlap_weight = 0.5 * (1 + np.cos(np.pi * x_norm))
                    else:
                        # This camera is on the right - low on left, high on right
                        x_norm = (x_overlap - x_min_overlap) / (x_max_overlap - x_min_overlap)
                        overlap_weight = 0.5 * (1 - np.cos(np.pi * x_norm))

                    weight[overlap] = overlap_weight
        else:  # vertical
            # Weight based on distance from camera center in y-direction
            cam_y_center = camera_interp[cam_idx]['y_center']
            for other_idx in camera_data_dict.keys():
                if other_idx == cam_idx:
                    continue

                # Find overlap region
                valid_this = camera_interp[cam_idx]['valid']
                valid_other = camera_interp[other_idx]['valid']
                overlap = valid_this & valid_other

                if np.any(overlap):
                    # Get y-coordinates of overlap
                    y_overlap = yg[overlap]
                    y_min_overlap = y_overlap.min()
                    y_max_overlap = y_overlap.max()

                    # Determine which camera is top/bottom
                    other_y_center = camera_interp[other_idx]['y_center']
                    if cam_y_center < other_y_center:
                        # This camera is on the bottom - high on bottom, low on top
                        y_norm = (y_overlap - y_min_overlap) / (y_max_overlap - y_min_overlap)
                        overlap_weight = 0.5 * (1 + np.cos(np.pi * y_norm))
                    else:
                        # This camera is on the top - low on bottom, high on top
                        y_norm = (y_overlap - y_min_overlap) / (y_max_overlap - y_min_overlap)
                        overlap_weight = 0.5 * (1 - np.cos(np.pi * y_norm))

                    weight[overlap] = overlap_weight

        camera_weights[cam_idx] = weight

    # Normalize weights so they sum to 1 at each point
    total_weight = np.zeros_like(xg)
    for cam_idx in camera_data_dict.keys():
        total_weight += camera_weights[cam_idx]

    for cam_idx in camera_data_dict.keys():
        # Avoid division by zero
        valid_total = total_weight > 0
        camera_weights[cam_idx] = np.where(
            valid_total,
            camera_weights[cam_idx] / total_weight,
            0
        )

    # Create merged fields by weighted sum
    logger.debug("Blending cameras...")
    ux_merged = np.zeros_like(xg)
    uy_merged = np.zeros_like(yg)

    for cam_idx in camera_data_dict.keys():
        ux_merged += camera_weights[cam_idx] * np.nan_to_num(camera_interp[cam_idx]['ux'], nan=0.0)
        uy_merged += camera_weights[cam_idx] * np.nan_to_num(camera_interp[cam_idx]['uy'], nan=0.0)

    # Set to NaN where no camera has valid data
    no_data = total_weight == 0
    ux_merged[no_data] = np.nan
    uy_merged[no_data] = np.nan

    # uz is not used in 2D PIV
    uz_merged = np.zeros_like(ux_merged)

    return xg, yg, ux_merged, uy_merged, uz_merged


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

        # Verify coordinates are not empty
        for camera, data in run_data.items():
            if data["x"].size == 0:
                logger.warning(f"Empty coordinates for run {run_num}, camera {camera}, skipping")
                continue

        # Use n-camera merge with production Hanning blend algorithm
        logger.debug(f"Merging {len(run_data)} cameras for run {run_num}")
        X_merged, Y_merged, ux_merged, uy_merged, uz_merged = merge_n_camera_fields(run_data)

        # Create b_mask (True where data is invalid/NaN)
        b_mask_merged = np.isnan(ux_merged) | np.isnan(uy_merged)

        # Replace NaN with 0 for saving (MATLAB compatibility)
        ux_merged_save = np.nan_to_num(ux_merged, nan=0.0)
        uy_merged_save = np.nan_to_num(uy_merged, nan=0.0)
        uz_merged_save = np.nan_to_num(uz_merged if uz_merged is not None else np.zeros_like(ux_merged), nan=0.0)

        # Flip arrays vertically to match Cartesian coordinates (smallest y at bottom)
        ux_merged_save = ux_merged_save[::-1, :]
        uy_merged_save = uy_merged_save[::-1, :]
        uz_merged_save = uz_merged_save[::-1, :]
        b_mask_merged = b_mask_merged[::-1, :]
        Y_merged = Y_merged[::-1, :]

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
                    # Get merged coordinates (already in Cartesian format with smallest y at bottom)
                    x_coords = merged_runs[run_num]["x"]
                    y_coords = merged_runs[run_num]["y"]
                    
                    # Convert to half precision for space saving
                    x_coords = _convert_to_half_precision(x_coords)
                    y_coords = _convert_to_half_precision(y_coords)
                    
                    coordinates[run_idx]["x"] = x_coords
                    coordinates[run_idx]["y"] = y_coords
                else:
                    # Empty run
                    coordinates[run_idx]["x"] = np.array([], dtype=np.float16)
                    coordinates[run_idx]["y"] = np.array([], dtype=np.float16)

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
    max_workers = min(os.cpu_count() or 4, num_images, 8)
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
                        # Get merged coordinates (already in Cartesian format with smallest y at bottom)
                        x_coords = last_merged_runs[run_num]["x"]
                        y_coords = last_merged_runs[run_num]["y"]
                        
                        # Convert to half precision for space saving
                        x_coords = _convert_to_half_precision(x_coords)
                        y_coords = _convert_to_half_precision(y_coords)
                        
                        coordinates[run_idx]["x"] = x_coords
                        coordinates[run_idx]["y"] = y_coords
                    else:
                        # Empty run
                        coordinates[run_idx]["x"] = np.array([], dtype=np.float16)
                        coordinates[run_idx]["y"] = np.array([], dtype=np.float16)

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
