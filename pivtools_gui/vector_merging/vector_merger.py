#!/usr/bin/env python3
"""
vector_merger.py

Multi-camera vector field merging using Hanning blend.
Can be used standalone (CLI) or via GUI with progress callbacks.

This class abstracts the vector merging logic from the GUI layer,
following the pattern established by MultiViewCalibrator in the
calibration module.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import scipy.io
from loguru import logger
from scipy.interpolate import RegularGridInterpolator

from pivtools_core.config import get_config, reload_config
from pivtools_core.paths import get_data_paths
from pivtools_core.vector_loading import (
    find_valid_piv_runs,
    load_coords_from_directory,
    read_mat_contents,
)

# ===================== CONFIGURATION =====================
# These settings are used when running standalone (USE_CONFIG_DIRECTLY=False)
# Or edit config.yaml and set USE_CONFIG_DIRECTLY=True

# BASE_DIR: Base directory where PIV data is stored
BASE_DIR = "/path/to/data"

# CAMERAS: List of camera numbers to merge
CAMERAS = [1, 2]

# TYPE_NAME: Vector type ("instantaneous", "ensemble", etc.)
TYPE_NAME = "instantaneous"

# ENDPOINT: Optional endpoint specification
ENDPOINT = ""

# MAX_WORKERS: Number of parallel workers for processing
MAX_WORKERS = 8

# USE_CONFIG_DIRECTLY: If True, load settings directly from config.yaml
USE_CONFIG_DIRECTLY = True

# LOGGING SETUP
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def apply_cli_settings_to_config():
    """Update config.yaml with CLI-mode hardcoded settings.

    This function writes the hardcoded configuration variables to config.yaml,
    ensuring the centralized path system uses the correct paths and settings.

    Returns
    -------
    Config
        The reloaded config object with updated settings
    """
    config = get_config()

    # Paths
    config.data["paths"]["base_paths"] = [BASE_DIR]
    config.data["paths"]["camera_numbers"] = CAMERAS

    # Merging settings
    if "merging" not in config.data:
        config.data["merging"] = {}
    config.data["merging"]["type_name"] = TYPE_NAME
    config.data["merging"]["endpoint"] = ENDPOINT
    config.data["merging"]["max_workers"] = MAX_WORKERS
    config.data["merging"]["cameras"] = CAMERAS

    # Save to disk so centralized loader picks up changes
    config.save()
    logger.info("Updated config.yaml with CLI settings")

    # Reload to ensure fresh state
    return reload_config()


def _convert_to_half_precision(arr: np.ndarray) -> np.ndarray:
    """Convert float arrays to half precision (float16) for space saving."""
    if arr is None or arr.size == 0:
        return arr
    if arr.dtype.kind == "f":
        return arr.astype(np.float16)
    return arr


def _process_frame_worker(args: tuple) -> tuple:
    """
    Worker function for parallel frame processing.
    Must be module-level for ProcessPoolExecutor.

    Args:
        args: Tuple of (frame_idx, merger_params)

    Returns:
        Tuple of (frame_idx, success, merged_runs_dict)
    """
    (
        frame_idx,
        base_dir,
        cameras,
        type_name,
        endpoint,
        num_frame_pairs,
        vector_format,
        valid_runs,
        total_runs,
    ) = args

    try:
        # Create a temporary merger instance for this worker
        merger = VectorMerger(
            base_dir=Path(base_dir),
            cameras=cameras,
            type_name=type_name,
            endpoint=endpoint,
            num_frame_pairs=num_frame_pairs,
            vector_format=vector_format,
        )

        # Merge the frame
        merged_runs = merger.merge_single_frame(frame_idx, valid_runs)

        if not merged_runs:
            logger.warning(f"No runs could be merged for frame {frame_idx}")
            return frame_idx, False, None

        # Save the result
        merger.save_frame_result(frame_idx, merged_runs, total_runs)

        return frame_idx, True, merged_runs

    except Exception as e:
        logger.error(f"Error processing frame {frame_idx}: {e}", exc_info=True)
        return frame_idx, False, None


class VectorMerger:
    """
    Multi-camera vector field merging using Hanning blend.

    This class can be used standalone for CLI operations or via the GUI
    with progress callbacks. It follows the same pattern as MultiViewCalibrator.

    Example CLI usage:
        merger = VectorMerger(
            base_dir=Path("/path/to/data"),
            cameras=[1, 2],
            type_name="instantaneous",
        )
        result = merger.run()

    Example GUI usage:
        merger = VectorMerger(base_dir, cameras, type_name, endpoint)
        result = merger.merge_all_frames(progress_callback=callback)
    """

    def __init__(
        self,
        base_dir: Path,
        cameras: list,
        type_name: str = "instantaneous",
        endpoint: str = "",
        num_frame_pairs: Optional[int] = None,
        vector_format: Optional[str] = None,
    ):
        """
        Initialize the VectorMerger.

        Args:
            base_dir: Base directory for data
            cameras: List of camera numbers to merge (e.g., [1, 2])
            type_name: Type of vectors ("instantaneous", "averaged", etc.)
            endpoint: Optional endpoint specification
            num_frame_pairs: Number of frame pairs (read from config if None)
            vector_format: Vector file format (read from config if None)
        """
        self.base_dir = Path(base_dir)
        self.cameras = cameras
        self.type_name = type_name
        self.endpoint = endpoint

        # Read from config if not provided
        cfg = get_config()
        self.num_frame_pairs = num_frame_pairs or cfg.num_frame_pairs
        self.vector_format = vector_format or cfg.vector_format

        # Setup output directory
        self.output_paths = get_data_paths(
            base_dir=self.base_dir,
            num_frame_pairs=self.num_frame_pairs,
            cam=self.cameras[0],
            type_name=self.type_name,
            endpoint=self.endpoint,
            use_merged=True,
        )
        self.output_dir = self.output_paths["data_dir"]

    def find_valid_runs(self) -> tuple:
        """
        Find which runs have valid (non-empty) vector data.

        Returns:
            Tuple of (list of valid run numbers, total number of runs)
        """
        first_cam_paths = get_data_paths(
            base_dir=self.base_dir,
            num_frame_pairs=self.num_frame_pairs,
            cam=self.cameras[0],
            type_name=self.type_name,
            endpoint=self.endpoint,
        )

        data_dir = first_cam_paths["data_dir"]
        if not data_dir.exists():
            return [], 0

        first_file = data_dir / (self.vector_format % 1)
        if not first_file.exists():
            return [], 0

        try:
            result = find_valid_piv_runs(first_file, one_based=True)
            return result.valid_runs, result.total_runs
        except Exception as e:
            logger.error(f"Error checking runs in {first_file}: {e}")
            return [], 0

    @staticmethod
    def merge_n_camera_fields(camera_data_dict: dict) -> tuple:
        """
        Merge n cameras using unified grid with distance-based Hanning blend.

        This is the core algorithm for vector field merging. It creates a unified
        grid spanning all cameras and uses Hanning window weighting for smooth
        blending in overlap regions.

        Args:
            camera_data_dict: Dict mapping camera_idx -> {
                'x': x coordinates (1D or 2D),
                'y': y coordinates (1D or 2D),
                'ux': x velocity (masked with NaN),
                'uy': y velocity (masked with NaN),
                'mask': boolean mask (True = invalid)
            }

        Returns:
            Tuple of (X_merged, Y_merged, ux_merged, uy_merged, uz_merged)

        Raises:
            ValueError: If fewer than 2 cameras provided
        """
        if len(camera_data_dict) < 2:
            raise ValueError(f"Need at least 2 cameras, got {len(camera_data_dict)}")

        # Get first camera for reference
        first_cam_idx = min(camera_data_dict.keys())
        first_cam = camera_data_dict[first_cam_idx]

        # Compute grid spacing from first camera
        x_first = np.asarray(first_cam["x"])
        y_first = np.asarray(first_cam["y"])

        if x_first.ndim == 1:
            x_first_vec = x_first
            y_first_vec = y_first
        else:
            x_first_vec = x_first[0, :]
            y_first_vec = y_first[:, 0]

        dx = abs(np.median(np.diff(x_first_vec)))
        dy = abs(np.median(np.diff(y_first_vec)))

        # Combined bounds from all cameras
        x_min = min(cam_data["x"].min() for cam_data in camera_data_dict.values())
        x_max = max(cam_data["x"].max() for cam_data in camera_data_dict.values())
        y_min = min(cam_data["y"].min() for cam_data in camera_data_dict.values())
        y_max = max(cam_data["y"].max() for cam_data in camera_data_dict.values())

        # Create unified grid
        nx = int(np.round((x_max - x_min) / dx)) + 1
        ny = int(np.round((y_max - y_min) / dy)) + 1
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        xg, yg = np.meshgrid(x_grid, y_grid, indexing="xy")

        logger.debug(
            f"Unified grid: {nx} x {ny}, X:[{x_min:.2f}, {x_max:.2f}], Y:[{y_min:.2f}, {y_max:.2f}]"
        )

        # Interpolate all cameras to unified grid
        points = np.stack([yg.ravel(), xg.ravel()], axis=-1)
        camera_interp = {}

        for cam_idx, cam_data in camera_data_dict.items():
            logger.debug(f"Interpolating camera {cam_idx}...")

            # Get camera coordinates and data
            x_cam = np.asarray(cam_data["x"])
            y_cam = np.asarray(cam_data["y"])
            ux_cam = np.asarray(cam_data["ux"])
            uy_cam = np.asarray(cam_data["uy"])
            mask_cam = np.asarray(cam_data["mask"])

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
                (y_vec, x_vec),
                valid_ux,
                method="nearest",
                bounds_error=False,
                fill_value=np.nan,
            )
            interp_uy = RegularGridInterpolator(
                (y_vec, x_vec),
                valid_uy,
                method="nearest",
                bounds_error=False,
                fill_value=np.nan,
            )
            interp_mask = RegularGridInterpolator(
                (y_vec, x_vec),
                mask_cam.astype(float),
                method="nearest",
                bounds_error=False,
                fill_value=1.0,
            )

            # Interpolate to unified grid
            ux_interp = interp_ux(points).reshape(yg.shape)
            uy_interp = interp_uy(points).reshape(yg.shape)
            mask_interp = interp_mask(points).reshape(yg.shape) > 0.5

            # Store interpolated data and valid region
            camera_interp[cam_idx] = {
                "ux": ux_interp,
                "uy": uy_interp,
                "mask": mask_interp,
                "valid": ~np.isnan(ux_interp) & ~mask_interp,
                "x_center": np.mean(x_cam),
                "y_center": np.mean(y_cam),
            }

        # Determine stacking direction (horizontal or vertical)
        cam_centers = [
            (camera_interp[idx]["x_center"], camera_interp[idx]["y_center"])
            for idx in sorted(camera_data_dict.keys())
        ]
        x_spread = max(c[0] for c in cam_centers) - min(c[0] for c in cam_centers)
        y_spread = max(c[1] for c in cam_centers) - min(c[1] for c in cam_centers)

        if x_spread >= y_spread:
            stack_direction = "horizontal"
            logger.debug(f"Detected horizontal stacking (x_spread={x_spread:.2f} mm)")
        else:
            stack_direction = "vertical"
            logger.debug(f"Detected vertical stacking (y_spread={y_spread:.2f} mm)")

        # Create weight maps for each camera using distance-based Hanning blend
        logger.debug("Computing blend weights...")
        camera_weights = {}

        for cam_idx in camera_data_dict.keys():
            # Initialize weight to 1 where this camera is valid
            weight = np.where(camera_interp[cam_idx]["valid"], 1.0, 0.0)

            # For overlap regions, use distance-based weighting
            if stack_direction == "horizontal":
                # Weight based on distance from camera center in x-direction
                cam_x_center = camera_interp[cam_idx]["x_center"]
                for other_idx in camera_data_dict.keys():
                    if other_idx == cam_idx:
                        continue

                    # Find overlap region
                    valid_this = camera_interp[cam_idx]["valid"]
                    valid_other = camera_interp[other_idx]["valid"]
                    overlap = valid_this & valid_other

                    if np.any(overlap):
                        # Get x-coordinates of overlap
                        x_overlap = xg[overlap]
                        x_min_overlap = x_overlap.min()
                        x_max_overlap = x_overlap.max()

                        # Determine which camera is left/right
                        other_x_center = camera_interp[other_idx]["x_center"]
                        if cam_x_center < other_x_center:
                            # This camera is on the left - high on left, low on right
                            x_norm = (x_overlap - x_min_overlap) / (
                                x_max_overlap - x_min_overlap
                            )
                            overlap_weight = 0.5 * (1 + np.cos(np.pi * x_norm))
                        else:
                            # This camera is on the right - low on left, high on right
                            x_norm = (x_overlap - x_min_overlap) / (
                                x_max_overlap - x_min_overlap
                            )
                            overlap_weight = 0.5 * (1 - np.cos(np.pi * x_norm))

                        weight[overlap] = overlap_weight
            else:  # vertical
                # Weight based on distance from camera center in y-direction
                cam_y_center = camera_interp[cam_idx]["y_center"]
                for other_idx in camera_data_dict.keys():
                    if other_idx == cam_idx:
                        continue

                    # Find overlap region
                    valid_this = camera_interp[cam_idx]["valid"]
                    valid_other = camera_interp[other_idx]["valid"]
                    overlap = valid_this & valid_other

                    if np.any(overlap):
                        # Get y-coordinates of overlap
                        y_overlap = yg[overlap]
                        y_min_overlap = y_overlap.min()
                        y_max_overlap = y_overlap.max()

                        # Determine which camera is top/bottom
                        other_y_center = camera_interp[other_idx]["y_center"]
                        if cam_y_center < other_y_center:
                            # This camera is on the bottom - high on bottom, low on top
                            y_norm = (y_overlap - y_min_overlap) / (
                                y_max_overlap - y_min_overlap
                            )
                            overlap_weight = 0.5 * (1 + np.cos(np.pi * y_norm))
                        else:
                            # This camera is on the top - low on bottom, high on top
                            y_norm = (y_overlap - y_min_overlap) / (
                                y_max_overlap - y_min_overlap
                            )
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
                valid_total, camera_weights[cam_idx] / total_weight, 0
            )

        # Create merged fields by weighted sum
        logger.debug("Blending cameras...")
        ux_merged = np.zeros_like(xg)
        uy_merged = np.zeros_like(yg)

        for cam_idx in camera_data_dict.keys():
            ux_merged += camera_weights[cam_idx] * np.nan_to_num(
                camera_interp[cam_idx]["ux"], nan=0.0
            )
            uy_merged += camera_weights[cam_idx] * np.nan_to_num(
                camera_interp[cam_idx]["uy"], nan=0.0
            )

        # Set to NaN where no camera has valid data
        no_data = total_weight == 0
        ux_merged[no_data] = np.nan
        uy_merged[no_data] = np.nan

        # uz is not used in 2D PIV
        uz_merged = np.zeros_like(ux_merged)

        return xg, yg, ux_merged, uy_merged, uz_merged

    def merge_single_frame(
        self,
        frame_idx: int,
        valid_runs: list,
    ) -> dict:
        """
        Merge vectors from multiple cameras for a single frame.

        Args:
            frame_idx: Frame index to process
            valid_runs: List of valid run numbers (1-based)

        Returns:
            Dictionary mapping run_num -> merged data dict containing:
            - ux, uy, uz: Velocity components
            - b_mask: Boolean mask
            - x, y: Coordinate grids
        """
        camera_data = {}

        # Load data from each camera
        for camera in self.cameras:
            paths = get_data_paths(
                base_dir=self.base_dir,
                num_frame_pairs=self.num_frame_pairs,
                cam=camera,
                type_name=self.type_name,
                endpoint=self.endpoint,
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

            # Load vector file using centralized utility
            vector_file = data_dir / (self.vector_format % frame_idx)
            if not vector_file.exists():
                logger.warning(f"Vector file does not exist: {vector_file}")
                continue

            try:
                # Load all runs at once - returns shape (R, 3, H, W)
                # where 3 channels are [ux, uy, b_mask]
                all_runs_data = read_mat_contents(str(vector_file), return_all_runs=True)
                camera_data[camera] = {
                    "vector_data": all_runs_data,  # (R, 3, H, W) or object array
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
        merged_runs = {}

        for run_idx, run_num in enumerate(valid_runs):
            logger.debug(f"Processing run {run_num} (index {run_idx})")
            run_data = {}

            for camera, data in camera_data.items():
                vector_data = data["vector_data"]
                array_idx = run_num - 1  # Convert 1-based run to 0-based index

                # Handle both regular arrays and object arrays from read_mat_contents
                if vector_data.dtype == object:
                    # Object array - each element is (3, H, W)
                    if array_idx >= len(vector_data):
                        continue
                    run_arr = vector_data[array_idx]
                    if run_arr.size == 0:
                        continue
                    ux = run_arr[0]
                    uy = run_arr[1]
                    b_mask = run_arr[2].astype(bool)
                    logger.debug(
                        f"Camera {camera}: Loaded from object array, ux.shape={ux.shape}"
                    )
                else:
                    # Regular array shape (R, 3, H, W)
                    if array_idx >= vector_data.shape[0]:
                        continue
                    ux = vector_data[array_idx, 0]
                    uy = vector_data[array_idx, 1]
                    b_mask = vector_data[array_idx, 2].astype(bool)
                    logger.debug(
                        f"Camera {camera}: Loaded from regular array, ux.shape={ux.shape}"
                    )

                # Skip empty runs
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
                    "mask": b_mask,
                }

            # Merge the fields for this run - need at least 2 cameras
            if len(run_data) < 2:
                logger.warning(
                    f"Could not merge run {run_num}: insufficient cameras "
                    f"with valid data (got {len(run_data)}), skipping"
                )
                continue

            # Verify coordinates are not empty
            skip_run = False
            for camera, data in run_data.items():
                if data["x"].size == 0:
                    logger.warning(
                        f"Empty coordinates for run {run_num}, camera {camera}, skipping"
                    )
                    skip_run = True
                    break
            if skip_run:
                continue

            # Merge using Hanning blend algorithm
            logger.debug(f"Merging {len(run_data)} cameras for run {run_num}")
            X_merged, Y_merged, ux_merged, uy_merged, uz_merged = (
                self.merge_n_camera_fields(run_data)
            )

            # Create b_mask (True where data is invalid/NaN)
            b_mask_merged = np.isnan(ux_merged) | np.isnan(uy_merged)

            # Replace NaN with 0 for saving (MATLAB compatibility)
            ux_merged_save = np.nan_to_num(ux_merged, nan=0.0)
            uy_merged_save = np.nan_to_num(uy_merged, nan=0.0)
            uz_merged_save = np.nan_to_num(
                uz_merged if uz_merged is not None else np.zeros_like(ux_merged),
                nan=0.0,
            )

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

    def save_frame_result(
        self,
        frame_idx: int,
        merged_runs: dict,
        total_runs: int,
    ) -> Path:
        """
        Save merged frame result to .mat file.

        Args:
            frame_idx: Frame index
            merged_runs: Dict mapping run_num -> merged data
            total_runs: Total number of runs in file

        Returns:
            Path to saved file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / (self.vector_format % frame_idx)

        # Create piv_result structure preserving run indices
        piv_dtype = np.dtype(
            [("ux", "O"), ("uy", "O"), ("uz", "O"), ("b_mask", "O")]
        )
        piv_result = np.empty(total_runs, dtype=piv_dtype)

        # Fill all runs (0-based array indices)
        for run_idx in range(total_runs):
            run_num = run_idx + 1  # 1-based run number
            if run_num in merged_runs:
                run_data = merged_runs[run_num]
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

        return output_file

    def save_coordinates(self, merged_runs: dict, total_runs: int) -> Path:
        """
        Save merged coordinates to .mat file.

        Args:
            merged_runs: Dict from a merged frame containing coordinate info
            total_runs: Total number of runs

        Returns:
            Path to saved coordinates file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        coords_file = self.output_dir / "coordinates.mat"

        # Create coordinates structure preserving run indices
        coords_dtype = np.dtype([("x", "O"), ("y", "O")])
        coordinates = np.empty(total_runs, dtype=coords_dtype)

        # Fill all runs
        for run_idx in range(total_runs):
            run_num = run_idx + 1
            if run_num in merged_runs:
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

        return coords_file

    def merge_all_frames(
        self,
        progress_callback: Optional[Callable[[dict], None]] = None,
        max_workers: Optional[int] = None,
    ) -> dict:
        """
        Process all frames with multiprocessing support.

        Args:
            progress_callback: Optional callback receiving dict with:
                - progress: int (0-100)
                - processed_frames: int
                - total_frames: int
                - message: str
            max_workers: Maximum number of parallel workers (default: from config)

        Returns:
            dict with:
                - success: bool
                - processed_count: int
                - output_dir: str
                - valid_runs: list
                - error: str (if failed)
        """
        # Read max_workers from config if not provided
        if max_workers is None:
            cfg = get_config()
            max_workers = cfg.merging_max_workers

        # Find valid runs
        valid_runs, total_runs = self.find_valid_runs()

        if not valid_runs:
            return {
                "success": False,
                "error": "No valid runs found in vector files",
            }

        logger.info(
            f"Found {len(valid_runs)} valid runs: {valid_runs} (total: {total_runs})"
        )

        # Report initial progress
        if progress_callback:
            progress_callback({
                "progress": 2,
                "processed_frames": 0,
                "total_frames": self.num_frame_pairs,
                "message": "Initializing merge operation...",
            })

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        # Limit workers
        max_workers = min(os.cpu_count() or 4, max_workers, 8)

        # Prepare arguments for all frames
        frame_args = [
            (
                frame_idx,
                str(self.base_dir),
                self.cameras,
                self.type_name,
                self.endpoint,
                self.num_frame_pairs,
                self.vector_format,
                valid_runs,
                total_runs,
            )
            for frame_idx in range(1, self.num_frame_pairs + 1)
        ]

        # Process frames in parallel
        processed_count = 0
        last_merged_runs = None

        if progress_callback:
            progress_callback({
                "progress": 5,
                "processed_frames": 0,
                "total_frames": self.num_frame_pairs,
                "message": f"Merging with {max_workers} workers...",
            })

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_process_frame_worker, args)
                    for args in frame_args
                ]

                for future in as_completed(futures):
                    frame_idx, success, merged_runs = future.result()
                    processed_count += 1

                    if success and merged_runs:
                        last_merged_runs = merged_runs

                    # Update progress
                    if progress_callback:
                        progress = int(
                            (processed_count / self.num_frame_pairs) * 90
                        ) + 5
                        progress_callback({
                            "progress": min(progress, 95),
                            "processed_frames": processed_count,
                            "total_frames": self.num_frame_pairs,
                            "message": f"Merged {processed_count}/{self.num_frame_pairs} frames",
                        })

                    if processed_count % 10 == 0:
                        logger.info(
                            f"Merged {processed_count}/{self.num_frame_pairs} frames"
                        )

            # Save coordinates
            if last_merged_runs:
                self.save_coordinates(last_merged_runs, total_runs)

            if progress_callback:
                progress_callback({
                    "progress": 100,
                    "processed_frames": processed_count,
                    "total_frames": self.num_frame_pairs,
                    "message": f"Complete: merged {processed_count} frames",
                })

            return {
                "success": True,
                "processed_count": processed_count,
                "output_dir": str(self.output_dir),
                "valid_runs": valid_runs,
            }

        except Exception as e:
            logger.error(f"Error in merge operation: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    def run(self) -> dict:
        """
        Run complete merge operation (CLI-friendly entry point).

        Returns:
            dict with success status and results
        """
        logger.info(
            f"Starting vector merge for cameras {self.cameras}, "
            f"{self.num_frame_pairs} frames"
        )

        result = self.merge_all_frames()

        if result["success"]:
            logger.info(
                f"Merge complete: {result['processed_count']} frames "
                f"saved to {result['output_dir']}"
            )
        else:
            logger.error(f"Merge failed: {result.get('error', 'Unknown error')}")

        return result


# CLI entry point for standalone usage
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Vector Merging - Starting")
    logger.info("=" * 60)

    if USE_CONFIG_DIRECTLY:
        # Load settings directly from existing config.yaml
        logger.info("Loading settings directly from config.yaml (USE_CONFIG_DIRECTLY=True)")
        config = get_config()

        # Extract settings from config (all from merging section)
        base_dir = Path(config.base_paths[0])
        cameras = config.merging_cameras
        type_name = config.merging_type_name
        endpoint = config.merging_endpoint
        max_workers = config.merging_max_workers
    else:
        # Apply CLI settings to config.yaml so centralized loaders work correctly
        config = apply_cli_settings_to_config()

        # Use hardcoded settings
        base_dir = Path(BASE_DIR)
        cameras = CAMERAS
        type_name = TYPE_NAME
        endpoint = ENDPOINT
        max_workers = MAX_WORKERS

    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Cameras: {cameras}")
    logger.info(f"Type: {type_name}")
    logger.info(f"Max workers: {max_workers}")

    # Run merging
    merger = VectorMerger(
        base_dir=base_dir,
        cameras=cameras,
        type_name=type_name,
        endpoint=endpoint,
    )

    result = merger.merge_all_frames(max_workers=max_workers)

    logger.info("=" * 60)
    if result["success"]:
        logger.info(f"Merge complete: {result['processed_count']} frames")
        logger.info(f"Output: {result['output_dir']}")
        logger.info("Vector merging completed successfully")
    else:
        logger.error(f"Merge failed: {result.get('error', 'Unknown error')}")

    exit(0 if result["success"] else 1)
