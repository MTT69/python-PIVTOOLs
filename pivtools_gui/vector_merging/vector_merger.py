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
from scipy.ndimage import distance_transform_edt

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
        args: Tuple containing pre-computed paths and parameters

    Returns:
        Tuple of (frame_idx, success, merged_runs_dict)
    """
    (
        frame_idx,
        camera_paths,
        valid_runs,
        total_runs,
        is_ensemble,
        vector_format,
        output_dir,
    ) = args

    # Call the static method directly - no VectorMerger instantiation needed
    return VectorMerger.process_frame_static(
        frame_idx=frame_idx,
        camera_paths=camera_paths,
        valid_runs=valid_runs,
        total_runs=total_runs,
        is_ensemble=is_ensemble,
        vector_format=vector_format,
        output_dir=Path(output_dir),
    )


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

    @property
    def is_ensemble(self) -> bool:
        """Check if this merger is processing ensemble data."""
        return self.type_name == "ensemble"

    @staticmethod
    def _get_vector_file_path_static(
        data_dir: Path, frame_idx: int, is_ensemble: bool, vector_format: str
    ) -> Path:
        """
        Get the correct vector file path based on type_name.

        For ensemble data, returns ensemble_result.mat.
        For instantaneous data, returns the frame-numbered file.
        """
        if is_ensemble:
            return data_dir / "ensemble_result.mat"
        else:
            return data_dir / (vector_format % frame_idx)

    def _get_vector_file_path(self, data_dir: Path, frame_idx: int) -> Path:
        """Instance method wrapper for backward compatibility."""
        return self._get_vector_file_path_static(
            data_dir, frame_idx, self.is_ensemble, self.vector_format
        )

    @staticmethod
    def _get_result_key_static(is_ensemble: bool) -> str:
        """Get the key name for the result based on type_name."""
        if is_ensemble:
            return "ensemble_result"
        else:
            return "piv_result"

    def _get_result_key(self) -> str:
        """Instance method wrapper for backward compatibility."""
        return self._get_result_key_static(self.is_ensemble)

    @staticmethod
    def _load_ensemble_run_data_static(
        mat_file: Path, run_idx: int, result_key: str
    ) -> Optional[dict]:
        """
        Load ensemble data for a specific run, including stress fields.

        Args:
            mat_file: Path to ensemble_result.mat
            run_idx: 0-based run index
            result_key: Key name in the mat file ('ensemble_result' or 'piv_result')

        Returns:
            Dict with ux, uy, b_mask, and optional stress fields, or None if invalid
        """
        mat = scipy.io.loadmat(str(mat_file), struct_as_record=False, squeeze_me=True)

        if result_key not in mat:
            logger.warning(f"'{result_key}' not found in {mat_file}")
            return None

        result = mat[result_key]

        # Handle array of structs
        if isinstance(result, np.ndarray) and result.dtype == object:
            if run_idx >= result.size:
                return None
            run = result[run_idx]
        else:
            if run_idx != 0:
                return None
            run = result

        # Extract required fields
        ux = getattr(run, "ux", None)
        uy = getattr(run, "uy", None)
        b_mask = getattr(run, "b_mask", None)

        if ux is None or uy is None:
            return None

        ux = np.asarray(ux)
        uy = np.asarray(uy)

        if ux.size == 0 or uy.size == 0:
            return None

        if b_mask is None:
            b_mask = np.zeros_like(ux, dtype=bool)
        else:
            b_mask = np.asarray(b_mask).astype(bool)

        data = {
            "ux": ux,
            "uy": uy,
            "b_mask": b_mask,
        }

        # Extract stress fields if present (ensemble data)
        for stress_field in ["UU_stress", "VV_stress", "UV_stress"]:
            val = getattr(run, stress_field, None)
            if val is not None:
                data[stress_field] = np.asarray(val)

        return data

    def _load_ensemble_run_data(self, mat_file: Path, run_idx: int) -> Optional[dict]:
        """Instance method wrapper for backward compatibility."""
        return self._load_ensemble_run_data_static(
            mat_file, run_idx, self._get_result_key()
        )

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

        # Use helper to get correct file path for ensemble vs instantaneous
        first_file = self._get_vector_file_path(data_dir, 1)
        if not first_file.exists():
            return [], 0

        try:
            result = find_valid_piv_runs(
                first_file,
                one_based=True,
                result_key=self._get_result_key(),
            )
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
                method="cubic",
                bounds_error=False,
                fill_value=np.nan,
            )
            interp_uy = RegularGridInterpolator(
                (y_vec, x_vec),
                valid_uy,
                method="cubic",
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

        # Create weight maps using Tukey window based on edge distance
        # This approach handles any camera arrangement (horizontal, vertical, or 2D grid)
        logger.debug("Computing Tukey window blend weights...")
        camera_weights = {}

        # Tukey window alpha parameter (0.5 = half cosine taper at edges)
        tukey_alpha = 0.5

        for cam_idx in camera_data_dict.keys():
            valid_mask = camera_interp[cam_idx]["valid"]

            # Compute distance from edge of valid region (in pixels)
            # distance_transform_edt gives distance to nearest False pixel
            edge_distance = distance_transform_edt(valid_mask)

            # Find maximum distance (center of valid region)
            max_dist = edge_distance.max()

            if max_dist > 0:
                # Normalize distance to [0, 1] where 0=edge, 1=center
                norm_dist = edge_distance / max_dist

                # Apply Tukey window: flat in center, cosine taper at edges
                # For pixels within the taper region (norm_dist < alpha/2),
                # weight ramps from 0 at edge to 1 at alpha/2
                taper_region = norm_dist < (tukey_alpha / 2)
                weight = np.ones_like(norm_dist)
                # Tukey taper: 0.5 * (1 - cos(2*pi*x/alpha)) for x in [0, alpha/2]
                weight[taper_region] = 0.5 * (
                    1 - np.cos(2 * np.pi * norm_dist[taper_region] / tukey_alpha)
                )
            else:
                # Single pixel or no valid data - uniform weight
                weight = np.where(valid_mask, 1.0, 0.0)

            # Zero weight where camera has no valid data
            weight = np.where(valid_mask, weight, 0.0)
            camera_weights[cam_idx] = weight

            logger.debug(
                f"Camera {cam_idx}: max_edge_dist={max_dist:.1f}px, "
                f"weight range=[{weight[valid_mask].min():.3f}, {weight[valid_mask].max():.3f}]"
            )

        # Normalize weights so they sum to 1 at each point (weighted accumulation)
        total_weight = np.zeros_like(xg)
        for cam_idx in camera_data_dict.keys():
            total_weight += camera_weights[cam_idx]

        for cam_idx in camera_data_dict.keys():
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

    @staticmethod
    def process_frame_static(
        frame_idx: int,
        camera_paths: dict,
        valid_runs: list,
        total_runs: int,
        is_ensemble: bool,
        vector_format: str,
        output_dir: Path,
    ) -> tuple:
        """
        Process a single frame without instantiating VectorMerger.

        This static method accepts pre-computed paths and coordinates,
        avoiding filesystem scans and config parsing for each frame.

        Args:
            frame_idx: Frame index to process
            camera_paths: Dict mapping camera_num -> {
                'data_dir': Path to data directory,
                'coords_x': List of x coordinate arrays per run,
                'coords_y': List of y coordinate arrays per run,
            }
            valid_runs: List of valid run numbers (1-based)
            total_runs: Total number of runs in file
            is_ensemble: Whether this is ensemble data
            vector_format: Format string for vector filenames (e.g., "%05d.mat")
            output_dir: Directory to save output files

        Returns:
            Tuple of (frame_idx, success, merged_runs_dict)
        """
        result_key = VectorMerger._get_result_key_static(is_ensemble)

        try:
            camera_data = {}

            # Load vector data from each camera (coordinates already pre-loaded)
            for camera, paths_info in camera_paths.items():
                data_dir = paths_info["data_dir"]

                # Get vector file path
                vector_file = VectorMerger._get_vector_file_path_static(
                    data_dir, frame_idx, is_ensemble, vector_format
                )
                if not vector_file.exists():
                    logger.warning(f"Vector file does not exist: {vector_file}")
                    continue

                try:
                    if is_ensemble:
                        camera_data[camera] = {
                            "vector_file": vector_file,
                            "coords_x": paths_info["coords_x"],
                            "coords_y": paths_info["coords_y"],
                            "is_ensemble": True,
                        }
                    else:
                        all_runs_data = read_mat_contents(
                            str(vector_file),
                            return_all_runs=True,
                            var_name=result_key,
                        )
                        camera_data[camera] = {
                            "vector_data": all_runs_data,
                            "coords_x": paths_info["coords_x"],
                            "coords_y": paths_info["coords_y"],
                            "is_ensemble": False,
                        }
                except Exception as e:
                    logger.error(f"Failed to load vector file {vector_file}: {e}")
                    continue

            if len(camera_data) < 2:
                logger.warning(
                    f"Frame {frame_idx}: Need at least 2 cameras, got {len(camera_data)}"
                )
                return frame_idx, False, None

            # Merge data for each run
            merged_runs = {}

            for run_idx, run_num in enumerate(valid_runs):
                run_data = {}
                run_stress_data = {}

                for camera, data in camera_data.items():
                    array_idx = run_num - 1

                    if data.get("is_ensemble"):
                        vec_data = VectorMerger._load_ensemble_run_data_static(
                            data["vector_file"], array_idx, result_key
                        )
                        if vec_data is None:
                            continue

                        ux = vec_data["ux"]
                        uy = vec_data["uy"]
                        b_mask = vec_data["b_mask"]

                        stress_fields = {}
                        for sf in ["UU_stress", "VV_stress", "UV_stress"]:
                            if sf in vec_data:
                                stress_fields[sf] = vec_data[sf]
                    else:
                        vector_data = data["vector_data"]

                        if vector_data.dtype == object:
                            if array_idx >= len(vector_data):
                                continue
                            run_arr = vector_data[array_idx]
                            if run_arr.size == 0:
                                continue
                            ux = run_arr[0]
                            uy = run_arr[1]
                            b_mask = run_arr[2].astype(bool)
                        else:
                            if array_idx >= vector_data.shape[0]:
                                continue
                            ux = vector_data[array_idx, 0]
                            uy = vector_data[array_idx, 1]
                            b_mask = vector_data[array_idx, 2].astype(bool)

                        stress_fields = {}

                    if ux.size == 0 or uy.size == 0:
                        continue

                    ux_masked = np.where(b_mask, np.nan, ux)
                    uy_masked = np.where(b_mask, np.nan, uy)

                    x_coords = data["coords_x"][run_idx]
                    y_coords = data["coords_y"][run_idx]

                    run_data[camera] = {
                        "x": x_coords,
                        "y": y_coords,
                        "ux": ux_masked,
                        "uy": uy_masked,
                        "mask": b_mask,
                    }

                    if stress_fields:
                        run_stress_data[camera] = {
                            "x": x_coords,
                            "y": y_coords,
                            "mask": b_mask,
                            **{sf: np.where(b_mask, np.nan, v) for sf, v in stress_fields.items()},
                        }

                if len(run_data) < 2:
                    continue

                skip_run = False
                for camera, data in run_data.items():
                    if data["x"].size == 0:
                        skip_run = True
                        break
                if skip_run:
                    continue

                X_merged, Y_merged, ux_merged, uy_merged, uz_merged = (
                    VectorMerger.merge_n_camera_fields(run_data)
                )

                b_mask_merged = np.isnan(ux_merged) | np.isnan(uy_merged)

                ux_merged_save = np.nan_to_num(ux_merged, nan=0.0)[::-1, :]
                uy_merged_save = np.nan_to_num(uy_merged, nan=0.0)[::-1, :]
                uz_merged_save = np.nan_to_num(
                    uz_merged if uz_merged is not None else np.zeros_like(ux_merged),
                    nan=0.0,
                )[::-1, :]
                b_mask_merged = b_mask_merged[::-1, :]
                Y_merged = Y_merged[::-1, :]

                merged_runs[run_num] = {
                    "ux": ux_merged_save,
                    "uy": uy_merged_save,
                    "uz": uz_merged_save,
                    "b_mask": b_mask_merged.astype(np.uint8),
                    "x": X_merged,
                    "y": Y_merged,
                }

                # Merge stress fields if present
                if run_stress_data and len(run_stress_data) >= 2:
                    for stress_name in ["UU_stress", "VV_stress", "UV_stress"]:
                        stress_camera_data = {}
                        for camera, sdata in run_stress_data.items():
                            if stress_name in sdata:
                                stress_camera_data[camera] = {
                                    "x": sdata["x"],
                                    "y": sdata["y"],
                                    "ux": sdata[stress_name],
                                    "uy": np.zeros_like(sdata[stress_name]),
                                    "mask": sdata["mask"],
                                }

                        if len(stress_camera_data) >= 2:
                            _, _, stress_merged, _, _ = VectorMerger.merge_n_camera_fields(
                                stress_camera_data
                            )
                            stress_merged_save = np.nan_to_num(stress_merged, nan=0.0)[::-1, :]
                            merged_runs[run_num][stress_name] = stress_merged_save

            if not merged_runs:
                return frame_idx, False, None

            # Save the result
            VectorMerger.save_frame_result_static(
                frame_idx, merged_runs, total_runs, output_dir, is_ensemble, vector_format
            )

            return frame_idx, True, merged_runs

        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {e}", exc_info=True)
            return frame_idx, False, None

    @staticmethod
    def save_frame_result_static(
        frame_idx: int,
        merged_runs: dict,
        total_runs: int,
        output_dir: Path,
        is_ensemble: bool,
        vector_format: str,
    ) -> Path:
        """
        Save merged frame result to .mat file (static version).

        Args:
            frame_idx: Frame index
            merged_runs: Dict mapping run_num -> merged data
            total_runs: Total number of runs in file
            output_dir: Directory to save to
            is_ensemble: Whether this is ensemble data
            vector_format: Format string for vector filenames

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = VectorMerger._get_vector_file_path_static(
            output_dir, frame_idx, is_ensemble, vector_format
        )
        result_key = VectorMerger._get_result_key_static(is_ensemble)

        has_stresses = any("UU_stress" in run_data for run_data in merged_runs.values())

        if has_stresses:
            piv_dtype = np.dtype([
                ("ux", "O"), ("uy", "O"), ("uz", "O"), ("b_mask", "O"),
                ("UU_stress", "O"), ("VV_stress", "O"), ("UV_stress", "O"),
            ])
        else:
            piv_dtype = np.dtype([("ux", "O"), ("uy", "O"), ("uz", "O"), ("b_mask", "O")])

        piv_result = np.empty(total_runs, dtype=piv_dtype)

        for run_idx in range(total_runs):
            run_num = run_idx + 1
            if run_num in merged_runs:
                run_data = merged_runs[run_num]
                piv_result[run_idx]["ux"] = run_data["ux"]
                piv_result[run_idx]["uy"] = run_data["uy"]
                piv_result[run_idx]["uz"] = run_data["uz"]
                piv_result[run_idx]["b_mask"] = run_data["b_mask"]

                if has_stresses:
                    piv_result[run_idx]["UU_stress"] = run_data.get("UU_stress", np.array([]))
                    piv_result[run_idx]["VV_stress"] = run_data.get("VV_stress", np.array([]))
                    piv_result[run_idx]["UV_stress"] = run_data.get("UV_stress", np.array([]))
            else:
                piv_result[run_idx]["ux"] = np.array([])
                piv_result[run_idx]["uy"] = np.array([])
                piv_result[run_idx]["uz"] = np.array([])
                piv_result[run_idx]["b_mask"] = np.array([])

                if has_stresses:
                    piv_result[run_idx]["UU_stress"] = np.array([])
                    piv_result[run_idx]["VV_stress"] = np.array([])
                    piv_result[run_idx]["UV_stress"] = np.array([])

        scipy.io.savemat(str(output_file), {result_key: piv_result}, do_compression=True)

        return output_file

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

            # Load vector file
            vector_file = self._get_vector_file_path(data_dir, frame_idx)
            if not vector_file.exists():
                logger.warning(f"Vector file does not exist: {vector_file}")
                continue

            try:
                if self.is_ensemble:
                    # Ensemble: store file path for on-demand loading with stress fields
                    camera_data[camera] = {
                        "vector_file": vector_file,
                        "coords_x": coords_x_list,
                        "coords_y": coords_y_list,
                        "is_ensemble": True,
                    }
                else:
                    # Instantaneous: load all runs at once - returns shape (R, 3, H, W)
                    all_runs_data = read_mat_contents(
                        str(vector_file),
                        return_all_runs=True,
                        var_name=self._get_result_key(),
                    )
                    camera_data[camera] = {
                        "vector_data": all_runs_data,
                        "coords_x": coords_x_list,
                        "coords_y": coords_y_list,
                        "is_ensemble": False,
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
        has_stress_fields = False  # Track if ensemble with stress data

        for run_idx, run_num in enumerate(valid_runs):
            logger.debug(f"Processing run {run_num} (index {run_idx})")
            run_data = {}
            run_stress_data = {}  # Stress fields for ensemble

            for camera, data in camera_data.items():
                array_idx = run_num - 1  # Convert 1-based run to 0-based index

                if data.get("is_ensemble"):
                    # Ensemble mode: load with stress fields
                    vec_data = self._load_ensemble_run_data(data["vector_file"], array_idx)
                    if vec_data is None:
                        continue

                    ux = vec_data["ux"]
                    uy = vec_data["uy"]
                    b_mask = vec_data["b_mask"]

                    # Extract stress fields if present
                    stress_fields = {}
                    for sf in ["UU_stress", "VV_stress", "UV_stress"]:
                        if sf in vec_data:
                            stress_fields[sf] = vec_data[sf]
                            has_stress_fields = True

                    logger.debug(
                        f"Camera {camera}: Loaded ensemble data, ux.shape={ux.shape}, "
                        f"stress fields: {list(stress_fields.keys())}"
                    )
                else:
                    # Instantaneous mode: use preloaded data
                    vector_data = data["vector_data"]

                    # Handle both regular arrays and object arrays from read_mat_contents
                    if vector_data.dtype == object:
                        if array_idx >= len(vector_data):
                            continue
                        run_arr = vector_data[array_idx]
                        if run_arr.size == 0:
                            continue
                        ux = run_arr[0]
                        uy = run_arr[1]
                        b_mask = run_arr[2].astype(bool)
                    else:
                        if array_idx >= vector_data.shape[0]:
                            continue
                        ux = vector_data[array_idx, 0]
                        uy = vector_data[array_idx, 1]
                        b_mask = vector_data[array_idx, 2].astype(bool)

                    stress_fields = {}
                    logger.debug(
                        f"Camera {camera}: Loaded instantaneous data, ux.shape={ux.shape}"
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

                # Store stress fields if present
                if stress_fields:
                    run_stress_data[camera] = {
                        "x": x_coords,
                        "y": y_coords,
                        "mask": b_mask,
                        **{sf: np.where(b_mask, np.nan, v) for sf, v in stress_fields.items()},
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

            # Merge stress fields if present (ensemble data)
            if run_stress_data and len(run_stress_data) >= 2:
                logger.debug(f"Merging stress fields for run {run_num}")

                # Use the same merging approach for each stress field
                for stress_name in ["UU_stress", "VV_stress", "UV_stress"]:
                    # Build camera data dict for this stress field (treating it as ux)
                    stress_camera_data = {}
                    for camera, sdata in run_stress_data.items():
                        if stress_name in sdata:
                            stress_camera_data[camera] = {
                                "x": sdata["x"],
                                "y": sdata["y"],
                                "ux": sdata[stress_name],  # Use stress as ux
                                "uy": np.zeros_like(sdata[stress_name]),  # Dummy uy
                                "mask": sdata["mask"],
                            }

                    if len(stress_camera_data) >= 2:
                        _, _, stress_merged, _, _ = self.merge_n_camera_fields(stress_camera_data)
                        stress_merged_save = np.nan_to_num(stress_merged, nan=0.0)
                        stress_merged_save = stress_merged_save[::-1, :]  # Flip
                        merged_runs[run_num][stress_name] = stress_merged_save

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
        output_file = self._get_vector_file_path(self.output_dir, frame_idx)
        result_key = self._get_result_key()

        # Check if stress fields are present (ensemble data)
        has_stresses = any(
            "UU_stress" in run_data
            for run_data in merged_runs.values()
        )

        # Create result structure with appropriate dtype
        if has_stresses:
            piv_dtype = np.dtype([
                ("ux", "O"), ("uy", "O"), ("uz", "O"), ("b_mask", "O"),
                ("UU_stress", "O"), ("VV_stress", "O"), ("UV_stress", "O"),
            ])
        else:
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

                if has_stresses:
                    piv_result[run_idx]["UU_stress"] = run_data.get("UU_stress", np.array([]))
                    piv_result[run_idx]["VV_stress"] = run_data.get("VV_stress", np.array([]))
                    piv_result[run_idx]["UV_stress"] = run_data.get("UV_stress", np.array([]))
            else:
                # Empty run - preserve structure
                piv_result[run_idx]["ux"] = np.array([])
                piv_result[run_idx]["uy"] = np.array([])
                piv_result[run_idx]["uz"] = np.array([])
                piv_result[run_idx]["b_mask"] = np.array([])

                if has_stresses:
                    piv_result[run_idx]["UU_stress"] = np.array([])
                    piv_result[run_idx]["VV_stress"] = np.array([])
                    piv_result[run_idx]["UV_stress"] = np.array([])

        scipy.io.savemat(
            str(output_file),
            {result_key: piv_result},
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
        max_workers: int = 8,
    ) -> dict:
        """
        Process all frames with multiprocessing support.

        Args:
            progress_callback: Optional callback receiving dict with:
                - progress: int (0-100)
                - processed_frames: int
                - total_frames: int
                - message: str
            max_workers: Maximum number of parallel workers (default: 8)

        Returns:
            dict with:
                - success: bool
                - processed_count: int
                - output_dir: str
                - valid_runs: list
                - error: str (if failed)
        """
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

        # For ensemble mode, there's only one file to process (ensemble_result.mat)
        # For instantaneous mode, process num_frame_pairs files (00001.mat, etc.)
        num_files_to_process = 1 if self.is_ensemble else self.num_frame_pairs

        # Report initial progress
        if progress_callback:
            progress_callback({
                "progress": 2,
                "processed_frames": 0,
                "total_frames": num_files_to_process,
                "message": "Initializing merge operation...",
            })

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        # Limit workers
        max_workers = min(os.cpu_count() or 4, max_workers, 8)

        # Pre-compute camera paths and coordinates ONCE (performance optimization)
        # This avoids calling get_data_paths() and load_coords_from_directory()
        # for every frame in every worker process
        logger.info("Pre-computing camera paths and loading coordinates...")
        camera_paths = {}
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

            try:
                coords_x_list, coords_y_list = load_coords_from_directory(
                    data_dir, runs=valid_runs
                )
                camera_paths[camera] = {
                    "data_dir": data_dir,
                    "coords_x": coords_x_list,
                    "coords_y": coords_y_list,
                }
                logger.debug(f"Camera {camera}: loaded {len(coords_x_list)} coordinate sets")
            except Exception as e:
                logger.error(f"Failed to load coordinates for camera {camera}: {e}")
                continue

        if len(camera_paths) < 2:
            return {
                "success": False,
                "error": f"Need at least 2 cameras with data, only found {len(camera_paths)}",
            }

        # Prepare arguments for all frames (or single file for ensemble)
        frame_indices = [1] if self.is_ensemble else list(range(1, self.num_frame_pairs + 1))
        frame_args = [
            (
                frame_idx,
                camera_paths,
                valid_runs,
                total_runs,
                self.is_ensemble,
                self.vector_format,
                str(self.output_dir),
            )
            for frame_idx in frame_indices
        ]

        # Process frames in parallel
        processed_count = 0
        last_merged_runs = None

        if progress_callback:
            progress_callback({
                "progress": 5,
                "processed_frames": 0,
                "total_frames": num_files_to_process,
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
                            (processed_count / num_files_to_process) * 90
                        ) + 5
                        progress_callback({
                            "progress": min(progress, 95),
                            "processed_frames": processed_count,
                            "total_frames": num_files_to_process,
                            "message": f"Merged {processed_count}/{num_files_to_process} files",
                        })

                    if processed_count % 10 == 0:
                        logger.info(
                            f"Merged {processed_count}/{num_files_to_process} files"
                        )

            # Save coordinates
            if last_merged_runs:
                self.save_coordinates(last_merged_runs, total_runs)

            if progress_callback:
                progress_callback({
                    "progress": 100,
                    "processed_frames": processed_count,
                    "total_frames": num_files_to_process,
                    "message": f"Complete: merged {processed_count} files",
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
        cameras = config.merging_cameras
        type_name = config.merging_type_name

        # Loop through active_paths for batch processing
        active_paths = config.active_paths
        logger.info(f"Processing {len(active_paths)} active path(s)")
        logger.info(f"Cameras: {cameras}")
        logger.info(f"Type: {type_name}")

        results = []
        for path_idx in active_paths:
            base_dir = Path(config.base_paths[path_idx])
            logger.info("")
            logger.info("=" * 40)
            logger.info(f"Path {path_idx + 1}/{len(active_paths)}: {base_dir}")
            logger.info("=" * 40)

            # Run merging for this path
            merger = VectorMerger(
                base_dir=base_dir,
                cameras=cameras,
                type_name=type_name,
            )

            result = merger.merge_all_frames()
            result["path_idx"] = path_idx
            result["base_dir"] = str(base_dir)
            results.append(result)

            if result["success"]:
                logger.info(f"Path {path_idx + 1}: Merged {result['processed_count']} frames")
            else:
                logger.error(f"Path {path_idx + 1}: FAILED - {result.get('error')}")

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        success_count = sum(1 for r in results if r["success"])
        for r in results:
            status = "OK" if r["success"] else "FAILED"
            logger.info(f"  Path {r['path_idx'] + 1}: {status}")
        logger.info(f"Total: {success_count}/{len(results)} paths succeeded")

        all_success = all(r["success"] for r in results)
        exit(0 if all_success else 1)
    else:
        # Apply CLI settings to config.yaml so centralized loaders work correctly
        config = apply_cli_settings_to_config()

        # Use hardcoded settings (single path)
        base_dir = Path(BASE_DIR)
        cameras = CAMERAS
        type_name = TYPE_NAME

        logger.info(f"Base directory: {base_dir}")
        logger.info(f"Cameras: {cameras}")
        logger.info(f"Type: {type_name}")

        # Run merging
        merger = VectorMerger(
            base_dir=base_dir,
            cameras=cameras,
            type_name=type_name,
        )

        result = merger.merge_all_frames()

        logger.info("=" * 60)
        if result["success"]:
            logger.info(f"Merge complete: {result['processed_count']} frames")
            logger.info(f"Output: {result['output_dir']}")
            logger.info("Vector merging completed successfully")
        else:
            logger.error(f"Merge failed: {result.get('error', 'Unknown error')}")

        exit(0 if result["success"] else 1)
