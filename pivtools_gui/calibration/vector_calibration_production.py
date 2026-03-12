#!/usr/bin/env python3
"""
vector_calibration_production.py

Production script for calibrating uncalibrated PIV vectors to physical units (m/s).
Converts pixel-based vectors to physical velocities using pinhole camera calibration models.
Supports both ChArUco and Planar (circle grid) calibration models.
"""

import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.io import loadmat, savemat

sys.path.append(str(Path(__file__).parent.parent))
from pivtools_core.config import get_config, reload_config
from pivtools_core.coordinate_utils import extract_coordinates, get_num_coordinate_runs
from pivtools_core.paths import get_data_paths
from pivtools_core.vector_loading import load_coords_from_directory, read_mat_contents
from pivtools_gui.utils.worker_pool import worker_initializer, get_max_workers

# ===================== CONFIGURATION VARIABLES =====================
# Set these variables for your calibration setup.
# These will be written to config.yaml before processing.
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/planar_images/test"
NUM_FRAME_PAIRS = 100  # Number of frame pairs (from config.yaml images.num_frame_pairs)
DT_SECONDS = 0.0057553  # Time step between frames in seconds
CAMERA_NUMS = [1]  # List of camera numbers to process (1-based), e.g. [1, 2] for stereo
MODEL_TYPE = "charuco"  # "charuco" or "dotboard" - sets calibration.active
VECTOR_PATTERN = "%05d.mat"  # Pattern for vector files (e.g. "B%05d.mat", "%05d.mat")
TYPE_NAME = "instantaneous"  # Type name for data directory (e.g. "instantaneous", "ensemble")
RUNS_TO_PROCESS = None  # List of 1-indexed runs to process, or None for all (e.g. [1, 2, 3])

# USE_CONFIG_DIRECTLY: If True, skip updating config.yaml with above parameters
# and load calibration settings directly from the existing config.yaml
USE_CONFIG_DIRECTLY = True
# ===================================================================


def apply_cli_settings_to_config():
    """Update config.yaml with CLI-mode hardcoded settings.

    This function writes the hardcoded configuration variables to config.yaml,
    ensuring the centralized paths and calibration systems use the correct settings.

    Returns
    -------
    Config
        The reloaded config object with updated settings
    """
    config = get_config()

    # Paths
    config.data["paths"]["base_paths"] = [BASE_DIR]
    config.data["paths"]["camera_numbers"] = CAMERA_NUMS
    config.data["paths"]["camera_count"] = len(CAMERA_NUMS)

    # Images
    config.data["images"]["num_frame_pairs"] = NUM_FRAME_PAIRS
    config.data["images"]["vector_format"] = [VECTOR_PATTERN]

    # Calibration - set active method based on MODEL_TYPE
    if MODEL_TYPE.lower() == "charuco":
        config.data["calibration"]["active"] = "charuco"
        config.data["calibration"]["charuco"]["dt"] = DT_SECONDS
    elif MODEL_TYPE.lower() == "dotboard":
        config.data["calibration"]["active"] = "dotboard"
        config.data["calibration"]["dotboard"]["dt"] = DT_SECONDS
    else:
        raise ValueError(f"MODEL_TYPE must be 'charuco' or 'dotboard', got '{MODEL_TYPE}'")

    # Save to disk so centralized systems pick up changes
    config.save()
    logger.info("Updated config.yaml with CLI settings")

    # Reload to ensure fresh state
    return reload_config()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _pixels_to_world_mm(
    pts_px: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
) -> np.ndarray:
    """
    Convert pixel coordinates to world coordinates (mm) on a plane.

    Uses the pinhole camera model with distortion correction to project
    pixel coordinates back to a world plane. The plane is defined by:
        Z = z_world + X * tan(tilt_y) + Y * tan(tilt_x)

    When z_world=0, tilt_x=0, tilt_y=0, this reduces to the Z=0 plane.

    Args:
        pts_px: Pixel coordinates, shape (N, 2)
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvec: Rotation vector (3,)
        tvec: Translation vector (3,)
        z_world: Z-offset of the plane from the calibration plane (mm)
        tilt_x: Tilt angle about the X-axis (radians)
        tilt_y: Tilt angle about the Y-axis (radians)

    Returns:
        World coordinates (mm) on the specified plane, shape (N, 2)
    """
    if pts_px.size == 0:
        return pts_px

    # Undistort points to normalized camera coordinates
    # Without P matrix, returns normalized coordinates (x/z, y/z) in camera frame
    pts_normalized = cv2.undistortPoints(
        pts_px.reshape(-1, 1, 2).astype(np.float32),
        camera_matrix,
        dist_coeffs,
        P=None,  # No rectification, get normalized coords
    ).reshape(-1, 2)

    # Build rotation matrix from rvec
    R, _ = cv2.Rodrigues(rvec)

    # Ray-plane intersection for plane: Z = z_world + X*tan(tilt_y) + Y*tan(tilt_x)
    # Camera ray: [x_norm, y_norm, 1] (normalized coords with z=1)
    # World point: P_world = R^T @ (s * ray - t) = s * ray_world - t_world
    # Plane equation: P_world[2] = z_world + P_world[0]*tan_ty + P_world[1]*tan_tx
    # Substituting and solving for s:
    #   s*rw[2] - tw[2] = z_world + (s*rw[0] - tw[0])*tan_ty + (s*rw[1] - tw[1])*tan_tx
    #   s*(rw[2] - rw[0]*tan_ty - rw[1]*tan_tx) = z_world + tw[2] - tw[0]*tan_ty - tw[1]*tan_tx

    R_inv = R.T
    t = tvec.flatten()

    tan_tx = math.tan(tilt_x)
    tan_ty = math.tan(tilt_y)

    world_pts = np.zeros((pts_normalized.shape[0], 2), dtype=np.float64)

    for i, (xn, yn) in enumerate(pts_normalized):
        ray = np.array([xn, yn, 1.0])

        ray_world = R_inv @ ray
        t_world = R_inv @ t

        denom = ray_world[2] - ray_world[0] * tan_ty - ray_world[1] * tan_tx
        if abs(denom) < 1e-10:
            world_pts[i] = [np.nan, np.nan]
            continue

        numer = z_world + t_world[2] - t_world[0] * tan_ty - t_world[1] * tan_tx
        s = numer / denom
        P_world = s * ray_world - t_world

        world_pts[i] = P_world[:2]

    return world_pts


def _process_single_vector_file(args: Tuple) -> Optional[Dict[str, Any]]:
    """
    Process a single vector file for calibration.

    Module-level function for multiprocessing compatibility.
    Automatically detects ensemble data (with stress tensors) and calibrates
    both velocities and stresses using spatially-varying pinhole model.

    Args:
        args: Tuple of (file_idx, vector_file_path, output_file_path,
                       coords_by_run, camera_matrix, dist_coeffs,
                       rvec, tvec, dt, max_run, valid_run_nums, invert_ux,
                       image_height, z_world, tilt_x, tilt_y)
               where coords_by_run is Dict[int, Tuple[ndarray, ndarray]]
               mapping 1-based run numbers to (x_coords, y_coords).
               invert_ux: bool - if True, negate ux and UV_stress after calibration.
               image_height: int - image height in pixels for uncal→raw conversion.
               z_world, tilt_x, tilt_y: self-cal laser sheet correction params.

    Returns:
        Dict with results or None if failed
    """
    (
        file_idx,
        vector_file_path,
        output_file_path,
        coords_by_run,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        dt,
        max_run,
        valid_run_nums,
        invert_ux,
        image_height,
        z_world,
        tilt_x,
        tilt_y,
    ) = args

    def _uncal_to_raw_local(x_uncal, y_uncal):
        """Convert uncalibrated coords (1-based, y-up) to raw pixels (0-based, y-down)."""
        return x_uncal - 1, image_height - y_uncal

    try:
        # Try loading as structured .mat file first (for ensemble data)
        mat = loadmat(str(vector_file_path), struct_as_record=False, squeeze_me=True)

        # Check for ensemble_result (ensemble data) or piv_result (instantaneous)
        is_ensemble = False
        has_stresses = False

        if "ensemble_result" in mat:
            piv_result_raw = mat["ensemble_result"]
            result_key = "ensemble_result"
            is_ensemble = True
        elif "piv_result" in mat:
            piv_result_raw = mat["piv_result"]
            result_key = "piv_result"
        else:
            # Fall back to read_mat_contents for simple array format
            vector_data = read_mat_contents(str(vector_file_path))

            # Handle different vector data formats
            if vector_data.ndim == 4 and vector_data.shape[0] == 1:
                ux_px = vector_data[0, 0, :, :]
                uy_px = vector_data[0, 1, :, :]
                b_mask = vector_data[0, 2, :, :]
            elif vector_data.ndim == 3 and vector_data.shape[0] == 3:
                ux_px = vector_data[0, :, :]
                uy_px = vector_data[1, :, :]
                b_mask = vector_data[2, :, :]
            else:
                return None

            # Simple array format - find matching coordinates by shape
            # Try to find coordinates that match the velocity data shape
            coords_x_px, coords_y_px = None, None
            for run_num in sorted(coords_by_run.keys()):
                cx, cy = coords_by_run[run_num]
                if cx.shape == ux_px.shape:
                    coords_x_px, coords_y_px = cx, cy
                    break

            # Fall back to first available if no shape match
            if coords_x_px is None and coords_by_run:
                first_run = min(coords_by_run.keys())
                coords_x_px, coords_y_px = coords_by_run[first_run]

            if coords_x_px is None:
                return {"frame": file_idx, "success": False, "error": "No coordinates available"}

            # Convert uncalibrated coords to raw pixels for the pinhole model
            raw_x, raw_y = _uncal_to_raw_local(coords_x_px, coords_y_px)
            coords_flat = np.stack(
                [raw_x.flatten(), raw_y.flatten()], axis=-1
            ).astype(np.float32)

            if coords_flat.size == 0 or ux_px.size == 0:
                return None

            coords_world = _pixels_to_world_mm(
                coords_flat, camera_matrix, dist_coeffs, rvec, tvec,
                z_world=z_world, tilt_x=tilt_x, tilt_y=tilt_y,
            )
            # Displaced positions: compute in uncal space, then convert to raw
            disp_x_uncal = coords_x_px + ux_px
            disp_y_uncal = coords_y_px + uy_px
            disp_raw_x, disp_raw_y = _uncal_to_raw_local(disp_x_uncal, disp_y_uncal)
            disp_flat = np.stack(
                [disp_raw_x.flatten(), disp_raw_y.flatten()], axis=-1
            ).astype(np.float32)
            disp_world = _pixels_to_world_mm(
                disp_flat, camera_matrix, dist_coeffs, rvec, tvec,
                z_world=z_world, tilt_x=tilt_x, tilt_y=tilt_y,
            )
            delta_mm = disp_world - coords_world
            ux_ms = (delta_mm[:, 0] / 1000.0) / dt
            uy_ms = -(delta_mm[:, 1] / 1000.0) / dt
            ux_ms = ux_ms.reshape(ux_px.shape)
            uy_ms = uy_ms.reshape(uy_px.shape)

            if invert_ux:
                ux_ms = -ux_ms

            piv_dtype = np.dtype([("ux", "O"), ("uy", "O"), ("b_mask", "O")])
            piv_result = np.empty(max_run, dtype=piv_dtype)
            for run_num in range(1, max_run + 1):
                if run_num in valid_run_nums:
                    piv_result[run_num - 1] = (ux_ms, uy_ms, b_mask)
                else:
                    piv_result[run_num - 1] = (np.array([]), np.array([]), np.array([]))

            savemat(str(output_file_path), {"piv_result": piv_result})
            return {"frame": file_idx, "success": True}

        # Handle structured .mat format (piv_result or ensemble_result)
        # Ensure result is iterable
        if not hasattr(piv_result_raw, '__len__') or isinstance(piv_result_raw, np.void):
            piv_result_raw = [piv_result_raw]

        # Check if this is ensemble data with stress tensors
        for cell in piv_result_raw:
            if hasattr(cell, 'UU_stress') or hasattr(cell, 'VV_stress') or hasattr(cell, 'UV_stress'):
                has_stresses = True
                break

        # Build output struct array with appropriate dtype
        if has_stresses:
            piv_dtype = np.dtype([
                ("ux", "O"), ("uy", "O"), ("b_mask", "O"),
                ("UU_stress", "O"), ("VV_stress", "O"), ("UV_stress", "O")
            ])
        else:
            piv_dtype = np.dtype([("ux", "O"), ("uy", "O"), ("b_mask", "O")])

        piv_result = np.empty(len(piv_result_raw), dtype=piv_dtype)

        for idx, cell in enumerate(piv_result_raw):
            run_num = idx + 1  # 1-based run number
            ux_px = getattr(cell, "ux", None)
            uy_px = getattr(cell, "uy", None)
            b_mask = getattr(cell, "b_mask", None)
            if b_mask is None and ux_px is not None:
                b_mask = np.zeros_like(ux_px)
            elif b_mask is None:
                b_mask = np.array([])

            if ux_px is not None and uy_px is not None and ux_px.size > 0:
                # Get coordinates for this specific run
                coords_x_px, coords_y_px = None, None
                if run_num in coords_by_run:
                    coords_x_px, coords_y_px = coords_by_run[run_num]

                # Check shape match
                if coords_x_px is None or coords_x_px.shape != ux_px.shape:
                    # No matching coordinates for this run - store empty
                    if has_stresses:
                        piv_result[idx] = (np.array([]), np.array([]), np.array([]),
                                          np.array([]), np.array([]), np.array([]))
                    else:
                        piv_result[idx] = (np.array([]), np.array([]), np.array([]))
                    continue

                # Calibrate velocities using pinhole model
                # Convert uncalibrated coords to raw pixels for the pinhole model
                raw_x, raw_y = _uncal_to_raw_local(coords_x_px, coords_y_px)
                coords_flat = np.stack(
                    [raw_x.flatten(), raw_y.flatten()], axis=-1
                ).astype(np.float32)

                coords_world = _pixels_to_world_mm(
                    coords_flat, camera_matrix, dist_coeffs, rvec, tvec,
                    z_world=z_world, tilt_x=tilt_x, tilt_y=tilt_y,
                )
                # Displaced positions: compute in uncal space, then convert to raw
                disp_x_uncal = coords_x_px + ux_px
                disp_y_uncal = coords_y_px + uy_px
                disp_raw_x, disp_raw_y = _uncal_to_raw_local(disp_x_uncal, disp_y_uncal)
                disp_flat = np.stack(
                    [disp_raw_x.flatten(), disp_raw_y.flatten()], axis=-1
                ).astype(np.float32)
                disp_world = _pixels_to_world_mm(
                    disp_flat, camera_matrix, dist_coeffs, rvec, tvec,
                    z_world=z_world, tilt_x=tilt_x, tilt_y=tilt_y,
                )
                delta_mm = disp_world - coords_world
                ux_ms = (delta_mm[:, 0] / 1000.0) / dt
                uy_ms = -(delta_mm[:, 1] / 1000.0) / dt
                ux_ms = ux_ms.reshape(ux_px.shape)
                uy_ms = uy_ms.reshape(uy_px.shape)

                if invert_ux:
                    ux_ms = -ux_ms

                if has_stresses:
                    UU_stress = getattr(cell, "UU_stress", None)
                    VV_stress = getattr(cell, "VV_stress", None)
                    UV_stress = getattr(cell, "UV_stress", None)

                    # Compute stress scale factor for this run's grid
                    delta_px = 1.0
                    coords_disp_x = coords_flat + np.array([delta_px, 0.0], dtype=np.float32)
                    world_disp_x = _pixels_to_world_mm(
                        coords_disp_x, camera_matrix, dist_coeffs, rvec, tvec,
                        z_world=z_world, tilt_x=tilt_x, tilt_y=tilt_y,
                    )
                    coords_disp_y = coords_flat + np.array([0.0, delta_px], dtype=np.float32)
                    world_disp_y = _pixels_to_world_mm(
                        coords_disp_y, camera_matrix, dist_coeffs, rvec, tvec,
                        z_world=z_world, tilt_x=tilt_x, tilt_y=tilt_y,
                    )
                    delta_world_x = np.linalg.norm(world_disp_x - coords_world, axis=1)
                    delta_world_y = np.linalg.norm(world_disp_y - coords_world, axis=1)
                    delta_world_avg = (delta_world_x + delta_world_y) / 2.0
                    velocity_scale = (delta_world_avg / delta_px) / 1000.0 / dt
                    stress_scale = (velocity_scale ** 2).reshape(coords_x_px.shape)

                    # Calibrate stresses: pixels²/frame² -> m²/s²
                    UU_calib = UU_stress * stress_scale if UU_stress is not None else np.array([])
                    VV_calib = VV_stress * stress_scale if VV_stress is not None else np.array([])
                    UV_calib = UV_stress * stress_scale if UV_stress is not None else np.array([])

                    if invert_ux and UV_calib.size > 0:
                        UV_calib = -UV_calib

                    piv_result[idx] = (ux_ms, uy_ms, b_mask, UU_calib, VV_calib, UV_calib)
                else:
                    piv_result[idx] = (ux_ms, uy_ms, b_mask)
            else:
                if has_stresses:
                    piv_result[idx] = (np.array([]), np.array([]), np.array([]),
                                      np.array([]), np.array([]), np.array([]))
                else:
                    piv_result[idx] = (np.array([]), np.array([]), np.array([]))

        # Save calibrated result
        savemat(str(output_file_path), {result_key: piv_result})

        return {"frame": file_idx, "success": True}

    except Exception as e:
        return {"frame": file_idx, "success": False, "error": str(e)}


class VectorCalibrator:
    """
    Calibrates PIV vectors from pixels/frame to m/s using pinhole camera model.

    Supports both ChArUco and Planar (circle grid) calibration models.
    Uses rvec/tvec from the first calibration view to project pixels to
    world coordinates on the Z=0 calibration plane.
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        camera_num: Optional[int] = None,
        model_type: Optional[str] = None,
        dt: Optional[float] = None,
        vector_pattern: Optional[str] = None,
        type_name: str = "instantaneous",
        runs: Optional[List[int]] = None,
        num_workers: Optional[int] = None,
        config=None,
        z_world: float = 0.0,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
    ):
        """
        Initialize vector calibrator.

        Parameters can be provided explicitly or read from config. When config
        is provided, it takes precedence for settings stored in config.yaml.

        Args:
            base_dir: Base directory containing data (or from config.base_paths[0])
            camera_num: Camera number (1-based) (or from config.camera_numbers[0])
            model_type: Calibration model type - "charuco" or "dotboard" (or from config.active_calibration_method)
            dt: Time step between frames in seconds (or from config.dt)
            vector_pattern: Pattern for vector files (or from config.vector_format)
            type_name: Type name for data directory (e.g. "instantaneous", "ensemble")
            runs: List of 1-indexed run numbers to process, or None for all runs
            num_workers: Number of parallel workers, None = os.cpu_count()
            config: Optional Config object to read settings from
            z_world: Self-cal Z-offset of laser sheet from calibration plane (mm)
            tilt_x: Self-cal tilt angle about X-axis (radians)
            tilt_y: Self-cal tilt angle about Y-axis (radians)
        """
        self._config = config

        # Read from config if provided, otherwise use explicit parameters
        if config is not None:
            self.base_dir = Path(base_dir) if base_dir else config.base_paths[0]
            self.camera_num = camera_num if camera_num else config.camera_numbers[0]
            # Map active calibration method to model type
            active_method = config.active_calibration_method
            if model_type:
                self.model_type = model_type.lower()
            elif active_method == "charuco":
                self.model_type = "charuco"
            elif active_method == "dotboard":
                self.model_type = "dotboard"
            else:
                raise ValueError(f"Cannot determine model_type from config.active_calibration_method: {active_method}")
            self.dt = dt if dt is not None else config.dt
            self.vector_pattern = vector_pattern if vector_pattern else config.vector_format
            self.num_frame_pairs = config.num_frame_pairs
        else:
            # Explicit parameters required when no config
            if base_dir is None:
                raise ValueError("base_dir is required when config is not provided")
            if camera_num is None:
                raise ValueError("camera_num is required when config is not provided")
            if model_type is None:
                raise ValueError("model_type is required when config is not provided")
            if dt is None:
                raise ValueError("dt is required when config is not provided")
            self.base_dir = Path(base_dir)
            self.camera_num = camera_num
            self.model_type = model_type.lower()
            self.dt = dt
            self.vector_pattern = vector_pattern if vector_pattern else "%05d.mat"
            self.num_frame_pairs = None  # Must be passed to process_run()

        self.type_name = type_name
        self.runs = runs  # 1-indexed
        self.num_workers = num_workers if num_workers else get_max_workers(999999)

        # Validate model type
        if self.model_type not in ("charuco", "dotboard"):
            raise ValueError(
                f"model_type must be 'charuco' or 'dotboard', got '{self.model_type}'"
            )

        # Load calibration model
        self.calibration_model = self._load_calibration_model()
        self.camera_matrix = self.calibration_model["camera_matrix"]
        self.dist_coeffs = self.calibration_model["dist_coeffs"]
        self.rvec = self.calibration_model["rvec"]  # First view
        self.tvec = self.calibration_model["tvec"]  # First view
        self.dot_spacing_mm = self.calibration_model["dot_spacing_mm"]
        self.image_height = self.calibration_model["image_height"]

        # Self-calibration sheet correction parameters
        self.z_world = z_world
        self.tilt_x = tilt_x
        self.tilt_y = tilt_y

        logger.info(f"Initialized calibrator for Camera {camera_num}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Time step: {self.dt} seconds")
        logger.info(f"Dot spacing: {self.dot_spacing_mm} mm")
        logger.info(f"Vector pattern: {self.vector_pattern}")
        logger.info(f"Type name: {type_name}")
        logger.info(f"Runs to process: {runs if runs else 'all'}")
        logger.info(f"Worker count: {self.num_workers}")

    def _load_calibration_model(self) -> Dict[str, Any]:
        """Load the calibration model based on model_type."""
        calib_paths = get_data_paths(
            self.base_dir,
            num_frame_pairs=1,
            cam=self.camera_num,
            type_name="",
            calibration=True,
        )

        calib_dir = calib_paths["calib_dir"]

        # Build model path based on model type
        if self.model_type == "charuco":
            model_path = calib_dir / "charuco_planar" / "model" / "camera_model.mat"
        else:  # dotboard
            model_path = calib_dir / "dotboard_planar" / "model" / "dotboard_model.mat"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Calibration model not found: {model_path}\n"
                f"Expected {self.model_type} model at this location."
            )

        logger.info(f"Loading {self.model_type} calibration model: {model_path}")
        model_data = loadmat(str(model_path), squeeze_me=True, struct_as_record=False)

        # Validate required fields
        required_fields = ["camera_matrix", "dist_coeffs", "rvecs", "tvecs"]
        missing_fields = [f for f in required_fields if f not in model_data]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in calibration model: {missing_fields}"
            )

        # Extract camera matrix and distortion coefficients
        camera_matrix = np.array(model_data["camera_matrix"]).astype(np.float64)
        dist_coeffs = np.array(model_data["dist_coeffs"]).flatten().astype(np.float64)

        # Extract rvec/tvec from first calibration view
        rvecs = model_data["rvecs"]
        tvecs = model_data["tvecs"]

        # Handle single vs multiple views
        if rvecs.ndim == 1:
            rvec = rvecs.astype(np.float64)
            tvec = tvecs.astype(np.float64)
        else:
            rvec = rvecs[0].flatten().astype(np.float64)
            tvec = tvecs[0].flatten().astype(np.float64)

        # Get dot spacing in mm (both model types now store dot_spacing_mm)
        dot_spacing_mm = float(model_data.get("dot_spacing_mm", 28.89))

        # Get image height — required for uncalibrated→raw coordinate conversion.
        # Dotboard saves "image_height" directly; charuco saves "image_size": [W, H].
        image_height = int(model_data.get("image_height", 0))
        if image_height == 0 and "image_size" in model_data:
            img_size = np.asarray(model_data["image_size"]).flatten()
            if img_size.size >= 2:
                image_height = int(img_size[1])  # [W, H] → H
        if image_height == 0:
            raise ValueError(
                f"Calibration model {model_path} does not contain 'image_height' or "
                f"'image_size'. Re-generate the model with the latest calibration code."
            )

        # Use dt from model if available
        if "dt" in model_data:
            logger.info(f"Using dt from calibration model: {model_data['dt']} seconds")
            self.dt = float(model_data["dt"])

        return {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "rvec": rvec,
            "tvec": tvec,
            "dot_spacing_mm": dot_spacing_mm,
            "image_height": image_height,
        }

    def _uncal_to_raw(self, x_uncal, y_uncal):
        """Convert uncalibrated coords (1-based, y-up) to raw pixels (0-based, y-down).

        Reverses the convention applied in save_results.py:
            x_uncal = x_raw + 1
            y_uncal = H - y_raw
        So the inverse is:
            x_raw = x_uncal - 1
            y_raw = H - y_uncal
        """
        return x_uncal - 1, self.image_height - y_uncal

    def calibrate_coordinates(
        self, coords_x: np.ndarray, coords_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel coordinates to physical coordinates in mm.

        Uses the pinhole camera model with distortion correction to project
        pixel coordinates to the Z=0 calibration plane.

        Args:
            coords_x, coords_y: Coordinate arrays in uncalibrated pixels (1-based, y-up)

        Returns:
            (x_mm, y_mm): Coordinate arrays in mm
        """
        # Convert uncalibrated coords (1-based, y-up) to raw pixels (0-based, y-down)
        # because the pinhole model was fitted using raw pixel coordinates.
        raw_x, raw_y = self._uncal_to_raw(coords_x, coords_y)
        pts = np.stack([raw_x.flatten(), raw_y.flatten()], axis=-1).astype(
            np.float32
        )

        if pts.size == 0:
            return coords_x, coords_y

        # Project to world coordinates (mm) on the calibration/sheet plane
        world_pts = _pixels_to_world_mm(
            pts, self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec,
            z_world=self.z_world, tilt_x=self.tilt_x, tilt_y=self.tilt_y,
        )

        x_mm = world_pts[:, 0].reshape(coords_x.shape)
        # Negate y: the pinhole model's world y-axis points opposite to the
        # physical y-up convention because we feed raw pixels (y-down).
        y_mm = -world_pts[:, 1].reshape(coords_y.shape)

        logger.info("Converted coordinates from pixels to mm (pinhole model)")

        return x_mm, y_mm

    def calibrate_vectors(
        self,
        ux_px: np.ndarray,
        uy_px: np.ndarray,
        coords_x_px: np.ndarray,
        coords_y_px: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel-based velocity vectors to m/s using pinhole camera model.

        Args:
            ux_px, uy_px: Velocity components in uncalibrated pixels/frame
            coords_x_px, coords_y_px: Grid coordinates in uncalibrated pixels (1-based, y-up)

        Returns:
            (ux_ms, uy_ms): Velocity components in m/s
        """
        if coords_x_px.size == 0 or ux_px.size == 0 or uy_px.size == 0:
            logger.warning("Empty coordinate or vector data, returning zeros")
            return np.zeros_like(ux_px), np.zeros_like(uy_px)

        if ux_px.shape != uy_px.shape or ux_px.shape != coords_x_px.shape:
            logger.error(
                f"Shape mismatch: ux_px={ux_px.shape}, uy_px={uy_px.shape}, "
                f"coords_x_px={coords_x_px.shape}"
            )
            return np.zeros_like(ux_px), np.zeros_like(uy_px)

        # Convert base positions: uncalibrated (1-based, y-up) → raw pixels (0-based, y-down)
        raw_x, raw_y = self._uncal_to_raw(coords_x_px, coords_y_px)
        coords_flat = np.stack(
            [raw_x.flatten(), raw_y.flatten()], axis=-1
        ).astype(np.float32)

        # Project original positions to world (mm)
        coords_world = _pixels_to_world_mm(
            coords_flat, self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec,
            z_world=self.z_world, tilt_x=self.tilt_x, tilt_y=self.tilt_y,
        )

        # Displaced positions: compute in uncalibrated space, then convert to raw.
        # ux/uy from .mat are in uncalibrated convention (uy positive = upward).
        # Adding them to uncal coords gives displaced position in uncal space.
        disp_x_uncal = coords_x_px + ux_px
        disp_y_uncal = coords_y_px + uy_px
        disp_raw_x, disp_raw_y = self._uncal_to_raw(disp_x_uncal, disp_y_uncal)
        disp_flat = np.stack(
            [disp_raw_x.flatten(), disp_raw_y.flatten()], axis=-1
        ).astype(np.float32)

        # Project displaced positions to world (mm)
        disp_world = _pixels_to_world_mm(
            disp_flat, self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec,
            z_world=self.z_world, tilt_x=self.tilt_x, tilt_y=self.tilt_y,
        )

        # Compute displacement in mm, convert to m/s
        # Negate uy: pinhole world y-axis points downward (raw pixel convention),
        # but physical convention is y-up.  Must match calibrate_coordinates()
        # which also negates y: y_mm = -world_pts[:, 1].
        delta_mm = disp_world - coords_world
        ux_ms = (delta_mm[:, 0] / 1000.0) / self.dt
        uy_ms = -(delta_mm[:, 1] / 1000.0) / self.dt

        ux_ms = ux_ms.reshape(ux_px.shape)
        uy_ms = uy_ms.reshape(uy_px.shape)

        logger.info("Converted vectors from pixels/frame to m/s (pinhole model)")

        return ux_ms, uy_ms

    def _compute_local_scale_factor(
        self,
        coords_x_px: np.ndarray,
        coords_y_px: np.ndarray,
    ) -> np.ndarray:
        """
        Compute local velocity scaling factor at each grid point.

        For pinhole calibration, the scaling factor varies spatially due to
        lens distortion and perspective effects. This method computes the
        local scaling factor by projecting a small pixel displacement to
        world coordinates and measuring the resulting world displacement.

        Args:
            coords_x_px, coords_y_px: Grid coordinates in uncalibrated pixels (1-based, y-up)

        Returns:
            scale_factor: 2D array of local scaling factors (m/s per pixel/frame)
        """
        delta_px = 1.0  # Small pixel displacement for computing local scale

        # Convert to raw pixels first, then perturb in raw pixel space
        # (the scale factor should reflect mm-per-raw-pixel)
        raw_x, raw_y = self._uncal_to_raw(coords_x_px, coords_y_px)
        coords_flat = np.stack(
            [raw_x.flatten(), raw_y.flatten()], axis=-1
        ).astype(np.float32)

        if coords_flat.size == 0:
            return np.ones_like(coords_x_px)

        # Project original positions to world (mm)
        coords_world = _pixels_to_world_mm(
            coords_flat, self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec,
            z_world=self.z_world, tilt_x=self.tilt_x, tilt_y=self.tilt_y,
        )

        # Project positions displaced by delta_px in x direction
        coords_disp_x = coords_flat + np.array([delta_px, 0.0], dtype=np.float32)
        world_disp_x = _pixels_to_world_mm(
            coords_disp_x, self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec,
            z_world=self.z_world, tilt_x=self.tilt_x, tilt_y=self.tilt_y,
        )

        # Project positions displaced by delta_px in y direction
        coords_disp_y = coords_flat + np.array([0.0, delta_px], dtype=np.float32)
        world_disp_y = _pixels_to_world_mm(
            coords_disp_y, self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec,
            z_world=self.z_world, tilt_x=self.tilt_x, tilt_y=self.tilt_y,
        )

        # Compute displacement magnitudes in world coordinates (mm)
        # Average the x and y displacement magnitudes for isotropic scaling
        delta_world_x = np.linalg.norm(world_disp_x - coords_world, axis=1)
        delta_world_y = np.linalg.norm(world_disp_y - coords_world, axis=1)
        delta_world_avg = (delta_world_x + delta_world_y) / 2.0

        # Convert to velocity scaling factor: mm/pixel * (m/mm) / dt = m/s per pixel/frame
        # scale_factor = (delta_world_mm / delta_px) / 1000 / dt
        scale_factor = (delta_world_avg / delta_px) / 1000.0 / self.dt

        return scale_factor.reshape(coords_x_px.shape)

    def calibrate_stresses(
        self,
        stress_px: np.ndarray,
        coords_x_px: np.ndarray,
        coords_y_px: np.ndarray,
    ) -> np.ndarray:
        """
        Calibrate stress tensor using spatially-varying scaling factor.

        For ensemble PIV, Reynolds stresses (UU, VV, UV) are computed before
        calibration and have units of velocity². The calibration factor must
        be squared compared to velocity calibration.

        Since pinhole calibration is spatially-varying, we compute the local
        velocity scaling factor at each grid point and square it for stress.

        Args:
            stress_px: Stress tensor in pixels²/frame²
            coords_x_px, coords_y_px: Grid coordinates in pixels

        Returns:
            Stress tensor in m²/s²
        """
        # Get local velocity scale factor at each grid point
        scale_factor = self._compute_local_scale_factor(coords_x_px, coords_y_px)

        # Stress scaling is velocity scaling squared
        stress_scale = scale_factor ** 2

        return stress_px * stress_scale

    def process_run(
        self,
        num_frame_pairs: Optional[int] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        alignment: Optional[Dict] = None,
    ):
        """
        Process and calibrate vectors for specified runs.

        Args:
            num_frame_pairs: Number of frame pairs. If None, uses self.num_frame_pairs from config.
            progress_cb: Optional callback for progress updates
            alignment: Optional dict from GlobalCoordinateAligner.precompute_camera_shifts().
                If provided, shifts are applied to coordinates and invert_ux is applied
                to vectors during calibration (fused single-pass).
                Keys: camera_shifts, invert_ux, datum_physical_x
        """
        # Use config value if not explicitly provided
        if num_frame_pairs is None:
            num_frame_pairs = self.num_frame_pairs
        if num_frame_pairs is None:
            raise ValueError("num_frame_pairs must be provided either to __init__ via config or to process_run()")

        logger.info(f"Processing run with {num_frame_pairs} frame pairs")

        # Get data paths for uncalibrated data
        paths = get_data_paths(
            self.base_dir,
            num_frame_pairs=num_frame_pairs,
            cam=self.camera_num,
            type_name=self.type_name,
            use_uncalibrated=True,
        )

        uncalib_data_dir = paths["data_dir"]
        logger.info(f"Uncalibrated data directory: {uncalib_data_dir}")

        # Get output paths for calibrated data
        calib_paths = get_data_paths(
            self.base_dir,
            num_frame_pairs=num_frame_pairs,
            cam=self.camera_num,
            type_name=self.type_name,
        )

        calib_data_dir = calib_paths["data_dir"]
        calib_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Calibrated data directory: {calib_data_dir}")

        if not uncalib_data_dir.exists():
            raise FileNotFoundError(
                f"Uncalibrated data directory not found: {uncalib_data_dir}"
            )

        # Load coordinates for requested runs (1-indexed)
        logger.info("Loading coordinates...")
        x_coords_list, y_coords_list = load_coords_from_directory(
            uncalib_data_dir, runs=self.runs
        )

        if not x_coords_list:
            logger.error("No coordinate data found!")
            raise ValueError("No coordinate data found")

        logger.info(f"Loaded coordinates for {len(x_coords_list)} runs")

        # Find runs with valid data
        valid_runs = []
        for i, (x_coords, y_coords) in enumerate(zip(x_coords_list, y_coords_list)):
            # Map back to original run number if filtering
            if self.runs:
                run_num = self.runs[i]
            else:
                run_num = i + 1

            if x_coords is None:
                x_coords = np.array([])
            if y_coords is None:
                y_coords = np.array([])

            valid_coords = np.sum(~np.isnan(x_coords)) + np.sum(~np.isnan(y_coords))
            logger.info(f"Run {run_num}: {valid_coords} valid coordinates")
            if valid_coords > 0:
                valid_runs.append((i, run_num, valid_coords, x_coords, y_coords))

        if not valid_runs:
            raise ValueError("No runs with valid coordinate data found")

        logger.info(
            f"Found {len(valid_runs)} runs with valid data: {[r[1] for r in valid_runs]}"
        )

        # Create coordinate structure
        max_run = max(r[1] for r in valid_runs)
        coord_dtype = np.dtype([("x", "O"), ("y", "O")])
        coordinates = np.empty(max_run, dtype=coord_dtype)

        # Initialize all runs with empty arrays
        for run_num in range(1, max_run + 1):
            coordinates[run_num - 1] = (np.array([]), np.array([]))

        # Resolve alignment params for this camera
        invert_ux = False
        coord_shift = None
        datum_physical_x = 0.0
        if alignment:
            invert_ux = alignment.get("invert_ux", False)
            datum_physical_x = alignment.get("datum_physical_x", 0.0)
            coord_shift = alignment.get("camera_shifts", {}).get(self.camera_num)

        # Process each valid run's coordinates
        for list_idx, run_num, valid_coord_count, x_coords_px, y_coords_px in valid_runs:
            logger.info(
                f"Processing run {run_num} with {valid_coord_count} valid coordinates"
            )
            x_coords_mm, y_coords_mm = self.calibrate_coordinates(
                x_coords_px, y_coords_px
            )

            # Apply alignment shift if provided
            if coord_shift is not None:
                x_coords_mm = x_coords_mm + coord_shift[0]
                y_coords_mm = y_coords_mm + coord_shift[1]

            # Apply x-reflection for invert_ux
            if invert_ux:
                x_coords_mm = 2.0 * datum_physical_x - x_coords_mm

            coordinates[run_num - 1] = (x_coords_mm, y_coords_mm)

        # Save calibrated coordinates
        coords_output = {"coordinates": coordinates}
        coords_path = calib_data_dir / "coordinates.mat"
        savemat(str(coords_path), coords_output)
        logger.info(f"Saved calibrated coordinates: {coords_path}")

        # Write alignment marker if alignment was applied, else clear it
        from pivtools_gui.calibration.global_coordinate_alignment import GlobalCoordinateAligner
        if alignment and coord_shift is not None:
            import json
            from datetime import datetime
            marker_path = calib_data_dir / GlobalCoordinateAligner.ALIGNMENT_MARKER
            marker_data = {
                "aligned_at": datetime.now().isoformat(),
                "shift_x": coord_shift[0],
                "shift_y": coord_shift[1],
                "method": self.model_type,
                "fused": True,
            }
            marker_path.write_text(json.dumps(marker_data, indent=2))
        else:
            GlobalCoordinateAligner.clear_alignment_marker(calib_data_dir)

        # Process vector files - build per-run coordinate mapping
        if valid_runs:
            # Build dict mapping run_num (1-based) -> (x_coords, y_coords)
            # Each run may have different grid sizes due to different window sizes
            coords_by_run = {
                run_num: (x_coords, y_coords)
                for _, run_num, _, x_coords, y_coords in valid_runs
            }
            valid_run_nums = set(r[1] for r in valid_runs)

            logger.info(f"Coordinates available for runs: {sorted(coords_by_run.keys())}")

            # Check if this is ensemble data (single file) vs instantaneous (many files)
            ensemble_file = uncalib_data_dir / "ensemble_result.mat"
            if self.type_name == "ensemble" or ensemble_file.exists():
                # Ensemble data: single file with potentially different grid per run
                self._process_ensemble_file(
                    uncalib_data_dir,
                    calib_data_dir,
                    coords_by_run,
                    max_run,
                    valid_run_nums,
                    progress_cb,
                    invert_ux=invert_ux,
                )
            else:
                # Instantaneous data: many files
                self._process_vector_files_parallel(
                    uncalib_data_dir,
                    calib_data_dir,
                    num_frame_pairs,
                    coords_by_run,
                    max_run,
                    valid_run_nums,
                    progress_cb,
                    invert_ux=invert_ux,
                )
        else:
            logger.error("No valid runs found for vector processing")

    def _process_ensemble_file(
        self,
        uncalib_dir: Path,
        calib_dir: Path,
        coords_by_run: Dict[int, Tuple[np.ndarray, np.ndarray]],
        max_run: int,
        valid_run_nums: set,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]],
        invert_ux: bool = False,
    ):
        """Process single ensemble_result.mat file."""
        logger.info("Processing ensemble result file...")

        ensemble_file = uncalib_dir / "ensemble_result.mat"
        output_file = calib_dir / "ensemble_result.mat"

        if not ensemble_file.exists():
            logger.error(f"Ensemble result file not found: {ensemble_file}")
            raise FileNotFoundError(f"Ensemble result file not found: {ensemble_file}")

        # Process the single ensemble file with per-run coordinates
        result = _process_single_vector_file((
            1,  # file_idx
            str(ensemble_file),
            str(output_file),
            coords_by_run,
            self.camera_matrix,
            self.dist_coeffs,
            self.rvec,
            self.tvec,
            self.dt,
            max_run,
            valid_run_nums,
            invert_ux,
            self.image_height,
            self.z_world,
            self.tilt_x,
            self.tilt_y,
        ))

        if result and result.get("success"):
            logger.info(f"Successfully calibrated ensemble result: {output_file}")
            if progress_cb:
                progress_cb({
                    "processed_frames": 1,
                    "total_frames": 1,
                    "progress": 100,
                    "successful_frames": 1,
                    "failed_frames": 0,
                })
        else:
            error_msg = result.get("error", "Unknown error") if result else "Processing returned None"
            logger.error(f"Failed to calibrate ensemble result: {error_msg}")
            raise RuntimeError(f"Failed to calibrate ensemble result: {error_msg}")

    def _process_vector_files_parallel(
        self,
        uncalib_dir: Path,
        calib_dir: Path,
        num_images: int,
        coords_by_run: Dict[int, Tuple[np.ndarray, np.ndarray]],
        max_run: int,
        valid_run_nums: set,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]],
        invert_ux: bool = False,
    ):
        """Process all vector files using parallel workers."""
        logger.info(f"Processing vector files with {self.num_workers} workers...")

        # Build list of files to process
        tasks = []
        for i in range(1, num_images + 1):
            vector_file = uncalib_dir / (self.vector_pattern % i)
            if not vector_file.exists():
                continue

            output_file = calib_dir / (self.vector_pattern % i)
            tasks.append(
                (
                    i,
                    str(vector_file),
                    str(output_file),
                    coords_by_run,
                    self.camera_matrix,
                    self.dist_coeffs,
                    self.rvec,
                    self.tvec,
                    self.dt,
                    max_run,
                    valid_run_nums,
                    invert_ux,
                    self.image_height,
                    self.z_world,
                    self.tilt_x,
                    self.tilt_y,
                )
            )

        if not tasks:
            error_msg = f"No vector files found to process in {uncalib_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Found {len(tasks)} vector files to process")

        successful = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=worker_initializer) as executor:
            futures = {
                executor.submit(_process_single_vector_file, task): task[0]
                for task in tasks
            }

            for future in as_completed(futures):
                result = future.result()
                if result and result.get("success"):
                    successful += 1
                else:
                    failed += 1
                    if result and "error" in result:
                        logger.debug(f"Frame {result['frame']} failed: {result['error']}")

                # Progress callback (approximate)
                if progress_cb:
                    total_done = successful + failed
                    progress_cb(
                        {
                            "processed_frames": total_done,
                            "total_frames": len(tasks),
                            "progress": (total_done / len(tasks)) * 100,
                            "successful_frames": successful,
                            "failed_frames": failed,
                        }
                    )

        logger.info(
            f"Successfully processed {successful} vector files, {failed} failed"
        )


def main():
    """Main entry point for vector calibration.

    When USE_CONFIG_DIRECTLY=True, loads settings from existing config.yaml instead
    of applying the hardcoded CLI settings.
    """
    logger.info("=" * 60)
    logger.info("Vector Calibration - Starting")
    logger.info("=" * 60)

    if USE_CONFIG_DIRECTLY:
        # Load settings directly from existing config.yaml
        logger.info("Loading settings directly from config.yaml (USE_CONFIG_DIRECTLY=True)")
        config = get_config()

        # Log settings from config
        logger.info(f"Base directory: {config.base_paths[0]}")
        logger.info(f"Num frame pairs: {config.num_frame_pairs}")
        logger.info(f"Time step: {config.dt} seconds")
        logger.info(f"Cameras: {config.camera_numbers}")
        logger.info(f"Active calibration: {config.active_calibration_method}")
        logger.info(f"Vector pattern: {config.vector_format}")
        logger.info(f"Type name: {TYPE_NAME}")
        logger.info(f"Runs to process: {RUNS_TO_PROCESS if RUNS_TO_PROCESS else 'all'}")
        logger.info(f"Worker count: {get_max_workers(999999)}")

        camera_nums = config.camera_numbers
    else:
        # Log hardcoded settings and apply to config
        logger.info(f"Base directory: {BASE_DIR}")
        logger.info(f"Num frame pairs: {NUM_FRAME_PAIRS}")
        logger.info(f"Time step: {DT_SECONDS} seconds")
        logger.info(f"Cameras: {CAMERA_NUMS}")
        logger.info(f"Model type: {MODEL_TYPE}")
        logger.info(f"Vector pattern: {VECTOR_PATTERN}")
        logger.info(f"Type name: {TYPE_NAME}")
        logger.info(f"Runs to process: {RUNS_TO_PROCESS if RUNS_TO_PROCESS else 'all'}")
        logger.info(f"Worker count: {get_max_workers(999999)}")

        # Apply CLI settings to config.yaml so centralized systems work correctly
        config = apply_cli_settings_to_config()
        camera_nums = CAMERA_NUMS

    failed_cameras = []

    for camera_num in camera_nums:
        logger.info(f"Processing Camera {camera_num}...")
        try:
            # Create calibrator using config - settings are now in config.yaml
            calibrator = VectorCalibrator(
                camera_num=camera_num,
                type_name=TYPE_NAME,
                runs=RUNS_TO_PROCESS,
                config=config,
            )

            calibrator.process_run()  # num_frame_pairs read from config
            logger.info(f"Camera {camera_num} completed successfully")

        except Exception as e:
            logger.error(f"Camera {camera_num} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_cameras.append(camera_num)

    logger.info("=" * 60)
    if failed_cameras:
        logger.error(f"Calibration failed for cameras: {failed_cameras}")
        sys.exit(1)
    else:
        logger.info("Vector calibration completed successfully for all cameras")


if __name__ == "__main__":
    main()
