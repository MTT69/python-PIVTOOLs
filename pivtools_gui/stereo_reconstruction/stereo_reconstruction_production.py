#!/usr/bin/env python3
"""
stereo_reconstruction_production.py

Production script for 3D velocity reconstruction from stereo camera pairs.
Takes uncalibrated 2D velocity fields from two cameras and reconstructs 3D velocities (ux, uy, uz).
Supports both ChArUco and Dotboard (circle grid) stereo calibration models.
Uses ProcessPoolExecutor for parallel frame processing.
"""

import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.io import loadmat, savemat

sys.path.append(str(Path(__file__).parent.parent))
from pivtools_core.config import get_config, reload_config
from pivtools_core.paths import get_data_paths
from pivtools_core.vector_loading import load_coords_from_directory, read_mat_contents

# ===================== CONFIGURATION VARIABLES =====================
# Set these variables for your stereo reconstruction setup.
# These will be written to config.yaml before processing.
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/stereo/processed"
CAMERA_PAIR = [1, 2]  # Single pair [cam1_num, cam2_num]
NUM_FRAME_PAIRS = 100  # Number of frame pairs to process
DT_SECONDS = 0.0057553   # Time step between frames in seconds
MODEL_TYPE = "dotboard"  # "charuco" or "dotboard" - determines stereo model expectation
VECTOR_PATTERN = "%05d.mat"  # Pattern for vector files
TYPE_NAME = "instantaneous"  # Type name for data directory
MIN_TRIANGULATION_ANGLE = 5.0  # Minimum angle in degrees for triangulation quality
RUNS_TO_PROCESS = None  # List of 1-indexed runs to process, or None for all
NUM_WORKERS = None  # Number of parallel workers, None = os.cpu_count()

# USE_CONFIG_DIRECTLY: If True, skip updating config.yaml with above parameters
# and load reconstruction settings directly from the existing config.yaml
USE_CONFIG_DIRECTLY = True
# ===================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    config.data["paths"]["camera_numbers"] = CAMERA_PAIR
    config.data["paths"]["camera_count"] = max(CAMERA_PAIR)

    # Images
    config.data["images"]["num_frame_pairs"] = NUM_FRAME_PAIRS
    config.data["images"]["vector_format"] = [VECTOR_PATTERN]

    # Calibration - set active to stereo and update stereo settings
    config.data["calibration"]["active"] = "stereo"
    config.data["calibration"]["stereo"]["camera_pair"] = CAMERA_PAIR
    config.data["calibration"]["stereo"]["stereo_model_type"] = MODEL_TYPE
    config.data["calibration"]["stereo"]["dt"] = DT_SECONDS

    # Save to disk so centralized systems pick up changes
    config.save()
    logger.info("Updated config.yaml with CLI settings")

    # Reload to ensure fresh state
    return reload_config()


# ===================== MODULE-LEVEL HELPER FUNCTIONS =====================
# These must be at module level for ProcessPoolExecutor pickling compatibility


def _extract_velocity_components(
    vector_data: np.ndarray, run_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ux, uy velocity components from vector data for specified run.

    Args:
        vector_data: Loaded vector data array
        run_idx: 1-indexed run number

    Returns:
        (ux_px, uy_px): Velocity components in pixels/frame
    """
    if vector_data.ndim == 4 and vector_data.shape[0] >= run_idx:
        # Multiple runs: (runs, 3, height, width)
        ux_px = vector_data[run_idx - 1, 0, :, :]
        uy_px = vector_data[run_idx - 1, 1, :, :]
    elif vector_data.ndim == 4 and vector_data.shape[0] == 1 and vector_data.shape[1] == 3:
        # Single run with extra dimension: (1, 3, height, width)
        ux_px = vector_data[0, 0, :, :]
        uy_px = vector_data[0, 1, :, :]
    elif vector_data.ndim == 3 and vector_data.shape[0] == 3:
        # Single run: (3, height, width)
        ux_px = vector_data[0, :, :]
        uy_px = vector_data[1, :, :]
    else:
        raise ValueError(f"Unexpected vector_data shape: {vector_data.shape}")

    return ux_px, uy_px


def _find_corresponding_points(
    coords1_px: Tuple[np.ndarray, np.ndarray],
    coords2_px: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find corresponding point indices between two cameras.

    Uses grid-based correspondence assuming same interrogation window layout.

    Args:
        coords1_px: (x, y) coordinate arrays for camera 1
        coords2_px: (x, y) coordinate arrays for camera 2

    Returns:
        (indices1, indices2): Flat indices into respective coordinate arrays
    """
    shape1 = coords1_px[0].shape
    shape2 = coords2_px[0].shape

    if shape1 != shape2:
        min_h = min(shape1[0], shape2[0])
        min_w = min(shape1[1], shape2[1])
        indices1, indices2 = [], []
        for i in range(min_h):
            for j in range(min_w):
                indices1.append(np.ravel_multi_index((i, j), shape1))
                indices2.append(np.ravel_multi_index((i, j), shape2))
        return np.array(indices1), np.array(indices2)
    else:
        total_points = np.prod(shape1)
        return np.arange(total_points), np.arange(total_points)


def _triangulate_3d_points(
    pts1_px: np.ndarray,
    pts2_px: np.ndarray,
    stereo_params: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Triangulate 3D points from stereo image points.

    Args:
        pts1_px: Pixel coordinates from camera 1, shape (N, 2)
        pts2_px: Pixel coordinates from camera 2, shape (N, 2)
        stereo_params: Dict with camera matrices and projection matrices

    Returns:
        (points_3d, pts1_rect, pts2_rect): 3D points and rectified 2D points
    """
    mtx1 = stereo_params["camera_matrix_1"]
    dist1 = stereo_params["dist_coeffs_1"]
    mtx2 = stereo_params["camera_matrix_2"]
    dist2 = stereo_params["dist_coeffs_2"]
    R1 = stereo_params["rectification_R1"]
    R2 = stereo_params["rectification_R2"]
    P1 = stereo_params["projection_P1"]
    P2 = stereo_params["projection_P2"]

    # Undistort and rectify points
    pts1_rect = cv2.undistortPoints(
        pts1_px.reshape(-1, 1, 2).astype(np.float32), mtx1, dist1, R=R1, P=P1
    ).reshape(-1, 2)
    pts2_rect = cv2.undistortPoints(
        pts2_px.reshape(-1, 1, 2).astype(np.float32), mtx2, dist2, R=R2, P=P2
    ).reshape(-1, 2)

    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, pts1_rect.T, pts2_rect.T)
    points_3d = (points_4d[:3] / points_4d[3]).T

    return points_3d, pts1_rect, pts2_rect


def _compute_triangulation_angles(
    pts_3d: np.ndarray, stereo_params: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Compute triangulation angles for quality filtering.

    Args:
        pts_3d: 3D point positions, shape (N, 3)
        stereo_params: Dict with rotation matrix and translation vector

    Returns:
        Triangulation angles in degrees, shape (N,)
    """
    R = stereo_params["rotation_matrix"]
    T = stereo_params["translation_vector"].reshape(3)

    cam1_center = np.array([0.0, 0.0, 0.0])
    cam2_center = -R.T @ T

    vec1 = pts_3d - cam1_center
    vec2 = pts_3d - cam2_center

    # Normalize vectors
    vec1_norm = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
    vec2_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)

    # Compute angles
    dot_products = np.sum(vec1_norm * vec2_norm, axis=1)
    angles_rad = np.arccos(np.clip(dot_products, -1, 1))

    return np.degrees(angles_rad)


def _reconstruct_3d_velocities(
    ux1: np.ndarray,
    uy1: np.ndarray,
    ux2: np.ndarray,
    uy2: np.ndarray,
    coords1_px: Tuple[np.ndarray, np.ndarray],
    coords2_px: Tuple[np.ndarray, np.ndarray],
    stereo_params: Dict[str, np.ndarray],
    min_angle: float,
    combined_input_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Reconstruct 3D velocities from stereo 2D velocity fields.

    Args:
        ux1, uy1: Velocity components from camera 1 in pixels/frame
        ux2, uy2: Velocity components from camera 2 in pixels/frame
        coords1_px: (x, y) pixel coordinates for camera 1
        coords2_px: (x, y) pixel coordinates for camera 2
        stereo_params: Stereo calibration parameters
        min_angle: Minimum triangulation angle in degrees
        combined_input_mask: Combined boolean mask from cam1 and cam2 (True = invalid)

    Returns:
        Dict with velocities_3d, positions_3d, indices1, num_valid, num_total
    """
    indices1, indices2 = _find_corresponding_points(coords1_px, coords2_px)

    if len(indices1) == 0:
        return {
            "velocities_3d": np.array([]),
            "positions_3d": np.array([]),
            "indices1": np.array([]),
            "triangulation_angles": np.array([]),
            "num_valid": 0,
            "num_total": 0,
        }

    shape1, shape2 = coords1_px[0].shape, coords2_px[0].shape
    row1, col1 = np.unravel_index(indices1, shape1)
    row2, col2 = np.unravel_index(indices2, shape2)

    # Get pixel coordinates and velocities at corresponding points
    pts1_px = np.column_stack([coords1_px[0][row1, col1], coords1_px[1][row1, col1]])
    pts2_px = np.column_stack([coords2_px[0][row2, col2], coords2_px[1][row2, col2]])
    vel1 = np.column_stack([ux1[row1, col1], uy1[row1, col1]])
    vel2 = np.column_stack([ux2[row2, col2], uy2[row2, col2]])

    # Triangulate original positions
    pts_3d, _, _ = _triangulate_3d_points(pts1_px, pts2_px, stereo_params)

    # Filter by triangulation angle
    angles = _compute_triangulation_angles(pts_3d, stereo_params)
    angle_mask = angles > min_angle

    # Combine with input mask (conservative: both cams must be valid)
    if combined_input_mask is not None:
        combined_flat = combined_input_mask.flatten()[indices1]
        valid_mask = angle_mask & (~combined_flat)  # True if angle OK AND not masked
    else:
        valid_mask = angle_mask

    # Triangulate displaced positions
    pts1_displaced = pts1_px + vel1
    pts2_displaced = pts2_px + vel2
    pts_3d_displaced, _, _ = _triangulate_3d_points(
        pts1_displaced, pts2_displaced, stereo_params
    )

    # Compute 3D velocity (displacement in mm per frame)
    vel_3d_mm = pts_3d_displaced - pts_3d

    return {
        "velocities_3d": vel_3d_mm[valid_mask],
        "positions_3d": pts_3d[valid_mask],
        "indices1": indices1[valid_mask],
        "triangulation_angles": angles[valid_mask],
        "num_valid": np.sum(valid_mask),
        "num_total": len(valid_mask),
    }


def _process_stereo_frame(args: Tuple) -> Optional[Dict[str, Any]]:
    """
    Process a single stereo frame for 3D velocity reconstruction.

    Processes ALL valid runs in a single pass (like dotboard's pattern).
    Module-level function for ProcessPoolExecutor compatibility.

    Args:
        args: Tuple containing all parameters needed for processing
              - frame_idx: Frame number
              - vector_file_path_cam1, vector_file_path_cam2: Paths to vector files
              - output_file_path: Path to save output
              - coords_by_run: Dict mapping run_num -> (x1, y1, x2, y2) coordinates
              - stereo_params: Stereo calibration parameters
              - dt: Time step
              - min_angle: Minimum triangulation angle
              - max_run: Maximum run number
              - valid_run_nums: Set of valid run numbers

    Returns:
        Dict with results including per-run num_valid counts, or None if failed
    """
    (
        frame_idx,
        vector_file_path_cam1,
        vector_file_path_cam2,
        output_file_path,
        coords_by_run,
        stereo_params,
        dt,
        min_angle,
        max_run,
        valid_run_nums,
    ) = args

    try:
        # Load uncalibrated vectors for both cameras using structured format
        # (matching dotboard's pattern from vector_calibration_production.py:200-210)
        # This allows each run to have different grid dimensions
        mat1 = loadmat(vector_file_path_cam1, struct_as_record=False, squeeze_me=True)
        mat2 = loadmat(vector_file_path_cam2, struct_as_record=False, squeeze_me=True)

        # Get piv_result struct arrays
        piv_result_raw1 = mat1.get("piv_result")
        piv_result_raw2 = mat2.get("piv_result")

        if piv_result_raw1 is None or piv_result_raw2 is None:
            return {"frame": frame_idx, "success": False, "error": "piv_result not found"}

        # Ensure iterable (handle single-run case)
        if not hasattr(piv_result_raw1, '__len__') or isinstance(piv_result_raw1, np.void):
            piv_result_raw1 = [piv_result_raw1]
        if not hasattr(piv_result_raw2, '__len__') or isinstance(piv_result_raw2, np.void):
            piv_result_raw2 = [piv_result_raw2]

        # Create output piv_result structure array (with uz for 3D)
        piv_dtype = np.dtype([("ux", "O"), ("uy", "O"), ("uz", "O"), ("b_mask", "O")])
        piv_result = np.empty(max_run, dtype=piv_dtype)

        # Initialize all runs with empty arrays
        for r in range(1, max_run + 1):
            piv_result[r - 1] = (np.array([]), np.array([]), np.array([]), np.array([]))

        # Track per-run results for coordinate saving
        run_results = {}
        total_valid = 0

        # Process ALL valid runs for this frame
        for run_num in sorted(valid_run_nums):
            if run_num not in coords_by_run:
                continue

            x1, y1, x2, y2 = coords_by_run[run_num]

            try:
                # Extract velocity components for this run directly from struct
                # (matching dotboard's pattern from vector_calibration_production.py:302-306)
                run_idx = run_num - 1  # 0-based index
                if run_idx >= len(piv_result_raw1) or run_idx >= len(piv_result_raw2):
                    continue

                cell1 = piv_result_raw1[run_idx]
                cell2 = piv_result_raw2[run_idx]

                ux1_px = getattr(cell1, "ux", None)
                uy1_px = getattr(cell1, "uy", None)
                ux2_px = getattr(cell2, "ux", None)
                uy2_px = getattr(cell2, "uy", None)

                # Skip if velocity data is missing or empty
                if ux1_px is None or uy1_px is None or ux2_px is None or uy2_px is None:
                    if frame_idx == 1:
                        logger.warning(f"Run {run_num}: velocity data is None")
                    continue
                if not hasattr(ux1_px, 'size') or ux1_px.size == 0:
                    if frame_idx == 1:
                        logger.warning(f"Run {run_num}: velocity data is empty")
                    continue

                # Convert to numpy arrays
                ux1_px = np.asarray(ux1_px)
                uy1_px = np.asarray(uy1_px)
                ux2_px = np.asarray(ux2_px)
                uy2_px = np.asarray(uy2_px)

                # Extract b_mask from each camera (True = invalid)
                b_mask_1 = getattr(cell1, "b_mask", None)
                b_mask_2 = getattr(cell2, "b_mask", None)

                # Combine masks: conservative (invalid if EITHER masked)
                if b_mask_1 is not None and b_mask_2 is not None:
                    b_mask_1 = np.asarray(b_mask_1).astype(bool)
                    b_mask_2 = np.asarray(b_mask_2).astype(bool)
                    combined_input_mask = b_mask_1 | b_mask_2
                elif b_mask_1 is not None:
                    combined_input_mask = np.asarray(b_mask_1).astype(bool)
                elif b_mask_2 is not None:
                    combined_input_mask = np.asarray(b_mask_2).astype(bool)
                else:
                    combined_input_mask = np.zeros_like(ux1_px, dtype=bool)

                # Shape validation (matching dotboard's pattern from vector_calibration_production.py:318-326)
                # Ensures coordinates and velocities have compatible grid sizes
                if x1.shape != ux1_px.shape or y1.shape != uy1_px.shape:
                    # Grid size mismatch for camera 1 - log and continue
                    # Only log once per run (on first frame) to avoid spam
                    if frame_idx == 1:
                        logger.warning(
                            f"Run {run_num}: Cam1 shape mismatch - "
                            f"coords {x1.shape} vs velocity {ux1_px.shape}"
                        )
                    continue

                if x2.shape != ux2_px.shape or y2.shape != uy2_px.shape:
                    # Grid size mismatch for camera 2 - log and continue
                    if frame_idx == 1:
                        logger.warning(
                            f"Run {run_num}: Cam2 shape mismatch - "
                            f"coords {x2.shape} vs velocity {ux2_px.shape}"
                        )
                    continue

                # Perform 3D reconstruction
                result_3d = _reconstruct_3d_velocities(
                    ux1_px,
                    uy1_px,
                    ux2_px,
                    uy2_px,
                    (x1, y1),
                    (x2, y2),
                    stereo_params,
                    min_angle,
                    combined_input_mask,
                )

                # Create output grid matching camera 1 coordinates
                ref_shape = x1.shape
                ux_grid = np.full(ref_shape, np.nan, dtype=np.float64)
                uy_grid = np.full(ref_shape, np.nan, dtype=np.float64)
                uz_grid = np.full(ref_shape, np.nan, dtype=np.float64)
                # Create output mask: True where NOT successfully reconstructed
                output_mask = np.full(ref_shape, True, dtype=bool)  # Default: all masked

                if result_3d["num_valid"] > 0:
                    # Convert mm to m and divide by dt for velocity (m/s)
                    velocities_mps = (result_3d["velocities_3d"] / 1000.0) / max(dt, 1e-12)
                    valid_indices = result_3d["indices1"]
                    row_indices, col_indices = np.unravel_index(valid_indices, ref_shape)
                    ux_grid[row_indices, col_indices] = velocities_mps[:, 0]
                    uy_grid[row_indices, col_indices] = velocities_mps[:, 1]
                    # Negate uz because OpenCV stereo Z-axis convention differs from physical coords
                    # This matches the uy negation done in save_results.py for planar PIV
                    uz_grid[row_indices, col_indices] = -velocities_mps[:, 2]
                    output_mask[row_indices, col_indices] = False  # Valid points: not masked

                    # Store result for this run (for coordinate saving)
                    run_results[run_num] = result_3d
                    total_valid += result_3d["num_valid"]

                # Store in piv_result array
                piv_result[run_num - 1] = (ux_grid, uy_grid, uz_grid, output_mask)

            except Exception as run_error:
                # Log error but continue with other runs
                logger.warning(f"Run {run_num}: Failed to process frame {frame_idx} - {run_error}")

        # Save result with ALL runs
        savemat(output_file_path, {"piv_result": piv_result})

        return {
            "frame": frame_idx,
            "success": True,
            "num_valid": total_valid,
            "run_results": run_results,  # Per-run 3D results for coordinate saving
        }

    except Exception as e:
        return {"frame": frame_idx, "success": False, "error": str(e)}


def _load_stereo_model(
    base_dir: Path, cam1: int, cam2: int, model_type: str
) -> Dict[str, np.ndarray]:
    """
    Load stereo calibration model from appropriate path.

    Args:
        base_dir: Base directory for calibrated data
        cam1, cam2: Camera numbers
        model_type: 'charuco' or 'dotboard' - sets expectation but doesn't change path

    Returns:
        Dict with stereo calibration parameters (numpy arrays)

    Raises:
        FileNotFoundError: If no stereo model found
        ValueError: If required fields missing
    """
    stereo_file = base_dir / "calibration" / f"stereo_cam{cam1}_cam{cam2}" / "model" / "stereo_model.mat"

    if not stereo_file.exists():
        raise FileNotFoundError(f"Stereo calibration not found at: {stereo_file}")

    logger.info(f"Loading stereo model from: {stereo_file}")

    stereo_data = loadmat(str(stereo_file), squeeze_me=True, struct_as_record=False)

    # Validate required fields
    required_fields = [
        "camera_matrix_1",
        "camera_matrix_2",
        "dist_coeffs_1",
        "dist_coeffs_2",
        "rotation_matrix",
        "translation_vector",
        "projection_P1",
        "projection_P2",
        "rectification_R1",
        "rectification_R2",
    ]

    missing = [f for f in required_fields if f not in stereo_data]
    if missing:
        raise ValueError(f"Missing required fields in stereo calibration: {missing}")

    # Optional: Check pattern_type matches model_type
    if "pattern_params" in stereo_data:
        params = stereo_data["pattern_params"]
        if hasattr(params, "pattern_type"):
            detected_type = params.pattern_type
            if detected_type == "circle_grid":
                detected_type = "dotboard"
            if detected_type != model_type:
                logger.warning(
                    f"Model type mismatch: config says '{model_type}', "
                    f"model has '{detected_type}'"
                )

    # Return as serializable dict for worker processes
    return {
        "camera_matrix_1": np.array(stereo_data["camera_matrix_1"]).astype(np.float64),
        "camera_matrix_2": np.array(stereo_data["camera_matrix_2"]).astype(np.float64),
        "dist_coeffs_1": np.array(stereo_data["dist_coeffs_1"]).flatten().astype(np.float64),
        "dist_coeffs_2": np.array(stereo_data["dist_coeffs_2"]).flatten().astype(np.float64),
        "rotation_matrix": np.array(stereo_data["rotation_matrix"]).astype(np.float64),
        "translation_vector": np.array(stereo_data["translation_vector"]).astype(np.float64),
        "projection_P1": np.array(stereo_data["projection_P1"]).astype(np.float64),
        "projection_P2": np.array(stereo_data["projection_P2"]).astype(np.float64),
        "rectification_R1": np.array(stereo_data["rectification_R1"]).astype(np.float64),
        "rectification_R2": np.array(stereo_data["rectification_R2"]).astype(np.float64),
        "disparity_to_depth_Q": np.array(
            stereo_data.get("disparity_to_depth_Q", np.eye(4))
        ).astype(np.float64),
        "tvecs_1": np.array(
            stereo_data.get("tvecs_1", [[0, 0, 0]])
        ).astype(np.float64),
    }


class StereoReconstructor:
    """
    Reconstructs 3D velocities from stereo camera pair PIV data.

    Supports both ChArUco and Pinhole stereo calibration models.
    Uses parallel processing with ProcessPoolExecutor.
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        camera_pair: Optional[List[int]] = None,
        model_type: Optional[str] = None,
        dt: Optional[float] = None,
        vector_pattern: Optional[str] = None,
        type_name: str = "instantaneous",
        runs: Optional[List[int]] = None,
        num_workers: Optional[int] = None,
        min_angle: float = 5.0,
        config=None,
    ):
        """
        Initialize stereo reconstructor.

        Parameters can be provided explicitly or read from config. When config
        is provided, it takes precedence for settings stored in config.yaml.

        Args:
            base_dir: Base directory containing data (or from config.base_paths[0])
            camera_pair: Camera pair [cam1, cam2] (or from config.stereo_calibration)
            model_type: 'charuco' or 'dotboard' (or from config.stereo_dotboard_calibration)
            dt: Time step between frames in seconds (or from config.dt)
            vector_pattern: Pattern for vector files (or from config.vector_format)
            type_name: Type name for data directory
            runs: List of 1-indexed run numbers to process, or None for all
            num_workers: Number of parallel workers, None = os.cpu_count()
            min_angle: Minimum triangulation angle in degrees
            config: Optional Config object to read settings from
        """
        self._config = config

        # Read from config if provided
        if config is not None:
            self.base_dir = Path(base_dir) if base_dir else config.base_paths[0]
            stereo_cfg = config.stereo_dotboard_calibration
            self.camera_pair = camera_pair or stereo_cfg.get("camera_pair", [1, 2])
            self.model_type = model_type or stereo_cfg.get("stereo_model_type", "charuco")
            self.dt = dt if dt is not None else config.dt
            self.vector_pattern = vector_pattern or config.vector_format
            self.num_frame_pairs = config.num_frame_pairs
        else:
            if base_dir is None:
                raise ValueError("base_dir required when config not provided")
            self.base_dir = Path(base_dir)
            self.camera_pair = camera_pair or [1, 2]
            self.model_type = model_type or "charuco"
            self.dt = dt or 1.0
            self.vector_pattern = vector_pattern or "%05d.mat"
            self.num_frame_pairs = None

        self.type_name = type_name
        self.runs = runs  # 1-indexed
        self.num_workers = num_workers or os.cpu_count()
        self.min_angle = min_angle

        # Validate model type
        if self.model_type not in ("charuco", "dotboard"):
            raise ValueError(f"model_type must be 'charuco' or 'dotboard', got '{self.model_type}'")

        # Load stereo calibration model
        self.stereo_params = _load_stereo_model(
            self.base_dir, self.camera_pair[0], self.camera_pair[1], self.model_type
        )

        logger.info("Initialized StereoReconstructor")
        logger.info(f"  Base directory: {self.base_dir}")
        logger.info(f"  Camera pair: {self.camera_pair}")
        logger.info(f"  Model type: {self.model_type}")
        logger.info(f"  Time step: {self.dt} seconds")
        logger.info(f"  Vector pattern: {self.vector_pattern}")
        logger.info(f"  Type name: {self.type_name}")
        logger.info(f"  Runs to process: {self.runs if self.runs else 'all'}")
        logger.info(f"  Worker count: {self.num_workers}")
        logger.info(f"  Min triangulation angle: {self.min_angle} degrees")

    def _find_valid_runs(
        self,
        x1_list: List[np.ndarray],
        y1_list: List[np.ndarray],
        x2_list: List[np.ndarray],
        y2_list: List[np.ndarray],
    ) -> List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Find runs with valid coordinate data in both cameras.

        Returns:
            List of tuples: (list_idx, run_num, x1, y1, x2, y2)
        """
        valid_runs = []

        for i, (x1, y1, x2, y2) in enumerate(zip(x1_list, y1_list, x2_list, y2_list)):
            # Map to original run number
            if self.runs:
                run_num = self.runs[i]
            else:
                run_num = i + 1

            # Handle None arrays
            if x1 is None:
                x1 = np.array([])
            if y1 is None:
                y1 = np.array([])
            if x2 is None:
                x2 = np.array([])
            if y2 is None:
                y2 = np.array([])

            valid_coords1 = np.sum(~np.isnan(x1)) if x1.size > 0 else 0
            valid_coords2 = np.sum(~np.isnan(x2)) if x2.size > 0 else 0

            logger.info(
                f"Run {run_num}: Cam{self.camera_pair[0]}={valid_coords1}, "
                f"Cam{self.camera_pair[1]}={valid_coords2} valid coordinates"
            )

            if valid_coords1 > 0 and valid_coords2 > 0:
                valid_runs.append((i, run_num, x1, y1, x2, y2))

        return valid_runs

    def _save_stereo_coordinates(
        self,
        valid_runs: List[Tuple],
        output_dir: Path,
        run_results: Dict[int, Dict[str, Any]],
    ):
        """
        Save 3D coordinates from stereo reconstruction.
        Matches dotboard's VectorCalibrator pattern (vector_calibration_production.py lines 830-853).

        Args:
            valid_runs: List of valid run tuples (list_idx, run_num, x1, y1, x2, y2)
            output_dir: Output directory
            run_results: Dict mapping run_num -> 3D reconstruction result
        """
        if not run_results:
            logger.warning("No valid 3D positions to save for coordinates")
            return

        # Step 1: Create coordinate structure (like dotboard lines 830-837)
        max_run = max(r[1] for r in valid_runs)
        coord_dtype = np.dtype([("x", "O"), ("y", "O"), ("z", "O")])
        coordinates = np.empty(max_run, dtype=coord_dtype)

        # Initialize all runs with empty arrays
        for run_num in range(1, max_run + 1):
            coordinates[run_num - 1] = (np.array([]), np.array([]), np.array([]))

        # Get Z offset from first calibration image (shared across all runs)
        z_offset = 0.0
        if "tvecs_1" in self.stereo_params:
            tvecs_1 = self.stereo_params["tvecs_1"]
            if tvecs_1.size > 0:
                z_offset = tvecs_1[0, 2]
                logger.info(f"Z reference (first calibration image): {z_offset:.2f} mm")

        # Step 2: Fill in each run using its OWN result (like dotboard lines 840-847)
        for _, run_num, x1, _, _, _ in valid_runs:
            if run_num not in run_results:
                continue

            result_3d = run_results[run_num]  # THIS run's result
            if result_3d["num_valid"] == 0:
                continue

            ref_shape = x1.shape
            x_grid = np.full(ref_shape, np.nan, dtype=np.float64)
            y_grid = np.full(ref_shape, np.nan, dtype=np.float64)
            z_grid = np.full(ref_shape, np.nan, dtype=np.float64)

            valid_indices = result_3d["indices1"]
            row_indices, col_indices = np.unravel_index(valid_indices, ref_shape)
            positions_3d = result_3d["positions_3d"]

            # Center X,Y only; reference Z to calibration plate plane
            mean_xy = np.mean(positions_3d[:, :2], axis=0)

            x_grid[row_indices, col_indices] = positions_3d[:, 0] - mean_xy[0]
            y_grid[row_indices, col_indices] = positions_3d[:, 1] - mean_xy[1]
            z_grid[row_indices, col_indices] = positions_3d[:, 2] - z_offset

            coordinates[run_num - 1] = (x_grid, y_grid, z_grid)
            logger.info(f"Run {run_num}: saved {result_3d['num_valid']} 3D positions")

        # Step 3: Save all coordinates (like dotboard lines 849-853)
        coords_path = output_dir / "coordinates.mat"
        savemat(str(coords_path), {"coordinates": coordinates})
        logger.info(f"Saved 3D coordinates: {coords_path}")

    def _process_all_frames_parallel(
        self,
        coords_by_run: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        uncalib_dir1: Path,
        uncalib_dir2: Path,
        output_dir: Path,
        num_frames: int,
        max_run: int,
        valid_run_nums: Set[int],
        progress_cb: Optional[Callable[[Dict[str, Any]], None]],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process all frames for ALL runs in a single pass (like dotboard's pattern).

        Args:
            coords_by_run: Dict mapping run_num -> (x1, y1, x2, y2) coordinates
            uncalib_dir1, uncalib_dir2: Uncalibrated data directories
            output_dir: Output directory
            num_frames: Total number of frames
            max_run: Maximum run number
            valid_run_nums: Set of valid run numbers
            progress_cb: Optional progress callback

        Returns:
            Dict mapping run_num -> 3D result for coordinate saving
        """
        logger.info(f"Processing all {len(valid_run_nums)} runs with {self.num_workers} workers")

        # Build task list - ONE task per frame (not per run!)
        tasks = []
        for frame_idx in range(1, num_frames + 1):
            vec_file1 = uncalib_dir1 / (self.vector_pattern % frame_idx)
            vec_file2 = uncalib_dir2 / (self.vector_pattern % frame_idx)

            if not vec_file1.exists() or not vec_file2.exists():
                continue

            output_file = output_dir / (self.vector_pattern % frame_idx)

            tasks.append((
                frame_idx,
                str(vec_file1),
                str(vec_file2),
                str(output_file),
                coords_by_run,  # Pass ALL runs' coordinates
                self.stereo_params,
                self.dt,
                self.min_angle,
                max_run,
                valid_run_nums,
            ))

        if not tasks:
            logger.warning("No vector files found")
            return {}

        logger.info(f"Processing {len(tasks)} frames for {len(valid_run_nums)} runs")

        successful = 0
        failed = 0
        # Accumulate per-run results across all frames
        all_run_results: Dict[int, Dict[str, Any]] = {}

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(_process_stereo_frame, task): task[0]
                for task in tasks
            }

            for future in as_completed(futures):
                result = future.result()
                if result and result.get("success"):
                    successful += 1

                    # Capture first successful result per run for coordinate saving
                    frame_run_results = result.get("run_results", {})
                    for run_num, run_result in frame_run_results.items():
                        if run_num not in all_run_results and run_result.get("num_valid", 0) > 0:
                            all_run_results[run_num] = run_result
                            logger.info(f"Run {run_num}: captured 3D result with {run_result['num_valid']} valid positions")
                else:
                    failed += 1
                    if result and "error" in result:
                        logger.debug(f"Frame {result['frame']} failed: {result['error']}")

                # Progress callback - single progress bar for all runs
                # Field names match dotboard's vector_calibration_production.py for consistency
                if progress_cb:
                    total_done = successful + failed
                    try:
                        progress_cb({
                            "camera_pair": self.camera_pair,
                            "processed_frames": total_done,
                            "total_frames": len(tasks),
                            "progress": (total_done / len(tasks)) * 100,
                            "successful_frames": successful,
                            "failed_frames": failed,
                        })
                    except Exception:
                        pass

        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        logger.info(f"Captured 3D results for {len(all_run_results)} runs: {sorted(all_run_results.keys())}")
        return all_run_results

    def process_run(
        self,
        num_frame_pairs: Optional[int] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Process stereo reconstruction for all frames.

        Args:
            num_frame_pairs: Number of frame pairs. If None, uses config value.
            progress_cb: Optional callback for progress updates
        """
        # Use config value if not explicitly provided
        if num_frame_pairs is None:
            num_frame_pairs = self.num_frame_pairs
        if num_frame_pairs is None:
            raise ValueError(
                "num_frame_pairs must be provided either to __init__ via config or to process_run()"
            )

        logger.info(f"Processing stereo reconstruction with {num_frame_pairs} frame pairs")

        cam1, cam2 = self.camera_pair

        # Get data paths for both cameras
        paths1 = get_data_paths(
            self.base_dir,
            num_frame_pairs,
            cam1,
            self.type_name,
            use_uncalibrated=True,
        )
        paths2 = get_data_paths(
            self.base_dir,
            num_frame_pairs,
            cam2,
            self.type_name,
            use_uncalibrated=True,
        )

        uncalib_dir1 = paths1["data_dir"]
        uncalib_dir2 = paths2["data_dir"]

        logger.info(f"Uncalibrated data camera {cam1}: {uncalib_dir1}")
        logger.info(f"Uncalibrated data camera {cam2}: {uncalib_dir2}")

        # Output uses dedicated stereo path structure
        output_paths = get_data_paths(
            self.base_dir,
            num_frame_pairs,
            cam=cam1,  # Reference camera (not used for stereo path construction)
            type_name=self.type_name,
            use_stereo=True,
            stereo_camera_pair=self.camera_pair,
        )
        output_dir = output_paths["data_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory (stereo): {output_dir}")

        # Check directories exist
        if not uncalib_dir1.exists():
            raise FileNotFoundError(f"Uncalibrated data not found: {uncalib_dir1}")
        if not uncalib_dir2.exists():
            raise FileNotFoundError(f"Uncalibrated data not found: {uncalib_dir2}")

        # Load coordinates for both cameras
        logger.info("Loading coordinates...")
        x1_list, y1_list = load_coords_from_directory(uncalib_dir1, runs=self.runs)
        x2_list, y2_list = load_coords_from_directory(uncalib_dir2, runs=self.runs)

        if not x1_list:
            raise ValueError(f"No coordinate data found for camera {cam1}")
        if not x2_list:
            raise ValueError(f"No coordinate data found for camera {cam2}")

        # Match number of runs between cameras
        if len(x1_list) != len(x2_list):
            min_runs = min(len(x1_list), len(x2_list))
            x1_list, y1_list = x1_list[:min_runs], y1_list[:min_runs]
            x2_list, y2_list = x2_list[:min_runs], y2_list[:min_runs]
            logger.warning(f"Adjusted to {min_runs} runs to match both cameras")

        logger.info(f"Loaded coordinates for {len(x1_list)} runs")

        # Find runs with valid data
        valid_runs = self._find_valid_runs(x1_list, y1_list, x2_list, y2_list)

        if not valid_runs:
            raise ValueError("No runs with valid coordinate data found")

        logger.info(f"Found {len(valid_runs)} runs with valid data: {[r[1] for r in valid_runs]}")

        max_run = max(r[1] for r in valid_runs)
        valid_run_nums = set(r[1] for r in valid_runs)

        # Build coords_by_run dict mapping run_num -> (x1, y1, x2, y2)
        # Following dotboard's pattern (vector_calibration_production.py lines 859-862)
        coords_by_run = {
            run_num: (x1, y1, x2, y2)
            for _, run_num, x1, y1, x2, y2 in valid_runs
        }
        logger.info(f"Coordinates available for runs: {sorted(coords_by_run.keys())}")

        # Process ALL frames in a single pass (like dotboard)
        # This ensures one progress bar and all runs processed together
        run_results = self._process_all_frames_parallel(
            coords_by_run,
            uncalib_dir1, uncalib_dir2, output_dir,
            num_frame_pairs, max_run, valid_run_nums,
            progress_cb,
        )

        # Save 3D coordinates using each run's own data
        if run_results:
            self._save_stereo_coordinates(valid_runs, output_dir, run_results)

        # Save reconstruction summary
        summary_data = {
            "reconstruction_summary": {
                "camera_pair": self.camera_pair,
                "model_type": self.model_type,
                "output_directory": str(output_dir),
                "configuration": {
                    "min_triangulation_angle": self.min_angle,
                    "vector_pattern": self.vector_pattern,
                    "type_name": self.type_name,
                    "num_frame_pairs": num_frame_pairs,
                    "dt": self.dt,
                    "num_workers": self.num_workers,
                },
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Strip private keys from stereo params for saving
        stereo_params_clean = {k: v for k, v in self.stereo_params.items() if not k.startswith("_")}
        summary_data["stereo_calibration"] = stereo_params_clean

        summary_file = output_dir / "stereo_reconstruction_summary.mat"
        savemat(str(summary_file), summary_data)
        logger.info(f"Saved reconstruction summary: {summary_file}")


def main():
    """Main entry point for stereo reconstruction.

    When USE_CONFIG_DIRECTLY=True, loads settings from existing config.yaml instead
    of applying the hardcoded CLI settings.
    """
    logger.info("=" * 60)
    logger.info("Stereo Reconstruction - Starting")
    logger.info("=" * 60)

    if USE_CONFIG_DIRECTLY:
        # Load settings directly from existing config.yaml
        logger.info("Loading settings directly from config.yaml (USE_CONFIG_DIRECTLY=True)")
        config = get_config()

        # Log settings from config
        stereo_cfg = config.stereo_calibration
        logger.info(f"Base directory: {config.base_paths[0]}")
        logger.info(f"Camera pair: {stereo_cfg.get('camera_pair', [1, 2])}")
        logger.info(f"Num frame pairs: {config.num_frame_pairs}")
        logger.info(f"Time step: {config.dt} seconds")
        logger.info(f"Model type: {stereo_cfg.get('stereo_model_type', 'charuco')}")
        logger.info(f"Vector pattern: {config.vector_format}")
        logger.info(f"Type name: {TYPE_NAME}")
        logger.info(f"Min triangulation angle: {MIN_TRIANGULATION_ANGLE} degrees")
        logger.info(f"Runs to process: {RUNS_TO_PROCESS if RUNS_TO_PROCESS else 'all'}")
        logger.info(f"Worker count: {NUM_WORKERS if NUM_WORKERS else 'auto'}")
    else:
        # Log hardcoded settings and apply to config
        logger.info(f"Base directory: {BASE_DIR}")
        logger.info(f"Camera pair: {CAMERA_PAIR}")
        logger.info(f"Num frame pairs: {NUM_FRAME_PAIRS}")
        logger.info(f"Time step: {DT_SECONDS} seconds")
        logger.info(f"Model type: {MODEL_TYPE}")
        logger.info(f"Vector pattern: {VECTOR_PATTERN}")
        logger.info(f"Type name: {TYPE_NAME}")
        logger.info(f"Min triangulation angle: {MIN_TRIANGULATION_ANGLE} degrees")
        logger.info(f"Runs to process: {RUNS_TO_PROCESS if RUNS_TO_PROCESS else 'all'}")
        logger.info(f"Worker count: {NUM_WORKERS if NUM_WORKERS else 'auto'}")

        # Apply CLI settings to config.yaml so centralized systems work correctly
        config = apply_cli_settings_to_config()

    try:
        reconstructor = StereoReconstructor(
            type_name=TYPE_NAME,
            runs=RUNS_TO_PROCESS,
            num_workers=NUM_WORKERS,
            min_angle=MIN_TRIANGULATION_ANGLE,
            config=config,
        )

        reconstructor.process_run()

        logger.info("=" * 60)
        logger.info("Stereo Reconstruction - Complete")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Stereo reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
