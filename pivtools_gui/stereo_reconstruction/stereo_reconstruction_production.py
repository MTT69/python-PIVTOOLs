#!/usr/bin/env python3
"""
stereo_reconstruction_production.py

Production script for 3D velocity reconstruction from stereo camera pairs.
Takes uncalibrated 2D velocity fields from two cameras and reconstructs 3D velocities (ux, uy, uz).

Uses the Willert (1997) / Soloff (1997) geometric reconstruction approach:
  - Coordinates: single-camera projection (cam1 grid → world plane at Z=z_world)
  - Velocities: Jacobian-based geometric decomposition of paired 2D displacements
  - Cross-camera correspondence: project cam1 world grid into cam2 image, interpolate cam2 displacements

Supports ChArUco, Dotboard, and Stepped Board stereo calibration models.
Uses ProcessPoolExecutor for parallel frame processing.
"""

import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat, savemat

sys.path.append(str(Path(__file__).parent.parent))
from pivtools_core.config import get_config, reload_config
from pivtools_core.paths import get_data_paths
from pivtools_core.vector_loading import load_coords_from_directory, read_mat_contents
from pivtools_gui.calibration.vector_calibration_production import _pixels_to_world_mm
from pivtools_gui.utils.worker_pool import worker_initializer, get_max_workers

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


def _project_coords_to_world(
    coords_x_uncal: np.ndarray,
    coords_y_uncal: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    image_height: int,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project cam1 PIV grid to world coordinates on the laser sheet plane.

    Uses single-camera ray-plane intersection via _pixels_to_world_mm().
    This is the same projection the planar (pinhole) calibration uses,
    ensuring consistency between planar and stereo coordinate systems.

    Args:
        coords_x_uncal: (H, W) uncalibrated x-coordinates (1-based)
        coords_y_uncal: (H, W) uncalibrated y-coordinates (y-up)
        camera_matrix: (3, 3) intrinsic K for reference camera
        dist_coeffs: Distortion coefficients for reference camera
        rvec: (3,) rotation vector for reference camera
        tvec: (3,) translation vector for reference camera
        image_height: Image height in pixels
        z_world: Z-offset of laser sheet from calibration plane (mm)
        tilt_x: Tilt about X-axis (radians, from self-calibration)
        tilt_y: Tilt about Y-axis (radians, from self-calibration)

    Returns:
        (x_world_mm, y_world_mm): World coordinates in mm, each shape (H, W).
        Y is negated for board-Y → physical Y-up convention.
    """
    shape = coords_x_uncal.shape

    # Convert uncalibrated (1-based, y-up) → raw pixels (0-based, y-down)
    raw_x = coords_x_uncal.flatten() - 1
    raw_y = image_height - coords_y_uncal.flatten()
    pts_raw = np.column_stack([raw_x, raw_y])

    # Project to world plane via single-camera ray-plane intersection
    world_pts = _pixels_to_world_mm(
        pts_raw, camera_matrix, dist_coeffs, rvec, tvec,
        z_world=z_world, tilt_x=tilt_x, tilt_y=tilt_y,
    )

    x_world = world_pts[:, 0].reshape(shape)
    # Board Y direction depends on how calibration target defines axes.
    # _pixels_to_world_mm returns world coords in the board frame directly.
    # No blanket negation — the board's Y convention is determined by the
    # calibration target orientation and fiducial layout.
    y_world = world_pts[:, 1].reshape(shape)

    return x_world, y_world


def _interpolate_cam2_displacements(
    world_pts: np.ndarray,
    ux2_px: np.ndarray,
    uy2_px: np.ndarray,
    coords2_x_uncal: np.ndarray,
    coords2_y_uncal: np.ndarray,
    cam2_K: np.ndarray,
    cam2_dist: np.ndarray,
    cam2_rvec: np.ndarray,
    cam2_tvec: np.ndarray,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into cam2 image and interpolate cam2 displacements.

    Establishes proper cross-camera correspondence by projecting cam1's
    world-space grid into cam2's image and interpolating cam2's displacement
    field at those locations.

    Args:
        world_pts: (N, 3) world XYZ coordinates on laser sheet
        ux2_px: (H2, W2) cam2 x-displacement in pixels/frame
        uy2_px: (H2, W2) cam2 y-displacement in pixels/frame
        coords2_x_uncal: (H2, W2) cam2 uncalibrated x-coordinates
        coords2_y_uncal: (H2, W2) cam2 uncalibrated y-coordinates
        cam2_K: (3, 3) cam2 intrinsic matrix
        cam2_dist: cam2 distortion coefficients
        cam2_rvec: (3,) cam2 rotation vector
        cam2_tvec: (3,) cam2 translation vector
        image_height: Image height in pixels

    Returns:
        (dx2_interp, dy2_interp, valid_mask): Interpolated cam2 displacements
        (N,) each, and boolean mask of successfully interpolated points.
    """
    N = world_pts.shape[0]
    dx2_out = np.full(N, np.nan)
    dy2_out = np.full(N, np.nan)
    valid_mask = np.zeros(N, dtype=bool)

    if N == 0:
        return dx2_out, dy2_out, valid_mask

    # Project world points into cam2's image (raw pixel coordinates)
    img_pts2, _ = cv2.projectPoints(
        world_pts.astype(np.float64),
        cam2_rvec.astype(np.float64),
        cam2_tvec.astype(np.float64),
        cam2_K.astype(np.float64),
        cam2_dist.astype(np.float64),
    )
    projected_raw = img_pts2.reshape(-1, 2)  # (N, 2) in raw pixels (0-based, y-down)

    # Convert projected raw pixels to uncalibrated convention (1-based, y-up)
    projected_uncal_x = projected_raw[:, 0] + 1
    projected_uncal_y = image_height - projected_raw[:, 1]

    # Build interpolator on cam2's uncalibrated grid.
    # The grid must be regular (evenly spaced PIV windows).
    # Extract unique sorted axis vectors from the 2D grid arrays.
    y_axis = coords2_y_uncal[:, 0]   # column of y values (varies along rows)
    x_axis = coords2_x_uncal[0, :]   # row of x values (varies along columns)

    # RegularGridInterpolator expects axes in ascending order
    y_ascending = np.all(np.diff(y_axis) > 0)
    if not y_ascending:
        y_axis = y_axis[::-1]
        ux2_px = ux2_px[::-1, :]
        uy2_px = uy2_px[::-1, :]

    x_ascending = np.all(np.diff(x_axis) > 0)
    if not x_ascending:
        x_axis = x_axis[::-1]
        ux2_px = ux2_px[:, ::-1]
        uy2_px = uy2_px[:, ::-1]

    try:
        interp_ux = RegularGridInterpolator(
            (y_axis, x_axis), ux2_px,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        interp_uy = RegularGridInterpolator(
            (y_axis, x_axis), uy2_px,
            method="linear", bounds_error=False, fill_value=np.nan,
        )

        # Evaluate at projected locations (y, x) order for RegularGridInterpolator
        query_pts = np.column_stack([projected_uncal_y, projected_uncal_x])
        dx2_out = interp_ux(query_pts)
        dy2_out = interp_uy(query_pts)

        valid_mask = np.isfinite(dx2_out) & np.isfinite(dy2_out)
    except Exception as e:
        logger.debug(f"Cam2 interpolation failed: {e}")

    return dx2_out, dy2_out, valid_mask


def _compute_projection_jacobian(
    world_pts: np.ndarray,
    cam_K: np.ndarray,
    cam_dist: np.ndarray,
    cam_rvec: np.ndarray,
    cam_tvec: np.ndarray,
    delta: float = 0.01,
) -> np.ndarray:
    """Compute Jacobian of camera projection d(pixel)/d(world) numerically.

    For each world point, perturbs by ±delta in X, Y, Z and projects through
    the camera model. Central differences give the 2×3 Jacobian per point.
    Uses 6 vectorized cv2.projectPoints calls (all N points per call).

    Args:
        world_pts: (N, 3) world coordinates
        cam_K: (3, 3) intrinsic matrix
        cam_dist: Distortion coefficients
        cam_rvec: (3,) rotation vector
        cam_tvec: (3,) translation vector
        delta: Perturbation size in mm (default 0.01)

    Returns:
        J: (N, 2, 3) Jacobian where J[i, :, j] = d(pixel_xy)/d(world_j) for point i
    """
    N = world_pts.shape[0]
    J = np.zeros((N, 2, 3), dtype=np.float64)

    rvec = cam_rvec.astype(np.float64)
    tvec = cam_tvec.astype(np.float64)
    K = cam_K.astype(np.float64)
    dist = cam_dist.astype(np.float64)

    for axis in range(3):
        pts_plus = world_pts.copy()
        pts_plus[:, axis] += delta
        pts_minus = world_pts.copy()
        pts_minus[:, axis] -= delta

        px_plus, _ = cv2.projectPoints(pts_plus, rvec, tvec, K, dist)
        px_minus, _ = cv2.projectPoints(pts_minus, rvec, tvec, K, dist)

        J[:, :, axis] = (px_plus.reshape(-1, 2) - px_minus.reshape(-1, 2)) / (2 * delta)

    return J


def _compute_stereo_angle(
    world_pts: np.ndarray,
    cam1_rvec: np.ndarray,
    cam1_tvec: np.ndarray,
    cam2_rvec: np.ndarray,
    cam2_tvec: np.ndarray,
) -> np.ndarray:
    """Compute the stereo angle (angle between viewing rays) at each point.

    Used as a quality metric: larger angles give better depth/Uz resolution.

    Args:
        world_pts: (N, 3) world coordinates
        cam1_rvec, cam1_tvec: Camera 1 extrinsics
        cam2_rvec, cam2_tvec: Camera 2 extrinsics

    Returns:
        angles_deg: (N,) stereo angles in degrees
    """
    R1, _ = cv2.Rodrigues(cam1_rvec)
    R2, _ = cv2.Rodrigues(cam2_rvec)

    # Camera centers in world frame: C = -R^T @ t
    cam1_center = -R1.T @ cam1_tvec.flatten()
    cam2_center = -R2.T @ cam2_tvec.flatten()

    vec1 = world_pts - cam1_center
    vec2 = world_pts - cam2_center

    vec1_norm = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
    vec2_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)

    dot_products = np.sum(vec1_norm * vec2_norm, axis=1)
    angles_rad = np.arccos(np.clip(dot_products, -1, 1))

    return np.degrees(angles_rad)


def _reconstruct_3d_velocities(
    dx1_px: np.ndarray,
    dy1_px: np.ndarray,
    dx2_interp: np.ndarray,
    dy2_interp: np.ndarray,
    world_pts: np.ndarray,
    stereo_params: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    """Reconstruct 3-component velocity from paired 2D displacements.

    Uses the Willert/Soloff geometric decomposition: for each grid point,
    stacks the projection Jacobians from both cameras into a 4×3 system
    and solves for (Ux, Uy, Uz) in world mm/frame via least-squares.

    Args:
        dx1_px, dy1_px: (N,) cam1 pixel displacements
        dx2_interp, dy2_interp: (N,) interpolated cam2 pixel displacements
        world_pts: (N, 3) world coordinates of grid points
        stereo_params: Dict with cam1/cam2 K, dist, rvec, tvec
        valid_mask: (N,) boolean — True where both cameras have valid data

    Returns:
        Dict with velocities_3d (M, 3) in mm/frame, valid_indices (M,),
        num_valid, num_total
    """
    N = world_pts.shape[0]
    valid_idx = np.where(valid_mask)[0]
    M = len(valid_idx)

    if M == 0:
        return {
            "velocities_3d": np.array([]).reshape(0, 3),
            "valid_indices": np.array([], dtype=int),
            "num_valid": 0,
            "num_total": N,
        }

    # Extract valid subset
    pts_valid = world_pts[valid_idx]
    dx1 = dx1_px[valid_idx]
    dy1 = dy1_px[valid_idx]
    dx2 = dx2_interp[valid_idx]
    dy2 = dy2_interp[valid_idx]

    # Compute projection Jacobians for both cameras at valid points
    J1 = _compute_projection_jacobian(
        pts_valid,
        stereo_params["camera_matrix_1"], stereo_params["dist_coeffs_1"],
        stereo_params["cam1_rvec"], stereo_params["cam1_tvec"],
    )  # (M, 2, 3)

    J2 = _compute_projection_jacobian(
        pts_valid,
        stereo_params["camera_matrix_2"], stereo_params["dist_coeffs_2"],
        stereo_params["cam2_rvec"], stereo_params["cam2_tvec"],
    )  # (M, 2, 3)

    # Stack into 4×3 system per point: A @ [Ux, Uy, Uz]^T = b
    A = np.concatenate([J1, J2], axis=1)  # (M, 4, 3)
    b = np.column_stack([dx1, dy1, dx2, dy2])  # (M, 4)

    # Vectorized least-squares: solve A^T A x = A^T b (normal equations)
    AtA = np.einsum('nij,nik->njk', A, A)  # (M, 3, 3)
    Atb = np.einsum('nij,ni->nj', A, b)     # (M, 3)

    # Solve for [Ux, Uy, Uz] in mm/frame
    # np.linalg.solve batched: (M, 3, 3) @ (M, 3, 1) -> (M, 3, 1)
    try:
        vel_3d = np.linalg.solve(AtA, Atb[..., np.newaxis]).squeeze(-1)  # (M, 3)
    except np.linalg.LinAlgError:
        # Batched solve fails if ANY matrix is singular. Use per-point solve
        # with lstsq fallback only for the degenerate points.
        vel_3d = np.zeros((M, 3), dtype=np.float64)
        for i in range(M):
            try:
                vel_3d[i] = np.linalg.solve(AtA[i], Atb[i])
            except np.linalg.LinAlgError:
                result = np.linalg.lstsq(A[i], b[i], rcond=None)
                vel_3d[i] = result[0]

    return {
        "velocities_3d": vel_3d,
        "valid_indices": valid_idx,
        "num_valid": M,
        "num_total": N,
    }


def _process_stereo_frame(args: Tuple) -> Optional[Dict[str, Any]]:
    """Process a single stereo frame for 3D velocity reconstruction.

    Processes ALL valid runs in a single pass.
    Module-level function for ProcessPoolExecutor compatibility.

    Args:
        args: Tuple containing all parameters needed for processing

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
        image_height,
        z_world,
        tilt_x,
        tilt_y,
    ) = args

    try:
        # Load uncalibrated vectors for both cameras
        mat1 = loadmat(vector_file_path_cam1, struct_as_record=False, squeeze_me=True)
        mat2 = loadmat(vector_file_path_cam2, struct_as_record=False, squeeze_me=True)

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

        total_valid = 0

        # Process ALL valid runs for this frame
        for run_num in sorted(valid_run_nums):
            if run_num not in coords_by_run:
                continue

            x1, y1, x2, y2, x1_world, y1_world = coords_by_run[run_num]

            try:
                run_idx = run_num - 1  # 0-based index
                if run_idx >= len(piv_result_raw1) or run_idx >= len(piv_result_raw2):
                    continue

                cell1 = piv_result_raw1[run_idx]
                cell2 = piv_result_raw2[run_idx]

                ux1_px = getattr(cell1, "ux", None)
                uy1_px = getattr(cell1, "uy", None)
                ux2_px = getattr(cell2, "ux", None)
                uy2_px = getattr(cell2, "uy", None)

                if ux1_px is None or uy1_px is None or ux2_px is None or uy2_px is None:
                    if frame_idx == 1:
                        logger.warning(f"Run {run_num}: velocity data is None")
                    continue
                if not hasattr(ux1_px, 'size') or ux1_px.size == 0:
                    if frame_idx == 1:
                        logger.warning(f"Run {run_num}: velocity data is empty")
                    continue

                ux1_px = np.asarray(ux1_px, dtype=np.float64)
                uy1_px = np.asarray(uy1_px, dtype=np.float64)
                ux2_px = np.asarray(ux2_px, dtype=np.float64)
                uy2_px = np.asarray(uy2_px, dtype=np.float64)

                # Extract and combine b_masks
                b_mask_1 = getattr(cell1, "b_mask", None)
                b_mask_2 = getattr(cell2, "b_mask", None)
                if b_mask_1 is not None and b_mask_2 is not None:
                    combined_input_mask = np.asarray(b_mask_1).astype(bool) | np.asarray(b_mask_2).astype(bool)
                elif b_mask_1 is not None:
                    combined_input_mask = np.asarray(b_mask_1).astype(bool)
                elif b_mask_2 is not None:
                    combined_input_mask = np.asarray(b_mask_2).astype(bool)
                else:
                    combined_input_mask = np.zeros_like(ux1_px, dtype=bool)

                # Shape validation
                ref_shape = x1.shape
                if x1.shape != ux1_px.shape:
                    if frame_idx == 1:
                        logger.warning(f"Run {run_num}: Cam1 shape mismatch - coords {x1.shape} vs velocity {ux1_px.shape}")
                    continue
                if x2.shape != ux2_px.shape:
                    if frame_idx == 1:
                        logger.warning(f"Run {run_num}: Cam2 shape mismatch - coords {x2.shape} vs velocity {ux2_px.shape}")
                    continue

                # Build world 3D points from pre-computed world XY + Z plane equation
                # x1_world, y1_world are in the board's world frame (from _pixels_to_world_mm).
                # The Jacobian uses the same board frame (camera extrinsics are world(board)→camera).
                board_x = x1_world.flatten()
                board_y = y1_world.flatten()
                tan_tx = math.tan(tilt_x) if tilt_x != 0 else 0.0
                tan_ty = math.tan(tilt_y) if tilt_y != 0 else 0.0
                board_z = z_world + board_x * tan_ty + board_y * tan_tx
                world_pts = np.column_stack([board_x, board_y, board_z])

                # Flatten cam1 displacements
                dx1 = ux1_px.flatten()
                dy1 = uy1_px.flatten()

                # Cross-camera correspondence: interpolate cam2 displacements
                dx2_interp, dy2_interp, interp_valid = _interpolate_cam2_displacements(
                    world_pts, ux2_px, uy2_px, x2, y2,
                    stereo_params["camera_matrix_2"], stereo_params["dist_coeffs_2"],
                    stereo_params["cam2_rvec"], stereo_params["cam2_tvec"],
                    image_height,
                )

                # Combine masks: cam1 b_mask + cam2 out-of-bounds
                valid = interp_valid & (~combined_input_mask.flatten())

                # Optional: stereo angle quality filter
                if min_angle > 0:
                    angles = _compute_stereo_angle(
                        world_pts, stereo_params["cam1_rvec"], stereo_params["cam1_tvec"],
                        stereo_params["cam2_rvec"], stereo_params["cam2_tvec"],
                    )
                    valid = valid & (angles > min_angle)

                # Reconstruct 3D velocities via Jacobian decomposition
                result_3d = _reconstruct_3d_velocities(
                    dx1, dy1, dx2_interp, dy2_interp,
                    world_pts, stereo_params, valid,
                )

                # Create output grids
                ux_grid = np.full(ref_shape, np.nan, dtype=np.float64)
                uy_grid = np.full(ref_shape, np.nan, dtype=np.float64)
                uz_grid = np.full(ref_shape, np.nan, dtype=np.float64)
                output_mask = np.full(ref_shape, True, dtype=bool)

                if result_3d["num_valid"] > 0:
                    # vel_3d is in board frame mm/frame → convert to m/s
                    vel_mps = (result_3d["velocities_3d"] / 1000.0) / max(dt, 1e-12)
                    valid_indices = result_3d["valid_indices"]
                    row_indices, col_indices = np.unravel_index(valid_indices, ref_shape)

                    ux_grid[row_indices, col_indices] = vel_mps[:, 0]
                    uy_grid[row_indices, col_indices] = vel_mps[:, 1]
                    # Negate uz: board-Z points from board toward cameras,
                    # physical convention is opposite (away from board surface)
                    uz_grid[row_indices, col_indices] = -vel_mps[:, 2]
                    output_mask[row_indices, col_indices] = False

                    total_valid += result_3d["num_valid"]

                piv_result[run_num - 1] = (ux_grid, uy_grid, uz_grid, output_mask)

            except Exception as run_error:
                logger.warning(f"Run {run_num}: Failed to process frame {frame_idx} - {run_error}")

        savemat(output_file_path, {"piv_result": piv_result})

        return {
            "frame": frame_idx,
            "success": True,
            "num_valid": total_valid,
        }

    except Exception as e:
        return {"frame": frame_idx, "success": False, "error": str(e)}


def _load_stereo_model(
    base_dir: Path, cam1: int, cam2: int, model_type: str
) -> Dict[str, np.ndarray]:
    """Load stereo calibration model and derive per-camera extrinsics.

    Loads the stereo model .mat file and derives individual camera
    extrinsics (rvec, tvec) for both cameras in a shared world frame.
    Cam1 uses its own calibration extrinsics; cam2 is derived from
    the stereo R/T composition (same pattern as camera_model_utils.py).

    Args:
        base_dir: Base directory for calibrated data
        cam1, cam2: Camera numbers
        model_type: 'charuco' or 'dotboard'

    Returns:
        Dict with stereo calibration parameters including per-camera extrinsics
    """
    stereo_file = base_dir / "calibration" / f"stereo_cam{cam1}_cam{cam2}" / "model" / "stereo_model.mat"

    if not stereo_file.exists():
        raise FileNotFoundError(f"Stereo calibration not found at: {stereo_file}")

    logger.info(f"Loading stereo model from: {stereo_file}")
    stereo_data = loadmat(str(stereo_file), squeeze_me=True, struct_as_record=False)

    # Validate required fields
    required_fields = [
        "camera_matrix_1", "camera_matrix_2",
        "dist_coeffs_1", "dist_coeffs_2",
        "rotation_matrix", "translation_vector",
    ]
    missing = [f for f in required_fields if f not in stereo_data]
    if missing:
        raise ValueError(f"Missing required fields in stereo calibration: {missing}")

    # Extract image_height
    img_size = np.asarray(stereo_data.get("image_size", [0, 0])).flatten()
    image_height = int(img_size[1]) if img_size.size >= 2 else 0
    if image_height == 0:
        raise ValueError(
            f"Stereo model {stereo_file} does not contain 'image_size'. "
            f"Re-generate the stereo model."
        )

    # --- Derive per-camera extrinsics (same pattern as camera_model_utils.py) ---

    # Cam1: use datum calibration view's extrinsics
    rvecs_1_raw = np.array(stereo_data.get("rvecs_1", [[0, 0, 0]])).astype(np.float64)
    tvecs_1_raw = np.array(stereo_data.get("tvecs_1", [[0, 0, 0]])).astype(np.float64)

    datum_frame = int(stereo_data["datum_frame"]) if "datum_frame" in stereo_data else 0

    if rvecs_1_raw.ndim == 1:
        rvec1 = rvecs_1_raw.astype(np.float64).flatten()
        tvec1 = tvecs_1_raw.astype(np.float64).flatten()
    else:
        idx = max(0, datum_frame - 1) if datum_frame > 0 else 0
        rvec1 = rvecs_1_raw[idx].flatten()
        tvec1 = tvecs_1_raw[idx].flatten()

    R1, _ = cv2.Rodrigues(rvec1)

    # Cam2: derive from stereo R/T
    # X_cam2 = R_stereo @ X_cam1 + T_stereo
    # Combined: X_cam2 = (R_stereo @ R1) @ X_world + (R_stereo @ t1 + T_stereo)
    K1 = np.array(stereo_data["camera_matrix_1"]).astype(np.float64)
    K2 = np.array(stereo_data["camera_matrix_2"]).astype(np.float64)
    dist1 = np.array(stereo_data["dist_coeffs_1"]).flatten().astype(np.float64)
    dist2 = np.array(stereo_data["dist_coeffs_2"]).flatten().astype(np.float64)
    R_stereo = np.array(stereo_data["rotation_matrix"]).astype(np.float64)
    T_stereo = np.array(stereo_data["translation_vector"]).astype(np.float64).reshape(3, 1)

    R2 = R_stereo @ R1
    t2 = R_stereo @ tvec1.reshape(3, 1) + T_stereo
    rvec2, _ = cv2.Rodrigues(R2)
    rvec2 = rvec2.flatten()
    tvec2 = t2.flatten()

    logger.info(f"Cam1 extrinsics: rvec={rvec1}, tvec={tvec1}")
    logger.info(f"Cam2 extrinsics (derived): rvec={rvec2}, tvec={tvec2}")

    return {
        "camera_matrix_1": K1,
        "camera_matrix_2": K2,
        "dist_coeffs_1": dist1,
        "dist_coeffs_2": dist2,
        "rotation_matrix": R_stereo,
        "translation_vector": T_stereo.flatten(),
        "cam1_rvec": rvec1,
        "cam1_tvec": tvec1,
        "cam2_rvec": rvec2,
        "cam2_tvec": tvec2,
        "image_height": image_height,
    }


class StereoReconstructor:
    """
    Reconstructs 3D velocities from stereo camera pair PIV data.

    Uses the Willert/Soloff geometric reconstruction:
    - Coordinates from single-camera projection to world plane
    - Velocities from Jacobian-based decomposition of paired 2D displacements
    - Cross-camera correspondence via projection + interpolation

    Supports ChArUco, Dotboard, and Stepped Board stereo calibration models.
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
        self._config = config

        if config is not None:
            self.base_dir = Path(base_dir) if base_dir else config.base_paths[0]
            stereo_cfg = config.stereo_dotboard_calibration
            self.camera_pair = camera_pair or stereo_cfg.get("camera_pair", [1, 2])
            self.model_type = model_type or stereo_cfg.get("stereo_model_type", "charuco")
            self.dt = dt if dt is not None else config.dt
            self.vector_pattern = vector_pattern or config.vector_format
            self.num_frame_pairs = config.num_frame_pairs
            # Self-calibration parameters
            self.z_world = config.self_calibration_z_offset
            self.tilt_x = config.self_calibration_tilt_x
            self.tilt_y = config.self_calibration_tilt_y
        else:
            if base_dir is None:
                raise ValueError("base_dir required when config not provided")
            self.base_dir = Path(base_dir)
            self.camera_pair = camera_pair or [1, 2]
            self.model_type = model_type or "charuco"
            self.dt = dt if dt is not None else 1.0
            self.vector_pattern = vector_pattern or "%05d.mat"
            self.num_frame_pairs = None
            self.z_world = 0.0
            self.tilt_x = 0.0
            self.tilt_y = 0.0

        self.type_name = type_name
        self.runs = runs  # 1-indexed
        self.num_workers = num_workers or get_max_workers(999999)
        self.min_angle = min_angle

        valid_types = ("charuco", "dotboard", "stepped_board")
        if self.model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}, got '{self.model_type}'")

        # Load stereo calibration model
        self.stereo_params = _load_stereo_model(
            self.base_dir, self.camera_pair[0], self.camera_pair[1], self.model_type
        )
        self.image_height = self.stereo_params["image_height"]

        logger.info("Initialized StereoReconstructor")
        logger.info(f"  Base directory: {self.base_dir}")
        logger.info(f"  Camera pair: {self.camera_pair}")
        logger.info(f"  Model type: {self.model_type}")
        logger.info(f"  Time step: {self.dt} seconds")
        logger.info(f"  Vector pattern: {self.vector_pattern}")
        logger.info(f"  Type name: {self.type_name}")
        logger.info(f"  Runs to process: {self.runs if self.runs else 'all'}")
        logger.info(f"  Worker count: {self.num_workers}")
        logger.info(f"  Min stereo angle: {self.min_angle} degrees")
        logger.info(f"  Self-cal: z_world={self.z_world}, tilt_x={self.tilt_x}, tilt_y={self.tilt_y}")

    def _find_valid_runs(
        self,
        x1_list: List[np.ndarray],
        y1_list: List[np.ndarray],
        x2_list: List[np.ndarray],
        y2_list: List[np.ndarray],
    ) -> List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Find runs with valid coordinate data in both cameras."""
        valid_runs = []

        for i, (x1, y1, x2, y2) in enumerate(zip(x1_list, y1_list, x2_list, y2_list)):
            if self.runs:
                run_num = self.runs[i]
            else:
                run_num = i + 1

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
        world_coords_by_run: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ):
        """Save world coordinates from single-camera projection.

        Coordinates come from _project_coords_to_world() — no triangulation.
        Already in physical convention (Y-up, mm).

        Args:
            valid_runs: List of valid run tuples
            output_dir: Output directory
            world_coords_by_run: Dict mapping run_num -> (x_world_mm, y_world_mm)
        """
        max_run = max(r[1] for r in valid_runs)
        coord_dtype = np.dtype([("x", "O"), ("y", "O"), ("z", "O")])
        coordinates = np.empty(max_run, dtype=coord_dtype)

        for run_num in range(1, max_run + 1):
            coordinates[run_num - 1] = (np.array([]), np.array([]), np.array([]))

        for _, run_num, x1, y1, _, _ in valid_runs:
            if run_num not in world_coords_by_run:
                continue

            x_world, y_world = world_coords_by_run[run_num]

            # Z from plane equation in board frame
            tan_tx = math.tan(self.tilt_x) if self.tilt_x != 0 else 0.0
            tan_ty = math.tan(self.tilt_y) if self.tilt_y != 0 else 0.0
            z_world_grid = self.z_world + x_world * tan_ty + y_world * tan_tx

            coordinates[run_num - 1] = (x_world, y_world, z_world_grid)
            valid_count = np.sum(np.isfinite(x_world))
            logger.info(f"Run {run_num}: saved {valid_count} world coordinates")

        coords_path = output_dir / "coordinates.mat"
        savemat(str(coords_path), {"coordinates": coordinates})
        logger.info(f"Saved stereo coordinates: {coords_path}")

    def _process_all_frames_parallel(
        self,
        coords_by_run: Dict[int, Tuple],
        uncalib_dir1: Path,
        uncalib_dir2: Path,
        output_dir: Path,
        num_frames: int,
        max_run: int,
        valid_run_nums: Set[int],
        progress_cb: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        """Process all frames for ALL runs in a single pass."""
        logger.info(f"Processing all {len(valid_run_nums)} runs with {self.num_workers} workers")

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
                coords_by_run,
                self.stereo_params,
                self.dt,
                self.min_angle,
                max_run,
                valid_run_nums,
                self.image_height,
                self.z_world,
                self.tilt_x,
                self.tilt_y,
            ))

        if not tasks:
            logger.warning("No vector files found")
            return

        logger.info(f"Processing {len(tasks)} frames for {len(valid_run_nums)} runs")

        successful = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=worker_initializer) as executor:
            futures = {
                executor.submit(_process_stereo_frame, task): task[0]
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
                    except Exception as e:
                        logger.debug(f"Progress callback error: {e}")

        logger.info(f"Processing complete: {successful} successful, {failed} failed")

    def process_run(
        self,
        num_frame_pairs: Optional[int] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Process stereo reconstruction for all frames."""
        if num_frame_pairs is None:
            num_frame_pairs = self.num_frame_pairs
        if num_frame_pairs is None:
            raise ValueError(
                "num_frame_pairs must be provided either to __init__ via config or to process_run()"
            )

        logger.info(f"Processing stereo reconstruction with {num_frame_pairs} frame pairs")

        cam1, cam2 = self.camera_pair

        paths1 = get_data_paths(self.base_dir, num_frame_pairs, cam1, self.type_name, use_uncalibrated=True)
        paths2 = get_data_paths(self.base_dir, num_frame_pairs, cam2, self.type_name, use_uncalibrated=True)

        uncalib_dir1 = paths1["data_dir"]
        uncalib_dir2 = paths2["data_dir"]

        logger.info(f"Uncalibrated data camera {cam1}: {uncalib_dir1}")
        logger.info(f"Uncalibrated data camera {cam2}: {uncalib_dir2}")

        output_paths = get_data_paths(
            self.base_dir, num_frame_pairs, cam=cam1,
            type_name=self.type_name, use_stereo=True, stereo_camera_pair=self.camera_pair,
        )
        output_dir = output_paths["data_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory (stereo): {output_dir}")

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

        if len(x1_list) != len(x2_list):
            min_runs = min(len(x1_list), len(x2_list))
            x1_list, y1_list = x1_list[:min_runs], y1_list[:min_runs]
            x2_list, y2_list = x2_list[:min_runs], y2_list[:min_runs]
            logger.warning(f"Adjusted to {min_runs} runs to match both cameras")

        logger.info(f"Loaded coordinates for {len(x1_list)} runs")

        valid_runs = self._find_valid_runs(x1_list, y1_list, x2_list, y2_list)

        if not valid_runs:
            raise ValueError("No runs with valid coordinate data found")

        logger.info(f"Found {len(valid_runs)} runs with valid data: {[r[1] for r in valid_runs]}")

        max_run = max(r[1] for r in valid_runs)
        valid_run_nums = set(r[1] for r in valid_runs)

        # Pre-compute world coordinates for each run (single-camera projection)
        world_coords_by_run = {}
        coords_by_run = {}
        for _, run_num, x1, y1, x2, y2 in valid_runs:
            x_world, y_world = _project_coords_to_world(
                x1, y1,
                self.stereo_params["camera_matrix_1"],
                self.stereo_params["dist_coeffs_1"],
                self.stereo_params["cam1_rvec"],
                self.stereo_params["cam1_tvec"],
                self.image_height,
                z_world=self.z_world,
                tilt_x=self.tilt_x,
                tilt_y=self.tilt_y,
            )
            world_coords_by_run[run_num] = (x_world, y_world)
            # Pack everything the worker needs: uncal coords for both cameras + world coords
            coords_by_run[run_num] = (x1, y1, x2, y2, x_world, y_world)

            logger.info(
                f"Run {run_num}: world coords x=[{np.nanmin(x_world):.1f}, {np.nanmax(x_world):.1f}], "
                f"y=[{np.nanmin(y_world):.1f}, {np.nanmax(y_world):.1f}] mm"
            )

        # Save coordinates (from single-camera projection, not triangulation)
        self._save_stereo_coordinates(valid_runs, output_dir, world_coords_by_run)

        # Process all frames in parallel
        self._process_all_frames_parallel(
            coords_by_run,
            uncalib_dir1, uncalib_dir2, output_dir,
            num_frame_pairs, max_run, valid_run_nums,
            progress_cb,
        )

        # Save reconstruction summary
        summary_data = {
            "reconstruction_summary": {
                "camera_pair": self.camera_pair,
                "model_type": self.model_type,
                "output_directory": str(output_dir),
                "configuration": {
                    "min_stereo_angle": self.min_angle,
                    "vector_pattern": self.vector_pattern,
                    "type_name": self.type_name,
                    "num_frame_pairs": num_frame_pairs,
                    "dt": self.dt,
                    "num_workers": self.num_workers,
                    "z_world": self.z_world,
                    "tilt_x": self.tilt_x,
                    "tilt_y": self.tilt_y,
                },
                "timestamp": datetime.now().isoformat(),
            },
        }

        stereo_params_clean = {k: v for k, v in self.stereo_params.items() if not k.startswith("_")}
        summary_data["stereo_calibration"] = stereo_params_clean

        summary_file = output_dir / "summary.mat"
        savemat(str(summary_file), summary_data)
        logger.info(f"Saved reconstruction summary: {summary_file}")


def main():
    """Main entry point for stereo reconstruction."""
    logger.info("=" * 60)
    logger.info("Stereo Reconstruction - Starting")
    logger.info("=" * 60)

    if USE_CONFIG_DIRECTLY:
        logger.info("Loading settings directly from config.yaml (USE_CONFIG_DIRECTLY=True)")
        config = get_config()

        stereo_cfg = config.stereo_calibration
        logger.info(f"Base directory: {config.base_paths[0]}")
        logger.info(f"Camera pair: {stereo_cfg.get('camera_pair', [1, 2])}")
        logger.info(f"Num frame pairs: {config.num_frame_pairs}")
        logger.info(f"Time step: {config.dt} seconds")
        logger.info(f"Model type: {stereo_cfg.get('stereo_model_type', 'charuco')}")
        logger.info(f"Vector pattern: {config.vector_format}")
        logger.info(f"Type name: {TYPE_NAME}")
        logger.info(f"Min stereo angle: {MIN_TRIANGULATION_ANGLE} degrees")
        logger.info(f"Runs to process: {RUNS_TO_PROCESS if RUNS_TO_PROCESS else 'all'}")
        logger.info(f"Worker count: {get_max_workers(999999)}")
    else:
        logger.info(f"Base directory: {BASE_DIR}")
        logger.info(f"Camera pair: {CAMERA_PAIR}")
        logger.info(f"Num frame pairs: {NUM_FRAME_PAIRS}")
        logger.info(f"Time step: {DT_SECONDS} seconds")
        logger.info(f"Model type: {MODEL_TYPE}")
        logger.info(f"Vector pattern: {VECTOR_PATTERN}")
        logger.info(f"Type name: {TYPE_NAME}")
        logger.info(f"Min stereo angle: {MIN_TRIANGULATION_ANGLE} degrees")
        logger.info(f"Runs to process: {RUNS_TO_PROCESS if RUNS_TO_PROCESS else 'all'}")
        logger.info(f"Worker count: {get_max_workers(999999)}")

        config = apply_cli_settings_to_config()

    try:
        reconstructor = StereoReconstructor(
            type_name=TYPE_NAME,
            runs=RUNS_TO_PROCESS,
            num_workers=get_max_workers(999999),
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
