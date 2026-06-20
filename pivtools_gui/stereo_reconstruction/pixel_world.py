"""stereo_reconstruction.pixel_world — pixel→world-mm ray-plane projection.

``_pixels_to_world_mm`` is the full plane-aware projection (supports ``z_world`` /
``tilt_x`` / ``tilt_y``) for stereo back-projection. It was relocated
here verbatim from the legacy ``calibration/vector_calibration_production.py`` during the
v1 retirement so stereo_reconstruction no longer depends on the v1 ``calibration`` package.
The math is unchanged (no silent algorithm change) — note this is the FULL version, not
the simplified Z=0-only copy that lived in ``global_coordinate_alignment``.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


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

    # Ray-plane intersection for plane: Z = z_world + X*tan(tilt_y) + Y*tan(tilt_x)
    # Camera ray: [x_norm, y_norm, 1] (normalized coords with z=1)
    # World point: P_world = R^T @ (s * ray - t) = s * ray_world - t_world
    # Plane equation: P_world[2] = z_world + P_world[0]*tan_ty + P_world[1]*tan_tx
    # Substituting and solving for s:
    #   s*(rw[2] - rw[0]*tan_ty - rw[1]*tan_tx) = z_world + tw[2] - tw[0]*tan_ty - tw[1]*tan_tx

    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_world = R_inv @ tvec.flatten()  # (3,) — constant across all points

    tan_tx = math.tan(tilt_x)
    tan_ty = math.tan(tilt_y)

    # Build rays [xn, yn, 1] in a single (N, 3) array
    N = pts_normalized.shape[0]
    rays = np.empty((N, 3), dtype=np.float64)
    rays[:, :2] = pts_normalized
    rays[:, 2] = 1.0

    rays_world = rays @ R_inv.T  # (N, 3)

    denom = rays_world[:, 2] - rays_world[:, 0] * tan_ty - rays_world[:, 1] * tan_tx
    numer = z_world + t_world[2] - t_world[0] * tan_ty - t_world[1] * tan_tx

    s = np.full(N, np.nan, dtype=np.float64)
    valid = np.abs(denom) >= 1e-10
    s[valid] = numer / denom[valid]

    world_3d = s[:, None] * rays_world - t_world  # (N, 3); NaN propagates for invalid
    return world_3d[:, :2]
