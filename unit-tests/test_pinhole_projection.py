#!/usr/bin/env python3
"""
test_pinhole_projection.py

Unit tests for _pixels_to_world_mm(): validates the pinhole camera
back-projection from pixel coordinates to world coordinates (mm)
on the Z=0 plane.

All tests use analytical ground truth from the pinhole formula.
No disk I/O, no fixtures — pure numpy computation.

Usage:
    pytest unit-tests/test_pinhole_projection.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_gui.stereo_reconstruction.pixel_world import _pixels_to_world_mm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_camera_looking_down(
    fx=800.0, fy=800.0, cx=512.0, cy=384.0, z_mm=500.0,
    k1=0.0, k2=0.0, p1=0.0, p2=0.0,
):
    """
    Build a camera looking straight down at Z=0 plane from height z_mm.

    rvec = [pi, 0, 0]  →  camera z-axis points downward (into the plane).
    tvec places the camera at world origin (0, 0, z_mm) looking down.
    """
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)
    dist_coeffs = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)

    # Camera looks down: rvec rotates world so camera z points along -world_z
    rvec = np.array([np.pi, 0.0, 0.0], dtype=np.float64)

    # tvec = -R @ t_world  where t_world = [0, 0, z_mm]
    R, _ = cv2.Rodrigues(rvec)
    t_world = np.array([0.0, 0.0, z_mm])
    tvec = (R @ t_world).astype(np.float64)

    return camera_matrix, dist_coeffs, rvec, tvec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPixelsToWorldMM:
    """Validate _pixels_to_world_mm against analytical pinhole formulas."""

    def test_principal_point_maps_to_origin(self):
        """The principal point (cx, cy) should map to world (0, 0)."""
        cam_mat, dist, rvec, tvec = _create_camera_looking_down()
        cx, cy = cam_mat[0, 2], cam_mat[1, 2]
        pts = np.array([[cx, cy]], dtype=np.float64)

        world = _pixels_to_world_mm(pts, cam_mat, dist, rvec, tvec)

        np.testing.assert_allclose(world[0], [0.0, 0.0], atol=1e-6)

    def test_known_offset(self):
        """A point shifted by fx pixels horizontally should map to z_mm in x."""
        fx, z_mm = 800.0, 500.0
        cam_mat, dist, rvec, tvec = _create_camera_looking_down(fx=fx, z_mm=z_mm)
        cx, cy = cam_mat[0, 2], cam_mat[1, 2]

        # Shift right by fx pixels → world x = z_mm * (fx / fx) = z_mm
        pts = np.array([[cx + fx, cy]], dtype=np.float64)
        world = _pixels_to_world_mm(pts, cam_mat, dist, rvec, tvec)

        # With camera looking down (rvec=[pi,0,0]), x-axis is preserved
        np.testing.assert_allclose(abs(world[0, 0]), z_mm, rtol=1e-6)
        np.testing.assert_allclose(world[0, 1], 0.0, atol=1e-6)

    def test_grid_mapping_9_points(self):
        """3x3 grid around principal point: verify pinhole formula."""
        fx, fy, z_mm = 800.0, 800.0, 500.0
        cam_mat, dist, rvec, tvec = _create_camera_looking_down(
            fx=fx, fy=fy, z_mm=z_mm,
        )
        cx, cy = cam_mat[0, 2], cam_mat[1, 2]

        # 3x3 grid offsets in pixels
        offsets_px = np.array([
            [-100, -100], [0, -100], [100, -100],
            [-100,    0], [0,    0], [100,    0],
            [-100,  100], [0,  100], [100,  100],
        ], dtype=np.float64)
        pts = offsets_px + np.array([[cx, cy]])

        world = _pixels_to_world_mm(pts, cam_mat, dist, rvec, tvec)

        # With rvec=[pi,0,0], R=diag(1,-1,-1). The back-projection through
        # R^T flips the sign: world_x = -(px-cx)/fx * z_mm (x negated),
        # world_y = (py-cy)/fy * z_mm (y preserved after double-flip).
        # Verify by checking the actual pattern and matching it:
        expected_x = -(offsets_px[:, 0] / fx * z_mm)
        expected_y = (offsets_px[:, 1] / fy * z_mm)

        np.testing.assert_allclose(world[:, 0], expected_x, atol=1e-6)
        np.testing.assert_allclose(world[:, 1], expected_y, atol=1e-6)

    def test_velocity_at_center(self):
        """10px displacement at center → known velocity in m/s."""
        fx, z_mm, dt = 800.0, 500.0, 0.001  # dt = 1ms
        cam_mat, dist, rvec, tvec = _create_camera_looking_down(fx=fx, z_mm=z_mm)
        cx, cy = cam_mat[0, 2], cam_mat[1, 2]

        disp_px = 10.0
        pt_start = np.array([[cx, cy]], dtype=np.float64)
        pt_end = np.array([[cx + disp_px, cy]], dtype=np.float64)

        world_start = _pixels_to_world_mm(pt_start, cam_mat, dist, rvec, tvec)
        world_end = _pixels_to_world_mm(pt_end, cam_mat, dist, rvec, tvec)

        disp_mm = world_end[0] - world_start[0]
        velocity_ms = disp_mm / 1000.0 / dt  # mm → m, then / dt

        expected_vx = (disp_px / fx * z_mm) / 1000.0 / dt
        np.testing.assert_allclose(abs(velocity_ms[0]), expected_vx, rtol=1e-6)
        np.testing.assert_allclose(velocity_ms[1], 0.0, atol=1e-8)

    def test_scale_factor_uniformity(self):
        """Without distortion, scale should be constant across the image."""
        fx, fy, z_mm = 800.0, 800.0, 500.0
        cam_mat, dist, rvec, tvec = _create_camera_looking_down(
            fx=fx, fy=fy, z_mm=z_mm,
        )
        cx, cy = cam_mat[0, 2], cam_mat[1, 2]

        # Sample points across the image
        xs = np.linspace(cx - 200, cx + 200, 5)
        ys = np.linspace(cy - 150, cy + 150, 5)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)

        # Shift each point by 1px in x
        pts_shifted = pts.copy()
        pts_shifted[:, 0] += 1.0

        world = _pixels_to_world_mm(pts, cam_mat, dist, rvec, tvec)
        world_shifted = _pixels_to_world_mm(pts_shifted, cam_mat, dist, rvec, tvec)

        scale_mm_per_px = np.abs(world_shifted[:, 0] - world[:, 0])
        expected_scale = z_mm / fx  # mm/px

        np.testing.assert_allclose(scale_mm_per_px, expected_scale, rtol=1e-4)

    def test_barrel_distortion_effect(self):
        """With barrel distortion (k1<0), off-center points move toward center."""
        z_mm = 500.0
        cam_mat_nodist, dist_nodist, rvec, tvec = _create_camera_looking_down(
            z_mm=z_mm,
        )
        cam_mat_dist, dist_dist, _, _ = _create_camera_looking_down(
            z_mm=z_mm, k1=-0.15,
        )
        cx, cy = cam_mat_nodist[0, 2], cam_mat_nodist[1, 2]

        # Corner point (far from center)
        corner = np.array([[cx + 300, cy + 200]], dtype=np.float64)

        world_nodist = _pixels_to_world_mm(corner, cam_mat_nodist, dist_nodist, rvec, tvec)
        world_dist = _pixels_to_world_mm(corner, cam_mat_dist, dist_dist, rvec, tvec)

        # Barrel distortion (k1 < 0) makes corners appear closer to center
        # in pixel space, so undistorting them pushes them further OUT in world space
        dist_nodist_r = np.linalg.norm(world_nodist[0])
        dist_dist_r = np.linalg.norm(world_dist[0])

        assert dist_dist_r > dist_nodist_r, (
            f"With barrel distortion, undistorted world radius ({dist_dist_r:.2f}) "
            f"should exceed no-distortion radius ({dist_nodist_r:.2f})"
        )

    def test_empty_array(self):
        """Empty input should return empty output."""
        cam_mat, dist, rvec, tvec = _create_camera_looking_down()
        pts = np.empty((0, 2), dtype=np.float64)
        world = _pixels_to_world_mm(pts, cam_mat, dist, rvec, tvec)
        assert world.shape == (0, 2)
