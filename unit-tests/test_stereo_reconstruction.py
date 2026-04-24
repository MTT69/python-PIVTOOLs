"""
Tests for stereo PIV reconstruction (Willert/Soloff Jacobian method).

Validates the three stages of reconstruction independently:
  1. Coordinate projection: uncalibrated PIV grid → world coordinates
  2. Cross-camera correspondence: cam1 world grid → cam2 interpolated displacements
  3. Velocity reconstruction: 2D pixel displacements → 3D world velocities

Uses synthetic cameras with known geometry (zero distortion) and analytically
computed displacements. Proves the reconstruction math is correct for two
stereo configurations: same-side (30°) and wide-angle (90°).

Known limitations (see plan for future work):
  - Zero distortion (isolates geometry from distortion handling)
  - Bypasses PIV (displacements computed analytically, not from correlation)
  - Does not test _process_stereo_frame orchestration or .mat I/O
"""

import math
from typing import Tuple

import cv2
import numpy as np
import pytest

from pivtools_gui.stereo_reconstruction.self_calibration import PinholeCamera
from pivtools_gui.stereo_reconstruction.stereo_reconstruction_production import (
    _compute_projection_jacobian,
    _interpolate_cam2_displacements,
    _project_coords_to_world,
    _reconstruct_3d_velocities,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_W, IMAGE_H = 2048, 2048
FOCAL_LENGTH_PX = 5000.0
CX, CY = IMAGE_W / 2.0, IMAGE_H / 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _look_at_rotation(cam_pos: np.ndarray) -> np.ndarray:
    """Rotation matrix for a camera at cam_pos looking at the origin.

    Camera Z-axis points from cam_pos toward origin.
    Camera Y-axis is approximately aligned with world -Y (image y-down).
    """
    forward = -cam_pos / np.linalg.norm(cam_pos)  # toward origin
    world_up = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    # R maps world → camera: rows are camera axes in world coords
    R = np.stack([right, down, forward], axis=0)
    return R


def _create_stereo_pair(
    full_angle_deg: float,
    distance_mm: float = 500.0,
) -> Tuple[PinholeCamera, PinholeCamera, dict]:
    """Create a symmetric stereo pair at the given full angle.

    Both cameras are at equal angles from the Z-axis, in the XZ plane,
    at the given distance from the origin, looking at (0, 0, 0).

    Returns (cam1, cam2, stereo_params_dict).
    """
    K = np.array([
        [FOCAL_LENGTH_PX, 0, CX],
        [0, FOCAL_LENGTH_PX, CY],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    half_angle = math.radians(full_angle_deg / 2.0)

    # Camera positions in world coords
    cam1_pos = np.array([
        distance_mm * math.sin(half_angle),
        0,
        -distance_mm * math.cos(half_angle),
    ])
    cam2_pos = np.array([
        -distance_mm * math.sin(half_angle),
        0,
        -distance_mm * math.cos(half_angle),
    ])

    R1 = _look_at_rotation(cam1_pos)
    R2 = _look_at_rotation(cam2_pos)
    t1 = (-R1 @ cam1_pos).reshape(3, 1)
    t2 = (-R2 @ cam2_pos).reshape(3, 1)

    rvec1, _ = cv2.Rodrigues(R1)
    rvec2, _ = cv2.Rodrigues(R2)

    cam1 = PinholeCamera(K=K, dist=dist, R=R1, t=t1, image_size=(IMAGE_W, IMAGE_H))
    cam2 = PinholeCamera(K=K, dist=dist, R=R2, t=t2, image_size=(IMAGE_W, IMAGE_H))

    stereo_params = {
        "camera_matrix_1": K.copy(),
        "camera_matrix_2": K.copy(),
        "dist_coeffs_1": dist.copy(),
        "dist_coeffs_2": dist.copy(),
        "cam1_rvec": rvec1.flatten(),
        "cam1_tvec": t1.flatten(),
        "cam2_rvec": rvec2.flatten(),
        "cam2_tvec": t2.flatten(),
        "image_height": IMAGE_H,
    }

    return cam1, cam2, stereo_params


def _world_to_uncal(
    world_pts: np.ndarray,
    cam: PinholeCamera,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project world points to uncalibrated pixel convention (1-based, y-up).

    Returns (x_uncal, y_uncal) each shape matching input.
    """
    raw_px = cam.project(world_pts.reshape(-1, 3))  # (N, 2) raw pixels
    x_uncal = raw_px[:, 0] + 1.0          # 0-based → 1-based
    y_uncal = IMAGE_H - raw_px[:, 1]      # y-down → y-up
    return x_uncal, y_uncal


def _make_world_grid(
    x_range: Tuple[float, float] = (-40.0, 40.0),
    y_range: Tuple[float, float] = (-30.0, 30.0),
    nx: int = 20,
    ny: int = 15,
    z: float = 0.0,
) -> np.ndarray:
    """Create a regular grid of world points. Returns (ny, nx, 3)."""
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    Xg, Yg = np.meshgrid(xs, ys)
    Zg = np.full_like(Xg, z)
    return np.stack([Xg, Yg, Zg], axis=-1)  # (ny, nx, 3)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=["same_side_30", "wide_angle_90"])
def stereo_setup(request):
    """Create synthetic stereo cameras + stereo_params + world grid.

    Parametrized over two geometries: 30° (typical PIV) and 90° (wide angle).
    """
    if request.param == "same_side_30":
        angle = 30.0
    else:
        angle = 90.0

    cam1, cam2, stereo_params = _create_stereo_pair(angle)
    world_grid = _make_world_grid()  # (ny, nx, 3)

    return {
        "cam1": cam1,
        "cam2": cam2,
        "stereo_params": stereo_params,
        "world_grid": world_grid,
        "angle_deg": angle,
        "label": request.param,
    }


# ===========================================================================
# Test 1: Coordinate Projection Round-Trip
# ===========================================================================

class TestCoordinateProjection:
    """Verify _project_coords_to_world recovers known world positions."""

    def test_roundtrip_z0(self, stereo_setup):
        """Project world → cam1 uncal → _project_coords_to_world → world. Z=0."""
        cam1 = stereo_setup["cam1"]
        sp = stereo_setup["stereo_params"]
        grid = stereo_setup["world_grid"]  # (ny, nx, 3)
        ny, nx = grid.shape[:2]

        # World → uncalibrated pixels
        x_uncal, y_uncal = _world_to_uncal(grid.reshape(-1, 3), cam1)
        X_uncal = x_uncal.reshape(ny, nx)
        Y_uncal = y_uncal.reshape(ny, nx)

        # Uncalibrated → world via production function
        x_world, y_world = _project_coords_to_world(
            X_uncal, Y_uncal,
            sp["camera_matrix_1"], sp["dist_coeffs_1"],
            sp["cam1_rvec"], sp["cam1_tvec"],
            IMAGE_H,
        )

        # Compare against known world positions
        err_x = np.abs(x_world - grid[:, :, 0])
        err_y = np.abs(y_world - grid[:, :, 1])

        assert err_x.max() < 0.001, f"X error {err_x.max():.6f} mm"
        assert err_y.max() < 0.001, f"Y error {err_y.max():.6f} mm"

    def test_roundtrip_with_z_offset(self, stereo_setup):
        """Same round-trip but with z_world=2.0 and tilt_x=0.01 rad."""
        cam1 = stereo_setup["cam1"]
        sp = stereo_setup["stereo_params"]
        z_world = 2.0
        tilt_x = 0.01

        # Create grid on the tilted plane: Z = z_world + Y * tan(tilt_x)
        grid = _make_world_grid(z=0.0)
        ny, nx = grid.shape[:2]
        grid[:, :, 2] = z_world + grid[:, :, 1] * math.tan(tilt_x)

        # World → uncalibrated
        x_uncal, y_uncal = _world_to_uncal(grid.reshape(-1, 3), cam1)
        X_uncal = x_uncal.reshape(ny, nx)
        Y_uncal = y_uncal.reshape(ny, nx)

        # Uncalibrated → world with plane params
        x_world, y_world = _project_coords_to_world(
            X_uncal, Y_uncal,
            sp["camera_matrix_1"], sp["dist_coeffs_1"],
            sp["cam1_rvec"], sp["cam1_tvec"],
            IMAGE_H,
            z_world=z_world, tilt_x=tilt_x,
        )

        err_x = np.abs(x_world - grid[:, :, 0])
        err_y = np.abs(y_world - grid[:, :, 1])

        assert err_x.max() < 0.01, f"X error {err_x.max():.6f} mm (with z_offset+tilt)"
        assert err_y.max() < 0.01, f"Y error {err_y.max():.6f} mm (with z_offset+tilt)"


# ===========================================================================
# Test 2: Cross-Camera Correspondence
# ===========================================================================

class TestCrossCameraCorrespondence:
    """Verify _interpolate_cam2_displacements maps correctly between cameras."""

    def test_uniform_displacement_field(self, stereo_setup):
        """Uniform cam2 displacement field should interpolate to the same value."""
        cam1 = stereo_setup["cam1"]
        cam2 = stereo_setup["cam2"]
        sp = stereo_setup["stereo_params"]
        grid = stereo_setup["world_grid"]
        ny, nx = grid.shape[:2]

        # Build cam1 world points (what the reconstruction uses)
        world_pts_flat = grid.reshape(-1, 3)

        # Build cam2's PIV grid in uncalibrated coords
        # Use a regular pixel grid for cam2
        win_spacing = 32
        x2_raw = np.arange(win_spacing / 2, IMAGE_W, win_spacing)
        y2_raw = np.arange(win_spacing / 2, IMAGE_H, win_spacing)
        X2_raw, Y2_raw = np.meshgrid(x2_raw, y2_raw)
        X2_uncal = X2_raw + 1.0
        Y2_uncal = IMAGE_H - Y2_raw

        # Uniform displacement field for cam2
        known_dx, known_dy = 2.5, -1.3
        ux2 = np.full(X2_uncal.shape, known_dx, dtype=np.float64)
        uy2 = np.full(X2_uncal.shape, known_dy, dtype=np.float64)

        # Interpolate
        dx2_interp, dy2_interp, valid = _interpolate_cam2_displacements(
            world_pts_flat,
            ux2, uy2,
            X2_uncal, Y2_uncal,
            sp["camera_matrix_2"], sp["dist_coeffs_2"],
            sp["cam2_rvec"], sp["cam2_tvec"],
            IMAGE_H,
        )

        # All valid points should recover the uniform field
        assert valid.sum() > 0, "No valid interpolated points"
        err_dx = np.abs(dx2_interp[valid] - known_dx)
        err_dy = np.abs(dy2_interp[valid] - known_dy)

        assert err_dx.max() < 0.01, (
            f"dx interpolation error {err_dx.max():.4f} px "
            f"(expected {known_dx}, {valid.sum()} valid points)"
        )
        assert err_dy.max() < 0.01, (
            f"dy interpolation error {err_dy.max():.4f} px "
            f"(expected {known_dy}, {valid.sum()} valid points)"
        )


# ===========================================================================
# Test 3: 3D Velocity Reconstruction — Component Isolation
# ===========================================================================

class TestVelocityReconstruction:
    """Verify _reconstruct_3d_velocities correctly decomposes 2D → 3D."""

    @pytest.mark.parametrize("velocity,label", [
        ([1.0, 0.0, 0.0], "Ux_only"),
        ([0.0, 1.0, 0.0], "Uy_only"),
        ([0.0, 0.0, 1.0], "Uz_only"),
        ([0.5, -0.3, 0.1], "combined"),
    ])
    def test_velocity_recovery(self, stereo_setup, velocity, label):
        """Known 3D velocity → project to 2D per camera → reconstruct → compare."""
        cam1 = stereo_setup["cam1"]
        cam2 = stereo_setup["cam2"]
        sp = stereo_setup["stereo_params"]
        grid = stereo_setup["world_grid"]
        angle = stereo_setup["angle_deg"]

        vel_3d = np.array(velocity, dtype=np.float64)
        world_pts = grid.reshape(-1, 3)  # (N, 3)
        N = world_pts.shape[0]

        # Compute what each camera would measure as pixel displacement
        pts_A = world_pts
        pts_B = world_pts + vel_3d[np.newaxis, :]  # displaced positions

        px1_A = cam1.project(pts_A)  # (N, 2) raw pixels
        px1_B = cam1.project(pts_B)
        dx1 = px1_B[:, 0] - px1_A[:, 0]
        dy1 = px1_B[:, 1] - px1_A[:, 1]

        px2_A = cam2.project(pts_A)
        px2_B = cam2.project(pts_B)
        dx2 = px2_B[:, 0] - px2_A[:, 0]
        dy2 = px2_B[:, 1] - px2_A[:, 1]

        valid_mask = np.ones(N, dtype=bool)

        # Reconstruct
        result = _reconstruct_3d_velocities(
            dx1, dy1, dx2, dy2,
            world_pts, sp, valid_mask,
        )

        assert result["num_valid"] == N, (
            f"Expected {N} valid, got {result['num_valid']}"
        )

        recovered = result["velocities_3d"]  # (N, 3) mm/frame
        mean_recovered = recovered.mean(axis=0)

        # Primary component accuracy
        for i, comp in enumerate(["Ux", "Uy", "Uz"]):
            expected = vel_3d[i]
            got = mean_recovered[i]
            if abs(expected) > 0.01:
                rel_err = abs(got - expected) / abs(expected)
                assert rel_err < 0.005, (
                    f"[{label} @ {angle}°] {comp}: expected {expected:.3f}, "
                    f"got {got:.3f} (rel error {rel_err:.4f})"
                )
            else:
                # Cross-talk: should be near zero
                assert abs(got) < 0.01, (
                    f"[{label} @ {angle}°] {comp} cross-talk: "
                    f"expected ~0, got {got:.4f}"
                )

        # Spatial uniformity: std should be very small
        std_recovered = recovered.std(axis=0)
        for i, comp in enumerate(["Ux", "Uy", "Uz"]):
            assert std_recovered[i] < 0.01, (
                f"[{label} @ {angle}°] {comp} spatial variation: "
                f"std={std_recovered[i]:.4f} (should be <0.01 for uniform field)"
            )
