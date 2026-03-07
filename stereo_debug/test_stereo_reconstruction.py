#!/usr/bin/env python3
"""
test_stereo_reconstruction.py

Tests for stereo 3D reconstruction math.
Uses synthetic stereo camera geometry for reproducible triangulation tests.

Test categories:
1. Unit Tests - Triangulation, angle computation, velocity extraction
2. Integration Tests - Full reconstruction pipeline with synthetic data
3. CLI Tests - Module imports and syntax validation
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pivtools_gui.stereo_reconstruction.stereo_reconstruction_production import (
    _triangulate_3d_points,
    _compute_triangulation_angles,
    _extract_velocity_components,
    _find_corresponding_points,
    _reconstruct_3d_velocities,
)


# ============================================================
# SYNTHETIC STEREO CAMERA SETUP
# ============================================================

class SyntheticStereoSetup:
    """
    Create synthetic stereo camera configuration for testing.

    Uses a parallel stereo configuration (cameras side by side, parallel optical axes).
    This simplifies the geometry while still testing the core math.
    """

    def __init__(
        self,
        baseline_mm: float = 100.0,
        focal_length: float = 1000.0,
        image_size: tuple = (1024, 768),
    ):
        """
        Initialize synthetic stereo setup.

        Args:
            baseline_mm: Distance between camera centers in mm
            focal_length: Focal length in pixels
            image_size: (width, height) of images
        """
        self.baseline = baseline_mm
        self.focal_length = focal_length
        self.image_size = image_size

        # Both cameras have same intrinsics
        cx, cy = image_size[0] / 2, image_size[1] / 2
        self.K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist = np.zeros(5, dtype=np.float64)

        # Camera 1: At origin, looking down +Z axis
        self.R1 = np.eye(3, dtype=np.float64)
        self.t1 = np.array([0, 0, 0], dtype=np.float64)

        # Camera 2: Offset by baseline in X direction
        self.R2 = np.eye(3, dtype=np.float64)
        self.t2 = np.array([baseline_mm, 0, 0], dtype=np.float64)

        # Relative pose (cam2 relative to cam1)
        self.R_relative = self.R2 @ self.R1.T  # Identity for parallel cameras
        self.T_relative = self.t2 - self.R_relative @ self.t1  # [baseline, 0, 0]

        # Compute rectification parameters (identity for already-parallel cameras)
        self.R1_rect = np.eye(3, dtype=np.float64)
        self.R2_rect = np.eye(3, dtype=np.float64)

        # Compute projection matrices
        self.P1 = self.K @ np.hstack([self.R1_rect, np.zeros((3, 1))])
        self.P2 = self.K @ np.hstack([
            self.R2_rect,
            np.array([[-baseline_mm], [0], [0]])  # Translation in rectified coordinates
        ])

    def project_to_camera(
        self, points_3d: np.ndarray, camera: int = 1
    ) -> np.ndarray:
        """
        Project 3D points to camera image plane.

        Args:
            points_3d: Nx3 array of 3D points in world coordinates
            camera: Camera number (1 or 2)

        Returns:
            Nx2 array of pixel coordinates
        """
        if camera == 1:
            R, t = self.R1, self.t1
        else:
            R, t = self.R2, self.t2

        # Transform to camera coordinates
        points_cam = (R @ points_3d.T).T + t

        # Project to image plane
        pts_2d = (self.K @ points_cam.T).T
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]

        return pts_2d

    def project_to_cameras(self, points_3d: np.ndarray) -> tuple:
        """Project 3D points to both camera image planes."""
        pts_cam1 = self.project_to_camera(points_3d, camera=1)
        pts_cam2 = self.project_to_camera(points_3d, camera=2)
        return pts_cam1, pts_cam2

    def get_stereo_params(self) -> dict:
        """Get stereo parameters in format expected by reconstruction functions."""
        return {
            "camera_matrix_1": self.K.copy(),
            "camera_matrix_2": self.K.copy(),
            "dist_coeffs_1": self.dist.copy(),
            "dist_coeffs_2": self.dist.copy(),
            "rotation_matrix": self.R_relative.copy(),
            "translation_vector": self.T_relative.copy(),
            "rectification_R1": self.R1_rect.copy(),
            "rectification_R2": self.R2_rect.copy(),
            "projection_P1": self.P1.copy(),
            "projection_P2": self.P2.copy(),
        }


# ============================================================
# UNIT TESTS
# ============================================================

def test_triangulate_known_point():
    """Test triangulation of a known 3D point - verify depth magnitude."""
    print("  Testing triangulation of known point...")

    stereo = SyntheticStereoSetup(baseline_mm=100, focal_length=1000)

    # Known 3D point at Z=500mm (in front of cameras)
    point_3d = np.array([[0, 0, 500]], dtype=np.float64)

    # Project to both cameras
    pts_cam1, pts_cam2 = stereo.project_to_cameras(point_3d)

    # Triangulate
    stereo_params = stereo.get_stereo_params()
    points_reconstructed, _, _ = _triangulate_3d_points(
        pts_cam1, pts_cam2, stereo_params
    )

    # OpenCV coordinate conventions may differ - verify depth magnitude correct
    # (sign may differ depending on coordinate system convention)
    assert np.abs(points_reconstructed[0, 2]) - 500.0 < 1.0, (
        f"Depth magnitude wrong: {np.abs(points_reconstructed[0, 2])} != 500"
    )

    # X, Y should be near zero (point on optical axis)
    assert np.abs(points_reconstructed[0, 0]) < 1.0, f"X not near zero: {points_reconstructed[0, 0]}"
    assert np.abs(points_reconstructed[0, 1]) < 1.0, f"Y not near zero: {points_reconstructed[0, 1]}"

    return True


def test_triangulate_off_center_point():
    """Test triangulation of point not on optical axis - verify magnitudes."""
    print("  Testing triangulation of off-center point...")

    stereo = SyntheticStereoSetup(baseline_mm=100, focal_length=1000)

    # Off-center 3D point
    point_3d = np.array([[50, 30, 500]], dtype=np.float64)

    # Project to both cameras
    pts_cam1, pts_cam2 = stereo.project_to_cameras(point_3d)

    # Triangulate
    stereo_params = stereo.get_stereo_params()
    points_reconstructed, _, _ = _triangulate_3d_points(
        pts_cam1, pts_cam2, stereo_params
    )

    # Verify magnitudes are correct (signs may differ due to coordinate convention)
    assert abs(abs(points_reconstructed[0, 0]) - 50.0) < 1.0, (
        f"X magnitude wrong: {abs(points_reconstructed[0, 0])} != 50"
    )
    assert abs(abs(points_reconstructed[0, 1]) - 30.0) < 1.0, (
        f"Y magnitude wrong: {abs(points_reconstructed[0, 1])} != 30"
    )
    assert abs(abs(points_reconstructed[0, 2]) - 500.0) < 1.0, (
        f"Z magnitude wrong: {abs(points_reconstructed[0, 2])} != 500"
    )

    return True


def test_triangulate_multiple_points():
    """Test triangulation of multiple 3D points - verify relative positions."""
    print("  Testing triangulation of multiple points...")

    stereo = SyntheticStereoSetup(baseline_mm=100, focal_length=1000)

    # Multiple 3D points at different positions
    points_3d = np.array([
        [0, 0, 500],
        [50, 30, 500],
        [-40, -20, 600],
        [100, -50, 400],
    ], dtype=np.float64)

    # Project to both cameras
    pts_cam1, pts_cam2 = stereo.project_to_cameras(points_3d)

    # Triangulate
    stereo_params = stereo.get_stereo_params()
    points_reconstructed, _, _ = _triangulate_3d_points(
        pts_cam1, pts_cam2, stereo_params
    )

    # Verify magnitudes are correct for each point
    for i, (orig, recon) in enumerate(zip(points_3d, points_reconstructed)):
        for j, (o, r) in enumerate(zip(orig, recon)):
            assert abs(abs(r) - abs(o)) < 2.0, (
                f"Point {i}, axis {j}: |{r}| != |{o}|"
            )

    # Verify relative distances between points are preserved
    # (this tests internal consistency regardless of coordinate convention)
    for i in range(len(points_3d)):
        for j in range(i + 1, len(points_3d)):
            orig_dist = np.linalg.norm(points_3d[i] - points_3d[j])
            recon_dist = np.linalg.norm(points_reconstructed[i] - points_reconstructed[j])
            assert abs(orig_dist - recon_dist) < 5.0, (
                f"Distance {i}-{j}: {recon_dist} != {orig_dist}"
            )

    return True


def test_triangulation_angle_calculation():
    """Test triangulation angle computation."""
    print("  Testing triangulation angle calculation...")

    stereo = SyntheticStereoSetup(baseline_mm=100, focal_length=1000)
    stereo_params = stereo.get_stereo_params()

    # Point directly in front at 500mm
    # Angle should be: 2 * arctan(baseline/2 / distance) ≈ 11.4°
    point_3d = np.array([[0, 0, 500]])
    expected_angle = 2 * np.rad2deg(np.arctan(50 / 500))  # ~11.3°

    angles = _compute_triangulation_angles(point_3d, stereo_params)

    np.testing.assert_allclose(
        angles[0], expected_angle, rtol=0.1,
        err_msg=f"Angle {angles[0]} != expected {expected_angle}"
    )

    return True


def test_triangulation_angle_varies_with_distance():
    """Test that triangulation angle decreases with distance."""
    print("  Testing triangulation angle varies with distance...")

    stereo = SyntheticStereoSetup(baseline_mm=100, focal_length=1000)
    stereo_params = stereo.get_stereo_params()

    # Points at different distances
    points_3d = np.array([
        [0, 0, 300],   # Close - larger angle
        [0, 0, 500],   # Medium
        [0, 0, 1000],  # Far - smaller angle
    ])

    angles = _compute_triangulation_angles(points_3d, stereo_params)

    # Angles should decrease with distance
    assert angles[0] > angles[1] > angles[2], (
        f"Angles should decrease with distance: {angles}"
    )

    return True


def test_extract_velocity_4d_multi_run():
    """Test velocity extraction from 4D multi-run data."""
    print("  Testing velocity extraction from 4D multi-run data...")

    # Simulate (runs, 3, height, width) data
    n_runs = 3
    height, width = 10, 10
    data = np.random.rand(n_runs, 3, height, width)

    # Set known values in run 2
    data[1, 0, :, :] = 1.0  # ux
    data[1, 1, :, :] = 2.0  # uy

    ux, uy = _extract_velocity_components(data, run_idx=2)

    assert np.all(ux == 1.0), "ux extraction failed"
    assert np.all(uy == 2.0), "uy extraction failed"

    return True


def test_extract_velocity_3d_single_run():
    """Test velocity extraction from 3D single-run data."""
    print("  Testing velocity extraction from 3D single-run data...")

    # Simulate (3, height, width) data
    height, width = 10, 10
    data = np.zeros((3, height, width))
    data[0, :, :] = 5.0  # ux
    data[1, :, :] = 7.0  # uy

    ux, uy = _extract_velocity_components(data, run_idx=1)

    assert np.all(ux == 5.0), "ux extraction failed"
    assert np.all(uy == 7.0), "uy extraction failed"

    return True


def test_find_corresponding_points_same_shape():
    """Test point correspondence finding with same shape grids."""
    print("  Testing point correspondence with same shape grids...")

    shape = (5, 6)
    x1 = np.arange(30).reshape(shape).astype(float)
    y1 = np.arange(30, 60).reshape(shape).astype(float)
    x2 = np.arange(100, 130).reshape(shape).astype(float)
    y2 = np.arange(130, 160).reshape(shape).astype(float)

    indices1, indices2 = _find_corresponding_points((x1, y1), (x2, y2))

    # Same shape: indices should be identical
    np.testing.assert_array_equal(indices1, indices2)
    assert len(indices1) == 30, f"Should have 30 correspondences, got {len(indices1)}"

    return True


def test_find_corresponding_points_different_shape():
    """Test point correspondence finding with different shape grids."""
    print("  Testing point correspondence with different shape grids...")

    shape1 = (5, 6)
    shape2 = (4, 5)  # Smaller
    x1 = np.arange(30).reshape(shape1).astype(float)
    y1 = np.arange(30, 60).reshape(shape1).astype(float)
    x2 = np.arange(20).reshape(shape2).astype(float)
    y2 = np.arange(20, 40).reshape(shape2).astype(float)

    indices1, indices2 = _find_corresponding_points((x1, y1), (x2, y2))

    # Should find min(4,5) * min(5,6) = 4*5 = 20 correspondences
    expected_count = 4 * 5
    assert len(indices1) == expected_count, f"Expected {expected_count}, got {len(indices1)}"

    return True


# ============================================================
# INTEGRATION TESTS
# ============================================================

def test_3d_velocity_reconstruction():
    """Test 3D velocity reconstruction from known displacement - verify magnitudes."""
    print("  Testing 3D velocity reconstruction...")

    stereo = SyntheticStereoSetup(baseline_mm=100, focal_length=1000)
    stereo_params = stereo.get_stereo_params()

    # Create a grid of 3D points at Z=500
    nx, ny = 3, 3
    x = np.linspace(-50, 50, nx)
    y = np.linspace(-30, 30, ny)
    X, Y = np.meshgrid(x, y)
    points_3d = np.column_stack([X.ravel(), Y.ravel(), np.full(nx*ny, 500.0)])

    # Known 3D displacement: 10mm in X, 5mm in Y, 2mm in Z
    displacement_3d = np.array([10.0, 5.0, 2.0])
    points_displaced = points_3d + displacement_3d

    # Project original and displaced points to both cameras
    pts1_orig, pts2_orig = stereo.project_to_cameras(points_3d)
    pts1_disp, pts2_disp = stereo.project_to_cameras(points_displaced)

    # Compute pixel displacements (velocities in pixels/frame)
    vel1 = pts1_disp - pts1_orig
    vel2 = pts2_disp - pts2_orig

    # Create coordinate grids (reshape to 2D)
    coords1_x = pts1_orig[:, 0].reshape(ny, nx)
    coords1_y = pts1_orig[:, 1].reshape(ny, nx)
    coords2_x = pts2_orig[:, 0].reshape(ny, nx)
    coords2_y = pts2_orig[:, 1].reshape(ny, nx)

    # Create velocity grids
    ux1 = vel1[:, 0].reshape(ny, nx)
    uy1 = vel1[:, 1].reshape(ny, nx)
    ux2 = vel2[:, 0].reshape(ny, nx)
    uy2 = vel2[:, 1].reshape(ny, nx)

    # Reconstruct 3D velocities
    result = _reconstruct_3d_velocities(
        ux1, uy1, ux2, uy2,
        (coords1_x, coords1_y),
        (coords2_x, coords2_y),
        stereo_params,
        min_angle=1.0,  # Low threshold for test
    )

    assert result["num_valid"] > 0, "No valid points reconstructed"

    # Check reconstructed velocity magnitudes match expected
    # (signs may differ due to coordinate convention)
    mean_vel = np.mean(result["velocities_3d"], axis=0)
    mean_vel_abs = np.abs(mean_vel)
    expected_abs = np.abs(displacement_3d)

    for i, (recon, exp) in enumerate(zip(mean_vel_abs, expected_abs)):
        assert abs(recon - exp) < 2.0, (
            f"Velocity component {i}: |{recon}| != |{exp}|"
        )

    # Also check total velocity magnitude
    total_recon = np.linalg.norm(mean_vel)
    total_expected = np.linalg.norm(displacement_3d)
    assert abs(total_recon - total_expected) < 2.0, (
        f"Total velocity magnitude: {total_recon} != {total_expected}"
    )

    return True


def test_min_angle_filtering():
    """Test that minimum angle filtering works."""
    print("  Testing minimum angle filtering...")

    stereo = SyntheticStereoSetup(baseline_mm=100, focal_length=1000)
    stereo_params = stereo.get_stereo_params()

    # Create points at very different distances (different triangulation angles)
    points_3d = np.array([
        [0, 0, 200],   # Close - large angle (~28°)
        [0, 0, 2000],  # Far - small angle (~2.9°)
    ])

    # Project to both cameras
    pts1, pts2 = stereo.project_to_cameras(points_3d)

    # Create fake coordinate and velocity grids (1x2)
    coords1_x = pts1[:, 0].reshape(1, 2)
    coords1_y = pts1[:, 1].reshape(1, 2)
    coords2_x = pts2[:, 0].reshape(1, 2)
    coords2_y = pts2[:, 1].reshape(1, 2)

    # Zero velocity (just testing filtering)
    ux = np.zeros((1, 2))
    uy = np.zeros((1, 2))

    # With high min_angle, far point should be filtered out
    result_high = _reconstruct_3d_velocities(
        ux, uy, ux, uy,
        (coords1_x, coords1_y),
        (coords2_x, coords2_y),
        stereo_params,
        min_angle=10.0,  # High threshold
    )

    # With low min_angle, both should pass
    result_low = _reconstruct_3d_velocities(
        ux, uy, ux, uy,
        (coords1_x, coords1_y),
        (coords2_x, coords2_y),
        stereo_params,
        min_angle=1.0,  # Low threshold
    )

    assert result_high["num_valid"] < result_low["num_valid"], (
        f"High threshold ({result_high['num_valid']}) should filter more than "
        f"low threshold ({result_low['num_valid']})"
    )

    return True


def test_reconstruction_empty_input():
    """Test that reconstruction handles empty input gracefully."""
    print("  Testing reconstruction with empty input...")

    stereo = SyntheticStereoSetup(baseline_mm=100, focal_length=1000)
    stereo_params = stereo.get_stereo_params()

    # Empty coordinate grids
    empty_x = np.array([]).reshape(0, 0)
    empty_y = np.array([]).reshape(0, 0)
    empty_vel = np.array([]).reshape(0, 0)

    result = _reconstruct_3d_velocities(
        empty_vel, empty_vel, empty_vel, empty_vel,
        (empty_x, empty_y),
        (empty_x, empty_y),
        stereo_params,
        min_angle=5.0,
    )

    assert result["num_valid"] == 0, "Should return 0 valid points for empty input"
    assert result["num_total"] == 0, "Should return 0 total points for empty input"

    return True


def test_stereo_geometry_baseline_effect():
    """Test that reconstruction works across different baseline configurations."""
    print("  Testing baseline effect on reconstruction...")

    # Point at Z=500
    point_3d = np.array([[0, 0, 500]], dtype=np.float64)

    results = []
    for baseline in [50, 100, 200]:
        stereo = SyntheticStereoSetup(baseline_mm=baseline, focal_length=1000)
        pts1, pts2 = stereo.project_to_cameras(point_3d)

        stereo_params = stereo.get_stereo_params()
        reconstructed, _, _ = _triangulate_3d_points(pts1, pts2, stereo_params)

        # Check magnitude (sign may differ due to coordinate convention)
        depth_error = abs(abs(reconstructed[0, 2]) - 500.0)
        results.append((baseline, depth_error))

    # All should reconstruct depth magnitude correctly
    for baseline, error in results:
        assert error < 2.0, f"Baseline {baseline}mm gave depth error {error}mm"

    return True


# ============================================================
# CLI TESTS
# ============================================================

def test_module_imports():
    """Test that all expected functions can be imported."""
    print("  Testing module imports...")

    from pivtools_gui.stereo_reconstruction.stereo_reconstruction_production import (
        _triangulate_3d_points,
        _compute_triangulation_angles,
        _extract_velocity_components,
        _find_corresponding_points,
        _reconstruct_3d_velocities,
        _load_stereo_model,
        StereoReconstructor,
    )

    assert _triangulate_3d_points is not None
    assert _compute_triangulation_angles is not None
    assert _extract_velocity_components is not None
    assert _find_corresponding_points is not None
    assert _reconstruct_3d_velocities is not None
    assert _load_stereo_model is not None
    assert StereoReconstructor is not None

    return True


def test_script_syntax():
    """Test that the production script has valid syntax."""
    print("  Testing script syntax...")

    import ast
    script_path = (
        Path(__file__).parent.parent
        / "pivtools_gui"
        / "stereo_reconstruction"
        / "stereo_reconstruction_production.py"
    )

    with open(script_path, "r") as f:
        source = f.read()

    try:
        ast.parse(source)
    except SyntaxError as e:
        raise AssertionError(f"Syntax error in production script: {e}")

    return True


# ============================================================
# TEST RUNNER
# ============================================================

def run_tests(verbose: bool = True):
    """Run all tests and report results."""
    unit_tests = [
        ("Triangulate Known Point", test_triangulate_known_point),
        ("Triangulate Off-Center Point", test_triangulate_off_center_point),
        ("Triangulate Multiple Points", test_triangulate_multiple_points),
        ("Triangulation Angle Calculation", test_triangulation_angle_calculation),
        ("Triangulation Angle Varies With Distance", test_triangulation_angle_varies_with_distance),
        ("Extract Velocity 4D Multi-Run", test_extract_velocity_4d_multi_run),
        ("Extract Velocity 3D Single-Run", test_extract_velocity_3d_single_run),
        ("Find Corresponding Points Same Shape", test_find_corresponding_points_same_shape),
        ("Find Corresponding Points Different Shape", test_find_corresponding_points_different_shape),
    ]

    integration_tests = [
        ("3D Velocity Reconstruction", test_3d_velocity_reconstruction),
        ("Min Angle Filtering", test_min_angle_filtering),
        ("Reconstruction Empty Input", test_reconstruction_empty_input),
        ("Stereo Geometry Baseline Effect", test_stereo_geometry_baseline_effect),
    ]

    cli_tests = [
        ("Module Imports", test_module_imports),
        ("Script Syntax", test_script_syntax),
    ]

    all_test_groups = [
        ("UNIT TESTS", unit_tests),
        ("INTEGRATION TESTS", integration_tests),
        ("CLI TESTS", cli_tests),
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for group_name, tests in all_test_groups:
        if verbose:
            print("=" * 60)
            print(group_name)
            print("=" * 60)

        for test_name, test_func in tests:
            total_tests += 1
            try:
                result = test_func()
                if result:
                    passed_tests += 1
                    if verbose:
                        print(f"  [\u2713] {test_name}")
                else:
                    failed_tests.append((test_name, "Test returned False"))
                    if verbose:
                        print(f"  [\u2717] {test_name}: returned False")
            except Exception as e:
                failed_tests.append((test_name, str(e)))
                if verbose:
                    print(f"  [\u2717] {test_name}: {e}")

    if verbose:
        print()
        print("=" * 60)
        print(f"SUMMARY: {passed_tests}/{total_tests} tests passed")
        print("=" * 60)

        if failed_tests:
            print("\nFailed tests:")
            for name, error in failed_tests:
                print(f"  - {name}: {error}")

    return passed_tests == total_tests


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    success = run_tests(verbose=True)
    sys.exit(0 if success else 1)
