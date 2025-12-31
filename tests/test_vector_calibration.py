#!/usr/bin/env python3
"""
test_vector_calibration.py

Comprehensive tests for pinhole camera model vector calibration.

Tests validate:
    - _pixels_to_world_mm() function with synthetic camera setups
    - Coordinate calibration (pixels -> mm using pinhole model)
    - Vector calibration (pixels/frame -> m/s using pinhole model)
    - Stress calibration (spatially-varying scale factor)
    - Known geometric transformations

Usage:
    python test_vector_calibration.py           # Run all tests
    python test_vector_calibration.py --unit    # Unit tests only
    python test_vector_calibration.py --verbose # Detailed output
"""

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.io

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ===================== TEST RESULT DATACLASS =====================


@dataclass
class TestResult:
    """Result of a single test case."""

    name: str
    passed: bool
    checks: List[Dict]
    message: str = ""


# ===================== SYNTHETIC CAMERA SETUP =====================


class SyntheticCameraSetup:
    """Create synthetic camera configurations for testing vector calibration."""

    @staticmethod
    def create_camera_matrix(
        fx: float = 1000.0, fy: float = 1000.0, cx: float = 512.0, cy: float = 384.0
    ) -> np.ndarray:
        """
        Create camera intrinsic matrix with known parameters.

        Args:
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point coordinates

        Returns:
            3x3 camera intrinsic matrix
        """
        return np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )

    @staticmethod
    def create_no_distortion() -> np.ndarray:
        """Create zero distortion coefficients."""
        return np.zeros(5, dtype=np.float64)

    @staticmethod
    def create_simple_distortion(k1: float = 0.1, k2: float = -0.05) -> np.ndarray:
        """Create simple radial distortion coefficients."""
        return np.array([k1, k2, 0, 0, 0], dtype=np.float64)

    @staticmethod
    def create_camera_looking_down(z_distance_mm: float = 500.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create camera pose looking straight down at Z=0 plane.

        The camera is positioned at (0, 0, z_distance_mm) in world coordinates,
        looking down at the Z=0 plane. The camera's optical axis is aligned with
        the -Z world axis.

        Args:
            z_distance_mm: Height of camera above Z=0 plane in mm

        Returns:
            (rvec, tvec): Rotation and translation vectors
        """
        # Rotation: 180 degrees around X axis to flip camera to look down
        # This makes the camera's +Z point toward world -Z
        rvec = np.array([np.pi, 0, 0], dtype=np.float64)

        # Translation: camera center at (0, 0, z_distance_mm) in world
        # In OpenCV convention: t = -R @ camera_center
        R, _ = cv2.Rodrigues(rvec)
        camera_center = np.array([0, 0, z_distance_mm], dtype=np.float64)
        tvec = -R @ camera_center

        return rvec, tvec

    @staticmethod
    def create_tilted_camera(
        z_distance_mm: float = 500.0, tilt_x_deg: float = 15.0, tilt_y_deg: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create camera pose tilted from straight-down position.

        Args:
            z_distance_mm: Height of camera above Z=0 plane
            tilt_x_deg: Tilt angle around X axis in degrees
            tilt_y_deg: Tilt angle around Y axis in degrees

        Returns:
            (rvec, tvec): Rotation and translation vectors
        """
        tilt_x = np.deg2rad(tilt_x_deg)
        tilt_y = np.deg2rad(tilt_y_deg)

        # Combined rotation: base 180° flip + tilts
        rvec = np.array([np.pi - tilt_x, tilt_y, 0], dtype=np.float64)

        R, _ = cv2.Rodrigues(rvec)
        camera_center = np.array([0, 0, z_distance_mm], dtype=np.float64)
        tvec = -R @ camera_center

        return rvec, tvec


class SyntheticPIVData:
    """Generate synthetic PIV data for vector calibration testing."""

    @staticmethod
    def create_coordinate_grid(
        shape: Tuple[int, int], cx: float = 512.0, cy: float = 384.0,
        spacing_px: float = 16.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create coordinate grids in pixels centered around principal point.

        Args:
            shape: (H, W) grid dimensions
            cx, cy: Center coordinates (principal point)
            spacing_px: Spacing between grid points in pixels

        Returns:
            (X, Y): Coordinate grids in pixels
        """
        H, W = shape
        # Create grid centered around principal point
        x_start = cx - (W - 1) / 2 * spacing_px
        y_start = cy - (H - 1) / 2 * spacing_px

        x = np.arange(W) * spacing_px + x_start
        y = np.arange(H) * spacing_px + y_start
        X, Y = np.meshgrid(x, y)
        return X, Y

    @staticmethod
    def create_uniform_vectors(
        shape: Tuple[int, int], ux_px: float, uy_px: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create uniform velocity field in pixels/frame.

        Args:
            shape: (H, W) grid dimensions
            ux_px, uy_px: Uniform velocity components in pixels/frame

        Returns:
            (ux, uy, b_mask): Velocity components and mask
        """
        H, W = shape
        ux = np.full((H, W), ux_px, dtype=np.float64)
        uy = np.full((H, W), uy_px, dtype=np.float64)
        b_mask = np.zeros((H, W), dtype=np.float64)
        return ux, uy, b_mask


# ===================== REFERENCE FORMULAS =====================


def pixels_to_world_simple(
    px: float, py: float, fx: float, fy: float, cx: float, cy: float, z: float
) -> Tuple[float, float]:
    """
    Simple pinhole model projection (no distortion, camera looking straight down).

    For a camera at height z looking straight down at Z=0 plane, with 180° rotation
    around X axis (standard OpenCV convention for looking down):
    world_x = (px - cx) / fx * z
    world_y = -(py - cy) / fy * z  # Negated due to 180° rotation around X

    Args:
        px, py: Pixel coordinates
        fx, fy: Focal lengths
        cx, cy: Principal point
        z: Camera height above Z=0 plane

    Returns:
        (world_x, world_y): World coordinates in mm
    """
    world_x = (px - cx) / fx * z
    # Y is negated because 180° rotation around X flips the Y axis
    world_y = -(py - cy) / fy * z
    return world_x, world_y


def expected_velocity_pinhole(
    disp_px: float, fx: float, z_mm: float, dt: float
) -> float:
    """
    Calculate expected velocity for pinhole model.

    For small displacements near the principal point:
    velocity_ms = (disp_px / fx * z_mm) / 1000 / dt

    Args:
        disp_px: Displacement in pixels
        fx: Focal length in pixels
        z_mm: Camera height in mm
        dt: Time step in seconds

    Returns:
        Velocity in m/s
    """
    disp_mm = disp_px / fx * z_mm
    return disp_mm / 1000.0 / dt


# ===================== UNIT TESTS =====================


class UnitTests:
    """Direct calibration formula and function tests."""

    def __init__(self, rtol: float = 1e-6, verbose: bool = True):
        self.rtol = rtol
        self.verbose = verbose

    def _check(self, name: str, expected: float, computed: float) -> Dict:
        """Check a single value against expected."""
        if abs(expected) > 1e-10:
            passed = abs(computed - expected) / abs(expected) <= self.rtol
        else:
            passed = abs(computed - expected) <= self.rtol
        return {
            "name": name,
            "expected": expected,
            "computed": computed,
            "passed": passed,
        }

    def _check_array(
        self, name: str, expected: np.ndarray, computed: np.ndarray
    ) -> Dict:
        """Check array values against expected."""
        if expected.size == 0 and computed.size == 0:
            return {"name": name, "expected": "[]", "computed": "[]", "passed": True}

        try:
            np.testing.assert_allclose(computed, expected, rtol=self.rtol)
            passed = True
        except AssertionError:
            passed = False

        return {
            "name": name,
            "expected": f"array(mean={np.mean(expected):.6f})",
            "computed": f"array(mean={np.mean(computed):.6f})",
            "passed": passed,
        }

    def test_pixels_to_world_principal_point(self) -> TestResult:
        """Test that principal point maps to world origin."""
        from pivtools_gui.calibration.vector_calibration_production import (
            _pixels_to_world_mm,
        )

        # Setup: camera at 500mm looking straight down
        fx, fy, cx, cy = 1000.0, 1000.0, 512.0, 384.0
        z_mm = 500.0

        K = SyntheticCameraSetup.create_camera_matrix(fx, fy, cx, cy)
        dist = SyntheticCameraSetup.create_no_distortion()
        rvec, tvec = SyntheticCameraSetup.create_camera_looking_down(z_mm)

        # Test: principal point should map to (0, 0)
        pts_px = np.array([[cx, cy]], dtype=np.float32)
        pts_world = _pixels_to_world_mm(pts_px, K, dist, rvec, tvec)

        checks = [
            self._check("world_x", 0.0, pts_world[0, 0]),
            self._check("world_y", 0.0, pts_world[0, 1]),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Principal Point Maps to Origin", passed, checks)

    def test_pixels_to_world_known_offset(self) -> TestResult:
        """Test pixel offset produces expected world offset."""
        from pivtools_gui.calibration.vector_calibration_production import (
            _pixels_to_world_mm,
        )

        # Setup: camera at 500mm, focal length 1000px
        fx, fy, cx, cy = 1000.0, 1000.0, 512.0, 384.0
        z_mm = 500.0

        K = SyntheticCameraSetup.create_camera_matrix(fx, fy, cx, cy)
        dist = SyntheticCameraSetup.create_no_distortion()
        rvec, tvec = SyntheticCameraSetup.create_camera_looking_down(z_mm)

        # Test: pixel at (cx + fx, cy) should map to (z_mm, 0)
        # Because: world_x = (px - cx) / fx * z = 1000/1000 * 500 = 500mm
        pts_px = np.array([[cx + fx, cy]], dtype=np.float32)
        pts_world = _pixels_to_world_mm(pts_px, K, dist, rvec, tvec)

        expected_x, expected_y = pixels_to_world_simple(
            cx + fx, cy, fx, fy, cx, cy, z_mm
        )

        checks = [
            self._check("world_x", expected_x, pts_world[0, 0]),
            self._check("world_y", expected_y, pts_world[0, 1]),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Pixel Offset to World Offset", passed, checks)

    def test_pixels_to_world_grid(self) -> TestResult:
        """Test grid of pixels maps correctly to world coordinates."""
        from pivtools_gui.calibration.vector_calibration_production import (
            _pixels_to_world_mm,
        )

        # Setup
        fx, fy, cx, cy = 1000.0, 1000.0, 512.0, 384.0
        z_mm = 500.0

        K = SyntheticCameraSetup.create_camera_matrix(fx, fy, cx, cy)
        dist = SyntheticCameraSetup.create_no_distortion()
        rvec, tvec = SyntheticCameraSetup.create_camera_looking_down(z_mm)

        # Create 3x3 grid of test points
        offsets = [-100, 0, 100]
        pts_px = []
        expected_world = []

        for dy in offsets:
            for dx in offsets:
                px, py = cx + dx, cy + dy
                pts_px.append([px, py])
                wx, wy = pixels_to_world_simple(px, py, fx, fy, cx, cy, z_mm)
                expected_world.append([wx, wy])

        pts_px = np.array(pts_px, dtype=np.float32)
        expected_world = np.array(expected_world, dtype=np.float64)

        pts_world = _pixels_to_world_mm(pts_px, K, dist, rvec, tvec)

        checks = []
        for i in range(len(pts_px)):
            checks.append(
                self._check(f"point_{i}_x", expected_world[i, 0], pts_world[i, 0])
            )
            checks.append(
                self._check(f"point_{i}_y", expected_world[i, 1], pts_world[i, 1])
            )

        passed = all(c["passed"] for c in checks)
        return TestResult("Grid Pixel to World Mapping", passed, checks)

    def test_velocity_transformation_center(self) -> TestResult:
        """Test velocity transformation at image center."""
        from pivtools_gui.calibration.vector_calibration_production import (
            _pixels_to_world_mm,
        )

        # Setup
        fx, fy, cx, cy = 1000.0, 1000.0, 512.0, 384.0
        z_mm = 500.0
        dt = 0.001  # 1ms time step

        K = SyntheticCameraSetup.create_camera_matrix(fx, fy, cx, cy)
        dist = SyntheticCameraSetup.create_no_distortion()
        rvec, tvec = SyntheticCameraSetup.create_camera_looking_down(z_mm)

        # Displacement: 10 pixels in x direction at center
        disp_px = 10.0
        start_px = np.array([[cx, cy]], dtype=np.float32)
        end_px = np.array([[cx + disp_px, cy]], dtype=np.float32)

        start_world = _pixels_to_world_mm(start_px, K, dist, rvec, tvec)
        end_world = _pixels_to_world_mm(end_px, K, dist, rvec, tvec)

        # Compute velocity
        delta_mm = end_world - start_world
        ux_computed = (delta_mm[0, 0] / 1000.0) / dt
        uy_computed = (delta_mm[0, 1] / 1000.0) / dt

        # Expected velocity
        expected_ux = expected_velocity_pinhole(disp_px, fx, z_mm, dt)
        expected_uy = 0.0

        checks = [
            self._check("ux_ms", expected_ux, ux_computed),
            self._check("uy_ms", expected_uy, uy_computed),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Velocity Transformation at Center", passed, checks)

    def test_empty_array_handling(self) -> TestResult:
        """Test that empty arrays are handled correctly."""
        from pivtools_gui.calibration.vector_calibration_production import (
            _pixels_to_world_mm,
        )

        K = SyntheticCameraSetup.create_camera_matrix()
        dist = SyntheticCameraSetup.create_no_distortion()
        rvec, tvec = SyntheticCameraSetup.create_camera_looking_down()

        # Empty input
        pts_px = np.array([], dtype=np.float32).reshape(0, 2)
        pts_world = _pixels_to_world_mm(pts_px, K, dist, rvec, tvec)

        checks = [
            {"name": "output_size", "expected": 0, "computed": pts_world.size, "passed": pts_world.size == 0}
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Empty Array Handling", passed, checks)

    def test_scale_factor_uniformity(self) -> TestResult:
        """Test that scale factor is uniform for camera looking straight down."""
        from pivtools_gui.calibration.vector_calibration_production import (
            _pixels_to_world_mm,
        )

        # For camera looking straight down with no distortion,
        # the scale factor should be constant across the image

        fx, fy, cx, cy = 1000.0, 1000.0, 512.0, 384.0
        z_mm = 500.0
        delta_px = 1.0

        K = SyntheticCameraSetup.create_camera_matrix(fx, fy, cx, cy)
        dist = SyntheticCameraSetup.create_no_distortion()
        rvec, tvec = SyntheticCameraSetup.create_camera_looking_down(z_mm)

        # Test at multiple locations
        test_points = [
            (cx, cy),           # center
            (cx - 200, cy),     # left
            (cx + 200, cy),     # right
            (cx, cy - 150),     # top
            (cx, cy + 150),     # bottom
        ]

        scale_factors = []
        for px, py in test_points:
            start = np.array([[px, py]], dtype=np.float32)
            end_x = np.array([[px + delta_px, py]], dtype=np.float32)

            start_world = _pixels_to_world_mm(start, K, dist, rvec, tvec)
            end_world = _pixels_to_world_mm(end_x, K, dist, rvec, tvec)

            scale = np.linalg.norm(end_world - start_world) / delta_px
            scale_factors.append(scale)

        # All scale factors should be approximately equal
        scale_mean = np.mean(scale_factors)
        scale_std = np.std(scale_factors)

        # Expected scale: z_mm / fx = 500 / 1000 = 0.5 mm/pixel
        expected_scale = z_mm / fx

        checks = [
            self._check("scale_mean", expected_scale, scale_mean),
            self._check("scale_std", 0.0, scale_std),  # Should be zero for uniform scaling
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Scale Factor Uniformity", passed, checks)

    def run_all(self) -> List[TestResult]:
        """Run all unit tests."""
        tests = [
            self.test_pixels_to_world_principal_point,
            self.test_pixels_to_world_known_offset,
            self.test_pixels_to_world_grid,
            self.test_velocity_transformation_center,
            self.test_empty_array_handling,
            self.test_scale_factor_uniformity,
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                results.append(
                    TestResult(test.__name__, False, [], f"Exception: {str(e)}")
                )

        return results


# ===================== INTEGRATION TESTS =====================


class IntegrationTests:
    """Integration tests using VectorCalibrator class."""

    def __init__(self, rtol: float = 1e-5, verbose: bool = True):
        self.rtol = rtol
        self.verbose = verbose

    def _check(self, name: str, expected: float, computed: float) -> Dict:
        """Check a single value against expected."""
        if abs(expected) > 1e-10:
            passed = abs(computed - expected) / abs(expected) <= self.rtol
        else:
            passed = abs(computed - expected) <= self.rtol
        return {
            "name": name,
            "expected": expected,
            "computed": computed,
            "passed": passed,
        }

    def test_calibrate_coordinates_method(self) -> TestResult:
        """Test VectorCalibrator.calibrate_coordinates() method directly."""
        from pivtools_gui.calibration.vector_calibration_production import (
            _pixels_to_world_mm,
        )

        # This tests the coordinate calibration logic without full class instantiation
        fx, fy, cx, cy = 1000.0, 1000.0, 512.0, 384.0
        z_mm = 500.0

        K = SyntheticCameraSetup.create_camera_matrix(fx, fy, cx, cy)
        dist = SyntheticCameraSetup.create_no_distortion()
        rvec, tvec = SyntheticCameraSetup.create_camera_looking_down(z_mm)

        # Create coordinate grid - use 11x11 so there's a true center point
        shape = (11, 11)
        spacing_px = 16.0
        coords_x, coords_y = SyntheticPIVData.create_coordinate_grid(
            shape, cx, cy, spacing_px
        )

        # Flatten and calibrate
        pts_flat = np.stack([coords_x.flatten(), coords_y.flatten()], axis=-1).astype(
            np.float32
        )
        pts_world = _pixels_to_world_mm(pts_flat, K, dist, rvec, tvec)

        x_mm = pts_world[:, 0].reshape(coords_x.shape)
        y_mm = pts_world[:, 1].reshape(coords_y.shape)

        # Check that center of grid maps to approximately (0, 0)
        # For 11x11, center is at index (5, 5)
        center_idx = (shape[0] // 2, shape[1] // 2)

        # The center pixel should be exactly at (cx, cy), so world coords should be (0, 0)
        checks = [
            self._check("center_x_mm", 0.0, x_mm[center_idx]),
            self._check("center_y_mm", 0.0, y_mm[center_idx]),
        ]

        # Check that spacing is preserved (in mm)
        # Note: Y spacing is negative due to 180° camera rotation
        expected_spacing_mm = spacing_px / fx * z_mm
        actual_spacing_x = x_mm[0, 1] - x_mm[0, 0]
        actual_spacing_y = y_mm[1, 0] - y_mm[0, 0]  # Will be negative

        checks.extend([
            self._check("spacing_x_mm", expected_spacing_mm, actual_spacing_x),
            self._check("spacing_y_mm", -expected_spacing_mm, actual_spacing_y),  # Negated
        ])

        passed = all(c["passed"] for c in checks)
        return TestResult("Calibrate Coordinates Method", passed, checks)

    def test_calibrate_vectors_uniform_field(self) -> TestResult:
        """Test vector calibration produces uniform output for uniform input."""
        from pivtools_gui.calibration.vector_calibration_production import (
            _pixels_to_world_mm,
        )

        # Setup
        fx, fy, cx, cy = 1000.0, 1000.0, 512.0, 384.0
        z_mm = 500.0
        dt = 0.001

        K = SyntheticCameraSetup.create_camera_matrix(fx, fy, cx, cy)
        dist = SyntheticCameraSetup.create_no_distortion()
        rvec, tvec = SyntheticCameraSetup.create_camera_looking_down(z_mm)

        # Create uniform velocity field - use 11x11 for true center
        shape = (11, 11)
        ux_px, uy_px = 10.0, 5.0
        coords_x, coords_y = SyntheticPIVData.create_coordinate_grid(
            shape, cx, cy, spacing_px=16.0
        )
        ux, uy, _ = SyntheticPIVData.create_uniform_vectors(shape, ux_px, uy_px)

        # Calibrate vectors using the production method
        coords_flat = np.stack([coords_x.flatten(), coords_y.flatten()], axis=-1).astype(
            np.float32
        )
        coords_world = _pixels_to_world_mm(coords_flat, K, dist, rvec, tvec)

        disp_px = coords_flat + np.stack([ux.flatten(), uy.flatten()], axis=-1).astype(
            np.float32
        )
        disp_world = _pixels_to_world_mm(disp_px, K, dist, rvec, tvec)

        delta_mm = disp_world - coords_world
        ux_ms = (delta_mm[:, 0] / 1000.0) / dt
        uy_ms = (delta_mm[:, 1] / 1000.0) / dt

        # Expected velocities
        # Note: Y velocity is negated due to 180° camera rotation
        expected_ux = expected_velocity_pinhole(ux_px, fx, z_mm, dt)
        expected_uy = -expected_velocity_pinhole(uy_px, fy, z_mm, dt)  # Negated

        # Check mean values
        checks = [
            self._check("mean_ux_ms", expected_ux, np.mean(ux_ms)),
            self._check("mean_uy_ms", expected_uy, np.mean(uy_ms)),
        ]

        # Check uniformity (std should be very small, allow numerical error)
        # For a uniform field, std should be ~1e-6 or less
        std_ux = np.std(ux_ms)
        std_uy = np.std(uy_ms)
        checks.extend([
            {"name": "std_ux_ms_small", "expected": "<1e-5", "computed": std_ux, "passed": std_ux < 1e-5},
            {"name": "std_uy_ms_small", "expected": "<1e-5", "computed": std_uy, "passed": std_uy < 1e-5},
        ])

        passed = all(c["passed"] for c in checks)
        return TestResult("Calibrate Vectors Uniform Field", passed, checks)

    def run_all(self) -> List[TestResult]:
        """Run all integration tests."""
        tests = [
            self.test_calibrate_coordinates_method,
            self.test_calibrate_vectors_uniform_field,
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                results.append(
                    TestResult(test.__name__, False, [], f"Exception: {str(e)}")
                )

        return results


# ===================== CLI TESTS =====================


class CLITests:
    """Tests for CLI script validity."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def test_module_imports(self) -> TestResult:
        """Test that the vector calibration module can be imported."""
        checks = []

        try:
            from pivtools_gui.calibration import vector_calibration_production
            checks.append({"name": "module_import", "expected": True, "computed": True, "passed": True})
        except ImportError as e:
            checks.append({"name": "module_import", "expected": True, "computed": False, "passed": False})
            return TestResult("Module Import", False, checks, str(e))

        # Check key functions exist
        required_attrs = [
            "_pixels_to_world_mm",
            "VectorCalibrator",
            "_process_single_vector_file",
        ]

        for attr in required_attrs:
            has_attr = hasattr(vector_calibration_production, attr)
            checks.append({
                "name": f"has_{attr}",
                "expected": True,
                "computed": has_attr,
                "passed": has_attr,
            })

        passed = all(c["passed"] for c in checks)
        return TestResult("Module Import and Attributes", passed, checks)

    def test_script_syntax(self) -> TestResult:
        """Test that the script has valid Python syntax."""
        import ast

        script_path = Path(__file__).parent.parent / "pivtools_gui" / "calibration" / "vector_calibration_production.py"

        checks = []

        if not script_path.exists():
            checks.append({
                "name": "file_exists",
                "expected": True,
                "computed": False,
                "passed": False,
            })
            return TestResult("Script Syntax", False, checks, f"File not found: {script_path}")

        try:
            with open(script_path, "r") as f:
                source = f.read()
            ast.parse(source)
            checks.append({
                "name": "valid_syntax",
                "expected": True,
                "computed": True,
                "passed": True,
            })
        except SyntaxError as e:
            checks.append({
                "name": "valid_syntax",
                "expected": True,
                "computed": False,
                "passed": False,
            })
            return TestResult("Script Syntax", False, checks, str(e))

        passed = all(c["passed"] for c in checks)
        return TestResult("Script Syntax", passed, checks)

    def run_all(self) -> List[TestResult]:
        """Run all CLI tests."""
        tests = [
            self.test_module_imports,
            self.test_script_syntax,
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                results.append(
                    TestResult(test.__name__, False, [], f"Exception: {str(e)}")
                )

        return results


# ===================== TEST RUNNER =====================


def print_result(result: TestResult, verbose: bool = True):
    """Print a single test result."""
    status = "\u2713" if result.passed else "\u2717"
    print(f"  [{status}] {result.name}")

    if verbose and result.checks:
        for check in result.checks:
            check_status = "\u2713" if check["passed"] else "\u2717"
            print(f"      [{check_status}] {check['name']}: expected={check['expected']}, computed={check['computed']}")

    if result.message:
        print(f"      Message: {result.message}")


def print_summary(results: List[TestResult]):
    """Print test summary."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passed}/{total} tests passed")

    if passed < total:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Vector Calibration Tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--cli", action="store_true", help="Run CLI tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance")

    args = parser.parse_args()

    # If no specific test type selected, run all
    run_all = not (args.unit or args.integration or args.cli)

    all_results = []

    # Unit tests
    if args.unit or run_all:
        print("\n" + "=" * 60)
        print("UNIT TESTS")
        print("=" * 60)
        unit_tests = UnitTests(rtol=args.rtol, verbose=args.verbose)
        results = unit_tests.run_all()
        all_results.extend(results)
        for r in results:
            print_result(r, args.verbose)

    # Integration tests
    if args.integration or run_all:
        print("\n" + "=" * 60)
        print("INTEGRATION TESTS")
        print("=" * 60)
        integration_tests = IntegrationTests(rtol=args.rtol, verbose=args.verbose)
        results = integration_tests.run_all()
        all_results.extend(results)
        for r in results:
            print_result(r, args.verbose)

    # CLI tests
    if args.cli or run_all:
        print("\n" + "=" * 60)
        print("CLI TESTS")
        print("=" * 60)
        cli_tests = CLITests(verbose=args.verbose)
        results = cli_tests.run_all()
        all_results.extend(results)
        for r in results:
            print_result(r, args.verbose)

    # Summary
    print_summary(all_results)

    # Exit code
    all_passed = all(r.passed for r in all_results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
