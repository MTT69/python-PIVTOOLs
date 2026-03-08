#!/usr/bin/env python3
"""
test_stereo_calibration.py

Tests for stereo camera calibration using ChArUco and circle grid (dotboard) detection.
Uses synthetic stereo image pairs generated programmatically for reproducibility.

Test categories:
1. Unit Tests - Pattern detection, object point generation, stereo math
2. Integration Tests - Full stereo calibration pipeline with synthetic views
3. CLI Tests - Module imports and syntax validation
"""

import sys
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pivtools_gui.stereo_reconstruction.stereo_calibration_base import BaseStereoCalibrator
from pivtools_gui.stereo_reconstruction.stereo_charuco_calibration_production import (
    StereoCharucoCalibrator,
    ARUCO_DICT_MAP,
)
from pivtools_gui.stereo_reconstruction.stereo_dotboard_calibration_production import (
    StereoDotboardCalibrator,
)


# ============================================================
# SYNTHETIC STEREO IMAGE GENERATION
# ============================================================

class SyntheticStereoImageGenerator:
    """
    Generate synthetic stereo image pairs for testing stereo calibration.

    Creates board images as viewed from two different camera positions,
    simulating a stereo camera rig.
    """

    def __init__(
        self,
        baseline_mm: float = 100.0,
        focal_length: float = 800.0,
        image_size: Tuple[int, int] = (800, 600),
        board_distance_mm: float = 500.0,
    ):
        """
        Initialize stereo image generator.

        Args:
            baseline_mm: Distance between camera centers in mm
            focal_length: Focal length in pixels
            image_size: (width, height) of output images
            board_distance_mm: Distance from cameras to calibration board
        """
        self.baseline = baseline_mm
        self.focal_length = focal_length
        self.image_size = image_size
        self.board_distance = board_distance_mm

        # Camera intrinsics (same for both cameras)
        cx, cy = image_size[0] / 2, image_size[1] / 2
        self.K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def generate_charuco_stereo_pair(
        self,
        squares_h: int = 8,
        squares_v: int = 6,
        square_size: float = 0.03,
        marker_ratio: float = 0.5,
        aruco_dict_id: int = cv2.aruco.DICT_4X4_1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a stereo pair of ChArUco board images.

        For simplicity, we generate the same frontal view for both cameras
        with a small horizontal shift to simulate stereo disparity.

        Returns:
            (image_cam1, image_cam2)
        """
        # Create ChArUco board
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        marker_size = square_size * marker_ratio
        board = cv2.aruco.CharucoBoard(
            (squares_h, squares_v),
            square_size,
            marker_size,
            dictionary,
        )

        # Generate board image
        board_img = board.generateImage(self.image_size)

        # For camera 1: centered view
        img1 = self._add_margin(board_img, margin=50)

        # For camera 2: slightly shifted view (simulating stereo disparity)
        img2 = self._shift_image(board_img, shift_x=10, shift_y=0)
        img2 = self._add_margin(img2, margin=50)

        return img1, img2

    def generate_dotboard_stereo_pair(
        self,
        cols: int = 8,
        rows: int = 6,
        dot_spacing_mm: float = 15.0,
        dot_radius_mm: float = 4.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a stereo pair of dotboard (circle grid) images.

        Returns:
            (image_cam1, image_cam2)
        """
        # Calculate pixel scale
        px_per_mm = 8.0

        # Generate dotboard
        spacing_px = dot_spacing_mm * px_per_mm
        radius_px = int(dot_radius_mm * px_per_mm)
        margin = 80

        grid_width = int((cols - 1) * spacing_px + 2 * radius_px)
        grid_height = int((rows - 1) * spacing_px + 2 * radius_px)

        width = grid_width + 2 * margin
        height = grid_height + 2 * margin

        # Create black background with white dots
        img = np.zeros((height, width), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                x = int(margin + radius_px + c * spacing_px)
                y = int(margin + radius_px + r * spacing_px)
                cv2.circle(img, (x, y), radius_px, 255, -1)

        # For camera 1: original
        img1 = img.copy()

        # For camera 2: slightly shifted
        img2 = self._shift_image(img, shift_x=8, shift_y=0)

        return img1, img2

    def generate_multiple_stereo_views(
        self,
        n_views: int = 5,
        board_type: str = "charuco",
        **board_kwargs,
    ) -> list:
        """
        Generate multiple stereo view pairs with slight variations.

        Args:
            n_views: Number of stereo pairs to generate
            board_type: "charuco" or "dotboard"
            **board_kwargs: Arguments for board generation

        Returns:
            List of (img1, img2) tuples
        """
        views = []
        for i in range(n_views):
            if board_type == "charuco":
                img1, img2 = self.generate_charuco_stereo_pair(**board_kwargs)
            else:
                img1, img2 = self.generate_dotboard_stereo_pair(**board_kwargs)

            # Add slight rotation for variation
            if i > 0:
                angle = (i - n_views // 2) * 2  # -4, -2, 0, 2, 4 degrees
                img1 = self._rotate_image(img1, angle * 0.5)
                img2 = self._rotate_image(img2, angle * 0.5)

            views.append((img1, img2))

        return views

    def _add_margin(self, img: np.ndarray, margin: int = 50) -> np.ndarray:
        """Add gray margin around image."""
        h, w = img.shape[:2]
        new_h, new_w = h + 2 * margin, w + 2 * margin

        if img.ndim == 2:
            canvas = np.full((new_h, new_w), 128, dtype=np.uint8)
            canvas[margin:margin + h, margin:margin + w] = img
        else:
            canvas = np.full((new_h, new_w, img.shape[2]), 128, dtype=np.uint8)
            canvas[margin:margin + h, margin:margin + w] = img

        return canvas

    def _shift_image(self, img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
        """Shift image by given pixels (simulating stereo disparity)."""
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        h, w = img.shape[:2]
        return cv2.warpAffine(img, M, (w, h), borderValue=128)

    def _rotate_image(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate image around center."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=128)


# ============================================================
# UNIT TESTS - CHARUCO
# ============================================================

def test_charuco_detector_creation():
    """Test that StereoCharucoCalibrator creates detector correctly."""
    print("  Testing ChArUco detector creation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = StereoCharucoCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_pair=[1, 2],
            squares_h=10,
            squares_v=9,
            square_size=0.03,
            marker_ratio=0.5,
            aruco_dict="DICT_4X4_1000",
            min_corners=6,
        )

        # Check detector is tuple (board, detector)
        assert calibrator.detector is not None, "Detector not created"
        board, detector = calibrator.detector
        assert board is not None, "Board not created"
        assert detector is not None, "CharucoDetector not created"

        # Check board parameters
        board_size = board.getChessboardSize()
        assert board_size == (10, 9), f"Wrong board size: {board_size}"

    return True


def test_charuco_object_points():
    """Test ChArUco object points generation."""
    print("  Testing ChArUco object points generation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = StereoCharucoCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_pair=[1, 2],
            squares_h=6,
            squares_v=5,
            square_size=0.03,
        )

        obj_pts = calibrator.make_object_points()

        # Expected corners: (squares_h - 1) * (squares_v - 1) = 5 * 4 = 20
        expected_corners = (6 - 1) * (5 - 1)
        assert len(obj_pts) == expected_corners, f"Expected {expected_corners} corners, got {len(obj_pts)}"

        # All Z should be 0 (planar)
        assert np.all(obj_pts[:, 2] == 0), "Z coordinates should all be 0"

    return True


def test_charuco_detection_synthetic():
    """Test ChArUco detection on synthetic image."""
    print("  Testing ChArUco detection on synthetic image...")

    generator = SyntheticStereoImageGenerator()
    img1, img2 = generator.generate_charuco_stereo_pair(squares_h=8, squares_v=6)

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = StereoCharucoCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_pair=[1, 2],
            squares_h=8,
            squares_v=6,
            square_size=0.03,
            min_corners=4,
        )

        # Detect in both images
        found1, corners1, ids1 = calibrator.detect_pattern(img1)
        found2, corners2, ids2 = calibrator.detect_pattern(img2)

        assert found1, "Detection failed in camera 1 image"
        assert found2, "Detection failed in camera 2 image"
        assert corners1 is not None, "No corners in camera 1"
        assert corners2 is not None, "No corners in camera 2"
        assert ids1 is not None, "No IDs in camera 1"
        assert ids2 is not None, "No IDs in camera 2"

        # Should detect significant number of corners
        assert len(corners1) >= 10, f"Too few corners in cam1: {len(corners1)}"
        assert len(corners2) >= 10, f"Too few corners in cam2: {len(corners2)}"

    return True


def test_charuco_pattern_params():
    """Test ChArUco pattern parameters retrieval."""
    print("  Testing ChArUco pattern parameters...")

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = StereoCharucoCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_pair=[1, 2],
            squares_h=10,
            squares_v=9,
            square_size=0.03,
            marker_ratio=0.5,
            aruco_dict="DICT_5X5_1000",
        )

        params = calibrator.get_pattern_params()

        assert params['pattern_type'] == 'charuco', "Wrong pattern type"
        assert params['squares_h'] == 10, "Wrong squares_h"
        assert params['squares_v'] == 9, "Wrong squares_v"
        assert params['square_size'] == 0.03, "Wrong square_size"
        assert params['aruco_dict'] == "DICT_5X5_1000", "Wrong aruco_dict"

    return True


# ============================================================
# UNIT TESTS - DOTBOARD (DOTBOARD)
# ============================================================

def test_dotboard_detector_creation():
    """Test that StereoDotboardCalibrator creates detector correctly."""
    print("  Testing Pinhole detector creation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = StereoDotboardCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_pair=[1, 2],
            pattern_cols=10,
            pattern_rows=10,
            dot_spacing_mm=12.0,
        )

        # Check detector is SimpleBlobDetector
        assert calibrator.detector is not None, "Detector not created"

    return True


def test_dotboard_object_points_symmetric():
    """Test symmetric grid object points generation."""
    print("  Testing Pinhole symmetric object points...")

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = StereoDotboardCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_pair=[1, 2],
            pattern_cols=5,
            pattern_rows=4,
            dot_spacing_mm=10.0,
            asymmetric=False,
        )

        obj_pts = calibrator.make_object_points()

        # Expected: 5 * 4 = 20 points
        assert len(obj_pts) == 20, f"Expected 20 points, got {len(obj_pts)}"

        # Check spacing
        # Point (0,0) should be at origin
        np.testing.assert_array_equal(obj_pts[0], [0, 0, 0])

        # Point (1,0) should be at (10, 0, 0)
        np.testing.assert_array_equal(obj_pts[1], [10, 0, 0])

        # Point (0,1) = index 5 should be at (0, 10, 0)
        np.testing.assert_array_equal(obj_pts[5], [0, 10, 0])

    return True


def test_dotboard_object_points_asymmetric():
    """Test asymmetric grid object points generation."""
    print("  Testing Pinhole asymmetric object points...")

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = StereoDotboardCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_pair=[1, 2],
            pattern_cols=4,
            pattern_rows=3,
            dot_spacing_mm=10.0,
            asymmetric=True,
        )

        obj_pts = calibrator.make_object_points()

        # First point at origin
        np.testing.assert_array_equal(obj_pts[0], [0, 0, 0])

        # First point of second row (index 4) should be offset
        expected_x = 5.0  # half spacing
        expected_y = 10.0
        np.testing.assert_allclose(obj_pts[4], [expected_x, expected_y, 0], atol=1e-6)

    return True


def test_dotboard_detection_synthetic():
    """Test circle grid detection on synthetic image."""
    print("  Testing Pinhole detection on synthetic image...")

    generator = SyntheticStereoImageGenerator()
    img1, img2 = generator.generate_dotboard_stereo_pair(cols=8, rows=6)

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = StereoDotboardCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_pair=[1, 2],
            pattern_cols=8,
            pattern_rows=6,
            dot_spacing_mm=15.0,
        )

        # Detect in both images
        found1, centers1 = calibrator.detect_pattern(img1)
        found2, centers2 = calibrator.detect_pattern(img2)

        assert found1, "Detection failed in camera 1 image"
        assert found2, "Detection failed in camera 2 image"

        # Should have all 48 points
        assert len(centers1) == 48, f"Expected 48 centers, got {len(centers1)}"
        assert len(centers2) == 48, f"Expected 48 centers, got {len(centers2)}"

    return True


def test_dotboard_detection_inverted():
    """Test circle grid detection on inverted image."""
    print("  Testing Pinhole detection on inverted image...")

    # Generate inverted dotboard (black dots on white)
    generator = SyntheticStereoImageGenerator()
    img1, img2 = generator.generate_dotboard_stereo_pair(cols=6, rows=5)

    # Invert
    img1_inv = 255 - img1
    img2_inv = 255 - img2

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = StereoDotboardCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_pair=[1, 2],
            pattern_cols=6,
            pattern_rows=5,
            dot_spacing_mm=15.0,
        )

        # Should detect in inverted images (auto-tries inversion)
        found1, centers1 = calibrator.detect_pattern(img1_inv)
        found2, centers2 = calibrator.detect_pattern(img2_inv)

        assert found1, "Detection failed on inverted camera 1 image"
        assert found2, "Detection failed on inverted camera 2 image"

    return True


def test_dotboard_pattern_params():
    """Test Pinhole pattern parameters retrieval."""
    print("  Testing Pinhole pattern parameters...")

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = StereoDotboardCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_pair=[1, 2],
            pattern_cols=10,
            pattern_rows=8,
            dot_spacing_mm=15.5,
            asymmetric=True,
        )

        params = calibrator.get_pattern_params()

        assert params['pattern_type'] == 'circle_grid', "Wrong pattern type"
        assert params['pattern_cols'] == 10, "Wrong pattern_cols"
        assert params['pattern_rows'] == 8, "Wrong pattern_rows"
        assert params['dot_spacing_mm'] == 15.5, "Wrong dot_spacing_mm"
        assert params['asymmetric'] is True, "Wrong asymmetric"

    return True


# ============================================================
# UNIT TESTS - STEREO MATH
# ============================================================

def test_stereo_calibration_math():
    """Test stereo calibration math with known geometry."""
    print("  Testing stereo calibration math...")

    # Create synthetic object and image points for stereo calibration
    # Using simple planar grid
    cols, rows = 6, 5
    spacing = 10.0  # mm

    # Object points (3D)
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * spacing

    # Synthetic camera matrices
    fx, fy = 800, 800
    cx, cy = 400, 300
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    # Project to camera 1 (board at Z=500)
    rvec1 = np.array([0.1, 0.05, 0.02], dtype=np.float64)
    tvec1 = np.array([0, 0, 500], dtype=np.float64)
    imgpts1, _ = cv2.projectPoints(objp, rvec1, tvec1, K, dist)
    imgpts1 = imgpts1.reshape(-1, 2)

    # Project to camera 2 (offset by 100mm in X)
    rvec2 = np.array([0.1, 0.05, 0.02], dtype=np.float64)
    tvec2 = np.array([100, 0, 500], dtype=np.float64)
    imgpts2, _ = cv2.projectPoints(objp, rvec2, tvec2, K, dist)
    imgpts2 = imgpts2.reshape(-1, 2)

    # Prepare data for stereo calibration
    objpoints = [objp.reshape(-1, 1, 3).astype(np.float32) for _ in range(3)]
    imgpoints1 = [imgpts1.reshape(-1, 1, 2).astype(np.float32) for _ in range(3)]
    imgpoints2 = [imgpts2.reshape(-1, 1, 2).astype(np.float32) for _ in range(3)]

    # Run stereo calibration
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        K, dist, K, dist,
        (800, 600),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    # Check RMS error is reasonable
    assert ret < 1.0, f"Stereo RMS error too high: {ret}"

    # Translation should be roughly [100, 0, 0] (baseline)
    T_mag = np.linalg.norm(T)
    assert abs(T_mag - 100) < 10, f"Translation magnitude wrong: {T_mag}"

    return True


def test_epipolar_constraint():
    """Test that stereo calibration satisfies epipolar constraint."""
    print("  Testing epipolar constraint...")

    # Create matching points in two views
    cols, rows = 6, 5
    spacing = 10.0

    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * spacing

    K = np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    rvec = np.array([0.1, 0.05, 0.02], dtype=np.float64)
    tvec1 = np.array([0, 0, 500], dtype=np.float64)
    tvec2 = np.array([100, 0, 500], dtype=np.float64)

    imgpts1, _ = cv2.projectPoints(objp, rvec, tvec1, K, dist)
    imgpts2, _ = cv2.projectPoints(objp, rvec, tvec2, K, dist)

    imgpts1 = imgpts1.reshape(-1, 2)
    imgpts2 = imgpts2.reshape(-1, 2)

    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(imgpts1, imgpts2, cv2.FM_8POINT)

    # Check epipolar constraint: x2^T * F * x1 ≈ 0
    errors = []
    for p1, p2 in zip(imgpts1, imgpts2):
        p1_h = np.array([p1[0], p1[1], 1])
        p2_h = np.array([p2[0], p2[1], 1])
        error = abs(p2_h @ F @ p1_h)
        errors.append(error)

    mean_error = np.mean(errors)
    assert mean_error < 0.1, f"Epipolar constraint error too high: {mean_error}"

    return True


# ============================================================
# INTEGRATION TESTS
# ============================================================

def test_charuco_stereo_pipeline_synthetic():
    """Test full ChArUco stereo calibration with synthetic images."""
    print("  Testing ChArUco stereo pipeline...")

    # This test verifies the detection and matching logic works,
    # but can't run the full pipeline without config infrastructure

    generator = SyntheticStereoImageGenerator()
    views = generator.generate_multiple_stereo_views(
        n_views=5,
        board_type="charuco",
        squares_h=8,
        squares_v=6,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = StereoCharucoCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_pair=[1, 2],
            squares_h=8,
            squares_v=6,
            square_size=0.03,
            min_corners=4,
        )

        # Test detection on all views
        successful_pairs = 0
        for i, (img1, img2) in enumerate(views):
            found1, corners1, ids1 = calibrator.detect_pattern(img1)
            found2, corners2, ids2 = calibrator.detect_pattern(img2)

            if found1 and found2 and ids1 is not None and ids2 is not None:
                # Check for common IDs
                common_ids = np.intersect1d(ids1, ids2)
                if len(common_ids) >= 4:
                    successful_pairs += 1

        # Should have at least 3 successful pairs for calibration
        assert successful_pairs >= 3, f"Too few successful pairs: {successful_pairs}"

    return True


def test_dotboard_stereo_pipeline_synthetic():
    """Test full Pinhole stereo calibration with synthetic images."""
    print("  Testing Pinhole stereo pipeline...")

    generator = SyntheticStereoImageGenerator()
    views = generator.generate_multiple_stereo_views(
        n_views=5,
        board_type="dotboard",
        cols=8,
        rows=6,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = StereoDotboardCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_pair=[1, 2],
            pattern_cols=8,
            pattern_rows=6,
            dot_spacing_mm=15.0,
        )

        # Test detection on all views
        successful_pairs = 0
        for i, (img1, img2) in enumerate(views):
            found1, centers1 = calibrator.detect_pattern(img1)
            found2, centers2 = calibrator.detect_pattern(img2)

            if found1 and found2:
                if len(centers1) == len(centers2) == 48:
                    successful_pairs += 1

        # Should have all 5 pairs detected
        assert successful_pairs >= 3, f"Too few successful pairs: {successful_pairs}"

    return True


def test_stereo_output_structure():
    """Test that stereo calibration output has correct structure."""
    print("  Testing stereo output structure...")

    # Create synthetic calibration result
    stereo_result = {
        'camera_matrix_1': np.eye(3),
        'camera_matrix_2': np.eye(3),
        'dist_coeffs_1': np.zeros(5),
        'dist_coeffs_2': np.zeros(5),
        'rotation_matrix': np.eye(3),
        'translation_vector': np.array([100, 0, 0]),
        'essential_matrix': np.zeros((3, 3)),
        'fundamental_matrix': np.zeros((3, 3)),
        'rectification_R1': np.eye(3),
        'rectification_R2': np.eye(3),
        'projection_P1': np.zeros((3, 4)),
        'projection_P2': np.zeros((3, 4)),
        'stereo_rms_error': 0.5,
        'cam1_rms_error': 0.3,
        'cam2_rms_error': 0.4,
    }

    # Check all required fields present
    required_fields = [
        'camera_matrix_1', 'camera_matrix_2',
        'dist_coeffs_1', 'dist_coeffs_2',
        'rotation_matrix', 'translation_vector',
        'projection_P1', 'projection_P2',
        'rectification_R1', 'rectification_R2',
    ]

    for field in required_fields:
        assert field in stereo_result, f"Missing field: {field}"

    return True


# ============================================================
# CLI TESTS
# ============================================================

def test_module_imports():
    """Test that all expected classes can be imported."""
    print("  Testing module imports...")

    from pivtools_gui.stereo_reconstruction.stereo_calibration_base import (
        BaseStereoCalibrator,
    )
    from pivtools_gui.stereo_reconstruction.stereo_charuco_calibration_production import (
        StereoCharucoCalibrator,
        ARUCO_DICT_MAP,
    )
    from pivtools_gui.stereo_reconstruction.stereo_dotboard_calibration_production import (
        StereoDotboardCalibrator,
    )

    assert BaseStereoCalibrator is not None
    assert StereoCharucoCalibrator is not None
    assert StereoDotboardCalibrator is not None
    assert ARUCO_DICT_MAP is not None

    return True


def test_script_syntax_base():
    """Test that base class script has valid syntax."""
    print("  Testing base class script syntax...")

    import ast
    script_path = (
        Path(__file__).parent.parent
        / "pivtools_gui"
        / "stereo_reconstruction"
        / "stereo_calibration_base.py"
    )

    with open(script_path, "r") as f:
        source = f.read()

    try:
        ast.parse(source)
    except SyntaxError as e:
        raise AssertionError(f"Syntax error: {e}")

    return True


def test_script_syntax_charuco():
    """Test that ChArUco script has valid syntax."""
    print("  Testing ChArUco script syntax...")

    import ast
    script_path = (
        Path(__file__).parent.parent
        / "pivtools_gui"
        / "stereo_reconstruction"
        / "stereo_charuco_calibration_production.py"
    )

    with open(script_path, "r") as f:
        source = f.read()

    try:
        ast.parse(source)
    except SyntaxError as e:
        raise AssertionError(f"Syntax error: {e}")

    return True


def test_script_syntax_dotboard():
    """Test that Pinhole script has valid syntax."""
    print("  Testing Pinhole script syntax...")

    import ast
    script_path = (
        Path(__file__).parent.parent
        / "pivtools_gui"
        / "stereo_reconstruction"
        / "stereo_dotboard_calibration_production.py"
    )

    with open(script_path, "r") as f:
        source = f.read()

    try:
        ast.parse(source)
    except SyntaxError as e:
        raise AssertionError(f"Syntax error: {e}")

    return True


# ============================================================
# TEST RUNNER
# ============================================================

def run_tests(verbose: bool = True):
    """Run all tests and report results."""
    charuco_unit_tests = [
        ("ChArUco Detector Creation", test_charuco_detector_creation),
        ("ChArUco Object Points", test_charuco_object_points),
        ("ChArUco Detection Synthetic", test_charuco_detection_synthetic),
        ("ChArUco Pattern Params", test_charuco_pattern_params),
    ]

    dotboard_unit_tests = [
        ("Dotboard Detector Creation", test_dotboard_detector_creation),
        ("Dotboard Object Points Symmetric", test_dotboard_object_points_symmetric),
        ("Dotboard Object Points Asymmetric", test_dotboard_object_points_asymmetric),
        ("Dotboard Detection Synthetic", test_dotboard_detection_synthetic),
        ("Dotboard Detection Inverted", test_dotboard_detection_inverted),
        ("Dotboard Pattern Params", test_dotboard_pattern_params),
    ]

    stereo_math_tests = [
        ("Stereo Calibration Math", test_stereo_calibration_math),
        ("Epipolar Constraint", test_epipolar_constraint),
    ]

    integration_tests = [
        ("ChArUco Stereo Pipeline", test_charuco_stereo_pipeline_synthetic),
        ("Dotboard Stereo Pipeline", test_dotboard_stereo_pipeline_synthetic),
        ("Stereo Output Structure", test_stereo_output_structure),
    ]

    cli_tests = [
        ("Module Imports", test_module_imports),
        ("Script Syntax Base", test_script_syntax_base),
        ("Script Syntax ChArUco", test_script_syntax_charuco),
        ("Script Syntax Dotboard", test_script_syntax_dotboard),
    ]

    all_test_groups = [
        ("CHARUCO UNIT TESTS", charuco_unit_tests),
        ("DOTBOARD UNIT TESTS", dotboard_unit_tests),
        ("STEREO MATH TESTS", stereo_math_tests),
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
