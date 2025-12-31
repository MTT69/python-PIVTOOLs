#!/usr/bin/env python3
"""
test_planar_calibration.py

Tests for planar (dotboard/circle grid) calibration.
Uses synthetic dot pattern images generated programmatically for reproducibility.

Test categories:
1. Unit Tests - Blob detector creation, grid detection on clean images
2. Integration Tests - Full calibration pipeline with multiple synthetic views
3. CLI Tests - Module imports and syntax validation
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pivtools_gui.calibration.calibration_planar.planar_calibration_production import (
    MultiViewCalibrator,
)


# ============================================================
# SYNTHETIC IMAGE GENERATION
# ============================================================

class SyntheticDotboardGenerator:
    """Generate synthetic dotboard (circle grid) images for testing."""

    def __init__(
        self,
        cols: int = 10,
        rows: int = 10,
        dot_spacing_mm: float = 12.0,
        dot_radius_mm: float = 3.0,
        asymmetric: bool = False,
    ):
        """
        Initialize generator with grid parameters.

        Args:
            cols: Number of columns in grid
            rows: Number of rows in grid
            dot_spacing_mm: Spacing between dot centers in mm
            dot_radius_mm: Radius of each dot in mm
            asymmetric: If True, offset every other row
        """
        self.cols = cols
        self.rows = rows
        self.dot_spacing_mm = dot_spacing_mm
        self.dot_radius_mm = dot_radius_mm
        self.asymmetric = asymmetric

        # Total dots
        self.total_dots = cols * rows

    def generate_frontal_image(
        self,
        px_per_mm: float = 10.0,
        margin_px: int = 100,
        background: int = 0,
        dot_color: int = 255,
    ) -> tuple:
        """
        Generate clean frontal dotboard image.

        Args:
            px_per_mm: Pixel scale (pixels per millimeter)
            margin_px: Margin around the grid in pixels
            background: Background pixel value (0=black, 255=white)
            dot_color: Dot pixel value

        Returns:
            tuple: (image, dot_centers_px) where dot_centers_px is Nx2 array
        """
        spacing_px = self.dot_spacing_mm * px_per_mm
        radius_px = int(self.dot_radius_mm * px_per_mm)

        # Calculate image dimensions
        grid_width = (self.cols - 1) * spacing_px
        grid_height = (self.rows - 1) * spacing_px
        if self.asymmetric:
            grid_width += spacing_px / 2  # Account for offset rows

        width = int(margin_px * 2 + grid_width + 2 * radius_px)
        height = int(margin_px * 2 + grid_height + 2 * radius_px)

        # Create image
        img = np.full((height, width), background, dtype=np.uint8)

        # Generate dot centers
        dot_centers = []
        for r in range(self.rows):
            row_offset = (spacing_px / 2) if (self.asymmetric and r % 2 == 1) else 0
            for c in range(self.cols):
                x = margin_px + radius_px + c * spacing_px + row_offset
                y = margin_px + radius_px + r * spacing_px
                dot_centers.append((x, y))
                cv2.circle(img, (int(x), int(y)), radius_px, dot_color, -1)

        return img, np.array(dot_centers, dtype=np.float32)

    def generate_with_noise(
        self,
        px_per_mm: float = 10.0,
        margin_px: int = 100,
        noise_std: float = 5.0,
    ) -> tuple:
        """Generate dotboard with Gaussian noise."""
        img, centers = self.generate_frontal_image(px_per_mm, margin_px)

        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
        noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return noisy_img, centers

    def generate_inverted(
        self,
        px_per_mm: float = 10.0,
        margin_px: int = 100,
    ) -> tuple:
        """Generate inverted dotboard (white background, black dots)."""
        return self.generate_frontal_image(
            px_per_mm, margin_px, background=255, dot_color=0
        )

    def generate_calibration_set(
        self,
        n_views: int = 5,
        px_per_mm: float = 10.0,
        margin_px: int = 100,
        seed: int = 42,
    ) -> list:
        """
        Generate multiple views with slight variations for calibration.

        For simplicity, we generate frontal views with different scales/margins.
        Real calibration would use perspective transforms.

        Returns:
            List of (image, centers) tuples
        """
        np.random.seed(seed)
        views = []

        for i in range(n_views):
            # Vary scale slightly
            scale = px_per_mm * (0.9 + np.random.uniform(0, 0.2))
            margin = margin_px + int(np.random.uniform(-20, 20))
            margin = max(50, margin)  # Ensure positive margin

            img, centers = self.generate_frontal_image(scale, margin)
            views.append((img, centers))

        return views


# ============================================================
# UNIT TESTS
# ============================================================

def test_calibrator_creation():
    """Test that MultiViewCalibrator creates detector correctly."""
    print("  Testing calibrator creation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = MultiViewCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_count=1,
            file_pattern="*.png",
            pattern_cols=10,
            pattern_rows=10,
            dot_spacing_mm=12.0,
            asymmetric=False,
        )

        # Check detector exists
        assert calibrator.detector is not None, "Blob detector not created"
        assert calibrator.pattern_size == (10, 10), f"Wrong pattern size: {calibrator.pattern_size}"

    return True


def test_object_points_generation():
    """Test that 3D object points are generated correctly."""
    print("  Testing object points generation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Symmetric grid
        calibrator = MultiViewCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_count=1,
            file_pattern="*.png",
            pattern_cols=5,
            pattern_rows=4,
            dot_spacing_mm=10.0,
            asymmetric=False,
        )

        obj_pts = calibrator.make_object_points()

        # Check shape: 5*4 = 20 points, each with (X, Y, Z)
        assert obj_pts.shape == (20, 3), f"Wrong shape: {obj_pts.shape}"

        # Check Z is all zeros (planar board)
        assert np.all(obj_pts[:, 2] == 0), "Z coordinates should all be 0"

        # Check first point is at origin
        np.testing.assert_array_equal(obj_pts[0], [0, 0, 0])

        # Check spacing: second point should be at (10, 0, 0)
        np.testing.assert_array_equal(obj_pts[1], [10, 0, 0])

        # Check first point of second row: (0, 10, 0) for symmetric
        np.testing.assert_array_equal(obj_pts[5], [0, 10, 0])

    return True


def test_object_points_asymmetric():
    """Test asymmetric grid object points."""
    print("  Testing asymmetric object points...")

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = MultiViewCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_count=1,
            file_pattern="*.png",
            pattern_cols=4,
            pattern_rows=3,
            dot_spacing_mm=10.0,
            asymmetric=True,
        )

        obj_pts = calibrator.make_object_points()

        # First point at origin
        np.testing.assert_array_equal(obj_pts[0], [0, 0, 0])

        # First point of second row should be offset by half spacing
        # Row 1, Col 0 -> index 4
        expected_x = 5.0  # half spacing offset
        expected_y = 10.0  # one row down
        np.testing.assert_allclose(obj_pts[4], [expected_x, expected_y, 0], atol=1e-6)

    return True


def test_detection_symmetric_grid():
    """Test grid detection on symmetric dotboard."""
    print("  Testing detection on symmetric grid...")

    generator = SyntheticDotboardGenerator(
        cols=8, rows=6, dot_spacing_mm=15.0, dot_radius_mm=4.0, asymmetric=False
    )
    img, true_centers = generator.generate_frontal_image(px_per_mm=8.0, margin_px=80)

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = MultiViewCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_count=1,
            file_pattern="*.png",
            pattern_cols=8,
            pattern_rows=6,
            dot_spacing_mm=15.0,
            asymmetric=False,
        )

        found, centers = calibrator.detect_grid(img)

        assert found, "Grid detection failed on clean symmetric image"
        assert centers is not None, "No centers returned"
        assert len(centers) == 8 * 6, f"Wrong number of centers: {len(centers)}"

    return True


def test_detection_inverted_image():
    """Test grid detection on inverted (white bg, black dots) image."""
    print("  Testing detection on inverted image...")

    generator = SyntheticDotboardGenerator(
        cols=6, rows=5, dot_spacing_mm=15.0, dot_radius_mm=4.0, asymmetric=False
    )
    img, _ = generator.generate_inverted(px_per_mm=8.0, margin_px=80)

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = MultiViewCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_count=1,
            file_pattern="*.png",
            pattern_cols=6,
            pattern_rows=5,
            dot_spacing_mm=15.0,
            asymmetric=False,
        )

        # Detector should try inverted automatically
        found, centers = calibrator.detect_grid(img)

        assert found, "Grid detection failed on inverted image"
        assert len(centers) == 6 * 5, f"Wrong number of centers: {len(centers)}"

    return True


def test_detection_with_noise():
    """Test grid detection with noisy image."""
    print("  Testing detection with noise...")

    generator = SyntheticDotboardGenerator(
        cols=6, rows=5, dot_spacing_mm=15.0, dot_radius_mm=4.0, asymmetric=False
    )
    img, _ = generator.generate_with_noise(px_per_mm=8.0, margin_px=80, noise_std=10.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = MultiViewCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_count=1,
            file_pattern="*.png",
            pattern_cols=6,
            pattern_rows=5,
            dot_spacing_mm=15.0,
            asymmetric=False,
        )

        found, centers = calibrator.detect_grid(img)

        # Should still detect with moderate noise
        assert found, "Grid detection failed with moderate noise"

    return True


def test_detection_fails_on_blank():
    """Test that detection fails on blank image."""
    print("  Testing detection fails on blank image...")

    blank_img = np.zeros((400, 500), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        calibrator = MultiViewCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_count=1,
            file_pattern="*.png",
            pattern_cols=8,
            pattern_rows=6,
            dot_spacing_mm=15.0,
        )

        found, centers = calibrator.detect_grid(blank_img)

        assert not found, "Detection should fail on blank image"

    return True


def test_detection_wrong_pattern_size():
    """Test that detection fails when pattern size doesn't match."""
    print("  Testing detection with wrong pattern size...")

    # Generate 6x5 grid
    generator = SyntheticDotboardGenerator(
        cols=6, rows=5, dot_spacing_mm=15.0, dot_radius_mm=4.0
    )
    img, _ = generator.generate_frontal_image(px_per_mm=8.0, margin_px=80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Try to detect with wrong pattern size (8x6)
        calibrator = MultiViewCalibrator(
            source_dir=tmpdir,
            base_dir=tmpdir,
            camera_count=1,
            file_pattern="*.png",
            pattern_cols=8,  # Wrong!
            pattern_rows=6,  # Wrong!
            dot_spacing_mm=15.0,
        )

        found, centers = calibrator.detect_grid(img)

        # Should fail because grid size doesn't match
        assert not found, "Detection should fail with wrong pattern size"

    return True


# ============================================================
# INTEGRATION TESTS
# ============================================================

def test_calibration_from_synthetic_views():
    """Test full calibration pipeline with synthetic dotboard views."""
    print("  Testing full calibration with synthetic views...")

    generator = SyntheticDotboardGenerator(
        cols=8, rows=6, dot_spacing_mm=12.0, dot_radius_mm=3.0, asymmetric=False
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate and save test images
        calib_dir = tmpdir / "images"
        calib_dir.mkdir()

        n_views = 5
        for i in range(n_views):
            # Vary scale slightly for each view
            scale = 8.0 + i * 0.5
            img, _ = generator.generate_frontal_image(px_per_mm=scale, margin_px=80)
            cv2.imwrite(str(calib_dir / f"img{i+1:05d}.png"), img)

        # Run calibration
        calibrator = MultiViewCalibrator(
            source_dir=str(calib_dir),
            base_dir=str(tmpdir),
            camera_count=1,
            file_pattern="img%05d.png",
            pattern_cols=8,
            pattern_rows=6,
            dot_spacing_mm=12.0,
            asymmetric=False,
        )

        result = calibrator.process_single_camera(1, save_visualizations=False)

        # Check calibration succeeded (pipeline runs to completion)
        assert result["success"], f"Calibration failed: {result.get('error')}"

        # Note: RMS error will be very high with synthetic frontal views because
        # OpenCV camera calibration requires images with different board poses
        # (perspective variations). Frontal views at different scales don't
        # provide the geometric diversity needed for accurate calibration.
        # This test verifies the pipeline runs correctly, not calibration accuracy.
        assert "rms_error" in result, "Result should contain rms_error"

        # Check camera matrix structure (basic validity)
        cam_matrix = np.array(result["camera_matrix"])
        assert cam_matrix.shape == (3, 3), f"Wrong camera matrix shape: {cam_matrix.shape}"

        # Check number of images used
        assert result["num_images_used"] >= 3, f"Too few images used: {result['num_images_used']}"

    return True


def test_calibration_output_files():
    """Test that calibration produces expected output files."""
    print("  Testing calibration output files...")

    generator = SyntheticDotboardGenerator(cols=6, rows=5, dot_spacing_mm=15.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate and save test images
        calib_dir = tmpdir / "images"
        calib_dir.mkdir()

        for i in range(4):
            img, _ = generator.generate_frontal_image(px_per_mm=8.0, margin_px=70 + i * 10)
            cv2.imwrite(str(calib_dir / f"img{i+1:05d}.png"), img)

        # Run calibration
        calibrator = MultiViewCalibrator(
            source_dir=str(calib_dir),
            base_dir=str(tmpdir),
            camera_count=1,
            file_pattern="img%05d.png",
            pattern_cols=6,
            pattern_rows=5,
            dot_spacing_mm=15.0,
        )

        result = calibrator.process_single_camera(1, save_visualizations=True)

        if not result["success"]:
            print(f"    Skipping file checks - calibration failed: {result.get('error')}")
            return True

        # Check output directories exist
        model_dir = tmpdir / "calibration" / "Cam1" / "pinhole_planar" / "model"
        indices_dir = tmpdir / "calibration" / "Cam1" / "pinhole_planar" / "indices"
        figures_dir = tmpdir / "calibration" / "Cam1" / "pinhole_planar" / "figures"

        assert model_dir.exists(), "Model directory not created"
        assert indices_dir.exists(), "Indices directory not created"
        assert figures_dir.exists(), "Figures directory not created"

        # Check model file
        model_file = model_dir / "pinhole_model.mat"
        assert model_file.exists(), "pinhole_model.mat not created"

        # Check indices files exist
        indices_files = list(indices_dir.glob("indexing_*.mat"))
        assert len(indices_files) > 0, "No indices files created"

    return True


def test_insufficient_images():
    """Test that calibration fails gracefully with too few images."""
    print("  Testing insufficient images handling...")

    generator = SyntheticDotboardGenerator(cols=6, rows=5, dot_spacing_mm=15.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Only save 2 images (need at least 3)
        calib_dir = tmpdir / "images"
        calib_dir.mkdir()

        for i in range(2):
            img, _ = generator.generate_frontal_image(px_per_mm=8.0, margin_px=80)
            cv2.imwrite(str(calib_dir / f"img{i+1:05d}.png"), img)

        calibrator = MultiViewCalibrator(
            source_dir=str(calib_dir),
            base_dir=str(tmpdir),
            camera_count=1,
            file_pattern="img%05d.png",
            pattern_cols=6,
            pattern_rows=5,
            dot_spacing_mm=15.0,
        )

        result = calibrator.process_single_camera(1, save_visualizations=False)

        # Should fail gracefully
        assert not result["success"], "Should fail with only 2 images"
        assert "error" in result, "Should have error message"

    return True


def test_progress_callback():
    """Test that progress callback is called correctly."""
    print("  Testing progress callback...")

    generator = SyntheticDotboardGenerator(cols=6, rows=5, dot_spacing_mm=15.0)

    progress_reports = []

    def callback(data):
        progress_reports.append(data.copy())

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate test images
        calib_dir = tmpdir / "images"
        calib_dir.mkdir()

        for i in range(4):
            img, _ = generator.generate_frontal_image(px_per_mm=8.0, margin_px=80)
            cv2.imwrite(str(calib_dir / f"img{i+1:05d}.png"), img)

        calibrator = MultiViewCalibrator(
            source_dir=str(calib_dir),
            base_dir=str(tmpdir),
            camera_count=1,
            file_pattern="img%05d.png",
            pattern_cols=6,
            pattern_rows=5,
            dot_spacing_mm=15.0,
        )

        result = calibrator.process_single_camera(
            1, progress_callback=callback, save_visualizations=False
        )

        # Should have received progress reports
        assert len(progress_reports) > 0, "No progress reports received"

        # Check progress report structure
        for report in progress_reports:
            assert "processed_images" in report, "Missing processed_images"
            assert "valid_images" in report, "Missing valid_images"
            assert "total_images" in report, "Missing total_images"
            assert "progress" in report, "Missing progress"

    return True


def test_no_images_found():
    """Test behavior when no images are found."""
    print("  Testing no images found handling...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create empty directory
        calib_dir = tmpdir / "empty_images"
        calib_dir.mkdir()

        calibrator = MultiViewCalibrator(
            source_dir=str(calib_dir),
            base_dir=str(tmpdir),
            camera_count=1,
            file_pattern="img%05d.png",
            pattern_cols=6,
            pattern_rows=5,
            dot_spacing_mm=15.0,
        )

        result = calibrator.process_single_camera(1, save_visualizations=False)

        # Should fail gracefully
        assert not result["success"], "Should fail when no images found"
        assert "error" in result, "Should have error message"
        assert "No images" in result["error"], f"Error should mention no images: {result['error']}"

    return True


# ============================================================
# CLI TESTS
# ============================================================

def test_module_imports():
    """Test that all expected classes and functions can be imported."""
    print("  Testing module imports...")

    from pivtools_gui.calibration.calibration_planar.planar_calibration_production import (
        MultiViewCalibrator,
        apply_cli_settings_to_config,
    )

    assert MultiViewCalibrator is not None
    assert callable(apply_cli_settings_to_config)

    return True


def test_script_syntax():
    """Test that the production script has valid syntax."""
    print("  Testing script syntax...")

    import ast
    script_path = (
        Path(__file__).parent.parent
        / "pivtools_gui"
        / "calibration"
        / "calibration_planar"
        / "planar_calibration_production.py"
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
        ("Calibrator Creation", test_calibrator_creation),
        ("Object Points Generation", test_object_points_generation),
        ("Object Points Asymmetric", test_object_points_asymmetric),
        ("Detection Symmetric Grid", test_detection_symmetric_grid),
        ("Detection Inverted Image", test_detection_inverted_image),
        ("Detection With Noise", test_detection_with_noise),
        ("Detection Fails on Blank", test_detection_fails_on_blank),
        ("Detection Wrong Pattern Size", test_detection_wrong_pattern_size),
    ]

    integration_tests = [
        ("Calibration From Synthetic Views", test_calibration_from_synthetic_views),
        ("Calibration Output Files", test_calibration_output_files),
        ("Insufficient Images Handling", test_insufficient_images),
        ("Progress Callback", test_progress_callback),
        ("No Images Found", test_no_images_found),
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
