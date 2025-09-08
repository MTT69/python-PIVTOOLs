#!/usr/bin/env python3
"""
planar_calibration_production.py

Production-ready planar calibration script for individual cameras.
Processes calibration images, saves grid indexing, calibration models, and dewarped images.
"""

import glob
import logging
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

# ===================== CONFIGURATION VARIABLES =====================
# Set these variables for your calibration setup
SOURCE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Stereo_Images"
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Stereo_Images/ProcessedPIV"
CAMERA_COUNT = 2
FILE_PATTERN = "planar_calibration_plate_*.tif"  # or 'B%05d.tif' for numbered files

# Grid pattern parameters
PATTERN_COLS = 10
PATTERN_ROWS = 10
DOT_SPACING_MM = 28.89
ASYMMETRIC = False
ENHANCE_DOTS = True

# ===================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PlanarCalibrator:
    def __init__(
        self,
        source_dir,
        base_dir,
        camera_count,
        file_pattern,
        pattern_cols=10,
        pattern_rows=10,
        dot_spacing_mm=28.89,
        asymmetric=False,
        enhance_dots=False,
        dt=1.0,
    ):
        """
        Initialize planar calibrator

        Args:
            source_dir: Source directory containing calibration subdirectory
            base_dir: Base output directory
            camera_count: Number of cameras to process
            file_pattern: File pattern (e.g., 'B%05d.tif', 'planar_calibration_plate_*.tif')
            pattern_cols: Number of columns in calibration grid
            pattern_rows: Number of rows in calibration grid
            dot_spacing_mm: Physical spacing between dots in mm
            asymmetric: Whether grid is asymmetric
            enhance_dots: Whether to apply dot enhancement
            dt: Time step between frames in seconds
        """
        self.source_dir = Path(source_dir)
        self.base_dir = Path(base_dir)
        self.camera_count = camera_count
        self.file_pattern = file_pattern
        self.pattern_size = (pattern_cols, pattern_rows)
        self.dot_spacing_mm = dot_spacing_mm
        self.asymmetric = asymmetric
        self.enable_dot_enhancement = enhance_dots
        self.dt = dt  # Add dt parameter

        # Create blob detector
        self.detector = self._create_blob_detector()

        # Create base directories
        self._setup_directories()

    def _create_blob_detector(self):
        """Create optimized blob detector for circle grid detection"""
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 200
        params.maxArea = 1000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minThreshold = 0
        params.maxThreshold = 255
        params.thresholdStep = 5
        return cv2.SimpleBlobDetector_create(params)

    def _setup_directories(self):
        """Create necessary output directories"""
        for cam_num in range(1, self.camera_count + 1):
            cam_base = self.base_dir / "calibration" / f"Cam{cam_num}"
            (cam_base / "grid").mkdir(parents=True, exist_ok=True)
            (cam_base / "models").mkdir(parents=True, exist_ok=True)
            (cam_base / "dewarped").mkdir(parents=True, exist_ok=True)

    def enhance_dots_image(self, img, fixed_radius=9):
        """
        Enhance white dots in calibration image for better detection

        Args:
            img: Input grayscale image
            fixed_radius: Radius for enhanced dots

        Returns:
            Enhanced image
        """
        # Threshold to binary to isolate white dots
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours (each white dot)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create output as copy of original
        output = img.copy()

        for cnt in contours:
            # Find center of each dot
            (x, y), _ = cv2.minEnclosingCircle(cnt)
            center = (int(round(x)), int(round(y)))

            # Draw filled white circle with fixed radius
            cv2.circle(output, center, fixed_radius, (255,), -1)

        return output

    def make_object_points(self):
        """Create 3D object points for calibration grid"""
        cols, rows = self.pattern_size
        objp = []
        for i in range(rows):
            for j in range(cols):
                if self.asymmetric:
                    x = j * self.dot_spacing_mm + (
                        0.5 * self.dot_spacing_mm if (i % 2 == 1) else 0.0
                    )
                    y = i * self.dot_spacing_mm
                else:
                    x = j * self.dot_spacing_mm
                    y = i * self.dot_spacing_mm
                objp.append([x, y, 0.0])
        return np.array(objp, dtype=np.float32)

    def detect_grid_in_image(self, img):
        """
        Detect circle grid in image

        Args:
            img: Input image

        Returns:
            (found, centers) - boolean and Nx2 array of points
        """
        # Convert to grayscale if needed
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Apply dot enhancement if requested
        if self.enable_dot_enhancement:
            gray = self.enhance_dots_image(gray)

        # Grid detection flags
        grid_flags = (
            cv2.CALIB_CB_ASYMMETRIC_GRID
            if self.asymmetric
            else cv2.CALIB_CB_SYMMETRIC_GRID
        )

        # Try both original and inverted images
        for test_img, label in [(gray, "Original"), (255 - gray, "Inverted")]:
            found, centers = cv2.findCirclesGrid(
                test_img,
                self.pattern_size,
                flags=grid_flags,
                blobDetector=self.detector,
            )

            if found:
                logger.info(f"Grid detected ({label} image)")
                return True, centers.reshape(-1, 2).astype(np.float32)

        return False, None

    def calculate_reprojection_error(self, grid_points, objp_2d, H):
        """Calculate reprojection error using homography"""
        H_inv = np.linalg.inv(H)
        objp_h = np.hstack([objp_2d, np.ones((objp_2d.shape[0], 1))])
        projected = (H_inv @ objp_h.T).T
        projected = projected[:, :2] / projected[:, 2:]
        errors = np.linalg.norm(grid_points - projected, axis=1)
        return errors.mean(), errors

    def calculate_dewarped_size(self, H, img_shape):
        """Calculate optimal output size for dewarped image"""
        h, w = img_shape[:2]
        corners = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        ).reshape(-1, 1, 2)
        physical_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        min_x, max_x = np.min(physical_corners[:, 0]), np.max(physical_corners[:, 0])
        min_y, max_y = np.min(physical_corners[:, 1]), np.max(physical_corners[:, 1])

        width_px = int(np.ceil(max_x - min_x))
        height_px = int(np.ceil(max_y - min_y))

        physical_to_pixel = np.array(
            [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32
        )
        combined_H = physical_to_pixel @ H

        return (width_px, height_px), combined_H

    def process_camera(self, cam_num):
        """
        Process all calibration images for one camera

        Args:
            cam_num: Camera number (1-based)
        """
        logger.info(f"Processing Camera {cam_num}")

        # Setup paths
        cam_input_dir = self.source_dir / "calibration" / f"Cam{cam_num}"
        cam_output_base = self.base_dir / "calibration" / f"Cam{cam_num}"

        if not cam_input_dir.exists():
            logger.error(f"Camera directory not found: {cam_input_dir}")
            return

        # Find calibration images
        if "%" in self.file_pattern:
            # Handle numbered patterns like B%05d.tif
            image_files = []
            i = 1
            while True:
                filename = self.file_pattern % i
                filepath = cam_input_dir / filename
                if filepath.exists():
                    image_files.append(str(filepath))
                    i += 1
                else:
                    break
        else:
            # Handle glob patterns like planar_calibration_plate_*.tif
            image_files = sorted(glob.glob(str(cam_input_dir / self.file_pattern)))

        if not image_files:
            logger.error(
                f"No images found in {cam_input_dir} with pattern {self.file_pattern}"
            )
            return

        logger.info(f"Found {len(image_files)} images for Camera {cam_num}")

        # Create object points
        objp = self.make_object_points()
        objp_2d = objp[:, :2]

        # Process each image
        for i, img_path in enumerate(image_files):
            try:
                self._process_single_image(img_path, i, cam_output_base, objp_2d)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {str(e)}")
                continue

    def _process_single_image(self, img_path, img_index, cam_output_base, objp_2d):
        """Process a single calibration image"""
        logger.info(f"Processing image {img_index}: {Path(img_path).name}")

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Detect grid
        found, grid_points = self.detect_grid_in_image(img)
        if not found:
            raise ValueError("Grid not detected")

        # Ensure we have the right number of points
        expected_points = self.pattern_size[0] * self.pattern_size[1]
        if len(grid_points) != expected_points:
            logger.warning(f"Expected {expected_points} points, got {len(grid_points)}")
            objp_2d_used = objp_2d[: len(grid_points)]
        else:
            objp_2d_used = objp_2d

        # Compute homography
        H, _ = cv2.findHomography(grid_points, objp_2d_used, cv2.RANSAC, 3.0)

        # Calculate reprojection error
        mean_error, _ = self.calculate_reprojection_error(grid_points, objp_2d_used, H)
        logger.info(f"Reprojection error: {mean_error:.2f} px")

        # Fit camera model
        camera_model = self._fit_camera_model(grid_points, objp_2d_used)

        # Calculate dewarped image
        output_size, combined_H = self.calculate_dewarped_size(H, img.shape)
        dewarped = cv2.warpPerspective(
            img, combined_H, output_size, flags=cv2.INTER_LANCZOS4
        )

        # Save results
        self._save_results(
            img_index,
            cam_output_base,
            grid_points,
            H,
            camera_model,
            dewarped,
            mean_error,
            Path(img_path).name,
        )

    def _fit_camera_model(self, image_points, object_points_2d):
        """Fit camera model using OpenCV"""
        # Convert to proper format for OpenCV
        img_pts = image_points.reshape(-1, 1, 2).astype(np.float32)

        # Create 3D object points (add z=0)
        obj_pts_3d = np.hstack(
            [object_points_2d, np.zeros((len(object_points_2d), 1))]
        ).astype(np.float32)
        obj_pts = obj_pts_3d.reshape(-1, 1, 3)

        # Prepare lists for calibrateCamera (it expects lists of arrays)
        objpoints = [obj_pts]
        imgpoints = [img_pts]

        # Estimate image size from image points
        w = int(np.max(image_points[:, 0])) + 100
        h = int(np.max(image_points[:, 1])) + 100

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), None, None
        )

        # Calculate reprojection error
        imgpoints2, _ = cv2.projectPoints(obj_pts, rvecs[0], tvecs[0], mtx, dist)
        error = cv2.norm(img_pts, imgpoints2, cv2.NORM_L2) / len(imgpoints2)

        return {
            "camera_matrix": mtx,
            "dist_coeffs": dist,
            "rvecs": rvecs[0],
            "tvecs": tvecs[0],
            "reprojection_error": error,
        }

    def _save_results(
        self,
        img_index,
        cam_output_base,
        grid_points,
        H,
        camera_model,
        dewarped,
        reprojection_error,
        original_filename,
    ):
        """Save all calibration results"""
        # Save grid indexing
        grid_data = {
            "grid_points": grid_points,
            "homography": H,
            "reprojection_error": reprojection_error,
            "original_filename": original_filename,
            "pattern_size": self.pattern_size,
            "dot_spacing_mm": self.dot_spacing_mm,
            "dt": self.dt,  # Add dt to grid data
            "timestamp": datetime.now().isoformat(),
        }
        savemat(cam_output_base / "grid" / f"indexing_{img_index}.mat", grid_data)

        # Save calibration model - CRITICAL: Include dt here
        model_data = {
            "camera_matrix": camera_model["camera_matrix"],
            "dist_coeffs": camera_model["dist_coeffs"],
            "rvecs": camera_model["rvecs"],
            "tvecs": camera_model["tvecs"],
            "reprojection_error": camera_model["reprojection_error"],
            "grid_points": grid_points,
            "homography": H,
            "original_filename": original_filename,
            "pattern_size": self.pattern_size,
            "dot_spacing_mm": self.dot_spacing_mm,
            "dt": self.dt,  # IMPORTANT: Save dt with the model for vector calibration
            "timestamp": datetime.now().isoformat(),
        }
        savemat(cam_output_base / "models" / f"{img_index}.mat", model_data)

        # Save dewarped image
        dewarped_path = (
            cam_output_base
            / "dewarped"
            / f"{Path(original_filename).stem}_dewarped.tif"
        )
        cv2.imwrite(str(dewarped_path), dewarped)

        # Save grid visualization with indices
        self._save_grid_visualization(
            img_index,
            cam_output_base,
            grid_points,
            original_filename,
            reprojection_error,
        )

        logger.info(f"Saved results for image {img_index}")

    def _save_grid_visualization(
        self,
        img_index,
        cam_output_base,
        grid_points,
        original_filename,
        reprojection_error,
    ):
        """Save a figure showing the detected grid with dot indices"""
        try:
            # Load original image for background
            img_path = (
                self.source_dir
                / "calibration"
                / cam_output_base.name
                / original_filename
            )
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            cols, rows = self.pattern_size

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))

            # Display image
            ax.imshow(img, cmap="gray", alpha=0.7)

            # Plot detected grid points with indices
            for idx, (x, y) in enumerate(grid_points):
                # Calculate grid coordinates (row, col)
                row = idx // cols
                col = idx % cols

                # Plot point
                ax.scatter(x, y, c="red", s=60, marker="o", alpha=0.8)

                # Add index label
                ax.text(
                    x + 10,
                    y - 10,
                    f"({row},{col})",
                    color="cyan",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
                )

            ax.set_title(
                f"Grid Detection: {original_filename}\n"
                f"Detected: {len(grid_points)} points | "
                f"Reprojection Error: {reprojection_error:.2f}px",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            ax.grid(True, alpha=0.3)

            # Invert y-axis to match image coordinates
            ax.invert_yaxis()

            # Save figure
            fig_path = cam_output_base / "grid" / f"grid_visualization_{img_index}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            logger.info(f"Saved grid visualization: {fig_path}")

        except Exception as e:
            logger.warning(f"Failed to save grid visualization: {str(e)}")

    def run(self):
        """Run calibration for all cameras"""
        logger.info(f"Starting planar calibration for {self.camera_count} cameras")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.base_dir}")
        logger.info(f"Pattern: {self.file_pattern}")
        logger.info(f"Grid size: {self.pattern_size}")
        logger.info(f"Dot spacing: {self.dot_spacing_mm} mm")
        logger.info(f"Dot enhancement: {self.enable_dot_enhancement}")

        for cam_num in range(1, self.camera_count + 1):
            try:
                self.process_camera(cam_num)
            except Exception as e:
                logger.error(f"Failed to process Camera {cam_num}: {str(e)}")
                continue

        logger.info("Planar calibration completed")


def main():
    calibrator = PlanarCalibrator(
        source_dir=SOURCE_DIR,
        base_dir=BASE_DIR,
        camera_count=CAMERA_COUNT,
        file_pattern=FILE_PATTERN,
        pattern_cols=PATTERN_COLS,
        pattern_rows=PATTERN_ROWS,
        dot_spacing_mm=DOT_SPACING_MM,
        asymmetric=ASYMMETRIC,
        enhance_dots=ENHANCE_DOTS,
    )

    calibrator.run()


if __name__ == "__main__":
    main()
