#!/usr/bin/env python3
"""
stereo_calibration_production.py

Production-ready stereo calibration script for camera pairs.
Processes calibration images, saves individual camera results and stereo reconstruction data.
"""

import glob
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from scipy.io import savemat

# ===================== CONFIGURATION VARIABLES =====================
# Set these variables for your calibration setup
SOURCE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Stereo_Images"
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Stereo_Images/ProcessedPIV"
CAMERA_PAIRS = [[1, 2]]  # Array of camera pairs
FILE_PATTERN = "planar_calibration_plate_*.tif"  # or 'B%05d.tif' for numbered files

# Grid pattern parameters
PATTERN_COLS = 10
PATTERN_ROWS = 10
DOT_SPACING_MM = 28.89  # Physical spacing between dots in mm
ASYMMETRIC = False
ENHANCE_DOTS = True

# ===================================================================


class StereoCalibrator:
    def __init__(
        self,
        source_dir,
        base_dir,
        camera_pairs,
        file_pattern,
        pattern_cols=10,
        pattern_rows=10,
        dot_spacing_mm=28.89,
        asymmetric=False,
        enhance_dots=False,
    ):
        """
        Initialize stereo calibrator

        Args:
            source_dir: Source directory containing calibration subdirectory
            base_dir: Base output directory
            camera_pairs: Array of camera pairs [[1,2], [3,4], ...]
            file_pattern: File pattern (e.g., 'B%05d.tif', 'planar_calibration_plate_*.tif')
            pattern_cols: Number of columns in calibration grid
            pattern_rows: Number of rows in calibration grid
            dot_spacing_mm: Physical spacing between dots in mm
            asymmetric: Whether grid is asymmetric
            enhance_dots: Whether to apply dot enhancement
        """
        self.source_dir = Path(source_dir)
        self.base_dir = Path(base_dir)
        self.camera_pairs = camera_pairs
        self.file_pattern = file_pattern
        self.pattern_size = (pattern_cols, pattern_rows)
        self.dot_spacing_mm = dot_spacing_mm
        self.asymmetric = asymmetric
        self.enhance_dots = enhance_dots

        # Get all unique cameras
        self.all_cameras = sorted(set([cam for pair in camera_pairs for cam in pair]))

        # Create blob detector
        self.detector = self._create_blob_detector()

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

    def enhance_dots_image(self, img, fixed_radius=9):
        """Enhance white dots in calibration image for better detection"""
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        output = img.copy()
        for cnt in contours:
            (x, y), _ = cv2.minEnclosingCircle(cnt)
            center = (int(round(x)), int(round(y)))
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
        if self.enhance_dots:
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
                return True, centers.reshape(-1, 2).astype(np.float32)

        return False, None

    def get_image_files(self, cam_dir):
        """Get list of calibration image files for a camera"""
        if "%" in self.file_pattern:
            # Handle numbered patterns like B%05d.tif
            image_files = []
            i = 1
            while True:
                filename = self.file_pattern % i
                filepath = cam_dir / filename
                if filepath.exists():
                    image_files.append(filepath)
                    i += 1
                else:
                    break
        else:
            # Handle glob patterns like planar_calibration_plate_*.tif
            image_files = sorted(
                [Path(p) for p in glob.glob(str(cam_dir / self.file_pattern))]
            )

        return image_files

    def process_camera_pair(self, cam1_num, cam2_num):
        """
        Process a camera pair for stereo calibration

        Args:
            cam1_num: First camera number
            cam2_num: Second camera number
        """
        logger.info(f"Processing stereo pair: Camera {cam1_num} and Camera {cam2_num}")

        # Setup paths
        cam1_input_dir = self.source_dir / "calibration" / f"Cam{cam1_num}"
        cam2_input_dir = self.source_dir / "calibration" / f"Cam{cam2_num}"

        if not cam1_input_dir.exists() or not cam2_input_dir.exists():
            logger.error(
                f"Camera directories not found: {cam1_input_dir} or {cam2_input_dir}"
            )
            return

        # Get image files for both cameras
        cam1_files = self.get_image_files(cam1_input_dir)
        cam2_files = self.get_image_files(cam2_input_dir)

        if not cam1_files or not cam2_files:
            logger.error(f"No images found for camera pair {cam1_num}-{cam2_num}")
            return

        # Match files by name
        cam1_dict = {f.name: f for f in cam1_files}
        cam2_dict = {f.name: f for f in cam2_files}
        common_names = sorted(set(cam1_dict.keys()) & set(cam2_dict.keys()))

        if not common_names:
            logger.error("No matching image pairs found between cameras")
            return

        logger.info(f"Found {len(common_names)} matching image pairs")

        # Process all pairs and collect successful ones
        objpoints = []
        imgpoints1 = []
        imgpoints2 = []
        successful_pairs = []
        objp = self.make_object_points()
        image_size = None

        for filename in common_names:
            cam1_file = cam1_dict[filename]
            cam2_file = cam2_dict[filename]

            # Process first image
            img1 = cv2.imread(str(cam1_file), cv2.IMREAD_UNCHANGED)
            if img1 is None:
                logger.warning(f"Could not load {cam1_file}")
                continue

            # Process second image
            img2 = cv2.imread(str(cam2_file), cv2.IMREAD_UNCHANGED)
            if img2 is None:
                logger.warning(f"Could not load {cam2_file}")
                continue

            # Set image size from first successful pair
            if image_size is None:
                image_size = img1.shape[:2][::-1]  # width, height

            # Detect grid in both images
            found1, grid1 = self.detect_grid_in_image(img1)
            found2, grid2 = self.detect_grid_in_image(img2)

            # Skip if grid not found in either image
            if not found1 or not found2:
                logger.warning(f"Grid not found in pair {filename}")
                continue

            # Verify that we have the correct number of points
            expected_points = self.pattern_size[0] * self.pattern_size[1]
            if len(grid1) != expected_points or len(grid2) != expected_points:
                logger.warning(f"Incorrect number of points detected in {filename}")
                continue

            # Add to calibration data
            objpoints.append(objp.reshape(-1, 1, 3))
            imgpoints1.append(grid1.reshape(-1, 1, 2))
            imgpoints2.append(grid2.reshape(-1, 1, 2))
            successful_pairs.append(filename)

            # Save individual results for this image pair (using 1-based indexing)
            img_index = len(successful_pairs)

            # Compute homographies for visualization
            objp_2d = objp[:, :2].astype(np.float32)
            if grid1 is not None and grid2 is not None:
                grid1_f32 = grid1.astype(np.float32)
                grid2_f32 = grid2.astype(np.float32)
                H1, _ = cv2.findHomography(grid1_f32, objp_2d, method=cv2.RANSAC)
                H2, _ = cv2.findHomography(grid2_f32, objp_2d, method=cv2.RANSAC)
            else:
                logger.warning(f"Grid points are None for {filename}")
                continue

            # Calculate individual reprojection errors
            def calc_reproj_error(grid_pts, H_matrix):
                objp_2d = objp[:, :2]
                H_inv = np.linalg.inv(H_matrix)
                objp_h = np.hstack([objp_2d, np.ones((objp_2d.shape[0], 1))])
                projected_h = (H_inv @ objp_h.T).T
                projected = projected_h[:, :2] / projected_h[:, 2:]
                error_vec = grid_pts - projected
                errors = np.linalg.norm(error_vec, axis=1)
                return errors.mean(), error_vec[:, 0], error_vec[:, 1]

            reproj_err1, reproj_x1, reproj_y1 = calc_reproj_error(grid1, H1)
            reproj_err2, reproj_x2, reproj_y2 = calc_reproj_error(grid2, H2)

            # Placeholder camera matrices (will be computed later)
            placeholder_matrix = np.eye(3, dtype=np.float32) * 1000
            placeholder_dist = np.zeros(5, dtype=np.float32)

            # Save individual results for both cameras
            self._save_individual_results(
                cam1_num,
                img_index,
                grid1,
                H1,
                placeholder_matrix,
                placeholder_dist,
                reproj_err1,
                reproj_x1,
                reproj_y1,
                filename,
            )
            self._save_individual_results(
                cam2_num,
                img_index,
                grid2,
                H2,
                placeholder_matrix,
                placeholder_dist,
                reproj_err2,
                reproj_x2,
                reproj_y2,
                filename,
            )

            logger.info(f"Successfully processed pair {filename}")

        # Check if we have enough pairs for calibration
        if len(successful_pairs) < 3:
            logger.error(
                f"Not enough successful image pairs for calibration: {len(successful_pairs)}"
            )
            return

        logger.info(f"Using {len(successful_pairs)} image pairs for calibration")

        # Calibrate each camera individually
        logger.info("Calibrating individual cameras...")
        ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(  # type: ignore
            objpoints, imgpoints1, image_size, None, None
        )
        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(  # type: ignore
            objpoints, imgpoints2, image_size, None, None
        )
        logger.info(f"Camera 1 reprojection error: {ret1:.5f}")
        logger.info(f"Camera 2 reprojection error: {ret2:.5f}")

        # Stereo calibration
        logger.info("Performing stereo calibration...")
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        flags = cv2.CALIB_FIX_INTRINSIC  # Use pre-calculated intrinsics

        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints1,
            imgpoints2,
            mtx1,
            dist1,
            mtx2,
            dist2,
            image_size,
            criteria=criteria,
            flags=flags,
        )

        # Stereo rectification with alpha=-1 and USE_INTRINSIC_GUESS
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            mtx1,
            dist1,
            mtx2,
            dist2,
            image_size,
            R,
            T,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
            alpha=-1,  # keep all pixels, don't crop
        )

        # Calculate relative angle
        angle_rad = math.acos((np.trace(R) - 1) / 2)
        angle_deg = np.degrees(angle_rad)

        # Save results
        self._save_stereo_results(
            cam1_num,
            cam2_num,
            {
                "camera_matrix_1": mtx1,
                "dist_coeffs_1": dist1,
                "camera_matrix_2": mtx2,
                "dist_coeffs_2": dist2,
                "rotation_matrix": R,
                "translation_vector": T,
                "essential_matrix": E,
                "fundamental_matrix": F,
                "rectification_R1": R1,
                "rectification_R2": R2,
                "projection_P1": P1,
                "projection_P2": P2,
                "disparity_to_depth_Q": Q,
                "valid_pixel_ROI1": validPixROI1,
                "valid_pixel_ROI2": validPixROI2,
                "stereo_reprojection_error": ret,
                "relative_angle_deg": angle_deg,
                "num_image_pairs": len(successful_pairs),
                "timestamp": datetime.now().isoformat(),
                "image_size": image_size,
                "successful_filenames": successful_pairs,
            },
        )

        logger.info(f"Stereo calibration completed successfully with error: {ret:.5f}")
        logger.info(f"Relative angle between cameras: {angle_deg:.2f} degrees")
        logger.info(f"Translation vector: {T.ravel()}")

    def _save_grid_visualization(
        self, cam_num, img_index, grid_points, original_filename, reprojection_error
    ):
        """Save a figure showing the detected grid with dot indices for a camera"""
        try:
            import matplotlib.pyplot as plt

            # Load original image for background
            cam_input_dir = self.source_dir / "calibration" / f"Cam{cam_num}"
            img_path = cam_input_dir / original_filename
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                logger.warning(f"Could not load image for visualization: {img_path}")
                return None

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
                f"Stereo Grid Detection - Cam{cam_num}: {original_filename}\n"
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
            stereo_dir = self.base_dir / "calibration" / f"Cam{cam_num}" / "stereo"
            stereo_dir.mkdir(parents=True, exist_ok=True)
            fig_path = stereo_dir / f"grid_detection_{img_index}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            logger.info(f"Saved stereo grid visualization: {fig_path}")
            return fig_path

        except Exception as e:
            logger.warning(f"Failed to save stereo grid visualization: {str(e)}")
            return None

    def _save_individual_results(
        self,
        cam_num,
        img_index,
        grid_points,
        H,
        camera_matrix,
        dist_coeffs,
        reprojection_error,
        reproj_errs_x,
        reproj_errs_y,
        original_filename,
    ):
        """Save individual camera calibration results for stereo"""
        stereo_dir = self.base_dir / "calibration" / f"Cam{cam_num}" / "stereo"
        stereo_dir.mkdir(parents=True, exist_ok=True)

        # Save grid data
        grid_data = {
            "grid_points": grid_points,
            "homography": H,
            "reprojection_error": reprojection_error,
            "reprojection_error_x_mean": float(np.mean(np.abs(reproj_errs_x))),
            "reprojection_error_y_mean": float(np.mean(np.abs(reproj_errs_y))),
            "reprojection_errors_x": reproj_errs_x,
            "reprojection_errors_y": reproj_errs_y,
            "original_filename": original_filename,
            "pattern_size": self.pattern_size,
            "dot_spacing_mm": self.dot_spacing_mm,
            "timestamp": datetime.now().isoformat(),
        }
        savemat(stereo_dir / f"grid_detection_{img_index}.mat", grid_data)

        # Save camera model
        model_data = {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "reprojection_error": reprojection_error,
            "reprojection_error_x_mean": float(np.mean(np.abs(reproj_errs_x))),
            "reprojection_error_y_mean": float(np.mean(np.abs(reproj_errs_y))),
            "grid_points": grid_points,
            "homography": H,
            "original_filename": original_filename,
            "pattern_size": self.pattern_size,
            "dot_spacing_mm": self.dot_spacing_mm,
            "timestamp": datetime.now().isoformat(),
        }
        savemat(stereo_dir / f"camera_model_{img_index}.mat", model_data)

        # Save visualization
        self._save_grid_visualization(
            cam_num, img_index, grid_points, original_filename, reprojection_error
        )

    def _save_stereo_results(self, cam1_num, cam2_num, stereo_data):
        """Save stereo calibration results"""
        stereo_filename = f"stereo_model_cam{cam1_num}-cam{cam2_num}.mat"

        # Save to both locations for compatibility
        savemat(self.base_dir / "calibration" / stereo_filename, stereo_data)

        # Also save to individual camera stereo directories
        stereo_dir1 = self.base_dir / "calibration" / f"Cam{cam1_num}" / "stereo"
        stereo_dir2 = self.base_dir / "calibration" / f"Cam{cam2_num}" / "stereo"
        stereo_dir1.mkdir(parents=True, exist_ok=True)
        stereo_dir2.mkdir(parents=True, exist_ok=True)

        savemat(stereo_dir1 / stereo_filename, stereo_data)
        savemat(stereo_dir2 / stereo_filename, stereo_data)

        logger.info(
            f"Saved stereo model: {self.base_dir / 'calibration' / stereo_filename}"
        )
        logger.info(
            f"Stereo calibration results saved at: {self.base_dir / 'calibration' / stereo_filename}"
        )

    def run(self):
        """Run stereo calibration for all camera pairs"""
        logger.info("Starting stereo calibration")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.base_dir}")
        logger.info(f"Camera pairs: {self.camera_pairs}")

        for cam1_num, cam2_num in self.camera_pairs:
            try:
                self.process_camera_pair(cam1_num, cam2_num)
            except Exception as e:
                logger.error(
                    f"Failed to process camera pair {cam1_num}-{cam2_num}: {str(e)}"
                )
                continue

        logger.info("Stereo calibration completed")


def main():
    calibrator = StereoCalibrator(
        source_dir=SOURCE_DIR,
        base_dir=BASE_DIR,
        camera_pairs=CAMERA_PAIRS,
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
