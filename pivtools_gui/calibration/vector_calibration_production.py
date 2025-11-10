#!/usr/bin/env python3
"""
vector_calibration_production.py

Production script for calibrating uncalibrated PIV vectors to physical units (m/s).
Converts pixel-based vectors to physical velocities using camera calibration models.
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat, savemat

sys.path.append(str(Path(__file__).parent.parent))
from ..paths import get_data_paths
from ..vector_loading import load_coords_from_directory, read_mat_contents

# ===================== CONFIGURATION VARIABLES =====================
# Set these variables for your calibration setup
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Planar_Images_with_wall/test"
image_count = 1000
DT_SECONDS = 1  # Time step between frames in seconds
CAMERA_NUM = 1  # Camera number (1-based)
MODEL_INDEX = 0  # Index of calibration model to use (0-based)
DOT_SPACING_MM = 28.89  # Physical spacing between calibration dots in mm
VECTOR_PATTERN = "%05d.mat"  # Pattern for vector files (e.g. "B%05d.mat", "%05d.mat")
TYPE_NAME = "instantaneous"  # Type name for uncalibrated data directory (e.g. "Instantaneous", "piv")
# Example: RUNS_TO_PROCESS = [1, 2, 3]  # Process only runs 1, 2, and 3
# ===================================================================

# Add src to path to import modules


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VectorCalibrator:
    def __init__(
        self,
        base_dir,
        camera_num,
        model_index,
        dt,
        dot_spacing_mm=28.89,
        vector_pattern="%05d.mat",
        type_name="Instantaneous",
    ):
        """
        Initialize vector calibrator

        Args:
            base_dir: Base directory containing data
            camera_num: Camera number (1-based)
            model_index: Index of calibration model to use (0-based)
            dt: Time step between frames in seconds
            dot_spacing_mm: Physical spacing of calibration dots in mm
            vector_pattern: Pattern for vector files (e.g. "B%05d.mat", "%05d.mat")
            type_name: Type name for uncalibrated data directory (e.g. "Instantaneous", "piv")
        """
        self.base_dir = Path(base_dir)
        self.camera_num = camera_num
        self.model_index = model_index
        self.dt = dt
        self.dot_spacing_mm = dot_spacing_mm
        self.vector_pattern = vector_pattern
        self.type_name = type_name

        # Load calibration model
        self.calibration_model = self._load_calibration_model()
        self.homography = self.calibration_model["homography"]
        self.camera_matrix = self.calibration_model["camera_matrix"]
        self.dist_coeffs = self.calibration_model["dist_coeffs"]
        self.dot_spacing_mm = dot_spacing_mm

        logger.info(f"Initialized calibrator for Camera {camera_num}")
        logger.info(f"Using calibration model index {model_index}")
        logger.info(f"Time step: {dt} seconds")
        logger.info(f"Dot spacing: {dot_spacing_mm} mm")
        logger.info(f"Vector pattern: {vector_pattern}")
        logger.info(f"Type name: {type_name}")

    def _load_calibration_model(self):
        """Load the specified calibration model"""
        calib_paths = get_data_paths(
            self.base_dir,
            num_images=1,  # Not used for calibration paths
            cam=self.camera_num,
            type_name="",  # Not used for calibration paths
            calibration=True,
        )

        calib_dir = calib_paths["calib_dir"]
        # Try common model filenames produced by planar calibrator
        # Always look for the model at "model/camera_model.mat"
        model_path = calib_dir / "model" / "camera_model.mat"
        if not model_path.exists():
            raise FileNotFoundError(f"Calibration model not found: {model_path}")

        logger.info(f"Loading calibration model: {model_path}")
        model_data = loadmat(str(model_path), squeeze_me=True, struct_as_record=False)

        required_fields = ["homography", "camera_matrix", "dist_coeffs"]
        missing_fields = [field for field in required_fields if field not in model_data]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in calibration model: {missing_fields}"
            )

        # Ensure homography is 3x3
        homography = np.array(model_data["homography"])
        if homography.shape != (3, 3):
            raise ValueError(f"Homography must be 3x3, got shape {homography.shape}")

        # Ensure dist_coeffs is 1D array
        dist_coeffs = np.array(model_data["dist_coeffs"]).flatten()

        model_data["homography"] = homography.astype(np.float32)
        model_data["dist_coeffs"] = dist_coeffs.astype(np.float32)

        # Use dt from model if available, otherwise use instance dt
        if "dt" in model_data:
            logger.info(f"Using dt from calibration model: {model_data['dt']} seconds")
            self.dt = float(model_data["dt"])
        else:
            logger.info(
                f"No dt in calibration model, using provided dt: {self.dt} seconds"
            )

        return model_data

    def calibrate_coordinates(self, coords_x, coords_y):
        """
        Convert pixel coordinates to physical coordinates in mm using homography and distortion correction.

        Args:
            coords_x, coords_y: Coordinate arrays in pixels

        Returns:
            (x_mm, y_mm): Coordinate arrays in mm
        """
        # Stack coordinates for transformation
        pts = np.stack([coords_x.flatten(), coords_y.flatten()], axis=-1).astype(
            np.float32
        )

        # Ensure we have valid points
        if pts.size == 0:
            return coords_x, coords_y

        # Undistort points if distortion coefficients are present
        if np.any(self.dist_coeffs):
            # Reshape for cv2.undistortPoints: (N, 1, 2)
            pts_reshaped = pts.reshape(-1, 1, 2)
            try:
                pts_ud = cv2.undistortPoints(
                    pts_reshaped,
                    self.camera_matrix,
                    self.dist_coeffs,
                    P=self.camera_matrix,
                )
                pts_ud = pts_ud.reshape(-1, 2)
            except cv2.error as e:
                logger.warning(f"Undistortion failed, using original points: {e}")
                pts_ud = pts
        else:
            pts_ud = pts

        # Apply homography using OpenCV for numerical stability
        pts_ud = pts_ud.reshape(1, -1, 2)  # shape (1, N, 2) for perspectiveTransform
        pts_mapped = cv2.perspectiveTransform(pts_ud, self.homography)[0]

        # Reshape back to original shape
        x_mm = pts_mapped[:, 0].reshape(coords_x.shape)
        y_mm = pts_mapped[:, 1].reshape(coords_y.shape)

        logger.info(
            "Converted coordinates from pixels to mm (undistorted + homography)"
        )

        return x_mm, y_mm

    def calibrate_vectors(self, ux_px, uy_px, coords_x_px, coords_y_px):
        """
        Convert pixel-based velocity vectors to m/s using camera matrix (pinhole model) and distortion correction.

        Args:
            ux_px, uy_px: Velocity components in pixels/frame
            coords_x_px, coords_y_px: Grid coordinates in pixels

        Returns:
            (ux_ms, uy_ms): Velocity components in m/s
        """
        # Stack coordinates
        coords_px = np.stack([coords_x_px, coords_y_px], axis=-1).astype(np.float32)
        shape = coords_px.shape[:-1]
        coords_flat = coords_px.reshape(-1, 2)

        # Check for valid data
        if coords_flat.size == 0 or ux_px.size == 0 or uy_px.size == 0:
            logger.warning("Empty coordinate or vector data, returning zeros")
            return np.zeros_like(ux_px), np.zeros_like(uy_px)

        # Ensure arrays have compatible shapes
        if ux_px.shape != uy_px.shape or ux_px.shape != coords_x_px.shape:
            logger.error(
                f"Shape mismatch: ux_px={ux_px.shape}, uy_px={uy_px.shape}, coords_x_px={coords_x_px.shape}"
            )
            return np.zeros_like(ux_px), np.zeros_like(uy_px)

        # Undistort grid points
        if np.any(self.dist_coeffs):
            # Reshape for cv2.undistortPoints: (N, 1, 2)
            coords_reshaped = coords_flat.reshape(-1, 1, 2)
            try:
                coords_ud = cv2.undistortPoints(
                    coords_reshaped,
                    self.camera_matrix,
                    self.dist_coeffs,
                    P=self.camera_matrix,
                )
                coords_ud = coords_ud.reshape(-1, 2)
            except cv2.error as e:
                logger.warning(f"Grid undistortion failed, using original points: {e}")
                coords_ud = coords_flat
        else:
            coords_ud = coords_flat

        # Apply homography
        coords_ud = coords_ud.reshape(1, -1, 2)
        coords_mm = cv2.perspectiveTransform(coords_ud, self.homography)[0]

        # Displaced points
        disp_px = coords_flat + np.stack([ux_px.flatten(), uy_px.flatten()], axis=-1)

        if np.any(self.dist_coeffs):
            # Reshape for cv2.undistortPoints: (N, 1, 2)
            disp_reshaped = disp_px.reshape(-1, 1, 2)
            try:
                disp_ud = cv2.undistortPoints(
                    disp_reshaped,
                    self.camera_matrix,
                    self.dist_coeffs,
                    P=self.camera_matrix,
                )
                disp_ud = disp_ud.reshape(-1, 2)
            except cv2.error as e:
                logger.warning(
                    f"Displacement undistortion failed, using original points: {e}"
                )
                disp_ud = disp_px
        else:
            disp_ud = disp_px

        disp_ud = disp_ud.reshape(1, -1, 2)
        disp_mm = cv2.perspectiveTransform(disp_ud, self.homography)[0]

        # Metric displacement
        delta_mm = disp_mm - coords_mm

        # Convert to m/s
        ux_ms = (delta_mm[:, 0] / 1000.0) / self.dt  # mm to m, frame to s
        uy_ms = (delta_mm[:, 1] / 1000.0) / self.dt

        ux_ms = ux_ms.reshape(shape)
        uy_ms = uy_ms.reshape(shape)

        logger.info(
            "Converted vectors from pixels/frame to m/s using undistortion + homography"
        )

        return ux_ms, uy_ms

    def process_run(self, image_count, progress_cb=None):
        """
        Process and calibrate vectors for all available runs

        Args:
            image_count: Number of images in the run
            progress_cb: Optional callback for progress updates
        """
        logger.info(f"Processing run with {image_count} images")

        # Get data paths for uncalibrated data - use configured type
        paths = get_data_paths(
            self.base_dir,
            num_images=image_count,
            cam=self.camera_num,
            type_name=self.type_name,
            use_uncalibrated=True,
        )

        uncalib_data_dir = paths["data_dir"]
        logger.info(f"Uncalibrated data directory: {uncalib_data_dir}")

        # Get output paths for calibrated data
        calib_paths = get_data_paths(
            self.base_dir,
            num_images=image_count,
            cam=self.camera_num,
            type_name=self.type_name,
        )

        calib_data_dir = calib_paths["data_dir"]
        calib_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Calibrated data directory: {calib_data_dir}")

        if not uncalib_data_dir.exists():
            raise FileNotFoundError(
                f"Uncalibrated data directory not found: {uncalib_data_dir}"
            )

        # Load coordinates for all runs
        logger.info("Loading coordinates...")
        x_coords_list, y_coords_list = load_coords_from_directory(
            uncalib_data_dir, runs=None  # Load all available runs
        )

        if not x_coords_list:
            logger.error("No coordinate data found!")
            raise ValueError("No coordinate data found")

        logger.info(f"Loaded coordinates for {len(x_coords_list)} runs")

        # Find runs with valid data
        valid_runs = []
        for i, (x_coords, y_coords) in enumerate(zip(x_coords_list, y_coords_list)):
            run_num = i + 1
            # Ensure None is replaced with empty arrays
            if x_coords is None:
                x_coords = np.array([])
            if y_coords is None:
                y_coords = np.array([])
            valid_coords = np.sum(~np.isnan(x_coords)) + np.sum(~np.isnan(y_coords))
            logger.info(f"Run {run_num}: {valid_coords} valid coordinates")
            if valid_coords > 0:
                valid_runs.append((i, run_num, valid_coords))

        if not valid_runs:
            raise ValueError("No runs with valid coordinate data found")

        logger.info(
            f"Found {len(valid_runs)} runs with valid data: {[r[1] for r in valid_runs]}"
        )

        # Create coordinate structure
        max_run = max([r[1] for r in valid_runs])
        coord_dtype = np.dtype([("x", "O"), ("y", "O")])
        coordinates = np.empty(max_run, dtype=coord_dtype)

        # Process each valid run
        for run_idx, run_num, valid_coord_count in valid_runs:
            logger.info(
                f"Processing run {run_num} with {valid_coord_count} valid coordinates"
            )
            x_coords_px = x_coords_list[run_idx]
            y_coords_px = y_coords_list[run_idx]

            # Calibrate coordinates to mm
            x_coords_mm, y_coords_mm = self.calibrate_coordinates(
                x_coords_px, y_coords_px
            )
            coordinates[run_num - 1] = (x_coords_mm, y_coords_mm)

        # Fill all runs: valid runs get data, others get empty arrays
        valid_run_indices = set(r[1] for r in valid_runs)
        for run_num in range(1, max_run + 1):
            if run_num in valid_run_indices:
                # Already set above for valid runs
                continue
            coordinates[run_num - 1] = (np.array([]), np.array([]))

        coords_output = {"coordinates": coordinates}

        # Save calibrated coordinates
        calibrated_dir = (
            self.base_dir
            / "calibrated_piv"
            / str(image_count)
            / f"Cam{self.camera_num}"
            / self.type_name
        )
        calibrated_dir.mkdir(parents=True, exist_ok=True)
        coords_path = calibrated_dir / "coordinates.mat"
        savemat(str(coords_path), coords_output)
        logger.info(f"Saved calibrated coordinates: {coords_path}")

        # Process vector files using the first valid run's coordinates
        if valid_runs:
            first_run_idx = valid_runs[0][0]
            x_coords_for_vectors = x_coords_list[first_run_idx]
            y_coords_for_vectors = y_coords_list[first_run_idx]

            self._process_vector_files(
                uncalib_data_dir,
                calib_data_dir,
                image_count,
                x_coords_for_vectors,
                y_coords_for_vectors,
                max_run,
                valid_runs,
                progress_cb,
            )
        else:
            logger.error("No valid runs found for vector processing")

    def _process_vector_files(
        self,
        uncalib_dir,
        calib_dir,
        num_images,
        coords_x_px,
        coords_y_px,
        max_run,
        valid_runs,
        progress_cb,
    ):
        """Process all vector files in the directory"""
        logger.info("Processing vector files...")

        # Use configured vector pattern
        vector_pattern = self.vector_pattern

        processed_vectors = []

        for i in range(1, num_images + 1):
            vector_file = uncalib_dir / (vector_pattern % i)

            if not vector_file.exists():
                if i <= 5:  # Only log first few missing files
                    logger.warning(f"Vector file not found: {vector_file}")
                continue

            try:
                # Load uncalibrated vectors
                vector_data = read_mat_contents(str(vector_file))  # Shape varies

                # Handle different vector data formats
                if vector_data.ndim == 4 and vector_data.shape[0] == 1:
                    # Single run format: (1, 3, H, W)
                    ux_px = vector_data[0, 0, :, :]
                    uy_px = vector_data[0, 1, :, :]
                    b_mask = vector_data[0, 2, :, :]
                elif vector_data.ndim == 3 and vector_data.shape[0] == 3:
                    # Single run format: (3, H, W)
                    ux_px = vector_data[0, :, :]
                    uy_px = vector_data[1, :, :]
                    b_mask = vector_data[2, :, :]
                else:
                    logger.warning(
                        f"Unexpected vector data shape in {vector_file.name}: {vector_data.shape}"
                    )
                    continue

                # Calibrate vectors
                ux_ms, uy_ms = self.calibrate_vectors(
                    ux_px, uy_px, coords_x_px, coords_y_px
                )

                # Create piv_result structure array with proper MATLAB struct format
                piv_dtype = np.dtype([("ux", "O"), ("uy", "O"), ("b_mask", "O")])
                piv_result = np.empty(max_run, dtype=piv_dtype)

                for run_num in range(1, max_run + 1):
                    run_idx = run_num - 1  # Convert to 0-based

                    # Check if this run has valid data
                    run_has_data = any(r[1] == run_num for r in valid_runs)

                    if run_has_data:
                        # This creates piv_result(run_idx).ux, piv_result(run_idx).uy, piv_result(run_idx).b_mask
                        piv_result[run_idx] = (ux_ms, uy_ms, b_mask)
                    else:
                        # Empty struct for runs not being processed
                        piv_result[run_idx] = (np.array([]), np.array([]), np.array([]))

                # Save calibrated piv_result into calibrated_piv output tree
                output_file = calib_dir / (self.vector_pattern % i)
                savemat(str(output_file), {"piv_result": piv_result})

                processed_vectors.append(
                    {"ux_ms": ux_ms, "uy_ms": uy_ms, "b_mask": b_mask, "frame": i}
                )

                # Progress callback
                if progress_cb:
                    progress = (i / num_images) * 100
                    progress_cb(
                        {
                            "processed_frames": i,
                            "total_frames": num_images,
                            "progress": progress,
                            "successful_frames": len(processed_vectors),
                        }
                    )

            except Exception as e:
                logger.error(f"Failed to process {vector_file.name}: {str(e)}")
                continue

        logger.info(
            f"Successfully processed {len(processed_vectors)} vector files into {calib_dir}"
        )


def main():
    logger.info("Starting vector calibration with configuration:")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Run number: {image_count}")
    logger.info(f"Time step: {DT_SECONDS} seconds")
    logger.info(f"Camera: {CAMERA_NUM}")
    logger.info(f"Model index: {MODEL_INDEX}")
    logger.info(f"Dot spacing: {DOT_SPACING_MM} mm")
    logger.info(f"Vector pattern: {VECTOR_PATTERN}")
    logger.info(f"Type name: {TYPE_NAME}")

    try:
        calibrator = VectorCalibrator(
            base_dir=BASE_DIR,
            camera_num=CAMERA_NUM,
            model_index=MODEL_INDEX,
            dt=DT_SECONDS,
            dot_spacing_mm=DOT_SPACING_MM,
            vector_pattern=VECTOR_PATTERN,
            type_name=TYPE_NAME,
        )

        calibrator.process_run(image_count)

        logger.info("Vector calibration completed successfully")

    except Exception as e:
        logger.error(f"Calibration failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
