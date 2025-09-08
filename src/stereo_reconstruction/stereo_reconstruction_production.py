#!/usr/bin/env python3
"""
stereo_reconstruction_production.py

Production script for 3D velocity reconstruction from stereo camera pairs.
Takes calibrated 2D velocity fields from two cameras and reconstructs 3D velocities (ux, uy, uz).
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat, savemat

from paths import get_data_paths

# ===================== CONFIGURATION VARIABLES =====================
# Set these variables for your stereo reconstruction setup
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Stereo_Images/ProcessedPIV"
CAMERA_PAIRS = [[1, 2]]  # Array of camera pairs to process
image_count = 1000
VECTOR_PATTERN = "%05d.mat"  # Pattern for vector files
TYPE_NAME = "instantaneous"  # Type name for calibrated data directory
MAX_CORRESPONDENCE_DISTANCE = 5.0  # Maximum distance in mm for point correspondence
MIN_TRIANGULATION_ANGLE = 5.0  # Minimum angle in degrees for triangulation
RUNS_TO_PROCESS = [6]  # List of run numbers to process (1-based)
# ===================================================================

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StereoReconstructor:
    def __init__(
        self,
        base_dir,
        camera_pairs,
        image_count,
        vector_pattern="%05d.mat",
        type_name="instantaneous",
        max_distance=5.0,
        min_angle=5.0,
    ):
        """
        Initialize stereo reconstructor

        Args:
            base_dir: Base directory containing calibrated data
            camera_pairs: List of camera pairs [[cam1, cam2], ...]
            image_count: Number of images to process
            vector_pattern: Pattern for vector files
            type_name: Type name for data directory
            max_distance: Maximum distance in mm for point correspondence
            min_angle: Minimum triangulation angle in degrees
        """
        self.base_dir = Path(base_dir)
        self.camera_pairs = camera_pairs
        self.image_count = image_count
        self.vector_pattern = vector_pattern
        self.type_name = type_name
        self.max_distance = max_distance
        self.min_angle = min_angle

        logger.info("Initialized stereo reconstructor")
        logger.info(f"Base directory: {base_dir}")
        logger.info(f"Camera pairs: {camera_pairs}")
        logger.info(f"Image count: {image_count}")
        logger.info(f"Max correspondence distance: {max_distance} mm")
        logger.info(f"Min triangulation angle: {min_angle} degrees")

    def load_stereo_calibration(self, cam1_num, cam2_num):
        """Load stereo calibration data for camera pair"""
        cam1_stereo_dir = self.base_dir / "calibration" / f"Cam{cam1_num}" / "stereo"
        stereo_file = cam1_stereo_dir / f"stereo_cam{cam1_num}_cam{cam2_num}.mat"

        if not stereo_file.exists():
            raise FileNotFoundError(f"Stereo calibration not found: {stereo_file}")

        logger.info(f"Loading stereo calibration: {stereo_file}")
        stereo_data = loadmat(str(stereo_file), squeeze_me=True, struct_as_record=False)

        required_fields = [
            "camera_matrix_1",
            "camera_matrix_2",
            "dist_coeffs_1",
            "dist_coeffs_2",
            "rotation_matrix",
            "translation_vector",
            "projection_P1",
            "projection_P2",
            "disparity_to_depth_Q",
        ]

        missing_fields = [
            field for field in required_fields if field not in stereo_data
        ]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in stereo calibration: {missing_fields}"
            )

        return stereo_data

    def load_calibrated_coordinates(self, cam_num, runs_to_process=None):
        """Load calibrated coordinates for a camera"""
        # Load from uncalibrated directory to get pixel coordinates
        paths = get_data_paths(
            self.base_dir,
            num_images=self.image_count,
            cam=cam_num,
            type_name=self.type_name,
            use_uncalibrated=True,  # This gets us the pixel coordinates
        )

        coords_file = paths["data_dir"] / "coordinates.mat"
        if not coords_file.exists():
            raise FileNotFoundError(
                f"Uncalibrated coordinates not found: {coords_file}"
            )

        logger.info(
            f"Loading uncalibrated coordinates for Camera {cam_num}: {coords_file}"
        )
        coords_data = loadmat(str(coords_file), squeeze_me=True, struct_as_record=False)

        # Extract coordinates for specified runs
        coordinates = coords_data["coordinates"]

        # Handle both single coordinate set and list of coordinates
        if not isinstance(coordinates, (list, np.ndarray)):
            # Single coordinate set - wrap in list
            coordinates = [coordinates]
        elif isinstance(coordinates, np.ndarray) and coordinates.ndim == 0:
            # Single object in ndarray - extract it
            coordinates = [coordinates.item()]

        filtered_coords = []
        for coord_data in coordinates:
            # Handle both structured and simple coordinate data
            if hasattr(coord_data, "run"):
                run_num = coord_data.run
            else:
                run_num = 1  # Default run number

            if runs_to_process is None or run_num in runs_to_process:
                # Extract coordinate arrays
                if hasattr(coord_data, "x_px"):
                    x_px = coord_data.x_px
                    y_px = coord_data.y_px
                elif hasattr(coord_data, "x_coords"):
                    x_px = coord_data.x_coords
                    y_px = coord_data.y_coords
                else:
                    # Try to find coordinate data in the structure
                    logger.warning(
                        f"Unexpected coordinate structure: {type(coord_data)}"
                    )
                    # Try to access as dict-like
                    if hasattr(coord_data, "__dict__"):
                        attrs = coord_data.__dict__
                        x_px = attrs.get("x_px", attrs.get("x_coords", None))
                        y_px = attrs.get("y_px", attrs.get("y_coords", None))
                    else:
                        continue

                if x_px is not None and y_px is not None:
                    filtered_coords.append({"x_px": x_px, "y_px": y_px, "run": run_num})

        if not filtered_coords:
            # If no structured coordinates found, try direct access
            logger.warning("No structured coordinates found, trying direct access")
            if "x_coords" in coords_data and "y_coords" in coords_data:
                filtered_coords = [
                    {
                        "x_px": coords_data["x_coords"],
                        "y_px": coords_data["y_coords"],
                        "run": 1,
                    }
                ]
            elif "x_px" in coords_data and "y_px" in coords_data:
                filtered_coords = [
                    {"x_px": coords_data["x_px"], "y_px": coords_data["y_px"], "run": 1}
                ]
            else:
                # Log available keys for debugging
                logger.error(
                    f"Could not find coordinate data. Available keys: {list(coords_data.keys())}"
                )
                raise ValueError(f"No valid coordinate data found in {coords_file}")

        logger.info(
            f"Loaded {len(filtered_coords)} coordinate sets for Camera {cam_num}"
        )
        return filtered_coords

    def load_calibrated_vectors(self, cam_num, frame_idx):
        """Load calibrated vector data for a specific frame"""
        paths = get_data_paths(
            self.base_dir,
            num_images=self.image_count,
            cam=cam_num,
            type_name=self.type_name,
        )

        vector_file = paths["data_dir"] / (self.vector_pattern % frame_idx)
        if not vector_file.exists():
            raise FileNotFoundError(f"Vector file not found: {vector_file}")

        vector_data = loadmat(str(vector_file), squeeze_me=True, struct_as_record=False)

        if "piv_result" in vector_data:
            piv_data = vector_data["piv_result"]
            return {
                "ux_ms": piv_data.ux_ms,
                "uy_ms": piv_data.uy_ms,
                "b_mask": piv_data.b_mask,
                "frame": piv_data.frame,
            }
        else:
            # Fallback for direct format
            return {
                "ux_ms": vector_data["ux_ms"],
                "uy_ms": vector_data["uy_ms"],
                "b_mask": vector_data["b_mask"],
                "frame": frame_idx,
            }

    def find_corresponding_points(self, coords1_px, coords2_px):
        """
        Find corresponding points between two camera views using grid structure

        Args:
            coords1_px: (x_px, y_px) coordinates from camera 1 in pixels
            coords2_px: (x_px, y_px) coordinates from camera 2 in pixels

        Returns:
            (indices1, indices2): Arrays of corresponding indices
        """
        # For PIV grids, points should correspond directly by grid position
        # Assuming both cameras have the same grid structure
        shape1 = coords1_px[0].shape
        shape2 = coords2_px[0].shape

        if shape1 != shape2:
            logger.warning(f"Grid shapes don't match: {shape1} vs {shape2}")
            # Use minimum dimensions
            min_h = min(shape1[0], shape2[0])
            min_w = min(shape1[1], shape2[1])

            # Create indices for overlapping region
            indices1 = []
            indices2 = []
            for i in range(min_h):
                for j in range(min_w):
                    idx1 = np.ravel_multi_index((i, j), shape1)
                    idx2 = np.ravel_multi_index((i, j), shape2)
                    indices1.append(idx1)
                    indices2.append(idx2)

            indices1 = np.array(indices1)
            indices2 = np.array(indices2)
        else:
            # Perfect grid correspondence - all points correspond
            total_points = np.prod(shape1)
            indices1 = np.arange(total_points)
            indices2 = np.arange(total_points)

        logger.info(f"Found {len(indices1)} corresponding grid points")

        return indices1, indices2

    def triangulate_3d_points(self, pts1_px, pts2_px, stereo_data):
        """
        Triangulate 3D points from corresponding 2D points

        Args:
            pts1_px: 2D points from camera 1 in pixels (Nx2)
            pts2_px: 2D points from camera 2 in pixels (Nx2)
            stereo_data: Stereo calibration data

        Returns:
            3D points in mm (Nx3)
        """
        # Undistort points
        pts1_undist = cv2.undistortPoints(
            pts1_px.reshape(-1, 1, 2).astype(np.float32),
            stereo_data["camera_matrix_1"],
            stereo_data["dist_coeffs_1"],
        ).reshape(-1, 2)

        pts2_undist = cv2.undistortPoints(
            pts2_px.reshape(-1, 1, 2).astype(np.float32),
            stereo_data["camera_matrix_2"],
            stereo_data["dist_coeffs_2"],
        ).reshape(-1, 2)

        # Triangulate points
        points_4d = cv2.triangulatePoints(
            stereo_data["projection_P1"],
            stereo_data["projection_P2"],
            pts1_undist.T,
            pts2_undist.T,
        )

        # Convert from homogeneous to 3D coordinates
        points_3d = points_4d[:3] / points_4d[3]

        return points_3d.T  # Shape: (N, 3)

    def compute_triangulation_angles(self, pts_3d, stereo_data):
        """
        Compute triangulation angles to filter unreliable reconstructions

        Args:
            pts_3d: 3D points (Nx3)
            stereo_data: Stereo calibration data

        Returns:
            Array of angles in degrees
        """
        # Camera centers (simplified - assumes cameras at origin after rectification)
        cam1_center = np.array([0, 0, 0])
        cam2_center = stereo_data["translation_vector"]

        # Vectors from cameras to 3D points
        vec1 = pts_3d - cam1_center
        vec2 = pts_3d - cam2_center

        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
        vec2_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)

        # Compute angles
        dot_products = np.sum(vec1_norm * vec2_norm, axis=1)
        angles_rad = np.arccos(np.clip(dot_products, -1, 1))
        angles_deg = np.degrees(angles_rad)

        return angles_deg

    def reconstruct_3d_velocities(
        self, ux1, uy1, ux2, uy2, coords1_px, coords2_px, stereo_data
    ):
        """
        Reconstruct 3D velocities from corresponding 2D velocity fields

        Args:
            ux1, uy1: 2D velocity components from camera 1 (m/s)
            ux2, uy2: 2D velocity components from camera 2 (m/s)
            coords1_px, coords2_px: Pixel coordinates from both cameras
            stereo_data: Stereo calibration data

        Returns:
            Dictionary with 3D reconstruction results
        """
        logger.info("Starting 3D velocity reconstruction")

        # Find corresponding points using grid structure
        indices1, indices2 = self.find_corresponding_points(coords1_px, coords2_px)

        if len(indices1) == 0:
            raise ValueError("No corresponding points found between cameras")

        # Get corresponding coordinates and velocities
        # Convert flat indices to 2D indices
        shape1 = coords1_px[0].shape
        shape2 = coords2_px[0].shape
        row1, col1 = np.unravel_index(indices1, shape1)
        row2, col2 = np.unravel_index(indices2, shape2)

        # Extract corresponding data
        pts1_px = np.column_stack(
            [coords1_px[0][row1, col1], coords1_px[1][row1, col1]]
        )
        pts2_px = np.column_stack(
            [coords2_px[0][row2, col2], coords2_px[1][row2, col2]]
        )

        vel1 = np.column_stack([ux1[row1, col1], uy1[row1, col1]])
        vel2 = np.column_stack([ux2[row2, col2], uy2[row2, col2]])

        # Triangulate 3D positions
        pts_3d = self.triangulate_3d_points(pts1_px, pts2_px, stereo_data)

        # Compute triangulation angles for quality filtering
        angles = self.compute_triangulation_angles(pts_3d, stereo_data)
        angle_mask = angles > self.min_angle

        # For 3D velocity reconstruction, we need to convert velocity to pixel displacement
        # Assume dt=1 for simplicity (velocities are already in m/s)
        # Convert velocity from m/s to pixels/frame using a scale factor
        # This is approximate - ideally we'd use the local Jacobian of the camera projection

        # Rough conversion: assume 1 m/s ≈ some pixels/frame
        # We'll use a small displacement approach
        displacement_scale = 1.0  # This may need tuning based on your setup

        # Create displaced points by adding small pixel displacements
        # Convert m/s to pixel displacement (this is an approximation)
        pts1_displaced_px = pts1_px + vel1 * displacement_scale
        pts2_displaced_px = pts2_px + vel2 * displacement_scale

        # Triangulate displaced positions
        pts_3d_displaced = self.triangulate_3d_points(
            pts1_displaced_px, pts2_displaced_px, stereo_data
        )

        # Compute 3D velocity as difference
        vel_3d = pts_3d_displaced - pts_3d  # This gives change in mm
        # Convert to m/s (assuming unit time step)
        vel_3d = vel_3d / 1000.0  # Convert mm to m

        # Create output grids (sparse representation)
        valid_mask = angle_mask

        logger.info(f"Reconstructed {np.sum(valid_mask)} valid 3D velocity vectors")

        return {
            "velocities_3d": vel_3d[valid_mask],  # (N, 3) array
            "positions_3d": pts_3d[valid_mask],  # (N, 3) array
            "indices1": indices1[valid_mask],  # Original indices in camera 1
            "indices2": indices2[valid_mask],  # Original indices in camera 2
            "triangulation_angles": angles[valid_mask],
            "num_valid": np.sum(valid_mask),
            "num_total": len(valid_mask),
        }

    def process_camera_pair(self, cam1_num, cam2_num, runs_to_process=None):
        """
        Process a camera pair for 3D reconstruction

        Args:
            cam1_num: First camera number
            cam2_num: Second camera number
            runs_to_process: List of run numbers to process
        """
        logger.info(f"Processing camera pair {cam1_num}-{cam2_num}")

        # Load stereo calibration
        stereo_data = self.load_stereo_calibration(cam1_num, cam2_num)

        # Load uncalibrated coordinates for both cameras (pixel coordinates)
        coords1_list = self.load_calibrated_coordinates(cam1_num, runs_to_process)
        coords2_list = self.load_calibrated_coordinates(cam2_num, runs_to_process)

        if len(coords1_list) == 0:
            raise ValueError(f"No coordinate data found for Camera {cam1_num}")
        if len(coords2_list) == 0:
            raise ValueError(f"No coordinate data found for Camera {cam2_num}")

        if len(coords1_list) != len(coords2_list):
            logger.warning(
                f"Mismatched number of coordinate sets: {len(coords1_list)} vs {len(coords2_list)}"
            )
            # Use minimum number
            min_sets = min(len(coords1_list), len(coords2_list))
            coords1_list = coords1_list[:min_sets]
            coords2_list = coords2_list[:min_sets]

        # Use first coordinate set for reconstruction (assuming same grid for all runs)
        coords1 = coords1_list[0]
        coords2 = coords2_list[0]

        coords1_px = (coords1["x_px"], coords1["y_px"])
        coords2_px = (coords2["x_px"], coords2["y_px"])

        logger.info(
            f"Using coordinate grids: Camera 1 shape {coords1_px[0].shape}, Camera 2 shape {coords2_px[0].shape}"
        )

        # Create output directory
        output_dir = (
            self.base_dir
            / f"stereo_reconstruction_cam{cam1_num}_cam{cam2_num}"
            / self.type_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each frame
        reconstructed_frames = []

        for frame_idx in range(1, self.image_count + 1):
            try:
                # Load vector data for both cameras
                vectors1 = self.load_calibrated_vectors(cam1_num, frame_idx)
                vectors2 = self.load_calibrated_vectors(cam2_num, frame_idx)

                # Reconstruct 3D velocities
                result_3d = self.reconstruct_3d_velocities(
                    vectors1["ux_ms"],
                    vectors1["uy_ms"],
                    vectors2["ux_ms"],
                    vectors2["uy_ms"],
                    coords1_px,
                    coords2_px,
                    stereo_data,
                )

                # Save individual frame
                frame_data = {
                    "velocities_3d": result_3d[
                        "velocities_3d"
                    ],  # (N, 3) - ux, uy, uz in m/s
                    "positions_3d": result_3d["positions_3d"],  # (N, 3) - x, y, z in mm
                    "indices_cam1": result_3d["indices1"],
                    "indices_cam2": result_3d["indices2"],
                    "triangulation_angles": result_3d["triangulation_angles"],
                    "num_valid_vectors": result_3d["num_valid"],
                    "frame": frame_idx,
                    "reconstruction_info": {
                        "camera_pair": [cam1_num, cam2_num],
                        "max_correspondence_distance": self.max_distance,
                        "min_triangulation_angle": self.min_angle,
                        "timestamp": datetime.now().isoformat(),
                    },
                }

                # Save frame
                frame_file = output_dir / (self.vector_pattern % frame_idx)
                savemat(str(frame_file), frame_data)

                reconstructed_frames.append(frame_data)

                logger.info(
                    f"Frame {frame_idx}: {result_3d['num_valid']}/{result_3d['num_total']} valid vectors"
                )

            except Exception as e:
                logger.error(f"Failed to process frame {frame_idx}: {str(e)}")
                continue

        # Save summary
        summary_data = {
            "stereo_calibration": stereo_data,
            "reconstruction_summary": {
                "total_frames_processed": len(reconstructed_frames),
                "camera_pair": [cam1_num, cam2_num],
                "configuration": {
                    "max_correspondence_distance": self.max_distance,
                    "min_triangulation_angle": self.min_angle,
                    "vector_pattern": self.vector_pattern,
                    "type_name": self.type_name,
                },
                "timestamp": datetime.now().isoformat(),
            },
        }

        summary_file = output_dir / "reconstruction_summary.mat"
        savemat(str(summary_file), summary_data)

        logger.info(f"Saved reconstruction summary: {summary_file}")
        logger.info(
            f"Completed camera pair {cam1_num}-{cam2_num}: {len(reconstructed_frames)} frames"
        )

    def run(self, runs_to_process=None):
        """Run stereo reconstruction for all camera pairs"""
        logger.info("Starting stereo reconstruction")

        for cam1_num, cam2_num in self.camera_pairs:
            try:
                self.process_camera_pair(cam1_num, cam2_num, runs_to_process)
            except Exception as e:
                logger.error(
                    f"Failed to process camera pair {cam1_num}-{cam2_num}: {str(e)}"
                )
                continue

        logger.info("Stereo reconstruction completed")


def main():
    logger.info("Starting 3D velocity reconstruction with configuration:")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Camera pairs: {CAMERA_PAIRS}")
    logger.info(f"Image count: {image_count}")
    logger.info(f"Vector pattern: {VECTOR_PATTERN}")
    logger.info(f"Type name: {TYPE_NAME}")
    logger.info(f"Max correspondence distance: {MAX_CORRESPONDENCE_DISTANCE} mm")
    logger.info(f"Min triangulation angle: {MIN_TRIANGULATION_ANGLE} degrees")
    logger.info(f"Runs to process: {RUNS_TO_PROCESS}")

    try:
        reconstructor = StereoReconstructor(
            base_dir=BASE_DIR,
            camera_pairs=CAMERA_PAIRS,
            image_count=image_count,
            vector_pattern=VECTOR_PATTERN,
            type_name=TYPE_NAME,
            max_distance=MAX_CORRESPONDENCE_DISTANCE,
            min_angle=MIN_TRIANGULATION_ANGLE,
        )

        reconstructor.run(RUNS_TO_PROCESS)

        logger.info("3D velocity reconstruction completed successfully")

    except Exception as e:
        logger.error(f"Reconstruction failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
