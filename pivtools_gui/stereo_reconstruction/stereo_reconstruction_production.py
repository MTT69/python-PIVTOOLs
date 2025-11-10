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

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent.parent))
from ..paths import get_data_paths
from ..vector_loading import load_coords_from_directory, read_mat_contents

# ===================== CONFIGURATION VARIABLES =====================
# Set these variables for your stereo reconstruction setup
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Stereo_Images/ProcessedPIV"
CAMERA_PAIRS = [[1, 2]]  # Array of camera pairs to process
IMAGE_COUNT = 1000  # Number of images to process for stereo reconstruction
VECTOR_PATTERN = "%05d.mat"  # Pattern for vector files
TYPE_NAME = "instantaneous"  # Type name for calibrated data directory
MAX_CORRESPONDENCE_DISTANCE = 5.0  # Maximum distance in mm for point correspondence
MIN_TRIANGULATION_ANGLE = 5.0  # Minimum angle in degrees for triangulation
DT = 0.01  # Time between frames in seconds
# ===================================================================

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
        progress_cb=None,
        dt=1.0,  # NEW: time between frames in seconds
    ):
        self.base_dir = Path(base_dir)
        self.camera_pairs = camera_pairs
        self.image_count = image_count
        self.vector_pattern = vector_pattern
        self.type_name = type_name
        self.max_distance = max_distance
        self.min_angle = min_angle
        self._stereo_diag_done = False
        self.progress_cb = progress_cb
        self.dt = dt  # Store dt

    def load_stereo_calibration(self, cam1_num, cam2_num):
        stereo_file = (
            self.base_dir
            / "calibration"
            / f"stereo_model_cam{cam1_num}-cam{cam2_num}.mat"
        )
        if not stereo_file.exists():
            raise FileNotFoundError(f"Stereo calibration not found: {stereo_file}")
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

    def load_uncalibrated_coordinates(self, cam_num):
        paths = get_data_paths(
            self.base_dir,
            num_images=self.image_count,
            cam=cam_num,
            type_name=self.type_name,
            use_uncalibrated=True,
        )
        coords_file = paths["data_dir"]
        logger.info(f"Loading coordinates from: {coords_file}")

        # First try to detect available runs by looking at coordinate files
        coord_files = list(coords_file.glob("coords_run*.mat"))
        if coord_files:
            # Extract run numbers from filenames
            available_runs = []
            for f in sorted(coord_files):
                try:
                    run_num = int(f.stem.split("_run")[-1])
                    available_runs.append(run_num)
                except ValueError:
                    continue
            logger.info(f"Found coordinate files for runs: {available_runs}")
            x_list, y_list = load_coords_from_directory(
                coords_file, runs=available_runs
            )
        else:
            # Fallback: try to load all runs without specifying
            logger.info("No specific run files found, trying to load all coordinates")
            x_list, y_list = load_coords_from_directory(coords_file, runs=None)
            available_runs = list(range(1, len(x_list) + 1)) if x_list else []

        logger.info(f"Loaded {len(x_list)} coordinate sets")
        filtered_coords = []
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            run_num = available_runs[i] if i < len(available_runs) else i + 1
            filtered_coords.append({"x_px": x, "y_px": y, "run": run_num})
        return filtered_coords

    def load_uncalibrated_vectors(self, cam_num, frame_idx, run_idx):
        paths = get_data_paths(
            self.base_dir,
            num_images=self.image_count,
            cam=cam_num,
            type_name=self.type_name,
            use_uncalibrated=True,
        )
        vector_file = paths["data_dir"] / (self.vector_pattern % frame_idx)
        logger.debug(f"Looking for vector file: {vector_file}")
        if not vector_file.exists():
            raise FileNotFoundError(f"Vector file not found: {vector_file}")
        logger.debug(f"Loading vector file: {vector_file}")
        vector_data = read_mat_contents(str(vector_file))
        logger.debug(f"Vector data shape: {vector_data.shape}")

        # Extract data for the specific run
        if vector_data.ndim == 4 and vector_data.shape[0] >= run_idx:
            # Multiple runs in file: (runs, 3, height, width)
            ux_px = vector_data[run_idx - 1, 0, :, :]
            uy_px = vector_data[run_idx - 1, 1, :, :]
            b_mask = vector_data[run_idx - 1, 2, :, :]
        elif (
            vector_data.ndim == 4
            and vector_data.shape[0] == 1
            and vector_data.shape[1] == 3
        ):
            # Single run with extra dimension: (1, 3, height, width)
            # Assume this single run corresponds to the requested run
            logger.debug(
                f"Vector file contains 1 run, assuming it corresponds to requested run {run_idx}"
            )
            ux_px = vector_data[0, 0, :, :]
            uy_px = vector_data[0, 1, :, :]
            b_mask = vector_data[0, 2, :, :]
        elif vector_data.ndim == 3 and vector_data.shape[0] == 3:
            # Single run: (3, height, width)
            # Assume this single run corresponds to the requested run
            logger.debug(
                f"Vector file contains 1 run, assuming it corresponds to requested run {run_idx}"
            )
            ux_px = vector_data[0, :, :]
            uy_px = vector_data[1, :, :]
            b_mask = vector_data[2, :, :]
        else:
            raise ValueError(f"Unexpected vector_data shape: {vector_data.shape}")
        return {
            "ux_px": ux_px,
            "uy_px": uy_px,
            "b_mask": b_mask,
            "frame": frame_idx,
            "run": run_idx,
        }

    def find_corresponding_points(self, coords1_px, coords2_px):
        shape1 = coords1_px[0].shape
        shape2 = coords2_px[0].shape
        if shape1 != shape2:
            min_h = min(shape1[0], shape2[0])
            min_w = min(shape1[1], shape2[1])
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
            total_points = np.prod(shape1)
            indices1 = np.arange(total_points)
            indices2 = np.arange(total_points)
        return indices1, indices2

    def triangulate_3d_points(self, pts1_px, pts2_px, stereo_data):
        mtx1 = stereo_data["camera_matrix_1"]
        dist1 = stereo_data["dist_coeffs_1"]
        mtx2 = stereo_data["camera_matrix_2"]
        dist2 = stereo_data["dist_coeffs_2"]
        R1 = stereo_data["rectification_R1"]
        R2 = stereo_data["rectification_R2"]
        P1 = stereo_data["projection_P1"]
        P2 = stereo_data["projection_P2"]
        pts1_rect = cv2.undistortPoints(
            pts1_px.reshape(-1, 1, 2).astype(np.float32), mtx1, dist1, R=R1, P=P1
        ).reshape(-1, 2)
        pts2_rect = cv2.undistortPoints(
            pts2_px.reshape(-1, 1, 2).astype(np.float32), mtx2, dist2, R=R2, P=P2
        ).reshape(-1, 2)
        points_4d = cv2.triangulatePoints(P1, P2, pts1_rect.T, pts2_rect.T)
        points_3d = points_4d[:3] / points_4d[3]
        points_3d = points_3d.T
        # Removed mean-centering to avoid artificial offset
        return points_3d, pts1_rect, pts2_rect

    def compute_triangulation_angles(self, pts_3d, stereo_data):
        R = stereo_data["rotation_matrix"]
        T = stereo_data["translation_vector"].reshape(3)
        cam1_center = np.array([0.0, 0.0, 0.0])
        cam2_center = -R.T @ T
        vec1 = pts_3d - cam1_center
        vec2 = pts_3d - cam2_center
        vec1_norm = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
        vec2_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
        dot_products = np.sum(vec1_norm * vec2_norm, axis=1)
        angles_rad = np.arccos(np.clip(dot_products, -1, 1))
        angles_deg = np.degrees(angles_rad)
        return angles_deg

    def reconstruct_3d_velocities(
        self, ux1, uy1, ux2, uy2, coords1_px, coords2_px, stereo_data
    ):
        indices1, indices2 = self.find_corresponding_points(coords1_px, coords2_px)
        if len(indices1) == 0:
            raise ValueError("No corresponding points found between cameras")
        shape1 = coords1_px[0].shape
        shape2 = coords2_px[0].shape
        row1, col1 = np.unravel_index(indices1, shape1)
        row2, col2 = np.unravel_index(indices2, shape2)
        pts1_px = np.column_stack(
            [coords1_px[0][row1, col1], coords1_px[1][row1, col1]]
        )
        pts2_px = np.column_stack(
            [coords2_px[0][row2, col2], coords2_px[1][row2, col2]]
        )
        vel1 = np.column_stack([ux1[row1, col1], uy1[row1, col1]])
        vel2 = np.column_stack([ux2[row2, col2], uy2[row2, col2]])
        pts_3d, pts1_rect, pts2_rect = self.triangulate_3d_points(
            pts1_px, pts2_px, stereo_data
        )
        angles = self.compute_triangulation_angles(pts_3d, stereo_data)
        angle_mask = angles > self.min_angle
        pts1_displaced_px = pts1_px + vel1
        pts2_displaced_px = pts2_px + vel2
        pts_3d_displaced, _, _ = self.triangulate_3d_points(
            pts1_displaced_px, pts2_displaced_px, stereo_data
        )
        vel_3d_mm_per_frame = pts_3d_displaced - pts_3d
        # Output in mm, coordinates centered around zero
        vel_3d_mm = vel_3d_mm_per_frame
        valid_mask = angle_mask
        return {
            "velocities_3d": vel_3d_mm[valid_mask],
            "positions_3d": pts_3d[valid_mask],
            "indices1": indices1[valid_mask],
            "indices2": indices2[valid_mask],
            "triangulation_angles": angles[valid_mask],
            "num_valid": np.sum(valid_mask),
            "num_total": len(valid_mask),
        }

    def save_calibrated_vectors_matlab_format(
        self,
        result_3d,
        coords1_px,
        coords2_px,
        frame_idx,
        output_dir,
        num_runs,
        current_run_num,
    ):
        ref_shape = coords1_px[0].shape
        ux_grid = np.full(ref_shape, np.nan, dtype=np.float64)
        uy_grid = np.full(ref_shape, np.nan, dtype=np.float64)
        uz_grid = np.full(ref_shape, np.nan, dtype=np.float64)
        # Convert mm to m for displacement and divide by dt for velocity
        velocities_3d_mps = (result_3d["velocities_3d"] / 1000.0) / max(self.dt, 1e-12)
        if result_3d["num_valid"] > 0:
            valid_indices = result_3d["indices1"]
            row_indices, col_indices = np.unravel_index(valid_indices, ref_shape)
            ux_grid[row_indices, col_indices] = velocities_3d_mps[:, 0]
            uy_grid[row_indices, col_indices] = velocities_3d_mps[:, 1]
            uz_grid[row_indices, col_indices] = velocities_3d_mps[:, 2]

        piv_dtype = np.dtype([("ux", "O"), ("uy", "O"), ("uz", "O")])
        piv_result = np.empty(num_runs, dtype=piv_dtype)
        for run_idx in range(num_runs):
            run_num = run_idx + 1
            if run_num == current_run_num:
                # This is the current run with data
                piv_result[run_idx] = (ux_grid, uy_grid, uz_grid)
            else:
                # Empty run - save empty arrays
                piv_result[run_idx] = (np.array([]), np.array([]), np.array([]))
        vector_file = output_dir / (self.vector_pattern % frame_idx)
        savemat(str(vector_file), {"piv_result": piv_result})
        return vector_file

    def save_stereo_coordinates(
        self, result_3d, coords1_px, output_dir, num_runs, current_run_num
    ):
        ref_shape = coords1_px[0].shape
        x_grid = np.full(ref_shape, np.nan, dtype=np.float64)
        y_grid = np.full(ref_shape, np.nan, dtype=np.float64)
        z_grid = np.full(ref_shape, np.nan, dtype=np.float64)
        if result_3d["num_valid"] > 0:
            valid_indices = result_3d["indices1"]
            row_indices, col_indices = np.unravel_index(valid_indices, ref_shape)
            positions_3d = result_3d["positions_3d"]
            # Center coordinates around zero
            mean_xyz = np.mean(positions_3d, axis=0)
            centered_xyz = positions_3d - mean_xyz
            x_grid[row_indices, col_indices] = centered_xyz[:, 0]
            y_grid[row_indices, col_indices] = centered_xyz[:, 1]
            z_grid[row_indices, col_indices] = centered_xyz[:, 2]

        coord_dtype = np.dtype([("x", "O"), ("y", "O"), ("z", "O")])
        coordinates = np.empty(num_runs, dtype=coord_dtype)
        for run_idx in range(num_runs):
            run_num = run_idx + 1
            if run_num == current_run_num:
                # This is the current run with data
                coordinates[run_idx] = (x_grid, y_grid, z_grid)
            else:
                # Empty run - save empty arrays
                coordinates[run_idx] = (np.array([]), np.array([]), np.array([]))
        coord_file = output_dir / "coordinates.mat"
        coords_output = {"coordinates": coordinates}
        savemat(str(coord_file), coords_output)
        return coord_file

    def determine_output_camera(self, cam1_num, cam2_num):
        return cam1_num

    def process_camera_pair(self, cam1_num, cam2_num):
        logger.info(
            f"Starting stereo 3D reconstruction for pair ({cam1_num},{cam2_num})"
        )
        stereo_data = self.load_stereo_calibration(cam1_num, cam2_num)
        # Strip private MATLAB keys to avoid MatWriteWarning
        stereo_data_sanitized = {
            k: v for k, v in stereo_data.items() if not k.startswith("_")
        }
        coords1_list = self.load_uncalibrated_coordinates(cam1_num)
        coords2_list = self.load_uncalibrated_coordinates(cam2_num)
        logger.info(f"Camera {cam1_num}: Found {len(coords1_list)} coordinate sets")
        logger.info(f"Camera {cam2_num}: Found {len(coords2_list)} coordinate sets")

        if len(coords1_list) == 0:
            raise ValueError(f"No coordinate data found for Camera {cam1_num}")
        if len(coords2_list) == 0:
            raise ValueError(f"No coordinate data found for Camera {cam2_num}")
        if len(coords1_list) != len(coords2_list):
            min_sets = min(len(coords1_list), len(coords2_list))
            coords1_list = coords1_list[:min_sets]
            coords2_list = coords2_list[:min_sets]
            logger.info(f"Adjusted to {min_sets} coordinate sets to match both cameras")

        # Check which runs have valid coordinate data
        valid_runs = []
        for i, (coords1, coords2) in enumerate(zip(coords1_list, coords2_list)):
            valid_coords1 = np.sum(~np.isnan(coords1["x_px"]))
            valid_coords2 = np.sum(~np.isnan(coords2["x_px"]))
            run_num = coords1["run"]
            logger.info(
                f"Run {run_num}: Cam{cam1_num}={valid_coords1}, Cam{cam2_num}={valid_coords2} valid coordinates"
            )
            if valid_coords1 > 0 and valid_coords2 > 0:
                valid_runs.append((i, run_num, valid_coords1 + valid_coords2))

        if not valid_runs:
            raise ValueError("No runs with valid coordinate data found")

        logger.info(
            f"Found {len(valid_runs)} runs with valid data: {[r[1] for r in valid_runs]}"
        )

        output_cam = self.determine_output_camera(cam1_num, cam2_num)
        output_dir = (
            self.base_dir
            / "calibrated_piv"
            / str(self.image_count)
            / f"cam{output_cam}"
            / self.type_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Process each valid run
        total_successful_frames = 0
        for run_idx, run_num, total_coords in valid_runs:
            logger.info(
                f"Processing run {run_num} (index {run_idx}) with {total_coords} coordinates"
            )

            coords1 = coords1_list[run_idx]
            coords2 = coords2_list[run_idx]
            coords1_px = (coords1["x_px"], coords1["y_px"])
            coords2_px = (coords2["x_px"], coords2["y_px"])

            logger.info(
                f"Run {run_num} coord shapes: Cam{cam1_num}={coords1_px[0].shape}, Cam{cam2_num}={coords2_px[0].shape}"
            )

            successful_frames = 0
            coordinates_saved = False
            for frame_idx in range(1, self.image_count + 1):
                try:
                    if frame_idx <= 5 or frame_idx % 100 == 0:
                        logger.info(f"Run {run_num}, Frame {frame_idx}")
                    vectors1 = self.load_uncalibrated_vectors(
                        cam1_num, frame_idx, run_num
                    )
                    vectors2 = self.load_uncalibrated_vectors(
                        cam2_num, frame_idx, run_num
                    )
                    result_3d = self.reconstruct_3d_velocities(
                        vectors1["ux_px"],
                        vectors1["uy_px"],
                        vectors2["ux_px"],
                        vectors2["uy_px"],
                        coords1_px,
                        coords2_px,
                        stereo_data,
                    )
                    self.save_calibrated_vectors_matlab_format(
                        result_3d,
                        coords1_px,
                        coords2_px,
                        frame_idx,
                        output_dir,
                        len(coords1_list),
                        run_num,
                    )
                    if not coordinates_saved and result_3d["num_valid"] > 0:
                        self.save_stereo_coordinates(
                            result_3d,
                            coords1_px,
                            output_dir,
                            len(coords1_list),
                            run_num,
                        )
                        coordinates_saved = True
                    successful_frames += 1
                except FileNotFoundError as e:
                    if frame_idx <= 3:
                        logger.warning(f"Run {run_num}, Frame {frame_idx}: {e}")
                    break  # Stop processing this run if vector files don't exist
                except Exception as e:
                    if frame_idx <= 5:
                        logger.error(f"Run {run_num}, Frame {frame_idx} failed: {e}")
                finally:
                    if self.progress_cb:
                        try:
                            self.progress_cb(
                                {
                                    "camera_pair": [cam1_num, cam2_num],
                                    "processed": frame_idx,
                                    "successful": successful_frames,
                                    "total": self.image_count,
                                    "current_run": run_num,
                                }
                            )
                        except Exception:
                            pass

            logger.info(f"Run {run_num}: {successful_frames} successful frames")
            total_successful_frames += successful_frames

        logger.info(
            f"All runs completed: {total_successful_frames} total successful frames"
        )
        summary_data = {
            "stereo_calibration": stereo_data_sanitized,
            "reconstruction_summary": {
                "total_frames_processed": total_successful_frames,
                "total_frames_attempted": self.image_count,
                "camera_pair": [cam1_num, cam2_num],
                "output_camera": output_cam,
                "output_directory": str(output_dir),
                "configuration": {
                    "max_correspondence_distance": self.max_distance,
                    "min_triangulation_angle": self.min_angle,
                    "vector_pattern": self.vector_pattern,
                    "type_name": self.type_name,
                    "image_count": self.image_count,
                },
                "timestamp": datetime.now().isoformat(),
            },
        }
        summary_file = output_dir / "stereo_reconstruction_summary.mat"
        savemat(str(summary_file), summary_data)

    def run(self):
        for cam1_num, cam2_num in self.camera_pairs:
            try:
                self.process_camera_pair(cam1_num, cam2_num)
            except Exception as e:
                logger.error(
                    f"Reconstruction failed for pair ({cam1_num},{cam2_num}): {e}"
                )


def main():
    reconstructor = StereoReconstructor(
        base_dir=BASE_DIR,
        camera_pairs=CAMERA_PAIRS,
        image_count=IMAGE_COUNT,
        vector_pattern=VECTOR_PATTERN,
        type_name=TYPE_NAME,
        max_distance=MAX_CORRESPONDENCE_DISTANCE,
        min_angle=MIN_TRIANGULATION_ANGLE,
        dt=DT,
    )
    reconstructor.run()


if __name__ == "__main__":
    main()
