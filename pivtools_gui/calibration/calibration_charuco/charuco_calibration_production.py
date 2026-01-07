#!/usr/bin/env python3
"""
charuco_calibration_production.py

Production-ready ChArUco board calibration for camera intrinsic parameters.
Uses OpenCV's ChArUco detection with multi-image aggregation for robust calibration.
Saves results to: {BASE_DIR}/calibration/Cam{N}/charuco_planar/
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from pivtools_core.config import get_config, reload_config
from pivtools_core.image_handling.load_images import read_image

# ===================== CONFIGURATION VARIABLES =====================

# -------------------- PATH CONFIGURATION --------------------
# SOURCE_DIR: Root directory containing your data.
SOURCE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Planar_Images_with_wall/Cam1"

# BASE_DIR: The output directory where calibration results will be saved.
#           Results are saved to: {BASE_DIR}/calibration/Cam{N}/charuco_planar/...
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Planar_Images_with_wall/test"

# CALIBRATION_SUBFOLDER: Subfolder within the source path for calibration images.
#                        Leave empty "" to look directly in SOURCE_DIR.
CALIBRATION_SUBFOLDER = ""

# -------------------- CAMERA CONFIGURATION --------------------
# CAMERA_NUMS: List of camera numbers to process (1-based), e.g. [1, 2] for stereo
CAMERA_NUMS = [1]

# CAMERA_SUBFOLDERS: List of subfolder names for each camera (index matches camera number - 1).
#                    e.g., ["Cam1", "Cam2"] means camera 1 uses "Cam1/", camera 2 uses "Cam2/"
#                    Set to [] (empty list) for container formats or when images are in SOURCE_DIR directly.
CAMERA_SUBFOLDERS = []

# FILE_PATTERN: The naming pattern for calibration images (e.g., "calib%05d.tif" or "*.tif")
FILE_PATTERN = "calib%05d.tif"

# -------------------- CHARUCO BOARD SETTINGS --------------------
# These must match your physical calibration target exactly.
SQUARES_H = 10              # Number of squares horizontally
SQUARES_V = 9               # Number of squares vertically
SQUARE_SIZE_M = 0.03        # Physical square size in METERS
MARKER_RATIO = 0.5          # Ratio of marker size to square size (usually 0.5)
ARUCO_DICT = "DICT_4X4_1000" # ArUco dictionary used
MIN_CORNERS = 6             # Minimum number of corners required to accept an image

# USE_CONFIG_DIRECTLY: If True, skip updating config.yaml with above parameters
# and load calibration settings directly from the existing config.yaml
USE_CONFIG_DIRECTLY = True

# ===================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Standard ArUco dictionaries mapping
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


def apply_cli_settings_to_config():
    """Update config.yaml with CLI-mode hardcoded settings.

    This function writes the hardcoded configuration variables to config.yaml,
    ensuring the centralized image loading system uses the correct paths and settings.

    Returns
    -------
    Config
        The reloaded config object with updated settings
    """
    config = get_config()

    # Paths
    config.data["paths"]["source_paths"] = [SOURCE_DIR]
    config.data["paths"]["base_paths"] = [BASE_DIR]
    config.data["paths"]["camera_subfolders"] = CAMERA_SUBFOLDERS
    config.data["paths"]["camera_count"] = len(CAMERA_NUMS)
    config.data["paths"]["camera_numbers"] = CAMERA_NUMS

    # Calibration settings
    config.data["calibration"]["image_format"] = FILE_PATTERN
    config.data["calibration"]["subfolder"] = CALIBRATION_SUBFOLDER

    # ChArUco-specific params
    config.data["calibration"]["charuco"]["squares_h"] = SQUARES_H
    config.data["calibration"]["charuco"]["squares_v"] = SQUARES_V
    config.data["calibration"]["charuco"]["square_size"] = SQUARE_SIZE_M
    config.data["calibration"]["charuco"]["marker_ratio"] = MARKER_RATIO
    config.data["calibration"]["charuco"]["aruco_dict"] = ARUCO_DICT
    config.data["calibration"]["charuco"]["min_corners"] = MIN_CORNERS

    # Save to disk so centralized loader picks up changes
    config.save()
    logger.info("Updated config.yaml with CLI settings")

    # Reload to ensure fresh state
    return reload_config()


class ChArUcoCalibrator:
    def __init__(
        self,
        source_dir,
        base_dir,
        camera_count=1,
        file_pattern="*.tif",
        squares_h=10,
        squares_v=9,
        square_size=0.03,
        marker_ratio=0.5,
        aruco_dict="DICT_4X4_1000",
        min_corners=6,
        dt=1.0,
        calibration_subfolder="",
        calibration_input_path=None,
        config=None,
    ):
        self.source_dir = Path(source_dir)
        self.base_dir = Path(base_dir)
        self.camera_count = camera_count
        self.file_pattern = file_pattern
        self.squares_h = squares_h
        self.squares_v = squares_v
        self.square_size = square_size
        self.marker_ratio = marker_ratio
        self.aruco_dict_name = aruco_dict
        self.min_corners = min_corners
        self.dt = dt
        self.calibration_subfolder = calibration_subfolder
        self.calibration_input_path = Path(calibration_input_path) if calibration_input_path else None
        self._config = config

        # Create board and detector
        self.board, self.detector = self._create_detector()

        # Setup output directories
        self._setup_directories()

    def _create_detector(self) -> Tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector]:
        marker_size = self.square_size * self.marker_ratio
        dict_id = ARUCO_DICT_MAP.get(self.aruco_dict_name, cv2.aruco.DICT_4X4_1000)
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

        board = cv2.aruco.CharucoBoard(
            (self.squares_h, self.squares_v),
            self.square_size,
            marker_size,
            dictionary,
        )

        detector = cv2.aruco.CharucoDetector(
            board,
            cv2.aruco.CharucoParameters(),
            cv2.aruco.DetectorParameters(),
        )

        return board, detector

    def _setup_directories(self):
        """Create necessary output directories including charuco_planar subfolder."""
        for cam_num in range(1, self.camera_count + 1):
            # Output path: .../calibration/CamX/charuco_planar/...
            cam_base = self.base_dir / "calibration" / f"Cam{cam_num}" / "charuco_planar"
            (cam_base / "detections").mkdir(parents=True, exist_ok=True)
            (cam_base / "model").mkdir(parents=True, exist_ok=True)
            (cam_base / "indices").mkdir(parents=True, exist_ok=True)

    def _get_camera_input_dir(self, cam_num: int) -> Path:
        """Get the input directory for calibration images.

        Path structure: source / camera_folder / calibration_subfolder
        """
        # If explicit calibration_input_path is provided, use it
        if self.calibration_input_path:
            return self.calibration_input_path

        base_path = self.source_dir

        # Get camera folder from config
        if self._config is not None:
            camera_folder = self._config.get_calibration_camera_folder(cam_num)
            if camera_folder:
                base_path = base_path / camera_folder

        # Add calibration subfolder if set
        if self.calibration_subfolder:
            base_path = base_path / self.calibration_subfolder

        return base_path

    def _is_container_format(self) -> bool:
        """Check if file pattern is a container format (.set, .im7, .cine)."""
        pattern_lower = self.file_pattern.lower()
        return ".set" in pattern_lower or ".im7" in pattern_lower or ".cine" in pattern_lower

    def _read_calibration_image(
        self, img_path: Path, camera: int = 1, img_index: int = 1
    ) -> Optional[np.ndarray]:
        """Read calibration image with container format support (.set, .im7, .cine, standard formats)."""
        try:
            if self._is_container_format():
                if ".set" in str(img_path).lower():
                    img = read_image(str(img_path), camera_no=camera, im_no=img_index)
                elif ".im7" in str(img_path).lower():
                    img = read_image(str(img_path), camera_no=camera)
                elif ".cine" in str(img_path).lower():
                    # .cine files: use dedicated single-frame reader
                    from pivtools_core.image_handling.readers.cine_reader import read_cine_single
                    img = read_cine_single(str(img_path), idx=img_index)
                else:
                    img = read_image(str(img_path))
            else:
                img = read_image(str(img_path))

            if img is None:
                return None

            # Normalize to uint8
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            elif img.dtype in [np.float32, np.float64]:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            elif img.dtype == np.bool_:
                img = img.astype(np.uint8) * 255

            return img

        except Exception as e:
            logger.warning(f"Failed to read image {img_path}: {e}")
            return None

    def _find_calibration_images(self, cam_input_dir: Path) -> List[Path]:
        """Find all calibration images matching the pattern."""
        if self._is_container_format():
            container_file = cam_input_dir / self.file_pattern
            if container_file.exists():
                return [container_file]
            return []

        # Numbered pattern (e.g., "calib%05d.tif")
        if "%" in self.file_pattern:
            files = []
            i = 1
            while True:
                try:
                    filename = self.file_pattern % i
                except TypeError:
                    break
                filepath = cam_input_dir / filename
                if filepath.exists():
                    files.append(filepath)
                    i += 1
                else:
                    break
            return files

        # Single file
        single = cam_input_dir / self.file_pattern
        return [single] if single.exists() else []

    def detect_charuco_corners(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect ChArUco corners in an image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(gray)

        if ids is None or len(corners) < self.min_corners:
            return False, None, None, marker_corners, marker_ids

        return True, corners, ids, marker_corners, marker_ids

    def _save_detection_visualization(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        ids: np.ndarray,
        marker_corners: Optional[np.ndarray],
        filename: str,
        output_dir: Path,
    ):
        """Save visualization of detected corners."""
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        if marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(vis, marker_corners)

        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{filename} - {len(corners)} corners detected")
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / f"{filename}_detection.png", dpi=150)
        plt.close(fig)

    def process_camera(
        self,
        cam_num: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        save_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all calibration images for one camera.

        Args:
            cam_num: Camera number to process
            progress_callback: Optional callback function receiving progress dict with:
                - processed_images: int
                - valid_images: int
                - total_images: int
                - progress: int (0-100)
            save_visualizations: Whether to save detection visualization PNGs

        Returns:
            Dict with success status and calibration results:
                - success: bool
                - camera_matrix: list (if success)
                - dist_coeffs: list (if success)
                - rms_error: float (if success)
                - num_images_used: int (if success)
                - model_path: str (if success)
                - error: str (if not success)
        """
        logger.info(f"Processing Camera {cam_num}")

        is_container = self._is_container_format()
        cam_input_dir = self._get_camera_input_dir(cam_num)

        # Output path structure: .../CamX/charuco_planar/...
        cam_output_base = self.base_dir / "calibration" / f"Cam{cam_num}" / "charuco_planar"
        detections_dir = cam_output_base / "detections"
        indices_dir = cam_output_base / "indices"

        # Ensure directories exist
        detections_dir.mkdir(parents=True, exist_ok=True)
        indices_dir.mkdir(parents=True, exist_ok=True)
        (cam_output_base / "model").mkdir(parents=True, exist_ok=True)

        if not cam_input_dir.exists():
            error_msg = f"Calibration directory not found: {cam_input_dir}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        image_files = self._find_calibration_images(cam_input_dir)
        if not image_files:
            error_msg = f"No calibration images found in {cam_input_dir}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        logger.info(f"Found {len(image_files)} images (or container files)")

        all_obj_points = []
        all_img_points = []
        img_size = None
        stats = {"empty": 0, "no_detect": 0, "valid": 0}
        valid_images = []

        # Store per-frame detection data for indices saving
        # Key: frame index (1-based), Value: dict with corners, ids, filename
        valid_indices_map: Dict[int, Dict[str, Any]] = {}

        # Count total images for progress tracking
        total_images = len(image_files)
        if is_container:
            # Estimate container size (will update as we read)
            total_images = 100  # Max container frames to try

        processed_count = 0

        for idx, img_path in enumerate(image_files):
            # Container logic
            if is_container:
                for img_idx in range(1, 101):  # Attempt reading up to 100 frames
                    image = self._read_calibration_image(img_path, camera=cam_num, img_index=img_idx)
                    if image is None:
                        # Update total_images when we hit end of container
                        total_images = img_idx - 1
                        break

                    processed_count += 1

                    if img_size is None and image is not None:
                        h, w = image.shape[:2]
                        img_size = (w, h)

                    detection_result = self._process_single_image_with_data(
                        image,
                        f"{img_path.stem}_img{img_idx:03d}",
                        all_obj_points,
                        all_img_points,
                        valid_images,
                        stats,
                        detections_dir if save_visualizations else None,
                    )

                    if detection_result is not None:
                        valid_indices_map[img_idx] = detection_result

                    # Report progress
                    if progress_callback:
                        progress_callback({
                            "processed_images": processed_count,
                            "valid_images": stats["valid"],
                            "total_images": max(total_images, processed_count),
                            "progress": int((processed_count / max(total_images, processed_count)) * 100),
                        })

            # Standard file logic
            else:
                image = self._read_calibration_image(img_path, camera=cam_num, img_index=idx + 1)
                processed_count += 1

                if image is None:
                    if progress_callback:
                        progress_callback({
                            "processed_images": processed_count,
                            "valid_images": stats["valid"],
                            "total_images": total_images,
                            "progress": int((processed_count / total_images) * 100),
                        })
                    continue

                if img_size is None:
                    h, w = image.shape[:2]
                    img_size = (w, h)

                frame_index = idx + 1  # 1-based frame index
                detection_result = self._process_single_image_with_data(
                    image,
                    img_path.stem,
                    all_obj_points,
                    all_img_points,
                    valid_images,
                    stats,
                    detections_dir if save_visualizations else None,
                )

                if detection_result is not None:
                    detection_result["original_filename"] = img_path.name
                    valid_indices_map[frame_index] = detection_result

                # Report progress
                if progress_callback:
                    progress_callback({
                        "processed_images": processed_count,
                        "valid_images": stats["valid"],
                        "total_images": total_images,
                        "progress": int((processed_count / total_images) * 100),
                    })

        logger.info(
            f"Valid: {stats['valid']}, Empty: {stats['empty']}, No detection: {stats['no_detect']}"
        )

        if stats["valid"] < 3:
            error_msg = f"Need at least 3 valid images, got {stats['valid']}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        # Run calibration
        logger.info(f"Calibrating with {len(all_obj_points)} images...")

        rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            all_obj_points, all_img_points, img_size, None, None
        )

        logger.info(f"RMS reprojection error: {rms:.4f} pixels")

        # Calculate errors
        all_errors = []
        all_errors_x = []
        all_errors_y = []

        for i in range(len(all_obj_points)):
            proj, _ = cv2.projectPoints(
                all_obj_points[i], rvecs[i], tvecs[i], mtx, dist
            )
            proj = proj.reshape(-1, 2)
            img_pts = all_img_points[i].reshape(-1, 2)
            err_vec = img_pts - proj
            all_errors.extend(np.linalg.norm(err_vec, axis=1).tolist())
            all_errors_x.extend(err_vec[:, 0].tolist())
            all_errors_y.extend(err_vec[:, 1].tolist())

        # Save per-frame indices
        logger.info(f"Saving per-frame detection indices for {len(valid_indices_map)} frames...")
        for frame_idx, detection_data in valid_indices_map.items():
            indices_data = {
                "corners": detection_data["corners"],
                "corner_ids": detection_data["ids"],
                "corner_count": len(detection_data["corners"]),
                "original_filename": detection_data.get("original_filename", ""),
                "frame_index": frame_idx,
                "board_params": {
                    "squares_h": self.squares_h,
                    "squares_v": self.squares_v,
                    "square_size": self.square_size,
                    "square_size_mm": self.square_size * 1000.0,
                    "marker_ratio": self.marker_ratio,
                    "aruco_dict": self.aruco_dict_name,
                },
            }
            indices_file = indices_dir / f"indexing_{frame_idx}.mat"
            savemat(str(indices_file), indices_data)

        # Save model
        model_data = {
            "camera_matrix": mtx,
            "dist_coeffs": dist,
            "rvecs": np.array([r.flatten() for r in rvecs]),
            "tvecs": np.array([t.flatten() for t in tvecs]),
            "reprojection_error": rms,
            "reprojection_error_x_mean": float(np.mean(np.abs(all_errors_x))),
            "reprojection_error_y_mean": float(np.mean(np.abs(all_errors_y))),
            "reprojection_errors": np.array(all_errors),
            "reprojection_errors_x": np.array(all_errors_x),
            "reprojection_errors_y": np.array(all_errors_y),
            "num_images": stats["valid"],
            "image_size": list(img_size),
            "timestamp": datetime.now().isoformat(),
            "dt": self.dt,
            "dot_spacing_mm": self.square_size * 1000.0,
            "board_params": {
                "squares_h": self.squares_h,
                "squares_v": self.squares_v,
                "square_size": self.square_size,
                "square_size_mm": self.square_size * 1000.0,
                "marker_ratio": self.marker_ratio,
                "aruco_dict": self.aruco_dict_name,
            },
        }

        model_path = cam_output_base / "model" / "camera_model.mat"
        savemat(str(model_path), model_data)
        logger.info(f"Saved camera model: {model_path}")

        # JSON save
        json_data = {
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "rms_error": float(rms),
            "image_size": list(img_size),
            "num_images_used": stats["valid"],
        }
        json_path = cam_output_base / "model" / "camera_model.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        return {
            "success": True,
            "camera_matrix": mtx.tolist(),
            "dist_coeffs": dist.flatten().tolist(),
            "rms_error": float(rms),
            "num_images_used": stats["valid"],
            "model_path": str(model_path),
        }

    def _process_single_image_with_data(
        self,
        image: np.ndarray,
        name: str,
        all_obj_points: List,
        all_img_points: List,
        valid_images: List,
        stats: Dict,
        detections_dir: Optional[Path],
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single calibration image and return detection data.

        Returns:
            Dict with corners and ids if detection successful, None otherwise.
        """
        if np.mean(image) < 10:
            stats["empty"] += 1
            return None

        found, corners, ids, marker_corners, marker_ids = self.detect_charuco_corners(image)

        if not found:
            stats["no_detect"] += 1
            return None

        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)

        if obj_pts is None or len(obj_pts) < self.min_corners:
            stats["no_detect"] += 1
            return None

        all_obj_points.append(obj_pts)
        all_img_points.append(img_pts)
        valid_images.append(name)
        stats["valid"] += 1

        logger.info(f"  {name}: OK ({len(corners)} corners)")

        if detections_dir is not None:
            self._save_detection_visualization(
                image, corners, ids, marker_corners, name, detections_dir
            )

        # Return detection data for indices saving
        # Reshape corners from (N, 1, 2) to (N, 2) for cleaner storage
        corners_2d = corners.reshape(-1, 2) if corners is not None else np.array([])
        ids_flat = ids.flatten() if ids is not None else np.array([])

        return {
            "corners": corners_2d,
            "ids": ids_flat,
            "name": name,
        }

    def _process_single_image(
        self,
        image: np.ndarray,
        name: str,
        all_obj_points: List,
        all_img_points: List,
        valid_images: List,
        stats: Dict,
        detections_dir: Path,
    ) -> bool:
        """Process a single calibration image (legacy wrapper)."""
        result = self._process_single_image_with_data(
            image, name, all_obj_points, all_img_points,
            valid_images, stats, detections_dir
        )
        return result is not None

    def process_all_cameras(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        save_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all cameras with progress tracking.

        Args:
            progress_callback: Optional callback function receiving progress dict with:
                - current_camera: int or None
                - processed_cameras: int
                - total_cameras: int
                - progress: int (0-100)
                - camera_results: dict of per-camera status
            save_visualizations: Whether to save detection visualization PNGs

        Returns:
            Dict with overall results:
                - success: bool
                - processed_cameras: int
                - camera_results: dict mapping camera number to result dict
        """
        camera_results = {}
        processed_cameras = 0
        total_cameras = self.camera_count

        for cam_num in range(1, self.camera_count + 1):
            # Report starting this camera
            if progress_callback:
                progress_callback({
                    "current_camera": cam_num,
                    "processed_cameras": processed_cameras,
                    "total_cameras": total_cameras,
                    "progress": int((processed_cameras / total_cameras) * 100),
                })

            try:
                # Create per-camera progress callback that reports to parent
                def camera_progress(data):
                    if progress_callback:
                        progress_callback({
                            "current_camera": cam_num,
                            "processed_cameras": processed_cameras,
                            "total_cameras": total_cameras,
                            "progress": int((processed_cameras / total_cameras) * 100),
                            "processed_images": data.get("processed_images", 0),
                            "valid_images": data.get("valid_images", 0),
                            "total_images": data.get("total_images", 0),
                        })

                result = self.process_camera(
                    cam_num,
                    progress_callback=camera_progress,
                    save_visualizations=save_visualizations,
                )
                camera_results[cam_num] = result
                processed_cameras += 1

            except Exception as e:
                logger.error(f"Failed to process Camera {cam_num}: {e}")
                camera_results[cam_num] = {"success": False, "error": str(e)}
                processed_cameras += 1

        # Final progress report
        if progress_callback:
            progress_callback({
                "current_camera": None,
                "processed_cameras": processed_cameras,
                "total_cameras": total_cameras,
                "progress": 100,
            })

        # Determine overall success
        success_count = sum(1 for r in camera_results.values() if r.get("success"))

        return {
            "success": success_count > 0,
            "processed_cameras": processed_cameras,
            "successful_cameras": success_count,
            "camera_results": camera_results,
        }

    def run(self):
        """Run calibration (CLI mode)."""
        logger.info("=" * 60)
        logger.info("ChArUco Calibration - Starting")
        logger.info("=" * 60)
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.base_dir}")
        logger.info(f"Board: {self.squares_h}x{self.squares_v} squares, {self.square_size}m size")

        results = self.process_all_cameras(save_visualizations=True)

        logger.info("=" * 60)
        logger.info("ChArUco Calibration - Complete")
        logger.info(f"Processed {results['processed_cameras']} cameras")
        logger.info(f"Successful: {results['successful_cameras']}")

        for cam_num, result in results["camera_results"].items():
            if result.get("success"):
                logger.info(f"  Camera {cam_num}: RMS={result['rms_error']:.4f} px, {result['num_images_used']} images")
            else:
                logger.error(f"  Camera {cam_num}: FAILED - {result.get('error', 'Unknown error')}")

        logger.info("=" * 60)


def main():
    """Main entry point using hardcoded configuration.

    Updates config.yaml with the hardcoded settings, then runs ChArUco calibration.
    """
    logger.info("=" * 60)
    logger.info("ChArUco Calibration - Starting")
    logger.info("=" * 60)

    if USE_CONFIG_DIRECTLY:
        # Load settings directly from existing config.yaml
        logger.info("Loading settings directly from config.yaml (USE_CONFIG_DIRECTLY=True)")
        config = get_config()

        # Extract settings from config
        source_dir = config.data["paths"]["source_paths"][0]
        base_dir = config.data["paths"]["base_paths"][0]
        camera_nums = config.data["paths"].get("camera_numbers", [1])
        file_pattern = config.data["calibration"]["image_format"]
        calibration_subfolder = config.data["calibration"].get("subfolder", "")
        squares_h = config.data["calibration"]["charuco"]["squares_h"]
        squares_v = config.data["calibration"]["charuco"]["squares_v"]
        square_size_m = config.data["calibration"]["charuco"]["square_size"]
        marker_ratio = config.data["calibration"]["charuco"].get("marker_ratio", 0.5)
        aruco_dict = config.data["calibration"]["charuco"].get("aruco_dict", "DICT_4X4_1000")
        min_corners = config.data["calibration"]["charuco"].get("min_corners", 6)
    else:
        # Apply CLI settings to config.yaml so centralized loaders work correctly
        config = apply_cli_settings_to_config()

        # Use hardcoded settings
        source_dir = SOURCE_DIR
        base_dir = BASE_DIR
        camera_nums = CAMERA_NUMS
        file_pattern = FILE_PATTERN
        calibration_subfolder = CALIBRATION_SUBFOLDER
        squares_h = SQUARES_H
        squares_v = SQUARES_V
        square_size_m = SQUARE_SIZE_M
        marker_ratio = MARKER_RATIO
        aruco_dict = ARUCO_DICT
        min_corners = MIN_CORNERS

    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {base_dir}")
    logger.info(f"Cameras: {camera_nums}")
    logger.info(f"Board: {squares_h}x{squares_v} squares, {square_size_m}m size")

    failed_cameras = []

    for camera_num in camera_nums:
        logger.info(f"Processing Camera {camera_num}...")
        try:
            # Create calibrator using config - all settings are now in config.yaml
            calibrator = ChArUcoCalibrator(
                source_dir=source_dir,
                base_dir=base_dir,
                camera_count=1,  # Process one at a time
                file_pattern=file_pattern,
                squares_h=squares_h,
                squares_v=squares_v,
                square_size=square_size_m,
                marker_ratio=marker_ratio,
                aruco_dict=aruco_dict,
                min_corners=min_corners,
                calibration_subfolder=calibration_subfolder,
                config=config,
            )
            result = calibrator.process_camera(camera_num, save_visualizations=True)
            if result.get("success"):
                logger.info(f"Camera {camera_num} completed: RMS={result['rms_error']:.4f} px, {result['num_images_used']} images")
            else:
                logger.error(f"Camera {camera_num} failed: {result.get('error', 'Unknown error')}")
                failed_cameras.append(camera_num)
        except Exception as e:
            logger.error(f"Camera {camera_num} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_cameras.append(camera_num)

    logger.info("=" * 60)
    if failed_cameras:
        logger.error(f"Calibration failed for cameras: {failed_cameras}")
    else:
        logger.info("ChArUco calibration completed successfully for all cameras")


if __name__ == "__main__":
    main()