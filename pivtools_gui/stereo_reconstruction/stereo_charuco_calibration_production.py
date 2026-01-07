#!/usr/bin/env python3
"""
stereo_charuco_calibration_production.py

Production-ready stereo calibration using ChArUco board detection.
Uses OpenCV's CharucoDetector for robust corner detection with ID matching.

Saves results to: {BASE_DIR}/calibration/stereo_cam{N}_cam{M}/
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

from pivtools_core.config import Config, get_config, reload_config
from pivtools_core.image_handling.calibration_loader import get_calibration_frame_count

from pivtools_gui.stereo_reconstruction.stereo_calibration_base import BaseStereoCalibrator


# ===================== CONFIGURATION VARIABLES =====================
# Set these variables for your calibration setup (CLI mode)

SOURCE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Stereo_Images"
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Stereo_Images/ProcessedPIV"
CAMERA_PAIR = [1, 2]
FILE_PATTERN = "planar_calibration_plate_%02d.tif"

# CAMERA_SUBFOLDERS: List of subfolder names for each camera (index matches camera number - 1).
#                    e.g., ["Cam1", "Cam2"] means camera 1 uses "Cam1/", camera 2 uses "Cam2/"
#                    Set to [] (empty list) for container formats or when images are in SOURCE_DIR directly.
CAMERA_SUBFOLDERS = ["Cam1", "Cam2"]

# CALIBRATION_SUBFOLDER: Subfolder within camera folders for calibration images.
#                        Leave empty "" to look directly in camera folders.
CALIBRATION_SUBFOLDER = ""

# ChArUco board parameters
SQUARES_H = 10
SQUARES_V = 9
SQUARE_SIZE = 0.03  # meters
MARKER_RATIO = 0.5
ARUCO_DICT = "DICT_4X4_1000"
MIN_CORNERS = 6

# Number of calibration images to use (set to None to use all available)
NUM_CALIBRATION_IMAGES = None

# USE_CONFIG_DIRECTLY: If True, skip updating config.yaml with above parameters
# and load calibration settings directly from the existing config.yaml
USE_CONFIG_DIRECTLY = True

# ===================================================================


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


def apply_cli_settings_to_config() -> Config:
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
    config.data["paths"]["camera_count"] = len(CAMERA_PAIR)
    config.data["paths"]["camera_numbers"] = CAMERA_PAIR

    # Calibration settings
    config.data["calibration"]["image_format"] = FILE_PATTERN
    config.data["calibration"]["subfolder"] = CALIBRATION_SUBFOLDER

    # Set calibration image count - explicit value or auto-detect from directory
    if NUM_CALIBRATION_IMAGES is not None:
        config.data["calibration"]["num_images"] = NUM_CALIBRATION_IMAGES
    else:
        # Auto-detect from first camera's calibration directory
        # Need to save and reload first so paths are correct for detection
        config.save()
        config = reload_config()
        detected_count = get_calibration_frame_count(camera=CAMERA_PAIR[0], config=config)
        if detected_count > 0:
            config.data["calibration"]["num_images"] = detected_count
            logger.info(f"Auto-detected {detected_count} calibration images")
        else:
            logger.warning("Could not auto-detect calibration image count, using default")

    # Stereo-specific params
    config.data["calibration"]["stereo"]["camera_pair"] = CAMERA_PAIR

    # ChArUco-specific params
    config.data["calibration"]["charuco"]["squares_h"] = SQUARES_H
    config.data["calibration"]["charuco"]["squares_v"] = SQUARES_V
    config.data["calibration"]["charuco"]["square_size"] = SQUARE_SIZE
    config.data["calibration"]["charuco"]["marker_ratio"] = MARKER_RATIO
    config.data["calibration"]["charuco"]["aruco_dict"] = ARUCO_DICT
    config.data["calibration"]["charuco"]["min_corners"] = MIN_CORNERS

    # Save to disk so centralized loader picks up changes
    config.save()
    logger.info(f"Updated config.yaml with CLI settings")

    # Reload to ensure fresh state
    return reload_config()


class StereoCharucoCalibrator(BaseStereoCalibrator):
    """Stereo calibration using ChArUco board detection.

    This calibrator detects ChArUco boards in calibration images and uses them
    for stereo camera calibration. ChArUco boards provide robust corner detection
    with unique IDs, allowing partial occlusion handling.

    Parameters
    ----------
    config : Config, optional
        Configuration object. If provided, settings are read from config.charuco_calibration
    squares_h : int
        Number of squares horizontally on the ChArUco board
    squares_v : int
        Number of squares vertically on the ChArUco board
    square_size : float
        Physical square size in meters
    marker_ratio : float
        Ratio of marker size to square size (usually 0.5)
    aruco_dict : str
        ArUco dictionary name (e.g., "DICT_4X4_1000")
    min_corners : int
        Minimum number of corners required to accept a detection
    **base_kwargs
        Additional arguments passed to BaseStereoCalibrator

    Example
    -------
    >>> calibrator = StereoCharucoCalibrator(
    ...     source_dir="/path/to/images",
    ...     base_dir="/path/to/output",
    ...     camera_pair=[1, 2],
    ...     squares_h=10,
    ...     squares_v=9,
    ...     square_size=0.03,
    ... )
    >>> result = calibrator.process_camera_pair()
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        # ChArUco-specific params (from config.charuco_calibration or explicit)
        squares_h: int = 10,
        squares_v: int = 9,
        square_size: float = 0.03,
        marker_ratio: float = 0.5,
        aruco_dict: str = "DICT_4X4_1000",
        min_corners: int = 6,
        # Base class params
        source_dir: Optional[Union[str, Path]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        camera_pair: Optional[List[int]] = None,
        file_pattern: Optional[str] = None,
        calibration_subfolder: str = "",
        camera_subfolders: Optional[List[str]] = None,
        source_path_idx: int = 0,
        dt: float = 1.0,
    ):
        # Get ChArUco params from config.charuco_calibration if available
        if config is not None:
            charuco_cfg = config.charuco_calibration
            squares_h = charuco_cfg.get('squares_h', squares_h)
            squares_v = charuco_cfg.get('squares_v', squares_v)
            square_size = charuco_cfg.get('square_size', square_size)
            marker_ratio = charuco_cfg.get('marker_ratio', marker_ratio)
            aruco_dict = charuco_cfg.get('aruco_dict', aruco_dict)
            min_corners = charuco_cfg.get('min_corners', min_corners)
            # Get dt from stereo_charuco config (not charuco_calibration)
            stereo_cfg = config.stereo_charuco_calibration
            dt = stereo_cfg.get('dt', dt)

        self.squares_h = squares_h
        self.squares_v = squares_v
        self.square_size = square_size
        self.marker_ratio = marker_ratio
        self.aruco_dict_name = aruco_dict
        self.min_corners = min_corners

        super().__init__(
            config=config,
            source_dir=source_dir,
            base_dir=base_dir,
            camera_pair=camera_pair,
            file_pattern=file_pattern,
            calibration_subfolder=calibration_subfolder,
            camera_subfolders=camera_subfolders,
            source_path_idx=source_path_idx,
            dt=dt,
        )

    def _create_detector(self) -> Tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector]:
        """Create ChArUco board and detector.

        Returns
        -------
        tuple
            (CharucoBoard, CharucoDetector)
        """
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

    def detect_pattern(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect ChArUco corners in image.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or color)

        Returns
        -------
        tuple
            (found: bool, corners: np.ndarray or None, ids: np.ndarray or None)
            corners shape: (N, 2) if found
            ids shape: (N,) if found
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        board, detector = self.detector
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)

        if ids is None or len(corners) < self.min_corners:
            return False, None, None

        # Reshape corners from (N, 1, 2) to (N, 2)
        corners_2d = corners.reshape(-1, 2).astype(np.float32)
        ids_flat = ids.flatten()

        return True, corners_2d, ids_flat

    def make_object_points(self) -> np.ndarray:
        """Generate 3D object points from ChArUco board geometry.

        Returns
        -------
        np.ndarray
            Shape (N, 3) with all chessboard corners
        """
        board, _ = self.detector
        return board.getChessboardCorners().astype(np.float32)

    def get_pattern_params(self) -> Dict[str, Any]:
        """Get pattern-specific parameters for saving to output files.

        Returns
        -------
        dict
            Pattern parameters
        """
        return {
            'pattern_type': 'charuco',
            'squares_h': self.squares_h,
            'squares_v': self.squares_v,
            'square_size': self.square_size,
            'square_size_mm': self.square_size * 1000,
            'marker_ratio': self.marker_ratio,
            'aruco_dict': self.aruco_dict_name,
            'min_corners': self.min_corners,
        }

    def _match_object_points(
        self,
        objp: np.ndarray,
        result1: Tuple,
        result2: Tuple,
    ) -> Optional[np.ndarray]:
        """Match object points between two cameras using corner IDs.

        For ChArUco, we need to find the intersection of detected corner IDs
        and return only the object points corresponding to those IDs.

        Parameters
        ----------
        objp : np.ndarray
            Full object points array from board.getChessboardCorners()
        result1 : tuple
            Detection result from camera 1: (found, corners, ids)
        result2 : tuple
            Detection result from camera 2: (found, corners, ids)

        Returns
        -------
        np.ndarray or None
            Matched object points, or None if matching failed
        """
        _, corners1, ids1 = result1
        _, corners2, ids2 = result2

        if ids1 is None or ids2 is None:
            return None

        # Find common IDs
        common_ids = np.intersect1d(ids1, ids2)

        if len(common_ids) < self.min_corners:
            logger.warning(f"Only {len(common_ids)} common corners found (need {self.min_corners})")
            return None

        # Get object points for common IDs
        return objp[common_ids].astype(np.float32)

    def process_camera_pair(
        self,
        cam1: Optional[int] = None,
        cam2: Optional[int] = None,
        progress_callback=None,
        save_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """Process a camera pair for stereo calibration with ChArUco ID matching.

        This overrides the base class method to handle ChArUco-specific
        corner ID matching between cameras.

        Parameters
        ----------
        cam1 : int, optional
            First camera number
        cam2 : int, optional
            Second camera number
        progress_callback : callable, optional
            Progress callback
        save_visualizations : bool
            Whether to save visualizations

        Returns
        -------
        dict
            Calibration result
        """
        # Use default camera pair if not specified
        cam1 = cam1 if cam1 is not None else self.camera_pair[0]
        cam2 = cam2 if cam2 is not None else self.camera_pair[1]

        logger.info(f"Processing ChArUco stereo pair: Camera {cam1} and Camera {cam2}")

        output_dir = self._get_output_dir(cam1, cam2)

        # Get number of calibration images from config
        if self._config is None:
            return {'success': False, 'error': 'Config required for stereo calibration'}

        num_frame_pairs = self._config.calibration_image_count
        if num_frame_pairs == 0:
            return {'success': False, 'error': 'No calibration images found (calibration_image_count=0)'}

        logger.info(f"Processing {num_frame_pairs} calibration frame pairs")

        # Collect calibration data
        objpoints = []
        imgpoints1 = []
        imgpoints2 = []
        successful_pairs = []
        indices_data = {}
        image_size = None

        board, _ = self.detector

        processed_count = 0

        # Iterate through image indices (1-based) using centralized reader
        for img_index in range(1, num_frame_pairs + 1):
            filename = f"frame_{img_index:05d}"

            # Use centralized image reader (handles all formats: standard, .set, .im7, .cine)
            img1 = self._read_calibration_image_centralized(camera=cam1, img_index=img_index)
            img2 = self._read_calibration_image_centralized(camera=cam2, img_index=img_index)

            if img1 is None:
                if img_index == 1:
                    return {'success': False, 'error': f'Could not read first image for camera {cam1}'}
                # For containers, None means we've reached the end
                if self._is_container_format():
                    logger.info(f"Reached end of images at index {img_index}")
                    break
                continue
            if img2 is None:
                if img_index == 1:
                    return {'success': False, 'error': f'Could not read first image for camera {cam2}'}
                if self._is_container_format():
                    break
                continue

            processed_count += 1

            if image_size is None:
                image_size = img1.shape[:2][::-1]

            # Detect ChArUco in both images
            found1, corners1, ids1 = self.detect_pattern(img1)
            found2, corners2, ids2 = self.detect_pattern(img2)

            if not found1 or not found2:
                logger.debug(f"ChArUco not found in pair {filename}")
                continue

            # Find common corner IDs
            common_ids = np.intersect1d(ids1, ids2)

            if len(common_ids) < self.min_corners:
                logger.warning(f"Only {len(common_ids)} common corners in {filename}")
                continue

            # Get indices of common IDs in each detection
            idx1 = [np.where(ids1 == cid)[0][0] for cid in common_ids]
            idx2 = [np.where(ids2 == cid)[0][0] for cid in common_ids]

            # Extract matched corners
            matched_corners1 = corners1[idx1].astype(np.float32)
            matched_corners2 = corners2[idx2].astype(np.float32)

            # Get object points for common IDs
            obj_pts = board.getChessboardCorners()[common_ids].astype(np.float32)

            # Add to calibration data
            objpoints.append(obj_pts)
            imgpoints1.append(matched_corners1)
            imgpoints2.append(matched_corners2)

            frame_idx = len(successful_pairs) + 1
            successful_pairs.append(filename)

            # Store indices data
            indices_data[frame_idx] = {
                'grid_points_cam1': matched_corners1,
                'grid_points_cam2': matched_corners2,
                'corner_ids': common_ids,
                'all_corners_cam1': corners1,
                'all_corners_cam2': corners2,
                'all_ids_cam1': ids1,
                'all_ids_cam2': ids2,
                'object_points': obj_pts,
                'frame_index': frame_idx,
                'original_filename': filename,
            }

            # Save visualizations
            if save_visualizations:
                self._save_detection_visualization(
                    img1, matched_corners1, cam1, frame_idx, output_dir, filename
                )
                self._save_detection_visualization(
                    img2, matched_corners2, cam2, frame_idx, output_dir, filename
                )

            logger.info(f"Processed pair {filename}: {len(common_ids)} matched corners")

            if progress_callback:
                progress_callback({
                    'progress': min(int((processed_count / max(num_frame_pairs, 1)) * 70), 70),
                    'stage': 'detecting',
                    'processed_pairs': processed_count,
                    'valid_pairs': len(successful_pairs),
                    'total_pairs': num_frame_pairs,
                })

        if len(successful_pairs) < 3:
            return {
                'success': False,
                'error': f'Insufficient valid image pairs: {len(successful_pairs)} (need >= 3)'
            }

        logger.info(f"Using {len(successful_pairs)} image pairs for ChArUco calibration")

        if progress_callback:
            progress_callback({
                'progress': 70,
                'stage': 'calibrating',
                'processed_pairs': processed_count,
                'valid_pairs': len(successful_pairs),
                'total_pairs': num_frame_pairs,
            })

        # Perform stereo calibration
        assert image_size is not None, "image_size should be set after processing pairs"
        try:
            calibration_result = self._perform_stereo_calibration(
                objpoints, imgpoints1, imgpoints2, image_size
            )
        except cv2.error as e:
            return {'success': False, 'error': f'OpenCV calibration failed: {e}'}
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return {'success': False, 'error': str(e)}

        if progress_callback:
            progress_callback({
                'progress': 90,
                'stage': 'saving',
                'processed_pairs': processed_count,
                'valid_pairs': len(successful_pairs),
                'total_pairs': num_frame_pairs,
            })

        # Save results
        model_path = self._save_stereo_results(
            cam1, cam2, calibration_result, successful_pairs, image_size, indices_data
        )

        if progress_callback:
            progress_callback({
                'progress': 100,
                'stage': 'complete',
                'processed_pairs': processed_count,
                'valid_pairs': len(successful_pairs),
                'total_pairs': num_frame_pairs,
            })

        return {
            'success': True,
            'stereo_rms_error': calibration_result['stereo_rms_error'],
            'cam1_rms_error': calibration_result['cam1_rms_error'],
            'cam2_rms_error': calibration_result['cam2_rms_error'],
            'num_pairs_used': len(successful_pairs),
            'model_path': str(model_path),
            'relative_angle_deg': calibration_result['relative_angle_deg'],
        }


def main():
    """Main entry point using hardcoded configuration.

    Updates config.yaml with the hardcoded settings, then runs stereo calibration.
    When USE_CONFIG_DIRECTLY=True, loads settings from existing config.yaml instead.
    """
    if USE_CONFIG_DIRECTLY:
        # Load settings directly from existing config.yaml
        logger.info("Loading settings directly from config.yaml (USE_CONFIG_DIRECTLY=True)")
        config = get_config()
    else:
        # Apply CLI settings to config.yaml so centralized loaders work correctly
        config = apply_cli_settings_to_config()

    # Create calibrator using config - all settings are now in config.yaml
    calibrator = StereoCharucoCalibrator(config=config)
    calibrator.run()


if __name__ == "__main__":
    main()
