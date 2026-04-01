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

from pivtools_gui.calibration.calibration_io import (
    ARUCO_DICT_MAP,
    create_charuco_detector,
)
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
    datum_camera : int
        Which camera defines the coordinate system origin (1 or 2, default: 1)
    datum_frame : int
        Which calibration image defines the world coordinate origin (1-based, default: 1)
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
        # Stereo-specific params
        datum_camera: int = 1,
        datum_frame: int = 1,
        # Base class params
        source_dir: Optional[Union[str, Path]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        camera_pair: Optional[List[int]] = None,
        file_pattern: Optional[str] = None,
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
            # Get stereo-specific params from stereo_charuco config
            stereo_cfg = config.stereo_charuco_calibration
            dt = stereo_cfg.get('dt', dt)
            datum_camera = stereo_cfg.get('datum_camera', datum_camera)
            datum_frame = stereo_cfg.get('datum_frame', datum_frame)

        self.squares_h = squares_h
        self.squares_v = squares_v
        self.square_size = square_size
        self.marker_ratio = marker_ratio
        self.aruco_dict_name = aruco_dict
        self.min_corners = min_corners
        self.datum_camera = datum_camera
        self.datum_frame = datum_frame

        super().__init__(
            config=config,
            source_dir=source_dir,
            base_dir=base_dir,
            camera_pair=camera_pair,
            file_pattern=file_pattern,
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
        return create_charuco_detector(
            self.squares_h, self.squares_v, self.square_size,
            self.marker_ratio, self.aruco_dict_name,
        )

    def detect_pattern(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Detect ChArUco corners in image.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or color)

        Returns
        -------
        tuple
            (found: bool, corners: np.ndarray or None, ids: np.ndarray or None, info: dict or None)
            corners shape: (N, 2) if found
            ids shape: (N,) if found
            info contains board_params for figure generation
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            if image.shape[-1] == 1:
                gray = image[:, :, 0]  # Squeeze singleton channel
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # ArUco detector requires uint8 input
        if gray.dtype != np.uint8:
            gmin, gmax = float(gray.min()), float(gray.max())
            if gmax > gmin:
                gray = ((gray.astype(np.float64) - gmin) / (gmax - gmin) * 255).astype(np.uint8)
            else:
                gray = np.zeros(gray.shape, dtype=np.uint8)

        board, detector = self.detector
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)

        # Build info dict for figure generation
        info = {
            'board_params': {
                'squares_h': self.squares_h,
                'squares_v': self.squares_v,
                'square_size': self.square_size,
                'square_size_mm': self.square_size * 1000,
                'marker_ratio': self.marker_ratio,
            },
        }

        if ids is None or len(corners) < self.min_corners:
            return False, None, None, info

        # Reshape corners from (N, 1, 2) to (N, 2)
        corners_2d = corners.reshape(-1, 2).astype(np.float32)
        ids_flat = ids.flatten()

        return True, corners_2d, ids_flat, info

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
            'datum_camera': self.datum_camera,
            'datum_frame': self.datum_frame,
        }

    def _match_object_points(
        self,
        objp: np.ndarray,
        result1: Tuple,
        result2: Tuple,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Match object points between two cameras using corner IDs.

        Finds the intersection of detected corner IDs and returns matched
        object points and image points for both cameras, matching the
        3-tuple contract used by StereoDotboardCalibrator.

        Parameters
        ----------
        objp : np.ndarray
            Full object points array from board.getChessboardCorners()
        result1 : tuple
            Detection result from camera 1: (found, corners, ids, info)
        result2 : tuple
            Detection result from camera 2: (found, corners, ids, info)

        Returns
        -------
        tuple or None
            (obj_pts, img_pts_1, img_pts_2) with matched points only
        """
        corners1 = result1[1]
        ids1 = result1[2]
        corners2 = result2[1]
        ids2 = result2[2]

        if ids1 is None or ids2 is None:
            return None

        # Find common IDs
        common_ids = np.intersect1d(ids1, ids2)

        if len(common_ids) < self.min_corners:
            logger.warning(f"Only {len(common_ids)} common corners found (need {self.min_corners})")
            return None

        # Get indices of common IDs in each detection
        idx1 = [np.where(ids1 == cid)[0][0] for cid in common_ids]
        idx2 = [np.where(ids2 == cid)[0][0] for cid in common_ids]

        matched_corners1 = corners1[idx1].astype(np.float32)
        matched_corners2 = corners2[idx2].astype(np.float32)
        obj_pts = objp[common_ids].astype(np.float32)

        return obj_pts, matched_corners1, matched_corners2


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
