#!/usr/bin/env python3
"""
stereo_dotboard_calibration_production.py

Production-ready stereo calibration using automatic RANSAC-based grid detection.
No longer requires specifying pattern_cols and pattern_rows - grid dimensions
are automatically detected.

Saves results to: {BASE_DIR}/calibration/stereo_cam{N}_cam{M}/
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

from pivtools_core.config import Config, get_config, reload_config
from pivtools_core.image_handling.calibration_loader import get_calibration_frame_count

from pivtools_gui.calibration.grid_detection import (
    to_grayscale_2d,
    detect_grid_automatic,
)
from pivtools_gui.stereo_reconstruction.stereo_calibration_base import BaseStereoCalibrator


# ===================== CONFIGURATION VARIABLES =====================
# Set these variables for your calibration setup (CLI mode)

SOURCE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/stereo"
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/stereo/processed"
CAMERA_PAIR = [1, 2]
FILE_PATTERN = "planar_calibration_plate_%02d.tif"

# CAMERA_SUBFOLDERS: List of subfolder names for each camera (index matches camera number - 1).
#                    e.g., ["Cam1", "Cam2"] means camera 1 uses "Cam1/", camera 2 uses "Cam2/"
#                    Set to [] (empty list) for container formats or when images are in SOURCE_DIR directly.
CAMERA_SUBFOLDERS = ["Cam1", "Cam2"]

# CALIBRATION_SUBFOLDER: Subfolder within camera folders for calibration images.
#                        Leave empty "" to look directly in camera folders.
CALIBRATION_SUBFOLDER = "calibration"

# Grid pattern parameters
# NOTE: PATTERN_COLS and PATTERN_ROWS are no longer required - auto-detected
DOT_SPACING_MM = 12.2222  # Physical dot spacing (required)

# Number of calibration images to use (set to None to use all available)
NUM_CALIBRATION_IMAGES = None

# USE_CONFIG_DIRECTLY: If True, skip updating config.yaml with above parameters
# and load calibration settings directly from the existing config.yaml
USE_CONFIG_DIRECTLY = True

# ===================================================================


def apply_cli_settings_to_config() -> Config:
    """Update config.yaml with CLI-mode hardcoded settings."""
    config = get_config()

    config.data["paths"]["source_paths"] = [SOURCE_DIR]
    config.data["paths"]["base_paths"] = [BASE_DIR]
    config.data["paths"]["camera_subfolders"] = CAMERA_SUBFOLDERS
    config.data["paths"]["camera_count"] = len(CAMERA_PAIR)
    config.data["paths"]["camera_numbers"] = CAMERA_PAIR

    config.data["calibration"]["image_format"] = FILE_PATTERN
    config.data["calibration"]["subfolder"] = CALIBRATION_SUBFOLDER

    if NUM_CALIBRATION_IMAGES is not None:
        config.data["calibration"]["num_images"] = NUM_CALIBRATION_IMAGES
    else:
        config.save()
        config = reload_config()
        detected_count = get_calibration_frame_count(camera=CAMERA_PAIR[0], config=config)
        if detected_count > 0:
            config.data["calibration"]["num_images"] = detected_count
            logger.info(f"Auto-detected {detected_count} calibration images")

    # Stereo-specific params - no longer need pattern_cols/rows
    config.data["calibration"]["stereo_dotboard"]["camera_pair"] = CAMERA_PAIR
    config.data["calibration"]["stereo_dotboard"]["dot_spacing_mm"] = DOT_SPACING_MM
    config.data["calibration"]["stereo_dotboard"]["datum_camera"] = 1  # Default

    config.save()
    logger.info("Updated config.yaml with CLI settings")

    return reload_config()


class StereoDotboardCalibrator(BaseStereoCalibrator):
    """Stereo calibration using automatic RANSAC-based grid detection.

    Grid dimensions (cols, rows) are automatically detected - no need to specify.
    Handles variable grid sizes between cameras by finding intersection of detected points.

    Parameters
    ----------
    config : Config, optional
        Configuration object
    dot_spacing_mm : float
        Physical spacing between dots in millimeters (required)
    datum_camera : int
        Which camera defines the coordinate system origin (1 or 2, default: 1)
    datum_frame : int
        Which calibration image defines the world coordinate origin (1-based, default: 1)
    **base_kwargs
        Additional arguments passed to BaseStereoCalibrator

    Example
    -------
    >>> calibrator = StereoDotboardCalibrator(
    ...     source_dir="/path/to/images",
    ...     base_dir="/path/to/output",
    ...     camera_pair=[1, 2],
    ...     dot_spacing_mm=28.89,
    ... )
    >>> result = calibrator.process_camera_pair()
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        # Pattern-specific params
        dot_spacing_mm: float = 28.89,
        datum_camera: int = 1,
        datum_frame: int = 1,
        # Legacy params (ignored)
        pattern_cols: Optional[int] = None,
        pattern_rows: Optional[int] = None,
        asymmetric: bool = False,
        # Base class params
        source_dir: Optional[Union[str, Path]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        camera_pair: Optional[List[int]] = None,
        file_pattern: Optional[str] = None,
        camera_subfolders: Optional[List[str]] = None,
        source_path_idx: int = 0,
        dt: float = 1.0,
    ):
        # Get params from config if available
        if config is not None:
            stereo_cfg = config.stereo_dotboard_calibration
            dot_spacing_mm = stereo_cfg.get('dot_spacing_mm', dot_spacing_mm)
            datum_camera = stereo_cfg.get('datum_camera', datum_camera)
            datum_frame = stereo_cfg.get('datum_frame', datum_frame)
            dt = stereo_cfg.get('dt', dt)

        self.dot_spacing_mm = dot_spacing_mm
        self.datum_camera = datum_camera
        self.datum_frame = datum_frame

        # Track detected dimensions per frame
        self._detected_cols: Optional[int] = None
        self._detected_rows: Optional[int] = None

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

    def _create_detector(self):
        """No external detector needed — flat-field pipeline is internal."""
        return None

    def detect_pattern(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Detect grid using automatic RANSAC-based detection.

        Returns
        -------
        tuple
            (found: bool, centers: np.ndarray or None, grid_data: dict or None, info: dict or None)
            grid_data includes grid_indices, n_cols, n_rows for point matching
            info contains diagnostic data from detect_grid_automatic()
        """
        gray = to_grayscale_2d(image)

        success, grid_data, info = detect_grid_automatic(
            gray, mask=None, grid_spacing_mm=self.dot_spacing_mm,
        )

        if success and grid_data is not None:
            self._detected_cols = grid_data['n_cols']
            self._detected_rows = grid_data['n_rows']
            logger.debug(f"Auto-detected {grid_data['n_cols']}x{grid_data['n_rows']} grid with {len(grid_data['centers'])} points")
            return True, grid_data['centers'], grid_data, info

        return False, None, None, info

    def make_object_points(self) -> np.ndarray:
        """Create placeholder object points - actual points created dynamically."""
        # This is called by base class but we override matching
        # Return empty array - actual points created in _match_object_points
        return np.array([], dtype=np.float32)

    def make_object_points_dynamic(
        self,
        grid_indices: np.ndarray,
    ) -> np.ndarray:
        """Create 3D object points from grid indices."""
        obj_points = np.zeros((len(grid_indices), 3), dtype=np.float32)
        obj_points[:, 0] = grid_indices[:, 0] * self.dot_spacing_mm
        obj_points[:, 1] = grid_indices[:, 1] * self.dot_spacing_mm
        return obj_points

    def _match_object_points(
        self,
        objp: np.ndarray,
        result1: Tuple,
        result2: Tuple,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Match object points between cameras using grid indices.

        Finds intersection of detected grid positions and returns matched points.

        Returns
        -------
        tuple or None
            (obj_pts, img_pts_1, img_pts_2) with matched points only
        """
        # Extract grid data from detection results
        # result format: (found, centers, grid_data, info)
        if len(result1) < 3 or len(result2) < 3:
            return None

        grid_data_1 = result1[2]
        grid_data_2 = result2[2]

        if grid_data_1 is None or grid_data_2 is None:
            return None

        centers_1 = grid_data_1['centers']
        centers_2 = grid_data_2['centers']
        indices_1 = grid_data_1['grid_indices']
        indices_2 = grid_data_2['grid_indices']

        # Build lookup from grid position to index
        pos_to_idx_1 = {(gi[0], gi[1]): i for i, gi in enumerate(indices_1)}
        pos_to_idx_2 = {(gi[0], gi[1]): i for i, gi in enumerate(indices_2)}

        # Find common grid positions
        common_positions = set(pos_to_idx_1.keys()) & set(pos_to_idx_2.keys())

        if len(common_positions) < 9:
            logger.warning(f"Too few common grid points: {len(common_positions)}")
            return None

        # Build matched arrays
        obj_pts = []
        img_pts_1 = []
        img_pts_2 = []

        for pos in sorted(common_positions):
            col, row = pos
            idx_1 = pos_to_idx_1[pos]
            idx_2 = pos_to_idx_2[pos]

            obj_pts.append([col * self.dot_spacing_mm, row * self.dot_spacing_mm, 0.0])
            img_pts_1.append(centers_1[idx_1])
            img_pts_2.append(centers_2[idx_2])

        logger.info(f"Matched {len(obj_pts)} common grid points between cameras")

        return (
            np.array(obj_pts, dtype=np.float32),
            np.array(img_pts_1, dtype=np.float32),
            np.array(img_pts_2, dtype=np.float32),
        )

    def get_pattern_params(self) -> Dict[str, Any]:
        """Get pattern-specific parameters for saving."""
        return {
            'pattern_type': 'circle_grid_automatic',
            'detected_cols': self._detected_cols,
            'detected_rows': self._detected_rows,
            'dot_spacing_mm': self.dot_spacing_mm,
            'datum_camera': self.datum_camera,
            'datum_frame': self.datum_frame,
        }


def main():
    """Main entry point using hardcoded configuration."""
    if USE_CONFIG_DIRECTLY:
        logger.info("Loading settings directly from config.yaml (USE_CONFIG_DIRECTLY=True)")
        config = get_config()
    else:
        config = apply_cli_settings_to_config()

    calibrator = StereoDotboardCalibrator(config=config)
    calibrator.run()


if __name__ == "__main__":
    main()
