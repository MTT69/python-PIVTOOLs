#!/usr/bin/env python3
"""
stereo_dotboard_calibration_production.py

Production-ready stereo calibration using circle grid (dotboard) detection.
Uses OpenCV's findCirclesGrid with blob detection for calibration dot detection.

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
PATTERN_COLS = 10
PATTERN_ROWS = 10
DOT_SPACING_MM = 12.2222
ASYMMETRIC = False
ENHANCE_DOTS = False

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
    config.data["calibration"]["stereo_dotboard"]["camera_pair"] = CAMERA_PAIR
    config.data["calibration"]["stereo_dotboard"]["pattern_cols"] = PATTERN_COLS
    config.data["calibration"]["stereo_dotboard"]["pattern_rows"] = PATTERN_ROWS
    config.data["calibration"]["stereo_dotboard"]["dot_spacing_mm"] = DOT_SPACING_MM
    config.data["calibration"]["stereo_dotboard"]["asymmetric"] = ASYMMETRIC
    config.data["calibration"]["stereo_dotboard"]["enhance_dots"] = ENHANCE_DOTS

    # Save to disk so centralized loader picks up changes
    config.save()
    logger.info(f"Updated config.yaml with CLI settings")

    # Reload to ensure fresh state
    return reload_config()


class StereoDotboardCalibrator(BaseStereoCalibrator):
    """Stereo calibration using circle grid (dotboard) detection.

    This calibrator detects circle grids (symmetric or asymmetric) in calibration
    images and uses them for stereo camera calibration.

    Parameters
    ----------
    config : Config, optional
        Configuration object. If provided, settings are read from config.stereo_dotboard_calibration
    pattern_cols : int
        Number of columns in the calibration grid
    pattern_rows : int
        Number of rows in the calibration grid
    dot_spacing_mm : float
        Physical spacing between dots in millimeters
    asymmetric : bool
        Whether the grid is asymmetric (offset alternating rows)
    enhance_dots : bool
        Whether to apply dot enhancement for better detection
    **base_kwargs
        Additional arguments passed to BaseStereoCalibrator

    Example
    -------
    >>> calibrator = StereoDotboardCalibrator(
    ...     source_dir="/path/to/images",
    ...     base_dir="/path/to/output",
    ...     camera_pair=[1, 2],
    ...     pattern_cols=10,
    ...     pattern_rows=10,
    ...     dot_spacing_mm=28.89,
    ... )
    >>> result = calibrator.process_camera_pair()
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        # Pattern-specific params (from config.stereo_calibration or explicit)
        pattern_cols: int = 10,
        pattern_rows: int = 10,
        dot_spacing_mm: float = 28.89,
        asymmetric: bool = False,
        enhance_dots: bool = False,
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
        # Get pattern params from config.stereo_dotboard_calibration if available
        if config is not None:
            stereo_cfg = config.stereo_dotboard_calibration
            pattern_cols = stereo_cfg.get('pattern_cols', pattern_cols)
            pattern_rows = stereo_cfg.get('pattern_rows', pattern_rows)
            dot_spacing_mm = stereo_cfg.get('dot_spacing_mm', dot_spacing_mm)
            asymmetric = stereo_cfg.get('asymmetric', asymmetric)
            enhance_dots = stereo_cfg.get('enhance_dots', enhance_dots)
            dt = stereo_cfg.get('dt', dt)

        self.pattern_size = (pattern_cols, pattern_rows)
        self.dot_spacing_mm = dot_spacing_mm
        self.asymmetric = asymmetric
        self.enhance_dots = enhance_dots

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

    def _create_detector(self) -> cv2.SimpleBlobDetector:
        """Create optimized blob detector for circle grid detection.

        Returns
        -------
        cv2.SimpleBlobDetector
            Configured blob detector
        """
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 200
        params.maxArea = 5000  # Increased for varying depths
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minThreshold = 0
        params.maxThreshold = 255
        params.thresholdStep = 5
        return cv2.SimpleBlobDetector_create(params)

    def _enhance_dots_image(self, img: np.ndarray, fixed_radius: int = 9) -> np.ndarray:
        """Enhance white dots in calibration image for better detection.

        Parameters
        ----------
        img : np.ndarray
            Grayscale image
        fixed_radius : int
            Radius to draw for enhanced dots

        Returns
        -------
        np.ndarray
            Enhanced image
        """
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = img.copy()
        for cnt in contours:
            (x, y), _ = cv2.minEnclosingCircle(cnt)
            center = (int(round(x)), int(round(y)))
            cv2.circle(output, center, fixed_radius, (255,), -1)
        return output

    def detect_pattern(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Detect circle grid in image.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or color)

        Returns
        -------
        tuple
            (found: bool, centers: np.ndarray or None)
            centers shape: (N, 2) if found
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply dot enhancement if requested
        if self.enhance_dots:
            gray = self._enhance_dots_image(gray)

        # Grid detection flags
        flags = cv2.CALIB_CB_ASYMMETRIC_GRID if self.asymmetric else cv2.CALIB_CB_SYMMETRIC_GRID

        # Try both original and inverted images
        for test_img, label in [(gray, "Original"), (255 - gray, "Inverted")]:
            found, centers = cv2.findCirclesGrid(
                test_img, self.pattern_size, flags=flags, blobDetector=self.detector
            )

            if found:
                centers = centers.reshape(-1, 2).astype(np.float32)

                # Subpixel refinement using cornerSubPix
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                win_size = (11, 11)
                zero_zone = (-1, -1)

                centers_refined = cv2.cornerSubPix(
                    gray if label == "Original" else (255 - gray),
                    centers.reshape(-1, 1, 2),
                    win_size,
                    zero_zone,
                    criteria
                )

                return True, centers_refined.reshape(-1, 2).astype(np.float32)

        return False, None

    def make_object_points(self) -> np.ndarray:
        """Create 3D object points for calibration grid.

        Returns
        -------
        np.ndarray
            Shape (N, 3) with Z=0 for all points
        """
        cols, rows = self.pattern_size
        objp = []
        for i in range(rows):
            for j in range(cols):
                if self.asymmetric:
                    x = j * self.dot_spacing_mm + (0.5 * self.dot_spacing_mm if (i % 2 == 1) else 0.0)
                else:
                    x = j * self.dot_spacing_mm
                y = i * self.dot_spacing_mm
                objp.append([x, y, 0.0])
        return np.array(objp, dtype=np.float32)

    def get_pattern_params(self) -> Dict[str, Any]:
        """Get pattern-specific parameters for saving to output files.

        Returns
        -------
        dict
            Pattern parameters
        """
        return {
            'pattern_type': 'circle_grid',
            'pattern_cols': self.pattern_size[0],
            'pattern_rows': self.pattern_size[1],
            'dot_spacing_mm': self.dot_spacing_mm,
            'asymmetric': self.asymmetric,
            'enhance_dots': self.enhance_dots,
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
    calibrator = StereoDotboardCalibrator(config=config)
    calibrator.run()


if __name__ == "__main__":
    main()
