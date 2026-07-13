"""Stereo reconstruction and calibration module.

This module provides stereo camera calibration and 3D reconstruction functionality.

Classes
-------
BaseStereoCalibrator
    Abstract base class for stereo calibration
StereoDotboardCalibrator
    Stereo calibration using circle grid (dotboard) detection
StereoCharucoCalibrator
    Stereo calibration using ChArUco board detection
PinholeCamera
    Pinhole camera model for self-calibration
SelfCalibrationResult
    Result container for self-calibration

Functions
---------
run_self_calibration
    Iterative stereo self-calibration (Wieneke 2005)
compute_dewarp_maps
    Build remap tables for dewarping camera images onto world plane
estimate_pixel_scale
    Estimate native pixel scale (mm/px) from camera pair
"""

from .self_calibration import (
    PinholeCamera,
    SelfCalibrationResult,
    compute_dewarp_maps,
    estimate_pixel_scale,
    run_self_calibration,
)
from .stereo_calibration_base import BaseStereoCalibrator
from .stereo_charuco_calibration_production import StereoCharucoCalibrator
from .stereo_dotboard_calibration_production import StereoDotboardCalibrator

__all__ = [
    "BaseStereoCalibrator",
    "StereoDotboardCalibrator",
    "StereoCharucoCalibrator",
    "PinholeCamera",
    "SelfCalibrationResult",
    "run_self_calibration",
    "compute_dewarp_maps",
    "estimate_pixel_scale",
]
