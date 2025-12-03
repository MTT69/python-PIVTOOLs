"""Stereo reconstruction and calibration module.

This module provides stereo camera calibration and 3D reconstruction functionality.

Classes
-------
BaseStereoCalibrator
    Abstract base class for stereo calibration
StereoPinholeCalibrator
    Stereo calibration using circle grid (pinhole) detection
StereoCharucoCalibrator
    Stereo calibration using ChArUco board detection
StereoReconstructor
    3D velocity reconstruction from stereo PIV data
"""

from .stereo_calibration_base import BaseStereoCalibrator
from .stereo_pinhole_calibration_production import StereoPinholeCalibrator
from .stereo_charuco_calibration_production import StereoCharucoCalibrator
from .stereo_reconstruction_production import StereoReconstructor

__all__ = [
    "BaseStereoCalibrator",
    "StereoPinholeCalibrator",
    "StereoCharucoCalibrator",
    "StereoReconstructor",
]
