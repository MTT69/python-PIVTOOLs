"""Stereo self-calibration and pixel↔world projection.

The legacy calibrator classes (BaseStereoCalibrator and the dotboard/ChArUco
production scripts) were removed — stereo model generation lives in
``pivtools_gui.calibration``. This package keeps the two leaves that
production still uses:

Classes
-------
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

from .pixel_world import _pixels_to_world_mm
from .self_calibration import (
    PinholeCamera,
    SelfCalibrationResult,
    compute_dewarp_maps,
    estimate_pixel_scale,
    run_self_calibration,
)

__all__ = [
    "PinholeCamera",
    "SelfCalibrationResult",
    "run_self_calibration",
    "compute_dewarp_maps",
    "estimate_pixel_scale",
    "_pixels_to_world_mm",
]
