"""
Calibration Views Router.

This module serves as the main entry point for all calibration-related routes.
It imports and aggregates blueprints from modular view files:

- scale_factor_views: Scale factor calibration (pixel to physical units)
- pinhole_views: Pinhole/planar calibration (grid detection, camera model)
- shared_views: Shared utilities (datum setting, status)
- charuco_views: ChArUco board calibration (when available)

Each sub-module uses the unified JobManager for background task tracking.
"""

from flask import Blueprint

from .pinhole_views import pinhole_bp
from .scale_factor_views import scale_factor_bp
from .shared_views import calibration_shared_bp

# Main calibration blueprint that aggregates all sub-blueprints
calibration_bp = Blueprint("calibration", __name__)

# Register sub-blueprints
# Note: Flask's nested blueprint registration uses register_blueprint
calibration_bp.register_blueprint(scale_factor_bp)
calibration_bp.register_blueprint(pinhole_bp)
calibration_bp.register_blueprint(calibration_shared_bp)

# Try to import charuco views if available
try:
    from .charuco_views import charuco_bp
    calibration_bp.register_blueprint(charuco_bp)
except ImportError:
    # ChArUco views not yet implemented
    pass

__all__ = ["calibration_bp"]
