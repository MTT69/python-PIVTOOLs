"""
Calibration Views Router.

This module serves as the main entry point for all calibration-related routes.
It imports and aggregates blueprints from modular view files:

- scale_factor_views: Scale factor calibration (pixel to physical units)
- dotboard_views: Dotboard/planar calibration (grid detection, camera model)
- shared_views: Shared utilities (datum setting, status)
- charuco_views: ChArUco board calibration
- polynomial_views: Polynomial/DaVis XML calibration

Each sub-module uses the unified JobManager for background task tracking.
"""

from flask import Blueprint

from pivtools_gui.calibration.app.dotboard_views import dotboard_bp
from pivtools_gui.calibration.app.scale_factor_views import scale_factor_bp
from pivtools_gui.calibration.app.shared_views import calibration_shared_bp
from pivtools_gui.calibration.app.stereo_dotboard_views import stereo_dotboard_bp
from pivtools_gui.calibration.app.stereo_charuco_views import stereo_charuco_bp
from pivtools_gui.calibration.app.polynomial_views import polynomial_bp

# Main calibration blueprint that aggregates all sub-blueprints
calibration_bp = Blueprint("calibration", __name__)

# Register sub-blueprints
# Note: Flask's nested blueprint registration uses register_blueprint
calibration_bp.register_blueprint(scale_factor_bp)
calibration_bp.register_blueprint(dotboard_bp)
calibration_bp.register_blueprint(stereo_dotboard_bp)
calibration_bp.register_blueprint(stereo_charuco_bp)
calibration_bp.register_blueprint(calibration_shared_bp)
calibration_bp.register_blueprint(polynomial_bp)

# Try to import charuco views if available
try:
    from .charuco_views import charuco_bp
    calibration_bp.register_blueprint(charuco_bp)
except ImportError:
    # ChArUco views not yet implemented
    pass

__all__ = ["calibration_bp"]
