"""calibration.detection — board detectors behind one interface.

Each detector answers only "what features are where" (image-down pixel points +
board-local mm points + a board-grid mapping). It never decides the world origin
or axis signs — that is ``world_frame``'s job. This separation is what lets one
mechanism (click origin/+X/+Y) serve every board type.
"""

from .base import DetectionResult, BoardDetector  # noqa: F401

__all__ = ["DetectionResult", "BoardDetector"]
