"""calibration.detection.base — the unified detection result + detector protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

import numpy as np


@dataclass
class DetectionResult:
    """What a board detector returns. One shape for every board type.

    Coordinate contract: ``image_points`` are image-down pixels (0-based, y-down);
    ``board_local_points`` are millimetres in the board's own canonical frame
    (col*spacing, row*spacing, 0) — NOT yet in the user world frame. The world
    frame is applied later by ``world_frame``.

    Attributes
    ----------
    success : whether a usable board was found
    board_type : 'dotboard' | 'charuco'
    image_points : (N,2) image-down pixels
    board_local_points : (N,3) mm, board canonical frame
    grid_indices : (N,2) integer (col, row) for every detected feature. Present for
        both dotboard (BFS) and charuco (derived from corner id). This is what the
        world-frame resolver operates on.
    point_ids : (N,) global feature ids (charuco corner ids) or None (dotboard).
        Globally consistent across cameras/views — used for stereo correspondence.
    board_to_pixel : (3,3) homography board-grid -> pixel, or None.
    spacing_mm : nominal feature spacing in mm.
    diagnostics : free-form detector diagnostics.
    """

    success: bool
    board_type: str
    image_points: np.ndarray
    board_local_points: np.ndarray
    grid_indices: Optional[np.ndarray] = None
    point_ids: Optional[np.ndarray] = None
    board_to_pixel: Optional[np.ndarray] = None
    spacing_mm: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image_points = np.asarray(self.image_points, dtype=np.float64).reshape(-1, 2)
        self.board_local_points = np.asarray(
            self.board_local_points, dtype=np.float64
        ).reshape(-1, 3)
        if self.grid_indices is not None:
            self.grid_indices = np.asarray(self.grid_indices).reshape(-1, 2)
        if self.point_ids is not None:
            self.point_ids = np.asarray(self.point_ids).reshape(-1)

    @property
    def n(self) -> int:
        return int(self.image_points.shape[0])


class BoardDetector(Protocol):
    """Detect a calibration board in one image."""

    board_type: str

    def detect(self, image: np.ndarray) -> DetectionResult: ...
