"""calibration2.detection.dotboard — dot-grid detector behind the unified interface.

Wraps the production BFS + homography detector
(``pivtools_gui.calibration.grid_detection.detect_grid_automatic``), which is the
hard-won perspective-robust dot finder (180/180 on tilted boards). We reuse it
verbatim and only repackage its output into a ``DetectionResult``.

Note on indexing: the BFS grid indices are internally consistent *per detection*
(origin at the min corner, orientation from the local neighbour walk) but are NOT
globally consistent across views/cameras. That is fine for bundled intrinsics
(each view solves its own extrinsic) and for the world frame on the datum view
(clicks resolve against that view's indices). For stereo, the user clicks the same
physical origin/+X/+Y on each camera, anchoring both to the shared world frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from pivtools_gui.calibration.grid_detection import detect_grid_automatic

from .base import DetectionResult


@dataclass
class DotboardParams:
    dot_spacing_mm: float = 15.0
    k_neighbors: int = 9


class DotboardDetector:
    """Detect a dot grid -> DetectionResult."""

    board_type = "dotboard"

    def __init__(self, params: DotboardParams):
        self.params = params

    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> DetectionResult:
        ok, grid, info = detect_grid_automatic(
            image, mask=mask,
            grid_spacing_mm=self.params.dot_spacing_mm,
            k_neighbors=self.params.k_neighbors,
        )
        if not ok or grid is None:
            return DetectionResult(
                success=False, board_type=self.board_type,
                image_points=np.empty((0, 2)), board_local_points=np.empty((0, 3)),
                diagnostics=info,
            )

        centers = np.asarray(grid["centers"], dtype=np.float64).reshape(-1, 2)
        gi = np.asarray(grid["grid_indices"], dtype=np.int64).reshape(-1, 2)
        sp = float(self.params.dot_spacing_mm)
        board_local = np.column_stack(
            [gi[:, 0] * sp, gi[:, 1] * sp, np.zeros(len(gi))]
        ).astype(np.float64)

        board_to_pixel = None
        if len(gi) >= 4:
            H, _ = cv2.findHomography(
                gi.astype(np.float32), centers.astype(np.float32), method=0
            )
            board_to_pixel = H

        return DetectionResult(
            success=True,
            board_type=self.board_type,
            image_points=centers,
            board_local_points=board_local,
            grid_indices=gi,
            point_ids=None,
            board_to_pixel=board_to_pixel,
            spacing_mm=sp,
            diagnostics={k: v for k, v in grid.items() if np.isscalar(v)},
        )
