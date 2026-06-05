"""calibration2.detection.charuco — ChArUco detector behind the unified interface.

Wraps ``cv2.aruco.CharucoDetector``. Converts object points from metres (the
ChArUco board native unit) to millimetres (the calibration2 contract), and derives
integer (col, row) grid indices from the chessboard corner id so the world-frame
resolver can treat charuco exactly like a dot grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import DetectionResult


@dataclass
class CharucoParams:
    squares_h: int = 10           # number of squares horizontally
    squares_v: int = 7            # number of squares vertically
    square_size_m: float = 0.030  # metres (ChArUco native unit)
    marker_ratio: float = 0.75    # marker_size / square_size
    aruco_dict: str = "DICT_4X4_1000"
    min_corners: int = 6

    @property
    def square_size_mm(self) -> float:
        return self.square_size_m * 1000.0

    @property
    def interior_cols(self) -> int:
        return self.squares_h - 1

    @property
    def interior_rows(self) -> int:
        return self.squares_v - 1


def build_board(params: CharucoParams):
    dict_id = getattr(cv2.aruco, params.aruco_dict)
    adict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (params.squares_h, params.squares_v),
        params.square_size_m,
        params.square_size_m * params.marker_ratio,
        adict,
    )
    return board, adict


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        mx = float(img.max()) if img.size else 1.0
        img = (img.astype(np.float64) / (mx if mx > 0 else 1.0) * 255.0).astype(np.uint8)
    return img


class CharucoBoardDetector:
    """Detect a ChArUco board -> DetectionResult (mm board points, grid indices)."""

    board_type = "charuco"

    def __init__(self, params: CharucoParams):
        self.params = params
        self.board, self.adict = build_board(params)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        # All chessboard corners in board frame, metres -> mm. Indexed by corner id.
        self._corners_mm = np.asarray(
            self.board.getChessboardCorners(), dtype=np.float64
        ).reshape(-1, 3) * 1000.0

    def _grid_index_for_id(self, cid: int) -> tuple:
        cols = self.params.interior_cols
        return (int(cid % cols), int(cid // cols))

    def detect(self, image: np.ndarray) -> DetectionResult:
        gray = _to_gray_u8(image)
        ch_corners, ch_ids, _m_corners, _m_ids = self.detector.detectBoard(gray)

        if ch_ids is None or len(ch_ids) < self.params.min_corners:
            return DetectionResult(
                success=False, board_type=self.board_type,
                image_points=np.empty((0, 2)), board_local_points=np.empty((0, 3)),
                diagnostics={"n_corners": 0 if ch_ids is None else int(len(ch_ids))},
            )

        ids = np.asarray(ch_ids, dtype=int).reshape(-1)
        image_points = np.asarray(ch_corners, dtype=np.float64).reshape(-1, 2)
        board_local = self._corners_mm[ids]
        grid_indices = np.array([self._grid_index_for_id(i) for i in ids], dtype=int)

        return DetectionResult(
            success=True,
            board_type=self.board_type,
            image_points=image_points,
            board_local_points=board_local,
            grid_indices=grid_indices,
            point_ids=ids,
            board_to_pixel=None,
            spacing_mm=self.params.square_size_mm,
            diagnostics={"n_corners": int(len(ids))},
        )
