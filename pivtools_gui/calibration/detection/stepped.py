"""calibration.detection.stepped — dual-level stepped dotboard detector.

A stepped board carries dots on two parallel Z-planes (peak / trough) separated by
a machined ``step_height_mm``; the trough grid is interleaved by half a dot spacing
in x and y. This detector separates the two interleaved levels, runs a per-level BFS
grid walk, and stitches them back into ONE pose-local grid frame, then repackages
the result into the unified ``DetectionResult``.

Coordinate contract (see ``detection.base`` + the calibration invariants):
- ``image_points`` are image-down pixels (0-based, y-down), both levels concatenated.
- ``grid_indices`` are integer (col, row) in the stitched pose-local frame; the two
  levels share this frame but the trough level is physically offset by half a
  spacing (encoded in ``board_local_points``, NOT in the integer index).
- ``board_local_points`` carry the board's own two-Z geometry under a NEUTRAL
  convention: the larger (reference) level at z=0, the other level at
  z=-step_height and +level_offset in x and y. The peak/trough -> absolute-Z
  decision is the user's (``clicked_level`` / per-pose labels) and is applied later
  by the stepped calibrator — the detector never guesses which face is which.
- ``diagnostics`` carry the per-level breakdown (``level_a`` / ``level_b`` centers +
  indices, for the GUI overlay and click-to-label), a per-point ``level_labels``
  array, and the stitch audit (``stitch``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .base import DetectionResult
from .grid_detection import detect_dotboard_blobs, to_grayscale_2d
from .stepped_levels import (
    SteppedBoardSpec,
    cluster_into_rows,
    run_single_level_detection,
    separate_levels,
    stitch_levels_pose_local,
)


@dataclass
class SteppedParams:
    dot_spacing_mm: float = 15.0
    step_height_mm: float = 3.0
    board_thickness_mm: float = 14.8
    level_offset_mm: Optional[float] = None  # defaults to dot_spacing_mm / 2

    def board(self) -> SteppedBoardSpec:
        return SteppedBoardSpec(
            dot_spacing_mm=self.dot_spacing_mm,
            step_height_mm=self.step_height_mm,
            level_offset_mm=self.level_offset_mm,
            board_thickness_mm=self.board_thickness_mm,
        )


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Grayscale + scale to uint8 by peak value (matches v1 detect_single_camera)."""
    gray = to_grayscale_2d(image)
    gray_f = gray.astype(np.float64)
    vmax = float(gray_f.max())
    if vmax > 0:
        return (gray_f / vmax * 255).astype(np.uint8)
    return np.zeros_like(gray, dtype=np.uint8)


class SteppedDetector:
    """Detect a dual-level stepped dot grid -> DetectionResult."""

    board_type = "stepped"

    def __init__(self, params: SteppedParams):
        self.params = params
        self._board = params.board()

    def detect(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> DetectionResult:
        gray_uint8 = _to_uint8(image)

        # Stage 1: blob detection (both polarities, best by count).
        blobs, blob_info = detect_dotboard_blobs(gray_uint8, mask=mask)
        if len(blobs) < 9:
            return self._fail("blob detection found < 9 dots", blob_info)

        # Stage 2: row-parity level separation.
        tree = cKDTree(blobs)
        nn_dists = tree.query(blobs, k=2)[0]
        spacing_px = float(np.median(nn_dists[:, 1]))
        row_labels, row_y_values = cluster_into_rows(blobs, spacing_px)
        level_info = separate_levels(blobs, row_labels, row_y_values)
        centers_clean = level_info["centers"]
        mask_a = level_info["mask_level_A"]
        mask_b = level_info["mask_level_B"]

        # Stage 3: per-level grid detection (k=5).
        flat_field = blob_info.get("flat_field")
        level_a = run_single_level_detection(
            centers_clean[mask_a], gray_uint8, flat_field=flat_field
        )
        level_b = run_single_level_detection(
            centers_clean[mask_b], gray_uint8, flat_field=flat_field
        )
        if level_a is None and level_b is None:
            return self._fail("no level produced a usable grid", blob_info)

        # Stage 4: stitch the two levels into one pose-local frame.
        stitched = stitch_levels_pose_local(level_a, level_b, self._board)
        if stitched is None:
            return self._fail("level stitch failed", blob_info)

        return self._assemble(stitched, level_a, level_b, blob_info)

    # ------------------------------------------------------------------
    def _assemble(
        self,
        stitched: dict,
        level_a: Optional[dict],
        level_b: Optional[dict],
        blob_info: dict,
    ) -> DetectionResult:
        sp = float(self.params.dot_spacing_mm)
        step = float(self.params.step_height_mm)
        offset = float(self._board.level_offset_mm)

        ref = stitched["reference"]
        other = stitched.get("other")
        meta = stitched["metadata"]

        # Reference level: board canonical (col*sp, row*sp, 0).
        ref_c = np.asarray(ref["centers"], dtype=np.float64).reshape(-1, 2)
        ref_gi = np.asarray(ref["grid_indices"], dtype=np.int64).reshape(-1, 2)
        ref_local = np.column_stack(
            [ref_gi[:, 0] * sp, ref_gi[:, 1] * sp, np.zeros(len(ref_gi))]
        )
        ref_label = np.full(len(ref_gi), ref["source_level"], dtype="<U1")

        if other is not None:
            # Other level: +half-spacing interleave in x,y and -step in z (neutral
            # convention; absolute Z + peak/trough resolved later by the calibrator).
            oth_c = np.asarray(other["centers"], dtype=np.float64).reshape(-1, 2)
            oth_gi = np.asarray(other["grid_indices"], dtype=np.int64).reshape(-1, 2)
            oth_local = np.column_stack(
                [
                    oth_gi[:, 0] * sp + offset,
                    oth_gi[:, 1] * sp + offset,
                    np.full(len(oth_gi), -step),
                ]
            )
            oth_label = np.full(len(oth_gi), other["source_level"], dtype="<U1")
            image_points = np.vstack([ref_c, oth_c])
            grid_indices = np.vstack([ref_gi, oth_gi])
            board_local = np.vstack([ref_local, oth_local])
            level_labels = np.concatenate([ref_label, oth_label])
        else:
            image_points = ref_c
            grid_indices = ref_gi
            board_local = ref_local
            level_labels = ref_label

        # The two physical planes as raw per-level grid dicts (centers + grid_indices +
        # H + vec1/vec2). The calibrator anchors the datum to the fiducial clicks via the
        # per-level homographies and re-stitches every non-datum pose against the datum's
        # orientation, so this is a first-class fit input on the DetectionResult (it is
        # persisted to the sidecar) — NOT diagnostics. 'a'/'b' track the stable A/B labels
        # the stitched ``source_level`` flag carries (level_a is source_level 'A').
        level_data = {"a": level_a, "b": level_b}

        diagnostics = {
            "level_labels": level_labels.tolist(),
            "stitch": dict(meta),
            "image_mode": blob_info.get("image_mode"),
            "n_blobs_detected": blob_info.get("n_blobs_detected"),
            "_blob_info": blob_info,
        }

        return DetectionResult(
            success=True,
            board_type=self.board_type,
            image_points=image_points,
            board_local_points=board_local,
            grid_indices=grid_indices,
            point_ids=None,
            board_to_pixel=None,
            spacing_mm=sp,
            level_data=level_data,
            diagnostics=diagnostics,
        )

    def _fail(self, why: str, blob_info: dict) -> DetectionResult:
        return DetectionResult(
            success=False,
            board_type=self.board_type,
            image_points=np.empty((0, 2)),
            board_local_points=np.empty((0, 3)),
            diagnostics={
                "error": why,
                **{k: v for k, v in blob_info.items() if np.isscalar(v)},
            },
        )
