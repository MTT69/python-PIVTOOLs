"""calibration.detection.stepped — dual-level stepped dotboard detector.

A stepped board carries dots on two parallel Z-planes (peak / trough) separated by
a machined ``step_height_mm``; the trough grid is interleaved by half a dot spacing
in x and y. This detector separates the levels WALK-FIRST: the BFS grid walk runs
directly on the full mixed blob set (with k=9 direction finding — see
``stepped_levels.find_grid_vectors``), locks onto one level's lattice and rejects
the other level's dots as off-lattice; the remainder is walked again for the second
level. The two levels are then stitched into ONE pose-local grid frame and
repackaged into the unified ``DetectionResult``. There is no explicit row/level
clustering — the v1 row-parity separation was removed 2026-07-28 because steep
off-axis views (image roll + step parallax) break its evenly-spaced-alternating-rows
assumption.

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
from loguru import logger
from scipy.spatial import cKDTree

from .base import DetectionResult
from .grid_detection import detect_dotboard_blobs, to_grayscale_2d
from .stepped_levels import (
    SteppedBoardSpec,
    run_single_level_detection,
    stitch_levels_pose_local,
)

# Direction-finding neighbourhood for the walk passes. k=9 is required on the
# full MIXED set (pass 1): each dot's 4 nearest neighbours are mostly cross-level
# diagonal partners, and only k=9 reaches the true same-level lattice neighbours.
# Pass 2's remainder is a single level plus stragglers, where k=9 is the planar
# default and equally valid.
DIRECTION_K = 9

# Fraction of the blob set pass 1 may consume before we conclude the walk tracked
# the combined two-level checkerboard with a diagonal basis instead of one level.
# One level holds at most ~55% of a stepped board's dots (n x n peak vs
# (n-1) x (n-1) trough), so >70% means both levels were walked. This happens near
# normal incidence, where no parallax smears the diagonal direction-histogram
# peaks and they can beat the lattice peaks.
COMBINED_LATTICE_FRACTION = 0.70

# A blob within this fraction of the pass-1 blob-set NN spacing of a walked
# centre is the same physical dot (pass-1 centers include template-rescued
# positions absent from the raw blob list). The nearest cross-level dot sits at
# ~1.0x the mixed-set NN spacing by definition, so 0.25x cannot eat pass-2 dots.
CONSUMED_RADIUS_FRACTION = 0.25


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
        flat_field = blob_info.get("flat_field")

        # Stage 2: walk-first level extraction — pass 1 on the full mixed set.
        # The BFS walk steps by full lattice vectors, so it extracts exactly the
        # seed dot's level; the other level sits near half-lattice positions and
        # loses every predicted-position candidate contest.
        level_a = run_single_level_detection(
            blobs, gray_uint8, flat_field=flat_field, k=DIRECTION_K
        )
        if level_a is None:
            return self._fail("no level produced a usable grid", blob_info)

        # Combined-lattice guard: if pass 1 consumed most of the blob set, the
        # direction finder locked onto the cross-level diagonals and the walk
        # tracked the combined checkerboard. The diagonal basis (p1, p2) relates
        # to the same-level lattice basis by L1 = p1 + p2, L2 = p1 - p2 — re-walk
        # with the derived basis injected.
        if level_a["n_points"] > COMBINED_LATTICE_FRACTION * len(blobs):
            v1 = np.asarray(level_a["vec1"], dtype=np.float64)
            v2 = np.asarray(level_a["vec2"], dtype=np.float64)
            lat1, lat2 = v1 + v2, v1 - v2
            # Match _find_grid_directions' output convention (vec1 = x-dominant
            # forced +x, vec2 forced +y): the stitch and the calibrator's datum
            # anchoring assume both levels' BFS bases share one convention, and
            # pass 2's basis is convention-normalised by direction finding.
            if abs(lat2[0]) > abs(lat1[0]):
                lat1, lat2 = lat2, lat1
            if lat1[0] < 0:
                lat1 = -lat1
            if lat2[1] < 0:
                lat2 = -lat2
            logger.info(
                f"Stepped pass 1 consumed {level_a['n_points']}/{len(blobs)} "
                "blobs — diagonal basis suspected, re-walking with derived "
                "lattice basis"
            )
            level_a = run_single_level_detection(
                blobs, gray_uint8, flat_field=flat_field, vectors=(lat1, lat2)
            )
            if level_a is None:
                return self._fail(
                    "combined-lattice re-walk produced no usable grid", blob_info
                )

        # Pass 2: remove pass-1 dots (template-rescued centres included) and walk
        # the remainder for the second level.
        tree_a = cKDTree(level_a["centers"])
        dist_to_a, _ = tree_a.query(blobs)
        remaining = blobs[
            dist_to_a > CONSUMED_RADIUS_FRACTION * level_a["spacing_px"]
        ]
        level_b = (
            run_single_level_detection(
                remaining, gray_uint8, flat_field=flat_field, k=DIRECTION_K
            )
            if len(remaining) >= 9
            else None
        )

        # Stage 3: stitch the two levels into one pose-local frame (falls back to
        # a degraded single-level result when pass 2 found nothing).
        stitched = stitch_levels_pose_local(level_a, level_b, self._board)

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
        # persisted to the sidecar) — NOT diagnostics. 'a'/'b' track the walk passes
        # ('a' = pass 1 = stitched ``source_level`` 'A'); which physical face lands in
        # which slot is per-pose arbitrary, resolved later by the user's clicks.
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
