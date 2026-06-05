"""calibration2.global_coords — N-camera global coordinate stitching (chain math).

Pure functions, no Flask / no config / no file I/O. The datum-chain algorithm is
ported from the v1 ``GlobalCoordinateAligner.precompute_camera_shifts``, but the
pixel->physical step uses calibration2's own model back-projection
(``apply.calibrate_coordinates``) — so there is NO v1 model format, and NO ``-y``
flip (the sign is carried by the fitted model, per the calibration2 contract).

Topology: the datum camera anchors the frame at a user-chosen pixel -> physical
point. Each overlap pair ``(cam_a, cam_b)`` carries the frame from an
already-placed ``cam_a`` to ``cam_b`` by matching the same physical feature seen
in both views. Pairs are processed in a stable order; a pair whose ``cam_a`` is
not yet placed is skipped (broken chain link) and reported.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .apply import calibrate_coordinates
from .record import MonoRecord


def _physical(record: MonoRecord, pixel, z: float, tx: float, ty: float) -> Tuple[float, float]:
    """Back-project a single pixel to world (X, Y) mm via the camera's model."""
    px = np.asarray(pixel, dtype=float).reshape(1, 2)
    w = calibrate_coordinates(record.camera_model, px, z, tx, ty)[0]
    if not np.all(np.isfinite(w)):
        raise ValueError("pixel back-projection failed (ray missed the sheet plane)")
    return float(w[0]), float(w[1])


def compute_camera_shifts(
    records_by_cam: Dict[int, MonoRecord],
    datum_camera: int,
    datum_pixel,
    datum_physical,
    overlap_pairs: List[dict],
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
) -> Dict[int, Tuple[float, float]]:
    """Per-camera (shift_x, shift_y) mm placing every camera in one global frame.

    ``records_by_cam`` maps camera number -> its mono model record. ``overlap_pairs``
    is a list of ``{camera_a, camera_b, pixel_on_a:[x,y], pixel_on_b:[x,y]}``.
    Returns ``{camera: (shift_x, shift_y)}``; cameras unreachable through the chain
    are absent. Raises ``KeyError`` if a referenced camera has no model record.
    """
    if datum_camera not in records_by_cam:
        raise KeyError(f"no model for datum camera {datum_camera}")
    datum_phys = (float(datum_physical[0]), float(datum_physical[1]))

    dc = _physical(records_by_cam[datum_camera], datum_pixel, z_world, tilt_x, tilt_y)
    shifts: Dict[int, Tuple[float, float]] = {
        datum_camera: (datum_phys[0] - dc[0], datum_phys[1] - dc[1]),
    }

    # Fixpoint pass: a pair places cam_b once cam_a is placed. Repeat until a full
    # pass adds nothing, so multi-hop chains resolve regardless of pair order. A pair
    # whose cam_a is never placed (broken link) is simply left out.
    ordered = sorted(overlap_pairs, key=lambda p: (int(p["camera_a"]), int(p["camera_b"])))
    progressed = True
    while progressed:
        progressed = False
        for pair in ordered:
            cam_a, cam_b = int(pair["camera_a"]), int(pair["camera_b"])
            pixel_a, pixel_b = pair.get("pixel_on_a"), pair.get("pixel_on_b")
            if pixel_a is None or pixel_b is None:
                continue  # incomplete pair
            if cam_a not in shifts or cam_b in shifts:
                continue  # parent not placed yet, or child already placed
            if cam_a not in records_by_cam or cam_b not in records_by_cam:
                raise KeyError(f"no model for camera {cam_a} or {cam_b} in pair")
            phys_a = _physical(records_by_cam[cam_a], pixel_a, z_world, tilt_x, tilt_y)
            sa = shifts[cam_a]
            phys_a_shifted = (phys_a[0] + sa[0], phys_a[1] + sa[1])
            phys_b = _physical(records_by_cam[cam_b], pixel_b, z_world, tilt_x, tilt_y)
            shifts[cam_b] = (phys_a_shifted[0] - phys_b[0], phys_a_shifted[1] - phys_b[1])
            progressed = True

    return shifts
