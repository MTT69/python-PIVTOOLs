"""calibration.world_frame — user-defined world frame from origin/+X/+Y clicks.

This is the keystone that lets one mechanism serve every board type. The user
clicks three points on camera 1 (origin, a +X point, a +Y point); each snaps to the
nearest DETECTED feature, and the world frame is built from the board's ORTHOGONAL
grid axes through the origin feature. The clicks only choose which grid axis is +X /
+Y and the sign — they can never introduce skew, so the axes are perpendicular by
construction (in board/world space). Generalised from the stepped board's
``assign_absolute_grid_indices`` (alignment of click vectors to grid axes).

Handedness note: +Z follows the right-hand rule from the chosen +X, +Y. If the user
picks a left-handed in-plane pair, the world frame is left-handed and the
out-of-plane (w / uz) sign follows that choice — this is honoured as the user's
explicit selection, not silently corrected.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .record import WorldFrame


def _snap(pts_px: np.ndarray, click_xy) -> int:
    """Index of the detected point nearest the click."""
    tree = cKDTree(np.asarray(pts_px, dtype=np.float64))
    _, idx = tree.query(np.asarray(click_xy, dtype=np.float64).reshape(2))
    return int(idx)


def resolve_world_frame(
    grid_indices: np.ndarray,
    image_points: np.ndarray,
    clicks: Optional[Dict[str, object]] = None,
) -> WorldFrame:
    """Resolve the world frame from clicks (or the documented default).

    Parameters
    ----------
    grid_indices : (N,2) integer (col, row)
    image_points : (N,2) image-down pixels (for snapping clicks)
    clicks : {'origin': [x,y], 'x_axis': [x,y], 'y_axis': [x,y]} in image-down px,
        or None for the default frame.

    Returns
    -------
    WorldFrame
    """
    gi = np.asarray(grid_indices, dtype=np.int64).reshape(-1, 2)

    if clicks is None:
        c0, r0 = int(gi[:, 0].min()), int(gi[:, 1].min())
        return WorldFrame(
            mode="default", swap_axes=False, col_sign=1, row_sign=1,
            origin_grid=np.array([c0, r0], dtype=np.float64),
        )

    io = _snap(image_points, clicks["origin"])
    ix = _snap(image_points, clicks["x_axis"])
    iy = _snap(image_points, clicks["y_axis"])
    g_o = gi[io].astype(np.int64)
    dx = gi[ix].astype(np.int64) - g_o
    dy = gi[iy].astype(np.int64) - g_o

    # Which grid axis (0=col, 1=row) does +X follow? +Y takes the other.
    ax_x = 0 if abs(dx[0]) >= abs(dx[1]) else 1
    ax_y = 1 - ax_x
    sx = int(np.sign(dx[ax_x])) or 1
    sy = int(np.sign(dy[ax_y])) or 1

    swap = ax_x == 1  # +X follows the row axis
    # col_sign = +X sign, row_sign = +Y sign (see WorldFrame docstring mapping).
    return WorldFrame(
        mode="clicks",
        origin_px=np.asarray(clicks["origin"], dtype=np.float64).reshape(2),
        x_axis_px=np.asarray(clicks["x_axis"], dtype=np.float64).reshape(2),
        y_axis_px=np.asarray(clicks["y_axis"], dtype=np.float64).reshape(2),
        swap_axes=swap, col_sign=sx, row_sign=sy,
        origin_grid=g_o.astype(np.float64),
    )


def resolve_world_frame_from_grid(origin_gi, x_axis_gi, y_axis_gi) -> WorldFrame:
    """Resolve the world frame directly from dot GRID indices (no pixel snapping).

    The headless / CLI analogue of clicking: name the origin dot and one dot along
    each of +X and +Y as ``(col, row)`` grid indices. Same orthogonal-grid-axis
    logic as the clicks path, so axes are perpendicular by construction. Use when
    there is no GUI to click on.

    Parameters
    ----------
    origin_gi, x_axis_gi, y_axis_gi : (col, row) integer grid indices.
    """
    g_o = np.asarray(origin_gi, dtype=np.int64).reshape(2)
    dx = np.asarray(x_axis_gi, dtype=np.int64).reshape(2) - g_o
    dy = np.asarray(y_axis_gi, dtype=np.int64).reshape(2) - g_o

    ax_x = 0 if abs(dx[0]) >= abs(dx[1]) else 1
    ax_y = 1 - ax_x
    sx = int(np.sign(dx[ax_x])) or 1
    sy = int(np.sign(dy[ax_y])) or 1
    swap = ax_x == 1
    return WorldFrame(
        mode="grid", swap_axes=swap, col_sign=sx, row_sign=sy,
        origin_grid=g_o.astype(np.float64),
    )


def apply_world_frame(
    grid_indices: np.ndarray, spacing_mm: float, wf: WorldFrame
) -> np.ndarray:
    """Map detected features' (col,row) -> world (X,Y,0) mm under a resolved frame.

    Works for any view/camera that shares the same grid indexing (charuco corner
    ids, or a board whose indexing is globally consistent).
    """
    gi = np.asarray(grid_indices, dtype=np.float64).reshape(-1, 2)
    if wf.origin_grid is not None:
        og = np.asarray(wf.origin_grid, dtype=np.float64).reshape(2)
    else:
        og = np.array([gi[:, 0].min(), gi[:, 1].min()])
    du = gi[:, 0] - og[0]
    dv = gi[:, 1] - og[1]
    sp = float(spacing_mm)
    if not wf.swap_axes:
        wx = wf.col_sign * du * sp
        wy = wf.row_sign * dv * sp
    else:
        wx = wf.col_sign * dv * sp
        wy = wf.row_sign * du * sp
    # Translate so the origin dot reads as the user-specified (X, Y) mm (default 0,0).
    if wf.origin_mm is not None:
        om = np.asarray(wf.origin_mm, dtype=np.float64).reshape(-1)
        if om.size >= 2:
            wx = wx + float(om[0])
            wy = wy + float(om[1])
    return np.column_stack([wx, wy, np.zeros(len(gi))])


def resolve_world_points(
    detection,
    clicks: Optional[Dict[str, object]] = None,
    spacing_mm: Optional[float] = None,
) -> Tuple[np.ndarray, WorldFrame]:
    """Convenience: resolve the frame for a detection and return its world points."""
    sp = spacing_mm if spacing_mm is not None else detection.spacing_mm
    if sp is None:
        raise ValueError("resolve_world_points: spacing_mm required")
    wf = resolve_world_frame(detection.grid_indices, detection.image_points, clicks)
    world = apply_world_frame(detection.grid_indices, sp, wf)
    return world, wf
