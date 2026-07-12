"""calibration.self_cal_frame — bake a self-cal sheet correction into the world frame.

Self-calibration (Wieneke 2005, ``calibration.self_cal``) recovers a laser-sheet
correction ``(z_offset, tilt_x, tilt_y)``: the true sheet is not the assumed Z=0
plane but the plane

    Z = z_offset + X*tan(tilt_y) + Y*tan(tilt_x)

in the calibration (clicked) world frame — the same plane equation as
``camera_model.back_project_to_plane`` (post-A1, pinhole and poly3d agree).

Two ways to use that correction. The legacy path keeps the three numbers and applies
them at reconstruction time (``regular_world_grid(..., z_world, tilt_x, tilt_y)``).
DaVis instead **redefines the world frame** so the corrected sheet becomes the new
Z=0 plane and bakes that redefinition into both cameras' extrinsics — no tilt fields
stored, intrinsics untouched. This module provides that frame redefinition as a pure
rigid transform so the CLI/GUI self-cal routes can rebake identically.

Conventions (load-bearing)
--------------------------
``g_corr`` maps NEW (sheet-aligned) world coords -> OLD (clicked) world coords::

    X_old = R_corr @ X_new + t_corr            # new Z=0 plane lands ON the sheet

so a camera that was ``X_cam = R @ X_old + t`` rebakes to::

    X_cam = R @ (R_corr @ X_new + t_corr) + t
          = (R @ R_corr) @ X_new + (R @ t_corr + t)
    R' = R @ R_corr ,   t' = R @ t_corr + t

``X_cam`` for a physical point is unchanged, so the cross-camera pose
``(R_stereo, T_stereo) = compose_stereo`` is invariant under the rebake, and
back-projecting onto the new Z=0 plane with the rebaked camera returns the same
physical point as back-projecting onto the sheet with the old camera (both asserted
in the unit tests).

``R_corr`` is the minimal, twist-free rotation that carries the new Z axis onto the
sheet's unit normal: its third column IS that normal, ``rz = 0`` exactly. The new
origin sits on the sheet at ``(0, 0, z_offset)``. This places the fitted sheet on the
new Z=0 plane to machine precision (exact reconstruction equivalence), and reproduces
the andre x25 DaVis baked correction (``pinhole_no_self`` vs ``Calibration_pinhole_self``,
world-rot (+1.8816, -0.3618, ~0) mrad / shift (0, 0, -0.1048) mm) to 1.1e-7 rad /
5e-5 mm — the residual being DaVis carrying a ~50 nm in-plane optimizer term the
analytic ideal drops. See ``unit-tests/test_self_cal_frame.py``.
"""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


def plane_to_world_correction(
    z_offset: float, tilt_x: float, tilt_y: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid transform that makes the self-cal sheet the new Z=0 world plane.

    Returns ``(R_corr, t_corr)`` mapping NEW (sheet-aligned) world coords to OLD
    (clicked) world coords: ``X_old = R_corr @ X_new + t_corr``. Feed straight into
    :func:`rebake_pose` for each camera.

    Parameters
    ----------
    z_offset : float
        Sheet Z at world (X, Y) = (0, 0), mm.
    tilt_x, tilt_y : float
        Sheet tilts (rad): ``Z = z_offset + X*tan(tilt_y) + Y*tan(tilt_x)``.
    """
    # Sheet unit normal in the OLD frame, pointing toward +Z. The plane
    # Z - X*tan(ty) - Y*tan(tx) = z_offset has gradient (-tan ty, -tan tx, 1).
    normal = np.array([-math.tan(tilt_y), -math.tan(tilt_x), 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # Minimal (twist-free) rotation taking the new Z axis e3 -> normal; its third
    # column is the normal, so the new Z=0 plane is exactly the sheet. rz = 0.
    e3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis = np.cross(e3, normal)
    sin_a = float(np.linalg.norm(axis))
    if sin_a < 1e-15:  # no tilt -> identity rotation
        R_corr = np.eye(3, dtype=np.float64)
    else:
        angle = math.atan2(sin_a, float(np.dot(e3, normal)))
        R_corr = cv2.Rodrigues((angle / sin_a) * axis)[0]

    t_corr = np.array([0.0, 0.0, float(z_offset)], dtype=np.float64)
    return R_corr, t_corr


def world_to_plane_correction(
    R_corr: np.ndarray, t_corr: np.ndarray
) -> Tuple[float, float, float]:
    """Inverse of :func:`plane_to_world_correction`.

    Recovers ``(z_offset, tilt_x, tilt_y)`` from a NEW->OLD rigid transform. The new
    Z axis in old coords is ``R_corr[:, 2]`` (the sheet normal); rescaling it to a
    unit Z component gives ``(-tan tilt_y, -tan tilt_x, 1)``. Only the Z component of
    ``t_corr`` carries information — the sheet model has no in-plane translation
    degree of freedom — so any in-plane part is ignored.
    """
    R = np.asarray(R_corr, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_corr, dtype=np.float64).reshape(3)
    normal = R[:, 2]
    tilt_y = math.atan2(-normal[0], normal[2])
    tilt_x = math.atan2(-normal[1], normal[2])
    z_offset = float(t[2])
    return z_offset, float(tilt_x), float(tilt_y)


def rebake_pose(
    R: np.ndarray, t: np.ndarray, R_corr: np.ndarray, t_corr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a world-frame redefinition to a world->camera pose.

    ``R' = R @ R_corr``, ``t' = R @ t_corr + t``. Applied to both cameras of a stereo
    record with the same ``(R_corr, t_corr)``, it leaves ``(R_stereo, T_stereo)``
    unchanged while moving the world origin onto the corrected sheet.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    R_corr = np.asarray(R_corr, dtype=np.float64).reshape(3, 3)
    t_corr = np.asarray(t_corr, dtype=np.float64).reshape(3, 1)
    R_new = R @ R_corr
    t_new = R @ t_corr + t
    return R_new, t_new
