"""calibration2.apply — apply a fitted model to PIV coordinates + vectors.

Core math (no implicit Y-flips; all sign carried by the model):

  pixel (image-down) --back_project_to_plane--> world mm on the sheet plane
  velocity = (world(pos+disp) - world(pos)) / 1000 / dt   [mm->m, per-frame->per-s]

The light-sheet plane is ``Z = z_world + X*tan(tilt_y) + Y*tan(tilt_x)``; for a
board placed in the sheet (the datum view) this is Z=0.

File drivers consume the production layout (``coordinates.mat`` 1-based MATLAB +
per-frame ``B*.mat`` with a ``piv_result`` per-pass struct) via ``frames`` to cross
the MATLAB/pixel boundary. The 3C stereo apply lives in ``stereo_model``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .camera_model import CameraModel


def calibrate_coordinates(
    model: CameraModel,
    coords_px: np.ndarray,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
) -> np.ndarray:
    """Image-down pixel coords -> world (X,Y) mm on the sheet plane.

    ``coords_px`` is (...,2); returns (...,2) world mm. NaN where the ray misses.
    """
    coords_px = np.asarray(coords_px, dtype=np.float64)
    shape = coords_px.shape
    flat = coords_px.reshape(-1, 2)
    world = model.back_project_to_plane(flat, z_world, tilt_x, tilt_y)[:, :2]
    return world.reshape(shape)


def calibrate_displacements(
    model: CameraModel,
    coords_px: np.ndarray,
    disp_px: np.ndarray,
    dt: float,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """2C velocity (m/s) from pixel positions + pixel displacements.

    Both ``coords_px`` and ``disp_px`` are (...,2) in image-down pixels.
    Returns (u, v) each shaped like ``coords_px[...,0]``. The sign of v comes
    entirely from the model — there is no manual negation.
    """
    coords_px = np.asarray(coords_px, dtype=np.float64)
    disp_px = np.asarray(disp_px, dtype=np.float64)
    base_shape = coords_px.shape[:-1]
    flat = coords_px.reshape(-1, 2)
    disp = disp_px.reshape(-1, 2)

    w0 = model.back_project_to_plane(flat, z_world, tilt_x, tilt_y)[:, :2]
    w1 = model.back_project_to_plane(flat + disp, z_world, tilt_x, tilt_y)[:, :2]
    delta_mm = w1 - w0
    u = (delta_mm[:, 0] / 1000.0) / dt
    v = (delta_mm[:, 1] / 1000.0) / dt
    return u.reshape(base_shape), v.reshape(base_shape)
