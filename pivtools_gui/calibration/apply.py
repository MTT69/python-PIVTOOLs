"""calibration.apply — apply a fitted model to PIV coordinates + vectors.

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

# Finite-difference step (pixels) for the local pixel->world Jacobian. One pixel is
# small vs a PIV window yet large enough to avoid float noise. The grid points fed in
# are real measurement windows, so the probes stay well inside the FOV in normal use.
_JACOBIAN_STEP_PX = 1.0


def calibrate_coordinates(
    model: CameraModel,
    coords_px: np.ndarray,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    offset_mm=None,
) -> np.ndarray:
    """Image-down pixel coords -> world (X,Y) mm on the sheet plane.

    ``coords_px`` is (...,2); returns (...,2) world mm. NaN where the ray misses.
    ``offset_mm`` is an optional (2,) translation added to every output point — the
    per-camera placement into a shared multi-camera rig frame (``world_offset_mm``).
    It is a pure constant, so it does NOT affect velocities (it cancels in the
    displacement difference); callers computing the offset itself pass None.
    """
    coords_px = np.asarray(coords_px, dtype=np.float64)
    shape = coords_px.shape
    flat = coords_px.reshape(-1, 2)
    world = model.back_project_to_plane(flat, z_world, tilt_x, tilt_y)[:, :2]
    if offset_mm is not None:
        off = np.asarray(offset_mm, dtype=np.float64).reshape(-1)
        if off.size >= 2:
            world = world + off[:2]
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


def local_jacobians(
    model: CameraModel,
    coords_px: np.ndarray,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    h: float = _JACOBIAN_STEP_PX,
) -> np.ndarray:
    """Local 2x2 Jacobian J = d(world_mm)/d(pixel) at each point, (...,2)->(...,2,2).

    Central finite-difference on ``back_project_to_plane`` (``h`` pixels), so it is
    model-agnostic — pinhole / polynomial / scale-factor all expose it. ``J[...,a,b]``
    is ``d world_a / d pixel_b``. This is the linearisation the velocity calibration
    already uses implicitly; here it is made explicit for the stress-tensor transform.

    NaN propagates here only if a probe pixel back-projects to NaN — i.e. a pinhole
    ray that misses the sheet near the FOV edge. The downstream stress is then NaN for
    that window (an honest "off-plane" marker), exactly as the pinhole coordinate path
    already does.
    """
    flat = np.asarray(coords_px, dtype=np.float64).reshape(-1, 2)
    hx = np.array([h, 0.0]); hy = np.array([0.0, h])

    def bp(p):
        return model.back_project_to_plane(p, z_world, tilt_x, tilt_y)[:, :2]

    jx = (bp(flat + hx) - bp(flat - hx)) / (2.0 * h)   # d world / d pixel_x  (N,2)
    jy = (bp(flat + hy) - bp(flat - hy)) / (2.0 * h)   # d world / d pixel_y  (N,2)
    return np.stack([jx, jy], axis=-1)                 # (N,2,2)


def calibrate_stress_tensor(
    model: CameraModel,
    coords_px: np.ndarray,
    UU: np.ndarray,
    VV: np.ndarray,
    UV: np.ndarray,
    dt: float,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate a 2x2 Reynolds-stress field (pixels^2/frame^2 -> m^2/s^2).

    A Reynolds stress is a tensor, so it transforms by the local pixel->world Jacobian
    as ``R_world = J R_px J^T`` (then mm^2->m^2 and per-frame^2->per-s^2):

        R_world = J [[UU, UV],[UV, VV]] J^T / (dt^2 * 1e6)

    ``UU/VV/UV`` are grid-shaped (same as the coordinate grid). Returns the calibrated
    ``(UU, VV, UV)`` in the same shape. With an isotropic ``J = s*I`` this reduces to
    the legacy scalar ``UU*s^2`` / ``VV*s^2`` / ``UV*s^2``; under a mirrored axis the
    off-diagonal carries ``sign(J00*J11)``, so the cross-stress sign is correct with no
    separate negation.
    """
    UU = np.asarray(UU, dtype=np.float64)
    shape = UU.shape
    uu = UU.reshape(-1)
    vv = np.asarray(VV, dtype=np.float64).reshape(-1)
    uv = np.asarray(UV, dtype=np.float64).reshape(-1)
    J = local_jacobians(model, coords_px, z_world, tilt_x, tilt_y)   # (N,2,2)
    n = uu.shape[0]
    R = np.empty((n, 2, 2), dtype=np.float64)
    R[:, 0, 0] = uu
    R[:, 1, 1] = vv
    R[:, 0, 1] = uv
    R[:, 1, 0] = uv
    Rw = J @ R @ np.transpose(J, (0, 2, 1))             # (N,2,2)
    scale = 1.0 / (dt * dt * 1.0e6)
    return (
        (Rw[:, 0, 0] * scale).reshape(shape),
        (Rw[:, 1, 1] * scale).reshape(shape),
        (Rw[:, 0, 1] * scale).reshape(shape),
    )
