"""Self-cal world-frame rebake adapter (plan item 16).

``plane_to_world_correction`` redefines the world frame so the recovered laser sheet
``Z = z_offset + X*tan(tilt_y) + Y*tan(tilt_x)`` becomes the new Z=0 plane (the DaVis
convention), as a rigid ``(R_corr, t_corr)`` that rebakes into both cameras' extrinsics.

The math is pinned three ways: it round-trips, it places the fitted sheet on the new
Z=0 plane to machine precision (so reconstruction on the rebaked camera equals
reconstruction on the old camera with the tilts applied), and it reproduces the andre
x25 DaVis baked correction extracted from ``pinhole_no_self`` vs
``Calibration_pinhole_self`` (world-rot (+1.8816, -0.3618, ~0) mrad, shift
(0, 0, -0.1048) mm).
"""

import math

import cv2
import numpy as np
import pytest

from pivtools_gui.calibration.camera_model import CameraModel
from pivtools_gui.calibration.self_cal_frame import (
    plane_to_world_correction,
    rebake_pose,
    world_to_plane_correction,
)
from pivtools_gui.calibration.stereo_model import compose_stereo

# (z_offset mm, tilt_x rad, tilt_y rad) — zero, andre-scale, and an exaggerated case.
SHEETS = [
    (0.0, 0.0, 0.0),
    (0.104764, -0.00188162, -0.00036179),   # andre x25 implied OUR params
    (-0.5, 0.012, -0.008),                   # exaggerated tilt
]


def _make_camera(rvec, tvec, f=4000.0, w=2560, h=2160) -> CameraModel:
    """A distortion-free pinhole looking roughly down +Z at the world plane."""
    K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
    R = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    return CameraModel(K=K, dist=np.zeros(5), R=R, t=t, image_size=(w, h))


@pytest.mark.parametrize("z,tx,ty", SHEETS)
def test_round_trip(z, tx, ty):
    R_corr, t_corr = plane_to_world_correction(z, tx, ty)
    z2, tx2, ty2 = world_to_plane_correction(R_corr, t_corr)
    assert (z2, tx2, ty2) == pytest.approx((z, tx, ty), abs=1e-12)


def test_r_corr_is_a_rotation():
    R_corr, _ = plane_to_world_correction(0.3, 0.012, -0.008)
    assert np.allclose(R_corr @ R_corr.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R_corr), 1.0, atol=1e-12)


def test_zero_correction_is_identity():
    R_corr, t_corr = plane_to_world_correction(0.0, 0.0, 0.0)
    assert np.allclose(R_corr, np.eye(3))
    assert np.allclose(t_corr, 0.0)


def test_no_twist_about_z():
    # The correction must not rotate the in-plane axes about Z (DaVis rz ~ 0).
    R_corr, _ = plane_to_world_correction(0.2, 0.012, -0.008)
    rz = cv2.Rodrigues(R_corr)[0].ravel()[2]
    assert abs(rz) < 1e-12


@pytest.mark.parametrize("z,tx,ty", SHEETS)
def test_new_plane_lands_exactly_on_sheet(z, tx, ty):
    R_corr, t_corr = plane_to_world_correction(z, tx, ty)
    xy = np.array([[0, 0], [20, 0], [0, 15], [-12, 8], [25, -18]], dtype=np.float64)
    x_new = np.column_stack([xy, np.zeros(len(xy))])
    x_old = (R_corr @ x_new.T).T + t_corr
    z_sheet = z + x_old[:, 0] * math.tan(ty) + x_old[:, 1] * math.tan(tx)
    assert np.allclose(x_old[:, 2], z_sheet, atol=1e-12)


@pytest.mark.parametrize("z,tx,ty", SHEETS)
def test_rebake_reconstruction_equivalence(z, tx, ty):
    # Back-projecting onto the new Z=0 plane with the rebaked camera must return the
    # same physical point (after mapping new->old) as back-projecting onto the sheet
    # with the old camera. This is the property the whole rebake exists to preserve.
    model = _make_camera(rvec=(0.02, -0.015, 0.01), tvec=(5.0, -3.0, 300.0))
    R_corr, t_corr = plane_to_world_correction(z, tx, ty)
    R_new, t_new = rebake_pose(model.R, model.t, R_corr, t_corr)
    rebaked = CameraModel(K=model.K, dist=model.dist, R=R_new, t=t_new,
                          image_size=model.image_size)

    px = np.array([[800, 700], [1700, 700], [1200, 1500], [600, 1400]],
                  dtype=np.float64)
    world_old = model.back_project_to_plane(px, z, tx, ty)         # on the sheet
    world_new = rebaked.back_project_to_plane(px, 0.0, 0.0, 0.0)   # on new Z=0
    world_new_in_old = (R_corr @ world_new.T).T + t_corr
    assert np.allclose(world_new_in_old, world_old, atol=1e-6)


def test_stereo_pose_invariant_under_rebake():
    m1 = _make_camera(rvec=(0.0, -0.30, 0.0), tvec=(-40.0, 0.0, 320.0))
    m2 = _make_camera(rvec=(0.0, 0.30, 0.0), tvec=(40.0, 0.0, 320.0))
    R_s, T_s = compose_stereo(m1, m2)

    R_corr, t_corr = plane_to_world_correction(0.25, 0.011, -0.006)
    R1, t1 = rebake_pose(m1.R, m1.t, R_corr, t_corr)
    R2, t2 = rebake_pose(m2.R, m2.t, R_corr, t_corr)
    m1r = CameraModel(K=m1.K, dist=m1.dist, R=R1, t=t1, image_size=m1.image_size)
    m2r = CameraModel(K=m2.K, dist=m2.dist, R=R2, t=t2, image_size=m2.image_size)
    R_s2, T_s2 = compose_stereo(m1r, m2r)

    assert np.allclose(R_s, R_s2, atol=1e-12)
    assert np.allclose(T_s, T_s2, atol=1e-9)


def test_numeric_matches_davis_andre():
    # andre x25 OUR-convention sheet params implied by decomposing the DaVis baked
    # correction (R_corr = R_no.T @ R_self, both cameras agree to 1e-16). The inverse
    # (old->new) of our adapter must reproduce DaVis's published correction.
    z, tx, ty = 0.104764, -0.00188162, -0.00036179
    R_corr, t_corr = plane_to_world_correction(z, tx, ty)

    R_old_to_new = R_corr.T
    t_old_to_new = -R_corr.T @ t_corr
    rotvec_mrad = cv2.Rodrigues(R_old_to_new)[0].ravel() * 1e3

    assert rotvec_mrad == pytest.approx([1.8816, -0.3618, 0.0], abs=1e-3)
    assert float(t_old_to_new[2]) == pytest.approx(-0.1048, abs=1e-4)
    # Inverting t_corr=(0,0,z_offset) rotates it into the new frame, so the shift
    # picks up in-plane terms ~ z_offset*tilt ~ 2e-4 mm. DaVis's own extracted shift
    # carries the same ~0.2 um in-plane part; its published (0,0,-0.1048) rounds it
    # away. Negligible vs the 0.1 mm Z shift.
    assert np.allclose(t_old_to_new[:2], 0.0, atol=1e-3)
