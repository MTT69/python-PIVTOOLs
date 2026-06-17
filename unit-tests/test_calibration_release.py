"""S1·Phase 2a — object-point release in fit_intrinsics is real, not a no-op.

The historical bug fixed reference-point index N-1 (outside OpenCV's valid [1, N-2]), so
``cv2.calibrateCameraRO`` silently fell back to plain ``calibrateCamera`` and the board was
never released. These tests feed a board with a KNOWN out-of-plane bow but pass NOMINAL FLAT
object points; a working release must (a) move the board off flat, (b) recover the true bow
shape, and (c) predict a held-out view better than a flat board.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pivtools_gui.calibration.camera_model import (
    DistortionModel, _release_gauge_index, fit_intrinsics, fit_pose, reprojection_rms,
)

SPACING = 15.0
COLS, ROWS = 12, 9
W, H = 1280, 1024
K_TRUE = np.array([[1200.0, 0, 640.0], [0, 1200.0, 512.0], [0, 0, 1.0]])
BOW_MM = 1.5  # dome amplitude at board centre


def _boards():
    """Return (nominal_flat (M,3), true_bowed (M,3)) for a centred dot grid."""
    gx, gy = np.meshgrid(np.arange(COLS), np.arange(ROWS))
    x = (gx.ravel() - (COLS - 1) / 2) * SPACING
    y = (gy.ravel() - (ROWS - 1) / 2) * SPACING
    nominal = np.column_stack([x, y, np.zeros_like(x)])
    half_x, half_y = (COLS - 1) / 2 * SPACING, (ROWS - 1) / 2 * SPACING
    z = BOW_MM * (1.0 - 0.5 * ((x / half_x) ** 2 + (y / half_y) ** 2))  # dome, ~0 at corners
    true = np.column_stack([x, y, z])
    return nominal.astype(np.float64), true.astype(np.float64)


def _views(true_board, n=14, seed=0):
    """Project the true board to n views; return list of image points."""
    rng = np.random.default_rng(seed)
    imgs = []
    for v in range(n):
        rvec = np.zeros(3) if v == 0 else rng.uniform(-0.30, 0.30, 3) * [1, 1, 0.25]
        tvec = np.array([rng.uniform(-20, 20), rng.uniform(-20, 20), 600 + rng.uniform(-40, 40)])
        p, _ = cv2.projectPoints(true_board, rvec, tvec, K_TRUE, np.zeros(5))
        imgs.append(p.reshape(-1, 2))
    return imgs


def _detrend(z, xy):
    """Remove the best-fit plane a + b*x + c*y from z (gauge: release fixes tilt/offset freely)."""
    A = np.column_stack([np.ones(len(z)), xy[:, 0], xy[:, 1]])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    return z - A @ coef


def test_release_gauge_index_is_interior():
    nominal, _ = _boards()
    i = _release_gauge_index(nominal)
    assert 1 <= i <= len(nominal) - 2


def test_release_is_not_a_noop_and_recovers_bow():
    nominal, true = _boards()
    imgs = _views(true)
    objs = [nominal] * len(imgs)
    K, dist, rvecs, tvecs, rms, pv, released = fit_intrinsics(
        objs, imgs, (W, H), DistortionModel.STANDARD, use_release_object=True)

    assert released is not None and released.shape == nominal.shape
    # (a) not a no-op: the board moved off flat by an amount comparable to the true bow.
    assert np.max(np.abs(released[:, 2] - nominal[:, 2])) > 0.3 * BOW_MM
    # (b) recovers the bow shape: detrended released Z correlates with detrended true Z.
    zt = _detrend(true[:, 2], true[:, :2])
    zr = _detrend(released[:, 2], released[:, :2])
    corr = float(np.corrcoef(zt, zr)[0, 1])
    assert corr > 0.9, f"released bow corr {corr:.3f}"
    # in-plane stays essentially nominal (no spurious lateral release with zero distortion)
    assert np.max(np.abs(released[:, :2] - nominal[:, :2])) < 0.3
    assert rms < 0.5


def test_release_beats_flat_on_held_out_view():
    nominal, true = _boards()
    imgs = _views(true, n=14, seed=1)
    held = len(imgs) - 1
    train_imgs = imgs[:held]

    flat = fit_intrinsics([nominal] * len(train_imgs), train_imgs, (W, H),
                          DistortionModel.STANDARD, use_release_object=False)
    K_f, d_f = flat[0], flat[1]
    K_r, d_r, _rv, _tv, _rms, _pv, released = fit_intrinsics(
        [nominal] * len(train_imgs), train_imgs, (W, H),
        DistortionModel.STANDARD, use_release_object=True)

    img = imgs[held]
    R_f, t_f = fit_pose(nominal, img, K_f, d_f, planar=True)
    R_r, t_r = fit_pose(released, img, K_r, d_r, planar=False)
    rms_f = reprojection_rms(nominal, img, K_f, d_f, cv2.Rodrigues(R_f)[0], t_f)
    rms_r = reprojection_rms(released, img, K_r, d_r, cv2.Rodrigues(R_r)[0], t_r)
    assert rms_r < rms_f, f"released held-out {rms_r:.4f} not better than flat {rms_f:.4f}"
