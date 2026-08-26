"""The rigid-rig parameterisation of the joint bundle: ``pose(c, v) = rig_c o board_v``.

What these tests pin, none of which a reprojection-RMS assertion can see:

- the analytic Jacobian of ``_JointBA`` (rig and board-pose columns chain-ruled through
  ``cv2.composeRT``) agrees with finite differences -- a wrong block does not crash, it
  converges slowly to a slightly wrong point, which looks exactly like the defect being fixed;
- the seed factorisation reproduces an exactly rigid pose set and pins the datum board pose to
  the identity;
- a camera chain with NO view common to every camera solves (traverse rigs), and a camera
  linked to nobody raises instead of returning an unobservable pose;
- ``board_release: none`` returns a rigid rig too (the bundle runs in every mode).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pivtools_gui.calibration.detection.base import DetectionResult
from pivtools_gui.calibration.global_grid import resolve_global_grid
from pivtools_gui.calibration.joint import (
    _compose,
    _factorise_rig_seed,
    _JointBA,
    _kabsch,
    run_joint,
)

SPACING = 20.0
COLS, ROWS = 14, 9
IMG_WH = (1280, 1024)
CX, CY = 640.0, 512.0
_FX = {1: 1300.0, 2: 1250.0, 3: 1350.0}
_CAM_RVEC = {
    1: np.array([0.0, 0.0, 0.0]),
    2: np.array([0.0, 0.25, 0.0]),
    3: np.array([0.0, -0.25, 0.0]),
}
_HALF = np.array([(COLS - 1) / 2 * SPACING, (ROWS - 1) / 2 * SPACING, 0.0])


def _K(cam):
    return np.array([[_FX[cam], 0, CX], [0, _FX[cam], CY], [0, 0, 1.0]])


def _rig(cam):
    """Truth world->camera pose of ``cam`` in the board (datum) frame."""
    Rc = cv2.Rodrigues(_CAM_RVEC[cam])[0]
    tc = np.array([0.0, 0.0, 700.0]) - Rc @ _HALF
    return Rc, tc.reshape(3, 1)


def _truth_centres():
    return {c: (-_rig(c)[0].T @ _rig(c)[1]).reshape(3) for c in _CAM_RVEC}


def _board_poses(n_views, seed):
    rng = np.random.default_rng(seed)
    poses = [(np.eye(3), np.zeros((3, 1)))]
    for _ in range(1, n_views):
        rb = cv2.Rodrigues(rng.uniform(-0.30, 0.30, 3) * [1, 1, 0.3])[0]
        poses.append((rb, rng.uniform(-12, 12, 3).reshape(3, 1)))
    return poses


def _dataset(views_of, n_views, seed=0, noise_px=0.0, windows=None):
    """Rigid 3-camera rig; camera ``c`` detects only the views in ``views_of[c]``.

    ``windows`` gives each camera's global-column window (overlapping, so corner ids link the
    cameras). Undetected views are ``success=False`` entries, which ``resolve_global_grid``
    skips, so the grid holds exactly the (camera, view) pairs in ``views_of``.
    """
    windows = windows or {1: range(0, 6), 2: range(4, 10), 3: range(8, 14)}
    poses = _board_poses(n_views, seed)
    noise_rng = np.random.default_rng(seed + 10_000)
    detections = {c: [] for c in _CAM_RVEC}
    for cam in _CAM_RVEC:
        Rc, tc = _rig(cam)
        gidx = np.array(
            [[gx, gy] for gx in windows[cam] for gy in range(ROWS)], dtype=np.int64
        )
        ids = np.array([gy * COLS + gx for gx, gy in gidx], dtype=np.int64)
        pts = np.column_stack([gidx * SPACING, np.zeros(len(gidx))])
        for v in range(n_views):
            if v not in views_of[cam]:
                detections[cam].append(
                    DetectionResult(
                        success=False,
                        board_type="charuco",
                        image_points=np.empty((0, 2)),
                        board_local_points=np.empty((0, 3)),
                    )
                )
                continue
            R, t = _compose((Rc, tc), poses[v])
            px = cv2.projectPoints(pts, cv2.Rodrigues(R)[0], t, _K(cam), None)[0]
            px = px.reshape(-1, 2)
            if noise_px:
                px = px + noise_rng.normal(0.0, noise_px, px.shape)
            detections[cam].append(
                DetectionResult(
                    success=True,
                    board_type="charuco",
                    image_points=px,
                    board_local_points=pts,
                    grid_indices=gidx,
                    point_ids=ids,
                    spacing_mm=SPACING,
                )
            )
    return detections


def _solve(detections, datum_camera=1, datum_view=0, board_release="full3d", **kw):
    grid = resolve_global_grid(detections, spec=None, spacing_mm=SPACING)
    return run_joint(
        detections,
        grid,
        SPACING,
        datum_camera=datum_camera,
        datum_view=datum_view,
        board_release=board_release,
        image_size_by_cam={c: IMG_WH for c in _CAM_RVEC},
        **kw,
    )


def _baseline_errors(res):
    truth = _truth_centres()
    solved = {c: (-m.R.T @ m.t).reshape(3) for c, m in res.models.items()}
    cams = sorted(truth)
    return {
        (a, b): abs(
            np.linalg.norm(solved[a] - solved[b]) - np.linalg.norm(truth[a] - truth[b])
        )
        for i, a in enumerate(cams)
        for b in cams[i + 1 :]
    }


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["full3d", "z_only", "none"])
def test_joint_ba_analytic_jacobian_matches_finite_differences(mode):
    """Every column block -- intrinsics, rig, board pose, free board -- against central FD.

    Small case (2 cameras, 3 views, 6x5 corners) so the dense comparison is cheap. The
    datum view's board pose must have no columns at all.
    """
    rng = np.random.default_rng(7)
    cams, views = [1, 2], [0, 1, 2]
    gx, gy = np.meshgrid(np.arange(6), np.arange(5))
    nominal = np.column_stack([gx.ravel() * SPACING, gy.ravel() * SPACING, np.zeros(30)])
    nominal[:, 2] += rng.normal(0, 0.3, 30)  # a slightly non-flat "released" board
    poses = _board_poses(3, 3)
    rig = {c: _rig(c) for c in cams}
    K_by = {c: _K(c) for c in cams}
    dist_by = {c: np.array([-0.05, 0.01, 0.001, -0.002, 0.0]) for c in cams}
    observations, view_keys = [], []
    for c in cams:
        for v in views:
            R, t = _compose(rig[c], poses[v])
            px = cv2.projectPoints(
                nominal, cv2.Rodrigues(R)[0], t, K_by[c], dist_by[c]
            )[0].reshape(-1, 2)
            px += rng.normal(0, 0.2, px.shape)
            view_keys.append((c, v))
            observations += [(c, v, i, float(px[i, 0]), float(px[i, 1])) for i in range(30)]
    free_rows = [] if mode == "none" else list(range(3, 30))
    ba = _JointBA(nominal, observations, cams, view_keys, free_rows, mode, datum_view=0)
    board_pose = {v: poses[v] for v in views}
    x0 = ba.pack(K_by, dist_by, rig, board_pose, nominal)
    # perturb so no block sits at a symmetric point
    x0 = x0 + rng.normal(0, 1e-3, x0.shape) * np.maximum(1.0, np.abs(x0)) * 1e-2

    assert ba.n_params == len(x0)
    J = ba.jac(x0).toarray()
    assert J.shape == (len(observations) * 2, len(x0))

    def fd(x, h_rel=1e-6):
        out = np.empty_like(J)
        for j in range(len(x)):
            h = h_rel * max(1.0, abs(x[j]))
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            out[:, j] = (ba.residuals(xp) - ba.residuals(xm)) / (2 * h)
        return out

    J_fd = fd(x0)
    scale = np.abs(J_fd).max(axis=0) + 1e-9
    rel = np.abs(J - J_fd).max(axis=0) / scale
    blocks = {
        "intrinsics": slice(0, ba._n_intr),
        "rig": slice(ba._n_intr, ba._n_intr + ba._n_rig),
        "board_pose": slice(ba._n_intr + ba._n_rig, ba._n_intr + ba._n_pose),
        "board_points": slice(ba._n_intr + ba._n_pose, len(x0)),
    }
    for name, sl in blocks.items():
        if sl.start == sl.stop:
            continue
        assert rel[sl].max() < 1e-4, f"{name} block: max rel error {rel[sl].max():.2e}"
    # the datum view's board pose is pinned: n_pose counts only the other views
    assert ba._n_pose == 6 * len(cams) + 6 * (len(views) - 1)


# ---------------------------------------------------------------------------
# Seed factorisation
# ---------------------------------------------------------------------------


def test_factorise_rig_seed_recovers_an_exactly_rigid_pose_set():
    """Given exactly rigid poses, the factorisation returns them and pins the datum board."""
    cams, n_views = [1, 2, 3], 5
    poses = _board_poses(n_views, 11)
    # make the truth board_datum NOT the identity, so the gauge move is exercised
    G = (cv2.Rodrigues(np.array([0.1, -0.2, 0.05]))[0], np.array([[3.0], [-4.0], [7.0]]))
    poses = [_compose(G, p) for p in poses]
    rig = {c: _compose(_rig(c), (G[0].T, -G[0].T @ G[1])) for c in cams}
    pose_by_view = {(c, v): _compose(rig[c], poses[v]) for c in cams for v in range(n_views)}
    # cam3 misses the datum view: the factorisation must still reach it through cam2
    del pose_by_view[(3, 0)]

    rig_out, board_out = _factorise_rig_seed(pose_by_view, cams, 1, 0)
    np.testing.assert_allclose(board_out[0][0], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(board_out[0][1], 0.0, atol=1e-12)
    for (c, v), (R, t) in pose_by_view.items():
        Rr, tr = _compose(rig_out[c], board_out[v])
        np.testing.assert_allclose(Rr, R, atol=1e-9)
        np.testing.assert_allclose(tr, t, atol=1e-7)
    # and the recovered rig is the truth rig expressed in the datum-board frame
    for c in cams:
        Rt, tt = _compose(rig[c], G)
        np.testing.assert_allclose(rig_out[c][0], Rt, atol=1e-9)
        np.testing.assert_allclose(rig_out[c][1], tt, atol=1e-7)


# ---------------------------------------------------------------------------
# run_joint: connectivity, traverse, modes
# ---------------------------------------------------------------------------


def test_joint_traverse_chain_solves_without_a_common_view():
    """No single view is seen by every camera; the datum view is seen by cam1 only.

    This is the traverse-rig case that used to raise "did not observe the datum view".
    """
    views_of = {1: {0, 1, 2, 3}, 2: {2, 3, 4, 5}, 3: {4, 5, 6, 7}}
    res = _solve(_dataset(views_of, 8, seed=5))
    assert (2, 0) not in res.view_poses and (3, 0) not in res.view_poses
    errs = _baseline_errors(res)
    assert max(errs.values()) < 0.1, f"noiseless traverse baselines off by {errs}"


def test_joint_disconnected_camera_raises():
    """cam3 shares no view with cam1 or cam2: its rig pose is unobservable, so it must raise."""
    views_of = {1: {0, 1, 2}, 2: {0, 1, 2}, 3: {3, 4}}
    with pytest.raises(ValueError, match="rigid rig"):
        _solve(_dataset(views_of, 5, seed=6))


def test_joint_output_is_rigid_in_every_board_release_mode():
    """``view_poses`` must factor exactly as ``rig_c o board_v`` for full3d, z_only AND none.

    Under ``none`` the old code never ran a bundle and returned the raw single-view datum
    pose; that path carried the full defect with every test green.
    """
    views_of = {c: set(range(6)) for c in (1, 2, 3)}
    det = _dataset(views_of, 6, seed=2, noise_px=0.3)
    for mode in ("full3d", "z_only", "none"):
        res = _solve(det, board_release=mode)
        rig = {c: (m.R, m.t) for c, m in res.models.items()}
        # board pose of view v from cam1, then every other camera must agree exactly
        for v in range(6):
            R1, t1 = res.view_poses[(1, v)]
            board_v = _compose((rig[1][0].T, -rig[1][0].T @ rig[1][1]), (R1, t1))
            for c in (2, 3):
                Rc, tc = _compose(rig[c], board_v)
                R, t = res.view_poses[(c, v)]
                np.testing.assert_allclose(Rc, R, atol=1e-9, err_msg=f"{mode} cam{c} v{v}")
                np.testing.assert_allclose(tc, t, atol=1e-6, err_msg=f"{mode} cam{c} v{v}")
        # datum view pose IS the rig pose (world = datum board plane)
        for c in (1, 2, 3):
            np.testing.assert_allclose(res.view_poses[(c, 0)][0], rig[c][0], atol=1e-12)
        assert res.info["bundle"]["applied"], f"{mode}: bundle not applied"


def test_joint_datum_choice_is_a_pure_relabel():
    """Baselines must not depend on which view is the datum: it is a gauge label now."""
    views_of = {c: set(range(6)) for c in (1, 2, 3)}
    det = _dataset(views_of, 6, seed=4, noise_px=0.3)
    bl = []
    for dv in (0, 2, 5):
        res = _solve(det, datum_view=dv)
        solved = {c: (-m.R.T @ m.t).reshape(3) for c, m in res.models.items()}
        bl.append(np.linalg.norm(solved[2] - solved[3]))
    # The objective is gauge-invariant, but the bundle stops on its evaluation cap rather than
    # at an exact optimum, and each datum choice seeds from a different root; measured swing
    # is ~0.01 mm on a 345 mm baseline (free per-view poses: 0.64 mm on the real pattern).
    assert max(bl) - min(bl) < 0.05, f"cam2-cam3 baseline swings with the datum: {bl}"


def test_kabsch_helper_still_exported():
    """The charuco tests import ``_kabsch``; keep the seam."""
    A = np.random.default_rng(0).normal(size=(5, 3))
    R, t = _kabsch(A, A)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(t, 0.0, atol=1e-12)
