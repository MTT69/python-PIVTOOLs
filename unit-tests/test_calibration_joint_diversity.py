"""Pose-diversity diagnostic (joint.py): metrics, detectors, storage round-trip, failures.

The headline test drives the production solve on a fronto-parallel synthetic and asserts, in
one place, that the reprojection rms is low AND the pose set is flagged degenerate -- the
exact situation the diagnostic exists for (figures/debug/
synth3cam_fx-error_fronto-vs-tilted-standoff.png: ~1000 % fx error at 0.4 px rms).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pivtools_gui.calibration.global_grid import resolve_global_grid
from pivtools_gui.calibration.joint import (
    PoseDiversity,
    _board_plane_normal,
    _circular_range_deg,
    format_pose_diversity,
    pose_diversity,
    pose_diversity_from_meta,
    pose_diversity_to_meta,
    run_joint,
)
from test_calibration_joint_charuco import _IMG_SIZES, SPACING, _dataset

_BOARD = np.array(
    [[x * 20.0, y * 20.0, 0.0] for x in range(8) for y in range(6)], dtype=np.float64
)
_ROWS = list(range(len(_BOARD)))


def _pose(tilt_deg: float, azimuth_deg: float, standoff: float):
    """World->camera pose: board centred at ``standoff`` on the axis, tilted about an
    in-plane axis so the normal leans towards ``azimuth_deg`` in the image plane."""
    az = np.radians(azimuth_deg)
    axis = np.array([-np.sin(az), np.cos(az), 0.0])  # rotate about this to lean towards az
    R = cv2.Rodrigues(axis * np.radians(tilt_deg))[0]
    centre = _BOARD.mean(axis=0)
    t = np.array([0.0, 0.0, standoff]) - R @ centre
    return R, t


def _rig(specs):
    """``specs[cam] = [(tilt, azimuth, standoff), ...]`` -> (pose_by_view, rows, rms)."""
    poses, rows, rms = {}, {}, {}
    for cam, views in specs.items():
        for v, (tl, az, so) in enumerate(views):
            poses[(cam, v)] = _pose(tl, az, so)
            rows[(cam, v)] = _ROWS
            rms[(cam, v)] = 0.3
    return poses, rows, rms


def test_circular_range_wraps():
    assert _circular_range_deg([10.0, 350.0]) == pytest.approx(20.0)
    assert _circular_range_deg([0.0, 90.0, 180.0, 270.0]) == pytest.approx(270.0)
    assert _circular_range_deg([45.0]) == 0.0
    # tilt AXES: a hinge leaned forward and back is one axis
    assert _circular_range_deg([0.0, 180.0], period=180.0) == 0.0
    assert _circular_range_deg([0.0, 90.0, 200.0], period=180.0) == pytest.approx(90.0)


def test_board_plane_normal_and_planarity():
    n, rms = _board_plane_normal(_BOARD)
    assert abs(n[2]) == pytest.approx(1.0)
    assert rms == 0.0
    bowed = _BOARD.copy()
    bowed[:, 2] = 0.5
    bowed[0, 2] = 1.5  # one point 1 mm off a plane at z=0.5
    _, rms = _board_plane_normal(bowed)
    assert 0.0 < rms < 1.0


def test_metrics_recover_hand_built_tilts():
    specs = {
        1: [(0.0, 0.0, 700.0), (20.0, 0.0, 650.0), (20.0, 90.0, 750.0), (35.0, 200.0, 700.0)]
    }
    pd = pose_diversity(_BOARD, *_rig(specs))
    assert pd.cameras == [1]
    assert pd.n_views[1] == 4 and pd.n_views_tilt_above_floor[1] == 3
    assert pd.tilt_deg_min[1] == pytest.approx(0.0, abs=1e-6)
    assert pd.tilt_deg_max[1] == pytest.approx(35.0, abs=1e-6)
    assert pd.tilt_azimuth_spread_deg[1] == pytest.approx(90.0, abs=1e-6)  # axes 0/90/20
    assert 0.3 < pd.tilt_anisotropy[1] < 1.0
    assert pd.standoff_mm_min[1] == pytest.approx(650.0, abs=2.0)
    assert pd.standoff_range_fraction[1] == pytest.approx(100.0 / 700.0, abs=0.01)
    assert not pd.flag_low_tilt[1]
    assert not pd.flag_single_azimuth[1]
    assert not pd.flag_constant_standoff[1]
    assert pd.degenerate is False
    assert pd.board_planarity_rms_mm == 0.0


def test_fronto_parallel_fires_low_tilt_and_azimuth_is_nan():
    specs = {1: [(0.0, 0.0, 700.0 + 50.0 * i) for i in range(6)]}
    pd = pose_diversity(_BOARD, *_rig(specs))
    assert pd.n_views_tilt_above_floor[1] == 0
    assert np.isnan(pd.tilt_azimuth_spread_deg[1])
    assert np.isnan(pd.tilt_anisotropy[1])
    assert pd.flag_low_tilt[1] and pd.degenerate
    assert not pd.flag_constant_standoff[1]  # it DID move in depth; that is not enough
    text = format_pose_diversity(pd)
    assert "DEGENERATE" in text and "NOT evidence" in text
    assert "azimuth undefined" in text
    assert "GOOD" not in text


def test_single_axis_and_constant_standoff_flag_but_do_not_set_degenerate():
    # leaned forward (0), back (180) and nearly back (185): ONE hinge axis
    specs = {1: [(15.0, 0.0, 700.0), (25.0, 180.0, 700.0), (20.0, 185.0, 700.0)]}
    pd = pose_diversity(_BOARD, *_rig(specs))
    assert pd.flag_single_azimuth[1]
    assert pd.flag_constant_standoff[1]
    assert not pd.flag_low_tilt[1]
    assert pd.degenerate is False
    assert "single tilt axis" in format_pose_diversity(pd)


def test_degenerate_is_rig_level():
    specs = {
        1: [(0.0, 0.0, 700.0), (0.0, 0.0, 750.0)],
        2: [(20.0, 0.0, 700.0), (20.0, 120.0, 700.0), (20.0, 240.0, 760.0)],
    }
    pd = pose_diversity(_BOARD, *_rig(specs))
    assert pd.flag_low_tilt == {1: True, 2: False}
    assert pd.degenerate is True


def test_per_view_rms_outlier_flagged():
    poses, rows, rms = _rig({1: [(20.0, 60.0 * i, 700.0) for i in range(6)]})
    rms[(1, 2)] = 1.0  # 3.3x the 0.3 median
    pd = pose_diversity(_BOARD, poses, rows, rms)
    assert pd.flagged_views == [(1, 2)]
    assert pd.view_rms_median_px[1] == pytest.approx(0.3)
    assert "view 2 (1.00 px)" in format_pose_diversity(pd)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda p, r, m: (p, {k: [] for k in r}, m), "no board rows"),
        (lambda p, r, m: (p, {}, m), "no board rows"),
        (
            lambda p, r, m: ({k: (v[0] * np.nan, v[1]) for k, v in p.items()}, r, m),
            "non-finite",
        ),
        (
            lambda p, r, m: ({k: (v[0], -v[1]) for k, v in p.items()}, r, m),
            "behind the camera",
        ),
    ],
)
def test_malformed_input_raises(mutate, match):
    with pytest.raises(ValueError, match=match):
        pose_diversity(_BOARD, *mutate(*_rig({1: [(10.0, 0.0, 700.0)]})))


def test_collinear_or_tiny_board_raises():
    line = np.array([[i * 10.0, 0.0, 0.0] for i in range(5)])
    with pytest.raises(ValueError, match="collinear"):
        _board_plane_normal(line)
    with pytest.raises(ValueError, match="need >= 3"):
        _board_plane_normal(_BOARD[:2])


def _squeeze(meta):
    """Mimic ``loadmat(squeeze_me=True)``: size-1 arrays become scalars."""
    return {
        k: (v.reshape(()).item() if isinstance(v, np.ndarray) and v.size == 1 else v)
        for k, v in meta.items()
    }


@pytest.mark.parametrize("n_cams", [1, 2])
def test_meta_round_trip_including_single_camera_squeeze(n_cams):
    specs = {
        c: [(0.0 if c == 1 else 20.0, 90.0 * i, 700.0 + 40 * i) for i in range(3)]
        for c in range(1, n_cams + 1)
    }
    poses, rows, rms = _rig(specs)
    rms[(1, 1)] = 2.0
    pd = pose_diversity(_BOARD, poses, rows, rms)
    meta = pose_diversity_to_meta(pd)
    for k, v in meta.items():
        assert not k[0].isdigit(), k
        assert not isinstance(v, str)
        if isinstance(v, np.ndarray):
            assert v.ndim == 2 and v.shape[0] == 1
    back = pose_diversity_from_meta(_squeeze(meta) if n_cams == 1 else meta)
    assert isinstance(back, PoseDiversity)
    assert back.cameras == pd.cameras
    assert back.degenerate == pd.degenerate
    assert back.flag_low_tilt == pd.flag_low_tilt
    assert back.flagged_views == pd.flagged_views == [(1, 1)]
    assert back.view_rms_px == pd.view_rms_px
    for name in ("tilt_deg_median", "standoff_mm_max", "view_rms_median_px"):
        assert getattr(back, name) == pytest.approx(getattr(pd, name))
    for c in pd.cameras:
        a, b = pd.tilt_azimuth_spread_deg[c], back.tilt_azimuth_spread_deg[c]
        assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b)


def test_meta_missing_field_raises_not_defaults():
    pd = pose_diversity(_BOARD, *_rig({1: [(20.0, 0.0, 700.0), (20.0, 90.0, 760.0)]}))
    meta = pose_diversity_to_meta(pd)
    del meta["depth_ratio_max"]
    with pytest.raises(KeyError):
        pose_diversity_from_meta(meta)


def test_fronto_parallel_solve_has_low_rms_and_is_degenerate():
    """The headline: a fit the rms would call healthy, flagged in the same assertion."""
    detections = _dataset(seed=3, fronto=True, noise_px=0.3)
    grid = resolve_global_grid(detections, spec=None, spacing_mm=SPACING)
    res = run_joint(
        detections,
        grid,
        SPACING,
        datum_camera=1,
        datum_view=0,
        board_release="full3d",
        image_size_by_cam=_IMG_SIZES,
    )
    pd = res.pose_diversity
    assert res.rms_px < 0.5 and pd.degenerate is True, (res.rms_px, pd)
    # The solved geometry is self-consistent and wrong (measured 2026-08-26, seed 3: fx
    # 145 % off, standoff 1700 mm for a 700 mm truth, cam2 tilt 25 deg for a 14 deg truth,
    # board bowed to 24 mm planarity rms), so the solved poses are NOT compared with truth.
    # The diagnostic must fire on them as they are.
    assert pd.flag_low_tilt[1] and pd.tilt_deg_median[1] < 5.0
    assert not pd.flag_constant_standoff[1]
    assert pd.board_planarity_rms_mm > 5.0  # a flat board absorbing the missing constraint
    assert set(pd.view_rms_px) == set(res.view_poses)
    text = format_pose_diversity(pd)
    assert "DEGENERATE" in text and "cam1" in text
