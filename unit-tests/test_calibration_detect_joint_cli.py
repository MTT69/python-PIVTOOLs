"""S1·Phase 4 — the ``detect-joint`` CLI command + the global_grid config reader.

Exercised: ``_global_grid_spec_from_cfg`` (the headless reader that turns the
``global_grid`` block into a ``GlobalGridSpec``) and ``detect_joint_command``'s
image-free guard rails (unknown board, missing cameras, model-fidelity checks).
The image-driven end-to-end solves were deleted 2026-07-29 — they depended on a
``stereo_charuco`` synthetic set that was never generated (see git history to
recover them if it ever is); the dotboard click path is covered at the resolver
level in ``test_calibration_global_grid``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pivtools_cli.calibration_cli as cli
from pivtools_core import calibration_settings as cs

_SYN = Path(__file__).parent / "synthetic_calibration"
_STEREO_CHARUCO = _SYN / "stereo_charuco"


# ---------------------------------------------------------------------------
# _global_grid_spec_from_cfg — config block -> GlobalGridSpec
# ---------------------------------------------------------------------------


def test_spec_from_cfg_parses_datum_and_cross_camera_anchor():
    gg = {
        "datum_camera": 1,
        "datum_view": 0,
        "datum_clicks": {
            "origin": [10.0, 20.0],
            "x_axis": [110.0, 20.0],
            "y_axis": [10.0, 120.0],
            "origin_mm": [5.0, -3.0],
        },
        "anchors": [
            {
                "camera": 2,
                "view": 0,
                "correspondences": [
                    {
                        "pixel": [50.0, 60.0],
                        "same_as": [1, 0],
                        "ref_pixel": [200.0, 210.0],
                    },
                    {
                        "pixel": [80.0, 60.0],
                        "same_as": [1, 0],
                        "ref_pixel": [240.0, 210.0],
                    },
                ],
            },
        ],
    }
    spec = cli._global_grid_spec_from_cfg(gg)
    assert spec.datum_camera == 1 and spec.datum_view == 0
    np.testing.assert_allclose(spec.datum_clicks["origin"], [10.0, 20.0])
    np.testing.assert_allclose(spec.datum_clicks["origin_mm"], [5.0, -3.0])
    assert len(spec.anchors) == 1
    a = spec.anchors[0]
    assert (a.camera, a.view) == (2, 0)
    assert len(a.correspondences) == 2
    # same_as must be coerced from a YAML list to a (camera, view) tuple.
    assert a.correspondences[0].same_as == (1, 0)
    assert isinstance(a.correspondences[0].same_as, tuple)
    np.testing.assert_allclose(a.correspondences[0].ref_pixel, [200.0, 210.0])


def test_spec_from_cfg_handles_origin_literal_and_no_ref_pixel():
    gg = {
        "datum_camera": 1,
        "datum_view": 0,
        "datum_clicks": {
            "origin": [0.0, 0.0],
            "x_axis": [1.0, 0.0],
            "y_axis": [0.0, 1.0],
        },
        "anchors": [
            {
                "camera": 1,
                "view": 1,
                "correspondences": [
                    {"pixel": [5.0, 6.0], "same_as": "origin"},
                ],
            },
        ],
    }
    spec = cli._global_grid_spec_from_cfg(gg)
    # origin_mm defaults to [0, 0] when omitted.
    np.testing.assert_allclose(spec.datum_clicks["origin_mm"], [0.0, 0.0])
    corr = spec.anchors[0].correspondences[0]
    assert corr.same_as == "origin"
    assert corr.ref_pixel is None


def test_spec_from_cfg_raises_on_missing_datum_click():
    gg = {
        "datum_camera": 1,
        "datum_view": 0,
        "datum_clicks": {"origin": None, "x_axis": [1.0, 0.0], "y_axis": [0.0, 1.0]},
    }
    with pytest.raises(SystemExit, match="datum_clicks.origin"):
        cli._global_grid_spec_from_cfg(gg)


# ---------------------------------------------------------------------------
# detect_joint_command — end-to-end on the two-camera ChArUco set
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal config double for detect-joint: the pointer block + rig camera count.

    Image sourcing + board geometry now live in the SOURCE's settings sidecar
    (written by ``_write_settings``); the clicked-coords block lives in the
    sidecar ``inputs.mat``. ChArUco needs no clicks (corner ids give the grid)
    and its cameras come from ``--cameras`` here.
    """

    camera_count = 2

    def __init__(self):
        self.calibration = {"active": "charuco"}


def _write_settings(src, fit=None):
    """The settings sidecar the command reads (replaces the old config block)."""
    cs.save_settings(
        src,
        {
            "image": {
                "image_format": "calib%05d.png",
                "image_type": "standard",
                "start_index": 1,
                "n_views": 10,
                "use_camera_subfolders": True,
                "camera_subfolders": ["cam1", "cam2"],
            },
            "rig": {"dt": 1.0},
            "fit": fit or {},
            "methods": {
                "charuco": {
                    "squares_h": 10,
                    "squares_v": 7,
                    "square_size": 0.030,
                    "marker_ratio": 0.5,
                    "aruco_dict": "DICT_4X4_1000",
                    "min_corners": 6,
                }
            },
        },
    )


def _args(source, **overrides):
    base = dict(
        source=str(source),
        board=None,
        cameras=None,
        image_format=None,
        n_views=None,
        model_type=None,
        distortion=None,
        board_release=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _install(monkeypatch, cfg):
    monkeypatch.setattr(cli, "get_config", lambda: cfg)
    monkeypatch.setattr(cli, "_cfg2", lambda c: c.calibration)


# The e2e solves (charuco end-to-end, polynomial per-camera, cameras override,
# drop-failed-view, blank-camera-fatal) were deleted 2026-07-29: they skipif'd on
# a ``stereo_charuco`` synthetic set that was never generated on any machine, so
# they had never run anywhere. If that fixture set is ever rendered, recover them
# from git history — the guard tests below run without images and stay.


def test_detect_joint_rejects_unknown_board(monkeypatch):
    cfg = _FakeConfig()
    _install(monkeypatch, cfg)
    with pytest.raises(SystemExit, match="board must be dotboard|charuco"):
        cli.detect_joint_command(_args(_STEREO_CHARUCO, board="stepped", cameras="1,2"))


def test_detect_joint_requires_cameras(monkeypatch):
    """No sidecar coords and no ``--cameras`` -> the rig is undefined and the solve refuses."""
    cfg = _FakeConfig()
    _install(monkeypatch, cfg)
    with pytest.raises(SystemExit, match="set the rig cameras"):
        cli.detect_joint_command(_args(_STEREO_CHARUCO, board="charuco"))


# The model-fidelity guards (these fire before any image is loaded, so no source is needed).


def test_detect_joint_rejects_non_standard_distortion(tmp_path, monkeypatch):
    cfg = _FakeConfig()
    _install(monkeypatch, cfg)
    _write_settings(tmp_path, fit={"distortion_model": "rational"})
    with pytest.raises(SystemExit, match="DaVis pinhole only"):
        cli.detect_joint_command(_args(tmp_path, board="charuco", cameras="1,2"))


def test_detect_joint_requires_fixed_aspect(tmp_path, monkeypatch):
    cfg = _FakeConfig()
    _install(monkeypatch, cfg)
    _write_settings(tmp_path, fit={"fix_aspect_ratio": False})
    with pytest.raises(SystemExit, match="fix_aspect_ratio"):
        cli.detect_joint_command(_args(tmp_path, board="charuco", cameras="1,2"))


def test_detect_joint_rejects_bad_board_release(tmp_path, monkeypatch):
    cfg = _FakeConfig()
    _install(monkeypatch, cfg)
    with pytest.raises(SystemExit, match="board_release must be"):
        cli.detect_joint_command(
            _args(tmp_path, board="charuco", cameras="1,2", board_release="bogus")
        )


