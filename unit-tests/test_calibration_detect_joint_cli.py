"""S1·Phase 4 — the ``detect-joint`` CLI command + the global_grid config reader.

Two layers are exercised:

- ``_global_grid_spec_from_cfg`` — the headless reader that turns the
  ``calibration.global_grid`` config block into a ``GlobalGridSpec`` (same_as list->tuple,
  the ``origin`` literal, ``ref_pixel`` None, and a loud failure on a missing datum click).
- ``detect_joint_command`` end-to-end on the pre-rendered two-camera ChArUco set: real
  image load -> real detect -> real ``resolve_global_grid`` (corner-id path, no clicks) ->
  real ``run_joint`` -> a unified ``JointRecord`` written and reloadable. ChArUco is used for
  the end-to-end because it needs no datum/overlap clicks; the dotboard click path is covered
  at the resolver level in ``test_calibration_global_grid``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import pivtools_cli.calibration_cli as cli
from pivtools_gui.calibration.camera_model import PolynomialModel
from pivtools_gui.calibration.record import load_joint, load_mono

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
    """Minimal config double for detect-joint: a calibration block + the resolvers it calls.

    The clicked-coords block (datum + anchors + cameras) is NOT config anymore — it lives in the
    sidecar ``inputs.mat``. ChArUco needs no clicks (corner ids give the grid) and its cameras come
    from ``--cameras`` here, so these tests need no sidecar; the dotboard click path is covered at
    the resolver level in ``test_calibration_global_grid``.
    """

    def __init__(self):
        self.calibration = {
            "active": "charuco",
            "image_format": "calib%05d.png",
            "start_index": 1,
            "n_views": 10,
            "distortion_model": "standard",
            "fix_aspect_ratio": True,
            "use_camera_subfolders": True,
            "camera_subfolders": ["cam1", "cam2"],
            "charuco": {
                "squares_h": 10,
                "squares_v": 7,
                "square_size": 0.030,
                "marker_ratio": 0.5,
                "aruco_dict": "DICT_4X4_1000",
                "min_corners": 6,
            },
        }

    def get_calibration_camera_folder(self, camera_num: int) -> str:
        c = self.calibration
        if not c.get("use_camera_subfolders", False):
            return ""
        subs = c.get("camera_subfolders", [])
        idx = camera_num - 1
        return subs[idx] if 0 <= idx < len(subs) and subs[idx] else ""


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


def _tmp_source(tmp_path):
    """A writable source whose cam1/cam2 symlink the read-only checked-in ChArUco fixture.

    detect-joint writes its record under ``<source>/calibration``; pointing source at a tmp dir
    keeps the solve's output out of the checked-in fixture tree.
    """
    src = tmp_path / "src"
    src.mkdir()
    for cam in ("cam1", "cam2"):
        (src / cam).symlink_to(_STEREO_CHARUCO / cam)
    return src


@pytest.mark.skipif(
    not _STEREO_CHARUCO.is_dir(), reason="synthetic stereo_charuco set absent"
)
def test_detect_joint_charuco_end_to_end(tmp_path, monkeypatch):
    cfg = _FakeConfig()
    _install(monkeypatch, cfg)

    path = cli.detect_joint_command(
        _args(_tmp_source(tmp_path), cameras="1,2", board_release="full3d")
    )
    rec = load_joint(Path(path))

    assert rec.cameras == [1, 2]
    assert rec.board_type == "charuco"
    assert rec.board_release == "full3d"
    # One shared board: every camera agrees on it by construction.
    assert rec.board_meta["cross_camera_board_agreement_mm"] == pytest.approx(
        0.0, abs=1e-9
    )
    # Synthetic data -> the joint solve should reproject tightly for both cameras.
    assert np.isfinite(rec.rms_px) and rec.rms_px < 2.0
    for c in (1, 2):
        assert c in rec.models
        assert rec.per_camera_rms[c] < 2.0
    # ChArUco spacing is the square size in mm (30 mm), not the raw 0.03 m.
    assert rec.spacing_mm == pytest.approx(30.0)
    # A real shared board with many dots.
    assert len(rec.board) > 40


@pytest.mark.skipif(
    not _STEREO_CHARUCO.is_dir(), reason="synthetic stereo_charuco set absent"
)
def test_detect_joint_polynomial_writes_per_camera_records(tmp_path, monkeypatch):
    """model_type=polynomial fits a per-camera single-plane map in the shared global frame."""
    cfg = _FakeConfig()
    _install(monkeypatch, cfg)

    src = _tmp_source(tmp_path)
    paths = cli.detect_joint_command(_args(src, model_type="polynomial", cameras="1,2"))
    assert len(paths) == 2
    for cam, p in zip((1, 2), paths):
        mono = load_mono(Path(p))
        assert isinstance(mono.camera_model, PolynomialModel)
        assert mono.camera == cam
        assert mono.world_frame.mode == "global_grid"
        # Synthetic data -> the planar polynomial should fit to well under a mm.
        assert mono.camera_model.rms_x_mm < 1.0 and mono.camera_model.rms_y_mm < 1.0


@pytest.mark.skipif(
    not _STEREO_CHARUCO.is_dir(), reason="synthetic stereo_charuco set absent"
)
def test_detect_joint_cameras_override_from_cli(tmp_path, monkeypatch):
    """``--cameras`` selects the rig; a single camera still solves (degenerate rig)."""
    cfg = _FakeConfig()
    _install(monkeypatch, cfg)
    path = cli.detect_joint_command(
        _args(_tmp_source(tmp_path), cameras="1", board_release="none")
    )
    rec = load_joint(Path(path))
    assert rec.cameras == [1]


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
    cfg.calibration["distortion_model"] = "rational"
    _install(monkeypatch, cfg)
    with pytest.raises(SystemExit, match="DaVis pinhole only"):
        cli.detect_joint_command(_args(tmp_path, board="charuco", cameras="1,2"))


def test_detect_joint_requires_fixed_aspect(tmp_path, monkeypatch):
    cfg = _FakeConfig()
    cfg.calibration["fix_aspect_ratio"] = False
    _install(monkeypatch, cfg)
    with pytest.raises(SystemExit, match="fix_aspect_ratio"):
        cli.detect_joint_command(_args(tmp_path, board="charuco", cameras="1,2"))


def test_detect_joint_rejects_bad_board_release(tmp_path, monkeypatch):
    cfg = _FakeConfig()
    _install(monkeypatch, cfg)
    with pytest.raises(SystemExit, match="board_release must be"):
        cli.detect_joint_command(
            _args(tmp_path, board="charuco", cameras="1,2", board_release="bogus")
        )


@pytest.mark.skipif(
    not _STEREO_CHARUCO.is_dir(), reason="synthetic stereo_charuco set absent"
)
def test_detect_joint_drops_failed_view(tmp_path, monkeypatch):
    """One bad (non-datum) frame is dropped, not fatal — the rest of the rig still calibrates.

    The user's requirement: a single bad image must not throw off the whole solve.
    """
    src = tmp_path / "src"
    (src / "cam1").mkdir(parents=True)
    (src / "cam2").mkdir(parents=True)
    for k in range(1, 11):
        name = "calib%05d.png" % k
        (src / "cam1" / name).symlink_to(_STEREO_CHARUCO / "cam1" / name)
        if k == 5:
            # a blank, non-datum frame the ChArUco detector cannot resolve -> a failed view
            assert cv2.imwrite(str(src / "cam2" / name), np.zeros((600, 800), np.uint8))
        else:
            (src / "cam2" / name).symlink_to(_STEREO_CHARUCO / "cam2" / name)
    cfg = _FakeConfig()
    _install(monkeypatch, cfg)

    # does NOT raise — the bad frame is dropped
    path = cli.detect_joint_command(_args(src, cameras="1,2", board_release="none"))
    rec = load_joint(Path(path))
    assert rec.cameras == [1, 2]
    assert np.isfinite(rec.rms_px)


@pytest.mark.skipif(
    not _STEREO_CHARUCO.is_dir(), reason="synthetic stereo_charuco set absent"
)
def test_detect_joint_blank_camera_fails_loudly(tmp_path, monkeypatch):
    """A camera that detects NOTHING in any image is fatal (almost always a wrong path/format)."""
    src = tmp_path / "src"
    (src / "cam1").mkdir(parents=True)
    (src / "cam2").mkdir(parents=True)
    for k in range(1, 11):
        name = "calib%05d.png" % k
        (src / "cam1" / name).symlink_to(_STEREO_CHARUCO / "cam1" / name)
        assert cv2.imwrite(str(src / "cam2" / name), np.zeros((600, 800), np.uint8))
    cfg = _FakeConfig()
    _install(monkeypatch, cfg)
    with pytest.raises(SystemExit, match="detected no calibration target"):
        cli.detect_joint_command(_args(src, cameras="1,2", board_release="none"))
