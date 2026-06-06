"""S8 — calibration stepped CLI commands, end-to-end through the headless surface.

Drives the REAL ``detect_stepped_mono_command`` / ``detect_stepped_stereo_command``
and the REAL ``calibrate_stepped_*`` fit. Two seams are faked: ``get_config`` (so the
CLI reads a minimal in-memory config) and the on-disk images (the synthetic renders are
written to ``tmp_path`` as PNGs the command then loads through its normal ``cv2.imread``
path — no image-loading mock, the real file round-trip is exercised).

The synthetic stepped geometry is reused from the calibrator-level tests so the numbers
match. What is asserted:
- the spec-driven mono command writes a reloadable model with sub-pixel RMS (pinhole);
- the polynomial3d mono command fits from a single datum view (1 image) and reloads;
- the stereo command auto-classifies same_side and writes a stereo model;
- a missing stepped spec fails loudly (SystemExit), never guesses fiducials;
- the spec parser is faithful to its JSON.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import pivtools_cli.calibration_cli as cli
from pivtools_gui.calibration.record import load_mono, load_stereo
from pivtools_cli.synthetic_calibration_common import make_camera_matrix

# Mono scene helpers (peak 9x9 z=0, trough 8x8 z=-step, +half-spacing xy interleave).
from test_calibration_stepped_mono import (  # noqa: E402
    PEAK_COLS,
    POSE_RVECS,
    SPACING_MM,
    STEP_MM,
    W,
    H,
    _board_points_mm,
    _level_a_label,
    _render_pose,
)
# Stereo scene helpers (a genuine two-camera same-side rig).
from test_calibration_stepped_stereo import (  # noqa: E402
    RVEC_EXTRA,
    POSE_RVECS as STEREO_RVECS,
    _cam1_pose,
    _cam2_pose,
    _fiducials as _stereo_fiducials,
    _gt_z as _stereo_gt_z,
    _level_a_label as _stereo_level_a_label,
    _render as _stereo_render,
)

IMAGE_FORMAT = "calib%05d.png"


class _FakeConfig:
    """Minimal stand-in for the app config the stepped CLI commands touch."""

    def __init__(self):
        self.calibration = {
            "camera": 1,
            "camera_pair": [1, 2],
            "image_format": IMAGE_FORMAT,
            "start_index": 1,
            "datum_index": 0,
            "distortion_model": "standard",
            "cam_subfolders": {1: "cam1", 2: "cam2"},
            "stepped": {"dot_spacing_mm": SPACING_MM, "step_height_mm": STEP_MM},
        }


def _write_views(images, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, img in enumerate(images, start=1):
        assert cv2.imwrite(str(out_dir / (IMAGE_FORMAT % k)), img)


def _mono_args(source, **overrides):
    base = dict(source=str(source), camera=None, image_format=None, n_views=None,
                distortion=None, model_type=None, stepped_spec=None,
                no_figures=True, force=True)
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(scope="module")
def mono_scene():
    """5 mono poses + the ground-truth fiducials / per-pose level labels."""
    K = make_camera_matrix(W, H)
    images, projs = [], []
    for rv in POSE_RVECS:
        img, proj = _render_pose(rv, K)
        images.append(img)
        projs.append(proj)
    datum = projs[0]
    fiducials = {
        "origin": datum[0].tolist(),
        "x_axis": datum[1].tolist(),
        "y_axis": datum[PEAK_COLS].tolist(),
    }
    peak, trough = _board_points_mm()
    gt_z = np.concatenate([np.zeros(len(peak)), np.full(len(trough), -STEP_MM)])
    from pivtools_gui.calibration.detection.stepped import SteppedDetector, SteppedParams
    det = SteppedDetector(SteppedParams(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM))
    pose_levels = [_level_a_label(det.detect(im), p, gt_z) for im, p in zip(images, projs)]
    return images, fiducials, pose_levels


def test_mono_pinhole_cli_writes_reloadable_model(mono_scene, tmp_path, monkeypatch):
    images, fiducials, pose_levels = mono_scene
    _write_views(images, tmp_path / "cam1")
    monkeypatch.setattr(cli, "get_config", lambda: _FakeConfig())
    cfg = _FakeConfig().calibration
    monkeypatch.setattr(cli, "_cfg2", lambda c: cfg)  # ensure both get_config calls agree

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(
        {"fiducials": fiducials, "clicked_level": "peak", "pose_levels": pose_levels}))
    cfg["cam_subfolders"] = {1: "cam1"}

    path = cli.detect_stepped_mono_command(
        _mono_args(tmp_path, stepped_spec=str(spec), model_type="pinhole"))
    assert path.exists()
    record = load_mono(path.parent)
    assert record.board_type == "stepped"
    assert record.camera_model.rms < 0.5  # sub-pixel on noise-free renders
    assert record.board_meta["n_views"] >= 3


def test_mono_polynomial3d_cli_single_view(mono_scene, tmp_path, monkeypatch):
    images, fiducials, pose_levels = mono_scene
    # Poly3d needs only the datum view; load exactly one image + its label.
    _write_views(images[:1], tmp_path / "cam1")
    monkeypatch.setattr(cli, "get_config", lambda: _FakeConfig())
    cfg = _FakeConfig().calibration
    cfg["cam_subfolders"] = {1: "cam1"}
    monkeypatch.setattr(cli, "_cfg2", lambda c: cfg)

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(
        {"fiducials": fiducials, "clicked_level": "peak", "pose_levels": pose_levels[:1]}))

    path = cli.detect_stepped_mono_command(
        _mono_args(tmp_path, stepped_spec=str(spec), model_type="polynomial3d"))
    assert path.exists()
    record = load_mono(path.parent)
    assert record.camera_model.model_type == "polynomial3d"
    assert record.board_meta["n_views"] == 1
    assert record.camera_model.rms_px < 0.5


def test_mono_cli_requires_spec(mono_scene, tmp_path, monkeypatch):
    images, _, _ = mono_scene
    _write_views(images[:1], tmp_path / "cam1")
    cfg = _FakeConfig().calibration
    cfg["cam_subfolders"] = {1: "cam1"}
    monkeypatch.setattr(cli, "get_config", lambda: _FakeConfig())
    monkeypatch.setattr(cli, "_cfg2", lambda c: cfg)
    with pytest.raises(SystemExit, match="stepped-spec"):
        cli.detect_stepped_mono_command(_mono_args(tmp_path, model_type="pinhole"))


@pytest.fixture(scope="module")
def stereo_scene():
    K = make_camera_matrix(W, H)
    R_extra, _ = cv2.Rodrigues(np.asarray(RVEC_EXTRA, np.float64))
    imgs1, projs1, imgs2, projs2 = [], [], [], []
    for rv in STEREO_RVECS:
        R1, t1, Z, _ = _cam1_pose(rv, K)
        im1, pr1 = _stereo_render(R1, t1, K)
        R2, t2 = _cam2_pose(R1, t1, Z, R_extra)
        im2, pr2 = _stereo_render(R2, t2, K)
        imgs1.append(im1); projs1.append(pr1)
        imgs2.append(im2); projs2.append(pr2)
    gt_z = _stereo_gt_z()
    from pivtools_gui.calibration.detection.stepped import SteppedDetector, SteppedParams
    det = SteppedDetector(SteppedParams(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM))
    poses1 = [_stereo_level_a_label(det.detect(im), p, gt_z) for im, p in zip(imgs1, projs1)]
    poses2 = [_stereo_level_a_label(det.detect(im), p, gt_z) for im, p in zip(imgs2, projs2)]
    spec = {
        "cam1": {"fiducials": _stereo_fiducials(projs1[0]), "clicked_level": "peak",
                 "pose_levels": poses1},
        "cam2": {"fiducials": _stereo_fiducials(projs2[0]), "clicked_level": "peak",
                 "pose_levels": poses2},
        "stereo_config": "auto",
    }
    return imgs1, imgs2, spec


def test_stereo_pinhole_cli_writes_reloadable_model(stereo_scene, tmp_path, monkeypatch):
    imgs1, imgs2, spec = stereo_scene
    _write_views(imgs1, tmp_path / "cam1")
    _write_views(imgs2, tmp_path / "cam2")
    cfg = _FakeConfig().calibration
    monkeypatch.setattr(cli, "get_config", lambda: _FakeConfig())
    monkeypatch.setattr(cli, "_cfg2", lambda c: cfg)

    spec_path = tmp_path / "stereo_spec.json"
    spec_path.write_text(json.dumps(spec))
    args = SimpleNamespace(source=str(tmp_path), camera_pair=None, image_format=None,
                           n_views=None, distortion=None, model_type="pinhole",
                           stepped_spec=str(spec_path), no_figures=True, force=True)
    path = cli.detect_stepped_stereo_command(args)
    assert path.exists()
    record = load_stereo(path.parent)
    assert record.board_type == "stepped"
    assert record.board_meta["stereo_config"] == "same_side"
    assert record.model1.rms < 0.5 and record.model2.rms < 0.5
    assert record.R_stereo is not None


def test_spec_parser_is_faithful():
    d = {"fiducials": {"origin": [1, 2], "x_axis": [3, 4], "y_axis": [5, 6]},
         "clicked_level": "trough", "pose_levels": ["peak", "trough", "peak"]}
    fid, lvl, poses = cli._parse_stepped_cam(d)
    assert fid["origin"] == [1.0, 2.0] and fid["x_axis"] == [3.0, 4.0]
    assert lvl == "trough"
    assert poses == ["peak", "trough", "peak"]
