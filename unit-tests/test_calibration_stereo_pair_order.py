"""Stereo consumers pair data with models in the RECORD's camera order.

``stereo_model_dir_for_source`` canonicalises the pair, so a pair solved as (2, 1) and
one requested as (1, 2) share one record. ``record.model1`` is ``record.cam1``'s model,
so every consumer must index camera data by the record's order, not the request's.
Before ``record.record_pair`` the self-cal route and CLI loaded camera images in request
order, correlating each camera's images against the other camera's model, and the model
summary labelled each camera's intrinsics with the other camera's number. No error was
raised in either case.
"""

from __future__ import annotations

import types

import pytest
from flask import Flask
from test_calibration_self_cal import _stereo_record

import pivtools_gui.calibration.app.views as V
from pivtools_cli import calibration_cli as CLI
from pivtools_gui.calibration import record as REC
from pivtools_gui.calibration import self_cal as SC

BOARD = "dotboard"


class _FakeConfig:
    """Only what the stereo routes and CLI read: the calibration block + source."""

    camera_count = 2

    def __init__(self, source):
        self._source = source
        self.calibration = {}

    def get_calibration_source(self, idx):
        return self._source


def _save_reversed(source):
    """A pinhole stereo record solved as (2, 1): model1 is Cam2's, model2 Cam1's."""
    record = _stereo_record()
    record.cam1, record.cam2 = 2, 1
    record.model1.rms, record.model2.rms = 0.21, 0.37
    # Fields run_stereo always stores; the model summary reports them, never a stand-in.
    record.board_meta.update(
        stereo_method="stereoCalibrate", n_stereo_views=3, relative_angle_deg=30.0
    )
    REC.save_stereo(record, REC.stereo_model_dir_for_source(source, 2, 1, BOARD))
    return record


class _LoadCaptured(Exception):
    """Raised by the patched image loader once the camera order is captured."""


def _capturing_loader(seen):
    def _load(full_cfg, base_idx, cam1, cam2, n_images, apply_filters):
        seen.append((cam1, cam2))
        raise _LoadCaptured

    return _load


# ---------------------------------------------------------------------------
# record_pair
# ---------------------------------------------------------------------------


def test_record_pair_returns_record_order_for_either_request():
    record = _stereo_record()
    record.cam1, record.cam2 = 2, 1
    assert REC.record_pair(record, [1, 2]) == (2, 1)
    assert REC.record_pair(record, (2, 1)) == (2, 1)


def test_record_pair_mismatch_raises_naming_both():
    record = _stereo_record()
    record.cam1, record.cam2 = 1, 3
    with pytest.raises(ValueError, match=r"\(1, 3\)") as exc:
        REC.record_pair(record, [1, 2])
    assert "[1, 2]" in str(exc.value) or "(1, 2)" in str(exc.value)


# ---------------------------------------------------------------------------
# Self-calibration: images load in the record's order
# ---------------------------------------------------------------------------


class _SyncThread:
    """Run the job body inline so the test sees the loader call deterministically."""

    def __init__(self, target, daemon=None):
        self._target = target

    def start(self):
        self._target()


def test_self_cal_route_loads_images_in_record_order(tmp_path, monkeypatch):
    _save_reversed(tmp_path)
    seen = []
    monkeypatch.setattr(V, "get_config", lambda: _FakeConfig(tmp_path))
    monkeypatch.setattr(V.c2sc, "load_particle_pairs", _capturing_loader(seen))
    monkeypatch.setattr(V, "threading", types.SimpleNamespace(Thread=_SyncThread))
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    resp = app.test_client().post(
        "/calibration/self_cal/run",
        json={"source_path_idx": 0, "board": BOARD, "camera_pair": [1, 2]},
    )
    assert resp.get_json()["success"] is True
    assert seen == [(2, 1)]


def test_self_calibrate_cli_loads_images_in_record_order(tmp_path, monkeypatch):
    _save_reversed(tmp_path)
    seen = []
    monkeypatch.setattr(CLI, "get_config", lambda: _FakeConfig(tmp_path))
    monkeypatch.setattr(SC, "load_particle_pairs", _capturing_loader(seen))
    args = types.SimpleNamespace(
        board=BOARD,
        source=str(tmp_path),
        camera_pair=[1, 2],
        base_path_idx=None,
        n_images=None,
        window_size=None,
        overlap=None,
        no_filters=False,
        no_figures=True,
        model_type=None,
    )
    with pytest.raises(_LoadCaptured):
        CLI.self_calibrate_command(args)
    assert seen == [(2, 1)]


# ---------------------------------------------------------------------------
# GET /calibration/model labels each camera's numbers with that camera
# ---------------------------------------------------------------------------


def test_model_summary_reports_record_camera_order(tmp_path, monkeypatch):
    _save_reversed(tmp_path)
    monkeypatch.setattr(V, "get_config", lambda: _FakeConfig(tmp_path))
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    data = (
        app.test_client()
        .get(
            "/calibration/model",
            query_string={
                "source_path_idx": 0,
                "board": BOARD,
                "stereo": "1",
                "camera_pair": "1,2",
            },
        )
        .get_json()
    )
    assert data["exists"] is True
    # rms_cam1 / intrinsics1 are model1's, which is Cam2's model.
    assert (data["cam1"], data["cam2"]) == (2, 1)
    assert data["rms_cam1"] == pytest.approx(0.21)
    assert data["rms_cam2"] == pytest.approx(0.37)
