"""YAML calibration block = pointer only — save-time strip + generate dt contract.

The sidecar migration leaves exactly four calibration keys in config.yaml
(calibration_sources, source, source_idx, active); everything else lives in the
per-source settings sidecar. These tests pin the two write paths that enforce
that (``Config.save()`` directly, and ``POST /backend/update_config`` which
persists through ``cfg.save()``), the generate-time dt resolution on both the
route and CLI paths (request/--dt > sidecar rig.dt > loud error, never a
default), and the stereo self-cal leg of the apply z/tilt precedence
(``StereoRecord.sc_*`` returns the stored self-cal, 0.0 when absent — the
request-override leg is a plain ``if key in request`` branch in the route/CLI).

Usage:
    pytest unit-tests/test_calibration_pointer_config.py -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import yaml
from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pivtools_core.calibration_settings as cs
import pivtools_gui.app as A
from pivtools_core.config import Config
from pivtools_cli.calibration_cli import _generate_dt_cli
from pivtools_gui.calibration.app.views import _generate_dt
from pivtools_gui.calibration.record import StereoRecord

POINTER_KEYS = {"calibration_sources", "source", "source_idx", "active"}


def _write_yaml(path: Path, calibration: dict) -> None:
    """Minimal config.yaml with the paths/images keys Config needs plus a
    calibration block under test."""
    cfg = {
        "paths": {
            "base_paths": ["/tmp/results"],
            "source_paths": ["/tmp/source"],
            "camera_count": 1,
            "camera_numbers": [1],
            "camera_subfolders": [],
        },
        "images": {
            "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
            "vector_format": ["%05d.mat"],
            "image_type": "standard",
            "num_images": 100,
            "start_index": 1,
            "frame_stride": 0,
            "pair_stride": 1,
            "num_loops": 1,
        },
        "calibration": calibration,
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


_STALE_BLOCK = {
    # pointer keys — must survive
    "calibration_sources": ["/data/calib_a", "/data/calib_b"],
    "source": "",
    "source_idx": 1,
    "active": "dotboard",
    # pre-sidecar keys — must be stripped on save
    "image_format": "calib%05d.tif",
    "image_type": "standard",
    "n_views": 10,
    "dt": 0.004,
    "piv_type": "instantaneous",
    "dotboard": {"dot_spacing_mm": 15.0, "model_type": "pinhole"},
    "global_coordinates": {"enabled": True, "datum_pixel": [1.0, 2.0]},
}


# ---------------------------------------------------------------------------
# Config.save() strips non-pointer keys
# ---------------------------------------------------------------------------


def test_config_save_strips_non_pointer_keys(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    _write_yaml(yaml_path, dict(_STALE_BLOCK))

    Config(path=str(yaml_path)).save()

    with open(yaml_path) as f:
        cal = yaml.safe_load(f)["calibration"]
    assert set(cal) == POINTER_KEYS
    assert cal["calibration_sources"] == ["/data/calib_a", "/data/calib_b"]
    assert cal["source_idx"] == 1
    assert cal["active"] == "dotboard"


def test_config_save_without_calibration_block(tmp_path):
    """No calibration block: save must neither crash nor invent one."""
    yaml_path = tmp_path / "config.yaml"
    _write_yaml(yaml_path, {})
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    del raw["calibration"]
    with open(yaml_path, "w") as f:
        yaml.dump(raw, f)

    Config(path=str(yaml_path)).save()

    with open(yaml_path) as f:
        assert "calibration" not in yaml.safe_load(f)


# ---------------------------------------------------------------------------
# POST /backend/update_config persists through cfg.save() -> same strip
# ---------------------------------------------------------------------------


@pytest.fixture
def route_env(tmp_path, monkeypatch):
    """Flask test client for api_bp with a per-call fresh Config on tmp yaml."""
    yaml_path = tmp_path / "config.yaml"
    monkeypatch.setattr(A, "get_config", lambda: Config(path=str(yaml_path)))
    monkeypatch.setattr(A, "reload_config", lambda: None)
    app = Flask(__name__)
    app.register_blueprint(A.api_bp)
    return app.test_client(), yaml_path


def test_update_config_route_strips_calibration_keys(route_env):
    """A stale full calibration block on disk is reduced to the pointer on the
    next GUI-driven persist, even when the request itself re-sends stale keys."""
    client, yaml_path = route_env
    _write_yaml(yaml_path, dict(_STALE_BLOCK))

    r = client.post(
        "/backend/update_config",
        json={"calibration": {"active": "charuco", "image_format": "x%03d.png"}},
    )
    assert r.status_code == 200

    with open(yaml_path) as f:
        cal = yaml.safe_load(f)["calibration"]
    assert set(cal) == POINTER_KEYS
    assert cal["active"] == "charuco"
    assert cal["calibration_sources"] == ["/data/calib_a", "/data/calib_b"]

    # The response echoes the PERSISTED block, not the request — a stripped key
    # must not be reported back as updated.
    echoed = r.get_json()["updated"]["calibration"]
    assert set(echoed) == POINTER_KEYS
    assert "image_format" not in echoed


# ---------------------------------------------------------------------------
# Generate-time dt: request/--dt > sidecar rig.dt > loud error
# ---------------------------------------------------------------------------


_VALID_IMAGE = {"image_format": "calib%05d.tif", "image_type": "standard"}


def _source_with_dt(tmp_path, dt):
    source = tmp_path / "calib"
    # image block must be valid — load_settings fail-louds on a null image_format
    cs.save_settings(source, {"image": dict(_VALID_IMAGE), "rig": {"dt": dt}})
    return source


def test_generate_dt_request_wins_over_sidecar(tmp_path):
    source = _source_with_dt(tmp_path, 0.25)
    assert _generate_dt({"dt": 0.5}.get, source) == 0.5


def test_generate_dt_falls_to_sidecar(tmp_path):
    source = _source_with_dt(tmp_path, 0.25)
    assert _generate_dt({}.get, source) == 0.25


def test_generate_dt_empty_string_is_unset(tmp_path):
    """The GUI sends '' while the field is being edited — not a value."""
    source = _source_with_dt(tmp_path, 0.25)
    assert _generate_dt({"dt": ""}.get, source) == 0.25


def test_generate_dt_raises_when_nowhere(tmp_path):
    unseeded = tmp_path / "no_sidecar"
    with pytest.raises(ValueError, match="dt is required"):
        _generate_dt({}.get, unseeded)

    seeded_null = tmp_path / "seeded"
    # valid image block, template rig.dt stays null
    cs.save_settings(seeded_null, {"image": dict(_VALID_IMAGE)})
    with pytest.raises(ValueError, match="dt is required"):
        _generate_dt({}.get, seeded_null)


def test_generate_dt_cli_flag_wins_then_sidecar_then_exit(tmp_path):
    source = _source_with_dt(tmp_path, 0.25)
    assert _generate_dt_cli(argparse.Namespace(dt=2.0), source) == 2.0
    assert _generate_dt_cli(argparse.Namespace(dt=None), source) == 0.25
    with pytest.raises(SystemExit, match="dt is required"):
        _generate_dt_cli(argparse.Namespace(dt=None), tmp_path / "no_sidecar")


def test_detect_stereo_reads_geometry_from_sidecar(tmp_path, monkeypatch):
    """Regression: detect-stereo built its board params from the pointer-only YAML
    block instead of the settings sidecar, so sidecar geometry never reached the
    detector. Drive the command up to the detector build and assert the sidecar
    geometry arrived."""
    import pivtools_cli.calibration_cli as cc

    source = tmp_path / "calib"
    cs.save_settings(
        source,
        {
            "image": {**_VALID_IMAGE, "n_views": 3},
            "rig": {"dt": 1.0, "camera_pair": [1, 2]},
            "methods": {
                "charuco": {"squares_h": 12, "squares_v": 9, "square_size": 0.04}
            },
        },
    )

    class _Stop(Exception):
        pass

    captured = {}

    def _capture_build(board, params):
        captured["board"], captured["params"] = board, params
        raise _Stop  # geometry resolved — stop before any image I/O

    monkeypatch.setattr(cc, "_build_detector", _capture_build)
    monkeypatch.setattr(
        cc, "get_config", lambda: argparse.Namespace(calibration={"active": "charuco"})
    )

    args = argparse.Namespace(
        board="charuco",
        source=str(source),
        camera_pair=None,
        image_format=None,
        n_views=None,
        distortion=None,
        world_frame=None,
        dt=None,
    )
    with pytest.raises(_Stop):
        cc.detect_stereo_command(args)

    assert captured["board"] == "charuco"
    assert captured["params"].squares_h == 12
    assert captured["params"].squares_v == 9
    assert captured["params"].square_size_m == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# Stereo apply z/tilt: the self-cal leg (request leg is a request-key branch)
# ---------------------------------------------------------------------------


def test_settings_post_persists_seed_before_partial(tmp_path):
    """Regression: the first POST for a source started from the defaults
    template, discarding the record-recovered seed shown by GET. The route now
    persists the seed as the base, then merges the client partial over it."""
    from flask import Flask

    import pivtools_gui.calibration.app.views as V
    from pivtools_gui.calibration import record as REC
    from pivtools_gui.calibration.pipeline import build_scale_factor_record

    source = tmp_path / "calib"
    source.mkdir()
    record = build_scale_factor_record(
        camera=1, origin_px=(0.0, 0.0), px_per_mm=13.65,
        image_size=(100, 100), dt=0.0057,
    )
    REC.save_mono(record, REC.mono_model_dir_for_source(source, 1, "scale_factor"))

    class _Cfg:
        calibration = {"source_idx": 0}

        def get_calibration_source(self, idx):
            return source

    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    client = app.test_client()
    import unittest.mock as mock

    with mock.patch.object(V, "get_config", lambda: _Cfg()):
        r = client.post(
            "/calibration/settings",
            json={"source_path_idx": 0, "settings": {"rig": {"camera": 2}}},
        )
    assert r.status_code == 200

    stored = cs.load_settings(source)
    assert stored["rig"]["camera"] == 2  # the client partial won its key
    # ...but the record-recovered seed survived underneath it
    assert stored["rig"]["dt"] == pytest.approx(0.0057)
    assert stored["methods"]["scale_factor"]["px_per_mm"] == pytest.approx(13.65)


def test_stereo_record_self_cal_defaults_to_zero():
    # model/R/T payloads are irrelevant to the sc_* properties under test
    rec = StereoRecord(
        cam1=1,
        cam2=2,
        board_type="dotboard",
        model1=None,
        model2=None,
        R_stereo=None,
        T_stereo=None,
    )
    assert rec.sc_z_offset == 0.0
    assert rec.sc_tilt_x == 0.0
    assert rec.sc_tilt_y == 0.0

    rec.self_cal = {"z_offset": 1.5, "tilt_x": 0.01}
    assert rec.sc_z_offset == 1.5
    assert rec.sc_tilt_x == 0.01
    assert rec.sc_tilt_y == 0.0  # absent key still defaults, not KeyError
