"""Image-less scale-factor generate: the GUI route and the CLI command.

Drives the REAL handlers against a tmp workspace with ``get_config`` mocked. The GUI's
"Use calibration images" toggle sends an explicit ``use_image: false``; the route must
then take the image size from the EXISTING saved model (never read an image) and skip
the proof figure. The CLI's ``--image-size W H`` supplies the size directly and must
skip the frame load the same way. Without either flag the image path must still be
the one taken — image-less is opt-in, not a silent rescue on a failed image load.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest
from flask import Flask

import pivtools_cli.calibration_cli as c2
import pivtools_gui.calibration.app.views as V
from pivtools_gui.calibration import record as rec
from pivtools_gui.calibration.pipeline import build_scale_factor_record


class _FakeConfig:
    camera_count = 1

    def __init__(self, base):
        self._base = base
        self.calibration = {}

    def get_calibration_source(self, idx):
        return self._base


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "get_config", lambda: _FakeConfig(tmp_path))
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    return app.test_client(), tmp_path


def _seed_model(source, camera=1, image_size=(1600, 1200)):
    record = build_scale_factor_record(
        camera=camera,
        origin_px=(10.0, 20.0),
        px_per_mm=2.0,
        image_size=image_size,
        dt=1.0,
    )
    model_dir = rec.mono_model_dir_for_source(source, camera, "scale_factor")
    return rec.save_mono(record, model_dir)


def _never_read_image(*args, **kwargs):
    raise AssertionError("image must not be read in image-less generate")


def test_generate_no_image_uses_existing_model_size(env, monkeypatch):
    client, base = env
    _seed_model(base, image_size=(1600, 1200))
    monkeypatch.setattr(V, "read_calibration_image", _never_read_image)

    r = client.post(
        "/calibration/scale_factor/generate",
        json={
            "source_path_idx": 0,
            "camera": 1,
            "use_image": False,
            "px_per_mm": 4.0,
            "origin_px": [30.0, 40.0],
            "dt": 0.5,
        },
    )
    data = r.get_json()
    assert data["success"] is True
    assert data["image_width"] == 1600
    assert data["image_height"] == 1200
    assert data["figures"] == []

    m = client.get(
        "/calibration/model",
        query_string={"board": "scale_factor", "camera": 1, "source_path_idx": 0},
    ).get_json()
    assert m["exists"] is True and m["model_type"] == "scale_factor"
    assert m["px_per_mm"] == pytest.approx(4.0)
    assert list(m["origin_px"]) == pytest.approx([30.0, 40.0])
    assert m["dt"] == pytest.approx(0.5)
    assert m["image_width"] == 1600 and m["image_height"] == 1200


def test_generate_no_image_no_model_errors_clearly(env, monkeypatch):
    client, _ = env
    monkeypatch.setattr(V, "read_calibration_image", _never_read_image)

    r = client.post(
        "/calibration/scale_factor/generate",
        json={
            "source_path_idx": 0,
            "camera": 1,
            "use_image": False,
            "px_per_mm": 4.0,
            "origin_px": [30.0, 40.0],
            "dt": 0.5,
        },
    )
    data = r.get_json()
    assert data["success"] is False
    assert "no existing scale-factor model" in data["error"]


def test_generate_default_still_loads_image(env, monkeypatch):
    """No ``use_image`` key -> the image path runs; the fallback is never silent."""
    client, _ = env
    monkeypatch.setattr(
        V,
        "read_calibration_image",
        lambda *a, **k: np.zeros((300, 400), np.uint8),
    )

    r = client.post(
        "/calibration/scale_factor/generate",
        json={
            "source_path_idx": 0,
            "camera": 1,
            "no_figures": True,
            "px_per_mm": 4.0,
            "origin_px": [30.0, 40.0],
            "dt": 0.5,
        },
    )
    data = r.get_json()
    assert data["success"] is True
    assert data["image_width"] == 400
    assert data["image_height"] == 300


def _cli_args(source, **overrides):
    base = dict(
        source=str(source),
        camera=1,
        image_format=None,
        frame=None,
        px_per_mm=4.0,
        dt=0.5,
        origin=[30.0, 40.0],
        origin_mm=None,
        x_dir="right",
        y_dir="up",
        swap=False,
        no_figures=False,
        image_size=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cli_image_size_skips_frame_load(tmp_path, monkeypatch):
    monkeypatch.setattr(c2, "get_config", lambda: _FakeConfig(tmp_path))
    monkeypatch.setattr(c2, "_load_one", _never_read_image)

    path = c2.scale_factor_command(_cli_args(tmp_path, image_size=[2048, 1080]))

    record = rec.load_mono(path, "scale_factor")
    assert record.camera_model.image_size == (2048, 1080)
    assert 1.0 / record.camera_model.mm_per_pixel == pytest.approx(4.0)
    assert not (path.parent.parent / "figures").exists()


def test_cli_default_still_loads_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(c2, "get_config", lambda: _FakeConfig(tmp_path))
    monkeypatch.setattr(
        c2, "_load_one", lambda *a, **k: np.zeros((300, 400), np.uint8)
    )

    # image_format now has no silent default — the frame-loading path requires
    # it from --image-format or the sidecar.
    path = c2.scale_factor_command(
        _cli_args(tmp_path, no_figures=True, image_format="calib%05d.png")
    )

    record = rec.load_mono(path, "scale_factor")
    assert record.camera_model.image_size == (400, 300)
