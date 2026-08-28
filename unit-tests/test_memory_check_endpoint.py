#!/usr/bin/env python3
"""
test_memory_check_endpoint.py

The worker-memory estimate is a Run PIV warning, not a setup gate.

Background: /backend/validate_files used to run the memory estimate and flip
``valid`` to False when the batch would not fit in one worker. The setup page
then showed "Validation Failed" and told the user to edit batch size in
Performance Settings — a control on the PIV tab, not on setup. Worse, the
setup page re-validates only on path/image keys, so a batch-size change never
cleared the red box. The estimate now lives behind GET /backend/memory_check,
which the Run PIV panel polls whenever batch size or memory limit change.

Usage:
    pytest unit-tests/test_memory_check_endpoint.py -v
"""

import sys
from pathlib import Path

import pytest
import yaml
from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pivtools_gui.app as gui_app  # noqa: E402
from pivtools_core.config import Config  # noqa: E402

# 17.8 MP, matching the case that motivated the change.
_BIG_SHAPE = (3648, 4872)
_SMALL_SHAPE = (64, 64)


def _make_config(tmp_path, batch_size, memory_limit):
    cfg = {
        "paths": {
            "base_paths": [str(tmp_path / "out")],
            "source_paths": [str(tmp_path / "src")],
            "camera_count": 1,
            "camera_numbers": [1],
            "camera_subfolders": [],
        },
        "images": {
            "image_type": "standard",
            "image_format": ["B%05d.tif"],
            "num_images": 100,  # batch_size is capped at per_loop_frame_pairs
            "start_index": 1,
            "pairing_preset": "pre_paired",
            "num_loops": 1,
            "use_camera_subfolders": False,
        },
        "batches": {"size": batch_size},
        "processing": {"dask_memory_limit": memory_limit},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return Config(path=str(config_path))


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The real api_bp route; image_shape is stubbed so no image is read."""

    def _build(batch_size, memory_limit, image_shape):
        config = _make_config(tmp_path, batch_size, memory_limit)
        if image_shape is None:

            def _raise(self):
                raise FileNotFoundError("no image")

            monkeypatch.setattr(Config, "image_shape", property(_raise))
        else:
            monkeypatch.setattr(
                Config, "image_shape", property(lambda self: image_shape)
            )
        monkeypatch.setattr(gui_app, "get_config", lambda: config)
        app = Flask(__name__)
        app.register_blueprint(gui_app.api_bp)
        return app.test_client()

    return _build


def test_warns_when_batch_exceeds_worker_memory(client):
    """50 pairs of 17.8 MP images need ~19.9 GB; 12 GB per worker is short."""
    response = client(50, "12GB", _BIG_SHAPE).get("/backend/memory_check")

    assert response.status_code == 200
    warning = response.get_json()["warning"]
    assert warning is not None
    assert "batch size 50" in warning
    assert "12GB" in warning


def test_silent_when_batch_fits(client):
    response = client(10, "12GB", _SMALL_SHAPE).get("/backend/memory_check")

    assert response.status_code == 200
    assert response.get_json()["warning"] is None


def test_silent_when_no_image_can_be_read(client):
    """No image yet means nothing to estimate — not an error."""
    response = client(50, "12GB", None).get("/backend/memory_check")

    assert response.status_code == 200
    assert response.get_json()["warning"] is None


def test_validate_files_no_longer_carries_the_memory_gate(client):
    """Setup validation reports files only; the memory key is gone."""
    response = client(50, "12GB", _BIG_SHAPE).post(
        "/backend/validate_files", json={}
    )

    assert response.status_code == 200
    assert "memory_warning" not in response.get_json()
