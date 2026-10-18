"""Calibration camera-subfolder resolution and the settings-sidecar prune.

Mirrors the PIV rule in ``Config.get_camera_folder`` (see
``test_camera_folder_resolution.py``): a single-camera setup always resolves calibration
images at the source root, and a stored ``camera_subfolders`` entry (stale state from a
previous multi-camera session) never overrides that. Multi-camera behaviour is unchanged:
an explicit entry wins, else ``Cam{N}``, and the toggle off keeps the flat layout.

The ``/calibration/settings`` POST keeps the sidecar consistent with the configured camera
count, as ``/update_config`` does for config.yaml
(``test_update_config_camera_reconciliation.py``).

Usage:
    pytest unit-tests/test_calibration_camera_folder.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from flask import Flask

import pivtools_gui.calibration.app.views as V
from pivtools_core import calibration_settings as cs
from pivtools_core.image_handling.path_utils import calibration_camera_folder


def _image(use_subfolders: bool, subfolders: list) -> dict:
    return {"use_camera_subfolders": use_subfolders, "camera_subfolders": subfolders}


# ---------------------------------------------------------------------------
# calibration_camera_folder
# ---------------------------------------------------------------------------


def test_single_camera_ignores_stale_subfolder():
    """Count 1 + stale ["Cam1"] + toggle on -> source root."""
    assert calibration_camera_folder(_image(True, ["Cam1"]), "standard", 1, 1) == ""


def test_single_camera_im7_ignores_stale_subfolder():
    assert calibration_camera_folder(_image(True, ["Cam1"]), "lavision_im7", 1, 1) == ""


def test_multi_camera_explicit_entries_win():
    image = _image(True, ["left", "right"])
    assert calibration_camera_folder(image, "standard", 1, 2) == "left"
    assert calibration_camera_folder(image, "standard", 2, 2) == "right"


def test_multi_camera_without_entries_uses_cam_n():
    image = _image(True, [])
    assert calibration_camera_folder(image, "standard", 1, 2) == "Cam1"
    assert calibration_camera_folder(image, "standard", 2, 2) == "Cam2"


def test_multi_camera_toggle_off_is_flat_layout():
    """The flat multi-camera layout (all cameras in the source root) stays allowed."""
    image = _image(False, ["left", "right"])
    assert calibration_camera_folder(image, "standard", 1, 2) == ""
    assert calibration_camera_folder(image, "standard", 2, 2) == ""


@pytest.mark.parametrize("image_type", ["lavision_set", "cine"])
@pytest.mark.parametrize("num_cameras", [1, 2])
def test_containers_never_use_subfolders(image_type, num_cameras):
    image = _image(True, ["left", "right"])
    assert calibration_camera_folder(image, image_type, 1, num_cameras) == ""


# ---------------------------------------------------------------------------
# /calibration/settings POST prunes camera_subfolders to the camera count
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, base: Path, camera_count: int) -> None:
        self._base = base
        self.camera_count = camera_count
        self.calibration = {"active": "dotboard"}

    def get_calibration_source(self, idx):
        return self._base


def _client(monkeypatch, tmp_path, camera_count: int):
    monkeypatch.setattr(V, "get_config", lambda: _FakeConfig(tmp_path, camera_count))
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    return app.test_client()


def _post(client, image: dict) -> dict:
    response = client.post(
        "/calibration/settings",
        json={"source_path_idx": 0, "settings": {"image": image}},
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _disk_subfolders(source: Path) -> list:
    """The stored list, read raw (``load_settings`` also validates unrelated fields)."""
    stored = yaml.safe_load(cs.settings_path(source).read_text(encoding="utf-8"))
    return stored["image"]["camera_subfolders"]


def test_settings_post_at_one_camera_prunes_and_echoes(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path, camera_count=1)
    data = _post(client, {"use_camera_subfolders": True, "camera_subfolders": ["Cam1"]})
    assert data["settings"]["image"]["camera_subfolders"] == []
    assert _disk_subfolders(tmp_path) == []


def test_settings_post_prunes_stale_entries_already_on_disk(monkeypatch, tmp_path):
    """A partial that never mentions subfolders still reconciles the stored list (the
    camera count may have dropped since it was written)."""
    cs.save_settings(tmp_path, {"image": {"camera_subfolders": ["Cam1", "Cam2"]}})
    client = _client(monkeypatch, tmp_path, camera_count=1)
    data = _post(client, {"zero_based_indexing": False})
    assert data["settings"]["image"]["camera_subfolders"] == []
    assert _disk_subfolders(tmp_path) == []


def test_settings_post_truncates_to_camera_count(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path, camera_count=2)
    data = _post(client, {"camera_subfolders": ["left", "right", "spare"]})
    assert data["settings"]["image"]["camera_subfolders"] == ["left", "right"]
    assert _disk_subfolders(tmp_path) == ["left", "right"]


def test_settings_post_keeps_valid_entries(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path, camera_count=2)
    data = _post(client, {"camera_subfolders": ["left", "right"]})
    assert data["settings"]["image"]["camera_subfolders"] == ["left", "right"]
