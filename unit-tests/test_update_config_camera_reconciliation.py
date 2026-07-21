"""POST /backend/update_config — camera state reconciliation.

Drives the REAL route handler with ``get_config``/``reload_config``
monkeypatched (in the ``pivtools_gui.app`` namespace) to re-read a tmp
config.yaml on every call, so the post-write ``get_config()`` inside the
route sees the persisted disk state exactly as production does.

Pins the camera-subfolder invariant introduced with the stale-``Cam1`` fix:
``paths.camera_subfolders`` is pruned on every persist ([] at count 1,
truncated to count entries otherwise), and the response echoes the
server-reconciled camera state under ``updated.paths`` whenever the request
contained a paths key — the frontend merges that echo into its config
context, keeping it in sync with disk.

Usage:
    pytest unit-tests/test_update_config_camera_reconciliation.py -v
"""

import sys
from pathlib import Path

import pytest
import yaml
from flask import Flask

# Ensure production code is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pivtools_gui.app as A
from pivtools_core.config import Config


def _write_yaml(path: Path, camera_count: int, camera_subfolders: list) -> None:
    """Write a config.yaml with the paths/images keys the route touches."""
    cfg = {
        "paths": {
            "base_paths": ["/tmp/results"],
            "source_paths": ["/tmp/source"],
            "camera_count": camera_count,
            "camera_numbers": list(range(1, camera_count + 1)),
            "camera_subfolders": list(camera_subfolders),
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
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Flask test client for api_bp with a per-call fresh Config on tmp yaml."""
    yaml_path = tmp_path / "config.yaml"
    # Fresh Config per call: the route's post-write get_config() must observe
    # the reconciled state exactly as written to disk.
    monkeypatch.setattr(A, "get_config", lambda: Config(path=str(yaml_path)))
    monkeypatch.setattr(A, "reload_config", lambda: None)
    app = Flask(__name__)
    app.register_blueprint(A.api_bp)
    return app.test_client(), yaml_path


def _disk(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def test_switch_to_one_camera_clears_subfolders(env):
    """The bug: stereo subfolders must not survive a switch to 1 camera."""
    client, yaml_path = env
    _write_yaml(yaml_path, camera_count=2, camera_subfolders=["Cam1", "Cam2"])

    r = client.post("/backend/update_config", json={"paths": {"camera_count": 1}})
    assert r.status_code == 200

    updated_paths = r.get_json()["updated"]["paths"]
    assert updated_paths["camera_count"] == 1
    assert updated_paths["camera_numbers"] == [1]
    assert updated_paths["camera_subfolders"] == []

    disk = _disk(yaml_path)["paths"]
    assert disk["camera_count"] == 1
    assert disk["camera_numbers"] == [1]
    assert disk["camera_subfolders"] == []


def test_count_decrease_truncates_subfolders(env):
    client, yaml_path = env
    _write_yaml(yaml_path, camera_count=3, camera_subfolders=["A", "B", "C"])

    r = client.post("/backend/update_config", json={"paths": {"camera_count": 2}})
    assert r.status_code == 200

    assert r.get_json()["updated"]["paths"]["camera_subfolders"] == ["A", "B"]
    assert _disk(yaml_path)["paths"]["camera_subfolders"] == ["A", "B"]


def test_unrelated_paths_update_keeps_valid_subfolders(env):
    """Explicit subfolders consistent with the count survive other updates."""
    client, yaml_path = env
    _write_yaml(yaml_path, camera_count=2, camera_subfolders=["Left", "Right"])

    r = client.post(
        "/backend/update_config", json={"paths": {"camera_numbers": [1, 2]}}
    )
    assert r.status_code == 200

    assert r.get_json()["updated"]["paths"]["camera_subfolders"] == ["Left", "Right"]
    assert _disk(yaml_path)["paths"]["camera_subfolders"] == ["Left", "Right"]


def test_non_paths_update_still_prunes_stale_state_on_disk(env):
    """The invariant runs on every persist: a stale count-1 yaml is healed
    even by an update that touches no paths keys. No paths echo in that case
    (callers that sent no paths keys don't merge updated.paths)."""
    client, yaml_path = env
    _write_yaml(yaml_path, camera_count=1, camera_subfolders=["Cam1"])

    r = client.post(
        "/backend/update_config", json={"images": {"num_images": 50}}
    )
    assert r.status_code == 200

    assert "paths" not in r.get_json()["updated"]
    assert _disk(yaml_path)["paths"]["camera_subfolders"] == []
