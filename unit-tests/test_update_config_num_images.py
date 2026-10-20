"""``POST /backend/update_config`` refuses a null or blank ``images.num_images``.

The GUI once sent ``num_images: null`` when the field was cleared. ``Config.num_images``
then returned ``None`` and ``per_loop_frame_pairs`` raised ``TypeError`` *after* the
file had been saved, so ``GET /backend/config`` 500'd and the page reported
"Cannot connect to backend server". The route must reject it before saving.
"""

import yaml
from flask import Flask

from pivtools_core.config import Config
from pivtools_gui import app as A


def _write_yaml(path, images):
    cfg = {
        "paths": {
            "base_paths": ["/x"],
            "source_paths": ["/y"],
            "camera_numbers": [1],
            "camera_count": 1,
            "camera_subfolders": [],
        },
        "images": images,
        "calibration": {
            "active": "scale_factor",
            "calibration_sources": [],
            "source": "",
            "source_idx": 0,
        },
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f)


def _client(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    _write_yaml(yaml_path, {"num_images": 10, "image_format": ["B%05d.tif"]})
    monkeypatch.setattr(A, "get_config", lambda: Config(path=str(yaml_path)))
    monkeypatch.setattr(A, "reload_config", lambda: None)
    app = Flask(__name__)
    app.register_blueprint(A.api_bp)
    return app.test_client(), yaml_path


def test_null_num_images_is_refused_and_not_saved(tmp_path, monkeypatch):
    client, yaml_path = _client(tmp_path, monkeypatch)
    r = client.post("/backend/update_config", json={"images": {"num_images": None}})
    assert r.status_code == 400
    assert "num_images" in r.get_json()["error"]
    with open(yaml_path) as f:
        assert yaml.safe_load(f)["images"]["num_images"] == 10


def test_blank_and_non_positive_num_images_are_refused(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    for bad in ("", 0, -3, "abc"):
        r = client.post("/backend/update_config", json={"images": {"num_images": bad}})
        assert r.status_code == 400, bad
        assert "num_images" in r.get_json()["error"]


def test_valid_num_images_is_saved(tmp_path, monkeypatch):
    client, yaml_path = _client(tmp_path, monkeypatch)
    r = client.post("/backend/update_config", json={"images": {"num_images": 7}})
    assert r.status_code == 200
    with open(yaml_path) as f:
        assert yaml.safe_load(f)["images"]["num_images"] == 7
