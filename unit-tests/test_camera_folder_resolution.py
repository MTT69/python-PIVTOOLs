"""Config.get_camera_folder — per-camera subfolder resolution.

Pins the single-camera rule introduced with the stale-``Cam1`` fix: when
``camera_count == 1`` images always resolve at the source root, and an
explicit ``camera_subfolders`` entry (stale state from a previous
multi-camera session) never overrides that. Multi-camera behaviour is
unchanged: explicit subfolder wins, otherwise ``Cam{n}`` is synthesised.

Usage:
    pytest unit-tests/test_camera_folder_resolution.py -v
"""

import sys
from pathlib import Path

import pytest
import yaml

# Ensure production code is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_core.config import Config


def _make_config(
    tmp_path,
    camera_count: int = 1,
    camera_subfolders: list = (),
    image_type: str = "standard",
    use_camera_subfolders: bool = False,
) -> Config:
    """Write a minimal config.yaml and return a Config bound to it."""
    cfg = {
        "paths": {
            "camera_count": camera_count,
            "camera_numbers": list(range(1, camera_count + 1)),
            "camera_subfolders": list(camera_subfolders),
        },
        "images": {
            "image_type": image_type,
            "use_camera_subfolders": use_camera_subfolders,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return Config(path=str(config_path))


def test_single_camera_ignores_stale_subfolders(tmp_path):
    """The bug: stale ['Cam1', 'Cam2'] from a stereo session must not force
    source/Cam1 once camera_count is 1."""
    config = _make_config(tmp_path, camera_count=1, camera_subfolders=["Cam1", "Cam2"])
    assert config.get_camera_folder(1) == ""


def test_single_camera_clean_resolves_root(tmp_path):
    config = _make_config(tmp_path, camera_count=1)
    assert config.get_camera_folder(1) == ""


def test_multi_camera_synthesizes_cam_n(tmp_path):
    config = _make_config(tmp_path, camera_count=2)
    assert config.get_camera_folder(1) == "Cam1"
    assert config.get_camera_folder(2) == "Cam2"


def test_multi_camera_explicit_subfolders_win(tmp_path):
    config = _make_config(tmp_path, camera_count=2, camera_subfolders=["Left", "Right"])
    assert config.get_camera_folder(1) == "Left"
    assert config.get_camera_folder(2) == "Right"


def test_multi_camera_short_or_empty_entry_falls_back_to_cam_n(tmp_path):
    """A missing or empty entry for a camera synthesises Cam{n} for it."""
    config = _make_config(tmp_path, camera_count=3, camera_subfolders=["Left", ""])
    assert config.get_camera_folder(1) == "Left"
    assert config.get_camera_folder(2) == "Cam2"  # empty entry
    assert config.get_camera_folder(3) == "Cam3"  # beyond list length


@pytest.mark.parametrize("image_type", ["lavision_set", "cine"])
def test_container_formats_never_use_subfolders(tmp_path, image_type):
    config = _make_config(
        tmp_path,
        camera_count=2,
        camera_subfolders=["Cam1", "Cam2"],
        image_type=image_type,
    )
    assert config.get_camera_folder(1) == ""
    assert config.get_camera_folder(2) == ""


def test_im7_without_subfolder_toggle_resolves_root(tmp_path):
    config = _make_config(
        tmp_path,
        camera_count=2,
        camera_subfolders=["Cam1", "Cam2"],
        image_type="lavision_im7",
        use_camera_subfolders=False,
    )
    assert config.get_camera_folder(1) == ""
