"""Calibration P4 brief 1c: paths, counts and forward-only config strictness.

Covers, in order:

1. The camera-subfolder rule is the PIV rule on both sides (user decision
   2026-10-19): containers never use a subfolder, IM7 is gated by
   ``use_camera_subfolders``, standard formats always resolve ``Cam{N}`` (or the
   explicit entry) at camera_count > 1. ``Config.get_camera_folder`` and
   ``calibration_camera_folder`` share one function.
2. AppleDouble / dot-files never count as calibration views.
3. A container (.set / .cine) holding fewer entries than the caller expects fails
   validation with "found N entries, expected M".
4. The CINE extension-mismatch text survives the PIV ``validate_files`` route.
5. ``zero_based_indexing`` on a container calibration source is refused, not
   silently ignored.
6. A null ``calibration:`` block reads as empty, so the designed
   "calibration.active is required" message fires.
10. The live pairing-key migration is gone: a config still carrying
    ``time_resolved`` / ``pairing_mode`` fails naming the keys to rewrite.
11. ``images.num_images: null`` raises instead of flowing on as None.

Usage:
    pytest unit-tests/test_calibration_paths_strict.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from flask import Flask

from pivtools_core.calibration_settings import save_settings
from pivtools_core.config import Config
from pivtools_core.image_handling import calibration_loader as CL
from pivtools_core.image_handling.path_utils import (
    calibration_camera_folder,
    camera_subfolder_name,
    validate_images_generic,
)
from pivtools_core.image_handling.readers import cine_reader, set_reader

STANDARD_FMT = "cal_%03d.tif"


def _write_config(tmp_path: Path, data: dict) -> Config:
    """Write ``data`` as config.yaml under ``tmp_path`` and load it."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return Config(path=str(path))


def _piv_config(
    tmp_path: Path,
    image_type: str,
    camera_count: int,
    use_camera_subfolders: bool,
    camera_subfolders: list,
) -> Config:
    fmt = {
        "standard": "B%05d.tif",
        "lavision_im7": "B%05d.im7",
        "lavision_set": "run.set",
        "cine": "cam%d.cine",
    }[image_type]
    return _write_config(
        tmp_path,
        {
            "paths": {
                "base_paths": [str(tmp_path / "out")],
                "source_paths": [str(tmp_path / "src")],
                "camera_count": camera_count,
                "camera_numbers": list(range(1, camera_count + 1)),
                "camera_subfolders": camera_subfolders,
            },
            "images": {
                "image_type": image_type,
                "image_format": [fmt],
                "num_images": 4,
                "start_index": 1,
                "frame_stride": 1,
                "pair_stride": 1,
                "use_camera_subfolders": use_camera_subfolders,
            },
        },
    )


# ---------------------------------------------------------------------------
# 1. Subfolder rule — PIV rule on both sides
# ---------------------------------------------------------------------------


def test_standard_multi_camera_uses_cam_n_with_flag_off():
    """The flag no longer gates standard formats: count 2 -> Cam{N}."""
    image = {"use_camera_subfolders": False, "camera_subfolders": []}
    assert calibration_camera_folder(image, "standard", 1, 2) == "Cam1"
    assert calibration_camera_folder(image, "standard", 2, 2) == "Cam2"


def test_standard_multi_camera_explicit_entry_with_flag_off():
    image = {"use_camera_subfolders": False, "camera_subfolders": ["left", "right"]}
    assert calibration_camera_folder(image, "standard", 2, 2) == "right"


def test_im7_still_gated_by_flag():
    off = {"use_camera_subfolders": False, "camera_subfolders": []}
    on = {"use_camera_subfolders": True, "camera_subfolders": []}
    assert calibration_camera_folder(off, "lavision_im7", 2, 2) == ""
    assert calibration_camera_folder(on, "lavision_im7", 2, 2) == "Cam2"


@pytest.mark.parametrize(
    "image_type", ["standard", "lavision_im7", "lavision_set", "cine"]
)
@pytest.mark.parametrize("camera_count", [1, 2])
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("subfolders", [[], ["left", "right"]])
def test_config_and_calibration_agree(
    tmp_path, image_type, camera_count, flag, subfolders
):
    """Config.get_camera_folder and calibration_camera_folder give the same answer."""
    cfg = _piv_config(tmp_path, image_type, camera_count, flag, subfolders)
    image = {"use_camera_subfolders": flag, "camera_subfolders": subfolders}
    for cam in range(1, camera_count + 1):
        expected = camera_subfolder_name(
            image_type,
            cam,
            camera_count,
            use_camera_subfolders=flag,
            camera_subfolders=subfolders,
        )
        assert cfg.get_camera_folder(cam) == expected
        assert (
            calibration_camera_folder(image, image_type, cam, camera_count) == expected
        )


def test_build_calibration_path_standard_multi_camera(tmp_path):
    from pivtools_core.image_handling.path_utils import build_calibration_camera_path

    image = {
        "image_format": STANDARD_FMT,
        "image_type": "standard",
        "use_camera_subfolders": False,
        "camera_subfolders": [],
    }
    assert build_calibration_camera_path(tmp_path, image, 2, 2) == tmp_path / "Cam2"


# ---------------------------------------------------------------------------
# 2. AppleDouble / dot-files do not count
# ---------------------------------------------------------------------------


class _FakeConfig:
    """The two Config members the calibration loader reads."""

    def __init__(self, source: Path, camera_count: int = 1):
        self._source = source
        self.camera_count = camera_count

    def get_calibration_source(self, idx: int) -> Path:
        return self._source


# A pattern that starts with the frame number globs to "*.tif", which also
# matches the AppleDouble sidecar "._00001.tif" macOS writes on exFAT/network
# volumes.
NUMBER_FIRST_FMT = "%05d.tif"


def _write_tifs(folder: Path, numbers) -> None:
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    img = np.full((8, 8), 100, dtype=np.uint8)
    for i in numbers:
        assert cv2.imwrite(str(folder / (NUMBER_FIRST_FMT % i)), img)
    (folder / "._00001.tif").write_bytes(b"\x00\x05\x16\x07")


def _number_first_sidecar(folder: Path) -> None:
    save_settings(
        folder,
        {"image": {"image_format": NUMBER_FIRST_FMT, "image_type": "standard"}},
    )


def test_appledouble_not_counted_as_view(tmp_path):
    _write_tifs(tmp_path, range(1, 4))
    _number_first_sidecar(tmp_path)

    count = CL.get_calibration_frame_count(1, _FakeConfig(tmp_path), 0)
    assert count.n == 3


def test_appledouble_not_counted_in_validate_expected(tmp_path):
    """Blank n_views: validate's expected count is the file count, minus dot-files."""
    _write_tifs(tmp_path, range(1, 4))
    _number_first_sidecar(tmp_path)

    result = CL.validate_calibration_images(1, _FakeConfig(tmp_path), 0)
    assert result["expected_count"] == 3
    assert result["valid"] is True, result["error"]


# ---------------------------------------------------------------------------
# 3. Container shortfall fails validation
# ---------------------------------------------------------------------------


def _frame(_idx):
    return np.zeros((8, 8), dtype=np.uint16)


def test_set_shortfall_is_invalid(tmp_path, monkeypatch):
    set_file = tmp_path / "data.set"
    set_file.write_bytes(b"stub")
    monkeypatch.setattr(set_reader, "get_set_entry_count", lambda _p: 5)

    result = validate_images_generic(
        camera_path=set_file,
        camera=1,
        image_format="data.set",
        image_type="lavision_set",
        expected_count=10,
        zero_based_indexing=False,
        read_frame_fn=_frame,
    )
    assert result["valid"] is False
    assert result["found_count"] == 5
    assert "found 5 entries, expected 10" in result["error"]


def test_cine_shortfall_is_invalid(tmp_path, monkeypatch):
    (tmp_path / "cam1.cine").write_bytes(b"stub")
    monkeypatch.setattr(cine_reader, "get_cine_frame_count", lambda _p: 3)

    result = validate_images_generic(
        camera_path=tmp_path,
        camera=1,
        image_format="cam%d.cine",
        image_type="cine",
        expected_count=7,
        zero_based_indexing=False,
        read_frame_fn=_frame,
    )
    assert result["valid"] is False
    assert "found 3 entries, expected 7" in result["error"]


def test_container_without_expected_count_is_not_checked(tmp_path, monkeypatch):
    """expected_count=None: the caller has no count to hold the container to."""
    set_file = tmp_path / "data.set"
    set_file.write_bytes(b"stub")
    monkeypatch.setattr(set_reader, "get_set_entry_count", lambda _p: 5)

    result = validate_images_generic(
        camera_path=set_file,
        camera=1,
        image_format="data.set",
        image_type="lavision_set",
        expected_count=None,
        zero_based_indexing=False,
        read_frame_fn=_frame,
    )
    assert result["valid"] is True, result["error"]
    assert result["found_count"] == 5


def test_calibration_validate_set_honours_n_views(tmp_path, monkeypatch):
    """A sidecar n_views larger than the .set index fails calibration validate."""
    set_file = tmp_path / "data.set"
    set_file.write_bytes(b"stub")
    save_settings(
        set_file,
        {
            "image": {
                "image_format": "data.set",
                "image_type": "lavision_set",
                "n_views": 9,
            }
        },
    )
    monkeypatch.setattr(set_reader, "get_set_entry_count", lambda _p: 4)
    monkeypatch.setattr(CL, "read_calibration_image", lambda *a, **k: _frame(1))

    result = CL.validate_calibration_images(1, _FakeConfig(set_file), 0)
    assert result["valid"] is False
    assert "found 4 entries, expected 9" in result["error"]


def test_calibration_validate_set_blank_n_views_passes(tmp_path, monkeypatch):
    set_file = tmp_path / "data.set"
    set_file.write_bytes(b"stub")
    save_settings(
        set_file, {"image": {"image_format": "data.set", "image_type": "lavision_set"}}
    )
    monkeypatch.setattr(set_reader, "get_set_entry_count", lambda _p: 4)
    monkeypatch.setattr(CL, "read_calibration_image", lambda *a, **k: _frame(1))

    result = CL.validate_calibration_images(1, _FakeConfig(set_file), 0)
    assert result["valid"] is True, result["error"]


# ---------------------------------------------------------------------------
# 4. CINE mismatch text survives the PIV validate_files route
# ---------------------------------------------------------------------------


def test_cine_mismatch_sets_read_error(tmp_path):
    (tmp_path / "cam1.png").write_bytes(b"\x89PNG")
    result = validate_images_generic(
        camera_path=tmp_path,
        camera=1,
        image_format="cam%d.png",
        image_type="cine",
        expected_count=4,
        zero_based_indexing=False,
        read_frame_fn=_frame,
    )
    assert result["valid"] is False
    assert "Image type is CINE" in result["read_error"]


def test_cine_mismatch_reaches_piv_route(tmp_path, monkeypatch):
    import pivtools_gui.app as gui_app

    src = tmp_path / "src"
    src.mkdir()
    (src / "cam1.png").write_bytes(b"\x89PNG")
    config = _write_config(
        tmp_path,
        {
            "paths": {
                "base_paths": [str(tmp_path / "out")],
                "source_paths": [str(src)],
                "camera_count": 1,
                "camera_numbers": [1],
                "camera_subfolders": [],
            },
            "images": {
                "image_type": "cine",
                "image_format": ["cam%d.png"],
                "num_images": 4,
                "start_index": 1,
                "frame_stride": 1,
                "pair_stride": 1,
                "pairing_preset": "time_resolved",
                "num_loops": 1,
            },
        },
    )
    monkeypatch.setattr(gui_app, "get_config", lambda: config)
    app = Flask(__name__)
    app.register_blueprint(gui_app.api_bp)
    response = app.test_client().post("/backend/validate_files", json={})
    assert response.status_code == 200
    cam = response.get_json()["details"]["camera_1"]
    assert cam["status"] == "error"
    assert "Image type is CINE" in cam["error"]


# ---------------------------------------------------------------------------
# 5. zero_based_indexing on a container calibration source is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image_type, fmt", [("lavision_set", "data.set"), ("cine", "cam%d.cine")]
)
def test_zero_based_container_refused(tmp_path, image_type, fmt):
    camera_path = tmp_path / "data.set" if image_type == "lavision_set" else tmp_path
    with pytest.raises(ValueError, match="zero_based_indexing"):
        CL.read_calibration_frame_at(
            camera_path=camera_path,
            camera=1,
            frame_idx=1,
            image_format=fmt,
            image_type=image_type,
            num_cameras=1,
            zero_based_indexing=True,
        )


# ---------------------------------------------------------------------------
# 6. Null calibration block
# ---------------------------------------------------------------------------


def test_null_calibration_block_reads_as_empty(tmp_path):
    cfg = _write_config(tmp_path, {"calibration": None})
    assert cfg.calibration == {}
    with pytest.raises(ValueError, match="calibration.active is required"):
        cfg.active_calibration_method


# ---------------------------------------------------------------------------
# 10. Pairing migration removed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["time_resolved", "pairing_mode"])
def test_legacy_pairing_keys_refused(tmp_path, key):
    with pytest.raises(ValueError, match=key):
        _write_config(tmp_path, {"images": {key: True, "num_images": 4}})


def test_legacy_pairing_config_not_rewritten(tmp_path):
    """No migration: the file on disk is left exactly as the user wrote it."""
    data = {"images": {"time_resolved": True, "num_images": 4}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        Config(path=str(path))
    assert path.read_text(encoding="utf-8") == before


def test_zero_based_indexing_shim_removed():
    assert not hasattr(Config, "zero_based_indexing")


# ---------------------------------------------------------------------------
# 11. num_images null
# ---------------------------------------------------------------------------


def test_num_images_null_raises(tmp_path):
    cfg = _write_config(tmp_path, {"images": {"num_images": None}})
    with pytest.raises(ValueError, match="images.num_images is required"):
        cfg.num_images


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
