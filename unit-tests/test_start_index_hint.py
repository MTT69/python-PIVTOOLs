#!/usr/bin/env python3
"""
test_start_index_hint.py

A wrong 0/1 start index is named as the cause on both the PIV and calibration
validation paths, success or failure.

Background: the PIV route computed an ``indexing_warning`` from the first digit run
of up to five ``sample_files`` and the GUI only showed it on success. Calibration
validation had no index check, and ``validate_single_pattern`` dropped a suggestion
equal to the current pattern, so "files 0..9, start index 1" read as a bare
"Pattern incomplete" / "Last frame not found". ``detect_start_index`` now reads the
minimum frame index from the FULL matching listing, parsed with the pattern itself,
and both validators report "Files start at X but the start index is Y" with
``suggested_zero_based``.

Usage:
    pytest unit-tests/test_start_index_hint.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pivtools_gui.app as gui_app  # noqa: E402
from pivtools_core.calibration_settings import save_settings  # noqa: E402
from pivtools_core.config import Config  # noqa: E402
from pivtools_core.image_handling.calibration_loader import (  # noqa: E402
    validate_calibration_images,
)
from pivtools_core.image_handling.path_utils import (  # noqa: E402
    detect_start_index,
    validate_single_pattern,
)

N_FILES = 10


def _write_frames(folder: Path, fmt: str, first: int, n: int = N_FILES) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    img = np.full((16, 16), 128, dtype=np.uint8)
    for i in range(first, first + n):
        assert cv2.imwrite(str(folder / (fmt % i)), img)


# ---------------------------------------------------------------------------
# detect_start_index
# ---------------------------------------------------------------------------


def test_detect_start_index_parses_with_the_pattern():
    """The camera digit in 'Cam1_' is not the frame index; the pattern says so."""
    names = [f"Cam1_{i:05d}.tif" for i in range(0, 12)]
    assert detect_start_index(names, "Cam1_%05d.tif") == 0


def test_detect_start_index_ignores_non_matching_and_unpadded():
    names = ["img10.tif", "img2.tif", "img3.tif", "other7.tif"]
    assert detect_start_index(names, "img%d.tif") == 2


def test_detect_start_index_none_without_matches():
    assert detect_start_index(["a.tif"], "B%05d_A.tif") is None


# ---------------------------------------------------------------------------
# Calibration validate (validate_images_generic)
# ---------------------------------------------------------------------------

CAL_FMT = "cal_%03d.tif"


class _FakeConfig:
    """The two Config members the calibration loader reads."""

    def __init__(self, source: Path):
        self._source = source
        self.camera_count = 1

    def get_calibration_source(self, idx: int) -> Path:
        return self._source


def _cal_validate(tmp_path, first_file, zero_based, n_views=N_FILES):
    _write_frames(tmp_path, CAL_FMT, first_file)
    save_settings(
        tmp_path,
        {
            "image": {
                "image_format": CAL_FMT,
                "image_type": "standard",
                "n_views": n_views,
                "zero_based_indexing": zero_based,
            }
        },
    )
    return validate_calibration_images(1, _FakeConfig(tmp_path), 0)


def test_calibration_files_from_zero_start_one(tmp_path):
    result = _cal_validate(tmp_path, first_file=0, zero_based=False)
    assert result["valid"] is False
    assert result["suggested_zero_based"] is True
    assert result["error"] == (
        "Files start at 0 but the start index is 1 — switch to 0-based indexing"
    )


def test_calibration_files_from_one_start_zero(tmp_path):
    result = _cal_validate(tmp_path, first_file=1, zero_based=True)
    assert result["valid"] is False
    assert result["suggested_zero_based"] is False
    assert "switch to 1-based indexing" in result["error"]


def test_calibration_blank_n_views_checks_the_auto_count(tmp_path):
    """Blank n_views: detection reads start..start+count-1 from the file count, so
    validation must check that range too, not just the first file."""
    result = _cal_validate(tmp_path, first_file=0, zero_based=False, n_views=None)
    assert result["valid"] is False
    assert result["suggested_zero_based"] is True
    assert result["expected_count"] == N_FILES


def test_calibration_blank_n_views_correct_index_valid(tmp_path):
    result = _cal_validate(tmp_path, first_file=1, zero_based=False, n_views=None)
    assert result["valid"] is True, result["error"]
    assert result["expected_count"] == N_FILES


def test_calibration_correct_index_no_hint(tmp_path):
    result = _cal_validate(tmp_path, first_file=1, zero_based=False)
    assert result["valid"] is True, result["error"]
    assert result["suggested_zero_based"] is None
    assert result["detected_start_index"] == 1


def test_calibration_zero_based_files_from_zero_valid(tmp_path):
    result = _cal_validate(tmp_path, first_file=0, zero_based=True)
    assert result["valid"] is True, result["error"]
    assert result["suggested_zero_based"] is None


# ---------------------------------------------------------------------------
# Calibration routes honour the sidecar's zero_based_indexing (GUI 0/1 control)
# ---------------------------------------------------------------------------


def _frame_value(index: int) -> int:
    """Distinct grey level per file index, so a read names the file it came from."""
    return 10 * index + 5


def _cal_client(tmp_path, monkeypatch):
    import pivtools_gui.calibration.app.views as V

    folder = tmp_path / "cal_src"
    folder.mkdir()
    for i in range(0, N_FILES):
        img = np.full((16, 16), _frame_value(i), dtype=np.uint8)
        assert cv2.imwrite(str(folder / (CAL_FMT % i)), img)
    save_settings(
        folder,
        {
            "image": {
                "image_format": CAL_FMT,
                "image_type": "standard",
                "zero_based_indexing": False,
            }
        },
    )
    config = _FakeConfig(folder)
    # The blueprint's helpers also read the calibration pointer block.
    config.calibration = {"active": "dotboard"}
    monkeypatch.setattr(V, "get_config", lambda: config)
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    return app.test_client()


def test_calibration_routes_follow_saved_zero_based(tmp_path, monkeypatch):
    """The GUI's First Frame Number control saves image.zero_based_indexing and
    re-validates; validate, frame reads and the frame count must all use it."""
    client = _cal_client(tmp_path, monkeypatch)
    body = {"camera": 1, "source_path_idx": 0}

    before = client.post("/calibration/validate", json=body).get_json()
    assert before["valid"] is False
    assert before["suggested_zero_based"] is True

    saved = client.post(
        "/calibration/settings",
        json={
            "source_path_idx": 0,
            "settings": {"image": {"zero_based_indexing": True}},
        },
    )
    assert saved.status_code == 200, saved.get_json()

    after = client.post("/calibration/validate", json=body).get_json()
    assert after["valid"] is True, after["error"]
    assert after["suggested_zero_based"] is None

    # Frame numbers stay 1-based in the GUI; 0-based files shift by one. The JSON
    # frame route reports raw stats (no display stretch), so the mean names the file.
    for frame, file_index in ((1, 0), (N_FILES, N_FILES - 1)):
        response = client.get(
            f"/calibration/frame?camera=1&source_path_idx=0&idx={frame}"
        )
        assert response.status_code == 200, response.get_json()
        data = response.get_json()
        assert data["stats"]["mean"] == _frame_value(file_index)
        assert data["frame_count"] == N_FILES


# ---------------------------------------------------------------------------
# validate_single_pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("first_file, zero_based", [(0, False), (1, True)])
def test_single_pattern_reports_start_index(tmp_path, first_file, zero_based):
    _write_frames(tmp_path, "B%05d_A.tif", first_file)
    result = validate_single_pattern(
        camera_path=tmp_path,
        pattern="B%05d_A.tif",
        image_type="standard",
        expected_count=N_FILES,
        zero_based_indexing=zero_based,
        role="A",
    )
    assert result["valid"] is False
    assert result["suggested_zero_based"] is (first_file == 0)
    assert result["error"].startswith(f"Files start at {first_file}")


def test_single_pattern_correct_no_hint(tmp_path):
    _write_frames(tmp_path, "B%05d_A.tif", 1)
    result = validate_single_pattern(
        camera_path=tmp_path,
        pattern="B%05d_A.tif",
        image_type="standard",
        expected_count=N_FILES,
        zero_based_indexing=False,
        role="A",
    )
    assert result["valid"] is True
    assert result["suggested_zero_based"] is None


# ---------------------------------------------------------------------------
# PIV route
# ---------------------------------------------------------------------------


def _piv_client(tmp_path, monkeypatch, first_file, start_index):
    src = tmp_path / "src"
    _write_frames(src, "B%05d_A.tif", first_file)
    _write_frames(src, "B%05d_B.tif", first_file)
    cfg = {
        "paths": {
            "base_paths": [str(tmp_path / "out")],
            "source_paths": [str(src)],
            "camera_count": 1,
            "camera_numbers": [1],
            "camera_subfolders": [],
        },
        "images": {
            "image_type": "standard",
            "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
            "num_images": N_FILES,
            "start_index": start_index,
            "frame_stride": 0,
            "pair_stride": 1,
            "pairing_preset": "ab_format",
            "num_loops": 1,
            "use_camera_subfolders": False,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    config = Config(path=str(config_path))
    monkeypatch.setattr(gui_app, "get_config", lambda: config)
    app = Flask(__name__)
    app.register_blueprint(gui_app.api_bp)
    response = app.test_client().post("/backend/validate_files", json={})
    assert response.status_code == 200
    return response.get_json()["details"]["camera_1"]


@pytest.mark.parametrize(
    "first_file, start_index, zero_based", [(0, 1, True), (1, 0, False)]
)
def test_piv_route_hint_on_failure(
    tmp_path, monkeypatch, first_file, start_index, zero_based
):
    result = _piv_client(tmp_path, monkeypatch, first_file, start_index)
    assert result["status"] == "error"
    assert result["suggested_zero_based"] is zero_based
    assert result["indexing_warning"] == (
        f"Files start at {first_file} but the start index is {start_index} — "
        f"switch to {first_file}-based indexing"
    )
    # The error itself names the cause, not only "first/last frame not found".
    assert result["indexing_warning"] in result["error"]


def test_piv_route_correct_index_no_hint(tmp_path, monkeypatch):
    result = _piv_client(tmp_path, monkeypatch, 1, 1)
    assert result["status"] == "ok", result["error"]
    assert result["indexing_warning"] is None
    assert result["suggested_zero_based"] is None


def test_piv_route_reports_why_no_pattern_was_suggested(tmp_path, monkeypatch):
    """First frame missing and no suggestion possible: the error says why."""
    src = tmp_path / "src"
    src.mkdir()
    img = np.full((16, 16), 128, dtype=np.uint8)
    for i in range(1, N_FILES + 1):
        # Both numbers increase, so neither can be singled out as the frame index.
        assert cv2.imwrite(str(src / f"run{i}_img{i:04d}.tif"), img)
    cfg = {
        "paths": {
            "base_paths": [str(tmp_path / "out")],
            "source_paths": [str(src)],
            "camera_count": 1,
            "camera_numbers": [1],
            "camera_subfolders": [],
        },
        "images": {
            "image_type": "standard",
            "image_format": ["frame%05d.tif"],
            "num_images": N_FILES,
            "start_index": 1,
            "frame_stride": 1,
            "pair_stride": 1,
            "pairing_preset": "time_resolved",
            "num_loops": 1,
            "use_camera_subfolders": False,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    config = Config(path=str(config_path))
    monkeypatch.setattr(gui_app, "get_config", lambda: config)
    app = Flask(__name__)
    app.register_blueprint(gui_app.api_bp)
    result = (
        app.test_client()
        .post("/backend/validate_files", json={})
        .get_json()["details"]["camera_1"]
    )
    assert result["status"] == "error"
    assert "could not tell which number is the frame index" in result["error"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
