"""Calibration view counts never guess (2026-10-18, plan Phase A).

``get_calibration_frame_count`` returns a ``FrameCount(n, source)`` read from the
images themselves -- a file glob, the ``.set`` index or the ``.cine`` header -- or
raises naming what it looked at. It no longer falls back to the sidecar's
``n_views`` or to ``0``: a guessed view count silently detects, persists and solves
the wrong set.

The GUI routes resolve the count through ``_frame_total`` (request value, else
settings ``n_views``, else the auto-count) and report it with its source. Stereo
routes count both cameras and refuse a mismatch, because views are paired by index.
The frame-serving routes keep serving a loadable frame when the count fails, but
report ``frame_count: None`` with the reason instead of a fake ``0``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from flask import Flask

import pivtools_gui.calibration.app.views as V
from pivtools_core.calibration_settings import save_settings
from pivtools_core.image_handling import calibration_loader as CL
from pivtools_core.image_handling.calibration_loader import (
    FrameCount,
    get_calibration_frame_count,
)

FMT = "cal_%03d.tif"
CINE_FMT = "cam%d.cine"


class _FakeConfig:
    """The two Config members the loader reads: the source pointer + rig camera count."""

    def __init__(self, source: Path, camera_count: int = 2):
        self._source = source
        self.camera_count = camera_count

    def get_calibration_source(self, idx: int) -> Path:
        return self._source


def _source(path: Path, **image) -> _FakeConfig:
    """Write a settings sidecar with the given image block and return a config for it."""
    save_settings(path, {"image": image})
    return _FakeConfig(path)


# ---------------------------------------------------------------------------
# get_calibration_frame_count -- standard / im7 file globs
# ---------------------------------------------------------------------------


def _cam1(source: Path) -> Path:
    """Camera 1's folder: a standard multi-camera rig (the fake's default 2) uses Cam{N}."""
    folder = source / "Cam1"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def test_files_counted_with_source(tmp_path):
    for i in range(1, 5):
        (_cam1(tmp_path) / f"cal_{i:03d}.tif").write_bytes(b"x")
    cfg = _source(tmp_path, image_format=FMT, image_type="standard")

    assert get_calibration_frame_count(1, cfg, 0) == FrameCount(4, "files")


def test_stored_n_views_is_not_returned(tmp_path):
    """The sidecar n_views is honoured upstream (_frame_total), never by the counter."""
    for i in range(1, 3):
        (_cam1(tmp_path) / f"cal_{i:03d}.tif").write_bytes(b"x")
    cfg = _source(tmp_path, image_format=FMT, image_type="standard", n_views=9)

    assert get_calibration_frame_count(1, cfg, 0).n == 2


def test_missing_camera_path_raises_naming_path(tmp_path):
    cfg = _source(
        tmp_path,
        image_format=FMT,
        image_type="standard",
        use_camera_subfolders=True,
    )
    with pytest.raises(FileNotFoundError, match="Cam1"):
        get_calibration_frame_count(1, cfg, 0)


def test_zero_matches_raises_naming_dir_and_glob(tmp_path):
    (_cam1(tmp_path) / "unrelated.png").write_bytes(b"x")
    cfg = _source(tmp_path, image_format=FMT, image_type="standard")

    with pytest.raises(ValueError) as err:
        get_calibration_frame_count(1, cfg, 0)
    assert str(tmp_path / "Cam1") in str(err.value) and "cal_*.tif" in str(err.value)


def test_im7_zero_matches_raises(tmp_path):
    cfg = _source(tmp_path, image_format="B%05d.im7", image_type="lavision_im7")

    with pytest.raises(ValueError, match=r"\*\.im7"):
        get_calibration_frame_count(1, cfg, 0)


# ---------------------------------------------------------------------------
# get_calibration_frame_count -- .set index
# ---------------------------------------------------------------------------


def test_set_count_read_from_index(tmp_path, monkeypatch):
    set_file = tmp_path / "board.set"
    set_file.write_bytes(b"stub")
    seen: list[str] = []

    def _count(path):
        seen.append(str(path))
        return 17

    monkeypatch.setattr(CL, "get_set_entry_count", _count)
    cfg = _source(set_file, image_format="board.set", image_type="lavision_set")

    assert get_calibration_frame_count(1, cfg, 0) == FrameCount(17, ".set index")
    assert seen == [str(set_file)]


def test_set_reader_error_raises_chained(tmp_path, monkeypatch):
    set_file = tmp_path / "board.set"
    set_file.write_bytes(b"stub")
    underlying = OSError("companion folder missing")

    def _boom(path):
        raise underlying

    monkeypatch.setattr(CL, "get_set_entry_count", _boom)
    cfg = _source(set_file, image_format="board.set", image_type="lavision_set")

    with pytest.raises(ValueError) as err:
        get_calibration_frame_count(1, cfg, 0)
    assert str(set_file) in str(err.value)
    assert "companion folder missing" in str(err.value)
    assert err.value.__cause__ is underlying


def test_missing_set_file_raises_file_not_found(tmp_path):
    set_file = tmp_path / "board.set"
    cfg = _source(set_file, image_format="board.set", image_type="lavision_set")

    with pytest.raises(FileNotFoundError, match="board.set"):
        get_calibration_frame_count(1, cfg, 0)


def test_set_with_zero_entries_raises(tmp_path, monkeypatch):
    set_file = tmp_path / "board.set"
    set_file.write_bytes(b"stub")
    monkeypatch.setattr(CL, "get_set_entry_count", lambda p: 0)
    cfg = _source(set_file, image_format="board.set", image_type="lavision_set")

    with pytest.raises(ValueError, match="board.set"):
        get_calibration_frame_count(1, cfg, 0)


# ---------------------------------------------------------------------------
# get_calibration_frame_count -- .cine header
# ---------------------------------------------------------------------------


def test_cine_count_read_from_header_per_camera(tmp_path, monkeypatch):
    (tmp_path / "cam2.cine").write_bytes(b"stub")
    seen: list[str] = []

    def _count(path):
        seen.append(str(path))
        return 23

    monkeypatch.setattr(CL, "get_cine_frame_count", _count)
    cfg = _source(tmp_path, image_format=CINE_FMT, image_type="cine")

    assert get_calibration_frame_count(2, cfg, 0) == FrameCount(23, ".cine header")
    assert seen == [str(tmp_path / "cam2.cine")]


def test_missing_cine_file_raises_naming_file(tmp_path):
    cfg = _source(tmp_path, image_format=CINE_FMT, image_type="cine")

    with pytest.raises(ValueError, match="cam1.cine"):
        get_calibration_frame_count(1, cfg, 0)


def test_cine_reader_error_raises_chained(tmp_path, monkeypatch):
    (tmp_path / "cam1.cine").write_bytes(b"stub")
    underlying = RuntimeError("bad header")

    def _boom(path):
        raise underlying

    monkeypatch.setattr(CL, "get_cine_frame_count", _boom)
    cfg = _source(tmp_path, image_format=CINE_FMT, image_type="cine")

    with pytest.raises(ValueError) as err:
        get_calibration_frame_count(1, cfg, 0)
    assert "cam1.cine" in str(err.value) and "bad header" in str(err.value)
    assert err.value.__cause__ is underlying


def test_cine_type_on_png_names_the_mismatch(tmp_path):
    """A CINE image type pointed at a PNG fails on the extension, not in the header reader."""
    png = tmp_path / "calib00001.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    cfg = _source(tmp_path, image_format="calib00001.png", image_type="cine")

    with pytest.raises(ValueError) as err:
        get_calibration_frame_count(1, cfg, 0)
    assert (
        str(err.value) == f"Image type is CINE but the pattern names a .png file: {png}"
    )


# ---------------------------------------------------------------------------
# _frame_total / _pair_frame_total
# ---------------------------------------------------------------------------


def _getter(**values):
    return values.get


@pytest.fixture
def counts(monkeypatch):
    """Per-camera auto-counts the views module sees; no settings n_views."""
    table: dict[int, FrameCount] = {}
    monkeypatch.setattr(V, "get_config", lambda: object())
    monkeypatch.setattr(V, "_settings_idx", lambda idx: {"image": {"n_views": None}})
    monkeypatch.setattr(
        V, "get_calibration_frame_count", lambda cam, cfg, src: table[int(cam)]
    )
    return table


def test_frame_total_request_wins(counts):
    counts[1] = FrameCount(5, "files")
    assert V._frame_total(_getter(frame_total=3), 1, 0) == (3, "request")


@pytest.mark.parametrize("bad", [0, -2, "0"])
def test_frame_total_non_positive_request_raises(counts, bad):
    counts[1] = FrameCount(5, "files")
    with pytest.raises(ValueError, match="frame_total"):
        V._frame_total(_getter(frame_total=bad), 1, 0)


def test_frame_total_settings_n_views(counts, monkeypatch):
    monkeypatch.setattr(V, "_settings_idx", lambda idx: {"image": {"n_views": 7}})
    assert V._frame_total(_getter(), 1, 0) == (7, "n_views setting")


def test_frame_total_auto_count_carries_loader_source(counts):
    counts[1] = FrameCount(12, ".set index")
    assert V._frame_total(_getter(), 1, 0) == (12, ".set index")


def test_frame_total_auto_count_failure_names_camera(counts, monkeypatch):
    def _boom(cam, cfg, src):
        raise FileNotFoundError("/data/Cam1 does not exist")

    monkeypatch.setattr(V, "get_calibration_frame_count", _boom)
    with pytest.raises(ValueError) as err:
        V._frame_total(_getter(), 1, 0)
    assert "camera 1" in str(err.value) and "/data/Cam1" in str(err.value)


def test_pair_frame_total_matching_counts(counts):
    counts[1] = FrameCount(10, "files")
    counts[2] = FrameCount(10, "files")
    assert V._pair_frame_total(_getter(), [1, 2], 0) == (10, "files")


def test_pair_frame_total_mismatch_raises(counts):
    counts[1] = FrameCount(10, "files")
    counts[2] = FrameCount(9, "files")
    with pytest.raises(ValueError) as err:
        V._pair_frame_total(_getter(), [1, 2], 0)
    msg = str(err.value)
    assert "Camera 1 has 10 views" in msg and "camera 2 has 9" in msg


def test_pair_frame_total_explicit_request_not_compared(counts):
    counts[1] = FrameCount(10, "files")
    counts[2] = FrameCount(9, "files")
    assert V._pair_frame_total(_getter(frame_total=4), [1, 2], 0) == (4, "request")


def test_pair_frame_total_settings_not_compared(counts, monkeypatch):
    counts[1] = FrameCount(10, "files")
    counts[2] = FrameCount(9, "files")
    monkeypatch.setattr(V, "_settings_idx", lambda idx: {"image": {"n_views": 6}})
    assert V._pair_frame_total(_getter(), [1, 2], 0) == (6, "n_views setting")


# ---------------------------------------------------------------------------
# Frame-serving routes: a failed count is reported, the frame still served
# ---------------------------------------------------------------------------


class _OneDetection:
    success = False
    diagnostics: dict = {}


class _NeverDetects:
    def detect(self, img):
        return _OneDetection()


@pytest.fixture
def frame_client(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "get_config", lambda: _FakeConfig(tmp_path))
    monkeypatch.setattr(V, "_cfg", lambda: {})
    monkeypatch.setattr(
        V,
        "_load_one",
        lambda cam, frame, src, fmt=None, typ=None, normalize_uint8=True: np.full(
            (8, 10), frame, np.uint8
        ),
    )
    monkeypatch.setattr(
        V,
        "_resolve_board",
        lambda get, overrides=None: ({}, "dotboard", None, _NeverDetects()),
    )

    def _boom(cam, cfg, src):
        raise ValueError("no files matching 'cal_*.tif' in /data/Cam1")

    monkeypatch.setattr(V, "get_calibration_frame_count", _boom)
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    return app.test_client()


def test_frame_json_reports_count_error_and_serves_frame(frame_client):
    resp = frame_client.get("/calibration/frame?camera=1&idx=2&format=png")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data["frame_count"] is None
    assert "cal_*.tif" in data["frame_count_error"]
    assert data["image"] and data["current_idx"] == 2


def test_frame_json_count_ok_has_null_error(frame_client, monkeypatch):
    monkeypatch.setattr(
        V, "get_calibration_frame_count", lambda cam, cfg, src: FrameCount(3, "files")
    )
    data = frame_client.get("/calibration/frame?camera=1&idx=2&format=png").get_json()

    assert data["frame_count"] == 3 and data["frame_count_error"] is None


def test_frame_json_index_beyond_count_rejected(frame_client, monkeypatch):
    monkeypatch.setattr(
        V, "get_calibration_frame_count", lambda cam, cfg, src: FrameCount(3, "files")
    )
    resp = frame_client.get("/calibration/frame?camera=1&idx=4")

    assert resp.status_code == 400
    assert resp.get_json()["frame_count"] == 3


def test_detect_frame_reports_count_error(frame_client):
    data = frame_client.post(
        "/calibration/detect_frame", json={"camera": 1, "frame": 1}
    ).get_json()

    assert data["frame_count"] is None
    assert "cal_*.tif" in data["frame_count_error"]
    assert data["error"] == "board not detected"
