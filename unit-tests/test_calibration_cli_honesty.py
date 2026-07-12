"""CLI honesty fixes (triage items A3, A4).

A3 — the CLI's native datum key is ``datum_index`` (0-based view position); the GUI
persists ``datum_frame`` (1-based, ``index = frame - 1``). ``_resolve_datum_index``
bridges the two so a config written by either tool resolves the same way, and refuses
to guess when both are present and disagree.

A4 — board geometry (dot spacing, square count/size) has no safe default: a wrong value
silently rescales every world coordinate. The CLI param builders now raise an actionable
error naming the config path instead of defaulting, unlike detector-tuning knobs which
keep defaults.
"""

from pathlib import Path

import pytest

from pivtools_cli.calibration_cli import (
    _cam_dir,
    _charuco_params_from,
    _dotboard_params_from,
    _resolve_datum_index,
    _stepped_params_from,
)

# ---------------------------------------------------------------------------
# A3 — datum bridge
# ---------------------------------------------------------------------------


def test_datum_index_used_when_present():
    assert _resolve_datum_index({"datum_index": 3}) == 3


def test_datum_frame_bridges_to_index():
    # Config with only the GUI's 1-based datum_frame -> 0-based index.
    assert _resolve_datum_index({"datum_frame": 5}) == 4


def test_datum_neither_defaults_to_first_view():
    assert _resolve_datum_index({}) == 0


def test_datum_both_consistent_is_accepted():
    # Template default datum_index:0 + GUI default datum_frame:1 agree (index 0).
    assert _resolve_datum_index({"datum_index": 0, "datum_frame": 1}) == 0
    assert _resolve_datum_index({"datum_index": 4, "datum_frame": 5}) == 4


def test_datum_both_conflicting_raises():
    with pytest.raises(SystemExit, match="disagree"):
        _resolve_datum_index({"datum_index": 0, "datum_frame": 5})


# ---------------------------------------------------------------------------
# A4 — required board geometry
# ---------------------------------------------------------------------------


def test_dotboard_missing_spacing_raises():
    with pytest.raises(ValueError, match="calibration.dotboard.dot_spacing_mm"):
        _dotboard_params_from({})


def test_dotboard_present_builds():
    p = _dotboard_params_from({"dot_spacing_mm": 15.0})
    assert p.dot_spacing_mm == 15.0
    assert p.k_neighbors == 9  # tuning knob keeps its default


def test_charuco_missing_squares_raises():
    with pytest.raises(ValueError, match="calibration.charuco.squares_h"):
        _charuco_params_from({"squares_v": 7, "square_size": 0.03})


def test_charuco_missing_square_size_raises():
    with pytest.raises(ValueError, match="calibration.charuco.square_size"):
        _charuco_params_from({"squares_h": 10, "squares_v": 7})


def test_charuco_present_builds():
    p = _charuco_params_from({"squares_h": 10, "squares_v": 7, "square_size": 0.03})
    assert (p.squares_h, p.squares_v) == (10, 7)
    assert p.min_corners == 6  # tuning knob keeps its default


def test_stepped_missing_spacing_raises():
    with pytest.raises(ValueError, match="calibration.stepped.dot_spacing_mm"):
        _stepped_params_from({"step_height_mm": 3.0})


def test_stepped_missing_step_height_raises():
    with pytest.raises(ValueError, match="calibration.stepped.step_height_mm"):
        _stepped_params_from({"dot_spacing_mm": 15.0, "board_thickness_mm": 14.8})


def test_stepped_missing_board_thickness_raises():
    # board_thickness only bites transmission stereo, but a silent default there
    # corrupts the opposite-face Z — so it is required like the rest of the geometry.
    with pytest.raises(ValueError, match="calibration.stepped.board_thickness_mm"):
        _stepped_params_from({"dot_spacing_mm": 15.0, "step_height_mm": 3.0})


def test_stepped_present_builds():
    p = _stepped_params_from(
        {"dot_spacing_mm": 15.0, "step_height_mm": 3.0, "board_thickness_mm": 14.8}
    )
    assert (p.dot_spacing_mm, p.step_height_mm, p.board_thickness_mm) == (
        15.0,
        3.0,
        14.8,
    )
    assert p.level_offset_mm is None  # derived (defaults to spacing/2) — stays optional


# ---------------------------------------------------------------------------
# cam_subfolders convergence — _cam_dir delegates the folder name to the same
# resolver the image loader uses (Config.get_calibration_camera_folder), so the
# headless CLI and the GUI/PIV loaders land on the identical per-camera directory.
# ---------------------------------------------------------------------------

ROOT = Path("/data/calib")


class _FolderConfig:
    """Stub Config exposing only the folder resolver _cam_dir delegates to."""

    def __init__(self, folders):
        self._folders = folders  # {camera: folder_name}

    def get_calibration_camera_folder(self, camera):
        return self._folders.get(camera, "")


def test_cam_dir_joins_delegated_folder():
    cfg = _FolderConfig({1: "cam1", 2: "cam2"})
    assert _cam_dir(cfg, ROOT, 1) == ROOT / "cam1"
    assert _cam_dir(cfg, ROOT, 2) == ROOT / "cam2"


def test_cam_dir_empty_folder_returns_source():
    assert _cam_dir(_FolderConfig({}), ROOT, 1) == ROOT
    assert _cam_dir(_FolderConfig({1: ""}), ROOT, 1) == ROOT
