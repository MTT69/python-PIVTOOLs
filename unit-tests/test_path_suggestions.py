"""Tests for detect_flat_source_files — stale multi-camera config detection.

When camera subfolder paths (source/Cam1, source/Cam2) don't exist but files
matching the image pattern sit directly in the source folder, the GUI suggests
switching to a single-camera setup. These tests cover the detection helper.
"""

from pivtools_core.image_handling.path_utils import detect_flat_source_files


def _touch(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")


def test_flat_layout_detected(tmp_path):
    _touch(tmp_path, [f"{i:05d}.tif" for i in range(1, 4)])
    result = detect_flat_source_files(tmp_path, ["%05d.tif"])
    assert result == ["00001.tif", "00002.tif", "00003.tif"]


def test_files_only_in_camera_subfolder_not_detected(tmp_path):
    _touch(tmp_path / "Cam1", [f"{i:05d}.tif" for i in range(1, 4)])
    assert detect_flat_source_files(tmp_path, ["%05d.tif"]) == []


def test_ab_patterns_detected_when_only_b_matches(tmp_path):
    _touch(tmp_path, [f"B{i:05d}_B.tif" for i in range(1, 3)])
    result = detect_flat_source_files(tmp_path, ["B%05d_A.tif", "B%05d_B.tif"])
    assert result == ["B00001_B.tif", "B00002_B.tif"]


def test_capped_at_five_files(tmp_path):
    _touch(tmp_path, [f"{i:05d}.tif" for i in range(1, 20)])
    assert len(detect_flat_source_files(tmp_path, ["%05d.tif"])) == 5


def test_empty_and_specifierless_patterns_do_not_crash(tmp_path):
    _touch(tmp_path, ["literal.tif"])
    assert detect_flat_source_files(tmp_path, ["", "   "]) == []
    # A pattern without %d globs as a literal filename
    assert detect_flat_source_files(tmp_path, ["literal.tif"]) == ["literal.tif"]


def test_missing_source_dir_returns_empty(tmp_path):
    assert detect_flat_source_files(tmp_path / "nope", ["%05d.tif"]) == []


def test_dotfiles_excluded(tmp_path):
    _touch(tmp_path, ["._00001.tif", "00001.tif"])
    assert detect_flat_source_files(tmp_path, ["%05d.tif"]) == ["00001.tif"]
