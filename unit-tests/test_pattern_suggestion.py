#!/usr/bin/env python3
"""
test_pattern_suggestion.py

Filename-pattern suggestion picks the digit run that VARIES across the folder.

Background: the old suggester read only the first sorted file, took its first
digit run and capped padding at %05d. ``Cam1_00001.tif`` became ``Cam%d_00001.tif``
(the camera number, not the frame index) and six-digit names became ``%05d``.
The suggester now takes the whole sample list, picks the run position whose value
changes between files, and keeps the run's true width. When it cannot tell which
number is the frame index it says so instead of guessing.

Usage:
    pytest unit-tests/test_pattern_suggestion.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_core.image_handling.path_utils import (  # noqa: E402
    _detect_ab_pair_pattern,
    _suggest_pattern,
    _suggest_pattern_for_role,
    validate_images_generic,
    validate_single_pattern,
)


def _names(fmt, start, stop):
    return [fmt % i for i in range(start, stop + 1)]


@pytest.mark.parametrize(
    "names, expected",
    [
        (_names("Cam1_%05d.tif", 1, 10), "Cam1_%05d.tif"),
        (_names("run3_img%04d_cam1.tif", 1, 10), "run3_img%04d_cam1.tif"),
        (_names("img%d.tif", 1, 12), "img%d.tif"),
        (_names("frame%06d.png", 0, 7), "frame%06d.png"),
        (_names("IMG%05dA.tif", 1, 5), "IMG%05dA.tif"),
    ],
)
def test_varying_run_is_the_frame_index(names, expected):
    """The run that changes between files is chosen, at its true width."""
    suggestion = _suggest_pattern(sorted(names))
    assert suggestion.pattern == expected
    assert suggestion.reason is None


def test_ab_pairs_give_both_patterns():
    names = sorted(_names("B%05d_A.tif", 1, 6) + _names("B%05d_B.tif", 1, 6))
    ab = _detect_ab_pair_pattern(names)
    assert ab is not None
    assert ab["pattern_a"] == "B%05d_A.tif"
    assert ab["pattern_b"] == "B%05d_B.tif"


def test_role_filter_digit_letter_suffix():
    """IMG00001A.tif style (letter directly after the digits) keeps working per role."""
    names = sorted(_names("IMG%05dA.tif", 1, 5) + _names("IMG%05dB.tif", 1, 5))
    assert _suggest_pattern_for_role(names, "A").pattern == "IMG%05dA.tif"
    assert _suggest_pattern_for_role(names, "B").pattern == "IMG%05dB.tif"


def test_single_file_one_run_uses_it():
    assert _suggest_pattern(["Cam_00007.tif"]).pattern == "Cam_%05d.tif"
    assert _suggest_pattern(["img7.tif"]).pattern == "img%d.tif"


def test_single_file_two_runs_is_ambiguous():
    suggestion = _suggest_pattern(["Cam1_00001.tif"])
    assert suggestion.pattern is None
    assert suggestion.reason == (
        "could not tell which number is the frame index in 'Cam1_00001.tif'"
    )


def test_two_runs_varying_together_resolved_by_sort_order():
    """Only the strictly increasing run is the frame index."""
    names = ["a2_0001.tif", "a1_0002.tif", "a3_0003.tif"]
    # Sorted, the first run reads 1,2,3 (strictly increasing) while the 4-digit
    # run reads 2,1,3, so the first run is the frame index.
    assert _suggest_pattern(sorted(names)).pattern == "a%d_0002.tif"


def test_two_runs_both_monotone_is_ambiguous():
    names = ["t1_img0001.tif", "t2_img0002.tif", "t3_img0003.tif"]
    suggestion = _suggest_pattern(names)
    assert suggestion.pattern is None
    assert "t1_img0001.tif" in suggestion.reason


def _fake_frame(_idx):
    return np.zeros((8, 8), dtype=np.uint8)


def test_set_suggestion_is_a_filename(tmp_path):
    """A missing .set suggests the sibling's filename, not its full path."""
    (tmp_path / "recording.set").write_bytes(b"stub")
    result = validate_images_generic(
        camera_path=tmp_path / "missing.set",
        camera=1,
        image_format="missing.set",
        image_type="lavision_set",
        expected_count=1,
        zero_based_indexing=False,
        read_frame_fn=_fake_frame,
    )
    assert result["suggested_pattern"] == "recording.set"


def test_ambiguous_message_reaches_validation_error(tmp_path):
    (tmp_path / "Cam1_00001.tif").write_bytes(b"")
    generic = validate_images_generic(
        camera_path=tmp_path,
        camera=1,
        image_format="wrong%05d.tif",
        image_type="standard",
        expected_count=1,
        zero_based_indexing=False,
        read_frame_fn=_fake_frame,
    )
    assert generic["suggested_pattern"] is None
    assert generic["suggestion_error"] == (
        "could not tell which number is the frame index in 'Cam1_00001.tif'"
    )
    assert generic["suggestion_error"] in generic["error"]

    single = validate_single_pattern(
        camera_path=tmp_path,
        pattern="wrong%05d.tif",
        image_type="standard",
        expected_count=1,
        zero_based_indexing=False,
    )
    assert single["suggested_pattern"] is None
    assert generic["suggestion_error"] in single["error"]


def test_validation_suggests_from_folder(tmp_path):
    for name in _names("Cam1_%05d.tif", 1, 10):
        (tmp_path / name).write_bytes(b"")
    result = validate_images_generic(
        camera_path=tmp_path,
        camera=1,
        image_format="nope%05d.tif",
        image_type="standard",
        expected_count=10,
        zero_based_indexing=False,
        read_frame_fn=_fake_frame,
    )
    assert result["suggested_pattern"] == "Cam1_%05d.tif"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
