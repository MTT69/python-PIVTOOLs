#!/usr/bin/env python3
"""
test_container_count_validation.py

Regression tests for the .set / container image-count validation fix.

Background: a run driven by a LaVision .set file passed validation with a green
"Matches detected count" tick even though the user entered more images than
existed. Root cause chain:

  * path_utils.py imported get_set_entry_count from the .readers *package*
    (where it is not re-exported) instead of the .readers.set_reader submodule,
    so every .set validation raised ImportError, silently fell back to the
    "container" sentinel, and the real entry count was never read.
  * app.py then substituted the user's own num_images for the sentinel and
    reported it as detected_count, so the frontend equality check always matched.

These tests lock in that the real container count now flows through
validate_images_generic (guards the import), so a regression to the broken
package-level import would surface here.

Usage:
    pytest unit-tests/test_container_count_validation.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_core.image_handling import path_utils
from pivtools_core.image_handling.path_utils import validate_images_generic
from pivtools_core.image_handling.readers import set_reader


def _fake_frame(_idx):
    """A tiny valid 2D image so the .set preview read succeeds."""
    return np.zeros((8, 8), dtype=np.uint16)


def test_get_set_entry_count_importable_from_submodule():
    """The exact import that regressed must work from the concrete submodule."""
    from pivtools_core.image_handling.readers.set_reader import get_set_entry_count

    assert callable(get_set_entry_count)


def test_set_validation_surfaces_real_count(tmp_path, monkeypatch):
    """A .set reports its real entry count, NOT the 'container' sentinel.

    Guards Defect A: if the import in path_utils regresses to the broken
    package-level form, found_count degrades to 'container' and this fails.
    """
    set_file = tmp_path / "data.set"
    set_file.write_bytes(b"stub")  # existence is all validate_images_generic checks
    monkeypatch.setattr(set_reader, "get_set_entry_count", lambda _p: 750)

    result = validate_images_generic(
        camera_path=set_file,
        camera=1,
        image_format="data.set",
        image_type="lavision_set",
        expected_count=2100,  # the original over-count
        zero_based_indexing=False,
        read_frame_fn=_fake_frame,
    )

    assert result["valid"] is True
    # The real, honest count — never the sentinel, never the user's 2100.
    assert result["found_count"] == 750


def test_set_validation_falls_back_only_on_genuine_read_failure(tmp_path, monkeypatch):
    """When the count truly can't be read, fall back to the sentinel (not a fabricated number)."""
    set_file = tmp_path / "data.set"
    set_file.write_bytes(b"stub")

    def _boom(_p):
        raise IOError("corrupt index")

    monkeypatch.setattr(set_reader, "get_set_entry_count", _boom)

    result = validate_images_generic(
        camera_path=set_file,
        camera=1,
        image_format="data.set",
        image_type="lavision_set",
        expected_count=100,
        zero_based_indexing=False,
        read_frame_fn=_fake_frame,
    )

    # Honest "unknown" — a non-int sentinel, so downstream detected_count is None
    # rather than an echo of the user's input.
    assert result["found_count"] == "container"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
