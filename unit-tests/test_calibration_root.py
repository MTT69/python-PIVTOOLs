"""calibration_settings.root_for_source — per-source calibration root layout.

Pins the root each source kind resolves to: a single-file container
(``.set`` / ``.cine``) gets its own ``<parent>/calibration/<stem>`` so several
containers in one folder never share settings, records or ``inputs.mat``; any
other path (including a directory whose name contains a dot) is a directory
source rooted at ``<source>/calibration``.
"""

from __future__ import annotations

import pytest

from pivtools_core import calibration_settings as cs

# ---------------------------------------------------------------------------
# root_for_source
# ---------------------------------------------------------------------------


def test_set_containers_in_one_folder_get_distinct_roots(tmp_path):
    a = cs.root_for_source(tmp_path / "Calib_A.set")
    b = cs.root_for_source(tmp_path / "Calib_B.set")
    assert a == tmp_path / "calibration" / "Calib_A"
    assert b == tmp_path / "calibration" / "Calib_B"
    assert a != b


def test_cine_containers_in_one_folder_get_distinct_roots(tmp_path):
    a = cs.root_for_source(tmp_path / "cam1.cine")
    b = cs.root_for_source(tmp_path / "cam2.cine")
    assert a == tmp_path / "calibration" / "cam1"
    assert b == tmp_path / "calibration" / "cam2"


def test_container_suffix_is_case_insensitive(tmp_path):
    assert cs.root_for_source(tmp_path / "Calib_A.SET") == (
        tmp_path / "calibration" / "Calib_A"
    )
    assert cs.root_for_source(tmp_path / "run.Cine") == tmp_path / "calibration" / "run"


@pytest.mark.parametrize("name", ["2026.10.17", "calib_dt1.5ms"])
def test_dotted_directory_is_a_directory_source(tmp_path, name):
    src = tmp_path / name
    src.mkdir()
    assert cs.root_for_source(src) == src / "calibration"
    assert cs.settings_path(src) == src / "calibration" / cs.SETTINGS_FILENAME


def test_plain_directory_unchanged(tmp_path):
    assert cs.root_for_source(tmp_path / "run1") == tmp_path / "run1" / "calibration"


def test_root_for_source_needs_no_disk(tmp_path):
    """Paths are typed before they exist: resolution is pure path arithmetic."""
    missing = tmp_path / "not_yet" / "Calib_A.set"
    assert (
        cs.root_for_source(missing) == tmp_path / "not_yet" / "calibration" / "Calib_A"
    )
