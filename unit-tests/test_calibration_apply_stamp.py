"""runio.stamp_applied_calibration — apply writes its provenance into the run snapshot.

A run's ``config_<stamp>.yaml`` is archived when PIV *starts*, before any calibration has
been chosen, so its ``calibration`` block still names whatever dataset was configured
before. Apply is the first moment the answer is known; these tests pin that it stamps the
right file, corrects the stale pointer, and leaves the rest of the config alone.
"""

from __future__ import annotations

import logging

import yaml

from pivtools_gui.calibration import runio


def _write_snapshot(base, stamp, calibration):
    """A minimal run snapshot with sections either side of the calibration block."""
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"config_{stamp}.yaml"
    data = {
        "paths": {"base_paths": [str(base)], "source_paths": [r"E:\Synthetic\Planar"]},
        "images": {"num_images": 4000, "image_shape": [2048, 2048]},
        "calibration": calibration,
        "masking": {"enabled": False},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return path


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


STALE = {
    "calibration_sources": [r"E:\Synthetic\calibration_charuco\stereo"],
    "source": r"E:\Synthetic\calibration_charuco\stereo",
    "source_idx": 0,
    "active": "stereo_charuco",
}


def test_stamps_the_newest_snapshot(tmp_path):
    """Several snapshots in one base path -> only the most recent is stamped."""
    base = tmp_path / "run"
    old = _write_snapshot(base, "2026-07-29_09-00-00", dict(STALE))
    mid = _write_snapshot(base, "2026-07-29_12-30-00", dict(STALE))
    new = _write_snapshot(base, "2026-07-29_16-58-41", dict(STALE))

    pointer = runio.build_applied_pointer(
        r"E:\Synthetic\calibration_charuco\planar", "charuco", False
    )
    stamped = runio.stamp_applied_calibration(base, pointer)

    assert stamped == new
    assert _read(new)["calibration"]["active"] == "charuco"
    # The older snapshots belong to earlier runs and must not be touched.
    assert _read(old)["calibration"] == STALE
    assert _read(mid)["calibration"] == STALE


def test_corrects_a_stale_stereo_pointer(tmp_path):
    """The live bug: a planar run whose snapshot names the previous stereo dataset."""
    base = tmp_path / "planar_no_noise"
    snap = _write_snapshot(base, "2026-07-29_16-58-41", dict(STALE))

    planar = r"E:\Synthetic\calibration_charuco\planar"
    runio.stamp_applied_calibration(
        base, runio.build_applied_pointer(planar, "charuco", False)
    )

    cal = _read(snap)["calibration"]
    assert cal == {
        "calibration_sources": [planar],
        "source": planar,
        "source_idx": 0,
        "active": "charuco",
    }


def test_leaves_other_sections_untouched(tmp_path):
    """Only the calibration block changes; every other section round-trips unchanged."""
    base = tmp_path / "run"
    snap = _write_snapshot(base, "2026-07-29_16-58-41", dict(STALE))
    before = _read(snap)

    runio.stamp_applied_calibration(
        base, runio.build_applied_pointer(tmp_path / "calib", "dotboard", False)
    )
    after = _read(snap)

    assert after["calibration"] != before["calibration"]
    for section in ("paths", "images", "masking"):
        assert after[section] == before[section]
    assert list(after) == list(before)  # key order preserved


def test_no_snapshot_warns_and_returns_none(tmp_path, caplog):
    """A base path with no snapshot is surfaced, not silently skipped."""
    base = tmp_path / "empty"
    base.mkdir()

    with caplog.at_level(logging.WARNING, logger=runio.__name__):
        result = runio.stamp_applied_calibration(
            base, runio.build_applied_pointer(tmp_path / "calib", "charuco", False)
        )

    assert result is None
    assert "not stamped" in caplog.text


def test_stereo_active_name_is_fused_and_not_double_prefixed():
    """Apply carries a bare board + a stereo flag; config stores the fused name."""
    assert runio.active_method_name("charuco", False) == "charuco"
    assert runio.active_method_name("charuco", True) == "stereo_charuco"
    assert runio.active_method_name("dotboard", True) == "stereo_dotboard"
    # Callers may pass an already-fused name (config's `active` is a valid --board).
    assert runio.active_method_name("stereo_stepped", True) == "stereo_stepped"


def test_stamp_unit_skips_explicit_units_without_a_base(tmp_path, caplog):
    """An ad-hoc --uncalibrated-dir/--calibrated-dir apply has no snapshot to stamp."""
    unit = {"label": "manual", "base": None}
    with caplog.at_level(logging.INFO, logger=runio.__name__):
        runio.stamp_unit(unit, runio.build_applied_pointer(tmp_path, "charuco", False))
    assert "no base path" in caplog.text


def test_stamp_unit_never_raises(tmp_path, caplog):
    """Vectors are already written — a provenance failure must not fail the apply."""
    unit = {"label": "runA/Cam1", "base": tmp_path / "does_not_exist"}
    with caplog.at_level(logging.WARNING, logger=runio.__name__):
        runio.stamp_unit(unit, runio.build_applied_pointer(tmp_path, "charuco", False))
