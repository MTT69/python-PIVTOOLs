"""calibration_settings — the per-source settings sidecar store + record seeding.

The store replaces the YAML config as the home of pre-model calibration state:
these tests pin the contract (partial deep-merge saves, knob defaults applied
only at load, required keys fail loudly, atomic write) and the GUI seed that
recovers a template from the model records on disk.
"""

from __future__ import annotations

import pytest
import yaml

from pivtools_core import calibration_settings as cs
from pivtools_gui.calibration import record as REC
from pivtools_gui.calibration.pipeline import build_scale_factor_record
from pivtools_gui.calibration.settings_seed import seed_settings


def _source(tmp_path):
    src = tmp_path / "calib_src"
    src.mkdir()
    return src


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_root_for_source_directory_and_container(tmp_path):
    assert cs.root_for_source(tmp_path / "run1") == tmp_path / "run1" / "calibration"
    assert (
        cs.root_for_source(tmp_path / "images.set") == tmp_path / "calibration"
    )


def test_round_trip_with_knob_defaults(tmp_path):
    src = _source(tmp_path)
    cs.save_settings(
        src,
        {
            "image": {"image_format": "calib%05d.tif", "image_type": "standard"},
            "rig": {"dt": 0.004},
        },
    )
    loaded = cs.load_settings(src)
    assert loaded["image"]["image_format"] == "calib%05d.tif"
    assert loaded["rig"]["dt"] == 0.004
    # knob defaults are layered in at load
    assert loaded["methods"]["dotboard"]["k_neighbors"] == 9
    assert loaded["rig"]["interpolator"] == "lanczos"
    assert loaded["schema_version"] == cs.SCHEMA_VERSION


def test_partial_save_preserves_other_blocks(tmp_path):
    src = _source(tmp_path)
    cs.save_settings(
        src,
        {
            "image": {"image_format": "a%03d.png", "image_type": "standard"},
            "methods": {"dotboard": {"dot_spacing_mm": 12.5}},
        },
    )
    cs.save_settings(src, {"methods": {"charuco": {"squares_h": 11}}})
    loaded = cs.load_settings(src)
    assert loaded["methods"]["dotboard"]["dot_spacing_mm"] == 12.5
    assert loaded["methods"]["charuco"]["squares_h"] == 11
    assert loaded["image"]["image_format"] == "a%03d.png"


def test_lists_replace_wholesale(tmp_path):
    src = _source(tmp_path)
    cs.save_settings(
        src,
        {
            "image": {
                "image_format": "a%03d.png",
                "image_type": "standard",
                "camera_subfolders": ["c1", "c2"],
            }
        },
    )
    cs.save_settings(src, {"image": {"camera_subfolders": ["only"]}})
    assert cs.load_settings(src)["image"]["camera_subfolders"] == ["only"]


def test_missing_file_raises_actionable(tmp_path):
    src = _source(tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        cs.load_settings(src)
    assert "init-settings" in str(e.value)
    assert str(src) in str(e.value)
    assert cs.try_load_settings(src) is None


def test_missing_required_image_keys_raise(tmp_path):
    src = _source(tmp_path)
    cs.save_settings(src, {})  # template: required keys still None
    with pytest.raises(ValueError) as e:
        cs.load_settings(src)
    assert "image.image_format" in str(e.value)
    # try_load_settings must NOT swallow a present-but-invalid file
    with pytest.raises(ValueError):
        cs.try_load_settings(src)


def test_schema_version_mismatch_raises(tmp_path):
    src = _source(tmp_path)
    path = cs.settings_path(src)
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.dump(
            {
                "schema_version": 99,
                "image": {"image_format": "a%03d.png", "image_type": "standard"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as e:
        cs.load_settings(src)
    assert "schema_version" in str(e.value)


def test_reload_after_external_edit(tmp_path):
    """The mtime cache must not serve stale values after the file changes."""
    src = _source(tmp_path)
    cs.save_settings(
        src, {"image": {"image_format": "a%03d.png", "image_type": "standard"}}
    )
    assert cs.load_settings(src)["rig"]["dt"] is None
    cs.save_settings(src, {"rig": {"dt": 0.01}})
    assert cs.load_settings(src)["rig"]["dt"] == 0.01


def test_no_tmp_file_left_behind(tmp_path):
    src = _source(tmp_path)
    cs.save_settings(
        src, {"image": {"image_format": "a%03d.png", "image_type": "standard"}}
    )
    leftovers = list(cs.root_for_source(src).glob("*.tmp"))
    assert leftovers == []


# ---------------------------------------------------------------------------
# Seed from records
# ---------------------------------------------------------------------------


def test_seed_defaults_when_no_records(tmp_path):
    src = _source(tmp_path)
    seeded = seed_settings(src)
    # visible starting guesses for the GUI form
    assert seeded["image"]["image_format"] == "calib%05d.tif"
    assert seeded["image"]["image_type"] == "standard"
    assert seeded["rig"]["dt"] is None


def test_seed_recovers_board_meta_from_mono_record(tmp_path):
    src = _source(tmp_path)
    record = build_scale_factor_record(
        camera=2,
        origin_px=(0.0, 0.0),
        px_per_mm=10.0,
        image_size=(100, 100),
        dt=1.0,
    )
    record.board_meta = {
        "geometry": {
            "board_type": "dotboard",
            "dot_spacing_mm": 12.5,
            "k_neighbors": 7,
            "model_type": "pinhole",
            "datum_frame": 3,
        },
        "dt": 0.004,
        "n_views": 24,
    }
    REC.save_mono(record, REC.mono_model_dir_for_source(src, 2, "dotboard"))
    seeded = seed_settings(src)
    assert seeded["methods"]["dotboard"]["dot_spacing_mm"] == 12.5
    assert seeded["methods"]["dotboard"]["k_neighbors"] == 7
    assert seeded["rig"]["dt"] == 0.004
    assert seeded["rig"]["datum_frame"] == 3
    assert seeded["image"]["n_views"] == 24
    assert seeded["rig"]["camera"] == 2


def test_seed_scale_factor_px_per_mm(tmp_path):
    src = _source(tmp_path)
    record = build_scale_factor_record(
        camera=1,
        origin_px=(0.0, 0.0),
        px_per_mm=13.65,
        image_size=(100, 100),
        dt=0.0057,
    )
    REC.save_mono(record, REC.mono_model_dir_for_source(src, 1, "scale_factor"))
    seeded = seed_settings(src)
    assert seeded["methods"]["scale_factor"]["px_per_mm"] == pytest.approx(13.65)
    assert seeded["rig"]["dt"] == pytest.approx(0.0057)


def test_seed_camera_pair_from_stereo_dir_even_if_record_unreadable(tmp_path):
    src = _source(tmp_path)
    model_dir = cs.root_for_source(src) / "stereo_cam2_cam3" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "stereo_model_pinhole.mat").write_bytes(b"not a mat file")
    seeded = seed_settings(src)
    assert seeded["rig"]["camera_pair"] == [2, 3]


def test_load_after_save_into_partial_file_layers_defaults(tmp_path):
    """Regression: save_settings primed the cache with the raw file merge (no
    defaults layered), so a save into a hand-edited partial file made every
    subsequent cached load serve a structure with whole blocks missing."""
    src = _source(tmp_path)
    p = cs.settings_path(src)
    p.parent.mkdir(parents=True)
    p.write_text(
        "schema_version: 1\nimage:\n  image_format: a%03d.tif\n  image_type: standard\n",
        encoding="utf-8",
    )
    cs.save_settings(src, {"rig": {"dt": 0.5}})
    loaded = cs.load_settings(src)
    assert loaded["rig"]["dt"] == 0.5
    assert loaded["methods"]["dotboard"]["k_neighbors"] == 9
    assert loaded["rig"]["interpolator"] == "lanczos"


def test_seed_shared_rig_keys_come_from_newest_record(tmp_path):
    """Regression: dt/datum_frame/n_views were applied per-record, so board
    insertion order (not record age) picked the winner. The rig keys must come
    from the single newest record; per-board geometry stays per-board."""
    import os
    import time

    src = _source(tmp_path)

    def _make(camera, board, dt):
        record = build_scale_factor_record(
            camera=camera,
            origin_px=(0.0, 0.0),
            px_per_mm=10.0,
            image_size=(100, 100),
            dt=dt,
        )
        record.board_meta["geometry"] = {"board_type": board}
        model_dir = REC.mono_model_dir_for_source(src, camera, board)
        REC.save_mono(record, model_dir)
        return next(model_dir.glob("model_*.mat"))

    now = time.time()
    # Cam1/dotboard is globbed FIRST (insertion order) but is the NEWER record;
    # Cam2/scale_factor is applied last but is a day old. Last-write-wins would
    # seed the stale 0.002.
    newer = _make(1, "dotboard", dt=0.001)
    older = _make(2, "scale_factor", dt=0.002)
    os.utime(newer, (now, now))
    os.utime(older, (now - 86400, now - 86400))

    seeded = seed_settings(src)
    assert seeded["rig"]["dt"] == pytest.approx(0.001)
