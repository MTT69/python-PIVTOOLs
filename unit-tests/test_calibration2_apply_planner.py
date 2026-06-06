"""runio.plan_apply_units — the config-derived apply planner shared by the Flask route
and the CLI's --all-paths apply.

The planner is pure (no Flask): given a config-like object + a calibration source, it loads
the model record per unit and derives every (base_path x camera) I/O directory via
get_data_paths. These tests pin: full mono derivation, an active_paths subset, and the
explicit single-unit override.
"""

from __future__ import annotations

from pivtools_gui.calibration2 import record as REC
from pivtools_gui.calibration2 import runio
from pivtools_gui.calibration2.pipeline import build_scale_factor_record


class _FakeCfg:
    """Minimal stand-in for Config exposing only what plan_apply_units reads."""

    def __init__(self, base_paths, camera_numbers, num_frame_pairs):
        self.base_paths = base_paths
        self.camera_numbers = camera_numbers
        self.num_frame_pairs = num_frame_pairs


def _make_source(tmp_path, cams=(1, 2)):
    """A calibration source with a saved scale-factor model per camera."""
    source = tmp_path / "calib"
    for cam in cams:
        recd = build_scale_factor_record(
            camera=cam, origin_px=(0.0, 0.0), px_per_mm=10.0, image_size=(100, 100), dt=1.0)
        REC.save_mono(recd, REC.mono_model_dir_for_source(source, cam, "scale_factor"))
    return source


def test_mono_all_paths_two_by_two(tmp_path):
    source = _make_source(tmp_path)
    cfg = _FakeCfg([tmp_path / "runA", tmp_path / "runB"], [1, 2], 10)
    units = runio.plan_apply_units(cfg, source, "scale_factor", False, "instantaneous")

    assert len(units) == 4
    assert sorted(u["label"] for u in units) == [
        "runA/Cam1", "runA/Cam2", "runB/Cam1", "runB/Cam2"]
    for u in units:
        assert u["stereo"] is False
        assert "uncalibrated_piv" in str(u["uncal"])
        assert "/calibrated_piv/" in str(u["out"]) and "uncalibrated" not in str(u["out"])
        assert u["record"].camera_model.model_type == "scale_factor"


def test_active_paths_subset(tmp_path):
    source = _make_source(tmp_path)
    cfg = _FakeCfg([tmp_path / "runA", tmp_path / "runB"], [1, 2], 10)
    units = runio.plan_apply_units(
        cfg, source, "scale_factor", False, "instantaneous", active_paths=[1])
    assert len(units) == 2
    assert all("runB" in u["label"] for u in units)


def test_explicit_single_unit_override(tmp_path):
    source = _make_source(tmp_path)
    cfg = _FakeCfg([tmp_path / "runA", tmp_path / "runB"], [1, 2], 10)
    units = runio.plan_apply_units(
        cfg, source, "scale_factor", False, "instantaneous",
        camera=1, explicit={"uncal": tmp_path / "u", "out": tmp_path / "o"})
    assert len(units) == 1
    u = units[0]
    assert u["label"] == "manual"
    assert str(u["uncal"]).endswith("/u") and str(u["out"]).endswith("/o")
