"""runio.plan_apply_units — the config-derived apply planner shared by the Flask route
and the CLI's --all-paths apply.

The planner is pure (no Flask): given a config-like object + a calibration source, it loads
the model record per unit and derives every (base_path x camera) I/O directory via
get_data_paths. These tests pin: full mono derivation, an active_paths subset, and the
explicit single-unit override.
"""

from __future__ import annotations

import pytest

from pivtools_core.paths import get_data_paths
from pivtools_gui.calibration import record as REC
from pivtools_gui.calibration import runio
from pivtools_gui.calibration.pipeline import build_scale_factor_record


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
            camera=cam,
            origin_px=(0.0, 0.0),
            px_per_mm=10.0,
            dt=1.0,
        )
        REC.save_mono(recd, REC.mono_model_dir_for_source(source, cam, "scale_factor"))
    return source


def test_mono_all_paths_two_by_two(tmp_path):
    source = _make_source(tmp_path)
    cfg = _FakeCfg([tmp_path / "runA", tmp_path / "runB"], [1, 2], 10)
    units = runio.plan_apply_units(cfg, source, "scale_factor", False, "instantaneous")

    assert len(units) == 4
    assert sorted(u["label"] for u in units) == [
        "runA/Cam1",
        "runA/Cam2",
        "runB/Cam1",
        "runB/Cam2",
    ]
    for u in units:
        assert u["stereo"] is False
        assert "uncalibrated_piv" in str(u["uncal"])
        # Path.parts is separator-agnostic (str() would need "/" vs "\\" per-OS).
        assert "calibrated_piv" in u["out"].parts and "uncalibrated" not in str(
            u["out"]
        )
        assert u["record"].camera_model.model_type == "scale_factor"

    # Each unit carries the base path it came from, so apply can stamp that run's
    # config snapshot with the calibration actually used.
    assert {u["base"] for u in units} == {tmp_path / "runA", tmp_path / "runB"}


def test_active_paths_subset(tmp_path):
    source = _make_source(tmp_path)
    cfg = _FakeCfg([tmp_path / "runA", tmp_path / "runB"], [1, 2], 10)
    units = runio.plan_apply_units(
        cfg, source, "scale_factor", False, "instantaneous", active_paths=[1]
    )
    assert len(units) == 2
    assert all("runB" in u["label"] for u in units)


def test_explicit_single_unit_override(tmp_path):
    source = _make_source(tmp_path)
    cfg = _FakeCfg([tmp_path / "runA", tmp_path / "runB"], [1, 2], 10)
    units = runio.plan_apply_units(
        cfg,
        source,
        "scale_factor",
        False,
        "instantaneous",
        camera=1,
        explicit={"uncal": tmp_path / "u", "out": tmp_path / "o"},
    )
    assert len(units) == 1
    u = units[0]
    assert u["label"] == "manual"
    assert u["uncal"].name == "u" and u["out"].name == "o"
    # Ad-hoc dirs have no base path, so there is no run snapshot to stamp.
    assert u["base"] is None


# ---------------------------------------------------------------------------
# Stereo: the record's camera order, not the request's, pairs data with models
# ---------------------------------------------------------------------------


def _make_stereo_source(tmp_path, cam1, cam2):
    """A calibration source holding one pinhole stereo record solved as (cam1, cam2).

    ``model1`` belongs to ``cam1`` and ``model2`` to ``cam2``; the two models differ
    (opposite yaw) so a swap is observable.
    """
    from test_calibration_self_cal import _stereo_record

    source = tmp_path / "calib"
    record = _stereo_record()
    record.cam1, record.cam2 = int(cam1), int(cam2)
    REC.save_stereo(
        record, REC.stereo_model_dir_for_source(source, cam1, cam2, "dotboard")
    )
    return source


def _uncal(base, cam, nfp=10):
    return get_data_paths(base, nfp, cam, "instantaneous", use_uncalibrated=True)[
        "data_dir"
    ]


def test_stereo_request_matches_record_order(tmp_path):
    """Solved (1, 2), requested (1, 2): Cam1's vectors pair with model1 as before."""
    source = _make_stereo_source(tmp_path, 1, 2)
    base = tmp_path / "runA"
    cfg = _FakeCfg([base], [1, 2], 10)
    (u,) = runio.plan_apply_units(
        cfg, source, "dotboard", True, "instantaneous", camera_pair=[1, 2]
    )
    assert (u["record"].cam1, u["record"].cam2) == (1, 2)
    assert u["uncal1"] == _uncal(base, 1)
    assert u["uncal2"] == _uncal(base, 2)


def test_stereo_reversed_request_follows_record_order(tmp_path):
    """Solved (2, 1), requested (1, 2) — the CLI default. ``uncal1`` must be Cam2's
    vectors because ``model1`` is Cam2's model; pairing by request order silently
    applied Cam2's model to Cam1's vectors."""
    source = _make_stereo_source(tmp_path, 2, 1)
    base = tmp_path / "runA"
    cfg = _FakeCfg([base], [1, 2], 10)
    (u,) = runio.plan_apply_units(
        cfg, source, "dotboard", True, "instantaneous", camera_pair=[1, 2]
    )
    assert (u["record"].cam1, u["record"].cam2) == (2, 1)
    assert u["uncal1"] == _uncal(base, 2)
    assert u["uncal2"] == _uncal(base, 1)
    # The output folder stays the requested pair's, where downstream readers
    # (plotting, statistics) look for it — the 3C field is in the shared world frame.
    assert (
        u["out"]
        == get_data_paths(
            base,
            10,
            1,
            "instantaneous",
            use_stereo=True,
            stereo_camera_pair=(1, 2),
        )["data_dir"]
    )


def test_stereo_explicit_label_names_record_order(tmp_path):
    """Explicit dirs are taken as given (record order); the label says which camera
    each dir is read as, so a reversed pair is visible in the job log."""
    source = _make_stereo_source(tmp_path, 2, 1)
    cfg = _FakeCfg([tmp_path / "runA"], [1, 2], 10)
    (u,) = runio.plan_apply_units(
        cfg,
        source,
        "dotboard",
        True,
        "instantaneous",
        camera_pair=[1, 2],
        explicit={
            "uncal1": tmp_path / "a",
            "uncal2": tmp_path / "b",
            "out": tmp_path / "o",
        },
    )
    assert u["uncal1"].name == "a" and u["uncal2"].name == "b"
    assert u["label"] == "manual/Cam2_Cam1"
    assert u["base"] is None


def test_stereo_record_for_other_cameras_raises(tmp_path):
    """A record whose cameras are not the requested pair (a misfiled model) must
    refuse to apply rather than pair the wrong cameras' vectors with its models."""
    from test_calibration_self_cal import _stereo_record

    source = tmp_path / "calib"
    record = _stereo_record()
    record.cam1, record.cam2 = 1, 3
    REC.save_stereo(record, REC.stereo_model_dir_for_source(source, 1, 2, "dotboard"))
    cfg = _FakeCfg([tmp_path / "runA"], [1, 2], 10)
    with pytest.raises(ValueError, match=r"cameras \(1, 3\) but cameras \[1, 2\]"):
        runio.plan_apply_units(
            cfg, source, "dotboard", True, "instantaneous", camera_pair=[1, 2]
        )
