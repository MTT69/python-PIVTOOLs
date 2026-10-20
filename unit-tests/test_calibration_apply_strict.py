"""Calibration apply refuses every silent-wrong-result path.

Each test pins one way apply used to write plausible-but-wrong output (or report a
success that wrote nothing) and now fails loudly instead:

- stereo frame sets that differ between cameras (used to be silently intersected);
- a run with zero vector files (used to "succeed" with only coordinates.mat);
- a filled pass with no coordinates (ensemble wrote raw px through, mono dropped it);
- a frame exception in the pool (used to let every queued frame run first);
- planner / route errors that were mislabelled, partial overrides that fell through to
  every dataset, a scalar ``active_paths``, a missing ``camera_pair``;
- a provenance-stamp failure that was swallowed without telling the user;
- the stereo summary angle, which fell back to a different quantity (the R_stereo
  rotation angle) when the record did not carry the optical-axis angle.
"""

from __future__ import annotations

import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import scipy.io
from fixtures.vector_sets import (
    ENSEMBLE_FILE,
    write_stereo_vector_sets,
    write_vector_set,
)
from flask import Flask

import pivtools_gui.calibration.app.views as V
from pivtools_gui.calibration import record as REC
from pivtools_gui.calibration import runio
from pivtools_gui.calibration.camera_model import CameraModel
from pivtools_gui.calibration.pipeline import build_scale_factor_record

VECTOR_GLOB = "*.mat"
WAIT_S = 10.0
# The real layout: pass 0 empty, passes 1 and 2 filled.
PASSES = (None, (4, 5), (8, 10))


def _sf_record(camera: int = 1) -> REC.MonoRecord:
    return build_scale_factor_record(
        camera=camera, origin_px=(0.0, 0.0), px_per_mm=10.0, dt=1.0
    )


# ---------------------------------------------------------------------------
# runio: frame sets, empty runs, passes without coordinates
# ---------------------------------------------------------------------------


def test_stereo_frame_set_mismatch_raises_naming_the_frames(tmp_path):
    cam1, cam2 = tmp_path / "cam1", tmp_path / "cam2"
    write_stereo_vector_sets(
        cam1, cam2, n_frames=6, passes=PASSES, missing_in_cam2=(2, 5)
    )
    out = tmp_path / "out"
    # The record is never reached: the frame-set check runs before any model work.
    with pytest.raises(ValueError) as info:
        runio.reconstruct_stereo_run(
            types.SimpleNamespace(), cam1, cam2, out, dt=1.0, vector_glob=VECTOR_GLOB
        )
    msg = str(info.value)
    assert "2 frame(s)" in msg and "00002.mat" in msg and "00005.mat" in msg
    assert not out.exists() or not any(out.iterdir())


def test_mono_run_with_no_vector_files_raises(tmp_path):
    uncal = tmp_path / "uncal"
    write_vector_set(uncal, n_frames=1, passes=PASSES, with_ensemble=False)
    (uncal / "00001.mat").unlink()
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError) as info:
        runio.calibrate_mono_run(_sf_record(), uncal, out, dt=1.0, vector_glob="B*.mat")
    assert str(uncal) in str(info.value) and "B*.mat" in str(info.value)
    # Nothing written: no orphan coordinates.mat claiming a calibrated run.
    assert not (out / "coordinates.mat").exists()


def test_stereo_run_with_no_vector_files_raises(tmp_path):
    cam1, cam2 = tmp_path / "cam1", tmp_path / "cam2"
    write_stereo_vector_sets(cam1, cam2, n_frames=2, passes=PASSES)
    with pytest.raises(FileNotFoundError, match=r"B\*\.mat"):
        runio.reconstruct_stereo_run(
            types.SimpleNamespace(),
            cam1,
            cam2,
            tmp_path / "out",
            dt=1.0,
            vector_glob="B*.mat",
        )


def _ensemble_ctx(uncal: Path, drop_pass=None):
    coords = runio.read_coordinates(uncal / "coordinates.mat")
    if drop_pass is not None:
        del coords[drop_pass]
    return coords


def test_ensemble_pass_without_coordinates_raises(tmp_path):
    uncal = tmp_path / "uncal"
    write_vector_set(uncal, n_frames=1, passes=PASSES, with_ensemble=True)
    coords = _ensemble_ctx(uncal, drop_pass=2)
    with pytest.raises(ValueError, match="pass 2"):
        runio.calibrate_ensemble_file(
            uncal / ENSEMBLE_FILE,
            tmp_path / ENSEMBLE_FILE,
            _sf_record().camera_model,
            coords,
            dt=1.0,
        )


def test_ensemble_empty_pass_is_skipped(tmp_path):
    """Pass 0 is empty in real runs and has no coordinates: that is not an error."""
    uncal = tmp_path / "uncal"
    write_vector_set(uncal, n_frames=1, passes=PASSES, with_ensemble=True)
    coords = _ensemble_ctx(uncal)
    assert 0 not in coords
    out = tmp_path / ENSEMBLE_FILE
    runio.calibrate_ensemble_file(
        uncal / ENSEMBLE_FILE, out, _sf_record().camera_model, coords, dt=1.0
    )
    er = scipy.io.loadmat(str(out), squeeze_me=True)["ensemble_result"]
    assert np.asarray(er["ux"][0]).size == 0
    assert np.asarray(er["ux"][2]).shape == (8, 10)


def test_mono_frame_pass_without_coordinates_raises(tmp_path):
    uncal = tmp_path / "uncal"
    write_vector_set(uncal, n_frames=1, passes=PASSES, with_ensemble=False)
    coords = _ensemble_ctx(uncal, drop_pass=1)
    ctx = {
        "model": _sf_record().camera_model,
        "coords_px": coords,
        "n_passes": len(PASSES),
        "dt": 1.0,
        "z_world": 0.0,
        "tilt_x": 0.0,
        "tilt_y": 0.0,
    }
    with pytest.raises(ValueError, match="pass 1"):
        runio._mono_frame_task(
            str(uncal / "00001.mat"), str(tmp_path / "00001.mat"), ctx=ctx
        )


# ---------------------------------------------------------------------------
# runio: a frame exception stops the pool and reports what was written
# ---------------------------------------------------------------------------

_FAIL_NAME = "f00.mat"


def _flaky_task(name: str, ctx=None):
    if name == _FAIL_NAME:
        raise RuntimeError("frame exploded")
    time.sleep(0.05)
    with _executed_lock:
        _executed.append(name)
    return name


_executed: list = []
_executed_lock = threading.Lock()


def test_pool_frame_failure_cancels_queue_and_reports_written(monkeypatch):
    """Threads stand in for processes so the task needs no pickling."""
    _executed.clear()
    monkeypatch.setattr(runio, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(runio, "get_max_workers", lambda total: 2)
    tasks = [(_FAIL_NAME,)] + [(f"f{i:02d}.mat",) for i in range(1, 20)]
    with pytest.raises(runio.ApplyFailed) as info:
        runio._run_frame_tasks(_flaky_task, tasks, {}, None)
    exc = info.value
    assert "frame exploded" in str(exc)
    assert isinstance(exc.__cause__, RuntimeError)
    # Frames still queued were cancelled, not run to completion first.
    assert len(_executed) < len(tasks) - 1
    assert sorted(exc.written) == sorted(_executed)
    assert f"{len(exc.written)} frame(s) written" in str(exc)


def test_inline_frame_failure_reports_written(monkeypatch):
    _executed.clear()
    monkeypatch.setattr(runio, "get_max_workers", lambda total: 1)
    tasks = [("f01.mat",), (_FAIL_NAME,), ("f02.mat",)]
    with pytest.raises(runio.ApplyFailed) as info:
        runio._run_frame_tasks(_flaky_task, tasks, {}, None)
    assert info.value.written == ["f01.mat"]
    assert _executed == ["f01.mat"]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class _FakeCfg:
    def __init__(self, base_paths, camera_numbers, num_frame_pairs=10):
        self.base_paths = base_paths
        self.camera_numbers = camera_numbers
        self.num_frame_pairs = num_frame_pairs


def test_planner_names_the_unmodelled_camera(tmp_path):
    source = tmp_path / "calib"
    REC.save_mono(
        _sf_record(1), REC.mono_model_dir_for_source(source, 1, "scale_factor")
    )
    cfg = _FakeCfg([tmp_path / "runA"], [1, 2])
    with pytest.raises(FileNotFoundError) as info:
        runio.plan_apply_units(cfg, source, "scale_factor", False, "instantaneous")
    assert "camera 2 has no scale_factor model — generate it first" in str(info.value)


@pytest.mark.parametrize("bad", [0, 1, "0"])
def test_planner_rejects_scalar_active_paths(tmp_path, bad):
    cfg = _FakeCfg([tmp_path / "runA", tmp_path / "runB"], [1])
    with pytest.raises(ValueError, match="active_paths must be a list"):
        runio.plan_apply_units(
            cfg, tmp_path, "scale_factor", False, "instantaneous", active_paths=bad
        )


def test_planner_stereo_requires_camera_pair(tmp_path):
    cfg = _FakeCfg([tmp_path / "runA"], [1, 2])
    with pytest.raises(ValueError, match="camera_pair is required"):
        runio.plan_apply_units(cfg, tmp_path, "dotboard", True, "instantaneous")


def test_planner_units_carry_their_cameras(tmp_path):
    source = tmp_path / "calib"
    for cam in (2, 3):
        REC.save_mono(
            _sf_record(cam), REC.mono_model_dir_for_source(source, cam, "scale_factor")
        )
    cfg = _FakeCfg([tmp_path / "runA"], [2, 3])
    units = runio.plan_apply_units(cfg, source, "scale_factor", False, "instantaneous")
    assert [u["cameras"] for u in units] == [(2,), (3,)]


def test_stamp_unit_failure_is_returned(monkeypatch, tmp_path):
    def boom(base, pointer):
        raise PermissionError("read-only snapshot")

    monkeypatch.setattr(runio, "stamp_applied_calibration", boom)
    warning = runio.stamp_unit({"base": tmp_path, "label": "runA/Cam1"}, {})
    assert "runA/Cam1" in warning and "read-only snapshot" in warning
    monkeypatch.setattr(runio, "stamp_applied_calibration", lambda b, p: None)
    assert runio.stamp_unit({"base": tmp_path, "label": "x"}, {}) is None


# ---------------------------------------------------------------------------
# Apply route
# ---------------------------------------------------------------------------


def _wait(client, job_id):
    deadline = time.time() + WAIT_S
    while time.time() < deadline:
        j = client.get(f"/calibration/apply/status/{job_id}").get_json()
        if j.get("status") in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.02)
    raise AssertionError(f"apply job {job_id} did not finish")


@pytest.fixture
def route(tmp_path, monkeypatch):
    """Real apply route; config, source and (per test) the planner/run are stubbed."""
    monkeypatch.setattr(
        V, "get_config", lambda: types.SimpleNamespace(vector_format="%05d.mat")
    )
    monkeypatch.setattr(V, "_cfg", lambda: {"active": "scale_factor"})
    monkeypatch.setattr(V, "_source_path", lambda idx: tmp_path)
    monkeypatch.setattr(
        V.c2runio, "build_applied_pointer", lambda source, board, stereo: {}
    )
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    return app.test_client()


def _stub_units(monkeypatch, tmp_path, cams=(1,)):
    def _units(data, full_cfg, source, board, stereo, type_name):
        return [
            {
                "stereo": False,
                "record": types.SimpleNamespace(board_meta={"dt": 1e-4}),
                "label": f"run/Cam{cam}",
                "uncal": tmp_path / "uncal",
                "out": tmp_path / f"out{cam}",
                "cameras": (cam,),
            }
            for cam in cams
        ]

    monkeypatch.setattr(V, "_apply_units", _units)


def _post(client, **body):
    body.setdefault("board", "scale_factor")
    body.setdefault("interpolator", "linear")
    return client.post("/calibration/apply", json=body)


def test_unit_with_zero_frames_never_completes(route, monkeypatch, tmp_path):
    _stub_units(monkeypatch, tmp_path)
    monkeypatch.setattr(V.c2runio, "calibrate_mono_run", lambda *a, **k: [])
    monkeypatch.setattr(V.c2runio, "stamp_unit", lambda u, p: None)
    job = _post(route).get_json()
    status = _wait(route, job["job_id"])
    assert status["status"] == "failed"
    assert "run/Cam1" in status["error"] and "no frames" in status["error"]


def test_stamp_failure_surfaces_as_job_warning(route, monkeypatch, tmp_path):
    _stub_units(monkeypatch, tmp_path, cams=(1, 2))
    monkeypatch.setattr(
        V.c2runio, "calibrate_mono_run", lambda *a, **k: [str(tmp_path / "00001.mat")]
    )
    monkeypatch.setattr(
        V.c2runio,
        "stamp_unit",
        lambda u, p: (
            f"{u['label']}: provenance not stamped" if u["cameras"] == (2,) else None
        ),
    )
    resp = _post(route).get_json()
    assert resp["cameras"] == [1, 2] and resp["n_cameras"] == 2
    status = _wait(route, resp["job_id"])
    assert status["status"] == "completed"
    assert status["warnings"] == ["run/Cam2: provenance not stamped"]
    assert status["cameras"] == [1, 2] and status["n_cameras"] == 2


@pytest.mark.parametrize(
    "exc, expected",
    [
        (
            ValueError("stereo record is for cameras (1, 3) but cameras [1, 2]"),
            "stereo record is for cameras (1, 3) but cameras [1, 2]",
        ),
        (IndexError("source index 4 out of range"), "source index 4 out of range"),
        (
            FileNotFoundError("camera 2 has no dotboard model — generate it first"),
            "camera 2 has no dotboard model — generate it first",
        ),
        (
            FileNotFoundError("calibration model not found: /x/model"),
            "calibration model not found: /x/model — generate it first",
        ),
    ],
)
def test_planner_errors_are_reported_verbatim(route, monkeypatch, exc, expected):
    def _units(*a, **k):
        raise exc

    monkeypatch.setattr(V, "_apply_units", _units)
    body = _post(route).get_json()
    assert body == {"success": False, "error": expected}


@pytest.mark.parametrize(
    "body, missing",
    [
        ({"calibrated_dir": "/o"}, "uncalibrated_dir"),
        ({"uncalibrated_dir": "/u"}, "calibrated_dir"),
        (
            {"stereo": True, "calibrated_dir": "/o", "uncalibrated_dir_cam1": "/a"},
            "uncalibrated_dir_cam2",
        ),
        (
            {
                "stereo": True,
                "uncalibrated_dir_cam1": "/a",
                "uncalibrated_dir_cam2": "/b",
            },
            "calibrated_dir",
        ),
    ],
)
def test_partial_explicit_override_is_a_400(route, monkeypatch, body, missing):
    monkeypatch.setattr(
        V.c2runio,
        "plan_apply_units",
        lambda *a, **k: pytest.fail("a partial override must not reach the planner"),
    )
    resp = _post(route, camera_pair=[1, 2], **body)
    assert resp.status_code == 400
    assert repr(missing) in resp.get_json()["error"]


@pytest.mark.parametrize("bad", [0, 2, "1"])
def test_scalar_active_paths_is_a_400(route, monkeypatch, bad):
    monkeypatch.setattr(
        V.c2runio,
        "plan_apply_units",
        lambda *a, **k: pytest.fail("a scalar active_paths must not reach the planner"),
    )
    resp = _post(route, active_paths=bad)
    assert resp.status_code == 400
    assert "active_paths" in resp.get_json()["error"]


def test_stereo_apply_without_camera_pair_is_a_400(route, monkeypatch):
    monkeypatch.setattr(V, "_rig", lambda get: {})
    monkeypatch.setattr(
        V.c2runio,
        "plan_apply_units",
        lambda *a, **k: pytest.fail("no camera_pair must not reach the planner"),
    )
    resp = _post(route, stereo=True, board="dotboard")
    assert resp.status_code == 400
    assert "camera_pair is required" in resp.get_json()["error"]


def test_stereo_interpolator_needs_the_settings_sidecar(route, monkeypatch):
    """No request interpolator and no settings sidecar: refuse, never assume lanczos."""
    monkeypatch.setattr(
        V.c2runio,
        "plan_apply_units",
        lambda *a, **k: pytest.fail("must refuse before planning"),
    )
    body = _post(route, stereo=True, board="dotboard", interpolator=None).get_json()
    assert body["success"] is False
    assert "No calibration settings" in body["error"]


# ---------------------------------------------------------------------------
# Model summary: stereo angle and camera pair
# ---------------------------------------------------------------------------


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


YAW = 0.2  # rad, each camera; optical axes 0.4 rad apart
ROLL = 1.0  # rad, cam2 only: changes the R_stereo angle, not the optical-axis angle


def _cam(R):
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 512.0], [0, 0, 1.0]])
    return CameraModel(
        K=K,
        dist=np.zeros(5),
        R=R,
        t=np.array([[0.0], [0.0], [700.0]]),
        image_size=(1280, 1024),
        rms=0.2,
    )


def test_stereo_summary_angle_is_the_optical_axis_angle(route, tmp_path):
    """A flat (stereoCalibrate) record does not stamp relative_angle_deg; the summary
    must still report the optical-axis angle, never the R_stereo rotation angle."""
    m1 = _cam(_rot_y(YAW))
    m2 = _cam(_rot_z(ROLL) @ _rot_y(-YAW))
    R = m2.R @ m1.R.T
    record = REC.StereoRecord(
        cam1=1,
        cam2=2,
        board_type="dotboard",
        model1=m1,
        model2=m2,
        R_stereo=R,
        T_stereo=m2.t - R @ m1.t,
        per_view_rms1=[0.1],
        per_view_rms2=[0.1],
        board_meta={
            "spacing_mm": 15.0,
            "n_views": 3,
            "dt": 1e-4,
            "stereo_method": "stereoCalibrate",
            "n_stereo_views": 3,
            "stereo_rms_px": 0.3,
        },
    )
    REC.save_stereo(record, REC.stereo_model_dir_for_source(tmp_path, 1, 2, "dotboard"))
    body = route.get(
        "/calibration/model?board=dotboard&stereo=1&camera_pair=1,2"
    ).get_json()
    assert body["exists"] is True
    rotation_angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    assert abs(rotation_angle - np.degrees(2 * YAW)) > 10.0  # the two really differ
    assert body["stereo_angle_deg"] == pytest.approx(np.degrees(2 * YAW))


def test_stereo_summary_without_camera_pair_reports_error(route, monkeypatch):
    monkeypatch.setattr(V, "_rig", lambda get: {})
    body = route.get("/calibration/model?board=dotboard&stereo=1").get_json()
    assert body["exists"] is False
    assert "camera_pair is required" in body["error"]


# ---------------------------------------------------------------------------
# Reading calibrated stereo output: a uz that does not match ux is an error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("uz_shape", [(2, 2), (0,)])
def test_stereo_vector_uz_shape_mismatch_raises(tmp_path, uz_shape):
    """A 3C file whose uz does not match ux used to read back silently as 2C."""
    from pivtools_core.vector_loading import read_mat_contents

    shape = (3, 4)
    pr = {
        "ux": np.ones(shape),
        "uy": np.ones(shape),
        "uz": np.ones(uz_shape) if uz_shape != (0,) else np.array([]),
        "b_mask": np.zeros(shape, dtype=np.uint8),
    }
    path = tmp_path / "00001.mat"
    scipy.io.savemat(str(path), {"piv_result": pr})
    with pytest.raises(ValueError, match="uz") as info:
        read_mat_contents(str(path))
    assert str(path) in str(info.value)


def test_stereo_vector_matching_uz_reads_as_3c(tmp_path):
    from pivtools_core.vector_loading import read_mat_contents

    shape = (3, 4)
    pr = {
        "ux": np.ones(shape),
        "uy": np.ones(shape),
        "uz": np.full(shape, 2.0),
        "b_mask": np.zeros(shape, dtype=np.uint8),
    }
    path = tmp_path / "00001.mat"
    scipy.io.savemat(str(path), {"piv_result": pr})
    arr = read_mat_contents(str(path))
    assert arr.shape == (1, 4, 3, 4)
    assert np.all(arr[0, 2] == 2.0)
