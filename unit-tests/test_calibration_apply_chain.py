"""Calibration apply, unstubbed: Flask route -> planner -> model load -> calibrate -> files.

``test_calibration_apply_jobs.py`` drives the route with the planner and the per-run
calibrate function replaced by fakes, so it cannot see a break between the pieces. This
file keeps every production step real:

* the planner (``plan_apply_units`` with the real ``get_data_paths`` layout),
* the model load (records written by ``save_mono`` / ``save_stereo``),
* ``calibrate_mono_run`` / ``reconstruct_stereo_run`` on vector sets written in the real
  on-disk layout (``fixtures.vector_sets``),
* ``stamp_unit`` (wrapped in a spy, still calling through).

Only the config source is faked, exactly as the existing env fixture does (``get_config``,
``_cfg``, ``_source_path``). Frame work runs inline (worker count patched to 1) so frame
counts are deterministic and no process pool is spawned.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
import scipy.io
from fixtures.vector_sets import (
    BASE_UX_PX,
    BASE_UY_PX,
    DEFAULT_PASSES,
    ENSEMBLE_UU_PX2,
    ENSEMBLE_UV_PX2,
    ENSEMBLE_VV_PX2,
    GridSpec,
    config_with_cameras,
    write_stereo_vector_sets,
    write_vector_set,
)
from flask import Flask
from test_calibration_self_cal import _stereo_record

import pivtools_gui.calibration.app.views as V
from pivtools_core.paths import get_data_paths
from pivtools_gui.calibration import record as REC
from pivtools_gui.calibration import runio
from pivtools_gui.calibration.pipeline import build_scale_factor_record

N_FRAMES = 4
NUM_FRAME_PAIRS = 10
WAIT_S = 20.0
TYPE_NAME = "instantaneous"
# Camera -> (pixels per mm, model dt s). Different per camera so a mix-up is visible.
SCALE_FACTOR_RIG = {1: (10.0, 1e-3), 2: (20.0, 2e-3)}
# Stereo frames written before the cancel lands.
FRAMES_BEFORE_CANCEL = 3
# Grid inside both stereo cameras' overlap (centred on the principal point of
# test_calibration_self_cal's 1600x1200 pair).
STEREO_GRID = GridSpec(origin_px=(600.0, 450.0), extent_px=(400.0, 300.0))


def _wait(client, job_id: str) -> dict:
    deadline = time.time() + WAIT_S
    while time.time() < deadline:
        j = client.get(f"/calibration/apply/status/{job_id}").get_json()
        if j.get("status") in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.02)
    raise AssertionError(f"apply job {job_id} did not finish within {WAIT_S}s")


def _uncal(base: Path, cam: int) -> Path:
    return get_data_paths(base, NUM_FRAME_PAIRS, cam, TYPE_NAME, use_uncalibrated=True)[
        "data_dir"
    ]


def _read_struct(path: Path, name: str) -> np.ndarray:
    return scipy.io.loadmat(str(path))[name].reshape(-1)


@pytest.fixture
def chain(tmp_path, monkeypatch):
    """Real apply route over a tmp dataset. Yields ``(client, base, source, stamps)``."""
    base = tmp_path / "run"
    source = tmp_path / "calib"
    cfg = config_with_cameras(
        [1, 2], [base], num_frame_pairs=NUM_FRAME_PAIRS, vector_format="%05d.mat"
    )
    monkeypatch.setattr(V, "get_config", lambda: cfg)
    monkeypatch.setattr(V, "_cfg", lambda: {"active": "scale_factor"})
    monkeypatch.setattr(V, "_source_path", lambda idx: source)
    monkeypatch.setattr(runio, "get_max_workers", lambda n_items=None, config=None: 1)

    stamps: List[dict] = []
    real_stamp = runio.stamp_unit

    def _spy(unit, pointer):
        stamps.append(
            {"label": unit["label"], "base": unit["base"], "pointer": pointer}
        )
        real_stamp(unit, pointer)

    monkeypatch.setattr(V.c2runio, "stamp_unit", _spy)

    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    yield app.test_client(), base, source, stamps


def _save_scale_factor_models(source: Path) -> None:
    for cam, (ppm, dt) in SCALE_FACTOR_RIG.items():
        rec = build_scale_factor_record(
            camera=cam, origin_px=(0.0, 0.0), px_per_mm=ppm, dt=dt
        )
        REC.save_mono(rec, REC.mono_model_dir_for_source(source, cam, "scale_factor"))


# ---------------------------------------------------------------------------
# Mono: real scale-factor model, two cameras
# ---------------------------------------------------------------------------


def test_scale_factor_apply_writes_calibrated_ms_for_every_camera(chain):
    client, base, source, stamps = chain
    _save_scale_factor_models(source)
    for cam in SCALE_FACTOR_RIG:
        write_vector_set(_uncal(base, cam), N_FRAMES)

    resp = client.post(
        "/calibration/apply", json={"board": "scale_factor", "interpolator": "linear"}
    ).get_json()
    assert resp["success"] is True, resp
    status = _wait(client, resp["job_id"])
    assert status["status"] == "completed", status

    # N vector files + the ensemble file, per camera.
    assert status["n_frames"] == 2 * (N_FRAMES + 1)
    pass_shape = DEFAULT_PASSES[2]
    for cam, (ppm, dt) in SCALE_FACTOR_RIG.items():
        out = get_data_paths(base, NUM_FRAME_PAIRS, cam, TYPE_NAME)["data_dir"]
        for k in range(1, N_FRAMES + 1):
            pr = _read_struct(out / f"{k:05d}.mat", "piv_result")
            assert pr["ux"][0].size == 0, "pass 0 stays empty"
            # x_dir right (col_sign +1), y_dir up (row_sign -1): v flips the image-down y.
            expect_u = k * BASE_UX_PX / ppm / 1000.0 / dt
            expect_v = -k * BASE_UY_PX / ppm / 1000.0 / dt
            np.testing.assert_allclose(pr["ux"][2], expect_u, rtol=1e-9)
            np.testing.assert_allclose(pr["uy"][2], expect_v, rtol=1e-9)
            assert pr["ux"][2].shape == pass_shape
            assert pr["b_mask"][2].dtype == np.uint8

        coords = _read_struct(out / "coordinates.mat", "coordinates")
        assert coords["x"][0].size == 0
        assert coords["x"][2].shape == pass_shape

        er = _read_struct(out / "ensemble_result.mat", "ensemble_result")
        scale = 1.0 / (ppm**2 * dt**2 * 1e6)
        np.testing.assert_allclose(er["UU"][2], ENSEMBLE_UU_PX2 * scale, rtol=1e-9)
        np.testing.assert_allclose(er["VV"][2], ENSEMBLE_VV_PX2 * scale, rtol=1e-9)
        # The y flip (row_sign -1) negates the cross term.
        np.testing.assert_allclose(er["UV"][2], -ENSEMBLE_UV_PX2 * scale, rtol=1e-9)


def test_scale_factor_apply_stamps_every_unit_once(chain):
    client, base, source, stamps = chain
    _save_scale_factor_models(source)
    for cam in SCALE_FACTOR_RIG:
        write_vector_set(_uncal(base, cam), 2, with_ensemble=False)

    resp = client.post(
        "/calibration/apply", json={"board": "scale_factor", "interpolator": "linear"}
    ).get_json()
    assert _wait(client, resp["job_id"])["status"] == "completed"

    assert [s["label"] for s in stamps] == ["run/Cam1", "run/Cam2"]
    assert all(s["base"] == base for s in stamps)
    assert {s["pointer"]["active"] for s in stamps} == {"scale_factor"}
    assert {s["pointer"]["source"] for s in stamps} == {str(source)}


# ---------------------------------------------------------------------------
# Stereo: real pinhole pair, cancel mid-run
# ---------------------------------------------------------------------------


def test_stereo_cancel_mid_run_leaves_exactly_the_written_files(chain, monkeypatch):
    client, base, source, stamps = chain
    REC.save_stereo(
        _stereo_record(),
        REC.stereo_model_dir_for_source(source, 1, 2, "dotboard"),
    )
    write_stereo_vector_sets(
        _uncal(base, 1),
        _uncal(base, 2),
        N_FRAMES + 3,
        grid=STEREO_GRID,
    )

    reached = threading.Event()
    real_run = runio.reconstruct_stereo_run

    def _blocks_after_some_frames(*args, progress_cb=None, cancel_event=None, **kw):
        """The real reconstruction; after FRAMES_BEFORE_CANCEL frames it waits for the cancel."""

        def cb(done, total):
            if progress_cb:
                progress_cb(done, total)
            if done == FRAMES_BEFORE_CANCEL:
                reached.set()
                deadline = time.time() + WAIT_S
                while not cancel_event.is_set() and time.time() < deadline:
                    time.sleep(0.01)

        return real_run(*args, progress_cb=cb, cancel_event=cancel_event, **kw)

    monkeypatch.setattr(V.c2runio, "reconstruct_stereo_run", _blocks_after_some_frames)

    resp = client.post(
        "/calibration/apply",
        json={
            "board": "dotboard",
            "stereo": True,
            "camera_pair": [1, 2],
            "interpolator": "linear",
            "dt": 1e-3,
        },
    ).get_json()
    assert resp["success"] is True, resp
    assert reached.wait(WAIT_S)
    assert client.post(f"/calibration/apply/cancel/{resp['job_id']}").get_json()[
        "success"
    ]
    status = _wait(client, resp["job_id"])

    assert status["status"] == "cancelled", status
    out = get_data_paths(
        base,
        NUM_FRAME_PAIRS,
        1,
        TYPE_NAME,
        use_stereo=True,
        stereo_camera_pair=(1, 2),
    )["data_dir"]
    on_disk = sorted(p.name for p in out.iterdir())
    assert on_disk == ["00001.mat", "00002.mat", "00003.mat", "coordinates.mat"]
    assert status["n_frames"] == FRAMES_BEFORE_CANCEL
    assert status["frame_done"] == FRAMES_BEFORE_CANCEL
    assert status["frame_total"] == N_FRAMES + 3
    # A half-finished unit is not stamped: its run config would claim a full calibration.
    assert stamps == []
    # Every kept frame is a real 3C field.
    pr = _read_struct(out / "00001.mat", "piv_result")
    assert {"ux", "uy", "uz", "b_mask"} <= set(pr.dtype.names)
    assert np.isfinite(np.asarray(pr["ux"][-1], dtype=float)).any()
