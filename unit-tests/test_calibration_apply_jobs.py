"""Calibration apply as a guarded, cancellable job.

- Two apply jobs writing the same calibrated output directory would interleave
  per-frame writes, so ``/calibration/apply`` refuses a second job whose output
  overlaps a running one, and names the running job. A different target runs.
- ``/calibration/apply/cancel/<job_id>`` stops a running job cooperatively: the
  worker checks between units, and ``runio._run_frame_tasks`` checks as frames
  complete and cancels the frames not yet started. The job ends ``cancelled`` with
  its frame counts kept; frames already written stay on disk.

The route tests drive the REAL Flask handlers with the unit planner and the
per-run calibrate function replaced by controllable fakes. ``_run_frame_tasks``
is tested directly, on both its inline and its process-pool path.
"""

from __future__ import annotations

import threading
import time
import types
from pathlib import Path
from typing import Optional

import pytest
from flask import Flask

import pivtools_gui.calibration.app.views as V
from pivtools_gui.calibration import runio

N_FRAMES = 10
# Frames the fake run writes before it blocks, so a cancel lands mid-unit.
FRAMES_BEFORE_BLOCK = 3
WAIT_S = 10.0


def _wait(client, url, timeout=WAIT_S):
    """Poll a job-status URL until it reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(url).get_json()
        if j.get("status") in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.02)
    raise AssertionError(f"job at {url} did not finish within {timeout}s")


class _GatedRun:
    """Stand-in for ``calibrate_mono_run``: writes a few frames, then blocks on a gate.

    Released by ``gate.set()`` (finishes all frames) or by the job's cancel event
    (raises ``ApplyCancelled`` with the frames written so far), mirroring how the
    real ``_run_frame_tasks`` reports a cancel.
    """

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.blocked = threading.Event()

    def __call__(
        self,
        record,
        uncal,
        out,
        dt,
        z,
        tx,
        ty,
        progress_cb=None,
        vector_glob=None,
        cancel_event: Optional[threading.Event] = None,
    ):
        written = []
        for k in range(1, N_FRAMES + 1):
            if k == FRAMES_BEFORE_BLOCK + 1:
                self.blocked.set()
                deadline = time.time() + WAIT_S
                while not self.gate.is_set() and time.time() < deadline:
                    if cancel_event is not None and cancel_event.is_set():
                        raise runio.ApplyCancelled(sorted(written), k - 1, N_FRAMES)
                    time.sleep(0.01)
            written.append(str(Path(out) / f"{k:05d}.mat"))
            if progress_cb:
                progress_cb(k, N_FRAMES)
        return written


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Apply route with units keyed on the request's ``out`` name, and a gated run."""
    run = _GatedRun()
    monkeypatch.setattr(
        V, "get_config", lambda: types.SimpleNamespace(vector_format="%05d.mat")
    )
    monkeypatch.setattr(V, "_cfg", lambda: {"active": "scale_factor"})
    monkeypatch.setattr(V, "_source_path", lambda idx: tmp_path)

    def _units(data, full_cfg, source, board, stereo, type_name):
        return [
            {
                "stereo": False,
                "record": types.SimpleNamespace(board_meta={"dt": 1e-4}),
                "label": f"unit {name}",
                "uncal": tmp_path / "uncal",
                "out": tmp_path / name,
            }
            for name in data["outs"]
        ]

    monkeypatch.setattr(V, "_apply_units", _units)
    monkeypatch.setattr(V.c2runio, "calibrate_mono_run", run)
    monkeypatch.setattr(V.c2runio, "stamp_unit", lambda unit, pointer: None)
    monkeypatch.setattr(
        V.c2runio, "build_applied_pointer", lambda source, board, stereo: {}
    )
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    client = app.test_client()
    yield client, tmp_path, run
    # Never leak a blocked worker thread (or a "running" job) into another test.
    run.gate.set()


def _apply(client, outs):
    body = {"board": "scale_factor", "interpolator": "linear", "outs": list(outs)}
    return client.post("/calibration/apply", json=body).get_json()


# ---------------------------------------------------------------------------
# Concurrency guard
# ---------------------------------------------------------------------------


def test_second_apply_on_same_target_is_refused(env):
    client, _base, run = env
    first = _apply(client, ["calibrated"])
    assert first["success"] is True
    assert run.blocked.wait(WAIT_S)

    second = _apply(client, ["calibrated"])

    assert second["success"] is False
    assert first["job_id"] in second["error"]
    run.gate.set()
    assert (
        _wait(client, f"/calibration/apply/status/{first['job_id']}")["status"]
        == "completed"
    )


def test_apply_overlapping_any_unit_is_refused(env):
    """Overlap is per output directory, not only the job's first one."""
    client, _base, run = env
    first = _apply(client, ["a", "b"])
    assert run.blocked.wait(WAIT_S)

    second = _apply(client, ["c", "b"])

    assert second["success"] is False
    assert first["job_id"] in second["error"]


def test_apply_on_different_target_is_allowed(env):
    client, _base, run = env
    first = _apply(client, ["calibrated_a"])
    assert run.blocked.wait(WAIT_S)

    second = _apply(client, ["calibrated_b"])

    assert second["success"] is True
    run.gate.set()
    for job in (first, second):
        assert (
            _wait(client, f"/calibration/apply/status/{job['job_id']}")["status"]
            == "completed"
        )


def test_apply_allowed_again_after_previous_job_finished(env):
    client, _base, run = env
    run.gate.set()
    first = _apply(client, ["calibrated"])
    assert (
        _wait(client, f"/calibration/apply/status/{first['job_id']}")["status"]
        == "completed"
    )

    second = _apply(client, ["calibrated"])

    assert second["success"] is True


def test_job_records_every_output_dir(env):
    client, base, run = env
    run.gate.set()
    job = _apply(client, ["a", "b"])
    status = _wait(client, f"/calibration/apply/status/{job['job_id']}")
    assert status["out_dirs"] == [str(base / "a"), str(base / "b")]


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancel_mid_run_ends_cancelled_with_counts_kept(env):
    client, _base, run = env
    job = _apply(client, ["calibrated"])
    assert run.blocked.wait(WAIT_S)

    resp = client.post(f"/calibration/apply/cancel/{job['job_id']}")

    assert resp.status_code == 200
    assert resp.get_json()["success"] is True
    status = _wait(client, f"/calibration/apply/status/{job['job_id']}")
    assert status["status"] == "cancelled"
    assert status["frame_done"] == FRAMES_BEFORE_BLOCK
    assert status["frame_total"] == N_FRAMES
    assert status["frame_done"] < status["frame_total"]
    assert "remain on disk" in status["error"]
    # The registry entry is dropped once the job ends.
    assert job["job_id"] not in V._apply_cancel_events


def test_cancel_between_units_skips_remaining_units(env, monkeypatch):
    """A cancel landing while a unit finishes stops before the next unit starts."""
    client, _base, run = env
    seen_units = []
    real_run = V.c2runio.calibrate_mono_run

    def _first_unit_then_cancel(record, uncal, out, *a, cancel_event=None, **k):
        seen_units.append(Path(out).name)
        cancel_event.set()
        run.gate.set()
        return real_run(record, uncal, out, *a, cancel_event=None, **k)

    monkeypatch.setattr(V.c2runio, "calibrate_mono_run", _first_unit_then_cancel)
    job = _apply(client, ["a", "b"])
    status = _wait(client, f"/calibration/apply/status/{job['job_id']}")
    assert status["status"] == "cancelled"
    assert seen_units == ["a"]
    assert status["processed"] == 1 and status["total"] == 2


def test_cancelled_target_can_be_applied_again(env):
    client, _base, run = env
    job = _apply(client, ["calibrated"])
    assert run.blocked.wait(WAIT_S)
    client.post(f"/calibration/apply/cancel/{job['job_id']}")
    assert (
        _wait(client, f"/calibration/apply/status/{job['job_id']}")["status"]
        == "cancelled"
    )

    run.gate.set()
    again = _apply(client, ["calibrated"])

    assert again["success"] is True


def test_cancel_unknown_job_is_404(env):
    client, _base, _run = env
    resp = client.post("/calibration/apply/cancel/calibration_apply_nope")
    assert resp.status_code == 404


def test_cancel_finished_job_reports_not_running(env):
    client, _base, run = env
    run.gate.set()
    job = _apply(client, ["calibrated"])
    _wait(client, f"/calibration/apply/status/{job['job_id']}")

    resp = client.post(f"/calibration/apply/cancel/{job['job_id']}")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is False and "completed" in data["error"]


# ---------------------------------------------------------------------------
# runio._run_frame_tasks cancel (inline and process pool)
# ---------------------------------------------------------------------------

# Long enough that the pool cannot drain every frame before the cancel lands.
POOL_TASK_SLEEP_S = 0.2
POOL_TASKS = 20
POOL_WORKERS = 2
CANCEL_AFTER = 2


def _write_frame_task(out_path: str, ctx: Optional[dict] = None) -> str:
    """Picklable per-frame task: sleep, then write one output file."""
    time.sleep(POOL_TASK_SLEEP_S)
    Path(out_path).write_text("frame")
    return out_path


@pytest.mark.parametrize("workers", [1, POOL_WORKERS])
def test_run_frame_tasks_cancel_stops_and_reports_written(
    tmp_path, monkeypatch, workers
):
    monkeypatch.setattr(runio, "get_max_workers", lambda n: workers)
    cancel = threading.Event()
    tasks = [(str(tmp_path / f"{k:05d}.mat"),) for k in range(POOL_TASKS)]

    def cb(done, total):
        if done >= CANCEL_AFTER:
            cancel.set()

    with pytest.raises(runio.ApplyCancelled) as info:
        runio._run_frame_tasks(_write_frame_task, tasks, {}, cb, cancel_event=cancel)

    exc = info.value
    on_disk = sorted(str(p) for p in tmp_path.glob("*.mat"))
    assert exc.total == POOL_TASKS
    assert CANCEL_AFTER <= exc.done < POOL_TASKS
    # Every frame counted done is on disk, and nothing else was written after return.
    assert exc.written == on_disk
    assert len(on_disk) == exc.done


def test_run_frame_tasks_unset_event_runs_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(runio, "get_max_workers", lambda n: 1)
    tasks = [(str(tmp_path / f"{k:05d}.mat"),) for k in range(3)]
    written = runio._run_frame_tasks(
        _write_frame_task, tasks, {}, None, cancel_event=threading.Event()
    )
    assert len(written) == 3
