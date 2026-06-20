"""S4 — calibration stepped Flask routes (the GUI API), end-to-end.

Drives the REAL route handlers, the REAL background job threads, and the REAL
``calibrate_stepped_mono`` / ``calibrate_stepped_stereo`` fit. Only the two
environment seams are mocked: ``read_calibration_image`` (returns the in-memory
synthetic renders instead of reading disk) and ``get_config().get_calibration_source``
(points the saved model at ``tmp_path``). The synthetic stepped scene is reused from
the S3 stereo test so the geometry is identical to the calibrator-level tests.

What is asserted:
- the detect -> label -> snap -> generate sequence completes and saves a model;
- the per-pose detection payload carries the level breakdown but NEVER the
  ``_``-prefixed internal diagnostics (the serialiser contract);
- mono + stereo both recover sub-pixel RMS through the HTTP surface;
- the stereo route auto-classifies same_side and writes proof figures;
- bad input (missing fiducial, expired sequence) fails loudly, not silently.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from flask import Flask

import pivtools_gui.calibration.app.stepped_views as sv
from pivtools_cli.synthetic_calibration_common import make_camera_matrix

# Reuse the S3 synthetic stepped scene (identical geometry to the calibrator tests).
from test_calibration_stepped_stereo import (  # noqa: E402
    PEAK_COLS,
    POSE_RVECS,
    RVEC_EXTRA,
    SPACING_MM,
    STEP_MM,
    W,
    H,
    _cam1_pose,
    _cam2_pose,
    _fiducials,
    _gt_z,
    _level_a_label,
    _render,
)
from pivtools_gui.calibration.detection.stepped import SteppedDetector, SteppedParams


# ---------------------------------------------------------------------------
# Fake environment: in-memory image source + tmp model dir
# ---------------------------------------------------------------------------

class _FakeConfig:
    """Minimal stand-in for the app config touched by the stepped routes."""

    def __init__(self, source):
        self.calibration = {
            "camera": 1,
            "camera_pair": [1, 2],
            "source_idx": 0,
            "stepped": {"dot_spacing_mm": SPACING_MM, "step_height_mm": STEP_MM,
                        "board_thickness_mm": 14.8},
        }
        self._source = source

    def get_calibration_source(self, idx):
        return self._source


def _build_scene():
    """Per-camera, per-pose synthetic renders + ground-truth fiducials / pose labels."""
    K = make_camera_matrix(W, H)
    R_extra, _ = cv2.Rodrigues(np.asarray(RVEC_EXTRA, np.float64))
    imgs1, projs1, imgs2, projs2 = [], [], [], []
    for rv in POSE_RVECS:
        R1, t1, Z, _ = _cam1_pose(rv, K)
        im1, pr1 = _render(R1, t1, K)
        R2, t2 = _cam2_pose(R1, t1, Z, R_extra)
        im2, pr2 = _render(R2, t2, K)
        imgs1.append(im1); projs1.append(pr1)
        imgs2.append(im2); projs2.append(pr2)

    det = SteppedDetector(SteppedParams(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM))
    gt_z = _gt_z()
    pl1 = [_level_a_label(det.detect(im), p, gt_z) for im, p in zip(imgs1, projs1)]
    pl2 = [_level_a_label(det.detect(im), p, gt_z) for im, p in zip(imgs2, projs2)]
    return {
        "images": {1: imgs1, 2: imgs2},
        "fiducials": {1: _fiducials(projs1[0]), 2: _fiducials(projs2[0])},
        "pose_levels": {1: pl1, 2: pl2},
    }


@pytest.fixture(scope="module")
def scene():
    return _build_scene()


@pytest.fixture
def client(scene, tmp_path, monkeypatch):
    """Flask test client with the image reader + config seams mocked."""
    cfg = _FakeConfig(tmp_path)
    monkeypatch.setattr(sv, "get_config", lambda: cfg)

    def _fake_read(frame, camera, config, source_idx, image_format=None, image_type=None):
        return scene["images"][int(camera)][int(frame) - 1]

    monkeypatch.setattr(sv, "read_calibration_image", _fake_read)

    app = Flask(__name__)
    app.register_blueprint(sv.calibration_stepped_bp)
    app.config.update(TESTING=True)
    return app.test_client()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait(client, url, timeout=60.0):
    """Poll a job-status URL until completed/failed (jobs run in daemon threads)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(url).get_json()
        if j.get("status") in ("completed", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job at {url} did not finish within {timeout}s")


FRAMES = list(range(1, len(POSE_RVECS) + 1))


def _detect(client, cameras):
    r = client.post("/calibration/stepped/detect_sequence", json={
        "source_path_idx": 0, "cameras": cameras,
        "num_frames": len(FRAMES), "start_frame_idx": 1, "datum_frame_idx": 1,
    })
    body = r.get_json()
    assert r.status_code == 200 and "sequence_id" in body, body
    done = _wait(client, f"/calibration/stepped/detect_sequence/status/{body['job_id']}")
    assert done["status"] == "completed", done
    return body["sequence_id"], done


def _spec(scene, camera):
    pose_levels = {str(fi): lbl for fi, lbl in zip(FRAMES, scene["pose_levels"][camera])}
    return {
        "fiducials": scene["fiducials"][camera],
        "clicked_level": "peak",
        "pose_levels": pose_levels,
    }


# ---------------------------------------------------------------------------
# 1. The serialiser contract: no _-prefixed diagnostics ever leak
# ---------------------------------------------------------------------------

def test_pose_detection_strips_internal_keys(client, scene):
    sid, done = _detect(client, [1])
    # Datum detection embedded in the completed job is JSON-safe.
    datum = done["datum_detection"]["1"]
    assert datum["ok"] and datum["level_a"] and datum["level_b"]
    # The per-pose endpoint payload carries the overlay data, not the heavy dicts.
    r = client.get("/calibration/stepped/sequence_pose_detection",
                   query_string={"sequence_id": sid, "camera": 1, "frame_idx": 1})
    payload = r.get_json()
    assert r.status_code == 200 and payload["ok"]
    assert "image_points" in payload and "level_labels" in payload
    flat = str(payload)
    assert "_level_a_full" not in flat and "_blob_info" not in flat
    assert not any(k.startswith("_") for k in payload)


def test_identify_pose_level_and_snap(client, scene):
    sid, _ = _detect(client, [1])
    fid = scene["fiducials"][1]
    # identify_pose_level returns a level letter + the snapped dot.
    r = client.post("/calibration/stepped/identify_pose_level", json={
        "sequence_id": sid, "camera": 1, "frame_idx": 1,
        "click_x": fid["origin"][0], "click_y": fid["origin"][1]})
    j = r.get_json()
    assert r.status_code == 200 and j["level"] in ("A", "B")
    assert np.hypot(j["snapped_x"] - fid["origin"][0],
                    j["snapped_y"] - fid["origin"][1]) < SPACING_MM
    # snap_fiducial snaps onto the datum pose and returns its grid index.
    r = client.post("/calibration/stepped/snap_fiducial", json={
        "sequence_id": sid, "camera": 1,
        "click_x": fid["origin"][0] + 3, "click_y": fid["origin"][1] - 3})
    s = r.get_json()
    assert r.status_code == 200 and "grid_col" in s and "grid_row" in s


# ---------------------------------------------------------------------------
# 2. Mono end-to-end through HTTP
# ---------------------------------------------------------------------------

def test_mono_generate_end_to_end(client, scene, tmp_path):
    sid, _ = _detect(client, [1])
    r = client.post("/calibration/stepped/generate_model", json={
        "sequence_id": sid, "stereo": False,
        "cameras": {"1": _spec(scene, 1)}})
    body = r.get_json()
    assert r.status_code == 200, body
    done = _wait(client, f"/calibration/stepped/generate_model/status/{body['job_id']}")
    assert done["status"] == "completed", done
    assert not done["stereo"]
    assert done["rms"] < 1.0, done["rms"]
    assert max(done["per_view_rms"]) < 1.0
    assert done["num_views_used"] == len(FRAMES)
    # Model saved to the (mocked) source, with proof figures.
    assert (tmp_path / "calibration").exists() or done["model_path"]
    assert any("reprojection" in f for f in done["figures"]), done["figures"]
    assert done["model_path"].endswith(".mat")


# ---------------------------------------------------------------------------
# 3. Stereo end-to-end through HTTP (auto same_side)
# ---------------------------------------------------------------------------

def test_stereo_generate_end_to_end(client, scene):
    sid, done_det = _detect(client, [1, 2])
    assert set(done_det["datum_detection"]) == {"1", "2"}
    r = client.post("/calibration/stepped/generate_model", json={
        "sequence_id": sid, "stereo": True, "stereo_config": "auto",
        "cameras": {"1": _spec(scene, 1), "2": _spec(scene, 2)}})
    body = r.get_json()
    assert r.status_code == 200, body
    done = _wait(client, f"/calibration/stepped/generate_model/status/{body['job_id']}")
    assert done["status"] == "completed", done
    assert done["stereo"] and done["stereo_config"] == "same_side"
    assert done["rms_cam1"] < 1.0 and done["rms_cam2"] < 1.0
    assert done["num_pairs_used"] == len(FRAMES)
    assert done["baseline_mm"] > 0
    assert any("cam1_reprojection" in f for f in done["figures"]), done["figures"]
    assert any("cameras_3d" in f for f in done["figures"]), done["figures"]


# ---------------------------------------------------------------------------
# 4. Loud failures
# ---------------------------------------------------------------------------

def test_generate_missing_fiducial_fails_loudly(client, scene):
    sid, _ = _detect(client, [1])
    spec = _spec(scene, 1)
    spec["fiducials"] = {"origin": scene["fiducials"][1]["origin"]}  # missing +X/+Y
    r = client.post("/calibration/stepped/generate_model", json={
        "sequence_id": sid, "stereo": False, "cameras": {"1": spec}})
    assert r.status_code == 400 and "fiducial" in r.get_json()["error"]


def test_expired_sequence_returns_410(client):
    r = client.get("/calibration/stepped/sequence_pose_detection",
                   query_string={"sequence_id": "deadbeef", "camera": 1, "frame_idx": 1})
    assert r.status_code == 410 and "not found" in r.get_json()["error"]


# ---------------------------------------------------------------------------
# 5. Restore the overlay from the sidecar after the in-memory cache is gone
# ---------------------------------------------------------------------------

def test_restore_sequence_rehydrates_from_sidecar(client, scene):
    """detect + generate persists detections + clicks to inputs.mat; after the in-memory
    cache is dropped (reload), restore_sequence rebuilds a live sequence from the sidecar so
    the GUI can re-show the overlay + fiducials + peak/trough without re-detecting."""
    sid, _ = _detect(client, [1, 2])
    gen = client.post("/calibration/stepped/generate_model", json={
        "sequence_id": sid, "stereo": True, "stereo_config": "auto",
        "cameras": {"1": _spec(scene, 1), "2": _spec(scene, 2)}})
    done = _wait(client, f"/calibration/stepped/generate_model/status/{gen.get_json()['job_id']}")
    assert done["status"] == "completed", done

    # Simulate a reload / new session: the in-memory cache is gone, only the sidecar remains.
    sv._sequence_cache.clear()

    r = client.get("/calibration/stepped/restore_sequence",
                   query_string={"camera_pair": "1,2", "source_path_idx": 0})
    body = r.get_json()
    assert r.status_code == 200 and body["exists"], body
    assert body["sequence_id"] and body["sequence_id"] != sid  # fresh, live id

    # Detection overlay restored for both cameras — incl. the per-level dot centres + grid
    # indices the GUI draws (these survive the sidecar round-trip, not just the `ok` flag).
    assert set(body["poses"]) == {"1", "2"}
    assert len(body["poses"]["1"]) == len(FRAMES)
    datum1 = body["datum_detection"]["1"]
    assert datum1["ok"] and body["datum_detection"]["2"]["ok"]
    assert datum1["level_a"] and datum1["level_a"]["centers"], "level_a centers missing"
    assert datum1["level_b"] and datum1["level_b"]["centers"], "level_b centers missing"
    assert datum1["level_a"]["grid_indices"], "level_a grid_indices missing"

    # Operator clicks restored: fiducials + datum face + per-pose peak/trough labels.
    assert set(body["clicks"]) == {"1", "2"}
    assert body["clicks"]["1"]["clicked_level"] == "peak"
    assert set(body["clicks"]["1"]["fiducials"]) == {"origin", "x_axis", "y_axis"}
    assert body["clicks"]["1"]["pose_levels"]

    # The restored sequence is live in the cache: per-pose detection works with the new id.
    pd = client.get("/calibration/stepped/sequence_pose_detection",
                    query_string={"sequence_id": body["sequence_id"], "camera": 1, "frame_idx": 1})
    assert pd.status_code == 200 and pd.get_json()["ok"]


def test_generate_after_reload_from_sidecar(client, scene):
    """The whole regression: detect + generate, drop the in-memory cache (reopen the tab),
    restore from the sidecar, then GENERATE AGAIN against the restored sequence. This fails
    if the fit's per-level grids (centers + grid_indices + H + vec1/vec2) don't survive the
    sidecar round-trip — the "datum pose has no detected grid level" bug. The earlier restore
    test only checks the overlay; this one re-solves."""
    sid, _ = _detect(client, [1, 2])
    first = client.post("/calibration/stepped/generate_model", json={
        "sequence_id": sid, "stereo": True, "stereo_config": "auto",
        "cameras": {"1": _spec(scene, 1), "2": _spec(scene, 2)}})
    done = _wait(client, f"/calibration/stepped/generate_model/status/{first.get_json()['job_id']}")
    assert done["status"] == "completed", done

    sv._sequence_cache.clear()  # reopen: in-memory cache gone, only the sidecar remains

    body = client.get("/calibration/stepped/restore_sequence",
                      query_string={"camera_pair": "1,2", "source_path_idx": 0}).get_json()
    assert body["exists"], body
    restored_sid = body["sequence_id"]

    # Re-solve from the sidecar-restored sequence — this is what was raising RuntimeError.
    again = client.post("/calibration/stepped/generate_model", json={
        "sequence_id": restored_sid, "stereo": True, "stereo_config": "auto",
        "cameras": {"1": _spec(scene, 1), "2": _spec(scene, 2)}})
    done2 = _wait(client, f"/calibration/stepped/generate_model/status/{again.get_json()['job_id']}")
    assert done2["status"] == "completed", done2
    assert done2["rms_cam1"] < 1.0 and done2["rms_cam2"] < 1.0
    assert done2["num_pairs_used"] == len(FRAMES)


def test_restore_sequence_absent_returns_exists_false(client):
    """No sidecar for this source -> the normal first-use state, not an error."""
    r = client.get("/calibration/stepped/restore_sequence",
                   query_string={"camera_pair": "1,2", "source_path_idx": 0})
    assert r.status_code == 200 and r.get_json() == {"exists": False}


def test_inputs_save_persists_clicks_without_generate(client, scene):
    """Clicks persist to the sidecar via inputs/save as they are picked (no generate, no
    config) — restore then brings them back. This is the per-pick persistence that replaces
    the old config write."""
    _detect(client, [1, 2])
    r = client.post("/calibration/stepped/inputs/save", json={
        "camera_pair": "1,2", "source_path_idx": 0, "stereo_config": "auto",
        "cameras": {"1": _spec(scene, 1), "2": _spec(scene, 2)},
    })
    assert r.status_code == 200 and r.get_json()["saved"], r.get_json()

    sv._sequence_cache.clear()  # simulate reload (in-memory cache gone, sidecar remains)

    body = client.get("/calibration/stepped/restore_sequence",
                      query_string={"camera_pair": "1,2", "source_path_idx": 0}).get_json()
    assert body["exists"] and set(body["clicks"]) == {"1", "2"}
    assert body["clicks"]["1"]["fiducials"]["origin"]
    assert body["clicks"]["1"]["clicked_level"] == "peak"
    assert body["clicks"]["1"]["pose_levels"]
    assert body["stereo_config"] == "auto"


def test_inputs_save_noop_without_detection(client, scene):
    """No detected sequence -> nothing to attach clicks to -> saved:false (not an error)."""
    r = client.post("/calibration/stepped/inputs/save", json={
        "camera_pair": "1,2", "source_path_idx": 0,
        "cameras": {"1": _spec(scene, 1)},
    })
    assert r.status_code == 200 and r.get_json() == {"saved": False}
