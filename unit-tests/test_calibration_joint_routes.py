"""S2·C0 — the /calibration/joint/* Flask routes, end-to-end.

Drives the REAL route handlers, the REAL generate job thread, and the REAL
``run_joint_from_spec`` driver (shared with the CLI). Only two environment seams are
mocked: ``read_calibration_image`` (returns the checked-in ChArUco fixture frames instead
of resolving disk paths) and ``get_config`` / ``_cfg`` (a small config double pointing the
saved model at ``tmp_path``). ChArUco needs zero clicks, so this proves the whole HTTP
pipeline — resolve_grid -> generate(job) -> model — without any dotboard click machinery
(that is exercised in C2).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from flask import Flask

import pivtools_gui.calibration.app.views as V

_SYN = Path(__file__).parent / "synthetic_calibration"
_STEREO_CHARUCO = _SYN / "stereo_charuco"
_N_VIEWS = 10

_CHARUCO = {"squares_h": 10, "squares_v": 7, "square_size": 0.030, "marker_ratio": 0.5,
            "aruco_dict": "DICT_4X4_1000", "min_corners": 6}


class _FakeConfig:
    """Config double for the joint routes: a calibration block + the resolvers they call."""

    def __init__(self, source, global_grid, active="charuco"):
        self.calibration = {
            "active": active,
            "image_format": "calib%05d.png",
            "image_type": "standard",
            "n_views": _N_VIEWS,
            "camera_numbers": [1, 2],
            "charuco": dict(_CHARUCO),
            "global_grid": global_grid,
        }
        self._source = Path(source)

    @property
    def global_grid_config(self) -> dict:
        return self.calibration.get("global_grid", {})

    def get_calibration_source(self, idx):
        return self._source


def _load_fixture():
    """{(camera, frame_1based): grayscale image} from the checked-in ChArUco set."""
    imgs = {}
    for cam in (1, 2):
        for k in range(1, _N_VIEWS + 1):
            p = _STEREO_CHARUCO / f"cam{cam}" / f"calib{k:05d}.png"
            imgs[(cam, k)] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return imgs


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A Flask test client with the calibration blueprint and the two env seams mocked."""
    if not _STEREO_CHARUCO.is_dir():
        pytest.skip("synthetic stereo_charuco set absent")
    gg = {"enabled": True, "datum_camera": 1, "datum_view": 0, "cameras": [1, 2],
          "board_release": "full3d"}
    cfg = _FakeConfig(tmp_path, gg)
    fixture = _load_fixture()

    def fake_read(frame, camera, config, source_idx, image_format=None, image_type=None):
        return fixture[(int(camera), int(frame))]

    monkeypatch.setattr(V, "get_config", lambda: cfg)
    monkeypatch.setattr(V, "_cfg", lambda: cfg.calibration)
    monkeypatch.setattr(V, "read_calibration_image", fake_read)
    V._joint_detect_cache.clear()  # no cross-test bleed (module-global cache)

    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    return app.test_client()


def _wait_job(client, job_id, timeout=120.0):
    """Poll the generate-status route until the job ends; return its final payload."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/calibration/joint/generate/status/{job_id}")
        data = r.get_json()
        if data.get("status") in ("completed", "failed"):
            return data
        time.sleep(0.2)
    raise AssertionError("joint generate job did not finish in time")


# ---------------------------------------------------------------------------
# resolve_grid
# ---------------------------------------------------------------------------


def test_resolve_grid_charuco_labels_every_view(client):
    r = client.post("/calibration/joint/resolve_grid", json={"board": "charuco"})
    data = r.get_json()
    assert data["success"] is True
    assert data["cameras"] == [1, 2]
    assert data["n_views_total"] == 2 * _N_VIEWS
    # ChArUco resolves every view from corner ids (no clicks, no errors).
    assert data["errors"] == []
    assert data["n_resolved"] == data["n_views_total"]
    for vw in data["views"]:
        assert vw["resolved"] is True
        assert vw["global_index"] is not None
        assert len(vw["points"]) == vw["n"] == len(vw["global_index"])
        # global indices are integer (gx, gy) pairs
        assert all(len(g) == 2 for g in vw["global_index"])


def test_resolve_grid_rejects_unknown_board(client):
    r = client.post("/calibration/joint/resolve_grid", json={"board": "stepped"})
    data = r.get_json()
    assert data["success"] is False
    assert "board must be dotboard|charuco" in data["error"]


# ---------------------------------------------------------------------------
# generate (job) + model
# ---------------------------------------------------------------------------


def test_generate_charuco_pinhole_then_model(client, tmp_path):
    r = client.post("/calibration/joint/generate", json={"board": "charuco"})
    start = r.get_json()
    assert start["success"] is True
    job = _wait_job(client, start["job_id"])
    assert job["status"] == "completed", job.get("error")
    assert job["model_type"] == "pinhole"
    assert sorted(int(c) for c in job["cameras"]) == [1, 2]
    assert job["rms_units"] == "px"
    assert np.isfinite(job["rms_px"]) and job["rms_px"] < 2.0
    for cam in ("1", "2"):
        assert job["per_camera_rms"][cam] < 2.0
    # one shared board => agreement is exactly 0 by construction
    assert job["cross_camera_board_agreement_mm"] == pytest.approx(0.0, abs=1e-9)
    assert job["paths"] and Path(job["paths"][0]).exists()

    # the model route now reports the saved record
    m = client.get("/calibration/joint/model", query_string={"board": "charuco"}).get_json()
    assert m["exists"] is True
    assert m["model_type"] == "pinhole"
    assert sorted(m["cameras"]) == [1, 2]
    assert m["rms_px"] < 2.0
    assert m["spacing_mm"] == pytest.approx(30.0)
    assert m["n_board_dots"] > 40


def test_model_absent_when_nothing_saved(client):
    m = client.get("/calibration/joint/model", query_string={"board": "charuco"}).get_json()
    assert m["exists"] is False


def test_generate_charuco_polynomial_writes_per_camera(client):
    r = client.post("/calibration/joint/generate",
                    json={"board": "charuco", "model_type": "polynomial"})
    job = _wait_job(client, r.get_json()["job_id"])
    assert job["status"] == "completed", job.get("error")
    assert job["model_type"] == "polynomial"
    assert job["rms_units"] == "mm"
    assert len(job["paths"]) == 2  # one per camera
    m = client.get("/calibration/joint/model",
                   query_string={"board": "charuco", "model_type": "polynomial"}).get_json()
    assert m["exists"] is True and m["model_type"] == "polynomial"
    assert sorted(m["cameras"]) == [1, 2]
    for cam in ("1", "2"):
        assert m["per_camera"][cam]["rms_x_mm"] < 1.0
        assert m["per_camera"][cam]["rms_y_mm"] < 1.0


def test_generate_status_unknown_job_404(client):
    r = client.get("/calibration/joint/generate/status/no-such-job")
    assert r.status_code == 404
    assert r.get_json()["error"]


def test_generate_job_failure_surfaces(client):
    """A camera with no frames fails INSIDE the job thread -> status 'failed' with an error,
    never a hung 'running' or a swallowed exception."""
    r = client.post("/calibration/joint/generate", json={"board": "charuco", "cameras": [1, 2, 3]})
    assert r.get_json()["success"] is True  # the job starts; the failure is reported via status
    job = _wait_job(client, r.get_json()["job_id"])
    assert job["status"] == "failed"
    assert job.get("error")


def test_resolve_then_generate_different_camera_sets(client):
    """Regression: the detection cache key includes the camera set, so resolving cam1 alone does
    NOT stale-serve a cam1-only detection to a cam1+2 generate (which would fail confusingly)."""
    r1 = client.post("/calibration/joint/resolve_grid",
                     json={"board": "charuco", "cameras": [1]})
    assert r1.get_json()["cameras"] == [1]
    r2 = client.post("/calibration/joint/generate", json={"board": "charuco", "cameras": [1, 2]})
    job = _wait_job(client, r2.get_json()["job_id"])
    assert job["status"] == "completed", job.get("error")
    assert sorted(int(c) for c in job["cameras"]) == [1, 2]
