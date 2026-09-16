"""Detection-cache honesty for the GUI calibration routes (2026-09-01).

Drives the REAL Flask handlers with ``get_config``, image loading and the board
detector mocked, against a tmp workspace:

- ``/calibration/detect_views`` (the Detect Dots button) WRITES the fresh detections
  and their ``det_key`` into the ``inputs.mat`` sidecar Generate reads, so what the
  user sees drawn is what the next solve uses. Stereo detects the pair in one call
  and writes the pair sidecar once; stored clicks survive; a partial set (a frame
  that failed to load) is never persisted over a full one.
- ``/calibration/generate_model`` with ``force_redetect`` bypasses the cached
  detections only -- it must not erase the clicked world frame stored beside them.
- ``_joint_detect``'s in-memory cache follows the resolved source PATH, so an
  in-place path edit at the same index cannot serve another directory's detections.
"""

from __future__ import annotations

import numpy as np
import pytest
from flask import Flask

import pivtools_gui.calibration.app.views as V
from pivtools_core.image_handling.path_utils import infer_image_type
from pivtools_gui.calibration import inputs_store as INP
from pivtools_gui.calibration import record as rec
from pivtools_gui.calibration.detection.dotboard import DotboardParams
from test_calibration_sidecar_resolve import (  # noqa: E402  (sibling-test reuse, as elsewhere)
    SPACING_MM,
    _clicks_from_detection,
    _detection,
)

W, H = 1600, 1200
FMT = "cal_%03d.tif"
# One board pose per view; three distinct tilts keep the pinhole bundle well posed.
TILTS = [10.0, 14.0, 18.0]
N_VIEWS = len(TILTS)


class _FakeConfig:
    camera_count = 2

    def __init__(self, base):
        self._base = base
        self.calibration = {}

    def get_calibration_source(self, idx):
        return self._base


# Camera 2 sees every board pose under one extra rotation about the board's own origin. That
# is a fixed rigid transform between the two camera frames for every view, so the pair forms
# a consistent stereo rig for run_stereo.
CAM2_EXTRA_TILT = 6.0


def _frame_image(frame: int, camera: int = 1) -> np.ndarray:
    """A stand-in image carrying its frame and camera number, so the fake detector is per-view."""
    img = np.full((H, W), frame, np.uint8)
    img[0, 1] = camera
    return img


def _tilt(frame: int, camera: int) -> float:
    return TILTS[frame - 1] + (CAM2_EXTRA_TILT if camera == 2 else 0.0)


class _FakeDetector:
    """Deterministic per view: the tilt is read from the frame + camera the image carries."""

    def detect(self, img):
        return _detection(_tilt(int(img[0, 0]), int(img[0, 1])))[0]


@pytest.fixture
def env(tmp_path, monkeypatch):
    params = DotboardParams(dot_spacing_mm=SPACING_MM)
    detector = _FakeDetector()
    monkeypatch.setattr(V, "get_config", lambda: _FakeConfig(tmp_path))
    monkeypatch.setattr(
        V, "_resolve_board", lambda get, overrides=None: ({}, "dotboard", params, detector)
    )
    monkeypatch.setattr(
        V, "_load_one", lambda cam, frame, src, fmt=None, typ=None: _frame_image(frame, cam)
    )
    monkeypatch.setattr(
        V,
        "_detect_parallel",
        lambda board, p, imgs, spacing_mm=None, on_done=None: [detector.detect(i) for i in imgs],
    )
    monkeypatch.setattr(V, "get_calibration_frame_count", lambda cam, cfg, src: N_VIEWS)
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    return app.test_client(), tmp_path, params


def _post_views(client, **extra):
    body = {
        "source_path_idx": 0,
        "board": "dotboard",
        "image_format": FMT,
        "frame_total": N_VIEWS,
        **extra,
    }
    return client.post("/calibration/detect_views", json=body).get_json()


def _mono_key(params, n_views=N_VIEWS, cameras=(1,)):
    return INP.joint_det_key(
        "dotboard", n_views, FMT, infer_image_type(FMT), list(cameras), params
    )


# ---------------------------------------------------------------------------
# detect_views writes through
# ---------------------------------------------------------------------------


def test_detect_views_mono_writes_sidecar_and_keeps_clicks(env):
    client, base, params = env
    mdir = rec.mono_model_dir_for_source(base, 1, "dotboard")
    clicks = _clicks_from_detection(_detection(TILTS[0])[0])
    INP.save_inputs(mdir, path_type="mono", board_type="dotboard", coords=clicks)

    data = _post_views(client, camera=1)

    assert data["success"] is True
    assert data["persisted"] is True
    assert list(data["by_camera"]) == ["1"]
    assert data["by_camera"]["1"]["n_detected"] == N_VIEWS
    assert data["by_camera"]["1"]["width"] == W and data["by_camera"]["1"]["height"] == H
    side = INP.load_inputs(mdir)
    assert side.det_key == _mono_key(params)
    assert [d.success for d in side.detections[1]] == [True] * N_VIEWS
    assert side.image_size_by_cam[1] == (W, H)
    assert side.board_params["dot_spacing_mm"] == SPACING_MM
    assert np.allclose(side.coords["origin"], clicks["origin"])


def test_detect_views_stereo_writes_pair_sidecar_once(env):
    client, base, params = env

    data = _post_views(client, stereo=True, camera_pair=[1, 2])

    assert data["success"] is True and data["persisted"] is True
    assert sorted(data["by_camera"]) == ["1", "2"]
    side = INP.load_inputs(rec.stereo_model_dir_for_source(base, 1, 2))
    assert sorted(side.detections) == [1, 2]
    assert len(side.detections[2]) == N_VIEWS
    assert side.det_key == _mono_key(params, cameras=(1, 2))
    # Nothing leaks into the mono sidecars.
    assert INP.try_load_inputs(rec.mono_model_dir_for_source(base, 1, "dotboard")) is None


def test_detect_views_failed_view_is_stored_as_failed(env, monkeypatch):
    """A view where the board is NOT found is a legitimate outcome: it is stored in place as
    a failed detection (the solve drops it), and the set still counts as complete."""
    client, base, params = env

    class _MissTwo(_FakeDetector):
        def detect(self, img):
            d = super().detect(img)
            if int(img[0, 0]) == 2:
                d.success = False
            return d

    miss = _MissTwo()
    monkeypatch.setattr(
        V, "_resolve_board", lambda get, overrides=None: ({}, "dotboard", params, miss)
    )
    # detect_views detects through _detect_parallel (which builds a detector per
    # task -- cv2 detectors are not documented thread-safe), so a custom detector
    # has to be injected at that seam, not only at _resolve_board.
    monkeypatch.setattr(
        V,
        "_detect_parallel",
        lambda board, p, imgs, spacing_mm=None, on_done=None: [miss.detect(i) for i in imgs],
    )
    data = _post_views(client, camera=1)

    assert data["persisted"] is True
    assert data["by_camera"]["1"]["n_detected"] == N_VIEWS - 1
    assert sorted(data["by_camera"]["1"]["frames"]) == ["1", "3"]
    side = INP.load_inputs(rec.mono_model_dir_for_source(base, 1, "dotboard"))
    assert [d.success for d in side.detections[1]] == [True, False, True]


def test_detect_views_unloadable_view_is_not_persisted(env, monkeypatch):
    """A frame that fails to LOAD leaves the set incomplete: the overlay still shows what
    detected, but the sidecar keeps its previous (complete) contents."""
    client, base, params = env
    mdir = rec.mono_model_dir_for_source(base, 1, "dotboard")
    INP.save_inputs(
        mdir,
        path_type="mono",
        board_type="dotboard",
        detections={1: [_detection(t)[0] for t in TILTS]},
        det_key="previous-complete-set",
    )

    def _load(cam, frame, src, fmt=None, typ=None):
        if frame == 2:
            raise FileNotFoundError("cal_002.tif")
        return _frame_image(frame)

    monkeypatch.setattr(V, "_load_one", _load)
    data = _post_views(client, camera=1)

    assert data["success"] is True
    assert data["persisted"] is False
    assert "2" in data["persist_skipped"]
    assert data["by_camera"]["1"]["n_detected"] == N_VIEWS - 1
    side = INP.load_inputs(mdir)
    assert side.det_key == "previous-complete-set"
    assert len(side.detections[1]) == N_VIEWS


# ---------------------------------------------------------------------------
# generate_model: force_redetect must not erase the stored clicks
# ---------------------------------------------------------------------------


def test_generate_force_redetect_keeps_stored_clicks(env):
    client, base, params = env
    mdir = rec.mono_model_dir_for_source(base, 1, "dotboard")
    clicks = _clicks_from_detection(_detection(TILTS[0])[0])
    INP.save_inputs(
        mdir,
        path_type="mono",
        board_type="dotboard",
        detections={1: [_detection(t)[0] for t in TILTS]},
        image_size_by_cam={1: (W, H)},
        det_key=_mono_key(params),
        coords=clicks,
    )

    data = client.post(
        "/calibration/generate_model",
        json={
            "source_path_idx": 0,
            "board": "dotboard",
            "camera": 1,
            "image_format": FMT,
            "frame_total": N_VIEWS,
            "datum_frame": 1,
            "dt": 1.0,
            "no_figures": True,
            "force_redetect": True,
            # no "clicks": the user pressed Re-detect without re-clicking the world frame
        },
    ).get_json()

    assert data["success"] is True, data
    assert data["detections_cached"] is False
    side = INP.load_inputs(mdir)
    assert side.coords is not None, "force_redetect erased the stored world-frame clicks"
    assert np.allclose(side.coords["origin"], clicks["origin"])
    assert side.coords["origin_mm"] == clicks["origin_mm"]


def test_generate_stereo_force_redetect_keeps_stored_clicks(env):
    client, base, params = env
    sdir = rec.stereo_model_dir_for_source(base, 1, 2)
    clicks = _clicks_from_detection(_detection(_tilt(1, 1))[0])
    INP.save_inputs(
        sdir,
        path_type="stereo",
        board_type="dotboard",
        detections={c: [_detection(_tilt(f, c))[0] for f in range(1, N_VIEWS + 1)] for c in (1, 2)},
        image_size_by_cam={1: (W, H), 2: (W, H)},
        det_key=_mono_key(params, cameras=(1, 2)),
        coords=clicks,
    )

    data = client.post(
        "/calibration/generate_model",
        json={
            "source_path_idx": 0,
            "board": "dotboard",
            "stereo": True,
            "camera_pair": [1, 2],
            "image_format": FMT,
            "frame_total": N_VIEWS,
            "datum_frame": 1,
            "dt": 1.0,
            "no_figures": True,
            "force_redetect": True,
        },
    ).get_json()

    assert data["success"] is True, data
    assert data["detections_cached"] is False
    side = INP.load_inputs(sdir)
    assert side.coords is not None, "force_redetect erased the stored world-frame clicks"
    assert np.allclose(side.coords["origin"], clicks["origin"])


# ---------------------------------------------------------------------------
# joint in-memory detection cache follows the source path
# ---------------------------------------------------------------------------


def test_joint_detect_memory_cache_follows_source_path(tmp_path, monkeypatch):
    dirs = [tmp_path / "rig_A", tmp_path / "rig_B"]
    for d in dirs:
        d.mkdir()
    current = {"path": dirs[0]}
    monkeypatch.setattr(V, "_source_path", lambda idx: current["path"])
    monkeypatch.setattr(
        V, "_load_views", lambda cam, n, src, fmt, typ: [_frame_image(k + 1) for k in range(n)]
    )
    runs = {"n": 0}

    def _detect(board, params, imgs, spacing_mm=None, on_done=None):
        runs["n"] += 1
        # Each source has its own board pose, so a served-from-cache result is detectable.
        return [_detection(5.0 * runs["n"])[0] for _ in imgs]

    monkeypatch.setattr(V, "_detect_parallel", _detect)
    V._joint_detect_cache.clear()
    params = DotboardParams(dot_spacing_mm=SPACING_MM)
    args = ("dotboard", params, [1], 2, 0, FMT, "standard", SPACING_MM)

    first, _ = V._joint_detect(*args)
    current["path"] = dirs[1]  # same index, edited in place to another directory
    second, _ = V._joint_detect(*args)

    assert runs["n"] == 2, "the second source was served the first source's detections"
    assert not np.allclose(first[1][0].image_points, second[1][0].image_points)
