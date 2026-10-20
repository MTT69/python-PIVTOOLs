"""``board_meta["n_views"]`` is the configured image count, not the number of views solved.

The GUI compares the loaded model's ``n_views`` with the form's image count and flags
"Settings changed since this model was generated (number of images)" when they differ;
``settings_seed`` copies it into ``image.n_views``. Stamping the count of *successfully
detected* views (pinhole) or 1 (polynomial) made a fresh generate on a set with one
undetectable view read as a settings change. These tests drive the real ``generate_model``
and ``load_model`` routes (the ``env`` fixture of the ``detect_views`` suite: config, image
loading and the detector mocked) with one view whose board is not found.
"""

from __future__ import annotations

# ruff: noqa: F811  (the imported ``env`` pytest fixture is re-bound by every test's parameter)
import pytest
import test_calibration_detect_views_routes as DVR
from test_calibration_detect_views_routes import env  # noqa: F401  (pytest fixture)
from test_calibration_detect_views_routes import FMT, _FakeDetector, _tilt
from test_calibration_sidecar_resolve import _clicks_from_detection, _detection

import pivtools_gui.calibration.app.views as V

# Four poses so the pinhole bundle still has the >= 3 successful views it needs after one
# view fails detection. The failed frame is the last one, never the datum (frame 1).
TILTS_4 = [10.0, 14.0, 18.0, 22.0]
FRAME_TOTAL = len(TILTS_4)
FAILED_FRAME = 4


class _MissLast(_FakeDetector):
    """The fake detector, except camera ``miss_camera`` does not find the board on FAILED_FRAME."""

    def __init__(self, miss_camera: int) -> None:
        self.miss_camera = miss_camera

    def detect(self, img):
        d = super().detect(img)
        if int(img[0, 0]) == FAILED_FRAME and int(img[0, 1]) == self.miss_camera:
            d.success = False
        return d


@pytest.fixture
def one_view_fails(env, monkeypatch):
    """``env`` with four views, the last of which camera 1 cannot detect."""
    client, base, params = env
    monkeypatch.setattr(DVR, "TILTS", TILTS_4)
    miss = _MissLast(miss_camera=1)
    monkeypatch.setattr(
        V, "_resolve_board", lambda get, overrides=None: ({}, "dotboard", params, miss)
    )
    # Detection runs through _detect_parallel (one detector per task), so the failing
    # detector is injected at that seam too.
    monkeypatch.setattr(
        V,
        "_detect_parallel",
        lambda board, p, imgs, spacing_mm=None, on_done=None: [
            miss.detect(i) for i in imgs
        ],
    )
    return client, base, params


def _gen(client, **extra):
    body = {
        "source_path_idx": 0,
        "board": "dotboard",
        "image_format": FMT,
        "frame_total": FRAME_TOTAL,
        "datum_frame": 1,
        "dt": 1.0,
        "no_figures": True,
        **extra,
    }
    return client.post("/calibration/generate_model", json=body).get_json()


def _load(client, **query):
    q = {"source_path_idx": 0, "board": "dotboard", **query}
    return client.get("/calibration/model", query_string=q).get_json()


def _clicks(camera: int) -> dict:
    return _clicks_from_detection(_detection(_tilt(1, camera))[0])


def test_mono_pinhole_n_views_is_frame_total(one_view_fails):
    client, _base, _params = one_view_fails

    body = _gen(client, camera=1, clicks=_clicks(1), model_type="pinhole")

    assert body["success"] is True, body
    # The fit really did drop the failed view ...
    assert body["num_images_used"] == FRAME_TOTAL - 1
    # ... but the model reports the configured image count.
    loaded = _load(client, camera=1)
    assert loaded["exists"] is True
    assert loaded["n_views"] == FRAME_TOTAL


def test_mono_polynomial_n_views_is_frame_total(one_view_fails):
    """The planar polynomial fits the datum view only; n_views is still the image count."""
    client, _base, _params = one_view_fails

    body = _gen(client, camera=1, clicks=_clicks(1), model_type="polynomial")

    assert body["success"] is True, body
    loaded = _load(client, camera=1, model_type="polynomial")
    assert loaded["exists"] is True
    assert loaded["n_views"] == FRAME_TOTAL


def test_stereo_n_views_is_frame_total(one_view_fails):
    client, _base, _params = one_view_fails

    body = _gen(
        client,
        stereo=True,
        camera_pair=[1, 2],
        clicks=_clicks(1),
        clicks2=_clicks(2),
    )

    assert body["success"] is True, body
    assert body["num_pairs_used"] == FRAME_TOTAL - 1
    loaded = _load(client, stereo=1, camera_pair="1,2")
    assert loaded["exists"] is True
    assert loaded["n_views"] == FRAME_TOTAL
