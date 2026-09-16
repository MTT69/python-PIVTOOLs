"""``/calibration/restore_detections`` — stored detections without detecting.

The route backs the overlay a calibration tab paints when it opens. Before it
existed, ``DotboardCalibration.tsx`` restored the world frame from ``inputs.mat``
and then re-detected the dots on the very next line, even though those dots were
sitting in the same sidecar. The public manual already promised otherwise
(``manual/global-coordinates``: "previously detected dots load into the viewer
automatically from the cache").

Two properties carry the whole design and each has a test that fails loudly if it
breaks:

- what the tab redraws on re-open is bit-identical to what Detect Dots drew, and
  it is obtained without opening a single image;
- when the stored ``det_key`` does not match, NO points come back. Drawing stale
  dots would misrepresent what Generate is about to solve, because
  ``generate_model`` gates its own cache on that same key.

Fixtures are reused from the ``detect_views`` suite so the writer and the reader
are exercised against one another rather than against a hand-built sidecar.
"""

from __future__ import annotations

import pivtools_gui.calibration.app.views as V
from pivtools_gui.calibration import inputs_store as INP
from pivtools_gui.calibration import record as rec
from test_calibration_detect_views_routes import (  # noqa: E402  (sibling-test reuse, as elsewhere)
    FMT,
    N_VIEWS,
    TILTS,
    _FakeDetector,
    _post_views,
    env,
)
from test_calibration_sidecar_resolve import (  # noqa: E402
    _clicks_from_detection,
    _detection,
)


def _restore(client, **extra):
    """Deliberately the SAME body ``_post_views`` sends: the det_key must be
    reproduced by construction, not by a parallel encoding that can drift."""
    body = {
        "source_path_idx": 0,
        "board": "dotboard",
        "image_format": FMT,
        "frame_total": N_VIEWS,
        **extra,
    }
    return client.post("/calibration/restore_detections", json=body).get_json()


# ---------------------------------------------------------------------------
# Nothing stored
# ---------------------------------------------------------------------------


def test_restore_with_no_sidecar_reports_absent_not_an_error(env):
    """First use of a source. Must be a 200 with exists false: this fires on every
    tab open, so a 404 would be noise rather than information."""
    client, _base, _params = env
    response = client.post(
        "/calibration/restore_detections",
        json={
            "source_path_idx": 0,
            "board": "dotboard",
            "image_format": FMT,
            "frame_total": N_VIEWS,
            "camera": 1,
        },
    )
    assert response.status_code == 200
    assert response.get_json() == {"exists": False}


def test_restore_missing_camera_reports_absent(env):
    """A half-restored stereo pair is worse than none: the pair is detected and
    persisted as ONE set under one det_key, and the solve expects both."""
    client, base, _params = env
    INP.save_inputs(
        rec.stereo_model_dir_for_source(base, 1, 2),
        path_type="stereo",
        board_type="dotboard",
        detections={1: [_detection(t)[0] for t in TILTS]},
        det_key="whatever",
    )
    data = _restore(client, stereo=True, camera_pair=[1, 2])
    assert data == {"exists": False}


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------


def test_restore_returns_exactly_what_detect_views_drew(env):
    """The headline regression: what the GUI redraws on re-open is bit-identical
    to what it drew when the user pressed Detect Dots."""
    client, _base, _params = env
    drawn = _post_views(client, camera=1)
    assert drawn["persisted"] is True

    restored = _restore(client, camera=1)

    assert restored["exists"] is True
    assert restored["fresh"] is True
    assert restored["by_camera"] == drawn["by_camera"]


def test_restore_opens_no_images_and_builds_no_detector(env, monkeypatch):
    """The core property. If this passes only because an image was quietly loaded,
    the route is not a restore at all -- so make loading and detecting explode."""
    client, _base, _params = env
    drawn = _post_views(client, camera=1)

    def _explode(*args, **kwargs):
        raise AssertionError("restore_detections must not read images or detect")

    monkeypatch.setattr(V, "_load_one", _explode)
    monkeypatch.setattr(V, "_detect_parallel", _explode)

    restored = _restore(client, camera=1)

    assert restored["fresh"] is True
    assert restored["by_camera"] == drawn["by_camera"]


def test_restore_stereo_returns_both_cameras_in_one_call(env):
    client, _base, _params = env
    drawn = _post_views(client, stereo=True, camera_pair=[1, 2])

    restored = _restore(client, stereo=True, camera_pair=[1, 2])

    assert restored["fresh"] is True
    assert sorted(restored["by_camera"]) == ["1", "2"]
    assert restored["by_camera"] == drawn["by_camera"]


def test_restore_omits_failed_views_so_that_frame_blanks(env, monkeypatch):
    """A view where the board was not found stays blank rather than restoring an
    empty point list. Both render as "no detection", but only omission is honest
    about there being nothing there. Also pins the index -> frame mapping: the
    stored list is dense, so position i is frame i+1."""
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
    # detect_views detects through the pool, so the fake detector goes in there.
    monkeypatch.setattr(
        V,
        "_detect_parallel",
        lambda board, p, imgs, spacing_mm=None, on_done=None: [miss.detect(i) for i in imgs],
    )
    _post_views(client, camera=1)
    side = INP.load_inputs(rec.mono_model_dir_for_source(base, 1, "dotboard"))
    assert [d.success for d in side.detections[1]] == [True, False, True]

    restored = _restore(client, camera=1)

    assert sorted(restored["by_camera"]["1"]["frames"]) == ["1", "3"]
    assert restored["by_camera"]["1"]["n_detected"] == N_VIEWS - 1
    assert restored["n_views"] == N_VIEWS


def test_restore_carries_the_stored_image_size(env):
    """The overlay needs the frame size to scale, and a restore has no image to
    measure -- it must come from the sidecar."""
    client, _base, _params = env
    drawn = _post_views(client, camera=1)
    restored = _restore(client, camera=1)
    assert restored["by_camera"]["1"]["width"] == drawn["by_camera"]["1"]["width"]
    assert restored["by_camera"]["1"]["height"] == drawn["by_camera"]["1"]["height"]
    assert restored["by_camera"]["1"]["width"] > 0


# ---------------------------------------------------------------------------
# Staleness — no points, ever
# ---------------------------------------------------------------------------


def test_restore_under_a_different_view_count_returns_no_points(env):
    """The no-silent-stale-data assertion. generate_model would refuse to reuse
    these detections, so the overlay must not draw them either."""
    client, _base, _params = env
    _post_views(client, camera=1)

    restored = _restore(client, camera=1, frame_total=N_VIEWS + 1)

    assert restored["exists"] is True
    assert restored["fresh"] is False
    assert restored["by_camera"] is None
    assert restored["stale_reason"]
    # Enough detail for the GUI to say WHAT changed, not merely "stale".
    assert restored["stored"]["n_views"] == N_VIEWS


def test_restore_under_different_board_geometry_returns_no_points(env, monkeypatch):
    client, _base, _params = env
    _post_views(client, camera=1)

    from pivtools_gui.calibration.detection.dotboard import DotboardParams

    other = DotboardParams(dot_spacing_mm=999.0)
    monkeypatch.setattr(
        V,
        "_resolve_board",
        lambda get, overrides=None: ({}, "dotboard", other, _FakeDetector()),
    )
    restored = _restore(client, camera=1)

    assert restored["fresh"] is False
    assert restored["by_camera"] is None


def test_restore_goes_stale_on_a_detector_version_bump(env, monkeypatch):
    """DETECTOR_VERSION is folded into the key precisely so an algorithm change
    cannot be papered over by a cached point set from the old detector."""
    client, _base, _params = env
    _post_views(client, camera=1)
    assert _restore(client, camera=1)["fresh"] is True

    monkeypatch.setattr(INP, "DETECTOR_VERSION", 999)
    restored = _restore(client, camera=1)

    assert restored["exists"] is True
    assert restored["fresh"] is False
    assert restored["by_camera"] is None


def test_restore_without_board_geometry_reports_absent(env, monkeypatch):
    """No dot spacing entered yet is the routine first-visit state. Without
    geometry there is no det_key to compare, so there is nothing to restore --
    and that is not an error."""
    client, _base, _params = env

    def _no_geometry(get, overrides=None):
        raise ValueError("dot_spacing_mm is required")

    monkeypatch.setattr(V, "_resolve_board", _no_geometry)
    response = client.post(
        "/calibration/restore_detections",
        json={"source_path_idx": 0, "board": "dotboard", "camera": 1},
    )
    assert response.status_code == 200
    assert response.get_json() == {"exists": False}


# ---------------------------------------------------------------------------
# The probe and the solve must read one cache under one key
# ---------------------------------------------------------------------------


def test_generate_after_restore_reuses_the_same_cached_detections(env):
    """The behavioural mirror of the stepped suite's
    ``test_generate_after_reload_from_sidecar``. A restored overlay is only
    truthful if Generate then solves from those same points: this is the test that
    catches a future drift between this route's det_key construction and
    ``generate_model``'s.
    """
    client, _base, _params = env
    _post_views(client, camera=1)
    restored = _restore(client, camera=1)
    assert restored["fresh"] is True

    generated = client.post(
        "/calibration/generate_model",
        json={
            "source_path_idx": 0,
            "board": "dotboard",
            "camera": 1,
            "image_format": FMT,
            "frame_total": N_VIEWS,
            "model_type": "pinhole",
            "dt": 1.0,
            "clicks": _clicks_from_detection(_detection(TILTS[0])[0]),
        },
    ).get_json()

    assert generated.get("detections_cached") is True, generated
