"""Generate / detect / restore routes: no fabricated values, no silent fallbacks (P4 1b).

Drives the REAL Flask handlers against a tmp workspace with ``get_config``, image loading
and the board detector mocked (the ``env`` fixture of the ``detect_views`` suite):

- Design A: a stereo dotboard needs world-frame picks on BOTH cameras (dotboard grid
  indices are re-zeroed per view); the sidecar stores them per camera
  (``{"cameras": {"<cam>": {...}}, "origin_mm": [X, Y]}``), ``load_model`` returns
  camera 2's frame as ``world_frame2``, and a stereo sidecar in the old flat shape is
  reported stale instead of being read as camera 1's picks.
- Stereo responses take ``method`` / ``num_pairs_used`` from the record or raise; every
  mono/stereo generate and load states ``rms_units`` (``px``, or ``mm`` for a planar
  polynomial).
- ``snap_fiducial`` requires the click, resolves the camera like ``detect_datum`` and
  refuses a datum detection made for another source/board/frame.
- ``model_type`` absent means the settings sidecar's ``methods.<board>.model_type``;
  ``null`` is an error. ``camera_pair`` absent from request and rig is an error.
- The generate write-back merges coords under the sidecar lock: a pick committed while
  the model was solving is never overwritten with the stale read.
- ``_joint_detect``: a read-only sidecar write is reported, a ValueError raises.
- ``DETECTOR_VERSION`` 4: detections cached by an older detector are stale.
"""

from __future__ import annotations

# ruff: noqa: F811  (the imported ``env`` pytest fixture is re-bound by every test's parameter)
import time

import numpy as np
import pytest
from flask import Flask
from test_calibration_detect_views_routes import env  # noqa: F401  (pytest fixture)
from test_calibration_detect_views_routes import (  # noqa: E402  (sibling-test reuse)
    FMT,
    N_VIEWS,
    TILTS,
    H,
    W,
    _frame_image,
    _mono_key,
    _post_views,
    _tilt,
)
from test_calibration_sidecar_resolve import (  # noqa: E402
    SPACING_MM,
    _clicks_from_detection,
    _detection,
)

import pivtools_gui.calibration.app.views as V
from pivtools_core import calibration_settings as cs
from pivtools_gui.calibration import inputs_store as INP
from pivtools_gui.calibration import record as rec
from pivtools_gui.calibration.detection import grid_detection as GD
from pivtools_gui.calibration.detection.dotboard import DotboardParams
from pivtools_gui.calibration.pipeline import Calibrator
from pivtools_gui.calibration.stereo_model import StereoCalibrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen(client, **extra):
    """POST generate_model with the suite's defaults (figures off, dt given)."""
    body = {
        "source_path_idx": 0,
        "board": "dotboard",
        "image_format": FMT,
        "frame_total": N_VIEWS,
        "datum_frame": 1,
        "dt": 1.0,
        "no_figures": True,
        **extra,
    }
    return client.post("/calibration/generate_model", json=body)


def _pair_clicks():
    """Picks on the same physical dots (the (0,0) corner, far +col, far +row) in each camera."""
    c1 = _clicks_from_detection(_detection(_tilt(1, 1))[0])
    c2 = _clicks_from_detection(_detection(_tilt(1, 2))[0])
    return c1, c2


def _stereo_dir(base):
    return rec.stereo_model_dir_for_source(base, 1, 2, "dotboard")


def _load_model(client, **query):
    q = {"source_path_idx": 0, "board": "dotboard", **query}
    return client.get("/calibration/model", query_string=q).get_json()


# ---------------------------------------------------------------------------
# Design A: stereo dotboard world-frame picks on both cameras
# ---------------------------------------------------------------------------


def test_stereo_dotboard_generate_refuses_without_cam2_picks(env):
    client, _base, _params = env
    c1, _c2 = _pair_clicks()

    r = _gen(client, stereo=True, camera_pair=[1, 2], clicks=c1)

    body = r.get_json()
    assert r.status_code == 400, body
    assert body["success"] is False
    assert "camera 2" in body["error"] and "re-zeroed" in body["error"]


def test_stereo_dotboard_generate_persists_per_camera_coords(env):
    client, base, _params = env
    c1, c2 = _pair_clicks()

    body = _gen(
        client, stereo=True, camera_pair=[1, 2], clicks=c1, clicks2=c2
    ).get_json()

    assert body["success"] is True, body
    coords = INP.load_inputs(_stereo_dir(base)).coords
    assert sorted(coords) == ["cameras", "origin_mm"]
    assert sorted(coords["cameras"]) == ["1", "2"]
    assert coords["origin_mm"] == [2.5, 5.0]
    for cam, clicks in (("1", c1), ("2", c2)):
        for role in ("origin", "x_axis", "y_axis"):
            assert coords["cameras"][cam][role] == clicks[role]
        assert sorted(coords["cameras"][cam]) == ["origin", "x_axis", "y_axis"]


def test_stereo_dotboard_regenerates_from_sidecar_picks(env):
    """No picks in the request: both cameras' frames come back from the sidecar."""
    client, _base, _params = env
    c1, c2 = _pair_clicks()
    first = _gen(client, stereo=True, camera_pair=[1, 2], clicks=c1, clicks2=c2)
    assert first.get_json()["success"] is True

    again = _gen(client, stereo=True, camera_pair=[1, 2]).get_json()

    assert again["success"] is True, again
    assert again["detections_cached"] is True


def test_stereo_sidecar_with_flat_coords_is_stale(env):
    """A stereo sidecar holding the old flat (cam1-only) shape is not read as cam1 picks."""
    client, base, params = env
    c1, _c2 = _pair_clicks()
    INP.save_inputs(
        _stereo_dir(base),
        path_type="stereo",
        board_type="dotboard",
        detections={
            c: [_detection(_tilt(f, c))[0] for f in range(1, N_VIEWS + 1)]
            for c in (1, 2)
        },
        image_size_by_cam={1: (W, H), 2: (W, H)},
        det_key=_mono_key(params, cameras=(1, 2)),
        coords=c1,
    )

    body = _gen(client, stereo=True, camera_pair=[1, 2]).get_json()

    assert body["success"] is False
    assert "stale" in body["error"] and "'cameras'" in body["error"]


def test_load_model_stereo_returns_world_frame2(env):
    client, base, _params = env
    c1, c2 = _pair_clicks()
    assert _gen(
        client, stereo=True, camera_pair=[1, 2], clicks=c1, clicks2=c2
    ).get_json()["success"]

    live = _load_model(client, stereo=1, camera_pair="1,2")
    assert live["exists"] is True
    assert live["world_frame2"]["origin"] == c2["origin"]
    assert live["world_frame2"]["origin_mm"] == [2.5, 5.0]
    assert live["rms_units"] == "px"

    # Model deleted: both frames come back from the sidecar picks.
    for p in _stereo_dir(base).glob("*.mat"):
        if p.name != INP.INPUTS_FILENAME:
            p.unlink()
    gone = _load_model(client, stereo=1, camera_pair="1,2")
    assert gone["exists"] is False
    assert gone["world_frame"]["origin"] == c1["origin"]
    assert gone["world_frame2"]["origin"] == c2["origin"]


def test_inputs_world_frame_per_camera(env):
    _client, base, _params = env
    c1, c2 = _pair_clicks()
    sdir = _stereo_dir(base)
    INP.save_inputs(
        sdir,
        path_type="stereo",
        board_type="dotboard",
        coords=V._stereo_coords(1, 2, c1, c2),
    )
    assert V._inputs_world_frame(sdir, camera=2)["origin"] == c2["origin"]
    assert V._inputs_world_frame(sdir, camera=1)["x_axis"] == c1["x_axis"]
    assert V._inputs_world_frame(sdir, camera=1)["origin_mm"] == [2.5, 5.0]

    mdir = rec.mono_model_dir_for_source(base, 1, "dotboard")
    INP.save_inputs(mdir, path_type="mono", board_type="dotboard", coords=c1)
    assert V._inputs_world_frame(mdir)["origin"] == c1["origin"]

    flat = rec.stereo_model_dir_for_source(base, 1, 3, "dotboard")
    INP.save_inputs(flat, path_type="stereo", board_type="dotboard", coords=c1)
    with pytest.raises(ValueError, match="'cameras'"):
        V._inputs_world_frame(flat, camera=1)


def test_inputs_store_round_trips_stereo_coords_shape(tmp_path):
    c1, c2 = _pair_clicks()
    coords = V._stereo_coords(1, 2, c1, c2)
    INP.save_inputs(tmp_path, path_type="stereo", board_type="dotboard", coords=coords)
    assert INP.load_inputs(tmp_path).coords == coords


def test_stereo_coords_requires_origin_mm():
    c1, c2 = _pair_clicks()
    del c1["origin_mm"]
    with pytest.raises(ValueError, match="origin_mm"):
        V._stereo_coords(1, 2, c1, c2)


# ---------------------------------------------------------------------------
# Item 5: nothing fabricated in the responses; rms units stated
# ---------------------------------------------------------------------------


def test_stereo_generate_response_states_method_and_units(env):
    client, _base, _params = env
    c1, c2 = _pair_clicks()

    body = _gen(
        client, stereo=True, camera_pair=[1, 2], clicks=c1, clicks2=c2
    ).get_json()

    assert body["success"] is True, body
    assert body["method"] == "stereoCalibrate"
    assert body["num_pairs_used"] == N_VIEWS
    assert body["rms_units"] == "px"


def test_stereo_generate_refuses_record_without_n_stereo_views(env, monkeypatch):
    client, _base, _params = env
    c1, c2 = _pair_clicks()
    real = StereoCalibrator.run_stereo

    def _no_count(self, *a, **k):
        record = real(self, *a, **k)
        record.board_meta.pop("n_stereo_views")
        return record

    monkeypatch.setattr(StereoCalibrator, "run_stereo", _no_count)

    body = _gen(
        client, stereo=True, camera_pair=[1, 2], clicks=c1, clicks2=c2
    ).get_json()

    assert body["success"] is False
    assert "n_stereo_views" in body["error"]


def test_load_model_stereo_without_stereo_method_reports_it(env):
    client, base, _params = env
    c1, c2 = _pair_clicks()
    body = _gen(
        client, stereo=True, camera_pair=[1, 2], clicks=c1, clicks2=c2
    ).get_json()
    record = rec.load_stereo(body["model_path"])
    record.board_meta.pop("stereo_method")
    rec.save_stereo(record, _stereo_dir(base))

    loaded = _load_model(client, stereo=1, camera_pair="1,2")

    assert loaded["exists"] is False
    assert "stereo_method" in loaded["error"]


def test_mono_generate_and_load_state_rms_units(env):
    client, _base, _params = env
    c1, _c2 = _pair_clicks()

    gen = _gen(client, camera=1, clicks=c1).get_json()
    assert gen["success"] is True, gen
    assert gen["model_type"] == "pinhole" and gen["rms_units"] == "px"

    loaded = _load_model(client, stereo=0, camera=1)
    assert loaded["exists"] is True and loaded["rms_units"] == "px"


# ---------------------------------------------------------------------------
# Item 6: snap_fiducial
# ---------------------------------------------------------------------------


def _detect_datum(client, **extra):
    body = {
        "source_path_idx": 0,
        "board": "dotboard",
        "camera": 1,
        "datum_frame": 1,
        "image_format": FMT,
        **extra,
    }
    r = client.post("/calibration/detect_datum", json=body).get_json()
    assert r["success"] is True, r
    return r


def _snap(client, **extra):
    body = {"source_path_idx": 0, "board": "dotboard", "datum_frame": 1, **extra}
    return client.post("/calibration/snap_fiducial", json=body)


def test_snap_fiducial_requires_the_click(env):
    client, _base, _params = env
    _detect_datum(client)

    r = _snap(client, camera=1, click_y=10.0)

    assert r.status_code == 400 and "click_x" in r.get_json()["error"]


def test_snap_fiducial_camera_defaults_to_rig(env):
    """No camera in the request: the rig camera (1 in the template), as detect_datum does."""
    client, _base, _params = env
    datum = _detect_datum(client)
    px = datum["image_points"][0]

    r = _snap(client, click_x=px[0] + 0.4, click_y=px[1] - 0.4)

    assert r.status_code == 200, r.get_json()
    assert r.get_json()["snapped_x"] == pytest.approx(px[0])


def test_snap_fiducial_refuses_a_datum_detected_for_another_frame(env):
    client, _base, _params = env
    _detect_datum(client, datum_frame=1)

    r = _snap(client, camera=1, datum_frame=2, click_x=10.0, click_y=10.0)

    assert r.status_code == 400
    assert "detect the datum first" in r.get_json()["error"]


def test_two_cameras_keep_separate_datum_detections(env):
    """Design A picks on both cameras at once: camera 2's detect must not evict camera 1's."""
    client, _base, _params = env
    d1 = _detect_datum(client, camera=1)
    _detect_datum(client, camera=2)

    px = d1["image_points"][5]
    r = _snap(client, camera=1, click_x=px[0], click_y=px[1])

    assert r.status_code == 200 and r.get_json()["snapped_x"] == pytest.approx(px[0])


# ---------------------------------------------------------------------------
# Item 7: a cached re-solve that needs images raises naming the frame
# ---------------------------------------------------------------------------


def test_cached_resolve_with_figures_and_missing_image_raises(env, monkeypatch):
    client, base, params = env
    c1, _c2 = _pair_clicks()
    INP.save_inputs(
        rec.mono_model_dir_for_source(base, 1, "dotboard"),
        path_type="mono",
        board_type="dotboard",
        detections={1: [_detection(t)[0] for t in TILTS]},
        image_size_by_cam={1: (W, H)},
        det_key=_mono_key(params),
        coords=c1,
    )

    def _load(cam, frame, src, fmt=None, typ=None, normalize_uint8=True):
        if frame == 2:
            raise FileNotFoundError("calibration image not found: cal_002.tif")
        return _frame_image(frame, cam)

    monkeypatch.setattr(V, "_load_one", _load)

    body = _gen(client, camera=1, no_figures=False).get_json()

    assert body["success"] is False
    assert "cal_002.tif" in body["error"]


# ---------------------------------------------------------------------------
# Item 8: model_type resolution
# ---------------------------------------------------------------------------


def test_generate_model_type_absent_uses_settings(env):
    client, base, _params = env
    settings = cs.default_settings()
    settings["image"].update(image_format=FMT, image_type="standard")
    settings["methods"]["dotboard"].update(
        dot_spacing_mm=SPACING_MM, model_type="polynomial"
    )
    cs.save_settings(base, settings)
    c1, _c2 = _pair_clicks()

    body = _gen(client, camera=1, clicks=c1).get_json()

    assert body["success"] is True, body
    assert body["model_type"] == "polynomial"
    assert body["rms_units"] == "mm"


def test_generate_model_type_null_is_an_error(env):
    client, _base, _params = env
    c1, _c2 = _pair_clicks()

    r = _gen(client, camera=1, clicks=c1, model_type=None)

    assert r.status_code == 400
    assert "model_type" in r.get_json()["error"]


# ---------------------------------------------------------------------------
# Item 10: camera_pair is required (request, else rig)
# ---------------------------------------------------------------------------


def test_detect_views_stereo_without_camera_pair_is_an_error(env, monkeypatch):
    client, _base, _params = env
    monkeypatch.setattr(V, "_sidecar_rig", lambda get: {})

    r = client.post(
        "/calibration/detect_views",
        json={
            "source_path_idx": 0,
            "board": "dotboard",
            "image_format": FMT,
            "frame_total": N_VIEWS,
            "stereo": True,
        },
    )

    assert r.status_code == 400
    assert "camera_pair is required" in r.get_json()["error"]


def test_generate_stereo_without_camera_pair_is_an_error(env, monkeypatch):
    client, _base, _params = env
    monkeypatch.setattr(V, "_sidecar_rig", lambda get: {})
    c1, c2 = _pair_clicks()

    body = _gen(client, stereo=True, clicks=c1, clicks2=c2).get_json()

    assert body["success"] is False
    assert "camera_pair is required" in body["error"]


def test_stereo_pair_never_comes_from_the_defaults_template(env, monkeypatch):
    """No sidecar + no request pair is an error even though the defaults template
    carries ``rig.camera_pair = [1, 2]`` (P6: the pair must be a user choice)."""
    client, _base, _params = env
    monkeypatch.setattr(V, "_rig", lambda get: {"camera_pair": [1, 2]})
    monkeypatch.setattr(V.cs, "try_load_settings", lambda source: None)

    r = client.post(
        "/calibration/detect_views",
        json={
            "source_path_idx": 0,
            "board": "dotboard",
            "image_format": FMT,
            "frame_total": N_VIEWS,
            "stereo": True,
        },
    )

    assert r.status_code == 400
    assert "no settings sidecar" in r.get_json()["error"]


def test_restore_detections_stereo_rig_pair_is_used(env, monkeypatch):
    """The rig pair (not a literal [1, 2]) decides which sidecar is restored."""
    client, base, _params = env
    monkeypatch.setattr(V, "_sidecar_rig", lambda get: {"camera_pair": [2, 3]})

    r = client.post(
        "/calibration/restore_detections",
        json={
            "source_path_idx": 0,
            "board": "dotboard",
            "image_format": FMT,
            "frame_total": N_VIEWS,
            "stereo": True,
        },
    ).get_json()

    assert r["exists"] is False


# ---------------------------------------------------------------------------
# Item 11: the generate write-back never overwrites a concurrent pick commit
# ---------------------------------------------------------------------------


def test_generate_does_not_write_back_a_stale_coords_read(env, monkeypatch):
    client, base, params = env
    mdir = rec.mono_model_dir_for_source(base, 1, "dotboard")
    c_old = _clicks_from_detection(_detection(TILTS[0])[0])
    c_new = dict(c_old, origin_mm=[40.0, 50.0])
    INP.save_inputs(
        mdir,
        path_type="mono",
        board_type="dotboard",
        detections={1: [_detection(t)[0] for t in TILTS]},
        image_size_by_cam={1: (W, H)},
        det_key=_mono_key(params),
        coords=c_old,
    )
    real = Calibrator.run_mono

    def _commit_during_solve(self, *a, **k):
        record = real(self, *a, **k)
        INP.save_inputs(mdir, path_type="mono", board_type="dotboard", coords=c_new)
        return record

    monkeypatch.setattr(Calibrator, "run_mono", _commit_during_solve)

    body = _gen(client, camera=1).get_json()

    assert body["success"] is False
    assert "changed while" in body["error"]
    assert INP.load_inputs(mdir).coords["origin_mm"] == [40.0, 50.0]


def test_generate_request_picks_are_written(env):
    client, base, _params = env
    c1, _c2 = _pair_clicks()
    body = _gen(client, camera=1, clicks=c1).get_json()
    assert body["success"] is True, body
    coords = INP.load_inputs(rec.mono_model_dir_for_source(base, 1, "dotboard")).coords
    assert coords["origin"] == c1["origin"] and coords["origin_mm"] == [2.5, 5.0]


# ---------------------------------------------------------------------------
# Item 14: _joint_detect persist
# ---------------------------------------------------------------------------


def _joint_env(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "_source_path", lambda idx: tmp_path)
    monkeypatch.setattr(
        V,
        "_load_views",
        lambda cam, n, src, fmt, typ: [_frame_image(k + 1, cam) for k in range(n)],
    )
    monkeypatch.setattr(
        V,
        "_detect_parallel",
        lambda board, p, imgs, spacing_mm=None, on_done=None: [
            _detection(TILTS[0])[0] for _ in imgs
        ],
    )
    V._joint_detect_cache.clear()
    params = DotboardParams(dot_spacing_mm=SPACING_MM)
    return ("dotboard", params, [1], 2, 0, FMT, "standard", SPACING_MM)


def test_joint_detect_read_only_persist_is_reported(tmp_path, monkeypatch):
    args = _joint_env(tmp_path, monkeypatch)

    def _read_only(*a, **k):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(V, "save_inputs", _read_only)

    dets, sizes, persist_skipped = V._joint_detect(*args)

    assert len(dets[1]) == 2
    assert "read-only file system" in persist_skipped


def test_joint_detect_persist_value_error_raises(tmp_path, monkeypatch):
    args = _joint_env(tmp_path, monkeypatch)

    def _bad(*a, **k):
        raise ValueError("cannot serialise detections")

    monkeypatch.setattr(V, "save_inputs", _bad)

    with pytest.raises(ValueError, match="cannot serialise"):
        V._joint_detect(*args)


def test_joint_detect_persisted_reports_none(tmp_path, monkeypatch):
    args = _joint_env(tmp_path, monkeypatch)
    _dets, _sizes, persist_skipped = V._joint_detect(*args)
    assert persist_skipped is None


# ---------------------------------------------------------------------------
# Joint generate / model: views excluded from the solve are reported
# ---------------------------------------------------------------------------

# The synthetic ChArUco rig of test_calibration_joint_charuco (corner ids link the cameras, so
# no click spec is needed). (camera, 0-based view) whose detection fails in the route tests.
_JOINT_FAILED_VIEW = (2, 3)
_JOINT_WAIT_S = 120.0


def _joint_route(tmp_path, monkeypatch):
    """Real joint routes over the synthetic ChArUco rig with one failed view.

    ``_joint_env`` supplies the source path and the per-(frame, camera) tagged images;
    ``_detect_parallel`` is replaced to hand back the rig's detection for each tagged image.
    Figure writing is stubbed (it is guarded and orthogonal to the reported fields).
    """
    from test_calibration_joint_charuco import _N_VIEWS as CH_VIEWS
    from test_calibration_joint_charuco import SPACING as CH_SPACING
    from test_calibration_joint_charuco import _dataset

    from pivtools_gui.calibration import figures as FIG
    from pivtools_gui.calibration.detection.base import DetectionResult

    _joint_env(tmp_path, monkeypatch)
    dets = _dataset(seed=3, flat=False)
    cam, view = _JOINT_FAILED_VIEW
    dets[cam][view] = DetectionResult(
        success=False,
        board_type="charuco",
        image_points=np.empty((0, 2)),
        board_local_points=np.empty((0, 3)),
    )
    monkeypatch.setattr(
        V,
        "_detect_parallel",
        # _frame_image tags pixel (0,0) with the 1-based frame and (0,1) with the camera.
        lambda board, p, imgs, spacing_mm=None, on_done=None: [
            dets[int(i[0, 1])][int(i[0, 0]) - 1] for i in imgs
        ],
    )
    monkeypatch.setattr(FIG, "write_joint_figures", lambda *a, **k: None)
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    body = {
        "source_path_idx": 0,
        "board": "charuco",
        "model_type": "pinhole",
        "cameras": [1, 2, 3],
        "n_views": CH_VIEWS,
        "image_format": FMT,
        "datum_camera": 1,
        "datum_view": 0,
        "dt": 1.0,
        "board_params": {
            "squares_h": 15,
            "squares_v": 10,
            "square_size": CH_SPACING / 1000.0,
        },
    }
    return app.test_client(), body


def _wait_joint(client, job_id: str) -> dict:
    deadline = time.time() + _JOINT_WAIT_S
    while time.time() < deadline:
        j = client.get(f"/calibration/joint/generate/status/{job_id}").get_json()
        if j.get("status") in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"joint job {job_id} did not finish within {_JOINT_WAIT_S}s")


def _joint_model(client) -> dict:
    return client.get(
        "/calibration/joint/model",
        query_string={
            "source_path_idx": 0,
            "board": "charuco",
            "model_type": "pinhole",
        },
    ).get_json()


def test_joint_generate_reports_excluded_views(tmp_path, monkeypatch):
    """The failed view is reported 1-based in the completed job AND by joint_model after save."""
    client, body = _joint_route(tmp_path, monkeypatch)
    resp = client.post("/calibration/joint/generate", json=body).get_json()
    assert resp["success"] is True, resp

    status = _wait_joint(client, resp["job_id"])

    assert status["status"] == "completed", status
    cam, view = _JOINT_FAILED_VIEW
    assert status["excluded_views"] == [[cam, view + 1]]
    model = _joint_model(client)
    assert model["exists"] is True, model
    assert model["excluded_views"] == [[cam, view + 1]]


def test_joint_model_without_excluded_views_is_stale(tmp_path, monkeypatch):
    """A joint record lacking board_meta.excluded_views is reported stale, naming the field."""
    client, body = _joint_route(tmp_path, monkeypatch)
    resp = client.post("/calibration/joint/generate", json=body).get_json()
    status = _wait_joint(client, resp["job_id"])
    assert status["status"] == "completed", status
    model_dir = rec.joint_model_dir_for_source(tmp_path, "charuco")
    record = rec.load_joint(rec.resolve_joint_path(model_dir, "pinhole"))
    record.board_meta.pop("excluded_views")
    rec.save_joint(record, model_dir)

    model = _joint_model(client)

    assert model["exists"] is False
    assert "excluded_views" in model["error"]


# ---------------------------------------------------------------------------
# DETECTOR_VERSION 4
# ---------------------------------------------------------------------------


def test_detections_from_detector_version_1_are_stale(env, monkeypatch):
    """The current detector (v4) returns different points: a v1 sidecar must not be reused."""
    client, _base, _params = env
    assert GD.DETECTOR_VERSION == 4
    monkeypatch.setattr(INP, "DETECTOR_VERSION", 1)
    _post_views(client, camera=1)
    monkeypatch.setattr(INP, "DETECTOR_VERSION", GD.DETECTOR_VERSION)

    restored = client.post(
        "/calibration/restore_detections",
        json={
            "source_path_idx": 0,
            "board": "dotboard",
            "image_format": FMT,
            "frame_total": N_VIEWS,
            "camera": 1,
        },
    ).get_json()

    assert restored["exists"] is True
    assert restored["fresh"] is False
    assert restored["by_camera"] is None
