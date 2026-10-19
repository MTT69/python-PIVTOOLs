"""Stepped-board routes and stitch: no silent fallbacks (P4 1b, item 12 + orientation).

Route tests drive the REAL stepped Flask handlers and job threads against the synthetic
stepped scene of ``test_calibration_stepped_routes`` (only the image reader, config and
frame count are mocked there). What is pinned:

- the stitch honours an ``orientation_override`` in every branch (no-anchor and
  single-level included), so the returned indices match the metadata it reports;
- generate merges the picks per camera key, refuses a live sequence whose sidecar was
  re-detected elsewhere, and a mono generate solves the camera it was asked for;
- detect counts the frames when ``num_frames`` is absent and refuses more than exist;
- a read-only detect keeps the previous stamp and the later save says why it failed;
- ``inputs/save`` always explains ``saved: false`` and rejects a malformed camera block;
- ``restore_sequence``: first use is a plain ``exists: false``; real errors keep their
  status and message.
"""

from __future__ import annotations

# ruff: noqa: F811  (imported pytest fixtures are re-bound by every test's parameter)
import numpy as np
import pytest
from test_calibration_stepped_routes import (  # noqa: F401  (fixtures); noqa: E402  (sibling-test reuse)
    FRAMES,
    _detect,
    _spec,
    _start_detect,
    _stereo_model_dir,
    _wait,
    client,
    scene,
)

import pivtools_gui.calibration.app.stepped_views as sv
from pivtools_gui.calibration import inputs_store as INP
from pivtools_gui.calibration.detection.stepped_levels import (
    SteppedBoardSpec,
    stitch_levels_pose_local,
)
from pivtools_gui.calibration.inputs_store import load_inputs, update_inputs

# ---------------------------------------------------------------------------
# Stitch: the orientation override reaches every branch
# ---------------------------------------------------------------------------

_OVERRIDE = {"swap_axes": True, "col_sign": -1, "row_sign": 1}


def _level(n_cols: int, n_rows: int, origin_px=(0.0, 0.0), pitch_px: float = 10.0):
    """A level's BFS output: centres on a pixel lattice, raw (col, row) indices."""
    cols, rows = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    gi = np.column_stack([cols.ravel(), rows.ravel()]).astype(np.int32)
    centers = np.asarray(origin_px) + gi * pitch_px
    return {"centers": centers.astype(float), "grid_indices": gi}


def _oriented(gi):
    out = np.asarray(gi)[:, ::-1].copy()
    out[:, 0] *= -1
    return out


def test_stitch_no_anchor_branch_applies_the_orientation_override():
    """Zero anchors: the reference indices must carry the override, as when anchored."""
    board = SteppedBoardSpec()
    ref = _level(5, 4)
    anchored_other = _level(4, 3, origin_px=(5.0, 5.0))  # cell centres -> anchors
    lonely_other = {"centers": np.array([[900.0, 900.0]]), "grid_indices": [[0, 0]]}

    anchored = stitch_levels_pose_local(ref, anchored_other, board, _OVERRIDE)
    no_anchor = stitch_levels_pose_local(ref, lonely_other, board, _OVERRIDE)

    assert anchored["metadata"]["n_anchors"] > 0
    assert no_anchor["metadata"]["n_anchors"] == 0
    np.testing.assert_array_equal(
        no_anchor["reference"]["grid_indices"], anchored["reference"]["grid_indices"]
    )
    np.testing.assert_array_equal(
        no_anchor["reference"]["grid_indices"], _oriented(ref["grid_indices"])
    )


@pytest.mark.parametrize("missing", ["A", "B"])
def test_stitch_single_level_honours_the_orientation_override(missing):
    board = SteppedBoardSpec()
    lone = _level(5, 4)
    a, b = (None, lone) if missing == "A" else (lone, None)

    out = stitch_levels_pose_local(a, b, board, _OVERRIDE)

    np.testing.assert_array_equal(
        out["reference"]["grid_indices"], _oriented(lone["grid_indices"])
    )
    meta = out["metadata"]
    assert (meta["swap_axes"], meta["col_sign"], meta["row_sign"]) == (True, -1, 1)
    assert meta["degraded_single_level"] is True


def test_stitch_single_level_without_override_is_identity():
    lone = _level(3, 2)
    out = stitch_levels_pose_local(lone, None, SteppedBoardSpec())
    np.testing.assert_array_equal(
        out["reference"]["grid_indices"], lone["grid_indices"]
    )


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def _generate(client, scene, sid, cams, stereo, **extra):
    body = {
        "sequence_id": sid,
        "stereo": stereo,
        "dt": 1.0,
        "no_figures": True,
        "stereo_config": "auto",
        "cameras": {str(c): _spec(scene, c) for c in cams},
        **extra,
    }
    return client.post("/calibration/stepped/generate_model", json=body)


def test_generate_merges_picks_per_camera_key(client, scene, tmp_path):
    sid, _ = _detect(client, [1, 2])
    mdir = _stereo_model_dir(tmp_path)
    extra = {"fiducials": {"origin": [1.0, 1.0]}, "clicked_level": "peak"}

    def _add_cam3(prev):
        coords = dict(prev.coords)
        coords["cameras"] = {"3": extra}
        return {"path_type": "stepped", "board_type": "stepped", "coords": coords}

    update_inputs(mdir, _add_cam3)

    r = _generate(client, scene, sid, (1, 2), True)
    done = _wait(
        client, f"/calibration/stepped/generate_model/status/{r.get_json()['job_id']}"
    )
    assert done["status"] == "completed", done

    cams = load_inputs(mdir).coords["cameras"]
    assert sorted(cams) == ["1", "2", "3"]
    assert cams["3"] == extra


def test_generate_refuses_a_live_sequence_redetected_elsewhere(client, scene):
    sid_old, _ = _detect(client, [1, 2])
    _detect(client, [1, 2])  # another tab re-detects: the sidecar now holds a new set

    r = _generate(client, scene, sid_old, (1, 2), True)

    assert r.status_code == 409, r.get_json()
    assert "re-detected" in r.get_json()["error"]


def test_mono_generate_solves_the_requested_camera(client, scene):
    sid, _ = _detect(client, [1, 2])

    r = _generate(client, scene, sid, (2,), False)

    body = r.get_json()
    assert r.status_code == 200, body
    done = _wait(client, f"/calibration/stepped/generate_model/status/{body['job_id']}")
    assert done["status"] == "completed", done
    assert done["camera"] == 2
    assert done["rms_units"] == "px"


def test_mono_generate_refuses_a_camera_not_in_the_sequence(client, scene):
    sid, _ = _detect(client, [1])

    r = _generate(client, scene, sid, (2,), False)

    assert r.status_code == 400
    assert "camera 2" in r.get_json()["error"]


def test_stereo_generate_requires_the_pair_count_in_the_record(
    client, scene, monkeypatch
):
    sid, _ = _detect(client, [1, 2])
    real = sv.calibrate_stepped_stereo

    def _no_count(*a, **k):
        record = real(*a, **k)
        record.board_meta.pop("n_stereo_views")
        return record

    monkeypatch.setattr(sv, "calibrate_stepped_stereo", _no_count)

    r = _generate(client, scene, sid, (1, 2), True)
    done = _wait(
        client, f"/calibration/stepped/generate_model/status/{r.get_json()['job_id']}"
    )

    assert done["status"] == "failed"
    assert "n_stereo_views" in done["error"]


# ---------------------------------------------------------------------------
# Detect: frame count
# ---------------------------------------------------------------------------


def _post_detect(client, **extra):
    body = {
        "source_path_idx": 0,
        "cameras": [1, 2],
        "start_frame_idx": 1,
        "datum_frame_idx": 1,
        **extra,
    }
    return client.post("/calibration/stepped/detect_sequence", json=body)


def test_detect_without_num_frames_uses_the_frame_count(client, scene):
    r = _post_detect(client)
    body = r.get_json()
    assert r.status_code == 200, body
    done = _wait(
        client, f"/calibration/stepped/detect_sequence/status/{body['job_id']}"
    )
    assert done["status"] == "completed", done
    assert done["frame_indices"] == FRAMES


def test_detect_refuses_more_frames_than_exist(client, scene):
    r = _post_detect(client, num_frames=len(FRAMES) + 1)

    assert r.status_code == 400
    err = r.get_json()["error"]
    assert f"has {len(FRAMES)}" in err and "camera 1" in err


# ---------------------------------------------------------------------------
# Read-only detect: previous stamp kept, the later save says why
# ---------------------------------------------------------------------------


def test_read_only_detect_keeps_previous_stamp_and_save_explains(
    client, scene, monkeypatch
):
    _sid, first = _detect(client, [1, 2])
    stamp = first["detection_stamp"]
    assert stamp

    def _read_only_save(source, entry):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(sv, "_save_sequence_sidecar", _read_only_save)
    done = _start_detect(client, [1, 2])
    assert done["status"] == "completed", done
    assert done["persisted"] is False
    assert done["detection_stamp"] == stamp
    assert "detections were not saved (read-only source)" in done["persist_skipped"]

    def _read_only_update(model_dir, mutate):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(sv, "update_inputs", _read_only_update)
    r = client.post(
        "/calibration/stepped/inputs/save",
        json={
            "camera_pair": "1,2",
            "source_path_idx": 0,
            "detection_stamp": stamp,
            "cameras": {"1": _spec(scene, 1)},
        },
    )
    body = r.get_json()
    assert body["saved"] is False
    assert "detections were not saved (read-only source)" in body["error"]


# ---------------------------------------------------------------------------
# inputs/save and restore_sequence explain every refusal
# ---------------------------------------------------------------------------


def test_inputs_save_without_detection_says_so(client, scene):
    r = client.post(
        "/calibration/stepped/inputs/save",
        json={
            "camera_pair": "1,2",
            "source_path_idx": 0,
            "detection_stamp": "0" * 32,
            "cameras": {"1": _spec(scene, 1)},
        },
    )
    body = r.get_json()
    assert body["saved"] is False and "detect" in body["error"].lower()


def test_inputs_save_rejects_a_non_dict_camera_block(client, scene):
    _sid, done = _detect(client, [1, 2])
    r = client.post(
        "/calibration/stepped/inputs/save",
        json={
            "camera_pair": "1,2",
            "source_path_idx": 0,
            "detection_stamp": done["detection_stamp"],
            "cameras": {"1": ["not", "a", "dict"]},
        },
    )
    assert r.status_code == 400
    assert "camera 1" in r.get_json()["error"]


def test_inputs_save_unconfigured_source_says_so(client, scene, monkeypatch):
    class _NoSource:
        calibration: dict = {}

        def get_calibration_source(self, idx):
            raise IndexError(f"no calibration source {idx}")

    monkeypatch.setattr(sv, "get_config", lambda: _NoSource())
    r = client.post(
        "/calibration/stepped/inputs/save",
        json={"camera_pair": "1,2", "source_path_idx": 3, "cameras": {}},
    )
    body = r.get_json()
    assert body["saved"] is False and "source" in body["error"]


def test_restore_sequence_first_use_is_a_plain_absent(client):
    r = client.get(
        "/calibration/stepped/restore_sequence",
        query_string={"camera_pair": "1,2", "source_path_idx": 0},
    )
    assert r.status_code == 200 and r.get_json() == {"exists": False}


def test_restore_sequence_reports_a_missing_descriptor(client, scene, tmp_path):
    _detect(client, [1, 2])
    sv._sequence_cache.clear()

    def _drop_sequence(prev):
        coords = {k: v for k, v in prev.coords.items() if k != "sequence"}
        return {"path_type": "stepped", "board_type": "stepped", "coords": coords}

    update_inputs(_stereo_model_dir(tmp_path), _drop_sequence)

    r = client.get(
        "/calibration/stepped/restore_sequence",
        query_string={"camera_pair": "1,2", "source_path_idx": 0},
    )
    assert r.status_code == 410
    assert "sequence descriptor" in r.get_json()["error"]


def test_restore_sequence_unconfigured_source_is_an_error(client, monkeypatch):
    class _NoSource:
        calibration: dict = {}

        def get_calibration_source(self, idx):
            raise IndexError(f"no calibration source {idx}")

    monkeypatch.setattr(sv, "get_config", lambda: _NoSource())
    r = client.get(
        "/calibration/stepped/restore_sequence",
        query_string={"camera_pair": "1,2", "source_path_idx": 3},
    )
    assert r.status_code == 400 and "not configured" in r.get_json()["error"]


# ---------------------------------------------------------------------------
# Item 13 (implemented by agent 1a in inputs_store): a read error is not "corrupt"
# ---------------------------------------------------------------------------


def test_try_load_inputs_lets_a_read_error_through(tmp_path, monkeypatch):
    """An OSError reading inputs.mat is a read error, not a damaged file: it must not be
    wrapped as InputsCorrupt (whose message tells the user to delete the sidecar)."""
    INP.save_inputs(tmp_path, path_type="stepped", board_type="stepped", coords={})

    def _unreadable(model_dir):
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(INP, "load_inputs", _unreadable)

    with pytest.raises(OSError) as info:
        INP.try_load_inputs(tmp_path)
    assert not isinstance(info.value, INP.InputsCorrupt)
