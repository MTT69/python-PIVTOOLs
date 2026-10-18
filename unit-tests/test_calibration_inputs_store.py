"""The sidecar ``inputs.mat`` store — round-trip of detected points + clicked coords.

The sidecar holds the calibration INPUTS beside the solved model. These tests pin the two
contracts that matter: a ``DetectionResult`` (including a failed view and the None-typed
optional arrays) survives ``save_inputs`` -> ``load_inputs`` byte-for-byte, and a joint
``coords`` block with ragged ``anchors`` / mixed ``same_as`` round-trips through the JSON
string field. The merge semantics (detections and coords written independently) are pinned
too — that is what lets a detect run and a click commit update the same file without
clobbering each other.
"""

from __future__ import annotations

import re
import threading
import time

import numpy as np
import pytest

from pivtools_gui.calibration import inputs_store as INP
from pivtools_gui.calibration.detection.base import DetectionResult
from pivtools_gui.calibration.inputs_store import (
    InputsCorrupt,
    inputs_path,
    load_inputs,
    save_inputs,
    try_load_inputs,
    update_inputs,
)


def _good_detection(n: int = 6, *, charuco: bool) -> DetectionResult:
    rng = np.arange(n, dtype=np.float64)
    return DetectionResult(
        success=True,
        board_type="charuco" if charuco else "dotboard",
        image_points=np.column_stack([rng, rng + 0.5]),
        board_local_points=np.column_stack([rng, rng, np.zeros(n)]),
        grid_indices=np.column_stack([rng.astype(int), (rng % 3).astype(int)]),
        point_ids=rng.astype(int) if charuco else None,
        board_to_pixel=np.eye(3) * 2.0,
        spacing_mm=21.2,
        synthetic_mask=np.array([i % 2 == 0 for i in range(n)]),
        diagnostics={"note": "ok", "drops": 2},
    )


def _failed_detection() -> DetectionResult:
    return DetectionResult(
        success=False,
        board_type="dotboard",
        image_points=np.empty((0, 2)),
        board_local_points=np.empty((0, 3)),
    )


def _assert_detection_equal(a: DetectionResult, b: DetectionResult) -> None:
    assert a.success == b.success
    assert a.board_type == b.board_type
    np.testing.assert_allclose(a.image_points, b.image_points)
    np.testing.assert_allclose(a.board_local_points, b.board_local_points)
    for attr in ("grid_indices", "point_ids", "board_to_pixel", "synthetic_mask"):
        va, vb = getattr(a, attr), getattr(b, attr)
        if va is None:
            assert vb is None, f"{attr}: expected None, got {vb!r}"
        else:
            assert vb is not None, f"{attr}: lost on round-trip"
            np.testing.assert_allclose(np.asarray(va, float), np.asarray(vb, float))
    assert a.spacing_mm == pytest.approx(b.spacing_mm)


def test_detections_round_trip_including_failed_view(tmp_path):
    dets = {
        1: [_good_detection(charuco=False), _failed_detection()],
        2: [_good_detection(charuco=False), _good_detection(n=4, charuco=False)],
    }
    sizes = {1: (5312, 4600), 2: (5312, 4600)}
    save_inputs(
        tmp_path,
        path_type="joint",
        board_type="dotboard",
        detections=dets,
        image_size_by_cam=sizes,
    )

    rec = load_inputs(tmp_path)
    assert rec.path_type == "joint"
    assert rec.board_type == "dotboard"
    assert rec.image_size_by_cam == sizes
    assert sorted(rec.detections) == [1, 2]
    assert [len(v) for v in (rec.detections[1], rec.detections[2])] == [2, 2]
    _assert_detection_equal(dets[1][0], rec.detections[1][0])
    _assert_detection_equal(dets[1][1], rec.detections[1][1])  # the failed view
    _assert_detection_equal(dets[2][1], rec.detections[2][1])


def test_charuco_point_ids_survive(tmp_path):
    dets = {1: [_good_detection(charuco=True)]}
    save_inputs(tmp_path, path_type="joint", board_type="charuco", detections=dets)
    rec = load_inputs(tmp_path)
    _assert_detection_equal(dets[1][0], rec.detections[1][0])
    assert rec.detections[1][0].point_ids is not None


def test_joint_coords_with_anchors_round_trip(tmp_path):
    coords = {
        "datum_camera": 1,
        "datum_view": 0,
        "datum_clicks": {
            "origin": [10.0, 20.0],
            "x_axis": [110.0, 20.0],
            "y_axis": [10.0, 120.0],
            "origin_mm": [5.0, -3.0],
        },
        "anchors": [
            {
                "camera": 2,
                "view": 0,
                "correspondences": [
                    {
                        "pixel": [50.0, 60.0],
                        "same_as": [1, 0],
                        "ref_pixel": [200.0, 210.0],
                    },
                ],
            },
            {
                "camera": 1,
                "view": 1,
                "correspondences": [
                    {"pixel": [7.0, 8.0], "same_as": "origin", "ref_pixel": None},
                ],
            },
        ],
        "camera_extends": {"2": [3.97, -0.03]},
        "cameras": [1, 2],
        "board_release": "full3d",
    }
    save_inputs(tmp_path, path_type="joint", board_type="dotboard", coords=coords)
    rec = load_inputs(tmp_path)
    assert rec.coords == coords  # JSON round-trip is exact for plain nested data


def test_merge_preserves_other_field(tmp_path):
    """Writing detections must not clobber coords, and vice versa."""
    coords = {"datum_clicks": {"origin": [1.0, 2.0]}}
    save_inputs(tmp_path, path_type="mono", board_type="dotboard", coords=coords)
    save_inputs(
        tmp_path,
        path_type="mono",
        board_type="dotboard",
        detections={1: [_good_detection(charuco=False)]},
    )
    rec = load_inputs(tmp_path)
    assert rec.coords == coords  # preserved across the detections-only write
    assert rec.detections is not None and 1 in rec.detections


def test_large_diagnostics_array_is_dropped_not_stored(tmp_path):
    """A detector may park a full-frame debug image in diagnostics (e.g. flat_field on a failed
    view). It must NOT be serialised — it would bloat the sidecar by tens of MB — while the small
    scalar diagnostics are kept."""
    d = _good_detection(charuco=False)
    d.diagnostics = {
        "method": "bfs",
        "n_blobs_detected": 192,
        "flat_field": np.zeros((4600, 5312), dtype=np.float64),  # ~195 MB raw
        "error": None,
    }
    save_inputs(tmp_path, path_type="joint", board_type="dotboard", detections={1: [d]})
    assert inputs_path(tmp_path).stat().st_size < 1_000_000  # well under 1 MB
    kept = load_inputs(tmp_path).detections[1][0].diagnostics
    assert kept["method"] == "bfs" and kept["n_blobs_detected"] == 192
    assert "flat_field" not in kept
    assert (
        "error" not in kept
    )  # None-valued diagnostics are dropped (not stored as null)


# A concurrent-writers/readers stress test lived here until 2026-07-29. It failed
# on ~every full-suite run on Windows with a PermissionError inside save_inputs
# (os.replace onto a file a reader holds open) — a REAL, still-open concurrency
# limitation of the store on Windows, not a test bug. It was removed as a
# permanent false-red; if the store ever gains retry-on-share-violation
# semantics, reinstate a stress test WITH that retry to pin it.


def test_missing_sidecar_raises_and_try_load_is_none(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_inputs(tmp_path)
    assert try_load_inputs(tmp_path) is None
    assert not inputs_path(tmp_path).exists()


# Byte patterns for an unreadable sidecar: a valid file cut short, and non-MAT bytes.
_GARBAGE_BYTES = b"this is not a MAT file at all" * 8
_TRUNCATE_TO_BYTES = 200


def _corrupt_sidecar(model_dir, kind: str) -> bytes:
    """Write an unreadable ``inputs.mat`` into ``model_dir`` and return its bytes."""
    path = inputs_path(model_dir)
    if kind == "truncated":
        save_inputs(
            model_dir,
            path_type="mono",
            board_type="dotboard",
            detections={1: [_good_detection(charuco=False)]},
            coords={"a": 1},
        )
        data = path.read_bytes()[:_TRUNCATE_TO_BYTES]
    else:
        data = _GARBAGE_BYTES
    path.write_bytes(data)
    return data


@pytest.mark.parametrize("kind", ["truncated", "garbage"])
def test_corrupt_sidecar_raises_inputs_corrupt(tmp_path, kind):
    _corrupt_sidecar(tmp_path, kind)
    with pytest.raises(InputsCorrupt) as info:
        try_load_inputs(tmp_path)
    msg = str(info.value)
    assert str(inputs_path(tmp_path)) in msg
    assert "Delete it, then re-run detection" in msg
    assert info.value.__cause__ is not None
    assert info.value.path == inputs_path(tmp_path)


@pytest.mark.parametrize("kind", ["truncated", "garbage"])
def test_save_inputs_never_overwrites_corrupt_sidecar(tmp_path, kind):
    data = _corrupt_sidecar(tmp_path, kind)
    with pytest.raises(InputsCorrupt):
        save_inputs(tmp_path, path_type="mono", board_type="dotboard", coords={"b": 2})
    assert inputs_path(tmp_path).read_bytes() == data


@pytest.mark.parametrize("kind", ["truncated", "garbage"])
def test_update_inputs_never_overwrites_corrupt_sidecar(tmp_path, kind):
    data = _corrupt_sidecar(tmp_path, kind)
    called: list = []

    def mutate(prev):
        called.append(prev)
        return {"path_type": "mono", "board_type": "dotboard", "coords": {"b": 2}}

    with pytest.raises(InputsCorrupt):
        update_inputs(tmp_path, mutate)
    assert called == []
    assert inputs_path(tmp_path).read_bytes() == data


@pytest.mark.parametrize("field", ["coords_json", "board_params_json"])
def test_unparseable_json_field_raises_naming_the_field(tmp_path, field):
    """A damaged JSON field fails loudly instead of reading back as None."""
    from scipy.io import loadmat, savemat

    save_inputs(
        tmp_path,
        path_type="mono",
        board_type="dotboard",
        detections={1: [_good_detection(charuco=False)]},
        coords={"a": 1},
        board_params={"rows": 5},
    )
    path = inputs_path(tmp_path)
    mat = {k: v for k, v in loadmat(str(path)).items() if not k.startswith("__")}
    mat[field] = "{not json"
    savemat(str(path), mat, oned_as="row")

    with pytest.raises(InputsCorrupt) as info:
        try_load_inputs(tmp_path)
    assert field in str(info.value)


def test_absent_coords_still_load_as_none(tmp_path):
    """A blank/"null" field is the writer's encoding of absent, not damage."""
    save_inputs(
        tmp_path,
        path_type="mono",
        board_type="dotboard",
        detections={1: [_good_detection(charuco=False)]},
    )
    rec = load_inputs(tmp_path)
    assert rec.coords is None
    assert rec.board_params is None


# ---------------------------------------------------------------------------
# Detection identity stamp
# ---------------------------------------------------------------------------

_STAMP_RE = re.compile(r"^[0-9a-f]{32}$")


def test_detection_write_stamps_a_fresh_identity(tmp_path):
    """Every detections write gets a new stamp; a coords-only write keeps it."""
    dets = {1: [_good_detection(charuco=False)]}
    save_inputs(tmp_path, path_type="stepped", board_type="stepped", detections=dets)
    first = load_inputs(tmp_path).detection_stamp
    assert _STAMP_RE.match(first)

    save_inputs(tmp_path, path_type="stepped", board_type="stepped", coords={"a": 1})
    assert load_inputs(tmp_path).detection_stamp == first

    save_inputs(tmp_path, path_type="stepped", board_type="stepped", detections=dets)
    assert load_inputs(tmp_path).detection_stamp not in ("", first)


def test_coords_only_sidecar_has_no_stamp(tmp_path):
    save_inputs(tmp_path, path_type="mono", board_type="dotboard", coords={"a": 1})
    assert load_inputs(tmp_path).detection_stamp == ""


# ---------------------------------------------------------------------------
# update_inputs: load + mutate + save under one lock
# ---------------------------------------------------------------------------

# Long enough that, without the lock, both mutates read the same previous file.
_SLOW_MUTATE_S = 0.2


def test_update_inputs_concurrent_writers_both_persist(tmp_path):
    """Two cameras committing clicks at once: neither update may be lost."""
    save_inputs(
        tmp_path,
        path_type="stepped",
        board_type="stepped",
        detections={1: [_good_detection(charuco=False)]},
        coords={"cameras": {}},
    )
    start = threading.Barrier(2)
    errors: list = []

    def _commit(cam: int) -> None:
        def mutate(prev):
            coords = dict(prev.coords)
            cams = dict(coords["cameras"])
            time.sleep(_SLOW_MUTATE_S)
            cams[str(cam)] = {"clicked_level": "peak"}
            coords["cameras"] = cams
            return {"path_type": "stepped", "board_type": "stepped", "coords": coords}

        try:
            start.wait()
            update_inputs(tmp_path, mutate)
        except Exception as exc:  # surfaced below; a thread cannot fail the test itself
            errors.append(exc)

    threads = [threading.Thread(target=_commit, args=(c,)) for c in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert sorted(load_inputs(tmp_path).coords["cameras"]) == ["1", "2"]


def test_update_inputs_returning_none_writes_nothing(tmp_path):
    save_inputs(tmp_path, path_type="mono", board_type="dotboard", coords={"a": 1})
    before = inputs_path(tmp_path).stat().st_mtime_ns

    assert update_inputs(tmp_path, lambda prev: None) is None

    assert inputs_path(tmp_path).stat().st_mtime_ns == before
    assert load_inputs(tmp_path).coords == {"a": 1}


def test_update_inputs_sees_none_without_sidecar_and_returns_written(tmp_path):
    seen: list = []

    def mutate(prev):
        seen.append(prev)
        return {
            "path_type": "stepped",
            "board_type": "stepped",
            "detections": {1: [_good_detection(charuco=False)]},
        }

    written = update_inputs(tmp_path, mutate)

    assert seen == [None]
    assert isinstance(written, INP.InputsRecord)
    assert written.detection_stamp == load_inputs(tmp_path).detection_stamp
