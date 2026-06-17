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

import numpy as np
import pytest

from pivtools_gui.calibration.detection.base import DetectionResult
from pivtools_gui.calibration.inputs_store import (
    inputs_path,
    load_inputs,
    save_inputs,
    try_load_inputs,
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
        tmp_path, path_type="joint", board_type="dotboard",
        detections=dets, image_size_by_cam=sizes,
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
            "origin": [10.0, 20.0], "x_axis": [110.0, 20.0],
            "y_axis": [10.0, 120.0], "origin_mm": [5.0, -3.0],
        },
        "anchors": [
            {"camera": 2, "view": 0, "correspondences": [
                {"pixel": [50.0, 60.0], "same_as": [1, 0], "ref_pixel": [200.0, 210.0]},
            ]},
            {"camera": 1, "view": 1, "correspondences": [
                {"pixel": [7.0, 8.0], "same_as": "origin", "ref_pixel": None},
            ]},
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
        tmp_path, path_type="mono", board_type="dotboard",
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
    assert "error" not in kept  # None-valued diagnostics are dropped (not stored as null)


def test_concurrent_writes_and_reads_never_corrupt(tmp_path):
    """Readers during a write must never see a truncated file (the live overlay reads the sidecar
    on every frame while detect/click commits write it). Atomic temp-then-replace + the write lock
    guarantee a reader gets either the old or the new file, never a half-written one."""
    import threading

    dets = {1: [_good_detection(charuco=False) for _ in range(6)],
            2: [_good_detection(n=200, charuco=False) for _ in range(6)]}
    save_inputs(tmp_path, path_type="joint", board_type="dotboard", detections=dets,
                image_size_by_cam={1: (5312, 4600), 2: (5312, 4600)})

    errors: list = []

    def writer():
        for i in range(15):
            try:
                save_inputs(tmp_path, path_type="joint", board_type="dotboard",
                            coords={"datum_clicks": {"origin": [float(i), float(i)]}})
            except Exception as e:  # noqa: BLE001 - test asserts no error
                errors.append(("write", e))

    def reader():
        for _ in range(40):
            try:
                load_inputs(tmp_path)  # strict: must NOT raise on a complete file
            except Exception as e:  # noqa: BLE001
                errors.append(("read", e))

    threads = [threading.Thread(target=writer) for _ in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    rec = load_inputs(tmp_path)  # final file is valid; detections survived the coord writes
    assert rec.detections is not None and len(rec.detections[1]) == 6
    assert not list(tmp_path.glob("*.tmp"))  # no temp leftovers


def test_missing_sidecar_raises_and_try_load_is_none(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_inputs(tmp_path)
    assert try_load_inputs(tmp_path) is None
    assert not inputs_path(tmp_path).exists()
