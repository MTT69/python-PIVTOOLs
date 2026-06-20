"""Stepped-board detection sidecar: persistence + reuse round-trip (Phase 1).

The stepped flow holds detections in a process-memory cache during a session; the
sidecar (``model/inputs.mat``) is the durable store that lets a model be regenerated
from disk — detections AND the operator's clicks — without re-detecting or re-clicking.
These tests pin the round-trip and the det_key invalidation that gates reuse.
"""

import numpy as np

from pivtools_gui.calibration import record as rec
from pivtools_gui.calibration.app import stepped_views as sv
from pivtools_gui.calibration.detection.base import DetectionResult
from pivtools_gui.calibration.detection.stepped_levels import SteppedBoardSpec
from pivtools_gui.calibration.inputs_store import save_inputs, try_load_inputs


def _det(ok: bool = True):
    """A minimal successful detection, or None for a failed pose."""
    if not ok:
        return None
    return DetectionResult(
        success=True,
        board_type="stepped",
        image_points=np.array([[10.0, 20.0], [30.0, 40.0]]),
        board_local_points=np.array([[0.0, 0.0, 0.0], [15.0, 0.0, 0.0]]),
        grid_indices=np.array([[0, 0], [1, 0]]),
    )


def _entry(params):
    return {
        "cameras": [1, 2],
        "frame_indices": [1, 2, 3],
        "datum_frame_idx": 1,
        "source_idx": 0,
        "params": params,
        "image_format": None,
        "image_type": None,
        # Pose 2 (cam1) and pose 3 (cam2) failed -> None, to prove the slots survive.
        "detections": {1: [_det(), None, _det()], 2: [_det(), _det(), None]},
        "images": {1: [None, None, None], 2: [None, None, None]},
        "image_size": {1: (640, 480), 2: (640, 480)},
    }


def test_sequence_sidecar_round_trip(tmp_path):
    params = SteppedBoardSpec()
    sv._save_sequence_sidecar(tmp_path, _entry(params))

    model_dir = rec.stereo_model_dir_for_source(tmp_path, 1, 2)
    assert (model_dir / "inputs.mat").is_file()

    side = try_load_inputs(model_dir)
    assert side is not None

    # det_key is stable and recomputable from the same inputs (gates reuse).
    assert side.det_key == sv._stepped_det_key([1, 2], [1, 2, 3], 1, None, None, params)

    # Sequence descriptor persisted so a later session regenerates from the file alone.
    seq = side.coords["sequence"]
    assert seq["frame_indices"] == [1, 2, 3]
    assert seq["datum_frame_idx"] == 1
    assert side.board_params["dot_spacing_mm"] == params.dot_spacing_mm

    # Detections round-trip with the failed-pose slots preserved as None (positional).
    recon = sv._dets_from_sidecar(side.detections)
    assert [d is None for d in recon[1]] == [False, True, False]
    assert [d is None for d in recon[2]] == [False, False, True]
    np.testing.assert_allclose(recon[1][0].image_points, _det().image_points)


def test_det_key_invalidates_on_frame_change():
    params = SteppedBoardSpec()
    base = sv._stepped_det_key([1, 2], [1, 2, 3], 1, None, None, params)
    moved_window = sv._stepped_det_key([1, 2], [1, 2, 4], 1, None, None, params)
    moved_datum = sv._stepped_det_key([1, 2], [1, 2, 3], 2, None, None, params)
    assert base != moved_window
    assert base != moved_datum


def test_coords_spec_merges_with_sequence(tmp_path):
    """The clicks block persists and merges with the sequence descriptor (no clobber)."""
    params = SteppedBoardSpec()
    sv._save_sequence_sidecar(tmp_path, _entry(params))
    model_dir = rec.stereo_model_dir_for_source(tmp_path, 1, 2)

    specs = {
        1: (
            {"origin": [1.0, 2.0], "x_axis": [3.0, 2.0], "y_axis": [1.0, 4.0]},
            "peak",
            {1: "peak", 2: "trough", 3: "peak"},
        ),
        2: (
            {"origin": [5.0, 6.0], "x_axis": [7.0, 6.0], "y_axis": [5.0, 8.0]},
            "trough",
            {1: "trough", 2: "peak", 3: "trough"},
        ),
    }
    coords = dict(try_load_inputs(model_dir).coords)
    coords.update(sv._spec_to_coords(specs, [1, 2], True, {"stereo_config": "same_side"}))
    save_inputs(model_dir, path_type="stepped", board_type="stepped", coords=coords)

    side = try_load_inputs(model_dir)
    # Sequence block survived the clicks save (independent merge).
    assert side.coords["sequence"]["frame_indices"] == [1, 2, 3]
    assert side.coords["stereo_config"] == "same_side"
    assert side.coords["cameras"]["1"]["clicked_level"] == "peak"
    assert side.coords["cameras"]["2"]["pose_levels"]["2"] == "peak"
