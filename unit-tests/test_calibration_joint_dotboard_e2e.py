"""S2·C4 — dotboard joint solve, end-to-end through the GUI/CLI driver.

The resolver (``test_calibration_global_grid``) proves a clicked spec -> correct global indices,
and ``test_calibration_joint`` proves run_joint recovers geometry from a global index. What was
NOT covered is the JOIN: a realistic clicked ``GlobalGridSpec`` (datum 3-click + cross-camera
2-click links + within-camera 1-click "rescue" anchors) driven all the way through
``run_joint_from_spec`` -> a saved, reloadable ``JointRecord`` / per-camera polynomial. That is
exactly the path the dotboard GUI (S2·C2/C3) and the CLI ``detect-joint`` share, so this is the
headless equivalent of driving the dotboard joint flow in the browser.

The synthetic three-camera dotboard + its spec are reused verbatim from the resolver test, so the
geometry and the click record are identical to what that test validates the resolver on.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_calibration_global_grid import (  # noqa: E402  (sibling-test reuse, as elsewhere)
    _N_VIEWS,
    SPACING,
    _make_dataset,
    _spec,
)

from pivtools_gui.calibration.camera_model import PolynomialModel
from pivtools_gui.calibration.detection.base import DetectionResult
from pivtools_gui.calibration.joint_driver import run_joint_from_spec
from pivtools_gui.calibration.record import (
    format_excluded_views,
    load_joint,
    load_mono,
    parse_excluded_views,
)

# The synthetic projects around principal point (640, 512) with a 1280x1024-ish sensor.
_IMAGE_SIZE = (1280, 1024)


def _image_size_by_cam(detections) -> dict:
    return {c: _IMAGE_SIZE for c in detections}


def test_dotboard_joint_pinhole_end_to_end(tmp_path):
    """A clicked dotboard spec -> one shared released board, jointly solved across 3 cameras.

    Noise-free synthetic projected with a pinhole (no distortion), so the joint solve must
    reproject to sub-pixel and every camera must agree on the one shared board by construction.
    """
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    cams = sorted(detections)

    res = run_joint_from_spec(
        detections,
        _image_size_by_cam(detections),
        source=tmp_path,
        board="dotboard",
        model_type="pinhole",
        spacing_mm=SPACING,
        dt=1.0,
        datum_camera=1,
        datum_view=0,
        spec=spec,
        cameras=cams,
        n_views=_N_VIEWS,
    )

    assert res.board_type == "dotboard"
    assert res.model_type == "pinhole"
    assert sorted(res.cameras) == cams
    # One shared board: cross-camera agreement is exact by construction.
    assert res.cross_camera_board_agreement_mm == pytest.approx(0.0, abs=1e-9)
    # Noise-free pinhole data -> the joint bundle reprojects tightly.
    assert np.isfinite(res.rms_px) and res.rms_px < 1.0
    assert res.rms_units == "px"
    for c in cams:
        assert res.per_camera_rms[c] < 1.0
    assert (
        res.n_board_dots > 40
    )  # a real shared board (cols 0..11 x rows 0..8, partial per cam)

    # Stage rms and the board mode actually run travel with the driver result.
    assert res.info["effective_board_release"] == "full3d"
    assert res.info["rms_after_alternation"] <= res.rms_px + 1e-9
    assert res.info["n_released"] <= res.n_board_dots
    assert res.pose_diversity is not None and res.pose_diversity.degenerate is False
    assert res.excluded_views == []  # every view detected and anchored
    # The record is on disk and reloads with the same shape.
    assert len(res.paths) == 1
    jr = load_joint(res.paths[0])
    assert jr.board_type == "dotboard"
    assert jr.pose_diversity  # stored, not {}
    assert sorted(jr.cameras) == cams
    assert jr.spacing_mm == pytest.approx(SPACING)
    for c in cams:
        assert c in jr.models
    assert parse_excluded_views(jr.board_meta["excluded_views"]) == []


def test_dotboard_joint_polynomial_end_to_end(tmp_path):
    """model_type=polynomial fits a per-camera single-plane map in the one shared global frame.

    Each camera's datum view (resolved into the shared frame via the cross-camera links) maps
    image px -> global (X, Y) mm; for a planar datum view the fit is essentially exact.
    """
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    cams = sorted(detections)

    res = run_joint_from_spec(
        detections,
        _image_size_by_cam(detections),
        source=tmp_path,
        board="dotboard",
        model_type="polynomial",
        spacing_mm=SPACING,
        dt=1.0,
        datum_camera=1,
        datum_view=0,
        spec=spec,
        cameras=cams,
        n_views=_N_VIEWS,
    )

    assert res.model_type == "polynomial"
    assert res.rms_units == "mm"
    # One per-camera mono record (polynomial has no single shared object to unify).
    assert len(res.paths) == len(cams)
    for c in cams:
        assert res.per_camera_rms[c] < 1.0

    for p in res.paths:
        mono = load_mono(p)
        assert isinstance(mono.camera_model, PolynomialModel)
        assert mono.world_frame.mode == "global_grid"
        assert mono.camera_model.rms_x_mm < 1.0 and mono.camera_model.rms_y_mm < 1.0


def test_dotboard_joint_shared_frame_invariant(tmp_path):
    """The defining shared-frame property: a physical dot two cameras both see gets ONE world
    position. Dot (5,3) is in cam1 (cols 0..6) and cam2 (cols 4..9) at view 0; after the joint
    solve its released-board coordinate is identical regardless of which camera observed it.
    """
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    cams = sorted(detections)

    res = run_joint_from_spec(
        detections,
        _image_size_by_cam(detections),
        source=tmp_path,
        board="dotboard",
        model_type="pinhole",
        spacing_mm=SPACING,
        dt=1.0,
        datum_camera=1,
        datum_view=0,
        spec=spec,
        cameras=cams,
        n_views=_N_VIEWS,
    )

    jr = load_joint(res.paths[0])
    # The shared board is keyed by global index; (5,3) resolves to one coordinate full stop.
    assert (5, 3) in jr.board
    xyz = np.asarray(jr.board[(5, 3)], dtype=float)
    assert xyz.shape == (3,)
    # In-plane it sits at the nominal lattice position (origin (0,0), in-plane gauge-locked);
    # z is the released bow. A 1 mm tolerance confirms the correct lattice cell (15 mm pitch).
    np.testing.assert_allclose(xyz[:2], [5 * SPACING, 3 * SPACING], atol=1.0)


# The view taken out of the solve in the exclusion tests: camera 1, 0-based view 3 (frame 4).
_DROP_CAM, _DROP_VIEW = 1, 3


def _drop_anchor(spec, cam: int, view: int):
    """The spec without the click anchoring ``(cam, view)`` to the global grid."""
    spec.anchors = [a for a in spec.anchors if (a.camera, a.view) != (cam, view)]
    return spec


def _failed_dotboard() -> DetectionResult:
    return DetectionResult(
        success=False,
        board_type="dotboard",
        image_points=np.empty((0, 2)),
        board_local_points=np.empty((0, 3)),
    )


@pytest.mark.parametrize("detected", [False, True], ids=["failed", "unanchored"])
def test_dotboard_joint_reports_excluded_view(tmp_path, detected):
    """A view outside the global grid is reported as (camera, 1-based frame) and persisted.

    ``failed``: the detection failed (and so carries no anchor). ``unanchored``: the view
    DETECTED but has no anchor click, so it never enters the grid either. Exclusion is read
    from the grid, not from ``det.success``; both must report the same frame.
    """
    detections, _truth, pixel_of = _make_dataset(0)
    spec = _drop_anchor(_spec(pixel_of), _DROP_CAM, _DROP_VIEW)
    if not detected:
        detections[_DROP_CAM][_DROP_VIEW] = _failed_dotboard()
    else:
        assert detections[_DROP_CAM][_DROP_VIEW].success
    cams = sorted(detections)

    res = run_joint_from_spec(
        detections,
        _image_size_by_cam(detections),
        source=tmp_path,
        board="dotboard",
        model_type="pinhole",
        spacing_mm=SPACING,
        dt=1.0,
        datum_camera=1,
        datum_view=0,
        spec=spec,
        cameras=cams,
        n_views=_N_VIEWS,
    )

    assert res.excluded_views == [(_DROP_CAM, _DROP_VIEW + 1)]
    assert res.converged and res.rms_px < 1.0  # the rest of the rig still solves
    jr = load_joint(res.paths[0])
    assert parse_excluded_views(jr.board_meta["excluded_views"]) == [
        (_DROP_CAM, _DROP_VIEW + 1)
    ]


def test_dotboard_joint_polynomial_excluded_view_is_per_camera(tmp_path):
    """Each per-camera polynomial record carries only its own camera's excluded frames."""
    detections, _truth, pixel_of = _make_dataset(0)
    spec = _drop_anchor(_spec(pixel_of), _DROP_CAM, _DROP_VIEW)
    detections[_DROP_CAM][_DROP_VIEW] = _failed_dotboard()
    cams = sorted(detections)

    res = run_joint_from_spec(
        detections,
        _image_size_by_cam(detections),
        source=tmp_path,
        board="dotboard",
        model_type="polynomial",
        spacing_mm=SPACING,
        dt=1.0,
        datum_camera=1,
        datum_view=0,
        spec=spec,
        cameras=cams,
        n_views=_N_VIEWS,
    )

    assert res.excluded_views == [(_DROP_CAM, _DROP_VIEW + 1)]
    for cam, p in zip(res.cameras, res.paths):
        stored = parse_excluded_views(load_mono(p).board_meta["excluded_views"])
        assert stored == ([(_DROP_CAM, _DROP_VIEW + 1)] if cam == _DROP_CAM else [])


@pytest.mark.parametrize("views", [[], [(2, 5)], [(2, 5), (3, 1)], [(10, 12)]], ids=str)
def test_excluded_views_format_round_trip(views):
    """format -> parse is the identity, including the empty list."""
    assert parse_excluded_views(format_excluded_views(views)) == views


def test_excluded_views_format_is_cam_colon_frame():
    assert format_excluded_views([(2, 5), (3, 1)]) == "2:5,3:1"
    assert format_excluded_views([]) == ""


def test_excluded_views_empty_char_array_parses_as_none():
    """``savemat`` stores ``""`` as an empty char array; it loads back as a size-0 <U array."""
    assert parse_excluded_views(np.array([], dtype="<U1")) == []


@pytest.mark.parametrize("bad", ["2", "2:x", "2:5,", ":5", 3, np.zeros(2)])
def test_excluded_views_malformed_raises(bad):
    with pytest.raises(ValueError, match="excluded_views"):
        parse_excluded_views(bad)
