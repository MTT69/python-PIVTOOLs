#!/usr/bin/env python3
"""
test_charuco_failure_hint.py

A failed ChArUco detection says WHY: wrong dictionary, wrong squares count, or no
markers at all.

Background: ``CharucoBoardDetector.detect`` discarded the marker ids and a failure
returned only ``{"n_corners": 0}``, so the GUI showed a bare "board not detected"
whether the dictionary, the board size or the image was wrong.
``diagnose_failure`` (run on the single-frame failure path only) re-detects the
markers and names the likely cause; detect_datum / detect_frame put the hint in
their ``error`` text.

Images come from the repo's synthetic ChArUco generator
(``pivtools_cli/generate_synthetic_charuco.py``): the committed 10x7 DICT_4X4_1000
set, and a fresh render with DICT_5X5_100.

Usage:
    pytest unit-tests/test_charuco_failure_hint.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pivtools_gui.calibration.app.views as V  # noqa: E402
from pivtools_cli.generate_synthetic_charuco import (  # noqa: E402
    SQUARE_SIZE,
    SQUARES_H,
    SQUARES_V,
    generate_charuco_images,
)
from pivtools_cli.synthetic_calibration_common import (  # noqa: E402
    make_camera_matrix,
    make_poses,
)
from pivtools_gui.calibration.detection.charuco import (  # noqa: E402
    GUI_ARUCO_DICTS,
    CharucoBoardDetector,
    CharucoParams,
)

COMMITTED_VIEW = (
    Path(__file__).resolve().parent
    / "synthetic_calibration"
    / "charuco"
    / "calib00001.png"
)
RENDER_PX = 600


def _committed_image() -> np.ndarray:
    img = cv2.imread(str(COMMITTED_VIEW), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"missing synthetic view {COMMITTED_VIEW}"
    return img


@pytest.fixture(scope="module")
def dict5x5_image(tmp_path_factory) -> np.ndarray:
    """One synthetic view of a 10x7 board printed with DICT_5X5_100."""
    out = tmp_path_factory.mktemp("charuco5x5")
    cam = make_camera_matrix(RENDER_PX, RENDER_PX)
    board_w = SQUARES_H * SQUARE_SIZE
    centre = np.array([board_w / 2, SQUARES_V * SQUARE_SIZE / 2, 0.0])
    poses = make_poses(1, centre, cam[0, 0], RENDER_PX, board_w)
    generate_charuco_images(
        out, cam, poses, RENDER_PX, RENDER_PX, dict_id=cv2.aruco.DICT_5X5_100
    )
    img = cv2.imread(str(out / "calib00001.png"), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    return img


def _diagnose(params: CharucoParams, img: np.ndarray) -> dict:
    detector = CharucoBoardDetector(params)
    det = detector.detect(img)
    assert not det.success
    return detector.diagnose_failure(img, det)


def test_gui_dictionary_list_has_twelve_entries():
    assert len(GUI_ARUCO_DICTS) == 12


def test_control_committed_view_detects():
    """Without this, a detector that always fails would satisfy the tests below."""
    assert CharucoBoardDetector(CharucoParams()).detect(_committed_image()).success


def test_wrong_dictionary_names_the_right_one(dict5x5_image):
    diag = _diagnose(CharucoParams(), dict5x5_image)
    assert diag["n_markers"] == 0
    assert diag["marker_ids_max"] is None
    # A 10x7 board uses marker ids 0..34. DICT_5X5_50 holds the same first 50
    # markers as DICT_5X5_100, so the smallest dictionary that holds them all is
    # named; configuring it detects the board.
    assert diag["hint"] == (
        "no markers with DICT_4X4_1000; 35 found with DICT_5X5_50 — check the dictionary"
    )
    # The hint is actionable: the named dictionary detects the board.
    fixed = CharucoBoardDetector(CharucoParams(aruco_dict="DICT_5X5_50"))
    assert fixed.detect(dict5x5_image).success


def test_wrong_squares_count_gives_board_size_hint():
    diag = _diagnose(CharucoParams(squares_h=6, squares_v=5), _committed_image())
    assert diag["n_markers"] > 0
    assert diag["marker_ids_max"] >= 15
    assert diag["hint"] == (
        f"marker IDs up to {diag['marker_ids_max']} found but a 6x5 board has only "
        f"15 markers — check the squares count"
    )


def test_blank_image_found_nothing():
    diag = _diagnose(CharucoParams(), np.full((400, 400), 255, dtype=np.uint8))
    assert diag["n_markers"] == 0
    assert diag["hint"] == "no ArUco markers found (tried 12 dictionaries)"


def test_single_frame_routes_carry_hint(monkeypatch, dict5x5_image):
    """detect_datum and detect_frame both say "board not detected — <hint>"."""
    detector = CharucoBoardDetector(CharucoParams())
    monkeypatch.setattr(
        V,
        "_resolve_board",
        lambda get, override: (None, "charuco", CharucoParams(), detector),
    )
    monkeypatch.setattr(V, "_load_one", lambda *a, **k: dict5x5_image)
    monkeypatch.setattr(V, "_frame_count_or_error", lambda *a, **k: (1, None))
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    client = app.test_client()
    body = {"camera": 1, "source_path_idx": 0, "datum_frame": 1, "frame": 1}

    for route in ("/calibration/detect_datum", "/calibration/detect_frame"):
        out = client.post(route, json=body).get_json()
        assert out["success"] is False, route
        assert out["error"].startswith(
            "board not detected — no markers with DICT_4X4_1000"
        ), route
        assert out["diagnostics"]["hint"] in out["error"], route


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
