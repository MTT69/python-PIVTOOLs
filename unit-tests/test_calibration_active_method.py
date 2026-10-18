"""``calibration.active`` parsed in one place: ``runio.parse_active_method``.

Calibration routes and CLI commands fall back to the active method's board when the request /
command line names none. The active id is the GUI tab id ("stereo_charuco"), not a board name,
so it must be split into (board, stereo) by the inverse of ``active_method_name`` — never used
verbatim as a board, and never guessed when the key is missing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from flask import Flask

import pivtools_gui.calibration.app.views as V
from pivtools_cli import calibration_cli as c2
from pivtools_gui.calibration.runio import active_method_name, parse_active_method
from pivtools_gui.errors import register_json_error_handlers

# Every id the GUI "Set as Active" buttons write (gui/src/components/setup/Calibration.tsx),
# with the (board, stereo) it stands for.
ACTIVE_IDS = {
    "charuco": ("charuco", False),
    "dotboard": ("dotboard", False),
    "scale_factor": ("scale_factor", False),
    "stereo_charuco": ("charuco", True),
    "stereo_dotboard": ("dotboard", True),
    "stereo_stepped": ("stepped", True),
}


@pytest.mark.parametrize("active", sorted(ACTIVE_IDS))
def test_parse_round_trips_with_active_method_name(active):
    board, stereo = parse_active_method(active)
    assert (board, stereo) == ACTIVE_IDS[active]
    assert active_method_name(board, stereo) == active


@pytest.mark.parametrize("active", ["stepped", "stereo_scale_factor", "Charuco", "x"])
def test_unknown_active_raises(active):
    with pytest.raises(ValueError, match="calibration.active"):
        parse_active_method(active)


@pytest.mark.parametrize("active", [None, "", "   "])
def test_missing_active_raises(active):
    with pytest.raises(ValueError, match="calibration.active is required"):
        parse_active_method(active)


# ---------------------------------------------------------------------------
# A route with neither a request board nor calibration.active -> visible error
# ---------------------------------------------------------------------------


class _NoActiveConfig:
    """A config whose calibration block has no ``active`` key."""

    calibration: dict = {"calibration_sources": []}
    camera_count = 1


def _backend_client():
    app = Flask(__name__)
    register_json_error_handlers(app)
    app.register_blueprint(V.calibration_bp, url_prefix="/backend")
    return app.test_client()


def test_model_route_without_board_or_active_reports_error(monkeypatch):
    monkeypatch.setattr(V, "get_config", lambda: _NoActiveConfig())
    response = _backend_client().get("/backend/calibration/model?camera=1")
    data = response.get_json()
    assert data["success"] is False
    assert "calibration.active is required" in data["error"]


def test_model_route_takes_stereo_from_active_when_request_is_silent(
    monkeypatch, tmp_path
):
    """No board and no stereo flag: both come from the active id, so a stereo active
    looks up the stereo model dir of its board, not a mono dir named 'stereo_charuco'.
    """
    seen = {}

    class _Cfg:
        calibration = {"active": "stereo_charuco"}
        camera_count = 2

        def get_calibration_source(self, idx):
            return tmp_path

    def _stereo_dir(source, cam1, cam2, board):
        seen["stereo"] = (cam1, cam2, board)
        return tmp_path / "none"

    monkeypatch.setattr(V, "get_config", lambda: _Cfg())
    monkeypatch.setattr(V.rec, "stereo_model_dir_for_source", _stereo_dir)
    data = _backend_client().get("/backend/calibration/model").get_json()
    assert data["exists"] is False
    assert seen["stereo"] == (1, 2, "charuco")


# ---------------------------------------------------------------------------
# CLI: no --board and no calibration.active -> raises before touching files
# ---------------------------------------------------------------------------


def test_cli_without_board_or_active_raises(monkeypatch):
    monkeypatch.setattr(c2, "get_config", lambda: _NoActiveConfig())
    with pytest.raises(ValueError, match="calibration.active is required"):
        c2.detect_mono_command(SimpleNamespace(board=None, source=None))
