"""Calibration CLI parity with the GUI routes (Calibration P4 brief 1c, items 7-9).

- 8(a) the CLI never reuses an existing model: every detect re-solves, as the GUI does.
- 8(b) the fit knobs come from ONE reader, ``calibration_settings.fit_knobs``, with
  ``use_release_object`` defaulting False for mono and stereo alike (GUI parity).
- 8(c) ``apply-calibration`` with a stereo ``calibration.active`` runs the stereo apply.
- 8(d) ``calibration.source`` is not read: ``calibration_sources[source_idx]`` only.
- 8(e) ``_load_clicks`` keeps ``origin_mm``.
- 8(f) the joint detections sidecar: OSError warns (persist skipped), ValueError raises.
- 8(g) zero planned apply units exit with "no datasets/cameras configured".
- 8(h) ``--camera`` with ``--all-paths`` is refused.
- 8(i) ``detect-stereo --model-type`` other than pinhole is refused.
- 8(j) dead helpers are gone.
- 9   ``main`` reports ValueError / FileNotFoundError / InputsCorrupt as one line and
  exit 1; the traceback only under ``--debug`` / ``PIVTOOLS_DEBUG=1``.

Usage:
    pytest unit-tests/test_calibration_cli_parity.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pivtools_cli.calibration_cli as cli
from pivtools_cli import cli as top_cli
from pivtools_core import calibration_settings as cs
from pivtools_gui.calibration.inputs_store import InputsCorrupt


class _Stop(Exception):
    """Raised by a stub to end a command once the behaviour under test is seen."""


class _FakeConfig:
    """The Config surface the calibration CLI commands read."""

    def __init__(self, source: Path, active: str = "charuco", camera_count: int = 2):
        self._source = source
        self.calibration = {
            "calibration_sources": [str(source)],
            "source_idx": 0,
            "active": active,
        }
        self.camera_count = camera_count
        self.camera_numbers = list(range(1, camera_count + 1))
        self.vector_format = ["%05d.mat"]

    def get_calibration_source(self, idx: int) -> Path:
        assert idx == 0
        return self._source


def _charuco_source(tmp_path: Path) -> Path:
    src = tmp_path / "calib"
    src.mkdir()
    cs.save_settings(
        src,
        {
            "image": {
                "image_format": "c%05d.tif",
                "image_type": "standard",
                "n_views": 2,
            },
            "rig": {"dt": 1e-3},
            "methods": {
                "charuco": {"squares_h": 5, "squares_v": 4, "square_size": 0.01}
            },
        },
    )
    return src


@pytest.fixture
def source(tmp_path, monkeypatch):
    src = _charuco_source(tmp_path)
    monkeypatch.setattr(cli, "get_config", lambda: _FakeConfig(src))
    return src


def _detect_args(**over) -> SimpleNamespace:
    base = dict(
        board="charuco",
        source=None,
        camera=None,
        camera_pair=None,
        image_format=None,
        n_views=None,
        distortion=None,
        model_type=None,
        world_frame=None,
        no_figures=True,
        force=False,
        dt=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _stub_loading(monkeypatch):
    """No image I/O and no real detection: the solve's inputs are not under test."""
    monkeypatch.setattr(
        cli, "_load_views", lambda *a, **k: [np.zeros((8, 8), np.uint8)] * 2
    )

    class _Det:
        def detect(self, _img):
            return SimpleNamespace(success=False)

    monkeypatch.setattr(cli, "_build_detector", lambda board, params: _Det())


# ---------------------------------------------------------------------------
# 8(a) always re-solve
# ---------------------------------------------------------------------------


def test_detect_mono_resolves_even_when_a_model_exists(source, monkeypatch):
    _stub_loading(monkeypatch)
    monkeypatch.setattr(
        cli.rec, "resolve_mono_path", lambda *a, **k: source / "existing.mat"
    )

    class _Cal:
        def __init__(self, **kwargs):
            raise _Stop(kwargs)

    monkeypatch.setattr(cli, "Calibrator", _Cal)
    with pytest.raises(_Stop):
        cli.detect_mono_command(_detect_args(camera=1))


def test_detect_stereo_resolves_even_when_a_model_exists(source, monkeypatch):
    _stub_loading(monkeypatch)
    monkeypatch.setattr(
        cli.rec, "resolve_stereo_path", lambda *a, **k: source / "existing.mat"
    )

    class _Cal:
        def __init__(self, **kwargs):
            raise _Stop(kwargs)

    monkeypatch.setattr(cli, "StereoCalibrator", _Cal)
    with pytest.raises(_Stop):
        cli.detect_stereo_command(_detect_args())


def test_reuse_helpers_removed():
    assert not hasattr(cli, "_existing_mono_path")
    assert not hasattr(cli, "_reusable_stereo")


# ---------------------------------------------------------------------------
# 8(b) one fit-knob reader, use_release_object False by default
# ---------------------------------------------------------------------------


def test_fit_knobs_defaults():
    knobs = cs.fit_knobs(cs.default_settings(), "charuco")
    assert knobs == {
        "distortion_model": "standard",
        "fix_aspect_ratio": True,
        "fix_k2": False,
        "use_release_object": False,
    }


def test_fit_knobs_reads_fit_and_board_blocks():
    settings = cs.default_settings()
    settings["fit"].update(
        distortion_model="rational", fix_aspect_ratio=False, use_release_object=True
    )
    settings["methods"]["dotboard"]["fix_k2"] = True
    assert cs.fit_knobs(settings, "dotboard") == {
        "distortion_model": "rational",
        "fix_aspect_ratio": False,
        "fix_k2": True,
        "use_release_object": True,
    }
    # fix_k2 is per board: another board's flag does not leak.
    assert cs.fit_knobs(settings, "charuco")["fix_k2"] is False


@pytest.mark.parametrize(
    "command, cal_attr",
    [
        ("detect_mono_command", "Calibrator"),
        ("detect_stereo_command", "StereoCalibrator"),
    ],
)
def test_cli_use_release_object_defaults_false(source, monkeypatch, command, cal_attr):
    """Stereo used to default True on the CLI while the GUI route defaults False."""
    _stub_loading(monkeypatch)
    seen = {}

    class _Cal:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            raise _Stop

    monkeypatch.setattr(cli, cal_attr, _Cal)
    with pytest.raises(_Stop):
        getattr(cli, command)(_detect_args(camera=1))
    assert seen["use_release_object"] is False
    assert seen["fix_k2"] is False
    assert seen["fix_aspect_ratio"] is True
    assert seen["distortion_model"].value == "standard"


def test_cli_distortion_flag_overrides_sidecar(source, monkeypatch):
    _stub_loading(monkeypatch)
    seen = {}

    class _Cal:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            raise _Stop

    monkeypatch.setattr(cli, "Calibrator", _Cal)
    with pytest.raises(_Stop):
        cli.detect_mono_command(_detect_args(camera=1, distortion="rational"))
    assert seen["distortion_model"].value == "rational"


# ---------------------------------------------------------------------------
# 8(c) apply-calibration honours a stereo active method
# ---------------------------------------------------------------------------


def _apply_args(**over) -> SimpleNamespace:
    base = dict(
        board=None,
        source=None,
        camera=None,
        dt=None,
        z_world=None,
        tilt_x=None,
        tilt_y=None,
        uncalibrated_dir=None,
        calibrated_dir=None,
        all_paths=True,
        type_name=None,
        model_type=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_apply_calibration_dispatches_stereo_active(tmp_path, monkeypatch):
    src = _charuco_source(tmp_path)
    monkeypatch.setattr(
        cli, "get_config", lambda: _FakeConfig(src, active="stereo_charuco")
    )
    seen = {}

    def _stereo(args):
        seen["args"] = args
        return 7

    monkeypatch.setattr(cli, "apply_stereo_command", _stereo)
    assert cli.apply_calibration_command(_apply_args()) == 7
    stereo_args = seen["args"]
    assert stereo_args.board == "charuco"
    assert stereo_args.all_paths is True
    assert stereo_args.camera_pair is None
    assert stereo_args.interpolator is None


def test_apply_calibration_stereo_active_refuses_explicit_mono_dirs(
    tmp_path, monkeypatch
):
    src = _charuco_source(tmp_path)
    monkeypatch.setattr(
        cli, "get_config", lambda: _FakeConfig(src, active="stereo_charuco")
    )
    with pytest.raises(SystemExit, match="apply-stereo"):
        cli.apply_calibration_command(
            _apply_args(all_paths=False, uncalibrated_dir="u", calibrated_dir="c")
        )


def test_apply_calibration_explicit_board_stays_mono(source, monkeypatch):
    """An explicit --board names a mono apply, as an explicit GUI board does."""
    monkeypatch.setattr(
        cli, "apply_stereo_command", lambda a: pytest.fail("dispatched to stereo")
    )

    def _plan(*a, **k):
        assert a[3] is False  # stereo flag passed to the planner
        raise _Stop

    monkeypatch.setattr(cli.runio, "plan_apply_units", _plan)
    with pytest.raises(_Stop):
        cli.apply_calibration_command(_apply_args(board="charuco"))


# ---------------------------------------------------------------------------
# 8(d) calibration.source is not read
# ---------------------------------------------------------------------------


def test_source_ignores_calibration_source_key(tmp_path, monkeypatch):
    src = _charuco_source(tmp_path)
    monkeypatch.setattr(cli, "get_config", lambda: _FakeConfig(src))
    cfg = {"source": str(tmp_path / "elsewhere"), "source_idx": 0}
    assert cli._source(cfg) == src
    assert cli._source(cfg, override=str(tmp_path / "flag")) == tmp_path / "flag"


# ---------------------------------------------------------------------------
# 8(e) origin_mm survives _load_clicks
# ---------------------------------------------------------------------------


def test_load_clicks_keeps_origin_mm(tmp_path):
    import json

    path = tmp_path / "clicks.json"
    path.write_text(
        json.dumps(
            {
                "origin": [10, 20],
                "x_axis": [30, 20],
                "y_axis": [10, 5],
                "origin_mm": [15.0, -7.5],
            }
        )
    )
    clicks = cli._load_clicks(str(path))
    np.testing.assert_allclose(clicks["origin_mm"], [15.0, -7.5])


def test_load_clicks_without_origin_mm(tmp_path):
    import json

    path = tmp_path / "clicks.json"
    path.write_text(json.dumps({"origin": [1, 2], "x_axis": [3, 2], "y_axis": [1, 0]}))
    assert "origin_mm" not in cli._load_clicks(str(path))


# ---------------------------------------------------------------------------
# 8(f) joint detections sidecar persistence
# ---------------------------------------------------------------------------


def _persist(monkeypatch, exc):
    def _boom(*a, **k):
        raise exc

    monkeypatch.setattr(cli, "save_inputs", _boom)
    return cli._persist_joint_detections(
        Path("/nowhere"),
        board="charuco",
        detections={},
        image_size_by_cam={},
        det_key="k",
        board_params={},
    )


def test_joint_persist_oserror_warns(monkeypatch, capsys):
    skipped = _persist(monkeypatch, OSError("read-only volume"))
    assert skipped == "sidecar not written: read-only volume"
    assert "sidecar not written: read-only volume" in capsys.readouterr().out


def test_joint_persist_valueerror_raises(monkeypatch):
    with pytest.raises(ValueError, match="bad shape"):
        _persist(monkeypatch, ValueError("bad shape"))


def test_joint_persist_success_returns_none(monkeypatch):
    monkeypatch.setattr(cli, "save_inputs", lambda *a, **k: None)
    assert (
        cli._persist_joint_detections(
            Path("/nowhere"),
            board="charuco",
            detections={},
            image_size_by_cam={},
            det_key="k",
            board_params={},
        )
        is None
    )


# ---------------------------------------------------------------------------
# 8(g) zero planned units, 8(h) --camera with --all-paths
# ---------------------------------------------------------------------------


def test_apply_calibration_zero_units_exits(source, monkeypatch):
    monkeypatch.setattr(cli.runio, "plan_apply_units", lambda *a, **k: [])
    with pytest.raises(SystemExit, match="no datasets/cameras configured"):
        cli.apply_calibration_command(_apply_args(board="charuco"))


def test_apply_stereo_zero_units_exits(source, monkeypatch):
    monkeypatch.setattr(cli.runio, "plan_apply_units", lambda *a, **k: [])
    args = _apply_args(
        board="charuco",
        camera_pair=None,
        interpolator=None,
        uncalibrated_dir_cam1=None,
        uncalibrated_dir_cam2=None,
    )
    with pytest.raises(SystemExit, match="no datasets/cameras configured"):
        cli.apply_stereo_command(args)


def test_camera_with_all_paths_refused(source, monkeypatch):
    monkeypatch.setattr(
        cli.runio, "plan_apply_units", lambda *a, **k: pytest.fail("planned")
    )
    with pytest.raises(
        SystemExit, match="--camera needs an explicit --uncalibrated-dir"
    ):
        cli.apply_calibration_command(_apply_args(board="charuco", camera=2))


# ---------------------------------------------------------------------------
# 8(i) detect-stereo is pinhole-only
# ---------------------------------------------------------------------------


def test_detect_stereo_refuses_polynomial(source, monkeypatch):
    _stub_loading(monkeypatch)
    with pytest.raises(SystemExit, match="pinhole"):
        cli.detect_stereo_command(_detect_args(model_type="polynomial"))


# ---------------------------------------------------------------------------
# 8(j) dead helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["_count_views", "_read_json_maybe", "_parse_fiducials"]
)
def test_dead_helpers_removed(name):
    assert not hasattr(cli, name)


# ---------------------------------------------------------------------------
# 9 CLI honesty: one-line errors, traceback only under debug
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bad value"),
        FileNotFoundError("no file"),
        InputsCorrupt(Path("/m"), Path("/m/inputs.mat"), OSError("torn")),
    ],
)
def test_main_reports_error_line(monkeypatch, capsys, exc):
    def _raise(_args):
        raise exc

    monkeypatch.delenv("PIVTOOLS_DEBUG", raising=False)
    monkeypatch.setattr(cli, "init_settings_command", _raise)
    monkeypatch.setattr(sys, "argv", ["pivtools-cli", "init-settings"])
    with pytest.raises(SystemExit) as info:
        top_cli.main()
    assert info.value.code == 1
    err = capsys.readouterr().err
    assert err.strip() == f"error: {exc}"
    assert "Traceback" not in err


@pytest.mark.parametrize("how", ["flag", "env"])
def test_main_debug_keeps_traceback(monkeypatch, how):
    def _raise(_args):
        raise ValueError("bad value")

    monkeypatch.setattr(cli, "init_settings_command", _raise)
    argv = ["pivtools-cli", "init-settings"]
    if how == "flag":
        monkeypatch.delenv("PIVTOOLS_DEBUG", raising=False)
        argv = ["pivtools-cli", "--debug", "init-settings"]
    else:
        monkeypatch.setenv("PIVTOOLS_DEBUG", "1")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="bad value"):
        top_cli.main()


# ---------------------------------------------------------------------------
# Stereo sidecar coords: the per-camera shape the GUI reads
# ---------------------------------------------------------------------------


def _dotboard_stereo_source(tmp_path: Path) -> Path:
    src = tmp_path / "calib"
    src.mkdir()
    cs.save_settings(
        src,
        {
            "image": {
                "image_format": "c%05d.tif",
                "image_type": "standard",
                "n_views": 2,
            },
            "rig": {"dt": 1e-3, "camera_pair": [1, 2]},
            "methods": {"dotboard": {"dot_spacing_mm": 5.0}},
        },
    )
    return src


def _clicks_file(tmp_path: Path, name: str, origin, origin_mm=None) -> str:
    import json

    data = {
        "origin": list(origin),
        "x_axis": [origin[0] + 40.0, origin[1]],
        "y_axis": [origin[0], origin[1] - 40.0],
    }
    if origin_mm is not None:
        data["origin_mm"] = list(origin_mm)
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def _stub_stereo_solve(monkeypatch, seen: list):
    """Real detections and a real inputs.mat; only the solve and record save are stubbed."""
    from pivtools_gui.calibration.detection.base import DetectionResult

    monkeypatch.setattr(
        cli, "_load_views", lambda *a, **k: [np.zeros((8, 8), np.uint8)] * 2
    )

    class _Det:
        def detect(self, _img):
            return DetectionResult(
                success=False,
                board_type="dotboard",
                image_points=np.zeros((0, 2)),
                board_local_points=np.zeros((0, 3)),
            )

    monkeypatch.setattr(cli, "_build_detector", lambda board, params: _Det())
    model = SimpleNamespace(image_size=(8, 8), rms=0.1)
    record = SimpleNamespace(
        model1=model,
        model2=model,
        R_stereo=np.eye(3),
        T_stereo=np.zeros(3),
        board_meta={},
    )

    class _Cal:
        def __init__(self, **kwargs):
            pass

        def run_stereo(self, *a, **kwargs):
            seen.append(kwargs)
            return record

    monkeypatch.setattr(cli, "StereoCalibrator", _Cal)
    monkeypatch.setattr(cli.rec, "save_stereo", lambda r, d: Path(d) / "stub.mat")


def test_cli_stereo_sidecar_round_trips_through_gui(tmp_path, monkeypatch):
    import pivtools_gui.calibration.app.views as V

    src = _dotboard_stereo_source(tmp_path)
    monkeypatch.setattr(cli, "get_config", lambda: _FakeConfig(src))
    seen: list = []
    _stub_stereo_solve(monkeypatch, seen)
    args = _detect_args(
        board="dotboard",
        world_frame=_clicks_file(tmp_path, "c1.json", (100, 200), (15.0, -5.0)),
        world_frame_cam2=_clicks_file(tmp_path, "c2.json", (300, 210)),
    )
    cli.detect_stereo_command(args)

    mdir = cli.rec.stereo_model_dir_for_source(src, 1, 2, "dotboard")
    cam2 = V._inputs_world_frame(mdir, camera=2)
    assert cam2["origin"] == [300.0, 210.0]
    assert cam2["origin_mm"] == [15.0, -5.0]  # the shared frame's, from camera 1
    cam1 = V._inputs_world_frame(mdir, camera=1)
    assert cam1["origin"] == [100.0, 200.0]

    kw = seen[0]
    np.testing.assert_allclose(kw["clicks"]["origin"], [100, 200])
    np.testing.assert_allclose(kw["clicks2"]["origin"], [300, 210])
    assert kw["origin_mm"] == (15.0, -5.0)

    # A re-run with no click files reads both cameras' picks back from the sidecar.
    cli.detect_stereo_command(_detect_args(board="dotboard"))
    kw2 = seen[1]
    np.testing.assert_allclose(kw2["clicks"]["origin"], [100, 200])
    np.testing.assert_allclose(kw2["clicks2"]["origin"], [300, 210])
    assert kw2["origin_mm"] == (15.0, -5.0)


def test_cli_dotboard_stereo_requires_cam2_picks(tmp_path, monkeypatch):
    src = _dotboard_stereo_source(tmp_path)
    monkeypatch.setattr(cli, "get_config", lambda: _FakeConfig(src))
    seen: list = []
    _stub_stereo_solve(monkeypatch, seen)
    args = _detect_args(
        board="dotboard",
        world_frame=_clicks_file(tmp_path, "c1.json", (100, 200), (0.0, 0.0)),
    )
    with pytest.raises(SystemExit, match="--world-frame-cam2"):
        cli.detect_stereo_command(args)
    assert seen == []


def test_main_reports_inputs_read_error(monkeypatch, capsys):
    from pivtools_gui.calibration.inputs_store import InputsReadError

    def _raise(_args):
        raise InputsReadError("inputs.mat could not be read: permission denied")

    monkeypatch.delenv("PIVTOOLS_DEBUG", raising=False)
    monkeypatch.setattr(cli, "init_settings_command", _raise)
    monkeypatch.setattr(sys, "argv", ["pivtools-cli", "init-settings"])
    with pytest.raises(SystemExit) as info:
        top_cli.main()
    assert info.value.code == 1
    err = capsys.readouterr().err
    assert err.strip() == "error: inputs.mat could not be read: permission denied"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
