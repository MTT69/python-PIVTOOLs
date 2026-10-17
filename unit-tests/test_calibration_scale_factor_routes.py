"""Image-less scale-factor generate: the GUI route and the CLI command.

Drives the REAL handlers against a tmp workspace with ``get_config`` mocked. The
scale-factor map needs no image: the GUI's "Use calibration images" toggle sends an
explicit ``use_image: false`` and the CLI takes ``--no-image``; both must build and
save the model without reading a frame and without an existing model, and skip the
proof figure. Without either flag the image path must still be the one taken —
image-less is opt-in, not a silent rescue on a failed image load.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest
from flask import Flask

import pivtools_cli.calibration_cli as c2
import pivtools_gui.calibration.app.views as V
from pivtools_gui.calibration import record as rec
from pivtools_gui.calibration.pipeline import build_scale_factor_record


class _FakeConfig:
    camera_count = 1

    def __init__(self, base):
        self._base = base
        self.calibration = {}

    def get_calibration_source(self, idx):
        return self._base


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "get_config", lambda: _FakeConfig(tmp_path))
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    return app.test_client(), tmp_path


def _never_read_image(*args, **kwargs):
    raise AssertionError("image must not be read in image-less generate")


class _FrameLoader:
    """Stand-in frame reader that counts its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return np.zeros((300, 400), np.uint8)


_GENERATE_BODY = {
    "source_path_idx": 0,
    "camera": 1,
    "px_per_mm": 4.0,
    "origin_px": [30.0, 40.0],
    "dt": 0.5,
}


def test_generate_no_image_needs_no_existing_model(env, monkeypatch):
    """First-time image-less generate: no frame read, no prior model, no figure."""
    client, base = env
    monkeypatch.setattr(V, "read_calibration_image", _never_read_image)
    model_dir = rec.mono_model_dir_for_source(base, 1, "scale_factor")
    assert not model_dir.exists()

    r = client.post(
        "/calibration/scale_factor/generate",
        json={**_GENERATE_BODY, "use_image": False},
    )
    data = r.get_json()
    assert data["success"] is True, data.get("error")
    assert data["figures"] == []
    assert not (model_dir.parent / "figures").exists()

    m = client.get(
        "/calibration/model",
        query_string={"board": "scale_factor", "camera": 1, "source_path_idx": 0},
    ).get_json()
    assert m["exists"] is True and m["model_type"] == "scale_factor"
    assert m["px_per_mm"] == pytest.approx(4.0)
    assert list(m["origin_px"]) == pytest.approx([30.0, 40.0])
    assert m["dt"] == pytest.approx(0.5)


def test_generate_no_image_overwrites_existing_model(env, monkeypatch):
    client, base = env
    monkeypatch.setattr(V, "read_calibration_image", _never_read_image)
    model_dir = rec.mono_model_dir_for_source(base, 1, "scale_factor")
    rec.save_mono(
        build_scale_factor_record(
            camera=1, origin_px=(10.0, 20.0), px_per_mm=2.0, dt=1.0
        ),
        model_dir,
    )

    r = client.post(
        "/calibration/scale_factor/generate",
        json={**_GENERATE_BODY, "use_image": False},
    )
    assert r.get_json()["success"] is True

    record = rec.load_mono(model_dir, "scale_factor")
    assert 1.0 / record.camera_model.mm_per_pixel == pytest.approx(4.0)
    assert record.board_meta["dt"] == pytest.approx(0.5)


def test_generate_default_still_loads_image(env, monkeypatch):
    """No ``use_image`` key -> the image path runs; image-less is never silent."""
    client, _ = env
    loader = _FrameLoader()
    monkeypatch.setattr(V, "read_calibration_image", loader)

    r = client.post(
        "/calibration/scale_factor/generate",
        json={**_GENERATE_BODY, "no_figures": True},
    )
    assert r.get_json()["success"] is True
    assert loader.calls == 1


def test_generate_default_image_failure_is_not_rescued(env, monkeypatch):
    """A failed frame read fails the request; it never falls back to image-less."""
    client, base = env

    def _missing(*args, **kwargs):
        raise FileNotFoundError("no calibration image")

    monkeypatch.setattr(V, "read_calibration_image", _missing)

    data = client.post(
        "/calibration/scale_factor/generate", json=_GENERATE_BODY
    ).get_json()
    assert data["success"] is False
    assert "no calibration image" in data["error"]
    assert not rec.mono_model_dir_for_source(base, 1, "scale_factor").exists()


def test_record_with_removed_image_size_fields_still_loads(tmp_path, monkeypatch):
    """Records written before ``image_size`` was dropped carry two extra fields.

    The reader does not read them, so such a record loads unchanged — no contract
    bump, no re-generate.
    """
    current_writer = rec._scale_factor_to_dict

    def _pre_change_writer(sf):
        return {**current_writer(sf), "image_width": 1600, "image_height": 1200}

    monkeypatch.setattr(rec, "_scale_factor_to_dict", _pre_change_writer)
    path = rec.save_mono(
        build_scale_factor_record(
            camera=1, origin_px=(10.0, 20.0), px_per_mm=2.0, dt=1.0
        ),
        tmp_path / "model",
    )

    record = rec.load_mono(path, "scale_factor")
    assert record.camera_model.mm_per_pixel == pytest.approx(0.5)
    assert list(record.camera_model.origin_px) == pytest.approx([10.0, 20.0])


def _cli_args(source, **overrides):
    base = dict(
        source=str(source),
        camera=1,
        image_format=None,
        frame=None,
        px_per_mm=4.0,
        dt=0.5,
        origin=[30.0, 40.0],
        origin_mm=None,
        x_dir="right",
        y_dir="up",
        swap=False,
        no_figures=False,
        no_image=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cli_no_image_skips_frame_load(tmp_path, monkeypatch):
    monkeypatch.setattr(c2, "get_config", lambda: _FakeConfig(tmp_path))
    monkeypatch.setattr(c2, "_load_one", _never_read_image)

    path = c2.scale_factor_command(_cli_args(tmp_path, no_image=True))

    record = rec.load_mono(path, "scale_factor")
    assert 1.0 / record.camera_model.mm_per_pixel == pytest.approx(4.0)
    assert not (path.parent.parent / "figures").exists()


def test_cli_default_still_loads_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(c2, "get_config", lambda: _FakeConfig(tmp_path))
    loader = _FrameLoader()
    monkeypatch.setattr(c2, "_load_one", loader)

    # image_format has no silent default — the frame-loading path requires it
    # from --image-format or the sidecar.
    path = c2.scale_factor_command(
        _cli_args(tmp_path, no_figures=True, image_format="calib%05d.png")
    )

    assert loader.calls == 1
    assert rec.load_mono(path, "scale_factor").camera_model.col_sign == 1


# ---- dt must be finite and > 0 (velocity ~ 1/dt) -------------------------------------


@pytest.mark.parametrize("bad", [0, -1e-3, float("nan"), float("inf")])
def test_resolve_dt_rejects_non_positive_explicit(bad):
    from pivtools_gui.calibration import runio

    with pytest.raises(ValueError, match="dt"):
        runio.resolve_dt(bad, None)


@pytest.mark.parametrize("bad", [0.0, -2e-4, float("nan")])
def test_resolve_dt_rejects_non_positive_stamped(bad):
    from pivtools_gui.calibration import runio

    with pytest.raises(ValueError, match="model-stamped"):
        runio.resolve_dt(None, bad)


def test_resolve_dt_accepts_positive():
    from pivtools_gui.calibration import runio

    assert runio.resolve_dt(None, 1e-4) == 1e-4
    assert runio.resolve_dt(2e-4, 1e-4) == 2e-4


@pytest.mark.parametrize("bad", [0, -1e-3, float("nan")])
def test_build_scale_factor_record_rejects_bad_dt(bad):
    with pytest.raises(ValueError, match="dt"):
        build_scale_factor_record(camera=1, origin_px=(0.0, 0.0), px_per_mm=4.0, dt=bad)


def test_generate_route_rejects_zero_dt(env):
    client, base = env
    data = client.post(
        "/calibration/scale_factor/generate",
        json={**_GENERATE_BODY, "use_image": False, "dt": 0},
    ).get_json()
    assert data["success"] is False
    assert "dt" in data["error"]
    assert not rec.mono_model_dir_for_source(base, 1, "scale_factor").exists()


def test_cli_scale_factor_rejects_zero_dt(tmp_path, monkeypatch):
    monkeypatch.setattr(c2, "get_config", lambda: _FakeConfig(tmp_path))
    monkeypatch.setattr(c2, "_load_one", _never_read_image)
    with pytest.raises(SystemExit, match="dt"):
        c2.scale_factor_command(_cli_args(tmp_path, no_image=True, dt=0.0))


def test_apply_route_rejects_zero_dt_before_job_starts(env, monkeypatch):
    """Request dt=0 -> success False, and no job is created."""
    import types

    client, base = env
    full_cfg = types.SimpleNamespace(vector_format="%05d.mat")
    monkeypatch.setattr(V, "get_config", lambda: full_cfg)
    monkeypatch.setattr(V, "_cfg", lambda: {"active": "scale_factor"})
    monkeypatch.setattr(V, "_source_path", lambda idx: base)
    unit = {
        "stereo": False,
        "record": types.SimpleNamespace(board_meta={"dt": 1e-4}),
        "label": "u",
        "out": base / "out",
    }
    monkeypatch.setattr(V, "_apply_units", lambda *a, **k: [unit])

    def _no_job(*a, **k):
        raise AssertionError("job must not start with an invalid dt")

    monkeypatch.setattr(V.job_manager, "create_job", _no_job)

    data = client.post(
        "/calibration/apply", json={"board": "scale_factor", "dt": 0}
    ).get_json()
    assert data["success"] is False
    assert "dt" in data["error"]
