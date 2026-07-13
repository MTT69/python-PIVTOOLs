"""POST /calibration/set_datum — the vector viewer's datum/offset coordinate rewrite.

Drives the REAL route handler against synthetic ``coordinates.mat`` files in a tmp
workspace; only ``get_config`` is mocked (base_paths + num_frame_pairs double). The
route was restored from the deleted v1 calibration package (commit 2f96833 removed it
while the GUI kept calling it), so these tests pin the v1 semantics: datum ``(x, y)``
subtracted first, then ``(x_offset, y_offset)`` added, all runs rewritten, stereo ``z``
untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io
from flask import Flask

import pivtools_gui.calibration.app.views as V

_NUM_FRAME_PAIRS = 10


class _FakeConfig:
    def __init__(self, base):
        self.base_paths = [str(base)]
        self.num_frame_pairs = _NUM_FRAME_PAIRS


def _write_coords(path, runs, with_z=False):
    """Write a coordinates.mat with one (x, y[, z]) grid pair per run."""
    fields = [("x", object), ("y", object)] + ([("z", object)] if with_z else [])
    struct = np.empty((len(runs),), dtype=fields)
    for i, (cx, cy) in enumerate(runs):
        struct["x"][i] = cx
        struct["y"][i] = cy
        if with_z:
            struct["z"][i] = np.full_like(cx, 3.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(path), {"coordinates": struct}, do_compression=True)


def _read_coords(path, run):
    mat = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    coords = mat["coordinates"]
    el = (
        coords[run - 1]
        if (isinstance(coords, np.ndarray) and coords.dtype == object)
        else coords
    )
    return el


def _grid(x0=0.0, y0=0.0):
    x, y = np.meshgrid(np.arange(4, dtype=float) + x0, np.arange(3, dtype=float) + y0)
    return x, y


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "get_config", lambda: _FakeConfig(tmp_path))
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    return app.test_client(), tmp_path


def _inst_coords_path(base, cam=1, type_name="instantaneous"):
    return (
        base
        / "calibrated_piv"
        / str(_NUM_FRAME_PAIRS)
        / f"Cam{cam}"
        / type_name
        / "coordinates.mat"
    )


def test_offset_adds_to_all_runs_and_is_additive(env):
    client, base = env
    coords_path = _inst_coords_path(base)
    _write_coords(coords_path, [_grid(), _grid(100.0, 50.0)])

    body = {
        "base_path_idx": 0,
        "camera": 1,
        "run": 1,
        "type_name": "instantaneous",
        "x_offset": 5.0,
        "y_offset": -2.0,
        "merged": 0,
    }
    r = client.post("/calibration/set_datum", json=body)
    assert r.status_code == 200 and r.get_json()["num_runs_updated"] == 2

    gx, gy = _grid()
    for run, (ex, ey) in enumerate([(gx, gy), (*_grid(100.0, 50.0),)], start=1):
        el = _read_coords(coords_path, run)
        np.testing.assert_allclose(np.asarray(el.x), ex + 5.0)
        np.testing.assert_allclose(np.asarray(el.y), ey - 2.0)

    # A second press adds again — offsets are additive, not absolute.
    r = client.post("/calibration/set_datum", json=body)
    assert r.status_code == 200
    el = _read_coords(coords_path, 1)
    np.testing.assert_allclose(np.asarray(el.x), gx + 10.0)
    np.testing.assert_allclose(np.asarray(el.y), gy - 4.0)


def test_datum_subtracts_then_offset_adds(env):
    client, base = env
    coords_path = _inst_coords_path(base)
    gx, gy = _grid()
    _write_coords(coords_path, [(gx, gy)])

    r = client.post(
        "/calibration/set_datum",
        json={
            "base_path_idx": 0,
            "camera": 1,
            "x": 2.0,
            "y": 1.0,
            "x_offset": 0.5,
            "y_offset": 0.0,
        },
    )
    assert r.status_code == 200
    el = _read_coords(coords_path, 1)
    np.testing.assert_allclose(np.asarray(el.x), gx - 2.0 + 0.5)
    np.testing.assert_allclose(np.asarray(el.y), gy - 1.0)


def test_stereo_z_preserved(env):
    client, base = env
    coords_path = (
        base
        / "stereo_calibrated"
        / str(_NUM_FRAME_PAIRS)
        / "Cam1_Cam2"
        / "instantaneous"
        / "coordinates.mat"
    )
    gx, gy = _grid()
    _write_coords(coords_path, [(gx, gy)], with_z=True)

    r = client.post(
        "/calibration/set_datum",
        json={
            "base_path_idx": 0,
            "camera": 1,
            "x_offset": 3.0,
            "y_offset": 3.0,
            "use_stereo": True,
            "camera_pair": [1, 2],
        },
    )
    assert r.status_code == 200
    el = _read_coords(coords_path, 1)
    np.testing.assert_allclose(np.asarray(el.x), gx + 3.0)
    np.testing.assert_allclose(np.asarray(el.z), np.full_like(gx, 3.5))


def test_merged_path_honored(env):
    client, base = env
    coords_path = (
        base
        / "calibrated_piv"
        / str(_NUM_FRAME_PAIRS)
        / "Merged"
        / "instantaneous"
        / "coordinates.mat"
    )
    gx, gy = _grid()
    _write_coords(coords_path, [(gx, gy)])

    r = client.post(
        "/calibration/set_datum",
        json={
            "base_path_idx": 0,
            "camera": 1,
            "x_offset": 1.0,
            "y_offset": 0.0,
            "merged": 1,
        },
    )
    assert r.status_code == 200
    assert "Merged" in r.get_json()["coords_path"]
    el = _read_coords(coords_path, 1)
    np.testing.assert_allclose(np.asarray(el.x), gx + 1.0)


def test_missing_coordinates_is_404(env):
    client, _ = env
    r = client.post(
        "/calibration/set_datum",
        json={
            "base_path_idx": 0,
            "camera": 1,
            "type_name": "ensemble",
            "x_offset": 1.0,
        },
    )
    assert r.status_code == 404
    assert "Coordinates file not found" in r.get_json()["error"]


def test_explicit_base_path_wins_over_idx(env):
    client, base = env
    other = base / "elsewhere"
    coords_path = _inst_coords_path(other)
    gx, gy = _grid()
    _write_coords(coords_path, [(gx, gy)])

    r = client.post(
        "/calibration/set_datum",
        json={
            "base_path": str(other),
            "base_path_idx": 0,
            "camera": 1,
            "x_offset": 2.0,
            "y_offset": 0.0,
        },
    )
    assert r.status_code == 200
    el = _read_coords(coords_path, 1)
    np.testing.assert_allclose(np.asarray(el.x), gx + 2.0)
