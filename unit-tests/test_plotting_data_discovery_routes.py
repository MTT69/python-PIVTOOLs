"""Data-discovery routes of the vector viewer: what the Results tab asks on mount.

Drives the REAL route handlers (``check_available_data``, ``check_limits``,
``check_runs``, ``check_all_vars``) against synthetic result trees in a tmp workspace;
only ``get_config`` is doubled. These endpoints had no coverage at all when their
per-frame ``Path.exists()`` sweep was replaced by a single ``os.scandir``
(``pivtools_core.paths.list_vector_files``), so this pins the contract that swap must
preserve:

* ``exists`` / ``frame_count`` per data source, including partial and empty runs,
* variables read from the LOWEST-numbered frame present, not an arbitrary one --
  ``scandir`` returns no particular order, whereas the old ``range(1, n + 1)`` loop
  gave frame order for free,
* non-result ``.mat`` files (``coordinates.mat``, ``mask.mat``) never counted.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io
from flask import Flask

import pivtools_gui.plotting.app.plotting_views as PV
import pivtools_gui.plotting.app.shared_utils as SU

NUM_FRAME_PAIRS = 8
VECTOR_FORMAT = "%05d.mat"


class _FakeConfig:
    """Minimal stand-in covering every attribute the plotting views read."""

    def __init__(self, base):
        self.base_paths = [str(base)]
        self.num_frame_pairs = NUM_FRAME_PAIRS
        self.vector_format = VECTOR_FORMAT
        self.camera_numbers = [1]
        self.is_stereo_setup = False
        self.stereo_pairs = []


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Flask test client with both get_config bindings doubled.

    plotting_views and shared_utils import get_config separately, so patching one
    leaves the other reading the real config.yaml.
    """
    monkeypatch.setattr(PV, "get_config", lambda: _FakeConfig(tmp_path))
    monkeypatch.setattr(SU, "get_config", lambda: _FakeConfig(tmp_path))
    app = Flask(__name__)
    app.register_blueprint(PV.vector_plot_bp, url_prefix="/plot")
    return app.test_client(), tmp_path


def _piv_result(extra_var=None, fill=1.0):
    """A one-run piv_result struct with ux/uy and an optional marker variable."""
    fields = [("ux", object), ("uy", object), ("b_mask", object)]
    if extra_var:
        fields.append((extra_var, object))
    struct = np.empty((1,), dtype=fields)
    grid = np.full((3, 4), fill, dtype=float)
    struct["ux"][0] = grid
    struct["uy"][0] = grid * 2.0
    struct["b_mask"][0] = np.zeros((3, 4), dtype=bool)
    if extra_var:
        struct[extra_var][0] = grid
    return struct


def _write_frames(folder, frames, extra_on=None, fill=1.0):
    """Write ``frames`` result files; ``extra_on`` frames also carry a marker var."""
    folder.mkdir(parents=True, exist_ok=True)
    for f in frames:
        extra = "marker_var" if extra_on and f in extra_on else None
        scipy.io.savemat(
            str(folder / (VECTOR_FORMAT % f)),
            {"piv_result": _piv_result(extra_var=extra, fill=fill + f)},
            do_compression=True,
        )
    return folder


def _uncal_dir(base, cam=1, type_name="instantaneous"):
    return base / "uncalibrated_piv" / str(NUM_FRAME_PAIRS) / f"Cam{cam}" / type_name


def _cal_dir(base, cam=1, type_name="instantaneous"):
    return base / "calibrated_piv" / str(NUM_FRAME_PAIRS) / f"Cam{cam}" / type_name


def _availability(client, base):
    res = client.get(f"/plot/check_available_data?base_path={base}&camera=1")
    assert res.status_code == 200, res.get_data(as_text=True)
    return res.get_json()["available"]


# --------------------------------------------------------------------------
# check_available_data
# --------------------------------------------------------------------------


def test_nothing_produced_reports_no_sources(env):
    client, base = env
    available = _availability(client, base)
    assert not any(src["exists"] for src in available.values())


def test_counts_a_complete_uncalibrated_run(env):
    client, base = env
    _write_frames(_uncal_dir(base), range(1, NUM_FRAME_PAIRS + 1))

    available = _availability(client, base)

    assert available["uncalibrated_instantaneous"]["exists"] is True
    assert available["uncalibrated_instantaneous"]["frame_count"] == NUM_FRAME_PAIRS
    # The calibration step has not run, which is exactly the post-PIV state that used
    # to aim the viewer's whole first request wave at a directory that is not there.
    assert available["calibrated_instantaneous"]["exists"] is False


def test_counts_a_partial_run(env):
    client, base = env
    _write_frames(_uncal_dir(base), [1, 2, 3])

    available = _availability(client, base)

    assert available["uncalibrated_instantaneous"]["frame_count"] == 3


def test_frames_beyond_num_frame_pairs_are_not_counted(env):
    client, base = env
    _write_frames(_uncal_dir(base), range(1, NUM_FRAME_PAIRS + 1))
    scipy.io.savemat(
        str(_uncal_dir(base) / (VECTOR_FORMAT % (NUM_FRAME_PAIRS + 5))),
        {"piv_result": _piv_result()},
        do_compression=True,
    )

    available = _availability(client, base)

    assert available["uncalibrated_instantaneous"]["frame_count"] == NUM_FRAME_PAIRS


def test_non_result_mat_files_are_not_counted(env):
    client, base = env
    folder = _write_frames(_uncal_dir(base), [1, 2])
    for stray in ("coordinates.mat", "mask.mat"):
        scipy.io.savemat(str(folder / stray), {"coordinates": np.zeros((2, 2))})

    available = _availability(client, base)

    assert available["uncalibrated_instantaneous"]["frame_count"] == 2


def test_variables_come_from_the_lowest_frame_present(env):
    """The ordering contract.

    Only frame 3 -- the lowest present -- carries ``marker_var``. Reading any other
    frame would miss it. The old code got frame order from ``range(1, n + 1)``;
    ``os.scandir`` does not, so ``list_vector_files`` has to sort.
    """
    client, base = env
    _write_frames(_uncal_dir(base), [3, 5, 7], extra_on={3})

    variables = _availability(client, base)["uncalibrated_instantaneous"]["variables"]

    assert "marker_var" in variables
    assert "ux" in variables


def test_variables_exclude_a_marker_on_a_later_frame(env):
    """The mirror of the test above: a marker on frame 7 must NOT be reported."""
    client, base = env
    _write_frames(_uncal_dir(base), [3, 5, 7], extra_on={7})

    variables = _availability(client, base)["uncalibrated_instantaneous"]["variables"]

    assert "marker_var" not in variables


def test_calibrated_and_uncalibrated_are_reported_independently(env):
    client, base = env
    _write_frames(_uncal_dir(base), range(1, NUM_FRAME_PAIRS + 1))
    _write_frames(_cal_dir(base), [1, 2])

    available = _availability(client, base)

    assert available["uncalibrated_instantaneous"]["frame_count"] == NUM_FRAME_PAIRS
    assert available["calibrated_instantaneous"]["frame_count"] == 2


# --------------------------------------------------------------------------
# check_limits
# --------------------------------------------------------------------------


def _limits(client, base, uncalibrated=True, var="ux"):
    return client.get(
        f"/plot/check_limits?base_path={base}&camera=1&var={var}"
        f"&is_uncalibrated={'1' if uncalibrated else '0'}"
    )


def test_limits_reports_percentiles_over_the_run(env):
    client, base = env
    _write_frames(_uncal_dir(base), range(1, NUM_FRAME_PAIRS + 1))

    res = _limits(client, base)

    assert res.status_code == 200, res.get_data(as_text=True)
    payload = res.get_json()
    assert payload["success"] is True
    # Frame f is filled with 1.0 + f, so the run spans 2.0 .. 9.0.
    assert payload["min"] == pytest.approx(2.0)
    assert payload["max"] == pytest.approx(9.0)
    assert payload["p5"] <= payload["p95"]


def test_limits_404s_on_an_empty_directory(env):
    """A missing source must 404, not raise - the viewer renders the message."""
    client, base = env
    res = _limits(client, base)

    assert res.status_code == 404
    assert res.get_json()["success"] is False


def test_limits_ignores_stray_mat_files(env):
    client, base = env
    folder = _write_frames(_uncal_dir(base), [1])
    scipy.io.savemat(str(folder / "coordinates.mat"), {"coordinates": np.zeros((2, 2))})

    res = _limits(client, base)

    assert res.status_code == 200
    assert res.get_json()["min"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# check_runs / check_all_vars
# --------------------------------------------------------------------------


def test_runs_reports_the_populated_run(env):
    client, base = env
    _write_frames(_uncal_dir(base), [1])

    res = client.get(
        f"/plot/check_runs?base_path={base}&camera=1&var=ux&is_uncalibrated=1&frame=1"
    )

    assert res.status_code == 200, res.get_data(as_text=True)
    assert res.get_json()["runs"] == [1]


def test_all_vars_groups_instantaneous_variables(env):
    client, base = env
    _write_frames(_uncal_dir(base), [1])

    res = client.get(
        f"/plot/check_all_vars?base_path={base}&camera=1&is_uncalibrated=1&frame=1"
    )

    assert res.status_code == 200, res.get_data(as_text=True)
    payload = res.get_json()
    assert "ux" in payload["instantaneous"]
    assert "uy" in payload["instantaneous"]


def test_all_vars_reads_inst_stats_from_its_lowest_frame(env):
    """instantaneous_stats used to build every path just to take ``[0]``."""
    client, base = env
    _write_frames(_uncal_dir(base), [1])
    stats_dir = (
        base
        / "statistics"
        / "uncalibrated"
        / str(NUM_FRAME_PAIRS)
        / "Cam1"
        / "instantaneous"
        / "instantaneous_stats"
    )
    _write_frames(stats_dir, [4, 6], extra_on={4})

    res = client.get(
        f"/plot/check_all_vars?base_path={base}&camera=1&is_uncalibrated=1&frame=1"
    )

    assert res.status_code == 200, res.get_data(as_text=True)
    assert "marker_var" in res.get_json()["instantaneous_stats"]
