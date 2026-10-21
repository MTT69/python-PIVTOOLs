"""Stereo apply job: frames skipped for a missing pass are counted and warned, not silent.

``reconstruct_stereo_run`` skips a frame pair whose reconstructed pass (the final populated
pass of ``coordinates.mat``) is absent from either camera's vector file. The apply route must
carry that count to the GUI (``n_frames_skipped``) and name the unit in ``warnings``. Drives
the real route through the ``chain`` fixture of ``test_calibration_apply_chain`` (planner,
model load, reconstruction and stamping all real; only the config source is faked).
"""

from __future__ import annotations

# ruff: noqa: F811  (the imported ``chain`` pytest fixture is re-bound by the test parameter)
import scipy.io
from fixtures.vector_sets import (
    DEFAULT_PASSES,
    VECTOR_NAME_FORMAT,
    _frame_piv_result,
    write_stereo_vector_sets,
)
from test_calibration_apply_chain import chain  # noqa: F401  (pytest fixture)
from test_calibration_apply_chain import (  # noqa: E402  (sibling-test reuse)
    N_FRAMES,
    STEREO_GRID,
    _uncal,
    _wait,
)
from test_calibration_self_cal import _stereo_record

from pivtools_gui.calibration import record as REC

# 1-based frame whose camera-2 file loses the reconstructed pass.
SKIPPED_FRAME = 2


def test_stereo_apply_reports_skipped_frames(chain):
    client, base, source, _stamps = chain
    REC.save_stereo(
        _stereo_record(), REC.stereo_model_dir_for_source(source, 1, 2, "dotboard")
    )
    write_stereo_vector_sets(
        _uncal(base, 1), _uncal(base, 2), N_FRAMES, grid=STEREO_GRID
    )
    no_final_pass = DEFAULT_PASSES[:-1] + (None,)
    scipy.io.savemat(
        str(_uncal(base, 2) / VECTOR_NAME_FORMAT.format(SKIPPED_FRAME)),
        {"piv_result": _frame_piv_result(SKIPPED_FRAME, no_final_pass, STEREO_GRID)},
    )

    resp = client.post(
        "/calibration/apply",
        json={
            "board": "dotboard",
            "stereo": True,
            "camera_pair": [1, 2],
            "interpolator": "linear",
            "dt": 1e-3,
        },
    ).get_json()
    assert resp["success"] is True, resp
    status = _wait(client, resp["job_id"])

    assert status["status"] == "completed", status
    assert status["n_frames"] == N_FRAMES - 1
    assert status["n_frames_skipped"] == 1
    assert any(
        "1 frame(s) skipped" in w and "absent in one or both cameras" in w
        for w in status["warnings"]
    ), status["warnings"]


def test_stereo_apply_without_skips_reports_zero(chain):
    client, base, source, _stamps = chain
    REC.save_stereo(
        _stereo_record(), REC.stereo_model_dir_for_source(source, 1, 2, "dotboard")
    )
    write_stereo_vector_sets(
        _uncal(base, 1), _uncal(base, 2), N_FRAMES, grid=STEREO_GRID
    )

    resp = client.post(
        "/calibration/apply",
        json={
            "board": "dotboard",
            "stereo": True,
            "camera_pair": [1, 2],
            "interpolator": "linear",
            "dt": 1e-3,
        },
    ).get_json()
    status = _wait(client, resp["job_id"])

    assert status["status"] == "completed", status
    assert status["n_frames_skipped"] == 0
    assert not any("skipped" in w for w in status["warnings"])
