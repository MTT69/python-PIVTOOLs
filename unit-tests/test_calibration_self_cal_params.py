"""Self-calibration count parameters must be positive integers, named on failure.

``/calibration/self_cal/run`` read ``int(data.get("n_images", 20))``: an explicit
null raised TypeError (HTTP 500) and 0 reached ``load_particle_pairs``, which
divides by it. The stored block was read with ``int(sc.get(..., 0))``, turning a
missing field into a fabricated 0. Both now raise ValueError naming the field and
the routes return ``success: false``.
"""

from __future__ import annotations

import pytest
from flask import Flask
from test_calibration_self_cal import (  # noqa: E402  (sibling-test reuse, as elsewhere)
    _FakeResult,
    _stereo_record,
)

import pivtools_gui.calibration.app.views as V
from pivtools_gui.calibration import self_cal as SC


@pytest.fixture
def client(tmp_path, monkeypatch):
    record = _stereo_record()
    monkeypatch.setattr(
        V,
        "_stereo_locators",
        lambda get: (tmp_path, 1, 2, "dotboard", tmp_path / "model", tmp_path / "fig"),
    )
    monkeypatch.setattr(V.rec, "load_stereo", lambda *a, **k: record)
    monkeypatch.setattr(V, "get_config", lambda: None)
    monkeypatch.setattr(V, "_model_type_arg", lambda *a, **k: None)

    def _no_job(*a, **k):
        raise AssertionError("an invalid request must not start a job")

    monkeypatch.setattr(V.job_manager, "create_job", _no_job)
    app = Flask(__name__)
    app.register_blueprint(V.calibration_bp)
    return app.test_client(), record


@pytest.mark.parametrize(
    "field, value",
    [("n_images", None), ("n_images", 0), ("window_size", -8), ("window_size", None)],
)
def test_run_rejects_non_positive_counts(client, field, value):
    c, _record = client
    out = c.post("/calibration/self_cal/run", json={field: value}).get_json()
    assert out["success"] is False
    assert out["error"] == f"{field} must be a positive integer (got {value!r})"


def _block(**overrides) -> dict:
    block = SC.baked_block(
        _FakeResult(z=1.0, tx=0.0, ty=0.0, conv=True, rms=0.1, n=3),
        {},
        n_images=20,
        window_size=64,
        overlap=50.0,
    )
    block.update(overrides)
    return block


@pytest.mark.parametrize("field", ["n_images", "window_size", "n_iterations"])
def test_payload_rejects_missing_or_zero_stored_count(field):
    block = _block()
    del block[field]
    with pytest.raises(ValueError, match=f"self_cal.{field} must be a positive"):
        V._self_cal_payload(_stereo_record(self_cal=block))
    with pytest.raises(ValueError, match=f"self_cal.{field} must be a positive"):
        V._self_cal_payload(_stereo_record(self_cal=_block(**{field: 0})))


def test_payload_valid_block():
    payload = V._self_cal_payload(_stereo_record(self_cal=_block()))
    assert (payload["n_images"], payload["window_size"], payload["n_iterations"]) == (
        20,
        64,
        3,
    )


def test_result_route_reports_bad_block(client):
    c, record = client
    record.self_cal = _block(n_images=0)
    out = c.get("/calibration/self_cal/result").get_json()
    assert out["success"] is False
    assert out["error"] == "self_cal.n_images must be a positive integer (got 0)"


@pytest.mark.parametrize("bad", [0, -3])
def test_load_particle_pairs_rejects_non_positive_n_images(bad):
    """The division by n_images is guarded for every caller (GUI and CLI)."""
    with pytest.raises(ValueError, match="n_images must be a positive integer"):
        SC.load_particle_pairs(None, 0, 1, 2, bad)
