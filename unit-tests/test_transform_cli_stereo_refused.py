"""``pivtools-cli transform`` does not offer stereo (3C) transforms yet.

The operations are written for 2C fields and their effect on uz / the right-handed stereo
frame is unverified, so a stereo request must stop before any file is written, never fall
through to transforming the per-camera data.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import pivtools_core.config as config_mod
import pivtools_gui.transforms.transform_production as tp
from pivtools_cli import cli


class _FakeConfig:
    """The config surface ``transform_command`` reads."""

    def __init__(self, source_endpoint):
        self.camera_numbers = [1, 2]
        self.transforms_cameras = {1: ["flip_ud"]}
        self.transforms_type_name = "instantaneous"
        self.transforms_source_endpoint = source_endpoint
        self.base_paths = ["/data/run"]
        self.active_paths = [0]
        self.data = {}

    def save(self):
        """Config persistence is not under test."""


def _run(monkeypatch, cfg, source_endpoint=None):
    calls: list = []

    class _Recorder:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def process_all_cameras(self):
            return {"success": True, "camera_results": {1: {"transformed_files": 1}}}

    monkeypatch.setattr(tp, "TransformProcessor", _Recorder)
    monkeypatch.setattr(config_mod, "get_config", lambda: cfg)
    args = SimpleNamespace(
        operations=None,
        camera=None,
        type_name=None,
        merged=False,
        source_endpoint=source_endpoint,
        active_paths=None,
    )
    with pytest.raises(SystemExit) as info:
        cli.transform_command(args)
    return calls, info.value.code


def test_stereo_from_config_is_refused_before_processing(monkeypatch, capsys):
    calls, code = _run(monkeypatch, _FakeConfig("stereo"))
    assert code == 1 and calls == []
    assert "stereo transforms are not supported yet" in capsys.readouterr().out


def test_stereo_is_not_a_cli_choice(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["pivtools-cli", "transform", "--source-endpoint", "stereo"]
    )
    with pytest.raises(SystemExit) as info:
        cli.main()
    assert info.value.code == 2  # argparse: invalid choice


def test_regular_endpoint_still_transforms(monkeypatch):
    calls, code = _run(monkeypatch, _FakeConfig("regular"))
    assert code == 0 and len(calls) == 1
    assert calls[0]["camera_transforms"] == {1: ["flip_ud"]}
