"""CLI stereo transforms: the stereo pair and a single op list, never a guessed one.

The GUI viewer does not offer transforms on stereo data (VectorViewer hides the panel), so
the CLI is the only stereo transform path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import pivtools_core.config as config_mod
import pivtools_gui.transforms.transform_production as tp
from pivtools_cli import cli
from pivtools_gui.transforms.transform_production import stereo_transform_target


class _FakeConfig:
    """The config surface the stereo rule and ``transform_command`` read."""

    def __init__(self, camera_numbers, transforms, base_path="/data/run"):
        self.camera_numbers = camera_numbers
        self.transforms_cameras = transforms
        self.transforms_type_name = "instantaneous"
        self.transforms_source_endpoint = "regular"
        self.base_paths = [base_path]
        self.active_paths = [0]
        self.data = {}

    @property
    def stereo_pairs(self):
        c = self.camera_numbers
        return [(c[i], c[i + 1]) for i in range(0, len(c) - 1, 2)]

    def save(self):
        """Config persistence is not under test."""


def test_shared_ops_on_both_cameras_give_pair_and_ops():
    cfg = _FakeConfig([1, 2], {})
    pair, ops = stereo_transform_target(cfg, {1: ["flip_ud"], 2: ["flip_ud"]})
    assert pair == (1, 2) and ops == ["flip_ud"]


def test_ops_on_one_camera_only_are_used():
    cfg = _FakeConfig([1, 2], {})
    pair, ops = stereo_transform_target(cfg, {1: [], 2: ["rotate_90_cw"]})
    assert pair == (1, 2) and ops == ["rotate_90_cw"]


def test_different_ops_per_camera_raise_naming_both():
    cfg = _FakeConfig([1, 2], {})
    with pytest.raises(ValueError, match=r"Cam1: \['flip_ud'\].*Cam2: \['flip_lr'\]"):
        stereo_transform_target(cfg, {1: ["flip_ud"], 2: ["flip_lr"]})


def test_equivalent_ops_after_simplification_are_not_ambiguous():
    cfg = _FakeConfig([1, 2], {})
    _pair, ops = stereo_transform_target(
        cfg, {1: ["flip_ud"], 2: ["flip_ud", "flip_lr", "flip_lr"]}
    )
    assert ops in (["flip_ud"], ["flip_ud", "flip_lr", "flip_lr"])


def test_single_camera_raises():
    with pytest.raises(ValueError, match="needs two cameras"):
        stereo_transform_target(_FakeConfig([1], {}), {1: ["flip_ud"]})


def test_no_pending_ops_raise():
    with pytest.raises(ValueError, match="no pending transforms"):
        stereo_transform_target(_FakeConfig([1, 2], {}), {1: [], 2: []})


def _run_cli(monkeypatch, cfg, **arg_overrides):
    """Run ``transform_command`` with the processor replaced by a recorder."""
    calls: list = []

    class _Recorder:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.cams = list(kwargs["camera_transforms"])

        def process_all_cameras(self):
            return {
                "success": True,
                "camera_results": {c: {"transformed_files": 3} for c in self.cams},
            }

    monkeypatch.setattr(tp, "TransformProcessor", _Recorder)
    monkeypatch.setattr(config_mod, "get_config", lambda: cfg)
    args = SimpleNamespace(
        operations=None,
        camera=None,
        type_name=None,
        merged=False,
        source_endpoint="stereo",
        active_paths=None,
    )
    for k, v in arg_overrides.items():
        setattr(args, k, v)
    with pytest.raises(SystemExit) as info:
        cli.transform_command(args)
    return calls, info.value.code


def test_cli_stereo_passes_stereo_flag_pair_and_one_op_list(monkeypatch, capsys):
    cfg = _FakeConfig([1, 2], {1: ["flip_ud"], 2: ["flip_ud"]})
    calls, code = _run_cli(monkeypatch, cfg)
    assert code == 0 and len(calls) == 1
    kw = calls[0]
    assert kw["use_stereo"] is True
    assert kw["stereo_camera_pair"] == (1, 2)
    assert kw["camera_transforms"] == {1: ["flip_ud"]}
    assert "Stereo Cam1_Cam2: 3 files" in capsys.readouterr().out


def test_cli_stereo_with_conflicting_ops_exits_before_processing(monkeypatch):
    cfg = _FakeConfig([1, 2], {1: ["flip_ud"], 2: ["flip_lr"]})
    calls, code = _run_cli(monkeypatch, cfg)
    assert code == 1 and calls == []


def test_cli_regular_endpoint_is_not_stereo(monkeypatch):
    cfg = _FakeConfig([1, 2], {1: ["flip_ud"], 2: ["flip_lr"]})
    calls, code = _run_cli(monkeypatch, cfg, source_endpoint="regular")
    assert code == 0
    assert calls[0]["use_stereo"] is False
    assert calls[0]["stereo_camera_pair"] is None
    assert calls[0]["camera_transforms"] == {1: ["flip_ud"], 2: ["flip_lr"]}
