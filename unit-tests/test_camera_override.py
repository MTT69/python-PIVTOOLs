#!/usr/bin/env python3
"""
test_camera_override.py

Tests for the single-camera override on the instantaneous/ensemble PIV
commands. Adding `--camera N` sets the PIV_CAMERA environment variable, which
the `Config.camera_numbers` property honours by returning `[N]`.

Covers:
  - PIV_CAMERA set -> camera_numbers == [N]
  - PIV_CAMERA unset -> config list returned unchanged
  - out-of-range -> ValueError (existing range check)
  - non-integer -> ValueError (fail loud, NO silent fallback)
  - argparse wiring: `instantaneous`/`ensemble --camera 5` -> args.camera == 5,
    default None

Usage:
    pytest unit-tests/test_camera_override.py -v
"""

import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

# Ensure production code is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_core.config import Config


def _write_config(tmp_path, camera_count=5, camera_numbers=(1, 2, 3, 4, 5)):
    """Write a minimal config.yaml with just the paths block the property reads."""
    cfg = {
        "paths": {
            "camera_count": camera_count,
            "camera_numbers": list(camera_numbers),
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return Config(path=str(config_path))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Guarantee PIV_CAMERA never leaks in or out of a test."""
    monkeypatch.delenv("PIV_CAMERA", raising=False)
    yield


def test_env_override_selects_single_camera(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    monkeypatch.setenv("PIV_CAMERA", "5")
    assert config.camera_numbers == [5]


def test_no_env_returns_config_list(tmp_path):
    config = _write_config(tmp_path, camera_numbers=(1, 2, 3, 4, 5))
    assert config.camera_numbers == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("bad", ["6", "0", "-1"])
def test_env_override_out_of_range_raises(tmp_path, monkeypatch, bad):
    config = _write_config(tmp_path, camera_count=5)
    monkeypatch.setenv("PIV_CAMERA", bad)
    with pytest.raises(ValueError):
        _ = config.camera_numbers


def test_env_override_non_integer_raises(tmp_path, monkeypatch):
    """Malformed value must fail loudly, not silently fall back to the config list."""
    config = _write_config(tmp_path)
    monkeypatch.setenv("PIV_CAMERA", "abc")
    with pytest.raises(ValueError):
        _ = config.camera_numbers


@pytest.mark.parametrize("command", ["instantaneous", "ensemble"])
def test_argparse_camera_flag(command, monkeypatch):
    """`<command> --camera 5` parses to args.camera == 5; default is None."""
    from pivtools_cli import cli

    captured = {}

    def _capture(args):
        captured["args"] = args

    # Route both handlers through the capture so dispatch never runs the pipeline.
    monkeypatch.setattr(cli, "instantaneous_command", _capture)
    monkeypatch.setattr(cli, "ensemble_command", _capture)

    monkeypatch.setattr(sys, "argv", ["pivtools-cli", command, "--camera", "5"])
    cli.main()
    assert captured["args"].camera == 5

    captured.clear()
    monkeypatch.setattr(sys, "argv", ["pivtools-cli", command])
    cli.main()
    assert captured["args"].camera is None


@pytest.mark.parametrize("command", ["instantaneous", "ensemble"])
def test_handler_sets_and_clears_env(command, monkeypatch):
    """The handler must set PIV_CAMERA from the flag, and clear a stale value
    when the flag is absent (no silent single-camera run)."""
    import types

    import pivtools_core
    from pivtools_cli import cli

    seen = {}

    def _fake_main():
        # Capture what main() would observe at run time.
        seen["piv_camera"] = os.environ.get("PIV_CAMERA")

    # Inject a stub so the handler's `from pivtools_core import <command>` binds
    # this instead of importing the real (heavy) pipeline module — we only want
    # to exercise the env-var glue in the handler. The sys.modules entry alone
    # is NOT enough: `from pivtools_core import <command>` prefers an existing
    # package ATTRIBUTE, and collecting any test module that imports the real
    # pipeline (e.g. test_multipass_convergence.py) sets that attribute before
    # this test runs — so stub the attribute too, or the REAL main() executes.
    fake = types.ModuleType(f"pivtools_core.{command}")
    fake.main = _fake_main
    monkeypatch.setitem(sys.modules, f"pivtools_core.{command}", fake)
    monkeypatch.setattr(pivtools_core, command, fake, raising=False)

    handler = getattr(cli, f"{command}_command")

    # Flag present -> env set, visible to main()
    handler(Namespace(active_paths=None, camera=5))
    assert seen["piv_camera"] == "5"

    # The handler set PIV_CAMERA=5 DIRECTLY (not via monkeypatch), so pop it
    # before monkeypatch.setenv below snapshots it — otherwise teardown
    # "restores" the leaked 5 into the process env and poisons every later
    # config-reading test ("Camera numbers [5] must be between 1 and 1").
    os.environ.pop("PIV_CAMERA", None)

    # Stale value inherited + flag absent -> cleared, main() sees no override
    monkeypatch.setenv("PIV_CAMERA", "3")
    handler(Namespace(active_paths=None, camera=None))
    assert seen["piv_camera"] is None
