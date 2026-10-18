"""Config.is_stereo_setup — which ``calibration.active`` values mean stereo data.

Statistics, plotting, the video maker, merging and transforms branch on this property, so
every stereo method the GUI can make active must be recognised, including stepped stereo.
"""

from __future__ import annotations

import pytest
import yaml

from pivtools_cli.cli import create_default_config
from pivtools_core.config import Config

# The values the GUI "Set as Active" buttons write (gui/src/components/setup/Calibration.tsx).
STEREO_METHODS = ("stereo_dotboard", "stereo_charuco", "stereo_stepped")
MONO_METHODS = ("dotboard", "charuco", "scale_factor")


def _config_with_active(tmp_path, active: str) -> Config:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"calibration": {"active": active}}), encoding="utf-8"
    )
    return Config(path=str(path))


@pytest.mark.parametrize("active", STEREO_METHODS)
def test_stereo_methods_are_stereo(tmp_path, active):
    assert _config_with_active(tmp_path, active).is_stereo_setup


@pytest.mark.parametrize("active", MONO_METHODS)
def test_mono_methods_are_not_stereo(tmp_path, active):
    assert not _config_with_active(tmp_path, active).is_stereo_setup


def test_cli_template_names_gui_active_values(tmp_path):
    """The template comment lists the active values the GUI writes, not stale spellings."""
    path = tmp_path / "config.yaml"
    create_default_config(str(path))
    text = path.read_text(encoding="utf-8")
    assert "stereo_stepped" in text
    assert "stepped_stereo" not in text
    for method in STEREO_METHODS + MONO_METHODS:
        assert method in text


@pytest.mark.parametrize("calibration", [{}, {"active": ""}, {"active": "  "}, None])
def test_missing_active_raises(tmp_path, calibration):
    """``calibration.active`` is required: no default method is guessed for downstream
    stereo/mono branching (the CLI template always writes it)."""
    path = tmp_path / "config.yaml"
    body = {} if calibration is None else {"calibration": calibration}
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    cfg = Config(path=str(path))
    with pytest.raises(ValueError, match=r"calibration\.active"):
        cfg.active_calibration_method
    with pytest.raises(ValueError, match="stereo_stepped"):
        cfg.is_stereo_setup


@pytest.mark.parametrize("active", ["stereo_charcuo", "stepped_stereo", "stepped"])
def test_unknown_active_raises(tmp_path, active):
    """A misspelt or retired id is not silently read as mono by downstream tools."""
    cfg = _config_with_active(tmp_path, active)
    with pytest.raises(ValueError, match="is not a calibration method"):
        cfg.active_calibration_method
    with pytest.raises(ValueError, match="is not a calibration method"):
        cfg.is_stereo_setup
