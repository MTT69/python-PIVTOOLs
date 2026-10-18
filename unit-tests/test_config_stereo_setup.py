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
