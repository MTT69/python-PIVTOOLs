"""The shipped profile configs carry a pointer-only calibration block.

``Config.active_calibration_method`` raises when ``calibration.active`` is missing,
and ``Config._normalize_calibration_block`` drops every non-pointer key on save, so
the profile YAMLs must hold exactly the pointer keys the CLI template writes.
"""

from pathlib import Path

import pytest

from pivtools_core.config import Config

PROFILE_DIR = Path(__file__).resolve().parent.parent / "profile"
PROFILE_CONFIGS = ("config.yaml", "config_iridis_planar.yaml")
CLI_TEMPLATE_BLOCK = {"calibration_sources": [], "source_idx": 0, "active": "charuco"}


@pytest.mark.parametrize("name", PROFILE_CONFIGS)
def test_profile_config_calibration_block_is_pointer_only(name: str) -> None:
    """Each profile config loads, resolves its active method, and has only pointer keys."""
    cfg = Config(path=str(PROFILE_DIR / name))

    assert cfg.active_calibration_method == "charuco"
    assert cfg.calibration_sources == []
    assert set(cfg.calibration) <= set(Config._CALIBRATION_POINTER_KEYS)
    assert cfg.calibration == CLI_TEMPLATE_BLOCK
