"""Config.stereo_ensemble_coc_reference: default, opt-in, and validation.

Guards the production-default flip back to the legacy 'ac' reference and the
no-silent-fallback validation of the stereo-CoC k-space reference toggle.
"""

from __future__ import annotations

import pytest

from pivtools_core.config import Config


def _config(tmp_path, yaml_text: str) -> Config:
    p = tmp_path / "config.yaml"
    p.write_text(yaml_text)
    return Config(str(p))


def test_defaults_to_ac_when_unset(tmp_path):
    # No stereo_ensemble_piv block at all -> legacy AC reference.
    cfg = _config(tmp_path, "instantaneous_piv: {}\n")
    assert cfg.stereo_ensemble_coc_reference == "ac"


def test_block_present_but_key_absent_defaults_to_ac(tmp_path):
    # stereo_ensemble_piv exists (other keys) but no coc_reference -> 'ac'.
    cfg = _config(
        tmp_path,
        "stereo_ensemble_piv:\n  background_subtraction_method: none\n",
    )
    assert cfg.stereo_ensemble_coc_reference == "ac"


def test_opt_in_coc_aa(tmp_path):
    cfg = _config(tmp_path, "stereo_ensemble_piv:\n  coc_reference: coc_aa\n")
    assert cfg.stereo_ensemble_coc_reference == "coc_aa"


def test_invalid_value_raises(tmp_path):
    cfg = _config(tmp_path, "stereo_ensemble_piv:\n  coc_reference: garbage\n")
    with pytest.raises(ValueError, match="Invalid stereo_ensemble_coc_reference"):
        _ = cfg.stereo_ensemble_coc_reference


def test_cli_default_template_round_trips_to_ac(tmp_path):
    # The create_default_config template must emit stereo_ensemble_piv with
    # coc_reference: ac, and Config must read it back as 'ac'.
    import yaml

    from pivtools_cli.cli import create_default_config

    p = tmp_path / "default_config.yaml"
    create_default_config(str(p))
    blk = yaml.safe_load(p.read_text()).get("stereo_ensemble_piv")
    assert blk == {"coc_reference": "ac"}, blk
    assert Config(str(p)).stereo_ensemble_coc_reference == "ac"
