#!/usr/bin/env python3
"""
test_validation.py

Tests for pivtools_core/validation.py.

Verifies all error paths in validate_ensemble_config() and
edge cases in validate_batch_size_for_pod().

Usage:
    pytest unit-tests/test_validation.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_core.fft_sizes import BUILT_FFT_SIZES
from pivtools_core.validation import (
    _check_built_sizes,
    validate_batch_size_for_pod,
    validate_ensemble_config,
    validate_window_sizes,
)

# ---------------------------------------------------------------------------
# Helper: build a mock Config with controllable properties
# ---------------------------------------------------------------------------


def _make_config(
    ensemble_type=None,
    window_sizes=None,
    overlaps=None,
    sum_window=None,
    sum_fitting_window_enabled=False,
    sum_fitting_window=None,
    fit_method="kspace",
    kspace_shape="gaussian",
    background_subtraction_method="correlation",
    per_pair_normalization=False,
    resume_from_pass=0,
    num_passes=None,
    filters=None,
    type_raise=None,
    window_raise=None,
    overlap_raise=None,
    sum_window_raise=None,
    sum_fitting_window_raise=None,
    fit_method_raise=None,
    kspace_shape_raise=None,
):
    """Create a mock Config object with the specified ensemble properties."""
    cfg = MagicMock()

    # ensemble_type
    if type_raise:
        type(cfg).ensemble_type = PropertyMock(side_effect=type_raise)
    else:
        type(cfg).ensemble_type = PropertyMock(return_value=ensemble_type or ["std"])

    # window_sizes
    if window_raise:
        type(cfg).ensemble_window_sizes = PropertyMock(side_effect=window_raise)
    else:
        type(cfg).ensemble_window_sizes = PropertyMock(
            return_value=window_sizes or [[64, 64]]
        )

    # overlaps
    if overlap_raise:
        type(cfg).ensemble_overlaps = PropertyMock(side_effect=overlap_raise)
    else:
        type(cfg).ensemble_overlaps = PropertyMock(return_value=overlaps or [50])

    # sum_window
    if sum_window_raise:
        type(cfg).ensemble_sum_window = PropertyMock(side_effect=sum_window_raise)
    else:
        type(cfg).ensemble_sum_window = PropertyMock(
            return_value=sum_window or [128, 128]
        )

    # sum_fitting_window
    type(cfg).ensemble_sum_fitting_window_enabled = PropertyMock(
        return_value=sum_fitting_window_enabled
    )
    if sum_fitting_window_raise:
        type(cfg).ensemble_sum_fitting_window = PropertyMock(
            side_effect=sum_fitting_window_raise
        )
    else:
        type(cfg).ensemble_sum_fitting_window = PropertyMock(
            return_value=sum_fitting_window or [64, 64]
        )

    # fit_method
    if fit_method_raise:
        type(cfg).ensemble_fit_method = PropertyMock(side_effect=fit_method_raise)
    else:
        type(cfg).ensemble_fit_method = PropertyMock(return_value=fit_method)

    # kspace_shape (check 7b: quartic terms are LM-fitter-only)
    if kspace_shape_raise:
        type(cfg).ensemble_kspace_shape = PropertyMock(side_effect=kspace_shape_raise)
    else:
        type(cfg).ensemble_kspace_shape = PropertyMock(return_value=kspace_shape)

    # background subtraction / per-pair normalization consistency (check 8)
    type(cfg).ensemble_background_subtraction_method = PropertyMock(
        return_value=background_subtraction_method
    )
    type(cfg).ensemble_per_pair_normalization = PropertyMock(
        return_value=per_pair_normalization
    )

    # resume / num_passes
    type(cfg).ensemble_resume_from_pass = PropertyMock(return_value=resume_from_pass)
    if num_passes is None:
        # Default: length of window_sizes or 1
        num_passes = len(window_sizes) if window_sizes else 1
    type(cfg).ensemble_num_passes = PropertyMock(return_value=num_passes)

    # filters (for POD test)
    type(cfg).filters = PropertyMock(return_value=filters or [])

    return cfg


# ===========================================================================
# Tests: validate_ensemble_config
# ===========================================================================


class TestValidateEnsembleConfig:
    """Test each error path in validate_ensemble_config()."""

    def test_valid_config_passes(self):
        """A well-formed config should return is_valid=True with no errors."""
        cfg = _make_config(
            ensemble_type=["std", "std"],
            window_sizes=[[64, 64], [32, 32]],
            overlaps=[50, 50],
        )
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert valid
        assert len(errors) == 0

    def test_invalid_ensemble_type_raises(self):
        """ValueError from ensemble_type property produces error."""
        cfg = _make_config(type_raise=ValueError("bad type list"))
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert not valid
        assert any("Ensemble type" in e for e in errors)

    def test_invalid_window_sizes_raises(self):
        """ValueError from ensemble_window_sizes produces error."""
        cfg = _make_config(window_raise=ValueError("bad window"))
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert not valid
        assert any("window sizes" in e for e in errors)

    def test_invalid_overlaps_raises(self):
        """ValueError from ensemble_overlaps produces error."""
        cfg = _make_config(overlap_raise=ValueError("bad overlap"))
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert not valid
        assert any("overlap" in e.lower() for e in errors)

    def test_overlap_out_of_range(self):
        """Overlap < 0 or > 95 should produce error."""
        cfg = _make_config(overlaps=[110])
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert not valid
        assert any("out of range" in e for e in errors)

    def test_overlap_negative(self):
        """Negative overlap should produce error."""
        cfg = _make_config(overlaps=[-5])
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert not valid
        assert any("out of range" in e for e in errors)

    def test_increasing_window_size_warns(self):
        """Window sizes that increase across passes produce a warning."""
        cfg = _make_config(
            window_sizes=[[32, 32], [64, 64]],
            overlaps=[50, 50],
            ensemble_type=["std", "std"],
        )
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert valid  # warning, not error
        assert any("larger" in w for w in warnings)

    def test_single_mode_invalid_sum_window(self):
        """Single mode with invalid sum_window produces error."""
        cfg = _make_config(
            ensemble_type=["single"],
            sum_window_raise=ValueError("bad sum window"),
        )
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert not valid
        assert any("sum window" in e.lower() for e in errors)

    def test_sum_fitting_window_invalid(self):
        """Enabled sum_fitting_window that raises ValueError produces error."""
        cfg = _make_config(
            sum_fitting_window_enabled=True,
            sum_fitting_window_raise=ValueError("bad fitting window"),
        )
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert not valid
        assert any("sum fitting window" in e.lower() for e in errors)

    def test_invalid_fit_method(self):
        """ValueError from ensemble_fit_method produces error."""
        cfg = _make_config(fit_method_raise=ValueError("bad fit method"))
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert not valid
        assert any("fit method" in e.lower() for e in errors)

    def test_kspace_valid_passes(self):
        """A valid k-space config passes validation with no errors."""
        cfg = _make_config(fit_method="kspace")
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert valid

    def test_invalid_kspace_shape(self):
        """ValueError from ensemble_kspace_shape produces error."""
        cfg = _make_config(kspace_shape_raise=ValueError("bad shape"))
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert not valid
        assert any("kspace shape" in e.lower() for e in errors)

    @pytest.mark.parametrize("shape", ["gaussian", "kx4", "ky4", "kx4+ky4"])
    def test_kspace_shape_valid_with_lm_fitter(self, shape):
        """All four shapes pass with fit_method kspace."""
        cfg = _make_config(fit_method="kspace", kspace_shape=shape)
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert valid, errors

    def test_kspace_shape_requires_lm_fitter(self):
        """Non-gaussian shape + kspace_linear is an error, not a silent no-op."""
        cfg = _make_config(fit_method="kspace_linear", kspace_shape="kx4+ky4")
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert not valid
        assert any("kspace_shape" in e for e in errors)

    def test_gaussian_shape_fine_with_linear_fitter(self):
        """The default shape never conflicts with kspace_linear."""
        cfg = _make_config(fit_method="kspace_linear", kspace_shape="gaussian")
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert valid, errors

    def test_resume_from_pass_out_of_range(self):
        """resume_from_pass beyond num_passes should produce error."""
        cfg = _make_config(
            window_sizes=[[64, 64], [32, 32]],
            overlaps=[50, 50],
            resume_from_pass=5,
            num_passes=2,
        )
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert not valid
        assert any("resume_from_pass" in e for e in errors)

    def test_resume_from_pass_zero_is_disabled(self):
        """resume_from_pass=0 means disabled, should pass."""
        cfg = _make_config(resume_from_pass=0)
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert valid

    def test_resume_from_pass_valid(self):
        """resume_from_pass within range should pass."""
        cfg = _make_config(
            window_sizes=[[64, 64], [32, 32]],
            overlaps=[50, 50],
            resume_from_pass=2,
            num_passes=2,
        )
        valid, errors, warnings = validate_ensemble_config(cfg)
        assert valid


# ===========================================================================
# Tests: validate_batch_size_for_pod
# ===========================================================================


class TestValidateBatchSizeForPod:
    """Test validate_batch_size_for_pod edge cases."""

    def test_no_pod_filter_always_valid(self):
        """Without POD filter, any batch size is valid."""
        cfg = _make_config(filters=[{"type": "gaussian", "sigma": 1.0}])
        valid, msg = validate_batch_size_for_pod(cfg, 1)
        assert valid
        assert msg == ""

    def test_pod_batch_size_too_small(self):
        """POD with batch_size < 10 should fail."""
        cfg = _make_config(filters=[{"type": "pod"}])
        valid, msg = validate_batch_size_for_pod(cfg, 5)
        assert not valid
        assert "batch_size" in msg.lower()

    def test_pod_batch_size_marginal(self):
        """POD with batch_size 10-19 should pass with warning."""
        cfg = _make_config(filters=[{"type": "pod"}])
        valid, msg = validate_batch_size_for_pod(cfg, 15)
        assert valid
        assert "suboptimal" in msg.lower()

    def test_pod_batch_size_sufficient(self):
        """POD with batch_size >= 20 should pass with no warning."""
        cfg = _make_config(filters=[{"type": "pod"}])
        valid, msg = validate_batch_size_for_pod(cfg, 50)
        assert valid
        assert msg == ""

    def test_empty_filters(self):
        """Empty filter list is fine."""
        cfg = _make_config(filters=[])
        valid, msg = validate_batch_size_for_pod(cfg, 1)
        assert valid

    def test_none_filters(self):
        """None filters should not crash."""
        cfg = _make_config(filters=None)
        # filters property returns None, validate should handle gracefully
        type(cfg).filters = PropertyMock(return_value=None)
        valid, msg = validate_batch_size_for_pod(cfg, 1)
        assert valid


# ---------------------------------------------------------------------------
# Window-size restriction (codelet FFT engine supports a fixed set of sizes)
# ---------------------------------------------------------------------------


def _window_config(
    instantaneous,
    ensemble=None,
    ensemble_enabled=False,
    ensemble_types=None,
    sum_window=None,
):
    """Minimal mock Config exposing only what validate_window_sizes reads."""
    cfg = MagicMock()
    type(cfg).window_sizes = PropertyMock(return_value=instantaneous)
    type(cfg).ensemble_window_sizes = PropertyMock(
        return_value=ensemble if ensemble is not None else instantaneous
    )
    data = {"processing": {"ensemble": ensemble_enabled}}
    ensemble_piv = {}
    if ensemble_types is not None:
        ensemble_piv["type"] = ensemble_types
    if sum_window is not None:
        ensemble_piv["sum_window"] = sum_window
    if ensemble_piv:
        data["ensemble_piv"] = ensemble_piv
    cfg.data = data
    return cfg


class TestCheckBuiltSizes:
    def test_all_built_sizes_pass(self):
        sizes = [[s, s] for s in BUILT_FFT_SIZES]
        assert _check_built_sizes(sizes, "x") == []

    def test_rectangular_built_sizes_pass(self):
        assert _check_built_sizes([[16, 32], [32, 16], [96, 128]], "x") == []

    def test_unsupported_size_reported(self):
        errors = _check_built_sizes([[100, 100]], "instantaneous_piv.window_size")
        assert len(errors) == 1
        assert "100" in errors[0]
        assert "instantaneous_piv.window_size pass 1" in errors[0]

    def test_one_bad_axis_reported(self):
        # 128 is built, 100 is not -> still an error
        errors = _check_built_sizes([[128, 100]], "x")
        assert len(errors) == 1
        assert "100" in errors[0]

    def test_common_256_rejected(self):
        # 256 is a common PIV size but is NOT built -> must fail loud
        assert 256 not in BUILT_FFT_SIZES
        errors = _check_built_sizes([[256, 256]], "x")
        assert len(errors) == 1

    def test_malformed_size_reported(self):
        errors = _check_built_sizes([[32]], "x")
        assert len(errors) == 1
        assert "malformed" in errors[0].lower()


class TestValidateWindowSizes:
    def test_valid_instantaneous_no_ensemble(self):
        cfg = _window_config([[128, 128], [64, 64], [32, 32]])
        assert validate_window_sizes(cfg) == []

    def test_bad_instantaneous_fails(self):
        cfg = _window_config([[128, 128], [100, 100]])
        errors = validate_window_sizes(cfg)
        assert any("100" in e for e in errors)

    def test_ensemble_checked_when_enabled(self):
        cfg = _window_config(
            instantaneous=[[64, 64]],
            ensemble=[[64, 64], [256, 256]],
            ensemble_enabled=True,
        )
        errors = validate_window_sizes(cfg)
        assert any("ensemble_piv.window_size" in e for e in errors)

    def test_ensemble_ignored_when_disabled(self):
        cfg = _window_config(
            instantaneous=[[64, 64]],
            ensemble=[[256, 256]],
            ensemble_enabled=False,
        )
        assert validate_window_sizes(cfg) == []

    def test_sum_window_checked_in_single_mode(self):
        # sum_window is the FFT computation size in 'single' mode -> must be built
        cfg = _window_config(
            instantaneous=[[64, 64]],
            ensemble=[[64, 64]],
            ensemble_enabled=True,
            ensemble_types=["single"],
            sum_window=[20, 20],
        )
        errors = validate_window_sizes(cfg)
        assert any("ensemble_piv.sum_window" in e and "20" in e for e in errors)

    def test_valid_sum_window_passes_in_single_mode(self):
        cfg = _window_config(
            instantaneous=[[64, 64]],
            ensemble=[[64, 64]],
            ensemble_enabled=True,
            ensemble_types=["single"],
            sum_window=[32, 32],
        )
        assert validate_window_sizes(cfg) == []

    def test_sum_window_ignored_in_std_mode(self):
        # No 'single' pass -> sum_window is unused, so it is not checked
        cfg = _window_config(
            instantaneous=[[64, 64]],
            ensemble=[[64, 64]],
            ensemble_enabled=True,
            ensemble_types=["std"],
            sum_window=[20, 20],
        )
        assert validate_window_sizes(cfg) == []
