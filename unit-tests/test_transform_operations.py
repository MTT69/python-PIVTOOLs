#!/usr/bin/env python3
"""
test_transform_operations.py

Tests for pivtools_gui/transforms/transform_operations.py.

Verifies:
  - simplify_transformations(): Z4 rotation group closure, flip composition,
    inversion composition, scale accumulation
  - apply_transformation_to_piv_result(): stress tensor rotation rules,
    velocity inversion, swap, and scale correctness
  - parse_parametric_transform(): edge cases
  - validate_transformations(): valid and invalid inputs

Usage:
    pytest unit-tests/test_transform_operations.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_gui.transforms.transform_operations import (
    apply_transformation_to_piv_result,
    parse_parametric_transform,
    simplify_transformations,
    validate_transformations,
)

# ---------------------------------------------------------------------------
# Helper: create a mock piv_result with known velocity + stress fields
# ---------------------------------------------------------------------------


def _make_piv_result(shape=(4, 6), ux_val=1.0, uy_val=2.0, with_stresses=False):
    """Create a SimpleNamespace mimicking a piv_result struct."""
    pr = SimpleNamespace()
    pr.ux = np.full(shape, ux_val, dtype=np.float64)
    pr.uy = np.full(shape, uy_val, dtype=np.float64)
    pr.b_mask = np.zeros(shape, dtype=np.float64)

    if with_stresses:
        pr.UU_stress = np.full(shape, 3.0, dtype=np.float64)
        pr.VV_stress = np.full(shape, 5.0, dtype=np.float64)
        pr.UV_stress = np.full(shape, 1.5, dtype=np.float64)

    return pr


# ===========================================================================
# Tests: simplify_transformations — algebraic identities
# ===========================================================================


class TestSimplifyTransformations:
    """Verify transform simplification uses algebraic group properties."""

    # --- Z4 rotation group ---

    def test_four_rotations_cw_is_identity(self):
        """4 × rotate_90_cw = identity."""
        result = simplify_transformations(["rotate_90_cw"] * 4)
        assert result == []

    def test_four_rotations_ccw_is_identity(self):
        """4 × rotate_90_ccw = identity."""
        result = simplify_transformations(["rotate_90_ccw"] * 4)
        assert result == []

    def test_cw_plus_ccw_is_identity(self):
        """rotate_90_cw + rotate_90_ccw = identity."""
        result = simplify_transformations(["rotate_90_cw", "rotate_90_ccw"])
        assert result == []

    def test_two_cw_equals_180(self):
        """2 × rotate_90_cw = rotate_180."""
        result = simplify_transformations(["rotate_90_cw", "rotate_90_cw"])
        assert result == ["rotate_180"]

    def test_two_180_is_identity(self):
        """2 × rotate_180 = identity."""
        result = simplify_transformations(["rotate_180", "rotate_180"])
        assert result == []

    def test_three_cw_equals_one_ccw(self):
        """3 × rotate_90_cw = 1 × rotate_90_ccw."""
        result = simplify_transformations(["rotate_90_cw"] * 3)
        assert result == ["rotate_90_ccw"]

    # --- Z2 flip group ---

    def test_double_flip_ud_is_identity(self):
        """flip_ud + flip_ud = identity."""
        result = simplify_transformations(["flip_ud", "flip_ud"])
        assert result == []

    def test_double_flip_lr_is_identity(self):
        """flip_lr + flip_lr = identity."""
        result = simplify_transformations(["flip_lr", "flip_lr"])
        assert result == []

    # --- Z2 inversion group ---

    def test_double_invert_ux_is_identity(self):
        """invert_ux + invert_ux = identity."""
        result = simplify_transformations(["invert_ux", "invert_ux"])
        assert result == []

    def test_double_invert_uy_is_identity(self):
        """invert_uy + invert_uy = identity."""
        result = simplify_transformations(["invert_uy", "invert_uy"])
        assert result == []

    def test_double_invert_ux_uy_is_identity(self):
        """invert_ux_uy + invert_ux_uy = identity."""
        result = simplify_transformations(["invert_ux_uy", "invert_ux_uy"])
        assert result == []

    def test_double_swap_ux_uy_is_identity(self):
        """swap_ux_uy + swap_ux_uy = identity."""
        result = simplify_transformations(["swap_ux_uy", "swap_ux_uy"])
        assert result == []

    def test_invert_ux_plus_invert_uy_equals_invert_both(self):
        """invert_ux + invert_uy = invert_ux_uy."""
        result = simplify_transformations(["invert_ux", "invert_uy"])
        assert result == ["invert_ux_uy"]

    # --- Scale accumulation ---

    def test_scale_velocity_accumulates(self):
        """scale_velocity:2 + scale_velocity:3 = scale_velocity:6."""
        result = simplify_transformations(["scale_velocity:2", "scale_velocity:3"])
        assert len(result) == 1
        assert result[0].startswith("scale_velocity:")
        assert abs(float(result[0].split(":")[1]) - 6.0) < 1e-10

    def test_scale_velocity_inverse_cancels(self):
        """scale_velocity:2 + scale_velocity:0.5 = identity."""
        result = simplify_transformations(["scale_velocity:2", "scale_velocity:0.5"])
        assert result == []

    def test_scale_coords_accumulates(self):
        """scale_coords:1000 + scale_coords:0.001 = identity."""
        result = simplify_transformations(["scale_coords:1000", "scale_coords:0.001"])
        assert result == []

    # --- Non-adjacent cancellation ---

    def test_non_adjacent_rotation_cancels(self):
        """rotate_90_cw, flip_ud, rotate_90_ccw → flip_ud only."""
        result = simplify_transformations(["rotate_90_cw", "flip_ud", "rotate_90_ccw"])
        assert "flip_ud" in result
        assert not any("rotate" in r for r in result)

    # --- Edge cases ---

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert simplify_transformations([]) == []

    def test_single_transform_preserved(self):
        """Single transform is preserved."""
        assert simplify_transformations(["flip_ud"]) == ["flip_ud"]


# ===========================================================================
# Tests: apply_transformation_to_piv_result — velocity transforms
# ===========================================================================


class TestApplyTransformVelocity:
    """Test velocity field transformations."""

    def test_invert_ux_negates_ux_only(self):
        """invert_ux should negate ux, leave uy unchanged."""
        pr = _make_piv_result(ux_val=5.0, uy_val=3.0)
        apply_transformation_to_piv_result(pr, "invert_ux")
        np.testing.assert_array_equal(pr.ux, -5.0)
        np.testing.assert_array_equal(pr.uy, 3.0)

    def test_invert_uy_negates_uy_only(self):
        """invert_uy should negate uy, leave ux unchanged."""
        pr = _make_piv_result(ux_val=5.0, uy_val=3.0)
        apply_transformation_to_piv_result(pr, "invert_uy")
        np.testing.assert_array_equal(pr.ux, 5.0)
        np.testing.assert_array_equal(pr.uy, -3.0)

    def test_invert_ux_uy_negates_both(self):
        """invert_ux_uy should negate both ux and uy."""
        pr = _make_piv_result(ux_val=5.0, uy_val=3.0)
        apply_transformation_to_piv_result(pr, "invert_ux_uy")
        np.testing.assert_array_equal(pr.ux, -5.0)
        np.testing.assert_array_equal(pr.uy, -3.0)

    def test_swap_ux_uy_swaps(self):
        """swap_ux_uy should swap ux and uy."""
        pr = _make_piv_result(ux_val=5.0, uy_val=3.0)
        apply_transformation_to_piv_result(pr, "swap_ux_uy")
        np.testing.assert_array_equal(pr.ux, 3.0)
        np.testing.assert_array_equal(pr.uy, 5.0)

    def test_scale_velocity(self):
        """scale_velocity:1000 should multiply ux, uy by 1000."""
        pr = _make_piv_result(ux_val=0.001, uy_val=0.002)
        apply_transformation_to_piv_result(pr, "scale_velocity:1000")
        np.testing.assert_allclose(pr.ux, 1.0)
        np.testing.assert_allclose(pr.uy, 2.0)

    def test_flip_ud_flips_spatial(self):
        """flip_ud should flip the 2D arrays vertically."""
        pr = _make_piv_result(shape=(3, 3))
        pr.ux = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        apply_transformation_to_piv_result(pr, "flip_ud")
        expected = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=np.float64)
        np.testing.assert_array_equal(pr.ux, expected)


# ===========================================================================
# Tests: apply_transformation_to_piv_result — stress tensor rules
# ===========================================================================


class TestApplyTransformStress:
    """Test stress tensor transformation rules."""

    def test_scale_velocity_scales_stress_squared(self):
        """scale_velocity:k should scale stresses by k^2."""
        pr = _make_piv_result(with_stresses=True)
        k = 2.0
        UU_before = pr.UU_stress.copy()
        VV_before = pr.VV_stress.copy()
        UV_before = pr.UV_stress.copy()

        apply_transformation_to_piv_result(pr, f"scale_velocity:{k}")

        np.testing.assert_allclose(pr.UU_stress, UU_before * k**2)
        np.testing.assert_allclose(pr.VV_stress, VV_before * k**2)
        np.testing.assert_allclose(pr.UV_stress, UV_before * k**2)

    def test_swap_ux_uy_swaps_UU_VV(self):
        """swap_ux_uy should swap UU_stress and VV_stress, UV unchanged."""
        pr = _make_piv_result(with_stresses=True)
        UU_before = pr.UU_stress.copy()
        VV_before = pr.VV_stress.copy()
        UV_before = pr.UV_stress.copy()

        apply_transformation_to_piv_result(pr, "swap_ux_uy")

        np.testing.assert_array_equal(pr.UU_stress, VV_before)
        np.testing.assert_array_equal(pr.VV_stress, UU_before)
        np.testing.assert_array_equal(pr.UV_stress, UV_before)

    def test_invert_ux_uy_leaves_stresses_unchanged(self):
        """invert_ux_uy: stresses unchanged (variance is sign-invariant)."""
        pr = _make_piv_result(with_stresses=True)
        UU_before = pr.UU_stress.copy()
        VV_before = pr.VV_stress.copy()
        UV_before = pr.UV_stress.copy()

        apply_transformation_to_piv_result(pr, "invert_ux_uy")

        np.testing.assert_array_equal(pr.UU_stress, UU_before)
        np.testing.assert_array_equal(pr.VV_stress, VV_before)
        np.testing.assert_array_equal(pr.UV_stress, UV_before)

    def test_invert_ux_negates_UV_stress(self):
        """invert_ux: UV_stress negated, UU/VV unchanged."""
        pr = _make_piv_result(with_stresses=True)
        UU_before = pr.UU_stress.copy()
        VV_before = pr.VV_stress.copy()
        UV_before = pr.UV_stress.copy()

        apply_transformation_to_piv_result(pr, "invert_ux")

        np.testing.assert_array_equal(pr.UU_stress, UU_before)
        np.testing.assert_array_equal(pr.VV_stress, VV_before)
        np.testing.assert_array_equal(pr.UV_stress, -UV_before)

    def test_invert_uy_negates_UV_stress(self):
        """invert_uy: UV_stress negated, UU/VV unchanged."""
        pr = _make_piv_result(with_stresses=True)
        UU_before = pr.UU_stress.copy()
        VV_before = pr.VV_stress.copy()
        UV_before = pr.UV_stress.copy()

        apply_transformation_to_piv_result(pr, "invert_uy")

        np.testing.assert_array_equal(pr.UU_stress, UU_before)
        np.testing.assert_array_equal(pr.VV_stress, VV_before)
        np.testing.assert_array_equal(pr.UV_stress, -UV_before)

    def test_flip_ud_flips_stress_spatially(self):
        """flip_ud should spatially flip stress fields (same as velocities)."""
        pr = _make_piv_result(shape=(3, 3), with_stresses=True)
        pr.UU_stress = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)

        apply_transformation_to_piv_result(pr, "flip_ud")

        expected = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=np.float64)
        np.testing.assert_array_equal(pr.UU_stress, expected)


# ===========================================================================
# Tests: parse_parametric_transform
# ===========================================================================


class TestParseParametricTransform:
    """Test parametric transform parsing edge cases."""

    def test_simple_transform(self):
        """Non-parametric transform returns (name, None)."""
        name, param = parse_parametric_transform("flip_ud")
        assert name == "flip_ud"
        assert param is None

    def test_scale_velocity_with_factor(self):
        """scale_velocity:2.5 returns correct name and factor."""
        name, param = parse_parametric_transform("scale_velocity:2.5")
        assert name == "scale_velocity"
        assert param == 2.5

    def test_scale_coords_with_factor(self):
        """scale_coords:0.001 returns correct factor."""
        name, param = parse_parametric_transform("scale_coords:0.001")
        assert name == "scale_coords"
        assert abs(param - 0.001) < 1e-10

    def test_scale_velocity_zero_factor_raises(self):
        """Zero scale factor should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be zero"):
            parse_parametric_transform("scale_velocity:0")

    def test_invalid_parametric_raises(self):
        """Unknown parametric transform raises ValueError."""
        with pytest.raises(ValueError, match="Unknown parametric"):
            parse_parametric_transform("bad_transform:123")

    def test_missing_factor_raises(self):
        """Parametric transform without numeric factor raises ValueError."""
        with pytest.raises(ValueError, match="Invalid parameter"):
            parse_parametric_transform("scale_velocity:abc")


# ===========================================================================
# Tests: validate_transformations
# ===========================================================================


class TestValidateTransformations:
    """Test transformation validation."""

    def test_valid_transforms_pass(self):
        """Known transforms should pass validation."""
        valid, err = validate_transformations(["flip_ud", "rotate_90_cw"])
        assert valid
        assert err is None

    def test_invalid_transform_fails(self):
        """Unknown transform should fail validation."""
        valid, err = validate_transformations(["flip_ud", "totally_bogus"])
        assert not valid
        assert "totally_bogus" in err

    def test_empty_list_fails_by_default(self):
        """Empty list fails without allow_empty."""
        valid, err = validate_transformations([])
        assert not valid

    def test_empty_list_allowed(self):
        """Empty list passes with allow_empty=True."""
        valid, err = validate_transformations([], allow_empty=True)
        assert valid

    def test_parametric_without_factor_fails(self):
        """scale_velocity without :factor should fail."""
        valid, err = validate_transformations(["scale_velocity"])
        assert not valid

    def test_parametric_with_factor_passes(self):
        """scale_velocity:1000 should pass."""
        valid, err = validate_transformations(["scale_velocity:1000"])
        assert valid
