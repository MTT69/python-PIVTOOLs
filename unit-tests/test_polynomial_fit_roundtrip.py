#!/usr/bin/env python3
"""
test_polynomial_fit_roundtrip.py

Unit tests for the polynomial calibration fitting and evaluation pipeline:
  - fit_polynomial_from_points: round-trip coefficient recovery
  - evaluate_polynomial_terms: direct term-ordering verification
  - PolynomialVectorCalibrator: coordinate/vector calibration math
  - Edge case: missing image_height

No external data required — all tests use synthetic point correspondences.

Usage:
    pytest unit-tests/test_polynomial_fit_roundtrip.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
    PolynomialVectorCalibrator,
    convert_davis_coeffs_to_array,
    evaluate_polynomial_terms,
    fit_polynomial_from_points,
    read_calibration_xml,
    save_polynomial_to_config,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_grid_points(nx=20, ny=20, spacing_mm=10.0):
    """Create a regular grid of world points (mm) and corresponding image points.

    Returns (image_points, world_points, image_shape, mm_per_pixel).

    The synthetic camera is a simple pinhole at the origin with known scale.
    We inject known polynomial distortion so we can verify recovery.
    """
    mm_per_pixel = 0.25  # known ground truth

    # World coords on a regular grid in mm
    wx = np.arange(nx) * spacing_mm
    wy = np.arange(ny) * spacing_mm
    WX, WY = np.meshgrid(wx, wy)
    world_points = np.column_stack([WX.ravel(), WY.ravel()])

    # Ideal image coords = world / mm_per_pixel
    ix = world_points[:, 0] / mm_per_pixel
    iy = world_points[:, 1] / mm_per_pixel

    image_shape = (int(ix.max() + 100), int(iy.max() + 100))  # (width, height)
    image_points = np.column_stack([ix, iy])

    return image_points, world_points, image_shape, mm_per_pixel


def _make_distorted_grid(
    nx=20, ny=20, spacing_mm=10.0, dx_coeffs=None, dy_coeffs=None
):
    """Create grid points with known polynomial distortion applied.

    Returns (image_points, world_points, image_shape, mm_per_pixel, dx_coeffs, dy_coeffs).
    """
    image_points, world_points, image_shape, mm_per_pixel = _make_grid_points(
        nx, ny, spacing_mm
    )
    width, height = image_shape
    x_origin = width / 2.0
    y_origin = height / 2.0

    # Default distortion: small but non-trivial
    if dx_coeffs is None:
        dx_coeffs = np.array([2.0, 0.5, -0.1, 0.01, 0.3, -0.05, 0.005, 0.02, -0.01, 0.008])
    if dy_coeffs is None:
        dy_coeffs = np.array([1.5, -0.3, 0.08, -0.005, 0.4, 0.1, -0.003, -0.015, 0.007, -0.006])

    # Compute normalized coords at ideal image positions
    s = 2.0 * (image_points[:, 0] - x_origin) / width
    t = 2.0 * (image_points[:, 1] - y_origin) / height

    # Apply distortion: distorted_pixel = ideal_pixel + polynomial(s, t)
    dx = evaluate_polynomial_terms(s, t, dx_coeffs)
    dy = evaluate_polynomial_terms(s, t, dy_coeffs)

    distorted_image_points = image_points.copy()
    distorted_image_points[:, 0] += dx
    distorted_image_points[:, 1] += dy

    return distorted_image_points, world_points, image_shape, mm_per_pixel, dx_coeffs, dy_coeffs


# ===========================================================================
# Test 1a: evaluate_polynomial_terms
# ===========================================================================

class TestEvaluatePolynomialTerms:
    """Direct tests for the polynomial term evaluation function."""

    def test_constant_term(self):
        """At s=0, t=0, only the constant term contributes."""
        coeffs = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        s = np.array([0.0])
        t = np.array([0.0])
        result = evaluate_polynomial_terms(s, t, coeffs)
        np.testing.assert_allclose(result, 5.0, atol=1e-14)

    def test_s_terms_at_s1_t0(self):
        """At s=1, t=0, only 1, s, s^2, s^3 terms contribute."""
        coeffs = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        s = np.array([1.0])
        t = np.array([0.0])
        # 1 + 2*1 + 3*1 + 4*1 = 10
        result = evaluate_polynomial_terms(s, t, coeffs)
        np.testing.assert_allclose(result, 10.0, atol=1e-14)

    def test_t_terms_at_s0_t1(self):
        """At s=0, t=1, only 1, t, t^2, t^3 terms contribute."""
        coeffs = np.array([1.0, 0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0])
        s = np.array([0.0])
        t = np.array([1.0])
        # 1 + 2*1 + 3*1 + 4*1 = 10
        result = evaluate_polynomial_terms(s, t, coeffs)
        np.testing.assert_allclose(result, 10.0, atol=1e-14)

    def test_cross_terms(self):
        """Test s*t, s^2*t, s*t^2 cross terms."""
        # Only cross terms: s*t, s^2*t, s*t^2
        coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        s = np.array([2.0])
        t = np.array([3.0])
        # 1*(2*3) + 2*(4*3) + 3*(2*9) = 6 + 24 + 54 = 84
        result = evaluate_polynomial_terms(s, t, coeffs)
        np.testing.assert_allclose(result, 84.0, atol=1e-14)

    def test_all_terms(self):
        """All 10 coefficients = 1, at s=1, t=1."""
        coeffs = np.ones(10)
        s = np.array([1.0])
        t = np.array([1.0])
        # Terms: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] = 10
        result = evaluate_polynomial_terms(s, t, coeffs)
        np.testing.assert_allclose(result, 10.0, atol=1e-14)

    def test_vectorized_evaluation(self):
        """Evaluate on multiple points simultaneously."""
        coeffs = np.array([1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        s = np.array([0.0, 1.0, -1.0])
        t = np.array([0.0, 0.0, 1.0])
        # s=0,t=0: 1
        # s=1,t=0: 1 + 2 = 3
        # s=-1,t=1: 1 + 2*(-1) + 3*1 = 2
        expected = np.array([1.0, 3.0, 2.0])
        result = evaluate_polynomial_terms(s, t, coeffs)
        np.testing.assert_allclose(result, expected, atol=1e-14)


# ===========================================================================
# Test 1b: fit_polynomial_from_points round-trip
# ===========================================================================

class TestFitPolynomialRoundTrip:
    """Verify that fitting recovers known distortion coefficients."""

    def test_zero_distortion_coefficients(self):
        """With no distortion, all coefficients should be near zero."""
        image_points, world_points, image_shape, mm_per_pixel = _make_grid_points()
        result = fit_polynomial_from_points(image_points, world_points, image_shape)

        np.testing.assert_allclose(result["coefficients_x"], np.zeros(10), atol=0.01)
        np.testing.assert_allclose(result["coefficients_y"], np.zeros(10), atol=0.01)
        assert abs(result["mm_per_pixel"] - mm_per_pixel) < 0.01
        assert result["rms_fit_error_px"] < 0.01

    def test_known_distortion_recovery(self):
        """Fit should recover known polynomial distortion coefficients.

        The mm_per_pixel estimation via median nearest-neighbor ratio absorbs
        some low-order distortion, shifting the constant (c[0]) and linear
        (c[1]) terms. Higher-order terms (c[2:]) recover well. This is
        expected and functionally harmless because the round-trip accuracy
        (test_coordinate_round_trip) is what matters operationally.
        """
        (
            distorted_pts, world_pts, img_shape, mm_per_pixel, dx_coeffs, dy_coeffs
        ) = _make_distorted_grid()

        result = fit_polynomial_from_points(distorted_pts, world_pts, img_shape)

        # mm_per_pixel bias absorbs into the constant (idx 0), linear-s (1),
        # and linear-t (4) terms — these are the terms proportional to
        # world coordinates which get scaled by 1/mm_per_pixel.
        # All other terms (quadratic+, cross-terms) should recover tightly.
        absorbed = [0, 1, 4]
        higher_order = [i for i in range(10) if i not in absorbed]

        cx = np.array(result["coefficients_x"])
        cy = np.array(result["coefficients_y"])

        np.testing.assert_allclose(
            cx[higher_order], dx_coeffs[higher_order], atol=0.01,
            err_msg="X higher-order coefficients not recovered",
        )
        np.testing.assert_allclose(
            cy[higher_order], dy_coeffs[higher_order], atol=0.01,
            err_msg="Y higher-order coefficients not recovered",
        )

        # Absorbed terms get wider tolerance
        np.testing.assert_allclose(
            cx[absorbed], dx_coeffs[absorbed], atol=0.5,
            err_msg="X absorbed coefficients too far off",
        )
        np.testing.assert_allclose(
            cy[absorbed], dy_coeffs[absorbed], atol=0.5,
            err_msg="Y absorbed coefficients too far off",
        )

        # mm_per_pixel should be close to ground truth
        assert abs(result["mm_per_pixel"] - mm_per_pixel) / mm_per_pixel < 0.05, (
            f"mm_per_pixel: got {result['mm_per_pixel']:.6f}, expected {mm_per_pixel}"
        )

        # RMS error should be near-zero for noise-free synthetic data
        assert result["rms_fit_error_px"] < 0.01, (
            f"RMS fit error {result['rms_fit_error_px']:.4f} px, expected ~0"
        )

    def test_coordinate_round_trip(self):
        """Fit + evaluate should recover world positions from distorted pixels."""
        (
            distorted_pts, world_pts, img_shape, mm_per_pixel_gt,
            dx_coeffs_gt, dy_coeffs_gt,
        ) = _make_distorted_grid()

        result = fit_polynomial_from_points(distorted_pts, world_pts, img_shape)

        # Reconstruct world coords using the fitted model
        width, height = img_shape
        x_origin = result["origin"]["x"]
        y_origin = result["origin"]["y"]
        nx = result["normalisation"]["nx"]
        ny = result["normalisation"]["ny"]
        mpp = result["mm_per_pixel"]
        cx = np.array(result["coefficients_x"])
        cy = np.array(result["coefficients_y"])

        # Normalize the distorted image points
        s = 2.0 * (distorted_pts[:, 0] - x_origin) / nx
        t = 2.0 * (distorted_pts[:, 1] - y_origin) / ny

        # Evaluate distortion
        dx = evaluate_polynomial_terms(s, t, cx)
        dy = evaluate_polynomial_terms(s, t, cy)

        # Back-map: x_world_px = x_raw - dx
        x_world_px = distorted_pts[:, 0] - dx
        y_world_px = distorted_pts[:, 1] - dy

        # Convert to mm
        x_mm = x_world_px * mpp
        y_mm = y_world_px * mpp

        # Compare to ground truth world coords
        np.testing.assert_allclose(x_mm, world_pts[:, 0], atol=0.5,
                                   err_msg="X world coord round-trip failed")
        np.testing.assert_allclose(y_mm, world_pts[:, 1], atol=0.5,
                                   err_msg="Y world coord round-trip failed")

    def test_minimum_points_error(self):
        """Fewer than 10 points should raise ValueError."""
        pts = np.random.rand(9, 2) * 100
        world = np.random.rand(9, 2) * 25
        with pytest.raises(ValueError, match="at least 10"):
            fit_polynomial_from_points(pts, world, (200, 200))

    def test_normalisation_values(self):
        """Origin should be at image centre, nx/ny should be image dimensions."""
        image_points, world_points, image_shape, _ = _make_grid_points()
        result = fit_polynomial_from_points(image_points, world_points, image_shape)

        width, height = image_shape
        assert abs(result["origin"]["x"] - width / 2.0) < 1e-10
        assert abs(result["origin"]["y"] - height / 2.0) < 1e-10
        assert abs(result["normalisation"]["nx"] - width) < 1e-10
        assert abs(result["normalisation"]["ny"] - height) < 1e-10


# ===========================================================================
# Test 1c: PolynomialVectorCalibrator coordinate calibration
# ===========================================================================

class TestCalibratorCoordinates:
    """Test calibrate_coordinates with explicit parameters (no config needed)."""

    def _make_calibrator(self, mm_per_pixel=0.25, image_width=1024.0,
                         image_height=768.0, dx_coeff=None, dy_coeff=None):
        """Create a calibrator with explicit parameters (no config file).

        image_height is passed as ny since the constructor derives
        image_height from ny when no config is present (ny > 1.0 fallback).
        """
        import tempfile
        base_dir = tempfile.mkdtemp()
        return PolynomialVectorCalibrator(
            base_dir=base_dir,
            camera_num=1,
            dt=0.001,
            mm_per_pixel=mm_per_pixel,
            dx_coeff=dx_coeff if dx_coeff is not None else np.zeros(10),
            dy_coeff=dy_coeff if dy_coeff is not None else np.zeros(10),
            x_origin=image_width / 2.0,
            y_origin=image_height / 2.0,
            nx=image_width,
            ny=image_height,
            config=None,
        )

    def test_zero_distortion_identity(self):
        """With zero distortion, calibrated coords = raw coords * mm_per_pixel.

        Uncal coords (1-based, y-up) are converted to raw (0-based, y-down)
        before applying the polynomial. With zero polynomial, the result is
        just raw * mm_per_pixel with y negated.
        """
        cal = self._make_calibrator(mm_per_pixel=0.25, image_width=1024.0, image_height=768.0)

        # Uncalibrated point at image centre: x_uncal=513, y_uncal=385
        # (1-based, y-up from bottom)
        x_uncal = np.array([513.0])
        y_uncal = np.array([385.0])

        x_mm, y_mm = cal.calibrate_coordinates(x_uncal, y_uncal)

        # Raw: x_raw = 513-1 = 512, y_raw = 768-385 = 383
        # World px (no distortion): same as raw
        # x_mm = 512 * 0.25 = 128.0
        # y_mm = -383 * 0.25 = -95.75
        np.testing.assert_allclose(x_mm, 128.0, atol=0.001)
        np.testing.assert_allclose(y_mm, -95.75, atol=0.001)

    def test_y_convention_negation(self):
        """y_mm should be negative of y_raw * mm_per_pixel."""
        cal = self._make_calibrator(mm_per_pixel=1.0, image_width=100.0, image_height=100.0)

        # Uncal y=1 → raw y = 100-1 = 99 → y_mm = -99
        # Uncal y=100 → raw y = 100-100 = 0 → y_mm = 0
        x_uncal = np.array([1.0, 1.0])
        y_uncal = np.array([1.0, 100.0])

        x_mm, y_mm = cal.calibrate_coordinates(x_uncal, y_uncal)

        np.testing.assert_allclose(y_mm[0], -99.0, atol=0.001)
        np.testing.assert_allclose(y_mm[1], 0.0, atol=0.001)


# ===========================================================================
# Test 1d: PolynomialVectorCalibrator vector calibration
# ===========================================================================

class TestCalibratorVectors:
    """Test calibrate_vectors velocity conversion and v-negation."""

    def _make_calibrator(self, mm_per_pixel=0.25, image_height=768.0, dt=0.001):
        import tempfile
        base_dir = tempfile.mkdtemp()
        return PolynomialVectorCalibrator(
            base_dir=base_dir,
            camera_num=1,
            dt=dt,
            mm_per_pixel=mm_per_pixel,
            dx_coeff=np.zeros(10),
            dy_coeff=np.zeros(10),
            x_origin=512.0,
            y_origin=384.0,
            nx=1024.0,
            ny=image_height,
            config=None,
        )

    def test_zero_velocity(self):
        """Zero displacement should give zero velocity."""
        cal = self._make_calibrator()
        u, v = cal.calibrate_vectors(
            np.array([[0.0]]), np.array([[0.0]]),
            np.array([[513.0]]), np.array([[385.0]]),
        )
        np.testing.assert_allclose(u, 0.0, atol=1e-10)
        np.testing.assert_allclose(v, 0.0, atol=1e-10)

    def test_u_positive_direction(self):
        """Positive ux_px should give positive u_ms (x is not negated)."""
        cal = self._make_calibrator(mm_per_pixel=1.0, dt=1.0)
        u, v = cal.calibrate_vectors(
            np.array([[10.0]]), np.array([[0.0]]),
            np.array([[513.0]]), np.array([[385.0]]),
        )
        # u = (10 px * 1.0 mm/px * 1e-3) / 1.0 = 0.01 m/s
        np.testing.assert_allclose(u, 0.01, atol=1e-6)

    def test_v_negation(self):
        """Positive uy_px in uncal (y-up) should give negative v_ms.

        In uncalibrated coords (y-up), positive uy means upward.
        After conversion to raw (y-down), this becomes negative displacement.
        The polynomial (in raw space) sees negative v_world_px.
        Then v_ms = -(v_world_mm * 1e-3) / dt, which should be positive
        for upward motion (because the raw-space displacement is negative
        and gets negated).

        Wait — let's trace through carefully:
          coords y_uncal=385, uy_px=+10 (upward in uncal)
          y1_uncal = 385+10 = 395
          y0_raw = 768-385 = 383
          y1_raw = 768-395 = 373
          v_world_px = y1_raw - y0_raw = 373-383 = -10 (downward in raw = upward physically)
          v_world_mm = -10 * mm_per_pixel
          v_ms = -(v_world_mm * 1e-3) / dt = -(-10 * 1.0 * 1e-3) / 1.0 = +0.01

        So positive uy_px (upward) → positive v_ms (upward). Good.
        """
        cal = self._make_calibrator(mm_per_pixel=1.0, dt=1.0)
        u, v = cal.calibrate_vectors(
            np.array([[0.0]]), np.array([[10.0]]),
            np.array([[513.0]]), np.array([[385.0]]),
        )
        # v_ms should be positive (upward motion stays upward)
        np.testing.assert_allclose(v, 0.01, atol=1e-6)

    def test_velocity_scaling(self):
        """Velocity should scale with mm_per_pixel and inversely with dt."""
        for mpp, dt in [(0.5, 0.001), (0.1, 0.01), (1.0, 0.005)]:
            cal = self._make_calibrator(mm_per_pixel=mpp, dt=dt)
            u, v = cal.calibrate_vectors(
                np.array([[1.0]]), np.array([[0.0]]),
                np.array([[513.0]]), np.array([[385.0]]),
            )
            expected_u = (1.0 * mpp * 1e-3) / dt
            np.testing.assert_allclose(u, expected_u, rtol=1e-6,
                                       err_msg=f"Failed for mpp={mpp}, dt={dt}")


# ===========================================================================
# Test 1e: Missing image_height edge case
# ===========================================================================

class TestMissingImageHeight:
    """Document behaviour when image_height is None."""

    def test_calibrate_coordinates_skips_uncal_conversion(self):
        """With image_height=None, uncal→raw conversion is skipped.

        The code should still run (not crash), but results differ from
        the correct path because the input coords are not converted
        from uncalibrated to raw convention.
        """
        import tempfile
        base_dir = tempfile.mkdtemp()

        # Force image_height=None by providing ny=1.0 (which triggers the
        # fallback path) and no config
        cal = PolynomialVectorCalibrator(
            base_dir=base_dir,
            camera_num=1,
            dt=0.001,
            mm_per_pixel=0.25,
            dx_coeff=np.zeros(10),
            dy_coeff=np.zeros(10),
            x_origin=512.0,
            y_origin=384.0,
            nx=1024.0,
            ny=1.0,  # <= 1.0, triggers fallback
            config=None,
        )

        # ny=1.0 but image_height falls back to ny when > 1.0. With ny=1.0,
        # falls through to config lookup which fails → image_height = None
        assert cal.image_height is None, (
            f"Expected image_height=None, got {cal.image_height}"
        )

        # Should not crash
        x_mm, y_mm = cal.calibrate_coordinates(np.array([100.0]), np.array([200.0]))
        assert np.isfinite(x_mm).all()
        assert np.isfinite(y_mm).all()

    def test_calibrate_vectors_skips_uncal_conversion(self):
        """Vectors also work (without crashing) when image_height is None."""
        import tempfile
        base_dir = tempfile.mkdtemp()

        cal = PolynomialVectorCalibrator(
            base_dir=base_dir,
            camera_num=1,
            dt=0.001,
            mm_per_pixel=0.25,
            dx_coeff=np.zeros(10),
            dy_coeff=np.zeros(10),
            x_origin=512.0,
            y_origin=384.0,
            nx=1024.0,
            ny=1.0,
            config=None,
        )

        assert cal.image_height is None

        u, v = cal.calibrate_vectors(
            np.array([[1.0]]), np.array([[1.0]]),
            np.array([[100.0]]), np.array([[200.0]]),
        )
        assert np.isfinite(u).all()
        assert np.isfinite(v).all()


# ===========================================================================
# Test 1f: save_polynomial_to_config / config round-trip
# ===========================================================================

class TestSavePolynomialToConfig:
    """Verify save_polynomial_to_config writes all required keys."""

    def test_config_structure(self, tmp_path):
        """After saving, config should have all polynomial calibration keys."""
        import yaml
        from pivtools_core.config import Config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"paths": {"base_paths": [str(tmp_path)]}}))
        cfg = Config(path=str(config_path))

        fit_result = {
            "mm_per_pixel": 0.25,
            "origin": {"x": 512.0, "y": 384.0},
            "normalisation": {"nx": 1024.0, "ny": 768.0},
            "coefficients_x": list(range(10)),
            "coefficients_y": list(range(10, 20)),
            "rms_fit_error_px": 0.1,
        }

        save_polynomial_to_config(
            camera_num=1, fit_result=fit_result, dt=0.001, config=cfg,
        )

        # Reload and verify
        cfg2 = Config(path=str(config_path))
        cam_params = cfg2.get_polynomial_camera_params(1)

        assert cam_params, "Camera params not found after save"
        assert abs(cam_params["mm_per_pixel"] - 0.25) < 1e-10
        assert cam_params["origin"]["x"] == 512.0
        assert cam_params["origin"]["y"] == 384.0
        assert cam_params["normalisation"]["nx"] == 1024.0
        assert cam_params["normalisation"]["ny"] == 768.0
        assert cam_params["coefficients_x"] == list(range(10))
        assert cam_params["coefficients_y"] == list(range(10, 20))
        assert cam_params["image_height"] == 768.0  # ny
        assert cfg2.polynomial_calibration.get("dt") == 0.001
        assert cfg2.data["calibration"]["active"] == "polynomial"

    def test_calibrator_from_saved_config(self, tmp_path):
        """PolynomialVectorCalibrator should load params from saved config."""
        import yaml
        from pivtools_core.config import Config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"paths": {"base_paths": [str(tmp_path)]}}))
        cfg = Config(path=str(config_path))

        dx_coeffs = [0.5, 0.1, -0.02, 0.001, 0.3, -0.05, 0.002, 0.01, -0.005, 0.003]
        dy_coeffs = [0.3, -0.2, 0.03, -0.002, 0.4, 0.06, -0.001, -0.02, 0.004, -0.002]

        fit_result = {
            "mm_per_pixel": 0.25,
            "origin": {"x": 512.0, "y": 384.0},
            "normalisation": {"nx": 1024.0, "ny": 768.0},
            "coefficients_x": dx_coeffs,
            "coefficients_y": dy_coeffs,
            "rms_fit_error_px": 0.1,
        }

        save_polynomial_to_config(camera_num=1, fit_result=fit_result, dt=0.001, config=cfg)

        # Reload config and create calibrator from it
        cfg2 = Config(path=str(config_path))
        cal = PolynomialVectorCalibrator(
            base_dir=str(tmp_path), camera_num=1, config=cfg2,
        )

        assert abs(cal.mm_per_pixel - 0.25) < 1e-10
        assert abs(cal.dt - 0.001) < 1e-10
        assert abs(cal.x_origin - 512.0) < 1e-10
        assert abs(cal.y_origin - 384.0) < 1e-10
        assert abs(cal.nx - 1024.0) < 1e-10
        assert abs(cal.ny - 768.0) < 1e-10
        assert abs(cal.image_height - 768.0) < 1e-10
        np.testing.assert_allclose(cal.dx_coeff, dx_coeffs, atol=1e-10)
        np.testing.assert_allclose(cal.dy_coeff, dy_coeffs, atol=1e-10)


# ===========================================================================
# Test 2: DaVis XML Reader
# ===========================================================================

class TestDavisXMLReader:
    """Tests for the DaVis XML import pipeline: coefficient mapping,
    XML parsing, and end-to-end coordinate calibration."""

    # ---- Test 1: convert_davis_coeffs_to_array with a_ prefix ----

    def test_convert_davis_coeffs_to_array_a_prefix(self):
        """Verify 10-element mapping with a_ prefix and cross-validate
        against evaluate_polynomial_terms."""
        coeff_dict = {
            'a_o': 10, 'a_s': 20, 'a_s2': 30, 'a_s3': 40,
            'a_t': 50, 'a_t2': 60, 'a_t3': 70,
            'a_st': 80, 'a_s2t': 90, 'a_st2': 100,
        }
        result = convert_davis_coeffs_to_array(coeff_dict)
        expected = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        np.testing.assert_array_equal(result, expected)

        # Cross-validate: at s=1, t=0 only constant + s-terms are active
        s = np.array([1.0])
        t = np.array([0.0])
        val = evaluate_polynomial_terms(s, t, result)
        # 10 + 20*1 + 30*1 + 40*1 = 100  (t-terms and cross-terms are zero)
        np.testing.assert_allclose(val, 100.0, atol=1e-14)

    # ---- Test 2: convert_davis_coeffs_to_array with b_ prefix ----

    def test_convert_davis_coeffs_to_array_b_prefix(self):
        """Same mapping but with b_ prefix — both prefixes use identical order."""
        coeff_dict = {
            'b_o': 10, 'b_s': 20, 'b_s2': 30, 'b_s3': 40,
            'b_t': 50, 'b_t2': 60, 'b_t3': 70,
            'b_st': 80, 'b_s2t': 90, 'b_st2': 100,
        }
        result = convert_davis_coeffs_to_array(coeff_dict)
        expected = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        np.testing.assert_array_equal(result, expected)

    # ---- Test 3: missing keys default to 0.0 ----

    def test_convert_davis_coeffs_missing_keys(self):
        """Partial dict — missing keys should default to 0.0."""
        coeff_dict = {'a_o': 5.0, 'a_s': 3.0}
        result = convert_davis_coeffs_to_array(coeff_dict)
        expected = np.zeros(10)
        expected[0] = 5.0
        expected[1] = 3.0
        np.testing.assert_array_equal(result, expected)

    # ---- Test 4: read_calibration_xml ----

    def test_read_calibration_xml(self, tmp_path):
        """Parse a synthetic Calibration.xml and verify all fields."""
        xml_content = """\
<?xml version="1.0" encoding="utf-8"?>
<CalibrationData>
  <CoordinateMapper CameraIdentifier="Camera1">
    <PolynomialParameters>
      <CommonParameters>
        <PixelPerMmFactor Value="4.0"/>
      </CommonParameters>
      <PolynomialMapping>
        <Origin s_o="500.0" t_o="400.0"/>
        <NormalisationFactor nx="1000.0" ny="800.0"/>
        <Polynomial3rdOrder>
          <CoefficientsA a_o="1.5" a_s="0.01" a_s2="0.0" a_s3="0.0" a_t="0.02" a_t2="0.0" a_t3="0.0" a_st="0.0" a_s2t="0.0" a_st2="0.0"/>
          <CoefficientsB b_o="2.5" b_s="0.0" b_s2="0.0" b_s3="0.0" b_t="0.03" b_t2="0.0" b_t3="0.0" b_st="0.0" b_s2t="0.0" b_st2="0.0"/>
        </Polynomial3rdOrder>
      </PolynomialMapping>
    </PolynomialParameters>
  </CoordinateMapper>
</CalibrationData>
"""
        xml_file = tmp_path / "Calibration.xml"
        xml_file.write_text(xml_content)

        result = read_calibration_xml(xml_path=str(xml_file))

        assert result["status"] == "success"
        assert "Camera1" in result["cameras"]

        cam = result["cameras"]["Camera1"]

        # mm_per_pixel = 1 / PixelPerMmFactor = 0.25
        assert abs(cam["mm_per_pixel"] - 0.25) < 1e-10

        # Origin
        assert cam["origin"]["s_o"] == 500.0
        assert cam["origin"]["t_o"] == 400.0

        # Normalisation
        assert cam["normalisation"]["nx"] == 1000.0
        assert cam["normalisation"]["ny"] == 800.0

        # Coefficients A
        assert cam["coefficients_a"]["a_o"] == 1.5
        assert cam["coefficients_a"]["a_s"] == 0.01
        assert cam["coefficients_a"]["a_t"] == 0.02

        # Coefficients B
        assert cam["coefficients_b"]["b_o"] == 2.5
        assert cam["coefficients_b"]["b_t"] == 0.03

    # ---- Test 5: XML → calibrator coordinate conventions ----

    def test_xml_to_calibrator_coordinate_conventions(self):
        """Full pipeline: XML params → PolynomialVectorCalibrator →
        calibrate_coordinates with uncalibrated convention inputs.

        Verifies the coordinate convention chain:
          uncal (1-based, y-up) → raw (0-based, y-down) → polynomial → mm
        including the Y-negation on output.
        """
        import tempfile
        base_dir = tempfile.mkdtemp()

        # Identity mapping: zero polynomial coefficients, zero origin
        cal = PolynomialVectorCalibrator(
            base_dir=base_dir,
            camera_num=1,
            dt=0.001,
            mm_per_pixel=0.25,
            dx_coeff=np.zeros(10),
            dy_coeff=np.zeros(10),
            x_origin=0.0,
            y_origin=0.0,
            nx=1000.0,
            ny=800.0,   # image height
            config=None,
        )

        # Verify image_height is set from ny
        assert cal.image_height == 800.0

        # Test point: raw pixel (200, 300)
        # → uncalibrated: x_uncal = 200+1 = 201, y_uncal = 800 - 300 = 500
        x_uncal = np.array([201.0])
        y_uncal = np.array([500.0])

        x_mm, y_mm = cal.calibrate_coordinates(x_uncal, y_uncal)

        # _uncal_to_raw: x_raw = 201-1 = 200, y_raw = 800-500 = 300
        # Polynomial is zero → x_world_px = 200, y_world_px = 300
        # x_mm = 200 * 0.25 = 50.0
        # y_mm = -300 * 0.25 = -75.0  (Y-negation!)
        np.testing.assert_allclose(x_mm, 50.0, atol=1e-10)
        np.testing.assert_allclose(y_mm, -75.0, atol=1e-10)

        # Second test point at opposite vertical extreme:
        # raw pixel (200, 600) → uncal: (201, 200)
        x_uncal2 = np.array([201.0])
        y_uncal2 = np.array([200.0])

        x_mm2, y_mm2 = cal.calibrate_coordinates(x_uncal2, y_uncal2)

        # y_raw = 800 - 200 = 600 → y_mm = -600 * 0.25 = -150.0
        np.testing.assert_allclose(x_mm2, 50.0, atol=1e-10)
        np.testing.assert_allclose(y_mm2, -150.0, atol=1e-10)

        # Lower in image (larger y_raw) → more negative y_mm: consistent
        assert y_mm2 < y_mm

    # ---- Test 6: multi-camera XML ----

    def test_xml_multi_camera(self, tmp_path):
        """XML with two cameras should parse both with distinct coefficients."""
        xml_content = """\
<?xml version="1.0" encoding="utf-8"?>
<CalibrationData>
  <CoordinateMapper CameraIdentifier="Camera1">
    <PolynomialParameters>
      <CommonParameters>
        <PixelPerMmFactor Value="4.0"/>
      </CommonParameters>
      <PolynomialMapping>
        <Origin s_o="500.0" t_o="400.0"/>
        <NormalisationFactor nx="1000.0" ny="800.0"/>
        <Polynomial3rdOrder>
          <CoefficientsA a_o="1.5" a_s="0.0" a_s2="0.0" a_s3="0.0" a_t="0.0" a_t2="0.0" a_t3="0.0" a_st="0.0" a_s2t="0.0" a_st2="0.0"/>
          <CoefficientsB b_o="2.5" b_s="0.0" b_s2="0.0" b_s3="0.0" b_t="0.0" b_t2="0.0" b_t3="0.0" b_st="0.0" b_s2t="0.0" b_st2="0.0"/>
        </Polynomial3rdOrder>
      </PolynomialMapping>
    </PolynomialParameters>
  </CoordinateMapper>
  <CoordinateMapper CameraIdentifier="Camera2">
    <PolynomialParameters>
      <CommonParameters>
        <PixelPerMmFactor Value="8.0"/>
      </CommonParameters>
      <PolynomialMapping>
        <Origin s_o="600.0" t_o="500.0"/>
        <NormalisationFactor nx="1200.0" ny="900.0"/>
        <Polynomial3rdOrder>
          <CoefficientsA a_o="3.0" a_s="0.05" a_s2="0.0" a_s3="0.0" a_t="0.0" a_t2="0.0" a_t3="0.0" a_st="0.0" a_s2t="0.0" a_st2="0.0"/>
          <CoefficientsB b_o="4.0" b_s="0.0" b_s2="0.0" b_s3="0.0" b_t="0.06" b_t2="0.0" b_t3="0.0" b_st="0.0" b_s2t="0.0" b_st2="0.0"/>
        </Polynomial3rdOrder>
      </PolynomialMapping>
    </PolynomialParameters>
  </CoordinateMapper>
</CalibrationData>
"""
        xml_file = tmp_path / "Calibration.xml"
        xml_file.write_text(xml_content)

        result = read_calibration_xml(xml_path=str(xml_file))

        assert result["status"] == "success"
        assert "Camera1" in result["cameras"]
        assert "Camera2" in result["cameras"]

        cam1 = result["cameras"]["Camera1"]
        cam2 = result["cameras"]["Camera2"]

        # Distinct mm_per_pixel
        assert abs(cam1["mm_per_pixel"] - 0.25) < 1e-10   # 1/4
        assert abs(cam2["mm_per_pixel"] - 0.125) < 1e-10  # 1/8

        # Distinct origins
        assert cam1["origin"]["s_o"] == 500.0
        assert cam2["origin"]["s_o"] == 600.0

        # Distinct coefficients
        assert cam1["coefficients_a"]["a_o"] == 1.5
        assert cam2["coefficients_a"]["a_o"] == 3.0
        assert cam2["coefficients_a"]["a_s"] == 0.05

        assert cam1["coefficients_b"]["b_o"] == 2.5
        assert cam2["coefficients_b"]["b_o"] == 4.0
        assert cam2["coefficients_b"]["b_t"] == 0.06
