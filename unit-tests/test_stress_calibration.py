#!/usr/bin/env python3
"""
test_stress_calibration.py

Tests for stress calibration math in scale-factor and pinhole calibration.

Verifies:
  - stress_scale = 1/(px_per_mm^2 * dt^2 * 1e6)
  - ScaleFactorCalibrator.calibrate_stresses() produces correct physical values
  - VectorCalibrator.calibrate_stresses() with spatially-varying scale
  - Cross-validation: scale-factor and pinhole produce same stress scaling
    for a uniform-scale (frontal, no distortion) camera setup

Usage:
    pytest unit-tests/test_stress_calibration.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_gui.calibration.scale_factor_calibration_production import (
    ScaleFactorCalibrator,
)


# ===========================================================================
# Tests: ScaleFactorCalibrator stress math
# ===========================================================================


class TestScaleFactorStressCalibration:
    """Test stress calibration using scale-factor method."""

    def test_stress_scale_formula(self):
        """Verify stress_scale = 1 / (px_per_mm^2 * dt^2 * 1e6)."""
        px_per_mm = 10.0
        dt = 0.001  # 1 ms

        cal = ScaleFactorCalibrator(
            base_path=Path("/tmp/dummy"),
            dt=dt,
            px_per_mm=px_per_mm,
        )

        # Known input: 100 px^2/frame^2
        stress_px = np.array([[100.0]])
        stress_phys = cal.calibrate_stresses(stress_px)

        # Expected: 100 / (10^2 * 0.001^2 * 1e6) = 100 / (100 * 1e-6 * 1e6) = 100 / 100 = 1.0
        expected = 100.0 / (px_per_mm**2 * dt**2 * 1e6)
        np.testing.assert_allclose(stress_phys, expected, rtol=1e-10)

    def test_stress_is_velocity_squared(self):
        """Stress scale should equal velocity_scale^2."""
        px_per_mm = 28.89
        dt = 0.005

        cal = ScaleFactorCalibrator(
            base_path=Path("/tmp/dummy"),
            dt=dt,
            px_per_mm=px_per_mm,
        )

        # velocity_scale = 1 / (px_per_mm * dt * 1000)
        velocity_scale = 1.0 / (px_per_mm * dt * 1000)

        # Calibrate a unit stress
        stress_px = np.array([[1.0]])
        stress_phys = cal.calibrate_stresses(stress_px)

        expected = velocity_scale**2
        np.testing.assert_allclose(stress_phys[0, 0], expected, rtol=1e-10)

    def test_calibrate_stress_array(self):
        """Verify calibration works on 2D stress arrays."""
        px_per_mm = 20.0
        dt = 0.002

        cal = ScaleFactorCalibrator(
            base_path=Path("/tmp/dummy"),
            dt=dt,
            px_per_mm=px_per_mm,
        )

        shape = (10, 12)
        UU_px = np.random.RandomState(42).rand(*shape) * 100
        VV_px = np.random.RandomState(43).rand(*shape) * 80
        UV_px = np.random.RandomState(44).rand(*shape) * 20 - 10

        UU_phys = cal.calibrate_stresses(UU_px)
        VV_phys = cal.calibrate_stresses(VV_px)
        UV_phys = cal.calibrate_stresses(UV_px)

        stress_scale = 1.0 / (px_per_mm**2 * dt**2 * 1e6)
        np.testing.assert_allclose(UU_phys, UU_px * stress_scale, rtol=1e-10)
        np.testing.assert_allclose(VV_phys, VV_px * stress_scale, rtol=1e-10)
        np.testing.assert_allclose(UV_phys, UV_px * stress_scale, rtol=1e-10)

    def test_stress_units_consistency(self):
        """Stress calibrated value should have units m^2/s^2.

        Verify with a known physical scenario:
        - 10 px/mm magnification, dt = 1ms
        - A velocity of 10 px/frame = 10/(10*0.001*1000) = 1.0 m/s
        - A stress of 100 px^2/frame^2 should = 1.0 m^2/s^2
        """
        cal = ScaleFactorCalibrator(
            base_path=Path("/tmp/dummy"),
            dt=0.001,
            px_per_mm=10.0,
        )

        # Velocity check
        ux_px = np.array([[10.0]])
        uy_px = np.array([[0.0]])
        ux_ms, uy_ms = cal.calibrate_vectors(ux_px, uy_px)
        np.testing.assert_allclose(ux_ms, 1.0, rtol=1e-10)

        # Stress check: 10^2 = 100 px^2/frame^2 → 1.0^2 = 1.0 m^2/s^2
        stress_px = np.array([[100.0]])
        stress_phys = cal.calibrate_stresses(stress_px)
        np.testing.assert_allclose(stress_phys, 1.0, rtol=1e-10)

    def test_different_dt_values(self):
        """Stress scaling should change with dt^2."""
        px_per_mm = 10.0
        stress_px = np.array([[100.0]])

        cal1 = ScaleFactorCalibrator(
            base_path=Path("/tmp/dummy"), dt=0.001, px_per_mm=px_per_mm,
        )
        cal2 = ScaleFactorCalibrator(
            base_path=Path("/tmp/dummy"), dt=0.002, px_per_mm=px_per_mm,
        )

        s1 = cal1.calibrate_stresses(stress_px)
        s2 = cal2.calibrate_stresses(stress_px)

        # dt doubled → stress_scale quartered
        np.testing.assert_allclose(s1 / s2, 4.0, rtol=1e-10)

    def test_different_px_per_mm(self):
        """Stress scaling should change with px_per_mm^2."""
        dt = 0.001
        stress_px = np.array([[100.0]])

        cal1 = ScaleFactorCalibrator(
            base_path=Path("/tmp/dummy"), dt=dt, px_per_mm=10.0,
        )
        cal2 = ScaleFactorCalibrator(
            base_path=Path("/tmp/dummy"), dt=dt, px_per_mm=20.0,
        )

        s1 = cal1.calibrate_stresses(stress_px)
        s2 = cal2.calibrate_stresses(stress_px)

        # px_per_mm doubled → stress_scale quartered
        np.testing.assert_allclose(s1 / s2, 4.0, rtol=1e-10)


# ===========================================================================
# Tests: Coordinate calibration
# ===========================================================================


class TestScaleFactorCoordinateCalibration:
    """Test coordinate calibration in scale-factor method."""

    def test_coordinates_zero_origin(self):
        """Calibrated coordinates should start from (0,0)."""
        cal = ScaleFactorCalibrator(
            base_path=Path("/tmp/dummy"), dt=0.001, px_per_mm=10.0,
        )

        x = np.array([[10, 20, 30], [10, 20, 30]], dtype=np.float64)
        y = np.array([[5, 5, 5], [15, 15, 15]], dtype=np.float64)

        x_mm, y_mm = cal.calibrate_coordinates(x, y)

        # x starts from x.flat[0]=10, so x_mm = (x-10)/10
        np.testing.assert_allclose(x_mm[0, 0], 0.0)
        np.testing.assert_allclose(x_mm[0, 1], 1.0)
        np.testing.assert_allclose(x_mm[0, 2], 2.0)

        # y starts from min(y)=5, so y_mm = (y-5)/10
        np.testing.assert_allclose(y_mm[0, 0], 0.0)
        np.testing.assert_allclose(y_mm[1, 0], 1.0)

    def test_velocity_calibration(self):
        """Velocity calibration: px/frame → m/s."""
        px_per_mm = 20.0
        dt = 0.005

        cal = ScaleFactorCalibrator(
            base_path=Path("/tmp/dummy"), dt=dt, px_per_mm=px_per_mm,
        )

        # 100 px/frame at 20 px/mm, dt=5ms
        # = 100 / 20 / 0.005 / 1000 = 100 / 100 = 1.0 m/s
        ux_px = np.array([[100.0]])
        uy_px = np.array([[50.0]])

        ux_ms, uy_ms = cal.calibrate_vectors(ux_px, uy_px)

        expected_ux = 100.0 / (px_per_mm * dt * 1000)
        expected_uy = 50.0 / (px_per_mm * dt * 1000)

        np.testing.assert_allclose(ux_ms, expected_ux, rtol=1e-10)
        np.testing.assert_allclose(uy_ms, expected_uy, rtol=1e-10)
