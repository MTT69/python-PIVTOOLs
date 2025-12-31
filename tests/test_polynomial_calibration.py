#!/usr/bin/env python3
"""
test_polynomial_calibration.py

Comprehensive tests for DAVIS polynomial calibration.

Tests validate:
    - evaluate_polynomial_terms() function with known inputs
    - convert_davis_coeffs_to_array() coefficient parsing
    - Coordinate calibration (pixels -> mm with polynomial correction)
    - Vector calibration (pixels/frame -> m/s)
    - Identity transform (zero coefficients)
    - Simple transforms (single coefficients)

Usage:
    python test_polynomial_calibration.py           # Run all tests
    python test_polynomial_calibration.py --unit    # Unit tests only
    python test_polynomial_calibration.py --verbose # Detailed output
"""

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ===================== TEST RESULT DATACLASS =====================


@dataclass
class TestResult:
    """Result of a single test case."""

    name: str
    passed: bool
    checks: List[Dict]
    message: str = ""


# ===================== REFERENCE FORMULAS =====================


def evaluate_polynomial_reference(s: float, t: float, coeffs: np.ndarray) -> float:
    """
    Reference implementation of DAVIS polynomial evaluation.

    Coefficient order: [1, s, s², s³, t, t², t³, s*t, s²*t, s*t²]

    Args:
        s: Normalized x coordinate
        t: Normalized y coordinate
        coeffs: 10 polynomial coefficients

    Returns:
        Polynomial value (dx or dy)
    """
    result = (
        coeffs[0] * 1.0 +        # constant
        coeffs[1] * s +          # s
        coeffs[2] * s**2 +       # s²
        coeffs[3] * s**3 +       # s³
        coeffs[4] * t +          # t
        coeffs[5] * t**2 +       # t²
        coeffs[6] * t**3 +       # t³
        coeffs[7] * s * t +      # s*t
        coeffs[8] * s**2 * t +   # s²*t
        coeffs[9] * s * t**2     # s*t²
    )
    return result


def expected_coordinate_mm(
    x_px: float, y_px: float, dx_coeffs: np.ndarray, dy_coeffs: np.ndarray,
    x_origin: float, y_origin: float, nx: float, ny: float, mm_per_pixel: float
) -> Tuple[float, float]:
    """
    Calculate expected calibrated coordinate.

    Args:
        x_px, y_px: Pixel coordinates
        dx_coeffs, dy_coeffs: Polynomial coefficients
        x_origin, y_origin: Normalization origins
        nx, ny: Normalization factors
        mm_per_pixel: Scale factor

    Returns:
        (x_mm, y_mm): Calibrated coordinates in mm
    """
    # Normalize
    s = 2 * (x_px - x_origin) / nx
    t = 2 * (y_px - y_origin) / ny

    # Evaluate polynomial
    dx = evaluate_polynomial_reference(s, t, dx_coeffs)
    dy = evaluate_polynomial_reference(s, t, dy_coeffs)

    # Apply correction
    x_world_px = x_px - dx
    y_world_px = y_px - dy

    # Convert to mm
    x_mm = x_world_px * mm_per_pixel
    y_mm = y_world_px * mm_per_pixel

    return x_mm, y_mm


# ===================== UNIT TESTS =====================


class UnitTests:
    """Direct calibration formula and function tests."""

    def __init__(self, rtol: float = 1e-10, verbose: bool = True):
        self.rtol = rtol
        self.verbose = verbose

    def _check(self, name: str, expected: float, computed: float) -> Dict:
        """Check a single value against expected."""
        if abs(expected) > 1e-10:
            passed = abs(computed - expected) / abs(expected) <= self.rtol
        else:
            passed = abs(computed - expected) <= self.rtol
        return {
            "name": name,
            "expected": expected,
            "computed": computed,
            "passed": passed,
        }

    def test_polynomial_identity(self) -> TestResult:
        """Test that zero coefficients give identity transform (no correction)."""
        from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
            evaluate_polynomial_terms,
        )

        # Zero coefficients should give zero output
        coeffs = np.zeros(10)
        s = np.array([0.5, -0.3, 1.0])
        t = np.array([0.2, 0.8, -0.5])

        result = evaluate_polynomial_terms(s, t, coeffs)

        checks = [
            self._check("zero_coeffs_0", 0.0, result[0]),
            self._check("zero_coeffs_1", 0.0, result[1]),
            self._check("zero_coeffs_2", 0.0, result[2]),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Polynomial Identity (Zero Coefficients)", passed, checks)

    def test_polynomial_constant_term(self) -> TestResult:
        """Test polynomial with only constant term (uniform offset)."""
        from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
            evaluate_polynomial_terms,
        )

        # Only constant term (index 0)
        coeffs = np.zeros(10)
        coeffs[0] = 5.0  # Constant offset

        s = np.array([0.0, 0.5, 1.0])
        t = np.array([0.0, 0.5, 1.0])

        result = evaluate_polynomial_terms(s, t, coeffs)

        # All outputs should be 5.0 regardless of s, t
        checks = [
            self._check("constant_at_origin", 5.0, result[0]),
            self._check("constant_at_0.5", 5.0, result[1]),
            self._check("constant_at_1.0", 5.0, result[2]),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Polynomial Constant Term Only", passed, checks)

    def test_polynomial_linear_s(self) -> TestResult:
        """Test polynomial with only linear s term."""
        from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
            evaluate_polynomial_terms,
        )

        # Only linear s term (index 1)
        coeffs = np.zeros(10)
        coeffs[1] = 2.0  # Linear s coefficient

        s = np.array([0.0, 0.5, 1.0, -1.0])
        t = np.array([0.0, 0.0, 0.0, 0.0])

        result = evaluate_polynomial_terms(s, t, coeffs)

        # Output should be 2.0 * s
        expected = 2.0 * s
        checks = [
            self._check(f"linear_s_{i}", expected[i], result[i])
            for i in range(len(s))
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Polynomial Linear S Term Only", passed, checks)

    def test_polynomial_quadratic_t(self) -> TestResult:
        """Test polynomial with only quadratic t term."""
        from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
            evaluate_polynomial_terms,
        )

        # Only t² term (index 5)
        coeffs = np.zeros(10)
        coeffs[5] = 3.0  # t² coefficient

        s = np.array([0.0, 0.0, 0.0, 0.0])
        t = np.array([0.0, 0.5, 1.0, -1.0])

        result = evaluate_polynomial_terms(s, t, coeffs)

        # Output should be 3.0 * t²
        expected = 3.0 * t ** 2
        checks = [
            self._check(f"quad_t_{i}", expected[i], result[i])
            for i in range(len(t))
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Polynomial Quadratic T Term Only", passed, checks)

    def test_polynomial_cross_term(self) -> TestResult:
        """Test polynomial with s*t cross term."""
        from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
            evaluate_polynomial_terms,
        )

        # Only s*t term (index 7)
        coeffs = np.zeros(10)
        coeffs[7] = 4.0  # s*t coefficient

        s = np.array([1.0, 2.0, -1.0, 0.5])
        t = np.array([1.0, 0.5, 2.0, -2.0])

        result = evaluate_polynomial_terms(s, t, coeffs)

        # Output should be 4.0 * s * t
        expected = 4.0 * s * t
        checks = [
            self._check(f"cross_term_{i}", expected[i], result[i])
            for i in range(len(s))
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Polynomial Cross Term (s*t)", passed, checks)

    def test_polynomial_full_evaluation(self) -> TestResult:
        """Test full polynomial evaluation against reference implementation."""
        from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
            evaluate_polynomial_terms,
        )

        # Full set of non-zero coefficients
        coeffs = np.array([1.0, 2.0, 0.5, 0.1, 1.5, 0.3, 0.05, 0.8, 0.2, 0.15])

        s_vals = np.array([0.0, 0.5, -0.5, 1.0])
        t_vals = np.array([0.0, 0.3, 0.7, -0.2])

        result = evaluate_polynomial_terms(s_vals, t_vals, coeffs)

        # Compare against reference implementation
        expected = np.array([
            evaluate_polynomial_reference(s_vals[i], t_vals[i], coeffs)
            for i in range(len(s_vals))
        ])

        checks = [
            self._check(f"full_poly_{i}", expected[i], result[i])
            for i in range(len(s_vals))
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Polynomial Full Evaluation", passed, checks)

    def test_coefficient_conversion(self) -> TestResult:
        """Test conversion of DAVIS coefficient dictionary to array."""
        from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
            convert_davis_coeffs_to_array,
        )

        # Test with 'a_' prefix
        coeff_dict_a = {
            'a_o': 1.0, 'a_s': 2.0, 'a_s2': 3.0, 'a_s3': 4.0,
            'a_t': 5.0, 'a_t2': 6.0, 'a_t3': 7.0,
            'a_st': 8.0, 'a_s2t': 9.0, 'a_st2': 10.0
        }

        result = convert_davis_coeffs_to_array(coeff_dict_a)

        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        checks = [
            self._check(f"coeff_{i}", expected[i], result[i])
            for i in range(10)
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Coefficient Conversion (a_ prefix)", passed, checks)

    def test_coefficient_conversion_b_prefix(self) -> TestResult:
        """Test conversion with 'b_' prefix."""
        from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
            convert_davis_coeffs_to_array,
        )

        # Test with 'b_' prefix
        coeff_dict_b = {
            'b_o': 10.0, 'b_s': 20.0, 'b_s2': 30.0, 'b_s3': 40.0,
            'b_t': 50.0, 'b_t2': 60.0, 'b_t3': 70.0,
            'b_st': 80.0, 'b_s2t': 90.0, 'b_st2': 100.0
        }

        result = convert_davis_coeffs_to_array(coeff_dict_b)

        expected = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])

        checks = [
            self._check(f"coeff_{i}", expected[i], result[i])
            for i in range(10)
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Coefficient Conversion (b_ prefix)", passed, checks)

    def test_coordinate_calibration_identity(self) -> TestResult:
        """Test coordinate calibration with identity transform (zero coefficients)."""
        from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
            PolynomialVectorCalibrator,
        )

        # Create calibrator with zero coefficients (identity transform)
        with tempfile.TemporaryDirectory() as temp_dir:
            calibrator = PolynomialVectorCalibrator(
                base_dir=Path(temp_dir),
                camera_num=1,
                dt=0.001,
                mm_per_pixel=0.1,  # 0.1 mm per pixel
                dx_coeff=np.zeros(10),
                dy_coeff=np.zeros(10),
                x_origin=0.0,
                y_origin=0.0,
                nx=1000.0,  # Large normalization to make s, t small
                ny=1000.0,
            )

            # Test coordinates
            x_px = np.array([[100.0, 200.0], [100.0, 200.0]])
            y_px = np.array([[50.0, 50.0], [100.0, 100.0]])

            x_mm, y_mm = calibrator.calibrate_coordinates(x_px, y_px)

            # With zero coefficients, x_mm = x_px * mm_per_pixel
            expected_x = x_px * 0.1
            expected_y = y_px * 0.1

            checks = [
                self._check("x_mm_00", expected_x[0, 0], x_mm[0, 0]),
                self._check("x_mm_01", expected_x[0, 1], x_mm[0, 1]),
                self._check("y_mm_00", expected_y[0, 0], y_mm[0, 0]),
                self._check("y_mm_10", expected_y[1, 0], y_mm[1, 0]),
            ]

            passed = all(c["passed"] for c in checks)
            return TestResult("Coordinate Calibration Identity", passed, checks)

    def test_coordinate_calibration_constant_offset(self) -> TestResult:
        """Test coordinate calibration with constant offset."""
        from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
            PolynomialVectorCalibrator,
        )

        # Create calibrator with constant offset
        dx_coeff = np.zeros(10)
        dx_coeff[0] = 10.0  # Constant offset of 10 pixels in x
        dy_coeff = np.zeros(10)
        dy_coeff[0] = 5.0   # Constant offset of 5 pixels in y

        with tempfile.TemporaryDirectory() as temp_dir:
            calibrator = PolynomialVectorCalibrator(
                base_dir=Path(temp_dir),
                camera_num=1,
                dt=0.001,
                mm_per_pixel=0.1,
                dx_coeff=dx_coeff,
                dy_coeff=dy_coeff,
                x_origin=0.0,
                y_origin=0.0,
                nx=1000.0,
                ny=1000.0,
            )

            # Test single point
            x_px = np.array([[100.0]])
            y_px = np.array([[50.0]])

            x_mm, y_mm = calibrator.calibrate_coordinates(x_px, y_px)

            # x_world = x_px - dx = 100 - 10 = 90, x_mm = 90 * 0.1 = 9.0
            # y_world = y_px - dy = 50 - 5 = 45, y_mm = 45 * 0.1 = 4.5
            expected_x = (100.0 - 10.0) * 0.1
            expected_y = (50.0 - 5.0) * 0.1

            checks = [
                self._check("x_mm", expected_x, x_mm[0, 0]),
                self._check("y_mm", expected_y, y_mm[0, 0]),
            ]

            passed = all(c["passed"] for c in checks)
            return TestResult("Coordinate Calibration Constant Offset", passed, checks)

    def test_velocity_calibration_identity(self) -> TestResult:
        """Test velocity calibration with identity transform."""
        from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
            PolynomialVectorCalibrator,
        )

        # Create calibrator with zero coefficients
        with tempfile.TemporaryDirectory() as temp_dir:
            mm_per_pixel = 0.1
            dt = 0.001  # 1 ms

            calibrator = PolynomialVectorCalibrator(
                base_dir=Path(temp_dir),
                camera_num=1,
                dt=dt,
                mm_per_pixel=mm_per_pixel,
                dx_coeff=np.zeros(10),
                dy_coeff=np.zeros(10),
                x_origin=0.0,
                y_origin=0.0,
                nx=1000.0,
                ny=1000.0,
            )

            # Test uniform velocity field
            shape = (5, 5)
            ux_px = np.full(shape, 10.0)  # 10 pixels/frame
            uy_px = np.full(shape, 5.0)   # 5 pixels/frame
            coords_x = np.tile(np.arange(5) * 16.0, (5, 1))
            coords_y = np.tile(np.arange(5).reshape(-1, 1) * 16.0, (1, 5))

            ux_ms, uy_ms = calibrator.calibrate_vectors(ux_px, uy_px, coords_x, coords_y)

            # With identity transform:
            # u_world_px = ux_px (no distortion change)
            # u_world_mm = u_world_px * mm_per_pixel = 10 * 0.1 = 1 mm
            # u_ms = u_world_mm * 1e-3 / dt = 1e-3 / 0.001 = 1 m/s
            expected_ux = 10.0 * mm_per_pixel * 1e-3 / dt  # = 1.0 m/s
            expected_uy = 5.0 * mm_per_pixel * 1e-3 / dt   # = 0.5 m/s

            checks = [
                self._check("mean_ux_ms", expected_ux, np.mean(ux_ms)),
                self._check("mean_uy_ms", expected_uy, np.mean(uy_ms)),
            ]

            # Check uniformity
            checks.extend([
                {"name": "std_ux", "expected": "<1e-10", "computed": np.std(ux_ms), "passed": np.std(ux_ms) < 1e-10},
                {"name": "std_uy", "expected": "<1e-10", "computed": np.std(uy_ms), "passed": np.std(uy_ms) < 1e-10},
            ])

            passed = all(c["passed"] for c in checks)
            return TestResult("Velocity Calibration Identity", passed, checks)

    def run_all(self) -> List[TestResult]:
        """Run all unit tests."""
        tests = [
            self.test_polynomial_identity,
            self.test_polynomial_constant_term,
            self.test_polynomial_linear_s,
            self.test_polynomial_quadratic_t,
            self.test_polynomial_cross_term,
            self.test_polynomial_full_evaluation,
            self.test_coefficient_conversion,
            self.test_coefficient_conversion_b_prefix,
            self.test_coordinate_calibration_identity,
            self.test_coordinate_calibration_constant_offset,
            self.test_velocity_calibration_identity,
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                results.append(
                    TestResult(test.__name__, False, [], f"Exception: {str(e)}")
                )

        return results


# ===================== CLI TESTS =====================


class CLITests:
    """Tests for CLI script validity."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def test_module_imports(self) -> TestResult:
        """Test that the polynomial calibration module can be imported."""
        checks = []

        try:
            from pivtools_gui.calibration.calibration_poly import polynomial_calibration_production
            checks.append({"name": "module_import", "expected": True, "computed": True, "passed": True})
        except ImportError as e:
            checks.append({"name": "module_import", "expected": True, "computed": False, "passed": False})
            return TestResult("Module Import", False, checks, str(e))

        # Check key functions exist
        required_attrs = [
            "evaluate_polynomial_terms",
            "convert_davis_coeffs_to_array",
            "PolynomialVectorCalibrator",
            "read_calibration_xml",
        ]

        for attr in required_attrs:
            has_attr = hasattr(polynomial_calibration_production, attr)
            checks.append({
                "name": f"has_{attr}",
                "expected": True,
                "computed": has_attr,
                "passed": has_attr,
            })

        passed = all(c["passed"] for c in checks)
        return TestResult("Module Import and Attributes", passed, checks)

    def test_script_syntax(self) -> TestResult:
        """Test that the script has valid Python syntax."""
        import ast

        script_path = (
            Path(__file__).parent.parent /
            "pivtools_gui" / "calibration" / "calibration_poly" /
            "polynomial_calibration_production.py"
        )

        checks = []

        if not script_path.exists():
            checks.append({
                "name": "file_exists",
                "expected": True,
                "computed": False,
                "passed": False,
            })
            return TestResult("Script Syntax", False, checks, f"File not found: {script_path}")

        try:
            with open(script_path, "r") as f:
                source = f.read()
            ast.parse(source)
            checks.append({
                "name": "valid_syntax",
                "expected": True,
                "computed": True,
                "passed": True,
            })
        except SyntaxError as e:
            checks.append({
                "name": "valid_syntax",
                "expected": True,
                "computed": False,
                "passed": False,
            })
            return TestResult("Script Syntax", False, checks, str(e))

        passed = all(c["passed"] for c in checks)
        return TestResult("Script Syntax", passed, checks)

    def run_all(self) -> List[TestResult]:
        """Run all CLI tests."""
        tests = [
            self.test_module_imports,
            self.test_script_syntax,
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                results.append(
                    TestResult(test.__name__, False, [], f"Exception: {str(e)}")
                )

        return results


# ===================== TEST RUNNER =====================


def print_result(result: TestResult, verbose: bool = True):
    """Print a single test result."""
    status = "\u2713" if result.passed else "\u2717"
    print(f"  [{status}] {result.name}")

    if verbose and result.checks:
        for check in result.checks:
            check_status = "\u2713" if check["passed"] else "\u2717"
            print(f"      [{check_status}] {check['name']}: expected={check['expected']}, computed={check['computed']}")

    if result.message:
        print(f"      Message: {result.message}")


def print_summary(results: List[TestResult]):
    """Print test summary."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passed}/{total} tests passed")

    if passed < total:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Polynomial Calibration Tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--cli", action="store_true", help="Run CLI tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--rtol", type=float, default=1e-10, help="Relative tolerance")

    args = parser.parse_args()

    # If no specific test type selected, run all
    run_all = not (args.unit or args.cli)

    all_results = []

    # Unit tests
    if args.unit or run_all:
        print("\n" + "=" * 60)
        print("UNIT TESTS")
        print("=" * 60)
        unit_tests = UnitTests(rtol=args.rtol, verbose=args.verbose)
        results = unit_tests.run_all()
        all_results.extend(results)
        for r in results:
            print_result(r, args.verbose)

    # CLI tests
    if args.cli or run_all:
        print("\n" + "=" * 60)
        print("CLI TESTS")
        print("=" * 60)
        cli_tests = CLITests(verbose=args.verbose)
        results = cli_tests.run_all()
        all_results.extend(results)
        for r in results:
            print_result(r, args.verbose)

    # Summary
    print_summary(all_results)

    # Exit code
    all_passed = all(r.passed for r in all_results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
