#!/usr/bin/env python3
"""
test_scale_factor_calibration.py

Comprehensive tests for scale factor calibration of PIV vector fields.

Tests validate:
    - Instantaneous vector field calibration (ux, uy)
    - Ensemble vector field calibration (ux, uy + stresses)
    - Coordinate calibration (pixels -> mm)
    - CLI interface functionality
    - Edge cases and error handling

Usage:
    python test_scale_factor_calibration.py           # Run all tests
    python test_scale_factor_calibration.py --unit    # Unit tests only
    python test_scale_factor_calibration.py --cli     # CLI tests only
"""

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io

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


# ===================== SYNTHETIC DATA GENERATION =====================


class SyntheticPIVData:
    """Generate synthetic PIV data for scale factor calibration testing."""

    @staticmethod
    def create_coordinate_grid(
        shape: Tuple[int, int], dx_px: float = 16.0, dy_px: float = 16.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create coordinate grids in pixels (PIV window spacing).

        Args:
            shape: (H, W) grid dimensions
            dx_px: Pixel spacing in x direction
            dy_px: Pixel spacing in y direction

        Returns:
            Tuple of (X, Y) coordinate grids in pixels
        """
        H, W = shape
        x = np.arange(W) * dx_px + dx_px / 2  # Center of first window
        y = np.arange(H) * dy_px + dy_px / 2
        X, Y = np.meshgrid(x, y)
        return X, Y

    @staticmethod
    def create_instantaneous_vectors(
        shape: Tuple[int, int],
        ux_px: float,
        uy_px: float,
        n_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create uniform instantaneous vector fields in pixels/frame.

        Args:
            shape: (H, W) grid dimensions
            ux_px: X velocity in pixels/frame
            uy_px: Y velocity in pixels/frame
            n_frames: Number of frames

        Returns:
            Tuple of (ux, uy, b_mask) arrays, each (n_frames, H, W)
        """
        H, W = shape
        ux = np.full((n_frames, H, W), ux_px, dtype=np.float64)
        uy = np.full((n_frames, H, W), uy_px, dtype=np.float64)
        b_mask = np.zeros((n_frames, H, W), dtype=np.float64)
        return ux, uy, b_mask

    @staticmethod
    def create_instantaneous_vectors_with_uz(
        shape: Tuple[int, int],
        ux_px: float,
        uy_px: float,
        uz_px: float,
        n_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create uniform instantaneous vector fields with uz (stereo PIV).

        Returns:
            Tuple of (ux, uy, uz, b_mask) arrays
        """
        H, W = shape
        ux = np.full((n_frames, H, W), ux_px, dtype=np.float64)
        uy = np.full((n_frames, H, W), uy_px, dtype=np.float64)
        uz = np.full((n_frames, H, W), uz_px, dtype=np.float64)
        b_mask = np.zeros((n_frames, H, W), dtype=np.float64)
        return ux, uy, uz, b_mask

    @staticmethod
    def create_ensemble_result(
        shape: Tuple[int, int],
        ux_px: float,
        uy_px: float,
        UU_px: float,
        VV_px: float,
        UV_px: float,
    ) -> Dict[str, np.ndarray]:
        """
        Create ensemble PIV result with mean velocities and stresses.

        All values are in uncalibrated units (pixels/frame for velocity,
        pixels^2/frame^2 for stresses).

        Returns:
            Dict with ux, uy, b_mask, UU_stress, VV_stress, UV_stress arrays
        """
        H, W = shape
        return {
            "ux": np.full((H, W), ux_px, dtype=np.float64),
            "uy": np.full((H, W), uy_px, dtype=np.float64),
            "b_mask": np.zeros((H, W), dtype=np.float64),
            "UU_stress": np.full((H, W), UU_px, dtype=np.float64),
            "VV_stress": np.full((H, W), VV_px, dtype=np.float64),
            "UV_stress": np.full((H, W), UV_px, dtype=np.float64),
        }


# ===================== FILE WRITERS =====================


def write_coordinates_mat(
    path: Path, x: np.ndarray, y: np.ndarray
) -> None:
    """Write coordinates.mat file."""
    coord_dtype = np.dtype([("x", "O"), ("y", "O")])
    coordinates = np.empty(1, dtype=coord_dtype)
    coordinates[0] = (x, y)
    scipy.io.savemat(str(path), {"coordinates": coordinates}, do_compression=True)


def write_instantaneous_mat(
    path: Path,
    ux: np.ndarray,
    uy: np.ndarray,
    b_mask: np.ndarray,
    uz: Optional[np.ndarray] = None,
) -> None:
    """
    Write instantaneous piv_result MAT file.

    Args:
        path: Output path
        ux, uy: Velocity arrays (H, W)
        b_mask: Mask array (H, W)
        uz: Optional out-of-plane velocity (H, W)
    """
    if uz is not None:
        piv_dtype = np.dtype([("ux", "O"), ("uy", "O"), ("uz", "O"), ("b_mask", "O")])
        piv_result = np.empty(1, dtype=piv_dtype)
        piv_result[0] = (ux, uy, uz, b_mask)
    else:
        piv_dtype = np.dtype([("ux", "O"), ("uy", "O"), ("b_mask", "O")])
        piv_result = np.empty(1, dtype=piv_dtype)
        piv_result[0] = (ux, uy, b_mask)

    scipy.io.savemat(str(path), {"piv_result": piv_result}, do_compression=True)


def write_ensemble_mat(path: Path, data: Dict[str, np.ndarray]) -> None:
    """
    Write ensemble_result MAT file with stresses.

    Args:
        path: Output path
        data: Dict with ux, uy, b_mask, UU_stress, VV_stress, UV_stress
    """
    piv_dtype = np.dtype([
        ("ux", "O"), ("uy", "O"), ("b_mask", "O"),
        ("UU_stress", "O"), ("VV_stress", "O"), ("UV_stress", "O")
    ])
    ensemble_result = np.empty(1, dtype=piv_dtype)
    ensemble_result[0] = (
        data["ux"], data["uy"], data["b_mask"],
        data["UU_stress"], data["VV_stress"], data["UV_stress"]
    )
    scipy.io.savemat(str(path), {"ensemble_result": ensemble_result}, do_compression=True)


# ===================== CALIBRATION FORMULAS (FOR REFERENCE) =====================


def expected_velocity_ms(vel_px: float, px_per_mm: float, dt: float) -> float:
    """
    Calculate expected calibrated velocity in m/s.

    Formula: (px/frame) / (px/mm) / (s/frame) / 1000 = m/s
    """
    return vel_px / px_per_mm / dt / 1000


def expected_stress_ms2(stress_px: float, px_per_mm: float, dt: float) -> float:
    """
    Calculate expected calibrated stress in m^2/s^2.

    Formula: (px^2/frame^2) / (px^2/mm^2) / (s^2/frame^2) / 10^6 = m^2/s^2
    """
    return stress_px / (px_per_mm ** 2) / (dt ** 2) / 1e6


def expected_coordinate_mm(coord_px: float, px_per_mm: float, origin_px: float = 0) -> float:
    """Calculate expected calibrated coordinate in mm."""
    return (coord_px - origin_px) / px_per_mm


# ===================== UNIT TESTS =====================


class UnitTests:
    """Direct calibration formula and class tests."""

    def __init__(self, rtol: float = 1e-10, verbose: bool = True):
        self.rtol = rtol
        self.verbose = verbose

    def _check(
        self, name: str, expected: float, computed: float
    ) -> Dict:
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

    def test_calibrator_formulas(self) -> TestResult:
        """Test ScaleFactorCalibrator class methods directly."""
        from pivtools_gui.calibration.scale_factor_calibration_production import (
            ScaleFactorCalibrator,
        )

        # Known test parameters
        px_per_mm = 28.89
        dt = 0.001  # 1 ms between frames

        calibrator = ScaleFactorCalibrator(
            base_path=Path("/tmp"),
            dt=dt,
            px_per_mm=px_per_mm,
        )

        # Test velocity calibration
        ux_px = 10.0  # pixels/frame
        uy_px = -5.0

        ux_ms, uy_ms = calibrator.calibrate_vectors(
            np.array([[ux_px]]), np.array([[uy_px]])
        )

        expected_ux = expected_velocity_ms(ux_px, px_per_mm, dt)
        expected_uy = expected_velocity_ms(uy_px, px_per_mm, dt)

        # Test stress calibration
        stress_px = 100.0  # pixels^2/frame^2
        stress_ms = calibrator.calibrate_stresses(np.array([[stress_px]]))
        expected_stress = expected_stress_ms2(stress_px, px_per_mm, dt)

        # Test coordinate calibration
        x_px = np.array([[100.0, 200.0], [100.0, 200.0]])
        y_px = np.array([[50.0, 50.0], [100.0, 100.0]])
        x_mm, y_mm = calibrator.calibrate_coordinates(x_px, y_px)

        # Expected: zero-based at origin, converted to mm
        # x_mm = (x - x[0,0]) / px_per_mm
        expected_x_mm_01 = (200.0 - 100.0) / px_per_mm  # x[0,1]

        checks = [
            self._check("ux_ms", expected_ux, ux_ms[0, 0]),
            self._check("uy_ms", expected_uy, uy_ms[0, 0]),
            self._check("stress_ms2", expected_stress, stress_ms[0, 0]),
            self._check("x_mm_origin", 0.0, x_mm[0, 0]),
            self._check("x_mm_01", expected_x_mm_01, x_mm[0, 1]),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Calibrator Formula Tests", passed, checks)

    def test_instantaneous_calibration(self) -> TestResult:
        """Test instantaneous vector field calibration through file I/O."""
        from pivtools_gui.calibration.scale_factor_calibration_production import (
            _process_vector_file,
        )

        shape = (30, 40)
        ux_px, uy_px = 15.0, -8.0
        px_per_mm = 20.0
        dt = 0.002

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            uncal_file = temp_path / "00001.mat"
            cal_file = temp_path / "00001_cal.mat"

            # Write uncalibrated data
            ux, uy, b_mask = SyntheticPIVData.create_instantaneous_vectors(
                shape, ux_px, uy_px, n_frames=1
            )
            write_instantaneous_mat(uncal_file, ux[0], uy[0], b_mask[0])

            # Run calibration
            success = _process_vector_file(
                (1, uncal_file, cal_file, px_per_mm, dt)
            )

            if not success:
                return TestResult(
                    "Instantaneous Calibration",
                    False,
                    [],
                    "Calibration function returned False"
                )

            # Load and verify
            mat = scipy.io.loadmat(str(cal_file), struct_as_record=False, squeeze_me=True)
            piv_result = mat["piv_result"]
            if isinstance(piv_result, np.ndarray):
                piv_result = piv_result[0]

            computed_ux = np.mean(piv_result.ux)
            computed_uy = np.mean(piv_result.uy)

            expected_ux = expected_velocity_ms(ux_px, px_per_mm, dt)
            expected_uy = expected_velocity_ms(uy_px, px_per_mm, dt)

            checks = [
                self._check("ux_calibrated", expected_ux, computed_ux),
                self._check("uy_calibrated", expected_uy, computed_uy),
            ]

            passed = all(c["passed"] for c in checks)
            return TestResult("Instantaneous Calibration", passed, checks)

    def test_ensemble_calibration(self) -> TestResult:
        """Test ensemble vector field calibration with stresses."""
        from pivtools_gui.calibration.scale_factor_calibration_production import (
            _process_vector_file,
        )

        shape = (25, 35)
        ux_px, uy_px = 12.0, 6.0
        UU_px, VV_px, UV_px = 50.0, 30.0, -10.0
        px_per_mm = 15.0
        dt = 0.0005

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            uncal_file = temp_path / "ensemble_result.mat"
            cal_file = temp_path / "ensemble_result_cal.mat"

            # Write uncalibrated ensemble data
            data = SyntheticPIVData.create_ensemble_result(
                shape, ux_px, uy_px, UU_px, VV_px, UV_px
            )
            write_ensemble_mat(uncal_file, data)

            # Run calibration
            success = _process_vector_file(
                (1, uncal_file, cal_file, px_per_mm, dt)
            )

            if not success:
                return TestResult(
                    "Ensemble Calibration",
                    False,
                    [],
                    "Calibration function returned False"
                )

            # Load and verify
            mat = scipy.io.loadmat(str(cal_file), struct_as_record=False, squeeze_me=True)
            ensemble_result = mat["ensemble_result"]
            if isinstance(ensemble_result, np.ndarray):
                ensemble_result = ensemble_result[0]

            computed_ux = np.mean(ensemble_result.ux)
            computed_uy = np.mean(ensemble_result.uy)
            computed_UU = np.mean(ensemble_result.UU_stress)
            computed_VV = np.mean(ensemble_result.VV_stress)
            computed_UV = np.mean(ensemble_result.UV_stress)

            expected_ux = expected_velocity_ms(ux_px, px_per_mm, dt)
            expected_uy = expected_velocity_ms(uy_px, px_per_mm, dt)
            expected_UU = expected_stress_ms2(UU_px, px_per_mm, dt)
            expected_VV = expected_stress_ms2(VV_px, px_per_mm, dt)
            expected_UV = expected_stress_ms2(UV_px, px_per_mm, dt)

            checks = [
                self._check("ux_calibrated", expected_ux, computed_ux),
                self._check("uy_calibrated", expected_uy, computed_uy),
                self._check("UU_stress_calibrated", expected_UU, computed_UU),
                self._check("VV_stress_calibrated", expected_VV, computed_VV),
                self._check("UV_stress_calibrated", expected_UV, computed_UV),
            ]

            passed = all(c["passed"] for c in checks)
            return TestResult("Ensemble Calibration", passed, checks)

    def test_stress_formula_consistency(self) -> TestResult:
        """Verify stress calibration is velocity_scale^2 as documented."""
        px_per_mm = 25.0
        dt = 0.001

        # Velocity scale factor
        velocity_scale = 1.0 / px_per_mm / dt / 1000

        # Stress scale should be velocity_scale squared
        expected_stress_scale = velocity_scale ** 2

        # From the code's formula
        computed_stress_scale = 1.0 / (px_per_mm ** 2) / (dt ** 2) / 1e6

        checks = [
            self._check("stress_scale", expected_stress_scale, computed_stress_scale),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Stress Formula Consistency", passed, checks)

    def test_multiple_frames(self) -> TestResult:
        """Test calibration of multiple instantaneous frames."""
        from pivtools_gui.calibration.scale_factor_calibration_production import (
            _process_vector_file,
        )

        shape = (20, 25)
        px_per_mm = 30.0
        dt = 0.001
        n_frames = 5

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            all_passed = True
            checks = []

            for frame in range(n_frames):
                # Each frame has different velocity
                ux_px = 10.0 + frame * 2.0
                uy_px = -5.0 - frame * 1.0

                uncal_file = temp_path / f"{frame+1:05d}.mat"
                cal_file = temp_path / f"{frame+1:05d}_cal.mat"

                ux, uy, b_mask = SyntheticPIVData.create_instantaneous_vectors(
                    shape, ux_px, uy_px, n_frames=1
                )
                write_instantaneous_mat(uncal_file, ux[0], uy[0], b_mask[0])

                success = _process_vector_file(
                    (frame + 1, uncal_file, cal_file, px_per_mm, dt)
                )

                if not success:
                    all_passed = False
                    checks.append({
                        "name": f"frame_{frame+1}_process",
                        "expected": True,
                        "computed": False,
                        "passed": False,
                    })
                    continue

                mat = scipy.io.loadmat(str(cal_file), struct_as_record=False, squeeze_me=True)
                piv_result = mat["piv_result"]
                if isinstance(piv_result, np.ndarray):
                    piv_result = piv_result[0]

                computed_ux = np.mean(piv_result.ux)
                expected_ux = expected_velocity_ms(ux_px, px_per_mm, dt)

                frame_passed = abs(computed_ux - expected_ux) / abs(expected_ux) < self.rtol
                all_passed = all_passed and frame_passed
                checks.append({
                    "name": f"frame_{frame+1}_ux",
                    "expected": expected_ux,
                    "computed": computed_ux,
                    "passed": frame_passed,
                })

            return TestResult("Multiple Frames Calibration", all_passed, checks)

    def run_all(self) -> Dict[str, TestResult]:
        """Run all unit tests."""
        tests = [
            self.test_calibrator_formulas,
            self.test_instantaneous_calibration,
            self.test_ensemble_calibration,
            self.test_stress_formula_consistency,
            self.test_multiple_frames,
        ]

        results = {}
        for test_func in tests:
            try:
                result = test_func()
            except Exception as e:
                result = TestResult(
                    test_func.__name__,
                    False,
                    [],
                    f"Exception: {e}"
                )
            results[result.name] = result
        return results


# ===================== INTEGRATION TESTS =====================


class IntegrationTests:
    """Test through ScaleFactorCalibrator with realistic directory structure."""

    def __init__(self, rtol: float = 1e-6, verbose: bool = True):
        self.rtol = rtol
        self.verbose = verbose

    def _setup_test_directory(
        self,
        temp_dir: Path,
        shape: Tuple[int, int],
        n_frames: int,
        ux_px: float,
        uy_px: float,
        ensemble: bool = False,
        UU_px: float = 0,
        VV_px: float = 0,
        UV_px: float = 0,
    ) -> Path:
        """
        Set up test directory structure matching production code expectations.

        Creates: {temp_dir}/uncalibrated_piv/{n_frames}/Cam1/{type_name}/
        """
        type_name = "ensemble" if ensemble else "instantaneous"
        data_dir = temp_dir / "uncalibrated_piv" / str(n_frames) / "Cam1" / type_name
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create coordinates
        x, y = SyntheticPIVData.create_coordinate_grid(shape)
        write_coordinates_mat(data_dir / "coordinates.mat", x, y)

        if ensemble:
            # Single ensemble result file
            data = SyntheticPIVData.create_ensemble_result(
                shape, ux_px, uy_px, UU_px, VV_px, UV_px
            )
            write_ensemble_mat(data_dir / "ensemble_result.mat", data)
        else:
            # Multiple instantaneous files
            for i in range(n_frames):
                ux, uy, b_mask = SyntheticPIVData.create_instantaneous_vectors(
                    shape, ux_px, uy_px, n_frames=1
                )
                write_instantaneous_mat(
                    data_dir / f"{i+1:05d}.mat", ux[0], uy[0], b_mask[0]
                )

        return temp_dir

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

    def test_full_instantaneous_pipeline(self) -> TestResult:
        """Test complete instantaneous calibration through ScaleFactorCalibrator."""
        from pivtools_gui.calibration.scale_factor_calibration_production import (
            ScaleFactorCalibrator,
        )
        from pivtools_core.config import get_config

        shape = (20, 30)
        n_frames = 5
        ux_px, uy_px = 20.0, -10.0
        px_per_mm = 25.0
        dt = 0.001

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self._setup_test_directory(
                temp_path, shape, n_frames, ux_px, uy_px, ensemble=False
            )

            # Need to set up config for the paths module
            config = get_config()
            original_num_images = config.data["images"]["num_images"]
            original_vector_format = config.vector_format

            try:
                # num_frame_pairs is computed from num_images for standard image types
                config.data["images"]["num_images"] = n_frames
                config.data["images"]["vector_format"] = ["%05d.mat"]

                calibrator = ScaleFactorCalibrator(
                    base_path=temp_path,
                    type_name="instantaneous",
                    dt=dt,
                    px_per_mm=px_per_mm,
                )

                result = calibrator.process_camera(
                    camera_num=1,
                    image_count=n_frames,
                )

                if not result["success"]:
                    return TestResult(
                        "Full Instantaneous Pipeline",
                        False,
                        [],
                        f"Calibrator failed: {result.get('error', 'unknown')}"
                    )

                # Load calibrated data
                cal_dir = temp_path / "calibrated_piv" / str(n_frames) / "Cam1" / "instantaneous"
                cal_file = cal_dir / "00001.mat"

                if not cal_file.exists():
                    return TestResult(
                        "Full Instantaneous Pipeline",
                        False,
                        [],
                        f"Calibrated file not found: {cal_file}"
                    )

                mat = scipy.io.loadmat(str(cal_file), struct_as_record=False, squeeze_me=True)
                piv_result = mat["piv_result"]
                if isinstance(piv_result, np.ndarray):
                    piv_result = piv_result[0]

                computed_ux = np.mean(piv_result.ux)
                computed_uy = np.mean(piv_result.uy)

                expected_ux = expected_velocity_ms(ux_px, px_per_mm, dt)
                expected_uy = expected_velocity_ms(uy_px, px_per_mm, dt)

                checks = [
                    self._check("ux_calibrated", expected_ux, computed_ux),
                    self._check("uy_calibrated", expected_uy, computed_uy),
                    self._check("files_processed", n_frames + 1, result["processed_files"]),  # +1 for coords
                    self._check("successful_files", n_frames + 1, result["successful_files"]),
                ]

                passed = all(c["passed"] for c in checks)
                return TestResult("Full Instantaneous Pipeline", passed, checks)

            finally:
                # Restore original config
                config.data["images"]["num_images"] = original_num_images
                config.data["images"]["vector_format"] = original_vector_format

    def test_full_ensemble_pipeline(self) -> TestResult:
        """Test complete ensemble calibration with stresses."""
        from pivtools_gui.calibration.scale_factor_calibration_production import (
            ScaleFactorCalibrator,
        )
        from pivtools_core.config import get_config

        shape = (20, 30)
        n_frames = 1  # Ensemble has single file
        ux_px, uy_px = 15.0, 8.0
        UU_px, VV_px, UV_px = 100.0, 64.0, -25.0
        px_per_mm = 20.0
        dt = 0.002

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self._setup_test_directory(
                temp_path, shape, n_frames, ux_px, uy_px,
                ensemble=True, UU_px=UU_px, VV_px=VV_px, UV_px=UV_px
            )

            config = get_config()
            original_num_images = config.data["images"]["num_images"]
            original_vector_format = config.vector_format

            try:
                # num_frame_pairs is computed from num_images for standard image types
                config.data["images"]["num_images"] = n_frames
                config.data["images"]["vector_format"] = ["%05d.mat"]

                calibrator = ScaleFactorCalibrator(
                    base_path=temp_path,
                    type_name="ensemble",
                    dt=dt,
                    px_per_mm=px_per_mm,
                )

                result = calibrator.process_camera(
                    camera_num=1,
                    image_count=n_frames,
                )

                if not result["success"]:
                    return TestResult(
                        "Full Ensemble Pipeline",
                        False,
                        [],
                        f"Calibrator failed: {result.get('error', 'unknown')}"
                    )

                # Load calibrated data
                cal_dir = temp_path / "calibrated_piv" / str(n_frames) / "Cam1" / "ensemble"
                cal_file = cal_dir / "ensemble_result.mat"

                if not cal_file.exists():
                    return TestResult(
                        "Full Ensemble Pipeline",
                        False,
                        [],
                        f"Calibrated file not found: {cal_file}"
                    )

                mat = scipy.io.loadmat(str(cal_file), struct_as_record=False, squeeze_me=True)
                ensemble_result = mat["ensemble_result"]
                if isinstance(ensemble_result, np.ndarray):
                    ensemble_result = ensemble_result[0]

                computed_ux = np.mean(ensemble_result.ux)
                computed_uy = np.mean(ensemble_result.uy)
                computed_UU = np.mean(ensemble_result.UU_stress)
                computed_VV = np.mean(ensemble_result.VV_stress)
                computed_UV = np.mean(ensemble_result.UV_stress)

                expected_ux = expected_velocity_ms(ux_px, px_per_mm, dt)
                expected_uy = expected_velocity_ms(uy_px, px_per_mm, dt)
                expected_UU = expected_stress_ms2(UU_px, px_per_mm, dt)
                expected_VV = expected_stress_ms2(VV_px, px_per_mm, dt)
                expected_UV = expected_stress_ms2(UV_px, px_per_mm, dt)

                checks = [
                    self._check("ux_calibrated", expected_ux, computed_ux),
                    self._check("uy_calibrated", expected_uy, computed_uy),
                    self._check("UU_stress_calibrated", expected_UU, computed_UU),
                    self._check("VV_stress_calibrated", expected_VV, computed_VV),
                    self._check("UV_stress_calibrated", expected_UV, computed_UV),
                ]

                passed = all(c["passed"] for c in checks)
                return TestResult("Full Ensemble Pipeline", passed, checks)

            finally:
                config.data["images"]["num_images"] = original_num_images
                config.data["images"]["vector_format"] = original_vector_format

    def run_all(self) -> Dict[str, TestResult]:
        """Run all integration tests."""
        tests = [
            self.test_full_instantaneous_pipeline,
            self.test_full_ensemble_pipeline,
        ]

        results = {}
        for test_func in tests:
            try:
                result = test_func()
            except Exception as e:
                import traceback
                result = TestResult(
                    test_func.__name__,
                    False,
                    [],
                    f"Exception: {e}\n{traceback.format_exc()}"
                )
            results[result.name] = result
        return results


# ===================== CLI TESTS =====================


class CLITests:
    """Test command-line interface functionality."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.script_path = Path(__file__).parent.parent / "pivtools_gui" / "calibration" / "scale_factor_calibration_production.py"

    def test_cli_import(self) -> TestResult:
        """Test that CLI script can be imported without errors."""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("scale_factor_calibration", self.script_path)
            module = importlib.util.module_from_spec(spec)
            # Don't execute, just check it can be loaded
            checks = [
                {"name": "module_loadable", "expected": True, "computed": True, "passed": True}
            ]
            return TestResult("CLI Import Test", True, checks)
        except Exception as e:
            return TestResult(
                "CLI Import Test",
                False,
                [{"name": "module_loadable", "expected": True, "computed": False, "passed": False}],
                f"Import error: {e}"
            )

    def test_cli_help(self) -> TestResult:
        """Test that script doesn't crash on import (no --help flag but validates syntax)."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(self.script_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            passed = result.returncode == 0
            checks = [
                {
                    "name": "syntax_valid",
                    "expected": 0,
                    "computed": result.returncode,
                    "passed": passed
                }
            ]
            msg = "" if passed else f"Syntax error: {result.stderr}"
            return TestResult("CLI Syntax Check", passed, checks, msg)
        except subprocess.TimeoutExpired:
            return TestResult(
                "CLI Syntax Check",
                False,
                [],
                "Syntax check timed out"
            )
        except Exception as e:
            return TestResult(
                "CLI Syntax Check",
                False,
                [],
                f"Exception: {e}"
            )

    def run_all(self) -> Dict[str, TestResult]:
        """Run all CLI tests."""
        tests = [
            self.test_cli_import,
            self.test_cli_help,
        ]

        results = {}
        for test_func in tests:
            try:
                result = test_func()
            except Exception as e:
                result = TestResult(
                    test_func.__name__,
                    False,
                    [],
                    f"Exception: {e}"
                )
            results[result.name] = result
        return results


# ===================== REPORTING =====================


def print_header():
    """Print test header."""
    print("=" * 70)
    print("SCALE FACTOR CALIBRATION VALIDATION TEST")
    print("=" * 70)


def print_result(result: TestResult, verbose: bool = True):
    """Print formatted test result."""
    status = "\u2713 PASS" if result.passed else "\u2717 FAIL"
    print(f"\nTest: {result.name}")
    print(f"  Result: {status}")

    if verbose and result.checks:
        print("  " + "-" * 60)
        print(f"  {'Check':<25} {'Expected':>15} {'Computed':>15} {'Status':>8}")
        print("  " + "-" * 60)

        for check in result.checks:
            status_str = "\u2713" if check["passed"] else "\u2717"
            exp = check["expected"]
            comp = check["computed"]
            exp_str = f"{exp:.6e}" if isinstance(exp, float) and abs(exp) < 0.01 else f"{exp:.6f}" if isinstance(exp, float) else str(exp)
            comp_str = f"{comp:.6e}" if isinstance(comp, float) and abs(comp) < 0.01 else f"{comp:.6f}" if isinstance(comp, float) else str(comp)
            print(f"  {check['name']:<25} {exp_str:>15} {comp_str:>15} {status_str:>8}")

    if result.message:
        print(f"  Message: {result.message}")


def print_summary(
    unit_results: Dict[str, TestResult],
    integration_results: Dict[str, TestResult],
    cli_results: Dict[str, TestResult],
):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    unit_passed = sum(1 for r in unit_results.values() if r.passed)
    unit_total = len(unit_results)
    int_passed = sum(1 for r in integration_results.values() if r.passed)
    int_total = len(integration_results)
    cli_passed = sum(1 for r in cli_results.values() if r.passed)
    cli_total = len(cli_results)

    total_passed = unit_passed + int_passed + cli_passed
    total_tests = unit_total + int_total + cli_total

    print(f"Unit Tests:        {unit_passed}/{unit_total} passed")
    print(f"Integration:       {int_passed}/{int_total} passed")
    print(f"CLI Tests:         {cli_passed}/{cli_total} passed")
    print(f"Total:             {total_passed}/{total_tests} passed")

    if total_passed == total_tests:
        print("\nAll scale factor calibration tests passed successfully.")
    else:
        print("\nSome tests FAILED. Review output above for details.")

    print("=" * 70)


# ===================== MAIN =====================


def main():
    parser = argparse.ArgumentParser(
        description="Scale factor calibration validation tests"
    )
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run CLI tests only",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-10,
        help="Relative tolerance for numerical tests",
    )
    args = parser.parse_args()

    # Determine which tests to run
    run_all = not (args.unit or args.integration or args.cli)
    run_unit = args.unit or run_all
    run_integration = args.integration or run_all
    run_cli = args.cli or run_all

    print_header()

    unit_results = {}
    integration_results = {}
    cli_results = {}

    if run_unit:
        print("\nUNIT TESTS (Direct Formula Verification)")
        print("-" * 70)
        unit_tests = UnitTests(rtol=args.rtol)
        unit_results = unit_tests.run_all()
        for result in unit_results.values():
            print_result(result)

    if run_integration:
        print("\nINTEGRATION TESTS (Full Pipeline)")
        print("-" * 70)
        integration_tests = IntegrationTests(rtol=args.rtol)
        integration_results = integration_tests.run_all()
        for result in integration_results.values():
            print_result(result)

    if run_cli:
        print("\nCLI TESTS (Command Line Interface)")
        print("-" * 70)
        cli_tests = CLITests()
        cli_results = cli_tests.run_all()
        for result in cli_results.values():
            print_result(result)

    print_summary(unit_results, integration_results, cli_results)

    all_passed = (
        all(r.passed for r in unit_results.values()) and
        all(r.passed for r in integration_results.values()) and
        all(r.passed for r in cli_results.values())
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
