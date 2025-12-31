#!/usr/bin/env python3
"""
test_vector_statistics.py

Comprehensive validation of PIV vector statistics calculations.
Generates synthetic vector fields with known statistics and verifies
computed results match expected values.

Usage:
    python test_vector_statistics.py           # Run all tests
    python test_vector_statistics.py --plot    # Run with diagnostic figures
    python test_vector_statistics.py --unit    # Unit tests only
    python test_vector_statistics.py --integration  # Integration tests only

Tests validate:
    - Mean velocity computation
    - Reynolds stresses (uu, vv, uv, ww, uw, vw)
    - Turbulent Kinetic Energy (TKE)
    - Vorticity and divergence
    - Gamma1/Gamma2 vortex detection
    - NaN/mask handling
"""

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import scipy.io

from pivtools_gui.vector_statistics.instantaneous_statistics import gamma1, gamma2


# ===================== SYNTHETIC DATA GENERATION =====================


class SyntheticVectorField:
    """Generate synthetic vector fields with known properties."""

    @staticmethod
    def uniform_flow(
        shape: Tuple[int, int], u0: float, v0: float, n_frames: int
    ) -> np.ndarray:
        """
        Generate uniform flow field: u = u0, v = v0 everywhere.

        Returns:
            Array of shape (n_frames, 2, H, W) with [ux, uy] components
        """
        H, W = shape
        ux = np.full((n_frames, H, W), u0, dtype=np.float64)
        uy = np.full((n_frames, H, W), v0, dtype=np.float64)
        return np.stack([ux, uy], axis=1)  # (N, 2, H, W)

    @staticmethod
    def linear_shear(
        shape: Tuple[int, int], k: float, coords: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        Generate linear shear flow: u = k*y, v = 0 (Couette-like).

        Args:
            shape: (H, W) grid dimensions
            k: Shear rate (du/dy)
            coords: (x, y) coordinate arrays, each of shape (H, W)

        Returns:
            Array of shape (1, 2, H, W) - single frame
        """
        x, y = coords
        ux = k * y
        uy = np.zeros_like(ux)
        return np.stack([ux[None, ...], uy[None, ...]], axis=1)  # (1, 2, H, W)

    @staticmethod
    def oscillating_flow(
        shape: Tuple[int, int], u0: float, amplitude: float, n_frames: int
    ) -> np.ndarray:
        """
        Generate sinusoidal oscillating flow: u_i = u0 + A*sin(2*pi*i/N).

        The fluctuations have known variance: uu = A^2/2.

        Returns:
            Array of shape (n_frames, 2, H, W)
        """
        H, W = shape
        phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
        ux = np.zeros((n_frames, H, W), dtype=np.float64)
        uy = np.zeros((n_frames, H, W), dtype=np.float64)

        for i, phase in enumerate(phases):
            ux[i, :, :] = u0 + amplitude * np.sin(phase)
            uy[i, :, :] = 0.0

        return np.stack([ux, uy], axis=1)

    @staticmethod
    def solid_body_rotation(
        shape: Tuple[int, int],
        omega: float,
        coords: Tuple[np.ndarray, np.ndarray],
        center: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Generate solid body rotation: u = -omega*(y-yc), v = omega*(x-xc).

        This produces constant vorticity = 2*omega everywhere.

        Returns:
            Array of shape (1, 2, H, W) - single frame
        """
        x, y = coords
        if center is None:
            center = (x.mean(), y.mean())
        xc, yc = center

        ux = -omega * (y - yc)
        uy = omega * (x - xc)
        return np.stack([ux[None, ...], uy[None, ...]], axis=1)

    @staticmethod
    def source_flow(
        shape: Tuple[int, int],
        k: float,
        coords: Tuple[np.ndarray, np.ndarray],
        center: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Generate radial source flow: u = k*(x-xc), v = k*(y-yc).

        This produces constant divergence = 2*k everywhere.

        Returns:
            Array of shape (1, 2, H, W) - single frame
        """
        x, y = coords
        if center is None:
            center = (x.mean(), y.mean())
        xc, yc = center

        ux = k * (x - xc)
        uy = k * (y - yc)
        return np.stack([ux[None, ...], uy[None, ...]], axis=1)

    @staticmethod
    def random_turbulence(
        shape: Tuple[int, int],
        mean_u: float,
        mean_v: float,
        sigma_u: float,
        sigma_v: float,
        rho_uv: float,
        n_frames: int,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate random turbulent field with prescribed statistics.

        Args:
            shape: (H, W) grid dimensions
            mean_u, mean_v: Mean velocities
            sigma_u, sigma_v: Standard deviations
            rho_uv: Correlation coefficient between u' and v'
            n_frames: Number of frames
            seed: Random seed for reproducibility

        Returns:
            Array of shape (n_frames, 2, H, W)
        """
        H, W = shape
        rng = np.random.default_rng(seed)

        # Construct covariance matrix
        cov = np.array(
            [
                [sigma_u**2, rho_uv * sigma_u * sigma_v],
                [rho_uv * sigma_u * sigma_v, sigma_v**2],
            ]
        )

        # Generate correlated random samples for each point
        ux = np.zeros((n_frames, H, W), dtype=np.float64)
        uy = np.zeros((n_frames, H, W), dtype=np.float64)

        # Generate all fluctuations at once for efficiency
        n_points = H * W
        for i in range(n_frames):
            # Generate correlated u', v' fluctuations
            fluctuations = rng.multivariate_normal(
                mean=[0, 0], cov=cov, size=n_points
            )  # (H*W, 2)
            ux[i, :, :] = mean_u + fluctuations[:, 0].reshape(H, W)
            uy[i, :, :] = mean_v + fluctuations[:, 1].reshape(H, W)

        return np.stack([ux, uy], axis=1)

    @staticmethod
    def stereo_random_turbulence(
        shape: Tuple[int, int],
        mean_u: float,
        mean_v: float,
        mean_w: float,
        sigma_u: float,
        sigma_v: float,
        sigma_w: float,
        rho_uv: float,
        rho_uw: float,
        rho_vw: float,
        n_frames: int,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate 3D random turbulent field for stereo PIV.

        Returns:
            Array of shape (n_frames, 3, H, W) with [ux, uy, uz]
        """
        H, W = shape
        rng = np.random.default_rng(seed)

        # Construct 3x3 covariance matrix
        cov = np.array(
            [
                [
                    sigma_u**2,
                    rho_uv * sigma_u * sigma_v,
                    rho_uw * sigma_u * sigma_w,
                ],
                [
                    rho_uv * sigma_u * sigma_v,
                    sigma_v**2,
                    rho_vw * sigma_v * sigma_w,
                ],
                [
                    rho_uw * sigma_u * sigma_w,
                    rho_vw * sigma_v * sigma_w,
                    sigma_w**2,
                ],
            ]
        )

        # Ensure covariance matrix is positive semi-definite
        # by using Cholesky decomposition approach
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Make matrix positive definite by adding small diagonal
            min_eig = np.min(np.linalg.eigvalsh(cov))
            if min_eig < 0:
                cov += (-min_eig + 1e-10) * np.eye(3)

        ux = np.zeros((n_frames, H, W), dtype=np.float64)
        uy = np.zeros((n_frames, H, W), dtype=np.float64)
        uz = np.zeros((n_frames, H, W), dtype=np.float64)

        n_points = H * W
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress numpy linalg warnings
            for i in range(n_frames):
                fluctuations = rng.multivariate_normal(mean=[0, 0, 0], cov=cov, size=n_points)
                ux[i, :, :] = mean_u + fluctuations[:, 0].reshape(H, W)
                uy[i, :, :] = mean_v + fluctuations[:, 1].reshape(H, W)
                uz[i, :, :] = mean_w + fluctuations[:, 2].reshape(H, W)

        return np.stack([ux, uy, uz], axis=1)

    @staticmethod
    def create_coordinate_grid(
        shape: Tuple[int, int], dx: float = 1.0, dy: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create coordinate grids for the vector field."""
        H, W = shape
        x = np.arange(W) * dx
        y = np.arange(H) * dy
        X, Y = np.meshgrid(x, y)
        return X, Y


# ===================== STATISTICS COMPUTATION (Mirror Production Code) =====================


def compute_statistics(
    vectors: np.ndarray,
    coords: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    stereo: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute statistics using the same formulas as instantaneous_statistics.py.

    Args:
        vectors: Shape (N, 2 or 3, H, W) - N frames of velocity components
        coords: Optional (x, y) coordinate grids for gradient calculations
        stereo: Whether vectors have 3 components (u, v, w)

    Returns:
        Dict with computed statistics
    """
    n_frames = vectors.shape[0]
    ux = vectors[:, 0, :, :]  # (N, H, W)
    uy = vectors[:, 1, :, :]

    # Mean velocities (instantaneous_statistics.py lines 751-752)
    mean_ux = np.nanmean(ux, axis=0)
    mean_uy = np.nanmean(uy, axis=0)

    result = {
        "mean_ux": mean_ux,
        "mean_uy": mean_uy,
        "n_frames": n_frames,
    }

    # Second moments for Reynolds stresses (lines 766-772)
    E_ux2 = np.nanmean(ux**2, axis=0)
    E_uy2 = np.nanmean(uy**2, axis=0)
    E_uxuy = np.nanmean(ux * uy, axis=0)

    result["uu"] = E_ux2 - mean_ux**2
    result["vv"] = E_uy2 - mean_uy**2
    result["uv"] = E_uxuy - (mean_ux * mean_uy)

    # TKE (lines 788-792)
    if stereo and vectors.shape[1] >= 3:
        uz = vectors[:, 2, :, :]
        mean_uz = np.nanmean(uz, axis=0)
        E_uz2 = np.nanmean(uz**2, axis=0)
        E_uxuz = np.nanmean(ux * uz, axis=0)
        E_uyuz = np.nanmean(uy * uz, axis=0)

        result["mean_uz"] = mean_uz
        result["ww"] = E_uz2 - mean_uz**2
        result["uw"] = E_uxuz - (mean_ux * mean_uz)
        result["vw"] = E_uyuz - (mean_uy * mean_uz)
        result["tke"] = 0.5 * (result["uu"] + result["vv"] + result["ww"])
    else:
        result["tke"] = 0.5 * (result["uu"] + result["vv"])

    # Divergence (lines 794-804)
    if coords is not None:
        cx, cy = coords
        dx = np.gradient(cx, axis=1)
        dy = np.gradient(cy, axis=0)
        dudx = np.gradient(mean_ux, axis=1) / dx
        dvdy = np.gradient(mean_uy, axis=0) / dy
    else:
        dudx = np.gradient(mean_ux, axis=1)
        dvdy = np.gradient(mean_uy, axis=0)
    result["divergence"] = dudx + dvdy

    # Vorticity (lines 806-816)
    if coords is not None:
        cx, cy = coords
        dx = np.gradient(cx, axis=1)
        dy = np.gradient(cy, axis=0)
        dvdx = np.gradient(mean_uy, axis=1) / dx
        dudy = np.gradient(mean_ux, axis=0) / dy
    else:
        dvdx = np.gradient(mean_uy, axis=1)
        dudy = np.gradient(mean_ux, axis=0)
    result["vorticity"] = dvdx - dudy

    return result


# ===================== TEST RESULT DATACLASS =====================


@dataclass
class TestResult:
    """Result of a single test case."""

    name: str
    passed: bool
    checks: List[Dict]  # List of {name, expected, computed, passed}
    message: str = ""


# ===================== UNIT TESTS =====================


class UnitTests:
    """Direct formula verification - no file I/O."""

    def __init__(self, rtol: float = 0.01, atol: float = 1e-10, verbose: bool = True):
        """
        Args:
            rtol: Relative tolerance for statistical tests (1% default)
            atol: Absolute tolerance for analytical tests
            verbose: Print detailed output
        """
        self.rtol = rtol
        self.atol = atol
        self.verbose = verbose

    def _check(
        self, name: str, expected: float, computed: float, use_rtol: bool = False
    ) -> Dict:
        """Check a single value against expected."""
        if use_rtol and abs(expected) > 1e-10:
            passed = abs(computed - expected) / abs(expected) <= self.rtol
        else:
            passed = abs(computed - expected) <= self.atol
        return {
            "name": name,
            "expected": expected,
            "computed": computed,
            "passed": passed,
        }

    def _check_array(
        self,
        name: str,
        expected: np.ndarray,
        computed: np.ndarray,
        use_rtol: bool = False,
        check_interior: bool = True,
    ) -> Dict:
        """Check array values, optionally excluding boundaries."""
        if check_interior and expected.ndim == 2:
            # Exclude boundaries affected by gradient edge effects
            exp = expected[2:-2, 2:-2]
            comp = computed[2:-2, 2:-2]
        else:
            exp = expected.flatten()
            comp = computed.flatten()

        # Use median for robustness against edge effects
        exp_val = np.nanmedian(exp)
        comp_val = np.nanmedian(comp)

        return self._check(name, exp_val, comp_val, use_rtol)

    def test_uniform_flow(self) -> TestResult:
        """Test uniform flow: constant velocity, zero fluctuations."""
        shape = (50, 50)
        u0, v0 = 5.0, 3.0
        n_frames = 100

        vectors = SyntheticVectorField.uniform_flow(shape, u0, v0, n_frames)
        stats = compute_statistics(vectors)

        checks = [
            self._check("mean_u", u0, np.mean(stats["mean_ux"])),
            self._check("mean_v", v0, np.mean(stats["mean_uy"])),
            self._check("uu", 0.0, np.mean(stats["uu"])),
            self._check("vv", 0.0, np.mean(stats["vv"])),
            self._check("uv", 0.0, np.mean(stats["uv"])),
            self._check("TKE", 0.0, np.mean(stats["tke"])),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Uniform Flow", passed, checks)

    def test_linear_shear(self) -> TestResult:
        """Test linear shear: u = k*y, constant vorticity."""
        shape = (50, 50)
        k = 0.1  # Shear rate
        dx, dy = 1.0, 1.0

        coords = SyntheticVectorField.create_coordinate_grid(shape, dx, dy)
        vectors = SyntheticVectorField.linear_shear(shape, k, coords)
        stats = compute_statistics(vectors, coords)

        # Expected vorticity = dv/dx - du/dy = 0 - k = -k
        expected_vorticity = -k

        checks = [
            self._check("uu", 0.0, np.mean(stats["uu"])),
            self._check("vv", 0.0, np.mean(stats["vv"])),
            self._check("uv", 0.0, np.mean(stats["uv"])),
            self._check_array(
                "vorticity",
                np.full(shape, expected_vorticity),
                stats["vorticity"],
                check_interior=True,
            ),
            self._check_array(
                "divergence",
                np.zeros(shape),
                stats["divergence"],
                check_interior=True,
            ),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Linear Shear Flow", passed, checks)

    def test_oscillating_flow(self) -> TestResult:
        """Test sinusoidal oscillation: known variance A^2/2."""
        shape = (50, 50)
        u0 = 5.0
        amplitude = 2.0
        n_frames = 1000  # Need many frames for accurate variance

        vectors = SyntheticVectorField.oscillating_flow(shape, u0, amplitude, n_frames)
        stats = compute_statistics(vectors)

        # Expected variance of sin wave: A^2/2
        expected_uu = amplitude**2 / 2
        expected_tke = 0.5 * expected_uu

        checks = [
            self._check("mean_u", u0, np.mean(stats["mean_ux"]), use_rtol=True),
            self._check("mean_v", 0.0, np.mean(stats["mean_uy"])),
            self._check("uu", expected_uu, np.mean(stats["uu"]), use_rtol=True),
            self._check("vv", 0.0, np.mean(stats["vv"])),
            self._check("uv", 0.0, np.mean(stats["uv"])),
            self._check("TKE", expected_tke, np.mean(stats["tke"]), use_rtol=True),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Oscillating Flow", passed, checks)

    def test_solid_body_rotation(self) -> TestResult:
        """Test solid body rotation: constant vorticity = 2*omega."""
        shape = (51, 51)  # Odd for center point
        omega = 0.1
        dx, dy = 1.0, 1.0

        coords = SyntheticVectorField.create_coordinate_grid(shape, dx, dy)
        vectors = SyntheticVectorField.solid_body_rotation(shape, omega, coords)
        stats = compute_statistics(vectors, coords)

        # Expected vorticity = 2*omega
        expected_vorticity = 2 * omega

        checks = [
            self._check("uu", 0.0, np.mean(stats["uu"])),
            self._check("vv", 0.0, np.mean(stats["vv"])),
            self._check_array(
                "vorticity",
                np.full(shape, expected_vorticity),
                stats["vorticity"],
                check_interior=True,
            ),
            self._check_array(
                "divergence",
                np.zeros(shape),
                stats["divergence"],
                check_interior=True,
            ),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Solid Body Rotation", passed, checks)

    def test_solid_body_rotation_gamma(self) -> TestResult:
        """Test gamma functions for solid body rotation vortex."""
        shape = (51, 51)
        omega = 0.5
        dx, dy = 1.0, 1.0

        coords = SyntheticVectorField.create_coordinate_grid(shape, dx, dy)
        vectors = SyntheticVectorField.solid_body_rotation(shape, omega, coords)
        x, y = coords
        u = vectors[0, 0, :, :]
        v = vectors[0, 1, :, :]

        # Compute gamma functions
        g1 = gamma1(x, y, u, v, d=5)
        g2 = gamma2(x, y, u, v, d=5)

        # At center, gamma1 and gamma2 should be close to 1 for solid rotation
        # Use 2% tolerance since gamma functions have inherent discretization error
        center_i, center_j = shape[0] // 2, shape[1] // 2
        g1_center = g1[center_i, center_j]
        g2_center = g2[center_i, center_j]

        # Gamma should be > 0.95 for solid body rotation at vortex center
        gamma_tol = 0.02  # 2% tolerance for gamma (discretization effects)
        g1_passed = abs(abs(g1_center) - 1.0) <= gamma_tol
        g2_passed = abs(abs(g2_center) - 1.0) <= gamma_tol

        checks = [
            {
                "name": "gamma1_center",
                "expected": 1.0,
                "computed": abs(g1_center),
                "passed": g1_passed,
            },
            {
                "name": "gamma2_center",
                "expected": 1.0,
                "computed": abs(g2_center),
                "passed": g2_passed,
            },
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Solid Body Rotation (Gamma)", passed, checks)

    def test_source_flow(self) -> TestResult:
        """Test radial source: constant divergence = 2*k."""
        shape = (51, 51)
        k = 0.1
        dx, dy = 1.0, 1.0

        coords = SyntheticVectorField.create_coordinate_grid(shape, dx, dy)
        vectors = SyntheticVectorField.source_flow(shape, k, coords)
        stats = compute_statistics(vectors, coords)

        # Expected divergence = du/dx + dv/dy = k + k = 2k
        expected_divergence = 2 * k

        checks = [
            self._check_array(
                "divergence",
                np.full(shape, expected_divergence),
                stats["divergence"],
                check_interior=True,
            ),
            self._check_array(
                "vorticity",
                np.zeros(shape),
                stats["vorticity"],
                check_interior=True,
            ),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Source Flow", passed, checks)

    def test_random_turbulence(self) -> TestResult:
        """Test random turbulence with prescribed statistics."""
        shape = (30, 30)
        mean_u, mean_v = 5.0, 2.0
        sigma_u, sigma_v = 1.0, 0.5
        rho_uv = 0.3
        n_frames = 2000  # Need many frames for 1% accuracy

        vectors = SyntheticVectorField.random_turbulence(
            shape, mean_u, mean_v, sigma_u, sigma_v, rho_uv, n_frames
        )
        stats = compute_statistics(vectors)

        # Expected values
        expected_uu = sigma_u**2
        expected_vv = sigma_v**2
        expected_uv = rho_uv * sigma_u * sigma_v
        expected_tke = 0.5 * (expected_uu + expected_vv)

        checks = [
            self._check("mean_u", mean_u, np.mean(stats["mean_ux"]), use_rtol=True),
            self._check("mean_v", mean_v, np.mean(stats["mean_uy"]), use_rtol=True),
            self._check("uu", expected_uu, np.mean(stats["uu"]), use_rtol=True),
            self._check("vv", expected_vv, np.mean(stats["vv"]), use_rtol=True),
            self._check("uv", expected_uv, np.mean(stats["uv"]), use_rtol=True),
            self._check("TKE", expected_tke, np.mean(stats["tke"]), use_rtol=True),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Random Turbulence", passed, checks)

    def test_stereo_3d(self) -> TestResult:
        """Test 3D/stereo statistics with w component."""
        shape = (30, 30)
        mean_u, mean_v, mean_w = 5.0, 2.0, 1.0
        sigma_u, sigma_v, sigma_w = 1.0, 0.5, 0.3
        rho_uv, rho_uw, rho_vw = 0.3, 0.1, 0.2
        n_frames = 2000

        vectors = SyntheticVectorField.stereo_random_turbulence(
            shape,
            mean_u,
            mean_v,
            mean_w,
            sigma_u,
            sigma_v,
            sigma_w,
            rho_uv,
            rho_uw,
            rho_vw,
            n_frames,
        )
        stats = compute_statistics(vectors, stereo=True)

        # Expected values
        expected_ww = sigma_w**2
        expected_uw = rho_uw * sigma_u * sigma_w
        expected_vw = rho_vw * sigma_v * sigma_w
        expected_tke = 0.5 * (sigma_u**2 + sigma_v**2 + sigma_w**2)

        checks = [
            self._check("mean_w", mean_w, np.mean(stats["mean_uz"]), use_rtol=True),
            self._check("ww", expected_ww, np.mean(stats["ww"]), use_rtol=True),
            self._check("uw", expected_uw, np.mean(stats["uw"]), use_rtol=True),
            self._check("vw", expected_vw, np.mean(stats["vw"]), use_rtol=True),
            self._check("TKE_3D", expected_tke, np.mean(stats["tke"]), use_rtol=True),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Stereo 3D", passed, checks)

    def test_nan_handling(self) -> TestResult:
        """Test that NaN values are properly handled."""
        shape = (50, 50)
        u0, v0 = 5.0, 3.0
        n_frames = 100

        vectors = SyntheticVectorField.uniform_flow(shape, u0, v0, n_frames)

        # Introduce NaN values (masked region)
        vectors[:, :, 10:20, 10:20] = np.nan

        stats = compute_statistics(vectors)

        # Check that non-NaN regions still have correct values
        valid_mask = ~np.isnan(stats["mean_ux"])

        checks = [
            self._check("mean_u_valid", u0, np.nanmean(stats["mean_ux"])),
            self._check("mean_v_valid", v0, np.nanmean(stats["mean_uy"])),
            self._check(
                "has_nan_in_mask",
                1.0,
                float(np.any(np.isnan(stats["mean_ux"][10:20, 10:20]))),
            ),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("NaN Handling", passed, checks)

    def run_all(self) -> Dict[str, TestResult]:
        """Run all unit tests."""
        tests = [
            self.test_uniform_flow,
            self.test_linear_shear,
            self.test_oscillating_flow,
            self.test_solid_body_rotation,
            self.test_solid_body_rotation_gamma,
            self.test_source_flow,
            self.test_random_turbulence,
            self.test_stereo_3d,
            self.test_nan_handling,
        ]

        results = {}
        for test_func in tests:
            result = test_func()
            results[result.name] = result
        return results


# ===================== INTEGRATION TESTS =====================


class IntegrationTests:
    """Test through VectorStatisticsProcessor with temp MAT files."""

    def __init__(self, rtol: float = 0.01, verbose: bool = True):
        self.rtol = rtol
        self.verbose = verbose

    def _create_temp_mat_files(
        self,
        vectors: np.ndarray,
        coords: Tuple[np.ndarray, np.ndarray],
        temp_dir: Path,
    ) -> int:
        """
        Write synthetic vectors to temp MAT files.

        Args:
            vectors: Shape (N, 2 or 3, H, W)
            coords: (x, y) coordinate grids
            temp_dir: Directory to write files

        Returns:
            Number of files created
        """
        n_frames = vectors.shape[0]
        stereo = vectors.shape[1] >= 3
        x, y = coords

        # Create coordinates.mat
        dt_coords = np.dtype([("x", object), ("y", object)])
        coordinates = np.empty((1,), dtype=dt_coords)
        coordinates[0]["x"] = x
        coordinates[0]["y"] = y
        scipy.io.savemat(temp_dir / "coordinates.mat", {"coordinates": coordinates})

        # Create vector files
        for i in range(n_frames):
            # Build piv_result structure
            dt_fields = [
                ("ux", object),
                ("uy", object),
                ("b_mask", object),
            ]
            if stereo:
                dt_fields.insert(2, ("uz", object))

            dt = np.dtype(dt_fields)
            piv_result = np.empty((1,), dtype=dt)

            piv_result[0]["ux"] = vectors[i, 0, :, :]
            piv_result[0]["uy"] = vectors[i, 1, :, :]
            if stereo:
                piv_result[0]["uz"] = vectors[i, 2, :, :]
            piv_result[0]["b_mask"] = np.zeros(vectors.shape[2:], dtype=np.float64)

            file_path = temp_dir / f"{i + 1:05d}.mat"
            scipy.io.savemat(file_path, {"piv_result": piv_result})

        return n_frames

    def _run_processor(self, temp_dir: Path, n_frames: int) -> Optional[Dict]:
        """Run VectorStatisticsProcessor and return loaded results."""
        try:
            from pivtools_gui.vector_statistics.instantaneous_statistics import (
                VectorStatisticsProcessor,
            )

            processor = VectorStatisticsProcessor(
                data_dir=temp_dir,
                base_dir=temp_dir,
                num_frame_pairs=n_frames,
                vector_format="%05d.mat",
                type_name="test",
                use_merged=False,
                camera=1,
                gamma_radius=5,
            )

            result = processor.process(
                requested_statistics=[
                    "mean_velocity",
                    "reynolds_stress",
                    "normal_stress",
                    "mean_tke",
                    "mean_vorticity",
                    "mean_divergence",
                ],
                save_figures=False,
            )

            if not result["success"]:
                return None

            # Load results from output file
            output_file = Path(result["output_file"])
            mat_data = scipy.io.loadmat(output_file, squeeze_me=True, struct_as_record=False)
            return mat_data

        except Exception as e:
            print(f"  Integration test error: {e}")
            return None

    def test_uniform_flow(self) -> TestResult:
        """Integration test: uniform flow through full pipeline."""
        shape = (30, 30)
        u0, v0 = 5.0, 3.0
        n_frames = 50

        vectors = SyntheticVectorField.uniform_flow(shape, u0, v0, n_frames)
        coords = SyntheticVectorField.create_coordinate_grid(shape)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self._create_temp_mat_files(vectors, coords, temp_path)
            result = self._run_processor(temp_path, n_frames)

        if result is None:
            return TestResult(
                "Integration: Uniform Flow",
                False,
                [],
                "Processor failed to run",
            )

        # Extract results
        piv_result = result["piv_result"]
        if hasattr(piv_result, "__len__") and len(piv_result) > 0:
            piv = piv_result[0]
        else:
            piv = piv_result

        computed_mean_ux = np.mean(piv.ux)
        computed_mean_uy = np.mean(piv.uy)

        checks = [
            {
                "name": "mean_u",
                "expected": u0,
                "computed": computed_mean_ux,
                "passed": abs(computed_mean_ux - u0) < 0.01,
            },
            {
                "name": "mean_v",
                "expected": v0,
                "computed": computed_mean_uy,
                "passed": abs(computed_mean_uy - v0) < 0.01,
            },
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Integration: Uniform Flow", passed, checks)

    def test_random_turbulence(self) -> TestResult:
        """Integration test: random turbulence through full pipeline."""
        shape = (30, 30)
        mean_u, mean_v = 5.0, 2.0
        sigma_u, sigma_v = 1.0, 0.5
        rho_uv = 0.3
        n_frames = 500  # Fewer frames for integration test

        vectors = SyntheticVectorField.random_turbulence(
            shape, mean_u, mean_v, sigma_u, sigma_v, rho_uv, n_frames
        )
        coords = SyntheticVectorField.create_coordinate_grid(shape)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self._create_temp_mat_files(vectors, coords, temp_path)
            result = self._run_processor(temp_path, n_frames)

        if result is None:
            return TestResult(
                "Integration: Random Turbulence",
                False,
                [],
                "Processor failed to run",
            )

        # Extract results
        piv_result = result["piv_result"]
        if hasattr(piv_result, "__len__") and len(piv_result) > 0:
            piv = piv_result[0]
        else:
            piv = piv_result

        computed_mean_ux = np.mean(piv.ux)
        computed_uu = np.mean(piv.uu) if hasattr(piv, "uu") else 0
        computed_tke = np.mean(piv.tke) if hasattr(piv, "tke") else 0

        expected_uu = sigma_u**2
        expected_tke = 0.5 * (sigma_u**2 + sigma_v**2)

        # Use 5% tolerance for integration test (fewer frames)
        rtol = 0.05

        checks = [
            {
                "name": "mean_u",
                "expected": mean_u,
                "computed": computed_mean_ux,
                "passed": abs(computed_mean_ux - mean_u) / mean_u < rtol,
            },
            {
                "name": "uu",
                "expected": expected_uu,
                "computed": computed_uu,
                "passed": abs(computed_uu - expected_uu) / expected_uu < rtol
                if expected_uu > 0
                else True,
            },
            {
                "name": "TKE",
                "expected": expected_tke,
                "computed": computed_tke,
                "passed": abs(computed_tke - expected_tke) / expected_tke < rtol
                if expected_tke > 0
                else True,
            },
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Integration: Random Turbulence", passed, checks)

    def run_all(self) -> Dict[str, TestResult]:
        """Run all integration tests."""
        tests = [
            self.test_uniform_flow,
            self.test_random_turbulence,
        ]

        results = {}
        for test_func in tests:
            result = test_func()
            results[result.name] = result
        return results


# ===================== VISUALIZATION =====================


class Visualizer:
    """Generate diagnostic plots for test cases."""

    @staticmethod
    def plot_vector_field(
        u: np.ndarray,
        v: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        title: str,
        save_path: Path,
    ):
        """Create quiver plot of vector field."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))

        # Subsample for clarity
        skip = max(1, min(u.shape) // 20)
        ax.quiver(
            x[::skip, ::skip],
            y[::skip, ::skip],
            u[::skip, ::skip],
            v[::skip, ::skip],
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_statistics_comparison(
        expected: Dict[str, float],
        computed: Dict[str, float],
        title: str,
        save_path: Path,
    ):
        """Create bar chart comparing expected vs computed values."""
        import matplotlib.pyplot as plt

        names = list(expected.keys())
        exp_vals = [expected[n] for n in names]
        comp_vals = [computed[n] for n in names]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, exp_vals, width, label="Expected", color="blue", alpha=0.7)
        bars2 = ax.bar(x + width / 2, comp_vals, width, label="Computed", color="orange", alpha=0.7)

        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_scalar_field(
        field: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        title: str,
        save_path: Path,
        cmap: str = "RdBu_r",
    ):
        """Create contour plot of scalar field."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        c = ax.pcolormesh(x, y, field, cmap=cmap, shading="auto")
        fig.colorbar(c, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_gamma_vortex(
        g1: np.ndarray,
        g2: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        save_path: Path,
    ):
        """Create side-by-side plot of gamma1 and gamma2."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        c1 = ax1.pcolormesh(x, y, g1, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
        fig.colorbar(c1, ax=ax1)
        ax1.set_title("Gamma1")
        ax1.set_aspect("equal")

        c2 = ax2.pcolormesh(x, y, g2, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
        fig.colorbar(c2, ax=ax2)
        ax2.set_title("Gamma2")
        ax2.set_aspect("equal")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ===================== REPORTING =====================


def print_header():
    """Print test header."""
    print("=" * 70)
    print("VECTOR STATISTICS VALIDATION TEST")
    print("=" * 70)


def print_result(result: TestResult, verbose: bool = True):
    """Print formatted test result."""
    status = "\u2713 PASS" if result.passed else "\u2717 FAIL"
    print(f"\nTest: {result.name}")
    print(f"  Result: {status}")

    if verbose and result.checks:
        # Print table header
        print("  " + "-" * 60)
        print(f"  {'Statistic':<15} {'Expected':>15} {'Computed':>15} {'Status':>10}")
        print("  " + "-" * 60)

        for check in result.checks:
            status_str = "\u2713" if check["passed"] else "\u2717"
            exp_str = f"{check['expected']:.6f}" if isinstance(check["expected"], float) else str(check["expected"])
            comp_str = f"{check['computed']:.6f}" if isinstance(check["computed"], float) else str(check["computed"])
            print(f"  {check['name']:<15} {exp_str:>15} {comp_str:>15} {status_str:>10}")

    if result.message:
        print(f"  Message: {result.message}")


def print_summary(
    unit_results: Dict[str, TestResult],
    integration_results: Dict[str, TestResult],
):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    unit_passed = sum(1 for r in unit_results.values() if r.passed)
    unit_total = len(unit_results)
    int_passed = sum(1 for r in integration_results.values() if r.passed)
    int_total = len(integration_results)

    print(f"Unit Tests:      {unit_passed}/{unit_total} passed")
    print(f"Integration:     {int_passed}/{int_total} passed")
    print(f"Total:           {unit_passed + int_passed}/{unit_total + int_total} passed")

    if unit_passed + int_passed == unit_total + int_total:
        print("\nAll statistics computations validated successfully.")
    else:
        print("\nSome tests FAILED. Review output above for details.")

    print("=" * 70)


def generate_plots(output_dir: Path):
    """Generate diagnostic plots for all test cases."""
    print(f"\nGenerating diagnostic plots in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    shape = (51, 51)
    coords = SyntheticVectorField.create_coordinate_grid(shape)
    x, y = coords

    # 1. Solid body rotation
    vectors = SyntheticVectorField.solid_body_rotation(shape, 0.5, coords)
    u, v = vectors[0, 0, :, :], vectors[0, 1, :, :]
    Visualizer.plot_vector_field(u, v, x, y, "Solid Body Rotation", output_dir / "rotation_vectors.png")

    stats = compute_statistics(vectors, coords)
    Visualizer.plot_scalar_field(
        stats["vorticity"], x, y, "Vorticity (expected: 1.0)", output_dir / "rotation_vorticity.png"
    )

    g1 = gamma1(x, y, u, v, d=5)
    g2 = gamma2(x, y, u, v, d=5)
    Visualizer.plot_gamma_vortex(g1, g2, x, y, output_dir / "rotation_gamma.png")

    # 2. Source flow
    vectors = SyntheticVectorField.source_flow(shape, 0.1, coords)
    u, v = vectors[0, 0, :, :], vectors[0, 1, :, :]
    Visualizer.plot_vector_field(u, v, x, y, "Source Flow", output_dir / "source_vectors.png")

    stats = compute_statistics(vectors, coords)
    Visualizer.plot_scalar_field(
        stats["divergence"], x, y, "Divergence (expected: 0.2)", output_dir / "source_divergence.png"
    )

    # 3. Linear shear
    vectors = SyntheticVectorField.linear_shear(shape, 0.1, coords)
    u, v = vectors[0, 0, :, :], vectors[0, 1, :, :]
    Visualizer.plot_vector_field(u, v, x, y, "Linear Shear Flow", output_dir / "shear_vectors.png")

    print(f"  Generated {len(list(output_dir.glob('*.png')))} plots")


# ===================== MAIN =====================


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive validation of PIV vector statistics calculations"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate diagnostic figures",
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
        "--rtol",
        type=float,
        default=0.01,
        help="Relative tolerance for statistical tests (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_output",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    # Determine which tests to run
    run_unit = args.unit or (not args.unit and not args.integration)
    run_integration = args.integration or (not args.unit and not args.integration)

    print_header()

    unit_results = {}
    integration_results = {}

    # Run unit tests
    if run_unit:
        print("\nUNIT TESTS (Direct Formula Verification)")
        print("-" * 70)
        unit_tests = UnitTests(rtol=args.rtol)
        unit_results = unit_tests.run_all()
        for result in unit_results.values():
            print_result(result)

    # Run integration tests
    if run_integration:
        print("\nINTEGRATION TESTS (via VectorStatisticsProcessor)")
        print("-" * 70)
        integration_tests = IntegrationTests(rtol=args.rtol)
        integration_results = integration_tests.run_all()
        for result in integration_results.values():
            print_result(result)

    # Print summary
    print_summary(unit_results, integration_results)

    # Generate plots if requested
    if args.plot:
        generate_plots(Path(args.output_dir))

    # Return exit code
    all_passed = all(r.passed for r in unit_results.values()) and all(
        r.passed for r in integration_results.values()
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
