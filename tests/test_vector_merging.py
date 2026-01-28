#!/usr/bin/env python3
"""
test_vector_merging.py

Comprehensive test suite for VectorMerger.merge_n_camera_fields().

Tests validate:
    - 2-camera horizontal merge
    - 3-camera L-shaped merge
    - Mask handling in overlap regions
    - Random turbulence field reconstruction
    - Edge cases

Strategy: Create continuous synthetic field, split into camera regions,
run merger, verify reconstruction quality against original.

Usage:
    python test_vector_merging.py           # Run all tests
    python test_vector_merging.py --unit    # Unit tests only
    python test_vector_merging.py --plot    # Generate diagnostic plots
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pivtools_gui.vector_merging.vector_merger import VectorMerger


# ===================== TEST RESULT DATACLASS =====================


@dataclass
class TestResult:
    """Result of a single test case."""

    name: str
    passed: bool
    checks: List[Dict]
    message: str = ""


# ===================== SYNTHETIC FIELD GENERATION =====================


class SyntheticFieldGenerator:
    """Generate synthetic PIV vector fields for testing."""

    @staticmethod
    def create_coordinate_grid(
        shape: Tuple[int, int], dx: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 2D coordinate grids.

        Args:
            shape: (ny, nx) grid dimensions
            dx: Grid spacing (same for x and y)

        Returns:
            Tuple of (X, Y) 2D coordinate grids
        """
        ny, nx = shape
        x = np.arange(nx) * dx
        y = np.arange(ny) * dx
        X, Y = np.meshgrid(x, y, indexing="xy")
        return X, Y

    @staticmethod
    def uniform_flow(
        shape: Tuple[int, int],
        dx: float,
        ux_value: float,
        uy_value: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate uniform flow field.

        Args:
            shape: (ny, nx) grid dimensions
            dx: Grid spacing
            ux_value: Constant x-velocity
            uy_value: Constant y-velocity

        Returns:
            Tuple of (X, Y, ux, uy) arrays
        """
        X, Y = SyntheticFieldGenerator.create_coordinate_grid(shape, dx)
        ux = np.full(shape, ux_value, dtype=np.float64)
        uy = np.full(shape, uy_value, dtype=np.float64)
        return X, Y, ux, uy

    @staticmethod
    def spatially_correlated_turbulence(
        shape: Tuple[int, int],
        dx: float,
        mean_ux: float,
        mean_uy: float,
        variance: float,
        correlation_length: float,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate spatially correlated turbulent velocity field using FFT.

        Uses Gaussian power spectrum with specified correlation length
        to create smooth, physically realistic velocity fluctuations.

        Args:
            shape: (ny, nx) grid dimensions
            dx: Grid spacing
            mean_ux, mean_uy: Mean velocities
            variance: Velocity variance for fluctuations
            correlation_length: Spatial correlation length in grid units
            seed: Random seed for reproducibility

        Returns:
            Tuple of (X, Y, ux, uy) arrays
        """
        ny, nx = shape
        rng = np.random.default_rng(seed)

        X, Y = SyntheticFieldGenerator.create_coordinate_grid(shape, dx)

        # Create wavenumber grids
        kx = np.fft.fftfreq(nx, d=1.0)
        ky = np.fft.fftfreq(ny, d=1.0)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)

        # Gaussian power spectrum for spatial correlation
        # P(k) ~ exp(-k^2 * L^2 / 2) where L is correlation length
        power_spectrum = np.exp(-0.5 * (K * correlation_length * 2 * np.pi) ** 2)
        power_spectrum[0, 0] = 0  # Zero mean fluctuation

        # Generate amplitude from spectrum
        amplitude = np.sqrt(power_spectrum)

        # Generate ux with correlated noise
        noise_ux = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))
        ux_fft = amplitude * noise_ux
        ux_fluctuation = np.real(np.fft.ifft2(ux_fft))
        # Scale to target variance
        if np.std(ux_fluctuation) > 1e-10:
            ux_fluctuation *= np.sqrt(variance) / np.std(ux_fluctuation)
        ux = mean_ux + ux_fluctuation

        # Generate uy with correlated noise (independent from ux)
        noise_uy = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))
        uy_fft = amplitude * noise_uy
        uy_fluctuation = np.real(np.fft.ifft2(uy_fft))
        if np.std(uy_fluctuation) > 1e-10:
            uy_fluctuation *= np.sqrt(variance) / np.std(uy_fluctuation)
        uy = mean_uy + uy_fluctuation

        return X, Y, ux, uy


# ===================== CAMERA REGION SPLITTING =====================


class CameraSplitter:
    """Split continuous fields into camera views with overlap."""

    @staticmethod
    def horizontal_2camera(
        X: np.ndarray,
        Y: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        overlap_fraction: float = 0.15,
    ) -> Dict:
        """
        Split continuous field into two camera views with horizontal overlap.

        Layout:
        +------------------+------------------+
        |     Camera 1     |     Camera 2     |
        +--------+---------+---------+--------+
                 |  overlap zone     |

        Args:
            X, Y: Full field coordinate grids (2D)
            ux, uy: Full field velocity components
            overlap_fraction: Fraction of total width that overlaps (0.15 = 15%)

        Returns:
            Dict with camera data in format expected by merge_n_camera_fields
        """
        ny, nx = ux.shape

        # Calculate overlap in pixels
        overlap_pixels = int(nx * overlap_fraction)

        # Camera 1: left side (columns 0 to midpoint + half overlap)
        cam1_end = nx // 2 + overlap_pixels // 2

        # Camera 2: right side (columns midpoint - half overlap to end)
        cam2_start = nx // 2 - overlap_pixels // 2

        camera_data = {
            1: {
                "x": X[:, :cam1_end].copy(),
                "y": Y[:, :cam1_end].copy(),
                "ux": ux[:, :cam1_end].copy(),
                "uy": uy[:, :cam1_end].copy(),
                "mask": np.zeros((ny, cam1_end), dtype=bool),
            },
            2: {
                "x": X[:, cam2_start:].copy(),
                "y": Y[:, cam2_start:].copy(),
                "ux": ux[:, cam2_start:].copy(),
                "uy": uy[:, cam2_start:].copy(),
                "mask": np.zeros((ny, nx - cam2_start), dtype=bool),
            },
        }

        return camera_data

    @staticmethod
    def L_shape_3camera(
        X: np.ndarray,
        Y: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        overlap_fraction: float = 0.15,
    ) -> Dict:
        """
        Split field into L-shaped 3-camera arrangement.

        Layout (as plotted with Y increasing upward):
        +--------+--------+
        | Cam 3  |        |
        +--------+        |
        | Cam 1  | Cam 2  |
        +--------+--------+

        Camera 1: bottom-left (low Y, low X)
        Camera 2: bottom-right (low Y, high X)
        Camera 3: top-left (high Y, low X)

        Args:
            X, Y: Full field coordinate grids (2D)
            ux, uy: Full field velocity components
            overlap_fraction: Fraction that overlaps in each dimension

        Returns:
            Dict with camera data in format expected by merge_n_camera_fields
        """
        ny, nx = ux.shape

        # Horizontal and vertical midpoints with overlap
        overlap_x = int(nx * overlap_fraction)
        overlap_y = int(ny * overlap_fraction)

        mid_x = nx // 2
        mid_y = ny // 2

        # Camera 1: bottom-left quadrant (low Y, low X) - with overlap into neighbors
        cam1_x_end = mid_x + overlap_x // 2
        cam1_y_end = mid_y + overlap_y // 2

        # Camera 2: bottom-right quadrant (low Y, high X)
        cam2_x_start = mid_x - overlap_x // 2
        cam2_y_end = mid_y + overlap_y // 2

        # Camera 3: top-left quadrant (high Y, low X)
        cam3_x_end = mid_x + overlap_x // 2
        cam3_y_start = mid_y - overlap_y // 2

        camera_data = {
            1: {
                "x": X[:cam1_y_end, :cam1_x_end].copy(),
                "y": Y[:cam1_y_end, :cam1_x_end].copy(),
                "ux": ux[:cam1_y_end, :cam1_x_end].copy(),
                "uy": uy[:cam1_y_end, :cam1_x_end].copy(),
                "mask": np.zeros((cam1_y_end, cam1_x_end), dtype=bool),
            },
            2: {
                "x": X[:cam2_y_end, cam2_x_start:].copy(),
                "y": Y[:cam2_y_end, cam2_x_start:].copy(),
                "ux": ux[:cam2_y_end, cam2_x_start:].copy(),
                "uy": uy[:cam2_y_end, cam2_x_start:].copy(),
                "mask": np.zeros((cam2_y_end, nx - cam2_x_start), dtype=bool),
            },
            3: {
                "x": X[cam3_y_start:, :cam3_x_end].copy(),
                "y": Y[cam3_y_start:, :cam3_x_end].copy(),
                "ux": ux[cam3_y_start:, :cam3_x_end].copy(),
                "uy": uy[cam3_y_start:, :cam3_x_end].copy(),
                "mask": np.zeros((ny - cam3_y_start, cam3_x_end), dtype=bool),
            },
        }

        return camera_data


# ===================== MASK PATTERNS =====================


class MaskPatterns:
    """Generate various mask patterns for testing."""

    @staticmethod
    def rectangular_region(
        shape: Tuple[int, int],
        y_slice: slice,
        x_slice: slice,
    ) -> np.ndarray:
        """Create rectangular mask in specified region."""
        mask = np.zeros(shape, dtype=bool)
        mask[y_slice, x_slice] = True
        return mask

    @staticmethod
    def circular_center(
        shape: Tuple[int, int],
        radius_fraction: float = 0.15,
    ) -> np.ndarray:
        """Create circular mask at center."""
        ny, nx = shape
        Y, X = np.ogrid[:ny, :nx]
        center_y, center_x = ny // 2, nx // 2
        radius = min(ny, nx) * radius_fraction

        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        return dist < radius

    @staticmethod
    def edge_strip(
        shape: Tuple[int, int],
        edge: str,
        width_fraction: float = 0.1,
    ) -> np.ndarray:
        """Create strip mask along specified edge."""
        ny, nx = shape
        mask = np.zeros(shape, dtype=bool)

        if edge == "left":
            w = int(nx * width_fraction)
            mask[:, :w] = True
        elif edge == "right":
            w = int(nx * width_fraction)
            mask[:, -w:] = True
        elif edge == "top":
            h = int(ny * width_fraction)
            mask[:h, :] = True
        elif edge == "bottom":
            h = int(ny * width_fraction)
            mask[-h:, :] = True

        return mask


# ===================== QUALITY METRICS =====================


class MergeQualityMetrics:
    """Compute quality metrics for merged fields."""

    @staticmethod
    def compute_rmse(
        merged: np.ndarray,
        reference: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Root Mean Square Error."""
        if mask is not None:
            valid = ~mask & ~np.isnan(merged) & ~np.isnan(reference)
        else:
            valid = ~np.isnan(merged) & ~np.isnan(reference)

        if not np.any(valid):
            return np.nan

        diff = merged[valid] - reference[valid]
        return np.sqrt(np.mean(diff**2))

    @staticmethod
    def compute_max_error(
        merged: np.ndarray,
        reference: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Maximum absolute error."""
        if mask is not None:
            valid = ~mask & ~np.isnan(merged) & ~np.isnan(reference)
        else:
            valid = ~np.isnan(merged) & ~np.isnan(reference)

        if not np.any(valid):
            return np.nan

        return np.max(np.abs(merged[valid] - reference[valid]))

    @staticmethod
    def compute_correlation(
        merged: np.ndarray,
        reference: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Pearson correlation coefficient."""
        if mask is not None:
            valid = ~mask & ~np.isnan(merged) & ~np.isnan(reference)
        else:
            valid = ~np.isnan(merged) & ~np.isnan(reference)

        if not np.any(valid):
            return np.nan

        return np.corrcoef(merged[valid].flatten(), reference[valid].flatten())[0, 1]


# ===================== HELPER FUNCTIONS =====================


def interpolate_to_merged_grid(
    X_orig: np.ndarray,
    Y_orig: np.ndarray,
    field: np.ndarray,
    X_merged: np.ndarray,
    Y_merged: np.ndarray,
) -> np.ndarray:
    """
    Interpolate original field to merged grid for comparison.

    Uses same interpolation method as the merger for fair comparison.
    """
    # Extract 1D vectors from 2D grids
    if X_orig.ndim == 2:
        x_vec = X_orig[0, :]
        y_vec = Y_orig[:, 0]
    else:
        x_vec = X_orig
        y_vec = Y_orig

    # Ensure ascending y for interpolator
    if len(y_vec) > 1 and y_vec[1] < y_vec[0]:
        y_vec = y_vec[::-1]
        field = np.flipud(field)

    interp = RegularGridInterpolator(
        (y_vec, x_vec),
        field,
        method="cubic",
        bounds_error=False,
        fill_value=np.nan,
    )

    points = np.stack([Y_merged.ravel(), X_merged.ravel()], axis=-1)
    return interp(points).reshape(Y_merged.shape)


# ===================== UNIT TESTS =====================


class UnitTests:
    """Direct formula verification tests."""

    def __init__(self, rtol: float = 0.01, verbose: bool = True):
        self.rtol = rtol
        self.verbose = verbose

    def _check(self, name: str, expected: float, computed: float, tolerance: float) -> Dict:
        """Check a single value against expected with tolerance."""
        if np.isnan(computed):
            passed = False
        elif abs(expected) > 1e-10:
            passed = abs(computed - expected) <= tolerance
        else:
            passed = abs(computed) <= tolerance
        return {
            "name": name,
            "expected": expected,
            "computed": computed,
            "passed": passed,
        }

    def test_uniform_2camera_horizontal(self) -> TestResult:
        """Test uniform flow merge with 2 horizontal cameras."""
        shape = (80, 160)
        dx = 1.0
        ux_value, uy_value = 5.0, 2.0

        # Generate field
        X, Y, ux, uy = SyntheticFieldGenerator.uniform_flow(shape, dx, ux_value, uy_value)

        # Split into cameras with 15% overlap
        camera_data = CameraSplitter.horizontal_2camera(X, Y, ux, uy, overlap_fraction=0.15)

        # Merge
        X_merged, Y_merged, ux_merged, uy_merged, uz_merged = VectorMerger.merge_n_camera_fields(
            camera_data
        )

        # For uniform field, reference is just the constant value
        ref_ux = np.full_like(ux_merged, ux_value)
        ref_uy = np.full_like(uy_merged, uy_value)

        # Compute metrics (excluding NaN regions at edges)
        rmse_ux = MergeQualityMetrics.compute_rmse(ux_merged, ref_ux)
        rmse_uy = MergeQualityMetrics.compute_rmse(uy_merged, ref_uy)

        checks = [
            self._check("rmse_ux", 0.0, rmse_ux, 1e-8),
            self._check("rmse_uy", 0.0, rmse_uy, 1e-8),
            self._check("mean_ux", ux_value, np.nanmean(ux_merged), 1e-8),
            self._check("mean_uy", uy_value, np.nanmean(uy_merged), 1e-8),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Uniform 2-Camera Horizontal", passed, checks)

    def test_uniform_3camera_L_shape(self) -> TestResult:
        """Test uniform flow merge with 3 L-shaped cameras."""
        shape = (100, 100)
        dx = 1.0
        ux_value, uy_value = 3.0, -1.5

        X, Y, ux, uy = SyntheticFieldGenerator.uniform_flow(shape, dx, ux_value, uy_value)
        camera_data = CameraSplitter.L_shape_3camera(X, Y, ux, uy, overlap_fraction=0.15)

        X_merged, Y_merged, ux_merged, uy_merged, uz_merged = VectorMerger.merge_n_camera_fields(
            camera_data
        )

        rmse_ux = MergeQualityMetrics.compute_rmse(ux_merged, np.full_like(ux_merged, ux_value))
        rmse_uy = MergeQualityMetrics.compute_rmse(uy_merged, np.full_like(uy_merged, uy_value))

        checks = [
            self._check("rmse_ux", 0.0, rmse_ux, 1e-8),
            self._check("rmse_uy", 0.0, rmse_uy, 1e-8),
            self._check("mean_ux", ux_value, np.nanmean(ux_merged), 1e-8),
            self._check("mean_uy", uy_value, np.nanmean(uy_merged), 1e-8),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Uniform 3-Camera L-Shape", passed, checks)

    def test_turbulence_2camera_horizontal(self) -> TestResult:
        """Test spatially correlated turbulence merge with 2 horizontal cameras."""
        shape = (100, 200)
        dx = 1.0
        mean_ux, mean_uy = 5.0, 2.0
        variance = 1.0
        correlation_length = 8.0

        X, Y, ux, uy = SyntheticFieldGenerator.spatially_correlated_turbulence(
            shape, dx, mean_ux, mean_uy, variance, correlation_length, seed=42
        )

        camera_data = CameraSplitter.horizontal_2camera(X, Y, ux, uy, overlap_fraction=0.15)

        X_merged, Y_merged, ux_merged, uy_merged, uz_merged = VectorMerger.merge_n_camera_fields(
            camera_data
        )

        # Interpolate reference to merged grid
        ref_ux = interpolate_to_merged_grid(X, Y, ux, X_merged, Y_merged)
        ref_uy = interpolate_to_merged_grid(X, Y, uy, X_merged, Y_merged)

        # Compute metrics
        rms_velocity = np.sqrt(np.nanmean(ux**2 + uy**2))
        rmse_ux = MergeQualityMetrics.compute_rmse(ux_merged, ref_ux)
        rmse_uy = MergeQualityMetrics.compute_rmse(uy_merged, ref_uy)
        corr_ux = MergeQualityMetrics.compute_correlation(ux_merged, ref_ux)
        corr_uy = MergeQualityMetrics.compute_correlation(uy_merged, ref_uy)

        # Tolerance: 1% of RMS velocity
        tolerance = 0.01 * rms_velocity

        checks = [
            self._check("rmse_ux", 0.0, rmse_ux, tolerance),
            self._check("rmse_uy", 0.0, rmse_uy, tolerance),
            self._check("correlation_ux", 1.0, corr_ux, 0.001),
            self._check("correlation_uy", 1.0, corr_uy, 0.001),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Turbulence 2-Camera Horizontal", passed, checks)

    def test_turbulence_3camera_L_shape(self) -> TestResult:
        """Test spatially correlated turbulence merge with 3 L-shaped cameras."""
        shape = (100, 100)
        dx = 1.0
        mean_ux, mean_uy = 4.0, 1.0
        variance = 0.5
        correlation_length = 6.0

        X, Y, ux, uy = SyntheticFieldGenerator.spatially_correlated_turbulence(
            shape, dx, mean_ux, mean_uy, variance, correlation_length, seed=123
        )

        camera_data = CameraSplitter.L_shape_3camera(X, Y, ux, uy, overlap_fraction=0.15)

        X_merged, Y_merged, ux_merged, uy_merged, uz_merged = VectorMerger.merge_n_camera_fields(
            camera_data
        )

        ref_ux = interpolate_to_merged_grid(X, Y, ux, X_merged, Y_merged)
        ref_uy = interpolate_to_merged_grid(X, Y, uy, X_merged, Y_merged)

        rms_velocity = np.sqrt(np.nanmean(ux**2 + uy**2))
        rmse_ux = MergeQualityMetrics.compute_rmse(ux_merged, ref_ux)
        rmse_uy = MergeQualityMetrics.compute_rmse(uy_merged, ref_uy)
        corr_ux = MergeQualityMetrics.compute_correlation(ux_merged, ref_ux)

        tolerance = 0.01 * rms_velocity

        checks = [
            self._check("rmse_ux", 0.0, rmse_ux, tolerance),
            self._check("rmse_uy", 0.0, rmse_uy, tolerance),
            self._check("correlation_ux", 1.0, corr_ux, 0.001),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Turbulence 3-Camera L-Shape", passed, checks)

    def test_mask_in_overlap_one_camera(self) -> TestResult:
        """Test that mask in overlap region (one camera) uses data from other camera."""
        shape = (80, 160)
        dx = 1.0
        ux_value, uy_value = 5.0, 2.0

        X, Y, ux, uy = SyntheticFieldGenerator.uniform_flow(shape, dx, ux_value, uy_value)
        camera_data = CameraSplitter.horizontal_2camera(X, Y, ux, uy, overlap_fraction=0.15)

        # Add mask to camera 1 in the overlap region (right side of cam1)
        cam1_shape = camera_data[1]["ux"].shape
        overlap_start = int(cam1_shape[1] * 0.8)  # Mask last 20% of cam1
        camera_data[1]["mask"][:, overlap_start:] = True
        camera_data[1]["ux"][:, overlap_start:] = np.nan
        camera_data[1]["uy"][:, overlap_start:] = np.nan

        X_merged, Y_merged, ux_merged, uy_merged, uz_merged = VectorMerger.merge_n_camera_fields(
            camera_data
        )

        # Check that overlap region still has valid data (from camera 2)
        valid_count = np.sum(~np.isnan(ux_merged))
        total_count = ux_merged.size
        valid_fraction = valid_count / total_count

        # Most of the field should still be valid (camera 2 covers the masked region)
        checks = [
            self._check("valid_fraction", 1.0, valid_fraction, 0.1),
            self._check("mean_ux", ux_value, np.nanmean(ux_merged), 0.01),
            self._check("mean_uy", uy_value, np.nanmean(uy_merged), 0.01),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Mask in Overlap (One Camera)", passed, checks)

    def test_mask_in_overlap_both_cameras(self) -> TestResult:
        """Test that mask in same location both cameras produces NaN."""
        shape = (80, 160)
        dx = 1.0
        ux_value, uy_value = 5.0, 2.0

        X, Y, ux, uy = SyntheticFieldGenerator.uniform_flow(shape, dx, ux_value, uy_value)
        camera_data = CameraSplitter.horizontal_2camera(X, Y, ux, uy, overlap_fraction=0.15)

        # Find overlap region coordinates
        # Cam1 covers columns 0 to ~95 (for 160 wide field with 15% overlap)
        # Cam2 covers columns ~65 to 159
        # Overlap is roughly columns 65-95

        # Add mask in center of field (which is in overlap)
        ny, nx = shape
        mask_y_start, mask_y_end = ny // 3, 2 * ny // 3
        mask_x_start, mask_x_end = nx // 2 - 10, nx // 2 + 10

        # Apply mask to camera 1 (convert to cam1 coordinates)
        cam1_mask_x_start = mask_x_start
        cam1_mask_x_end = min(mask_x_end, camera_data[1]["ux"].shape[1])
        if cam1_mask_x_start < camera_data[1]["ux"].shape[1]:
            camera_data[1]["mask"][mask_y_start:mask_y_end, cam1_mask_x_start:cam1_mask_x_end] = True
            camera_data[1]["ux"][mask_y_start:mask_y_end, cam1_mask_x_start:cam1_mask_x_end] = np.nan
            camera_data[1]["uy"][mask_y_start:mask_y_end, cam1_mask_x_start:cam1_mask_x_end] = np.nan

        # Apply mask to camera 2 (convert to cam2 coordinates)
        cam2_start = nx // 2 - int(nx * 0.15 / 2)
        cam2_mask_x_start = max(0, mask_x_start - cam2_start)
        cam2_mask_x_end = min(mask_x_end - cam2_start, camera_data[2]["ux"].shape[1])
        if cam2_mask_x_start < cam2_mask_x_end:
            camera_data[2]["mask"][mask_y_start:mask_y_end, cam2_mask_x_start:cam2_mask_x_end] = True
            camera_data[2]["ux"][mask_y_start:mask_y_end, cam2_mask_x_start:cam2_mask_x_end] = np.nan
            camera_data[2]["uy"][mask_y_start:mask_y_end, cam2_mask_x_start:cam2_mask_x_end] = np.nan

        X_merged, Y_merged, ux_merged, uy_merged, uz_merged = VectorMerger.merge_n_camera_fields(
            camera_data
        )

        # Should have NaN in the masked region
        has_nan = np.any(np.isnan(ux_merged))

        # Unmasked regions should still have correct values
        valid_mean_ux = np.nanmean(ux_merged)
        valid_mean_uy = np.nanmean(uy_merged)

        checks = [
            self._check("has_nan_in_masked_region", 1.0, float(has_nan), 0.0),
            self._check("valid_mean_ux", ux_value, valid_mean_ux, 0.1),
            self._check("valid_mean_uy", uy_value, valid_mean_uy, 0.1),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Mask in Overlap (Both Cameras)", passed, checks)

    def test_mask_at_boundary(self) -> TestResult:
        """Test mask at camera boundary (non-overlap region)."""
        shape = (80, 160)
        dx = 1.0
        ux_value, uy_value = 5.0, 2.0

        X, Y, ux, uy = SyntheticFieldGenerator.uniform_flow(shape, dx, ux_value, uy_value)
        camera_data = CameraSplitter.horizontal_2camera(X, Y, ux, uy, overlap_fraction=0.15)

        # Mask left edge of camera 1 (not in overlap)
        camera_data[1]["mask"][:, :10] = True
        camera_data[1]["ux"][:, :10] = np.nan
        camera_data[1]["uy"][:, :10] = np.nan

        X_merged, Y_merged, ux_merged, uy_merged, uz_merged = VectorMerger.merge_n_camera_fields(
            camera_data
        )

        # Should have NaN at the left edge
        has_nan_at_edge = np.any(np.isnan(ux_merged[:, :5]))

        # Rest of field should be valid
        valid_fraction = np.sum(~np.isnan(ux_merged)) / ux_merged.size

        checks = [
            self._check("has_nan_at_edge", 1.0, float(has_nan_at_edge), 0.0),
            self._check("valid_fraction", 0.9, valid_fraction, 0.15),
        ]

        passed = all(c["passed"] for c in checks)
        return TestResult("Mask at Boundary", passed, checks)

    def run_all(self) -> Dict[str, TestResult]:
        """Run all unit tests."""
        tests = [
            self.test_uniform_2camera_horizontal,
            self.test_uniform_3camera_L_shape,
            self.test_turbulence_2camera_horizontal,
            self.test_turbulence_3camera_L_shape,
            self.test_mask_in_overlap_one_camera,
            self.test_mask_in_overlap_both_cameras,
            self.test_mask_at_boundary,
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
                    f"Exception: {e}\n{traceback.format_exc()}",
                )
            results[result.name] = result
        return results


# ===================== VISUALIZATION =====================


class MergeVisualizer:
    """Generate diagnostic plots for merge validation."""

    @staticmethod
    def plot_2camera_horizontal_demo(
        output_dir: Path,
        seed: int = 42,
    ) -> None:
        """
        Generate comprehensive visualization of 2-camera horizontal merge.

        Creates a figure showing:
        - Original continuous field
        - Camera 1 and Camera 2 regions with overlap
        - Merged result
        - Error map
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # Generate test field
        shape = (80, 160)
        dx = 1.0
        mean_ux, mean_uy = 5.0, 2.0
        variance = 1.0
        correlation_length = 8.0

        X, Y, ux_orig, uy_orig = SyntheticFieldGenerator.spatially_correlated_turbulence(
            shape, dx, mean_ux, mean_uy, variance, correlation_length, seed=seed
        )

        # Split and merge
        camera_data = CameraSplitter.horizontal_2camera(X, Y, ux_orig, uy_orig, overlap_fraction=0.15)
        X_merged, Y_merged, ux_merged, uy_merged, _ = VectorMerger.merge_n_camera_fields(camera_data)

        # Interpolate original to merged grid for comparison
        ux_ref = interpolate_to_merged_grid(X, Y, ux_orig, X_merged, Y_merged)
        uy_ref = interpolate_to_merged_grid(X, Y, uy_orig, X_merged, Y_merged)

        # Compute error
        error_ux = ux_merged - ux_ref
        error_uy = uy_merged - uy_ref

        # Create figure
        fig = plt.figure(figsize=(16, 12))

        # Velocity magnitude for visualization
        vel_mag_orig = np.sqrt(ux_orig**2 + uy_orig**2)
        vel_mag_merged = np.sqrt(ux_merged**2 + uy_merged**2)
        vmin, vmax = np.nanpercentile(vel_mag_orig, [2, 98])

        # Row 1: Original field and camera regions
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.pcolormesh(X, Y, vel_mag_orig, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ax1.set_title('Original Continuous Field\n(Velocity Magnitude)', fontsize=11)
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=ax1, label='|V| (m/s)')

        # Camera 1 region
        ax2 = fig.add_subplot(2, 3, 2)
        cam1_ux = camera_data[1]['ux']
        cam1_x = camera_data[1]['x']
        cam1_y = camera_data[1]['y']
        cam1_mag = np.sqrt(cam1_ux**2 + camera_data[1]['uy']**2)
        im2 = ax2.pcolormesh(cam1_x, cam1_y, cam1_mag, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ax2.set_title('Camera 1 (Left)\nwith overlap region', fontsize=11)
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        # Mark overlap region
        overlap_start = cam1_x[0, -1] - 30  # approximate
        ax2.axvline(overlap_start, color='red', linestyle='--', linewidth=2, label='Overlap start')
        ax2.legend(loc='upper right', fontsize=8)
        plt.colorbar(im2, ax=ax2, label='|V| (m/s)')

        # Camera 2 region
        ax3 = fig.add_subplot(2, 3, 3)
        cam2_ux = camera_data[2]['ux']
        cam2_x = camera_data[2]['x']
        cam2_y = camera_data[2]['y']
        cam2_mag = np.sqrt(cam2_ux**2 + camera_data[2]['uy']**2)
        im3 = ax3.pcolormesh(cam2_x, cam2_y, cam2_mag, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ax3.set_title('Camera 2 (Right)\nwith overlap region', fontsize=11)
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        # Mark overlap region
        overlap_end = cam2_x[0, 0] + 30
        ax3.axvline(overlap_end, color='red', linestyle='--', linewidth=2, label='Overlap end')
        ax3.legend(loc='upper left', fontsize=8)
        plt.colorbar(im3, ax=ax3, label='|V| (m/s)')

        # Row 2: Merged result and error
        ax4 = fig.add_subplot(2, 3, 4)
        im4 = ax4.pcolormesh(X_merged, Y_merged, vel_mag_merged, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ax4.set_title('Merged Result\n(Velocity Magnitude)', fontsize=11)
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        plt.colorbar(im4, ax=ax4, label='|V| (m/s)')

        # Error in ux
        ax5 = fig.add_subplot(2, 3, 5)
        err_max = max(np.nanmax(np.abs(error_ux)), 1e-10)
        im5 = ax5.pcolormesh(X_merged, Y_merged, error_ux, cmap='RdBu_r',
                             vmin=-err_max, vmax=err_max, shading='auto')
        ax5.set_title(f'Error in ux (merged - original)\nRMSE = {np.sqrt(np.nanmean(error_ux**2)):.2e}', fontsize=11)
        ax5.set_xlabel('X (pixels)')
        ax5.set_ylabel('Y (pixels)')
        plt.colorbar(im5, ax=ax5, label='Error (m/s)')

        # Error histogram
        ax6 = fig.add_subplot(2, 3, 6)
        valid_err = error_ux[~np.isnan(error_ux)].flatten()
        ax6.hist(valid_err, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax6.axvline(0, color='red', linestyle='--', linewidth=2)
        ax6.set_title('Error Distribution\n(ux component)', fontsize=11)
        ax6.set_xlabel('Error (m/s)')
        ax6.set_ylabel('Count')
        ax6.text(0.95, 0.95, f'Mean: {np.mean(valid_err):.2e}\nStd: {np.std(valid_err):.2e}',
                 transform=ax6.transAxes, ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_path = output_dir / 'merge_2camera_horizontal.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_path}")

    @staticmethod
    def plot_3camera_L_shape_demo(
        output_dir: Path,
        seed: int = 123,
    ) -> None:
        """
        Generate comprehensive visualization of 3-camera L-shaped merge.
        """
        import matplotlib.pyplot as plt

        shape = (100, 100)
        dx = 1.0
        mean_ux, mean_uy = 4.0, 1.0
        variance = 0.5
        correlation_length = 6.0

        X, Y, ux_orig, uy_orig = SyntheticFieldGenerator.spatially_correlated_turbulence(
            shape, dx, mean_ux, mean_uy, variance, correlation_length, seed=seed
        )

        camera_data = CameraSplitter.L_shape_3camera(X, Y, ux_orig, uy_orig, overlap_fraction=0.15)
        X_merged, Y_merged, ux_merged, uy_merged, _ = VectorMerger.merge_n_camera_fields(camera_data)

        ux_ref = interpolate_to_merged_grid(X, Y, ux_orig, X_merged, Y_merged)
        error_ux = ux_merged - ux_ref

        vel_mag_orig = np.sqrt(ux_orig**2 + uy_orig**2)
        vel_mag_merged = np.sqrt(ux_merged**2 + uy_merged**2)
        vmin, vmax = np.nanpercentile(vel_mag_orig, [2, 98])

        fig = plt.figure(figsize=(16, 10))

        # Original field with camera boundaries overlaid
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.pcolormesh(X, Y, vel_mag_orig, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ax1.set_title('Original Field with Camera Layout\n(L-shaped configuration)', fontsize=11)
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        # Draw camera boundaries
        mid = 50
        ax1.axhline(mid, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax1.axvline(mid, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax1.text(25, 25, 'Cam 1', color='white', fontsize=12, ha='center', va='center', fontweight='bold')
        ax1.text(75, 25, 'Cam 2', color='white', fontsize=12, ha='center', va='center', fontweight='bold')
        ax1.text(25, 75, 'Cam 3', color='white', fontsize=12, ha='center', va='center', fontweight='bold')
        ax1.text(75, 75, '(no cam)', color='white', fontsize=10, ha='center', va='center', alpha=0.7)
        plt.colorbar(im1, ax=ax1, label='|V| (m/s)')

        # Camera 1
        ax2 = fig.add_subplot(2, 3, 2)
        cam1_mag = np.sqrt(camera_data[1]['ux']**2 + camera_data[1]['uy']**2)
        im2 = ax2.pcolormesh(camera_data[1]['x'], camera_data[1]['y'], cam1_mag,
                             cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ax2.set_title('Camera 1 (Bottom-Left)', fontsize=11)
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2, label='|V| (m/s)')

        # Camera 2
        ax3 = fig.add_subplot(2, 3, 3)
        cam2_mag = np.sqrt(camera_data[2]['ux']**2 + camera_data[2]['uy']**2)
        im3 = ax3.pcolormesh(camera_data[2]['x'], camera_data[2]['y'], cam2_mag,
                             cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ax3.set_title('Camera 2 (Bottom-Right)', fontsize=11)
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        plt.colorbar(im3, ax=ax3, label='|V| (m/s)')

        # Camera 3
        ax4 = fig.add_subplot(2, 3, 4)
        cam3_mag = np.sqrt(camera_data[3]['ux']**2 + camera_data[3]['uy']**2)
        im4 = ax4.pcolormesh(camera_data[3]['x'], camera_data[3]['y'], cam3_mag,
                             cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ax4.set_title('Camera 3 (Top-Left)', fontsize=11)
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        plt.colorbar(im4, ax=ax4, label='|V| (m/s)')

        # Merged result
        ax5 = fig.add_subplot(2, 3, 5)
        im5 = ax5.pcolormesh(X_merged, Y_merged, vel_mag_merged, cmap='viridis',
                             vmin=vmin, vmax=vmax, shading='auto')
        ax5.set_title('Merged Result', fontsize=11)
        ax5.set_xlabel('X (pixels)')
        ax5.set_ylabel('Y (pixels)')
        plt.colorbar(im5, ax=ax5, label='|V| (m/s)')

        # Error map
        ax6 = fig.add_subplot(2, 3, 6)
        err_max = max(np.nanmax(np.abs(error_ux)), 1e-10)
        im6 = ax6.pcolormesh(X_merged, Y_merged, error_ux, cmap='RdBu_r',
                             vmin=-err_max, vmax=err_max, shading='auto')
        rmse = np.sqrt(np.nanmean(error_ux**2))
        ax6.set_title(f'Error Map (ux)\nRMSE = {rmse:.2e}', fontsize=11)
        ax6.set_xlabel('X (pixels)')
        ax6.set_ylabel('Y (pixels)')
        plt.colorbar(im6, ax=ax6, label='Error (m/s)')

        plt.tight_layout()
        output_path = output_dir / 'merge_3camera_L_shape.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_path}")

    @staticmethod
    def plot_mask_handling_demo(
        output_dir: Path,
    ) -> None:
        """
        Demonstrate mask handling in overlap regions.
        """
        import matplotlib.pyplot as plt

        shape = (80, 160)
        dx = 1.0
        ux_value, uy_value = 5.0, 2.0

        X, Y, ux, uy = SyntheticFieldGenerator.uniform_flow(shape, dx, ux_value, uy_value)

        # Case 1: Mask in one camera only (should use other camera's data)
        camera_data_1 = CameraSplitter.horizontal_2camera(X, Y, ux, uy, overlap_fraction=0.15)
        cam1_shape = camera_data_1[1]['ux'].shape
        overlap_start = int(cam1_shape[1] * 0.7)
        camera_data_1[1]['mask'][:, overlap_start:] = True
        camera_data_1[1]['ux'][:, overlap_start:] = np.nan
        camera_data_1[1]['uy'][:, overlap_start:] = np.nan

        _, _, ux_merged_1, _, _ = VectorMerger.merge_n_camera_fields(camera_data_1)

        # Case 2: Mask in both cameras (should produce NaN)
        camera_data_2 = CameraSplitter.horizontal_2camera(X, Y, ux, uy, overlap_fraction=0.15)
        ny, nx = shape
        mask_y_start, mask_y_end = ny // 3, 2 * ny // 3
        mask_x_center = nx // 2
        mask_half_width = 15

        # Mask camera 1
        cam1_end = camera_data_2[1]['ux'].shape[1]
        if mask_x_center - mask_half_width < cam1_end:
            x_start = max(0, mask_x_center - mask_half_width)
            x_end = min(cam1_end, mask_x_center + mask_half_width)
            camera_data_2[1]['mask'][mask_y_start:mask_y_end, x_start:x_end] = True
            camera_data_2[1]['ux'][mask_y_start:mask_y_end, x_start:x_end] = np.nan

        # Mask camera 2
        cam2_start_coord = camera_data_2[2]['x'][0, 0]
        cam2_mask_x_start = int(mask_x_center - mask_half_width - cam2_start_coord)
        cam2_mask_x_end = int(mask_x_center + mask_half_width - cam2_start_coord)
        cam2_mask_x_start = max(0, cam2_mask_x_start)
        cam2_mask_x_end = min(camera_data_2[2]['ux'].shape[1], cam2_mask_x_end)
        if cam2_mask_x_start < cam2_mask_x_end:
            camera_data_2[2]['mask'][mask_y_start:mask_y_end, cam2_mask_x_start:cam2_mask_x_end] = True
            camera_data_2[2]['ux'][mask_y_start:mask_y_end, cam2_mask_x_start:cam2_mask_x_end] = np.nan

        X_merged_2, Y_merged_2, ux_merged_2, _, _ = VectorMerger.merge_n_camera_fields(camera_data_2)

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: Mask in one camera
        ax = axes[0, 0]
        cam1_display = camera_data_1[1]['ux'].copy()
        cam1_display[camera_data_1[1]['mask']] = np.nan
        ax.pcolormesh(camera_data_1[1]['x'], camera_data_1[1]['y'], cam1_display,
                      cmap='viridis', vmin=4, vmax=6, shading='auto')
        ax.set_title('Camera 1: Masked Region\n(right side masked)', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax = axes[0, 1]
        ax.pcolormesh(camera_data_1[2]['x'], camera_data_1[2]['y'], camera_data_1[2]['ux'],
                      cmap='viridis', vmin=4, vmax=6, shading='auto')
        ax.set_title('Camera 2: Full Data\n(covers masked region)', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax = axes[0, 2]
        im = ax.pcolormesh(X, Y[:ux_merged_1.shape[0], :ux_merged_1.shape[1]], ux_merged_1,
                           cmap='viridis', vmin=4, vmax=6, shading='auto')
        valid_frac = np.sum(~np.isnan(ux_merged_1)) / ux_merged_1.size * 100
        ax.set_title(f'Merged Result\n({valid_frac:.1f}% valid - other camera used)', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='ux (m/s)')

        # Row 2: Mask in both cameras
        ax = axes[1, 0]
        cam1_display_2 = camera_data_2[1]['ux'].copy()
        ax.pcolormesh(camera_data_2[1]['x'], camera_data_2[1]['y'], cam1_display_2,
                      cmap='viridis', vmin=4, vmax=6, shading='auto')
        ax.set_title('Camera 1: Center Masked', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax = axes[1, 1]
        cam2_display_2 = camera_data_2[2]['ux'].copy()
        ax.pcolormesh(camera_data_2[2]['x'], camera_data_2[2]['y'], cam2_display_2,
                      cmap='viridis', vmin=4, vmax=6, shading='auto')
        ax.set_title('Camera 2: Center Masked', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax = axes[1, 2]
        im = ax.pcolormesh(X_merged_2, Y_merged_2, ux_merged_2, cmap='viridis',
                           vmin=4, vmax=6, shading='auto')
        nan_frac = np.sum(np.isnan(ux_merged_2)) / ux_merged_2.size * 100
        ax.set_title(f'Merged Result\n(NaN where both masked, {nan_frac:.1f}% invalid)', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='ux (m/s)')

        plt.suptitle('Mask Handling Demonstration', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = output_dir / 'merge_mask_handling.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_path}")

    @staticmethod
    def plot_blend_weights_demo(
        output_dir: Path,
    ) -> None:
        """
        Visualize the Tukey window blending weights used in overlap regions.
        """
        import matplotlib.pyplot as plt
        from scipy.ndimage import distance_transform_edt

        shape = (80, 160)
        dx = 1.0
        ux_value, uy_value = 5.0, 2.0

        X, Y, ux, uy = SyntheticFieldGenerator.uniform_flow(shape, dx, ux_value, uy_value)
        camera_data = CameraSplitter.horizontal_2camera(X, Y, ux, uy, overlap_fraction=0.15)

        # Compute weights manually (mimicking merge_n_camera_fields logic)
        ny, nx = shape
        alpha = 0.5

        weights = {}
        for cam_id, cam in camera_data.items():
            # Create valid mask
            valid = ~np.isnan(cam['ux']) & ~cam['mask']

            # Distance from edges
            edge_distance = distance_transform_edt(valid)
            max_dist = np.max(edge_distance)

            if max_dist > 0:
                norm_dist = edge_distance / max_dist
                weight = np.ones_like(norm_dist)
                taper_region = norm_dist < alpha / 2
                weight[taper_region] = 0.5 * (1 - np.cos(2 * np.pi * norm_dist[taper_region] / alpha))
                weight[~valid] = 0
            else:
                weight = np.zeros_like(edge_distance)

            weights[cam_id] = weight

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        # Camera 1 weight
        ax = axes[0, 0]
        im = ax.pcolormesh(camera_data[1]['x'], camera_data[1]['y'], weights[1],
                           cmap='viridis', vmin=0, vmax=1, shading='auto')
        ax.set_title('Camera 1 Blend Weight\n(Tukey window)', fontsize=11)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Weight')

        # Camera 2 weight
        ax = axes[0, 1]
        im = ax.pcolormesh(camera_data[2]['x'], camera_data[2]['y'], weights[2],
                           cmap='viridis', vmin=0, vmax=1, shading='auto')
        ax.set_title('Camera 2 Blend Weight\n(Tukey window)', fontsize=11)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Weight')

        # Weight cross-section at mid-height
        ax = axes[0, 2]
        mid_y = shape[0] // 2

        # Get x coordinates and weights for cross-section
        cam1_x = camera_data[1]['x'][mid_y, :]
        cam1_w = weights[1][mid_y, :]
        cam2_x = camera_data[2]['x'][mid_y, :]
        cam2_w = weights[2][mid_y, :]

        ax.plot(cam1_x, cam1_w, 'b-', linewidth=2, label='Camera 1')
        ax.plot(cam2_x, cam2_w, 'r-', linewidth=2, label='Camera 2')
        ax.axvline(cam1_x[-1], color='gray', linestyle='--', alpha=0.5)
        ax.axvline(cam2_x[0], color='gray', linestyle='--', alpha=0.5)
        ax.fill_between([cam2_x[0], cam1_x[-1]], [0, 0], [1.1, 1.1], alpha=0.2, color='green', label='Overlap')
        ax.set_xlim(0, 160)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Blend Weight')
        ax.set_title('Weight Profile at Mid-Height\n(cross-section)', fontsize=11)
        ax.legend(loc='center')
        ax.grid(True, alpha=0.3)

        # Weight sum visualization (should be ~1 everywhere)
        # Need to interpolate to common grid first
        full_x = np.arange(160)
        full_y = np.arange(80)
        Xf, Yf = np.meshgrid(full_x, full_y)

        weight_sum = np.zeros((80, 160))

        # Add camera 1 weight
        cam1_x_range = slice(0, camera_data[1]['x'].shape[1])
        weight_sum[:, cam1_x_range] += weights[1]

        # Add camera 2 weight (offset by start position)
        cam2_start = int(camera_data[2]['x'][0, 0])
        weight_sum[:, cam2_start:] += weights[2]

        ax = axes[1, 0]
        im = ax.pcolormesh(Xf, Yf, weight_sum, cmap='RdYlGn', vmin=0.8, vmax=1.2, shading='auto')
        ax.set_title('Sum of Weights\n(should be ~1.0 everywhere)', fontsize=11)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Weight Sum')

        # Zoom on overlap region
        ax = axes[1, 1]
        overlap_slice = slice(60, 110)
        im = ax.pcolormesh(Xf[:, overlap_slice], Yf[:, overlap_slice], weight_sum[:, overlap_slice],
                           cmap='RdYlGn', vmin=0.95, vmax=1.05, shading='auto')
        ax.set_title('Weight Sum in Overlap Region\n(zoomed)', fontsize=11)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Weight Sum')

        # Histogram of weight sum
        ax = axes[1, 2]
        ax.hist(weight_sum.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal = 1.0')
        ax.set_xlabel('Weight Sum')
        ax.set_ylabel('Count')
        ax.set_title(f'Weight Sum Distribution\nMean: {np.mean(weight_sum):.4f}', fontsize=11)
        ax.legend()

        plt.suptitle('Tukey Window Blend Weights Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = output_dir / 'merge_blend_weights.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_path}")


def generate_all_plots(output_dir: Path) -> None:
    """Generate all diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating diagnostic plots in {output_dir}...")

    MergeVisualizer.plot_2camera_horizontal_demo(output_dir)
    MergeVisualizer.plot_3camera_L_shape_demo(output_dir)
    MergeVisualizer.plot_mask_handling_demo(output_dir)
    MergeVisualizer.plot_blend_weights_demo(output_dir)

    print(f"\nGenerated {len(list(output_dir.glob('*.png')))} plots")


# ===================== REPORTING =====================


def print_header():
    """Print test header."""
    print("=" * 70)
    print("VECTOR MERGING VALIDATION TEST")
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
            exp_str = (
                f"{exp:.6e}"
                if isinstance(exp, float) and (abs(exp) < 0.001 or abs(exp) > 1000)
                else f"{exp:.6f}"
                if isinstance(exp, float)
                else str(exp)
            )
            comp_str = (
                f"{comp:.6e}"
                if isinstance(comp, float) and (abs(comp) < 0.001 or abs(comp) > 1000)
                else f"{comp:.6f}"
                if isinstance(comp, float)
                else str(comp)
            )
            print(f"  {check['name']:<25} {exp_str:>15} {comp_str:>15} {status_str:>8}")

    if result.message:
        print(f"  Message: {result.message}")


def print_summary(unit_results: Dict[str, TestResult]):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    unit_passed = sum(1 for r in unit_results.values() if r.passed)
    unit_total = len(unit_results)

    print(f"Unit Tests:        {unit_passed}/{unit_total} passed")

    if unit_passed == unit_total:
        print("\nAll vector merging tests passed successfully.")
    else:
        print("\nSome tests FAILED. Review output above for details.")

    print("=" * 70)


# ===================== MAIN =====================


def main():
    parser = argparse.ArgumentParser(description="Vector merging validation tests")
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate diagnostic plots",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_output/vector_merging",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.01,
        help="Relative tolerance for numerical tests",
    )
    args = parser.parse_args()

    print_header()

    unit_results = {}

    print("\nUNIT TESTS (Vector Merging Validation)")
    print("-" * 70)
    unit_tests = UnitTests(rtol=args.rtol)
    unit_results = unit_tests.run_all()
    for result in unit_results.values():
        print_result(result)

    print_summary(unit_results)

    # Generate plots if requested
    if args.plot:
        generate_all_plots(Path(args.output_dir))

    all_passed = all(r.passed for r in unit_results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
