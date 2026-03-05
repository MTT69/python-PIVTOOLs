#!/usr/bin/env python3
"""
test_vector_statistics.py

Tests for VectorStatisticsProcessor and gamma1/gamma2 vortex detection.

Formula-level tests use the production code directly via temp .mat files —
no local reimplementation of statistics formulas.

Usage:
    pytest unit-tests/test_vector_statistics.py -v
    pytest unit-tests/test_vector_statistics.py -v --make-figures
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_gui.vector_statistics.instantaneous_statistics import (
    VectorStatisticsProcessor,
    gamma1,
    gamma2,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coords(shape, dx=1.0, dy=1.0):
    """Create coordinate grids for synthetic fields."""
    ny, nx = shape
    x = np.arange(nx, dtype=np.float64) * dx
    y = np.arange(ny, dtype=np.float64) * dy
    return np.meshgrid(x, y, indexing="xy")


def _write_mat_files(data_dir, ux_frames, uy_frames, coords_x, coords_y):
    """Write synthetic vectors to .mat files in production format.

    Parameters
    ----------
    data_dir : Path
        Output directory
    ux_frames : ndarray, shape (N, H, W)
        x-velocity per frame
    uy_frames : ndarray, shape (N, H, W)
        y-velocity per frame
    coords_x, coords_y : ndarray, shape (H, W)
        Coordinate grids
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    n_frames = ux_frames.shape[0]

    for i in range(n_frames):
        # Build piv_result struct: single run (not multi-run object array)
        dt = np.dtype([("ux", object), ("uy", object), ("b_mask", object)])
        piv_result = np.empty((1,), dtype=dt)
        piv_result[0]["ux"] = ux_frames[i].astype(np.float64)
        piv_result[0]["uy"] = uy_frames[i].astype(np.float64)
        piv_result[0]["b_mask"] = np.zeros_like(ux_frames[i], dtype=np.float64)

        scipy.io.savemat(
            str(data_dir / f"B{i+1:05d}.mat"),
            {"piv_result": piv_result},
        )

    # Write coordinates.mat
    dt_c = np.dtype([("x", object), ("y", object)])
    coordinates = np.empty((1,), dtype=dt_c)
    coordinates[0]["x"] = coords_x.astype(np.float64)
    coordinates[0]["y"] = coords_y.astype(np.float64)
    scipy.io.savemat(str(data_dir / "coordinates.mat"), {"coordinates": coordinates})


def _run_statistics_processor(data_dir, n_frames, requested_stats, base_dir=None):
    """Run VectorStatisticsProcessor and return loaded results.

    Returns
    -------
    dict
        Loaded mean_stats.mat piv_result fields as dict
    """
    data_dir = Path(data_dir)
    if base_dir is None:
        base_dir = data_dir.parent

    proc = VectorStatisticsProcessor(
        data_dir=data_dir,
        base_dir=base_dir,
        num_frame_pairs=n_frames,
        vector_format="B%05d.mat",
        type_name="instantaneous",
        use_merged=False,
        camera=1,
        gamma_radius=5,
    )
    result = proc.process(
        requested_statistics=requested_stats,
        save_figures=False,
    )
    assert result["success"], f"Statistics processor failed: {result.get('error')}"

    # Load and return results
    out_file = proc.mean_stats_dir / "mean_stats.mat"
    mat = scipy.io.loadmat(str(out_file), struct_as_record=False, squeeze_me=True)
    piv = mat["piv_result"]

    # Handle multi-run array (our files have 1 run, index 0)
    if isinstance(piv, np.ndarray) and piv.dtype == object:
        piv = piv[0]

    return piv


# ---------------------------------------------------------------------------
# Gamma function tests
# ---------------------------------------------------------------------------

class TestGammaFunctions:
    """Test gamma1/gamma2 with solid-body rotation field."""

    @pytest.fixture(scope="class")
    def rotation_field(self):
        """Solid-body rotation: u=-omega*y, v=omega*x (centered)."""
        ny, nx = 61, 61
        X, Y = _make_coords((ny, nx))
        # Center the coordinates
        X = X - X.mean()
        Y = Y - Y.mean()
        omega = 2.0
        u = -omega * Y
        v = omega * X
        return X, Y, u, v

    def test_solid_body_rotation_gamma1(self, rotation_field):
        """gamma1 ~ 1.0 at the center of solid-body rotation."""
        X, Y, u, v = rotation_field
        G1 = gamma1(X, Y, u, v, d=8)
        cy, cx = G1.shape[0] // 2, G1.shape[1] // 2
        # Check center region (3x3)
        center = G1[cy-1:cy+2, cx-1:cx+2]
        assert np.all(np.abs(center) > 0.85), \
            f"gamma1 at center should be ~1.0, got {center}"

    def test_solid_body_rotation_gamma2(self, rotation_field):
        """gamma2 ~ 1.0 at the center (removes convective velocity)."""
        X, Y, u, v = rotation_field
        G2 = gamma2(X, Y, u, v, d=8)
        cy, cx = G2.shape[0] // 2, G2.shape[1] // 2
        center = G2[cy-1:cy+2, cx-1:cx+2]
        assert np.all(np.abs(center) > 0.85), \
            f"gamma2 at center should be ~1.0, got {center}"

    def test_gamma_away_from_vortex(self, rotation_field):
        """gamma1 ~ 0 far from the rotation center."""
        X, Y, u, v = rotation_field
        G1 = gamma1(X, Y, u, v, d=5)
        # Corners (far from center, but inside valid region after padding)
        ny, nx = G1.shape
        d = 5
        # Sample a point far from center but within the valid region
        corner_val = G1[d+2, d+2]
        # Far from vortex center, gamma1 should be small
        # (not exactly 0 for solid body rotation everywhere, but smaller)
        center_val = abs(G1[ny//2, nx//2])
        assert abs(corner_val) < center_val, \
            f"Corner gamma1 ({corner_val:.3f}) should be < center ({center_val:.3f})"


# ---------------------------------------------------------------------------
# Formula verification via production processor
# ---------------------------------------------------------------------------

class TestFormulaVerification:
    """Test statistics formulas by running the production processor."""

    def test_uniform_flow_mean_and_stresses(self, tmp_path):
        """Uniform flow: mean=exact, all stresses=0."""
        ny, nx, n_frames = 20, 30, 50
        X, Y = _make_coords((ny, nx))
        ux_frames = np.full((n_frames, ny, nx), 5.0)
        uy_frames = np.full((n_frames, ny, nx), 3.0)

        data_dir = tmp_path / "instantaneous" / str(n_frames) / "Cam1" / "instantaneous"
        _write_mat_files(data_dir, ux_frames, uy_frames, X, Y)

        piv = _run_statistics_processor(
            data_dir, n_frames,
            ["mean_velocity", "mean_stresses", "mean_tke"],
            base_dir=tmp_path,
        )

        np.testing.assert_allclose(piv.ux, 5.0, atol=1e-10)
        np.testing.assert_allclose(piv.uy, 3.0, atol=1e-10)
        np.testing.assert_allclose(piv.uu, 0.0, atol=1e-10)
        np.testing.assert_allclose(piv.vv, 0.0, atol=1e-10)
        np.testing.assert_allclose(piv.uv, 0.0, atol=1e-10)
        np.testing.assert_allclose(piv.tke, 0.0, atol=1e-10)

    def test_linear_shear_vorticity(self, tmp_path):
        """Linear shear u=k*y, v=0: vorticity=-k, divergence=0."""
        ny, nx = 30, 30
        k = 2.5
        X, Y = _make_coords((ny, nx), dx=1.0, dy=1.0)
        ux = k * Y
        uy = np.zeros_like(X)

        data_dir = tmp_path / "instantaneous" / "1" / "Cam1" / "instantaneous"
        _write_mat_files(data_dir, ux[None, ...], uy[None, ...], X, Y)

        piv = _run_statistics_processor(
            data_dir, 1,
            ["mean_velocity", "mean_vorticity", "mean_divergence"],
            base_dir=tmp_path,
        )

        # Interior only (avoid np.gradient edge effects)
        interior = (slice(3, -3), slice(3, -3))
        vort = np.asarray(piv.vorticity)
        div = np.asarray(piv.divergence)

        # vorticity = dv/dx - du/dy = 0 - k = -k
        np.testing.assert_allclose(vort[interior], -k, atol=0.1)
        np.testing.assert_allclose(div[interior], 0.0, atol=0.1)

    def test_solid_body_rotation_vorticity(self, tmp_path):
        """Solid-body rotation u=-omega*y, v=omega*x: vorticity=2*omega."""
        ny, nx = 40, 40
        omega = 1.5
        X, Y = _make_coords((ny, nx), dx=1.0, dy=1.0)
        X_c = X - X.mean()
        Y_c = Y - Y.mean()
        ux = -omega * Y_c
        uy = omega * X_c

        data_dir = tmp_path / "instantaneous" / "1" / "Cam1" / "instantaneous"
        _write_mat_files(data_dir, ux[None, ...], uy[None, ...], X, Y)

        piv = _run_statistics_processor(
            data_dir, 1,
            ["mean_vorticity", "mean_divergence"],
            base_dir=tmp_path,
        )

        interior = (slice(3, -3), slice(3, -3))
        vort = np.asarray(piv.vorticity)
        div = np.asarray(piv.divergence)

        np.testing.assert_allclose(vort[interior], 2 * omega, atol=0.1)
        np.testing.assert_allclose(div[interior], 0.0, atol=0.1)

    def test_source_flow_divergence(self, tmp_path):
        """Source flow u=k*x, v=k*y: divergence=2k, vorticity=0."""
        ny, nx = 30, 30
        k = 0.5
        X, Y = _make_coords((ny, nx))
        ux = k * X
        uy = k * Y

        data_dir = tmp_path / "instantaneous" / "1" / "Cam1" / "instantaneous"
        _write_mat_files(data_dir, ux[None, ...], uy[None, ...], X, Y)

        piv = _run_statistics_processor(
            data_dir, 1,
            ["mean_vorticity", "mean_divergence"],
            base_dir=tmp_path,
        )

        interior = (slice(3, -3), slice(3, -3))
        div = np.asarray(piv.divergence)
        vort = np.asarray(piv.vorticity)

        np.testing.assert_allclose(div[interior], 2 * k, atol=0.1)
        np.testing.assert_allclose(vort[interior], 0.0, atol=0.1)

    def test_oscillating_flow_variance(self, tmp_path):
        """Oscillating flow: uu = A^2/2, tke = A^2/4."""
        ny, nx, n_frames = 15, 20, 1000
        A = 2.0
        u0 = 5.0
        X, Y = _make_coords((ny, nx))

        phases = np.linspace(0, 2 * np.pi * 10, n_frames, endpoint=False)
        ux_frames = np.full((n_frames, ny, nx), u0) + A * np.sin(phases)[:, None, None]
        uy_frames = np.zeros((n_frames, ny, nx))

        data_dir = tmp_path / "instantaneous" / str(n_frames) / "Cam1" / "instantaneous"
        _write_mat_files(data_dir, ux_frames, uy_frames, X, Y)

        piv = _run_statistics_processor(
            data_dir, n_frames,
            ["mean_velocity", "mean_stresses", "mean_tke"],
            base_dir=tmp_path,
        )

        np.testing.assert_allclose(piv.ux, u0, atol=0.05)
        np.testing.assert_allclose(piv.uu, A**2 / 2, rtol=0.02)
        np.testing.assert_allclose(piv.vv, 0.0, atol=1e-10)
        np.testing.assert_allclose(piv.tke, A**2 / 4, rtol=0.02)

    def test_random_turbulence_stresses(self, tmp_path):
        """Correlated turbulence: uu~sigma_u^2, vv~sigma_v^2, uv~rho*sigma_u*sigma_v."""
        ny, nx, n_frames = 10, 10, 2000
        sigma_u, sigma_v, rho = 3.0, 2.0, 0.4
        mean_u, mean_v = 10.0, -1.0
        X, Y = _make_coords((ny, nx))

        rng = np.random.default_rng(123)

        # Generate correlated Gaussian samples
        cov = np.array([
            [sigma_u**2, rho * sigma_u * sigma_v],
            [rho * sigma_u * sigma_v, sigma_v**2],
        ])
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((n_frames, 2))
        uv_samples = z @ L.T  # (N, 2)

        ux_frames = np.full((n_frames, ny, nx), mean_u) + uv_samples[:, 0:1, None]
        uy_frames = np.full((n_frames, ny, nx), mean_v) + uv_samples[:, 1:2, None]

        data_dir = tmp_path / "instantaneous" / str(n_frames) / "Cam1" / "instantaneous"
        _write_mat_files(data_dir, ux_frames, uy_frames, X, Y)

        piv = _run_statistics_processor(
            data_dir, n_frames,
            ["mean_velocity", "mean_stresses", "mean_tke"],
            base_dir=tmp_path,
        )

        # Statistical tolerance: 2% relative for N=2000
        np.testing.assert_allclose(piv.ux.mean(), mean_u, rtol=0.02)
        np.testing.assert_allclose(piv.uy.mean(), mean_v, rtol=0.02)
        np.testing.assert_allclose(piv.uu.mean(), sigma_u**2, rtol=0.05)
        np.testing.assert_allclose(piv.vv.mean(), sigma_v**2, rtol=0.05)
        np.testing.assert_allclose(
            piv.uv.mean(), rho * sigma_u * sigma_v, rtol=0.10,
        )
        expected_tke = 0.5 * (sigma_u**2 + sigma_v**2)
        np.testing.assert_allclose(piv.tke.mean(), expected_tke, rtol=0.05)


class TestNaNHandling:
    """Verify NaN regions don't corrupt valid statistics."""

    def test_nan_region_preserves_valid_stats(self, tmp_path):
        """NaN block in frames: valid-region mean unaffected."""
        ny, nx, n_frames = 20, 30, 50
        X, Y = _make_coords((ny, nx))
        ux_frames = np.full((n_frames, ny, nx), 7.0)
        uy_frames = np.full((n_frames, ny, nx), 2.0)

        # Insert NaN block in upper-left corner of all frames
        ux_frames[:, :5, :5] = np.nan
        uy_frames[:, :5, :5] = np.nan

        data_dir = tmp_path / "instantaneous" / str(n_frames) / "Cam1" / "instantaneous"
        _write_mat_files(data_dir, ux_frames, uy_frames, X, Y)

        piv = _run_statistics_processor(
            data_dir, n_frames,
            ["mean_velocity", "mean_stresses"],
            base_dir=tmp_path,
        )

        # Valid region should have correct mean
        ux_arr = np.asarray(piv.ux)
        valid = ~np.isnan(ux_arr)
        np.testing.assert_allclose(ux_arr[valid], 7.0, atol=1e-10)

        # NaN region should remain NaN
        assert np.all(np.isnan(ux_arr[:5, :5]))


class TestDiagnosticFigures:
    """Diagnostic figure generation (gated by --make-figures)."""

    def test_make_figures(self, make_figures, output_dir, tmp_path):
        """Generate 2x3 diagnostic figure from random turbulence statistics."""
        if not make_figures:
            pytest.skip("--make-figures not set")

        import matplotlib.pyplot as plt

        ny, nx, n_frames = 30, 40, 500
        sigma_u, sigma_v, rho = 3.0, 2.0, 0.4
        mean_u, mean_v = 10.0, -1.0
        X, Y = _make_coords((ny, nx))

        rng = np.random.default_rng(77)
        cov = np.array([
            [sigma_u**2, rho * sigma_u * sigma_v],
            [rho * sigma_u * sigma_v, sigma_v**2],
        ])
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((n_frames, 2))
        uv = z @ L.T

        ux_frames = np.full((n_frames, ny, nx), mean_u) + uv[:, 0:1, None]
        uy_frames = np.full((n_frames, ny, nx), mean_v) + uv[:, 1:2, None]

        data_dir = tmp_path / "instantaneous" / str(n_frames) / "Cam1" / "instantaneous"
        _write_mat_files(data_dir, ux_frames, uy_frames, X, Y)

        piv = _run_statistics_processor(
            data_dir, n_frames,
            ["mean_velocity", "mean_stresses", "mean_tke", "mean_vorticity"],
            base_dir=tmp_path,
        )

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        im0 = axes[0, 0].pcolormesh(X, Y, np.asarray(piv.ux))
        axes[0, 0].set_title(f"Mean ux (expected {mean_u})")
        plt.colorbar(im0, ax=axes[0, 0])

        im1 = axes[0, 1].pcolormesh(X, Y, np.asarray(piv.uy))
        axes[0, 1].set_title(f"Mean uy (expected {mean_v})")
        plt.colorbar(im1, ax=axes[0, 1])

        vort = np.asarray(piv.vorticity) if hasattr(piv, "vorticity") and piv.vorticity.size > 0 else np.zeros_like(X)
        im2 = axes[0, 2].pcolormesh(X, Y, vort, cmap="RdBu_r")
        axes[0, 2].set_title("Vorticity (expected ~0)")
        plt.colorbar(im2, ax=axes[0, 2])

        im3 = axes[1, 0].pcolormesh(X, Y, np.asarray(piv.uu))
        axes[1, 0].set_title(f"uu stress (expected {sigma_u**2:.1f})")
        plt.colorbar(im3, ax=axes[1, 0])

        im4 = axes[1, 1].pcolormesh(X, Y, np.asarray(piv.vv))
        axes[1, 1].set_title(f"vv stress (expected {sigma_v**2:.1f})")
        plt.colorbar(im4, ax=axes[1, 1])

        im5 = axes[1, 2].pcolormesh(X, Y, np.asarray(piv.tke))
        axes[1, 2].set_title(f"TKE (expected {0.5*(sigma_u**2+sigma_v**2):.1f})")
        plt.colorbar(im5, ax=axes[1, 2])

        fig.tight_layout()
        fig.savefig(output_dir / "vector_statistics_verification.png", dpi=150)
        plt.close(fig)
