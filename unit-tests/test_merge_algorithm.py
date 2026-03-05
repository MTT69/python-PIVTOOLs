#!/usr/bin/env python3
"""
test_merge_algorithm.py

Direct tests of VectorMerger.merge_n_camera_fields() Tukey-window blending
with synthetic numpy arrays.  No disk I/O, no Dask, no .mat files.

Complementary to test_multicam_merge_pinhole.py (full pipeline integration).

Usage:
    pytest unit-tests/test_merge_algorithm.py -v
    pytest unit-tests/test_merge_algorithm.py -v --make-figures
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_gui.vector_merging.vector_merger import VectorMerger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(ny, nx, dx=1.0):
    """Create 2D coordinate grids with uniform spacing."""
    x = np.arange(nx, dtype=np.float64) * dx
    y = np.arange(ny, dtype=np.float64) * dx
    return np.meshgrid(x, y, indexing="xy")


def _split_horizontal_2cam(X, Y, ux, uy, overlap_frac=0.15):
    """Split a field into 2 cameras with horizontal overlap.

    Returns dict suitable for merge_n_camera_fields.
    """
    ny, nx = X.shape
    overlap_px = max(int(nx * overlap_frac), 4)
    mid = nx // 2

    # Cam1: left half + overlap into right
    c1_end = mid + overlap_px // 2
    # Cam2: right half + overlap into left
    c2_start = mid - overlap_px // 2

    cam1 = {
        "x": X[:, :c1_end],
        "y": Y[:, :c1_end],
        "ux": ux[:, :c1_end].copy(),
        "uy": uy[:, :c1_end].copy(),
        "mask": np.zeros((ny, c1_end), dtype=bool),
    }
    cam2 = {
        "x": X[:, c2_start:],
        "y": Y[:, c2_start:],
        "ux": ux[:, c2_start:].copy(),
        "uy": uy[:, c2_start:].copy(),
        "mask": np.zeros((ny, nx - c2_start), dtype=bool),
    }
    return {1: cam1, 2: cam2}


def _split_L_shape_3cam(X, Y, ux, uy, overlap_frac=0.15):
    """Split field into 3 cameras in L-shape arrangement.

    Cam1=bottom-left, Cam2=bottom-right, Cam3=top-left.
    """
    ny, nx = X.shape
    olap_x = max(int(nx * overlap_frac), 4)
    olap_y = max(int(ny * overlap_frac), 4)
    mid_x = nx // 2
    mid_y = ny // 2

    # Horizontal boundaries
    c1_xe = mid_x + olap_x // 2
    c2_xs = mid_x - olap_x // 2
    # Vertical boundaries
    c1_ye = mid_y + olap_y // 2
    c3_ys = mid_y - olap_y // 2

    def _cam(ys, ye, xs, xe):
        return {
            "x": X[ys:ye, xs:xe],
            "y": Y[ys:ye, xs:xe],
            "ux": ux[ys:ye, xs:xe].copy(),
            "uy": uy[ys:ye, xs:xe].copy(),
            "mask": np.zeros((ye - ys, xe - xs), dtype=bool),
        }

    return {
        1: _cam(0, c1_ye, 0, c1_xe),           # bottom-left
        2: _cam(0, c1_ye, c2_xs, nx),           # bottom-right
        3: _cam(c3_ys, ny, 0, c1_xe),           # top-left
    }


def _interpolate_to_merged(X_orig, Y_orig, field, X_merged, Y_merged):
    """Interpolate reference field to merged grid for comparison."""
    x_vec = X_orig[0, :]
    y_vec = Y_orig[:, 0]
    if y_vec[1] < y_vec[0]:
        y_vec = y_vec[::-1]
        field = np.flipud(field)

    interp = RegularGridInterpolator(
        (y_vec, x_vec), field,
        method="cubic", bounds_error=False, fill_value=np.nan,
    )
    pts = np.stack([Y_merged.ravel(), X_merged.ravel()], axis=-1)
    return interp(pts).reshape(X_merged.shape)


def _make_turbulent_field(ny, nx, dx=1.0, rng=None):
    """Create spatially-correlated random velocity field via FFT."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Gaussian power spectrum for spatial correlation
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    sigma_k = 0.05  # Correlation length in k-space
    power = np.exp(-K2 / (2 * sigma_k**2))

    def _gen_field():
        noise = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))
        return np.real(np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(power)))

    ux = _gen_field() * 5.0  # ~5 m/s RMS
    uy = _gen_field() * 3.0  # ~3 m/s RMS
    return ux, uy


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestMergeUniform:
    """Merge uniform fields — exact ground truth."""

    def test_uniform_2cam_horizontal(self):
        """Constant field, 2-cam horizontal: RMSE should be ~0."""
        ny, nx = 60, 100
        X, Y = _make_grid(ny, nx)
        ux = np.full((ny, nx), 7.5)
        uy = np.full((ny, nx), -2.3)

        cam_data = _split_horizontal_2cam(X, Y, ux, uy)
        Xm, Ym, uxm, uym, uzm = VectorMerger.merge_n_camera_fields(cam_data)

        valid = ~np.isnan(uxm)
        assert valid.sum() > 0.8 * Xm.size
        np.testing.assert_allclose(uxm[valid], 7.5, atol=1e-6)
        np.testing.assert_allclose(uym[valid], -2.3, atol=1e-6)

    def test_uniform_3cam_L_shape(self):
        """Constant field, 3-cam L-shape: RMSE should be ~0."""
        ny, nx = 80, 80
        X, Y = _make_grid(ny, nx)
        ux = np.full((ny, nx), 4.0)
        uy = np.full((ny, nx), 1.0)

        cam_data = _split_L_shape_3cam(X, Y, ux, uy)
        Xm, Ym, uxm, uym, uzm = VectorMerger.merge_n_camera_fields(cam_data)

        valid = ~np.isnan(uxm)
        assert valid.sum() > 0.5 * Xm.size  # L-shape covers ~3/4 of bounding box
        np.testing.assert_allclose(uxm[valid], 4.0, atol=1e-6)
        np.testing.assert_allclose(uym[valid], 1.0, atol=1e-6)


class TestMergeTurbulence:
    """Merge spatially-varying fields — statistical tolerance."""

    def test_turbulence_2cam_horizontal(self):
        """Correlated random field, 2-cam horizontal merge."""
        ny, nx = 80, 120
        X, Y = _make_grid(ny, nx)
        ux, uy = _make_turbulent_field(ny, nx)

        cam_data = _split_horizontal_2cam(X, Y, ux, uy, overlap_frac=0.20)
        Xm, Ym, uxm, uym, uzm = VectorMerger.merge_n_camera_fields(cam_data)

        # Interpolate reference to merged grid
        ux_ref = _interpolate_to_merged(X, Y, ux, Xm, Ym)
        uy_ref = _interpolate_to_merged(X, Y, uy, Xm, Ym)

        valid = ~np.isnan(uxm) & ~np.isnan(ux_ref)
        rms = np.sqrt(np.nanmean(ux**2))
        rmse_ux = np.sqrt(np.mean((uxm[valid] - ux_ref[valid])**2))

        assert rmse_ux < 0.01 * rms, f"RMSE {rmse_ux:.4f} > 1% of RMS {rms:.4f}"

        # Correlation check
        corr = np.corrcoef(uxm[valid], ux_ref[valid])[0, 1]
        assert corr > 0.999, f"Correlation {corr:.6f} < 0.999"

    def test_turbulence_3cam_L_shape(self):
        """Correlated random field, 3-cam L-shape merge."""
        ny, nx = 80, 80
        X, Y = _make_grid(ny, nx)
        ux, uy = _make_turbulent_field(ny, nx, rng=np.random.default_rng(99))

        cam_data = _split_L_shape_3cam(X, Y, ux, uy, overlap_frac=0.20)
        Xm, Ym, uxm, uym, uzm = VectorMerger.merge_n_camera_fields(cam_data)

        ux_ref = _interpolate_to_merged(X, Y, ux, Xm, Ym)
        valid = ~np.isnan(uxm) & ~np.isnan(ux_ref)
        rms = np.sqrt(np.nanmean(ux**2))
        rmse = np.sqrt(np.mean((uxm[valid] - ux_ref[valid])**2))

        assert rmse < 0.01 * rms, f"RMSE {rmse:.4f} > 1% of RMS {rms:.4f}"


class TestMergeMasks:
    """Validate mask handling during merge."""

    def test_mask_one_camera_in_overlap(self):
        """Masked region on cam1 overlap → cam2 data used."""
        ny, nx = 60, 100
        X, Y = _make_grid(ny, nx)
        ux = np.full((ny, nx), 5.0)
        uy = np.full((ny, nx), 0.0)

        cam_data = _split_horizontal_2cam(X, Y, ux, uy, overlap_frac=0.20)

        # Mask a block in cam1's overlap region (right side of cam1)
        cam1_nx = cam_data[1]["ux"].shape[1]
        cam_data[1]["mask"][:, cam1_nx - 5:] = True
        cam_data[1]["ux"][:, cam1_nx - 5:] = np.nan
        cam_data[1]["uy"][:, cam1_nx - 5:] = np.nan

        Xm, Ym, uxm, uym, uzm = VectorMerger.merge_n_camera_fields(cam_data)

        # The overlap region should still have valid data from cam2
        valid = ~np.isnan(uxm)
        # Most of the field should be valid
        assert valid.sum() > 0.7 * Xm.size

    def test_mask_both_cameras_produces_nan(self):
        """Same region masked on both cameras → NaN in merged output."""
        ny, nx = 60, 100
        X, Y = _make_grid(ny, nx)
        ux = np.full((ny, nx), 5.0)
        uy = np.full((ny, nx), 0.0)

        cam_data = _split_horizontal_2cam(X, Y, ux, uy, overlap_frac=0.20)

        # Mask the same rows on both cameras (entire row)
        for cam_idx in cam_data:
            cam_data[cam_idx]["mask"][25:35, :] = True
            cam_data[cam_idx]["ux"][25:35, :] = np.nan
            cam_data[cam_idx]["uy"][25:35, :] = np.nan

        Xm, Ym, uxm, uym, uzm = VectorMerger.merge_n_camera_fields(cam_data)

        # The masked rows should produce NaN in the merged field
        # Find merged rows closest to original masked rows
        y_masked_min = Y[25, 0]
        y_masked_max = Y[34, 0]
        row_mask = (Ym[:, 0] >= y_masked_min) & (Ym[:, 0] <= y_masked_max)
        assert np.all(np.isnan(uxm[row_mask, :])), \
            "Fully-masked rows should be NaN in merged output"

    def test_mask_at_boundary(self):
        """Mask at cam1 left edge (non-overlap) → NaN at edge, rest valid."""
        ny, nx = 60, 100
        X, Y = _make_grid(ny, nx)
        ux = np.full((ny, nx), 3.0)
        uy = np.full((ny, nx), 1.0)

        cam_data = _split_horizontal_2cam(X, Y, ux, uy)

        # Mask left edge of cam1
        cam_data[1]["mask"][:, :3] = True
        cam_data[1]["ux"][:, :3] = np.nan
        cam_data[1]["uy"][:, :3] = np.nan

        Xm, Ym, uxm, uym, uzm = VectorMerger.merge_n_camera_fields(cam_data)

        # Left edge should be NaN (only cam1 covers it, and it's masked)
        valid = ~np.isnan(uxm)
        # But the bulk should be valid
        assert valid.sum() > 0.85 * Xm.size


class TestDiagnosticFigures:
    """Diagnostic figure generation (gated by --make-figures)."""

    def test_make_figures(self, make_figures, output_dir):
        """Generate 2x3 diagnostic figure for merge verification."""
        if not make_figures:
            pytest.skip("--make-figures not set")

        import matplotlib.pyplot as plt

        ny, nx = 80, 120
        X, Y = _make_grid(ny, nx)
        ux, uy = _make_turbulent_field(ny, nx)

        cam_data = _split_horizontal_2cam(X, Y, ux, uy, overlap_frac=0.20)
        Xm, Ym, uxm, uym, uzm = VectorMerger.merge_n_camera_fields(cam_data)
        ux_ref = _interpolate_to_merged(X, Y, ux, Xm, Ym)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        vmin, vmax = np.nanpercentile(ux, [2, 98])

        axes[0, 0].pcolormesh(X, Y, ux, vmin=vmin, vmax=vmax)
        axes[0, 0].set_title("Original ux")

        axes[0, 1].pcolormesh(
            cam_data[1]["x"], cam_data[1]["y"], cam_data[1]["ux"],
            vmin=vmin, vmax=vmax,
        )
        axes[0, 1].set_title("Cam 1")

        axes[0, 2].pcolormesh(
            cam_data[2]["x"], cam_data[2]["y"], cam_data[2]["ux"],
            vmin=vmin, vmax=vmax,
        )
        axes[0, 2].set_title("Cam 2")

        axes[1, 0].pcolormesh(Xm, Ym, uxm, vmin=vmin, vmax=vmax)
        axes[1, 0].set_title("Merged ux")

        error = uxm - ux_ref
        err_lim = np.nanpercentile(np.abs(error), 99)
        axes[1, 1].pcolormesh(Xm, Ym, error, vmin=-err_lim, vmax=err_lim, cmap="RdBu_r")
        axes[1, 1].set_title("Error (merged - ref)")

        valid_err = error[~np.isnan(error)]
        axes[1, 2].hist(valid_err, bins=50, edgecolor="k", alpha=0.7)
        axes[1, 2].set_title(f"Error histogram (RMSE={np.std(valid_err):.4f})")

        for ax in axes.flat:
            ax.set_aspect("equal") if ax != axes[1, 2] else None

        fig.tight_layout()
        fig.savefig(output_dir / "merge_algorithm_verification.png", dpi=150)
        plt.close(fig)
