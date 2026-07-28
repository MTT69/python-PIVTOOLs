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
        center = G1[cy - 1 : cy + 2, cx - 1 : cx + 2]
        assert np.all(
            np.abs(center) > 0.85
        ), f"gamma1 at center should be ~1.0, got {center}"

    def test_solid_body_rotation_gamma2(self, rotation_field):
        """gamma2 ~ 1.0 at the center (removes convective velocity)."""
        X, Y, u, v = rotation_field
        G2 = gamma2(X, Y, u, v, d=8)
        cy, cx = G2.shape[0] // 2, G2.shape[1] // 2
        center = G2[cy - 1 : cy + 2, cx - 1 : cx + 2]
        assert np.all(
            np.abs(center) > 0.85
        ), f"gamma2 at center should be ~1.0, got {center}"

    def test_gamma_away_from_vortex(self, rotation_field):
        """gamma1 ~ 0 far from the rotation center."""
        X, Y, u, v = rotation_field
        G1 = gamma1(X, Y, u, v, d=5)
        # Corners (far from center, but inside valid region after padding)
        ny, nx = G1.shape
        d = 5
        # Sample a point far from center but within the valid region
        corner_val = G1[d + 2, d + 2]
        # Far from vortex center, gamma1 should be small
        # (not exactly 0 for solid body rotation everywhere, but smaller)
        center_val = abs(G1[ny // 2, nx // 2])
        assert (
            abs(corner_val) < center_val
        ), f"Corner gamma1 ({corner_val:.3f}) should be < center ({center_val:.3f})"


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
            data_dir,
            n_frames,
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
            data_dir,
            1,
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
            data_dir,
            1,
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
            data_dir,
            1,
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
            data_dir,
            n_frames,
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
        cov = np.array(
            [
                [sigma_u**2, rho * sigma_u * sigma_v],
                [rho * sigma_u * sigma_v, sigma_v**2],
            ]
        )
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((n_frames, 2))
        uv_samples = z @ L.T  # (N, 2)

        ux_frames = np.full((n_frames, ny, nx), mean_u) + uv_samples[:, 0:1, None]
        uy_frames = np.full((n_frames, ny, nx), mean_v) + uv_samples[:, 1:2, None]

        data_dir = tmp_path / "instantaneous" / str(n_frames) / "Cam1" / "instantaneous"
        _write_mat_files(data_dir, ux_frames, uy_frames, X, Y)

        piv = _run_statistics_processor(
            data_dir,
            n_frames,
            ["mean_velocity", "mean_stresses", "mean_tke"],
            base_dir=tmp_path,
        )

        # Statistical tolerance: 2% relative for N=2000
        np.testing.assert_allclose(piv.ux.mean(), mean_u, rtol=0.02)
        np.testing.assert_allclose(piv.uy.mean(), mean_v, rtol=0.02)
        np.testing.assert_allclose(piv.uu.mean(), sigma_u**2, rtol=0.05)
        np.testing.assert_allclose(piv.vv.mean(), sigma_v**2, rtol=0.05)
        np.testing.assert_allclose(
            piv.uv.mean(),
            rho * sigma_u * sigma_v,
            rtol=0.10,
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
            data_dir,
            n_frames,
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
        cov = np.array(
            [
                [sigma_u**2, rho * sigma_u * sigma_v],
                [rho * sigma_u * sigma_v, sigma_v**2],
            ]
        )
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((n_frames, 2))
        uv = z @ L.T

        ux_frames = np.full((n_frames, ny, nx), mean_u) + uv[:, 0:1, None]
        uy_frames = np.full((n_frames, ny, nx), mean_v) + uv[:, 1:2, None]

        data_dir = tmp_path / "instantaneous" / str(n_frames) / "Cam1" / "instantaneous"
        _write_mat_files(data_dir, ux_frames, uy_frames, X, Y)

        piv = _run_statistics_processor(
            data_dir,
            n_frames,
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

        vort = (
            np.asarray(piv.vorticity)
            if hasattr(piv, "vorticity") and piv.vorticity.size > 0
            else np.zeros_like(X)
        )
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


# ---------------------------------------------------------------------------
# Correlation-quality tests
# ---------------------------------------------------------------------------

from pivtools_gui.vector_statistics.correlation_quality import (  # noqa: E402
    aggregate,
    extract_frame_record,
    load_timeseries_mat,
)


def _write_quality_mat(
    path,
    peak_mag,
    nan_mask,
    b_mask,
    nan_reason,
    peak_ratio=None,
):
    """Write one uncalibrated-style frame file with quality channels."""
    fields = [
        ("ux", object),
        ("uy", object),
        ("b_mask", object),
        ("nan_mask", object),
        ("nan_reason", object),
        ("peak_mag", object),
    ]
    if peak_ratio is not None:
        fields.append(("peak_ratio", object))
    dt = np.dtype(fields)
    piv_result = np.empty((1,), dtype=dt)
    shape = np.asarray(peak_mag).shape
    piv_result[0]["ux"] = np.ones(shape, dtype=np.float64)
    piv_result[0]["uy"] = np.ones(shape, dtype=np.float64)
    piv_result[0]["b_mask"] = np.asarray(b_mask, dtype=np.uint8)
    piv_result[0]["nan_mask"] = np.asarray(nan_mask, dtype=np.uint8)
    piv_result[0]["nan_reason"] = np.asarray(nan_reason, dtype=np.int8)
    piv_result[0]["peak_mag"] = np.asarray(peak_mag, dtype=np.float32)
    if peak_ratio is not None:
        piv_result[0]["peak_ratio"] = np.asarray(peak_ratio, dtype=np.float32)
    scipy.io.savemat(str(path), {"piv_result": piv_result})


class TestCorrelationQualityCompute:
    """Compute-layer tests: extract_frame_record + aggregate on known data."""

    @pytest.fixture()
    def quality_dir(self, tmp_path):
        """3 frames on a 2x3 grid, one masked window, known NaN pattern.

        Window (0,0) is statically masked. Frame 1: no NaNs. Frame 2:
        window (1,1) NaN (code 1). Frame 3: windows (1,1) and (0,2) NaN
        (codes 1 and 10).
        """
        b_mask = np.zeros((2, 3), dtype=bool)
        b_mask[0, 0] = True

        for i, nan_windows in enumerate([[], [(1, 1)], [(1, 1), (0, 2)]]):
            peak_mag = np.full((2, 3), 0.5, dtype=np.float32)
            peak_mag[0, 1] = 0.9  # asymmetry so the mean is nontrivial
            nan_mask = np.zeros((2, 3), dtype=bool)
            nan_reason = np.zeros((2, 3), dtype=np.int8)
            nan_reason[b_mask] = -1
            nan_mask[b_mask] = True
            for j, (r, c) in enumerate(nan_windows):
                nan_mask[r, c] = True
                nan_reason[r, c] = 1 if j == 0 else 10
                peak_mag[r, c] = np.nan
            ratio = np.full((2, 3), 3.0, dtype=np.float32)
            _write_quality_mat(
                tmp_path / f"B{i+1:05d}.mat",
                peak_mag,
                nan_mask,
                b_mask,
                nan_reason,
                peak_ratio=ratio,
            )
        return tmp_path

    def test_known_values(self, quality_dir):
        files = sorted(quality_dir.glob("B*.mat"))
        records = [extract_frame_record(f, pass_idx=0) for f in files]
        agg = aggregate(records)

        # 5 unmasked windows (6 minus 1 masked)
        assert agg.n_unmasked == 5
        # Frame 1: mean of [0.9, 0.5, 0.5, 0.5, 0.5] = 0.58
        np.testing.assert_allclose(agg.mean_peak_mag[0], 0.58, rtol=1e-6)
        # NaN %: 0/5, 1/5, 2/5
        np.testing.assert_allclose(agg.nan_pct, [0.0, 20.0, 40.0])
        # Masked window is NaN in both maps
        assert np.isnan(agg.nan_pct_map[0, 0])
        assert np.isnan(agg.peak_mag_map[0, 0])
        # Window (1,1): NaN in 2 of 3 frames
        np.testing.assert_allclose(agg.nan_pct_map[1, 1], 200.0 / 3.0)
        # Reason breakdown: codes 1 and 10, masked (-1) and valid (0) excluded
        assert list(agg.reason_codes) == [1, 10]
        np.testing.assert_array_equal(agg.reason_counts[0], [0, 1, 1])
        np.testing.assert_array_equal(agg.reason_counts[1], [0, 0, 1])
        # Ratio present everywhere -> median is 3.0 each frame
        np.testing.assert_allclose(agg.median_peak_ratio, [3.0, 3.0, 3.0])

    def test_missing_peak_ratio_guard(self, tmp_path):
        b_mask = np.zeros((2, 2), dtype=bool)
        for i in range(2):
            _write_quality_mat(
                tmp_path / f"B{i+1:05d}.mat",
                np.full((2, 2), 0.5),
                np.zeros((2, 2), dtype=bool),
                b_mask,
                np.zeros((2, 2), dtype=np.int8),
                peak_ratio=None,
            )
        files = sorted(tmp_path.glob("B*.mat"))
        records = [extract_frame_record(f, pass_idx=0) for f in files]
        agg = aggregate(records)
        assert agg.median_peak_ratio is None
        assert agg.ratio_map is None

    def test_calibrated_file_raises(self, tmp_path):
        """A file without peak_mag (calibrated-style) must fail loudly."""
        dt = np.dtype([("ux", object), ("uy", object), ("b_mask", object)])
        piv_result = np.empty((1,), dtype=dt)
        piv_result[0]["ux"] = np.ones((2, 2))
        piv_result[0]["uy"] = np.ones((2, 2))
        piv_result[0]["b_mask"] = np.zeros((2, 2))
        path = tmp_path / "B00001.mat"
        scipy.io.savemat(str(path), {"piv_result": piv_result})

        with pytest.raises(ValueError, match="peak_mag"):
            extract_frame_record(path, pass_idx=0)


class TestCorrelationQualityProcessor:
    """Processor-level test: the correlation_quality statistic end-to-end."""

    def test_processor_outputs(self, tmp_path):
        n_frames = 3
        shape = (4, 3)
        base_dir = tmp_path

        # Calibrated data dir (ux/uy/b_mask only, as production writes it)
        cal_dir = (
            base_dir / "calibrated_piv" / str(n_frames) / "Cam1" / "instantaneous"
        )
        coords_x, coords_y = _make_coords(shape)
        ux = np.ones((n_frames,) + shape)
        uy = np.ones((n_frames,) + shape)
        _write_mat_files(cal_dir, ux, uy, coords_x, coords_y)

        # Uncalibrated data dir with the quality channels
        uncal_dir = (
            base_dir / "uncalibrated_piv" / str(n_frames) / "Cam1" / "instantaneous"
        )
        uncal_dir.mkdir(parents=True, exist_ok=True)
        b_mask = np.zeros(shape, dtype=bool)
        for i in range(n_frames):
            nan_mask = np.zeros(shape, dtype=bool)
            nan_reason = np.zeros(shape, dtype=np.int8)
            peak_mag = np.full(shape, 0.7, dtype=np.float32)
            if i == 1:
                nan_mask[2, 1] = True
                nan_reason[2, 1] = 1
                peak_mag[2, 1] = np.nan
            _write_quality_mat(
                uncal_dir / f"B{i+1:05d}.mat",
                peak_mag,
                nan_mask,
                b_mask,
                nan_reason,
                peak_ratio=np.full(shape, 2.5, dtype=np.float32),
            )

        proc = VectorStatisticsProcessor(
            data_dir=cal_dir,
            base_dir=base_dir,
            num_frame_pairs=n_frames,
            vector_format="B%05d.mat",
            type_name="instantaneous",
            use_merged=False,
            camera=1,
        )
        result = proc.process(
            requested_statistics=["mean_velocity", "correlation_quality"],
            save_figures=False,
        )
        assert result["success"], result.get("error")

        # mean_stats.mat gains the 2D maps
        mat = scipy.io.loadmat(
            str(proc.mean_stats_dir / "mean_stats.mat"),
            struct_as_record=False,
            squeeze_me=True,
        )
        piv = mat["piv_result"]
        if isinstance(piv, np.ndarray) and piv.dtype == object:
            piv = piv[0]
        nan_pct_map = np.asarray(piv.nan_pct)
        assert nan_pct_map.shape == shape
        np.testing.assert_allclose(nan_pct_map[2, 1], 100.0 / 3.0)
        assert np.asarray(piv.peak_ratio_median).shape == shape

        # Time-series file exists and round-trips through the loader
        ts_file = proc.mean_stats_dir / "corr_quality_timeseries.mat"
        assert ts_file.exists()
        agg = load_timeseries_mat(ts_file, run=1)
        assert agg.frames.size == n_frames
        np.testing.assert_allclose(agg.nan_pct, [0.0, 100.0 / 12.0, 0.0])
        np.testing.assert_allclose(agg.mean_peak_mag[0], 0.7, rtol=1e-3)
        np.testing.assert_allclose(agg.median_peak_ratio, [2.5] * 3)
        assert list(agg.reason_codes) == [1]

        # Requesting a run with no data names the available ones
        with pytest.raises(ValueError, match="available runs"):
            load_timeseries_mat(ts_file, run=5)

    def test_merged_target_skips_visibly(self, tmp_path, caplog):
        """Merged targets have no uncalibrated source: skip, don't fail."""
        import logging as _logging

        n_frames = 2
        shape = (3, 3)
        base_dir = tmp_path
        cal_dir = (
            base_dir / "calibrated_piv" / str(n_frames) / "Merged" / "instantaneous"
        )
        coords_x, coords_y = _make_coords(shape)
        _write_mat_files(
            cal_dir,
            np.ones((n_frames,) + shape),
            np.ones((n_frames,) + shape),
            coords_x,
            coords_y,
        )
        proc = VectorStatisticsProcessor(
            data_dir=cal_dir,
            base_dir=base_dir,
            num_frame_pairs=n_frames,
            vector_format="B%05d.mat",
            type_name="instantaneous",
            use_merged=True,
            camera=1,
        )
        with caplog.at_level(_logging.WARNING):
            result = proc.process(
                requested_statistics=["mean_velocity", "correlation_quality"],
                save_figures=False,
            )
        assert result["success"], result.get("error")
        assert any(
            "correlation quality skipped" in r.getMessage() for r in caplog.records
        )
        assert not (proc.mean_stats_dir / "corr_quality_timeseries.mat").exists()
