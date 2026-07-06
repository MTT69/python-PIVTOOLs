"""Tests for the mean-flow-only ensemble peak fit (fit_method: peak).

Convention under test (must match the C bulkxcorr2d planes): a zero-pixel
displacement puts the AB correlation peak at 0-based index (h//2, w//2).
fit_windows_peak returns displacements relative to that centre, so the
full-path 1-based ``corr_size/2 + 1`` bookkeeping never enters.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pivtools_cli.piv.piv_backend.peak_fitting import fit_windows_peak


def gaussian_plane(corr_size, dx, dy, sigma=2.0, amplitude=1.0):
    """Axis-aligned Gaussian AB plane with peak at centre + (dx, dy)."""
    h, w = corr_size
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    cx = w // 2 + dx
    cy = h // 2 + dy
    return (amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))).astype(np.float32)


def run_fit(planes, corr_size, mask=None):
    """Stack a list of (h, w) planes into the flat layout finalize uses."""
    n = len(planes)
    flat = np.concatenate([p.ravel() for p in planes]).astype(np.float32)
    if mask is None:
        mask = np.zeros(n, dtype=bool)
    return fit_windows_peak(flat, mask, tuple(corr_size))


class TestZeroShiftParity:
    """A centred peak must read as exactly zero displacement (risk #1)."""

    @pytest.mark.parametrize("corr_size", [(32, 32), (16, 16), (17, 17), (16, 32), (33, 17)])
    def test_zero_shift(self, corr_size):
        ux, uy, status = run_fit([gaussian_plane(corr_size, 0.0, 0.0)], corr_size)
        assert status[0] == 0
        assert abs(ux[0]) < 1e-5
        assert abs(uy[0]) < 1e-5


class TestSubpixelRecovery:
    @pytest.mark.parametrize("dx,dy", [
        (0.3, 0.0), (0.0, -0.4), (1.3, -2.7), (-3.45, 2.15), (0.49, 0.49),
    ])
    @pytest.mark.parametrize("corr_size", [(32, 32), (17, 17)])
    def test_known_shift_recovered(self, corr_size, dx, dy):
        # 3-point Gaussian fit is exact on a noiseless Gaussian (log is a
        # parabola), so tolerance is numerical, well under the 0.02 px target.
        ux, uy, status = run_fit([gaussian_plane(corr_size, dx, dy)], corr_size)
        assert status[0] == 0
        assert ux[0] == pytest.approx(dx, abs=0.02)
        assert uy[0] == pytest.approx(dy, abs=0.02)

    def test_batch_of_windows(self):
        corr_size = (32, 32)
        shifts = [(0.0, 0.0), (1.25, -0.75), (-2.5, 3.1), (0.4, 0.4)]
        planes = [gaussian_plane(corr_size, dx, dy) for dx, dy in shifts]
        ux, uy, status = run_fit(planes, corr_size)
        assert status.tolist() == [0, 0, 0, 0]
        for i, (dx, dy) in enumerate(shifts):
            assert ux[i] == pytest.approx(dx, abs=0.02)
            assert uy[i] == pytest.approx(dy, abs=0.02)

    def test_amplitude_invariance(self):
        corr_size = (32, 32)
        ux1, uy1, _ = run_fit([gaussian_plane(corr_size, 0.7, -0.2, amplitude=1.0)], corr_size)
        ux2, uy2, _ = run_fit([gaussian_plane(corr_size, 0.7, -0.2, amplitude=1e-4)], corr_size)
        assert ux1[0] == pytest.approx(ux2[0], abs=1e-5)
        assert uy1[0] == pytest.approx(uy2[0], abs=1e-5)


class TestFailureModes:
    """No fallbacks: bad windows fail loudly with NaN + nonzero status."""

    def test_border_peak_fails(self):
        corr_size = (16, 16)
        plane = np.zeros(corr_size, dtype=np.float32)
        plane[0, 5] = 1.0  # peak on the border row
        ux, uy, status = run_fit([plane], corr_size)
        assert status[0] != 0
        assert np.isnan(ux[0]) and np.isnan(uy[0])

    def test_nonpositive_neighbour_fails(self):
        corr_size = (16, 16)
        plane = np.zeros(corr_size, dtype=np.float32)
        plane[8, 8] = 1.0  # delta: neighbours are 0 -> log undefined
        ux, uy, status = run_fit([plane], corr_size)
        assert status[0] != 0
        assert np.isnan(ux[0]) and np.isnan(uy[0])

    def test_all_zero_plane_fails(self):
        corr_size = (16, 16)
        plane = np.zeros(corr_size, dtype=np.float32)
        ux, uy, status = run_fit([plane], corr_size)
        assert status[0] != 0
        assert np.isnan(ux[0]) and np.isnan(uy[0])

    def test_nan_plane_fails(self):
        corr_size = (16, 16)
        plane = np.full(corr_size, np.nan, dtype=np.float32)
        ux, uy, status = run_fit([plane], corr_size)
        assert status[0] != 0
        assert np.isnan(ux[0]) and np.isnan(uy[0])

    def test_failure_does_not_poison_neighbours(self):
        corr_size = (16, 16)
        bad = np.zeros(corr_size, dtype=np.float32)
        good = gaussian_plane(corr_size, 0.5, -0.5)
        ux, uy, status = run_fit([bad, good, bad], corr_size)
        assert status[1] == 0
        assert ux[1] == pytest.approx(0.5, abs=0.02)
        assert uy[1] == pytest.approx(-0.5, abs=0.02)
        assert status[0] != 0 and status[2] != 0


class TestMasking:
    def test_masked_windows_skipped(self):
        corr_size = (16, 16)
        # Masked window content is garbage on purpose — it must not matter.
        garbage = np.full(corr_size, np.nan, dtype=np.float32)
        good = gaussian_plane(corr_size, 1.0, 1.0)
        mask = np.array([True, False], dtype=bool)
        ux, uy, status = run_fit([garbage, good], corr_size, mask=mask)
        assert status[0] == 0          # masked: not a failure
        assert ux[0] == 0.0 and uy[0] == 0.0
        assert ux[1] == pytest.approx(1.0, abs=0.02)


class TestOutputContract:
    def test_shapes_and_dtypes(self):
        corr_size = (16, 32)
        planes = [gaussian_plane(corr_size, 0, 0)] * 5
        ux, uy, status = run_fit(planes, corr_size)
        assert ux.shape == uy.shape == status.shape == (5,)
        assert ux.dtype == np.float32 and uy.dtype == np.float32

    def test_size_mismatch_raises(self):
        flat = np.zeros(100, dtype=np.float32)  # not divisible by 16*16
        with pytest.raises(ValueError):
            fit_windows_peak(flat, np.zeros(1, dtype=bool), (16, 16))
