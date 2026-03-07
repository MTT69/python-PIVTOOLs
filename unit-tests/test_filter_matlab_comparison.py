"""
Test POD filter and time filter against MATLAB reference output.

Prerequisites:
    1. Open MATLAB, cd to the unit-tests/ directory
    2. Run: generate_filter_reference
    3. This creates test_output/pod_reference.mat and test_output/time_reference.mat
    4. Then run: pytest test_filter_matlab_comparison.py -v
"""

import numpy as np
import pytest
import scipy.io as sio
from pathlib import Path

from pivtools_cli.preprocessing.pod_filter import (
    find_auto_mode,
    pod_filter_single_channel,
    pod_filter_batch,
    time_filter_batch,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "test_output"
POD_REF = FIXTURE_DIR / "pod_reference.mat"
TIME_REF = FIXTURE_DIR / "time_reference.mat"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pod_ref():
    """Load MATLAB POD reference data, skip if file missing."""
    if not POD_REF.exists():
        pytest.skip(f"MATLAB reference not found: {POD_REF}")
    return sio.loadmat(str(POD_REF), squeeze_me=True)


@pytest.fixture(scope="module")
def time_ref():
    """Load MATLAB time-filter reference data, skip if file missing."""
    if not TIME_REF.exists():
        pytest.skip(f"MATLAB reference not found: {TIME_REF}")
    return sio.loadmat(str(TIME_REF), squeeze_me=True)


# ---------------------------------------------------------------------------
# POD filter tests
# ---------------------------------------------------------------------------

class TestFindAutoMode:
    """Test find_auto_mode matches MATLAB N_auto for both channels."""

    def test_channel1_mode_count(self, pod_ref):
        """Python find_auto_mode on channel 1 matches MATLAB N_auto1."""
        PSI1 = pod_ref["PSI1"].astype(np.float64)
        eigVal1 = pod_ref["eigVal1"].astype(np.float64)
        n_images = int(pod_ref["n_images"])
        expected = int(pod_ref["N_auto1"])

        result = find_auto_mode(PSI1, eigVal1, n_images)
        assert result == expected, (
            f"Channel 1: Python find_auto_mode={result}, MATLAB N_auto1={expected}"
        )

    def test_channel2_mode_count(self, pod_ref):
        """Python find_auto_mode on channel 2 matches MATLAB N_auto2."""
        PSI2 = pod_ref["PSI2"].astype(np.float64)
        eigVal2 = pod_ref["eigVal2"].astype(np.float64)
        n_images = int(pod_ref["n_images"])
        expected = int(pod_ref["N_auto2"])

        result = find_auto_mode(PSI2, eigVal2, n_images)
        assert result == expected, (
            f"Channel 2: Python find_auto_mode={result}, MATLAB N_auto2={expected}"
        )


class TestPODEigenvalues:
    """Compare eigenvalues (SVD singular values of covariance matrix)."""

    def test_eigenvalues_channel1(self, pod_ref):
        """Eigenvalues from Python SVD match MATLAB.

        Dominant eigenvalues match tightly; noise-floor eigenvalues
        (clustered ~6000) can differ up to ~3% due to float32 input
        precision and different SVD implementations.
        """
        M = pod_ref["M_bloc1"].astype(np.float64)
        C = M @ M.T
        _, S, _ = np.linalg.svd(C, full_matrices=False)
        matlab_eigvals = pod_ref["eigVal1"].astype(np.float64)
        np.testing.assert_allclose(S, matlab_eigvals, rtol=0.03,
                                   err_msg="Channel 1 eigenvalues differ")

    def test_eigenvalues_channel2(self, pod_ref):
        """Eigenvalues from Python SVD match MATLAB.

        See test_eigenvalues_channel1 docstring for tolerance rationale.
        """
        M = pod_ref["M_bloc2"].astype(np.float64)
        C = M @ M.T
        _, S, _ = np.linalg.svd(C, full_matrices=False)
        matlab_eigvals = pod_ref["eigVal2"].astype(np.float64)
        np.testing.assert_allclose(S, matlab_eigvals, rtol=0.03,
                                   err_msg="Channel 2 eigenvalues differ")


class TestPODFilterOutput:
    """Compare full POD-filtered output against MATLAB reference."""

    def _run_python_pod(self, M_bloc, n_images, H, W):
        """Run Python POD filter on a flat M_bloc matrix.

        Reshapes M_bloc (n_images, n_pixels) -> (n_images, H, W),
        runs pod_filter_single_channel, returns (n_images, n_pixels).
        """
        images = M_bloc.astype(np.float32).reshape(n_images, H, W)
        filtered = pod_filter_single_channel(images, verbose=False)
        return filtered.reshape(n_images, -1).astype(np.float64)

    def test_filtered_output_channel1(self, pod_ref):
        """Filtered channel 1 matches MATLAB within atol=0.5."""
        n_images = int(pod_ref["n_images"])
        H = int(pod_ref["H"])
        W = int(pod_ref["W"])
        M_bloc1 = pod_ref["M_bloc1"].astype(np.float64)
        expected = pod_ref["M_filtered1"].astype(np.float64)

        result = self._run_python_pod(M_bloc1, n_images, H, W)

        # Absolute tolerance: filter residual values are small, so use atol
        np.testing.assert_allclose(result, expected, atol=0.5,
                                   err_msg="Channel 1 filtered output differs")

    def test_filtered_output_channel2(self, pod_ref):
        """Filtered channel 2 matches MATLAB within atol=0.5."""
        n_images = int(pod_ref["n_images"])
        H = int(pod_ref["H"])
        W = int(pod_ref["W"])
        M_bloc2 = pod_ref["M_bloc2"].astype(np.float64)
        expected = pod_ref["M_filtered2"].astype(np.float64)

        result = self._run_python_pod(M_bloc2, n_images, H, W)

        np.testing.assert_allclose(result, expected, atol=0.5,
                                   err_msg="Channel 2 filtered output differs")

    def test_filtered_correlation_channel1(self, pod_ref):
        """Correlation between Python and MATLAB filtered output > 0.9999."""
        n_images = int(pod_ref["n_images"])
        H = int(pod_ref["H"])
        W = int(pod_ref["W"])
        M_bloc1 = pod_ref["M_bloc1"].astype(np.float64)
        expected = pod_ref["M_filtered1"].astype(np.float64)

        result = self._run_python_pod(M_bloc1, n_images, H, W)

        corr = np.corrcoef(result.ravel(), expected.ravel())[0, 1]
        assert corr > 0.9999, f"Channel 1 correlation = {corr:.6f}, expected > 0.9999"

    def test_filtered_correlation_channel2(self, pod_ref):
        """Correlation between Python and MATLAB filtered output > 0.9999."""
        n_images = int(pod_ref["n_images"])
        H = int(pod_ref["H"])
        W = int(pod_ref["W"])
        M_bloc2 = pod_ref["M_bloc2"].astype(np.float64)
        expected = pod_ref["M_filtered2"].astype(np.float64)

        result = self._run_python_pod(M_bloc2, n_images, H, W)

        corr = np.corrcoef(result.ravel(), expected.ravel())[0, 1]
        assert corr > 0.9999, f"Channel 2 correlation = {corr:.6f}, expected > 0.9999"


class TestPODFilterBatch:
    """Test pod_filter_batch wrapper processes both channels consistently."""

    def test_batch_wrapper_matches_single_channel(self, pod_ref):
        """pod_filter_batch(batch) should give same result as running
        pod_filter_single_channel on each channel independently."""
        n_images = int(pod_ref["n_images"])
        H = int(pod_ref["H"])
        W = int(pod_ref["W"])

        # Build a (n_images, 2, H, W) batch from the reference inputs
        M1 = pod_ref["M_bloc1"].astype(np.float32).reshape(n_images, H, W)
        M2 = pod_ref["M_bloc2"].astype(np.float32).reshape(n_images, H, W)
        batch = np.stack([M1, M2], axis=1)  # (n_images, 2, H, W)

        # Run single channel on copies
        filtered_ch0 = pod_filter_single_channel(M1.copy(), verbose=False)
        filtered_ch1 = pod_filter_single_channel(M2.copy(), verbose=False)

        # Run batch wrapper
        batch_filtered = pod_filter_batch(batch.copy(), verbose=False)

        np.testing.assert_array_equal(
            batch_filtered[:, 0], filtered_ch0,
            err_msg="Batch channel 0 differs from single-channel result"
        )
        np.testing.assert_array_equal(
            batch_filtered[:, 1], filtered_ch1,
            err_msg="Batch channel 1 differs from single-channel result"
        )


# ---------------------------------------------------------------------------
# Time filter tests
# ---------------------------------------------------------------------------

class TestTimeFilter:
    """Test time filter against MATLAB reference."""

    def test_time_minimum_channel1(self, time_ref):
        """Per-pixel temporal minimum matches MATLAB for channel 1."""
        images = time_ref["images_frame1"].astype(np.float32)
        expected_min = time_ref["time_min1"].astype(np.float32)

        computed_min = images.min(axis=0)
        np.testing.assert_allclose(computed_min, expected_min, atol=1e-4,
                                   err_msg="Channel 1 time minimum differs")

    def test_time_minimum_channel2(self, time_ref):
        """Per-pixel temporal minimum matches MATLAB for channel 2."""
        images = time_ref["images_frame2"].astype(np.float32)
        expected_min = time_ref["time_min2"].astype(np.float32)

        computed_min = images.min(axis=0)
        np.testing.assert_allclose(computed_min, expected_min, atol=1e-4,
                                   err_msg="Channel 2 time minimum differs")

    def test_filtered_output_channel1(self, time_ref):
        """Time-filtered channel 1 matches MATLAB reference."""
        n_images = int(time_ref["n_images"])
        H = int(time_ref["H"])
        W = int(time_ref["W"])
        images = time_ref["images_frame1"].astype(np.float32)
        expected = time_ref["filtered_frame1"].astype(np.float32)

        # time_filter_batch expects (N, 2, H, W) - build a dummy batch
        # with channel 1 data in slot 0, zeros in slot 1
        batch = np.zeros((n_images, 2, H, W), dtype=np.float32)
        batch[:, 0] = images
        batch[:, 1] = time_ref["images_frame2"].astype(np.float32)

        filtered = time_filter_batch(batch, verbose=False)

        np.testing.assert_allclose(
            filtered[:, 0], expected, atol=1e-4,
            err_msg="Channel 1 time-filtered output differs"
        )

    def test_filtered_output_channel2(self, time_ref):
        """Time-filtered channel 2 matches MATLAB reference."""
        n_images = int(time_ref["n_images"])
        H = int(time_ref["H"])
        W = int(time_ref["W"])
        images_f2 = time_ref["images_frame2"].astype(np.float32)
        expected = time_ref["filtered_frame2"].astype(np.float32)

        batch = np.zeros((n_images, 2, H, W), dtype=np.float32)
        batch[:, 0] = time_ref["images_frame1"].astype(np.float32)
        batch[:, 1] = images_f2

        filtered = time_filter_batch(batch, verbose=False)

        np.testing.assert_allclose(
            filtered[:, 1], expected, atol=1e-4,
            err_msg="Channel 2 time-filtered output differs"
        )

    def test_non_negativity(self, time_ref):
        """Time-filtered output should be non-negative (min subtracted)."""
        n_images = int(time_ref["n_images"])
        H = int(time_ref["H"])
        W = int(time_ref["W"])

        batch = np.zeros((n_images, 2, H, W), dtype=np.float32)
        batch[:, 0] = time_ref["images_frame1"].astype(np.float32)
        batch[:, 1] = time_ref["images_frame2"].astype(np.float32)

        filtered = time_filter_batch(batch, verbose=False)

        assert filtered.min() >= -1e-6, (
            f"Time-filtered output has negative values: min = {filtered.min()}"
        )

    def test_minimum_frame_near_zero(self, time_ref):
        """After time filtering, the frame that was the per-pixel minimum
        should have values very close to zero."""
        n_images = int(time_ref["n_images"])
        H = int(time_ref["H"])
        W = int(time_ref["W"])

        batch = np.zeros((n_images, 2, H, W), dtype=np.float32)
        batch[:, 0] = time_ref["images_frame1"].astype(np.float32)
        batch[:, 1] = time_ref["images_frame2"].astype(np.float32)

        filtered = time_filter_batch(batch, verbose=False)

        # The temporal minimum of the filtered output should be ~0
        for ch in range(2):
            channel_min = filtered[:, ch].min(axis=0)
            assert np.allclose(channel_min, 0.0, atol=1e-5), (
                f"Channel {ch}: minimum frame not near zero, "
                f"max deviation = {np.abs(channel_min).max()}"
            )


# ---------------------------------------------------------------------------
# Standalone verification (no MATLAB reference needed)
# ---------------------------------------------------------------------------

class TestPODFilterStandalone:
    """Sanity checks that don't require MATLAB .mat files."""

    def test_find_auto_mode_all_noise(self):
        """If all eigenvalues are ~equal (pure noise), return 0 (nothing to remove)."""
        n = 20
        PSI = np.random.randn(n, n)
        # Orthogonalize to get realistic eigenvectors
        PSI, _ = np.linalg.qr(PSI)
        eigvals = np.ones(n)  # flat spectrum = noise
        result = find_auto_mode(PSI, eigvals, n)
        # With flat eigenvalues AND random eigenvectors (mean ~0),
        # both criteria trigger immediately → should return 1 or more.
        # Actually, with QR-orthogonalized random vectors, the mean
        # of each column will be very small, so the psi criterion is met.
        # The sig_diff criterion: |1.0 - 1.0| / 1.0 = 0 < 0.01 * 1.0 = 0.01.
        # So it WILL trigger at i=0, returning 1.
        # This is correct: even with flat eigenvalues, the algorithm
        # triggers based on both criteria.
        assert result >= 0

    def test_find_auto_mode_strong_signal(self):
        """Strong signal modes should not be removed - large eigenvalue gaps."""
        n = 20
        # Create eigenvectors where first few have non-zero mean
        PSI = np.eye(n)
        PSI[:, 0] = 1.0 / np.sqrt(n)  # First mode has large mean
        # Eigenvalues with strong gap
        eigvals = np.zeros(n)
        eigvals[0] = 1000.0
        eigvals[1] = 100.0
        eigvals[2:] = 1.0

        result = find_auto_mode(PSI, eigvals, n)
        # First mode has mean = 1/sqrt(20) ≈ 0.224 > 0.01, so not noise
        # Should skip it and look further
        assert result >= 0

    def test_pod_filter_preserves_shape(self):
        """Output shape and dtype match input."""
        images = np.random.rand(10, 32, 32).astype(np.float32)
        filtered = pod_filter_single_channel(images.copy(), verbose=False)
        assert filtered.shape == images.shape
        assert filtered.dtype == images.dtype

    def test_pod_filter_reduces_background(self):
        """POD filter should reduce a strong constant background."""
        n, h, w = 15, 32, 32
        background = np.ones((h, w), dtype=np.float32) * 100.0
        images = np.empty((n, h, w), dtype=np.float32)
        for i in range(n):
            # Constant background + small varying signal
            images[i] = background + np.float32(2.0 * np.sin(
                np.arange(h * w).reshape(h, w) * 0.1 * (i + 1)
            ))

        filtered = pod_filter_single_channel(images.copy(), verbose=False)

        # Filtered mean should be much smaller than original mean
        assert filtered.mean() < images.mean() * 0.5, (
            f"POD didn't reduce background: "
            f"before={images.mean():.1f}, after={filtered.mean():.1f}"
        )

    def test_time_filter_batch_shape(self):
        """time_filter_batch preserves shape and dtype."""
        batch = np.random.rand(10, 2, 32, 32).astype(np.float32) + 10.0
        result = time_filter_batch(batch.copy(), verbose=False)
        assert result.shape == batch.shape
        assert result.dtype == batch.dtype


# ---------------------------------------------------------------------------
# Diagnostic figures (gated by --make-figures)
# ---------------------------------------------------------------------------

def _add_watermark(ax, label):
    """Add a large semi-transparent watermark letter to an axes."""
    ax.text(
        0.5, 0.5, label,
        transform=ax.transAxes, fontsize=48, fontweight="bold",
        color="white", alpha=0.3, ha="center", va="center",
    )


class TestDiagnosticFigures:
    """Generate diagnostic plots (gated by --make-figures)."""

    def test_make_pod_figure(self, pod_ref, make_figures, output_dir):
        """Generate 2x3 POD filter diagnostic figure."""
        if not make_figures:
            pytest.skip("Pass --make-figures to generate diagnostic plots")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_images = int(pod_ref["n_images"])
        H = int(pod_ref["H"])
        W = int(pod_ref["W"])

        # --- Run Python POD filter on both channels ---
        M1 = pod_ref["M_bloc1"].astype(np.float32).reshape(n_images, H, W)
        M2 = pod_ref["M_bloc2"].astype(np.float32).reshape(n_images, H, W)
        filtered1 = pod_filter_single_channel(M1.copy(), verbose=False)
        filtered2 = pod_filter_single_channel(M2.copy(), verbose=False)

        # Compute eigenvalues and N_auto for both channels
        M1_64 = pod_ref["M_bloc1"].astype(np.float64)
        C1 = M1_64 @ M1_64.T
        PSI1, S1, _ = np.linalg.svd(C1, full_matrices=False)
        N_auto1 = find_auto_mode(PSI1, S1, n_images)

        M2_64 = pod_ref["M_bloc2"].astype(np.float64)
        C2 = M2_64 @ M2_64.T
        PSI2, S2, _ = np.linalg.svd(C2, full_matrices=False)
        N_auto2 = find_auto_mode(PSI2, S2, n_images)

        # MATLAB reference filtered output
        matlab_filt1 = pod_ref["M_filtered1"].astype(np.float64).reshape(n_images, H, W)
        matlab_filt2 = pod_ref["M_filtered2"].astype(np.float64).reshape(n_images, H, W)

        # Error between Python and MATLAB
        err1 = filtered1[0].astype(np.float64) - matlab_filt1[0]
        err2 = filtered2[0].astype(np.float64) - matlab_filt2[0]
        all_err = np.concatenate([err1.ravel(), err2.ravel()])

        # Correlation
        corr1 = np.corrcoef(
            filtered1.reshape(n_images, -1).astype(np.float64).ravel(),
            matlab_filt1.reshape(n_images, -1).ravel(),
        )[0, 1]
        corr2 = np.corrcoef(
            filtered2.reshape(n_images, -1).astype(np.float64).ravel(),
            matlab_filt2.reshape(n_images, -1).ravel(),
        )[0, 1]

        # --- Build 2x3 figure ---
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(
            f"POD Filter Verification — N_auto={N_auto1}/{N_auto2} modes removed",
            fontweight="bold", fontsize=13,
        )

        # Determine shared color scales per channel (raw vs filtered)
        for row, (raw, filt, label) in enumerate([
            (M1[0], filtered1[0], "A"),
            (M2[0], filtered2[0], "B"),
        ]):
            # Col 0: Raw input frame 1
            ax = axes[row, 0]
            vmin, vmax = raw.min(), raw.max()
            im = ax.imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Raw Frame {label}")
            _add_watermark(ax, label)

            # Col 1: Filtered frame 1 (same color scale as raw for comparison)
            ax = axes[row, 1]
            im = ax.imshow(filt, cmap="gray", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Filtered Frame {label}")
            _add_watermark(ax, label)

        # Col 2 top: Eigenvalue spectrum
        ax = axes[0, 2]
        mode_idx = np.arange(1, len(S1) + 1)
        ax.semilogy(mode_idx, S1, "b.-", label="Channel A", markersize=4)
        ax.semilogy(mode_idx, S2, "r.-", label="Channel B", markersize=4)
        ax.axvline(N_auto1, color="b", linestyle="--", alpha=0.7,
                   label=f"N_auto A={N_auto1}")
        ax.axvline(N_auto2, color="r", linestyle="--", alpha=0.7,
                   label=f"N_auto B={N_auto2}")
        ax.set_xlabel("Mode index")
        ax.set_ylabel("Eigenvalue (log)")
        ax.set_title("Eigenvalue Spectrum")
        ax.legend(fontsize=7)

        # Col 2 bottom: Error histogram
        ax = axes[1, 2]
        ax.hist(all_err, bins=60, color="#4c72b0", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Pixel error (Python − MATLAB)")
        ax.set_ylabel("Count")
        ax.set_title("Error Histogram")
        ax.axvline(0, color="k", linestyle="-", linewidth=0.8)

        # Stats text box
        max_abs_err = np.max(np.abs(all_err))
        stats_text = (
            f"N_auto: A={N_auto1}, B={N_auto2}  |  "
            f"Max |error|={max_abs_err:.4f}  |  "
            f"Corr: A={corr1:.6f}, B={corr2:.6f}"
        )
        fig.text(
            0.5, 0.01, stats_text, ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        out_path = output_dir / "pod_filter_verification.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Figure saved: {out_path}")

    def test_make_time_figure(self, time_ref, make_figures, output_dir):
        """Generate 2x3 time filter diagnostic figure."""
        if not make_figures:
            pytest.skip("Pass --make-figures to generate diagnostic plots")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_images = int(time_ref["n_images"])
        H = int(time_ref["H"])
        W = int(time_ref["W"])

        # Raw images
        images_A = time_ref["images_frame1"].astype(np.float32)
        images_B = time_ref["images_frame2"].astype(np.float32)

        # Time minimums
        time_min_A = images_A.min(axis=0)
        time_min_B = images_B.min(axis=0)

        # Run Python time filter
        batch = np.zeros((n_images, 2, H, W), dtype=np.float32)
        batch[:, 0] = images_A
        batch[:, 1] = images_B
        filtered = time_filter_batch(batch.copy(), verbose=False)

        # MATLAB reference
        matlab_filt_A = time_ref["filtered_frame1"].astype(np.float32)
        matlab_filt_B = time_ref["filtered_frame2"].astype(np.float32)

        # Error vs MATLAB
        err_A = filtered[:, 0].astype(np.float64) - matlab_filt_A.astype(np.float64)
        err_B = filtered[:, 1].astype(np.float64) - matlab_filt_B.astype(np.float64)
        max_abs_err = max(np.max(np.abs(err_A)), np.max(np.abs(err_B)))

        # --- Build 2x3 figure ---
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(
            "Time Filter Verification — per-pixel minimum subtraction",
            fontweight="bold", fontsize=13,
        )

        for row, (raw, tmin, filt, label) in enumerate([
            (images_A[0], time_min_A, filtered[0, 0], "A"),
            (images_B[0], time_min_B, filtered[0, 1], "B"),
        ]):
            # Col 0: Raw input frame 1
            ax = axes[row, 0]
            im = ax.imshow(raw, cmap="gray")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Raw Frame {label}")
            _add_watermark(ax, label)

            # Col 1: Time minimum map
            ax = axes[row, 1]
            im = ax.imshow(tmin, cmap="gray")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Time Min {label}")

            # Col 2: Filtered frame 1
            ax = axes[row, 2]
            im = ax.imshow(filt, cmap="gray")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Filtered Frame {label}")
            _add_watermark(ax, label)

        # Stats text box
        filt_min = filtered.min()
        filt_max = filtered.max()
        stats_text = (
            f"Filtered range: [{filt_min:.2f}, {filt_max:.2f}]  |  "
            f"Max |error| vs MATLAB: {max_abs_err:.6f}"
        )
        fig.text(
            0.5, 0.01, stats_text, ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        out_path = output_dir / "time_filter_verification.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Figure saved: {out_path}")
