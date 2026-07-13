"""
Tests for lsqpeaklocate_lm from libbulkxcorr2d — instantaneous PIV peak finder.

Validates position and sigma recovery for all 4 fit types (3-pt parabolic,
4-DOF circular, 5-DOF elliptical, 6-DOF rotated) with noise-free synthetic
Gaussians.

Fit types:
  3 — 3-pt parabolic (each axis independently, no sigma output)
  4 — 4-DOF circular Gaussian: A·exp(-(i²+j²)/s²)
  5 — 5-DOF elliptical: A·exp(-(i²/sr² + j²/sc²))
  6 — 6-DOF rotated: A·exp(-0.5·(i²/vr + j²/vc + 2ij·cov))
"""

import ctypes
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_bulkxcorr_lib = None


def _load_bulkxcorr_lib():
    """Load libbulkxcorr2d, caching the result."""
    global _bulkxcorr_lib
    if _bulkxcorr_lib is not None:
        return _bulkxcorr_lib

    lib_extension = ".dll" if os.name == "nt" else ".so"

    possible_paths = [
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "pivtools_cli",
                "lib",
                f"libbulkxcorr2d{lib_extension}",
            )
        ),
        os.path.abspath(
            os.path.join(
                "pivtools_cli",
                "lib",
                f"libbulkxcorr2d{lib_extension}",
            )
        ),
    ]

    for path in possible_paths:
        if os.path.isfile(path):
            break
    else:
        raise FileNotFoundError(f"libbulkxcorr2d not found. Tried: {possible_paths}")

    lib = ctypes.CDLL(path)

    lib.lsqpeaklocate_lm.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_void_p,  # fPlaneWeight (nullable; None = no compensation)
    ]
    lib.lsqpeaklocate_lm.restype = None

    lib.peak_detectability.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
    ]
    lib.peak_detectability.restype = ctypes.c_float

    _bulkxcorr_lib = lib
    return lib


try:
    _load_bulkxcorr_lib()
    _LIB_AVAILABLE = True
except Exception:
    _LIB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _LIB_AVAILABLE,
    reason="libbulkxcorr2d C library not available",
)


# ---------------------------------------------------------------------------
# Gaussian generators (matching C code models)
# ---------------------------------------------------------------------------


def _generate_gaussian_4dof(shape, amp, i0, j0, s):
    """4-DOF circular: A·exp(-(di²+dj²)/s²). i0,j0 are offsets from center."""
    h, w = shape
    ci, cj = (h - 1) / 2, (w - 1) / 2
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    di = ii - ci - i0
    dj = jj - cj - j0
    return (amp * np.exp(-(di * di + dj * dj) / (s * s))).astype(np.float32)


def _generate_gaussian_5dof(shape, amp, i0, j0, sigma_row, sigma_col):
    """5-DOF elliptical: A·exp(-(di²/sr² + dj²/sc²))."""
    h, w = shape
    ci, cj = (h - 1) / 2, (w - 1) / 2
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    di = ii - ci - i0
    dj = jj - cj - j0
    Q = di * di / (sigma_row**2) + dj * dj / (sigma_col**2)
    return (amp * np.exp(-Q)).astype(np.float32)


def _generate_gaussian_6dof(shape, amp, i0, j0, var_row, var_col, cov_rowcol):
    """6-DOF rotated: A·exp(-0.5·(di²/vr + dj²/vc + 2·di·dj·cov))."""
    h, w = shape
    ci, cj = (h - 1) / 2, (w - 1) / 2
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    di = ii - ci - i0
    dj = jj - cj - j0
    Q = di * di / var_row + dj * dj / var_col + 2.0 * di * dj * cov_rowcol
    return (amp * np.exp(-0.5 * Q)).astype(np.float32)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_single_fit(xcorr, fit_type):
    """Run lsqpeaklocate_lm on one correlation plane.

    Returns
    -------
    peak_loc : ndarray, shape (3,)
        [row_position, col_position, peak_height]
    std_dev : ndarray, shape (3,)
        [sigma_row, sigma_col, sxy]
    """
    lib = _load_bulkxcorr_lib()

    xcorr = np.ascontiguousarray(xcorr, dtype=np.float32)
    N = np.array(xcorr.shape, dtype=np.int32)
    peak_loc = np.zeros(3, dtype=np.float32)
    std_dev = np.zeros(3, dtype=np.float32)

    lib.lsqpeaklocate_lm(xcorr, N, peak_loc, 1, fit_type, std_dev, None)

    return peak_loc, std_dev


# ═══════════════════════════════════════════════════════════════════════════
# Test classes
# ═══════════════════════════════════════════════════════════════════════════


class TestLibraryLoad:
    """Verify C library loads."""

    def test_load(self):
        lib = _load_bulkxcorr_lib()
        assert lib is not None


class TestPositionRecovery:
    """Test sub-pixel position recovery for all fit types."""

    @pytest.mark.parametrize("fit_type", [3, 4, 5, 6])
    @pytest.mark.parametrize(
        "shape",
        [
            (17, 17),
            (33, 33),
            (65, 65),
            (33, 65),
        ],
    )
    @pytest.mark.parametrize(
        "displacement",
        [
            (0.0, 0.0),
            (0.35, 0.27),
            (0.1, -0.4),
        ],
    )
    def test_position(self, fit_type, shape, displacement):
        i0_true, j0_true = displacement
        amp = 1000.0
        sigma = 1.5

        if fit_type in (3, 4):
            xcorr = _generate_gaussian_4dof(shape, amp, i0_true, j0_true, sigma)
        elif fit_type == 5:
            xcorr = _generate_gaussian_5dof(
                shape,
                amp,
                i0_true,
                j0_true,
                sigma,
                sigma,
            )
        else:  # 6
            var = sigma**2
            xcorr = _generate_gaussian_6dof(
                shape,
                amp,
                i0_true,
                j0_true,
                var,
                var,
                0.0,
            )

        peak_loc, _ = _run_single_fit(xcorr, fit_type)

        ci = (shape[0] - 1) / 2
        cj = (shape[1] - 1) / 2
        fit_i0 = peak_loc[0] - ci
        fit_j0 = peak_loc[1] - cj

        assert (
            abs(fit_i0 - i0_true) < 0.01
        ), f"row error: true={i0_true:.3f}, fit={fit_i0:.4f}"
        assert (
            abs(fit_j0 - j0_true) < 0.01
        ), f"col error: true={j0_true:.3f}, fit={fit_j0:.4f}"


class TestSigmaRecovery:
    """Test sigma/variance recovery for fit types that output widths."""

    @pytest.mark.parametrize("fit_type", [4, 5, 6])
    @pytest.mark.parametrize(
        "sigma_row, sigma_col",
        [
            (1.5, 1.5),
            (1.5, 2.0),
            (1.0, 3.0),
        ],
    )
    @pytest.mark.parametrize("shape", [(33, 33), (65, 65)])
    def test_sigma(self, fit_type, sigma_row, sigma_col, shape):
        amp = 1000.0
        i0, j0 = 0.35, 0.27

        if fit_type == 4:
            # Circular model — use average sigma
            s_avg = np.sqrt((sigma_row**2 + sigma_col**2) / 2)
            xcorr = _generate_gaussian_4dof(shape, amp, i0, j0, s_avg)
            true_sr = s_avg
            true_sc = s_avg
        elif fit_type == 5:
            xcorr = _generate_gaussian_5dof(
                shape,
                amp,
                i0,
                j0,
                sigma_row,
                sigma_col,
            )
            true_sr = sigma_row
            true_sc = sigma_col
        else:  # 6
            var_r = sigma_row**2
            var_c = sigma_col**2
            xcorr = _generate_gaussian_6dof(
                shape,
                amp,
                i0,
                j0,
                var_r,
                var_c,
                0.0,
            )
            # Type 6 outputs variances
            true_sr = var_r
            true_sc = var_c

        _, std_dev = _run_single_fit(xcorr, fit_type)
        fit_sr = std_dev[0]
        fit_sc = std_dev[1]

        tol = 0.01 if fit_type != 4 else 0.05  # Circular model less precise

        if fit_type == 4:
            # Check averaged sigma is close
            assert (
                abs(fit_sr - true_sr) / true_sr < tol
            ), f"sigma_row: true={true_sr:.3f}, fit={fit_sr:.4f}"
        else:
            assert (
                abs(fit_sr - true_sr) / true_sr < tol
            ), f"sigma_row: true={true_sr:.3f}, fit={fit_sr:.4f}"
            assert (
                abs(fit_sc - true_sc) / true_sc < tol
            ), f"sigma_col: true={true_sc:.3f}, fit={fit_sc:.4f}"


class TestAstigmatismChallenge:
    """All fitters on ONE elliptical Gaussian.

    Reveals which fitters introduce position bias when the peak is not
    circular. Type 4 (circular model) is expected to fail.
    """

    SIGMA_ROW = 1.5
    SIGMA_COL = 2.5
    I0 = 0.35
    J0 = 0.27
    SHAPE = (33, 33)

    @pytest.fixture
    def elliptical_xcorr(self):
        return _generate_gaussian_5dof(
            self.SHAPE,
            1000.0,
            self.I0,
            self.J0,
            self.SIGMA_ROW,
            self.SIGMA_COL,
        )

    @pytest.mark.parametrize("fit_type", [3, 5, 6])
    def test_correct_fitters_pass(self, fit_type, elliptical_xcorr):
        """Types 3, 5, 6 should handle elliptical peaks correctly."""
        peak_loc, _ = _run_single_fit(elliptical_xcorr, fit_type)

        ci = (self.SHAPE[0] - 1) / 2
        cj = (self.SHAPE[1] - 1) / 2

        err_i = abs(peak_loc[0] - ci - self.I0)
        err_j = abs(peak_loc[1] - cj - self.J0)
        pos_err = np.sqrt(err_i**2 + err_j**2)

        assert pos_err < 0.01, f"Type {fit_type} position error={pos_err:.4f} px"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Type 4 circular model cannot represent elliptical peaks — "
            "fitting all 25 points with r²/s² introduces position bias"
        ),
    )
    def test_type4_fails(self, elliptical_xcorr):
        """Type 4 should fail on elliptical input."""
        peak_loc, _ = _run_single_fit(elliptical_xcorr, 4)

        ci = (self.SHAPE[0] - 1) / 2
        cj = (self.SHAPE[1] - 1) / 2

        err_i = abs(peak_loc[0] - ci - self.I0)
        err_j = abs(peak_loc[1] - cj - self.J0)
        pos_err = np.sqrt(err_i**2 + err_j**2)

        assert (
            pos_err < 0.01
        ), f"Type 4 position error={pos_err:.4f} px (expected > 0.01)"


class TestFitFailureSentinel:
    """Failure honesty: invalid results must be NaN, never silently finite.

    Two failure layers exist in lsqpeaklocate_lm:
    1. The pre-fit peak-search gate (flat plane, border peak, non-local-max)
       — always emitted NaN.
    2. The LM fit itself (Cholesky breakdown, max_iter without convergence,
       stagnation with a large residual) — silently returned last-best params
       until 2026-07-06; now emits the same NaN sentinel.
    Both directions are tested: pathological planes -> NaN, and clean/noisy
    Gaussians -> finite (guards against over-aggressive failure marking,
    especially the lambda-blowup path which also fires on perfect seeds).
    """

    SHAPE = (33, 33)

    @staticmethod
    def _assert_nan_sentinel(peak_loc, std_dev, label):
        assert np.isnan(peak_loc[0]) and np.isnan(
            peak_loc[1]
        ), f"{label}: expected NaN row/col, got {peak_loc[:2]}"
        assert peak_loc[2] == 0, f"{label}: expected height 0, got {peak_loc[2]}"
        assert np.all(std_dev == 0), f"{label}: expected std_dev 0, got {std_dev}"

    # ── layer 1: the pre-fit search gate (pre-existing behaviour) ──────────

    @pytest.mark.parametrize("fit_type", [3, 4, 5, 6])
    def test_flat_zero_plane_is_nan(self, fit_type):
        """All-zero plane: no positive peak -> gate NaN."""
        xcorr = np.zeros(self.SHAPE, dtype=np.float32)
        peak_loc, std_dev = _run_single_fit(xcorr, fit_type)
        self._assert_nan_sentinel(peak_loc, std_dev, "flat zero")

    @pytest.mark.parametrize("fit_type", [4, 5, 6])
    def test_monotone_ramp_is_nan(self, fit_type):
        """Monotone ramp: search max sits on the search-region edge with a
        larger neighbour outside -> fails the strict local-max checks."""
        h, w = self.SHAPE
        xcorr = (
            np.arange(h, dtype=np.float32)[:, None]
            * np.ones(w, dtype=np.float32)[None, :]
        ) + 1.0
        peak_loc, std_dev = _run_single_fit(xcorr, fit_type)
        self._assert_nan_sentinel(peak_loc, std_dev, "monotone ramp")

    # ── layer 2: the LM fit itself (new sentinel) ──────────────────────────
    # Note: planes where LM converges to the least-squares optimum of the
    # model (plateau, delta, checkerboard) are KEPT — the fitter's job is
    # convergence, not model adequacy (Q-factor filtering handles that
    # downstream). Failure means: did not converge AND the residual is large.

    @pytest.mark.parametrize("fit_type", [4, 5, 6])
    def test_nan_poisoned_subwindow_is_nan(self, fit_type):
        """A NaN inside the 5x5 fit window (e.g. from a masked correlation
        plane) poisons every residual, so no step is ever accepted. The old
        code returned finite clamped garbage here; it must be NaN."""
        xcorr = _generate_gaussian_4dof(self.SHAPE, 1000.0, 0.0, 0.0, 1.5)
        c = (self.SHAPE[0] - 1) // 2
        xcorr[c + 1, c + 1] = (
            np.nan
        )  # inside the subwindow, off the gate's 4-neighbours
        peak_loc, std_dev = _run_single_fit(xcorr, fit_type)
        self._assert_nan_sentinel(peak_loc, std_dev, "NaN-poisoned window")

    @pytest.mark.parametrize("fit_type", [4, 5, 6])
    def test_uniform_noise_spike(self, fit_type):
        """Uniform noise with a spiked maximum — CONTRACT CHANGED 2026-07-13.

        With the double-precision improvement statistic + max_iter 50 the LM
        genuinely converges on this patch: it stalls on a degenerate wide
        blob and honestly measures a sub-tolerance step (the old float32
        arithmetic rejected this case only by accident — its noise floor hid
        the microscopic descent). Deliberate new contract (Westerweel-style,
        wiki sessions/2026-07-13-code1-lm-convergence-race.md):

        - gauss4/5: the converged blob has its width pinned exactly ON the
          step clamp (sigma = 4.0) -> the degenerate-width guard rejects it
          (NaN sentinel, as before).
        - gauss6: the blob converges with interior (huge but unclamped)
          widths -> the fitter EMITS a finite vector; rejecting dubious but
          well-formed fits is owned by the downstream validation stack
          (peak_mag threshold / normalized median), not the fitter.
        """
        rng = np.random.RandomState(0)
        xcorr = rng.uniform(0.0, 1.0, self.SHAPE).astype(np.float32)
        c = (self.SHAPE[0] - 1) // 2
        xcorr[c, c] = 2.0
        peak_loc, std_dev = _run_single_fit(xcorr, fit_type)
        if fit_type in (4, 5):
            self._assert_nan_sentinel(peak_loc, std_dev, "uniform noise + spike")
        else:
            assert np.all(np.isfinite(peak_loc[:2])), (
                "gauss6 noise blob must be emitted for downstream validation, "
                f"got {peak_loc[:2]}"
            )

    # ── no-regression guards: good fits must stay finite ───────────────────

    @pytest.mark.parametrize("fit_type", [4, 5, 6])
    def test_clean_gaussian_stays_finite(self, fit_type):
        """Noise-free Gaussian: the residual decays geometrically, so the
        relative-improvement test never fires and the loop runs to max_iter
        with a superb fit (or stagnates on an already-exact seed). Neither
        exit may be marked failed — the residual gate must keep them."""
        if fit_type in (4, 5):
            xcorr = _generate_gaussian_4dof(self.SHAPE, 1000.0, 0.35, 0.27, 1.5)
        else:
            xcorr = _generate_gaussian_6dof(
                self.SHAPE, 1000.0, 0.35, 0.27, 1.5**2, 1.5**2, 0.0
            )
        peak_loc, _ = _run_single_fit(xcorr, fit_type)
        assert np.all(
            np.isfinite(peak_loc)
        ), f"clean Gaussian marked failed: {peak_loc}"
        ci, cj = (self.SHAPE[0] - 1) / 2, (self.SHAPE[1] - 1) / 2
        assert abs(peak_loc[0] - ci - 0.35) < 0.01
        assert abs(peak_loc[1] - cj - 0.27) < 0.01

    @pytest.mark.parametrize("fit_type", [4, 5, 6])
    def test_noisy_gaussian_stays_finite(self, fit_type):
        """Gaussian + 1% deterministic noise: a realistic good fit must not
        be marked failed by the max_iter or stagnation classification."""
        rng = np.random.RandomState(42)
        xcorr = _generate_gaussian_4dof(self.SHAPE, 1000.0, 0.35, 0.27, 1.5)
        xcorr = xcorr + rng.normal(0.0, 10.0, self.SHAPE).astype(np.float32)
        peak_loc, _ = _run_single_fit(xcorr, fit_type)
        assert np.all(
            np.isfinite(peak_loc)
        ), f"noisy Gaussian marked failed: {peak_loc}"
        ci, cj = (self.SHAPE[0] - 1) / 2, (self.SHAPE[1] - 1) / 2
        assert abs(peak_loc[0] - ci - 0.35) < 0.1
        assert abs(peak_loc[1] - cj - 0.27) < 0.1


# ═══════════════════════════════════════════════════════════════════════════
# peak_detectability (PIVware peak1/peak2, 2026-07-13)
# ═══════════════════════════════════════════════════════════════════════════


class TestPeakDetectability:
    """Direct ctypes tests of the exported peak_detectability helper.

    Semantics under test (PIVware pwInterrogateDisplacement SNR type 2, over
    the same candidate set as the lsqpeaklocate_lm search): candidates are
    positive non-strict 3x3 local maxima inside the quarter region
    N/4..3N/4; ratio = tallest / second-tallest; 0 when there is no second
    positive candidate. Single-pixel spikes on a zero background are valid
    local maxima. Diagnostic only — stored per window, never gated on."""

    SHAPE = (33, 33)  # quarter region rows/cols 8..24

    def _ratio(self, plane):
        lib = _load_bulkxcorr_lib()
        plane = np.ascontiguousarray(plane, dtype=np.float32)
        n = np.array(plane.shape, dtype=np.int32)
        return float(lib.peak_detectability(plane, n))

    def test_two_known_peaks(self):
        plane = np.zeros(self.SHAPE, dtype=np.float32)
        plane[16, 16] = 10.0
        plane[16, 20] = 4.0
        assert self._ratio(plane) == pytest.approx(2.5)

    def test_single_peak_returns_zero(self):
        plane = np.zeros(self.SHAPE, dtype=np.float32)
        plane[16, 16] = 10.0
        assert self._ratio(plane) == 0.0

    def test_no_positive_candidate_returns_zero(self):
        assert self._ratio(np.full(self.SHAPE, -1.0, dtype=np.float32)) == 0.0
        assert self._ratio(np.zeros(self.SHAPE, dtype=np.float32)) == 0.0

    def test_equal_peaks_ratio_one(self):
        plane = np.zeros(self.SHAPE, dtype=np.float32)
        plane[12, 12] = 7.0
        plane[20, 20] = 7.0
        assert self._ratio(plane) == pytest.approx(1.0)

    def test_shoulder_is_not_second_peak(self):
        """A pixel adjacent to peak 1 is not a 3x3 local max and must not be
        counted as peak 2 (PIVware: the local-maxima map excludes shoulders
        by construction)."""
        plane = np.zeros(self.SHAPE, dtype=np.float32)
        plane[16, 16] = 10.0
        plane[16, 17] = 9.0  # shoulder of peak 1, NOT a local max
        plane[16, 21] = 4.0  # the true second peak
        assert self._ratio(plane) == pytest.approx(2.5)

    def test_peak_outside_quarter_region_ignored(self):
        plane = np.zeros(self.SHAPE, dtype=np.float32)
        plane[16, 16] = 10.0
        plane[2, 2] = 9.0  # outside rows/cols 8..24 — not a candidate
        assert self._ratio(plane) == 0.0
