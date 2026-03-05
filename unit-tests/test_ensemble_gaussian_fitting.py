"""
Tests for the C Gaussian fitting library (marquadt_gaussian.c).

Verifies that the Levenberg-Marquardt fitter correctly recovers all 16
stacked Gaussian parameters. Tests both direct C calls (isolating the
fitter from initial-guess generation) and the production API
(fit_windows_openmp).

The 16-parameter stacked Gaussian model (INPUT convention — delta):
  [0-2]:   amp_A, amp_B, amp_AB          — peak amplitudes
  [3-5]:   c_A, c_B, c_AB                — DC offsets
  [6-8]:   sx_A, sy_A, sxy_A             — auto-correlation covariance
  [9-11]:  delta_x, delta_y, delta_xy     — extra AB covariance (delta = sigma_AB - sigma_A)
  [12-13]: x0_A, y0_A                    — auto-correlation center
  [14-15]: x0_AB, y0_AB                  — cross-correlation center

The C fitter OUTPUT converts [9-11] to total sigma: output[9] = sigma_A + delta.
"""

import ctypes

import numpy as np
import pytest

from tests.test_initial_guess import generate_2d_gaussian
from synthetic_correlations import flatten_for_gaussian, make_mock_config

# ---------------------------------------------------------------------------
# Library loading — skip entire module if C library not available
# ---------------------------------------------------------------------------

try:
    from pivtools_cli.piv.piv_backend.gaussian_fitting import (
        _load_marquadt_lib,
        set_offset_fitting,
        set_center_masking,
        fit_windows_openmp,
    )
    _load_marquadt_lib()
    _LIB_AVAILABLE = True
except Exception:
    _LIB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _LIB_AVAILABLE,
    reason="libmarquadt C library not available",
)

# ---------------------------------------------------------------------------
# Parameter name mapping
# ---------------------------------------------------------------------------

_EMPTY_SIGMA_DICT = {
    'sig_AB_x': None, 'sig_AB_y': None, 'sig_AB_xy': None,
    'sig_A_x': None, 'sig_A_y': None, 'sig_A_xy': None,
}

PARAM_NAMES = [
    'amp_A', 'amp_B', 'amp_AB',
    'c_A', 'c_B', 'c_AB',
    'sx_A', 'sy_A', 'sxy_A',
    'delta_x', 'delta_y', 'delta_xy',
    'x0_A', 'y0_A', 'x0_AB', 'y0_AB',
]


# ---------------------------------------------------------------------------
# Test-local helpers
# ---------------------------------------------------------------------------

def _generate_stacked_planes(shape, params, noise_std=0.0, rng=None):
    """Generate stacked AA, BB, AB planes using 1-based coordinate grids.

    Parameters
    ----------
    params : ndarray, shape (16,)
        Input-convention params: [9-11] are delta (extra covariance).
    """
    h, w = shape
    Y, X = np.meshgrid(np.arange(1, h + 1), np.arange(1, w + 1), indexing='ij')

    amp_A, amp_B, amp_AB = params[0], params[1], params[2]
    c_A, c_B, c_AB = params[3], params[4], params[5]
    sx_A, sy_A, sxy_A = params[6], params[7], params[8]
    delta_x, delta_y, delta_xy = params[9], params[10], params[11]
    x0_A, y0_A = params[12], params[13]
    x0_AB, y0_AB = params[14], params[15]

    AA = generate_2d_gaussian(X, Y, amp_A, x0_A, y0_A, sx_A, sy_A, sxy_A, c_A)
    BB = generate_2d_gaussian(X, Y, amp_B, x0_A, y0_A, sx_A, sy_A, sxy_A, c_B)

    # AB uses combined covariance: sigma_A + delta
    AB = generate_2d_gaussian(
        X, Y, amp_AB, x0_AB, y0_AB,
        sx_A + delta_x, sy_A + delta_y, sxy_A + delta_xy, c_AB,
    )

    if noise_std > 0 and rng is not None:
        noise_scale = amp_A * noise_std
        AA = AA + rng.normal(0, noise_scale, shape)
        BB = BB + rng.normal(0, noise_scale, shape)
        AB = AB + rng.normal(0, noise_scale, shape)

    return AA, BB, AB


def _build_expected_output(params):
    """Build expected C output from input params.

    The C fitter outputs total sigma for [9-11]:
        output[9] = sigma_A[6] + delta[9]
        output[10] = sigma_A[7] + delta[10]
        output[11] = sigma_A[8] + delta[11]
    All other params are unchanged.
    """
    expected = params.copy()
    expected[9] = params[6] + params[9]    # total_x = sx_A + delta_x
    expected[10] = params[7] + params[10]  # total_y = sy_A + delta_y
    expected[11] = params[8] + params[11]  # total_xy = sxy_A + delta_xy
    return expected


def _fit_single_window_direct(AA, BB, AB, initial_guess, win_size):
    """Call the C fitter directly with a known initial guess.

    Uses the 12-argument production signature (with uniform weights
    and pass_idx=0).
    """
    lib = _load_marquadt_lib()

    h, w = win_size
    n_per_window = h * w

    # 1-based coordinate grids matching C code convention
    Y, X = np.meshgrid(np.arange(1, h + 1), np.arange(1, w + 1), indexing='ij')
    X1 = Y.ravel(order='C').astype(np.float64)  # Y coordinates
    X2 = X.ravel(order='C').astype(np.float64)  # X coordinates

    # Pack: [AA | BB | AB]
    y_all = np.concatenate([AA.ravel(), BB.ravel(), AB.ravel()]).astype(np.float64)

    # Uniform weights (1.0)
    weights = np.ones(n_per_window, dtype=np.float64)

    result = np.zeros(16, dtype=np.float64)
    status = np.zeros(1, dtype=np.int32)

    lib.fit_stacked_gaussian_batch_export(
        ctypes.c_size_t(1),
        ctypes.c_size_t(n_per_window),
        ctypes.c_size_t(h),
        ctypes.c_size_t(w),
        X2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        initial_guess.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(0),  # pass_idx=0
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        status.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    return result, status[0]


def _make_default_params(win_size, sx_A=4.0, sy_A=4.0, delta_x=2.0, delta_y=2.0,
                         dx=0.0, dy=0.0, c=5.0):
    """Create a standard set of input-convention parameters.

    delta_x, delta_y are the extra covariance for AB beyond sigma_A.
    """
    center_x = win_size[1] / 2 + 1
    center_y = win_size[0] / 2 + 1
    return np.array([
        100.0, 100.0, 80.0,        # amplitudes
        c, c, c,                    # offsets
        sx_A, sy_A, 0.0,           # sigma_A
        delta_x, delta_y, 0.0,     # delta (extra AB covariance)
        center_x, center_y,        # center_A
        center_x + dx, center_y + dy,  # center_AB
    ])


def _assert_params_match(result, expected, rtol=0.005, atol=0.01, label=""):
    """Assert all 16 fitted params match expected values."""
    for i, name in enumerate(PARAM_NAMES):
        exp_val = expected[i]
        fit_val = result[i]
        if abs(exp_val) > 1e-6:
            rel_err = abs(fit_val - exp_val) / abs(exp_val)
            assert rel_err < rtol, (
                f"{label}{name}: expected={exp_val:.6f}, fit={fit_val:.6f}, "
                f"rel_err={rel_err:.4f}"
            )
        else:
            assert abs(fit_val - exp_val) < atol, (
                f"{label}{name}: expected={exp_val:.6f}, fit={fit_val:.6f}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Test classes
# ═══════════════════════════════════════════════════════════════════════════

class TestGaussianLibraryLoad:
    """Verify the C library loads successfully."""

    def test_load_library(self):
        lib = _load_marquadt_lib()
        assert lib is not None


class TestParameterRecovery:
    """Zero-noise parameter recovery via direct C call.

    With true parameters as initial guess, the fitter should converge to
    the exact input values. The C output converts delta[9-11] to total
    sigma (sigma_A + delta).
    """

    @pytest.mark.parametrize("win_size", [(32, 32), (64, 64), (32, 64)])
    @pytest.mark.parametrize("astigmatism", [
        ("circular", 4.0, 4.0),
        ("anisotropic", 6.0, 3.0),
    ], ids=lambda x: x[0])
    @pytest.mark.parametrize("displacement", [
        ("zero", 0.0, 0.0),
        ("sub_px", 2.5, 1.5),
    ], ids=lambda x: x[0])
    def test_exact_recovery(self, win_size, astigmatism, displacement):
        _, sx_A, sy_A = astigmatism
        _, dx, dy = displacement

        set_offset_fitting(True)
        set_center_masking(False)

        true_params = _make_default_params(
            win_size, sx_A=sx_A, sy_A=sy_A, dx=dx, dy=dy,
        )
        AA, BB, AB = _generate_stacked_planes(win_size, true_params)

        result, status = _fit_single_window_direct(
            AA, BB, AB, true_params.copy(), win_size,
        )

        assert status == 1, f"Fitter failed with status={status}"

        expected = _build_expected_output(true_params)
        _assert_params_match(result, expected)


class TestOffsetFitting:
    """Test offset (+c) fitting on vs off."""

    @pytest.mark.parametrize("offset_enabled", [True, False])
    @pytest.mark.parametrize("offset_value", [0.0, 5.0, 50.0])
    def test_offset_recovery(self, offset_enabled, offset_value):
        win_size = (32, 32)

        set_offset_fitting(offset_enabled)
        set_center_masking(False)

        true_params = _make_default_params(
            win_size, dx=2.0, dy=1.0, c=offset_value,
        )
        AA, BB, AB = _generate_stacked_planes(win_size, true_params)

        result, status = _fit_single_window_direct(
            AA, BB, AB, true_params.copy(), win_size,
        )

        assert status == 1, f"Fitter failed with status={status}"

        if offset_enabled or offset_value == 0.0:
            expected = _build_expected_output(true_params)
            _assert_params_match(result, expected)
        else:
            # Offset disabled + nonzero offset: expect model mismatch.
            # Positions should still be reasonable, sigmas may be biased.
            # Large offsets (50+) cause ~1-2 px position bias — model limitation.
            pos_tol = 2.0 if offset_value >= 50.0 else 1.0
            for idx in [12, 13, 14, 15]:
                exp_val = true_params[idx]
                fit_val = result[idx]
                assert abs(fit_val - exp_val) < pos_tol, (
                    f"{PARAM_NAMES[idx]}: position error > {pos_tol} px"
                )

        # Restore default
        set_offset_fitting(True)


class TestCenterMasking:
    """Test center pixel masking on vs off."""

    @pytest.mark.parametrize("mask_center", [True, False])
    @pytest.mark.parametrize("win_size", [(32, 32), (64, 64)])
    def test_convergence_on_clean_data(self, mask_center, win_size):
        """Both should converge on clean synthetic data."""
        set_offset_fitting(True)
        set_center_masking(mask_center)

        true_params = _make_default_params(win_size, dx=2.5, dy=1.5)
        AA, BB, AB = _generate_stacked_planes(win_size, true_params)

        result, status = _fit_single_window_direct(
            AA, BB, AB, true_params.copy(), win_size,
        )

        assert status == 1, f"Fitter failed with mask_center={mask_center}"

        expected = _build_expected_output(true_params)

        # Check positions are accurate
        for idx in [12, 13, 14, 15]:
            exp_val = expected[idx]
            fit_val = result[idx]
            rel_err = abs(fit_val - exp_val) / abs(exp_val)
            assert rel_err < 0.005, (
                f"{PARAM_NAMES[idx]}: rel_err={rel_err:.4f}"
            )

        # Restore default
        set_center_masking(True)


class TestReynoldsStressRecovery:
    """Verify delta recovery for Reynolds stress calculation.

    The C output [9] = sigma_A + delta. We check delta = result[9] - result[6].
    R_uu = delta^2, so even small errors compound into stress errors.
    """

    @pytest.mark.parametrize("delta", [0.3, 0.5, 1.0, 2.0, 3.0])
    @pytest.mark.parametrize("win_size", [(32, 32), (64, 64)])
    def test_delta_recovery(self, delta, win_size):
        set_offset_fitting(True)
        set_center_masking(False)

        true_params = _make_default_params(
            win_size, delta_x=delta, delta_y=delta, dx=2.0, dy=1.0,
        )
        AA, BB, AB = _generate_stacked_planes(win_size, true_params)

        result, status = _fit_single_window_direct(
            AA, BB, AB, true_params.copy(), win_size,
        )

        assert status == 1

        # Recover delta from output: total_sigma - sigma_A
        fit_delta_x = result[9] - result[6]
        rel_err = abs(fit_delta_x - delta) / delta
        assert rel_err < 0.01, (
            f"delta_x: true={delta:.3f}, fit={fit_delta_x:.6f}, "
            f"rel_err={rel_err:.4f}"
        )

        # Check R_uu = delta^2
        R_true = delta ** 2
        R_fit = fit_delta_x ** 2
        R_err = abs(R_fit - R_true) / R_true
        assert R_err < 0.02, f"R_uu error: {R_err:.4f}"


class TestPerturbedInitialGuess:
    """Test convergence with perturbed initial guesses."""

    @pytest.mark.parametrize("perturbation_pct", [0.1, 0.2, 0.5])
    def test_convergence_with_perturbation(self, perturbation_pct):
        win_size = (32, 32)
        rng = np.random.default_rng(42)

        set_offset_fitting(True)
        set_center_masking(False)

        true_params = _make_default_params(win_size, dx=2.5, dy=1.5)
        AA, BB, AB = _generate_stacked_planes(win_size, true_params)

        # Perturb initial guess
        perturbation = 1.0 + perturbation_pct * (2 * rng.random(16) - 1)
        initial_guess = true_params * perturbation

        # Keep positions close (within 2 pixels)
        initial_guess[12:16] = true_params[12:16] + 2 * (rng.random(4) - 0.5)

        # Ensure sigmas stay positive
        initial_guess[6:12] = np.maximum(initial_guess[6:12], 0.1)

        result, status = _fit_single_window_direct(
            AA, BB, AB, initial_guess, win_size,
        )

        assert status == 1, f"Fitter failed with {perturbation_pct*100:.0f}% perturbation"

        # Check delta recovery — zero noise, exact model → must converge <2%
        fit_delta_x = result[9] - result[6]
        true_delta_x = true_params[9]
        rel_err = abs(fit_delta_x - true_delta_x) / true_delta_x

        assert rel_err < 0.02, (
            f"delta_x error={rel_err*100:.2f}% at {perturbation_pct*100:.0f}% "
            f"perturbation (true={true_delta_x:.3f}, fit={fit_delta_x:.4f})"
        )


class TestFitWindowsOpenmp:
    """Integration test for the production fit_windows_openmp() API.

    Tests the full Python+C pipeline including initial-guess generation,
    grid setup, and post-fit validation.
    """

    @pytest.mark.parametrize("win_size", [(32, 32), (64, 64)])
    def test_output_shapes(self, win_size):
        """Output arrays should have correct shapes."""
        n_windows = 4

        config = make_mock_config(
            ensemble_window_sizes=[list(win_size)],
            ensemble_type=['std'],
            ensemble_fit_offset=True,
            ensemble_mask_center_pixel=False,
        )

        true_params = _make_default_params(win_size, dx=2.0, dy=1.0)
        AA, BB, AB = _generate_stacked_planes(win_size, true_params)

        R_AA_flat = np.tile(AA.ravel(), n_windows)
        R_BB_flat = np.tile(BB.ravel(), n_windows)
        R_AB_flat = np.tile(AB.ravel(), n_windows)
        mask_flat = np.zeros(n_windows, dtype=bool)

        gauss, status, initial = fit_windows_openmp(
            R_AA_flat, R_BB_flat, R_AB_flat, mask_flat,
            sigma_dict=_EMPTY_SIGMA_DICT, corr_size=win_size,
            config=config, pass_idx=0, num_threads=1,
        )

        assert gauss.shape == (n_windows, 16)
        assert status.shape == (n_windows,)
        assert initial.shape == (n_windows, 16)

    @pytest.mark.parametrize("win_size", [(32, 32), (64, 64)])
    def test_displacement_recovery(self, win_size):
        """Production API should recover displacement direction."""
        n_windows = 1
        dx, dy = 2.0, 1.0

        config = make_mock_config(
            ensemble_window_sizes=[list(win_size)],
            ensemble_type=['std'],
            ensemble_fit_offset=True,
            ensemble_mask_center_pixel=False,
        )

        true_params = _make_default_params(win_size, dx=dx, dy=dy)
        AA, BB, AB = _generate_stacked_planes(win_size, true_params)

        R_AA_flat = AA.ravel()
        R_BB_flat = BB.ravel()
        R_AB_flat = AB.ravel()
        mask_flat = np.zeros(n_windows, dtype=bool)

        gauss, status, _ = fit_windows_openmp(
            R_AA_flat, R_BB_flat, R_AB_flat, mask_flat,
            sigma_dict=_EMPTY_SIGMA_DICT, corr_size=win_size,
            config=config, pass_idx=0, num_threads=1,
        )

        # Status 0 = success after Python validation
        assert status[0] == 0, f"Fit failed with status={status[0]}"

        # Check that AB center moved in the right direction
        x0_A = gauss[0, 12]
        x0_AB = gauss[0, 14]
        y0_A = gauss[0, 13]
        y0_AB = gauss[0, 15]

        assert (x0_AB - x0_A) * dx >= 0, "x displacement has wrong sign"
        assert (y0_AB - y0_A) * dy >= 0, "y displacement has wrong sign"

    def test_masked_windows(self):
        """Masked windows should get status=-1 and zero result."""
        win_size = (32, 32)
        n_windows = 4

        config = make_mock_config(
            ensemble_window_sizes=[list(win_size)],
            ensemble_type=['std'],
            ensemble_fit_offset=True,
            ensemble_mask_center_pixel=False,
        )

        true_params = _make_default_params(win_size)
        AA, BB, AB = _generate_stacked_planes(win_size, true_params)

        R_AA_flat = np.tile(AA.ravel(), n_windows)
        R_BB_flat = np.tile(BB.ravel(), n_windows)
        R_AB_flat = np.tile(AB.ravel(), n_windows)

        mask_flat = np.array([False, True, False, True], dtype=bool)

        gauss, status, _ = fit_windows_openmp(
            R_AA_flat, R_BB_flat, R_AB_flat, mask_flat,
            sigma_dict=_EMPTY_SIGMA_DICT, corr_size=win_size,
            config=config, pass_idx=0, num_threads=1,
        )

        assert status[1] == -1
        assert status[3] == -1
        assert np.all(gauss[1] == 0)
        assert np.all(gauss[3] == 0)

    def test_all_masked(self):
        """All-masked edge case should return immediately."""
        win_size = (32, 32)
        n_windows = 2

        config = make_mock_config(
            ensemble_window_sizes=[list(win_size)],
            ensemble_type=['std'],
        )

        AA = np.zeros(win_size)
        R_AA_flat = np.tile(AA.ravel(), n_windows)

        mask_flat = np.ones(n_windows, dtype=bool)

        gauss, status, initial = fit_windows_openmp(
            R_AA_flat, R_AA_flat, R_AA_flat, mask_flat,
            sigma_dict=_EMPTY_SIGMA_DICT, corr_size=win_size,
            config=config, pass_idx=0, num_threads=1,
        )

        assert np.all(status == -1)
        assert np.all(gauss == 0)
