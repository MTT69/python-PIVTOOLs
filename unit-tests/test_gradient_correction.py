"""Tests for the window/particle gradient correction, including the
deformation-aware residual-gradient behaviour (2026-07-20).

Physics under test: the L^2/12 window term models mean-shear broadening of the
ensemble correlation peak. Under image deformation the frames are resampled by
the predictor before correlation, so only the RESIDUAL gradient (U - pred)
broadens the peak. A converged warped pass (pred == U) must therefore receive
~zero window correction; pred=None / zeros must reproduce the historical
laboratory-frame formula exactly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pivtools_cli.piv.gradient_correction import (  # noqa: E402
    apply_gradient_correction_to_pass,
    compute_gradient_corrections,
)


def _shear_fields(ny=20, nx=30, shear=0.25, dy=4.0):
    """Linear shear U = shear*y, V = 0 on a window-centre grid."""
    y = np.arange(ny)[:, None] * dy
    U = np.broadcast_to(shear * y, (ny, nx)).copy()
    V = np.zeros((ny, nx))
    return U, V


COMMON = dict(dx=8.0, dy=4.0, window_size=(8, 96))  # (L_y, L_x)


def _window_corr(U, V, pred_U=None, pred_V=None):
    zeros = np.zeros_like(U)
    out = compute_gradient_corrections(
        U=U, V=V, sig_A_x=zeros, sig_A_y=zeros,
        UU_stress=np.full_like(U, 5.0), VV_stress=np.full_like(U, 2.0),
        UV_stress=np.full_like(U, -1.0),
        pred_U=pred_U, pred_V=pred_V, **COMMON,
    )
    return out  # 9-tuple


class TestUnwarpedBehaviourUnchanged:
    def test_linear_shear_matches_analytic_L2_12(self):
        shear = 0.25
        U, V = _shear_fields(shear=shear)
        (_, _, _, UUw, VVw, UVw, UUp, VVp, UVp) = _window_corr(U, V)
        expected = (8**2 / 12.0) * shear**2  # L_y^2/12 * (dU/dy)^2
        assert np.allclose(UUw, expected)
        assert np.allclose(VVw, 0.0)
        assert np.allclose(UVw, 0.0)
        assert np.allclose(UUp, 0.0)  # sig_A = 0 (k-space mode)

    def test_pred_none_equals_pred_zeros(self):
        U, V = _shear_fields()
        ref = _window_corr(U, V)
        zer = _window_corr(U, V, pred_U=np.zeros_like(U),
                           pred_V=np.zeros_like(V))
        for a, b in zip(ref, zer):
            assert np.allclose(a, b, equal_nan=True)


class TestResidualGradient:
    def test_fully_tracked_predictor_zeroes_window_correction(self):
        """Converged warped pass: pred == U -> no shear reaches the planes."""
        U, V = _shear_fields()
        (UUc, _, _, UUw, VVw, UVw, *_ ) = _window_corr(U, V, pred_U=U.copy(),
                                                       pred_V=V.copy())
        assert np.allclose(UUw, 0.0)
        assert np.allclose(VVw, 0.0)
        assert np.allclose(UVw, 0.0)
        assert np.allclose(UUc, 5.0)  # corrected == raw stress

    def test_partial_predictor_uses_residual(self):
        shear = 0.25
        U, V = _shear_fields(shear=shear)
        (_, _, _, UUw, *_ ) = _window_corr(U, V, pred_U=0.6 * U,
                                           pred_V=V.copy())
        expected = (8**2 / 12.0) * (0.4 * shear) ** 2
        assert np.allclose(UUw, expected)

    def test_nan_predictor_windows_fall_back_to_full_gradient(self):
        """NaN pred = un-warped window -> full laboratory-frame correction
        there; interior tracked windows still get ~zero."""
        shear = 0.25
        U, V = _shear_fields(shear=shear)
        pred = U.copy()
        pred[:, :5] = np.nan
        (_, _, _, UUw, *_ ) = _window_corr(U, V, pred_U=pred, pred_V=V.copy())
        full = (8**2 / 12.0) * shear**2
        # NaN block interior (away from the NaN/tracked seam) -> full corr
        assert np.allclose(UUw[:, :3], full)
        # tracked interior (away from the seam) -> zero
        assert np.allclose(UUw[:, 8:], 0.0)
        assert np.all(np.isfinite(UUw))


class TestApplyWrapper:
    def _apply(self, pred_x=None, pred_y=None):
        U, V = _shear_fields()
        zeros = np.zeros_like(U)
        return apply_gradient_correction_to_pass(
            ux=U, uy=V,
            UU_stress=np.full_like(U, 5.0), VV_stress=np.full_like(U, 2.0),
            UV_stress=np.full_like(U, -1.0),
            sig_A_x=zeros, sig_A_y=zeros,
            win_ctrs_x=np.arange(U.shape[1]) * 8.0,
            win_ctrs_y=np.arange(U.shape[0]) * 4.0,
            image_height=2048, window_size=(8, 96),
            pred_x=pred_x, pred_y=pred_y,
        )

    def test_wrapper_threads_predictor(self):
        U, V = _shear_fields()
        res = self._apply(pred_x=U.copy(), pred_y=V.copy())
        UU_window_corr = res[3]
        assert np.allclose(UU_window_corr, 0.0)

    def test_wrapper_without_predictor_is_historical(self):
        res = self._apply()
        UU_window_corr = res[3]
        assert np.allclose(UU_window_corr, (8**2 / 12.0) * 0.25**2)

    def test_shape_mismatch_raises(self):
        U, V = _shear_fields()
        with pytest.raises(ValueError, match="pred_x shape"):
            self._apply(pred_x=np.zeros((3, 3)), pred_y=V.copy())
