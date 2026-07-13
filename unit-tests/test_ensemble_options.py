"""Tests for the ensemble noise-control options (2026-07-10).

Covers the two new C-correlator flags on ``bulkxcorr2d_accumulate_triple``
(per-window weighted-mean subtraction for the 'window_mean' background method,
and per-pair zero-lag-energy normalization), the ``round_shifts`` flag on the
fused symmetric warp, the ``kspace_linear`` fit-method dispatch recipe, and
the per-pair-normalization config validation rule.

The C libraries are exercised through raw ctypes (same pattern as
``test_peakfit_batch.py``) so the invariances are tested at the kernel
boundary, independent of the Dask pipeline.
"""

import ctypes
import platform
from pathlib import Path

import numpy as np
import pytest

_LIB_DIR = Path(__file__).resolve().parent.parent / "pivtools_cli" / "lib"
_LIB_EXT = ".dll" if platform.system() == "Windows" else ".so"
_XCORR_LIB = _LIB_DIR / f"libbulkxcorr2d{_LIB_EXT}"
_WARP_LIB = _LIB_DIR / f"libfusedwarp{_LIB_EXT}"

WIN = 32
IMG = 64
CENTER = WIN // 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def triple_lib():
    """Load libbulkxcorr2d and declare the accumulate_triple signature."""
    if not _XCORR_LIB.is_file():
        pytest.skip(f"libbulkxcorr2d{_LIB_EXT} not found at {_XCORR_LIB}")
    lib = ctypes.CDLL(str(_XCORR_LIB))
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    lib.bulkxcorr2d_accumulate_triple.restype = ctypes.c_ubyte
    lib.bulkxcorr2d_accumulate_triple.argtypes = [
        f32,
        f32,
        f32,
        i32,
        ctypes.c_int,
        f32,
        f32,
        i32,
        f32,
        f32,
        f32,
        f32,
        i32,
        i32,
        ctypes.c_int,
        ctypes.c_int,  # bMeanSubtract, bPerPairNorm
        f32,
        f32,
        f32,
    ]
    return lib


@pytest.fixture(scope="module")
def image_pairs():
    """Random particle-like image stacks, N pairs of (IMG, IMG)."""
    rng = np.random.default_rng(42)
    n = 5
    imgs_a = rng.random((n, IMG, IMG)).astype(np.float32) * 100
    imgs_b = rng.random((n, IMG, IMG)).astype(np.float32) * 100
    return imgs_a, imgs_b


def run_triple(lib, imgs_a, imgs_b, mean_subtract, per_pair_norm):
    """One centered WINxWIN window over an image stack; returns AB, AA, BB."""
    n = imgs_a.shape[0]
    weights = np.ones(WIN * WIN, dtype=np.float32)
    mask = np.zeros(1, dtype=np.float32)
    ab = np.zeros(WIN * WIN, dtype=np.float32)
    aa = np.zeros(WIN * WIN, dtype=np.float32)
    bb = np.zeros(WIN * WIN, dtype=np.float32)
    err = lib.bulkxcorr2d_accumulate_triple(
        np.ascontiguousarray(imgs_a),
        np.ascontiguousarray(imgs_b),
        mask,
        np.array([IMG, IMG], dtype=np.int32),
        n,
        np.array([IMG / 2 - 0.5], dtype=np.float32),
        np.array([IMG / 2 - 0.5], dtype=np.float32),
        np.array([1, 1], dtype=np.int32),
        weights,
        weights,
        weights,
        weights,
        np.array([WIN, WIN], dtype=np.int32),
        np.array([WIN, WIN], dtype=np.int32),
        mean_subtract,
        per_pair_norm,
        ab,
        aa,
        bb,
    )
    assert err == 0, f"C error code {err}"
    shape = (WIN, WIN)
    return ab.reshape(shape), aa.reshape(shape), bb.reshape(shape)


# ---------------------------------------------------------------------------
# Legacy regression: default flags reproduce the plain circular correlation
# ---------------------------------------------------------------------------
class TestLegacyRegression:
    def test_flags_off_match_numpy_reference(self, triple_lib, image_pairs):
        imgs_a, imgs_b = image_pairs
        ab, aa, _ = run_triple(triple_lib, imgs_a[:1], imgs_b[:1], 0, 0)

        lo = IMG // 2 - WIN // 2
        hi = lo + WIN
        wa = imgs_a[0, lo:hi, lo:hi].astype(np.float64)
        wb = imgs_b[0, lo:hi, lo:hi].astype(np.float64)
        # Kernel convention: B*conj(A), numpy-ifft normalization, fftshifted
        ref_ab = np.fft.fftshift(
            np.real(np.fft.ifft2(np.fft.fft2(wb) * np.conj(np.fft.fft2(wa))))
        )
        rel = np.abs(ab - ref_ab).max() / np.abs(ref_ab).max()
        assert rel < 1e-5

        # AA zero-lag equals the window energy sum(x^2) — the identity the
        # per-pair normalization relies on
        assert np.isclose(aa[CENTER, CENTER], (wa**2).sum(), rtol=1e-5)


# ---------------------------------------------------------------------------
# window_mean: per-pair offset invariance
# ---------------------------------------------------------------------------
class TestWindowMeanSubtraction:
    def test_constant_offset_invariance(self, triple_lib, image_pairs):
        """Adding a constant to one pair's frames must not change the planes."""
        imgs_a, imgs_b = image_pairs
        offs_a = imgs_a.copy()
        offs_b = imgs_b.copy()
        offs_a[2] += 37.0
        offs_b[2] += 12.0

        ab1, aa1, bb1 = run_triple(triple_lib, imgs_a, imgs_b, 1, 0)
        ab2, aa2, bb2 = run_triple(triple_lib, offs_a, offs_b, 1, 0)

        for p1, p2 in ((ab1, ab2), (aa1, aa2), (bb1, bb2)):
            rel = np.abs(p1 - p2).max() / np.abs(p1).max()
            assert rel < 1e-4

    def test_pedestal_removed(self, triple_lib, image_pairs):
        """With mean subtraction the plane far from the peak drops toward zero
        relative to the raw plane (the mean^2 pedestal is gone)."""
        imgs_a, imgs_b = image_pairs
        ab_raw, _, _ = run_triple(triple_lib, imgs_a, imgs_b, 0, 0)
        ab_ms, _, _ = run_triple(triple_lib, imgs_a, imgs_b, 1, 0)
        # Uncorrelated random frames: raw AB is dominated by the mean pedestal;
        # mean subtraction should remove almost all of it
        assert np.abs(ab_ms).mean() < 0.05 * np.abs(ab_raw).mean()


# ---------------------------------------------------------------------------
# per-pair normalization: gain invariance + unit auto peaks
# ---------------------------------------------------------------------------
class TestPerPairNormalization:
    def test_gain_invariance(self, triple_lib, image_pairs):
        """Scaling one pair's frames by gains must not change the sums."""
        imgs_a, imgs_b = image_pairs
        gain_a = imgs_a.copy()
        gain_b = imgs_b.copy()
        gain_a[1] *= 7.0
        gain_b[1] *= 3.0

        ab1, aa1, bb1 = run_triple(triple_lib, imgs_a, imgs_b, 1, 1)
        ab2, aa2, bb2 = run_triple(triple_lib, gain_a, gain_b, 1, 1)

        for p1, p2 in ((ab1, ab2), (aa1, aa2), (bb1, bb2)):
            rel = np.abs(p1 - p2).max() / np.abs(p1).max()
            assert rel < 1e-4

    def test_unit_auto_peaks(self, triple_lib, image_pairs):
        """Each pair contributes unit auto zero-lag: summed peak = N."""
        imgs_a, imgs_b = image_pairs
        n = imgs_a.shape[0]
        _, aa, bb = run_triple(triple_lib, imgs_a, imgs_b, 1, 1)
        assert np.isclose(aa[CENTER, CENTER], n, rtol=1e-3)
        assert np.isclose(bb[CENTER, CENTER], n, rtol=1e-3)

    def test_t_ratio_invariant_noise_free(self, triple_lib):
        """On identical (fully correlated) pairs, per-pair normalization must
        leave T = F_AB / sqrt(F_AA * F_BB) unchanged."""
        rng = np.random.default_rng(7)
        img = rng.random((1, IMG, IMG)).astype(np.float32) * 100
        ab0, aa0, bb0 = run_triple(triple_lib, img, img, 1, 0)
        ab1, aa1, bb1 = run_triple(triple_lib, img, img, 1, 1)

        def t_ratio(ab, aa, bb):
            fab = np.fft.fft2(np.fft.ifftshift(ab))
            faa = np.abs(np.fft.fft2(np.fft.ifftshift(aa)))
            fbb = np.abs(np.fft.fft2(np.fft.ifftshift(bb)))
            ref = np.sqrt(faa * fbb)
            return np.abs(fab) / np.maximum(ref, 1e-12)

        t0 = t_ratio(ab0, aa0, bb0)
        t1 = t_ratio(ab1, aa1, bb1)
        # compare where the reference spectrum carries signal
        sig = np.abs(np.fft.fft2(np.fft.ifftshift(aa0)))
        m = sig > 1e-3 * sig.max()
        assert np.abs(t0[m] - t1[m]).max() < 1e-3


# ---------------------------------------------------------------------------
# predictor rounding: rounded warp equals an integer roll
# ---------------------------------------------------------------------------
class TestRoundShifts:
    @pytest.fixture(scope="class")
    def warp_lib(self):
        if not _WARP_LIB.is_file():
            pytest.skip(f"libfusedwarp{_LIB_EXT} not found at {_WARP_LIB}")
        lib = ctypes.CDLL(str(_WARP_LIB))
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_int = ctypes.c_int
        lib.fused_symmetric_warp_batch.argtypes = [
            c_float_p,
            c_float_p,
            c_float_p,
            c_float_p,
            c_float_p,
            c_float_p,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_float_p,
            c_float_p,
            c_int,
            c_int,
            c_int,  # interp_mode, shared_predictor, round_shifts
        ]
        lib.fused_symmetric_warp_batch.restype = c_int
        return lib

    def test_rounded_warp_is_integer_roll(self, warp_lib):
        """Uniform even predictor + round_shifts=1 -> exact pixel roll of
        each frame by half the predictor (interior pixels)."""
        rng = np.random.default_rng(3)
        H = W = 64
        img_a = rng.random((1, H, W)).astype(np.float32)
        img_b = rng.random((1, H, W)).astype(np.float32)
        out_a = np.zeros_like(img_a)
        out_b = np.zeros_like(img_b)

        n_p = 4
        shift = 6.0  # even total -> half-shift 3 px, integer
        pred_dy = np.full((n_p, n_p), shift, dtype=np.float32)
        pred_dx = np.full((n_p, n_p), 0.0, dtype=np.float32)
        ctrs = np.linspace(8, 56, n_p).astype(np.float32)

        c_float_p = ctypes.POINTER(ctypes.c_float)
        ret = warp_lib.fused_symmetric_warp_batch(
            img_a.ctypes.data_as(c_float_p),
            img_b.ctypes.data_as(c_float_p),
            out_a.ctypes.data_as(c_float_p),
            out_b.ctypes.data_as(c_float_p),
            pred_dy.ctypes.data_as(c_float_p),
            pred_dx.ctypes.data_as(c_float_p),
            ctypes.c_int(1),
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_int(n_p),
            ctypes.c_int(n_p),
            ctrs.ctypes.data_as(c_float_p),
            ctrs.ctypes.data_as(c_float_p),
            ctypes.c_int(0),
            ctypes.c_int(1),
            ctypes.c_int(1),
        )
        assert ret == 0

        half = int(shift / 2)
        interior = slice(8, H - 8)
        # A sampled at i - half -> content shifts down by +half (np.roll +half)
        expect_a = np.roll(img_a[0], half, axis=0)
        expect_b = np.roll(img_b[0], -half, axis=0)
        assert (
            np.abs(out_a[0][interior, interior] - expect_a[interior, interior]).max()
            < 1e-5
        )
        assert (
            np.abs(out_b[0][interior, interior] - expect_b[interior, interior]).max()
            < 1e-5
        )


# ---------------------------------------------------------------------------
# kspace_linear dispatch recipe + validation rule
# ---------------------------------------------------------------------------
class TestFitterSelectionAndValidation:
    def test_kspace_linear_production_recipe_contract(self):
        """The dispatch recipe (joint/refc/gauss) recovers a synthetic
        Gaussian displacement on the 16-element output contract."""
        from synthetic_correlations import make_mock_config

        from pivtools_cli.piv.piv_backend.kspace_linear_fitting import (
            fit_windows_kspace_linear,
        )

        h = w = 32
        n = 4
        y, x = np.mgrid[0:h, 0:w]
        cy = cx = h // 2

        def g(sx, sy, dx=0.0, dy=0.0):
            return np.exp(
                -((x - cx - dx) ** 2 / (2 * sx**2) + (y - cy - dy) ** 2 / (2 * sy**2))
            )

        aa = np.stack([g(2.2, 2.2) for _ in range(n)])
        bb = aa.copy()
        ab = np.stack([0.8 * g(2.5, 2.5, 1.3, -0.7) for _ in range(n)])

        gauss, status, _ = fit_windows_kspace_linear(
            aa.ravel(),
            bb.ravel(),
            ab.ravel(),
            np.zeros(n, np.float32),
            (h, w),
            make_mock_config(),
            0,
            floor_mode="joint",
            weight_mode="refc",
            shape_mode="gauss",
        )
        assert gauss.shape == (n, 16)
        assert (status == 0).all()
        mu_x = gauss[:, 14] - gauss[:, 12]
        mu_y = gauss[:, 15] - gauss[:, 13]
        assert np.allclose(mu_x, 1.3, atol=0.02)
        assert np.allclose(mu_y, -0.7, atol=0.02)
        # sigma_AB^2 - sigma_A^2 = 2.5^2 - 2.2^2 = 1.41
        assert np.allclose(gauss[:, 9], 1.41, atol=0.05)

    def test_per_pair_norm_requires_window_mean(self):
        from pivtools_core.validation import validate_ensemble_config

        class FakeCfg:
            ensemble_type = ["std"]
            ensemble_window_sizes = [[32, 32]]
            ensemble_overlaps = [50]
            ensemble_sum_window = [32, 32]
            ensemble_sum_fitting_window_enabled = False
            ensemble_sum_fitting_window = None
            ensemble_fit_method = "kspace"
            ensemble_background_subtraction_method = "correlation"
            ensemble_per_pair_normalization = True
            ensemble_resume_from_pass = 0
            ensemble_num_passes = 1

        ok, errors, _ = validate_ensemble_config(FakeCfg())
        assert not ok
        assert any("per_pair_normalization" in e for e in errors)

        FakeCfg.ensemble_background_subtraction_method = "window_mean"
        ok, errors, _ = validate_ensemble_config(FakeCfg())
        assert ok, errors
