"""Quintic B-spline ('bspline5') kernel for the analytic coloured noise floor.

A prefiltered quintic B-spline warp is an IIR prefilter followed by the
6-tap beta5 basis, so its effective interpolator has infinite, exponentially
decaying support. The floor needs the per-shift effective weights c_n(f)
(``KERNEL_TAPS['bspline5']``), truncated at a proven tolerance.

Evidence layers:
  1. The effective weights: delta at f = 0, partition of unity, the closed
     form against scipy.ndimage.map_coordinates(order=5, mode='mirror') on a
     delta image (an independent implementation of the same interpolator),
     and the truncation bound.
  2. The analytic floor against the empirical expected spectrum of white
     noise warped by scipy's quintic spline, shared and per-image
     predictors, with a tolerance set by the estimator's own standard error.
"""

import numpy as np
import pytest
from scipy import ndimage

from pivtools_cli.piv.piv_backend.interpolation_noise_psd import (
    BSPLINE5_OFFSETS,
    BSPLINE5_POLY,
    BSPLINE5_TAIL_TOL,
    BSPLINE5_Z1,
    BSPLINE5_Z2,
    bspline5_tail_bound,
    bspline5_weights,
)
from pivtools_cli.piv.piv_backend.kspace_floor_psd import (
    KERNEL_TAPS,
    analytic_floor_single,
    tap_autocorr,
)

FRACS = np.linspace(0.0, 1.0, 41, endpoint=False)
# scipy evaluates the same interpolator in float64 by an independent route
# (its own recursive prefilter with mirror boundary); agreement is limited by
# float64 round-off on the O(1) weights, ~1e-15. 1e-12 leaves headroom.
SCIPY_WEIGHT_ATOL = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 1. Effective weights
# ─────────────────────────────────────────────────────────────────────────────


class TestBspline5Weights:
    def test_registered_in_kernel_taps(self):
        offsets, weights_fun = KERNEL_TAPS["bspline5"]
        assert np.array_equal(offsets, BSPLINE5_OFFSETS)
        assert weights_fun is bspline5_weights

    def test_closed_form_poles_are_the_prefilter_roots(self):
        """The two closed-form poles are the in-circle roots of
        z^4 + 26 z^3 + 66 z^2 + 26 z + 1 (120 z^2 b(z))."""
        roots = np.roots(BSPLINE5_POLY)
        inside = np.sort(roots[np.abs(roots) < 1.0].real)
        assert np.allclose(inside, sorted([BSPLINE5_Z1, BSPLINE5_Z2]), rtol=1e-13)

    def test_offsets_contiguous_and_centred(self):
        """Tap layout of the other kernels: first tap at floor - (T/2 - 1)."""
        T = BSPLINE5_OFFSETS.size
        assert T % 2 == 0
        assert np.array_equal(BSPLINE5_OFFSETS, np.arange(-(T // 2 - 1), T // 2 + 1))

    def test_f0_is_delta(self):
        """Interpolating spline: integer samples are reproduced exactly."""
        c = bspline5_weights(0.0)
        expect = (BSPLINE5_OFFSETS == 0).astype(float)
        assert np.max(np.abs(c - expect)) < 1e-12

    @pytest.mark.parametrize("f", FRACS)
    def test_partition_of_unity(self, f):
        """Constants are reproduced: the weights sum to 1 at every f."""
        assert abs(bspline5_weights(f).sum() - 1.0) < 1e-9

    @pytest.mark.parametrize("f", [0.0, 0.1, 0.25, 0.37, 0.5, 0.75, 0.93])
    def test_matches_scipy_map_coordinates(self, f):
        """c_n(f) is the weight the quintic interpolator puts on pixel
        floor + n when sampling at floor + f. scipy, sampling a unit delta at
        x0 + n at the position x0 + f, returns exactly that weight."""
        L = 401
        x0 = L // 2
        img = np.zeros(L)
        img[x0] = 1.0
        # delta at x0, sample at x0 + f - n  <=>  delta at x0 + n, sample x0 + f
        vals = ndimage.map_coordinates(
            img, [x0 + f - BSPLINE5_OFFSETS.astype(float)], order=5, mode="mirror"
        )
        assert np.max(np.abs(vals - bspline5_weights(f))) < SCIPY_WEIGHT_ATOL

    def test_truncation_bound_holds(self):
        """The dropped tail (|n| beyond the stored taps), summed far out with
        the closed form, is below the proven bound, which is below the
        tolerance the tap count was chosen for."""
        bound = bspline5_tail_bound(BSPLINE5_OFFSETS)
        assert bound <= BSPLINE5_TAIL_TOL
        far = np.arange(-400, 401)
        keep = (far >= BSPLINE5_OFFSETS[0]) & (far <= BSPLINE5_OFFSETS[-1])
        for f in FRACS:
            c_far = bspline5_weights(f, offsets=far)
            assert np.abs(c_far[~keep]).sum() <= bound

    def test_symmetry_reverses_weights(self):
        """c_n(1 - f) = c_{1-n}(f): the shift f and 1 - f give mirrored
        weights, so the tap autocorrelation depends only on frac_distance."""
        f = 0.3
        far = np.arange(-60, 62)
        c_f = bspline5_weights(f, offsets=far)
        c_mirror = bspline5_weights(1.0 - f, offsets=1 - far)
        assert np.max(np.abs(c_f - c_mirror)) < 1e-14
        _, r_a = tap_autocorr(f, "bspline5")
        _, r_b = tap_autocorr(1.0 - f, "bspline5")
        assert np.max(np.abs(r_a - r_b)) < 1e-14

    def test_out_of_range_fraction_raises(self):
        with pytest.raises(ValueError, match="fractional"):
            bspline5_weights(1.0)
        with pytest.raises(ValueError, match="fractional"):
            bspline5_weights(-0.1)

    def test_unknown_kernel_still_raises(self):
        with pytest.raises(ValueError, match="unknown kernel"):
            tap_autocorr(0.2, "bspline3")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Analytic floor vs the empirical spectrum of quintic-warped white noise
# ─────────────────────────────────────────────────────────────────────────────
#
# Truth: unit white noise warped by scipy's quintic spline (an implementation
# independent of KERNEL_TAPS), frame A sampled at x - d/2 and frame B at
# x + d/2 as the symmetric warp does. The compared quantity is the expected
# weighted, mean-removed auto spectrum E_S(k) = E|DFT2(w_B (u - m))|^2: with
# env = 1 and no crop analytic_floor_single returns |fftshift(E_S)| at the
# absolute level (unit noise variance), so there is no free scale.
#
# Estimator: the mean periodogram over windows. Windows sit _MARGIN px inside
# the image (the mirror-boundary prefilter transient decays as |z1|^distance,
# 0.43^40 = 2e-15) and _GAP px apart (warped-noise covariance at lag 40 is
# below 1e-13, measured), so they are independent samples; the standard
# error per bin is the sample std over windows / sqrt(n).

_IMG = 400
_MARGIN = 40
_GAP = 40
# Bonferroni bound for the max per-bin |z| over the (sh * sw) bins at
# family-wise false-alarm 1e-4: Phi^-1(1 - 1e-4 / (2 * 256)) = 4.6 for 16x16.
_Z_FAMILY_ALPHA = 1e-4


def _z_max(n_bins: int) -> float:
    from scipy.stats import norm

    return float(norm.isf(_Z_FAMILY_ALPHA / (2.0 * n_bins)))


def _frac_distance(x: float) -> float:
    return abs(x - round(x))


def _warp_pair_scipy(rng, dy: float, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """White-noise A and B frames warped symmetrically by (dy, dx) with the
    quintic spline (prefiltered, mirror boundary), float64."""
    yy, xx = np.mgrid[0:_IMG, 0:_IMG].astype(np.float64)
    out = []
    for sign in (-1.0, 1.0):
        u = rng.standard_normal((_IMG, _IMG))
        out.append(
            ndimage.map_coordinates(
                u, [yy + sign * dy / 2.0, xx + sign * dx / 2.0], order=5, mode="mirror"
            )
        )
    return out[0], out[1]


def _window_periodograms(frame: np.ndarray, w_B: np.ndarray, hole: bool) -> np.ndarray:
    """|DFT2(w_B (u - m))|^2 for every interior window, (n, sh, sw), fftshifted."""
    sh, sw = w_B.shape
    starts_y = range(_MARGIN, _IMG - _MARGIN - sh + 1, sh + _GAP)
    starts_x = range(_MARGIN, _IMG - _MARGIN - sw + 1, sw + _GAP)
    wins = np.stack(
        [frame[y : y + sh, x : x + sw] for y in starts_y for x in starts_x]
    )
    if hole:
        wins = wins - (wins * w_B).sum(axis=(1, 2), keepdims=True) / w_B.sum()
    return np.fft.fftshift(np.abs(np.fft.fft2(wins * w_B)) ** 2, axes=(-2, -1))


def _z_scores(per_window: np.ndarray, model: np.ndarray) -> np.ndarray:
    """(empirical mean - model) / standard error, on bins with signal."""
    n = per_window.shape[0]
    mean = per_window.mean(axis=0)
    se = per_window.std(axis=0, ddof=1) / np.sqrt(n)
    live = model > 1e-12 * model.max()  # drops the exact DC hole
    return (mean[live] - model[live]) / se[live]


def _flat_w(n=16):
    return np.ones((n, n))


def _bsingle_w(n=24):
    from pivtools_cli.piv.piv_backend.base import CrossCorrelator

    return np.asarray(
        CrossCorrelator._window_weight_fun((n, n), "bsingle", (n, n)), dtype=np.float64
    )


_N_IMAGES = 24


class TestFloorVsQuinticWarpedNoise:
    @pytest.mark.parametrize(
        "dy, dx",
        [(0.0, 0.5), (0.0, 1.0), (1.0, 0.5), (-1.7, 3.3)],
        ids=["fx0.25", "fx0.5", "fx0.25_fy0.5", "fx0.35_fy0.15"],
    )
    @pytest.mark.parametrize("weight, seed", [("flat16", 31), ("bsingle24", 37)])
    def test_shared_predictor(self, dy, dx, weight, seed):
        """Every pair warped by the same (dy, dx): frames A and B each match
        analytic_floor_single at (frac_distance(dx/2), frac_distance(dy/2))."""
        w_B = _flat_w() if weight == "flat16" else _bsingle_w()
        rng = np.random.default_rng([seed, int(100 * dy) % 1000, int(100 * dx) % 1000])
        fx, fy = _frac_distance(dx / 2.0), _frac_distance(dy / 2.0)
        model = analytic_floor_single(fx, fy, np.ones(w_B.shape), w_B, "bspline5", True)
        pa, pb = [], []
        for _ in range(_N_IMAGES):
            a, b = _warp_pair_scipy(rng, dy, dx)
            pa.append(_window_periodograms(a, w_B, True))
            pb.append(_window_periodograms(b, w_B, True))
        z_max = _z_max(w_B.size)
        for per_window in (np.concatenate(pa), np.concatenate(pb)):
            z = _z_scores(per_window, model)
            assert np.max(np.abs(z)) < z_max, f"max |z| {np.max(np.abs(z)):.2f}"

    @pytest.mark.parametrize("hole", [True, False])
    def test_per_image_predictor(self, hole):
        """Each pair warped by its own (dy_i, dx_i): the ensemble spectrum is
        the mean of the per-pair floors (the floor is linear in the noise
        covariance), so it matches the mean analytic floor over the pairs'
        fractions."""
        w_B = _flat_w()
        rng = np.random.default_rng(23 + int(hole))
        per_window, models = [], []
        for _ in range(_N_IMAGES):
            dy, dx = rng.uniform(-3.0, 3.0, 2)
            a, _ = _warp_pair_scipy(rng, dy, dx)
            pw = _window_periodograms(a, w_B, hole)
            per_window.append(pw)
            fx, fy = _frac_distance(dx / 2.0), _frac_distance(dy / 2.0)
            m = analytic_floor_single(fx, fy, np.ones(w_B.shape), w_B, "bspline5", hole)
            models.append(np.broadcast_to(m, pw.shape))
        per_window = np.concatenate(per_window)
        model = np.concatenate(models).mean(axis=0)
        z = _z_scores(per_window, model)
        assert np.max(np.abs(z)) < _z_max(w_B.size), f"max |z| {np.max(np.abs(z)):.2f}"

    def test_wrong_kernel_is_rejected(self):
        """Power control: the same estimator and tolerance reject the Lanczos-3
        and Keys floors for quintic-warped noise at f = 0.5, so the gate above
        can tell kernels apart."""
        w_B = _flat_w()
        rng = np.random.default_rng(99)
        per_window = np.concatenate(
            [
                _window_periodograms(_warp_pair_scipy(rng, 0.0, 1.0)[0], w_B, True)
                for _ in range(_N_IMAGES)
            ]
        )
        for kernel in ("lanczos", "cubic"):
            wrong = analytic_floor_single(
                0.5, 0.0, np.ones(w_B.shape), w_B, kernel, True
            )
            assert np.max(np.abs(_z_scores(per_window, wrong))) > _z_max(w_B.size)


class TestTruncationAtFloorLevel:
    def test_floor_matches_long_reference(self, monkeypatch):
        """The 60-tap floor equals a 120-tap reference (tail bound 7e-21, i.e.
        exact in float64) to within BSPLINE5_TAIL_TOL of max P, through the
        full single chain (crop + envelope divide), hole on and off."""
        import functools

        from pivtools_cli.piv.piv_backend import kspace_floor_psd
        from pivtools_cli.piv.piv_backend.single_pass_accumulator import (
            _linear_pair_envelope,
        )

        ref_offsets = np.arange(-59, 61)
        assert bspline5_tail_bound(ref_offsets) < 1e-19
        monkeypatch.setitem(
            kspace_floor_psd.KERNEL_TAPS,
            "bspline5_ref",
            (ref_offsets, functools.partial(bspline5_weights, offsets=ref_offsets)),
        )
        w_B = _bsingle_w(24)
        env = _linear_pair_envelope(w_B, w_B, (16, 16))
        for fx, fy in ((0.5, 0.5), (0.25, 0.1), (0.37, 0.0)):
            for hole in (True, False):
                P = analytic_floor_single(fx, fy, env, w_B, "bspline5", hole)
                P_ref = analytic_floor_single(fx, fy, env, w_B, "bspline5_ref", hole)
                assert np.abs(P - P_ref).max() <= BSPLINE5_TAIL_TOL * P_ref.max()
