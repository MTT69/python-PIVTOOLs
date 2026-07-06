"""
Tests for lsqpeaklocate_lm_batch — the lockstep one-window-per-SIMD-lane LM
peak fitter (peak_locate_lm_batch.c) — against the scalar oracle
lsqpeaklocate_lm from the same libbulkxcorr2d build.

The heavy numerical gates live in the standalone C harness
(pivtools_cli/lib/test_peakfit_gate.c, run at build verification time); this
suite proves the DLL-exported surface: oracle agreement through ctypes,
NaN-mask equality, partial batches, and the selector's no-silent-fallback
semantics.
"""

import ctypes
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Library loading (pattern: test_instantaneous_peaks.py)
# ---------------------------------------------------------------------------

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib_extension = ".dll" if os.name == "nt" else ".so"
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'pivtools_cli', 'lib',
        f'libbulkxcorr2d{lib_extension}',
    ))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"libbulkxcorr2d not found at {path}")

    lib = ctypes.CDLL(path)

    lib.lsqpeaklocate_lm.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
        ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    ]
    lib.lsqpeaklocate_lm.restype = None

    lib.peakfit_batch_available.argtypes = []
    lib.peakfit_batch_available.restype = ctypes.c_int
    lib.peakfit_batch_lanes.argtypes = []
    lib.peakfit_batch_lanes.restype = ctypes.c_int
    lib.bulkxcorr2d_set_peakfit_impl.argtypes = [ctypes.c_int]
    lib.bulkxcorr2d_set_peakfit_impl.restype = ctypes.c_int
    lib.bulkxcorr2d_get_peakfit_impl.argtypes = []
    lib.bulkxcorr2d_get_peakfit_impl.restype = ctypes.c_int
    lib.bulkxcorr2d_peakfit_batch_available.argtypes = []
    lib.bulkxcorr2d_peakfit_batch_available.restype = ctypes.c_int

    lib.lsqpeaklocate_lm_batch.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),  # planes
        ctypes.c_int,                                                    # L_real
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),    # N
        ctypes.c_int,                                                    # iFitType
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),  # peak_loc [3][W]
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),  # std_dev  [3][W]
    ]
    lib.lsqpeaklocate_lm_batch.restype = None

    _lib = lib
    return lib


try:
    _LIB = _load_lib()
    _AVAILABLE = bool(_LIB.peakfit_batch_available())
    _LANES = int(_LIB.peakfit_batch_lanes())
except Exception:
    _LIB, _AVAILABLE, _LANES = None, False, 0

pytestmark = pytest.mark.skipif(
    _LIB is None, reason="libbulkxcorr2d C library not available",
)

needs_batch = pytest.mark.skipif(
    not _AVAILABLE, reason="batch peak fitter not compiled in (plain MSVC cl build)",
)

PLANE_N = 33
CENTER = (PLANE_N - 1) / 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gauss_plane(rng, fit_type):
    """One clean synthetic plane matching the C model for the fit type."""
    di, dj = rng.uniform(-0.45, 0.45, 2)
    amp = rng.uniform(100.0, 1000.0)
    s1, s2 = rng.uniform(1.0, 2.5, 2)
    ii, jj = np.meshgrid(np.arange(PLANE_N), np.arange(PLANE_N), indexing='ij')
    y = ii - CENTER - di
    x = jj - CENTER - dj
    if fit_type == 4:
        z = amp * np.exp(-(y * y + x * x) / (s1 * s1))
    elif fit_type == 5:
        z = amp * np.exp(-(y * y / (s1 * s1) + x * x / (s2 * s2)))
    else:
        cov = rng.uniform(-0.2, 0.2) / (s1 * s2)
        z = amp * np.exp(-0.5 * (y * y / (s1 * s1) + x * x / (s2 * s2) + 2 * y * x * cov))
    return z.astype(np.float32)


def _run_scalar(plane, fit_type):
    N = np.array(plane.shape, dtype=np.int32)
    loc = np.zeros(3, dtype=np.float32)
    std = np.zeros(3, dtype=np.float32)
    _LIB.lsqpeaklocate_lm(np.ascontiguousarray(plane), N, loc, 1, fit_type, std)
    return loc, std


def _run_batch(planes, L_real, fit_type):
    """planes: (n, H, W) with n <= lanes."""
    n, h, w = planes.shape
    buf = np.zeros((_LANES, h, w), dtype=np.float32)
    buf[:n] = planes
    N = np.array([h, w], dtype=np.int32)
    loc = np.zeros(3 * _LANES, dtype=np.float32)
    std = np.zeros(3 * _LANES, dtype=np.float32)
    _LIB.lsqpeaklocate_lm_batch(np.ascontiguousarray(buf), L_real, N, fit_type, loc, std)
    return loc.reshape(3, _LANES), std.reshape(3, _LANES)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelector:
    """No-silent-fallback semantics of the runtime selector."""

    def test_availability_consistent(self):
        assert _LIB.bulkxcorr2d_peakfit_batch_available() == _LIB.peakfit_batch_available()

    def test_default_is_scalar(self):
        assert _LIB.bulkxcorr2d_get_peakfit_impl() == 0

    @needs_batch
    def test_select_and_restore(self):
        assert _LIB.bulkxcorr2d_set_peakfit_impl(1) == 0
        assert _LIB.bulkxcorr2d_get_peakfit_impl() == 1
        assert _LIB.bulkxcorr2d_set_peakfit_impl(0) == 0
        assert _LIB.bulkxcorr2d_get_peakfit_impl() == 0

    @pytest.mark.skipif(_AVAILABLE, reason="stub-only semantics")
    def test_stub_refuses_batch(self):
        assert _LIB.bulkxcorr2d_set_peakfit_impl(1) == -1
        assert _LIB.bulkxcorr2d_get_peakfit_impl() == 0

    @needs_batch
    def test_lanes_sane(self):
        assert _LANES in (4, 8, 16)


@needs_batch
class TestOracleAgreement:
    """A1: batch results agree with the scalar oracle on clean batteries."""

    @pytest.mark.parametrize("fit_type", [4, 5, 6])
    def test_clean_battery(self, fit_type):
        rng = np.random.RandomState(1000 + fit_type)
        n_cases = 8 * _LANES
        dmax = 0.0
        for _ in range(n_cases // _LANES):
            planes = np.stack([_gauss_plane(rng, fit_type) for _ in range(_LANES)])
            bloc, _ = _run_batch(planes, _LANES, fit_type)
            for l in range(_LANES):
                sloc, _ = _run_scalar(planes[l], fit_type)
                assert np.isnan(sloc[0]) == np.isnan(bloc[0, l])
                if not np.isnan(sloc[0]):
                    d = np.hypot(sloc[0] - bloc[0, l], sloc[1] - bloc[1, l])
                    dmax = max(dmax, float(d))
        assert dmax < 1e-3, f"type {fit_type}: max |dloc| {dmax:.2e} px vs oracle"

    @pytest.mark.parametrize("fit_type", [4, 5, 6])
    def test_noisy_battery(self, fit_type):
        """A2: noisy planes — kept positions agree; NaN classification may
        jitter ONLY in the trust-boundary band.

        The production batch fitter uses the vectorized polynomial exp
        (rel err ~8e-8). Its tiny value differences perturb LM iteration
        paths on planes whose residual sits near the trust rule's NaN
        boundary (residual ~ 1e-3*A^2*npix, i.e. noise sigma/A near 3.16%),
        flipping the NaN-vs-kept call on fits that are marginal either way.
        Measured (2026-07-06, 1536 fits): ZERO flips for sigma/A below 0.8x
        the boundary — good fits are never misclassified — and a few % on
        at/beyond-boundary junk. Under the libm-exp reference build
        (PIVTOOLS_PEAKFIT_LIBM_EXP=1) agreement is exact (0 flips).
        This test pins both facts: no flips in the safe band, bounded
        jitter on the deliberately-marginal remainder.
        """
        rng = np.random.RandomState(2000 + fit_type)
        sigma = 15.0
        boundary = np.sqrt(1e-3)              # trust rule: sigma/A at the NaN boundary
        n_batches = 16
        safe_flips = 0
        marginal_flips = 0
        marginal_total = 0
        p99_pool = []
        for _ in range(n_batches):
            amps = rng.uniform(100.0, 1000.0, _LANES)
            planes = []
            for l in range(_LANES):
                rng2 = np.random.RandomState(rng.randint(2**31))
                p = _gauss_plane(rng2, fit_type)
                p *= amps[l] / p.max()
                planes.append(p)
            planes = np.stack(planes) + rng.normal(0, sigma, (_LANES, PLANE_N, PLANE_N)).astype(np.float32)
            bloc, _ = _run_batch(planes, _LANES, fit_type)
            for l in range(_LANES):
                sloc, _ = _run_scalar(planes[l], fit_type)
                sn, bn = bool(np.isnan(sloc[0])), bool(np.isnan(bloc[0, l]))
                in_safe_band = (sigma / amps[l]) < 0.8 * boundary
                if sn != bn:
                    if in_safe_band:
                        safe_flips += 1
                    else:
                        marginal_flips += 1
                if not in_safe_band:
                    marginal_total += 1
                if not sn and not bn:
                    p99_pool.append(np.hypot(sloc[0] - bloc[0, l], sloc[1] - bloc[1, l]))
        assert safe_flips == 0, (
            f"type {fit_type}: {safe_flips} NaN flips on GOOD fits (safe band) — "
            "this must never happen; suspect a lockstep-logic regression "
            "(verify with a PIVTOOLS_PEAKFIT_LIBM_EXP=1 build)"
        )
        if marginal_total:
            rate = marginal_flips / marginal_total
            assert rate < 0.10, (
                f"type {fit_type}: boundary-band flip rate {rate:.1%} exceeds 10%"
            )
        if p99_pool:
            assert float(np.percentile(p99_pool, 99)) < 1e-3


@needs_batch
class TestFailureMasks:
    """A3: pathological planes produce identical NaN masks."""

    def _pathological(self, kind, rng):
        c = int(CENTER)
        if kind == "flat":
            return np.zeros((PLANE_N, PLANE_N), dtype=np.float32)
        if kind == "noise_spike":
            p = rng.uniform(0, 1, (PLANE_N, PLANE_N)).astype(np.float32)
            p[c, c] = 2.0
            return p
        if kind == "nan_window":
            p = _gauss_plane(rng, 4)
            p[c + 1, c + 1] = np.nan
            return p
        return _gauss_plane(rng, 4)  # "clean"

    @pytest.mark.parametrize("fit_type", [4, 5, 6])
    def test_masks_match(self, fit_type):
        rng = np.random.RandomState(3000 + fit_type)
        kinds = (["flat", "noise_spike", "nan_window", "clean"] * ((_LANES // 4) + 1))[:_LANES]
        planes = np.stack([self._pathological(k, rng) for k in kinds])
        bloc, bstd = _run_batch(planes, _LANES, fit_type)
        for l in range(_LANES):
            sloc, _ = _run_scalar(planes[l], fit_type)
            assert np.isnan(sloc[0]) == np.isnan(bloc[0, l]), (
                f"lane {l} ({kinds[l]}): scalar "
                f"{'NaN' if np.isnan(sloc[0]) else 'finite'} vs batch "
                f"{'NaN' if np.isnan(bloc[0, l]) else 'finite'}"
            )
            if np.isnan(bloc[0, l]):
                assert bloc[2, l] == 0 and np.all(bstd[:, l] == 0)


@needs_batch
class TestPartialBatch:
    """Tail semantics: L_real < lanes leaves real lanes bit-identical and
    tail lanes NaN."""

    def test_partial(self):
        rng = np.random.RandomState(4000)
        planes = np.stack([_gauss_plane(rng, 6) for _ in range(_LANES)])
        full_loc, _ = _run_batch(planes, _LANES, 6)
        L = max(1, _LANES - 2)
        part_loc, _ = _run_batch(planes, L, 6)
        for l in range(L):
            for comp in range(3):
                a, b = full_loc[comp, l], part_loc[comp, l]
                assert (a == b) or (np.isnan(a) and np.isnan(b))
        for l in range(L, _LANES):
            assert np.isnan(part_loc[0, l])
