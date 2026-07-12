"""
Correctness tests for the fused symmetric warp C kernel (libfusedwarp).

Tests both interpolation modes:
  - mode 0: bicubic (Keys a=-0.75, 4x4) — compared against cv2.INTER_CUBIC
  - mode 1: Lanczos-3 (windowed sinc, 6x6) — compared against numpy reference

Test classes:
  - TestSyntheticCorrectness: grid-pattern images + Poiseuille predictor (in-memory)

Pass criteria (relative to image value range):
  - mean_err  < 0.1% of range  (catches systematic bugs)
  - p99.9_err < 5%  of range   (allows bicubic ringing at sharp edges)
"""

import ctypes
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Platform-correct shared-library extension (was hard-coded .dll → Windows-only).
_LIB_EXT = ".dll" if sys.platform.startswith("win") else ".so"
_LIB_PATH = _PROJECT_ROOT / "pivtools_cli" / "lib" / f"libfusedwarp{_LIB_EXT}"

INTERP_NAMES = {0: "bicubic", 1: "lanczos3"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fused_warp_lib():
    """Load libfusedwarp (platform shared lib) via ctypes. Skip if not found."""
    if not _LIB_PATH.is_file():
        pytest.skip(f"libfusedwarp{_LIB_EXT} not found at {_LIB_PATH}")

    lib = ctypes.CDLL(str(_LIB_PATH))

    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int = ctypes.c_int

    lib.fused_symmetric_warp.argtypes = [
        c_float_p,
        c_float_p,  # img_a, img_b
        c_float_p,
        c_float_p,  # out_a, out_b
        c_float_p,
        c_float_p,  # pred_dy, pred_dx
        c_int,
        c_int,  # H, W
        c_int,
        c_int,  # nPY, nPX
        c_float_p,
        c_float_p,  # ctrs_y, ctrs_x
        c_int,  # interp_mode
        c_int,  # round_shifts
    ]
    lib.fused_symmetric_warp.restype = c_int

    lib.fused_symmetric_warp_batch.argtypes = [
        c_float_p,
        c_float_p,  # imgs_a, imgs_b   (N, H, W)
        c_float_p,
        c_float_p,  # outs_a, outs_b   (N, H, W)
        c_float_p,
        c_float_p,  # pred_dy, pred_dx (nPY,nPX) or (N,nPY,nPX)
        c_int,
        c_int,
        c_int,  # N, H, W
        c_int,
        c_int,  # nPY, nPX
        c_float_p,
        c_float_p,  # ctrs_y, ctrs_x
        c_int,
        c_int,  # interp_mode, shared_predictor
        c_int,  # round_shifts
    ]
    lib.fused_symmetric_warp_batch.restype = c_int

    # Runtime impl selector (0 = scalar reference, 1 = interior/SIMD path).
    lib.fused_warp_set_impl.argtypes = [c_int]
    lib.fused_warp_set_impl.restype = None
    lib.fused_warp_get_impl.argtypes = []
    lib.fused_warp_get_impl.restype = c_int

    return lib


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def call_kernel(
    lib, img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x, interp_mode=0, round_shifts=0
):
    """Call the C kernel and return (out_a, out_b)."""
    H, W = img_a.shape
    nPY, nPX = pred_dy.shape

    img_a = np.ascontiguousarray(img_a, dtype=np.float32)
    img_b = np.ascontiguousarray(img_b, dtype=np.float32)
    pred_dy = np.ascontiguousarray(pred_dy, dtype=np.float32)
    pred_dx = np.ascontiguousarray(pred_dx, dtype=np.float32)
    ctrs_y = np.ascontiguousarray(ctrs_y, dtype=np.float32)
    ctrs_x = np.ascontiguousarray(ctrs_x, dtype=np.float32)

    out_a = np.zeros((H, W), dtype=np.float32)
    out_b = np.zeros((H, W), dtype=np.float32)

    c_float_p = ctypes.POINTER(ctypes.c_float)
    ret = lib.fused_symmetric_warp(
        img_a.ctypes.data_as(c_float_p),
        img_b.ctypes.data_as(c_float_p),
        out_a.ctypes.data_as(c_float_p),
        out_b.ctypes.data_as(c_float_p),
        pred_dy.ctypes.data_as(c_float_p),
        pred_dx.ctypes.data_as(c_float_p),
        ctypes.c_int(H),
        ctypes.c_int(W),
        ctypes.c_int(nPY),
        ctypes.c_int(nPX),
        ctrs_y.ctypes.data_as(c_float_p),
        ctrs_x.ctypes.data_as(c_float_p),
        ctypes.c_int(interp_mode),
        ctypes.c_int(round_shifts),
    )
    if ret != 0:
        raise RuntimeError(f"C kernel returned error code {ret}")

    return out_a, out_b


def call_kernel_batch(
    lib,
    imgs_a,
    imgs_b,
    pred_dy,
    pred_dx,
    ctrs_y,
    ctrs_x,
    interp_mode=0,
    shared_predictor=1,
    round_shifts=0,
):
    """Call the batch C kernel and return (outs_a, outs_b), each (N, H, W).

    imgs_* are (N, H, W). pred_* are (nPY, nPX) when shared_predictor=1, else
    (N, nPY, nPX).
    """
    N, H, W = imgs_a.shape
    nPY, nPX = pred_dy.shape if shared_predictor else pred_dy.shape[1:]

    imgs_a = np.ascontiguousarray(imgs_a, dtype=np.float32)
    imgs_b = np.ascontiguousarray(imgs_b, dtype=np.float32)
    pred_dy = np.ascontiguousarray(pred_dy, dtype=np.float32)
    pred_dx = np.ascontiguousarray(pred_dx, dtype=np.float32)
    ctrs_y = np.ascontiguousarray(ctrs_y, dtype=np.float32)
    ctrs_x = np.ascontiguousarray(ctrs_x, dtype=np.float32)

    outs_a = np.zeros((N, H, W), dtype=np.float32)
    outs_b = np.zeros((N, H, W), dtype=np.float32)

    c_float_p = ctypes.POINTER(ctypes.c_float)
    ret = lib.fused_symmetric_warp_batch(
        imgs_a.ctypes.data_as(c_float_p),
        imgs_b.ctypes.data_as(c_float_p),
        outs_a.ctypes.data_as(c_float_p),
        outs_b.ctypes.data_as(c_float_p),
        pred_dy.ctypes.data_as(c_float_p),
        pred_dx.ctypes.data_as(c_float_p),
        ctypes.c_int(N),
        ctypes.c_int(H),
        ctypes.c_int(W),
        ctypes.c_int(nPY),
        ctypes.c_int(nPX),
        ctrs_y.ctypes.data_as(c_float_p),
        ctrs_x.ctypes.data_as(c_float_p),
        ctypes.c_int(interp_mode),
        ctypes.c_int(shared_predictor),
        ctypes.c_int(round_shifts),
    )
    if ret != 0:
        raise RuntimeError(f"C batch kernel returned error code {ret}")

    return outs_a, outs_b


def _lanczos3_warp_numpy(img, map_y, map_x):
    """Vectorized Lanczos-3 image warp with BORDER_CONSTANT=0.

    Loops over the 6x6 stencil (36 iterations) but each iteration is
    fully vectorized over all output pixels.
    """
    H, W = img.shape

    def _lanczos_kernel(t):
        at = np.abs(t)
        result = np.zeros_like(at)
        small = at < 1e-6
        result[small] = 1.0
        mid = (~small) & (at < 3.0)
        pi_t = np.pi * at[mid]
        pi_t_a = pi_t / 3.0
        result[mid] = (np.sin(pi_t) / pi_t) * (np.sin(pi_t_a) / pi_t_a)
        return result

    fy_floor = np.floor(map_y)
    fx_floor = np.floor(map_x)
    dy = map_y - fy_floor
    dx = map_x - fx_floor
    iy_base = fy_floor.astype(np.int32) - 2
    ix_base = fx_floor.astype(np.int32) - 2

    out = np.zeros((H, W), dtype=np.float64)
    for m in range(6):
        wy = _lanczos_kernel(dy - (m - 2))
        rows = iy_base + m
        row_valid = (rows >= 0) & (rows < H)
        for n in range(6):
            wx = _lanczos_kernel(dx - (n - 2))
            cols = ix_base + n
            col_valid = (cols >= 0) & (cols < W)
            valid = row_valid & col_valid
            r = np.clip(rows, 0, H - 1)
            c = np.clip(cols, 0, W - 1)
            vals = img[r, c].astype(np.float64)
            vals = np.where(valid, vals, 0.0)
            out += wy * wx * vals
    return out.astype(np.float32)


def reference_warp(img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x, interp_mode=0):
    """Reference warp matching the C kernel's behaviour.

    Predictor upsampling: ALWAYS bicubic (cv2.INTER_CUBIC + BORDER_REPLICATE).
    Image warping: mode 0 -> cv2 bicubic, mode 1 -> numpy Lanczos-3.

    Convention:
        map_A = im_mesh - delta/2   (sample backward from A)
        map_B = im_mesh + delta/2   (sample forward  from B)
    """
    H, W = img_a.shape
    nPY = len(ctrs_y)
    nPX = len(ctrs_x)

    # Step 1: Upsample predictor to dense (H, W) — ALWAYS bicubic
    pix_y = np.arange(H, dtype=np.float32)
    pix_x = np.arange(W, dtype=np.float32)
    pred_frac_y = np.interp(pix_y, ctrs_y, np.arange(nPY, dtype=np.float32))
    pred_frac_x = np.interp(pix_x, ctrs_x, np.arange(nPX, dtype=np.float32))
    pred_frac_x = pred_frac_x.astype(np.float32)
    pred_frac_y = pred_frac_y.astype(np.float32)
    map_y_pred, map_x_pred = np.meshgrid(pred_frac_y, pred_frac_x, indexing="ij")

    dense_dy = cv2.remap(
        pred_dy.astype(np.float32),
        map_x_pred,
        map_y_pred,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    dense_dx = cv2.remap(
        pred_dx.astype(np.float32),
        map_x_pred,
        map_y_pred,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Step 2: Build symmetric coordinate maps
    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    half_dy = dense_dy / 2.0
    half_dx = dense_dx / 2.0
    map_a_y = yy - half_dy
    map_a_x = xx - half_dx
    map_b_y = yy + half_dy
    map_b_x = xx + half_dx

    # Step 3: Warp images (mode-dependent)
    if interp_mode == 0:
        out_a = cv2.remap(
            img_a.astype(np.float32),
            map_a_x,
            map_a_y,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out_b = cv2.remap(
            img_b.astype(np.float32),
            map_b_x,
            map_b_y,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        out_a = _lanczos3_warp_numpy(img_a.astype(np.float32), map_a_y, map_a_x)
        out_b = _lanczos3_warp_numpy(img_b.astype(np.float32), map_b_y, map_b_x)

    return out_a, out_b


def make_grid_image(H, W, spacing=16):
    """Create a grid-pattern test image with smooth variation."""
    img = np.zeros((H, W), dtype=np.float32)
    for y in range(0, H, spacing):
        img[y, :] = 255.0
    for x in range(0, W, spacing):
        img[:, x] = 255.0
    yy, xx = np.meshgrid(
        np.linspace(0, 2 * np.pi, H),
        np.linspace(0, 2 * np.pi, W),
        indexing="ij",
    )
    img += 50.0 * np.sin(yy) * np.cos(xx)
    return img.astype(np.float32)


def make_poiseuille_predictor(nPY, nPX, ctrs_y, ctrs_x, u_max=5.0, H=None):
    """Poiseuille (parabolic) displacement field on the predictor grid.

    u_x(y) = u_max * (1 - (2*y/H - 1)^2), u_y = 0
    """
    if H is None:
        H = int(ctrs_y[-1] + ctrs_y[0])
    y_norm = 2.0 * ctrs_y / float(H) - 1.0
    u_profile = u_max * (1.0 - y_norm**2)
    pred_dx = np.tile(u_profile[:, None], (1, nPX)).astype(np.float32)
    pred_dy = np.zeros((nPY, nPX), dtype=np.float32)
    return pred_dy, pred_dx


def make_window_centres(H, W, nPY, nPX):
    """Create evenly spaced window centres."""
    spacing_y = H / nPY
    spacing_x = W / nPX
    ctrs_y = np.arange(nPY, dtype=np.float32) * spacing_y + spacing_y / 2.0
    ctrs_x = np.arange(nPX, dtype=np.float32) * spacing_x + spacing_x / 2.0
    return ctrs_y, ctrs_x


def evaluate_errors(err_a, err_b, img_range):
    """Compute error stats and pass/fail against relative thresholds.

    Pass criteria (relative to image value range):
      - mean_err  < 0.1% of range   (catches systematic bugs)
      - p99.9_err < 5%  of range    (allows bicubic ringing at sharp edges)
    """
    all_err = np.concatenate([err_a.ravel(), err_b.ravel()])
    max_err = float(all_err.max())
    mean_err = float(all_err.mean())
    p999_err = float(np.percentile(all_err, 99.9))
    passed = (mean_err < img_range * 0.001) and (p999_err < img_range * 0.05)
    return max_err, mean_err, p999_err, passed


# ---------------------------------------------------------------------------
# Test class: synthetic grid images + Poiseuille predictor
# ---------------------------------------------------------------------------
class TestSyntheticCorrectness:
    """C kernel vs reference on synthetic grid images with Poiseuille predictor."""

    @pytest.mark.parametrize("interp_mode", [0, 1], ids=["bicubic", "lanczos3"])
    @pytest.mark.parametrize(
        "H, W, nPY, nPX",
        [
            (512, 512, 8, 8),
            (1000, 1000, 16, 16),
        ],
        ids=["512x512_8x8", "1000x1000_16x16"],
    )
    def test_correctness(self, fused_warp_lib, interp_mode, H, W, nPY, nPX):
        ctrs_y, ctrs_x = make_window_centres(H, W, nPY, nPX)
        pred_dy, pred_dx = make_poiseuille_predictor(
            nPY,
            nPX,
            ctrs_y,
            ctrs_x,
            u_max=5.0,
            H=H,
        )
        img_a = make_grid_image(H, W)
        img_b = make_grid_image(H, W, spacing=20)

        c_out_a, c_out_b = call_kernel(
            fused_warp_lib,
            img_a,
            img_b,
            pred_dy,
            pred_dx,
            ctrs_y,
            ctrs_x,
            interp_mode=interp_mode,
        )
        ref_out_a, ref_out_b = reference_warp(
            img_a,
            img_b,
            pred_dy,
            pred_dx,
            ctrs_y,
            ctrs_x,
            interp_mode=interp_mode,
        )

        err_a = np.abs(c_out_a - ref_out_a)
        err_b = np.abs(c_out_b - ref_out_b)
        img_range = max(img_a.max() - img_a.min(), 1.0)
        max_err, mean_err, p999_err, passed = evaluate_errors(
            err_a,
            err_b,
            img_range,
        )

        mode_name = INTERP_NAMES[interp_mode]
        assert passed, (
            f"{H}x{W} grid={nPY}x{nPX} [{mode_name}]: "
            f"max={max_err:.2f}  mean={mean_err:.4f}  p99.9={p999_err:.2f}"
        )


# ---------------------------------------------------------------------------
# Scalar-vs-SIMD equivalence
# ---------------------------------------------------------------------------
class TestScalarSimdEquivalence:
    """The optimised path (impl=1: interior/border split, later SIMD) must match the
    scalar reference (impl=0: always bounds-checked) to within FP-reassociation noise.

    Includes a high-frequency 0/255 checkerboard — the worst case for scalar-vs-SIMD
    divergence, since FMA's single rounding vs the reference's double rounding diverges
    most on high-frequency content. The checkerboard is built by integer modulus (NOT a
    sin/cos wave): trig carries platform ULP jitter that FMA contraction would amplify
    into false-positive failures.
    """

    @pytest.mark.parametrize("interp_mode", [0, 1], ids=["bicubic", "lanczos3"])
    @pytest.mark.parametrize("pattern", ["grid", "checkerboard"])
    def test_equivalence(self, fused_warp_lib, interp_mode, pattern):
        H = W = 256
        nPY = nPX = 16
        ctrs_y, ctrs_x = make_window_centres(H, W, nPY, nPX)
        pred_dy, pred_dx = make_poiseuille_predictor(
            nPY, nPX, ctrs_y, ctrs_x, u_max=5.0, H=H
        )

        if pattern == "grid":
            img_a = make_grid_image(H, W, spacing=16)
            img_b = make_grid_image(H, W, spacing=20)
        else:
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            img_a = (((yy + xx) % 2) * 255.0).astype(np.float32)
            img_b = (((yy + xx + 1) % 2) * 255.0).astype(np.float32)

        try:
            fused_warp_lib.fused_warp_set_impl(0)
            ref_a, ref_b = call_kernel(
                fused_warp_lib,
                img_a,
                img_b,
                pred_dy,
                pred_dx,
                ctrs_y,
                ctrs_x,
                interp_mode,
            )
            fused_warp_lib.fused_warp_set_impl(1)
            opt_a, opt_b = call_kernel(
                fused_warp_lib,
                img_a,
                img_b,
                pred_dy,
                pred_dx,
                ctrs_y,
                ctrs_x,
                interp_mode,
            )
        finally:
            fused_warp_lib.fused_warp_set_impl(1)  # restore default

        max_d = float(max(np.max(np.abs(opt_a - ref_a)), np.max(np.abs(opt_b - ref_b))))
        # 1e-3 on 0..255 data (~4e-6 relative) absorbs FMA/reassociation; today's
        # interior split is exact (0.0), the headroom is for the SIMD path.
        assert (
            max_d < 1e-3
        ), f"{pattern}/{INTERP_NAMES[interp_mode]}: scalar-vs-opt max|Δ|={max_d:.3e}"


# ---------------------------------------------------------------------------
# Batch entry point vs single-pair (striping / dispatch coverage)
# ---------------------------------------------------------------------------
class TestBatchMatchesSinglePair:
    """The batch entry must reproduce, for each slot k, exactly what the single-pair
    entry produces for that slot's inputs.

    This guards the batch-only machinery the single-pair path never runs: the
    flattened (image, row) loop (ti -> ni, i) and the per-image pointer arithmetic
    (imgs + ni*H*W, pred + ni*pred_stride), across BOTH shared (ensemble) and
    per-image (instantaneous) predictor striping. A wrong stride or a swapped slot
    shows up as a mismatch because every image and every per-image predictor is
    distinct.

    Note this is deliberately NOT a scalar-vs-SIMD (impl 0 vs 1) test: the impl flag
    only gates the interior sampler, while the batch striping is identical for both
    impls — so an impl 0/1 batch comparison would pass even with a stride bug. The
    striping is only caught by comparing against the trusted single-pair entry.

    Results are bit-identical (max|Δ| == 0): both entries run the same per-pixel
    kernels on the same data, and OpenMP row-scheduling does not reassociate any
    per-pixel sum. Run under the default impl (SIMD), which also confirms the SIMD
    path is correctly plumbed through the batch call sites.
    """

    @pytest.mark.parametrize("interp_mode", [0, 1], ids=["bicubic", "lanczos3"])
    @pytest.mark.parametrize(
        "shared_predictor", [1, 0], ids=["shared_pred", "per_image_pred"]
    )
    def test_batch_matches_single(self, fused_warp_lib, interp_mode, shared_predictor):
        N = 3
        H = W = 256
        nPY = nPX = 16
        ctrs_y, ctrs_x = make_window_centres(H, W, nPY, nPX)

        # Distinct random content per slot so a swapped slot is detectable.
        rng = np.random.default_rng(0)
        imgs_a = (rng.random((N, H, W), dtype=np.float32) * 255.0).astype(np.float32)
        imgs_b = (rng.random((N, H, W), dtype=np.float32) * 255.0).astype(np.float32)

        # Distinct predictor per slot (varying u_max) so a per-image stride bug shows.
        preds = [
            make_poiseuille_predictor(
                nPY, nPX, ctrs_y, ctrs_x, u_max=3.0 + 2.0 * k, H=H
            )
            for k in range(N)
        ]
        if shared_predictor:
            pred_dy_b, pred_dx_b = preds[0]
        else:
            pred_dy_b = np.stack([p[0] for p in preds]).astype(np.float32)
            pred_dx_b = np.stack([p[1] for p in preds]).astype(np.float32)

        outs_a, outs_b = call_kernel_batch(
            fused_warp_lib,
            imgs_a,
            imgs_b,
            pred_dy_b,
            pred_dx_b,
            ctrs_y,
            ctrs_x,
            interp_mode,
            shared_predictor,
        )

        for k in range(N):
            pdy, pdx = preds[0] if shared_predictor else preds[k]
            ref_a, ref_b = call_kernel(
                fused_warp_lib,
                imgs_a[k],
                imgs_b[k],
                pdy,
                pdx,
                ctrs_y,
                ctrs_x,
                interp_mode,
            )
            d = float(
                max(
                    np.max(np.abs(outs_a[k] - ref_a)), np.max(np.abs(outs_b[k] - ref_b))
                )
            )
            pred_kind = "shared" if shared_predictor else "per-image"
            assert d == 0.0, (
                f"batch[{pred_kind}]/{INTERP_NAMES[interp_mode]} slot {k}: "
                f"batch vs single-pair max|Δ|={d:.3e}"
            )
