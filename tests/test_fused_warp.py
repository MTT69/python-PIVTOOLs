"""
Correctness + performance tests for the fused symmetric warp C kernel.

Tests both interp modes:
  - mode 0: bicubic (Keys a=-0.75, 4x4) — compared against cv2.INTER_CUBIC
  - mode 1: Lanczos-3 (windowed sinc, 6x6) — compared against numpy reference

Tests:
  A) Synthetic correctness — grid-pattern images, Poiseuille displacement field
  B) Real data correctness + timing — uses profiler image presets
  C) Synthetic timing benchmark — 1 MP, 4 MP, 25 MP (bicubic vs lanczos)

Usage:
    python manual_tools/test_fused_warp.py                     # All tests
    python manual_tools/test_fused_warp.py --synthetic-only     # Skip real data
    python manual_tools/test_fused_warp.py --threads 8          # Set OMP thread count
"""

import argparse
import ctypes
import os
import sys
import time

import cv2
import numpy as np

# Ensure project root is on the path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Image presets (same as profile/profile_piv.py)
# ---------------------------------------------------------------------------
IMAGE_PRESETS = {
    "4mp": {
        "path": (
            r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
            r"\#current_processing\4000_images_channel\planar_images"
        ),
        "shape": [2048, 2048],
        "label": "4 MP (2048x2048)",
    },
    "25mp": {
        "path": (
            r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
            r"\#current_processing\query_JHTDB\download_from_jhtdb\efe_images"
        ),
        "shape": [4600, 5312],
        "label": "25 MP (4600x5312)",
    },
}


# ---------------------------------------------------------------------------
# Load the C kernel
# ---------------------------------------------------------------------------
def load_kernel():
    """Load libfusedwarp.dll and set up the ctypes signature."""
    lib_dir = os.path.join(_project_root, "pivtools_cli", "lib")
    dll_path = os.path.join(lib_dir, "libfusedwarp.dll")
    if not os.path.isfile(dll_path):
        raise FileNotFoundError(
            f"DLL not found: {dll_path}\n"
            "Run: python manual_tools/build_fused_warp.py"
        )
    lib = ctypes.CDLL(dll_path)

    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int = ctypes.c_int

    lib.fused_symmetric_warp.argtypes = [
        c_float_p, c_float_p,    # img_a, img_b
        c_float_p, c_float_p,    # out_a, out_b
        c_float_p, c_float_p,    # pred_dy, pred_dx
        c_int, c_int,            # H, W
        c_int, c_int,            # nPY, nPX
        c_float_p, c_float_p,    # ctrs_y, ctrs_x
        c_int,                   # interp_mode
    ]
    lib.fused_symmetric_warp.restype = c_int

    return lib


def call_kernel(lib, img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x, interp_mode=0):
    """Call the C kernel and return (out_a, out_b)."""
    H, W = img_a.shape
    nPY, nPX = pred_dy.shape

    # Ensure contiguous float32
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
        ctypes.c_int(H), ctypes.c_int(W),
        ctypes.c_int(nPY), ctypes.c_int(nPX),
        ctrs_y.ctypes.data_as(c_float_p),
        ctrs_x.ctypes.data_as(c_float_p),
        ctypes.c_int(interp_mode),
    )
    if ret != 0:
        raise RuntimeError(f"C kernel returned error code {ret}")

    return out_a, out_b


# ---------------------------------------------------------------------------
# Lanczos-3 vectorized numpy reference (no cv2 equivalent — cv2 only has
# Lanczos-4 with 8×8 stencil, ours is Lanczos-3 with 6×6)
# ---------------------------------------------------------------------------
def _lanczos3_warp_numpy(img, map_y, map_x):
    """Vectorized Lanczos-3 image warp with BORDER_CONSTANT=0.

    Loops over the 6×6 stencil (36 iterations) but each iteration is
    fully vectorized over all output pixels.  Fast enough for test images.
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


# ---------------------------------------------------------------------------
# Reference implementation (matches production sign convention)
# ---------------------------------------------------------------------------
INTERP_NAMES = {0: "bicubic", 1: "lanczos3"}


def reference_warp(img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x, interp_mode=0):
    """
    Reference warp matching the C kernel's behaviour:
      - Predictor upsampling: ALWAYS bicubic (cv2.INTER_CUBIC + BORDER_REPLICATE)
      - Image warping: mode 0 → cv2 bicubic, mode 1 → numpy Lanczos-3

    Production convention:
        map_A = im_mesh - delta/2   (sample backward from A)
        map_B = im_mesh + delta/2   (sample forward from B)
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
        pred_dy.astype(np.float32), map_x_pred, map_y_pred,
        cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    dense_dx = cv2.remap(
        pred_dx.astype(np.float32), map_x_pred, map_y_pred,
        cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
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
        # Bicubic — cv2 reference
        out_a = cv2.remap(
            img_a.astype(np.float32), map_a_x, map_a_y,
            cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        out_b = cv2.remap(
            img_b.astype(np.float32), map_b_x, map_b_y,
            cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    else:
        # Lanczos-3 — numpy reference (no cv2 equivalent)
        out_a = _lanczos3_warp_numpy(img_a.astype(np.float32), map_a_y, map_a_x)
        out_b = _lanczos3_warp_numpy(img_b.astype(np.float32), map_b_y, map_b_x)

    return out_a, out_b


# ---------------------------------------------------------------------------
# Test data generators
# ---------------------------------------------------------------------------
def make_grid_image(H, W, spacing=16):
    """Create a grid-pattern test image."""
    img = np.zeros((H, W), dtype=np.float32)
    # Horizontal lines
    for y in range(0, H, spacing):
        img[y, :] = 255.0
    # Vertical lines
    for x in range(0, W, spacing):
        img[:, x] = 255.0
    # Add some smooth variation
    yy, xx = np.meshgrid(
        np.linspace(0, 2 * np.pi, H),
        np.linspace(0, 2 * np.pi, W),
        indexing="ij",
    )
    img += 50.0 * np.sin(yy) * np.cos(xx)
    return img.astype(np.float32)


def make_poiseuille_predictor(nPY, nPX, ctrs_y, ctrs_x, u_max=5.0, H=None):
    """
    Poiseuille (parabolic) displacement field on the predictor grid.
    u_x(y) = u_max * (1 - (2*y/H - 1)^2), u_y = 0
    """
    if H is None:
        H = int(ctrs_y[-1] + ctrs_y[0])  # rough estimate
    # Normalised y positions [-1, 1]
    y_norm = 2.0 * ctrs_y / float(H) - 1.0
    u_profile = u_max * (1.0 - y_norm ** 2)
    # Broadcast to (nPY, nPX)
    pred_dx = np.tile(u_profile[:, None], (1, nPX)).astype(np.float32)
    pred_dy = np.zeros((nPY, nPX), dtype=np.float32)
    return pred_dy, pred_dx


def make_window_centres(H, W, nPY, nPX):
    """Create evenly spaced window centres."""
    # Centres are spaced within the image, with half-spacing margin
    spacing_y = H / nPY
    spacing_x = W / nPX
    ctrs_y = np.arange(nPY, dtype=np.float32) * spacing_y + spacing_y / 2.0
    ctrs_x = np.arange(nPX, dtype=np.float32) * spacing_x + spacing_x / 2.0
    return ctrs_y, ctrs_x


def load_image_pair(source_dir, pair_idx=1):
    """Load one AB-format image pair, return (img_a, img_b) as float32."""
    a_path = os.path.join(source_dir, f"B{pair_idx:05d}_A.tif")
    b_path = os.path.join(source_dir, f"B{pair_idx:05d}_B.tif")
    if not os.path.isfile(a_path):
        raise FileNotFoundError(f"Image A not found: {a_path}")
    if not os.path.isfile(b_path):
        raise FileNotFoundError(f"Image B not found: {b_path}")
    img_a = cv2.imread(a_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img_b = cv2.imread(b_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return img_a, img_b


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------
def best_of_n(func, n=3):
    """Run func n times, return (best_time, result_from_first_run)."""
    result = None
    best = float("inf")
    for k in range(n):
        t0 = time.perf_counter()
        r = func()
        elapsed = time.perf_counter() - t0
        if k == 0:
            result = r
        best = min(best, elapsed)
    return best, result


# ---------------------------------------------------------------------------
# Error reporting + pass/fail
# ---------------------------------------------------------------------------
def evaluate_errors(err_a, err_b, img_range):
    """Compute error stats and pass/fail against relative thresholds.

    Pass criteria (relative to image value range):
      - mean_err  < 0.1% of range   (catches systematic bugs)
      - p99.9_err < 5%  of range    (allows bicubic ringing at sharp edges)
    These are practical thresholds: bicubic implementations differ at sharp
    edges and image borders, but the differences are irrelevant to PIV
    (the correlation peak location is unaffected by sub-percent pixel errors).
    """
    all_err = np.concatenate([err_a.ravel(), err_b.ravel()])
    max_err = float(all_err.max())
    mean_err = float(all_err.mean())
    p999_err = float(np.percentile(all_err, 99.9))

    # Relative thresholds
    passed = (mean_err < img_range * 0.001) and (p999_err < img_range * 0.05)
    return max_err, mean_err, p999_err, passed


def format_error_line(label, max_err, mean_err, p999_err, passed):
    status = "PASS" if passed else "FAIL"
    return (
        f"  {label}: max={max_err:.2f}  mean={mean_err:.4f}  "
        f"p99.9={p999_err:.2f}  [{status}]"
    )


# ---------------------------------------------------------------------------
# Test A: Synthetic correctness
# ---------------------------------------------------------------------------
def test_synthetic_correctness(lib):
    """Test C kernel vs reference on synthetic data (both interp modes)."""
    print("=" * 60)
    print("TEST A: Synthetic Correctness")
    print("=" * 60)

    configs = [
        (512, 512, 8, 8),
        (1000, 1000, 16, 16),
        (2048, 2048, 32, 32),
    ]

    all_pass = True
    for mode in (0, 1):
        mode_name = INTERP_NAMES[mode]
        print(f"\n  --- interp_mode={mode} ({mode_name}) ---")
        for H, W, nPY, nPX in configs:
            ctrs_y, ctrs_x = make_window_centres(H, W, nPY, nPX)
            pred_dy, pred_dx = make_poiseuille_predictor(nPY, nPX, ctrs_y, ctrs_x, u_max=5.0, H=H)
            img_a = make_grid_image(H, W)
            img_b = make_grid_image(H, W, spacing=20)

            # C kernel
            c_out_a, c_out_b = call_kernel(
                lib, img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x, interp_mode=mode,
            )

            # Reference
            ref_out_a, ref_out_b = reference_warp(
                img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x, interp_mode=mode,
            )

            # Compare
            err_a = np.abs(c_out_a - ref_out_a)
            err_b = np.abs(c_out_b - ref_out_b)
            img_range = max(img_a.max() - img_a.min(), 1.0)
            max_err, mean_err, p999_err, passed = evaluate_errors(err_a, err_b, img_range)
            if not passed:
                all_pass = False

            print(format_error_line(f"{H}x{W} grid={nPY}x{nPX}", max_err, mean_err, p999_err, passed))

    print()
    return all_pass


# ---------------------------------------------------------------------------
# Test B: Real data correctness + timing
# ---------------------------------------------------------------------------
def test_real_data(lib, n_timing=3):
    """Test C kernel vs reference on real images (both interp modes)."""
    print("=" * 60)
    print("TEST B: Real Data Correctness + Timing")
    print("=" * 60)

    any_tested = False
    all_pass = True

    for name, preset in IMAGE_PRESETS.items():
        source_dir = preset["path"]
        if not os.path.isdir(source_dir):
            print(f"  {preset['label']}: SKIPPED (path not found)")
            continue

        try:
            img_a, img_b = load_image_pair(source_dir)
        except FileNotFoundError as e:
            print(f"  {preset['label']}: SKIPPED ({e})")
            continue

        any_tested = True
        H, W = img_a.shape[:2]
        if H * W < 5_000_000:
            nPY, nPX = 32, 32
        else:
            nPY, nPX = 80, 80

        ctrs_y, ctrs_x = make_window_centres(H, W, nPY, nPX)
        pred_dy, pred_dx = make_poiseuille_predictor(nPY, nPX, ctrs_y, ctrs_x, u_max=3.0, H=H)

        for mode in (0, 1):
            mode_name = INTERP_NAMES[mode]

            # Correctness
            c_out_a, c_out_b = call_kernel(
                lib, img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x, interp_mode=mode,
            )
            ref_out_a, ref_out_b = reference_warp(
                img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x, interp_mode=mode,
            )

            err_a = np.abs(c_out_a - ref_out_a)
            err_b = np.abs(c_out_b - ref_out_b)
            img_range = max(img_a.max() - img_a.min(), 1.0)
            max_err, mean_err, p999_err, passed = evaluate_errors(err_a, err_b, img_range)
            if not passed:
                all_pass = False

            # Timing — C kernel
            _mode = mode  # capture for lambda
            t_c, _ = best_of_n(
                lambda: call_kernel(
                    lib, img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x, interp_mode=_mode,
                ),
                n=n_timing,
            )

            status = "PASS" if passed else "FAIL"
            print(f"  {preset['label']} [{mode_name}]:")
            print(f"    Correctness: max={max_err:.2f}  mean={mean_err:.4f}  p99.9={p999_err:.2f}  [{status}]")
            print(f"    C timing (best of {n_timing}): {t_c*1000:.1f}ms")

    if not any_tested:
        print("  No real data found — all presets skipped.")

    print()
    return all_pass


# ---------------------------------------------------------------------------
# Test C: Synthetic timing benchmark
# ---------------------------------------------------------------------------
def test_synthetic_timing(lib, n_timing=3):
    """Benchmark both interp modes on synthetic images at 1 MP, 4 MP, 25 MP."""
    print("=" * 60)
    print("TEST C: Synthetic Timing Benchmark")
    print("=" * 60)

    configs = [
        ("1 MP",   1024,  1024,  16, 16),
        ("4 MP",   2048,  2048,  32, 32),
        ("25 MP",  4600,  5312,  80, 80),
    ]

    print(f"  {'Size':<10} {'Bicubic (ms)':>14} {'Lanczos3 (ms)':>14} {'Ratio L/B':>10}")
    print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*10}")

    for label, H, W, nPY, nPX in configs:
        img_a = np.random.rand(H, W).astype(np.float32) * 255.0
        img_b = np.random.rand(H, W).astype(np.float32) * 255.0

        ctrs_y, ctrs_x = make_window_centres(H, W, nPY, nPX)
        pred_dy, pred_dx = make_poiseuille_predictor(nPY, nPX, ctrs_y, ctrs_x, u_max=3.0, H=H)

        t_bicubic, _ = best_of_n(
            lambda: call_kernel(lib, img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x, interp_mode=0),
            n=n_timing,
        )
        t_lanczos, _ = best_of_n(
            lambda: call_kernel(lib, img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x, interp_mode=1),
            n=n_timing,
        )
        ratio = t_lanczos / t_bicubic if t_bicubic > 0 else float("inf")

        print(f"  {label:<10} {t_bicubic*1000:>14.1f} {t_lanczos*1000:>14.1f} {ratio:>9.2f}x")

    print()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def generate_figures(lib):
    """Generate comparison figures: original, warped (C vs cv2), difference maps."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    # --- Decide which dataset to use: prefer real 4MP, fall back to synthetic ---
    use_real = False
    for name in ("4mp", "25mp"):
        preset = IMAGE_PRESETS[name]
        if os.path.isdir(preset["path"]):
            try:
                img_a, img_b = load_image_pair(preset["path"])
                use_real = True
                data_label = preset["label"]
                H, W = img_a.shape[:2]
                nPY, nPX = (32, 32) if H * W < 5_000_000 else (80, 80)
                break
            except FileNotFoundError:
                pass

    if not use_real:
        H, W, nPY, nPX = 1024, 1024, 16, 16
        img_a = make_grid_image(H, W)
        img_b = make_grid_image(H, W, spacing=20)
        data_label = f"Synthetic {H}x{W}"

    ctrs_y, ctrs_x = make_window_centres(H, W, nPY, nPX)
    pred_dy, pred_dx = make_poiseuille_predictor(nPY, nPX, ctrs_y, ctrs_x, u_max=5.0, H=H)

    # Run both methods
    c_out_a, c_out_b = call_kernel(lib, img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x)
    ref_out_a, ref_out_b = reference_warp(img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x)

    diff_a = c_out_a - ref_out_a
    diff_b = c_out_b - ref_out_b

    # Clamp display to [0, vmax] for image panels
    vmax = max(img_a.max(), img_b.max())
    vmin = 0

    # --- Figure 1: Full-image comparison for image A ---
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 11))
    fig1.suptitle(f"Fused Warp Comparison — Image A — {data_label}", fontsize=14, fontweight="bold")

    ax = axes1[0, 0]
    ax.imshow(img_a, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Original A")
    ax.axis("off")

    ax = axes1[0, 1]
    ax.imshow(c_out_a, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Warped A (C kernel)")
    ax.axis("off")

    ax = axes1[0, 2]
    ax.imshow(ref_out_a, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Warped A (cv2 reference)")
    ax.axis("off")

    # Difference map (symmetric colorbar centred at 0)
    abs_max_a = max(abs(diff_a.min()), abs(diff_a.max()), 1e-6)
    ax = axes1[1, 0]
    im = ax.imshow(diff_a, cmap="RdBu_r", vmin=-abs_max_a, vmax=abs_max_a)
    ax.set_title(f"Difference (C - cv2)\nmax={abs(diff_a).max():.2f}, mean={abs(diff_a).mean():.4f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Histogram of errors
    ax = axes1[1, 1]
    ax.hist(diff_a.ravel(), bins=200, color="steelblue", edgecolor="none", log=True)
    ax.set_xlabel("Pixel error (C - cv2)")
    ax.set_ylabel("Count (log)")
    ax.set_title("Error distribution (A)")
    ax.axvline(0, color="red", linewidth=0.5, linestyle="--")

    # Zoomed difference — centre crop
    crop = min(H, W) // 4
    cy, cx = H // 2, W // 2
    diff_crop = diff_a[cy - crop:cy + crop, cx - crop:cx + crop]
    abs_max_crop = max(abs(diff_crop.min()), abs(diff_crop.max()), 1e-6)
    ax = axes1[1, 2]
    im = ax.imshow(diff_crop, cmap="RdBu_r", vmin=-abs_max_crop, vmax=abs_max_crop)
    ax.set_title(f"Centre crop difference ({2*crop}x{2*crop})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig1.tight_layout()

    # --- Figure 2: Full-image comparison for image B ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
    fig2.suptitle(f"Fused Warp Comparison — Image B — {data_label}", fontsize=14, fontweight="bold")

    ax = axes2[0, 0]
    ax.imshow(img_b, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Original B")
    ax.axis("off")

    ax = axes2[0, 1]
    ax.imshow(c_out_b, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Warped B (C kernel)")
    ax.axis("off")

    ax = axes2[0, 2]
    ax.imshow(ref_out_b, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Warped B (cv2 reference)")
    ax.axis("off")

    abs_max_b = max(abs(diff_b.min()), abs(diff_b.max()), 1e-6)
    ax = axes2[1, 0]
    im = ax.imshow(diff_b, cmap="RdBu_r", vmin=-abs_max_b, vmax=abs_max_b)
    ax.set_title(f"Difference (C - cv2)\nmax={abs(diff_b).max():.2f}, mean={abs(diff_b).mean():.4f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes2[1, 1]
    ax.hist(diff_b.ravel(), bins=200, color="steelblue", edgecolor="none", log=True)
    ax.set_xlabel("Pixel error (C - cv2)")
    ax.set_ylabel("Count (log)")
    ax.set_title("Error distribution (B)")
    ax.axvline(0, color="red", linewidth=0.5, linestyle="--")

    diff_crop_b = diff_b[cy - crop:cy + crop, cx - crop:cx + crop]
    abs_max_crop_b = max(abs(diff_crop_b.min()), abs(diff_crop_b.max()), 1e-6)
    ax = axes2[1, 2]
    im = ax.imshow(diff_crop_b, cmap="RdBu_r", vmin=-abs_max_crop_b, vmax=abs_max_crop_b)
    ax.set_title(f"Centre crop difference ({2*crop}x{2*crop})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig2.tight_layout()

    # --- Figure 3: Predictor field + warp displacement ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle(f"Predictor Field — {data_label}", fontsize=14, fontweight="bold")

    ax = axes3[0]
    im = ax.imshow(pred_dx, cmap="RdBu_r", aspect="auto")
    ax.set_title(f"Predictor dx (coarse {nPY}x{nPX})")
    ax.set_xlabel("Grid x")
    ax.set_ylabel("Grid y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes3[1]
    im = ax.imshow(pred_dy, cmap="RdBu_r", aspect="auto")
    ax.set_title(f"Predictor dy (coarse {nPY}x{nPX})")
    ax.set_xlabel("Grid x")
    ax.set_ylabel("Grid y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Show the dense upsampled field from the C kernel (reconstruct by comparing warp offsets)
    # Dense dx ≈ 2 * (src_B_x - j) = 2 * (warped_pos - pixel_pos)
    # Instead, recompute via reference to show the upsampled field
    pix_y = np.arange(H, dtype=np.float32)
    pix_x = np.arange(W, dtype=np.float32)
    pred_frac_y = np.interp(pix_y, ctrs_y, np.arange(nPY, dtype=np.float32)).astype(np.float32)
    pred_frac_x = np.interp(pix_x, ctrs_x, np.arange(nPX, dtype=np.float32)).astype(np.float32)
    map_y_pred, map_x_pred = np.meshgrid(pred_frac_y, pred_frac_x, indexing="ij")
    dense_dx = cv2.remap(
        pred_dx.astype(np.float32), map_x_pred, map_y_pred,
        cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    ax = axes3[2]
    im = ax.imshow(dense_dx, cmap="RdBu_r")
    ax.set_title(f"Dense dx (upsampled to {H}x{W})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig3.tight_layout()

    plt.show()
    print("Figures displayed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test the fused symmetric warp C kernel."
    )
    parser.add_argument(
        "--synthetic-only", action="store_true",
        help="Skip real data tests (Test B)",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Set OMP_NUM_THREADS (default: system default)",
    )
    parser.add_argument(
        "--timing-runs", type=int, default=3,
        help="Number of timing runs for best-of-N (default: 3)",
    )
    parser.add_argument(
        "--figures", action="store_true",
        help="Generate comparison figures (original, warped, difference maps)",
    )
    args = parser.parse_args()

    if args.threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        print(f"OMP_NUM_THREADS = {args.threads}")

    print()
    lib = load_kernel()
    print(f"Loaded: libfusedwarp.dll")
    print()

    # Test A: Synthetic correctness
    pass_a = test_synthetic_correctness(lib)

    # Test B: Real data (unless --synthetic-only)
    pass_b = True
    if not args.synthetic_only:
        pass_b = test_real_data(lib, n_timing=args.timing_runs)

    # Test C: Synthetic timing
    test_synthetic_timing(lib, n_timing=args.timing_runs)

    # Figures
    if args.figures:
        print()
        generate_figures(lib)

    # Summary
    print("=" * 60)
    all_pass = pass_a and pass_b
    if all_pass:
        print("ALL CORRECTNESS TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
