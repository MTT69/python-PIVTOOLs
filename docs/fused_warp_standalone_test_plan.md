# Plan: Standalone Fused Warp C Kernel — Manual Build & Test Tool

## Goal

Create a self-contained manual tool that:
1. Compiles `fused_warp.c` independently (no setup.py needed)
2. Loads it via ctypes
3. Runs it on synthetic and real PIV data
4. Compares output pixel-by-pixel against the current cv2.remap pipeline
5. Benchmarks both paths and reports timings

This allows iterating on the C code without touching the main pipeline.

---

## Files to Create

| File | Purpose |
|------|---------|
| `pivtools_cli/lib/fused_warp.h` | C header |
| `pivtools_cli/lib/fused_warp.c` | C implementation |
| `manual_tools/build_fused_warp.py` | Compile script (detects MSVC/GCC, builds DLL/SO) |
| `manual_tools/test_fused_warp.py` | Load library, run correctness + timing tests |

---

## Step 1: `build_fused_warp.py`

Standalone compile script. Detects platform, finds compiler, builds the shared library into `pivtools_cli/lib/`.

```python
"""
Build the fused_warp C library standalone.

Usage:
    python manual_tools/build_fused_warp.py

Produces:
    pivtools_cli/lib/libfusedwarp.dll   (Windows)
    pivtools_cli/lib/libfusedwarp.so    (Linux/macOS)
"""
import os, platform, subprocess, shutil, pathlib

pkg_dir = pathlib.Path(__file__).parent.parent
src_dir = pkg_dir / "pivtools_cli" / "lib"
src_file = src_dir / "fused_warp.c"
sys_name = platform.system().lower()

if sys_name == "windows":
    # MSVC
    out = src_dir / "libfusedwarp.dll"
    cmd = [
        "cl", "/O2", "/std:c11", "/openmp:experimental", "/MT", "/LD",
        f"/Fo{src_dir}/",
        str(src_file),
        f"/I{src_dir}",
        f"/Fe{out}",
    ]
elif sys_name == "darwin":
    # macOS GCC (Homebrew)
    compiler = shutil.which("gcc-15") or shutil.which("gcc-14") or shutil.which("gcc-13") or "gcc"
    out = src_dir / "libfusedwarp.so"
    cmd = [
        compiler, "-O3", "-fPIC", "-fopenmp", "-shared",
        str(src_file), f"-I{src_dir}",
        "-o", str(out), "-lm", "-fopenmp",
    ]
else:
    # Linux GCC
    out = src_dir / "libfusedwarp.so"
    cmd = [
        "gcc", "-O3", "-fPIC", "-fopenmp", "-shared",
        str(src_file), f"-I{src_dir}",
        "-o", str(out), "-lm", "-fopenmp",
    ]

print("BUILD:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    raise RuntimeError(f"Build failed (exit {result.returncode})")

# Clean up MSVC intermediates
for ext in ["*.obj", "*.exp", "*.lib"]:
    for f in src_dir.glob(ext):
        f.unlink()

print(f"OK -> {out}  ({out.stat().st_size / 1024:.0f} KB)")
```

---

## Step 2: `test_fused_warp.py`

Loads the compiled library and runs three test modes:

### Test A: Synthetic correctness

1. Create a 2048x2048 synthetic image (same grid pattern as the warp comparison script)
2. Create a Poiseuille displacement field on a coarse 16x16 window grid
3. Run the **current Python/cv2.remap pipeline** (dense remap + coord arithmetic + image warp)
4. Run the **C fused warp**
5. Compute max absolute difference and print pass/fail

### Test B: Real data correctness

1. Load a real PIV image pair from the validation dataset
2. Load a real predictor field from a pass-1 result (or fabricate one from known velocities)
3. Run both paths, compare pixel-by-pixel

### Test C: Timing benchmark

1. Run both paths on 1 MP, 4 MP, 25 MP synthetic images
2. Time each path (best of 3 repeats)
3. Print speedup table
4. Generate a bar chart

### Outline

```python
"""
Test and benchmark the fused_warp C kernel against cv2.remap.

Usage:
    python manual_tools/test_fused_warp.py [--real-data PATH]
"""
import ctypes, os, time, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Load library ---
lib_dir = os.path.join(os.path.dirname(__file__), "..", "pivtools_cli", "lib")
ext = ".dll" if os.name == "nt" else ".so"
lib = ctypes.CDLL(os.path.join(lib_dir, f"libfusedwarp{ext}"))

# Set argtypes for fused_symmetric_warp
from numpy.ctypeslib import ndpointer
lib.fused_symmetric_warp.argtypes = [
    ndpointer(np.float32, flags="C"),  # img_a
    ndpointer(np.float32, flags="C"),  # img_b
    ndpointer(np.float32, flags="C"),  # out_a
    ndpointer(np.float32, flags="C"),  # out_b
    ndpointer(np.float32, flags="C"),  # pred_dy
    ndpointer(np.float32, flags="C"),  # pred_dx
    ctypes.c_int, ctypes.c_int,        # H, W
    ctypes.c_int, ctypes.c_int,        # nPY, nPX
    ndpointer(np.float32, flags="C"),  # ctrs_y
    ndpointer(np.float32, flags="C"),  # ctrs_x
    ctypes.c_int,                       # interp_mode
]
lib.fused_symmetric_warp.restype = ctypes.c_int


def reference_warp_cv2(img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x, H, W):
    """
    Reproduce the exact Python/cv2 pipeline:
    1. cv2.remap to upsample predictor to (H, W)
    2. Coordinate arithmetic to build im_mesh_A, im_mesh_B
    3. cv2.remap to warp both images
    """
    nPY, nPX = pred_dy.shape

    # Build dense interpolation maps (same as _cache_interpolation_grids_unified)
    y_coords = np.arange(H, dtype=np.float32)
    x_coords = np.arange(W, dtype=np.float32)
    map_x_1d = np.interp(x_coords, ctrs_x, np.arange(nPX)).astype(np.float32)
    map_y_1d = np.interp(y_coords, ctrs_y, np.arange(nPY)).astype(np.float32)
    map_y_2d, map_x_2d = np.meshgrid(map_y_1d, map_x_1d, indexing="ij")

    # Upsample predictor to pixel resolution
    dense_dy = cv2.remap(pred_dy, map_x_2d, map_y_2d, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    dense_dx = cv2.remap(pred_dx, map_x_2d, map_y_2d, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)

    # Build coordinate maps
    y_mesh, x_mesh = np.meshgrid(y_coords, x_coords, indexing="ij")
    im_mesh = np.stack([y_mesh, x_mesh], axis=-1)
    delta_dense = np.stack([dense_dy, dense_dx], axis=-1)

    delta_0b = delta_dense / 2
    im_mesh_A = im_mesh + delta_0b    # note: +pred/2 for source of A
    im_mesh_B = im_mesh - delta_0b    # -pred/2 for source of B

    # Warp images
    out_a = cv2.remap(img_a, im_mesh_A[..., 1], im_mesh_A[..., 0],
                      cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out_b = cv2.remap(img_b, im_mesh_B[..., 1], im_mesh_B[..., 0],
                      cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out_a, out_b


def fused_warp_c(img_a, img_b, pred_dy, pred_dx, ctrs_y, ctrs_x, H, W):
    """Call the C fused warp kernel."""
    out_a = np.empty((H, W), dtype=np.float32)
    out_b = np.empty((H, W), dtype=np.float32)
    err = lib.fused_symmetric_warp(
        np.ascontiguousarray(img_a, np.float32),
        np.ascontiguousarray(img_b, np.float32),
        out_a, out_b,
        np.ascontiguousarray(pred_dy, np.float32),
        np.ascontiguousarray(pred_dx, np.float32),
        H, W,
        pred_dy.shape[0], pred_dy.shape[1],
        np.ascontiguousarray(ctrs_y, np.float32),
        np.ascontiguousarray(ctrs_x, np.float32),
        0  # bicubic
    )
    assert err == 0, f"C kernel returned error {err}"
    return out_a, out_b


def make_test_data(H, W, n_win_y, n_win_x, u_max=12.0):
    """Create synthetic image, predictor field, and window centres."""
    # Grid image
    y = np.arange(H, dtype=np.float64)
    x = np.arange(W, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    img = 0.5 + 0.3 * np.sin(2*np.pi*xx/12) + 0.2 * np.sin(2*np.pi*yy/12)
    img = (np.clip(img, 0, 1) * 255).astype(np.float32)

    # Poiseuille predictor on coarse grid
    ctrs_y = np.linspace(H/(2*n_win_y), H - H/(2*n_win_y), n_win_y).astype(np.float32)
    ctrs_x = np.linspace(W/(2*n_win_x), W - W/(2*n_win_x), n_win_x).astype(np.float32)

    y_norm = np.linspace(-1, 1, n_win_y, dtype=np.float32)
    pred_dx = np.broadcast_to(
        (u_max * (1 - y_norm**2))[:, None],
        (n_win_y, n_win_x)
    ).copy().astype(np.float32)
    pred_dy = np.zeros_like(pred_dx)

    return img, pred_dy, pred_dx, ctrs_y, ctrs_x


def test_correctness():
    """Compare C kernel vs cv2.remap on synthetic data."""
    print("=" * 60)
    print("  CORRECTNESS TEST")
    print("=" * 60)

    for label, H, W, nwy, nwx in [
        ("Small",  512,  512,  8,  8),
        ("1 MP",  1000, 1000, 16, 16),
        ("4 MP",  2048, 2048, 32, 32),
    ]:
        img, pred_dy, pred_dx, ctrs_y, ctrs_x = make_test_data(H, W, nwy, nwx)

        ref_a, ref_b = reference_warp_cv2(img, img, pred_dy, pred_dx, ctrs_y, ctrs_x, H, W)
        fus_a, fus_b = fused_warp_c(img, img, pred_dy, pred_dx, ctrs_y, ctrs_x, H, W)

        diff_a = np.abs(ref_a - fus_a)
        diff_b = np.abs(ref_b - fus_b)
        max_err = max(diff_a.max(), diff_b.max())
        mean_err = (diff_a.mean() + diff_b.mean()) / 2

        status = "PASS" if max_err < 2.0 else "FAIL"
        print(f"  {label:8s}  max_err={max_err:.3f}  mean_err={mean_err:.4f}  [{status}]")


def test_timing():
    """Benchmark C kernel vs cv2.remap across resolutions."""
    print("\n" + "=" * 60)
    print("  TIMING BENCHMARK")
    print("=" * 60)

    results = []
    for label, H, W, nwy, nwx in [
        ("1 MP",   1000, 1000,  16,  16),
        ("4 MP",   2048, 2048,  32,  32),
        ("25 MP",  5000, 5000,  80,  80),
    ]:
        img, pred_dy, pred_dx, ctrs_y, ctrs_x = make_test_data(H, W, nwy, nwx)

        # Warm up
        reference_warp_cv2(img, img, pred_dy, pred_dx, ctrs_y, ctrs_x, H, W)
        fused_warp_c(img, img, pred_dy, pred_dx, ctrs_y, ctrs_x, H, W)

        # Time cv2 path
        times_cv2 = []
        for _ in range(3):
            t0 = time.perf_counter()
            reference_warp_cv2(img, img, pred_dy, pred_dx, ctrs_y, ctrs_x, H, W)
            times_cv2.append(time.perf_counter() - t0)

        # Time C path
        times_c = []
        for _ in range(3):
            t0 = time.perf_counter()
            fused_warp_c(img, img, pred_dy, pred_dx, ctrs_y, ctrs_x, H, W)
            times_c.append(time.perf_counter() - t0)

        t_cv2 = min(times_cv2) * 1000
        t_c = min(times_c) * 1000
        speedup = t_cv2 / t_c

        print(f"  {label:8s}  cv2={t_cv2:7.1f} ms   C={t_c:7.1f} ms   speedup={speedup:.2f}x")
        results.append((label, t_cv2, t_c))

    return results


if __name__ == "__main__":
    test_correctness()
    results = test_timing()
```

---

## Step 3: Implementation order

1. Write `fused_warp.h` and `fused_warp.c`
2. Write `manual_tools/build_fused_warp.py`
3. Run build script -> produces `libfusedwarp.dll`
4. Write `manual_tools/test_fused_warp.py`
5. Run test script -> correctness + timing results
6. Iterate on C code if needed (fix edge cases, tune OpenMP scheduling)
7. Once verified, proceed with main pipeline integration (see `fused_warp_kernel_plan.md`)

---

## Notes

- The reference `reference_warp_cv2()` reproduces the **exact** pipeline from `cpu_instantaneous.py` lines 837-896: dense remap, coordinate arithmetic, symmetric image warp
- **Expected difference source**: cv2.remap uses a slightly different bicubic boundary handling (reflects vs clamps at edges). Interior pixels should match to <0.5 grey levels; edge pixels may differ by up to ~2
- The C kernel uses OpenMP `parallel for` over rows, which releases the GIL automatically when called from Python via ctypes
- **No changes to the main pipeline** are needed for this testing phase
