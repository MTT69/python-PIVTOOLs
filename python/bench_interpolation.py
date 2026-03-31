"""
GPU vs CPU bicubic interpolation (symmetric warp) benchmark.

Usage:
    python python/bench_interpolation.py
    python python/bench_interpolation.py --sizes 1024,2048 --batches 1,5,10
    python python/bench_interpolation.py --csv results_interp.csv
"""

import argparse
import ctypes
import os
import sys
import time
import numpy as np
from numpy.ctypeslib import ndpointer

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_BENCH_DIR)
_EXT = ".dll" if os.name == "nt" else ".so"


def _add_cuda_dll_dirs():
    """Register CUDA runtime DLL directories on Windows."""
    if os.name != "nt":
        return
    cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")
    for subdir in ("bin", os.path.join("bin", "x64")):
        d = os.path.join(cuda_path, subdir)
        if os.path.isdir(d):
            os.add_dll_directory(d)


def load_gpu_lib():
    """Load the compiled bench_warp CUDA library."""
    path = os.path.join(_ROOT_DIR, f"bench_warp{_EXT}")
    if not os.path.isfile(path):
        print(f"ERROR: GPU library not found at {path}")
        print(f"Build it first: see build.bat or build.sh")
        sys.exit(1)

    _add_cuda_dll_dirs()
    lib = ctypes.CDLL(path)
    lib.gpu_bicubic_warp_bench.argtypes = [
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # h_imgs_a
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # h_imgs_b
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # h_outs_a
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # h_outs_b
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # h_dy
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # h_dx
        ctypes.c_int, ctypes.c_int, ctypes.c_int,           # N, H, W
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # out_times[4]
    ]
    lib.gpu_bicubic_warp_bench.restype = ctypes.c_int
    return lib


# ── CPU baseline: try C library, fall back to scipy ───────────────────────

def try_load_c_warp():
    """Attempt to load libfusedwarp for CPU baseline."""
    sys.path.insert(0, _BENCH_DIR)
    from load_cpu_baseline import load_fused_warp_lib
    return load_fused_warp_lib()


def cpu_warp_c(lib, imgs_a, imgs_b, dy, dx, N, H, W):
    """
    Call the C fused_symmetric_warp_batch() as CPU baseline.

    The C function expects predictor fields + grid centres for its fused upsampling.
    Since our benchmark uses a pre-computed dense displacement field, we pass it
    as the "predictor" on a 1:1 grid (nPY=H, nPX=W, centres = pixel indices).
    """
    outs_a = np.zeros_like(imgs_a)
    outs_b = np.zeros_like(imgs_b)
    ctrs_y = np.arange(H, dtype=np.float32)
    ctrs_x = np.arange(W, dtype=np.float32)

    err = lib.fused_symmetric_warp_batch(
        np.ascontiguousarray(imgs_a),
        np.ascontiguousarray(imgs_b),
        outs_a, outs_b,
        np.ascontiguousarray(dy),
        np.ascontiguousarray(dx),
        N, H, W,
        H, W,           # nPY=H, nPX=W (dense grid)
        ctrs_y, ctrs_x,
        0,               # interp_mode=0 (bicubic)
        1,               # shared_predictor=1
    )
    if err != 0:
        print(f"Warning: fused_symmetric_warp_batch returned error {err}")
    return outs_a, outs_b


def cpu_warp_scipy(imgs_a, imgs_b, dy, dx, N, H, W):
    """Scipy fallback: map_coordinates with order=3 (cubic spline)."""
    from scipy.ndimage import map_coordinates

    outs_a = np.zeros_like(imgs_a)
    outs_b = np.zeros_like(imgs_b)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    for n in range(N):
        half_dy = 0.5 * dy
        half_dx = 0.5 * dx
        coords_a = np.array([yy - half_dy, xx - half_dx])
        coords_b = np.array([yy + half_dy, xx + half_dx])
        outs_a[n] = map_coordinates(imgs_a[n], coords_a, order=3, mode='constant').astype(np.float32)
        outs_b[n] = map_coordinates(imgs_b[n], coords_b, order=3, mode='constant').astype(np.float32)

    return outs_a, outs_b


# ── Correctness check ─────────────────────────────────────────────────────

def check_correctness(gpu_lib, c_lib, H=256, W=256):
    """Verify GPU warp output matches C library (or scipy if C unavailable)."""
    np.random.seed(42)
    imgs_a = np.random.randn(1, H, W).astype(np.float32)
    imgs_b = np.random.randn(1, H, W).astype(np.float32)
    # Small smooth displacement field (typical for PIV: ~2-5 pixel shifts)
    dy = (3.0 * np.sin(2 * np.pi * np.arange(H)[:, None] / H) *
          np.cos(2 * np.pi * np.arange(W)[None, :] / W)).astype(np.float32)
    dx = (2.5 * np.cos(2 * np.pi * np.arange(H)[:, None] / H) *
          np.sin(2 * np.pi * np.arange(W)[None, :] / W)).astype(np.float32)

    # GPU
    gpu_a = np.zeros_like(imgs_a)
    gpu_b = np.zeros_like(imgs_b)
    times = np.zeros(4, dtype=np.float32)
    err = gpu_lib.gpu_bicubic_warp_bench(
        np.ascontiguousarray(imgs_a), np.ascontiguousarray(imgs_b),
        gpu_a, gpu_b,
        np.ascontiguousarray(dy), np.ascontiguousarray(dx),
        1, H, W, times)
    if err != 0:
        print(f"  GPU error!")
        return False

    # Reference
    if c_lib is not None:
        ref_a, ref_b = cpu_warp_c(c_lib, imgs_a, imgs_b, dy, dx, 1, H, W)
        label = "C library"
    else:
        ref_a, ref_b = cpu_warp_scipy(imgs_a, imgs_b, dy, dx, 1, H, W)
        label = "scipy (approx)"

    # Compare only interior pixels (boundaries have different clamping)
    margin = 4
    s = np.s_[:, margin:-margin, margin:-margin]
    max_err_a = np.max(np.abs(gpu_a[s] - ref_a[s]))
    max_err_b = np.max(np.abs(gpu_b[s] - ref_b[s]))
    max_ref = max(np.max(np.abs(ref_a[s])), np.max(np.abs(ref_b[s]))) + 1e-10
    rel_err = max(max_err_a, max_err_b) / max_ref

    # scipy uses cubic spline (not Keys a=-0.75), so allow larger tolerance
    tol = 1e-4 if c_lib is not None else 0.05
    ok = rel_err < tol
    status = "PASS" if ok else "FAIL"
    print(f"  vs {label}: max_abs_err={max(max_err_a, max_err_b):.2e}  "
          f"rel_err={rel_err:.2e}  [{status}]")
    return ok


# ── Benchmark loop ────────────────────────────────────────────────────────

def run_benchmark(gpu_lib, c_lib, image_sizes, batch_sizes, n_warmup, n_runs):
    results = []

    for img_size in image_sizes:
        H = W = img_size
        for N in batch_sizes:
            # Check GPU memory (rough estimate)
            mem_gb = N * H * W * 4 * 6 / 1e9  # 6 arrays (2 in, 2 out, 2 disp)
            if mem_gb > 7.0:  # RTX 4060 has 8GB
                print(f"  Skipping {img_size}^2 x {N}: estimated {mem_gb:.1f}GB > 7GB VRAM limit")
                continue

            np.random.seed(0)
            imgs_a = np.random.randn(N, H, W).astype(np.float32)
            imgs_b = np.random.randn(N, H, W).astype(np.float32)
            dy = (3.0 * np.sin(2 * np.pi * np.arange(H)[:, None] / H) *
                  np.cos(2 * np.pi * np.arange(W)[None, :] / W)).astype(np.float32)
            dx = (2.5 * np.cos(2 * np.pi * np.arange(H)[:, None] / H) *
                  np.sin(2 * np.pi * np.arange(W)[None, :] / W)).astype(np.float32)

            gpu_a = np.zeros_like(imgs_a)
            gpu_b = np.zeros_like(imgs_b)
            times = np.zeros(4, dtype=np.float32)

            # GPU warmup
            for _ in range(n_warmup):
                gpu_lib.gpu_bicubic_warp_bench(
                    np.ascontiguousarray(imgs_a), np.ascontiguousarray(imgs_b),
                    gpu_a, gpu_b,
                    np.ascontiguousarray(dy), np.ascontiguousarray(dx),
                    N, H, W, times)

            # GPU timed runs
            gpu_total, gpu_compute, gpu_h2d, gpu_d2h = [], [], [], []
            for _ in range(n_runs):
                gpu_lib.gpu_bicubic_warp_bench(
                    np.ascontiguousarray(imgs_a), np.ascontiguousarray(imgs_b),
                    gpu_a, gpu_b,
                    np.ascontiguousarray(dy), np.ascontiguousarray(dx),
                    N, H, W, times)
                gpu_total.append(times[0])
                gpu_compute.append(times[1])
                gpu_h2d.append(times[2])
                gpu_d2h.append(times[3])

            # CPU baseline
            if c_lib is not None:
                # Warmup
                cpu_warp_c(c_lib, imgs_a, imgs_b, dy, dx, N, H, W)
                cpu_times = []
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    cpu_warp_c(c_lib, imgs_a, imgs_b, dy, dx, N, H, W)
                    cpu_times.append((time.perf_counter() - t0) * 1000)
                cpu_ms = np.median(cpu_times)
                cpu_label = "C (fused_warp)"
            else:
                # Warmup
                cpu_warp_scipy(imgs_a, imgs_b, dy, dx, N, H, W)
                cpu_times = []
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    cpu_warp_scipy(imgs_a, imgs_b, dy, dx, N, H, W)
                    cpu_times.append((time.perf_counter() - t0) * 1000)
                cpu_ms = np.median(cpu_times)
                cpu_label = "scipy"

            row = {
                "image_size": img_size,
                "batch_size": N,
                "cpu_ms": cpu_ms,
                "cpu_label": cpu_label,
                "gpu_total_ms": np.median(gpu_total),
                "gpu_compute_ms": np.median(gpu_compute),
                "gpu_h2d_ms": np.median(gpu_h2d),
                "gpu_d2h_ms": np.median(gpu_d2h),
                "speedup_total": cpu_ms / np.median(gpu_total),
                "speedup_compute": cpu_ms / np.median(gpu_compute),
            }
            results.append(row)

    return results


def print_results(results):
    print()
    print("Bicubic Interpolation Benchmark (Symmetric Warp)")
    print("=" * 100)

    header = (f"{'Image':>8s}  {'Batch':>6s}  "
              f"{'CPU':>10s}  "
              f"{'GPU+xfer':>9s}  {'GPU only':>9s}  {'H2D':>7s}  {'D2H':>7s}  "
              f"{'Speedup':>9s}  {'(compute)':>9s}")
    print(header)
    print("-" * len(header))

    for r in results:
        img = f"{r['image_size']}^2"
        cpu_str = f"{r['cpu_ms']:.2f}ms"
        print(f"{img:>8s}  {r['batch_size']:>6d}  "
              f"{cpu_str:>10s}  "
              f"{r['gpu_total_ms']:>8.2f}ms  {r['gpu_compute_ms']:>8.2f}ms  "
              f"{r['gpu_h2d_ms']:>6.2f}ms  {r['gpu_d2h_ms']:>6.2f}ms  "
              f"{r['speedup_total']:>8.1f}x  {r['speedup_compute']:>8.1f}x")

    if results:
        print(f"\nCPU baseline: {results[0]['cpu_label']}")
    print()


def save_csv(results, path):
    import csv
    keys = results[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


def load_real_images(image_dir, batch_size):
    """Load batch_size A/B pairs from a directory of tif images."""
    import cv2
    files = sorted(os.listdir(image_dir))
    a_files = [f for f in files if f.endswith("_A.tif")][:batch_size]
    b_files = [f for f in files if f.endswith("_B.tif")][:batch_size]
    N = min(len(a_files), len(b_files), batch_size)

    imgs_a = []
    imgs_b = []
    for i in range(N):
        a = cv2.imread(os.path.join(image_dir, a_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        b = cv2.imread(os.path.join(image_dir, b_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        imgs_a.append(a)
        imgs_b.append(b)

    imgs_a = np.stack(imgs_a)
    imgs_b = np.stack(imgs_b)
    H, W = imgs_a.shape[1], imgs_a.shape[2]
    print(f"  Loaded {N} pairs, shape={H}x{W}  ({H*W/1e6:.1f} MP)")
    return imgs_a, imgs_b


def main():
    parser = argparse.ArgumentParser(description="GPU vs CPU bicubic warp benchmark")
    parser.add_argument("--sizes", default="1024,2048,4096",
                        help="Comma-separated image sizes for synthetic test (default: 1024,2048,4096)")
    parser.add_argument("--batches", default="1,5,10,20",
                        help="Comma-separated batch sizes (default: 1,5,10,20)")
    parser.add_argument("--threads", type=int, default=0,
                        help="OMP threads for C library (0=all cores, default: 0)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument("--runs", type=int, default=10, help="Timed iterations (default: 10)")
    parser.add_argument("--image-dir", default=None, help="Directory with A/B .tif pairs for real images")
    parser.add_argument("--csv", default=None, help="Save results to CSV file")
    args = parser.parse_args()

    image_sizes = [int(x) for x in args.sizes.split(",")]
    batch_sizes = [int(x) for x in args.batches.split(",")]

    # Set CPU thread counts for saturation
    omp_threads = args.threads if args.threads > 0 else os.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    print(f"CPU threads: OMP_NUM_THREADS={omp_threads}")
    print()

    print("Loading GPU library...")
    gpu_lib = load_gpu_lib()

    print("Loading CPU C library...")
    c_lib = try_load_c_warp()
    if c_lib is not None:
        print(f"  libfusedwarp loaded — OpenMP ({omp_threads} threads)")
    else:
        print("  Not available — using scipy map_coordinates as CPU baseline")

    # Correctness
    print("\nCorrectness check (GPU vs CPU):")
    check_correctness(gpu_lib, c_lib)

    # Real image benchmark
    if args.image_dir:
        print(f"\n--- Real images from {args.image_dir} ---")
        real_results = []
        for N in batch_sizes:
            imgs_a, imgs_b = load_real_images(args.image_dir, N)
            actual_N, H, W = imgs_a.shape
            dy = (3.0 * np.sin(2 * np.pi * np.arange(H)[:, None] / H) *
                  np.cos(2 * np.pi * np.arange(W)[None, :] / W)).astype(np.float32)
            dx = (2.5 * np.cos(2 * np.pi * np.arange(H)[:, None] / H) *
                  np.sin(2 * np.pi * np.arange(W)[None, :] / W)).astype(np.float32)

            gpu_a = np.zeros_like(imgs_a)
            gpu_b = np.zeros_like(imgs_b)
            times = np.zeros(4, dtype=np.float32)

            # Check VRAM
            mem_gb = actual_N * H * W * 4 * 6 / 1e9
            if mem_gb > 7.0:
                print(f"  Skipping batch={actual_N}: ~{mem_gb:.1f}GB > 7GB VRAM")
                continue

            # GPU
            for _ in range(args.warmup):
                gpu_lib.gpu_bicubic_warp_bench(
                    np.ascontiguousarray(imgs_a), np.ascontiguousarray(imgs_b),
                    gpu_a, gpu_b, np.ascontiguousarray(dy), np.ascontiguousarray(dx),
                    actual_N, H, W, times)
            gpu_compute = []
            for _ in range(args.runs):
                gpu_lib.gpu_bicubic_warp_bench(
                    np.ascontiguousarray(imgs_a), np.ascontiguousarray(imgs_b),
                    gpu_a, gpu_b, np.ascontiguousarray(dy), np.ascontiguousarray(dx),
                    actual_N, H, W, times)
                gpu_compute.append(times[1])

            # CPU
            if c_lib is not None:
                cpu_warp_c(c_lib, imgs_a, imgs_b, dy, dx, actual_N, H, W)
                cpu_times = []
                for _ in range(args.runs):
                    t0 = time.perf_counter()
                    cpu_warp_c(c_lib, imgs_a, imgs_b, dy, dx, actual_N, H, W)
                    cpu_times.append((time.perf_counter() - t0) * 1000)
                cpu_ms = np.median(cpu_times)
            else:
                cpu_ms = 0.0

            gpu_ms = np.median(gpu_compute)
            real_results.append({
                "image_size": f"{H}x{W}",
                "batch_size": actual_N,
                "cpu_ms": cpu_ms,
                "gpu_compute_ms": gpu_ms,
                "speedup": cpu_ms / gpu_ms if gpu_ms > 0 else 0,
            })

        if real_results:
            print(f"\nBicubic Warp — Real {real_results[0]['image_size']} images  (compute-only)")
            print(f"CPU: {os.cpu_count()} cores, OMP_NUM_THREADS={omp_threads}")
            print("=" * 70)
            print(f"{'Image':>12s}  {'Batch':>6s}  {'CPU(C)':>10s}  {'GPU':>10s}  {'Speedup':>10s}")
            print("-" * 70)
            for r in real_results:
                print(f"{r['image_size']:>12s}  {r['batch_size']:>6d}  "
                      f"{r['cpu_ms']:>9.2f}ms  {r['gpu_compute_ms']:>9.2f}ms  "
                      f"{r['speedup']:>9.1f}x")
            print()

    # Synthetic benchmark
    print(f"\nRunning synthetic benchmark: {args.warmup} warmup + {args.runs} timed runs (median)")
    results = run_benchmark(gpu_lib, c_lib, image_sizes, batch_sizes,
                            args.warmup, args.runs)
    print_results(results)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()
