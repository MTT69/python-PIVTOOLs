"""
GPU vs CPU bicubic warp — real image benchmark.

Compares compute-only times with full CPU saturation (all OMP threads)
and full GPU saturation on real PIV images.

Usage:
    python python/bench_warp_real.py <image_dir>
    python python/bench_warp_real.py <image_dir> --batches 1,5,10 --threads 20 --runs 10
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
_LIB_DIR = os.path.join(_ROOT_DIR, "pivtools_cli", "lib")
_EXT = ".dll" if os.name == "nt" else ".so"


# ── Set OMP threads BEFORE any library loads ──────────────────────────────

def setup_threads(n_threads):
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OMP_PROC_BIND"] = "spread"
    os.environ["OMP_PLACES"] = "cores"


# ── Load libraries ────────────────────────────────────────────────────────

def load_gpu_lib():
    if os.name == "nt":
        cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")
        for sub in ("bin", os.path.join("bin", "x64")):
            d = os.path.join(cuda_path, sub)
            if os.path.isdir(d):
                os.add_dll_directory(d)

    path = os.path.join(_ROOT_DIR, f"bench_warp{_EXT}")
    if not os.path.isfile(path):
        print(f"ERROR: {path} not found. Run build.bat first.")
        sys.exit(1)

    lib = ctypes.CDLL(path)
    lib.gpu_bicubic_warp_bench.argtypes = [
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ]
    lib.gpu_bicubic_warp_bench.restype = ctypes.c_int
    return lib


def load_cpu_lib():
    path = os.path.join(_LIB_DIR, f"libfusedwarp{_EXT}")
    if not os.path.isfile(path):
        print(f"ERROR: {path} not found. Build libfusedwarp first.")
        sys.exit(1)

    lib = ctypes.CDLL(path)
    lib.fused_symmetric_warp_batch.argtypes = [
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int,
    ]
    lib.fused_symmetric_warp_batch.restype = ctypes.c_int
    return lib


# ── Image loading ─────────────────────────────────────────────────────────

def load_images(image_dir, n_pairs):
    import cv2
    files = sorted(os.listdir(image_dir))
    a_files = [f for f in files if f.endswith("_A.tif")][:n_pairs]
    b_files = [f for f in files if f.endswith("_B.tif")][:n_pairs]
    N = min(len(a_files), len(b_files))

    imgs_a, imgs_b = [], []
    for i in range(N):
        a = cv2.imread(os.path.join(image_dir, a_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        b = cv2.imread(os.path.join(image_dir, b_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        imgs_a.append(a)
        imgs_b.append(b)

    return np.stack(imgs_a), np.stack(imgs_b)


# ── CPU warp ──────────────────────────────────────────────────────────────

def cpu_warp(lib, imgs_a, imgs_b, dy, dx):
    N, H, W = imgs_a.shape
    outs_a = np.zeros_like(imgs_a)
    outs_b = np.zeros_like(imgs_b)
    ctrs_y = np.arange(H, dtype=np.float32)
    ctrs_x = np.arange(W, dtype=np.float32)

    lib.fused_symmetric_warp_batch(
        np.ascontiguousarray(imgs_a), np.ascontiguousarray(imgs_b),
        outs_a, outs_b,
        np.ascontiguousarray(dy), np.ascontiguousarray(dx),
        N, H, W, H, W,
        ctrs_y, ctrs_x,
        0, 1,  # bicubic, shared_predictor
    )
    return outs_a, outs_b


# ── GPU warp ──────────────────────────────────────────────────────────────

def gpu_warp(lib, imgs_a, imgs_b, dy, dx):
    N, H, W = imgs_a.shape
    outs_a = np.zeros_like(imgs_a)
    outs_b = np.zeros_like(imgs_b)
    times = np.zeros(4, dtype=np.float32)

    lib.gpu_bicubic_warp_bench(
        np.ascontiguousarray(imgs_a), np.ascontiguousarray(imgs_b),
        outs_a, outs_b,
        np.ascontiguousarray(dy), np.ascontiguousarray(dx),
        N, H, W, times)
    return outs_a, outs_b, times


# ── Correctness ───────────────────────────────────────────────────────────

def check_correctness(gpu_lib, cpu_lib, H=512, W=512):
    np.random.seed(42)
    imgs_a = np.random.randn(1, H, W).astype(np.float32)
    imgs_b = np.random.randn(1, H, W).astype(np.float32)
    dy = (3.0 * np.sin(2 * np.pi * np.arange(H)[:, None] / H) *
          np.cos(2 * np.pi * np.arange(W)[None, :] / W)).astype(np.float32)
    dx = (2.5 * np.cos(2 * np.pi * np.arange(H)[:, None] / H) *
          np.sin(2 * np.pi * np.arange(W)[None, :] / W)).astype(np.float32)

    ref_a, ref_b = cpu_warp(cpu_lib, imgs_a, imgs_b, dy, dx)
    gpu_a, gpu_b, _ = gpu_warp(gpu_lib, imgs_a, imgs_b, dy, dx)

    margin = 4
    s = np.s_[:, margin:-margin, margin:-margin]
    err_a = np.max(np.abs(gpu_a[s] - ref_a[s]))
    err_b = np.max(np.abs(gpu_b[s] - ref_b[s]))
    max_ref = max(np.max(np.abs(ref_a[s])), np.max(np.abs(ref_b[s]))) + 1e-10
    rel_err = max(err_a, err_b) / max_ref

    ok = rel_err < 1e-4
    print(f"  GPU vs C fused_warp: rel_err={rel_err:.2e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPU vs CPU bicubic warp — real images")
    parser.add_argument("image_dir", help="Directory with B*_A.tif / B*_B.tif pairs")
    parser.add_argument("--batches", default="1,5,10,20",
                        help="Comma-separated batch sizes (default: 1,5,10,20)")
    parser.add_argument("--threads", type=int, default=0,
                        help="OMP threads (0=all cores, default: 0)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs (default: 3)")
    parser.add_argument("--runs", type=int, default=10, help="Timed runs (default: 10)")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batches.split(",")]
    omp_threads = args.threads if args.threads > 0 else os.cpu_count()

    # MUST set before loading any library
    setup_threads(omp_threads)

    print(f"Bicubic Warp Benchmark — GPU (RTX 4060) vs CPU ({os.cpu_count()} cores, {omp_threads} OMP threads)")
    print(f"Compute-only times (no PCIe transfer)")
    print()

    gpu_lib = load_gpu_lib()
    cpu_lib = load_cpu_lib()
    print("Libraries loaded.\n")

    # Correctness
    print("Correctness check:")
    check_correctness(gpu_lib, cpu_lib)
    print()

    # Generate displacement field once (shared across all batches)
    # Load one image to get dimensions
    test_a, _ = load_images(args.image_dir, 1)
    _, H, W = test_a.shape
    del test_a

    dy = (3.0 * np.sin(2 * np.pi * np.arange(H)[:, None] / H) *
          np.cos(2 * np.pi * np.arange(W)[None, :] / W)).astype(np.float32)
    dx = (2.5 * np.cos(2 * np.pi * np.arange(H)[:, None] / H) *
          np.sin(2 * np.pi * np.arange(W)[None, :] / W)).astype(np.float32)

    print(f"Image size: {H}x{W} ({H*W/1e6:.1f} MP)")
    print(f"Warmup: {args.warmup}, Timed runs: {args.runs} (median reported)")
    print()

    # Header
    print(f"{'Batch':>6s}  {'CPU (C)':>12s}  {'GPU':>12s}  {'Speedup':>10s}")
    print("-" * 50)

    for N in batch_sizes:
        # Check VRAM
        mem_gb = N * H * W * 4 * 6 / 1e9
        if mem_gb > 7.0:
            print(f"{N:>6d}  {'SKIP':>12s}  {'SKIP':>12s}  ~{mem_gb:.1f}GB > 7GB VRAM")
            continue

        imgs_a, imgs_b = load_images(args.image_dir, N)
        actual_N = imgs_a.shape[0]

        outs_a_cpu = np.zeros_like(imgs_a)
        outs_b_cpu = np.zeros_like(imgs_b)
        outs_a_gpu = np.zeros_like(imgs_a)
        outs_b_gpu = np.zeros_like(imgs_b)
        times = np.zeros(4, dtype=np.float32)

        # ── CPU warmup + timed ────────────────────────────────────────
        for _ in range(args.warmup):
            cpu_warp(cpu_lib, imgs_a, imgs_b, dy, dx)

        cpu_times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            cpu_warp(cpu_lib, imgs_a, imgs_b, dy, dx)
            cpu_times.append((time.perf_counter() - t0) * 1000)

        # ── GPU warmup + timed (compute-only) ─────────────────────────
        for _ in range(args.warmup):
            gpu_lib.gpu_bicubic_warp_bench(
                np.ascontiguousarray(imgs_a), np.ascontiguousarray(imgs_b),
                outs_a_gpu, outs_b_gpu,
                np.ascontiguousarray(dy), np.ascontiguousarray(dx),
                actual_N, H, W, times)

        gpu_compute_times = []
        for _ in range(args.runs):
            gpu_lib.gpu_bicubic_warp_bench(
                np.ascontiguousarray(imgs_a), np.ascontiguousarray(imgs_b),
                outs_a_gpu, outs_b_gpu,
                np.ascontiguousarray(dy), np.ascontiguousarray(dx),
                actual_N, H, W, times)
            gpu_compute_times.append(times[1])  # compute-only

        cpu_ms = np.median(cpu_times)
        gpu_ms = np.median(gpu_compute_times)
        speedup = cpu_ms / gpu_ms

        print(f"{actual_N:>6d}  {cpu_ms:>11.2f}ms  {gpu_ms:>11.2f}ms  {speedup:>9.1f}x")

    print()


if __name__ == "__main__":
    main()
