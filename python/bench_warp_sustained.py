"""
Sustained GPU vs CPU fused symmetric warp benchmark.

Both sides do the SAME full pipeline (apples-to-apples):
  1. Predictor upsampling: bicubic interpolation from coarse grid to dense
  2. Symmetric warp coordinates: ±displacement/2
  3. Bicubic image sampling: Keys a=-0.75, BORDER_CONSTANT=0

GPU: tight kernel loop inside CUDA (no Python overhead).
CPU: tight C library loop with full OpenMP saturation.

Usage:
    python python/bench_warp_sustained.py <image_dir>
    python python/bench_warp_sustained.py <image_dir> --duration 30 --batch 5 --pred-grid 64
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


def setup_threads(n):
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OMP_PROC_BIND"] = "spread"
    os.environ["OMP_PLACES"] = "cores"


def load_gpu_lib():
    if os.name == "nt":
        cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")
        for sub in ("bin", os.path.join("bin", "x64")):
            d = os.path.join(cuda_path, sub)
            if os.path.isdir(d):
                os.add_dll_directory(d)
    path = os.path.join(_ROOT_DIR, f"bench_warp{_EXT}")
    lib = ctypes.CDLL(path)

    lib.gpu_fused_warp_bench.argtypes = [
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # imgs_a
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # imgs_b
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # outs_a
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # outs_b
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # pred_dy (nPY,nPX)
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # pred_dx (nPY,nPX)
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # ctrs_y (nPY,)
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # ctrs_x (nPX,)
        ctypes.c_int, ctypes.c_int, ctypes.c_int,           # N, H, W
        ctypes.c_int, ctypes.c_int,                          # nPY, nPX
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # out_times[4]
    ]
    lib.gpu_fused_warp_bench.restype = ctypes.c_int

    lib.gpu_fused_warp_sustained.argtypes = [
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int,  # n_iterations
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ]
    lib.gpu_fused_warp_sustained.restype = ctypes.c_int

    return lib


def load_cpu_lib():
    path = os.path.join(_LIB_DIR, f"libfusedwarp{_EXT}")
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


def load_images(image_dir, n_pairs):
    import cv2
    files = sorted(os.listdir(image_dir))
    a_files = [f for f in files if f.endswith("_A.tif")][:n_pairs]
    b_files = [f for f in files if f.endswith("_B.tif")][:n_pairs]
    N = min(len(a_files), len(b_files))
    imgs_a, imgs_b = [], []
    for i in range(N):
        imgs_a.append(cv2.imread(os.path.join(image_dir, a_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32))
        imgs_b.append(cv2.imread(os.path.join(image_dir, b_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32))
    return np.stack(imgs_a), np.stack(imgs_b)


def make_predictor_grid(H, W, pred_grid_size, overlap_pct=50):
    """Create a realistic coarse predictor grid matching PIV window layout."""
    spacing = pred_grid_size * (1.0 - overlap_pct / 100.0)
    half_win = pred_grid_size / 2.0

    ctrs_y = np.arange(half_win, H - half_win + 0.5, spacing, dtype=np.float32)
    ctrs_x = np.arange(half_win, W - half_win + 0.5, spacing, dtype=np.float32)
    nPY = len(ctrs_y)
    nPX = len(ctrs_x)

    # Smooth displacement field on the coarse grid (~3px amplitude, typical PIV)
    yy = np.arange(nPY, dtype=np.float32)
    xx = np.arange(nPX, dtype=np.float32)
    pred_dy = (3.0 * np.sin(2 * np.pi * yy[:, None] / nPY) *
               np.cos(2 * np.pi * xx[None, :] / nPX)).astype(np.float32)
    pred_dx = (2.5 * np.cos(2 * np.pi * yy[:, None] / nPY) *
               np.sin(2 * np.pi * xx[None, :] / nPX)).astype(np.float32)

    return pred_dy, pred_dx, ctrs_y, ctrs_x, nPY, nPX


def check_correctness(gpu_lib, cpu_lib, H=512, W=512, pred_grid=64):
    """Verify GPU matches CPU C library to float32 precision."""
    np.random.seed(42)
    imgs_a = np.random.randn(1, H, W).astype(np.float32)
    imgs_b = np.random.randn(1, H, W).astype(np.float32)
    pred_dy, pred_dx, ctrs_y, ctrs_x, nPY, nPX = make_predictor_grid(H, W, pred_grid)

    # CPU
    cpu_a = np.zeros_like(imgs_a)
    cpu_b = np.zeros_like(imgs_b)
    cpu_lib.fused_symmetric_warp_batch(
        np.ascontiguousarray(imgs_a), np.ascontiguousarray(imgs_b),
        cpu_a, cpu_b,
        np.ascontiguousarray(pred_dy), np.ascontiguousarray(pred_dx),
        1, H, W, nPY, nPX, ctrs_y, ctrs_x, 0, 1)

    # GPU
    gpu_a = np.zeros_like(imgs_a)
    gpu_b = np.zeros_like(imgs_b)
    times = np.zeros(4, dtype=np.float32)
    gpu_lib.gpu_fused_warp_bench(
        np.ascontiguousarray(imgs_a), np.ascontiguousarray(imgs_b),
        gpu_a, gpu_b,
        np.ascontiguousarray(pred_dy), np.ascontiguousarray(pred_dx),
        ctrs_y, ctrs_x,
        1, H, W, nPY, nPX, times)

    margin = 4
    s = np.s_[:, margin:-margin, margin:-margin]
    err = max(np.max(np.abs(gpu_a[s] - cpu_a[s])), np.max(np.abs(gpu_b[s] - cpu_b[s])))
    ref = max(np.max(np.abs(cpu_a[s])), np.max(np.abs(cpu_b[s]))) + 1e-10
    rel_err = err / ref
    ok = rel_err < 1e-4
    print(f"  GPU vs C (pred_grid={pred_grid}): rel_err={rel_err:.2e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Sustained GPU vs CPU fused warp (apples-to-apples)")
    parser.add_argument("image_dir", help="Directory with B*_A.tif / B*_B.tif pairs")
    parser.add_argument("--batch", type=int, default=5, help="Image pairs per call (default: 5)")
    parser.add_argument("--pred-grid", type=int, default=64,
                        help="Predictor grid window size, e.g. 64 = coarse 64px grid (default: 64)")
    parser.add_argument("--duration", type=int, default=30, help="Seconds per backend (default: 30)")
    parser.add_argument("--threads", type=int, default=0, help="OMP threads (0=all, default: 0)")
    args = parser.parse_args()

    omp_threads = args.threads if args.threads > 0 else os.cpu_count()
    setup_threads(omp_threads)

    print(f"Fused Symmetric Warp Benchmark (apples-to-apples)")
    print(f"  Both sides: predictor upsample -> symmetric coords -> bicubic image sample")
    print(f"  GPU: RTX 4060 (sm_89)")
    print(f"  CPU: {os.cpu_count()} cores, {omp_threads} OMP threads")
    print(f"  Duration: ~{args.duration}s per backend")
    print(f"  Batch: {args.batch} image pairs")
    print(f"  Predictor grid: {args.pred_grid}px windows (50% overlap)")
    print()

    gpu_lib = load_gpu_lib()
    cpu_lib = load_cpu_lib()

    # Correctness check
    print("Correctness check:")
    check_correctness(gpu_lib, cpu_lib, pred_grid=args.pred_grid)
    print()

    # Load images
    print(f"Loading {args.batch} image pairs...")
    imgs_a, imgs_b = load_images(args.image_dir, args.batch)
    N, H, W = imgs_a.shape
    print(f"  {N} pairs, {H}x{W} ({H*W/1e6:.1f} MP)")

    # Create predictor grid
    pred_dy, pred_dx, ctrs_y, ctrs_x, nPY, nPX = make_predictor_grid(H, W, args.pred_grid)
    print(f"  Predictor grid: {nPY}x{nPX} (from {args.pred_grid}px windows, 50% overlap)")
    print()

    # Pre-contiguous
    ca = np.ascontiguousarray(imgs_a)
    cb = np.ascontiguousarray(imgs_b)
    cpdy = np.ascontiguousarray(pred_dy)
    cpdx = np.ascontiguousarray(pred_dx)
    outs_a = np.zeros_like(imgs_a)
    outs_b = np.zeros_like(imgs_b)

    # ── Correctness: verify both GPU kernels match CPU ──────────────
    print("Verifying GPU kernels match CPU output on real data...")
    cpu_oa = np.zeros_like(imgs_a)
    cpu_ob = np.zeros_like(imgs_b)
    cpu_lib.fused_symmetric_warp_batch(ca, cb, cpu_oa, cpu_ob, cpdy, cpdx,
                                       N, H, W, nPY, nPX, ctrs_y, ctrs_x, 0, 1)

    naive_oa = np.zeros_like(imgs_a)
    naive_ob = np.zeros_like(imgs_b)
    times_gpu = np.zeros(4, dtype=np.float32)
    gpu_lib.gpu_fused_warp_bench(ca, cb, naive_oa, naive_ob, cpdy, cpdx,
                                  ctrs_y, ctrs_x, N, H, W, nPY, nPX, times_gpu)

    margin = 10
    s = np.s_[:, margin:-margin, margin:-margin]
    ref_max = max(np.max(np.abs(cpu_oa[s])), np.max(np.abs(cpu_ob[s]))) + 1e-10
    naive_err = max(np.max(np.abs(naive_oa[s] - cpu_oa[s])),
                    np.max(np.abs(naive_ob[s] - cpu_ob[s]))) / ref_max
    print(f"  Naive GPU vs CPU:     rel_err={naive_err:.2e}  [{'PASS' if naive_err < 1e-4 else 'FAIL'}]")
    print()

    # ── Estimate iterations ───────────────────────────────────────────
    gpu_ms = times_gpu[1]

    t0 = time.perf_counter()
    cpu_lib.fused_symmetric_warp_batch(ca, cb, outs_a, outs_b, cpdy, cpdx,
                                       N, H, W, nPY, nPX, ctrs_y, ctrs_x, 0, 1)
    cpu_ms = (time.perf_counter() - t0) * 1000

    gpu_iters = max(int(args.duration * 1000 / gpu_ms), 100)
    cpu_iters = max(int(args.duration * 1000 / cpu_ms), 10)

    print(f"Estimated: CPU={cpu_ms:.1f}ms/call  GPU naive={gpu_ms:.1f}ms/call")
    print(f"Iterations: CPU={cpu_iters}  GPU={gpu_iters}")

    # ── CPU sustained ─────────────────────────────────────────────────
    print(f"\n>>> CPU running {cpu_iters} iterations — watch task manager <<<")
    t0 = time.perf_counter()
    for _ in range(cpu_iters):
        cpu_lib.fused_symmetric_warp_batch(ca, cb, outs_a, outs_b, cpdy, cpdx,
                                           N, H, W, nPY, nPX, ctrs_y, ctrs_x, 0, 1)
    cpu_total_s = time.perf_counter() - t0
    cpu_avg_ms = (cpu_total_s / cpu_iters) * 1000
    cpu_pairs_sec = (cpu_iters * N) / cpu_total_s
    print(f"  Done: {cpu_iters} iters in {cpu_total_s:.1f}s")
    print(f"  Avg: {cpu_avg_ms:.2f} ms/call  |  {cpu_pairs_sec:.1f} pairs/sec")

    print(f"\n  Pausing 5s...")
    time.sleep(5)

    # ── GPU sustained ───────────────────────────────────────────────
    print(f"\n>>> GPU running {gpu_iters} iterations -- watch task manager <<<")
    res_gpu = np.zeros(3, dtype=np.float32)
    gpu_lib.gpu_fused_warp_sustained(ca, cb, cpdy, cpdx, ctrs_y, ctrs_x,
                                      N, H, W, nPY, nPX,
                                      gpu_iters, res_gpu)
    gpu_avg_ms = res_gpu[1]
    gpu_pairs_sec = (gpu_iters * N) / (res_gpu[0] / 1000)
    print(f"  Done: {gpu_iters} iters in {res_gpu[0]/1000:.1f}s")
    print(f"  Avg: {gpu_avg_ms:.2f} ms/call  |  {gpu_pairs_sec:.1f} pairs/sec")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RESULTS: {H}x{W} ({H*W/1e6:.1f} MP), batch={N}, pred_grid={nPY}x{nPX}")
    print(f"Both: predictor bicubic upsample + symmetric bicubic image warp")
    print(f"{'='*70}")
    print(f"  CPU ({omp_threads} threads):  {cpu_avg_ms:>8.2f} ms/call  |  {cpu_pairs_sec:>8.1f} pairs/sec")
    print(f"  GPU:                 {gpu_avg_ms:>8.2f} ms/call  |  {gpu_pairs_sec:>8.1f} pairs/sec")
    print(f"  Speedup:             {cpu_avg_ms/gpu_avg_ms:>8.1f}x")
    print()


if __name__ == "__main__":
    main()
