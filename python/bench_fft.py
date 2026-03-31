"""
GPU vs CPU FFT cross-correlation benchmark.

Compares:
  - GPU: cuFFT batched (compute-only, no transfer)
  - CPU: FFTW3f via C convolve() with OpenMP (all cores)
  - CPU: scipy.fft with workers=-1 (all cores)

Usage:
    python python/bench_fft.py
    python python/bench_fft.py --windows 32,64 --counts 1024,4096 --threads 20
    python python/bench_fft.py --image-dir <path_to_tifs>  --windows 32,64 --counts 4096
    python python/bench_fft.py --csv results_fft.csv
"""

import argparse
import ctypes
import os
import sys
import time
import numpy as np
from numpy.ctypeslib import ndpointer

# ── Paths ──────────────────────────────────────────────────────────────────

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


# ── GPU library ────────────────────────────────────────────────────────────

def load_gpu_lib():
    path = os.path.join(_ROOT_DIR, f"bench_xcorr{_EXT}")
    if not os.path.isfile(path):
        print(f"ERROR: GPU library not found at {path}")
        print(f"Build it first: see build.bat or build.sh")
        sys.exit(1)
    _add_cuda_dll_dirs()
    lib = ctypes.CDLL(path)
    lib.gpu_xcorr_bench.argtypes = [
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ]
    lib.gpu_xcorr_bench.restype = ctypes.c_int
    return lib


# ── CPU C library (FFTW3f + OpenMP) ───────────────────────────────────────

def load_c_lib():
    """Load libbulkxcorr2d with exported convolve()."""
    sys.path.insert(0, _BENCH_DIR)
    from load_cpu_baseline import load_xcorr_lib
    return load_xcorr_lib()


def cpu_xcorr_c_batch(lib, windowsA, windowsB, win_h, win_w):
    """Call C convolve() for each window pair. OpenMP parallelism is INSIDE each call."""
    N = windowsA.shape[0]
    out = np.zeros((N, win_h, win_w), dtype=np.float32)
    n_arr = np.array([win_h, win_w], dtype=np.int32)
    for i in range(N):
        lib.convolve(np.ascontiguousarray(windowsA[i]),
                     np.ascontiguousarray(windowsB[i]),
                     out[i], n_arr)
    return out


# ── CPU scipy baseline ────────────────────────────────────────────────────

def cpu_xcorr_scipy_batch(windowsA, windowsB, win_h, win_w, workers):
    """Batched cross-correlation using scipy FFT with multi-threading."""
    from scipy.fft import fft2, ifft2, fftshift

    N = windowsA.shape[0]
    pad_h, pad_w = win_h * 2, win_w * 2
    oh, ow = win_h // 2, win_w // 2

    padA = np.zeros((N, pad_h, pad_w), dtype=np.float32)
    padB = np.zeros((N, pad_h, pad_w), dtype=np.float32)
    padA[:, oh:oh + win_h, ow:ow + win_w] = windowsA
    padB[:, oh:oh + win_h, ow:ow + win_w] = windowsB

    FA = fft2(padA, axes=(-2, -1), workers=workers)
    FB = fft2(padB, axes=(-2, -1), workers=workers)
    FC = FA * np.conj(FB)
    C = np.real(ifft2(FC, axes=(-2, -1), workers=workers)).astype(np.float32)
    C = fftshift(C, axes=(-2, -1))
    return C[:, oh:oh + win_h, ow:ow + win_w].copy()


# ── Image loading ─────────────────────────────────────────────────────────

def load_image_pair(image_dir):
    """Load the first A/B pair from a directory of tif images."""
    import cv2
    files = sorted(os.listdir(image_dir))
    a_file = next((f for f in files if f.endswith("_A.tif")), None)
    b_file = next((f for f in files if f.endswith("_B.tif")), None)
    if not a_file or not b_file:
        print(f"ERROR: No A/B tif pair found in {image_dir}")
        sys.exit(1)

    img_a = cv2.imread(os.path.join(image_dir, a_file), cv2.IMREAD_UNCHANGED).astype(np.float32)
    img_b = cv2.imread(os.path.join(image_dir, b_file), cv2.IMREAD_UNCHANGED).astype(np.float32)
    print(f"  Loaded: {a_file} + {b_file}  shape={img_a.shape}  ({img_a.shape[0]*img_a.shape[1]/1e6:.1f} MP)")
    return img_a, img_b


def extract_windows_from_images(img_a, img_b, win_size, n_windows):
    """Extract window pairs from real images on a regular grid with 50% overlap."""
    H, W = img_a.shape
    stride = win_size // 2
    n_y = (H - win_size) // stride + 1
    n_x = (W - win_size) // stride + 1
    available = n_y * n_x

    if available < n_windows:
        print(f"    Warning: only {available} windows available for {win_size}x{win_size} "
              f"on {H}x{W} image (requested {n_windows}). Using {available}.")
        n_windows = available

    winsA = np.zeros((n_windows, win_size, win_size), dtype=np.float32)
    winsB = np.zeros((n_windows, win_size, win_size), dtype=np.float32)
    idx = 0
    for iy in range(n_y):
        for ix in range(n_x):
            if idx >= n_windows:
                break
            y0 = iy * stride
            x0 = ix * stride
            winsA[idx] = img_a[y0:y0 + win_size, x0:x0 + win_size]
            winsB[idx] = img_b[y0:y0 + win_size, x0:x0 + win_size]
            idx += 1
        if idx >= n_windows:
            break

    return winsA[:idx], winsB[:idx], idx


# ── Correctness check ─────────────────────────────────────────────────────

def check_correctness(gpu_lib, c_lib, win_h, win_w):
    """Verify GPU output matches C library (or scipy if C unavailable)."""
    np.random.seed(42)
    A = np.random.randn(1, win_h, win_w).astype(np.float32)
    B = np.random.randn(1, win_h, win_w).astype(np.float32)

    # GPU
    gpu_out = np.zeros_like(A)
    times = np.zeros(4, dtype=np.float32)
    gpu_lib.gpu_xcorr_bench(np.ascontiguousarray(A), np.ascontiguousarray(B),
                            gpu_out, 1, win_h, win_w, times)

    # Reference
    if c_lib is not None:
        ref = cpu_xcorr_c_batch(c_lib, A, B, win_h, win_w)
        label = "FFTW"
    else:
        ref = cpu_xcorr_scipy_batch(A, B, win_h, win_w, 1)
        label = "scipy"

    max_abs_err = np.max(np.abs(gpu_out - ref))
    max_ref = np.max(np.abs(ref)) + 1e-10
    rel_err = max_abs_err / max_ref
    ok = rel_err < 1e-4
    print(f"  {win_h:3d}x{win_w:<3d}  vs {label:5s}  rel_err={rel_err:.2e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


# ── Benchmark ─────────────────────────────────────────────────────────────

def run_benchmark(gpu_lib, c_lib, window_sizes, window_counts, n_warmup, n_runs,
                  omp_threads, scipy_workers, img_a=None, img_b=None):
    results = []

    for win_size in window_sizes:
        for n_win in window_counts:
            # Get window data
            if img_a is not None:
                A, B, actual_n = extract_windows_from_images(img_a, img_b, win_size, n_win)
                n_win = actual_n
            else:
                np.random.seed(0)
                A = np.random.randn(n_win, win_size, win_size).astype(np.float32)
                B = np.random.randn(n_win, win_size, win_size).astype(np.float32)

            gpu_out = np.zeros_like(A)
            times = np.zeros(4, dtype=np.float32)

            # ── GPU (compute-only) ────────────────────────────────────
            for _ in range(n_warmup):
                gpu_lib.gpu_xcorr_bench(np.ascontiguousarray(A), np.ascontiguousarray(B),
                                        gpu_out, n_win, win_size, win_size, times)
            gpu_compute_times = []
            for _ in range(n_runs):
                gpu_lib.gpu_xcorr_bench(np.ascontiguousarray(A), np.ascontiguousarray(B),
                                        gpu_out, n_win, win_size, win_size, times)
                gpu_compute_times.append(times[1])  # compute-only

            # ── CPU FFTW (C library) ──────────────────────────────────
            c_time_ms = None
            if c_lib is not None:
                cpu_xcorr_c_batch(c_lib, A, B, win_size, win_size)  # warmup
                c_times = []
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    cpu_xcorr_c_batch(c_lib, A, B, win_size, win_size)
                    c_times.append((time.perf_counter() - t0) * 1000)
                c_time_ms = np.median(c_times)

            # ── CPU scipy ─────────────────────────────────────────────
            cpu_xcorr_scipy_batch(A, B, win_size, win_size, scipy_workers)  # warmup
            scipy_times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                cpu_xcorr_scipy_batch(A, B, win_size, win_size, scipy_workers)
                scipy_times.append((time.perf_counter() - t0) * 1000)

            gpu_ms = np.median(gpu_compute_times)
            scipy_ms = np.median(scipy_times)

            row = {
                "win_size": win_size,
                "pad_size": win_size * 2,
                "n_windows": n_win,
                "gpu_compute_ms": gpu_ms,
                "cpu_fftw_ms": c_time_ms,
                "cpu_scipy_ms": scipy_ms,
                "speedup_vs_fftw": c_time_ms / gpu_ms if c_time_ms else None,
                "speedup_vs_scipy": scipy_ms / gpu_ms,
            }
            results.append(row)

    return results


def print_results(results, omp_threads, has_fftw):
    print()
    print(f"FFT Cross-Correlation Benchmark  (compute-only, no transfer overhead)")
    print(f"CPU: {os.cpu_count()} cores, OMP_NUM_THREADS={omp_threads}")
    print("=" * 105)

    header = (f"{'Window':>8s}  {'Padded':>8s}  {'N_win':>7s}  "
              f"{'GPU':>10s}  {'FFTW(C)':>10s}  {'scipy':>10s}  "
              f"{'GPU/FFTW':>10s}  {'GPU/scipy':>10s}")
    print(header)
    print("-" * len(header))

    for r in results:
        win = f"{r['win_size']}x{r['win_size']}"
        pad = f"{r['pad_size']}x{r['pad_size']}"
        fftw_str = f"{r['cpu_fftw_ms']:.2f}ms" if r['cpu_fftw_ms'] else "N/A"
        vs_fftw = f"{r['speedup_vs_fftw']:.1f}x" if r['speedup_vs_fftw'] else "N/A"
        print(f"{win:>8s}  {pad:>8s}  {r['n_windows']:>7d}  "
              f"{r['gpu_compute_ms']:>9.2f}ms  {fftw_str:>10s}  {r['cpu_scipy_ms']:>9.2f}ms  "
              f"{vs_fftw:>10s}  {r['speedup_vs_scipy']:>9.1f}x")
    print()


def save_csv(results, path):
    import csv
    keys = results[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPU vs CPU FFT cross-correlation benchmark")
    parser.add_argument("--windows", default="16,32,64,128",
                        help="Comma-separated window sizes (default: 16,32,64,128)")
    parser.add_argument("--counts", default="256,1024,4096",
                        help="Comma-separated window counts (default: 256,1024,4096)")
    parser.add_argument("--threads", type=int, default=0,
                        help="OMP threads for C library (0=all cores, default: 0)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument("--runs", type=int, default=10, help="Timed iterations (default: 10)")
    parser.add_argument("--image-dir", default=None, help="Directory with A/B .tif pairs for real image windows")
    parser.add_argument("--csv", default=None, help="Save results to CSV file")
    args = parser.parse_args()

    window_sizes = [int(x) for x in args.windows.split(",")]
    window_counts = [int(x) for x in args.counts.split(",")]

    # Set CPU thread counts for saturation
    omp_threads = args.threads if args.threads > 0 else os.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    scipy_workers = -1  # all cores

    print(f"CPU threads: OMP_NUM_THREADS={omp_threads}, scipy workers=all")
    print(f"GPU: RTX 4060 (sm_89)")
    print()

    print("Loading GPU library...")
    gpu_lib = load_gpu_lib()

    print("Loading FFTW C library...")
    c_lib = load_c_lib()
    if c_lib is not None:
        print(f"  libbulkxcorr2d loaded — FFTW3f + OpenMP ({omp_threads} threads)")
    else:
        print("  Not available — FFTW comparison will be skipped")

    # Load real images if specified
    img_a, img_b = None, None
    if args.image_dir:
        print(f"\nLoading real images from {args.image_dir}...")
        img_a, img_b = load_image_pair(args.image_dir)

    # Correctness
    print("\nCorrectness check:")
    all_ok = True
    for ws in window_sizes:
        if not check_correctness(gpu_lib, c_lib, ws, ws):
            all_ok = False
    if not all_ok:
        print("WARNING: Correctness check failed!")
    else:
        print("All passed.\n")

    # Benchmark
    print(f"Running benchmark: {args.warmup} warmup + {args.runs} timed runs (median)")
    results = run_benchmark(gpu_lib, c_lib, window_sizes, window_counts,
                            args.warmup, args.runs, omp_threads, scipy_workers,
                            img_a, img_b)
    print_results(results, omp_threads, c_lib is not None)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()
