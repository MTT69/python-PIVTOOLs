"""
I/O decomposition benchmark: separates image reading, result writing, and Dask overhead.

The "residual I/O" in the pipeline benchmark lumps TIFF decoding, Dask scheduling,
data serialization, and OneDrive latency into one number. This script measures each
component independently so the profile report can tell a clear story.

Test 1: Image read — time tifffile.imread() for N pairs (cold + warm cache)
Test 2: Result write — 4 save mode combos (full/minimal x compressed/uncompressed)
Test 3: Dask overhead — cluster startup + empty task round-trip + scatter cost
Test 4: Compression speedup — ratio table for the profile report

Usage:
    python profile/benchmark_io_decomposition.py
    python profile/benchmark_io_decomposition.py --test 1      # Image read only
    python profile/benchmark_io_decomposition.py --test 2      # Result write only
    python profile/benchmark_io_decomposition.py --test 3      # Dask overhead only
    python profile/benchmark_io_decomposition.py --threads 10  # OMP threads for correlator
    python profile/benchmark_io_decomposition.py --pairs 20    # Number of pairs
"""

import argparse
import csv
import gc
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import yaml

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SOURCE_4MP = (
    r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
    r"\#current_processing\4000_images_channel\planar_images"
)

N_PAIRS = 20
OMP_THREADS = 10
N_ITERATIONS = 5


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def make_csv_path(test_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RESULTS_DIR, f"{test_name}_{timestamp}.csv")


def write_csv(csv_path, fieldnames, rows):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Test 1: Image read decomposition
# ---------------------------------------------------------------------------
def test_image_read():
    """Decompose TIFF reading into: file open+decode, dtype conversion, stacking.

    Sub-tests:
      1a. End-to-end: tifffile.imread() + .astype(float32) — what the pipeline does
      1b. Raw decode only: tifffile.imread() without dtype conversion
      1c. Dtype conversion only: pre-loaded uint16 → float32 copy
      1d. File open overhead: open() + close() with no read (measures syscall cost)
      1e. os.path.exists() overhead: the redundant stat() call in the reader
      1f. Threaded pair read: read A and B concurrently via ThreadPoolExecutor
      1g. np.fromfile baseline: raw binary read (theoretical floor for this file size)
      1h. Cold vs warm: first read vs OS page cache warm
      1i. cv2.imread comparison
      1j. Production path: read_pair() through the full pivtools reader stack
    """
    print("\n" + "=" * 70)
    print("TEST 1: Image read decomposition")
    print(f"        {N_PAIRS} pairs, 4MP (2048x2048), {N_ITERATIONS} iterations each")
    print("=" * 70)

    import tifffile
    from concurrent.futures import ThreadPoolExecutor

    # Build file list
    file_paths = []
    pair_paths = []  # [(path_a, path_b), ...]
    for idx in range(1, N_PAIRS + 1):
        a_path = os.path.join(SOURCE_4MP, f"B{idx:05d}_A.tif")
        b_path = os.path.join(SOURCE_4MP, f"B{idx:05d}_B.tif")
        if not os.path.isfile(a_path):
            raise FileNotFoundError(f"Missing: {a_path}")
        if not os.path.isfile(b_path):
            raise FileNotFoundError(f"Missing: {b_path}")
        file_paths.extend([a_path, b_path])
        pair_paths.append((a_path, b_path))

    n_files = len(file_paths)
    sample_size = os.path.getsize(file_paths[0])
    raw_data_size = 2048 * 2048 * 2  # uint16 = 2 bytes per pixel
    total_size = sum(os.path.getsize(p) for p in file_paths)
    pair_data_mb = (sample_size * 2) / 1e6

    print(f"  {n_files} files ({N_PAIRS} pairs x 2)")
    print(f"  File size: {sample_size / 1024:.0f} KB on disk, "
          f"{raw_data_size / 1024:.0f} KB raw pixels (uint16)")
    print(f"  TIFF compression ratio: {sample_size / raw_data_size:.2f}x "
          f"({'compressed' if sample_size < raw_data_size * 0.95 else 'uncompressed'})")

    def _time_loop(func, label, iterations=N_ITERATIONS):
        """Run func() N_ITERATIONS times, return (mean_per_pair_ms, std_per_pair_ms)."""
        times = []
        for _ in range(iterations):
            gc.collect()
            t0 = time.perf_counter()
            func()
            elapsed = time.perf_counter() - t0
            times.append(elapsed / N_PAIRS * 1000.0)
        mean = np.mean(times)
        std = np.std(times)
        print(f"    {label:<42s} {mean:7.2f} +/- {std:5.2f} ms/pair")
        return round(mean, 2), round(std, 2)

    results = {}

    # --- 1h. Cold read (before anything warms the page cache) ---
    print("\n  [1h] Cold read (first pass, page cache may be cold)...")
    gc.collect()
    t0 = time.perf_counter()
    for path in file_paths:
        img = tifffile.imread(path).astype(np.float32)
        del img
    t_cold = time.perf_counter() - t0
    cold_per_pair = t_cold / N_PAIRS * 1000.0
    print(f"    Cold read (single pass)                  {cold_per_pair:7.2f} ms/pair")
    results["cold_per_pair_ms"] = round(cold_per_pair, 2)

    # All subsequent tests are warm (page cache populated by cold read above)
    print("\n  Warm measurements (all subsequent, OS page cache populated):")

    # --- 1a. End-to-end: what the pipeline actually does ---
    def _read_e2e():
        for path in file_paths:
            img = tifffile.imread(path).astype(np.float32)
            del img

    mean, std = _time_loop(_read_e2e, "[1a] tifffile + astype(float32)")
    results["e2e_per_pair_ms"] = mean
    results["e2e_std_ms"] = std

    # --- 1b. Raw decode only (no dtype conversion) ---
    def _read_raw_decode():
        for path in file_paths:
            img = tifffile.imread(path)
            del img

    mean, std = _time_loop(_read_raw_decode, "[1b] tifffile only (native dtype)")
    results["decode_per_pair_ms"] = mean
    results["decode_std_ms"] = std

    # --- 1c. Dtype conversion only (pre-loaded data) ---
    # Pre-load one image to measure conversion cost in isolation
    sample_img = tifffile.imread(file_paths[0])
    sample_dtype = sample_img.dtype
    print(f"\n    Native dtype: {sample_dtype}, shape: {sample_img.shape}")

    # Stack N_PAIRS * 2 copies to measure per-file conversion
    preloaded = [tifffile.imread(p) for p in file_paths]

    def _dtype_convert():
        for img in preloaded:
            _ = img.astype(np.float32)

    mean, std = _time_loop(_dtype_convert, f"[1c] .astype(float32) from {sample_dtype}")
    results["dtype_per_pair_ms"] = mean
    results["dtype_std_ms"] = std

    del preloaded
    gc.collect()

    # --- 1d. File open overhead (open + close, no read) ---
    def _file_open_only():
        for path in file_paths:
            f = open(path, "rb")
            f.close()

    mean, std = _time_loop(_file_open_only, "[1d] open() + close() only")
    results["open_close_per_pair_ms"] = mean
    results["open_close_std_ms"] = std

    # --- 1e. os.path.exists() overhead ---
    def _exists_check():
        for path in file_paths:
            os.path.exists(path)

    mean, std = _time_loop(_exists_check, "[1e] os.path.exists() only")
    results["exists_per_pair_ms"] = mean
    results["exists_std_ms"] = std

    # --- 1f. Threaded pair read (A and B concurrently) ---
    def _read_one_file(path):
        return tifffile.imread(path).astype(np.float32)

    def _read_threaded():
        with ThreadPoolExecutor(max_workers=2) as pool:
            for a_path, b_path in pair_paths:
                fa = pool.submit(_read_one_file, a_path)
                fb = pool.submit(_read_one_file, b_path)
                img_a = fa.result()
                img_b = fb.result()
                del img_a, img_b

    mean, std = _time_loop(_read_threaded, "[1f] Threaded A+B (2 threads)")
    results["threaded_per_pair_ms"] = mean
    results["threaded_std_ms"] = std

    # --- 1g. np.fromfile baseline (theoretical floor for this data size) ---
    # Write a raw binary file to measure pure sequential read bandwidth
    raw_tmpdir = tempfile.mkdtemp(prefix="piv_raw_bench_")
    raw_path = os.path.join(raw_tmpdir, "raw_pair.bin")
    raw_pair = np.zeros((2, 2048, 2048), dtype=np.float32)
    raw_pair.tofile(raw_path)
    raw_file_size = os.path.getsize(raw_path)

    # Create N_PAIRS copies to match the file-open-per-pair pattern
    raw_paths = []
    for i in range(N_PAIRS):
        p = os.path.join(raw_tmpdir, f"pair_{i:04d}.bin")
        if i == 0:
            pass  # already written as raw_path, just rename
        raw_pair.tofile(p)
        raw_paths.append(p)

    def _read_np_fromfile():
        for p in raw_paths:
            data = np.fromfile(p, dtype=np.float32).reshape(2, 2048, 2048)
            del data

    mean, std = _time_loop(_read_np_fromfile, f"[1g] np.fromfile (raw float32, {raw_file_size/1e6:.0f}MB)")
    results["fromfile_per_pair_ms"] = mean
    results["fromfile_std_ms"] = std

    # Single-file memmap (no file-open-per-pair overhead)
    big_raw_path = os.path.join(raw_tmpdir, "all_pairs.bin")
    all_raw = np.zeros((N_PAIRS, 2, 2048, 2048), dtype=np.float32)
    all_raw.tofile(big_raw_path)
    big_raw_size_gb = os.path.getsize(big_raw_path) / 1e9

    def _read_memmap():
        data = np.memmap(big_raw_path, dtype=np.float32, mode='r',
                         shape=(N_PAIRS, 2, 2048, 2048))
        # Force a full read to measure actual I/O (memmap is lazy)
        _ = data.sum()
        del data

    mean, std = _time_loop(_read_memmap, f"[1g2] np.memmap full read ({big_raw_size_gb:.1f}GB)")
    results["memmap_per_pair_ms"] = mean
    results["memmap_std_ms"] = std

    import shutil
    shutil.rmtree(raw_tmpdir, ignore_errors=True)
    del all_raw, raw_pair
    gc.collect()

    # --- 1i. cv2.imread comparison ---
    def _read_cv2():
        for path in file_paths:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            del img

    mean, std = _time_loop(_read_cv2, "[1i] cv2.imread + astype(float32)")
    results["cv2_per_pair_ms"] = mean
    results["cv2_std_ms"] = std

    # --- 1j. Production path: pivtools read_pair() ---
    print("\n  Production reader stack:")

    from pivtools_core.config import Config
    from pivtools_core.image_handling.load_images import read_pair
    from pathlib import Path

    # Create minimal config for standard TIFF reading
    cfg = _make_config([2048, 2048], [[64, 64]], [50], OMP_THREADS)
    camera_path = Path(SOURCE_4MP)

    def _read_production():
        for idx in range(1, N_PAIRS + 1):
            pair = read_pair(idx, camera_path, 1, cfg)
            del pair

    mean, std = _time_loop(_read_production, "[1j] pivtools read_pair() (full stack)")
    results["production_per_pair_ms"] = mean
    results["production_std_ms"] = std

    # --- Summary ---
    print(f"\n  {'─' * 70}")
    print(f"  File info:")
    print(f"    TIFF on disk:       {sample_size / 1024:,.0f} KB per file")
    print(f"    Raw pixels (uint16): {raw_data_size / 1024:,.0f} KB per file")
    print(f"    Float32 in memory:  {2048*2048*4 / 1024:,.0f} KB per file")
    print(f"    Per pair on disk:   {pair_data_mb:.1f} MB (2 files)")
    print(f"    Total dataset:      {total_size / 1e6:.0f} MB ({n_files} files)")

    e2e = results["e2e_per_pair_ms"]
    e2e_total_s = e2e * N_PAIRS / 1000.0
    bandwidth = (total_size / 1e6) / e2e_total_s if e2e_total_s > 0 else 0
    results["file_size_kb"] = round(sample_size / 1024, 0)
    results["raw_data_size_kb"] = round(raw_data_size / 1024, 0)
    results["pair_data_mb"] = round(pair_data_mb, 1)
    results["bandwidth_mbs"] = round(bandwidth, 0)

    print(f"    Effective read BW:  {bandwidth:.0f} MB/s")

    # Decomposition waterfall
    decode = results["decode_per_pair_ms"]
    dtype = results["dtype_per_pair_ms"]
    exists = results["exists_per_pair_ms"]
    open_close = results["open_close_per_pair_ms"]
    fromfile = results["fromfile_per_pair_ms"]
    prod = results["production_per_pair_ms"]

    print(f"\n  Decomposition waterfall (warm, ms/pair):")
    print(f"    File open+close overhead:     {open_close:6.2f} ms")
    print(f"    os.path.exists() overhead:    {exists:6.2f} ms")
    print(f"    TIFF decode (native dtype):   {decode:6.2f} ms")
    print(f"    + dtype conversion to f32:    {dtype:6.2f} ms")
    print(f"    = End-to-end (tifffile+f32):  {e2e:6.2f} ms")
    print(f"    Production (read_pair stack):  {prod:6.2f} ms")
    print(f"    Theoretical floor (fromfile):  {fromfile:6.2f} ms")
    overhead_vs_floor = e2e / fromfile if fromfile > 0 else float('inf')
    print(f"    Overhead vs raw binary:        {overhead_vs_floor:.1f}x")

    csv_path = make_csv_path("image_read")
    write_csv(csv_path, list(results.keys()), [results])

    return results


# ---------------------------------------------------------------------------
# Test 2: Result write timing (separate from correlation)
# ---------------------------------------------------------------------------
def test_result_write():
    """Time .mat writing for all 4 save mode combos, using real PIV results."""
    print("\n" + "=" * 70)
    print("TEST 2: Result write timing (scipy.io.savemat)")
    print(f"        4MP, 64->32, {N_PAIRS} pairs, {N_ITERATIONS} iterations per combo")
    print("=" * 70)

    from pivtools_core.config import Config
    from pivtools_cli.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU
    from pivtools_cli.piv.save_results import save_piv_result_distributed

    # Load images and correlate once to get real PIV results
    print("\n  Loading images and running correlation (one-time setup)...")
    images = _load_image_pairs(SOURCE_4MP, N_PAIRS)
    config = _make_config(
        [2048, 2048], [[64, 64], [32, 32]], [50, 50], OMP_THREADS,
        save_mode="full", save_compression=True,
    )
    correlator = InstantaneousCorrelatorCPU(config)
    correlator.correlate_batch(images, config)  # warmup
    piv_results = correlator.correlate_batch(images, config)
    print(f"  Got {len(piv_results)} results")

    save_combos = [
        ("full + compressed",      "full",    True),
        ("full + uncompressed",    "full",    False),
        ("minimal + compressed",   "minimal", True),
        ("minimal + uncompressed", "minimal", False),
    ]

    runs_to_save = config.instantaneous_runs_0based
    results = []

    for label, save_mode, save_compression in save_combos:
        print(f"\n  --- {label} ---")

        save_times_ms = []
        file_size_kb = 0.0

        for it in range(N_ITERATIONS):
            save_tmpdir = tempfile.mkdtemp(prefix="piv_io_bench_")
            t0 = time.perf_counter()
            for i, piv_result in enumerate(piv_results):
                save_piv_result_distributed(
                    piv_result, save_tmpdir, i + 1, runs_to_save,
                    save_mode=save_mode,
                    do_compression=save_compression,
                )
            t_save = time.perf_counter() - t0
            save_times_ms.append(t_save / N_PAIRS * 1000.0)

            # Measure file size on first iteration
            if it == 0:
                sample_file = os.path.join(save_tmpdir, "B00001.mat")
                if os.path.exists(sample_file):
                    file_size_kb = os.path.getsize(sample_file) / 1024.0

            shutil.rmtree(save_tmpdir, ignore_errors=True)

        mean_ms = np.mean(save_times_ms)
        std_ms = np.std(save_times_ms)

        print(f"    Per pair: {mean_ms:.2f} +/- {std_ms:.2f} ms")
        print(f"    File size: {file_size_kb:.0f} KB")

        results.append({
            "label": label,
            "save_mode": save_mode,
            "compressed": save_compression,
            "save_per_pair_ms": round(mean_ms, 2),
            "save_std_ms": round(std_ms, 2),
            "file_size_kb": round(file_size_kb, 0),
        })

    # Summary
    print("\n\n  Save I/O Summary:")
    print(f"  {'Mode':<26} {'Per pair (ms)':>14} {'File (KB)':>10} {'4000x disk':>12}")
    print("  " + "-" * 66)
    for r in results:
        disk_gb = r["file_size_kb"] * 4000 / 1024 / 1024
        print(f"  {r['label']:<26} {r['save_per_pair_ms']:>10.2f} ms {r['file_size_kb']:>8.0f} {disk_gb:>10.1f} GB")

    # Compression speedup
    print("\n  Compression speedup:")
    for mode in ("full", "minimal"):
        comp = next(r for r in results if r["save_mode"] == mode and r["compressed"])
        uncomp = next(r for r in results if r["save_mode"] == mode and not r["compressed"])
        time_ratio = comp["save_per_pair_ms"] / uncomp["save_per_pair_ms"] if uncomp["save_per_pair_ms"] > 0 else 0
        size_ratio = uncomp["file_size_kb"] / comp["file_size_kb"] if comp["file_size_kb"] > 0 else 0
        print(f"    {mode}: {time_ratio:.1f}x slower write, {size_ratio:.1f}x smaller files")

    csv_path = make_csv_path("result_write")
    write_csv(csv_path, list(results[0].keys()), results)

    return results


# ---------------------------------------------------------------------------
# Test 3: Dask overhead measurement
# ---------------------------------------------------------------------------
def test_dask_overhead():
    """Measure Dask cluster startup, round-trip task latency, and scatter cost."""
    print("\n" + "=" * 70)
    print("TEST 3: Dask overhead decomposition")
    print(f"        Cluster startup, task round-trip, scatter cost")
    print("=" * 70)

    from dask.distributed import Client, LocalCluster
    import logging as _logging
    for name in ["distributed", "distributed.worker", "distributed.scheduler",
                 "distributed.nanny", "distributed.core", "distributed.comm",
                 "tornado.application", "tornado.general"]:
        _logging.getLogger(name).setLevel(_logging.ERROR)

    worker_configs = [
        (1, 10, "1w x 10t"),
        (5, 2, "5w x 2t"),
        (10, 2, "10w x 2t"),
    ]

    results = []

    for n_workers, n_threads, label in worker_configs:
        print(f"\n  --- {label} ---")

        # Cluster startup
        t0 = time.perf_counter()
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=n_threads,
            memory_limit="4GB",
            silence_logs=50,
        )
        client = Client(cluster)
        client.wait_for_workers(n_workers, timeout=30)
        t_startup = time.perf_counter() - t0
        print(f"    Cluster startup: {t_startup:.2f}s")

        # Empty task round-trip (measures scheduling + IPC)
        n_tasks = 100
        t0 = time.perf_counter()
        futures = [client.submit(lambda: None, pure=False) for _ in range(n_tasks)]
        client.gather(futures)
        t_roundtrip = (time.perf_counter() - t0) / n_tasks * 1000.0
        print(f"    Task round-trip: {t_roundtrip:.2f} ms/task")

        # Scatter cost (broadcast config-sized object ~10KB)
        dummy_config = {"key": np.zeros(1000, dtype=np.float32)}  # ~4KB
        scatter_futures = []
        t0 = time.perf_counter()
        for _ in range(20):
            f = client.scatter(dummy_config, broadcast=True)
            scatter_futures.append(f)
        t_scatter = (time.perf_counter() - t0) / 20 * 1000.0
        print(f"    Scatter (4KB, broadcast): {t_scatter:.1f} ms")
        # Cancel all scatter futures before next test to avoid stale replication
        for f in scatter_futures:
            client.cancel(f)
        del scatter_futures

        # Scatter cost (large array ~32MB, like images)
        big_array = np.zeros((2, 2048, 2048), dtype=np.float32)  # 32MB
        scatter_futures = []
        t0 = time.perf_counter()
        for _ in range(5):
            f = client.scatter(big_array, broadcast=True)
            scatter_futures.append(f)
        t_scatter_big = (time.perf_counter() - t0) / 5 * 1000.0
        print(f"    Scatter (32MB, broadcast): {t_scatter_big:.1f} ms")
        # Cancel all scatter futures before shutdown
        for f in scatter_futures:
            client.cancel(f)
        del scatter_futures, big_array

        try:
            client.close(timeout=5)
        except Exception:
            pass
        try:
            cluster.close(timeout=5)
        except Exception:
            pass

        results.append({
            "config": label,
            "workers": n_workers,
            "threads": n_threads,
            "startup_s": round(t_startup, 2),
            "task_roundtrip_ms": round(t_roundtrip, 2),
            "scatter_small_ms": round(t_scatter, 1),
            "scatter_large_ms": round(t_scatter_big, 1),
        })
        gc.collect()
        time.sleep(1)

    # Summary
    print("\n\n  Dask Overhead Summary:")
    print(f"  {'Config':<14} {'Startup':>10} {'Task RT':>10} {'Scatter 4KB':>12} {'Scatter 32MB':>13}")
    print("  " + "-" * 63)
    for r in results:
        print(f"  {r['config']:<14} {r['startup_s']:>8.2f}s {r['task_roundtrip_ms']:>8.2f}ms {r['scatter_small_ms']:>10.1f}ms {r['scatter_large_ms']:>11.1f}ms")

    csv_path = make_csv_path("dask_overhead")
    write_csv(csv_path, list(results[0].keys()), results)

    return results


# ---------------------------------------------------------------------------
# Helpers (shared with other benchmarks)
# ---------------------------------------------------------------------------
def _load_image_pairs(source_dir, n_pairs):
    pairs = []
    for idx in range(1, n_pairs + 1):
        a = cv2.imread(os.path.join(source_dir, f"B{idx:05d}_A.tif"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        b = cv2.imread(os.path.join(source_dir, f"B{idx:05d}_B.tif"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        pairs.append(np.stack([a, b]))
    return np.stack(pairs)


def _make_config(image_shape, window_sizes, overlaps, omp_threads,
                 save_mode="full", save_compression=True):
    cfg_dict = {
        "images": {
            "shape": image_shape, "num_images": 100,
            "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
            "type": "standard", "start_index": 1, "frame_stride": 0,
            "pair_stride": 1, "pairing_preset": "ab_format",
        },
        "paths": {"source_paths": ["."], "base_paths": ["."], "camera_count": 1},
        "processing": {"backend": "cpu", "omp_threads": omp_threads},
        "instantaneous_piv": {
            "window_size": window_sizes, "overlap": overlaps,
            "peak_finder": "gauss6", "secondary_peak": False,
            "window_type": "gaussian",
            "runs": list(range(1, len(window_sizes) + 1)),
            "save_mode": save_mode, "save_compression": save_compression,
        },
        "outlier_detection": {
            "enabled": True,
            "methods": [{"type": "median_2d", "threshold": 2.0, "epsilon": 0.2}],
        },
        "infilling": {
            "mid_pass": {"enabled": True, "method": "local_median", "parameters": {"ksize": 3}},
            "final_pass": {"enabled": True, "method": "local_median", "parameters": {"ksize": 3}},
        },
    }
    tmpdir = tempfile.mkdtemp(prefix="piv_io_bench_")
    cfg_path = os.path.join(tmpdir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
    from pivtools_core.config import Config
    return Config(cfg_path)


# ---------------------------------------------------------------------------
# Test 4: Full pipeline waterfall (all components, independently measured)
# ---------------------------------------------------------------------------
def test_pipeline_waterfall(read_results=None, write_results=None):
    """Combine independently-measured components into a pipeline waterfall.

    Uses correlator profiling (same as profile_piv.py) + read/write results
    from Tests 1-2 to produce a complete per-pair time decomposition.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Full pipeline waterfall (independent measurements)")
    print(f"        4MP, 64->32->16, 10 OMP threads, {N_PAIRS} pairs")
    print("=" * 70)

    from pivtools_cli.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU

    # --- Correlator timing (isolated, same as profile_piv.py) ---
    print("\n  Running correlator benchmark (images pre-loaded)...")
    images = _load_image_pairs(SOURCE_4MP, N_PAIRS)
    config = _make_config(
        [2048, 2048], [[64, 64], [32, 32], [16, 16]], [50, 50, 50], OMP_THREADS,
        save_mode="minimal", save_compression=False,
    )
    correlator = InstantaneousCorrelatorCPU(config)
    correlator.profiling_enabled = True
    correlator.correlate_batch(images, config)  # warmup

    PC_SUB_SECTIONS = ["pc_gaussian_smooth", "pc_predictor_remap", "pc_fused_warp"]

    corr_times = []
    for _ in range(N_ITERATIONS):
        correlator.correlate_batch(images, config)
        profile = correlator.get_profile_summary()
        total = 0.0
        for pass_idx in profile:
            total += sum(v for k, v in profile[pass_idx].items() if k not in PC_SUB_SECTIONS)
        corr_times.append(total / N_PAIRS * 1000.0)

    corr_mean = np.mean(corr_times)
    corr_std = np.std(corr_times)
    print(f"    Correlator: {corr_mean:.1f} +/- {corr_std:.1f} ms/pair")

    del images, correlator
    gc.collect()

    # --- Gather read/write results ---
    if read_results is None:
        read_ms = float(input("  Enter image read time (ms/pair) from Test 1 [1a]: "))
    else:
        read_ms = read_results["e2e_per_pair_ms"]

    if write_results is None:
        save_ms = float(input("  Enter save time (ms/pair) for minimal+uncomp from Test 2: "))
    else:
        # Use minimal + uncompressed as the production default
        uncomp = next(r for r in write_results
                      if r["save_mode"] == "minimal" and not r["compressed"])
        save_ms = uncomp["save_per_pair_ms"]

    # --- Print waterfall ---
    measured_total = read_ms + corr_mean + save_ms

    # The actual pipeline total from benchmark_scaling.py (1w x 10t) was 494 ms/pair.
    # If we have it, show the residual = pipeline - (read + corr + save) = Dask overhead
    print(f"\n  Pipeline waterfall (ms/pair):")
    print(f"  {'─' * 55}")
    print(f"  {'Image read (TIFF decode + f32)':<38} {read_ms:7.1f} ms")
    print(f"  {'Correlator (3-pass compute)':<38} {corr_mean:7.1f} ms")
    print(f"  {'Result save (minimal, no compress)':<38} {save_ms:7.1f} ms")
    print(f"  {'─' * 55}")
    print(f"  {'Sum of measured components':<38} {measured_total:7.1f} ms")
    print(f"")
    print(f"  Compare against benchmark_scaling.py (1w x 10t) for full")
    print(f"  pipeline total. The gap = Dask scheduling + task overhead.")

    # As percentage of measured sum
    print(f"\n  Component shares (of measured sum):")
    for label, val in [("Image read", read_ms), ("Correlator", corr_mean), ("Result save", save_ms)]:
        pct = val / measured_total * 100 if measured_total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {label:<20s} {val:6.1f} ms  ({pct:4.1f}%)  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global N_PAIRS, OMP_THREADS, N_ITERATIONS

    parser = argparse.ArgumentParser(description="I/O decomposition benchmark")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4],
                        help="Run only a specific test (1=read, 2=write, 3=dask, 4=waterfall)")
    parser.add_argument("--threads", type=int, default=10, help="OMP threads (default: 10)")
    parser.add_argument("--pairs", type=int, default=20, help="Number of pairs (default: 20)")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations (default: 5)")
    args = parser.parse_args()

    OMP_THREADS = args.threads
    N_PAIRS = args.pairs
    N_ITERATIONS = args.iterations

    print(f"\nI/O Decomposition Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Source: {SOURCE_4MP}")
    print(f"  Pairs: {N_PAIRS}, Threads: {OMP_THREADS}, Iterations: {N_ITERATIONS}")

    read_results = None
    write_results = None

    if args.test is None or args.test == 1:
        read_results = test_image_read()

    if args.test is None or args.test == 2:
        write_results = test_result_write()

    if args.test is None or args.test == 3:
        test_dask_overhead()

    if args.test is None or args.test == 4:
        test_pipeline_waterfall(read_results, write_results)

    print("\n" + "=" * 70)
    print("I/O DECOMPOSITION BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
