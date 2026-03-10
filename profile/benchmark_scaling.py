"""
Worker x Thread scaling benchmark for instantaneous PIV pipeline.

Runs the full Dask pipeline (cluster -> load -> scatter -> correlate -> save)
with varying worker/thread combinations on 100 4MP image pairs, batch_size=10.

Four sweeps:
  1. Thread scaling:    1 worker, threads = 1,2,4,8,10,16,20
  2. Worker scaling:    4 threads/worker, workers = 1,2,4,5
  3. Combined matrix:   workers x threads where workers*threads <= 20
  4. Oversubscription:  workers x threads > 20 (1.5x, 2x, 4-5x)

Results saved to CSV row-by-row (crash-safe). Auto-generates matplotlib plots.

Usage:
    python profile/benchmark_scaling.py --dry-run                  # Preview test matrix
    python profile/benchmark_scaling.py --source "C:\\path\\images"  # Run all sweeps
    python profile/benchmark_scaling.py --sweep threads            # Thread scaling only
    python profile/benchmark_scaling.py --sweep oversub            # Oversubscription only
    python profile/benchmark_scaling.py --resume results.csv       # Resume after crash
    python profile/benchmark_scaling.py --plots-only results.csv   # Re-generate plots
    python profile/benchmark_scaling.py --iterations 3             # Error bars
"""

import argparse
import csv
import gc
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Configuration — edit these for your machine
# ---------------------------------------------------------------------------
TOTAL_CORES = 20
N_PAIRS = 100
BATCH_SIZE = 10
N_ITERATIONS = 1
IMAGE_SHAPE = [2048, 2048]
WINDOW_SIZES = [[64, 64], [32, 32]]
OVERLAPS = [50, 50]

SOURCE_DIR = (
    r"C:\path\to\your\4mp\images"  # UPDATE THIS or use --source flag
)


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------
def get_thread_sweep():
    """1 worker, vary threads: pure thread scaling."""
    return [
        {"workers": 1, "threads": t, "label": "thread_sweep"}
        for t in [1, 2, 4, 8, 10, 16, 20]
    ]


def get_worker_sweep():
    """Fixed 4 threads/worker, vary workers: pure worker scaling."""
    return [
        {"workers": w, "threads": 4, "label": "worker_sweep"}
        for w in [1, 2, 4, 5]
    ]


def get_matrix_sweep():
    """All valid w x t combos where w*t <= TOTAL_CORES."""
    worker_options = [1, 2, 4, 5, 10, 20]
    thread_options = [1, 2, 4, 5, 10, 20]
    configs = []
    seen = set()
    for w in worker_options:
        for t in thread_options:
            if w * t <= TOTAL_CORES and (w, t) not in seen:
                seen.add((w, t))
                configs.append({"workers": w, "threads": t, "label": "matrix"})
    configs.sort(key=lambda c: (c["workers"] * c["threads"], c["workers"]))
    return configs


def get_oversub_sweep():
    """Oversubscribed configs where w*t > TOTAL_CORES."""
    return [
        # 1.5x oversubscription (30 logical on 20 physical)
        {"workers": 5, "threads": 6, "label": "oversub_1.5x"},
        {"workers": 10, "threads": 3, "label": "oversub_1.5x"},
        {"workers": 2, "threads": 15, "label": "oversub_1.5x"},
        # 2x oversubscription (40 logical)
        {"workers": 4, "threads": 10, "label": "oversub_2x"},
        {"workers": 10, "threads": 4, "label": "oversub_2x"},
        {"workers": 20, "threads": 2, "label": "oversub_2x"},
        {"workers": 2, "threads": 20, "label": "oversub_2x"},
        # Heavy oversubscription (80-100 logical)
        {"workers": 10, "threads": 10, "label": "oversub_5x"},
        {"workers": 20, "threads": 4, "label": "oversub_4x"},
    ]


def build_config_list(sweep):
    """Build deduplicated config list for the requested sweep(s)."""
    configs = []
    if sweep in ("threads", "all"):
        configs.extend(get_thread_sweep())
    if sweep in ("workers", "all"):
        configs.extend(get_worker_sweep())
    if sweep in ("matrix", "all"):
        configs.extend(get_matrix_sweep())
    if sweep in ("oversub", "all"):
        configs.extend(get_oversub_sweep())

    # Deduplicate by (workers, threads), keep first label
    seen = set()
    deduped = []
    for cfg in configs:
        key = (cfg["workers"], cfg["threads"])
        if key not in seen:
            seen.add(key)
            deduped.append(cfg)
    return deduped


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "workers", "threads", "total_cores", "oversub_ratio",
    "label", "iteration",
    "n_pairs", "batch_size", "n_saved",
    "cluster_startup_s", "load_s", "scatter_s", "correlate_s", "pipeline_total_s",
    "per_pair_ms", "pairs_per_s",
    "valid", "error",
]


def write_csv_header(csv_path):
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_row(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore").writerow(row)


def load_csv_results(csv_path):
    results = []
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            for key in row:
                if key in ("error", "label", "valid"):
                    continue
                try:
                    row[key] = float(row[key]) if "." in str(row[key]) else int(row[key])
                except (ValueError, TypeError):
                    pass
            results.append(row)
    return results


def get_completed_keys(csv_path):
    """Return set of (workers, threads, iteration) already completed."""
    if not os.path.exists(csv_path):
        return set()
    return {
        (r["workers"], r["threads"], r["iteration"])
        for r in load_csv_results(csv_path)
    }


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def print_dry_run(configs, n_iterations):
    """Print the full test matrix without running anything."""
    total = len(configs) * n_iterations

    print(f"\n{'='*70}")
    print(f"DRY RUN: {len(configs)} unique configs x {n_iterations} iterations = {total} runs")
    print(f"  {N_PAIRS} pairs, batch_size={BATCH_SIZE}, {IMAGE_SHAPE[0]}x{IMAGE_SHAPE[1]}")
    print(f"{'='*70}")

    print(f"\n{'#':>4} {'Workers':>8} {'Threads':>8} {'Cores':>6} {'Oversub':>8} {'Label':<16}")
    print("-" * 56)
    for i, cfg in enumerate(configs, 1):
        w, t = cfg["workers"], cfg["threads"]
        cores = w * t
        oversub = f"{cores/TOTAL_CORES:.1f}x" if cores > TOTAL_CORES else "-"
        print(f"{i:>4} {w:>8} {t:>8} {cores:>6} {oversub:>8} {cfg['label']:<16}")

    # Time estimate
    # Rough model: fastest ~20s, slowest (1w1t) ~160s, average ~45s
    est_avg = 45
    est_total = total * est_avg
    print(f"\nEstimated total time: ~{est_total/60:.0f} minutes ({est_total/3600:.1f} hours)")
    print(f"  (based on ~{est_avg}s average per run)")


# ---------------------------------------------------------------------------
# Config file builder
# ---------------------------------------------------------------------------
def make_config_yaml(tmpdir, workers, threads):
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir, exist_ok=True)

    cfg_dict = {
        "paths": {
            "source_paths": [SOURCE_DIR],
            "base_paths": [output_dir],
            "camera_count": 1,
            "camera_subfolders": False,
            "active_paths": [0],
        },
        "images": {
            "shape": IMAGE_SHAPE,
            "num_images": N_PAIRS * 2,
            "format": ["B%05d_A.tif", "B%05d_B.tif"],
            "type": "standard",
            "start_index": 1,
            "frame_stride": 0,
            "pair_stride": 1,
            "pairing_preset": "ab_format",
        },
        "batches": {"batch_size": BATCH_SIZE},
        "processing": {
            "backend": "cpu",
            "omp_threads": threads,
            "dask_workers_per_node": workers,
            "dask_memory_limit": f"{max(2, 16 // workers)}GB",
            "dask_max_in_flight_per_worker": 3,
            "cluster_type": "local",
            "open_dashboard": False,
        },
        "instantaneous_piv": {
            "window_size": WINDOW_SIZES,
            "overlap": OVERLAPS,
            "peak_finder": "gauss6",
            "secondary_peak": False,
            "window_type": "gaussian",
            "runs": list(range(1, len(WINDOW_SIZES) + 1)),
            "predictor_smoothing": True,
            "image_warp_interpolation": "cubic",
        },
        "outlier_detection": {
            "enabled": True,
            "methods": [{"type": "median_2d", "threshold": 2.0, "epsilon": 0.2}],
        },
        "infilling": {
            "mid_pass": {"enabled": True, "method": "local_median", "parameters": {"ksize": 3}},
            "final_pass": {"enabled": True, "method": "local_median", "parameters": {"ksize": 3}},
        },
        "masking": {"enabled": False},
        "filters": [],
        "logging": {"level": "WARNING"},
    }

    cfg_path = os.path.join(tmpdir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
    return cfg_path


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def _quiet_shutdown(client, cluster):
    import logging as _logging
    for name in ["distributed", "distributed.worker", "distributed.scheduler",
                 "distributed.nanny", "distributed.core", "distributed.comm",
                 "tornado.application", "tornado.general"]:
        _logging.getLogger(name).setLevel(_logging.CRITICAL)
    try:
        client.close(timeout=10)
    except Exception:
        pass
    time.sleep(0.5)
    try:
        cluster.close(timeout=10)
    except Exception:
        pass
    time.sleep(0.5)


def run_warmup(source_dir):
    """Run a tiny 2-pair job to prime FFTW wisdom cache. Only needed once."""
    print("Warmup: priming FFTW wisdom cache...", flush=True)
    tmpdir = tempfile.mkdtemp(prefix="piv_warmup_")
    try:
        import cv2
        import numpy as np
        from pivtools_core.config import Config
        from pivtools_cli.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU

        # Load 2 image pairs
        pairs = []
        for idx in [1, 2]:
            a = cv2.imread(os.path.join(source_dir, f"B{idx:05d}_A.tif"), cv2.IMREAD_UNCHANGED).astype(np.float32)
            b = cv2.imread(os.path.join(source_dir, f"B{idx:05d}_B.tif"), cv2.IMREAD_UNCHANGED).astype(np.float32)
            pairs.append(np.stack([a, b]))
        images = np.stack(pairs)

        cfg_dict = {
            "images": {"shape": IMAGE_SHAPE, "num_images": 4, "format": ["B%05d_A.tif", "B%05d_B.tif"],
                        "type": "standard", "start_index": 1, "frame_stride": 0, "pair_stride": 1,
                        "pairing_preset": "ab_format"},
            "paths": {"source_paths": ["."], "base_paths": ["."], "camera_count": 1},
            "processing": {"backend": "cpu", "omp_threads": 4},
            "instantaneous_piv": {"window_size": WINDOW_SIZES, "overlap": OVERLAPS,
                                   "peak_finder": "gauss6", "secondary_peak": False,
                                   "window_type": "gaussian",
                                   "runs": list(range(1, len(WINDOW_SIZES) + 1))},
            "outlier_detection": {"enabled": True,
                                   "methods": [{"type": "median_2d", "threshold": 2.0, "epsilon": 0.2}]},
            "infilling": {"mid_pass": {"enabled": True, "method": "local_median", "parameters": {"ksize": 3}},
                          "final_pass": {"enabled": True, "method": "local_median", "parameters": {"ksize": 3}}},
        }
        cfg_path = os.path.join(tmpdir, "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)
        config = Config(cfg_path)

        correlator = InstantaneousCorrelatorCPU(config)
        correlator.correlate_batch(images, config)
        print("  FFTW wisdom cached. Warmup complete.\n", flush=True)
    except Exception as e:
        print(f"  Warmup failed (non-fatal): {e}\n", flush=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()


def _run_sliding_window(
    client, images, num_chunks, max_in_flight,
    scattered_config, scattered, output_path,
    runs_0based, vector_format,
):
    """
    Sliding window correlation, inlined to avoid importing instantaneous.py
    (which has module-level side effects: Config(), signal handlers, OMP_NUM_THREADS).
    """
    from dask.distributed import as_completed
    from pivtools_cli.processing.dask_pipeline import correlate_and_save_batch

    pending = {}
    next_to_submit = 0
    all_saved_paths = []

    # Fill initial window
    while next_to_submit < min(max_in_flight, num_chunks):
        chunk_start = sum(images.chunks[0][:next_to_submit])
        filter_future = client.compute(images.blocks[next_to_submit])
        corr_future = client.submit(
            correlate_and_save_batch,
            filter_future, chunk_start + 1,
            scattered_config, scattered["cache"], scattered["masks"],
            output_path, runs_0based, vector_format,
            pure=False,
        )
        pending[corr_future] = next_to_submit
        next_to_submit += 1

    # Drain as completed, refill to keep window full
    ac = as_completed(list(pending.keys()))
    for completed in ac:
        all_saved_paths.extend(completed.result())
        del pending[completed]

        if next_to_submit < num_chunks:
            chunk_start = sum(images.chunks[0][:next_to_submit])
            filter_future = client.compute(images.blocks[next_to_submit])
            corr_future = client.submit(
                correlate_and_save_batch,
                filter_future, chunk_start + 1,
                scattered_config, scattered["cache"], scattered["masks"],
                output_path, runs_0based, vector_format,
                pure=False,
            )
            pending[corr_future] = next_to_submit
            ac.add(corr_future)
            next_to_submit += 1

    return all_saved_paths


def run_single_benchmark(workers, threads, label, iteration=0):
    """Run the full Dask pipeline. Returns a result dict (always, even on failure)."""
    tmpdir = tempfile.mkdtemp(prefix=f"piv_scale_w{workers}_t{threads}_")
    total_cores = workers * threads
    oversub_ratio = round(total_cores / TOTAL_CORES, 2)

    base_result = {
        "workers": workers,
        "threads": threads,
        "total_cores": total_cores,
        "oversub_ratio": oversub_ratio,
        "label": label,
        "iteration": iteration,
        "n_pairs": N_PAIRS,
        "batch_size": BATCH_SIZE,
        "n_saved": 0,
        "cluster_startup_s": 0, "load_s": 0, "scatter_s": 0,
        "correlate_s": 0, "pipeline_total_s": 0,
        "per_pair_ms": 0, "pairs_per_s": 0,
        "valid": "false", "error": "",
    }

    try:
        cfg_path = make_config_yaml(tmpdir, workers, threads)

        from pivtools_core.config import Config
        os.environ["OMP_NUM_THREADS"] = str(threads)
        config = Config(cfg_path)

        from pivtools_cli.piv_cluster.cluster import start_cluster

        t_cluster_start = time.perf_counter()
        cluster, client = start_cluster(config=config, worker_omp_threads=str(threads))
        cluster_startup = time.perf_counter() - t_cluster_start

        try:
            # Import only from modules WITHOUT module-level side effects.
            # DO NOT import from pivtools_core.instantaneous — it runs Config()
            # and registers signal handlers at import time.
            from dask.distributed import as_completed
            from pivtools_core.image_handling.load_images import load_images
            from pivtools_cli.piv.save_results import get_output_path
            from pivtools_cli.processing.dask_pipeline import (
                create_filter_pipeline, scatter_immutable_data,
                correlate_and_save_batch,
            )

            source_path = Path(SOURCE_DIR)
            output_path = get_output_path(
                config, 1, use_uncalibrated=True, base_path_idx=0, piv_type="instantaneous",
            )

            t_pipeline_start = time.perf_counter()

            # Load (lazy)
            t0 = time.perf_counter()
            images = load_images(1, config, source=source_path, batch_size=BATCH_SIZE)
            t_load = time.perf_counter() - t0

            # Scatter
            t0 = time.perf_counter()
            scattered = scatter_immutable_data(client, config, None, None, ensemble=False)
            t_scatter = time.perf_counter() - t0

            # Filter pipeline (passthrough, no filters configured)
            images = create_filter_pipeline(images, config, None)
            scattered_config = client.scatter(config, broadcast=True)

            # Correlation + save via sliding window
            # (inlined from instantaneous.py to avoid its module-level side effects)
            num_chunks = len(images.chunks[0])
            num_workers = config.dask_workers_per_node
            max_in_flight = min(
                num_workers * config.dask_max_in_flight_per_worker, num_chunks,
            )
            runs_0based = config.instantaneous_runs_0based
            vector_format = config.vector_format

            t0 = time.perf_counter()
            saved_paths = _run_sliding_window(
                client, images, num_chunks, max_in_flight,
                scattered_config, scattered, output_path,
                runs_0based, vector_format,
            )
            t_correlate = time.perf_counter() - t0

            t_pipeline_total = time.perf_counter() - t_pipeline_start
            n_saved = len(saved_paths)
            per_pair_ms = (t_correlate / N_PAIRS) * 1000.0
            pairs_per_s = N_PAIRS / t_correlate if t_correlate > 0 else 0

            # Sanity check
            valid = n_saved == N_PAIRS
            error = "" if valid else f"Expected {N_PAIRS} saved, got {n_saved}"

            base_result.update({
                "n_saved": n_saved,
                "cluster_startup_s": round(cluster_startup, 2),
                "load_s": round(t_load, 3),
                "scatter_s": round(t_scatter, 3),
                "correlate_s": round(t_correlate, 3),
                "pipeline_total_s": round(t_pipeline_total, 3),
                "per_pair_ms": round(per_pair_ms, 2),
                "pairs_per_s": round(pairs_per_s, 2),
                "valid": str(valid).lower(),
                "error": error,
            })

        finally:
            _quiet_shutdown(client, cluster)

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}", flush=True)
        traceback.print_exc()
        base_result["error"] = str(e)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()

    return base_result


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------
def run_sweep(configs, csv_path, n_iterations=1):
    """Run all configs sequentially, appending to CSV after each run."""
    completed = get_completed_keys(csv_path)
    if not os.path.exists(csv_path):
        write_csv_header(csv_path)

    tasks = []
    for cfg in configs:
        for it in range(n_iterations):
            if (cfg["workers"], cfg["threads"], it) not in completed:
                tasks.append((cfg, it))

    if not tasks:
        print("All configurations already completed. Nothing to run.")
        return

    total = len(tasks)
    elapsed_times = []
    started_at = datetime.now()

    print(f"\n{'='*70}")
    print(f"SCALING BENCHMARK: {total} runs to go")
    print(f"  {N_PAIRS} pairs, batch_size={BATCH_SIZE}, {IMAGE_SHAPE[0]}x{IMAGE_SHAPE[1]}")
    print(f"  Results: {csv_path}")
    print(f"  Started: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for i, (cfg, iteration) in enumerate(tasks):
        w, t = cfg["workers"], cfg["threads"]

        # ETA from running average
        if elapsed_times:
            avg = sum(elapsed_times) / len(elapsed_times)
            remaining = (total - i) * avg
            eta = datetime.now() + timedelta(seconds=remaining)
            eta_str = f"  ETA: {eta.strftime('%H:%M:%S')} (~{remaining/60:.0f}m left)"
        else:
            eta_str = ""

        print(f"[{i+1}/{total}] {cfg['label']}: {w}w x {t}t "
              f"= {w*t} cores{eta_str}", flush=True)

        t0 = time.perf_counter()
        result = run_single_benchmark(w, t, cfg["label"], iteration)
        wall = time.perf_counter() - t0
        elapsed_times.append(wall)

        # Save immediately (crash-safe)
        append_csv_row(csv_path, result)

        if result["valid"] == "true":
            print(f"  -> {result['per_pair_ms']:.1f} ms/pair, "
                  f"{result['pairs_per_s']:.1f} pairs/s, "
                  f"correlate={result['correlate_s']:.1f}s "
                  f"(wall {wall:.0f}s)\n", flush=True)
        else:
            print(f"  -> INVALID: {result['error']} (wall {wall:.0f}s)\n", flush=True)

    total_wall = (datetime.now() - started_at).total_seconds()
    print(f"\nAll {total} runs complete in {total_wall/60:.1f} minutes.")
    print(f"Results: {csv_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def generate_plots(csv_path):
    """Generate scaling plots from CSV results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    results = [r for r in load_csv_results(csv_path) if r.get("valid") == "true"]
    if not results:
        print("No valid results to plot.")
        return

    out_dir = os.path.dirname(csv_path) or "."
    stem = os.path.splitext(os.path.basename(csv_path))[0]

    # Group by (workers, threads), aggregate iterations
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["workers"], r["threads"])].append(r)

    def agg(rows, field):
        vals = [r[field] for r in rows if isinstance(r.get(field), (int, float)) and r[field] > 0]
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)

    # ---- Plot 1: Thread scaling (workers=1) ----
    thread_data = sorted(
        [(k, v) for k, v in grouped.items() if k[0] == 1],
        key=lambda x: x[0][1],
    )

    if len(thread_data) >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Thread Scaling (1 worker, {N_PAIRS} pairs, batch={BATCH_SIZE})", fontsize=14)

        threads = [k[1] for k, _ in thread_data]
        tp_mean = [agg(v, "pairs_per_s")[0] for _, v in thread_data]
        tp_std = [agg(v, "pairs_per_s")[1] for _, v in thread_data]

        ax1.errorbar(threads, tp_mean, yerr=tp_std, marker="o", linewidth=2, capsize=4)
        if tp_mean[0] > 0:
            ax1.plot(threads, [tp_mean[0] * t for t in threads],
                     "--", color="gray", alpha=0.5, label="Linear scaling")
        # Mark physical core limit
        ax1.axvline(x=TOTAL_CORES, color="red", linestyle=":", alpha=0.4, label=f"{TOTAL_CORES} cores")
        ax1.legend()
        ax1.set_xlabel("OMP Threads")
        ax1.set_ylabel("Pairs / second")
        ax1.set_title("Throughput")
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(threads)

        if tp_mean[0] > 0:
            eff = [t / (tp_mean[0] * n) * 100 for t, n in zip(tp_mean, threads)]
            colors = ["tab:red" if th > TOTAL_CORES else "steelblue" for th in threads]
            ax2.bar(range(len(threads)), eff, tick_label=[str(t) for t in threads], color=colors)
            ax2.set_ylabel("Parallel efficiency (%)")
            ax2.set_xlabel("OMP Threads")
            ax2.set_title("Scaling Efficiency")
            ax2.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
            ax2.set_ylim(0, max(120, max(eff) + 10))
            ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = os.path.join(out_dir, f"{stem}_thread_scaling.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    # ---- Plot 2: Worker scaling (one plot per unique thread count with 2+ data points) ----
    thread_groups = defaultdict(list)
    for (w, t), v in grouped.items():
        thread_groups[t].append(((w, t), v))

    for fixed_t, data in sorted(thread_groups.items()):
        if len(data) < 2:
            continue

        data.sort(key=lambda x: x[0][0])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Worker Scaling ({fixed_t} threads/worker, {N_PAIRS} pairs, batch={BATCH_SIZE})",
                     fontsize=14)

        workers = [k[0] for k, _ in data]
        tp_mean = [agg(v, "pairs_per_s")[0] for _, v in data]
        tp_std = [agg(v, "pairs_per_s")[1] for _, v in data]

        ax1.errorbar(workers, tp_mean, yerr=tp_std, marker="s", linewidth=2, capsize=4, color="tab:orange")
        if tp_mean[0] > 0:
            ax1.plot(workers, [tp_mean[0] * w for w in workers],
                     "--", color="gray", alpha=0.5, label="Linear scaling")
        # Mark where oversubscription begins
        oversub_boundary = TOTAL_CORES / fixed_t
        if oversub_boundary < max(workers):
            ax1.axvline(x=oversub_boundary, color="red", linestyle=":", alpha=0.4,
                        label=f"Oversub boundary ({oversub_boundary:.0f}w)")
        ax1.legend()
        ax1.set_xlabel("Dask Workers")
        ax1.set_ylabel("Pairs / second")
        ax1.set_title("Throughput")
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(workers)

        if tp_mean[0] > 0:
            eff = [t / (tp_mean[0] * w) * 100 for t, w in zip(tp_mean, workers)]
            colors = ["tab:red" if w * fixed_t > TOTAL_CORES else "tab:orange" for w in workers]
            ax2.bar(range(len(workers)), eff, tick_label=[str(w) for w in workers], color=colors)
            ax2.set_ylabel("Parallel efficiency (%)")
            ax2.set_xlabel("Dask Workers")
            ax2.set_title("Scaling Efficiency")
            ax2.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
            ax2.set_ylim(0, max(120, max(eff) + 10))
            ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = os.path.join(out_dir, f"{stem}_worker_scaling_t{fixed_t}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    # ---- Plot 3: Heatmap (workers x threads -> throughput) ----
    if len(grouped) >= 4:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"Worker x Thread Scaling ({N_PAIRS} pairs, batch={BATCH_SIZE})", fontsize=14)

        all_w = sorted(set(k[0] for k in grouped))
        all_t = sorted(set(k[1] for k in grouped))

        tp_grid = np.full((len(all_w), len(all_t)), np.nan)
        pp_grid = np.full((len(all_w), len(all_t)), np.nan)

        for (w, t), v in grouped.items():
            wi, ti = all_w.index(w), all_t.index(t)
            tp_grid[wi, ti] = agg(v, "pairs_per_s")[0]
            pp_grid[wi, ti] = agg(v, "per_pair_ms")[0]

        # Throughput heatmap
        im1 = ax1.imshow(tp_grid, aspect="auto", cmap="YlOrRd", origin="lower", interpolation="nearest")
        ax1.set_xticks(range(len(all_t)))
        ax1.set_xticklabels(all_t)
        ax1.set_yticks(range(len(all_w)))
        ax1.set_yticklabels(all_w)
        ax1.set_xlabel("OMP Threads")
        ax1.set_ylabel("Dask Workers")
        ax1.set_title("Throughput (pairs/s)")
        plt.colorbar(im1, ax=ax1)
        for wi in range(len(all_w)):
            for ti in range(len(all_t)):
                val = tp_grid[wi, ti]
                if not np.isnan(val):
                    cores = all_w[wi] * all_t[ti]
                    oversub = "*" if cores > TOTAL_CORES else ""
                    ax1.text(ti, wi, f"{val:.0f}\n{cores}c{oversub}",
                             ha="center", va="center", fontsize=7,
                             color="white" if val > np.nanmax(tp_grid) * 0.6 else "black")

        # Per-pair heatmap
        im2 = ax2.imshow(pp_grid, aspect="auto", cmap="YlOrRd_r", origin="lower", interpolation="nearest")
        ax2.set_xticks(range(len(all_t)))
        ax2.set_xticklabels(all_t)
        ax2.set_yticks(range(len(all_w)))
        ax2.set_yticklabels(all_w)
        ax2.set_xlabel("OMP Threads")
        ax2.set_ylabel("Dask Workers")
        ax2.set_title("Time per pair (ms)")
        plt.colorbar(im2, ax=ax2)
        for wi in range(len(all_w)):
            for ti in range(len(all_t)):
                val = pp_grid[wi, ti]
                if not np.isnan(val):
                    cores = all_w[wi] * all_t[ti]
                    oversub = "*" if cores > TOTAL_CORES else ""
                    ax2.text(ti, wi, f"{val:.0f}\n{cores}c{oversub}",
                             ha="center", va="center", fontsize=7,
                             color="white" if val < np.nanmin(pp_grid) * 1.5 else "black")

        plt.tight_layout()
        path = os.path.join(out_dir, f"{stem}_scaling_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    # ---- Plot 4: Total cores vs throughput (scatter, all configs) ----
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(f"Throughput vs Total Cores ({N_PAIRS} pairs, batch={BATCH_SIZE})", fontsize=14)

    # Color by whether oversubscribed
    for (w, t), v in sorted(grouped.items()):
        tp_mean, tp_std = agg(v, "pairs_per_s")
        if tp_mean <= 0:
            continue
        cores = w * t
        color = "tab:red" if cores > TOTAL_CORES else "steelblue"
        ax.errorbar(cores, tp_mean, yerr=tp_std, marker="o", capsize=3,
                    color=color, markersize=8, zorder=3)
        ax.annotate(f"{w}w x {t}t", (cores, tp_mean),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Ideal scaling from (1,1) baseline
    baseline = grouped.get((1, 1))
    if baseline:
        base_tp = agg(baseline, "pairs_per_s")[0]
        if base_tp > 0:
            max_cores = max(k[0] * k[1] for k in grouped)
            x = np.arange(1, max_cores + 1)
            ax.plot(x, base_tp * x, "--", color="gray", alpha=0.5, label="Linear scaling")

    # Vertical line at physical core count
    ax.axvline(x=TOTAL_CORES, color="red", linestyle=":", alpha=0.6,
               label=f"{TOTAL_CORES} physical cores")
    ax.legend()
    ax.set_xlabel("Total logical cores (workers x threads)")
    ax.set_ylabel("Pairs / second")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{stem}_cores_vs_throughput.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ---- Summary table (sorted by throughput, best first) ----
    print(f"\n{'Workers':>8} {'Threads':>8} {'Cores':>6} {'Oversub':>8} "
          f"{'pairs/s':>10} {'ms/pair':>10} {'corr_s':>8} {'Label':<16}")
    print("-" * 86)
    ranked = sorted(grouped.items(), key=lambda x: -agg(x[1], "pairs_per_s")[0])
    for (w, t), v in ranked:
        tp_m, _ = agg(v, "pairs_per_s")
        pp_m, _ = agg(v, "per_pair_ms")
        cr_m, _ = agg(v, "correlate_s")
        cores = w * t
        oversub = f"{cores/TOTAL_CORES:.1f}x" if cores > TOTAL_CORES else "-"
        label = v[0].get("label", "")
        print(f"{w:>8} {t:>8} {cores:>6} {oversub:>8} "
              f"{tp_m:>10.1f} {pp_m:>10.1f} {cr_m:>8.1f} {label:<16}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Worker x Thread scaling benchmark for instantaneous PIV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sweep", choices=["threads", "workers", "matrix", "oversub", "all"],
                        default="all", help="Which sweep(s) to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print test matrix and time estimate without running")
    parser.add_argument("--resume", metavar="CSV",
                        help="Resume from existing CSV (skips completed configs)")
    parser.add_argument("--plots-only", metavar="CSV",
                        help="Re-generate plots from an existing CSV file")
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS,
                        help=f"Iterations per config for error bars (default: {N_ITERATIONS})")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to directory containing B00001_A.tif etc.")
    args = parser.parse_args()

    if args.source:
        global SOURCE_DIR
        SOURCE_DIR = args.source

    # --- Plots only ---
    if args.plots_only:
        print("Generating plots from existing results...")
        generate_plots(args.plots_only)
        return

    # --- Build config list ---
    configs = build_config_list(args.sweep)

    # --- Dry run ---
    if args.dry_run:
        print_dry_run(configs, args.iterations)
        return

    # --- Validate source ---
    if not os.path.isdir(SOURCE_DIR):
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        print("Use --source <path> or edit SOURCE_DIR in the script")
        sys.exit(1)

    first_file = os.path.join(SOURCE_DIR, "B00001_A.tif")
    last_file = os.path.join(SOURCE_DIR, f"B{N_PAIRS:05d}_A.tif")
    if not os.path.exists(first_file):
        print(f"ERROR: First image not found: {first_file}")
        print(f"Need B00001_A.tif through B{N_PAIRS:05d}_A.tif in {SOURCE_DIR}")
        sys.exit(1)
    if not os.path.exists(last_file):
        print(f"ERROR: Last image not found: {last_file}")
        print(f"Need {N_PAIRS} image pairs but source directory has fewer")
        sys.exit(1)

    # --- CSV path ---
    if args.resume:
        csv_path = args.resume
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(os.path.dirname(__file__), f"scaling_{timestamp}.csv")

    print(f"Source:     {SOURCE_DIR}")
    print(f"CSV:       {csv_path}")
    print(f"Configs:   {len(configs)} unique x {args.iterations} iter = {len(configs) * args.iterations} runs")

    # --- Warmup (prime FFTW wisdom so first real run isn't penalised) ---
    run_warmup(SOURCE_DIR)

    # --- Run ---
    run_sweep(configs, csv_path, n_iterations=args.iterations)

    # --- Plots ---
    print("\nGenerating plots...")
    generate_plots(csv_path)
    print("\nDone!")


if __name__ == "__main__":
    main()
