"""
Worker × Thread scaling benchmark for instantaneous PIV pipeline.

Runs the full Dask pipeline (cluster → load → scatter → correlate → save)
with varying worker/thread combinations on 100 4MP image pairs, batch_size=10.

Three sweeps:
  1. Thread scaling:  1 worker, threads = 1,2,4,8,10,16,20
  2. Worker scaling:  4 threads/worker, workers = 1,2,4,5,8,10
  3. Combined matrix: workers × threads where workers*threads <= 20

Results saved to CSV + auto-generates matplotlib scaling plots.

Usage:
    python profile/benchmark_scaling.py                    # Run all sweeps
    python profile/benchmark_scaling.py --sweep threads    # Thread scaling only
    python profile/benchmark_scaling.py --sweep workers    # Worker scaling only
    python profile/benchmark_scaling.py --sweep matrix     # Combined matrix only
    python profile/benchmark_scaling.py --resume results.csv  # Resume from partial run
    python profile/benchmark_scaling.py --plots-only results.csv  # Re-generate plots
    python profile/benchmark_scaling.py --iterations 3     # Multiple iterations per config
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
# Configuration
# ---------------------------------------------------------------------------
TOTAL_CORES = 20
N_PAIRS = 100
BATCH_SIZE = 10
N_ITERATIONS = 1  # Default; override with --iterations
IMAGE_SHAPE = [2048, 2048]
WINDOW_SIZES = [[64, 64], [32, 32]]
OVERLAPS = [50, 50]

# Source path — update this for your machine
SOURCE_DIR = (
    r"C:\path\to\your\4mp\images"  # Windows path — UPDATE THIS
)


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------
def get_thread_sweep():
    """Sweep 1: 1 worker, vary threads."""
    configs = []
    for threads in [1, 2, 4, 8, 10, 16, 20]:
        configs.append({"workers": 1, "threads": threads, "label": "thread_sweep"})
    return configs


def get_worker_sweep():
    """Sweep 2: fixed 4 threads/worker, vary workers."""
    configs = []
    for workers in [1, 2, 4, 5]:
        configs.append({"workers": workers, "threads": 4, "label": "worker_sweep"})
    return configs


def get_matrix_sweep():
    """Sweep 3: all valid worker × thread combos where w*t <= TOTAL_CORES."""
    worker_options = [1, 2, 4, 5, 10, 20]
    thread_options = [1, 2, 4, 5, 10, 20]
    configs = []
    seen = set()
    for w in worker_options:
        for t in thread_options:
            if w * t <= TOTAL_CORES and (w, t) not in seen:
                seen.add((w, t))
                configs.append({"workers": w, "threads": t, "label": "matrix"})
    # Sort by total cores used (ascending), then workers
    configs.sort(key=lambda c: (c["workers"] * c["threads"], c["workers"]))
    return configs


def get_oversub_sweep():
    """Sweep 4: oversubscribed configs where w*t > TOTAL_CORES."""
    configs = [
        # 1.5x oversubscription (30 logical cores on 20 physical)
        {"workers": 5, "threads": 6, "label": "oversub_1.5x"},
        {"workers": 10, "threads": 3, "label": "oversub_1.5x"},
        {"workers": 2, "threads": 15, "label": "oversub_1.5x"},
        # 2x oversubscription (40 logical cores)
        {"workers": 4, "threads": 10, "label": "oversub_2x"},
        {"workers": 10, "threads": 4, "label": "oversub_2x"},
        {"workers": 20, "threads": 2, "label": "oversub_2x"},
        {"workers": 2, "threads": 20, "label": "oversub_2x"},
        # Heavy oversubscription (60-100 logical cores)
        {"workers": 10, "threads": 10, "label": "oversub_5x"},
        {"workers": 20, "threads": 4, "label": "oversub_4x"},
    ]
    return configs


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def make_config_yaml(tmpdir, workers, threads):
    """Create a config.yaml for this benchmark run."""
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
        "batches": {
            "batch_size": BATCH_SIZE,
        },
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


def run_single_benchmark(workers, threads, iteration=0):
    """
    Run the full instantaneous PIV pipeline with given worker/thread config.

    Returns dict with timing results, or None on failure.
    """
    tmpdir = tempfile.mkdtemp(prefix=f"piv_scale_w{workers}_t{threads}_")

    try:
        cfg_path = make_config_yaml(tmpdir, workers, threads)

        # Import here to avoid top-level side effects (Config() at module level)
        from pivtools_core.config import Config

        # Set OMP_NUM_THREADS before any C library loads
        os.environ["OMP_NUM_THREADS"] = str(threads)

        config = Config(cfg_path)

        # Start cluster
        from pivtools_cli.piv_cluster.cluster import start_cluster

        t_cluster_start = time.perf_counter()
        cluster, client = start_cluster(
            config=config,
            worker_omp_threads=str(threads),
        )
        t_cluster_ready = time.perf_counter()
        cluster_startup = t_cluster_ready - t_cluster_start

        try:
            # Import pipeline components
            from pivtools_core.image_handling.load_images import load_images
            from pivtools_cli.piv.save_results import get_output_path
            from pivtools_cli.processing.dask_pipeline import (
                create_filter_pipeline,
                scatter_immutable_data,
            )
            from pivtools_core.instantaneous import (
                process_instantaneous_sliding_window,
            )

            source_path = Path(SOURCE_DIR)
            camera_num = 1

            output_path = get_output_path(
                config, camera_num,
                use_uncalibrated=True,
                base_path_idx=0,
                piv_type="instantaneous",
            )

            # --- Timed pipeline ---
            t_pipeline_start = time.perf_counter()

            # 1. Load images (lazy)
            t0 = time.perf_counter()
            images = load_images(camera_num, config, source=source_path, batch_size=BATCH_SIZE)
            t_load = time.perf_counter() - t0

            # 2. Scatter immutable data
            t0 = time.perf_counter()
            scattered = scatter_immutable_data(
                client, config, None, None, ensemble=False
            )
            t_scatter = time.perf_counter() - t0

            # 3. Filter pipeline (no filters, but keeps the pipeline shape)
            images = create_filter_pipeline(images, config, None)

            # 4. Scatter config
            scattered_config = client.scatter(config, broadcast=True)

            # 5. Sliding window correlation + save
            num_chunks = len(images.chunks[0])
            t0 = time.perf_counter()
            saved_paths = process_instantaneous_sliding_window(
                client=client,
                images=images,
                num_chunks=num_chunks,
                scattered_config=scattered_config,
                scattered=scattered,
                output_path=output_path,
                config=config,
            )
            t_correlate = time.perf_counter() - t0

            t_pipeline_total = time.perf_counter() - t_pipeline_start

            n_saved = len(saved_paths)
            per_pair_ms = (t_correlate / N_PAIRS) * 1000.0
            pairs_per_s = N_PAIRS / t_correlate if t_correlate > 0 else 0

            return {
                "workers": workers,
                "threads": threads,
                "total_cores": workers * threads,
                "iteration": iteration,
                "n_pairs": N_PAIRS,
                "batch_size": BATCH_SIZE,
                "n_saved": n_saved,
                "cluster_startup_s": round(cluster_startup, 2),
                "load_s": round(t_load, 3),
                "scatter_s": round(t_scatter, 3),
                "correlate_s": round(t_correlate, 3),
                "pipeline_total_s": round(t_pipeline_total, 3),
                "per_pair_ms": round(per_pair_ms, 2),
                "pairs_per_s": round(pairs_per_s, 2),
            }

        finally:
            # Clean shutdown
            _quiet_shutdown(client, cluster)

    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {
            "workers": workers,
            "threads": threads,
            "total_cores": workers * threads,
            "iteration": iteration,
            "n_pairs": N_PAIRS,
            "batch_size": BATCH_SIZE,
            "n_saved": 0,
            "cluster_startup_s": 0,
            "load_s": 0,
            "scatter_s": 0,
            "correlate_s": 0,
            "pipeline_total_s": 0,
            "per_pair_ms": 0,
            "pairs_per_s": 0,
            "error": str(e),
        }

    finally:
        # Clean up temp output
        shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()


def _quiet_shutdown(client, cluster):
    """Shut down Dask without noisy logs."""
    import logging as _logging
    for name in ["distributed", "distributed.worker", "distributed.scheduler",
                 "distributed.nanny", "distributed.core", "distributed.comm",
                 "tornado.application", "tornado.general"]:
        _logging.getLogger(name).setLevel(_logging.CRITICAL)

    try:
        client.close(timeout=10)
    except Exception:
        pass
    time.sleep(0.3)
    try:
        cluster.close(timeout=10)
    except Exception:
        pass
    time.sleep(0.2)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "workers", "threads", "total_cores", "iteration", "n_pairs", "batch_size",
    "n_saved", "cluster_startup_s", "load_s", "scatter_s", "correlate_s",
    "pipeline_total_s", "per_pair_ms", "pairs_per_s", "error",
]


def write_csv_header(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_csv_row(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writerow(row)


def load_csv_results(csv_path):
    results = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key == "error":
                    continue
                try:
                    if "." in str(row[key]):
                        row[key] = float(row[key])
                    else:
                        row[key] = int(row[key])
                except (ValueError, TypeError):
                    pass
            results.append(row)
    return results


def get_completed_configs(csv_path):
    """Return set of (workers, threads, iteration) already completed."""
    if not os.path.exists(csv_path):
        return set()
    results = load_csv_results(csv_path)
    return {(r["workers"], r["threads"], r["iteration"]) for r in results}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_sweep(configs, csv_path, n_iterations=1):
    """Run all configs, appending results to CSV. Supports resume."""
    completed = get_completed_configs(csv_path)
    if not os.path.exists(csv_path):
        write_csv_header(csv_path)

    # Build task list
    tasks = []
    for cfg in configs:
        for it in range(n_iterations):
            key = (cfg["workers"], cfg["threads"], it)
            if key not in completed:
                tasks.append((cfg, it))

    if not tasks:
        print("All configurations already completed. Nothing to run.")
        return

    total = len(tasks)
    elapsed_times = []

    print(f"\n{'='*70}")
    print(f"SCALING BENCHMARK: {total} configurations to run")
    print(f"  {N_PAIRS} pairs, batch_size={BATCH_SIZE}, {IMAGE_SHAPE[0]}x{IMAGE_SHAPE[1]}")
    print(f"  Results: {csv_path}")
    print(f"{'='*70}\n")

    for i, (cfg, iteration) in enumerate(tasks):
        w, t = cfg["workers"], cfg["threads"]
        label = cfg["label"]

        # ETA
        if elapsed_times:
            avg_time = sum(elapsed_times) / len(elapsed_times)
            remaining = (total - i) * avg_time
            eta = datetime.now() + timedelta(seconds=remaining)
            eta_str = f"  ETA: {eta.strftime('%H:%M:%S')} (~{remaining/60:.0f} min remaining)"
        else:
            eta_str = ""

        print(f"[{i+1}/{total}] {label}: {w} workers x {t} threads "
              f"({w*t} cores) iter={iteration}{eta_str}", flush=True)

        t0 = time.perf_counter()
        result = run_single_benchmark(w, t, iteration)
        elapsed = time.perf_counter() - t0
        elapsed_times.append(elapsed)

        if result:
            result.setdefault("error", "")
            append_csv_row(csv_path, result)
            if result.get("pairs_per_s", 0) > 0:
                print(f"  -> {result['per_pair_ms']:.1f} ms/pair, "
                      f"{result['pairs_per_s']:.1f} pairs/s, "
                      f"correlate={result['correlate_s']:.1f}s "
                      f"(run took {elapsed:.0f}s)\n", flush=True)
            else:
                print(f"  -> FAILED: {result.get('error', 'unknown')}\n", flush=True)

    print(f"\nAll {total} runs complete. Results in: {csv_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def generate_plots(csv_path):
    """Generate scaling plots from CSV results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = load_csv_results(csv_path)
    if not results:
        print("No results to plot.")
        return

    out_dir = os.path.dirname(csv_path) or "."
    timestamp = os.path.splitext(os.path.basename(csv_path))[0]

    # Aggregate iterations: mean ± std
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        if r.get("pairs_per_s", 0) <= 0:
            continue
        key = (r["workers"], r["threads"], r.get("label", ""))
        grouped[(r["workers"], r["threads"])].append(r)

    def agg(rows, field):
        vals = [r[field] for r in rows if r.get(field, 0) > 0]
        if not vals:
            return 0, 0
        return np.mean(vals), np.std(vals) if len(vals) > 1 else 0

    # ---- Plot 1: Thread scaling (workers=1) ----
    thread_data = [(k, v) for k, v in grouped.items() if k[0] == 1]
    thread_data.sort(key=lambda x: x[0][1])

    if thread_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Thread Scaling (1 worker, {N_PAIRS} pairs, batch={BATCH_SIZE})", fontsize=14)

        threads = [k[1] for k, _ in thread_data]
        throughput_mean = [agg(v, "pairs_per_s")[0] for _, v in thread_data]
        throughput_std = [agg(v, "pairs_per_s")[1] for _, v in thread_data]
        per_pair_mean = [agg(v, "per_pair_ms")[0] for _, v in thread_data]
        per_pair_std = [agg(v, "per_pair_ms")[1] for _, v in thread_data]

        # Throughput
        ax1.errorbar(threads, throughput_mean, yerr=throughput_std, marker="o", linewidth=2, capsize=4)
        # Ideal scaling line
        if throughput_mean[0] > 0:
            ideal = [throughput_mean[0] * t for t in threads]
            ax1.plot(threads, ideal, "--", color="gray", alpha=0.5, label="Linear scaling")
            ax1.legend()
        ax1.set_xlabel("OMP Threads")
        ax1.set_ylabel("Pairs / second")
        ax1.set_title("Throughput")
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(threads)

        # Efficiency
        if throughput_mean[0] > 0:
            efficiency = [tp / (throughput_mean[0] * t) * 100 for tp, t in zip(throughput_mean, threads)]
            ax2.bar(range(len(threads)), efficiency, tick_label=[str(t) for t in threads], color="steelblue")
            ax2.set_ylabel("Parallel efficiency (%)")
            ax2.set_xlabel("OMP Threads")
            ax2.set_title("Scaling Efficiency")
            ax2.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
            ax2.set_ylim(0, max(120, max(efficiency) + 10))
            ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = os.path.join(out_dir, f"{timestamp}_thread_scaling.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    # ---- Plot 2: Worker scaling (fixed threads) ----
    # Find configs with consistent thread count (most common)
    thread_counts = defaultdict(list)
    for (w, t), v in grouped.items():
        if w > 1 or t == 4:  # Include single-worker baseline at matching thread count
            thread_counts[t].append(((w, t), v))

    for fixed_threads, data in thread_counts.items():
        if len(data) < 2:
            continue

        data.sort(key=lambda x: x[0][0])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Worker Scaling ({fixed_threads} threads/worker, {N_PAIRS} pairs, batch={BATCH_SIZE})",
                     fontsize=14)

        workers = [k[0] for k, _ in data]
        throughput_mean = [agg(v, "pairs_per_s")[0] for _, v in data]
        throughput_std = [agg(v, "pairs_per_s")[1] for _, v in data]

        ax1.errorbar(workers, throughput_mean, yerr=throughput_std, marker="s", linewidth=2, capsize=4,
                     color="tab:orange")
        if throughput_mean[0] > 0:
            ideal = [throughput_mean[0] * w for w in workers]
            ax1.plot(workers, ideal, "--", color="gray", alpha=0.5, label="Linear scaling")
            ax1.legend()
        ax1.set_xlabel("Dask Workers")
        ax1.set_ylabel("Pairs / second")
        ax1.set_title("Throughput")
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(workers)

        if throughput_mean[0] > 0:
            efficiency = [tp / (throughput_mean[0] * w) * 100 for tp, w in zip(throughput_mean, workers)]
            ax2.bar(range(len(workers)), efficiency, tick_label=[str(w) for w in workers], color="tab:orange")
            ax2.set_ylabel("Parallel efficiency (%)")
            ax2.set_xlabel("Dask Workers")
            ax2.set_title("Scaling Efficiency")
            ax2.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
            ax2.set_ylim(0, max(120, max(efficiency) + 10))
            ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = os.path.join(out_dir, f"{timestamp}_worker_scaling_t{fixed_threads}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    # ---- Plot 3: Heatmap (workers × threads → throughput) ----
    if len(grouped) >= 4:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"Worker × Thread Scaling ({N_PAIRS} pairs, batch={BATCH_SIZE})", fontsize=14)

        all_workers = sorted(set(k[0] for k in grouped))
        all_threads = sorted(set(k[1] for k in grouped))

        throughput_grid = np.full((len(all_workers), len(all_threads)), np.nan)
        perpair_grid = np.full((len(all_workers), len(all_threads)), np.nan)

        for (w, t), v in grouped.items():
            wi = all_workers.index(w)
            ti = all_threads.index(t)
            throughput_grid[wi, ti] = agg(v, "pairs_per_s")[0]
            perpair_grid[wi, ti] = agg(v, "per_pair_ms")[0]

        # Throughput heatmap
        im1 = ax1.imshow(throughput_grid, aspect="auto", cmap="YlOrRd",
                         origin="lower", interpolation="nearest")
        ax1.set_xticks(range(len(all_threads)))
        ax1.set_xticklabels(all_threads)
        ax1.set_yticks(range(len(all_workers)))
        ax1.set_yticklabels(all_workers)
        ax1.set_xlabel("OMP Threads")
        ax1.set_ylabel("Dask Workers")
        ax1.set_title("Throughput (pairs/s)")
        plt.colorbar(im1, ax=ax1)
        # Annotate cells
        for wi in range(len(all_workers)):
            for ti in range(len(all_threads)):
                val = throughput_grid[wi, ti]
                if not np.isnan(val):
                    cores = all_workers[wi] * all_threads[ti]
                    ax1.text(ti, wi, f"{val:.0f}\n({cores}c)",
                             ha="center", va="center", fontsize=8,
                             color="white" if val > np.nanmax(throughput_grid) * 0.6 else "black")

        # Per-pair heatmap
        im2 = ax2.imshow(perpair_grid, aspect="auto", cmap="YlOrRd_r",
                         origin="lower", interpolation="nearest")
        ax2.set_xticks(range(len(all_threads)))
        ax2.set_xticklabels(all_threads)
        ax2.set_yticks(range(len(all_workers)))
        ax2.set_yticklabels(all_workers)
        ax2.set_xlabel("OMP Threads")
        ax2.set_ylabel("Dask Workers")
        ax2.set_title("Time per pair (ms)")
        plt.colorbar(im2, ax=ax2)
        for wi in range(len(all_workers)):
            for ti in range(len(all_threads)):
                val = perpair_grid[wi, ti]
                if not np.isnan(val):
                    cores = all_workers[wi] * all_threads[ti]
                    ax2.text(ti, wi, f"{val:.0f}\n({cores}c)",
                             ha="center", va="center", fontsize=8,
                             color="white" if val < np.nanmin(perpair_grid) * 1.5 else "black")

        plt.tight_layout()
        path = os.path.join(out_dir, f"{timestamp}_scaling_heatmap.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    # ---- Plot 4: Total cores vs throughput (all configs) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Throughput vs Total Cores ({N_PAIRS} pairs, batch={BATCH_SIZE})")

    for (w, t), v in sorted(grouped.items()):
        tp_mean, tp_std = agg(v, "pairs_per_s")
        if tp_mean > 0:
            ax.errorbar(w * t, tp_mean, yerr=tp_std, marker="o", capsize=3,
                        color="steelblue", markersize=8)
            ax.annotate(f"{w}w×{t}t", (w * t, tp_mean),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Ideal line from single-thread baseline
    baseline_configs = [v for (w, t), v in grouped.items() if w == 1 and t == 1]
    if baseline_configs:
        base_tp = agg(baseline_configs[0], "pairs_per_s")[0]
        if base_tp > 0:
            core_range = range(1, TOTAL_CORES + 1)
            ax.plot(core_range, [base_tp * c for c in core_range], "--",
                    color="gray", alpha=0.5, label="Linear scaling")
            ax.legend()

    ax.set_xlabel("Total cores (workers × threads)")
    ax.set_ylabel("Pairs / second")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{timestamp}_cores_vs_throughput.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # ---- Summary table ----
    print(f"\n{'Workers':>8} {'Threads':>8} {'Cores':>6} {'pairs/s':>10} {'ms/pair':>10} {'correlate_s':>12}")
    print("-" * 60)
    for (w, t), v in sorted(grouped.items(), key=lambda x: (-agg(x[1], "pairs_per_s")[0])):
        tp_mean, _ = agg(v, "pairs_per_s")
        pp_mean, _ = agg(v, "per_pair_ms")
        corr_mean, _ = agg(v, "correlate_s")
        print(f"{w:>8} {t:>8} {w*t:>6} {tp_mean:>10.1f} {pp_mean:>10.1f} {corr_mean:>12.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Worker × Thread scaling benchmark")
    parser.add_argument("--sweep", choices=["threads", "workers", "matrix", "oversub", "all"],
                        default="all", help="Which sweep to run (default: all)")
    parser.add_argument("--resume", metavar="CSV", help="Resume from existing CSV file")
    parser.add_argument("--plots-only", metavar="CSV", help="Re-generate plots from CSV")
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS,
                        help=f"Iterations per config (default: {N_ITERATIONS})")
    parser.add_argument("--source", type=str, default=None,
                        help="Override SOURCE_DIR for image location")
    args = parser.parse_args()

    if args.source:
        global SOURCE_DIR
        SOURCE_DIR = args.source

    # Plots only mode
    if args.plots_only:
        print("Generating plots from existing results...")
        generate_plots(args.plots_only)
        return

    # Determine output CSV
    if args.resume:
        csv_path = args.resume
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(
            os.path.dirname(__file__),
            f"scaling_{timestamp}.csv",
        )

    # Build config list
    configs = []
    if args.sweep in ("threads", "all"):
        configs.extend(get_thread_sweep())
    if args.sweep in ("workers", "all"):
        configs.extend(get_worker_sweep())
    if args.sweep in ("matrix", "all"):
        configs.extend(get_matrix_sweep())
    if args.sweep in ("oversub", "all"):
        configs.extend(get_oversub_sweep())

    # Deduplicate (keep label from first occurrence)
    seen = set()
    deduped = []
    for cfg in configs:
        key = (cfg["workers"], cfg["threads"])
        if key not in seen:
            seen.add(key)
            deduped.append(cfg)
    configs = deduped

    # Validate source directory
    if not os.path.isdir(SOURCE_DIR):
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        print("Update SOURCE_DIR in this script or use --source <path>")
        sys.exit(1)

    # Check we have enough images
    expected_files = [os.path.join(SOURCE_DIR, f"B{i:05d}_A.tif") for i in range(1, N_PAIRS + 1)]
    missing = [f for f in expected_files[:3] if not os.path.exists(f)]
    if missing:
        print(f"ERROR: Expected image files not found. First missing: {missing[0]}")
        print(f"Need B00001_A.tif through B{N_PAIRS:05d}_A.tif in {SOURCE_DIR}")
        sys.exit(1)

    print(f"Source: {SOURCE_DIR}")
    print(f"CSV: {csv_path}")
    print(f"Configs: {len(configs)} unique, {args.iterations} iteration(s) each")
    print(f"Total runs: {len(configs) * args.iterations}")

    run_sweep(configs, csv_path, n_iterations=args.iterations)

    print("\nGenerating plots...")
    generate_plots(csv_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
