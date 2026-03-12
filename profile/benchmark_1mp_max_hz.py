"""
Maximum Hz benchmark for 1MP images.

Runs 1 worker, 20 OMP threads, minimal save, no compression.
Tests both 64->32 and 64->32->16 pass configurations.
Uses 20 image pairs with 5 iterations for stable statistics.

Usage:
    python profile/benchmark_1mp_max_hz.py
"""

import csv
import os
import sys
import tempfile
import time
import shutil
from datetime import datetime

import cv2
import numpy as np
import yaml

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pivtools_core.config import Config
from pivtools_cli.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU
from pivtools_cli.piv.save_results import save_piv_result_distributed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SOURCE_1MP = (
    r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
    r"\#current_processing\4000_images_channel\Profile_images\1mp"
)

N_PAIRS = 20
OMP_THREADS = 20
N_ITERATIONS = 5

PASS_CONFIGS = [
    ("64-32",     [[64, 64], [32, 32]],                [50, 50]),
    ("64-32-16",  [[64, 64], [32, 32], [16, 16]],      [50, 50, 50]),
]

PC_SUB_SECTIONS = ["pc_gaussian_smooth", "pc_predictor_remap", "pc_fused_warp"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_image_pairs(source_dir, n_pairs):
    pairs = []
    for idx in range(1, n_pairs + 1):
        a = cv2.imread(os.path.join(source_dir, f"B{idx:05d}_A.tif"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        b = cv2.imread(os.path.join(source_dir, f"B{idx:05d}_B.tif"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        pairs.append(np.stack([a, b]))
    return np.stack(pairs)


def make_config(image_shape, window_sizes, overlaps, omp_threads):
    cfg_dict = {
        "images": {
            "shape": image_shape,
            "num_images": 100,
            "format": ["B%05d_A.tif", "B%05d_B.tif"],
            "type": "standard",
            "start_index": 1,
            "frame_stride": 0,
            "pair_stride": 1,
            "pairing_preset": "ab_format",
        },
        "paths": {"source_paths": ["."], "base_paths": ["."], "camera_count": 1},
        "processing": {"backend": "cpu", "omp_threads": omp_threads},
        "instantaneous_piv": {
            "window_size": window_sizes,
            "overlap": overlaps,
            "peak_finder": "gauss6",
            "secondary_peak": False,
            "window_type": "gaussian",
            "runs": list(range(1, len(window_sizes) + 1)),
            "save_mode": "minimal",
            "save_compression": False,
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
    tmpdir = tempfile.mkdtemp(prefix="piv_bench_1mp_")
    cfg_path = os.path.join(tmpdir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
    return Config(cfg_path)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    print(f"{'='*70}")
    print(f"1MP MAX Hz BENCHMARK")
    print(f"  Source: {SOURCE_1MP}")
    print(f"  Pairs: {N_PAIRS}, Threads: {OMP_THREADS}, Iterations: {N_ITERATIONS}")
    print(f"  Save: minimal, uncompressed")
    print(f"{'='*70}")

    images = load_image_pairs(SOURCE_1MP, N_PAIRS)
    print(f"Loaded {N_PAIRS} pairs, shape={images.shape}, dtype={images.dtype}")
    h, w = images.shape[2], images.shape[3]

    save_tmpdir = tempfile.mkdtemp(prefix="piv_bench_1mp_save_")
    all_results = []

    for cfg_label, windows, overlaps in PASS_CONFIGS:
        print(f"\n{'─'*70}")
        print(f"  {cfg_label}  ({h}x{w}, {OMP_THREADS} threads, {N_PAIRS} pairs)")
        print(f"{'─'*70}")

        config = make_config([h, w], windows, overlaps, OMP_THREADS)
        correlator = InstantaneousCorrelatorCPU(config)
        correlator.profiling_enabled = True
        runs_to_save = config.instantaneous_runs_0based
        n_passes = len(windows)
        last_pass = n_passes - 1

        # Warmup (FFTW plans)
        print("  Warmup...", end=" ", flush=True)
        t0 = time.perf_counter()
        correlator.correlate_batch(images, config)
        print(f"done ({time.perf_counter()-t0:.1f}s)")

        # Timed iterations
        iter_times_ms = []
        save_times_ms = []
        pass_breakdowns = []  # list of dicts per iteration

        for it in range(N_ITERATIONS):
            piv_results = correlator.correlate_batch(images, config)
            profile = correlator.get_profile_summary()

            # Correlation time (per pair)
            corr_total = 0.0
            for pass_idx in profile:
                corr_total += sum(v for k, v in profile[pass_idx].items() if k not in PC_SUB_SECTIONS)
            corr_per_pair_ms = corr_total / N_PAIRS * 1000.0
            iter_times_ms.append(corr_per_pair_ms)

            # Save time (per pair)
            t0 = time.perf_counter()
            for i, piv_result in enumerate(piv_results):
                save_piv_result_distributed(
                    piv_result, save_tmpdir, i + 1, runs_to_save,
                    save_mode=config.instantaneous_save_mode,
                    do_compression=config.instantaneous_save_compression,
                )
            save_per_pair_ms = (time.perf_counter() - t0) / N_PAIRS * 1000.0
            save_times_ms.append(save_per_pair_ms)

            # Per-pass breakdown
            breakdown = {}
            for pass_idx in sorted(profile.keys()):
                sections = profile[pass_idx]
                pass_total = sum(v for k, v in sections.items() if k not in PC_SUB_SECTIONS)
                breakdown[pass_idx] = {
                    "total": pass_total / N_PAIRS * 1000.0,
                    "warp": sections.get("predictor_corrector", 0) / N_PAIRS * 1000.0,
                    "xcorr": sections.get("bulkxcorr2d", 0) / N_PAIRS * 1000.0,
                    "outlier": sections.get("outlier_detection", 0) / N_PAIRS * 1000.0,
                    "infill": sections.get("infilling", 0) / N_PAIRS * 1000.0,
                }
                breakdown[pass_idx]["other"] = (
                    breakdown[pass_idx]["total"]
                    - breakdown[pass_idx]["warp"]
                    - breakdown[pass_idx]["xcorr"]
                    - breakdown[pass_idx]["outlier"]
                    - breakdown[pass_idx]["infill"]
                )
            pass_breakdowns.append(breakdown)

            print(f"  Iter {it+1}/{N_ITERATIONS}: {corr_per_pair_ms:.1f} ms/pair corr + {save_per_pair_ms:.1f} ms/pair save")

        # Aggregate
        corr_mean = np.mean(iter_times_ms)
        corr_std = np.std(iter_times_ms)
        save_mean = np.mean(save_times_ms)
        save_std = np.std(save_times_ms)
        total_mean = corr_mean + save_mean
        hz_corr = 1000.0 / corr_mean
        hz_total = 1000.0 / total_mean

        print(f"\n  Results ({N_ITERATIONS} iterations):")
        print(f"    Correlation: {corr_mean:.1f} +/- {corr_std:.1f} ms/pair  ({hz_corr:.1f} Hz)")
        print(f"    Save:        {save_mean:.1f} +/- {save_std:.1f} ms/pair")
        print(f"    Total:       {total_mean:.1f} ms/pair  ({hz_total:.1f} Hz)")

        # Average per-pass breakdown
        print(f"\n    Per-pass breakdown (mean across iterations):")
        for pass_idx in sorted(pass_breakdowns[0].keys()):
            avg = {}
            for key in ["total", "warp", "xcorr", "outlier", "infill", "other"]:
                avg[key] = np.mean([bd[pass_idx][key] for bd in pass_breakdowns])
            win = windows[pass_idx]
            print(f"      Pass {pass_idx+1} ({win[0]}x{win[1]}): {avg['total']:.1f} ms  "
                  f"(warp={avg['warp']:.1f}, xcorr={avg['xcorr']:.1f}, "
                  f"outlier={avg['outlier']:.1f}, infill={avg['infill']:.1f}, other={avg['other']:.1f})")

        all_results.append({
            "passes": cfg_label,
            "resolution": f"{h}x{w}",
            "threads": OMP_THREADS,
            "n_pairs": N_PAIRS,
            "corr_ms": round(corr_mean, 1),
            "corr_std_ms": round(corr_std, 1),
            "save_ms": round(save_mean, 1),
            "total_ms": round(total_mean, 1),
            "hz_corr": round(hz_corr, 1),
            "hz_total": round(hz_total, 1),
        })

    shutil.rmtree(save_tmpdir, ignore_errors=True)

    # CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(os.path.dirname(__file__), f"1mp_max_hz_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  Results saved to: {csv_path}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"1MP MAX Hz SUMMARY ({OMP_THREADS} threads, minimal save, no compression)")
    print(f"{'='*70}")
    print(f"{'Passes':<12} {'Corr (ms)':>10} {'Save (ms)':>10} {'Total (ms)':>11} {'Hz (corr)':>10} {'Hz (total)':>11}")
    print("-" * 68)
    for r in all_results:
        print(f"{r['passes']:<12} {r['corr_ms']:>8.1f} {r['save_ms']:>8.1f} {r['total_ms']:>9.1f} {r['hz_corr']:>10.1f} {r['hz_total']:>10.1f}")
    print()


if __name__ == "__main__":
    main()
