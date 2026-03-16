"""
Benchmark instantaneous PIV: batch size, resolution, and save I/O.

Test 1: Batch size sweep — 4MP, batch sizes 1..20, 10 OMP threads, 64->32
Test 2: Resolution + pass config — 4MP/1MP x 64->32/64->32->16
Test 3: Save I/O — 4 save mode combos (full/minimal x compressed/uncompressed)

All correlation tests use minimal save + no compression for best raw speed.
Results saved to CSV for write-up.

Thread and worker scaling is handled by benchmark_scaling.py (full Dask pipeline).
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
# Paths
# ---------------------------------------------------------------------------
SOURCE_4MP = (
    r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
    r"\#current_processing\4000_images_channel\planar_images"
)
SOURCE_1MP = os.path.join(SOURCE_4MP, "1mp")

N_PAIRS = 20
OMP_THREADS = 10  # overridden by --threads
N_ITERATIONS = 3


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def make_csv_path(test_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.path.dirname(__file__), f"{test_name}_{timestamp}.csv")


def write_csv(csv_path, fieldnames, rows):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_image_pair(source_dir, pair_idx=1):
    a_path = os.path.join(source_dir, f"B{pair_idx:05d}_A.tif")
    b_path = os.path.join(source_dir, f"B{pair_idx:05d}_B.tif")
    img_a = cv2.imread(a_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img_b = cv2.imread(b_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return np.stack([img_a, img_b])


def load_image_pairs(source_dir, n_pairs):
    pairs = [load_image_pair(source_dir, idx) for idx in range(1, n_pairs + 1)]
    return np.stack(pairs)


def make_config(image_shape, window_sizes, overlaps, omp_threads,
                save_mode="full", save_compression=True,
                outlier_enabled=True, infill_enabled=True):
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
            "save_mode": save_mode,
            "save_compression": save_compression,
        },
        "outlier_detection": {
            "enabled": outlier_enabled,
            "methods": [{"type": "median_2d", "threshold": 2.0, "epsilon": 0.2}],
        },
        "infilling": {
            "mid_pass": {"enabled": infill_enabled, "method": "local_median", "parameters": {"ksize": 3}},
            "final_pass": {"enabled": infill_enabled, "method": "local_median", "parameters": {"ksize": 3}},
        },
    }
    tmpdir = tempfile.mkdtemp(prefix="piv_bench_")
    cfg_path = os.path.join(tmpdir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
    return Config(cfg_path)


PC_SUB_SECTIONS = ["pc_gaussian_smooth", "pc_predictor_remap", "pc_fused_warp"]


def get_per_pair_ms(profiles, n_pairs):
    """Return mean per-pair time in ms across iterations."""
    per_iter = []
    for profile in profiles:
        total = 0.0
        for pass_idx in profile:
            total += sum(v for k, v in profile[pass_idx].items() if k not in PC_SUB_SECTIONS)
        per_iter.append(total / n_pairs * 1000.0)
    return np.mean(per_iter), np.std(per_iter) if len(per_iter) > 1 else 0.0


def get_per_pass_breakdown(profiles, n_pairs):
    """Return per-pass per-pair ms breakdown {pass_idx: {section: mean_ms}}."""
    n_passes = max(max(p.keys()) for p in profiles) + 1
    result = {}
    for pass_idx in range(n_passes):
        section_times = {}
        for profile in profiles:
            if pass_idx not in profile:
                continue
            for section, elapsed in profile[pass_idx].items():
                section_times.setdefault(section, []).append(elapsed / n_pairs * 1000.0)
        result[pass_idx] = {k: np.mean(v) for k, v in section_times.items()}
    return result


def run_benchmark(images, image_shape, window_sizes, overlaps, omp_threads,
                  n_iterations, save_mode="minimal", save_compression=False,
                  outlier_enabled=True, infill_enabled=True):
    """Run benchmark, return list of profile dicts."""
    config = make_config(image_shape, window_sizes, overlaps, omp_threads,
                         save_mode=save_mode, save_compression=save_compression,
                         outlier_enabled=outlier_enabled, infill_enabled=infill_enabled)
    correlator = InstantaneousCorrelatorCPU(config)
    correlator.profiling_enabled = True

    # Warmup
    correlator.correlate_batch(images, config)

    profiles = []
    for _ in range(n_iterations):
        correlator.correlate_batch(images, config)
        profiles.append(correlator.get_profile_summary())
    return profiles, config


def run_benchmark_with_save(images, image_shape, window_sizes, overlaps, omp_threads,
                            n_iterations, save_mode="full", save_compression=True,
                            outlier_enabled=True, infill_enabled=True):
    """Run benchmark including save timing. Returns (profiles, config, file_size_kb)."""
    config = make_config(image_shape, window_sizes, overlaps, omp_threads,
                         save_mode=save_mode, save_compression=save_compression,
                         outlier_enabled=outlier_enabled, infill_enabled=infill_enabled)
    correlator = InstantaneousCorrelatorCPU(config)
    correlator.profiling_enabled = True

    runs_to_save = config.instantaneous_runs_0based
    n_passes = len(window_sizes)
    last_pass = n_passes - 1
    save_tmpdir = tempfile.mkdtemp(prefix="piv_bench_save_")

    # Warmup (correlation only)
    correlator.correlate_batch(images, config)

    profiles = []
    for _ in range(n_iterations):
        piv_results = correlator.correlate_batch(images, config)
        profile = correlator.get_profile_summary()

        # Time the save
        t0 = time.perf_counter()
        for i, piv_result in enumerate(piv_results):
            save_piv_result_distributed(
                piv_result, save_tmpdir, i + 1, runs_to_save,
                save_mode=config.instantaneous_save_mode,
                do_compression=config.instantaneous_save_compression,
            )
        t_save = time.perf_counter() - t0
        profile.setdefault(last_pass, {})["save"] = t_save

        profiles.append(profile)

    # Measure file size from the last saved file
    sample_file = os.path.join(save_tmpdir, "B00001.mat")
    file_size_kb = os.path.getsize(sample_file) / 1024.0 if os.path.exists(sample_file) else 0.0

    shutil.rmtree(save_tmpdir, ignore_errors=True)
    return profiles, config, file_size_kb


# ---------------------------------------------------------------------------
# Create 1MP crops
# ---------------------------------------------------------------------------
def create_1mp_crops():
    os.makedirs(SOURCE_1MP, exist_ok=True)
    for idx in range(1, N_PAIRS + 1):
        for suffix in ("A", "B"):
            src = os.path.join(SOURCE_4MP, f"B{idx:05d}_{suffix}.tif")
            dst = os.path.join(SOURCE_1MP, f"B{idx:05d}_{suffix}.tif")
            if os.path.exists(dst):
                continue
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
            cy, cx = h // 2, w // 2
            crop = img[cy - 500 : cy + 500, cx - 500 : cx + 500]
            cv2.imwrite(dst, crop)
            print(f"  Created {dst} ({crop.shape})")
    print(f"  1MP crops ready in {SOURCE_1MP}")


# ---------------------------------------------------------------------------
# Test 1: Batch size sweep (4MP, 64->32)
# ---------------------------------------------------------------------------
def test_batch_sizes():
    print("\n" + "=" * 70)
    print("TEST 1: Batch size sweep (4MP 2048x2048, 64->32, 10 OMP threads)")
    print("        save_mode=minimal, save_compression=off")
    print("=" * 70)

    all_images = load_image_pairs(SOURCE_4MP, N_PAIRS)
    print(f"Loaded {N_PAIRS} pairs, shape={all_images.shape}")

    batch_sizes = [1, 2, 3, 5, 10, 20]
    results = []

    for bs in batch_sizes:
        print(f"\n--- Batch size = {bs} ---")
        images = all_images[:bs]
        profiles, config = run_benchmark(
            images,
            image_shape=[2048, 2048],
            window_sizes=[[64, 64], [32, 32]],
            overlaps=[50, 50],
            omp_threads=OMP_THREADS,
            n_iterations=N_ITERATIONS,
        )
        mean_ms, std_ms = get_per_pair_ms(profiles, bs)
        pairs_per_s = 1000.0 / mean_ms if mean_ms > 0 else 0
        mem_mb = bs * 2 * 2048 * 2048 * 4 / 1e6
        print(f"  Per pair: {mean_ms:.1f} +/- {std_ms:.1f} ms  ({pairs_per_s:.1f} pairs/s)  Working set: {mem_mb:.0f} MB")
        results.append({
            "batch_size": bs,
            "per_pair_ms": round(mean_ms, 1),
            "std_ms": round(std_ms, 1),
            "pairs_per_s": round(pairs_per_s, 1),
            "working_set_mb": round(mem_mb, 0),
        })

    # Summary
    print("\n\nBatch Size Summary (4MP, 64->32, 10 threads):")
    print(f"{'Batch':>6} {'Per pair (ms)':>14} {'Pairs/s':>10} {'Speedup':>10} {'Memory':>10}")
    base = results[0]["per_pair_ms"]
    for r in results:
        speedup = base / r["per_pair_ms"]
        print(f"{r['batch_size']:>6} {r['per_pair_ms']:>10.1f} ms {r['pairs_per_s']:>10.1f} {speedup:>9.2f}x {r['working_set_mb']:>8.0f} MB")

    # CSV
    csv_path = make_csv_path("batch_sweep")
    write_csv(csv_path, ["batch_size", "per_pair_ms", "std_ms", "pairs_per_s", "working_set_mb"], results)

    return results


# ---------------------------------------------------------------------------
# Test 2: Resolution + pass config comparison
# ---------------------------------------------------------------------------
def test_resolution_and_passes(outlier_enabled=True, infill_enabled=True):
    outlier_label = "ON" if outlier_enabled else "OFF"
    print("\n" + "=" * 70)
    print(f"TEST 2: Resolution (4MP vs 1MP) x Pass config (64->32 vs 64->32->16)")
    print(f"        {OMP_THREADS} OMP threads, outlier={outlier_label}, save_mode=minimal, compression=off")
    print("=" * 70)

    print("\nLoading 4MP images...")
    images_4mp = load_image_pairs(SOURCE_4MP, N_PAIRS)
    print(f"  Shape: {images_4mp.shape}")

    print("Loading 1MP images...")
    images_1mp = load_image_pairs(SOURCE_1MP, N_PAIRS)
    print(f"  Shape: {images_1mp.shape}")

    configs = [
        ("64-32", [[64, 64], [32, 32]], [50, 50]),
        ("64-32-16", [[64, 64], [32, 32], [16, 16]], [50, 50, 50]),
    ]

    resolutions = [
        ("4MP", [2048, 2048], images_4mp),
        ("1MP", [1000, 1000], images_1mp),
    ]

    summary_rows = []
    breakdown_rows = []

    for res_label, shape, images in resolutions:
        for cfg_label, windows, overlaps in configs:
            print(f"\n--- {res_label}, {cfg_label} ---")
            profiles, config = run_benchmark(
                images,
                image_shape=shape,
                window_sizes=windows,
                overlaps=overlaps,
                omp_threads=OMP_THREADS,
                n_iterations=N_ITERATIONS,
                outlier_enabled=outlier_enabled,
                infill_enabled=infill_enabled,
            )
            mean_ms, std_ms = get_per_pair_ms(profiles, N_PAIRS)
            pairs_per_s = 1000.0 / mean_ms
            breakdown = get_per_pass_breakdown(profiles, N_PAIRS)

            print(f"  Per pair: {mean_ms:.1f} +/- {std_ms:.1f} ms  ({pairs_per_s:.1f} Hz)")

            summary_rows.append({
                "resolution": res_label,
                "passes": cfg_label,
                "per_pair_ms": round(mean_ms, 1),
                "std_ms": round(std_ms, 1),
                "pairs_per_s": round(pairs_per_s, 1),
            })

            for pass_idx in sorted(breakdown.keys()):
                sections = breakdown[pass_idx]
                pass_total = sum(v for k, v in sections.items() if k not in PC_SUB_SECTIONS)
                warp = sections.get("predictor_corrector", 0)
                xcorr = sections.get("bulkxcorr2d", 0)
                outlier = sections.get("outlier_detection", 0)
                infill = sections.get("infilling", 0)
                other = pass_total - warp - xcorr - outlier - infill
                print(f"  Pass {pass_idx+1}: {pass_total:.1f} ms  (warp={warp:.1f}, xcorr={xcorr:.1f}, outlier={outlier:.1f}, infill={infill:.1f}, other={other:.1f})")

                win_size = windows[pass_idx]
                from pivtools_core.window_utils import compute_window_centers
                wc = compute_window_centers(
                    image_shape=tuple(shape), window_size=tuple(win_size),
                    overlap=overlaps[pass_idx], validate=False,
                )

                breakdown_rows.append({
                    "resolution": res_label,
                    "passes": cfg_label,
                    "pass_num": pass_idx + 1,
                    "window_size": f"{win_size[0]}x{win_size[1]}",
                    "grid_size": f"{wc.n_win_y}x{wc.n_win_x}",
                    "total_ms": round(pass_total, 1),
                    "warp_ms": round(warp, 1),
                    "xcorr_ms": round(xcorr, 1),
                    "outlier_ms": round(outlier, 1),
                    "infill_ms": round(infill, 1),
                    "other_ms": round(other, 1),
                })

    # Summary
    print("\n\nSummary Table:")
    print(f"{'Resolution':<12} {'Config':<12} {'Per pair (ms)':>14} {'Hz':>8}")
    print("-" * 50)
    for r in summary_rows:
        print(f"{r['resolution']:<12} {r['passes']:<12} {r['per_pair_ms']:>10.1f} ms {r['pairs_per_s']:>8.1f}")

    # CSVs
    csv_summary = make_csv_path("resolution_summary")
    write_csv(csv_summary, ["resolution", "passes", "per_pair_ms", "std_ms", "pairs_per_s"], summary_rows)

    csv_breakdown = make_csv_path("resolution_breakdown")
    write_csv(csv_breakdown,
              ["resolution", "passes", "pass_num", "window_size", "grid_size",
               "total_ms", "warp_ms", "xcorr_ms", "outlier_ms", "infill_ms", "other_ms"],
              breakdown_rows)

    return summary_rows, breakdown_rows


# ---------------------------------------------------------------------------
# Test 3: Save I/O benchmark (4 combos)
# ---------------------------------------------------------------------------
def test_save_io():
    print("\n" + "=" * 70)
    print("TEST 3: Save I/O benchmark (4MP 2048x2048, 64->32, 20 pairs, 10 threads)")
    print("        Comparing 4 save mode combinations")
    print("=" * 70)

    images = load_image_pairs(SOURCE_4MP, N_PAIRS)
    print(f"Loaded {N_PAIRS} pairs, shape={images.shape}")

    save_combos = [
        ("full + compressed",      "full",    True),
        ("full + uncompressed",    "full",    False),
        ("minimal + compressed",   "minimal", True),
        ("minimal + uncompressed", "minimal", False),
    ]

    results = []

    for label, save_mode, save_compression in save_combos:
        print(f"\n--- {label} ---")
        profiles, config, file_size_kb = run_benchmark_with_save(
            images,
            image_shape=[2048, 2048],
            window_sizes=[[64, 64], [32, 32]],
            overlaps=[50, 50],
            omp_threads=OMP_THREADS,
            n_iterations=N_ITERATIONS,
            save_mode=save_mode,
            save_compression=save_compression,
        )

        # Extract save time (attributed to last pass)
        save_times_ms = []
        corr_times_ms = []
        for profile in profiles:
            save_t = 0.0
            corr_t = 0.0
            for pass_idx in profile:
                for section, elapsed in profile[pass_idx].items():
                    if section == "save":
                        save_t += elapsed
                    elif section not in PC_SUB_SECTIONS:
                        corr_t += elapsed
            # corr_t includes save since it's in the profile dict, subtract it
            corr_t -= save_t
            save_times_ms.append(save_t / N_PAIRS * 1000.0)
            corr_times_ms.append(corr_t / N_PAIRS * 1000.0)

        save_mean = np.mean(save_times_ms)
        save_std = np.std(save_times_ms) if len(save_times_ms) > 1 else 0.0
        corr_mean = np.mean(corr_times_ms)
        total_mean = save_mean + corr_mean

        print(f"  Correlation: {corr_mean:.1f} ms/pair")
        print(f"  Save:        {save_mean:.1f} +/- {save_std:.1f} ms/pair")
        print(f"  File size:   {file_size_kb:.0f} KB")
        print(f"  Total:       {total_mean:.1f} ms/pair")

        results.append({
            "label": label,
            "save_mode": save_mode,
            "compressed": save_compression,
            "corr_ms": round(corr_mean, 1),
            "save_ms": round(save_mean, 1),
            "save_std_ms": round(save_std, 1),
            "total_ms": round(total_mean, 1),
            "file_kb": round(file_size_kb, 0),
        })

    # Summary
    print("\n\nSave I/O Summary (4MP, 64->32, 20 pairs, 10 threads):")
    print(f"{'Mode':<26} {'Corr (ms)':>10} {'Save (ms)':>10} {'Total (ms)':>11} {'File (KB)':>10} {'Save %':>8}")
    print("-" * 80)
    for r in results:
        save_pct = r["save_ms"] / r["total_ms"] * 100 if r["total_ms"] > 0 else 0
        print(f"{r['label']:<26} {r['corr_ms']:>8.1f} {r['save_ms']:>8.1f} {r['total_ms']:>9.1f} {r['file_kb']:>8.0f} {save_pct:>7.1f}%")

    # CSV
    csv_path = make_csv_path("save_io")
    write_csv(csv_path,
              ["label", "save_mode", "compressed", "corr_ms", "save_ms", "save_std_ms", "total_ms", "file_kb"],
              results)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PIV benchmark suite")
    parser.add_argument("--test", type=int, choices=[1, 2, 3],
                        help="Run only a specific test (1=batch, 2=resolution, 3=save)")
    parser.add_argument("--threads", type=int, default=10,
                        help="OMP threads (default: 10)")
    parser.add_argument("--no-outlier", action="store_true",
                        help="Disable outlier detection and infilling")
    args = parser.parse_args()

    OMP_THREADS = args.threads
    outlier_on = not args.no_outlier

    if args.test is None or args.test == 2:
        print("Creating 1MP centre crops...")
        create_1mp_crops()

    if args.test is None or args.test == 1:
        test_batch_sizes()

    if args.test is None or args.test == 2:
        test_resolution_and_passes(outlier_enabled=outlier_on, infill_enabled=outlier_on)

    if args.test is None or args.test == 3:
        test_save_io()

    print("\n\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)
