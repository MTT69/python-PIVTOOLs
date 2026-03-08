"""
Benchmark instantaneous PIV: batch size sweep + resolution comparison.

Test 1: 4MP images, batch sizes 1,2,3,5,10,20, 1 worker, 10 OMP threads, 64-32 passes
Test 2: 4MP and 1MP (centre 1000x1000 crop) at 64-32 and 64-32-16, 10 OMP threads
"""

import os
import sys
import tempfile
import time

import cv2
import numpy as np
import yaml

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pivtools_core.config import Config
from pivtools_cli.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SOURCE_4MP = (
    "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton"
    "/Documents/#current_processing/4000_images_channel/Profile_images"
)
SOURCE_1MP = os.path.join(SOURCE_4MP, "1mp")

N_PAIRS = 20
OMP_THREADS = 10
N_ITERATIONS = 3


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


def run_benchmark(images, image_shape, window_sizes, overlaps, omp_threads, n_iterations):
    """Run benchmark, return list of profile dicts."""
    config = make_config(image_shape, window_sizes, overlaps, omp_threads)
    correlator = InstantaneousCorrelatorCPU(config)
    correlator.profiling_enabled = True

    # Warmup
    correlator.correlate_batch(images, config)

    profiles = []
    for _ in range(n_iterations):
        correlator.correlate_batch(images, config)
        profiles.append(correlator.get_profile_summary())
    return profiles, config


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
# Test 1: Batch size sweep (4MP, 64-32)
# ---------------------------------------------------------------------------
def test_batch_sizes():
    print("\n" + "=" * 70)
    print("TEST 1: Batch size sweep (4MP 2048x2048, 64->32, 10 OMP threads)")
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
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "pairs_per_s": pairs_per_s,
            "mem_mb": mem_mb,
        })

    print("\n\nBatch Size Summary (4MP, 64->32, 10 threads):")
    print(f"{'Batch':>6} {'Per pair (ms)':>14} {'Pairs/s':>10} {'Speedup':>10} {'Memory':>10}")
    base = results[0]["mean_ms"]
    for r in results:
        speedup = base / r["mean_ms"]
        print(f"{r['batch_size']:>6} {r['mean_ms']:>10.1f} ms {r['pairs_per_s']:>10.1f} {speedup:>9.2f}x {r['mem_mb']:>8.0f} MB")

    return results


# ---------------------------------------------------------------------------
# Test 2: Resolution + pass config comparison
# ---------------------------------------------------------------------------
def test_resolution_and_passes():
    print("\n" + "=" * 70)
    print("TEST 2: Resolution (4MP vs 1MP) x Pass config (64-32 vs 64-32-16)")
    print("=" * 70)

    # Load images
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
        ("4MP (2048x2048)", [2048, 2048], images_4mp),
        ("1MP (1000x1000)", [1000, 1000], images_1mp),
    ]

    all_results = []

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
            )
            mean_ms, std_ms = get_per_pair_ms(profiles, N_PAIRS)
            pairs_per_s = 1000.0 / mean_ms
            breakdown = get_per_pass_breakdown(profiles, N_PAIRS)

            print(f"  Per pair: {mean_ms:.1f} +/- {std_ms:.1f} ms  ({pairs_per_s:.1f} Hz)")

            # Print per-pass breakdown
            for pass_idx in sorted(breakdown.keys()):
                sections = breakdown[pass_idx]
                pass_total = sum(v for k, v in sections.items() if k not in PC_SUB_SECTIONS)
                warp = sections.get("predictor_corrector", 0)
                xcorr = sections.get("bulkxcorr2d", 0)
                outlier = sections.get("outlier_detection", 0)
                infill = sections.get("infilling", 0)
                other = pass_total - warp - xcorr - outlier - infill
                print(f"  Pass {pass_idx+1}: {pass_total:.1f} ms  (warp={warp:.1f}, xcorr={xcorr:.1f}, outlier={outlier:.1f}, infill={infill:.1f}, other={other:.1f})")

            all_results.append({
                "resolution": res_label,
                "config": cfg_label,
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "pairs_per_s": pairs_per_s,
                "breakdown": breakdown,
            })

    print("\n\nSummary Table:")
    print(f"{'Resolution':<20} {'Config':<12} {'Per pair (ms)':>14} {'Hz':>8}")
    print("-" * 58)
    for r in all_results:
        print(f"{r['resolution']:<20} {r['config']:<12} {r['mean_ms']:>10.1f} ms {r['pairs_per_s']:>8.1f}")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Creating 1MP centre crops...")
    create_1mp_crops()

    batch_results = test_batch_sizes()
    resolution_results = test_resolution_and_passes()

    print("\n\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)
