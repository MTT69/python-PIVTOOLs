"""
Standalone Ensemble PIV profiling script.

Bypasses Dask entirely — loads real images from disk, creates a minimal Config,
and calls EnsembleCorrelatorCPU.correlate_batch_for_accumulation() directly
with per-section timing instrumentation.

Profiles the two main phases:
  1. CORRELATION: accumulate correlation planes across all batches (per pass)
  2. FINALIZATION: Gaussian/k-space fitting + outlier detection + infilling (per pass)

Usage:
    python profile/profile_ensemble.py 4mp
    python profile/profile_ensemble.py 4mp --pairs 20 --batch-size 10
    python profile/profile_ensemble.py 4mp --passes 3 --threads 4
    python profile/profile_ensemble.py 4mp --fit-method kspace
    python profile/profile_ensemble.py 4mp --windows 96,32,16 --types std,single,single
"""

import argparse
import os
import sys
import tempfile
import time

import cv2
import numpy as np
import yaml

# Ensure the project root is on the path so imports work when running as a script
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pivtools_core.config import Config

# ---------------------------------------------------------------------------
# Hardcoded image presets (same as profile_piv.py)
# ---------------------------------------------------------------------------
IMAGE_PRESETS = {
    "25mp": {
        "path": (
            r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
            r"\#current_processing\query_JHTDB\download_from_jhtdb\efe_images"
        ),
        "shape": [4600, 5312],
        "label": "25 MP (4600x5312)",
    },
    "4mp": {
        "path": (
            r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
            r"\#current_processing\4000_images_channel\planar_images"
        ),
        "shape": [2048, 2048],
        "label": "4 MP (2048x2048)",
    },
}


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_image_pair(source_dir: str, pair_idx: int = 1) -> np.ndarray:
    """Load one AB-format image pair, return (2, H, W) float32."""
    a_path = os.path.join(source_dir, f"B{pair_idx:05d}_A.tif")
    b_path = os.path.join(source_dir, f"B{pair_idx:05d}_B.tif")
    if not os.path.isfile(a_path):
        raise FileNotFoundError(f"Image A not found: {a_path}")
    if not os.path.isfile(b_path):
        raise FileNotFoundError(f"Image B not found: {b_path}")
    img_a = cv2.imread(a_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img_b = cv2.imread(b_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return np.stack([img_a, img_b])  # (2, H, W)


def load_image_pairs(source_dir: str, n_pairs: int) -> np.ndarray:
    """Load N AB-format image pairs, return (N, 2, H, W) float32."""
    pairs = []
    for idx in range(1, n_pairs + 1):
        pairs.append(load_image_pair(source_dir, idx))
    return np.stack(pairs)  # (N, 2, H, W)


# ---------------------------------------------------------------------------
# Config creation
# ---------------------------------------------------------------------------
def make_ensemble_config(
    image_shape: list,
    window_sizes: list,
    overlaps: list,
    ensemble_types: list,
    sum_window: list,
    omp_threads: int,
    fit_method: str = "gaussian",
    outlier_enabled: bool = True,
    infilling_method: str = "local_median",
    bg_subtraction: str = "correlation",
    gradient_correction: bool = False,
    sum_fitting_window: list = None,
    sum_fitting_window_enabled: bool = False,
) -> Config:
    """Create a minimal Config from a temporary YAML file for ensemble profiling."""
    runs = list(range(1, len(window_sizes) + 1))

    cfg_dict = {
        "images": {
            "shape": image_shape,
            "num_images": 100,  # placeholder
            "format": ["B%05d_A.tif", "B%05d_B.tif"],
            "type": "standard",
            "start_index": 1,
            "frame_stride": 0,
            "pair_stride": 1,
            "pairing_preset": "ab_format",
        },
        "paths": {
            "source_paths": ["."],
            "base_paths": ["."],
            "camera_count": 1,
        },
        "processing": {
            "backend": "cpu",
            "omp_threads": omp_threads,
            "ensemble": True,
        },
        "ensemble_piv": {
            "window_size": window_sizes,
            "overlap": overlaps,
            "type": ensemble_types,
            "sum_window": sum_window,
            "runs": runs,
            "fit_method": fit_method,
            "background_subtraction_method": bg_subtraction,
            "gradient_correction": gradient_correction,
            "mask_center_pixel": True,
            "window_type": "square",
            "sum_fitting_window_enabled": sum_fitting_window_enabled,
            "sum_fitting_window": sum_fitting_window or sum_window,
        },
        "outlier_detection": {
            "enabled": outlier_enabled,
            "methods": [
                {"type": "median_2d", "threshold": 2.0, "epsilon": 0.2},
            ],
        },
        "ensemble_outlier_detection": {
            "enabled": outlier_enabled,
            "methods": [
                {"type": "median_2d", "threshold": 2.0, "epsilon": 0.2},
            ],
        },
        "infilling": {
            "mid_pass": {
                "enabled": True,
                "method": infilling_method,
                "parameters": {"ksize": 3},
            },
            "final_pass": {
                "enabled": True,
                "method": infilling_method,
                "parameters": {"ksize": 3},
            },
        },
        "ensemble_infilling": {
            "mid_pass": {
                "enabled": True,
                "method": infilling_method,
                "parameters": {"ksize": 3},
            },
            "final_pass": {
                "enabled": True,
                "method": infilling_method,
                "parameters": {"ksize": 3},
            },
        },
    }

    tmpdir = tempfile.mkdtemp(prefix="piv_ensemble_profile_")
    cfg_path = os.path.join(tmpdir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

    return Config(cfg_path)


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------
def format_bar(pct: float, max_width: int = 25) -> str:
    n = int(round(pct / 100.0 * max_width))
    return "=" * n


def print_results(pass_timings: list, config: Config, label: str, n_pairs: int, batch_size: int):
    """Print profiling results."""
    from pivtools_core.window_utils import compute_window_centers, compute_window_centers_single_mode

    n_passes = len(pass_timings)
    total_time = sum(t["total"] for t in pass_timings)

    print(f"\n{'#' * 70}")
    print(f"# {label}  ({n_pairs} pairs, batch_size={batch_size})")
    print(f"{'#' * 70}")

    for pass_idx, pt in enumerate(pass_timings):
        win_size = config.ensemble_window_sizes[pass_idx]
        overlap = config.ensemble_overlaps[pass_idx]
        runtype = config.ensemble_type[pass_idx]

        # Compute window grid size
        if runtype == "single":
            result = compute_window_centers_single_mode(
                image_shape=tuple(config.image_shape),
                window_size=tuple(win_size),
                sum_window=tuple(config.ensemble_sum_window),
                overlap=overlap,
                validate=False,
            )
        else:
            result = compute_window_centers(
                image_shape=tuple(config.image_shape),
                window_size=tuple(win_size),
                overlap=overlap,
                validate=False,
            )
        grid_str = f"{result.n_win_y}x{result.n_win_x} windows"

        pass_total = pt["total"]
        pct_of_total = (pass_total / total_time * 100) if total_time > 0 else 0

        print(f"\nPass {pass_idx + 1} ({win_size[0]}x{win_size[1]}, {overlap}% overlap, {runtype}) -- {grid_str}")
        print("-" * 70)

        sections = [
            ("correlation", pt["correlation"]),
            ("  predictor_corrector", pt.get("predictor_corrector", 0)),
            ("  xcorr (3x C lib)", pt.get("xcorr", 0)),
            ("finalize", pt["finalize"]),
            ("  bg_subtraction", pt.get("bg_subtraction", 0)),
            ("  fitting", pt.get("fitting", 0)),
            ("  outlier_detection", pt.get("outlier_detection", 0)),
            ("  infilling", pt.get("infilling", 0)),
        ]

        for name, elapsed in sections:
            pct = (elapsed / pass_total * 100) if pass_total > 0 else 0
            print(f"  {name:<24s} {elapsed:7.3f}s  {pct:5.1f}%  {format_bar(pct)}")

        print(f"  {'TOTAL':<24s} {pass_total:7.3f}s  ({pct_of_total:.0f}% of grand total)")

    print(f"\n{'=' * 70}")
    print(f"Grand total: {total_time:.3f}s")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main profiling loop
# ---------------------------------------------------------------------------
def run_profile(
    preset_name: str,
    n_pairs: int,
    batch_size: int,
    omp_threads: int,
    window_sizes: list,
    overlaps: list,
    ensemble_types: list,
    sum_window: list,
    fit_method: str,
    outlier_enabled: bool,
    do_warmup: bool,
    infilling_method: str = "local_median",
    bg_subtraction: str = "correlation",
    gradient_correction: bool = False,
    sum_fitting_window: list = None,
    sum_fitting_window_enabled: bool = False,
):
    preset = IMAGE_PRESETS[preset_name]
    print(f"\nLoading images from: {preset['path']}")
    print(f"  Preset: {preset['label']}")
    print(f"  Pairs: {n_pairs}")

    t0 = time.perf_counter()
    images = load_image_pairs(preset["path"], n_pairs)
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.2f}s  shape={images.shape}  dtype={images.dtype}")

    config = make_ensemble_config(
        image_shape=preset["shape"],
        window_sizes=window_sizes,
        overlaps=overlaps,
        ensemble_types=ensemble_types,
        sum_window=sum_window,
        omp_threads=omp_threads,
        fit_method=fit_method,
        outlier_enabled=outlier_enabled,
        infilling_method=infilling_method,
        bg_subtraction=bg_subtraction,
        gradient_correction=gradient_correction,
        sum_fitting_window=sum_fitting_window,
        sum_fitting_window_enabled=sum_fitting_window_enabled,
    )

    print(f"\nCreating ensemble correlator...")
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU
    from pivtools_cli.piv.piv_backend.single_pass_accumulator import SinglePassAccumulator
    from pivtools_cli.processing.dask_pipeline import reduce_ensemble_results, extract_predictor_field

    t0 = time.perf_counter()
    correlator = EnsembleCorrelatorCPU(config)
    create_time = time.perf_counter() - t0
    print(f"  Created in {create_time:.2f}s")

    print(f"\nProcessing config:")
    print(f"  Window sizes: {window_sizes}")
    print(f"  Overlaps: {overlaps}")
    print(f"  Types: {ensemble_types}")
    print(f"  Sum window: {sum_window}")
    print(f"  Fit method: {fit_method}")
    print(f"  OMP threads: {omp_threads}")
    print(f"  Batch size: {batch_size}")
    print(f"  Outlier detection: {'ON' if outlier_enabled else 'OFF'}")
    print(f"  Infilling: {infilling_method}")
    print(f"  BG subtraction: {bg_subtraction}")

    # Split images into batches (mimics Dask batch-loading)
    n_batches = (n_pairs + batch_size - 1) // batch_size
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_pairs)
        batches.append(images[start:end])
    print(f"  Batches: {n_batches} (sizes: {[b.shape[0] for b in batches]})")

    if do_warmup:
        print(f"\nWarmup pass (FFTW plan creation)...")
        t0 = time.perf_counter()
        correlator.correlate_batch_for_accumulation(
            batches[0], config, pass_idx=0, predictor_field=None, is_first_batch=True
        )
        warmup_time = time.perf_counter() - t0
        print(f"  Warmup completed in {warmup_time:.2f}s")

    # --- Run multi-pass ensemble profiling ---
    print(f"\nRunning profiled ensemble ({len(window_sizes)} passes)...")
    accumulator = SinglePassAccumulator(config)
    predictor_field = None
    pass_timings = []

    # Enable profiling on correlator and accumulator
    correlator.profiling_enabled = True
    accumulator.profiling_enabled = True

    # Need a dummy Dask client for finalize_pass (it uses client for distributed fitting)
    # For profiling, we create a local client with 1 worker
    from dask.distributed import Client
    client = Client(processes=False, n_workers=1, threads_per_worker=omp_threads, silence_logs=50)

    try:
        for pass_idx in range(len(window_sizes)):
            timing = {
                "correlation": 0, "predictor_corrector": 0, "xcorr": 0,
                "finalize": 0, "bg_subtraction": 0, "fitting": 0,
                "outlier_detection": 0, "infilling": 0, "total": 0,
            }

            # Reset profiling data for this pass
            correlator.reset_profile_data()
            accumulator.reset_profile_data()

            print(f"\n  Pass {pass_idx + 1}/{len(window_sizes)}...")
            pass_start = time.perf_counter()

            # Phase 1: Correlation accumulation (across all batches)
            corr_start = time.perf_counter()
            batch_results = []
            for batch_idx, batch in enumerate(batches):
                result = correlator.correlate_batch_for_accumulation(
                    batch, config,
                    pass_idx=pass_idx,
                    predictor_field=predictor_field,
                    is_first_batch=(batch_idx == 0),
                )
                batch_results.append(result)

            # Reduce batch results (simulates worker reduction)
            accumulated = batch_results[0]
            for r in batch_results[1:]:
                accumulated = reduce_ensemble_results(accumulated, r)

            timing["correlation"] = time.perf_counter() - corr_start

            # Collect sub-section timings from correlator (accumulated across batches)
            corr_profile = correlator.get_profile_summary()
            if pass_idx in corr_profile:
                timing["predictor_corrector"] = corr_profile[pass_idx].get("predictor_corrector", 0)
                timing["xcorr"] = corr_profile[pass_idx].get("xcorr", 0)

            # Phase 2: Finalization (fitting + outlier + infill)
            finalize_start = time.perf_counter()
            accumulator.accumulate_batch(accumulated, pass_idx=pass_idx)
            pass_result = accumulator.finalize_pass(
                client=client, pass_idx=pass_idx,
                predictor_field=predictor_field,
            )
            timing["finalize"] = time.perf_counter() - finalize_start

            # Collect sub-section timings from accumulator
            accum_profile = accumulator.get_profile_summary()
            if pass_idx in accum_profile:
                timing["bg_subtraction"] = accum_profile[pass_idx].get("bg_subtraction", 0)
                timing["fitting"] = accum_profile[pass_idx].get("fitting", 0)
                timing["outlier_detection"] = accum_profile[pass_idx].get("outlier_detection", 0)
                timing["infilling"] = accum_profile[pass_idx].get("infilling", 0)

            # Extract predictor for next pass
            if pass_idx < len(window_sizes) - 1:
                predictor_field = extract_predictor_field(pass_result)

            timing["total"] = time.perf_counter() - pass_start
            pass_timings.append(timing)

            print(f"    correlation: {timing['correlation']:.2f}s "
                  f"(pc={timing['predictor_corrector']:.2f}s, xcorr={timing['xcorr']:.2f}s)")
            print(f"    finalize:    {timing['finalize']:.2f}s "
                  f"(bg={timing['bg_subtraction']:.2f}s, fit={timing['fitting']:.2f}s, "
                  f"outlier={timing['outlier_detection']:.2f}s, infill={timing['infilling']:.2f}s)")

            # Cleanup between passes
            accumulator.clear_pass_data(pass_idx)
            del accumulated, batch_results
    finally:
        client.close()

    print_results(pass_timings, config, preset["label"], n_pairs, batch_size)


def main():
    parser = argparse.ArgumentParser(
        description="Profile Ensemble PIV processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python profile/profile_ensemble.py 4mp
  python profile/profile_ensemble.py 4mp --pairs 20 --batch-size 10
  python profile/profile_ensemble.py 4mp --passes 3 --threads 4
  python profile/profile_ensemble.py 4mp --fit-method kspace
  python profile/profile_ensemble.py 4mp --windows 96,32,16 --types std,single,single
        """,
    )
    parser.add_argument(
        "preset",
        choices=["4mp", "25mp", "both"],
        help="Image preset to use",
    )
    parser.add_argument(
        "--pairs", type=int, default=20,
        help="Number of image pairs to process (default: 20)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Batch size — images per batch (default: 10)",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="OMP thread count (default: 4)",
    )
    parser.add_argument(
        "--windows", type=str, default="96,32,16",
        help="Comma-separated square window sizes per pass (default: 96,32,16)",
    )
    parser.add_argument(
        "--overlaps", type=str, default=None,
        help="Comma-separated overlaps per pass (default: 75 for first, 50 for rest)",
    )
    parser.add_argument(
        "--types", type=str, default=None,
        help="Comma-separated ensemble types: std or single (default: std for first, single for rest)",
    )
    parser.add_argument(
        "--sum-window", type=str, default="48,48",
        help="Sum window size for single mode (default: 48,48)",
    )
    parser.add_argument(
        "--fit-method", type=str, default="gaussian",
        choices=["gaussian", "kspace"],
        help="Fitting method (default: gaussian)",
    )
    parser.add_argument(
        "--no-outlier", action="store_true",
        help="Disable outlier detection",
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip FFTW warmup pass",
    )
    parser.add_argument(
        "--infilling", type=str, default="local_median",
        help="Infilling method (default: local_median)",
    )
    parser.add_argument(
        "--bg-method", type=str, default="correlation",
        choices=["correlation", "image"],
        help="Background subtraction method (default: correlation)",
    )
    parser.add_argument(
        "--gradient-correction", action="store_true",
        help="Enable gradient correction for Reynolds stresses",
    )

    args = parser.parse_args()

    # Parse window sizes
    win_sizes = [[int(w), int(w)] for w in args.windows.split(",")]
    n_passes = len(win_sizes)

    # Parse overlaps
    if args.overlaps:
        overlaps = [int(o) for o in args.overlaps.split(",")]
    else:
        overlaps = [75] + [50] * (n_passes - 1)

    # Parse types
    if args.types:
        ensemble_types = args.types.split(",")
    else:
        ensemble_types = ["std"] + ["single"] * (n_passes - 1) if n_passes > 1 else ["std"]

    # Parse sum window
    sum_window = [int(x) for x in args.sum_window.split(",")]

    presets = ["4mp", "25mp"] if args.preset == "both" else [args.preset]

    for preset_name in presets:
        run_profile(
            preset_name=preset_name,
            n_pairs=args.pairs,
            batch_size=args.batch_size,
            omp_threads=args.threads,
            window_sizes=win_sizes,
            overlaps=overlaps,
            ensemble_types=ensemble_types,
            sum_window=sum_window,
            fit_method=args.fit_method,
            outlier_enabled=not args.no_outlier,
            do_warmup=not args.no_warmup,
            infilling_method=args.infilling,
            bg_subtraction=args.bg_method,
            gradient_correction=args.gradient_correction,
        )


if __name__ == "__main__":
    main()
