"""
Ensemble PIV Fitting/Finalization Phase Profiler.

Benchmarks the finalization phase (fitting + outlier + infill) in isolation.
Runs correlation once (without profiling) to generate accumulated data,
then re-runs finalization with profiling for each iteration.

Uses Client(processes=False) to avoid serialization overhead — isolates
pure fitting computation cost.

Multiple iterations with mean +/- std statistics.

Usage:
    python profile/profile_ensemble_fitting.py 4mp
    python profile/profile_ensemble_fitting.py 4mp --pairs 10 --iterations 3
    python profile/profile_ensemble_fitting.py 4mp --fit-method kspace
    python profile/profile_ensemble_fitting.py 4mp --workers 4
"""

import argparse
import os
import sys
import time

import numpy as np

# Ensure the project root is on the path so imports work when running as a script
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from profile_ensemble import (
    IMAGE_PRESETS,
    load_image_pairs,
    make_ensemble_config,
)

# ---------------------------------------------------------------------------
# Section display order for finalization profiling
# ---------------------------------------------------------------------------
SECTION_ORDER = [
    "mean_computation",
    "bg_subtraction",
    "normalization",
    "sigma_propagation",
    "scatter",
    "fitting",
    "velocity_extraction",
    "parameter_extraction",
    "outlier_detection",
    "infilling",
    "stress_outlier_detection",
    "result_construction",
]

DISPLAY_NAMES = {
    "mean_computation": "mean_computation",
    "bg_subtraction": "bg_subtraction",
    "normalization": "normalization",
    "sigma_propagation": "sigma_propagation",
    "scatter": "scatter (local)",
    "fitting": "fitting",
    "velocity_extraction": "velocity_extraction",
    "parameter_extraction": "parameter_extraction",
    "outlier_detection": "outlier_detection",
    "infilling": "infilling",
    "stress_outlier_detection": "stress_outlier_detection",
    "result_construction": "result_construction",
}


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------
def format_bar(pct: float, max_width: int = 20) -> str:
    n = int(round(pct / 100.0 * max_width))
    return "=" * n


def print_results(
    all_profiles: list,
    config,
    label: str,
    n_pairs: int,
    n_workers: int,
):
    """Print averaged finalization profiling results across iterations."""
    from pivtools_core.window_utils import compute_window_centers, compute_window_centers_single_mode

    n_iter = len(all_profiles)
    n_passes = config.ensemble_num_passes

    print(f"\n{'#' * 70}")
    print(f"# FITTING/FINALIZATION PHASE: {label}")
    print(f"#   {n_pairs} pairs, {n_workers} Dask worker{'s' if n_workers > 1 else ''} "
          f"(processes=False), {n_iter} iteration{'s' if n_iter > 1 else ''}")
    print(f"{'#' * 70}")

    grand_totals = []

    for pass_idx in range(n_passes):
        win_size = config.ensemble_window_sizes[pass_idx]
        overlap = config.ensemble_overlaps[pass_idx]
        runtype = config.ensemble_type[pass_idx]

        # Collect timings across iterations for this pass
        pass_timings = {}
        for profile in all_profiles:
            if pass_idx not in profile:
                continue
            for section, elapsed in profile[pass_idx].items():
                pass_timings.setdefault(section, []).append(elapsed)

        if not pass_timings:
            continue

        # Compute pass total per iteration
        pass_total_per_iter = []
        for profile in all_profiles:
            if pass_idx not in profile:
                continue
            total = sum(v for v in profile[pass_idx].values())
            pass_total_per_iter.append(total)

        pass_mean = np.mean(pass_total_per_iter)
        pass_std = np.std(pass_total_per_iter) if n_iter > 1 else 0.0
        grand_totals.extend(pass_total_per_iter)

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

        print(f"\nPass {pass_idx + 1} ({win_size[0]}x{win_size[1]}, {overlap}% overlap, {runtype}) -- {grid_str}")
        print("-" * 70)

        for section in SECTION_ORDER:
            if section not in pass_timings:
                continue

            times = pass_timings[section]
            mean = np.mean(times)
            std = np.std(times) if n_iter > 1 else 0.0
            pct = (mean / pass_mean * 100) if pass_mean > 0 else 0.0

            display_name = DISPLAY_NAMES.get(section, section)

            if n_iter > 1:
                line = f"  {display_name:<28s} {mean:7.3f}s +/- {std:.3f}s  {pct:5.1f}%  {format_bar(pct)}"
            else:
                line = f"  {display_name:<28s} {mean:7.3f}s            {pct:5.1f}%  {format_bar(pct)}"

            # Annotate stress_outlier_detection (final pass only)
            if section == "stress_outlier_detection" and pass_idx < n_passes - 1:
                line += "  (skip)"

            print(line)

        if n_iter > 1:
            print(f"  {'TOTAL':<28s} {pass_mean:7.3f}s +/- {pass_std:.3f}s")
        else:
            print(f"  {'TOTAL':<28s} {pass_mean:7.3f}s")

    # Grand total
    if grand_totals:
        per_iter_totals = []
        for profile in all_profiles:
            t = sum(
                sum(v for v in profile[pass_idx].values())
                for pass_idx in profile
            )
            per_iter_totals.append(t)

        grand_mean = np.mean(per_iter_totals)
        grand_std = np.std(per_iter_totals) if n_iter > 1 else 0.0
        print(f"\n{'=' * 70}")
        if n_iter > 1:
            print(f"Grand total (finalization only): {grand_mean:.3f}s +/- {grand_std:.3f}s")
        else:
            print(f"Grand total (finalization only): {grand_mean:.3f}s")
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
    n_iterations: int,
    n_workers: int,
    infilling_method: str = "local_median",
    bg_subtraction: str = "correlation",
    gradient_correction: bool = False,
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
    print(f"  Dask workers: {n_workers} (processes=False)")
    print(f"  Iterations: {n_iterations}")

    # Split images into batches
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

    # --- Phase 1: Run correlation ONCE to get accumulated data per pass ---
    print(f"\nRunning correlation (one-time, not profiled)...")
    # We need to cache per-pass: {pass_idx: (accumulated_result, predictor_field)}
    cached_pass_data = {}
    predictor_field = None

    for pass_idx in range(len(window_sizes)):
        print(f"  Pass {pass_idx + 1}/{len(window_sizes)}...", end=" ", flush=True)
        t0 = time.perf_counter()

        batch_results = []
        for batch_idx, batch in enumerate(batches):
            result = correlator.correlate_batch_for_accumulation(
                batch, config,
                pass_idx=pass_idx,
                predictor_field=predictor_field,
                is_first_batch=(batch_idx == 0),
            )
            batch_results.append(result)

        # Reduce batch results
        accumulated = batch_results[0]
        for r in batch_results[1:]:
            accumulated = reduce_ensemble_results(accumulated, r)

        cached_pass_data[pass_idx] = {
            "accumulated": accumulated,
            "predictor_field": predictor_field,
        }

        # Run finalization to get predictor for next pass
        accumulator = SinglePassAccumulator(config)
        accumulator.accumulate_batch(accumulated, pass_idx=pass_idx)

        from dask.distributed import Client
        client = Client(processes=False, n_workers=n_workers, threads_per_worker=omp_threads, silence_logs=50)
        try:
            pass_result = accumulator.finalize_pass(
                client=client, pass_idx=pass_idx,
                predictor_field=predictor_field,
            )
        finally:
            client.close()

        if pass_idx < len(window_sizes) - 1:
            predictor_field = extract_predictor_field(pass_result)

        elapsed = time.perf_counter() - t0
        print(f"{elapsed:.2f}s")

        del batch_results, accumulator

    # --- Phase 2: Profile finalization for each iteration ---
    print(f"\nProfiling finalization ({n_iterations} iterations)...")
    all_profiles = []

    for iteration in range(n_iterations):
        print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

        client = Client(processes=False, n_workers=n_workers, threads_per_worker=omp_threads, silence_logs=50)
        try:
            for pass_idx in range(len(window_sizes)):
                print(f"  Pass {pass_idx + 1}/{len(window_sizes)}...", end=" ", flush=True)

                # Create fresh accumulator and accumulate cached data
                accumulator = SinglePassAccumulator(config)
                accumulator.profiling_enabled = True

                # Load any previous passes results (needed for sigma propagation)
                if pass_idx > 0 and iteration > 0:
                    # We need previous pass results. Run finalization for passes 0..pass_idx-1
                    # without profiling to build up the results chain
                    prev_accumulator = SinglePassAccumulator(config)
                    prev_predictor = None
                    for prev_idx in range(pass_idx):
                        prev_data = cached_pass_data[prev_idx]
                        prev_accumulator.accumulate_batch(
                            prev_data["accumulated"], pass_idx=prev_idx
                        )
                        prev_result = prev_accumulator.finalize_pass(
                            client=client, pass_idx=prev_idx,
                            predictor_field=prev_data["predictor_field"],
                        )
                        prev_accumulator.clear_pass_data(prev_idx)
                    # Now copy previous results to our profiled accumulator
                    accumulator.passes_results = list(prev_accumulator.passes_results)
                    del prev_accumulator
                elif pass_idx > 0:
                    # First iteration: re-build previous pass results
                    prev_accumulator = SinglePassAccumulator(config)
                    for prev_idx in range(pass_idx):
                        prev_data = cached_pass_data[prev_idx]
                        prev_accumulator.accumulate_batch(
                            prev_data["accumulated"], pass_idx=prev_idx
                        )
                        prev_result = prev_accumulator.finalize_pass(
                            client=client, pass_idx=prev_idx,
                            predictor_field=prev_data["predictor_field"],
                        )
                        prev_accumulator.clear_pass_data(prev_idx)
                    accumulator.passes_results = list(prev_accumulator.passes_results)
                    del prev_accumulator

                pass_data = cached_pass_data[pass_idx]
                accumulator.accumulate_batch(
                    pass_data["accumulated"], pass_idx=pass_idx
                )

                # Finalize with profiling
                pass_result = accumulator.finalize_pass(
                    client=client, pass_idx=pass_idx,
                    predictor_field=pass_data["predictor_field"],
                )

                # Collect profile
                accum_profile = accumulator.get_profile_summary()
                if iteration >= len(all_profiles):
                    all_profiles.append({})
                if pass_idx in accum_profile:
                    all_profiles[iteration][pass_idx] = accum_profile[pass_idx]

                # Report inline
                prof = accum_profile.get(pass_idx, {})
                fit_t = prof.get("fitting", 0)
                bg_t = prof.get("bg_subtraction", 0)
                print(f"bg={bg_t:.2f}s, fit={fit_t:.2f}s")

                accumulator.clear_pass_data(pass_idx)
                del accumulator

        finally:
            client.close()

    print_results(all_profiles, config, preset["label"], n_pairs, n_workers)


def main():
    parser = argparse.ArgumentParser(
        description="Profile Ensemble PIV fitting/finalization phase with granular sub-sections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python profile/profile_ensemble_fitting.py 4mp
  python profile/profile_ensemble_fitting.py 4mp --pairs 10 --iterations 3
  python profile/profile_ensemble_fitting.py 4mp --fit-method kspace
  python profile/profile_ensemble_fitting.py 4mp --workers 4
        """,
    )
    parser.add_argument(
        "preset", choices=["4mp", "25mp", "both"],
        help="Image preset to use",
    )
    parser.add_argument("--pairs", type=int, default=20, help="Number of image pairs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size (default: 10)")
    parser.add_argument("--threads", type=int, default=4, help="OMP thread count (default: 4)")
    parser.add_argument("--iterations", type=int, default=3, help="Number of profiling iterations (default: 3)")
    parser.add_argument("--workers", type=int, default=1, help="Number of Dask workers (default: 1)")
    parser.add_argument("--windows", type=str, default="96,32,16", help="Window sizes per pass (default: 96,32,16)")
    parser.add_argument("--overlaps", type=str, default=None, help="Overlaps per pass")
    parser.add_argument("--types", type=str, default=None, help="Ensemble types: std or single")
    parser.add_argument("--sum-window", type=str, default="48,48", help="Sum window for single mode (default: 48,48)")
    parser.add_argument("--fit-method", type=str, default="gaussian", choices=["gaussian", "kspace"])
    parser.add_argument("--no-outlier", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--infilling", type=str, default="local_median")
    parser.add_argument("--bg-method", type=str, default="correlation", choices=["correlation", "image"])
    parser.add_argument("--gradient-correction", action="store_true")

    args = parser.parse_args()

    win_sizes = [[int(w), int(w)] for w in args.windows.split(",")]
    n_passes = len(win_sizes)

    if args.overlaps:
        overlaps = [int(o) for o in args.overlaps.split(",")]
    else:
        overlaps = [75] + [50] * (n_passes - 1)

    if args.types:
        ensemble_types = args.types.split(",")
    else:
        ensemble_types = ["std"] + ["single"] * (n_passes - 1) if n_passes > 1 else ["std"]

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
            n_iterations=args.iterations,
            n_workers=args.workers,
            infilling_method=args.infilling,
            bg_subtraction=args.bg_method,
            gradient_correction=args.gradient_correction,
        )


if __name__ == "__main__":
    main()
