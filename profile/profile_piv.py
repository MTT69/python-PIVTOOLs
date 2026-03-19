"""
Standalone PIV profiling script.

Bypasses Dask entirely — loads real images from disk, creates a minimal Config,
and calls InstantaneousCorrelatorCPU.correlate_batch() directly with per-section
timing instrumentation.

Usage:
    python scripts/profile_piv.py 4mp
    python scripts/profile_piv.py 25mp
    python scripts/profile_piv.py both
    python scripts/profile_piv.py 4mp --pairs 4
    python scripts/profile_piv.py 25mp --iterations 5
    python scripts/profile_piv.py 25mp --threads 8
    python scripts/profile_piv.py 4mp --windows 128,64,32,16
    python scripts/profile_piv.py 4mp --no-outlier
    python scripts/profile_piv.py 4mp --no-warmup
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
from pivtools_cli.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU
from pivtools_cli.piv.save_results import save_piv_result_distributed

# ---------------------------------------------------------------------------
# Hardcoded image presets
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
def make_config(
    image_shape: list,
    window_sizes: list,
    overlaps: list,
    omp_threads: int,
    outlier_enabled: bool,
    infilling_method: str = "local_median",
    peak_finder: str = "gauss6",
    save_mode: str = "minimal",
    save_compression: bool = False,
) -> Config:
    """Create a minimal Config from a temporary YAML file."""
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
        },
        "instantaneous_piv": {
            "window_size": window_sizes,
            "overlap": overlaps,
            "peak_finder": peak_finder,
            "secondary_peak": False,
            "window_type": "gaussian",
            "runs": list(range(1, len(window_sizes) + 1)),
            "save_mode": save_mode,
            "save_compression": save_compression,
        },
        "outlier_detection": {
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
    }

    tmpdir = tempfile.mkdtemp(prefix="piv_profile_")
    cfg_path = os.path.join(tmpdir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

    return Config(cfg_path)


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------
PC_SUB_SECTIONS = [
    "pc_gaussian_smooth",
    "pc_predictor_remap",
    "pc_fused_warp",
]

SECTION_ORDER = [
    "predictor_corrector",
    "set_lib_args",
    "bulkxcorr2d",
    "post_processing",
    "outlier_detection",
    "secondary_peaks",
    "infilling",
    "padding_stacking",
    "result_construction",
    "save",
]


def format_bar(pct: float, max_width: int = 20) -> str:
    n = int(round(pct / 100.0 * max_width))
    return "=" * n


def print_results(
    all_profiles: list[dict],
    config: Config,
    label: str,
):
    """Print averaged profiling results across iterations."""
    n_iter = len(all_profiles)
    n_passes = len(config.window_sizes)

    print(f"\n{'#' * 70}")
    print(f"# {label}  ({n_iter} iteration{'s' if n_iter > 1 else ''})")
    print(f"{'#' * 70}")

    grand_totals = []

    for pass_idx in range(n_passes):
        win_size = config.window_sizes[pass_idx]
        overlap = config.overlap[pass_idx]

        # Collect timings across iterations for this pass
        pass_timings = {}  # section -> list of times
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
            total = sum(
                v for k, v in profile[pass_idx].items() if k not in PC_SUB_SECTIONS
            )
            pass_total_per_iter.append(total)

        pass_mean = np.mean(pass_total_per_iter)
        pass_std = np.std(pass_total_per_iter) if n_iter > 1 else 0.0
        grand_totals.extend(pass_total_per_iter)

        # Compute window grid size
        from pivtools_core.window_utils import compute_window_centers

        result = compute_window_centers(
            image_shape=tuple(config.image_shape),
            window_size=tuple(win_size),
            overlap=overlap,
            validate=False,
        )
        grid_str = f"{result.n_win_y}x{result.n_win_x} windows"

        print(f"\nPass {pass_idx + 1} ({win_size[0]}x{win_size[1]}, {overlap}% overlap) -- {grid_str}")
        print("-" * 70)

        for section in SECTION_ORDER:
            if section not in pass_timings:
                continue

            times = pass_timings[section]
            mean = np.mean(times)
            std = np.std(times) if n_iter > 1 else 0.0
            pct = (mean / pass_mean * 100) if pass_mean > 0 else 0.0

            if n_iter > 1:
                line = f"  {section:<28s} {mean:7.3f}s +/- {std:.3f}s  {pct:5.1f}%  {format_bar(pct)}"
            else:
                line = f"  {section:<28s} {mean:7.3f}s            {pct:5.1f}%  {format_bar(pct)}"

            # Special annotation for pass 0 predictor_corrector (skip)
            if section == "predictor_corrector" and pass_idx == 0:
                line += "  (skip)"

            print(line)

            # Print sub-sections for predictor_corrector
            if section == "predictor_corrector" and pass_idx > 0:
                for sub in PC_SUB_SECTIONS:
                    if sub not in pass_timings:
                        continue
                    sub_times = pass_timings[sub]
                    sub_mean = np.mean(sub_times)
                    sub_std = np.std(sub_times) if n_iter > 1 else 0.0
                    sub_pct = (sub_mean / pass_mean * 100) if pass_mean > 0 else 0.0
                    short_name = sub.replace("pc_", "")
                    if n_iter > 1:
                        print(f"    {short_name:<26s} {sub_mean:7.3f}s +/- {sub_std:.3f}s  {sub_pct:5.1f}%  {format_bar(sub_pct)}")
                    else:
                        print(f"    {short_name:<26s} {sub_mean:7.3f}s            {sub_pct:5.1f}%  {format_bar(sub_pct)}")

        if n_iter > 1:
            print(f"  {'TOTAL':<28s} {pass_mean:7.3f}s +/- {pass_std:.3f}s")
        else:
            print(f"  {'TOTAL':<28s} {pass_mean:7.3f}s")

    # Grand total
    if grand_totals:
        # Group per iteration
        per_iter_totals = []
        for profile in all_profiles:
            t = 0.0
            for pass_idx in profile:
                t += sum(
                    v
                    for k, v in profile[pass_idx].items()
                    if k not in PC_SUB_SECTIONS
                )
            per_iter_totals.append(t)

        grand_mean = np.mean(per_iter_totals)
        grand_std = np.std(per_iter_totals) if n_iter > 1 else 0.0
        print(f"\n{'=' * 70}")
        if n_iter > 1:
            print(f"Grand total: {grand_mean:.3f}s +/- {grand_std:.3f}s")
        else:
            print(f"Grand total: {grand_mean:.3f}s")
        print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_profile(
    preset_name: str,
    n_pairs: int,
    n_iterations: int,
    omp_threads: int,
    window_sizes: list,
    overlaps: list,
    outlier_enabled: bool,
    do_warmup: bool,
    infilling_method: str = "local_median",
    peak_finder: str = "gauss6",
    threading_enabled: bool = True,
    save_mode: str = "minimal",
    save_compression: bool = False,
):
    preset = IMAGE_PRESETS[preset_name]
    print(f"\nLoading images from: {preset['path']}")
    print(f"  Preset: {preset['label']}")
    print(f"  Pairs: {n_pairs}")

    t0 = time.perf_counter()
    images = load_image_pairs(preset["path"], n_pairs)
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.2f}s  shape={images.shape}  dtype={images.dtype}")

    config = make_config(
        image_shape=preset["shape"],
        window_sizes=window_sizes,
        overlaps=overlaps,
        omp_threads=omp_threads,
        outlier_enabled=outlier_enabled,
        infilling_method=infilling_method,
        peak_finder=peak_finder,
        save_mode=save_mode,
        save_compression=save_compression,
    )

    print(f"\nCreating correlator...")
    t0 = time.perf_counter()
    correlator = InstantaneousCorrelatorCPU(config)
    correlator.profiling_enabled = True
    correlator.threading_enabled = threading_enabled
    create_time = time.perf_counter() - t0
    print(f"  Created in {create_time:.2f}s")

    print(f"\nProcessing config:")
    print(f"  Window sizes: {window_sizes}")
    print(f"  Overlaps: {overlaps}")
    print(f"  OMP threads: {omp_threads}")
    print(f"  Outlier detection: {'ON' if outlier_enabled else 'OFF'}")
    print(f"  Infilling: {infilling_method}")
    print(f"  Peak finder: {peak_finder}")
    print(f"  Threading: {'ON' if threading_enabled else 'OFF'}")
    print(f"  Save mode: {save_mode}")
    print(f"  Save compression: {'ON' if save_compression else 'OFF'}")

    # Create temp output dir for save profiling
    save_tmpdir = tempfile.mkdtemp(prefix="piv_profile_save_")

    if do_warmup:
        print(f"\nWarmup pass (FFTW plan creation)...")
        t0 = time.perf_counter()
        correlator.correlate_batch(images, config)
        warmup_time = time.perf_counter() - t0
        print(f"  Warmup completed in {warmup_time:.2f}s")

    all_profiles = []
    runs_to_save = config.instantaneous_runs_0based

    for iteration in range(n_iterations):
        if n_iterations > 1:
            print(f"\nIteration {iteration + 1}/{n_iterations}...")
        else:
            print(f"\nRunning profiled correlation...")

        piv_results = correlator.correlate_batch(images, config)
        profile = correlator.get_profile_summary()

        # Profile save for each image in the batch
        n_passes = len(config.window_sizes)
        last_pass = n_passes - 1
        t_save_start = time.perf_counter()
        for i, piv_result in enumerate(piv_results):
            save_piv_result_distributed(
                piv_result, save_tmpdir, i + 1, runs_to_save,
                save_mode=config.instantaneous_save_mode,
                do_compression=config.instantaneous_save_compression,
            )
        t_save = time.perf_counter() - t_save_start
        # Attribute save time to the last pass (it saves all passes at once)
        profile.setdefault(last_pass, {})["save"] = t_save

        all_profiles.append(profile)

    # Clean up temp save files
    import shutil
    shutil.rmtree(save_tmpdir, ignore_errors=True)

    print_results(all_profiles, config, preset["label"])


def main():
    parser = argparse.ArgumentParser(
        description="Profile PIV processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/profile_piv.py 4mp
  python scripts/profile_piv.py 25mp --pairs 4 --iterations 5
  python scripts/profile_piv.py both --threads 8
  python scripts/profile_piv.py 4mp --windows 128,64,32 --no-outlier
        """,
    )
    parser.add_argument(
        "preset",
        choices=["4mp", "25mp", "both"],
        help="Image preset to use",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=1,
        help="Number of image pairs to process (default: 1)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of profiling iterations (default: 3)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="OMP thread count (default: 4)",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="128,64,32,16",
        help="Comma-separated square window sizes per pass (default: 128,64,32,16)",
    )
    parser.add_argument(
        "--no-outlier",
        action="store_true",
        help="Disable outlier detection",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip FFTW warmup pass",
    )
    parser.add_argument(
        "--infilling",
        type=str,
        default="local_median",
        help="Infilling method: local_median, biharmonic, etc. (default: local_median)",
    )
    parser.add_argument(
        "--peak-finder",
        type=str,
        default="gauss6",
        help="Peak finder: gauss3, gauss4, gauss5, gauss6 (default: gauss6)",
    )
    parser.add_argument(
        "--no-threading",
        action="store_true",
        help="Disable thread pool (direct sequential calls)",
    )
    parser.add_argument(
        "--save-mode",
        type=str,
        default="minimal",
        choices=["full", "minimal"],
        help="Save mode: full (11 fields) or minimal (ux, uy, b_mask only) (default: minimal)",
    )
    parser.add_argument(
        "--no-save-compression",
        action="store_true",
        default=True,
        help="Disable zlib compression on .mat save (default: disabled)",
    )

    args = parser.parse_args()

    # Parse window sizes
    win_sizes = [[int(w), int(w)] for w in args.windows.split(",")]
    overlaps = [50] * len(win_sizes)

    presets = ["4mp", "25mp"] if args.preset == "both" else [args.preset]

    for preset_name in presets:
        run_profile(
            preset_name=preset_name,
            n_pairs=args.pairs,
            n_iterations=args.iterations,
            omp_threads=args.threads,
            window_sizes=win_sizes,
            overlaps=overlaps,
            outlier_enabled=not args.no_outlier,
            do_warmup=not args.no_warmup,
            infilling_method=args.infilling,
            peak_finder=args.peak_finder,
            threading_enabled=not args.no_threading,
            save_mode=args.save_mode,
            save_compression=not args.no_save_compression,
        )


if __name__ == "__main__":
    main()
