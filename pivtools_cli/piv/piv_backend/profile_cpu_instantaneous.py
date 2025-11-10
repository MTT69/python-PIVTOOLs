import cProfile
import io
import pstats
import time
from pathlib import Path
import sys
import numpy as np

# Add src to path for unified imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from config import Config
from image_handling.load_images import load_images, load_mask_for_camera
from pivtools_cli.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU


def profile_cpu_instantaneous():
    """
    Profile the CPU instantaneous correlator with comprehensive timing breakdown.
    Runs 10 images: first one warms up caches, next 9 are profiled.
    Provides timing info per pass, memory usage, and detailed profiling.
    """
    overall_start = time.perf_counter()

    config = Config()

    # Use first camera and source path
    camera_num = config.camera_numbers[0]
    source_path = config.source_paths[0]

    print("=" * 60)
    print("COMPREHENSIVE CPU INSTANTANEOUS CORRELATOR PROFILING")
    print("=" * 60)

    # PHASE 1: Data Loading
    print("\n[PHASE 1] Loading images...")
    phase_start = time.perf_counter()
    all_images = load_images(camera_num, config, source=source_path)

    # Handle color images: convert to grayscale if needed
    if len(all_images.shape) == 5:
        print("Detected color images, converting to grayscale...")
        import dask.array as da
        all_images = da.mean(all_images, axis=-1, dtype=np.float32)

    # all_images is now (num_images, 2, H, W)
    num_images = min(10, all_images.shape[0])
    images = all_images[:num_images]

    print(f"Slicing {num_images} image pairs from dask array")

    # CRITICAL FIX: Compute dask array to numpy array BEFORE profiling
    # This matches the real workflow where image_block.compute() is called in _piv_single_pass
    print("Computing dask array to numpy (this may take a moment)...")
    start_compute = time.perf_counter()
    images = images.compute()  # Convert from dask to numpy array
    compute_time = time.perf_counter() - start_compute
    print(".2f")
    print(f"Images shape: {images.shape}, dtype: {images.dtype}")

    data_loading_time = time.perf_counter() - phase_start
    print(".2f")

    # PHASE 2: Mask Loading
    print("\n[PHASE 2] Loading masks...")
    phase_start = time.perf_counter()
    vector_masks = None
    if config.masking_enabled:
        mask = load_mask_for_camera(camera_num, config, source_path_idx=0)
        if mask is not None:
            from image_handling.load_images import compute_vector_mask
            vector_masks = compute_vector_mask(mask, config)
            print("Loaded mask and computed vector masks")
        else:
            print("Masking enabled but no mask found")
    else:
        print("Masking disabled")

    mask_loading_time = time.perf_counter() - phase_start
    print(".2f")

    # PHASE 3: Correlator Initialization
    print("\n[PHASE 3] Initializing correlator...")
    phase_start = time.perf_counter()
    correlator = InstantaneousCorrelatorCPU(config)
    correlator_init_time = time.perf_counter() - phase_start
    print(".2f")

    # PHASE 4: Warmup
    print("\n[PHASE 4] Warming up caches with first image...")
    phase_start = time.perf_counter()
    warmup_result = correlator.correlate_batch(images[:1], config, vector_masks)
    warmup_time = time.perf_counter() - phase_start
    print(".2f")

    # PHASE 5: Profiling
    print(f"\n[PHASE 5] Profiling correlation on {num_images-1} images...")
    images_to_profile = images[1:num_images]  # Take remaining images
    num_to_profile = len(images_to_profile)

    if num_to_profile == 0:
        print("Not enough images to profile (need at least 2 total)")
        return

    print(f"Profiling {num_to_profile} images...")

    # Use cProfile for detailed line-by-line profiling
    pr = cProfile.Profile()

    def run_correlation_batch():
        results = []
        all_pass_times = []
        global_img_idx = 1  # Start from 1 since 0 is warmup
        for i, img_pair in enumerate(images_to_profile):
            # Removed print statement for cleaner output
            result = correlator.correlate_batch(img_pair[np.newaxis], config, vector_masks)
            # Extend with global image index
            for pass_data in correlator.pass_times:
                all_pass_times.append((global_img_idx, pass_data[1], pass_data[2]))
            global_img_idx += 1
            results.append(result)
        return results, all_pass_times

    # Run with profiling
    pr.enable()
    start_profile = time.perf_counter()
    results, all_pass_times = run_correlation_batch()
    end_profile = time.perf_counter()
    pr.disable()

    total_profile_time = end_profile - start_profile
    avg_time_per_image = total_profile_time / num_to_profile

    print(".2f")
    print(".4f")

    # PHASE 6: Analysis and Output
    print("\n[PHASE 6] Analysis and Output...")

    # Comprehensive timing summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TIMING SUMMARY")
    print("=" * 60)
    print(f"{'Phase':<25} {'Time (s)':<10} {'% of Total':<10}")
    print("-" * 60)

    total_time = data_loading_time + mask_loading_time + correlator_init_time + warmup_time + total_profile_time
    phases = [
        ("Data Loading", data_loading_time),
        ("Mask Loading", mask_loading_time),
        ("Correlator Init", correlator_init_time),
        ("Warmup", warmup_time),
        ("Profiling", total_profile_time),
    ]

    for phase_name, phase_time in phases:
        pct = (phase_time / total_time * 100) if total_time > 0 else 0
        print(f"{phase_name:<25} {phase_time:<10.2f} {pct:<10.1f}")

    print("-" * 60)
    print(f"{'TOTAL TIME':<25} {total_time:<10.2f} {'100.0':<10}")
    print(f"{'Avg time per image':<25} {avg_time_per_image:<10.4f}")

    # Write line profiling results to file
    print("\nWriting detailed profiling results...")
    profile_txt_path = Path(__file__).parent / "profile_results.txt"
    with open(profile_txt_path, 'w') as f:
        f.write("Detailed profiling statistics (top 20 functions by cumulative time):\n")
        ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
        ps.print_stats(20)
        f.write("\nTop 20 functions by total time:\n")
        ps2 = pstats.Stats(pr, stream=f).sort_stats('time')
        ps2.print_stats(20)

    print(f"Line profiling results saved to: {profile_txt_path}")
    
    # Collect and analyze per-pass times (excluding warmup)
    pass_times_data = all_pass_times
    print(f"Collected {len(pass_times_data)} pass timing records")
    if pass_times_data:
        print(f"Sample record: {pass_times_data[0] if pass_times_data else 'None'}")
        # Filter out warmup (image 0) - but since we start from 1, all are valid
        profiled_times = [(pass_idx, time_val) for img_idx, pass_idx, time_val in pass_times_data]
        print(f"After filtering warmup: {len(profiled_times)} records")
        if profiled_times:
            print(f"Sample filtered: {profiled_times[0]}")
        # Filter out warmup (image 0)
        profiled_times = [(pass_idx, time_val) for img_idx, pass_idx, time_val in pass_times_data if img_idx > 0]
        
        # Group by pass and compute averages
        from collections import defaultdict
        pass_stats = defaultdict(list)
        for pass_idx, time_val in profiled_times:
            pass_stats[pass_idx].append(time_val)
        
        # Compute averages and sort by time descending
        avg_times = []
        for pass_idx, times in pass_stats.items():
            avg_time = sum(times) / len(times)
            avg_times.append((pass_idx, avg_time, len(times)))
        
        avg_times.sort(key=lambda x: x[1], reverse=True)  # Sort by average time descending
        
        # Write per-pass averages to file
        pass_avg_txt_path = Path(__file__).parent / "pass_averages.txt"
        with open(pass_avg_txt_path, 'w') as f:
            f.write("Per-Pass Average Times (excluding warmup, ranked by longest time)\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Pass':<5} {'Avg Time (s)':<12} {'Samples':<8} {'% of Total':<10}\n")
            f.write("-" * 60 + "\n")
            
            total_avg_time = sum(avg for _, avg, _ in avg_times)
            for pass_idx, avg_time, count in avg_times:
                pct = (avg_time / total_avg_time * 100) if total_avg_time > 0 else 0
                f.write(f"{pass_idx:<5} {avg_time:<12.4f} {count:<8} {pct:<10.1f}%\n")
        
        print(f"Per-pass averages saved to: {pass_avg_txt_path}")
    
    try:
        import line_profiler
        print("\nFor detailed line-by-line profiling, run:")
        print("kernprof -l -v profile_cpu_instantaneous.py")
    except ImportError:
        print("\nFor detailed line-by-line profiling, install line_profiler:")
        print("pip install line_profiler")
        print("Then run: kernprof -l -v profile_cpu_instantaneous.py")


if __name__ == "__main__":
    profile_cpu_instantaneous()