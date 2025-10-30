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
from pypivtools.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU


def profile_cpu_instantaneous():
    """
    Profile the CPU instantaneous correlator with line-by-line timing breakdown.
    Runs 10 images: first one warms up caches, next 9 are profiled.
    Provides timing info per pass and overall.
    """
    config = Config()

    # Use first camera and source path
    camera_num = config.camera_numbers[0]
    source_path = config.source_paths[0]

    print("Loading images...")
    # Load images (assuming we have at least 10 image pairs)
    all_images = load_images(camera_num, config, source=source_path)

    # Handle color images: convert to grayscale if needed
    if len(all_images.shape) == 5:
        print("Detected color images, converting to grayscale...")
        import dask.array as da
        all_images = da.mean(all_images, axis=-1, dtype=np.float32)

    # all_images is now (num_images, 2, H, W)
    num_images = min(10, all_images.shape[0])
    images = all_images[:num_images]

    print(f"Loaded {num_images} image pairs")

    # Load mask and compute vector masks if enabled
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

    # Instantiate correlator
    correlator = InstantaneousCorrelatorCPU(config)
    print("Instantiated correlator")

    # Warm up caches with first image
    print("\nWarming up caches with first image...")
    start_warmup = time.perf_counter()
    warmup_result = correlator.correlate_batch(images[:1], config, vector_masks)
    warmup_time = time.perf_counter() - start_warmup
    print(".2f")

    # Now profile the remaining 9 images
    images_to_profile = images[1:10]  # Take next 9, or fewer if not available
    num_to_profile = len(images_to_profile)

    if num_to_profile == 0:
        print("Not enough images to profile (need at least 2 total)")
        return

    print(f"\nProfiling {num_to_profile} images...")

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

    # Write line profiling results to file
    print("Profile complete.")
    
    # Write line profiling results to file
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