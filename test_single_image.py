"""
Test script to process a single image pair without Dask.
Prints all relevant array shapes to diagnose dimension issues.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import logging
import numpy as np
# Add src to path for unified imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import Config
from image_handling.load_images import read_pair, load_mask_for_camera

from pypivtools.piv.piv_backend.factory import make_correlator_backend

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def print_array_info(name, arr, indent=0):
    """Print array name and shape with optional indentation."""
    prefix = "  " * indent
    if arr is None:
        print(f"{prefix}{name}: None")
    elif isinstance(arr, (list, tuple)):
        print(f"{prefix}{name}: {type(arr).__name__} of length {len(arr)}")
        if len(arr) > 0 and isinstance(arr[0], np.ndarray):
            print(f"{prefix}  First element shape: {arr[0].shape}")
    else:
        print(f"{prefix}{name}: shape={arr.shape}, dtype={arr.dtype}")


def test_single_image():
    """Process a single image pair and print all relevant sizes."""
    
    print("=" * 80)
    print("SINGLE IMAGE PIV TEST - SHAPE DIAGNOSTICS")
    print("=" * 80)
    print()
    
    # Load configuration
    config = Config()
    print(f"Configuration loaded from: config.yaml")
    print(f"Image shape from config: {config.image_shape}")
    print(f"  - Height (Ny): {config.image_shape[0]}")
    print(f"  - Width (Nx): {config.image_shape[1]}")
    print()
    
    # Get paths
    camera_num = config.camera_numbers[0]
    source_path = config.source_paths[0]
    camera_path = source_path / f"Cam{camera_num}"
    
    print(f"Camera: Cam{camera_num}")
    print(f"Source path: {camera_path}")
    print()
    
    # Load a single image pair
    print("-" * 80)
    print("LOADING IMAGE PAIR")
    print("-" * 80)
    image_idx = 1
    image_pair = read_pair(image_idx, camera_path, config)
    print_array_info("image_pair", image_pair)
    print(f"  - Frame A shape: {image_pair[0].shape}")
    print(f"  - Frame B shape: {image_pair[1].shape}")
    print()
    
    # Add batch dimension: (1, 2, H, W)
    images = image_pair[np.newaxis, ...]
    print_array_info("images (with batch dimension)", images)
    print()
    
    # Load mask if enabled
    print("-" * 80)
    print("LOADING MASK")
    print("-" * 80)
    mask = load_mask_for_camera(camera_num, config, source_path_idx=0)
    print_array_info("mask", mask)
    if mask is not None:
        masked_pixels = np.sum(mask)
        total_pixels = mask.size
        print(f"  - Masked pixels: {masked_pixels}/{total_pixels} ({100*masked_pixels/total_pixels:.1f}%)")
    print()
    
    # Create correlator
    print("-" * 80)
    print("CREATING CORRELATOR")
    print("-" * 80)
    correlator = make_correlator_backend(config)
    print(f"Backend: {config.backend}")
    print(f"Correlator type: {type(correlator).__name__}")
    print()
    
    # Print PIV configuration
    print("-" * 80)
    print("PIV CONFIGURATION")
    print("-" * 80)
    print(f"Number of passes: {len(config.window_sizes)}")
    for pass_idx, win_size in enumerate(config.window_sizes):
        print(f"Pass {pass_idx}:")
        print(f"  - Window size: {win_size}")
        print(f"  - Overlap: {config.overlap[pass_idx]}%")
    print()
    
    # Inspect correlator's window center arrays
    print("-" * 80)
    print("WINDOW CENTERS (from correlator cache)")
    print("-" * 80)
    for pass_idx in range(len(config.window_sizes)):
        print(f"Pass {pass_idx}:")
        print(f"  - win_ctrs_x length: {len(correlator.win_ctrs_x[pass_idx])} (n_x)")
        print(f"  - win_ctrs_y length: {len(correlator.win_ctrs_y[pass_idx])} (n_y)")
        print(f"  - win_spacing_x: {correlator.win_spacing_x[pass_idx]}")
        print(f"  - win_spacing_y: {correlator.win_spacing_y[pass_idx]}")
        print(f"  - Total windows: {len(correlator.win_ctrs_x[pass_idx]) * len(correlator.win_ctrs_y[pass_idx])}")
        print()
    
    # Export win_ctrs_x and win_ctrs_y for pass 0
    if len(config.window_sizes) > 0:
        # Removed: np.savetxt('win_ctrs_x_pass0.csv', correlator.win_ctrs_x[0], delimiter=',')
        # Removed: np.savetxt('win_ctrs_y_pass0.csv', correlator.win_ctrs_y[0], delimiter=',')
        # Removed: print("Exported win_ctrs_x and win_ctrs_y for pass 0 to .csv files")
        
        # Compute and export edge_mask for pass 0
        EDGE_MARGIN = 64
        win_ctrs_x_grid, win_ctrs_y_grid = np.meshgrid(
            correlator.win_ctrs_x[0],
            correlator.win_ctrs_y[0],
        )
        edge_mask = (
            (win_ctrs_x_grid < EDGE_MARGIN) |
            (win_ctrs_x_grid > config.image_shape[1] - EDGE_MARGIN - 1) |
            (win_ctrs_y_grid < EDGE_MARGIN) |
            (win_ctrs_y_grid > config.image_shape[0] - EDGE_MARGIN - 1)
        )
        print(f"Edge mask shape: {edge_mask.shape}")
        # Removed: np.savetxt('edge_mask_pass0.csv', edge_mask.astype(int), delimiter=',')
        # Removed: print("Exported edge_mask for pass 0 to .csv file")
        print()
    
    # Run PIV
    print("-" * 80)
    print("RUNNING PIV")
    print("-" * 80)
    
    # Monkey-patch the correlator to print shapes during processing
    original_set_lib_arguments = correlator._set_lib_arguments
    
    def instrumented_set_lib_arguments(config, win_size, pass_idx):
        print(f"\n  Pass {pass_idx} - Setting library arguments:")
        result = original_set_lib_arguments(config, win_size, pass_idx)
        (
            win_size_out, n_windows, b_mask, n_peaks, i_peak_finder, 
            b_ensemble, pk_loc_x, pk_loc_y, pk_height, sx, sy, sxy, correl_plane_out
        ) = result
        
        print(f"    Window size: {win_size_out}")
        print(f"    n_windows: {n_windows} -> [n_x={n_windows[0]}, n_y={n_windows[1]}]")
        print_array_info("b_mask", b_mask, indent=2)
        print_array_info("pk_loc_x", pk_loc_x, indent=2)
        print_array_info("pk_loc_y", pk_loc_y, indent=2)
        print_array_info("pk_height", pk_height, indent=2)
        
        return result
    
    correlator._set_lib_arguments = instrumented_set_lib_arguments
    
    # Monkey-patch predictor_corrector to print shapes
    original_predictor = correlator._predictor_corrector
    
    def instrumented_predictor(pass_idx, image_a, image_b, interpolator="cubic", win_type="A"):
        print(f"\n  Pass {pass_idx} - Predictor-Corrector:")
        result = original_predictor(pass_idx, image_a, image_b, interpolator, win_type)
        image_a_prime, image_b_prime, delta_ab_pred = result
        
        print_array_info("image_a_prime", image_a_prime, indent=2)
        print_array_info("image_b_prime", image_b_prime, indent=2)
        print_array_info("delta_ab_pred", delta_ab_pred, indent=2)
        
        if hasattr(correlator, 'delta_ab_old') and correlator.delta_ab_old is not None:
            print_array_info("delta_ab_old", correlator.delta_ab_old, indent=2)
        
        return result
    
    correlator._predictor_corrector = instrumented_predictor
    
    try:
        piv_result = correlator.correlate_batch(images, config=config, mask=mask)
        
        print("\n" + "-" * 80)
        print("PIV RESULTS")
        print("-" * 80)
        print(f"Number of passes: {len(piv_result.passes)}")
        
        for pass_idx, pass_result in enumerate(piv_result.passes):
            print(f"\nPass {pass_idx}:")
            print_array_info("n_windows", pass_result.n_windows, indent=1)
            print_array_info("ux_mat", pass_result.ux_mat, indent=1)
            print_array_info("uy_mat", pass_result.uy_mat, indent=1)
            print_array_info("nan_mask", pass_result.nan_mask, indent=1)
            print_array_info("Q", pass_result.Q, indent=1)
            print_array_info("peak_mag", pass_result.peak_mag, indent=1)
            print_array_info("predictor_field", pass_result.predictor_field, indent=1)
            
            # Print statistics
            if pass_result.ux_mat is not None:
                valid_vectors = ~np.isnan(pass_result.ux_mat)
                n_valid = np.sum(valid_vectors)
                n_total = pass_result.ux_mat.size
                print(f"  Valid vectors: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
                
                if n_valid > 0:
                    print(f"  Ux range: [{np.nanmin(pass_result.ux_mat):.3f}, {np.nanmax(pass_result.ux_mat):.3f}]")
                    print(f"  Uy range: [{np.nanmin(pass_result.uy_mat):.3f}, {np.nanmax(pass_result.uy_mat):.3f}]")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR DURING PIV PROCESSING")
        print("=" * 80)
        print(f"Error: {e}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = test_single_image()
    sys.exit(exit_code)
