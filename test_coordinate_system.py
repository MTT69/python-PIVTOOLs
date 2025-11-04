"""
Simple test to verify C library coordinate system after rebuild
"""
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Set up minimal config
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from pypivtools.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU

# Create simple test images
H, W = 256, 256
test_images = np.zeros((1, 2, H, W), dtype=np.float32)

# Create a simple pattern in image A
test_images[0, 0, 100:150, 100:150] = 1.0

# Shift it by 5 pixels in X and 3 pixels in Y for image B
test_images[0, 1, 103:153, 105:155] = 1.0

# Minimal config
config = Config()
config.image_shape = (H, W)
config.window_sizes = [[64, 64]]  # Single pass, 64x64 window
config.overlap = [50]  # 50% overlap
config.window_type = "A"
config.num_peaks = 1
config.peak_finder = 6
config.outlier_detection = False
config.secondary_peak = False
config.ensemble_piv = False
config.debug = True

print(f"Test config:")
print(f"  Image shape: {config.image_shape}")
print(f"  Window size: {config.window_sizes[0]}")
print(f"  Expected displacement: X=+5, Y=+3")
print()

try:
    correlator = InstantaneousCorrelatorCPU(config)
    result = correlator.correlate_batch(test_images, config, vector_masks=None)
    
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    
    if result.passes:
        pass_result = result.passes[0]
        ux = pass_result.ux_mat
        uy = pass_result.uy_mat
        
        # Find valid (non-NaN) vectors
        valid = ~np.isnan(ux)
        if valid.any():
            median_ux = np.median(ux[valid])
            median_uy = np.median(uy[valid])
            print(f"Median displacement: X={median_ux:.2f}, Y={median_uy:.2f}")
            print(f"Expected: X=+5.0, Y=+3.0")
            print(f"Error: X={abs(median_ux - 5.0):.2f}, Y={abs(median_uy - 3.0):.2f}")
            
            if abs(median_ux - 5.0) < 0.5 and abs(median_uy - 3.0) < 0.5:
                print("\n✓ COORDINATE SYSTEM CORRECT!")
            else:
                print("\n✗ COORDINATE SYSTEM ERROR - Check indexing")
        else:
            print("No valid vectors computed!")
    else:
        print("No pass results!")
        
except Exception as e:
    print(f"\n✗ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
