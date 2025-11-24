import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from scipy.interpolate import UnivariateSpline # Added for precise FWHM

# ==========================================
# CONFIGURATION
# ==========================================

# Path to your images
IMAGE_DIR = '/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Planar_Images_with_wall/Cam1'

# Output directory for results
OUTPUT_DIR = 'PIV_Method_Comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image settings
NUM_IMAGES = 1000
CROP_SIZE = 128 # Size of the central crop (128x128)

# Filename pattern: B00001_A.tif / B00001_B.tif
get_filename = lambda i, pair: f"B{i:05d}_{pair}.tif" 

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def read_center_crop(idx, crop_size):
    """Reads image pair, converts to float, and crops the center."""
    try:
        path_a = os.path.join(IMAGE_DIR, get_filename(idx, 'A'))
        path_b = os.path.join(IMAGE_DIR, get_filename(idx, 'B'))
        if not os.path.exists(path_b):
            path_b = os.path.join(IMAGE_DIR, get_filename(idx, 'b'))
            
        img_a = np.array(Image.open(path_a)).astype(np.float64)
        img_b = np.array(Image.open(path_b)).astype(np.float64)
        
        h, w = img_a.shape
        cy, cx = h // 2, w // 2
        hw = crop_size // 2
        
        crop_a = img_a[cy-hw:cy+hw, cx-hw:cx+hw]
        crop_b = img_b[cy-hw:cy+hw, cx-hw:cx+hw]
        
        return crop_a, crop_b
    except Exception as e:
        print(f"Error reading image {idx}: {e}")
        return None, None

def fft_correlate(im1, im2):
    """Performs standard FFT-based cross-correlation."""
    f1 = np.fft.fft2(im1)
    f2 = np.fft.fft2(im2)
    corr = np.fft.ifft2(f1 * np.conj(f2))
    return np.fft.fftshift(np.real(corr))

def measure_fwhm(data):
    """
    Calculates the Full Width at Half Maximum (FWHM) of the peak 
    using spline interpolation for sub-pixel accuracy.
    """
    # Find peak index
    idx = np.unravel_index(np.argmax(data, axis=None), data.shape)
    y_idx, x_idx = idx
    
    # Extract slices
    slice_x = data[y_idx, :]
    slice_y = data[:, x_idx]
    
    def get_width(slice_data):
        x = np.arange(len(slice_data))
        half_max = np.max(slice_data) / 2.0
        
        # Create a spline to find exactly where it crosses half_max
        spline = UnivariateSpline(x, slice_data - half_max, s=0)
        roots = spline.roots()
        
        # We need exactly 2 roots for a clean bell curve
        if len(roots) >= 2:
            # Take the two roots closest to the peak index
            peak_loc = np.argmax(slice_data)
            roots.sort()
            
            # Find root before peak
            left_roots = roots[roots < peak_loc]
            right_roots = roots[roots > peak_loc]
            
            if len(left_roots) > 0 and len(right_roots) > 0:
                return right_roots[0] - left_roots[-1]
                
        return 0.0 # Failed to find width

    fwhm_x = get_width(slice_x)
    fwhm_y = get_width(slice_y)
    
    return fwhm_x, fwhm_y

# ==========================================
# MAIN PROCESSING
# ==========================================

def run_comparison():
    print(f"Starting comparison on {NUM_IMAGES} images...")
    print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")
    
    sum_A = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float64)
    sum_B = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float64)
    sum_corr_raw = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float64)
    
    # ==========================================
    # PASS 1: Single Sweep
    # ==========================================
    print("\n--- Starting Sweep 1 (Loading + Method 2 Accumulation) ---")
    start_time = time.time()
    valid_count = 0
    
    for i in range(1, NUM_IMAGES + 1):
        crop_a, crop_b = read_center_crop(i, CROP_SIZE)
        if crop_a is None: continue
        
        sum_A += crop_a
        sum_B += crop_b
        corr_raw = fft_correlate(crop_a, crop_b)
        sum_corr_raw += corr_raw
        valid_count += 1
        if i % 100 == 0: print(f"Processed {i}/{NUM_IMAGES}...")

    mean_A = sum_A / valid_count
    mean_B = sum_B / valid_count
    
    avg_corr_raw = sum_corr_raw / valid_count
    corr_of_means = fft_correlate(mean_A, mean_B)
    result_method_2 = avg_corr_raw - corr_of_means
    
    print(f"Sweep 1 finished in {time.time() - start_time:.2f}s")

    # ==========================================
    # PASS 2: Standard Method
    # ==========================================
    print("\n--- Starting Sweep 2 (Method 1: Two-Pass Logic) ---")
    sum_corr_fluct = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float64)
    
    for i in range(1, NUM_IMAGES + 1):
        crop_a, crop_b = read_center_crop(i, CROP_SIZE)
        if crop_a is None: continue
        
        fluct_a = crop_a - mean_A
        fluct_b = crop_b - mean_B
        corr_fluct = fft_correlate(fluct_a, fluct_b)
        sum_corr_fluct += corr_fluct

    result_method_1 = sum_corr_fluct / valid_count
    
    # ==========================================
    # SPREAD & SHAPE COMPARISON
    # ==========================================
    
    # Calculate Difference
    diff_plane = result_method_1 - result_method_2
    max_diff = np.max(np.abs(diff_plane))
    
    # Measure Spread (FWHM)
    fwhm_1_x, fwhm_1_y = measure_fwhm(result_method_1)
    fwhm_2_x, fwhm_2_y = measure_fwhm(result_method_2)
    
    print("\n--- Shape & Spread Analysis ---")
    print(f"{'Metric':<20} | {'Method 1 (Two-Pass)':<20} | {'Method 2 (Single-Pass)':<20} | {'Difference':<15}")
    print("-" * 85)
    print(f"{'Peak Value':<20} | {np.max(result_method_1):<20.6f} | {np.max(result_method_2):<20.6f} | {np.max(result_method_1)-np.max(result_method_2):.2e}")
    print(f"{'FWHM X (px)':<20} | {fwhm_1_x:<20.6f} | {fwhm_2_x:<20.6f} | {abs(fwhm_1_x - fwhm_2_x):.2e}")
    print(f"{'FWHM Y (px)':<20} | {fwhm_1_y:<20.6f} | {fwhm_2_y:<20.6f} | {abs(fwhm_1_y - fwhm_2_y):.2e}")
    print("-" * 85)
    print(f"\nMax Absolute Difference in Planes: {max_diff:.2e}")
    
    # PLOT
    mid = CROP_SIZE // 2
    plt.figure(figsize=(12, 6))
    
    # Plot normalized profiles to see shape only
    prof1 = result_method_1[mid, :] / np.max(result_method_1)
    prof2 = result_method_2[mid, :] / np.max(result_method_2)
    
    plt.plot(prof1, 'k-', linewidth=3, alpha=0.5, label='Method 1 (Normalized)')
    plt.plot(prof2, 'r--', linewidth=1.5, label='Method 2 (Normalized)')
    plt.plot((prof1 - prof2) * 1e5, 'g:', label='Difference x 100,000') 
    
    plt.title('Normalized Peak Shape Comparison (Check for Spread)')
    plt.xlabel('Pixel')
    plt.ylabel('Normalized Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'spread_comparison.png'), dpi=300)
    plt.close()
    
    print(f"\nSpread comparison plot saved to {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    run_comparison()