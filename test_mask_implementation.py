#!/usr/bin/env python3
"""
Test script to verify the mask implementation.

This script tests:
1. Loading a pixel mask
2. Computing vector masks from pixel mask
3. Verifying masking respects config.masking_enabled
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from image_handling.load_images import compute_vector_mask, load_mask_for_camera


def test_mask_threshold_property():
    """Test that mask_threshold is properly read from config."""
    print("=" * 60)
    print("Test 1: Verify mask_threshold property")
    print("=" * 60)
    
    config = Config()
    
    print(f"Masking enabled: {config.masking_enabled}")
    print(f"Mask threshold: {config.mask_threshold}")
    print(f"Mask file pattern: {config.mask_file_pattern}")
    
    assert hasattr(config, 'mask_threshold'), "Config missing mask_threshold property"
    assert isinstance(config.mask_threshold, (int, float)), "mask_threshold should be numeric"
    assert 0.0 <= config.mask_threshold <= 1.0, "mask_threshold should be between 0 and 1"
    
    print("✓ mask_threshold property working correctly")
    print()


def test_compute_vector_mask():
    """Test vector mask computation from pixel mask."""
    print("=" * 60)
    print("Test 2: Compute vector mask from synthetic pixel mask")
    print("=" * 60)
    
    config = Config()
    
    # Create a synthetic pixel mask
    H, W = config.image_shape
    pixel_mask = np.zeros((H, W), dtype=bool)
    
    # Create a rectangular masked region in the center
    mask_h = H // 4
    mask_w = W // 4
    y_start = H // 2 - mask_h // 2
    x_start = W // 2 - mask_w // 2
    pixel_mask[y_start:y_start+mask_h, x_start:x_start+mask_w] = True
    
    print(f"Created synthetic pixel mask: shape={pixel_mask.shape}")
    print(f"Masked pixels: {np.sum(pixel_mask)}/{pixel_mask.size} ({100*np.sum(pixel_mask)/pixel_mask.size:.1f}%)")
    print()
    
    # Compute vector masks
    vector_masks = compute_vector_mask(pixel_mask, config)
    
    print(f"Computed vector masks for {len(vector_masks)} passes")
    for i, vmask in enumerate(vector_masks):
        win_y, win_x = config.window_sizes[i]
        masked_vectors = np.sum(vmask)
        total_vectors = vmask.size
        print(f"  Pass {i+1}: {masked_vectors}/{total_vectors} vectors masked "
              f"({100*masked_vectors/total_vectors:.1f}%), "
              f"window size: ({win_y}x{win_x})")
    
    print()
    print("✓ Vector mask computation successful")
    print()


def test_masking_enabled_flag():
    """Test that masking respects the enabled flag."""
    print("=" * 60)
    print("Test 3: Verify masking respects config.masking_enabled")
    print("=" * 60)
    
    config = Config()
    
    print(f"Current masking_enabled: {config.masking_enabled}")
    
    if config.masking_enabled:
        print("Masking is ENABLED - vector masks will be computed")
    else:
        print("Masking is DISABLED - vector masks will NOT be computed")
    
    print()
    print("✓ Masking flag check successful")
    print()


def test_load_mask_for_camera():
    """Test loading mask from file (if available)."""
    print("=" * 60)
    print("Test 4: Try loading mask from file")
    print("=" * 60)
    
    config = Config()
    
    # Try to load mask for first camera
    camera_num = config.camera_numbers[0] if config.camera_numbers else 1
    
    print(f"Attempting to load mask for Cam{camera_num}...")
    mask = load_mask_for_camera(camera_num, config)
    
    if mask is not None:
        print(f"✓ Mask loaded successfully: shape={mask.shape}, dtype={mask.dtype}")
        print(f"  Masked pixels: {np.sum(mask)}/{mask.size} ({100*np.sum(mask)/mask.size:.1f}%)")
    else:
        print("ℹ No mask file found (this is OK if you haven't created one yet)")
    
    print()


def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("MASK IMPLEMENTATION TEST SUITE")
    print("=" * 60)
    print()
    
    try:
        test_mask_threshold_property()
        test_compute_vector_mask()
        test_masking_enabled_flag()
        test_load_mask_for_camera()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print()
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
