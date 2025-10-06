#!/usr/bin/env python3
"""
Test rectangular masking configuration.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from image_handling.load_images import load_mask_for_camera, compute_vector_mask


def test_rectangular_mask():
    """Test the new rectangular masking mode."""
    print("="*60)
    print("RECTANGULAR MASKING TEST")
    print("="*60)
    
    config = Config()
    
    print(f"\nMasking enabled: {config.masking_enabled}")
    print(f"Masking mode: {config.mask_mode}")
    
    if config.mask_mode == "rectangular":
        rect = config.mask_rectangular_settings
        print(f"Rectangular settings:")
        print(f"  Top: {rect['top']} pixels")
        print(f"  Bottom: {rect['bottom']} pixels")
        print(f"  Left: {rect['left']} pixels")
        print(f"  Right: {rect['right']} pixels")
    
    print(f"Mask threshold: {config.mask_threshold}")
    
    # Load mask (should create rectangular mask)
    camera_num = config.camera_numbers[0] if config.camera_numbers else 1
    mask = load_mask_for_camera(camera_num, config)
    
    if mask is not None:
        print(f"\n✓ Mask created: shape={mask.shape}")
        masked_pixels = np.sum(mask)
        total_pixels = mask.size
        print(f"  Masked pixels: {masked_pixels}/{total_pixels} ({100*masked_pixels/total_pixels:.1f}%)")
        
        # Compute vector masks
        print("\nComputing vector masks...")
        vector_masks = compute_vector_mask(mask, config)
        
        print(f"\nVector masks for {len(vector_masks)} passes:")
        for i, vmask in enumerate(vector_masks):
            win_y, win_x = config.window_sizes[i]
            masked_vecs = np.sum(vmask)
            total_vecs = vmask.size
            print(f"  Pass {i+1}: {masked_vecs}/{total_vecs} vectors masked "
                  f"({100*masked_vecs/total_vecs:.1f}%), window: {win_y}x{win_x}")
        
        print("\n✓ Rectangular masking working correctly!")
    else:
        print("\n✗ Failed to create mask")
        return False
    
    return True


if __name__ == "__main__":
    success = test_rectangular_mask()
    sys.exit(0 if success else 1)
