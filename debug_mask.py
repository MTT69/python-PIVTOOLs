#!/usr/bin/env python3
"""
Debug script to visualize and verify mask computation.

This script will:
1. Load the actual mask file
2. Visualize the pixel mask
3. Compute vector masks
4. Visualize vector masks for each pass
5. Compare with expected behavior
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from image_handling.load_images import compute_vector_mask, load_mask_for_camera


def visualize_pixel_mask(pixel_mask, title="Pixel Mask"):
    """Visualize the pixel mask."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Show mask (True = masked = white, False = valid = black)
    im = ax.imshow(pixel_mask, cmap='gray', origin='upper')
    ax.set_title(f"{title}\n{np.sum(pixel_mask)}/{pixel_mask.size} pixels masked ({100*np.sum(pixel_mask)/pixel_mask.size:.2f}%)")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    plt.colorbar(im, ax=ax, label="Masked (1=True, 0=False)")
    
    return fig


def visualize_vector_masks(vector_masks, config, pixel_mask):
    """Visualize vector masks for all passes."""
    n_passes = len(vector_masks)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for pass_idx, vmask in enumerate(vector_masks):
        if pass_idx >= len(axes):
            break
            
        ax = axes[pass_idx]
        
        # Show vector mask
        im = ax.imshow(vmask, cmap='RdYlGn_r', origin='upper', interpolation='nearest')
        
        win_y, win_x = config.window_sizes[pass_idx]
        masked_vecs = np.sum(vmask)
        total_vecs = vmask.size
        
        ax.set_title(f"Pass {pass_idx+1}: Window {win_y}x{win_x}\n"
                    f"{masked_vecs}/{total_vecs} masked ({100*masked_vecs/total_vecs:.1f}%)")
        ax.set_xlabel(f"Vector X (spacing ~{config.overlap[pass_idx]}% overlap)")
        ax.set_ylabel("Vector Y")
        plt.colorbar(im, ax=ax, label="Masked")
    
    # Hide unused subplots
    for idx in range(n_passes, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_mask_overlay(pixel_mask, vector_masks, config):
    """Overlay vector mask on pixel mask to verify alignment."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for pass_idx, vmask in enumerate(vector_masks):
        if pass_idx >= len(axes):
            break
            
        ax = axes[pass_idx]
        
        # Show pixel mask as background
        ax.imshow(pixel_mask, cmap='gray', alpha=0.3, origin='upper')
        
        # Compute window centers (matching the mask computation)
        win_y, win_x = config.window_sizes[pass_idx]
        overlap = config.overlap[pass_idx]
        
        win_spacing_x = round((1 - overlap / 100) * win_x)
        win_spacing_y = round((1 - overlap / 100) * win_y)
        
        H, W = pixel_mask.shape
        start_x = win_x / 2 - 0.5
        start_y = win_y / 2 - 0.5
        
        EDGE_MARGIN = 32
        start_x = max(start_x, EDGE_MARGIN)
        start_y = max(start_y, EDGE_MARGIN)
        
        max_x = W - EDGE_MARGIN - 1
        max_y = H - EDGE_MARGIN - 1
        
        n_win_x = int(np.floor((max_x - start_x) / win_spacing_x)) + 1
        n_win_y = int(np.floor((max_y - start_y) / win_spacing_y)) + 1
        
        win_ctrs_x = np.linspace(start_x, start_x + win_spacing_x * (n_win_x - 1), n_win_x)
        win_ctrs_y = np.linspace(start_y, start_y + win_spacing_y * (n_win_y - 1), n_win_y)
        
        # Plot window centers, colored by mask status
        for iy, cy in enumerate(win_ctrs_y):
            for ix, cx in enumerate(win_ctrs_x):
                is_masked = vmask[iy, ix]
                color = 'red' if is_masked else 'green'
                marker = 'x' if is_masked else '.'
                ax.plot(cx, cy, marker, color=color, markersize=4, alpha=0.7)
        
        ax.set_title(f"Pass {pass_idx+1}: Window {win_y}x{win_x}")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # Invert y-axis to match image coordinates
    
    # Hide unused subplots
    for idx in range(len(vector_masks), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def debug_mask_computation(pixel_mask, config):
    """Debug the mask computation step by step for first pass."""
    print("\n" + "="*60)
    print("DETAILED MASK COMPUTATION DEBUG - Pass 1")
    print("="*60)
    
    pass_idx = 0
    win_y, win_x = config.window_sizes[pass_idx]
    overlap = config.overlap[pass_idx]
    H, W = pixel_mask.shape
    
    print(f"\nImage shape: {H} x {W}")
    print(f"Window size: {win_y} x {win_x}")
    print(f"Overlap: {overlap}%")
    print(f"Mask threshold: {config.mask_threshold}")
    
    # Calculate spacing
    win_spacing_x = round((1 - overlap / 100) * win_x)
    win_spacing_y = round((1 - overlap / 100) * win_y)
    print(f"Window spacing: {win_spacing_y} x {win_spacing_x}")
    
    # Pixel mask statistics
    masked_pixels = np.sum(pixel_mask)
    total_pixels = pixel_mask.size
    print(f"\nPixel mask: {masked_pixels}/{total_pixels} masked ({100*masked_pixels/total_pixels:.2f}%)")
    
    # Perform convolution
    from scipy.ndimage import convolve
    
    im_mask = pixel_mask.astype(np.float32)
    box_filter_y = np.ones((win_y, 1), dtype=np.float32) / win_y
    f_mask = convolve(im_mask, box_filter_y, mode='constant', cval=0.0)
    
    box_filter_x = np.ones((1, win_x), dtype=np.float32) / win_x
    f_mask = convolve(f_mask, box_filter_x, mode='constant', cval=0.0)
    
    print(f"\nFiltered mask statistics:")
    print(f"  Min: {np.min(f_mask):.4f}")
    print(f"  Max: {np.max(f_mask):.4f}")
    print(f"  Mean: {np.mean(f_mask):.4f}")
    print(f"  Values > threshold ({config.mask_threshold}): {np.sum(f_mask > config.mask_threshold)}/{f_mask.size}")
    
    # Show some sample values
    print(f"\nSample filtered mask values (center region):")
    cy, cx = H // 2, W // 2
    sample = f_mask[cy-5:cy+5, cx-5:cx+5]
    print(sample)
    
    # Calculate window centers
    start_x = win_x / 2 - 0.5
    start_y = win_y / 2 - 0.5
    
    EDGE_MARGIN = 32
    start_x = max(start_x, EDGE_MARGIN)
    start_y = max(start_y, EDGE_MARGIN)
    
    max_x = W - EDGE_MARGIN - 1
    max_y = H - EDGE_MARGIN - 1
    
    n_win_x = int(np.floor((max_x - start_x) / win_spacing_x)) + 1
    n_win_y = int(np.floor((max_y - start_y) / win_spacing_y)) + 1
    
    print(f"\nWindow centers:")
    print(f"  Number of windows: {n_win_y} x {n_win_x} = {n_win_y * n_win_x}")
    print(f"  Start position: ({start_y:.1f}, {start_x:.1f})")
    
    win_ctrs_x = np.linspace(start_x, start_x + win_spacing_x * (n_win_x - 1), n_win_x)
    win_ctrs_y = np.linspace(start_y, start_y + win_spacing_y * (n_win_y - 1), n_win_y)
    
    # Sample at window centers
    win_y_grid, win_x_grid = np.meshgrid(win_ctrs_y, win_ctrs_x, indexing='ij')
    win_y_idx = np.clip(np.round(win_y_grid).astype(int), 0, H - 1)
    win_x_idx = np.clip(np.round(win_x_grid).astype(int), 0, W - 1)
    
    sampled_values = f_mask[win_y_idx, win_x_idx]
    b_mask_pass = sampled_values > config.mask_threshold
    
    print(f"\nSampled values at window centers:")
    print(f"  Min: {np.min(sampled_values):.4f}")
    print(f"  Max: {np.max(sampled_values):.4f}")
    print(f"  Mean: {np.mean(sampled_values):.4f}")
    print(f"  Values > threshold: {np.sum(b_mask_pass)}/{b_mask_pass.size}")
    
    print(f"\nFinal vector mask: {np.sum(b_mask_pass)}/{b_mask_pass.size} masked ({100*np.sum(b_mask_pass)/b_mask_pass.size:.1f}%)")
    
    return f_mask, sampled_values, b_mask_pass


def main():
    """Main debug routine."""
    print("\n" + "="*60)
    print("MASK COMPUTATION DEBUG TOOL")
    print("="*60)
    
    config = Config()
    
    # Try to load real mask
    camera_num = config.camera_numbers[0] if config.camera_numbers else 1
    print(f"\nLoading mask for Cam{camera_num}...")
    
    pixel_mask = load_mask_for_camera(camera_num, config)
    
    if pixel_mask is None:
        print("\n⚠ No mask file found. Creating synthetic mask for testing...")
        H, W = config.image_shape
        pixel_mask = np.zeros((H, W), dtype=bool)
        
        # Create mask matching real data: top and bottom 64 pixels
        mask_height = 64
        pixel_mask[:mask_height, :] = True  # Top 64 pixels
        pixel_mask[-mask_height:, :] = True  # Bottom 64 pixels
        
        print(f"Created synthetic mask: top and bottom {mask_height} pixels")
        print(f"Total masked: {np.sum(pixel_mask)}/{pixel_mask.size} pixels ({100*np.sum(pixel_mask)/pixel_mask.size:.1f}%)")
    
    # Visualize pixel mask
    print("\nGenerating pixel mask visualization...")
    fig1 = visualize_pixel_mask(pixel_mask, "Loaded Pixel Mask")
    fig1.savefig("debug_pixel_mask.png", dpi=150, bbox_inches='tight')
    print("Saved: debug_pixel_mask.png")
    
    # Compute vector masks
    print("\nComputing vector masks...")
    vector_masks = compute_vector_mask(pixel_mask, config)
    
    # Detailed debug for first pass
    f_mask, sampled, b_mask = debug_mask_computation(pixel_mask, config)
    
    # Visualize vector masks
    print("\nGenerating vector mask visualizations...")
    fig2 = visualize_vector_masks(vector_masks, config, pixel_mask)
    fig2.savefig("debug_vector_masks.png", dpi=150, bbox_inches='tight')
    print("Saved: debug_vector_masks.png")
    
    # Visualize overlay
    print("\nGenerating overlay visualization...")
    fig3 = visualize_mask_overlay(pixel_mask, vector_masks, config)
    fig3.savefig("debug_mask_overlay.png", dpi=150, bbox_inches='tight')
    print("Saved: debug_mask_overlay.png")
    
    # Create filtered mask visualization
    fig4, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(pixel_mask, cmap='gray', origin='upper')
    axes[0].set_title("Original Pixel Mask")
    
    axes[1].imshow(f_mask, cmap='hot', origin='upper')
    axes[1].set_title(f"Filtered Mask (Pass 1)\nWindow {config.window_sizes[0][0]}x{config.window_sizes[0][1]}")
    plt.colorbar(axes[1].images[0], ax=axes[1], label="Fraction masked")
    
    axes[2].imshow(b_mask, cmap='RdYlGn_r', origin='upper')
    axes[2].set_title(f"Binary Vector Mask (threshold={config.mask_threshold})")
    
    plt.tight_layout()
    fig4.savefig("debug_filtering_process.png", dpi=150, bbox_inches='tight')
    print("Saved: debug_filtering_process.png")
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - debug_pixel_mask.png")
    print("  - debug_vector_masks.png")
    print("  - debug_mask_overlay.png")
    print("  - debug_filtering_process.png")
    print("\nCheck these files to verify mask computation is correct.")
    
    plt.show()


if __name__ == "__main__":
    main()
