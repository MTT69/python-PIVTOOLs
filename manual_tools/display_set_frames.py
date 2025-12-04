#!/usr/bin/env python3
"""
Manual tool to open and display LaVision .set files - FRAME PAIR FORMAT.
For double-frame PIV data where each camera has A/B frame pairs.
Structure: set_file[image_no].frames[2*camera + frame_idx]

Example: 2 cameras, 100 image pairs
  - set_file[0].frames[0] = Camera 0, Frame A, Image 0
  - set_file[0].frames[1] = Camera 0, Frame B, Image 0
  - set_file[0].frames[2] = Camera 1, Frame A, Image 0
  - set_file[0].frames[3] = Camera 1, Frame B, Image 0
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal

def read_lavision_set_frame_pair(set_file, camera_idx: int, im_no: int) -> np.ndarray:
    """
    Read a frame pair from a double-frame LaVision .set file.

    Args:
        set_file: Already opened set file object
        camera_idx: Camera index (0-based)
        im_no: Image number (0-based index)

    Returns:
        np.ndarray: Array of shape (2, H, W) containing frame A and B
    """
    im = set_file[im_no]
    
    # Extract frames for this camera (A/B pairs)
    data = np.zeros((2, *im.frames[0].components["PIXEL"].planes[0].shape), dtype=np.float64)

    for j in range(2):
        # Frame indexing: 2*camera_idx + j (0=A, 1=B)
        frame_idx = 2 * camera_idx + j
        frame = im.frames[frame_idx]

        # Apply scaling
        i_scale = frame.scales.i.slope
        i_offset = frame.scales.i.offset
        u_arr = frame.components["PIXEL"].planes[0] * i_scale + i_offset

        data[j, :, :] = u_arr

    return data.astype(np.float32)

def main():
    try:
        import lvpyio as lv
    except ImportError:
        print("Error: lvpyio library not available. Please install it with: pip install lvpyio")
        sys.exit(1)

    # Hardcoded path for this specific request
    set_file_path = r"C:\Users\mtt1e23\Downloads\prismTR\a FreeSt_Cyl__PLIF_STB_33Hz_L1_75_Att100_Cs50_Qs10_P10_Run_6.set"

    print(f"Attempting to open: {set_file_path}")

    if not Path(set_file_path).exists():
        print(f"Error: Set file not found: {set_file_path}")
        sys.exit(1)

    try:
        # Read the set file
        set_file = lv.read_set(set_file_path)
        num_images = len(set_file)
        print(f"Set file contains {num_images} images")
        
        # Inspect structure
        print("\nInspecting set file structure:")
        if num_images > 0:
            first_im = set_file[0]
            num_frames_per_image = len(first_im.frames)
            num_cameras = num_frames_per_image // 2
            print(f"Each image has {num_frames_per_image} frames")
            print(f"Number of cameras (assuming A/B pairs): {num_cameras}")
            print(f"Total images: {num_images}")
            
            print(f"\nFrame details for first image:")
            for frame_idx in range(num_frames_per_image):
                frame = first_im.frames[frame_idx]
                shape = frame.components["PIXEL"].planes[0].shape
                cam = frame_idx // 2
                ab = "A" if frame_idx % 2 == 0 else "B"
                print(f"  Frame {frame_idx} (Cam {cam}, {ab}): shape {shape}")
                if hasattr(frame, 'scales'):
                    print(f"    Scales: i.slope={frame.scales.i.slope}, i.offset={frame.scales.i.offset}")
        
    except Exception as e:
        print(f"Error reading set file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============ USER SETTINGS ============
    camera_to_display = 2  # Which camera to display (0-indexed)
    num_frames_to_display = 5  # How many image pairs to show
    corr_size = 128  # Size of central region for cross-correlation
    # =========================================
    
    if camera_to_display >= num_cameras:
        print(f"Error: Camera {camera_to_display} does not exist. Max camera index is {num_cameras - 1}")
        sys.exit(1)
    
    num_frames_to_display = min(num_frames_to_display, num_images)
    
    # Create subplot grid (2 rows: Frame A images and A vs B correlations)
    fig, axes = plt.subplots(2, num_frames_to_display, figsize=(4 * num_frames_to_display, 8))
    
    # Handle single frame case
    if num_frames_to_display == 1:
        axes = axes.reshape(2, 1)

    # Read and display each frame pair for the selected camera
    for i in range(num_frames_to_display):
        try:
            frames = read_lavision_set_frame_pair(set_file, camera_idx=camera_to_display, im_no=i)
            frame_a = frames[0]
            frame_b = frames[1]
            
            # Display Frame A
            vmin, vmax = np.percentile(frame_a, [1, 99])
            axes[0, i].imshow(frame_a, cmap='gray', vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f'Im {i}, Frame A')
            axes[0, i].axis('off')
            
            # Add red box showing correlation region
            h, w = frame_a.shape
            cy, cx = h // 2, w // 2
            half = corr_size // 2
            rect = Rectangle((cx - half, cy - half), corr_size, corr_size, 
                            linewidth=2, edgecolor='red', facecolor='none')
            axes[0, i].add_patch(rect)
            
            # Extract central region for correlation
            region_a = frame_a[cy-half:cy+half, cx-half:cx+half]
            region_b = frame_b[cy-half:cy+half, cx-half:cx+half]
            
            # Normalize regions
            region_a = region_a - np.mean(region_a)
            region_b = region_b - np.mean(region_b)
            
            # Cross-correlation A vs B using FFT
            corr = signal.correlate2d(region_a, region_b, mode='full')
            
            axes[1, i].imshow(corr, cmap='jet')
            axes[1, i].set_title(f'Corr A vs B (Im {i})')
            axes[1, i].axis('off')
            
            print(f"Loaded image {i}, camera {camera_to_display}, computed A vs B correlation")
        except Exception as e:
            print(f"Error reading image {i}, camera {camera_to_display}: {e}")
            axes[0, i].set_title(f'Im {i} (Error)')
            axes[0, i].axis('off')
            axes[1, i].set_title(f'Corr (Error)')
            axes[1, i].axis('off')

    set_file.close()

    plt.suptitle(f'Frame Pairs: Camera {camera_to_display}, A vs B correlation, first {num_frames_to_display} images\n{Path(set_file_path).name}', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
