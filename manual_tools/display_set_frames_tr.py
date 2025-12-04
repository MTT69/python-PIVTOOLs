#!/usr/bin/env python3
"""
Manual tool to open and display LaVision .set files - TIME RESOLVED FORMAT.
For time-resolved data where each time step contains one frame per camera.
Structure: set_file[time_step].frames[camera_idx]

Example: 5 cameras, 855 time steps
  - set_file[0].frames[0] = Camera 0, t=0
  - set_file[0].frames[4] = Camera 4, t=0
  - set_file[1].frames[0] = Camera 0, t=1
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal

def read_lavision_set_frame_tr(set_file, camera_idx: int, time_step: int) -> np.ndarray:
    """
    Read a single frame from a time-resolved LaVision .set file.

    Args:
        set_file: Already opened set file object
        camera_idx: Camera index (0-based)
        time_step: Time step index (0-based)

    Returns:
        np.ndarray: 2D array (H, W) containing the frame
    """
    im = set_file[time_step]
    frame = im.frames[camera_idx]

    # Apply scaling
    i_scale = frame.scales.i.slope
    i_offset = frame.scales.i.offset
    u_arr = frame.components["PIXEL"].planes[0] * i_scale + i_offset

    return u_arr.astype(np.float32)

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
        num_time_steps = len(set_file)
        print(f"Set file contains {num_time_steps} time steps")
        
        # Inspect structure
        print("\nInspecting set file structure:")
        if num_time_steps > 0:
            first_im = set_file[0]
            num_cameras = len(first_im.frames)
            print(f"Each time step has {num_cameras} camera frames")
            print(f"Total time steps: {num_time_steps}")
            
            print(f"\nFrame details for first time step (t=0):")
            for cam_idx in range(num_cameras):
                frame = first_im.frames[cam_idx]
                shape = frame.components["PIXEL"].planes[0].shape
                print(f"  Camera {cam_idx}: shape {shape}")
                if hasattr(frame, 'scales'):
                    print(f"    Scales: i.slope={frame.scales.i.slope}, i.offset={frame.scales.i.offset}")
        
    except Exception as e:
        print(f"Error reading set file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ============ USER SETTINGS ============
    camera_to_display = 2  # Which camera to display (0-indexed)
    num_frames_to_display = 5  # How many time steps to show
    corr_size = 128  # Size of central region for cross-correlation
    # =========================================
    
    if camera_to_display >= num_cameras:
        print(f"Error: Camera {camera_to_display} does not exist. Max camera index is {num_cameras - 1}")
        sys.exit(1)
    
    num_frames_to_display = min(num_frames_to_display, num_time_steps)
    # Need one extra frame for correlations (t vs t+1)
    num_correlations = num_frames_to_display - 1
    
    # Create subplot grid (2 rows: images and correlations)
    fig, axes = plt.subplots(2, num_frames_to_display, figsize=(4 * num_frames_to_display, 8))
    
    # Store frames for correlation
    frames_list = []

    # Read and display each frame for the selected camera
    for t in range(num_frames_to_display):
        try:
            frame = read_lavision_set_frame_tr(set_file, camera_idx=camera_to_display, time_step=t)
            frames_list.append(frame)
            vmin, vmax = np.percentile(frame, [1, 99])
            axes[0, t].imshow(frame, cmap='gray', vmin=vmin, vmax=vmax)
            axes[0, t].set_title(f't={t}')
            axes[0, t].axis('off')
            
            # Add red box showing correlation region
            h, w = frame.shape
            cy, cx = h // 2, w // 2
            half = corr_size // 2
            rect = Rectangle((cx - half, cy - half), corr_size, corr_size, 
                            linewidth=2, edgecolor='red', facecolor='none')
            axes[0, t].add_patch(rect)
            
            print(f"Loaded t={t}, camera {camera_to_display}")
        except Exception as e:
            print(f"Error reading t={t}, camera {camera_to_display}: {e}")
            frames_list.append(None)
            axes[0, t].set_title(f't={t} (Error)')
            axes[0, t].axis('off')

    # Compute and display cross-correlations (t vs t+1)
    for i in range(num_correlations):
        try:
            if frames_list[i] is not None and frames_list[i+1] is not None:
                frame1 = frames_list[i]
                frame2 = frames_list[i+1]
                
                # Extract central region
                h, w = frame1.shape
                cy, cx = h // 2, w // 2
                half = corr_size // 2
                
                region1 = frame1[cy-half:cy+half, cx-half:cx+half]
                region2 = frame2[cy-half:cy+half, cx-half:cx+half]
                
                # Normalize regions
                region1 = region1 - np.mean(region1)
                region2 = region2 - np.mean(region2)
                
                # Cross-correlation using FFT
                corr = signal.correlate2d(region1, region2, mode='full')
                
                axes[1, i].imshow(corr, cmap='jet')
                axes[1, i].set_title(f'Corr t={i} vs t={i+1}')
                axes[1, i].axis('off')
                print(f"Computed correlation t={i} vs t={i+1}")
            else:
                axes[1, i].set_title(f'Corr (Error)')
                axes[1, i].axis('off')
        except Exception as e:
            print(f"Error computing correlation {i}: {e}")
            axes[1, i].set_title(f'Corr (Error)')
            axes[1, i].axis('off')
    
    # Hide last correlation subplot (no t+1 for last frame)
    axes[1, -1].axis('off')
    axes[1, -1].set_visible(False)

    set_file.close()

    plt.suptitle(f'Time-Resolved: Camera {camera_to_display}, first {num_frames_to_display} frames (corr: t vs t+1)\n{Path(set_file_path).name}', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
