#!/usr/bin/env python3
"""
Manual tool to open and display LaVision .set files.
Displays the first two frames (A and B) from the first camera.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def read_lavision_set_frames(set_file_path: str, camera_no: int = 1, im_no: int = 1) -> np.ndarray:
    """
    Read LaVision .set file and extract frames for a specific camera and image number.

    Args:
        set_file_path: Path to the .set file
        camera_no: Camera number (1-based)
        im_no: Image number (1-based)

    Returns:
        np.ndarray: Array of shape (2, H, W) containing frame A and B
    """
    try:
        import lvpyio as lv
    except ImportError:
        print("Error: lvpyio library not available. Please install it with: pip install lvpyio")
        sys.exit(1)

    if not Path(set_file_path).exists():
        print(f"Error: Set file not found: {set_file_path}")
        sys.exit(1)

    try:
        # Read the set file
        set_file = lv.read_set(set_file_path)
        im = set_file[im_no - 1]  # 0-based indexing in Python
    except Exception as e:
        print(f"Error reading set file: {e}")
        sys.exit(1)

    # Extract frames for this camera
    data = np.zeros((2, *im.frames[0].components["PIXEL"].planes[0].shape), dtype=np.float64)

    for j in range(2):
        # Frame indexing: 2*cameraNo-(2-j)
        frame_idx = 2 * camera_no - (2 - j)
        frame = im.frames[frame_idx]

        # Apply scaling
        i_scale = frame.scales.i.slope
        i_offset = frame.scales.i.offset
        u_arr = frame.components["PIXEL"].planes[0] * i_scale + i_offset

        data[j, :, :] = u_arr

    set_file.close()
    return data.astype(np.float32)

def main():
    # Hardcoded path for this specific request
    # set_file_path = r"C:\Users\mtt1e23\Downloads\OneDrive_1_29-10-2025\loop=1.set"
    set_file_path = r"D:\Wake_PIV_16_8\Case_A_Phase_003_Run_1.set"

    print(f"Attempting to open: {set_file_path}")

    # Read the frames
    try:
        frames = read_lavision_set_frames(set_file_path, camera_no=1, im_no=0)
        print(f"Successfully read frames with shape: {frames.shape}")
    except Exception as e:
        print(f"Error reading frames: {e}")
        sys.exit(1)

    # Display the frames
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Frame A (first frame)
    ax1.imshow(frames[0], cmap='gray')
    ax1.set_title('Frame A (Image 1)')
    ax1.axis('off')

    # Frame B (second frame)
    ax2.imshow(frames[1], cmap='gray')
    ax2.set_title('Frame B (Image 2)')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()