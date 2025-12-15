#!/usr/bin/env python3
"""
Manual tool to export LaVision .set frame pairs to TIFF format.
For double-frame PIV data where each camera has A/B frame pairs.

Output naming convention: B%05d_A.tif and B%05d_B.tif for camera 1
"""

import sys
from pathlib import Path
import numpy as np

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

    try:
        from PIL import Image
    except ImportError:
        print("Error: PIL/Pillow not available. Please install it with: pip install Pillow")
        sys.exit(1)

    # ============ USER SETTINGS ============
    input_set_file = r"/path/to/your/file.set"  # Path to input .set file
    output_directory = r"/path/to/output"        # Output directory for TIFFs
    camera_to_export = 0                         # Camera index (0-based), camera 1 = index 0
    frame_pair_start = 0                         # Starting frame pair index
    frame_pair_end = None                        # Ending frame pair index (None = all)
    # =========================================

    input_path = Path(input_set_file)
    output_path = Path(output_directory)

    print(f"Input set file: {input_path}")
    print(f"Output directory: {output_path}")

    if not input_path.exists():
        print(f"Error: Set file not found: {input_path}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Read the set file
        set_file = lv.read_set(str(input_path))
        num_images = len(set_file)
        print(f"Set file contains {num_images} images")

        # Get structure info
        if num_images > 0:
            first_im = set_file[0]
            num_frames_per_image = len(first_im.frames)
            num_cameras = num_frames_per_image // 2
            print(f"Number of cameras: {num_cameras}")

            if camera_to_export >= num_cameras:
                print(f"Error: Camera {camera_to_export} does not exist. Max camera index is {num_cameras - 1}")
                sys.exit(1)

        # Determine frame range
        start_idx = frame_pair_start
        end_idx = frame_pair_end if frame_pair_end is not None else num_images
        end_idx = min(end_idx, num_images)

        print(f"\nExporting frame pairs {start_idx} to {end_idx - 1} from camera {camera_to_export}")
        print(f"Output format: B%05d_A.tif and B%05d_B.tif")
        print("-" * 50)

        # Export each frame pair
        for i in range(start_idx, end_idx):
            try:
                frames = read_lavision_set_frame_pair(set_file, camera_idx=camera_to_export, im_no=i)
                frame_a = frames[0]
                frame_b = frames[1]

                # Convert to 16-bit for TIFF (preserving dynamic range)
                # Normalize to 16-bit range
                def to_uint16(arr):
                    arr_min = arr.min()
                    arr_max = arr.max()
                    if arr_max > arr_min:
                        normalized = (arr - arr_min) / (arr_max - arr_min)
                    else:
                        normalized = np.zeros_like(arr)
                    return (normalized * 65535).astype(np.uint16)

                frame_a_16 = to_uint16(frame_a)
                frame_b_16 = to_uint16(frame_b)

                # Save as TIFF
                filename_a = output_path / f"B{i:05d}_A.tif"
                filename_b = output_path / f"B{i:05d}_B.tif"

                Image.fromarray(frame_a_16).save(filename_a)
                Image.fromarray(frame_b_16).save(filename_b)

                print(f"Exported frame pair {i}: {filename_a.name}, {filename_b.name}")

            except Exception as e:
                print(f"Error exporting frame pair {i}: {e}")

        set_file.close()
        print("-" * 50)
        print(f"Export complete. Files saved to: {output_path}")

    except Exception as e:
        print(f"Error reading set file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
