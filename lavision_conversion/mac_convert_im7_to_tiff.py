#!/usr/bin/env python
"""Convert LaVision .im7 files to TIFF images (macOS / pure-Python version).

Uses im7_reader (pure-Python, no lvpyio dependency) instead of lvpyio.
Same CLI interface and output structure as convert_im7_to_tiff.py.

Reads .im7 files from a directory (optionally with Cam{N}/ subdirectories)
and saves extracted frames as 32-bit float TIFF files (or optionally uint16).
Supports both multi-camera .im7 files and single-camera subdirectory layouts.

Examples:
    # Multi-camera .im7 files in a flat directory (PIV mode)
    python mac_convert_im7_to_tiff.py data/ output/ --cameras 1 2 3 4

    # Single-camera subdirectories (Cam1/, Cam2/, etc.)
    python mac_convert_im7_to_tiff.py data/ output/ --camera-subfolders

    # Calibration mode (1 frame per camera per file)
    python mac_convert_im7_to_tiff.py data/ output/ --mode calibration

    # Custom filename pattern
    python mac_convert_im7_to_tiff.py data/ output/ --pattern "B%05d.im7"

    # Save as uint16
    python mac_convert_im7_to_tiff.py data/ output/ --uint16
"""

import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np

try:
    import tifffile
except ImportError:
    sys.exit("Error: tifffile is required. Install with: pip install tifffile")

# Import the pure-Python im7 reader (same directory)
try:
    from im7_reader import read_im7
except ImportError:
    # Handle running from different working directories
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from im7_reader import read_im7


def natural_sort_key(path: str):
    """Sort key for natural ordering (B00001.im7 before B00010.im7)."""
    basename = os.path.basename(path)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", basename)]


def find_camera_subfolder(input_dir: str, cam_num: int) -> str | None:
    """Find the subfolder for a given camera number, supporting multiple naming conventions.

    Checks for: Cam{N}, camera{N}, cam{N} (case-insensitive).
    Returns the full path if found, None otherwise.
    """
    candidates = [f"Cam{cam_num}", f"camera{cam_num}", f"cam{cam_num}"]
    for name in os.listdir(input_dir):
        if name.lower() in [c.lower() for c in candidates]:
            full = os.path.join(input_dir, name)
            if os.path.isdir(full):
                return full
    return None


def discover_camera_numbers(input_dir: str) -> list:
    """Discover camera numbers from subdirectory names (Cam{N}, camera{N}, etc.)."""
    cameras = []
    for name in sorted(os.listdir(input_dir)):
        if not os.path.isdir(os.path.join(input_dir, name)):
            continue
        m = re.match(r"(?:cam|camera)(\d+)$", name, re.IGNORECASE)
        if m:
            cameras.append(int(m.group(1)))
    return sorted(cameras)


def discover_im7_files(directory: str, pattern: str = None) -> list:
    """Find and sort .im7 files in a directory."""
    if pattern:
        glob_pattern = re.sub(r"%\d*d", "*", pattern)
        files = glob.glob(os.path.join(directory, glob_pattern))
    else:
        files = glob.glob(os.path.join(directory, "*.im7"))

    return sorted(files, key=natural_sort_key)


def read_im7_frames(filepath: str, camera_no: int = 1, frames_per_camera: int = 1) -> list:
    """Extract frames for a specific camera from an .im7 file.

    Calls read_im7(), applies the intensity scale (slope/offset), ensures
    the data is at least 3D (F, H, W), and extracts frames for the
    requested camera.

    Parameters
    ----------
    filepath : str
        Path to the .im7 file.
    camera_no : int
        1-based camera number. In multi-camera .im7 files, frames are
        interleaved: camera 1 gets frames 0..fpc-1, camera 2 gets
        frames fpc..2*fpc-1, etc.
    frames_per_camera : int
        Number of frames stored per camera in each .im7 file.

    Returns
    -------
    list of np.ndarray
        List of float32 2D arrays (H, W), one per frame for this camera.
    """
    header, pixels, scales = read_im7(filepath)

    # Apply intensity scale: physical = raw * slope + offset
    pixels = (pixels.astype(np.float32) * scales.slope + scales.offset).astype(np.float32)

    # Ensure at least 3D: (F, H, W)
    if pixels.ndim == 2:
        pixels = pixels[np.newaxis, :, :]  # (1, H, W)
    elif pixels.ndim == 4:
        # (F, Z, H, W) -> collapse Z dimension by taking first plane
        pixels = pixels[:, 0, :, :]  # (F, H, W)

    total_frames = pixels.shape[0]

    # Extract frames for the requested camera
    start_frame = (camera_no - 1) * frames_per_camera
    end_frame = start_frame + frames_per_camera

    if start_frame >= total_frames:
        return []

    end_frame = min(end_frame, total_frames)
    result = [pixels[i] for i in range(start_frame, end_frame)]

    return result


def detect_total_frames(filepath: str) -> int:
    """Count total frames in an .im7 file."""
    header, pixels, _ = read_im7(filepath)
    # sizeF gives the frame count directly
    return max(header.size_f, 1)


def save_tiff(arr: np.ndarray, path: str, as_uint16: bool = False) -> None:
    """Save array as TIFF."""
    if as_uint16:
        arr = np.clip(arr, 0, None)
        if arr.max() > 0:
            arr = (arr / arr.max() * 65535).astype(np.uint16)
        else:
            arr = arr.astype(np.uint16)
    tifffile.imwrite(path, arr)


def convert_multi_camera_piv(
    files: list, output_dir: str, cameras: list, frames_per_camera: int,
    as_uint16: bool = False
) -> None:
    """Convert multi-camera .im7 files to TIFF pairs."""
    for cam in cameras:
        cam_dir = os.path.join(output_dir, f"Cam{cam}")
        os.makedirs(cam_dir, exist_ok=True)

    for file_num, filepath in enumerate(files, start=1):
        basename = os.path.splitext(os.path.basename(filepath))[0]

        for cam in cameras:
            try:
                frames = read_im7_frames(filepath, cam, frames_per_camera)
                cam_dir = os.path.join(output_dir, f"Cam{cam}")

                if frames_per_camera == 2 and len(frames) == 2:
                    path_a = os.path.join(cam_dir, f"B{file_num:05d}_A.tif")
                    path_b = os.path.join(cam_dir, f"B{file_num:05d}_B.tif")
                    save_tiff(frames[0], path_a, as_uint16)
                    save_tiff(frames[1], path_b, as_uint16)
                    print(f"  {basename} Cam{cam}: saved A+B")
                elif len(frames) >= 1:
                    path = os.path.join(cam_dir, f"B{file_num:05d}.tif")
                    save_tiff(frames[0], path, as_uint16)
                    print(f"  {basename} Cam{cam}: saved")
                else:
                    warnings.warn(f"  {basename} Cam{cam}: no frames found")
            except Exception as e:
                warnings.warn(f"  {basename} Cam{cam}: FAILED ({e})")


def convert_multi_camera_calibration(
    files: list, output_dir: str, cameras: list, frames_per_camera: int,
    as_uint16: bool = False
) -> None:
    """Convert multi-camera .im7 files in calibration mode (single frame)."""
    for cam in cameras:
        cam_dir = os.path.join(output_dir, f"Cam{cam}")
        os.makedirs(cam_dir, exist_ok=True)

    for entry_num, filepath in enumerate(files, start=1):
        basename = os.path.splitext(os.path.basename(filepath))[0]

        for cam in cameras:
            try:
                frames = read_im7_frames(filepath, cam, frames_per_camera)
                if not frames:
                    warnings.warn(f"  {basename} Cam{cam}: no frames found")
                    continue

                cam_dir = os.path.join(output_dir, f"Cam{cam}")
                path = os.path.join(cam_dir, f"cal_{entry_num:03d}.tif")
                save_tiff(frames[0], path, as_uint16)
                print(f"  {basename} Cam{cam}: saved")
            except Exception as e:
                warnings.warn(f"  {basename} Cam{cam}: FAILED ({e})")


def convert_subfolder_piv(
    input_dir: str, output_dir: str, cameras: list, as_uint16: bool = False,
    pattern: str = None
) -> None:
    """Convert single-camera .im7 files from camera subdirectories."""
    for cam in cameras:
        cam_input = find_camera_subfolder(input_dir, cam)
        if cam_input is None:
            warnings.warn(f"  Camera {cam} subdirectory not found in {input_dir}")
            continue

        files = discover_im7_files(cam_input, pattern)
        if not files:
            warnings.warn(f"  Cam{cam}: no .im7 files found")
            continue

        cam_dir = os.path.join(output_dir, f"Cam{cam}")
        os.makedirs(cam_dir, exist_ok=True)

        print(f"  Cam{cam}: {len(files)} files")

        for file_num, filepath in enumerate(files, start=1):
            basename = os.path.splitext(os.path.basename(filepath))[0]

            try:
                # Single-camera files: camera_no=1, read all frames
                frames = read_im7_frames(filepath, camera_no=1, frames_per_camera=2)

                if len(frames) == 2:
                    path_a = os.path.join(cam_dir, f"B{file_num:05d}_A.tif")
                    path_b = os.path.join(cam_dir, f"B{file_num:05d}_B.tif")
                    save_tiff(frames[0], path_a, as_uint16)
                    save_tiff(frames[1], path_b, as_uint16)
                    print(f"    {basename}: saved A+B")
                elif len(frames) == 1:
                    path = os.path.join(cam_dir, f"B{file_num:05d}.tif")
                    save_tiff(frames[0], path, as_uint16)
                    print(f"    {basename}: saved")
                else:
                    # More than 2 frames — save all individually
                    for i, frame in enumerate(frames):
                        path = os.path.join(cam_dir, f"B{file_num:05d}_f{i}.tif")
                        save_tiff(frame, path, as_uint16)
                    print(f"    {basename}: saved {len(frames)} frames")
            except Exception as e:
                warnings.warn(f"    {basename}: FAILED ({e})")


def convert_subfolder_calibration(
    input_dir: str, output_dir: str, cameras: list, as_uint16: bool = False,
    pattern: str = None
) -> None:
    """Convert single-camera .im7 calibration files from camera subdirectories."""
    for cam in cameras:
        cam_input = find_camera_subfolder(input_dir, cam)
        if cam_input is None:
            warnings.warn(f"  Camera {cam} subdirectory not found in {input_dir}")
            continue

        files = discover_im7_files(cam_input, pattern)
        if not files:
            warnings.warn(f"  Cam{cam}: no .im7 files found")
            continue

        cam_dir = os.path.join(output_dir, f"Cam{cam}")
        os.makedirs(cam_dir, exist_ok=True)

        print(f"  Cam{cam}: {len(files)} files")

        for entry_num, filepath in enumerate(files, start=1):
            basename = os.path.splitext(os.path.basename(filepath))[0]

            try:
                # Single-camera: camera_no=1, 1 frame
                frames = read_im7_frames(filepath, camera_no=1, frames_per_camera=1)
                if not frames:
                    warnings.warn(f"    {basename}: no frames found")
                    continue

                path = os.path.join(cam_dir, f"cal_{entry_num:03d}.tif")
                save_tiff(frames[0], path, as_uint16)
                print(f"    {basename}: saved")
            except Exception as e:
                warnings.warn(f"    {basename}: FAILED ({e})")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LaVision .im7 files to TIFF images (macOS pure-Python version).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1] if "Examples:" in __doc__ else "",
    )
    parser.add_argument("input_dir", help="Directory containing .im7 files")
    parser.add_argument("output_dir", help="Output directory for TIFF files")
    parser.add_argument(
        "--cameras", type=int, nargs="+", default=None,
        help="Camera numbers to extract (1-based). Default: auto-detect."
    )
    parser.add_argument(
        "--mode", choices=["piv", "calibration"], default="piv",
        help="Extraction mode: 'piv' for frame pairs (default), 'calibration' for single frames."
    )
    parser.add_argument(
        "--pattern", type=str, default=None,
        help="Filename pattern for .im7 files (e.g. 'B%%05d.im7'). Default: auto-detect all .im7."
    )
    parser.add_argument(
        "--frames-per-camera", type=int, default=None,
        help="Frames stored per camera in each .im7 file. Default: 2 for PIV, 1 for calibration."
    )
    parser.add_argument(
        "--camera-subfolders", action="store_true",
        help="Expect Cam1/, Cam2/, ... subdirectories with single-camera .im7 files."
    )
    parser.add_argument(
        "--uint16", action="store_true",
        help="Save as uint16 instead of float32 (for compatibility with some viewers)."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"Error: input directory not found: {args.input_dir}")

    frames_per_camera = args.frames_per_camera
    if frames_per_camera is None:
        frames_per_camera = 1 if args.mode == "calibration" else 2

    os.makedirs(args.output_dir, exist_ok=True)

    if args.camera_subfolders:
        # Discover cameras from subdirectories
        if args.cameras:
            cameras = args.cameras
        else:
            cameras = discover_camera_numbers(args.input_dir)
            if not cameras:
                sys.exit("Error: no camera subdirectories found (expected Cam{N} or camera{N}).")

        print(f"Camera subfolder mode (pure-Python im7_reader)")
        print(f"  Cameras: {cameras}")
        print(f"  Mode: {args.mode}")
        print(f"  Output format: {'uint16' if args.uint16 else 'float32'}")
        print()

        if args.mode == "calibration":
            convert_subfolder_calibration(
                args.input_dir, args.output_dir, cameras, args.uint16, args.pattern
            )
        else:
            convert_subfolder_piv(
                args.input_dir, args.output_dir, cameras, args.uint16, args.pattern
            )
    else:
        # Multi-camera .im7 files in flat directory
        files = discover_im7_files(args.input_dir, args.pattern)
        if not files:
            sys.exit(f"Error: no .im7 files found in {args.input_dir}")

        print(f"Found {len(files)} .im7 files in {args.input_dir}")

        # Auto-detect camera count from first file
        if args.cameras:
            cameras = args.cameras
        else:
            total_frames = detect_total_frames(files[0])
            n_cameras = total_frames // frames_per_camera
            cameras = list(range(1, n_cameras + 1))
            print(f"  Frames in first file: {total_frames}")

        print(f"  Cameras: {cameras}")
        print(f"  Frames per camera: {frames_per_camera}")
        print(f"  Mode: {args.mode}")
        print(f"  Output format: {'uint16' if args.uint16 else 'float32'}")
        print()

        if args.mode == "calibration":
            convert_multi_camera_calibration(
                files, args.output_dir, cameras, frames_per_camera, args.uint16
            )
        else:
            convert_multi_camera_piv(
                files, args.output_dir, cameras, frames_per_camera, args.uint16
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
