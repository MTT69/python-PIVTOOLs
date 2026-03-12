#!/usr/bin/env python
"""Convert LaVision .set files to TIFF images.

Extracts camera frames from a .set container and saves them as 32-bit float
TIFF files (or optionally uint16). Supports both PIV frame-pair and
calibration (single-frame) modes.

Examples:
    # Convert all entries, all cameras, PIV mode (frame pairs)
    python convert_set_to_tiff.py experiment.set output/

    # Convert cameras 1 and 3, entries 1-50
    python convert_set_to_tiff.py experiment.set output/ --cameras 1 3 --entries 1-50

    # Calibration mode (single frame per entry per camera)
    python convert_set_to_tiff.py calibration.set output/ --mode calibration

    # Time-resolved PIV (1 frame/entry/camera, paired sequentially)
    python convert_set_to_tiff.py experiment.set output/ --time-resolved

    # Save as uint16 instead of float32
    python convert_set_to_tiff.py experiment.set output/ --uint16
"""

import argparse
import os
import sys
import warnings

import numpy as np

try:
    import tifffile
except ImportError:
    sys.exit("Error: tifffile is required. Install with: pip install tifffile")

try:
    import lvpyio as lv
except ImportError:
    sys.exit(
        "Error: lvpyio is required. Install with: pip install lvpyio\n"
        "Note: lvpyio is only available on Windows."
    )


def parse_entry_range(entry_str: str, max_entries: int) -> range:
    """Parse an entry range string like '1-100' or '5' into a range (1-based)."""
    parts = entry_str.split("-")
    if len(parts) == 1:
        val = int(parts[0])
        return range(val, val + 1)
    elif len(parts) == 2:
        start, end = int(parts[0]), int(parts[1])
        if start < 1:
            start = 1
        if end > max_entries:
            end = max_entries
        return range(start, end + 1)
    else:
        raise ValueError(f"Invalid entry range: {entry_str}")


def extract_frame(frame) -> np.ndarray:
    """Extract and scale a single frame from a .set entry."""
    pixel_data = frame.components["PIXEL"].planes[0]
    slope = frame.scales.i.slope
    offset = frame.scales.i.offset
    return (pixel_data * slope + offset).astype(np.float32)


def save_tiff(arr: np.ndarray, path: str, as_uint16: bool = False) -> None:
    """Save array as TIFF."""
    if as_uint16:
        # Clip negatives, scale to uint16 range
        arr = np.clip(arr, 0, None)
        if arr.max() > 0:
            arr = (arr / arr.max() * 65535).astype(np.uint16)
        else:
            arr = arr.astype(np.uint16)
    tifffile.imwrite(path, arr)


def detect_cameras(set_file) -> int:
    """Detect the number of cameras from the first entry's frame count."""
    first_entry = set_file[0]
    n_frames = len(first_entry.frames)
    return n_frames


def convert_piv_prepaired(
    set_file, output_dir: str, cameras: list, entries: range,
    as_uint16: bool = False
) -> None:
    """Convert pre-paired PIV .set (2 frames per camera per entry)."""
    n_frames_per_entry = len(set_file[0].frames)
    n_cameras = n_frames_per_entry // 2

    for cam in cameras:
        if cam < 1 or cam > n_cameras:
            warnings.warn(f"Camera {cam} out of range (1-{n_cameras}), skipping.")
            continue

        cam_dir = os.path.join(output_dir, f"Cam{cam}")
        os.makedirs(cam_dir, exist_ok=True)

        for entry_num in entries:
            entry_idx = entry_num - 1  # 0-based
            if entry_idx >= len(set_file):
                warnings.warn(f"Entry {entry_num} out of range, skipping.")
                continue

            try:
                entry = set_file[entry_idx]
                frame_a_idx = 2 * (cam - 1)
                frame_b_idx = frame_a_idx + 1

                img_a = extract_frame(entry.frames[frame_a_idx])
                img_b = extract_frame(entry.frames[frame_b_idx])

                path_a = os.path.join(cam_dir, f"B{entry_num:05d}_A.tif")
                path_b = os.path.join(cam_dir, f"B{entry_num:05d}_B.tif")

                save_tiff(img_a, path_a, as_uint16)
                save_tiff(img_b, path_b, as_uint16)

                print(f"  Cam{cam} entry {entry_num}: saved A+B")
            except Exception as e:
                warnings.warn(f"  Cam{cam} entry {entry_num}: FAILED ({e})")


def convert_piv_time_resolved(
    set_file, output_dir: str, cameras: list, entries: range,
    as_uint16: bool = False
) -> None:
    """Convert time-resolved PIV .set (1 frame per camera per entry, paired sequentially)."""
    n_frames_per_entry = len(set_file[0].frames)
    n_cameras = n_frames_per_entry  # 1 frame per camera

    # Sequential pairing: entry N = frame A, entry N+1 = frame B
    entry_list = list(entries)
    n_pairs = len(entry_list) - 1
    if n_pairs < 1:
        print("Error: need at least 2 entries for time-resolved pairing.")
        return

    for cam in cameras:
        if cam < 1 or cam > n_cameras:
            warnings.warn(f"Camera {cam} out of range (1-{n_cameras}), skipping.")
            continue

        cam_dir = os.path.join(output_dir, f"Cam{cam}")
        os.makedirs(cam_dir, exist_ok=True)

        cam_idx = cam - 1  # 0-based frame index within entry

        for pair_num, i in enumerate(range(n_pairs), start=1):
            entry_a_num = entry_list[i]
            entry_b_num = entry_list[i + 1]
            entry_a_idx = entry_a_num - 1
            entry_b_idx = entry_b_num - 1

            if entry_a_idx >= len(set_file) or entry_b_idx >= len(set_file):
                warnings.warn(f"Entry {entry_a_num}/{entry_b_num} out of range, skipping.")
                continue

            try:
                img_a = extract_frame(set_file[entry_a_idx].frames[cam_idx])
                img_b = extract_frame(set_file[entry_b_idx].frames[cam_idx])

                path_a = os.path.join(cam_dir, f"B{pair_num:05d}_A.tif")
                path_b = os.path.join(cam_dir, f"B{pair_num:05d}_B.tif")

                save_tiff(img_a, path_a, as_uint16)
                save_tiff(img_b, path_b, as_uint16)

                print(f"  Cam{cam} pair {pair_num} (entries {entry_a_num}+{entry_b_num}): saved A+B")
            except Exception as e:
                warnings.warn(
                    f"  Cam{cam} pair {pair_num} (entries {entry_a_num}+{entry_b_num}): FAILED ({e})"
                )


def convert_calibration(
    set_file, output_dir: str, cameras: list, entries: range,
    as_uint16: bool = False
) -> None:
    """Convert calibration .set (single frame per camera per entry)."""
    n_frames_per_entry = len(set_file[0].frames)

    # Calibration may store 1 frame per camera, or 2 (A+B) — use first frame
    if n_frames_per_entry >= 2 * max(cameras):
        # Pre-paired layout: take frame A only
        frame_index_fn = lambda cam: 2 * (cam - 1)
        n_cameras = n_frames_per_entry // 2
    else:
        # Single frame per camera
        frame_index_fn = lambda cam: cam - 1
        n_cameras = n_frames_per_entry

    for cam in cameras:
        if cam < 1 or cam > n_cameras:
            warnings.warn(f"Camera {cam} out of range (1-{n_cameras}), skipping.")
            continue

        cam_dir = os.path.join(output_dir, f"Cam{cam}")
        os.makedirs(cam_dir, exist_ok=True)

        for entry_num in entries:
            entry_idx = entry_num - 1
            if entry_idx >= len(set_file):
                warnings.warn(f"Entry {entry_num} out of range, skipping.")
                continue

            try:
                frame_idx = frame_index_fn(cam)
                img = extract_frame(set_file[entry_idx].frames[frame_idx])

                path = os.path.join(cam_dir, f"cal_{entry_num:03d}.tif")
                save_tiff(img, path, as_uint16)

                print(f"  Cam{cam} entry {entry_num}: saved")
            except Exception as e:
                warnings.warn(f"  Cam{cam} entry {entry_num}: FAILED ({e})")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LaVision .set files to TIFF images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1] if "Examples:" in __doc__ else "",
    )
    parser.add_argument("input_path", help="Path to the .set file")
    parser.add_argument("output_dir", help="Output directory for TIFF files")
    parser.add_argument(
        "--cameras", type=int, nargs="+", default=None,
        help="Camera numbers to extract (1-based). Default: all cameras."
    )
    parser.add_argument(
        "--mode", choices=["piv", "calibration"], default="piv",
        help="Extraction mode: 'piv' for frame pairs (default), 'calibration' for single frames."
    )
    parser.add_argument(
        "--time-resolved", action="store_true",
        help="For PIV: treat as time-resolved (1 frame/entry/camera, paired sequentially)."
    )
    parser.add_argument(
        "--entries", type=str, default=None,
        help="Entry range to convert (1-based, e.g. '1-100' or '5'). Default: all."
    )
    parser.add_argument(
        "--uint16", action="store_true",
        help="Save as uint16 instead of float32 (for compatibility with some viewers)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        sys.exit(f"Error: input file not found: {args.input_path}")

    print(f"Opening: {args.input_path}")
    set_file = lv.read_set(args.input_path)

    n_entries = len(set_file)
    n_frames_per_entry = len(set_file[0].frames)
    print(f"  Entries: {n_entries}")
    print(f"  Frames per entry: {n_frames_per_entry}")

    # Determine camera count
    if args.time_resolved or args.mode == "calibration":
        # Could be 1 frame/camera or 2 frames/camera for calibration
        if args.mode == "calibration" and n_frames_per_entry >= 2:
            # Heuristic: if even number, likely pre-paired
            n_cameras = n_frames_per_entry // 2 if n_frames_per_entry % 2 == 0 else n_frames_per_entry
        elif args.time_resolved:
            n_cameras = n_frames_per_entry
        else:
            n_cameras = n_frames_per_entry
    else:
        n_cameras = n_frames_per_entry // 2

    print(f"  Detected cameras: {n_cameras}")

    cameras = args.cameras if args.cameras else list(range(1, n_cameras + 1))
    print(f"  Extracting cameras: {cameras}")

    # Determine entry range
    entry_range = parse_entry_range(args.entries, n_entries) if args.entries else range(1, n_entries + 1)
    print(f"  Entries: {entry_range.start}-{entry_range.stop - 1}")
    print(f"  Mode: {args.mode}{'(time-resolved)' if args.time_resolved else ''}")
    print(f"  Output format: {'uint16' if args.uint16 else 'float32'}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "calibration":
        convert_calibration(set_file, args.output_dir, cameras, entry_range, args.uint16)
    elif args.time_resolved:
        convert_piv_time_resolved(set_file, args.output_dir, cameras, entry_range, args.uint16)
    else:
        convert_piv_prepaired(set_file, args.output_dir, cameras, entry_range, args.uint16)

    set_file.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
