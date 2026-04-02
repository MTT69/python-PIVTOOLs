"""Compare custom im7_reader against lvpyio to verify identical output.

Requires lvpyio to be installed (Windows only). If not available, exits
gracefully — this script is for validation only, not production use.
"""
import sys
import os
import numpy as np

# Custom reader (package path, fallback to local)
try:
    from pivtools_core.image_handling.readers.im7_reader import read_im7, read_im7_camera
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from im7_reader import read_im7, read_im7_camera

# lvpyio is optional — only needed for comparison
try:
    import lvpyio as lv
    HAS_LVPYIO = True
except ImportError:
    HAS_LVPYIO = False


def compare_single_frame(filepath: str, label: str = ""):
    """Compare readers on a single-frame .im7 file (e.g. calibration image)."""
    print(f"\n{'='*60}")
    print(f"TEST: {label or filepath}")
    print(f"{'='*60}")

    # --- Custom reader ---
    header, pixels_custom, scales = read_im7(filepath)
    # Apply intensity scale to match lvpyio behavior
    pixels_custom_scaled = (pixels_custom * scales.slope + scales.offset).astype(np.float32)

    print(f"Custom reader:")
    print(f"  Header: {header.size_x}x{header.size_y}, sizeF={header.size_f}, sizeZ={header.size_z}")
    print(f"  pack_type={header.pack_type}, buffer_format={header.buffer_format}")
    print(f"  Scale: slope={scales.slope}, offset={scales.offset}, unit='{scales.unit}'")
    print(f"  Pixels shape={pixels_custom.shape}, dtype={pixels_custom.dtype}")
    print(f"  Scaled shape={pixels_custom_scaled.shape}, dtype={pixels_custom_scaled.dtype}")
    print(f"  Range: [{pixels_custom_scaled.min():.4f}, {pixels_custom_scaled.max():.4f}]")
    print(f"  Mean: {pixels_custom_scaled.mean():.4f}")

    # --- lvpyio reader ---
    buffer = lv.read_buffer(filepath)
    frames_lv = []
    for img in buffer:
        i_scale = img.scales.i.slope
        i_offset = img.scales.i.offset
        arr = img.components["PIXEL"].planes[0] * i_scale + i_offset
        frames_lv.append(arr.astype(np.float32))

    if len(frames_lv) == 1:
        pixels_lv = frames_lv[0]
    else:
        pixels_lv = np.stack(frames_lv)

    print(f"\nlvpyio reader:")
    print(f"  Num frames: {len(frames_lv)}")
    print(f"  Pixels shape={pixels_lv.shape}, dtype={pixels_lv.dtype}")
    print(f"  Range: [{pixels_lv.min():.4f}, {pixels_lv.max():.4f}]")
    print(f"  Mean: {pixels_lv.mean():.4f}")

    # --- Compare ---
    print(f"\nComparison:")
    print(f"  Shape match: {pixels_custom_scaled.shape == pixels_lv.shape}")
    print(f"  Dtype match: {pixels_custom_scaled.dtype == pixels_lv.dtype}")

    if pixels_custom_scaled.shape == pixels_lv.shape:
        exact_match = np.array_equal(pixels_custom_scaled, pixels_lv)
        print(f"  Exact match (array_equal): {exact_match}")

        if not exact_match:
            diff = np.abs(pixels_custom_scaled - pixels_lv)
            print(f"  Max abs diff: {diff.max():.10f}")
            print(f"  Mean abs diff: {diff.mean():.10f}")
            print(f"  Num different pixels: {np.count_nonzero(diff)}")

            close = np.allclose(pixels_custom_scaled, pixels_lv, atol=1e-5)
            print(f"  np.allclose(atol=1e-5): {close}")

            # Show some differing pixels
            if diff.max() > 0:
                idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"  Worst pixel at {idx}: custom={pixels_custom_scaled[idx]}, lvpyio={pixels_lv[idx]}")
        else:
            print("  PERFECT MATCH!")
    else:
        print(f"  SHAPE MISMATCH: custom={pixels_custom_scaled.shape} vs lvpyio={pixels_lv.shape}")

    return pixels_custom_scaled.shape == pixels_lv.shape and np.array_equal(pixels_custom_scaled, pixels_lv)


def compare_multi_frame(filepath: str, label: str = ""):
    """Compare readers on a multi-frame .im7 (multi-camera) file."""
    print(f"\n{'='*60}")
    print(f"TEST: {label or filepath}")
    print(f"{'='*60}")

    # --- Custom reader ---
    header, pixels_custom, scales = read_im7(filepath)
    print(f"Custom reader:")
    print(f"  Header: {header.size_x}x{header.size_y}, sizeF={header.size_f}, sizeZ={header.size_z}")
    print(f"  pack_type={header.pack_type}, buffer_format={header.buffer_format}")
    print(f"  Scale: slope={scales.slope}, offset={scales.offset}, unit='{scales.unit}'")
    print(f"  Raw pixels shape={pixels_custom.shape}, dtype={pixels_custom.dtype}")

    # Apply scale - keep full dimensions for multi-frame
    if pixels_custom.ndim == 2:
        # Single frame was squeezed
        pixels_custom_scaled = (pixels_custom * scales.slope + scales.offset).astype(np.float32)
    elif pixels_custom.ndim == 3:
        # (Z, H, W) or (F, H, W)
        pixels_custom_scaled = (pixels_custom * scales.slope + scales.offset).astype(np.float32)
    else:
        # (F, Z, H, W)
        pixels_custom_scaled = (pixels_custom * scales.slope + scales.offset).astype(np.float32)

    print(f"  Scaled shape={pixels_custom_scaled.shape}")
    print(f"  Range: [{pixels_custom_scaled.min():.4f}, {pixels_custom_scaled.max():.4f}]")

    # --- lvpyio reader ---
    buffer = lv.read_buffer(filepath)
    frames_lv = []
    for img in buffer:
        i_scale = img.scales.i.slope
        i_offset = img.scales.i.offset
        arr = img.components["PIXEL"].planes[0] * i_scale + i_offset
        frames_lv.append(arr.astype(np.float32))
        print(f"  lvpyio frame {len(frames_lv)-1}: shape={arr.shape}, scale={i_scale}, offset={i_offset}")

    print(f"\nlvpyio reader:")
    print(f"  Total frames: {len(frames_lv)}")

    # --- Compare frame by frame ---
    all_match = True

    if header.size_f > 1 and header.size_z == 1:
        # Multi-frame: custom returns (F, H, W) after squeeze
        for i, lv_frame in enumerate(frames_lv):
            if i >= pixels_custom_scaled.shape[0]:
                print(f"  Frame {i}: MISSING in custom reader (only {pixels_custom_scaled.shape[0]} frames)")
                all_match = False
                continue

            custom_frame = pixels_custom_scaled[i]
            match = np.array_equal(custom_frame, lv_frame)
            print(f"  Frame {i}: shape custom={custom_frame.shape} vs lv={lv_frame.shape}, exact_match={match}")

            if not match and custom_frame.shape == lv_frame.shape:
                diff = np.abs(custom_frame - lv_frame)
                print(f"    Max diff: {diff.max():.10f}, Mean diff: {diff.mean():.10f}")
                all_match = False
    elif header.size_f == 1:
        # Single frame - compare directly
        if len(frames_lv) == 1:
            match = np.array_equal(pixels_custom_scaled, frames_lv[0])
            print(f"  Single frame: exact_match={match}")
            if not match:
                diff = np.abs(pixels_custom_scaled - frames_lv[0])
                print(f"    Max diff: {diff.max():.10f}")
                all_match = False

    if all_match:
        print("\n  ALL FRAMES MATCH PERFECTLY!")
    else:
        print("\n  MISMATCHES DETECTED")

    return all_match


if __name__ == "__main__":
    if not HAS_LVPYIO:
        print("lvpyio not installed — cannot run comparison tests.")
        print("Install with: pip install lvpyio (Windows only)")
        sys.exit(0)

    cal_dir = (
        r"C:\Users\mtt1e23\OneDrive - University of Southampton"
        r"\Documents\#current_processing\Properties\Calibration\camera1"
    )
    multi_cam_file = (
        r"C:\Users\mtt1e23\OneDrive - University of Southampton"
        r"\General - MT RAW data\BFS_1500_1\B00001.im7"
    )

    results = []

    # Test 1: All calibration single-frame images
    print("\n" + "#"*60)
    print("# PART 1: Single-frame calibration images (camera1)")
    print("#"*60)
    for i in range(1, 7):
        fpath = os.path.join(cal_dir, f"B{i:05d}.im7")
        if os.path.isfile(fpath):
            ok = compare_single_frame(fpath, f"camera1/B{i:05d}.im7")
            results.append(("cal_cam1_B{:05d}".format(i), ok))
        else:
            print(f"\n  SKIP: {fpath} not found")

    # Test 2: Multi-camera file (read_im7 — all frames)
    print("\n" + "#"*60)
    print("# PART 2: Multi-camera .im7 file (full read)")
    print("#"*60)
    if os.path.isfile(multi_cam_file):
        ok = compare_multi_frame(multi_cam_file, "BFS_1500_1/B00001.im7")
        results.append(("multi_cam_full_read", ok))
    else:
        print(f"\n  SKIP: {multi_cam_file} not found")

    # Test 3: read_im7_camera with frame skipping vs lvpyio
    print("\n" + "#"*60)
    print("# PART 3: Per-camera frame skipping (read_im7_camera)")
    print("#"*60)
    if os.path.isfile(multi_cam_file):
        for cam in range(1, 5):
            custom = read_im7_camera(multi_cam_file, camera_no=cam, frames_per_camera=2)
            # lvpyio ground truth: read_lavision_im7 now delegates to custom reader,
            # so read directly from lvpyio for a true comparison
            buffer = lv.read_buffer(multi_cam_file)
            start = (cam - 1) * 2
            lv_frames = []
            for idx, img in enumerate(buffer):
                if idx >= start and idx < start + 2:
                    arr = (img.components["PIXEL"].planes[0] * img.scales.i.slope + img.scales.i.offset).astype(np.float32)
                    lv_frames.append(arr)
                elif idx >= start + 2:
                    break
            lvpyio_result = np.stack(lv_frames)
            match = np.array_equal(custom, lvpyio_result)
            print(f"  Camera {cam}: shape={custom.shape}, dtype={custom.dtype}, exact_match={match}")
            if not match:
                diff = np.abs(custom - lvpyio_result)
                print(f"    Max diff: {diff.max():.10f}")
            results.append((f"camera_{cam}_skip", match))

    # Summary
    print("\n" + "#"*60)
    print("# SUMMARY")
    print("#"*60)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    print(f"\n  {passed}/{total} tests passed")
