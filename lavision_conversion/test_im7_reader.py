"""Tests for im7_reader.py against real LaVision calibration .im7 files.

All tests skip gracefully if the test files are not available on the machine.
"""
import os

import numpy as np
import pytest

from im7_reader import (
    BUFFER_FORMAT_FLOAT,
    HEADER_SIZE,
    IM7Header,
    IM7Scales,
    PACK_UNCOMPRESSED,
    _parse_header,
    read_im7,
)

TEST_DIR = (
    "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton"
    "/Documents/#current_processing/Properties/Calibration"
)
CAM1_DIR = os.path.join(TEST_DIR, "camera1")
CAM2_DIR = os.path.join(TEST_DIR, "camera2")
CAM1_FILE = os.path.join(CAM1_DIR, "B00001.im7")
CAM2_FILE = os.path.join(CAM2_DIR, "B00001.im7")

FILES_AVAILABLE = os.path.isfile(CAM1_FILE)
skip_no_files = pytest.mark.skipif(
    not FILES_AVAILABLE, reason="Calibration .im7 files not available"
)


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

@skip_no_files
def test_header_parsing():
    """Header fields match known values for DaVis calibration images."""
    with open(CAM1_FILE, "rb") as f:
        raw = f.read(HEADER_SIZE)
    header = _parse_header(raw)

    assert isinstance(header, IM7Header)
    assert header.size_x == 5312
    assert header.size_y == 4600
    assert header.size_z == 1
    assert header.size_f == 1
    assert header.pack_type == PACK_UNCOMPRESSED
    assert header.buffer_format == BUFFER_FORMAT_FLOAT
    assert header.is_sparse == 0


@skip_no_files
def test_header_extra_flags():
    """extraFlags bit 0 indicates tail offsets are present."""
    with open(CAM1_FILE, "rb") as f:
        raw = f.read(HEADER_SIZE)
    header = _parse_header(raw)
    assert header.extra_flags & 1 == 1  # tail offsets present


def test_header_wrong_size():
    """_parse_header rejects data that is not exactly 256 bytes."""
    with pytest.raises(ValueError, match="256"):
        _parse_header(b"\x00" * 128)


def test_header_roundtrip_synthetic():
    """_parse_header correctly unpacks a synthetic 256-byte header."""
    import struct

    fmt = "<hhhh iiii hhh"
    fields = (0, 0, -3, 0, 100, 200, 1, 1, 0, 1, 0)
    packed = struct.pack(fmt, *fields)
    padded = packed + b"\x00" * (HEADER_SIZE - len(packed))
    header = _parse_header(padded)

    assert header.size_x == 100
    assert header.size_y == 200
    assert header.buffer_format == -3
    assert header.pack_type == 0


# ---------------------------------------------------------------------------
# Full read: pixel data shape and range
# ---------------------------------------------------------------------------

@skip_no_files
def test_pixel_data_shape_and_range():
    """Pixel array has expected shape, dtype, and value range."""
    header, pixels, scales = read_im7(CAM1_FILE)

    # Single-frame should be squeezed to 2D
    assert pixels.shape == (4600, 5312)
    assert pixels.dtype == np.float32

    # Pixel values should be non-negative counts, within 16-bit range
    assert pixels.min() >= 0
    assert pixels.max() < 65536

    # Sanity: mean should be reasonable for a calibration target image
    assert pixels.mean() > 100


@skip_no_files
def test_scales_parsed():
    """Intensity scale should have slope=1.0, offset=0.0 for these files."""
    header, pixels, scales = read_im7(CAM1_FILE)

    assert isinstance(scales, IM7Scales)
    assert scales.slope == 1.0
    assert scales.offset == 0.0
    assert scales.unit == "counts"


@skip_no_files
def test_physical_pixel_values():
    """Applying scale (slope*raw + offset) should not change values when
    slope=1, offset=0."""
    header, pixels, scales = read_im7(CAM1_FILE)
    physical = pixels * scales.slope + scales.offset
    np.testing.assert_array_equal(physical, pixels)


# ---------------------------------------------------------------------------
# All 6 calibration frames for camera 1
# ---------------------------------------------------------------------------

@skip_no_files
def test_all_six_calibration_frames():
    """All 6 camera1 calibration frames load with correct shape."""
    for i in range(1, 7):
        filepath = os.path.join(CAM1_DIR, f"B{i:05d}.im7")
        if not os.path.isfile(filepath):
            pytest.skip(f"File {filepath} not available")

        header, pixels, scales = read_im7(filepath)
        assert pixels.shape == (4600, 5312), f"Wrong shape for B{i:05d}.im7"
        assert pixels.dtype == np.float32
        assert pixels.min() >= 0
        assert pixels.max() < 65536


# ---------------------------------------------------------------------------
# Camera 2
# ---------------------------------------------------------------------------

@skip_no_files
def test_camera2_loads():
    """camera2/B00001.im7 loads successfully with the same dimensions."""
    if not os.path.isfile(CAM2_FILE):
        pytest.skip("camera2/B00001.im7 not available")

    header, pixels, scales = read_im7(CAM2_FILE)
    assert pixels.shape == (4600, 5312)
    assert pixels.dtype == np.float32
    assert pixels.min() >= 0


@skip_no_files
def test_camera2_all_frames():
    """All 6 camera2 calibration frames load successfully."""
    for i in range(1, 7):
        filepath = os.path.join(CAM2_DIR, f"B{i:05d}.im7")
        if not os.path.isfile(filepath):
            pytest.skip(f"File {filepath} not available")

        header, pixels, scales = read_im7(filepath)
        assert pixels.shape == (4600, 5312), f"Wrong shape for camera2 B{i:05d}.im7"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_file_not_found():
    """read_im7 raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        read_im7("/nonexistent/path/to/file.im7")


def test_accepts_path_object():
    """read_im7 accepts a pathlib.Path as well as str."""
    from pathlib import Path

    with pytest.raises(FileNotFoundError):
        read_im7(Path("/nonexistent/path.im7"))


# ---------------------------------------------------------------------------
# Consistency checks across frames
# ---------------------------------------------------------------------------

@skip_no_files
def test_frames_have_different_content():
    """Different calibration frames should contain different pixel data
    (the board is at different z-positions)."""
    f1 = os.path.join(CAM1_DIR, "B00001.im7")
    f2 = os.path.join(CAM1_DIR, "B00002.im7")
    if not (os.path.isfile(f1) and os.path.isfile(f2)):
        pytest.skip("Need at least 2 calibration frames")

    _, px1, _ = read_im7(f1)
    _, px2, _ = read_im7(f2)

    # They should not be identical (different z-positions)
    assert not np.array_equal(px1, px2)
