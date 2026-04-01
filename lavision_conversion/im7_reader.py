"""Pure-Python reader for LaVision .im7 image files.

Reads the Image_Header_7 binary format (256-byte header + pixel data + attributes).
No dependency on lvpyio -- works on macOS, Linux, and Windows.

Supports:
- Uncompressed float32 and uint16 pixel data (pack_type=0)
- Zlib-compressed data (pack_type=2)
- Fixed 12-bit packed data (pack_type=3)
- Intensity scale attributes (slope/offset)
- Multi-frame files (sizeF > 1)

Reference: LaVision ReadIM7.h / ReadIM7.cpp (liorshig/readim on GitHub)
"""
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np

# buffer_format values
BUFFER_FORMAT_FLOAT = -3
BUFFER_FORMAT_WORD = -4

# pack_type values
PACK_UNCOMPRESSED = 0
PACK_ZLIB = 2
PACK_FIXED_12BIT = 3

# Extended header record types (int32)
IEH_END = 0
IEH_SCALE_X = 1
IEH_SCALE_Y = 2
IEH_SCALE_Z = 3
IEH_SCALE_I = 4
IEH_COMMENT = 5
IEH_ATTRIBUTE = 6

HEADER_SIZE = 256
HEADER_FORMAT = "<hhhh iiii hhh"  # 30 bytes of fields + 226 reserved


@dataclass
class IM7Header:
    """Parsed Image_Header_7 from an .im7 file."""
    version: int
    pack_type: int
    buffer_format: int
    is_sparse: int
    size_x: int
    size_y: int
    size_z: int
    size_f: int
    scalar_n: int
    vector_grid: int
    extra_flags: int


@dataclass
class IM7Scales:
    """Intensity scale information extracted from .im7 attributes.

    The physical pixel value is: raw_pixel * slope + offset.
    """
    slope: float = 1.0
    offset: float = 0.0
    description: str = ""
    unit: str = ""


def _parse_header(raw: bytes) -> IM7Header:
    """Unpack the 256-byte Image_Header_7.

    Parameters
    ----------
    raw : bytes
        Exactly 256 bytes from the start of the .im7 file.

    Returns
    -------
    IM7Header
        Parsed header fields.

    Raises
    ------
    ValueError
        If raw is not 256 bytes.
    """
    if len(raw) != HEADER_SIZE:
        raise ValueError(f"Header must be {HEADER_SIZE} bytes, got {len(raw)}")

    values = struct.unpack_from(HEADER_FORMAT, raw, 0)
    return IM7Header(
        version=values[0],
        pack_type=values[1],
        buffer_format=values[2],
        is_sparse=values[3],
        size_x=values[4],
        size_y=values[5],
        size_z=values[6],
        size_f=values[7],
        scalar_n=values[8],
        vector_grid=values[9],
        extra_flags=values[10],
    )


def _pixel_dtype(header: IM7Header) -> np.dtype:
    """Return the numpy dtype for the pixel data based on buffer_format."""
    if header.buffer_format == BUFFER_FORMAT_FLOAT:
        return np.dtype("<f4")  # float32 little-endian
    elif header.buffer_format == BUFFER_FORMAT_WORD:
        return np.dtype("<u2")  # uint16 little-endian
    else:
        raise ValueError(
            f"Unsupported buffer_format: {header.buffer_format}. "
            f"Expected {BUFFER_FORMAT_FLOAT} (FLOAT) or {BUFFER_FORMAT_WORD} (WORD/uint16)."
        )


def _read_pixels_uncompressed(f, header: IM7Header) -> np.ndarray:
    """Read uncompressed pixel data (pack_type=0).

    Parameters
    ----------
    f : file object
        File positioned at the start of pixel data (after 256-byte header).
    header : IM7Header
        Parsed header.

    Returns
    -------
    np.ndarray
        Pixel array with shape (sizeF, sizeZ, sizeY, sizeX).
    """
    dt = _pixel_dtype(header)
    n_pixels = header.size_x * header.size_y * header.size_z * header.size_f
    n_bytes = n_pixels * dt.itemsize
    raw = f.read(n_bytes)
    if len(raw) != n_bytes:
        raise IOError(
            f"Expected {n_bytes} bytes of pixel data, got {len(raw)}. "
            f"File may be truncated."
        )
    pixels = np.frombuffer(raw, dtype=dt)
    return pixels.reshape(header.size_f, header.size_z, header.size_y, header.size_x)


def _read_pixels_zlib(f, header: IM7Header) -> np.ndarray:
    """Read zlib-compressed pixel data (pack_type=2).

    Format: for each row, a 4-byte little-endian int32 giving the compressed
    size, followed by that many bytes of zlib-compressed row data.

    Parameters
    ----------
    f : file object
        File positioned at the start of pixel data.
    header : IM7Header
        Parsed header.

    Returns
    -------
    np.ndarray
        Pixel array with shape (sizeF, sizeZ, sizeY, sizeX).
    """
    dt = _pixel_dtype(header)
    row_bytes = header.size_x * dt.itemsize
    total_rows = header.size_f * header.size_z * header.size_y

    rows = []
    for _ in range(total_rows):
        comp_size_raw = f.read(4)
        if len(comp_size_raw) < 4:
            raise IOError("Unexpected EOF reading compressed size")
        comp_size = struct.unpack("<i", comp_size_raw)[0]
        comp_data = f.read(comp_size)
        if len(comp_data) != comp_size:
            raise IOError("Unexpected EOF reading compressed data")
        decompressed = zlib.decompress(comp_data)
        if len(decompressed) != row_bytes:
            raise IOError(
                f"Decompressed row size {len(decompressed)} != expected {row_bytes}"
            )
        rows.append(np.frombuffer(decompressed, dtype=dt))

    pixels = np.stack(rows)
    return pixels.reshape(header.size_f, header.size_z, header.size_y, header.size_x)


def _read_pixels_fixed12(f, header: IM7Header) -> np.ndarray:
    """Read fixed 12-bit packed pixel data (pack_type=3).

    Packing: 4 pixels are stored in 3 uint16 words (48 bits = 4 x 12 bits).
    From 3 words (w0, w1, w2):
        pixel[0] = w0 & 0x0FFF
        pixel[1] = ((w0 >> 12) & 0x000F) | ((w1 & 0x00FF) << 4)
        pixel[2] = ((w1 >> 8) & 0x00FF) | ((w2 & 0x000F) << 8)
        pixel[3] = (w2 >> 4) & 0x0FFF

    Parameters
    ----------
    f : file object
        File positioned at the start of pixel data.
    header : IM7Header
        Parsed header.

    Returns
    -------
    np.ndarray
        Pixel array with shape (sizeF, sizeZ, sizeY, sizeX), dtype uint16.
    """
    n_pixels = header.size_x * header.size_y * header.size_z * header.size_f
    # Pad to multiple of 4 for the packing
    n_groups = (n_pixels + 3) // 4
    n_words = n_groups * 3
    raw = f.read(n_words * 2)
    if len(raw) != n_words * 2:
        raise IOError("Unexpected EOF reading 12-bit packed data")

    words = np.frombuffer(raw, dtype=np.uint16)
    words = words.reshape(-1, 3)
    w0 = words[:, 0].astype(np.uint32)
    w1 = words[:, 1].astype(np.uint32)
    w2 = words[:, 2].astype(np.uint32)

    p0 = (w0 & 0x0FFF).astype(np.uint16)
    p1 = (((w0 >> 12) & 0x000F) | ((w1 & 0x00FF) << 4)).astype(np.uint16)
    p2 = (((w1 >> 8) & 0x00FF) | ((w2 & 0x000F) << 8)).astype(np.uint16)
    p3 = ((w2 >> 4) & 0x0FFF).astype(np.uint16)

    pixels = np.column_stack([p0, p1, p2, p3]).ravel()[:n_pixels]
    return pixels.reshape(header.size_f, header.size_z, header.size_y, header.size_x)


def _read_attributes(f) -> IM7Scales:
    """Read extended header records (attributes) after the pixel data.

    The extended header is a sequence of records, each with:
        int32  type   (IEH_END=0 terminates)
        int32  size   (byte count of data that follows)
        bytes  data   (size bytes)

    Scale records (types 1-4) contain: 'slope offset\\0unit\\0'
    Attribute records (type 6) contain: 'name=value' (single string)

    Only IEH_SCALE_I (type 4) is extracted for the intensity scale.

    Parameters
    ----------
    f : file object
        Positioned at the start of the attribute section.

    Returns
    -------
    IM7Scales
        Intensity scale (slope, offset, unit). Defaults to slope=1, offset=0
        if no SCALE_I record is found.
    """
    scales = IM7Scales()

    while True:
        type_raw = f.read(4)
        if len(type_raw) < 4:
            break  # EOF

        ieh_type = struct.unpack("<i", type_raw)[0]
        if ieh_type == IEH_END:
            break

        size_raw = f.read(4)
        if len(size_raw) < 4:
            break
        ieh_size = struct.unpack("<i", size_raw)[0]

        if ieh_size < 0 or ieh_size > 100_000_000:
            # Corrupt or unexpected record; stop parsing
            break

        data = f.read(ieh_size)
        if len(data) != ieh_size:
            break  # Truncated

        if ieh_type == IEH_SCALE_I:
            _parse_scale_record(data, scales)

    return scales


def _parse_scale_record(data: bytes, scales: IM7Scales) -> None:
    """Parse a scale record into slope/offset/unit.

    The data format is: 'slope offset\\0unit\\0[padding]'
    where slope and offset are space-separated ASCII floats.
    """
    parts = data.split(b"\x00")
    if not parts:
        return

    # First part: "slope offset"
    values_str = parts[0].decode("utf-8", errors="replace").strip()
    tokens = values_str.split()
    if len(tokens) >= 2:
        try:
            scales.slope = float(tokens[0])
            scales.offset = float(tokens[1])
        except ValueError:
            pass
    elif len(tokens) == 1:
        try:
            scales.slope = float(tokens[0])
        except ValueError:
            pass

    # Second part: unit string
    if len(parts) > 1:
        scales.unit = parts[1].decode("utf-8", errors="replace").strip()

    # Store the raw description
    scales.description = values_str


def read_im7(filepath: Union[str, Path]) -> tuple:
    """Read a LaVision .im7 image file.

    Returns the parsed header, pixel data array, and intensity scales.
    Single-frame (sizeF=1), single-plane (sizeZ=1) data is squeezed
    to 2D (H, W). Multi-frame data retains all dimensions.

    Parameters
    ----------
    filepath : str or Path
        Path to the .im7 file.

    Returns
    -------
    header : IM7Header
        Parsed binary header.
    pixels : np.ndarray
        Pixel data. Shape is (H, W) for single-frame single-plane,
        or (F, Z, H, W) for multi-frame/multi-plane.
    scales : IM7Scales
        Intensity scale (slope, offset, unit).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the header is invalid or the pack_type/buffer_format is unsupported.
    IOError
        If pixel data is truncated or corrupt.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        # 1. Parse header
        raw_header = f.read(HEADER_SIZE)
        if len(raw_header) != HEADER_SIZE:
            raise IOError(f"File too small for header ({len(raw_header)} < {HEADER_SIZE})")
        header = _parse_header(raw_header)

        # 2. Read pixel data
        if header.pack_type == PACK_UNCOMPRESSED:
            pixels = _read_pixels_uncompressed(f, header)
        elif header.pack_type == PACK_ZLIB:
            pixels = _read_pixels_zlib(f, header)
        elif header.pack_type == PACK_FIXED_12BIT:
            pixels = _read_pixels_fixed12(f, header)
        else:
            raise ValueError(
                f"Unsupported pack_type: {header.pack_type}. "
                f"Expected 0 (uncompressed), 2 (zlib), or 3 (fixed 12-bit)."
            )

        # 3. Read attributes (scale info)
        scales = _read_attributes(f)

    # 4. Squeeze single-frame, single-plane to 2D
    if header.size_f == 1 and header.size_z == 1:
        pixels = pixels[0, 0]  # (H, W)
    elif header.size_f == 1:
        pixels = pixels[0]  # (Z, H, W)
    # else keep (F, Z, H, W)

    return header, pixels, scales
