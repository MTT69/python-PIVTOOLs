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
# DaVis delta+nibble RLE: a 3-byte preamble then a token stream of signed-int8
# pixel deltas, with 0x81 entering a 4-bit packed-delta run (terminated by a 0x8
# nibble) and 0x80 introducing a 16-bit little-endian absolute. Reverse-engineered
# and verified bit-exact against LaVision lvpyio on 5 real DaVis cameras
# (ALK x25 calibration plates, 2026-06-22). See _decode_packtype1_frame.
PACK_RLE = 1
PACK_ZLIB = 2
PACK_FIXED_12BIT = 3
# DaVis 10/11 lossless compression: one LZ4 *block* (not framed) over the whole
# pixel buffer, prefixed by an int64 little-endian compressed-size. Layout was
# reverse-engineered and verified bit-exact against LaVision's own lvpyio on
# merle + andre-x25 calibration frames (2026-06-12, see the calibration PRD).
PACK_LZ4 = 20

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


def _frame_byte_size(header: IM7Header) -> int:
    """Byte count for one frame (one sizeZ * sizeY plane) of uncompressed data."""
    return header.size_x * header.size_y * header.size_z * _pixel_dtype(header).itemsize


def _read_pixels_uncompressed(
    f, header: IM7Header, frame_range: tuple = None,
) -> np.ndarray:
    """Read uncompressed pixel data (pack_type=0).

    Parameters
    ----------
    f : file object
        File positioned at the start of pixel data (after 256-byte header).
    header : IM7Header
        Parsed header.
    frame_range : tuple of (start, end), optional
        If given, only read frames [start:end) (0-based). Seeks past the rest.

    Returns
    -------
    np.ndarray
        Pixel array with shape (n_frames, sizeZ, sizeY, sizeX).
    """
    dt = _pixel_dtype(header)
    frame_pixels = header.size_x * header.size_y * header.size_z
    frame_bytes = frame_pixels * dt.itemsize

    if frame_range is None:
        start, end = 0, header.size_f
    else:
        start, end = frame_range

    # Seek past leading frames
    if start > 0:
        f.seek(start * frame_bytes, 1)  # relative seek

    # Read only the frames we need
    n_frames = end - start
    n_bytes = n_frames * frame_bytes
    raw = f.read(n_bytes)
    if len(raw) != n_bytes:
        raise IOError(
            f"Expected {n_bytes} bytes of pixel data, got {len(raw)}. "
            f"File may be truncated."
        )
    pixels = np.frombuffer(raw, dtype=dt)
    return pixels.reshape(n_frames, header.size_z, header.size_y, header.size_x)


try:  # optional C accelerator; the pure-python path below is correct without it
    import lz4.block as _lz4_block_mod
except ImportError:
    _lz4_block_mod = None


def _lz4_block_decompress(src: bytes, expected_size: int) -> bytes:
    """Decompress one raw LZ4 *block* (no frame header, no checksums).

    Uses the ``lz4`` package when installed; otherwise a pure-python decoder
    (~30 MB/s — fine for frame-at-a-time calibration reads). Raises if the
    output size does not match ``expected_size`` exactly: a truncated or
    misread buffer must fail visibly, never produce a short image.
    """
    if _lz4_block_mod is not None:
        out = _lz4_block_mod.decompress(src, uncompressed_size=expected_size)
        if len(out) != expected_size:
            raise IOError(
                f"LZ4 pixel buffer decompressed to {len(out)} bytes, "
                f"expected {expected_size}"
            )
        return out

    out = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        out += src[i:i + lit]
        i += lit
        if i >= n:
            break  # final sequence carries literals only
        off = src[i] | (src[i + 1] << 8)
        i += 2
        mlen = (token & 0xF) + 4
        if (token & 0xF) == 15:
            while True:
                b = src[i]
                i += 1
                mlen += b
                if b != 255:
                    break
        start = len(out) - off
        if off == 0 or start < 0:
            raise IOError(f"corrupt LZ4 stream: match offset {off} at {len(out)}")
        if off >= mlen:
            out += out[start:start + mlen]
        else:
            # Overlapping match: LZ4 semantics require byte-serial copy.
            for _ in range(mlen):
                out.append(out[start])
                start += 1
        if len(out) > expected_size:
            raise IOError(
                f"corrupt LZ4 stream: output exceeds expected {expected_size} bytes"
            )
    if len(out) != expected_size:
        raise IOError(
            f"LZ4 pixel buffer decompressed to {len(out)} bytes, "
            f"expected {expected_size}"
        )
    return bytes(out)


def _read_pixels_lz4(f, header: IM7Header, frame_range: tuple = None) -> np.ndarray:
    """Read LZ4-packed pixel data (pack_type=20).

    The file stores ONE int64 compressed-size then one LZ4 block covering the
    whole multi-frame pixel buffer, so the full buffer is decompressed and the
    requested frames sliced out. Leaves ``f`` positioned at the attribute
    records (the byte after the compressed block).
    """
    dt = _pixel_dtype(header)
    frame_pixels = header.size_x * header.size_y * header.size_z
    total_bytes = frame_pixels * dt.itemsize * header.size_f

    comp_size = struct.unpack("<q", f.read(8))[0]
    if comp_size <= 0:
        raise IOError(f"invalid LZ4 compressed size: {comp_size}")
    comp = f.read(comp_size)
    if len(comp) != comp_size:
        raise IOError(
            f"Expected {comp_size} bytes of LZ4 data, got {len(comp)}. "
            f"File may be truncated."
        )
    out = _lz4_block_decompress(comp, total_bytes)
    pixels = np.frombuffer(out, dtype=dt).reshape(
        header.size_f, header.size_z, header.size_y, header.size_x)

    if frame_range is None:
        return pixels
    start, end = frame_range
    return pixels[start:end]


def _skip_zlib_rows(f, n_rows: int) -> None:
    """Skip n_rows of zlib-compressed data by reading size prefixes and seeking."""
    for _ in range(n_rows):
        comp_size_raw = f.read(4)
        if len(comp_size_raw) < 4:
            raise IOError("Unexpected EOF skipping compressed rows")
        comp_size = struct.unpack("<i", comp_size_raw)[0]
        f.seek(comp_size, 1)


def _read_pixels_zlib(
    f, header: IM7Header, frame_range: tuple = None,
) -> np.ndarray:
    """Read zlib-compressed pixel data (pack_type=2).

    Format: for each row, a 4-byte little-endian int32 giving the compressed
    size, followed by that many bytes of zlib-compressed row data.

    Parameters
    ----------
    f : file object
        File positioned at the start of pixel data.
    header : IM7Header
        Parsed header.
    frame_range : tuple of (start, end), optional
        If given, only decompress frames [start:end). Skips the rest.

    Returns
    -------
    np.ndarray
        Pixel array with shape (n_frames, sizeZ, sizeY, sizeX).
    """
    dt = _pixel_dtype(header)
    row_bytes = header.size_x * dt.itemsize
    rows_per_frame = header.size_z * header.size_y

    if frame_range is None:
        start, end = 0, header.size_f
    else:
        start, end = frame_range

    # Skip leading frames
    if start > 0:
        _skip_zlib_rows(f, start * rows_per_frame)

    # Read only the frames we need
    n_frames = end - start
    rows = []
    for _ in range(n_frames * rows_per_frame):
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
    return pixels.reshape(n_frames, header.size_z, header.size_y, header.size_x)


def _read_pixels_fixed12(
    f, header: IM7Header, frame_range: tuple = None,
) -> np.ndarray:
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
    frame_range : tuple of (start, end), optional
        If given, only unpack frames [start:end). Seeks past the rest.

    Returns
    -------
    np.ndarray
        Pixel array with shape (n_frames, sizeZ, sizeY, sizeX), dtype uint16.
    """
    frame_pixels = header.size_x * header.size_y * header.size_z

    if frame_range is None:
        start, end = 0, header.size_f
    else:
        start, end = frame_range

    # 12-bit: 4 pixels → 3 uint16 words (6 bytes)
    frame_groups = (frame_pixels + 3) // 4
    frame_packed_bytes = frame_groups * 3 * 2  # 3 words * 2 bytes each

    # Seek past leading frames
    if start > 0:
        f.seek(start * frame_packed_bytes, 1)

    # Read only the frames we need
    n_frames = end - start
    n_pixels = frame_pixels * n_frames
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
    return pixels.reshape(n_frames, header.size_z, header.size_y, header.size_x)


def _decode_packtype1_frame(data: bytes, off: int, npix: int):
    """Decode one pack_type-1 (DaVis delta+nibble RLE) frame.

    Token stream (after a 3-byte preamble): the running pixel value starts at 0;
    each token updates it and emits one pixel, row-major:
      * 0x80  -> next 2 bytes are a 16-bit little-endian ABSOLUTE value
      * 0x81  -> enter a 4-bit packed-delta run: read signed nibbles (high then
                 low) as deltas until a nibble of value 0x8 terminates the run
      * else  -> the byte is a signed int8 DELTA

    Returns (np.ndarray[int32] of length ``npix``, new byte offset). See PACK_RLE.
    """
    out = np.empty(npix, dtype=np.int32)
    off += 3  # 3-byte preamble (observed "01 01 00")
    prev = 0
    i = 0
    while i < npix:
        b = data[off]; off += 1
        if b == 0x81:                       # 4-bit packed-delta run
            done = False
            while not done:
                byte = data[off]; off += 1
                for nib in ((byte >> 4) & 0xF, byte & 0xF):
                    if nib == 0x8:          # run terminator
                        done = True
                        break
                    prev += nib - 16 if nib >= 8 else nib  # signed nibble
                    out[i] = prev; i += 1
                    if i >= npix:
                        done = True
                        break
        elif b == 0x80:                     # 16-bit little-endian absolute
            prev = data[off] | (data[off + 1] << 8); off += 2
            out[i] = prev; i += 1
        else:                               # signed int8 delta
            prev += b - 256 if b >= 128 else b
            out[i] = prev; i += 1
    return out, off


def _read_pixels_packtype1(f, header: IM7Header, frame_range: tuple = None) -> np.ndarray:
    """Read pack_type=1 (delta+nibble RLE) pixel data.

    Frames are concatenated, self-delimited RLE streams that cannot be seeked into
    (like zlib), so frames are decoded sequentially from frame 0; only the requested
    range is returned. Leaves ``f`` positioned at the attribute records.
    """
    dt = _pixel_dtype(header)
    npix = header.size_x * header.size_y * header.size_z
    nf = header.size_f
    start, end = frame_range if frame_range is not None else (0, nf)

    base = f.tell()
    data = f.read()  # remaining bytes: every frame's RLE stream + the attributes
    off = 0
    frames = []
    for fi in range(end):
        pix, off = _decode_packtype1_frame(data, off, npix)
        if fi >= start:
            frames.append(pix)
    # Position f at the attribute records (right after the last decoded frame).
    f.seek(base + off)

    arr = np.asarray(frames, dtype=dt)
    return arr.reshape(len(frames), header.size_z, header.size_y, header.size_x)


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


def _read_im7_internal(
    filepath: Path, frame_range: tuple = None,
) -> tuple:
    """Core reader. Optionally reads only frames [start:end).

    Parameters
    ----------
    filepath : Path
        Path to the .im7 file (must exist).
    frame_range : tuple of (start, end), optional
        0-based frame range. None reads all frames.

    Returns
    -------
    header, pixels, scales
    """
    with open(filepath, "rb") as f:
        raw_header = f.read(HEADER_SIZE)
        if len(raw_header) != HEADER_SIZE:
            raise IOError(f"File too small for header ({len(raw_header)} < {HEADER_SIZE})")
        header = _parse_header(raw_header)

        if header.pack_type == PACK_UNCOMPRESSED:
            pixels = _read_pixels_uncompressed(f, header, frame_range)
        elif header.pack_type == PACK_RLE:
            pixels = _read_pixels_packtype1(f, header, frame_range)
        elif header.pack_type == PACK_ZLIB:
            pixels = _read_pixels_zlib(f, header, frame_range)
        elif header.pack_type == PACK_FIXED_12BIT:
            pixels = _read_pixels_fixed12(f, header, frame_range)
        elif header.pack_type == PACK_LZ4:
            pixels = _read_pixels_lz4(f, header, frame_range)
        else:
            raise ValueError(
                f"Unsupported pack_type: {header.pack_type}. "
                f"Expected 0 (uncompressed), 1 (RLE), 2 (zlib), 3 (fixed 12-bit), "
                f"or 20 (LZ4)."
            )

        scales = _read_attributes(f)

    return header, pixels, scales


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
        or (F, H, W) for multi-frame with single plane,
        or (F, Z, H, W) for multi-frame/multi-plane.
    scales : IM7Scales
        Intensity scale (slope, offset, unit).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    header, pixels, scales = _read_im7_internal(filepath)

    # Squeeze singleton dimensions
    if pixels.shape[0] == 1 and header.size_z == 1:
        pixels = pixels[0, 0]  # (H, W)
    elif pixels.shape[0] == 1:
        pixels = pixels[0]  # (Z, H, W)
    elif header.size_z == 1:
        pixels = pixels[:, 0]  # (F, H, W)

    return header, pixels, scales


def get_im7_frame_count(filepath: Union[str, Path]) -> int:
    """Return the number of frames (size_f) in an .im7 file.

    Reads only the 256-byte header — no pixel decode. Used to detect how many
    frames a multi-camera buffer holds so ``frames_per_camera`` can be derived
    from the configured camera count rather than hard-coded.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "rb") as f:
        raw_header = f.read(HEADER_SIZE)
    return _parse_header(raw_header).size_f


def read_im7_camera(
    filepath: Union[str, Path],
    camera_no: int = 1,
    frames_per_camera: int = 2,
) -> np.ndarray:
    """Read a specific camera's frames from a multi-camera .im7 file.

    Only reads the frames belonging to the requested camera — skips the
    rest on disk (seek for uncompressed/12-bit, skip for zlib).

    Parameters
    ----------
    filepath : str or Path
        Path to the .im7 file.
    camera_no : int
        Camera number (1-based).
    frames_per_camera : int
        Frames stored per camera (2 for standard PIV A/B pairs, 1 for
        time-resolved or calibration snapshots).

    Returns
    -------
    np.ndarray
        Pixel data with shape (frames_per_camera, H, W), dtype float32,
        with intensity scale applied.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Parse header first to validate frame range
    with open(filepath, "rb") as f:
        raw_header = f.read(HEADER_SIZE)
    header = _parse_header(raw_header)

    start = (camera_no - 1) * frames_per_camera
    end = start + frames_per_camera

    if start >= header.size_f:
        raise ValueError(
            f"Camera {camera_no} with {frames_per_camera} frames/camera "
            f"requires frame index {start}, but file only has "
            f"{header.size_f} frames."
        )

    # Clamp to available frames (single-frame calibration .im7 files have
    # sizeF=1 but callers may request frames_per_camera=2 by default)
    end = min(end, header.size_f)
    n_frames = end - start

    # Pre-allocate float32 output, read one frame at a time so only one
    # raw-dtype frame is in memory at any point (freed before the next).
    result = np.empty((n_frames, header.size_y, header.size_x),
                      dtype=np.float32)

    with open(filepath, "rb") as f:
        f.seek(HEADER_SIZE)  # skip header

        if header.pack_type == PACK_UNCOMPRESSED:
            dt = _pixel_dtype(header)
            frame_bytes = header.size_x * header.size_y * header.size_z * dt.itemsize
            # Seek past earlier cameras' frames
            if start > 0:
                f.seek(start * frame_bytes, 1)
            for i in range(n_frames):
                raw = f.read(frame_bytes)
                if len(raw) != frame_bytes:
                    raise IOError("Unexpected EOF reading frame data")
                frame = np.frombuffer(raw, dtype=dt).reshape(
                    header.size_z, header.size_y, header.size_x)
                result[i] = frame[0] if header.size_z == 1 else frame.reshape(
                    header.size_y, header.size_x)
                del frame, raw  # free before next iteration's read

        elif header.pack_type == PACK_ZLIB:
            dt = _pixel_dtype(header)
            row_bytes = header.size_x * dt.itemsize
            rows_per_frame = header.size_z * header.size_y
            if start > 0:
                _skip_zlib_rows(f, start * rows_per_frame)
            for i in range(n_frames):
                rows = []
                for _ in range(rows_per_frame):
                    csz = struct.unpack("<i", f.read(4))[0]
                    rows.append(np.frombuffer(
                        zlib.decompress(f.read(csz)), dtype=dt))
                frame = np.stack(rows).reshape(
                    header.size_z, header.size_y, header.size_x)
                result[i] = frame[0] if header.size_z == 1 else frame.reshape(
                    header.size_y, header.size_x)
                del frame, rows

        elif header.pack_type == PACK_FIXED_12BIT:
            frame_pixels = header.size_x * header.size_y * header.size_z
            frame_groups = (frame_pixels + 3) // 4
            frame_packed_bytes = frame_groups * 3 * 2
            if start > 0:
                f.seek(start * frame_packed_bytes, 1)
            for i in range(n_frames):
                raw = f.read(frame_packed_bytes)
                if len(raw) != frame_packed_bytes:
                    raise IOError("Unexpected EOF reading 12-bit packed data")
                words = np.frombuffer(raw, dtype=np.uint16).reshape(-1, 3)
                w0 = words[:, 0].astype(np.uint32)
                w1 = words[:, 1].astype(np.uint32)
                w2 = words[:, 2].astype(np.uint32)
                p0 = (w0 & 0x0FFF).astype(np.uint16)
                p1 = (((w0 >> 12) & 0xF) | ((w1 & 0xFF) << 4)).astype(np.uint16)
                p2 = (((w1 >> 8) & 0xFF) | ((w2 & 0xF) << 8)).astype(np.uint16)
                p3 = ((w2 >> 4) & 0x0FFF).astype(np.uint16)
                frame = np.column_stack([p0, p1, p2, p3]).ravel()[:frame_pixels]
                frame = frame.reshape(header.size_z, header.size_y, header.size_x)
                result[i] = frame[0] if header.size_z == 1 else frame.reshape(
                    header.size_y, header.size_x)
                del frame, raw, words, w0, w1, w2, p0, p1, p2, p3

        elif header.pack_type == PACK_LZ4:
            # One LZ4 block covers the whole buffer; decompress once, slice.
            frames = _read_pixels_lz4(f, header, (start, end))
            for i in range(n_frames):
                frame = frames[i]
                result[i] = frame[0] if header.size_z == 1 else frame.reshape(
                    header.size_y, header.size_x)
            del frames

        elif header.pack_type == PACK_RLE:
            # RLE frames are sequential (not seekable); decode 0..end and slice.
            frames = _read_pixels_packtype1(f, header, (start, end))
            for i in range(n_frames):
                frame = frames[i]
                result[i] = frame[0] if header.size_z == 1 else frame.reshape(
                    header.size_y, header.size_x)
            del frames
        else:
            raise ValueError(
                f"Unsupported pack_type: {header.pack_type}. "
                f"Expected 0, 1 (RLE), 2 (zlib), 3 (12-bit), or 20 (LZ4)."
            )

        # Read attributes for intensity scale
        scales = _read_attributes(f)

    # Apply intensity scale in-place
    if scales.slope != 1.0:
        result *= scales.slope
    if scales.offset != 0.0:
        result += scales.offset
    return result
