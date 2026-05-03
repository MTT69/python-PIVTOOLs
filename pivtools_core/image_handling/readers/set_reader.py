"""Pure-Python reader for LaVision .set image containers.

Reads the .set companion directory structure: index files (Frame{N}-0.ims),
data files (Frame{N}-1.ims), decoder XML, and scale XML.
No dependency on lvpyio -- works on macOS, Linux, and Windows.

Supports:
- Mono12Packed (mono-12p) pixel encoding
- Raw uint16 (mono-16) pixel encoding
- Pre-paired mode: entry[im_no].frames[2*(cam-1) + 0/1]
- Time-resolved mode: two entries, one frame each
- Per-camera frame extraction with seek-based skipping

Reference: LaVision DaVis 10.x .set recording format
"""
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IMSIndexEntry:
    """One entry in a Frame{N}-0.ims index file."""
    flag: int
    offset: int
    size: int


@dataclass
class IMSFrameInfo:
    """Metadata for one frame stream (one Frame{N} set of files)."""
    frame_idx: int          # 0-based frame number
    data_path: Path         # Frame{N}-1.ims
    index_path: Path        # Frame{N}-0.ims
    decoder: str            # e.g. "mono-12p", "mono-16"
    width: int
    height: int
    n_entries: int
    scale_slope: float
    scale_offset: float
    scale_unit: str
    entries: List[IMSIndexEntry]


@dataclass
class SetInfo:
    """Parsed .set container metadata."""
    set_dir: Path
    frames: List[IMSFrameInfo]
    n_entries: int          # number of images (same across all frames)


# ---------------------------------------------------------------------------
# Index parsing
# ---------------------------------------------------------------------------

def _parse_ims_index(index_path: Path) -> Tuple[int, int, int, List[IMSIndexEntry]]:
    """Parse a Frame{N}-0.ims index file.

    Returns (width, height, n_entries, entries).
    """
    data = index_path.read_bytes()
    if len(data) < 256:
        raise IOError(f"Index file too small: {index_path}")

    # Header: 256 bytes. Dimensions at int32 offsets [3] and [4] (12 and 16).
    width = struct.unpack_from("<i", data, 12)[0]
    height = struct.unpack_from("<i", data, 16)[0]

    # Entry table starts at offset 1024 (256 header + 768 padding)
    # Each entry: int32 flag + int64 offset + int64 size = 20 bytes
    table_offset = 256 + 768
    if table_offset > len(data):
        # Smaller padding — try to detect
        table_offset = 256
        # Scan for first non-zero after header
        for i in range(256, len(data), 4):
            val = struct.unpack_from("<i", data, i)[0]
            if val != 0:
                table_offset = i
                break

    entry_size = 20
    n_entries = (len(data) - table_offset) // entry_size

    entries = []
    for i in range(n_entries):
        pos = table_offset + i * entry_size
        flag = struct.unpack_from("<i", data, pos)[0]
        offset = struct.unpack_from("<q", data, pos + 4)[0]
        size = struct.unpack_from("<q", data, pos + 12)[0]
        entries.append(IMSIndexEntry(flag=flag, offset=offset, size=size))

    return width, height, n_entries, entries


# ---------------------------------------------------------------------------
# Decoder detection
# ---------------------------------------------------------------------------

def _read_decoder(decoder_path: Path) -> str:
    """Read pixel format from Frame{N}-decoder.xml."""
    if not decoder_path.exists():
        return "mono-16"  # default to raw uint16
    tree = ET.parse(decoder_path)
    root = tree.getroot()
    id_elem = root.find("id")
    if id_elem is not None and id_elem.text:
        return id_elem.text.strip()
    return "mono-16"


# ---------------------------------------------------------------------------
# Scale parsing
# ---------------------------------------------------------------------------

def _read_scales(scales_path: Path) -> Tuple[float, float, str]:
    """Read intensity scale from FrameScales{N}.scales XML."""
    if not scales_path.exists():
        return 1.0, 0.0, ""
    tree = ET.parse(scales_path)
    root = tree.getroot()
    scale_i = root.find(".//ScaleI")
    if scale_i is None:
        return 1.0, 0.0, ""
    slope = float(scale_i.get("Factor", "1"))
    offset = float(scale_i.get("Offset", "0"))
    unit = scale_i.get("Unit", "")
    return slope, offset, unit


# ---------------------------------------------------------------------------
# Pixel decoding
# ---------------------------------------------------------------------------

def _decode_mono12p(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode Mono12Packed (GigE Vision) bytes to uint16 array.

    Packing: 2 pixels in 3 bytes.
        pixel0 = byte0 | (byte1 & 0x0F) << 8
        pixel1 = (byte1 >> 4) | (byte2 << 4)
    """
    n_pixels = width * height
    arr = np.frombuffer(raw, dtype=np.uint8)
    b0 = arr[0::3].astype(np.uint16)
    b1 = arr[1::3].astype(np.uint16)
    b2 = arr[2::3].astype(np.uint16)

    pixels = np.empty(n_pixels, dtype=np.uint16)
    pixels[0::2] = b0 | ((b1 & 0x0F) << 8)
    pixels[1::2] = (b1 >> 4) | (b2 << 4)
    return pixels.reshape(height, width)


def _decode_mono16(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode raw uint16 little-endian bytes."""
    return np.frombuffer(raw, dtype=np.dtype("<u2")).reshape(height, width)


def _decode_pixels(raw: bytes, decoder: str, width: int, height: int) -> np.ndarray:
    """Dispatch to the correct pixel decoder."""
    if decoder == "mono-12p":
        return _decode_mono12p(raw, width, height)
    elif decoder in ("mono-16", ""):
        return _decode_mono16(raw, width, height)
    else:
        raise ValueError(f"Unsupported pixel decoder: '{decoder}'")


# ---------------------------------------------------------------------------
# Phantom CINE companion detection and reading
# ---------------------------------------------------------------------------

def _find_cine_companions(set_dir: Path) -> dict:
    """Return {camera_num: Path} for Camera*.cine files in set_dir, or {} if none."""
    result = {}
    for p in sorted(set_dir.glob("Camera*.cine")):
        stem = p.stem
        if stem.startswith("Camera"):
            try:
                result[int(stem[len("Camera"):])] = p
            except ValueError:
                continue
    return result


def _read_cine_set_pair(
    cine_map: dict,
    camera_no: int,
    im_no: int,
    time_resolved: bool,
    im_no_b: Optional[int],
) -> np.ndarray:
    """Read a frame pair from a Phantom-CINE-based .set container.

    Pre-paired: A/B frames interleaved within each camera's .cine file.
        Pair im_no=1 → cine frames 1 and 2; pair im_no=2 → frames 3 and 4.

    Time-resolved: each entry is one frame; im_no and im_no_b select both.
    """
    from .cine_reader import read_cine_single

    if camera_no not in cine_map:
        raise ValueError(
            f"Camera {camera_no} not found in SET companion directory. "
            f"Available cameras: {sorted(cine_map.keys())}"
        )

    cine_path = str(cine_map[camera_no])

    if time_resolved:
        if im_no_b is None:
            raise ValueError("im_no_b required for time_resolved mode")
        frame_a = read_cine_single(cine_path, im_no)
        frame_b = read_cine_single(cine_path, im_no_b)
    else:
        idx_a = 2 * (im_no - 1) + 1
        frame_a = read_cine_single(cine_path, idx_a)
        frame_b = read_cine_single(cine_path, idx_a + 1)

    result = np.empty((2, frame_a.shape[0], frame_a.shape[1]), dtype=np.float32)
    result[0] = frame_a
    result[1] = frame_b
    return result


# ---------------------------------------------------------------------------
# Set container parsing
# ---------------------------------------------------------------------------

def _parse_set(set_path: Union[str, Path]) -> SetInfo:
    """Parse a .set container, returning metadata for all frame streams.

    Parameters
    ----------
    set_path : str or Path
        Path to the .set file (the companion directory is derived from it).
    """
    set_path = Path(set_path)
    set_dir = set_path.with_suffix("")  # companion directory

    if not set_dir.is_dir():
        raise FileNotFoundError(f"Companion directory not found: {set_dir}")

    # Discover frame streams from files on disk
    frame_indices = sorted({
        int(p.name.split("-")[0].replace("Frame", ""))
        for p in set_dir.glob("Frame*-1.ims")
    })

    if not frame_indices:
        raise FileNotFoundError(f"No Frame*-1.ims data files in {set_dir}")

    frames = []
    n_entries = None

    for fi in frame_indices:
        index_path = set_dir / f"Frame{fi}-0.ims"
        data_path = set_dir / f"Frame{fi}-1.ims"
        decoder_path = set_dir / f"Frame{fi}-decoder.xml"
        scales_path = set_dir / f"FrameScales{fi}.scales"

        if not data_path.exists():
            raise FileNotFoundError(f"Data file missing: {data_path}")

        width, height, n_ent, entries = _parse_ims_index(index_path)
        decoder = _read_decoder(decoder_path)
        slope, offset, unit = _read_scales(scales_path)

        if n_entries is None:
            n_entries = n_ent
        elif n_ent != n_entries:
            raise ValueError(
                f"Frame{fi} has {n_ent} entries but Frame{frame_indices[0]} "
                f"has {n_entries}"
            )

        frames.append(IMSFrameInfo(
            frame_idx=fi,
            data_path=data_path,
            index_path=index_path,
            decoder=decoder,
            width=width,
            height=height,
            n_entries=n_ent,
            scale_slope=slope,
            scale_offset=offset,
            scale_unit=unit,
            entries=entries,
        ))

    return SetInfo(set_dir=set_dir, frames=frames, n_entries=n_entries)


# ---------------------------------------------------------------------------
# Single-image reader (one entry from one frame stream)
# ---------------------------------------------------------------------------

def _read_single_image(
    frame_info: IMSFrameInfo,
    entry_idx: int,
) -> np.ndarray:
    """Read one image from a frame stream, returning uint16.

    Parameters
    ----------
    frame_info : IMSFrameInfo
        Frame stream metadata.
    entry_idx : int
        0-based entry index.

    Returns
    -------
    np.ndarray
        Image array (H, W) uint16.
    """
    if entry_idx < 0 or entry_idx >= frame_info.n_entries:
        raise IndexError(
            f"Entry {entry_idx} out of range [0, {frame_info.n_entries})"
        )

    entry = frame_info.entries[entry_idx]
    with open(frame_info.data_path, "rb") as f:
        f.seek(entry.offset)
        raw = f.read(entry.size)

    if len(raw) != entry.size:
        raise IOError(
            f"Expected {entry.size} bytes at offset {entry.offset}, "
            f"got {len(raw)}. File may be truncated."
        )

    return _decode_pixels(raw, frame_info.decoder,
                          frame_info.width, frame_info.height)


# ---------------------------------------------------------------------------
# Scale helpers
# ---------------------------------------------------------------------------

def _apply_scale_inplace(arr: np.ndarray, fi: IMSFrameInfo) -> None:
    """Apply intensity scale (slope/offset) in-place on a float32 array."""
    if fi.scale_slope != 1.0:
        arr *= fi.scale_slope
    if fi.scale_offset != 0.0:
        arr += fi.scale_offset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_set_info(set_path: Union[str, Path]) -> SetInfo:
    """Parse .set metadata without reading any pixel data.

    Useful for getting entry count, dimensions, frame count.
    """
    return _parse_set(set_path)


def get_set_entry_count(set_path: Union[str, Path]) -> int:
    """Get the number of entries (image pairs) in a .set file."""
    set_path = Path(set_path)
    set_dir = set_path.with_suffix("")
    if set_dir.is_dir():
        cine_map = _find_cine_companions(set_dir)
        if cine_map:
            from .cine_reader import get_cine_frame_count
            total = get_cine_frame_count(str(cine_map[sorted(cine_map)[0]]))
            return total // 2
    info = _parse_set(set_path)
    return info.n_entries


def read_set_pair(
    set_path: Union[str, Path],
    camera_no: int,
    im_no: int,
    time_resolved: bool = False,
    im_no_b: Optional[int] = None,
    set_info: Optional[SetInfo] = None,
) -> np.ndarray:
    """Read a frame pair from a .set container.

    Matches the API of lavision_reader.read_lavision_ims().

    Pre-paired mode (time_resolved=False):
        Reads frames[2*(camera_no-1)] and frames[2*(camera_no-1)+1]
        from entry im_no.

    Time-resolved mode (time_resolved=True):
        Reads frames[camera_no-1] from entries im_no and im_no_b.

    Parameters
    ----------
    set_path : str or Path
        Path to the .set file.
    camera_no : int
        Camera number (1-based).
    im_no : int
        Image/entry number (1-based).
    time_resolved : bool
        If True, use time-resolved pairing.
    im_no_b : int, optional
        Second entry number for time-resolved mode (1-based).
    set_info : SetInfo, optional
        Pre-parsed container metadata. If provided, skips re-parsing the
        index/XML files. Use read_set_info() once, then pass it here for
        every pair in a batch.

    Returns
    -------
    np.ndarray
        Array of shape (2, H, W), dtype float32, with intensity scale applied.
    """
    set_path = Path(set_path)

    # Phantom CINE-based .set: companion dir has Camera{N}.cine files.
    # Only check when set_info was not pre-provided (set_info is always IMS-based).
    if set_info is None:
        set_dir = set_path.with_suffix("")
        if set_dir.is_dir():
            cine_map = _find_cine_companions(set_dir)
            if cine_map:
                return _read_cine_set_pair(cine_map, camera_no, im_no, time_resolved, im_no_b)

    info = set_info if set_info is not None else _parse_set(set_path)

    if time_resolved:
        if im_no_b is None:
            raise ValueError("im_no_b required for time_resolved mode")

        frame_idx = camera_no - 1
        if frame_idx >= len(info.frames):
            raise ValueError(
                f"Camera {camera_no} requires frame index {frame_idx}, "
                f"but only {len(info.frames)} frames exist"
            )

        fi = info.frames[frame_idx]
        entry_a = im_no - 1
        entry_b = im_no_b - 1

        result = np.empty((2, fi.height, fi.width), dtype=np.float32)

        img_a = _read_single_image(fi, entry_a)
        result[0] = img_a
        del img_a

        img_b = _read_single_image(fi, entry_b)
        result[1] = img_b
        del img_b

        # Apply per-frame-stream intensity scale
        _apply_scale_inplace(result[0], fi)
        _apply_scale_inplace(result[1], fi)

    else:
        frame_idx_a = 2 * (camera_no - 1)
        frame_idx_b = frame_idx_a + 1

        if frame_idx_b >= len(info.frames):
            raise ValueError(
                f"Camera {camera_no} requires frames [{frame_idx_a}, "
                f"{frame_idx_b}], but only {len(info.frames)} frames exist"
            )

        fi_a = info.frames[frame_idx_a]
        fi_b = info.frames[frame_idx_b]
        entry_idx = im_no - 1

        result = np.empty((2, fi_a.height, fi_a.width), dtype=np.float32)

        img_a = _read_single_image(fi_a, entry_idx)
        result[0] = img_a
        del img_a

        img_b = _read_single_image(fi_b, entry_idx)
        result[1] = img_b
        del img_b

        # Apply per-frame-stream intensity scale (A and B may differ)
        _apply_scale_inplace(result[0], fi_a)
        _apply_scale_inplace(result[1], fi_b)

    return result


def read_set_frame(
    set_path: Union[str, Path],
    entry_no: int,
    frame_idx: int,
    set_info: Optional[SetInfo] = None,
) -> np.ndarray:
    """Read a single frame from a .set container.

    Parameters
    ----------
    set_path : str or Path
        Path to the .set file.
    entry_no : int
        Entry number (1-based).
    frame_idx : int
        Frame index within the entry (0-based, maps to Frame{N} files).
    set_info : SetInfo, optional
        Pre-parsed metadata (avoids re-parsing).

    Returns
    -------
    np.ndarray
        Image (H, W) float32 with intensity scale applied.
    """
    info = set_info if set_info is not None else _parse_set(set_path)

    if frame_idx < 0 or frame_idx >= len(info.frames):
        raise ValueError(
            f"Frame index {frame_idx} out of range [0, {len(info.frames)})"
        )

    fi = info.frames[frame_idx]
    img = _read_single_image(fi, entry_no - 1)

    result = img.astype(np.float32)
    del img
    _apply_scale_inplace(result, fi)
    return result
