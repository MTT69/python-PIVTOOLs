"""Pure-Python reader for LaVision .set image containers.

Reads the .set companion directory structure: index files (Frame{N}-0.ims),
data files (Frame{N}-1.ims), decoder XML, and scale XML.
No dependency on lvpyio -- works on macOS, Linux, and Windows.

Pixel encodings: DaVis names the encoding in Frame{N}-decoder.xml. The full set of
ids DaVis 10/11 can write is in ``_DECODER_BITS_PER_PX`` below; ``mono-12p``,
``raw-16-bit`` and ``raw-8-bit`` are decoded here. The rest are recognised and
refused by name -- their bit or channel order cannot be derived from the id alone,
and a wrong guess yields a plausible-looking wrong image rather than a failure.

Stream transformers: StreamSet.xml declares post-decode corrections DaVis applies
on read (see ``_IMPLEMENTED_TRANSFORMERS``). ``dark-image-subtraction`` is applied
here as ``clamp(decoded - Transformer{N}-dark.im7, 0)``; a declared correction this
reader does not implement raises rather than being skipped, because skipping one
returns different pixels from DaVis with nothing to indicate it happened.

Also supports:
- Pre-paired mode: entry[im_no].frames[2*(cam-1) + 0/1]
- Time-resolved mode: two entries, one frame each
- Per-camera frame extraction with seek-based skipping

Reference: LaVision DaVis 10.x .set recording format
"""

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
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

    frame_idx: int  # 0-based frame number
    data_path: Path  # Frame{N}-1.ims
    index_path: Path  # Frame{N}-0.ims
    decoder: str  # DaVis encoding id, e.g. "mono-12p", "raw-16-bit"
    width: int
    height: int
    n_entries: int
    scale_slope: float
    scale_offset: float
    scale_unit: str
    entries: List[IMSIndexEntry]
    dark_path: Optional[Path] = None  # Transformer{N}-dark.im7, if declared


@dataclass
class SetInfo:
    """Parsed .set container metadata."""

    set_dir: Path
    frames: List[IMSFrameInfo]
    n_entries: int  # number of images (same across all frames)


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
    """Read the pixel encoding id from Frame{N}-decoder.xml.

    The encoding is never guessed. This used to default to ``"mono-16"`` (raw
    uint16) when the XML was missing or held no ``<id>``, which silently decoded
    packed data as unpacked and produced a scrambled image instead of an error.
    ``mono-16`` is not a DaVis id at all -- it was our own placeholder.

    Raises:
        FileNotFoundError: If the decoder XML is absent.
        ValueError: If the XML holds no usable ``<id>``.
    """
    if not decoder_path.exists():
        raise FileNotFoundError(
            f"Pixel encoding unknown: {decoder_path.name} is missing from "
            f"{decoder_path.parent}. DaVis writes this file with every recording. "
            f"The encoding will not be guessed -- re-export the recording from DaVis."
        )
    root = ET.parse(decoder_path).getroot()
    id_elem = root.find("id")
    if id_elem is None or not id_elem.text or not id_elem.text.strip():
        raise ValueError(
            f"Pixel encoding unknown: {decoder_path} has no <id> element. "
            f"The encoding will not be guessed."
        )
    return id_elem.text.strip()


# ---------------------------------------------------------------------------
# Stream transformers (post-decode corrections declared by DaVis)
# ---------------------------------------------------------------------------

# Transformer IDs this reader implements. StreamSet.xml declares a per-stream
# pipeline of post-decode corrections; DaVis (and lvpyio) apply them on read, so
# skipping one silently returns different pixels from what DaVis shows for the
# same recording. An undeclared pipeline is normal -- plenty of recordings have
# none -- but a DECLARED step we do not implement must fail loudly rather than
# be dropped.
_IMPLEMENTED_TRANSFORMERS = ("dark-image-subtraction",)


def _parse_stream_transformers(set_dir: Path) -> dict:
    """Map frame-stream index to its dark-image file, from StreamSet.xml.

    DaVis declares each correction as, for example::

        <Transformer ID="dark-image-subtraction" Label="Dark image subtraction"
                     FilePrefix="Transformer0" MinDaVisVersion="10.2.0">
            <ContentPurpose IsAssociatedToFrames="true" StartFrame="0" EndFrame="0"/>
        </Transformer>

    ``StartFrame``/``EndFrame`` carry the association to the frame stream, so the
    mapping comes from those rather than from digits in ``FilePrefix``.

    Args:
        set_dir: The .set companion directory.

    Returns:
        dict: {stream index: Path to Transformer{N}-dark.im7}. Empty when no
        StreamSet.xml exists or it declares no transformers.

    Raises:
        ValueError: If a declared transformer is one this reader does not
            implement, or spans more than one frame stream.
        FileNotFoundError: If a declared dark image is not on disk.
    """
    stream_xml = set_dir / "StreamSet.xml"
    if not stream_xml.exists():
        return {}

    try:
        root = ET.parse(stream_xml).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{stream_xml} is not valid XML: {exc}") from exc

    darks = {}
    for node in root.iter("Transformer"):
        tid = (node.get("ID") or "").strip()
        prefix = (node.get("FilePrefix") or "").strip()
        if tid not in _IMPLEMENTED_TRANSFORMERS:
            raise ValueError(
                f"{set_dir.name} declares the stream transformer '{tid}' "
                f"(label '{node.get('Label', '')}'), which PIVTOOLs does not "
                f"implement. Skipping it would return different pixels from DaVis "
                f"for the same recording. PIVTOOLs implements: "
                f"{', '.join(_IMPLEMENTED_TRANSFORMERS)}."
            )

        purpose = node.find("ContentPurpose")
        if purpose is None:
            raise ValueError(
                f"Transformer '{prefix}' in {stream_xml} has no ContentPurpose, "
                f"so the frame stream it applies to is unknown."
            )
        start, end = purpose.get("StartFrame"), purpose.get("EndFrame")
        if start is None or end is None or start != end:
            raise ValueError(
                f"Transformer '{prefix}' in {stream_xml} spans frames "
                f"{start}..{end}. This reader only handles a correction bound to "
                f"exactly one frame stream."
            )

        dark_path = set_dir / f"{prefix}-dark.im7"
        if not dark_path.exists():
            raise FileNotFoundError(
                f"{stream_xml.name} declares dark-image subtraction for frame "
                f"stream {start}, but {dark_path.name} is missing from {set_dir}. "
                f"Copy the complete companion folder -- the correction is part of "
                f"the recording, not an optional extra."
            )
        darks[int(start)] = dark_path

    return darks


@lru_cache(maxsize=4)
def _load_dark(dark_path_str: str) -> np.ndarray:
    """Load and cache one Transformer{N}-dark.im7 as uint16.

    A dark image is the full sensor -- 37 MB as uint16 for a 5312x3528 camera --
    so the cache trades memory for not re-reading it on every pair, which would
    otherwise dominate read time. The cache is per process, which is what Dask
    workers need.

    Size is a compromise, not a fit. Four entries hold two cameras' worth of
    pre-paired streams (~150 MB). A rig with three or more dark-bearing cameras
    whose worker interleaves them will evict and re-read; the alk235 recordings
    have ten such streams, so sizing to hold them all would cost ~370 MB per
    worker. Raise this only with a measurement showing the re-reads matter more
    than the resident memory.

    Args:
        dark_path_str: Path to the dark .im7, as a string so it is hashable.

    Returns:
        np.ndarray: (H, W) uint16 dark image.

    Raises:
        ValueError: If the dark image is not 2D, or is not integer counts in
            [0, 65535] -- the range the uint16 subtraction below assumes.
    """
    from .im7_reader import read_im7

    _header, pixels, _scales = read_im7(dark_path_str)
    dark = np.squeeze(np.asarray(pixels))
    if dark.ndim != 2:
        raise ValueError(
            f"Dark image {Path(dark_path_str).name} is {dark.ndim}-dimensional "
            f"(shape {dark.shape}); expected a single 2D frame."
        )

    # read_im7 hands back float32 for a WORD buffer, so the integrality check is
    # needed -- but only for a float dtype. Running `dark == np.floor(dark)` on an
    # integer array promotes a 19 Mpx frame to float64 (~150 MB) to learn nothing.
    lo, hi = dark.min(), dark.max()
    integral = np.issubdtype(dark.dtype, np.integer) or bool(
        np.all(dark == np.floor(dark))
    )
    if not integral or lo < 0 or hi > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Dark image {Path(dark_path_str).name} is not integer counts in "
            f"[0, 65535] (min {lo}, max {hi}, dtype {dark.dtype}). The uint16 "
            f"subtraction below would wrap and silently return wrong pixels."
        )
    return dark.astype(np.uint16)


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


# Every pixel encoding DaVis 10/11 can write, with its bit depth per pixel.
# Recovered from LaVision's own reader: lvpyio 1.3.1, io/Core.DataObjects.dll,
# UTF-16LE string table at 0x2ec5a8-0x2ec8f8, stored as id/label pairs. Bits
# rather than bytes keeps the 10-bit (1.25 B/px) and 12-bit (1.5 B/px) sizes
# exact under integer arithmetic.
_DECODER_BITS_PER_PX = {
    "mono-10p": 10,
    "mono-10pmsb": 10,
    "mono-12p": 12,
    "mono-12packed": 12,
    "mono-12pmsb": 12,
    "raw-16-bit": 16,
    "raw-8-bit": 8,
    "rgb-24": 24,
}

# The label DaVis shows for each id, so an error message names what the user sees
# in DaVis rather than only the internal id.
_DECODER_LABELS = {
    "mono-10p": "Mono10p to 16 bit",
    "mono-10pmsb": "Mono10pmsb to 16 bit",
    "mono-12p": "Mono12p to 16 bit",
    "mono-12packed": "Mono12packed to 16 bit",
    "mono-12pmsb": "Mono12pmsb to 16 bit",
    "raw-16-bit": "Raw 16 bit",
    "raw-8-bit": "8 to 16 bit",
    "rgb-24": "RGB 24",
}

# What this reader decodes. The others are refused by name: the three 12-bit
# encodings share a byte count and differ only in bit order, and rgb-24's channel
# order is not derivable from the id, so size alone cannot disambiguate them. Each
# needs a real sample file and a cross-check against lvpyio before it can be added.
#
# That cross-check IS a plain equality assertion, with one adjustment: lvpyio pads
# the frame height up to a multiple of 16 with zero rows (3528 -> 3536), so compare
# against its first `height` rows. Everything else now agrees bit for bit, the
# declared dark-image subtraction included. See [[piv-data-formats]].
_IMPLEMENTED_DECODERS = ("mono-12p", "raw-16-bit", "raw-8-bit")


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


def _decode_raw16(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode raw uint16 little-endian bytes (DaVis 'raw-16-bit')."""
    return np.frombuffer(raw, dtype=np.dtype("<u2")).reshape(height, width)


def _decode_raw8(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode raw uint8 bytes to uint16 (DaVis 'raw-8-bit').

    Values are widened, NOT rescaled: an 8-bit count of 200 stays 200, it does not
    become 51200. This matches :func:`_decode_mono12p`, which returns 0-4095 even
    though its DaVis label also reads "to 16 bit". Every encoding in this module
    therefore returns raw sensor counts, so intensity thresholds (masking, peak
    magnitude) mean the same thing whichever encoding a recording used.
    """
    return np.frombuffer(raw, dtype=np.uint8).astype(np.uint16).reshape(height, width)


def _decode_pixels(raw: bytes, decoder: str, width: int, height: int) -> np.ndarray:
    """Dispatch to the correct pixel decoder, refusing anything unproven.

    Args:
        raw: Pixel bytes for one entry, as stored in Frame{N}-1.ims.
        decoder: Encoding id from Frame{N}-decoder.xml.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        np.ndarray: (height, width) uint16 of raw sensor counts.

    Raises:
        ValueError: If the id is unknown to DaVis 10/11, if it is a known DaVis
            encoding this reader does not implement, or if the byte count does not
            match the id's bit depth.
    """
    implemented = ", ".join(_IMPLEMENTED_DECODERS)

    if decoder not in _DECODER_BITS_PER_PX:
        raise ValueError(
            f"Unknown .set pixel encoding '{decoder}'. It is not one of the eight "
            f"encodings DaVis 10/11 writes ({', '.join(_DECODER_BITS_PER_PX)}). "
            f"Check the <id> in the Frame{{N}}-decoder.xml of this recording. "
            f"PIVTOOLs decodes: {implemented}."
        )

    if decoder not in _IMPLEMENTED_DECODERS:
        raise ValueError(
            f"PIVTOOLs cannot decode this .set: pixel encoding '{decoder}' "
            f"(DaVis '{_DECODER_LABELS[decoder]}') is a recognised DaVis encoding "
            f"but is not implemented. PIVTOOLs decodes: {implemented}. Adding "
            f"'{decoder}' needs a sample recording to verify against -- send one to "
            f"the PIVTOOLs maintainers. Re-exporting from DaVis as Raw 16 bit is a "
            f"workaround in the meantime."
        )

    bits = _DECODER_BITS_PER_PX[decoder]
    expected = (width * height * bits) // 8
    if len(raw) != expected:
        raise ValueError(
            f"Pixel data size mismatch for encoding '{decoder}': entry holds "
            f"{len(raw)} bytes but a {width}x{height} frame at {bits} bits/pixel "
            f"needs {expected}. The container may be truncated, or the encoding id "
            f"may not match the stored data."
        )

    if decoder == "mono-12p":
        return _decode_mono12p(raw, width, height)
    if decoder == "raw-16-bit":
        return _decode_raw16(raw, width, height)
    return _decode_raw8(raw, width, height)


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
        raise FileNotFoundError(
            f"{set_path.name} has no companion data folder (expected {set_dir}). "
            f"A DaVis recording is a .set file plus a folder of the same name holding "
            f"its Frame*.ims streams; both must be copied together. A .set with no "
            f"folder is usually a DaVis result set (processed vectors), which holds no "
            f"images -- point at the recording .set instead."
        )

    # Discover frame streams from files on disk
    frame_indices = sorted(
        {
            int(p.name.split("-")[0].replace("Frame", ""))
            for p in set_dir.glob("Frame*-1.ims")
        }
    )

    if not frame_indices:
        # The folder exists but holds no image streams. DaVis reuses the .set
        # extension for project and calibration nodes, and stores some recordings
        # as .im7 files instead, so name which one this is -- each has a different
        # fix and the bare "no data files" message fitted none of them.
        im7_files = sorted(set_dir.glob("*.im7"))
        if im7_files:
            shown = im7_files[0].name
            if len(im7_files) > 1:
                shown += f" and {len(im7_files) - 1} more"
            raise FileNotFoundError(
                f"{set_dir} holds .im7 files ({shown}), not .ims frame streams. This "
                f"recording is in DaVis .im7 layout: set image_type to 'lavision_im7' "
                f"and point the source path at the folder {set_dir} rather than at the "
                f".set file."
            )
        found = sorted(p.name for p in set_dir.iterdir())
        raise FileNotFoundError(
            f"{set_dir} holds no image data -- no Frame*-1.ims streams and no .im7 "
            f"files. This is a DaVis project or calibration node, not an image "
            f"recording. Found: {', '.join(found[:5])}"
            f"{f' (+{len(found) - 5} more)' if len(found) > 5 else ''}. Point at a "
            f"recording .set whose companion folder holds Frame*-1.ims."
        )

    # Post-decode corrections DaVis declares for this recording (usually none).
    dark_paths = _parse_stream_transformers(set_dir)

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

        frames.append(
            IMSFrameInfo(
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
                dark_path=dark_paths.get(fi),
            )
        )

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
        raise IndexError(f"Entry {entry_idx} out of range [0, {frame_info.n_entries})")

    entry = frame_info.entries[entry_idx]
    with open(frame_info.data_path, "rb") as f:
        f.seek(entry.offset)
        raw = f.read(entry.size)

    if len(raw) != entry.size:
        raise IOError(
            f"Expected {entry.size} bytes at offset {entry.offset}, "
            f"got {len(raw)}. File may be truncated."
        )

    img = _decode_pixels(raw, frame_info.decoder, frame_info.width, frame_info.height)

    if frame_info.dark_path is None:
        return img

    # Dark-image subtraction, clamped at zero -- DaVis and lvpyio both floor the
    # result, and on this data ~0.001% of pixels would otherwise go negative.
    # Verified bit-exact against lvpyio 1.3.1 across three frame streams and two
    # entries of E:\Softwarex\alk235\images\loop=0.set (2026-08-22).
    dark = _load_dark(str(frame_info.dark_path))
    if dark.shape != img.shape:
        raise ValueError(
            f"Dark image {frame_info.dark_path.name} is {dark.shape} but frame "
            f"stream {frame_info.frame_idx} is {img.shape}. The dark image does "
            f"not belong to this recording."
        )
    # The uint16 subtraction wraps where dark > img, so the wrapped elements are
    # then zeroed in place -- they never reach the output. Writing into the
    # subtraction's own result keeps this to one full-frame allocation plus the
    # mask.
    #
    # Not done in place on `img`: _decode_raw16 returns a np.frombuffer VIEW of
    # the packed bytes, which is read-only, so `np.subtract(img, dark, out=img)`
    # raises "output array is read-only" for raw-16-bit while working fine for
    # mono-12p (which builds a fresh array). Measured on a 5312x3528 frame:
    # 23.0 ms here, against 42.9 ms for the equivalent
    # `np.where(img > dark, img - dark, 0).astype(np.uint16)`, which allocates
    # four full-frame temporaries. That was 35% of a frame read.
    out = img - dark
    np.copyto(out, 0, where=img <= dark)
    return out


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
