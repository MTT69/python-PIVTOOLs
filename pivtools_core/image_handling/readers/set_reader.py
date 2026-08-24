"""Pure-Python reader for LaVision .set image containers.

Reads the .set companion directory structure: per frame stream an index file
({prefix}-0.ims), a data file ({prefix}-1.ims), a decoder XML and a scale XML.
No dependency on lvpyio -- works on macOS, Linux, and Windows.

Stream discovery: StreamSet.xml is the companion folder's manifest. Its
ReaderInfo nodes name each stream's file prefix and bind it to a frame-stream
index; the prefix is DaVis's choice per recording ("Frame0".. on one, "Camera1"..
on another), not a format constant. A recording whose manifest declares no
streams falls back to the ``Frame{N}`` naming convention.

Pixel encodings: DaVis names the encoding in {prefix}-decoder.xml. The full set of
ids DaVis 10/11 can write is in ``_DECODER_BITS_PER_PX`` below; ``mono-10p``,
``mono-12p``, ``raw-16-bit`` and ``raw-8-bit`` are decoded here. The rest are
recognised and refused by name -- their bit or channel order cannot be derived
from the id alone, and a wrong guess yields a plausible-looking wrong image rather
than a failure.

Stream transformers: StreamSet.xml declares a post-decode correction per stream
that DaVis applies on read (see ``_IMPLEMENTED_TRANSFORMERS``).
``dark-image-subtraction`` is applied as ``clamp(decoded - Transformer{N}-dark.im7,
0)`` and ``rotate-180`` as a 180-degree rotation. A declared correction this reader
does not implement is refused when that stream is read -- not skipped, because
skipping one returns different pixels from DaVis with nothing to indicate it
happened, and not at parse time, because the other streams of the recording are
still readable.

Also supports:
- Pre-paired pairs: entry[im_no].frames[2*(cam-1) + 0/1] via read_set_pair
- Single frames: read_set_frame; time-resolved pairing is assembled by callers
  from two single-frame reads at different entries (load_images.read_pair)
- Per-camera frame extraction with seek-based skipping

Reference: LaVision DaVis 10.x .set recording format
"""

import struct
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .out_buffer import check_out

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class IMSIndexEntry:
    """One entry in a Frame{N}-0.ims index file."""

    flag: int
    offset: int
    size: int


@dataclass(frozen=True)
class StreamTransformer:
    """One post-decode correction StreamSet.xml binds to a frame stream.

    ``id`` is the DaVis transformer id (``dark-image-subtraction``,
    ``rotate-180``, ...); ``label`` is what the DaVis UI shows for it, kept so a
    refusal names what the user can see. ``dark_path`` is set only for
    dark-image subtraction. A stream carries at most one transformer: the order
    DaVis applies two in is not knowable from the manifest, so two are refused
    at parse time rather than guessed.
    """

    id: str
    label: str
    dark_path: Optional[Path] = None


@dataclass
class IMSFrameInfo:
    """Metadata for one frame stream (one {prefix}-0/1.ims pair plus sidecars)."""

    frame_idx: int  # 0-based frame-stream index
    data_path: Path  # {prefix}-1.ims
    index_path: Path  # {prefix}-0.ims
    decoder: str  # DaVis encoding id, e.g. "mono-12p", "raw-16-bit"
    width: int
    height: int
    n_entries: int
    scale_slope: float
    scale_offset: float
    scale_unit: str
    entries: List[IMSIndexEntry]
    transformer: Optional[StreamTransformer] = None


@dataclass(frozen=True)
class _StreamSource:
    """Where one frame stream's files live, before any of them is opened.

    ``declared`` is True when the manifest (StreamSet.xml) named this stream,
    so a missing file is an incomplete copy and must raise. Under the glob
    fallback the scale file is a naming convention and its absence is
    legitimately "no scale".
    """

    index: int
    prefix: str
    scale_prefix: Optional[str]
    declared: bool


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
# post-decode correction; DaVis (and lvpyio) apply it on read, so skipping one
# silently returns different pixels from what DaVis shows for the same
# recording. An undeclared correction is normal -- plenty of recordings have
# none -- but a DECLARED one we do not implement must fail loudly rather than be
# dropped. The refusal is per STREAM, at read time (see _read_single_image): a
# five-camera recording whose PLIF camera carries an unimplemented correction
# still serves its four other cameras.
#
# rotate-180 declares a FilePrefix like every transformer but has no data file
# on disk; only dark-image-subtraction owns a Transformer{N}-dark.im7. Both
# verified bit-exact against lvpyio 1.3.1 on real recordings (2026-08-22 and
# 2026-08-24).
_IMPLEMENTED_TRANSFORMERS = ("dark-image-subtraction", "rotate-180")

# StreamSet.xml is the companion folder's manifest. ReaderInfo nodes declare each
# file group by FilePrefix and bind it to a frame-stream index through
# ContentPurpose StartFrame. The prefix is DaVis's choice per recording, not a
# format constant: "Frame0".. on one recording, "Camera1".. on another, with
# scale files "FrameScales0" / "CameraScale1" respectively. Both seen in the wild.
_FRAME_READER_TYPE = "Core.Set.Recording.FrameReader"
_SCALE_READER_TYPE = "Core.Set.Recording.ScaleReader"


def _load_stream_manifest(set_dir: Path) -> Optional[ET.Element]:
    """Parse StreamSet.xml once, or return None when the recording has none.

    Raises:
        ValueError: If the file exists but is not valid XML. The manifest drives
            both stream discovery and the corrections, so a broken one cannot be
            worked around.
    """
    stream_xml = set_dir / "StreamSet.xml"
    if not stream_xml.exists():
        return None
    try:
        return ET.parse(stream_xml).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{stream_xml} is not valid XML: {exc}") from exc


def _bound_stream_index(node: ET.Element, what: str, stream_xml: Path) -> int:
    """The frame-stream index a ReaderInfo/Transformer node is bound to.

    ``ContentPurpose StartFrame``/``EndFrame`` carry the binding. Every failure
    names the node and what is wrong: a bare ``int()`` traceback or a KeyError
    would be the only unnamed failure in this module.
    """
    purpose = node.find("ContentPurpose")
    if purpose is None:
        raise ValueError(
            f"{what} in {stream_xml} has no ContentPurpose, so the frame stream "
            f"it applies to is unknown."
        )
    start, end = purpose.get("StartFrame"), purpose.get("EndFrame")
    if start is None or end is None or start != end:
        raise ValueError(
            f"{what} in {stream_xml} spans frames {start}..{end}. This reader only "
            f"handles a declaration bound to exactly one frame stream."
        )
    try:
        index = int(start)
    except ValueError as exc:
        raise ValueError(
            f"{what} in {stream_xml} gives StartFrame '{start}', which is not a "
            f"frame-stream number. The stream it belongs to cannot be determined."
        ) from exc
    if index < 0:
        raise ValueError(
            f"{what} in {stream_xml} gives StartFrame {index}; frame-stream "
            f"indices start at 0."
        )
    return index


def _streams_from_manifest(
    set_dir: Path, root: ET.Element, frame_readers: List[ET.Element]
) -> List[_StreamSource]:
    """Frame streams as StreamSet.xml declares them, sorted by stream index.

    Callers locate a camera's stream positionally (``(camera-1) * fpc``), so the
    declared indices must be exactly 0..N-1. DaVis can write a sparse manifest
    when a camera is disabled mid-session; a gap is reported as such rather than
    silently shifting every camera after it.
    """
    stream_xml = set_dir / "StreamSet.xml"

    prefixes: dict = {}
    for node in frame_readers:
        prefix = (node.get("FilePrefix") or "").strip()
        if not prefix:
            raise ValueError(
                f"A FrameReader in {stream_xml} has no FilePrefix, so its stream "
                f"files cannot be located."
            )
        index = _bound_stream_index(node, f"FrameReader '{prefix}'", stream_xml)
        if index in prefixes:
            raise ValueError(
                f"{stream_xml.name} binds two image streams to frame stream "
                f"{index}: '{prefixes[index]}' and '{prefix}'. Camera mapping is "
                f"positional, so this cannot be resolved by choosing one."
            )
        prefixes[index] = prefix

    found = sorted(prefixes)
    if found != list(range(len(found))):
        raise ValueError(
            f"{stream_xml.name} declares StartFrames {found}; expected contiguous "
            f"0..{len(found) - 1}. Cameras are located positionally, so a gap "
            f"would silently shift every camera after it. DaVis writes a sparse "
            f"manifest when a camera is disabled mid-session -- check the "
            f"recording, this is its manifest, not a reader fault."
        )

    scale_prefixes: dict = {}
    for node in root.iter("ReaderInfo"):
        if node.get("Type") != _SCALE_READER_TYPE:
            continue
        prefix = (node.get("FilePrefix") or "").strip()
        index = _bound_stream_index(node, f"ScaleReader '{prefix}'", stream_xml)
        if index in scale_prefixes:
            raise ValueError(
                f"{stream_xml.name} binds two scale files to frame stream {index}: "
                f"'{scale_prefixes[index]}' and '{prefix}'."
            )
        scale_prefixes[index] = prefix

    return [
        _StreamSource(
            index=i, prefix=prefixes[i], scale_prefix=scale_prefixes.get(i), declared=True
        )
        for i in found
    ]


def _streams_from_glob(set_dir: Path) -> List[_StreamSource]:
    """Frame streams by the ``Frame{N}`` naming convention, for recordings whose
    StreamSet.xml is absent or declares no FrameReader (older exports)."""
    indices = sorted(
        {
            int(p.name.split("-")[0].replace("Frame", ""))
            for p in set_dir.glob("Frame*-1.ims")
        }
    )
    return [
        _StreamSource(
            index=i, prefix=f"Frame{i}", scale_prefix=f"FrameScales{i}", declared=False
        )
        for i in indices
    ]


def _discover_streams(set_dir: Path, root: Optional[ET.Element]) -> List[_StreamSource]:
    """Locate every frame stream: manifest-driven when the manifest names any,
    ``Frame{N}`` glob otherwise.

    The condition is "declares at least one FrameReader", not "StreamSet.xml
    exists": a manifest that only declares transformers (our synthetic fixtures,
    and possibly older DaVis exports) still relies on the naming convention.
    """
    if root is not None:
        frame_readers = [
            n for n in root.iter("ReaderInfo") if n.get("Type") == _FRAME_READER_TYPE
        ]
        if frame_readers:
            return _streams_from_manifest(set_dir, root, frame_readers)
    return _streams_from_glob(set_dir)


def _parse_stream_transformers(set_dir: Path, root: Optional[ET.Element]) -> dict:
    """Map frame-stream index to its declared correction, from StreamSet.xml.

    DaVis declares each correction as, for example::

        <Transformer ID="dark-image-subtraction" Label="Dark image subtraction"
                     FilePrefix="Transformer0" MinDaVisVersion="10.2.0">
            <ContentPurpose IsAssociatedToFrames="true" StartFrame="0" EndFrame="0"/>
        </Transformer>

    Unimplemented ids are recorded here and refused when THAT stream is read;
    structural faults are refused now, because a correction that cannot be bound
    to a stream cannot be deferred to one.

    Args:
        set_dir: The .set companion directory.
        root: Parsed StreamSet.xml, or None when the recording has none.

    Returns:
        dict: {stream index: StreamTransformer}. Empty when nothing is declared.

    Raises:
        ValueError: If a transformer's binding is missing, spans several streams,
            or a stream is given more than one transformer.
        FileNotFoundError: If a declared dark image is not on disk.
    """
    if root is None:
        return {}
    stream_xml = set_dir / "StreamSet.xml"

    transformers: dict = {}
    for node in root.iter("Transformer"):
        tid = (node.get("ID") or "").strip()
        label = (node.get("Label") or "").strip()
        prefix = (node.get("FilePrefix") or "").strip()
        stream_index = _bound_stream_index(node, f"Transformer '{prefix}'", stream_xml)

        dark_path = None
        if tid == "dark-image-subtraction":
            dark_path = set_dir / f"{prefix}-dark.im7"
            if not dark_path.exists():
                raise FileNotFoundError(
                    f"{stream_xml.name} declares dark-image subtraction for frame "
                    f"stream {stream_index}, but {dark_path.name} is missing from "
                    f"{set_dir}. Copy the complete companion folder -- the "
                    f"correction is part of the recording, not an optional extra."
                )

        if stream_index in transformers:
            # Which order DaVis applies two corrections in is not knowable from the
            # file, and dark subtraction and rotation do not commute, so neither
            # order is guessed.
            other = transformers[stream_index]
            raise ValueError(
                f"{stream_xml.name} declares two transformers for frame stream "
                f"{stream_index}: '{other.id}' and '{tid}'. This reader applies at "
                f"most one correction per stream and will not choose an order."
            )

        transformers[stream_index] = StreamTransformer(
            id=tid, label=label, dark_path=dark_path
        )

    return transformers


def _load_dark(dark_path_str: str) -> np.ndarray:
    """Load one Transformer{N}-dark.im7 as uint16, cached and revalidated by mtime.

    A dark image is the full sensor -- 37 MB as uint16 for a 5312x3528 camera --
    so the cache trades memory for not re-reading it on every pair, which would
    otherwise dominate read time. The cache is per process, which is what Dask
    workers need.

    The mtime is part of the cache key, not decoration. Keying on the path alone
    means re-exporting a recording leaves the previous dark image cached in a
    long-lived process -- the Flask GUI outlives many recordings -- and every
    subsequent read then subtracts the wrong pixels with nothing to indicate it.
    Silently wrong data is the one outcome this reader refuses everywhere else, so
    it pays one stat per frame to make it impossible. A superseded entry ages out
    of the LRU normally; it is never served again.

    Args:
        dark_path_str: Path to the dark .im7, as a string so it is hashable.

    Returns:
        np.ndarray: (H, W) uint16 dark image.

    Raises:
        ValueError: If the dark image is not 2D, or is not integer counts in
            [0, 65535] -- the range the uint16 subtraction below assumes.
        OSError: If the dark image is missing. _parse_stream_transformers already
            checks existence at parse time, so reaching this means it was deleted
            mid-session.
    """
    return _load_dark_cached(dark_path_str, Path(dark_path_str).stat().st_mtime)


@lru_cache(maxsize=4)
def _load_dark_cached(dark_path_str: str, mtime: float) -> np.ndarray:
    """Cached body of :func:`_load_dark`. Call that, not this.

    ``mtime`` is a cache key only and is deliberately unused in the body: a changed
    dark image produces a different key and therefore a fresh read.

    Size is a compromise, not a fit. Four entries hold two cameras' worth of
    pre-paired streams (~150 MB). A rig with three or more dark-bearing cameras
    whose worker interleaves them will evict and re-read; the alk235 recordings
    have ten such streams, so sizing to hold them all would cost ~370 MB per
    worker. Raise this only with a measurement showing the re-reads matter more
    than the resident memory.
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


def _read_scales(scales_path: Optional[Path]) -> Tuple[float, float, str]:
    """Read intensity scale from a {scaleprefix}.scales XML.

    ``None`` (the manifest declares no scale file for this stream) and a
    missing conventional file (glob fallback) both mean identity scale -- that
    is the declared state, not a fallback. A DECLARED file that is missing is
    caught before this is called.
    """
    if scales_path is None or not scales_path.exists():
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
# encodings share a byte count and differ only in bit order, the two 10-bit ones
# likewise, and rgb-24's channel order is not derivable from the id, so size alone
# cannot disambiguate them. Each needs a real sample file and a cross-check
# against lvpyio before it can be added.
#
# That cross-check IS a plain equality assertion, with two adjustments: lvpyio
# pads the frame height up to a multiple of 16 with zero rows (3528 -> 3536), and
# in a multi-camera buffer it pads every frame to the LARGEST frame's shape,
# unmasked (a 1984x1264 stream comes back as 2160x2560 with zeros outside the
# top-left block). Compare against its top-left `height` x `width` block.
# Everything else agrees bit for bit, declared corrections included. See
# [[piv-data-formats]].
_IMPLEMENTED_DECODERS = ("mono-10p", "mono-12p", "raw-16-bit", "raw-8-bit")


def _decode_mono10p(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode Mono10p (USB3 Vision packed 10-bit) bytes to uint16 array.

    Packing: 4 pixels in 5 bytes, least-significant bits first, as one continuous
    little-endian bit stream::

        pixel0 =  b0       | (b1 & 0x03) << 8
        pixel1 = (b1 >> 2) | (b2 & 0x0F) << 6
        pixel2 = (b2 >> 4) | (b3 & 0x3F) << 4
        pixel3 = (b3 >> 6) |  b4         << 2

    Each is a little-endian uint16 window over two adjacent bytes, shifted down
    by its bit offset and masked to 10 bits -- the same memory-access trick as
    :func:`_decode_mono12p`, four overlapping windows at stride 5 instead of two
    at stride 3. Verified bit-exact against lvpyio 1.3.1 on the 1984x1264
    MiniShaker streams of a real five-camera recording (2026-08-24).

    Raises:
        ValueError: If the pixel count is not a multiple of 4. Guarded here
            because the generic byte-count check in :func:`_decode_pixels`
            floor-divides ``w*h*10/8`` and would accept a truncated payload for a
            malformed geometry.
    """
    n_pixels = width * height
    if n_pixels % 4:
        raise ValueError(
            f"mono-10p packs four pixels into five bytes, so a frame must hold a "
            f"multiple of four pixels; {width}x{height} is {n_pixels}. The declared "
            f"frame geometry does not match a mono-10p payload."
        )
    n_quads = n_pixels // 4

    # The final window starts at byte 5*n_quads - 2 and reads two bytes, ending
    # exactly at the end of the payload -- no over-read.
    def window(offset: int) -> np.ndarray:
        return np.ndarray(
            (n_quads,), dtype="<u2", buffer=raw, offset=offset, strides=(5,)
        )

    pixels = np.empty((n_quads, 4), dtype=np.uint16)
    np.bitwise_and(window(0), 0x03FF, out=pixels[:, 0])
    np.right_shift(window(1), 2, out=pixels[:, 1])
    np.bitwise_and(pixels[:, 1], 0x03FF, out=pixels[:, 1])
    np.right_shift(window(2), 4, out=pixels[:, 2])
    np.bitwise_and(pixels[:, 2], 0x03FF, out=pixels[:, 2])
    # (b3 | b4<<8) >> 6 is at most 10 bits wide, so no mask is needed.
    np.right_shift(window(3), 6, out=pixels[:, 3])
    return pixels.reshape(height, width)


def _decode_mono12p(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode Mono12Packed (GigE Vision) bytes to uint16 array.

    Packing: 2 pixels in 3 bytes.
        pixel0 = byte0 | (byte1 & 0x0F) << 8
        pixel1 = (byte1 >> 4) | (byte2 << 4)

    Those two expressions are the same bytes read as two OVERLAPPING little-endian
    uint16 windows, because little-endian stores the low byte first::

        uint16 at byte 3k+0  =  b0 | b1<<8   ->  & 0x0FFF  ==  pixel0
        uint16 at byte 3k+1  =  b1 | b2<<8   ->  >> 4      ==  pixel1

    Reading the windows directly replaces three stride-3 byte-plane gathers and six
    full-frame temporaries with two gathers and two ufuncs writing straight into the
    output columns. This is a memory-access change, not an arithmetic one: measured
    on a 5312x3528 frame, 69.6 -> 17.5 ms on x86-64 and 26.4 -> 11.8 ms on an Apple
    M4, bit-identical to the byte-plane form on six edge-case bit patterns, on random
    full-frame data, and on real recordings.

    The windows are deliberately unaligned (numpy reports ``ALIGNED=False``). Free on
    x86-64, mildly penalised on ARM -- which is why the ARM gain is the smaller one --
    but this remains the fastest of the tried formulations on both. ``<u2`` pins the
    byte order, so a big-endian host still decodes correctly, just via numpy's
    byte-swapping path.

    Raises:
        ValueError: If the frame holds an odd number of pixels. mono-12p packs two
            pixels per three bytes, so an odd count means the container's declared
            geometry is wrong. Guarded explicitly because the pair-wise form below
            would otherwise drop the trailing pixel silently.
    """
    n_pixels = width * height
    if n_pixels % 2:
        raise ValueError(
            f"mono-12p packs two pixels into three bytes, so a frame must hold an "
            f"even number of pixels; {width}x{height} is {n_pixels}. The declared "
            f"frame geometry does not match a mono-12p payload."
        )
    n_pairs = n_pixels // 2

    # The final window starts at byte 3*n_pairs - 2 and reads two bytes, ending
    # exactly at the end of the payload -- no over-read. A short buffer raises from
    # np.ndarray itself rather than decoding garbage.
    lo = np.ndarray((n_pairs,), dtype="<u2", buffer=raw, offset=0, strides=(3,))
    hi = np.ndarray((n_pairs,), dtype="<u2", buffer=raw, offset=1, strides=(3,))

    pixels = np.empty((n_pairs, 2), dtype=np.uint16)
    np.bitwise_and(lo, 0x0FFF, out=pixels[:, 0])
    np.right_shift(hi, 4, out=pixels[:, 1])
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

    if decoder == "mono-10p":
        return _decode_mono10p(raw, width, height)
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

    root = _load_stream_manifest(set_dir)
    sources = _discover_streams(set_dir, root)

    if not sources:
        _raise_no_streams(set_dir)

    # Post-decode corrections DaVis declares for this recording (usually none).
    transformers = _parse_stream_transformers(set_dir, root)

    frames = []
    n_entries = None

    for src in sources:
        index_path = set_dir / f"{src.prefix}-0.ims"
        data_path = set_dir / f"{src.prefix}-1.ims"
        decoder_path = set_dir / f"{src.prefix}-decoder.xml"

        for required in (index_path, data_path):
            if not required.exists():
                if src.declared:
                    raise FileNotFoundError(
                        f"StreamSet.xml declares frame stream {src.index} with "
                        f"prefix '{src.prefix}', but {required.name} is missing "
                        f"from {set_dir}. Copy the complete companion folder."
                    )
                raise FileNotFoundError(f"Data file missing: {required}")

        scales_path = None
        if src.scale_prefix is not None:
            scales_path = set_dir / f"{src.scale_prefix}.scales"
            if src.declared and not scales_path.exists():
                raise FileNotFoundError(
                    f"StreamSet.xml declares scale file '{src.scale_prefix}' for "
                    f"frame stream {src.index}, but {scales_path.name} is missing "
                    f"from {set_dir}. Copy the complete companion folder."
                )

        width, height, n_ent, entries = _parse_ims_index(index_path)
        decoder = _read_decoder(decoder_path)
        slope, offset, unit = _read_scales(scales_path)

        if n_entries is None:
            n_entries = n_ent
        elif n_ent != n_entries:
            raise ValueError(
                f"{src.prefix} has {n_ent} entries but {sources[0].prefix} "
                f"has {n_entries}"
            )

        frames.append(
            IMSFrameInfo(
                frame_idx=src.index,
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
                transformer=transformers.get(src.index),
            )
        )

    return SetInfo(set_dir=set_dir, frames=frames, n_entries=n_entries)


def _raise_no_streams(set_dir: Path) -> None:
    """Name which non-recording shape a stream-less companion folder is.

    DaVis reuses the .set extension for project and calibration nodes, stores
    some recordings as .im7 files, and names stream files by a per-recording
    prefix the manifest must declare -- each has a different fix, and the bare
    "no data files" message fitted none of them.
    """
    ims_prefixes = sorted({p.name[: -len("-1.ims")] for p in set_dir.glob("*-1.ims")})
    if ims_prefixes:
        shown = ", ".join(ims_prefixes[:5])
        raise FileNotFoundError(
            f"{set_dir} holds image streams ({shown}) whose prefix is not "
            f"'Frame{{N}}', and its StreamSet.xml does not declare them (no "
            f"FrameReader entry). The stream-to-camera mapping comes from that "
            f"manifest, so it cannot be inferred from the file names. Re-export the "
            f"recording from DaVis so the companion folder carries its StreamSet.xml."
        )

    # Transformer data files (Transformer1-scmos-1.im7 and the like) are .im7 too,
    # and are NOT evidence of .im7 layout; advising a switch to lavision_im7 on
    # their account would read a correction map as image data.
    im7_files = sorted(
        p for p in set_dir.glob("*.im7") if not p.name.startswith("Transformer")
    )
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
        f"{set_dir} holds no image data -- no *-1.ims streams and no .im7 files. "
        f"This is a DaVis project or calibration node, not an image recording. "
        f"Found: {', '.join(found[:5])}"
        f"{f' (+{len(found) - 5} more)' if len(found) > 5 else ''}. Point at a "
        f"recording .set whose companion folder holds the image streams."
    )


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

    # Refuse before the 37 MB read, not after it. Per stream: a container whose
    # other streams carry only implemented corrections stays readable.
    tf = frame_info.transformer
    if tf is not None and tf.id not in _IMPLEMENTED_TRANSFORMERS:
        container = frame_info.data_path.parent.name
        raise ValueError(
            f"Frame stream {frame_info.frame_idx} of {container} "
            f"declares the stream transformer '{tf.id}' (DaVis label '{tf.label}'), "
            f"which PIVTOOLs does not implement. Skipping it would return different "
            f"pixels from DaVis for the same recording. PIVTOOLs implements: "
            f"{', '.join(_IMPLEMENTED_TRANSFORMERS)}. Other streams of this "
            f"recording are unaffected."
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

    img = _decode_pixels(raw, frame_info.decoder, frame_info.width, frame_info.height)

    if tf is None:
        return img

    if tf.id == "rotate-180":
        # A negative-stride view, deliberately not materialised: every caller
        # performs exactly one full-frame widening copy into float32 (result[i] =
        # img / np.copyto / astype), which absorbs arbitrary source strides for
        # free, and none of them mutates the returned array (the raw-16-bit path
        # already returns a read-only frombuffer view on that same contract).
        # Verified bit-exact against lvpyio 1.3.1 on real entries (2026-08-24).
        return img[::-1, ::-1]

    # Dark-image subtraction, clamped at zero -- DaVis and lvpyio both floor the
    # result, and on this data ~0.001% of pixels would otherwise go negative.
    # Verified bit-exact against lvpyio 1.3.1 across three frame streams and two
    # entries of E:\Softwarex\alk235\images\loop=0.set (2026-08-22).
    dark = _load_dark(str(tf.dark_path))
    if dark.shape != img.shape:
        raise ValueError(
            f"Dark image {tf.dark_path.name} is {dark.shape} but frame "
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


# ---------------------------------------------------------------------------
# Parsed-container cache
# ---------------------------------------------------------------------------

# Parsing a .set opens every stream's index file and its two XML sidecars: 21 opens
# and ~43 stat calls on a 10-stream recording. read_pair needs that same metadata
# three times for one time-resolved pair (once to derive frames-per-camera, once per
# frame), so an uncached read spends 65 opens and 128 stats to deliver two frames --
# against 3 opens for the equivalent .im7 pair. Caching reduces it to two stats.
#
# Same shape as the .cine reader's _metadata_cache (see readers/cine_reader.py):
# per-process, keyed by path, invalidated on mtime.
#
# Bounded, unlike the .cine one. A 20,000-entry 10-stream SetInfo holds 200,000
# IMSIndexEntry objects, and each Dask worker gets its own copy of this dict, so an
# unbounded cache would let a long time-resolved run pin hundreds of MB per worker.
# Four covers a multi-loop acquisition's working set. Same reasoning, and the same
# "raise it only with a measurement" caveat, as _load_dark's lru_cache above.
_SET_INFO_CACHE_SIZE = 4
_set_info_cache: "OrderedDict[str, Tuple[Tuple[float, float], SetInfo]]" = (
    OrderedDict()
)

# The Flask server is threaded, so this dict needs a lock. Each OrderedDict operation
# is individually atomic under the GIL, but the sequences here are not: a hit's
# move_to_end can race another thread's eviction and raise KeyError out of what
# should be a read. The Dask path cannot hit it (piv_cluster pins
# threads_per_worker=1) and functools.lru_cache is internally locked; only this dict
# is exposed. _parse_set deliberately runs OUTSIDE the lock -- it is slow file I/O,
# and two threads racing to parse the same container is wasteful, never wrong.
_set_info_lock = threading.Lock()


def _set_info_cache_key(set_path: Path) -> Tuple[float, float]:
    """Modification times that must both still hold for a cached SetInfo to be valid.

    The .set file alone is not enough. It is a few hundred bytes of project XML and
    does not change when the streams in its companion folder do. The folder's mtime
    moves whenever a stream file is added or removed.

    KNOWN LIMITATION, accepted rather than hidden: rewriting a {prefix}-0.ims or
    StreamSet.xml in place changes neither mtime, so a cached SetInfo would
    survive it. Acquisition data is written once and read many times, so paying
    two stats per read is the right trade. Call :func:`clear_set_info_cache` if a
    container is ever edited in place during a session.

    Raises:
        OSError: If the .set or its companion directory is missing. Callers should
            fall through to _parse_set, which diagnoses that case by name.
    """
    return (set_path.stat().st_mtime, set_path.with_suffix("").stat().st_mtime)


def clear_set_info_cache() -> None:
    """Drop every cached SetInfo. Mirrors cine_reader.clear_metadata_cache()."""
    with _set_info_lock:
        _set_info_cache.clear()


def read_set_info(set_path: Union[str, Path]) -> SetInfo:
    """Parse .set metadata without reading any pixel data.

    Useful for getting entry count, dimensions, frame count. Results are cached per
    process and revalidated by mtime, so repeated calls for the same container cost
    two stat calls rather than a full re-parse. See :func:`_set_info_cache_key` for
    what invalidates an entry and the one case it cannot see.
    """
    set_path = Path(set_path)

    try:
        key = _set_info_cache_key(set_path)
    except OSError:
        # Missing .set or companion folder. Let _parse_set raise: it names which of
        # the non-recording .set shapes this is, and each has a different fix. A
        # bare stat error from here would lose that.
        return _parse_set(set_path)

    cache_id = str(set_path.absolute())
    with _set_info_lock:
        cached = _set_info_cache.get(cache_id)
        if cached is not None and cached[0] == key:
            _set_info_cache.move_to_end(cache_id)
            return cached[1]

    info = _parse_set(set_path)  # outside the lock: slow I/O, idempotent

    with _set_info_lock:
        _set_info_cache[cache_id] = (key, info)
        _set_info_cache.move_to_end(cache_id)
        while len(_set_info_cache) > _SET_INFO_CACHE_SIZE:
            _set_info_cache.popitem(last=False)
    return info


def get_set_entry_count(set_path: Union[str, Path]) -> int:
    """Get the number of entries (image pairs) in a .set file."""
    return read_set_info(set_path).n_entries


def read_set_pair(
    set_path: Union[str, Path],
    camera_no: int,
    im_no: int,
    set_info: Optional[SetInfo] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Read a pre-paired frame pair from a .set container.

    Reads frames[2*(camera_no-1)] and frames[2*(camera_no-1)+1] from entry
    im_no. Time-resolved pairing does NOT live here: it is two single-frame
    reads at different entries, and the production path builds it from
    read_set_frame (see load_images.read_pair), which derives the camera
    stride instead of assuming it.

    Parameters
    ----------
    set_path : str or Path
        Path to the .set file.
    camera_no : int
        Camera number (1-based).
    im_no : int
        Image/entry number (1-based).
    set_info : SetInfo, optional
        Pre-parsed container metadata. If provided, skips re-parsing the
        index/XML files. Use read_set_info() once, then pass it here for
        every pair in a batch.
    out : np.ndarray, optional
        Destination (2, H, W) float32 C-contiguous buffer, typically one slot of
        a batch. Validated by :func:`readers.out_buffer.check_out`, written in
        place and returned; undefined after an exception.

    Returns
    -------
    np.ndarray
        Array of shape (2, H, W), dtype float32, with intensity scale applied.
        ``out`` itself when given.
    """
    # read_set_info, not _parse_set: callers that do not thread set_info through
    # (the pre-paired PIV path and the calibration loader) would otherwise re-parse
    # the whole container on every single frame.
    info = set_info if set_info is not None else read_set_info(set_path)

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

    # Streams legitimately differ in shape across cameras (alk235's five span
    # 3472-3536 rows), so an A/B pair from mismatched streams is a wrong camera
    # mapping, not a broadcast error to be read off a numpy traceback.
    if (fi_a.height, fi_a.width) != (fi_b.height, fi_b.width):
        raise ValueError(
            f"Camera {camera_no} pairs frame streams {frame_idx_a} "
            f"({fi_a.height}x{fi_a.width}) and {frame_idx_b} "
            f"({fi_b.height}x{fi_b.width}), which differ in shape. These are not "
            f"one camera's A/B streams -- check camera_count against the recording."
        )

    if out is None:
        result = np.empty((2, fi_a.height, fi_a.width), dtype=np.float32)
    else:
        result = check_out(
            out,
            (2, fi_a.height, fi_a.width),
            f"read_set_pair camera {camera_no} of {Path(set_path).name}",
        )

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
    out: Optional[np.ndarray] = None,
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
    out : np.ndarray, optional
        Destination (H, W) float32 array to decode into, typically one slice of a
        caller's ``(2, H, W)`` pair buffer. Without it this allocates a fresh frame
        via ``astype``, and a caller stacking two frames then copies both again --
        two full-frame passes where one suffices. Writing into ``out`` performs the
        uint16-to-float32 widening straight into its final location, which is the
        same thing :func:`read_set_pair` does with ``result[0] = img_a``.

    Returns
    -------
    np.ndarray
        Image (H, W) float32 with intensity scale applied. ``out`` itself when given.

    Raises
    ------
    ValueError
        If ``frame_idx`` is out of range, or ``out`` does not match the frame
        stream's shape or is not float32.
    """
    # read_set_info, not _parse_set: callers that do not thread set_info through
    # (the pre-paired PIV path and the calibration loader) would otherwise re-parse
    # the whole container on every single frame.
    info = set_info if set_info is not None else read_set_info(set_path)

    if frame_idx < 0 or frame_idx >= len(info.frames):
        raise ValueError(
            f"Frame index {frame_idx} out of range [0, {len(info.frames)})"
        )

    fi = info.frames[frame_idx]

    if out is None:
        result = None
    else:
        if out.shape != (fi.height, fi.width):
            raise ValueError(
                f"out has shape {out.shape}, but frame stream {fi.frame_idx} of "
                f"{Path(set_path).name} is {(fi.height, fi.width)}."
            )
        if out.dtype != np.float32:
            raise ValueError(
                f"out has dtype {out.dtype}; read_set_frame produces float32."
            )
        result = out

    img = _read_single_image(fi, entry_no - 1)

    if result is None:
        result = img.astype(np.float32)
    else:
        # Widens uint16 -> float32 directly into the caller's buffer.
        np.copyto(result, img)
    del img

    _apply_scale_inplace(result, fi)
    return result
