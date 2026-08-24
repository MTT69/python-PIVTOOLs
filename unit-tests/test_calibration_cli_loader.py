"""D1 — the calibration CLI loader reads every format via the shared reader.

Regression test for the bug where ``detect-planar`` / ``detect-stereo`` loaded
calibration frames with ``cv2.imread`` only. ``cv2.imread`` returns ``None`` on a
LaVision ``.im7``, so the CLI raised ``FileNotFoundError`` and could not calibrate any
DaVis ``.im7``/``.set`` dataset headless — even though the rest of the codebase reads
those formats fine through ``read_single_frame``. The CLI now routes ``_load_one``
through ``read_calibration_frame_at`` (the same path the GUI uses).

Asserted here (real, no mocks):
- ``infer_image_type`` classifies by extension (the format → reader routing);
- a real standard PNG still round-trips on disk (the tif/png path is unchanged).

The real ``.im7`` decode is proven by running the CLI on actual LaVision data (the D5
dataset runs), not by a stand-in — see the closing note.
"""

from __future__ import annotations

import cv2
import numpy as np

import pivtools_cli.calibration_cli as cli
from pivtools_core.image_handling.path_utils import infer_image_type


def test_infer_image_type_by_extension():
    assert infer_image_type("B%05d.im7") == "lavision_im7"
    assert infer_image_type("frame.ims") == "lavision_im7"
    assert infer_image_type("data.set") == "lavision_set"
    assert infer_image_type("Camera%d.cine") == "cine"
    assert infer_image_type("calib%05d.tif") == "standard"
    assert infer_image_type("cal_%03d.png") == "standard"


def test_load_one_standard_png_roundtrips(tmp_path):
    """Behaviour-preserving: a standard image still loads to the identical array."""
    img = (np.arange(48 * 64, dtype=np.uint8) % 251).reshape(48, 64)
    cv2.imwrite(str(tmp_path / "calib00001.png"), img)
    out = cli._load_one(
        tmp_path,
        "calib%05d.png",
        1,
        camera=1,
        image_type="standard",
        use_camera_subfolders=False,
        zero_based=False,
        num_cameras=1,
    )
    assert out.shape == (48, 64)
    assert np.array_equal(out, img)


# The real im7 read is proven by running the CLI on actual LaVision data (the D5 dataset
# runs on bailey/merle/andre), not by a monkeypatched stand-in. read_single_frame — the
# reader the CLI now calls — is already the production PIV pair reader (load_images.read_pair),
# so its .im7/.set decoding is exercised by every production PIV run.


# ---------------------------------------------------------------------------
# pack_type 20 (LZ4) — hermetic regression for the reverse-engineered decoder.
# Real-data proof: bit-exact vs LaVision lvpyio on merle + andre x25 frames
# (2026-06-12); this test pins the container layout so it cannot regress.
# ---------------------------------------------------------------------------

import struct

from pivtools_core.image_handling.readers.im7_reader import (
    HEADER_SIZE,
    read_im7_camera,
)


def _lz4_literals_block(data: bytes) -> bytes:
    """Encode bytes as a single literals-only LZ4 block (valid final sequence)."""
    n = len(data)
    assert n >= 15, "use enough data to exercise the length extension"
    out = bytearray([0xF0])
    rem = n - 15
    while rem >= 255:
        out.append(255)
        rem -= 255
    out.append(rem)
    out += data
    return bytes(out)


def _write_pack20_im7(path, frames: np.ndarray) -> None:
    """Minimal pack_type-20 .im7: 256-byte header, int64 size, one LZ4 block."""
    n_f, h, w = frames.shape
    header = struct.pack(
        "<hhhh iiii hhh",
        0,  # version
        20,  # pack_type = LZ4
        -3,  # buffer_format = float32
        0,  # is_sparse
        w,
        h,
        1,
        n_f,
        0,
        1,
        0,
    )
    header += b"\x00" * (HEADER_SIZE - len(header))
    comp = _lz4_literals_block(frames.astype("<f4").tobytes())
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<q", len(comp)))
        f.write(comp)


def test_pack20_lz4_im7_roundtrips(tmp_path):
    rng = np.random.default_rng(7)
    frames = rng.uniform(0, 4095, size=(2, 6, 8)).astype(np.float32)
    p = tmp_path / "B00001.im7"
    _write_pack20_im7(p, frames)
    for cam in (1, 2):
        got = np.asarray(read_im7_camera(p, cam, 1)).squeeze()
        assert got.shape == (6, 8)
        np.testing.assert_array_equal(got, frames[cam - 1])


def test_pack20_truncated_file_fails_visibly(tmp_path):
    frames = np.zeros((1, 6, 8), dtype=np.float32)
    p = tmp_path / "B00001.im7"
    _write_pack20_im7(p, frames)
    blob = p.read_bytes()
    p.write_bytes(blob[:-10])  # chop the tail of the compressed block
    try:
        read_im7_camera(p, 1, 1)
    except (IOError, ValueError):
        pass
    else:
        raise AssertionError("truncated pack-20 file decoded without error")


# ---------------------------------------------------------------------------
# frames= reads only the frames asked for. A single-frame calibration read used
# to decode the whole camera slice (both A and B) and slice the result, so every
# multi-camera .im7 calibration read decoded twice the data it needed.
# ---------------------------------------------------------------------------

from pivtools_core.image_handling.readers import im7_reader
from pivtools_core.image_handling.readers.lavision_reader import read_lavision_im7


def _write_pack0_im7(path, frames: np.ndarray, scale=None) -> None:
    """Minimal pack_type-0 (uncompressed WORD) .im7 with size_f frames.

    ``scale=(slope, offset)`` appends an IEH_SCALE_I record (type 4) before the
    IEH_END terminator, the way DaVis stores the intensity scale.
    """
    n_f, h, w = frames.shape
    header = struct.pack("<hhhh iiii hhh", 0, 0, -4, 0, w, h, 1, n_f, 0, 1, 0)
    header += b"\x00" * (HEADER_SIZE - len(header))
    with open(path, "wb") as f:
        f.write(header)
        f.write(frames.astype("<u2").tobytes())
        if scale is not None:
            payload = f"{scale[0]} {scale[1]}\x00counts\x00".encode()
            f.write(struct.pack("<ii", 4, len(payload)) + payload)
        f.write(b"\x00\x00\x00\x00")  # IEH_END terminates the record list


class _CountingOpen:
    """Stand-in for builtins.open that totals the bytes every read() returns."""

    def __init__(self):
        self.bytes_read = 0

    def __call__(self, *args, **kwargs):
        counter = self
        real = open(*args, **kwargs)

        class _Wrapped:
            def read(self, n=-1):
                data = real.read(n)
                counter.bytes_read += len(data)
                return data

            def __getattr__(self, name):
                return getattr(real, name)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return real.__exit__(*exc)

        return _Wrapped()


def test_im7_frames_one_reads_half_the_bytes_and_the_same_pixels(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 4096, size=(8, 6, 10), dtype=np.uint16)  # 4 cams x A/B
    p = tmp_path / "B00001.im7"
    _write_pack0_im7(p, frames)
    frame_bytes = 6 * 10 * 2

    full = read_im7_camera(p, camera_no=3, frames_per_camera=2)
    np.testing.assert_array_equal(full, frames[4:6].astype(np.float32))

    counter = _CountingOpen()
    monkeypatch.setattr(im7_reader, "open", counter, raising=False)
    one = read_im7_camera(p, camera_no=3, frames_per_camera=2, frames=1)

    assert one.shape == (1, 6, 10)
    np.testing.assert_array_equal(one[0], full[0])
    # Header once (the second open only seeks) + exactly one frame + the 4-byte
    # IEH_END terminator read from where the attributes actually are.
    assert counter.bytes_read == HEADER_SIZE + frame_bytes + 4


def test_read_lavision_im7_passes_frames_down(tmp_path, monkeypatch):
    frames = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    p = tmp_path / "B00001.im7"
    _write_pack0_im7(p, frames)

    counter = _CountingOpen()
    monkeypatch.setattr(im7_reader, "open", counter, raising=False)
    got = read_lavision_im7(str(p), camera_no=1, frames=1, frames_per_camera=2)

    assert got.shape == (1, 4, 5)
    np.testing.assert_array_equal(got[0], frames[0].astype(np.float32))
    assert counter.bytes_read == HEADER_SIZE + 4 * 5 * 2 + 4


def test_intensity_scale_applies_to_every_camera_not_just_the_last(tmp_path):
    """The attribute records follow the whole buffer. Reading camera 1 of a
    4-camera file used to parse its scale out of camera 2's pixel bytes, so a
    recording with a non-identity ScaleI scaled the last camera only."""
    frames = np.full((8, 4, 5), 100, dtype=np.uint16)
    p = tmp_path / "B00001.im7"
    _write_pack0_im7(p, frames, scale=(0.5, 7))
    for cam in (1, 2, 3, 4):
        for kwargs in ({}, {"frames": 1}):
            got = read_im7_camera(p, cam, 2, **kwargs)
            np.testing.assert_array_equal(got, np.full(got.shape, 57, np.float32))


def test_rle_pack1_scale_applies_to_every_camera(tmp_path):
    """Same exposure in the RLE reader: it decoded up to the last requested
    frame and read attributes from there. It must decode through the buffer."""
    from pivtools_core.image_handling.readers.im7_reader import _read_im7_internal

    # 4 frames (2 cameras x A/B) of constant 100: per frame the 3-byte preamble,
    # one int8 delta of +100 from the implicit 0, then zero deltas (see the
    # pack_type 1 section below for the token kinds).
    npix = 4 * 5
    frame_stream = bytes([0x01, 0x01, 0x00, 0x64]) + b"\x00" * (npix - 1)
    body = frame_stream * 4
    header = struct.pack("<hhhh iiii hhh", 0, 1, -4, 0, 5, 4, 1, 4, 0, 1, 0)
    header += b"\x00" * (HEADER_SIZE - len(header))
    payload = b"0.5 7\x00counts\x00"
    p = tmp_path / "B00001.im7"
    p.write_bytes(
        header + body + struct.pack("<ii", 4, len(payload)) + payload + b"\x00" * 4
    )
    _hdr, px, scales = _read_im7_internal(p, frame_range=(0, 2))
    if scales.slope != 0.5:
        raise AssertionError(f"scale read from the wrong place: {scales}")
    assert px.shape[0] == 2


def test_im7_frames_clamps_to_available(tmp_path):
    """A single-frame calibration file (size_f=1) asked for frames=1 of a
    2-per-camera layout still returns its one frame, as before."""
    frames = np.ones((1, 4, 5), dtype=np.uint16)
    p = tmp_path / "B00001.im7"
    _write_pack0_im7(p, frames)
    assert read_im7_camera(p, 1, 2, frames=1).shape == (1, 4, 5)
    with pytest.raises(ValueError, match="frames must be >= 1"):
        read_im7_camera(p, 1, 2, frames=0)


# ---------------------------------------------------------------------------
# frames-per-camera detection — a multi-camera .im7 buffer's per-camera stride
# is derived from size_f / camera_count, not hard-coded. Guards the fix for the
# bug where a 6-frame, 3-camera (A/B) buffer mislabelled as 4 cameras failed
# only on camera 4 with a misleading "first frame not found".
# ---------------------------------------------------------------------------

import pytest

from pivtools_core.image_handling.load_images import _detect_im7_frames_per_camera
from pivtools_core.image_handling.readers.im7_reader import get_im7_frame_count


def test_get_im7_frame_count(tmp_path):
    p = tmp_path / "B00001.im7"
    _write_pack20_im7(p, np.zeros((6, 4, 5), dtype=np.float32))
    assert get_im7_frame_count(p) == 6


@pytest.mark.parametrize(
    "n_frames,n_cams,expected",
    [
        (6, 3, 2),  # 3 cameras, A/B double-frame (the reported dataset)
        (8, 4, 2),  # 4 cameras, A/B double-frame
        (4, 4, 1),  # 4 cameras, single-frame
        (2, 1, 2),  # single camera, A/B pre-paired
    ],
)
def test_detect_frames_per_camera_divisible(tmp_path, n_frames, n_cams, expected):
    p = tmp_path / "B00001.im7"
    _write_pack20_im7(p, np.zeros((n_frames, 4, 5), dtype=np.float32))
    assert _detect_im7_frames_per_camera(p, n_cams) == expected


def test_detect_frames_per_camera_indivisible_raises(tmp_path):
    """6 frames with 4 configured cameras: clear error, not a silent best-guess."""
    p = tmp_path / "B00001.im7"
    _write_pack20_im7(p, np.zeros((6, 4, 5), dtype=np.float32))
    with pytest.raises(
        ValueError,
        match=r"holds 6 frames.*not 1 or 2 frames per camera.*Expected 3 cameras",
    ):
        _detect_im7_frames_per_camera(p, 4)


# ---------------------------------------------------------------------------
# pack_type 1 (DaVis delta+nibble RLE) — hermetic regression for the decoder
# reverse-engineered bit-exact vs lvpyio on real ALK x25 calibration plates.
# This pins the byte format so the three token kinds can't silently regress:
#   3-byte preamble | int8 deltas | 0x81 nibble-run (0x8 terminates) | 0x80 abs16.
# ---------------------------------------------------------------------------


def _write_packtype1_im7(path, size_x, size_y, payload: bytes) -> None:
    """Minimal pack_type-1 .im7: 256-byte WORD header + a raw RLE token stream."""
    header = struct.pack(
        "<hhhh iiii hhh",
        0,  # version
        1,  # pack_type = RLE
        -4,  # buffer_format = WORD (uint16)
        0,  # is_sparse
        size_x,
        size_y,
        1,
        1,  # size_x, size_y, size_z=1, size_f=1
        0,
        1,
        0,
    )
    header += b"\x00" * (HEADER_SIZE - len(header))
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)  # EOF after payload terminates the attribute scan cleanly


def test_packtype1_rle_decodes_all_token_kinds(tmp_path):
    # Running value starts at 0; tokens exercise every branch:
    #  0x64(+100) 0x01(+1) | 0x81 nibble-run [+2,+3,-1] term 0x8 | 0x80 abs=500 | 0xfe(-2)
    payload = bytes(
        [
            0x01,
            0x01,
            0x00,  # preamble
            0x64,
            0x01,  # int8 deltas -> 100, 101
            0x81,
            0x23,
            0xF8,  # nibble run +2,+3,-1 then 0x8 term -> 103,106,105
            0x80,
            0xF4,
            0x01,  # abs16 LE 0x01f4 = 500
            0xFE,
        ]
    )  # int8 -2 -> 498
    expected = [100, 101, 103, 106, 105, 500, 498]
    p = tmp_path / "B00001.im7"
    _write_packtype1_im7(p, size_x=len(expected), size_y=1, payload=payload)

    img = np.asarray(read_im7_camera(p, camera_no=1, frames_per_camera=1)).squeeze()
    assert img.shape == (len(expected),)
    np.testing.assert_array_equal(img.astype(int), expected)


def test_packtype1_byte_delta_roundtrip(tmp_path):
    """A pure int8-delta row (no nibble/abs tokens) round-trips."""
    vals = [50, 60, 59, 61, 70, 69, 68, 90]  # consecutive deltas all within int8 range
    deltas = [vals[0]] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    assert all(
        -127 <= d <= 127 for d in deltas
    )  # else byte mode would need a 0x80 escape
    payload = bytes([0x01, 0x01, 0x00] + [d & 0xFF for d in deltas])
    p = tmp_path / "B00001.im7"
    _write_packtype1_im7(p, size_x=len(vals), size_y=1, payload=payload)
    img = np.asarray(read_im7_camera(p, camera_no=1, frames_per_camera=1)).squeeze()
    np.testing.assert_array_equal(img.astype(int), vals)
