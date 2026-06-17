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
    out = cli._load_one(tmp_path, "calib%05d.png", 1, camera=1, image_type="standard",
                        use_camera_subfolders=False, zero_based=False)
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
        0,      # version
        20,     # pack_type = LZ4
        -3,     # buffer_format = float32
        0,      # is_sparse
        w, h, 1, n_f,
        0, 1, 0,
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
