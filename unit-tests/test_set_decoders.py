#!/usr/bin/env python3
"""
test_set_decoders.py

Pixel-encoding support and container diagnostics for the LaVision .set reader.

Background: a user's two-camera .set failed validation with "First frame not found"
while the container was present and intact. The real cause was
``ValueError: Unsupported pixel decoder: 'raw-16-bit'`` — the reader knew only
``mono-12p`` and ``mono-16``. Scanning LaVision's own lvpyio 1.3.1 binaries
recovered the full DaVis 10/11 encoding table (eight ids); ``mono-16`` is not in it
at all, it was our own placeholder returned when Frame{N}-decoder.xml was missing,
which silently decoded packed data as unpacked.

These tests lock in:
  * ``raw-16-bit`` and ``raw-8-bit`` decode, and ``mono-12p`` still decodes.
  * ``raw-8-bit`` widens without rescaling (raw sensor counts, as mono-12p does).
  * A known-but-unimplemented encoding is refused BY NAME, not silently aliased.
  * The byte count is checked against the id's bit depth before decoding.
  * A missing or empty decoder XML raises instead of guessing ``mono-16``.
  * The three non-recording .set shapes each name themselves.

Usage:
    pytest unit-tests/test_set_decoders.py -v
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_core.image_handling.readers.set_reader import (  # noqa: E402
    _IMPLEMENTED_DECODERS,
    read_set_frame,
    read_set_info,
    read_set_pair,
)

# Index layout, mirroring _parse_ims_index: 256-byte header (width at offset 12,
# height at 16) then 768 bytes of padding, then 20-byte entries.
_TABLE_OFFSET = 256 + 768
_ENTRY_STRUCT = "<iqq"  # flag int32, offset int64, size int64


def _build_set(
    tmp_path,
    decoder,
    streams,
    width,
    height,
    name="synthetic",
    write_decoder_xml=True,
    declared_sizes=None,
):
    """Write a minimal .set container and return the path to the .set file.

    Args:
        tmp_path: pytest tmp_path.
        decoder: encoding id written into Frame{N}-decoder.xml.
        streams: list (one per frame stream) of lists of entry payload bytes.
        width: frame width written into the index header.
        height: frame height written into the index header.
        name: container stem.
        write_decoder_xml: when False, omit the decoder XML entirely.
        declared_sizes: optional per-stream list of per-entry sizes to declare in
            the index, overriding the true payload length. Used to build a
            container whose declared geometry disagrees with its payload.

    Returns:
        Path: the .set file (its companion directory sits alongside).
    """
    set_file = tmp_path / f"{name}.set"
    set_file.write_bytes(b"stub")
    set_dir = tmp_path / name
    set_dir.mkdir()

    for stream_idx, entries in enumerate(streams):
        index = bytearray(_TABLE_OFFSET)
        struct.pack_into("<i", index, 12, width)
        struct.pack_into("<i", index, 16, height)

        offset = 0
        for entry_idx, payload in enumerate(entries):
            size = len(payload)
            if declared_sizes is not None:
                size = declared_sizes[stream_idx][entry_idx]
            index += struct.pack(_ENTRY_STRUCT, 0, offset, size)
            offset += len(payload)

        (set_dir / f"Frame{stream_idx}-0.ims").write_bytes(bytes(index))
        (set_dir / f"Frame{stream_idx}-1.ims").write_bytes(b"".join(entries))

        if write_decoder_xml:
            (set_dir / f"Frame{stream_idx}-decoder.xml").write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                f"<FrameDecoder>\n<id>{decoder}</id>\n</FrameDecoder>\n"
            )

    return set_file


# ---------------------------------------------------------------------------
# Encodings that must decode
# ---------------------------------------------------------------------------


def test_raw16_round_trips(tmp_path):
    """'raw-16-bit' — the encoding the user's DaVis wrote — decodes as <u2."""
    expected = np.array([[0, 3, 1021, 65535], [7, 256, 4095, 12]], dtype=np.uint16)
    set_file = _build_set(
        tmp_path, "raw-16-bit", [[expected.astype("<u2").tobytes()]], width=4, height=2
    )

    img = read_set_frame(set_file, entry_no=1, frame_idx=0)

    np.testing.assert_array_equal(img, expected.astype(np.float32))


def test_raw8_widens_without_rescaling(tmp_path):
    """'raw-8-bit' widens to 16-bit but keeps raw counts: 200 stays 200, not 51200.

    Matches _decode_mono12p, which returns 0-4095 despite its DaVis label also
    reading "to 16 bit". Intensity thresholds must mean the same thing whichever
    encoding a recording used.
    """
    expected = np.array([[0, 1, 200, 255], [17, 42, 128, 254]], dtype=np.uint8)
    set_file = _build_set(
        tmp_path, "raw-8-bit", [[expected.tobytes()]], width=4, height=2
    )

    img = read_set_frame(set_file, entry_no=1, frame_idx=0)

    np.testing.assert_array_equal(img, expected.astype(np.float32))
    assert img.max() == 255  # not 255 << 8


def test_mono12p_still_decodes(tmp_path):
    """Regression: the one encoding that already worked is unchanged.

    Hand-computed from the Mono12Packed rule (2 pixels in 3 bytes):
        b0=0x34 b1=0x12 b2=0xAB
        p0 = 0x34 | (0x2 << 8) = 0x234 = 564
        p1 = 0x1  | (0xAB << 4) = 0xAB1 = 2737
    """
    set_file = _build_set(
        tmp_path, "mono-12p", [[bytes([0x34, 0x12, 0xAB])]], width=2, height=1
    )

    img = read_set_frame(set_file, entry_no=1, frame_idx=0)

    np.testing.assert_array_equal(img, np.array([[564.0, 2737.0]], dtype=np.float32))


def test_pre_paired_pair_reads_both_streams(tmp_path):
    """The pair path works end to end for a newly supported encoding."""
    frame_a = np.full((2, 4), 100, dtype="<u2")
    frame_b = np.full((2, 4), 200, dtype="<u2")
    set_file = _build_set(
        tmp_path,
        "raw-16-bit",
        [[frame_a.tobytes()], [frame_b.tobytes()]],
        width=4,
        height=2,
    )

    pair = read_set_pair(set_file, camera_no=1, im_no=1)

    assert pair.shape == (2, 2, 4)
    assert pair.dtype == np.float32
    assert pair[0].max() == 100
    assert pair[1].max() == 200


# ---------------------------------------------------------------------------
# Encodings that must be refused, by name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decoder,label",
    [
        ("mono-10p", "Mono10p to 16 bit"),
        ("mono-12packed", "Mono12packed to 16 bit"),
        ("mono-12pmsb", "Mono12pmsb to 16 bit"),
        ("rgb-24", "RGB 24"),
    ],
)
def test_known_but_unimplemented_encoding_names_itself(tmp_path, decoder, label):
    """A real DaVis encoding we cannot decode says so, and says what we can decode.

    These share byte counts with encodings we DO support (the three 12-bit ids are
    all 1.5 bytes/pixel) and differ only in bit order, so aliasing one onto another
    would produce a scrambled image rather than an error.
    """
    set_file = _build_set(tmp_path, decoder, [[b"\x00" * 6]], width=4, height=1)

    with pytest.raises(ValueError) as exc:
        read_set_frame(set_file, entry_no=1, frame_idx=0)

    message = str(exc.value)
    assert decoder in message
    assert label in message
    for implemented in _IMPLEMENTED_DECODERS:
        assert implemented in message


def test_unknown_encoding_is_rejected(tmp_path):
    """An id DaVis never writes is refused as unknown, not decoded on a guess."""
    set_file = _build_set(tmp_path, "bogus-42", [[b"\x00" * 16]], width=4, height=2)

    with pytest.raises(ValueError, match="Unknown .set pixel encoding 'bogus-42'"):
        read_set_frame(set_file, entry_no=1, frame_idx=0)


def test_mono16_is_no_longer_accepted(tmp_path):
    """'mono-16' was our own invention, not a DaVis id. It must not decode.

    Guards the removal of the silent unpacked-uint16 fallback.
    """
    set_file = _build_set(tmp_path, "mono-16", [[b"\x00" * 16]], width=4, height=2)

    with pytest.raises(ValueError, match="Unknown .set pixel encoding 'mono-16'"):
        read_set_frame(set_file, entry_no=1, frame_idx=0)


def test_byte_count_mismatch_reports_both_numbers(tmp_path):
    """The payload size is checked against the id's bit depth before decoding.

    A 4x2 frame at 16 bits/pixel needs 16 bytes; this container declares 10.
    """
    set_file = _build_set(
        tmp_path,
        "raw-16-bit",
        [[b"\x00" * 10]],
        width=4,
        height=2,
        declared_sizes=[[10]],
    )

    with pytest.raises(ValueError) as exc:
        read_set_frame(set_file, entry_no=1, frame_idx=0)

    message = str(exc.value)
    assert "10 bytes" in message
    assert "needs 16" in message
    assert "4x2" in message


# ---------------------------------------------------------------------------
# The encoding is never guessed
# ---------------------------------------------------------------------------


def test_missing_decoder_xml_raises(tmp_path):
    """No decoder XML means unknown encoding — never an assumed uint16."""
    set_file = _build_set(
        tmp_path,
        "raw-16-bit",
        [[b"\x00" * 16]],
        width=4,
        height=2,
        write_decoder_xml=False,
    )

    with pytest.raises(FileNotFoundError, match="will not be guessed"):
        read_set_info(set_file)


def test_empty_decoder_id_raises(tmp_path):
    """A decoder XML with no usable <id> is an error, not a default."""
    set_file = _build_set(tmp_path, "raw-16-bit", [[b"\x00" * 16]], width=4, height=2)
    (tmp_path / "synthetic" / "Frame0-decoder.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<FrameDecoder>\n</FrameDecoder>\n'
    )

    with pytest.raises(ValueError, match="no <id> element"):
        read_set_info(set_file)


# ---------------------------------------------------------------------------
# The three non-recording .set shapes
# ---------------------------------------------------------------------------


def test_missing_companion_folder_explains_result_sets(tmp_path):
    """A .set with no companion folder is usually a DaVis result set."""
    set_file = tmp_path / "PIV_MPd.set"
    set_file.write_bytes(b"stub")

    with pytest.raises(FileNotFoundError) as exc:
        read_set_info(set_file)

    message = str(exc.value)
    assert "no companion data folder" in message
    assert "result set" in message


def test_im7_companion_points_at_the_im7_reader(tmp_path):
    """A companion folder of .im7 files is a recording in the other DaVis layout.

    Real example: DaVis calibration recordings store B00001.im7 inside
    Calibration/camera1/, under a camera1.set node.
    """
    set_file = tmp_path / "camera1.set"
    set_file.write_bytes(b"stub")
    companion = tmp_path / "camera1"
    companion.mkdir()
    (companion / "B00001.im7").write_bytes(b"stub")

    with pytest.raises(FileNotFoundError) as exc:
        read_set_info(set_file)

    message = str(exc.value)
    assert "lavision_im7" in message
    assert "B00001.im7" in message
    assert "and 0 more" not in message  # singular reads correctly


def test_project_node_lists_what_it_found(tmp_path):
    """A DaVis project node holds sub-nodes and XML, not image streams."""
    set_file = tmp_path / "Properties.set"
    set_file.write_bytes(b"stub")
    companion = tmp_path / "Properties"
    companion.mkdir()
    (companion / "Calibration.set").write_bytes(b"stub")
    (companion / "ProjectConfiguration.xml").write_bytes(b"stub")

    with pytest.raises(FileNotFoundError) as exc:
        read_set_info(set_file)

    message = str(exc.value)
    assert "project or calibration node" in message
    assert "ProjectConfiguration.xml" in message


# ---------------------------------------------------------------------------
# Declared stream transformers (dark-image subtraction)
# ---------------------------------------------------------------------------
#
# StreamSet.xml declares post-decode corrections DaVis applies on read. Skipping
# a declared one silently returns different pixels from DaVis for the same
# recording -- which is exactly what this reader did until 2026-08-22, and why a
# comparison against lvpyio showed a 19.6-count offset that looked like a decode
# fault and was not.


def _write_stream_set(set_dir, entries):
    """Write a StreamSet.xml declaring one Transformer per (id, prefix, frame)."""
    body = "".join(
        f'<Transformer ID="{tid}" Label="L" FilePrefix="{prefix}" '
        f'MinDaVisVersion="10.2.0">'
        f'<ContentPurpose IsAssociatedToFrames="true" StartFrame="{frame}" '
        f'EndFrame="{frame}"/></Transformer>'
        for tid, prefix, frame in entries
    )
    (set_dir / "StreamSet.xml").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?><StreamSet>{body}</StreamSet>'
    )


def _write_dark_im7(path, arr):
    """Write a minimal uncompressed uint16 .im7 the in-tree reader can read."""
    import struct as _struct

    h, w = arr.shape
    header = bytearray(256)
    # Image_Header_7, "<hhhh iiii hhh": version, pack_type, buffer_format,
    # is_sparse, size_x, size_y, size_z, size_f, scalar_n, vector_grid, extra_flags.
    _struct.pack_into("<hhhh", header, 0, 0, 0, -4, 0)  # buffer_format -4 = WORD
    _struct.pack_into("<iiii", header, 8, w, h, 1, 1)
    # Trailing int32 0 terminates the extended-header record list (IEH_END).
    path.write_bytes(bytes(header) + arr.astype("<u2").tobytes() + b"\x00\x00\x00\x00")


def _write_dark_im7_float(path, arr):
    """Write a float32 .im7, for values that do not fit uint16."""
    import struct as _struct

    h, w = arr.shape
    header = bytearray(256)
    _struct.pack_into("<hhhh", header, 0, 0, 0, -3, 0)  # buffer_format -3 = FLOAT
    _struct.pack_into("<iiii", header, 8, w, h, 1, 1)
    path.write_bytes(
        bytes(header) + arr.astype("<f4").tobytes() + b"\x00\x00\x00\x00"
    )


def test_declared_dark_is_subtracted_and_clamped(tmp_path):
    """The declared dark image is applied, and the result floors at zero."""
    raw = np.array([[100, 50, 10], [200, 30, 0]], dtype=np.uint16)
    dark = np.array([[20, 20, 20], [20, 20, 20]], dtype=np.uint16)
    set_file = _build_set(
        tmp_path, "raw-16-bit", [[raw.astype("<u2").tobytes()]], width=3, height=2
    )
    set_dir = tmp_path / "synthetic"
    _write_dark_im7(set_dir / "Transformer0-dark.im7", dark)
    _write_stream_set(set_dir, [("dark-image-subtraction", "Transformer0", 0)])

    img = read_set_frame(set_file, entry_no=1, frame_idx=0)

    # 10-20 and 0-20 clamp to 0 rather than wrapping through uint16.
    expected = np.array([[80, 30, 0], [180, 10, 0]], dtype=np.float32)
    np.testing.assert_array_equal(img, expected)


def test_no_stream_set_means_no_correction(tmp_path):
    """A recording that declares nothing is read raw. This is the common case."""
    raw = np.array([[100, 50, 10], [200, 30, 0]], dtype=np.uint16)
    set_file = _build_set(
        tmp_path, "raw-16-bit", [[raw.astype("<u2").tobytes()]], width=3, height=2
    )

    img = read_set_frame(set_file, entry_no=1, frame_idx=0)

    np.testing.assert_array_equal(img, raw.astype(np.float32))


def test_unimplemented_transformer_is_refused(tmp_path):
    """A declared correction we cannot apply must fail, not be skipped.

    Silently dropping it would return pixels that differ from DaVis with no
    indication anything was omitted.
    """
    raw = np.zeros((2, 3), dtype=np.uint16)
    set_file = _build_set(
        tmp_path, "raw-16-bit", [[raw.astype("<u2").tobytes()]], width=3, height=2
    )
    _write_stream_set(
        tmp_path / "synthetic", [("flat-field-correction", "Transformer0", 0)]
    )

    with pytest.raises(ValueError) as exc:
        read_set_frame(set_file, entry_no=1, frame_idx=0)

    assert "flat-field-correction" in str(exc.value)
    assert "dark-image-subtraction" in str(exc.value)


def test_declared_dark_file_missing_raises(tmp_path):
    """A declared dark image that is not on disk is an incomplete copy."""
    raw = np.zeros((2, 3), dtype=np.uint16)
    set_file = _build_set(
        tmp_path, "raw-16-bit", [[raw.astype("<u2").tobytes()]], width=3, height=2
    )
    _write_stream_set(
        tmp_path / "synthetic", [("dark-image-subtraction", "Transformer0", 0)]
    )

    with pytest.raises(FileNotFoundError, match="Transformer0-dark.im7"):
        read_set_frame(set_file, entry_no=1, frame_idx=0)


def test_out_of_range_dark_is_refused(tmp_path):
    """A dark whose counts exceed uint16 must fail, not wrap.

    ``np.where(img > dark, img - dark, 0)`` works in uint16, so a dark value of
    70000 narrowed to 4464 would subtract almost nothing and return plausible
    wrong pixels — the failure mode this module exists to refuse.
    """
    raw = np.zeros((2, 3), dtype=np.uint16)
    set_file = _build_set(
        tmp_path, "raw-16-bit", [[raw.astype("<u2").tobytes()]], width=3, height=2
    )
    set_dir = tmp_path / "synthetic"
    big = np.full((2, 3), 70000, dtype=np.int32)
    _write_dark_im7_float(set_dir / "Transformer0-dark.im7", big)
    _write_stream_set(set_dir, [("dark-image-subtraction", "Transformer0", 0)])

    with pytest.raises(ValueError, match=r"not integer counts in \[0, 65535\]"):
        read_set_frame(set_file, entry_no=1, frame_idx=0)


def test_dark_shape_mismatch_raises(tmp_path):
    """A dark image from a different camera must not be applied."""
    raw = np.zeros((2, 3), dtype=np.uint16)
    set_file = _build_set(
        tmp_path, "raw-16-bit", [[raw.astype("<u2").tobytes()]], width=3, height=2
    )
    set_dir = tmp_path / "synthetic"
    _write_dark_im7(set_dir / "Transformer0-dark.im7", np.zeros((4, 5), np.uint16))
    _write_stream_set(set_dir, [("dark-image-subtraction", "Transformer0", 0)])

    with pytest.raises(ValueError, match="does not belong to this recording"):
        read_set_frame(set_file, entry_no=1, frame_idx=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
