"""_read_batch pre-allocates one buffer; the Dask graph it feeds is unchanged.

Pins the Batch-4 fix from the image-reading audit (2026-08-24): the batch was
collected in a list and np.stack-ed, holding every pair twice at the peak. It is
now one np.empty buffer filled pair by pair. The output must be identical to
stacking the individual read_pair results, and load_images must still build one
independent delayed task per batch with the same chunks and a truthful float32
dtype (the removed ``images.dtype`` knob let a config tell Dask uint16 while
float32 arrived, with no error).

Usage:
    pytest unit-tests/test_read_batch.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_frames_per_camera_and_shape import _build_raw16_set, _make_config  # noqa: E402

from pivtools_core.image_handling.load_images import (  # noqa: E402
    _read_batch,
    load_images,
    read_pair,
)
from pivtools_core.image_handling.readers.set_reader import (  # noqa: E402
    clear_set_info_cache,
)


@pytest.fixture()
def _clean_set_cache():
    clear_set_info_cache()
    yield
    clear_set_info_cache()


def _time_resolved_set(tmp_path):
    """One-stream .set with 5 entries of distinct constant frames, and a config
    pairing consecutive entries (frame_stride 1 -> 4 pairs)."""
    import struct

    set_file = _build_raw16_set(tmp_path, [(4, 6)], name="batch")
    payloads = [np.full((4, 6), 10 * (k + 1), "<u2").tobytes() for k in range(5)]
    index = bytearray(1024)
    struct.pack_into("<i", index, 12, 6)
    struct.pack_into("<i", index, 16, 4)
    for k, payload in enumerate(payloads):
        index += struct.pack("<iqq", 0, k * len(payload), len(payload))
    (tmp_path / "batch" / "Frame0-0.ims").write_bytes(bytes(index))
    (tmp_path / "batch" / "Frame0-1.ims").write_bytes(b"".join(payloads))

    cfg = _make_config(tmp_path, set_file, camera_count=1, camera_numbers=[1])
    cfg.data["images"]["num_images"] = 5
    cfg.data["images"]["pair_stride"] = 1
    return set_file, cfg


def test_read_batch_matches_stacked_read_pair(tmp_path, _clean_set_cache):
    set_file, cfg = _time_resolved_set(tmp_path)

    batch = _read_batch(2, 3, set_file, 1, cfg)
    expected = np.stack(
        [read_pair(idx, set_file, 1, cfg) for idx in (2, 3, 4)], axis=0
    )

    assert batch.shape == (3, 2, 4, 6) and batch.dtype == np.float32
    np.testing.assert_array_equal(batch, expected)
    # Entry values are 10*k, so pair 2 is entries (2, 3) -> 20, 30.
    assert batch[0, 0, 0, 0] == 20 and batch[0, 1, 0, 0] == 30


def test_load_images_graph_is_lazy_batched_float32(tmp_path, _clean_set_cache):
    set_file, cfg = _time_resolved_set(tmp_path)

    arr = load_images(1, cfg, batch_size=3)

    assert arr.dtype == np.float32
    assert arr.shape == (4, 2, 4, 6)
    assert arr.chunks == ((3, 1), (2,), (4,), (6,))  # one task per batch
    np.testing.assert_array_equal(
        arr.compute(),
        np.stack([read_pair(i, set_file, 1, cfg) for i in range(1, 5)]),
    )


# ---------------------------------------------------------------------------
# read_pair(out=): every branch writes into the caller's buffer and returns it
# ---------------------------------------------------------------------------
#
# The contract (readers/out_buffer.py): (2, H, W) float32 C-contiguous, written
# in place, returned, undefined after an exception. _read_batch relies on it to
# decode straight into the batch with no per-pair copy (47 ms of a 298 ms .set
# pair on alk235). .cine is the one branch not covered here: cinereader is not
# installed on the development machine and no synthetic builder exists.

from test_calibration_cli_loader import _write_pack0_im7  # noqa: E402
from test_frames_per_camera_and_shape import _make_im7_config  # noqa: E402

from pivtools_core.image_handling.readers import set_reader  # noqa: E402
from pivtools_core.image_handling.readers.im7_reader import read_im7_camera  # noqa: E402


def _assert_out_contract(cfg, camera_path, camera):
    ref = read_pair(1, camera_path, camera, cfg)
    buf = np.empty(ref.shape, np.float32)
    assert read_pair(1, camera_path, camera, cfg, out=buf) is buf
    np.testing.assert_array_equal(buf, ref)
    with pytest.raises(ValueError, match="out has shape"):
        read_pair(1, camera_path, camera, cfg, out=np.empty((2, 3, 3), np.float32))
    with pytest.raises(ValueError, match="out has dtype"):
        read_pair(1, camera_path, camera, cfg, out=np.empty(ref.shape))


def test_out_set_time_resolved(tmp_path, _clean_set_cache):
    set_file, cfg = _time_resolved_set(tmp_path)
    _assert_out_contract(cfg, set_file, 1)


def test_out_set_pre_paired(tmp_path, _clean_set_cache):
    set_file = _build_raw16_set(tmp_path, [(4, 6), (4, 6)], name="paired")
    cfg = _make_config(tmp_path, set_file, camera_count=1, camera_numbers=[1], frame_stride=0)
    _assert_out_contract(cfg, set_file, 1)


def test_out_im7_pre_paired_multi_camera(tmp_path):
    frames = np.arange(8 * 7 * 9, dtype=np.uint16).reshape(8, 7, 9)
    _write_pack0_im7(tmp_path / "B00001.im7", frames)
    cfg = _make_im7_config(tmp_path, camera_count=4, camera_numbers=[3])
    cfg.data["images"]["frame_stride"] = 0
    _assert_out_contract(cfg, tmp_path, 3)
    buf = np.empty((2, 7, 9), np.float32)
    read_pair(1, tmp_path, 3, cfg, out=buf)
    np.testing.assert_array_equal(buf, frames[4:6].astype(np.float32))


def test_out_im7_time_resolved_multi_camera(tmp_path):
    rng = np.random.default_rng(5)
    for k in (1, 2):
        _write_pack0_im7(
            tmp_path / f"B{k:05d}.im7", rng.integers(0, 4096, (4, 7, 9), np.uint16)
        )
    cfg = _make_im7_config(tmp_path, camera_count=4, camera_numbers=[2])
    _assert_out_contract(cfg, tmp_path, 2)


def test_out_im7_single_camera_subfolder_pre_paired(tmp_path):
    (tmp_path / "Cam2").mkdir()
    _write_pack0_im7(tmp_path / "Cam2" / "B00001.im7", np.ones((2, 7, 9), np.uint16))
    cfg = _make_im7_config(tmp_path, camera_count=2, camera_numbers=[2], subfolders=True)
    cfg.data["images"]["frame_stride"] = 0
    _assert_out_contract(cfg, tmp_path / "Cam2", 2)


def test_out_standard_tif(tmp_path):
    import tifffile

    rng = np.random.default_rng(6)
    for suffix in ("A", "B"):
        tifffile.imwrite(
            tmp_path / f"B00001_{suffix}.tif", rng.integers(0, 4096, (7, 9), np.uint16)
        )
    cfg = _make_config(tmp_path, tmp_path / "unused.set", camera_count=1, camera_numbers=[1])
    cfg.data["images"].update(
        {"image_type": "standard", "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
         "frame_stride": 0}
    )
    _assert_out_contract(cfg, tmp_path, 1)


def test_reader_that_ignores_out_is_caught(tmp_path, _clean_set_cache, monkeypatch):
    """A reader that accepts out= and allocates anyway would leave the batch slot
    as uninitialised memory with no error. read_pair checks identity."""
    set_file = _build_raw16_set(tmp_path, [(4, 6), (4, 6)], name="ignores")
    cfg = _make_config(tmp_path, set_file, camera_count=1, camera_numbers=[1], frame_stride=0)
    real = set_reader.read_set_pair
    monkeypatch.setattr(
        set_reader, "read_set_pair", lambda *a, out=None, **k: real(*a, **k)
    )
    with pytest.raises(RuntimeError, match="did not write into"):
        read_pair(1, set_file, 1, cfg, out=np.empty((2, 4, 6), np.float32))


def test_read_batch_refuses_a_camera_whose_shape_differs(tmp_path, _clean_set_cache):
    """config.image_shape is detected once per process from the first configured
    camera. A second camera with another shape must fail by name here, not
    produce a batch that disagrees with what Dask was told."""
    set_file = _build_raw16_set(tmp_path, [(4, 6), (8, 10)], name="mixed")
    cfg = _make_config(tmp_path, set_file, camera_count=2, camera_numbers=[1, 2])
    assert cfg.image_shape == (4, 6)
    with pytest.raises(ValueError, match=r"detected from camera 1"):
        _read_batch(1, 1, set_file, 2, cfg)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
