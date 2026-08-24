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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
