"""Decoded-frame cache in ``pivtools_core.image_handling.calibration_loader``.

Every test here counts ACTUAL decodes by wrapping ``read_single_frame``, the
lowest seam under the cache, and drives the real reader against real files on
disk. That is the deliberate difference from
``test_calibration_detect_views_routes.py``, which monkeypatches ``_load_one``
one layer above the cache: patching there would make every assertion here
vacuous.

The cases that matter most are the staleness ones. A frame cache that serves an
old image after the user re-exports their calibration set is worse than no cache
at all, because the wrong board silently calibrates.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

import pivtools_core.calibration_settings as cs
import pivtools_core.image_handling.calibration_loader as CL


@pytest.fixture(autouse=True)
def _clean_cache():
    """No cross-test hits: a leaked entry could mask a reader bug."""
    CL.clear_calibration_frame_cache()
    yield
    CL.clear_calibration_frame_cache()


@pytest.fixture
def reads(monkeypatch):
    """Counter of real decodes, wrapping the reader under the cache."""
    counter = {"n": 0}
    real = CL.read_single_frame

    def counting(**kwargs):
        counter["n"] += 1
        return real(**kwargs)

    monkeypatch.setattr(CL, "read_single_frame", counting)
    return counter


def _write_frame(path: Path, value: int, size: int = 32) -> None:
    """A 16-bit TIFF of a constant value, so content is trivially identifiable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((size, size), value, dtype=np.uint16))


def _bump_mtime(path: Path) -> None:
    """Move mtime forward past the Windows ~15 ms tick, deterministically."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


def _kwargs(camera_path: Path, **over):
    base = dict(
        camera_path=camera_path,
        camera=1,
        frame_idx=1,
        image_format="calib%05d.tif",
        image_type="standard",
        num_cameras=1,
    )
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# The win
# ---------------------------------------------------------------------------


def test_repeat_read_decodes_once(tmp_path, reads):
    _write_frame(tmp_path / "calib00001.tif", 100)
    first = CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path))
    second = CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path))
    assert reads["n"] == 1
    assert second is first


def test_display_then_detection_share_one_decode(tmp_path, reads):
    """The display path reads native and the detection path reads uint8, but
    they read the same file. Whichever arrives first pays the decode."""
    _write_frame(tmp_path / "calib00001.tif", 100)
    native = CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path))
    as_u8 = CL.read_calibration_frame_at(normalize_uint8=True, **_kwargs(tmp_path))
    assert reads["n"] == 1
    assert as_u8.dtype == np.uint8
    assert native.dtype != np.uint8


def test_detection_then_display_share_one_decode(tmp_path, reads):
    """The reverse order must also hold: a uint8 miss caches the NATIVE form on
    its way through, so the display path still gets the real bit depth. Serving
    it the uint8 form would collapse the contrast window to a dead [0, 100]."""
    _write_frame(tmp_path / "calib00001.tif", 100)
    as_u8 = CL.read_calibration_frame_at(normalize_uint8=True, **_kwargs(tmp_path))
    native = CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path))
    assert reads["n"] == 1
    assert as_u8.dtype == np.uint8
    assert native.dtype != np.uint8


# ---------------------------------------------------------------------------
# Staleness — the cases that would corrupt a calibration
# ---------------------------------------------------------------------------


def test_overwritten_image_is_re_read(tmp_path, reads):
    """The headline failure: re-export a calibration image, and the viewer must
    show the new board, not the cached old one."""
    path = tmp_path / "calib00001.tif"
    _write_frame(path, 100)
    CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path))

    _write_frame(path, 7)
    _bump_mtime(path)
    after = CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path))

    assert reads["n"] == 2
    assert after.max() == 7


def test_overwrite_invalidates_the_derived_uint8_form_too(tmp_path, reads):
    """A stamp mismatch must kill every form derived from the file, not only the
    one whose key was probed.

    The flat images ``_write_frame`` produces are useless here: the TIFF reader
    returns float32, and ``_normalize_to_uint8`` min-max normalises float input,
    so ANY constant image collapses to all zeros and the assertion would hold
    with invalidation completely broken. Identify the content by WHERE the
    brightest row is instead, which survives normalisation.
    """
    path = tmp_path / "calib00001.tif"
    size = 32

    def write_with_peak_at(row: int) -> None:
        arr = np.tile(np.arange(size, dtype=np.uint16), (size, 1))
        arr[row, :] = 65535
        cv2.imwrite(str(path), arr)

    write_with_peak_at(0)
    before = CL.read_calibration_frame_at(normalize_uint8=True, **_kwargs(tmp_path))
    assert int(before.argmax()) // size == 0

    write_with_peak_at(size - 1)
    _bump_mtime(path)
    after = CL.read_calibration_frame_at(normalize_uint8=True, **_kwargs(tmp_path))

    assert reads["n"] == 2
    assert int(after.argmax()) // size == size - 1


def test_source_path_edit_at_same_index_is_not_served_from_cache(tmp_path, reads):
    """Two directories, each with its own calib00001.tif. Editing the source path
    in place must not serve the previous directory's frames -- the same lesson
    ``_joint_detect``'s memory cache already carries."""
    a, b = tmp_path / "a", tmp_path / "b"
    _write_frame(a / "calib00001.tif", 11)
    _write_frame(b / "calib00001.tif", 22)

    first = CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(a))
    second = CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(b))

    assert reads["n"] == 2
    assert first.max() == 11
    assert second.max() == 22


def test_image_format_change_is_not_served_from_cache(tmp_path, reads):
    """The format resolves into the file path, so it needs no separate key term
    -- but only if the path really is in the key. Prove it."""
    _write_frame(tmp_path / "calib00001.tif", 11)
    _write_frame(tmp_path / "alt00001.tif", 22)

    first = CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path))
    second = CL.read_calibration_frame_at(
        normalize_uint8=False, **_kwargs(tmp_path, image_format="alt%05d.tif")
    )

    assert reads["n"] == 2
    assert first.max() == 11
    assert second.max() == 22


def test_num_cameras_is_part_of_the_key(tmp_path, monkeypatch):
    """A multi-camera container holds every camera behind ONE file path, and
    num_cameras alone sets the stride. Leaving it out of the key would serve
    camera 1's pixels for a camera 2 request with no error at all -- silently
    wrong data, which is the worst failure this cache could have."""
    path = tmp_path / "calib00001.tif"
    _write_frame(path, 100)

    seen = []

    def fake_read(**kwargs):
        seen.append(kwargs["frames_per_camera"])
        return np.full((4, 4), kwargs["frames_per_camera"], dtype=np.uint16)

    monkeypatch.setattr(CL, "read_single_frame", fake_read)

    # Stand in for a container whose stride follows the rig camera count.
    monkeypatch.setattr(
        CL, "_read_native_frame", _stride_reader(fake_read), raising=True
    )

    one = CL.read_calibration_frame_at(
        normalize_uint8=False, **_kwargs(tmp_path, num_cameras=1)
    )
    two = CL.read_calibration_frame_at(
        normalize_uint8=False, **_kwargs(tmp_path, num_cameras=2)
    )

    assert one.max() == 1 and two.max() == 2, (
        "num_cameras missing from the cache key: the second read was served the "
        "first camera-count's pixels"
    )


def _stride_reader(fake_read):
    """A reader whose output depends on num_cameras, standing in for the
    container stride logic without needing a real .set/.im7 fixture."""

    def _inner(*, file_path, camera, frame_idx, image_type, num_cameras,
               use_camera_subfolders):
        return fake_read(
            file_path=file_path,
            camera=camera,
            frame_idx=frame_idx,
            image_type=image_type,
            frames_per_camera=num_cameras,
        )

    return _inner


def test_missing_file_raises_and_does_not_serve_the_cache(tmp_path, reads):
    """Deleting the file must surface the reader's own error naming it, never a
    cached array. Serving the cache here is a textbook silent fallback."""
    path = tmp_path / "calib00001.tif"
    _write_frame(path, 100)
    CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path))
    path.unlink()

    with pytest.raises((FileNotFoundError, ValueError)):
        CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path))


# ---------------------------------------------------------------------------
# The cached array is shared, so it must be read-only
# ---------------------------------------------------------------------------


def test_cached_array_is_read_only(tmp_path, reads):
    """Every later request for this frame gets the SAME object, so an in-place
    write would corrupt all of them silently and far from the offending line."""
    _write_frame(tmp_path / "calib00001.tif", 100)
    arr = CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path))
    with pytest.raises(ValueError):
        arr[0, 0] = 0


def test_display_path_accepts_a_read_only_frame(tmp_path, reads):
    """The gate on making it read-only. `numpy_to_base64` and
    `get_display_contrast_stats` both begin with
    `arr.astype(np.float32, copy=False)`, which returns the ORIGINAL array when it
    is already float32 -- which is exactly what the .im7 reader hands back. If
    either ever writes in place, this fails instead of serving wrong pixels.
    """
    from pivtools_gui.utils import get_display_contrast_stats, numpy_to_base64

    _write_frame(tmp_path / "calib00001.tif", 100, size=64)
    arr = CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path))
    assert not arr.flags.writeable

    stats = get_display_contrast_stats(arr)
    assert "vmin_pct" in stats
    assert numpy_to_base64(arr, format="jpeg")


def test_the_real_detectors_accept_a_read_only_frame(tmp_path, reads):
    """Same gate for the detection path: OpenCV is handed the cached array
    directly by ChArUco's `_to_gray_u8`, which passes a uint8 2D input straight
    through. Detection is expected to FAIL to find a board in this noise -- what
    is being asserted is that it does not raise on a non-writeable input.
    """
    from pivtools_gui.calibration.detection.charuco import (
        CharucoBoardDetector,
        CharucoParams,
    )
    from pivtools_gui.calibration.detection.dotboard import (
        DotboardDetector,
        DotboardParams,
    )

    _write_frame(tmp_path / "calib00001.tif", 100, size=256)
    arr = CL.read_calibration_frame_at(normalize_uint8=True, **_kwargs(tmp_path))
    assert not arr.flags.writeable

    DotboardDetector(DotboardParams(dot_spacing_mm=5.0)).detect(arr)
    CharucoBoardDetector(CharucoParams()).detect(arr)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_lru_evicts_oldest_and_stays_inside_the_budget(tmp_path, reads, monkeypatch):
    for k in range(1, 6):
        _write_frame(tmp_path / f"calib{k:05d}.tif", k, size=64)

    one_frame = CL.read_calibration_frame_at(
        normalize_uint8=False, **_kwargs(tmp_path, frame_idx=1)
    ).nbytes
    CL.clear_calibration_frame_cache()
    reads["n"] = 0
    monkeypatch.setattr(CL, "_FRAME_CACHE_BYTES_BUDGET", one_frame * 3)

    for k in range(1, 6):
        CL.read_calibration_frame_at(
            normalize_uint8=False, **_kwargs(tmp_path, frame_idx=k)
        )
        assert CL._frame_cache_bytes <= one_frame * 3
    # Frame 1 was evicted by 4 and 5, so it costs a sixth decode.
    CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path, frame_idx=1))
    assert reads["n"] == 6


def test_entry_cap_bounds_a_long_run_of_small_frames(tmp_path, reads, monkeypatch):
    """The byte budget alone would let thousands of tiny frames accumulate."""
    monkeypatch.setattr(CL, "_FRAME_CACHE_MAX_ENTRIES", 3)
    for k in range(1, 7):
        _write_frame(tmp_path / f"calib{k:05d}.tif", k, size=8)
        CL.read_calibration_frame_at(
            normalize_uint8=False, **_kwargs(tmp_path, frame_idx=k)
        )
    assert len(CL._frame_cache) <= 3


def test_frame_larger_than_the_budget_is_not_cached(tmp_path, reads, monkeypatch):
    """Caching it would flush everything on every request and drive the hit rate
    to zero while the eviction loop churns."""
    _write_frame(tmp_path / "calib00001.tif", 100, size=64)
    monkeypatch.setattr(CL, "_FRAME_CACHE_BYTES_BUDGET", 16)

    CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path))
    CL.read_calibration_frame_at(normalize_uint8=False, **_kwargs(tmp_path))

    assert reads["n"] == 2
    assert len(CL._frame_cache) == 0


# ---------------------------------------------------------------------------
# use_cache=False (the CLI's one-pass sweep)
# ---------------------------------------------------------------------------


def test_use_cache_false_neither_reads_nor_populates(tmp_path, reads):
    _write_frame(tmp_path / "calib00001.tif", 100)
    CL.read_calibration_frame_at(
        normalize_uint8=False, use_cache=False, **_kwargs(tmp_path)
    )
    CL.read_calibration_frame_at(
        normalize_uint8=False, use_cache=False, **_kwargs(tmp_path)
    )
    assert reads["n"] == 2
    assert len(CL._frame_cache) == 0


# ---------------------------------------------------------------------------
# Budget parsing
# ---------------------------------------------------------------------------


def test_budget_env_override_is_honoured(monkeypatch):
    monkeypatch.setenv("PIV_CALIB_FRAME_CACHE_MB", "7")
    assert CL._frame_cache_budget_bytes() == 7 * 1024 * 1024


def test_budget_env_rejects_nonsense_loudly(monkeypatch):
    """A typo must not silently fall back to the default budget."""
    monkeypatch.setenv("PIV_CALIB_FRAME_CACHE_MB", "lots")
    with pytest.raises(ValueError, match="PIV_CALIB_FRAME_CACHE_MB"):
        CL._frame_cache_budget_bytes()


def test_budget_env_rejects_negative(monkeypatch):
    monkeypatch.setenv("PIV_CALIB_FRAME_CACHE_MB", "-1")
    with pytest.raises(ValueError, match="PIV_CALIB_FRAME_CACHE_MB"):
        CL._frame_cache_budget_bytes()


# ---------------------------------------------------------------------------
# Sidecar resolution — the trap this cache's placement exists to avoid
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal stand-in: the loader only asks for the source and camera count."""

    def __init__(self, source: Path, camera_count: int = 1):
        self._source = source
        self.camera_count = camera_count

    def get_calibration_source(self, idx: int = 0) -> Path:
        return self._source


def test_sidecar_format_change_is_not_served_from_cache(tmp_path, reads):
    """The whole reason the cache sits in read_calibration_frame_at rather than
    in the GUI route layer. image_format lives in the settings sidecar and never
    appears in the request URL, so a cache keyed further up would serve the old
    file's pixels after this edit."""
    _write_frame(tmp_path / "calib00001.tif", 11)
    _write_frame(tmp_path / "alt00001.tif", 22)
    cfg = _FakeConfig(tmp_path)

    cs.save_settings(
        tmp_path, {"image": {"image_format": "calib%05d.tif", "image_type": "standard"}}
    )
    first = CL.read_calibration_image(1, 1, cfg, 0, normalize_uint8=False)

    cs.save_settings(
        tmp_path, {"image": {"image_format": "alt%05d.tif", "image_type": "standard"}}
    )
    second = CL.read_calibration_image(1, 1, cfg, 0, normalize_uint8=False)

    assert reads["n"] == 2
    assert first.max() == 11
    assert second.max() == 22
