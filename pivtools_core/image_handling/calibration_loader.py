"""Calibration image loading utilities.

This module provides functions for loading and validating calibration images
using the centralized image handling system. It supports all image formats
including standard formats (TIFF, PNG, JPEG) and container formats
(.set, .im7, .cine).

Key Functions:
- read_calibration_image: Read a single calibration image
- read_calibration_frame_at: Read one frame from an already-resolved camera path
- validate_calibration_images: Validate calibration images exist and are readable
- get_calibration_frame_count: Auto-detect number of calibration images
- clear_calibration_frame_cache: Drop the process-wide decoded-frame cache

Reads are cached per process and revalidated by mtime and size on every hit —
see the decoded-frame cache block below for the key, the budget and the one
invalidation case it cannot see.
"""

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..calibration_settings import default_settings, try_load_settings
from ..config import Config
from .load_images import read_single_frame
from .path_utils import (
    build_calibration_camera_path,
    format_to_glob,
    infer_image_type,
    resolve_file_path,
    validate_images_generic,
)

# ---------------------------------------------------------------------------
# Decoded-frame cache
#
# The GUI re-reads the same calibration frames constantly: the viewer paints a
# frame, its prefetch pulls the neighbours, a detection route reads that same
# frame again as uint8, and every tab revisit starts over. Each miss is a full
# decode of a sensor-sized image off whatever the source lives on -- in
# practice a USB HDD or a OneDrive path.
#
# The cache lives HERE, in read_calibration_frame_at, rather than in the GUI
# route layer, because this is the first point at which every field that can
# change the pixels has been resolved into an argument. image_format,
# image_type, use_camera_subfolders, camera_subfolders, zero_based_indexing
# and camera_count all come from the source's settings sidecar and never
# appear in the request URL, so a key built further up would be blind to them
# and would serve stale pixels after a sidecar edit. Here the arguments ARE
# the key, which makes sidecar invalidation structural rather than remembered.
# The frontend cache has to solve the same problem by hand -- see gotcha #6.
#
# Same shape as readers/set_reader.py's _set_info_cache: per-process,
# OrderedDict insertion order as LRU, one lock, slow I/O outside the lock.
#
# Budget arithmetic. An entry costs H * W * itemsize, so how many fit depends
# entirely on the sensor AND on the dtype the reader hands back:
#
#     1024x1024  uint16   ( 1.0 MP)    2.0 MiB   -> the 64-entry cap binds first
#     2048x2048  uint16   ( 4.2 MP)    8.0 MiB   -> the 64-entry cap binds first
#     2560x2160  uint16   ( 5.5 MP)   10.6 MiB   ->  48 frames
#     4872x3248  uint16   (15.8 MP)   30.2 MiB   ->  16 frames
#     4872x3248  float32  (15.8 MP)   60.4 MiB   ->   8 frames
#
# That last row is not hypothetical: the .im7 reader promotes to float32, so a
# 16 MP LaVision frame really is 60 MiB and 512 MiB really is 8 of them. The
# entry cap is what binds on ordinary sensors, where it stops a long run of
# small frames growing the dict without bound.
#
# 512 MiB is sized for what this cache is FOR -- the frames being worked with
# right now:
#   * the validate preview (uint8) and the viewer's first frame (native)
#     sharing ONE decode,
#   * a detection route re-reading a frame the viewer already loaded,
#   * a browser-cache miss (a reopened browser, or a frame past its 40 entries).
# Eight frames covers all three even on the largest sensor here.
#
# It is deliberately NOT sized to hold a whole multi-camera warmup or a 40-view
# Detect Dots sweep. Both are one-pass: the browser keeps the encoded results in
# its own LRU and never re-asks, so being evicted during such a sweep costs
# nothing on screen. Sizing for them would pin memory in the long-lived Flask
# process against a running PIV job, which Config.auto_compute_params hands 90%
# of total RAM to.
# ---------------------------------------------------------------------------

_FRAME_CACHE_MAX_ENTRIES = 64


def _frame_cache_budget_bytes() -> int:
    """Byte budget for the decoded-frame cache, from ``PIV_CALIB_FRAME_CACHE_MB``.

    An environment variable rather than a config key on purpose: a ``config.yaml``
    knob would have to agree across the CLI template, the ``Config`` property and
    the frontend fallback (the three-layer-defaults gotcha), and this has no GUI
    control and no per-source meaning. It is a machine-tuning escape hatch.
    """
    raw = os.environ.get("PIV_CALIB_FRAME_CACHE_MB")
    if raw is None:
        return 512 * 1024 * 1024
    try:
        mb = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"PIV_CALIB_FRAME_CACHE_MB must be an integer number of MiB, got {raw!r}"
        ) from exc
    if mb < 0:
        raise ValueError(f"PIV_CALIB_FRAME_CACHE_MB must not be negative, got {mb}")
    return mb * 1024 * 1024


_FRAME_CACHE_BYTES_BUDGET = _frame_cache_budget_bytes()

# key -> (stamp, array). Insertion order is LRU order; a hit re-inserts.
_frame_cache: "OrderedDict[Tuple, Tuple[Tuple, np.ndarray]]" = OrderedDict()
_frame_cache_bytes = 0
# The Flask server is threaded, so the dict needs a lock: each OrderedDict
# operation is individually atomic under the GIL, but the sequences here are
# not, and a hit's move_to_end can race an eviction and raise KeyError out of
# what should be a read. Decoding runs OUTSIDE the lock -- two threads racing
# to decode the same frame is wasteful, never wrong.
_frame_cache_lock = threading.Lock()


def _frame_stamp(file_path: Path, image_type: str) -> Tuple:
    """What must still hold for a cached frame to be valid.

    ``st_mtime_ns`` AND ``st_size``, not mtime alone: Windows last-write time
    ticks at roughly 15 ms, so a rewrite inside one tick would otherwise be
    served stale. Size catches nearly all of those.

    For ``.set`` the companion stream directory is stamped too. The ``.set``
    itself is a few hundred bytes of project XML and does not change when the
    streams beside it do, so its mtime alone would miss a stream swap entirely
    (the same limitation ``set_reader._set_info_cache_key`` documents).

    KNOWN LIMITATION, accepted rather than hidden: an in-place rewrite that
    preserves both mtime and size survives this check. That is not reachable
    for a multi-megabyte frame, whose write takes longer than the mtime tick.
    Call :func:`clear_calibration_frame_cache` if a file is ever edited in
    place during a session.

    Raises:
        OSError: If the file is missing. Callers let the reader raise instead,
            so the error names the file rather than the cache.
    """
    st = file_path.stat()
    if image_type == "lavision_set":
        return (st.st_mtime_ns, st.st_size, file_path.with_suffix("").stat().st_mtime_ns)
    return (st.st_mtime_ns, st.st_size)


def _frame_cache_key(
    file_path: Path,
    image_type: str,
    camera: int,
    frame_idx: int,
    num_cameras: int,
    use_camera_subfolders: bool,
    normalize_uint8: bool,
) -> Tuple:
    """Everything that decides which pixels a read returns.

    ``file_path`` absorbs ``image_format``, ``zero_based_indexing``, the camera
    subfolder and the source directory itself, so a source edited in place at
    the same ``source_path_idx`` cannot be served the previous directory's
    frames.

    ``camera``, ``frame_idx`` and ``num_cameras`` are all separately required
    for the container formats: a ``.set`` or multi-camera ``.im7`` holds every
    camera and every frame behind ONE ``file_path``, and ``num_cameras`` alone
    sets the stride that selects this camera's slice. A wrong stride returns a
    different camera's pixels without failing, so leaving ``num_cameras`` out
    of the key would make a ``camera_count`` change silently serve the wrong
    camera.

    ``use_camera_subfolders`` is a branch selector inside the reader, not only
    a path input: with one camera and no explicit subfolder both branches
    resolve the same ``file_path`` but read different pixels out of it.
    """
    return (
        os.path.normcase(os.path.abspath(str(file_path))),
        image_type,
        int(camera),
        int(frame_idx),
        int(num_cameras),
        bool(use_camera_subfolders),
        bool(normalize_uint8),
    )


def _frame_cache_get(key: Tuple, stamp: Tuple) -> Optional[np.ndarray]:
    """Cached array for ``key`` if its stamp still holds, else ``None``.

    A stamp mismatch drops the entry: the file changed, so every form derived
    from it is dead too.
    """
    global _frame_cache_bytes
    with _frame_cache_lock:
        entry = _frame_cache.get(key)
        if entry is None:
            return None
        cached_stamp, arr = entry
        if cached_stamp != stamp:
            del _frame_cache[key]
            _frame_cache_bytes -= arr.nbytes
            return None
        _frame_cache.move_to_end(key)
        return arr


def _frame_cache_put(key: Tuple, stamp: Tuple, arr: np.ndarray) -> None:
    """Insert ``arr``, then evict from the front until inside both bounds.

    The array is marked read-only first, and callers are handed this master
    rather than a copy. A cached frame is shared by every later request for it,
    so an in-place write would corrupt all of them silently and at a distance;
    read-only turns that into a ``ValueError`` on the offending line instead.

    This is not hypothetical. ``numpy_to_base64`` and ``get_display_contrast_stats``
    both start with ``arr.astype(np.float32, copy=False)``, which returns the
    ORIGINAL array when it is already float32 — and the ``.im7`` reader returns
    float32. They are safe today only because the ``np.maximum`` that follows
    allocates; adding an ``out=`` there would poison the cache. The flag makes
    that a loud failure rather than wrong pixels.
    """
    global _frame_cache_bytes
    arr.setflags(write=False)
    budget = _FRAME_CACHE_BYTES_BUDGET
    if arr.nbytes > budget:
        # A single frame larger than the whole budget would flush the cache on
        # every request and drive the hit rate to zero. Serve it uncached.
        return
    with _frame_cache_lock:
        previous = _frame_cache.pop(key, None)
        if previous is not None:
            _frame_cache_bytes -= previous[1].nbytes
        _frame_cache[key] = (stamp, arr)
        _frame_cache_bytes += arr.nbytes
        # Never evict the entry just inserted: the caller is about to use it.
        while len(_frame_cache) > 1 and (
            _frame_cache_bytes > budget
            or len(_frame_cache) > _FRAME_CACHE_MAX_ENTRIES
        ):
            _, (_, evicted) = _frame_cache.popitem(last=False)
            _frame_cache_bytes -= evicted.nbytes


def clear_calibration_frame_cache() -> None:
    """Drop every cached frame. Mirrors ``set_reader.clear_set_info_cache()``.

    Needed by tests, and by the in-place-rewrite case ``_frame_stamp`` cannot
    see. Routine invalidation does not go through here: a sidecar edit that
    matters changes the key, which makes the old entries unreachable rather
    than stale, and they age out through the LRU.
    """
    global _frame_cache_bytes
    with _frame_cache_lock:
        _frame_cache.clear()
        _frame_cache_bytes = 0


def _resolve_image_settings(
    config: Config,
    source_path_idx: int,
    image_format: Optional[str] = None,
    image_type: Optional[str] = None,
    num_images: Optional[int] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """(source, effective image block) for a calibration read.

    The image block comes from the source's settings sidecar; explicit
    ``image_format`` / ``image_type`` / ``num_images`` arguments override it
    (live form values from the GUI). A source with no sidecar yet is readable
    ONLY with an explicit ``image_format`` (the first-visit validate flow,
    before anything is saved) — the remaining knobs then take the documented
    defaults. Without an override, a missing sidecar raises.
    """
    source = config.get_calibration_source(source_path_idx)
    settings = try_load_settings(source)
    if settings is None:
        if not image_format:
            from ..calibration_settings import _missing_message, settings_path

            raise FileNotFoundError(_missing_message(source, settings_path(source)))
        image = default_settings()["image"]
    else:
        image = dict(settings["image"])
    if image_format is not None:
        image["image_format"] = image_format
        # An explicit format re-infers the type unless the type is also explicit
        # (a stale stored type must not win over the live pattern).
        if image_type is None:
            image["image_type"] = infer_image_type(image_format)
    if image_type is not None:
        image["image_type"] = image_type
    if num_images is not None:
        image["n_views"] = num_images
    return source, image


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize image array to uint8 for OpenCV detection.

    Parameters
    ----------
    img : np.ndarray
        Input image of any dtype

    Returns
    -------
    np.ndarray
        Image normalized to uint8 (0-255)
    """
    if img.dtype == np.uint8:
        return img
    elif img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return np.zeros_like(img, dtype=np.uint8)
    elif img.dtype == np.bool_:
        return img.astype(np.uint8) * 255
    return img.astype(np.uint8)


def read_calibration_image(
    idx: int,
    camera: int,
    config: Config,
    source_path_idx: int = 0,
    image_format: Optional[str] = None,
    image_type: Optional[str] = None,
    normalize_uint8: bool = True,
) -> np.ndarray:
    """Read a single calibration image.

    This function uses the unified read_single_frame() core reader,
    eliminating duplicated format handling. It handles all image formats:
    - Standard formats (.tif, .png, .jpg) with numbered patterns
    - LaVision .set containers (all cameras in one file)
    - LaVision .im7 files (one per frame)
    - Phantom .cine video files (one per camera)

    Parameters
    ----------
    idx : int
        Image index (1-based unless the sidecar's zero_based_indexing is True)
    camera : int
        Camera number (1-based)
    config : Config
        Configuration object supplying the calibration source pointer
    source_path_idx : int, optional
        Index into calibration_sources list, defaults to 0
    image_format : str, optional
        Override for the sidecar's image.image_format
    image_type : str, optional
        Override for the sidecar's image.image_type
    normalize_uint8 : bool, optional
        If True, normalize output to uint8 for OpenCV detection (default True)

    Returns
    -------
    np.ndarray
        Image data as 2D array (H, W), normalized to uint8 if normalize_uint8=True

    Raises
    ------
    FileNotFoundError
        If the image file does not exist, or the source has no settings
        sidecar (and no explicit format/type overrides were given)
    ValueError
        If the image cannot be read or calibration_sources not configured
    """
    # Image sourcing comes from the source's settings sidecar (explicit args
    # override); config supplies only the source pointer + rig camera count.
    source, image = _resolve_image_settings(
        config, source_path_idx, image_format, image_type
    )
    camera_path = build_calibration_camera_path(
        source, image, camera, config.camera_count
    )

    return read_calibration_frame_at(
        camera_path=camera_path,
        camera=camera,
        frame_idx=idx,
        image_format=image["image_format"],
        image_type=image["image_type"],
        zero_based_indexing=bool(image.get("zero_based_indexing", False)),
        use_camera_subfolders=bool(image.get("use_camera_subfolders", False)),
        normalize_uint8=normalize_uint8,
        num_cameras=config.camera_count,
    )


def read_calibration_frame_at(
    camera_path: Path,
    camera: int,
    frame_idx: int,
    image_format: str,
    image_type: str,
    *,
    num_cameras: int,
    zero_based_indexing: bool = False,
    use_camera_subfolders: bool = False,
    normalize_uint8: bool = True,
    use_cache: bool = True,
) -> np.ndarray:
    """Read one calibration frame from an ALREADY-RESOLVED camera path/container.

    The format-dispatch half of :func:`read_calibration_image`, factored out so callers
    that resolve the camera directory themselves can share it. The GUI/config path goes
    through ``read_calibration_image`` (resolves ``camera_path`` from ``config`` +
    ``source_path_idx``); the calibration CLI calls this directly with its
    ``--source``-derived directory, so both read the same formats (standard tif/png,
    LaVision ``.im7``/``.set``, Phantom ``.cine``) through one code path.

    Parameters mirror :func:`read_calibration_image` except ``camera_path`` is supplied
    directly (a directory for per-file formats, or the container file for ``.set``), and
    ``num_cameras`` (the rig camera count) is required — it locates this camera's slice
    inside a multi-camera ``.im7`` buffer or ``.set`` container.

    ``use_cache`` (default True) serves repeat reads of the same frame from the
    process-wide decoded-frame cache above, revalidated by mtime and size on every
    hit. Pass False for a one-pass sweep that will never re-read a frame — the
    calibration CLI does, because it holds the whole view list itself and caching
    would only retain dead arrays after that list is dropped.

    A cached array is shared between callers. Copy it before writing.
    """
    # Pure path arithmetic (no I/O), so building the cache key below is free.
    file_path = resolve_file_path(
        camera_path=camera_path,
        camera=camera,
        frame_idx=frame_idx,
        format_pattern=image_format,
        image_type=image_type,
        zero_based_indexing=zero_based_indexing,
    )

    def _uncached() -> np.ndarray:
        img = _read_native_frame(
            file_path=file_path,
            camera=camera,
            frame_idx=frame_idx,
            image_type=image_type,
            num_cameras=num_cameras,
            use_camera_subfolders=use_camera_subfolders,
        )
        return _normalize_to_uint8(img) if normalize_uint8 else img

    if not use_cache:
        return _uncached()

    try:
        stamp = _frame_stamp(Path(file_path), image_type)
    except OSError:
        # Missing file, or a .set with no companion folder. Fall through to the
        # reader so IT raises: its error names the file and the shape of the
        # problem, which a bare stat error here would lose. A cached array is
        # never served for a file that has gone.
        return _uncached()

    def _key(as_uint8: bool) -> Tuple:
        return _frame_cache_key(
            file_path=file_path,
            image_type=image_type,
            camera=camera,
            frame_idx=frame_idx,
            num_cameras=num_cameras,
            use_camera_subfolders=use_camera_subfolders,
            normalize_uint8=as_uint8,
        )

    wanted = _key(normalize_uint8)
    hit = _frame_cache_get(wanted, stamp)
    if hit is not None:
        return hit

    # A uint8 miss still avoids the disk when the native form is cached. The
    # display path reads native (it needs the real bit depth for the contrast
    # window) and the detection path reads uint8, but they read the SAME file:
    # whichever arrives first pays the decode and the other derives from it.
    native_key = _key(False)
    native = _frame_cache_get(native_key, stamp) if normalize_uint8 else None
    if native is None:
        native = _read_native_frame(
            file_path=file_path,
            camera=camera,
            frame_idx=frame_idx,
            image_type=image_type,
            num_cameras=num_cameras,
            use_camera_subfolders=use_camera_subfolders,
        )
        _frame_cache_put(native_key, stamp, native)
    if not normalize_uint8:
        return native

    # When the native frame is already uint8 this returns it unchanged, so both
    # keys hold the one array and its bytes are counted twice. The over-count is
    # conservative (the budget under-fills) and cheaper than refcounting.
    img = _normalize_to_uint8(native)
    _frame_cache_put(wanted, stamp, img)
    return img


def _read_native_frame(
    file_path: Path,
    camera: int,
    frame_idx: int,
    image_type: str,
    num_cameras: int,
    use_camera_subfolders: bool,
) -> np.ndarray:
    """Decode one frame at its native bit depth. The uncached inner half of
    :func:`read_calibration_frame_at`, split out so the cache has one thing to
    call and normalisation stays a pure function of the result."""
    # For IM7 with camera subfolders, each file is single-camera - don't pass
    # camera_no. One frame is wanted, so ask for one: the reader defaults would
    # decode the whole A/B pair and discard B.
    if image_type == "lavision_im7" and use_camera_subfolders:
        from .load_images import read_image

        img = read_image(str(file_path), frames=1, frames_per_camera=1)
        if img.ndim == 3:
            img = img[0]  # Extract single frame
        return img

    # Multi-camera containers (.im7 buffers, .set frame streams) interleave every
    # camera's frames, so this camera's slice is located as
    # (camera-1) * frames_per_camera. Detect that stride rather than assume it —
    # the same rule the PIV path uses. ``num_cameras`` is required (not defaulted
    # to 1) precisely because a stride of 1 on a multi-camera container returns a
    # DIFFERENT camera's pixels without failing: silently wrong data, not a
    # visible error.
    fpc = 1
    if image_type == "lavision_im7":
        from .load_images import _detect_im7_frames_per_camera

        fpc = _detect_im7_frames_per_camera(Path(file_path), num_cameras)
    elif image_type == "lavision_set":
        from .load_images import _detect_set_frames_per_camera

        fpc = _detect_set_frames_per_camera(Path(file_path), num_cameras)

    # Unified core reader (locates this camera's slice in multi-camera containers).
    return read_single_frame(
        file_path=file_path,
        camera=camera,
        frame_idx=frame_idx,
        image_type=image_type,
        frames_per_camera=fpc,
    )


def validate_calibration_images(
    camera: int,
    config: Config,
    source_path_idx: int = 0,
    image_format: Optional[str] = None,
    num_images: Optional[int] = None,
    image_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate calibration images exist and are readable.

    Uses the generic validate_images_generic() function with calibration-specific
    parameters.

    Parameters
    ----------
    camera : int
        Camera number (1-based)
    config : Config
        Configuration object supplying the calibration source pointer
    source_path_idx : int, optional
        Index into calibration_sources list, defaults to 0
    image_format : str, optional
        Override for the sidecar's image.image_format
    num_images : int, optional
        Override for the sidecar's image.n_views
    image_type : str, optional
        Override for the sidecar's image.image_type

    Returns
    -------
    dict
        Validation result with keys:
        - valid: bool - Overall validation result
        - found_count: int or 'container' - Number of files found
        - expected_count: int - Expected number of files
        - camera_path: str - Path to camera directory
        - first_image_preview: str - Base64 PNG of first image (if valid)
        - image_size: tuple - (width, height) of images
        - sample_files: list - Sample of matching filenames
        - format_detected: str - Detected file format
        - error: str or None - Error message if validation failed
        - suggested_pattern: str or None - Suggested pattern if files don't match
    """
    # Image sourcing comes from the source's settings sidecar (explicit args
    # override); config supplies only the source pointer + rig camera count.
    source, image = _resolve_image_settings(
        config, source_path_idx, image_format, image_type, num_images
    )
    fmt = image["image_format"]
    cal_image_type = image["image_type"]
    expected_count = int(image.get("n_views") or 1)
    camera_path = build_calibration_camera_path(
        source, image, camera, config.camera_count
    )

    # Create a frame reader function for preview generation
    def read_frame(idx: int) -> np.ndarray:
        return read_calibration_image(
            idx,
            camera,
            config,
            source_path_idx,
            image_format=fmt,
            image_type=cal_image_type,
        )

    # Use the generic validator
    return validate_images_generic(
        camera_path=camera_path,
        camera=camera,
        image_format=fmt,
        image_type=cal_image_type,
        expected_count=expected_count,
        zero_based_indexing=bool(image.get("zero_based_indexing", False)),
        read_frame_fn=read_frame,
    )


def get_calibration_frame_count(
    camera: int, config: Config, source_path_idx: int = 0
) -> int:
    """Auto-detect number of calibration images from directory.

    Counts matching files or returns container frame count.

    Parameters
    ----------
    camera : int
        Camera number (1-based)
    config : Config
        Configuration object with calibration settings
    source_path_idx : int, optional
        Index into source_paths list, defaults to 0

    Returns
    -------
    int
        Number of calibration images found
    """
    # Image sourcing comes from the source's settings sidecar.
    source, image = _resolve_image_settings(config, source_path_idx)
    camera_path = build_calibration_camera_path(
        source, image, camera, config.camera_count
    )

    image_type = image["image_type"]
    fmt = image["image_format"]
    stored_count = int(image.get("n_views") or 1)

    if not camera_path.exists():
        return 0

    if image_type == "lavision_set":
        # For .set files, we would need to read the file to get count
        # Return the stored count as fallback
        return stored_count

    elif image_type == "cine":
        # Get frame count from .cine file
        try:
            from .readers.cine_reader import get_cine_frame_count

            if "%" in fmt:
                cine_filename = fmt % camera
            else:
                cine_filename = fmt
            cine_path = camera_path / cine_filename
            if cine_path.exists():
                return get_cine_frame_count(str(cine_path))
        except Exception:
            pass
        return stored_count

    elif image_type == "lavision_im7":
        pattern = format_to_glob(fmt)
        return len(list(camera_path.glob(pattern)))

    else:
        # Standard formats
        pattern = format_to_glob(fmt)
        return len(list(camera_path.glob(pattern)))
