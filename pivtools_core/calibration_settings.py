"""pivtools_core.calibration_settings — per-source calibration settings sidecar.

One ``settings.yaml`` per calibration source, living in the same
``<source>/calibration`` root as the model records, holding the pre-model
form state that used to sit (and go stale) in ``config.yaml``:

    <root>/settings.yaml        root = root_for_source(source)

The persisted YAML config keeps only the pointer (``calibration_sources``,
``source``, ``source_idx``, ``active``); everything else about a calibration
lives with the calibration data so it can never point at the wrong source.

Contract
--------
- ``image.image_format`` and ``image.image_type`` are REQUIRED: a file
  missing them is stale/unset and ``load_settings`` raises. Board geometry
  and ``rig.dt`` are required at *generate* time and enforced there.
- Detector-tuning knobs (``k_neighbors``, ``marker_ratio``, ``min_corners``,
  ``fix_k2``, ``interpolator``, ``start_index``, ...) have their defaults
  applied here and nowhere else — this file replaces the old three-layer
  (CLI template / Config property / frontend fallback) default system.
- Writes are partial deep-merges: callers send only the block they own and
  the rest of the file is preserved. Lists replace wholesale.
- Atomic write (tmp + ``os.replace``) with a PermissionError retry loop —
  sources routinely live on OneDrive/network shares.

Seeding a missing file from the model records on disk is a GUI-layer concern
(records are read by ``pivtools_gui.calibration``); see
``pivtools_gui/calibration/settings_seed.py``.
"""

from __future__ import annotations

import copy
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

SCHEMA_VERSION = 1
SETTINGS_FILENAME = "settings.yaml"

PathLike = Union[str, Path]


def root_for_source(source: PathLike) -> Path:
    """Calibration-output root for an explicit source path.

    Directory sources -> ``<source>/calibration``. Container-file sources
    (``.set``/``.cine``) -> ``<source>.parent/calibration``.
    """
    source = Path(source)
    base = source.parent if source.suffix else source
    return base / "calibration"


def settings_path(source: PathLike) -> Path:
    """Path of the settings sidecar for a calibration source."""
    return root_for_source(source) / SETTINGS_FILENAME


# ---------------------------------------------------------------------------
# Defaults template
# ---------------------------------------------------------------------------

# Required-at-read keys carry None so a hand-stripped file fails loudly;
# knob defaults are real values and are applied ONLY here.
_DEFAULTS: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "image": {
        "image_format": None,  # REQUIRED — no default, missing => raise
        "image_type": None,  # REQUIRED — standard | cine | lavision_set | lavision_im7
        "n_views": None,  # optional; frame-count auto-detect is the fallback
        "start_index": 1,
        "zero_based_indexing": False,
        "use_camera_subfolders": False,
        "camera_subfolders": [],
    },
    "rig": {
        "camera": 1,
        "camera_pair": [1, 2],
        "dt": None,  # REQUIRED before generate; never defaulted
        "datum_frame": 1,  # 1-based (GUI native; CLI derives datum_index)
        "interpolator": "lanczos",
        "piv_type": "instantaneous",
    },
    # use_release_object is deliberately NOT templated: its default is
    # context-dependent (stereo True, mono False) and a stored value would
    # override both. It is honoured when a user sets it explicitly.
    "fit": {
        "distortion_model": "standard",
        "fix_aspect_ratio": True,
        "self_cal_n_images": 20,
        "world_frame": "default",
        # Headless world-frame inputs for the CLI detect commands: a clicks-JSON
        # path for cam2, and grid-index specs ({origin,x_axis,y_axis} by dot
        # [col,row], or a path to such JSON). None = absent (GUI clicks flow).
        "world_frame_cam2": None,
        "world_frame_grid": None,
        "world_frame_grid_cam2": None,
    },
    "global_coordinates": {
        "enabled": False,
        "datum_camera": 1,
        "datum_pixel": None,
        "datum_physical": [0.0, 0.0],
        "datum_frame": 1,
        "overlap_pairs": [],
    },
    # One block per physical BOARD, shared by the mono and stereo flows of that
    # board (the same physical target has one geometry) — the same sharing the
    # per-method config blocks had. The stereo/mono distinction lives in
    # ``calibration.active`` (the YAML pointer), not here.
    #
    # fix_k2 is deliberately NOT templated (like use_release_object): its GUI
    # default is context-dependent (stereo few-view True, mono/CLI False) and a
    # stored value would override both. Readers default it False when absent;
    # it is honoured when a user sets it explicitly.
    "methods": {
        "dotboard": {
            "dot_spacing_mm": None,  # board geometry — required at generate
            "k_neighbors": 9,
            "model_type": "pinhole",
        },
        "charuco": {
            "squares_h": None,
            "squares_v": None,
            "square_size": None,  # metres (ChArUco native unit)
            "marker_ratio": 0.5,
            "aruco_dict": "DICT_4X4_1000",
            "min_corners": 6,
            "model_type": "pinhole",
        },
        "stepped": {
            "dot_spacing_mm": None,
            "step_height_mm": None,
            "board_thickness_mm": None,
            "level_offset_mm": None,  # None => derive dot_spacing_mm / 2
        },
        "stepped_stereo": {
            "stereo_config": "auto",
            "model_type": "pinhole",
        },
        "scale_factor": {
            "px_per_mm": None,
        },
    },
}


def default_settings() -> Dict[str, Any]:
    """A fresh deep copy of the defaults template."""
    return copy.deepcopy(_DEFAULTS)


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
# path -> (mtime_ns, parsed dict). Guarded by _LOCK; values are deep-copied out.
_CACHE: Dict[Path, tuple] = {}


def _deep_merge(base: Dict[str, Any], partial: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``partial`` into ``base`` recursively; lists/scalars replace."""
    for key, value in partial.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _missing_message(source: Path, path: Path) -> str:
    return (
        f"No calibration settings for source '{source}' (expected {path}). "
        "Open this source in the GUI Calibration tab (which seeds and saves "
        "settings) or run 'pivtools-cli calibration init-settings "
        f"--source \"{source}\"' and fill in the required fields."
    )


def _validate_loaded(data: Dict[str, Any], path: Path) -> None:
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Calibration settings {path} has schema_version={version!r}, "
            f"expected {SCHEMA_VERSION}. The file is stale — re-save it from "
            "the GUI or re-run init-settings."
        )
    image = data.get("image") or {}
    for key in ("image_format", "image_type"):
        if not image.get(key):
            raise ValueError(
                f"Calibration settings {path} is missing required "
                f"'image.{key}'. Set it in the GUI Calibration tab or edit "
                "the file directly."
            )


def try_load_settings(source: PathLike) -> Optional[Dict[str, Any]]:
    """Like :func:`load_settings` but returns None when the file is absent.

    Validation errors on an existing file still raise — a present-but-broken
    sidecar must never be silently ignored.
    """
    if not settings_path(source).exists():
        return None
    return load_settings(source)


def load_settings(source: PathLike) -> Dict[str, Any]:
    """Load the settings sidecar for ``source``.

    Knob defaults are layered underneath the stored values so callers see a
    complete structure; required keys are validated (fail loud, no silent
    fallback). Raises ``FileNotFoundError`` when the sidecar does not exist.
    """
    source = Path(source)
    path = settings_path(source)
    with _LOCK:
        try:
            st = path.stat()
        except FileNotFoundError:
            _CACHE.pop(path, None)
            raise FileNotFoundError(_missing_message(source, path))
        # mtime+size key: Windows last-write time ticks at ~15 ms, so an
        # external write inside one tick would otherwise be served stale.
        stamp = (st.st_mtime_ns, st.st_size)
        cached = _CACHE.get(path)
        if cached is not None and cached[0] == stamp:
            # The cache holds validated, defaults-layered dicts, but an
            # external editor can write invalid content that keeps the stamp —
            # cheap re-validation keeps the fail-loud guarantee airtight.
            merged = copy.deepcopy(cached[1])
            _validate_loaded(merged, path)
            return merged
        with open(path, "r", encoding="utf-8") as f:
            stored = yaml.safe_load(f)
        if not isinstance(stored, dict):
            raise ValueError(
                f"Calibration settings {path} is not a YAML mapping — the "
                "file is corrupt. Re-save it from the GUI or re-run "
                "init-settings."
            )
        merged = _deep_merge(default_settings(), stored)
        _validate_loaded(merged, path)
        _CACHE[path] = (stamp, copy.deepcopy(merged))
    return merged


def save_settings(source: PathLike, partial: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge ``partial`` into the stored settings and write atomically.

    Blocks not present in ``partial`` are preserved; a missing file starts
    from the defaults template, so a sidecar PINS the knob defaults current at
    its creation (deliberate: each file is complete and self-describing; later
    template changes reach only new sources). Returns the merged settings.
    No validation on save — partial in-progress edits (e.g. an empty
    image_format while the user types) are legal on disk; readers validate.
    """
    source = Path(source)
    path = settings_path(source)
    with _LOCK:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                current = yaml.safe_load(f) or {}
            if not isinstance(current, dict):
                raise ValueError(
                    f"Calibration settings {path} is not a YAML mapping — "
                    "refusing to merge into a corrupt file."
                )
        else:
            current = default_settings()
        merged = _deep_merge(current, copy.deepcopy(partial))
        merged["schema_version"] = SCHEMA_VERSION
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = str(path) + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
        # Retry os.replace to ride out transient OneDrive/cloud-sync locks.
        for attempt in range(5):
            try:
                os.replace(tmp_path, path)
                break
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise
        # Invalidate rather than prime: the merge above is file-contents only
        # (no knob defaults layered), and the loader's cache-hit path returns
        # cached dicts as-is — priming here would serve un-defaulted structures
        # for any partial on-disk file until its mtime next changed.
        _CACHE.pop(path, None)
    return merged
