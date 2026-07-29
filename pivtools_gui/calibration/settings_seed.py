"""calibration.settings_seed — build a settings template from the records on disk.

When a calibration source has no ``settings.yaml`` yet (all pre-existing
datasets), the GUI needs something to show. The persisted YAML config is
presumed stale and is never consulted; the model records in the source are
the truth about what actually produced a calibration, so the seed recovers
what it can from the newest records:

- board geometry / ``model_type`` / ``datum_frame``  from ``board_meta["geometry"]``
- ``dt`` / ``n_views`` / ``px_per_mm``               from ``board_meta``
- ``camera`` / ``camera_pair``                       from the record folder names

The image block (format / type / subfolders) is never stamped into records,
so it stays at template defaults — the existing validate → suggested-pattern
flow walks the user to the right format in one click.

Seeding is a best-effort convenience: a record that fails to load is skipped
with a warning (the user reviews the seeded form before anything runs), and
nothing is written to disk here — the seed persists only when the GUI saves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pivtools_core.calibration_settings import default_settings, root_for_source

from . import record as rec

logger = logging.getLogger(__name__)


def _mtime(path: Path) -> float:
    """mtime, tolerant of a file vanishing between glob and stat (best-effort seed)."""
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def _newest(paths: List[Path]) -> Optional[Path]:
    return max(paths, key=_mtime, default=None)


def _geometry_to_method_block(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Map a record's ``board_meta.geometry`` onto a settings ``methods`` block."""
    board = geometry.get("board_type", "")
    block: Dict[str, Any] = {}
    if board == "dotboard":
        for key in ("dot_spacing_mm", "k_neighbors"):
            if geometry.get(key) is not None:
                block[key] = geometry[key]
    elif board == "charuco":
        for key in ("squares_h", "squares_v", "marker_ratio", "aruco_dict", "min_corners"):
            if geometry.get(key) is not None:
                block[key] = geometry[key]
        # geometry stamps metres as square_size_m; the settings block uses square_size
        if geometry.get("square_size_m") is not None:
            block["square_size"] = geometry["square_size_m"]
    elif board == "stepped":
        for key in ("dot_spacing_mm", "step_height_mm", "board_thickness_mm", "level_offset_mm"):
            if geometry.get(key) is not None:
                block[key] = geometry[key]
    if geometry.get("model_type") is not None:
        block["model_type"] = geometry["model_type"]
    return block


def _apply_method_block(
    settings: Dict[str, Any], board_meta: Dict[str, Any], method_key: str
) -> None:
    """Fold one record's per-board geometry into its methods block (in place)."""
    geometry = board_meta.get("geometry") or {}
    if geometry and method_key in settings["methods"]:
        settings["methods"][method_key].update(_geometry_to_method_block(geometry))
    if board_meta.get("px_per_mm") is not None and method_key == "scale_factor":
        settings["methods"]["scale_factor"]["px_per_mm"] = float(board_meta["px_per_mm"])


def _apply_shared_rig(settings: Dict[str, Any], board_meta: Dict[str, Any]) -> None:
    """Fold the rig/image keys shared across boards into the template (in place).

    Called exactly once, with the newest record's meta — these keys describe the
    rig, not a board, so the last real calibration run is the only valid witness.
    """
    geometry = board_meta.get("geometry") or {}
    if geometry.get("datum_frame") is not None:
        settings["rig"]["datum_frame"] = int(geometry["datum_frame"])
    if board_meta.get("dt") is not None:
        settings["rig"]["dt"] = float(board_meta["dt"])
    if board_meta.get("n_views") is not None:
        settings["image"]["n_views"] = int(board_meta["n_views"])


def _mono_candidates(root: Path) -> List[Tuple[Path, int, str]]:
    """(record_path, camera, board) for every mono record under the root."""
    out: List[Tuple[Path, int, str]] = []
    for cam_dir in root.glob("Cam*"):
        try:
            camera = int(cam_dir.name[3:])
        except ValueError:
            continue
        for board_dir in cam_dir.glob("*_planar"):
            board = board_dir.name[: -len("_planar")]
            out.extend(
                (p, camera, board) for p in board_dir.glob("model/model_*.mat")
            )
    return out


def _stereo_candidates(root: Path) -> List[Tuple[Path, int, int]]:
    """(record_path, cam1, cam2) for every stereo record under the root."""
    out: List[Tuple[Path, int, int]] = []
    for pair_dir in root.glob("stereo_cam*_cam*"):
        parts = pair_dir.name.split("_")  # stereo, cam{A}, cam{B}
        try:
            cam1, cam2 = int(parts[1][3:]), int(parts[2][3:])
        except (IndexError, ValueError):
            continue
        out.extend((p, cam1, cam2) for p in pair_dir.glob("model/stereo_model_*.mat"))
    return out


def seed_settings(source: Path) -> Dict[str, Any]:
    """Settings template for a source with no sidecar, upgraded from records.

    Returns the defaults template enriched with whatever the newest model
    records on disk can prove. Never writes to disk and never reads the
    persisted YAML config (which is presumed stale).
    """
    settings = default_settings()
    # Visible starting guesses for the GUI form (the user reviews these before
    # saving); the store's own defaults keep these None so a file that never
    # had them set still fails loudly at load time.
    settings["image"]["image_format"] = "calib%05d.tif"
    settings["image"]["image_type"] = "standard"
    root = root_for_source(Path(source))
    if not root.is_dir():
        return settings

    # The shared rig/image keys (dt, datum_frame, n_views) come from the single
    # newest readable record overall — they describe the rig, and applying them
    # per-record would let insertion order pick a stale witness.
    newest_shared: Optional[Tuple[float, Dict[str, Any]]] = None

    def _consider_shared(path: Path, board_meta: Dict[str, Any]) -> None:
        nonlocal newest_shared
        mtime = _mtime(path)
        if newest_shared is None or mtime > newest_shared[0]:
            newest_shared = (mtime, board_meta)

    # Per-board mono geometry: newest record of each board wins for its block.
    monos = _mono_candidates(root)
    by_board: Dict[str, List[Tuple[Path, int, str]]] = {}
    for item in monos:
        by_board.setdefault(item[2], []).append(item)
    newest_mono: Optional[Tuple[Path, int, str]] = None
    for board, items in by_board.items():
        path = _newest([p for p, _, _ in items])
        entry = next(i for i in items if i[0] == path)
        if newest_mono is None or _mtime(path) > _mtime(newest_mono[0]):
            newest_mono = entry
        try:
            record = rec.load_mono(path)
        except Exception:
            logger.warning("settings seed: skipping unreadable record %s", path)
            continue
        meta = record.board_meta or {}
        _apply_method_block(settings, meta, board)
        _consider_shared(path, meta)

    if newest_mono is not None:
        settings["rig"]["camera"] = newest_mono[1]

    # Stereo: newest pair wins camera_pair + its board's stereo method block.
    stereos = _stereo_candidates(root)
    newest_stereo = _newest([p for p, _, _ in stereos])
    if newest_stereo is not None:
        _, cam1, cam2 = next(i for i in stereos if i[0] == newest_stereo)
        settings["rig"]["camera_pair"] = [cam1, cam2]
        try:
            record = rec.load_stereo(newest_stereo)
        except Exception:
            logger.warning(
                "settings seed: skipping unreadable record %s", newest_stereo
            )
        else:
            # Method blocks are per physical BOARD (shared mono/stereo), so a
            # stereo record seeds its base board's block.
            meta = record.board_meta or {}
            _apply_method_block(settings, meta, record.board_type)
            _consider_shared(newest_stereo, meta)

    if newest_shared is not None:
        _apply_shared_rig(settings, newest_shared[1])

    return settings
