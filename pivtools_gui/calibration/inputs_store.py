"""calibration.inputs_store — the sidecar ``inputs.mat`` that lives beside a model record.

A calibration model ``.mat`` stores the solved OUTPUT (camera params + released board). This
module stores the INPUTS that produced it — the detected dot points and the user-clicked
world-frame coordinates — in a sibling ``inputs.mat`` in the same model dir. That makes
"delete the model, regenerate without re-detecting or re-clicking" work: the detections and
clicks survive the model.

Layout (one file per model dir, ``<model_dir>/inputs.mat``):

    schema_version     int
    path_type          'joint' | 'mono' | 'stereo' | 'stepped'
    board_type         'dotboard' | 'charuco' | 'stepped'
    coords_json        JSON string of the path-specific clicked inputs (None -> "null")
    det_count          number of (camera, view) detections stored (0 -> field absent)
    detections         flat struct array, one element per (camera, view)
    image_size_by_cam  (M, 3) int rows [camera, width, height]

The clicked inputs are stored as ONE JSON string rather than a MATLAB struct: they are
already plain nested dicts (the same shape the config block used), JSON round-trips the
ragged ``anchors`` / ``same_as`` shapes losslessly, and the existing
``_global_grid_spec_from_cfg`` consumes that exact dict unchanged. Detections are stored as
native ``.mat`` arrays (not pickle) so the file is a real model artefact.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat, savemat

from .detection.base import DetectionResult
from .record import _empty_if_none, _scalar

SCHEMA_VERSION = 1
INPUTS_FILENAME = "inputs.mat"

# Serialises the read-merge-write in save_inputs across the Flask worker threads: the detection
# save and each click-commit both merge into the same file, so without this two writers could
# interleave and lose an update. Reads are made safe from a half-written file by the atomic
# temp-then-replace below, not by this lock.
_write_lock = threading.Lock()

# Sentinel for save_inputs: a field left unset is preserved from the existing file rather
# than overwritten. None is a real value (clear the coords), so it cannot double as "unset".
_UNSET = object()


@dataclass
class InputsRecord:
    """The deserialised sidecar: clicked coords + detected points for one model dir."""

    path_type: str
    board_type: str
    coords: Optional[Dict[str, Any]] = None
    detections: Optional[Dict[int, List[DetectionResult]]] = None
    image_size_by_cam: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    # Board geometry (the ``record.geometry_meta`` dict) that produced the detections, so a model
    # can be regenerated — and the GUI/CLI can read geometry — without re-typing it into config.
    # None when the writer did not supply it (e.g. a legacy sidecar).
    board_params: Optional[Dict[str, Any]] = None
    # Identifies the parameters (n_views, format, board params, ...) that produced the stored
    # detections, so a caller can tell whether they still match the current request before
    # reusing them instead of re-detecting. Empty when detections were not param-keyed.
    det_key: str = ""


def inputs_path(model_dir) -> Path:
    """Sidecar path for a model dir (the dir that also holds the solved model ``.mat``)."""
    return Path(model_dir) / INPUTS_FILENAME


def joint_det_key(board, n_views, image_format, image_type, cameras, params) -> str:
    """Stable id for the parameters that produced a joint detection set.

    SOURCE-INDEPENDENT on purpose: the sidecar already lives in the source's model dir, so the
    source need not be in the key. Stored next to the detections (``InputsRecord.det_key``) so a
    load reuses them only when the current request's params still match — and so the GUI route
    and the headless CLI, which compute it from the same fields, share one detection cache. A
    changed n_views / format / board param yields a new key and forces a re-detect.
    """
    sig = (
        str(board), int(n_views), str(image_format), str(image_type),
        tuple(sorted(int(c) for c in cameras)), repr(params),
    )
    return hashlib.md5(repr(sig).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# JSON helpers (coords block + free-form detector diagnostics)
# ---------------------------------------------------------------------------

def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default)


# Cap on a stored diagnostics list so a debug array (e.g. a full-frame flat_field image dumped
# into a failed view's diagnostics) cannot bloat the sidecar — see _diagnostics_for_storage.
_MAX_DIAG_LIST = 64

# Array fields of a raw per-level grid dict that must come back as ndarrays for the fit.
_LEVEL_ARRAY_FIELDS = ("centers", "grid_indices", "H", "vec1", "vec2")


def _diagnostics_for_storage(diag: Dict[str, Any]) -> Dict[str, Any]:
    """Keep small scalar diagnostics; drop bulky arrays / images / nested dicts.

    Diagnostics are debug metadata, NOT a solve input — the resolvers and solve never read them.
    A detector can park a whole image in here (a failed view's ``flat_field``), which JSON would
    inflate to tens of MB; this keeps the useful scalars (counts, angles, the error string) and
    drops the bulk so the sidecar stays small. Genuine fit inputs (e.g. the stepped per-level
    grids) live on first-class ``DetectionResult`` fields, not here.
    """
    out: Dict[str, Any] = {}
    for k, v in (diag or {}).items():
        if isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (np.integer, np.floating, np.bool_)):
            out[k] = v.item()
        elif isinstance(v, (list, tuple)) and len(v) <= _MAX_DIAG_LIST and all(
            isinstance(x, (bool, int, float, str)) for x in v
        ):
            out[k] = list(v)
        # else (ndarray, long list, nested dict, image): dropped — not needed to re-solve
    return out


def _rehydrate_level(lv: Any) -> Optional[Dict[str, Any]]:
    """One loaded per-level grid dict -> arrays restored to ndarray (or None)."""
    if not isinstance(lv, dict):
        return None
    out = dict(lv)
    if "centers" in out:
        out["centers"] = np.asarray(out["centers"], dtype=np.float64).reshape(-1, 2)
    if "grid_indices" in out:
        out["grid_indices"] = np.asarray(out["grid_indices"], dtype=np.int64).reshape(-1, 2)
    if out.get("H") is not None:
        out["H"] = np.asarray(out["H"], dtype=np.float64).reshape(3, 3)
    for k in ("vec1", "vec2"):
        if out.get(k) is not None:
            out[k] = np.asarray(out[k], dtype=np.float64).reshape(-1)
    return out


def _level_data_from(json_str: str) -> Optional[Dict[str, Any]]:
    """Inverse of ``_dumps(level_data)`` — rehydrate the two per-level grids to ndarrays."""
    try:
        ld = json.loads(json_str or "null")
    except (ValueError, TypeError):
        return None
    if not isinstance(ld, dict):
        return None
    return {"a": _rehydrate_level(ld.get("a")), "b": _rehydrate_level(ld.get("b"))}


def _str_field(v: Any) -> str:
    """A loaded ``.mat`` string field -> python str ('' for an empty/absent field)."""
    if v is None:
        return ""
    s = _scalar(v)
    if s is None:
        return ""
    if isinstance(s, np.ndarray):  # squeeze left an empty/odd array — no usable string
        return ""
    if isinstance(s, (bytes, np.bytes_)):  # some scipy versions load char arrays as bytes
        return s.decode("utf-8", errors="replace")
    return str(s)


# ---------------------------------------------------------------------------
# DetectionResult <-> mat struct
# ---------------------------------------------------------------------------

def _detection_to_dict(camera: int, view: int, d: DetectionResult) -> Dict[str, Any]:
    """One DetectionResult -> a flat dict (one struct-array element). None arrays become
    empty arrays; ``spacing_mm`` None becomes NaN; ``diagnostics`` becomes a JSON string."""
    return {
        "camera": int(camera),
        "view": int(view),
        "success": int(bool(d.success)),
        "board_type": str(d.board_type),
        "image_points": np.asarray(d.image_points, dtype=np.float64).reshape(-1, 2),
        "board_local_points": np.asarray(d.board_local_points, dtype=np.float64).reshape(-1, 3),
        "grid_indices": _empty_if_none(d.grid_indices),
        "point_ids": _empty_if_none(d.point_ids),
        "board_to_pixel": _empty_if_none(d.board_to_pixel),
        "spacing_mm": float(d.spacing_mm) if d.spacing_mm is not None else float("nan"),
        "synthetic_mask": _empty_if_none(d.synthetic_mask),
        "level_data_json": _dumps(d.level_data),
        "diagnostics_json": _dumps(_diagnostics_for_storage(d.diagnostics)),
    }


def _arr_or_none(obj, name: str) -> Optional[np.ndarray]:
    a = np.asarray(getattr(obj, name, []), dtype=np.float64)
    return None if a.size == 0 else a


def _detection_from(obj) -> Tuple[int, int, DetectionResult]:
    """Inverse of ``_detection_to_dict`` for one loaded struct-array element."""
    ip = np.asarray(getattr(obj, "image_points", []), dtype=np.float64)
    ip = ip.reshape(-1, 2) if ip.size else np.empty((0, 2))
    blp = np.asarray(getattr(obj, "board_local_points", []), dtype=np.float64)
    blp = blp.reshape(-1, 3) if blp.size else np.empty((0, 3))

    gi = _arr_or_none(obj, "grid_indices")
    gi = None if gi is None else gi.astype(np.int64).reshape(-1, 2)
    pid = _arr_or_none(obj, "point_ids")
    pid = None if pid is None else pid.astype(np.int64).reshape(-1)
    b2p = _arr_or_none(obj, "board_to_pixel")
    b2p = None if b2p is None else b2p.reshape(3, 3)
    sm = _arr_or_none(obj, "synthetic_mask")
    sm = None if sm is None else sm.astype(bool).reshape(-1)

    sp = _scalar(getattr(obj, "spacing_mm", float("nan")))
    spacing = None if (sp is None or sp != sp) else float(sp)  # NaN -> None
    try:
        diag = json.loads(_str_field(getattr(obj, "diagnostics_json", "")) or "{}")
    except (ValueError, TypeError):
        diag = {}
    level_data = _level_data_from(_str_field(getattr(obj, "level_data_json", "")))

    d = DetectionResult(
        success=bool(int(_scalar(getattr(obj, "success", 0)))),
        board_type=str(_scalar(getattr(obj, "board_type", ""))),
        image_points=ip,
        board_local_points=blp,
        grid_indices=gi,
        point_ids=pid,
        board_to_pixel=b2p,
        spacing_mm=spacing,
        synthetic_mask=sm,
        level_data=level_data,
        diagnostics=diag,
    )
    return int(_scalar(obj.camera)), int(_scalar(obj.view)), d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_inputs(model_dir) -> InputsRecord:
    """Load the sidecar for a model dir. Raises ``FileNotFoundError`` if absent."""
    path = inputs_path(model_dir)
    if not path.is_file():
        raise FileNotFoundError(f"calibration inputs sidecar not found: {path}")
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)

    coords_json = _str_field(mat.get("coords_json")) or "null"
    try:
        coords = json.loads(coords_json)
    except (ValueError, TypeError):
        coords = None

    board_params_json = _str_field(mat.get("board_params_json")) or "null"
    try:
        board_params = json.loads(board_params_json)
    except (ValueError, TypeError):
        board_params = None

    detections: Optional[Dict[int, List[DetectionResult]]] = None
    det_count = int(_scalar(mat.get("det_count", 0))) if "det_count" in mat else 0
    if "detections" in mat and det_count > 0:
        grouped: Dict[int, List[Tuple[int, DetectionResult]]] = {}
        for obj in np.atleast_1d(mat["detections"]):
            cam, view, d = _detection_from(obj)
            grouped.setdefault(cam, []).append((view, d))
        detections = {
            cam: [d for _, d in sorted(views, key=lambda t: t[0])]
            for cam, views in grouped.items()
        }

    image_size: Dict[int, Tuple[int, int]] = {}
    if "image_size_by_cam" in mat:
        rows = np.asarray(mat["image_size_by_cam"], dtype=np.int64)
        if rows.size:
            for r in rows.reshape(-1, 3):
                image_size[int(r[0])] = (int(r[1]), int(r[2]))

    return InputsRecord(
        path_type=str(_scalar(mat.get("path_type", ""))),
        board_type=str(_scalar(mat.get("board_type", ""))),
        coords=coords,
        detections=detections,
        image_size_by_cam=image_size,
        det_key=_str_field(mat.get("det_key")),
        board_params=board_params,
    )


def try_load_inputs(model_dir) -> Optional[InputsRecord]:
    """``load_inputs`` or ``None`` if the sidecar is absent OR unreadable.

    A corrupt / truncated file (e.g. caught mid-write before the atomic replace landed, or a
    stale incompatible schema) is treated as "no sidecar" rather than propagated — a missing
    cache is always recoverable by re-detecting / re-clicking, and must never 500 the live
    overlay. The strict ``load_inputs`` still raises so explicit callers can tell the difference.
    """
    try:
        return load_inputs(model_dir)
    except Exception:
        return None


def save_inputs(
    model_dir,
    *,
    path_type: str,
    board_type: str,
    coords: Any = _UNSET,
    detections: Any = _UNSET,
    image_size_by_cam: Any = _UNSET,
    det_key: Any = _UNSET,
    board_params: Any = _UNSET,
) -> Path:
    """Write the sidecar, MERGING with any existing file.

    Fields left at ``_UNSET`` are preserved from the existing file — detection runs and click
    commits update the sidecar independently without clobbering each other. The merge is done
    by reloading into python objects and rewriting the whole file, so no loaded mat-struct is
    ever re-serialised.
    """
    model_dir = Path(model_dir)
    with _write_lock:
        return _save_inputs_locked(
            model_dir,
            path_type=path_type,
            board_type=board_type,
            coords=coords,
            detections=detections,
            image_size_by_cam=image_size_by_cam,
            det_key=det_key,
            board_params=board_params,
        )


def _save_inputs_locked(
    model_dir: Path,
    *,
    path_type: str,
    board_type: str,
    coords: Any,
    detections: Any,
    image_size_by_cam: Any,
    det_key: Any,
    board_params: Any,
) -> Path:
    prev = try_load_inputs(model_dir)

    fin_coords = coords if coords is not _UNSET else (prev.coords if prev else None)
    fin_dets = detections if detections is not _UNSET else (prev.detections if prev else None)
    if image_size_by_cam is not _UNSET:
        fin_sizes = image_size_by_cam
    else:
        fin_sizes = prev.image_size_by_cam if prev else {}
    fin_key = det_key if det_key is not _UNSET else (prev.det_key if prev else "")
    fin_params = (
        board_params if board_params is not _UNSET else (prev.board_params if prev else None)
    )

    data: Dict[str, Any] = {
        "schema_version": int(SCHEMA_VERSION),
        "path_type": str(path_type),
        "board_type": str(board_type),
        "coords_json": _dumps(fin_coords),  # None -> "null"
        "board_params_json": _dumps(fin_params),  # None -> "null"
        "det_key": str(fin_key or ""),
    }

    items: List[Dict[str, Any]] = []
    if fin_dets:
        for cam in sorted(fin_dets):
            for view, d in enumerate(fin_dets[cam]):
                items.append(_detection_to_dict(cam, view, d))
    data["det_count"] = int(len(items))
    if items:
        data["detections"] = items

    rows = [[int(c), int(wh[0]), int(wh[1])] for c, wh in sorted((fin_sizes or {}).items())]
    if rows:
        data["image_size_by_cam"] = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    else:
        data["image_size_by_cam"] = np.empty((0, 3), dtype=np.int64)

    model_dir.mkdir(parents=True, exist_ok=True)
    path = inputs_path(model_dir)
    # Atomic write: a concurrent reader (the live resolve_grid loop reads this file on every
    # frame) must never see a half-written .mat. savemat to a temp file, then os.replace — an
    # atomic rename on the same filesystem — so readers see either the old file or the new one.
    tmp = path.with_name(path.name + ".tmp")
    savemat(str(tmp), data, oned_as="row")
    os.replace(tmp, path)
    return path
