"""pivtools_cli.calibration_cli — CLI for the unified calibration package.

Fully YAML-driven (config block ``calibration:``) with argparse overrides, mirroring
the v1 command pattern but writing models into the calibration SOURCE folder. New
subcommands: ``detect-planar``, ``detect-charuco``, ``detect-stereo``, ``apply-calibration``, ``apply-stereo``.

Example config:

    calibration:
      active: charuco            # charuco | dotboard
      source: /data/calib        # directory of calibration images (model saved here)
      image_format: "calib%05d.png"
      n_views: 10
      start_index: 1
      datum_index: 0
      distortion_model: standard
      fix_aspect_ratio: true
      world_frame: default       # "default" or path to clicks JSON {origin,x_axis,y_axis}
      dt: 1.0
      camera: 1
      camera_pair: [1, 2]
      use_camera_subfolders: true       # gate: subfolders only apply when this is true
      camera_subfolders: [cam1, cam2]   # per-camera dirs (index = camera-1); else Cam{N} fallback
      charuco: {squares_h: 10, squares_v: 7, square_size: 0.03, marker_ratio: 0.5,
                aruco_dict: DICT_4X4_1000, min_corners: 6}
      dotboard: {dot_spacing_mm: 15.0}
      uncalibrated_dir: /proc/uncalibrated_piv/.../Cam1/instantaneous
      calibrated_dir:   /proc/calibrated_piv/.../Cam1/instantaneous
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pivtools_core import calibration_settings as cs
from pivtools_core.config import get_config
from pivtools_core.image_handling.calibration_loader import read_calibration_frame_at
from pivtools_core.image_handling.path_utils import (
    calibration_camera_folder,
    infer_image_type,
)
from pivtools_core.paths import vector_glob_from_format
from pivtools_gui.calibration import global_coords as gc2
from pivtools_gui.calibration import record as rec
from pivtools_gui.calibration import runio
from pivtools_gui.calibration import self_cal as c2sc
from pivtools_gui.calibration.camera_model import DistortionModel
from pivtools_gui.calibration.detection.base import DetectionResult
from pivtools_gui.calibration.detection.charuco import (
    CharucoBoardDetector,
    CharucoParams,
)
from pivtools_gui.calibration.detection.dotboard import DotboardDetector, DotboardParams
from pivtools_gui.calibration.detection.stepped import SteppedDetector, SteppedParams
from pivtools_gui.calibration.global_grid import (
    Anchor,
    Correspondence,
    GlobalGridSpec,
)
from pivtools_gui.calibration.inputs_store import (
    joint_det_key,
    save_inputs,
    try_load_inputs,
)
from pivtools_gui.calibration.joint_driver import run_joint_from_spec
from pivtools_gui.calibration.pipeline import Calibrator, build_scale_factor_record
from pivtools_gui.calibration.stereo_model import StereoCalibrator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _cfg2(config) -> dict:
    """The YAML calibration block — now just the pointer (sources, source_idx, active)."""
    return config.calibration


def _settings_cfg(source: Path) -> dict:
    """Flat legacy-key view of the SOURCE's settings sidecar.

    ``image`` + ``rig`` + ``fit`` knobs at top level, per-board method blocks
    keyed by board name — the key shape the command bodies (and
    ``_board_params`` / ``_resolve_datum_index`` / ``_loader_kwargs``) have
    always consumed. Built from the source's sidecar (defaults template when
    absent — required keys stay None there and fail loudly downstream); the
    YAML config contributes nothing here.
    """
    settings = cs.try_load_settings(source) or cs.default_settings()
    flat: Dict[str, Any] = {}
    flat.update(settings.get("image") or {})
    flat.update(settings.get("rig") or {})
    flat.update(settings.get("fit") or {})
    flat.update(settings.get("methods") or {})
    return flat


def _image_format_cli(args, scfg: dict, source: Path) -> str:
    """``--image-format`` > sidecar ``image.image_format`` > loud error (no default)."""
    v = getattr(args, "image_format", None) or scfg.get("image_format")
    if not v:
        raise SystemExit(
            "calibration: image.image_format is required — pass --image-format "
            f"or set it in {cs.settings_path(source)} "
            "(`pivtools-cli calibration init-settings` seeds a template)"
        )
    return v


def _n_views_cli(args, scfg: dict, source: Path) -> int:
    """``--n-views`` > sidecar ``image.n_views`` > loud error.

    The CLI detect commands load a fixed number of views, so unlike the GUI
    (which frame-counts the source) an unset count has no fallback here.
    """
    v = getattr(args, "n_views", None) or scfg.get("n_views")
    if not v:
        raise SystemExit(
            "calibration: image.n_views is required for CLI detection — pass "
            f"--n-views or set it in {cs.settings_path(source)}"
        )
    return int(v)


def _source(cfg: dict, override=None) -> Path:
    """Resolve the calibration SOURCE directory (where the model is saved and read).

    This is the folder that holds the calibration images, so the model lives WITH
    them and is shared by every PIV run that references this calibration — regardless
    of the run's ``base_path``. Precedence:

        explicit override  >  calibration.source  >  calibration.calibration_sources[idx]

    The final fallback ties v2 to the SAME calibration source the rest of pivtools
    loads images from, so no second config is needed for the shared-model behaviour.
    """
    if override:
        return Path(override)
    if cfg.get("source"):
        return Path(cfg["source"])
    return get_config().get_calibration_source(int(cfg.get("source_idx", 0)))


def _resolve_datum_index(cfg: Dict[str, Any]) -> int:
    """0-based datum view index, bridging the GUI's 1-based ``datum_frame``.

    The CLI's native key is ``datum_index`` (0-based position into the loaded views);
    the GUI persists ``datum_frame`` (1-based frame number, ``index = frame - 1``). A
    config written by either tool must resolve the same way. Both present and disagreeing
    is a real ambiguity, not something to guess past — raise. Neither present -> 0 (the
    first loaded view is the datum).
    """
    has_index = cfg.get("datum_index") is not None
    has_frame = cfg.get("datum_frame") is not None
    if has_index and has_frame:
        idx = int(cfg["datum_index"])
        frame_idx = int(cfg["datum_frame"]) - 1
        if idx != frame_idx:
            raise SystemExit(
                f"calibration: datum_index={idx} and datum_frame={cfg['datum_frame']} "
                f"(index {frame_idx}) disagree — set one, or make them consistent"
            )
        return idx
    if has_index:
        return int(cfg["datum_index"])
    if has_frame:
        idx = int(cfg["datum_frame"]) - 1
        logger.info(
            "calibration: datum_frame=%s (1-based) -> datum_index=%d",
            cfg["datum_frame"],
            idx,
        )
        return idx
    return 0


def init_settings_command(args):
    """Write a fresh per-source settings sidecar template (headless seeding).

    Refuses to overwrite an existing file — settings are edited in place (GUI
    or text editor), never regenerated over the top of real values.
    """
    config = get_config()
    source = _source(_cfg2(config), getattr(args, "source", None))
    # save_settings mkdirs parents, so a typo'd --source would silently create
    # a sidecar in a directory tree that holds no calibration images.
    if not (source.is_dir() or source.is_file()):
        raise SystemExit(
            f"calibration: source {source} does not exist — init-settings "
            "seeds an existing calibration image location"
        )
    path = cs.settings_path(source)
    if path.exists():
        raise SystemExit(
            f"calibration: {path} already exists — edit it directly "
            "(init-settings never overwrites)"
        )
    cs.save_settings(source, {})
    print(
        f"calibration: wrote settings template {path}\n"
        "Fill in image.image_format, image.image_type, rig.dt and the board "
        "geometry under methods.<board> before detecting/generating."
    )


def _generate_dt_cli(args, source: Path) -> float:
    """dt to stamp into a generated record: ``--dt`` > settings ``rig.dt`` > error.

    Velocity scales linearly with dt so it has no safe default; every generated
    record carries it, and apply resolves request > model-stamped > error.
    """
    v = getattr(args, "dt", None)
    if v is not None:
        return float(v)
    settings = cs.try_load_settings(source)
    dt = ((settings or {}).get("rig") or {}).get("dt")
    if dt is None:
        raise SystemExit(
            "calibration: dt is required to generate a model — pass --dt or set "
            f"rig.dt in {cs.settings_path(source)} (velocity has no safe default)"
        )
    return float(dt)


# ---------------------------------------------------------------------------
# Board registry — the single dispatch point for board types.
#
# A board is described declaratively (how to build its params, its detector class,
# how to read its spacing). Detection/generation never branch on the board name;
# they go through this table. Adding a board later (e.g. ``stepped``) is one entry
# here plus its detector — no edits to the CLI commands or the Flask views.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardSpec:
    params_from: Callable[[dict], object]  # merged config dict -> Params dataclass
    detector_cls: type  # detector constructed from the Params
    spacing_mm: Callable[[object], float]  # physical feature spacing in mm


def _require_geometry(d: dict, key: str, where: str):
    """Pull a required board-geometry value; absent -> actionable error naming the path.

    Board geometry (dot spacing, square count/size) has no safe default — a wrong value
    silently rescales every world coordinate, the exact failure that survives review
    unnoticed. So the CLI refuses to guess, unlike detector-tuning knobs (k_neighbors,
    marker_ratio, ...) which keep defaults. This bites an unseeded or hand-edited
    settings sidecar whose ``methods.<board>`` block omits the geometry.

    Raises ``ValueError`` (not ``SystemExit``): the board-param builders are shared with
    the Flask routes (``views._resolve_board`` / ``stepped_views`` call ``_board_params``),
    where a ``SystemExit`` — a ``BaseException`` — escapes the app's ``except Exception``
    handling. ``ValueError`` is caught there and still surfaces on the CLI.
    """
    if d.get(key) is None:
        raise ValueError(
            f"calibration: {where}.{key} is required — set it in the source's "
            f"calibration/settings.yaml (GUI Calibration tab, or "
            f"`pivtools-cli calibration init-settings` then edit), or pass the "
            f"matching CLI flag (board geometry has no default)"
        )
    return d[key]


def _charuco_params_from(d: dict) -> CharucoParams:
    return CharucoParams(
        squares_h=int(_require_geometry(d, "squares_h", "methods.charuco")),
        squares_v=int(_require_geometry(d, "squares_v", "methods.charuco")),
        square_size_m=float(_require_geometry(d, "square_size", "methods.charuco")),
        marker_ratio=float(d.get("marker_ratio", 0.5)),
        aruco_dict=str(d.get("aruco_dict", "DICT_4X4_1000")),
        min_corners=int(d.get("min_corners", 6)),
    )


def _dotboard_params_from(d: dict) -> DotboardParams:
    return DotboardParams(
        dot_spacing_mm=float(
            _require_geometry(d, "dot_spacing_mm", "methods.dotboard")
        ),
        k_neighbors=int(d.get("k_neighbors", 9)),
    )


def _stepped_params_from(d: dict) -> SteppedParams:
    # dot_spacing + step_height + board_thickness are all rig geometry with no safe
    # default: step_height sets the level Z-separation (every pose), board_thickness sets
    # the opposite-face Z on a transmission stereo fit — a wrong value there silently
    # corrupts cam2's world points. level_offset is genuinely derived (defaults to
    # dot_spacing/2 in SteppedParams), so it stays optional.
    lo = d.get("level_offset_mm")
    return SteppedParams(
        dot_spacing_mm=float(
            _require_geometry(d, "dot_spacing_mm", "methods.stepped")
        ),
        step_height_mm=float(
            _require_geometry(d, "step_height_mm", "methods.stepped")
        ),
        board_thickness_mm=float(
            _require_geometry(d, "board_thickness_mm", "methods.stepped")
        ),
        level_offset_mm=None if lo is None else float(lo),
    )


BOARD_REGISTRY: Dict[str, BoardSpec] = {
    "charuco": BoardSpec(
        _charuco_params_from, CharucoBoardDetector, lambda p: p.square_size_mm
    ),
    "dotboard": BoardSpec(
        _dotboard_params_from, DotboardDetector, lambda p: p.dot_spacing_mm
    ),
    "stepped": BoardSpec(
        _stepped_params_from, SteppedDetector, lambda p: p.dot_spacing_mm
    ),
}


def _board_spec(board: str) -> BoardSpec:
    spec = BOARD_REGISTRY.get(board)
    if spec is None:
        raise ValueError(
            f"unknown board '{board}' (expected {'|'.join(BOARD_REGISTRY)})"
        )
    return spec


def _model_rms_str(cm) -> str:
    """One-line RMS summary for the console.

    Pinhole reports px RMS + fx; the single-plane polynomial reports its mm RMS; the
    stepped 3D polynomial (``Polynomial3DModel``) has no K, so it reports its px RMS
    plus the per-plane split (the DaVis poly diagnostic).
    """
    mt = getattr(cm, "model_type", "pinhole")
    if mt == "polynomial":
        return f"RMS_x={cm.rms_x_mm:.4f}mm RMS_y={cm.rms_y_mm:.4f}mm"
    if mt == "polynomial3d":
        planes = " ".join(f"{v:.3f}" for v in getattr(cm, "plane_rms_px", ()))
        return f"RMS={cm.rms_px:.4f}px (poly3d; per-plane {planes})"
    return f"RMS={cm.rms:.4f}px fx={cm.K[0, 0]:.1f}"


def _geometry_to_config(geo: Optional[dict]) -> dict:
    """Map a sidecar geometry dict (``record.geometry_meta`` shape) to the config-block key
    shape the ``params_from`` builders consume. Only ``square_size_m`` needs renaming (the
    config key is ``square_size``, also metres); ``board_type``/``model_type`` are provenance,
    not builder inputs, so they are dropped."""
    out = dict(geo or {})
    if "square_size_m" in out:
        out["square_size"] = out.pop("square_size_m")
    # Provenance-only keys, not board-geometry builder inputs — drop so they can never be
    # mistaken for a current-run value if a params_from builder later grows such a field.
    for k in ("board_type", "model_type", "datum_frame", "datum_camera"):
        out.pop(k, None)
    return out


def _board_params(
    cfg: dict,
    board: str,
    overrides: Optional[dict] = None,
    sidecar: Optional[dict] = None,
):
    """Build board params for a calibration run, source order ``overrides`` > ``sidecar`` > config.

    ``overrides`` are per-call values (the GUI panel / CLI ``--dot-spacing-mm`` args) and win;
    they are NOT persisted, so detection stays a pure function of its inputs. ``sidecar`` is the
    geometry recovered from an existing model's sidecar (``record.geometry_meta`` shape) so a
    re-run reads the geometry that produced the model rather than config. The
    ``calibration.<board>`` config block is the last-resort fallback. None layers are skipped.
    """
    merged = dict(cfg.get(board, {}) or {})
    if sidecar:
        merged.update(
            {k: v for k, v in _geometry_to_config(sidecar).items() if v is not None}
        )
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return _board_spec(board).params_from(merged)


def _build_detector(board: str, params):
    return _board_spec(board).detector_cls(params)


def _spacing_mm(board: str, params) -> float:
    return _board_spec(board).spacing_mm(params)


def _load_clicks(spec) -> Optional[Dict[str, object]]:
    if spec is None or spec == "default" or spec == "":
        return None
    p = Path(spec)
    data = json.loads(p.read_text())
    return {
        "origin": np.asarray(data["origin"], dtype=float),
        "x_axis": np.asarray(data["x_axis"], dtype=float),
        "y_axis": np.asarray(data["y_axis"], dtype=float),
    }


def _load_grid(spec) -> Optional[Dict[str, object]]:
    """Headless world-frame spec by dot GRID indices instead of pixel clicks.

    ``spec`` is a dict ``{origin: [col,row], x_axis: [col,row], y_axis: [col,row]}``
    (or a path to such a JSON). Returns None when absent.
    """
    if spec is None or spec == "" or spec == "default":
        return None
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text())
    return {
        "origin": [int(spec["origin"][0]), int(spec["origin"][1])],
        "x_axis": [int(spec["x_axis"][0]), int(spec["x_axis"][1])],
        "y_axis": [int(spec["y_axis"][0]), int(spec["y_axis"][1])],
    }


def _read_json_maybe(spec):
    """Return ``spec`` itself if it is already a dict, else parse it as a JSON path."""
    if isinstance(spec, (str, Path)):
        return json.loads(Path(spec).read_text())
    return spec


def _parse_fiducials(d: dict) -> Dict[str, List[float]]:
    """Pull the {origin, x_axis, y_axis} image-down pixel clicks from a spec dict."""
    return {k: [float(d[k][0]), float(d[k][1])] for k in ("origin", "x_axis", "y_axis")}


def _cam_dir(scfg: dict, source: Path, camera: int, num_cameras: int) -> Path:
    """Per-camera image directory: the CLI-resolved ``source`` + the loader's camera folder.

    The folder name is delegated to ``path_utils.calibration_camera_folder`` — the same
    resolver ``build_calibration_camera_path`` uses for the GUI/PIV image loaders — so the
    headless CLI lands on the same per-camera directory: the ``use_camera_subfolders`` gate
    and the ``camera_subfolders`` list are honoured, and a multi-camera rig with no explicit
    list falls back to ``Cam{N}``. Only ``source`` is CLI-specific (it may come from
    ``--source``), so just the folder is delegated, not the whole path.

    Caveat: the container-format gate keys off the settings ``image_type``/``image_format``,
    NOT the per-call ``--image-format``. If they disagree on format class (container vs
    per-file), the subfolder decision here can disagree with the file the loader then
    reads — a contradiction that surfaces as a loud FileNotFoundError, not silent wrong data.
    """
    image_type = scfg.get("image_type") or infer_image_type(
        scfg.get("image_format") or ""
    )
    folder = calibration_camera_folder(scfg, image_type, int(camera), num_cameras)
    return Path(source) / folder if folder else Path(source)


def _load_one(
    cam_dir: Path,
    image_format: str,
    frame_number: int,
    *,
    camera: int = 1,
    image_type: str,
    use_camera_subfolders: bool,
    zero_based: bool,
) -> np.ndarray:
    """Load a single calibration frame by image index (grayscale; 1-based unless ``zero_based``).

    Routes through the shared all-format reader (``read_calibration_frame_at``) so the
    CLI reads exactly what the GUI/PIV pipeline reads — standard tif/png AND LaVision
    ``.im7``/``.set`` and Phantom ``.cine`` — instead of the cv2.imread-only path that
    silently failed on ``.im7``. Full dynamic range is preserved (``normalize_uint8=False``);
    the detectors promote to float/uint8 themselves, so this is behaviour-preserving for the
    tif datasets. The loader flags (``image_type``, ``use_camera_subfolders``, ``zero_based``)
    are required and resolved in one place — :func:`_loader_kwargs` — so there is no second
    set of defaults to drift out of sync; every caller spreads ``**_loader_kwargs(...)``.
    """
    img = read_calibration_frame_at(
        camera_path=cam_dir,
        camera=int(camera),
        frame_idx=int(frame_number),
        image_format=image_format,
        image_type=image_type,
        zero_based_indexing=zero_based,
        use_camera_subfolders=use_camera_subfolders,
        normalize_uint8=False,
    )
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _load_views(
    cam_dir: Path,
    image_format: str,
    n_views: int,
    start_index: int,
    *,
    camera: int = 1,
    image_type: str,
    use_camera_subfolders: bool,
    zero_based: bool,
) -> List[np.ndarray]:
    return [
        _load_one(
            cam_dir,
            image_format,
            start_index + k,
            camera=camera,
            image_type=image_type,
            use_camera_subfolders=use_camera_subfolders,
            zero_based=zero_based,
        )
        for k in range(n_views)
    ]


def _loader_kwargs(cfg: dict, image_format: str) -> dict:
    """The single source of the loader flags ``_load_views``/``_load_one`` need.

    The image type is inferred from the *format extension* (authoritative — a
    ``--image-format`` ending ``.im7`` is im7 regardless of the config's legacy
    ``image_type`` default), so pointing the CLI at a LaVision dataset just needs the
    right ``--image-format``. ``_load_one``/``_load_views`` take these as required keyword
    args (no own defaults), so this dict is the one place the values are decided.
    """
    return {
        "image_type": infer_image_type(image_format),
        "use_camera_subfolders": bool(cfg.get("use_camera_subfolders", False)),
        "zero_based": bool(cfg.get("zero_based_indexing", False)),
    }


def _count_views(
    cam_dir: Path, image_format: str, start_index: int, max_views: int = 100000
) -> int:
    """Count consecutive calibration frames present from ``start_index`` upward."""
    n = 0
    while n < max_views and (cam_dir / (image_format % (start_index + n))).exists():
        n += 1
    return n


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _existing_mono_path(
    model_dir: Path, model_type: Optional[str], force: bool
) -> Optional[Path]:
    """Existing mono record of the requested type to reuse, or None to compute."""
    if force:
        return None
    try:
        return rec.resolve_mono_path(model_dir, model_type)
    except (FileNotFoundError, ValueError):
        # FileNotFoundError: nothing of the requested type. ValueError: ambiguous
        # (several types, none requested). Either way there is nothing safe to
        # reuse — recompute rather than guess.
        return None


def _reusable_stereo(
    model_dir: Path, board: str, model_type: Optional[str], force: bool
) -> Optional[Tuple[Path, "rec.StereoRecord"]]:
    """``(path, record)`` of an existing stereo model to reuse, or None to compute.

    Reuse requires the requested model_type present AND the stored board_type to
    match: the stereo model dir is shared across board types (unlike mono dirs,
    which embed the board in their path), so a stale other-board record must not
    satisfy a detect run.
    """
    if force:
        return None
    try:
        path = rec.resolve_stereo_path(model_dir, model_type)
    except (FileNotFoundError, ValueError):
        # Nothing of the requested type, or ambiguous with none requested —
        # recompute rather than guess which record to reuse.
        return None
    existing = rec.load_stereo(path)
    if existing.board_type != board:
        print(
            f"[calibration] existing stereo model at {path} is "
            f"board_type={existing.board_type!r} but {board!r} was requested — recomputing"
        )
        return None
    return path, existing


def _mono_model_type(args, cfg: Dict[str, Any], board: str) -> Optional[str]:
    """Requested mono record type: --model-type, else the board's configured
    model_type. None lets the resolver pick the single record present (and raise
    on ambiguity rather than guess)."""
    return getattr(args, "model_type", None) or (cfg.get(board) or {}).get("model_type")


def _stereo_model_type(args, cfg: Dict[str, Any], board: str) -> Optional[str]:
    """Requested stereo record type: --model-type, else the board's configured
    model_type when it is a stereo-capable type (the mono-only 'polynomial' /
    'scale_factor' must not be forwarded to a stereo load). None lets the
    resolver pick the single record present (ambiguity raises)."""
    mt = _mono_model_type(args, cfg, board)
    return mt if mt in rec.STEREO_MODEL_TYPES else None


def detect_mono_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    scfg = _settings_cfg(source)
    camera = int(args.camera if args.camera is not None else scfg.get("camera") or 1)
    image_format = _image_format_cli(args, scfg, source)
    n_views = _n_views_cli(args, scfg, source)
    # `is not None`, not `or`: start_index 0 is a legal value for zero-indexed sets
    start_index = int(scfg.get("start_index") if scfg.get("start_index") is not None else 1)
    datum_index = _resolve_datum_index(scfg)
    dm = DistortionModel(args.distortion or scfg.get("distortion_model", "standard"))
    model_type = args.model_type or scfg.get(board, {}).get("model_type", "pinhole")
    fix_aspect = bool(scfg.get("fix_aspect_ratio", True))
    fix_k2 = bool(scfg.get(board, {}).get("fix_k2", False))
    use_ro = bool(scfg.get("use_release_object", False))
    clicks = _load_clicks(args.world_frame or scfg.get("world_frame", "default"))
    frame_grid = _load_grid(scfg.get("world_frame_grid"))

    model_dir = rec.mono_model_dir_for_source(source, camera, board)

    # Reuse: the model lives WITH the calibration source images, so cases sharing
    # the same calibration input read it from this shared folder instead of recomputing.
    existing_path = _existing_mono_path(
        model_dir, model_type, getattr(args, "force", False)
    )
    if existing_path is not None:
        existing = rec.load_mono(existing_path)
        print(
            f"[calibration] {board} cam{camera} reusing existing model "
            f"({_model_rms_str(existing.camera_model)}) -> {existing_path} "
            f"(--force to recompute)"
        )
        return existing_path

    fig_dir = (
        None if getattr(args, "no_figures", False) else model_dir.parent / "figures"
    )

    # Resolve dt before the expensive detection pass — a missing dt must fail now.
    gen_dt = _generate_dt_cli(args, source)
    params = _board_params(scfg, board, overrides=_geometry_overrides(args))
    detector = _build_detector(board, params)
    images = _load_views(
        _cam_dir(scfg, source, camera, config.camera_count),
        image_format,
        n_views,
        start_index,
        camera=camera,
        **_loader_kwargs(scfg, image_format),
    )

    # Detection sidecar (parity with the GUI): reuse stored detections when the params match,
    # else detect fresh and persist — so a sidecar written here is reused by the GUI and
    # vice-versa, and a re-run skips re-detection.
    det_key = joint_det_key(
        board, n_views, image_format, infer_image_type(image_format), [camera], params
    )
    side = None if getattr(args, "force", False) else try_load_inputs(model_dir)
    cached = (side.detections or {}).get(camera) if side else None
    cache_hit = side is not None and side.det_key == det_key and bool(cached)
    dets = list(cached) if cache_hit else [detector.detect(im) for im in images]
    clicks_payload = clicks or (side.coords if side else None)
    origin_mm = (
        clicks_payload.get("origin_mm") if isinstance(clicks_payload, dict) else None
    )

    calr = Calibrator(
        detector=detector,
        board_type=board,
        model_type=model_type,
        distortion_model=dm,
        fix_aspect_ratio=fix_aspect,
        fix_k2=fix_k2,
        use_release_object=use_ro,
    )
    record = calr.run_mono(
        images,
        camera=camera,
        clicks=clicks_payload,
        datum_index=datum_index,
        spacing_mm=_spacing_mm(board, params),
        figure_dir=fig_dir,
        frame_grid=frame_grid,
        origin_mm=origin_mm,
        detections=dets,
    )
    record.board_meta["dt"] = gen_dt
    if image_format:
        record.board_meta["image_format"] = str(image_format)
    path = rec.save_mono(record, model_dir)
    isz = record.camera_model.image_size
    save_inputs(
        model_dir,
        path_type="mono",
        board_type=board,
        detections={camera: list(dets)},
        image_size_by_cam={camera: (int(isz[0]), int(isz[1]))},
        det_key=det_key,
        board_params=rec.geometry_meta(board, params),
        coords=clicks_payload,
    )
    figmsg = f" figures->{fig_dir}" if fig_dir else ""
    print(
        f"[calibration] {board} cam{camera} {_model_rms_str(record.camera_model)} "
        f"-> {path}{figmsg}"
    )
    return path


def detect_stereo_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    scfg = _settings_cfg(source)
    pair = args.camera_pair or scfg.get("camera_pair") or [1, 2]
    if isinstance(pair, str):
        pair = [int(x) for x in pair.split(",")]
    cam1, cam2 = int(pair[0]), int(pair[1])
    image_format = _image_format_cli(args, scfg, source)
    n_views = _n_views_cli(args, scfg, source)
    # `is not None`, not `or`: start_index 0 is a legal value for zero-indexed sets
    start_index = int(scfg.get("start_index") if scfg.get("start_index") is not None else 1)
    datum_index = _resolve_datum_index(scfg)
    dm = DistortionModel(args.distortion or scfg.get("distortion_model", "standard"))
    fix_aspect = bool(scfg.get("fix_aspect_ratio", True))
    fix_k2 = bool(scfg.get(board, {}).get("fix_k2", False))
    # Flat same-side stereo defaults to release-object intrinsics (DaVis-style); the GUI
    # route uses the StereoCalibrator default. An explicit settings value still overrides.
    use_ro = bool(scfg.get("use_release_object", True))
    clicks = _load_clicks(args.world_frame or scfg.get("world_frame", "default"))
    clicks2 = (
        _load_clicks(scfg.get("world_frame_cam2", "default"))
        if board == "dotboard"
        else None
    )
    frame_grid = _load_grid(scfg.get("world_frame_grid"))
    frame_grid2 = (
        _load_grid(scfg.get("world_frame_grid_cam2")) if board == "dotboard" else None
    )

    model_dir = rec.stereo_model_dir_for_source(source, cam1, cam2)

    # Reuse the shared model beside the calibration source images (see detect_mono).
    # This command's stereo fit is pinhole-only, so the reuse is too.
    reuse = _reusable_stereo(model_dir, board, "pinhole", getattr(args, "force", False))
    if reuse is not None:
        path, existing = reuse
        print(
            f"[calibration] stereo {board} cam{cam1}-cam{cam2} reusing existing model "
            f"(rms=({existing.model1.rms:.4f},{existing.model2.rms:.4f})px) -> "
            f"{path} (--force to recompute)"
        )
        return path

    fig_dir = (
        None if getattr(args, "no_figures", False) else model_dir.parent / "figures"
    )

    # Resolve dt before the expensive detection pass — a missing dt must fail now.
    gen_dt = _generate_dt_cli(args, source)
    params = _board_params(scfg, board, overrides=_geometry_overrides(args))
    detector = _build_detector(board, params)
    imgs1 = _load_views(
        _cam_dir(scfg, source, cam1, config.camera_count),
        image_format,
        n_views,
        start_index,
        camera=cam1,
        **_loader_kwargs(scfg, image_format),
    )
    imgs2 = _load_views(
        _cam_dir(scfg, source, cam2, config.camera_count),
        image_format,
        n_views,
        start_index,
        camera=cam2,
        **_loader_kwargs(scfg, image_format),
    )

    # Detection sidecar (parity with the GUI): reuse stored detections when params match,
    # else detect fresh and persist.
    det_key = joint_det_key(
        board,
        n_views,
        image_format,
        infer_image_type(image_format),
        [cam1, cam2],
        params,
    )
    side = None if getattr(args, "force", False) else try_load_inputs(model_dir)
    cache_hit = (
        side is not None
        and side.det_key == det_key
        and bool(side.detections)
        and cam1 in side.detections
        and cam2 in side.detections
    )
    if cache_hit:
        det1, det2 = side.detections[cam1], side.detections[cam2]
    else:
        det1 = [detector.detect(im) for im in imgs1]
        det2 = [detector.detect(im) for im in imgs2]
    clicks_payload = clicks or (side.coords if side else None)

    sc = StereoCalibrator(
        detector=detector,
        board_type=board,
        distortion_model=dm,
        fix_aspect_ratio=fix_aspect,
        fix_k2=fix_k2,
        use_release_object=use_ro,
    )
    record = sc.run_stereo(
        imgs1,
        imgs2,
        cam1=cam1,
        cam2=cam2,
        clicks=clicks_payload,
        clicks2=clicks2,
        datum_index=datum_index,
        spacing_mm=_spacing_mm(board, params),
        figure_dir=fig_dir,
        frame_grid=frame_grid,
        frame_grid2=frame_grid2,
        det1=det1,
        det2=det2,
    )
    record.board_meta["dt"] = gen_dt
    if image_format:
        record.board_meta["image_format"] = str(image_format)
    path = rec.save_stereo(record, model_dir)
    save_inputs(
        model_dir,
        path_type="stereo",
        board_type=board,
        detections={cam1: list(det1), cam2: list(det2)},
        image_size_by_cam={
            cam1: (int(record.model1.image_size[0]), int(record.model1.image_size[1])),
            cam2: (int(record.model2.image_size[0]), int(record.model2.image_size[1])),
        },
        det_key=det_key,
        board_params=rec.geometry_meta(board, params),
        coords=clicks_payload,
    )
    ang = np.degrees(np.arccos(np.clip((np.trace(record.R_stereo) - 1) / 2, -1, 1)))
    print(
        f"[calibration] stereo {board} cam{cam1}-cam{cam2} "
        f"rms=({record.model1.rms:.4f},{record.model2.rms:.4f})px "
        f"stereo_angle={ang:.3f}deg |T|={np.linalg.norm(record.T_stereo):.2f}mm -> {path}"
    )
    return path


def _global_grid_spec_from_cfg(gg: dict) -> GlobalGridSpec:
    """Build a ``GlobalGridSpec`` from the ``calibration.global_grid`` config block (dotboard).

    The datum clicks (origin/+X/+Y pixels) and the per-view anchor correspondences are the
    same record the GUI persists; this is the headless reader. ChArUco needs no spec (corner
    ids give the global grid), so this is dotboard-only. Geometry is required: a missing datum
    click raises rather than guessing an origin.
    """
    dc = dict(gg.get("datum_clicks", {}) or {})
    for key in ("origin", "x_axis", "y_axis"):
        if dc.get(key) is None:
            raise SystemExit(
                f"detect-joint: calibration.global_grid.datum_clicks.{key} is unset — set the "
                f"origin/+X/+Y pixels on the datum view first"
            )
    datum_clicks = {
        "origin": np.asarray(dc["origin"], dtype=float),
        "x_axis": np.asarray(dc["x_axis"], dtype=float),
        "y_axis": np.asarray(dc["y_axis"], dtype=float),
        "origin_mm": np.asarray(dc.get("origin_mm", [0.0, 0.0]), dtype=float),
    }
    anchors = []
    for i, a in enumerate(gg.get("anchors", []) or []):
        try:
            camera, view = int(a["camera"]), int(a["view"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"detect-joint: calibration.global_grid.anchors[{i}] needs integer 'camera' "
                f"and 'view' ({exc})"
            )
        corr = []
        for j, c in enumerate(a.get("correspondences", []) or []):
            where = f"calibration.global_grid.anchors[{i}].correspondences[{j}]"
            if c.get("pixel") is None:
                raise SystemExit(f"detect-joint: {where} is missing 'pixel'")
            same = c.get("same_as")
            if same is None:
                raise SystemExit(
                    f"detect-joint: {where} is missing 'same_as' ('origin' or [camera, view])"
                )
            if same != "origin":
                if not (isinstance(same, (list, tuple)) and len(same) == 2):
                    raise SystemExit(
                        f"detect-joint: {where}.same_as must be 'origin' or [camera, view], "
                        f"got {same!r}"
                    )
                same = (int(same[0]), int(same[1]))
            ref = c.get("ref_pixel")
            # A cross-view link must name the SAME physical dot in the reference view, else the
            # global index it inherits is wrong — so ref_pixel is mandatory for a view link.
            if same != "origin" and ref is None:
                raise SystemExit(
                    f"detect-joint: {where}.same_as is view {same} but 'ref_pixel' is missing "
                    f"(click the same physical dot in that view)"
                )
            corr.append(
                Correspondence(
                    pixel=np.asarray(c["pixel"], dtype=float),
                    same_as=same,
                    ref_pixel=None if ref is None else np.asarray(ref, dtype=float),
                )
            )
        if not corr:
            raise SystemExit(
                f"detect-joint: calibration.global_grid.anchors[{i}] has no correspondences"
            )
        anchors.append(Anchor(camera=camera, view=view, correspondences=corr))
    # Optional coarse rig arrangement: {camera: [dx, dy]} global-index direction this camera's
    # view extends, used only to break the lattice-symmetry tie for a camera's first view when
    # the overlap is a single collinear strip. Absent/zero is fine (the resolver then needs a
    # non-collinear pair or a prior).
    camera_extends = {}
    for k, v in (gg.get("camera_extends") or {}).items():
        try:
            camera_extends[int(k)] = (float(v[0]), float(v[1]))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise SystemExit(
                f"detect-joint: calibration.global_grid.camera_extends[{k!r}] must be "
                f"[dx, dy] numbers ({exc})"
            )
    return GlobalGridSpec(
        datum_camera=int(gg.get("datum_camera", 1)),
        datum_view=int(gg.get("datum_view", 0)),
        datum_clicks=datum_clicks,
        anchors=anchors,
        camera_extends=camera_extends,
    )


def detect_joint_command(args) -> "Path | List[Path]":
    """Joint multi-camera shared-board calibration — the DaVis-equivalent solve.

    Reads the global-grid spec from the sidecar ``inputs.mat`` (dotboard, written by the GUI
    wizard) or derives it from corner ids (ChArUco), detects every view of every camera, and
    resolves one global dot index across all of them. Then, by ``model_type``:

    - ``pinhole`` (default): the joint solve — per-camera intrinsics + ONE released board + all
      poses in a single shared frame — written as one unified ``JointRecord`` at the rig-level
      ``joint_<board>/model`` dir. The DaVis pinhole only (standard distortion, fx==fy, k3=0).
    - ``polynomial``: a per-camera single-plane polynomial on the datum view, fitted against the
      SHARED global frame (no released board, no bundle), written as per-camera mono records.

    Geometry (spacing, datum clicks, anchors) is required, never defaulted.
    """
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "dotboard")
    if board not in ("dotboard", "charuco"):
        raise SystemExit(f"detect-joint: board must be dotboard|charuco, got {board!r}")
    source = _source(cfg, args.source)
    scfg = _settings_cfg(source)
    # Clicked coords (datum + anchors + camera_extends + cameras) come from the sidecar
    # inputs.mat the GUI wizard writes — not config. Same dict shape, so the gg.get(...) reads
    # below and _global_grid_spec_from_cfg are unchanged. ChArUco needs no clicks (corner ids
    # give the grid); a missing dotboard datum is caught by _global_grid_spec_from_cfg.
    side = try_load_inputs(rec.joint_model_dir_for_source(source, board))
    gg = dict(side.coords) if (side and side.coords) else {}

    cams_arg = (
        [int(x) for x in str(args.cameras).split(",") if x.strip()]
        if args.cameras
        else None
    )
    cameras = cams_arg or [int(c) for c in (gg.get("cameras") or [])]
    if not cameras:
        raise SystemExit(
            "detect-joint: set the rig cameras in the joint wizard (or --cameras 1,2,3)"
        )
    datum_camera = int(gg.get("datum_camera", cameras[0]))
    datum_view = int(gg.get("datum_view", 0))
    model_type = args.model_type or scfg.get(board, {}).get("model_type", "pinhole")
    if model_type not in ("pinhole", "polynomial"):
        raise SystemExit(
            f"detect-joint: model_type must be pinhole|polynomial, got {model_type!r}"
        )
    board_release = args.board_release or gg.get("board_release", "full3d")
    if board_release not in ("full3d", "z_only", "none"):
        raise SystemExit(
            f"detect-joint: board_release must be full3d|z_only|none, got {board_release!r}"
        )

    n_views = _n_views_cli(args, scfg, source)
    if n_views < 1:
        raise SystemExit("detect-joint: n_views must be >= 1")
    # `is not None`, not `or`: start_index 0 is a legal value for zero-indexed sets
    start_index = int(scfg.get("start_index") if scfg.get("start_index") is not None else 1)
    image_format = _image_format_cli(args, scfg, source)
    # Distortion / fixed-aspect bind the pinhole bundle only; the polynomial map has no K/dist.
    dm = DistortionModel(args.distortion or scfg.get("distortion_model", "standard"))
    fix_aspect = bool(scfg.get("fix_aspect_ratio", True))
    if model_type == "pinhole":
        if dm != DistortionModel.STANDARD:
            raise SystemExit(
                f"detect-joint: the pinhole joint solve is the DaVis pinhole only "
                f"(distortion_model: standard); got {dm.value}"
            )
        if not fix_aspect:
            raise SystemExit(
                "detect-joint: the pinhole joint solve requires fix_aspect_ratio: true (fx==fy)"
            )
    # Geometry source order: CLI args > the source's existing sidecar (the GUI wizard wrote it
    # next to the clicks this command already reads) > config. So a config without geometry still
    # solves a previously-set-up joint rig.
    params = _board_params(
        scfg,
        board,
        overrides=_geometry_overrides(args),
        sidecar=(side.board_params if side else None),
    )
    spacing = _spacing_mm(board, params)
    # Resolve dt before the expensive detection pass — a missing dt must fail now.
    gen_dt = _generate_dt_cli(args, source)

    # Detect every view of every camera; tag each detection with the known spacing and record
    # the real image size (the joint solve seeds the principal point from it).
    detections: Dict[int, List[DetectionResult]] = {}
    image_size_by_cam: Dict[int, Tuple[int, int]] = {}
    for cam in cameras:
        detector = _build_detector(board, params)
        images = _load_views(
            _cam_dir(scfg, source, cam, config.camera_count),
            image_format,
            n_views,
            start_index,
            camera=cam,
            **_loader_kwargs(scfg, image_format),
        )
        if not images:
            raise SystemExit(f"detect-joint: no images loaded for cam{cam}")
        dets: List[DetectionResult] = []
        for img in images:
            d = detector.detect(img)
            d.spacing_mm = spacing
            dets.append(d)
        detections[cam] = dets
        h, w = images[0].shape[:2]
        image_size_by_cam[cam] = (int(w), int(h))
        nok = sum(1 for d in dets if d.success)
        print(f"[calibration] joint cam{cam}: {nok}/{len(dets)} views detected")

    # Individual failed views are tolerated — they are dropped (never anchored on dotboard, skipped
    # on ChArUco) and the solve uses the views that detected. Only a camera with NO successful view
    # is fatal (almost always a wrong path/format); report the dropped frames so they are not silent.
    failed = [
        (cam, v)
        for cam in cameras
        for v, d in enumerate(detections[cam])
        if not d.success
    ]
    if failed:
        print(
            f"[calibration] joint: dropping {len(failed)} undetected view(s): {failed}"
        )
    blank = [cam for cam in cameras if not any(d.success for d in detections[cam])]
    if blank:
        raise SystemExit(
            f"detect-joint: camera(s) {blank} detected no calibration target in any image — "
            f"check the image path, format and board parameters"
        )

    # Persist the detections into the sidecar (merging, so saved coords are untouched) so a
    # later GUI/CLI run can reuse them. det_key is computed from the same fields the GUI uses
    # (source-independent), so the GUI's reuse guard matches a CLI-written sidecar — real parity.
    try:
        save_inputs(
            rec.joint_model_dir_for_source(source, board),
            path_type="joint",
            board_type=board,
            detections=detections,
            image_size_by_cam=image_size_by_cam,
            det_key=joint_det_key(
                board,
                n_views,
                image_format,
                infer_image_type(image_format),
                cameras,
                params,
            ),
            board_params=rec.geometry_meta(board, params),
        )
    except (OSError, ValueError):
        pass

    spec = None if board == "charuco" else _global_grid_spec_from_cfg(gg)

    # origin_mm is the world (x, y) offset of the origin dot; the solve consumes it
    # (resolve_global_grid uses only the origin/+X/+Y clicks). ChArUco has no datum clicks, so
    # its origin is the corner-id frame origin at (0, 0).
    if board == "dotboard":
        origin_mm = tuple(
            float(x)
            for x in (gg.get("datum_clicks", {}) or {}).get("origin_mm", [0.0, 0.0])
        )
    else:
        origin_mm = (0.0, 0.0)

    # The resolve -> solve -> save core is the driver shared with the GUI routes (CLI/GUI
    # parity); only image loading above is CLI-specific.
    res = run_joint_from_spec(
        detections,
        image_size_by_cam,
        source=source,
        board=board,
        model_type=model_type,
        spacing_mm=spacing,
        dt=gen_dt,
        datum_camera=datum_camera,
        datum_view=datum_view,
        board_release=board_release,
        origin_mm=origin_mm,
        spec=spec,
        cameras=cameras,
        distortion_model=dm,
        fix_aspect_ratio=fix_aspect,
        n_views=n_views,
        board_params=params,
    )

    if res.model_type == "polynomial":
        for cam, p in zip(res.cameras, res.paths):
            print(
                f"[calibration] joint-polynomial {board} cam{cam} "
                f"rms={res.per_camera_rms[cam]:.4f}mm -> {p}"
            )
        return res.paths
    path = res.paths[0]
    rms_str = ", ".join(
        f"cam{c}={res.per_camera_rms.get(c, float('nan')):.4f}" for c in res.cameras
    )
    print(
        f"[calibration] joint {board} cams={res.cameras} rms={res.rms_px:.4f}px "
        f"({rms_str}) release={board_release} converged={res.converged} -> {path}"
    )
    return path


def apply_calibration_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    # Light-sheet plane for a mono apply: flags > 0.0. Zero is the calibration-board
    # plane — a geometric identity, not a guessed default. No config source.
    z = float(args.z_world) if args.z_world is not None else 0.0
    tx = float(args.tilt_x) if args.tilt_x is not None else 0.0
    ty = float(args.tilt_y) if args.tilt_y is not None else 0.0
    vector_glob = vector_glob_from_format(config.vector_format)
    type_name = args.type_name or "instantaneous"
    scfg = _settings_cfg(source)
    # Which record to load when several model types coexist in the model dir.
    # None -> the single one present (ambiguity raises before any unit runs).
    model_type = _mono_model_type(args, scfg, board)

    # Explicit dirs (flags only) -> one ad-hoc unit. Otherwise --all-paths derives
    # every base_path x camera from config (mirrors the GUI).
    explicit = None
    if args.uncalibrated_dir and args.calibrated_dir:
        explicit = {"uncal": args.uncalibrated_dir, "out": args.calibrated_dir}
    if explicit is None and not args.all_paths:
        raise SystemExit(
            "apply-calibration: pass --uncalibrated-dir + --calibrated-dir, or --all-paths "
            "to derive every base_path x camera from config"
        )

    camera = args.camera if args.camera is not None else scfg.get("camera") or 1
    units = runio.plan_apply_units(
        config,
        source,
        board,
        False,
        type_name,
        camera=camera,
        explicit=explicit,
        model_type=model_type,
    )
    total = 0
    for u in units:
        # dt: --dt override > model-stamped. No config source and no silent 1.0
        # fallback — velocity scales with dt, so an unresolved dt raises. Per
        # unit, so a multi-camera rig uses each camera's own stamped dt.
        dt = runio.resolve_dt(args.dt, u["record"].board_meta.get("dt"))
        written = runio.calibrate_mono_run(
            u["record"], u["uncal"], u["out"], dt, z, tx, ty, vector_glob=vector_glob
        )
        total += len(written)
        print(
            f"[calibration] applied {board} {u['label']} -> {len(written)} frame(s) in {u['out']}"
        )
    return total


def apply_stereo_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    scfg = _settings_cfg(source)
    pair = args.camera_pair or scfg.get("camera_pair") or [1, 2]
    if isinstance(pair, str):
        pair = [int(x) for x in pair.split(",")]
    cam1, cam2 = int(pair[0]), int(pair[1])
    type_name = args.type_name or "instantaneous"

    # Explicit dirs (flags only) -> one ad-hoc unit; otherwise --all-paths derives every base path.
    explicit = None
    if not args.all_paths:
        if not (args.uncalibrated_dir_cam1 and args.uncalibrated_dir_cam2):
            raise SystemExit(
                "apply-stereo: pass --uncalibrated-dir-cam1 + --uncalibrated-dir-cam2, or --all-paths"
            )
        if not args.calibrated_dir:
            raise SystemExit(
                "apply-stereo: pass --calibrated-dir, or --all-paths"
            )
        explicit = {
            "uncal1": args.uncalibrated_dir_cam1,
            "uncal2": args.uncalibrated_dir_cam2,
            "out": args.calibrated_dir,
        }

    units = runio.plan_apply_units(
        config,
        source,
        board,
        True,
        type_name,
        camera_pair=[cam1, cam2],
        explicit=explicit,
        model_type=_stereo_model_type(args, scfg, board),
    )
    # Laser-sheet plane: flags > the record's saved self-cal > 0.0 (an empty self_cal
    # already means sheet-at-datum). All stereo units share one record; no config source.
    rec0 = units[0]["record"]
    z = float(args.z_world) if args.z_world is not None else rec0.sc_z_offset
    tx = float(args.tilt_x) if args.tilt_x is not None else rec0.sc_tilt_x
    ty = float(args.tilt_y) if args.tilt_y is not None else rec0.sc_tilt_y
    vector_glob = vector_glob_from_format(config.vector_format)
    # --interpolator > the source's settings sidecar knob (defaulted there).
    # argparse choices only constrain the flag — the sidecar value needs the
    # same check so a typo'd rig.interpolator fails here, not mid-reconstruction.
    interpolator = args.interpolator or scfg.get("interpolator") or "lanczos"
    if interpolator not in ("linear", "cubic", "lanczos"):
        raise SystemExit(
            f"calibration: rig.interpolator must be linear|cubic|lanczos, "
            f"got {interpolator!r} (fix it in {cs.settings_path(source)})"
        )
    total = 0
    for u in units:
        # dt: --dt override > model-stamped (stereo records stamp dt at generate).
        dt = runio.resolve_dt(args.dt, u["record"].board_meta.get("dt"))
        written = runio.reconstruct_stereo_run(
            u["record"],
            u["uncal1"],
            u["uncal2"],
            u["out"],
            dt,
            None,
            z,
            tx,
            ty,
            vector_glob=vector_glob,
            interpolator=interpolator,
        )
        total += len(written)
        print(
            f"[calibration] stereo 3C {u['label']} -> {len(written)} frame(s) in {u['out']}"
        )
    return total


def self_calibrate_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    scfg = _settings_cfg(source)
    pair = args.camera_pair or scfg.get("camera_pair") or [1, 2]
    if isinstance(pair, str):
        pair = [int(x) for x in pair.split(",")]
    cam1, cam2 = int(pair[0]), int(pair[1])
    base_idx = int(args.base_path_idx if args.base_path_idx is not None else 0)
    n_images = int(
        args.n_images
        if args.n_images is not None
        else scfg.get("self_cal_n_images") or 20
    )
    window_size = int(args.window_size if args.window_size is not None else 64)
    overlap = float(args.overlap if args.overlap is not None else 50.0)
    apply_filters = not getattr(args, "no_filters", False)

    model_dir = rec.stereo_model_dir_for_source(source, cam1, cam2)
    record = rec.load_stereo(
        model_dir, model_type=_stereo_model_type(args, scfg, board)
    )
    figdir = (
        None
        if getattr(args, "no_figures", False)
        else model_dir.parent / "figures" / "self_cal"
    )

    imgs1, imgs2 = c2sc.load_particle_pairs(
        config, base_idx, cam1, cam2, n_images, apply_filters
    )
    result = c2sc.run(
        record,
        imgs1,
        imgs2,
        window_size=window_size,
        overlap=overlap,
        figure_dir=figdir,
    )
    c2sc.rebake_record(record, result.z_offset, result.tilt_x, result.tilt_y)
    record.self_cal = c2sc.baked_block(
        result, n_images=len(imgs1), window_size=window_size, overlap=overlap
    )
    path = rec.save_stereo(record, model_dir)
    print(
        f"[calibration] self-cal cam{cam1}-cam{cam2} baked into extrinsics "
        f"z={result.z_offset:.4f}mm "
        f"tilt_x={np.degrees(result.tilt_x):+.4f}deg "
        f"tilt_y={np.degrees(result.tilt_y):+.4f}deg "
        f"rms={result.final_rms_disparity:.4f}px converged={result.converged} "
        f"({result.n_iterations} iters) -> {path}"
    )
    return path


def scale_factor_command(args):
    """Build a scale-factor mono model (uniform pixel->mm) from CLI params.

    No board, no detection — the user names the origin pixel, the axis directions,
    px/mm and dt. One frame is loaded only to stamp the image size and draw the
    proof figure; an explicit ``--image-size W H`` supplies the size instead, so the
    model can be built with no images on disk at all (the proof figure is skipped —
    it needs a frame to draw on).
    """
    config = get_config()
    cfg = _cfg2(config)
    source = _source(cfg, args.source)
    scfg = _settings_cfg(source)
    camera = int(args.camera if args.camera is not None else scfg.get("camera") or 1)
    # `is not None`, not `or`: start_index 0 is a legal value for zero-indexed sets
    start_index = int(scfg.get("start_index") if scfg.get("start_index") is not None else 1)
    frame = int(args.frame if args.frame is not None else start_index)
    px_per_mm = float(args.px_per_mm)
    dt = _generate_dt_cli(args, source)
    origin = [float(args.origin[0]), float(args.origin[1])]
    origin_mm = (
        (float(args.origin_mm[0]), float(args.origin_mm[1]))
        if getattr(args, "origin_mm", None) is not None
        else (0.0, 0.0)
    )

    if getattr(args, "image_size", None) is not None:
        image = None
        image_size = (int(args.image_size[0]), int(args.image_size[1]))
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError(f"--image-size must be positive, got {image_size}")
    else:
        # image_format is only needed when a frame is actually loaded — the
        # --image-size path builds the model with no images on disk at all.
        image_format = _image_format_cli(args, scfg, source)
        image = _load_one(
            _cam_dir(scfg, source, camera, config.camera_count),
            image_format,
            frame,
            camera=camera,
            **_loader_kwargs(scfg, image_format),
        )
        h, w = np.asarray(image).shape[:2]
        image_size = (int(w), int(h))
    record = build_scale_factor_record(
        camera=camera,
        origin_px=origin,
        px_per_mm=px_per_mm,
        image_size=image_size,
        dt=dt,
        x_dir=args.x_dir,
        y_dir=args.y_dir,
        swap_axes=bool(args.swap),
        frame_idx=frame,
        origin_mm=origin_mm,
    )
    model_dir = rec.mono_model_dir_for_source(source, camera, "scale_factor")
    path = rec.save_mono(record, model_dir)

    fig_dir = (
        None
        if (getattr(args, "no_figures", False) or image is None)
        else model_dir.parent / "figures"
    )
    if fig_dir is not None:
        from pivtools_gui.calibration import figures as c2figs

        sf = record.camera_model
        # Draw at the PICKED origin (world_frame) — the model's own origin_px is the
        # world-zero pixel, off the picked point when origin_mm != 0.
        c2figs.write_scale_factor_figure(
            fig_dir,
            image=image,
            origin_px=record.world_frame.origin_px,
            col_sign=sf.col_sign,
            row_sign=sf.row_sign,
            swap_axes=bool(sf.swap_axes),
            mm_per_pixel=sf.mm_per_pixel,
            dt=dt,
            origin_mm=origin_mm,
        )
    figmsg = f" figures->{fig_dir}" if fig_dir else ""
    mm_msg = f"=({origin_mm[0]:g},{origin_mm[1]:g})mm" if any(origin_mm) else ""
    print(
        f"[calibration] scale_factor cam{camera} "
        f"origin=({origin[0]:.1f},{origin[1]:.1f})px{mm_msg} {px_per_mm:.4f}px/mm "
        f"+X={args.x_dir} +Y={args.y_dir}{' swap' if args.swap else ''} dt={dt:g}s "
        f"-> {path}{figmsg}"
    )
    return path


def global_frame_command(args):
    """Bake the multi-camera global frame into each mono model (headless analogue of the
    GUI's "Compute + Save Global Frame").

    Reads the datum + overlap-pair chain from the source's settings sidecar
    (``global_coordinates`` block — the same one the GUI persists), computes
    per-camera shifts via the shared chain math, and writes ``world_offset_mm``
    into each camera's model record so apply emits the shared rig frame.
    Re-run after recalibrating any camera (regen clears the offset).
    """
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    scfg = _settings_cfg(source)
    settings = cs.try_load_settings(source) or cs.default_settings()
    gc = settings.get("global_coordinates") or {}
    datum_camera = int(gc.get("datum_camera") or 1)
    datum_pixel = gc.get("datum_pixel")
    datum_physical = gc.get("datum_physical") or [0.0, 0.0]
    overlap_pairs = gc.get("overlap_pairs") or []
    if not datum_pixel:
        raise SystemExit(
            "[calibration] no datum_pixel in the source's calibration settings "
            f"({cs.settings_path(source)}, global_coordinates block) — set the "
            "datum + overlap pairs in the GUI first"
        )

    cams = {datum_camera}
    for p in overlap_pairs:
        cams.add(int(p["camera_a"]))
        cams.add(int(p["camera_b"]))
    model_type = _mono_model_type(args, scfg, board)
    dirs = {cam: rec.mono_model_dir_for_source(source, cam, board) for cam in cams}
    records = {cam: rec.load_mono(d, model_type=model_type) for cam, d in dirs.items()}
    shifts = gc2.compute_camera_shifts(
        records, datum_camera, datum_pixel, datum_physical, overlap_pairs
    )

    for cam, (sx, sy) in shifts.items():
        r = records[cam]
        r.world_frame.world_offset_mm = np.array(
            [float(sx), float(sy)], dtype=np.float64
        )
        saved = rec.save_mono(r, dirs[cam])
        print(
            f"[calibration] global-frame {board} cam{cam} "
            f"offset=({sx:+.2f}, {sy:+.2f}) mm -> {saved}"
        )
    return shifts


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------


def _add_geometry_args(p):
    """ChArUco geometry overrides (all default None -> only override when given). These let a
    headless-from-scratch ChArUco run supply geometry without a config block; the GUI passes the
    same values per-request. ``--square-size`` is in metres (ChArUco native unit), matching config.
    (CLI detection is ChArUco-only; dotboard/stepped geometry args went with those paths.)
    """
    p.add_argument("--squares-h", type=int, default=None)
    p.add_argument("--squares-v", type=int, default=None)
    p.add_argument("--square-size", type=float, default=None, help="metres")
    p.add_argument("--marker-ratio", type=float, default=None)
    p.add_argument("--aruco-dict", default=None)
    p.add_argument("--min-corners", type=int, default=None)


def _geometry_overrides(args) -> dict:
    """Non-None ChArUco geometry args as a config-key-shaped override dict for ``_board_params``."""
    m = {
        "squares_h": getattr(args, "squares_h", None),
        "squares_v": getattr(args, "squares_v", None),
        "square_size": getattr(args, "square_size", None),
        "marker_ratio": getattr(args, "marker_ratio", None),
        "aruco_dict": getattr(args, "aruco_dict", None),
        "min_corners": getattr(args, "min_corners", None),
    }
    return {k: v for k, v in m.items() if v is not None}


def _add_common(p):
    p.add_argument(
        "--source", default=None, help="calibration source dir (model saved here)"
    )
    _add_geometry_args(p)
    p.add_argument("--board", default="charuco", choices=["charuco"])
    p.add_argument(
        "--dt",
        type=float,
        default=None,
        help="frame dt stamped into the model (else rig.dt from settings.yaml)",
    )
    p.add_argument("--image-format", default=None)
    p.add_argument("--n-views", type=int, default=None)
    p.add_argument(
        "--distortion", default=None, choices=["standard", "rational", "tilted"]
    )
    p.add_argument(
        "--model-type",
        default=None,
        choices=["pinhole", "polynomial"],
        help="mono camera model (polynomial is single-plane / planar only)",
    )
    p.add_argument(
        "--world-frame", default=None, help="'default' or path to clicks JSON"
    )
    p.add_argument(
        "--no-figures",
        action="store_true",
        help="skip the archival proof figures saved beside the model",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="recompute even if a model already exists in the source folder",
    )


def register_calibration_subparsers(subparsers):
    p = subparsers.add_parser(
        "init-settings",
        help="calibration: write a settings.yaml template into the source's calibration folder",
    )
    p.add_argument(
        "--source",
        default=None,
        help="calibration source dir (else calibration.source / calibration_sources[source_idx])",
    )
    p.set_defaults(func=init_settings_command)

    # CLI detection is ChArUco-only: ChArUco needs no interactive datum/anchor clicks, so it
    # solves headless. Dotboard and stepped calibration require the GUI (world-frame / fiducial /
    # level picking) and are intentionally not exposed here. The CLI can still APPLY any saved
    # model (apply / apply-stereo) regardless of how it was calibrated.
    p = subparsers.add_parser("detect-charuco", help="calibration: detect mono charuco")
    _add_common(p)
    p.add_argument("--camera", type=int, default=None)
    p.set_defaults(func=detect_mono_command, board="charuco")

    p = subparsers.add_parser(
        "detect-stereo", help="calibration: detect stereo charuco pair"
    )
    _add_common(p)
    p.add_argument("--camera-pair", default=None, help="'1,2'")
    p.set_defaults(func=detect_stereo_command, board="charuco")

    p = subparsers.add_parser(
        "detect-joint",
        help="calibration: joint multi-camera shared-board solve (DaVis-equivalent) -> JointRecord",
    )
    p.add_argument(
        "--source", default=None, help="calibration source dir (model saved here)"
    )
    p.add_argument("--board", default="charuco", choices=["charuco"])
    p.add_argument(
        "--cameras",
        default=None,
        help="comma-separated cameras, e.g. '1,2,3' (else calibration.global_grid.cameras)",
    )
    p.add_argument("--image-format", default=None)
    p.add_argument("--n-views", type=int, default=None)
    p.add_argument(
        "--dt",
        type=float,
        default=None,
        help="frame dt stamped into the model (else rig.dt from settings.yaml)",
    )
    p.add_argument(
        "--model-type",
        default=None,
        choices=["pinhole", "polynomial"],
        help="pinhole joint bundle (shared released board) or per-camera polynomial "
        "in the shared global frame",
    )
    p.add_argument(
        "--distortion",
        default=None,
        choices=["standard"],
        help="the pinhole joint solve is the DaVis pinhole only (standard)",
    )
    p.add_argument(
        "--board-release",
        default=None,
        choices=["full3d", "z_only", "none"],
        help="released-board DOF for pinhole (default full3d, matches DaVis)",
    )
    _add_geometry_args(p)
    p.set_defaults(func=detect_joint_command)

    p = subparsers.add_parser(
        "apply-calibration", help="calibration: apply mono model to vectors"
    )
    p.add_argument("--source", default=None)
    p.add_argument(
        "--board",
        default=None,
        choices=["charuco", "dotboard", "stepped", "scale_factor"],
    )
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument(
        "--z-world",
        type=float,
        default=None,
        help="light-sheet plane Z in mm (default 0.0 = the calibration-board plane)",
    )
    p.add_argument("--tilt-x", type=float, default=None, help="sheet tilt (rad)")
    p.add_argument("--tilt-y", type=float, default=None, help="sheet tilt (rad)")
    p.add_argument("--uncalibrated-dir", default=None)
    p.add_argument("--calibrated-dir", default=None)
    p.add_argument(
        "--all-paths",
        action="store_true",
        help="derive every base_path x camera from config (like the GUI) instead of explicit dirs",
    )
    p.add_argument(
        "--type-name",
        default=None,
        help="PIV result type for --all-paths (default instantaneous)",
    )
    p.add_argument(
        "--model-type",
        default=None,
        choices=["pinhole", "polynomial", "polynomial3d", "scale_factor"],
        help="which model record to load when several types exist in the model dir",
    )
    p.set_defaults(func=apply_calibration_command)

    p = subparsers.add_parser(
        "apply-stereo", help="calibration: 3C stereo reconstruction"
    )
    p.add_argument("--source", default=None)
    p.add_argument("--board", default=None, choices=["charuco", "dotboard", "stepped"])
    p.add_argument("--camera-pair", default=None, help="'1,2'")
    p.add_argument("--dt", type=float, default=None)
    p.add_argument(
        "--z-world",
        type=float,
        default=None,
        help="light-sheet plane Z in mm (default: the record's self-cal, else 0.0)",
    )
    p.add_argument("--tilt-x", type=float, default=None, help="sheet tilt (rad)")
    p.add_argument("--tilt-y", type=float, default=None, help="sheet tilt (rad)")
    p.add_argument(
        "--interpolator",
        default=None,
        choices=["cubic", "lanczos"],
        help="stereo cam2 resample kernel (default: lanczos / settings.yaml)",
    )
    p.add_argument("--uncalibrated-dir-cam1", default=None)
    p.add_argument("--uncalibrated-dir-cam2", default=None)
    p.add_argument("--calibrated-dir", default=None)
    p.add_argument(
        "--all-paths",
        action="store_true",
        help="derive every base_path from config (like the GUI) instead of explicit dirs",
    )
    p.add_argument(
        "--type-name",
        default=None,
        help="PIV result type for --all-paths (default instantaneous)",
    )
    p.add_argument(
        "--model-type",
        default=None,
        choices=["pinhole", "polynomial3d"],
        help="which stereo record to load when several types exist in the model dir",
    )
    p.set_defaults(func=apply_stereo_command)

    p = subparsers.add_parser(
        "self-calibrate",
        help="calibration: stereo self-calibration (Wieneke disparity minimisation)",
    )
    p.add_argument(
        "--source", default=None, help="calibration source dir (stereo model location)"
    )
    p.add_argument("--board", default=None, choices=["charuco", "dotboard"])
    p.add_argument("--camera-pair", default=None, help="'1,2'")
    p.add_argument(
        "--base-path-idx",
        type=int,
        default=None,
        help="index into config.base_paths for the PIV particle images",
    )
    p.add_argument(
        "--n-images", type=int, default=None, help="number of frame pairs to correlate"
    )
    p.add_argument(
        "--window-size", type=int, default=None, help="correlation window (px)"
    )
    p.add_argument("--overlap", type=float, default=None, help="window overlap (%%)")
    p.add_argument(
        "--no-filters",
        action="store_true",
        help="skip the PIV pre-filters on the particle frames",
    )
    p.add_argument(
        "--no-figures", action="store_true", help="skip writing diagnostic figures"
    )
    p.add_argument(
        "--model-type",
        default=None,
        choices=["pinhole", "polynomial3d"],
        help="which stereo record to load when several types exist in the model dir",
    )
    p.set_defaults(func=self_calibrate_command)

    p = subparsers.add_parser(
        "scale-factor",
        help="calibration: build a scale-factor mono model (uniform pixel->mm)",
    )
    p.add_argument(
        "--source", default=None, help="calibration source dir (model saved here)"
    )
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--image-format", default=None)
    p.add_argument(
        "--frame", type=int, default=None, help="1-based frame for image size + figure"
    )
    p.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="image size in px; skips loading a frame entirely (no proof figure) — "
        "for sources with no calibration images on disk",
    )
    p.add_argument(
        "--px-per-mm", type=float, required=True, help="pixels per millimetre"
    )
    p.add_argument("--dt", type=float, default=None, help="time between frames (s)")
    p.add_argument(
        "--origin",
        type=float,
        nargs=2,
        required=True,
        metavar=("X", "Y"),
        help="world-origin pixel (image-down)",
    )
    p.add_argument(
        "--origin-mm",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="world (X, Y) mm assigned to the origin pixel (default 0 0)",
    )
    p.add_argument(
        "--x-dir", default="right", choices=["right", "left"], help="+X direction"
    )
    p.add_argument("--y-dir", default="up", choices=["up", "down"], help="+Y direction")
    p.add_argument(
        "--swap", action="store_true", help="+X follows the vertical pixel axis"
    )
    p.add_argument("--no-figures", action="store_true", help="skip the proof figure")
    p.set_defaults(func=scale_factor_command)

    p = subparsers.add_parser(
        "global-frame",
        help="calibration: bake the multi-camera global frame (datum+overlap from the source's settings sidecar) into each model",
    )
    p.add_argument(
        "--source", default=None, help="calibration source dir (models live here)"
    )
    p.add_argument(
        "--board", default=None, choices=["charuco", "dotboard", "scale_factor"]
    )
    p.add_argument(
        "--model-type",
        default=None,
        choices=["pinhole", "polynomial", "polynomial3d", "scale_factor"],
        help="which model record to load when several types exist in the model dir",
    )
    p.set_defaults(func=global_frame_command)
