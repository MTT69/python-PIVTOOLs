"""pivtools_cli.calibration2_cli — CLI for the unified calibration2 package.

Fully YAML-driven (config block ``calibration2:``) with argparse overrides, mirroring
the v1 command pattern but writing models into the calibration SOURCE folder. New
subcommands coexist with v1 (suffixed ``2``): ``detect-planar2``, ``detect-charuco2``,
``detect-stereo2``, ``apply-calibration2``, ``apply-stereo2``.

Example config:

    calibration2:
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
      cam_subfolders: {1: cam1, 2: cam2}   # optional, for stereo / multi-cam
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
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from pivtools_core.config import get_config
from pivtools_core.paths import vector_glob_from_format
from pivtools_gui.calibration2.camera_model import DistortionModel
from pivtools_gui.calibration2.detection.charuco import CharucoBoardDetector, CharucoParams
from pivtools_gui.calibration2.detection.dotboard import DotboardDetector, DotboardParams
from pivtools_gui.calibration2.detection.stepped import SteppedDetector, SteppedParams
from pivtools_gui.calibration2.pipeline import Calibrator, build_scale_factor_record
from pivtools_gui.calibration2.stepped_calibrate import (
    calibrate_stepped_mono,
    calibrate_stepped_stereo,
)
from pivtools_gui.calibration2.stereo_model import StereoCalibrator
from pivtools_gui.calibration2 import global_coords as gc2
from pivtools_gui.calibration2 import record as rec
from pivtools_gui.calibration2 import runio
from pivtools_gui.calibration2 import self_cal as c2sc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cfg2(config) -> dict:
    return config.calibration2


def _source(cfg: dict, override=None) -> Path:
    """Resolve the calibration SOURCE directory (where the model is saved and read).

    This is the folder that holds the calibration images, so the model lives WITH
    them and is shared by every PIV run that references this calibration — regardless
    of the run's ``base_path``. Precedence:

        explicit override  >  calibration2.source  >  calibration.calibration_sources[idx]

    The final fallback ties v2 to the SAME calibration source the rest of pivtools
    loads images from, so no second config is needed for the shared-model behaviour.
    """
    if override:
        return Path(override)
    if cfg.get("source"):
        return Path(cfg["source"])
    return get_config().get_calibration_source(int(cfg.get("source_idx", 0)))


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
    params_from: Callable[[dict], object]   # merged config dict -> Params dataclass
    detector_cls: type                      # detector constructed from the Params
    spacing_mm: Callable[[object], float]   # physical feature spacing in mm


def _charuco_params_from(d: dict) -> CharucoParams:
    return CharucoParams(
        squares_h=int(d.get("squares_h", 10)),
        squares_v=int(d.get("squares_v", 7)),
        square_size_m=float(d.get("square_size", 0.030)),
        marker_ratio=float(d.get("marker_ratio", 0.5)),
        aruco_dict=str(d.get("aruco_dict", "DICT_4X4_1000")),
        min_corners=int(d.get("min_corners", 6)),
    )


def _dotboard_params_from(d: dict) -> DotboardParams:
    return DotboardParams(
        dot_spacing_mm=float(d.get("dot_spacing_mm", 15.0)),
        k_neighbors=int(d.get("k_neighbors", 9)),
    )


def _stepped_params_from(d: dict) -> SteppedParams:
    lo = d.get("level_offset_mm")
    return SteppedParams(
        dot_spacing_mm=float(d.get("dot_spacing_mm", 15.0)),
        step_height_mm=float(d.get("step_height_mm", 3.0)),
        board_thickness_mm=float(d.get("board_thickness_mm", 14.8)),
        level_offset_mm=None if lo is None else float(lo),
    )


BOARD_REGISTRY: Dict[str, BoardSpec] = {
    "charuco": BoardSpec(_charuco_params_from, CharucoBoardDetector, lambda p: p.square_size_mm),
    "dotboard": BoardSpec(_dotboard_params_from, DotboardDetector, lambda p: p.dot_spacing_mm),
    "stepped": BoardSpec(_stepped_params_from, SteppedDetector, lambda p: p.dot_spacing_mm),
}


def _board_spec(board: str) -> BoardSpec:
    spec = BOARD_REGISTRY.get(board)
    if spec is None:
        raise ValueError(f"unknown board '{board}' (expected {'|'.join(BOARD_REGISTRY)})")
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


def _board_params(cfg: dict, board: str, overrides: Optional[dict] = None):
    """Build board params from the ``calibration2.<board>`` config block.

    A GUI request may pass ``overrides`` (e.g. squares/spacing typed in the panel);
    these take precedence per-call but are NOT persisted, so detection stays a pure
    function of its inputs and the shared config is never silently mutated.
    """
    merged = dict(cfg.get(board, {}) or {})
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


def _parse_stepped_cam(d: dict):
    """One camera's stepped inputs: (fiducials, clicked_level, pose_levels).

    These are the interactive picks the GUI gathers (origin/+X/+Y snap, the datum-face
    selector, and the per-pose click-to-label). Headless they arrive as data — one entry
    per loaded pose in ``pose_levels`` (the level-A label of that pose), the datum's entry
    cross-checked against the fiducial-derived label by the calibrator.
    """
    fiducials = _parse_fiducials(d["fiducials"])
    clicked_level = str(d["clicked_level"])
    pose_levels = [str(x) for x in d["pose_levels"]]
    return fiducials, clicked_level, pose_levels


def _cam_dir(source: Path, cfg: dict, camera: int, subfolder: Optional[str] = None) -> Path:
    """Resolve the per-camera image directory.

    ``subfolder`` (when not None) is an explicit per-request override — used by the
    GUI, which knows the camera subfolder for the active source without relying on
    the server-side ``cam_subfolders`` map being in sync. ``""`` means no subfolder.
    """
    if subfolder is not None:
        return Path(source) / subfolder if subfolder else Path(source)
    subs = cfg.get("cam_subfolders", {}) or {}
    sub = subs.get(camera, subs.get(str(camera), ""))
    return source / sub if sub else source


def _load_one(cam_dir: Path, image_format: str, frame_number: int) -> np.ndarray:
    """Load a single calibration frame by its 1-based image index (grayscale)."""
    path = cam_dir / (image_format % int(frame_number))
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"calibration image not found: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _load_views(cam_dir: Path, image_format: str, n_views: int, start_index: int) -> List[np.ndarray]:
    return [_load_one(cam_dir, image_format, start_index + k) for k in range(n_views)]


def _count_views(cam_dir: Path, image_format: str, start_index: int, max_views: int = 100000) -> int:
    """Count consecutive calibration frames present from ``start_index`` upward."""
    n = 0
    while n < max_views and (cam_dir / (image_format % (start_index + n))).exists():
        n += 1
    return n


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def detect_mono2_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    camera = int(args.camera if args.camera is not None else cfg.get("camera", 1))
    image_format = args.image_format or cfg.get("image_format", "calib%05d.png")
    n_views = int(args.n_views or cfg.get("n_views", 10))
    start_index = int(cfg.get("start_index", 1))
    datum_index = int(cfg.get("datum_index", 0))
    dm = DistortionModel(args.distortion or cfg.get("distortion_model", "standard"))
    model_type = args.model_type or cfg.get(board, {}).get("model_type", "pinhole")
    fix_aspect = bool(cfg.get("fix_aspect_ratio", True))
    use_ro = bool(cfg.get("use_release_object", False))
    clicks = _load_clicks(args.world_frame or cfg.get("world_frame", "default"))
    frame_grid = _load_grid(cfg.get("world_frame_grid"))

    model_dir = rec.mono_model_dir_for_source(source, camera, board)

    # Reuse: the model lives WITH the calibration source images, so cases sharing
    # the same calibration input read it from this shared folder instead of recomputing.
    if (model_dir / "model.mat").exists() and not getattr(args, "force", False):
        existing = rec.load_mono(model_dir)
        print(f"[calibration2] {board} cam{camera} reusing existing model "
              f"({_model_rms_str(existing.camera_model)}) -> {model_dir / 'model.mat'} "
              f"(--force to recompute)")
        return model_dir / "model.mat"

    fig_dir = None if getattr(args, "no_figures", False) else model_dir.parent / "figures"

    params = _board_params(cfg, board)
    detector = _build_detector(board, params)
    images = _load_views(_cam_dir(source, cfg, camera), image_format, n_views, start_index)

    calr = Calibrator(detector=detector, board_type=board, model_type=model_type,
                      distortion_model=dm, fix_aspect_ratio=fix_aspect,
                      use_release_object=use_ro)
    record = calr.run_mono(images, camera=camera, clicks=clicks,
                           datum_index=datum_index, spacing_mm=_spacing_mm(board, params),
                           figure_dir=fig_dir, frame_grid=frame_grid)
    path = rec.save_mono(record, model_dir)
    figmsg = f" figures->{fig_dir}" if fig_dir else ""
    print(f"[calibration2] {board} cam{camera} {_model_rms_str(record.camera_model)} "
          f"-> {path}{figmsg}")
    return path


def detect_stereo2_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    pair = args.camera_pair or cfg.get("camera_pair", [1, 2])
    if isinstance(pair, str):
        pair = [int(x) for x in pair.split(",")]
    cam1, cam2 = int(pair[0]), int(pair[1])
    image_format = args.image_format or cfg.get("image_format", "calib%05d.png")
    n_views = int(args.n_views or cfg.get("n_views", 10))
    start_index = int(cfg.get("start_index", 1))
    datum_index = int(cfg.get("datum_index", 0))
    dm = DistortionModel(args.distortion or cfg.get("distortion_model", "standard"))
    fix_aspect = bool(cfg.get("fix_aspect_ratio", True))
    use_ro = bool(cfg.get("use_release_object", False))
    clicks = _load_clicks(args.world_frame or cfg.get("world_frame", "default"))
    clicks2 = _load_clicks(cfg.get("world_frame_cam2", "default")) if board == "dotboard" else None
    frame_grid = _load_grid(cfg.get("world_frame_grid"))
    frame_grid2 = _load_grid(cfg.get("world_frame_grid_cam2")) if board == "dotboard" else None

    model_dir = rec.stereo_model_dir_for_source(source, cam1, cam2)

    # Reuse the shared model beside the calibration source images (see detect_mono2).
    if (model_dir / "stereo_model.mat").exists() and not getattr(args, "force", False):
        existing = rec.load_stereo(model_dir)
        print(f"[calibration2] stereo {board} cam{cam1}-cam{cam2} reusing existing model "
              f"(rms=({existing.model1.rms:.4f},{existing.model2.rms:.4f})px) -> "
              f"{model_dir / 'stereo_model.mat'} (--force to recompute)")
        return model_dir / "stereo_model.mat"

    fig_dir = None if getattr(args, "no_figures", False) else model_dir.parent / "figures"

    params = _board_params(cfg, board)
    detector = _build_detector(board, params)
    imgs1 = _load_views(_cam_dir(source, cfg, cam1), image_format, n_views, start_index)
    imgs2 = _load_views(_cam_dir(source, cfg, cam2), image_format, n_views, start_index)

    sc = StereoCalibrator(detector=detector, board_type=board, distortion_model=dm,
                          fix_aspect_ratio=fix_aspect, use_release_object=use_ro)
    record = sc.run_stereo(imgs1, imgs2, cam1=cam1, cam2=cam2, clicks=clicks,
                           clicks2=clicks2, datum_index=datum_index,
                           spacing_mm=_spacing_mm(board, params), figure_dir=fig_dir,
                           frame_grid=frame_grid, frame_grid2=frame_grid2)
    path = rec.save_stereo(record, model_dir)
    ang = np.degrees(np.arccos(np.clip((np.trace(record.R_stereo) - 1) / 2, -1, 1)))
    print(f"[calibration2] stereo {board} cam{cam1}-cam{cam2} "
          f"rms=({record.model1.rms:.4f},{record.model2.rms:.4f})px "
          f"stereo_angle={ang:.3f}deg |T|={np.linalg.norm(record.T_stereo):.2f}mm -> {path}")
    return path


def _stepped_spec_source(args, scfg: dict, key_attr: str = "stepped_spec"):
    """Resolve the stepped spec input: ``--stepped-spec PATH`` > inline config block.

    The stepped board cannot route through the on-demand single-detector path the other
    boards use (its fit needs per-pose levels + fiducial clicks), so the headless inputs
    arrive either as a JSON file or inline under ``calibration2.stepped``. Raises loudly
    when neither is present — never guesses fiducials.
    """
    spec = getattr(args, key_attr, None)
    if spec:
        return _read_json_maybe(spec)
    if scfg.get("fiducials") and scfg.get("clicked_level") and scfg.get("pose_levels"):
        return scfg
    return None


def detect_stepped_mono2_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = "stepped"
    source = _source(cfg, args.source)
    camera = int(args.camera if args.camera is not None else cfg.get("camera", 1))
    image_format = args.image_format or cfg.get("image_format", "calib%05d.png")
    start_index = int(cfg.get("start_index", 1))
    datum_index = int(cfg.get("datum_index", 0))
    dm = DistortionModel(args.distortion or cfg.get("distortion_model", "standard"))
    scfg = dict(cfg.get("stepped", {}) or {})
    model_type = args.model_type or scfg.get("model_type", "pinhole")

    model_dir = rec.mono_model_dir_for_source(source, camera, board)
    if (model_dir / "model.mat").exists() and not getattr(args, "force", False):
        existing = rec.load_mono(model_dir)
        print(f"[calibration2] stepped cam{camera} reusing existing model "
              f"({_model_rms_str(existing.camera_model)}) -> {model_dir / 'model.mat'} "
              f"(--force to recompute)")
        return model_dir / "model.mat"

    spec = _stepped_spec_source(args, scfg)
    if spec is None:
        raise SystemExit(
            "detect-stepped2: provide --stepped-spec PATH (JSON with fiducials, "
            "clicked_level, pose_levels) or set those under calibration2.stepped")
    fiducials, clicked_level, pose_levels = _parse_stepped_cam(spec)
    # n_views follows the spec's pose count unless explicitly overridden — the spec is
    # the source of truth for how many poses were labelled.
    n_views = int(args.n_views) if args.n_views else len(pose_levels)

    params = _board_params(cfg, board)        # SteppedParams
    detector = _build_detector(board, params)
    images = _load_views(_cam_dir(source, cfg, camera), image_format, n_views, start_index)
    h, w = np.asarray(images[datum_index]).shape[:2]
    image_size = (int(w), int(h))
    detections = [detector.detect(im) for im in images]

    fig_dir = None if getattr(args, "no_figures", False) else model_dir.parent / "figures"
    record = calibrate_stepped_mono(
        detections=detections, fiducials=fiducials, clicked_level=clicked_level,
        pose_levels=pose_levels, board=params.board(), image_size=image_size,
        camera=camera, datum_index=datum_index, distortion_model=dm,
        model_type=model_type, images=images, figure_dir=fig_dir,
    )
    path = rec.save_mono(record, model_dir)
    figmsg = f" figures->{fig_dir}" if fig_dir else ""
    print(f"[calibration2] stepped cam{camera} [{model_type}] "
          f"{_model_rms_str(record.camera_model)} -> {path}{figmsg}")
    return path


def detect_stepped_stereo2_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = "stepped"
    source = _source(cfg, args.source)
    pair = args.camera_pair or cfg.get("camera_pair", [1, 2])
    if isinstance(pair, str):
        pair = [int(x) for x in pair.split(",")]
    cam1, cam2 = int(pair[0]), int(pair[1])
    image_format = args.image_format or cfg.get("image_format", "calib%05d.png")
    start_index = int(cfg.get("start_index", 1))
    datum_index = int(cfg.get("datum_index", 0))
    dm = DistortionModel(args.distortion or cfg.get("distortion_model", "standard"))
    scfg = dict(cfg.get("stepped", {}) or {})
    model_type = args.model_type or scfg.get("model_type", "pinhole")

    model_dir = rec.stereo_model_dir_for_source(source, cam1, cam2)
    if (model_dir / "stereo_model.mat").exists() and not getattr(args, "force", False):
        existing = rec.load_stereo(model_dir)
        print(f"[calibration2] stepped stereo cam{cam1}-cam{cam2} reusing existing model "
              f"(cam{cam1} {_model_rms_str(existing.model1)}; "
              f"cam{cam2} {_model_rms_str(existing.model2)}) -> "
              f"{model_dir / 'stereo_model.mat'} (--force to recompute)")
        return model_dir / "stereo_model.mat"

    spec = _stepped_spec_source(args, scfg)
    if spec is None or "cam1" not in spec or "cam2" not in spec:
        raise SystemExit(
            "detect-stepped-stereo2: provide --stepped-spec PATH (JSON with cam1/cam2 "
            "blocks, each {fiducials, clicked_level, pose_levels}, optional stereo_config)")
    fid1, lvl1, poses1 = _parse_stepped_cam(spec["cam1"])
    fid2, lvl2, poses2 = _parse_stepped_cam(spec["cam2"])
    stereo_config = str(spec.get("stereo_config", "auto"))
    n_views = int(args.n_views) if args.n_views else len(poses1)

    params = _board_params(cfg, board)
    detector = _build_detector(board, params)
    imgs1 = _load_views(_cam_dir(source, cfg, cam1), image_format, n_views, start_index)
    imgs2 = _load_views(_cam_dir(source, cfg, cam2), image_format, n_views, start_index)
    sz1 = tuple(int(v) for v in np.asarray(imgs1[datum_index]).shape[:2][::-1])
    sz2 = tuple(int(v) for v in np.asarray(imgs2[datum_index]).shape[:2][::-1])
    det1 = [detector.detect(im) for im in imgs1]
    det2 = [detector.detect(im) for im in imgs2]

    fig_dir = None if getattr(args, "no_figures", False) else model_dir.parent / "figures"
    record = calibrate_stepped_stereo(
        detections1=det1, detections2=det2, fiducials1=fid1, fiducials2=fid2,
        clicked_level1=lvl1, clicked_level2=lvl2, pose_levels1=poses1, pose_levels2=poses2,
        board=params.board(), image_size1=sz1, image_size2=sz2, cam1=cam1, cam2=cam2,
        datum_index=datum_index, stereo_config=stereo_config, distortion_model=dm,
        model_type=model_type, images1=imgs1, images2=imgs2, figure_dir=fig_dir,
    )
    path = rec.save_stereo(record, model_dir)
    meta = record.board_meta
    if model_type == "polynomial3d":
        geo = f"config={meta.get('stereo_config')} baseline/angle n/a (polynomial)"
    else:
        geo = (f"config={meta.get('stereo_config')} "
               f"stereo_angle={meta.get('relative_angle_deg', float('nan')):.3f}deg "
               f"|T|={meta.get('baseline_mm', float('nan')):.2f}mm")
    print(f"[calibration2] stepped stereo cam{cam1}-cam{cam2} [{model_type}] "
          f"cam{cam1} {_model_rms_str(record.model1)}; "
          f"cam{cam2} {_model_rms_str(record.model2)} {geo} -> {path}")
    return path


def apply_calibration2_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    z = float(cfg.get("z_world", 0.0))
    tx = float(cfg.get("tilt_x", 0.0))
    ty = float(cfg.get("tilt_y", 0.0))
    vector_glob = vector_glob_from_format(config.vector_format)
    type_name = args.type_name or "instantaneous"

    # Explicit dirs (--args, then single config keys) -> one ad-hoc unit. Otherwise
    # --all-paths derives every base_path x camera from config (mirrors the GUI).
    explicit = None
    if args.uncalibrated_dir and args.calibrated_dir:
        explicit = {"uncal": args.uncalibrated_dir, "out": args.calibrated_dir}
    elif not args.all_paths and cfg.get("uncalibrated_dir") and cfg.get("calibrated_dir"):
        explicit = {"uncal": cfg["uncalibrated_dir"], "out": cfg["calibrated_dir"]}
    if explicit is None and not args.all_paths:
        raise SystemExit(
            "apply-calibration2: pass --uncalibrated-dir + --calibrated-dir, or --all-paths "
            "to derive every base_path x camera from config")

    camera = args.camera if args.camera is not None else cfg.get("camera", 1)
    units = runio.plan_apply_units(config, source, board, False, type_name,
                                   camera=camera, explicit=explicit)
    total = 0
    for u in units:
        # dt: --dt override > model-stamped (scale-factor records carry it) > config. No
        # silent 1.0 fallback — velocity scales with dt, so an unresolved dt raises. Per
        # unit, so a multi-camera rig uses each camera's own stamped dt.
        dt = runio.resolve_dt(args.dt, u["record"].board_meta.get("dt"), cfg.get("dt"))
        written = runio.calibrate_mono_run(
            u["record"], u["uncal"], u["out"], dt, z, tx, ty, vector_glob=vector_glob)
        total += len(written)
        print(f"[calibration2] applied {board} {u['label']} -> {len(written)} frame(s) in {u['out']}")
    return total


def apply_stereo2_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    pair = args.camera_pair or cfg.get("camera_pair", [1, 2])
    if isinstance(pair, str):
        pair = [int(x) for x in pair.split(",")]
    cam1, cam2 = int(pair[0]), int(pair[1])
    type_name = args.type_name or "instantaneous"

    # Explicit config dirs -> one ad-hoc unit; otherwise --all-paths derives every base path.
    explicit = None
    if not args.all_paths:
        if not (cfg.get("uncalibrated_dir_cam1") and cfg.get("uncalibrated_dir_cam2")):
            raise SystemExit(
                "apply-stereo2: set uncalibrated_dir_cam1/uncalibrated_dir_cam2 in config, or --all-paths")
        out = args.calibrated_dir or cfg.get("stereo_calibrated_dir")
        if not out:
            raise SystemExit(
                "apply-stereo2: set --calibrated-dir or stereo_calibrated_dir in config, or --all-paths")
        explicit = {"uncal1": cfg["uncalibrated_dir_cam1"], "uncal2": cfg["uncalibrated_dir_cam2"],
                    "out": out}

    units = runio.plan_apply_units(config, source, board, True, type_name,
                                   camera_pair=[cam1, cam2], explicit=explicit)
    # The laser sheet defaults from the saved self-cal unless config overrides (all stereo
    # units share one record). Stereo records do not stamp dt; --dt > config, no 1.0 fallback.
    rec0 = units[0]["record"]
    z = float(cfg["z_world"]) if "z_world" in cfg else rec0.sc_z_offset
    tx = float(cfg["tilt_x"]) if "tilt_x" in cfg else rec0.sc_tilt_x
    ty = float(cfg["tilt_y"]) if "tilt_y" in cfg else rec0.sc_tilt_y
    vector_glob = vector_glob_from_format(config.vector_format)
    total = 0
    for u in units:
        dt = runio.resolve_dt(args.dt, None, cfg.get("dt"))
        written = runio.reconstruct_stereo_run(
            u["record"], u["uncal1"], u["uncal2"], u["out"], dt, None, z, tx, ty,
            vector_glob=vector_glob)
        total += len(written)
        print(f"[calibration2] stereo 3C {u['label']} -> {len(written)} frame(s) in {u['out']}")
    return total


def self_calibrate2_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    pair = args.camera_pair or cfg.get("camera_pair", [1, 2])
    if isinstance(pair, str):
        pair = [int(x) for x in pair.split(",")]
    cam1, cam2 = int(pair[0]), int(pair[1])
    base_idx = int(args.base_path_idx if args.base_path_idx is not None else 0)
    n_images = int(args.n_images if args.n_images is not None else cfg.get("self_cal_n_images", 20))
    window_size = int(args.window_size if args.window_size is not None else 64)
    overlap = float(args.overlap if args.overlap is not None else 50.0)
    apply_filters = not getattr(args, "no_filters", False)

    model_dir = rec.stereo_model_dir_for_source(source, cam1, cam2)
    record = rec.load_stereo(model_dir)
    figdir = None if getattr(args, "no_figures", False) else model_dir.parent / "figures" / "self_cal"

    imgs1, imgs2 = c2sc.load_particle_pairs(config, base_idx, cam1, cam2, n_images, apply_filters)
    result = c2sc.run(record, imgs1, imgs2, window_size=window_size, overlap=overlap,
                      figure_dir=figdir)
    record.self_cal = c2sc.result_to_block(
        result, n_images=len(imgs1), window_size=window_size, overlap=overlap)
    path = rec.save_stereo(record, model_dir)
    print(f"[calibration2] self-cal cam{cam1}-cam{cam2} "
          f"z={result.z_offset:.4f}mm "
          f"tilt_x={np.degrees(result.tilt_x):+.4f}deg "
          f"tilt_y={np.degrees(result.tilt_y):+.4f}deg "
          f"rms={result.final_rms_disparity:.4f}px converged={result.converged} "
          f"({result.n_iterations} iters) -> {path}")
    return path


def scale_factor2_command(args):
    """Build a scale-factor mono model (uniform pixel->mm) from CLI params.

    No board, no detection — the user names the origin pixel, the axis directions,
    px/mm and dt. One frame is loaded only to stamp the image size and draw the
    proof figure.
    """
    config = get_config()
    cfg = _cfg2(config)
    source = _source(cfg, args.source)
    camera = int(args.camera if args.camera is not None else cfg.get("camera", 1))
    image_format = args.image_format or cfg.get("image_format", "calib%05d.png")
    start_index = int(cfg.get("start_index", 1))
    frame = int(args.frame if args.frame is not None else start_index)
    px_per_mm = float(args.px_per_mm)
    dt = float(args.dt if args.dt is not None else cfg.get("scale_factor", {}).get("dt", 1.0))
    origin = [float(args.origin[0]), float(args.origin[1])]

    image = _load_one(_cam_dir(source, cfg, camera), image_format, frame)
    h, w = np.asarray(image).shape[:2]
    record = build_scale_factor_record(
        camera=camera, origin_px=origin, px_per_mm=px_per_mm,
        image_size=(int(w), int(h)), dt=dt,
        x_dir=args.x_dir, y_dir=args.y_dir, swap_axes=bool(args.swap), frame_idx=frame)
    model_dir = rec.mono_model_dir_for_source(source, camera, "scale_factor")
    path = rec.save_mono(record, model_dir)

    fig_dir = None if getattr(args, "no_figures", False) else model_dir.parent / "figures"
    if fig_dir is not None:
        from pivtools_gui.calibration2 import figures as c2figs
        sf = record.camera_model
        c2figs.write_scale_factor_figure(
            fig_dir, image=image, origin_px=sf.origin_px, col_sign=sf.col_sign,
            row_sign=sf.row_sign, swap_axes=bool(sf.swap_axes),
            mm_per_pixel=sf.mm_per_pixel, dt=dt)
    figmsg = f" figures->{fig_dir}" if fig_dir else ""
    print(f"[calibration2] scale_factor cam{camera} "
          f"origin=({origin[0]:.1f},{origin[1]:.1f})px {px_per_mm:.4f}px/mm "
          f"+X={args.x_dir} +Y={args.y_dir}{' swap' if args.swap else ''} dt={dt:g}s "
          f"-> {path}{figmsg}")
    return path


def global_frame2_command(args):
    """Bake the multi-camera global frame into each mono model (headless analogue of the
    GUI's "Compute + Save Global Frame").

    Reads the datum + overlap-pair chain from ``config.calibration.global_coordinates``
    (the same block the GUI persists), computes per-camera shifts via the shared chain
    math, and writes ``world_offset_mm`` into each camera's ``model.mat`` so apply emits
    the shared rig frame. Re-run after recalibrating any camera (regen clears the offset).
    """
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    gc = config.global_coordinates_config
    datum_camera = int(gc.get("datum_camera", 1))
    datum_pixel = gc.get("datum_pixel")
    datum_physical = gc.get("datum_physical", [0.0, 0.0])
    overlap_pairs = gc.get("overlap_pairs", []) or []
    if not datum_pixel:
        raise SystemExit(
            "[calibration2] no datum_pixel in config.calibration.global_coordinates — "
            "set the datum + overlap pairs in the GUI (or config) first")

    cams = {datum_camera}
    for p in overlap_pairs:
        cams.add(int(p["camera_a"]))
        cams.add(int(p["camera_b"]))
    dirs = {cam: rec.mono_model_dir_for_source(source, cam, board) for cam in cams}
    records = {cam: rec.load_mono(d) for cam, d in dirs.items()}
    shifts = gc2.compute_camera_shifts(
        records, datum_camera, datum_pixel, datum_physical, overlap_pairs)

    for cam, (sx, sy) in shifts.items():
        r = records[cam]
        r.world_frame.world_offset_mm = np.array([float(sx), float(sy)], dtype=np.float64)
        rec.save_mono(r, dirs[cam])
        print(f"[calibration2] global-frame {board} cam{cam} "
              f"offset=({sx:+.2f}, {sy:+.2f}) mm -> {dirs[cam] / 'model.mat'}")
    return shifts


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------

def _add_common(p):
    p.add_argument("--source", default=None, help="calibration source dir (model saved here)")
    p.add_argument("--board", default=None, choices=["charuco", "dotboard"])
    p.add_argument("--image-format", default=None)
    p.add_argument("--n-views", type=int, default=None)
    p.add_argument("--distortion", default=None, choices=["standard", "rational", "tilted"])
    p.add_argument("--model-type", default=None, choices=["pinhole", "polynomial"],
                   help="mono camera model (polynomial is single-plane / planar only)")
    p.add_argument("--world-frame", default=None, help="'default' or path to clicks JSON")
    p.add_argument("--no-figures", action="store_true",
                   help="skip the archival proof figures saved beside the model")
    p.add_argument("--force", action="store_true",
                   help="recompute even if a model already exists in the source folder")


def register_calibration2_subparsers(subparsers):
    p = subparsers.add_parser("detect-planar2", help="calibration2: detect mono dotboard")
    _add_common(p)
    p.add_argument("--camera", type=int, default=None)
    p.set_defaults(func=detect_mono2_command, board="dotboard")

    p = subparsers.add_parser("detect-charuco2", help="calibration2: detect mono charuco")
    _add_common(p)
    p.add_argument("--camera", type=int, default=None)
    p.set_defaults(func=detect_mono2_command, board="charuco")

    p = subparsers.add_parser("detect-stereo2", help="calibration2: detect stereo pair")
    _add_common(p)
    p.add_argument("--camera-pair", default=None, help="'1,2'")
    p.set_defaults(func=detect_stereo2_command)

    # Stepped (dual-level) board — its own commands because the fit needs per-pose level
    # labels + fiducial clicks (the GUI gathers these interactively; headless they come
    # from --stepped-spec JSON or calibration2.stepped). model_type adds polynomial3d.
    p = subparsers.add_parser("detect-stepped2", help="calibration2: detect stepped mono")
    p.add_argument("--source", default=None, help="calibration source dir (model saved here)")
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--image-format", default=None)
    p.add_argument("--n-views", type=int, default=None,
                   help="poses to load (default: the spec's pose_levels count)")
    p.add_argument("--distortion", default=None, choices=["standard", "rational", "tilted"])
    p.add_argument("--model-type", default=None, choices=["pinhole", "polynomial3d"],
                   help="pinhole (multi-pose) or polynomial3d (single datum view, 3D cubic)")
    p.add_argument("--stepped-spec", default=None,
                   help="JSON with {fiducials:{origin,x_axis,y_axis}, clicked_level, pose_levels}")
    p.add_argument("--no-figures", action="store_true",
                   help="skip the archival proof figures saved beside the model")
    p.add_argument("--force", action="store_true",
                   help="recompute even if a model already exists in the source folder")
    p.set_defaults(func=detect_stepped_mono2_command)

    p = subparsers.add_parser("detect-stepped-stereo2", help="calibration2: detect stepped stereo pair")
    p.add_argument("--source", default=None, help="calibration source dir (model saved here)")
    p.add_argument("--camera-pair", default=None, help="'1,2'")
    p.add_argument("--image-format", default=None)
    p.add_argument("--n-views", type=int, default=None,
                   help="poses to load (default: the cam1 spec's pose_levels count)")
    p.add_argument("--distortion", default=None, choices=["standard", "rational", "tilted"])
    p.add_argument("--model-type", default=None, choices=["pinhole", "polynomial3d"],
                   help="pinhole (rig R/t) or polynomial3d (reconstruction-only, no rig geometry)")
    p.add_argument("--stepped-spec", default=None,
                   help="JSON with cam1/cam2 blocks {fiducials, clicked_level, pose_levels} + stereo_config")
    p.add_argument("--no-figures", action="store_true",
                   help="skip the archival proof figures saved beside the model")
    p.add_argument("--force", action="store_true",
                   help="recompute even if a model already exists in the source folder")
    p.set_defaults(func=detect_stepped_stereo2_command)

    p = subparsers.add_parser("apply-calibration2", help="calibration2: apply mono model to vectors")
    p.add_argument("--source", default=None)
    p.add_argument("--board", default=None, choices=["charuco", "dotboard", "stepped", "scale_factor"])
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--uncalibrated-dir", default=None)
    p.add_argument("--calibrated-dir", default=None)
    p.add_argument("--all-paths", action="store_true",
                   help="derive every base_path x camera from config (like the GUI) instead of explicit dirs")
    p.add_argument("--type-name", default=None, help="PIV result type for --all-paths (default instantaneous)")
    p.set_defaults(func=apply_calibration2_command)

    p = subparsers.add_parser("apply-stereo2", help="calibration2: 3C stereo reconstruction")
    p.add_argument("--source", default=None)
    p.add_argument("--board", default=None, choices=["charuco", "dotboard", "stepped"])
    p.add_argument("--camera-pair", default=None, help="'1,2'")
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--calibrated-dir", default=None)
    p.add_argument("--all-paths", action="store_true",
                   help="derive every base_path from config (like the GUI) instead of explicit dirs")
    p.add_argument("--type-name", default=None, help="PIV result type for --all-paths (default instantaneous)")
    p.set_defaults(func=apply_stereo2_command)

    p = subparsers.add_parser(
        "self-calibrate2",
        help="calibration2: stereo self-calibration (Wieneke disparity minimisation)")
    p.add_argument("--source", default=None, help="calibration source dir (stereo model location)")
    p.add_argument("--board", default=None, choices=["charuco", "dotboard"])
    p.add_argument("--camera-pair", default=None, help="'1,2'")
    p.add_argument("--base-path-idx", type=int, default=None,
                   help="index into config.base_paths for the PIV particle images")
    p.add_argument("--n-images", type=int, default=None, help="number of frame pairs to correlate")
    p.add_argument("--window-size", type=int, default=None, help="correlation window (px)")
    p.add_argument("--overlap", type=float, default=None, help="window overlap (%%)")
    p.add_argument("--no-filters", action="store_true",
                   help="skip the PIV pre-filters on the particle frames")
    p.add_argument("--no-figures", action="store_true", help="skip writing diagnostic figures")
    p.set_defaults(func=self_calibrate2_command)

    p = subparsers.add_parser(
        "scale-factor2",
        help="calibration2: build a scale-factor mono model (uniform pixel->mm)")
    p.add_argument("--source", default=None, help="calibration source dir (model saved here)")
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--image-format", default=None)
    p.add_argument("--frame", type=int, default=None, help="1-based frame for image size + figure")
    p.add_argument("--px-per-mm", type=float, required=True, help="pixels per millimetre")
    p.add_argument("--dt", type=float, default=None, help="time between frames (s)")
    p.add_argument("--origin", type=float, nargs=2, required=True, metavar=("X", "Y"),
                   help="world-origin pixel (image-down)")
    p.add_argument("--x-dir", default="right", choices=["right", "left"], help="+X direction")
    p.add_argument("--y-dir", default="up", choices=["up", "down"], help="+Y direction")
    p.add_argument("--swap", action="store_true", help="+X follows the vertical pixel axis")
    p.add_argument("--no-figures", action="store_true", help="skip the proof figure")
    p.set_defaults(func=scale_factor2_command)

    p = subparsers.add_parser(
        "global-frame2",
        help="calibration2: bake the multi-camera global frame (datum+overlap from config) into each model")
    p.add_argument("--source", default=None, help="calibration source dir (models live here)")
    p.add_argument("--board", default=None, choices=["charuco", "dotboard", "scale_factor"])
    p.set_defaults(func=global_frame2_command)
