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
from pivtools_gui.calibration2.pipeline import Calibrator
from pivtools_gui.calibration2.stereo_model import StereoCalibrator
from pivtools_gui.calibration2 import record as rec
from pivtools_gui.calibration2 import runio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cfg2(config) -> dict:
    return dict(config.data.get("calibration2", {}) or {})


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


BOARD_REGISTRY: Dict[str, BoardSpec] = {
    "charuco": BoardSpec(_charuco_params_from, CharucoBoardDetector, lambda p: p.square_size_mm),
    "dotboard": BoardSpec(_dotboard_params_from, DotboardDetector, lambda p: p.dot_spacing_mm),
}


def _board_spec(board: str) -> BoardSpec:
    spec = BOARD_REGISTRY.get(board)
    if spec is None:
        raise ValueError(f"unknown board '{board}' (expected {'|'.join(BOARD_REGISTRY)})")
    return spec


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
              f"(RMS={existing.camera_model.rms:.4f}px) -> {model_dir / 'model.mat'} "
              f"(--force to recompute)")
        return model_dir / "model.mat"

    fig_dir = None if getattr(args, "no_figures", False) else model_dir.parent / "figures"

    params = _board_params(cfg, board)
    detector = _build_detector(board, params)
    images = _load_views(_cam_dir(source, cfg, camera), image_format, n_views, start_index)

    calr = Calibrator(detector=detector, board_type=board, distortion_model=dm,
                      fix_aspect_ratio=fix_aspect, use_release_object=use_ro)
    record = calr.run_mono(images, camera=camera, clicks=clicks,
                           datum_index=datum_index, spacing_mm=_spacing_mm(board, params),
                           figure_dir=fig_dir, frame_grid=frame_grid)
    path = rec.save_mono(record, model_dir)
    figmsg = f" figures->{fig_dir}" if fig_dir else ""
    print(f"[calibration2] {board} cam{camera} RMS={record.camera_model.rms:.4f}px "
          f"fx={record.camera_model.K[0,0]:.1f} -> {path}{figmsg}")
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


def apply_calibration2_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    camera = int(args.camera if args.camera is not None else cfg.get("camera", 1))
    dt = float(args.dt if args.dt is not None else cfg.get("dt", 1.0))
    uncal = Path(args.uncalibrated_dir or cfg["uncalibrated_dir"])
    out = Path(args.calibrated_dir or cfg["calibrated_dir"])
    z = float(cfg.get("z_world", 0.0))
    tx = float(cfg.get("tilt_x", 0.0))
    ty = float(cfg.get("tilt_y", 0.0))

    model_dir = rec.mono_model_dir_for_source(source, camera, board)
    record = rec.load_mono(model_dir)
    written = runio.calibrate_mono_run(
        record, uncal, out, dt, z, tx, ty,
        vector_glob=vector_glob_from_format(config.vector_format))
    print(f"[calibration2] applied {board} cam{camera} to {len(written)} frame(s) -> {out}")
    return written


def apply_stereo2_command(args):
    config = get_config()
    cfg = _cfg2(config)
    board = args.board or cfg.get("active", "charuco")
    source = _source(cfg, args.source)
    pair = args.camera_pair or cfg.get("camera_pair", [1, 2])
    if isinstance(pair, str):
        pair = [int(x) for x in pair.split(",")]
    cam1, cam2 = int(pair[0]), int(pair[1])
    dt = float(args.dt if args.dt is not None else cfg.get("dt", 1.0))
    uncal1 = Path(cfg["uncalibrated_dir_cam1"])
    uncal2 = Path(cfg["uncalibrated_dir_cam2"])
    out = Path(args.calibrated_dir or cfg["stereo_calibrated_dir"])
    z = float(cfg.get("z_world", 0.0))
    tx = float(cfg.get("tilt_x", 0.0))
    ty = float(cfg.get("tilt_y", 0.0))

    model_dir = rec.stereo_model_dir_for_source(source, cam1, cam2)
    record = rec.load_stereo(model_dir)
    written = runio.reconstruct_stereo_run(
        record, uncal1, uncal2, out, dt, None, z, tx, ty,
        vector_glob=vector_glob_from_format(config.vector_format))
    print(f"[calibration2] stereo 3C cam{cam1}-cam{cam2} -> {len(written)} frame(s) in {out}")
    return written


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------

def _add_common(p):
    p.add_argument("--source", default=None, help="calibration source dir (model saved here)")
    p.add_argument("--board", default=None, choices=["charuco", "dotboard"])
    p.add_argument("--image-format", default=None)
    p.add_argument("--n-views", type=int, default=None)
    p.add_argument("--distortion", default=None, choices=["standard", "rational", "tilted"])
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

    p = subparsers.add_parser("apply-calibration2", help="calibration2: apply mono model to vectors")
    p.add_argument("--source", default=None)
    p.add_argument("--board", default=None, choices=["charuco", "dotboard"])
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--uncalibrated-dir", default=None)
    p.add_argument("--calibrated-dir", default=None)
    p.set_defaults(func=apply_calibration2_command)

    p = subparsers.add_parser("apply-stereo2", help="calibration2: 3C stereo reconstruction")
    p.add_argument("--source", default=None)
    p.add_argument("--board", default=None, choices=["charuco", "dotboard"])
    p.add_argument("--camera-pair", default=None, help="'1,2'")
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--calibrated-dir", default=None)
    p.set_defaults(func=apply_stereo2_command)
