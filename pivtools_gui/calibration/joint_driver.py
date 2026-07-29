"""calibration.joint_driver — the one resolve -> solve -> save path the CLI and GUI share.

``calibration_cli.detect_joint_command`` (headless) and the ``/calibration/joint/*`` Flask
routes (GUI) must produce byte-identical joint calibrations from the same inputs. The image
LOADING legitimately differs (argparse + the CLI loader vs request JSON + the app-wide reader),
but the CORE — resolve the global grid, run the joint/polynomial solve, assemble and save the
record — must not drift between the two front-ends (this codebase has been bitten by replay
drift before). That core lives here; both callers build ``detections_by_cam`` +
``image_size_by_cam`` their own way, then hand off to ``run_joint_from_spec``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import record as rec
from .camera_model import DistortionModel
from .detection.base import DetectionResult
from .global_grid import GlobalGridSpec, resolve_global_grid
from .joint import run_joint, run_joint_polynomial

ViewKey = Tuple[int, int]


@dataclass
class JointDriverResult:
    """What the shared driver produced — enough for both the CLI print and the route JSON."""

    board_type: str  # 'dotboard' | 'charuco'
    model_type: str  # 'pinhole' | 'polynomial'
    cameras: List[int]
    paths: List[Path]  # saved record path(s): one (pinhole) or per-camera (poly)
    per_camera_rms: Dict[
        int, float
    ]  # px for pinhole reprojection, in-plane mm for polynomial
    rms_units: str  # 'px' | 'mm' — per_camera_rms units (do not compare across)
    rms_px: float  # overall reprojection RMS (pinhole); NaN for polynomial
    converged: bool
    cross_camera_board_agreement_mm: (
        float  # 0 by construction for pinhole; NaN for polynomial
    )
    n_board_dots: int  # released board size (pinhole); 0 for polynomial
    info: Dict[str, object] = field(default_factory=dict)


def _check_some_detected(detections_by_cam: Dict[int, List[DetectionResult]]) -> None:
    """Raise only if a camera detected the target in NO image — that camera cannot be calibrated.

    Individual failed views are tolerated: they are never anchored (dotboard) / are skipped
    (ChArUco), so they contribute no observations and the solve uses the views that detected. A
    camera with zero successful views, on the other hand, is almost always a wrong path/format and
    must fail loudly rather than silently calibrate a subset of the rig.
    """
    blank = [
        cam
        for cam, dets in detections_by_cam.items()
        if not any(d.success for d in dets)
    ]
    if blank:
        raise ValueError(
            f"joint: camera(s) {blank} detected no calibration target in any image — check the "
            f"image path, format and board parameters"
        )


def run_joint_from_spec(
    detections_by_cam: Dict[int, List[DetectionResult]],
    image_size_by_cam: Dict[int, Tuple[int, int]],
    *,
    source: Path,
    board: str,
    model_type: str,
    spacing_mm: float,
    dt: float,
    datum_camera: int,
    datum_view: int,
    board_release: str = "full3d",
    origin_mm: Tuple[float, float] = (0.0, 0.0),
    spec: Optional[GlobalGridSpec] = None,
    cameras: Optional[Sequence[int]] = None,
    distortion_model: DistortionModel = DistortionModel.STANDARD,
    fix_aspect_ratio: bool = True,
    n_views: Optional[int] = None,
    figure_dir: Optional[Path] = None,
    image_loader: Optional[Callable[[int, int], Optional[np.ndarray]]] = None,
    board_params: Optional[Any] = None,
) -> JointDriverResult:
    """Resolve the global grid, run the joint/polynomial solve, save, and report.

    ``spec`` is the dotboard click record (``None`` for ChArUco, whose corner ids give the grid
    directly). ``cameras`` is the EXPECTED camera set (defaults to the cameras present in
    ``image_size_by_cam``); it is asserted against the cameras the grid actually resolves so a
    silently-dropped camera fails loudly rather than calibrating a subset. Raises ``ValueError``
    on any failed view, an unresolvable grid, or a camera/board mismatch — callers translate that
    to their own error surface (CLI ``SystemExit`` / route JSON).

    ``figure_dir`` writes the proof-figure suite beside the record — the pinhole bundle for a
    pinhole solve, the per-camera polynomial figures (detection, fit residual, dewarped board,
    dot-agreement scatter) for a polynomial solve. ``image_loader(cam, view) -> ndarray`` supplies
    raw images for the image-based figures (detection overlays + dewarp) and may be ``None``
    (geometry figures still write). Figure failures are swallowed and never abort the calibration.
    """
    if board not in ("dotboard", "charuco"):
        raise ValueError(f"joint: board must be dotboard|charuco, got {board!r}")
    if model_type not in ("pinhole", "polynomial"):
        raise ValueError(
            f"joint: model_type must be pinhole|polynomial, got {model_type!r}"
        )
    if board_release not in ("full3d", "z_only", "none"):
        raise ValueError(
            f"joint: board_release must be full3d|z_only|none, got {board_release!r}"
        )
    if board == "dotboard" and spec is None:
        raise ValueError("joint: a GlobalGridSpec is required for a dotboard solve")

    _check_some_detected(detections_by_cam)

    cams = sorted(
        int(c) for c in (cameras if cameras is not None else image_size_by_cam.keys())
    )
    if not cams:
        raise ValueError("joint: no cameras to calibrate")
    # n_views is the per-camera view count stamped into board_meta; require it rather than infer
    # (inferring max(len) could diverge from the CLI's value — the very drift this driver prevents).
    if n_views is None:
        raise ValueError("joint: n_views is required")
    n_views = int(n_views)
    origin_mm = (float(origin_mm[0]), float(origin_mm[1]))

    global_index = resolve_global_grid(detections_by_cam, spec, spacing_mm=spacing_mm)

    # Board geometry stamped into every record so the joint model is self-describing (the
    # GUI/CLI read it back instead of config). None when the caller passed no params.
    geo = (
        rec.geometry_meta(board, board_params, model_type=model_type)
        if board_params is not None
        else None
    )

    if model_type == "polynomial":
        # board_release and datum_camera are pinhole-only (the polynomial map has no released
        # board and no datum-camera pose — only datum_view matters); they are intentionally not
        # consulted here, not silently mis-applied. Callers pass them with their defaults.
        poly = run_joint_polynomial(
            detections_by_cam,
            global_index,
            spacing_mm,
            cams,
            datum_view,
            origin_mm=origin_mm,
            image_size_by_cam=image_size_by_cam,
        )
        wf = rec.WorldFrame(
            mode="global_grid", origin_mm=np.asarray(origin_mm, dtype=np.float64)
        )
        paths: List[Path] = []
        per_cam_rms: Dict[int, float] = {}
        for cam in cams:
            m = poly[cam]
            # per_view_rms for a polynomial record is the combined in-plane fit residual in mm
            # (per-axis rms_x/rms_y live on the model); a pinhole record's per_view_rms is
            # reprojection px — different units, never compare across model types.
            per_cam_rms[cam] = float(np.hypot(m.rms_x_mm, m.rms_y_mm))
            mono = rec.MonoRecord(
                camera=cam,
                board_type=board,
                camera_model=m,
                world_frame=wf,
                per_view_rms=[per_cam_rms[cam]],
                board_meta={
                    "spacing_mm": spacing_mm,
                    "n_views": 1,
                    "joint": 1,
                    "datum_view": datum_view,
                    "dt": float(dt),
                    **({"geometry": geo} if geo else {}),
                },
            )
            paths.append(
                rec.save_mono(mono, rec.mono_model_dir_for_source(source, cam, board))
            )

        if figure_dir is not None:
            # Proof figures beside the per-camera records: detection overlay, polynomial-fit
            # residual (the reprojection analogue), the dewarped board, and a back-projected-dots
            # agreement scatter for >=2 cameras. Guarded so a figure failure never aborts the solve.
            try:
                from . import figures

                figures.write_joint_polynomial_figures(
                    figure_dir,
                    models_by_cam=poly,
                    detections_by_cam=detections_by_cam,
                    global_index=global_index,
                    spacing=spacing_mm,
                    origin_mm=origin_mm,
                    datum_view=datum_view,
                    image_loader=image_loader,
                )
            except (
                Exception
            ):  # pragma: no cover - diagnostics must never abort a calibration
                from loguru import logger

                logger.warning("joint polynomial figures failed (calibration kept)")

        return JointDriverResult(
            board_type=board,
            model_type="polynomial",
            cameras=cams,
            paths=paths,
            per_camera_rms=per_cam_rms,
            rms_units="mm",
            rms_px=float("nan"),
            converged=True,
            cross_camera_board_agreement_mm=float("nan"),
            n_board_dots=0,
            info={"datum_view": datum_view},
        )

    result = run_joint(
        detections_by_cam,
        global_index,
        spacing_mm,
        datum_camera,
        datum_view,
        origin_mm=origin_mm,
        board_release=board_release,
        image_size_by_cam=image_size_by_cam,
        expected_cameras=cams,
        distortion_model=distortion_model,
        fix_aspect_ratio=fix_aspect_ratio,
    )

    record = rec.JointRecord(
        cameras=result.cameras,
        board_type=board,
        models=result.models,
        board=result.board,
        world_frame=result.world_frame,
        spacing_mm=result.spacing_mm,
        board_release=result.board_release,
        per_camera_rms=result.per_camera_rms,
        rms_px=result.rms_px,
        board_meta={
            "converged": int(result.converged),
            "n_views": n_views,
            "dt": float(dt),
            "datum_camera": datum_camera,
            "datum_view": datum_view,
            "n_board_dots": len(result.board),
            "cross_camera_board_agreement_mm": float(
                result.cross_camera_board_agreement_mm
            ),
            "spacing_mm": result.spacing_mm,
            **({"geometry": geo} if geo else {}),
        },
    )
    path = rec.save_joint(record, rec.joint_model_dir_for_source(source, board))

    if figure_dir is not None:
        # Proof figures beside the record: detection overlays, per-camera reprojection, the
        # cameras-relative-to-board scene, and the dewarp agreement figure. Drawn from the
        # in-memory result (the record drops per-view poses); each sub-figure swallows its own
        # errors, and the whole block is guarded so a figure failure never fails the solve.
        try:
            from . import figures

            figures.write_joint_figures(
                figure_dir,
                result=result,
                detections_by_cam=detections_by_cam,
                global_index=global_index,
                spacing=spacing_mm,
                board_type=board,
                datum_view=datum_view,
                image_loader=image_loader,
            )
        except (
            Exception
        ):  # pragma: no cover - diagnostics must never abort a calibration
            from loguru import logger

            logger.warning("joint figures failed (calibration kept): {}", path)

    return JointDriverResult(
        board_type=board,
        model_type="pinhole",
        cameras=result.cameras,
        paths=[path],
        per_camera_rms=result.per_camera_rms,
        rms_units="px",
        rms_px=result.rms_px,
        converged=bool(result.converged),
        cross_camera_board_agreement_mm=float(result.cross_camera_board_agreement_mm),
        n_board_dots=len(result.board),
        info={
            "datum_camera": datum_camera,
            "datum_view": datum_view,
            "board_release": board_release,
            "bootstrap": result.info.get("bootstrap", {}),
        },
    )
