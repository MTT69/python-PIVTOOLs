"""calibration.pipeline — the single calibration orchestrator.

detect each view -> fit bundled intrinsics -> resolve the user world frame on the
datum view -> solve the datum pose in that frame -> assemble a CameraModel. Board
type and distortion model are data; there is one code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera_model import (
    CameraModel,
    DistortionModel,
    ScaleFactorModel,
    fit_intrinsics,
    fit_polynomial,
    fit_pose,
)
from .detection.base import BoardDetector, DetectionResult
from .record import MonoRecord, WorldFrame, geometry_meta
from .world_frame import (
    apply_world_frame,
    resolve_world_frame,
    resolve_world_frame_from_grid,
)


def view_diagnostics_summary(
    detections: Sequence[DetectionResult],
) -> Dict[str, object]:
    """Per-view detection diagnostics as mat-safe parallel arrays.

    One entry per view, index-aligned with the input image list. Counts default
    to 0 for detectors that don't report them (e.g. ChArUco has no rescue path).
    Warnings are joined into one string keyed by view index; the key is omitted
    when no view warned. Stored under ``board_meta["view_diagnostics"]`` so the
    record carries the same honesty data the detection figures show.
    """
    diags = [d.diagnostics or {} for d in detections]
    out: Dict[str, object] = {
        "view_index": np.arange(len(detections), dtype=np.int64),
        "success": np.array([int(d.success) for d in detections], dtype=np.int64),
        "n_points": np.array([d.n for d in detections], dtype=np.int64),
        "n_synthetic": np.array(
            [
                (
                    0
                    if d.synthetic_mask is None
                    else int(np.count_nonzero(d.synthetic_mask))
                )
                for d in detections
            ],
            dtype=np.int64,
        ),
        "n_rescued": np.array(
            [int(g.get("n_rescued", 0) or 0) for g in diags], dtype=np.int64
        ),
        "n_infilled": np.array(
            [int(g.get("n_infilled", 0) or 0) for g in diags], dtype=np.int64
        ),
        "ransac_n_rejected": np.array(
            [int(g.get("ransac_n_rejected", 0) or 0) for g in diags], dtype=np.int64
        ),
        "edge_fraction": np.array(
            [float(g.get("edge_fraction", 0.0) or 0.0) for g in diags], dtype=np.float64
        ),
    }
    warnings = [
        f"view {i}: {g['warning']}" for i, g in enumerate(diags) if g.get("warning")
    ]
    if warnings:
        out["warnings"] = "; ".join(warnings)
    return out


@dataclass
class Calibrator:
    """Calibrate one camera from multiple board views."""

    detector: BoardDetector
    board_type: str
    model_type: str = "pinhole"  # "pinhole" (3D) or "polynomial" (single-plane)
    distortion_model: DistortionModel = DistortionModel.STANDARD
    fix_aspect_ratio: bool = True
    fix_k3: bool = True
    fix_k2: bool = False  # pin r⁴ radial term — set for few-view (<3) fits
    use_release_object: bool = False

    def detect_views(self, images: Sequence[np.ndarray]) -> List[DetectionResult]:
        return [self.detector.detect(im) for im in images]

    def run_mono(
        self,
        images: Sequence[np.ndarray],
        camera: int = 1,
        clicks: Optional[Dict[str, object]] = None,
        datum_index: int = 0,
        spacing_mm: Optional[float] = None,
        board_meta: Optional[dict] = None,
        image_size: Optional[Tuple[int, int]] = None,
        figure_dir: Optional[Path] = None,
        figure_prefix: str = "",
        frame_grid: Optional[Dict[str, object]] = None,
        origin_mm: Optional[Tuple[float, float]] = None,
        detections: Optional[Sequence[DetectionResult]] = None,
    ) -> MonoRecord:
        # ``detections`` lets a caller reuse a detection pass it already ran (e.g. the
        # stereo driver, which needs both cameras' detections for correspondence) instead
        # of re-detecting. Default ``None`` preserves the detect-internally behaviour.
        detections = (
            self.detect_views(images) if detections is None else list(detections)
        )
        ok = [d.success for d in detections]
        if not ok[datum_index]:
            raise RuntimeError(
                f"datum view (index {datum_index}) failed detection; cannot define world frame"
            )

        datum = detections[datum_index]
        sp = spacing_mm if spacing_mm is not None else datum.spacing_mm
        if image_size is None:
            h, w = np.asarray(images[datum_index]).shape[:2]
            image_size = (int(w), int(h))

        # World frame is resolved the same way for both model types: it defines the
        # world-mm targets (clicked origin / +X / +Y / origin_mm) on the datum view.
        if frame_grid is not None:
            wf = resolve_world_frame_from_grid(
                frame_grid["origin"], frame_grid["x_axis"], frame_grid["y_axis"]
            )
        else:
            wf = resolve_world_frame(datum.grid_indices, datum.image_points, clicks)
        if origin_mm is not None:
            wf.origin_mm = np.asarray(origin_mm, dtype=np.float64).reshape(2)
        world_pts = apply_world_frame(datum.grid_indices, sp, wf)

        meta = dict(board_meta or {})
        meta.setdefault("spacing_mm", float(sp))
        # Stamp the board geometry that produced this model so the record is self-describing
        # (the GUI/CLI read it back instead of config). detector.params carries the live params;
        # a detector without one (e.g. a test double) simply contributes no geometry block.
        det_params = getattr(self.detector, "params", None)
        if det_params is not None:
            meta.setdefault(
                "geometry",
                geometry_meta(self.board_type, det_params, model_type=self.model_type),
            )
        meta["view_diagnostics"] = view_diagnostics_summary(detections)

        if self.model_type == "polynomial":
            # Single-plane fit: only the datum view, no intrinsics/pose. The world
            # frame is baked into the coefficients via world_pts.
            model = fit_polynomial(datum.image_points, world_pts, image_size)
            meta.setdefault("n_views", 1)
            per_view = [float(np.hypot(model.rms_x_mm, model.rms_y_mm))]
            if figure_dir is not None:
                from . import figures

                figures.write_polynomial_figures(
                    figure_dir,
                    image=images[datum_index],
                    detection=datum,
                    world_pts=world_pts,
                    model=model,
                    wf=wf,
                    spacing=sp,
                    prefix=figure_prefix,
                )
            return MonoRecord(
                camera=camera,
                board_type=self.board_type,
                camera_model=model,
                world_frame=wf,
                per_view_rms=per_view,
                board_meta=meta,
            )

        # Pinhole: bundled intrinsics across all successful views, then datum pose.
        used = [(i, d) for i, d in enumerate(detections) if d.success]
        if len(used) < 3:
            raise RuntimeError(
                f"need >=3 successful views for intrinsics, got {len(used)}"
            )

        objs = [d.board_local_points for _, d in used]
        imgs = [d.image_points for _, d in used]

        K, dist, rvecs, tvecs, rms, per_view, _released = fit_intrinsics(
            objs,
            imgs,
            image_size,
            distortion_model=self.distortion_model,
            fix_aspect_ratio=self.fix_aspect_ratio,
            fix_k3=self.fix_k3,
            fix_k2=self.fix_k2,
            use_release_object=self.use_release_object,
        )

        R, t = fit_pose(world_pts, datum.image_points, K, dist, planar=True)

        cam = CameraModel(
            K=K,
            dist=dist,
            R=R,
            t=t,
            image_size=image_size,
            distortion_model=self.distortion_model,
            rms=rms,
        )
        meta.setdefault("n_views", len(used))

        if figure_dir is not None:
            # Drawn while detections + per-view poses are live; never persisted to the
            # record. Each figure swallows its own errors, so this cannot abort the fit.
            from . import figures

            figures.write_mono_figures(
                figure_dir,
                images=images,
                detections=detections,
                used=used,
                K=K,
                dist=dist,
                rvecs=rvecs,
                tvecs=tvecs,
                per_view=per_view,
                rms=rms,
                cam=cam,
                wf=wf,
                spacing=sp,
                board_type=self.board_type,
                datum_index=datum_index,
                board_meta=meta,
                prefix=figure_prefix,
                world_pts=world_pts,
            )

        return MonoRecord(
            camera=camera,
            board_type=self.board_type,
            camera_model=cam,
            world_frame=wf,
            per_view_rms=list(per_view),
            board_meta=meta,
        )


def build_scale_factor_record(
    camera: int,
    origin_px: Tuple[float, float],
    px_per_mm: float,
    image_size: Tuple[int, int],
    dt: float,
    x_dir: str = "right",
    y_dir: str = "up",
    swap_axes: bool = False,
    frame_idx: Optional[int] = None,
) -> MonoRecord:
    """Build a scale-factor mono record directly from UI/CLI params (no detection).

    The user picks the origin pixel and the +X / +Y directions on the image. The
    direction strings map to the shared sign convention (col_sign is +X, row_sign is
    +Y). Image-down y means "+Y up" is row_sign = -1. ``px_per_mm`` and ``dt`` are
    stamped into ``board_meta`` for provenance (``mm_per_pixel = 1/px_per_mm`` is what
    the model carries; ``dt`` is consumed at apply time like every other method).
    ``frame_idx`` (1-based), when given, is stamped too so the GUI can restore the
    origin/axis overlay on the SAME frame it was picked on.
    """
    if px_per_mm <= 0:
        raise ValueError(f"px_per_mm must be > 0, got {px_per_mm}")
    if x_dir not in ("right", "left"):
        raise ValueError(f"x_dir must be 'right' or 'left', got {x_dir!r}")
    if y_dir not in ("up", "down"):
        raise ValueError(f"y_dir must be 'up' or 'down', got {y_dir!r}")

    col_sign = 1 if x_dir == "right" else -1
    row_sign = -1 if y_dir == "up" else 1  # image-down: up = decreasing pixel row
    origin = np.asarray(origin_px, dtype=np.float64).reshape(2)
    meta = {"px_per_mm": float(px_per_mm), "dt": float(dt)}
    if frame_idx is not None:
        meta["frame_idx"] = int(frame_idx)

    model = ScaleFactorModel(
        origin_px=origin,
        mm_per_pixel=1.0 / float(px_per_mm),
        image_size=(int(image_size[0]), int(image_size[1])),
        swap_axes=int(bool(swap_axes)),
        col_sign=col_sign,
        row_sign=row_sign,
    )
    wf = WorldFrame(
        mode="scale_factor",
        origin_px=origin,
        swap_axes=bool(swap_axes),
        col_sign=col_sign,
        row_sign=row_sign,
    )
    return MonoRecord(
        camera=int(camera),
        board_type="scale_factor",
        camera_model=model,
        world_frame=wf,
        per_view_rms=[],
        board_meta=meta,
    )
