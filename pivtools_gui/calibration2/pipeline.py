"""calibration2.pipeline — the single calibration orchestrator.

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
    fit_intrinsics,
    fit_pose,
)
from .detection.base import BoardDetector, DetectionResult
from .record import MonoRecord, WorldFrame
from .world_frame import resolve_world_frame, resolve_world_frame_from_grid, apply_world_frame


@dataclass
class Calibrator:
    """Calibrate one camera from multiple board views."""

    detector: BoardDetector
    board_type: str
    distortion_model: DistortionModel = DistortionModel.STANDARD
    fix_aspect_ratio: bool = True
    fix_k3: bool = True
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
    ) -> MonoRecord:
        detections = self.detect_views(images)
        ok = [d.success for d in detections]
        if not ok[datum_index]:
            raise RuntimeError(
                f"datum view (index {datum_index}) failed detection; cannot define world frame"
            )

        used = [(i, d) for i, d in enumerate(detections) if d.success]
        if len(used) < 3:
            raise RuntimeError(
                f"need >=3 successful views for intrinsics, got {len(used)}"
            )

        objs = [d.board_local_points for _, d in used]
        imgs = [d.image_points for _, d in used]

        if image_size is None:
            h, w = np.asarray(images[datum_index]).shape[:2]
            image_size = (int(w), int(h))

        K, dist, rvecs, tvecs, rms, per_view = fit_intrinsics(
            objs, imgs, image_size,
            distortion_model=self.distortion_model,
            fix_aspect_ratio=self.fix_aspect_ratio,
            fix_k3=self.fix_k3,
            use_release_object=self.use_release_object,
        )

        datum = detections[datum_index]
        sp = spacing_mm if spacing_mm is not None else datum.spacing_mm
        if frame_grid is not None:
            wf = resolve_world_frame_from_grid(
                frame_grid["origin"], frame_grid["x_axis"], frame_grid["y_axis"])
        else:
            wf = resolve_world_frame(datum.grid_indices, datum.image_points, clicks)
        if origin_mm is not None:
            wf.origin_mm = np.asarray(origin_mm, dtype=np.float64).reshape(2)
        world_pts = apply_world_frame(datum.grid_indices, sp, wf)
        R, t = fit_pose(world_pts, datum.image_points, K, dist, planar=True)

        cam = CameraModel(
            K=K, dist=dist, R=R, t=t, image_size=image_size,
            distortion_model=self.distortion_model, rms=rms,
        )
        meta = dict(board_meta or {})
        meta.setdefault("spacing_mm", float(sp))
        meta.setdefault("n_views", len(used))

        if figure_dir is not None:
            # Drawn while detections + per-view poses are live; never persisted to the
            # record. Each figure swallows its own errors, so this cannot abort the fit.
            from . import figures
            figures.write_mono_figures(
                figure_dir, images=images, detections=detections, used=used,
                K=K, dist=dist, rvecs=rvecs, tvecs=tvecs, per_view=per_view, rms=rms,
                cam=cam, wf=wf, spacing=sp, board_type=self.board_type,
                datum_index=datum_index, board_meta=meta, prefix=figure_prefix,
            )

        return MonoRecord(
            camera=camera,
            board_type=self.board_type,
            camera_model=cam,
            world_frame=wf,
            per_view_rms=list(per_view),
            board_meta=meta,
        )
