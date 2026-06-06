"""calibration.stereo_model — stereo calibration + 3C reconstruction.

Stereo is built the stepped-board way (CLAUDE.md gotcha #7): each camera's
intrinsics are fit independently, then each camera's datum pose is solved into the
SHARED, cam1-defined world frame, and the stereo relation is derived from the two
poses:

    R_stereo = R2 @ R1.T ,   T_stereo = t2 - R_stereo @ t1

This needs NO ``cv2.stereoCalibrate`` and works even when the two cameras see
different boards (transmission PIV / two-plate), because correspondence is via the
shared world frame, not shared image points. For a shared board, the world frame
defined on cam1 is carried to cam2 by feature correspondence (ChArUco corner ids,
or any globally consistent grid indexing).

3C reconstruction keeps the Willert/Soloff geometric method: for each grid point,
stack the two cameras' projection Jacobians into a 4x3 system and solve for
(Ux,Uy,Uz) by least squares. No uz sign flip — the sign is carried by the models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .camera_model import CameraModel, DistortionModel, fit_intrinsics, fit_pose
from .detection.base import BoardDetector
from .pipeline import Calibrator
from .record import StereoRecord, WorldFrame
from .world_frame import apply_world_frame, resolve_world_frame, resolve_world_frame_from_grid


def compose_stereo(
    model1: CameraModel, model2: CameraModel
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive (R_stereo, T_stereo) relating cam1 -> cam2 from two world-frame poses."""
    R1, t1 = model1.R, model1.t
    R2, t2 = model2.R, model2.t
    R_stereo = R2 @ R1.T
    T_stereo = t2 - R_stereo @ t1
    return R_stereo, T_stereo.reshape(3, 1)


def camera_z_sign(model1, model2) -> float:
    """Convention sign for the out-of-plane axis: +Z (and +w) points TOWARD the cameras.

    The world frame's +Z = +X x +Y from the clicked axes; its physical direction is
    not otherwise pinned (and a reflected/synthetic projection can recover it either
    way). We define +w as toward the cameras and enforce it.

    Pinhole: from the recovered geometry -- camera centre C = -R^T t; if the cameras
    lie on the -Z side of the measurement plane the reconstructed w is negated.

    Polynomial (``Polynomial3DModel``): there is no camera centre, so the sign is the
    convention stored at fit time (``world_z_toward_camera``) -- for the stepped board
    the dotted peak face (Z=0) is nearer the camera than the trough (Z=-step), so +Z
    is toward the cameras (+1). Both models carry the same shared-frame convention.

    Returns +1 (no change) or -1 (negate w). Leaves the in-plane u,v untouched.
    """
    if hasattr(model1, "world_z_toward_camera") or hasattr(model2, "world_z_toward_camera"):
        s1 = getattr(model1, "world_z_toward_camera", None)
        s2 = getattr(model2, "world_z_toward_camera", None)
        signs = [s for s in (s1, s2) if s is not None]
        # Average the carried convention; agreement is expected in the shared frame.
        return 1.0 if float(np.mean(signs)) >= 0.0 else -1.0
    c1 = (-model1.R.T @ model1.t).ravel()
    c2 = (-model2.R.T @ model2.t).ravel()
    return 1.0 if (c1[2] + c2[2]) >= 0.0 else -1.0


@dataclass
class StereoCalibrator:
    """Calibrate a stereo pair into one shared world frame."""

    detector: BoardDetector
    board_type: str
    distortion_model: DistortionModel = DistortionModel.STANDARD
    fix_aspect_ratio: bool = True
    fix_k3: bool = True
    use_release_object: bool = False

    def _mono(self) -> Calibrator:
        return Calibrator(
            detector=self.detector, board_type=self.board_type,
            distortion_model=self.distortion_model,
            fix_aspect_ratio=self.fix_aspect_ratio, fix_k3=self.fix_k3,
            use_release_object=self.use_release_object,
        )

    def run_stereo(
        self,
        images1: Sequence[np.ndarray],
        images2: Sequence[np.ndarray],
        cam1: int = 1,
        cam2: int = 2,
        clicks: Optional[Dict[str, object]] = None,
        clicks2: Optional[Dict[str, object]] = None,
        datum_index: int = 0,
        spacing_mm: Optional[float] = None,
        board_meta: Optional[dict] = None,
        figure_dir: Optional[Path] = None,
        frame_grid: Optional[Dict[str, object]] = None,
        frame_grid2: Optional[Dict[str, object]] = None,
        origin_mm: Optional[Tuple[float, float]] = None,
    ) -> StereoRecord:
        """Calibrate a stereo pair into one shared world frame.

        ``clicks`` defines the world frame on camera 1. ``clicks2`` (optional)
        defines camera 2's frame independently — required when the two cameras do
        NOT share globally consistent feature indexing (dot boards, transmission /
        two-plate). When ``clicks2`` is None, camera 2 inherits camera 1's frame by
        feature correspondence (valid for ChArUco, whose corner ids are global).
        """
        mono = self._mono()

        # Camera 1 defines the shared world frame (and writes its own figures).
        cam1_fig_prefix = "cam1_" if figure_dir is not None else ""
        rec1 = mono.run_mono(
            images1, camera=cam1, clicks=clicks, datum_index=datum_index,
            spacing_mm=spacing_mm, board_meta=board_meta,
            figure_dir=figure_dir, figure_prefix=cam1_fig_prefix, frame_grid=frame_grid,
            origin_mm=origin_mm,
        )
        wf = rec1.world_frame  # carries origin_mm; cam2 inherits it via this shared frame
        sp = spacing_mm if spacing_mm is not None else rec1.board_meta["spacing_mm"]

        # Camera 2: independent intrinsics, datum pose into the SAME world frame.
        det2 = mono.detect_views(images2)
        if not det2[datum_index].success:
            raise RuntimeError("cam2 datum view failed detection")
        used2 = [(i, d) for i, d in enumerate(det2) if d.success]
        objs2 = [d.board_local_points for _, d in used2]
        imgs2 = [d.image_points for _, d in used2]
        h, w = np.asarray(images2[datum_index]).shape[:2]
        K2, dist2, rv2, tv2, rms2, pv2 = fit_intrinsics(
            objs2, imgs2, (int(w), int(h)),
            distortion_model=self.distortion_model,
            fix_aspect_ratio=self.fix_aspect_ratio, fix_k3=self.fix_k3,
            use_release_object=self.use_release_object,
        )
        datum2 = det2[datum_index]
        if frame_grid2 is not None:
            # Independent frame on cam2 by dot grid indices (headless / CLI).
            wf2 = resolve_world_frame_from_grid(
                frame_grid2["origin"], frame_grid2["x_axis"], frame_grid2["y_axis"])
            world2 = apply_world_frame(datum2.grid_indices, sp, wf2)
        elif clicks2 is not None:
            # Independent frame on cam2 (dotboard / transmission / two-plate).
            wf2 = resolve_world_frame(datum2.grid_indices, datum2.image_points, clicks2)
            world2 = apply_world_frame(datum2.grid_indices, sp, wf2)
        else:
            # ChArUco: inherit cam1's frame via globally-consistent corner ids.
            wf2 = wf
            world2 = apply_world_frame(datum2.grid_indices, sp, wf)
        R2, t2 = fit_pose(world2, datum2.image_points, K2, dist2, planar=True)
        model2 = CameraModel(K2, dist2, R2, t2, (int(w), int(h)),
                             self.distortion_model, rms2)

        R_stereo, T_stereo = compose_stereo(rec1.camera_model, model2)

        meta = dict(board_meta or {})
        meta.setdefault("spacing_mm", float(sp))
        meta["z_sign_toward_cameras"] = camera_z_sign(rec1.camera_model, model2)

        if figure_dir is not None:
            # cam2's own proof figures + the stereo-only geometry/dewarp pair. Drawn
            # while cam2's per-view poses are live; nothing here reaches the record.
            from . import figures
            figures.write_mono_figures(
                figure_dir, images=images2, detections=det2, used=used2,
                K=K2, dist=dist2, rvecs=rv2, tvecs=tv2, per_view=pv2, rms=rms2,
                cam=model2, wf=wf2, spacing=sp, board_type=self.board_type,
                datum_index=datum_index, board_meta=meta, prefix="cam2_",
            )
            det1_datum = mono.detector.detect(images1[datum_index])
            datum_board_world = apply_world_frame(det1_datum.grid_indices, sp, wf)
            figures.write_stereo_figures(
                figure_dir, model1=rec1.camera_model, model2=model2,
                R_stereo=R_stereo, T_stereo=T_stereo,
                img1=images1[datum_index], img2=images2[datum_index],
                datum_board_world=datum_board_world, spacing=sp,
            )

        return StereoRecord(
            cam1=cam1, cam2=cam2, board_type=self.board_type,
            model1=rec1.camera_model, model2=model2,
            R_stereo=R_stereo, T_stereo=T_stereo,
            world_frame=wf,
            per_view_rms1=rec1.per_view_rms, per_view_rms2=list(pv2),
            board_meta=meta,
        )


# ---------------------------------------------------------------------------
# 3C reconstruction
# ---------------------------------------------------------------------------

def reconstruct_3c_at_points(
    model1: CameraModel,
    model2: CameraModel,
    world_pts: np.ndarray,
    disp1_px: np.ndarray,
    disp2_px: np.ndarray,
    z_toward_cameras: bool = True,
) -> np.ndarray:
    """Solve (Ux,Uy,Uz) mm at each world point from paired pixel displacements.

    For each point, stack J1 (2x3) and J2 (2x3) into a 4x3 system A and solve
    A u = [d1x,d1y,d2x,d2y] by least squares (normal equations). NaN rows where
    either displacement is NaN. When ``z_toward_cameras`` the out-of-plane sign is
    set so +w points toward the cameras (see ``camera_z_sign``).
    """
    wp = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
    d1 = np.asarray(disp1_px, dtype=np.float64).reshape(-1, 2)
    d2 = np.asarray(disp2_px, dtype=np.float64).reshape(-1, 2)
    n = wp.shape[0]
    out = np.full((n, 3), np.nan, dtype=np.float64)

    valid = np.isfinite(wp).all(axis=1) & np.isfinite(d1).all(axis=1) & np.isfinite(d2).all(axis=1)
    if not valid.any():
        return out

    J1 = model1.jacobian(wp[valid])  # (M,2,3)
    J2 = model2.jacobian(wp[valid])
    A = np.concatenate([J1, J2], axis=1)               # (M,4,3)
    b = np.concatenate([d1[valid], d2[valid]], axis=1)  # (M,4)
    AtA = np.einsum("nij,nik->njk", A, A)              # (M,3,3)
    Atb = np.einsum("nij,ni->nj", A, b)               # (M,3)
    sol = np.linalg.solve(AtA, Atb[..., None]).squeeze(-1)
    out[valid] = sol
    if z_toward_cameras:
        out[:, 2] *= camera_z_sign(model1, model2)
    return out


def _interpolate_field(coords_px: np.ndarray, field: np.ndarray, query_px: np.ndarray) -> np.ndarray:
    """Bilinear-interpolate a per-camera field defined on a regular pixel grid.

    coords_px : (H,W,2) image-down pixel centres of the grid (ascending).
    field : (H,W) values. query_px : (M,2) image-down pixels.
    """
    y_axis = coords_px[:, 0, 1]
    x_axis = coords_px[0, :, 0]
    fld = field
    if y_axis[0] > y_axis[-1]:
        y_axis = y_axis[::-1]
        fld = fld[::-1, :]
    if x_axis[0] > x_axis[-1]:
        x_axis = x_axis[::-1]
        fld = fld[:, ::-1]
    interp = RegularGridInterpolator(
        (y_axis, x_axis), fld, method="linear", bounds_error=False, fill_value=np.nan
    )
    q = np.column_stack([query_px[:, 1], query_px[:, 0]])
    return interp(q)


def reconstruct_3c_field(
    model1: CameraModel,
    model2: CameraModel,
    coords1_px: np.ndarray,
    ux1_px: np.ndarray,
    uy1_px: np.ndarray,
    coords2_px: np.ndarray,
    ux2_px: np.ndarray,
    uy2_px: np.ndarray,
    dt: float,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    z_toward_cameras: bool = True,
):
    """Full-field 3C reconstruction from two cameras' PIV grids.

    cam1's grid is back-projected to world, projected into cam2, and cam2's
    displacement field is interpolated there; the 4x3 solve then gives (U,V,W).

    Returns (world_x, world_y, world_z, U, V, W) on cam1's grid, U/V/W in m/s.
    """
    H, W = ux1_px.shape
    c1 = np.asarray(coords1_px, dtype=np.float64).reshape(H, W, 2)
    flat_px = c1.reshape(-1, 2)
    d1 = np.column_stack([ux1_px.reshape(-1), uy1_px.reshape(-1)])

    world = model1.back_project_to_plane(flat_px, z_world, tilt_x, tilt_y)  # (N,3)

    # Project world grid into cam2 and interpolate cam2 displacements there.
    proj2 = model2.project(world)  # (N,2) image-down px
    dx2 = _interpolate_field(np.asarray(coords2_px).reshape(*ux2_px.shape, 2), ux2_px, proj2)
    dy2 = _interpolate_field(np.asarray(coords2_px).reshape(*uy2_px.shape, 2), uy2_px, proj2)
    d2 = np.column_stack([dx2, dy2])

    vel_mm = reconstruct_3c_at_points(
        model1, model2, world, d1, d2, z_toward_cameras=z_toward_cameras
    )  # (N,3) mm/frame
    vel_ms = (vel_mm / 1000.0) / dt

    wx = world[:, 0].reshape(H, W)
    wy = world[:, 1].reshape(H, W)
    wz = world[:, 2].reshape(H, W)
    U = vel_ms[:, 0].reshape(H, W)
    V = vel_ms[:, 1].reshape(H, W)
    Wc = vel_ms[:, 2].reshape(H, W)
    return wx, wy, wz, U, V, Wc
