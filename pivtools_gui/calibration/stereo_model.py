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
(Ux,Uy,Uz) by least squares. The result is returned in ONE right-handed world frame —
the user's clicked +X/+Y with +Z = +X x +Y — so W (= Uz) shares the z-axis with the
world_z coordinate. There is no independent uz sign flip: the sign is carried by the
4x3 solve, so W and world_z are always in the same frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from .camera_model import CameraModel, DistortionModel, fit_intrinsics
from .detection.base import BoardDetector, DetectionResult
from .pipeline import Calibrator, view_diagnostics_summary
from .record import StereoRecord, WorldFrame
from .world_frame import apply_world_frame, resolve_world_frame, resolve_world_frame_from_grid

# Cross-camera stereo-calibration thresholds (flat same-side dotboard/charuco). A genuine
# same-side pair shares the whole board, so most views contribute corresponding points;
# zero usable shared views means the pair is NOT same-side (wrong pair/datum, or a
# transmission rig that belongs to the stepped-board path) — we fail loudly rather than
# fabricate a relative pose.
MIN_STEREO_VIEWS = 1       # raise below this many usable shared views
MIN_SHARED_POINTS = 6      # a view must share >= this many features to contribute
_STEREO_VIEWS_WEAK = 3     # warn below this many shared views (pose weakly constrained)


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


def _views_share_object_points(objs: Sequence[np.ndarray]) -> bool:
    """True when every view's board-local points are identical.

    This is ``cv2.calibrateCameraRO``'s precondition (release-object refines ONE shared
    board, so every view must present the same object points). Holds for a board fully in
    frame each pose; fails for partial boards. Used to decide RO-or-plain per camera.
    """
    if len(objs) < 3:
        return False
    first = np.asarray(objs[0])
    return all(
        np.asarray(o).shape == first.shape and np.allclose(o, first) for o in objs[1:]
    )


def _stereo_correspondence(
    det1: Sequence[DetectionResult],
    det2: Sequence[DetectionResult],
    wf1: WorldFrame,
    wf2: WorldFrame,
    spacing: float,
) -> Tuple[list, list, list]:
    """Per-shared-view corresponding ``(object_world, image1, image2)`` point sets.

    Matches the same physical feature in both cameras by its world coordinate under each
    camera's resolved frame. For a same-side board those coordinates are exact integer
    multiples of ``spacing`` (plus ``origin_mm``), so the rounded key is robust for both
    dotboard (grid indices) and ChArUco (global corner ids). Returns three parallel lists
    (one entry per usable view) of float32 arrays; views sharing fewer than
    ``MIN_SHARED_POINTS`` features are dropped. The object points are expressed in cam1's
    world frame — the single 3D frame ``cv2.stereoCalibrate`` needs.
    """
    obj_out: list = []
    img1_out: list = []
    img2_out: list = []
    for d1, d2 in zip(det1, det2):
        if not (d1.success and d2.success):
            continue
        w1 = apply_world_frame(d1.grid_indices, spacing, wf1)
        w2 = apply_world_frame(d2.grid_indices, spacing, wf2)
        key2 = {
            (round(float(x), 3), round(float(y), 3)): i
            for i, (x, y, _z) in enumerate(w2)
        }
        obj, i1, i2 = [], [], []
        for i, (x, y, z) in enumerate(w1):
            j = key2.get((round(float(x), 3), round(float(y), 3)))
            if j is not None:
                obj.append([float(x), float(y), float(z)])
                i1.append(d1.image_points[i])
                i2.append(d2.image_points[j])
        if len(obj) >= MIN_SHARED_POINTS:
            obj_out.append(np.asarray(obj, dtype=np.float32))
            img1_out.append(np.asarray(i1, dtype=np.float32).reshape(-1, 2))
            img2_out.append(np.asarray(i2, dtype=np.float32).reshape(-1, 2))
    return obj_out, img1_out, img2_out


def _stereo_calibrate(
    obj: list,
    img1: list,
    img2: list,
    K1: np.ndarray,
    dist1: np.ndarray,
    K2: np.ndarray,
    dist2: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Joint cam1->cam2 ``(R, T)`` over all shared views via ``cv2.stereoCalibrate``.

    ``CALIB_FIX_INTRINSIC`` holds each camera's intrinsics fixed (each was calibrated from
    all its own views) and optimises only the rigid relative pose, minimising joint
    reprojection across both cameras over every shared view — the gold-standard estimate,
    far better-conditioned than composing two single-view ``solvePnP`` poses. Returns
    ``(R (3,3), T (3,1), rms_px)``. Convention matches ``compose_stereo``: ``Xc2 = R·Xc1 + T``.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, _, _, _, _, R, T, _e, _f = cv2.stereoCalibrate(
        obj,
        img1,
        img2,
        np.asarray(K1, dtype=np.float64),
        np.asarray(dist1, dtype=np.float64),
        np.asarray(K2, dtype=np.float64),
        np.asarray(dist2, dtype=np.float64),
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=criteria,
    )
    return (
        np.asarray(R, dtype=np.float64),
        np.asarray(T, dtype=np.float64).reshape(3, 1),
        float(rms),
    )


@dataclass
class StereoCalibrator:
    """Calibrate a stereo pair into one shared world frame."""

    detector: BoardDetector
    board_type: str
    distortion_model: DistortionModel = DistortionModel.STANDARD
    fix_aspect_ratio: bool = True
    fix_k3: bool = True
    fix_k2: bool = False  # pin r⁴ radial term — set for few-view (<3) fits
    # Flat same-side dot grids default to the DaVis-style release-object intrinsics fit
    # (Strobl-Hirzinger); ``run_stereo`` downgrades per camera if a board is partial.
    use_release_object: bool = True

    def _mono(self, use_release_object: Optional[bool] = None) -> Calibrator:
        return Calibrator(
            detector=self.detector, board_type=self.board_type,
            distortion_model=self.distortion_model,
            fix_aspect_ratio=self.fix_aspect_ratio, fix_k3=self.fix_k3,
            fix_k2=self.fix_k2,
            use_release_object=(
                self.use_release_object
                if use_release_object is None
                else use_release_object
            ),
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
        det1: Optional[Sequence[DetectionResult]] = None,
        det2: Optional[Sequence[DetectionResult]] = None,
    ) -> StereoRecord:
        """Calibrate a flat same-side stereo pair into one shared world frame.

        Camera 1 defines the world frame (its clicked datum). The cross-camera pose comes
        from ``cv2.stereoCalibrate(CALIB_FIX_INTRINSIC)`` over ALL shared views (the
        gold-standard joint estimate), and camera 2's world pose is composed onto camera
        1's anchor. ``clicks2`` (dotboard) / inheritance (ChArUco global corner ids) only
        LABELS camera 2's features so correspondence can match the same physical dot — it
        no longer fits camera 2's pose. Raises if the pair shares no board features (not
        same-side; transmission/opposite-face rigs belong to the stepped-board path).
        """
        mono = self._mono()

        # Detect both cameras up front: cam1's detections seed run_mono (no re-detect),
        # cam2's drive its intrinsics, and BOTH feed the cross-camera correspondence.
        # Caller-supplied detections (the sidecar cache) are reused when given.
        det1 = mono.detect_views(images1) if det1 is None else list(det1)
        det2 = mono.detect_views(images2) if det2 is None else list(det2)
        if not det1[datum_index].success:
            raise RuntimeError("cam1 datum view failed detection")
        if not det2[datum_index].success:
            raise RuntimeError("cam2 datum view failed detection")

        # Release-object intrinsics need identical board points in every view; downgrade
        # per camera (visibly) to a plain fit when a board is partial, so RO-default never
        # aborts the solve.
        ro1 = self.use_release_object and _views_share_object_points(
            [d.board_local_points for d in det1 if d.success]
        )
        ro2 = self.use_release_object and _views_share_object_points(
            [d.board_local_points for d in det2 if d.success]
        )
        if self.use_release_object and not (ro1 and ro2):
            logger.warning(
                "stereo: release-object disabled for camera(s) without identical per-view "
                "board points (cam1 RO={}, cam2 RO={}) — using plain calibrateCamera",
                ro1,
                ro2,
            )

        # Camera 1 defines the shared world frame (and writes its own figures).
        cam1_fig_prefix = "cam1_" if figure_dir is not None else ""
        rec1 = self._mono(use_release_object=ro1).run_mono(
            images1, camera=cam1, clicks=clicks, datum_index=datum_index,
            spacing_mm=spacing_mm, board_meta=board_meta,
            figure_dir=figure_dir, figure_prefix=cam1_fig_prefix, frame_grid=frame_grid,
            origin_mm=origin_mm, detections=det1,
        )
        wf = rec1.world_frame  # carries origin_mm; cam2 is labelled into this shared frame
        sp = spacing_mm if spacing_mm is not None else rec1.board_meta["spacing_mm"]
        K1, dist1 = rec1.camera_model.K, rec1.camera_model.dist
        R1, t1 = rec1.camera_model.R, rec1.camera_model.t

        # Camera 2: independent intrinsics. Its datum pose is NOT solved here — for a flat
        # same-side pair the cross-camera pose comes from cv2.stereoCalibrate over all
        # shared views, then cam2's world pose is composed onto cam1's anchor.
        used2 = [(i, d) for i, d in enumerate(det2) if d.success]
        objs2 = [d.board_local_points for _, d in used2]
        imgs2 = [d.image_points for _, d in used2]
        h, w = np.asarray(images2[datum_index]).shape[:2]
        image_size = (int(w), int(h))
        K2, dist2, rv2, tv2, rms2, pv2, _rel2 = fit_intrinsics(
            objs2, imgs2, image_size,
            distortion_model=self.distortion_model,
            fix_aspect_ratio=self.fix_aspect_ratio, fix_k3=self.fix_k3,
            fix_k2=self.fix_k2,
            use_release_object=ro2,
        )

        # Resolve cam2's world frame ONLY to LABEL its features in cam1's frame so
        # correspondence can match the same physical dot — no pose is fit from it.
        datum2 = det2[datum_index]
        if frame_grid2 is not None:
            wf2 = resolve_world_frame_from_grid(
                frame_grid2["origin"], frame_grid2["x_axis"], frame_grid2["y_axis"])
            wf2.origin_mm = wf.origin_mm
        elif clicks2 is not None:
            wf2 = resolve_world_frame(datum2.grid_indices, datum2.image_points, clicks2)
            wf2.origin_mm = wf.origin_mm
        else:
            wf2 = wf  # ChArUco: globally-consistent corner ids share cam1's frame

        # Cross-camera correspondence over ALL shared views, then the joint relative pose.
        obj_pts, img1_pts, img2_pts = _stereo_correspondence(det1, det2, wf, wf2, sp)
        if len(obj_pts) < MIN_STEREO_VIEWS:
            raise ValueError(
                "flat stereo expects a same-side pair sharing board features, but found "
                f"{len(obj_pts)} usable shared view(s) (need >= {MIN_STEREO_VIEWS}). Check "
                "the camera pair, the datum frame, and that both cameras see the same board "
                "face. Transmission / opposite-face rigs belong to the stepped-board path."
            )
        if len(obj_pts) < _STEREO_VIEWS_WEAK:
            logger.warning(
                "stereo: only {} shared view(s) for stereoCalibrate — the relative pose "
                "may be weakly constrained",
                len(obj_pts),
            )
        R_stereo, T_stereo, stereo_rms = _stereo_calibrate(
            obj_pts, img1_pts, img2_pts, K1, dist1, K2, dist2, image_size
        )

        # cam2 world pose composed onto cam1's anchor: world->cam2 = (cam1->cam2) o (world->cam1).
        R2 = R_stereo @ R1
        t2 = (R_stereo @ np.asarray(t1).reshape(3, 1) + T_stereo).reshape(3, 1)
        model2 = CameraModel(K2, dist2, R2, t2, image_size,
                             self.distortion_model, rms2)

        meta = dict(board_meta or {})
        meta.setdefault("spacing_mm", float(sp))
        # Carry cam1's view count so the stereo record (and the model GET) report n_views.
        meta.setdefault("n_views", rec1.board_meta.get("n_views"))
        # Carry the board geometry stamped into cam1's mono record so the stereo model is
        # self-describing too (both cameras share the one board).
        if rec1.board_meta.get("geometry"):
            meta.setdefault("geometry", rec1.board_meta["geometry"])
        meta["z_sign_toward_cameras"] = camera_z_sign(rec1.camera_model, model2)
        meta["stereo_rms_px"] = float(stereo_rms)
        meta["stereo_method"] = "stereoCalibrate"
        # The number of shared views cv2.stereoCalibrate actually ran on — can be fewer
        # than cam1's mono view count when a pose lacks board overlap in cam2.
        meta["n_stereo_views"] = int(len(obj_pts))
        # Keys are valid MATLAB struct field names ("cam1"/"cam2") — scipy.savemat drops
        # field names that start with a digit, so a bare camera number would be lost.
        meta["ro_applied"] = {f"cam{cam1}": bool(ro1), f"cam{cam2}": bool(ro2)}
        # cam1's summary was built inside run_mono; cam2's detections are local.
        meta["view_diagnostics"] = {
            "cam1": rec1.board_meta.get("view_diagnostics", {}),
            "cam2": view_diagnostics_summary(det2),
        }

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
            # Reuse the datum detection the calibration ran on (may be cached) rather than
            # re-detecting, so the stereo figure's board overlay matches the solve exactly.
            datum_board_world = apply_world_frame(
                det1[datum_index].grid_indices, sp, wf
            )
            figures.write_stereo_figures(
                figure_dir, model1=rec1.camera_model, model2=model2,
                R_stereo=R_stereo, T_stereo=T_stereo,
                img1=images1[datum_index], img2=images2[datum_index],
                datum_board_world=datum_board_world, spacing=sp,
                cam1_num=cam1, cam2_num=cam2,
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
) -> np.ndarray:
    """Solve (Ux,Uy,Uz) mm at each world point from paired pixel displacements.

    For each point, stack J1 (2x3) and J2 (2x3) into a 4x3 system A and solve
    A u = [d1x,d1y,d2x,d2y] by least squares (normal equations). NaN rows where
    either displacement is NaN.

    The velocity is returned in the SAME right-handed world frame as the coordinates —
    +Uz along (+X x +Y) from the user's clicked axes. The 4x3 solve already carries the
    correct sign (PIV is right-handed, W is u_z along that z-axis); there is no
    independent toward-cameras flip, so W and the returned ``world_z`` always share one
    frame.
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
    return out


def _interpolate_field(
    coords_px: np.ndarray, field: np.ndarray, query_px: np.ndarray, method: str = "linear"
) -> np.ndarray:
    """Interpolate a per-camera field defined on a regular pixel grid at query points.

    coords_px : (H,W,2) image-down pixel centres of the grid (ascending).
    field : (H,W) values. query_px : (M,2) image-down pixels.
    method :
        "linear"  — scipy bilinear, NaN-propagating (legacy default).
        "cubic"/"lanczos" — cv2.remap Keys-cubic / Lanczos-4. NaN holes are nearest-filled
        so the wider kernel has finite data, then the result is re-masked to the bilinear
        valid domain (only the kernel changes, not the valid region). The higher-order
        kernels remove the grid-locked variance ringing the bilinear resample imprints on
        stereo statistics.
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

    if method == "linear":
        interp = RegularGridInterpolator(
            (y_axis, x_axis), fld, method="linear", bounds_error=False, fill_value=np.nan
        )
        q = np.column_stack([query_px[:, 1], query_px[:, 0]])
        return interp(q)

    if method not in ("cubic", "lanczos"):
        raise ValueError(f"interpolator must be linear|cubic|lanczos, got {method!r}")

    flag = cv2.INTER_CUBIC if method == "cubic" else cv2.INTER_LANCZOS4
    m = query_px.shape[0]

    valid = np.isfinite(fld)
    if not valid.any():
        # All-NaN field (e.g. dropped camera frame): no background for the distance
        # transform, and nothing to interpolate from.
        return np.full(m, np.nan)

    # Nearest-fill NaN holes so the cubic/lanczos stencil reads finite values.
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    fld_filled = fld[tuple(idx)]

    # cv2.remap requires the source laid out (Y=axis0, X=axis1).
    assert fld_filled.shape[:2] == (len(y_axis), len(x_axis)), (
        "remap requires (Y, X) source layout"
    )

    dx = (x_axis[-1] - x_axis[0]) / (len(x_axis) - 1)
    dy = (y_axis[-1] - y_axis[0]) / (len(y_axis) - 1)
    # cv2.remap arg order is map1 = X (cols), map2 = Y (rows).
    col = ((query_px[:, 0] - x_axis[0]) / dx).astype(np.float32)
    row = ((query_px[:, 1] - y_axis[0]) / dy).astype(np.float32)

    # cv2.remap caps each map dimension at SHRT_MAX (32767), so pack the M query points
    # into a near-square 2D map (both sides << 32767) and unpack afterwards.
    side = int(np.ceil(np.sqrt(m)))
    pad = side * side - m
    col2 = np.concatenate([col, np.zeros(pad, np.float32)]).reshape(side, side)
    row2 = np.concatenate([row, np.zeros(pad, np.float32)]).reshape(side, side)

    # NaN border: a query whose cubic/lanczos stencil reaches past the grid edge can't be
    # built from clean data, so it becomes NaN rather than reading an extrapolated edge.
    # This erodes the outer ~stencil-radius ring vs the bilinear path (deliberate — never
    # fabricate edge values).
    out = cv2.remap(
        fld_filled.astype(np.float32), col2, row2, flag,
        borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"),
    ).reshape(-1)[:m]

    # Re-mask to the bilinear valid domain (NaN from holes / out-of-grid). cv2 quantises
    # the bilinear weights to a 1/32 fixed-point table, so a fully-valid cell can read
    # ~0.9999988 instead of 1.0; threshold at 0.99 (any NaN-touching cell reads <= 0.75).
    maskq = cv2.remap(
        valid.astype(np.float32), col2, row2, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    ).reshape(-1)[:m]
    out = out.copy()
    out[maskq < 0.99] = np.nan
    return out.astype(np.float64)


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
    interpolator: str = "linear",
    bmask1: Optional[np.ndarray] = None,
    bmask2: Optional[np.ndarray] = None,
):
    """Full-field 3C reconstruction from two cameras' PIV grids.

    cam1's grid is back-projected to world, projected into cam2, and cam2's
    displacement field is interpolated there; the 4x3 solve then gives (U,V,W).
    ``interpolator`` selects the cam2 resample kernel: "linear" (legacy bilinear,
    rings) or "cubic"/"lanczos" (cv2.remap, anti-ring) — see ``_interpolate_field``.

    ``bmask1``/``bmask2`` are the two cameras' per-vector validity masks (True = masked,
    the per-camera PIV ``b_mask`` convention). When supplied, the returned ``mask`` flags
    every output point that is unreliable: either camera flagged its contributing vector,
    or the 3C solve could not be formed (cam1 NaN input, or cam2's projection fell outside
    its grid).

    Returns (world_x, world_y, world_z, U, V, W, mask) on cam1's grid, U/V/W in m/s, mask
    a (H,W) bool. W is u_z in the same right-handed world frame as world_z (no sign flip) —
    coordinates and velocity always share one frame.
    """
    H, W = ux1_px.shape
    c1 = np.asarray(coords1_px, dtype=np.float64).reshape(H, W, 2)
    flat_px = c1.reshape(-1, 2)
    d1 = np.column_stack([ux1_px.reshape(-1), uy1_px.reshape(-1)])

    world = model1.back_project_to_plane(flat_px, z_world, tilt_x, tilt_y)  # (N,3)

    # Project world grid into cam2 and interpolate cam2 displacements there.
    proj2 = model2.project(world)  # (N,2) image-down px
    c2 = np.asarray(coords2_px).reshape(*ux2_px.shape, 2)
    dx2 = _interpolate_field(c2, ux2_px, proj2, method=interpolator)
    dy2 = _interpolate_field(c2, uy2_px, proj2, method=interpolator)
    d2 = np.column_stack([dx2, dy2])

    vel_mm = reconstruct_3c_at_points(model1, model2, world, d1, d2)  # (N,3) mm/frame
    vel_ms = (vel_mm / 1000.0) / dt

    wx = world[:, 0].reshape(H, W)
    wy = world[:, 1].reshape(H, W)
    wz = world[:, 2].reshape(H, W)
    U = vel_ms[:, 0].reshape(H, W)
    V = vel_ms[:, 1].reshape(H, W)
    Wc = vel_ms[:, 2].reshape(H, W)

    # Output validity mask (True = masked). Invalid where the 3C solve produced a
    # non-finite vector (cam1 NaN input, or cam2's projection left its grid -> NaN), or
    # where either camera flagged its contributing vector. cam1's mask is already on this
    # grid; cam2's is resampled through the same cam1->cam2 projection used for the
    # displacements (bilinear, so a query reaching cam2's masked/out-of-grid region is
    # masked).
    mask = ~(np.isfinite(U) & np.isfinite(V) & np.isfinite(Wc))
    if bmask1 is not None:
        mask |= np.asarray(bmask1, dtype=bool).reshape(H, W)
    if bmask2 is not None:
        m2 = _interpolate_field(
            c2, np.asarray(bmask2, dtype=np.float64), proj2, method="linear"
        ).reshape(H, W)
        mask |= (~np.isfinite(m2)) | (m2 >= 0.5)
    return wx, wy, wz, U, V, Wc, mask
