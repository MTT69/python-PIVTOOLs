"""calibration2.camera_model — pinhole camera model + DaVis-matching fit.

The single camera-model family for calibration2. Pinhole only. All world<->pixel
conversion lives here (``project`` / ``back_project_to_plane`` / ``projection_jacobian``);
``frames`` handles only the integer pixel<->matlab offset. No implicit Y-flips:
the sign of every axis is carried by ``R``, ``t`` and the projection math.

Fit policy reproduces the DaVis gold standard for the FlowMaster Scheimpflug rig
(see ``docs/calibration-v2/PRD.md`` sec 8.2): 4-coefficient distortion
``[k1, k2, p1, p2, 0]`` (k3 fixed), ``fx == fy`` (fixed aspect ratio), principal
point free, bundled intrinsics across all views (two-stage: intrinsics from every
view, then fix intrinsics and solve the measurement pose). RMS reported in pixels.

Lineage: the ray-plane back-projection generalises
``global_coordinate_alignment._pixels_to_world_mm`` (Z=0) to an arbitrary tilted
sheet plane; the projection Jacobian mirrors
``stereo_reconstruction_production._compute_projection_jacobian``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Tuple

import cv2
import numpy as np


# OpenCV LM termination: match the existing stereo calibration code.
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-5)


class DistortionModel(str, Enum):
    """Supported distortion models. STANDARD is the DaVis-matching default."""

    STANDARD = "standard"   # k1, k2, p1, p2, k3  (k3 fixed to 0 -> DaVis 4-coeff)
    RATIONAL = "rational"   # + k4, k5, k6 (wide-angle only)
    TILTED = "tilted"       # + tauX, tauY (Scheimpflug); opt-in only


@dataclass
class CameraModel:
    """A fitted pinhole camera in the user-defined world frame.

    Attributes
    ----------
    K : (3,3) intrinsic matrix
    dist : OpenCV distortion vector (length 5 for STANDARD/`[k1,k2,p1,p2,k3]`,
        14 for TILTED). For STANDARD the k3 slot is 0.
    R : (3,3) rotation, world -> camera
    t : (3,1) translation, world -> camera
    image_size : (width, height) in pixels
    distortion_model : which model was fitted
    rms : overall reprojection RMS (pixels) from the intrinsic fit
    """

    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_size: Tuple[int, int]
    distortion_model: DistortionModel = DistortionModel.STANDARD
    rms: float = float("nan")

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3, 1)
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        if not isinstance(self.distortion_model, DistortionModel):
            self.distortion_model = DistortionModel(str(self.distortion_model))

    @property
    def rvec(self) -> np.ndarray:
        r, _ = cv2.Rodrigues(self.R)
        return r.reshape(3)

    def project(self, world_pts: np.ndarray) -> np.ndarray:
        """Project world points (N,3) mm -> image-down pixels (N,2)."""
        wp = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
        px, _ = cv2.projectPoints(
            wp, self.rvec, self.t.reshape(3), self.K, self.dist
        )
        return px.reshape(-1, 2)

    def back_project_to_plane(
        self,
        pts_px: np.ndarray,
        z_world: float = 0.0,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
    ) -> np.ndarray:
        return back_project_to_plane(self, pts_px, z_world, tilt_x, tilt_y)

    def jacobian(self, world_pts: np.ndarray, delta: float = 0.01) -> np.ndarray:
        return projection_jacobian(self, world_pts, delta)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def distortion_flags(
    model: DistortionModel,
    fix_aspect_ratio: bool = True,
    fix_k3: bool = True,
) -> int:
    """Build the OpenCV calibration flag bitmask for a distortion model.

    Defaults reproduce DaVis: ``CALIB_FIX_ASPECT_RATIO`` (fx==fy) and, for the
    STANDARD model, ``CALIB_FIX_K3`` (4-coefficient k1,k2,p1,p2). Principal point
    is left free (DaVis solved it off-centre). Tangential p1,p2 stay free.
    """
    flags = 0
    if fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if model == DistortionModel.STANDARD and fix_k3:
        flags |= cv2.CALIB_FIX_K3
    elif model == DistortionModel.RATIONAL:
        flags |= cv2.CALIB_RATIONAL_MODEL
    elif model == DistortionModel.TILTED:
        flags |= cv2.CALIB_TILTED_MODEL
    return flags


def _initial_camera_matrix(image_size: Tuple[int, int]) -> np.ndarray:
    """Seed K with fx == fy so CALIB_FIX_ASPECT_RATIO locks the ratio to 1."""
    w, h = int(image_size[0]), int(image_size[1])
    f0 = float(max(w, h))
    return np.array(
        [[f0, 0.0, w / 2.0], [0.0, f0, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def per_view_rms(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
) -> List[float]:
    """RMS reprojection error (px) for each view."""
    out: List[float] = []
    for o, i, r, t in zip(object_points, image_points, rvecs, tvecs):
        out.append(reprojection_rms(o, i, K, dist, r, t))
    return out


def reprojection_rms(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    """RMS reprojection error (px) for a single (rvec, tvec)."""
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    proj, _ = cv2.projectPoints(obj, np.asarray(rvec, np.float64).reshape(3),
                                np.asarray(tvec, np.float64).reshape(3), K, dist)
    proj = proj.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))


def fit_intrinsics(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: Tuple[int, int],
    distortion_model: DistortionModel = DistortionModel.STANDARD,
    fix_aspect_ratio: bool = True,
    fix_k3: bool = True,
    use_release_object: bool = False,
):
    """Stage-1 bundled intrinsic fit across all views (DaVis-matching).

    Parameters
    ----------
    object_points : list of (M,3) arrays, WORLD-frame mm (user frame), one per view
    image_points : list of (M,2) arrays, image-down pixels, one per view
    image_size : (width, height)
    distortion_model, fix_aspect_ratio, fix_k3 : fit policy (defaults = DaVis)
    use_release_object : if True and >=3 views, use ``cv2.calibrateCameraRO``
        (Strobl-Hirzinger release-object, better for planar dot grids)

    Returns
    -------
    (K, dist, rvecs, tvecs, rms, per_view_rms_list)
        ``rms`` is the overall reprojection RMS in px (DaVis FitError analogue).
    """
    objp = [np.asarray(o, dtype=np.float32).reshape(-1, 3) for o in object_points]
    imgp = [np.asarray(i, dtype=np.float32).reshape(-1, 2) for i in image_points]
    if len(objp) == 0:
        raise ValueError("fit_intrinsics: no views provided")

    # Seed K with fx==fy so CALIB_FIX_ASPECT_RATIO locks the ratio to 1. We do NOT
    # set CALIB_USE_INTRINSIC_GUESS: with FIX_ASPECT_RATIO and no guess, OpenCV
    # takes only the fx/fy ratio from K0 and estimates the actual focal length from
    # the view homographies (robust), rather than starting LM from a guessed focal.
    K0 = _initial_camera_matrix(image_size)
    dist0 = np.zeros(5, dtype=np.float64)
    flags = distortion_flags(distortion_model, fix_aspect_ratio, fix_k3)
    image_size = (int(image_size[0]), int(image_size[1]))

    if use_release_object and len(objp) >= 3:
        # Fix the last object point of the (planar) board as the reference.
        i_fixed = objp[0].shape[0] - 1
        rms, K, dist, rvecs, tvecs, _new_obj = cv2.calibrateCameraRO(
            objp, imgp, image_size, i_fixed, K0, dist0,
            flags=flags, criteria=CRITERIA,
        )
    else:
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objp, imgp, image_size, K0, dist0,
            flags=flags, criteria=CRITERIA,
        )

    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    rvecs = [np.asarray(r, dtype=np.float64).reshape(3) for r in rvecs]
    tvecs = [np.asarray(t, dtype=np.float64).reshape(3) for t in tvecs]
    pv = per_view_rms(objp, imgp, K, dist, rvecs, tvecs)
    return K, dist, rvecs, tvecs, float(rms), pv


def fit_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    planar: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stage-2 pose with intrinsics fixed (world -> camera R, t).

    ``planar=True`` uses ``SOLVEPNP_IPPE`` (board points coplanar in their own
    frame); otherwise ``SOLVEPNP_SQPNP`` (general 3D, e.g. stepped). Refined with
    Levenberg-Marquardt.
    """
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)

    # IPPE requires planar object points; if z varies, force SQPNP.
    if planar and not np.allclose(obj[:, 0, 2], obj[0, 0, 2], atol=1e-6):
        planar = False

    flag = cv2.SOLVEPNP_IPPE if planar else cv2.SOLVEPNP_SQPNP
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flag)
    except cv2.error:
        ok = False
        rvec = tvec = None
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed for all methods")

    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return R, np.asarray(tvec, dtype=np.float64).reshape(3, 1)


# ---------------------------------------------------------------------------
# World <-> pixel
# ---------------------------------------------------------------------------

def back_project_to_plane(
    camera: CameraModel,
    pts_px: np.ndarray,
    z_world: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
) -> np.ndarray:
    """Back-project image-down pixels onto a world plane via ray-plane intersection.

    The plane is ``Z = z_world + X*tan(tilt_y) + Y*tan(tilt_x)`` (the laser sheet;
    z_world=tilt=0 is the Z=0 plane). Returns (N,3) world mm. NaN rows where the
    ray is parallel to the plane. Generalises
    ``global_coordinate_alignment._pixels_to_world_mm``.
    """
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    norm = cv2.undistortPoints(
        pts.reshape(-1, 1, 2), camera.K, camera.dist, P=None
    ).reshape(-1, 2)

    R = camera.R
    R_inv = R.T
    t_world = R_inv @ camera.t.reshape(3)  # (3,) camera centre = -t_world

    n = norm.shape[0]
    rays = np.empty((n, 3), dtype=np.float64)
    rays[:, :2] = norm
    rays[:, 2] = 1.0

    tan_tx = math.tan(tilt_x)
    tan_ty = math.tan(tilt_y)
    # Points whose undistorted ray diverges (far outside the image) produce
    # non-finite intermediate values; these resolve to NaN and are filtered by
    # callers, so silence the expected overflow/invalid warnings.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rays_world = rays @ R_inv.T  # (N,3)
        denom = rays_world[:, 2] - rays_world[:, 0] * tan_ty - rays_world[:, 1] * tan_tx
        numer = z_world + t_world[2] - t_world[0] * tan_ty - t_world[1] * tan_tx
        s = np.full(n, np.nan, dtype=np.float64)
        valid = np.isfinite(denom) & (np.abs(denom) >= 1e-12)
        s[valid] = numer / denom[valid]
        world3 = s[:, None] * rays_world - t_world
    world3[~np.isfinite(world3).all(axis=1)] = np.nan
    return world3


def projection_jacobian(
    camera: CameraModel, world_pts: np.ndarray, delta: float = 0.01
) -> np.ndarray:
    """Numerical d(pixel)/d(world) Jacobian, shape (N,2,3), via central differences.

    Mirrors ``stereo_reconstruction_production._compute_projection_jacobian``: six
    vectorised ``projectPoints`` calls (±delta on X, Y, Z).
    """
    wp = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
    n = wp.shape[0]
    jac = np.zeros((n, 2, 3), dtype=np.float64)
    rvec = camera.rvec
    tvec = camera.t.reshape(3)
    for ax in range(3):
        plus = wp.copy()
        plus[:, ax] += delta
        minus = wp.copy()
        minus[:, ax] -= delta
        a, _ = cv2.projectPoints(plus, rvec, tvec, camera.K, camera.dist)
        b, _ = cv2.projectPoints(minus, rvec, tvec, camera.K, camera.dist)
        jac[:, :, ax] = (a.reshape(-1, 2) - b.reshape(-1, 2)) / (2.0 * delta)
    return jac
