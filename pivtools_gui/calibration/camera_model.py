"""calibration.camera_model — pinhole camera model + DaVis-matching fit.

The single camera-model family for calibration. Pinhole only. All world<->pixel
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


@dataclass
class PolynomialModel:
    """A single-plane 3rd-order polynomial map image-px -> world-mm (planar only).

    A direct degree-3 bivariate map fitted on the datum plane, the alternative to
    the 3D pinhole model. Two coefficient sets (X_mm, Y_mm), each a 10-term cubic in
    normalised image coords ``s = (x-x0)/sx``, ``t = (y-y0)/sy``:

        basis(s,t) = [1, s, s^2, s^3, t, t^2, t^3, s*t, s^2*t, s*t^2]

    The targets are world mm in the resolved (clicked) frame, so the origin / +X /
    +Y / origin_mm choice is baked into the coefficients — there is no separate
    R, t, K. It exposes the same ``back_project_to_plane`` signature as
    ``CameraModel`` so the mono apply path (apply.py / runio.py) is shared verbatim;
    ``z_world`` / ``tilt_*`` are accepted and ignored (the fit is for one plane).

    No ``project`` / ``jacobian`` — those are stereo-only, and polynomial is planar
    only.
    """

    coeffs_x: np.ndarray   # (10,)
    coeffs_y: np.ndarray   # (10,)
    x0: float              # normalisation: s = (x - x0) / sx
    sx: float
    y0: float              # normalisation: t = (y - y0) / sy
    sy: float
    image_size: Tuple[int, int]
    rms_x_mm: float = float("nan")
    rms_y_mm: float = float("nan")
    model_type: str = "polynomial"

    def __post_init__(self) -> None:
        self.coeffs_x = np.asarray(self.coeffs_x, dtype=np.float64).reshape(-1)
        self.coeffs_y = np.asarray(self.coeffs_y, dtype=np.float64).reshape(-1)
        self.x0 = float(self.x0)
        self.sx = float(self.sx)
        self.y0 = float(self.y0)
        self.sy = float(self.sy)
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))

    def back_project_to_plane(
        self,
        pts_px: np.ndarray,
        z_world: float = 0.0,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
    ) -> np.ndarray:
        """Evaluate the polynomial: image-down pixels (N,2) -> world (X,Y,0) mm.

        ``z_world`` / ``tilt_x`` / ``tilt_y`` are accepted for signature parity with
        ``CameraModel`` and ignored (the polynomial is a single-plane map). Unlike
        the pinhole ray-plane intersection this never returns NaN — a polynomial
        extrapolates silently outside the fit region.
        """
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        s = (pts[:, 0] - self.x0) / self.sx
        t = (pts[:, 1] - self.y0) / self.sy
        basis = _poly_basis(s, t)
        # The inputs are finite, so the result is finite; silence the spurious
        # matmul over/invalid warnings some BLAS backends (macOS Accelerate) raise,
        # matching the errstate guard in the pinhole back_project_to_plane above.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            wx = basis @ self.coeffs_x
            wy = basis @ self.coeffs_y
        return np.column_stack([wx, wy, np.zeros(len(wx))])


@dataclass
class Polynomial3DModel:
    """A single-view 3rd-order 3D polynomial map world-mm -> image-px (DaVis Polynomial3rdOrder).

    The DaVis ``poly`` model, and the stepped board's single-datum-view alternative to
    the multi-view pinhole fit. UNLIKE the single-plane ``PolynomialModel`` (which maps
    image->world on one plane and is mono-only), this maps the OPPOSITE direction,
    world ``(X, Y, Z)`` mm -> image ``(u, v)`` px, so it exposes ``project`` and
    ``jacobian`` and therefore supports stereo 3C reconstruction. The Z sensitivity
    comes from the board's two physical planes (clicked level Z=0, other level
    Z=-step), so the fit is degree-3 in ``(X, Y)`` x degree-1 in ``Z`` -- a 20-term
    basis built from the 10-term cubic ``_poly_basis`` and that same cubic times ``r``::

        phi(p,q,r) = [ cubic10(p,q) , cubic10(p,q)*r ]                 (20 terms)
        cubic10(p,q) = [1,p,p^2,p^3,q,q^2,q^3,p*q,p^2*q,p*q^2]         (== _poly_basis)

    on world coords normalised by the datum board extent
    ``p=(X-x0)/sx, q=(Y-y0)/sy, r=(Z-z0)/sz`` with ``z0,sz`` placing the two planes at
    ``r = +-1``. Two coefficient sets (``u``, ``v``) absorb the clicked origin / +X /
    +Y, so there is no ``R, t, K``.

    ``back_project_to_plane`` inverts the forward map at a fixed sheet (a polynomial is
    not analytically invertible) by 2D Newton iteration on ``(X, Y)``; ``z_world`` /
    ``tilt_x`` / ``tilt_y`` define the sheet ``z = z_world + tan(tx)*X + tan(ty)*Y``.

    ``world_z_toward_camera`` is the +-1 the stereo W convention needs in place of the
    pinhole ``stereo_model.camera_z_sign`` -- a polynomial has no camera centre to test.
    +1 means world +Z already points toward the cameras, which is the stepped
    clicked-level convention (the dotted peak face Z=0 is nearer the camera than the
    trough Z=-step).
    """

    coeffs_u: np.ndarray   # (20,) normalised-world basis -> image u (px)
    coeffs_v: np.ndarray   # (20,) normalised-world basis -> image v (px)
    x0: float              # world normalisation: p = (X - x0) / sx
    sx: float
    y0: float              # q = (Y - y0) / sy
    sy: float
    z0: float              # r = (Z - z0) / sz  (the two planes land at r = +-1)
    sz: float
    image_size: Tuple[int, int]
    rms_px: float = float("nan")
    plane_rms_px: Tuple[float, ...] = ()   # per-plane reprojection RMS (px), Z-ascending
    world_z_toward_camera: float = 1.0
    model_type: str = "polynomial3d"

    def __post_init__(self) -> None:
        self.coeffs_u = np.asarray(self.coeffs_u, dtype=np.float64).reshape(-1)
        self.coeffs_v = np.asarray(self.coeffs_v, dtype=np.float64).reshape(-1)
        for name in ("x0", "sx", "y0", "sy", "z0", "sz"):
            setattr(self, name, float(getattr(self, name)))
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        self.plane_rms_px = tuple(float(v) for v in self.plane_rms_px)
        self.world_z_toward_camera = float(self.world_z_toward_camera)

    def _norm(self, world_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        wp = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
        return ((wp[:, 0] - self.x0) / self.sx,
                (wp[:, 1] - self.y0) / self.sy,
                (wp[:, 2] - self.z0) / self.sz)

    def project(self, world_pts: np.ndarray) -> np.ndarray:
        """Project world points (N,3) mm -> image-down pixels (N,2)."""
        wp = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
        if wp.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        p, q, r = self._norm(wp)
        basis = _poly_basis_3d(p, q, r)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            u = basis @ self.coeffs_u
            v = basis @ self.coeffs_v
        return np.column_stack([u, v])

    def jacobian(self, world_pts: np.ndarray, delta: float = 0.01) -> np.ndarray:
        """Analytic d(u,v)/d(X,Y,Z) -> (N,2,3). ``delta`` is ignored (parity with pinhole).

        The Z column is the out-of-plane (W) sensitivity the stereo 4x3 solve needs;
        for a stepped board it is weak (only the 3 mm step constrains it), which is the
        documented short-baseline poly characteristic.
        """
        wp = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
        if wp.size == 0:
            return np.empty((0, 2, 3), dtype=np.float64)
        p, q, r = self._norm(wp)
        dphi_dp, dphi_dq, dphi_dr = _poly_basis_3d_grad(p, q, r)
        out = np.empty((len(wp), 2, 3), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for axis, coeffs in ((0, self.coeffs_u), (1, self.coeffs_v)):
                out[:, axis, 0] = (dphi_dp @ coeffs) / self.sx
                out[:, axis, 1] = (dphi_dq @ coeffs) / self.sy
                out[:, axis, 2] = (dphi_dr @ coeffs) / self.sz
        return out

    def back_project_to_plane(
        self,
        pts_px: np.ndarray,
        z_world: float = 0.0,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
    ) -> np.ndarray:
        """Image-down pixels (N,2) -> world (X,Y,Z) mm on the sheet, by 2D Newton.

        Inverts the forward cubic at the sheet ``z = z_world + tan(tilt_x)*X +
        tan(tilt_y)*Y``. Each Newton step solves the in-plane 2x2 system (the Z column
        of the Jacobian is folded into d/dX, d/dY via the tilt chain rule). Iterates
        from the board centre; converges in a few steps over the fitted FOV. Like the
        2D ``PolynomialModel`` it extrapolates silently (never NaN) outside the fit
        region -- unlike the pinhole ray-plane intersection.
        """
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        tan_x, tan_y = math.tan(tilt_x), math.tan(tilt_y)
        # Start at the board centre (the normalisation origin in X, Y).
        x = np.full(len(pts), self.x0, dtype=np.float64)
        y = np.full(len(pts), self.y0, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for _ in range(POLY3D_NEWTON_ITERS):
                z = z_world + tan_x * x + tan_y * y
                wp = np.column_stack([x, y, z])
                res = self.project(wp) - pts            # (N,2)
                jac = self.jacobian(wp)                 # (N,2,3)
                # In-plane derivatives with the sheet-tilt chain rule.
                d_dx = jac[:, :, 0] + jac[:, :, 2] * tan_x   # (N,2) d(u,v)/dX
                d_dy = jac[:, :, 1] + jac[:, :, 2] * tan_y   # (N,2) d(u,v)/dY
                det = d_dx[:, 0] * d_dy[:, 1] - d_dy[:, 0] * d_dx[:, 1]
                det = np.where(np.abs(det) < 1e-12, np.nan, det)
                # Cramer's rule for M @ [dx, dy] = -res.
                dx = (-res[:, 0] * d_dy[:, 1] + d_dy[:, 0] * res[:, 1]) / det
                dy = (-d_dx[:, 0] * res[:, 1] + res[:, 0] * d_dx[:, 1]) / det
                x = x + np.nan_to_num(dx)
                y = y + np.nan_to_num(dy)
                if np.nanmax(np.abs(res)) < POLY3D_NEWTON_TOL_PX:
                    break
        z = z_world + tan_x * x + tan_y * y
        return np.column_stack([x, y, z])


@dataclass
class ScaleFactorModel:
    """A uniform pixel->world scale (the simplest planar map, no board, no fit).

    The world point of an image-down pixel is a single scale about a user-picked
    origin, with axis-aligned signs the user chooses on the image:

        du = px_x - origin_px[0],  dv = px_y - origin_px[1]
        not swap:  X = col_sign*du*mm_per_pixel,  Y = row_sign*dv*mm_per_pixel
            swap:  X = col_sign*dv*mm_per_pixel,  Y = row_sign*du*mm_per_pixel

    This is the same sign algebra as ``world_frame.apply_world_frame`` (col_sign is
    the +X sign, row_sign the +Y sign, swap selects which pixel delta feeds X), so
    the convention matches the board path exactly. ``mm_per_pixel = 1/px_per_mm``;
    fed through ``apply.calibrate_displacements`` it reproduces the v1 scale-factor
    velocity ``disp_px/px_per_mm/dt/1000`` exactly.

    Image-down y means the user's "+Y up" choice is ``row_sign = -1`` (world Y grows
    as the pixel row decreases). It exposes the same ``back_project_to_plane``
    signature as ``CameraModel`` so the mono apply / measure / global-coords paths
    are shared verbatim; ``z_world`` / ``tilt_*`` are accepted and ignored (the map
    is a single flat plane). No ``project`` / ``jacobian`` — those are stereo only.
    """

    origin_px: np.ndarray   # (2,) image-down pixel of world (0, 0)
    mm_per_pixel: float
    image_size: Tuple[int, int]
    swap_axes: int = 0
    col_sign: int = 1
    row_sign: int = 1
    model_type: str = "scale_factor"

    def __post_init__(self) -> None:
        self.origin_px = np.asarray(self.origin_px, dtype=np.float64).reshape(2)
        self.mm_per_pixel = float(self.mm_per_pixel)
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        self.swap_axes = int(self.swap_axes)
        self.col_sign = int(self.col_sign)
        self.row_sign = int(self.row_sign)

    def back_project_to_plane(
        self,
        pts_px: np.ndarray,
        z_world: float = 0.0,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
    ) -> np.ndarray:
        """Image-down pixels (N,2) -> world (X,Y,0) mm under the uniform scale.

        ``z_world`` / ``tilt_x`` / ``tilt_y`` are accepted for signature parity with
        ``CameraModel`` and ignored (a single flat plane). Linear, so never NaN.
        """
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        du = pts[:, 0] - self.origin_px[0]
        dv = pts[:, 1] - self.origin_px[1]
        mmpp = self.mm_per_pixel
        if not self.swap_axes:
            wx = self.col_sign * du * mmpp
            wy = self.row_sign * dv * mmpp
        else:
            wx = self.col_sign * dv * mmpp
            wy = self.row_sign * du * mmpp
        return np.column_stack([wx, wy, np.zeros(len(pts))])


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _poly_basis(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build the 10-term cubic design matrix (N,10) from normalised coords s, t.

    Term order: [1, s, s^2, s^3, t, t^2, t^3, s*t, s^2*t, s*t^2].
    """
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    s2 = s * s
    s3 = s2 * s
    t2 = t * t
    t3 = t2 * t
    return np.column_stack(
        [np.ones_like(s), s, s2, s3, t, t2, t3, s * t, s2 * t, s * t2]
    )


def fit_polynomial(
    image_points: np.ndarray,
    world_xy: np.ndarray,
    image_size: Tuple[int, int],
) -> PolynomialModel:
    """Fit a single-plane 3rd-order polynomial image-px -> world-mm by least squares.

    Parameters
    ----------
    image_points : (N,2) image-down pixels (detector convention, 0-based y-down).
    world_xy : (N,2) or (N,3) world mm in the resolved frame (only X, Y are used).
        Comes from ``world_frame.apply_world_frame`` so the clicked origin / +X / +Y
        / origin_mm is already encoded in the targets.
    image_size : (width, height); normalisation is centred (x0,y0 = w/2,h/2) and
        scaled by the half-dimensions, so s, t are ~[-1, 1].

    Returns
    -------
    PolynomialModel with per-axis RMS fit residual in mm.
    """
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    world = np.asarray(world_xy, dtype=np.float64).reshape(len(img), -1)[:, :2]
    n = img.shape[0]
    if n < 10:
        raise RuntimeError(
            f"polynomial fit needs >=10 points (10 coeffs per axis), got {n}"
        )
    w, h = int(image_size[0]), int(image_size[1])
    x0, y0 = w / 2.0, h / 2.0
    sx, sy = w / 2.0, h / 2.0
    s = (img[:, 0] - x0) / sx
    t = (img[:, 1] - y0) / sy
    a = _poly_basis(s, t)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cx, _, _, _ = np.linalg.lstsq(a, world[:, 0], rcond=None)
        cy, _, _, _ = np.linalg.lstsq(a, world[:, 1], rcond=None)
        rms_x = float(np.sqrt(np.mean((a @ cx - world[:, 0]) ** 2)))
        rms_y = float(np.sqrt(np.mean((a @ cy - world[:, 1]) ** 2)))
    return PolynomialModel(
        coeffs_x=cx, coeffs_y=cy, x0=x0, sx=sx, y0=y0, sy=sy,
        image_size=(w, h), rms_x_mm=rms_x, rms_y_mm=rms_y,
    )

# --- 3D polynomial (world -> image), single datum view ---------------------

# Newton inversion for Polynomial3DModel.back_project_to_plane.
POLY3D_NEWTON_ITERS = 30
POLY3D_NEWTON_TOL_PX = 1e-6


def _poly_basis_3d(s: np.ndarray, t: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Build the 20-term 3D cubic design matrix (N,20) from normalised s, t, r.

    Term order: the 10 ``_poly_basis(s,t)`` terms, then those same 10 multiplied by
    ``r`` (degree-1 in z). So columns 0..9 are the in-plane cubic and 10..19 are its
    linear-in-z partners.
    """
    base = _poly_basis(s, t)                       # (N,10)
    r = np.asarray(r, dtype=np.float64).reshape(-1, 1)
    return np.concatenate([base, base * r], axis=1)  # (N,20)


def _poly_basis_3d_grad(
    s: np.ndarray, t: np.ndarray, r: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partials of the 20-term basis wrt normalised (s,t,r); each return is (N,20).

    Matches the term order of ``_poly_basis_3d`` exactly. ``d/dr`` is zero on the
    in-plane block and equals the in-plane cubic on the linear-in-z block.
    """
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    r = np.asarray(r, dtype=np.float64).reshape(-1, 1)
    zeros = np.zeros_like(s)
    ones = np.ones_like(s)
    s2, t2 = s * s, t * t
    # d/ds and d/dt of cubic10 = [1,s,s^2,s^3,t,t^2,t^3,s*t,s^2*t,s*t^2].
    dbase_ds = np.column_stack(
        [zeros, ones, 2 * s, 3 * s2, zeros, zeros, zeros, t, 2 * s * t, t2]
    )
    dbase_dt = np.column_stack(
        [zeros, zeros, zeros, zeros, ones, 2 * t, 3 * t2, s, s2, 2 * s * t]
    )
    base = _poly_basis(s, t)
    dphi_ds = np.concatenate([dbase_ds, dbase_ds * r], axis=1)
    dphi_dt = np.concatenate([dbase_dt, dbase_dt * r], axis=1)
    dphi_dr = np.concatenate([np.zeros_like(base), base], axis=1)
    return dphi_ds, dphi_dt, dphi_dr


def fit_polynomial3d(
    world_pts: np.ndarray,
    image_points: np.ndarray,
    image_size: Tuple[int, int],
    world_z_toward_camera: float = 1.0,
) -> "Polynomial3DModel":
    """Fit a single-view 3D cubic world(X,Y,Z) mm -> image(u,v) px by least squares.

    Parameters
    ----------
    world_pts : (N,3) world mm in the resolved (clicked) frame, spanning >=2 distinct
        Z (the two stepped board levels). Z must vary or the linear-in-z block is
        unidentifiable.
    image_points : (N,2) image-down pixels (detector convention, 0-based y-down).
    image_size : (width, height); stored for the model, not used in the fit (the
        normalisation is on world coords, not pixels).
    world_z_toward_camera : +-1 W-sign convention carried to stereo reconstruction.

    Returns
    -------
    Polynomial3DModel with overall per-point reprojection RMS (px, the cv2 convention
    ``sqrt(mean(du^2+dv^2))``) and per-plane RMS in Z-ascending order.
    """
    wp = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(wp) != len(img):
        raise ValueError(
            f"world_pts ({len(wp)}) and image_points ({len(img)}) length mismatch"
        )
    n = len(wp)
    if n < 20:
        raise RuntimeError(
            f"3D polynomial fit needs >=20 points (20 coeffs per axis), got {n}"
        )
    zs = np.unique(np.round(wp[:, 2], 6))
    if zs.size < 2:
        raise RuntimeError(
            f"3D polynomial fit needs >=2 distinct Z planes, got {zs.size}. The stepped "
            f"datum view must include both the peak and trough levels."
        )
    # Normalise on the board extent; z0,sz place the two planes symmetrically at r=+-1.
    x0, y0 = float(np.mean(wp[:, 0])), float(np.mean(wp[:, 1]))
    sx = float(np.ptp(wp[:, 0])) / 2.0 or 1.0
    sy = float(np.ptp(wp[:, 1])) / 2.0 or 1.0
    z0 = float(zs.min() + zs.max()) / 2.0
    sz = float(zs.max() - zs.min()) / 2.0 or 1.0
    s = (wp[:, 0] - x0) / sx
    t = (wp[:, 1] - y0) / sy
    r = (wp[:, 2] - z0) / sz
    a = _poly_basis_3d(s, t, r)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cu, _, _, _ = np.linalg.lstsq(a, img[:, 0], rcond=None)
        cv, _, _, _ = np.linalg.lstsq(a, img[:, 1], rcond=None)
        sq = (a @ cu - img[:, 0]) ** 2 + (a @ cv - img[:, 1]) ** 2  # per-point du^2+dv^2
        rms = float(np.sqrt(np.mean(sq)))
    z_round = np.round(wp[:, 2], 6)
    plane_rms = tuple(float(np.sqrt(np.mean(sq[z_round == zv]))) for zv in zs)
    w, h = int(image_size[0]), int(image_size[1])
    return Polynomial3DModel(
        coeffs_u=cu, coeffs_v=cv, x0=x0, sx=sx, y0=y0, sy=sy, z0=z0, sz=sz,
        image_size=(w, h), rms_px=rms, plane_rms_px=plane_rms,
        world_z_toward_camera=float(world_z_toward_camera),
    )


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
