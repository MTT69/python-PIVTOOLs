"""calibration.joint — joint multi-camera calibration over one shared released board.

This is the DaVis-matching solve. Every camera observes the SAME rigid board (each its own
part), tied together by the global grid (``global_grid.resolve_global_grid``). We solve, in
one shared world frame: per-camera intrinsics ``(K_c, dist_c)``, ONE rigid-rig pose per camera,
ONE board pose per view, and ONE released board ``{P_g}`` that all cameras agree on.

The rig is rigid and the board moves: ``pose(c, v) = rig_c o board_v``, i.e.
``R(c,v) = R_c R_v`` and ``t(c,v) = R_c t_v + t_c``. A free 6-DOF pose per ``(camera, view)``
would describe camera motions that cannot happen and absorb corner noise into the reported
camera positions while the reprojection residual sits on the noise floor (measured: 6.29 mm
baseline scatter at 0.3 px noise on a 3-camera rig, 389x the rigid-rig value at the same rms).

Gauge: the factorisation is ambiguous under ``rig_c -> rig_c o G``, ``board_v -> G^-1 o board_v``.
It is pinned by ``board_pose[datum_view] = (I, 0)`` and excluding that pose from the parameter
vector, so the world frame IS the datum board plane -- the frame ``apply.py`` back-projects onto
at ``z_world = 0``, the wizard's clicked origin, and ``stereo_model``'s camera-height sign test.
Under that pin ``pose(c, datum) = rig_c``, so the per-camera model is the rig pose itself. The
datum view only has to exist in the grid; every camera need not observe it, which is what lets a
traverse chain (no view common to all cameras) solve. What IS required is that the camera graph
linked by shared views is connected -- otherwise a component's rig pose is unobservable, and the
solve raises rather than returning a rank-deficient answer.

Why this converges where a from-scratch monolithic bundle did not: we do not start the joint
bundle from a flat board with free intrinsics. Instead:

  1. BOOTSTRAP per-camera intrinsics with ``cv2.calibrateCameraRO`` on that camera's global
     common core (dots it sees in every one of its views); starved cameras fall back to plain
     ``calibrateCamera`` on per-view subsets (logged, never silent).
  2. ALTERNATE, intrinsics held: triangulate each dot seen by >= 2 rays from all cameras/views,
     gauge-fix the board to the world frame, re-solve every pose. This is a robust coordinate
     descent that lands the board + poses very close to the optimum. Its poses are still free
     per ``(camera, view)``; they are only a seed.
  3. FACTORISE the seed into rig poses and board poses (``_factorise_rig_seed``), then run the
     FINAL JOINT BUNDLE (the DaVis-equivalent step): one sparse least-squares refining
     intrinsics + board + rig poses + board poses together. A 3-anchor gauge removes the
     board-point similarity freedom. The bundle runs under every ``board_release`` mode -- under
     ``none`` it refines intrinsics and poses over a fixed board. Guarded: if it does not reduce
     the RMS of the rigid seed, the rigid seed is kept and the regression is logged. The
     factorisation is lossy, so the bundle's RMS is expected to sit slightly ABOVE
     ``rms_after_alternation``: fewer parameters cannot fit noise with fictitious camera motion.

Board release modes (``board_release``):
  - ``full3d`` (default, DaVis-literal): release all three coordinates of each triangulated dot.
  - ``z_only``: lock the in-plane (x, y) to the certified grid, release only the out-of-plane
    bow (z). More physical, far better conditioned; kept for comparison.
  - ``none``: do not release — a joint flat-board calibration (regression anchor / baseline).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from .camera_model import (
    CameraModel,
    DistortionModel,
    PolynomialModel,
    fit_intrinsics,
    fit_polynomial,
    fit_pose,
)
from .detection.base import DetectionResult
from .record import WorldFrame

log = logging.getLogger(__name__)

ViewKey = Tuple[int, int]
_MIN_COMMON_CORE = 6  # dots a camera must see in ALL its views to use calibrateCameraRO
_MIN_VIEW_DOTS = 4  # dots a view must have to contribute a pose
_MIN_RELEASE_RAYS = 2  # a dot is triangulable (released) only if seen by >= 2 rays
_MAX_ALT_ITERS = 25
_CONV_TOL_MM = 1e-4
_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-11)


@dataclass
class JointResult:
    """Output of the joint solve, in one shared world frame."""

    cameras: List[int]
    models: Dict[int, CameraModel]  # per-camera K, dist, rig pose (world = datum board plane)
    board: Dict[Tuple[int, int], np.ndarray]  # global index -> released (x,y,z) mm
    view_poses: Dict[ViewKey, Tuple[np.ndarray, np.ndarray]]  # (cam,view) -> rig_c o board_v
    world_frame: WorldFrame
    spacing_mm: float
    board_release: str
    rms_px: float  # overall reprojection RMS
    per_camera_rms: Dict[int, float]
    cross_camera_board_agreement_mm: float  # 0 by construction (one shared board)
    converged: bool
    info: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-frame camera centre C from world->camera (R, t):  X_cam = R X + t -> C = -R^T t."""
    return (-R.T @ np.asarray(t, dtype=np.float64).reshape(3, 1)).reshape(3)


def _pixel_rays_world(
    pixels: np.ndarray, K: np.ndarray, dist: np.ndarray, R: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (origin (3,), directions (M,3) unit) of world-frame rays through ``pixels``."""
    und = cv2.undistortPoints(
        np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2), K, dist
    ).reshape(-1, 2)
    cam = np.column_stack([und, np.ones(len(und))])  # normalised camera rays
    world = cam @ R  # R^T applied row-wise: (R^T d)^T = d^T R
    world /= np.linalg.norm(world, axis=1, keepdims=True) + 1e-12
    return _camera_center(R, t), world


def _triangulate(origins: np.ndarray, dirs: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Least-squares intersection of rays {(o_i, d_i)} (d_i unit): min sum |(I - d d^T)(p-o)|^2.

    Returns (point, ok); ok is False when the rays are near-parallel (ill-conditioned A), so
    the caller can leave that dot at its current estimate rather than solving garbage.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    if np.linalg.cond(A) > 1e8:
        return np.zeros(3), False
    return np.linalg.solve(A, b), True


def _kabsch(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Best rigid (rotation+translation, scale fixed = 1) mapping A -> B. Returns (R, t)."""
    mA, mB = A.mean(0), B.mean(0)
    H = (A - mA).T @ (B - mB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, mB - R @ mA


def _deplane_z(P: np.ndarray) -> np.ndarray:
    """Out-of-plane z with the best-fit plane removed (fixes the z offset+tilt gauge)."""
    A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
    coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    return P[:, 2] - A @ coef


def _gauge_anchor_rows(nominal: np.ndarray, candidate_rows: np.ndarray) -> List[int]:
    """Three non-collinear, widely-spread rows to hold fixed (removes the similarity gauge)."""
    xy = nominal[candidate_rows, :2]
    a = candidate_rows[int(np.argmin(xy[:, 0] + xy[:, 1]))]
    b = candidate_rows[int(np.argmax(xy[:, 0] + xy[:, 1]))]
    pa, pb = nominal[a, :2], nominal[b, :2]
    line = (pb - pa) / (np.linalg.norm(pb - pa) + 1e-9)
    perp = np.abs((xy - pa) @ np.array([-line[1], line[0]]))
    c = candidate_rows[int(np.argmax(perp))]
    return [int(a), int(b), int(c)]


# ---------------------------------------------------------------------------
# Rigid rig: pose(c, v) = rig_c o board_v
# ---------------------------------------------------------------------------

Pose = Tuple[np.ndarray, np.ndarray]  # (R (3,3), t (3,1)) world -> camera, X_cam = R X + t


def _compose(a: Pose, b: Pose) -> Pose:
    """``a o b``: apply ``b`` then ``a``. ``X -> R_a (R_b X + t_b) + t_a``."""
    Ra, ta = a[0], np.asarray(a[1], dtype=np.float64).reshape(3, 1)
    Rb, tb = b[0], np.asarray(b[1], dtype=np.float64).reshape(3, 1)
    return Ra @ Rb, Ra @ tb + ta


def _invert(p: Pose) -> Pose:
    R, t = p[0], np.asarray(p[1], dtype=np.float64).reshape(3, 1)
    return R.T, -R.T @ t


def _mean_rotation(Rs: Sequence[np.ndarray]) -> np.ndarray:
    """Chordal mean: the SO(3) projection of the summed matrices (Frobenius-nearest rotation)."""
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def _camera_components(view_keys: Sequence[ViewKey], cams: Sequence[int]) -> List[List[int]]:
    """Connected components of the camera graph whose edges are shared views."""
    by_view: Dict[int, set] = {}
    for c, v in view_keys:
        by_view.setdefault(v, set()).add(c)
    adj: Dict[int, set] = {c: set() for c in cams}
    for members in by_view.values():
        for a in members:
            adj[a] |= members - {a}
    seen: set = set()
    comps: List[List[int]] = []
    for c in cams:
        if c in seen:
            continue
        comp, stack = set(), [c]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adj[x] - comp)
        seen |= comp
        comps.append(sorted(comp))
    return comps


def _check_rig_connectivity(view_keys: Sequence[ViewKey], cams: Sequence[int]) -> None:
    """Every rig pose must be observable: the shared-view camera graph must be connected.

    A camera that shares no view with the rest has a rig pose the observations cannot tie to
    the others' -- the bundle Jacobian is rank-deficient by exactly that camera's 6 DOF. That
    is a data defect the user must fix (add a view both sides see), never something to solve
    around.
    """
    comps = _camera_components(view_keys, cams)
    if len(comps) > 1:
        raise ValueError(
            f"joint: the cameras do not form one rigid rig in the data -- no chain of shared "
            f"views links the groups {comps}; each group's position relative to the others is "
            f"unobservable. Add board positions seen by cameras on both sides."
        )


def _factorise_rig_seed(
    pose_by_view: Dict[ViewKey, Pose],
    cams: Sequence[int],
    datum_camera: int,
    datum_view: int,
) -> Tuple[Dict[int, Pose], Dict[int, Pose]]:
    """Split free per-``(camera, view)`` poses into ``rig_c o board_v`` with ``board_datum = I``.

    Seed for the rigid bundle. The free poses are not exactly rigid (they absorbed noise), so
    this is lossy by construction; averaging over shared views keeps the loss small and the
    bundle removes the rest.

    Grows from ``datum_camera`` (which observes ``datum_view`` -- checked by the caller) one
    camera at a time, always taking the unplaced camera with the most views shared with placed
    cameras (best-conditioned edge first). For camera ``c`` and each shared ``(p, v)``:
    ``rel = pose(c,v) o pose(p,v)^-1`` and ``rig_c = rel o rig_p``; rotations are averaged
    chordally, translations averaged AFTER the mean rotation is fixed (so every translation
    sample is consistent with the same ``R_c``). Board poses are then
    ``board_v = rig_c^-1 o pose(c,v)`` averaged over the cameras observing ``v``. Finally the
    gauge is moved so ``board_datum`` is exactly the identity.
    """
    views_of: Dict[int, set] = {c: set() for c in cams}
    cams_of: Dict[int, set] = {}
    for c, v in pose_by_view:
        views_of[c].add(v)
        cams_of.setdefault(v, set()).add(c)

    rig: Dict[int, Pose] = {datum_camera: pose_by_view[(datum_camera, datum_view)]}
    while len(rig) < len(cams):
        best_c, best_shared = None, []
        for c in cams:
            if c in rig:
                continue
            shared = [(p, v) for p in rig for v in sorted(views_of[c] & views_of[p])]
            if len(shared) > len(best_shared):
                best_c, best_shared = c, shared
        if best_c is None:
            raise ValueError(
                "joint: rig factorisation found an unlinked camera -- connectivity check missed it"
            )
        Rs = []
        for p, v in best_shared:
            rel = _compose(pose_by_view[(best_c, v)], _invert(pose_by_view[(p, v)]))
            Rs.append(_compose(rel, rig[p])[0])
        R_c = _mean_rotation(Rs)
        ts = []
        for p, v in best_shared:
            board_v = _compose(_invert(rig[p]), pose_by_view[(p, v)])
            t_cv = np.asarray(pose_by_view[(best_c, v)][1], dtype=np.float64).reshape(3, 1)
            ts.append(t_cv - R_c @ board_v[1])
        rig[best_c] = (R_c, np.mean(ts, axis=0))

    board_pose: Dict[int, Pose] = {}
    for v, members in cams_of.items():
        samples = [_compose(_invert(rig[c]), pose_by_view[(c, v)]) for c in sorted(members)]
        R_v = _mean_rotation([s[0] for s in samples])
        board_pose[v] = (R_v, np.mean([s[1] for s in samples], axis=0))

    # Gauge: rig_c -> rig_c o G, board_v -> G^-1 o board_v with G = board_datum, so the datum
    # board pose becomes the identity exactly (not just to rounding) and world = datum board.
    G = board_pose[datum_view]
    G_inv = _invert(G)
    rig = {c: _compose(p, G) for c, p in rig.items()}
    board_pose = {v: _compose(G_inv, p) for v, p in board_pose.items()}
    board_pose[datum_view] = (np.eye(3), np.zeros((3, 1)))
    return rig, board_pose


# ---------------------------------------------------------------------------
# Multi-camera bundle adjustment (final joint refinement)
# ---------------------------------------------------------------------------


class _JointBA:
    """Sparse multi-camera release bundle: intrinsics + rig poses + board poses + board.

    Parameter vector =
        [ (fx,cx,cy,k1,k2,p1,p2) x n_cam, rig (rvec,tvec) x n_cam,
          board (rvec,tvec) x (n_view - 1), free board ].
    The datum view's board pose is the identity and is NOT in the vector (gauge pin). Free board
    entries are 3 per row for ``full3d``, 1 (z) per row for ``z_only``, none for ``none``; gauge
    anchors and never-released dots are held at their current value.
    """

    def __init__(
        self,
        nominal: np.ndarray,
        observations: list,
        cams: Sequence[int],
        view_keys: Sequence[ViewKey],
        free_rows: Sequence[int],
        mode: str,
        datum_view: int,
    ):
        if mode not in ("full3d", "z_only", "none"):
            raise ValueError(f"_JointBA supports full3d|z_only|none, got {mode!r}")
        self.nominal = nominal  # (N,3) current board (used for held rows)
        self.N = nominal.shape[0]
        self.cams = list(cams)
        self.cam_index = {c: i for i, c in enumerate(self.cams)}
        self.view_keys = list(view_keys)
        self.views = sorted({v for _, v in self.view_keys})
        if datum_view not in self.views:
            raise ValueError(f"_JointBA: datum view {datum_view} is not in the grid")
        self.datum_view = datum_view
        self.free_views = [v for v in self.views if v != datum_view]
        self.fview_index = {v: i for i, v in enumerate(self.free_views)}
        self.mode = mode
        self.free_rows = sorted(free_rows)
        self.free_index = {j: i for i, j in enumerate(self.free_rows)}
        self._per_board = {"full3d": 3, "z_only": 1, "none": 0}[mode]
        self._n_intr = 7 * len(self.cams)
        self._n_rig = 6 * len(self.cams)
        self._n_pose = self._n_rig + 6 * len(self.free_views)
        # group observations per view for vectorised projection
        self.obs = observations  # list of (cam, view, row, u, v)
        self.grp_rows: Dict[ViewKey, np.ndarray] = {}
        self.grp_meas: Dict[ViewKey, np.ndarray] = {}
        tmp: Dict[ViewKey, list] = {k: [] for k in self.view_keys}
        for cam, view, row, u, v in observations:
            tmp[(cam, view)].append((row, u, v))
        for k in self.view_keys:
            arr = np.asarray(tmp[k], dtype=np.float64)
            self.grp_rows[k] = arr[:, 0].astype(int)
            self.grp_meas[k] = arr[:, 1:3]

    @property
    def n_params(self) -> int:
        return self._n_intr + self._n_pose + self._per_board * len(self.free_rows)

    def pack(self, K_by_cam, dist_by_cam, rig_by_cam, board_pose_by_view, board):
        intr = []
        for c in self.cams:
            K, d = K_by_cam[c], dist_by_cam[c]
            intr += [K[0, 0], K[0, 2], K[1, 2], d[0], d[1], d[2], d[3]]
        pose = []
        for c in self.cams:
            R, t = rig_by_cam[c]
            pose += list(cv2.Rodrigues(R)[0].ravel()) + list(np.asarray(t).ravel())
        for v in self.free_views:
            R, t = board_pose_by_view[v]
            pose += list(cv2.Rodrigues(R)[0].ravel()) + list(np.asarray(t).ravel())
        if self._per_board == 3:
            free = board[self.free_rows].ravel()
        elif self._per_board == 1:
            free = board[self.free_rows, 2].ravel()
        else:
            free = np.empty(0)
        return np.concatenate([np.asarray(intr), np.asarray(pose), free])

    def _K_dist(self, x, c):
        o = 7 * self.cam_index[c]
        fx, cx, cy = x[o], x[o + 1], x[o + 2]
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])
        dist = np.array([x[o + 3], x[o + 4], x[o + 5], x[o + 6], 0.0])
        return K, dist

    def _rig_col(self, c) -> int:
        return self._n_intr + 6 * self.cam_index[c]

    def _board_col(self, v) -> Optional[int]:
        """Column of view ``v``'s board pose, or ``None`` for the pinned datum view."""
        if v == self.datum_view:
            return None
        return self._n_intr + self._n_rig + 6 * self.fview_index[v]

    def _rig(self, x, c):
        o = self._rig_col(c)
        return x[o : o + 3].reshape(3, 1), x[o + 3 : o + 6].reshape(3, 1)

    def _board_pose(self, x, v):
        o = self._board_col(v)
        if o is None:
            return np.zeros((3, 1)), np.zeros((3, 1))
        return x[o : o + 3].reshape(3, 1), x[o + 3 : o + 6].reshape(3, 1)

    def _pose_with_jac(self, x, k):
        """Composed ``(rvec, tvec)`` of ``rig_c o board_v`` plus its 6x6 Jacobians.

        ``cv2.composeRT(r1, t1, r2, t2)`` applies ``(r1, t1)`` first, so ``(r1, t1)`` is the
        board pose and ``(r2, t2)`` the rig pose: ``R3 = R_c R_v``, ``t3 = R_c t_v + t_c``. It
        returns ``d(r3,t3)/d(r1,t1,r2,t2)`` as eight 3x3 blocks (rows = outputs).
        """
        rc, tc = self._rig(x, k[0])
        rv, tv = self._board_pose(x, k[1])
        r3, t3, dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2 = cv2.composeRT(
            rv, tv, rc, tc
        )
        J_board = np.block([[dr3dr1, dr3dt1], [dt3dr1, dt3dt1]])
        J_rig = np.block([[dr3dr2, dr3dt2], [dt3dr2, dt3dt2]])
        return r3.reshape(3, 1), t3.reshape(3, 1), J_rig, J_board

    def _pose(self, x, k):
        r3, t3, _, _ = self._pose_with_jac(x, k)
        return r3, t3

    def board_from(self, x):
        board = self.nominal.copy()
        b = x[self._n_intr + self._n_pose :]
        if self._per_board == 3:
            board[self.free_rows] = b.reshape(-1, 3)
        elif self._per_board == 1:
            board[self.free_rows, 2] = b
        return board

    def residuals(self, x):
        board = self.board_from(x)
        out = []
        for k in self.view_keys:
            cam = k[0]
            K, dist = self._K_dist(x, cam)
            rvec, tvec = self._pose(x, k)
            proj, _ = cv2.projectPoints(board[self.grp_rows[k]], rvec, tvec, K, dist)
            out.append((proj.reshape(-1, 2) - self.grp_meas[k]).ravel())
        return np.concatenate(out)

    def jac(self, x):
        """Analytic sparse Jacobian from cv2.projectPoints' own derivatives.

        projectPoints returns d(proj)/d[rvec(3), tvec(3), fx, fy, cx, cy, k1, k2, p1, p2, ...]
        for the COMPOSED pose. We map: fx column = (fx + fy) columns (aspect locked, one shared
        param); cx, cy, k1..p2 direct; the composed-pose block is chain-ruled into the rig and
        board-pose blocks through ``composeRT``'s own derivatives (``_pose_with_jac``); and the
        board point Jacobian d(proj)/dP_world = d(proj)/dtvec @ R (since X_cam = R X_world + t,
        so dX_cam/dP = R and dX_cam/dt = I). Far faster and far more convergent than finite
        differences over ~1000 parameters. ``test_calibration_joint_rig.py`` asserts this
        against a finite-difference Jacobian.
        """
        board = self.board_from(x)
        data, ridx, cidx = [], [], []
        r0 = 0
        for k in self.view_keys:
            cam = k[0]
            ci = 7 * self.cam_index[cam]
            K, dist = self._K_dist(x, cam)
            rvec, tvec, J_rig, J_board = self._pose_with_jac(x, k)
            rows = self.grp_rows[k]
            M = len(rows)
            _, jc = cv2.projectPoints(board[rows], rvec, tvec, K, dist)
            jc = np.asarray(jc, dtype=np.float64)  # (2M, ncols)
            R = cv2.Rodrigues(rvec)[0]
            rr = r0 + np.arange(2 * M)

            def add(col, vals):
                data.append(np.asarray(vals).ravel())
                ridx.append(rr)
                cidx.append(np.full(2 * M, col))

            add(ci + 0, jc[:, 6] + jc[:, 7])  # fx (== fy, aspect locked)
            add(ci + 1, jc[:, 8])  # cx
            add(ci + 2, jc[:, 9])  # cy
            add(ci + 3, jc[:, 10])  # k1
            add(ci + 4, jc[:, 11])  # k2
            add(ci + 5, jc[:, 12])  # p1
            add(ci + 6, jc[:, 13])  # p2
            d_rig = jc[:, :6] @ J_rig  # (2M, 6): d(proj)/d(rig rvec, tvec)
            ri = self._rig_col(cam)
            for j in range(6):
                add(ri + j, d_rig[:, j])
            bi = self._board_col(k[1])
            if bi is not None:  # the datum view's board pose is pinned, no columns
                d_board = jc[:, :6] @ J_board  # (2M, 6): d(proj)/d(board rvec, tvec)
                for j in range(6):
                    add(bi + j, d_board[:, j])

            dP = jc[:, 3:6].reshape(M, 2, 3) @ R  # d(proj)/dP_world, (M,2,3)
            for n_i in range(M):
                row = int(rows[n_i])
                if row not in self.free_index:
                    continue
                c = self._n_intr + self._n_pose + self._per_board * self.free_index[row]
                base = r0 + 2 * n_i
                if self._per_board == 3:
                    for cc in range(3):
                        data.append(dP[n_i, :, cc])
                        ridx.append([base, base + 1])
                        cidx.append([c + cc, c + cc])
                else:
                    data.append(dP[n_i, :, 2])
                    ridx.append([base, base + 1])
                    cidx.append([c, c])
            r0 += 2 * M

        from scipy.sparse import coo_matrix

        n = self.n_params
        return coo_matrix(
            (
                np.concatenate([np.asarray(d).ravel() for d in data]),
                (
                    np.concatenate([np.asarray(r).ravel() for r in ridx]),
                    np.concatenate([np.asarray(c).ravel() for c in cidx]),
                ),
            ),
            shape=(2 * len(self.obs), n),
        ).tocsr()

    def rms(self, x):
        res = self.residuals(x)
        return float(np.sqrt((res**2).sum() / (len(res) / 2)))


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def _seed_reproj(
    cam: int,
    views: Sequence[int],
    det_of: Dict[ViewKey, DetectionResult],
    global_of: Dict[ViewKey, np.ndarray],
    row_of: Dict[Tuple[int, int], int],
    nominal: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    """Mean reprojection RMS (px) of a candidate (K, dist) over the camera's views.

    Poses are solved per view against the nominal board; the bow biases every candidate
    equally, so the comparison reflects intrinsic/focal quality — the thing that decides
    which seed lands the downstream bundle in the right basin.
    """
    sq, n = 0.0, 0
    for v in views:
        rows = [row_of[tuple(g)] for g in global_of[(cam, v)]]
        if len(rows) < _MIN_VIEW_DOTS:
            continue
        obj = nominal[rows].astype(np.float64)
        img = det_of[(cam, v)].image_points
        try:
            R, t = fit_pose(obj, img, K, dist, planar=True)
        except RuntimeError:
            return np.inf
        proj, _ = cv2.projectPoints(
            obj, cv2.Rodrigues(R)[0], np.asarray(t).reshape(3), K, dist
        )
        sq += float(((proj.reshape(-1, 2) - img) ** 2).sum())
        n += len(rows)
    return float(np.sqrt(sq / n)) if n else np.inf


def _bootstrap_camera(
    cam: int,
    views: Sequence[int],
    det_of: Dict[ViewKey, DetectionResult],
    global_of: Dict[ViewKey, np.ndarray],
    row_of: Dict[Tuple[int, int], int],
    nominal: np.ndarray,
    image_size: Tuple[int, int],
    distortion_model: DistortionModel,
    fix_aspect_ratio: bool,
    fix_k3: bool,
):
    """Per-camera intrinsics seed for the joint solve, picking the better-conditioned of two.

    The bundle is non-convex and seed-sensitive, and no single seed wins on every rig: the
    all-view common core released by ``calibrateCameraRO`` gives a clean focal on telephoto
    rigs (bailey) but starves on small cores; plain ``calibrateCamera`` on the UNION of dots
    is better-conditioned on normal lenses (merle) but can mis-estimate a long telephoto focal
    from a flat board. So we compute both candidates and keep whichever reprojects better — the
    data picks the basin, not a lens-specific rule. The release itself happens later.
    """
    candidates = []  # (label, K, dist)

    # Candidate A: union, plain calibrateCamera (most dots, best on normal lenses)
    objs, imgs = [], []
    for v in views:
        rows = [row_of[tuple(g)] for g in global_of[(cam, v)]]
        if len(rows) < _MIN_VIEW_DOTS:
            continue
        objs.append(nominal[rows].astype(np.float64))
        imgs.append(det_of[(cam, v)].image_points)
    if len(objs) < 3:
        raise ValueError(
            f"joint: camera {cam} has too few usable views ({len(objs)}) to seed intrinsics"
        )
    Ka, da, *_ = fit_intrinsics(
        objs,
        imgs,
        image_size,
        distortion_model,
        fix_aspect_ratio,
        fix_k3,
        use_release_object=False,
    )
    candidates.append(
        ("calibrateCamera(union=%d dots)" % sum(len(o) for o in objs), Ka, da)
    )

    # Candidate B: all-view common core released by calibrateCameraRO (clean focal on telephoto)
    per_view_rows = [set(row_of[tuple(g)] for g in global_of[(cam, v)]) for v in views]
    common = sorted(set.intersection(*per_view_rows)) if per_view_rows else []
    if len(common) >= _MIN_COMMON_CORE:
        objp = nominal[common].astype(np.float64)
        imgp = []
        for v in views:
            rows = [row_of[tuple(g)] for g in global_of[(cam, v)]]
            idx = {r: i for i, r in enumerate(rows)}
            imgp.append(det_of[(cam, v)].image_points[[idx[r] for r in common]])
        Kb, db, *_ = fit_intrinsics(
            [objp] * len(views),
            imgp,
            image_size,
            distortion_model,
            fix_aspect_ratio,
            fix_k3,
            use_release_object=True,
        )
        candidates.append(("calibrateCameraRO(common-core=%d)" % len(common), Kb, db))

    scored = [
        (
            lab,
            K,
            dist,
            _seed_reproj(cam, views, det_of, global_of, row_of, nominal, K, dist),
        )
        for (lab, K, dist) in candidates
    ]
    scored.sort(key=lambda s: s[3])
    lab, K, dist, r = scored[0]
    if not np.isfinite(r):
        raise ValueError(
            f"joint: camera {cam} — no intrinsic seed produced a finite reprojection"
        )
    return K, dist, lab


def run_joint(
    detections_by_cam: Dict[int, List[DetectionResult]],
    global_index: Dict[ViewKey, np.ndarray],
    spacing_mm: float,
    datum_camera: int,
    datum_view: int,
    origin_mm: Tuple[float, float] = (0.0, 0.0),
    board_release: str = "full3d",
    image_size_by_cam: Optional[Dict[int, Tuple[int, int]]] = None,
    expected_cameras: Optional[Sequence[int]] = None,
    distortion_model: DistortionModel = DistortionModel.STANDARD,
    fix_aspect_ratio: bool = True,
    fix_k3: bool = True,
    max_iters: int = _MAX_ALT_ITERS,
    tol_mm: float = _CONV_TOL_MM,
    run_bundle: bool = True,
) -> JointResult:
    """Solve the joint multi-camera shared-board calibration. See module docstring.

    ``image_size_by_cam`` is the real ``(width, height)`` per camera (callers know it from the
    images) and is REQUIRED — inferring it from dot extents biases the principal-point seed and
    the stored size. ``expected_cameras``, when given, must exactly match the camera set the
    resolved grid covers; this catches a camera silently dropped from the grid (e.g. the GUI
    deselects one, or no observation for it resolved) rather than calibrating a quiet subset.
    Contract: for each (camera, view), ``global_index`` rows correspond one-for-one with that
    detection's ``image_points``.
    """
    if board_release not in ("full3d", "z_only", "none"):
        raise ValueError(
            f"board_release must be full3d|z_only|none, got {board_release!r}"
        )
    if (
        distortion_model != DistortionModel.STANDARD
        or not fix_aspect_ratio
        or not fix_k3
    ):
        raise NotImplementedError(
            "joint solve models the DaVis pinhole only (STANDARD distortion, fx==fy, k3=0); the "
            "final bundle would silently differ for other distortion/aspect settings"
        )
    ox, oy = float(origin_mm[0]), float(origin_mm[1])

    # ---- index the union of global dots ----
    keys = sorted(
        {tuple(int(c) for c in g) for gi in global_index.values() for g in gi}
    )
    row_of = {g: i for i, g in enumerate(keys)}
    N = len(keys)
    nominal = np.array(
        [[g[0] * spacing_mm + ox, g[1] * spacing_mm + oy, 0.0] for g in keys]
    )

    det_of = {k: detections_by_cam[k[0]][k[1]] for k in global_index}
    global_of = {
        k: np.asarray(v, dtype=np.int64).reshape(-1, 2) for k, v in global_index.items()
    }

    # observations: (cam, view, row, u, v)
    observations = []
    for k, gi in global_of.items():
        det = det_of[k]
        if len(gi) != len(det.image_points):
            raise ValueError(
                f"joint: ({k}) global_index has {len(gi)} rows but the detection has "
                f"{len(det.image_points)} image points — they must correspond one-for-one"
            )
        for i in range(len(gi)):
            r = row_of[tuple(int(c) for c in gi[i])]
            u, v = det.image_points[i]
            observations.append((k[0], k[1], r, float(u), float(v)))

    cams = sorted({k[0] for k in global_index})
    if expected_cameras is not None:
        exp = sorted(int(c) for c in expected_cameras)
        if exp != cams:
            raise ValueError(
                f"run_joint: the resolved global grid covers cameras {cams} but {exp} were "
                f"expected — a camera was dropped from the grid (no observations resolved for "
                f"it); fix its anchors/links rather than calibrate a silent subset"
            )
    view_keys = sorted(global_index.keys())
    if (datum_camera, datum_view) not in view_keys:
        raise ValueError(
            f"joint: datum ({datum_camera},{datum_view}) not in the resolved grid"
        )
    # Every camera need NOT observe the datum view (that is what lets a traverse chain solve);
    # what the rigid rig needs is one connected chain of shared views across all cameras.
    _check_rig_connectivity(view_keys, cams)

    # Real image size per camera is REQUIRED (callers know it from the images); inferring from
    # dot extents would bias the principal-point seed and the stored size, so fail loudly.
    if image_size_by_cam is None:
        raise ValueError(
            "run_joint: image_size_by_cam is required (real (width, height) per camera) — "
            "inferring from dot extents biases the principal-point seed and stored image_size"
        )
    missing = [c for c in cams if c not in image_size_by_cam]
    if missing:
        raise ValueError(f"run_joint: image_size_by_cam missing camera(s) {missing}")
    image_size_by_cam = {
        c: (int(image_size_by_cam[c][0]), int(image_size_by_cam[c][1])) for c in cams
    }

    # ---- 1. bootstrap intrinsics per camera ----
    K_by_cam, dist_by_cam, boot_info = {}, {}, {}
    for c in cams:
        views_c = sorted(v for (cc, v) in view_keys if cc == c)
        K, dist, how = _bootstrap_camera(
            c,
            views_c,
            det_of,
            global_of,
            row_of,
            nominal,
            image_size_by_cam[c],
            distortion_model,
            fix_aspect_ratio,
            fix_k3,
        )
        K_by_cam[c], dist_by_cam[c], boot_info[c] = K, dist, how

    # ---- seed board + poses ----
    board = nominal.copy()
    pose_by_view: Dict[ViewKey, Tuple[np.ndarray, np.ndarray]] = {}
    for k in view_keys:
        rows = [row_of[tuple(int(c) for c in g)] for g in global_of[k]]
        R, t = fit_pose(
            board[rows],
            det_of[k].image_points,
            K_by_cam[k[0]],
            dist_by_cam[k[0]],
            planar=True,
        )
        pose_by_view[k] = (R, t)

    # rows observed by >= 2 rays are triangulable (released); others stay nominal
    ray_count = np.zeros(N, int)
    for _, _, r, _, _ in observations:
        ray_count[r] += 1
    released_rows = np.array([r for r in range(N) if ray_count[r] >= _MIN_RELEASE_RAYS])

    rows_of_view = {
        k: [row_of[tuple(int(c) for c in g)] for g in global_of[k]] for k in view_keys
    }

    def _overall_rms(Kc, dc, poses, brd) -> float:
        sq, n = 0.0, 0
        for k in view_keys:
            R, t = poses[k]
            proj, _ = cv2.projectPoints(
                brd[rows_of_view[k]],
                cv2.Rodrigues(R)[0],
                np.asarray(t).reshape(3),
                Kc[k[0]],
                dc[k[0]],
            )
            d = proj.reshape(-1, 2) - det_of[k].image_points
            sq += float((d**2).sum())
            n += len(rows_of_view[k])
        return float(np.sqrt(sq / n))

    converged = False
    n_skipped_tri = 0
    if board_release != "none" and len(released_rows):
        # ---- 2. alternate: triangulate -> gauge-fix -> re-solve poses ----
        obs_by_row: Dict[int, list] = {r: [] for r in released_rows.tolist()}
        rel_set = set(released_rows.tolist())
        for cam, view, r, u, v in observations:
            if r in rel_set:
                obs_by_row[r].append((cam, view, u, v))

        prev_rms = _overall_rms(K_by_cam, dist_by_cam, pose_by_view, board)
        for it in range(max_iters):
            new_board = board.copy()
            for r in released_rows:
                origins, dirs = [], []
                for cam, view, u, v in obs_by_row[r]:
                    R, t = pose_by_view[(cam, view)]
                    o, d = _pixel_rays_world(
                        np.array([[u, v]]), K_by_cam[cam], dist_by_cam[cam], R, t
                    )
                    origins.append(o)
                    dirs.append(d[0])
                pt, ok = _triangulate(np.array(origins), np.array(dirs))
                if ok:
                    new_board[r] = pt
                else:
                    n_skipped_tri += 1  # near-parallel rays: keep the current estimate

            # gauge-fix to the world frame
            if board_release == "full3d":
                R_g, t_g = _kabsch(new_board[released_rows], nominal[released_rows])
                new_board = new_board @ R_g.T + t_g
            else:  # z_only: lock in-plane to nominal, deplane the released z
                new_board[:, :2] = nominal[:, :2]
                z = new_board[released_rows, 2].copy()
                new_board[released_rows, 2] = _deplane_z(
                    np.column_stack([nominal[released_rows, :2], z])
                )
                mask = np.ones(N, bool)
                mask[released_rows] = False
                new_board[mask, 2] = 0.0
            board = new_board

            # re-solve poses against the updated board
            for k in view_keys:
                R, t = fit_pose(
                    board[rows_of_view[k]],
                    det_of[k].image_points,
                    K_by_cam[k[0]],
                    dist_by_cam[k[0]],
                    planar=False,
                )
                pose_by_view[k] = (R, t)

            # gauge-invariant convergence: stop when reprojection RMS stops improving
            rms_now = _overall_rms(K_by_cam, dist_by_cam, pose_by_view, board)
            if abs(prev_rms - rms_now) < 1e-4:
                converged = True
                prev_rms = rms_now
                break
            prev_rms = rms_now

    rms_alt = _overall_rms(K_by_cam, dist_by_cam, pose_by_view, board)
    info = {
        "bootstrap": boot_info,
        "rms_after_alternation": rms_alt,
        "n_released": int(len(released_rows)),
        "n_union": N,
        "n_skipped_triangulations": n_skipped_tri,
    }

    # ---- 3. factorise the seed into a rigid rig, then the final joint bundle, guarded ----
    # The alternation's poses are free per (cam, view); the OUTPUT is always rigid. Even when
    # the bundle is skipped or rejected, the rigid seed is what is returned -- never a
    # single-view datum pose.
    rig_by_cam, board_pose_by_view = _factorise_rig_seed(
        pose_by_view, cams, datum_camera, datum_view
    )
    # Board-point gauge (rigid motion of the free rows against every board pose) -- distinct
    # from the rig/board-pose gauge, which the datum pin removes. Needs >= 3 released rows.
    if board_release != "none" and len(released_rows) >= 3:
        anchors = set(_gauge_anchor_rows(nominal, released_rows))
        free_rows = [int(r) for r in released_rows if r not in anchors]
        ba_mode = board_release
    else:
        free_rows, ba_mode = [], "none"
    ba = _JointBA(board, observations, cams, view_keys, free_rows, ba_mode, datum_view)
    x_seed = ba.pack(K_by_cam, dist_by_cam, rig_by_cam, board_pose_by_view, board)
    rms_rigid_seed = ba.rms(x_seed)  # rigid, so >= rms_alt in general (lossy factorisation)
    info["rms_rigid_seed"] = rms_rigid_seed
    x_out = x_seed
    rms_final = rms_rigid_seed
    if run_bundle:
        sol = least_squares(
            ba.residuals,
            x_seed,
            jac=ba.jac,
            method="trf",
            x_scale="jac",
            tr_solver="lsmr",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=300,
            verbose=0,
        )
        rms_bundle = ba.rms(sol.x)
        # Like-for-like: both sides are the rigid parameterisation. Comparing against rms_alt
        # (free poses) would reject good bundles, since a rigid fit cannot match a free one.
        applied = rms_bundle <= rms_rigid_seed + 1e-9
        if applied:
            x_out, rms_final = sol.x, rms_bundle
        else:
            log.warning(
                "joint: final bundle did not improve on the rigid seed (%.4f -> %.4f) — "
                "keeping the rigid seed",
                rms_rigid_seed,
                rms_bundle,
            )
        info["bundle"] = {
            "status": int(sol.status),
            "nfev": int(sol.nfev),
            "rms": rms_bundle,
            "applied": applied,
        }
    board = ba.board_from(x_out)
    for c in cams:
        K_by_cam[c], dist_by_cam[c] = ba._K_dist(x_out, c)
    rig_by_cam = {}
    for c in cams:
        rvec, tvec = ba._rig(x_out, c)
        rig_by_cam[c] = (cv2.Rodrigues(rvec)[0], tvec)
    for k in view_keys:
        rvec, tvec = ba._pose(x_out, k)
        pose_by_view[k] = (cv2.Rodrigues(rvec)[0], tvec)

    # ---- assemble per-camera models (the rig pose; world = datum board plane) + RMS ----
    models, per_cam_rms = {}, {}
    for c in cams:
        R, t = rig_by_cam[c]
        sq, n = 0.0, 0
        for k in view_keys:
            if k[0] != c:
                continue
            Rk, tk = pose_by_view[k]
            proj, _ = cv2.projectPoints(
                board[rows_of_view[k]],
                cv2.Rodrigues(Rk)[0],
                np.asarray(tk).reshape(3),
                K_by_cam[c],
                dist_by_cam[c],
            )
            d = proj.reshape(-1, 2) - det_of[k].image_points
            sq += float((d**2).sum())
            n += len(rows_of_view[k])
        per_cam_rms[c] = float(np.sqrt(sq / n))
        models[c] = CameraModel(
            K=K_by_cam[c],
            dist=dist_by_cam[c],
            R=R,
            t=np.asarray(t).reshape(3, 1),
            image_size=image_size_by_cam[c],
            distortion_model=distortion_model,
            rms=per_cam_rms[c],
        )

    board_dict = {keys[i]: board[i].copy() for i in range(N)}
    wf = WorldFrame(mode="global_grid", origin_mm=np.array([ox, oy], dtype=np.float64))
    # Cross-camera board agreement is EXACTLY 0: every camera references the one shared ``board``
    # object above, so there is no second board to disagree with (this is the DaVis 0.0000mm
    # property by construction, not a measured residual). It is surfaced so a consumer can assert
    # the single-board invariant; a non-zero value here would mean the assembly logic regressed.
    return JointResult(
        cameras=cams,
        models=models,
        board=board_dict,
        view_poses=pose_by_view,
        world_frame=wf,
        spacing_mm=float(spacing_mm),
        board_release=board_release,
        rms_px=float(rms_final),
        per_camera_rms=per_cam_rms,
        cross_camera_board_agreement_mm=0.0,
        converged=converged,
        info=info,
    )


def run_joint_polynomial(
    detections_by_cam: Dict[int, List[DetectionResult]],
    global_index: Dict[ViewKey, np.ndarray],
    spacing_mm: float,
    cameras: Sequence[int],
    datum_view: int,
    origin_mm: Tuple[float, float] = (0.0, 0.0),
    image_size_by_cam: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[int, PolynomialModel]:
    """Per-camera single-plane polynomial map, all in ONE shared global frame.

    The polynomial alternative to the pinhole joint bundle. There is no released board and no
    cross-camera bundle: each camera fits a 3rd-order ``fit_polynomial`` on its ``datum_view``
    detection, but the world targets come from the SHARED global index ``(gx*spacing+ox,
    gy*spacing+oy)`` rather than a per-camera clicked frame. So every camera's polynomial
    already emits coordinates in the same global frame, and the old per-camera ``world_offset_mm``
    stitch is unnecessary (the cameras tile by construction). Returns one ``PolynomialModel`` per
    camera; the caller stores them as per-camera records (polynomial has no shared object to
    unify into a JointRecord). Raises if a camera's datum view is not in the resolved grid.
    """
    if image_size_by_cam is None:
        raise ValueError("run_joint_polynomial: image_size_by_cam is required")
    ox, oy = float(origin_mm[0]), float(origin_mm[1])
    out: Dict[int, PolynomialModel] = {}
    for cam in cameras:
        key = (int(cam), int(datum_view))
        if key not in global_index:
            raise ValueError(
                f"run_joint_polynomial: no resolved global grid for camera {cam} view "
                f"{datum_view} — the polynomial datum view must be in the global grid (add an "
                f"anchor for it, or pick a datum_view every camera shares)"
            )
        if cam not in image_size_by_cam:
            raise ValueError(
                f"run_joint_polynomial: image_size_by_cam missing camera {cam}"
            )
        if cam not in detections_by_cam or datum_view >= len(detections_by_cam[cam]):
            raise ValueError(
                f"run_joint_polynomial: camera {cam} has no detection for datum view {datum_view}"
            )
        det = detections_by_cam[cam][datum_view]
        gi = np.asarray(global_index[key], dtype=np.float64).reshape(-1, 2)
        if len(gi) != len(det.image_points):
            raise ValueError(
                f"run_joint_polynomial: camera {cam} view {datum_view} has {len(gi)} global "
                f"indices but {len(det.image_points)} image points"
            )
        world = np.column_stack(
            [gi[:, 0] * spacing_mm + ox, gi[:, 1] * spacing_mm + oy, np.zeros(len(gi))]
        )
        out[int(cam)] = fit_polynomial(det.image_points, world, image_size_by_cam[cam])
    return out
