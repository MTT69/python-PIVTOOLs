"""
Shared utilities for synthetic calibration board generation.

Used by the ChArUco, stereo, dotboard and stepped generators and by the stepped
unit tests.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_camera_matrix(W: int, H: int) -> np.ndarray:
    """Camera with fx=fy=W (~60 deg FOV), principal point at image centre."""
    fx = fy = float(W)
    cx, cy = W / 2.0, H / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def make_poses(
    n_views: int,
    board_centre_3d: np.ndarray,
    fx: float,
    image_w: int,
    board_width_3d: float,
    fill_fraction_range: tuple = (0.75, 0.75),
    seed: int = 42,
) -> list:
    """
    Deterministic poses.  View 0 is frontal; views 1-9 have progressive tilts.

    The board is centred in the image by offsetting tvec so that board_centre_3d
    projects to the image centre.  tz is chosen so the board fills a fraction
    of the image width controlled by *fill_fraction_range*.

    fill_fraction_range = (high, low):
      - View 0 (frontal) uses *high* -- board nearly fills the frame.
      - Tilted views interpolate from *high* down to *low* based on tilt magnitude.
      This ensures edge coverage in near-frontal views while preventing dot merging
      in heavily tilted views.

    Returns list of (rvec, tvec) tuples.
    """
    rng = np.random.default_rng(seed)

    fill_high, fill_low = fill_fraction_range

    tx_centre = -board_centre_3d[0]
    ty_centre = -board_centre_3d[1]

    max_tilt = 0.25

    poses = []
    for i in range(n_views):
        if i == 0:
            rvec = np.zeros(3, dtype=np.float64)
            fill = fill_high
        else:
            rx = rng.uniform(-max_tilt, max_tilt)
            ry = rng.uniform(-max_tilt, max_tilt)
            rz = rng.uniform(-0.10, 0.10)
            rvec = np.array([rx, ry, rz], dtype=np.float64)
            tilt_mag = np.sqrt(rx**2 + ry**2)
            t = min(tilt_mag / max_tilt, 1.0)
            fill = fill_high + t * (fill_low - fill_high)

        tz = fx * board_width_3d / (fill * image_w) + rng.uniform(-0.02, 0.02)
        tx = tx_centre + rng.uniform(-0.005, 0.005)
        ty = ty_centre + rng.uniform(-0.005, 0.005)
        tvec = np.array([tx, ty, tz], dtype=np.float64)
        poses.append((rvec, tvec))
    return poses


def warp_board(
    frontal_img: np.ndarray,
    src_corners_px: np.ndarray,
    dst_corners_px: np.ndarray,
    out_size: tuple,
) -> np.ndarray:
    """
    Warp *frontal_img* so that *src_corners_px* map to *dst_corners_px*.

    Uses a least-squares homography from ALL corners (not just 4).
    """
    H, _ = cv2.findHomography(
        src_corners_px.astype(np.float32),
        dst_corners_px.astype(np.float32),
        method=0,
    )
    if H is None:
        raise RuntimeError("findHomography returned None")
    return cv2.warpPerspective(
        frontal_img,
        H,
        out_size,
        flags=cv2.INTER_LINEAR,
        borderValue=200,
    )


def save_ground_truth(
    out_path: Path, camera_matrix: np.ndarray, poses: list, **extra_arrays
):
    """Save ground-truth camera parameters and poses to a .npz file."""
    rvecs = np.array([p[0] for p in poses])
    tvecs = np.array([p[1] for p in poses])
    np.savez(
        str(out_path),
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros(5),
        rvecs=rvecs,
        tvecs=tvecs,
        **extra_arrays,
    )


# ---------------------------------------------------------------------------
# Anti-aliased dot rendering (shared by the dotboard and stepped generators
# and by the stepped unit tests)
# ---------------------------------------------------------------------------

DOT_OVERSAMPLE = 4  # hi-res render factor before INTER_AREA downsampling
DOT_SHIFT_BITS = 4  # cv2.circle fractional bits (1/16 hi-res pixel)
STRIP_HEIGHT_PX = 256  # output rows rendered per strip (bounds peak memory)


def adaptive_dot_radii(
    points_px: np.ndarray, fraction: float = 0.22, min_radius_px: float = 1.5
) -> np.ndarray:
    """Per-dot radius proportional to the local nearest-neighbour distance.

    Keeps dots from merging under perspective (neighbours get closer, radius
    shrinks with them).  Radii are floats -- never rounded -- so the rendered
    disc area follows the geometry continuously.
    """
    from scipy.spatial import cKDTree

    nn = cKDTree(points_px).query(points_px, k=2)[0][:, 1]
    return np.maximum(fraction * nn, min_radius_px)


def render_dots(
    points_px: np.ndarray,
    radii_px: np.ndarray,
    size: tuple,
    foreground: int = 255,
    background: int = 0,
    oversample: int = DOT_OVERSAMPLE,
) -> np.ndarray:
    """Render anti-aliased filled discs at exact sub-pixel centres.

    Discs are drawn at ``oversample`` x resolution with ``cv2.circle(shift=)``
    fixed-point centres and radii (no integer rounding of either), then reduced
    with ``INTER_AREA``.  Pixel-centre convention matches ``cv2.projectPoints``
    (pixel ``i`` is centred on coordinate ``i``), so a disc centred on
    ``points_px[k]`` has its image centroid at ``points_px[k]``.

    The frame is rendered in horizontal strips so a 25 MP image never needs the
    400 MP hi-res buffer.

    Args:
        points_px: (N, 2) centres, x then y, in output pixels.
        radii_px: (N,) radii in output pixels.
        size: (width, height) of the output.
        foreground: grey level inside the discs (0-255).
        background: grey level elsewhere (0-255).
        oversample: hi-res factor.

    Returns:
        (height, width) uint8 image.
    """
    W, H = int(size[0]), int(size[1])
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    rad = np.asarray(radii_px, dtype=np.float64).reshape(-1)
    if pts.shape[0] != rad.shape[0]:
        raise ValueError(f"{pts.shape[0]} centres but {rad.shape[0]} radii")
    scale = 1 << DOT_SHIFT_BITS
    img = np.empty((H, W), np.uint8)
    for y0 in range(0, H, STRIP_HEIGHT_PX):
        y1 = min(y0 + STRIP_HEIGHT_PX, H)
        strip = np.full(((y1 - y0) * oversample, W * oversample), 0, np.uint8)
        # discs whose vertical extent touches [y0 - 0.5, y1 - 0.5]
        sel = np.nonzero((pts[:, 1] + rad >= y0 - 0.5) & (pts[:, 1] - rad <= y1 - 0.5))[
            0
        ]
        for k in sel:
            cx = (pts[k, 0] + 0.5) * oversample - 0.5
            cy = (pts[k, 1] - y0 + 0.5) * oversample - 0.5
            cv2.circle(
                strip,
                (int(round(cx * scale)), int(round(cy * scale))),
                int(round(rad[k] * oversample * scale)),
                255,
                -1,
                cv2.LINE_8,
                DOT_SHIFT_BITS,
            )
        cover = cv2.resize(strip, (W, y1 - y0), interpolation=cv2.INTER_AREA)
        img[y0:y1] = np.round(
            background + (foreground - background) * (cover.astype(np.float64) / 255.0)
        ).astype(np.uint8)
    return img


def centred_pose_depth(
    fx: float, fit_px: float, board_width_3d: float, fill: float
) -> float:
    """Depth ``Z`` at which a frontal board ``board_width_3d`` wide spans ``fill * fit_px`` px."""
    return fx * board_width_3d / (fill * fit_px)


def make_centred_poses(
    n_views: int,
    board_centre_3d: np.ndarray,
    fx: float,
    fit_px: float,
    board_width_3d: float,
    fill: float = 0.70,
    max_tilt: float = 0.15,
    seed: int = 42,
) -> list:
    """Poses with the board centroid on the optical axis at a common depth.

    View 0 is frontal in the board-centred rig frame (rvec = 0); a stereo rig
    built with ``symmetric_stereo_rig`` then turns each camera by its half angle
    about the centroid.  Every view shares one depth ``Z``
    (``centred_pose_depth``: board width fills ``fill * fit_px`` pixels when
    frontal) so the rig has a pose-independent ``T_stereo``.  Unlike
    ``make_poses`` the translation is ``[0, 0, Z] - R @ centre`` so a tilted
    board stays centred.

    Returns a list of ``(rvec, tvec)`` (float64, shape (3,)).
    """
    rng = np.random.default_rng(seed)
    Z = centred_pose_depth(fx, fit_px, board_width_3d, fill)
    poses = []
    for i in range(n_views):
        if i == 0:
            rvec = np.zeros(3)
        else:
            rvec = np.array(
                [
                    rng.uniform(-max_tilt, max_tilt),
                    rng.uniform(-max_tilt, max_tilt),
                    rng.uniform(-0.10, 0.10),
                ]
            )
        R, _ = cv2.Rodrigues(rvec)
        tvec = np.array([0.0, 0.0, Z]) - R @ np.asarray(board_centre_3d, np.float64)
        poses.append((rvec, tvec))
    return poses


def symmetric_stereo_rig(
    poses_rig: list,
    half_rotation: np.ndarray,
    depth: float,
    transmission: bool = False,
) -> tuple:
    """Two cameras turned symmetrically about the board centroid.

    ``poses_rig`` are board -> rig-frame poses from ``make_centred_poses``: the
    rig frame is the frontal camera of view 0, with the board centroid at
    ``P = [0, 0, depth]``.  Each camera ``k`` is the rig frame rotated by
    ``Q_k`` about ``P``, so ``R_k = Q_k R`` and ``t_k = Q_k (t - P) + P``.

        same_side     Q1 = half_rotation.T,  Q2 = half_rotation
        transmission  Q1 = half_rotation.T,  Q2 = Ry(180 deg) @ half_rotation
                      (camera 2 looks at the back face)

    With ``half_rotation = Ry(theta / 2)`` the cameras sit at -theta/2 and
    +theta/2 (same_side) or -theta/2 and 180 deg + theta/2 (transmission), and
    the centroid stays on both optical axes.

    ``R_stereo = R2 @ R1.T`` and ``T_stereo = t2 - R_stereo @ t1`` (cam 1 ->
    cam 2) are taken from the view-0 pair; both are pose independent
    (``R_stereo = Q2 Q1^T``, ``T_stereo = P - Q2 Q1^T P``) so every view obeys
    ``R2 = R_stereo R1``, ``t2 = R_stereo t1 + T_stereo``.

    Returns ``(poses_cam1, poses_cam2, R_stereo (3, 3), T_stereo (3,))``.
    """
    Q1 = np.asarray(half_rotation, np.float64).T
    Q2 = np.asarray(half_rotation, np.float64)
    if transmission:
        Q2 = cv2.Rodrigues(np.array([0.0, np.pi, 0.0]))[0] @ Q2
    P = np.array([0.0, 0.0, float(depth)])
    out1, out2 = [], []
    for rvec, tvec in poses_rig:
        R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
        t = np.asarray(tvec, np.float64).reshape(3)
        for Q, out in ((Q1, out1), (Q2, out2)):
            rvec_k, _ = cv2.Rodrigues(Q @ R)
            out.append((rvec_k.reshape(3), Q @ (t - P) + P))
    R1 = cv2.Rodrigues(out1[0][0])[0]
    R2 = cv2.Rodrigues(out2[0][0])[0]
    R_stereo = R2 @ R1.T
    T_stereo = out2[0][1] - R_stereo @ out1[0][1]
    return out1, out2, R_stereo, T_stereo


def project_points(
    obj_pts: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """Distortion-free projection of (N, 3) points; returns (N, 2) pixels."""
    return cv2.projectPoints(
        np.asarray(obj_pts, np.float64),
        np.asarray(rvec, np.float64),
        np.asarray(tvec, np.float64),
        K,
        np.zeros(5),
    )[0].reshape(-1, 2)


def fully_inside(
    points_px: np.ndarray, radii_px: np.ndarray, size: tuple, margin_px: float = 1.0
) -> np.ndarray:
    """True where a disc lies wholly inside the frame by at least ``margin_px``.

    The detector drops any blob touching the frame edge, so these are the dots
    it is entitled to find.
    """
    W, H = size
    return (
        (points_px[:, 0] - radii_px >= margin_px)
        & (points_px[:, 0] + radii_px <= W - 1 - margin_px)
        & (points_px[:, 1] - radii_px >= margin_px)
        & (points_px[:, 1] + radii_px <= H - 1 - margin_px)
    )


def save_overview_figure(
    image: np.ndarray,
    gt_points_px: np.ndarray,
    out_path: Path,
    title: str,
    detected_px: np.ndarray | None = None,
) -> None:
    """Datum-view overview: image with ground-truth dots (and detections) overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.plot(
        gt_points_px[:, 0],
        gt_points_px[:, 1],
        "o",
        mfc="none",
        mec="tab:red",
        ms=7,
        mew=0.8,
        label=f"ground truth ({len(gt_points_px)})",
    )
    if detected_px is not None:
        ax.plot(
            detected_px[:, 0],
            detected_px[:, 1],
            "+",
            color="tab:cyan",
            ms=6,
            label=f"detected ({len(detected_px)})",
        )
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaVision 309-15-3 stepped plate
# ---------------------------------------------------------------------------

STEPPED_SPACING_MM = 15.0
STEPPED_STEP_MM = 3.0
STEPPED_DOT_DIAMETER_MM = 3.0
STEPPED_PEAK_SHAPE = (21, 21)  # (cols, rows) of the z = 0 level
STEPPED_TROUGH_SHAPE = (20, 20)  # (cols, rows) of the z = -step level
# Plate thickness, front peak plane to back peak plane.  Matches the stepped
# calibrator default (SteppedBoardSpec.board_thickness_mm) and the 309-15-3
# plates used on Andre's and Zefanya's data.
STEPPED_THICKNESS_MM = 14.8


def stepped_board_points_mm(
    peak_shape: tuple = STEPPED_PEAK_SHAPE,
    trough_shape: tuple = STEPPED_TROUGH_SHAPE,
    spacing_mm: float = STEPPED_SPACING_MM,
    step_mm: float = STEPPED_STEP_MM,
) -> tuple:
    """Two-level stepped plate dot centres in mm (board frame, row-major).

    Peak dots sit on the integer grid at z = 0; trough dots on the
    half-spacing-offset grid at z = -step (LaVision 309-15-3 layout: 21 x 21
    over 20 x 20, 15 mm pitch, 3 mm step).  Peak (col 0, row 0) is the origin.

    Returns (peak (Np, 3), trough (Nt, 3)).
    """
    off = spacing_mm / 2.0
    peak = [
        [c * spacing_mm, r * spacing_mm, 0.0]
        for r in range(peak_shape[1])
        for c in range(peak_shape[0])
    ]
    trough = [
        [c * spacing_mm + off, r * spacing_mm + off, -step_mm]
        for r in range(trough_shape[1])
        for c in range(trough_shape[0])
    ]
    return np.array(peak, np.float64), np.array(trough, np.float64)


def render_stepped_view(
    peak_mm: np.ndarray,
    trough_mm: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    size: tuple,
    dot_diameter_mm: float = STEPPED_DOT_DIAMETER_MM,
    radius_fraction: float | None = None,
) -> tuple:
    """Project both levels through one pose and render anti-aliased white dots.

    Dot radius is either physical (``dot_diameter_mm`` / 2 scaled by each dot's
    own depth, ``radius_fraction=None``) or ``radius_fraction`` x the local
    nearest-neighbour pixel distance (the fraction the legacy test renderers
    used, kept so their detection margins are unchanged).

    Returns (image uint8 (H, W), points_px (Np + Nt, 2) peak first, radii_px (Np + Nt,)).
    """
    allpts = np.vstack([peak_mm, trough_mm])
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    px = project_points(allpts, rvec, tvec, K)
    if radius_fraction is None:
        depth = np.einsum("j,nj->n", R[2], allpts) + float(
            np.asarray(tvec).reshape(3)[2]
        )
        radii = 0.5 * dot_diameter_mm * float(K[0, 0]) / depth
    else:
        radii = adaptive_dot_radii(px, radius_fraction, min_radius_px=2.0)
    return render_dots(px, radii, size), px, radii
