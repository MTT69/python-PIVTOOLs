"""Import DaVis PinholeOpenCV calibration models into PIVTOOLs format.

Parses DaVis Calibration.xml files containing PinholeOpenCV parameters and
saves them as camera_model.mat files compatible with VectorCalibrator.

Coordinate system notes:
  - DaVis FocalLengthPixel is stored in MILLIMETRES (physical focal length),
    despite the "Pixel" name. Convert to sensor pixels:
      f_sensor_px = FocalLengthPixel_mm / SensorPixelSizeMm
    Evidence: all cameras in a panoramic setup have the same physical magnification
    Tz×SensorPixelSizeMm/FocalLengthPixel ≈ 0.059 mm/px, consistent only when
    FocalLengthPixel is treated as mm. Using it directly as pixels gives 21 mm/px.
  - DaVis PrincipalPoint is in original sensor pixel coordinates — used directly.
  - DaVis RotationAngles (Rx, Ry, Rz) are OpenCV Rodrigues rotation vectors.
  - DaVis TranslationMm (Tx, Ty, Tz) is in mm — same units as VectorCalibrator expects.
  - Multiple <CoordinateSystem> elements = multiple calibration plate poses;
    each is stored as a row in rvecs/tvecs. VectorCalibrator uses row 0 (datum_frame=1).

1-view degeneracy and conditional f_px normalisation:
  - A single calibration plate pose (SideBySide2D panoramic) leaves f and Tz individually
    unobservable. Only their ratio f/Tz (the mm/px scale) is well-determined.
  - DaVis reports wildly wrong individual f and Tz values, but their ratio is correct.
  - Consequence: the normalised coordinates x_n = (x_raw - cx) / f_davis can be huge
    (e.g. x_n ≈ 1037 for camera 1 with f_px = 2.56). The R-matrix off-diagonal terms
    (from small Rz tilts) multiply x_n and can flip the sign of the ray-plane denominator
    mid-sensor, sending world coordinates to ±infinity.
  - Fix: applied ONLY when the denominator is unsafe.
    For each camera, compute the ray-plane denominator at all four image corners using the
    DaVis f_px. If the minimum denominator exceeds −0.5 (zero crossing risk), normalise:
      f_px_norm  = image_width / 2
      Tz_norm    = Tz_davis × (f_px_norm / f_px_davis)
    Tx and Ty are NOT scaled — they encode the correct lateral camera positions.
  - Cameras with large DaVis f_px (and hence tiny x_n) have safe denominators and are
    used with DaVis values unchanged.

Mark-based model fitting (MarkPositionTable.xml):
  - Only applied for single-pose calibrations (one <CoordinateSystem> block per camera).
    Multi-pose calibrations are well-determined; the DaVis model is used as-is.
  - For single-pose setups: if MarkPositionTable.xml is found in a camera1/ subdirectory
    next to Calibration.xml, the actual DaVis-detected pixel↔world correspondences
    (~590 per camera) are used to fit accurate rvec/tvec via cv2.solvePnP. This corrects
    residual errors in DaVis's global optimisation and the degenerate f/Tz scaling for
    camera 1.
  - The K matrix is first set via the denominator-safety rule above (f_px=W/2 for
    degenerate cameras, DaVis f_px for safe cameras), then solvePnP finds the rvec/tvec
    that minimises reprojection error against the mark data.
  - When marks are available all cameras benefit: degenerate cameras get a correct model
    from the mark fit (force=True), and non-degenerate cameras get their DaVis rvec/tvec
    refined if solvePnP achieves lower reprojection error (force=False).
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.io import savemat


def parse_davis_pinhole_xml(xml_path: str) -> Dict[int, dict]:
    """Parse DaVis Calibration.xml with PinholeOpenCV parameters.

    Returns:
        {camera_no: {
            "K": np.ndarray (3, 3),       intrinsic matrix in sensor pixels
            "dist": np.ndarray (4,),      [k1, k2, p1, p2]
            "rvecs": np.ndarray (N, 3),   Rodrigues vectors, one per pose
            "tvecs": np.ndarray (N, 3),   translations in mm, one per pose
            "image_width": int,
            "image_height": int,
        }}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # One entry per camera: accumulate (K, dist, w, h) once, then append poses
    intrinsics: Dict[int, tuple] = {}      # cam_id -> (K, dist, W, H)
    poses: Dict[int, list] = {}            # cam_id -> [(rvec, tvec), ...]

    for cs in root.findall(".//CoordinateSystem"):
        for mapper in cs.findall("CoordinateMapper"):
            cam_id = int(mapper.get("CameraIdentifier"))
            params = mapper.find("PinholeParameters")
            if params is None:
                continue

            # Image size
            common = params.find("CommonParameters")
            orig = common.find("OriginalImageSize")
            W = int(orig.get("Width"))
            H = int(orig.get("Height"))

            # Intrinsics (same across all poses for this camera)
            internal = params.find("InternalCameraParameters")
            # FocalLengthPixel is in mm (physical focal length). Convert to sensor pixels.
            sensor_px_mm = float(internal.find("SensorPixelSizeMm").get("Value"))
            f_px = float(internal.find("FocalLengthPixel").get("x")) / sensor_px_mm
            cx = float(internal.find("PrincipalPoint").get("x"))
            cy = float(internal.find("PrincipalPoint").get("y"))
            rad = internal.find("RadialDistortion")
            tan = internal.find("TangentialDistortion")
            k1 = float(rad.get("radialDistortionCoefficient1", 0.0)) if rad is not None else 0.0
            k2 = float(rad.get("radialDistortionCoefficient2", 0.0)) if rad is not None else 0.0
            p1 = float(tan.get("tangentialDistortionCoefficient1", 0.0)) if tan is not None else 0.0
            p2 = float(tan.get("tangentialDistortionCoefficient2", 0.0)) if tan is not None else 0.0

            K = np.array([[f_px, 0.0, cx],
                          [0.0, f_px, cy],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            dist = np.array([k1, k2, p1, p2], dtype=np.float64)

            if cam_id not in intrinsics:
                intrinsics[cam_id] = (K, dist, W, H)
                poses[cam_id] = []

            # Extrinsics (one per CoordinateSystem block = one calibration pose)
            external = params.find("ExternalCameraParameters")
            r_elem = external.find("RotationAngles")
            t_elem = external.find("TranslationMm")
            rvec = np.array([
                float(r_elem.get("Rx")),
                float(r_elem.get("Ry")),
                float(r_elem.get("Rz")),
            ], dtype=np.float64)
            tvec = np.array([
                float(t_elem.get("Tx")),
                float(t_elem.get("Ty")),
                float(t_elem.get("Tz")),
            ], dtype=np.float64)
            poses[cam_id].append((rvec, tvec))

    result = {}
    for cam_id in sorted(intrinsics.keys()):
        K, dist, W, H = intrinsics[cam_id]
        cam_poses = poses[cam_id]
        result[cam_id] = {
            "K": K,
            "dist": dist,
            "rvecs": np.array([p[0] for p in cam_poses], dtype=np.float64),  # (N, 3)
            "tvecs": np.array([p[1] for p in cam_poses], dtype=np.float64),  # (N, 3)
            "image_width": W,
            "image_height": H,
        }
    return result


def parse_mark_position_table(xml_path: str) -> Dict[int, dict]:
    """Parse DaVis MarkPositionTable.xml to get pixel<->world correspondences.

    The file contains one <Camera> section per camera with <Mark> elements,
    each carrying <RawPos> (sensor pixels) and <WorldPos> (mm on the plate).

    Returns:
        {cam_id: {
            "image_pts": np.ndarray (N, 2),  RawPos (x, y) in sensor pixels
            "world_pts": np.ndarray (N, 3),  WorldPos (x, y, 0.0) in mm
        }}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = {}
    for camera in root.findall("Camera"):
        cam_id = int(camera.get("CameraNumber"))
        image_pts: list = []
        world_pts: list = []
        for mark in camera.findall(".//Mark"):
            raw_pos = mark.find("RawPos")
            world_pos = mark.find("WorldPos")
            if raw_pos is None or world_pos is None:
                continue
            image_pts.append([float(raw_pos.get("x")), float(raw_pos.get("y"))])
            world_pts.append([float(world_pos.get("x")), float(world_pos.get("y")), 0.0])
        if image_pts:
            result[cam_id] = {
                "image_pts": np.array(image_pts, dtype=np.float64),
                "world_pts": np.array(world_pts, dtype=np.float64),
            }
    return result


def _find_mark_position_table(xml_path: str) -> Optional[str]:
    """Search for MarkPositionTable.xml adjacent to the Calibration.xml.

    DaVis places it in a camera1/ subdirectory; the single file covers all cameras.
    """
    calib_dir = Path(xml_path).parent
    candidate = calib_dir / "camera1" / "MarkPositionTable.xml"
    if candidate.exists():
        return str(candidate)
    candidate2 = calib_dir / "MarkPositionTable.xml"
    if candidate2.exists():
        return str(candidate2)
    return None


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-10:
        return np.eye(3, dtype=np.float64)
    axis = rvec / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    return np.cos(angle) * np.eye(3) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * K


def _maybe_normalise_focal(cam_data: dict) -> dict:
    """Normalise f_px to W/2 only when the ray-plane denominator would be unsafe.

    Computes the denominator (rays_world[2]) at all four image corners using the
    DaVis f_px. If the minimum denominator exceeds −0.5, the denominator may cross
    zero across the sensor and the camera needs normalisation. Only Tz is scaled;
    Tx/Ty are unchanged because they represent correct lateral camera positions.
    """
    f_px = cam_data["K"][0, 0]
    cx = cam_data["K"][0, 2]
    cy = cam_data["K"][1, 2]
    W = cam_data["image_width"]
    H = cam_data["image_height"]
    rvec = cam_data["rvecs"][0]  # datum pose

    R_inv = _rodrigues(rvec).T
    corner_xn = [(0 - cx) / f_px, (W - 1 - cx) / f_px]
    corner_yn = [(0 - cy) / f_px, (H - 1 - cy) / f_px]
    min_denom = min(
        float((R_inv @ np.array([xn, yn, 1.0]))[2])
        for xn in corner_xn
        for yn in corner_yn
    )

    if min_denom > -0.5:
        f_px_norm = W / 2.0
        tz_scale = f_px_norm / f_px
        K_norm = cam_data["K"].copy()
        K_norm[0, 0] = f_px_norm
        K_norm[1, 1] = f_px_norm
        tvecs_norm = cam_data["tvecs"].copy()
        tvecs_norm[:, 2] *= tz_scale
        return {**cam_data, "K": K_norm, "tvecs": tvecs_norm}
    return cam_data


def _reprojection_rms(world_pts, image_pts, rvec, tvec, K, dist) -> float:
    """RMS reprojection error (pixels) of rvec/tvec on the given correspondences."""
    proj, _ = cv2.projectPoints(world_pts, rvec, tvec, K, dist)
    diffs = proj.reshape(-1, 2) - image_pts.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))


def _fit_with_solvepnp(cam_data: dict, marks: dict, force: bool = False) -> dict:
    """Fit rvec/tvec using cv2.solvePnP on DaVis-detected mark correspondences.

    force=True: always adopt the solvePnP result (default in production — anchors
      every camera to the shared WorldPos global coordinate system).
    force=False: only adopt solvePnP if it achieves lower reprojection error than
      the original DaVis model (kept for callers that want a conservative update).

    Returns cam_data unchanged if solvePnP fails or the original model is better.
    """
    K = cam_data["K"].astype(np.float64)
    dist = cam_data["dist"].astype(np.float64)
    world_pts = marks["world_pts"].reshape(-1, 1, 3).astype(np.float64)
    image_pts = marks["image_pts"].reshape(-1, 1, 2).astype(np.float64)

    rvec_orig = cam_data["rvecs"][0].reshape(3, 1).astype(np.float64)
    tvec_orig = cam_data["tvecs"][0].reshape(3, 1).astype(np.float64)
    rms_orig = _reprojection_rms(world_pts, image_pts, rvec_orig, tvec_orig, K, dist)

    # SOLVEPNP_ITERATIVE: computes homography-based init for coplanar points
    # (all Z=0) then refines with Levenberg-Marquardt.
    success, rvec, tvec = cv2.solvePnP(
        world_pts, image_pts, K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return cam_data

    rms_pnp = _reprojection_rms(world_pts, image_pts, rvec, tvec, K, dist)

    if force or rms_pnp < rms_orig:
        rvecs_new = cam_data["rvecs"].copy()
        tvecs_new = cam_data["tvecs"].copy()
        rvecs_new[0] = rvec.flatten()
        tvecs_new[0] = tvec.flatten()
        return {**cam_data, "rvecs": rvecs_new, "tvecs": tvecs_new}

    return cam_data


def save_davis_pinhole_models(
    xml_path: str,
    base_paths: List[Path],
    dt: float,
    camera_map: Optional[Dict[int, int]] = None,
) -> dict:
    """Parse DaVis Calibration.xml and save camera_model.mat for each camera.

    Saves to: {base_path}/calibration/Cam{N}/davis_pinhole/model/camera_model.mat

    Args:
        xml_path: Path to DaVis Calibration.xml
        base_paths: PIVTOOLs base paths to save models into
        dt: Time step in seconds (stored in model for apply-calibration)
        camera_map: Optional {davis_cam_id: pivtools_cam_no}.
                    Defaults to identity (1→1, 2→2, ...).

    Returns:
        {"cameras": [{davis_cam_id, pivtools_cam, model_path, n_poses}],
         "errors": [str]}
    """
    from pivtools_core.paths import get_data_paths

    try:
        cameras = parse_davis_pinhole_xml(xml_path)
    except Exception as e:
        return {"cameras": [], "errors": [f"XML parse error: {e}"]}

    # Try to load mark correspondences for solvePnP-based fitting (single-pose only).
    marks: Dict[int, dict] = {}
    mark_xml = _find_mark_position_table(xml_path)
    if mark_xml is not None:
        try:
            marks = parse_mark_position_table(mark_xml)
        except Exception:
            marks = {}

    saved = []
    errors = []

    for davis_cam_id, cam_data in cameras.items():
        pivtools_cam = (
            camera_map.get(davis_cam_id, davis_cam_id) if camera_map else davis_cam_id
        )

        n_poses = len(cam_data["rvecs"])

        if n_poses == 1:
            # Single-pose: may be degenerate. Normalise K if unsafe, then fit
            # rvec/tvec from marks. force=True so every camera's pose is always
            # anchored to the shared WorldPos global coordinate system regardless
            # of whether solvePnP marginally improves reprojection — this gives
            # consistent inter-camera alignment across all 5 cameras.
            cam_data = _maybe_normalise_focal(cam_data)

            if davis_cam_id in marks:
                cam_data = _fit_with_solvepnp(
                    cam_data, marks[davis_cam_id], force=True,
                )
        # n_poses > 1: multi-pose calibration is well-determined — use DaVis as-is.

        for base_path in base_paths:
            try:
                calib_paths = get_data_paths(
                    base_path,
                    num_frame_pairs=1,
                    cam=pivtools_cam,
                    type_name="",
                    calibration=True,
                )
                model_dir = calib_paths["calib_dir"] / "davis_pinhole" / "model"
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / "camera_model.mat"

                savemat(str(model_path), {
                    "camera_matrix": cam_data["K"],
                    "dist_coeffs": cam_data["dist"],
                    "rvecs": cam_data["rvecs"],     # (N, 3) — row 0 is datum
                    "tvecs": cam_data["tvecs"],     # (N, 3)
                    "image_width": cam_data["image_width"],
                    "image_height": cam_data["image_height"],
                    "datum_frame": 1,               # 1-based, row 0 used by VectorCalibrator
                    "dt": dt,
                    "source_xml": str(xml_path),
                    "timestamp": datetime.now().isoformat(),
                    # DaVis Rx≈±π already encodes y-up world frame, so VectorCalibrator
                    # must NOT negate y at output (unlike native dotboard/charuco models).
                    "y_negate": 0,
                    "mark_xml": str(mark_xml) if mark_xml else "",
                })

                saved.append({
                    "davis_cam_id": davis_cam_id,
                    "pivtools_cam": pivtools_cam,
                    "model_path": str(model_path),
                    "n_poses": len(cam_data["rvecs"]),
                })
            except Exception as e:
                errors.append(
                    f"Camera {davis_cam_id} → {pivtools_cam}: {e}"
                )

    return {"cameras": saved, "errors": errors}
