"""Import DaVis PinholeOpenCV calibration models into PIVTOOLs format.

Parses DaVis Calibration.xml files containing PinholeOpenCV parameters and
saves them as camera_model.mat files compatible with VectorCalibrator.

Coordinate system notes:
  - DaVis FocalLengthPixel is in MILLIMETRES (despite the name — it is the
    physical focal length, computed from pixel correspondences). Conversion to
    sensor pixels: f_sensor_px = FocalLengthPixel_mm / SensorPixelSizeMm.
    Do NOT divide by PixelPerMmFactor — that is a derived property of the
    stitched corrected image, not a needed conversion factor.
  - DaVis PrincipalPoint is in original sensor pixel coordinates — used directly.
  - DaVis RotationAngles (Rx, Ry, Rz) are OpenCV Rodrigues rotation vectors.
  - DaVis TranslationMm (Tx, Ty, Tz) is in mm — same units as VectorCalibrator expects.
  - Multiple <CoordinateSystem> elements = multiple calibration plate poses;
    each is stored as a row in rvecs/tvecs. VectorCalibrator uses row 0 (datum_frame=1).
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
            sensor_px_mm = float(internal.find("SensorPixelSizeMm").get("Value"))
            # FocalLengthPixel is in mm (physical focal length). Convert to sensor pixels.
            f_mm = float(internal.find("FocalLengthPixel").get("x"))
            f_px = f_mm / sensor_px_mm
            cx = float(internal.find("PrincipalPoint").get("x"))
            cy = float(internal.find("PrincipalPoint").get("y"))
            k1 = float(internal.find("RadialDistortion").get("radialDistortionCoefficient1"))
            k2 = float(internal.find("RadialDistortion").get("radialDistortionCoefficient2"))
            p1 = float(internal.find("TangentialDistortion").get("tangentialDistortionCoefficient1"))
            p2 = float(internal.find("TangentialDistortion").get("tangentialDistortionCoefficient2"))

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

    cameras = parse_davis_pinhole_xml(xml_path)
    saved = []
    errors = []

    for davis_cam_id, cam_data in cameras.items():
        pivtools_cam = (
            camera_map.get(davis_cam_id, davis_cam_id) if camera_map else davis_cam_id
        )

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
