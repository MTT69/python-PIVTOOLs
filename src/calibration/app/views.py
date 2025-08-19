from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

from calibration.calibration_planar.planar_calibration import (
    calculate_homography,
)
from calibration.calibration_planar.planar_calibration import (
    detect_dots as calib_detect_dots,
)
from calibration.calibration_planar.planar_calibration import (
    dewarp_image,
)
from calibration.calibration_planar.planar_calibration import (
    load_image as calib_load_image,
)
from calibration.calibration_planar.planar_calibration import (
    organize_grid_points,
    save_calibration_results,
)
from common.utils import camera_number, numpy_to_png_base64
from config import get_config
from paths import get_data_paths


def cache_key(source_path_idx, camera):
    return (int(source_path_idx), str(camera))


calibration_cache = {}
calibration_bp = Blueprint("calibration", __name__)


def _cfg():
    return get_config()


def _resolve_calibration_image(source_path_idx: int, camera: int, index: int = 1):
    cfg = get_config()
    source_root = Path(cfg.source_paths[source_path_idx])
    # Use get_data_paths to get calibration directory
    paths = get_data_paths(
        base_dir=source_root,
        num_images=1,
        cam=camera,
        type_name="",
        calibration=True,
    )
    calib_dir = paths["calib_dir"]
    filename = cfg.calibration_filename(index)
    img_path = calib_dir / filename
    return img_path


@calibration_bp.route("/calibration/get_image", methods=["GET"])
def calibration_get_image():
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))
    index = request.args.get("index", default=1, type=int)
    try:
        img_path = _resolve_calibration_image(source_path_idx, camera, index)
        img = calib_load_image(img_path)
        # Normalize to 0-255 uint8 for display
        disp = img - img.min()
        if disp.max() > 0:
            disp = disp / disp.max()
        disp8 = (disp * 255).astype(np.uint8)
        # Use the new numpy_to_png_base64 function
        b64 = numpy_to_png_base64(disp8)
        # Cache base info
        k = cache_key(source_path_idx, camera)
        calibration_cache.setdefault(k, {})
        calibration_cache[k]["image_path"] = img_path
        calibration_cache[k]["image"] = img  # keep float32
        return jsonify(
            {
                "image": b64,
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "path": str(img_path),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@calibration_bp.route("/calibration/detect_dots", methods=["GET"])
def calibration_detect_dots():
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = camera_number(request.args.get("camera", default=1, type=int))
    k = cache_key(source_path_idx, camera)
    try:
        cache = calibration_cache.get(k)
        if not cache or "image" not in cache:
            return jsonify({"error": "Calibration image not loaded"}), 400
        img = cache["image"]
        dots = calib_detect_dots(img, debug=False)
        calibration_cache[k]["dots"] = dots
        return jsonify({"dots": dots.tolist(), "count": int(len(dots))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/calibration/compute", methods=["POST"])
def calibration_compute():
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = camera_number(data.get("camera", 1))
    dot_distance_mm = float(data.get("dot_distance_mm", 28.9))
    grid_tolerance = float(data.get("grid_tolerance", 0.5))
    ransac_threshold = float(data.get("ransac_threshold", 3.0))
    datum = data.get("datum")
    right = data.get("right")
    above = data.get("above")
    k = cache_key(source_path_idx, camera)
    cache = calibration_cache.get(k)
    try:
        img = cache["image"]
        dots = cache["dots"]
        datum_np = np.array(datum, dtype=np.float32)
        right_np = np.array(right, dtype=np.float32)
        above_np = np.array(above, dtype=np.float32)
        grid_points, grid_indices, scale_x, scale_y, all_proj = organize_grid_points(
            dots,
            datum_np,
            right_np,
            above_np,
            dot_distance_mm,
            tolerance=grid_tolerance,
        )
        H, world_points, inlier_mask = calculate_homography(
            grid_points, grid_indices, dot_distance_mm, ransac_threshold
        )
        dewarped, transform, effective_resolution = dewarp_image(
            img, H, mm_per_pixel=0.1
        )
        # Prepare quick PNGs
        dewarped_disp = dewarped - dewarped.min()
        if dewarped_disp.max() > 0:
            dewarped_disp = dewarped_disp / dewarped_disp.max()
        dewarped_png = numpy_to_png_base64((dewarped_disp * 255).astype(np.uint8))
        # Save results to file in calibration folder
        img_path = cache.get("image_path")
        save_dir = Path(img_path).parent
        out_base = save_dir / "calibration_results"
        results = {
            "image_path": str(img_path),
            "grid_points": grid_points,
            "grid_indices": grid_indices,
            "inlier_mask": inlier_mask,
            "homography": H,
            "transform": transform,
            "dewarped": dewarped,
            "dot_distance_mm": dot_distance_mm,
            "effective_resolution": effective_resolution,
            "datum": datum_np,
            "right": right_np,
            "above": above_np,
            "world_points": world_points,
            "grid_tolerance": grid_tolerance,
        }
        save_calibration_results(str(out_base), results, format="mat")
        return jsonify(
            {
                "status": "ok",
                "grid_points": grid_points.tolist(),
                "grid_indices": grid_indices.tolist(),
                "inlier_mask": inlier_mask.astype(int).tolist(),
                "homography": H.tolist() if H is not None else None,
                "dewarped": dewarped_png,
                "effective_resolution": effective_resolution,
                "world_points": world_points.tolist(),
                "output_base": str(out_base),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
