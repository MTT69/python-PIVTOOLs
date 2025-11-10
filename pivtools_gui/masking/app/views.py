from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

from ...config import get_config
from ...utils import camera_folder, camera_number
from ...vector_loading import read_mask_from_mat, save_mask_to_mat

masking_bp = Blueprint("masking", __name__)


def _cfg():
    return get_config()


@masking_bp.route("/save_mask_array", methods=["POST"])
def upload_mask():
    """
    Expects JSON with: meta (basePathIdx, camera, index, frame), width, height, data (flat mask), polygons (optional).
    Saves mask as .mat file.
    """
    payload = request.get_json(silent=True) or {}
    width, height, flat = (
        payload.get("width"),
        payload.get("height"),
        payload.get("data"),
    )
    meta = payload.get("meta", {})
    polygons = payload.get("polygons", None)

    # Validate input
    if not (
        isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0
    ):
        return jsonify({"error": "width and height must be positive integers"}), 400
    if not (isinstance(flat, list) and len(flat) == width * height):
        return jsonify({"error": "data must be a list of length width*height"}), 400

    try:
        mask = np.asarray(flat, dtype=bool).reshape((height, width))
    except Exception as e:
        return jsonify({"error": f"invalid mask data: {e}"}), 400

    try:
        basePathIdx = meta["basePathIdx"]
        camera = meta["camera"]
        cfg = _cfg()
        source_paths = cfg.source_paths
        try:
            camera_num = camera_number(camera)
        except Exception:
            camera_num = camera
        mask_path = source_paths[basePathIdx] / f"mask_{camera_folder(camera_num)}.mat"
        save_mask_to_mat(mask_path, mask, np.asarray(polygons))
    except Exception:
        return jsonify({"error": "invalid or missing meta fields"}), 400

    true_count = int(mask.sum())
    return jsonify(
        {
            "status": "ok",
            "shape": [height, width],
            "true_count": true_count,
            "fraction_true": true_count / (width * height),
            "meta": meta,
        }
    )


@masking_bp.route("/load_mask", methods=["GET"])
def load_mask():
    """
    Loads a mask and polygon data from a .mat file.
    Query params:
      - path: full path to mask .mat file (preferred)
      - basepath_idx, camera: optional, used to construct path if 'path' not given
    Returns: { mask: [0|1,...], width, height, polygons: [...] }
    """
    cfg = _cfg()
    path = request.args.get("path", default=None, type=str)
    # Optionally reconstruct path if not provided
    if not path or not Path(path).exists():
        try:
            basepath_idx = int(request.args.get("basepath_idx", 0))
            camera = request.args.get("camera")
            base_paths = cfg.source_paths
            if basepath_idx < 0 or basepath_idx >= len(base_paths):
                return jsonify({"error": "basepath_idx out of range"}), 400
            camera = camera_number(camera)
            mask_filename = f"mask_{camera_folder(camera)}.mat"
            path = str(base_paths[basepath_idx] / mask_filename)
        except Exception as e:
            return jsonify({"error": f"Could not resolve mask path: {e}"}), 400

    if not Path(path).exists():
        return jsonify({"error": f"Mask file not found: {path}"}), 404

    try:
        mask, polygons = read_mask_from_mat(path)

        def serialize_polygon(poly):
            return {
                "index": int(poly["index"]),
                "name": str(poly["name"]),
                "points": [list(map(float, pt)) for pt in poly["points"]],
            }

        polygons_serializable = [serialize_polygon(p) for p in polygons]
        mask_arr = np.asarray(mask)
        mask_flat = mask_arr.astype(np.uint8).flatten().tolist()
        height, width = mask_arr.shape
        return jsonify(
            {
                "mask": mask_flat,
                "width": width,
                "height": height,
                "polygons": polygons_serializable,
            }
        )
    except Exception as e:
        print("Exception in load_mask:", e)
        return jsonify({"error": f"Failed to load mask: {e}"}), 500
