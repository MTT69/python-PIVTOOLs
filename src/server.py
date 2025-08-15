import base64
import threading
from io import BytesIO
from pathlib import Path

import dask.array as da
import numpy as np
import yaml  # Add this import
from dask import config as dask_config
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from calibration_planar.planar_calibration import (
    calculate_homography,
)
from calibration_planar.planar_calibration import detect_dots as calib_detect_dots
from calibration_planar.planar_calibration import (
    dewarp_image,
)
from calibration_planar.planar_calibration import load_image as calib_load_image
from calibration_planar.planar_calibration import (
    organize_grid_points,
    save_calibration_results,
)
from config import Config
from image_handling.load_images import read_pair
from plotting.app.views import vector_plot_bp
from post_processing.vector_loading import read_mask_from_mat, save_mask_to_mat
from pre_processing.filters import filter_images  # use full filter pipeline

app = Flask(__name__)
CORS(app)  # enable CORS for frontend dev (Next.js on a different port)

dask_config.set(scheduler="threads")

config = Config()

app.register_blueprint(vector_plot_bp)

# In-memory storage for processed results and processing status
# Reworked: cache by (source_path_idx, cam_folder) -> { frame_index:int -> np.ndarray (2,H,W) }
processed_store = {
    "original": {},  # dict[tuple[int,str], dict[int, np.ndarray]]
    "processed": {},  # dict[tuple[int,str], dict[int, np.ndarray]]
}
processing = False
calibration_cache = (
    {}
)  # key: (source_path_idx, camera) -> { 'image_path': Path, 'dots': np.ndarray }


# Helpers to normalize camera folder and build cache key
def cam_folder_key(camera: str) -> str:
    return camera if str(camera).lower().startswith("cam") else f"Cam{camera}"


def cache_key(source_path_idx: int, camera: str):
    return (int(source_path_idx), cam_folder_key(camera))


# Utility to convert numpy array to PNG bytes
def numpy_to_png_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@app.route("/get_frame_pair", methods=["GET"])
def get_frame_pair():
    camera = request.args.get("camera")
    idx = request.args.get("idx", type=int)
    # NEW: optional source path index (defaults to 0)
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    print(
        f"Received request for camera: {camera}, idx: {idx}, source_path_idx: {source_path_idx}"
    )
    if camera is None or idx is None:
        print("Missing camera or idx in request")
        return jsonify({"error": "camera and idx required"}), 400
    try:
        # Normalise camera folder (accepts '1' or 'Cam1')
        cam_folder = camera if str(camera).lower().startswith("cam") else f"Cam{camera}"
        # Bounds check for source_path_idx
        if source_path_idx < 0 or source_path_idx >= len(config.source_paths):
            return (
                jsonify(
                    {
                        "error": f"source_path_idx out of range (0..{len(config.source_paths) - 1})"
                    }
                ),
                400,
            )
        camera_path = config.source_paths[source_path_idx] / cam_folder
        print(f"Reading images from: {camera_path}")
        pair = read_pair(idx, camera_path, config)
        img_a, img_b = pair[0], pair[1]
        png_a = numpy_to_png_bytes(img_a)
        png_b = numpy_to_png_bytes(img_b)
        b64_a = base64.b64encode(png_a).decode("utf-8")
        b64_b = base64.b64encode(png_b).decode("utf-8")

        # Build response with PNGs
        resp = {"A": b64_a, "B": b64_b}

        # Optionally include raw + meta if dtype supported (uint8/uint16)
        dtype_str = None
        bit_depth = None
        if pair.dtype == np.uint16:
            dtype_str, bit_depth = "uint16", 16
        elif pair.dtype == np.uint8:
            dtype_str, bit_depth = "uint8", 8

        if dtype_str is not None:
            H, W = int(pair.shape[1]), int(pair.shape[2])
            resp["meta"] = {
                "width": W,
                "height": H,
                "bitDepth": bit_depth,
                "dtype": dtype_str,
            }
            resp["A_raw"] = base64.b64encode(img_a.tobytes()).decode("utf-8")
            resp["B_raw"] = base64.b64encode(img_b.tobytes()).decode("utf-8")

        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/filter", methods=["POST"])
def filter_images_endpoint():
    """
    Expects JSON body:
      - camera: camera folder name (e.g. 'Cam1' or '1')
      - start_idx: target frame pair index (1-based)
      - count: number of target pairs to return (currently only 1 is supported; ignored otherwise)
      - filters: list of filter dicts. Temporal filters ('time', 'POD') may include:
          { "type": "time", "batch_size": 50 }
          { "type": "POD",  "batch_size": 100 }
      - source_path_idx or base_path_idx: integer index into config.source_paths
    Behavior:
      - Computes the timeline batch (non-overlapping window) that contains start_idx using max temporal batch_size.
      - Loads that batch, applies filters (rechunking per-filter to its batch_size), then caches each frame of the window.
      - Subsequent GET /get_processed_pair?type=processed&frame=<idx>&camera=<CamX>&source_path_idx=<i> returns from cache.
    """
    global processing
    if processing:
        return (
            jsonify(
                {"status": "processing", "message": "Processing already in progress"}
            ),
            409,
        )

    data = request.get_json() or {}
    camera = data.get("camera")
    start_idx = int(data.get("start_idx", 1))
    count = int(data.get("count", 1))
    filters = data.get("filters", None)
    # NEW: choose source path index (frontend may send base_path_idx)
    source_path_idx = data.get("source_path_idx", data.get("base_path_idx", 0))
    # NEW: shared temporal batch length for all temporal filters (time & pod)
    shared_temporal_bs = data.get("temporal_batch_filter", None)

    if camera is None:
        return jsonify({"error": "camera required"}), 400
    if start_idx < 1 or start_idx > config.num_images:
        return (
            jsonify({"error": f"start_idx out of range (1..{config.num_images})"}),
            400,
        )
    if (
        not isinstance(source_path_idx, int)
        or source_path_idx < 0
        or source_path_idx >= len(config.source_paths)
    ):
        return (
            jsonify(
                {
                    "error": f"Invalid source_path_idx/base_path_idx (0..{len(config.source_paths) - 1})"
                }
            ),
            400,
        )

    # Update backend config filters with what the frontend sends (in-memory for this server process only)
    if isinstance(filters, list):
        # If a shared temporal batch size is provided, apply it to all temporal filters
        if isinstance(shared_temporal_bs, int) and shared_temporal_bs > 0:
            _new_filters = []
            for f in filters:
                try:
                    ftype = str(f.get("type")).lower()
                except Exception:
                    ftype = ""
                if ftype in ("time", "pod"):
                    nf = dict(f)
                    nf["batch_size"] = int(shared_temporal_bs)
                    _new_filters.append(nf)
                else:
                    _new_filters.append(f)
            filters = _new_filters
        config.data["filters"] = filters
        print(f"/filter: updated config.filters = {config.filters}")
    else:
        print("/filter: no filters provided; using existing config.filters")
        filters = config.filters

    # Determine max temporal batch size across temporal filters
    def is_temporal(ftype: str) -> bool:
        return ftype in ("time", "pod")

    temporal_sizes = []
    for f in filters or []:
        ftype = str(f.get("type") or "").lower()
        if is_temporal(ftype):
            bs = f.get("batch_size", None)
            if isinstance(bs, int) and bs > 0:
                temporal_sizes.append(bs)
    max_batch_size = max(temporal_sizes) if temporal_sizes else max(1, count)
    print(f"/filter: computed max temporal batch_size = {max_batch_size}")

    # Compute batch window that contains start_idx using non-overlapping blocks of size max_batch_size
    def compute_batch_window(target_idx: int, batch_size: int, total: int):
        block = (target_idx - 1) // batch_size
        s = block * batch_size + 1
        e = min(s + batch_size - 1, total)
        return s, e

    batch_start, batch_end = compute_batch_window(
        start_idx, max_batch_size, config.num_images
    )
    indices = list(range(batch_start, batch_end + 1))
    print(
        f"/filter: temporal window [{batch_start}..{batch_end}] for target {start_idx}"
    )

    # Resolve camera path
    cam_folder = cam_folder_key(camera)
    camera_path = config.source_paths[source_path_idx] / cam_folder

    def load_pairs():
        pairs = [
            read_pair(idx, camera_path, config) for idx in indices
        ]  # list of (2,H,W) np arrays
        import numpy as _np

        arr = _np.stack(pairs, axis=0)  # (N, 2, H, W)
        # chunk first axis by max_batch_size (or the actual length if shorter)
        chunks_n = min(max_batch_size, arr.shape[0])
        darr = da.from_array(arr, chunks=(chunks_n, 2, *config.image_shape))
        return darr

    def process_and_store():
        global processing
        try:
            darr = load_pairs()
            # Apply filter stack with per-filter temporal batching
            processed_all = filter_images(
                darr, config, filters_override=filters
            ).compute()  # (N,2,H,W)
            original_all = darr.compute()  # (N,2,H,W)

            k = cache_key(source_path_idx, camera)
            processed_store["original"].setdefault(k, {})
            processed_store["processed"].setdefault(k, {})

            # Store each absolute frame in the cache
            for rel, abs_idx in enumerate(indices):
                # Each entry is shape (2,H,W)
                processed_store["original"][k][abs_idx] = original_all[rel]
                processed_store["processed"][k][abs_idx] = processed_all[rel]
        except Exception as e:
            print(f"Error during /filter processing: {e}")
        finally:
            processing = False

    processing = True
    thread = threading.Thread(target=process_and_store, daemon=True)
    thread.start()
    return jsonify({"status": "processing"})


@app.route("/get_processed_pair", methods=["GET"])
def get_processed_pair():
    """
    Query params:
      - frame: absolute 1-based frame index (required)
      - type: 'original' or 'processed' (default 'processed')
      - camera: camera folder name or number (required)
      - source_path_idx: integer index into config.source_paths (default 0)
    Returns: PNGs as base64 in JSON if cached; 404 if not cached.
    """
    frame = request.args.get("frame", type=int)
    typ = request.args.get("type", "processed")
    camera = request.args.get("camera")
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)

    if typ not in ["original", "processed"]:
        return jsonify({"error": "type must be 'original' or 'processed'"}), 400
    if frame is None or camera is None:
        return jsonify({"error": "frame and camera are required"}), 400
    if source_path_idx < 0 or source_path_idx >= len(config.source_paths):
        return (
            jsonify(
                {
                    "error": f"source_path_idx out of range (0..{len(config.source_paths) - 1})"
                }
            ),
            400,
        )

    k = cache_key(source_path_idx, camera)
    bucket = processed_store.get(typ, {}).get(k, {})
    pair = bucket.get(frame)

    if pair is None:
        return (
            jsonify(
                {"error": "processed frame not cached; run /filter for this index"}
            ),
            404,
        )

    # pair shape: (2, H, W)
    img_a = pair[0]
    img_b = pair[1]
    png_a = numpy_to_png_bytes(img_a)
    png_b = numpy_to_png_bytes(img_b)
    b64_a = base64.b64encode(png_a).decode("utf-8")
    b64_b = base64.b64encode(png_b).decode("utf-8")
    return jsonify({"A": b64_a, "B": b64_b})


# Endpoint to check processing status
@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"processing": processing})


# New endpoint: render a vector field from .mat files and return a PNG (base64)
# Query params:
#   data: absolute or relative path to data .mat file containing 'piv_result'
#   coords: absolute or relative path to coordinates .mat containing 'coordinates'
#   var: 'ux' or 'uy' (which scalar to plot)
#   run: 1-based pass index (default 1)
#   lower_limit, upper_limit: optional floats for fixed color scale
#
# Response: { image: <base64-png>, meta: { run, var, width, height } }


@app.route("/update_paths", methods=["POST"])
def update_paths():
    """
        Expects JSON body:
            - base_paths: list of base path strings
            - source_paths: list of source path strings
            Optional (will also update formats in config.yaml):
            - image_format: string OR list of two strings (raw image patterns)
            - vector_format: string (processed file pattern) or list with one string
            - calibration_image_format: string calibration image pattern
    Updates config in-memory and writes to config.yaml.
    """
    data = request.get_json() or {}
    base_paths = data.get("base_paths")
    source_paths = data.get("source_paths")
    if not isinstance(base_paths, list) or not isinstance(source_paths, list):
        return jsonify({"error": "base_paths and source_paths must be lists"}), 400

    # Update in-memory config
    config.data["paths"]["base_paths"] = base_paths
    config.data["paths"]["source_paths"] = source_paths

    # --- Optional format updates ---
    # Raw image formats
    if "image_format" in data:
        img_fmt = data.get("image_format")
        if isinstance(img_fmt, (list, tuple)):
            # store as list of strings
            config.data.setdefault("images", {})["image_format"] = list(img_fmt)
            # two formats implies not time-resolved
            config.data.setdefault("images", {})["time_resolved"] = False
        elif isinstance(img_fmt, str) and img_fmt.strip():
            # single pattern (time-resolved)
            config.data.setdefault("images", {})["image_format"] = img_fmt.strip()
            config.data.setdefault("images", {})["time_resolved"] = True
    # Vector / processed format
    if "vector_format" in data:
        vec_fmt = data.get("vector_format")
        if isinstance(vec_fmt, (list, tuple)):
            if vec_fmt:
                config.data.setdefault("images", {})["vector_format"] = [
                    str(vec_fmt[0])
                ]
        elif isinstance(vec_fmt, str) and vec_fmt.strip():
            config.data.setdefault("images", {})["vector_format"] = [vec_fmt.strip()]
    # Calibration image format
    if "calibration_image_format" in data:
        calib_fmt = data.get("calibration_image_format")
        if isinstance(calib_fmt, str) and calib_fmt.strip():
            config.data.setdefault("calibration", {})[
                "image_format"
            ] = calib_fmt.strip()

    # Write to config.yaml
    config_path = "config.yaml"
    try:
        with open(config_path, "w") as f:
            yaml.dump(
                config.data,
                f,
                default_flow_style=False,
                sort_keys=False,
            )
    except Exception as e:
        return jsonify({"error": f"Failed to write config.yaml: {e}"}), 500

    return jsonify(
        {
            "status": "success",
            "base_paths": base_paths,
            "source_paths": source_paths,
            "image_format": config.data.get("images", {}).get("image_format"),
            "vector_format": config.data.get("images", {}).get("vector_format"),
            "calibration_image_format": config.data.get("calibration", {}).get(
                "image_format"
            ),
            "time_resolved": config.data.get("images", {}).get("time_resolved"),
        }
    )


@app.route("/config", methods=["GET"])
def get_config():
    """Return the full current YAML configuration (read-only)."""
    try:
        return jsonify(config.data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/update_images", methods=["POST"])
def update_images():
    """Update image-related entries (num_images, shape, image_type) in YAML."""
    data = request.get_json() or {}
    images_block = config.data.setdefault("images", {})
    changed = {}
    if "num_images" in data:
        try:
            n = int(data["num_images"])
            if n > 0:
                images_block["num_images"] = n
                changed["num_images"] = n
        except Exception:
            return jsonify({"error": "num_images must be positive int"}), 400
    if "shape" in data:
        shp = data["shape"]
        if (
            isinstance(shp, (list, tuple))
            and len(shp) == 2
            and all(isinstance(v, int) and v > 0 for v in shp)
        ):
            images_block["shape"] = list(shp)
            changed["shape"] = list(shp)
        else:
            return jsonify({"error": "shape must be [H,W] positive ints"}), 400
    if "image_type" in data:
        it = data["image_type"]
        if isinstance(it, str) and it.strip():
            images_block["image_type"] = it.strip()
            changed["image_type"] = it.strip()

    # Persist
    try:
        with open("config.yaml", "w") as f:
            yaml.dump(
                config.data,
                f,
                default_flow_style=False,
                sort_keys=False,
            )
    except Exception as e:
        return jsonify({"error": f"Failed to write config.yaml: {e}"}), 500
    return jsonify({"status": "success", "images": images_block, "changed": changed})


@app.route("/update_instantaneous", methods=["POST"])
def update_instantaneous():
    """Update instantaneous_piv (window_size, overlap, runs) in YAML."""
    data = request.get_json() or {}
    inst = config.data.setdefault("instantaneous_piv", {})
    changed = {}
    if "window_size" in data:
        ws = data["window_size"]
        if isinstance(ws, list) and all(
            isinstance(p, (list, tuple)) and len(p) == 2 for p in ws
        ):
            inst["window_size"] = [list(map(int, p)) for p in ws]
            changed["window_size"] = inst["window_size"]
        else:
            return jsonify({"error": "window_size must be list of [x,y]"}), 400
    if "overlap" in data:
        ov = data["overlap"]
        if isinstance(ov, list) and all(isinstance(v, (int, float)) for v in ov):
            inst["overlap"] = [int(v) for v in ov]
            changed["overlap"] = inst["overlap"]
        else:
            return jsonify({"error": "overlap must be list of numbers"}), 400
    if "runs" in data:
        rn = data["runs"]
        if isinstance(rn, list) and all(isinstance(v, int) for v in rn):
            inst["runs"] = rn
            changed["runs"] = rn
        else:
            return jsonify({"error": "runs must be list of ints"}), 400
    try:
        with open("config.yaml", "w") as f:
            yaml.dump(config.data, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        return jsonify({"error": f"Failed to write config.yaml: {e}"}), 500
    return jsonify({"status": "success", "instantaneous_piv": inst, "changed": changed})


# ------------- Masking Endpoints -------------

# Endpoint to serve raw image for masking (with colormap)
@app.route("/get_raw_image", methods=["GET"])
def get_raw_image():
    basepath_idx = request.args.get("basepath_idx", default=0, type=int)
    camera = request.args.get("camera", default="Cam1", type=str)
    index = request.args.get("index", default=0, type=int)
    # Simplified: frame is always 'A' or 'B' (uppercase)
    frame = request.args.get("frame", default="A")
    if frame not in ("A", "B"):
        return jsonify({"error": "frame must be 'A' or 'B'"}), 400
    frame_idx = 0 if frame == "A" else 1

    # Get base path from config
    try:
        base_paths = config.source_paths
        if basepath_idx < 0 or basepath_idx >= len(base_paths):
            return jsonify({"error": "basepath_idx out of range"}), 400
        camera_path = base_paths[basepath_idx] / camera
        # Read image pair (A/B)
        pair = read_pair(index, camera_path, config)
        if frame_idx >= pair.shape[0]:
            return jsonify({"error": "requested frame index out of bounds"}), 400
        img = pair[frame_idx]
    except Exception as e:
        print("Exception in get_raw_image:", e)
        return jsonify({"error": str(e)}), 500

    vmin = float(np.min(img))
    vmax = float(np.max(img))

    im_pil = Image.fromarray(img)
    buf = BytesIO()
    im_pil.save(buf, format="PNG")
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode("utf-8")

    return jsonify({"image": b64_img, "vmin": vmin, "vmax": vmax})


@app.route("/save_mask_array", methods=["POST"])
def upload_mask():
    """
    Expects JSON:
      {
        meta: {
          basePathIdx: int,
          camera: str,
          index: int,
          frame: "A"|"B"
        }
        width: int
        height: int
        data: [0|1, ...] length = width*height
      }
    Converts to boolean numpy array of shape (height, width) and saves to .mat file.
    Returns summary.
    """
    payload = request.get_json(silent=True) or {}
    width = payload.get("width")
    height = payload.get("height")
    flat = payload.get("data")
    meta = payload.get("meta", {})
    polygons = payload.get("polygons", None)

    # Validate dimensions
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        return jsonify({"error": "width and height must be positive integers"}), 400
    if not isinstance(flat, list):
        return jsonify({"error": "data must be a list"}), 400
    expected = width * height
    if len(flat) != expected:
        return (
            jsonify(
                {
                    "error": f"data length {len(flat)} does not match width*height={expected}"
                }
            ),
            400,
        )

    # Convert to numpy boolean mask
    try:
        arr = np.asarray(flat, dtype=np.uint8)
    except Exception as e:
        return jsonify({"error": f"failed to convert data list to array: {e}"}), 400

    if not np.isin(arr, (0, 1)).all():
        return jsonify({"error": "data must contain only 0 or 1 values"}), 400

    try:
        mask = arr.reshape((height, width)).astype(bool)
    except Exception as e:
        return jsonify({"error": f"failed to reshape to (height,width): {e}"}), 400

    # Extract directory details from meta
    try:
        basePathIdx = meta["basePathIdx"]
        camera = meta["camera"]
        index = meta["index"]
        frame = meta["frame"]
    except Exception as e:
        return jsonify({"error": f"missing or invalid meta fields: {e}"}), 400

    # Validate frame
    if frame not in ("A", "B"):
        return jsonify({"error": 'frame must be "A" or "B"'}), 400

    # Get base path from config
    try:
        base_paths = config.source_paths
        if basePathIdx < 0 or basePathIdx >= len(base_paths):
            return jsonify({"error": "basePathIdx out of range"}), 400
        camera_path = base_paths[basePathIdx] / camera
    except Exception as e:
        return jsonify({"error": f"failed to resolve base path: {e}"}), 400

    # Save mask to mat file
    try:
        mask_filename = f"mask_{frame}_{index}.mat"
        mask_path = camera_path / mask_filename
        save_mask_to_mat(
            mask_path, mask, np.asarray(polygons)
        )  # Pass polygons to save_mask_to_mat
    except Exception as e:
        print("Exception in save_mask_array:", e)
        return jsonify({"error": f"failed to save mask to mat: {e}"}), 500

    true_count = int(mask.sum())
    return jsonify(
        {
            "status": "ok",
            "shape": [height, width],
            "true_count": true_count,
            "fraction_true": true_count / expected if expected else 0.0,
            "meta": meta,
        }
    )


@app.route("/load_mask", methods=["GET"])
def load_mask():
    """
    Loads a mask and polygon data from a .mat file.
    Query params:
      - path: full path to mask .mat file (preferred)
      - basepath_idx, camera, index, frame: optional, used to construct path if 'path' not given
    Returns: { mask: [0|1,...], width, height, polygons: [...] }
    """
    path = request.args.get("path", default=None, type=str)
    # Optionally reconstruct path if not provided
    if not path or not Path(path).exists():
        # Try to build path from meta info
        try:
            basepath_idx = int(request.args.get("basepath_idx", 0))
            camera = request.args.get("camera")
            index = int(request.args.get("index", 0))
            frame = request.args.get("frame")
            if frame not in ("A", "B"):
                return jsonify({"error": 'frame must be "A" or "B"'}), 400
            base_paths = config.source_paths
            if basepath_idx < 0 or basepath_idx >= len(base_paths):
                return jsonify({"error": "basepath_idx out of range"}), 400
            camera_path = base_paths[basepath_idx] / camera
            mask_filename = f"mask_{frame}_{index}.mat"
            path = str(camera_path / mask_filename)
        except Exception as e:
            return jsonify({"error": f"Could not resolve mask path: {e}"}), 400

    if not Path(path).exists():
        return jsonify({"error": f"Mask file not found: {path}"}), 404

    try:
        mask, polygons = read_mask_from_mat(path)
        # Convert all numpy arrays in polygons to lists for JSON serialization

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

# ------------- Run PIV Endpoints -------------

@app.route("/run_piv", methods=["POST"])
def run_piv():
    """
    Placeholder for run_piv functionality.
    Accepts POST requests and replies with 200 OK.
    Prints the incoming request JSON.
    """
    req_json = request.get_json(silent=True)
    print("Received /run_piv request:", req_json)
    return jsonify({"status": "ok", "message": "run_piv placeholder"}), 200


@app.route("/cancel_run", methods=["POST"])
def cancel_run():
    """
    Placeholder for cancel_run functionality.
    Accepts POST requests and replies with 200 OK.
    """
    return jsonify({"status": "ok", "message": "cancel_run placeholder"}), 200


@app.route("/check_status", methods=["GET"])
def check_status():
    """
    Returns a random integer between 0 and 100.
    """
    if not hasattr(check_status, "counter"):
        check_status.counter = 0
    value = check_status.counter
    check_status.counter = (check_status.counter + 10) % 110
    return jsonify({"status": value})


@app.route("/check_status_image", methods=["GET"])
def check_status_image():
    """
    Receives the same parameters as /get_raw_image.
    For now, always returns morgan.jpeg as base64 PNG.
    """
    # Accept parameters for future compatibility
    # basepath_idx = request.args.get("basepath_idx", default=0, type=int)
    # camera = request.args.get("camera", default="Cam1", type=str)
    # index = request.args.get("index", default=0, type=int)
    # frame = request.args.get("frame", default="A")
    # Placeholder: always send morgan.jpeg
    try:
        img_path = Path("morgan.jpeg")  # PLACEHOLDER
        if not img_path.exists():  # PLACEHOLDER
            return jsonify({"error": "morgan.jpeg not found"}), 404  # PLACEHOLDER

        im_pil = Image.open(img_path)
        buf = BytesIO()
        im_pil.save(buf, format="PNG")
        buf.seek(0)
        b64_img = base64.b64encode(buf.read()).decode("utf-8")
        # Optionally, send dummy vmin/vmax
        return jsonify({"image": b64_img, "vmin": 0, "vmax": 255})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------- Calibration Endpoints -------------


def _resolve_calibration_image(source_path_idx: int, camera: str, index: int = 1):
    cam_folder = cam_folder_key(camera)
    if source_path_idx < 0 or source_path_idx >= len(config.source_paths):
        raise ValueError("source_path_idx out of range")
    source_root = config.source_paths[source_path_idx]
    calib_dir = source_root / "calibration" / cam_folder
    # Accept both nested structure (Calibration/Cam1/Calib00001.tif) or flat (Calibration/Cam1.tif)
    config.calibration_image_format
    filename = config.calibration_filename(index)
    img_path = calib_dir / filename
    if not img_path.exists():
        # Fallback: try directly under Calibration folder without extra cam subfolder
        alt_dir = source_root / "calibration" / cam_folder
        alt_path = alt_dir / filename
        if alt_path.exists():
            img_path = alt_path
    if not img_path.exists():
        raise FileNotFoundError(f"calibration image not found: {img_path}")
    return img_path


@app.route("/calibration/get_image", methods=["GET"])
def calibration_get_image():
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = request.args.get("camera", default="Cam1")
    index = request.args.get("index", default=1, type=int)
    try:
        img_path = _resolve_calibration_image(source_path_idx, camera, index)
        img = calib_load_image(img_path)
        # Normalize to 0-255 uint8 for display
        disp = img - img.min()
        if disp.max() > 0:
            disp = disp / disp.max()
        disp8 = (disp * 255).astype(np.uint8)
        b64 = base64.b64encode(numpy_to_png_bytes(disp8)).decode("utf-8")
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


@app.route("/calibration/detect_dots", methods=["GET"])
def calibration_detect_dots():
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    camera = request.args.get("camera", default="Cam1")
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


@app.route("/calibration/compute", methods=["POST"])
def calibration_compute():
    data = request.get_json() or {}
    source_path_idx = int(data.get("source_path_idx", 0))
    camera = data.get("camera", "Cam1")
    dot_distance_mm = float(data.get("dot_distance_mm", 28.9))
    grid_tolerance = float(data.get("grid_tolerance", 0.5))
    ransac_threshold = float(data.get("ransac_threshold", 3.0))
    # Points provided by frontend after user clicks (already snapped) in pixel coordinates
    datum = data.get("datum")
    right = data.get("right")
    above = data.get("above")
    if not (datum and right and above):
        return jsonify({"error": "datum, right, above points required"}), 400
    k = cache_key(source_path_idx, camera)
    cache = calibration_cache.get(k)
    if not cache or "image" not in cache or "dots" not in cache:
        return jsonify({"error": "Calibration image/dots not loaded"}), 400
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
        dewarped_png = base64.b64encode(
            numpy_to_png_bytes((dewarped_disp * 255).astype(np.uint8))
        ).decode("utf-8")
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

if __name__ == "__main__":
    app.run(debug=True)
