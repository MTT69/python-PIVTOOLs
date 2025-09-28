import threading

import dask.array as da
import numpy as np
import yaml
from dask import config as dask_config
from flask import Flask, jsonify, request
from flask_cors import CORS
from loguru import logger

from calibration.app.views import calibration_bp
from common.utils import camera_folder, camera_number, numpy_to_png_base64
from config import get_config, reload_config
from image_handling.load_images import read_pair
from masking.app.views import masking_bp
from paths import get_data_paths
from plotting.app.views import vector_plot_bp
from post_processing.POD.app.views import POD_bp
from pre_processing.filters import filter_images
from stereo_reconstruction.app.views import stereo_bp
from video_maker.app.views import video_maker_bp

app = Flask(__name__)
CORS(app)
dask_config.set(scheduler="threads")

app.register_blueprint(vector_plot_bp)
app.register_blueprint(masking_bp)
app.register_blueprint(POD_bp)
app.register_blueprint(calibration_bp)
app.register_blueprint(video_maker_bp)
app.register_blueprint(stereo_bp)

# --- In-memory stores ---
processed_store = {"original": {}, "processed": {}}
processing = False

# --- Utility Functions ---


def cam_folder_key(camera):  # backward compat helper
    return camera_folder(camera)


def cache_key(source_path_idx, camera):
    return (int(source_path_idx), str(camera))


def get_cached_pair(frame, typ, camera, source_path_idx):
    """Fetch a cached pair (A, B) for given frame/type/camera/source_path_idx."""
    k = cache_key(source_path_idx, camera)
    bucket = processed_store.get(typ, {}).get(k, {})
    pair = bucket.get(frame)
    if pair is None:
        return None, None
    return numpy_to_png_base64(pair[0]), numpy_to_png_base64(pair[1])


def compute_batch_window(target_idx: int, batch_size: int, total: int):
    block = (target_idx - 1) // batch_size
    s = block * batch_size + 1
    e = min(s + batch_size - 1, total)
    return s, e


def recursive_update(d, u):
    for k, v in u.items():
        # Remove debug print statements
        # print(f"Updating key: {k}, value type: {type(v)}, current value: {d.get(k, 'MISSING')}")
        if isinstance(v, dict):
            if not isinstance(d.get(k), dict):
                # print(f"Key '{k}' is missing or not a dict, initializing as dict.")
                d[k] = {}
            recursive_update(d[k], v)
        else:
            d[k] = v


def get_active_calibration_params(cfg):
    """
    Returns (active_method, params_dict) from config['calibration'].
    Updated to work with new calibration structure.
    """
    cal = cfg.data.get("calibration", {})
    active = cal.get("active", "pinhole")
    params = cal.get(active, {})
    return active, params


def get_calibration_method_params(cfg, method: str):
    """
    Get parameters for a specific calibration method.
    """
    cal = cfg.data.get("calibration", {})
    return cal.get(method, {})


# --- Endpoints ---


@app.route("/get_frame_pair", methods=["GET"])
def get_frame_pair():
    cfg = get_config()
    camera = request.args.get("camera", type=int)
    idx = request.args.get("idx", type=int)
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    source_path = cfg.source_paths[source_path_idx] / camera_folder(camera)
    try:
        pair = read_pair(idx, source_path, cfg)
    except FileNotFoundError as e:
        return jsonify({"error": "File not found", "file": str(e)}), 404

    return jsonify(
        {"A": numpy_to_png_base64(pair[0]), "B": numpy_to_png_base64(pair[1])}
    )


@app.route("/filter", methods=["POST"])
def filter_images_endpoint():
    global processing
    data = request.get_json() or {}
    cfg = get_config()
    camera = camera_number(data.get("camera"))
    start_idx = int(data.get("start_idx", 1))
    filters = data.get("filters", None)
    source_path_idx = data.get("source_path_idx")
    temporal_batch_filter = data.get("temporal_batch_filter")
    if filters is not None:
        cfg.data["filters"] = filters
    # Derive desired temporal window length
    if temporal_batch_filter:
        batch_length = int(temporal_batch_filter)
        batch_len_reason = "request.temporal_batch_filter"
    else:
        # Fall back to largest batch_size among temporal filters if provided
        if filters:
            temporal_sizes = [
                int(f.get("batch_size", 1))
                for f in filters
                if str(f.get("type", "")).lower() in ("time", "pod")
            ]
            if temporal_sizes:
                batch_length = max(temporal_sizes)
                batch_len_reason = "max(filter.batch_size)"
            else:
                batch_length = int(data.get("batch_length", cfg.piv_chunk_size))
                batch_len_reason = "fallback.batch_length_or_config"
        else:
            batch_length = int(data.get("batch_length", cfg.piv_chunk_size))
            batch_len_reason = "fallback.no_filters"
    if batch_length < 1:
        batch_length = 1
    batch_start, batch_end = compute_batch_window(
        start_idx, batch_length, cfg.num_images
    )
    indices = list(range(batch_start, batch_end + 1))
    source_path = cfg.source_paths[source_path_idx] / camera_folder(camera)

    def load_pairs():
        pairs = [read_pair(i, source_path, cfg) for i in indices]
        arr = np.stack(pairs, axis=0)
        # Create single temporal chunk covering entire window so time/POD operate over it
        return da.from_array(arr, chunks=(arr.shape[0], 2, *cfg.image_shape))

    def process_and_store():
        global processing
        logger.debug("/filter processing thread started")
        try:
            darr = load_pairs()
            processed_all = filter_images(darr, cfg, filters_override=filters).compute()
            original_all = darr.compute()
            k = cache_key(source_path_idx, camera)
            processed_store["original"].setdefault(k, {})
            processed_store["processed"].setdefault(k, {})
            for rel, abs_idx in enumerate(indices):
                processed_store["original"][k][abs_idx] = original_all[rel]
                processed_store["processed"][k][abs_idx] = processed_all[rel]
        except Exception as e:
            logger.exception(f"Error during /filter processing: {e}")
        finally:
            processing = False
            logger.debug("/filter processing thread finished (processing=False)")

    processing = True
    threading.Thread(target=process_and_store, daemon=True).start()
    return jsonify(
        {
            "status": "processing",
            "window_start": batch_start,
            "window_end": batch_end,
            "window_size": len(indices),
            "batch_length": batch_length,
            "batch_length_reason": batch_len_reason,
        }
    )


@app.route("/get_processed_pair", methods=["GET"])
def get_processed_pair():
    frame = request.args.get("frame", type=int)
    typ = request.args.get("type", "processed")
    camera = camera_number(request.args.get("camera"))
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    b64_a, b64_b = get_cached_pair(frame, typ, camera, source_path_idx)
    return jsonify({"status": "ok", "A": b64_a, "B": b64_b})


@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"processing": processing})


@app.route("/config", methods=["GET"])
def config_endpoint():
    cfg = get_config()
    # Already returns full nested config as JSON
    return jsonify(cfg.data)


@app.route("/update_config", methods=["POST"])
def update_config():  # noqa: D401 simple update
    data = request.get_json() or {}
    cfg = get_config()

    # Special handling: merge post_processing entries by type and deep-merge their settings
    incoming_pp = data.get("post_processing", None)
    if isinstance(incoming_pp, list):
        current_pp = list(cfg.data.get("post_processing", []) or [])
        # Build index by type for current entries
        idx_by_type = {}
        for i, entry in enumerate(current_pp):
            t = (entry or {}).get("type")
            if t is not None and t not in idx_by_type:
                idx_by_type[t] = i

        def deep_merge_dict(a, b):
            for k, v in (b or {}).items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    deep_merge_dict(a[k], v)
                else:
                    a[k] = v
            return a

        for new_entry in incoming_pp:
            if not isinstance(new_entry, dict):
                continue
            t = new_entry.get("type")
            if t in idx_by_type:
                i = idx_by_type[t]
                cur = current_pp[i] or {}
                # Merge non-settings keys shallowly
                for k, v in new_entry.items():
                    if k == "settings" and isinstance(v, dict):
                        cur.setdefault("settings", {})
                        deep_merge_dict(cur["settings"], v)
                    elif k != "type":
                        cur[k] = v
                current_pp[i] = cur
            else:
                # New type -> append
                current_pp.append(new_entry)

        # Replace the post_processing in data with merged result to allow generic recursion below
        data = dict(data)
        data["post_processing"] = current_pp

    recursive_update(cfg.data, data)
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg.data, f, default_flow_style=False, sort_keys=False)
    reload_config()
    return jsonify({"status": "success", "updated": data})


@app.route("/run_piv", methods=["POST"])
def run_piv():
    return jsonify({"status": "ok", "message": "run_piv acknowledged"}), 200


@app.route("/cancel_run", methods=["POST"])
def cancel_run():
    return jsonify({"status": "ok", "message": "cancel_run placeholder"}), 200


@app.route("/get_uncalibrated_count", methods=["GET"])
def get_uncalibrated_count():
    cfg = get_config()
    basepath_idx = request.args.get("basepath_idx", default=0, type=int)
    cam = camera_number(request.args.get("camera", default=1, type=int))
    type_name = request.args.get("type", default="instantaneous")
    base_paths = cfg.base_paths
    base = base_paths[basepath_idx]
    num_images = cfg.num_images
    paths = get_data_paths(base, num_images, cam, type_name, use_uncalibrated=True)
    folder_uncal = paths["data_dir"]
    vector_fmt = cfg.vector_format
    expected_names = set([vector_fmt % i for i in range(1, num_images + 1)])
    found = (
        [
            p.name
            for p in sorted(folder_uncal.iterdir())
            if p.is_file() and p.name in expected_names
        ]
        if folder_uncal.exists() and folder_uncal.is_dir()
        else []
    )
    percent = int((len(found) / num_images) * 100) if num_images else 0
    return jsonify({"count": len(found), "files": found, "percent": percent})


if __name__ == "__main__":
    # Optionally set logger level elsewhere; keep debug for dev
    app.run(debug=True)
