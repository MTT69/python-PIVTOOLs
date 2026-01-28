import sys
from pathlib import Path

# Add parent directory to path so pivtools_gui can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
import time
import threading
import webbrowser
import dask
import dask.array as da
import numpy as np
import yaml
from dask import config as dask_config
from flask import Blueprint, Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from loguru import logger
import os
from pivtools_gui.calibration.app.views import calibration_bp
from pivtools_core.config import get_config, reload_config, Config
from pivtools_core.image_handling.load_images import read_pair, load_mask_for_camera
from pivtools_core.image_handling.path_utils import build_piv_camera_path, validate_images_generic
from pivtools_gui.masking.app.views import masking_bp
from pivtools_core.paths import get_data_paths
from pivtools_gui.piv_runner import get_runner
from pivtools_gui.plotting.app.plotting_views import vector_plot_bp
# Old per-file transform storage (kept for backwards compatibility)
from pivtools_gui.plotting.app.transform_views import transform_bp
# New config-based transform storage
from pivtools_gui.transforms.app.transform_views import transform_bp as transform_new_bp
from pivtools_cli.preprocessing.preprocess import preprocess_images, apply_filters_to_batch
# from pivtools_gui.stereo_reconstruction.app.views import stereo_bp
from pivtools_gui.utils import camera_folder, camera_number, numpy_to_png_base64, numpy_to_base64
from pivtools_gui.vector_statistics.app.views import statistics_bp
from pivtools_gui.vector_merging.app.views import merging_bp
from pivtools_gui.video_maker.app.views import video_maker_bp

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)
dask_config.set(scheduler="threads")

# Create API blueprint with /backend prefix
api_bp = Blueprint('api', __name__, url_prefix='/backend')

# Register existing blueprints with /backend prefix
app.register_blueprint(vector_plot_bp, url_prefix='/backend/plot')
app.register_blueprint(transform_bp, url_prefix='/backend/plot')
app.register_blueprint(masking_bp, url_prefix='/backend')
app.register_blueprint(calibration_bp, url_prefix='/backend')
app.register_blueprint(video_maker_bp, url_prefix='/backend/video')
# app.register_blueprint(stereo_bp, url_prefix='/backend')
app.register_blueprint(statistics_bp, url_prefix='/backend')
app.register_blueprint(merging_bp, url_prefix='/backend')
# --- In-memory stores ---
processed_store = {"processed": {}}
processing = False

# Raw image cache: {cache_key: (pair_array, base64_A, base64_B, stats, last_access_time)}
# Cache key includes format, so jpeg and png entries are separate
raw_image_cache = {}
RAW_CACHE_MAX_SIZE = 20  # Maximum number of frame pairs to cache
FRAME_1_KEYS = set()  # Track frame 1 keys to pin them in cache

# Thread safety for cache access
import threading
cache_lock = threading.Lock()

# --- Utility Functions ---

def manage_cache_size():
    """LRU eviction with frame 1 pinning. Thread-safe."""
    global raw_image_cache
    with cache_lock:
        if len(raw_image_cache) <= RAW_CACHE_MAX_SIZE:
            return

        # Sort by access time (5th element), excluding pinned frame 1 entries
        evictable = []
        for k, v in raw_image_cache.items():
            if k in FRAME_1_KEYS:
                continue  # Never evict frame 1
            # v = (pair, b64_a, b64_b, stats, last_access_time)
            access_time = v[4] if len(v) > 4 else 0
            evictable.append((k, access_time))

        evictable.sort(key=lambda x: x[1])  # Sort by access time (oldest first)

        # Remove oldest until under limit
        to_remove = len(raw_image_cache) - RAW_CACHE_MAX_SIZE
        for key, access_time in evictable[:to_remove]:
            del raw_image_cache[key]


def _log_cache_contents():
    """Log cache contents: raw frames + processed frames (if any)."""
    with cache_lock:
        # Get raw frame indices
        raw_frames = sorted([key[2] for key in raw_image_cache.keys()]) if raw_image_cache else []

        # Get processed frame indices
        proc_frames = sorted(processed_store.get("processed", {}).keys()) if processed_store.get("processed") else []

        if not raw_frames:
            return

        # Format: "Cache: raw 1-10 | processed 1-10" or "Cache: raw 1-10" if no processed
        raw_str = f"raw {min(raw_frames)}-{max(raw_frames)}"
        if proc_frames:
            proc_str = f"processed {min(proc_frames)}-{max(proc_frames)}"
            logger.info(f"Cache: {raw_str} | {proc_str}")
        else:
            logger.info(f"Cache: {raw_str}")


def cam_folder_key(camera, cfg):
    """Get camera folder using config to respect custom subfolders."""
    return cfg.get_camera_folder(camera_number(camera))


def make_raw_cache_key(source_path_idx: int, camera: int, idx: int, img_format: str, cfg) -> tuple:
    """Generate consistent cache key for raw images. Include all parameters that affect output."""
    image_type = cfg.image_type
    # For .set files, source_path IS the .set file; for others, may need camera folder
    if image_type in ("lavision_set", "lavision_im7"):
        source_path = cfg.source_paths[source_path_idx]
    else:
        folder = cfg.get_camera_folder(camera)
        source_path = cfg.source_paths[source_path_idx] / folder if folder else cfg.source_paths[source_path_idx]
    return (str(source_path), camera, idx, img_format)


def cache_key(source_path_idx, camera, cfg):
    """Generate cache key for processed images (no frame index - stores dict of frames)."""
    image_type = cfg.image_type
    # For .set files, source_path IS the .set file; for others, may need camera folder
    if image_type in ("lavision_set", "lavision_im7"):
        source_path = cfg.source_paths[source_path_idx]
    else:
        folder = cfg.get_camera_folder(camera_number(camera))
        source_path = cfg.source_paths[source_path_idx] / folder if folder else cfg.source_paths[source_path_idx]
    return (str(source_path), str(camera))


def get_percentile_stats(img_array):
    """Calculate vmin/vmax as percentages (0-100) of the data range.

    This allows the frontend to work with any bit depth since the image
    is normalized to 8-bit before sending, and contrast is expressed as
    percentages of the display range.
    """
    p1 = float(np.percentile(img_array, 1))
    p99 = float(np.percentile(img_array, 99))
    img_min = float(img_array.min())
    img_max = float(img_array.max())
    data_range = img_max - img_min
    if data_range > 0:
        vmin_pct = 100.0 * (p1 - img_min) / data_range
        vmax_pct = 100.0 * (p99 - img_min) / data_range
    else:
        vmin_pct = 0.0
        vmax_pct = 100.0
    return {"vmin_pct": round(vmin_pct, 2), "vmax_pct": round(vmax_pct, 2)}


def get_cached_pair(frame, typ, camera, source_path_idx, cfg, auto_limits=False):
    """Fetch a cached pair (A, B) for given frame/type/camera/source_path_idx."""
    k = cache_key(source_path_idx, camera, cfg)
    bucket = processed_store.get(typ, {}).get(k, {})
    pair = bucket.get(frame)
    if pair is None:
        return None, None, None

    b64_a = numpy_to_png_base64(pair[0])
    b64_b = numpy_to_png_base64(pair[1])

    stats = None
    if auto_limits:
        stats = {
            "A": get_percentile_stats(pair[0]),
            "B": get_percentile_stats(pair[1])
        }

    return b64_a, b64_b, stats


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
    active = cal.get("active", "dotboard")
    params = cal.get(active, {})
    return active, params


def get_calibration_method_params(cfg, method: str):
    """
    Get parameters for a specific calibration method.
    """
    cal = cfg.data.get("calibration", {})
    return cal.get(method, {})


def _preload_surrounding_frames(source_path_idx: int, camera: int, current_idx: int, cfg, window: int = 10, img_format: str = "jpeg", auto_limits: bool = False):
    """
    Background task to preload frames starting from current_idx.
    Loads 'window' frames forward from current_idx. Thread-safe.
    """
    import time as time_module
    try:
        format_str = cfg.image_format[0]
        image_type = cfg.image_type
        if image_type in ("lavision_set", "lavision_im7"):
            source_path = cfg.source_paths[source_path_idx]
        else:
            folder = cfg.get_camera_folder(camera)
            source_path = cfg.source_paths[source_path_idx] / folder if folder else cfg.source_paths[source_path_idx]

        num_pairs = cfg.num_frame_pairs

        # Calculate range to preload: current_idx through current_idx + window - 1
        start_idx = max(1, current_idx)
        end_idx = min(num_pairs, current_idx + window - 1)

        preloaded = 0
        for idx in range(start_idx, end_idx + 1):

            # Use consistent cache key function
            cache_key_tuple = make_raw_cache_key(source_path_idx, camera, idx, img_format, cfg)

            # Thread-safe check if already cached
            with cache_lock:
                if cache_key_tuple in raw_image_cache:
                    continue  # Already cached

            try:
                pair = read_pair(idx, source_path, camera, cfg)
                b64_a = numpy_to_base64(pair[0], format=img_format)
                b64_b = numpy_to_base64(pair[1], format=img_format)

                stats = None
                if auto_limits:
                    stats = {
                        "A": get_percentile_stats(pair[0]),
                        "B": get_percentile_stats(pair[1])
                    }

                # Thread-safe cache write with timestamp
                access_time = time_module.time()
                with cache_lock:
                    raw_image_cache[cache_key_tuple] = (pair, b64_a, b64_b, stats, access_time)
                    # Pin frame 1 entries
                    if idx == 1:
                        FRAME_1_KEYS.add(cache_key_tuple)
                preloaded += 1
            except Exception as e:
                logger.debug(f"Failed to preload frame {idx}: {e}")
                continue

        manage_cache_size()
        if preloaded > 0:
            logger.info(f"Preloaded {preloaded} {img_format} frames for camera {camera}")
            _log_cache_contents()
    except Exception as e:
        logger.debug(f"Error in background preload: {e}")


# --- Endpoints ---


@api_bp.route("/get_frame_pair", methods=["GET"])
def get_frame_pair():
    import time as time_module
    cfg = get_config()
    camera = request.args.get("camera", type=int)
    idx = request.args.get("idx", type=int)
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    img_format = request.args.get("format", default="jpeg", type=str).lower()
    auto_limits = request.args.get("auto_limits", default="false").lower() == "true"

    # Validate format - default to jpeg for speed
    if img_format not in ("png", "jpeg"):
        img_format = "jpeg"

    # Use consistent cache key function
    cache_key_tuple = make_raw_cache_key(source_path_idx, camera, idx, img_format, cfg)

    # Determine source path for reading (if cache miss)
    format_str = cfg.image_format[0]
    image_type = cfg.image_type
    if image_type in ("lavision_set", "lavision_im7"):
        source_path = cfg.source_paths[source_path_idx]
    else:
        folder = cfg.get_camera_folder(camera)
        source_path = cfg.source_paths[source_path_idx] / folder if folder else cfg.source_paths[source_path_idx]

    # Thread-safe cache check
    with cache_lock:
        if cache_key_tuple in raw_image_cache:
            cached_data = raw_image_cache[cache_key_tuple]

            # Handle variable cache structure (5 elements = has timestamp)
            if len(cached_data) == 5:
                pair, b64_a, b64_b, stats, _ = cached_data
            elif len(cached_data) == 4:
                pair, b64_a, b64_b, stats = cached_data
            else:
                pair, b64_a, b64_b = cached_data
                stats = None

            # Calculate stats if requested but missing
            if auto_limits and stats is None:
                stats = {
                    "A": get_percentile_stats(pair[0]),
                    "B": get_percentile_stats(pair[1])
                }

            # Update cache with new access time
            access_time = time_module.time()
            raw_image_cache[cache_key_tuple] = (pair, b64_a, b64_b, stats, access_time)

            # NOTE: No preload on cache hit - frontend handles prefetching

            response = {"A": b64_a, "B": b64_b}
            if stats:
                response["stats"] = stats
            return jsonify(response)

    try:
        pair = read_pair(idx, source_path, camera, cfg)
    except FileNotFoundError as e:
        # Provide detailed error with search path and patterns
        image_format = cfg.image_format
        patterns_info = f"patterns: {image_format}" if isinstance(image_format, list) else f"pattern: {image_format}"
        return jsonify({
            "error": f"File not found in {source_path}",
            "file": str(e),
            "source_path": str(source_path),
            "patterns": image_format,
            "detail": f"Searched in {source_path} using {patterns_info}"
        }), 404
    except Exception as e:
        return jsonify({
            "error": f"Error reading image: {str(e)}",
            "source_path": str(source_path)
        }), 500

    b64_a = numpy_to_base64(pair[0], format=img_format)
    b64_b = numpy_to_base64(pair[1], format=img_format)

    stats = None
    if auto_limits:
        stats = {
            "A": get_percentile_stats(pair[0]),
            "B": get_percentile_stats(pair[1])
        }

    # Thread-safe cache write with timestamp
    access_time = time_module.time()
    with cache_lock:
        raw_image_cache[cache_key_tuple] = (pair, b64_a, b64_b, stats, access_time)
        # Pin frame 1 entries
        if idx == 1:
            FRAME_1_KEYS.add(cache_key_tuple)
    manage_cache_size()
    _log_cache_contents()

    response = {"A": b64_a, "B": b64_b}
    if stats:
        response["stats"] = stats
    return jsonify(response)


@api_bp.route("/preload_images", methods=["POST"])
def preload_images():
    """
    Preload a range of images into the cache.
    This endpoint triggers background loading and returns immediately.

    Request body:
    {
        "camera": 1,
        "start_idx": 1,
        "count": 30,
        "source_path_idx": 0,
        "format": "jpeg"
    }
    """
    data = request.get_json() or {}
    camera = data.get("camera", 1)
    start_idx = data.get("start_idx", 1)
    count = data.get("count", 30)
    source_path_idx = data.get("source_path_idx", 0)
    img_format = data.get("format", "jpeg").lower()
    auto_limits = data.get("auto_limits", False)

    # Validate format
    if img_format not in ("png", "jpeg"):
        img_format = "jpeg"

    cfg = get_config()

    # Start background preload
    # Note: _preload_surrounding_frames loads `count` frames forward and some backward
    # So we pass count directly, not count // 2
    threading.Thread(
        target=_preload_surrounding_frames,
        args=(source_path_idx, camera, start_idx, cfg, count, img_format, auto_limits),
        daemon=True
    ).start()

    return jsonify({
        "status": "preloading",
        "camera": camera,
        "start_idx": start_idx,
        "count": count,
        "source_path_idx": source_path_idx,
        "format": img_format,
        "auto_limits": auto_limits
    })


@api_bp.route("/filter", methods=["POST"])
def filter_images_endpoint():
    global processing
    data = request.get_json() or {}
    # Use fresh config instance to avoid polluting global state with preview overrides
    cfg = Config()
    camera = camera_number(data.get("camera"))
    start_idx = int(data.get("start_idx", 1))
    filters = data.get("filters", None)
    masking = data.get("masking", None)
    
    # Handle source_path_idx safely (default to 0 if missing or None)
    source_path_idx = data.get("source_path_idx")
    if source_path_idx is None:
        source_path_idx = 0
    source_path_idx = int(source_path_idx)
    
    if filters is not None:
        cfg.data["filters"] = filters

    if masking is not None:
        if "masking" not in cfg.data:
            cfg.data["masking"] = {}
        recursive_update(cfg.data["masking"], masking)
        logger.info(f"Preview masking config: {cfg.data['masking']}")

    # Use batch size from config
    batch_length = cfg.data.get("batches", {}).get("size", 30)
    batch_len_reason = "config.batches.size"
    
    if batch_length < 1:
        batch_length = 1
    
    batch_start, batch_end = compute_batch_window(
        start_idx, batch_length, cfg.num_frame_pairs
    )
    indices = list(range(batch_start, batch_end + 1))
    
    # For .set and .im7 files, don't append camera folder - all cameras are in the source directory
    format_str = cfg.image_format[0]
    image_type = cfg.image_type

    if image_type in ("lavision_set", "lavision_im7"):
        source_path = cfg.source_paths[source_path_idx]
    else:
        folder = cfg.get_camera_folder(camera_number(camera))
        source_path = cfg.source_paths[source_path_idx] / folder if folder else cfg.source_paths[source_path_idx]

    def load_pairs_parallel():
        """Load pairs in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        
        # Use thread pool for I/O-bound image reading
        max_workers = min(os.cpu_count(), len(indices), 8)
        pairs = [None] * len(indices)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(read_pair, idx, source_path, camera, cfg): i
                for i, idx in enumerate(indices)
            }
            
            for future in as_completed(future_to_idx):
                pos = future_to_idx[future]
                pairs[pos] = future.result()
        
        arr = np.stack(pairs, axis=0)
        return da.from_array(arr, chunks=(arr.shape[0], 2, *cfg.image_shape))

    def process_and_store():
        global processing
        try:
            # Load with parallel I/O
            darr = load_pairs_parallel()

            # Compute to numpy array first (required for apply_filters_to_batch)
            batch_np = dask.compute(darr, scheduler='threads')[0]

            # Load mask
            mask = load_mask_for_camera(camera, cfg, source_path_idx)
            if mask is not None:
                logger.info(f"Preview mask loaded: shape={mask.shape}, dtype={mask.dtype}")
                if batch_np.shape[-2:] != mask.shape:
                    logger.error(f"Mask shape mismatch! Batch: {batch_np.shape}, Mask: {mask.shape}")
            else:
                if not cfg.masking_enabled:
                    logger.info("Masking is DISABLED in config.")
                else:
                    expected_path = cfg.get_mask_path(camera, source_path_idx)
                    logger.info(f"Masking ENABLED. Expected mask path: {expected_path}. Exists: {expected_path.exists()}")
                logger.info("No mask loaded for preview (masking disabled or file not found)")

            # Apply ALL filters (spatial + batch) using unified function
            processed_all = apply_filters_to_batch(
                batch_np,
                cfg,
                save_diagnostics=False,
                output_dir=None,
                batch_idx=0,
                pixel_mask=mask
            )

            # Store results
            k = cache_key(source_path_idx, camera, cfg)
            processed_store["processed"].setdefault(k, {})

            # Batch update dictionary (faster than individual updates)
            processed_store["processed"][k].update({
                abs_idx: processed_all[rel]
                for rel, abs_idx in enumerate(indices)
            })
            _log_cache_contents()

        except Exception as e:
            logger.exception(f"Error during /filter processing: {e}")
        finally:
            processing = False

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


@api_bp.route("/processing_status", methods=["GET", "POST"])
def processing_status():
    """
    Check or modify processing status.

    GET: Returns current processing state.
    POST: Can cancel processing by setting {"cancel": true}.
    """
    global processing
    if request.method == "POST":
        data = request.get_json() or {}
        if data.get("cancel"):
            processing = False
            logger.info("Processing cancelled by user request")
            return jsonify({"status": "cancelled"})
    return jsonify({"processing": processing})


@api_bp.route("/get_processed_pair", methods=["GET"])
def get_processed_pair():
    cfg = get_config()
    frame = request.args.get("frame", type=int)
    typ = request.args.get("type", "processed")
    camera = camera_number(request.args.get("camera"))
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    auto_limits = request.args.get("auto_limits", default="false").lower() == "true"

    b64_a, b64_b, stats = get_cached_pair(frame, typ, camera, source_path_idx, cfg, auto_limits)

    response = {"status": "ok", "A": b64_a, "B": b64_b}
    if stats:
        response["stats"] = stats
    return jsonify(response)


@api_bp.route("/filter_single_frame", methods=["POST"])
def filter_single_frame():
    """
    Process a single frame with spatial filters only (no batching required).
    Returns processed images immediately without caching.
    """
    data = request.get_json() or {}
    cfg = get_config()
    camera = camera_number(data.get("camera"))
    frame_idx = int(data.get("frame_idx", 1))
    filters = data.get("filters", [])
    source_path_idx = data.get("source_path_idx", 0)
    auto_limits = data.get("auto_limits", False)
    
    # Check if any batch filters are present (should use /filter endpoint instead)
    batch_filters = [f for f in filters if f.get("type") in ("time", "pod")]
    if batch_filters:
        return jsonify({
            "error": "Batch filters (time, pod) not supported in single-frame mode. Use /filter endpoint."
        }), 400
    
    # For .set and .im7 files, don't append camera folder
    format_str = cfg.image_format[0]
    image_type = cfg.image_type

    if image_type in ("lavision_set", "lavision_im7"):
        source_path = cfg.source_paths[source_path_idx]
    else:
        folder = cfg.get_camera_folder(camera_number(camera))
        source_path = cfg.source_paths[source_path_idx] / folder if folder else cfg.source_paths[source_path_idx]
    
    try:
        # Read the single pair
        pair = read_pair(frame_idx, source_path, camera, cfg)
        
        # Convert to dask array with single frame
        arr = np.stack([pair], axis=0)  # Shape: (1, 2, H, W)
        images_da = da.from_array(arr, chunks=(1, 2, *cfg.image_shape))
        
        # Apply spatial filters
        if filters:
            # Temporarily set filters in config
            old_filters = cfg.data.get("filters", [])
            cfg.data["filters"] = filters
            
            try:
                filtered = preprocess_images(images_da, cfg)
                result = filtered.compute()
            finally:
                # Restore original filters
                cfg.data["filters"] = old_filters
        else:
            result = arr
        
        # Extract the single processed pair
        processed_pair = result[0]  # Shape: (2, H, W)
        
        response = {
            "status": "ok",
            "A": numpy_to_png_base64(processed_pair[0]),
            "B": numpy_to_png_base64(processed_pair[1])
        }
        
        if auto_limits:
            response["stats"] = {
                "A": get_percentile_stats(processed_pair[0]),
                "B": get_percentile_stats(processed_pair[1])
            }
            
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing single frame: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/download_image", methods=["POST"])
def download_image():
    """
    Download raw or processed image as PNG with proper headers.
    """
    data = request.get_json() or {}
    image_type = data.get("type", "raw")  # "raw" or "processed"
    frame = data.get("frame", "A")  # "A" or "B"
    base64_data = data.get("data")  # Base64 PNG data
    frame_idx = data.get("frame_idx", 1)
    camera = data.get("camera", 1)
    
    if not base64_data:
        return jsonify({"error": "No image data provided"}), 400
    
    try:
        import base64
        from io import BytesIO
        from flask import send_file
        
        # Decode base64 to binary
        image_bytes = base64.b64decode(base64_data)
        
        # Create filename
        filename = f"Cam{camera}_frame{frame_idx:05d}_{frame}_{image_type}.png"
        
        # Send as downloadable file
        return send_file(
            BytesIO(image_bytes),
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/status", methods=["GET"])
def get_status():
    return jsonify({"processing": processing})


@api_bp.route("/validate_files", methods=["POST"])
def validate_files():
    """
    Smart validation: Check first frame, last frame, and count files.

    Uses the generic validate_images_generic() for core file detection,
    with PIV-specific additions for pair validation, color detection,
    indexing warnings, and background preloading.

    Returns per-camera validation with:
    - first_frame: "exists" or "missing"
    - last_frame: "exists" or "missing"
    - expected_count: number of expected files
    - actual_count: number of matching files found
    - status: "ok", "warning", or "error"
    - color_detected: True if images are color (will be converted)
    """
    import re as re_module

    data = request.get_json() or {}
    source_path_idx = data.get("source_path_idx", 0)

    cfg = get_config()
    camera_numbers = cfg.camera_numbers
    results = {}
    overall_valid = True

    for camera_num in camera_numbers:
        try:
            # Build camera path using shared utility
            camera_path = build_piv_camera_path(cfg, source_path_idx, camera_num)
            format_str = cfg.image_format[0]
            image_type = cfg.image_type
            num_images = cfg.num_images
            num_pairs = cfg.num_frame_pairs

            # Create a frame reader function for preview generation
            def read_frame(idx: int):
                pair = read_pair(idx, camera_path, camera_num, cfg)
                return pair[0]  # Return first frame of pair for preview

            # Use generic validator for core file detection
            validation = validate_images_generic(
                camera_path=camera_path,
                camera=camera_num,
                image_format=format_str,
                image_type=image_type,
                expected_count=num_images,
                zero_based_indexing=cfg.zero_based_indexing,
                read_frame_fn=read_frame,
            )

            # PIV-specific: Test first and last pairs (more thorough than generic)
            first_frame_status = "missing"
            last_frame_status = "missing"
            color_detected = False

            try:
                first_pair = read_pair(1, camera_path, camera_num, cfg)
                first_frame_status = "exists"
                # Check if color (ndim > 2 means we got a color image before conversion)
                if first_pair.ndim > 2 and first_pair.shape[-1] > 1:
                    color_detected = True
            except Exception as e:
                logger.debug(f"First frame check failed for camera {camera_num}: {e}")

            try:
                read_pair(num_pairs, camera_path, camera_num, cfg)
                last_frame_status = "exists"
            except Exception as e:
                logger.debug(f"Last frame check failed for camera {camera_num}: {e}")

            # PIV-specific: Check for indexing mismatch (only for standard formats)
            indexing_warning = None
            if image_type == "standard" and validation["sample_files"]:
                try:
                    indices = []
                    for fname in validation["sample_files"]:
                        match = re_module.search(r'(\d+)', fname)
                        if match:
                            indices.append(int(match.group(1)))
                    if indices:
                        min_idx = min(indices)
                        expected_min = 0 if cfg.zero_based_indexing else 1
                        if min_idx != expected_min:
                            indexing_warning = (
                                f"File indexing mismatch: found files starting at {min_idx}, "
                                f"but zero_based_indexing is "
                                f"{'enabled' if cfg.zero_based_indexing else 'disabled'} "
                                f"(expects {expected_min})"
                            )
                except Exception as e:
                    logger.debug(f"Indexing check failed: {e}")

            # Determine status based on both generic validation and pair checks
            actual_count = validation["found_count"]
            if actual_count == "container":
                actual_count = num_images  # For containers, assume expected count

            error_msg = validation.get("error")
            status = "ok" if validation["valid"] else "error"

            # Override with PIV-specific pair validation
            if first_frame_status == "missing":
                status = "error"
                start_idx = 0 if cfg.zero_based_indexing else 1
                if image_type == "lavision_set":
                    error_msg = f"First frame not found. Container file: {format_str}"
                elif image_type == "cine":
                    error_msg = f"First frame not found. CINE file: {format_str % camera_num}"
                else:
                    error_msg = f"First frame not found. Looking for: {format_str % start_idx}"
                # Append folder contents hint from generic validator
                if validation.get("sample_files"):
                    error_msg += f". Found files: {', '.join(validation['sample_files'][:5])}"
                if validation.get("suggested_pattern"):
                    error_msg += f". Try pattern: {validation['suggested_pattern']}"

            elif last_frame_status == "missing":
                status = "error"
                end_idx = (0 if cfg.zero_based_indexing else 1) + num_images - 1
                if image_type == "lavision_set":
                    error_msg = f"Last frame not found. Container file: {format_str}"
                elif image_type == "cine":
                    error_msg = f"Last frame not found. CINE file: {format_str % camera_num}"
                else:
                    error_msg = f"Last frame not found. Expected: {format_str % end_idx}"

            elif isinstance(actual_count, int) and actual_count > num_images:
                # More files than expected - user processing subset (this is fine!)
                status = "ok"
                error_msg = f"Processing subset: {num_images} of {actual_count} files available"

            if status == "error":
                overall_valid = False

            results[f"camera_{camera_num}"] = {
                "first_frame": first_frame_status,
                "last_frame": last_frame_status,
                "expected_count": num_images,
                "actual_count": actual_count,
                "status": status,
                "camera_path": str(camera_path),
                "color_detected": color_detected,
                "indexing_warning": indexing_warning,
                "error": error_msg,
                "first_image_preview": validation.get("first_image_preview"),
                "image_size": validation.get("image_size"),
                "suggested_pattern": validation.get("suggested_pattern"),
                "suggested_pattern_b": validation.get("suggested_pattern_b"),
                "suggested_mode": validation.get("suggested_mode"),
            }

        except Exception as e:
            logger.error(f"Validation error for camera {camera_num}: {e}")
            results[f"camera_{camera_num}"] = {
                "status": "error",
                "error": str(e)
            }
            overall_valid = False

    # If validation passed, preload first 10 frames for each camera in background
    if overall_valid:
        for camera_num in camera_numbers:
            threading.Thread(
                target=_preload_surrounding_frames,
                args=(source_path_idx, camera_num, 1, cfg, 10, "jpeg", True),
                daemon=True
            ).start()
        logger.info(f"Validation passed - preloading first 10 frames for {len(camera_numbers)} camera(s)")

    return jsonify({
        "valid": overall_valid,
        "details": results
    })


@api_bp.route("/config", methods=["GET"])
def config_endpoint():
    cfg = get_config()
    # Return full nested config as JSON, including computed properties
    config_data = cfg.data.copy()
    config_data["images"]["num_frame_pairs"] = cfg.num_frame_pairs
    return jsonify(config_data)


@api_bp.route("/update_config", methods=["POST"])
def update_config():
    global raw_image_cache, processed_store, FRAME_1_KEYS
    data = request.get_json() or {}
    cfg = get_config()

    # Check if paths or image format are changing (these affect cache validity)
    paths_changing = "paths" in data and (
        "source_paths" in data.get("paths", {}) or
        "base_paths" in data.get("paths", {})
    )
    format_changing = "images" in data and "image_format" in data.get("images", {})

    # Check if filters are changing (affects processed cache only)
    filters_changing = "filters" in data

    # Thread-safe cache clearing
    if paths_changing or format_changing:
        logger.info("Clearing all image caches due to config change (paths or format)")
        with cache_lock:
            raw_image_cache.clear()
            FRAME_1_KEYS.clear()
        processed_store["processed"] = {}
    elif filters_changing:
        logger.info("Clearing processed image cache due to filter change")
        processed_store["processed"] = {}

    # No special handling needed for filters - save them as-is including batch_size

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

    # Store old camera_count to detect changes
    old_camera_count = cfg.data["paths"].get("camera_count", 1)

    # Check if camera_numbers was explicitly provided in the update
    camera_numbers_provided = "camera_numbers" in data.get("paths", {})

    recursive_update(cfg.data, data)

    # Normalize camera keys in calibration.polynomial.cameras to integers
    # (JSON keys are always strings, but we want integer keys in YAML)
    poly_cameras = cfg.data.get("calibration", {}).get("polynomial", {}).get("cameras")
    if poly_cameras and isinstance(poly_cameras, dict):
        normalized = {}
        for k, v in poly_cameras.items():
            try:
                int_key = int(k)
                # If both string and int versions exist, prefer the one being updated
                # (string key is the one just sent from frontend)
                if int_key in normalized and isinstance(k, str):
                    # String key is new data, overwrite
                    normalized[int_key] = v
                elif int_key not in normalized:
                    normalized[int_key] = v
            except (ValueError, TypeError):
                # Keep non-numeric keys as-is (shouldn't happen)
                normalized[k] = v
        cfg.data["calibration"]["polynomial"]["cameras"] = normalized

    # Normalize camera keys in transforms.cameras to integers
    transforms_cameras = cfg.data.get("transforms", {}).get("cameras")
    if transforms_cameras and isinstance(transforms_cameras, dict):
        normalized = {}
        for k, v in transforms_cameras.items():
            try:
                int_key = int(k)
                if int_key in normalized and isinstance(k, str):
                    normalized[int_key] = v
                elif int_key not in normalized:
                    normalized[int_key] = v
            except (ValueError, TypeError):
                normalized[k] = v
        cfg.data["transforms"]["cameras"] = normalized

    # Handle camera_numbers based on camera_count changes
    new_camera_count = cfg.data["paths"].get("camera_count", 1)

    if new_camera_count != old_camera_count:
        # Camera count changed
        if not camera_numbers_provided:
            # Only reset to default if user didn't explicitly provide camera_numbers
            cfg.data["paths"]["camera_numbers"] = list(range(1, new_camera_count + 1))
        else:
            # User provided camera_numbers with new camera_count, just validate them
            camera_numbers = cfg.data["paths"].get("camera_numbers", [])
            valid_numbers = [n for n in camera_numbers if 1 <= n <= new_camera_count]
            if not valid_numbers:
                # If none are valid, use full range
                valid_numbers = list(range(1, new_camera_count + 1))
            cfg.data["paths"]["camera_numbers"] = valid_numbers
    else:
        # Camera count unchanged - validate existing camera_numbers
        camera_numbers = cfg.data["paths"].get("camera_numbers", [])
        valid_numbers = [n for n in camera_numbers if 1 <= n <= new_camera_count]
        if not valid_numbers:
            valid_numbers = list(range(1, new_camera_count + 1))
        cfg.data["paths"]["camera_numbers"] = valid_numbers
    
    with open(cfg.config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg.data, f, default_flow_style=False, sort_keys=False)
    reload_config()
    return jsonify({"status": "success", "updated": data})


@api_bp.route("/run_piv", methods=["POST"])
def run_piv():
    """
    Start a PIV computation job as a subprocess.

    This spawns the PIV computation outside of Flask for full computational
    performance while keeping the server responsive.

    Request body (optional):
    {
        "cameras": [1, 2, 3],    // List of camera numbers to process (optional)
        "source_path_idx": 0,    // Index of source path (legacy, use active_paths)
        "base_path_idx": 0,      // Index of base path (legacy, use active_paths)
        "active_paths": [0, 1],  // List of path indices to process (optional)
        "mode": "instantaneous"  // PIV mode: "instantaneous" or "ensemble"
    }
    """
    data = request.get_json() or {}

    # Extract parameters
    cameras = data.get("cameras")
    source_path_idx = data.get("source_path_idx", 0)
    base_path_idx = data.get("base_path_idx", 0)
    active_paths = data.get("active_paths")  # List of path indices to process
    mode = data.get("mode", "instantaneous")  # PIV mode: instantaneous or ensemble

    # Get the runner and start the job
    runner = get_runner()
    result = runner.start_piv_job(
        cameras=cameras,
        source_path_idx=source_path_idx,
        base_path_idx=base_path_idx,
        active_paths=active_paths,
        mode=mode,
    )

    return jsonify(result), 200 if result.get("status") == "started" else 500


@api_bp.route("/piv_status", methods=["GET"])
def piv_status():
    """
    Get status of PIV job(s).
    
    Query parameters:
    - job_id: Specific job ID (optional, if omitted returns all jobs)
    """
    runner = get_runner()
    job_id = request.args.get("job_id")
    
    if job_id:
        status = runner.get_job_status(job_id)
        if status:
            return jsonify(status)
        return jsonify({"error": "Job not found"}), 404
    else:
        # Return all jobs
        jobs = runner.list_jobs()
        return jsonify({"jobs": jobs})


@api_bp.route("/cancel_run", methods=["POST"])
def cancel_piv():
    """
    Cancel a running PIV job.
    
    Request body:
    {
        "job_id": "piv_20231005_143022"
    }
    """
    data = request.get_json() or {}
    job_id = data.get("job_id")
    
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    runner = get_runner()
    success = runner.cancel_job(job_id)
    
    if success:
        return jsonify({"status": "cancelled", "job_id": job_id})
    return jsonify({"error": "Failed to cancel job or job not found"}), 404


@api_bp.route("/piv_logs", methods=["GET"])
def get_piv_logs():
    """
    Get log content for a PIV job.
    
    Query parameters:
    - job_id: Specific job ID (optional)
    - lines: Number of lines to return from end (optional, default all)
    - offset: Line offset from end (optional, for pagination)
    """
    runner = get_runner()
    job_id = request.args.get("job_id")
    lines = request.args.get("lines", type=int)
    offset = request.args.get("offset", default=0, type=int)
    
    if not job_id:
        # If no job_id, try to get the most recent job
        jobs = runner.list_jobs()
        if not jobs:
            return jsonify({"error": "No PIV jobs found"}), 404
        # Sort by start time and get most recent
        jobs.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        job_id = jobs[0].get("job_id")
    
    status = runner.get_job_status(job_id)
    if not status:
        return jsonify({"error": "Job not found"}), 404
    
    log_file = Path(status["log_file"])
    if not log_file.exists():
        return jsonify({"logs": "", "job_id": job_id, "running": status["running"]})
    
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        
        # Apply offset and line limit
        if lines:
            start_idx = max(0, len(all_lines) - lines - offset)
            end_idx = len(all_lines) - offset
            log_lines = all_lines[start_idx:end_idx]
        else:
            log_lines = all_lines
        
        log_content = "".join(log_lines)
        
        return jsonify({
            "logs": log_content,
            "job_id": job_id,
            "running": status["running"],
            "total_lines": len(all_lines),
            "returned_lines": len(log_lines),
        })
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return jsonify({"error": f"Failed to read log file: {str(e)}"}), 500


def get_ensemble_progress_from_logs(job_id: str, cfg) -> dict:
    """Parse logs to determine ensemble progress."""
    import re

    runner = get_runner()
    status = runner.get_job_status(job_id)

    if not status:
        return {"percent": 0, "error": "Job not found", "running": False}

    # Read the full log file for parsing
    log_file = Path(status.get("log_file", ""))
    log_text = ""
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                log_text = f.read()
        except Exception:
            pass

    # Parse pass progress: "======== PASS 2/4 ========"
    pass_matches = re.findall(r"PASS (\d+)/(\d+)", log_text)
    current_pass = 1
    total_passes = cfg.ensemble_num_passes or 1

    if pass_matches:
        # Get the last match (most recent pass)
        current_pass = int(pass_matches[-1][0])
        total_passes = int(pass_matches[-1][1])

    # Parse batch progress within pass: "[Pass 2, Batch 5/10]" or similar patterns
    batch_matches = re.findall(r"\[Pass \d+,?\s*Batch (\d+)/(\d+)\]", log_text)
    current_batch = 0
    total_batches = 1

    if batch_matches:
        # Get the last batch match
        current_batch = int(batch_matches[-1][0])
        total_batches = int(batch_matches[-1][1])

    # Calculate overall progress
    # Progress = (completed_passes * 100 + current_pass_progress) / total_passes
    pass_progress = (current_batch / total_batches) * 100 if total_batches > 0 else 0
    overall_progress = ((current_pass - 1) * 100 + pass_progress) / total_passes

    # Check if complete
    is_running = status.get("running", True)
    return_code = status.get("return_code")
    if not is_running and return_code == 0:
        overall_progress = 100

    return {
        "percent": int(overall_progress),
        "mode": "ensemble",
        "current_pass": current_pass,
        "total_passes": total_passes,
        "current_batch": current_batch,
        "total_batches": total_batches,
        "running": is_running,
    }


@api_bp.route("/get_uncalibrated_count", methods=["GET"])
def get_uncalibrated_count():
    cfg = get_config()
    basepath_idx = request.args.get("basepath_idx", default=0, type=int)
    cam = camera_number(request.args.get("camera", default=1, type=int))
    type_name = request.args.get("type", default="instantaneous")
    job_id = request.args.get("job_id")  # NEW: accept job_id for log-based progress

    # Check if ensemble mode - use log-based progress if job_id provided
    is_ensemble = cfg.data.get("processing", {}).get("ensemble", False)
    if is_ensemble and job_id:
        progress_data = get_ensemble_progress_from_logs(job_id, cfg)
        return jsonify(progress_data)

    base_paths = cfg.base_paths
    base = base_paths[basepath_idx]
    num_pairs = cfg.num_frame_pairs  # Vector files correspond to frame pairs

    # Get all cameras that should be processed
    camera_numbers = cfg.camera_numbers
    total_cameras = len(camera_numbers)

    # Calculate progress across all cameras
    total_expected_files = num_pairs * total_cameras
    total_found_files = 0
    camera_progress = {}

    vector_fmt = cfg.vector_format
    expected_names = set([vector_fmt % i for i in range(1, num_pairs + 1)])

    # Count files for each camera and collect all available files
    all_files = []
    for camera_num in camera_numbers:
        paths = get_data_paths(
            base,
            cfg.num_frame_pairs,
            camera_num,
            type_name,
            use_uncalibrated=True,
        )
        folder_uncal = paths["data_dir"]

        found = (
            [
                p.name
                for p in sorted(folder_uncal.iterdir())
                if p.is_file() and p.name in expected_names
            ]
            if folder_uncal.exists() and folder_uncal.is_dir()
            else []
        )

        # If this is the requested camera, add its files to the list
        if camera_num == cam:
            all_files = found

        camera_progress[f"Cam{camera_num}"] = {
            "count": len(found),
            "percent": int((len(found) / num_pairs) * 100) if num_pairs else 0
        }
        total_found_files += len(found)

    # Calculate overall progress across all cameras
    percent = int((total_found_files / total_expected_files) * 100) if total_expected_files else 0

    return jsonify({
        "count": total_found_files,
        "percent": percent,
        "total_expected": total_expected_files,
        "camera_progress": camera_progress,
        "cameras": camera_numbers,
        "files": all_files,
    })


@api_bp.route("/check_output_exists", methods=["GET"])
def check_output_exists():
    """Check if output data exists for given paths/cameras."""
    cfg = get_config()
    active_paths_str = request.args.get("active_paths", "")
    active_paths = [int(p) for p in active_paths_str.split(",") if p.strip()]

    logger.info(f"[check_output_exists] Checking paths: {active_paths}")

    if not active_paths:
        logger.info("[check_output_exists] No active paths provided")
        return jsonify({"exists": False, "details": {}})

    exists = False
    details = {}

    for path_idx in active_paths:
        if path_idx >= len(cfg.base_paths):
            logger.warning(f"[check_output_exists] Path index {path_idx} out of range")
            continue
        base = cfg.base_paths[path_idx]
        path_key = f"path_{path_idx}"
        details[path_key] = {}
        logger.info(f"[check_output_exists] Checking base path: {base}")

        for cam_num in cfg.camera_numbers:
            cam_key = f"Cam{cam_num}"
            details[path_key][cam_key] = {"instantaneous": False, "ensemble": False}

            # Check instantaneous output
            inst_paths = get_data_paths(base, cfg.num_frame_pairs, cam_num, "instantaneous", True)
            inst_dir = inst_paths["data_dir"]
            logger.info(f"[check_output_exists] Instantaneous dir: {inst_dir}, exists: {inst_dir.exists()}")
            if inst_dir.exists():
                try:
                    files = list(inst_dir.iterdir())
                    logger.info(f"[check_output_exists] Found {len(files)} files in instantaneous dir")
                    if files:
                        exists = True
                        details[path_key][cam_key]["instantaneous"] = True
                except Exception as e:
                    logger.error(f"[check_output_exists] Error listing instantaneous dir: {e}")

            # Check ensemble output
            ens_paths = get_data_paths(base, cfg.num_frame_pairs, cam_num, "ensemble", True)
            ens_dir = ens_paths["data_dir"]
            logger.info(f"[check_output_exists] Ensemble dir: {ens_dir}, exists: {ens_dir.exists()}")
            if ens_dir.exists():
                try:
                    files = list(ens_dir.iterdir())
                    logger.info(f"[check_output_exists] Found {len(files)} files in ensemble dir")
                    if files:
                        exists = True
                        details[path_key][cam_key]["ensemble"] = True
                except Exception as e:
                    logger.error(f"[check_output_exists] Error listing ensemble dir: {e}")

    logger.info(f"[check_output_exists] Final result: exists={exists}")
    return jsonify({"exists": exists, "details": details})


@api_bp.route("/clear_output", methods=["POST"])
def clear_output():
    """Clear output data for given paths/cameras."""
    import shutil

    data = request.get_json() or {}
    active_paths = data.get("active_paths", [])
    camera_numbers = data.get("camera_numbers", [])

    cfg = get_config()
    cleared = []
    errors = []

    for path_idx in active_paths:
        if path_idx >= len(cfg.base_paths):
            continue
        base = cfg.base_paths[path_idx]

        for cam_num in camera_numbers:
            for type_name in ["instantaneous", "ensemble"]:
                try:
                    paths = get_data_paths(base, cfg.num_frame_pairs, cam_num, type_name, True)
                    data_dir = paths["data_dir"]
                    if data_dir.exists():
                        shutil.rmtree(data_dir)
                        data_dir.mkdir(parents=True, exist_ok=True)
                        cleared.append(str(data_dir))
                        logger.info(f"Cleared output directory: {data_dir}")
                except Exception as e:
                    errors.append(f"Failed to clear {type_name} for path {path_idx}, cam {cam_num}: {str(e)}")
                    logger.error(f"Error clearing output: {e}")

    return jsonify({
        "status": "cleared" if not errors else "partial",
        "directories": cleared,
        "errors": errors
    })

# Register the main API blueprint
app.register_blueprint(api_bp)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        # This serves static files like .js, .css, images
        return send_from_directory(app.static_folder, path)
    else:
        # This serves 'index.html' for any page request
        # that isn't an API route or a static file.
        return send_from_directory(app.static_folder, 'index.html')


def ensure_config_exists():
    """Ensure config.yaml exists in the current directory, create if not."""
    import shutil
    cwd = Path.cwd()
    config_path = cwd / "config.yaml"

    if config_path.exists():
        return  # Config already exists

    # Try to copy from pivtools_core package
    try:
        import pivtools_core
        default_config = Path(pivtools_core.__file__).parent / "config.yaml"

        if default_config.exists():
            shutil.copy2(default_config, config_path)
            print(f"Created config.yaml at {config_path}")
        else:
            print(f"Warning: Default config not found at {default_config}")
            print("Run 'pivtools-cli init' to create a config file")
    except ImportError:
        print("Warning: pivtools_core not found. Run 'pivtools-cli init' to create a config file")


def main():
    """Run the PIVTOOLs GUI"""
    # Ensure config.yaml exists before starting
    ensure_config_exists()

    # Suppress Flask development server warning by setting production environment
    import os
    os.environ['FLASK_ENV'] = 'production'

    print("Starting PIVTOOLs GUI...")
    print("Open your browser to http://localhost:5000")
    
    # Automatically open browser after a short delay
    def open_browser():
        webbrowser.open('http://localhost:5000')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()