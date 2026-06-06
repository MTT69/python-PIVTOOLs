import copy
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
from flask_compress import Compress
from loguru import logger
import os
from pivtools_core.config import get_config, reload_config, Config
from pivtools_core.image_handling.load_images import create_piv_frame_reader, read_pair, load_mask_for_camera
from pivtools_core.image_handling.path_utils import build_piv_camera_path, validate_images_generic
from pivtools_gui.masking.app.views import masking_bp
from pivtools_core.paths import get_data_paths
from pivtools_gui.piv_runner import get_runner
from pivtools_gui.plotting.app.plotting_views import vector_plot_bp
# Transform endpoints (per-file storage with backup/restore, used by frontend)
from pivtools_gui.plotting.app.transform_views import transform_bp
# Note: transforms/app/transform_views.py exists for CLI batch usage via TransformProcessor
# but is NOT registered as a Flask blueprint - the frontend uses the plotting transform API
from pivtools_cli.preprocessing.preprocess import apply_filters_to_batch
# from pivtools_gui.stereo_reconstruction.app.views import stereo_bp
from pivtools_gui.utils import camera_number, numpy_to_png_base64, numpy_to_base64, get_display_contrast_stats
from pivtools_gui.vector_statistics.app.views import statistics_bp
from pivtools_gui.vector_merging.app.views import merging_bp
from pivtools_gui.video_maker.app.views import video_maker_bp

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)
Compress(app)
app.config['COMPRESS_MIN_SIZE'] = 500
dask_config.set(scheduler="threads")

# Configure loguru logging level from config
def configure_logging():
    """Configure loguru to use the log level from config.yaml."""
    try:
        cfg = get_config()
        log_level = cfg.log_level  # e.g., "INFO", "DEBUG", "WARNING"
        # Remove default handler and add one with the configured level
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),  # Console output
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )
        logger.debug(f"Loguru configured with level: {log_level}")
    except Exception as e:
        # If config isn't available yet, keep default loguru config
        logger.warning(f"Could not configure logging from config: {e}")

configure_logging()

# Create API blueprint with /backend prefix
api_bp = Blueprint('api', __name__, url_prefix='/backend')

# Register existing blueprints with /backend prefix
app.register_blueprint(vector_plot_bp, url_prefix='/backend/plot')
app.register_blueprint(transform_bp, url_prefix='/backend/plot')
app.register_blueprint(masking_bp, url_prefix='/backend')
# Legacy calibration v1 blueprint deactivated 2026-06-05 — superseded by calibration2.
# The pivtools_gui.calibration package remains on disk (still imports job_manager for
# video_maker and global_coordinate_alignment for PIV runs); only its GUI routes are off.
from pivtools_gui.calibration2.app import calibration2_bp, calibration2_stepped_bp
app.register_blueprint(calibration2_bp, url_prefix='/backend')
app.register_blueprint(calibration2_stepped_bp, url_prefix='/backend')
app.register_blueprint(video_maker_bp, url_prefix='/backend/video')
# app.register_blueprint(stereo_bp, url_prefix='/backend')
app.register_blueprint(statistics_bp, url_prefix='/backend')
app.register_blueprint(merging_bp, url_prefix='/backend')
# --- In-memory stores ---
processed_store = {"processed": {}}
processing = False

# Raw image cache: {cache_key: (base64_A, base64_B, stats, last_access_time)}
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

        # Sort by access time (4th element), excluding pinned frame 1 entries
        evictable = []
        for k, v in raw_image_cache.items():
            if k in FRAME_1_KEYS:
                continue  # Never evict frame 1
            # v = (b64_a, b64_b, stats, last_access_time)
            access_time = v[3] if len(v) > 3 else 0
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
            logger.debug(f"Cache: {raw_str} | {proc_str}")
        else:
            logger.debug(f"Cache: {raw_str}")


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


def _resolve_loop_read(cfg, source_path_idx, pair_idx, camera):
    """Resolve multi-loop pair to (source_path, local_pair_idx) for read_pair.

    For single-loop, returns the normal source_path and original pair_idx.
    For multi-loop, resolves the global pair to the correct loop source
    and local pair index within that loop.

    Always operates on the raw cfg.source_paths entry (before camera folder),
    so get_loop_source_path modifies the correct path component.
    """
    base_source = cfg.source_paths[source_path_idx]
    image_type = cfg.image_type

    if cfg.num_loops > 1:
        loop_idx, local_pair = cfg.resolve_loop_for_pair(pair_idx)
        effective_base = cfg.get_loop_source_path(base_source, loop_idx)
    else:
        effective_base = base_source
        local_pair = pair_idx

    # Derive camera-specific path
    if image_type in ("lavision_set", "lavision_im7", "cine"):
        return effective_base, local_pair
    folder = cfg.get_camera_folder(camera)
    source = effective_base / folder if folder else effective_base
    return source, local_pair



def get_cached_pair(frame, typ, camera, source_path_idx, cfg, auto_limits=False):
    """Fetch a cached pair (A, B) for given frame/type/camera/source_path_idx."""
    k = cache_key(source_path_idx, camera, cfg)
    bucket = processed_store.get(typ, {}).get(k, {})
    entry = bucket.get(frame)
    if entry is None:
        return None, None, None

    # entry is (b64_a, b64_b, stats) — encoded + stats computed at storage time
    if len(entry) == 3:
        b64_a, b64_b, stats = entry
    else:
        # Legacy 2-tuple fallback
        b64_a, b64_b = entry
        stats = None

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
                # Multi-loop: resolve global pair index to loop-specific source
                resolved_source, local_idx = _resolve_loop_read(cfg, source_path_idx, idx, camera)
                pair = read_pair(local_idx, resolved_source, camera, cfg)

                # numpy_to_base64 handles sqrt + percentile clipping internally
                b64_a = numpy_to_base64(pair[0], format=img_format)
                b64_b = numpy_to_base64(pair[1], format=img_format)
                stats = {
                    "A": get_display_contrast_stats(pair[0]),
                    "B": get_display_contrast_stats(pair[1]),
                }

                # Thread-safe cache write with timestamp
                access_time = time_module.time()
                with cache_lock:
                    raw_image_cache[cache_key_tuple] = (b64_a, b64_b, stats, access_time)
                    # Pin frame 1 entries
                    if idx == 1:
                        FRAME_1_KEYS.add(cache_key_tuple)
                preloaded += 1
            except Exception as e:
                logger.debug(f"Failed to preload frame {idx}: {e}")
                continue

        manage_cache_size()
        if preloaded > 0:
            logger.debug(f"Preloaded {preloaded} {img_format} frames for camera {camera}")
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
    if not cfg.image_format:
        return jsonify({"error": "No image format configured"}), 400
    format_str = cfg.image_format[0]
    image_type = cfg.image_type
    if image_type in ("lavision_set", "lavision_im7"):
        source_path = cfg.source_paths[source_path_idx]
    else:
        folder = cfg.get_camera_folder(camera)
        source_path = cfg.source_paths[source_path_idx] / folder if folder else cfg.source_paths[source_path_idx]

    # Thread-safe cache check
    hit_response = None
    with cache_lock:
        if cache_key_tuple in raw_image_cache:
            cached_data = raw_image_cache[cache_key_tuple]
            b64_a, b64_b, stats, _ = cached_data

            # Update cache with new access time
            access_time = time_module.time()
            raw_image_cache[cache_key_tuple] = (b64_a, b64_b, stats, access_time)

            hit_response = {"A": b64_a, "B": b64_b}
            if stats:
                hit_response["stats"] = stats

    if hit_response is not None:
        return jsonify(hit_response)

    try:
        # Multi-loop: resolve global pair index to loop-specific source and local index
        resolved_source, local_idx = _resolve_loop_read(cfg, source_path_idx, idx, camera)
        pair = read_pair(local_idx, resolved_source, camera, cfg)
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

    # numpy_to_base64 handles sqrt + percentile clipping internally.
    # get_display_contrast_stats computes a tighter auto-scale window
    # so the slider starts at a useful narrowed range.
    b64_a = numpy_to_base64(pair[0], format=img_format)
    b64_b = numpy_to_base64(pair[1], format=img_format)
    stats = {
        "A": get_display_contrast_stats(pair[0]),
        "B": get_display_contrast_stats(pair[1]),
    }

    # Thread-safe cache write with timestamp
    access_time = time_module.time()
    with cache_lock:
        raw_image_cache[cache_key_tuple] = (b64_a, b64_b, stats, access_time)
        # Pin frame 1 entries
        if idx == 1:
            FRAME_1_KEYS.add(cache_key_tuple)
    manage_cache_size()
    _log_cache_contents()

    response = {"A": b64_a, "B": b64_b}
    if stats:
        response["stats"] = stats
    return jsonify(response)


@api_bp.route("/clear_raw_cache", methods=["POST"])
def clear_raw_cache():
    """Clear the raw image cache, optionally keeping entries for a specific camera.

    Request JSON (optional):
        exclude_camera: int - Camera number to keep (clear everything else)

    Also clears processed_store entries for evicted cameras.
    """
    data = request.get_json(silent=True) or {}
    exclude_camera = data.get("exclude_camera")

    with cache_lock:
        if exclude_camera is not None:
            cam = camera_number(exclude_camera)
            # Cache key is (source_path, camera, idx, img_format) — camera is index 1
            keys_to_remove = [k for k in raw_image_cache if k[1] != cam]
        else:
            keys_to_remove = list(raw_image_cache.keys())

        for k in keys_to_remove:
            raw_image_cache.pop(k, None)
            FRAME_1_KEYS.discard(k)

    # Also clear processed_store entries for evicted cameras (Step 10)
    if exclude_camera is not None:
        cam_str = str(camera_number(exclude_camera))
        proc = processed_store.get("processed", {})
        # Processed cache key is (source_path, camera_str) — camera is index 1
        keys_to_remove_proc = [k for k in proc if k[1] != cam_str]
        for k in keys_to_remove_proc:
            proc.pop(k, None)

    return jsonify({"status": "ok"})


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

        def _read_one_pair(idx):
            """Read a single pair, resolving loop if needed."""
            resolved_source, local_idx = _resolve_loop_read(cfg, source_path_idx, idx, camera)
            return read_pair(local_idx, resolved_source, camera, cfg)

        # Use thread pool for I/O-bound image reading
        max_workers = min(os.cpu_count(), len(indices), 8)
        pairs = [None] * len(indices)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_read_one_pair, idx): i
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

            # Batch update dictionary — store as (b64_A, b64_B, stats) at insertion time.
            # numpy_to_png_base64 handles sqrt + percentile clipping internally.
            # get_display_contrast_stats provides a tighter auto-scale window.
            processed_store["processed"][k].update({
                abs_idx: (
                    numpy_to_png_base64(processed_all[rel][0]),
                    numpy_to_png_base64(processed_all[rel][1]),
                    {
                        "A": get_display_contrast_stats(processed_all[rel][0]),
                        "B": get_display_contrast_stats(processed_all[rel][1]),
                    },
                )
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
    # Fresh Config instance (not get_config() singleton) for thread safety:
    # the /filter route's daemon thread mutates cfg.data, so concurrent
    # requests to this endpoint must not share state.
    cfg = Config()
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
        # Multi-loop: resolve global pair index to loop-specific source and local index
        resolved_source, local_idx = _resolve_loop_read(cfg, source_path_idx, frame_idx, camera)
        pair = read_pair(local_idx, resolved_source, camera, cfg)

        arr = np.stack([pair], axis=0)  # Shape: (1, 2, H, W)

        # Load mask (same as /filter route - mask should apply to all filter types)
        mask = load_mask_for_camera(camera, cfg, source_path_idx)

        if filters or mask is not None:
            # Single-frame preview: exclude temporal filters (need multiple frames)
            from pivtools_cli.processing.dask_pipeline import apply_all_filters_slim, TEMPORAL_FILTERS
            single_frame_specs = [f for f in filters if f.get("type") not in TEMPORAL_FILTERS]
            arr = apply_all_filters_slim(arr, filter_specs=single_frame_specs, pixel_mask=mask)

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
                "A": get_display_contrast_stats(processed_pair[0]),
                "B": get_display_contrast_stats(processed_pair[1]),
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

    # Import per-pattern validator
    from pivtools_core.image_handling.path_utils import validate_single_pattern

    for camera_num in camera_numbers:
        try:
            # Build camera path using shared utility
            camera_path = build_piv_camera_path(cfg, source_path_idx, camera_num)
            image_type = cfg.image_type
            num_images = cfg.num_images
            num_pairs = cfg.num_frame_pairs

            # Get all patterns (may be single pattern or A/B patterns)
            patterns = cfg.image_format if isinstance(cfg.image_format, (list, tuple)) else [cfg.image_format]
            format_str = patterns[0]  # Primary pattern for generic validation

            # Create a frame reader function for preview generation
            read_frame = create_piv_frame_reader(camera_path, camera_num, cfg)

            # --- Per-pattern validation ---
            pattern_validations = []
            for idx, pattern in enumerate(patterns):
                # Determine role based on position: first=A, second=B (for A/B mode)
                role = None
                label = "pattern"
                if len(patterns) == 2:
                    role = "A" if idx == 0 else "B"
                    label = role

                pattern_result = validate_single_pattern(
                    camera_path=camera_path,
                    pattern=pattern,
                    image_type=image_type,
                    expected_count=num_images,
                    zero_based_indexing=(cfg.start_index == 0),
                    role=role,
                    camera_num=camera_num,
                )
                pattern_result["index"] = idx
                pattern_result["label"] = label
                pattern_result["pattern"] = pattern
                pattern_validations.append(pattern_result)

            # Check A/B count consistency (warn if counts differ by >10%)
            ab_count_warning = None
            if len(pattern_validations) == 2:
                count_a = pattern_validations[0].get("found_count", 0)
                count_b = pattern_validations[1].get("found_count", 0)
                if isinstance(count_a, int) and isinstance(count_b, int) and count_a > 0 and count_b > 0:
                    diff_ratio = abs(count_a - count_b) / max(count_a, count_b)
                    if diff_ratio > 0.1:
                        ab_count_warning = f"A/B count mismatch: {count_a} A files vs {count_b} B files"

            # Use generic validator for core file detection (preview, image size, etc.)
            validation = validate_images_generic(
                camera_path=camera_path,
                camera=camera_num,
                image_format=format_str,
                image_type=image_type,
                expected_count=num_images,
                zero_based_indexing=(cfg.start_index == 0),
                read_frame_fn=read_frame,
            )

            # PIV-specific: Test first and last pairs
            first_frame_status = "missing"
            last_frame_status = "missing"
            color_detected = False

            if cfg.is_container_format:
                # Container formats (.set, .cine): validate_images_generic already
                # read frame 1 for preview. If the container exists and frame 1
                # reads, all frames exist — skip expensive redundant reads.
                if validation.get("first_image_preview"):
                    first_frame_status = "exists"
                    last_frame_status = "exists"
            else:
                # Standard/im7: individual files can be missing, test both ends
                try:
                    first_pair = read_pair(1, camera_path, camera_num, cfg)
                    first_frame_status = "exists"
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
                        expected_min = cfg.start_index
                        if min_idx != expected_min:
                            indexing_warning = (
                                f"File indexing mismatch: found files starting at {min_idx}, "
                                f"but start_index is {expected_min}"
                            )
                except Exception as e:
                    logger.debug(f"Indexing check failed: {e}")

            # Multi-loop validation: check all loop sources exist
            loop_errors = []
            if cfg.num_loops > 1:
                base_source = cfg.source_paths[source_path_idx]
                for loop_idx in range(cfg.num_loops):
                    loop_path = cfg.get_loop_source_path(base_source, loop_idx)
                    if not loop_path.exists():
                        loop_errors.append(
                            f"Loop {loop_idx} source not found: {loop_path.name}"
                        )

            # Determine status based on both generic validation and pair checks
            actual_count = validation["found_count"]
            if actual_count == "container":
                actual_count = num_images  # For containers, assume expected count

            error_msg = validation.get("error")
            status = "ok" if validation["valid"] else "error"

            # Override with PIV-specific pair validation
            if first_frame_status == "missing":
                status = "error"
                start_idx = cfg.start_index
                # Handle empty pattern case
                if not format_str or not format_str.strip():
                    error_msg = "Pattern is empty"
                elif image_type == "lavision_set":
                    error_msg = f"First frame not found. Container file: {format_str}"
                elif image_type == "cine":
                    error_msg = f"First frame not found. CINE file: {format_str % camera_num}"
                else:
                    error_msg = f"First frame not found. Looking for: {format_str % start_idx}"
                # Append folder contents hint from generic validator
                if validation.get("sample_files"):
                    error_msg += f". Found files: {', '.join(validation['sample_files'][:5])}"
                if validation.get("suggested_pattern"):
                    error_msg += f". Did you mean: {validation['suggested_pattern']}"

            elif last_frame_status == "missing":
                status = "error"
                end_idx = cfg.start_index + num_images - 1
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

            # Check if any per-pattern validation failed
            any_pattern_invalid = any(not pv.get("valid", True) for pv in pattern_validations)
            if any_pattern_invalid:
                status = "error"
                # Build per-pattern error summary
                pattern_errors = [
                    f"Pattern {pv['label']}: {pv.get('error', 'invalid')}"
                    for pv in pattern_validations
                    if not pv.get("valid", True)
                ]
                if pattern_errors and not error_msg:
                    error_msg = "; ".join(pattern_errors)

            # Multi-loop file errors
            if loop_errors:
                status = "error"
                loop_error_msg = "; ".join(loop_errors)
                error_msg = f"{error_msg}; {loop_error_msg}" if error_msg else loop_error_msg

            if status == "error":
                overall_valid = False

            # detected_count: the actual number of frames/files found (integer),
            # or None if detection wasn't possible (e.g. .set on macOS)
            detected_count = actual_count if isinstance(actual_count, int) else None

            results[f"camera_{camera_num}"] = {
                "first_frame": first_frame_status,
                "last_frame": last_frame_status,
                "expected_count": num_images,
                "actual_count": actual_count,
                "detected_count": detected_count,
                "status": status,
                "camera_path": str(camera_path),
                "color_detected": color_detected,
                "indexing_warning": indexing_warning,
                "error": error_msg,
                "first_image_preview": validation.get("first_image_preview"),
                "image_size": validation.get("image_size"),
                # Legacy fields for backward compatibility
                "suggested_pattern": validation.get("suggested_pattern"),
                "suggested_pattern_b": validation.get("suggested_pattern_b"),
                "suggested_mode": validation.get("suggested_mode"),
                # Subfolder suggestion (when camera folder doesn't exist)
                "suggested_subfolder": validation.get("suggested_subfolder"),
                # Files found in folder (for error context)
                "sample_files": validation.get("sample_files", []),
                # New per-pattern validation fields
                "pattern_validations": pattern_validations,
                "ab_count_warning": ab_count_warning,
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
        logger.debug(f"Validation passed - preloading first 10 frames for {len(camera_numbers)} camera(s)")

    # Extract detected_count and image_shape from first camera (all cameras should match)
    top_detected_count = None
    for cam_key, cam_result in results.items():
        dc = cam_result.get("detected_count")
        if isinstance(dc, int):
            top_detected_count = dc
            break

    # Store detected image_shape in config data so /config endpoint can return it
    # (avoids expensive re-read; image_size is [W, H] from validation, store as [H, W])
    for cam_key, cam_result in results.items():
        img_size = cam_result.get("image_size")
        if img_size and len(img_size) == 2:
            w, h = img_size
            cfg.data.setdefault("images", {})["image_shape"] = [h, w]
            break

    # Memory estimation check — fail validation if worker memory is insufficient
    memory_warning = None
    if overall_valid:
        from pivtools_core.validation import validate_memory_for_images
        memory_warning = validate_memory_for_images(cfg) or None
        if memory_warning:
            overall_valid = False

    return jsonify({
        "valid": overall_valid,
        "detected_count": top_detected_count,
        "memory_warning": memory_warning,
        "details": results
    })


def _normalize_dict_keys(obj):
    """Recursively convert all dict keys to strings.

    Flask's jsonify uses sort_keys=True which raises TypeError when a dict
    has mixed int/str keys. This happens when YAML loads camera numbers as
    both int (1:) and str ('1':). Normalizing to str prevents the crash.
    """
    if isinstance(obj, dict):
        return {str(k): _normalize_dict_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_dict_keys(item) for item in obj]
    return obj


@api_bp.route("/config", methods=["GET"])
def config_endpoint():
    cfg = get_config()
    # Return full nested config as JSON, including computed properties
    # Use deep copy to avoid mutating cfg.data (shallow copy shares nested dicts)
    config_data = copy.deepcopy(cfg.data)
    config_data["images"]["num_frame_pairs"] = cfg.num_frame_pairs
    config_data["images"]["per_loop_frame_pairs"] = cfg.per_loop_frame_pairs
    config_data["images"]["start_index"] = cfg.start_index
    config_data["images"]["frame_stride"] = cfg.frame_stride
    config_data["images"]["pair_stride"] = cfg.pair_stride
    config_data["images"]["pairing_preset"] = cfg.pairing_preset
    # Normalize image_format through property (handles empty lists → defaults)
    config_data["images"]["image_format"] = list(cfg.image_format)
    # Normalize all keys to strings to prevent jsonify sort_keys TypeError
    # when YAML loads camera numbers as mixed int/str keys
    config_data = _normalize_dict_keys(config_data)
    return jsonify(config_data)


@api_bp.route("/preview_frame_pairs", methods=["GET"])
def preview_frame_pairs():
    """Return first N frame pairs with resolved filenames for UI preview.

    When num_loops > 1, shows which loop each pair comes from.
    """
    cfg = get_config()
    count = request.args.get("count", 5, type=int)
    count = min(count, 20)  # Cap at 20

    num_pairs = cfg.num_frame_pairs
    num_loops = cfg.num_loops
    per_loop = cfg.per_loop_frame_pairs
    img_format = cfg.image_format
    if not img_format:
        return jsonify({"pairs": [], "total_pairs": 0, "num_images": 0,
                        "preset": cfg.pairing_preset, "frame_stride": 0,
                        "pair_stride": 1, "start_index": 0})
    format_str = img_format[0]
    has_ab = len(img_format) == 2
    is_multi_loop = num_loops > 1
    pairs = []

    for p in range(1, min(count + 1, num_pairs + 1)):
        # For multi-loop, resolve to local pair within the loop
        if is_multi_loop:
            loop_idx, local_pair = cfg.resolve_loop_for_pair(p)
            local_idx_a, local_idx_b = cfg.get_frame_pair_indices(local_pair)
            try:
                source_path = cfg.source_paths[0]
                loop_source = cfg.get_loop_source_path(source_path, loop_idx)
                loop_label = loop_source.name
            except (IndexError, ValueError):
                loop_label = f"loop={loop_idx}"
            pair_entry = {
                "pair": p,
                "frame_a": f"{loop_label} frame {local_idx_a}",
                "frame_b": f"{loop_label} frame {local_idx_b}" if cfg.frame_stride > 0 else "(internal B)",
                "loop": loop_idx,
                "local_pair": local_pair,
            }
        else:
            idx_a, idx_b = cfg.get_frame_pair_indices(p)
            if cfg.is_container_format:
                # Container files (.cine, .set): frames come from within
                # a single file, not separate files. Format string uses
                # camera number (not frame index).
                try:
                    container_name = format_str % cfg.camera_numbers[0]
                except TypeError:
                    container_name = format_str
                if cfg.frame_stride == 0:
                    name_a = f"{container_name} frame {idx_a}"
                    name_b = "(internal B)"
                else:
                    name_a = f"{container_name} frame {idx_a}"
                    name_b = f"{container_name} frame {idx_b}"
            elif has_ab:
                try:
                    name_a = cfg.image_format[0] % idx_a
                    name_b = cfg.image_format[1] % idx_a
                except TypeError:
                    name_a = cfg.image_format[0]
                    name_b = cfg.image_format[1]
            elif cfg.frame_stride == 0:
                # Pre-paired standard files: same file, internal A+B
                try:
                    name_a = format_str % idx_a
                except TypeError:
                    name_a = format_str
                name_b = "(internal B)"
            else:
                try:
                    name_a = format_str % idx_a
                    name_b = format_str % idx_b
                except TypeError:
                    name_a = format_str
                    name_b = format_str

            pair_entry = {"pair": p, "frame_a": name_a, "frame_b": name_b}

        pairs.append(pair_entry)

    response = {
        "pairs": pairs,
        "total_pairs": num_pairs,
        "num_images": cfg.num_images,
        "preset": cfg.pairing_preset,
        "frame_stride": cfg.frame_stride,
        "pair_stride": cfg.pair_stride,
        "start_index": cfg.start_index,
    }

    if is_multi_loop:
        response["num_loops"] = num_loops
        response["per_loop_pairs"] = per_loop

    return jsonify(response)


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

    # Detect calibration source changes -> reset pixel-dependent GC fields
    incoming_sources = data.get("calibration", {}).get("calibration_sources")
    if incoming_sources is not None:
        old_sources = sorted(str(s) for s in cfg.calibration_sources)
        new_sources = sorted(str(s).strip() for s in incoming_sources if s)
        if new_sources != old_sources and old_sources:
            gc = cfg.global_coordinates_config
            data.setdefault("calibration", {})["global_coordinates"] = {
                "enabled": gc.get("enabled", False),
                "datum_camera": 1,
                "datum_pixel": None,
                "datum_physical": gc.get("datum_physical", [0.0, 0.0]),
                "datum_frame": gc.get("datum_frame", 1),
                "invert_ux": False,
                "overlap_points": [],
                "overlap_pairs": [],
            }
            logger.info("Calibration sources changed — reset pixel-dependent global coordinate fields")

            # Reset derived calibration results that depend on calibration images
            cal_data = data.setdefault("calibration", {})

            # Polynomial coefficients (fitted from detection points on old images)
            cal_data.setdefault("polynomial", {})["cameras"] = {}

            # Stepped board derived data (fiducials, clicked levels, pose labels)
            sb = cal_data.setdefault("stepped_board", {})
            sb["cam1_fiducials"] = None
            sb["cam2_fiducials"] = None
            sb["cam1_clicked_level"] = None
            sb["cam2_clicked_level"] = None
            sb["cam1_pose_levels"] = {}
            sb["cam2_pose_levels"] = {}

            # Stepped planar derived data
            sp = cal_data.setdefault("stepped_planar", {})
            sp["fiducials"] = {}
            sp["clicked_level"] = {}
            sp["pose_levels"] = {}

            # Self-calibration config copy (canonical source is the file, not config.yaml)
            cal_data["self_calibration"] = {}

            logger.info(
                "Calibration sources changed — reset derived calibration fields "
                "(polynomial cameras, stepped fiducials/pose levels, self-calibration)"
            )

    # Detect active calibration method change -> clean up inactive derived data
    incoming_active = data.get("calibration", {}).get("active")
    if incoming_active is not None:
        old_active = cfg.data.get("calibration", {}).get("active")
        if old_active and incoming_active != old_active:
            cal_data = data.setdefault("calibration", {})
            # Clear self-calibration config copy (only relevant to stereo methods)
            cal_data["self_calibration"] = {}
            # Clear polynomial cameras dict when switching AWAY from polynomial
            if old_active == "polynomial":
                cal_data.setdefault("polynomial", {})["cameras"] = {}
            logger.info(
                f"Active calibration method changed {old_active!r} -> {incoming_active!r} "
                "— cleared derived data"
            )

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

    # Inject computed image properties into response so frontend stays in sync
    cfg = get_config()
    if "images" in data:
        data["images"]["num_frame_pairs"] = cfg.num_frame_pairs
        data["images"]["per_loop_frame_pairs"] = cfg.per_loop_frame_pairs

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


# Time-based cache for get_uncalibrated_count directory scans
_uncalibrated_count_cache: dict = {}  # key: (basepath_idx, type_name) -> {"time": float, "result": dict}
_UNCALIBRATED_COUNT_CACHE_TTL = 1.0   # seconds


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

    # Return cached result if fresh enough
    cache_key = (basepath_idx, type_name)
    now = time.time()
    cached = _uncalibrated_count_cache.get(cache_key)
    if cached and (now - cached["time"]) < _UNCALIBRATED_COUNT_CACHE_TTL:
        return jsonify(cached["result"])

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
    min_mtime = now - 5  # Skip files younger than 5 seconds to avoid truncated reads
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
                if p.is_file()
                and p.name in expected_names
                and p.stat().st_mtime < min_mtime
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

    result = {
        "count": total_found_files,
        "percent": percent,
        "total_expected": total_expected_files,
        "camera_progress": camera_progress,
        "cameras": camera_numbers,
        "files": all_files,
    }

    # Cache the result
    _uncalibrated_count_cache[cache_key] = {"time": now, "result": result}

    return jsonify(result)


@api_bp.route("/check_output_exists", methods=["GET"])
def check_output_exists():
    """Check if output data exists for given paths/cameras."""
    cfg = get_config()
    active_paths_str = request.args.get("active_paths", "")
    active_paths = [int(p) for p in active_paths_str.split(",") if p.strip()]
    mode = request.args.get("mode", "")  # "instantaneous" or "ensemble"
    types_to_check = [mode] if mode in ("instantaneous", "ensemble") else ["instantaneous", "ensemble"]

    logger.info(f"[check_output_exists] Checking paths: {active_paths}, mode: {mode}, types: {types_to_check}")

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

            for type_name in types_to_check:
                data_paths = get_data_paths(base, cfg.num_frame_pairs, cam_num, type_name)
                data_dir = data_paths["data_dir"]
                logger.info(f"[check_output_exists] {type_name} dir: {data_dir}, exists: {data_dir.exists()}")
                if data_dir.exists():
                    try:
                        files = list(data_dir.iterdir())
                        logger.info(f"[check_output_exists] Found {len(files)} files in {type_name} dir")
                        if files:
                            exists = True
                            details[path_key][cam_key][type_name] = True
                    except Exception as e:
                        logger.error(f"[check_output_exists] Error listing {type_name} dir: {e}")

    logger.info(f"[check_output_exists] Final result: exists={exists}")
    return jsonify({"exists": exists, "details": details})


@api_bp.route("/clear_output", methods=["POST"])
def clear_output():
    """Clear output data for given paths/cameras."""
    import shutil

    data = request.get_json() or {}
    active_paths = data.get("active_paths", [])
    camera_numbers = data.get("camera_numbers", [])
    mode = data.get("mode", "")
    types_to_clear = [mode] if mode in ("instantaneous", "ensemble") else ["instantaneous", "ensemble"]

    cfg = get_config()
    cleared = []
    errors = []

    for path_idx in active_paths:
        if path_idx >= len(cfg.base_paths):
            continue
        base = cfg.base_paths[path_idx]

        for cam_num in camera_numbers:
            for type_name in types_to_clear:
                try:
                    paths = get_data_paths(base, cfg.num_frame_pairs, cam_num, type_name)
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

@api_bp.route("/system_info", methods=["GET"])
def system_info():
    """Return system diagnostics: version, platform, Dask config, C library status."""
    import platform
    import ctypes

    cfg = get_config()

    # Version
    try:
        from pivtools_cli import __version__
        version = __version__
    except Exception:
        version = "unknown"

    # Dask config
    dask_info = {
        "workers_per_node": cfg.data.get("processing", {}).get("dask_workers_per_node", 1),
        "memory_limit": cfg.data.get("processing", {}).get("dask_memory_limit", "12GB"),
        "backend": cfg.data.get("processing", {}).get("backend", "cpu"),
        "omp_threads": cfg.data.get("processing", {}).get("omp_threads", 1),
    }

    # C library availability
    lib_extension = ".dll" if os.name == "nt" else ".so"
    lib_dir = Path(__file__).parent.parent / "pivtools_cli" / "lib"
    c_libs = {}
    for lib_name in ["libbulkxcorr2d", "libmarquadt", "libfusedwarp"]:
        lib_path = lib_dir / f"{lib_name}{lib_extension}"
        c_libs[lib_name] = {
            "found": lib_path.is_file(),
            "path": str(lib_path) if lib_path.is_file() else None,
        }

    # FFTW wisdom
    wisdom_path = Path.home() / ".pypivtools_fftw_wisdom"
    fftw_wisdom = {
        "path": str(wisdom_path),
        "exists": wisdom_path.is_file(),
        "size_bytes": wisdom_path.stat().st_size if wisdom_path.is_file() else 0,
    }

    return jsonify({
        "version": version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "dask": dask_info,
        "c_libraries": c_libs,
        "fftw_wisdom": fftw_wisdom,
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
    # Suppress Werkzeug startup logs (uses standard logging, not loguru)
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    # Ensure config.yaml exists before starting
    ensure_config_exists()

    print("Starting PIVTOOLs GUI...")
    print("Open your browser to http://localhost:5000")
    
    # Automatically open browser after a short delay
    def open_browser():
        webbrowser.open('http://localhost:5000')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()