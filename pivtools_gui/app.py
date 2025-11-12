import threading
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import yaml
from dask import config as dask_config
from flask import Blueprint, Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from loguru import logger
import os
from .calibration.app.views import calibration_bp
from pivtools_core.config import get_config, reload_config
from pivtools_core.image_handling.load_images import read_pair
from .masking.app.views import masking_bp
from pivtools_core.paths import get_data_paths
from .piv_runner import get_runner
from .plotting.app.views import vector_plot_bp
from .post_processing.POD.app.views import POD_bp
from pivtools_cli.preprocessing.preprocess import preprocess_images
from .stereo_reconstruction.app.views import stereo_bp
from .utils import camera_folder, camera_number, numpy_to_png_base64
from pivtools_gui.vector_statistics.app.views import statistics_bp
from pivtools_gui.vector_merging.app.views import merging_bp
from .video_maker.app.views import video_maker_bp

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)
dask_config.set(scheduler="threads")

# Create API blueprint with /backend prefix
api_bp = Blueprint('api', __name__, url_prefix='/backend')

# Register existing blueprints with /backend prefix
app.register_blueprint(vector_plot_bp, url_prefix='/backend')
app.register_blueprint(masking_bp, url_prefix='/backend')
app.register_blueprint(POD_bp, url_prefix='/backend')
app.register_blueprint(calibration_bp, url_prefix='/backend')
app.register_blueprint(video_maker_bp, url_prefix='/backend')
app.register_blueprint(stereo_bp, url_prefix='/backend')
app.register_blueprint(statistics_bp, url_prefix='/backend')
app.register_blueprint(merging_bp, url_prefix='/backend')

# --- In-memory stores ---
processed_store = {"processed": {}}
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


@api_bp.route("/get_frame_pair", methods=["GET"])
def get_frame_pair():
    cfg = get_config()
    camera = request.args.get("camera", type=int)
    idx = request.args.get("idx", type=int)
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    
    # For .set and .im7 files, don't append camera folder - all cameras are in the source directory
    image_format = cfg.image_format
    if isinstance(image_format, tuple):
        format_str = image_format[0]
    else:
        format_str = image_format
    
    if '.set' in str(format_str) or '.im7' in str(format_str):
        source_path = cfg.source_paths[source_path_idx]
    else:
        source_path = cfg.source_paths[source_path_idx] / camera_folder(camera)
    
    try:
        pair = read_pair(idx, source_path, camera, cfg)
    except FileNotFoundError as e:
        return jsonify({"error": "File not found", "file": str(e)}), 404

    return jsonify(
        {"A": numpy_to_png_base64(pair[0]), "B": numpy_to_png_base64(pair[1])}
    )



@api_bp.route("/filter", methods=["POST"])
def filter_images_endpoint():
    global processing
    data = request.get_json() or {}
    cfg = get_config()
    camera = camera_number(data.get("camera"))
    start_idx = int(data.get("start_idx", 1))
    filters = data.get("filters", None)
    source_path_idx = data.get("source_path_idx")
    
    if filters is not None:
        # Remove batch_size from filters before storing (it's configured in batches.size)
        cleaned_filters = []
        for f in filters:
            cleaned = {k: v for k, v in f.items() if k != 'batch_size'}
            cleaned_filters.append(cleaned)
        cfg.data["filters"] = cleaned_filters
    
    # Use batch size from config
    batch_length = cfg.data.get("batches", {}).get("size", 30)
    batch_len_reason = "config.batches.size"
    
    if batch_length < 1:
        batch_length = 1
    
    batch_start, batch_end = compute_batch_window(
        start_idx, batch_length, cfg.num_images
    )
    indices = list(range(batch_start, batch_end + 1))
    
    # For .set and .im7 files, don't append camera folder - all cameras are in the source directory
    image_format = cfg.image_format
    if isinstance(image_format, tuple):
        format_str = image_format[0]
    else:
        format_str = image_format
    
    if '.set' in str(format_str) or '.im7' in str(format_str):
        source_path = cfg.source_paths[source_path_idx]
    else:
        source_path = cfg.source_paths[source_path_idx] / camera_folder(camera)

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
        logger.debug("/filter processing thread started")
        try:
            # Load with parallel I/O
            darr = load_pairs_parallel()
            
            # Process with dask, then compute both in parallel
            processed_darr = preprocess_images(darr, cfg)
            
            # Compute processed
            processed_all = dask.compute(processed_darr, scheduler='threads')[0]
            
            # Store results
            k = cache_key(source_path_idx, camera)
            processed_store["processed"].setdefault(k, {})
            
            # Batch update dictionary (faster than individual updates)
            processed_store["processed"][k].update({
                abs_idx: processed_all[rel] 
                for rel, abs_idx in enumerate(indices)
            })
            
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


@api_bp.route("/get_processed_pair", methods=["GET"])
def get_processed_pair():
    frame = request.args.get("frame", type=int)
    typ = request.args.get("type", "processed")
    camera = camera_number(request.args.get("camera"))
    source_path_idx = request.args.get("source_path_idx", default=0, type=int)
    
    logger.debug(f"Checking cache for processed frame {frame}, type {typ}, camera {camera}, source_path_idx {source_path_idx}")
    b64_a, b64_b = get_cached_pair(frame, typ, camera, source_path_idx)
    
    if b64_a is not None and b64_b is not None:
        logger.debug(f"Cache hit for processed frame {frame}, type {typ}, camera {camera}")
    else:
        logger.debug(f"Cache miss for processed frame {frame}, type {typ}, camera {camera}")
    
    return jsonify({"status": "ok", "A": b64_a, "B": b64_b})


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
    
    # Check if any batch filters are present (should use /filter endpoint instead)
    batch_filters = [f for f in filters if f.get("type") in ("time", "pod")]
    if batch_filters:
        return jsonify({
            "error": "Batch filters (time, pod) not supported in single-frame mode. Use /filter endpoint."
        }), 400
    
    # For .set and .im7 files, don't append camera folder
    image_format = cfg.image_format
    if isinstance(image_format, tuple):
        format_str = image_format[0]
    else:
        format_str = image_format
    
    if '.set' in str(format_str) or '.im7' in str(format_str):
        source_path = cfg.source_paths[source_path_idx]
    else:
        source_path = cfg.source_paths[source_path_idx] / camera_folder(camera)
    
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
        
        return jsonify({
            "status": "ok",
            "A": numpy_to_png_base64(processed_pair[0]),
            "B": numpy_to_png_base64(processed_pair[1])
        })
        
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


@api_bp.route("/config", methods=["GET"])
def config_endpoint():
    cfg = get_config()
    # Already returns full nested config as JSON
    return jsonify(cfg.data)


@api_bp.route("/update_config", methods=["POST"])
def update_config(): 
    data = request.get_json() or {}
    cfg = get_config()
    
    # Special handling for filters: remove batch_size before saving
    if "filters" in data:
        cleaned_filters = []
        for f in data["filters"]:
            if isinstance(f, dict):
                cleaned = {k: v for k, v in f.items() if k != 'batch_size'}
                cleaned_filters.append(cleaned)
        data["filters"] = cleaned_filters

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

    recursive_update(cfg.data, data)
    
    # Handle camera_numbers based on camera_count changes
    new_camera_count = cfg.data["paths"].get("camera_count", 1)
    if new_camera_count != old_camera_count:
        # Reset camera_numbers to default range when camera_count changes
        cfg.data["paths"]["camera_numbers"] = list(range(1, new_camera_count + 1))
    else:
        # Fix camera_numbers if camera_count was not updated
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
        "cameras": [1, 2, 3],  // List of camera numbers to process (optional)
        "source_path_idx": 0,   // Index of source path (optional, default 0)
        "base_path_idx": 0      // Index of base path (optional, default 0)
    }
    """
    data = request.get_json() or {}
    
    # Extract parameters
    cameras = data.get("cameras")
    source_path_idx = data.get("source_path_idx", 0)
    base_path_idx = data.get("base_path_idx", 0)
    
    # Get the runner and start the job
    runner = get_runner()
    result = runner.start_piv_job(
        cameras=cameras,
        source_path_idx=source_path_idx,
        base_path_idx=base_path_idx,
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


@api_bp.route("/get_uncalibrated_count", methods=["GET"])
def get_uncalibrated_count():
    cfg = get_config()
    basepath_idx = request.args.get("basepath_idx", default=0, type=int)
    cam = camera_number(request.args.get("camera", default=1, type=int))
    type_name = request.args.get("type", default="instantaneous")
    base_paths = cfg.base_paths
    base = base_paths[basepath_idx]
    num_images = cfg.num_images
    
    # Get all cameras that should be processed
    camera_numbers = cfg.camera_numbers
    total_cameras = len(camera_numbers)
    
    # Calculate progress across all cameras
    total_expected_files = num_images * total_cameras
    total_found_files = 0
    camera_progress = {}
    
    vector_fmt = cfg.vector_format
    expected_names = set([vector_fmt % i for i in range(1, num_images + 1)])
    
    # Count files for each camera and collect all available files
    all_files = []
    for camera_num in camera_numbers:
        paths = get_data_paths(base, num_images, camera_num, type_name, use_uncalibrated=True)
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
            "percent": int((len(found) / num_images) * 100) if num_images else 0
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


def main():
    """Run the PIVTOOLs GUI"""
    # Suppress Flask development server warning by setting production environment
    import os
    os.environ['FLASK_ENV'] = 'production'
    
    print("Starting PIVTOOLs GUI...")
    print("Open your browser to http://localhost:5000")
    
    # Automatically open browser after a short delay
    def open_browser():
        import time
        import webbrowser
        time.sleep(2)  # Wait for server to start
        webbrowser.open('http://localhost:5000')
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()
