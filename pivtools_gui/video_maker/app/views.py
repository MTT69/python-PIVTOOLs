import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request, send_file
from loguru import logger

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from ..video_maker import PlotSettings, make_video_from_scalar

video_maker_bp = Blueprint("video_maker", __name__, url_prefix="/video")

# Constants
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
MAX_DEPTH = 5  # For deep search

# In-memory video job state with thread-safety
_video_state: Dict[str, Any] = {
    "processing": False,
    "progress": 0,
    "message": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "meta": None,
    "out_path": None,
    "current_frame": 0,
    "total_frames": 0,
}
_video_thread: Optional[threading.Thread] = None
_video_cancel_event = threading.Event()
_video_state_lock = threading.RLock()  # Reentrant lock for safety


def _video_set_state(**kwargs):
    with _video_state_lock:
        _video_state.update(kwargs)


def _video_reset_state():
    with _video_state_lock:
        _video_state.update(
            {
                "processing": False,
                "progress": 0,
                "message": None,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "meta": None,
                "out_path": None,
                "current_frame": 0,
                "total_frames": 0,
            }
        )


def progress_callback(current_frame: int, total_frames: int, message: str = ""):
    """Thread-safe progress update."""
    _video_set_state(
        progress=int((current_frame / max(total_frames, 1)) * 100),
        current_frame=current_frame,
        total_frames=total_frames,
        message=f"Processing frame {current_frame}/{total_frames}"
        + (f" - {message}" if message else ""),
    )


def _run_video_job(
    base: Path,
    cam: int,
    num_images: int,  # Number of images/files in the folder
    run: int,  # Run number (1-based) for run_index
    source_type: str,
    endpoint: str,
    merged_flag: bool,
    var: str,
    pattern: str,
    ps: PlotSettings,
    test_mode: bool = False,
    test_frames: int = 50,
):
    """Optimized job with better error handling."""
    try:
        _video_set_state(
            processing=True,
            progress=0,
            started_at=datetime.utcnow().isoformat(),
            message="Initializing video creation",
            error=None,
            meta=None,
            current_frame=0,
        )

        logger.info(
            f"[VIDEO] Starting video job | base='{base}', cam={cam}, num_images={num_images}, run={run}, var={var}, test_mode={test_mode}"
        )

        paths = get_data_paths(
            base, num_images, cam, source_type, endpoint, merged_flag
        )

        data_dir = Path(paths.get("data_dir"))
        video_dir = Path(paths.get("video_dir"))

        video_dir.mkdir(parents=True, exist_ok=True)

        if not Path(ps.out_path).is_absolute():
            ps.out_path = str(video_dir / ps.out_path)

        ps.progress_callback = progress_callback
        ps.test_mode = test_mode
        ps.test_frames = test_frames if test_mode else None

        _video_set_state(message="Starting video generation...")

        meta = make_video_from_scalar(
            data_dir,
            var=var,
            pattern=pattern,
            settings=ps,
            cancel_event=_video_cancel_event,
            run_index=run - 1,  # Convert run (1-based) to run_index (0-based)
        )

        if _video_cancel_event.is_set():
            _video_set_state(
                processing=False,
                progress=0,
                message="Video creation was cancelled",
                finished_at=datetime.utcnow().isoformat(),
                error="Cancelled by user",
            )
            return

        _video_set_state(
            progress=100,
            message="Video completed successfully",
            processing=False,
            finished_at=datetime.utcnow().isoformat(),
            meta=meta,
            out_path=ps.out_path,
            computed_limits={
                "lower": meta.get("vmin"),
                "upper": meta.get("vmax"),
                "actual_min": meta.get("actual_min"),
                "actual_max": meta.get("actual_max"),
                "percentile_based": ps.lower_limit is None or ps.upper_limit is None,
            },
        )
        logger.info(f"[VIDEO] Job completed successfully. Output: {ps.out_path}")

    except Exception as e:
        logger.exception(f"[VIDEO] Job failed: {e}")
        _video_set_state(
            processing=False,
            error=str(e),
            message=f"Video creation failed: {str(e)}",
            finished_at=datetime.utcnow().isoformat(),
        )


@video_maker_bp.route("/list_videos", methods=["GET"])
def list_videos():
    """Optimized video listing with glob and caching."""
    try:
        base_path_str = request.args.get("base_path")
        cfg = get_config(refresh=True)

        base = Path(base_path_str).expanduser() if base_path_str else cfg.base_paths[0]

        logger.info(f"[VIDEO] Listing videos under base path: {base}")

        videos: List[str] = []

        videos_dir = base / "videos"
        if videos_dir.exists():
            for ext in VIDEO_EXTENSIONS:
                videos.extend([str(f) for f in videos_dir.glob(f"**/*{ext}")])

        cam_dirs = [d for d in base.glob("**/Cam*") if d.is_dir()]
        for cam_dir in cam_dirs:
            for video_subdir in ["videos", "merged/videos"]:
                video_dir = cam_dir / video_subdir
                if video_dir.exists():
                    for ext in VIDEO_EXTENSIONS:
                        videos.extend([str(f) for f in video_dir.glob(f"*{ext}")])

        if not videos:

            def find_videos(directory: Path, current_depth: int = 0) -> List[str]:
                if current_depth > MAX_DEPTH:
                    return []
                found = []
                try:
                    for item in directory.iterdir():
                        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
                            found.append(str(item))
                        elif item.is_dir():
                            found.extend(find_videos(item, current_depth + 1))
                except (PermissionError, OSError):
                    pass
                return found

            videos = find_videos(base)

        videos.sort(
            key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True
        )

        logger.info(f"[VIDEO] Found {len(videos)} videos")
        return jsonify({"videos": videos})
    except Exception as e:
        logger.exception(f"[VIDEO] Failed to list videos: {e}")
        return jsonify({"error": str(e), "videos": []}), 500


@video_maker_bp.route("/start_video", methods=["POST"])
def start_video():
    """
    Start video job with validation.
    
    Expected JSON parameters:
    - base_path: str - Base directory path for data
    - camera: int - Camera number (1-based)
    - run: int - Run number (1-based)
    - var: str - Variable to visualize ("ux", "uy", "mag")
    - fps: int (optional) - Video frame rate (1-120, default: 30)
    - test_mode: bool (optional) - Create test video with limited frames
    - test_frames: int (optional) - Number of frames for test mode (default: 50)
    - lower/upper: float (optional) - Custom color scale limits
    - cmap: str (optional) - Matplotlib colormap name
    - resolution: str (optional) - Video resolution ("4k" or default)
    - out_name: str (optional) - Custom output filename
    """
    global _video_thread

    data = request.get_json(silent=True) or {}
    cfg = get_config(refresh=True)

    # Validate inputs
    base_path_str = data.get("base_path")
    if not base_path_str:
        return jsonify({"error": "base_path is required"}), 400
    base = Path(base_path_str).expanduser()
    if not base.exists():
        return jsonify({"error": "Invalid base_path"}), 400

    cam_raw = data.get("camera")
    if cam_raw is None:
        return jsonify({"error": "camera is required"}), 400
    try:
        cam = int(cam_raw)
        if cam < 1:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Invalid camera number"}), 400

    test_mode = data.get("test_mode", False)
    if not isinstance(test_mode, bool):
        return jsonify({"error": "test_mode must be boolean"}), 400
    test_frames = int(data.get("test_frames", 50))
    if test_frames < 1:
        return jsonify({"error": "test_frames must be positive"}), 400

    # Parse run as the run number (1-based)
    run_raw = data.get("run")
    if run_raw is None:
        return jsonify({"error": "run is required"}), 400
    try:
        run = int(run_raw)
        if run < 1:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Invalid run number"}), 400

    num_images = int(data.get("num_images", 1))  # Keep for other uses, e.g., if needed elsewhere
    if num_images < 1:
        return jsonify({"error": "num_images must be positive"}), 400
    merged_flag = str(data.get("merged", "0")) in ("1", "true", "True")
    endpoint = data.get("endpoint", "") or ""
    source_type = data.get("type", "instantaneous") or "instantaneous"
    if source_type not in ["instantaneous", "ensemble"]:  # Add allowed types
        return jsonify({"error": "Invalid source_type"}), 400

    var = data.get("var", None) or data.get("var", "uy")
    if var not in ("ux", "uy", "mag"):
        return jsonify({"error": "Invalid var"}), 400
    pattern = data.get("pattern", "[0-9]*.mat")

    ps = PlotSettings()

    # Parse FPS with validation (frames per second for video output)
    fps = data.get("fps", 30)  # Default to 30 FPS if not provided
    try:
        fps = int(fps)
        if fps < 1 or fps > 120:  # Reasonable range: 1-120 FPS
            return jsonify({"error": "FPS must be between 1 and 120"}), 400
        ps.fps = fps
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid FPS value"}), 400

    ps.crf = 18
    ps.upscale = (1080, 1920) if data.get("resolution") != "4k" else (2160, 3840)
    ps.out_path = data.get(
        "out_name",
        f"run{run}_Cam{cam}_{var}{'_test' if test_mode else ''}.mp4",  # Use run for filename
    )

    try:
        lower = data.get("lower")
        upper = data.get("upper")
        ps.lower_limit = float(lower) if lower and str(lower).strip() else None
        ps.upper_limit = float(upper) if upper and str(upper).strip() else None
    except ValueError:
        return jsonify({"error": "Invalid lower/upper limits"}), 400

    cmap = data.get("cmap")
    if cmap and cmap != "default":
        ps.cmap = cmap

    with _video_state_lock:
        running = _video_thread is not None and _video_thread.is_alive()
    if running:
        with _video_state_lock:
            st = {k: _video_state.get(k) for k in ("processing", "progress", "message")}
        return jsonify({"status": "busy", **st}), 409

    _video_cancel_event.clear()
    _video_reset_state()
    _video_set_state(message="Video queued")

    _video_thread = threading.Thread(
        target=_run_video_job,
        args=(
            base,
            cam,
            num_images,  # Pass num_images for folder selection
            run,  # Pass run for run_index
            source_type,
            endpoint,
            merged_flag,
            var,
            pattern,
            ps,
            test_mode,
            test_frames,
        ),
        daemon=True,
    )
    _video_thread.start()

    return jsonify({"status": "started", "processing": True, "progress": 0}), 202


@video_maker_bp.route("/cancel_video", methods=["POST"])
def cancel_video():
    """Cancel video job safely."""
    _video_cancel_event.set()
    with _video_state_lock:
        is_running = bool(_video_thread is not None and _video_thread.is_alive())
    if is_running:
        _video_set_state(message="Cancellation requested")
        return jsonify({"status": "cancelling", "processing": True}), 202
    _video_reset_state()
    return jsonify({"status": "idle", "processing": False}), 200


@video_maker_bp.route("/video_status", methods=["GET"])
def video_status():
    """Return thread-safe status."""
    with _video_state_lock:
        st = dict(_video_state)
        st["processing"] = bool(
            st.get("processing", False)
            or (_video_thread is not None and _video_thread.is_alive())
        )
    st["progress"] = int(max(0, min(100, int(st.get("progress", 0)))))
    if st.get("out_path"):
        st["out_path"] = st["out_path"]
    elif st.get("meta") and isinstance(st["meta"], dict) and "out_path" in st["meta"]:
        st["out_path"] = st["meta"]["out_path"]
    if st.get("computed_limits"):
        st["computed_limits"] = st["computed_limits"]
    return jsonify(st), 200


@video_maker_bp.route("/download", methods=["GET"])
def download_video():
    """Stream video file with range support."""
    try:
        abs_path = Path(request.args.get("path", "")).resolve()
        if not abs_path.is_file() or abs_path.suffix.lower() not in VIDEO_EXTENSIONS:
            return jsonify({"error": "Invalid file"}), 400
        user_home = Path.home()
        cwd = Path.cwd()
        
        # Get configured base paths for data access
        cfg = get_config(refresh=True)
        config_base_paths = [Path(bp).resolve() for bp in cfg.base_paths if Path(bp).exists()]
        
        allowed_roots = [
            user_home,
            cwd,
            Path("/tmp"),
            Path("/var/tmp"),
            Path("/Users"),
            Path("/home"),
        ]
        
        # Add configured base paths to allowed roots
        allowed_roots.extend(config_base_paths)
        
        if os.name == "nt":
            allowed_roots.extend([Path("C:\\Users"), Path("C:\\temp"), Path("C:\\tmp")])
        path_allowed = any(
            allowed_root in abs_path.parents or abs_path == allowed_root
            for allowed_root in allowed_roots
        )
        if not path_allowed:
            logger.warning(f"Attempted download of disallowed path: {abs_path}")
            logger.debug(f"Allowed roots: {allowed_roots}")
            logger.debug(f"File parents: {list(abs_path.parents)}")
            return jsonify({"error": "File not allowed"}), 403
        response = send_file(
            str(abs_path), mimetype="video/mp4", conditional=True, as_attachment=True
        )
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Range")
        return response
    except Exception as e:
        logger.error(f"Error serving video file: {e}")
        return jsonify({"error": f"Error serving file: {str(e)}"}), 500
