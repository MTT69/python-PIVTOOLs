import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request
from loguru import logger

from config import get_config
from paths import get_data_paths
from video_maker.video_maker import PlotSettings, make_video_from_scalar

video_maker_bp = Blueprint("video_maker", __name__, url_prefix="/video")

# In-memory video job state
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
_video_state_lock = threading.Lock()


def _video_set_state(**kwargs):
    with _video_state_lock:
        _video_state.update(kwargs)


def _video_reset_state():
    _video_set_state(
        processing=False,
        progress=0,
        message=None,
        started_at=None,
        finished_at=None,
        error=None,
        meta=None,
        out_path=None,
        current_frame=0,
        total_frames=0,
    )


def progress_callback(current_frame: int, total_frames: int, message: str = ""):
    """Callback function to update progress during video creation"""
    progress = int((current_frame / max(total_frames, 1)) * 100)
    _video_set_state(
        progress=progress,
        current_frame=current_frame,
        total_frames=total_frames,
        message=f"Processing frame {current_frame}/{total_frames}"
        + (f" - {message}" if message else ""),
    )


def _run_video_job(
    base: Path,
    cam: int,
    num_images: int,
    source_type: str,
    endpoint: str,
    merged_flag: bool,
    pick: str,
    pattern: str,
    ps: PlotSettings,
    test_mode: bool = False,
    test_frames: int = 50,
):
    """Background job to call make_video_from_scalar and update state."""
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
            f"[VIDEO] Starting video job | base='{base}', cam={cam}, num_images={num_images}, run={pick}, test_mode={test_mode}"
        )

        # Resolve data/video dirs
        paths = get_data_paths(
            base, num_images, cam, source_type, endpoint, merged_flag
        )

        data_dir = Path(paths.get("data_dir"))
        video_dir = Path(paths.get("video_dir"))

        video_dir.mkdir(parents=True, exist_ok=True)

        # Ensure output path set
        if not Path(ps.out_path).is_absolute():
            ps.out_path = str(video_dir / ps.out_path)

        # Set progress callback
        ps.progress_callback = progress_callback
        ps.test_mode = test_mode
        ps.test_frames = test_frames if test_mode else None

        _video_set_state(message="Starting video generation...")

        # Call video creation with progress tracking
        meta = make_video_from_scalar(
            data_dir,
            pick=pick,
            pattern=pattern,
            settings=ps,
            cancel_event=_video_cancel_event,
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


@video_maker_bp.route("/start_video", methods=["POST"])
def start_video():
    """Start a video job with JSON payload

    Returns 202 when queued, 409 if a job is already running.
    """
    global _video_thread

    data = request.get_json(silent=True) or {}
    cfg = get_config(refresh=True)

    # Resolve base directory
    base_path_str = data.get("base_path")
    if isinstance(base_path_str, str) and base_path_str.strip():
        base = Path(base_path_str).expanduser()
    else:
        idx = int(data.get("basepath_idx", 0))
        try:
            base = cfg.base_paths[idx]
        except Exception:
            base = cfg.base_paths[0]

    # Resolve camera
    cam_raw = data.get("camera")
    try:
        cam = int(cam_raw) if cam_raw is not None else int(cfg.camera_numbers[0])
    except (ValueError, TypeError, IndexError):
        cam = int(cfg.camera_numbers[0])

    # Test mode parameters
    test_mode = data.get("test_mode", False)
    test_frames = int(data.get("test_frames", 50))

    # other params
    num_images = int(data.get("num_images", data.get("run", 1)))
    merged_flag = str(data.get("merged", "0")) in ("1", "true", "True")
    endpoint = data.get("endpoint", "") or ""
    source_type = data.get("type", "instantaneous") or "instantaneous"

    # prefer 'var' (new client param); fall back to 'pick' for backward-compatibility
    pick = data.get("var", None)
    if pick is None:
        pick = data.get("pick", "uy")
    if pick not in ("ux", "uy", "mag"):
        pick = "uy"
    pattern = data.get("pattern", "[0-9]*.mat")

    ps = PlotSettings()
    try:
        fps = data.get("fps")
        if fps is not None:
            ps.fps = int(fps)
    except Exception:
        pass

    # Create output filename
    out_name = data.get("out_name")
    if not out_name:
        test_suffix = "_test" if test_mode else ""
        out_name = f"run{num_images}_Cam{cam}_{pick}{test_suffix}.mp4"
    ps.out_path = out_name

    try:
        lower = data.get("lower")
        upper = data.get("upper")
        if lower is not None and str(lower).strip():
            ps.lower_limit = float(lower)
        if upper is not None and str(upper).strip():
            ps.upper_limit = float(upper)
    except Exception:
        pass

    cmap = data.get("cmap")
    if cmap and cmap != "default":
        ps.cmap = cmap

    dither = data.get("dither")
    if dither is not None:
        ps.dither = dither.lower() in ("1", "true", "t", "yes")

    upscale = data.get("upscale", 1)
    try:
        ps.upscale = float(upscale) if upscale else 1.0
    except Exception:
        ps.upscale = 1.0

    # Start background job if none running
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
            num_images,
            source_type,
            endpoint,
            merged_flag,
            pick,
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
    """Request cancellation of running video job."""
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
    """Return current in-memory video job state for polling."""
    with _video_state_lock:
        st = dict(_video_state)
        st["processing"] = bool(
            st.get("processing", False)
            or (_video_thread is not None and _video_thread.is_alive())
        )
    try:
        st["progress"] = int(max(0, min(100, int(st.get("progress", 0)))))
    except Exception:
        st["progress"] = 0
    st["status"] = st["progress"]

    # Include out_path directly if available
    if st.get("out_path"):
        st["out_path"] = st["out_path"]
    elif st.get("meta") and isinstance(st["meta"], dict) and "out_path" in st["meta"]:
        st["out_path"] = st["meta"]["out_path"]

    # Include computed limits if available
    if st.get("computed_limits"):
        st["computed_limits"] = st["computed_limits"]

    return jsonify(st), 200


@video_maker_bp.route("/download", methods=["GET"])
def download_video():
    """Download or stream a video file by absolute path (with security check)."""
    try:
        import os
        import urllib.parse

        from flask import request, send_file

        abs_path = request.args.get("path", "")
        abs_path = urllib.parse.unquote(abs_path)
        logger.info(f"[VIDEO] Download request for path: {abs_path}")

        # Security: allow files under reasonable locations
        # Allow files under the current working directory, user home, and common data directories
        user_home = Path.home()
        cwd = Path.cwd()

        allowed_roots = [
            user_home,  # User's home directory
            cwd,  # Current working directory
            Path("/tmp"),
            Path("/var/tmp"),  # Temp directories (Unix)
            Path("/Users"),
            Path("/home"),  # User directories (macOS/Linux)
        ]

        # Add Windows paths if on Windows
        if os.name == "nt":
            allowed_roots.extend(
                [
                    Path("C:\\Users"),
                    Path("C:\\temp"),
                    Path("C:\\tmp"),
                ]
            )

        file_path = Path(abs_path).resolve()
        logger.info(f"[VIDEO] Resolved file path: {file_path}")
        logger.info(f"[VIDEO] File exists: {file_path.is_file()}")

        path_allowed = any(
            allowed_root in file_path.parents or file_path == allowed_root
            for allowed_root in allowed_roots
        )
        logger.info(f"[VIDEO] Path allowed: {path_allowed}")

        if not file_path.is_file() or not path_allowed:
            return (
                jsonify({"error": f"File not found or not allowed: {file_path}"}),
                404,
            )
        if not str(file_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            return jsonify({"error": "Invalid file type"}), 400
        return send_file(str(file_path), as_attachment=False, mimetype="video/mp4")
    except Exception as e:
        logger.error(f"Error serving video file: {e}")
        return jsonify({"error": "Error serving file"}), 500
