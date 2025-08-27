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
):
    """Background job to call make_video_from_scalar and update state."""
    # cfg = get_config(refresh=True)
    try:
        _video_set_state(
            processing=True,
            progress=0,
            started_at=datetime.utcnow().isoformat(),
            message="Video job running",
            error=None,
            meta=None,
        )
        logger.info(
            f"[VIDEO] Starting video job | base='{base}', cam={cam}, num_images={num_images}, pick={pick}"
        )

        # Resolve data/video dirs
        paths = get_data_paths(
            base, num_images, cam, source_type, endpoint, merged_flag
        )

        data_dir = Path(paths.get("data_dir"))
        video_dir = Path(paths.get("video_dir"))

        video_dir.mkdir(parents=True, exist_ok=True)

        # Ensure output path set
        if not getattr(ps, "out_path", None):
            out_name = f"run{num_images}_Cam{cam}_{pick}.mp4"
            ps.out_path = str(video_dir / out_name)

        # Note: make_video_from_scalar is not cancellable; this is best-effort
        meta = make_video_from_scalar(data_dir, pick=pick, pattern=pattern, settings=ps)

        _video_set_state(
            progress=100,
            message="Video completed",
            processing=False,
            finished_at=datetime.utcnow().isoformat(),
            meta=meta,
        )
        logger.info("[VIDEO] Job completed successfully")
    except Exception as e:
        logger.exception(f"[VIDEO] Job failed: {e}")
        _video_set_state(
            processing=False,
            error=str(e),
            message="Video failed",
            finished_at=datetime.utcnow().isoformat(),
        )


@video_maker_bp.route("/make_video", methods=["GET"])
def make_video():
    """Queue a video job (backwards-compatible GET). Returns 202 if queued.

    Use `/video/video_status` to poll job state and `/video/cancel_video` to request cancellation.
    """
    cfg = get_config(refresh=True)

    # Resolve base directory
    base_path_str = request.args.get("base_path")
    if base_path_str and base_path_str.strip():
        base = Path(base_path_str).expanduser()
    else:
        try:
            idx = int(request.args.get("basepath_idx", 0))
        except Exception:
            idx = 0
        try:
            base = cfg.base_paths[idx]
        except Exception:
            base = cfg.base_paths[0]

    # Resolve camera
    try:
        cam = int(request.args.get("camera", cfg.camera_numbers[0]))
    except Exception:
        cam = int(cfg.camera_numbers[0])

    # Run label
    try:
        run_label = int(request.args.get("run", 1))
    except Exception:
        run_label = 1

    # Merged flag
    merged_flag = request.args.get("merged", "0") in ("1", "true", "True")

    # Find endpoint/source_type
    endpoint = request.args.get("endpoint", "") or ""
    source_type = request.args.get("type", "instantaneous") or "instantaneous"

    # Request args for video creation
    # prefer 'var' (new client param); fall back to 'pick' for backward-compatibility
    pick = request.args.get("var", None)
    if pick not in ("ux", "uy"):
        pick = "uy"
    pattern = request.args.get("pattern", "[0-9]*.mat")

    # Build PlotSettings from optional request params
    ps = PlotSettings()
    try:
        fps = request.args.get("fps")
        if fps is not None:
            ps.fps = int(fps)
    except Exception:
        pass

    out_name = request.args.get("out_name") or f"run{run_label}_Cam{cam}_{pick}.mp4"
    ps.out_path = out_name

    try:
        lower = request.args.get("lower_limit")
        upper = request.args.get("upper_limit")
        if lower is not None:
            ps.lower_limit = float(lower)
        if upper is not None:
            ps.upper_limit = float(upper)
    except Exception:
        pass
    cmap = request.args.get("cmap")
    if cmap:
        ps.cmap = cmap
    dither = request.args.get("dither")
    if dither is not None:
        ps.dither = dither.lower() in ("1", "true", "t", "yes")

    upscale = request.args.get("upscale")
    if upscale is not None:
        try:
            if "," in upscale:
                h, w = upscale.split(",", 1)
                ps.upscale = (int(h.strip()), int(w.strip()))
            else:
                ps.upscale = float(upscale)
        except Exception:
            pass

    # Start background job if none running
    global _video_thread
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
            run_label,
            source_type,
            endpoint,
            merged_flag,
            pick,
            pattern,
            ps,
        ),
        daemon=True,
    )
    _video_thread.start()

    return jsonify({"status": "started", "processing": True, "progress": 0}), 202


@video_maker_bp.route("/start_video", methods=["POST"])
def start_video():
    """Start a video job with JSON payload (same params as GET for /make_video).

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
    except Exception:
        cam = int(cfg.camera_numbers[0])

    # other params
    run_label = int(data.get("run", 1))
    merged_flag = str(data.get("merged", "0")) in ("1", "true", "True")
    endpoint = data.get("endpoint", "") or ""
    source_type = data.get("type", "instantaneous") or "instantaneous"
    # prefer 'var' (new client param); fall back to 'pick' for backward-compatibility
    pick = data.get("var", None)
    if pick is None:
        pick = data.get("pick", "uy")
    if pick not in ("ux", "uy"):
        pick = "uy"
    pattern = data.get("pattern", "[0-9]*.mat")

    ps = PlotSettings()
    try:
        fps = data.get("fps")
        if fps is not None:
            ps.fps = int(fps)
    except Exception:
        pass
    out_name = data.get("out_name") or f"run{run_label}_Cam{cam}_{pick}.mp4"
    ps.out_path = out_name

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
            run_label,
            source_type,
            endpoint,
            merged_flag,
            pick,
            pattern,
            ps,
        ),
        daemon=True,
    )
    _video_thread.start()

    return jsonify({"status": "started", "processing": True, "progress": 0}), 202


@video_maker_bp.route("/cancel_video", methods=["POST"])
def cancel_video():
    """Request cancellation of running video job. Best-effort only."""
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
    return jsonify(st), 200
